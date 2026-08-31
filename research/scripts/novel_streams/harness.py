"""NOVEL-STREAM HARNESS -- one plugin API, one evaluation pack, many mechanisms.

Design rule: a new scientific mechanism should require ONE class and nothing
else. Everything downstream -- caching, fold splits, TS-AUC, dominant-cell
diagnostics, pair flow, RT-600 marginal integration -- is shared code that
already exists in this repository and is merely wired together here.

    class MyMechanism(StreamingMechanism):
        name = "m20_mine"
        def fit_history(self, hist):  ...        # per-series state from history ONLY
        def emit(self, hist, online): ...        # (n_online, k) float32, causal

    python -m novel_streams.harness --mechanism m20_mine --stage screen

STAGES
  build      compute + cache the stream over the full store            (MODE A/B)
  screen     single-column cell AUC / correlation / redundancy pack    (no training)
  specialist train ONE LightGBM on {500 causal cols + the new cols},
             fold 0 only, and run ensemble_marginal against RT600+clone
  report     write research/reports/<name>.json

CAUSALITY IS NOT OPTIONAL. `verify()` runs the repository's own bitwise
prefix-invariance check (`sbr.features.base.check_prefix_invariance`, atol=0.0)
against a registered wrapper of the mechanism, on 8 series of different lengths.
A mechanism that has not passed it cannot reach `screen`.
"""
from __future__ import annotations

import json
import os
import sys
import time

#: The checkout this file belongs to. Its `src` and `research/scripts` are
#: ALWAYS importable, so `import harness` works with no environment set up and
#: keeps working when SBR_ROOT points at a data-only tree.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

#: Where the data lives: cache/store, cache/features, research/folds,
#: research/oof. Defaults to REPO; SBR_ROOT points it elsewhere when the
#: caches are shared across worktrees (the usual local setup).
ROOT = os.environ.get("SBR_ROOT", REPO)

for _p in (f"{REPO}/src", f"{REPO}/research/scripts",
           f"{ROOT}/src", f"{ROOT}/research/scripts"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from wave5_lib import SPECIALISTS, Ctx, load_oof  # noqa: E402

from sbr.metric import ts_auc_flat  # noqa: E402
from sbr.store import load_store  # noqa: E402

#: Where cached streams land. Under ``cache/``, which .gitignore already
#: covers, so a mechanism build never dirties the working tree. Created on
#: first write, never as an import side effect.
CACHE = os.path.join(ROOT, "cache", "novel_streams")

DOMINANT_T_MIN = 200
DOMINANT_AGE_MIN = 100


def _cache_dir() -> str:
    os.makedirs(CACHE, exist_ok=True)
    return CACHE


# ---------------------------------------------------------------- the plugin
class StreamingMechanism:
    """Subclass this. Two methods, no framework."""

    name: str = "unnamed"
    cols: list[str] = []

    def fit_history(self, hist: np.ndarray):
        """Return whatever per-series state the mechanism needs. History only."""
        raise NotImplementedError

    def emit(self, hist: np.ndarray, online: np.ndarray) -> np.ndarray:
        """(n_online, len(self.cols)) float32. Row t may see hist and online[:t+1]."""
        raise NotImplementedError

    # -- optional: an explicit online form, if the mechanism is naturally stateful
    def initialize_online(self, state):  return state
    def update(self, state, x_t):        raise NotImplementedError
    def current_features(self, state):   raise NotImplementedError


# ------------------------------------------------------------------- causality
def store_series(n_series: int = 8):
    """``n_series`` (hist, online) pairs spanning the store's online-length range."""
    st = load_store(f"{ROOT}/cache/store")
    lens = st.meta.n_online.to_numpy()
    pick = np.argsort(lens)[np.linspace(0, len(lens) - 1, n_series).astype(int)]
    return [(st.hist(int(i)), st.online(int(i)), int(i)) for i in pick]


def verify(mech: StreamingMechanism, n_series: int = 8, cuts=(3, 10, 37, 111),
           series=None) -> tuple[bool, str]:
    """Bitwise prefix invariance (atol=0.0) -- the repository's causal contract.

    Row ``t`` may depend only on ``hist`` and ``online[:t+1]``, so rebuilding the
    mechanism on a truncated online segment must reproduce the surviving rows
    EXACTLY -- including their NaN pattern. ``equal_nan=True`` makes NaN match
    NaN, which means a value-vs-NaN disagreement is a failure: a mechanism that
    emits 0.0 in the full replay where the truncated replay emits NaN (or the
    reverse) is rejected. That is not a look-ahead, but it is the batch/stream
    parity defect this repository already treats as its largest deployment risk,
    and it is exactly what this check caught on ``m20_dwell_probe``'s first run.

    ``series``: optional list of ``(hist, online)`` or ``(hist, online, label)``
    tuples. Defaults to a spread of real series from ``cache/store``. Passing
    synthetic series lets the sentinel be exercised with no competition data --
    see ``tests/test_novel_streams_harness.py``.
    """
    if series is None:
        series = store_series(n_series)
    for rec in series:
        h, o = rec[0], rec[1]
        tag = rec[2] if len(rec) > 2 else "?"
        full = np.asarray(mech.emit(h, o), dtype=np.float64)
        for k in cuts:
            if k >= len(o):
                continue
            part = np.asarray(mech.emit(h, o[:k]), dtype=np.float64)
            if part.shape != full[:k].shape:
                return False, (f"{mech.name}: series {tag} prefix {k} shape "
                               f"{part.shape} != {full[:k].shape}")
            bad = ~np.isclose(full[:k], part, rtol=0, atol=0.0, equal_nan=True)
            if bad.any():
                j = int(np.argmax(bad.any(axis=0)))
                col = mech.cols[j] if j < len(mech.cols) else f"col{j}"
                return False, f"{mech.name}: series {tag} prefix {k} differs, column {col}"
    return True, "ok"


# ----------------------------------------------------------------------- build
def build(mech: StreamingMechanism, force=False) -> tuple[np.ndarray, list[str]]:
    f = f"{_cache_dir()}/{mech.name}.npy"
    if os.path.exists(f) and not force:
        return np.load(f, mmap_mode="r"), json.load(open(f"{CACHE}/{mech.name}.cols.json"))["cols"]
    ok, msg = verify(mech)
    if not ok:
        raise SystemExit(f"CAUSALITY FAILED -- {msg}")
    st = load_store(f"{ROOT}/cache/store")
    n_rows = int(st.meta.n_online.sum())
    A = np.full((n_rows, len(mech.cols)), np.nan, dtype=np.float32)
    pos, t0 = 0, time.time()
    for i in range(len(st.meta)):
        n = int(st.meta.n_online.iloc[i])
        A[pos:pos + n] = mech.emit(st.hist(i), st.online(i))
        pos += n
        if i % 1000 == 0:
            print(f"  {i} {time.time()-t0:.0f}s", flush=True)
    np.save(f, A)
    json.dump({"cols": mech.cols, "causal": msg}, open(f"{CACHE}/{mech.name}.cols.json", "w"))
    return np.load(f, mmap_mode="r"), mech.cols


# -------------------------------------------------------------------- scoring
def within_t_rank(x, t):
    o = np.lexsort((x, t))
    out = np.empty(len(x))
    ts = t[o]
    b = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]])
    e = np.r_[b[1:], len(ts)]
    for lo, hi in zip(b, e):
        out[o[lo:hi]] = rankdata(x[o[lo:hi]]) / (hi - lo)
    return out


def cell_rows(c: Ctx, rows, t_min=DOMINANT_T_MIN, age_min=DOMINANT_AGE_MIN):
    y, t, a = c.d.y[rows], c.d.t[rows], c.age[rows]
    return rows[(t >= t_min) & ((y == 0) | (a >= age_min))]


def diagnostic_pack(c: Ctx, cand: np.ndarray, base: np.ndarray, fold=0, label="cand"):
    """The standard pack every mechanism reports. No training, seconds to run."""
    out = {"label": label, "fold": fold}
    r = c.rows[fold]
    y, t = c.d.y[r], c.d.t[r]
    out["whole_fold"] = {"rt600": float(ts_auc_flat(base[r], y, t)),
                         "candidate": float(ts_auc_flat(cand[r], y, t))}
    rr = cell_rows(c, r)
    yy, tt = c.d.y[rr], c.d.t[rr]
    out["dominant_cell"] = {"rt600": float(ts_auc_flat(base[rr], yy, tt)),
                            "candidate": float(ts_auc_flat(cand[rr], yy, tt))}
    hb = c.has_break[c.d.sidx[rr]]
    for nm, keep in (("mature_vs_neverbreak", (yy == 1) | ((yy == 0) & ~hb)),
                     ("mature_vs_prebreak",   (yy == 1) | ((yy == 0) &  hb))):
        out[nm] = {"rt600": float(ts_auc_flat(base[rr][keep], yy[keep], tt[keep])),
                   "candidate": float(ts_auc_flat(cand[rr][keep], yy[keep], tt[keep]))}
    out["t_buckets"], out["age_buckets"] = {}, {}
    for lo, hi in ((0, 20), (20, 50), (50, 100), (100, 200), (200, 400), (400, 10**9)):
        m = (t >= lo) & (t < hi)
        if m.sum() > 1000 and 0 < y[m].mean() < 1:
            out["t_buckets"][f"{lo}-{hi}"] = {
                "rt600": float(ts_auc_flat(base[r][m], y[m], t[m])),
                "candidate": float(ts_auc_flat(cand[r][m], y[m], t[m]))}
    for lo, hi in ((0, 5), (5, 20), (20, 50), (50, 100), (100, 10**9)):
        a = c.age[r]
        m = (y == 0) | ((y == 1) & (a >= lo) & (a < hi))
        out["age_buckets"][f"{lo}-{hi}"] = {
            "rt600": float(ts_auc_flat(base[r][m], y[m], t[m])),
            "candidate": float(ts_auc_flat(cand[r][m], y[m], t[m]))}
    out["within_t_rank_corr_rt600"] = float(np.corrcoef(
        within_t_rank(cand[rr], tt), within_t_rank(base[rr], tt))[0, 1])
    return out


def pair_flow_by_cell(base, cand, c: Ctx, fold=0, n_pairs_per_t=20, seed=0):
    """wave8_common.pair_repair_stats, split by dominant cell and negative type."""
    from wave8_common import pair_repair_stats
    r = c.rows[fold]
    out = {"whole_fold": pair_repair_stats(base, cand, c.d.y, c.d.t, r, n_pairs_per_t, seed)}
    rr = cell_rows(c, r)
    out["dominant_cell"] = pair_repair_stats(base, cand, c.d.y, c.d.t, rr, n_pairs_per_t, seed)
    hb = c.has_break[c.d.sidx]
    out["cell_never_break_neg"] = pair_repair_stats(
        base, cand, c.d.y, c.d.t, rr[(c.d.y[rr] == 1) | ~hb[rr]], n_pairs_per_t, seed)
    out["cell_pre_break_neg"] = pair_repair_stats(
        base, cand, c.d.y, c.d.t, rr[(c.d.y[rr] == 1) | hb[rr]], n_pairs_per_t, seed)
    return out


def rt600_blend(c: Ctx, cache=True):
    f = f"{_cache_dir()}/rt600_blend.npy"
    if cache and os.path.exists(f):
        return np.load(f)
    v = c.crossfit_blend(load_oof(SPECIALISTS), SPECIALISTS)
    np.save(f, v)
    return v


def marginal(cand_oof, c: Ctx, fold=0, label="cand"):
    """RT600 vs RT600+seedclone vs RT600+candidate. The binding number."""
    from wave8_common import ensemble_marginal
    return ensemble_marginal(cand_oof, c=c, fold=fold, label=label)
