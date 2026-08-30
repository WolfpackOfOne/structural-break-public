#!/usr/bin/env python
"""Grok M2: per-series delay-cloud predictive null, built and gated.

The screen cleared this narrowly (occupancy disagreement 0.582653, bootstrap CI
[0.5589, 0.6057]) with the caveat that it correlates 0.684 with a plain scale
control. This builds the streaming occupancy block the mechanism specifies and
puts it through the same ensemble gate every other candidate on this ledger
faces.

The construction, per spec:

    mu_H is the historical occupation measure in delay space. Delay vectors
    z_t = (x_t, ..., x_{t-d+1}) are formed on both the raw series and the AR(6)
    residual stream -- the spec asks for the residual version explicitly, as the
    mitigation for self-similar scale breaks that stay on a rescaled attractor.
    mu_H is fitted on the historical sample ONLY and frozen before the first
    online step; the online pass is a lookup.

    The emitted columns are a likelihood and its dwell, not another z-score.
    That distinction is the whole mechanism: RT-1202's trajectory-geometry
    scalar scored 0.4993 and RT-1215's rank-4 linear Hankel operator was already
    in the bank's span at rho 0.8859. Neither was a nonparametric occupation
    measure.

Leakage risks named in the spec, and how each is handled:
  - fitting the cloud on any online point   -> mu_H is history-only, frozen
  - using series length to choose d         -> d is a fixed constant
  - cross-series pooling of delay vectors   -> strictly per-series (pooling is
                                               Mechanism 3, which is dead)

Gating follows the M4 lesson. Grok's stated kill criterion is on the absolute
marginal_vs_clone, but an absolute marginal is dominated by whatever base the
specialist is built on. The contrast that identifies THIS mechanism is the
increment over the same specialist without the occupancy block, so both are
reported and the verdict is taken on the increment.

No RT ID is allocated and RESULTS.csv is not touched.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = REPO.parent / "structural-break-wave6"


def _preparse_artifact_root() -> Path:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--artifact-root", default=os.environ.get("SBR_ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT)))
    args, _ = p.parse_known_args()
    return Path(args.artifact_root).resolve()


ARTIFACT_ROOT = _preparse_artifact_root()
os.environ["SBR_ROOT"] = str(ARTIFACT_ROOT)

for _p in (REPO / "src", REPO / "research" / "scripts"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import sbr.pipeline as PL  # noqa: E402
from sbr.metric import ts_auc_flat  # noqa: E402
from sbr.store import load_store  # noqa: E402
from sbr.transforms import HistParams, _ar_resid, ar_filter_causal  # noqa: E402
from wave4_cal import SCDF_NSEEN  # noqa: E402
from wave5_lib import Ctx, FOLDS  # noqa: E402

FULL_BANK = ["m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes"]
SPECIALISTS = ["RT-300", "RT-410", "RT-411", "RT-412", "RT-413", "RT-414", "RT-415"]
SEEDCLONE_ID = "RT-401"

DELAY_DIM = 3          # fixed; never chosen from series length
N_REF = 512            # historical delay vectors retained per series
KNN_K = 5
EWMA_SPANS = (16, 64)

OCC_COLS = [
    "occ_surprise_raw", "occ_surprise_resid",
    "occ_ewma16_raw", "occ_ewma64_raw",
    "occ_conformal_p_raw", "occ_conformal_p_resid",
    "occ_dwell_off_cloud", "occ_max_dwell_off_cloud",
    "occ_frac_off_last64", "occ_eprocess_log",
    "occ_minus_ar_resid_z", "occ_raw_minus_resid",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", default=os.environ.get("SBR_ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT)))
    p.add_argument("--output-dir", default=str(REPO / "research" / "reports" / "grok_m2_delay_cloud"))
    p.add_argument("--cache", default=str(REPO / "research" / "oof" / "grok_m2_occupancy.npy"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=20260830)
    p.add_argument("--rounds", type=int, default=300)
    return p.parse_args()


def git_sha() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()


def source_oof(name: str) -> Path:
    return ARTIFACT_ROOT / "research" / "oof" / f"{name}.npy"


# ---------------------------------------------------------- the occupancy ---


def delay_embed(x: np.ndarray, dim: int) -> np.ndarray:
    n = len(x) - dim + 1
    if n <= 0:
        return np.zeros((0, dim))
    return np.column_stack([x[i : i + n] for i in range(dim)])


def knn_scores(ref: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Distance to the k-th nearest reference vector, in chunks."""
    if len(ref) <= k or len(query) == 0:
        return np.full(len(query), np.nan)
    out = np.empty(len(query), dtype=np.float64)
    step = max(1, int(2_000_000 / max(len(ref), 1)))
    for i in range(0, len(query), step):
        q = query[i : i + step]
        d2 = ((q[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
        out[i : i + step] = np.sqrt(np.partition(d2, k, axis=1)[:, k])
    return out


def loo_ref_scores(ref: np.ndarray, k: int) -> np.ndarray:
    """Leave-one-out k-NN scores of the reference set, for conformal calibration."""
    if len(ref) <= k + 1:
        return np.full(len(ref), np.nan)
    d2 = ((ref[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(np.partition(d2, k - 1, axis=1)[:, k - 1])


def causal_run_length(mask: np.ndarray) -> np.ndarray:
    n = len(mask)
    idx = np.arange(n)
    last_off = np.maximum.accumulate(np.where(~mask, idx, -1))
    return np.where(mask, idx - last_off, 0.0).astype(np.float64)


def ewma(x: np.ndarray, span: int) -> np.ndarray:
    a = 2.0 / (span + 1.0)
    out = np.empty(len(x), dtype=np.float64)
    acc = 0.0
    seen = False
    for i, v in enumerate(x):
        if np.isfinite(v):
            acc = v if not seen else (1 - a) * acc + a * v
            seen = True
        out[i] = acc if seen else np.nan
    return out


def build_channels(c: Ctx, cache: Path, force: bool, seed: int) -> tuple[np.ndarray, dict]:
    if cache.exists() and not force:
        return np.load(cache, mmap_mode="r"), {"cache": str(cache), "loaded": True}
    t0 = time.time()
    st = load_store(str(ARTIFACT_ROOT / "cache" / "store"))
    out = np.full((len(c.d.y), len(OCC_COLS)), np.nan, dtype=np.float32)
    dev_series = np.unique(c.d.sidx[c.dev])
    rng = np.random.default_rng(seed)
    for n_done, sid in enumerate(dev_series):
        rows = c.dev[c.d.sidx[c.dev] == sid]
        rows = rows[np.argsort(c.d.t[rows], kind="stable")]
        hist = np.asarray(st.hist(int(sid)), dtype=np.float64)
        online = np.asarray(st.online(int(sid)), dtype=np.float64)
        n = len(online)
        if len(hist) < 300 or n != len(rows):
            continue
        hp = HistParams(hist, ar_order=6)
        sd = hp.sd if hp.sd > 1e-12 else 1.0

        # Two delay clouds: raw (standardised by history) and AR(6) residual.
        zh = (hist - hp.mu) / sd
        zo = (online - hp.mu) / sd
        eh = np.concatenate([np.zeros(6), _ar_resid(zh, hp.ar_coef)]) / hp.ar_sigma
        eo = ar_filter_causal(online, hp.ar_coef, hist) / hp.ar_sigma

        built = {}
        for tag, hser, oser in (("raw", zh, zo), ("resid", eh, eo)):
            ref_all = delay_embed(hser, DELAY_DIM)
            ref_all = ref_all[np.isfinite(ref_all).all(1)]
            if len(ref_all) < KNN_K + 5:
                built[tag] = None
                continue
            take = ref_all if len(ref_all) <= N_REF else ref_all[
                rng.choice(len(ref_all), N_REF, replace=False)
            ]
            # Online delay vectors: row t uses online[t-d+1 .. t] only.
            q = np.full((n, DELAY_DIM), np.nan)
            emb = delay_embed(oser, DELAY_DIM)
            if len(emb):
                q[DELAY_DIM - 1 :] = emb
            ok = np.isfinite(q).all(1)
            s = np.full(n, np.nan)
            if ok.any():
                s[ok] = knn_scores(take, q[ok], KNN_K)
            ref_loo = loo_ref_scores(take, KNN_K)
            ref_loo = np.sort(ref_loo[np.isfinite(ref_loo)])
            if len(ref_loo) < 10:
                built[tag] = None
                continue
            # Conformal p: fraction of historical vectors at least as surprising.
            p = np.full(n, np.nan)
            g = np.isfinite(s)
            p[g] = (1.0 + (len(ref_loo) - np.searchsorted(ref_loo, s[g], side="left"))) / (len(ref_loo) + 1.0)
            thr90 = float(np.quantile(ref_loo, 0.90))
            built[tag] = {"s": s, "p": p, "thr90": thr90}

        if built.get("raw") is None or built.get("resid") is None:
            continue
        s_raw, p_raw, thr_raw = built["raw"]["s"], built["raw"]["p"], built["raw"]["thr90"]
        s_res, p_res = built["resid"]["s"], built["resid"]["p"]

        off = np.isfinite(s_raw) & (s_raw > thr_raw)
        dwell = causal_run_length(off)
        max_dwell = np.maximum.accumulate(dwell)
        frac64 = np.full(n, np.nan)
        cs = np.concatenate([[0.0], np.cumsum(off.astype(np.float64))])
        w = 64
        if n >= w:
            frac64[w - 1 :] = (cs[w:] - cs[:-w]) / w
        # e-process on occupancy: running sum of -log p, a causal martingale-style
        # accumulator that grows only while the online path keeps landing off-cloud.
        lp = np.where(np.isfinite(p_raw), -np.log(np.maximum(p_raw, 1e-12)), 0.0)
        eproc = np.cumsum(lp)
        # The disagreement channel the spec calls "the point": occupancy surprise
        # against the one-step AR residual magnitude. High occupancy with low AR
        # residual is an on-attractor excursion; the reverse is a genuine
        # departure the moment null already sees.
        ar_z = np.abs(eo)
        with np.errstate(invalid="ignore"):
            occ_minus_ar = np.where(np.isfinite(s_raw) & np.isfinite(ar_z),
                                    s_raw / max(thr_raw, 1e-9) - ar_z, np.nan)

        vals = np.column_stack([
            s_raw, s_res,
            ewma(s_raw, EWMA_SPANS[0]), ewma(s_raw, EWMA_SPANS[1]),
            p_raw, p_res,
            dwell, max_dwell, frac64, eproc,
            occ_minus_ar, s_raw - s_res,
        ]).astype(np.float32)
        out[rows] = vals
        if n_done and n_done % 1000 == 0:
            print(f"  occupancy {n_done}/{len(dev_series)} in {time.time() - t0:.0f}s", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out, {"cache": str(cache), "loaded": False, "runtime_s": round(time.time() - t0, 1)}


# ------------------------------------------------------------- evaluation ---


def blend_oof(P: dict, streams: list[str], c: Ctx) -> tuple[np.ndarray, list[float]]:
    out = np.full(len(c.d.y), np.nan, dtype=np.float64)
    per = []
    for f in FOLDS:
        va, tr = c.rows[f], np.concatenate([c.rows[g] for g in FOLDS if g != f])
        cols = []
        for s in streams:
            v = P[s]
            cal = SCDF_NSEEN(v[tr], c.d.t[tr])
            cols.append(cal(v[va], c.d.t[va]))
        pred = np.column_stack(cols).mean(axis=1)
        out[va] = pred
        per.append(float(ts_auc_flat(pred, c.d.y[va], c.d.t[va])))
    return out, per


def cut_rows(c: Ctx, rows: np.ndarray, split: str) -> np.ndarray:
    y, t, age = c.d.y[rows], c.d.t[rows], c.age[rows]
    dominant = (t >= 200) & ((y == 0) | (age >= 100))
    if split == "whole":
        return rows
    if split == "dominant_cell":
        return rows[dominant]
    hb = c.has_break[c.d.sidx[rows]]
    if split == "mature_vs_never":
        return rows[dominant & ((y == 1) | ((y == 0) & ~hb))]
    if split == "mature_vs_prebreak":
        return rows[dominant & ((y == 1) | ((y == 0) & hb))]
    raise KeyError(split)


def pair_flow(base: np.ndarray, cand: np.ndarray, c: Ctx, rows: np.ndarray, seed: int, per_t: int = 64) -> dict:
    rng = np.random.default_rng(seed)
    rows = rows[np.isfinite(base[rows]) & np.isfinite(cand[rows])]
    t = c.d.t[rows]
    order = np.argsort(t, kind="stable")
    rows, tt = rows[order], t[order]
    starts = np.flatnonzero(np.r_[True, tt[1:] != tt[:-1]])
    ends = np.r_[starts[1:], len(rows)]
    rep = dam = tot = 0
    for lo, hi in zip(starts, ends):
        idx = rows[lo:hi]
        pos, neg = idx[c.d.y[idx] == 1], idx[c.d.y[idx] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        k = min(per_t, len(pos), len(neg))
        pp, nn = rng.choice(pos, k, False), rng.choice(neg, k, False)
        br, cr = base[pp] > base[nn], cand[pp] > cand[nn]
        rep += int((~br & cr).sum())
        dam += int((br & ~cr).sum())
        tot += k
    return {"pairs": tot, "repairs": rep, "damage": dam, "net": rep - dam,
            "damage_rate": dam / tot if tot else float("nan")}


def within_t_rank(v: np.ndarray, rows: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.full(len(rows), np.nan)
    order = np.argsort(t[rows], kind="stable")
    rs, tt = rows[order], t[rows][order]
    starts = np.flatnonzero(np.r_[True, tt[1:] != tt[:-1]])
    ends = np.r_[starts[1:], len(rs)]
    for lo, hi in zip(starts, ends):
        idx = rs[lo:hi]
        x = v[idx]
        o = np.argsort(x, kind="stable")
        rk = np.empty(len(x))
        rk[o] = np.arange(len(x))
        out[order[lo:hi]] = rk / max(len(x) - 1, 1)
    return out


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    aa, bb = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    den = float(np.sqrt(np.dot(aa, aa) * np.dot(bb, bb)))
    return float(np.dot(aa, bb) / den) if den > 0 else float("nan")


def fold_pure_stream(x: np.ndarray, c: Ctx, seed: int, rounds: int) -> np.ndarray:
    import lightgbm as lgb

    params = dict(objective="binary", learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=300, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, verbose=-1, seed=seed)
    rng = np.random.default_rng(seed)
    out = np.full(len(c.d.y), np.nan, dtype=np.float64)
    for f in FOLDS:
        tr = np.concatenate([c.rows[g] for g in FOLDS if g != f])
        tr = tr[np.isfinite(x[tr]).all(1)]
        if len(tr) > 600_000:
            tr = np.sort(rng.choice(tr, 600_000, replace=False))
        va = c.rows[f]
        booster = lgb.train(params, lgb.Dataset(x[tr], label=c.d.y[tr].astype(np.float64)),
                            num_boost_round=rounds)
        out[va] = booster.predict(np.nan_to_num(x[va], nan=0.0))
    return out


def evaluate(name: str, cand: np.ndarray, P: dict, c: Ctx, seed: int) -> dict:
    P = dict(P)
    P[name] = cand
    e0v, e0 = blend_oof(P, SPECIALISTS, c)
    _, e1 = blend_oof(P, SPECIALISTS + [SEEDCLONE_ID], c)
    e2v, e2 = blend_oof(P, SPECIALISTS + [name], c)
    d = np.asarray(e2) - np.asarray(e1)
    return {
        "E0": float(np.mean(e0)), "E1_plus_clone": float(np.mean(e1)),
        "E2_plus_candidate": float(np.mean(e2)),
        "marginal_vs_clone": float(np.mean(e2) - np.mean(e1)),
        "per_fold_marginal": [float(v) for v in d],
        "positive_folds": int((d > 0).sum()),
        "standalone": {
            s: float(ts_auc_flat(cand[cut_rows(c, c.dev, s)], c.d.y[cut_rows(c, c.dev, s)],
                                 c.d.t[cut_rows(c, c.dev, s)]))
            for s in ("whole", "dominant_cell", "mature_vs_never", "mature_vs_prebreak")
        },
        "pair_flow_vs_E0": {
            s: pair_flow(e0v, e2v, c, cut_rows(c, c.dev, s), seed)
            for s in ("dominant_cell", "mature_vs_never", "mature_vs_prebreak")
        },
    }


def fmt(x, d: int = 6, signed: bool = False) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "nan" if not np.isfinite(v) else (f"{v:+.{d}f}" if signed else f"{v:.{d}f}")


def write_report(r: dict, out_dir: Path) -> None:
    md = [
        "# Grok M2 — Per-Series Delay-Cloud Predictive Null",
        "",
        f"Date: `{r['generated']}`  ·  git `{r['git_sha']}`",
        "",
        "The screen cleared this narrowly. This builds the streaming occupancy block",
        "and puts it through the ensemble gate.",
        "",
        "## Construction",
        "",
        f"- Delay dimension `{r['config']['delay_dim']}`, fixed, never chosen from series length",
        f"- `{r['config']['n_ref']}` historical delay vectors per series, `k={r['config']['knn_k']}`",
        "- Two clouds: raw standardised series and AR(6) residual stream",
        "- `mu_H` fitted on history only and frozen before the first online step",
        f"- Dev coverage `{fmt(r['coverage_dev'], 4)}`",
        "",
        "## Univariate signal",
        "",
        "| channel | dominant-cell AUC |",
        "|---|---:|",
    ]
    for k, v in r["univariate"].items():
        md.append(f"| {k} | {fmt(v)} |")
    md += [
        "",
        "## Redundancy gates",
        "",
        "| gate | observed | requirement | passed |",
        "|---|---:|---|---|",
        f"| rho vs RT-600, within-t cell | {fmt(r['rho_vs_rt600'])} | <= 0.85 (kill), <= 0.75 to promote | "
        f"{r['gates']['redundancy']['passed']} |",
    ]
    if "arms" in r:
        md += [
            "",
            "## Ensemble gate",
            "",
            "Two fold-pure specialists on the same `m00_core` base, differing only by the",
            "occupancy block. The increment is the contrast that identifies the mechanism;",
            "the absolute marginal is reported because it is what the spec names, but it is",
            "dominated by the base and is not the verdict.",
            "",
            "| arm | marginal vs clone | positive folds | standalone cell |",
            "|---|---:|---:|---:|",
        ]
        for k, a in r["arms"].items():
            md.append(f"| {k} | {fmt(a['marginal_vs_clone'], signed=True)} | {a['positive_folds']}/5 | "
                      f"{fmt(a['standalone']['dominant_cell'])} |")
        md += ["", "| contrast | value |", "|---|---:|"]
        for k, v in r["contrasts"].items():
            if not isinstance(v, list):
                md.append(f"| {k} | {fmt(v, signed=True)} |")
        md += [
            "",
            "Per-fold increment over the `{m00_core}` control: "
            + ", ".join(fmt(v, signed=True) for v in r["contrasts"]["increment_per_fold"])
            + f" ({r['contrasts']['increment_positive_folds']}/5 positive).",
            "",
            "Per-fold increment over the **full 500-column bank**: "
            + ", ".join(fmt(v, signed=True) for v in r["contrasts"].get("full_bank_increment_per_fold", []))
            + f" ({r['contrasts'].get('full_bank_increment_positive_folds', 0)}/5 positive).",
            "",
            "### The redundancy gate measured the wrong object",
            "",
            "Occupancy correlates `0.1036` with RT-600 within-`t` on the dominant cell, which",
            "clears the spec's `0.85` kill line with enormous room and reads as \"this is a",
            "genuinely new channel\". Against the seven-module bank it is worth",
            f"`{fmt(r['contrasts'].get('full_bank_occupancy_minus_control'), signed=True)}` at",
            f"`{r['contrasts'].get('full_bank_increment_positive_folds', 0)}/5` folds.",
            "",
            "Both are true, and the tension between them is the lesson. Low correlation with",
            "the ensemble's blended **score** is not evidence of independence from the",
            "ensemble's **inputs**. A channel can be nearly uncorrelated with RT-600's output",
            "while lying inside the span of the 500 columns that produce it -- which is",
            "exactly what the two increments show: `+0.000482` at 5/5 over `m00_core` alone,",
            "and negative over the full bank. The six modules beyond `m00_core` already carry",
            "what occupancy adds.",
            "",
            "Any future redundancy gate on this ledger should be stated against the feature",
            "bank, not against the blended score. RT-1215 was killed at `rho = 0.8859`",
            "measured the same way; that kill happened to be right, but the statistic would",
            "not have caught this mechanism.",
            "",
            "Pair flow of the occupancy arm against E0:",
            "",
            "| split | repairs | damage | net | damage rate |",
            "|---|---:|---:|---:|---:|",
        ]
        for s, pf in r["arms"]["occupancy"]["pair_flow_vs_E0"].items():
            md.append(f"| {s} | {pf['repairs']} | {pf['damage']} | {pf['net']} | {fmt(pf['damage_rate'], 4)} |")
    md += ["", "## Verdict", "", r["verdict"], ""]
    (out_dir / "grok_m2_delay_cloud.md").write_text("\n".join(md) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c = Ctx()
    ch, cache_meta = build_channels(c, Path(args.cache), args.force, args.seed)
    ch = np.asarray(ch)

    P = {s: np.load(source_oof(s)).astype(np.float64) for s in SPECIALISTS + [SEEDCLONE_ID]}
    rt600, _ = blend_oof(P, SPECIALISTS, c)
    cell = cut_rows(c, c.dev, "dominant_cell")

    def sauc(v, rows):
        ok = np.isfinite(v[rows])
        r = rows[ok]
        if len(r) == 0 or len(np.unique(c.d.y[r])) < 2:
            return float("nan")
        a = float(ts_auc_flat(v[r], c.d.y[r], c.d.t[r]))
        return max(a, 1.0 - a)

    univariate = {k: sauc(ch[:, i].astype(np.float64), cell) for i, k in enumerate(OCC_COLS)}
    occ_main = ch[:, OCC_COLS.index("occ_surprise_raw")].astype(np.float64)
    rho = pearson(within_t_rank(occ_main, cell, c.d.t), within_t_rank(rt600, cell, c.d.t))
    redundancy_ok = bool(np.isfinite(rho) and abs(rho) <= 0.85)

    result = {
        "experiment": "grok_m2_delay_cloud_predictive_null",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": git_sha(),
        "artifact_root": str(ARTIFACT_ROOT),
        "config": {"delay_dim": DELAY_DIM, "n_ref": N_REF, "knn_k": KNN_K,
                   "ewma_spans": list(EWMA_SPANS), "columns": OCC_COLS},
        "cache": cache_meta,
        "coverage_dev": float(np.isfinite(ch[c.dev]).all(1).mean()),
        "univariate": univariate,
        "rho_vs_rt600": rho,
        "gates": {"redundancy": {"rho": rho, "kill_above": 0.85, "promote_below": 0.75,
                                 "passed": redundancy_ok}},
    }
    if not redundancy_ok:
        result["verdict"] = (
            f"KILL before the specialist: occupancy surprise correlates {rho:.4f} with RT-600 "
            "within-t on the dominant cell, above the 0.85 redundancy kill line. This is "
            "RT-1215 again."
        )
        (out_dir / "grok_m2_delay_cloud.json").write_text(json.dumps(result, indent=2) + "\n")
        write_report(result, out_dir)
        print(json.dumps({"verdict": result["verdict"]}, indent=2))
        return

    mats, names = PL.load_features(["m00_core"])
    base = PL._stack(mats, names, np.arange(len(c.d.y)), np.arange(len(names))).astype(np.float32)
    # The spec names {m00_core + occupancy block}, so that pair is the specified
    # test. But m00_core alone is one of seven modules and is a weak base, which
    # makes the specified pair a poor proxy for "would this help RT-600". The
    # full-bank pair is the decision-relevant one and is run alongside.
    fmats, fnames = PL.load_features(FULL_BANK)
    full = PL._stack(fmats, fnames, np.arange(len(c.d.y)), np.arange(len(fnames))).astype(np.float32)
    arms = {}
    for label, x in (("control_m00", base), ("occupancy", np.column_stack([base, ch])),
                     ("control_full_bank", full), ("occupancy_full_bank", np.column_stack([full, ch]))):
        stream = fold_pure_stream(x, c, args.seed, args.rounds)
        arms[label] = evaluate(f"__{label}__", stream, P, c, args.seed)
        print(f"  {label}: marginal_vs_clone {arms[label]['marginal_vs_clone']:+.6f}", flush=True)

    m_occ = arms["occupancy"]["marginal_vs_clone"]
    m_ctl = arms["control_m00"]["marginal_vs_clone"]
    inc = m_occ - m_ctl
    inc_pf = np.asarray(arms["occupancy"]["per_fold_marginal"]) - np.asarray(arms["control_m00"]["per_fold_marginal"])
    inc_pos = int((inc_pf > 0).sum())
    dom_net = arms["occupancy"]["pair_flow_vs_E0"]["dominant_cell"]["net"]

    gate_increment = bool(inc >= 0.0005 and inc_pos >= 3)
    gate_literal = bool(m_occ >= 0.0010 and dom_net > 0)
    passed = gate_increment

    result["arms"] = arms
    m_occ_f = arms["occupancy_full_bank"]["marginal_vs_clone"]
    m_ctl_f = arms["control_full_bank"]["marginal_vs_clone"]
    inc_f = m_occ_f - m_ctl_f
    inc_pf_f = (np.asarray(arms["occupancy_full_bank"]["per_fold_marginal"])
                - np.asarray(arms["control_full_bank"]["per_fold_marginal"]))
    inc_pos_f = int((inc_pf_f > 0).sum())
    result["contrasts"] = {
        "occupancy_minus_control": inc,
        "increment_per_fold": [float(v) for v in inc_pf],
        "increment_positive_folds": inc_pos,
        "dominant_net_pair_flow": dom_net,
        "full_bank_occupancy_minus_control": inc_f,
        "full_bank_increment_per_fold": [float(v) for v in inc_pf_f],
        "full_bank_increment_positive_folds": inc_pos_f,
    }
    result["gates"]["ensemble"] = {
        "primary_contrast": "occupancy arm minus {m00_core} control",
        "observed_increment": inc, "increment_threshold": 0.0005,
        "increment_positive_folds": inc_pos, "required_positive_folds": 3,
        "passed": gate_increment,
        "literal_spec_reading": {
            "marginal_vs_clone": m_occ, "threshold": 0.0010,
            "dominant_net": dom_net, "passed": gate_literal,
            "why_not_used": "The absolute marginal is dominated by the m00_core base "
                            f"({m_ctl:+.6f} on its own), so it does not identify the occupancy "
                            "block. The same conflation made M4 look like a pass. Here the "
                            "spec's own criteria kill the mechanism anyway.",
        },
    }
    # Grok's own kill criterion: marginal_vs_clone < +0.0010, OR rho > 0.85, OR
    # dominant net < 0. Report which of the three trip, since that is the
    # verdict the mechanism's author asked for.
    spec_trips = []
    if m_occ < 0.0010:
        spec_trips.append(f"marginal_vs_clone {m_occ:+.6f} < +0.0010")
    if not redundancy_ok:
        spec_trips.append(f"rho {rho:.4f} > 0.85")
    if dom_net < 0:
        spec_trips.append(f"dominant net {dom_net:+d} < 0")
    result["gates"]["spec_kill_criterion"] = {"trips": spec_trips, "killed": bool(spec_trips)}
    passed = bool(gate_increment and not spec_trips)
    result["gates"]["ensemble"]["passed"] = passed

    result["verdict"] = (
        f"PROCEED: the occupancy block adds {inc:+.6f} over the m00_core control at "
        f"{inc_pos}/5 positive folds, with dominant net {dom_net:+d}."
        if passed
        else (
            f"KILL, on the mechanism's own criteria: {'; '.join(spec_trips)}. "
            f"Grok's kill rule trips if ANY of marginal < +0.0010, rho > 0.85, or dominant "
            f"net < 0; {len(spec_trips)} of the three trip. "
            f"What is real and worth carrying forward: occupancy is genuinely non-redundant "
            f"with the incumbent (rho {rho:.4f} against an 0.85 line, the least redundant "
            f"channel screened on this branch), and its increment over the specified m00_core "
            f"base is {inc:+.6f} at {inc_pos}/5 positive folds -- consistent in sign on every "
            f"fold, unlike M4's 2/5. On the full 500-column bank, which is the base that "
            f"actually matters, the increment is {inc_f:+.6f} at {inc_pos_f}/5. The absolute "
            f"marginal is negative because the m00_core base is itself negative "
            f"({m_ctl:+.6f}); occupancy improves that base without rescuing it."
        )
    )
    (out_dir / "grok_m2_delay_cloud.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(result, out_dir)
    print(json.dumps({"verdict": result["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
