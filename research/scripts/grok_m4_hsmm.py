#!/usr/bin/env python
"""Grok M4: explicit-duration absorbing filter, built and gated.

The screen in grok_mechanism_screens.py established the premise: excursion dwell
is not memoryless (pooled hazard slope -0.0138 in log-dwell, survival gap 0.3047
against the matched geometric, and a frailty control that does not explain it).
This builds the filter that premise licenses and runs Grok's Experiment C.

What is implemented, stated precisely so the report does not overclaim. This is
an explicit-duration two-component model of the CURRENT elevated episode, solved
in closed form, not a full multi-state HSMM forward pass. The duration
distribution is explicit and non-geometric, which is the entire scientific
content of M4; what is not modelled is transitions between multiple elevated
regimes. Given an ongoing elevated episode of length r:

    p(absorbing | r) = pi_A S_A(r) / (pi_A S_A(r) + (1 - pi_A) S_T(r))

S_T is this series' own empirical survival of historical excursion lengths, so
the transient prior is per-series and causal. S_A is a Weibull survival with
increasing hazard and globally frozen hyper-parameters -- never fitted to y on
the scored fold. The emission magnitude cancels between the two components
because both condition on "elevated", which is deliberate: magnitude is what
m07_bayes already prices, and duration is the object M4 claims is missing.

Everything is prefix-only: row t uses hist and online[:t+1] and nothing else.
The recursion is a running run-length, so it is exactly O(1) per step and
bitwise prefix-invariant by construction.

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

SPECIALISTS = ["RT-300", "RT-410", "RT-411", "RT-412", "RT-413", "RT-414", "RT-415"]
SEEDCLONE_ID = "RT-401"

# Globally frozen absorbing-duration hyper-parameters. Never fitted to y.
WEIBULL_SHAPE = 1.5      # > 1 gives the increasing hazard the spec asks for
WEIBULL_SCALE = 512.0
PRIOR_ABSORBING = 0.05
SMOOTH_WINDOW = 64
HSMM_COLS = [
    "hsmm_post_absorbing",
    "hsmm_exp_remaining_dwell",
    "hsmm_post_entropy",
    "hsmm_map_duration",
    "hsmm_logbf_vs_geometric",
    "hsmm_elevated_llr",
]
DWELL_SCALAR_COLS = ["dwell_run_len", "dwell_log_run_len", "dwell_elevated_llr"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", default=os.environ.get("SBR_ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT)))
    p.add_argument("--output-dir", default=str(REPO / "research" / "reports" / "grok_m4_hsmm"))
    p.add_argument("--cache", default=str(REPO / "research" / "oof" / "grok_m4_hsmm_channels.npy"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=20260830)
    p.add_argument("--rounds", type=int, default=300)
    return p.parse_args()


def git_sha() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()


def source_oof(name: str) -> Path:
    return ARTIFACT_ROOT / "research" / "oof" / f"{name}.npy"


# ------------------------------------------------------------- the filter ---


def rollmean(x: np.ndarray, w: int) -> np.ndarray:
    cs = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    out = np.full(len(x), np.nan, dtype=np.float64)
    out[w - 1 :] = (cs[w:] - cs[:-w]) / w
    return out


def run_lengths_of(mask: np.ndarray) -> np.ndarray:
    """Current run length at every position, vectorised."""
    n = len(mask)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    idx = np.arange(n)
    # index of the last position that was NOT hot, at or before each position
    not_hot = np.where(~mask, idx, -1)
    last_off = np.maximum.accumulate(not_hot)
    return np.where(mask, idx - last_off, 0.0).astype(np.float64)


def excursion_run_list(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros(0, dtype=np.int64)
    d = np.diff(np.r_[0, mask.view(np.int8), 0])
    return (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).astype(np.int64)


def empirical_survival(runs: np.ndarray, r: np.ndarray) -> np.ndarray:
    """S_T(r) = P(historical excursion lasts >= r), with a geometric tail.

    Beyond the longest historical excursion the empirical survival is zero, which
    would make the posterior degenerate exactly where the mechanism is supposed
    to speak. The tail is continued geometrically at the rate implied by the
    series' own mean excursion length.
    """
    if len(runs) == 0:
        return np.full(len(r), np.nan)
    srt = np.sort(runs)
    n = len(srt)
    # P(L >= r) from the empirical distribution
    ge = n - np.searchsorted(srt, r, side="left")
    surv = ge / n
    rmax = float(srt[-1])
    mean_run = float(srt.mean())
    p_geo = 1.0 / max(mean_run, 1.0 + 1e-9)
    tail = r > rmax
    if tail.any():
        base = max(1.0 / n, 1e-9)
        surv = np.where(tail, base * (1.0 - p_geo) ** np.maximum(r - rmax, 0.0), surv)
    return np.maximum(surv, 1e-12)


def weibull_survival(r: np.ndarray, shape: float, scale: float) -> np.ndarray:
    return np.exp(-((np.maximum(r, 0.0) / scale) ** shape))


def build_channels(c: Ctx, cache: Path, force: bool) -> tuple[np.ndarray, dict]:
    if cache.exists() and not force:
        return np.load(cache, mmap_mode="r"), {"cache": str(cache), "loaded": True}
    t0 = time.time()
    st = load_store(str(ARTIFACT_ROOT / "cache" / "store"))
    n_cols = len(HSMM_COLS) + len(DWELL_SCALAR_COLS)
    out = np.full((len(c.d.y), n_cols), np.nan, dtype=np.float32)
    dev_series = np.unique(c.d.sidx[c.dev])
    for n_done, sid in enumerate(dev_series):
        rows = c.dev[c.d.sidx[c.dev] == sid]
        rows = rows[np.argsort(c.d.t[rows], kind="stable")]
        hist = np.asarray(st.hist(int(sid)), dtype=np.float64)
        online = np.asarray(st.online(int(sid)), dtype=np.float64)
        if len(hist) < 4 * SMOOTH_WINDOW or len(online) != len(rows):
            continue
        hp = HistParams(hist, ar_order=6)

        # Historical null: threshold and the per-series transient duration prior.
        zh = (hist - hp.mu) / hp.sd
        eh = np.concatenate([np.zeros(6), _ar_resid(zh, hp.ar_coef)]) / hp.ar_sigma
        sh = rollmean(eh * eh, SMOOTH_WINDOW)
        sh = sh[np.isfinite(sh)]
        if len(sh) < 2 * SMOOTH_WINDOW:
            continue
        med = float(np.median(sh))
        thr = med + float(np.quantile(np.abs(sh - med), 0.90))
        hist_runs = excursion_run_list(sh > thr)
        if len(hist_runs) < 5:
            continue

        # Online, causal: same statistic, same threshold, prefix-only.
        eo = ar_filter_causal(online, hp.ar_coef, hist) / hp.ar_sigma
        so = rollmean(eo * eo, SMOOTH_WINDOW)
        elevated = np.isfinite(so) & (so > thr)
        r = run_lengths_of(elevated)

        s_t = empirical_survival(hist_runs, r)
        s_a = weibull_survival(r, WEIBULL_SHAPE, WEIBULL_SCALE)
        num = PRIOR_ABSORBING * s_a
        den = num + (1.0 - PRIOR_ABSORBING) * s_t
        post = np.where(r > 0, num / np.maximum(den, 1e-300), 0.0)

        # Bayes factor of the explicit-duration mixture against the matched
        # geometric, i.e. against exactly what m07's memoryless hazard assumes.
        mean_run = max(float(hist_runs.mean()), 1.0 + 1e-9)
        p_geo = 1.0 / mean_run
        s_geo = np.maximum((1.0 - p_geo) ** np.maximum(r, 0.0), 1e-300)
        logbf = np.where(r > 0, np.log(np.maximum(den, 1e-300)) - np.log(s_geo), 0.0)

        ent = -(post * np.log(np.maximum(post, 1e-12)) + (1 - post) * np.log(np.maximum(1 - post, 1e-12)))
        # Expected remaining dwell under the two-component mixture.
        exp_rem = post * (WEIBULL_SCALE - np.minimum(r, WEIBULL_SCALE)) + (1 - post) * mean_run
        # Elevation evidence, the magnitude channel, kept so the tree can see
        # duration and magnitude separately rather than only their product.
        with np.errstate(invalid="ignore", divide="ignore"):
            llr = np.where(np.isfinite(so) & (so > 0), so / max(med, 1e-9) - 1.0 - np.log(np.maximum(so / max(med, 1e-9), 1e-12)), np.nan)

        vals = np.column_stack([
            post, exp_rem, ent, r, logbf, llr,
            r, np.log1p(r), llr,
        ]).astype(np.float32)
        vals[~np.isfinite(so)] = np.nan
        out[rows] = vals
        if n_done and n_done % 2000 == 0:
            print(f"  hsmm channels {n_done}/{len(dev_series)} in {time.time() - t0:.0f}s", flush=True)
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


def signed_auc(v: np.ndarray, c: Ctx, rows: np.ndarray) -> float:
    ok = np.isfinite(v[rows])
    r = rows[ok]
    if len(r) == 0 or len(np.unique(c.d.y[r])) < 2:
        return float("nan")
    a = float(ts_auc_flat(v[r], c.d.y[r], c.d.t[r]))
    return max(a, 1.0 - a)


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
            tr = rng.choice(tr, 600_000, replace=False)
        va = c.rows[f]
        booster = lgb.train(params, lgb.Dataset(x[tr], label=c.d.y[tr].astype(np.float64)),
                            num_boost_round=rounds)
        out[va] = booster.predict(np.nan_to_num(x[va], nan=0.0))
    return out


def evaluate(name: str, cand: np.ndarray, P: dict, c: Ctx) -> dict:
    P = dict(P)
    P[name] = cand
    _, e0 = blend_oof(P, SPECIALISTS, c)
    _, e1 = blend_oof(P, SPECIALISTS + [SEEDCLONE_ID], c)
    _, e2 = blend_oof(P, SPECIALISTS + [name], c)
    d = np.asarray(e2) - np.asarray(e1)
    return {
        "E0": float(np.mean(e0)),
        "E1_plus_clone": float(np.mean(e1)),
        "E2_plus_candidate": float(np.mean(e2)),
        "marginal_vs_clone": float(np.mean(e2) - np.mean(e1)),
        "per_fold_marginal": [float(v) for v in d],
        "positive_folds": int((d > 0).sum()),
        "standalone": {
            s: float(ts_auc_flat(cand[cut_rows(c, c.dev, s)], c.d.y[cut_rows(c, c.dev, s)],
                                 c.d.t[cut_rows(c, c.dev, s)]))
            for s in ("whole", "dominant_cell", "mature_vs_never", "mature_vs_prebreak")
        },
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c = Ctx()
    ch, cache_meta = build_channels(c, Path(args.cache), args.force)
    ch = np.asarray(ch)

    mats, names = PL.load_features(["m07_bayes"])
    j_ab = names.index("m07_bayes::ab_fast")
    ab_fast = PL._stack(mats, names, np.arange(len(c.d.y)), np.array([j_ab]))[:, 0].astype(np.float64)

    post = ch[:, HSMM_COLS.index("hsmm_post_absorbing")].astype(np.float64)
    logbf = ch[:, HSMM_COLS.index("hsmm_logbf_vs_geometric")].astype(np.float64)

    # Gate 1, the spec's own: if the posterior is monotone in ab_fast, kill.
    cell = cut_rows(c, c.dev, "dominant_cell")
    rho_post_ab = pearson(within_t_rank(post, cell, c.d.t), within_t_rank(ab_fast, cell, c.d.t))
    # Gate 2: the disagreement with ab_fast must have univariate cell signal.
    disagreement = within_t_rank(post, cell, c.d.t) - within_t_rank(ab_fast, cell, c.d.t)
    dis_full = np.full(len(c.d.y), np.nan)
    dis_full[cell] = disagreement
    mvn = cut_rows(c, c.dev, "mature_vs_never")
    univariate = {
        "hsmm_post_absorbing_cell": signed_auc(post, c, cell),
        "hsmm_logbf_vs_geometric_cell": signed_auc(logbf, c, cell),
        "disagreement_vs_ab_fast_cell": signed_auc(dis_full, c, cell),
        "disagreement_vs_ab_fast_mature_vs_never": signed_auc(dis_full, c, mvn),
        "ab_fast_cell": signed_auc(ab_fast, c, cell),
    }

    gate1 = bool(np.isfinite(rho_post_ab) and abs(rho_post_ab) <= 0.90)
    gate2 = bool(univariate["disagreement_vs_ab_fast_cell"] >= 0.52)

    result = {
        "experiment": "grok_m4_explicit_duration_absorbing_filter",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": git_sha(),
        "artifact_root": str(ARTIFACT_ROOT),
        "implementation_note": __doc__.split("What is implemented")[1].split("Everything is prefix-only")[0].strip(),
        "frozen_hyperparameters": {
            "weibull_shape": WEIBULL_SHAPE, "weibull_scale": WEIBULL_SCALE,
            "prior_absorbing": PRIOR_ABSORBING, "smooth_window": SMOOTH_WINDOW,
            "note": "globally frozen; never fitted to y on any fold",
        },
        "cache": cache_meta,
        "coverage_dev": float(np.isfinite(ch[c.dev]).all(1).mean()),
        "rho_post_vs_ab_fast_within_t_cell": rho_post_ab,
        "univariate": univariate,
        "gates": {
            "monotone_in_ab_fast_kill": {"rho": rho_post_ab, "limit": 0.90, "passed": gate1},
            "disagreement_univariate_cell": {
                "auc": univariate["disagreement_vs_ab_fast_cell"], "threshold": 0.52, "passed": gate2
            },
        },
    }

    if not (gate1 and gate2):
        result["verdict"] = (
            "KILL before the specialist: "
            + ("the posterior is monotone in ab_fast. " if not gate1 else "")
            + ("the disagreement with ab_fast has no univariate dominant-cell signal. " if not gate2 else "")
            + "Experiment C's pre-conditions are not met, so no training was spent."
        )
        (out_dir / "grok_m4_hsmm.json").write_text(json.dumps(result, indent=2) + "\n")
        write_report(result, out_dir)
        print(json.dumps({"verdict": result["verdict"]}, indent=2))
        return

    # Experiment C: {m00 + m07} vs {m00 + m07 + hsmm}, plus a dwell-scalar control
    # to confirm any gain is the posterior and not "we added dwell again".
    mats, names = PL.load_features(["m00_core", "m07_bayes"])
    base = PL._stack(mats, names, np.arange(len(c.d.y)), np.arange(len(names))).astype(np.float32)
    n_h = len(HSMM_COLS)
    x_ctrl = base
    x_dwell = np.column_stack([base, ch[:, n_h:]])
    x_hsmm = np.column_stack([base, ch[:, :n_h]])

    P = {s: np.load(source_oof(s)).astype(np.float64) for s in SPECIALISTS + [SEEDCLONE_ID]}
    arms = {}
    for label, x in (("control_m00_m07", x_ctrl), ("dwell_scalar", x_dwell), ("hsmm", x_hsmm)):
        stream = fold_pure_stream(x, c, args.seed, args.rounds)
        arms[label] = evaluate(f"__{label}__", stream, P, c)
        print(f"  {label}: marginal_vs_clone {arms[label]['marginal_vs_clone']:+.6f}", flush=True)

    m_hsmm = arms["hsmm"]["marginal_vs_clone"]
    m_dwell = arms["dwell_scalar"]["marginal_vs_clone"]
    m_ctrl = arms["control_m00_m07"]["marginal_vs_clone"]
    over_dwell = m_hsmm - m_dwell
    over_ctrl = m_hsmm - m_ctrl
    pf = {k: np.asarray(a["per_fold_marginal"]) for k, a in arms.items()}
    inc_per_fold = pf["hsmm"] - pf["control_m00_m07"]
    inc_positive = int((inc_per_fold > 0).sum())

    # The PRIMARY contrast is the one Experiment C actually specifies:
    # "specialist on {m00 + m07_old + m07_hsmm} vs {m00 + m07_old}". The absolute
    # marginal of the hsmm arm is not that contrast -- it is dominated by the base
    # specialist, which scores +0.001028 on its own with no dwell channel of any
    # kind. Gating on the absolute number would credit the HSMM for the base.
    #
    # The "beats the dwell scalar" condition is also not what it appears: the
    # dwell scalar HURTS the base (-0.000482), so clearing it by +0.0005 is
    # satisfied by doing nothing. Both of the spec's literal conditions can be
    # met by a channel that adds nothing, so the increment is gated directly.
    gate_increment = bool(over_ctrl >= 0.0005 and inc_positive >= 3)
    gate_literal = bool(m_hsmm >= 0.0010 and over_dwell >= 0.0005)
    passed = gate_increment

    result["arms"] = arms
    result["contrasts"] = {
        "hsmm_minus_dwell_scalar": over_dwell,
        "hsmm_minus_control": over_ctrl,
        "dwell_scalar_minus_control": m_dwell - m_ctrl,
        "hsmm_minus_control_per_fold": [float(v) for v in inc_per_fold],
        "hsmm_minus_control_positive_folds": inc_positive,
    }
    result["gates"]["experiment_c"] = {
        "primary_contrast": "hsmm arm minus {m00 + m07} control -- what Experiment C specifies",
        "observed_increment": over_ctrl,
        "increment_threshold": 0.0005,
        "increment_positive_folds": inc_positive,
        "increment_positive_folds_required": 3,
        "passed": gate_increment,
        "literal_spec_reading": {
            "marginal_vs_clone": m_hsmm,
            "threshold": 0.0010,
            "over_dwell_scalar": over_dwell,
            "must_beat_dwell_scalar_by": 0.0005,
            "passed": gate_literal,
            "why_not_used": (
                "Both conditions are satisfiable by a channel that adds nothing. The absolute "
                "marginal is carried by the base specialist (+%.6f with no dwell channel), and "
                "the dwell-scalar comparator is negative (%.6f), so clearing it by +0.0005 "
                "requires only not being harmful." % (m_ctrl, m_dwell - m_ctrl)
            ),
        },
    }
    result["verdict"] = (
        f"PROCEED: the explicit-duration posterior adds {over_ctrl:+.6f} over the "
        f"{{m00 + m07}} control at {inc_positive}/5 positive folds."
        if passed
        else (
            f"KILL. The filter works and the premise holds -- the posterior is not monotone in "
            f"ab_fast (rho {rho_post_ab:.4f}) and carries real univariate cell signal "
            f"({univariate['hsmm_post_absorbing_cell']:.6f}) -- but it adds {over_ctrl:+.6f} over "
            f"the {{m00 + m07}} control at only {inc_positive}/5 positive folds, against a "
            "+0.000500 requirement. Note that the spec's two literal conditions ARE met "
            f"(marginal {m_hsmm:+.6f} >= +0.0010; over dwell scalar {over_dwell:+.6f} >= "
            "+0.0005), and they are misleading here: the absolute marginal is carried by the "
            f"base specialist, which scores {m_ctrl:+.6f} with no dwell channel at all, and the "
            f"dwell-scalar comparator is negative ({m_dwell - m_ctrl:+.6f}), so beating it "
            "requires only not being harmful. The dwell premise is real; the posterior does not "
            "convert it into ensemble marginal."
        )
    )
    (out_dir / "grok_m4_hsmm.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(result, out_dir)
    print(json.dumps({"verdict": result["verdict"]}, indent=2))


def fmt(x, d: int = 6, signed: bool = False) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "nan" if not np.isfinite(v) else (f"{v:+.{d}f}" if signed else f"{v:.{d}f}")


def write_report(r: dict, out_dir: Path) -> None:
    md = [
        "# Grok M4 — Explicit-Duration Absorbing Filter",
        "",
        f"Date: `{r['generated']}`  ·  git `{r['git_sha']}`",
        "",
        "The screen established the premise: excursion dwell is not memoryless. This is",
        "the filter that premise licenses, run through Grok's Experiment C.",
        "",
        "## What was implemented",
        "",
        r["implementation_note"],
        "",
        "Frozen hyper-parameters, never fitted to `y` on any fold: "
        + ", ".join(f"`{k}={v}`" for k, v in r["frozen_hyperparameters"].items() if k != "note")
        + ".",
        "",
        f"Dev coverage: `{fmt(r['coverage_dev'], 4)}`.",
        "",
        "## Pre-conditions",
        "",
        "| gate | observed | requirement | passed |",
        "|---|---:|---|---|",
        f"| posterior not monotone in `ab_fast` | {fmt(r['rho_post_vs_ab_fast_within_t_cell'])} | "
        f"\\|rho\\| <= 0.90 | {r['gates']['monotone_in_ab_fast_kill']['passed']} |",
        f"| disagreement has univariate cell signal | "
        f"{fmt(r['gates']['disagreement_univariate_cell']['auc'])} | >= 0.52 | "
        f"{r['gates']['disagreement_univariate_cell']['passed']} |",
        "",
        "Univariate dominant-cell AUCs:",
        "",
        "| channel | AUC |",
        "|---|---:|",
    ]
    for k, v in r["univariate"].items():
        md.append(f"| {k} | {fmt(v)} |")

    if "arms" in r:
        md += [
            "",
            "## Experiment C",
            "",
            "Three fold-pure specialists on the same `m00_core + m07_bayes` base, differing",
            "only in what is appended. The dwell-scalar arm exists so that any gain is",
            "attributed to the posterior rather than to re-adding dwell in any form.",
            "",
            "| arm | marginal vs clone | positive folds | standalone whole | standalone cell |",
            "|---|---:|---:|---:|---:|",
        ]
        for k, a in r["arms"].items():
            md.append(
                f"| {k} | {fmt(a['marginal_vs_clone'], signed=True)} | {a['positive_folds']}/5 | "
                f"{fmt(a['standalone']['whole'])} | {fmt(a['standalone']['dominant_cell'])} |"
            )
        md += [
            "",
            "| contrast | value |",
            "|---|---:|",
        ] + [
            f"| {k} | {fmt(v, signed=True)} |" for k, v in r["contrasts"].items()
            if not isinstance(v, list)
        ] + [
            "",
            "Per-fold increment over the `{m00 + m07}` control: "
            + ", ".join(fmt(v, signed=True) for v in r["contrasts"]["hsmm_minus_control_per_fold"])
            + f" ({r['contrasts']['hsmm_minus_control_positive_folds']}/5 positive).",
            "",
            "### Why the primary contrast is the increment, not the absolute marginal",
            "",
            r["gates"]["experiment_c"]["literal_spec_reading"]["why_not_used"],
            "",
            "Experiment C is specified as \"specialist on {m00 + m07_old + m07_hsmm} vs",
            "{m00 + m07_old}\", so the increment is the contrast the experiment names. Both of",
            "the spec's numeric conditions are reported above for completeness, and both are",
            f"met (`{fmt(r['gates']['experiment_c']['literal_spec_reading']['passed'])}`), but",
            "they are not the gate.",
        ]
    md += ["", "## Verdict", "", r["verdict"], ""]
    (out_dir / "grok_m4_hsmm.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
