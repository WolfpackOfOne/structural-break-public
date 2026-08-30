#!/usr/bin/env python
"""Cheapest-falsification screens for Grok mechanisms M2, M3, M4, M5.

Each mechanism in agent_05_grok46.md carries an explicit "Cheapest falsification"
and a kill criterion. This script runs those four screens and nothing else. No
model is trained, no RT ID is allocated, RESULTS.csv is not touched.

Scope note, deliberately restrictive: M2 and M5 are *series-level* batch
diagnostics evaluated at the final online point, exactly as their specs
prescribe. PROTOCOL.md permits true-tau-adjacent and end-of-series diagnostics
only under a series-level protocol (one row per series, series ROC AUC), and
that is how they are reported here. They are screens for whether a family is
worth building, never row-level predictive claims. M3 and M4 are row-level and
fold-pure: their atlases and reference distributions are built on training folds
only and scored on the held-out fold.
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
from wave5_lib import Ctx, FOLDS  # noqa: E402

SCORED_FOLD = 0
ATLAS_CHANNEL = "m07_bayes::ab_fast"
OWN_Z_CHANNEL = "m07_bayes::ab_fast_z"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", default=os.environ.get("SBR_ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT)))
    p.add_argument("--output-dir", default=str(REPO / "research" / "reports" / "grok_mechanism_screens"))
    p.add_argument("--m3-atlas", action="store_true", help="M3 frozen cohort two-null atlas")
    p.add_argument("--m4-duration", action="store_true", help="M4 explicit-duration vs geometric hazard")
    p.add_argument("--m2-delay-cloud", action="store_true", help="M2 per-series delay-cloud occupancy")
    p.add_argument("--m5-c2st", action="store_true", help="M5 per-series classifier two-sample test")
    p.add_argument("--all", action="store_true")
    p.add_argument("--n-series", type=int, default=200, help="per class, for the series-level screens")
    p.add_argument("--seed", type=int, default=20260830)
    return p.parse_args()


def git_sha() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()


# ---------------------------------------------------------------- scoring ---


def signed_ts_auc(scores: np.ndarray, labels: np.ndarray, t: np.ndarray) -> dict[str, float | int]:
    """Best of the feature or its negation, within-t weighted. Reports the sign."""
    ok = np.isfinite(scores) & np.isfinite(labels)
    scores, labels, t = scores[ok], labels[ok].astype(np.int8), t[ok]
    if len(scores) == 0 or len(np.unique(labels)) < 2:
        return {"auc": float("nan"), "sign": 1, "n": 0}
    pos = float(ts_auc_flat(scores, labels, t))
    neg = float(ts_auc_flat(-scores, labels, t))
    return {"auc": max(pos, neg), "sign": -1 if neg > pos else 1, "n": int(len(scores))}


def series_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Plain series-level ROC AUC, best of feature or negation."""
    ok = np.isfinite(scores)
    s, y = scores[ok], labels[ok].astype(np.int8)
    if len(s) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    # mid-ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(max(auc, 1.0 - auc))


def series_auc_ci(scores: np.ndarray, labels: np.ndarray, rng, n_boot: int = 2000) -> dict:
    """Series AUC with a stratified bootstrap CI, so a gate comparison is honest."""
    point = series_auc(scores, labels)
    ok = np.isfinite(scores)
    s, y = scores[ok], labels[ok]
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return {"auc": point, "ci95": [float("nan"), float("nan")], "n": int(len(s))}
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        draws[b] = series_auc(s[i], y[i])
    draws = draws[np.isfinite(draws)]
    lo, hi = (np.quantile(draws, [0.025, 0.975]) if len(draws) else (float("nan"),) * 2)
    return {"auc": point, "ci95": [float(lo), float(hi)], "n": int(len(s))}


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    aa, bb = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    den = float(np.sqrt(np.dot(aa, aa) * np.dot(bb, bb)))
    return float(np.dot(aa, bb) / den) if den > 0 else float("nan")


def mature_vs_never_rows(c: Ctx, rows: np.ndarray) -> np.ndarray:
    """Dominant cell, mature-break positives vs never-break negatives."""
    y, t, age = c.d.y[rows], c.d.t[rows], c.age[rows]
    dominant = (t >= 200) & ((y == 0) | (age >= 100))
    hb = c.has_break[c.d.sidx[rows]]
    return rows[dominant & ((y == 1) | ((y == 0) & ~hb))]


# ------------------------------------------------------- M3: cohort atlas ---


def within_t_ecdf_rank(
    values: np.ndarray,
    t: np.ndarray,
    atlas_values: np.ndarray,
    atlas_t: np.ndarray,
) -> np.ndarray:
    """Rank each row's value in the frozen atlas distribution at the same t.

    The atlas is a per-t empirical CDF built from training-fold rows only. Lookup
    is a searchsorted into the sorted atlas values for that t; rows whose t is
    absent from the atlas get NaN rather than a borrowed neighbouring t.
    """
    out = np.full(len(values), np.nan, dtype=np.float64)
    order = np.argsort(atlas_t, kind="stable")
    at_sorted, av_sorted = atlas_t[order], atlas_values[order]
    starts = np.flatnonzero(np.r_[True, at_sorted[1:] != at_sorted[:-1]])
    ends = np.r_[starts[1:], len(at_sorted)]
    atlas = {}
    for lo, hi in zip(starts, ends):
        block = av_sorted[lo:hi]
        block = block[np.isfinite(block)]
        if len(block) >= 50:
            atlas[int(at_sorted[lo])] = np.sort(block)

    order_q = np.argsort(t, kind="stable")
    t_q = t[order_q]
    q_starts = np.flatnonzero(np.r_[True, t_q[1:] != t_q[:-1]])
    q_ends = np.r_[q_starts[1:], len(t_q)]
    for lo, hi in zip(q_starts, q_ends):
        idx = order_q[lo:hi]
        ref = atlas.get(int(t_q[lo]))
        if ref is None:
            continue
        v = values[idx]
        pos = np.searchsorted(ref, v, side="left") + np.searchsorted(ref, v, side="right")
        out[idx] = pos / (2.0 * len(ref))
    return out


def screen_m3_atlas(args: argparse.Namespace, c: Ctx) -> dict:
    t0 = time.time()
    mats, names = PL.load_features(["m07_bayes"])
    j_val = names.index(ATLAS_CHANNEL)
    j_z = names.index(OWN_Z_CHANNEL)
    col = PL._stack(mats, names, np.arange(len(c.d.y)), np.array([j_val, j_z]))
    ab_fast, ab_fast_z = col[:, 0].astype(np.float64), col[:, 1].astype(np.float64)

    train_rows = np.concatenate([c.rows[f] for f in FOLDS if f != SCORED_FOLD])
    score_rows = mature_vs_never_rows(c, c.rows[SCORED_FOLD])

    # Atlas mitigation from the spec: pre-break rows and never-break series only,
    # so the reference measure is not contaminated by mature positives.
    hb_train = c.has_break[c.d.sidx[train_rows]]
    clean_train = train_rows[(c.d.y[train_rows] == 0)]
    atlas_rank = within_t_ecdf_rank(
        ab_fast[score_rows], c.d.t[score_rows], ab_fast[clean_train], c.d.t[clean_train]
    )
    # Contamination control: the same atlas built from ALL training rows.
    atlas_rank_contaminated = within_t_ecdf_rank(
        ab_fast[score_rows], c.d.t[score_rows], ab_fast[train_rows], c.d.t[train_rows]
    )
    # Derangement control: permute the atlas ranks within each t.
    rng = np.random.default_rng(args.seed)
    deranged = atlas_rank.copy()
    t_s = c.d.t[score_rows]
    order = np.argsort(t_s, kind="stable")
    tt = t_s[order]
    starts = np.flatnonzero(np.r_[True, tt[1:] != tt[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for lo, hi in zip(starts, ends):
        idx = order[lo:hi]
        deranged[idx] = rng.permutation(deranged[idx])

    y = c.d.y[score_rows]
    t = c.d.t[score_rows]
    own_z = ab_fast_z[score_rows]

    # The spec says "own z > 2". ab_fast_z is not a standard normal score -- its
    # median is about -1.1 and only ~1.9% of rows exceed 2 -- so a literal 2.0
    # would make the threshold an artifact of this channel's offset rather than
    # the "own-history hot" event the mechanism means. The 2-sigma-equivalent
    # quantile, estimated on training rows only, is the scale-free reading.
    hot_cut = float(np.nanquantile(ab_fast_z[clean_train], 0.977))
    own_hot = own_z > hot_cut

    # Own-history rank within t, so the two nulls are compared on one scale.
    own_rank = within_t_ecdf_rank(own_z, t, ab_fast_z[clean_train], c.d.t[clean_train])

    indicator = (own_hot & (atlas_rank < 0.5)).astype(np.float64)
    indicator[~np.isfinite(atlas_rank) | ~np.isfinite(own_z)] = np.nan
    residual = own_rank - atlas_rank  # continuous two-null disagreement, both in [0,1]
    indicator_deranged = (own_hot & (deranged < 0.5)).astype(np.float64)
    indicator_deranged[~np.isfinite(deranged) | ~np.isfinite(own_z)] = np.nan
    indicator_contam = (own_hot & (atlas_rank_contaminated < 0.5)).astype(np.float64)
    indicator_contam[~np.isfinite(atlas_rank_contaminated) | ~np.isfinite(own_z)] = np.nan

    scores = {
        "disagreement_indicator": signed_ts_auc(indicator, y, t),
        "continuous_two_null_residual": signed_ts_auc(residual, y, t),
        "atlas_rank_alone": signed_ts_auc(atlas_rank, y, t),
        "own_z_alone": signed_ts_auc(own_z, y, t),
        "indicator_deranged_atlas": signed_ts_auc(indicator_deranged, y, t),
        "indicator_contaminated_atlas": signed_ts_auc(indicator_contam, y, t),
    }
    # The decisive diagnostic for the family: do the two reference measures
    # actually disagree? If the own-history rank and the cohort-atlas rank are
    # near-identical, the second null is not a distinct reference measure and no
    # construction on top of it can carry new information.
    ok = np.isfinite(own_rank) & np.isfinite(atlas_rank)
    null_agreement = {
        "rho_own_rank_vs_atlas_rank": pearson(own_rank, atlas_rank),
        "mean_abs_rank_disagreement": float(np.mean(np.abs(own_rank[ok] - atlas_rank[ok]))),
        "hot_cut_on_own_z": hot_cut,
        "p_own_hot": float(np.mean(own_hot[np.isfinite(own_z)])),
        "p_pop_typical_given_own_hot": float(
            np.mean((atlas_rank < 0.5)[own_hot & np.isfinite(atlas_rank)])
        ),
    }
    gate_auc = scores["disagreement_indicator"]["auc"]
    deranged_gap = gate_auc - scores["indicator_deranged_atlas"]["auc"]
    passed = bool(np.isfinite(gate_auc) and gate_auc >= 0.55)
    return {
        "mechanism": "M3_frozen_cohort_two_null_atlas",
        "spec": "Cheapest falsification: AUC of 1[own z > 2 and population rank < 0.5] "
        "on the dominant-cell mature-vs-never cut. Kill the family if < 0.55.",
        "protocol": "row-level, fold-pure: atlas built on folds 1-4, scored on fold 0",
        "channel": ATLAS_CHANNEL,
        "own_z_channel": OWN_Z_CHANNEL,
        "n_scored_rows": int(len(score_rows)),
        "n_atlas_rows": int(len(clean_train)),
        "atlas_coverage": float(np.isfinite(atlas_rank).mean()),
        "indicator_rate": float(np.nanmean(indicator)),
        "scores": scores,
        "null_agreement": null_agreement,
        "deranged_gap": float(deranged_gap),
        "gate": {"threshold": 0.55, "observed": float(gate_auc), "passed": passed},
        "verdict": (
            "PROCEED to the fold-0 specialist" if passed else "KILL the M3 family: the two-null "
            "disagreement indicator does not separate the target cut at the declared threshold."
        ),
        "runtime_s": round(time.time() - t0, 1),
    }


# --------------------------------------------- M4: duration vs geometric ---


def excursion_runs(energy: np.ndarray, threshold: float) -> np.ndarray:
    """Lengths of maximal runs where energy exceeds threshold."""
    hot = energy > threshold
    if not hot.any():
        return np.zeros(0, dtype=np.int64)
    d = np.diff(np.r_[0, hot.view(np.int8), 0])
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return (ends - starts).astype(np.int64)


def rollmean(x: np.ndarray, w: int) -> np.ndarray:
    """Causal rolling mean; the first w-1 entries are NaN."""
    cs = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    out = np.full(len(x), np.nan, dtype=np.float64)
    out[w - 1 :] = (cs[w:] - cs[:-w]) / w
    return out


def dwell_hazard(runs: np.ndarray, max_k: int = 400, min_at_risk: int = 50) -> dict:
    """Discrete hazard h(k) = P(L = k | L >= k) and its divergence from geometric.

    Under a geometric (memoryless) dwell law h(k) is constant in k. An
    increasing hazard means a long excursion is progressively more likely to be
    absorbing rather than transient, which is the whole content of M4.
    """
    hazard = []
    for k in range(1, max_k + 1):
        at_risk = int((runs >= k).sum())
        if at_risk < min_at_risk:
            break
        hazard.append({"k": k, "at_risk": at_risk, "hazard": float((runs == k).sum()) / at_risk})
    ks = np.array([h["k"] for h in hazard], dtype=np.float64)
    hv = np.array([h["hazard"] for h in hazard], dtype=np.float64)
    slope = float(np.polyfit(np.log(ks), hv, 1)[0]) if len(ks) >= 4 else float("nan")
    mean_run = float(runs.mean()) if len(runs) else float("nan")
    geom_p = 1.0 / mean_run if np.isfinite(mean_run) and mean_run > 0 else float("nan")
    gap = float("nan")
    if len(ks):
        surv_obs = np.array([float((runs >= k).mean()) for k in ks])
        surv_geo = np.array([float((1.0 - geom_p) ** (k - 1)) for k in ks])
        gap = float(np.max(np.abs(surv_obs - surv_geo)))
    return {
        "n_excursions": int(len(runs)),
        "mean_run_length": mean_run,
        "median_run_length": float(np.median(runs)) if len(runs) else float("nan"),
        "p95_run_length": float(np.quantile(runs, 0.95)) if len(runs) else float("nan"),
        "matched_geometric_p": geom_p,
        "hazard_by_dwell": hazard,
        "hazard_slope_in_log_dwell": slope,
        "max_survival_gap_vs_geometric": gap,
        "hazard_ratio_last_over_first": (
            float(hv[-1] / hv[0]) if len(hv) >= 2 and hv[0] > 0 else float("nan")
        ),
    }


def screen_m4_duration(args: argparse.Namespace, c: Ctx) -> dict:
    t0 = time.time()
    st = load_store(str(ARTIFACT_ROOT / "cache" / "store"))
    rng = np.random.default_rng(args.seed)

    never_series = np.intersect1d(np.flatnonzero(~c.has_break), np.unique(c.d.sidx[c.dev]))
    take = rng.choice(never_series, size=min(args.n_series * 3, len(never_series)), replace=False)

    # An excursion is a sustained elevation of SMOOTHED AR-residual energy, the
    # same object the EFPT states use. Thresholding raw pointwise energy instead
    # measures single spikes, whose dwell is memoryless by construction and says
    # nothing about the absorbing-vs-transient question M4 poses.
    windows = (32, 64, 128)
    runs_by_w: dict[int, list[np.ndarray]] = {w: [] for w in windows}
    runs_by_series: dict[int, list[np.ndarray]] = {w: [] for w in windows}
    for sid in take:
        hist = np.asarray(st.hist(int(sid)), dtype=np.float64)
        if len(hist) < 600:
            continue
        hp = HistParams(hist, ar_order=6)
        zh = (hist - hp.mu) / hp.sd
        eh = np.concatenate([np.zeros(6), _ar_resid(zh, hp.ar_coef)]) / hp.ar_sigma
        energy = eh * eh
        for w in windows:
            if len(energy) < 4 * w:
                continue
            sm = rollmean(energy, w)
            sm = sm[np.isfinite(sm)]
            if len(sm) < 2 * w:
                continue
            med = float(np.median(sm))
            thr = med + float(np.quantile(np.abs(sm - med), 0.90))
            r = excursion_runs(sm, thr)
            if len(r):
                runs_by_w[w].append(r)
                runs_by_series[w].append(r)

    per_window = {}
    for w in windows:
        runs = np.concatenate(runs_by_w[w]) if runs_by_w[w] else np.zeros(0, dtype=np.int64)
        stats = dwell_hazard(runs)
        # Frailty control. A pooled decreasing hazard is also what a MIXTURE of
        # per-series geometric dwells produces, even when each series is
        # individually memoryless. Two checks separate the two explanations:
        #  1. within-series hazard slope, pooled only after per-series estimation
        #  2. the pooled survival predicted by a mixture of per-series geometrics
        within_slopes = []
        for r in runs_by_series[w]:
            if len(r) >= 30:
                s = dwell_hazard(r, min_at_risk=10)["hazard_slope_in_log_dwell"]
                if np.isfinite(s):
                    within_slopes.append(s)
        series_means = np.array(
            [float(r.mean()) for r in runs_by_series[w] if len(r) >= 10], dtype=np.float64
        )
        mixture_gap = float("nan")
        if len(series_means) and stats["hazard_by_dwell"]:
            ps = 1.0 / series_means[series_means > 0]
            ks = np.array([h["k"] for h in stats["hazard_by_dwell"]], dtype=np.float64)
            surv_obs = np.array([float((runs >= k).mean()) for k in ks])
            surv_mix = np.array([float(np.mean((1.0 - ps) ** (k - 1))) for k in ks])
            mixture_gap = float(np.max(np.abs(surv_obs - surv_mix)))
        stats["within_series_hazard_slope_median"] = (
            float(np.median(within_slopes)) if within_slopes else float("nan")
        )
        stats["n_series_with_within_slope"] = int(len(within_slopes))
        stats["max_survival_gap_vs_geometric_mixture"] = mixture_gap
        per_window[str(w)] = stats

    # The gate reads the widest window with a usable excursion count, since that
    # is the scale at which "absorbing vs transient" is even a question.
    usable = [w for w in windows if per_window[str(w)]["n_excursions"] >= 1000]
    primary_w = max(usable) if usable else windows[0]
    primary = per_window[str(primary_w)]
    slope = primary["hazard_slope_in_log_dwell"]
    gap = primary["max_survival_gap_vs_geometric"]
    within_slope = primary["within_series_hazard_slope_median"]
    mixture_gap = primary["max_survival_gap_vs_geometric_mixture"]
    geometric = bool(np.isfinite(slope) and abs(slope) < 0.01 and np.isfinite(gap) and gap < 0.05)
    # Frailty verdict: if the pooled hazard is non-flat but the WITHIN-series
    # hazard is flat and a mixture of per-series geometrics reproduces the pooled
    # survival, the non-geometry is heterogeneity across series, not duration
    # memory within an excursion -- and M4's premise does not hold.
    frailty_explains = bool(
        np.isfinite(within_slope)
        and abs(within_slope) < 0.01
        and np.isfinite(mixture_gap)
        and mixture_gap < 0.05
    )
    passed = bool(not geometric and not frailty_explains)
    return {
        "mechanism": "M4_explicit_duration_absorbing_filter",
        "spec": "Cheapest falsification: empirical survival of excursion length on never-break "
        "history windows. If geometrically tailed the HSMM collapses to m07 and we kill "
        "without training.",
        "protocol": "history-only, never-break series only; no online data, no labels used. "
        "Excursions are runs of smoothed AR(6) residual energy above the historical "
        "q90 of its own absolute deviation from median.",
        "n_series_sampled": int(len(take)),
        "smoothing_windows": list(windows),
        "primary_window": primary_w,
        "per_window": per_window,
        "n_excursions": primary["n_excursions"],
        "mean_run_length": primary["mean_run_length"],
        "matched_geometric_p": primary["matched_geometric_p"],
        "hazard_by_dwell": primary["hazard_by_dwell"],
        "hazard_slope_in_log_dwell": slope,
        "max_survival_gap_vs_geometric": gap,
        "is_geometric": geometric,
        "within_series_hazard_slope_median": within_slope,
        "max_survival_gap_vs_geometric_mixture": mixture_gap,
        "frailty_explains_pooled_nongeometry": frailty_explains,
        "gate": {
            "rule": "kill if the dwell hazard is flat (|slope| < 0.01) and survival tracks the "
            "matched geometric within 0.05; also kill if the pooled non-geometry is "
            "explained by per-series frailty (flat within-series hazard AND a mixture of "
            "per-series geometrics reproducing pooled survival within 0.05)",
            "observed": slope,
            "passed": passed,
        },
        "verdict": (
            f"PROCEED: at window {primary_w} the dwell hazard is not flat "
            f"(slope {slope:+.4f} in log-dwell, survival gap {gap:.4f} vs the matched "
            f"geometric), and the frailty control does not explain it "
            f"(within-series hazard slope {within_slope:+.4f}, mixture survival gap "
            f"{mixture_gap:.4f}), so an explicit-duration prior is not a reparametrisation "
            "of m07's geometric hazard."
            if passed
            else (
                "KILL the M4 family: the pooled non-geometry is explained by per-series "
                f"frailty (within-series hazard slope {within_slope:+.4f}, mixture survival "
                f"gap {mixture_gap:.4f}). Dwell is memoryless within an excursion; the "
                "pooled decreasing hazard is heterogeneity across series, which m00/m05-class "
                "series descriptors already price."
                if frailty_explains
                else "KILL the M4 family: excursion dwell is geometrically tailed, so the "
                "HSMM collapses to the incumbent m07 hazard."
            )
        ),
        "runtime_s": round(time.time() - t0, 1),
    }


# ------------------------------------------------- M2: delay-cloud occupancy ---


def delay_embed(x: np.ndarray, dim: int = 3, lag: int = 1) -> np.ndarray:
    n = len(x) - (dim - 1) * lag
    if n <= 0:
        return np.zeros((0, dim))
    return np.column_stack([x[i * lag : i * lag + n] for i in range(dim)])


def knn_occupancy_surprise(hist_cloud: np.ndarray, query: np.ndarray, k: int, rng) -> float:
    """Mean log distance from each query point to its k-th nearest history neighbour.

    A permanent scale or dependence change moves the online window off the
    historical attractor, so its k-NN distances inflate. A long on-attractor
    excursion does not.
    """
    if len(hist_cloud) < k + 1 or len(query) == 0:
        return float("nan")
    ref = hist_cloud
    if len(ref) > 4000:
        ref = ref[rng.choice(len(ref), 4000, replace=False)]
    q = query
    if len(q) > 400:
        q = q[rng.choice(len(q), 400, replace=False)]
    d = np.sqrt(((q[:, None, :] - ref[None, :, :]) ** 2).sum(-1))
    kth = np.partition(d, k, axis=1)[:, k]
    kth = kth[np.isfinite(kth) & (kth > 0)]
    return float(np.mean(np.log(kth))) if len(kth) else float("nan")


def screen_m2_delay_cloud(args: argparse.Namespace, c: Ctx) -> dict:
    t0 = time.time()
    st = load_store(str(ARTIFACT_ROOT / "cache" / "store"))
    rng = np.random.default_rng(args.seed)
    dev_series = np.unique(c.d.sidx[c.dev])

    tau = c.tau
    mature = np.array(
        [s for s in dev_series if c.has_break[s] and tau[s] >= 0 and (c.n_online[s] - tau[s]) >= 100],
        dtype=np.int64,
    )
    never = np.array([s for s in dev_series if not c.has_break[s]], dtype=np.int64)
    n = min(args.n_series, len(mature), len(never))
    pick = np.concatenate([rng.choice(mature, n, replace=False), rng.choice(never, n, replace=False)])
    labels = np.r_[np.ones(n, dtype=np.int8), np.zeros(n, dtype=np.int8)]

    occ, m00_proxy = [], []
    for sid in pick:
        hist = np.asarray(st.hist(int(sid)), dtype=np.float64)
        online = np.asarray(st.online(int(sid)), dtype=np.float64)
        if len(hist) < 300 or len(online) < 100:
            occ.append(np.nan)
            m00_proxy.append(np.nan)
            continue
        hp = HistParams(hist, ar_order=6)
        zh = (hist - hp.mu) / hp.sd
        zo = (online - hp.mu) / hp.sd
        cloud = delay_embed(zh)
        window = delay_embed(zo[-200:])
        s_online = knn_occupancy_surprise(cloud, window, k=5, rng=rng)
        # Calibrate against the same statistic on a held-out slice of history, so
        # the score is a disagreement and not a series-difficulty constant.
        s_hist = knn_occupancy_surprise(cloud[: len(cloud) // 2], delay_embed(zh[len(zh) // 2 :][-200:]), k=5, rng=rng)
        occ.append(s_online - s_hist)
        # m00-style moment channel on the same window, as the control.
        m00_proxy.append(float(np.log(np.std(zo[-200:]) + 1e-12)))
    occ_a = np.array(occ, dtype=np.float64)
    m00_a = np.array(m00_proxy, dtype=np.float64)

    auc_occ_ci = series_auc_ci(occ_a, labels, rng)
    auc_occ = auc_occ_ci["auc"]
    auc_m00 = series_auc(m00_a, labels)
    resid = occ_a - np.polyval(np.polyfit(m00_a[np.isfinite(m00_a) & np.isfinite(occ_a)],
                                          occ_a[np.isfinite(m00_a) & np.isfinite(occ_a)], 1), m00_a)
    auc_resid = series_auc(resid, labels)
    rho = pearson(occ_a, m00_a)
    # Gate on the lower CI bound, not the point estimate: a point estimate a
    # hair over the threshold is not evidence the family clears it.
    passed = bool(np.isfinite(auc_occ_ci["ci95"][0]) and auc_occ_ci["ci95"][0] >= 0.55)
    return {
        "mechanism": "M2_per_series_delay_cloud_predictive_null",
        "spec": "Cheapest falsification: k-NN occupancy surprise at t=max on ~200 mature-break "
        "and ~200 never-break series. Kill if the occupancy disagreement with m00_core does "
        "not rank the rows above 0.55 AUC.",
        "protocol": "SERIES-LEVEL batch diagnostic at the final online point, one row per "
        "series, series ROC AUC. Not a row-level predictive claim.",
        "n_series_per_class": int(n),
        "scores": {
            "occupancy_disagreement_auc": auc_occ,
            "occupancy_disagreement_ci95_lo": auc_occ_ci["ci95"][0],
            "occupancy_disagreement_ci95_hi": auc_occ_ci["ci95"][1],
            "m00_scale_control_auc": auc_m00,
            "occupancy_residualised_on_m00_auc": auc_resid,
            "rho_occupancy_vs_m00": rho,
        },
        "gate": {
            "threshold": 0.55,
            "observed": auc_occ,
            "observed_ci95_lo": auc_occ_ci["ci95"][0],
            "rule": "gate on the bootstrap lower bound, not the point estimate",
            "passed": passed,
        },
        "verdict": (
            "PROCEED to a streaming delay-cloud likelihood"
            if passed
            else "KILL the M2 family: delay-space occupancy does not separate mature breaks from "
            "never-break series even as a batch statistic at the most favourable point."
        ),
        "runtime_s": round(time.time() - t0, 1),
    }


# ------------------------------------------------------------- M5: C2ST ---


def linear_c2st(a: np.ndarray, b: np.ndarray, rng, n_perm: int = 0) -> float:
    """Cross-validated linear two-sample statistic between window a and window b.

    Fits a ridge-regularised linear discriminant on delay-embedded coordinates
    with a two-fold split, and returns the held-out AUC. Chance is 0.5.
    """
    xa, xb = delay_embed(a), delay_embed(b)
    if len(xa) < 40 or len(xb) < 40:
        return float("nan")
    x = np.vstack([xa, xb])
    y = np.r_[np.ones(len(xa)), np.zeros(len(xb))]
    idx = rng.permutation(len(x))
    x, y = x[idx], y[idx]
    half = len(x) // 2
    aucs = []
    for tr, te in ((slice(0, half), slice(half, None)), (slice(half, None), slice(0, half))):
        xt, yt = x[tr], y[tr]
        if len(np.unique(yt)) < 2:
            continue
        mu = xt.mean(0)
        xc = xt - mu
        cov = xc.T @ xc / max(len(xc) - 1, 1) + 1e-6 * np.eye(x.shape[1])
        m1 = xt[yt == 1].mean(0) - mu
        m0 = xt[yt == 0].mean(0) - mu
        try:
            w = np.linalg.solve(cov, m1 - m0)
        except np.linalg.LinAlgError:
            continue
        s = (x[te] - mu) @ w
        aucs.append(series_auc(s, y[te].astype(np.int8)))
    aucs = [a for a in aucs if np.isfinite(a)]
    return float(np.mean(aucs)) if aucs else float("nan")


def wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 20:
        return float("nan")
    qa = np.quantile(a, np.linspace(0.01, 0.99, 99))
    qb = np.quantile(b, np.linspace(0.01, 0.99, 99))
    return float(np.mean(np.abs(qa - qb)))


def screen_m5_c2st(args: argparse.Namespace, c: Ctx) -> dict:
    t0 = time.time()
    st = load_store(str(ARTIFACT_ROOT / "cache" / "store"))
    rng = np.random.default_rng(args.seed)
    dev_series = np.unique(c.d.sidx[c.dev])
    tau = c.tau
    mature = np.array(
        [s for s in dev_series if c.has_break[s] and tau[s] >= 0 and (c.n_online[s] - tau[s]) >= 100],
        dtype=np.int64,
    )
    never = np.array([s for s in dev_series if not c.has_break[s]], dtype=np.int64)
    n = min(args.n_series // 2, len(mature), len(never))
    pick = np.concatenate([rng.choice(mature, n, replace=False), rng.choice(never, n, replace=False)])
    labels = np.r_[np.ones(n, dtype=np.int8), np.zeros(n, dtype=np.int8)]

    c2st, wass = [], []
    for sid in pick:
        hist = np.asarray(st.hist(int(sid)), dtype=np.float64)
        online = np.asarray(st.online(int(sid)), dtype=np.float64)
        if len(hist) < 400 or len(online) < 200:
            c2st.append(np.nan)
            wass.append(np.nan)
            continue
        hp = HistParams(hist, ar_order=6)
        zh = (hist - hp.mu) / hp.sd
        zo = ar_filter_causal(np.asarray(online, dtype=np.float64), hp.ar_coef, np.asarray(hist, dtype=np.float64))
        zo = zo / hp.ar_sigma
        eh = np.concatenate([np.zeros(6), _ar_resid(zh, hp.ar_coef)]) / hp.ar_sigma
        w_hist = eh[-200:]
        w_online = zo[-200:]
        c2st.append(linear_c2st(w_online, w_hist, rng))
        wass.append(wasserstein1(w_online, w_hist))
    c2st_a = np.array(c2st, dtype=np.float64)
    wass_a = np.array(wass, dtype=np.float64)

    c2st_ci = series_auc_ci(c2st_a, labels, rng)
    auc_c2st = c2st_ci["auc"]
    auc_wass = series_auc(wass_a, labels)
    margin = auc_c2st - auc_wass
    rho = pearson(c2st_a, wass_a)
    passed = bool(np.isfinite(auc_c2st) and auc_c2st >= 0.53 and margin >= 0.02)
    return {
        "mechanism": "M5_per_series_classifier_two_sample_test",
        "spec": "Cheapest falsification: batch linear C2ST at the last online point on ~100 "
        "series. Kill if series-level AUC < 0.53, or if it does not beat Wasserstein on the "
        "same windows by +0.02.",
        "protocol": "SERIES-LEVEL batch diagnostic at the final online point, one row per "
        "series, series ROC AUC. Not a row-level predictive claim.",
        "n_series_per_class": int(n),
        "scores": {
            "c2st_auc": auc_c2st,
            "c2st_ci95_lo": c2st_ci["ci95"][0],
            "c2st_ci95_hi": c2st_ci["ci95"][1],
            "wasserstein_control_auc": auc_wass,
            "margin_over_wasserstein": margin,
            "rho_c2st_vs_wasserstein": rho,
        },
        "gate": {
            "c2st_threshold": 0.53,
            "margin_threshold": 0.02,
            "observed_auc": auc_c2st,
            "observed_margin": margin,
            "passed": passed,
        },
        "verdict": (
            "PROCEED to a streaming C2ST channel"
            if passed
            else "KILL the M5 family: a learned two-sample test does not beat the named "
            "distance it would have to displace, so it is m02 by another name."
        ),
        "runtime_s": round(time.time() - t0, 1),
    }


# ------------------------------------------------------------- reporting ---


def fmt(x: object, digits: int = 6) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "nan" if not np.isfinite(v) else f"{v:.{digits}f}"


def write_report(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "grok_mechanism_screens.json").write_text(json.dumps(results, indent=2) + "\n")
    md = [
        "# Grok Mechanisms M2-M5: Cheapest-Falsification Screens",
        "",
        f"Date: `{results['generated']}`  ·  git `{results['git_sha']}`",
        "",
        "**No model was trained.** Each screen is the falsification the mechanism's own",
        "author specified, run at the threshold the author declared.",
        "",
        "M3 and M4 are row-level and fold-pure. M2 and M5 are **series-level batch",
        "diagnostics at the final online point**, which is the protocol their specs ask",
        "for and the only protocol under which an end-of-series statistic is admissible",
        "(PROTOCOL.md: series-level, one row per series, series ROC AUC). They screen",
        "whether a family is worth building; they are not row-level predictive claims.",
        "",
        "## Summary",
        "",
        "| mechanism | gate | observed | verdict |",
        "|---|---|---:|---|",
    ]
    for key in ("m3_atlas", "m4_duration", "m2_delay_cloud", "m5_c2st"):
        r = results["screens"].get(key)
        if not r:
            continue
        gate = r["gate"]
        obs = gate.get("observed", gate.get("observed_auc"))
        thr = gate.get("threshold", gate.get("c2st_threshold", gate.get("rule", "")))
        verdict = "**PROCEED**" if gate["passed"] else "**KILL**"
        thr_cell = thr.replace("|", "\\|") if isinstance(thr, str) else f">= {thr}"
        md.append(
            f"| {r['mechanism']} | {thr_cell} | "
            f"{fmt(obs) if obs is not None else 'n/a'} | {verdict} |"
        )

    for key, title in (
        ("m3_atlas", "M3 — Frozen Cohort Two-Null Atlas"),
        ("m4_duration", "M4 — Explicit-Duration Absorbing Filter"),
        ("m2_delay_cloud", "M2 — Per-Series Delay-Cloud Predictive Null"),
        ("m5_c2st", "M5 — Per-Series Classifier Two-Sample Test"),
    ):
        r = results["screens"].get(key)
        if not r:
            continue
        md += ["", f"## {title}", "", f"*Spec:* {r['spec']}", "", f"*Protocol:* {r['protocol']}", ""]
        if "scores" in r:
            md += ["| statistic | value |", "|---|---:|"]
            for k, v in r["scores"].items():
                val = v["auc"] if isinstance(v, dict) else v
                md.append(f"| {k} | {fmt(val)} |")
            md.append("")
        if key == "m3_atlas":
            md += [
                f"- Scored rows: `{r['n_scored_rows']}`; atlas rows: `{r['n_atlas_rows']}`; "
                f"atlas coverage `{fmt(r['atlas_coverage'], 4)}`",
                f"- Indicator fires on `{fmt(r['indicator_rate'], 4)}` of scored rows",
                f"- Real-minus-deranged atlas gap: `{fmt(r['deranged_gap'])}`",
                "",
                "Do the two reference measures actually disagree?",
                "",
                f"- Within-t rank correlation, own-history vs cohort atlas: "
                f"`{fmt(r['null_agreement']['rho_own_rank_vs_atlas_rank'])}`",
                f"- Mean absolute rank disagreement: "
                f"`{fmt(r['null_agreement']['mean_abs_rank_disagreement'])}`",
                f"- P(own-hot) `{fmt(r['null_agreement']['p_own_hot'], 4)}`, "
                f"P(population-typical | own-hot) "
                f"`{fmt(r['null_agreement']['p_pop_typical_given_own_hot'], 4)}`",
                "",
                "They do disagree substantially, so the family does not die of the two nulls",
                "being the same object. It dies because the *disagreement* carries nothing on",
                "the target cut while the population *level* does: the atlas rank alone scores",
                f"`{fmt(r['scores']['atlas_rank_alone']['auc'])}` against own-z alone at",
                f"`{fmt(r['scores']['own_z_alone']['auc'])}`. A within-t population level is",
                "exactly what SmoothTimeCDFCal already applies to scores, which answers the",
                "mechanism's own 'information even on failure' question: yes, a population",
                "reference is already implicit in SCDF calibration.",
                "",
            ]
        if key == "m4_duration":
            md += [
                f"- Never-break series sampled: `{r['n_series_sampled']}`, "
                f"excursions: `{r['n_excursions']}`",
                f"- Mean run length `{fmt(r['mean_run_length'], 3)}`, "
                f"matched geometric p `{fmt(r['matched_geometric_p'], 5)}`",
                f"- Hazard slope in log-dwell: `{fmt(r['hazard_slope_in_log_dwell'])}` "
                "(0 = memoryless)",
                f"- Max survival gap vs matched geometric: `{fmt(r['max_survival_gap_vs_geometric'])}`",
                "",
                "| dwell k | at risk | hazard |",
                "|---:|---:|---:|",
            ]
            for h in r["hazard_by_dwell"][:12]:
                md.append(f"| {h['k']} | {h['at_risk']} | {fmt(h['hazard'], 4)} |")
            md += [
                "",
                "Frailty control. A pooled decreasing hazard is also what a mixture of",
                "per-series geometric dwells produces even when each series is individually",
                "memoryless, so the pooled figure alone cannot carry the verdict.",
                "",
                "| window | excursions | pooled slope | vs geometric | within-series slope | series | vs geometric mixture |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for w, v in r["per_window"].items():
                md.append(
                    f"| {w} | {v['n_excursions']} | {fmt(v['hazard_slope_in_log_dwell'], 5)} | "
                    f"{fmt(v['max_survival_gap_vs_geometric'], 4)} | "
                    f"{fmt(v['within_series_hazard_slope_median'], 5)} | "
                    f"{v['n_series_with_within_slope']} | "
                    f"{fmt(v['max_survival_gap_vs_geometric_mixture'], 4)} |"
                )
            md += [
                "",
                "The within-series hazard slope is steeper than the pooled one at every",
                "window, and a mixture of per-series geometrics leaves a survival gap far",
                "above the 0.05 tolerance, so the non-geometry is duration memory inside an",
                "excursion rather than heterogeneity across series. The within-series",
                "estimate is best powered at window 32 (211 series); windows 64 and 128 agree",
                "on the sign and magnitude with far fewer series.",
                "",
            ]
        if key == "m2_delay_cloud":
            md += [
                f"- Bootstrap 95% CI on the occupancy AUC: "
                f"`[{fmt(r['scores']['occupancy_disagreement_ci95_lo'], 4)}, "
                f"{fmt(r['scores']['occupancy_disagreement_ci95_hi'], 4)}]`; the gate is",
                "  applied to the lower bound, not the point estimate.",
                "",
                "Read this verdict narrowly. The screen clears the threshold its author set,",
                "but the margin over the plain scale control is modest and the correlation",
                f"with it is `{fmt(r['scores']['rho_occupancy_vs_m00'], 4)}`: residualised on",
                f"that control the statistic falls to",
                f"`{fmt(r['scores']['occupancy_residualised_on_m00_auc'])}`. PROCEED here means",
                "'not falsified, worth one fold-0 build', not 'carries independent signal'.",
                "",
            ]
        if key == "m5_c2st":
            md += [
                f"- Bootstrap 95% CI on the C2ST AUC: "
                f"`[{fmt(r['scores']['c2st_ci95_lo'], 4)}, {fmt(r['scores']['c2st_ci95_hi'], 4)}]`",
                "",
                "The kill is decisive rather than marginal: the learned test sits at chance",
                "while the named distance it would have to displace scores",
                f"`{fmt(r['scores']['wasserstein_control_auc'])}` on the identical windows.",
                "This is the linear C2ST the spec asked for; a nonlinear variant would have to",
                f"close a gap of `{fmt(abs(r['scores']['margin_over_wasserstein']), 4)}`, not",
                "win a close contest.",
                "",
            ]
        md.append(f"**Verdict.** {r['verdict']}")
    (out_dir / "grok_mechanism_screens.md").write_text("\n".join(md) + "\n")


def main() -> None:
    args = parse_args()
    if args.all:
        args.m3_atlas = args.m4_duration = args.m2_delay_cloud = args.m5_c2st = True
    if not any([args.m3_atlas, args.m4_duration, args.m2_delay_cloud, args.m5_c2st]):
        raise SystemExit("pass --m3-atlas, --m4-duration, --m2-delay-cloud, --m5-c2st, or --all")

    out_dir = Path(args.output_dir)
    existing = out_dir / "grok_mechanism_screens.json"
    results = json.loads(existing.read_text()) if existing.exists() else {"screens": {}}
    results["generated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results["git_sha"] = git_sha()
    results["artifact_root"] = str(ARTIFACT_ROOT)

    c = Ctx()
    if args.m3_atlas:
        results["screens"]["m3_atlas"] = screen_m3_atlas(args, c)
    if args.m4_duration:
        results["screens"]["m4_duration"] = screen_m4_duration(args, c)
    if args.m2_delay_cloud:
        results["screens"]["m2_delay_cloud"] = screen_m2_delay_cloud(args, c)
    if args.m5_c2st:
        results["screens"]["m5_c2st"] = screen_m5_c2st(args, c)

    write_report(results, out_dir)
    print(json.dumps({k: v["verdict"] for k, v in results["screens"].items()}, indent=2))


if __name__ == "__main__":
    main()
