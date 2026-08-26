#!/usr/bin/env python3
"""SS-04 specialist disagreement micro-router execution.

Implements the frozen protocol in
research/reports/new_avenues_2026/second_sweep/SS04_EXECUTION_PREREG.md.
No core sbr files are edited by this script.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(os.environ.get("SBR_ROOT", Path(__file__).resolve().parents[2]))
os.environ.setdefault("SBR_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(ROOT / "research" / "scripts" / "novel_streams"))
sys.path.insert(0, str(ROOT / "research" / "scripts" / "novel_streams" / "pilots"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from harness import rt600_blend  # noqa: E402
from pilot01_failure_manifolds import history_fingerprints, per_series_loss  # noqa: E402
from second_sweep_ss01 import (  # noqa: E402
    append_result_append_only,
    calibrated_for_outer,
    concat_rows,
    evaluate_stream,
    extended_pair_stats,
    finite_float,
    git_sha,
    logit01,
    pair_flow_pack,
    peak_rss,
    rt600_sentinel,
    sample_eval_pairs_for_label,
)
from sbr.metric import ts_auc_flat  # noqa: E402
from wave5_lib import Ctx, FOLDS, SPECIALISTS, load_oof  # noqa: E402

PREREG = ROOT / "research" / "reports" / "new_avenues_2026" / "second_sweep" / "SS04_EXECUTION_PREREG.md"
OUTDIR = ROOT / "research" / "reports" / "new_avenues_2026" / "second_sweep"
OOFDIR = ROOT / "research" / "oof"

EXP_ID = "RT-1230"
GLOBAL_ID = "RT-1231"
STATIC_ID = "RT-1232"
SHUFFLED_ID = "RT-1233"
CONTROL_IDS = (GLOBAL_ID, STATIC_ID, SHUFFLED_ID)
SEEDCLONE = "RT-401"
EXECUTION_PREREG_SHA = "25f40ac"

DOMINANT_T_MIN = 200
DOMINANT_AGE_MIN = 100
ACTION_ALPHA = 0.25
DELTA_CLIP = 0.35
DAMAGE_PENALTY = 2.0
DOMINANT_PAIR_WEIGHT = 3.0
TRAIN_PAIRS_PER_T = 128
EVAL_PAIRS_PER_T = 64
TRAIN_PAIR_SEED = 2026082510
GLOBAL_ROW_SEED = 2026082511
SHUFFLE_SEED = 2026082512
MAX_GLOBAL_TRAIN_ROWS = 400_000
ACTION_PROB_GAP = 0.05
MIN_ACTION_PROB = 0.20
LOGIT_C = 0.5
LOGIT_MAX_ITER = 200
STATIC_FINGERPRINT = "exc_max_run64"
STATIC_N_BINS = 5

SCREEN_GATE_MARGIN = 0.0010
CONTROL_GAP = 0.0005

FEATURE_SETS = {
    EXP_ID: "RT600_specialists,RT-401,row_score_state,specialist_deltas,t",
    GLOBAL_ID: "RT600_specialists",
    STATIC_ID: f"RT600_specialists,Pilot1::{STATIC_FINGERPRINT}",
    SHUFFLED_ID: "RT600_specialists,RT-401,row_score_state,specialist_deltas,t",
}
FEATURE_COUNTS = {
    EXP_ID: 35,
    GLOBAL_ID: 7,
    STATIC_ID: 8,
    SHUFFLED_ID: 35,
}


def load_required_scores(c: Ctx) -> tuple[dict[str, np.ndarray], dict]:
    names = list(SPECIALISTS) + [SEEDCLONE]
    scores = load_oof(names)
    dev = c.dev
    lockbox = c.d.rows_for([-1])
    checks = {}
    for name, x in scores.items():
        checks[name] = {
            "finite_dev_rows": int(np.isfinite(x[dev]).sum()),
            "expected_dev_rows": int(len(dev)),
            "finite_lockbox_rows": int(np.isfinite(x[lockbox]).sum()),
        }
        if checks[name]["finite_dev_rows"] != len(dev):
            raise SystemExit(f"{name} is not finite on all dev rows: {checks[name]}")
        if checks[name]["finite_lockbox_rows"] != 0:
            raise SystemExit(f"{name} unexpectedly fills lockbox rows: {checks[name]}")
    return scores, checks


def time_groups(rows: np.ndarray, t: np.ndarray):
    order = np.argsort(t[rows], kind="stable")
    sorted_rows = rows[order]
    sorted_t = t[sorted_rows]
    starts = np.flatnonzero(np.r_[True, sorted_t[1:] != sorted_t[:-1]])
    ends = np.r_[starts[1:], len(sorted_t)]
    for lo, hi in zip(starts, ends):
        yield int(sorted_t[lo]), sorted_rows[lo:hi]


def score_state(cal: dict[str, np.ndarray], rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spec = np.column_stack([cal[s][rows] for s in SPECIALISTS]).astype(np.float32, copy=False)
    base = spec.mean(axis=1).astype(np.float32, copy=False)
    seed = cal[SEEDCLONE][rows].astype(np.float32, copy=False)
    delta = np.clip(spec - base[:, None], -DELTA_CLIP, DELTA_CLIP).astype(np.float32, copy=False)
    if not np.isfinite(spec).all() or not np.isfinite(base).all() or not np.isfinite(seed).all():
        raise SystemExit("non-finite SS-04 score state")
    return spec, base, seed, delta


def router_features(c: Ctx, cal: dict[str, np.ndarray], rows: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    spec, base, seed, delta = score_state(cal, rows)
    t = c.d.t[rows].astype(np.float32, copy=False)
    spec_sorted = np.sort(spec, axis=1)
    cols: list[np.ndarray] = []
    names: list[str] = []

    def add(name: str, arr: np.ndarray) -> None:
        names.append(name)
        cols.append(np.asarray(arr, dtype=np.float32))

    add("rt600_cal", base)
    add("rt600_margin_abs", np.abs(base - 0.5))
    add("rt600_logit", logit01(base))
    add("seed_minus_rt600", seed - base)
    add("abs_seed_minus_rt600", np.abs(seed - base))
    add("specialist_std", spec.std(axis=1))
    add("specialist_range", spec.max(axis=1) - spec.min(axis=1))
    add("specialist_min", spec.min(axis=1))
    add("specialist_max", spec.max(axis=1))
    add("specialists_above_rt600_frac", (spec > base[:, None]).mean(axis=1))
    add("specialists_above_half_frac", (spec > 0.5).mean(axis=1))
    add("top_minus_second", spec_sorted[:, -1] - spec_sorted[:, -2])
    add("second_minus_bottom", spec_sorted[:, 1] - spec_sorted[:, 0])
    add("max_positive_delta", np.maximum(delta.max(axis=1), 0.0))
    add("max_negative_delta_abs", np.maximum(-delta.min(axis=1), 0.0))
    add("log1p_t", np.log1p(t))
    for cut in (50, 100, 200, 400, 800):
        add(f"t_ge_{cut}", (t >= cut).astype(np.float32))
    for sid, values in zip(SPECIALISTS, delta.T):
        lab = sid.replace("-", "")
        add(f"{lab}_delta", values)
        add(f"{lab}_abs_delta", np.abs(values))
    X = np.column_stack(cols).astype(np.float32, copy=False)
    if X.shape[1] != FEATURE_COUNTS[EXP_ID]:
        raise SystemExit(f"SS-04 feature-count mismatch: {X.shape[1]} vs {FEATURE_COUNTS[EXP_ID]}")
    if not np.isfinite(X).all():
        raise SystemExit("non-finite SS-04 router features")
    return X, names, base, delta


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0) * 1.4826
    sd = np.nanstd(x, axis=0)
    scale = np.where(mad > 1e-12, mad, sd)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return med.astype(np.float32), scale.astype(np.float32)


def apply_standardizer(x: np.ndarray, med: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (x - med) / scale
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def sample_training_pairs(c: Ctx, train_rows: np.ndarray, outer_fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pp_all: list[np.ndarray] = []
    nn_all: list[np.ndarray] = []
    ww_all: list[np.ndarray] = []
    for tt, idx in time_groups(train_rows, c.d.t):
        if tt < DOMINANT_T_MIN:
            continue
        pos = idx[(c.d.y[idx] == 1) & (c.age[idx] >= DOMINANT_AGE_MIN)]
        neg = idx[c.d.y[idx] == 0]
        k = min(TRAIN_PAIRS_PER_T, len(pos), len(neg))
        if k <= 0:
            continue
        rng = np.random.default_rng(TRAIN_PAIR_SEED + 1000 * int(outer_fold) + int(tt))
        pp = rng.choice(pos, k, replace=False)
        nn = rng.choice(neg, k, replace=False)
        pp_all.append(pp.astype(np.int64, copy=False))
        nn_all.append(nn.astype(np.int64, copy=False))
        ww_all.append(np.full(k, DOMINANT_PAIR_WEIGHT, dtype=np.float32))
    if not pp_all:
        raise SystemExit(f"outer fold {outer_fold}: no SS-04 training pairs sampled")
    return np.concatenate(pp_all), np.concatenate(nn_all), np.concatenate(ww_all)


def build_action_targets(c: Ctx, cal: dict[str, np.ndarray], train_rows: np.ndarray, outer_fold: int) -> dict:
    row_to_local = np.full(len(c.d.y), -1, dtype=np.int32)
    row_to_local[train_rows] = np.arange(len(train_rows), dtype=np.int32)
    pp, nn, pair_weight = sample_training_pairs(c, train_rows, outer_fold)
    spec_p, base_p, _, delta_p = score_state(cal, pp)
    spec_n, base_n, _, delta_n = score_state(cal, nn)
    base_right = base_p > base_n
    base_wrong = ~base_right

    util = np.zeros((len(train_rows), len(SPECIALISTS)), dtype=np.float32)
    exposure = np.zeros(len(train_rows), dtype=np.float32)
    pos_local = row_to_local[pp]
    neg_local = row_to_local[nn]
    if (pos_local < 0).any() or (neg_local < 0).any():
        raise SystemExit("SS-04 pair sampler selected row outside train matrix")

    pos_contrib = np.zeros((len(pp), len(SPECIALISTS)), dtype=np.float32)
    neg_contrib = np.zeros((len(pp), len(SPECIALISTS)), dtype=np.float32)
    for j in range(len(SPECIALISTS)):
        score_p = base_p + ACTION_ALPHA * delta_p[:, j]
        score_n = base_n + ACTION_ALPHA * delta_n[:, j]
        pos_contrib[base_wrong, j] = (score_p[base_wrong] > base_n[base_wrong]).astype(np.float32)
        neg_contrib[base_wrong, j] = (base_p[base_wrong] > score_n[base_wrong]).astype(np.float32)
        pos_contrib[base_right, j] = -DAMAGE_PENALTY * (score_p[base_right] <= base_n[base_right]).astype(np.float32)
        neg_contrib[base_right, j] = -DAMAGE_PENALTY * (base_p[base_right] <= score_n[base_right]).astype(np.float32)
    pos_contrib *= pair_weight[:, None]
    neg_contrib *= pair_weight[:, None]
    np.add.at(util, pos_local, pos_contrib)
    np.add.at(util, neg_local, neg_contrib)
    np.add.at(exposure, pos_local, pair_weight)
    np.add.at(exposure, neg_local, pair_weight)

    exposed = exposure > 0
    util_norm = util[exposed] / exposure[exposed, None]
    max_util = util_norm.max(axis=1)
    labels = np.where(max_util > 0.0, util_norm.argmax(axis=1).astype(np.int16) + 1, 0).astype(np.int16)
    rows = train_rows[exposed]
    weights = exposure[exposed].astype(np.float32)
    weights = weights / max(float(weights.mean()), 1e-12)
    spec_correct_count = (spec_p > spec_n).sum(axis=1)
    counts = np.bincount(labels, minlength=len(SPECIALISTS) + 1)
    return {
        "rows": rows,
        "labels": labels,
        "weights": weights,
        "target_counts": {str(i): int(v) for i, v in enumerate(counts)},
        "target_specialist_counts": {SPECIALISTS[i - 1]: int(counts[i]) for i in range(1, len(SPECIALISTS) + 1)},
        "n_pairs": int(len(pp)),
        "rt600_wrong_pairs": int(base_wrong.sum()),
        "rt600_right_pairs": int(base_right.sum()),
        "specialist_correct_count_histogram": {str(i): int((spec_correct_count == i).sum()) for i in range(8)},
        "n_exposed_rows": int(len(rows)),
        "n_rt600_only_rows": int(counts[0]),
    }


def fit_logistic_multiclass(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    med, scale = fit_standardizer(x)
    xz = apply_standardizer(x, med, scale)
    if len(np.unique(y)) < 2:
        return None, med, scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model = LogisticRegression(
            C=LOGIT_C,
            solver="lbfgs",
            max_iter=LOGIT_MAX_ITER,
            fit_intercept=True,
            multi_class="multinomial",
        )
        model.fit(xz, y, sample_weight=w)
    return model, med, scale


def predict_action_scores(
    model,
    med: np.ndarray,
    scale: np.ndarray,
    feature_names: list[str],
    x_val: np.ndarray,
    base: np.ndarray,
    delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    n = len(base)
    if model is None:
        actions = np.zeros(n, dtype=np.int16)
        proba_full = np.zeros((n, len(SPECIALISTS) + 1), dtype=np.float32)
        proba_full[:, 0] = 1.0
    else:
        z = apply_standardizer(x_val, med, scale)
        p = model.predict_proba(z).astype(np.float32)
        proba_full = np.zeros((n, len(SPECIALISTS) + 1), dtype=np.float32)
        for j, cls in enumerate(model.classes_):
            proba_full[:, int(cls)] = p[:, j]
        nonzero = proba_full[:, 1:]
        best = np.argmax(nonzero, axis=1).astype(np.int16) + 1
        best_p = nonzero[np.arange(n), best - 1]
        keep_p = proba_full[:, 0]
        take = (best_p >= MIN_ACTION_PROB) & ((best_p - keep_p) >= ACTION_PROB_GAP)
        actions = np.where(take, best, 0).astype(np.int16)
    score = base.astype(np.float32, copy=True)
    for j in range(len(SPECIALISTS)):
        m = actions == (j + 1)
        if np.any(m):
            score[m] = base[m] + ACTION_ALPHA * delta[m, j]
    counts = np.bincount(actions, minlength=len(SPECIALISTS) + 1)
    return score, actions, {
        "action_counts": {str(i): int(v) for i, v in enumerate(counts)},
        "action_specialist_counts": {SPECIALISTS[i - 1]: int(counts[i]) for i in range(1, len(SPECIALISTS) + 1)},
        "n_non_rt600_actions": int(counts[1:].sum()),
        "feature_audit": coefficient_audit(model, feature_names),
    }


def coefficient_audit(model, feature_names: list[str]) -> dict:
    if model is None:
        return {"classes": [0], "top_coefficients": [], "family_l2": {}}
    coefs = np.asarray(model.coef_, dtype=np.float64)
    top = []
    for ci, cls in enumerate(model.classes_):
        order = np.argsort(np.abs(coefs[ci]))[::-1][:8]
        top.append(
            {
                "class": int(cls),
                "top": [
                    {"feature": feature_names[int(j)], "coef": float(coefs[ci, int(j)])}
                    for j in order
                ],
            }
        )
    groups = {
        "rt600_seed_state": [i for i, n in enumerate(feature_names) if "rt600" in n or "seed" in n],
        "dispersion_state": [i for i, n in enumerate(feature_names) if "specialist_" in n or "top_" in n or "second_" in n or "max_" in n],
        "time_state": [i for i, n in enumerate(feature_names) if n.startswith("t_") or n == "log1p_t"],
        "specialist_delta": [i for i, n in enumerate(feature_names) if n.endswith("_delta") or n.endswith("_abs_delta")],
    }
    family_l2 = {
        name: float(np.sqrt((coefs[:, idx] ** 2).sum())) if idx else 0.0
        for name, idx in groups.items()
    }
    return {"classes": [int(x) for x in model.classes_], "top_coefficients": top, "family_l2": family_l2}


def fit_router_fold(c: Ctx, scores: dict[str, np.ndarray], outer_fold: int, shuffled: bool) -> dict:
    names = list(SPECIALISTS) + [SEEDCLONE]
    cal = calibrated_for_outer(c, scores, int(outer_fold), names)
    train_rows = concat_rows(c, [g for g in FOLDS if int(g) != int(outer_fold)])
    val_rows = c.rows[int(outer_fold)]
    target = build_action_targets(c, cal, train_rows, int(outer_fold))
    labels = target["labels"].astype(np.int16, copy=True)
    if shuffled:
        rng = np.random.default_rng(SHUFFLE_SEED + int(outer_fold))
        labels = labels[rng.permutation(len(labels))]
    xtr, feature_names, _, _ = router_features(c, cal, target["rows"])
    model, med, scale = fit_logistic_multiclass(xtr, labels, target["weights"])
    xva, _, base_va, delta_va = router_features(c, cal, val_rows)
    score, actions, audit = predict_action_scores(model, med, scale, feature_names, xva, base_va, delta_va)
    return {
        "fold": int(outer_fold),
        "val_rows": val_rows,
        "score": score,
        "actions": actions,
        "target": target,
        "shuffled": bool(shuffled),
        "classes": [int(x) for x in (model.classes_ if model is not None else [0])],
        **audit,
    }


def train_router_oof(c: Ctx, scores: dict[str, np.ndarray], shuffled: bool, label: str) -> dict:
    t0 = time.time()
    oof = np.full(len(c.d.y), np.nan, dtype=np.float32)
    actions = np.full(len(c.d.y), -1, dtype=np.int16)
    fold_summaries = []
    for f in FOLDS:
        fold_start = time.time()
        res = fit_router_fold(c, scores, int(f), shuffled=shuffled)
        oof[res["val_rows"]] = res["score"]
        actions[res["val_rows"]] = res["actions"]
        fold_summaries.append(
            {
                "fold": int(f),
                "train_rows": int(len(concat_rows(c, [g for g in FOLDS if int(g) != int(f)]))),
                "valid_rows": int(len(res["val_rows"])),
                "target": {
                    "n_pairs": res["target"]["n_pairs"],
                    "rt600_wrong_pairs": res["target"]["rt600_wrong_pairs"],
                    "rt600_right_pairs": res["target"]["rt600_right_pairs"],
                    "specialist_correct_count_histogram": res["target"]["specialist_correct_count_histogram"],
                    "n_exposed_rows": res["target"]["n_exposed_rows"],
                    "n_rt600_only_rows": res["target"]["n_rt600_only_rows"],
                    "target_counts": res["target"]["target_counts"],
                    "target_specialist_counts": res["target"]["target_specialist_counts"],
                },
                "action_counts": res["action_counts"],
                "action_specialist_counts": res["action_specialist_counts"],
                "n_non_rt600_actions": res["n_non_rt600_actions"],
                "classes": res["classes"],
                "feature_audit": res["feature_audit"],
                "runtime_s": float(time.time() - fold_start),
            }
        )
        print(f"{label}: fold {f} non-RT600 actions {res['n_non_rt600_actions']} ({time.time() - fold_start:.1f}s)", flush=True)
    if np.isfinite(oof[c.d.rows_for([-1])]).any():
        raise SystemExit(f"{label}: produced lockbox predictions")
    return {"oof": oof, "actions": actions, "fold_summaries": fold_summaries, "runtime_s": float(time.time() - t0)}


def fit_binary_logit(xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    med, scale = fit_standardizer(xtr)
    xtrz = apply_standardizer(xtr, med, scale)
    xvaz = apply_standardizer(xva, med, scale)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=LOGIT_MAX_ITER, fit_intercept=True, random_state=seed)
        model.fit(xtrz, ytr)
    pred = model.predict_proba(xvaz)[:, 1].astype(np.float32)
    return pred, {
        "intercept": float(model.intercept_[0]),
        "coefficients": {SPECIALISTS[i]: float(model.coef_[0, i]) for i in range(len(SPECIALISTS))},
    }


def train_global_reweight_oof(c: Ctx, scores: dict[str, np.ndarray]) -> dict:
    t0 = time.time()
    oof = np.full(len(c.d.y), np.nan, dtype=np.float32)
    summaries = []
    for f in FOLDS:
        fold_start = time.time()
        cal = calibrated_for_outer(c, scores, int(f), list(SPECIALISTS))
        train_rows = concat_rows(c, [g for g in FOLDS if int(g) != int(f)])
        val_rows = c.rows[int(f)]
        if len(train_rows) > MAX_GLOBAL_TRAIN_ROWS:
            rng = np.random.default_rng(GLOBAL_ROW_SEED + int(f))
            fit_rows = np.sort(rng.choice(train_rows, MAX_GLOBAL_TRAIN_ROWS, replace=False))
        else:
            fit_rows = train_rows
        xtr = np.column_stack([cal[s][fit_rows] for s in SPECIALISTS]).astype(np.float32)
        xva = np.column_stack([cal[s][val_rows] for s in SPECIALISTS]).astype(np.float32)
        pred, coef = fit_binary_logit(xtr, c.d.y[fit_rows], xva, GLOBAL_ROW_SEED + int(f))
        oof[val_rows] = pred
        summaries.append(
            {
                "fold": int(f),
                "fit_rows": int(len(fit_rows)),
                "valid_rows": int(len(val_rows)),
                "row_seed": GLOBAL_ROW_SEED + int(f),
                "coef": coef,
                "runtime_s": float(time.time() - fold_start),
            }
        )
        print(f"{GLOBAL_ID}: fold {f} global logistic control ({time.time() - fold_start:.1f}s)", flush=True)
    if np.isfinite(oof[c.d.rows_for([-1])]).any():
        raise SystemExit(f"{GLOBAL_ID}: produced lockbox predictions")
    return {"oof": oof, "fold_summaries": summaries, "runtime_s": float(time.time() - t0)}


def static_bins(values: np.ndarray, train_series: np.ndarray) -> tuple[np.ndarray, list[float]]:
    finite = train_series & np.isfinite(values)
    qs = np.quantile(values[finite], np.linspace(0, 1, STATIC_N_BINS + 1))
    edges = np.unique(qs[1:-1])
    bins = np.full(len(values), -1, dtype=np.int16)
    ok = np.isfinite(values)
    bins[ok] = np.digitize(values[ok], edges, right=True).astype(np.int16)
    return bins, [float(x) for x in edges]


def train_static_selector_oof(c: Ctx, scores: dict[str, np.ndarray]) -> dict:
    t0 = time.time()
    fp, fp_names = history_fingerprints()
    if STATIC_FINGERPRINT not in fp_names:
        raise SystemExit(f"missing Pilot-1 fingerprint {STATIC_FINGERPRINT}")
    fp_values = fp[:, fp_names.index(STATIC_FINGERPRINT)]
    folds_series = pd.read_parquet(ROOT / "research" / "folds" / "folds.parquet").fold.to_numpy()
    oof = np.full(len(c.d.y), np.nan, dtype=np.float32)
    summaries = []
    for f in FOLDS:
        fold_start = time.time()
        names = list(SPECIALISTS)
        cal = calibrated_for_outer(c, scores, int(f), names)
        train_rows = concat_rows(c, [g for g in FOLDS if int(g) != int(f)])
        val_rows = c.rows[int(f)]
        train_series = np.isin(folds_series, [g for g in FOLDS if int(g) != int(f)])
        bins, edges = static_bins(fp_values, train_series)
        losses = {}
        weights = None
        for sid in SPECIALISTS:
            sl, sw = per_series_loss(cal[sid], c, train_rows)
            losses[sid] = sl
            if weights is None:
                weights = sw
        assert weights is not None
        choices = {}
        base_va = np.column_stack([cal[s][val_rows] for s in SPECIALISTS]).mean(axis=1)
        oof[val_rows] = base_va.astype(np.float32)
        for b in range(STATIC_N_BINS):
            m_series = train_series & (bins == b) & (weights > 0)
            if float(weights[m_series].sum()) <= 0:
                choices[str(b)] = "RT600"
                continue
            rates = {sid: float(losses[sid][m_series].sum() / max(float(weights[m_series].sum()), 1.0)) for sid in SPECIALISTS}
            pick = min(rates, key=rates.get)
            choices[str(b)] = pick
            val_mask = bins[c.d.sidx[val_rows]] == b
            oof[val_rows[val_mask]] = cal[pick][val_rows[val_mask]].astype(np.float32)
        summaries.append(
            {
                "fold": int(f),
                "fingerprint": STATIC_FINGERPRINT,
                "edges": edges,
                "choices": choices,
                "valid_rows": int(len(val_rows)),
                "runtime_s": float(time.time() - fold_start),
            }
        )
        print(f"{STATIC_ID}: fold {f} static selector choices {choices} ({time.time() - fold_start:.1f}s)", flush=True)
    if np.isfinite(oof[c.d.rows_for([-1])]).any():
        raise SystemExit(f"{STATIC_ID}: produced lockbox predictions")
    return {"oof": oof, "fold_summaries": summaries, "runtime_s": float(time.time() - t0)}


def write_oof(exp_id: str, score: np.ndarray) -> str:
    OOFDIR.mkdir(parents=True, exist_ok=True)
    path = OOFDIR / f"{exp_id}.npy"
    np.save(path, score.astype(np.float32, copy=False))
    return str(path)


def pair_stats_arrays(base: np.ndarray, cand: np.ndarray, pp: np.ndarray, nn: np.ndarray) -> dict:
    base_right = base[pp] > base[nn]
    cand_right = cand[pp] > cand[nn]
    rt600_wrong = ~base_right
    repairs = int((rt600_wrong & cand_right).sum())
    damage = int((base_right & ~cand_right).sum())
    wrong_n = int(rt600_wrong.sum())
    right_n = int(base_right.sum())
    return {
        "total_pairs_sampled": int(len(pp)),
        "rt600_wrong": wrong_n,
        "rt600_right": right_n,
        "repairs": repairs,
        "damage": damage,
        "net_pair_lift": repairs - damage,
        "repair_rate_of_rt600_wrong": float(repairs / wrong_n) if wrong_n else None,
        "damage_rate_of_rt600_right": float(damage / right_n) if right_n else None,
    }


def specialist_disagreement_pair_flow(c: Ctx, scores: dict[str, np.ndarray], base: np.ndarray, cand: np.ndarray) -> dict:
    cal = calibrated_for_outer(c, scores, 0, list(SPECIALISTS))
    pp, nn = sample_eval_pairs_for_label(c, "dominant_cell", fold=0)
    spec_p = np.column_stack([cal[s][pp] for s in SPECIALISTS])
    spec_n = np.column_stack([cal[s][nn] for s in SPECIALISTS])
    correct_count = (spec_p > spec_n).sum(axis=1)
    out = {
        "all_dominant": pair_stats_arrays(base, cand, pp, nn),
        "specialist_correct_count_histogram": {str(i): int((correct_count == i).sum()) for i in range(8)},
    }
    for name, mask in (
        ("majority_correct", correct_count >= 4),
        ("near_split", (correct_count == 3) | (correct_count == 4)),
        ("at_least_one_correct", correct_count >= 1),
    ):
        out[name] = pair_stats_arrays(base, cand, pp[mask], nn[mask])
    return out


def control_independent_pass(ev: dict, subgroup: dict) -> bool:
    margin = ev["ensemble_marginal"]["marginal_vs_clone"]
    dom = ev["pair_flow"]["dominant_cell"]["net_pair_lift"]
    maj = subgroup["majority_correct"]["net_pair_lift"]
    near = subgroup["near_split"]["net_pair_lift"]
    return bool(margin >= SCREEN_GATE_MARGIN and dom >= 0 and maj > 0 and near > 0)


def gate_verdict(candidate: dict, controls: dict, subgroups: dict, control_subgroups: dict, audits_ok: bool) -> tuple[str, list[dict]]:
    failures = []
    margin = candidate["ensemble_marginal"]["marginal_vs_clone"]
    dom = candidate["pair_flow"]["dominant_cell"]["net_pair_lift"]
    maj = subgroups[EXP_ID]["majority_correct"]["net_pair_lift"]
    near = subgroups[EXP_ID]["near_split"]["net_pair_lift"]
    if margin < SCREEN_GATE_MARGIN:
        failures.append({"gate": "marginal_vs_clone", "threshold": SCREEN_GATE_MARGIN, "observed": margin})
    if dom < 0:
        failures.append({"gate": "dominant_cell_net_pair_lift", "threshold": ">=0", "observed": dom})
    if maj <= 0:
        failures.append({"gate": "majority_correct_pair_net", "threshold": ">0", "observed": maj})
    if near <= 0:
        failures.append({"gate": "near_split_pair_net", "threshold": ">0", "observed": near})
    for eid in CONTROL_IDS:
        cm = controls[eid]["ensemble_marginal"]["marginal_vs_clone"]
        if margin - cm < CONTROL_GAP:
            failures.append(
                {
                    "gate": f"{eid}_control_gap",
                    "threshold": CONTROL_GAP,
                    "observed_candidate_minus_control": margin - cm,
                    "control_marginal_vs_clone": cm,
                }
            )
    if control_independent_pass(controls[SHUFFLED_ID], control_subgroups[SHUFFLED_ID]):
        failures.append({"gate": f"{SHUFFLED_ID}_independent_screen", "threshold": "shuffled target must not pass gates 1-4", "observed": True})
    if not audits_ok:
        failures.append({"gate": "causal_fold_purity_lockbox_audit", "threshold": True, "observed": False})
    return ("KILL" if failures else "PASS_SCREEN"), failures


def markdown_pair_table(ev: dict) -> list[str]:
    rows = [
        "| split | pairs | RT600 wrong | RT600 right | repairs | damage | net | damage rate on RT600-right |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in ev["pair_flow"].items():
        dr = row["damage_rate_of_rt600_right"]
        dr_text = f"{dr:.4f}" if dr is not None else "n/a"
        rows.append(
            f"| `{name}` | {row['total_pairs_sampled']} | {row['rt600_wrong']} | {row['rt600_right']} | "
            f"{row['repairs']} | {row['damage']} | {row['net_pair_lift']} | {dr_text} |"
        )
    return rows


def write_json(result: dict) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with (OUTDIR / "ss04_specialist_router.json").open("w") as f:
        json.dump(finite_float(result), f, indent=2, sort_keys=True)
        f.write("\n")


def write_markdown(result: dict) -> None:
    cand = result[EXP_ID]
    controls = result["controls"]
    cm = cand["ensemble_marginal"]
    sub = result["specialist_disagreement_pair_flow"][EXP_ID]
    md = [
        "# SS-04 -- Specialist Disagreement Micro-Router",
        "",
        f"Execution preregistration: `research/reports/new_avenues_2026/second_sweep/SS04_EXECUTION_PREREG.md` at `{result['execution_prereg_sha']}`.",
        f"Experiment IDs: `{EXP_ID}` candidate, `{GLOBAL_ID}` global reweighting control, `{STATIC_ID}` Pilot-1 static selector control, `{SHUFFLED_ID}` shuffled target control.",
        "",
        "## RT-600 Sentinel",
        "",
        f"* Dev mean TS-AUC: `{result['rt600_sentinel']['mean']:.6f}`.",
        f"* Dev pooled TS-AUC: `{result['rt600_sentinel']['pooled']:.6f}`.",
        f"* Dev dominant-cell TS-AUC: `{result['rt600_sentinel']['dominant_cell']:.6f}`.",
        f"* Fold-0 E0 RT600: `{result['rt600_sentinel']['fold0_e0']:.6f}`.",
        f"* Fold-0 E1 RT600 + RT-401: `{result['rt600_sentinel']['fold0_e1_seedclone']:.6f}`.",
        "",
        "## Binding Result",
        "",
        "| arm | standalone fold-0 TS-AUC | E2 TS-AUC | marginal vs clone | gain vs RT600 | dominant net | majority net | near-split net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| RT600 |  | {cm['rt600_7stream']:.6f} |  |  |  |  |  |",
        f"| RT600 + RT-401 seed clone |  | {cm['rt600_plus_seedclone']:.6f} |  |  |  |  |  |",
        (
            f"| RT600 + {EXP_ID} | {cand['fold0_standalone_ts_auc']:.6f} | "
            f"{cm[f'rt600_plus_{EXP_ID}']:.6f} | {cm['marginal_vs_clone']:+.6f} | "
            f"{cm['gain_vs_base']:+.6f} | {cand['pair_flow']['dominant_cell']['net_pair_lift']} | "
            f"{sub['majority_correct']['net_pair_lift']} | {sub['near_split']['net_pair_lift']} |"
        ),
    ]
    for eid, ctrl in controls.items():
        em = ctrl["ensemble_marginal"]
        sg = result["specialist_disagreement_pair_flow"][eid]
        md.append(
            f"| RT600 + {eid} | {ctrl['fold0_standalone_ts_auc']:.6f} | "
            f"{em[f'rt600_plus_{eid}']:.6f} | {em['marginal_vs_clone']:+.6f} | "
            f"{em['gain_vs_base']:+.6f} | {ctrl['pair_flow']['dominant_cell']['net_pair_lift']} | "
            f"{sg['majority_correct']['net_pair_lift']} | {sg['near_split']['net_pair_lift']} |"
        )
    md += [
        "",
        f"Verdict: **{result['verdict']}**.",
        "Gate failures: "
        + ("; ".join(f"`{x['gate']}` observed `{x.get('observed', x.get('observed_candidate_minus_control'))}`" for x in result["gate_failures"]) if result["gate_failures"] else "none"),
        "",
        "## Candidate Pair Flow",
        "",
        *markdown_pair_table(cand),
        "",
        "## Specialist-Disagreement Pair Flow",
        "",
        "| subset | pairs | RT600 wrong | RT600 right | repairs | damage | net |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("majority_correct", "near_split", "at_least_one_correct"):
        row = sub[name]
        md.append(
            f"| `{name}` | {row['total_pairs_sampled']} | {row['rt600_wrong']} | {row['rt600_right']} | "
            f"{row['repairs']} | {row['damage']} | {row['net_pair_lift']} |"
        )
    md += [
        "",
        "## Action And Coefficient Audits",
        "",
        f"* Fold-0 candidate action counts: `{cand['fit']['fold_summaries'][0]['action_counts']}`.",
        f"* Fold-0 candidate selected specialists: `{cand['fit']['fold_summaries'][0]['action_specialist_counts']}`.",
        f"* Fold-0 target counts: `{cand['fit']['fold_summaries'][0]['target']['target_counts']}`.",
        f"* Fold-0 coefficient family L2: `{cand['fit']['fold_summaries'][0]['feature_audit']['family_l2']}`.",
        "",
        "## Controls",
        "",
        "| control | marginal vs clone | candidate minus control | dominant net |",
        "|---|---:|---:|---:|",
    ]
    cand_margin = cm["marginal_vs_clone"]
    for eid, ctrl in controls.items():
        em = ctrl["ensemble_marginal"]["marginal_vs_clone"]
        md.append(f"| `{eid}` | {em:+.6f} | {cand_margin - em:+.6f} | {ctrl['pair_flow']['dominant_cell']['net_pair_lift']} |")
    md += [
        "",
        "## Implementation Audits",
        "",
        "* Score coverage: all required frozen specialist and seed-clone OOF scores finite on dev rows and zero finite on lockbox rows.",
        f"* Pair-sample reproduction: `{result['pair_sample_reproduction']}`.",
        f"* New OOF lockbox finite counts: `{result['new_oof_lockbox_finite_counts']}`.",
        f"* Peak RSS: `{result['peak_rss']}`.",
        f"* Runtime: `{result['runtime_s']:.1f}s`.",
        "",
        "## Interpretation",
        "",
        result["interpretation"],
    ]
    (OUTDIR / "ss04_specialist_router.md").write_text("\n".join(md) + "\n")


def result_row(exp_id: str, status: str, notes: str, runtime_s: float, train_rows: int) -> dict:
    return {
        "experiment_id": exp_id,
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "git_sha": git_sha(),
        "agent": "codex-second-sweep",
        "hypothesis": "SS-04 tests whether bounded row-level specialist disagreement routing repairs RT600 residual pairs without global specialist weighting.",
        "falsification_condition": "KILL if any SS-04 mandatory fold-0 screen gate fails; global/static/shuffled controls must lose.",
        "feature_set": FEATURE_SETS[exp_id],
        "n_features": FEATURE_COUNTS[exp_id],
        "model": "multinomial_logistic_router" if exp_id in (EXP_ID, SHUFFLED_ID) else ("binary_logistic_global_stack" if exp_id == GLOBAL_ID else "static_fingerprint_selector"),
        "objective": "row_local_repair_damage_action" if exp_id in (EXP_ID, SHUFFLED_ID) else "control",
        "folds": "0",
        "random_seed": SHUFFLE_SEED if exp_id == SHUFFLED_ID else 20260825,
        "train_series": 8000,
        "train_rows": train_rows,
        "mean_oof_ts_auc": "",
        "pooled_oof_ts_auc": "",
        "per_fold_ts_auc": "",
        "fold_std": 0.0,
        "persistence": "none",
        "sample_mode": f"{TRAIN_PAIRS_PER_T}_train_pairs_per_t_{EVAL_PAIRS_PER_T}_eval_pairs_per_t",
        "training_runtime_s": round(runtime_s, 1),
        "causal_verified": "frozen specialist OOF score state + outer/inner fold-pure SCDF calibration + no killed first-sweep candidate scores + zero lockbox fill",
        "test_reduced_touched": "no",
        "lockbox_touched": "no",
        "status": status,
        "notes": notes,
        "protocol": "second_sweep_ss04_fold0_screen",
    }


def append_rows(c: Ctx, rows: list[dict], scores: dict[str, np.ndarray]) -> None:
    existing = set()
    with (ROOT / "research" / "RESULTS.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            existing.add((row.get("experiment_id") or "").strip())
    for row in rows:
        eid = row["experiment_id"]
        if eid in existing:
            raise SystemExit(f"RESULTS.csv already contains {eid}")
        sc = scores[eid]
        r = c.rows[0]
        auc = float(ts_auc_flat(sc[r], c.d.y[r], c.d.t[r]))
        row["mean_oof_ts_auc"] = auc
        row["pooled_oof_ts_auc"] = auc
        row["per_fold_ts_auc"] = f"{auc:.5f}"
        append_result_append_only(row)
        existing.add(eid)


def preflight() -> None:
    if not PREREG.exists():
        raise SystemExit(f"missing execution preregistration: {PREREG}")
    c = Ctx()
    scores, checks = load_required_scores(c)
    base = rt600_blend(c)
    sentinel = rt600_sentinel(c, base)
    cal = calibrated_for_outer(c, scores, 0, list(SPECIALISTS) + [SEEDCLONE])
    train_rows = concat_rows(c, [g for g in FOLDS if int(g) != 0])
    target = build_action_targets(c, cal, train_rows, 0)
    _, fp_names = history_fingerprints()
    dom_self = extended_pair_stats(c, base, base, "dominant_cell", fold=0)
    out = {
        "git_sha": git_sha(),
        "branch": "research/new-avenues-pilots-2026",
        "prereg_exists": PREREG.exists(),
        "rt600_sentinel": sentinel,
        "score_coverage": checks,
        "pilot1_static_fingerprint_available": STATIC_FINGERPRINT in fp_names,
        "fold0_target": {
            "n_pairs": target["n_pairs"],
            "rt600_wrong_pairs": target["rt600_wrong_pairs"],
            "rt600_right_pairs": target["rt600_right_pairs"],
            "specialist_correct_count_histogram": target["specialist_correct_count_histogram"],
            "n_exposed_rows": target["n_exposed_rows"],
            "target_counts": target["target_counts"],
            "target_specialist_counts": target["target_specialist_counts"],
        },
        "dominant_pair_sample_self": dom_self,
    }
    print(json.dumps(finite_float(out), indent=2, sort_keys=True))


def run_score() -> None:
    t0 = time.time()
    if not PREREG.exists():
        raise SystemExit(f"missing execution preregistration: {PREREG}")
    c = Ctx()
    scores, checks = load_required_scores(c)
    base = rt600_blend(c)
    sentinel = rt600_sentinel(c, base)

    candidate_fit = train_router_oof(c, scores, shuffled=False, label=EXP_ID)
    write_oof(EXP_ID, candidate_fit["oof"])
    global_fit = train_global_reweight_oof(c, scores)
    write_oof(GLOBAL_ID, global_fit["oof"])
    static_fit = train_static_selector_oof(c, scores)
    write_oof(STATIC_ID, static_fit["oof"])
    shuffled_fit = train_router_oof(c, scores, shuffled=True, label=SHUFFLED_ID)
    write_oof(SHUFFLED_ID, shuffled_fit["oof"])

    oofs = {
        EXP_ID: candidate_fit["oof"],
        GLOBAL_ID: global_fit["oof"],
        STATIC_ID: static_fit["oof"],
        SHUFFLED_ID: shuffled_fit["oof"],
    }
    fits = {
        EXP_ID: candidate_fit,
        GLOBAL_ID: global_fit,
        STATIC_ID: static_fit,
        SHUFFLED_ID: shuffled_fit,
    }
    evals = {eid: evaluate_stream(c, base, score, eid, fits[eid]["runtime_s"]) for eid, score in oofs.items()}
    for eid in evals:
        evals[eid]["fit"] = {
            "runtime_s": fits[eid]["runtime_s"],
            "fold_summaries": fits[eid]["fold_summaries"],
            "oof_artifact": str(OOFDIR / f"{eid}.npy"),
        }
    controls = {eid: evals[eid] for eid in CONTROL_IDS}
    subgroup = {eid: specialist_disagreement_pair_flow(c, scores, base, score) for eid, score in oofs.items()}
    lock_counts = {eid: int(np.isfinite(score[c.d.rows_for([-1])]).sum()) for eid, score in oofs.items()}
    audits_ok = bool(all(v == 0 for v in lock_counts.values()))
    verdict, failures = gate_verdict(evals[EXP_ID], controls, subgroup, {eid: subgroup[eid] for eid in CONTROL_IDS}, audits_ok)

    if verdict == "KILL":
        interpretation = (
            "SS-04 is KILL under the preregistered fold-0 screen. The bounded "
            "row-level specialist action router did not satisfy all mandatory "
            "marginal, dominant-pair, targeted-disagreement, and control-gap gates. "
            "Specialist-disagreement routing is closed under this Second Sweep."
        )
    else:
        interpretation = (
            "SS-04 passes the fold-0 screen. Per preregistration, breadth execution "
            "must stop and a separate confirmation execution note is required before "
            "any 5-fold confirmation."
        )

    result = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "branch": "research/new-avenues-pilots-2026",
        "execution_prereg_sha": EXECUTION_PREREG_SHA,
        "head_sha": git_sha(),
        "exp_ids": [EXP_ID, *CONTROL_IDS],
        "constants": {
            "dominant_t_min": DOMINANT_T_MIN,
            "dominant_age_min": DOMINANT_AGE_MIN,
            "action_alpha": ACTION_ALPHA,
            "delta_clip": DELTA_CLIP,
            "damage_penalty": DAMAGE_PENALTY,
            "dominant_pair_weight": DOMINANT_PAIR_WEIGHT,
            "train_pairs_per_t": TRAIN_PAIRS_PER_T,
            "eval_pairs_per_t": EVAL_PAIRS_PER_T,
            "train_pair_seed": TRAIN_PAIR_SEED,
            "global_row_seed": GLOBAL_ROW_SEED,
            "shuffle_seed": SHUFFLE_SEED,
            "action_prob_gap": ACTION_PROB_GAP,
            "min_action_prob": MIN_ACTION_PROB,
            "control_gap": CONTROL_GAP,
        },
        "rt600_sentinel": sentinel,
        "score_coverage": checks,
        "pair_sample_reproduction": {
            "dominant_pairs": pair_flow_pack(c, base, base, fold=0)["dominant_cell"]["total_pairs_sampled"],
            "dominant_rt600_wrong": pair_flow_pack(c, base, base, fold=0)["dominant_cell"]["rt600_wrong"],
            "dominant_rt600_right": pair_flow_pack(c, base, base, fold=0)["dominant_cell"]["rt600_right"],
        },
        EXP_ID: evals[EXP_ID],
        "controls": controls,
        "specialist_disagreement_pair_flow": subgroup,
        "verdict": verdict,
        "gate_failures": failures,
        "interpretation": interpretation,
        "new_oof_lockbox_finite_counts": lock_counts,
        "peak_rss": peak_rss(),
        "runtime_s": float(time.time() - t0),
    }
    write_json(result)
    write_markdown(result)

    train_rows = int(fits[EXP_ID]["fold_summaries"][0]["target"]["n_exposed_rows"])
    rows = [
        result_row(EXP_ID, "recorded", f"SS-04 candidate verdict {verdict}.", fits[EXP_ID]["runtime_s"], train_rows),
        result_row(GLOBAL_ID, "control", "SS-04 global logistic specialist reweighting control.", fits[GLOBAL_ID]["runtime_s"], MAX_GLOBAL_TRAIN_ROWS),
        result_row(STATIC_ID, "control", "SS-04 Pilot-1 static history-fingerprint selector replay control.", fits[STATIC_ID]["runtime_s"], train_rows),
        result_row(SHUFFLED_ID, "control", "SS-04 shuffled disagreement-target router control.", fits[SHUFFLED_ID]["runtime_s"], train_rows),
    ]
    append_rows(c, rows, oofs)
    print(
        json.dumps(
            finite_float(
                {
                    "verdict": verdict,
                    "gate_failures": failures,
                    "marginal_vs_clone": evals[EXP_ID]["ensemble_marginal"]["marginal_vs_clone"],
                    "dominant_pair_flow": evals[EXP_ID]["pair_flow"]["dominant_cell"],
                    "majority_correct_pair_flow": subgroup[EXP_ID]["majority_correct"],
                    "near_split_pair_flow": subgroup[EXP_ID]["near_split"],
                    "control_margins": {
                        eid: controls[eid]["ensemble_marginal"]["marginal_vs_clone"]
                        for eid in CONTROL_IDS
                    },
                    "runtime_s": result["runtime_s"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("preflight", "score"), default="preflight")
    args = ap.parse_args()
    if args.mode == "preflight":
        preflight()
    else:
        run_score()


if __name__ == "__main__":
    main()
