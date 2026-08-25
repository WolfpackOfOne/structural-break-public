#!/usr/bin/env python3
"""SS-01 repair-damage arbiter execution.

Implements the frozen protocol in
research/reports/new_avenues_2026/second_sweep/SS01_EXECUTION_PREREG.md.
No core sbr files are edited by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("SBR_ROOT", Path(__file__).resolve().parents[2]))
os.environ.setdefault("SBR_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(ROOT / "research" / "scripts" / "novel_streams"))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

from harness import diagnostic_pack, marginal, rt600_blend  # noqa: E402
from sbr.metric import ts_auc_flat  # noqa: E402
from wave4_cal import SCDF_NSEEN  # noqa: E402
from wave5_lib import Ctx, FOLDS, SPECIALISTS, load_oof  # noqa: E402

PREREG = ROOT / "research" / "reports" / "new_avenues_2026" / "second_sweep" / "SS01_EXECUTION_PREREG.md"
OUTDIR = ROOT / "research" / "reports" / "new_avenues_2026" / "second_sweep"
OOFDIR = ROOT / "research" / "oof"

EXP_ID = "RT-1219"
GLOBAL_AVG_ID = "RT-1220"
SHUFFLED_ID = "RT-1221"
SINGLE_BEST_ID = "RT-1222"
CONTROL_IDS = (GLOBAL_AVG_ID, SHUFFLED_ID, SINGLE_BEST_ID)

SEEDCLONE = "RT-401"
SENSOR_IDS = [
    "RT-1200",
    "RT-1201",
    "RT-1202",
    "RT-1204",
    "RT-1206",
    "RT-1208",
    "RT-1210",
    "RT-1212",
    "RT-1214",
    "RT-1215",
    "RT-1216",
    "RT-1218",
]
SENSOR_LABELS = {
    "RT-1200": "relay_score_state",
    "RT-1201": "im2_dwell",
    "RT-1202": "trajectory_geometry",
    "RT-1204": "scale_survival",
    "RT-1206": "spectral_impulse",
    "RT-1208": "ordinal_irreversibility",
    "RT-1210": "joint_rarity",
    "RT-1212": "scalar_difficulty",
    "RT-1214": "kalman_nis",
    "RT-1215": "hankel_dmd",
    "RT-1216": "weighted_ctm",
    "RT-1218": "direct_evalue",
}

ACTION_ALPHA = 0.20
DELTA_CLIP = 0.25
DAMAGE_PENALTY = 2.0
TRAIN_PAIRS_PER_T = 64
EVAL_PAIRS_PER_T = 64
PAIR_TRAIN_SEED = 2026082501
PAIR_EVAL_SEED = 20260825
SHUFFLE_SEED = 2026082502
CONTRIB_MIN_ROWS = 100
CONTRIB_MIN_SHARE = 0.01
DOMINANT_T_MIN = 200
DOMINANT_AGE_MIN = 100

LGB_PARAMS = {
    "objective": "multiclass",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 4,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "num_threads": 2,
    "max_bin": 127,
    "seed": 20260825,
    "verbose": -1,
}
NUM_BOOST_ROUND = 150

FIRST_SWEEP_DOMINANT_REPAIRS = 14868
FIRST_SWEEP_DOMINANT_WRONG = 16193
FIRST_SWEEP_DOMINANT_DAMAGE = 28505
FIRST_SWEEP_DOMINANT_RIGHT = 34347
FIRST_SWEEP_UNION_DAMAGE_RATE = 0.830
SCREEN_GATE_MARGIN = 0.0015
CONTROL_MATCH_GAP = 0.0005


def finite_float(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        y = float(x)
        return y if np.isfinite(y) else None
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, np.ndarray):
        return finite_float(x.tolist())
    if isinstance(x, dict):
        return {str(k): finite_float(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite_float(v) for v in x]
    return x


def git_sha(short: bool = True) -> str:
    args = ["git", "-C", str(ROOT), "rev-parse", "--short" if short else "HEAD", "HEAD"]
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def peak_rss() -> dict:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB. Keep the raw number too.
    mib = raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024
    return {"ru_maxrss": raw, "approx_mib": float(mib), "platform": sys.platform}


def append_result_append_only(res: dict) -> None:
    import fcntl

    path = ROOT / "research" / "RESULTS.csv"
    with open(str(path) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with path.open(newline="") as f:
            fieldnames = next(csv.reader(f))
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
            writer.writerow({k: res.get(k, "") for k in fieldnames})
        fcntl.flock(lk, fcntl.LOCK_UN)


def concat_rows(c: Ctx, folds: list[int] | tuple[int, ...]) -> np.ndarray:
    return np.concatenate([c.rows[int(f)] for f in folds]).astype(np.int64, copy=False)


def logit01(x: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(x, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def load_scores(c: Ctx) -> dict[str, np.ndarray]:
    names = list(dict.fromkeys(list(SPECIALISTS) + [SEEDCLONE] + SENSOR_IDS))
    scores = load_oof(names)
    dev = c.dev
    lockbox = c.d.rows_for([-1])
    checks = {}
    for name in names:
        x = scores[name]
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


def calibrated_for_outer(
    c: Ctx,
    scores: dict[str, np.ndarray],
    outer_fold: int,
    names: list[str],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    outer_train_folds = [int(g) for g in FOLDS if int(g) != int(outer_fold)]
    outer_train_rows = concat_rows(c, outer_train_folds)
    val_rows = c.rows[int(outer_fold)]
    for name in names:
        raw = scores[name]
        z = np.full(len(c.d.y), np.nan, dtype=np.float32)
        for g in outer_train_folds:
            fit_folds = [h for h in outer_train_folds if h != g]
            fit_rows = concat_rows(c, fit_folds)
            pred_rows = c.rows[g]
            cal = SCDF_NSEEN(raw[fit_rows], c.d.t[fit_rows])
            z[pred_rows] = cal(raw[pred_rows], c.d.t[pred_rows]).astype(np.float32)
        cal = SCDF_NSEEN(raw[outer_train_rows], c.d.t[outer_train_rows])
        z[val_rows] = cal(raw[val_rows], c.d.t[val_rows]).astype(np.float32)
        out[name] = z
    return out


def base_from_cal(cal: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    spec = np.column_stack([cal[s][rows] for s in SPECIALISTS]).astype(np.float32, copy=False)
    return spec.mean(axis=1).astype(np.float32, copy=False)


def sensor_matrix(cal: dict[str, np.ndarray], rows: np.ndarray, sensors: list[str]) -> np.ndarray:
    return np.column_stack([cal[s][rows] for s in sensors]).astype(np.float32, copy=False)


def make_features(
    c: Ctx,
    cal: dict[str, np.ndarray],
    rows: np.ndarray,
    sensors: list[str],
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    spec = np.column_stack([cal[s][rows] for s in SPECIALISTS]).astype(np.float32, copy=False)
    base = spec.mean(axis=1).astype(np.float32, copy=False)
    seed = cal[SEEDCLONE][rows].astype(np.float32, copy=False)
    smat = sensor_matrix(cal, rows, sensors)
    delta = np.clip(smat - base[:, None], -DELTA_CLIP, DELTA_CLIP).astype(np.float32, copy=False)
    t = c.d.t[rows].astype(np.float32, copy=False)

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
    add("log1p_t", np.log1p(t))
    for cut in (20, 50, 100, 200, 400):
        add(f"t_ge_{cut}", (t >= cut).astype(np.float32))

    add("sensor_delta_std", delta.std(axis=1))
    add("sensor_delta_range", delta.max(axis=1) - delta.min(axis=1))
    add("sensor_positive_delta_frac", (delta > 0.0).mean(axis=1))
    add("sensor_negative_delta_frac", (delta < 0.0).mean(axis=1))
    add("sensor_max_abs_delta", np.abs(delta).max(axis=1))

    for j, sid in enumerate(sensors):
        lab = SENSOR_LABELS[sid]
        add(f"{lab}_delta", delta[:, j])
        add(f"{lab}_abs_delta", np.abs(delta[:, j]))

    X = np.column_stack(cols).astype(np.float32, copy=False)
    if not np.isfinite(X).all():
        raise SystemExit("non-finite arbiter feature matrix")
    return X, names, base, delta


def time_groups(rows: np.ndarray, t: np.ndarray):
    order = np.argsort(t[rows], kind="stable")
    sorted_rows = rows[order]
    sorted_t = t[sorted_rows]
    starts = np.flatnonzero(np.r_[True, sorted_t[1:] != sorted_t[:-1]])
    ends = np.r_[starts[1:], len(sorted_t)]
    for lo, hi in zip(starts, ends):
        yield int(sorted_t[lo]), sorted_rows[lo:hi]


def sample_training_pairs(c: Ctx, rows: np.ndarray, outer_fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pp_all: list[np.ndarray] = []
    nn_all: list[np.ndarray] = []
    ww_all: list[np.ndarray] = []
    for tt, idx in time_groups(rows, c.d.t):
        pos = idx[c.d.y[idx] == 1]
        neg = idx[c.d.y[idx] == 0]
        k = min(TRAIN_PAIRS_PER_T, len(pos), len(neg))
        if k <= 0:
            continue
        rng = np.random.default_rng(PAIR_TRAIN_SEED + 1000 * int(outer_fold) + int(tt))
        pp = rng.choice(pos, k, replace=False)
        nn = rng.choice(neg, k, replace=False)
        w = np.ones(k, dtype=np.float32)
        if tt >= DOMINANT_T_MIN:
            w[c.age[pp] >= DOMINANT_AGE_MIN] = 3.0
        pp_all.append(pp.astype(np.int64, copy=False))
        nn_all.append(nn.astype(np.int64, copy=False))
        ww_all.append(w)
    if not pp_all:
        raise SystemExit(f"outer fold {outer_fold}: no train pairs sampled")
    return np.concatenate(pp_all), np.concatenate(nn_all), np.concatenate(ww_all)


def build_targets(
    c: Ctx,
    cal: dict[str, np.ndarray],
    train_rows: np.ndarray,
    outer_fold: int,
    sensors: list[str],
) -> dict:
    row_to_local = np.full(len(c.d.y), -1, dtype=np.int32)
    row_to_local[train_rows] = np.arange(len(train_rows), dtype=np.int32)
    pp, nn, w = sample_training_pairs(c, train_rows, outer_fold)
    base_p = base_from_cal(cal, pp)
    base_n = base_from_cal(cal, nn)
    sm_p = sensor_matrix(cal, pp, sensors)
    sm_n = sensor_matrix(cal, nn, sensors)
    delta_p = np.clip(sm_p - base_p[:, None], -DELTA_CLIP, DELTA_CLIP)
    delta_n = np.clip(sm_n - base_n[:, None], -DELTA_CLIP, DELTA_CLIP)
    base_diff = base_p - base_n
    action_diff = base_diff[:, None] + ACTION_ALPHA * (delta_p - delta_n)
    base_wrong = base_diff <= 0.0
    base_right = ~base_wrong
    contrib = np.zeros_like(action_diff, dtype=np.float32)
    contrib[base_wrong] = (action_diff[base_wrong] > 0.0).astype(np.float32)
    contrib[base_right] = -DAMAGE_PENALTY * (action_diff[base_right] <= 0.0).astype(np.float32)
    contrib *= w[:, None]

    util = np.zeros((len(train_rows), len(sensors)), dtype=np.float32)
    exposure = np.zeros(len(train_rows), dtype=np.float32)
    lp = row_to_local[pp]
    ln = row_to_local[nn]
    np.add.at(util, lp, contrib)
    np.add.at(util, ln, contrib)
    np.add.at(exposure, lp, w)
    np.add.at(exposure, ln, w)

    exposed = exposure > 0.0
    util_norm = util[exposed] / exposure[exposed, None]
    max_util = util_norm.max(axis=1)
    best = util_norm.argmax(axis=1).astype(np.int16) + 1
    labels = np.where(max_util > 0.0, best, 0).astype(np.int16)
    exposed_rows = train_rows[exposed]
    weights = exposure[exposed].astype(np.float32)

    counts = np.bincount(labels, minlength=len(sensors) + 1)
    return {
        "rows": exposed_rows,
        "labels": labels,
        "weights": weights,
        "target_counts": {str(i): int(v) for i, v in enumerate(counts)},
        "target_sensor_counts": {sensors[i - 1]: int(counts[i]) for i in range(1, len(sensors) + 1)},
        "n_pairs": int(len(pp)),
        "rt600_wrong_pairs": int(base_wrong.sum()),
        "rt600_right_pairs": int(base_right.sum()),
        "mean_pair_weight": float(w.mean()),
        "n_exposed_rows": int(len(exposed_rows)),
        "n_rt600_only_rows": int(counts[0]),
    }


def corrected_score(base: np.ndarray, delta: np.ndarray, cls: np.ndarray) -> np.ndarray:
    out = base.astype(np.float32, copy=True)
    for j in range(delta.shape[1]):
        m = cls == (j + 1)
        if np.any(m):
            out[m] = out[m] + ACTION_ALPHA * delta[m, j]
    return out


def fit_one_outer(
    c: Ctx,
    scores: dict[str, np.ndarray],
    outer_fold: int,
    sensors: list[str],
    shuffled: bool = False,
) -> dict:
    names = list(dict.fromkeys(list(SPECIALISTS) + [SEEDCLONE] + sensors))
    cal = calibrated_for_outer(c, scores, outer_fold, names)
    train_folds = [int(g) for g in FOLDS if int(g) != int(outer_fold)]
    train_rows = concat_rows(c, train_folds)
    val_rows = c.rows[int(outer_fold)]

    target = build_targets(c, cal, train_rows, outer_fold, sensors)
    labels = target["labels"].astype(np.int32, copy=True)
    if shuffled:
        rng = np.random.default_rng(SHUFFLE_SEED + int(outer_fold))
        labels = labels[rng.permutation(len(labels))]

    Xtr, feature_names, _, _ = make_features(c, cal, target["rows"], sensors)
    params = dict(LGB_PARAMS)
    params["num_class"] = len(sensors) + 1
    ds = lgb.Dataset(
        Xtr,
        label=labels,
        weight=target["weights"],
        feature_name=feature_names,
        params=params,
    )
    booster = lgb.train(params, ds, num_boost_round=NUM_BOOST_ROUND)
    del Xtr, ds

    Xva, _, base_va, delta_va = make_features(c, cal, val_rows, sensors)
    pred = booster.predict(Xva)
    cls = np.asarray(np.argmax(pred, axis=1), dtype=np.int16)
    score = corrected_score(base_va, delta_va, cls)
    del Xva

    action_counts = np.bincount(cls, minlength=len(sensors) + 1)
    imp = np.asarray(booster.feature_importance("gain"), dtype=np.float64)
    return {
        "fold": int(outer_fold),
        "val_rows": val_rows,
        "score": score,
        "actions": cls,
        "sensors": list(sensors),
        "target": target,
        "action_counts": {str(i): int(v) for i, v in enumerate(action_counts)},
        "action_sensor_counts": {sensors[i - 1]: int(action_counts[i]) for i in range(1, len(sensors) + 1)},
        "n_non_rt600_actions": int(len(cls) - action_counts[0]),
        "feature_names": feature_names,
        "feature_importance_gain": imp,
    }


def train_arbiter_oof(
    c: Ctx,
    scores: dict[str, np.ndarray],
    sensors: list[str],
    shuffled: bool = False,
    label: str = "arbiter",
) -> dict:
    t0 = time.time()
    oof = np.full(len(c.d.y), np.nan, dtype=np.float32)
    actions = np.full(len(c.d.y), -1, dtype=np.int16)
    fold_summaries = []
    imp_sum = None
    feature_names = None
    for f in FOLDS:
        fold_start = time.time()
        res = fit_one_outer(c, scores, int(f), sensors, shuffled=shuffled)
        oof[res["val_rows"]] = res["score"]
        actions[res["val_rows"]] = res["actions"]
        imp_sum = res["feature_importance_gain"] if imp_sum is None else imp_sum + res["feature_importance_gain"]
        feature_names = res["feature_names"]
        fold_summaries.append(
            {
                "fold": int(f),
                "sensors": list(sensors),
                "target": res["target"],
                "action_counts": res["action_counts"],
                "action_sensor_counts": res["action_sensor_counts"],
                "n_non_rt600_actions": res["n_non_rt600_actions"],
                "runtime_s": float(time.time() - fold_start),
            }
        )
        print(
            f"{label}: fold {f} trained, non-RT600 actions {res['n_non_rt600_actions']} "
            f"({time.time() - fold_start:.1f}s)",
            flush=True,
        )

    if np.isfinite(oof[c.d.rows_for([-1])]).any():
        raise SystemExit(f"{label}: produced lockbox predictions")
    imp_rows = []
    if imp_sum is not None and feature_names is not None:
        order = np.argsort(-imp_sum)
        imp_rows = [
            {"feature": feature_names[int(i)], "gain": float(imp_sum[int(i)])}
            for i in order[:30]
        ]
    return {
        "oof": oof,
        "actions": actions,
        "fold_summaries": fold_summaries,
        "top_feature_importance": imp_rows,
        "runtime_s": float(time.time() - t0),
        "shuffled": bool(shuffled),
        "sensors": list(sensors),
    }


def global_average_oof(c: Ctx, scores: dict[str, np.ndarray]) -> dict:
    t0 = time.time()
    oof = np.full(len(c.d.y), np.nan, dtype=np.float32)
    for f in FOLDS:
        cal = calibrated_for_outer(c, scores, int(f), SENSOR_IDS)
        rows = c.rows[int(f)]
        oof[rows] = sensor_matrix(cal, rows, SENSOR_IDS).mean(axis=1)
    if np.isfinite(oof[c.d.rows_for([-1])]).any():
        raise SystemExit("global average control produced lockbox predictions")
    return {"oof": oof, "runtime_s": float(time.time() - t0)}


def single_best_oof(c: Ctx, scores: dict[str, np.ndarray]) -> dict:
    t0 = time.time()
    oof = np.full(len(c.d.y), np.nan, dtype=np.float32)
    choices = []
    names = list(dict.fromkeys(list(SPECIALISTS) + [SEEDCLONE] + SENSOR_IDS))
    for f in FOLDS:
        cal = calibrated_for_outer(c, scores, int(f), names)
        train_folds = [int(g) for g in FOLDS if int(g) != int(f)]
        train_rows = concat_rows(c, train_folds)
        spec_sum = np.zeros(len(train_rows), dtype=np.float32)
        for s in SPECIALISTS:
            spec_sum += cal[s][train_rows]
        seed_blend = (spec_sum + cal[SEEDCLONE][train_rows]) / 8.0
        y = c.d.y[train_rows]
        t = c.d.t[train_rows]
        seed_auc = float(ts_auc_flat(seed_blend, y, t))
        margins = {}
        for sid in SENSOR_IDS:
            cand_blend = (spec_sum + cal[sid][train_rows]) / 8.0
            margins[sid] = float(ts_auc_flat(cand_blend, y, t) - seed_auc)
        best = max(SENSOR_IDS, key=lambda sid: (margins[sid], -SENSOR_IDS.index(sid)))
        val_rows = c.rows[int(f)]
        oof[val_rows] = cal[best][val_rows]
        choices.append({"fold": int(f), "chosen_sensor": best, "train_marginal_vs_clone": margins[best], "all_margins": margins})
        print(f"single-best control: fold {f} chose {best} train marginal {margins[best]:+.6f}", flush=True)
    if np.isfinite(oof[c.d.rows_for([-1])]).any():
        raise SystemExit("single-best control produced lockbox predictions")
    return {"oof": oof, "choices": choices, "runtime_s": float(time.time() - t0)}


def stable_seed(label: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).hexdigest()
    return int(digest[:16], 16) % (2**32)


def sample_eval_pairs_for_label(c: Ctx, label: str, fold: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rows = c.rows[int(fold)]
    t = c.d.t[rows]
    y = c.d.y[rows]
    age = c.age[rows]
    has_break = c.has_break[c.d.sidx[rows]]

    if label == "whole_fold":
        pos_mask = y == 1
        neg_mask = y == 0
    elif label == "dominant_cell":
        pos_mask = (y == 1) & (age >= DOMINANT_AGE_MIN) & (t >= DOMINANT_T_MIN)
        neg_mask = (y == 0) & (t >= DOMINANT_T_MIN)
    elif label == "cell_never_break_neg":
        pos_mask = (y == 1) & (age >= DOMINANT_AGE_MIN) & (t >= DOMINANT_T_MIN)
        neg_mask = (y == 0) & (t >= DOMINANT_T_MIN) & (has_break == 0)
    elif label == "cell_pre_break_neg":
        pos_mask = (y == 1) & (age >= DOMINANT_AGE_MIN) & (t >= DOMINANT_T_MIN)
        neg_mask = (y == 0) & (t >= DOMINANT_T_MIN) & (has_break == 1)
    else:
        raise ValueError(label)

    rng = np.random.default_rng(stable_seed(label, PAIR_EVAL_SEED))
    pos_by_t: dict[int, np.ndarray] = {}
    neg_by_t: dict[int, np.ndarray] = {}
    for tt in np.unique(t[pos_mask | neg_mask]):
        pos = np.flatnonzero(pos_mask & (t == tt))
        neg = np.flatnonzero(neg_mask & (t == tt))
        if len(pos) and len(neg):
            pos_by_t[int(tt)] = pos
            neg_by_t[int(tt)] = neg

    pos_rows: list[int] = []
    neg_rows: list[int] = []
    for tt in sorted(pos_by_t):
        pos = pos_by_t[tt]
        neg = neg_by_t[tt]
        total = len(pos) * len(neg)
        k = min(EVAL_PAIRS_PER_T, total)
        if total <= EVAL_PAIRS_PER_T:
            chosen = [(int(p), int(n)) for p in pos for n in neg]
        else:
            chosen_set: set[tuple[int, int]] = set()
            max_trials = max(k * 20, 200)
            trials = 0
            while len(chosen_set) < k and trials < max_trials:
                pp = int(pos[rng.integers(0, len(pos))])
                nn = int(neg[rng.integers(0, len(neg))])
                chosen_set.add((pp, nn))
                trials += 1
            if len(chosen_set) < k:
                for pp in pos:
                    for nn in neg:
                        chosen_set.add((int(pp), int(nn)))
                        if len(chosen_set) >= k:
                            break
                    if len(chosen_set) >= k:
                        break
            chosen = sorted(chosen_set)
        for pp, nn in chosen:
            pos_rows.append(int(rows[pp]))
            neg_rows.append(int(rows[nn]))

    return np.asarray(pos_rows, dtype=np.int64), np.asarray(neg_rows, dtype=np.int64)


def extended_pair_stats(c: Ctx, base: np.ndarray, cand: np.ndarray, label: str, fold: int = 0) -> dict:
    pp, nn = sample_eval_pairs_for_label(c, label, fold=fold)
    if len(pp) == 0:
        return {
            "total_pairs_sampled": 0,
            "rt600_wrong": 0,
            "rt600_right": 0,
            "repairs": 0,
            "damage": 0,
            "net_pair_lift": 0,
            "repair_rate_of_rt600_wrong": None,
            "damage_rate_of_rt600_right": None,
        }
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


def dominant_rows(c: Ctx, rows: np.ndarray) -> np.ndarray:
    return rows[
        (c.d.t[rows] >= DOMINANT_T_MIN)
        & ((c.d.y[rows] == 0) | (c.age[rows] >= DOMINANT_AGE_MIN))
    ]


def pair_flow_pack(c: Ctx, base: np.ndarray, cand: np.ndarray, fold: int = 0) -> dict:
    return {
        "whole_fold": extended_pair_stats(c, base, cand, "whole_fold", fold=fold),
        "dominant_cell": extended_pair_stats(c, base, cand, "dominant_cell", fold=fold),
        "mature_vs_never": extended_pair_stats(c, base, cand, "cell_never_break_neg", fold=fold),
        "mature_vs_prebreak": extended_pair_stats(c, base, cand, "cell_pre_break_neg", fold=fold),
    }


def rt600_sentinel(c: Ctx, base: np.ndarray) -> dict:
    mean_auc, per_fold = c.score(base)
    pooled = c.pooled(base)
    dom = dominant_rows(c, c.dev)
    dominant = float(ts_auc_flat(base[dom], c.d.y[dom], c.d.t[dom]))
    marg = marginal(base, c, fold=0, label="rt600_self")
    out = {
        "mean": mean_auc,
        "per_fold": per_fold,
        "pooled": pooled,
        "dominant_cell": dominant,
        "fold0_e0": marg["rt600_7stream"],
        "fold0_e1_seedclone": marg["rt600_plus_seedclone"],
    }
    expected = {
        "mean": 0.625811,
        "pooled": 0.625627,
        "dominant_cell": 0.664277,
        "fold0_e0": 0.638276,
        "fold0_e1_seedclone": 0.638586,
    }
    out["expected"] = expected
    out["diffs"] = {k: out[k] - expected[k] for k in expected}
    out["ok"] = bool(all(abs(v) < 0.005 for v in out["diffs"].values()))
    if not out["ok"]:
        raise SystemExit(f"RT600 sentinel materially differs: {out}")
    return out


def fold0_standalone(c: Ctx, score: np.ndarray) -> float:
    rows = c.rows[0]
    return float(ts_auc_flat(score[rows], c.d.y[rows], c.d.t[rows]))


def summarize_actions(c: Ctx, actions: np.ndarray, sensors: list[str]) -> dict:
    rows = c.rows[0]
    a = actions[rows]
    counts = np.bincount(a[a >= 0], minlength=len(sensors) + 1)
    nonzero = int(counts[1:].sum())
    families = {}
    contributing = []
    for i, sid in enumerate(sensors, start=1):
        n = int(counts[i])
        share = float(n / nonzero) if nonzero else 0.0
        hit = n >= CONTRIB_MIN_ROWS and share >= CONTRIB_MIN_SHARE
        families[sid] = {"rows": n, "share_of_non_rt600_actions": share, "counts_for_gate": bool(hit)}
        if hit:
            contributing.append(sid)
    return {
        "fold0_total_rows": int(len(rows)),
        "rt600_only_rows": int(counts[0]),
        "non_rt600_action_rows": nonzero,
        "families": families,
        "contributing_families": contributing,
        "n_contributing_families": int(len(contributing)),
    }


def evaluate_stream(c: Ctx, base: np.ndarray, score: np.ndarray, exp_id: str, runtime_s: float) -> dict:
    pack = diagnostic_pack(c, score, base, fold=0, label=exp_id)
    pair_flow = pair_flow_pack(c, base, score, fold=0)
    marg = marginal(score, c, fold=0, label=exp_id)
    return {
        "diagnostic_pack": pack,
        "pair_flow": pair_flow,
        "ensemble_marginal": marg,
        "fold0_standalone_ts_auc": fold0_standalone(c, score),
        "runtime_s": runtime_s,
    }


def result_row(
    exp_id: str,
    status: str,
    notes: str,
    runtime_s: float,
    model: str,
    objective: str,
    n_features: int,
    train_rows: int | str,
) -> dict:
    return {
        "experiment_id": exp_id,
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "git_sha": git_sha(),
        "agent": "codex-second-sweep",
        "hypothesis": "SS-01 repair-damage arbitration over frozen first-sweep sensors.",
        "falsification_condition": "KILL if any SS-01 mandatory screen gate fails; controls must not match within +0.0005.",
        "feature_set": "RT600_specialists,RT-401," + ",".join(SENSOR_IDS),
        "n_features": n_features,
        "model": model,
        "objective": objective,
        "folds": "0",
        "random_seed": 20260825,
        "train_series": 8000,
        "train_rows": train_rows,
        "mean_oof_ts_auc": "",
        "pooled_oof_ts_auc": "",
        "per_fold_ts_auc": "",
        "fold_std": 0.0,
        "persistence": "none",
        "sample_mode": f"{EVAL_PAIRS_PER_T}_same_t_pairs_per_t_eval",
        "training_runtime_s": round(runtime_s, 1),
        "causal_verified": "frozen causal OOF sensors + SCDF_NSEEN outer/inner fold-pure calibration; no validation ranks as features",
        "test_reduced_touched": "no",
        "lockbox_touched": "no",
        "status": status,
        "notes": notes,
        "protocol": "second_sweep_ss01_fold0_screen",
    }


def write_oof(exp_id: str, score: np.ndarray) -> str:
    OOFDIR.mkdir(parents=True, exist_ok=True)
    path = OOFDIR / f"{exp_id}.npy"
    np.save(path, score.astype(np.float32, copy=False))
    return str(path)


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
        auc = fold0_standalone(c, sc)
        row["mean_oof_ts_auc"] = auc
        row["pooled_oof_ts_auc"] = auc
        row["per_fold_ts_auc"] = f"{auc:.5f}"
        append_result_append_only(row)
        existing.add(eid)


def gate_verdict(candidate: dict, controls: dict, action_summary: dict) -> tuple[str, list[dict]]:
    failures = []
    margin = candidate["ensemble_marginal"]["marginal_vs_clone"]
    dom = candidate["pair_flow"]["dominant_cell"]
    damage_rate = dom["damage_rate_of_rt600_right"]
    if margin < SCREEN_GATE_MARGIN:
        failures.append({"gate": "marginal_vs_clone", "threshold": SCREEN_GATE_MARGIN, "observed": margin})
    if dom["net_pair_lift"] <= 0:
        failures.append({"gate": "dominant_net_pair_lift", "threshold": ">0", "observed": dom["net_pair_lift"]})
    if damage_rate is None or damage_rate >= FIRST_SWEEP_UNION_DAMAGE_RATE / 2.0:
        failures.append(
            {
                "gate": "rt600_right_dominant_damage_rate",
                "threshold": FIRST_SWEEP_UNION_DAMAGE_RATE / 2.0,
                "observed": damage_rate,
            }
        )
    if action_summary["n_contributing_families"] < 2:
        failures.append(
            {
                "gate": "contributing_sensor_families",
                "threshold": 2,
                "observed": action_summary["n_contributing_families"],
            }
        )
    for name, ctrl in controls.items():
        cm = ctrl["ensemble_marginal"]["marginal_vs_clone"]
        if cm >= margin - CONTROL_MATCH_GAP:
            failures.append(
                {
                    "gate": f"{name}_control_gap",
                    "threshold": f"candidate - control > {CONTROL_MATCH_GAP}",
                    "observed_candidate_minus_control": margin - cm,
                    "control_marginal_vs_clone": cm,
                }
            )
    return ("KILL" if failures else "PASS_SCREEN"), failures


def write_markdown(result: dict) -> None:
    path = OUTDIR / "ss01_repair_damage_arbiter.md"
    cand = result[EXP_ID]
    controls = result["controls"]
    marg = cand["ensemble_marginal"]
    dom = cand["pair_flow"]["dominant_cell"]
    action = result["action_summary"]
    md = [
        "# SS-01 -- Repair-Damage Arbiter",
        "",
        f"Execution preregistration: `research/reports/new_avenues_2026/second_sweep/SS01_EXECUTION_PREREG.md` at `{result['execution_prereg_sha']}`.",
        f"Experiment IDs: `{EXP_ID}` candidate, `{GLOBAL_AVG_ID}` global-average control, `{SHUFFLED_ID}` shuffled-target control, `{SINGLE_BEST_ID}` single-best-arm control.",
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
        "| arm | standalone fold-0 TS-AUC | E2 TS-AUC | marginal vs clone | gain vs RT600 |",
        "|---|---:|---:|---:|---:|",
        f"| RT600 |  | {marg['rt600_7stream']:.6f} |  |  |",
        f"| RT600 + RT-401 seed clone |  | {marg['rt600_plus_seedclone']:.6f} |  |  |",
        (
            f"| RT600 + {EXP_ID} | {cand['fold0_standalone_ts_auc']:.6f} | "
            f"{marg[f'rt600_plus_{EXP_ID}']:.6f} | {marg['marginal_vs_clone']:+.6f} | "
            f"{marg['gain_vs_base']:+.6f} |"
        ),
    ]
    for eid, ctrl in controls.items():
        em = ctrl["ensemble_marginal"]
        md.append(
            f"| RT600 + {eid} | {ctrl['fold0_standalone_ts_auc']:.6f} | "
            f"{em[f'rt600_plus_{eid}']:.6f} | {em['marginal_vs_clone']:+.6f} | "
            f"{em['gain_vs_base']:+.6f} |"
        )
    md += [
        "",
        f"Verdict: **{result['verdict']}**.",
        "Gate failures: "
        + ("; ".join(f"`{x['gate']}` observed `{x.get('observed', x.get('observed_candidate_minus_control'))}`" for x in result["gate_failures"]) if result["gate_failures"] else "none"),
        "",
        "## Pair Flow",
        "",
        f"### {EXP_ID}",
        "",
        "| split | pairs | RT600 wrong | RT600 right | repairs | damage | net | damage rate on RT600-right |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in cand["pair_flow"].items():
        dr = row["damage_rate_of_rt600_right"]
        md.append(
            f"| `{name}` | {row['total_pairs_sampled']} | {row['rt600_wrong']} | {row['rt600_right']} | "
            f"{row['repairs']} | {row['damage']} | {row['net_pair_lift']} | "
            f"{dr:.4f} |" if dr is not None else
            f"| `{name}` | {row['total_pairs_sampled']} | {row['rt600_wrong']} | {row['rt600_right']} | "
            f"{row['repairs']} | {row['damage']} | {row['net_pair_lift']} | n/a |"
        )
    md += [
        "",
        "## Repair Reservoir",
        "",
        f"* Original dominant repair reservoir: `{FIRST_SWEEP_DOMINANT_REPAIRS} / {FIRST_SWEEP_DOMINANT_WRONG}`.",
        f"* Original dominant any-damage baseline: `{FIRST_SWEEP_DOMINANT_DAMAGE} / {FIRST_SWEEP_DOMINANT_RIGHT}`.",
        f"* SS-01 dominant repairs: `{dom['repairs']}`.",
        f"* SS-01 dominant damage: `{dom['damage']}`.",
        f"* Repair reservoir retained: `{result['repair_reservoir']['repair_retention_ratio']:.4f}`.",
        f"* Original candidate-union damage rejected: `{result['repair_reservoir']['damage_rejection_ratio']:.4f}`.",
        "",
        "## Action Use",
        "",
        f"* Non-RT600 action rows: `{action['non_rt600_action_rows']}`.",
        f"* Contributing families for gate: `{action['contributing_families']}`.",
        "",
        "| family | fold-0 rows | share of non-RT600 actions | counts for gate |",
        "|---|---:|---:|---|",
    ]
    for sid, row in action["families"].items():
        md.append(f"| `{sid}` {SENSOR_LABELS[sid]} | {row['rows']} | {row['share_of_non_rt600_actions']:.4f} | {row['counts_for_gate']} |")
    md += [
        "",
        "## Controls",
        "",
        "| control | marginal vs clone | candidate minus control | dominant net |",
        "|---|---:|---:|---:|",
    ]
    cand_margin = marg["marginal_vs_clone"]
    for eid, ctrl in controls.items():
        em = ctrl["ensemble_marginal"]["marginal_vs_clone"]
        md.append(f"| `{eid}` | {em:+.6f} | {cand_margin - em:+.6f} | {ctrl['pair_flow']['dominant_cell']['net_pair_lift']} |")
    md += [
        "",
        "## Family Holdouts",
        "",
        "| removed family | marginal vs clone | candidate minus holdout | dominant net |",
        "|---|---:|---:|---:|",
    ]
    for sid, hold in result["family_holdouts"].items():
        em = hold["ensemble_marginal"]["marginal_vs_clone"]
        md.append(f"| `{sid}` {SENSOR_LABELS[sid]} | {em:+.6f} | {cand_margin - em:+.6f} | {hold['pair_flow']['dominant_cell']['net_pair_lift']} |")
    md += [
        "",
        "## Implementation Audits",
        "",
        f"* Pair-sample reproduction: `{result['pair_sample_reproduction']}`.",
        f"* Sensor coverage: all frozen sensors finite on dev rows and zero finite on lockbox rows.",
        f"* New OOF lockbox finite counts: `{result['new_oof_lockbox_finite_counts']}`.",
        f"* Peak RSS: `{result['peak_rss']}`.",
        f"* Runtime: `{result['runtime_s']:.1f}s`.",
        "",
        "## Interpretation",
        "",
        result["interpretation"],
    ]
    path.write_text("\n".join(md) + "\n")


def write_json(result: dict) -> None:
    path = OUTDIR / "ss01_repair_damage_arbiter.json"
    path.write_text(json.dumps(finite_float(result), indent=2, sort_keys=True) + "\n")


def preflight() -> None:
    c = Ctx()
    scores, checks = load_scores(c)
    base = rt600_blend(c)
    sentinel = rt600_sentinel(c, base)
    dom_stats = extended_pair_stats(c, base, base, "dominant_cell", fold=0)
    out = {
        "git_sha": git_sha(),
        "prereg_exists": PREREG.exists(),
        "rt600_sentinel": sentinel,
        "sensor_coverage": checks,
        "dominant_eval_sample_with_rt600_self": dom_stats,
        "expected_first_sweep_sample_counts": {
            "pairs": 50540,
            "rt600_wrong": FIRST_SWEEP_DOMINANT_WRONG,
            "rt600_right": FIRST_SWEEP_DOMINANT_RIGHT,
        },
    }
    print(json.dumps(finite_float(out), indent=2, sort_keys=True))


def run_score(include_holdouts: bool = True) -> None:
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    c = Ctx()
    scores, sensor_checks = load_scores(c)
    base = rt600_blend(c)
    sentinel = rt600_sentinel(c, base)
    dom_self = extended_pair_stats(c, base, base, "dominant_cell", fold=0)
    pair_sample_reproduction = {
        "dominant_pairs": dom_self["total_pairs_sampled"],
        "dominant_rt600_wrong": dom_self["rt600_wrong"],
        "dominant_rt600_right": dom_self["rt600_right"],
        "matches_first_sweep_counts": bool(
            dom_self["total_pairs_sampled"] == 50540
            and dom_self["rt600_wrong"] == FIRST_SWEEP_DOMINANT_WRONG
            and dom_self["rt600_right"] == FIRST_SWEEP_DOMINANT_RIGHT
        ),
    }
    if not pair_sample_reproduction["matches_first_sweep_counts"]:
        raise SystemExit(f"evaluation pair sample did not reproduce first-sweep counts: {pair_sample_reproduction}")

    print("training RT-1219 candidate arbiter", flush=True)
    candidate_fit = train_arbiter_oof(c, scores, SENSOR_IDS, shuffled=False, label=EXP_ID)
    write_oof(EXP_ID, candidate_fit["oof"])

    print("building RT-1220 global average control", flush=True)
    global_fit = global_average_oof(c, scores)
    write_oof(GLOBAL_AVG_ID, global_fit["oof"])

    print("training RT-1221 shuffled-target control", flush=True)
    shuffled_fit = train_arbiter_oof(c, scores, SENSOR_IDS, shuffled=True, label=SHUFFLED_ID)
    write_oof(SHUFFLED_ID, shuffled_fit["oof"])

    print("building RT-1222 single-best killed-arm control", flush=True)
    single_fit = single_best_oof(c, scores)
    write_oof(SINGLE_BEST_ID, single_fit["oof"])

    candidate_eval = evaluate_stream(c, base, candidate_fit["oof"], EXP_ID, candidate_fit["runtime_s"])
    control_evals = {
        GLOBAL_AVG_ID: evaluate_stream(c, base, global_fit["oof"], GLOBAL_AVG_ID, global_fit["runtime_s"]),
        SHUFFLED_ID: evaluate_stream(c, base, shuffled_fit["oof"], SHUFFLED_ID, shuffled_fit["runtime_s"]),
        SINGLE_BEST_ID: evaluate_stream(c, base, single_fit["oof"], SINGLE_BEST_ID, single_fit["runtime_s"]),
    }
    action_summary = summarize_actions(c, candidate_fit["actions"], SENSOR_IDS)

    family_holdouts = {}
    if include_holdouts:
        for sid in SENSOR_IDS:
            active = [x for x in SENSOR_IDS if x != sid]
            label = f"holdout_{sid}"
            print(f"training family holdout without {sid}", flush=True)
            fit = train_arbiter_oof(c, scores, active, shuffled=False, label=label)
            ev = evaluate_stream(c, base, fit["oof"], label, fit["runtime_s"])
            ev["fit"] = {
                "runtime_s": fit["runtime_s"],
                "fold_summaries": fit["fold_summaries"],
                "top_feature_importance": fit["top_feature_importance"],
                "removed_sensor": sid,
            }
            family_holdouts[sid] = ev

    verdict, failures = gate_verdict(candidate_eval, control_evals, action_summary)
    dom = candidate_eval["pair_flow"]["dominant_cell"]
    repair_retention_ratio = dom["repairs"] / FIRST_SWEEP_DOMINANT_REPAIRS
    damage_rejection_ratio = (FIRST_SWEEP_DOMINANT_DAMAGE - dom["damage"]) / FIRST_SWEEP_DOMINANT_DAMAGE
    repair_reservoir = {
        "first_sweep_dominant_repair_reservoir_repairs": FIRST_SWEEP_DOMINANT_REPAIRS,
        "first_sweep_dominant_repair_reservoir_wrong": FIRST_SWEEP_DOMINANT_WRONG,
        "first_sweep_dominant_union_damage": FIRST_SWEEP_DOMINANT_DAMAGE,
        "first_sweep_dominant_union_right": FIRST_SWEEP_DOMINANT_RIGHT,
        "ss01_dominant_repairs": dom["repairs"],
        "ss01_dominant_damage": dom["damage"],
        "repair_retention_ratio": float(repair_retention_ratio),
        "damage_rejection_ratio": float(damage_rejection_ratio),
    }

    if verdict == "KILL":
        interpretation = (
            "SS-01 is KILL under the preregistered screen. Fold-pure repair-vs-damage "
            "arbitration over frozen first-sweep sensors did not satisfy all mandatory "
            "same-t pair-flow and marginal-vs-clone gates. Under the frozen state "
            "representation, first-sweep killed arms should not be treated as useful "
            "production arbitration sensors."
        )
    else:
        interpretation = (
            "SS-01 passes the fold-0 screen. Per the program preregistration, breadth "
            "execution must stop and the next step is the SS-01 5-fold confirmation."
        )

    result = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "branch": "research/new-avenues-pilots-2026",
        "execution_prereg_sha": "b451aa7",
        "head_sha": git_sha(),
        "exp_ids": [EXP_ID, *CONTROL_IDS],
        "sensor_ids": SENSOR_IDS,
        "constants": {
            "action_alpha": ACTION_ALPHA,
            "delta_clip": DELTA_CLIP,
            "damage_penalty": DAMAGE_PENALTY,
            "train_pairs_per_t": TRAIN_PAIRS_PER_T,
            "eval_pairs_per_t": EVAL_PAIRS_PER_T,
            "pair_train_seed": PAIR_TRAIN_SEED,
            "pair_eval_seed": PAIR_EVAL_SEED,
            "shuffle_seed": SHUFFLE_SEED,
            "contrib_min_rows": CONTRIB_MIN_ROWS,
            "contrib_min_share": CONTRIB_MIN_SHARE,
        },
        "rt600_sentinel": sentinel,
        "sensor_coverage": sensor_checks,
        "pair_sample_reproduction": pair_sample_reproduction,
        EXP_ID: {
            **candidate_eval,
            "fit": {
                "runtime_s": candidate_fit["runtime_s"],
                "fold_summaries": candidate_fit["fold_summaries"],
                "top_feature_importance": candidate_fit["top_feature_importance"],
                "oof_artifact": str(OOFDIR / f"{EXP_ID}.npy"),
            },
        },
        "controls": {
            GLOBAL_AVG_ID: {**control_evals[GLOBAL_AVG_ID], "oof_artifact": str(OOFDIR / f"{GLOBAL_AVG_ID}.npy")},
            SHUFFLED_ID: {
                **control_evals[SHUFFLED_ID],
                "fit": {
                    "runtime_s": shuffled_fit["runtime_s"],
                    "fold_summaries": shuffled_fit["fold_summaries"],
                    "top_feature_importance": shuffled_fit["top_feature_importance"],
                    "oof_artifact": str(OOFDIR / f"{SHUFFLED_ID}.npy"),
                },
            },
            SINGLE_BEST_ID: {
                **control_evals[SINGLE_BEST_ID],
                "choices": single_fit["choices"],
                "oof_artifact": str(OOFDIR / f"{SINGLE_BEST_ID}.npy"),
            },
        },
        "action_summary": action_summary,
        "family_holdouts": family_holdouts,
        "repair_reservoir": repair_reservoir,
        "verdict": verdict,
        "gate_failures": failures,
        "interpretation": interpretation,
        "new_oof_lockbox_finite_counts": {
            eid: int(np.isfinite(arr[c.d.rows_for([-1])]).sum())
            for eid, arr in {
                EXP_ID: candidate_fit["oof"],
                GLOBAL_AVG_ID: global_fit["oof"],
                SHUFFLED_ID: shuffled_fit["oof"],
                SINGLE_BEST_ID: single_fit["oof"],
            }.items()
        },
        "peak_rss": peak_rss(),
        "runtime_s": float(time.time() - t0),
    }
    write_json(result)
    write_markdown(result)

    feature_count = len(candidate_fit["top_feature_importance"])
    if candidate_fit["fold_summaries"]:
        feature_count = len(candidate_fit["fold_summaries"][0]["target"]["target_sensor_counts"]) * 2 + 22
    candidate_train_rows = candidate_fit["fold_summaries"][0]["target"]["n_exposed_rows"]
    shuffled_train_rows = shuffled_fit["fold_summaries"][0]["target"]["n_exposed_rows"]
    control_train_rows = int(len(concat_rows(c, [1, 2, 3, 4])))
    rows = [
        result_row(EXP_ID, "recorded", f"SS-01 candidate verdict {verdict}.", candidate_fit["runtime_s"], "lgbm_multiclass_action_policy", "repair_damage_pair_target", feature_count, candidate_train_rows),
        result_row(GLOBAL_AVG_ID, "control", "SS-01 global average of frozen first-sweep sensors control.", global_fit["runtime_s"], "global_average_frozen_sensors", "none", len(SENSOR_IDS), control_train_rows),
        result_row(SHUFFLED_ID, "control", "SS-01 shuffled repair/damage target arbiter control.", shuffled_fit["runtime_s"], "lgbm_multiclass_action_policy", "shuffled_repair_damage_pair_target", feature_count, shuffled_train_rows),
        result_row(SINGLE_BEST_ID, "control", "SS-01 single-best killed-arm selected on train folds only control.", single_fit["runtime_s"], "train_selected_single_sensor", "train_fold_marginal_selection", len(SENSOR_IDS), control_train_rows),
    ]
    append_rows(
        c,
        rows,
        {
            EXP_ID: candidate_fit["oof"],
            GLOBAL_AVG_ID: global_fit["oof"],
            SHUFFLED_ID: shuffled_fit["oof"],
            SINGLE_BEST_ID: single_fit["oof"],
        },
    )
    print(
        json.dumps(
            finite_float(
                {
                    "verdict": verdict,
                    "gate_failures": failures,
                    "marginal_vs_clone": candidate_eval["ensemble_marginal"]["marginal_vs_clone"],
                    "dominant_pair_flow": candidate_eval["pair_flow"]["dominant_cell"],
                    "repair_reservoir": repair_reservoir,
                    "controls": {
                        eid: ev["ensemble_marginal"]["marginal_vs_clone"]
                        for eid, ev in control_evals.items()
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
    ap.add_argument("--skip-holdouts", action="store_true")
    args = ap.parse_args()
    if args.mode == "preflight":
        preflight()
    else:
        run_score(include_holdouts=not args.skip_holdouts)


if __name__ == "__main__":
    main()
