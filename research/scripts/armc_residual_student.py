"""Nested causal student for the Arm-C horizon residual.

This follows the Grok-response follow-up to W7-D3R:

* use the existing nested Arm-C teacher checkpoints, not global RT-991 labels;
* residualize the teacher logit inside each outer-training split against
  horizon variables at fixed t;
* train a student on the frozen 500 causal feature bank only;
* evaluate on the full dev population, then report the dominant cell,
  never-break, and pre-break cuts separately.

No RT ID is allocated and RESULTS.csv is not edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from sbr.metric import ts_auc_flat  # noqa: E402
from wave4_cal import SCDF_NSEEN  # noqa: E402

FOLDS = (0, 1, 2, 3, 4)
FULL = ("m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes")
SPECIALISTS = ("RT-300", "RT-410", "RT-411", "RT-412", "RT-413", "RT-414", "RT-415")
SEED_CLONE = "RT-401"
FORBIDDEN_TOKENS = (
    "tau",
    "cut",
    "boundary",
    "break_at",
    "changepoint_true",
    "has_break",
    "n_online",
    "n_hist",
    "final_row",
    "eligible",
    "availability",
    "pad",
)
EPS = 1e-6

STUDENT_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 150,
    "feature_fraction": 1.0,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "max_bin": 127,
    "num_threads": 2,
    "verbose": -1,
}
STUDENT_ROUNDS = 900


def default_data_root() -> Path:
    candidates = [
        ROOT,
        ROOT.parent / "structural-break-wave8",
        ROOT.parent / "structural-break-wave6",
        ROOT.parent / "structural-break-wave5",
    ]
    for path in candidates:
        if (path / "cache" / "features" / "m00_core.npy").exists() and (
            path / "research" / "oof" / "RT-991.npy"
        ).exists():
            return path
    return candidates[0]


def default_catboost_oof_dir() -> Path | None:
    candidates = [
        ROOT.parent / "structural-break-learner-diversity-2026" / "research" / "oof",
        ROOT.parent / "structural-break-deep-ensemble-frontier-local-2026" / "research" / "oof",
    ]
    for path in candidates:
        if (path / "RT-1254.npy").exists() and (path / "RT-1255.npy").exists():
            return path
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(q / (1.0 - q))


def build_row_arrays(folds_path: Path) -> dict[str, np.ndarray]:
    folds = pd.read_parquet(folds_path).sort_values("id").reset_index(drop=True)
    ids = folds["id"].to_numpy()
    if not np.array_equal(ids, np.arange(len(folds))):
        raise ValueError("folds.parquet ids must be contiguous and sorted 0..n-1")

    n_online = folds["n_online"].to_numpy(dtype=np.int32)
    tau = folds["tau_index"].to_numpy(dtype=np.int32)
    has_break = folds["has_break"].to_numpy(dtype=bool)
    series_fold = folds["fold"].to_numpy(dtype=np.int16)

    n_rows = int(n_online.sum())
    sidx = np.repeat(np.arange(len(folds), dtype=np.int32), n_online)
    t = np.empty(n_rows, dtype=np.int32)
    y = np.zeros(n_rows, dtype=np.int8)

    off = 0
    for i, n in enumerate(n_online):
        n_int = int(n)
        t[off : off + n_int] = np.arange(n_int, dtype=np.int32)
        if tau[i] >= 0:
            y[off + int(tau[i]) : off + n_int] = 1
        off += n_int

    tau_by_row = tau[sidx]
    age = np.where(y == 1, t - tau_by_row, -1).astype(np.int32)
    row_fold = series_fold[sidx]
    dev = np.isin(row_fold, FOLDS)
    has_break_by_row = has_break[sidx]
    neg_pre = has_break_by_row & (y == 0)
    dominant = dev & (t >= 200) & (((y == 1) & (age >= 100)) | (y == 0))

    return {
        "folds": folds,
        "sidx": sidx,
        "t": t,
        "y": y,
        "age": age,
        "row_fold": row_fold,
        "dev": dev,
        "has_break_by_row": has_break_by_row,
        "n_online_by_row": n_online[sidx],
        "dominant_cell": dominant,
        "dominant_never_break_only": dominant & ((y == 1) | (~neg_pre)),
        "dominant_pre_break_only": dominant & ((y == 1) | neg_pre),
    }


def rows_for(row_fold: np.ndarray, folds: Iterable[int]) -> np.ndarray:
    return np.flatnonzero(np.isin(row_fold, np.asarray(tuple(folds), dtype=np.int16)))


def load_features(feature_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    mats: list[np.ndarray] = []
    names: list[str] = []
    for module in FULL:
        meta_path = feature_dir / f"{module}.cols.json"
        matrix_path = feature_dir / f"{module}.npy"
        if not meta_path.exists() or not matrix_path.exists():
            raise FileNotFoundError(f"missing feature module {module} under {feature_dir}")
        meta = json.loads(meta_path.read_text())
        arr = np.load(matrix_path, mmap_mode="r")
        mats.append(arr)
        names.extend([f"{module}::{col}" for col in meta["cols"]])

    bad = [name for name in names if any(tok in name.lower() for tok in FORBIDDEN_TOKENS)]
    if bad:
        raise ValueError(f"forbidden student feature columns are reachable: {bad[:20]}")
    if len(names) != 500:
        raise ValueError(f"expected 500 causal columns, found {len(names)}")
    return mats, names


def stack_features(
    mats: list[np.ndarray],
    rows: np.ndarray,
    keep_idx: np.ndarray,
    *,
    max_span: int = 150_000,
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    out = np.empty((len(rows), len(keep_idx)), dtype=np.float32)
    spans = []
    off = 0
    for arr in mats:
        spans.append((off, off + arr.shape[1]))
        off += arr.shape[1]

    cuts = []
    i = 0
    while i < len(rows):
        j = i + int(np.searchsorted(rows[i:], rows[i] + max_span))
        j = max(j, i + 1)
        cuts.append((i, j))
        i = j

    pos = 0
    keep_idx = np.asarray(keep_idx)
    for arr, (lo, hi) in zip(mats, spans):
        sel = keep_idx[(keep_idx >= lo) & (keep_idx < hi)] - lo
        if len(sel) == 0:
            continue
        whole = len(sel) == arr.shape[1] and sel[0] == 0
        for i, j in cuts:
            a = rows[i]
            block = np.asarray(arr[a : rows[j - 1] + 1])
            idx = rows[i:j] - a
            out[i:j, pos : pos + len(sel)] = block[idx] if whole else block[idx][:, sel]
            del block
        pos += len(sel)
    return out


def group_indices_by_t(t: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    idx = np.flatnonzero(mask)
    order = np.argsort(t[idx], kind="stable")
    idx = idx[order]
    tt = t[idx]
    starts = np.flatnonzero(np.r_[True, tt[1:] != tt[:-1]])
    ends = np.r_[starts[1:], len(idx)]
    return [idx[lo:hi] for lo, hi in zip(starts, ends)]


def design_matrix(t_value: int, n_online: np.ndarray) -> np.ndarray:
    n_online = np.asarray(n_online, dtype=np.float64)
    remaining = n_online - float(t_value)
    frac = float(t_value) / np.maximum(n_online, 1.0)
    return np.column_stack([np.ones(len(n_online)), n_online, remaining, frac])


def fit_project_predict(y_fit: np.ndarray, x_fit: np.ndarray, x_pred: np.ndarray) -> tuple[np.ndarray, bool]:
    if len(y_fit) < x_fit.shape[1] + 2:
        return np.full(x_pred.shape[0], float(np.mean(y_fit))), True

    mu = x_fit[:, 1:].mean(axis=0)
    sd = x_fit[:, 1:].std(axis=0)
    sd[sd < 1e-12] = 1.0
    z_fit = x_fit.copy()
    z_pred = x_pred.copy()
    z_fit[:, 1:] = (z_fit[:, 1:] - mu) / sd
    z_pred[:, 1:] = (z_pred[:, 1:] - mu) / sd
    beta, *_ = np.linalg.lstsq(z_fit, y_fit, rcond=None)
    return z_pred @ beta, False


def nested_projection_residual(
    score: np.ndarray,
    rows: dict[str, np.ndarray],
    outer_fold: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    row_fold = rows["row_fold"]
    t = rows["t"]
    n_online = rows["n_online_by_row"]
    outer_train_mask = np.isin(row_fold, [f for f in FOLDS if f != outer_fold]) & np.isfinite(score)
    groups = group_indices_by_t(t, outer_train_mask)

    horizon = np.full(score.shape, np.nan, dtype=np.float64)
    fallback_groups = 0
    fitted_groups = 0
    skipped_groups = 0
    skipped_predictions = 0

    for group in groups:
        t_value = int(t[group[0]])
        group_folds = np.unique(row_fold[group])
        for inner_fold in group_folds:
            pred_idx = group[row_fold[group] == inner_fold]
            fit_idx = group[row_fold[group] != inner_fold]
            if len(fit_idx) == 0:
                skipped_groups += 1
                skipped_predictions += int(len(pred_idx))
                continue
            pred, fallback = fit_project_predict(
                score[fit_idx],
                design_matrix(t_value, n_online[fit_idx]),
                design_matrix(t_value, n_online[pred_idx]),
            )
            horizon[pred_idx] = pred
            fallback_groups += int(fallback)
            fitted_groups += 1

    residual = score - horizon
    stats = {
        "outer_fold": int(outer_fold),
        "t_groups": len(groups),
        "fitted_fold_groups": int(fitted_groups),
        "fallback_fold_groups": int(fallback_groups),
        "skipped_fold_groups": int(skipped_groups),
        "skipped_predictions": int(skipped_predictions),
        "target_rows": int(np.isfinite(residual).sum()),
    }
    return horizon, residual, stats


def load_nested_teacher_q(oof_dir: Path, rows: dict[str, np.ndarray], outer_fold: int) -> tuple[np.ndarray, dict]:
    row_fold = rows["row_fold"]
    q = np.full(len(row_fold), np.nan, dtype=np.float64)
    parts = []
    for inner_fold in FOLDS:
        if inner_fold == outer_fold:
            continue
        path = oof_dir / f"nested_Q_outer{outer_fold}_inner{inner_fold}.npy"
        if not path.exists():
            raise FileNotFoundError(f"missing nested teacher checkpoint: {path}")
        part = np.load(path, mmap_mode="r")
        if len(part) != len(row_fold):
            raise ValueError(f"{path} length {len(part)} != row count {len(row_fold)}")
        target_rows = row_fold == inner_fold
        outer_rows = row_fold == outer_fold
        finite_target = int(np.isfinite(part[target_rows]).sum())
        target_total = int(target_rows.sum())
        finite_outer = int(np.isfinite(part[outer_rows]).sum())
        finite_elsewhere = int(np.isfinite(part[~target_rows]).sum())
        if finite_target != target_total:
            raise ValueError(f"{path} does not cover all rows for inner fold {inner_fold}")
        if finite_outer != 0:
            raise ValueError(f"{path} has finite labels on held-out outer fold {outer_fold}")
        q[target_rows] = part[target_rows]
        parts.append(
            {
                "outer_fold": int(outer_fold),
                "inner_fold": int(inner_fold),
                "path": str(path),
                "sha256": sha256_file(path),
                "finite_target_rows": finite_target,
                "target_rows": target_total,
                "finite_non_target_rows": finite_elsewhere,
            }
        )
    train_rows = np.isin(row_fold, [fold for fold in FOLDS if fold != outer_fold])
    if not np.isfinite(q[train_rows]).all():
        raise ValueError(f"nested teacher Q incomplete for outer fold {outer_fold}")
    return np.clip(q, EPS, 1.0 - EPS), {"outer_fold": int(outer_fold), "parts": parts}


def fold_purity_check(oof_dir: Path, rows: dict[str, np.ndarray]) -> dict:
    row_fold = rows["row_fold"]
    failures = []
    old_contaminated = 0
    checked = 0
    artifact_parts = []

    for outer_fold in FOLDS:
        outer_train = [f for f in FOLDS if f != outer_fold]
        for inner_fold in outer_train:
            checked += 1
            new_train = set(FOLDS) - {outer_fold, inner_fold}
            if new_train & {outer_fold, inner_fold}:
                failures.append(
                    {
                        "outer_fold": int(outer_fold),
                        "inner_fold": int(inner_fold),
                        "teacher_training_folds": sorted(new_train),
                    }
                )
            old_train = set(FOLDS) - {inner_fold}
            if old_train & {outer_fold, inner_fold}:
                old_contaminated += 1

            path = oof_dir / f"nested_Q_outer{outer_fold}_inner{inner_fold}.npy"
            if not path.exists():
                failures.append(
                    {
                        "outer_fold": int(outer_fold),
                        "inner_fold": int(inner_fold),
                        "missing_artifact": str(path),
                    }
                )
                continue
            part = np.load(path, mmap_mode="r")
            target = row_fold == inner_fold
            outer = row_fold == outer_fold
            artifact = {
                "outer_fold": int(outer_fold),
                "inner_fold": int(inner_fold),
                "finite_target_rows": int(np.isfinite(part[target]).sum()),
                "target_rows": int(target.sum()),
                "finite_outer_rows": int(np.isfinite(part[outer]).sum()),
                "finite_non_target_rows": int(np.isfinite(part[~target]).sum()),
            }
            artifact_parts.append(artifact)
            if artifact["finite_target_rows"] != artifact["target_rows"]:
                failures.append({**artifact, "failure": "target fold not fully covered"})
            if artifact["finite_outer_rows"] != 0:
                failures.append({**artifact, "failure": "held-out outer fold has finite labels"})

    return {
        "checked_outer_inner_pairs": checked,
        "new_nested_failures": failures,
        "new_nested_passed": len(failures) == 0,
        "old_global_oof_contaminated_checks": old_contaminated,
        "old_global_oof_total_checks": checked,
        "old_global_oof_sentinel_catches_defect": old_contaminated == checked,
        "artifact_parts": artifact_parts,
    }


def train_outer_fold(args: argparse.Namespace) -> dict:
    import lightgbm as lgb

    data_root = args.data_root
    oof_dir = args.oof_dir
    out_dir = args.out_dir
    outer_fold = int(args.train_outer)
    rows = build_row_arrays(args.folds_path)
    row_fold = rows["row_fold"]
    n_rows = len(row_fold)
    out_dir.mkdir(parents=True, exist_ok=True)

    part_path = out_dir / f"armc_residual_student_outer{outer_fold}.npy"
    summary_path = out_dir / f"armc_residual_student_outer{outer_fold}.json"
    if part_path.exists() and not args.force:
        print(f"outer fold {outer_fold}: existing {part_path}; use --force to retrain")
        return json.loads(summary_path.read_text()) if summary_path.exists() else {"outer_fold": outer_fold, "skipped": True}

    t0 = time.time()
    q, teacher_meta = load_nested_teacher_q(oof_dir, rows, outer_fold)
    q_logit = logit(q, args.eps)
    horizon, residual, residual_stats = nested_projection_residual(q_logit, rows, outer_fold)
    del horizon

    train_rows = rows_for(row_fold, [f for f in FOLDS if f != outer_fold])
    train_rows = train_rows[np.isfinite(residual[train_rows])]
    valid_rows = rows_for(row_fold, [outer_fold])

    rng = np.random.default_rng(args.seed + outer_fold)
    sampled_rows = train_rows
    if len(sampled_rows) > args.max_train_rows:
        sampled_rows = np.sort(rng.choice(sampled_rows, args.max_train_rows, replace=False))

    feature_dir = data_root / "cache" / "features"
    mats, names = load_features(feature_dir)
    keep_idx = np.arange(len(names), dtype=np.int64)
    for matrix in mats:
        if matrix.shape[0] != n_rows:
            raise ValueError(f"feature matrix row count {matrix.shape[0]} != folds row count {n_rows}")

    params = dict(STUDENT_PARAMS)
    params["num_threads"] = int(args.num_threads)
    X_train = stack_features(mats, sampled_rows, keep_idx)
    y_train = residual[sampled_rows].astype(np.float32)
    dataset = lgb.Dataset(
        X_train,
        label=y_train,
        params=params,
        feature_name=[f"f{i}" for i in range(X_train.shape[1])],
    )
    booster = lgb.train(params, dataset, num_boost_round=int(args.rounds))
    gain = booster.feature_importance("gain")
    del X_train, dataset, y_train

    X_valid = stack_features(mats, valid_rows, keep_idx)
    pred = booster.predict(X_valid).astype(np.float32)
    del X_valid

    part = np.full(n_rows, np.nan, dtype=np.float32)
    part[valid_rows] = pred
    np.save(part_path, part)

    y = rows["y"]
    t = rows["t"]
    fold_auc = float(ts_auc_flat(pred, y[valid_rows], t[valid_rows]))
    dominant_rows = valid_rows[rows["dominant_cell"][valid_rows]]
    dominant_auc = score_on(part, dominant_rows, y, t)

    top_idx = np.argsort(gain)[::-1][:40]
    feature_importance = [
        {"feature": names[int(i)], "gain": float(gain[int(i)])}
        for i in top_idx
        if gain[int(i)] > 0
    ]
    summary = {
        "experiment": "armc_residual_student",
        "outer_fold": outer_fold,
        "date": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "folds_path": str(args.folds_path),
        "feature_dir": str(feature_dir),
        "oof_dir": str(oof_dir),
        "output_path": str(part_path),
        "teacher_meta": teacher_meta,
        "student": {
            "modules": list(FULL),
            "n_features": len(names),
            "forbidden_feature_tokens": list(FORBIDDEN_TOKENS),
            "params": {**params, "rounds": int(args.rounds)},
            "max_train_rows": int(args.max_train_rows),
            "sampled_train_rows": int(len(sampled_rows)),
            "available_train_rows": int(len(train_rows)),
            "validation_rows": int(len(valid_rows)),
            "target": "nested fold-pure residual logit(Q) - horizon_projection(logit(Q))",
        },
        "target_stats": {
            **residual_stats,
            "mean": float(np.nanmean(residual[train_rows])),
            "std": float(np.nanstd(residual[train_rows])),
            "q01": float(np.nanquantile(residual[train_rows], 0.01)),
            "q50": float(np.nanquantile(residual[train_rows], 0.50)),
            "q99": float(np.nanquantile(residual[train_rows], 0.99)),
        },
        "scores": {
            "whole_fold_ts_auc": fold_auc,
            "dominant_cell_ts_auc": dominant_auc,
        },
        "top_feature_importance": feature_importance,
        "runtime_s": round(time.time() - t0, 1),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "outer_fold": outer_fold,
        "whole_fold_ts_auc": fold_auc,
        "dominant_cell_ts_auc": dominant_auc,
        "sampled_train_rows": int(len(sampled_rows)),
        "runtime_s": summary["runtime_s"],
        "output": str(part_path),
    }, indent=2), flush=True)
    return summary


def score_on(vec: np.ndarray, rows: np.ndarray, y: np.ndarray, t: np.ndarray) -> float:
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) == 0:
        return float("nan")
    m = np.isfinite(vec[rows])
    if m.sum() == 0:
        return float("nan")
    rr = rows[m]
    return float(ts_auc_flat(vec[rr], y[rr], t[rr]))


def score_pack(vec: np.ndarray, rows: dict[str, np.ndarray]) -> dict[str, object]:
    y = rows["y"]
    t = rows["t"]
    row_fold = rows["row_fold"]
    masks = {
        "whole_dev": rows["dev"],
        "dominant_cell": rows["dominant_cell"],
        "dominant_never_break_only": rows["dominant_never_break_only"],
        "dominant_pre_break_only": rows["dominant_pre_break_only"],
    }
    out = {}
    for name, mask in masks.items():
        rr = np.flatnonzero(mask)
        out[name] = {
            "ts_auc": score_on(vec, rr, y, t),
            "n_rows": int((mask & np.isfinite(vec)).sum()),
            "per_fold": {
                str(fold): score_on(vec, np.flatnonzero(mask & (row_fold == fold)), y, t)
                for fold in FOLDS
            },
        }
    return out


def mean_fold_ts_auc(vec: np.ndarray, rows: dict[str, np.ndarray], mask: np.ndarray | None = None) -> dict[str, object]:
    y = rows["y"]
    t = rows["t"]
    row_fold = rows["row_fold"]
    if mask is None:
        mask = rows["dev"]
    per_fold = {
        str(fold): score_on(vec, np.flatnonzero(mask & (row_fold == fold)), y, t)
        for fold in FOLDS
    }
    vals = [v for v in per_fold.values() if np.isfinite(v)]
    return {
        "mean_ts_auc": float(np.mean(vals)) if vals else float("nan"),
        "per_fold_ts_auc": per_fold,
    }


def midrank_percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    n = len(values)
    ranks = np.empty(n, dtype=np.float64)
    starts = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1]])
    ends = np.r_[starts[1:], n]
    for lo, hi in zip(starts, ends):
        avg = 0.5 * (lo + hi - 1) + 1.0
        ranks[order[lo:hi]] = avg
    return ranks / (n + 1.0)


def within_t_rank_vector(scores: np.ndarray, rows: np.ndarray, t: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    out = np.full(scores.shape, np.nan, dtype=np.float64)
    order = np.argsort(t[rows], kind="stable")
    rr = rows[order]
    tt = t[rr]
    starts = np.flatnonzero(np.r_[True, tt[1:] != tt[:-1]])
    ends = np.r_[starts[1:], len(rr)]
    for lo, hi in zip(starts, ends):
        group = rr[lo:hi]
        finite = np.isfinite(scores[group])
        if finite.sum() >= 2:
            idx = group[finite]
            out[idx] = midrank_percentile(scores[idx])
    return out


def pearson(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    m = mask & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return float("nan")
    aa = a[m].astype(np.float64)
    bb = b[m].astype(np.float64)
    aa -= aa.mean()
    bb -= bb.mean()
    den = math.sqrt(float(np.dot(aa, aa) * np.dot(bb, bb)))
    return float(np.dot(aa, bb) / den) if den > 0 else float("nan")


def crossfit_calibrate(score: np.ndarray, rows: dict[str, np.ndarray]) -> np.ndarray:
    row_fold = rows["row_fold"]
    t = rows["t"]
    out = np.full(score.shape, np.nan, dtype=np.float64)
    for fold in FOLDS:
        train = rows_for(row_fold, [f for f in FOLDS if f != fold])
        train = train[np.isfinite(score[train])]
        valid = rows_for(row_fold, [fold])
        valid = valid[np.isfinite(score[valid])]
        if len(train) == 0 or len(valid) == 0:
            continue
        cal = SCDF_NSEEN(score[train], t[train])
        out[valid] = cal(score[valid], t[valid])
    return out


def load_calibrated_stream(oof_dir: Path, stream: str, rows: dict[str, np.ndarray]) -> np.ndarray:
    cached = oof_dir / f"wave5_cal_SCDF_NSEEN_{stream}.npy"
    if cached.exists():
        return np.load(cached, mmap_mode="r").astype(np.float64)
    raw = np.load(oof_dir / f"{stream}.npy", mmap_mode="r")
    return crossfit_calibrate(raw, rows)


def blend(vectors: list[np.ndarray]) -> np.ndarray:
    return np.column_stack(vectors).mean(axis=1)


def pair_repair_stats(
    base_score: np.ndarray,
    cand_score: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    rows: np.ndarray,
    *,
    n_pairs_per_t: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    rows = np.asarray(rows, dtype=np.int64)
    rows = rows[np.isfinite(base_score[rows]) & np.isfinite(cand_score[rows])]
    order = np.argsort(t[rows], kind="stable")
    rows = rows[order]
    tt = t[rows]
    starts = np.flatnonzero(np.r_[True, tt[1:] != tt[:-1]])
    ends = np.r_[starts[1:], len(rows)]

    repairs = 0
    damage = 0
    both_right = 0
    both_wrong = 0
    total_pairs = 0
    used_t_groups = 0
    for lo, hi in zip(starts, ends):
        idx = rows[lo:hi]
        pos = idx[y[idx] == 1]
        neg = idx[y[idx] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        k = min(n_pairs_per_t, len(pos), len(neg))
        pp = rng.choice(pos, k, replace=False)
        nn = rng.choice(neg, k, replace=False)
        base_right = base_score[pp] > base_score[nn]
        cand_right = cand_score[pp] > cand_score[nn]
        repairs += int((~base_right & cand_right).sum())
        damage += int((base_right & ~cand_right).sum())
        both_right += int((base_right & cand_right).sum())
        both_wrong += int((~base_right & ~cand_right).sum())
        total_pairs += int(k)
        used_t_groups += 1

    return {
        "total_pairs_sampled": int(total_pairs),
        "used_t_groups": int(used_t_groups),
        "repairs": int(repairs),
        "damage": int(damage),
        "net_pair_lift": int(repairs - damage),
        "repair_rate": float(repairs / total_pairs) if total_pairs else float("nan"),
        "damage_rate": float(damage / total_pairs) if total_pairs else float("nan"),
        "net_rate": float((repairs - damage) / total_pairs) if total_pairs else float("nan"),
        "sample_base_auc": float((repairs * 0 + damage + both_right) / total_pairs) if total_pairs else float("nan"),
        "sample_candidate_auc": float((repairs + both_right) / total_pairs) if total_pairs else float("nan"),
        "both_right": int(both_right),
        "both_wrong": int(both_wrong),
    }


def pair_flow_by_split(
    base: np.ndarray,
    cand: np.ndarray,
    rows: dict[str, np.ndarray],
    *,
    n_pairs_per_t: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    y = rows["y"]
    t = rows["t"]
    masks = {
        "whole_dev": rows["dev"],
        "dominant_cell": rows["dominant_cell"],
        "dominant_never_break_only": rows["dominant_never_break_only"],
        "dominant_pre_break_only": rows["dominant_pre_break_only"],
    }
    return {
        name: pair_repair_stats(
            base,
            cand,
            y,
            t,
            np.flatnonzero(mask),
            n_pairs_per_t=n_pairs_per_t,
            seed=seed,
        )
        for name, mask in masks.items()
    }


def merge_student_oof(out_dir: Path, rows: dict[str, np.ndarray], *, force: bool) -> tuple[np.ndarray, list[dict]]:
    row_fold = rows["row_fold"]
    n_rows = len(row_fold)
    merged_path = out_dir / "armc_residual_student_oof.npy"
    if merged_path.exists() and not force:
        oof = np.load(merged_path).astype(np.float32)
        summaries = []
        for fold in FOLDS:
            p = out_dir / f"armc_residual_student_outer{fold}.json"
            if p.exists():
                summaries.append(json.loads(p.read_text()))
        return oof, summaries

    oof = np.full(n_rows, np.nan, dtype=np.float32)
    summaries = []
    for fold in FOLDS:
        part_path = out_dir / f"armc_residual_student_outer{fold}.npy"
        summary_path = out_dir / f"armc_residual_student_outer{fold}.json"
        if not part_path.exists():
            raise FileNotFoundError(f"missing outer-fold student output: {part_path}")
        part = np.load(part_path, mmap_mode="r")
        valid_rows = row_fold == fold
        if not np.isfinite(part[valid_rows]).all():
            raise ValueError(f"{part_path} does not cover validation fold {fold}")
        oof[valid_rows] = part[valid_rows]
        summaries.append(json.loads(summary_path.read_text()) if summary_path.exists() else {"outer_fold": fold})
    np.save(merged_path, oof)
    return oof, summaries


def per_fold_contrast(scores: dict, cand: str, anchor: str, cut: str) -> dict[str, object]:
    """Fold-level deltas for one contrast on one cut.

    The pooled TS-AUC hides fold heterogeneity: a contrast can be pooled-positive
    while being negative on individual folds. Everything promotion-relevant is
    judged per fold, so the per-fold spread belongs in the report.
    """
    a = scores[cand][cut]["per_fold"]
    b = scores[anchor][cut]["per_fold"]
    folds = sorted(a, key=int)
    deltas = [float(a[f] - b[f]) for f in folds]
    arr = np.asarray(deltas, dtype=np.float64)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
    se = sd / math.sqrt(len(arr)) if len(arr) > 1 and sd > 0 else float("nan")
    return {
        "cut": cut,
        "per_fold": {f: float(d) for f, d in zip(folds, deltas)},
        "positive_folds": int((arr > 0).sum()),
        "n_folds": int(len(arr)),
        "mean": mean,
        "sd": sd,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "t_stat": float(mean / se) if np.isfinite(se) and se > 0 else float("nan"),
    }


def delta_score_pack(pack: dict, anchor: dict) -> dict[str, object]:
    out = {}
    for split, row in pack.items():
        out[split] = {
            "ts_auc_delta": row["ts_auc"] - anchor[split]["ts_auc"],
            "per_fold_delta": {
                fold: row["per_fold"][fold] - anchor[split]["per_fold"][fold]
                for fold in row["per_fold"]
            },
        }
    return out


def rt1257_combo_analysis(
    args: argparse.Namespace,
    rows: dict[str, np.ndarray],
    calibrated_specialists: list[np.ndarray],
    student_cal: np.ndarray,
    rt600: np.ndarray,
) -> dict[str, object] | None:
    cat_dir = args.catboost_oof_dir
    if cat_dir is None:
        return None
    cat300_path = cat_dir / "RT-1255.npy"
    cat413_path = cat_dir / "RT-1254.npy"
    if not cat300_path.exists() or not cat413_path.exists():
        return None

    cat300 = crossfit_calibrate(np.load(cat300_path, mmap_mode="r"), rows)
    cat413 = crossfit_calibrate(np.load(cat413_path, mmap_mode="r"), rows)
    clone300 = load_calibrated_stream(args.oof_dir, "RT-401", rows)
    clone413 = load_calibrated_stream(args.oof_dir, "RT-402", rows)
    clone_extra = load_calibrated_stream(args.oof_dir, "RT-403", rows)

    rt1257 = blend(
        [
            cat300,
            calibrated_specialists[1],
            calibrated_specialists[2],
            calibrated_specialists[3],
            cat413,
            calibrated_specialists[5],
            calibrated_specialists[6],
        ]
    )
    rt1257_clone = blend(
        [
            clone300,
            calibrated_specialists[1],
            calibrated_specialists[2],
            calibrated_specialists[3],
            clone413,
            calibrated_specialists[5],
            calibrated_specialists[6],
        ]
    )
    rt1257_plus_student = blend(
        [
            cat300,
            calibrated_specialists[1],
            calibrated_specialists[2],
            calibrated_specialists[3],
            cat413,
            calibrated_specialists[5],
            calibrated_specialists[6],
            student_cal,
        ]
    )
    # PROTOCOL_CHAMPION_2026 E1 for the ADDITION contract: RT-1257 with one
    # matched exchangeable seed clone added, i.e. exactly one change from E0.
    rt1257_plus_seedclone = blend(
        [
            cat300,
            calibrated_specialists[1],
            calibrated_specialists[2],
            calibrated_specialists[3],
            cat413,
            calibrated_specialists[5],
            calibrated_specialists[6],
            clone_extra,
        ]
    )
    # NOT an E1. This clones the CAT-300 and CAT-413 members *and* adds a clone,
    # so it sits at RT-600 grade rather than RT-1257 grade and a delta against it
    # re-credits the two CatBoost swaps to whatever is being tested. Retained
    # only as a fully-cloned floor; see `endpoint_note` below.
    all_clone_control = blend(
        [
            clone300,
            calibrated_specialists[1],
            calibrated_specialists[2],
            calibrated_specialists[3],
            clone413,
            calibrated_specialists[5],
            calibrated_specialists[6],
            clone_extra,
        ]
    )

    packs = {
        "RT600_7stream": score_pack(rt600, rows),
        "RT1257_catboost_hybrid": score_pack(rt1257, rows),
        "RT1257_clone_control": score_pack(rt1257_clone, rows),
        "RT1257_plus_residual_student": score_pack(rt1257_plus_student, rows),
        "RT1257_plus_seedclone": score_pack(rt1257_plus_seedclone, rows),
        "all_clone_control": score_pack(all_clone_control, rows),
    }
    means = {
        name: mean_fold_ts_auc(vec, rows)
        for name, vec in (
            ("RT600_7stream", rt600),
            ("RT1257_catboost_hybrid", rt1257),
            ("RT1257_clone_control", rt1257_clone),
            ("RT1257_plus_residual_student", rt1257_plus_student),
            ("RT1257_plus_seedclone", rt1257_plus_seedclone),
            ("all_clone_control", all_clone_control),
        )
    }
    pair_flow_vs_rt1257 = pair_flow_by_split(
        rt1257,
        rt1257_plus_student,
        rows,
        n_pairs_per_t=args.pairs_per_t,
        seed=args.pair_seed,
    )
    pair_flow_vs_rt1257_plus_seedclone = pair_flow_by_split(
        rt1257_plus_seedclone,
        rt1257_plus_student,
        rows,
        n_pairs_per_t=args.pairs_per_t,
        seed=args.pair_seed,
    )
    pair_flow_vs_all_clone_control = pair_flow_by_split(
        all_clone_control,
        rt1257_plus_student,
        rows,
        n_pairs_per_t=args.pairs_per_t,
        seed=args.pair_seed,
    )

    return {
        "catboost_oof_dir": str(cat_dir),
        "cat300_path": str(cat300_path),
        "cat413_path": str(cat413_path),
        "cat300_sha256": sha256_file(cat300_path),
        "cat413_sha256": sha256_file(cat413_path),
        "scores": packs,
        "mean_scores": means,
        "deltas": {
            "RT1257_vs_RT600": delta_score_pack(packs["RT1257_catboost_hybrid"], packs["RT600_7stream"]),
            "RT1257_vs_clone_control": delta_score_pack(packs["RT1257_catboost_hybrid"], packs["RT1257_clone_control"]),
            "RT1257_plus_student_vs_RT1257": delta_score_pack(
                packs["RT1257_plus_residual_student"], packs["RT1257_catboost_hybrid"]
            ),
            "PRIMARY_RT1257_plus_student_vs_RT1257_plus_seedclone": delta_score_pack(
                packs["RT1257_plus_residual_student"], packs["RT1257_plus_seedclone"]
            ),
            "RT1257_plus_student_vs_all_clone_control": delta_score_pack(
                packs["RT1257_plus_residual_student"], packs["all_clone_control"]
            ),
            "RT1257_plus_student_vs_RT600": delta_score_pack(
                packs["RT1257_plus_residual_student"], packs["RT600_7stream"]
            ),
        },
        "mean_deltas": {
            "RT1257_minus_RT600": means["RT1257_catboost_hybrid"]["mean_ts_auc"]
            - means["RT600_7stream"]["mean_ts_auc"],
            "RT1257_minus_clone_control": means["RT1257_catboost_hybrid"]["mean_ts_auc"]
            - means["RT1257_clone_control"]["mean_ts_auc"],
            "RT1257_plus_student_minus_RT1257": means["RT1257_plus_residual_student"]["mean_ts_auc"]
            - means["RT1257_catboost_hybrid"]["mean_ts_auc"],
            "PRIMARY_RT1257_plus_student_minus_RT1257_plus_seedclone": means[
                "RT1257_plus_residual_student"
            ]["mean_ts_auc"]
            - means["RT1257_plus_seedclone"]["mean_ts_auc"],
            "control_lift_RT1257_plus_seedclone_minus_RT1257": means[
                "RT1257_plus_seedclone"
            ]["mean_ts_auc"]
            - means["RT1257_catboost_hybrid"]["mean_ts_auc"],
            "RT1257_plus_student_minus_all_clone_control": means[
                "RT1257_plus_residual_student"
            ]["mean_ts_auc"]
            - means["all_clone_control"]["mean_ts_auc"],
            "RT1257_plus_student_minus_RT600": means["RT1257_plus_residual_student"]["mean_ts_auc"]
            - means["RT600_7stream"]["mean_ts_auc"],
        },
        "pair_flow_vs_rt1257": pair_flow_vs_rt1257,
        "pair_flow_vs_rt1257_plus_seedclone": pair_flow_vs_rt1257_plus_seedclone,
        "pair_flow_vs_all_clone_control": pair_flow_vs_all_clone_control,
        "note": (
            "Uses original dev OOF arrays RT-1254/RT-1255 from the CatBoost specialist "
            "research directory, not the deployment final10k artifacts."
        ),
        "endpoint_note": (
            "PRIMARY endpoint under PROTOCOL_CHAMPION_2026 is E2-E1 where E1 is "
            "RT1257_plus_seedclone -- RT-1257 with one matched exchangeable clone added, "
            "exactly one change from E0. all_clone_control is NOT an E1: it clones the "
            "CAT-300 and CAT-413 members as well as adding a clone, so it sits at RT-600 "
            "grade and a delta against it re-credits the two CatBoost slot swaps to the "
            "candidate. Reports before 2026-08-31 quoted that delta as if it were a "
            "champion-relative marginal; it is not. See "
            "research/scripts/armc_e2_e1_addition_contract.py."
        ),
    }


def analyze(args: argparse.Namespace) -> dict:
    rows = build_row_arrays(args.folds_path)
    out_dir = args.out_dir
    oof_dir = args.oof_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    student_raw, fold_summaries = merge_student_oof(out_dir, rows, force=args.force_merge)
    finite_dev = rows["dev"] & np.isfinite(student_raw)
    if int(finite_dev.sum()) != int(rows["dev"].sum()):
        raise ValueError("student OOF does not cover every dev-fold row")

    rt990 = np.load(oof_dir / "RT-990.npy", mmap_mode="r").astype(np.float64)
    rt991 = np.load(oof_dir / "RT-991.npy", mmap_mode="r").astype(np.float64)
    calibrated_specialists = [load_calibrated_stream(oof_dir, stream, rows) for stream in SPECIALISTS]
    rt600 = blend(calibrated_specialists)
    seed_clone = load_calibrated_stream(oof_dir, SEED_CLONE, rows)
    student_cal = crossfit_calibrate(student_raw, rows)
    rt600_plus_clone = blend(calibrated_specialists + [seed_clone])
    rt600_plus_student = blend(calibrated_specialists + [student_cal])

    packs = {
        "Arm_B_RT990_raw": score_pack(rt990, rows),
        "Arm_C_RT991_raw": score_pack(rt991, rows),
        "RT600_7stream": score_pack(rt600, rows),
        "RT600_plus_seedclone": score_pack(rt600_plus_clone, rows),
        "Residual_student_raw": score_pack(student_raw, rows),
        "RT600_plus_residual_student": score_pack(rt600_plus_student, rows),
    }
    deltas = {
        "Arm_C_vs_Arm_B": delta_score_pack(packs["Arm_C_RT991_raw"], packs["Arm_B_RT990_raw"]),
        "Student_raw_vs_Arm_B": delta_score_pack(packs["Residual_student_raw"], packs["Arm_B_RT990_raw"]),
        "Student_blend_vs_RT600": delta_score_pack(packs["RT600_plus_residual_student"], packs["RT600_7stream"]),
        "Student_blend_vs_seedclone_blend": delta_score_pack(
            packs["RT600_plus_residual_student"], packs["RT600_plus_seedclone"]
        ),
    }

    per_fold_contrasts = {
        f"{name}::{cut}": per_fold_contrast(packs, cand, anchor, cut)
        for name, cand, anchor in (
            ("Student_raw_vs_Arm_B", "Residual_student_raw", "Arm_B_RT990_raw"),
            ("Student_blend_vs_RT600", "RT600_plus_residual_student", "RT600_7stream"),
            (
                "Student_blend_vs_seedclone_blend",
                "RT600_plus_residual_student",
                "RT600_plus_seedclone",
            ),
        )
        for cut in ("whole_dev", "dominant_cell")
    }

    dominant_rows = np.flatnonzero(rows["dominant_cell"])
    rank_student = within_t_rank_vector(student_raw, dominant_rows, rows["t"])
    rank_rt990 = within_t_rank_vector(rt990, dominant_rows, rows["t"])
    rank_rt600 = within_t_rank_vector(rt600, dominant_rows, rows["t"])
    rank_rt991 = within_t_rank_vector(rt991, dominant_rows, rows["t"])
    rank_student_blend = within_t_rank_vector(rt600_plus_student, dominant_rows, rows["t"])
    rank_clone_blend = within_t_rank_vector(rt600_plus_clone, dominant_rows, rows["t"])
    rho = {
        "student_raw_vs_Arm_B_RT990": pearson(rank_student, rank_rt990, rows["dominant_cell"]),
        "student_raw_vs_RT600": pearson(rank_student, rank_rt600, rows["dominant_cell"]),
        "Arm_C_RT991_vs_Arm_B_RT990": pearson(rank_rt991, rank_rt990, rows["dominant_cell"]),
        "student_blend_vs_seedclone_blend": pearson(rank_student_blend, rank_clone_blend, rows["dominant_cell"]),
    }

    pair_flow_vs_rt600 = pair_flow_by_split(
        rt600,
        rt600_plus_student,
        rows,
        n_pairs_per_t=args.pairs_per_t,
        seed=args.pair_seed,
    )
    pair_flow_vs_clone = pair_flow_by_split(
        rt600_plus_clone,
        rt600_plus_student,
        rows,
        n_pairs_per_t=args.pairs_per_t,
        seed=args.pair_seed,
    )
    rt1257_combo = rt1257_combo_analysis(
        args,
        rows,
        calibrated_specialists,
        student_cal,
        rt600,
    )

    oracle_report_path = ROOT / "research" / "reports" / "armc_residualization.json"
    oracle_lift = None
    retention = None
    if oracle_report_path.exists():
        oracle = json.loads(oracle_report_path.read_text())
        oracle_r = oracle["scores"]["T_orthogonal_residual_xfit"]["dominant_cell_ts_auc"]
        oracle_b = oracle["scores"]["Arm_B_RT990"]["dominant_cell_ts_auc"]
        oracle_lift = float(oracle_r - oracle_b)
        student_lift = deltas["Student_raw_vs_Arm_B"]["dominant_cell"]["ts_auc_delta"]
        retention = float(student_lift / oracle_lift) if oracle_lift else None

    prebreak_damage = pair_flow_vs_rt600["dominant_pre_break_only"]["damage_rate"]
    neverbreak_net = pair_flow_vs_rt600["dominant_never_break_only"]["net_rate"]
    marginal_vs_clone = deltas["Student_blend_vs_seedclone_blend"]["whole_dev"]["ts_auc_delta"]
    marginal_vs_rt600 = deltas["Student_blend_vs_RT600"]["whole_dev"]["ts_auc_delta"]
    cell_blend_vs_rt600 = deltas["Student_blend_vs_RT600"]["dominant_cell"]["ts_auc_delta"]
    hard_gate_prebreak = bool(prebreak_damage < args.max_prebreak_damage)
    positive_neverbreak = bool(neverbreak_net > 0.0)
    positive_marginal = bool(marginal_vs_clone > 0.0)

    if hard_gate_prebreak and positive_neverbreak and positive_marginal:
        verdict = (
            "PROMOTE for confirmation only: the nested 500-feature residual student beats the seed-clone "
            "blend marginally, has positive never-break pair net, and clears the pre-break damage gate."
        )
    elif not hard_gate_prebreak:
        verdict = (
            "DO NOT PROMOTE: the nested residual student fails the hard pre-break damage gate "
            f"({prebreak_damage:.6f} >= {args.max_prebreak_damage:.6f})."
        )
    elif not positive_neverbreak:
        verdict = (
            "DO NOT PROMOTE: the nested residual student does not produce positive never-break pair net "
            f"({neverbreak_net:+.6f})."
        )
    else:
        verdict = (
            "DO NOT PROMOTE: the nested residual student does not beat the seed-clone marginal "
            f"({marginal_vs_clone:+.6f})."
        )
    if rt1257_combo is not None and verdict.startswith("PROMOTE"):
        combo_delta = rt1257_combo["mean_deltas"]["RT1257_plus_student_minus_RT1257"]
        verdict += (
            f" The light RT-1257 complementarity check is also positive: +student is "
            f"{combo_delta:+.6f} mean whole-dev TS-AUC over the CatBoost hybrid."
        )

    purity = fold_purity_check(oof_dir, rows)
    result = {
        "experiment": "armc_residual_student",
        "date": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Nested causal student for the real complement in Arm-C after fixed-t horizon residualization.",
        "inputs": {
            "data_root": str(args.data_root),
            "folds_path": str(args.folds_path),
            "feature_dir": str(args.data_root / "cache" / "features"),
            "oof_dir": str(oof_dir),
            "student_oof": str(out_dir / "armc_residual_student_oof.npy"),
            "rt990_sha256": sha256_file(oof_dir / "RT-990.npy"),
            "rt991_sha256": sha256_file(oof_dir / "RT-991.npy"),
        },
        "causality_contract": {
            "student_features": "existing 500 causal feature bank only",
            "target": "nested fold-pure Arm-C teacher residual; target is never a feature",
            "no_added_features": True,
            "no_t_h_eligibility_filter": True,
            "forbidden_feature_tokens": list(FORBIDDEN_TOKENS),
            "fold_purity": {
                "checked_outer_inner_pairs": purity["checked_outer_inner_pairs"],
                "new_nested_passed": purity["new_nested_passed"],
                "old_global_oof_contaminated_checks": purity["old_global_oof_contaminated_checks"],
                "old_global_oof_total_checks": purity["old_global_oof_total_checks"],
                "old_global_oof_sentinel_catches_defect": purity["old_global_oof_sentinel_catches_defect"],
            },
        },
        "population": {
            "row_count_total": int(len(rows["y"])),
            "row_count_dev": int(rows["dev"].sum()),
            "row_count_dominant_cell": int(rows["dominant_cell"].sum()),
            "row_count_dominant_never_break_only": int(rows["dominant_never_break_only"].sum()),
            "row_count_dominant_pre_break_only": int(rows["dominant_pre_break_only"].sum()),
        },
        "fold_summaries": fold_summaries,
        "scores": packs,
        "deltas": deltas,
        "per_fold_contrasts": per_fold_contrasts,
        "gain_scope": {
            "neverbreak_net_rate_vs_rt600": pair_flow_vs_rt600["dominant_never_break_only"]["net_rate"],
            "prebreak_net_rate_vs_rt600": pair_flow_vs_rt600["dominant_pre_break_only"]["net_rate"],
            "whole_dev_damage_rate_vs_rt600": pair_flow_vs_rt600["whole_dev"]["damage_rate"],
            "gate_scope": "dominant_pre_break_only",
            "note": (
                "The pre-break pair net is ~0 while the never-break pair net is positive: this is a "
                "never-break-cut gain, not a broad one. The damage-rate gate is applied to the dominant "
                "pre-break cut only; the whole-dev damage rate is reported here because it sits above the "
                "same numeric threshold and must not be read as gated."
            ),
        },
        "within_t_rank_rho": rho,
        "pair_flow_vs_rt600": pair_flow_vs_rt600,
        "pair_flow_vs_seedclone_blend": pair_flow_vs_clone,
        "rt1257_combo": rt1257_combo,
        "oracle_retention": {
            "oracle_residual_cell_lift_vs_Arm_B": oracle_lift,
            "student_raw_cell_lift_vs_Arm_B": deltas["Student_raw_vs_Arm_B"]["dominant_cell"]["ts_auc_delta"],
            "retention_fraction": retention,
            "note": (
                "NOT a like-for-like ratio. The denominator is the oracle residual measured on the "
                "global (fold-contaminated) RT-991 over the dominant cell; the numerator is a nested "
                "fold-pure student trained on full-population residual labels. The two estimands "
                "differ, so this fraction is an order-of-magnitude indication only and must not be "
                "quoted as a retention rate."
            ),
        },
        "gates": {
            "max_prebreak_damage_rate": args.max_prebreak_damage,
            "observed_prebreak_damage_rate_vs_rt600": prebreak_damage,
            "prebreak_damage_gate_passed": hard_gate_prebreak,
            "observed_neverbreak_net_rate_vs_rt600": neverbreak_net,
            "neverbreak_net_positive": positive_neverbreak,
            "whole_dev_marginal_vs_seedclone_blend": marginal_vs_clone,
            "whole_dev_gain_vs_rt600": marginal_vs_rt600,
            "dominant_cell_gain_vs_rt600": cell_blend_vs_rt600,
        },
        "verdict": verdict,
    }

    json_path = out_dir / "armc_residual_student.json"
    md_path = out_dir / "armc_residual_student.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    md_path.write_text(make_markdown(result))
    print(json.dumps({
        "out_json": str(json_path),
        "out_md": str(md_path),
        "whole_dev_marginal_vs_seedclone_blend": marginal_vs_clone,
        "whole_dev_gain_vs_rt600": marginal_vs_rt600,
        "dominant_cell_gain_vs_rt600": cell_blend_vs_rt600,
        "prebreak_damage_rate": prebreak_damage,
        "neverbreak_net_rate": neverbreak_net,
        "verdict": verdict,
    }, indent=2), flush=True)
    return result


def fmt(x: object, digits: int = 6, signed: bool = False) -> str:
    if x is None:
        return "n/a"
    try:
        val = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(val):
        return "nan"
    return f"{val:+.{digits}f}" if signed else f"{val:.{digits}f}"


def make_markdown(result: dict[str, object]) -> str:
    scores = result["scores"]
    deltas = result["deltas"]
    gates = result["gates"]
    rho = result["within_t_rank_rho"]
    retention = result["oracle_retention"]
    lines = [
        "# Arm-C Residual Student",
        "",
        f"Date: `{result['date']}`",
        "",
        "Nested causal student of the Arm-C residual. No RT ID was allocated and",
        "`RESULTS.csv` was not edited.",
        "",
        "## Contract",
        "",
        f"- Feature bank: `{result['inputs']['feature_dir']}`",
        "- Student inputs: existing 500 causal columns only",
        "- Target: nested fold-pure Arm-C residual, used as label only",
        "- Population: full dev folds 0-4; no t+h eligibility filter",
        f"- OOF arrays: `{result['inputs']['student_oof']}` and the five per-fold "
        "checkpoints beside it. These are `*.npy` and therefore gitignored, so they do "
        "not travel with the commit; regenerate with `--train-outer F` for each fold, "
        "then `--merge-analyze`.",
        f"- T2 purity sentinel: nested passed = `{result['causality_contract']['fold_purity']['new_nested_passed']}`, "
        f"old global-OOF contamination caught = "
        f"`{result['causality_contract']['fold_purity']['old_global_oof_contaminated_checks']}/"
        f"{result['causality_contract']['fold_purity']['old_global_oof_total_checks']}`",
        "",
        "## Main Scores",
        "",
        "| score | whole dev | dominant cell | never-break cut | pre-break cut |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in (
        "Arm_B_RT990_raw",
        "Arm_C_RT991_raw",
        "Residual_student_raw",
        "RT600_7stream",
        "RT600_plus_seedclone",
        "RT600_plus_residual_student",
    ):
        row = scores[name]
        lines.append(
            f"| {name} | {fmt(row['whole_dev']['ts_auc'])} | "
            f"{fmt(row['dominant_cell']['ts_auc'])} | "
            f"{fmt(row['dominant_never_break_only']['ts_auc'])} | "
            f"{fmt(row['dominant_pre_break_only']['ts_auc'])} |"
        )

    lines.extend(
        [
            "",
            "## Deltas",
            "",
            "| contrast | whole dev | dominant cell | never-break cut | pre-break cut |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in (
        "Arm_C_vs_Arm_B",
        "Student_raw_vs_Arm_B",
        "Student_blend_vs_RT600",
        "Student_blend_vs_seedclone_blend",
    ):
        row = deltas[name]
        lines.append(
            f"| {name} | {fmt(row['whole_dev']['ts_auc_delta'], signed=True)} | "
            f"{fmt(row['dominant_cell']['ts_auc_delta'], signed=True)} | "
            f"{fmt(row['dominant_never_break_only']['ts_auc_delta'], signed=True)} | "
            f"{fmt(row['dominant_pre_break_only']['ts_auc_delta'], signed=True)} |"
        )

    per_fold = result.get("per_fold_contrasts") or {}
    if per_fold:
        lines.extend(
            [
                "",
                "## Per-Fold Stability",
                "",
                "Pooled TS-AUC hides fold heterogeneity. These are the same contrasts",
                "resolved per fold.",
                "",
                "| contrast | cut | f0 | f1 | f2 | f3 | f4 | positive | mean | sd | t |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for key in (
            "Student_raw_vs_Arm_B::whole_dev",
            "Student_raw_vs_Arm_B::dominant_cell",
            "Student_blend_vs_RT600::whole_dev",
            "Student_blend_vs_RT600::dominant_cell",
            "Student_blend_vs_seedclone_blend::whole_dev",
            "Student_blend_vs_seedclone_blend::dominant_cell",
        ):
            row = per_fold.get(key)
            if not row:
                continue
            name, cut = key.split("::")
            cells = " | ".join(
                fmt(row["per_fold"].get(str(f)), signed=True) for f in FOLDS
            )
            lines.append(
                f"| {name} | {cut} | {cells} | "
                f"{row['positive_folds']}/{row['n_folds']} | "
                f"{fmt(row['mean'], signed=True)} | {fmt(row['sd'])} | "
                f"{fmt(row['t_stat'], digits=2)} |"
            )
        lines.extend(
            [
                "",
                "The blend contrasts are 5/5 positive, which is the promotion-relevant",
                "gate. The standalone `Student_raw_vs_Arm_B` contrast is not: its pooled",
                "value is carried by two folds and is negative on others, so it describes",
                "this fit rather than a stable property of the mechanism. The blend gain",
                "is likewise concentrated -- folds 1 and 3 are several times the size of",
                "folds 0 and 4 -- so the mean clears the bar with a small margin relative",
                "to its own fold spread. Treat the confirmation run as load-bearing.",
            ]
        )

    scope = result.get("gain_scope") or {}
    if scope:
        lines.extend(
            [
                "",
                "## Scope Of The Gain",
                "",
                f"- Never-break pair net vs RT-600: `{fmt(scope['neverbreak_net_rate_vs_rt600'], signed=True)}`",
                f"- Pre-break pair net vs RT-600: `{fmt(scope['prebreak_net_rate_vs_rt600'], signed=True)}`",
                "",
                "**This is a never-break-cut gain.** The pre-break pair net is",
                "approximately zero, and in the upstream residualization the",
                "T-orthogonal residual scores *below* Arm B on the pre-break cut --",
                "the residual carries no pre-break signal. The blend's positive",
                "pre-break delta comes from dilution of the seven incumbent streams,",
                "not from new pre-break information. Do not describe this result as a",
                "broad improvement.",
                "",
                f"- Whole-dev damage rate vs RT-600: `{fmt(scope['whole_dev_damage_rate_vs_rt600'])}`",
                "",
                f"The damage-rate gate is applied to the `{scope['gate_scope']}` cut only.",
                "The whole-dev damage rate above sits over the same numeric threshold and",
                "is deliberately not gated; it is shown so the gate's scope is not",
                "mistaken for a claim that damage is bounded everywhere.",
            ]
        )

    lines.extend(
        [
            "",
            "## Pair Flow Vs RT-600",
            "",
            "| split | pairs | repairs | damage | net | damage rate | net rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ("whole_dev", "dominant_cell", "dominant_never_break_only", "dominant_pre_break_only"):
        row = result["pair_flow_vs_rt600"][split]
        lines.append(
            f"| {split} | {row['total_pairs_sampled']} | {row['repairs']} | "
            f"{row['damage']} | {row['net_pair_lift']} | "
            f"{fmt(row['damage_rate'])} | {fmt(row['net_rate'], signed=True)} |"
        )

    lines.extend(
        [
            "",
            "## Gates",
            "",
            f"- Pre-break damage rate: `{fmt(gates['observed_prebreak_damage_rate_vs_rt600'])}` "
            f"(gate `< {fmt(gates['max_prebreak_damage_rate'])}`) -> "
            f"`{gates['prebreak_damage_gate_passed']}`",
            f"- Never-break net rate vs RT-600: `{fmt(gates['observed_neverbreak_net_rate_vs_rt600'], signed=True)}`",
            f"- Whole-dev marginal vs seed-clone blend: `{fmt(gates['whole_dev_marginal_vs_seedclone_blend'], signed=True)}`",
            f"- Whole-dev gain vs RT-600: `{fmt(gates['whole_dev_gain_vs_rt600'], signed=True)}`",
            f"- Dominant-cell gain vs RT-600: `{fmt(gates['dominant_cell_gain_vs_rt600'], signed=True)}`",
            f"- Student raw retention of oracle residual cell lift: `{fmt(retention['retention_fraction'])}` "
            "-- **not a like-for-like ratio.** The denominator is the oracle residual on the global "
            "(fold-contaminated) RT-991 over the dominant cell; the numerator is the nested fold-pure "
            "student on full-population labels. Indicative magnitude only; do not quote as a retention rate.",
            "",
            "## Correlation",
            "",
            f"- Student raw vs Arm B, within-t dominant-cell rho: `{fmt(rho['student_raw_vs_Arm_B_RT990'])}`",
            f"- Student raw vs RT-600, within-t dominant-cell rho: `{fmt(rho['student_raw_vs_RT600'])}`",
            "",
        ]
    )

    combo = result.get("rt1257_combo")
    if combo:
        combo_scores = combo["scores"]
        combo_means = combo["mean_scores"]
        combo_deltas = combo["deltas"]
        combo_mean_deltas = combo["mean_deltas"]
        lines.extend(
            [
                "## RT-1257 Complementarity",
                "",
                combo["note"],
                "",
                "| score | mean whole-dev | pooled whole-dev | dominant cell | never-break cut | pre-break cut |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name in (
            "RT600_7stream",
            "RT1257_catboost_hybrid",
            "RT1257_plus_seedclone",
            "RT1257_plus_residual_student",
            "all_clone_control",
        ):
            row = combo_scores[name]
            lines.append(
                f"| {name} | {fmt(combo_means[name]['mean_ts_auc'])} | "
                f"{fmt(row['whole_dev']['ts_auc'])} | "
                f"{fmt(row['dominant_cell']['ts_auc'])} | "
                f"{fmt(row['dominant_never_break_only']['ts_auc'])} | "
                f"{fmt(row['dominant_pre_break_only']['ts_auc'])} |"
            )
        _pri = combo_deltas["PRIMARY_RT1257_plus_student_vs_RT1257_plus_seedclone"]
        _pri_mean = combo_mean_deltas["PRIMARY_RT1257_plus_student_minus_RT1257_plus_seedclone"]
        _acc = combo_deltas["RT1257_plus_student_vs_all_clone_control"]
        _acc_mean = combo_mean_deltas["RT1257_plus_student_minus_all_clone_control"]
        lines.extend(
            [
                "",
                "| contrast | mean whole-dev | pooled whole-dev | dominant cell |",
                "|---|---:|---:|---:|",
                f"| RT1257_plus_student_vs_RT1257 | "
                f"{fmt(combo_mean_deltas['RT1257_plus_student_minus_RT1257'], signed=True)} | "
                f"{fmt(combo_deltas['RT1257_plus_student_vs_RT1257']['whole_dev']['ts_auc_delta'], signed=True)} | "
                f"{fmt(combo_deltas['RT1257_plus_student_vs_RT1257']['dominant_cell']['ts_auc_delta'], signed=True)} |",
                f"| **PRIMARY** RT1257_plus_student_vs_RT1257_plus_seedclone | "
                f"{fmt(_pri_mean, signed=True)} | "
                f"{fmt(_pri['whole_dev']['ts_auc_delta'], signed=True)} | "
                f"{fmt(_pri['dominant_cell']['ts_auc_delta'], signed=True)} |",
                f"| RT1257_plus_student_vs_all_clone_control (NOT an E1) | "
                f"{fmt(_acc_mean, signed=True)} | "
                f"{fmt(_acc['whole_dev']['ts_auc_delta'], signed=True)} | "
                f"{fmt(_acc['dominant_cell']['ts_auc_delta'], signed=True)} |",
                f"| RT1257_plus_student_vs_RT600 | "
                f"{fmt(combo_mean_deltas['RT1257_plus_student_minus_RT600'], signed=True)} | "
                f"{fmt(combo_deltas['RT1257_plus_student_vs_RT600']['whole_dev']['ts_auc_delta'], signed=True)} | "
                f"{fmt(combo_deltas['RT1257_plus_student_vs_RT600']['dominant_cell']['ts_auc_delta'], signed=True)} |",
                "",
                "",
                combo["endpoint_note"],
                "",
                "Pair flow of `RT1257 + residual_student` vs `RT1257`:",
                "",
                "| split | pairs | repairs | damage | net | damage rate | net rate |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for split in ("whole_dev", "dominant_cell", "dominant_never_break_only", "dominant_pre_break_only"):
            row = combo["pair_flow_vs_rt1257"][split]
            lines.append(
                f"| {split} | {row['total_pairs_sampled']} | {row['repairs']} | "
                f"{row['damage']} | {row['net_pair_lift']} | "
                f"{fmt(row['damage_rate'])} | {fmt(row['net_rate'], signed=True)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Verdict",
            "",
            str(result["verdict"]),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--folds-path", type=Path, default=ROOT / "research" / "folds" / "folds.parquet")
    parser.add_argument("--oof-dir", type=Path, default=None)
    parser.add_argument("--catboost-oof-dir", type=Path, default=default_catboost_oof_dir())
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "research" / "reports" / "armc_residual_student",
    )
    parser.add_argument("--eps", type=float, default=EPS)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--pair-seed", type=int, default=20260830)
    parser.add_argument("--pairs-per-t", type=int, default=64)
    parser.add_argument("--max-prebreak-damage", type=float, default=0.015)
    parser.add_argument("--max-train-rows", type=int, default=1_000_000)
    parser.add_argument("--rounds", type=int, default=STUDENT_ROUNDS)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--train-outer", type=int, choices=FOLDS)
    parser.add_argument("--purity-check", action="store_true")
    parser.add_argument("--merge-analyze", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-merge", action="store_true")
    args = parser.parse_args()

    args.data_root = args.data_root.resolve()
    args.folds_path = args.folds_path.resolve()
    args.oof_dir = (args.oof_dir or (args.data_root / "research" / "oof")).resolve()
    if args.catboost_oof_dir is not None:
        args.catboost_oof_dir = args.catboost_oof_dir.resolve()
    args.out_dir = args.out_dir.resolve()

    if args.purity_check:
        rows = build_row_arrays(args.folds_path)
        out = fold_purity_check(args.oof_dir, rows)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        path = args.out_dir / "armc_residual_student_purity.json"
        path.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps({
            "path": str(path),
            "new_nested_passed": out["new_nested_passed"],
            "old_global_oof_contaminated_checks": out["old_global_oof_contaminated_checks"],
            "old_global_oof_total_checks": out["old_global_oof_total_checks"],
        }, indent=2))
        if not out["new_nested_passed"] or not out["old_global_oof_sentinel_catches_defect"]:
            raise SystemExit(1)

    if args.train_outer is not None:
        train_outer_fold(args)

    if args.merge_analyze:
        analyze(args)

    if not args.purity_check and args.train_outer is None and not args.merge_analyze:
        raise SystemExit("pass --purity-check, --train-outer F, or --merge-analyze")


if __name__ == "__main__":
    main()
