"""Crunch cloud final-10k training job for RT-1320.

RESEARCH / ARTIFACT-BUILD JOB ONLY - NOT A LEADERBOARD CANDIDATE.

This submission uses the Crunch training environment to regenerate the
fold-pure Arm-C residual target on ``folds_final10k`` and fit the RT-1320
student model. It writes artifacts under ``model_directory_path`` for download
and local assembly into an 8-member RT-1257+RT-1320 production artifact.

``infer`` deliberately returns deterministic dummy probabilities. Do not read
the leaderboard score from this job as a model result.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NUM_THREADS = max(1, int(os.environ.get("SBR_RT1320_NUM_THREADS", "16")))

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_k, str(NUM_THREADS))

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT / "src", _REPO_ROOT / "research" / "scripts"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

# @crunch/keep:on
INFER_PARALLELISM = 1

LABEL = "RT-1320 FINAL10K FIT - RESEARCH ARTIFACT BUILD ONLY"
SMOKE_ENV = "SBR_RT1320_FINAL10K_SMOKE"
KEEP_ARTIFACT_ENV = "SBR_RT1320_KEEP_ARTIFACT_ROOT"
OUTPUT_NAME = "rt1320_final10k_fit"

FOLDS = (0, 1, 2, 3, 4)
FULL_MODULES = (
    "m00_core",
    "m01_seq",
    "m02_dist",
    "m03_dyn",
    "m04_resid",
    "m06_loc",
    "m07_bayes",
)
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

SEED = 20260901
TEACHER_MAX_TRAIN_ROWS = 1_000_000
STUDENT_OOF_MAX_TRAIN_ROWS = 1_250_000
STUDENT_FINAL_MAX_TRAIN_ROWS = 1_250_000
TEACHER_ROUNDS = 900
STUDENT_ROUNDS = 900
FEATURE_WORKERS = int(os.environ.get("SBR_RT1320_FEATURE_WORKERS", "16"))
FEATURE_CHUNK = int(os.environ.get("SBR_RT1320_FEATURE_CHUNK", "64"))
EPS = 1e-6

RT600_FEATURE_MANIFEST_SHA256 = (
    "1646c3b9e09d8a7fb3c564483a1d1d999680caeefe848b092f907a6cac80cced"
)
FOLDS_FINAL10K_SHA256 = (
    "bf0cdf642bde018a632663ae7d211714a173fb64ef824b15416caf2649e0c716"
)

TEACHER_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 150,
    "feature_fraction": 1.0,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "max_bin": 127,
    "num_threads": NUM_THREADS,
    "verbose": -1,
}
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
    "num_threads": NUM_THREADS,
    "verbose": -1,
}


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _series_values(obj: Any) -> Any:
    import numpy as np

    if hasattr(obj, "columns") and "value" in obj.columns:
        return obj["value"].to_numpy(dtype=np.float32)
    if isinstance(obj, dict) and "value" in obj:
        return np.asarray(obj["value"], dtype=np.float32).reshape(-1)
    return np.asarray(obj, dtype=np.float32).reshape(-1)


def _normalise_tau(value: Any, n_online: int) -> int:
    import numpy as np

    if value is None:
        return -1
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return -1
    tau = int(value)
    if tau < -1 or tau >= n_online:
        raise RuntimeError(f"Invalid tau_index={tau}; expected -1 or [0, {n_online})")
    return tau


def _package_versions() -> dict[str, str]:
    import importlib.metadata as md
    import platform

    packages = ("numpy", "pandas", "pyarrow", "lightgbm", "scipy", "scikit-learn")
    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for package in packages:
        try:
            out[package] = md.version(package)
        except Exception:
            out[package] = "missing"
    return out


def _artifact_contract() -> dict[str, Any]:
    return {
        "experiment_id": "RT-1320",
        "what": "RT-1257 plus one Arm-C residual-student member",
        "member_contract": "addition of an 8th exchangeable member to RT-1257",
        "student": {
            "slot": 7,
            "id": "RT-1320",
            "kind": "lightgbm",
            "objective": "regression",
            "model_file": "model.txt.7",
            "calibration_file": "RT-1320_student_scdf.json",
            "input_features": "same 500 causal columns as RT-1257",
            "target": "fold-pure Arm-C teacher residual on folds_final10k",
        },
        "feature_bank": {
            "modules": list(FULL_MODULES),
            "n_columns": 500,
            "feature_manifest_sha256": RT600_FEATURE_MANIFEST_SHA256,
        },
        "expected_provenance": {
            "n_series": 10000,
            "partition": "folds_final10k",
            "folds_sha256": FOLDS_FINAL10K_SHA256,
            "n_boosters_after_local_assembly": 8,
            "calibration_kind": "scdf",
            "calibration_time_coord": "log_n_seen",
        },
    }


def _checkpoint_contract() -> dict[str, Any]:
    return {
        "schema": "sbr.rt1320_final10k_fit.checkpoints/1",
        "folds": list(FOLDS),
        "folds_final10k_sha256": FOLDS_FINAL10K_SHA256,
        "feature_modules": list(FULL_MODULES),
        "teacher_rounds": TEACHER_ROUNDS,
        "student_rounds": STUDENT_ROUNDS,
        "teacher_max_train_rows": TEACHER_MAX_TRAIN_ROWS,
        "student_oof_max_train_rows": STUDENT_OOF_MAX_TRAIN_ROWS,
        "student_final_max_train_rows": STUDENT_FINAL_MAX_TRAIN_ROWS,
        "lightgbm_num_threads": NUM_THREADS,
        "seed": SEED,
    }


def _prepare_output_root(model_directory_path: str) -> Path:
    out_root = Path(model_directory_path).resolve() / OUTPUT_NAME
    stamp_payload = _checkpoint_contract()
    stamp = out_root / "CHECKPOINT_PROVENANCE.json"
    if stamp.exists():
        try:
            existing = json.loads(stamp.read_text(encoding="utf-8"))
        except Exception:
            existing = None
        if existing != stamp_payload:
            quarantine_parent = Path(
                tempfile.mkdtemp(prefix=f"{out_root.name}_stale_{int(time.time())}_")
            ).resolve()
            quarantine = quarantine_parent / out_root.name
            print(
                "REFUSING TO REUSE RT-1320 CHECKPOINTS: provenance mismatch. "
                f"Quarantining {out_root} -> {quarantine}.",
                flush=True,
            )
            shutil.move(str(out_root), str(quarantine))
    out_root.mkdir(parents=True, exist_ok=True)
    _dump_json(stamp, stamp_payload)
    return out_root


def _fold_purity_report() -> dict[str, Any]:
    new_failures: list[dict[str, Any]] = []
    old_contaminated = 0
    total = 0
    for outer_fold in FOLDS:
        outer_train = [fold for fold in FOLDS if fold != outer_fold]
        for inner_fold in outer_train:
            total += 1
            new_train = set(FOLDS) - {outer_fold, inner_fold}
            if new_train & {outer_fold, inner_fold}:
                new_failures.append(
                    {
                        "outer_fold": int(outer_fold),
                        "inner_fold": int(inner_fold),
                        "teacher_training_folds": sorted(int(x) for x in new_train),
                    }
                )
            old_train = set(FOLDS) - {inner_fold}
            if old_train & {outer_fold, inner_fold}:
                old_contaminated += 1
    return {
        "scope": (
            "static fold-index plan only; actual saved-array leakage is checked "
            "by _load_nested_q"
        ),
        "actual_prediction_gate": "_load_nested_q",
        "checked_outer_inner_pairs": int(total),
        "new_nested_failures": new_failures,
        "new_nested_passed": len(new_failures) == 0,
        "old_global_oof_contaminated_checks": int(old_contaminated),
        "old_global_oof_total_checks": int(total),
        "old_global_oof_sentinel_catches_defect": old_contaminated == total,
    }


def _materialize_final10k_store(
    datasets: Iterable[tuple[int, Any, Any, Any]],
    artifact_root: Path,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    folds_src = _REPO_ROOT / "research" / "folds" / "folds_final10k.parquet"
    folds_sha = _sha256_file(folds_src)
    if folds_sha != FOLDS_FINAL10K_SHA256:
        raise RuntimeError(f"folds_final10k sha mismatch: {folds_sha}")

    final_folds = pd.read_parquet(folds_src).sort_values("id").reset_index(drop=True)
    if len(final_folds) != 10000:
        raise RuntimeError(f"folds_final10k must contain 10,000 series; got {len(final_folds)}")
    if set(int(x) for x in final_folds["fold"].unique()) != set(FOLDS):
        raise RuntimeError("folds_final10k must contain only folds 0..4")
    ids = final_folds["id"].to_numpy()
    if not np.array_equal(ids, np.arange(len(final_folds))):
        raise RuntimeError("folds_final10k ids must be contiguous and sorted 0..9999")

    selected_ids = set(int(x) for x in ids)
    seen: dict[int, tuple[Any, Any, int]] = {}
    duplicate_ids = 0
    n_total_items = 0
    for item in datasets:
        n_total_items += 1
        if len(item) != 4:
            raise RuntimeError(
                "Expected Crunch real-time train item "
                "(id, x_historical, x_online, tau_index)"
            )
        sid_raw, x_historical, x_online, tau_raw = item
        sid = int(sid_raw)
        if sid not in selected_ids:
            continue
        if sid in seen:
            duplicate_ids += 1
        hist = _series_values(x_historical)
        online = _series_values(x_online)
        tau = _normalise_tau(tau_raw, len(online))
        seen[sid] = (hist, online, tau)

    missing = [int(x) for x in ids if int(x) not in seen]
    if missing:
        raise RuntimeError(
            "Crunch train data is missing folds_final10k ids; "
            f"first missing ids={missing[:10]} total_missing={len(missing)}"
        )

    store_dir = artifact_root / "cache" / "store"
    folds_dir = artifact_root / "research" / "folds"
    store_dir.mkdir(parents=True, exist_ok=True)
    folds_dir.mkdir(parents=True, exist_ok=True)

    total_values = int(
        sum(len(seen[int(sid)][0]) + len(seen[int(sid)][1]) for sid in ids)
    )
    values = np.lib.format.open_memmap(
        store_dir / "values.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_values,),
    )

    rows = []
    pos = 0
    for sid_raw in ids:
        sid = int(sid_raw)
        hist, online, tau = seen[sid]
        n_hist = int(len(hist))
        n_online = int(len(online))
        values[pos : pos + n_hist] = hist
        values[pos + n_hist : pos + n_hist + n_online] = online
        rows.append((sid, pos, n_hist, n_online, tau, int(tau >= 0)))
        pos += n_hist + n_online
    values.flush()

    meta = pd.DataFrame(
        rows,
        columns=["id", "off", "n_hist", "n_online", "tau_index", "has_break"],
    )
    meta.to_parquet(store_dir / "meta.parquet", index=False)

    fold_cols = [
        col
        for col in final_folds.columns
        if col not in {"n_hist", "n_online", "tau_index", "has_break"}
    ]
    folds_out = final_folds[fold_cols].merge(
        meta[["id", "n_hist", "n_online", "tau_index", "has_break"]],
        on="id",
        how="left",
    )
    folds_out.to_parquet(folds_dir / "folds_final10k.parquet", index=False)

    return {
        "n_train_items_seen": int(n_total_items),
        "duplicate_selected_ids_seen": int(duplicate_ids),
        "store_dir": str(store_dir),
        "folds_path": str(folds_dir / "folds_final10k.parquet"),
        "folds_final10k_source_sha256": folds_sha,
        "selected_folds": list(FOLDS),
        "selected_series": int(len(meta)),
        "selected_online_rows": int(meta["n_online"].sum()),
        "selected_total_values": int(total_values),
    }


def _configure_runtime(artifact_root: Path):
    os.environ["SBR_ROOT"] = str(artifact_root)
    os.environ["SBR_STORE"] = str(artifact_root / "cache" / "store")
    os.environ["SBR_FEATURES"] = str(artifact_root / "cache" / "features")

    import sbr.pipeline as PL

    PL.ROOT = str(artifact_root)
    PL.FEAT = str(artifact_root / "cache" / "features")
    PL.FOLDS = str(artifact_root / "research" / "folds" / "folds_final10k.parquet")
    PL.STORE = str(artifact_root / "cache" / "store")
    PL.OOF = str(artifact_root / "research" / "oof")
    return PL


def _build_feature_cache(artifact_root: Path) -> dict[str, Any]:
    import numpy as np

    from sbr.features.driver import build_features
    from sbr.stream.engine import StreamEngine

    features_dir = artifact_root / "cache" / "features"
    t0 = time.time()
    manifest = build_features(
        list(FULL_MODULES),
        store=str(artifact_root / "cache" / "store"),
        out=str(features_dir),
        workers=FEATURE_WORKERS,
        chunk=FEATURE_CHUNK,
        limit=0,
    )
    feature_count = int(sum(v["columns"] for v in manifest["modules"].values()))
    if feature_count != 500:
        raise RuntimeError(f"Expected 500 causal features, built {feature_count}")

    engine = StreamEngine(FULL_MODULES).fit_historical(
        np.arange(1200, dtype=np.float64) % 7 - 3.0
    )
    engine_manifest = engine.manifest()
    stream_sha = engine_manifest.get("feature_manifest_sha256")
    if stream_sha != RT600_FEATURE_MANIFEST_SHA256:
        raise RuntimeError(f"feature manifest sha mismatch: {stream_sha}")

    return {
        **manifest,
        "wall_seconds": float(time.time() - t0),
        "feature_modules": list(FULL_MODULES),
        "feature_count": feature_count,
        "stream_feature_manifest_sha256": stream_sha,
    }


def _load_features(PL) -> tuple[list[Any], list[str], Any]:
    import numpy as np

    mats, names = PL.load_features(FULL_MODULES)
    bad = [name for name in names if any(tok in name.lower() for tok in FORBIDDEN_TOKENS)]
    if bad:
        raise RuntimeError(f"forbidden student feature columns are reachable: {bad[:20]}")
    if len(names) != 500:
        raise RuntimeError(f"expected 500 causal columns, found {len(names)}")
    keep_idx = np.arange(len(names), dtype=np.int64)
    return mats, names, keep_idx


def _row_arrays(d) -> dict[str, Any]:
    import numpy as np

    tau = d.folds["tau_index"].to_numpy(dtype=np.int32)
    has_break = d.folds["has_break"].to_numpy(dtype=bool)
    n_online = d.st.meta.n_online.to_numpy(dtype=np.int32)
    tau_by_row = tau[d.sidx]
    age = np.where(d.y == 1, d.t - tau_by_row, -1).astype(np.int32)
    has_break_by_row = has_break[d.sidx]
    dev = np.isin(d.row_fold, FOLDS)
    neg_pre = has_break_by_row & (d.y == 0)
    dominant = dev & (d.t >= 200) & (((d.y == 1) & (age >= 100)) | (d.y == 0))
    return {
        "sidx": d.sidx,
        "t": d.t,
        "y": d.y,
        "age": age,
        "row_fold": d.row_fold,
        "dev": dev,
        "has_break_by_row": has_break_by_row,
        "n_online_by_row": n_online[d.sidx],
        "dominant_cell": dominant,
        "dominant_never_break_only": dominant & ((d.y == 1) | (~neg_pre)),
        "dominant_pre_break_only": dominant & ((d.y == 1) | neg_pre),
    }


def _last_row_lookup(d):
    import numpy as np

    sidx = d.sidx
    n_series = int(sidx.max()) + 1
    order = np.argsort(sidx, kind="stable")
    s_sorted = sidx[order]
    starts = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])
    ends = np.r_[starts[1:], len(s_sorted)]
    last_row = np.zeros(n_series, dtype=np.int64)
    for lo, hi in zip(starts, ends):
        group = order[lo:hi]
        last_row[s_sorted[lo]] = group[np.argmax(d.t[group])]
    return last_row


def _augmented_stack(PL, mats, names, keep_idx, rows, final_of_row):
    import numpy as np

    own = PL._stack(mats, names, rows, keep_idx)
    final_rows = final_of_row[rows]
    uniq, inv = np.unique(final_rows, return_inverse=True)
    fut_u = PL._stack(mats, names, uniq, keep_idx)
    if own.shape[1] != fut_u.shape[1]:
        raise RuntimeError(
            f"teacher feature width mismatch: own={own.shape[1]} final={fut_u.shape[1]}"
        )
    n_cols = int(own.shape[1])
    out = np.empty((own.shape[0], n_cols * 2), dtype=np.result_type(own.dtype, fut_u.dtype))
    out[:, :n_cols] = own
    del own
    right = out[:, n_cols:]
    # Avoid a second full rows x features gather before LightGBM allocates its Dataset.
    for start in range(0, len(inv), 100_000):
        end = min(start + 100_000, len(inv))
        right[start:end] = fut_u[inv[start:end]]
    return out


def _sample_rows(rows, budget: int, seed: int):
    import numpy as np

    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) <= budget:
        return rows
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(rows, budget, replace=False))


def _train_lgb(params: dict[str, Any], x_train, y_train, rounds: int):
    import lightgbm as lgb

    dataset = lgb.Dataset(
        x_train,
        label=y_train,
        params=params,
        feature_name=[f"f{i}" for i in range(x_train.shape[1])],
    )
    return lgb.train(params, dataset, num_boost_round=int(rounds))


def _predict_rows(PL, booster, mats, names, rows, keep_idx, final_of_row=None, chunk_rows=150_000):
    import numpy as np

    rows = np.asarray(rows, dtype=np.int64)
    out = np.empty(len(rows), dtype=np.float32)
    for start in range(0, len(rows), int(chunk_rows)):
        end = min(start + int(chunk_rows), len(rows))
        rr = rows[start:end]
        if final_of_row is None:
            x_pred = PL._stack(mats, names, rr, keep_idx)
        else:
            x_pred = _augmented_stack(PL, mats, names, keep_idx, rr, final_of_row)
        out[start:end] = booster.predict(x_pred).astype(np.float32)
        del x_pred
        gc.collect()
    return out


def _validate_partial_vector(path: Path, row_count: int, target_rows, forbidden_rows) -> bool:
    import numpy as np

    if not path.exists():
        return False
    part = np.load(path, mmap_mode="r")
    if len(part) != row_count:
        return False
    if not np.isfinite(part[target_rows]).all():
        return False
    if np.isfinite(part[forbidden_rows]).any():
        return False
    return True


def _train_teacher_checkpoint(
    PL,
    d,
    mats,
    names,
    keep_idx,
    final_of_row,
    out_root: Path,
    outer_fold: int,
    inner_fold: int,
) -> dict[str, Any]:
    import numpy as np

    npy_path = out_root / f"nested_Q_outer{outer_fold}_inner{inner_fold}.npy"
    json_path = out_root / f"nested_Q_outer{outer_fold}_inner{inner_fold}.json"
    target_rows = d.rows_for([inner_fold])
    forbidden_rows = d.rows_for([outer_fold])
    if json_path.exists() and _validate_partial_vector(
        npy_path, len(d.y), target_rows, forbidden_rows
    ):
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        meta["status"] = "reused"
        return meta

    train_folds = [fold for fold in FOLDS if fold not in (outer_fold, inner_fold)]
    if set(train_folds) & {outer_fold, inner_fold}:
        raise RuntimeError("teacher fold-purity violation")
    train_rows = _sample_rows(
        d.rows_for(train_folds),
        TEACHER_MAX_TRAIN_ROWS,
        SEED + 100 * int(outer_fold) + int(inner_fold),
    )

    t0 = time.time()
    params = dict(TEACHER_PARAMS)
    x_train = _augmented_stack(PL, mats, names, keep_idx, train_rows, final_of_row)
    if x_train.shape[1] != 1000:
        raise RuntimeError(f"expected 1000 teacher columns, got {x_train.shape[1]}")
    y_train = d.y[train_rows]
    booster = _train_lgb(params, x_train, y_train, TEACHER_ROUNDS)
    del x_train, y_train
    gc.collect()

    pred = np.clip(
        _predict_rows(PL, booster, mats, names, target_rows, keep_idx, final_of_row),
        EPS,
        1.0 - EPS,
    )
    del booster
    gc.collect()

    part = np.full(len(d.y), np.nan, dtype=np.float32)
    part[target_rows] = pred
    np.save(npy_path, part)
    del part

    meta = {
        "kind": "nested_teacher_q",
        "outer_fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "teacher_training_folds": [int(x) for x in train_folds],
        "target_fold": int(inner_fold),
        "train_rows": int(len(train_rows)),
        "target_rows": int(len(target_rows)),
        "params": params,
        "rounds": int(TEACHER_ROUNDS),
        "q_mean": float(np.mean(pred)),
        "q_min": float(np.min(pred)),
        "q_max": float(np.max(pred)),
        "path": str(npy_path),
        "sha256": _sha256_file(npy_path),
        "runtime_s": round(time.time() - t0, 1),
        "score_computed": False,
        "status": "trained",
    }
    _dump_json(json_path, meta)
    print(
        "nested teacher "
        f"outer={outer_fold} inner={inner_fold}: {len(train_rows)} train rows, "
        f"{len(target_rows)} target rows, {meta['runtime_s']}s",
        flush=True,
    )
    return meta


def _train_final_teacher_checkpoint(
    PL,
    d,
    mats,
    names,
    keep_idx,
    final_of_row,
    out_root: Path,
    fold: int,
) -> dict[str, Any]:
    import numpy as np

    npy_path = out_root / f"final_Q_inner{fold}.npy"
    json_path = out_root / f"final_Q_inner{fold}.json"
    target_rows = d.rows_for([fold])
    forbidden_rows = np.flatnonzero(d.row_fold != fold)
    if json_path.exists() and _validate_partial_vector(
        npy_path, len(d.y), target_rows, forbidden_rows
    ):
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        meta["status"] = "reused"
        return meta

    train_folds = [x for x in FOLDS if x != fold]
    train_rows = _sample_rows(
        d.rows_for(train_folds),
        TEACHER_MAX_TRAIN_ROWS,
        SEED + 1000 + int(fold),
    )

    t0 = time.time()
    params = dict(TEACHER_PARAMS)
    x_train = _augmented_stack(PL, mats, names, keep_idx, train_rows, final_of_row)
    if x_train.shape[1] != 1000:
        raise RuntimeError(f"expected 1000 teacher columns, got {x_train.shape[1]}")
    y_train = d.y[train_rows]
    booster = _train_lgb(params, x_train, y_train, TEACHER_ROUNDS)
    del x_train, y_train
    gc.collect()

    pred = np.clip(
        _predict_rows(PL, booster, mats, names, target_rows, keep_idx, final_of_row),
        EPS,
        1.0 - EPS,
    )
    del booster
    gc.collect()

    part = np.full(len(d.y), np.nan, dtype=np.float32)
    part[target_rows] = pred
    np.save(npy_path, part)
    del part

    meta = {
        "kind": "final_teacher_q",
        "fold": int(fold),
        "teacher_training_folds": [int(x) for x in train_folds],
        "target_fold": int(fold),
        "train_rows": int(len(train_rows)),
        "target_rows": int(len(target_rows)),
        "params": params,
        "rounds": int(TEACHER_ROUNDS),
        "q_mean": float(np.mean(pred)),
        "q_min": float(np.min(pred)),
        "q_max": float(np.max(pred)),
        "path": str(npy_path),
        "sha256": _sha256_file(npy_path),
        "runtime_s": round(time.time() - t0, 1),
        "score_computed": False,
        "status": "trained",
    }
    _dump_json(json_path, meta)
    print(
        f"final teacher target_fold={fold}: {len(train_rows)} train rows, "
        f"{len(target_rows)} target rows, {meta['runtime_s']}s",
        flush=True,
    )
    return meta


def _logit(p):
    import numpy as np

    q = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


def _group_indices_by_t(t, mask) -> list[Any]:
    import numpy as np

    idx = np.flatnonzero(mask)
    order = np.argsort(t[idx], kind="stable")
    idx = idx[order]
    tt = t[idx]
    starts = np.flatnonzero(np.r_[True, tt[1:] != tt[:-1]])
    ends = np.r_[starts[1:], len(idx)]
    return [idx[lo:hi] for lo, hi in zip(starts, ends)]


def _design_matrix(t_value: int, n_online):
    import numpy as np

    n_online = np.asarray(n_online, dtype=np.float64)
    remaining = n_online - float(t_value)
    frac = float(t_value) / np.maximum(n_online, 1.0)
    return np.column_stack([np.ones(len(n_online)), n_online, remaining, frac])


def _fit_project_predict(y_fit, x_fit, x_pred):
    import numpy as np

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


def _projection_residual(score, d, *, base_folds: Iterable[int]) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    base_folds = tuple(int(x) for x in base_folds)
    base_mask = np.isin(d.row_fold, base_folds) & np.isfinite(score)
    groups = _group_indices_by_t(d.t, base_mask)
    horizon = np.full(score.shape, np.nan, dtype=np.float64)
    fallback_groups = 0
    fitted_groups = 0
    skipped_predictions = 0

    for group in groups:
        t_value = int(d.t[group[0]])
        group_folds = np.unique(d.row_fold[group])
        for pred_fold in group_folds:
            pred_idx = group[d.row_fold[group] == pred_fold]
            fit_idx = group[d.row_fold[group] != pred_fold]
            if len(fit_idx) == 0:
                skipped_predictions += int(len(pred_idx))
                continue
            pred, fallback = _fit_project_predict(
                score[fit_idx],
                _design_matrix(t_value, d.st.meta.n_online.to_numpy()[d.sidx[fit_idx]]),
                _design_matrix(t_value, d.st.meta.n_online.to_numpy()[d.sidx[pred_idx]]),
            )
            horizon[pred_idx] = pred
            fallback_groups += int(fallback)
            fitted_groups += 1

    residual = score - horizon
    target_rows = np.flatnonzero(base_mask)
    return residual, {
        "base_folds": [int(x) for x in base_folds],
        "t_groups": int(len(groups)),
        "fitted_fold_groups": int(fitted_groups),
        "fallback_fold_groups": int(fallback_groups),
        "skipped_predictions": int(skipped_predictions),
        "target_rows": int(np.isfinite(residual[target_rows]).sum()),
    }


def _load_nested_q(out_root: Path, d, outer_fold: int):
    import numpy as np

    q = np.full(len(d.y), np.nan, dtype=np.float64)
    parts = []
    for inner_fold in FOLDS:
        if inner_fold == outer_fold:
            continue
        path = out_root / f"nested_Q_outer{outer_fold}_inner{inner_fold}.npy"
        part = np.load(path, mmap_mode="r")
        target_rows = d.row_fold == inner_fold
        outer_rows = d.row_fold == outer_fold
        if not np.isfinite(part[target_rows]).all():
            raise RuntimeError(f"{path} does not cover target fold {inner_fold}")
        if np.isfinite(part[outer_rows]).any():
            raise RuntimeError(f"{path} has finite labels on held-out outer fold {outer_fold}")
        q[target_rows] = part[target_rows]
        parts.append(
            {
                "inner_fold": int(inner_fold),
                "path": str(path),
                "sha256": _sha256_file(path),
            }
        )
    train_rows = np.isin(d.row_fold, [f for f in FOLDS if f != outer_fold])
    if not np.isfinite(q[train_rows]).all():
        raise RuntimeError(f"nested Q incomplete for outer fold {outer_fold}")
    return np.clip(q, EPS, 1.0 - EPS), parts


def _load_final_q(out_root: Path, d):
    import numpy as np

    q = np.full(len(d.y), np.nan, dtype=np.float64)
    parts = []
    for fold in FOLDS:
        path = out_root / f"final_Q_inner{fold}.npy"
        part = np.load(path, mmap_mode="r")
        target_rows = d.row_fold == fold
        if not np.isfinite(part[target_rows]).all():
            raise RuntimeError(f"{path} does not cover final target fold {fold}")
        q[target_rows] = part[target_rows]
        parts.append({"fold": int(fold), "path": str(path), "sha256": _sha256_file(path)})
    all_rows = d.rows_for(FOLDS)
    if not np.isfinite(q[all_rows]).all():
        raise RuntimeError("final Q target is incomplete")
    return np.clip(q, EPS, 1.0 - EPS), parts


def _target_stats(values, rows) -> dict[str, Any]:
    import numpy as np

    vv = np.asarray(values[rows], dtype=np.float64)
    vv = vv[np.isfinite(vv)]
    return {
        "n": int(len(vv)),
        "mean": float(np.mean(vv)),
        "std": float(np.std(vv)),
        "q01": float(np.quantile(vv, 0.01)),
        "q50": float(np.quantile(vv, 0.50)),
        "q99": float(np.quantile(vv, 0.99)),
    }


def _train_student_oof_fold(PL, d, mats, names, keep_idx, out_root: Path, outer_fold: int):
    import numpy as np

    npy_path = out_root / f"RT-1320_final10k_outer{outer_fold}.npy"
    json_path = out_root / f"RT-1320_final10k_outer{outer_fold}.json"
    model_path = out_root / f"RT-1320_student_outer{outer_fold}.txt"
    valid_rows = d.rows_for([outer_fold])
    forbidden_rows = np.flatnonzero(d.row_fold != outer_fold)
    if (
        json_path.exists()
        and model_path.exists()
        and _validate_partial_vector(npy_path, len(d.y), valid_rows, forbidden_rows)
    ):
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        meta["status"] = "reused"
        return meta

    q, teacher_parts = _load_nested_q(out_root, d, outer_fold)
    residual, residual_meta = _projection_residual(
        _logit(q),
        d,
        base_folds=[f for f in FOLDS if f != outer_fold],
    )
    train_rows = d.rows_for([f for f in FOLDS if f != outer_fold])
    train_rows = train_rows[np.isfinite(residual[train_rows])]
    sampled_rows = _sample_rows(
        train_rows,
        STUDENT_OOF_MAX_TRAIN_ROWS,
        SEED + 2000 + int(outer_fold),
    )

    t0 = time.time()
    params = dict(STUDENT_PARAMS)
    x_train = PL._stack(mats, names, sampled_rows, keep_idx)
    if x_train.shape[1] != 500:
        raise RuntimeError(f"expected 500 student columns, got {x_train.shape[1]}")
    y_train = residual[sampled_rows].astype(np.float32)
    booster = _train_lgb(params, x_train, y_train, STUDENT_ROUNDS)
    booster.save_model(str(model_path))
    del x_train, y_train
    gc.collect()

    pred = _predict_rows(PL, booster, mats, names, valid_rows, keep_idx)
    del booster
    gc.collect()

    part = np.full(len(d.y), np.nan, dtype=np.float32)
    part[valid_rows] = pred
    np.save(npy_path, part)
    del part

    meta = {
        "kind": "student_oof_fold",
        "outer_fold": int(outer_fold),
        "teacher_parts": teacher_parts,
        "residual": {
            **residual_meta,
            "target_stats": _target_stats(residual, train_rows),
        },
        "train_rows_available": int(len(train_rows)),
        "train_rows_sampled": int(len(sampled_rows)),
        "valid_rows": int(len(valid_rows)),
        "params": params,
        "rounds": int(STUDENT_ROUNDS),
        "prediction_path": str(npy_path),
        "prediction_sha256": _sha256_file(npy_path),
        "model_path": str(model_path),
        "model_sha256": _sha256_file(model_path),
        "runtime_s": round(time.time() - t0, 1),
        "score_computed": False,
        "status": "trained",
    }
    _dump_json(json_path, meta)
    print(
        f"student OOF outer={outer_fold}: {len(sampled_rows)} train rows, "
        f"{len(valid_rows)} valid rows, {meta['runtime_s']}s",
        flush=True,
    )
    return meta


def _assemble_student_oof(out_root: Path, d) -> dict[str, Any]:
    import numpy as np

    merged_path = out_root / "RT-1320_final10k_oof.npy"
    oof = np.full(len(d.y), np.nan, dtype=np.float32)
    parts = []
    for fold in FOLDS:
        path = out_root / f"RT-1320_final10k_outer{fold}.npy"
        part = np.load(path, mmap_mode="r")
        valid_rows = d.row_fold == fold
        if not np.isfinite(part[valid_rows]).all():
            raise RuntimeError(f"{path} does not cover OOF fold {fold}")
        oof[valid_rows] = part[valid_rows]
        parts.append({"fold": int(fold), "path": str(path), "sha256": _sha256_file(path)})
    all_rows = d.rows_for(FOLDS)
    if not np.isfinite(oof[all_rows]).all():
        raise RuntimeError("RT-1320 OOF vector incomplete")
    np.save(merged_path, oof)
    return {
        "path": str(merged_path),
        "sha256": _sha256_file(merged_path),
        "parts": parts,
        "finite_rows": int(np.isfinite(oof[all_rows]).sum()),
        "score_computed": False,
    }


def _fit_student_calibration(out_root: Path, d) -> dict[str, Any]:
    import numpy as np

    from sbr.production.calibration import DEFAULT_COORD, SmoothTimeCDFCal

    oof_path = out_root / "RT-1320_final10k_oof.npy"
    payload_path = out_root / "RT-1320_student_scdf.json"
    oof = np.load(oof_path, mmap_mode="r")
    all_rows = d.rows_for(FOLDS)
    cal = SmoothTimeCDFCal.fit(oof[all_rows], d.t[all_rows], time_coord=DEFAULT_COORD)
    payload = {
        "kind": "scdf",
        "time_coord": DEFAULT_COORD,
        "source_oof": str(oof_path),
        "source_oof_sha256": _sha256_file(oof_path),
        "model": cal.to_json(),
        "fitted_rows": int(len(all_rows)),
        "score_computed": False,
    }
    _dump_json(payload_path, payload)
    return {"path": str(payload_path), "sha256": _sha256_file(payload_path), **payload}


def _train_final_student(PL, d, mats, names, keep_idx, out_root: Path) -> dict[str, Any]:
    import lightgbm as lgb
    import numpy as np

    model_path = out_root / "model.txt.7"
    json_path = out_root / "RT-1320_final_student.json"
    if model_path.exists() and json_path.exists():
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        current_q, current_teacher_parts = _load_final_q(out_root, d)
        del current_q
        residual_path = out_root / "RT-1320_final10k_final_residual.npy"
        expected_teacher_parts = [
            {"fold": int(part["fold"]), "sha256": part["sha256"]}
            for part in current_teacher_parts
        ]
        found_teacher_parts = [
            {"fold": int(part["fold"]), "sha256": part["sha256"]}
            for part in meta.get("teacher_parts", [])
            if "fold" in part and "sha256" in part
        ]
        residual_meta = meta.get("residual", {})
        if (
            meta.get("kind") == "final_student"
            and meta.get("rounds") == STUDENT_ROUNDS
            and meta.get("params") == dict(STUDENT_PARAMS)
            and meta.get("model_sha256") == _sha256_file(model_path)
            and residual_path.exists()
            and residual_meta.get("sha256") == _sha256_file(residual_path)
            and expected_teacher_parts == found_teacher_parts
        ):
            meta["status"] = "reused"
            return meta

    q, teacher_parts = _load_final_q(out_root, d)
    residual, residual_meta = _projection_residual(_logit(q), d, base_folds=FOLDS)
    train_rows = d.rows_for(FOLDS)
    train_rows = train_rows[np.isfinite(residual[train_rows])]
    sampled_rows = _sample_rows(train_rows, STUDENT_FINAL_MAX_TRAIN_ROWS, SEED + 3000)

    t0 = time.time()
    params = dict(STUDENT_PARAMS)
    x_train = PL._stack(mats, names, sampled_rows, keep_idx)
    if x_train.shape[1] != 500:
        raise RuntimeError(f"expected 500 student columns, got {x_train.shape[1]}")
    y_train = residual[sampled_rows].astype(np.float32)
    booster = _train_lgb(params, x_train, y_train, STUDENT_ROUNDS)
    booster.save_model(str(model_path))

    sample = sampled_rows[: min(4096, len(sampled_rows))]
    before = booster.predict(PL._stack(mats, names, sample, keep_idx))
    loaded = lgb.Booster(model_file=str(model_path))
    after = loaded.predict(PL._stack(mats, names, sample, keep_idx))
    save_load_max_abs = float(np.max(np.abs(before - after))) if len(sample) else 0.0
    del x_train, y_train, booster, loaded, before, after
    gc.collect()

    residual_path = out_root / "RT-1320_final10k_final_residual.npy"
    np.save(residual_path, residual.astype(np.float32))

    meta = {
        "kind": "final_student",
        "teacher_parts": teacher_parts,
        "residual": {
            **residual_meta,
            "target_stats": _target_stats(residual, train_rows),
            "path": str(residual_path),
            "sha256": _sha256_file(residual_path),
        },
        "train_rows_available": int(len(train_rows)),
        "train_rows_sampled": int(len(sampled_rows)),
        "params": params,
        "rounds": int(STUDENT_ROUNDS),
        "model_path": str(model_path),
        "model_sha256": _sha256_file(model_path),
        "save_load_repro_max_abs": save_load_max_abs,
        "runtime_s": round(time.time() - t0, 1),
        "score_computed": False,
        "status": "trained",
    }
    _dump_json(json_path, meta)
    print(
        f"final student: {len(sampled_rows)} train rows, {meta['runtime_s']}s -> {model_path}",
        flush=True,
    )
    return meta


def _training_run(datasets: Iterable[tuple[int, Any, Any, Any]], model_directory_path: str) -> None:
    import numpy as np

    t_start = time.time()
    out_root = _prepare_output_root(model_directory_path)
    artifact_root = Path(tempfile.mkdtemp(prefix="sbr_rt1320_final10k_")).resolve()
    try:
        preparation = _materialize_final10k_store(datasets, artifact_root)
        PL = _configure_runtime(artifact_root)
        feature_manifest = _build_feature_cache(artifact_root)
        d = PL.Data()
        mats, names, keep_idx = _load_features(PL)
        rows = _row_arrays(d)
        all_rows = d.rows_for(FOLDS)
        if len(np.unique(d.sidx[all_rows])) != 10000:
            raise RuntimeError("final10k fit must see all 10,000 labelled series")

        final_of_row = _last_row_lookup(d)[d.sidx]
        purity = _fold_purity_report()

        teacher_final = [
            _train_final_teacher_checkpoint(
                PL,
                d,
                mats,
                names,
                keep_idx,
                final_of_row,
                out_root,
                fold,
            )
            for fold in FOLDS
        ]
        final_student = _train_final_student(PL, d, mats, names, keep_idx, out_root)
        _dump_json(
            out_root / "RT1320_FINAL_STUDENT_READY.json",
            {
                "schema": "sbr.rt1320_final10k_fit.final_student_ready/1",
                "timestamp": _utc_timestamp(),
                "label": LABEL,
                "teacher_final": teacher_final,
                "final_student": final_student,
                "score_computed": False,
            },
        )

        teacher_nested = []
        for outer_fold in FOLDS:
            for inner_fold in FOLDS:
                if inner_fold == outer_fold:
                    continue
                teacher_nested.append(
                    _train_teacher_checkpoint(
                        PL,
                        d,
                        mats,
                        names,
                        keep_idx,
                        final_of_row,
                        out_root,
                        outer_fold,
                        inner_fold,
                    )
                )

        student_oof = [
            _train_student_oof_fold(PL, d, mats, names, keep_idx, out_root, outer_fold)
            for outer_fold in FOLDS
        ]
        oof_meta = _assemble_student_oof(out_root, d)
        calibration = _fit_student_calibration(out_root, d)

        total_runtime_s = float(time.time() - t_start)
        report = {
            "schema": "sbr.rt1320_final10k_fit/1",
            "label": LABEL,
            "timestamp": _utc_timestamp(),
            "artifact_contract": _artifact_contract(),
            "package_versions": _package_versions(),
            "runtime_config": {
                "lightgbm_num_threads": int(NUM_THREADS),
                "feature_workers": int(FEATURE_WORKERS),
                "feature_chunk": int(FEATURE_CHUNK),
            },
            "feature_preparation": {
                "series": preparation["selected_series"],
                "online_rows": preparation["selected_online_rows"],
                "wall_seconds": feature_manifest["wall_seconds"],
                "feature_count": feature_manifest["feature_count"],
                "feature_manifest_sha256": feature_manifest["stream_feature_manifest_sha256"],
            },
            "store_preparation": preparation,
            "run_order": [
                "materialize_final10k_store",
                "build_feature_cache",
                "train_final_teachers",
                "train_final_student_model_txt_7",
                "train_nested_teachers",
                "train_student_oof",
                "fit_student_calibration",
            ],
            "row_population": {
                "total_rows": int(len(d.y)),
                "all_rows": int(len(all_rows)),
                "dominant_cell_rows": int(rows["dominant_cell"].sum()),
                "dominant_never_break_rows": int(rows["dominant_never_break_only"].sum()),
                "dominant_pre_break_rows": int(rows["dominant_pre_break_only"].sum()),
            },
            "fold_purity": purity,
            "teacher_nested": teacher_nested,
            "teacher_final": teacher_final,
            "student_oof": student_oof,
            "assembled_oof": oof_meta,
            "student_calibration": {
                "path": calibration["path"],
                "sha256": calibration["sha256"],
                "kind": calibration["kind"],
                "time_coord": calibration["time_coord"],
            },
            "final_student": final_student,
            "artifact_files": sorted(p.name for p in out_root.iterdir() if p.is_file()),
            "score_computed": False,
            "total_runtime_s": round(total_runtime_s, 1),
        }
        _dump_json(out_root / "RT1320_FINAL10K_FIT_RESULT.json", report)
        (out_root / "RESEARCH_ONLY_NOT_LEADERBOARD.txt").write_text(
            "RT-1320 final10k artifact-build job. infer() returns dummy 0.5 scores.\n",
            encoding="utf-8",
        )
        print("=== RT1320_FINAL10K_FIT_RESULT_BEGIN ===", flush=True)
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default), flush=True)
        print("=== RT1320_FINAL10K_FIT_RESULT_END ===", flush=True)
    finally:
        if os.environ.get(KEEP_ARTIFACT_ENV) != "1":
            shutil.rmtree(artifact_root, ignore_errors=True)


def _run_smoke(model_directory_path: str) -> None:
    out_root = _prepare_output_root(model_directory_path)
    purity = _fold_purity_report()
    report = {
        "schema": "sbr.rt1320_final10k_fit.smoke/1",
        "label": LABEL,
        "timestamp": _utc_timestamp(),
        "fold_purity": purity,
        "artifact_contract": _artifact_contract(),
        "score_computed": False,
        "PASS": True,
    }
    _dump_json(out_root / "SMOKE_RESULT.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def train(
    datasets: Iterable[tuple[int, Any, Any, Any]],
    model_directory_path: str,
) -> None:
    print(LABEL, flush=True)
    if os.environ.get(SMOKE_ENV) == "1":
        _run_smoke(model_directory_path)
        return
    _training_run(datasets, model_directory_path)


def _online_stream_from_infer_item(item: Any):
    """Return the online point stream WITHOUT materialising it.

    Crunch's real-time ``infer`` hands back ``x_online`` as a generator that
    yields points as they arrive, so ``np.asarray`` on it raises TypeError.
    The stream has to be consumed lazily, one point at a time, exactly as
    H_gpu_tabular_full_oof.py does -- materialising it up front would break
    the streaming protocol even where it did not raise.
    """
    if isinstance(item, (tuple, list)):
        if len(item) == 2:
            return item[1]
        if len(item) >= 3:
            return item[2]
    if isinstance(item, dict) and "x_online" in item:
        return item["x_online"]
    raise RuntimeError(f"Unsupported infer item shape: {type(item).__name__}")


def infer(
    datasets: Iterable[Any],
    model_directory_path: str,
):
    del model_directory_path
    yield

    for item in datasets:
        for _point in _online_stream_from_infer_item(item):
            yield 0.5
