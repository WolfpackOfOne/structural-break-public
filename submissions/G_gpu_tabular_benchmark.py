"""Crunch cloud GPU benchmark for GPU-01 TabM and GPU-02 RealMLP.

BENCHMARK ONLY - NOT FOR SCORING.

This file intentionally exposes the same Real-Time Crunch public interface as
the deployed RT600 submission:

    train(datasets, model_directory_path)
    infer(datasets, model_directory_path)
    INFER_PARALLELISM

The training phase characterizes RTX hardware for the preregistered GPU
tabular package. It does not compute TS-AUC, does not train five folds, does
not touch test/reduced data, does not allocate RT IDs, and does not persist a
scoring model. The inference phase emits deterministic dummy probabilities only
because the platform requires a contract-valid infer() generator.
"""

from __future__ import annotations

# ruff: noqa: UP006, UP035, UP045
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_k, "16")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (
    _REPO_ROOT / "src",
    _REPO_ROOT / "research" / "scripts",
    _REPO_ROOT / "research" / "scripts" / "gpu_tabular",
):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

# If Crunch's uploaded code tree has no .git directory, this preserves the
# branch tip the benchmark wrapper was created from. Prefer SBR_SOURCE_GIT_SHA
# when launching a cloud run from a later committed revision.
_SOURCE_GIT_SHA_FALLBACK = "e39a69c9eca8d35cbbf42211151f8e77ee3a3a96"

# Series are independent and infer() is a dummy stream, so one worker is enough
# and keeps determinism checks simple.
# @crunch/keep:on
INFER_PARALLELISM = 1

BENCHMARK_LABEL = "BENCHMARK ONLY - NOT FOR SCORING"
BENCHMARK_FOLD = 0
SEED = 1
MAX_TRAIN_ROWS = 1_000_000
BENCHMARK_EPOCHS = int(os.environ.get("SBR_GPU_BENCHMARK_EPOCHS", "3"))
QUOTA_HOURS = float(os.environ.get("SBR_GPU_QUOTA_HOURS", "15.0"))
QUOTA_LEARNERS = int(os.environ.get("SBR_GPU_QUOTA_LEARNERS", "2"))
QUOTA_FRACTION = float(os.environ.get("SBR_GPU_QUOTA_FRACTION", "0.90"))
FEATURE_WORKERS = int(os.environ.get("SBR_GPU_FEATURE_WORKERS", "16"))
FEATURE_CHUNK = int(os.environ.get("SBR_GPU_FEATURE_CHUNK", "64"))


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _source_git_sha() -> str:
    for key in ("SBR_SOURCE_GIT_SHA", "GITHUB_SHA"):
        value = os.environ.get(key)
        if value:
            return value
    try:
        return subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return _SOURCE_GIT_SHA_FALLBACK


def _source_git_sha_kind() -> str:
    if os.environ.get("SBR_SOURCE_GIT_SHA"):
        return "SBR_SOURCE_GIT_SHA"
    if os.environ.get("GITHUB_SHA"):
        return "GITHUB_SHA"
    try:
        subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return "git"
    except Exception:
        return "embedded_fallback"


def _require_benchmark_epoch_bound() -> None:
    if BENCHMARK_EPOCHS < 2 or BENCHMARK_EPOCHS > 3:
        raise RuntimeError(
            f"SBR_GPU_BENCHMARK_EPOCHS must be 2 or 3 for this benchmark, got {BENCHMARK_EPOCHS}"
        )


def _torch_gpu_report_subprocess() -> dict[str, Any]:
    code = r"""
import json
import torch
if not torch.cuda.is_available():
    raise SystemExit("GPU REQUIRED: torch.cuda.is_available() is False")
props = torch.cuda.get_device_properties(0)
print(json.dumps({
    "cuda_available": True,
    "device": "cuda:0",
    "device_name": torch.cuda.get_device_name(0),
    "cuda_device_capability": torch.cuda.get_device_capability(0),
    "cuda_runtime_version": getattr(torch.version, "cuda", None),
    "torch_version": torch.__version__,
    "total_vram_gb": props.total_memory / 1e9,
}, sort_keys=True))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(msg or "GPU REQUIRED: torch CUDA preflight failed")
    return json.loads(proc.stdout)


def _print_device_report(label: str, report: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "benchmark": label,
                "cuda_available": report.get("cuda_available"),
                "gpu_name": report.get("device_name"),
                "cuda_version": report.get("cuda_runtime_version"),
                "torch_version": report.get("torch_version"),
                "total_vram_gb": report.get("total_vram_gb"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _series_values(obj: Any) -> Any:
    import numpy as np

    if hasattr(obj, "columns") and "value" in obj.columns:
        return obj["value"].to_numpy(dtype=np.float32)
    if isinstance(obj, dict) and "value" in obj:
        return np.asarray(obj["value"], dtype=np.float32).reshape(-1)
    return np.asarray(obj, dtype=np.float32).reshape(-1)


def _normalise_tau(value: Any, n_online: int) -> int:
    import math

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


def _materialize_outer_train_store(
    datasets: Iterable[Tuple[int, List[float], List[float], Optional[int]]],
    artifact_root: Path,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    folds_src = _REPO_ROOT / "research" / "folds" / "folds.parquet"
    canonical = pd.read_parquet(folds_src).sort_values("id").reset_index(drop=True)
    outer_train_folds = [1, 2, 3, 4]
    selected_folds = canonical[canonical["fold"].isin(outer_train_folds)].copy()
    selected_ids = set(int(x) for x in selected_folds["id"].to_numpy())

    seen: dict[int, tuple[Any, Any, int]] = {}
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
        hist = _series_values(x_historical)
        online = _series_values(x_online)
        tau = _normalise_tau(tau_raw, len(online))
        seen[sid] = (hist, online, tau)

    missing = [int(x) for x in selected_folds["id"].to_numpy() if int(x) not in seen]
    if missing:
        raise RuntimeError(
            "Crunch train data is missing canonical fold-0 outer-training ids; "
            f"first missing ids={missing[:10]} total_missing={len(missing)}"
        )

    store_dir = artifact_root / "cache" / "store"
    folds_dir = artifact_root / "research" / "folds"
    store_dir.mkdir(parents=True, exist_ok=True)
    folds_dir.mkdir(parents=True, exist_ok=True)

    total_values = int(
        sum(
            len(seen[int(sid)][0]) + len(seen[int(sid)][1])
            for sid in selected_folds["id"]
        )
    )
    values = np.lib.format.open_memmap(
        store_dir / "values.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_values,),
    )

    rows = []
    pos = 0
    for sid_raw in selected_folds["id"].to_numpy():
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
        c
        for c in selected_folds.columns
        if c not in {"n_hist", "n_online", "tau_index", "has_break"}
    ]
    folds_out = selected_folds[fold_cols].merge(
        meta[["id", "n_hist", "n_online", "tau_index", "has_break"]],
        on="id",
        how="left",
    )
    folds_out.to_parquet(folds_dir / "folds.parquet", index=False)

    return {
        "n_train_items_seen": int(n_total_items),
        "store_dir": str(store_dir),
        "folds_path": str(folds_dir / "folds.parquet"),
        "selected_folds": outer_train_folds,
        "selected_series": int(len(meta)),
        "selected_online_rows": int(meta["n_online"].sum()),
        "selected_total_values": int(total_values),
        "lockbox_series_materialized": 0,
        "validation_fold0_series_materialized": 0,
    }


def _build_feature_cache(artifact_root: Path, store_manifest: dict[str, Any]) -> dict[str, Any]:
    from common import FULL_MODULES

    from sbr.features.driver import build_features

    features_dir = artifact_root / "cache" / "features"
    t0 = time.time()
    manifest = build_features(
        list(FULL_MODULES),
        store=store_manifest["store_dir"],
        out=str(features_dir),
        workers=FEATURE_WORKERS,
        chunk=FEATURE_CHUNK,
        limit=0,
    )
    manifest["wall_seconds"] = float(time.time() - t0)
    manifest["feature_modules"] = list(FULL_MODULES)
    manifest["feature_count"] = int(sum(v["columns"] for v in manifest["modules"].values()))
    if manifest["feature_count"] != 500:
        raise RuntimeError(f"Expected 500 causal features, built {manifest['feature_count']}")
    return manifest


def _prepare_artifact_root(
    datasets: Iterable[Tuple[int, List[float], List[float], Optional[int]]],
) -> tuple[Path, dict[str, Any]]:
    artifact_root = Path(tempfile.mkdtemp(prefix="sbr_gpu_tabular_benchmark_")).resolve()
    os.environ["SBR_ROOT"] = str(artifact_root)
    os.environ["SBR_STORE"] = str(artifact_root / "cache" / "store")
    os.environ["SBR_FEATURES"] = str(artifact_root / "cache" / "features")

    t0 = time.time()
    store_manifest = _materialize_outer_train_store(datasets, artifact_root)
    feature_manifest = _build_feature_cache(artifact_root, store_manifest)
    prepared = {
        "artifact_root": str(artifact_root),
        "store": store_manifest,
        "features": feature_manifest,
        "seconds": float(time.time() - t0),
    }
    return artifact_root, prepared


def _benchmark_args(learner: str, artifact_root: Path, output_root: Path) -> Namespace:
    report_root = artifact_root / "research" / "reports" / "gpu_tabular_2026"
    return Namespace(
        learner=learner,
        artifact_root=str(artifact_root),
        output_root=str(output_root),
        config=str(report_root / "FROZEN_GPU_CONFIG.json"),
        benchmark_log=str(report_root / "benchmark_results.json"),
        fold=BENCHMARK_FOLD,
        max_train_rows=MAX_TRAIN_ROWS,
        benchmark_epochs=BENCHMARK_EPOCHS,
        batch_size=None,
        quota_hours=QUOTA_HOURS,
        quota_learners=QUOTA_LEARNERS,
        quota_fraction=QUOTA_FRACTION,
        allow_cpu_smoke=False,
        synthetic_smoke=False,
        no_write_config=True,
    )


def _synthetic_smoke_args(learner: str, output_root: Path) -> Namespace:
    return Namespace(
        learner=learner,
        artifact_root=str(_REPO_ROOT),
        output_root=str(output_root),
        config=str(output_root / "FROZEN_GPU_CONFIG.json"),
        benchmark_log=str(output_root / "benchmark_results.json"),
        fold=BENCHMARK_FOLD,
        max_train_rows=512,
        benchmark_epochs=max(2, min(BENCHMARK_EPOCHS, 3)),
        batch_size=None,
        quota_hours=QUOTA_HOURS,
        quota_learners=QUOTA_LEARNERS,
        quota_fraction=QUOTA_FRACTION,
        allow_cpu_smoke=True,
        synthetic_smoke=True,
        no_write_config=True,
    )


def _selected_attempt(entry: dict[str, Any]) -> dict[str, Any] | None:
    selected = entry.get("selected_config") or {}
    selected_training = selected.get("training") or {}
    selected_model = selected.get("model") or {}
    for attempt in entry.get("attempts") or []:
        if attempt.get("status") != "BENCHMARKED_NO_SCORE":
            continue
        cfg = attempt.get("config") or {}
        if (cfg.get("training") or {}).get("batch_size") != selected_training.get("batch_size"):
            continue
        if (cfg.get("model") or {}) == selected_model:
            return attempt
    for attempt in entry.get("attempts") or []:
        if attempt.get("status") == "BENCHMARKED_NO_SCORE":
            return attempt
    return None


def _learner_freeze(entry: dict[str, Any]) -> dict[str, Any]:
    attempt = _selected_attempt(entry) or {}
    benchmark_config = copy.deepcopy(attempt.get("config") or entry.get("selected_config") or {})
    benchmark_config.setdefault("training", {})
    benchmark_config["training"]["benchmark_epochs_run"] = int(entry["benchmark_epochs"])
    bench = entry.get("selected_benchmark") or attempt.get("benchmark") or {}
    projection = entry.get("selected_runtime_projection") or attempt.get("projection") or {}
    return {
        "status": entry.get("status"),
        "source": entry.get("source"),
        "fold_benchmarked": int(entry.get("fold_benchmarked", BENCHMARK_FOLD)),
        "benchmark_config": benchmark_config,
        "selected_final_config": entry.get("selected_config"),
        "preprocessing_seconds": entry.get("preprocessing_seconds"),
        "feature_stack_seconds": entry.get("feature_stack_seconds"),
        "array_preprocess_seconds": entry.get("array_preprocess_seconds"),
        "seconds_per_epoch": bench.get("seconds_per_epoch"),
        "batches_per_second": bench.get("batches_per_second"),
        "rows_per_second": bench.get("rows_per_second"),
        "peak_RAM_GB": attempt.get("peak_ram_gb") or entry.get("peak_ram_gb"),
        "peak_VRAM_GB": attempt.get("peak_vram_gb") or entry.get("peak_vram_gb"),
        "device": entry.get("device"),
        "projected_runtime_per_fold_hours": projection.get("projected_runtime_per_fold_hours"),
        "projected_5fold_hours": projection.get("projected_5fold_runtime_hours"),
        "runtime_projection": projection,
        "score_computed": False,
    }


def _freeze_json(
    tabm_entry: dict[str, Any],
    realmlp_entry: dict[str, Any],
    preparation: dict[str, Any],
    synthetic_smoke: bool,
) -> dict[str, Any]:
    from common import env_versions

    tabm = _learner_freeze(tabm_entry)
    realmlp = _learner_freeze(realmlp_entry)
    tabm_hours = float(tabm.get("projected_5fold_hours") or 0.0)
    realmlp_hours = float(realmlp.get("projected_5fold_hours") or 0.0)
    device = tabm_entry.get("device") or realmlp_entry.get("device") or {}
    return {
        "label": BENCHMARK_LABEL,
        "timestamp": _utc_timestamp(),
        "git_sha": _source_git_sha(),
        "git_sha_source": _source_git_sha_kind(),
        "gpu_name": device.get("device_name"),
        "cuda_version": device.get("cuda_runtime_version"),
        "torch_version": device.get("torch_version"),
        "total_vram_gb": device.get("total_vram_gb"),
        "package_versions": env_versions(),
        "data_contract": {
            "feature_count": 500,
            "feature_modules": [
                "m00_core",
                "m01_seq",
                "m02_dist",
                "m03_dyn",
                "m04_resid",
                "m06_loc",
                "m07_bayes",
            ],
            "canonical_outer_fold": 0,
            "outer_training_folds": [1, 2, 3, 4],
            "max_train_rows": MAX_TRAIN_ROWS,
            "row_sampling_convention": "RT-401 sequential rng choice over outer-training rows",
            "seed": SEED,
            "test_or_reduced_test_touched": False,
            "lockbox_touched": False,
            "validation_scoring": False,
            "rt_ids_allocated": False,
        },
        "benchmark_epochs": BENCHMARK_EPOCHS,
        "quota_hours": QUOTA_HOURS,
        "quota_learners": QUOTA_LEARNERS,
        "quota_fraction": QUOTA_FRACTION,
        "artifact_preparation": preparation,
        "tabm": tabm,
        "realmlp": realmlp,
        "projected_combined_hours": tabm_hours + realmlp_hours,
        "projected_combined_runtime": {
            "hours": tabm_hours + realmlp_hours,
            "seconds": (tabm_hours + realmlp_hours) * 3600.0,
        },
        "score_computed": False,
        "synthetic_smoke": bool(synthetic_smoke),
    }


def _write_small_model_artifacts(model_directory_path: str, freeze: dict[str, Any]) -> None:
    model_dir = Path(model_directory_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "BENCHMARK_ONLY_NOT_FOR_SCORING.txt").write_text(
        "This run benchmarks GPU tabular training only. infer() returns dummy probabilities.\n",
        encoding="utf-8",
    )
    (model_dir / "GPU_TABULAR_FREEZE.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _print_freeze_block(freeze: dict[str, Any]) -> None:
    print("=== GPU_TABULAR_FREEZE_JSON_BEGIN ===", flush=True)
    print(json.dumps(freeze, indent=2, sort_keys=True), flush=True)
    print("=== GPU_TABULAR_FREEZE_JSON_END ===", flush=True)


def train(
    datasets: List[Tuple[int, List[float], List[float], Optional[int]]],
    model_directory_path: str,
) -> None:
    print(BENCHMARK_LABEL, flush=True)
    _require_benchmark_epoch_bound()

    synthetic_smoke = os.environ.get("SBR_GPU_BENCHMARK_SYNTHETIC_SMOKE") == "1"
    artifact_root: Path | None = None
    preparation: dict[str, Any]

    if synthetic_smoke:
        preparation = {"synthetic_smoke": True, "seconds": 0.0}
        output_root = Path(tempfile.mkdtemp(prefix="sbr_gpu_tabular_smoke_oof_")).resolve()
    else:
        preflight = _torch_gpu_report_subprocess()
        _print_device_report("preflight", preflight)
        artifact_root, preparation = _prepare_artifact_root(datasets)
        output_root = artifact_root / "research" / "oof" / "gpu_tabular_2026"

    try:
        import realmlp_runner
        import tabm_runner
        from common import clear_memory

        if synthetic_smoke:
            tabm_args = _synthetic_smoke_args("tabm", output_root)
            realmlp_args = _synthetic_smoke_args("realmlp", output_root)
        else:
            assert artifact_root is not None
            tabm_args = _benchmark_args("tabm", artifact_root, output_root)
            realmlp_args = _benchmark_args("realmlp", artifact_root, output_root)

        print("Running GPU-01 TabM benchmark", flush=True)
        tabm_entry = tabm_runner.benchmark(tabm_args)
        _print_device_report("GPU-01 TabM", tabm_entry["device"])
        clear_memory()

        print("Running GPU-02 RealMLP benchmark", flush=True)
        realmlp_entry = realmlp_runner.benchmark(realmlp_args)
        _print_device_report("GPU-02 RealMLP", realmlp_entry["device"])
        clear_memory()

        freeze = _freeze_json(tabm_entry, realmlp_entry, preparation, synthetic_smoke)
        _write_small_model_artifacts(model_directory_path, freeze)
        _print_freeze_block(freeze)
    finally:
        if artifact_root is not None and os.environ.get("SBR_GPU_KEEP_ARTIFACT_ROOT") != "1":
            shutil.rmtree(artifact_root, ignore_errors=True)


def infer(
    datasets: Iterable[Tuple[List[float], Iterable[float]]],
    model_directory_path: str,
):
    yield

    for _x_historical, x_online in datasets:
        for _point in x_online:
            yield 0.5
