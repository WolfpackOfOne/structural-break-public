#!/usr/bin/env python
"""Shared helpers for the GPU tabular learner-diversity package.

This package is deliberately separate from prior learner-diversity scripts:
TabM and RealMLP are reopened only after RTX hardware benchmarking, and no
validation TS-AUC is computed by the training runners.
"""

from __future__ import annotations

import csv
import fcntl
import gc
import hashlib
import importlib.metadata as md
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO / "research" / "scripts" / "gpu_tabular"
REPORT_DIR = REPO / "research" / "reports" / "gpu_tabular_2026"
LOCAL_OOF_ROOT = REPO / "research" / "oof" / "gpu_tabular_2026"
FROZEN_CONFIG = REPORT_DIR / "FROZEN_GPU_CONFIG.json"
BENCHMARK_JSON = REPORT_DIR / "benchmark_results.json"
RESULTS_JSON = REPORT_DIR / "results.json"
RESULTS_CSV = REPORT_DIR / "results.csv"
FINAL_MD = REPORT_DIR / "FINAL.md"

FOLDS = (0, 1, 2, 3, 4)
FULL_MODULES = ("m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes")
SPECIALISTS = ("RT-300", "RT-410", "RT-411", "RT-412", "RT-413", "RT-414", "RT-415")
MATCHED_LGBM = "RT-401"
SEED = 1
MAX_TRAIN_ROWS = 1_000_000
PAIR_SEED = 20260827
PAIRS_PER_T = 64

PACKAGE_NAMES = (
    "torch",
    "tabm",
    "pytabkit",
    "rtdl_num_embeddings",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "pyarrow",
    "psutil",
)


def configure_roots(artifact_root: str | os.PathLike[str] | None = None) -> Path:
    """Set import/data roots before importing repo modules with module globals."""
    root = Path(artifact_root or os.environ.get("SBR_ARTIFACT_ROOT") or REPO).resolve()
    os.environ["SBR_ROOT"] = str(root)
    for p in (
        SCRIPT_DIR,
        REPO / "research" / "scripts",
        REPO / "src",
        root / "research" / "scripts",
        root / "src",
    ):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


def git_sha(short: bool = True) -> str:
    args = ["git", "-C", str(REPO), "rev-parse"]
    args.append("--short" if short else "HEAD")
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "nogit"


def git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "branch", "--show-current"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "nogit"


def env_versions() -> dict[str, str]:
    out: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for pkg in PACKAGE_NAMES:
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            out[pkg] = "missing"
    return out


def set_reproducibility(seed: int, torch_mod: Any | None = None) -> dict[str, Any]:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    report: dict[str, Any] = {
        "python_seed": seed,
        "numpy_seed": seed,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    }
    if torch_mod is not None:
        torch_mod.manual_seed(seed)
        if torch_mod.cuda.is_available():
            torch_mod.cuda.manual_seed_all(seed)
        try:
            torch_mod.use_deterministic_algorithms(True, warn_only=True)
            deterministic = "warn_only"
        except TypeError:
            torch_mod.use_deterministic_algorithms(True)
            deterministic = "strict"
        torch_mod.backends.cudnn.benchmark = False
        torch_mod.backends.cudnn.deterministic = True
        report.update(
            {
                "torch_seed": seed,
                "torch_deterministic_algorithms": deterministic,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cuda_strict_determinism_note": (
                    "CUDA deterministic algorithms are enabled with warn_only when supported; "
                    "PyTorch may warn for kernels without strict deterministic implementations."
                ),
            }
        )
    return report


def require_torch_device(require_gpu: bool, allow_cpu_smoke: bool):
    import torch

    has_cuda = bool(torch.cuda.is_available())
    if require_gpu and not has_cuda:
        raise SystemExit(
            "GPU REQUIRED: torch.cuda.is_available() is False. "
            "Use --allow-cpu-smoke only for tiny local smoke tests."
        )
    if not has_cuda and not allow_cpu_smoke:
        raise SystemExit(
            "CUDA unavailable. This package is built for the RTX 4090 GPU environment; "
            "pass --allow-cpu-smoke only for tiny local checks."
        )
    device = torch.device("cuda:0" if has_cuda else "cpu")
    if has_cuda:
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        props = torch.cuda.get_device_properties(device)
        total_vram_gb = float(props.total_memory / 1e9)
    else:
        total_vram_gb = 0.0
    report = {
        "cuda_available": has_cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if has_cuda else "cpu",
        "cuda_device_capability": torch.cuda.get_device_capability(0) if has_cuda else None,
        "cuda_runtime_version": getattr(torch.version, "cuda", None),
        "torch_version": torch.__version__,
        "total_vram_gb": total_vram_gb,
    }
    return torch, device, report


def rss_gb() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / 1e9)
    except Exception:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        scale = 1e9 if sys.platform == "darwin" else 1e6
        return float(ru / scale)


def peak_ram_gb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1e9 if sys.platform == "darwin" else 1e6
    return float(ru / scale)


def peak_vram_gb(torch_mod: Any, device: Any) -> float:
    if getattr(device, "type", None) != "cuda":
        return 0.0
    return float(torch_mod.cuda.max_memory_allocated(device) / 1e9)


def clear_memory(torch_mod: Any | None = None) -> None:
    gc.collect()
    if torch_mod is not None and torch_mod.cuda.is_available():
        torch_mod.cuda.empty_cache()


def jsonable(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def update_frozen_config(path: Path, learner: str, entry: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_id = hashlib.sha256(str(path).encode()).hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / f"sbr_gpu_tabular_config_{lock_id}.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            data = load_json(path)
        else:
            data = {
                "program": "GPU TABULAR 2026",
                "created_by": "research/scripts/gpu_tabular/benchmark_gpu.py",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "branch": git_branch(),
                "base_git_sha": git_sha(short=False),
                "seed": SEED,
                "matched_contract": {
                    "feature_modules": list(FULL_MODULES),
                    "folds": list(FOLDS),
                    "max_train_rows_per_fold": MAX_TRAIN_ROWS,
                    "matched_lgbm_control": MATCHED_LGBM,
                    "calibration": "SCDF_NSEEN",
                    "rt600_specialists": list(SPECIALISTS),
                    "no_test_or_lockbox": True,
                },
                "score_computed_during_benchmark": False,
                "learners": {},
            }
        data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        data["learners"][learner] = entry
        atomic_write_json(path, data)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return data


def load_frozen_learner_config(path: Path, learner: str) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"MISSING FROZEN CONFIG: {path}")
    data = load_json(path)
    try:
        entry = data["learners"][learner]
        return entry["selected_config"]
    except KeyError as exc:
        raise SystemExit(f"FROZEN CONFIG has no selected config for {learner}: {path}") from exc


def require_committed_and_pushed(
    path: Path, allow_uncommitted: bool = False, allow_unpushed: bool = False
) -> None:
    if allow_uncommitted:
        return
    rel = str(path.relative_to(REPO))
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "--error-unmatch", rel],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if tracked.returncode != 0:
        raise SystemExit(
            f"{rel} is not tracked. Commit/push FROZEN_GPU_CONFIG.json before scoring."
        )
    for args, label in (
        (["git", "-C", str(REPO), "diff", "--quiet", "--", rel], "unstaged"),
        (["git", "-C", str(REPO), "diff", "--cached", "--quiet", "--", rel], "staged"),
    ):
        proc = subprocess.run(args)
        if proc.returncode != 0:
            raise SystemExit(
                f"{rel} has {label} changes. Commit/push FROZEN_GPU_CONFIG.json before scoring."
            )
    if allow_unpushed:
        return
    upstream = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        text=True,
        capture_output=True,
    )
    if upstream.returncode != 0:
        raise SystemExit("No upstream branch is configured. Push the branch before scoring.")
    proc = subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", "HEAD", upstream.stdout.strip()]
    )
    if proc.returncode != 0:
        raise SystemExit(
            "HEAD is not contained in the upstream branch. "
            "Push FROZEN_GPU_CONFIG.json before scoring."
        )


def load_data_and_features(artifact_root: str | os.PathLike[str] | None = None):
    root = configure_roots(artifact_root)
    from sbr.pipeline import Data, load_features

    d = Data()
    mats, names = load_features(FULL_MODULES)
    if len(names) != 500:
        raise SystemExit(f"EXPECTED 500 canonical features, found {len(names)} from {FULL_MODULES}")
    keep = np.arange(len(names), dtype=np.int64)
    return root, d, mats, names, keep


def stack_features(mats, names, rows, keep_idx) -> np.ndarray:
    from sbr.pipeline import _stack

    return _stack(mats, names, rows, keep_idx)


def champ_fold_rows(
    d, fold: int, max_train_rows: int = MAX_TRAIN_ROWS, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    selected: tuple[np.ndarray, np.ndarray] | None = None
    for f in FOLDS:
        tr_rows = d.rows_for([g for g in FOLDS if g != f])
        va_rows = d.rows_for([f])
        if len(tr_rows) > max_train_rows:
            tr_rows = np.sort(rng.choice(tr_rows, max_train_rows, replace=False))
        if int(f) == int(fold):
            selected = (tr_rows, va_rows)
    if selected is None:
        raise KeyError(f"Unknown fold {fold}")
    return selected


def inner_train_val_split(
    rows: np.ndarray,
    y: np.ndarray,
    seed: int,
    val_fraction: float,
    max_val_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    if val_fraction <= 0:
        return rows, np.array([], dtype=np.int64)
    n_val = min(max(int(round(len(rows) * val_fraction)), 1), int(max_val_rows))
    rng = np.random.default_rng(seed)
    yy = y[rows]
    pos = np.flatnonzero(yy == 1)
    neg = np.flatnonzero(yy == 0)
    if len(pos) and len(neg):
        n_pos = min(len(pos), max(1, int(round(n_val * len(pos) / len(rows)))))
        n_neg = min(len(neg), n_val - n_pos)
        if n_neg <= 0:
            n_neg = min(len(neg), 1)
            n_pos = min(len(pos), n_val - n_neg)
        val_local = np.concatenate(
            [
                rng.choice(pos, n_pos, replace=False),
                rng.choice(neg, n_neg, replace=False),
            ]
        )
        if len(val_local) < n_val:
            remaining = np.setdiff1d(np.arange(len(rows)), val_local, assume_unique=False)
            extra = rng.choice(remaining, n_val - len(val_local), replace=False)
            val_local = np.concatenate([val_local, extra])
    else:
        val_local = rng.choice(np.arange(len(rows)), n_val, replace=False)
    val_local = np.sort(np.unique(val_local))
    mask = np.ones(len(rows), dtype=bool)
    mask[val_local] = False
    return rows[mask], rows[val_local]


@dataclass
class FittedPreprocessor:
    mode: str
    center: np.ndarray
    scale: np.ndarray | None = None
    transformer: Any | None = None

    def transform(self, X: np.ndarray) -> np.ndarray:
        X2 = np.asarray(X, dtype=np.float32).copy()
        bad = ~np.isfinite(X2)
        if bad.any():
            cols = np.where(bad)[1]
            X2[bad] = self.center[cols]
        if self.mode == "median_iqr":
            assert self.scale is not None
            X2 = (X2 - self.center) / self.scale
            np.clip(X2, -20.0, 20.0, out=X2)
            return X2.astype(np.float32, copy=False)
        if self.mode == "quantile_normal":
            assert self.transformer is not None
            return self.transformer.transform(X2).astype(np.float32, copy=False)
        raise KeyError(self.mode)


def fit_preprocessor(
    X_train: np.ndarray, mode: str, seed: int, n_quantiles_max: int = 1000, noise_std: float = 1e-5
) -> FittedPreprocessor:
    center = np.nanmedian(np.where(np.isfinite(X_train), X_train, np.nan), axis=0).astype(
        np.float32
    )
    center[~np.isfinite(center)] = 0.0
    if mode == "median_iqr":
        q = np.nanquantile(np.where(np.isfinite(X_train), X_train, np.nan), [0.25, 0.75], axis=0)
        scale = (q[1] - q[0]).astype(np.float32)
        scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
        return FittedPreprocessor(mode=mode, center=center, scale=scale)
    if mode == "quantile_normal":
        from sklearn.preprocessing import QuantileTransformer

        X_fit = np.asarray(X_train, dtype=np.float32).copy()
        bad = ~np.isfinite(X_fit)
        if bad.any():
            cols = np.where(bad)[1]
            X_fit[bad] = center[cols]
        if noise_std > 0:
            rng = np.random.default_rng(seed)
            X_fit += rng.normal(0.0, noise_std, size=X_fit.shape).astype(np.float32)
        n_quantiles = max(min(len(X_fit) // 30, int(n_quantiles_max)), 10)
        transformer = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            subsample=10**9,
            random_state=seed,
        ).fit(X_fit)
        del X_fit
        return FittedPreprocessor(mode=mode, center=center, transformer=transformer)
    raise KeyError(mode)


def preprocess_arrays(
    X_train: np.ndarray,
    X_val: np.ndarray | None,
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None, float, str]:
    t0 = time.time()
    pconf = dict(config.get("preprocessing", {}))
    mode = pconf.get("mode", "median_iqr")
    prep = fit_preprocessor(
        X_train,
        mode=mode,
        seed=seed,
        n_quantiles_max=int(pconf.get("n_quantiles_max", 1000)),
        noise_std=float(pconf.get("noise_std", 1e-5)),
    )
    Xtr = prep.transform(X_train)
    Xva = prep.transform(X_val) if X_val is not None else None
    return Xtr, Xva, float(time.time() - t0), mode


def synthetic_classification(
    n_train: int = 512, n_val: int = 128, n_features: int = 16, seed: int = SEED
):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_train + n_val, n_features)).astype(np.float32)
    beta = rng.normal(size=n_features).astype(np.float32)
    logits = X @ beta + 0.1 * rng.normal(size=len(X))
    y = (logits > np.median(logits)).astype(np.int64)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def fold_dir(learner: str, output_root: str | os.PathLike[str] | None = None) -> Path:
    return Path(output_root or LOCAL_OOF_ROOT).resolve() / learner


def fold_paths(
    learner: str, fold: int, output_root: str | os.PathLike[str] | None = None
) -> tuple[Path, Path]:
    fd = fold_dir(learner, output_root)
    return fd / f"fold_{fold}_pred.npy", fd / f"fold_{fold}_meta.json"


def save_completed_fold(
    learner: str,
    fold: int,
    pred: np.ndarray,
    meta: dict[str, Any],
    output_root: str | os.PathLike[str] | None = None,
) -> None:
    pred_path, meta_path = fold_paths(learner, fold, output_root)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pred_path.with_suffix(".npy.tmp")
    with tmp.open("wb") as fh:
        np.save(fh, np.asarray(pred, dtype=np.float32))
    tmp.replace(pred_path)
    atomic_write_json(meta_path, meta)


def completed_fold_valid(
    learner: str,
    fold: int,
    expected_len: int,
    output_root: str | os.PathLike[str] | None = None,
) -> bool:
    pred_path, meta_path = fold_paths(learner, fold, output_root)
    if not pred_path.exists() or not meta_path.exists():
        return False
    try:
        pred = np.load(pred_path, mmap_mode="r")
        return pred.shape == (expected_len,) and bool(np.isfinite(pred).all())
    except Exception:
        return False


def assemble_oof(
    learner: str,
    d: Any,
    output_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    root = Path(output_root or LOCAL_OOF_ROOT).resolve()
    oof = np.full(len(d.y), np.nan, dtype=np.float32)
    completed: list[int] = []
    for f in FOLDS:
        pred_path, _ = fold_paths(learner, int(f), root)
        rows = d.rows_for([f])
        if pred_path.exists():
            pred = np.load(pred_path)
            if pred.shape != (len(rows),):
                raise SystemExit(
                    f"Bad {learner} fold {f} pred shape {pred.shape}, expected {(len(rows),)}"
                )
            oof[rows] = pred.astype(np.float32, copy=False)
            completed.append(int(f))
    out_path = root / f"{learner}_oof.npy"
    if completed:
        tmp = out_path.with_suffix(".npy.tmp")
        with tmp.open("wb") as fh:
            np.save(fh, oof)
        tmp.replace(out_path)
    return {
        "oof_path": str(out_path),
        "completed_folds": completed,
        "all_folds_complete": completed == list(FOLDS),
        "score_computed": False,
    }


def config_runtime_projection(
    preprocess_seconds: float,
    seconds_per_epoch: float,
    max_epochs: int,
    quota_hours: float,
    quota_learners: int,
    quota_fraction: float,
) -> dict[str, Any]:
    per_learner_budget = (
        float(quota_hours) * 3600.0 * float(quota_fraction) / max(int(quota_learners), 1)
    )
    projected_per_fold = float(preprocess_seconds) + float(seconds_per_epoch) * int(max_epochs)
    projected = 5.0 * projected_per_fold
    max_epochs_fit = max(
        1,
        int(
            math.floor(
                ((per_learner_budget / 5.0) - float(preprocess_seconds))
                / max(float(seconds_per_epoch), 1e-9)
            )
        ),
    )
    return {
        "quota_hours_total": float(quota_hours),
        "quota_learners": int(quota_learners),
        "quota_fraction_for_scored_training": float(quota_fraction),
        "per_learner_budget_seconds": per_learner_budget,
        "projected_runtime_per_fold_seconds": projected_per_fold,
        "projected_runtime_per_fold_hours": projected_per_fold / 3600.0,
        "projected_5fold_runtime_seconds": projected,
        "projected_5fold_runtime_hours": projected / 3600.0,
        "max_epochs_fit_under_budget": max_epochs_fit,
        "fits_budget_at_configured_max_epochs": projected <= per_learner_budget,
    }


def load_eval_context(artifact_root: str | os.PathLike[str] | None = None):
    configure_roots(artifact_root)
    from wave5_lib import Ctx

    return Ctx()


def load_control_oof(artifact_root: str | os.PathLike[str] | None = None) -> dict[str, np.ndarray]:
    root = configure_roots(artifact_root)
    oof_dir = root / "research" / "oof"
    need = list(SPECIALISTS) + [MATCHED_LGBM]
    missing = [name for name in need if not (oof_dir / f"{name}.npy").exists()]
    if missing:
        raise SystemExit(f"MISSING CONTROL OOF {missing} in {oof_dir}")
    return {name: np.load(oof_dir / f"{name}.npy") for name in need}


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    exists = path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        if not exists:
            w.writeheader()
        for row in rows:
            w.writerow(row)


def manifest_base(artifact_root: Path, learner: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "program": "GPU TABULAR 2026",
        "learner": learner,
        "branch": git_branch(),
        "git_sha": git_sha(short=False),
        "artifact_root": str(artifact_root),
        "versions": env_versions(),
        "seed": int(config.get("seed", SEED)),
        "feature_modules": list(FULL_MODULES),
        "folds": list(FOLDS),
        "max_train_rows": int(config.get("data", {}).get("max_train_rows", MAX_TRAIN_ROWS)),
        "calibration_for_evaluation": "SCDF_NSEEN",
        "score_computed_by_runner": False,
        "test_or_lockbox_touched": False,
        "rt_ids_allocated": False,
    }
