#!/usr/bin/env python
"""GPU RealMLP runner for the matched learner-diversity contract.

This uses pytabkit's RealMLP_TD_Classifier. The runner checkpoints fold OOF
predictions and never computes validation TS-AUC.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from common import (
    FROZEN_CONFIG,
    LOCAL_OOF_ROOT,
    MAX_TRAIN_ROWS,
    SEED,
    assemble_oof,
    champ_fold_rows,
    clear_memory,
    completed_fold_valid,
    config_runtime_projection,
    configure_roots,
    env_versions,
    fold_paths,
    jsonable,
    load_data_and_features,
    load_frozen_learner_config,
    manifest_base,
    peak_ram_gb,
    peak_vram_gb,
    preprocess_arrays,
    require_committed_and_pushed,
    require_torch_device,
    rss_gb,
    save_completed_fold,
    set_reproducibility,
    stack_features,
    synthetic_classification,
)

LEARNER = "realmlp"


def default_config() -> dict[str, Any]:
    return {
        "seed": SEED,
        "data": {"max_train_rows": MAX_TRAIN_ROWS},
        "preprocessing": {"mode": "median_iqr", "n_quantiles_max": 1000, "noise_std": 0.0},
        "model": {
            "variant": "RealMLP_TD_Classifier",
            "n_cv": 1,
            "n_refit": 0,
            "val_fraction": 0.2,
            "hidden_sizes": [256, 256, 256],
            "use_ls": False,
            "val_metric_name": "cross_entropy",
            "lr": 0.04,
        },
        "training": {
            "batch_size": 8192,
            "predict_batch_size": 32768,
            "max_epochs": 256,
            "min_epochs_for_quota_selection": 8,
            "early_stopping": True,
            "verbosity": 2,
            "n_threads": 16,
        },
    }


def smoke_config() -> dict[str, Any]:
    c = default_config()
    c["model"].update({"hidden_sizes": [16, 16], "val_fraction": 0.2, "lr": 0.02})
    c["training"].update(
        {
            "batch_size": 32,
            "predict_batch_size": 128,
            "max_epochs": 2,
            "min_epochs_for_quota_selection": 1,
            "early_stopping": False,
            "verbosity": 0,
            "n_threads": 2,
        }
    )
    return c


def config_candidates(batch_size: int | None = None) -> list[dict[str, Any]]:
    base = default_config()
    if batch_size is not None:
        base["training"]["batch_size"] = int(batch_size)
    hidden_sets = [
        [256, 256, 256],
        [192, 192, 192],
        [128, 128, 128],
        [128, 128],
    ]
    out = []
    for hidden in hidden_sets:
        c = copy.deepcopy(base)
        c["model"]["hidden_sizes"] = hidden
        out.append(c)
    return out


def make_model(config: dict[str, Any], device_name: str, tmp_folder: Path, benchmark: bool = False):
    from pytabkit import RealMLP_TD_Classifier

    m = config["model"]
    t = config["training"]
    return RealMLP_TD_Classifier(
        device=device_name,
        random_state=int(config.get("seed", SEED)),
        n_cv=int(m.get("n_cv", 1)),
        n_refit=int(m.get("n_refit", 0)),
        val_fraction=float(m.get("val_fraction", 0.2)),
        n_threads=int(t.get("n_threads", 16)),
        tmp_folder=tmp_folder,
        verbosity=int(t.get("verbosity", 2)),
        val_metric_name=m.get("val_metric_name", "cross_entropy"),
        n_epochs=int(t.get("max_epochs", 256)),
        batch_size=int(t.get("batch_size", 8192)),
        predict_batch_size=int(t.get("predict_batch_size", 32768)),
        hidden_sizes=list(m.get("hidden_sizes", [256, 256, 256])),
        use_ls=bool(m.get("use_ls", False)),
        lr=float(m.get("lr", 0.04)),
        use_early_stopping=False if benchmark else bool(t.get("early_stopping", True)),
    )


def fit_realmlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    config: dict[str, Any],
    device_name: str,
    tmp_folder: Path,
    benchmark: bool = False,
) -> tuple[Any, np.ndarray | None, dict[str, Any]]:
    tmp_folder.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = make_model(config, device_name, tmp_folder, benchmark=benchmark)
    model.fit(X_train, y_train.astype(np.int64, copy=False))
    pred = None
    predict_seconds = 0.0
    if X_val is not None:
        tp = time.time()
        proba = model.predict_proba(X_val)
        predict_seconds = float(time.time() - tp)
        proba = np.asarray(proba)
        pred = (proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()).astype(
            np.float32
        )
    fit_seconds = float(time.time() - t0)
    epochs = int(config["training"]["max_epochs"])
    meta = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "seconds_per_epoch": fit_seconds / max(epochs, 1),
        "batches_per_second": (
            len(X_train) / max(int(config["training"]["batch_size"]), 1) * epochs
        )
        / max(fit_seconds, 1e-9),
        "rows_per_second": float(len(X_train) * epochs / max(fit_seconds, 1e-9)),
        "configured_epochs": epochs,
        "benchmark_mode": bool(benchmark),
        "score_computed": False,
    }
    for attr in ("best_epoch_", "best_epochs_", "n_epochs_", "n_epochs"):
        if hasattr(model, attr):
            try:
                meta[attr] = jsonable(getattr(model, attr))
            except Exception:
                pass
    return model, pred, meta


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    require_gpu = not bool(args.allow_cpu_smoke)
    torch_mod, device, device_report = require_torch_device(
        require_gpu=require_gpu, allow_cpu_smoke=bool(args.allow_cpu_smoke)
    )
    reproducibility = set_reproducibility(SEED, torch_mod)
    device_name = "cuda" if device.type == "cuda" else "cpu"
    if args.synthetic_smoke:
        X_train, y_train, _, _ = synthetic_classification(
            n_train=384, n_val=96, n_features=12, seed=SEED
        )
        artifact_root = configure_roots(args.artifact_root)
        source = "synthetic_smoke"
        feature_stack_seconds = 0.0
    else:
        artifact_root, d, mats, names, keep = load_data_and_features(args.artifact_root)
        tr_rows, _ = champ_fold_rows(d, int(args.fold), int(args.max_train_rows), SEED)
        t_load = time.time()
        X_train = stack_features(mats, names, tr_rows, keep)
        feature_stack_seconds = float(time.time() - t_load)
        y_train = d.y[tr_rows].astype(np.int64)
        source = "canonical_fold"
        print(
            f"loaded fold {args.fold} train matrix {X_train.shape} in {feature_stack_seconds:.1f}s",
            flush=True,
        )

    raw_rss = rss_gb()
    prep_config = smoke_config() if args.synthetic_smoke else default_config()
    X_proc, _, array_preprocess_seconds, prep_mode = preprocess_arrays(
        X_train, None, prep_config, SEED
    )
    prep_seconds = float(feature_stack_seconds + array_preprocess_seconds)
    print(
        f"realmlp preprocessing mode={prep_mode} feature_stack={feature_stack_seconds:.1f}s "
        f"array_preprocess={array_preprocess_seconds:.1f}s total={prep_seconds:.1f}s "
        f"rss_gb={rss_gb():.2f}",
        flush=True,
    )
    del X_train
    clear_memory(torch_mod)

    attempts = []
    selected = None
    selected_projection = None
    selected_train_meta = None
    for cand in (
        config_candidates(args.batch_size) if not args.synthetic_smoke else [smoke_config()]
    ):
        batch_sizes = [int(cand["training"]["batch_size"])]
        if device.type == "cuda":
            while batch_sizes[-1] > 256:
                batch_sizes.append(batch_sizes[-1] // 2)
        for bs in batch_sizes:
            trial = copy.deepcopy(cand)
            trial["training"]["batch_size"] = int(bs)
            trial["training"]["max_epochs"] = int(args.benchmark_epochs)
            try:
                clear_memory(torch_mod)
                _, _, meta = fit_realmlp(
                    X_proc,
                    y_train,
                    None,
                    trial,
                    device_name,
                    Path(args.output_root).resolve() / "realmlp_benchmark_tmp",
                    benchmark=True,
                )
                clear_memory(torch_mod)
            except RuntimeError as exc:
                msg = str(exc)
                attempts.append(
                    {
                        "config": trial,
                        "status": "OOM" if "out of memory" in msg.lower() else "ERROR",
                        "error": msg[:1000],
                    }
                )
                clear_memory(torch_mod)
                if "out of memory" in msg.lower():
                    continue
                raise
            selected_trial = copy.deepcopy(cand)
            selected_trial["training"]["batch_size"] = int(bs)
            seconds_per_epoch = float(meta["seconds_per_epoch"])
            projection = config_runtime_projection(
                preprocess_seconds=prep_seconds,
                seconds_per_epoch=seconds_per_epoch,
                max_epochs=int(selected_trial["training"]["max_epochs"]),
                quota_hours=float(args.quota_hours),
                quota_learners=int(args.quota_learners),
                quota_fraction=float(args.quota_fraction),
            )
            min_epochs = int(selected_trial["training"].get("min_epochs_for_quota_selection", 8))
            epoch_cap = min(
                int(selected_trial["training"]["max_epochs"]),
                int(projection["max_epochs_fit_under_budget"]),
            )
            attempt = {
                "config": selected_trial,
                "status": "BENCHMARKED_NO_SCORE",
                "benchmark": meta,
                "projection": projection,
                "peak_vram_gb": peak_vram_gb(torch_mod, device),
                "peak_ram_gb": peak_ram_gb(),
                "rss_gb_after_benchmark": rss_gb(),
            }
            attempts.append(attempt)
            if epoch_cap >= min_epochs:
                selected = copy.deepcopy(selected_trial)
                selected["training"]["max_epochs"] = int(epoch_cap)
                selected["selection"] = {
                    "basis": "runtime_vram_only_no_ts_auc",
                    "reduction_order_applied": (
                        "batch_size_for_gpu_utilization_or_oom, then epoch cap; "
                        "hidden width/layers unchanged for this selected candidate"
                    ),
                    "epoch_cap_was_reduced": epoch_cap
                    < int(selected_trial["training"]["max_epochs"]),
                }
                selected_projection = config_runtime_projection(
                    preprocess_seconds=prep_seconds,
                    seconds_per_epoch=seconds_per_epoch,
                    max_epochs=int(selected["training"]["max_epochs"]),
                    quota_hours=float(args.quota_hours),
                    quota_learners=int(args.quota_learners),
                    quota_fraction=float(args.quota_fraction),
                )
                selected_train_meta = meta
                break
        if selected is not None:
            break
    if selected is None:
        raise SystemExit(
            "No RealMLP benchmarked configuration fit the quota floor; "
            "rerun on RTX 4090 with adjusted quota/batch settings."
        )

    entry = {
        "learner": "GPU-02 RealMLP",
        "status": "FROZEN_AFTER_HARDWARE_BENCHMARK_NO_SCORE"
        if not args.synthetic_smoke
        else "SYNTHETIC_SMOKE_ONLY",
        "artifact_root": str(artifact_root),
        "source": source,
        "fold_benchmarked": int(args.fold),
        "benchmark_epochs": int(args.benchmark_epochs),
        "preprocessing_seconds": prep_seconds,
        "feature_stack_seconds": feature_stack_seconds,
        "array_preprocess_seconds": array_preprocess_seconds,
        "preprocessing_mode": prep_mode,
        "raw_rss_gb_before_preprocessing": raw_rss,
        "rss_gb_after_preprocessing": rss_gb(),
        "peak_ram_gb": peak_ram_gb(),
        "peak_vram_gb": peak_vram_gb(torch_mod, device),
        "device": device_report,
        "versions": env_versions(),
        "reproducibility": reproducibility,
        "attempts": attempts,
        "selected_config": selected,
        "selected_runtime_projection": selected_projection,
        "selected_benchmark": selected_train_meta,
        "score_computed": False,
    }
    return entry


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-root", default=None)
    p.add_argument("--config", default=str(FROZEN_CONFIG))
    p.add_argument("--output-root", default=str(LOCAL_OOF_ROOT))
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--allow-cpu-smoke", action="store_true")
    p.add_argument("--allow-uncommitted-config", action="store_true")
    p.add_argument("--allow-unpushed-config", action="store_true")
    p.add_argument("--synthetic-smoke", action="store_true")
    return p.parse_args()


def run_synthetic_smoke(args: argparse.Namespace) -> dict[str, Any]:
    torch_mod, device, device_report = require_torch_device(require_gpu=False, allow_cpu_smoke=True)
    config = smoke_config()
    set_reproducibility(SEED, torch_mod)
    device_name = "cuda" if device.type == "cuda" else "cpu"
    X_train, y_train, X_val, _ = synthetic_classification(
        n_train=384, n_val=96, n_features=12, seed=SEED
    )
    Xtr, Xva, prep_seconds, prep_mode = preprocess_arrays(X_train, X_val, config, SEED)
    _, pred, meta = fit_realmlp(
        Xtr,
        y_train,
        Xva,
        config,
        device_name,
        Path(args.output_root).resolve() / "realmlp_smoke_tmp",
        benchmark=True,
    )
    pred_path = Path(args.output_root).resolve() / "realmlp_smoke_pred.npy"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(pred_path, pred)
    return {
        "learner": LEARNER,
        "synthetic_smoke": True,
        "pred_path": str(pred_path),
        "prediction_shape": list(pred.shape),
        "preprocessing_seconds": prep_seconds,
        "preprocessing_mode": prep_mode,
        "train_meta": meta,
        "device": device_report,
        "score_computed": False,
    }


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        print(json.dumps(run_synthetic_smoke(args), indent=2, sort_keys=True))
        return

    config_path = Path(args.config).resolve()
    require_committed_and_pushed(
        config_path,
        allow_uncommitted=bool(args.allow_uncommitted_config),
        allow_unpushed=bool(args.allow_unpushed_config),
    )
    config = load_frozen_learner_config(config_path, LEARNER)
    torch_mod, device, device_report = require_torch_device(
        require_gpu=True, allow_cpu_smoke=bool(args.allow_cpu_smoke)
    )
    reproducibility = set_reproducibility(int(config.get("seed", SEED)), torch_mod)
    device_name = "cuda" if device.type == "cuda" else "cpu"
    artifact_root, d, mats, names, keep = load_data_and_features(args.artifact_root)
    out_root = Path(args.output_root).resolve()
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    manifest = manifest_base(artifact_root, LEARNER, config)
    manifest.update(
        {
            "device": device_report,
            "reproducibility": reproducibility,
            "folds_requested": folds,
            "frozen_config": str(config_path),
            "output_root": str(out_root),
        }
    )

    fold_summaries = []
    for fold in folds:
        tr_rows, va_rows = champ_fold_rows(
            d,
            fold,
            int(config.get("data", {}).get("max_train_rows", MAX_TRAIN_ROWS)),
            int(config.get("seed", SEED)),
        )
        if (
            args.resume
            and not args.force
            and completed_fold_valid(LEARNER, fold, len(va_rows), out_root)
        ):
            pred_path, meta_path = fold_paths(LEARNER, fold, out_root)
            print(f"realmlp fold {fold}: reusing completed checkpoint {pred_path}", flush=True)
            fold_summaries.append(
                {"fold": fold, "status": "SKIPPED_COMPLETE", "meta": str(meta_path)}
            )
            assemble_oof(LEARNER, d, out_root)
            continue

        t0 = time.time()
        Xtr_raw = stack_features(mats, names, tr_rows, keep)
        ytr = d.y[tr_rows].astype(np.int64)
        Xva_raw = stack_features(mats, names, va_rows, keep)
        Xtr, Xva, prep_seconds, prep_mode = preprocess_arrays(
            Xtr_raw, Xva_raw, config, int(config.get("seed", SEED)) + fold
        )
        del Xtr_raw, Xva_raw
        _, pred, train_meta = fit_realmlp(
            Xtr,
            ytr,
            Xva,
            config,
            device_name,
            out_root / LEARNER / f"tmp_fold_{fold}",
            benchmark=False,
        )
        fold_meta = {
            **manifest,
            "fold": int(fold),
            "train_rows": int(len(tr_rows)),
            "validation_rows_predicted": int(len(va_rows)),
            "preprocessing_seconds": prep_seconds,
            "preprocessing_mode": prep_mode,
            "train_meta": train_meta,
            "fold_runtime_seconds": float(time.time() - t0),
            "rss_gb": rss_gb(),
            "peak_vram_gb": peak_vram_gb(torch_mod, device),
            "score_computed": False,
        }
        save_completed_fold(LEARNER, fold, pred, fold_meta, out_root)
        fold_summaries.append(
            {
                "fold": fold,
                "status": "COMPLETED_NO_SCORE",
                "fold_runtime_seconds": fold_meta["fold_runtime_seconds"],
            }
        )
        del Xtr, Xva, pred
        clear_memory(torch_mod)
        assembly = assemble_oof(LEARNER, d, out_root)
        print(
            f"realmlp fold {fold}: checkpointed; completed={assembly['completed_folds']}",
            flush=True,
        )

    assembly = assemble_oof(LEARNER, d, out_root)
    print(
        json.dumps(
            {
                "learner": LEARNER,
                "folds": fold_summaries,
                "assembly": assembly,
                "score_computed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
