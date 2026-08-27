#!/usr/bin/env python
"""GPU TabM runner for the matched learner-diversity contract.

Training writes fold OOF predictions and metadata only. It never computes
validation TS-AUC; evaluation is deferred to evaluate_gpu_oof.py after all five
folds are complete.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from contextlib import nullcontext
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
    inner_train_val_split,
    load_data_and_features,
    load_frozen_learner_config,
    manifest_base,
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

LEARNER = "tabm"


def default_config() -> dict[str, Any]:
    return {
        "seed": SEED,
        "data": {"max_train_rows": MAX_TRAIN_ROWS},
        "preprocessing": {
            "mode": "quantile_normal",
            "n_quantiles_max": 1000,
            "noise_std": 1e-5,
        },
        "model": {
            "arch_type": "tabm",
            "k": 32,
            "n_blocks": 3,
            "d_block": 512,
            "dropout": 0.1,
            "num_embeddings": None,
        },
        "training": {
            "batch_size": 8192,
            "max_epochs": 256,
            "min_epochs_for_quota_selection": 8,
            "early_stopping_patience": 16,
            "inner_val_fraction": 0.05,
            "inner_val_max_rows": 100_000,
            "eval_batch_size": 32768,
            "num_workers": 0,
            "amp": False,
            "torch_threads": 16,
        },
        "optimizer": {"lr": 0.002, "weight_decay": 0.0003, "grad_clip_norm": 1.0},
    }


def smoke_config() -> dict[str, Any]:
    c = default_config()
    c["preprocessing"] = {"mode": "median_iqr", "n_quantiles_max": 32, "noise_std": 0.0}
    c["model"].update({"k": 2, "n_blocks": 1, "d_block": 16, "dropout": 0.0})
    c["training"].update(
        {
            "batch_size": 64,
            "max_epochs": 2,
            "min_epochs_for_quota_selection": 1,
            "early_stopping_patience": 1,
            "inner_val_fraction": 0.2,
            "inner_val_max_rows": 64,
            "eval_batch_size": 128,
            "torch_threads": 2,
            "amp": False,
        }
    )
    return c


def config_candidates(batch_size: int | None = None) -> list[dict[str, Any]]:
    base = default_config()
    if batch_size is not None:
        base["training"]["batch_size"] = int(batch_size)
    specs = [
        {"k": 32, "d_block": 512, "n_blocks": 3},
        {"k": 16, "d_block": 512, "n_blocks": 3},
        {"k": 16, "d_block": 256, "n_blocks": 3},
        {"k": 8, "d_block": 256, "n_blocks": 3},
        {"k": 8, "d_block": 128, "n_blocks": 2},
    ]
    out = []
    for spec in specs:
        c = copy.deepcopy(base)
        c["model"].update(spec)
        out.append(c)
    return out


def make_model(n_features: int, config: dict[str, Any], device: Any):
    import tabm

    mconf = dict(config["model"])
    num_embeddings = mconf.pop("num_embeddings", None)
    emb = None
    if num_embeddings == "linear_relu":
        from rtdl_num_embeddings import LinearReLUEmbeddings

        emb = LinearReLUEmbeddings(n_features, d_embedding=int(mconf.pop("d_embedding", 32)))
    elif num_embeddings is not None:
        raise KeyError(f"Unsupported TabM num_embeddings={num_embeddings!r}")
    model = tabm.TabM.make(
        n_num_features=n_features,
        cat_cardinalities=[],
        d_out=2,
        num_embeddings=emb,
        **mconf,
    )
    return model.to(device)


def _autocast(torch_mod: Any, device: Any, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    dtype = torch_mod.bfloat16 if torch_mod.cuda.is_bf16_supported() else torch_mod.float16
    return torch_mod.autocast(device_type=device.type, dtype=dtype)


def _batch_indices(n: int, batch_size: int, shuffle: bool, seed: int, epoch: int):
    idx = np.arange(n, dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


def _loss_on_arrays(
    model: Any, X: np.ndarray, y: np.ndarray, config: dict[str, Any], torch_mod: Any, device: Any
) -> float:
    import torch.nn.functional as F

    batch_size = int(config["training"]["eval_batch_size"])
    amp = bool(config["training"].get("amp", False))
    total = 0.0
    n_seen = 0
    model.eval()
    with torch_mod.inference_mode():
        for idx in _batch_indices(len(X), batch_size, False, int(config["seed"]), 0):
            xb = torch_mod.from_numpy(np.ascontiguousarray(X[idx])).to(device, non_blocking=True)
            yb = torch_mod.from_numpy(np.ascontiguousarray(y[idx].astype(np.int64))).to(
                device, non_blocking=True
            )
            with _autocast(torch_mod, device, amp):
                logits = model(xb).float()
                loss = F.cross_entropy(logits.flatten(0, 1), yb.repeat_interleave(model.k))
            total += float(loss.item()) * len(idx)
            n_seen += len(idx)
    return total / max(n_seen, 1)


def fit_tabm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_inner_val: np.ndarray | None,
    y_inner_val: np.ndarray | None,
    config: dict[str, Any],
    torch_mod: Any,
    device: Any,
    benchmark_epochs: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    seed = int(config["seed"])
    set_reproducibility(seed, torch_mod)
    torch_mod.set_num_threads(int(config["training"].get("torch_threads", 16)))
    model = make_model(X_train.shape[1], config, device)
    opt_conf = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_conf.get("lr", 0.002)),
        weight_decay=float(opt_conf.get("weight_decay", 0.0003)),
    )
    max_epochs = int(benchmark_epochs or config["training"]["max_epochs"])
    batch_size = int(config["training"]["batch_size"])
    grad_clip = opt_conf.get("grad_clip_norm")
    patience = int(config["training"].get("early_stopping_patience", 0))
    has_inner_val = X_inner_val is not None and y_inner_val is not None and len(X_inner_val) > 0
    amp = bool(config["training"].get("amp", False))
    use_scaler = bool(amp and device.type == "cuda" and not torch_mod.cuda.is_bf16_supported())
    scaler = torch_mod.cuda.amp.GradScaler() if use_scaler else None

    best_state = None
    best_epoch = -1
    best_val_loss = math.inf
    stale = 0
    epoch_seconds: list[float] = []
    batch_counts: list[int] = []
    train_losses: list[float] = []

    for epoch in range(max_epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_seen = 0
        n_batches = 0
        for idx in _batch_indices(len(X_train), batch_size, True, seed, epoch):
            xb = torch_mod.from_numpy(np.ascontiguousarray(X_train[idx])).to(
                device, non_blocking=True
            )
            yb = torch_mod.from_numpy(np.ascontiguousarray(y_train[idx].astype(np.int64))).to(
                device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast(torch_mod, device, amp):
                logits = model(xb).float()
                loss = F.cross_entropy(logits.flatten(0, 1), yb.repeat_interleave(model.k))
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()
            total_loss += float(loss.detach().cpu().item()) * len(idx)
            n_seen += len(idx)
            n_batches += 1
        elapsed = float(time.time() - t0)
        epoch_seconds.append(elapsed)
        batch_counts.append(n_batches)
        train_losses.append(total_loss / max(n_seen, 1))

        if benchmark_epochs is not None:
            print(
                f"tabm benchmark epoch {epoch + 1}/{max_epochs}: "
                f"{elapsed:.2f}s batches={n_batches}",
                flush=True,
            )
            continue
        if has_inner_val:
            val_loss = _loss_on_arrays(model, X_inner_val, y_inner_val, config, torch_mod, device)
            print(
                f"tabm epoch {epoch + 1}/{max_epochs}: "
                f"train_bce={train_losses[-1]:.5f} inner_bce={val_loss:.5f}",
                flush=True,
            )
            if val_loss < best_val_loss - 1e-7:
                best_val_loss = float(val_loss)
                best_epoch = epoch
                best_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in model.state_dict().items()}
                )
                stale = 0
            else:
                stale += 1
                if stale > patience:
                    break
        else:
            print(
                f"tabm epoch {epoch + 1}/{max_epochs}: train_bce={train_losses[-1]:.5f}", flush=True
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    meta = {
        "epochs_completed": len(epoch_seconds),
        "seconds_per_epoch": float(np.mean(epoch_seconds)) if epoch_seconds else 0.0,
        "batches_per_second": float(np.sum(batch_counts) / max(np.sum(epoch_seconds), 1e-9)),
        "epoch_seconds": epoch_seconds,
        "train_loss_last": train_losses[-1] if train_losses else None,
        "best_epoch_by_inner_bce": best_epoch if best_epoch >= 0 else None,
        "best_inner_bce": best_val_loss if np.isfinite(best_val_loss) else None,
        "score_computed": False,
    }
    return model, meta


def predict_tabm(
    model: Any, X: np.ndarray, config: dict[str, Any], torch_mod: Any, device: Any
) -> np.ndarray:
    batch_size = int(config["training"]["eval_batch_size"])
    amp = bool(config["training"].get("amp", False))
    out = []
    model.eval()
    with torch_mod.inference_mode():
        for idx in _batch_indices(len(X), batch_size, False, int(config["seed"]), 0):
            xb = torch_mod.from_numpy(np.ascontiguousarray(X[idx])).to(device, non_blocking=True)
            with _autocast(torch_mod, device, amp):
                logits = model(xb).float()
            prob = logits.softmax(dim=-1)[:, :, 1].mean(dim=1).detach().cpu().numpy()
            out.append(prob.astype(np.float32))
    return np.concatenate(out)


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    require_gpu = not bool(args.allow_cpu_smoke)
    torch_mod, device, device_report = require_torch_device(
        require_gpu=require_gpu, allow_cpu_smoke=bool(args.allow_cpu_smoke)
    )
    reproducibility = set_reproducibility(SEED, torch_mod)
    if args.synthetic_smoke:
        X_train, y_train, _, _ = synthetic_classification(
            n_train=384, n_val=96, n_features=12, seed=SEED
        )
        artifact_root = configure_roots(args.artifact_root)
        source = "synthetic_smoke"
    else:
        artifact_root, d, mats, names, keep = load_data_and_features(args.artifact_root)
        tr_rows, _ = champ_fold_rows(d, int(args.fold), int(args.max_train_rows), SEED)
        t_load = time.time()
        X_train = stack_features(mats, names, tr_rows, keep)
        y_train = d.y[tr_rows].astype(np.int64)
        source = "canonical_fold"
        print(
            f"loaded fold {args.fold} train matrix {X_train.shape} in {time.time() - t_load:.1f}s",
            flush=True,
        )

    raw_rss = rss_gb()
    prep_config = default_config()
    if args.synthetic_smoke:
        prep_config = smoke_config()
    X_proc, _, prep_seconds, prep_mode = preprocess_arrays(X_train, None, prep_config, SEED)
    print(
        f"tabm preprocessing mode={prep_mode} seconds={prep_seconds:.1f} rss_gb={rss_gb():.2f}",
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
            try:
                clear_memory(torch_mod)
                _, meta = fit_tabm(
                    X_proc,
                    y_train,
                    None,
                    None,
                    trial,
                    torch_mod,
                    device,
                    benchmark_epochs=int(args.benchmark_epochs),
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
            seconds_per_epoch = float(meta["seconds_per_epoch"])
            projection = config_runtime_projection(
                preprocess_seconds=prep_seconds,
                seconds_per_epoch=seconds_per_epoch,
                max_epochs=int(trial["training"]["max_epochs"]),
                quota_hours=float(args.quota_hours),
                quota_learners=int(args.quota_learners),
                quota_fraction=float(args.quota_fraction),
            )
            min_epochs = int(trial["training"].get("min_epochs_for_quota_selection", 8))
            epoch_cap = min(
                int(trial["training"]["max_epochs"]), int(projection["max_epochs_fit_under_budget"])
            )
            attempt = {
                "config": trial,
                "status": "BENCHMARKED_NO_SCORE",
                "benchmark": meta,
                "projection": projection,
                "peak_vram_gb": peak_vram_gb(torch_mod, device),
                "rss_gb_after_benchmark": rss_gb(),
            }
            attempts.append(attempt)
            if epoch_cap >= min_epochs:
                selected = copy.deepcopy(trial)
                selected["training"]["max_epochs"] = int(epoch_cap)
                selected["selection"] = {
                    "basis": "runtime_vram_only_no_ts_auc",
                    "reduction_order_applied": (
                        "batch_size_for_gpu_utilization_or_oom, then epoch cap; "
                        "k/d_block unchanged for this selected candidate"
                    ),
                    "epoch_cap_was_reduced": epoch_cap < int(trial["training"]["max_epochs"]),
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
            "No TabM benchmarked configuration fit the quota floor; "
            "rerun on RTX 4090 with adjusted quota/batch settings."
        )

    entry = {
        "learner": "GPU-01 TabM",
        "status": "FROZEN_AFTER_HARDWARE_BENCHMARK_NO_SCORE"
        if not args.synthetic_smoke
        else "SYNTHETIC_SMOKE_ONLY",
        "artifact_root": str(artifact_root),
        "source": source,
        "fold_benchmarked": int(args.fold),
        "benchmark_epochs": int(args.benchmark_epochs),
        "preprocessing_seconds": prep_seconds,
        "preprocessing_mode": prep_mode,
        "raw_rss_gb_before_preprocessing": raw_rss,
        "rss_gb_after_preprocessing": rss_gb(),
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
    X_train, y_train, X_val, _ = synthetic_classification(
        n_train=384, n_val=96, n_features=12, seed=SEED
    )
    Xtr, Xva, prep_seconds, prep_mode = preprocess_arrays(X_train, X_val, config, SEED)
    tr_idx, va_idx = inner_train_val_split(np.arange(len(Xtr)), y_train, SEED, 0.2, 64)
    mask = np.isin(np.arange(len(Xtr)), va_idx)
    model, meta = fit_tabm(
        Xtr[~mask], y_train[~mask], Xtr[mask], y_train[mask], config, torch_mod, device
    )
    pred = predict_tabm(model, Xva, config, torch_mod, device)
    pred_path = Path(args.output_root).resolve() / "tabm_smoke_pred.npy"
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
            print(f"tabm fold {fold}: reusing completed checkpoint {pred_path}", flush=True)
            fold_summaries.append(
                {"fold": fold, "status": "SKIPPED_COMPLETE", "meta": str(meta_path)}
            )
            assemble_oof(LEARNER, d, out_root)
            continue

        t0 = time.time()
        Xtr_raw = stack_features(mats, names, tr_rows, keep)
        ytr_all = d.y[tr_rows].astype(np.int64)
        Xva_raw = stack_features(mats, names, va_rows, keep)
        Xtr, Xva, prep_seconds, prep_mode = preprocess_arrays(
            Xtr_raw, Xva_raw, config, int(config.get("seed", SEED)) + fold
        )
        del Xtr_raw, Xva_raw
        inner_train_rows, inner_val_rows = inner_train_val_split(
            tr_rows,
            d.y,
            int(config.get("seed", SEED)) + 1000 + fold,
            float(config["training"].get("inner_val_fraction", 0.05)),
            int(config["training"].get("inner_val_max_rows", 100_000)),
        )
        val_mask = np.isin(tr_rows, inner_val_rows)
        model, train_meta = fit_tabm(
            Xtr[~val_mask],
            ytr_all[~val_mask],
            Xtr[val_mask] if val_mask.any() else None,
            ytr_all[val_mask] if val_mask.any() else None,
            config,
            torch_mod,
            device,
        )
        pred = predict_tabm(model, Xva, config, torch_mod, device)
        fold_meta = {
            **manifest,
            "fold": int(fold),
            "train_rows": int(len(tr_rows)),
            "gradient_train_rows": int((~val_mask).sum()),
            "inner_early_stop_rows": int(val_mask.sum()),
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
        del Xtr, Xva, pred, model
        clear_memory(torch_mod)
        assembly = assemble_oof(LEARNER, d, out_root)
        print(
            f"tabm fold {fold}: checkpointed; completed={assembly['completed_folds']}", flush=True
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
