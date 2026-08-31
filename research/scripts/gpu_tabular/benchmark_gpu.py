#!/usr/bin/env python
"""Stage A hardware benchmark for GPU-01 TabM and GPU-02 RealMLP.

This script measures preprocessing, per-epoch training time, throughput, RAM,
and VRAM. It does not compute TS-AUC or any replacement score. On the RTX host,
it writes/updates research/reports/gpu_tabular_2026/FROZEN_GPU_CONFIG.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import realmlp_runner
import tabm_runner
from common import (
    BENCHMARK_JSON,
    FROZEN_CONFIG,
    LOCAL_OOF_ROOT,
    atomic_write_json,
    load_json,
    update_frozen_config,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--learner", choices=("tabm", "realmlp"), required=True)
    p.add_argument("--artifact-root", default=None)
    p.add_argument("--output-root", default=str(LOCAL_OOF_ROOT))
    p.add_argument("--config", default=None)
    p.add_argument("--benchmark-log", default=None)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--max-train-rows", type=int, default=1_000_000)
    p.add_argument("--benchmark-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--quota-hours", type=float, default=15.0)
    p.add_argument("--quota-learners", type=int, default=2)
    p.add_argument("--quota-fraction", type=float, default=0.90)
    p.add_argument("--allow-cpu-smoke", action="store_true")
    p.add_argument("--synthetic-smoke", action="store_true")
    p.add_argument("--no-write-config", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.learner == "tabm":
        entry = tabm_runner.benchmark(args)
    else:
        entry = realmlp_runner.benchmark(args)

    config_path = Path(args.config).resolve() if args.config else FROZEN_CONFIG
    wrote_config = False
    if not args.no_write_config and (not args.synthetic_smoke or args.config):
        data = update_frozen_config(config_path, args.learner, entry)
        wrote_config = True
    else:
        data = {"learners": {args.learner: entry}}

    if args.benchmark_log:
        log_path = Path(args.benchmark_log).resolve()
    elif args.synthetic_smoke and args.config:
        log_path = config_path.parent / "benchmark_results.json"
    elif args.synthetic_smoke:
        log_path = None
    else:
        log_path = BENCHMARK_JSON
    if log_path is not None:
        benchmark_log = load_json(log_path) if log_path.exists() else {"learners": {}}
        benchmark_log["learners"][args.learner] = entry
        atomic_write_json(log_path, benchmark_log)

    print(
        json.dumps(
            {
                "learner": args.learner,
                "status": entry["status"],
                "score_computed": False,
                "config_path": str(config_path),
                "config_written": wrote_config,
                "benchmark_log": str(log_path) if log_path is not None else None,
                "learners_in_config": sorted(data.get("learners", {}).keys()),
                "selected_projection_hours": entry["selected_runtime_projection"][
                    "projected_5fold_runtime_hours"
                ],
                "selected_max_epochs": entry["selected_config"]["training"]["max_epochs"],
                "selected_batch_size": entry["selected_config"]["training"]["batch_size"],
                "peak_vram_gb": entry["peak_vram_gb"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
