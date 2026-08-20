#!/usr/bin/env python3
"""Run only the 2025 offline/known-boundary benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path

from oracle_information_frontier import default_2025_data_path, run_2025_benchmark


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data2025", default=default_2025_data_path())
    parser.add_argument("--n-estimators", type=int, default=500)
    args = parser.parse_args()
    if not args.data2025:
        raise SystemExit("2025 data not found; pass --data2025")
    run_2025_benchmark(
        args.data2025,
        root / "research/reports/benchmark_2025_vs_2026",
        n_estimators=args.n_estimators,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
