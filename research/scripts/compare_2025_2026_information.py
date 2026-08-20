#!/usr/bin/env python3
"""Regenerate the 2025-vs-2026 markdown comparison from existing JSON outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from oracle_information_frontier import write_comparison_report


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-json",
        default=str(root / "research/reports/oracle_information_frontier_2026.json"),
    )
    parser.add_argument(
        "--benchmark-json",
        default=str(root / "research/reports/benchmark_2025_vs_2026.json"),
    )
    parser.add_argument(
        "--out",
        default=str(root / "research/reports/benchmark_2025_vs_2026.md"),
    )
    args = parser.parse_args()
    result_2026 = json.load(open(args.oracle_json)) if Path(args.oracle_json).exists() else None
    result_2025 = json.load(open(args.benchmark_json)) if Path(args.benchmark_json).exists() else None
    write_comparison_report(result_2026, result_2025, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
