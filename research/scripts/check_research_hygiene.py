#!/usr/bin/env python3
"""Lightweight sanity checks for research/RESULTS.csv.

Deliberately minimal: duplicate experiment IDs and a malformed header are
the two failure modes that have actually bitten a multi-agent research
process (an ID collision silently overwrites the meaning of an earlier
row). Exits non-zero on failure so it can gate CI.
"""
import csv
import sys
from pathlib import Path

RESULTS_CSV = Path(__file__).resolve().parents[1] / "RESULTS.csv"
REQUIRED_COLUMNS = {"experiment_id", "date", "status"}


def main() -> int:
    if not RESULTS_CSV.exists():
        print(f"FAIL: {RESULTS_CSV} not found")
        return 1

    with RESULTS_CSV.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            print(f"FAIL: RESULTS.csv is missing required columns: {sorted(missing)}")
            return 1

        seen: dict[str, int] = {}
        for row in reader:
            eid = (row.get("experiment_id") or "").strip()
            if not eid:
                continue
            seen[eid] = seen.get(eid, 0) + 1

    dupes = {eid: count for eid, count in seen.items() if count > 1}
    if dupes:
        print(f"FAIL: duplicate experiment_id values in RESULTS.csv: {dupes}")
        return 1

    print(f"OK: RESULTS.csv has {sum(seen.values())} experiment rows, no duplicate IDs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
