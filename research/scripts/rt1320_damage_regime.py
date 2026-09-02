"""Where does the RT-1320 student do damage, across every partition and fold?

Phase 1 §1.5 only required characterising a damage regime *if* fold 0 came out
negative on multiple partitions. It did not (canonical alone), so the conditional
never fired and the gate is satisfied. But the decomposition that was computed
anyway showed something worth chasing: canonical's fold-0 loss sits in
**never-break negatives** (-0.001419 vs -0.000154 pre-break) and concentrates at
**200 <= t < 400** (-0.003206) -- which is exactly W7-D0's dominant remaining-loss
region, ~45% of total pairwise inversion loss.

That is one cell of one fold of one partition. This script asks whether it is a
property of the mechanism that happens to be outweighed elsewhere, or genuinely
local to canonical fold 0.

The question matters because RT-1320 is licensed but unpromoted. If the student
systematically damages never-break negatives in the highest-value horizon band
and merely wins on net, that is a targeted repair opportunity -- and a known
weakness to watch if the external score disappoints.

Method: reuses rt1320_fold0_diagnosis's blend construction, so E1 and E2 are the
same quantities as the headline endpoint. mean_fold_ts_auc already returns every
fold, so sweeping all 20 (partition, fold) cells costs no more than the fold-0
run did -- it just stops discarding four fifths of the result.

Nothing here gates anything. It is exploratory.

Usage:
    python research/scripts/rt1320_damage_regime.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "research" / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import armc_residual_student as A  # noqa: E402
import rt1320_fold0_diagnosis as D  # noqa: E402

PARTITIONS = ["canonical", "alt1", "alt2", "alt3"]
OUT = REPO / "research" / "reports" / "rt1320_promotion" / "PHASE1_DAMAGE_REGIME.json"


def per_fold_delta(pack, mask) -> dict[str, float | None]:
    """E2-E1 on every fold, restricted to `mask`."""
    e1 = A.mean_fold_ts_auc(pack["E1"], pack["rows"], mask=mask)["per_fold_ts_auc"]
    e2 = A.mean_fold_ts_auc(pack["E2"], pack["rows"], mask=mask)["per_fold_ts_auc"]
    out: dict[str, float | None] = {}
    for f in sorted(e1):
        a, b = e1[f], e2[f]
        out[f] = (float(b - a)
                  if a is not None and b is not None
                  and np.isfinite(a) and np.isfinite(b) else None)
    return out


def sweep(pack) -> dict:
    rows = pack["rows"]
    dev, t = rows["dev"], rows["t"]
    res: dict = {"overall": per_fold_delta(pack, dev), "cells": {}}

    # The two negative classes crossed with horizon. Each mask keeps all positives
    # and admits only one negative class, so the contrast isolates that class.
    for cls, base in (("never_break", rows["dominant_never_break_only"]),
                      ("pre_break", rows["dominant_pre_break_only"])):
        for name, lo, hi in D.HORIZON_BUCKETS:
            m = base & (t >= lo) & (t < hi)
            n = int(m.sum())
            if n == 0:
                continue
            res["cells"][f"{cls} | {name}"] = {
                "n_rows": n,
                "per_fold": per_fold_delta(pack, m),
            }
    return res


def main() -> int:
    results: dict = {}
    for part in PARTITIONS:
        print(f"=== {part} ===", flush=True)
        pack, err = D.build_blends(part, D.student_dir_for(part, None))
        if pack is None:
            print(f"  SKIP: {err}", flush=True)
            continue
        results[part] = sweep(pack)
        ov = results[part]["overall"]
        print("  overall per fold: "
              + " ".join(f"{v:+.5f}" if v is not None else "  n/a"
                         for _, v in sorted(ov.items())), flush=True)
        del pack

    # Aggregate each cell over all (partition, fold) pairs that were computed.
    cells: dict[str, list[float]] = {}
    for r in results.values():
        for cell, d in r["cells"].items():
            for v in d["per_fold"].values():
                if v is not None:
                    cells.setdefault(cell, []).append(v)

    summary = []
    for cell, vals in cells.items():
        arr = np.array(vals, dtype=float)
        summary.append({
            "cell": cell,
            "n_partition_fold_cells": int(arr.size),
            "mean_delta": float(arr.mean()),
            "median_delta": float(np.median(arr)),
            "n_negative": int((arr < 0).sum()),
            "frac_negative": float((arr < 0).mean()),
            "worst": float(arr.min()),
            "best": float(arr.max()),
        })
    summary.sort(key=lambda d: d["mean_delta"])

    payload = {
        "question": ("is the never-break / 200<=t<400 loss seen on canonical fold 0 "
                     "a property of the mechanism, or local to that cell?"),
        "gates": False,
        "note": "Exploratory. Phase 1 §1.5's conditional did not fire; this is follow-up.",
        "partitions": sorted(results),
        "cell_summary_worst_first": summary,
        "per_partition": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print("\n=== cells ranked worst-first, over all partition/fold pairs ===")
    print(f"{'cell':34s} {'mean':>10s} {'worst':>10s} {'neg':>8s}")
    for s in summary:
        print(f"{s['cell']:34s} {s['mean_delta']:+10.6f} {s['worst']:+10.6f} "
              f"{s['n_negative']:3d}/{s['n_partition_fold_cells']:<4d}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
