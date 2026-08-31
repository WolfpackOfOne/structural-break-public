#!/usr/bin/env python
"""Protocol-correct primary endpoint for RT-1320 under the addition contract.

`PROTOCOL_CHAMPION_2026.md` defines the primary endpoint as E2-E1, where E1 is
RT-1257 with a *matched exchangeable control under the same contract*. Because
RT-1320 adds an 8th member, E1 must be RT-1257 plus one seed clone -- exactly
one change from E0.

The `rt1257_combo_analysis` block in `armc_residual_student.py` does not build
that. Its `clone_plus_seed_control` degrades three slots at once (it clones the
CAT-300 and CAT-413 members *and* adds a clone), so the delta against it is not
E2-E1; it is the RT-1264/CSA-04 shape the protocol explicitly names as an
inflation mode. This script computes the endpoint the protocol actually asks
for, reusing that module's own calibration and blending so nothing else moves.

Usage:
    python research/scripts/armc_e2_e1_addition_contract.py \
        --student-dir research/reports/armc_residual_student_confirm_s20260901
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "research" / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import armc_residual_student as A  # noqa: E402

CONTROL_STREAM = "RT-403"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student-dir", type=Path, required=True)
    ap.add_argument("--folds-path", type=Path,
                    default=REPO / "research" / "folds" / "folds.parquet")
    ap.add_argument("--noise-floor", type=float, default=0.0011)
    args = ap.parse_args()

    data_root = A.default_data_root()
    oof_dir = data_root / "research" / "oof"
    cat_dir = A.default_catboost_oof_dir()

    rows = A.build_row_arrays(args.folds_path)
    student_raw, _ = A.merge_student_oof(args.student_dir, rows, force=False)
    student = A.crossfit_calibrate(student_raw, rows)

    spec = [A.load_calibrated_stream(oof_dir, s, rows) for s in A.SPECIALISTS]
    cat300 = A.crossfit_calibrate(np.load(cat_dir / "RT-1255.npy", mmap_mode="r"), rows)
    cat413 = A.crossfit_calibrate(np.load(cat_dir / "RT-1254.npy", mmap_mode="r"), rows)
    control = A.load_calibrated_stream(oof_dir, CONTROL_STREAM, rows)

    base = [cat300, spec[1], spec[2], spec[3], cat413, spec[5], spec[6]]
    packs = {
        "E0": A.mean_fold_ts_auc(A.blend(base), rows),
        "E1": A.mean_fold_ts_auc(A.blend(base + [control]), rows),
        "E2": A.mean_fold_ts_auc(A.blend(base + [student]), rows),
    }
    pf = {k: v["per_fold_ts_auc"] for k, v in packs.items()}
    folds = sorted(pf["E0"])
    d21 = [pf["E2"][f] - pf["E1"][f] for f in folds]
    d20 = [pf["E2"][f] - pf["E0"][f] for f in folds]
    d10 = [pf["E1"][f] - pf["E0"][f] for f in folds]

    out = {
        "contract": "addition of an 8th exchangeable member to RT-1257",
        "E1_control_stream": f"{CONTROL_STREAM} seed clone",
        "means": {k: packs[k]["mean_ts_auc"] for k in ("E0", "E1", "E2")},
        "PRIMARY_E2_minus_E1": float(np.mean(d21)),
        "primary_per_fold": [float(x) for x in d21],
        "primary_folds_positive": int(sum(x > 0 for x in d21)),
        "SECONDARY_E2_minus_E0": float(np.mean(d20)),
        "secondary_per_fold": [float(x) for x in d20],
        "secondary_folds_positive": int(sum(x > 0 for x in d20)),
        "control_lift_E1_minus_E0": float(np.mean(d10)),
        "noise_floor_paired_bootstrap": args.noise_floor,
    }
    out["primary_clears_noise_floor"] = bool(out["PRIMARY_E2_minus_E1"] >= args.noise_floor)
    print(json.dumps(out, indent=2))
    (args.student_dir / "E2_E1_addition_contract.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
