"""W4-E2: does the ENSEMBLE DELTA survive the fold-partition draw?

Wave 2 measured the champion's ABSOLUTE sensitivity to the partition
(RT-221/222/223).  That is not the quantity at risk.  The claim under test is a
DELTA -- ensemble minus single model -- and a delta can be stable while levels
move, or vanish while levels look fine.  This measures the delta itself on
alt1, alt2 and alt3.

PROTOCOL NOTE, and it matters.  An earlier draft of this script forced every arm
onto the reduced ABL protocol to save compute.  That was wrong: ABL overrides
num_leaves, min_data_in_leaf and feature_fraction, which are exactly the knobs
that MAKE a specialist stream a specialist.  Under it RT-412 lost its 255 leaves
and RT-410 its 127, and every "specialist" collapsed towards the champion --
the experiment would have measured nothing.  Each stream therefore runs at its
OWN native configuration, identical to the canonical run, and only the fold
partition changes.  The canonical arm is not re-run here; the canonical numbers
come from the W4-E1 runs, which use these same configurations.

Candidates were declared in RDOF_LEDGER.md before any alternate score was read:
exactly three -- the single champion, the seven-way seed ensemble, the
seven-way specialist ensemble.  The alternate partitions select nothing; they
are a robustness diagnostic.

Usage:  wave4_partitions.py <alt1|alt2|alt3> <base-exp-id> [...]
"""
from __future__ import annotations

import os, sys

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

from sbr.pipeline import run
from wave2_lib import alt_folds
from wave4_lib import job

PARTS = ("alt1", "alt2", "alt3")


def main():
    part = sys.argv[1]
    assert part in PARTS, PARTS
    for base in sys.argv[2:]:
        kw = job(base)
        kw["exp_id"] = f"{base}.{part}"
        kw["notes"] = (f"W4-E2 partition study, partition={part}, NATIVE protocol "
                       f"(unchanged from the canonical run of {base}). ROBUSTNESS "
                       f"DIAGNOSTIC ONLY -- never used to select anything. " + kw["notes"])
        kw["hypothesis"] = ("The ensemble delta is a property of the method, not of the "
                            "canonical fold draw: " + kw["hypothesis"])
        kw["falsification"] = ("the ensemble delta is negative on any alternate partition, "
                               "or its across-partition SD exceeds its canonical mean")
        with alt_folds(part):
            run(**kw)


if __name__ == "__main__":
    main()
