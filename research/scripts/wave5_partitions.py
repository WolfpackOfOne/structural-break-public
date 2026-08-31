"""Alternate-partition robustness for a wave-5 candidate stream.

Same discipline as W4-E2: NATIVE protocol, only the fold partition changes, and
the partitions select nothing -- they are a robustness diagnostic. The canonical
arm is not re-run here.

The seven specialists and the RT-401 seed control already have alt1/alt2/alt3
OOF vectors on disk from wave 4, so only the candidate needs training, and the
partition comparison is then the same `(S+C) - (S+N)` contrast on a different
draw of series into folds.

Usage:  wave5_partitions.py <alt1|alt2|alt3> <candidate-exp-id> ...
"""
from __future__ import annotations

import os, sys

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

from sbr.pipeline import run
from wave2_lib import alt_folds, ABL, CHAMP, FULL

PARTS = ("alt1", "alt2", "alt3")

#: candidate id -> (modules, protocol)
CAND = {
    "RT-731": (FULL + ["m11_focus"], CHAMP),
    "RT-730": (FULL + ["m11_focus"], ABL),
    "RT-740": (FULL + ["m10_persist"], ABL),
    "RT-750": (FULL + ["m12_rdep"], ABL),
    "RT-760": (FULL + ["m10_persist", "m11_focus", "m12_rdep"], ABL),
}


def main():
    part = sys.argv[1]
    assert part in PARTS, PARTS
    for base in sys.argv[2:]:
        mods, proto = CAND[base]
        with alt_folds(part):
            run(exp_id=f"{base}.{part}", modules=mods, agent="claude-wave5", seed=0,
                hypothesis=f"The candidate's contribution is a property of the method, not of the "
                           f"canonical fold draw: {base} on partition {part}, native protocol",
                falsification="the candidate's delta over its matched seed control is negative on "
                              "an alternate partition, or its across-partition SD exceeds its "
                              "canonical mean",
                notes=f"WAVE-5 partition study, partition={part}, NATIVE protocol unchanged from "
                      f"the canonical run of {base}. ROBUSTNESS DIAGNOSTIC ONLY -- never used to "
                      f"select anything.",
                **proto)


if __name__ == "__main__":
    main()
