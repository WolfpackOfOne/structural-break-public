"""Alternate-partition read on the strongest surviving block, at the ABL protocol.

Section 4 of the pre-registration requires alternate partitions to be positive
or at minimum directionally stable, and no wave-5 candidate had that evidence.
The wave-3/wave-5 ABL control arms exist only on the canonical partition, so all
three arms are re-run here on the alternate draw; nothing is compared across
partitions except the DELTA, which is the quantity at risk (W4-E2's finding:
levels move ~0.009 across partitions while deltas move ~0.003).

ROBUSTNESS DIAGNOSTIC ONLY -- the partitions select nothing.

Usage:  wave5_alt_blocks.py <alt1|alt2|alt3>
"""
from __future__ import annotations

import os, sys, time

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

from sbr.pipeline import run
from wave2_lib import ABL, FULL, alt_folds

#: id suffixed by the partition -> (modules, seed, what it is)
ARMS = [
    ("RT-301", FULL, 0, "control, 7 production modules"),
    ("RT-303", FULL, 1, "seed clone of the control -- carries no new information"),
    ("RT-750", FULL + ["m12_rdep"], 0, "the candidate block"),
]


def main(part):
    assert part in ("alt1", "alt2", "alt3")
    for base, mods, seed, what in ARMS:
        t0 = time.time()
        with alt_folds(part):
            run(exp_id=f"{base}.{part}", modules=mods, agent="claude-wave5", seed=seed,
                hypothesis="m12_rdep's advantage over a same-strength seed clone is a property "
                           "of the method, not of the canonical fold draw",
                falsification="the candidate's blend margin over the seed clone's blend is "
                              "negative on an alternate partition",
                notes=f"WAVE-5 partition study, partition={part}, ABL protocol, {what}. "
                      f"ROBUSTNESS DIAGNOSTIC ONLY -- never used to select anything.",
                **ABL)
        print(f"[{base}.{part}] {time.time()-t0:.0f}s\n", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
