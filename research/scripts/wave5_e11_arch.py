"""W5-E11: does m12_rdep improve the ARCHITECTURE, or only add a member?

Pre-registered in research/RDOF_LEDGER.md before any arm was trained.

Each arm is the EXACT incumbent specialist configuration -- verbatim from
research/scripts/wave2_streams.py, the manifest the Crunch-tested artifact was
built from -- with `m12_rdep` appended to its module list and nothing else
changed.  Same seed, same rows, same leaves, same feature_fraction, same
sampling, same objective, same boosting type.

S' is the equal-weight cross-fitted SCDF blend of the seven rebuilt streams; S
is the incumbent seven.  They differ by 57 columns per stream and by nothing
else -- no member count, no weight, no seed -- so there is no bagging channel
for a gain to arrive through.
"""
from __future__ import annotations

import os, sys, time

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

from sbr.pipeline import run
from wave2_streams import JOBS
from wave5_lib import SPECIALISTS  # noqa: F401  (documents the incumbent set)

NEW = "m12_rdep"

#: new id -> the wave-2 stream whose configuration it copies verbatim
REBUILD = {
    "RT-811": "RT-120R",
    "RT-812": "RT-121R",
    "RT-813": "RT-122R",
    "RT-814": "RT-123R",
    "RT-815": "RT-124R",
    "RT-816": "RT-125R",
}
#: RT-751 (champion configuration + m12_rdep) is declared under stage D and is
#: member 1 of S', exactly as RT-300 is member 1 of S.
MEMBER1 = "RT-751"

HYP = ("m12_rdep improves the architecture rather than adding an eighth member. "
       "An eighth exchangeable member is worth +0.00003 (W5-NULLTEST), so the "
       "eighth-member comparison cannot resolve a feature block; rebuilding every "
       "stream with the block available to it can.")


def main(which):
    for exp in which:
        old = REBUILD[exp]
        j = dict(JOBS[old])
        note = j.pop("note")
        t0 = time.time()
        j["modules"] = list(j["modules"]) + [NEW]
        run(exp_id=exp, agent="claude-wave5", folds=(0, 1, 2, 3, 4),
            hypothesis=HYP,
            falsification="S' - S < +0.0030 over the five canonical folds, or positive on "
                          "< 4/5 folds, or the paired bootstrap CI includes zero",
            notes=f"W5-E11: {old} configuration VERBATIM plus {NEW}; {note}",
            **j)
        print(f"[{exp}] rebuilt {old} + {NEW}  {time.time()-t0:.0f}s\n", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(REBUILD))
