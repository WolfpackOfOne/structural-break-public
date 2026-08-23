"""Stage D: promote a surviving block to a CHAMP-protocol ENSEMBLE STREAM.

Only blocks that cleared stage C -- i.e. whose two-model ABL blend beat the
RT-303 seed clone's -- reach here.  Stage C outcomes, from
research/reports/wave5_abl_blocks.json:

    RT-730  m11_focus   +0.00055 vs the seed clone, 4/5   SURVIVES
    RT-740  m10_persist -0.00041 vs the seed clone, 2/5   REJECTED, no stage D
    RT-750  m12_rdep    +0.00141 vs the seed clone, 4/5   SURVIVES

Each stage-D arm is the CHAMP configuration -- the champion's own protocol,
seed 0 -- over the seven production modules PLUS the block, so it differs from
RT-300 by the new columns and nothing else.  Its matched control is RT-401: the
champion configuration at seed 1, which differs from RT-300 by the seed and
nothing else and therefore carries no new information at all.

    RT-731  + m11_focus
    RT-751  + m12_rdep
    RT-761  + all three blocks   (only if RT-760 clears stage C)
"""
from __future__ import annotations

import os, sys, time

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

from sbr.pipeline import run
from wave2_lib import CHAMP, FULL

JOBS = {
    "RT-731": (["m11_focus"],
               "Exact maximisation over the candidate change point adds information an "
               "eighth ensemble member cannot get from a seed change."),
    "RT-751": (["m12_rdep"],
               "Residualised distribution distances, residual CUSUM/CUSUMSQ paths and a "
               "dependence likelihood ratio with the innovation variance profiled out add "
               "information an eighth ensemble member cannot get from a seed change."),
    "RT-761": (["m10_persist", "m11_focus", "m12_rdep"],
               "The blocks target different mechanisms and different break ages, so their "
               "union should carry more information than either alone."),
}


def main(which):
    for exp in which:
        mods, hyp = JOBS[exp]
        t0 = time.time()
        run(exp_id=exp, modules=FULL + mods, agent="claude-wave5", seed=0,
            hypothesis=hyp,
            falsification="(S + this stream) - (S + RT-401) < +0.0030, or positive on < 4/5 "
                          "canonical folds; RT-401 is the champion configuration at seed 1 and "
                          "carries no new information",
            notes=f"W5 stage D: CHAMP protocol, FULL + {'+'.join(mods)}; matched control RT-401",
            **CHAMP)
        print(f"[{exp}] {time.time()-t0:.0f}s\n", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
