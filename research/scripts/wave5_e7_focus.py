"""W5-E7: does EXACT maximisation over the candidate change point add anything?

Pre-registered in research/WAVE5_PREREG.md section 6.

STAGE C -- the paired ABL arm, deliberately the SAME protocol wave 3 used for
m09_back, so the two feature families are directly comparable:

    RT-301   control, 7 production modules, ABL 400k        (already on disk)
    RT-303   seed clone of the control, seed 1, ABL 400k    (already on disk)
    RT-730   RT-301 + m11_focus                             <- the candidate

STAGE D/E happen only if stage C survives:

    RT-731   CHAMP protocol, FULL + m11_focus, seed 0 -- the 8th ensemble stream.
             Its matched control is RT-401 (CHAMP, FULL, seed 1): a stream that
             differs from RT-300 only by its seed and therefore carries no new
             information at all.  `S + RT-731` against `S + RT-401` is the
             comparison section 4 requires; `S + RT-731` against `S` is not.
"""
from __future__ import annotations

import os, sys, time

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

from sbr.pipeline import run
from wave2_lib import ABL, CHAMP, FULL

NEW = "m11_focus"

HYP = ("The incumbent bank never maximises over the candidate change point: m01_seq's "
       "GLR uses a dyadic grid (its own docstring bounds the loss at sqrt(2) in effective "
       "segment length) and m06_loc uses a geometric bank of trailing windows. An EXACT "
       "max over tau of the normalised likelihood improvement, calibrated against a "
       "per-series historical null at matched elapsed length, should recover evidence the "
       "fixed-window bank dilutes -- most of all at small post-break age.")


def stage_c():
    run(exp_id="RT-730", modules=FULL + [NEW], agent="claude-wave5", seed=0,
        hypothesis=HYP,
        falsification="mean ABL delta vs RT-301 <= 0, OR positive on <= 2 of 5 folds, OR "
                      "the deployable blend delta does not beat the RT-303 seed clone's",
        notes=f"W5-E7 stage C: RT-301 + {NEW} (76 cols, exact max over tau, 6 channels); "
              f"ABL protocol, identical arms to the wave-3 m09_back test", **ABL)


def stage_d():
    run(exp_id="RT-731", modules=FULL + [NEW], agent="claude-wave5", seed=0,
        hypothesis=HYP + " As an 8th ensemble stream it must beat a same-strength seed clone.",
        falsification="S+RT-731 minus S+RT-401 < +0.0030, or positive on < 4/5 folds",
        notes=f"W5-E7 stage D: CHAMP protocol, FULL + {NEW}; matched control is RT-401",
        **CHAMP)


if __name__ == "__main__":
    t0 = time.time()
    for s in (sys.argv[1:] or ["c"]):
        {"c": stage_c, "d": stage_d}[s]()
    print(f"\nW5-E7 done in {(time.time()-t0)/60:.1f} min", flush=True)
