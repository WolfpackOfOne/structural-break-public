"""WAVE-4 experiment definitions.  Pre-registered in research/RDOF_LEDGER.md
BEFORE any of them was run.  Never edits sbr core files.

Two seven-member sets are defined here and nothing else.  They exist to answer
one question: is the deployable seven-stream ensemble's gain over the single
champion bought by SPECIALIST DIVERSITY, or by ORDINARY BAGGING?

  SPECIALIST set  the wave-2 streams, configurations taken verbatim from
                  research/scripts/wave2_streams.py (which is itself the
                  manifest the Crunch-tested RT-150 artifact was built from)
  SEED set        the champion configuration seven times, changing ONLY the
                  seed, from a seed list fixed before the first run

Member 1 of BOTH sets is the champion configuration at seed 0, which on this
machine is already on disk as RT-300.  The sets are therefore maximally paired:
they share a member, the folds, the feature cache and the protocol.
"""
from __future__ import annotations

import os, sys

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

from wave2_lib import CHAMP, FULL
from wave2_streams import JOBS

#: PRE-REGISTERED SEED LIST.  Fixed 2026-08-20 before the first wave-4 run.
#: Seven seeds, no substitutions, no "best seven of ten".
SEEDS = [0, 1, 7, 42, 2026, 31415, 271828]

#: seed clones: champion protocol, champion modules, ONLY the seed varies.
SEED_CLONES = {
    "RT-300": 0,        # already on disk -- the champion itself
    "RT-401": 1,
    "RT-402": 7,
    "RT-403": 42,
    "RT-404": 2026,
    "RT-405": 31415,
    "RT-406": 271828,
}

#: specialist streams: RT-300 is the wave-2 RT-100R configuration exactly, and
#: RT-410..RT-415 are wave-2 RT-120R..RT-125R re-run on this machine.
SPECIALIST_ALIAS = {
    "RT-300": "RT-100R",
    "RT-410": "RT-120R",
    "RT-411": "RT-121R",
    "RT-412": "RT-122R",
    "RT-413": "RT-123R",
    "RT-414": "RT-124R",
    "RT-415": "RT-125R",
}

SPECIALISTS = {new: dict(JOBS[old]) for new, old in SPECIALIST_ALIAS.items() if old != "RT-100R"}

SEED_SET = list(SEED_CLONES)
SPECIALIST_SET = list(SPECIALIST_ALIAS)


def job(exp_id):
    """Return the kwargs for one wave-4 run."""
    if exp_id in SPECIALISTS:
        j = dict(SPECIALISTS[exp_id])
        note = j.pop("note")
        old = SPECIALIST_ALIAS[exp_id]
        return dict(
            exp_id=exp_id, agent="claude-wave4", folds=(0, 1, 2, 3, 4),
            hypothesis=f"macOS/arm64 reconstruction of the wave-2 specialist stream {old}, "
                       f"config verbatim from research/scripts/wave2_streams.py, so that the "
                       f"specialist ensemble can be scored against a seed-clone ensemble on ONE platform",
            falsification="n/a (reconstruction, not a candidate); the ensemble comparison it feeds "
                          "is falsified if the seed-clone ensemble is not beaten by more than 0.0030",
            notes=f"WAVE-4 specialist stream, alias of {old}; {note}", **j)
    if exp_id in SEED_CLONES:
        return dict(
            exp_id=exp_id, modules=FULL, agent="claude-wave4", seed=SEED_CLONES[exp_id],
            hypothesis="A seed clone of the champion carries no new information, so a seven-way "
                       "seed-clone ensemble measures the BAGGING component of the seven-stream "
                       "ensemble's gain",
            falsification="n/a (control arm)",
            notes=f"WAVE-4 seed clone, seed {SEED_CLONES[exp_id]}, champion protocol/modules",
            **CHAMP)
    raise KeyError(exp_id)


if __name__ == "__main__":
    from sbr.pipeline import run
    for e in sys.argv[1:]:
        run(**job(e))
