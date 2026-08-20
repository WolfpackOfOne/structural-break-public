"""Wave-3 queue A: the environment anchor and the backward-CUSUM experiment.

One launch, three runs, no supervision in between (see the supervision trap in
HANDOFF_WAVE3 section 7).

  RT-300  reproduction of RT-100 on macOS/arm64            CHAMP protocol, 1M rows
  RT-301  matched control, 7 production modules            ABL protocol, 400k rows
  RT-302  RT-301 + m09_back                                ABL protocol, 400k rows

RT-301/RT-302 are the paired arms: identical folds, rows, seed and parameters,
run in the same session, differing only in the presence of one module.  Both are
pre-registered in research/RDOF_LEDGER.md with two falsification conditions.
"""
from __future__ import annotations

import os, sys, time

ROOT = os.environ.get("SBR_ROOT", "/home/claude/sb")
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

from sbr.pipeline import run
from wave2_lib import ABL, CHAMP, FULL

SECTION = sys.argv[1] if len(sys.argv) > 1 else "all"
NEW = "m09_back"


def anchor():
    """RT-300: is this machine's champion the same number as the ledger's?"""
    run(exp_id="RT-300", modules=FULL, agent="claude-wave3", seed=0,
        hypothesis="RT-100 reproduces on macOS/arm64 from a clean checkout to within seed-scale noise",
        falsification="|mean OOF - 0.615103| > 0.0050, i.e. the environment is not comparable "
                      "and every wave-3 delta must be re-based",
        notes="WAVE-3 environment anchor; identical config to RT-100/RT-100R; num_threads=2 as in wave 2",
        **CHAMP)


def backcusum():
    """RT-301/RT-302: does suffix-vs-prefix evidence add anything to the bank?"""
    run(exp_id="RT-301", modules=FULL, agent="claude-wave3", seed=0,
        hypothesis="Matched control for the m09_back test: 7 production modules at the ABL protocol",
        falsification="n/a (control)",
        notes="WAVE-3 ABL control for RT-302", **ABL)
    run(exp_id="RT-302", modules=FULL + [NEW], agent="claude-wave3", seed=0,
        hypothesis="Backward suffix-vs-prefix two-sample evidence detects late breaks that the "
                   "one-sample trailing-window bank cannot, because it differences out the "
                   "series-level offset between the online segment and the historical null",
        falsification="mean ABL delta vs RT-301 <= 0, OR the delta is positive on <= 2 of 5 folds",
        notes="WAVE-3 treatment: RT-301 + m09_back (51 cols, suffix-vs-prefix contrast)", **ABL)


if __name__ == "__main__":
    t0 = time.time()
    if SECTION in ("all", "anchor"):
        anchor()
    if SECTION in ("all", "backcusum"):
        backcusum()
    print(f"\nQUEUE A COMPLETE in {(time.time()-t0)/60:.1f} min", flush=True)
