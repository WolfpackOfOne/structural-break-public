"""Wave-3 queue B: the negative control the RT-302 ensemble claim needs.

RT-302's standalone delta is +0.00156 with a paired-bootstrap CI of
[-0.0019, +0.0049] -- straddling zero.  Its deployable logit-blend delta,
however, is +0.00371 over the control, which under VALIDATION_V2 section 7 is
the alternative promotion route.

That route is only valid if the blend gain is attributable to `m09_back` rather
than to the generic variance reduction any two-model average buys.  So:

  RT-303  the SAME 7 modules at the SAME protocol with seed=1 -- a stream that
          contains no new information whatsoever, only a different bagging and
          feature-sampling draw.

If blending RT-301 with its own seed-clone buys as much as blending it with
RT-302, the module's ensemble delta is not evidence for the module and the
promotion route closes.  Pre-registered before running.
"""
from __future__ import annotations

import os, sys, time

ROOT = os.environ.get("SBR_ROOT", "/home/claude/sb")
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

from sbr.pipeline import run
from wave2_lib import ABL, FULL

if __name__ == "__main__":
    t0 = time.time()
    run(exp_id="RT-303", modules=FULL, agent="claude-wave3", seed=1,
        hypothesis="NEGATIVE CONTROL for the RT-302 ensemble claim: a seed-clone of RT-301 "
                   "carries no new information, so blending it with RT-301 measures the generic "
                   "two-model averaging gain that RT-302's blend delta must beat",
        falsification="the RT-301+RT-303 logit blend gains as much as the RT-301+RT-302 blend "
                      "(within 0.0005), i.e. m09_back's ensemble delta is not attributable to the module",
        notes="WAVE-3 negative control: identical modules/protocol to RT-301, seed 1 only",
        **ABL)
    print(f"\nQUEUE B COMPLETE in {(time.time()-t0)/60:.1f} min", flush=True)
