"""W6-E2: what is knowing tau actually worth to OUR model?

Pre-registered in research/WAVE6_PREREG.md section 4.

ORACLE / DIAGNOSTIC -- NOT DEPLOYABLE.  RT-900 trains on a block that uses the
true break location (and a placebo cut for no-break series, so the block does
not encode the label).  It can never ship; it exists to decide what W6-E3 is.

    RT-900 - RT-300 at age 100+  >=  +0.010  ->  knowing tau IS the lever
                                                 -> W6-E3 = causal localisation
    RT-900 - RT-300 at age 100+  <   +0.010  ->  tau knowledge is NOT the gap
                                                 -> the remaining headroom, if any,
                                                    is in the REPRESENTATION
                                                 -> W6-E3 = model family / neural
"""
from __future__ import annotations

import os, sys, time

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

from sbr.pipeline import run
from wave2_lib import CHAMP, FULL

if __name__ == "__main__":
    t0 = time.time()
    run(exp_id="RT-900", modules=FULL + ["w6oracle"], agent="claude-wave6", seed=0,
        hypothesis="ORACLE DIAGNOSTIC. Handing the champion's own 500-column engine the "
                   "true break location (placebo cut for no-break series) prices the "
                   "LOCALISATION lever: it separates 'the oracle frontier's FULL-horizon "
                   "headroom is about knowing tau' from 'it is about better features in "
                   "that regime'. These imply opposite wave-6 programmes.",
        falsification="n/a -- this is a diagnostic and can never be promoted. The DECISION "
                      "rule is pre-registered: >= +0.010 at post-break age 100+ means "
                      "localisation is the lever; below that it is not.",
        notes="W6-E2 ORACLE / DIAGNOSTIC -- NOT DEPLOYABLE. Uses tau. w6oracle is not a "
              "registered feature module and cannot reach a production manifest.",
        **CHAMP)
    print(f"\nRT-900 done in {(time.time()-t0)/60:.1f} min", flush=True)
