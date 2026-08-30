"""Confirm the surprising leave-one-module-out results at the CHAMPION protocol.

The battery runs at 400,000 training rows for compute reasons. If a module's
value is partly "more columns for a model that already has enough data", then a
row-starved protocol will systematically UNDERSTATE module value -- the deltas
would be biased toward "this module is useless".

Wave 1 saw the mirror image of exactly this failure (three screen-level
architectural wins reversed at full scale, all because they helped a data-starved
model). So any module whose removal looks HARMLESS or HELPFUL at 400k rows must be
re-tested at 1,000,000 rows before anything is concluded, let alone deployed.

Usage: wave2_lomo_confirm.py m00_core m01_seq ...
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/home/claude/sb/src")
sys.path.insert(0, "/home/claude/sb/research/scripts")
from sbr.pipeline import run
from wave2_lib import CHAMP, FULL

TARGETS = sys.argv[1:] or ["m00_core", "m01_seq"]
IDS = {"m00_core": "RT-231", "m01_seq": "RT-232", "m02_dist": "RT-233", "m03_dyn": "RT-234",
       "m04_resid": "RT-235", "m06_loc": "RT-236", "m07_bayes": "RT-237"}

for m in TARGETS:
    keep = [x for x in FULL if x != m]
    run(exp_id=IDS[m], modules=keep, agent="ablation-confirm",
        hypothesis=f"The battery's finding for {m} survives the CHAMPION protocol "
                   f"(1M rows), i.e. it is a real module effect and not an artefact "
                   f"of a row-starved model preferring fewer columns",
        falsification=f"the sign of the delta vs RT-100R (0.615103) flips relative to "
                      f"the battery result for {m}",
        notes=f"CHAMPION-protocol leave-one-module-out: minus {m}; compare to RT-100R 0.615103",
        **CHAMP)
