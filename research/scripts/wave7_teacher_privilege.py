"""WAVE-7 W7-D2 -- how much does a full-sequence teacher actually KNOW that the
causal student does not?

LANE A's premise is that an offline teacher can see the rest of the series and
distil that into a prefix-only student.  Its ceiling is therefore bounded by the
amount of series the student has NOT yet seen, at the places where the metric
puts its weight.  If in the high-mass cells the student already holds 90% of the
online segment, the teacher has almost nothing to teach there and Lane A cannot
pay for itself no matter how good the distillation is.

This is exactly the section-31 question -- what is the maximum plausible gain if
this mechanism were PERFECT -- asked before building the mechanism.

EXACT.  Everything here comes from research/folds/folds.parquet: true tau, true
n_online, true fold.  No model, no store, no prediction.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wave7_alpha_budget import (AGE_EDGES, AGE_NAMES, T_EDGES, T_NAMES,  # noqa: E402
                                weight_geometry)

OUT = f"{ROOT}/research/reports/wave7_teacher_privilege.json"


def main() -> int:
    meta = pd.read_parquet(f"{ROOT}/research/folds/folds.parquet")
    dev = meta[meta["split"] == "dev"]
    geo = weight_geometry(dev)
    nneg = geo["nneg"]
    W = geo["W"]

    n_on = dev["n_online"].to_numpy(np.int64)
    tau = dev["tau_index"].to_numpy(np.int64)
    br = dev["has_break"].to_numpy(np.int64) == 1

    # Accumulators over POSITIVE rows, weighted by that row's pair mass nneg(t).
    cells = {tn: {an: {"w": 0.0, "unseen": 0.0, "unseen_frac": 0.0,
                       "seen_post": 0.0} for an in AGE_NAMES} for tn in T_NAMES}
    tot = {"w": 0.0, "unseen": 0.0, "unseen_frac": 0.0}

    for s in np.flatnonzero(br):
        N, tv = int(n_on[s]), int(tau[s])
        ts = np.arange(tv, N)
        if ts.size == 0:
            continue
        w = nneg[ts].astype(np.float64)
        unseen = (N - 1 - ts).astype(np.float64)        # points after t
        frac = unseen / float(N)
        seen_post = (ts - tv + 1).astype(np.float64)    # post-break points held
        ab = np.digitize(ts - tv, AGE_EDGES[1:-1], right=False)
        tb = np.digitize(ts, T_EDGES[1:-1], right=False)
        for i, an in enumerate(AGE_NAMES):
            for j, tn in enumerate(T_NAMES):
                m = (ab == i) & (tb == j)
                if not m.any():
                    continue
                c = cells[tn][an]
                c["w"] += float(w[m].sum())
                c["unseen"] += float((w[m] * unseen[m]).sum())
                c["unseen_frac"] += float((w[m] * frac[m]).sum())
                c["seen_post"] += float((w[m] * seen_post[m]).sum())
        tot["w"] += float(w.sum())
        tot["unseen"] += float((w * unseen).sum())
        tot["unseen_frac"] += float((w * frac).sum())

    for tn in T_NAMES:
        for an in AGE_NAMES:
            c = cells[tn][an]
            if c["w"] > 0:
                for k in ("unseen", "unseen_frac", "seen_post"):
                    c[k] /= c["w"]
            # Every unseen point of a POSITIVE row is post-break, so the teacher's
            # advantage on the detection question is exactly how many more times
            # the post-break regime it gets to observe.
            c["privilege_ratio"] = (
                (c["seen_post"] + c["unseen"]) / c["seen_post"] if c["seen_post"] else float("nan"))
            c["weight_share"] = c["w"] / W

    overall = {"mean_unseen_points": tot["unseen"] / tot["w"],
               "mean_unseen_fraction": tot["unseen_frac"] / tot["w"],
               "weighted_mean_privilege_ratio": sum(
                   cells[tn][an]["weight_share"] * cells[tn][an]["privilege_ratio"]
                   for tn in T_NAMES for an in AGE_NAMES if cells[tn][an]["w"] > 0)}

    # Privilege-weighted mass: how much of the LOSS sits where the teacher still
    # holds a material amount of future.  "material" is fixed here at >= 25% of
    # the online segment unseen, declared before looking at the answer.
    THRESH = 0.25
    mass_material = sum(cells[tn][an]["weight_share"]
                        for tn in T_NAMES for an in AGE_NAMES
                        if cells[tn][an]["unseen_frac"] >= THRESH)

    out = {"schema": "wave7_teacher_privilege/1", "threshold_unseen_fraction": THRESH,
           "overall": overall, "weight_share_with_material_privilege": mass_material,
           "cells": cells}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)

    print("=" * 78)
    print("W7-D2  TEACHER PRIVILEGE -- unseen future at the metric's weight")
    print("=" * 78)
    print(f"weighted mean unseen points  {overall['mean_unseen_points']:8.1f}")
    print(f"weighted mean unseen frac    {overall['mean_unseen_fraction']:8.3f}")
    print(f"weight share with >= {THRESH:.0%} of the online segment still unseen: "
          f"{mass_material:.1%}")

    print(f"weighted mean privilege ratio {overall['weighted_mean_privilege_ratio']:8.2f}x")

    for lbl, key, fmt in (("TEACHER PRIVILEGE RATIO  (post-break points teacher/student)", "privilege_ratio", "{:8.1f}"),
                          ("MEAN UNSEEN FRACTION OF THE ONLINE SEGMENT", "unseen_frac", "{:8.2f}"),
                          ("MEAN UNSEEN POINTS", "unseen", "{:8.0f}"),
                          ("MEAN POST-BREAK POINTS ALREADY HELD", "seen_post", "{:8.0f}")):
        print(f"\n--- {lbl}   (blank = no weight) ---")
        print(f"{'t \\ age':>10}" + "".join(f"{a:>9}" for a in AGE_NAMES))
        for tn in T_NAMES:
            cs = []
            for an in AGE_NAMES:
                c = cells[tn][an]
                cs.append("        ." if c["w"] == 0 else fmt.format(c[key]))
            print(f"{tn:>10}" + "".join(cs))

    print("\n--- WEIGHT SHARE x PRIVILEGE, the two cells that own half the loss ---")
    for tn in ("200-400", "400-1000"):
        c = cells[tn]["100+"]
        print(f"  t {tn:>9} x age 100+ : weight {100*c['weight_share']:5.2f}%  "
              f"unseen {c['unseen']:5.0f} pts ({c['unseen_frac']:.0%})  "
              f"post-break already held {c['seen_post']:5.0f} pts  "
              f"privilege {c['privilege_ratio']:.2f}x")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
