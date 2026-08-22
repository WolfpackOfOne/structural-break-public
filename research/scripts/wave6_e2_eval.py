"""W6-E2 verdict: the value of knowing tau, by post-break age."""
from __future__ import annotations

import json, time
import numpy as np

from wave5_lib import Ctx, load_oof, REPORTS, fmt

if __name__ == "__main__":
    t0 = time.time()
    c = Ctx()
    P = load_oof(["RT-300", "RT-900"])
    base_m, base_pf = c.score(P["RT-300"])
    or_m, or_pf = c.score(P["RT-900"])
    d = [a - b for a, b in zip(or_pf, base_pf)]
    print("=== W6-E2: what knowing tau is worth to our own engine ===\n")
    print(f"  RT-300 legal champion   {base_m:.5f}   {fmt(base_pf)}")
    print(f"  RT-900 + true tau       {or_m:.5f}   {fmt(or_pf)}")
    print(f"  aggregate delta         {or_m-base_m:+.5f}  ({sum(x>0 for x in d)}/5)\n")

    ab, ao = c.score_by_age(P["RT-300"]), c.score_by_age(P["RT-900"])
    print(f"  {'age':>8s}{'RT-300':>12s}{'RT-900':>12s}{'delta':>11s}{'n_pos':>10s}")
    res = {"experiment": "W6-E2", "aggregate": {"RT-300": base_m, "RT-900": or_m,
                                                "delta": or_m - base_m}, "age": {}}
    for k in ab:
        a, b = ab[k]["ts_auc"], ao[k]["ts_auc"]
        res["age"][k] = {"RT-300": a, "RT-900": b, "delta": b - a, "n_pos": ab[k]["n_pos"]}
        print(f"  {k:>8s}{a:>12.5f}{b:>12.5f}{b-a:>+11.5f}{ab[k]['n_pos']:>10d}")

    mature = res["age"]["100-"]["delta"]
    res["decision"] = {
        "mature_delta_age_100plus": mature, "threshold": 0.010,
        "branch": ("LOCALISATION -- tau knowledge is the lever" if mature >= 0.010
                   else "REPRESENTATION -- tau knowledge is not the gap; W6-E3 = model family / neural"),
    }
    print(f"\n  age 100+ delta = {mature:+.5f}  vs threshold +0.010")
    print(f"  BRANCH: {res['decision']['branch']}")
    json.dump(res, open(f"{REPORTS}/wave6_e2_tau.json", "w"), indent=2, default=float)
    print(f"\nwrote research/reports/wave6_e2_tau.json  ({time.time()-t0:.0f}s)")
