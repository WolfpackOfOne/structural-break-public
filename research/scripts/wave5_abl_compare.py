"""Stage-C/D verdict for the new feature blocks, at the wave-3 ABL protocol.

Reproduces exactly the comparison that rejected m09_back, so the new blocks are
judged by the standard that has already killed one candidate:

  1. paired delta of the treated arm against RT-301, per fold;
  2. the DEPLOYABLE two-model blend RT-301 + candidate, against the SAME blend
     built with RT-303 -- a seed clone of the control that contains no new
     information whatsoever.  Wave 3's lesson was that the clone blends BETTER
     than a real new module did, so this is the number that decides;
  3. TS-AUC by post-break age, because m09_back's aggregate gain turned out to
     be entirely mature-break and its young-break effect was negative.
"""
from __future__ import annotations

import json, sys, time
import numpy as np

from wave5_lib import Ctx, load_oof, REPORTS, fmt
from wave4_cal import logit, sigmoid

CONTROL = "RT-301"
SEEDCTL = "RT-303"


def main(cands):
    t0 = time.time()
    c = Ctx()
    need = [CONTROL, SEEDCTL] + list(cands)
    P = load_oof(need)
    res = {"control": CONTROL, "seed_control": SEEDCTL, "arms": {}}

    base_m, base_pf = c.score(P[CONTROL])
    print(f"=== ABL stage C/D  (protocol identical to the wave-3 m09_back test) ===")
    print(f"  {CONTROL} control     {base_m:.5f}   {fmt(base_pf)}")

    def blend(a, b):
        return sigmoid(0.5 * (logit(P[a]) + logit(P[b])))

    sc_v = blend(CONTROL, SEEDCTL)
    sc_m, sc_pf = c.score(sc_v)
    scl_m, scl_pf = c.score(P[SEEDCTL])
    print(f"  {SEEDCTL} seed clone  {scl_m:.5f}   {fmt(scl_pf)}")
    print(f"  blend {CONTROL}+{SEEDCTL} {sc_m:.5f}  vs control {sc_m-base_m:+.5f}   <- the bar")
    res["seed_clone_blend"] = {"mean": sc_m, "per_fold": sc_pf, "delta": sc_m - base_m}

    vecs = {"control": P[CONTROL], "seedblend": sc_v}
    for e in cands:
        m, pf = c.score(P[e])
        d = [x - y for x, y in zip(pf, base_pf)]
        bv = blend(CONTROL, e)
        bm, bpf = c.score(bv)
        vecs[e] = bv
        a = {"standalone": {"mean": m, "per_fold": pf, "delta_vs_control": m - base_m,
                            "per_fold_delta": d, "n_folds_better": int(sum(x > 0 for x in d))},
             "blend": {"mean": bm, "per_fold": bpf, "delta_vs_control": bm - base_m,
                       "delta_vs_seed_clone_blend": bm - sc_m,
                       "n_folds_better_than_seed_blend":
                           int(sum(x > y for x, y in zip(bpf, sc_pf)))}}
        a["verdict"] = ("beats the seed clone" if bm - sc_m > 0 else
                        "LOSES to a seed clone -- the blend gain is bagging")
        res["arms"][e] = a
        print(f"\n  {e}  standalone {m:.5f}  delta {m-base_m:+.5f} "
              f"({a['standalone']['n_folds_better']}/5)   {fmt(pf)}")
        print(f"  {' ':9s}blend      {bm:.5f}  vs control {bm-base_m:+.5f}   "
              f"vs SEED CLONE {bm-sc_m:+.5f} "
              f"({a['blend']['n_folds_better_than_seed_blend']}/5)  -> {a['verdict']}")

    print("\n  -- TS-AUC by post-break age (standalone arms) --")
    ages = {"control": c.score_by_age(P[CONTROL])}
    for e in cands:
        ages[e] = c.score_by_age(P[e])
    res["age"] = ages
    hdr = ["control"] + list(cands)
    print(f"  {'age':>8s}" + "".join(f"{h:>13s}" for h in hdr) +
          "".join(f"{'d '+h[-3:]:>10s}" for h in cands))
    for kk in list(ages["control"]):
        row = [ages[h][kk]["ts_auc"] for h in hdr]
        print(f"  {kk:>8s}" + "".join(f"{x:>13.5f}" for x in row) +
              "".join(f"{row[i+1]-row[0]:>+10.5f}" for i in range(len(cands))))

    json.dump(res, open(f"{REPORTS}/wave5_abl_blocks.json", "w"), indent=2, default=float)
    print(f"\nwrote research/reports/wave5_abl_blocks.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1:])
