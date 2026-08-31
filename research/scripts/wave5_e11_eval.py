"""W5-E11 verdict: S' (every stream rebuilt with m12_rdep) against S.

S and S' share member count, weights, seeds, protocol and calibration.  They
differ by 57 columns per stream and by nothing else, so there is no bagging
channel for a gain to arrive through -- which is why this comparison, and not
the eighth-member one, is what decides the block.
"""
from __future__ import annotations

import json, time
import numpy as np

from wave5_lib import Ctx, load_oof, SPECIALISTS, SEEDCLONES, REPORTS, ROOT, fmt

SPRIME = ["RT-751", "RT-811", "RT-812", "RT-813", "RT-814", "RT-815", "RT-816"]
PAIRS = list(zip(SPECIALISTS, SPRIME))


def main():
    t0 = time.time()
    c = Ctx()
    need = sorted(set(SPECIALISTS) | set(SPRIME) | set(SEEDCLONES))
    P = load_oof(need)
    Q = c.crossfit_streams(P, need)

    S = c.blend(Q, SPECIALISTS)
    Sp = c.blend(Q, SPRIME)
    B = c.blend(Q, SEEDCLONES)
    S_m, S_pf = c.score(S)
    Sp_m, Sp_pf = c.score(Sp)
    B_m, B_pf = c.score(B)
    d = [a - b for a, b in zip(Sp_pf, S_pf)]
    nb = int(sum(x > 0 for x in d))

    print("=== W5-E11: does m12_rdep improve the ARCHITECTURE? ===\n")
    print("  -- member by member, same configuration, one module added --")
    print(f"  {'incumbent':>10s} {'score':>9s}   {'rebuilt':>10s} {'score':>9s}   {'delta':>9s}")
    per_member = {}
    for a, b in PAIRS:
        ma, _ = c.score(P[a])
        mb, _ = c.score(P[b])
        per_member[b] = {"incumbent": a, "incumbent_score": ma, "rebuilt_score": mb,
                         "delta": mb - ma}
        print(f"  {a:>10s} {ma:9.5f}   {b:>10s} {mb:9.5f}   {mb-ma:+9.5f}")

    print(f"\n  S  seven specialists          {S_m:.5f}   {fmt(S_pf)}")
    print(f"  S' rebuilt with m12_rdep      {Sp_m:.5f}   {fmt(Sp_pf)}")
    print(f"  B  seven seed clones          {B_m:.5f}   {fmt(B_pf)}")
    print(f"\n  S' - S = {Sp_m-S_m:+.5f}  on {nb}/5 folds   {' '.join(f'{x:+.5f}' for x in d)}")

    res = {"experiment": "W5-E11", "S": {"mean": S_m, "per_fold": S_pf},
           "S_prime": {"mean": Sp_m, "per_fold": Sp_pf},
           "B_seed_clone": {"mean": B_m, "per_fold": B_pf},
           "per_member": per_member,
           "delta": Sp_m - S_m, "per_fold_delta": d, "n_folds_better": nb,
           "prereg_bar": {"threshold": 0.0030, "min_folds": 4}}

    print("\n  -- paired series bootstrap (200 reps, common random numbers) --")
    res["bootstrap"] = c.bootstrap({"S": S, "Sp": Sp, "B": B},
                                   {"Sprime_minus_S": ("Sp", "S"),
                                    "Sprime_minus_B": ("Sp", "B")}, n=200, seed=0)
    for k, v in res["bootstrap"].items():
        print(f"    {k:20s} {v['mean']:+.5f}  CI [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]"
              f"  {100*v['fraction_positive']:.0f}% positive")

    ci = res["bootstrap"]["Sprime_minus_S"]["ci95"]
    res["outcome"] = ("H1 CONFIRMED: the block improves the architecture"
                      if (Sp_m - S_m) >= 0.0030 and nb >= 4 and ci[0] > 0
                      else "EXPLORATORY" if (Sp_m - S_m) >= 0.0010 and nb >= 4
                      else "H0 ACCEPTED: the block adds nothing the bank did not already reach")
    print(f"\n  VERDICT: {res['outcome']}")

    print("\n  -- TS-AUC by post-break age --")
    ages = {"S": c.score_by_age(S), "S_prime": c.score_by_age(Sp)}
    res["age"] = ages
    print(f"  {'age':>8s}{'S':>13s}{'S prime':>13s}{'delta':>11s}")
    for kk in list(ages["S"]):
        a, b = ages["S"][kk]["ts_auc"], ages["S_prime"][kk]["ts_auc"]
        print(f"  {kk:>8s}{a:>13.5f}{b:>13.5f}{b-a:>+11.5f}")

    np.save(f"{ROOT}/research/oof/wave5_Sprime.npy", Sp)
    json.dump(res, open(f"{REPORTS}/wave5_e11_architecture.json", "w"), indent=2, default=float)
    print(f"\nwrote research/reports/wave5_e11_architecture.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
