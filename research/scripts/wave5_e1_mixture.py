"""W5-E1: does a SMALL bagging component improve the specialist ensemble?

Pre-registered in research/WAVE5_PREREG.md section 6 BEFORE this ran.

W4-E6 rejected the 13-booster UNION, and its mechanism was diagnosed: under an
equal-weight mean the union hands the RT-100R configuration 54% of the blend
weight instead of 14%.  A small bagging component is a different point in the
space -- at lambda=0.90 that configuration carries 22.9% -- so W4-E6 does not
answer this.  The grid is fixed at four values and is NOT enlarged after seeing
results.

No training.  Pure recombination of OOF vectors that already exist.
"""
from __future__ import annotations

import json, time
import numpy as np

from wave5_lib import Ctx, load_oof, SPECIALISTS, SEEDCLONES, FOLDS, REPORTS, fmt

LAMBDAS = [1.00, 0.90, 0.80, 0.70]     # FIXED. Not enlarged.


def main():
    t0 = time.time()
    c = Ctx()
    P = load_oof(sorted(set(SPECIALISTS) | set(SEEDCLONES)))

    out = {"experiment": "W5-E1", "lambdas": LAMBDAS,
           "specialists": SPECIALISTS, "seedclones": SEEDCLONES}

    # ---- baselines A / B / C, same platform, same session --------------
    print("=== BASELINES (macOS/arm64, canonical partition) ===")
    A_m, A_pf = c.score(P["RT-300"])
    print(f"  A single RT-300      {A_m:.5f}   {fmt(A_pf)}")
    Sv = c.crossfit_blend(P, SPECIALISTS)
    Bv = c.crossfit_blend(P, SEEDCLONES)
    S_m, S_pf = c.score(Sv)
    B_m, B_pf = c.score(Bv)
    print(f"  B seven seed clones  {B_m:.5f}   {fmt(B_pf)}")
    print(f"  C seven specialists  {S_m:.5f}   {fmt(S_pf)}")
    out["baselines"] = {
        "A_single_RT300": {"mean": A_m, "per_fold": A_pf},
        "B_seed_clone_7": {"mean": B_m, "per_fold": B_pf},
        "C_specialist_7": {"mean": S_m, "per_fold": S_pf},
        "bagging_delta": B_m - A_m,
        "specialisation_delta": S_m - B_m,
        "total_delta": S_m - A_m,
    }

    # ---- the mixture grid ---------------------------------------------
    print("\n=== W5-E1 MIXTURE  lambda*S + (1-lambda)*B ===")
    vecs = {"S": Sv, "B": Bv}
    out["grid"] = {}
    for lam in LAMBDAS:
        v = lam * Sv + (1.0 - lam) * Bv
        m, pf = c.score(v)
        d = [x - y for x, y in zip(pf, S_pf)]
        nb = sum(x > 0 for x in d)
        out["grid"][f"{lam:.2f}"] = {
            "mean": m, "per_fold": pf, "delta_vs_S": m - S_m,
            "per_fold_delta": d, "n_folds_better": int(nb),
            # weight the RT-100R configuration carries in the mixture
            "rt100R_blend_weight": lam / 7.0 + (1.0 - lam),
        }
        vecs[f"mix{lam:.2f}"] = v
        print(f"  lambda={lam:.2f}  {m:.5f}  delta {m-S_m:+.5f}  "
              f"{nb}/5 folds  (RT-100R weight {lam/7+1-lam:.1%})   {fmt(pf)}")

    # ---- verdict against the pre-registered falsification --------------
    best = max(LAMBDAS[1:], key=lambda l: out["grid"][f"{l:.2f}"]["delta_vs_S"])
    g = out["grid"][f"{best:.2f}"]
    passed = g["delta_vs_S"] >= 0.0010 and g["n_folds_better"] >= 4
    out["verdict"] = {
        "best_lambda_below_1": best, "best_delta": g["delta_vs_S"],
        "best_n_folds": g["n_folds_better"],
        "prereg_bar": {"delta": 0.0010, "min_folds": 4},
        "outcome": "SURVIVES screening bar" if passed else "REJECTED",
    }
    print(f"\n  best lambda<1 = {best:.2f}, delta {g['delta_vs_S']:+.5f}, "
          f"{g['n_folds_better']}/5  ->  {out['verdict']['outcome']}")

    # ---- bootstrap on the best mixture, whatever the verdict -----------
    print("\n=== paired series bootstrap (200 reps, common random numbers) ===")
    out["bootstrap"] = c.bootstrap(
        vecs,
        {"best_mix_minus_S": (f"mix{best:.2f}", "S"),
         "S_minus_B": ("S", "B")},
        n=200, seed=0)
    for k, v in out["bootstrap"].items():
        print(f"  {k:22s} {v['mean']:+.5f}  CI [{v['ci95'][0]:+.5f}, "
              f"{v['ci95'][1]:+.5f}]  {100*v['fraction_positive']:.0f}% positive")

    json.dump(out, open(f"{REPORTS}/wave5_e1_mixture.json", "w"), indent=2)
    np.save(f"{Ctx.__module__ and REPORTS}/../oof/wave5_S_specialist.npy", Sv)
    np.save(f"{REPORTS}/../oof/wave5_B_seedclone.npy", Bv)
    print(f"\nwrote research/reports/wave5_e1_mixture.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
