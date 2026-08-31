"""W6-N controlled analysis: every comparison the promotion bar names, and no others.

research/WAVE6_NEURAL_PREREG.md section 6 is the bar.  The binding question is
NOT "does the neural model beat RT-300" -- it is

    does it add information beyond ordinary LightGBM bagging?

so every neural arm is measured against a MATCHED SEED CLONE occupying the same
ensemble slot.  W5-NULLTEST priced an eighth exchangeable member at +0.00003.
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/research/scripts")

from wave5_lib import (AGE_BUCKETS, CANON_CAL, Ctx, FOLDS, OOFDIR, SEEDCLONES,
                       SPECIALISTS, fmt, load_oof)

#: the eighth-member seed clone -- an exchangeable CHAMP-protocol LightGBM
#: stream that is NOT already in S.  This is the control the bar is written
#: against, not RT-300.
EIGHTH_CLONE = "RT-401"
NEURAL = ("RT-960", "RT-961", "RT-970", "RT-971")


def within_time_rank_corr(ctx, a, b, min_alive=8):
    """Pair-weighted mean Spearman between two scores INSIDE each timestep.

    Cross-time correlation is meaningless for TS-AUC: the metric never compares
    rows at different t.  This is the correlation the ensemble actually feels.
    """
    r = ctx.dev
    t = ctx.d.t[r]
    y = ctx.d.y[r]
    av, bv = a[r], b[r]
    order = np.argsort(t, kind="stable")
    t, y, av, bv = t[order], y[order], av[order], bv[order]
    st = np.flatnonzero(np.r_[True, t[1:] != t[:-1]])
    en = np.r_[st[1:], len(t)]
    num = den = 0.0
    for s, e in zip(st, en):
        n = e - s
        if n < min_alive:
            continue
        npos = int(y[s:e].sum())
        w = npos * (n - npos)
        if w == 0:
            continue
        ra = np.argsort(np.argsort(av[s:e]))
        rb = np.argsort(np.argsort(bv[s:e]))
        ra = ra - ra.mean(); rb = rb - rb.mean()
        d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
        if d <= 0:
            continue
        num += w * float((ra * rb).sum() / d)
        den += w
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(NEURAL))
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--out", default=f"{ROOT}/research/reports/wave6_neural_analysis.json")
    a = ap.parse_args()

    arms = [x.strip() for x in a.arms.split(",") if x.strip()
            and os.path.exists(f"{OOFDIR}/{x.strip()}.npy")]
    print(f"neural arms found: {arms}")
    ctx = Ctx()
    names = sorted(set(SPECIALISTS) | set(SEEDCLONES) | set(arms))
    P = load_oof(names)

    out = {"controls": {}, "arms": {}, "eighth_clone": EIGHTH_CLONE,
           "calibration": CANON_CAL.__name__, "n_boot": a.boot}

    # ---------------------------------------------------------- controls --
    print("\n=== CONTROLS (raw TS-AUC; SCDF calibration is monotone within t, "
          "so single-stream numbers are calibration-invariant) ===")
    mA, pA = ctx.score(P["RT-300"])
    print(f"  A  RT-300 single LightGBM   {mA:.5f}   {fmt(pA)}")
    Q = ctx.crossfit_streams(P, names)
    vS = Ctx.blend(Q, SPECIALISTS)
    vB = Ctx.blend(Q, SEEDCLONES)
    mS, pS = ctx.score(vS)
    mB, pB = ctx.score(vB)
    print(f"  B  seven seed clones        {mB:.5f}   {fmt(pB)}")
    print(f"  C  seven specialists (S)    {mS:.5f}   {fmt(pS)}")
    v8clone = Ctx.blend(Q, SPECIALISTS + [EIGHTH_CLONE])
    m8, p8 = ctx.score(v8clone)
    print(f"  D  S + {EIGHTH_CLONE} (8th seed clone) {m8:.5f}   {fmt(p8)}"
          f"   delta vs S {m8 - mS:+.5f}   <-- THE BAR TO BEAT")
    out["controls"] = {
        "A_RT300": {"mean": mA, "per_fold": pA},
        "B_seedclone_ensemble": {"mean": mB, "per_fold": pB},
        "S_specialists": {"mean": mS, "per_fold": pS},
        "S_plus_eighth_seed_clone": {"mean": m8, "per_fold": p8,
                                     "delta_vs_S": m8 - mS,
                                     "per_fold_delta": [x - z for x, z in zip(p8, pS)]},
        "age_RT300": ctx.score_by_age(P["RT-300"]),
        "age_S": ctx.score_by_age(vS),
    }

    # ------------------------------------------------------------- arms --
    for n in arms:
        print(f"\n=== {n} ===")
        mN, pN = ctx.score(P[n])
        print(f"  standalone                 {mN:.5f}   {fmt(pN)}"
              f"   delta vs RT-300 {mN - mA:+.5f}  "
              f"{sum(x > z for x, z in zip(pN, pA))}/5 folds")
        vSN = Ctx.blend(Q, SPECIALISTS + [n])
        mSN, pSN = ctx.score(vSN)
        print(f"  S + {n}               {mSN:.5f}   {fmt(pSN)}"
              f"   delta vs S {mSN - mS:+.5f}  "
              f"{sum(x > z for x, z in zip(pSN, pS))}/5 folds")
        print(f"  vs the 8th seed clone      {mSN - m8:+.5f}  "
              f"{sum(x > z for x, z in zip(pSN, p8))}/5 folds   <-- THE BAR")
        v2 = Ctx.blend(Q, ["RT-300", n])
        m2, p2 = ctx.score(v2)
        v2c = Ctx.blend(Q, ["RT-300", EIGHTH_CLONE])
        m2c, p2c = ctx.score(v2c)
        print(f"  two-model RT-300 + {n} {m2:.5f}   vs RT-300 + {EIGHTH_CLONE} "
              f"{m2c:.5f}   delta {m2 - m2c:+.5f}")
        rc_lgb = within_time_rank_corr(ctx, Q[n], Q["RT-300"])
        rc_ens = within_time_rank_corr(ctx, Q[n], vS)
        print(f"  within-time rank corr      vs RT-300 {rc_lgb:.4f}   vs S {rc_ens:.4f}")

        boot = ctx.bootstrap(
            {"N": P[n], "A": P["RT-300"], "SN": vSN, "S": vS, "S8": v8clone},
            {"standalone_minus_RT300": ("N", "A"),
             "ensemble_minus_S": ("SN", "S"),
             "ensemble_minus_eighth_seed_clone": ("SN", "S8")},
            n=a.boot, seed=0)
        for k, v in boot.items():
            print(f"  bootstrap {k:<34s} {v['mean']:+.5f} "
                  f"CI [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}] "
                  f"{v['fraction_positive'] * 100:.1f}% positive")

        ageN = ctx.score_by_age(P[n])
        ageSN = ctx.score_by_age(vSN)
        age8 = ctx.score_by_age(v8clone)
        print("  age buckets (positives restricted; negatives held fixed)")
        print(f"    {'bucket':<10s} {'n_pos':>8s}  {'RT-300':>8s} {n:>9s} {'delta':>9s}"
              f"   {'S':>8s} {'S+N':>8s} {'S+clone':>8s} {'N-clone':>9s}")
        for k in ageN:
            print(f"    {k:<10s} {ageN[k]['n_pos']:>8d}  "
                  f"{out['controls']['age_RT300'][k]['ts_auc']:>8.5f} "
                  f"{ageN[k]['ts_auc']:>9.5f} "
                  f"{ageN[k]['ts_auc'] - out['controls']['age_RT300'][k]['ts_auc']:>+9.5f}   "
                  f"{out['controls']['age_S'][k]['ts_auc']:>8.5f} "
                  f"{ageSN[k]['ts_auc']:>8.5f} {age8[k]['ts_auc']:>8.5f} "
                  f"{ageSN[k]['ts_auc'] - age8[k]['ts_auc']:>+9.5f}")

        out["arms"][n] = {
            "standalone": {"mean": mN, "per_fold": pN,
                           "delta_vs_RT300": mN - mA,
                           "folds_positive_vs_RT300": int(sum(x > z for x, z in zip(pN, pA)))},
            "ensemble_S_plus_N": {"mean": mSN, "per_fold": pSN,
                                  "delta_vs_S": mSN - mS,
                                  "folds_positive_vs_S": int(sum(x > z for x, z in zip(pSN, pS))),
                                  "delta_vs_eighth_seed_clone": mSN - m8,
                                  "folds_positive_vs_eighth": int(sum(x > z for x, z in zip(pSN, p8)))},
            "two_model": {"with_N": m2, "with_clone": m2c, "delta": m2 - m2c},
            "within_time_rank_corr": {"vs_RT300": rc_lgb, "vs_S": rc_ens},
            "bootstrap": boot,
            "age_standalone": ageN, "age_ensemble": ageSN, "age_eighth_clone": age8,
        }
        json.dump(out, open(a.out, "w"), indent=1)

    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
