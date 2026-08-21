"""The Wave-5 promotion battery: does a candidate stream beat a SEED CLONE?

Pre-registered in research/WAVE5_PREREG.md section 4.  This is the only test
that decides anything in wave 5.

    S        the RT-600 architecture, seven specialists
    C        the candidate stream
    N        the MATCHED CONTROL -- a stream of the same strength carrying no
             new information at all (RT-401: the champion configuration at
             seed 1, differing from RT-300 by nothing but the seed)

The number that matters is  (S + C) - (S + N),  not (S + C) - S.  Wave 3 and
wave 4 both produced candidates that beat S and lost to N; "beats the champion
blend" is not evidence and is not reported as the headline here.

Usage:  wave5_stream_eval.py <tag> <candidate> [<extra controls>...]
"""
from __future__ import annotations

import json, sys, time
import numpy as np

from wave5_lib import (Ctx, load_oof, SPECIALISTS, SEEDCLONES, FOLDS, REPORTS,
                       ROOT, fmt)

SEED_CONTROL = "RT-401"          # champion config, seed 1: no new information


def within_t_rank(scores, t):
    order = np.lexsort((scores, t))
    s, tt = scores[order], t[order]
    n = len(s)
    gs = np.flatnonzero(np.r_[True, tt[1:] != tt[:-1]])
    gl = np.r_[gs[1:], n] - gs
    pos = np.arange(n) - np.repeat(gs, gl)
    new = np.r_[True, (tt[1:] != tt[:-1]) | (s[1:] != s[:-1])]
    rs = np.flatnonzero(new)
    rl = np.r_[rs[1:], n] - rs
    out = np.empty(n)
    out[order] = np.repeat(pos[rs] + (rl - 1) / 2.0, rl) / np.maximum(np.repeat(gl, gl) - 1, 1)
    return out


def main(tag, cand, extras):
    t0 = time.time()
    c = Ctx()
    need = sorted(set(SPECIALISTS) | {SEED_CONTROL, cand} | set(extras))
    P = load_oof(need)
    Q = c.crossfit_streams(P, need)

    S = c.blend(Q, SPECIALISTS)
    S_m, S_pf = c.score(S)
    res = {"experiment": tag, "candidate": cand, "seed_control": SEED_CONTROL,
           "specialists": SPECIALISTS, "S": {"mean": S_m, "per_fold": S_pf},
           "standalone": {}, "compositions": {}}
    print(f"=== {tag}: {cand} against the {SEED_CONTROL} seed clone ===")
    print(f"  S (7 specialists)          {S_m:.5f}   {fmt(S_pf)}")

    print("\n  -- standalone streams --")
    for e in [cand, SEED_CONTROL] + list(extras):
        m, pf = c.score(P[e])
        res["standalone"][e] = {"mean": m, "per_fold": pf}
        print(f"  {e:10s} {m:.5f}   {fmt(pf)}")

    print("\n  -- eight-member compositions (equal weight, cross-fitted SCDF) --")
    vecs = {"S": S}
    for e in [cand, SEED_CONTROL] + list(extras):
        v = c.blend(Q, SPECIALISTS + [e])
        m, pf = c.score(v)
        d = [a - b for a, b in zip(pf, S_pf)]
        res["compositions"][f"S+{e}"] = {
            "mean": m, "per_fold": pf, "delta_vs_S": m - S_m,
            "per_fold_delta_vs_S": d, "n_folds_better_than_S": int(sum(x > 0 for x in d))}
        vecs[f"S+{e}"] = v
        print(f"  S+{e:10s} {m:.5f}  vs S {m-S_m:+.5f} ({sum(x>0 for x in d)}/5)   {fmt(pf)}")

    # ---- THE BINDING COMPARISON --------------------------------------
    a = res["compositions"][f"S+{cand}"]
    b = res["compositions"][f"S+{SEED_CONTROL}"]
    dd = [x - y for x, y in zip(a["per_fold"], b["per_fold"])]
    nb = int(sum(x > 0 for x in dd))
    res["verdict"] = {
        "delta_vs_seed_control": a["mean"] - b["mean"],
        "per_fold_delta": dd, "n_folds_better": nb,
        "prereg_bar": {"threshold": 0.0030, "min_folds": 4},
        "outcome": ("PROMOTE candidate" if a["mean"] - b["mean"] >= 0.0030 and nb >= 4
                    else "EXPLORATORY" if a["mean"] - b["mean"] >= 0.0010
                    else "REJECTED: no information beyond seed diversity"),
    }
    print(f"\n  BINDING: (S+{cand}) - (S+{SEED_CONTROL}) = "
          f"{a['mean']-b['mean']:+.5f}  on {nb}/5 folds  ->  {res['verdict']['outcome']}")

    # ---- diversity, the credential that wave 3 proved is worthless -----
    dev = c.dev
    rS = within_t_rank(S[dev], c.d.t[dev])
    res["within_t_rank_corr_with_S"] = {
        e: float(np.corrcoef(rS, within_t_rank(P[e][dev], c.d.t[dev]))[0, 1])
        for e in [cand, SEED_CONTROL] + list(extras)}
    print("\n  within-t rank correlation with S (NOT a promotion credential):")
    for e, v in res["within_t_rank_corr_with_S"].items():
        print(f"    {e:10s} {v:.4f}")

    # ---- age buckets ---------------------------------------------------
    print("\n  -- TS-AUC by post-break age --")
    ages = {k: c.score_by_age(v) for k, v in vecs.items()}
    res["age"] = ages
    keys = list(ages["S"])
    hdr = ["S", f"S+{cand}", f"S+{SEED_CONTROL}"]
    print(f"  {'age':>8s}" + "".join(f"{h:>16s}" for h in hdr) + f"{'cand-ctrl':>12s}")
    for kk in keys:
        row = [ages[h][kk]["ts_auc"] for h in hdr]
        print(f"  {kk:>8s}" + "".join(f"{x:>16.5f}" for x in row) + f"{row[1]-row[2]:>+12.5f}")

    # ---- bootstrap -----------------------------------------------------
    print("\n  -- paired series bootstrap (200 reps, common random numbers) --")
    res["bootstrap"] = c.bootstrap(
        vecs, {"cand_minus_seedctrl": (f"S+{cand}", f"S+{SEED_CONTROL}"),
               "cand_minus_S": (f"S+{cand}", "S")}, n=200, seed=0)
    for k, v in res["bootstrap"].items():
        print(f"    {k:22s} {v['mean']:+.5f}  CI [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]"
              f"  {100*v['fraction_positive']:.0f}% positive")

    json.dump(res, open(f"{REPORTS}/wave5_{tag}.json", "w"), indent=2, default=float)
    print(f"\nwrote research/reports/wave5_{tag}.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
