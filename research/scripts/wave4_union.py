"""W4-E6: do specialisation and bagging stack?

Pre-registered in research/RDOF_LEDGER.md before any composition was scored.
Three compositions, all unions, none selected:

    RT-420  seven specialists           (the incumbent)
    RT-421  seven seed clones
    RT-422  the union, deduplicated     (13 boosters)

Zero new training -- every OOF vector already exists.  Calibration is the same
cross-fitted SCDF, fitted on folds != k for validation fold k.
"""
from __future__ import annotations

import json, os, sys

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

import numpy as np
from sbr.metric import ts_auc_flat
from sbr.pipeline import Data
from wave4_ensemble import Blends, FOLDS, score
from wave4_lib import SEED_SET, SPECIALIST_SET

OUT = f"{ROOT}/research/reports/wave4_union_ensemble.json"
SINGLE = "RT-300"
COMPOSITIONS = {
    "RT-420_specialist7": list(SPECIALIST_SET),
    "RT-421_seedclone7": list(SEED_SET),
    "RT-422_union13": list(dict.fromkeys(list(SPECIALIST_SET) + list(SEED_SET))),
}


def main():
    d = Data()
    rows = {f: d.rows_for([f]) for f in FOLDS}
    dev = d.rows_for(list(FOLDS))
    need = sorted({e for v in COMPOSITIONS.values() for e in v})
    P = {e: np.load(f"{ROOT}/research/oof/{e}.npy") for e in need}
    B = Blends(P, d, rows)

    res = {"compositions": {k: v for k, v in COMPOSITIONS.items()}, "results": {}, "vectors": {}}
    for name, streams in COMPOSITIONS.items():
        v = B.crossfit("scdf_t", streams)
        m, pf = score(v, d, rows)
        res["results"][name] = {"n_boosters": len(streams), "mean": m, "per_fold": pf}
        res["vectors"][name] = v
        print(f"  {name:22s} n={len(streams):2d}  {m:.5f}   " + " ".join(f"{x:.5f}" for x in pf))

    a = res["results"]["RT-422_union13"]; b = res["results"]["RT-420_specialist7"]
    diff = a["mean"] - b["mean"]
    per = [x - y for x, y in zip(a["per_fold"], b["per_fold"])]

    # paired bootstrap, common random numbers
    sid = d.sidx[dev]
    order = np.argsort(sid, kind="stable"); sid_s = sid[order]
    bnd = np.flatnonzero(np.r_[True, sid_s[1:] != sid_s[:-1]])
    ends = np.r_[bnd[1:], len(sid_s)]
    srows = {sid_s[i]: dev[order[i:j]] for i, j in zip(bnd, ends)}
    uniq = np.unique(sid)
    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(200):
        r = np.concatenate([srows[s] for s in rng.choice(uniq, len(uniq), replace=True)])
        y, tt = d.y[r], d.t[r]
        deltas.append(float(ts_auc_flat(res["vectors"]["RT-422_union13"][r], y, tt)
                            - ts_auc_flat(res["vectors"]["RT-420_specialist7"][r], y, tt)))
    dv = np.array(deltas)
    ci = [float(np.quantile(dv, .025)), float(np.quantile(dv, .975))]
    nfold = int(sum(x > 0 for x in per))
    ok = diff > 0.0020 and nfold >= 4 and ci[0] > 0
    res["verdict"] = {
        "union_minus_specialist": diff, "per_fold": per, "n_folds_positive": nfold,
        "bootstrap": {"mean": float(dv.mean()), "ci95": ci,
                      "fraction_positive": float((dv > 0).mean())},
        "prereg_bar": {"threshold": 0.0020, "min_folds": 4, "ci_excludes_zero": True},
        "prereg_outcome": ("H1 CONFIRMED: specialisation and bagging stack"
                           if ok else "H0 ACCEPTED: the gains overlap; incumbent stands"),
        "note": "Promotion additionally requires W4-E4 to show 13 boosters fit the runtime budget.",
    }
    print(f"\n  union - specialist: {diff:+.5f}  ({nfold}/5 folds)  "
          f"bootstrap {dv.mean():+.5f} CI [{ci[0]:+.5f}, {ci[1]:+.5f}] "
          f"{100*(dv>0).mean():.0f}% positive")
    print("  " + res["verdict"]["prereg_outcome"])
    del res["vectors"]
    json.dump(res, open(OUT, "w"), indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
