"""W4-E2 analysis: does the ENSEMBLE DELTA survive the fold-partition draw?

The quantity at risk is a DELTA, not a level.  Wave 2 measured the champion's
absolute sensitivity to the partition (RT-221/222/223) and stopped there; a
delta can be stable while levels move by 0.012, or vanish while levels look fine.

For each partition, and using ONLY that partition's own OOF vectors and its own
cross-fitted calibration:

    single           the champion alone
    seed-clone 7     the bagging control
    specialist 7     the incumbent

    bagging delta        seed-clone - single
    specialisation delta specialist - seed-clone   <- the W4-E1 claim
    total delta          specialist - single

Candidates were declared in RDOF_LEDGER.md before any alternate score was read.
The alternate partitions select nothing; if the answer is unwelcome it is still
the answer.
"""
from __future__ import annotations

import json, os, sys

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

import numpy as np
import pandas as pd
import sbr.pipeline as PL
from sbr.pipeline import Data
from wave4_ensemble import Blends, FOLDS, score
from wave4_lib import SEED_SET, SPECIALIST_SET

OUT = f"{ROOT}/research/reports/ensemble_partition_stability.json"
SINGLE = "RT-300"
PARTS = ("canonical", "alt1", "alt2", "alt3")


def materialise(name):
    """The alt files hold only the 8,000 dev series; build a full-length table."""
    base = pd.read_parquet(f"{ROOT}/research/folds/folds.parquet")
    alt = pd.read_parquet(f"{ROOT}/research/folds/folds_{name}.parquet")
    mp = dict(zip(alt.id.tolist(), alt.fold.tolist()))
    new = base.copy()
    new["fold"] = [mp.get(i, -1) for i in base.id.tolist()]
    tmp = f"{ROOT}/cache/_folds_{name}.parquet"
    new.to_parquet(tmp, index=False)
    return tmp


def analyse(part):
    old = PL.FOLDS
    if part != "canonical":
        PL.FOLDS = materialise(part)
    try:
        d = Data()
        rows = {f: d.rows_for([f]) for f in FOLDS}
        suffix = "" if part == "canonical" else f".{part}"
        need = sorted(set(SEED_SET) | set(SPECIALIST_SET))
        paths = {e: f"{ROOT}/research/oof/{e}{suffix}.npy" for e in need}
        missing = [e for e, p in paths.items() if not os.path.exists(p)]
        if missing:
            return {"status": "INCOMPLETE", "missing": missing}
        P = {e: np.load(p) for e, p in paths.items()}
        B = Blends(P, d, rows)
        single_m, single_pf = score(P[SINGLE], d, rows)
        out = {"status": "ok",
               "single": {"mean": single_m, "per_fold": single_pf},
               "members": {e: score(P[e], d, rows)[0] for e in need}}
        for arm, streams in (("seed_clone", SEED_SET), ("specialist", SPECIALIST_SET)):
            m, pf = score(B.crossfit("scdf_t", streams), d, rows)
            out[arm] = {"mean": m, "per_fold": pf}
        out["deltas"] = {
            "bagging": out["seed_clone"]["mean"] - single_m,
            "specialisation": out["specialist"]["mean"] - out["seed_clone"]["mean"],
            "total": out["specialist"]["mean"] - single_m,
        }
        return out
    finally:
        PL.FOLDS = old


def main():
    res = {"partitions": {}}
    for part in PARTS:
        print(f"=== {part} ===", flush=True)
        r = analyse(part)
        res["partitions"][part] = r
        if r.get("status") != "ok":
            print(f"  INCOMPLETE, missing {len(r['missing'])}: {r['missing'][:5]}")
            continue
        print(f"  single      {r['single']['mean']:.5f}")
        print(f"  seed clone  {r['seed_clone']['mean']:.5f}")
        print(f"  specialist  {r['specialist']['mean']:.5f}")
        print(f"  deltas: bagging {r['deltas']['bagging']:+.5f}  "
              f"specialisation {r['deltas']['specialisation']:+.5f}  "
              f"total {r['deltas']['total']:+.5f}", flush=True)

    ok = {k: v for k, v in res["partitions"].items() if v.get("status") == "ok"}
    if len(ok) > 1:
        for key in ("bagging", "specialisation", "total"):
            vals = np.array([v["deltas"][key] for v in ok.values()])
            res.setdefault("summary", {})[key] = {
                "by_partition": {k: v["deltas"][key] for k, v in ok.items()},
                "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
                "min": float(vals.min()), "max": float(vals.max()),
                "all_positive": bool((vals > 0).all()),
                "sd_exceeds_canonical": bool(
                    vals.std(ddof=1) > abs(ok["canonical"]["deltas"][key])) if "canonical" in ok else None,
            }
        lv = np.array([v["single"]["mean"] for v in ok.values()])
        res["summary"]["single_level_spread"] = {
            "min": float(lv.min()), "max": float(lv.max()), "sd": float(lv.std(ddof=1))}

        print("\n=== SUMMARY ===")
        for key in ("bagging", "specialisation", "total"):
            s = res["summary"][key]
            print(f"  {key:16s} mean {s['mean']:+.5f}  sd {s['sd']:.5f}  "
                  f"range [{s['min']:+.5f}, {s['max']:+.5f}]  "
                  f"all positive: {s['all_positive']}")
        s = res["summary"]["single_level_spread"]
        print(f"  single LEVEL spread across partitions: {s['max']-s['min']:.5f} "
              f"(sd {s['sd']:.5f}) -- compare to the delta sds above")

        spec = res["summary"]["specialisation"]
        res["verdict"] = {
            "falsification": "the ensemble delta is negative on any alternate partition, "
                             "or its across-partition SD exceeds its canonical mean",
            "specialisation_negative_anywhere": not spec["all_positive"],
            "specialisation_sd_exceeds_canonical": spec["sd_exceeds_canonical"],
            "survives": bool(spec["all_positive"] and not spec["sd_exceeds_canonical"]),
        }
        print(f"\n  W4-E2 VERDICT: specialisation delta "
              f"{'SURVIVES' if res['verdict']['survives'] else 'DOES NOT SURVIVE'} the partition draw")

    json.dump(res, open(OUT, "w"), indent=2)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
