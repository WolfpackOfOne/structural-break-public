"""Assemble the W6-E2R report artefacts from the runner's JSON.

Adds the one number the runner does not produce: the LEGAL row-level champion
stream's series AUC on exactly the same eligible population, so "our features
given the boundary" and "our shipped model given nothing" sit in one table.

SERIES ROC AUC ONLY.  No row-level TS-AUC anywhere in this file.
"""
from __future__ import annotations

import csv, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.environ.get("SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/research/scripts")
from wave6_e2r import (FOLDS, FOLDS_PQ, HORIZON, MIN_POST, OOF_RT300, STORE,
                       load_frontier)

J = f"{ROOT}/research/reports/wave6_corrected_oracle.json"
res = json.load(open(J))
F, _, _ = load_frontier()

series = F.load_2026_series(STORE, FOLDS_PQ)
F.derive_break_families(series, seed=0)

rows = []
rt300 = {}
for r in res["per_seed"]:
    s = r["pseudo_seed"]
    df = F.build_2026_records(series, HORIZON, s)
    e = df[df["post_len"] >= MIN_POST]
    sc = F.current_model_scores(e, f"{STORE}/meta.parquet", OOF_RT300, HORIZON)
    y = e["target"].to_numpy(dtype=int); fo = e["fold"].to_numpy(dtype=int)
    auc, per = F.auc_with_folds(y, sc, fo)
    rt300[s] = {"auc": auc, "per_fold": per}
    print(f"  seed {s}  RT-300 legal, no boundary, eligible population: {auc:.5f}", flush=True)

v = np.array([rt300[s]["auc"] for s in rt300])
res["arms"]["legal_RT300_no_boundary"] = {
    "mean": float(v.mean()), "std": float(v.std(ddof=1)), "min": float(v.min()),
    "max": float(v.max()),
    "note": "the SHIPPED row-level stream's prediction at the final online row, "
            "scored as a series score on the same eligible population. It is "
            "given NO boundary. This is the legal reference, not an arm.",
}
for r in res["per_seed"]:
    r["legal_RT300_no_boundary"] = rt300[r["pseudo_seed"]]
json.dump(res, open(J, "w"), indent=1)

# ---- csv ----
keys = ["A_rich_full_population", "A_rich", "A_basic", "B_causal",
        "B_causal_withpos", "C_nobound", "AB", "legal_RT300_no_boundary"]
out = f"{ROOT}/research/reports/wave6_corrected_oracle.csv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["arm", "rt_id", "mean_series_roc_auc", "std_across_pseudo_seeds",
                "min", "max"] + [f"seed_{r['pseudo_seed']}" for r in res["per_seed"]])
    ids = {"A_rich": "RT-940", "A_rich_full_population": "RT-940", "A_basic": "RT-941",
           "B_causal": "RT-942", "B_causal_withpos": "RT-942", "C_nobound": "RT-943",
           "AB": "RT-944", "legal_RT300_no_boundary": "RT-300"}
    for k in keys:
        g = res["arms"][k]
        w.writerow([k, ids[k], f"{g['mean']:.6f}", f"{g['std']:.6f}",
                    f"{g['min']:.6f}", f"{g['max']:.6f}"]
                   + [f"{r[k]['auc']:.6f}" for r in res["per_seed"]])
    for k, g in sorted(res.get("sentinels", {}).items()):
        w.writerow([f"SENTINEL {k}", "", f"{g['auc']:.6f}", "", "", ""])
print(f"wrote {out}")
