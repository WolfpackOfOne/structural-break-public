"""W6-E2R post-hoc control: is arm B's advantage just more columns?

NOT PRE-REGISTERED.  Declared as post-hoc in the report, because the column-count
asymmetry (498 vs 280) is a real alternative explanation and pretending it was
anticipated would be worse than labelling it.

Arm B is randomly subsampled to arm A's exact width, three independent draws,
and re-evaluated with everything else identical.  Wave 5's own W5-E7 result --
175 new columns doing less than 57 -- is the reason this is not a formality.

SERIES ROC AUC ONLY.
"""
from __future__ import annotations

import json, os, sys
import numpy as np

ROOT = os.environ.get("SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/research/scripts")
from wave6_e2r import (FOLDS_PQ, HORIZON, MIN_POST, N_ESTIMATORS, STORE,
                       bank_matrix, build_causal_bank, causal_bank_columns,
                       load_frontier)

F, _, _ = load_frontier()
cols = causal_bank_columns()
series = F.load_2026_series(STORE, FOLDS_PQ)
F.derive_break_families(series, seed=0)

SEEDS = (0, 2026)
DRAWS = (101, 102, 103)
out = {"label": "W6-E2R post-hoc width control -- NOT pre-registered", "runs": []}
for ps in SEEDS:
    df = F.build_2026_records(series, HORIZON, ps)
    df = df[df["post_len"] >= MIN_POST].reset_index(drop=True)
    y = df["target"].to_numpy(dtype=int)
    fo = df["fold"].to_numpy(dtype=int)
    jobs = list(zip(df["id"].astype(int), df["boundary"].astype(int)))
    bank = build_causal_bank(jobs, 1, "full")        # already cached; no rebuild
    XB = bank_matrix(df, bank, cols, drop_position=True)
    nA = len(F.model_feature_columns(df))
    print(f"\n--- pseudo seed {ps}: arm A width {nA}, arm B width {XB.shape[1]} ---", flush=True)
    for d in DRAWS:
        rng = np.random.default_rng(d)
        pick = rng.choice(XB.shape[1], nA, replace=False)
        Xs = XB.iloc[:, sorted(pick)]
        oof = F.crossfit_model(Xs, y, fo, "lgbm", ps + 17, n_estimators=N_ESTIMATORS)
        auc, per = F.auc_with_folds(y, oof, fo)
        print(f"    B_causal subsampled to {nA} cols, draw {d}:  {auc:.5f}   "
              + " ".join(f"{p:.5f}" for p in per), flush=True)
        out["runs"].append({"pseudo_seed": ps, "draw": d, "width": int(nA),
                            "auc": auc, "per_fold": per})
p = f"{ROOT}/research/reports/wave6_corrected_oracle_width.json"
json.dump(out, open(p, "w"), indent=1)
print(f"\nwrote {p}")
