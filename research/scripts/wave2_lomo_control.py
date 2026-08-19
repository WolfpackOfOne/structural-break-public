"""The control that makes the leave-one-module-out battery interpretable.

Observed: at the battery protocol (400k rows) EVERY module removal IMPROVES the
score. That is not seven independent discoveries that seven modules are useless.
It is one fact -- at 400k rows a 500-column model is over-wide, so dropping ~60
columns helps whatever they are.

Leave-one-module-out therefore confounds two effects:
    (a) "this model prefers fewer columns"   <- nuisance
    (b) "these particular columns carried information"  <- what we want

This script measures (a) directly by dropping a RANDOM set of columns of the same
size, at the same protocol, with the same seed. Module value is then the gap:

    value(module) = delta(random drop of k columns) - delta(drop this module of k)

A module whose removal hurts MORE than a random drop of the same size is carrying
real information. A module whose removal hurts LESS is dead weight.
"""
from __future__ import annotations

import json, sys
import numpy as np

sys.path.insert(0, "/home/claude/sb/src")
sys.path.insert(0, "/home/claude/sb/research/scripts")
import lightgbm as lgb

from sbr.metric import ts_auc_flat
from sbr.pipeline import Data, load_features, _stack, append_result
from wave2_lib import ABL, FULL

SIZES = [(60, 3), (151, 2)]      # (n_columns_dropped, n_random_repeats)
FOLDS = (0, 1, 2, 3, 4)

d = Data()
mats, names = load_features(FULL)
p = dict(ABL["params"]); n_round = int(p.pop("n_estimators"))
p.update(objective="binary", num_threads=2, verbose=-1, bagging_freq=1)
out = {}

for k, reps in SIZES:
    for rep in range(reps):
        drop_rng = np.random.default_rng(5000 + 17 * k + rep)
        drop = drop_rng.choice(len(names), k, replace=False)
        keep = np.setdiff1d(np.arange(len(names)), drop)
        per = []
        for f in FOLDS:
            rng = np.random.default_rng(0)          # SAME row sample as the battery
            tr = d.rows_for([x for x in FOLDS if x != f])
            tr = np.sort(rng.choice(tr, min(ABL["max_train_rows"], len(tr)), replace=False))
            va = d.rows_for([f])
            X = _stack(mats, names, tr, keep)
            b = lgb.train(p, lgb.Dataset(X, label=d.y[tr], params=p), num_boost_round=n_round)
            del X
            Xv = _stack(mats, names, va, keep)
            s = float(ts_auc_flat(b.predict(Xv), d.y[va], d.t[va]))
            del Xv, b
            per.append(s)
            print(f"  random-drop k={k} rep={rep} fold {f}: {s:.5f}", flush=True)
        key = f"random_drop_{k}_rep{rep}"
        out[key] = {"per_fold": per, "mean": float(np.mean(per)), "n_dropped": k,
                    "n_features": int(len(keep))}
        append_result({
            "experiment_id": f"RT-24{k//60}{rep}", "date": __import__("time").strftime("%Y-%m-%d %H:%M"),
            "git_sha": __import__("subprocess").check_output(
                ["git", "-C", "/home/claude/sb", "rev-parse", "--short", "HEAD"]).decode().strip(),
            "agent": "ablation-control",
            "hypothesis": f"Dropping {k} RANDOM columns measures the pure column-count effect "
                          f"that leave-one-module-out confounds with module value",
            "falsification_condition": "n/a (control)",
            "feature_set": f"random {len(keep)} of 500", "n_features": int(len(keep)),
            "model": "lgbm", "objective": "binary", "folds": "0,1,2,3,4", "random_seed": 0,
            "mean_oof_ts_auc": float(np.mean(per)), "pooled_oof_ts_auc": float(np.mean(per)),
            "per_fold_ts_auc": ";".join(f"{x:.5f}" for x in per),
            "fold_std": float(np.std(per)),
            "lockbox_touched": "no", "test_reduced_touched": "no", "status": "recorded",
            "notes": f"BATTERY control: random drop of {k} columns, repeat {rep}",
            "protocol": "battery-400k",
        })

json.dump(out, open("/home/claude/sb/research/reports/lomo_column_count_control.json", "w"), indent=2)
print(json.dumps({k: round(v["mean"], 5) for k, v in out.items()}, indent=2))
