"""Negative control: does a block of 60 pure-noise columns improve the model?

Memory-light rerun of the arm that was OOM-killed in the battery. The original
held X, Xv and their augmented copies simultaneously (~5 GB on a 7 GB box). Here
the augmented matrix is allocated ONCE at full width and the noise columns are
overwritten in place, so peak memory is one training matrix plus one validation
matrix.

Expected: delta ~= 0. A consistent improvement would mean the model is exploiting
capacity/regularisation noise rather than signal, and would invalidate every small
positive delta in the ledger.
"""
from __future__ import annotations

import json, sys
import numpy as np

sys.path.insert(0, "/home/claude/sb/src")
sys.path.insert(0, "/home/claude/sb/research/scripts")
import lightgbm as lgb

from sbr.metric import ts_auc_flat
from sbr.pipeline import Data, load_features, _stack
from wave2_lib import ABL, FULL

EXTRA = 60
FOLDS = (0, 1, 2)

d = Data()
mats, names = load_features(FULL)
p = dict(ABL["params"]); n_round = int(p.pop("n_estimators"))
p.update(objective="binary", num_threads=2, verbose=-1, bagging_freq=1)
keep = np.arange(len(names))
res = {"with": [], "without": []}

for f in FOLDS:
    rng = np.random.default_rng(0)
    tr = d.rows_for([x for x in (0, 1, 2, 3, 4) if x != f])
    tr = np.sort(rng.choice(tr, min(ABL["max_train_rows"], len(tr)), replace=False))
    va = d.rows_for([f])

    Xtr = np.empty((len(tr), len(names) + EXTRA), dtype=np.float32)
    Xtr[:, :len(names)] = _stack(mats, names, tr, keep)
    Xva = np.empty((len(va), len(names) + EXTRA), dtype=np.float32)
    Xva[:, :len(names)] = _stack(mats, names, va, keep)

    for arm in ("without", "with"):
        if arm == "with":
            g = np.random.default_rng(1000 + f)
            Xtr[:, len(names):] = g.standard_normal((len(tr), EXTRA)).astype(np.float32)
            Xva[:, len(names):] = g.standard_normal((len(va), EXTRA)).astype(np.float32)
            cols = slice(0, len(names) + EXTRA)
        else:
            cols = slice(0, len(names))
        b = lgb.train(p, lgb.Dataset(Xtr[:, cols], label=d.y[tr], params=p), num_boost_round=n_round)
        s = float(ts_auc_flat(b.predict(Xva[:, cols]), d.y[va], d.t[va]))
        res[arm].append(s)
        print(f"  random-block {arm:8s} fold {f}: {s:.5f}", flush=True)
        del b
    del Xtr, Xva

res["delta_mean"] = float(np.mean(res["with"]) - np.mean(res["without"]))
res["folds"] = list(FOLDS)
res["n_random_columns"] = EXTRA
res["verdict"] = ("PASS - noise does not help" if abs(res["delta_mean"]) < 0.002
                  else "INVESTIGATE - noise moved the score materially")
print(json.dumps(res, indent=2))

out = "/home/claude/sb/research/reports/negative_controls.json"
try:
    cur = json.load(open(out))
except Exception:
    cur = {}
cur["random_feature_block"] = res
json.dump(cur, open(out, "w"), indent=2)
print("updated", out)
