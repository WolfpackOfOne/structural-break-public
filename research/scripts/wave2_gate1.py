"""PRODUCTION GATE 1 -- is the streaming RT-100 actually submittable?

Runs the real `infer()` through the local Crunch-contract harness on real
held-out series and checks legality, determinism, order-independence, future
poisoning, output range, and runtime.  Also checks that the streaming scores
equal the batch-pipeline scores for the same booster, which is the end-to-end
statement that the streaming port did not change the model.
"""
import argparse, json, sys, time
sys.path.insert(0, "/home/claude/sb/src")
sys.path.insert(0, "/home/claude/sb/research/scripts")
import numpy as np

from local_runner import check_all
from sbr.production.submission import infer
from sbr.production.model import ProductionModel
from sbr.store import load_store
from sbr.pipeline import Data, load_features, _stack
from sbr.metric import ts_auc_flat

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/home/claude/sb/models/rt100_smoke")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--fold", type=int, default=0)
a = ap.parse_args()

st = load_store()
d = Data()
idx = np.flatnonzero(d.series_fold == a.fold)
rng = np.random.default_rng(0)
pick = rng.choice(idx, min(a.n, len(idx)), replace=False)

series, labels = [], []
for i in pick:
    h, o, tau = st.series(int(i))
    series.append((h, o))
    labels.append(st.labels(int(i)))

print(f"=== streaming legality + runtime on {len(series)} fold-{a.fold} series ===")
res = check_all(infer, series, a.model, labels=labels)

# --- end-to-end equivalence: streaming scores == batch-pipeline scores -------
import lightgbm as lgb
m = ProductionModel.load(a.model)
mats, names = load_features(list(m.modules))
keep = np.arange(len(names))
rows = np.concatenate([np.arange(st.orow_off[i], st.orow_off[i] + st.meta.n_online.iloc[i])
                       for i in pick]).astype(np.int64)
X = _stack(mats, names, np.sort(rows), keep)
order = np.argsort(rows)
pb = np.empty(len(rows))
pb[order] = m.boosters[0].predict(X)

from local_runner import run_infer
sc = np.concatenate(run_infer(infer, series, a.model))
mx = float(np.max(np.abs(sc - pb)))
print(f"\nmax |streaming score - batch score| over {len(sc)} points: {mx:.3e}")
res["stream_vs_batch_max_abs"] = mx
res["stream_equals_batch"] = mx < 1e-12
json.dump(res, open("/home/claude/sb/research/reports/gate1.json", "w"), indent=2)
print(json.dumps({k: res[k] for k in ("PASS", "deterministic", "order_independent",
                                      "future_poison_safe", "all_finite", "in_range",
                                      "ms_per_point", "n_points", "stream_equals_batch")}, indent=2))
