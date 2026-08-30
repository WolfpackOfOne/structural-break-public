"""RT-110 -- the metric-aligned objective at full scale.

Agent 9 found (screen, fold 0, 211 features) that a pairwise logistic loss whose
pairs are drawn WITHIN the same online index t -- exactly the stratification the
official metric uses -- beats binary logloss by +0.0072.  This promotes that
result to the champion feature set and the full 5-fold protocol.
"""
import sys, time, json; sys.path.insert(0, "/home/claude/sb/scripts"); sys.path.insert(0, "/home/claude/sb/src")
import numpy as np, lightgbm as lgb
from agent09_lib import Bench, log, make_pairwise_obj, BASE
from sbr.metric import ts_auc_flat

FULL = ["m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes"]
P = dict(BASE); P.update(learning_rate=0.05, num_leaves=63, min_data_in_leaf=300,
                         feature_fraction=0.5, bagging_fraction=0.7)
N_ROUND = 600

b = Bench(FULL, folds=(0, 1, 2, 3, 4), max_train_rows=800_000, seed=0, screen=False)
oof = np.full(len(b.d.y), np.nan, dtype=np.float32)
per = []
t0 = time.time()
for f in (0, 1, 2, 3, 4):
    pk = b.get(f)
    obj, info = make_pairwise_obj(pk["ttr"], pk["ytr"], m_neg=8, seed=0)
    ds = lgb.Dataset(pk["Xtr"], label=pk["ytr"], params=P)
    boo = lgb.train(dict(P, objective=obj), ds, num_boost_round=N_ROUND)
    pred = boo.predict(pk["Xva"]).astype(np.float32)
    oof[pk["va"]] = pred
    s = b.score(pred, pk)
    per.append(s)
    print(f"fold {f}: pairwise TS-AUC {s:.5f}  ({info})", flush=True)
    b.cache.clear()

dev = b.d.rows_for([0, 1, 2, 3, 4])
pooled = float(ts_auc_flat(oof[dev], b.d.y[dev], b.d.t[dev]))
np.save("/home/claude/sb/research/oof/RT-110.npy", oof)
log("RT-110", b, per, hypothesis="Pairwise logistic ranking within online index t beats binary logloss at full scale",
    falsification="mean OOF TS-AUC <= RT-100's 0.61510", model="lgbm", objective="pairwise_logistic_t_groups",
    notes="promotion of agent9's screen finding to champion features + full 5 folds", sample_mode="uniform",
    runtime=time.time() - t0, agent="agent0", oof=pooled)
print(json.dumps({"mean": float(np.mean(per)), "pooled": pooled, "per_fold": per,
                  "champion_binary": 0.6151034246587253}, indent=2))
