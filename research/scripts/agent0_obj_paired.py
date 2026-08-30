"""RT-111 -- properly paired objective comparison at full scale, fold 0.

The first promotion attempt compared a pairwise model trained on 800k rows with
the champion's 1M rows, which is not a paired test.  Here both objectives see
the IDENTICAL materialised matrix, so the only thing that differs is the loss.
"""
import sys, time, json; sys.path.insert(0, "/home/claude/sb/scripts"); sys.path.insert(0, "/home/claude/sb/src")
import numpy as np, lightgbm as lgb
from agent09_lib import Bench, log, make_pairwise_obj, BASE

FULL = ["m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes"]
P = dict(BASE); P.update(learning_rate=0.05, num_leaves=63, min_data_in_leaf=300,
                         feature_fraction=0.5, bagging_fraction=0.7)
N = 600
b = Bench(FULL, folds=(0,), max_train_rows=800_000, seed=0, screen=False)
pk = b.get(0)
res = {}
for tag, obj in [("binary", "binary"), ("pairwise_t", None)]:
    t0 = time.time()
    pars = dict(P)
    if obj == "binary":
        pars["objective"] = "binary"
        ds = lgb.Dataset(pk["Xtr"], label=pk["ytr"], params=pars)
        boo = lgb.train(pars, ds, num_boost_round=N)
    else:
        fobj, info = make_pairwise_obj(pk["ttr"], pk["ytr"], m_neg=8, seed=0)
        pars["objective"] = fobj
        ds = lgb.Dataset(pk["Xtr"], label=pk["ytr"], params=dict(P, objective="binary"))
        boo = lgb.train(pars, ds, num_boost_round=N)
    s = b.score(boo.predict(pk["Xva"]), pk)
    res[tag] = {"ts_auc": s, "runtime_s": round(time.time() - t0, 1)}
    print(tag, s, flush=True)
    del ds, boo
res["delta_pairwise_minus_binary"] = res["pairwise_t"]["ts_auc"] - res["binary"]["ts_auc"]
json.dump(res, open("/home/claude/sb/research/reports/objective_paired_full.json", "w"), indent=2)
log("RT-111", b, [res["pairwise_t"]["ts_auc"]],
    hypothesis="Pairwise-t ranking beats binary logloss on the champion feature set at full scale",
    falsification="delta <= 0 in a properly paired fold-0 comparison", model="lgbm",
    objective="pairwise_logistic_t_groups", sample_mode="uniform", runtime=0, agent="agent0",
    notes=f"PAIRED fold-0: binary {res['binary']['ts_auc']:.5f} vs pairwise {res['pairwise_t']['ts_auc']:.5f} "
          f"(delta {res['delta_pairwise_minus_binary']:+.5f}); identical 800k-row matrix for both")
print(json.dumps(res, indent=2))
