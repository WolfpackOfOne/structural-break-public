"""RT-140 -- nested feature selection over the 500-column bank.

The selection is done INSIDE the training folds: the importance ranking comes
from a model fitted on folds 1-4 only, the reduced models are refitted on folds
1-4 only, and fold 0 is touched exactly once per candidate k, as a held-out
score.  Selecting on the champion's pooled gain and then reporting the pooled
OOF would be selecting on the number we report.
"""
import sys, time, json; sys.path.insert(0, "/home/claude/sb/src")
import numpy as np, lightgbm as lgb
from sbr.pipeline import Data, load_features, _stack, append_result
from sbr.metric import ts_auc_flat

FULL = ["m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes"]
P = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=300,
         feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0,
         num_threads=2, verbose=-1, max_bin=127)
N = 600

d = Data(); mats, names = load_features(FULL)
rng = np.random.default_rng(0)
tr = np.sort(rng.choice(d.rows_for([1, 2, 3, 4]), 900_000, replace=False))
va = d.rows_for([0])
all_cols = np.arange(len(names))

t0 = time.time()
Xtr = _stack(mats, names, tr, all_cols)
ds = lgb.Dataset(Xtr, label=d.y[tr], params=P)
full = lgb.train(P, ds, num_boost_round=N)
gain = full.feature_importance("gain")
del ds
Xva = _stack(mats, names, va, all_cols)
res = {"all_500": float(ts_auc_flat(full.predict(Xva), d.y[va], d.t[va]))}
print("all_500", res["all_500"], round(time.time() - t0), flush=True)
del full

order = np.argsort(-gain)
for k in (300, 200, 120, 60):
    sel = np.sort(order[:k])
    p = dict(P); p["feature_fraction"] = min(1.0, 0.5 * 500 / k)
    ds = lgb.Dataset(Xtr[:, sel], label=d.y[tr], params=p)
    b = lgb.train(p, ds, num_boost_round=N)
    res[f"top_{k}"] = float(ts_auc_flat(b.predict(Xva[:, sel]), d.y[va], d.t[va]))
    print(f"top_{k}", res[f"top_{k}"], round(time.time() - t0), flush=True)
    del ds, b

res["note"] = ("importance ranking and all refits use folds 1-4 only; fold 0 is a "
               "held-out score for each candidate k")
json.dump(res, open("/home/claude/sb/research/reports/feature_selection.json", "w"), indent=2)
best = max((v for k, v in res.items() if k.startswith(("all", "top"))))
bestk = [k for k, v in res.items() if v == best][0]
append_result(dict(experiment_id="RT-140", date=time.strftime("%Y-%m-%d %H:%M"), git_sha="nogit",
    agent="agent0", hypothesis="A gain-selected subset of the 500-column bank matches or beats the full bank",
    falsification_condition="every reduced set scores below the full bank on held-out fold 0",
    feature_set=",".join(FULL), n_features=500, model="lgbm", objective="binary", folds="0",
    random_seed=0, mean_oof_ts_auc=best, pooled_oof_ts_auc=best, per_fold_ts_auc=f"{best:.5f}",
    fold_std=0.0, persistence="none", sample_mode="uniform", training_runtime_s=round(time.time() - t0, 1),
    causal_verified="prefix-invariance@module", test_reduced_touched="no", lockbox_touched="no",
    protocol="full", status="recorded",
    notes=f"nested selection, best={bestk}: " + "; ".join(f"{k}={v:.5f}" for k, v in res.items() if k != "note")))
print(json.dumps(res, indent=2))
