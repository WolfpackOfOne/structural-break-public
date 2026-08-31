"""LOCKBOX CONFIRMATION -- run exactly once, after all decisions are made.

Trains the champion configuration on all 5 development folds and scores the
2,000 lockbox series that no experiment, no feature-selection step and no
hyperparameter choice has ever seen.  Its only job is to tell us how much of
the development-fold number is research overfit.
"""
import sys, json, time; sys.path.insert(0, "/home/claude/sb/src")
import numpy as np, lightgbm as lgb
from sbr.pipeline import Data, load_features, _stack, append_result
from sbr.metric import ts_auc_flat

FULL = ["m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes"]
P = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=300,
         feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0,
         num_threads=2, verbose=-1, max_bin=127)
N_ROUND = 600

t0 = time.time()
d = Data()
mats, names = load_features(FULL)
cols = np.arange(len(names))
rng = np.random.default_rng(0)

tr = d.rows_for([0, 1, 2, 3, 4])
tr = np.sort(rng.choice(tr, 1_000_000, replace=False))
X = _stack(mats, names, tr, cols)
ds = lgb.Dataset(X, label=d.y[tr], params=P)
b = lgb.train(P, ds, num_boost_round=N_ROUND)
del X, ds
print("trained", round(time.time() - t0), "s", flush=True)

lb = d.rows_for([-1])
pred = np.empty(len(lb), dtype=np.float32)
step = 400_000
for s in range(0, len(lb), step):
    pred[s:s + step] = b.predict(_stack(mats, names, lb[s:s + step], cols)).astype(np.float32)

score = float(ts_auc_flat(pred, d.y[lb], d.t[lb]))
dev_mean = 0.6151034246587253
out = {"lockbox_ts_auc": score, "dev_mean_oof": dev_mean, "gap": score - dev_mean,
       "n_lockbox_series": 2000, "n_lockbox_rows": int(len(lb)), "runtime_s": round(time.time() - t0)}
np.save("/home/claude/sb/research/oof/RT-100.lockbox.npy", pred)
json.dump(out, open("/home/claude/sb/research/reports/lockbox_confirmation.json", "w"), indent=2)
append_result(dict(experiment_id="RT-100-LOCKBOX", date=time.strftime("%Y-%m-%d %H:%M"), git_sha="nogit",
    agent="agent0", hypothesis="Champion dev-fold TS-AUC is not research overfit",
    falsification_condition="lockbox TS-AUC materially below the 0.6151 dev mean",
    feature_set=",".join(FULL), n_features=len(names), model="lgbm", objective="binary",
    folds="train=0-4, eval=lockbox", random_seed=0, mean_oof_ts_auc=score, pooled_oof_ts_auc=score,
    per_fold_ts_auc=f"{score:.5f}", fold_std=0.0, persistence="none", sample_mode="uniform",
    training_runtime_s=round(time.time() - t0, 1), causal_verified="prefix-invariance@module",
    test_reduced_touched="no", lockbox_touched="YES (single authorised touch)", protocol="lockbox",
    status="recorded", notes="Single authorised lockbox evaluation, after all decisions were fixed."))
print(json.dumps(out, indent=2))
