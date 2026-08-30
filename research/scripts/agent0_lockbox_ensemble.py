"""RT-130-LOCKBOX -- final confirmation of the ensemble on the untouched 2,000 series.

The blend is a parameter-free equal-weight average of within-timestep rank
percentiles, so there is nothing here that could have been tuned on the lockbox.
Each of the four streams is retrained on all five development folds and applied
once to the lockbox.
"""
import sys, time, json; sys.path.insert(0, "/home/claude/sb/src")
import numpy as np, lightgbm as lgb
from sbr.pipeline import Data, load_features, _stack, append_result
from sbr.metric import ts_auc_flat

FULL = ["m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes"]
STREAMS = [
    ("RT-100", FULL, dict(n_estimators=600, learning_rate=0.05, num_leaves=63, min_data_in_leaf=300,
                          feature_fraction=0.5, bagging_fraction=0.7, max_bin=127), 0, "uniform"),
    ("RT-120", ["m00_core", "m01_seq", "m07_bayes"],
     dict(n_estimators=700, learning_rate=0.04, num_leaves=127, min_data_in_leaf=150,
          feature_fraction=0.35, bagging_fraction=0.7, max_bin=127), 0, "uniform"),
    ("RT-121", ["m02_dist", "m03_dyn", "m04_resid", "m06_loc"],
     dict(n_estimators=500, learning_rate=0.08, num_leaves=31, min_data_in_leaf=500,
          feature_fraction=0.7, bagging_fraction=0.7, max_bin=127), 1, "uniform"),
    ("RT-122", FULL, dict(n_estimators=500, learning_rate=0.06, num_leaves=255, min_data_in_leaf=400,
                          feature_fraction=0.3, bagging_fraction=0.6, max_bin=63, extra_trees=True), 7, "per_series"),
]
COMMON = dict(objective="binary", bagging_freq=1, lambda_l2=5.0, num_threads=2, verbose=-1)

d = Data()
lb = d.rows_for([-1])
t_lb = d.t[lb].astype(np.int64); y_lb = d.y[lb]


def within_t_rank(s, t):
    order = np.lexsort((s, t)); n = len(s); ts = t[order]
    g = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]]); glen = np.r_[g[1:], n] - g
    pos = np.arange(n) - np.repeat(g, glen)
    new = np.r_[True, (ts[1:] != ts[:-1]) | (s[order][1:] != s[order][:-1])]
    rs = np.flatnonzero(new); re = np.r_[rs[1:], n]
    avg = np.repeat(pos[rs] + (re - rs - 1) / 2.0, re - rs)
    out = np.empty(n); out[order] = avg / np.maximum(np.repeat(glen, glen) - 1, 1)
    return out


t0 = time.time()
ranks, singles = [], {}
for tag, mods, pars, seed, mode in STREAMS:
    mats, names = load_features(mods)
    cols = np.arange(len(names))
    rng = np.random.default_rng(seed)
    tr = d.rows_for([0, 1, 2, 3, 4])
    if mode == "per_series":
        s = d.sidx[tr]; bnd = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
        per = 900_000 // len(bnd); ends = np.r_[bnd[1:], len(s)]
        sel = [np.arange(b, e) if e - b <= per else rng.choice(np.arange(b, e), per, replace=False)
               for b, e in zip(bnd, ends)]
        tr = np.sort(tr[np.concatenate(sel)])
    else:
        tr = np.sort(rng.choice(tr, 900_000, replace=False))
    p = dict(COMMON); p.update({k: v for k, v in pars.items() if k != "n_estimators"})
    X = _stack(mats, names, tr, cols)
    ds = lgb.Dataset(X, label=d.y[tr], params=p)
    b = lgb.train(p, ds, num_boost_round=int(pars["n_estimators"]))
    del X, ds
    pred = np.empty(len(lb), dtype=np.float64)
    for s0 in range(0, len(lb), 400_000):
        pred[s0:s0 + 400_000] = b.predict(_stack(mats, names, lb[s0:s0 + 400_000], cols))
    del b
    singles[tag] = float(ts_auc_flat(pred, y_lb, t_lb))
    ranks.append(within_t_rank(pred, t_lb))
    print(tag, singles[tag], round(time.time() - t0), flush=True)

blend = np.mean(np.column_stack(ranks), axis=1)
score = float(ts_auc_flat(blend, y_lb, t_lb))
out = {"lockbox_ensemble_ts_auc": score, "lockbox_singles": singles,
       "dev_oof_ensemble": 0.6237438123699494, "dev_oof_champion_single": 0.6149965423542226,
       "gap_vs_dev": score - 0.6237438123699494, "runtime_s": round(time.time() - t0)}
np.save("/home/claude/sb/research/oof/RT-130.lockbox.npy", blend.astype(np.float32))
json.dump(out, open("/home/claude/sb/research/reports/lockbox_ensemble.json", "w"), indent=2)
append_result(dict(experiment_id="RT-130-LOCKBOX", date=time.strftime("%Y-%m-%d %H:%M"), git_sha="nogit",
    agent="agent0", hypothesis="The four-stream rank-average ensemble holds up on series never used for anything",
    falsification_condition="lockbox ensemble below the single-model lockbox score of 0.60791",
    feature_set="RT-100+RT-120+RT-121+RT-122", n_features=500, model="rank-average ensemble",
    objective="n/a", folds="train=0-4, eval=lockbox", random_seed=0, mean_oof_ts_auc=score,
    pooled_oof_ts_auc=score, per_fold_ts_auc=f"{score:.5f}", fold_std=0.0, persistence="none",
    sample_mode="mixed", training_runtime_s=round(time.time() - t0, 1),
    causal_verified="prefix-invariance@module", test_reduced_touched="no",
    lockbox_touched="YES (second authorised touch, parameter-free blend)", protocol="lockbox",
    status="recorded", notes=json.dumps(singles)))
print(json.dumps(out, indent=2))
