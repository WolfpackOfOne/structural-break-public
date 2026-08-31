"""BASELINE 0 -- the currently shipped streaming detector, on OUR permanent folds.

Reproduces src/legacy_structural_break/realtime.py exactly (EWMA z + two-sided
CUSUM + EWMA variance ratio, noisy-OR) and scores it with the official
TS-AUC on each dev fold.  This is the number every later experiment is
measured against.
"""
import sys, math, json, time
import numpy as np
sys.path.insert(0, "/home/claude/sb/src")
from sbr.store import load_store
from sbr.pipeline import Data, evaluate_scores, append_result

ALPHA, SLACK, KM, KV = 0.05, 0.5, 3.0, 1.5

def detector(hist, online):
    mu = float(hist.mean()); sd = max(float(hist.std(ddof=1)), 1e-8)
    mu_e = mu; n_eff = 0.0; cp = 0.0; cn = 0.0; var_e = sd**2
    out = np.empty(len(online), dtype=np.float32)
    for i, x in enumerate(online):
        mu_e = (1-ALPHA)*mu_e + ALPHA*x
        n_eff = (1-ALPHA)*n_eff + 1.0
        se = sd/math.sqrt(max(n_eff,1.0))
        z = (mu_e-mu)/max(se,1e-8)
        ew = math.tanh(abs(z)/KM)
        r = (x-mu)/sd
        cp = max(0.0, cp+r-SLACK); cn = min(0.0, cn+r+SLACK)
        cu = math.tanh(max(cp,-cn)/(2.0*KM))
        me = max(ew, cu)
        var_e = (1-ALPHA)*var_e + ALPHA*(x-mu)**2
        ve = math.tanh(abs(math.log(max(var_e/sd**2,1e-8)))/KV)
        out[i] = 1.0-(1.0-me)*(1.0-ve)
    return out

st = load_store(); d = Data()
scores = np.full(len(d.y), np.nan, dtype=np.float32)
t0 = time.time()
n_on = st.meta.n_online.to_numpy()
for i in range(st.n_series):
    if d.series_fold[i] == -1:
        continue
    h, o, _ = st.series(i)
    s = int(st.orow_off[i])
    scores[s:s+n_on[i]] = detector(h, o)
mean, per, pooled = evaluate_scores(scores, d=d)
np.save("/home/claude/sb/research/oof/RT-000.npy", scores)
res = dict(experiment_id="RT-000", date=time.strftime("%Y-%m-%d %H:%M"), git_sha="baseline",
           agent="agent0", hypothesis="Shipped EWMA/CUSUM/var noisy-OR detector is a real signal on 2026 data",
           falsification_condition="TS-AUC <= 0.50 on dev folds", feature_set="none(handcrafted)",
           n_features=0, model="handcrafted_noisyOR", objective="none", folds="0,1,2,3,4",
           random_seed=0, train_series=0, train_rows=0, mean_oof_ts_auc=mean, pooled_oof_ts_auc=pooled,
           per_fold_ts_auc=";".join(f"{x:.5f}" for x in per), fold_std=float(np.std(per)),
           persistence="none", sample_mode="n/a", training_runtime_s=round(time.time()-t0,1),
           causal_verified="by construction (single pass)", test_reduced_touched="no",
           lockbox_touched="no", status="recorded", notes="BASELINE 0 replication on permanent folds")
append_result(res)
print(json.dumps({k: res[k] for k in ("mean_oof_ts_auc","pooled_oof_ts_auc","per_fold_ts_auc","fold_std","training_runtime_s")}, indent=2))
