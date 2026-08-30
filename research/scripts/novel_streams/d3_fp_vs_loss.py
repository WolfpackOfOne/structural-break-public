"""D3: does RT-600's dominant-cell loss concentrate by historical DGP fingerprint?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import numpy as np
from scipy.stats import spearmanr

z = np.load(f"{OUT}/d1_series_loss.npz"); fp = np.load(f"{OUT}/d2_fingerprint.npz")
F, NAMES = fp["F"], list(fp["names"])
sl, sw, hb = z["series_loss"], z["series_weight"], z["has_break"]
rate = np.where(sw > 0, sl/np.maximum(sw,1), np.nan)

import pandas as pd
folds = pd.read_parquet(f"{ROOT}/research/folds/folds.parquet").fold.to_numpy()
dev = folds >= 0

for grp, name in ((~hb, "NEVER-BREAK (loss as negative)"), (hb, "BREAK (loss as mature positive)")):
    m = grp & dev & (sw > 500)
    print(f"\n=== {name}  n={m.sum()} mean loss-rate={rate[m].mean():.4f} sd={rate[m].std():.4f}")
    rows = []
    for j, nm in enumerate(NAMES):
        x = F[m, j]
        ok = np.isfinite(x)
        rho, p = spearmanr(x[ok], rate[m][ok])
        rows.append((abs(rho), rho, p, nm))
    rows.sort(reverse=True)
    for a, rho, p, nm in rows[:12]:
        print(f"   {nm:>14s}  spearman {rho:+.4f}   p={p:.2e}")

# --- out-of-fold predictability of the loss rate from history alone ---
import lightgbm as lgb
print("\n=== OOF predictability of dominant-cell loss rate from HISTORY ONLY ===")
for grp, name in ((~hb, "never-break"), (hb, "break")):
    m = np.flatnonzero(grp & dev & (sw > 500))
    X = F[m]; yv = rate[m]; f = folds[m]
    pred = np.full(len(m), np.nan)
    for k in range(5):
        tr, va = f != k, f == k
        b = lgb.train(dict(objective="regression", learning_rate=0.05, num_leaves=15,
                           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                           bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=8),
                      lgb.Dataset(X[tr], label=yv[tr]), num_boost_round=250)
        pred[va] = b.predict(X[va])
    rho, p = spearmanr(pred, yv)
    ss = 1 - ((yv-pred)**2).sum()/((yv-yv.mean())**2).sum()
    # permutation control
    rng = np.random.default_rng(0); yp = rng.permutation(yv)
    predp = np.full(len(m), np.nan)
    for k in range(5):
        tr, va = f != k, f == k
        b = lgb.train(dict(objective="regression", learning_rate=0.05, num_leaves=15,
                           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                           bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=8),
                      lgb.Dataset(X[tr], label=yp[tr]), num_boost_round=250)
        predp[va] = b.predict(X[va])
    rhop, _ = spearmanr(predp, yp)
    print(f"  {name:>12s}: OOF spearman {rho:+.4f} (perm control {rhop:+.4f})  R2={ss:+.4f}  n={len(m)}")
