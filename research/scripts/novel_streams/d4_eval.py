import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import numpy as np
from scipy.stats import rankdata

c = Ctx(); v = rt600(c)
M = np.load(f"{OUT}/d4_exc.npy", mmap_mode="r"); COLS = list(np.load(f"{OUT}/d4_cols.npy"))
r = c.rows[0]                      # fold 0 only
k = cell_mask(c, r); rr = r[k]
y, t = c.d.y[rr], c.d.t[rr]
base = v[rr]
print(f"fold-0 cell rows {len(rr):,}  RT600 cell AUC {ts_auc_flat(base,y,t):.5f}")

# within-t rank helper
def wt_rank(x, t):
    o = np.lexsort((x, t)); out = np.empty(len(x))
    ts = t[o]; b = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]]); e = np.r_[b[1:], len(ts)]
    for lo, hi in zip(b, e):
        out[o[lo:hi]] = rankdata(x[o[lo:hi]]) / (hi - lo)
    return out

br = wt_rank(base, t)
print(f"{'column':>22s} {'cellAUC':>8s} {'corr_RT600':>11s} {'blendAUC':>9s} {'delta':>8s}")
res = []
for j, nm in enumerate(COLS):
    x = np.asarray(M[rr, j], dtype=np.float64)
    if not np.isfinite(x).any(): continue
    x = np.nan_to_num(x, nan=-1.0)
    if x.std() == 0: continue
    a = ts_auc_flat(x, y, t)
    xr = wt_rank(x, t)
    corr = float(np.corrcoef(xr, br)[0,1])
    bl = ts_auc_flat(0.75*br + 0.25*xr, y, t)
    res.append((bl, nm, a, corr))
    print(f"{nm:>22s} {a:8.5f} {corr:+11.4f} {bl:9.5f} {bl-ts_auc_flat(br,y,t):+8.5f}")
print(f"\nRT600 rank-only cell AUC {ts_auc_flat(br,y,t):.5f}")
# all 12 columns jointly, on top of RT600 rank, tiny LGBM, fold-0-internal 2-fold by series
import lightgbm as lgb
sid = c.d.sidx[rr]; us = np.unique(sid); half = set(us[::2].tolist())
m1 = np.array([s in half for s in sid])
X = np.column_stack([np.nan_to_num(np.asarray(M[rr, j], dtype=np.float64), nan=-1.0) for j in range(len(COLS))])
for nmX, XX in (("RT600+t only", np.column_stack([br, t])), ("RT600+t+exc12", np.column_stack([br, t, X]))):
    p = np.empty(len(rr))
    for msk in (m1, ~m1):
        b = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
                           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                           verbose=-1, num_threads=8),
                      lgb.Dataset(XX[~msk], label=y[~msk]), num_boost_round=200)
        p[msk] = b.predict(XX[msk])
    print(f"  {nmX:>16s}: cell AUC {ts_auc_flat(p,y,t):.5f}")
