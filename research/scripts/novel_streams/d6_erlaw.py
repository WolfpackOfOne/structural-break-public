"""D6 (descriptive, pre-declared family): normalise the longest observed exceedance
run against the series' OWN historical run-length distribution, extrapolated to the
elapsed online length via the Erdos-Renyi log-n growth law. Fold 0, dominant cell."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import numpy as np
from sbr.store import load_store
from sbr.transforms import HistParams, ar_filter_causal, _ar_resid

st = load_store(f"{ROOT}/cache/store"); S = len(st.meta)
W = 64
n_rows = int(st.meta.n_online.sum())
COLS = ["maxrun_raw", "maxrun_minus_ERexp", "maxrun_over_ERexp", "gumbel_z", "cur_run_gumbel_z"]
A = np.full((n_rows, len(COLS)), np.nan, dtype=np.float32)

def rollmean(c, w): return (c[w:] - c[:-w]) / w
def runs_of(mask):
    if not mask.any(): return np.array([0])
    ch = np.diff(np.r_[0, mask.view(np.int8), 0])
    return np.flatnonzero(ch == -1) - np.flatnonzero(ch == 1)

t0 = time.time(); pos = 0
for i in range(S):
    h = st.hist(i); o = st.online(i); n = len(o)
    hp = HistParams(h, ar_order=2)
    zh = (h - hp.mu)/hp.sd; zo = (o - hp.mu)/hp.sd
    eh = np.concatenate([np.zeros(2), _ar_resid(zh, hp.ar_coef)])/hp.ar_sigma
    eo = ar_filter_causal(zo, hp.ar_coef, zh)/hp.ar_sigma
    sh, so = eh*eh, eo*eo
    if len(sh) < 2*W or n < W:
        pos += n; continue
    nul = rollmean(np.concatenate([[0.0], np.cumsum(sh)]), W)
    med = np.median(nul); q90 = np.quantile(np.abs(nul - med), 0.90)
    # historical run-length law for the SAME exceedance event: p = P(exceed) = 0.10
    hr = runs_of(np.abs(nul - med) > q90).astype(float)
    H = len(nul)
    # Erdos-Renyi: E[longest run] ~ log_{1/p}(m) ; fit the scale from history's own longest run
    p_exc = 0.10
    lam = np.log(1.0/p_exc)
    er_hist = np.log(max(H,2)) / lam
    obs_hist = hr.max() if len(hr) else 0.0
    scale = obs_hist / max(er_hist, 1e-9)            # per-series dependence inflation factor
    # Gumbel scale for the longest head run is ~1/lam
    on = np.full(n, np.nan); on[W-1:] = rollmean(np.concatenate([[0.0], np.cumsum(so)]), W)
    hot = np.abs(on - med) > q90; hot[:W-1] = False
    cur = np.zeros(n); acc = 0
    for k in range(n):
        acc = acc + 1 if hot[k] else 0
        cur[k] = acc
    mx = np.maximum.accumulate(cur)
    m_eff = np.maximum(np.arange(1, n+1) - (W-1), 1)
    er_exp = scale * np.log(m_eff) / lam
    gz = (mx - er_exp) * lam                          # Gumbel-standardised
    gzc = (cur - er_exp) * lam
    A[pos:pos+n, 0] = mx
    A[pos:pos+n, 1] = mx - er_exp
    A[pos:pos+n, 2] = mx / np.maximum(er_exp, 1e-6)
    A[pos:pos+n, 3] = gz
    A[pos:pos+n, 4] = gzc
    pos += n
    if i % 2000 == 0: print(i, f"{time.time()-t0:.0f}s", flush=True)

c = Ctx(); v = rt600(c)
r = c.rows[0]; k = cell_mask(c, r); rr = r[k]
y, t = c.d.y[rr], c.d.t[rr]
from scipy.stats import rankdata
def wt_rank(x, t):
    o = np.lexsort((x, t)); out = np.empty(len(x)); ts = t[o]
    b = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]]); e = np.r_[b[1:], len(ts)]
    for lo, hi in zip(b, e): out[o[lo:hi]] = rankdata(x[o[lo:hi]])/(hi-lo)
    return out
br = wt_rank(v[rr], t)
print(f"\nRT600 fold-0 cell AUC {ts_auc_flat(v[rr],y,t):.5f}\n")
print(f"{'column':>22s} {'cellAUC':>8s} {'corr_RT600':>11s}")
for j, nm in enumerate(COLS):
    x = np.nan_to_num(np.asarray(A[rr, j], np.float64), nan=0.0)
    xr = wt_rank(x, t)
    print(f"{nm:>22s} {ts_auc_flat(x,y,t):8.5f} {float(np.corrcoef(xr,br)[0,1]):+11.4f}")
np.save(f"{OUT}/d6_er.npy", A)
