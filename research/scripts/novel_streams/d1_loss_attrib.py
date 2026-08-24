"""D1: exact per-series inversion-loss attribution inside the dominant cell.

Uses the same pairwise machinery as sbr.metric (mid-rank ties), decomposed per row.
For each timestep t: positives lose (n_neg - concordant), negatives lose
(n_pos - concordant). Sum over rows -> per-series loss and per-series pair weight.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import numpy as np

c = Ctx()
v = rt600(c)
r = c.dev
k = cell_mask(c, r)
rr = r[k]
y = c.d.y[rr].astype(np.int8); t = c.d.t[rr]; s = v[rr]; sid = c.d.sidx[rr]

order = np.lexsort((s, t))
y_o, t_o, s_o, sid_o = y[order], t[order], s[order], sid[order]
gb = np.flatnonzero(np.r_[True, t_o[1:] != t_o[:-1]]); ge = np.r_[gb[1:], len(t_o)]

row_loss = np.zeros(len(t_o)); row_weight = np.zeros(len(t_o))
for lo, hi in zip(gb, ge):
    yy = y_o[lo:hi]; ss = s_o[lo:hi]
    p = yy == 1; n = ~p
    npos = int(p.sum()); nneg = int(n.sum())
    if npos == 0 or nneg == 0:
        continue
    sp = ss[p]; sn = np.sort(ss[n])
    # concordant count for each positive vs negatives (mid-rank ties)
    lo_i = np.searchsorted(sn, sp, side="left"); hi_i = np.searchsorted(sn, sp, side="right")
    conc_p = 0.5 * (lo_i + hi_i)
    idx = np.arange(lo, hi)
    row_loss[idx[p]] = nneg - conc_p
    row_weight[idx[p]] = nneg
    # for negatives: concordant = # positives ABOVE it
    spx = np.sort(sp)
    lo_j = np.searchsorted(spx, ss[n], side="left"); hi_j = np.searchsorted(spx, ss[n], side="right")
    conc_n = npos - 0.5 * (lo_j + hi_j)
    row_loss[idx[n]] = npos - conc_n
    row_weight[idx[n]] = npos

tot_w = row_weight.sum(); tot_l = row_loss.sum()
print(f"cell pair weight {tot_w:,.0f}  loss {tot_l:,.1f}  AUC {1-tot_l/tot_w:.5f}")

n_series = len(c.tau)
sl = np.bincount(sid_o, weights=row_loss, minlength=n_series)
sw = np.bincount(sid_o, weights=row_weight, minlength=n_series)
np.savez(f"{OUT}/d1_series_loss.npz", series_loss=sl, series_weight=sw,
         has_break=c.has_break, tau=c.tau, n_online=c.n_online, n_hist=c.n_hist)

hb = c.has_break
nb = ~hb
# negatives split: never-break series contribute all their rows as negatives;
# break series contribute pre-break rows as negatives and mature rows as positives.
print(f"never-break series: n={nb.sum()}  weight={sw[nb].sum():,.0f} ({sw[nb].sum()/tot_w:.3f})  loss={sl[nb].sum():,.0f} ({sl[nb].sum()/tot_l:.3f})")
print(f"break series:       n={hb.sum()}  weight={sw[hb].sum():,.0f} ({sw[hb].sum()/tot_w:.3f})  loss={sl[hb].sum():,.0f} ({sl[hb].sum()/tot_l:.3f})")

# concentration: how much of the cell's loss lives in the worst k% of series?
for grp, name in ((nb, "never-break"), (hb, "break")):
    idx = np.flatnonzero(grp & (sw > 0))
    rate = sl[idx] / np.maximum(sw[idx], 1)
    o = np.argsort(-sl[idx])
    cum = np.cumsum(sl[idx][o]) / sl[idx].sum()
    marks = [int(len(idx)*p) for p in (0.05, 0.10, 0.20, 0.50)]
    print(f"{name}: loss share of worst  5%={cum[marks[0]]:.3f} 10%={cum[marks[1]]:.3f} "
          f"20%={cum[marks[2]]:.3f} 50%={cum[marks[3]]:.3f}   mean loss-rate={rate.mean():.3f}")
