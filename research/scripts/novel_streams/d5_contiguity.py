"""D5: at MATCHED exceedance fraction, is contiguity the discriminator?
Compares the incumbent-style 'fraction of time above threshold' against the
'longest contiguous run' the bank does not compute."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import numpy as np

c = Ctx(); v = rt600(c)
M = np.load(f"{OUT}/d4_exc.npy", mmap_mode="r"); COLS = list(np.load(f"{OUT}/d4_cols.npy"))
jrun = COLS.index("res64_maxrun90")
r = c.rows[0]; k = cell_mask(c, r); rr = r[k]
y, t = c.d.y[rr], c.d.t[rr]
maxrun = np.nan_to_num(np.asarray(M[rr, jrun], np.float64), nan=0.0)
# incumbent-style companion: fraction of elapsed time spent above the same threshold
jr = COLS.index("res64_run90")
run_now = np.nan_to_num(np.asarray(M[rr, jr], np.float64), nan=0.0)
frac = maxrun / np.maximum(t, 1)     # crude proxy: longest run / elapsed

print(f"{'stat':>26s} {'cellAUC':>8s}")
for nm, x in (("res64_maxrun90 (longest run)", maxrun),
              ("res64_run90 (current run)", run_now),
              ("longest-run / elapsed", frac)):
    print(f"{nm:>26s} {ts_auc_flat(x, y, t):8.5f}")

# CONTIGUITY AT MATCHED EXCEEDANCE LOAD.
# Stratify by t-decile x exceedance-load decile; inside each stratum compare
# the longest contiguous run of positives vs negatives.
load = run_now  # current dwell
tb = np.digitize(t, [200,300,400,500,650,800,1000])
print("\nWithin t-bucket, mature-break vs never-break, matched on total exceedance load:")
hb_row = c.has_break[c.d.sidx[rr]]
nb = (y == 0) & ~hb_row      # never-break negatives only
pos = y == 1
for b in np.unique(tb):
    mb = tb == b
    if mb.sum() < 5000: continue
    lp, ln = maxrun[mb & pos], maxrun[mb & nb]
    if len(lp) < 100 or len(ln) < 100: continue
    # matched: restrict both to rows in the same total-elapsed band and compare medians
    print(f"  t-bucket {b}: n_pos={len(lp):7d} n_neg={len(ln):7d}  "
          f"median longest-run pos={np.median(lp):6.1f} neg={np.median(ln):6.1f}  "
          f"p90 pos={np.quantile(lp,0.9):6.1f} neg={np.quantile(ln,0.9):6.1f}")

# how many NEVER-BREAK series carry a long excursion at all?
sid = c.d.sidx[rr]
import collections
per_series_max = collections.defaultdict(float)
for s, mr in zip(sid, maxrun):
    if mr > per_series_max[s]: per_series_max[s] = mr
ser = np.array(sorted(per_series_max)); vals = np.array([per_series_max[s] for s in ser])
isnb = ~c.has_break[ser]
for thr in (25, 50, 100, 200):
    print(f"  share of series whose longest res64 excursion >= {thr:3d}: "
          f"never-break {np.mean(vals[isnb] >= thr):.3f}   break {np.mean(vals[~isnb] >= thr):.3f}")
