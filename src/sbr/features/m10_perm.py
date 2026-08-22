"""m10_perm -- TRANSIENT vs PERMANENT: is the displacement sustained or concentrated?

THE GAP THIS TARGETS
--------------------
`STATE_OF_RESEARCH_V4.md` §12 lists "transient vs permanent" as NOT RUN, and the
`m09_back` post-mortem measured why it matters: TS-AUC by post-break age is
**0.513 at age 0-5** and 0.646 at age 100+.  Young breaks are where the metric's
mass is lost, and the reason is not that the evidence is weak -- it is that at
small post-break age a genuine level shift and a two-point outlier burst produce
*the same* trailing-window mean.

WHY THIS IS NOT `m00_core` WITH DIFFERENT COLUMNS
-------------------------------------------------
`m00_core` monitors the **mean** of a transform over a trailing window against a
length-matched historical null.  A window mean is invariant to the ARRANGEMENT of
its points: 60% of points displaced by 1 sigma and 6% displaced by 10 sigma give
the same mean, and the same trailing-window surprise.  They are different events.
One is a regime change; the other is a burst.

Every column here is a functional of that arrangement and is therefore
**information the window-mean bank cannot express at any window length**:

* `xn` -- crossing count.  How often the series changes side of the historical
  median inside the window.  A permanent shift moves the whole window to one
  side and CROSSINGS COLLAPSE; a burst leaves the crossing rate untouched.
  Note this is not the mean of `sign`: a window with 60% positives scattered and
  one with 60% positives in a single block have the identical mean sign and very
  different crossing counts.
* `mx` -- run length.  The longest one-sided run visible in the window, the dual
  statistic to crossings and the one that survives when the median itself is
  poorly estimated.
* `cn` -- concentration.  The share of the window's total |z| mass carried by its
  single largest point.  A burst is concentrated (share -> 1); a sustained shift
  is diffuse (share -> 1/w).  This is the direct transient/permanent contrast.
* `md` -- mean-vs-MEDIAN displacement contrast.  A window median is not an
  affine function of the window mean, so the two respond differently to a burst:
  a sustained shift moves both, a three-point spike moves only the mean.  This
  is the transient/permanent contrast in its most direct form.

  A NEGATIVE RESULT IS BAKED INTO THIS CHOICE.  The obvious version -- mean
  versus `m00_core`'s ROBUST mean `rmean` -- is **identically zero and can never
  carry information**.  `(x-mu)/sd` and `(x-med)/mad` are affine images of one
  another; a window mean of an affine map is affine in the window mean of `x`;
  and null-standardising each arm against its own null removes any affine map
  exactly.  Measured max |null_z(mean) - null_z(rmean)| = 1.8e-15 on heavy-tailed
  t(3) data -- floating-point noise.  The median version measures 4.0 on the same
  input.  Any future "robust vs plain" contrast in this project must use a
  non-affine robust statistic or it is a column of zeros.

FALSE SIGNAL
    (a) A heavy-tailed series crosses its median often for reasons that have
    nothing to do with a break, so raw crossing counts are meaningless across
    series -- every column is emitted as a per-series historical-null surprise
    at the SAME window length, which is what makes them comparable at a fixed
    online index (the quantity TS-AUC actually ranks).
    (b) A slow drift raises `mx` without any break.  `xn` and `mx` are duals and
    disagree under drift, which is why both are emitted.
    (c) Near-degenerate nulls: the scale is floored by a fraction of the null's
    own range, per FAILED_EXPERIMENTS N7, never by a bare epsilon.

WHAT THIS DELIBERATELY DOES NOT DO
    No slope or trend channel (FAILED_EXPERIMENTS N1 and agent7: dead, twice).
    No fixed thresholding of a continuous statistic (N2: discretising for a GBM
    is self-harm).  No re-statement of a divergence already in `m02_dist`, which
    already carries Wasserstein, energy, KS, CvM, JS, Hellinger and TV against a
    length-matched null.

Prefix invariance: row t reads only `online[:t+1]` and historical objects.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from sbr.features.base import register

#: Trailing windows.  Chosen a priori to match `m00_core`'s existing grid
#: (8..256).  NEVER tuned against a validation score.
WGRID = (8, 16, 32, 64, 128, 256)

#: Windows for the mean-vs-median contrast.  Restricted to the SHORT rungs on a
#: mechanism argument fixed in advance, not on any score: a burst of <=3 points
#: cannot move a 128- or 256-point window mean far enough for the contrast to
#: mean anything, so the long rungs would be noise columns.
MGRID = (8, 16, 32, 64)

#: Deterministic cap on historical null start positions per window length.
MAX_NULL = 400

#: Features are meaningless past this magnitude (FAILED_EXPERIMENTS N7).
CLIP = 40.0


def _trail(x: np.ndarray, w: int) -> np.ndarray | None:
    """Exact trailing windows of length w: row i covers x[i .. i+w-1]."""
    if w > len(x):
        return None
    return sliding_window_view(x, w)


def _null_z(online_val: np.ndarray, null_vals: np.ndarray) -> np.ndarray:
    """Standardise an online statistic against its historical null distribution.

    The scale is floored by a fraction of the null's OWN range rather than by a
    bare epsilon: FAILED_EXPERIMENTS N7 records a column that reached 3e7
    because a near-degenerate null was floored at 1e-9.
    """
    if null_vals.size < 2:
        return np.zeros_like(online_val)
    mu = float(np.mean(null_vals))
    sd = float(np.std(null_vals))
    rng = float(np.max(null_vals) - np.min(null_vals))
    sd = max(sd, 0.01 * rng, 1e-6)
    return np.clip((online_val - mu) / sd, -CLIP, CLIP)


def _subsample(a: np.ndarray) -> np.ndarray:
    """Deterministically thin a null sample to MAX_NULL rows (no RNG)."""
    if len(a) <= MAX_NULL:
        return a
    idx = np.linspace(0, len(a) - 1, MAX_NULL).astype(np.int64)
    return a[idx]


def _runlen(side: np.ndarray) -> np.ndarray:
    """Current consecutive-run length of a +/-1 sign array. Causal by construction."""
    n = len(side)
    out = np.ones(n, dtype=np.float64)
    for i in range(1, n):
        out[i] = out[i - 1] + 1.0 if side[i] == side[i - 1] else 1.0
    return out


def _stats_for_windows(z: np.ndarray, zr: np.ndarray, side: np.ndarray,
                       run: np.ndarray, w: int):
    """(xn, mx, cn) over every trailing window of length w. NaN before w points."""
    n = len(z)
    xn = np.full(n, np.nan)
    mx = np.full(n, np.nan)
    cn = np.full(n, np.nan)
    if w > n:
        return xn, mx, cn
    # crossings: indicator that the side changed at i, summed over the window
    ch = np.zeros(n)
    ch[1:] = (side[1:] != side[:-1]).astype(np.float64)
    c = np.concatenate([[0.0], np.cumsum(ch)])
    xn[w - 1:] = c[w:] - c[:-w]
    # longest run visible in the window (run is itself causal)
    mx[w - 1:] = _trail(run, w).max(axis=1)
    # concentration: share of total |z| mass held by the single largest point
    az = np.abs(z)
    ca = np.concatenate([[0.0], np.cumsum(az)])
    tot = ca[w:] - ca[:-w]
    top = _trail(az, w).max(axis=1)
    cn[w - 1:] = np.where(tot > 1e-12, top / np.maximum(tot, 1e-12), np.nan)
    return xn, mx, cn


@register("m10_perm", version="1", owner="wave5")
def build(ctx):
    """Transient-vs-permanent arrangement statistics, null-calibrated per series."""
    hp = ctx.hp
    n = ctx.n

    # ---- online channels -------------------------------------------------
    z = ctx.tr["mean"]                       # (x - mu_hist) / sd_hist
    zr = ctx.tr["rmean"]                     # (x - med_hist) / mad_hist
    side_o = np.where(ctx.online > hp.med, 1.0, -1.0)
    run_o = _runlen(side_o)

    # ---- historical counterparts (the null sample) -----------------------
    h = ctx.hist
    zh = (h - hp.mu) / hp.sd
    zrh = (h - hp.med) / hp.mad
    side_h = np.where(h > hp.med, 1.0, -1.0)
    run_h = _runlen(side_h)

    names: list[str] = []
    cols: list[np.ndarray] = []

    # ---- global (window-free) run state ----------------------------------
    # log1p keeps a 400-long run from dominating the split search.
    names.append("run_cur")
    cols.append(np.log1p(run_o).astype(np.float64))
    names.append("run_sgn")
    cols.append(side_o.astype(np.float64))

    for w in WGRID:
        xn_o, mx_o, cn_o = _stats_for_windows(z, zr, side_o, run_o, w)
        xn_h, mx_h, cn_h = _stats_for_windows(zh, zrh, side_h, run_h, w)
        valid_h = ~np.isnan(xn_h)
        xn_hn = _subsample(xn_h[valid_h])
        mx_hn = _subsample(mx_h[valid_h])
        cn_hn = _subsample(cn_h[~np.isnan(cn_h)])

        names.append(f"xn{w}")
        cols.append(_null_z(xn_o, xn_hn))
        names.append(f"mx{w}")
        cols.append(_null_z(np.log1p(mx_o), np.log1p(mx_hn)))
        names.append(f"cn{w}")
        cols.append(_null_z(cn_o, cn_hn))

        # mean-vs-median contrast: see the module docstring for why the
        # robust-mean version of this is provably a column of zeros.
        if w in MGRID:
            mo = ctx.roll("mean", w)
            mh = _trail(zh, w)
            mo_win = _trail(z, w)
            if mh is None or mo_win is None:
                names.append(f"md{w}")
                cols.append(np.full(n, np.nan))
            else:
                med_o = np.full(n, np.nan)
                med_o[w - 1:] = np.median(mo_win, axis=1)
                zmean = _null_z(mo, _subsample(mh.mean(axis=1)))
                zmed = _null_z(med_o, _subsample(np.median(mh, axis=1)))
                names.append(f"md{w}")
                cols.append(np.clip(zmean - zmed, -CLIP, CLIP))

    out = np.empty((n, len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        out[:, j] = c
    return names, out
