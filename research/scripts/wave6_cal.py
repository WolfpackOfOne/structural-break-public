"""Calibration with EXPLICIT anchor placement.  W6-E1 only.

`SCDF_ANCH` is a near-copy of `wave4_cal._SCDF` in its `SCDF_NSEEN` form --
`n_seen = t + 1`, no clamp, windows cut on the shifted coordinate -- with ONE
change: the anchors are supplied rather than computed as 12 log-spaced points.

**Everything else is held at the shipped values on purpose.**  The window
half-width is still `mean(diff(log(anchors)))`, a single global constant, which
is exactly the incumbent's rule and reduces to it identically when the anchors
are log-spaced.  Grid size (256) and `min_n` (400) are untouched.  So the only
quantity that varies across schemes is WHERE the anchors sit.

That the generalisation is faithful is not assumed: `CAL-LOG` must reproduce the
shipped `SCDF_NSEEN` score, and the driver asserts it.

A consequence worth watching rather than hiding: clustering anchors makes the
mean log gap smaller, so the windows narrow, so a grid can fall below `min_n`
and fall back to the pooled grid.  The driver counts those fallbacks and reports
them -- that is a real cost of clustering, not an artefact to be tuned away.
"""
from __future__ import annotations

import numpy as np

from wave4_cal import quantile_grid, GRID, MIN_N


class SCDF_ANCH:
    """Smooth time-conditional CDF calibration at explicit anchors."""

    def __init__(self, s, t, anchors, grid=GRID, min_n=MIN_N):
        s = np.asarray(s, dtype=np.float64)
        u = np.asarray(t, dtype=np.float64) + 1.0            # n_seen coordinate
        self.anchors = np.unique(np.asarray(anchors, dtype=np.int64))
        self.anchors = self.anchors[self.anchors >= 1]
        if len(self.anchors) == 0:
            self.anchors = np.array([1], dtype=np.int64)
        self._la = np.log(self.anchors.astype(np.float64))
        half = np.diff(self._la).mean() if len(self.anchors) > 1 else 1.0
        allg = quantile_grid(s, grid)
        self.grids = []
        self.n_fallback = 0
        for a in self.anchors:
            m = (u >= a * np.exp(-half)) & (u <= a * np.exp(half))
            g = s[m]
            if len(g) >= min_n:
                self.grids.append(quantile_grid(g, grid))
            else:
                self.grids.append(allg)
                self.n_fallback += 1

    def __call__(self, s, t):
        s = np.asarray(s, dtype=np.float64)
        lt = np.log(np.asarray(t, dtype=np.float64) + 1.0)
        la = self._la
        if len(la) == 1:
            g = self.grids[0]
            return np.searchsorted(g, s, side="left") / len(g)
        j = np.clip(np.searchsorted(la, lt, side="right"), 1, len(la) - 1)
        w = np.clip((lt - la[j - 1]) / np.maximum(la[j] - la[j - 1], 1e-12), 0.0, 1.0)
        lo = np.empty(len(s)); hi = np.empty(len(s))
        for idx in np.unique(j):
            m = j == idx
            glo, ghi = self.grids[idx - 1], self.grids[idx]
            lo[m] = np.searchsorted(glo, s[m], side="left") / len(glo)
            hi[m] = np.searchsorted(ghi, s[m], side="left") / len(ghi)
        return (1.0 - w) * lo + w * hi


# --------------------------------------------------------------- anchor schemes
N_ANCHOR = 12
SPLIT = 168          # the t below which the first 25% of pair weight sits (W5-D1)


def _pair_weight(t, y):
    """Official n_pos(t)*n_neg(t) per timestep, over the rows handed in."""
    o = np.lexsort((y, t))
    ts, ys = t[o], y[o]
    g = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]])
    e = np.r_[g[1:], len(ts)]
    npos = np.add.reduceat(ys.astype(np.int64), g)
    nall = e - g
    return ts[g], (npos * (nall - npos)).astype(np.float64)


def anchors_log(t, y, n=N_ANCHOR):
    """CAL-LOG -- the incumbent: n log-spaced anchors on [1, max n_seen]."""
    top = max(float((t + 1).max()), 2.0)
    return np.unique(np.round(np.exp(np.linspace(np.log(1.0), np.log(top), n))).astype(np.int64))


def anchors_weight(t, y, n=N_ANCHOR):
    """CAL-WT -- the (i+0.5)/n quantiles of the cumulative pair weight."""
    tv, w = _pair_weight(t, y)
    c = np.cumsum(w) / w.sum()
    q = (np.arange(n) + 0.5) / n
    a = tv[np.searchsorted(c, q, side="left").clip(0, len(tv) - 1)] + 1
    return np.unique(np.maximum(a, 1).astype(np.int64))


def anchors_hybrid(t, y, n=N_ANCHOR):
    """CAL-HYB -- half log-spaced below SPLIT, half weight-quantiles above it.

    Exists because CAL-WT may leave small t with no anchor at all, and those rows
    still have to be scored even though they carry little weight.
    """
    k = n // 2
    lo = np.unique(np.round(np.exp(
        np.linspace(np.log(1.0), np.log(float(SPLIT)), k))).astype(np.int64))
    tv, w = _pair_weight(t, y)
    m = tv > SPLIT
    if not m.any():
        return lo
    c = np.cumsum(w[m]) / w[m].sum()
    q = (np.arange(n - len(lo)) + 0.5) / (n - len(lo))
    hi = tv[m][np.searchsorted(c, q, side="left").clip(0, int(m.sum()) - 1)] + 1
    return np.unique(np.r_[lo, hi].astype(np.int64))


def anchors_null(t, y, seed, n=N_ANCHOR):
    """NULL-ANCHOR -- n log-uniform draws on [1, max n_seen].

    The seed-clone analogue for a calibration change: a DIFFERENT placement that
    carries no information about where the metric's weight is.  If a random
    re-placement gains as much as a weight-aware one, the incumbent placement was
    merely unlucky and the weight-aware story is unsupported.
    """
    top = max(float((t + 1).max()), 2.0)
    rng = np.random.default_rng(seed)
    a = np.exp(rng.uniform(np.log(1.0), np.log(top), size=n))
    return np.unique(np.maximum(np.round(a), 1).astype(np.int64))
