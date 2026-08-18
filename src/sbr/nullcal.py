"""Per-series historical-null calibration engine.

The historical segment is guaranteed break-free, so it is an empirical null for
*any* statistic we compute online.  This engine makes detector evidence
comparable across heterogeneous series -- which is exactly what Time-Stratified
AUC rewards, since scores are only ever compared cross-sectionally at a fixed
online index.

Everything here is a rolling mean of a per-point transform of the series, which
is what makes it cheap:  mean, second moment, variance, absolute deviation,
tail rates, sign, PIT moments, lag products (ACF) and slopes are all obtainable
from cumulative sums of a handful of transforms.

For a statistic S computed over a trailing window of length w, the null is the
empirical distribution of S over every contiguous length-w window of the
historical segment.  We store those nulls sorted for a log-spaced grid of w and
expose

    pct(name, w, value)   exact empirical percentile (grid w only)
    z(name, L, value)     robust z vs a null whose location/scale is
                          log-interpolated across the grid (any L)

Causality: only historical data enters the null; online values only ever enter
as the query.
"""
from __future__ import annotations

import numpy as np

# log-spaced window grid, capped at run time by the historical length
WINDOW_GRID = np.array([5, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024])


def _rolling_mean(c: np.ndarray, w: int) -> np.ndarray:
    """Rolling mean of the series whose cumsum-with-leading-zero is ``c``."""
    return (c[w:] - c[:-w]) / w


class NullCal:
    """Historical-null distributions for a single series."""

    __slots__ = ("transforms", "grid", "null_sorted", "null_mu", "null_sd", "n_hist")

    def __init__(self, hist_transforms: dict[str, np.ndarray], max_window: int | None = None):
        h = len(next(iter(hist_transforms.values())))
        self.n_hist = h
        cap = h // 2 if max_window is None else min(max_window, h // 2)
        self.grid = WINDOW_GRID[WINDOW_GRID <= max(cap, 5)]
        if len(self.grid) == 0:
            self.grid = np.array([5])
        self.transforms = list(hist_transforms)
        self.null_sorted: dict[tuple[str, int], np.ndarray] = {}
        self.null_mu: dict[str, np.ndarray] = {}
        self.null_sd: dict[str, np.ndarray] = {}
        for name, arr in hist_transforms.items():
            a = np.asarray(arr, dtype=np.float64)
            c = np.concatenate([[0.0], np.cumsum(a)])
            mus = np.empty(len(self.grid))
            sds = np.empty(len(self.grid))
            for j, w in enumerate(self.grid):
                w = int(w)
                if w >= h:
                    rm = np.array([a.mean()])
                else:
                    rm = _rolling_mean(c, w)
                rs = np.sort(rm)
                self.null_sorted[(name, w)] = rs
                mus[j] = np.median(rs)
                q1, q3 = np.quantile(rs, [0.25, 0.75])
                iqr_sd = (q3 - q1) / 1.349
                sds[j] = max(iqr_sd, rs.std(), 1e-12)
            self.null_mu[name] = mus
            self.null_sd[name] = sds

    def nearest_w(self, L: int) -> int:
        return int(self.grid[np.argmin(np.abs(np.log(self.grid) - np.log(max(L, 1))))])

    def pct(self, name: str, w: int, value) -> np.ndarray:
        """Empirical percentile in [0,1] of ``value`` under the length-w null."""
        rs = self.null_sorted[(name, int(w))]
        v = np.atleast_1d(np.asarray(value, dtype=np.float64))
        lo = np.searchsorted(rs, v, side="left")
        hi = np.searchsorted(rs, v, side="right")
        return (0.5 * (lo + hi)) / len(rs)

    def _interp(self, name: str, L):
        lg = np.log(self.grid.astype(np.float64))
        x = np.log(np.clip(np.atleast_1d(np.asarray(L, dtype=np.float64)), 1, None))
        mu = np.interp(x, lg, self.null_mu[name])
        sd = np.exp(np.interp(x, lg, np.log(self.null_sd[name])))
        return mu, sd

    def z(self, name: str, L, value) -> np.ndarray:
        """Robust z of ``value`` against the length-L null (L may be any length)."""
        mu, sd = self._interp(name, L)
        v = np.atleast_1d(np.asarray(value, dtype=np.float64))
        return (v - mu) / np.maximum(sd, 1e-12)

    def surprise(self, name: str, w: int, value) -> np.ndarray:
        """Two-sided -log10 empirical p-value, floored by the null sample size."""
        p = self.pct(name, w, value)
        n = len(self.null_sorted[(name, int(w))])
        two = 2.0 * np.minimum(p, 1.0 - p)
        return -np.log10(np.maximum(two, 1.0 / (2.0 * n)))
