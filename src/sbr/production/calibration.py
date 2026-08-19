"""Deployable score calibration.

The problem this solves: TS-AUC compares series against each other WITHIN a
timestep, so a blend of several models is only as good as the common scale it
puts them on. The offline oracle put them on that scale by ranking each model's
scores across the live cross-section at time t -- an object that does not exist
during a legal single-pass inference run.

The fix is a FROZEN transform, fitted once on training-fold OOF scores and
applied per series at inference:

    F_m(s | t)  ~=  P[ model m scores below s, among rows at online index ~t ]

estimated at a log-spaced set of time anchors and interpolated in log t between
the two neighbouring anchors. Each anchor's CDF is a 256-point quantile grid, so
the whole payload is 7 models x 12 anchors x 256 floats.

Measured: this recovers 99.7% of the oracle rank-average's gain over the best
single model, and the family is selected by nested CV rather than by looking at
the evaluation folds. See research/reports/deployable_ensemble_v2.json.

Cost at inference: two binary searches per model per observation.
"""
from __future__ import annotations

import numpy as np

DEFAULT_ANCHORS = 12
DEFAULT_GRID = 256
MIN_N = 400


class SmoothTimeCDFCal:
    """F_m(s | t): log-spaced time anchors, quantile grids, linear in log t."""

    def __init__(self, anchors, grids):
        self.anchors = np.asarray(anchors, dtype=np.int64)
        self.grids = [np.asarray(g, dtype=np.float64) for g in grids]
        self._la = np.log(np.maximum(self.anchors, 1).astype(np.float64))

    # ------------------------------------------------------------------- fit
    @classmethod
    def fit(cls, scores, t, n_anchor=DEFAULT_ANCHORS, grid=DEFAULT_GRID, min_n=MIN_N):
        scores = np.asarray(scores, dtype=np.float64)
        t = np.asarray(t)
        anchors = np.unique(np.round(np.exp(
            np.linspace(np.log(1), np.log(max(int(t.max()), 2)), n_anchor))).astype(int))
        half = np.diff(np.log(np.maximum(anchors, 1))).mean() if len(anchors) > 1 else 1.0
        allg = _quantile_grid(scores, grid)
        grids = []
        for a in anchors:
            m = (t >= a * np.exp(-half)) & (t <= a * np.exp(half))
            g = scores[m]
            grids.append(_quantile_grid(g, grid) if len(g) >= min_n else allg)
        return cls(anchors, grids)

    # ------------------------------------------------------------- inference
    def __call__(self, s: float, t: int) -> float:
        la = self._la
        lt = np.log(max(float(t), 1.0))
        j = int(np.searchsorted(la, lt, side="right"))
        j = min(max(j, 1), len(la) - 1)
        denom = la[j] - la[j - 1]
        w = 0.0 if denom <= 0 else min(max((lt - la[j - 1]) / denom, 0.0), 1.0)
        glo, ghi = self.grids[j - 1], self.grids[j]
        lo = float(np.searchsorted(glo, s, side="left")) / len(glo)
        hi = float(np.searchsorted(ghi, s, side="left")) / len(ghi)
        return (1.0 - w) * lo + w * hi

    # ------------------------------------------------------------------- io
    def to_json(self):
        return {"anchors": self.anchors.tolist(),
                "grids": [g.tolist() for g in self.grids]}

    @classmethod
    def from_json(cls, d):
        return cls(d["anchors"], d["grids"])


def _quantile_grid(x, q):
    x = np.sort(np.asarray(x, dtype=np.float64))
    if len(x) == 0:
        return np.array([0.0])
    if len(x) <= q:
        return x
    return x[np.linspace(0, len(x) - 1, q).astype(np.int64)]
