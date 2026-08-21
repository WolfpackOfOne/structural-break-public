"""Deployable score calibration.

The problem this solves: TS-AUC compares series against each other WITHIN a
timestep, so a blend of several models is only as good as the common scale it
puts them on. The offline oracle put them on that scale by ranking each model's
scores across the live cross-section at time t -- an object that does not exist
during a legal single-pass inference run.

The fix is a FROZEN transform, fitted once on training-fold OOF scores and
applied per series at inference:

    F_m(s | t)  ~=  P[ model m scores below s, among rows at online index ~t ]

estimated at a log-spaced set of time anchors and interpolated in log time
between the two neighbouring anchors. Each anchor's CDF is a 256-point quantile
grid, so the whole payload is 7 models x 12 anchors x 256 floats.

WHAT THIS IS ACTUALLY WORTH, measured in wave 4 (W4-E1). On seven heterogeneous
specialist streams the calibration family moves the blend by +0.00267 -- raw
mean 0.62314, logit mean 0.62479, global CDF 0.62550, this 0.62581. On seven
seed clones of one model it moves it by 0.00016. So this is not a general
improvement to blending: it is specifically a fix for members whose score scales
disagree, which heterogeneous streams have and seed clones do not. On the same
comparison the blend reached 0.62581 against the ILLEGAL within-timestep rank
oracle's 0.62580 -- it matches the ceiling rather than recovering a fraction of
it, and the older "recovers 99.7% of the oracle gain" phrasing is retired.

Cost at inference: two binary searches per model per observation.

TIME COORDINATE -- read before changing anything here.
The competition's online index is ZERO-BASED: the first scored point of every
series is t = 0. Two coordinates are supported and the payload records which one
it was fitted with, because a payload fitted under one and evaluated under the
other is silently wrong.

    "log_t_clamped"  the ORIGINAL. Anchor windows cut on raw t, interpolation on
                     log(max(t, 1)). It maps t=0 and t=1 to the same position,
                     and scores t=0 rows against an anchor grid that excluded
                     them. This is what the wave-2 artifact shipped with, so it
                     is preserved exactly and remains the default for any
                     payload that does not say otherwise.

    "log_n_seen"     the CORRECTED default for new fits. n_seen = t + 1 is the
                     number of online observations seen, strictly positive by
                     construction, used consistently for both the anchor windows
                     and the interpolation.

Cross-fitted over five folds on both wave-4 arms the two agree to six decimals
(specialist 0.625814 vs 0.625815, seed clone identical). The corrected one is
adopted for correctness at t=0, not for score -- TS-AUC at post-break age 0-5 is
0.513, so there was never much there to win. See research/RDOF_LEDGER.md W4-E3.
"""
from __future__ import annotations

import numpy as np

DEFAULT_ANCHORS = 12
DEFAULT_GRID = 256
MIN_N = 400

#: coordinate used for any payload that predates the field
LEGACY_COORD = "log_t_clamped"
#: coordinate used for new fits
DEFAULT_COORD = "log_n_seen"
COORDS = (LEGACY_COORD, DEFAULT_COORD)


class SmoothTimeCDFCal:
    """F_m(s | t): log-spaced time anchors, quantile grids, linear in log time."""

    def __init__(self, anchors, grids, time_coord=LEGACY_COORD):
        if time_coord not in COORDS:
            raise ValueError(f"unknown time_coord {time_coord!r}, expected one of {COORDS}")
        self.anchors = np.asarray(anchors, dtype=np.int64)
        self.grids = [np.asarray(g, dtype=np.float64) for g in grids]
        self.time_coord = time_coord
        self._la = np.log(np.maximum(self.anchors, 1).astype(np.float64))

    # ------------------------------------------------------------ coordinate
    def _u(self, t):
        """Online index -> evaluation coordinate."""
        if self.time_coord == DEFAULT_COORD:
            return float(t) + 1.0
        return max(float(t), 1.0)

    # ------------------------------------------------------------------- fit
    @classmethod
    def fit(cls, scores, t, n_anchor=DEFAULT_ANCHORS, grid=DEFAULT_GRID, min_n=MIN_N,
            time_coord=DEFAULT_COORD):
        scores = np.asarray(scores, dtype=np.float64)
        t = np.asarray(t)
        if time_coord == DEFAULT_COORD:
            cut = t.astype(np.float64) + 1.0        # n_seen, used for windows too
            top = max(float(cut.max()), 2.0)
        elif time_coord == LEGACY_COORD:
            cut = t.astype(np.float64)              # raw t, as originally shipped
            top = max(float(t.max()), 2.0)
        else:
            raise ValueError(f"unknown time_coord {time_coord!r}")
        anchors = np.unique(np.round(np.exp(
            np.linspace(np.log(1.0), np.log(top), n_anchor))).astype(int))
        la = np.log(np.maximum(anchors, 1).astype(np.float64))
        half = np.diff(la).mean() if len(anchors) > 1 else 1.0
        allg = _quantile_grid(scores, grid)
        grids = []
        for a in anchors:
            m = (cut >= a * np.exp(-half)) & (cut <= a * np.exp(half))
            g = scores[m]
            grids.append(_quantile_grid(g, grid) if len(g) >= min_n else allg)
        return cls(anchors, grids, time_coord=time_coord)

    # ------------------------------------------------------------- inference
    def __call__(self, s: float, t: int) -> float:
        la = self._la
        lt = np.log(self._u(t))
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
                "grids": [g.tolist() for g in self.grids],
                "time_coord": self.time_coord}

    @classmethod
    def from_json(cls, d):
        # a payload without the field predates it and is therefore legacy;
        # defaulting to the CORRECTED coordinate here would silently re-map
        # every shipped grid.
        return cls(d["anchors"], d["grids"], time_coord=d.get("time_coord", LEGACY_COORD))


def _quantile_grid(x, q):
    x = np.sort(np.asarray(x, dtype=np.float64))
    if len(x) == 0:
        return np.array([0.0])
    if len(x) <= q:
        return x
    return x[np.linspace(0, len(x) - 1, q).astype(np.int64)]
