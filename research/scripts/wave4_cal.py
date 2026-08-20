"""Calibration families for wave 4, vectorised for research use.

Two smooth time-conditional CDF variants live here:

  SCDF_T     the INCUMBENT.  Anchors and interpolation in log(max(t, 1)).
  SCDF_NSEEN the CANDIDATE.  Anchors and interpolation in log(t + 1).

Why the candidate exists.  The competition's online index is ZERO-BASED, so the
first scored point of every series is t = 0.  `log(max(t,1))` maps t = 0 and
t = 1 to the same position, which silently merges the two youngest online
timesteps and leaves the lowest anchor doing double duty.  `n_seen = t + 1` is
the number of online observations the detector has actually seen, is strictly
positive by construction, and needs no clamp.

This is a BUGFIX CANDIDATE under the W4-E3 pre-registration, not a tuning knob.
Anchor count, grid size and min_n are FROZEN at the production values (12 / 256 /
400) in both variants; only the coordinate changes.  The production class
sbr.production.calibration.SmoothTimeCDFCal is byte-compatible with SCDF_T and is
what actually ships; SCDF_T here is the vectorised twin used for research, and
test_wave4_cal_parity asserts they agree.
"""
from __future__ import annotations

import numpy as np

N_ANCHOR, GRID, MIN_N = 12, 256, 400


def quantile_grid(x, q=GRID):
    x = np.sort(np.asarray(x, dtype=np.float64))
    if len(x) == 0:
        return np.array([0.0])
    if len(x) <= q:
        return x
    return x[np.linspace(0, len(x) - 1, q).astype(np.int64)]


class _SCDF:
    """Shared machinery.  Subclasses only define the time coordinate.

    Note the deliberate asymmetry in the incumbent: its anchor WINDOWS are cut on
    raw t while its INTERPOLATION uses log(max(t,1)).  That is what
    sbr.production.calibration.SmoothTimeCDFCal does and what shipped, so the
    control arm has to reproduce it exactly, bug and all.  The candidate uses
    n_seen = t + 1 consistently in both places, which is the actual fix: under
    the incumbent a t = 0 row is scored by the anchor-1 grid but was excluded
    from building it.
    """

    SHIFT = 0        #: added to t before logs
    CLAMP = True     #: clamp to >= 1 at evaluation (incumbent only)
    FIT_ON_SHIFTED = False   #: cut anchor windows on the shifted coordinate too

    def __init__(self, s, t, n_anchor=N_ANCHOR, grid=GRID, min_n=MIN_N):
        s = np.asarray(s, dtype=np.float64)
        t = np.asarray(t)
        top = self._u(t).max() if self.FIT_ON_SHIFTED else max(float(t.max()), 2.0)
        top = max(float(top), 2.0)
        self.anchors = np.unique(np.round(np.exp(
            np.linspace(np.log(1.0), np.log(top), n_anchor))).astype(np.int64))
        self._la = np.log(self.anchors.astype(np.float64))
        half = np.diff(self._la).mean() if len(self.anchors) > 1 else 1.0
        cut = self._u(t) if self.FIT_ON_SHIFTED else t.astype(np.float64)
        allg = quantile_grid(s, grid)
        self.grids = []
        for a in self.anchors:
            m = (cut >= a * np.exp(-half)) & (cut <= a * np.exp(half))
            g = s[m]
            self.grids.append(quantile_grid(g, grid) if len(g) >= min_n else allg)

    @classmethod
    def _u(cls, t):
        """Map the online index t to the calibration's evaluation coordinate."""
        u = np.asarray(t, dtype=np.float64) + cls.SHIFT
        return np.maximum(u, 1.0) if cls.CLAMP else u

    def __call__(self, s, t):
        s = np.asarray(s, dtype=np.float64)
        lt = np.log(self._u(np.asarray(t)))
        la = self._la
        j = np.clip(np.searchsorted(la, lt, side="right"), 1, len(la) - 1)
        w = np.clip((lt - la[j - 1]) / np.maximum(la[j] - la[j - 1], 1e-12), 0.0, 1.0)
        lo = np.empty(len(s)); hi = np.empty(len(s))
        for idx in np.unique(j):
            m = j == idx
            glo, ghi = self.grids[idx - 1], self.grids[idx]
            lo[m] = np.searchsorted(glo, s[m], side="left") / len(glo)
            hi[m] = np.searchsorted(ghi, s[m], side="left") / len(ghi)
        return (1.0 - w) * lo + w * hi


class SCDF_T(_SCDF):
    """INCUMBENT: log(max(t, 1)), anchor windows cut on raw t.  Byte-compatible
    with the shipped SmoothTimeCDFCal.  t=0 and t=1 land on the same position."""
    SHIFT, CLAMP, FIT_ON_SHIFTED = 0, True, False


class SCDF_NSEEN(_SCDF):
    """CANDIDATE: log(t + 1) everywhere.  Zero-based online index, no clamp."""
    SHIFT, CLAMP, FIT_ON_SHIFTED = 1, False, True


class GlobalCDF:
    """F_m(s), no time conditioning."""

    def __init__(self, s, grid=GRID):
        self.grid = quantile_grid(s, grid)

    def __call__(self, s, t=None):
        return np.searchsorted(self.grid, np.asarray(s, dtype=np.float64),
                               side="left") / len(self.grid)


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))
