"""Per-point transforms shared by the null-calibration engine and the feature
modules.

Every transform maps a raw series to a same-length array whose *rolling mean*
is a statistic we care about.  Historical transforms are built from the
historical segment alone; online transforms must be built with historical
parameters only (never online ones), which keeps everything causal.
"""
from __future__ import annotations

import numpy as np


class HistParams:
    """Break-free historical characterisation used to standardise everything."""

    __slots__ = (
        "mu", "sd", "med", "mad", "q05", "q25", "q75", "q95", "q01", "q99",
        "iqr", "ecdf_x", "n", "ar1", "ar_coef", "ar_sigma", "abs_mu", "ewma_var",
    )

    def __init__(self, hist: np.ndarray, ar_order: int = 2):
        h = np.asarray(hist, dtype=np.float64)
        self.n = len(h)
        self.mu = float(h.mean())
        self.sd = max(float(h.std(ddof=1)) if len(h) > 1 else 1.0, 1e-9)
        self.med = float(np.median(h))
        self.mad = max(float(np.median(np.abs(h - self.med))) * 1.4826, 1e-9)
        qs = np.quantile(h, [0.01, 0.05, 0.25, 0.75, 0.95, 0.99])
        self.q01, self.q05, self.q25, self.q75, self.q95, self.q99 = map(float, qs)
        self.iqr = max(self.q75 - self.q25, 1e-9)
        self.abs_mu = float(np.mean(np.abs(h - self.med)))
        self.ecdf_x = np.sort(h)
        # AR fit on standardised history (fixed order; no per-series selection)
        z = (h - self.mu) / self.sd
        self.ar_coef = _fit_ar(z, ar_order)
        r = _ar_resid(z, self.ar_coef)
        self.ar_sigma = max(float(r.std(ddof=1)) if len(r) > 1 else 1.0, 1e-9)
        self.ar1 = float(self.ar_coef[0]) if len(self.ar_coef) else 0.0
        self.ewma_var = 1.0

    def pit(self, x: np.ndarray) -> np.ndarray:
        """Historical-ECDF probability integral transform, in (0,1)."""
        n = len(self.ecdf_x)
        lo = np.searchsorted(self.ecdf_x, x, side="left")
        hi = np.searchsorted(self.ecdf_x, x, side="right")
        return (0.5 * (lo + hi) + 0.5) / (n + 1.0)


def _fit_ar(z: np.ndarray, p: int) -> np.ndarray:
    if p <= 0 or len(z) < 10 * p + 10:
        return np.zeros(p)
    X = np.column_stack([z[p - k - 1: len(z) - k - 1] for k in range(p)])
    y = z[p:]
    XtX = X.T @ X + 1e-6 * np.eye(p) * len(y)
    try:
        return np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        return np.zeros(p)


def _ar_resid(z: np.ndarray, coef: np.ndarray) -> np.ndarray:
    p = len(coef)
    if p == 0:
        return z.copy()
    if len(z) <= p:
        return z.copy()
    X = np.column_stack([z[p - k - 1: len(z) - k - 1] for k in range(p)])
    return z[p:] - X @ coef


def ar_filter_causal(z: np.ndarray, coef: np.ndarray, warm: np.ndarray) -> np.ndarray:
    """AR residuals of ``z`` using ``warm`` (the tail of history) for lags.

    Fully causal: residual at position t uses only z[<=t] and warm.
    """
    p = len(coef)
    if p == 0:
        return z.copy()
    pad = np.concatenate([warm[-p:], z]) if len(warm) >= p else np.concatenate([np.zeros(p), z])
    X = np.column_stack([pad[p - k - 1: len(pad) - k - 1] for k in range(p)])
    return z - X @ coef


def build_transforms(x: np.ndarray, hp: HistParams, ar_resid: np.ndarray | None = None) -> dict:
    """Transforms whose rolling means are the statistics we monitor.

    ``x`` is raw; standardisation uses historical parameters only.
    """
    z = (x - hp.mu) / hp.sd
    zr = (x - hp.med) / hp.mad
    u = hp.pit(x)
    out = {
        "mean": z,                                   # location
        "sq": z * z,                                 # second moment about hist mean
        "abs": np.abs(z),                            # robust scale
        "rmean": zr,                                 # robust location
        "rabs": np.abs(zr),
        "tail_hi": (x > hp.q95).astype(np.float64),  # upper tail occupancy
        "tail_lo": (x < hp.q05).astype(np.float64),
        "tail_x": ((x > hp.q99) | (x < hp.q01)).astype(np.float64),
        "center": ((x > hp.q25) & (x < hp.q75)).astype(np.float64),
        "sign": np.sign(z),
        "u": u,                                      # PIT mean
        "u2": (u - 0.5) ** 2,                        # PIT dispersion
        "cube": np.clip(z, -8, 8) ** 3,              # skew-ish
        "quart": np.clip(z, -8, 8) ** 4,             # kurtosis-ish
        "logabs": np.log(np.abs(z) + 1e-3),
    }
    if ar_resid is not None:
        e = ar_resid / hp.ar_sigma
        out["res_mean"] = e
        out["res_sq"] = e * e
        out["res_abs"] = np.abs(e)
        out["res_lag1"] = np.concatenate([[0.0], e[1:] * e[:-1]])
        out["res_sq_lag1"] = np.concatenate([[0.0], (e[1:] ** 2) * (e[:-1] ** 2)])
    # lag products for ACF-style monitoring on the standardised raw series
    out["lag1"] = np.concatenate([[0.0], z[1:] * z[:-1]])
    out["lag2"] = np.concatenate([[0.0, 0.0], z[2:] * z[:-2]])
    out["abslag1"] = np.concatenate([[0.0], np.abs(z[1:]) * np.abs(z[:-1])])
    return out
