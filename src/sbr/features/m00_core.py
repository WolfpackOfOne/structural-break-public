"""m00_core -- calibrated multi-scale online-vs-historical-null evidence.

SIGNAL / ALPHA
    After a structural break the distribution of the online points stops
    matching the historical one.  Whatever moment or occupancy statistic the
    break moves (location, scale, tails, dependence, PIT shape), a trailing or
    expanding window of the online segment drifts away from the range that the
    same statistic occupies over break-free historical windows of the same
    length.  Calibrating against the *per-series* historical null is what makes
    the evidence comparable across series with wildly different volatility,
    tails and dependence -- and TS-AUC only ever compares series against each
    other at a fixed online index, so comparability is the whole game.

FALSE SIGNAL
    An isolated outlier, a transient volatility burst, or a temporary level
    excursion moves short windows exactly the same way.  The companion
    information that disambiguates them is scale: a real break moves the long
    windows and keeps moving them, a transient moves only the short ones and
    reverts.  That is why every statistic is emitted at several window lengths
    plus an expanding window, and why persistence/peak-decay channels are
    exposed in m01_seq rather than being collapsed into one number here.
"""
from __future__ import annotations

import warnings

import numpy as np

from sbr.features.base import register

WINDOWS = (8, 16, 32, 64, 128, 256)
# transforms monitored at every scale
TR_MULTI = ("mean", "sq", "abs", "u", "u2", "tail_x", "lag1")
# transforms monitored on the expanding window only (cheaper, slower-moving)
TR_EXP = ("mean", "sq", "abs", "rmean", "rabs", "u", "u2", "tail_hi", "tail_lo",
          "tail_x", "center", "sign", "cube", "quart", "logabs", "lag1", "lag2", "abslag1")

CLIP = 12.0


def _signed_surprise(p: np.ndarray) -> np.ndarray:
    two = 2.0 * np.minimum(p, 1.0 - p)
    s = -np.log10(np.maximum(two, 1e-6))
    return np.sign(p - 0.5) * s


@register("m00_core", version="1", owner="agent0")
def build(ctx):
    n = ctx.n
    cols: list[str] = []
    out: list[np.ndarray] = []

    L = ctx.idx()

    # ---- expanding-window evidence, robust-z against the length-matched null
    for name in TR_EXP:
        if name not in ctx.cum:
            continue
        v = ctx.expand(name)
        z = np.clip(ctx.nc.z(name, L, v), -CLIP, CLIP)
        cols.append(f"exp_z_{name}")
        out.append(z)
        cols.append(f"exp_absz_{name}")
        out.append(np.abs(z))

    # ---- trailing multi-scale evidence: exact empirical percentile + robust z
    grid = set(int(g) for g in ctx.nc.grid)
    for w in WINDOWS:
        if w not in grid:
            continue
        for name in TR_MULTI:
            if name not in ctx.cum:
                continue
            v = ctx.roll(name, w)
            ok = ~np.isnan(v)
            ss = np.full(n, np.nan)
            zz = np.full(n, np.nan)
            if ok.any():
                p = ctx.nc.pct(name, w, v[ok])
                ss[ok] = np.clip(_signed_surprise(p), -6, 6)
                zz[ok] = np.clip(ctx.nc.z(name, w, v[ok]), -CLIP, CLIP)
            cols.append(f"w{w}_sur_{name}")
            out.append(ss)
            cols.append(f"w{w}_z_{name}")
            out.append(zz)

    # ---- cross-scale aggregation: does the evidence agree across scales?
    for name in TR_MULTI:
        idx = [i for i, c in enumerate(cols) if c.startswith("w") and c.endswith(f"_z_{name}")]
        if len(idx) >= 2:
            M = np.abs(np.column_stack([out[i] for i in idx]))
            # every scale can still be NaN early in a series (not enough online
            # points for even the shortest window) -- an all-NaN row is the
            # correct answer here, not a problem, so do not shout about it
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                cols.append(f"xs_max_{name}")
                out.append(np.nanmax(M, axis=1))
                cols.append(f"xs_mean_{name}")
                out.append(np.nanmean(M, axis=1))
                cols.append(f"xs_min_{name}")
                out.append(np.nanmin(M, axis=1))

    # ---- short-vs-long contrast (transient vs persistent discrimination)
    for name in ("mean", "sq", "abs", "u2"):
        s = ctx.roll(name, 16)
        l = ctx.roll(name, 128)
        e = ctx.expand(name)
        cols.append(f"sl_{name}_16_128")
        out.append(s - l)
        cols.append(f"se_{name}_16_exp")
        out.append(s - e)

    # ---- how much online evidence exists at all (a conditioning variable,
    #      not a detector: TS-AUC compares at fixed t so this is constant
    #      within a timestep and cannot by itself change any ranking)
    cols.append("t_online")
    out.append(L)
    cols.append("log_t_online")
    out.append(np.log1p(L))

    A = np.column_stack(out).astype(np.float32)
    A[~np.isfinite(A)] = np.nan
    return cols, A
