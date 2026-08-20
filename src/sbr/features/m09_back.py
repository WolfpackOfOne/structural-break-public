"""m09_back -- backward (suffix-vs-prefix) two-sample evidence, for LATE breaks.

WHY THIS IS NOT `m00_core` WITH DIFFERENT WINDOWS
-------------------------------------------------
`m00_core` monitors a trailing window of length w against the **historical**
null: ``(M_suffix(w) - mu_null(w)) / sd_null(w)``.  That is a ONE-sample test.
Every column here is a TWO-sample test of the last ``k`` online points against
the *earlier online points of the same series*:

    D_k(t) = ( M_suffix(k) - M_prefix(t+1-k) )
             / sqrt( sd_null(k)^2 + sd_null(t+1-k)^2 )

The two differ exactly when the online segment as a whole sits somewhere other
than the historical null -- which the wave-1 context work established is common
(the series-level DGP fingerprint is worth AUC 0.53-0.54 on its own).  In that
situation a one-sample trailing statistic carries a persistent per-series offset
that is present *before and after* any break, while the suffix-prefix contrast
differences it out.  The pre-break online segment is a break-free sample drawn
from the same online regime, so for a LATE break it is a better-matched
reference than history is.

The scale comes from `ctx.nc`, whose nulls are empirical distributions of
rolling means over the historical segment, so the autocorrelation inflation of
a mean's variance is inherited per series and per length for free.  Variances
add because the suffix and prefix windows are disjoint.

SIGNAL / ALPHA
    A break at tau leaves the suffix drawn from the post-break law and the
    prefix (mostly) from the pre-break law.  Maximising over suffix length k
    adapts the test to an unknown tau instead of hoping a fixed window contains
    it, and the argmax doubles as a localisation estimate tau_hat = t - k*.

FALSE SIGNAL
    (a) A transient burst in the last few points inflates short-k D.  (b) Heavy
    tails do the same.  (c) Very short prefixes are noisy, so early t is
    unreliable.
DISAMBIGUATOR
    `_dkf` (where the change sits) separates "the last 4 points are odd" from
    "everything since the middle of the segment is different"; `_dsh` isolates
    the short-k arm so the tree can discount it; `_pre` reports how far the
    prefix itself sits from the historical null, which is the nuisance term the
    contrast removes and therefore the conditioning variable for whether the
    removal mattered; and rank (`u`) and robust (`abs`) channels bound the
    influence of any single outlier.

Prefix invariance: row t reads only ``cum[:t+2]`` and historical objects.
"""
from __future__ import annotations

import numpy as np

from sbr.features.base import register

#: Suffix lengths.  Chosen a priori to match `m00_core`'s window grid (8..256)
#: plus one shorter rung; NEVER tuned against a validation score.
KGRID = (4, 8, 16, 32, 64, 128, 256)

#: A prefix shorter than this is too noisy to be a reference sample.
MIN_PREFIX = 8

#: Channels spanning the break families: level, scale, distribution, and the
#: dependence-adjusted versions of the first two.
CHANNELS = (
    ("lv", "mean"),      # level shift
    ("sc", "abs"),       # scale change (robust)
    ("pit", "u"),        # distribution shift, rank-based
    ("rl", "res_mean"),  # level shift after AR whitening
    ("rs", "res_abs"),   # innovation scale
    ("dp", "lag1"),      # autocorrelation change
)

#: Features are meaningless past this magnitude; clipping stops a near-degenerate
#: null from producing a 1e7 column (see FAILED_EXPERIMENTS N7).
CLIP = 40.0


def _null_loc_scale(nc, name: str, L):
    """(mu, sd) of the length-L rolling-mean null, log-interpolated across the grid."""
    return nc._interp(name, L)


def _nanmax(A: np.ndarray, axis: int) -> np.ndarray:
    """np.nanmax that returns NaN for all-NaN slices instead of warning."""
    ok = np.isfinite(A).any(axis=axis)
    out = np.full(ok.shape, np.nan)
    if ok.any():
        sel = A[ok] if axis == 1 else A[:, ok]
        out[ok] = np.nanmax(sel, axis=axis)
    return out


@register("m09_back", version="1", owner="claude-wave3")
def build(ctx):
    n = ctx.n
    names: list[str] = []
    cols: list[np.ndarray] = []
    t = np.arange(n)              # online index
    L = t + 1.0                   # points observed at row t

    ks = [k for k in KGRID if k + MIN_PREFIX <= n]
    per_chan_absmax = []

    for tag, tname in CHANNELS:
        if tname not in ctx.cum:
            continue
        c = ctx.cum[tname]                       # cumsum with leading zero, len n+1
        tot = c[1:]                              # sum of online[0..t]
        # historical per-point scale of this transform, for the denominator floor
        s1 = float(np.std(ctx.hist_tr[tname])) if tname in ctx.hist_tr else 1.0
        floor = max(1e-4 * s1, 1e-9)

        D = np.full((n, len(ks)), np.nan)
        for j, k in enumerate(ks):
            m = L - k                            # prefix length at row t
            ok = m >= MIN_PREFIX                 # also guarantees t >= k
            if not ok.any():
                continue
            idx = np.flatnonzero(ok)
            suf = (c[idx + 1] - c[idx + 1 - k]) / k
            pre = (c[idx + 1 - k] - c[0]) / m[idx]
            _, sd_k = _null_loc_scale(ctx.nc, tname, k)
            _, sd_m = _null_loc_scale(ctx.nc, tname, m[idx])
            den = np.sqrt(np.asarray(sd_k, float) ** 2 + np.asarray(sd_m, float) ** 2)
            D[idx, j] = (suf - pre) / np.maximum(den, floor)
        np.clip(D, -CLIP, CLIP, out=D)

        nan = np.full(n, np.nan)
        if D.shape[1] == 0:
            # series (or truncated prefix) too short for any suffix/prefix split at all
            dmax = dkf = dsh = dlg = pre_z = nan
        else:
            A = np.abs(D)
            valid = np.isfinite(A).any(axis=1)
            star = np.zeros(n, dtype=np.int64)
            if valid.any():                   # a short prefix can leave no valid k at all
                star[valid] = np.nanargmax(A[valid], axis=1)
            rows = np.arange(n)

            dmax = np.where(valid, D[rows, star], np.nan)        # signed value at |argmax|
            kstar = np.asarray(ks, float)[star]
            dkf = np.where(valid, kstar / L, np.nan)             # tau_hat = t - k*, as a fraction

            short = [j for j, k in enumerate(ks) if k <= 16]
            long_ = [j for j, k in enumerate(ks) if k >= 64]
            dsh = _nanmax(A[:, short], axis=1) if short else nan
            dlg = _nanmax(A[:, long_], axis=1) if long_ else nan

            # the nuisance term the contrast removes: prefix vs HISTORY at the winning length
            mstar = np.maximum(L - kstar, 1.0)
            pre_star = np.where(valid, (c[np.maximum(rows + 1 - kstar.astype(np.int64), 0)]) / mstar, np.nan)
            pre_z = np.asarray(ctx.nc.z(tname, mstar, np.nan_to_num(pre_star, nan=0.0)), float)
            pre_z = np.where(valid, np.clip(pre_z, -CLIP, CLIP), np.nan)

        for nm, v in ((f"{tag}_d8", D[:, ks.index(8)] if 8 in ks else nan),
                      (f"{tag}_d32", D[:, ks.index(32)] if 32 in ks else nan),
                      (f"{tag}_d128", D[:, ks.index(128)] if 128 in ks else nan),
                      (f"{tag}_dmax", dmax),
                      (f"{tag}_dkf", dkf),
                      (f"{tag}_dsh", dsh),
                      (f"{tag}_dlg", dlg),
                      (f"{tag}_pre", pre_z)):
            names.append(nm)
            cols.append(v)
        per_chan_absmax.append(np.abs(dmax))

    # cross-channel agreement: how many families point the same way, continuously
    if per_chan_absmax:
        M = np.vstack(per_chan_absmax)
        ok = np.isfinite(M).any(axis=0)
        mx = _nanmax(M, axis=0)
        mn = -_nanmax(-M, axis=0)
        mean = np.full(n, np.nan)
        if ok.any():
            mean[ok] = np.nanmean(M[:, ok], axis=0)
        names.append("xb_max"); cols.append(mx)
        names.append("xb_mean"); cols.append(mean)
        names.append("xb_rng"); cols.append(mx - mn)

    out = np.empty((n, len(cols)), dtype=np.float32)
    for j, v in enumerate(cols):
        out[:, j] = np.asarray(v, dtype=np.float32)
    return names, out
