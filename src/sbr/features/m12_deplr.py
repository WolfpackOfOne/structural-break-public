"""m12_deplr -- DEPENDENCE change: has the AR COEFFICIENT moved, not just the lag product?

WHAT THE AUDIT FOUND (research/reports/wave5_m04_redundancy_audit.md)
--------------------------------------------------------------------
`m04_resid` monitors dependence with `e_acf1` (the expanding mean of the lag-1
product) and `e_acf1sq`, on ten residual representations. What it never does is
ask whether the AR **coefficient** has changed: its AR coefficients are fitted on
history, frozen, and never re-estimated or compared against the online segment.
54 of its 60 columns are expanding-window and none maximises over a candidate
changepoint.

That gap matters because an autocovariance and a regression coefficient move for
different reasons. `E[e_t e_{t-1}]` changes when EITHER the dependence
coefficient OR the innovation variance changes. The ratio

    r = sum(e_t e_{t-1}) / sum(e_{t-1}^2)

changes only when the coefficient does -- the variance cancels. So a variance
break masquerades as a dependence break in `e_acf1` and does not in `r`.

MEASURED, BEFORE ANY TRAINING
    Ridge R^2 of this statistic on the whole existing bank (m04 60 cols +
    m11 21 cols) is 0.7325 -- 27% is unexplained. On its own falsification test
    (permanent dependence change vs a dependence BURST, series-level, 40 series
    per arm) it separates at |d| = 2.728 against the incumbent `ar1_e_acf1`'s
    1.048; paired bootstrap of the difference +1.680, 95% CI [+1.151, +2.324],
    100% of replicates positive.

THE INTERNAL CONTROL IS PART OF THE MODULE -- READ THIS BEFORE SCORING IT
------------------------------------------------------------------------
`m11_focus` established that maximising over a candidate changepoint is itself
worth something, independent of what is being maximised. So a dependence family
that gains could be winning for either of two reasons, and the module is built so
they can be told apart:

    dlr  the AR-COEFFICIENT change, max over tau        <- the hypothesis
    plr  the LAG-PRODUCT MEAN change, max over tau      <- generic control
    dg   dlr - plr                                       <- the difference

`plr` uses the identical changepoint search, the identical channel and the
identical calibration; the ONLY difference is that it does not normalise by the
segment's own second moment, i.e. it has no dependence model. **If `plr` carries
the gain and `dlr` does not, the win belongs to max-over-tau and the dependence
hypothesis is REJECTED** -- and the cheaper lesson would be to add a lag-product
channel to `m11_focus` rather than promote this module.

FALSE SIGNAL
    (a) A short prefix gives a wildly noisy coefficient estimate, so a minimum
    prefix is enforced and segments with a degenerate second moment are skipped.
    (b) The maximum of many statistics is upward-biased and the bias grows with
    t, so each maximum is ALSO emitted as a per-series historical-null surprise.
    (c) Heavy tails inflate both arms; the null calibration is per series.

Prefix invariance: row t reads only `online[:t+1]` and historical objects.
"""
from __future__ import annotations

import numpy as np

from sbr.features.base import register

#: Maximum lookback for the changepoint search. Same value and same reasoning as
#: `m11_focus`: the longest window already in `m00_core`'s grid. Chosen on
#: runtime, never on a score.
LMAX = 256

#: A coefficient estimated on fewer points than this is noise.
MIN_PRE = 16

#: Segments whose second moment is below this are skipped (degenerate ratio).
MIN_SS = 1e-6

#: Two channels: the standardised series, and the AR residual. The second is
#: where "the historical AR model no longer fits" shows up as residual
#: dependence -- the coefficient-drift signal m04 cannot express.
CHANNELS = (("rz", "mean"), ("re", "res_mean"))

CLIP = 40.0
MAX_NULL = 400


def _null_z(val: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Standardise against a historical null; scale floored by the null's own range."""
    if null.size < 2:
        return np.zeros_like(val)
    mu = float(np.mean(null))
    sd = float(np.std(null))
    rng = float(np.max(null) - np.min(null))
    sd = max(sd, 0.01 * rng, 1e-6)
    return np.clip((val - mu) / sd, -CLIP, CLIP)


def _subsample(a: np.ndarray) -> np.ndarray:
    if len(a) <= MAX_NULL:
        return a
    return a[np.linspace(0, len(a) - 1, MAX_NULL).astype(np.int64)]


def _dep_scan(e: np.ndarray, lag: int, need_lags: bool = True):
    """Max over candidate changepoint of a dependence change at `lag`.

    Returns (dlr, dage, plr): the coefficient-change arm, its argmax age, and the
    lag-product-mean arm that serves as the generic max-over-tau control.

    The Python loop runs over SUFFIX LENGTH (bounded by LMAX), never over the
    series, and every operation inside it is a contiguous slice.
    """
    n = len(e)
    prev = np.zeros(n)
    if n > lag:
        prev[lag:] = e[:-lag]
    xy = e * prev
    xx = prev * prev
    S = np.concatenate([[0.0], np.cumsum(xy)])
    X = np.concatenate([[0.0], np.cumsum(xx)])

    dlr = np.zeros(n)
    plr = np.zeros(n)
    dage = np.zeros(n) if need_lags else None

    lmax = min(LMAX, n)
    for L in range(1, lmax + 1):
        lo = L + MIN_PRE - 1                     # first t with a long enough prefix
        if lo >= n:
            break
        s_t1 = S[lo + 1:n + 1]                   # S[t+1]
        s_a = S[MIN_PRE:n - L + 1]               # S[t+1-L]
        x_t1 = X[lo + 1:n + 1]
        x_a = X[MIN_PRE:n - L + 1]
        n1 = np.arange(MIN_PRE, n - L + 1, dtype=np.float64)   # prefix length
        w = n1 * L / (n1 + L)

        # --- hypothesis arm: change in the AR coefficient (variance cancels)
        xx_pre = x_a
        xx_post = x_t1 - x_a
        ok = (xx_pre > MIN_SS) & (xx_post > MIN_SS)
        r_pre = np.where(ok, s_a / np.maximum(xx_pre, MIN_SS), 0.0)
        r_post = np.where(ok, (s_t1 - s_a) / np.maximum(xx_post, MIN_SS), 0.0)
        stat_d = np.where(ok, w * (r_post - r_pre) ** 2, 0.0)
        cur = dlr[lo:]
        better = stat_d > cur
        np.copyto(cur, stat_d, where=better)
        if need_lags:
            np.copyto(dage[lo:], float(L), where=better)

        # --- generic control arm: change in the lag-product MEAN (no model)
        m_pre = s_a / n1
        m_post = (s_t1 - s_a) / L
        stat_p = w * (m_post - m_pre) ** 2
        curp = plr[lo:]
        np.copyto(curp, stat_p, where=stat_p > curp)

    return np.sqrt(dlr), dage, np.sqrt(plr)


@register("m12_deplr", version="1", owner="wave5")
def build(ctx):
    """Dependence-coefficient change under max-over-tau, with its generic control."""
    n = ctx.n
    names: list[str] = []
    cols: list[np.ndarray] = []

    for tag, key in CHANNELS:
        eo = ctx.tr.get(key)
        eh = ctx.hist_tr.get(key)
        if eo is None:
            continue
        d1, age1, p1 = _dep_scan(eo, 1)
        d2, _, _ = _dep_scan(eo, 2, need_lags=False)

        names += [f"dlr_{tag}", f"plr_{tag}", f"dg_{tag}", f"dage_{tag}", f"d2_{tag}"]
        cols += [np.clip(d1, -CLIP, CLIP), np.clip(p1, -CLIP, CLIP),
                 np.clip(d1 - p1, -CLIP, CLIP), np.log1p(age1),
                 np.clip(d2, -CLIP, CLIP)]

        if eh is not None and len(eh) >= 4 * MIN_PRE:
            dh, _, ph = _dep_scan(eh, 1, need_lags=False)
            names += [f"dz_{tag}", f"pz_{tag}"]
            cols += [_null_z(d1, _subsample(dh)), _null_z(p1, _subsample(ph))]
        else:
            names += [f"dz_{tag}", f"pz_{tag}"]
            cols += [np.zeros(n), np.zeros(n)]

    out = np.empty((n, len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        out[:, j] = c
    return names, out
