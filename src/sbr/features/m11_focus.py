"""m11_focus -- exact maximum-likelihood change detection with an ADAPTIVE baseline.

WHAT IS ACTUALLY NEW HERE, AND WHAT IS NOT
------------------------------------------
`m01_seq` already carries a max-GLR channel, and it is strong: `glz_pk` takes
3.56% of total gain and `gle_pkr` 3.86%. That statistic maximises the Gaussian
GLR over a DYADIC grid of candidate windows, with the pre-change mean taken from
the HISTORICAL segment. Two things are approximated there, and this module
removes them one at a time so their contributions can be told apart:

1. **The grid.** `hg` maximises over EVERY lag up to `LMAX`, not over dyadic
   rungs. This is a refinement of an existing strong channel and is expected to
   be worth little on its own; it is emitted mainly as the matched control for
   the column below, so that any gain from (2) cannot be an artefact of (1).

2. **The baseline — this is the hypothesis.** `ag` estimates the PRE-change
   level from the series' own online prefix instead of from history, and
   maximises the two-sample statistic

       n1*n2/(n1+n2) * (mean_post - mean_pre)^2 / sigma^2

   jointly over the changepoint. Every detector in the production bank
   calibrates against the HISTORICAL null, so each carries the series'
   persistent online-vs-historical offset in every window. Wave 1 established
   that offset is real and large (the series-level DGP fingerprint is worth
   AUC 0.53-0.54 on its own). An adaptive baseline differences it out.

`dg = hg - ag` is therefore a direct read on the nuisance term: large when the
apparent break is mostly the online segment sitting somewhere history did not
predict, small when the break is visible against the series' own recent regime.

WHY THIS IS NOT `m09_back`, WHICH FAILED
----------------------------------------
`m09_back` also used an online-prefix reference, and was rejected. Its post
mortem is specific about the mechanism of failure: it fixed the suffix length
`k` on an a-priori grid and contrasted suffix against prefix, so at small
post-break age the suffix was short and the contrast was dominated by prefix
noise; six of its top eight columns turned out to be the `_pre` nuisance term
rather than the contrast. Here the changepoint is not fixed and not gridded --
it is MAXIMISED over, which is what makes the estimator adaptive to an unknown
tau instead of hoping a fixed window contains it, and the maximiser itself
(`hl`, `al`) is emitted as a post-change age estimate rather than discarded.

That is a different estimator, not a rerun. It may still fail; W5-E2 is
pre-registered with the seed-clone bar that killed `m09_back`.

FALSE SIGNAL
    (a) The maximum of many statistics is upward-biased and the bias grows with
    `t`, so raw maxima are not comparable across online index -- every maximum
    is ALSO emitted as a per-series historical-null surprise computed by running
    the identical statistic over the break-free historical segment.
    (b) A heavy-tailed series produces large maxima with no break at all; the
    null calibration is per series and absorbs this.
    (c) An adaptive baseline is itself contaminated once the prefix contains the
    break, which is exactly why the historical-baseline arm is retained rather
    than replaced.

Prefix invariance: row t reads only `online[:t+1]` and historical objects.
Cost is bounded by LMAX and vectorised over lags, never over a Python loop in t.
"""
from __future__ import annotations

import numpy as np

from sbr.features.base import register

#: Maximum lookback for the changepoint search: the longest window already in
#: `m00_core`'s grid. A break older than this is comfortably visible to the
#: expanding-window bank, so searching further buys little and costs linearly.
#:
#: Set from a RUNTIME measurement taken before any score existed -- at 512 the
#: module cost 85.8 ms/series against the ~80 ms budget in PROTOCOL.md. It has
#: NEVER been tuned against a validation score, and no validation data existed in
#: the environment when it was chosen.
LMAX = 256

#: Minimum prefix length before an adaptive baseline is meaningful.
MIN_PRE = 8

#: Channels: level, AR-innovation level, robust scale.
CHANNELS = (("lv", "mean"), ("rs", "res_mean"), ("sc", "abs"))

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


def _focus(x: np.ndarray, need_lags: bool = True):
    """Exact max-GLR over every lag <= LMAX, under both baselines.

    The Python loop runs over LAGS (bounded by LMAX) and never over the series.
    Every operation inside it is a CONTIGUOUS SLICE, not fancy indexing: for lag
    l the valid rows are t = l-1 .. n-1, which is exactly `s[l:] - s[:-l]`. The
    fancy-indexed form of this cost 1.6x the per-series budget.

    Returns (hg, hl, ag, al); hg/ag are on a sqrt (standard-deviation) scale.

    hist baseline : pre-change mean is 0 in the caller's units, so the statistic
                    for lag l is (sum of the last l values)^2 / l.
    adaptive      : both means estimated from the online segment, giving the
                    two-sample statistic n1*n2/(n1+n2) * (post-pre)^2.
    """
    n = len(x)
    s = np.concatenate([[0.0], np.cumsum(x)])          # s[k] = sum(x[:k])
    hg = np.zeros(n)
    ag = np.zeros(n)
    hl = np.zeros(n) if need_lags else None
    al = np.zeros(n) if need_lags else None
    lmax = min(LMAX, n)
    for lag in range(1, lmax + 1):
        # rows t = lag-1 .. n-1, all contiguous slices
        d = s[lag:] - s[:n + 1 - lag]                      # sum over the last lag points
        stat_h = d * d / lag
        cur = hg[lag - 1:]
        better = stat_h > cur
        np.copyto(cur, stat_h, where=better)
        if need_lags:
            np.copyto(hl[lag - 1:], float(lag), where=better)

        n1 = np.arange(n + 1 - lag, dtype=np.float64)    # prefix length, t+1-lag
        ok = n1 >= MIN_PRE
        if not ok.any():
            continue
        denom = np.maximum(n1, 1.0)
        pre = s[:n + 1 - lag] / denom
        post = d / lag
        w = n1 * lag / (n1 + lag)
        stat_a = w * (post - pre) ** 2
        cura = ag[lag - 1:]
        better_a = ok & (stat_a > cura)
        np.copyto(cura, stat_a, where=better_a)
        if need_lags:
            np.copyto(al[lag - 1:], float(lag), where=better_a)
    if not need_lags:
        return np.sqrt(hg), None, np.sqrt(ag), None
    return np.sqrt(hg), hl, np.sqrt(ag), al


@register("m11_focus", version="1", owner="wave5")
def build(ctx):
    """Exact changepoint-maximised evidence under historical vs adaptive baselines."""
    n = ctx.n
    names: list[str] = []
    cols: list[np.ndarray] = []

    for tag, key in CHANNELS:
        xo = ctx.tr[key]
        xh = ctx.hist_tr.get(key)
        # centre both arms on the historical mean of the channel so that the
        # "historical baseline" arm really has a zero pre-change mean.
        mu_h = float(np.mean(xh)) if xh is not None and len(xh) else 0.0
        hg, hl, ag, al = _focus(xo - mu_h)

        names += [f"hg_{tag}", f"ag_{tag}", f"dg_{tag}",
                  f"hl_{tag}", f"al_{tag}"]
        cols += [np.clip(hg, -CLIP, CLIP), np.clip(ag, -CLIP, CLIP),
                 np.clip(hg - ag, -CLIP, CLIP), np.log1p(hl), np.log1p(al)]

        # per-series historical null: identical statistic over the break-free
        # historical segment, which is what makes the maxima comparable across
        # series at a fixed online index.
        if xh is not None and len(xh) >= 4 * MIN_PRE:
            hgh, _, agh, _ = _focus(xh - mu_h, need_lags=False)
            names += [f"hz_{tag}", f"az_{tag}"]
            cols += [_null_z(hg, _subsample(hgh)), _null_z(ag, _subsample(agh))]
        else:
            names += [f"hz_{tag}", f"az_{tag}"]
            cols += [np.zeros(n), np.zeros(n)]

    out = np.empty((n, len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        out[:, j] = c
    return names, out
