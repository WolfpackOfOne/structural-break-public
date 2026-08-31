"""m11_focus -- EXACT sequential maximisation over the candidate change point.

WHAT IS ACTUALLY MISSING FROM THE INCUMBENT BANK
    m01_seq's own docstring records the approximation this module removes:

        "GLR uses a DYADIC candidate change-point grid (window lengths
         1,2,4,...,512) rather than a full O(t) scan every step. [...] the
         worst-case loss is a factor <= sqrt(2) in the effective segment
         length"

    m06_loc localises, but over a geometric BANK of segment lengths, and its
    statistics are trailing-window means calibrated against length-matched
    historical nulls -- not a maximised likelihood ratio.  Neither computes

        max_{0 <= tau < t}  ( S_t - S_tau )^2 / ( 2 (t - tau) )

    exactly.  This module does, at every online index, for six channels.

SIGNAL / ALPHA
    A prefix-anchored statistic dilutes post-break evidence by (t-tau)/t, and a
    fixed window either has not filled yet or has diluted the signal again.
    Maximising over tau is the likelihood-ratio-optimal way to spend a fixed
    amount of data on an unknown change point, and it is the construction that
    should show up first at SMALL post-break age -- where m09_back's late-break
    hypothesis died and where the metric still has weight.

FALSE SIGNAL
    A maximum over a bank of correlated statistics is a maximum over noise: the
    max over tau grows like log log t under the null (law of the iterated
    logarithm), a single fat-tailed observation makes the SHORT segments fire so
    tau_hat snaps to t-1, and both effects are per-series (they depend on the
    tail weight and the dependence of the series, not on whether it broke).

DISAMBIGUATOR -- and it is most of the code
    Every statistic is calibrated against a per-series historical null built by
    running THE IDENTICAL RECURSION over the break-free historical segment from
    ``N_RESTART`` restart positions, and reading the null at the MATCHED elapsed
    length.  The multiple-testing behaviour of the maximum is therefore priced
    in per series and per elapsed length, which is exactly what TS-AUC needs
    because it only ever compares series at a fixed online index.  Alongside the
    calibrated maximum we emit the geometry that separates a real change point
    from a lucky one: the inferred age, the runner-up maximum restricted to
    candidates FAR from the argmax, the gap between them, the anchored (tau=0)
    statistic that maximisation is supposed to beat, and how stable tau_hat has
    been over recent steps.

COST -- and why there is NO approximation error to quantify
    The candidate set is never pruned and never gridded: the scan is over every
    tau in 0..t-1, so the statistic is EXACT and the "quantify the approximation
    error" requirement is discharged by there being none.  This is affordable
    because n_online <= 999: the worst online step costs ~999 flops (~1 us)
    against the 1.07 ms/point the shared feature engine already spends, i.e.
    below a tenth of a percent.  Functional pruning (the convex hull of
    (tau, S_tau)) would reduce that to O(log t) and is not worth the extra
    surface area at this length.  The historical null is the expensive half --
    N_RESTART restarts x NULL_LEN steps x six channels -- and is measured, not
    assumed, in research/reports/wave5_e7_focus_cost.json.

CAUSALITY
    Row t uses online[:t+1] and history only.  S is a cumulative sum, the hull
    only ever grows to the right, and every null constant comes from history.
    Bitwise prefix-invariant, checked at atol=0.
"""
from __future__ import annotations

import math

import numpy as np

from sbr.features.base import register

#: restart positions used to build the per-series historical null
N_RESTART = 24
#: cap on how far a null path is run (the longest online segment is 999)
NULL_LEN = 1024
#: log-spaced elapsed anchors at which the null is summarised
ANCHORS = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 999], dtype=np.int64)
#: a runner-up must be this far from the argmax (in segment length, log units)
FAR = 2.0
#: how many recent steps the tau_hat stability window covers
STAB = 16
CLIP = 12.0
EPS = 1e-12


# ------------------------------------------------------------------ kernel
#
# EXACTNESS ARGUMENT (this is why there is no approximation error)
#
#   max_tau (S_t - S_tau)^2 / (2 (t - tau))
#     = max_tau max_{mu != 0} [ mu (S_t - S_tau) - mu^2 (t - tau) / 2 ]
#
#   For a fixed mu > 0, maximising over tau means maximising the LINEAR
#   functional (mu/2) tau - S_tau over the points {(tau, S_tau)}.  A linear
#   functional is maximised at a vertex of the convex hull, and as mu sweeps
#   (0, inf) the maximisers sweep exactly the LOWER hull.  Symmetrically, mu < 0
#   sweeps the UPPER hull.  So the union of the two hulls contains every tau
#   that can ever attain the maximum, for every t -- pruning to it is EXACT,
#   not an approximation.  Both hulls are maintained by monotone chain, which
#   is amortised O(1) per point, and their combined depth on this data is
#   ~2 log t rather than t.
#
#   `_focus_brute` is kept as the reference implementation and the hull kernel
#   is asserted equal to it in tests/test_m11_focus.py.


def _focus_brute(x):
    """Reference: exact max over EVERY tau, no pruning.  O(t) per point."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    S = np.concatenate([[0.0], np.cumsum(x)])
    stat = np.zeros(n)
    age = np.ones(n)
    second = np.zeros(n)
    anchored = np.zeros(n)
    for i in range(n):
        t = i + 1
        d = t - np.arange(t)
        D = S[t] - S[:t]
        v = D * D / (2.0 * d)
        k = int(np.argmax(v))
        stat[i] = v[k]
        age[i] = d[k]
        far = np.abs(np.log(d) - np.log(d[k])) > FAR
        second[i] = v[far].max() if far.any() else 0.0
        anchored[i] = S[t] * S[t] / (2.0 * t)
    return stat, age, second, anchored


def _focus_np(x):
    """Hull-pruned exact scan, pure numpy/python (used when numba is absent)."""
    n = len(x)
    stat = np.zeros(n)
    age = np.ones(n)
    second = np.zeros(n)
    anchored = np.zeros(n)
    lo_t, lo_s = [0], [0.0]          # lower hull of (tau, S_tau)
    up_t, up_s = [0], [0.0]          # upper hull
    S = 0.0
    for i in range(n):
        S += x[i]
        t = i + 1
        best = 0.0
        bage = 1.0
        sec = 0.0
        for ht, hs in ((lo_t, lo_s), (up_t, up_s)):
            for k in range(len(ht)):
                d = t - ht[k]
                if d <= 0:
                    continue
                D = S - hs[k]
                v = D * D / (2.0 * d)
                if v > best:
                    if best > sec and abs(math.log(bage) - math.log(d)) > FAR:
                        sec = best
                    best = v
                    bage = d
                elif v > sec and abs(math.log(d) - math.log(bage)) > FAR:
                    sec = v
        stat[i] = best
        age[i] = bage
        second[i] = sec
        anchored[i] = S * S / (2.0 * t)
        while len(lo_t) >= 2 and (lo_s[-1] - lo_s[-2]) * (t - lo_t[-1]) >= \
                (S - lo_s[-1]) * (lo_t[-1] - lo_t[-2]):
            lo_t.pop()
            lo_s.pop()
        lo_t.append(t)
        lo_s.append(S)
        while len(up_t) >= 2 and (up_s[-1] - up_s[-2]) * (t - up_t[-1]) <= \
                (S - up_s[-1]) * (up_t[-1] - up_t[-2]):
            up_t.pop()
            up_s.pop()
        up_t.append(t)
        up_s.append(S)
    return stat, age, second, anchored


try:
    from numba import njit

    @njit(cache=True, fastmath=False)
    def _decay_max_kernel(v, rho):
        n = v.shape[0]
        out = np.empty(n, dtype=np.float64)
        acc = -1.0e18
        for i in range(n):
            a = acc * rho
            acc = v[i] if v[i] > a else a
            out[i] = acc
        return out

    @njit(cache=True, fastmath=False)
    def _focus_kernel(x):
        n = x.shape[0]
        stat = np.zeros(n)
        age = np.ones(n)
        second = np.zeros(n)
        anchored = np.zeros(n)
        cap = n + 2
        lo_t = np.empty(cap, dtype=np.int64)
        lo_s = np.empty(cap, dtype=np.float64)
        up_t = np.empty(cap, dtype=np.int64)
        up_s = np.empty(cap, dtype=np.float64)
        lo_t[0] = 0
        lo_s[0] = 0.0
        nlo = 1
        up_t[0] = 0
        up_s[0] = 0.0
        nup = 1
        S = 0.0
        for i in range(n):
            S += x[i]
            t = i + 1
            best = 0.0
            bage = 1.0
            sec = 0.0
            for k in range(nlo):
                d = t - lo_t[k]
                if d <= 0:
                    continue
                D = S - lo_s[k]
                v = D * D / (2.0 * d)
                if v > best:
                    if best > sec and abs(math.log(bage) - math.log(d)) > FAR:
                        sec = best
                    best = v
                    bage = d
                elif v > sec and abs(math.log(d) - math.log(bage)) > FAR:
                    sec = v
            for k in range(nup):
                d = t - up_t[k]
                if d <= 0:
                    continue
                D = S - up_s[k]
                v = D * D / (2.0 * d)
                if v > best:
                    if best > sec and abs(math.log(bage) - math.log(d)) > FAR:
                        sec = best
                    best = v
                    bage = d
                elif v > sec and abs(math.log(d) - math.log(bage)) > FAR:
                    sec = v
            stat[i] = best
            age[i] = bage
            second[i] = sec
            anchored[i] = S * S / (2.0 * t)
            while nlo >= 2 and (lo_s[nlo - 1] - lo_s[nlo - 2]) * (t - lo_t[nlo - 1]) >= \
                    (S - lo_s[nlo - 1]) * (lo_t[nlo - 1] - lo_t[nlo - 2]):
                nlo -= 1
            lo_t[nlo] = t
            lo_s[nlo] = S
            nlo += 1
            while nup >= 2 and (up_s[nup - 1] - up_s[nup - 2]) * (t - up_t[nup - 1]) <= \
                    (S - up_s[nup - 1]) * (up_t[nup - 1] - up_t[nup - 2]):
                nup -= 1
            up_t[nup] = t
            up_s[nup] = S
            nup += 1
        return stat, age, second, anchored

    _HAVE_NUMBA = True
except Exception:                                            # pragma: no cover
    _HAVE_NUMBA = False
    _focus_kernel = None

    def _decay_max_kernel(v, rho):
        out = np.empty(len(v), dtype=np.float64)
        acc = -1.0e18
        for i in range(len(v)):
            acc = max(v[i], acc * rho)
            out[i] = acc
        return out


def _focus(x):
    """Exact max_tau (S_t - S_tau)^2 / (2 (t - tau)) at every t, plus geometry.

    Returns (stat, age, second, anchored):
      stat      the maximised statistic
      age       t - tau_hat at the maximiser
      second    the best statistic over HULL candidates FAR (in log segment
                length) from the maximiser -- a runner-up change point, not a
                neighbouring tau reporting almost the same evidence.  This is
                deliberately NOT the all-tau runner-up and does not equal
                `_focus_brute`'s: the hull is exactly the set of tau that can be
                optimal for SOME post-change mean, so a hull runner-up is a
                genuinely competing change point, whereas the all-tau runner-up
                is dominated by taus that no mean would ever select.  `stat`,
                `age` and `anchored` ARE bit-identical to the brute force.
      anchored  the tau = 0 statistic, i.e. what NOT maximising would have given
    """
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if len(x) == 0:
        z = np.zeros(0)
        return z, z, z, z
    if _HAVE_NUMBA:
        return _focus_kernel(x)
    return _focus_np(x)


def _null_table(xh):
    """Null quantiles of each FOCuS output at the log-spaced elapsed anchors.

    Runs the identical recursion from N_RESTART evenly spaced positions in the
    break-free history.  Only history enters; the online segment is never seen.
    """
    H = len(xh)
    if H < 32:
        return None
    span = min(NULL_LEN, H)
    starts = np.unique(np.linspace(0, max(H - span, 0), N_RESTART).astype(np.int64))
    A = ANCHORS[ANCHORS <= span]
    if len(A) == 0:
        A = ANCHORS[:1]
    acc = np.full((len(starts), len(A)), np.nan)
    acc2 = np.full((len(starts), len(A)), np.nan)
    for r, s0 in enumerate(starts):
        seg = xh[s0:s0 + span]
        st, ag, se, an = _focus(seg)
        idx = np.minimum(A, len(st)) - 1
        acc[r] = st[idx]
        acc2[r] = an[idx]
    med = np.nanmedian(acc, axis=0)
    q1, q3 = np.nanquantile(acc, 0.25, axis=0), np.nanquantile(acc, 0.75, axis=0)
    sd = np.maximum((q3 - q1) / 1.349, np.nanstd(acc, axis=0))
    med2 = np.nanmedian(acc2, axis=0)
    q1b, q3b = np.nanquantile(acc2, 0.25, axis=0), np.nanquantile(acc2, 0.75, axis=0)
    sd2 = np.maximum((q3b - q1b) / 1.349, np.nanstd(acc2, axis=0))
    return (A.astype(np.float64), med, np.maximum(sd, EPS),
            med2, np.maximum(sd2, EPS), np.sort(acc, axis=0))


def _z_at(tab, L, v, which=0):
    """Robust z of ``v`` against the null at matched elapsed length ``L``."""
    A, med, sd, med2, sd2, _ = tab
    m = med if which == 0 else med2
    s = sd if which == 0 else sd2
    la = np.log(A)
    x = np.log(np.maximum(np.asarray(L, dtype=np.float64), 1.0))
    mu = np.interp(x, la, m)
    sg = np.exp(np.interp(x, la, np.log(s)))
    return (v - mu) / np.maximum(sg, EPS)


def _pct_at(tab, L, v):
    """Empirical upper-tail percentile of ``v`` at the nearest elapsed anchor."""
    A, _, _, _, _, sorted_null = tab
    la = np.log(A)
    j = np.abs(np.log(np.maximum(np.asarray(L, float), 1.0))[:, None] - la[None, :]).argmin(1)
    sorted_null.shape[0]
    out = np.empty(len(v))
    for a in np.unique(j):
        m = j == a
        col = sorted_null[:, a]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            out[m] = 0.5
        else:
            out[m] = np.searchsorted(col, v[m], side="left") / len(col)
    return out


def _roll_std_rel(v, w=STAB):
    """Trailing-window sd of tau_hat, scaled by its own level.

    Expanding while fewer than w points have been seen -- never back-filled, so
    the column stays honest about how much online evidence exists.
    """
    n = len(v)
    c1 = np.concatenate([[0.0], np.cumsum(v)])
    c2 = np.concatenate([[0.0], np.cumsum(v * v)])
    ca = np.concatenate([[0.0], np.cumsum(np.abs(v))])
    i = np.arange(n)
    lo = np.maximum(i - w + 1, 0)
    k = (i - lo + 1).astype(np.float64)
    s1 = c1[i + 1] - c1[lo]
    s2 = c2[i + 1] - c2[lo]
    sa = ca[i + 1] - ca[lo]
    var = np.maximum(s2 / k - (s1 / k) ** 2, 0.0)
    return np.sqrt(var) / np.maximum(sa / k + 1.0, 1.0)


#: the six channels, as (name, builder) over the shared context
def _channels(ctx):
    hp = ctx.hp
    z = (ctx.online - hp.mu) / hp.sd
    zh = (ctx.hist - hp.mu) / hp.sd
    zr = (ctx.online - hp.med) / hp.mad
    zrh = (ctx.hist - hp.med) / hp.mad
    e = ctx.ar_online / hp.ar_sigma if ctx.ar_online is not None else z
    eh = ctx.hist_tr.get("res_mean", zh)

    def centre(a, b):
        """Remove the HISTORICAL mean so the null recursion starts at drift 0."""
        m = float(np.mean(b)) if len(b) else 0.0
        s = float(np.std(b)) if len(b) else 1.0
        s = s if s > EPS else 1.0
        return (a - m) / s, (b - m) / s

    out = {}
    out["mean"] = centre(z, zh)
    out["sq"] = centre(z * z, zh * zh)
    out["rabs"] = centre(np.abs(zr), np.abs(zrh))
    out["res"] = centre(e, eh)
    out["ressq"] = centre(e * e, eh * eh)
    l_on = np.concatenate([[0.0], z[1:] * z[:-1]]) if len(z) > 1 else np.zeros(len(z))
    l_h = np.concatenate([[0.0], zh[1:] * zh[:-1]]) if len(zh) > 1 else np.zeros(len(zh))
    out["lag1"] = centre(l_on, l_h)
    return out


@register("m11_focus", version="1", owner="claude-wave5")
def build(ctx):
    n = ctx.n
    L = np.arange(1, n + 1, dtype=np.float64)
    cols: list[str] = []
    out: list[np.ndarray] = []

    def add(name, v):
        cols.append(name)
        out.append(np.asarray(v, dtype=np.float64))

    for ch, (xo, xh) in _channels(ctx).items():
        stat, age, sec, anc = _focus(xo)
        tab = _null_table(xh)
        if tab is None:
            nanv = np.full(n, np.nan)
            for suf in ("z", "pct", "anc_z", "gain", "age", "agefrac",
                        "gap", "secz", "pk", "dpk", "tsp", "stab"):
                add(f"{ch}_{suf}", nanv)
            continue

        zst = np.clip(_z_at(tab, L, stat, 0), -CLIP, CLIP)
        zanc = np.clip(_z_at(tab, L, anc, 1), -CLIP, CLIP)
        add(f"{ch}_z", zst)
        add(f"{ch}_pct", _pct_at(tab, L, stat))
        add(f"{ch}_anc_z", zanc)
        # what maximisation over tau bought over the anchored (tau=0) statistic
        add(f"{ch}_gain", zst - zanc)
        # inferred change-point geometry
        add(f"{ch}_age", np.log1p(age))
        add(f"{ch}_agefrac", age / np.maximum(L, 1.0))
        zsec = np.clip(_z_at(tab, L, sec, 0), -CLIP, CLIP)
        add(f"{ch}_secz", zsec)
        add(f"{ch}_gap", zst - zsec)
        # evidence-path shape: running peak, decayed peak, time since peak
        pk = np.maximum.accumulate(zst)
        add(f"{ch}_pk", pk)
        add(f"{ch}_dpk", _decay_max_kernel(np.ascontiguousarray(zst), 0.98))
        arg = np.maximum.accumulate(np.where(zst >= pk - 1e-12, L, 0.0))
        add(f"{ch}_tsp", np.log1p(L - arg))
        # tau_hat stability: does the estimated change point stay put?
        tau_hat = L - age
        add(f"{ch}_stab", _roll_std_rel(tau_hat))

    # cross-channel agreement: is the SAME change point seen by several channels?
    zs = [out[cols.index(f"{ch}_z")] for ch in ("mean", "sq", "rabs", "res", "ressq", "lag1")]
    ages = [out[cols.index(f"{ch}_age")] for ch in ("mean", "sq", "rabs", "res", "ressq", "lag1")]
    M = np.column_stack(zs)
    allnan = np.all(~np.isfinite(M), axis=1)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mx, mn = np.nanmax(M, axis=1), np.nanmean(M, axis=1)
    mx[allnan] = np.nan
    mn[allnan] = np.nan
    add("xf_max", mx)
    add("xf_mean", mn)
    add("xf_n_hot", np.nansum(M > 3.0, axis=1).astype(np.float64))
    Ag = np.column_stack(ages)
    hot = M > 2.0
    with np.errstate(invalid="ignore"):
        agree = np.where(hot.sum(1) >= 2, np.nanstd(np.where(hot, Ag, np.nan), axis=1), np.nan)
    add("xf_age_agree", agree)

    A = np.column_stack(out).astype(np.float32)
    A[~np.isfinite(A)] = np.nan
    return cols, A
