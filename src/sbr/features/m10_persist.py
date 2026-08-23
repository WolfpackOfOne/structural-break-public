"""m10_persist -- is the evidence carried by a FEW POINTS or by the BULK, and
does it persist?

WHY THIS MODULE LOOKS LIKE THIS AND NOT LIKE THE BRIEF
    The wave-5 brief asked for a generic transient-vs-persistent family: peak,
    current, peak/current ratio, time since peak, decay rate, persistence.
    m01_seq already emits all of those, for six detector channels, with
    historical-path null calibration -- its own docstring is an essay on exactly
    that discrimination.  Rebuilding it would have measured nothing.

    So this module was designed AFTER the W5-D2 false-positive forensics
    (research/reports/wave5_diagnostics.json), against the mechanism that
    actually fools the champion.  Among the top 1% highest-ranked NO-BREAK
    series, the descriptors that separate them from the rest are
    `tail_rate_online` (+1.89 IQR), `shock_max_absz` (+1.50) and `burst_ratio`
    (+1.38); the heuristic mechanism mix puts heavy-tail/outlier at 40% against
    a 26% base rate.  The champion's false positives are dominated by online
    segments carrying MORE EXTREME VALUES than their own history predicts.

SIGNAL / ALPHA
    A genuine scale break moves the whole distribution: the trimmed variance
    moves, the median absolute deviation moves, exceedances become frequent and
    contiguous.  An outlier or a short burst moves the untrimmed variance and
    leaves the trimmed one alone, concentrates the window's energy in one or two
    points, and produces scattered rather than contiguous exceedances.  Nothing
    in the incumbent bank computes a trimmed statistic, an energy-concentration
    statistic, or an exceedance run length, so the two cases arrive at the
    booster looking alike.

    The load-bearing column is the CONTRAST, not either half: `*_gap` is the
    calibrated untrimmed scale minus the calibrated trimmed scale.  Boosters
    approximate differences of two features badly and read one column well, and
    both halves are already implicitly present -- what is missing is their
    difference.

FALSE SIGNAL
    A real break that happens to be small will also have low concentration and a
    low trimmed/untrimmed gap, so a low gap is not evidence of stability; it is
    only evidence that whatever moved, moved in the bulk.  A break INTO a
    heavy-tailed regime is genuinely outlier-driven and this module will call it
    transient.  That is a real cost and the age-bucket diagnostic is where it
    would show up.

DISAMBIGUATOR
    Every statistic is calibrated against the same statistic over length-matched
    windows of the break-free history, so a series that is heavy-tailed ALREADY
    has a heavy-tailed null and does not score.  This is the whole reason the
    per-series historical null exists.

CAUSALITY
    Row t uses online[:t+1] and history only; every window statistic is a
    trailing window ending at t and every null constant comes from history.
    Bitwise prefix-invariant, checked at atol=0.
"""
from __future__ import annotations

import numpy as np

from sbr.features.base import register
# the null machinery is shared with m12_rdep rather than duplicated: same
# construction, same tests, one place for it to be wrong.
from sbr.features.m12_rdep import _GridNull, _WinNull, _cum, _roll, CLIP, EPS

WINDOWS = (32, 128)
TRIM = 2                      #: points dropped from each tail-end of a window
Q_EXC = 0.95                  #: exceedance threshold quantile, from history


def _win(a, w):
    """Trailing windows of length w ending at each index; NaN before w points."""
    a = np.asarray(a, dtype=np.float64)
    n = len(a)
    if w > n:
        return None
    return np.lib.stride_tricks.sliding_window_view(a, w)      # (n-w+1, w)


def _conc_trim(a, w):
    """(concentration, trimmed mean, untrimmed mean) of ``a`` over trailing w."""
    n = len(a)
    W = _win(a, w)
    conc = np.full(n, np.nan)
    trim = np.full(n, np.nan)
    full = np.full(n, np.nan)
    if W is None:
        return conc, trim, full
    tot = W.sum(axis=1)
    mx = W.max(axis=1)
    conc[w - 1:] = mx / np.maximum(tot, EPS)
    k = min(TRIM, max(w // 8, 1))
    part = np.partition(W, w - k - 1, axis=1)
    trim[w - 1:] = part[:, :w - k].mean(axis=1)
    full[w - 1:] = tot / w
    return conc, trim, full


def _exp_conc_trim(a):
    """Expanding version: running max, running second max, running mean."""
    a = np.asarray(a, dtype=np.float64)
    n = len(a)
    L = np.arange(1, n + 1, dtype=np.float64)
    c = _cum(a)
    full = c[1:] / L
    mx = np.maximum.accumulate(a)
    # running second maximum, so the trimmed mean drops the single largest point
    m2 = np.full(n, -np.inf)
    best = -np.inf
    sec = -np.inf
    for i in range(n):
        if a[i] > best:
            sec, best = best, a[i]
        elif a[i] > sec:
            sec = a[i]
        m2[i] = sec
    conc = mx / np.maximum(c[1:], EPS)
    trim = np.where(L > 1, (c[1:] - mx) / np.maximum(L - 1, 1.0), full)
    return conc, trim, full, m2


def _max_run(flag):
    """Longest run of consecutive True in flag[:t+1], at every t."""
    n = len(flag)
    out = np.empty(n)
    cur = 0
    best = 0
    for i in range(n):
        cur = cur + 1 if flag[i] else 0
        if cur > best:
            best = cur
        out[i] = best
    return out


try:
    from numba import njit
    _max_run = njit(cache=True, fastmath=False)(_max_run)
except Exception:                                            # pragma: no cover
    pass


@register("m10_persist", version="1", owner="claude-wave5")
def build(ctx):
    n = ctx.n
    L = np.arange(1, n + 1, dtype=np.float64)
    hp = ctx.hp
    cols: list[str] = []
    out: list[np.ndarray] = []

    def add(name, v):
        v = np.asarray(v, dtype=np.float64)
        assert v.shape == (n,), f"{name}: {v.shape} != ({n},)"
        cols.append(name)
        out.append(v)

    z = (ctx.online - hp.mu) / hp.sd
    zh = (ctx.hist - hp.mu) / hp.sd
    e = (ctx.ar_online / hp.ar_sigma) if ctx.ar_online is not None else z
    eh = np.asarray(ctx.hist_tr.get("res_mean", zh), dtype=np.float64)

    for sname, xo, xh in (("z", z, zh), ("e", e, eh)):
        so, sh = xo * xo, xh * xh

        # ---- trailing windows: concentration, trimmed vs untrimmed scale ----
        for w in WINDOWS:
            conc, trim, full = _conc_trim(so, w)
            nc_ = _WinNull(sh, w, lambda a, ww: _conc_trim(a, ww)[0])
            nt = _WinNull(sh, w, lambda a, ww: _conc_trim(a, ww)[1])
            nf = _WinNull(sh, w, lambda a, ww: _conc_trim(a, ww)[2])
            zt, zf = nt.z(trim), nf.z(full)
            add(f"{sname}_w{w}_conc", nc_.z(conc))
            add(f"{sname}_w{w}_trim", zt)
            add(f"{sname}_w{w}_full", zf)
            # THE contrast: untrimmed minus trimmed evidence.  Large and positive
            # means the apparent scale change lives in the few largest points.
            add(f"{sname}_w{w}_gap", zf - zt)

        # ---- expanding versions -------------------------------------------
        conc, trim, full, _ = _exp_conc_trim(so)
        gn_c = _GridNull(sh, lambda seg: _exp_conc_trim(seg)[0])
        gn_t = _GridNull(sh, lambda seg: _exp_conc_trim(seg)[1])
        gn_f = _GridNull(sh, lambda seg: _exp_conc_trim(seg)[2])
        zt, zf = gn_t.z(L, trim), gn_f.z(L, full)
        add(f"{sname}_exp_conc", gn_c.z(L, conc))
        add(f"{sname}_exp_trim", zt)
        add(f"{sname}_exp_full", zf)
        add(f"{sname}_exp_gap", zf - zt)

        # ---- exceedances: how many, how big, and how contiguous ------------
        thr = float(np.quantile(np.abs(xh), Q_EXC))
        exc = (np.abs(xo) > thr).astype(np.float64)
        exc_h = (np.abs(xh) > thr).astype(np.float64)
        mag = np.where(exc > 0, np.abs(xo) - thr, 0.0)
        mag_h = np.where(exc_h > 0, np.abs(xh) - thr, 0.0)
        c_e, c_m = _cum(exc), _cum(mag)
        gn_e = _GridNull(exc_h, lambda seg: _cum(seg)[1:] / np.arange(1, len(seg) + 1))
        gn_m = _GridNull(mag_h, lambda seg: _cum(seg)[1:] / np.arange(1, len(seg) + 1))
        z_n = gn_e.z(L, c_e[1:] / L)
        z_g = gn_m.z(L, c_m[1:] / L)
        add(f"{sname}_exc_n", z_n)
        add(f"{sname}_exc_mag", z_g)
        # many moderate exceedances (a regime change) vs few huge ones (outliers)
        add(f"{sname}_exc_shape", z_g - z_n)
        # a burst is CONTIGUOUS; scattered outliers are not
        gn_r = _GridNull(exc_h, lambda seg: _max_run((seg > 0)))
        add(f"{sname}_exc_run", gn_r.z(L, _max_run(exc > 0)))

        # ---- persistence of the ROBUST scale channel -----------------------
        # Built on the trimmed evidence, so a single outlier cannot create the
        # "still elevated" reading the incumbent persistence channels can.
        conc32, trim32, _ = _conc_trim(so, WINDOWS[0])
        nt32 = _WinNull(sh, WINDOWS[0], lambda a, ww: _conc_trim(a, ww)[1])
        r = nt32.z(trim32)
        rr = np.where(np.isfinite(r), r, 0.0)
        pk = np.maximum.accumulate(rr)
        add(f"{sname}_rb_cur", r)
        add(f"{sname}_rb_pk", pk)
        # recovery: how far the robust scale has fallen back from its own peak
        add(f"{sname}_rb_rel", rr - pk)
        arg = np.maximum.accumulate(np.where(rr >= pk - 1e-12, L, 0.0))
        add(f"{sname}_rb_tsp", np.log1p(L - arg))
        # what fraction of the elapsed segment has the robust scale stayed hot
        add(f"{sname}_rb_per", np.cumsum(rr > 2.0) / L)

    A = np.column_stack(out).astype(np.float32)
    A[~np.isfinite(A)] = np.nan
    return cols, A
