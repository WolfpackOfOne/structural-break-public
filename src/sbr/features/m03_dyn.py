"""m03_dyn -- ONLINE temporal-dynamics evidence, calibrated against the
per-series historical null.

The marginal-distribution channel is already covered (m00_core).  This module
watches the *dynamics*: dependence structure, volatility clustering, trend,
spectral shape, multi-scale (wavelet) energy allocation and ordinal
complexity.  A structural break can leave the marginal law almost untouched
(same mean, same variance, same tails) while completely rewiring how one point
relates to the next -- an AR coefficient flip, a GARCH regime change, a switch
from a random walk to a mean-reverting process, a change of characteristic
frequency.  Those breaks are invisible to moment/occupancy statistics and are
exactly what this module targets.

DESIGN CONSTRAINTS THAT SHAPED EVERY CHOICE
    * causal by construction: every statistic is a function of trailing-window
      means of per-point transforms, so row t only ever touches hist and
      online[:t+1].  No stride, no hold, no FFT: the windowed DFT is obtained
      from cumulative sums of z*cos(2*pi*f*j) and z*sin(2*pi*f*j), which is an
      exact Goertzel-equivalent at fixed frequencies for O(1) per point.
    * ADAPTIVE windows instead of fixed ones.  Online length ranges 10..999, so
      a fixed w=128 is useless for a third of the data and sluggish for the
      rest.  Every statistic is evaluated on a window that is a fixed FRACTION
      of the online prefix (quarter / half / full).  This keeps the feature
      defined early, keeps it responsive late, and -- crucially -- the null is
      then a *function of window length*, which we interpolate.
    * every statistic is emitted (where budget allows) both raw and as a
      robust z against the distribution of the SAME statistic over
      length-matched windows of the break-free history.  The calibrated
      version is the H2-flavoured construction: it asks "is this dynamic
      unusual FOR THIS SERIES", which is what a cross-sectional metric like
      TS-AUC actually needs.
    * lag products, differences, Haar details and ordinal patterns at the very
      start of the online segment are warmed up from the TAIL OF THE HISTORY
      (known at t=0, therefore causal) rather than zero-padded, so no artificial
      zeros contaminate the early windows.
"""
from __future__ import annotations

import numpy as np

from sbr.features.base import register

# ---------------------------------------------------------------- constants
PAD = 40                      # hist points prepended to online / trimmed from hist
ZLAGS = (1, 2, 5, 10, 20)     # ACF lags on the standardised raw series
ELAGS = (1, 2, 5, 10)         # ACF lags on the AR residual
ALAGS = (1, 5, 20)            # ACF lags on |residual| (volatility clustering)
FREQS = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625])
NF = len(FREQS)
HSCALES = (1, 2, 4, 8)        # Haar dyadic scales
GRID = (16, 64, 256)          # null-calibration window grid (log-spaced)
NULL_N = 700                  # historical null windows sampled per grid point
LG = np.log(np.asarray(GRID, dtype=np.float64))
PCODES = (0, 1, 3, 4, 6, 7)   # the 6 realisable order-3 ordinal patterns
CLIP = 10.0
EPS = 1e-9


# ---------------------------------------------------------------- helpers
def _shiftmul(a: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(a)
    out[k:] = a[k:] * a[:-k]
    return out


def _cumz(a: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(a)))


def _local(z: np.ndarray, e: np.ndarray, off: int) -> dict:
    """Per-point transforms whose rolling means give every statistic here.

    ``z``/``e`` include ``off`` leading warm-up points which are trimmed from
    the returned arrays; every transform therefore has a fully-defined value at
    its first returned index.
    """
    n = len(z)
    # index origin at the FIRST RETURNED point so that the trend statistic
    # (which mixes sum(x) and sum(j*x)) is consistent with the end indices the
    # callers use.  Spectral phase is irrelevant: |windowed DFT| is
    # origin-invariant.
    j = np.arange(n, dtype=np.float64) - float(off)
    ae = np.abs(e)
    e2 = e * e
    d = {}

    # --- dependence ------------------------------------------------------
    d["x"] = z
    d["x2"] = z * z
    for k in ZLAGS:
        d[f"xl{k}"] = _shiftmul(z, k)
    d["e"] = e
    d["e2"] = e2
    d["e4"] = e2 * e2
    for k in ELAGS:
        d[f"el{k}"] = _shiftmul(e, k)
    d["ae"] = ae
    for k in ALAGS:
        d[f"ael{k}"] = _shiftmul(ae, k)
    d["e2l1"] = _shiftmul(e2, 1)
    sg = np.sign(z)
    d["sg"] = sg
    d["sg2"] = sg * sg
    d["sgl1"] = _shiftmul(sg, 1)

    # --- trend -----------------------------------------------------------
    d["tx"] = j * z
    dz = np.zeros_like(z)
    dz[1:] = z[1:] - z[:-1]
    ddz = np.zeros_like(z)
    ddz[2:] = z[2:] - 2.0 * z[1:-1] + z[:-2]
    d["d1"] = dz
    d["d1_2"] = dz * dz
    d["d2"] = ddz
    d["d2_2"] = ddz * ddz
    d["mk1"] = np.sign(dz)
    mk8 = np.zeros_like(z)
    mk8[8:] = np.sign(z[8:] - z[:-8])
    d["mk8"] = mk8

    # --- spectral: windowed-DFT quadratures at fixed frequencies ---------
    ang = 2.0 * np.pi * np.outer(FREQS, j)
    cs = np.cos(ang) * z
    sn = np.sin(ang) * z
    for i in range(NF):
        d[f"c{i}"] = cs[i]
        d[f"s{i}"] = sn[i]

    # --- wavelet: Haar detail energies at dyadic scales ------------------
    cz = _cumz(z)
    for s in HSCALES:
        dd = np.zeros_like(z)
        idx = np.arange(2 * s - 1, n)
        dd[idx] = (cz[idx + 1] - 2.0 * cz[idx + 1 - s] + cz[idx + 1 - 2 * s]) / np.sqrt(2.0 * s)
        d[f"h{s}"] = dd * dd

    # --- complexity: order-3 ordinal patterns ----------------------------
    a = np.empty_like(z); a[:2] = 0.0; a[2:] = z[:-2]
    b = np.empty_like(z); b[:1] = 0.0; b[1:] = z[:-1]
    code = 4 * (a > b).astype(np.int64) + 2 * (a > z).astype(np.int64) + (b > z).astype(np.int64)
    for i, pc in enumerate(PCODES):
        d[f"p{i}"] = (code == pc).astype(np.float64)

    return {k: np.ascontiguousarray(v[off:]) for k, v in d.items()}


class _R:
    """Rolling-mean accessor: R(name, mult) = mean over a window of mult*W."""

    __slots__ = ("cum", "W", "tend", "cache", "vec", "sl")

    def __init__(self, cum, W, tend, vec, sl=None):
        self.cum = cum
        self.W = W
        self.tend = tend
        self.vec = vec
        self.sl = sl          # (start, stop, step) of end indices, hist only
        self.cache = {}

    def __call__(self, name, mult=1.0):
        key = (name, mult)
        v = self.cache.get(key)
        if v is None:
            c = self.cum[name]
            if self.vec:
                w = np.maximum(np.rint(mult * self.W).astype(np.int64), 1)
                w = np.minimum(w, self.tend + 1)
                hi = self.tend + 1
                v = (c[hi] - c[hi - w]) / w
            else:
                w = max(int(round(mult * self.W)), 1)
                a, b, st = self.sl                    # basic slicing: no gather
                v = (c[a + 1:b + 1:st] - c[a + 1 - w:b + 1 - w:st]) / w
            self.cache[key] = v
        return v


def _loglin(y, x):
    """Piecewise-linear in log-window with linear extrapolation at both ends."""
    out = np.interp(x, LG, y)
    lo = x < LG[0]
    if lo.any():
        out[lo] = y[0] + (y[1] - y[0]) / (LG[1] - LG[0]) * (x[lo] - LG[0])
    hi = x > LG[-1]
    if hi.any():
        out[hi] = y[-1] + (y[-1] - y[-2]) / (LG[-1] - LG[-2]) * (x[hi] - LG[-1])
    return out


class _Cal:
    """Historical-null calibrator for arbitrary window-scaled statistics."""

    def __init__(self, cum_h, nh):
        self.R = {}
        for W in GRID:
            m = nh - 2 * W + 1
            if m <= 0:
                tend = np.array([nh - 1], dtype=np.int64)
                sl = (nh - 1, nh, 1)
            else:
                # subsample the null windows: a median/IQR needs ~NULL_N draws,
                # not every one of the ~3000 historical windows.  The stride is
                # a fixed function of the historical length only, so it is
                # deterministic and cannot affect prefix invariance.
                step = max(1, m // NULL_N)
                tend = np.arange(2 * W - 1, nh, step, dtype=np.int64)
                sl = (2 * W - 1, nh, step)
            self.R[W] = _R(cum_h, float(W), tend, False, sl)
        self.cache = {}

    def params(self, key, fn):
        p = self.cache.get(key)
        if p is None:
            mus = np.empty(len(GRID))
            sds = np.empty(len(GRID))
            for i, W in enumerate(GRID):
                R = self.R[W]
                v = np.asarray(fn(R, float(W), R.tend), dtype=np.float64)
                v = v[np.isfinite(v)]
                if v.size < 24:
                    mus[i] = 0.0
                    sds[i] = 1.0
                else:
                    v = np.sort(v)          # cheaper than 5x np.quantile
                    k = v.size - 1
                    p05 = v[int(0.05 * k)]; q1 = v[int(0.25 * k)]
                    m = v[int(0.5 * k)]
                    q3 = v[int(0.75 * k)]; p95 = v[int(0.95 * k)]
                    mus[i] = m
                    sds[i] = max((q3 - q1) / 1.349, (p95 - p05) / 3.29, 1e-9)
            p = (mus, np.log(sds))
            self.cache[key] = p
        return p

    def z(self, key, fn, value, W):
        mus, lsds = self.params(key, fn)
        lw = np.log(np.maximum(W, 1.0))
        mu = _loglin(mus, lw)
        sd = np.exp(_loglin(lsds, lw))
        return np.clip((value - mu) / np.maximum(sd, 1e-9), -CLIP, CLIP)


# ---------------------------------------------------------------- statistics
def _acf(num, den_sq, mean_name, lag_key):
    """Windowed lag-k autocorrelation of a process, demeaned inside the window.

    Demeaning inside the window is what makes this a DEPENDENCE detector rather
    than a level detector: a pure mean shift leaves it unchanged.
    """
    def f(R, W, tend):
        m = R(mean_name)
        v = R(den_sq) - m * m
        c = R(lag_key) - m * m
        return np.clip(c / np.maximum(v, 1e-8), -2.0, 2.0)
    return f


def _ratio(a, ma, b, mb):
    def f(R, W, tend):
        return np.log(np.maximum(R(a, ma), 1e-10) / np.maximum(R(b, mb), 1e-10))
    return f


def _drift(R, W, tend):
    """Total drift (in historical sd units) across the trailing window.

    OLS slope from cumulative sums of x and t*x -- never a regression per step.
    """
    w = np.rint(W).astype(np.int64) if np.ndim(W) else int(round(W))
    w = np.maximum(w, 2)
    s0 = R("x") * w
    s1 = R("tx") * w
    jbar = tend - (w - 1) / 2.0
    den = w * (w * w - 1.0) / 12.0
    return (s1 - jbar * s0) / np.maximum(den, 1e-9) * w


def _mk(name):
    def f(R, W, tend):
        return R(name)
    return f


def _spec_p(R):
    """Relative band powers over the fixed dyadic frequency set."""
    P = []
    for i in range(NF):
        a = R(f"c{i}")
        b = R(f"s{i}")
        P.append(a * a + b * b)
    P = np.stack(P)
    tot = P.sum(axis=0)
    return P / np.maximum(tot, 1e-14)


def _sp_low(R, W, tend):
    p = _spec_p(R)
    return p[4] + p[5]


def _sp_high(R, W, tend):
    p = _spec_p(R)
    return p[0] + p[1]


def _sp_ent(R, W, tend):
    p = _spec_p(R)
    return -(p * np.log(np.maximum(p, 1e-12))).sum(axis=0) / np.log(NF)


def _sp_cen(R, W, tend):
    p = _spec_p(R)
    return (p * np.arange(NF).reshape(-1, *([1] * (p.ndim - 1)))).sum(axis=0)


def _hw_logE(R):
    return [np.log(np.maximum(R(f"h{s}"), 1e-12)) for s in HSCALES]


def _hw_r(i, j):
    def f(R, W, tend):
        L = _hw_logE(R)
        return L[i] - L[j]
    return f


def _hw_hurst(R, W, tend):
    L = _hw_logE(R)
    return (-1.5 * L[0] - 0.5 * L[1] + 0.5 * L[2] + 1.5 * L[3]) / 5.0 / np.log(2.0)


def _pe(R, W, tend):
    q = np.stack([R(f"p{i}") for i in range(6)])
    q = q / np.maximum(q.sum(axis=0), 1e-12)
    return -(q * np.log(np.maximum(q, 1e-12))).sum(axis=0) / np.log(6.0)


def _hj_mob(R, W, tend):
    vx = np.maximum(R("x2") - R("x") ** 2, 1e-10)
    vd = np.maximum(R("d1_2") - R("d1") ** 2, 1e-10)
    return 0.5 * np.log(vd / vx)


def _hj_comp(R, W, tend):
    vx = np.maximum(R("x2") - R("x") ** 2, 1e-10)
    vd = np.maximum(R("d1_2") - R("d1") ** 2, 1e-10)
    vdd = np.maximum(R("d2_2") - R("d2") ** 2, 1e-10)
    return 0.5 * (np.log(vdd / vd) - np.log(vd / vx))


# ---------------------------------------------------------------- module
@register("m03_dyn", version="1", owner="agent7")
def build(ctx):
    n = ctx.n
    hp = ctx.hp

    zh = np.asarray(ctx.hist_tr["mean"], dtype=np.float64)
    eh = np.asarray(ctx.hist_tr.get("res_mean", ctx.hist_tr["mean"]), dtype=np.float64)
    zo = np.asarray(ctx.tr["mean"], dtype=np.float64)
    eo = np.asarray(ctx.tr.get("res_mean", ctx.tr["mean"]), dtype=np.float64)

    pad = min(PAD, len(zh))
    zp = np.concatenate([zh[len(zh) - pad:], zo])
    ep = np.concatenate([eh[len(eh) - pad:], eo])

    trh = _local(zh, eh, PAD if len(zh) > PAD + 64 else 0)
    tro = _local(zp, ep, pad)

    nh = len(next(iter(trh.values())))
    cum_h = {k: _cumz(v) for k, v in trh.items()}
    cum_o = {k: _cumz(v) for k, v in tro.items()}
    cal = _Cal(cum_h, nh)

    tend = np.arange(n, dtype=np.int64)
    L = tend + 1.0
    W_exp = tend + 1
    W_half = np.maximum(W_exp // 2, 1)
    W_quart = np.maximum(W_exp // 4, 1)
    R_exp = _R(cum_o, W_exp, tend, True)
    R_half = _R(cum_o, W_half, tend, True)
    R_quart = _R(cum_o, W_quart, tend, True)
    SCHEME = {"exp": (R_exp, W_exp.astype(np.float64)),
              "half": (R_half, W_half.astype(np.float64)),
              "quart": (R_quart, W_quart.astype(np.float64))}

    cols: list[str] = []
    out: list[np.ndarray] = []

    def emit(name, arr):
        cols.append(name)
        out.append(np.asarray(arr, dtype=np.float64))

    def raw(fn, scheme, minW):
        R, W = SCHEME[scheme]
        v = np.asarray(fn(R, W, tend), dtype=np.float64)
        v = np.where(W >= minW, v, np.nan)
        return v, W

    def both(key, fn, scheme, minW, name, want_raw=True, want_z=True):
        v, W = raw(fn, scheme, minW)
        if want_raw:
            emit(name, v)
        if want_z:
            z = cal.z(key, fn, v, W)
            emit(name + "_z", np.where(np.isfinite(v), z, np.nan))

    # ============ A. DEPENDENCE (incremental ACF, calibrated) ============
    for k in ZLAGS:
        f = _acf("x", "x2", "x", f"xl{k}")
        both(f"acf_x_l{k}", f, "exp", max(12, 2 * k), f"az_x_exp_l{k}", want_raw=False)
    for k in ELAGS:
        f = _acf("e", "e2", "e", f"el{k}")
        both(f"acf_e_l{k}", f, "exp", max(12, 2 * k), f"az_e_exp_l{k}", want_raw=False)
    for k in ALAGS:
        f = _acf("ae", "e2", "ae", f"ael{k}")
        both(f"acf_ae_l{k}", f, "exp", max(12, 2 * k), f"az_ae_exp_l{k}", want_raw=False)
    both("acf_e2_l1", _acf("e2", "e4", "e2", "e2l1"), "exp", 12, "az_e2_exp_l1", want_raw=False)
    both("acf_sg_l1", _acf("sg", "sg2", "sg", "sgl1"), "exp", 12, "az_sg_exp_l1", want_raw=False)
    # trailing-half versions of the three headline dependence channels: a
    # break late in a long series barely moves the expanding ACF.
    both("acf_x_l1", _acf("x", "x2", "x", "xl1"), "half", 12, "az_x_half_l1", want_raw=False)
    both("acf_e_l1", _acf("e", "e2", "e", "el1"), "half", 12, "az_e_half_l1", want_raw=False)
    both("acf_ae_l1", _acf("ae", "e2", "ae", "ael1"), "half", 12, "az_ae_half_l1", want_raw=False)

    # ============ B. VOLATILITY CLUSTERING / VARIANCE RATIO ==============
    both("vr_qh", _ratio("e2", 0.5, "e2", 1.0), "half", 16, "vr_qh")
    both("vr_he", _ratio("e2", 1.0, "e2", 2.0), "half", 16, "vr_he")
    both("vra_qh", _ratio("ae", 0.5, "ae", 1.0), "half", 16, "vra_qh", want_raw=False)

    # ============ C. TREND ===============================================
    dz_by = {}
    for sch, mw, nm in (("quart", 12, "drift_q"), ("half", 12, "drift_h"), ("exp", 12, "drift_e")):
        v, W = raw(_drift, sch, mw)
        emit(nm, v)
        z = np.where(np.isfinite(v), cal.z("drift", _drift, v, W), np.nan)
        emit(nm + "_z", z)
        dz_by[sch] = z
    emit("dslope_qh", dz_by["quart"] - dz_by["half"])
    emit("dslope_he", dz_by["half"] - dz_by["exp"])
    both("mk1", _mk("mk1"), "half", 12, "mk1")
    both("mk8", _mk("mk8"), "half", 16, "mk8")

    # ============ D. SPECTRAL ============================================
    both("sp_low", _sp_low, "half", 32, "sp_low")
    both("sp_high", _sp_high, "half", 32, "sp_high")
    both("sp_ent", _sp_ent, "half", 32, "sp_ent")
    both("sp_cen", _sp_cen, "half", 32, "sp_cen")
    plo, _ = raw(_sp_low, "half", 32)
    phi, _ = raw(_sp_high, "half", 32)
    emit("sp_lr", np.log((plo + 1e-6) / (phi + 1e-6)))

    def _sp_dom(R, W, tend):
        return np.argmax(_spec_p(R), axis=0).astype(np.float64)
    v, _ = raw(_sp_dom, "half", 32)
    emit("sp_dom", v)

    # ============ E. WAVELET (Haar) ======================================
    both("hw_r12", _hw_r(0, 1), "half", 24, "hw_r12")
    both("hw_r24", _hw_r(1, 2), "half", 24, "hw_r24")
    both("hw_r48", _hw_r(2, 3), "half", 32, "hw_r48")
    both("hw_hurst", _hw_hurst, "half", 32, "hw_hurst")

    # ============ F. COMPLEXITY ==========================================
    both("pe", _pe, "half", 24, "pe_half")
    both("pe", _pe, "exp", 24, "pe_exp")
    both("hj_mob", _hj_mob, "half", 16, "hj_mob")
    both("hj_comp", _hj_comp, "half", 16, "hj_comp")

    A = np.column_stack(out).astype(np.float32)
    A[~np.isfinite(A)] = np.nan
    return cols, A
