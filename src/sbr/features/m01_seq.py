"""m01_seq -- multi-scale sequential change-detection evidence bank.

WHAT THIS IS
    A bank of classical online change detectors (CUSUM, CUSUM-SQ, Page-Hinkley,
    Shiryaev-Roberts, dyadic GLR / mixture-GLR, multi-half-life EWMA) run as
    *features*, not as a collapsed alarm.  Each detector is exposed through the
    SHAPE of its evidence path -- current level, running peak, decayed peak,
    time since peak, persistence above a historical threshold, local slope, and
    the current/peak ratio -- because that shape is exactly what separates a
    permanent regime change from a transient anomaly.  A transient spike drives
    a detector up and then lets it decay: current/peak collapses, time-since-peak
    grows, persistence stays low.  A real break keeps the detector pinned at or
    near its peak, so current/peak stays ~1, the slope stays positive and the
    persistence fraction climbs.  We never force monotonicity and never OR the
    channels together; LightGBM gets the raw geometry.

CALIBRATION (the load-bearing part)
    A raw CUSUM value is meaningless across series: its scale depends on the
    per-series noise, tail weight and dependence.  TS-AUC only ever compares
    series against each other at a fixed online index, so cross-series
    comparability *is* the metric.  These statistics are path dependent, so
    ``ctx.nc`` (which calibrates rolling means of transforms) cannot be used
    directly.  Instead, for every channel we run the *identical recursion* over
    the break-free historical segment, using the identical historical
    standardisation, and calibrate against that path:

      * marginal null  -- the sorted empirical distribution of the historical
        detector path (after a burn-in) gives an exact upper-tail empirical
        p-value for the current detector value.
      * running-peak null -- trailing rolling maxima of the historical detector
        path over dyadic windows (32/128/512, computed by doubling in O(H log H))
        give the null for "largest value this detector reached in the last L
        steps"; the online running peak at step t is scored against the grid
        window nearest t in log space.
      * threshold/scale constants -- q50/q99/max of the historical path give the
        persistence threshold and the slope normaliser.

    Everything is O(n_hist) up to the log factor and a handful of sorts.

APPROXIMATIONS (documented, deliberate)
    * GLR uses a DYADIC candidate change-point grid (window lengths
      1,2,4,...,512) rather than a full O(t) scan every step.  This costs
      O(log n) per point instead of O(n), and the worst-case loss is a factor
      <= sqrt(2) in the effective segment length, i.e. the GLR statistic is
      recovered to within ~sqrt(2)/2 in amplitude for change points that fall
      between grid nodes.  The mixture-GLR is the log-mean-exp over the same
      dyadic set.
    * The running-peak null is taken from a historical path that is *not*
      reset to zero at the window start, so it is very slightly conservative
      relative to the online peak, which does start from zero.  The bias is a
      smooth function of window length and affects all series the same way.
    * The marginal null is the stationary (converged) distribution of the
      detector, whereas the online detector at step t has only run for t steps.
      This makes early online rows look "quiet"; since TS-AUC compares at fixed
      t, a t-monotone distortion shared by all series is harmless.

CAUSALITY
    Every online quantity is a cumulative/sequential function of
    ``online[:t+1]`` plus historical constants; every null is built from history
    only.  All time-axis reductions are strictly sequential numpy accumulators
    (cumsum / minimum.accumulate / maximum.accumulate) or an IIR filter, all of
    which are prefix-stable bitwise.  All cross-channel reductions are row-wise.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

from sbr.features.base import register

# ---------------------------------------------------------------- constants
DYADIC = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)     # GLR candidate lengths
PEAK_WIN_LOG2 = (5, 7, 9)                             # 32 / 128 / 512
SLOPE_K = 16
DECAY_HL = 64.0
EWMA_HL = (8.0, 32.0, 128.0)
SUR_CAP = 6.0
ZCAP = 12.0
EPS = 1e-12

_LOG_PEAK_WIN = np.log(np.array([2.0 ** m for m in PEAK_WIN_LOG2]))


# ---------------------------------------------------------------- kernels
try:
    from numba import njit

    @njit(cache=True, fastmath=False)
    def _sr_log_kernel(l):                                   # noqa: E741
        n = l.shape[0]
        out = np.empty(n, dtype=np.float64)
        r = -1.0e300
        for i in range(n):
            if r > 30.0:
                s = r + math.log1p(math.exp(-r))
            elif r < -30.0:
                s = math.exp(r)
            else:
                s = math.log1p(math.exp(r))
            r = l[i] + s
            out[i] = r
        return out

    _HAVE_NUMBA = True
except Exception:                                            # pragma: no cover
    _HAVE_NUMBA = False

    def _sr_log_kernel(l):                                   # noqa: E741
        out = np.empty(len(l), dtype=np.float64)
        r = -1.0e300
        for i in range(len(l)):
            if r > 30.0:
                s = r + math.log1p(math.exp(-r))
            elif r < -30.0:
                s = math.exp(r)
            else:
                s = math.log1p(math.exp(r))
            r = l[i] + s
            out[i] = r
        return out


def _sr_log(x: np.ndarray, delta: float) -> np.ndarray:
    """log Shiryaev-Roberts statistic for a mean shift of size ``delta``.

    R_t = (1 + R_{t-1}) * exp(delta*x_t - delta^2/2), kept in log space so it
    never overflows (R grows exponentially once a break starts).
    """
    l = np.ascontiguousarray(delta * x - 0.5 * delta * delta)
    return _sr_log_kernel(l)


def _reflect(y: np.ndarray) -> np.ndarray:
    """S_t = max(0, S_{t-1} + y_t), S_0 = 0  --  the CUSUM/Page recursion.

    Closed form S_t = C_t - min_{0<=j<=t} C_j with C the cumsum, so it is one
    cumsum plus one running minimum: fully vectorised and prefix-stable.
    """
    c = np.empty(y.shape[0] + 1, dtype=np.float64)
    c[0] = 0.0
    np.cumsum(y, out=c[1:])
    return c[1:] - np.minimum.accumulate(c)[1:]


def _cusum_pair(x: np.ndarray, k: float):
    """(positive, negative) CUSUM of standardised stream ``x`` with slack ``k``."""
    return _reflect(x - k), _reflect(-x - k)


def _page_hinkley(x: np.ndarray, delta: float):
    """Page-Hinkley up/down using the *running* online mean as the reference."""
    t = np.arange(1, x.shape[0] + 1, dtype=np.float64)
    rm = np.cumsum(x) / t
    d = x - rm
    return _reflect(d - delta), _reflect(-d - delta)


def _glr_dyadic(x: np.ndarray):
    """Dyadic-grid GLR and mixture-GLR for a mean shift in a unit-variance stream.

    For a candidate change point t-L the GLR statistic is
    (sum of the last L points)^2 / (2 L).  We evaluate it only on dyadic L.
    """
    n = x.shape[0]
    c = np.empty(n + 1, dtype=np.float64)
    c[0] = 0.0
    np.cumsum(x, out=c[1:])
    best = np.zeros(n, dtype=np.float64)
    acc = np.zeros(n, dtype=np.float64)
    cnt = np.zeros(n, dtype=np.float64)
    for L in DYADIC:
        if L > n:
            break
        s = np.zeros(n, dtype=np.float64)
        s[L - 1:] = (c[L:] - c[:-L]) ** 2 / (2.0 * L)
        best = np.maximum(best, s)
        # only rows that actually have L points contribute -- keeping the
        # mixture row-local is what makes it prefix-invariant
        acc[L - 1:] += np.exp(np.minimum(s[L - 1:], 40.0))
        cnt[L - 1:] += 1.0
    mix = np.log(acc / np.maximum(cnt, 1.0) + EPS)
    return best, mix


def _ewma(x: np.ndarray, hl: float) -> np.ndarray:
    """Bias-corrected EWMA (IIR, sequential -> prefix-stable bitwise)."""
    a = 1.0 - 2.0 ** (-1.0 / hl)
    y = lfilter(np.array([a]), np.array([1.0, -(1.0 - a)]), x)
    den = 1.0 - (1.0 - a) ** np.arange(1, x.shape[0] + 1, dtype=np.float64)
    return y / den


def _trailing_max_pow2(a: np.ndarray, logs) -> dict:
    """Trailing rolling maxima for window sizes 2**m, computed by doubling."""
    out = {}
    cur = a.astype(np.float64, copy=True)
    step = 1
    mx = max(logs)
    for m in range(1, mx + 1):
        prev = np.concatenate([np.full(step, -np.inf), cur[:-step]]) if step < len(cur) \
            else np.full(len(cur), -np.inf)
        cur = np.maximum(cur, prev)
        step *= 2
        if m in logs:
            out[m] = cur
    return out


def _decayed_peak(d: np.ndarray, hl: float) -> np.ndarray:
    """P_t = max(D_t, rho P_{t-1}) with rho = 2^{-1/hl}, done in log space."""
    lr = -math.log(2.0) / hl
    idx = np.arange(d.shape[0], dtype=np.float64)
    a = np.log(np.maximum(d, 0.0) + EPS) - idx * lr
    return np.exp(idx * lr + np.maximum.accumulate(a))


# ---------------------------------------------------------------- calibration
class _Chan:
    """One detector channel: its historical null and the online path scorer."""

    __slots__ = ("srt", "n", "q50", "q99", "qmax", "scale", "pk_srt")

    def __init__(self, dh: np.ndarray, peak_null: bool):
        h = dh.shape[0]
        burn = min(64, h // 4)
        v = np.sort(np.asarray(dh[burn:], dtype=np.float64))
        if v.shape[0] < 8:
            v = np.sort(np.asarray(dh, dtype=np.float64))
        self.srt = v
        self.n = v.shape[0]
        self.q50 = float(v[self.n // 2])
        self.q99 = float(v[min(self.n - 1, int(0.99 * self.n))])
        self.qmax = float(v[-1])
        # robust spread of the historical detector path.  The floor matters:
        # a near-degenerate null (q99 == q50) otherwise makes every derived
        # ratio explode and swamps the column with meaningless outliers.
        self.scale = max(self.q99 - self.q50, 0.1 * (self.qmax - self.q50),
                         1e-3 * max(abs(self.q50), 1.0))
        self.pk_srt = None
        if peak_null:
            rm = _trailing_max_pow2(np.asarray(dh, dtype=np.float64), set(PEAK_WIN_LOG2))
            self.pk_srt = []
            for m in PEAK_WIN_LOG2:
                w = 1 << m
                seg = rm[m][w - 1:]
                if seg.shape[0] < 8:
                    seg = v
                self.pk_srt.append(np.sort(np.asarray(seg, dtype=np.float64)))

    def usur(self, x: np.ndarray) -> np.ndarray:
        """Upper-tail empirical surprise -log10 P(null >= x).

        The empirical p-value cannot resolve below 1/N, so beyond the historical
        maximum every value would collapse to the same capped number -- exactly
        the region where the breaks live.  We therefore extend the scale above
        the historical maximum by a logarithmic exceedance term in units of the
        historical spread, which keeps the ordering informative and stays
        roughly comparable across series.
        """
        s = self.srt
        n = self.n
        lo = np.searchsorted(s, x, side="left")
        hi = np.searchsorted(s, x, side="right")
        p = 0.5 * (lo + hi) / n
        base = np.minimum(-np.log10(np.maximum(1.0 - p, 1.0 / (2.0 * n))), SUR_CAP)
        ext = np.log1p(np.maximum(x - self.qmax, 0.0) / self.scale)
        return np.minimum(base + ext, ZCAP)

    def usur_peak(self, x: np.ndarray, widx: np.ndarray) -> np.ndarray:
        """Upper-tail surprise of a running peak against a window-matched null.

        ``widx`` selects, per online step, the dyadic window whose length is
        closest to t+1 in log space (a function of t only -> causal).
        """
        if self.pk_srt is None:
            return self.usur(x)
        cols = []
        for s in self.pk_srt:
            n = s.shape[0]
            lo = np.searchsorted(s, x, side="left")
            hi = np.searchsorted(s, x, side="right")
            p = 0.5 * (lo + hi) / n
            base = np.minimum(-np.log10(np.maximum(1.0 - p, 1.0 / (2.0 * n))), SUR_CAP)
            cols.append(np.minimum(base + np.log1p(np.maximum(x - s[-1], 0.0) / self.scale), ZCAP))
        M = np.column_stack(cols)
        return np.take_along_axis(M, widx[:, None], axis=1)[:, 0]

    def usur_win(self, x: np.ndarray, j: int) -> np.ndarray:
        """Upper-tail surprise against the fixed dyadic-window peak null ``j``."""
        if self.pk_srt is None:
            return self.usur(x)
        s = self.pk_srt[j]
        n = s.shape[0]
        lo = np.searchsorted(s, x, side="left")
        hi = np.searchsorted(s, x, side="right")
        p = 0.5 * (lo + hi) / n
        base = np.minimum(-np.log10(np.maximum(1.0 - p, 1.0 / (2.0 * n))), SUR_CAP)
        return np.minimum(base + np.log1p(np.maximum(x - s[-1], 0.0) / self.scale), ZCAP)

    def rz(self, x: np.ndarray) -> np.ndarray:
        """Robust z of a two-sided statistic against the historical path."""
        s = self.srt
        n = self.n
        q1 = s[int(0.25 * n)]
        q3 = s[min(n - 1, int(0.75 * n))]
        sd = max((q3 - q1) / 1.349, 1e-9)
        return np.clip((x - self.q50) / sd, -ZCAP, ZCAP)


# ---------------------------------------------------------------- module
@register("m01_seq", version="1", owner="agent5")
def build(ctx):
    n = ctx.n
    idx = np.arange(n, dtype=np.float64)
    tt = idx + 1.0
    cols: list[str] = []
    out: list[np.ndarray] = []

    # window index used to score the running peak (function of t only)
    widx = np.argmin(np.abs(np.log(tt)[:, None] - _LOG_PEAK_WIN[None, :]), axis=1).astype(np.intp)

    # ---- streams, standardised by HISTORICAL params only -------------------
    z = np.asarray(ctx.tr["mean"], dtype=np.float64)
    zh = np.asarray(ctx.hist_tr["mean"], dtype=np.float64)
    if "res_mean" in ctx.tr:
        e = np.asarray(ctx.tr["res_mean"], dtype=np.float64)
        eh = np.asarray(ctx.hist_tr["res_mean"], dtype=np.float64)
    else:                                                       # pragma: no cover
        e, eh = z, zh
    sq_o = np.asarray(ctx.tr["sq"], dtype=np.float64)
    sq_h = np.asarray(ctx.hist_tr["sq"], dtype=np.float64)
    m_sq = float(sq_h.mean())
    s_sq = max(float(sq_h.std()), 1e-9)
    v = (sq_o - m_sq) / s_sq                 # standardised squared stream
    vh = (sq_h - m_sq) / s_sq

    cur_pool: list[np.ndarray] = []           # for the cross-channel summary
    pk_pool: list[np.ndarray] = []

    # ------------------------------------------------------------------ helpers
    def full_pack(p: str, d: np.ndarray, dh: np.ndarray):
        """7 shape columns for one detector, all historically calibrated."""
        ch = _Chan(dh, peak_null=True)
        peak = np.maximum.accumulate(d)
        cur_s = ch.usur(d)
        pk_s = ch.usur_peak(peak, widx)
        dpk = _decayed_peak(d, DECAY_HL)
        dpk_s = ch.usur_win(dpk, 1)                     # 128-step window null
        last = np.maximum.accumulate(np.where(d >= peak, idx, -1.0))
        tsp = (idx - last) / tt
        per = np.cumsum(d > ch.q99) / tt
        slp = np.full(n, np.nan)
        if n > SLOPE_K:
            slp[SLOPE_K:] = np.clip((d[SLOPE_K:] - d[:-SLOPE_K]) / (SLOPE_K * ch.scale),
                                    -ZCAP, ZCAP)
        rel = d / (peak + EPS)
        for suf, arr in (("cur", cur_s), ("pk", pk_s), ("dpk", dpk_s), ("tsp", tsp),
                         ("per", per), ("slp", slp), ("rel", rel)):
            cols.append(f"{p}_{suf}")
            out.append(arr)
        cur_pool.append(cur_s)
        pk_pool.append(pk_s)
        return ch

    def small_pack(p: str, d: np.ndarray, dh: np.ndarray, pool=True):
        """2 columns: calibrated current level + peak relative to historical q99."""
        ch = _Chan(dh, peak_null=False)
        cur_s = ch.usur(d)
        peak = np.maximum.accumulate(d)
        pr = np.minimum(np.log1p(np.maximum(peak - ch.q99, 0.0) / ch.scale), ZCAP)
        cols.append(f"{p}_cur")
        out.append(cur_s)
        cols.append(f"{p}_pkr")
        out.append(pr)
        if pool:
            cur_pool.append(cur_s)
        return ch

    def one_col(name: str, arr: np.ndarray):
        cols.append(name)
        out.append(arr)

    # ------------------------------------------------------------------ CUSUM (raw)
    cp50, cn50 = _cusum_pair(z, 0.5)
    hp50, hn50 = _cusum_pair(zh, 0.5)
    ch50 = full_pack("cz50", np.maximum(cp50, cn50), np.maximum(hp50, hn50))
    one_col("cz50_up", _Chan(hp50, False).usur(cp50))
    one_col("cz50_dn", _Chan(hn50, False).usur(cn50))

    cp25, cn25 = _cusum_pair(z, 0.25)
    hp25, hn25 = _cusum_pair(zh, 0.25)
    small_pack("cz25", np.maximum(cp25, cn25), np.maximum(hp25, hn25))

    cp100, cn100 = _cusum_pair(z, 1.0)
    hp100, hn100 = _cusum_pair(zh, 1.0)
    small_pack("cz100", np.maximum(cp100, cn100), np.maximum(hp100, hn100))

    # ------------------------------------------------------------------ CUSUM (AR residual)
    ep50, en50 = _cusum_pair(e, 0.5)
    fp50, fn50 = _cusum_pair(eh, 0.5)
    full_pack("ce50", np.maximum(ep50, en50), np.maximum(fp50, fn50))
    one_col("ce50_up", _Chan(fp50, False).usur(ep50))
    one_col("ce50_dn", _Chan(fn50, False).usur(en50))

    ep25, en25 = _cusum_pair(e, 0.25)
    fp25, fn25 = _cusum_pair(eh, 0.25)
    small_pack("ce25", np.maximum(ep25, en25), np.maximum(fp25, fn25))

    # ------------------------------------------------------------------ CUSUM-SQ (variance)
    vp50, vn50 = _cusum_pair(v, 0.5)
    wp50, wn50 = _cusum_pair(vh, 0.5)
    full_pack("sq50", np.maximum(vp50, vn50), np.maximum(wp50, wn50))
    one_col("sq50_up", _Chan(wp50, False).usur(vp50))
    one_col("sq50_dn", _Chan(wn50, False).usur(vn50))

    vp25, vn25 = _cusum_pair(v, 0.25)
    wp25, wn25 = _cusum_pair(vh, 0.25)
    small_pack("sq25", np.maximum(vp25, vn25), np.maximum(wp25, wn25))

    # ------------------------------------------------------------------ Page-Hinkley
    pu, pd = _page_hinkley(z, 0.5)
    qu, qd = _page_hinkley(zh, 0.5)
    small_pack("phu", pu, qu)
    small_pack("phd", pd, qd)

    # ------------------------------------------------------------------ Shiryaev-Roberts
    sr = _sr_log(z, 0.5)
    srh = _sr_log(zh, 0.5)
    small_pack("srz", sr, srh)

    # ------------------------------------------------------------------ GLR / mixture GLR
    g, gm = _glr_dyadic(z)
    gh, gmh = _glr_dyadic(zh)
    full_pack("glz", g, gh)
    one_col("glz_mix", _Chan(gmh, False).usur(gm))

    ge, gem = _glr_dyadic(e)
    geh, gemh = _glr_dyadic(eh)
    small_pack("gle", ge, geh)


    # ------------------------------------------------------------------ EWMA bank
    for hl in EWMA_HL:
        w = _ewma(z, hl)
        wh = _ewma(zh, hl)
        chw = _Chan(np.abs(wh), False)
        one_col(f"ew{int(hl)}_z", chw.rz(np.abs(w)) * np.sign(w))
    ew32 = _ewma(z, 32.0)
    a32 = np.abs(ew32)
    one_col("ew32_pkrel", a32 / (np.maximum.accumulate(a32) + EPS))
    slp = np.full(n, np.nan)
    if n > SLOPE_K:
        slp[SLOPE_K:] = (ew32[SLOPE_K:] - ew32[:-SLOPE_K]) / SLOPE_K
    one_col("ew32_slp", slp)

    # EWMA variance-ratio channel
    zs = np.asarray(ctx.tr["sq"], dtype=np.float64)
    r32 = np.log((_ewma(zs, 32.0) + 1e-6) / (m_sq + 1e-6))
    rh32 = np.log((_ewma(sq_h, 32.0) + 1e-6) / (m_sq + 1e-6))
    chv = _Chan(np.abs(rh32), False)
    one_col("ewv32_z", chv.rz(np.abs(r32)) * np.sign(r32))

    # ------------------------------------------------------------------ cross-channel shape
    M = np.column_stack(cur_pool)
    one_col("xc_max_cur", M.max(axis=1))
    one_col("xc_n_hot", (M > 2.0).sum(axis=1).astype(np.float64))
    P = np.column_stack(pk_pool)
    one_col("xc_max_pk", P.max(axis=1))

    A = np.column_stack(out).astype(np.float32)
    A[~np.isfinite(A)] = np.nan
    assert len(cols) <= 60, f"m01_seq column budget exceeded: {len(cols)}"
    return cols, A
