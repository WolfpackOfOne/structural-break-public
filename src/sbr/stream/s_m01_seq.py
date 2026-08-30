"""StreamM01Seq -- exact incremental port of ``sbr.features.m01_seq``.

Design
------
``m01_seq`` is, underneath the presentation layer, a bank of *recursions*:

    CUSUM / Page / Page-Hinkley   S_t = max(0, S_{t-1} + y_t)
    Shiryaev-Roberts (log)        r_t = l_t + log1p(exp(r_{t-1}))
    EWMA                          y_t = a x_t + (1-a) y_{t-1}
    dyadic GLR                    a fixed set of 10 cumsum differences at t
    running / decayed peak        running max, running max in log space

Every one of those is prefix-stable *by construction*, and the batch module
implements them with strictly sequential numpy accumulators
(``np.cumsum`` / ``np.minimum.accumulate`` / ``np.maximum.accumulate`` /
``scipy.signal.lfilter``).  Sequential accumulators are bitwise reproducible
one element at a time, so this port carries the scalar state and applies the
identical arithmetic -- there is no reformulation anywhere.

The three places that needed care:

1.  ``np.minimum.accumulate`` / ``np.maximum.accumulate`` NaN semantics.
    ``npy_minimum(a, b) = (a < b || isnan(a)) ? a : b``.  ``_mn``/``_mx`` below
    reproduce that exactly, so a NaN that enters a detector poisons its state
    for ever in the stream just as it does in batch.
2.  The GLR loop is written over ``L in DYADIC`` in *increasing* order, exactly
    as batch does, because ``acc`` is a floating-point sum whose value depends
    on summation order.  Batch's ``if L > n: break`` uses the full online
    length, but for row ``t`` only ``L <= t+1`` writes anything (``s[L-1:]``);
    the larger ``L`` only contribute ``np.maximum(best, 0.0)``, which is a
    no-op because ``best`` starts at ``0.0``.  So iterating ``L <= t+1`` is
    bitwise equivalent and needs no knowledge of the future length.
3.  ``(1-a) ** np.arange(...)`` (the EWMA bias correction) is *not* bitwise
    equal to the scalar ``(1-a) ** float(t+1)``.  We therefore precompute the
    same vectorised expression into a table and index it, growing by doubling.

The historical half (every ``_Chan`` null) is built by calling the batch
module's own private helpers on the historical arrays, so there is exactly one
implementation of the calibration and it cannot drift.

Cost: O(log n) per observation (the 10-node dyadic GLR grid) plus O(log H) per
empirical-p lookup.  No rescan of the online prefix anywhere.
"""
from __future__ import annotations

import math

import numpy as np

from sbr.features.m01_seq import (
    DECAY_HL,
    DYADIC,
    EPS,
    EWMA_HL,
    PEAK_WIN_LOG2,
    SLOPE_K,
    SUR_CAP,
    ZCAP,
    _LOG_PEAK_WIN,
    _Chan,
    _cusum_pair,
    _ewma,
    _glr_dyadic,
    _page_hinkley,
    _sr_log,
)
from sbr.stream._fp import fma as _fma

_TABLE0 = 1024


# --------------------------------------------------------------- scalar prims
def _mn(a, b):
    """Bitwise ``np.minimum`` for scalars (NaN propagates from either side)."""
    if a != a:
        return a
    return a if a < b else b


def _mx(a, b):
    """Bitwise ``np.maximum`` for scalars (NaN propagates from either side)."""
    if a != a:
        return a
    return a if a > b else b


def _clip(x, lo, hi):
    """Bitwise ``np.clip`` for scalars."""
    if x != x:
        return x
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


# --------------------------------------------------------------- _Chan scoring
class _Null:
    """Flattened, read-optimised view of a fitted ``_Chan``.

    Nothing here is recomputed: every field is copied out of the ``_Chan`` the
    batch module itself built from the historical detector path.  The only
    reason this class exists is to hoist the attribute lookups and the
    ``1/(2n)`` / ``pk_srt[j][-1]`` constants out of the per-observation path.
    """

    __slots__ = ("srt", "n", "inv2n", "qmax", "scale", "q50", "q99", "rzsd",
                 "pk_srt", "pk_n", "pk_inv2n", "pk_max")

    def __init__(self, ch):
        self.srt = ch.srt
        self.n = ch.n
        self.inv2n = 1.0 / (2.0 * ch.n)
        self.qmax = ch.qmax
        self.scale = ch.scale
        self.q50 = ch.q50
        self.q99 = ch.q99
        s, n = ch.srt, ch.n
        q1 = s[int(0.25 * n)]
        q3 = s[min(n - 1, int(0.75 * n))]
        self.rzsd = max((q3 - q1) / 1.349, 1e-9)
        self.pk_srt = ch.pk_srt
        if ch.pk_srt is None:
            self.pk_n = self.pk_inv2n = self.pk_max = None
        else:
            self.pk_n = [a.shape[0] for a in ch.pk_srt]
            self.pk_inv2n = [1.0 / (2.0 * a.shape[0]) for a in ch.pk_srt]
            self.pk_max = [a[-1] for a in ch.pk_srt]


def _usur(nl, x):
    """Scalar form of ``_Chan.usur``.

    ``x`` NaN short-circuits to NaN: batch reaches the same answer because
    ``np.maximum(nan - qmax, 0.0)`` is NaN and NaN survives ``log1p`` and both
    ``np.minimum`` calls.
    """
    if x != x:
        return x
    s = nl.srt
    p = 0.5 * (s.searchsorted(x, "left") + s.searchsorted(x, "right")) / nl.n
    m = 1.0 - p
    if m < nl.inv2n:                      # == np.maximum (neither side is NaN)
        m = nl.inv2n
    base = -np.log10(m)
    if base > SUR_CAP:                    # == np.minimum
        base = SUR_CAP
    d = x - nl.qmax
    if not (d > 0.0):                     # == np.maximum(d, 0.0), incl. -0.0
        d = 0.0
    r = base + np.log1p(d / nl.scale)
    return r if r < ZCAP else ZCAP


def _usur_win(nl, x, j):
    """Scalar form of ``_Chan.usur_win`` (== ``usur_peak`` at column ``j``)."""
    if nl.pk_srt is None:
        return _usur(nl, x)
    if x != x:
        return x
    s = nl.pk_srt[j]
    p = 0.5 * (s.searchsorted(x, "left") + s.searchsorted(x, "right")) / nl.pk_n[j]
    inv = nl.pk_inv2n[j]
    m = 1.0 - p
    if m < inv:
        m = inv
    base = -np.log10(m)
    if base > SUR_CAP:
        base = SUR_CAP
    d = x - nl.pk_max[j]
    if not (d > 0.0):
        d = 0.0
    r = base + np.log1p(d / nl.scale)
    return r if r < ZCAP else ZCAP


def _rz(nl, x):
    """Scalar form of ``_Chan.rz`` (the robust sd is a historical constant)."""
    return _clip((x - nl.q50) / nl.rzsd, -ZCAP, ZCAP)


# --------------------------------------------------------------- recursions
class _Refl:
    """``S_t = max(0, S_{t-1} + y_t)`` written exactly as batch computes it:
    a running cumsum minus its running minimum (seeded with the leading 0)."""

    __slots__ = ("c", "m")

    def __init__(self):
        self.c = 0.0
        self.m = 0.0

    def push(self, y):
        c = self.c + y
        self.c = c
        self.m = _mn(self.m, c)
        return c - self.m


class _Ewma:
    """Bias-corrected EWMA; identical to ``lfilter`` + the arange-power table.

    ``lfilter`` runs the *transposed direct form II* -- ``y = a*x + z`` then
    ``z = (1-a)*y`` -- and contracts the multiply-add into an FMA.  Carrying
    ``z`` and fusing here reproduces it bitwise; the previous
    ``a*x + b*y_prev`` form rounded twice and drifted by 1 ULP.
    """

    __slots__ = ("a", "b", "z", "den")

    def __init__(self, hl, den):
        a = 1.0 - 2.0 ** (-1.0 / hl)
        self.a = a
        self.b = 1.0 - a
        self.z = 0.0            # lfilter's zi when no initial state is given
        self.den = den

    def push(self, x, t):
        y = _fma(self.a, x, self.z)
        self.z = self.b * y
        return y / self.den[t]


class _Sr:
    """log Shiryaev-Roberts; mirrors ``_sr_log_kernel`` branch for branch."""

    __slots__ = ("r",)

    def __init__(self):
        self.r = -1.0e300

    def push(self, l):                                       # noqa: E741
        r = self.r
        if r > 30.0:
            s = r + math.log1p(math.exp(-r))
        elif r < -30.0:
            s = math.exp(r)
        else:
            s = math.log1p(math.exp(r))
        r = l + s
        self.r = r
        return r


class _Peak:
    """``np.maximum.accumulate`` as a scalar."""

    __slots__ = ("p", "first")

    def __init__(self):
        self.p = 0.0
        self.first = True

    def push(self, d):
        if self.first:
            self.p = d
            self.first = False
        else:
            self.p = _mx(self.p, d)
        return self.p


# --------------------------------------------------------------- packs
class _FullPack:
    """The 7 shape columns of ``full_pack`` for one detector."""

    __slots__ = ("ch", "pk", "dacc", "dfirst", "last", "cnt", "ring", "lr", "slope_den")

    def __init__(self, ch):
        self.ch = _Null(ch)
        self.pk = _Peak()
        self.dacc = 0.0
        self.dfirst = True
        self.last = -1.0
        self.cnt = 0
        self.ring = []
        self.lr = -math.log(2.0) / DECAY_HL
        self.slope_den = SLOPE_K * self.ch.scale

    def step(self, d, t, tt, widx, out):
        ch = self.ch
        peak = self.pk.push(d)

        cur_s = _usur(ch, d)

        pk_s = _usur_win(ch, peak, widx)

        # decayed peak, in log space, exactly as ``_decayed_peak``
        ft = float(t)
        a = np.log(_mx(d, 0.0) + EPS) - ft * self.lr
        if self.dfirst:
            self.dacc = a
            self.dfirst = False
        else:
            self.dacc = _mx(self.dacc, a)
        dpk = np.exp(ft * self.lr + self.dacc)
        dpk_s = _usur_win(ch, dpk, 1)

        if d >= peak:
            self.last = ft
        tsp = (ft - self.last) / tt

        if d > ch.q99:
            self.cnt += 1
        per = self.cnt / tt

        r = self.ring
        r.append(d)
        if len(r) > SLOPE_K + 1:
            del r[0]
        if t >= SLOPE_K:
            slp = _clip((d - r[0]) / self.slope_den, -ZCAP, ZCAP)
        else:
            slp = np.nan

        rel = d / (peak + EPS)

        out.append(cur_s)
        out.append(pk_s)
        out.append(dpk_s)
        out.append(tsp)
        out.append(per)
        out.append(slp)
        out.append(rel)
        return cur_s, pk_s


class _SmallPack:
    """The 2 columns of ``small_pack`` for one detector."""

    __slots__ = ("ch", "pk")

    def __init__(self, ch):
        self.ch = _Null(ch)
        self.pk = _Peak()

    def step(self, d, out):
        ch = self.ch
        cur_s = _usur(ch, d)
        peak = self.pk.push(d)
        pr = _mn(np.log1p(_mx(peak - ch.q99, 0.0) / ch.scale), ZCAP)
        out.append(cur_s)
        out.append(pr)
        return cur_s


# --------------------------------------------------------------- module
COLS = [
    "cz50_cur", "cz50_pk", "cz50_dpk", "cz50_tsp", "cz50_per", "cz50_slp", "cz50_rel",
    "cz50_up", "cz50_dn",
    "cz25_cur", "cz25_pkr",
    "cz100_cur", "cz100_pkr",
    "ce50_cur", "ce50_pk", "ce50_dpk", "ce50_tsp", "ce50_per", "ce50_slp", "ce50_rel",
    "ce50_up", "ce50_dn",
    "ce25_cur", "ce25_pkr",
    "sq50_cur", "sq50_pk", "sq50_dpk", "sq50_tsp", "sq50_per", "sq50_slp", "sq50_rel",
    "sq50_up", "sq50_dn",
    "sq25_cur", "sq25_pkr",
    "phu_cur", "phu_pkr",
    "phd_cur", "phd_pkr",
    "srz_cur", "srz_pkr",
    "glz_cur", "glz_pk", "glz_dpk", "glz_tsp", "glz_per", "glz_slp", "glz_rel",
    "glz_mix",
    "gle_cur", "gle_pkr",
    "ew8_z", "ew32_z", "ew128_z",
    "ew32_pkrel", "ew32_slp", "ewv32_z",
    "xc_max_cur", "xc_n_hot", "xc_max_pk",
]


class StreamM01Seq:
    """Incremental, bitwise-identical port of the ``m01_seq`` batch module."""

    MODULE = "m01_seq"

    def __init__(self):
        self._fitted = False

    # ------------------------------------------------------------------ cols
    @property
    def cols(self) -> list[str]:
        return list(COLS)

    # -------------------------------------------------------------- tables
    def _grow_tables(self, need: int) -> None:
        size = self._tab_n
        while size < need:
            size *= 2
        idx = np.arange(1, size + 1, dtype=np.float64)
        self._den = {hl: 1.0 - (1.0 - (1.0 - 2.0 ** (-1.0 / hl))) ** idx for hl in self._hls}
        tt = np.arange(size, dtype=np.float64) + 1.0
        self._widx = np.argmin(
            np.abs(np.log(tt)[:, None] - _LOG_PEAK_WIN[None, :]), axis=1
        ).astype(np.intp)
        self._tab_n = size
        for e, hl in zip(self._ew_objs, self._ew_hls):
            e.den = self._den[hl]

    # ---------------------------------------------------------------- fit
    def fit_historical(self, ctx) -> None:
        zh = np.asarray(ctx.hist_tr["mean"], dtype=np.float64)
        if "res_mean" in ctx.hist_tr:
            eh = np.asarray(ctx.hist_tr["res_mean"], dtype=np.float64)
        else:                                                # pragma: no cover
            eh = zh
        self._has_res = "res_mean" in ctx.hist_tr

        sq_h = np.asarray(ctx.hist_tr["sq"], dtype=np.float64)
        self._m_sq = float(sq_h.mean())
        self._s_sq = max(float(sq_h.std()), 1e-9)
        vh = (sq_h - self._m_sq) / self._s_sq

        # ---- historical detector paths (batch code, verbatim) -------------
        hp50, hn50 = _cusum_pair(zh, 0.5)
        hp25, hn25 = _cusum_pair(zh, 0.25)
        hp100, hn100 = _cusum_pair(zh, 1.0)
        fp50, fn50 = _cusum_pair(eh, 0.5)
        fp25, fn25 = _cusum_pair(eh, 0.25)
        wp50, wn50 = _cusum_pair(vh, 0.5)
        wp25, wn25 = _cusum_pair(vh, 0.25)
        qu, qd = _page_hinkley(zh, 0.5)
        srh = _sr_log(zh, 0.5)
        gh, gmh = _glr_dyadic(zh)
        geh, gemh = _glr_dyadic(eh)

        # ---- nulls --------------------------------------------------------
        self._f_cz50 = _FullPack(_Chan(np.maximum(hp50, hn50), peak_null=True))
        self._c_cz50_up = _Null(_Chan(hp50, False))
        self._c_cz50_dn = _Null(_Chan(hn50, False))
        self._s_cz25 = _SmallPack(_Chan(np.maximum(hp25, hn25), peak_null=False))
        self._s_cz100 = _SmallPack(_Chan(np.maximum(hp100, hn100), peak_null=False))

        self._f_ce50 = _FullPack(_Chan(np.maximum(fp50, fn50), peak_null=True))
        self._c_ce50_up = _Null(_Chan(fp50, False))
        self._c_ce50_dn = _Null(_Chan(fn50, False))
        self._s_ce25 = _SmallPack(_Chan(np.maximum(fp25, fn25), peak_null=False))

        self._f_sq50 = _FullPack(_Chan(np.maximum(wp50, wn50), peak_null=True))
        self._c_sq50_up = _Null(_Chan(wp50, False))
        self._c_sq50_dn = _Null(_Chan(wn50, False))
        self._s_sq25 = _SmallPack(_Chan(np.maximum(wp25, wn25), peak_null=False))

        self._s_phu = _SmallPack(_Chan(qu, peak_null=False))
        self._s_phd = _SmallPack(_Chan(qd, peak_null=False))
        self._s_srz = _SmallPack(_Chan(srh, peak_null=False))

        self._f_glz = _FullPack(_Chan(gh, peak_null=True))
        self._c_glz_mix = _Null(_Chan(gmh, False))
        self._s_gle = _SmallPack(_Chan(geh, peak_null=False))

        # ---- tables (EWMA bias correction + peak-null window index) --------
        self._hls = sorted({*EWMA_HL, 32.0})
        self._tab_n = _TABLE0
        self._ew_objs = []
        self._ew_hls = []
        self._grow_tables(_TABLE0)

        # ---- EWMA channels ------------------------------------------------
        self._ew = []
        for hl in EWMA_HL:
            wh = _ewma(zh, hl)
            nl = _Null(_Chan(np.abs(wh), False))
            e = _Ewma(hl, self._den[hl])
            self._ew.append((e, nl))
            self._ew_objs.append(e)
            self._ew_hls.append(hl)
        # note: batch calls _ewma(z, 32.0) twice (once in the bank loop, once
        # for ew32_pkrel/ew32_slp); the two paths are the same recursion on the
        # same input, so one accumulator serves both.
        rh32 = np.log((_ewma(sq_h, 32.0) + 1e-6) / (self._m_sq + 1e-6))
        self._chv = _Null(_Chan(np.abs(rh32), False))
        self._ewv = _Ewma(32.0, self._den[32.0])
        self._ew_objs.append(self._ewv)
        self._ew_hls.append(32.0)

        # ---- online recursion state ---------------------------------------
        self._r = {k: _Refl() for k in (
            "zp50", "zn50", "zp25", "zn25", "zp100", "zn100",
            "ep50", "en50", "ep25", "en25",
            "vp50", "vn50", "vp25", "vn25",
            "phu", "phd",
        )}
        self._sr = _Sr()
        self._ew32_pk = _Peak()
        self._ew32_ring = []
        self._fitted = True

    # --------------------------------------------------------------- step
    def _glr(self, c, t):
        """Dyadic GLR (best, mix) at index ``t`` from a cumsum-with-leading-zero."""
        best = 0.0
        acc = 0.0
        cnt = 0.0
        ct = c[t + 1]
        for L in DYADIC:
            if L > t + 1:
                break
            diff = ct - c[t + 1 - L]
            s = diff * diff / (2.0 * L)
            best = _mx(best, s)
            acc = acc + np.exp(_mn(s, 40.0))
            cnt = cnt + 1.0
        mix = np.log(acc / _mx(cnt, 1.0) + EPS)
        return best, mix

    def step(self, ctx) -> np.ndarray:
        t = ctx.t
        if t + 1 > self._tab_n:
            self._grow_tables(t + 1)
        tt = float(t) + 1.0
        widx = int(self._widx[t])

        z = ctx.tr["mean"][t]
        e = ctx.tr["res_mean"][t] if self._has_res else z
        v = (ctx.tr["sq"][t] - self._m_sq) / self._s_sq

        R = self._r
        out: list = []
        cur_pool: list = []
        pk_pool: list = []

        # ------------------------------------------------------ CUSUM (raw z)
        cp50 = R["zp50"].push(z - 0.5)
        cn50 = R["zn50"].push(-z - 0.5)
        c50, p50 = self._f_cz50.step(_mx(cp50, cn50), t, tt, widx, out)
        cur_pool.append(c50)
        pk_pool.append(p50)
        out.append(_usur(self._c_cz50_up, cp50))
        out.append(_usur(self._c_cz50_dn, cn50))

        cp25 = R["zp25"].push(z - 0.25)
        cn25 = R["zn25"].push(-z - 0.25)
        cur_pool.append(self._s_cz25.step(_mx(cp25, cn25), out))

        cp100 = R["zp100"].push(z - 1.0)
        cn100 = R["zn100"].push(-z - 1.0)
        cur_pool.append(self._s_cz100.step(_mx(cp100, cn100), out))

        # ------------------------------------------------- CUSUM (AR residual)
        ep50 = R["ep50"].push(e - 0.5)
        en50 = R["en50"].push(-e - 0.5)
        c, p = self._f_ce50.step(_mx(ep50, en50), t, tt, widx, out)
        cur_pool.append(c)
        pk_pool.append(p)
        out.append(_usur(self._c_ce50_up, ep50))
        out.append(_usur(self._c_ce50_dn, en50))

        ep25 = R["ep25"].push(e - 0.25)
        en25 = R["en25"].push(-e - 0.25)
        cur_pool.append(self._s_ce25.step(_mx(ep25, en25), out))

        # ----------------------------------------------- CUSUM-SQ (variance)
        vp50 = R["vp50"].push(v - 0.5)
        vn50 = R["vn50"].push(-v - 0.5)
        c, p = self._f_sq50.step(_mx(vp50, vn50), t, tt, widx, out)
        cur_pool.append(c)
        pk_pool.append(p)
        out.append(_usur(self._c_sq50_up, vp50))
        out.append(_usur(self._c_sq50_dn, vn50))

        vp25 = R["vp25"].push(v - 0.25)
        vn25 = R["vn25"].push(-v - 0.25)
        cur_pool.append(self._s_sq25.step(_mx(vp25, vn25), out))

        # ------------------------------------------------------ Page-Hinkley
        rm = ctx.cum["mean"][t + 1] / tt
        dph = z - rm
        cur_pool.append(self._s_phu.step(R["phu"].push(dph - 0.5), out))
        cur_pool.append(self._s_phd.step(R["phd"].push(-dph - 0.5), out))

        # -------------------------------------------------- Shiryaev-Roberts
        cur_pool.append(self._s_srz.step(self._sr.push(0.5 * z - 0.5 * 0.5 * 0.5), out))

        # ------------------------------------------------------------- GLR
        g, gm = self._glr(ctx.cum["mean"], t)
        c, p = self._f_glz.step(g, t, tt, widx, out)
        cur_pool.append(c)
        pk_pool.append(p)
        out.append(_usur(self._c_glz_mix, gm))

        if self._has_res:
            ge, _gem = self._glr(ctx.cum["res_mean"], t)
        else:                                                # pragma: no cover
            ge, _gem = g, gm
        cur_pool.append(self._s_gle.step(ge, out))

        # ------------------------------------------------------- EWMA bank
        w32 = None
        for (eo, nl), hl in zip(self._ew, EWMA_HL):
            w = eo.push(z, t)
            out.append(_rz(nl, np.abs(w)) * np.sign(w))
            if hl == 32.0:
                w32 = w
        a32 = np.abs(w32)
        out.append(a32 / (self._ew32_pk.push(a32) + EPS))
        ring = self._ew32_ring
        ring.append(w32)
        if len(ring) > SLOPE_K + 1:
            del ring[0]
        out.append((w32 - ring[0]) / SLOPE_K if t >= SLOPE_K else np.nan)

        # ------------------------------------------- EWMA variance ratio
        r32 = np.log((self._ewv.push(ctx.tr["sq"][t], t) + 1e-6) / (self._m_sq + 1e-6))
        out.append(_rz(self._chv, np.abs(r32)) * np.sign(r32))

        # -------------------------------------------------- cross-channel
        mx = cur_pool[0]
        hot = 0.0
        for cval in cur_pool:
            mx = _mx(mx, cval)
            if cval > 2.0:
                hot += 1.0
        out.append(mx)
        out.append(hot)
        mp = pk_pool[0]
        for pval in pk_pool:
            mp = _mx(mp, pval)
        out.append(mp)

        # batch casts to float32 and only then replaces non-finite with NaN
        row = np.asarray(out, dtype=np.float64).astype(np.float32)
        row[~np.isfinite(row)] = np.nan
        return row.astype(np.float64)
