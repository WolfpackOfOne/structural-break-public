"""Incremental (streaming) port of the batch feature module ``m02_dist``.

Design
------
The batch module is built on exactly three kinds of object:

1. **Historical nulls.** Every calibrated column looks the online statistic up
   in a sorted vector of the *same* statistic evaluated over contiguous
   historical windows.  Those vectors depend on ``hist`` only, so they are
   computed once in :meth:`StreamM02Dist.fit_historical` -- literally by calling
   the batch helpers (``_Stream.null``, ``_roll_null``, ``_cusum_null``) on a
   ``_Stream`` whose *online* half is empty.  There is therefore exactly one
   definition of the null, and no chance of drift.

2. **Cumulative online counters.**  ``_onehot_cum`` / ``_cum`` are plain
   sequential cumulative sums, so ``c[t+1] = c[t] + v[t]`` reproduces them
   bit for bit (this is the same argument ``StreamCtx`` uses).  The occupancy
   counters are *integer* counts stored in float64, so windowed differences
   ``C[t+1] - C[t+1-w]`` are exact; the PIT means/second moments are float
   sums, but batch itself forms them as differences of the very same cumulative
   array, so we are not "maintaining a running float" -- we keep the cumulative
   array and difference it exactly as batch does.

3. **Row-wise statistics.**  ``_stats`` is fully row-wise: every expression is
   elementwise plus a reduction along axis 1.  Reductions along the last
   (contiguous) axis of a ``(m, B)`` array give, row by row, the same float as
   the reduction on a ``(1, B)`` array (verified empirically for B = 10, 20).
   So we call the *batch* ``_stats`` on a one-row matrix.

The only non-obvious causality point is ``_Emit.exp``: it buckets the expanding
window to the nearest null length in log space via
``argmin |log(L) - log(g)|``.  That is a function of ``t`` alone, so we
precompute the whole map once over ``L = 1 .. cap`` using the identical
expression (``np.log`` is length-stable, checked).

Cost per observation
--------------------
O(B) with B <= 20 and a bounded number of numpy calls:

  * the ten window/expanding statistic rows are stacked into exactly TWO
    ``_stats`` calls, one per bin resolution (rows are independent, so this is
    bitwise identical and costs one numpy dispatch instead of ten);
  * every calibrated column is a table lookup.  ``_up``/``_sg`` depend on the
    null only through ``m = len(rs)`` and the integer ``c = lo + hi``, so the
    whole surprise curve is precomputed at fit time as 2m+1 <= 1001 floats and
    the per-observation work is two ``bisect`` calls and one index;
  * the historical-ECDF PIT of the |x-med| and AR-residual streams is likewise
    tabulated over ``c``;
  * the quantile block of ``_stats`` (a gather plus five scalar expressions,
    but ~150 us/obs of numpy dispatch) is evaluated in scalar arithmetic.

Nothing is rescanned: no sorting, no re-null-ing, no FFT, no revisiting of the
online prefix.  Every fast path above is pinned bitwise against the expression
it replaces by a dedicated differential fuzz test.  Measured throughput is
reported by the test suite.
"""
from __future__ import annotations

from bisect import bisect_left as _bl, bisect_right as _br
from math import sqrt as _sqrt

import numpy as np

from sbr.features.m02_dist import (  # single source of truth -- do not reimplement
    B_COARSE,
    B_FINE,
    CLIP,
    G_EXP,
    QLEV,
    W_FIX,
    W_TAIL,
    _cum,
    _cusum_null,
    _pit_against,
    _roll_null,
    _stats,
    _Stream,
)

_EMPTY = np.zeros(0, dtype=np.float64)
_TAIL_KEYS = ("asym05", "asym01", "ctr")
nan = np.nan


# --------------------------------------------------------------------------
# Scalar fast paths for the two null-lookup kernels.  These are the SAME
# expressions as ``m02_dist._up`` / ``m02_dist._sg`` evaluated on one value,
# with numpy's array dispatch (``np.searchsorted`` -> ``_wrapfunc``,
# ``np.clip`` -> ``_methods._clip``) replaced by scalar equivalents.  Every
# arithmetic step is IEEE-identical: ``0.5*(lo+hi)/m`` is the same float64
# expression, ``np.maximum``/``np.minimum``/``np.clip`` against non-NaN bounds
# are ordinary comparisons, and ``np.log10`` is verified to give the identical
# float for a scalar and for the corresponding element of an array.
# ``tests/test_stream_parity_m02_dist.py::test_null_lookup_kernels_bitwise``
# pins this against the batch functions over 200k random inputs.
# --------------------------------------------------------------------------
def _up_ct(m: int, c: int) -> float:
    """``_up`` as a function of ``m = len(rs)`` and ``c = lo + hi`` only."""
    p = 1.0 - 0.5 * c / m
    floor = 1.0 / (2.0 * m)
    if p < floor:
        p = floor
    s = -np.log10(p)
    if s < 0.0:
        s = 0.0
    elif s > CLIP:
        s = CLIP
    return float(s)


def _sg_ct(m: int, c: int) -> float:
    """``_sg`` as a function of ``m = len(rs)`` and ``c = lo + hi`` only."""
    p = 0.5 * c / m
    q = 1.0 - p
    two = 2.0 * (p if p < q else q)
    floor = 1.0 / (2.0 * m)
    if two < floor:
        two = floor
    s = -np.log10(two)
    if s < 0.0:
        s = 0.0
    elif s > CLIP:
        s = CLIP
    return float(np.sign(p - 0.5) * s)


def _up1(rs: np.ndarray, v) -> float:
    return _up_ct(len(rs), rs.searchsorted(v, side="left")
                  + rs.searchsorted(v, side="right"))


def _sg1(rs: np.ndarray, v) -> float:
    return _sg_ct(len(rs), rs.searchsorted(v, side="left")
                  + rs.searchsorted(v, side="right"))


# --------------------------------------------------------------------------
# A calibrated column is  table[bisect_left(rs, v) + bisect_right(rs, v)].
#
# Two observations make the per-observation cost constant:
#   * ``_up``/``_sg`` depend on the null ONLY through the integer
#     ``c = lo + hi in [0, 2m]`` and the length ``m``, so the whole surprise
#     curve is a table of 2m+1 <= 1001 floats.  The table is filled by calling
#     ``_up_ct``/``_sg_ct`` -- the same code ``_up1``/``_sg1`` run, which the
#     test suite pins bitwise against the batch ``_up``/``_sg``.
#   * ``bisect`` on a Python list of the same floats returns exactly the same
#     indices as ``np.searchsorted`` for every non-NaN key (both are plain IEEE
#     ``<`` binary searches; fuzz-tested), and is ~3x cheaper.  NaN is the one
#     case where they differ (numpy orders NaN last, bisect orders it first),
#     so a NaN key -- and any null vector that itself contains a NaN -- falls
#     back to the numpy path.
# --------------------------------------------------------------------------
class _NT:
    """One historical null: sorted values + the two tabulated surprise curves."""

    __slots__ = ("arr", "lst", "up", "sg", "m")

    def __init__(self, rs: np.ndarray, cache: dict):
        self.arr = rs
        self.m = m = len(rs)
        # bisect is only safe when the null itself is NaN-free
        self.lst = rs.tolist() if not np.isnan(rs).any() else None
        tabs = cache.get(m)
        if tabs is None:
            tabs = ([_up_ct(m, c) for c in range(2 * m + 1)],
                    [_sg_ct(m, c) for c in range(2 * m + 1)])
            cache[m] = tabs
        self.up, self.sg = tabs

    def up_of(self, v: float) -> float:
        lst = self.lst
        if lst is None or v != v:
            return _up1(self.arr, v)
        return self.up[_bl(lst, v) + _br(lst, v)]

    def sg_of(self, v: float) -> float:
        lst = self.lst
        if lst is None or v != v:
            return _sg1(self.arr, v)
        return self.sg[_bl(lst, v) + _br(lst, v)]


def _wrap(rs, cache):
    return None if rs is None else _NT(np.asarray(rs, dtype=np.float64), cache)


def _wrap_d(d, cache):
    """Wrap a ``{key: sorted_null}`` dict (or None)."""
    return None if d is None else {k: _NT(v, cache) for k, v in d.items()}


def _pit_table(sorted_ref: np.ndarray) -> tuple:
    """Tabulate ``_pit_against`` as a function of ``lo + hi``.

    ``_pit_against`` is ``(0.5*(lo+hi) + 0.5) / (n+1.0)``; evaluating that same
    ufunc chain on ``c = 0 .. 2n`` gives the identical float64 for every input.
    """
    n = len(sorted_ref)
    c = np.arange(2 * n + 1, dtype=np.intp)
    tab = ((0.5 * c + 0.5) / (n + 1.0)).tolist()
    lst = sorted_ref.tolist() if not np.isnan(sorted_ref).any() else None
    return sorted_ref, lst, tab


def _pit1(pt, x: float) -> float:
    ref, lst, tab = pt
    if lst is None or x != x:
        return float(_pit_against(ref, np.array([x]))[0])
    return tab[_bl(lst, x) + _br(lst, x)]


def _column_names() -> list[str]:
    """Column names in the exact order the batch ``_Emit`` appends them."""
    c: list[str] = []
    for k in ("chi2", "js", "ks", "ene"):
        c.append(f"dv_u_w32_{k}")
    for k in ("chi2", "js", "ks", "cvm", "w1", "ene"):
        c.append(f"dv_u_w128_{k}")
    for k in ("chi2", "js", "ks", "ene"):
        c.append(f"dv_u_exp_{k}")
    c += ["dv_u_w128_js_raw", "dv_u_exp_js_raw"]
    c += ["oc_u_w32_ent", "oc_u_w32_maxdev", "oc_u_w128_ent", "oc_u_w128_maxdev",
          "oc_u_w128_amax", "oc_u_w128_hell", "oc_u_exp_ent", "oc_u_exp_maxdev",
          "oc_u_exp_tv", "oc_u_exp_amax"]
    for k in ("q50", "qiqr", "q9010"):
        c.append(f"qd_w128_{k}_raw")
    for k in ("q50", "qiqr", "q9010"):
        c.append(f"qd_exp_{k}_raw")
    for k in ("q50", "qiqr", "q9010"):
        c.append(f"qd_w128_{k}")
    for k in ("q50", "qiqr"):
        c.append(f"qd_exp_{k}")
    for k in _TAIL_KEYS:
        c.append(f"tl_w{W_TAIL}_{k}")
    for k in ("asym05", "asym01"):
        c.append(f"tl_exp_{k}")
    c.append("tl_exp_asym05_raw")
    c += ["rk_w128_var", "rk_exp_var"]
    for tag in ("u", "r"):
        c += [f"rk_cusum_{tag}_raw", f"rk_cusum_{tag}"]
    c += ["ab_w128_chi2", "ab_w128_ks", "ab_exp_chi2", "ab_exp_js", "ab_w128_mean"]
    c += ["rs_w128_chi2", "rs_w128_ks", "rs_w128_js", "rs_exp_chi2", "rs_exp_js"]
    return c


COLS = _column_names()
NCOL = len(COLS)


class StreamM02Dist:
    """Exact incremental implementation of the ``m02_dist`` batch module."""

    MODULE = "m02_dist"

    def __init__(self, cap: int = 1024):
        self._cap0 = int(cap)
        self.cap = int(cap)
        self._fitted = False

    # ------------------------------------------------------------------ cols
    @property
    def cols(self) -> list[str]:
        return list(COLS)

    # ------------------------------------------------------------------- fit
    def fit_historical(self, ctx) -> None:
        hp = ctx.hp
        hist = np.asarray(ctx.hist, dtype=np.float64)
        nh = len(hist)
        self.nh = nh
        self.hp = hp

        uh = np.asarray(ctx.hist_tr["u"], dtype=np.float64)

        lens_fine = sorted(set(W_FIX) | set(G_EXP))
        lens_alt = sorted({W_FIX[1]} | set(G_EXP))

        # ---- occupancy nulls (history only) -----------------------------
        s20 = _Stream(uh, _EMPTY, B_FINE, quant=True, ene=True)
        self.n20 = {w: s20.null(w) for w in lens_fine}
        s10 = _Stream(uh, _EMPTY, B_COARSE)
        self.n10 = {w: s10.null(w) for w in lens_fine}

        # ---- tail-indicator nulls ---------------------------------------
        hi05, lo05 = (uh > 0.95).astype(float), (uh < 0.05).astype(float)
        hi01, lo01 = (uh > 0.99).astype(float), (uh < 0.01).astype(float)
        ctr_h = ((uh > 0.25) & (uh < 0.75)).astype(float)
        hist_tail = {"asym05": hi05 - lo05, "asym01": hi01 - lo01, "ctr": ctr_h}
        self.tail_fix = {}
        self.tail_exp = {}
        for k, a in hist_tail.items():
            ch = _cum(a)
            self.tail_fix[k] = _roll_null(ch, nh, W_TAIL)
            if k != "ctr":
                self.tail_exp[k] = {g: _roll_null(ch, nh, g) for g in G_EXP}

        # ---- rank-variance nulls ----------------------------------------
        chu, chu2 = _cum(uh), _cum(uh * uh)

        def _var_null(w):
            from sbr.features.m02_dist import _starts
            idx = _starts(nh, w)
            if idx is None:
                return None
            m1 = (chu[idx + w] - chu[idx]) / w
            m2 = (chu2[idx + w] - chu2[idx]) / w
            return np.sort(m2 - m1 * m1)

        self.var_fix = _var_null(W_FIX[1])
        self.var_exp = {g: _var_null(g) for g in G_EXP}

        # ---- AR-residual PIT reference ----------------------------------
        p_ar = len(hp.ar_coef)
        self._has_res = ("res_mean" in ctx.hist_tr) and (nh - p_ar > 50)
        if self._has_res:
            rh = np.asarray(ctx.hist_tr["res_mean"], dtype=np.float64)[p_ar:]
            self._sref = np.sort(rh)
            urh = _pit_against(self._sref, rh)
        else:
            self._sref = None
            urh = uh

        # ---- CUSUM nulls (u stream and residual stream) ------------------
        self.cusum_nulls = {
            "u": {g: _cusum_null(uh, g) for g in G_EXP},
            "r": {g: _cusum_null(urh, g) for g in G_EXP},
        }

        # ---- |x - med| PIT stream ---------------------------------------
        av_h = np.abs(hist - hp.med)
        self._svh = np.sort(av_h)
        uah = _pit_against(self._svh, av_h)
        sab = _Stream(uah, _EMPTY, B_COARSE)
        self.nab = {w: sab.null(w) for w in lens_alt}
        self.cah_null = _roll_null(_cum(uah), len(uah), 128)

        # ---- AR-residual occupancy stream -------------------------------
        srs = _Stream(urh, _EMPTY, B_COARSE)
        self.nrs = {w: srs.null(w) for w in lens_alt}

        # ---- log-bucket map for every expanding family -------------------
        self._logL = np.log(np.arange(1, self.cap + 1, dtype=np.float64))
        self._jcache: dict[tuple, list] = {}

        # ---- tabulate every null once ------------------------------------
        tc: dict = {}
        self.n20 = {w: _wrap_d(d, tc) for w, d in self.n20.items()}
        self.n10 = {w: _wrap_d(d, tc) for w, d in self.n10.items()}
        self.nab = {w: _wrap_d(d, tc) for w, d in self.nab.items()}
        self.nrs = {w: _wrap_d(d, tc) for w, d in self.nrs.items()}
        self.tail_fix = {k: _wrap(v, tc) for k, v in self.tail_fix.items()}
        self.tail_exp = {k: {g: _wrap(v, tc) for g, v in d.items()}
                         for k, d in self.tail_exp.items()}
        self.var_fix = _wrap(self.var_fix, tc)
        self.var_exp = {g: _wrap(v, tc) for g, v in self.var_exp.items()}
        self.cusum_nulls = {tag: {g: _wrap(v, tc) for g, v in d.items()}
                            for tag, d in self.cusum_nulls.items()}
        self.cah_null = _wrap(self.cah_null, tc)

        # ---- PIT reference tables ----------------------------------------
        self._pt_ab = _pit_table(self._svh)
        self._pt_rs = _pit_table(self._sref) if self._has_res else None

        # ---- resolve every calibrated column to a table reference ---------
        n20, n10, nab, nrs = self.n20, self.n10, self.nab, self.nrs

        def fx(d, key):
            return None if d is None or key not in d else d[key]

        F = []
        for k in ("chi2", "js", "ks", "ene"):
            F.append(fx(n20[32], k))                                # 0-3
        for k in ("chi2", "js", "ks", "cvm", "w1", "ene"):
            F.append(fx(n20[128], k))                               # 4-9
        F.append(fx(n10[32], "ent"))                                # 10
        F.append(fx(n10[32], "maxdev"))                             # 11
        F.append(fx(n10[128], "ent"))                               # 12
        F.append(fx(n10[128], "maxdev"))                            # 13
        F.append(fx(n10[128], "hell"))                              # 14
        for k in ("q50", "qiqr", "q9010"):
            F.append(fx(n20[128], k))                               # 15-17
        for k in _TAIL_KEYS:
            F.append(self.tail_fix[k])                              # 18-20
        F.append(self.var_fix)                                      # 21
        F.append(fx(nab[128], "chi2"))                              # 22
        F.append(fx(nab[128], "ks"))                                # 23
        F.append(self.cah_null)                                     # 24
        F.append(fx(nrs[128], "chi2"))                              # 25
        F.append(fx(nrs[128], "ks"))                                # 26
        F.append(fx(nrs[128], "js"))                                # 27
        self._F = tuple(F)

        def xr(d, key=None):
            """(jmap list, NT tuple) for one expanding family/key."""
            if key is None:
                gs = tuple(sorted(g for g in G_EXP if d[g] is not None))
                nts = tuple(d[g] for g in gs)
            else:
                gs = tuple(sorted(g for g in G_EXP
                                  if d[g] is not None and key in d[g]))
                nts = tuple(d[g][key] for g in gs)
            if not gs:
                Xgs.append(())
                return (None, ())
            Xgs.append(gs)
            return (self._jmap(gs), nts)

        ne20 = {g: n20[g] for g in G_EXP}
        ne10 = {g: n10[g] for g in G_EXP}
        neab = {g: nab[g] for g in G_EXP}
        ners = {g: nrs[g] for g in G_EXP}
        X = []
        Xgs: list = []
        for k in ("chi2", "js", "ks", "ene"):
            X.append(xr(ne20, k))                                   # 0-3
        for k in ("ent", "maxdev", "tv"):
            X.append(xr(ne10, k))                                   # 4-6
        for k in ("q50", "qiqr"):
            X.append(xr(ne20, k))                                   # 7-8
        for k in ("asym05", "asym01"):
            X.append(xr(self.tail_exp[k]))                          # 9-10
        X.append(xr(self.var_exp))                                  # 11
        X.append(xr(self.cusum_nulls["u"]))                         # 12
        X.append(xr(self.cusum_nulls["r"]))                         # 13
        for k in ("chi2", "js"):
            X.append(xr(neab, k))                                   # 14-15
        for k in ("chi2", "js"):
            X.append(xr(ners, k))                                   # 16-17
        self._X = tuple(X)
        self._Xgs = tuple(Xgs)

        self._reset_online()
        self._fitted = True

    # --------------------------------------------------------------- online
    def _reset_online(self):
        cap = self.cap
        z1 = lambda b: np.zeros((cap + 1, b), dtype=np.float64)  # noqa: E731
        self.C20 = z1(B_FINE)
        self.C10 = z1(B_COARSE)
        self.Cab = z1(B_COARSE)
        self.Crs = z1(B_COARSE)
        self.cu = np.zeros(cap + 1, dtype=np.float64)
        self.cu2 = np.zeros(cap + 1, dtype=np.float64)
        self.c_a05 = np.zeros(cap + 1, dtype=np.float64)
        self.c_a01 = np.zeros(cap + 1, dtype=np.float64)
        self.c_ctr = np.zeros(cap + 1, dtype=np.float64)
        self.cao = np.zeros(cap + 1, dtype=np.float64)
        self._Su = self._Sr = self._Mu = self._Mr = 0.0
        self._olist = [0.0] * NCOL
        # scratch matrices for the two batched `_stats` calls
        # 20-bin rows: [w=32, w=128, expanding]
        self._M20 = np.zeros((3, B_FINE), dtype=np.float64)
        self._wl20 = np.array([32.0, 128.0, 1.0], dtype=np.float64)
        self._mu20 = np.zeros(3, dtype=np.float64)
        self._mu220 = np.zeros(3, dtype=np.float64)
        # 10-bin rows: [u w32, u w128, |x-med| w128, resid w128,
        #               u exp, |x-med| exp, resid exp]
        self._M10 = np.zeros((7, B_COARSE), dtype=np.float64)
        self._wl10 = np.array([32.0, 128.0, 128.0, 128.0, 1.0, 1.0, 1.0],
                              dtype=np.float64)

    def _grow(self):
        old = self.cap
        self.cap = old * 2
        def g2(a, b):
            n = np.zeros((self.cap + 1, b), dtype=np.float64)
            n[:old + 1] = a
            return n
        def g1(a):
            n = np.zeros(self.cap + 1, dtype=np.float64)
            n[:old + 1] = a
            return n
        self.C20 = g2(self.C20, B_FINE)
        self.C10 = g2(self.C10, B_COARSE)
        self.Cab = g2(self.Cab, B_COARSE)
        self.Crs = g2(self.Crs, B_COARSE)
        self.cu = g1(self.cu)
        self.cu2 = g1(self.cu2)
        self.c_a05 = g1(self.c_a05)
        self.c_a01 = g1(self.c_a01)
        self.c_ctr = g1(self.c_ctr)
        self.cao = g1(self.cao)
        self._logL = np.log(np.arange(1, self.cap + 1, dtype=np.float64))
        self._jcache = {}
        # the log-bucket maps are indexed by t, so they must be re-tabulated
        self._X = tuple((self._jmap(gs) if gs else None, nts)
                        for gs, (jm, nts) in zip(self._Xgs, self._X))

    # ----------------------------------------------------------------- util
    def _jmap(self, gs: tuple) -> list:
        j = self._jcache.get(gs)
        if j is None:
            ga = np.asarray(gs, dtype=np.float64)
            j = np.argmin(np.abs(self._logL[:, None] - np.log(ga)[None, :]),
                          axis=1).tolist()
            self._jcache[gs] = j
        return j

    # ------------------------------------------------------------- quantiles
    @staticmethod
    def _quant(Crow, w):
        """The `quant` block of ``_stats`` for ONE row, in scalar arithmetic.

        ``_stats``'s quantile block is a *gather* (``take_along_axis``) plus a
        handful of scalar expressions per level, which costs ~150 us/obs of
        pure numpy dispatch for 20 bins.  Every step below is the identical
        float64 expression on the identical operands:

          * ``(F < p).sum(1)`` == ``bisect_left(F, p)`` because ``F`` is a
            cumulative sum of non-negative values and is therefore exactly
            non-decreasing;
          * ``take_along_axis`` is a pure gather -- no arithmetic, no rounding;
          * ``np.maximum(d, 1e-12)`` on a non-NaN ``d`` is ``d if d > 1e-12``;
          * ``k + (p - Fkm)/d`` then ``/B`` are the same two IEEE operations,
            in the same order, on the same values (``k`` is an exact integer).

        Pinned bitwise against ``_stats(..., quant=True)`` by
        ``test_quant_block_bitwise`` over random count matrices.
        """
        F = np.cumsum(Crow / w).tolist()
        B = len(F)
        Bm1 = B - 1
        Q = []
        for p in QLEV:
            k = _bl(F, p)
            if k > Bm1:
                k = Bm1
            Fk = F[k]
            Fkm = F[k - 1] if k > 0 else 0.0
            d = Fk - Fkm
            if d < 1e-12:
                d = 1e-12
            Q.append((k + (p - Fkm) / d) / B)
        return Q[2] - 0.5, (Q[3] - Q[1]) - 0.5, (Q[4] - Q[0]) - 0.8

    # ----------------------------------------------------------------- step
    def step(self, ctx) -> np.ndarray:
        t = ctx.t
        if t + 1 > self.cap:
            self._grow()
        n = t + 1
        L = float(n)
        t1 = t + 1

        u = float(ctx.tr["u"][t])
        x = float(ctx._raw[t])

        # ---------------- update the online cumulative structures --------
        b20 = int(u * B_FINE)
        if b20 > 19:
            b20 = 19
        b10 = int(u * B_COARSE)
        if b10 > 9:
            b10 = 9
        C20, C10, Cab, Crs = self.C20, self.C10, self.Cab, self.Crs
        cu, cu2 = self.cu, self.cu2
        C20[t1] = C20[t]
        C20[t1, b20] += 1.0
        C10[t1] = C10[t]
        C10[t1, b10] += 1.0
        cu[t1] = cu[t] + u
        cu2[t1] = cu2[t] + u * u

        c = self.c_a05
        c[t1] = c[t] + (float(u > 0.95) - float(u < 0.05))
        c = self.c_a01
        c[t1] = c[t] + (float(u > 0.99) - float(u < 0.01))
        c = self.c_ctr
        c[t1] = c[t] + float(0.25 < u < 0.75)

        # |x - med| PIT
        ua = _pit1(self._pt_ab, abs(x - self.hp.med))
        cao = self.cao
        cao[t1] = cao[t] + ua
        bab = int(ua * B_COARSE)
        if bab > 9:
            bab = 9
        Cab[t1] = Cab[t]
        Cab[t1, bab] += 1.0

        # AR-residual PIT
        if self._has_res:
            ur = _pit1(self._pt_rs, float(ctx.tr["res_mean"][t]))
        else:
            ur = u
        brs = int(ur * B_COARSE)
        if brs > 9:
            brs = 9
        Crs[t1] = Crs[t]
        Crs[t1, brs] += 1.0

        # CUSUM state (sequential cumsum + running max -> exact)
        Su = self._Su + (u - 0.5)
        self._Su = Su
        au = -Su if Su < 0.0 else Su
        if au > self._Mu:
            self._Mu = au
        Sr = self._Sr + (ur - 0.5)
        self._Sr = Sr
        ar = -Sr if Sr < 0.0 else Sr
        if ar > self._Mr:
            self._Mr = ar
        sq12 = _sqrt(L / 12.0)
        cu_u = self._Mu / sq12
        cu_r = self._Mr / sq12

        # ---------------- statistics ------------------------------------
        # `_stats` is strictly row-wise (every expression is elementwise plus a
        # reduction along the contiguous axis 1), so stacking several windows
        # into one call yields, row by row, exactly the floats the batch module
        # produces -- and costs one numpy dispatch instead of ten.
        # Rows whose window is not yet full are computed from a zero count row
        # and simply never read.
        M20, wl20, mu20, mu220 = self._M20, self._wl20, self._mu20, self._mu220
        top = C20[t1]
        ge32 = n >= 32
        ge128 = n >= 128
        if ge32:
            np.subtract(top, C20[t1 - 32], out=M20[0])
            mu20[0] = (cu[t1] - cu[t1 - 32]) / 32
            mu220[0] = (cu2[t1] - cu2[t1 - 32]) / 32
        else:
            M20[0] = 0.0; mu20[0] = 0.0; mu220[0] = 0.0
        if ge128:
            np.subtract(top, C20[t1 - 128], out=M20[1])
            mu20[1] = (cu[t1] - cu[t1 - 128]) / 128
            mu220[1] = (cu2[t1] - cu2[t1 - 128]) / 128
        else:
            M20[1] = 0.0; mu20[1] = 0.0; mu220[1] = 0.0
        M20[2] = top
        wl20[2] = L
        mu20[2] = cu[t1] / L
        mu220[2] = cu2[t1] / L
        s20 = _stats(M20, wl20, B_FINE, mu20, mu220, False)

        M10, wl10 = self._M10, self._wl10
        if ge32:
            np.subtract(C10[t1], C10[t1 - 32], out=M10[0])
        else:
            M10[0] = 0.0
        if ge128:
            np.subtract(C10[t1], C10[t1 - 128], out=M10[1])
            np.subtract(Cab[t1], Cab[t1 - 128], out=M10[2])
            np.subtract(Crs[t1], Crs[t1 - 128], out=M10[3])
        else:
            M10[1] = 0.0; M10[2] = 0.0; M10[3] = 0.0
        M10[4] = C10[t1]
        M10[5] = Cab[t1]
        M10[6] = Crs[t1]
        wl10[4] = wl10[5] = wl10[6] = L
        s10 = _stats(M10, wl10, B_COARSE, None, None, False)

        # the quantile block of `_stats`, for the two rows that need it
        qe = self._quant(M20[2], L)
        qf = self._quant(M20[1], 128.0) if ge128 else (nan, nan, nan)

        # ---- pull every needed statistic out as python floats ------------
        chi2 = s20["chi2"].tolist(); js = s20["js"].tolist()
        ks = s20["ks"].tolist(); ene = s20["ene"].tolist()
        cvm = s20["cvm"].tolist(); w1 = s20["w1"].tolist()
        ent = s10["ent"].tolist(); mxd = s10["maxdev"].tolist()
        amax = s10["amax"].tolist(); hell = s10["hell"].tolist()
        chi2b = s10["chi2"].tolist(); ksb = s10["ks"].tolist()
        tv = s10["tv"].tolist(); jsb = s10["js"].tolist()

        F = self._F
        X = self._X
        o = self._olist

        # ---- divergence family ------------------------------------------
        if ge32:
            nt = F[0]; o[0] = nan if nt is None else nt.up_of(chi2[0])
            nt = F[1]; o[1] = nan if nt is None else nt.up_of(js[0])
            nt = F[2]; o[2] = nan if nt is None else nt.up_of(ks[0])
            nt = F[3]; o[3] = nan if nt is None else nt.up_of(ene[0])
        else:
            o[0] = o[1] = o[2] = o[3] = nan
        if ge128:
            nt = F[4]; o[4] = nan if nt is None else nt.up_of(chi2[1])
            nt = F[5]; o[5] = nan if nt is None else nt.up_of(js[1])
            nt = F[6]; o[6] = nan if nt is None else nt.up_of(ks[1])
            nt = F[7]; o[7] = nan if nt is None else nt.up_of(cvm[1])
            nt = F[8]; o[8] = nan if nt is None else nt.up_of(w1[1])
            nt = F[9]; o[9] = nan if nt is None else nt.up_of(ene[1])
        else:
            o[4] = o[5] = o[6] = o[7] = o[8] = o[9] = nan
        for sl, val, xi in ((10, chi2[2], 0), (11, js[2], 1),
                            (12, ks[2], 2), (13, ene[2], 3)):
            jm, nts = X[xi]
            o[sl] = nan if jm is None else nts[jm[t]].up_of(val)
        o[14] = js[1] if ge128 else nan
        o[15] = js[2]

        # ---- occupancy-shape family --------------------------------------
        if ge32:
            nt = F[10]; o[16] = nan if nt is None else nt.sg_of(ent[0])
            nt = F[11]; o[17] = nan if nt is None else nt.up_of(mxd[0])
        else:
            o[16] = o[17] = nan
        if ge128:
            nt = F[12]; o[18] = nan if nt is None else nt.sg_of(ent[1])
            nt = F[13]; o[19] = nan if nt is None else nt.up_of(mxd[1])
            o[20] = amax[1]
            nt = F[14]; o[21] = nan if nt is None else nt.up_of(hell[1])
        else:
            o[18] = o[19] = o[20] = o[21] = nan
        jm, nts = X[4]; o[22] = nan if jm is None else nts[jm[t]].sg_of(ent[4])
        jm, nts = X[5]; o[23] = nan if jm is None else nts[jm[t]].up_of(mxd[4])
        jm, nts = X[6]; o[24] = nan if jm is None else nts[jm[t]].up_of(tv[4])
        o[25] = amax[4]

        # ---- quantile-deviation family ------------------------------------
        o[26], o[27], o[28] = qf
        o[29], o[30], o[31] = qe
        if ge128:
            nt = F[15]; o[32] = nan if nt is None else nt.sg_of(qf[0])
            nt = F[16]; o[33] = nan if nt is None else nt.sg_of(qf[1])
            nt = F[17]; o[34] = nan if nt is None else nt.sg_of(qf[2])
        else:
            o[32] = o[33] = o[34] = nan
        jm, nts = X[7]; o[35] = nan if jm is None else nts[jm[t]].sg_of(qe[0])
        jm, nts = X[8]; o[36] = nan if jm is None else nts[jm[t]].sg_of(qe[1])

        # ---- tail family --------------------------------------------------
        if n >= W_TAIL:
            c = self.c_a05; nt = F[18]
            o[37] = nan if nt is None else nt.sg_of((c[t1] - c[t1 - W_TAIL]) / W_TAIL)
            c = self.c_a01; nt = F[19]
            o[38] = nan if nt is None else nt.sg_of((c[t1] - c[t1 - W_TAIL]) / W_TAIL)
            c = self.c_ctr; nt = F[20]
            o[39] = nan if nt is None else nt.sg_of((c[t1] - c[t1 - W_TAIL]) / W_TAIL)
        else:
            o[37] = o[38] = o[39] = nan
        ev05 = self.c_a05[t1] / L
        jm, nts = X[9]; o[40] = nan if jm is None else nts[jm[t]].sg_of(ev05)
        jm, nts = X[10]
        o[41] = nan if jm is None else nts[jm[t]].sg_of(self.c_a01[t1] / L)
        o[42] = ev05

        # ---- rank family ---------------------------------------------------
        nt = F[21]
        if nt is None or not ge128:
            o[43] = nan
        else:
            m1 = (cu[t1] - cu[t1 - 128]) / 128
            m2 = (cu2[t1] - cu2[t1 - 128]) / 128
            o[43] = nt.sg_of(m2 - m1 * m1)
        jm, nts = X[11]
        if jm is None:
            o[44] = nan
        else:
            o[44] = nts[jm[t]].sg_of(cu2[t1] / L - (cu[t1] / L) ** 2)

        # ---- CUSUM ----------------------------------------------------------
        o[45] = 20.0 if cu_u > 20.0 else (0.0 if cu_u < 0.0 else cu_u)
        jm, nts = X[12]; o[46] = nan if jm is None else nts[jm[t]].up_of(cu_u)
        o[47] = 20.0 if cu_r > 20.0 else (0.0 if cu_r < 0.0 else cu_r)
        jm, nts = X[13]; o[48] = nan if jm is None else nts[jm[t]].up_of(cu_r)

        # ---- |x - med| stream ------------------------------------------------
        if ge128:
            nt = F[22]; o[49] = nan if nt is None else nt.up_of(chi2b[2])
            nt = F[23]; o[50] = nan if nt is None else nt.up_of(ksb[2])
        else:
            o[49] = o[50] = nan
        jm, nts = X[14]; o[51] = nan if jm is None else nts[jm[t]].up_of(chi2b[5])
        jm, nts = X[15]; o[52] = nan if jm is None else nts[jm[t]].up_of(jsb[5])
        nt = F[24]
        if nt is None or not ge128:
            o[53] = nan
        else:
            o[53] = nt.sg_of((cao[t1] - cao[t1 - 128]) / 128.0)

        # ---- AR-residual stream ----------------------------------------------
        if ge128:
            nt = F[25]; o[54] = nan if nt is None else nt.up_of(chi2b[3])
            nt = F[26]; o[55] = nan if nt is None else nt.up_of(ksb[3])
            nt = F[27]; o[56] = nan if nt is None else nt.up_of(jsb[3])
        else:
            o[54] = o[55] = o[56] = nan
        jm, nts = X[16]; o[57] = nan if jm is None else nts[jm[t]].up_of(chi2b[6])
        jm, nts = X[17]; o[58] = nan if jm is None else nts[jm[t]].up_of(jsb[6])

        out = np.array(o, dtype=np.float32).astype(np.float64)
        bad = ~np.isfinite(out)
        if bad.any():
            out[bad] = np.nan
        return out
