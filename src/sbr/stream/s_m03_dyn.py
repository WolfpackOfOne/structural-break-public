"""Incremental port of the ``m03_dyn`` batch feature module.

What the batch module actually does
-----------------------------------
``sbr.features.m03_dyn`` builds, for every online index ``t``, 60 columns of
temporal-dynamics evidence: windowed ACFs (raw / AR-residual / |residual| /
squared residual / sign), variance ratios, OLS drift, Mann-Kendall style
monotonicity, a six-band windowed DFT, Haar detail energies at four dyadic
scales, and order-3 permutation entropy -- each evaluated on an *adaptive*
window (expanding / trailing-half / trailing-quarter), and most of them
additionally z-scored against the distribution of the SAME statistic over
length-matched windows of the break-free history.

Crucially the batch author already wrote it in streaming form: **every**
statistic is a closed-form function of trailing means of 51 per-point
transforms (``_local``), read out of cumulative sums by ``_R``.  There is no
FFT, no per-step regression, no sort, no window rescan.  The port is therefore
O(1) per observation -- not even a bounded O(w).

How bitwise parity is obtained
------------------------------
1. The *historical* half (``_local`` on the history, ``_cumz``, ``_Cal`` and its
   per-grid null medians/IQRs) is the batch code called on the batch input,
   once, in :meth:`fit_historical`.  Identical input, identical call.  The null
   calibration of every statistic is driven by the **batch** statistic function,
   never by the scalar re-expression used per step.
2. The *online* per-point transforms are recomputed here one point at a time.
   They are elementwise except for three details, each handled explicitly:
     * lag products / differences / ordinal codes look back up to 20 points,
       which is why the batch module prepends ``pad = min(40, n_hist)`` points
       of history in front of the online segment.  We keep exactly that padded
       buffer, so the early online points are warmed from history the same way
       and the ``i >= k`` zero-fill guards fire on the same indices.
     * the Haar details are differences of a cumulative sum taken over the
       *whole* padded array (``_cumz(zp)``).  Differences of cumsums are NOT
       invariant to where the cumsum started, so we carry that same running sum
       from the same origin, extending it by ``cz[i+1] = cz[i] + zp[i]`` --
       exactly what ``np.cumsum`` does (sequential accumulation).
     * the DFT quadratures need ``cos(2*pi*f*j)`` at the absolute online index
       ``j = t``; we form the angle with the same expression on a length-6
       vector.  ``np.cos``/``np.sin``/``np.log``/``np.interp`` are strictly
       elementwise in numpy -- fuzz-verified, not assumed.
3. The *statistics* are re-expressed in scalar form (see below), and every one
   of those re-expressions is pinned by a differential fuzz against the batch
   expression in ``tests/test_stream_parity_m03_dyn.py``.
4. The batch module ends with ``A = ...astype(np.float32)`` and only then
   ``A[~np.isfinite(A)] = np.nan``, so the finiteness scrub sees float32 values
   (a float64 that overflows float32 becomes NaN).  We reproduce that: the mask
   is computed on the float32 cast, the returned row stays float64.

Why a scalar re-expression is safe here
---------------------------------------
Batch evaluates each statistic on length-``n`` vectors; we evaluate it at one
index.  For an expression that is **elementwise** -- no reduction whose
summation order numpy chooses -- the scalar form performs the *same IEEE
operations on the same operands in the same order*, so it is not an
approximation, it is the same arithmetic.  That covers ``_acf``, ``_ratio``,
``_drift``, ``_mk``, ``_hj_mob``, ``_hj_comp`` and the Haar read-outs.  The two
statistics that DO reduce (``_spec_p``/``_pe``: ``sum(axis=0)`` over the
6-band stack, and the entropy sum) keep the reduction **in numpy on an
identically-shaped ``(6, 1)`` block**; only the construction of that block
changes, from ``np.stack`` of six length-1 arrays to a reshaped contiguous
slice, which is the same memory layout and therefore the same reduction path.
Nothing hand-rolls a floating-point sum.

Transcendentals are never moved off numpy: every ``log`` here is ``np.log``.

The deviations from a literal transcription of ``build`` -- all of them
value-preserving, all of them covered by the bitwise parity tests and by the
per-expression fuzz:
  * ``_R`` is replaced by :class:`_R1`.  All 51 transforms share one window
    length per multiplier, so ONE vectorised cumsum difference over the name
    axis produces every rolling mean at once; scalars are read out of it with
    ``.item``, and the four multi-band statistics read contiguous slices of it.
  * ``_Cal.z``'s ``mu(W)``/``sd(W)`` interpolation is tabulated over the integer
    window range at fit time (``_loglin`` is elementwise, so a length-1 query
    and a length-cap query agree bitwise), and the final affine + clip is done
    in scalar arithmetic -- ``np.clip(x, -c, c)`` is ``min(max(x, -c), c)``
    including the NaN and infinity cases.
  * ``np.where(W >= minW, v, nan)`` becomes a branch: below the minimum window
    the statistic is not evaluated at all and the row entry is NaN, which is
    what the ``np.where`` produced.
  * ``_spec_p`` (5 read-outs) and the Haar log-energies (4 read-outs) are
    memoised on the accessor for the duration of one step; both are pure
    functions of the accessor.

Cost per observation: 51 scalar transform updates, one 51-wide vector add, up
to five 51-wide cumsum differences, 39 scalar statistic evaluations and 30
table lookups.  Independent of ``t`` and of the window length.
"""
from __future__ import annotations

import numpy as np

from sbr.features.m03_dyn import (
    ALAGS,
    CLIP,
    ELAGS,
    FREQS,
    HSCALES,
    NF,
    PAD,
    PCODES,
    ZLAGS,
    _acf,
    _Cal,
    _cumz,
    _drift,
    _hj_comp,
    _hj_mob,
    _hw_hurst,
    _hw_r,
    _local,
    _loglin,
    _mk,
    _pe,
    _ratio,
    _sp_cen,
    _sp_ent,
    _sp_high,
    _sp_low,
)

_TWO_PI = 2.0 * np.pi
_INF = float("inf")
#: dyadic-scale normalisers, exactly the batch ``np.sqrt(2.0 * s)``
_SQ2S = tuple(float(np.sqrt(2.0 * s)) for s in HSCALES)
#: constants the batch recomputes inside the hot loop; identical values
_LOG_NF = np.log(NF)
_LOG_6 = np.log(6.0)
_LOG_2 = np.log(2.0)
_BANDS = np.arange(NF).reshape(-1, 1)      # == np.arange(NF).reshape(-1, *[1])


def _sign(x: float) -> float:
    """``float(np.sign(x))`` for a scalar, including the NaN and -0.0 cases."""
    if x != x:
        return x
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


def _tnames() -> tuple[str, ...]:
    """Transform names in the order :meth:`StreamM03Dyn._absorb` writes them.

    The four multi-band families (``c*``, ``s*``, ``h*``, ``p*``) are kept
    contiguous so the spectral / wavelet / permutation-entropy read-outs can
    take a slice of the rolling-mean vector instead of gathering names.
    Checked against the batch ``_local`` output at fit time, so a change to the
    batch transform set is caught immediately instead of silently drifting.
    """
    n = ["x", "x2"]
    n += [f"xl{k}" for k in ZLAGS]
    n += ["e", "e2", "e4"]
    n += [f"el{k}" for k in ELAGS]
    n += ["ae"]
    n += [f"ael{k}" for k in ALAGS]
    n += ["e2l1", "sg", "sg2", "sgl1", "tx", "d1", "d1_2", "d2", "d2_2", "mk1", "mk8"]
    n += [f"c{q}" for q in range(NF)]
    n += [f"s{q}" for q in range(NF)]
    n += [f"h{s}" for s in HSCALES]
    n += [f"p{q}" for q in range(len(PCODES))]
    return tuple(n)


class _R1:
    """``m03_dyn._R`` evaluated at ONE time index.

    ``_R(name, mult)`` is ``(c[t+1] - c[t+1-w]) / w`` with
    ``w = min(max(rint(mult * W), 1), t + 1)`` -- and ``w`` does not depend on
    ``name``.  So one vectorised difference of two rows of the (time x name)
    cumsum matrix yields every transform's rolling mean for that multiplier;
    ``s(name)`` reads one element out of it and ``band(base, k)`` a contiguous
    ``k``-slice.  ``specp`` / ``hwlog`` memoise the two derived blocks that the
    batch module rebuilds for each of its read-outs.
    """

    __slots__ = ("C", "W", "t1", "ix", "vecs", "specp", "hwlog", "ic", "isn",
                 "ih", "ip")

    def __init__(self, C, W: int, t1: int, ix: dict, bases: tuple):
        self.C = C
        self.W = W
        self.t1 = t1
        self.ix = ix
        self.vecs: dict = {}
        self.specp = None
        self.hwlog = None
        self.ic, self.isn, self.ih, self.ip = bases

    def vec(self, mult: float = 1.0):
        V = self.vecs.get(mult)
        if V is None:
            w = int(round(mult * self.W))      # round() is round-half-even, as np.rint
            if w < 1:
                w = 1
            t1 = self.t1
            if w > t1:
                w = t1
            V = (self.C[t1] - self.C[t1 - w]) / w
            self.vecs[mult] = V
        return V

    def s(self, name: str, mult: float = 1.0) -> float:
        """The batch ``R(name, mult)[t]`` as a python float."""
        V = self.vecs.get(mult)
        if V is None:
            V = self.vec(mult)
        return V.item(self.ix[name])

    def band(self, base: int, k: int):
        """Contiguous ``k`` rolling means starting at column ``base``, mult 1."""
        V = self.vecs.get(1.0)
        if V is None:
            V = self.vec(1.0)
        return V[base:base + k]

    # --- API-compatibility with the batch _R (unused on the hot path) -------
    def __call__(self, name, mult=1.0):
        i = self.ix[name]
        return self.vec(mult)[i:i + 1]


# ---------------------------------------------------------------------------
# Scalar re-expressions of the batch statistics.  Each is the batch body with
# length-1 arrays replaced by floats -- same operations, same order -- and each
# is pinned by a differential fuzz against the batch original (see
# ``tests/test_stream_parity_m03_dyn.py::test_fuzz_*``).  Signature is
# ``(R, Wi, t) -> float`` where ``Wi`` is the integer window (batch's
# ``np.rint(W)``) and ``t`` the current online index (batch's ``tend``).
# ---------------------------------------------------------------------------
def _acf_s(num, den_sq, mean_name, lag_key):
    """== ``m03_dyn._acf``.  ``np.maximum``/``np.clip`` become branches that
    keep NaN (a NaN comparison is False, so NaN falls through unchanged)."""
    def f(R, Wi, t):
        m = R.s(mean_name)
        mm = m * m
        v = R.s(den_sq) - mm
        c = R.s(lag_key) - mm
        if v < 1e-8:
            v = 1e-8
        r = c / v
        if r < -2.0:
            return -2.0
        if r > 2.0:
            return 2.0
        return r
    return f


def _ratio_s(a, ma, b, mb):
    """== ``m03_dyn._ratio``.  The log stays on numpy."""
    def f(R, Wi, t):
        p = R.s(a, ma)
        q = R.s(b, mb)
        if p < 1e-10:
            p = 1e-10
        if q < 1e-10:
            q = 1e-10
        return np.log(p / q)
    return f


def _drift_s(R, Wi, t):
    """== ``m03_dyn._drift`` (``np.rint(W).astype(int64)`` is ``Wi``)."""
    w = Wi if Wi > 2 else 2
    s0 = R.s("x") * w
    s1 = R.s("tx") * w
    jbar = t - (w - 1) / 2.0
    den = w * (w * w - 1.0) / 12.0
    if den < 1e-9:
        den = 1e-9
    return (s1 - jbar * s0) / den * w


def _mk_s(name):
    """== ``m03_dyn._mk``."""
    def f(R, Wi, t):
        return R.s(name)
    return f


def _specp_s(R):
    """== ``m03_dyn._spec_p``, memoised, on a ``(6, 1)`` block.

    The band powers are elementwise; only the *construction* of the stacked
    block changes (contiguous slice + reshape instead of ``np.stack`` of six
    length-1 arrays -- same dtype, same C-contiguous ``(6, 1)`` layout).  The
    ``sum(axis=0)`` reduction is left to numpy on that identical block.
    """
    p = R.specp
    if p is None:
        a = R.band(R.ic, NF)
        b = R.band(R.isn, NF)
        P = (a * a + b * b).reshape(NF, 1)
        p = P / np.maximum(P.sum(axis=0), 1e-14)
        R.specp = p
    return p


def _sp_low_s(R, Wi, t):
    p = _specp_s(R)
    return p.item(4) + p.item(5)


def _sp_high_s(R, Wi, t):
    p = _specp_s(R)
    return p.item(0) + p.item(1)


def _sp_ent_s(R, Wi, t):
    p = _specp_s(R)
    return (-(p * np.log(np.maximum(p, 1e-12))).sum(axis=0) / _LOG_NF).item(0)


def _sp_cen_s(R, Wi, t):
    p = _specp_s(R)
    return (p * _BANDS).sum(axis=0).item(0)


def _sp_dom_s(R, Wi, t):
    return float(np.argmax(_specp_s(R), axis=0).item(0))


def _hwlog_s(R):
    """== ``m03_dyn._hw_logE``, memoised, as one length-4 vector."""
    L = R.hwlog
    if L is None:
        L = np.log(np.maximum(R.band(R.ih, len(HSCALES)), 1e-12))
        R.hwlog = L
    return L


def _hw_r_s(i, j):
    """== ``m03_dyn._hw_r``."""
    def f(R, Wi, t):
        L = _hwlog_s(R)
        return L[i] - L[j]
    return f


def _hw_hurst_s(R, Wi, t):
    """== ``m03_dyn._hw_hurst``."""
    L = _hwlog_s(R)
    return (-1.5 * L[0] - 0.5 * L[1] + 0.5 * L[2] + 1.5 * L[3]) / 5.0 / _LOG_2


def _pe_s(R, Wi, t):
    """== ``m03_dyn._pe``; the two reductions stay in numpy on a (6, 1) block."""
    q = R.band(R.ip, 6).reshape(6, 1)
    q = q / np.maximum(q.sum(axis=0), 1e-12)
    return (-(q * np.log(np.maximum(q, 1e-12))).sum(axis=0) / _LOG_6).item(0)


def _hj_mob_s(R, Wi, t):
    """== ``m03_dyn._hj_mob``."""
    m = R.s("x")
    vx = R.s("x2") - m * m
    d = R.s("d1")
    vd = R.s("d1_2") - d * d
    if vx < 1e-10:
        vx = 1e-10
    if vd < 1e-10:
        vd = 1e-10
    return 0.5 * np.log(vd / vx)


def _hj_comp_s(R, Wi, t):
    """== ``m03_dyn._hj_comp``."""
    m = R.s("x")
    vx = R.s("x2") - m * m
    d = R.s("d1")
    vd = R.s("d1_2") - d * d
    d2 = R.s("d2")
    vdd = R.s("d2_2") - d2 * d2
    if vx < 1e-10:
        vx = 1e-10
    if vd < 1e-10:
        vd = 1e-10
    if vdd < 1e-10:
        vdd = 1e-10
    return 0.5 * (np.log(vdd / vd) - np.log(vd / vx))


def _sp_dom_batch(R, W, tend):
    """The closure defined inside ``m03_dyn.build`` (used for fuzzing only)."""
    from sbr.features.m03_dyn import _spec_p
    return np.argmax(_spec_p(R), axis=0).astype(np.float64)


#: scheme index -> (accessor, integer window) in :meth:`StreamM03Dyn.step`
_EXP, _HALF, _QUART = 0, 1, 2


def _plan() -> list[tuple]:
    """The batch ``build`` emission sequence, as data.

    Each ``"b"`` op carries BOTH the batch statistic (used only at fit time, to
    calibrate the historical null exactly as batch does) and its scalar
    re-expression (used per step)::

        ("b", key, fn_batch, fn_scalar, scheme, minW, name, want_raw, want_z)

    plus ``("sub", name, a, b)`` for the dslope pair and ``("splr",)`` /
    ``("spdom",)`` for the two bespoke spectral columns.
    """
    P: list[tuple] = []
    # ---- A. dependence ---------------------------------------------------
    for k in ZLAGS:
        P.append(("b", f"acf_x_l{k}", _acf("x", "x2", "x", f"xl{k}"),
                  _acf_s("x", "x2", "x", f"xl{k}"), _EXP,
                  max(12, 2 * k), f"az_x_exp_l{k}", False, True))
    for k in ELAGS:
        P.append(("b", f"acf_e_l{k}", _acf("e", "e2", "e", f"el{k}"),
                  _acf_s("e", "e2", "e", f"el{k}"), _EXP,
                  max(12, 2 * k), f"az_e_exp_l{k}", False, True))
    for k in ALAGS:
        P.append(("b", f"acf_ae_l{k}", _acf("ae", "e2", "ae", f"ael{k}"),
                  _acf_s("ae", "e2", "ae", f"ael{k}"), _EXP,
                  max(12, 2 * k), f"az_ae_exp_l{k}", False, True))
    P.append(("b", "acf_e2_l1", _acf("e2", "e4", "e2", "e2l1"),
              _acf_s("e2", "e4", "e2", "e2l1"), _EXP, 12, "az_e2_exp_l1", False, True))
    P.append(("b", "acf_sg_l1", _acf("sg", "sg2", "sg", "sgl1"),
              _acf_s("sg", "sg2", "sg", "sgl1"), _EXP, 12, "az_sg_exp_l1", False, True))
    P.append(("b", "acf_x_l1", _acf("x", "x2", "x", "xl1"),
              _acf_s("x", "x2", "x", "xl1"), _HALF, 12, "az_x_half_l1", False, True))
    P.append(("b", "acf_e_l1", _acf("e", "e2", "e", "el1"),
              _acf_s("e", "e2", "e", "el1"), _HALF, 12, "az_e_half_l1", False, True))
    P.append(("b", "acf_ae_l1", _acf("ae", "e2", "ae", "ael1"),
              _acf_s("ae", "e2", "ae", "ael1"), _HALF, 12, "az_ae_half_l1", False, True))
    # ---- B. volatility clustering / variance ratio ------------------------
    P.append(("b", "vr_qh", _ratio("e2", 0.5, "e2", 1.0), _ratio_s("e2", 0.5, "e2", 1.0),
              _HALF, 16, "vr_qh", True, True))
    P.append(("b", "vr_he", _ratio("e2", 1.0, "e2", 2.0), _ratio_s("e2", 1.0, "e2", 2.0),
              _HALF, 16, "vr_he", True, True))
    P.append(("b", "vra_qh", _ratio("ae", 0.5, "ae", 1.0), _ratio_s("ae", 0.5, "ae", 1.0),
              _HALF, 16, "vra_qh", False, True))
    # ---- C. trend ----------------------------------------------------------
    for sch, mw, nm in ((_QUART, 12, "drift_q"), (_HALF, 12, "drift_h"),
                        (_EXP, 12, "drift_e")):
        P.append(("b", "drift", _drift, _drift_s, sch, mw, nm, True, True))
    P.append(("sub", "dslope_qh", "drift_q_z", "drift_h_z"))
    P.append(("sub", "dslope_he", "drift_h_z", "drift_e_z"))
    P.append(("b", "mk1", _mk("mk1"), _mk_s("mk1"), _HALF, 12, "mk1", True, True))
    P.append(("b", "mk8", _mk("mk8"), _mk_s("mk8"), _HALF, 16, "mk8", True, True))
    # ---- D. spectral --------------------------------------------------------
    P.append(("b", "sp_low", _sp_low, _sp_low_s, _HALF, 32, "sp_low", True, True))
    P.append(("b", "sp_high", _sp_high, _sp_high_s, _HALF, 32, "sp_high", True, True))
    P.append(("b", "sp_ent", _sp_ent, _sp_ent_s, _HALF, 32, "sp_ent", True, True))
    P.append(("b", "sp_cen", _sp_cen, _sp_cen_s, _HALF, 32, "sp_cen", True, True))
    P.append(("splr",))
    P.append(("spdom",))
    # ---- E. wavelet ----------------------------------------------------------
    P.append(("b", "hw_r12", _hw_r(0, 1), _hw_r_s(0, 1), _HALF, 24, "hw_r12", True, True))
    P.append(("b", "hw_r24", _hw_r(1, 2), _hw_r_s(1, 2), _HALF, 24, "hw_r24", True, True))
    P.append(("b", "hw_r48", _hw_r(2, 3), _hw_r_s(2, 3), _HALF, 32, "hw_r48", True, True))
    P.append(("b", "hw_hurst", _hw_hurst, _hw_hurst_s, _HALF, 32, "hw_hurst", True, True))
    # ---- F. complexity --------------------------------------------------------
    P.append(("b", "pe", _pe, _pe_s, _HALF, 24, "pe_half", True, True))
    P.append(("b", "pe", _pe, _pe_s, _EXP, 24, "pe_exp", True, True))
    P.append(("b", "hj_mob", _hj_mob, _hj_mob_s, _HALF, 16, "hj_mob", True, True))
    P.append(("b", "hj_comp", _hj_comp, _hj_comp_s, _HALF, 16, "hj_comp", True, True))
    return P


class StreamM03Dyn:
    """Streaming (one row per observation) implementation of ``m03_dyn``."""

    MODULE = "m03_dyn"

    __slots__ = ("_plan", "_cols", "_cal", "_ztab", "_names", "_ix", "_bases",
                 "_C", "_vb", "_pad", "_cap", "_zp", "_ep", "_ae", "_e2", "_sg",
                 "_cz", "_wi", "_ekey", "_t", "_fitted")

    def __init__(self) -> None:
        self._plan = _plan()
        self._cols: list[str] = []
        self._fitted = False
        self._t = -1

    # -------------------------------------------------------------------- fit
    def fit_historical(self, ctx) -> None:
        """Historical null calibration + history-warmed online buffers.

        Everything here is a function of the history alone and follows the batch
        code path: ``_local`` on the historical z / AR-residual pair, ``_cumz``
        of each transform, ``_Cal`` over the (16, 64, 256) window grid, driven
        by the *batch* statistic functions.
        """
        self._ekey = "res_mean" if "res_mean" in ctx.hist_tr else "mean"
        zh = np.asarray(ctx.hist_tr["mean"], dtype=np.float64)
        eh = np.asarray(ctx.hist_tr[self._ekey], dtype=np.float64)

        # --- historical null (identical call to batch) ---------------------
        trh = _local(zh, eh, PAD if len(zh) > PAD + 64 else 0)
        nh = len(next(iter(trh.values())))
        cum_h = {k: _cumz(v) for k, v in trh.items()}
        self._cal = _Cal(cum_h, nh)

        self._names = _tnames()
        if set(self._names) != set(trh.keys()):
            raise RuntimeError("m03_dyn transform set changed; streaming port is stale")
        self._ix = {k: i for i, k in enumerate(self._names)}
        self._bases = (self._ix["c0"], self._ix["s0"],
                       self._ix[f"h{HSCALES[0]}"], self._ix["p0"])

        # --- online-side buffers, warm-started from the tail of history ------
        pad = min(PAD, len(zh))
        self._pad = pad
        self._cap = max(int(getattr(ctx, "cap", 1024)), 64)
        self._alloc()
        self._build_ztab()
        zt = zh[len(zh) - pad:] if pad else zh[:0]
        et = eh[len(eh) - pad:] if pad else eh[:0]
        self._zp[:pad] = zt
        self._ep[:pad] = et
        self._ae[:pad] = np.abs(et)
        self._e2[:pad] = et * et
        self._sg[:pad] = np.sign(zt)
        self._cz[:pad + 1] = _cumz(zt)

        self._cols = self._build_cols()
        self._t = -1
        self._fitted = True

    def _alloc(self) -> None:
        cap = self._cap
        n = self._pad + cap
        self._zp = np.zeros(n, dtype=np.float64)
        self._ep = np.zeros(n, dtype=np.float64)
        self._ae = np.zeros(n, dtype=np.float64)
        self._e2 = np.zeros(n, dtype=np.float64)
        self._sg = np.zeros(n, dtype=np.float64)
        self._cz = np.zeros(n + 1, dtype=np.float64)
        # (time x name) cumulative sums: row t+1 is the running sum through t
        self._C = np.zeros((cap + 2, len(self._names)), dtype=np.float64)
        self._vb = np.zeros(len(self._names), dtype=np.float64)
        self._mkwin()

    def _mkwin(self) -> None:
        """Adaptive windows per index: batch's W_exp / W_half / W_quart."""
        tend = np.arange(self._cap + 2, dtype=np.int64)
        we = tend + 1
        wh = np.maximum(we // 2, 1)
        wq = np.maximum(we // 4, 1)
        # python ints: the hot path never needs the float/array form
        self._wi = (we.tolist(), wh.tolist(), wq.tolist())

    def _build_ztab(self) -> None:
        """Tabulate the null location/scale of every calibrated statistic.

        ``_Cal.z`` maps a window length ``W`` to ``mu(W)``/``sd(W)`` by
        piecewise-linear interpolation in ``log W`` over the three-point grid.
        The adaptive windows are always *integers* in ``1 .. cap`` and
        ``_loglin`` (``np.interp`` plus two elementwise extrapolation branches)
        is strictly elementwise, so evaluating the batch expression once over
        the whole integer range at fit time and indexing it per step removes
        every interpolation call from the hot path without changing a bit.
        The null itself is built from the BATCH statistic (``op[2]``).
        """
        cal = self._cal
        wv = np.arange(self._cap + 2, dtype=np.float64)
        lw = np.log(np.maximum(wv, 1.0))
        self._ztab = {}
        for op in self._plan:
            if op[0] != "b" or not op[8]:
                continue
            key = op[1]
            if key in self._ztab:
                continue
            mus, lsds = cal.params(key, op[2])
            self._ztab[key] = (_loglin(mus, lw).tolist(),
                               np.maximum(np.exp(_loglin(lsds, lw)), 1e-9).tolist())

    def _grow(self) -> None:
        old = self._cap
        self._cap *= 2
        for nm in ("_zp", "_ep", "_ae", "_e2", "_sg", "_cz"):
            a = getattr(self, nm)
            b = np.zeros(len(a) + old, dtype=np.float64)
            b[:len(a)] = a
            setattr(self, nm, b)
        C = np.zeros((self._cap + 2, len(self._names)), dtype=np.float64)
        C[:len(self._C)] = self._C
        self._C = C
        self._mkwin()
        self._build_ztab()

    def _build_cols(self) -> list[str]:
        cols: list[str] = []
        for op in self._plan:
            if op[0] == "b":
                name, want_raw, want_z = op[6], op[7], op[8]
                if want_raw:
                    cols.append(name)
                if want_z:
                    cols.append(name + "_z")
            elif op[0] == "sub":
                cols.append(op[1])
            elif op[0] == "splr":
                cols.append("sp_lr")
            else:
                cols.append("sp_dom")
        return cols

    # ---------------------------------------------------------- per-point work
    def _absorb(self, ctx) -> None:
        """Fold observation ``t`` into the 51 per-point transforms of ``_local``.

        Index origin: ``i = pad + t`` in the padded buffer, so ``j = i - pad = t``
        -- the same absolute index the batch ``_local`` gives online point ``t``
        (its ``j = arange(n) - off``).
        """
        t = ctx.t
        if t >= self._cap:
            self._grow()
        pad = self._pad
        i = pad + t
        zp = self._zp; ep = self._ep; aeb = self._ae; e2b = self._e2
        sgb = self._sg; cz = self._cz
        vb = self._vb

        z = ctx.tr["mean"].item(t)
        e = ctx.tr[self._ekey].item(t)
        zp[i] = z
        ep[i] = e
        ae = abs(e); aeb[i] = ae
        e2 = e * e; e2b[i] = e2
        sg = _sign(z); sgb[i] = sg
        cz[i + 1] = cz[i] + z

        # --- dependence -----------------------------------------------------
        vb[0] = z
        vb[1] = z * z
        k = 2
        for q in ZLAGS:
            vb[k] = z * zp[i - q] if i >= q else 0.0
            k += 1
        vb[k] = e; k += 1
        vb[k] = e2; k += 1
        vb[k] = e2 * e2; k += 1
        for q in ELAGS:
            vb[k] = e * ep[i - q] if i >= q else 0.0
            k += 1
        vb[k] = ae; k += 1
        for q in ALAGS:
            vb[k] = ae * aeb[i - q] if i >= q else 0.0
            k += 1
        vb[k] = e2 * e2b[i - 1] if i >= 1 else 0.0; k += 1
        vb[k] = sg; k += 1
        vb[k] = sg * sg; k += 1
        vb[k] = sg * sgb[i - 1] if i >= 1 else 0.0; k += 1

        # --- trend ------------------------------------------------------------
        vb[k] = (float(i) - float(pad)) * z; k += 1
        dz = (z - zp[i - 1]) if i >= 1 else 0.0
        ddz = (z - 2.0 * zp[i - 1] + zp[i - 2]) if i >= 2 else 0.0
        vb[k] = dz; k += 1
        vb[k] = dz * dz; k += 1
        vb[k] = ddz; k += 1
        vb[k] = ddz * ddz; k += 1
        vb[k] = _sign(dz); k += 1
        vb[k] = _sign(z - zp[i - 8]) if i >= 8 else 0.0; k += 1

        # --- spectral: windowed-DFT quadratures ---------------------------------
        ang = _TWO_PI * (FREQS * float(t))
        vb[k:k + NF] = np.cos(ang) * z
        vb[k + NF:k + 2 * NF] = np.sin(ang) * z
        k += 2 * NF

        # --- wavelet: Haar detail energies ----------------------------------------
        for q, s in enumerate(HSCALES):
            if i >= 2 * s - 1:
                dd = (cz[i + 1] - 2.0 * cz[i + 1 - s] + cz[i + 1 - 2 * s]) / _SQ2S[q]
            else:
                dd = 0.0
            vb[k] = dd * dd; k += 1

        # --- complexity: order-3 ordinal patterns -------------------------------
        a = zp[i - 2] if i >= 2 else 0.0
        b = zp[i - 1] if i >= 1 else 0.0
        code = 4 * (a > b) + 2 * (a > z) + (b > z)
        for pc in PCODES:
            vb[k] = 1.0 if code == pc else 0.0
            k += 1

        C = self._C
        np.add(C[t], vb, out=C[t + 1])
        self._t = t

    # ------------------------------------------------------------------- step
    def step(self, ctx) -> np.ndarray:
        """Row ``t`` of the batch output, float64, non-finite mapped to NaN."""
        self._absorb(ctx)
        t = ctx.t
        t1 = t + 1
        C = self._C
        ix = self._ix
        bases = self._bases
        wi = self._wi
        we = wi[0][t]; wh = wi[1][t]; wq = wi[2][t]
        scheme = ((_R1(C, we, t1, ix, bases), we),
                  (_R1(C, wh, t1, ix, bases), wh),
                  (_R1(C, wq, t1, ix, bases), wq))
        ztab = self._ztab
        store: dict[str, float] = {}
        vals: list[float] = []
        nan = np.nan

        for op in self._plan:
            kind = op[0]
            if kind == "b":
                _, key, _fnb, fn, sch, minW, name, want_raw, want_z = op
                R, Wi = scheme[sch]
                v = fn(R, Wi, t) if Wi >= minW else nan   # == np.where(W>=minW, .., nan)
                if want_raw:
                    store[name] = v
                    vals.append(v)
                if want_z:
                    if -_INF < v < _INF:                  # == np.isfinite(v)
                        mu, sd = ztab[key]
                        r = (v - mu[Wi]) / sd[Wi]
                        if r < -CLIP:                     # == np.clip(r, -CLIP, CLIP)
                            r = -CLIP
                        elif r > CLIP:
                            r = CLIP
                    else:
                        r = nan
                    store[name + "_z"] = r
                    vals.append(r)
            elif kind == "sub":
                vals.append(store[op[2]] - store[op[3]])
            elif kind == "splr":
                vals.append(np.log((store["sp_low"] + 1e-6) / (store["sp_high"] + 1e-6)))
            else:
                R, Wi = scheme[_HALF]
                vals.append(_sp_dom_s(R, Wi, t) if Wi >= 32 else nan)

        a = np.asarray(vals, dtype=np.float64)
        # batch scrubs non-finites AFTER the float32 cast
        a[~np.isfinite(a.astype(np.float32))] = np.nan
        return a

    # ------------------------------------------------------------------- cols
    @property
    def cols(self) -> list[str]:
        if not self._fitted:
            raise RuntimeError("fit_historical() must be called before cols")
        return self._cols
