"""Exact incremental port of the batch feature module ``m04_resid``.

Design
------
Everything in ``sbr.features.m04_resid`` that depends on the HISTORICAL segment
only -- AR coefficients (plain / ridge / Huber), the residual scale factors, the
EWMA / GARCH filter states at the end of history, the tail thresholds, and the
whole null-calibration machinery -- is computed once in :meth:`fit_historical`
by calling *the batch module's own helpers on the same inputs*.  There is
therefore no second implementation of the historical half and no drift.

The online half is a set of scalar recursions:

* ``raw``                         elementwise standardisation, O(1);
* ``ar1/ar2/ar5/arR/arH``         one ``ar_filter_causal`` call on a CONSTANT
  length-6 causal window of the standardised stream.  This is the same trick
  ``StreamCtx.push`` uses: a hand-rolled dot product disagrees with batch's BLAS
  ``gemv`` in the last ulp on ~1 % of points, whereas calling the identical
  function on a constant-height window is bitwise identical (verified for
  p = 1,2,3,5 in the parity test);
* ``vol / volM / volG / cmb``     the predictive EWMA / GARCH recursions.
  ``scipy.signal.lfilter`` evaluates ``y = b0*x + z ; z = -a1*y`` sequentially in
  double precision, so carrying the two scalars ``(y, z)`` reproduces it bit for
  bit (verified in the parity test).

The null-calibration query ``(_Null.z)`` is ``np.interp`` on a 6-node log grid.
Because ``np.interp`` is elementwise and ``np.log``/``np.exp`` are
array-length-invariant, the interpolation is tabulated once at fit time for
every window length ``L = 1..cap`` and looked up in O(1) at query time; the
arithmetic is the same as batch's.

Cost per observation: 5 short ``ar_filter_causal`` calls + ~60 scalar monitor
updates.  No rescan of the online prefix, no refit, no FFT.
"""
from __future__ import annotations

import math

import numpy as np

from sbr.features.m04_resid import (
    A_MON,
    B_MON,
    BURN_MAX,
    CLIP,
    ECLIP,
    GRID,
    TIER_A,
    TIER_B,
    W_TR,
    _GA,
    _GB,
    _HL22,
    _HL63,
    _ewma_pred,
    _fit_ar_huber,
    _fit_ar_ridge,
    _garch_pred,
    _null_llr,
    _null_mean,
    _point_transforms,
    _std,
)
from sbr.transforms import _ar_resid, _fit_ar, ar_filter_causal

#: number of trailing historical z values kept in front of the online buffer and
#: the constant causal-window length used for every AR order.  _AR_PAD must
#: exceed _AR_WIN + max_ar_order (6 + 5).
_AR_PAD = 16
_AR_WIN = 6

#: representation order == batch column order
_REPS = TIER_A + TIER_B
_MAX_ONLINE = 1024
#: integer dispatch codes for the per-point monitor transforms
_MON_CODE = {"mean": 0, "var": 1, "abs": 2, "tail": 3, "acf1": 4, "acf1sq": 5}
_NAN = float("nan")


# --------------------------------------------------------------------- scalars
def _clip1(v: float, lo: float, hi: float) -> float:
    """Scalar ``np.clip`` (NaN passes through, as ``minimum(maximum(...))`` does)."""
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _max1(v: float, lo: float) -> float:
    """Scalar ``np.maximum(v, lo)`` (NaN propagates)."""
    return lo if v < lo else v


def _llr1(m: float, q: float) -> float:
    """Scalar version of ``_llr_stat`` -- same operation order as the batch one."""
    s2 = q - m * m
    if s2 != s2:                       # NaN -> q - log(NaN) - 1 == NaN
        return s2
    if s2 < 1e-6:
        s2 = 1e-6
    return q - float(np.log(s2)) - 1.0


def _tabulate(nl, cap: int):
    """Tabulate ``_Null.z``'s interpolation for L = 1..cap.

    Uses exactly the expressions in ``_Null.z``; ``np.interp``/``np.log``/
    ``np.exp`` are elementwise and length-invariant, so entry ``L-1`` equals what
    batch computes for that ``L``.
    """
    L = np.arange(1, cap + 1, dtype=np.float64)
    x = np.log(np.clip(L, 1.0, None))
    mu = np.interp(x, nl.lg, nl.mu)
    sd = np.exp(np.interp(x, nl.lg, np.log(nl.sd)))
    return mu, np.maximum(sd, 1e-12)


class StreamM04Resid:
    """Incremental ``m04_resid``.  See :mod:`sbr.stream.CONTRACT`."""

    MODULE = "m04_resid"

    def __init__(self, cap: int = _MAX_ONLINE):
        self.cap = int(cap)
        self._cols: list[str] = []

    # ------------------------------------------------------------------ cols
    @property
    def cols(self) -> list[str]:
        return list(self._cols)

    # ------------------------------------------------------------------- fit
    def fit_historical(self, ctx) -> None:
        hp = ctx.hp
        hist = np.asarray(ctx.hist, dtype=np.float64)
        zh = (hist - hp.mu) / hp.sd
        h = len(zh)
        burn = min(BURN_MAX, max(h // 5, 1))

        grid = GRID[GRID <= max(h // 2, 5)]
        if len(grid) == 0:
            grid = np.array([5])
        self._grid = grid

        # ---------------------------------------------------------- reps
        eh: dict[str, np.ndarray] = {}

        # raw ---------------------------------------------------------------
        eh["raw"] = zh

        # AR family ---------------------------------------------------------
        ar_specs = []          # (rep index, coef, p, s)
        ar_defs = [("ar1", _fit_ar(zh, 1)),
                   ("ar2", hp.ar_coef),
                   ("ar5", _fit_ar(zh, 5)),
                   ("arR", _fit_ar_ridge(zh, 3, 0.05)),
                   ("arH", _fit_ar_huber(zh, 2))]
        ar_hist_resid: dict[str, np.ndarray] = {}
        for tag, coef in ar_defs:
            coef = np.asarray(coef, dtype=np.float64)
            rh = _ar_resid(zh, coef)
            s = _std(rh)
            eh[tag] = rh / s
            ar_hist_resid[tag] = rh
            ar_specs.append((_REPS.index(tag), coef, len(coef), s))
        self._ar_specs = ar_specs

        # the constant-length causal AR window buffer: the last _AR_PAD historical
        # z values, followed by the online z values.
        off = min(h, _AR_PAD)
        self._off = off
        self._zf = np.zeros(off + self.cap, dtype=np.float64)
        if off:
            self._zf[:off] = zh[-off:]
        self._zh_full = zh
        # per-order window length (constant across t, which is what keeps the
        # BLAS gemv path identical to batch's).  w < 1 can only happen for
        # h < 5, where `_fit_ar*` always returns a zero coefficient vector and
        # the residual is exactly z regardless of the warm-up -- the O(1)
        # single-point fallback is then bitwise correct as well.
        self._ar_win = []
        for _, _, p, _ in ar_specs:
            self._ar_win.append(min(_AR_WIN, off - p + 1))

        x2h = zh * zh

        # vol : z / sqrt(EWMA_pred(z^2, hl 22)) --------------------------------
        v_hist = _ewma_pred(x2h, _HL22, 1.0)
        e = zh / np.sqrt(np.maximum(v_hist, 1e-9))
        s_vol = _std(e[burn:h])
        eh["vol"] = e[burn:h] / s_vol
        self._vol_s = s_vol
        self._vol_y, self._vol_z = self._ewma_state(_HL22, x2h, v_hist)

        # volM : z / EWMA_pred(min(|z|,4), hl 63) ------------------------------
        azh = np.minimum(np.abs(zh), 4.0)
        x0m = float(np.mean(azh))
        sm_hist = _ewma_pred(azh, _HL63, x0m)
        e = zh / np.maximum(sm_hist, 1e-9)
        s_volM = _std(e[burn:h])
        eh["volM"] = e[burn:h] / s_volM
        self._volM_s = s_volM
        self._volM_y, self._volM_z = self._ewma_state(_HL63, azh, sm_hist)

        # volG : variance-targeted GARCH(1,1), 12-point deterministic QMLE grid
        best, ba, bb = np.inf, _GA[0], _GB[0]
        for a in _GA:
            for b in _GB:
                if a + b > 0.995:
                    continue
                vv = _garch_pred(x2h, a, b)[burn:]
                vv = np.maximum(vv, 1e-9)
                loss = float(np.sum(np.log(vv) + x2h[burn:] / vv))
                if loss < best:
                    best, ba, bb = loss, a, b
        vg_raw = _garch_pred(x2h, ba, bb)
        v = np.maximum(vg_raw, 1e-9)
        e = zh / np.sqrt(v)
        s_volG = _std(e[burn:h])
        eh["volG"] = e[burn:h] / s_volG
        self._volG_s = s_volG
        self._g_a, self._g_b = float(ba), float(bb)
        self._g_om = (1.0 - ba - bb) * 1.0
        self._g_z = float(bb * vg_raw[-1]) if h else float(bb * 1.0)
        self._g_x2prev = float(x2h[-1]) if h else 1.0

        # cmb : AR(2) residual then EWMA(22) volatility normalisation ----------
        rh2 = eh["ar2"]
        hb = len(rh2)
        r2 = rh2 * rh2
        vr_hist = _ewma_pred(r2, _HL22, 1.0)
        e = rh2 / np.sqrt(np.maximum(vr_hist, 1e-9))
        b2 = min(BURN_MAX, max(hb // 5, 1))
        s_cmb = _std(e[b2:hb])
        eh["cmb"] = e[b2:hb] / s_cmb
        self._cmb_s = s_cmb
        self._cmb_y, self._cmb_z = self._ewma_state(_HL22, r2, vr_hist)

        # -------------------------------------------------- monitors / nulls
        self._tierA = []
        self._want = []
        self._thr = []
        self._lag = []
        self._msum = []
        self._mmu = []
        self._msd = []
        self._midx = []
        self._nmu = []
        self._nsd = []
        self._llr_idx = []
        self._w32_idx = []
        self._nulls = []            # kept so the tables can be re-tabulated on grow
        self._nulls_llr = []

        cols: list[str] = []
        cap = self.cap
        for ri, tag in enumerate(_REPS):
            e_h = eh[tag]
            tier_a = tag in TIER_A
            want = A_MON if tier_a else B_MON
            ec_h = np.clip(e_h, -ECLIP, ECLIP)
            thr = float(np.quantile(np.abs(ec_h), 0.95)) if len(ec_h) else 3.0
            th = _point_transforms(e_h, e_h[0] if len(e_h) else 0.0, thr, want)

            mus, sds, idxs, nulls = [], [], [], []
            for m in want:
                ah = th[m][1:] if m in ("acf1", "acf1sq") else th[m]
                nl = _null_mean(ah, self._grid)
                mu, sd = _tabulate(nl, cap)
                mus.append(mu.tolist())
                sds.append(sd.tolist())
                nulls.append(nl)
                idxs.append(len(cols))
                cols.append(f"{tag}_e_{m}")

            nll = _null_llr(e_h, np.clip(e_h, -ECLIP, ECLIP) ** 2, self._grid)
            nmu, nsd = _tabulate(nll, cap)
            self._nulls_llr.append(nll)
            self._nmu.append(nmu.tolist())
            self._nsd.append(nsd.tolist())
            self._llr_idx.append(len(cols))
            cols.append(f"{tag}_e_llr")

            if tier_a:
                self._w32_idx.append(len(cols))
                cols.append(f"{tag}_w{W_TR}_llr")
            else:
                self._w32_idx.append(-1)

            self._tierA.append(tier_a)
            self._want.append(tuple(_MON_CODE[m] for m in want))
            self._thr.append(thr)
            self._lag.append(_clip1(float(e_h[-1]) if len(e_h) else 0.0, -ECLIP, ECLIP))
            self._msum.append([0.0] * len(want))
            self._mmu.append(mus)
            self._msd.append(sds)
            self._midx.append(idxs)
            self._nulls.append(nulls)

        self._cols = cols
        self._k = len(cols)
        self._row = [0.0] * self._k
        self._c1 = [[0.0] for _ in _REPS]
        self._c2 = [[0.0] for _ in _REPS]
        self._t = -1
        self._e = [0.0] * len(_REPS)
        self._i_vol = _REPS.index("vol")
        self._i_volM = _REPS.index("volM")
        self._i_volG = _REPS.index("volG")
        self._i_cmb = _REPS.index("cmb")
        self._i_ar2 = _REPS.index("ar2")
        self._nrep = len(_REPS)

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _ewma_state(alpha: float, xh: np.ndarray, out_hist: np.ndarray):
        """Filter state at the end of history for ``_ewma_pred``.

        ``_ewma_pred`` returns ``out[t] = y[t-1]`` (predictive).  ``out[-1]`` is
        therefore ``y[H-2]`` and ``lfilter``'s carried state at that moment is
        exactly ``(1-alpha)*y[H-2]``; for ``H == 1`` ``out[-1] == x0`` and the
        carried state is the initial ``zi == (1-alpha)*x0``.  Both cases collapse
        to the same two lines.
        """
        if len(xh) == 0:
            y = 1.0
            return y, (1.0 - alpha) * y
        z = (1.0 - alpha) * float(out_hist[-1])
        y = alpha * float(xh[-1]) + z
        return y, (1.0 - alpha) * y

    def _grow(self):
        """Double the capacity.  ``np.interp``/``np.log``/``np.exp`` are
        length-invariant, so re-tabulating a longer table leaves every already
        emitted value untouched."""
        self.cap *= 2
        cap = self.cap
        zf = np.zeros(self._off + cap, dtype=np.float64)
        zf[:len(self._zf)] = self._zf
        self._zf = zf
        for ri in range(len(_REPS)):
            for a, nl in enumerate(self._nulls[ri]):
                mu, sd = _tabulate(nl, cap)
                self._mmu[ri][a] = mu.tolist()
                self._msd[ri][a] = sd.tolist()
            mu, sd = _tabulate(self._nulls_llr[ri], cap)
            self._nmu[ri] = mu.tolist()
            self._nsd[ri] = sd.tolist()

    # -------------------------------------------------------------- step
    def step(self, ctx) -> np.ndarray:
        t = self._t + 1
        if t >= self.cap:
            self._grow()
        self._t = t
        L = t + 1
        Lf = float(L)
        z = float(ctx.tr["mean"][ctx.t])          # (x - mu_h)/sd_h, bitwise as batch
        x2 = z * z

        zf = self._zf
        i = self._off + t
        zf[i] = z

        e = self._e
        e[0] = z                                   # raw

        # ---- AR residuals (constant-height BLAS window, see module docstring)
        for j, (ri, coef, p, s) in enumerate(self._ar_specs):
            w = self._ar_win[j]
            if w >= 1:
                lo = i - w + 1
                e[ri] = ar_filter_causal(zf[lo:i + 1], coef, zf[lo - p:lo])[-1] / s
            else:                                  # h < 5: coef is all zeros
                e[ri] = ar_filter_causal(zf[i:i + 1], coef, self._zh_full)[-1] / s

        # ---- vol : predictive EWMA of z^2
        y = self._vol_y
        e[self._i_vol] = (z / math.sqrt(_max1(y, 1e-9))) / self._vol_s
        y = _HL22 * x2 + self._vol_z
        self._vol_y, self._vol_z = y, (1.0 - _HL22) * y

        # ---- volM : predictive EWMA of min(|z|, 4)
        y = self._volM_y
        e[self._i_volM] = (z / _max1(y, 1e-9)) / self._volM_s
        az = 4.0 if abs(z) > 4.0 else abs(z)       # np.minimum(np.abs(z), 4.0)
        y = _HL63 * az + self._volM_z
        self._volM_y, self._volM_z = y, (1.0 - _HL63) * y

        # ---- volG : predictive variance-targeted GARCH(1,1)
        u = self._g_om + self._g_a * self._g_x2prev
        vg = 1.0 * u + self._g_z
        self._g_z = self._g_b * vg
        self._g_x2prev = x2
        e[self._i_volG] = (z / math.sqrt(_max1(vg, 1e-9))) / self._volG_s

        # ---- cmb : AR(2) residual, then EWMA(22) normalisation of the residual
        rc = e[self._i_ar2]
        y = self._cmb_y
        e[self._i_cmb] = (rc / math.sqrt(_max1(y, 1e-9))) / self._cmb_s
        y = _HL22 * (rc * rc) + self._cmb_z
        self._cmb_y, self._cmb_z = y, (1.0 - _HL22) * y

        # ------------------------------------------------------- monitors
        row = self._row
        lags = self._lag
        nan = _NAN
        for ri in range(self._nrep):
            ev = e[ri]
            ec = -ECLIP if ev < -ECLIP else (ECLIP if ev > ECLIP else ev)
            e2 = ec * ec
            lag = lags[ri]
            lags[ri] = ec
            aec = -ec if ec < 0.0 else ec          # abs(ec); NaN passes through

            sums = self._msum[ri]
            mus = self._mmu[ri]
            sds = self._msd[ri]
            idxs = self._midx[ri]
            thr = self._thr[ri]
            for a, m in enumerate(self._want[ri]):
                if m == 1:
                    v = e2
                elif m == 5:
                    v = (e2 - 1.0) * (lag * lag - 1.0)
                elif m == 0:
                    v = ev
                elif m == 2:
                    v = abs(ec)
                elif m == 3:
                    v = 1.0 if aec > thr else 0.0
                else:                              # 4 -> acf1
                    v = ec * lag
                s = sums[a] + v
                sums[a] = s
                q = (s / Lf - mus[a][t]) / sds[a][t]
                row[idxs[a]] = -CLIP if q < -CLIP else (CLIP if q > CLIP else q)

            c1 = self._c1[ri]
            c2 = self._c2[ri]
            s1 = c1[t] + ev
            s2 = c2[t] + e2
            c1.append(s1)
            c2.append(s2)

            nmu = self._nmu[ri]
            nsd = self._nsd[ri]
            if L < 8:
                row[self._llr_idx[ri]] = nan
            else:
                st = _llr1(s1 / Lf, s2 / Lf)
                q = (st - nmu[t]) / nsd[t]
                row[self._llr_idx[ri]] = -CLIP if q < -CLIP else (CLIP if q > CLIP else q)

            wi = self._w32_idx[ri]
            if wi >= 0:
                if L < W_TR:
                    row[wi] = nan
                else:
                    st = _llr1((s1 - c1[L - W_TR]) / W_TR, (s2 - c2[L - W_TR]) / W_TR)
                    q = (st - nmu[W_TR - 1]) / nsd[W_TR - 1]
                    row[wi] = -CLIP if q < -CLIP else (CLIP if q > CLIP else q)

        out = np.array(row, dtype=np.float64)
        out[~np.isfinite(out)] = np.nan            # batch does this at the end
        return out
