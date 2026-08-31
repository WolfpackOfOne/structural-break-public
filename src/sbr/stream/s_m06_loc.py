"""Incremental port of the ``m06_loc`` batch feature module.

The batch module (``sbr.features.m06_loc``) does, at every online index ``t``:

  1. AR(6) residuals ``e_t`` of the standardised series, filtered with the
     coefficients fitted on history (a *module-local* AR fit -- the shared
     ``StreamCtx`` carries AR(2), which is not what this module wants);
  2. three per-point streams of that residual (``rsq``, ``rlab``, ``rl1``);
  3. for each of two engines (H = "segment vs history", P = "segment vs its
     own preceding segment") a scan over a geometric bank of segment lengths,
     every candidate statistic being a difference of a cumulative sum;
  4. max / argmax / runner-up over the candidates, the multiplicity-matched
     calibration of the max, and the trailing-16 dispersion of ``log2(m_hat)``;
  5. a control family at four FIXED trailing offsets.

Everything the online half needs is either (a) elementwise in ``t``, or (b) a
difference of cumulative sums, so the port is exact and cheap.

Why bitwise parity is reachable here
------------------------------------
* **Historical half.** ``_fit_ar``, ``_ar_apply``, ``_streams`` and
  ``_hist_null`` are *imported from the batch module* and run on the same
  input, so there is literally one implementation.  Nothing is re-derived.
* **The candidate grid does not depend on ``n_online``.**  ``_hist_null``
  truncates the grid to ``K = #{m : mult*m <= n_online}``, but row ``t`` only
  ever consults nodes with ``need <= t+1 <= n_online``, and the null of the
  max over the first ``k`` nodes (``CUS[k-1]``) involves only those same
  nodes.  So the streaming engine builds the null on the FULL grid at fit time
  (``n_online`` unknown) and gets bitwise-identical rows.  Every per-node
  quantity (``med``, ``sd``, ``SZ``) and every ``CUS`` row is independent of
  ``K`` because the null end-point set is pinned to ``mult*M[-1]-1`` of the
  *full* grid, never to the truncated one.
* **The AR filter.** ``_ar_apply`` is written by the batch module as an
  explicit accumulation over lags (``out -= coef[k] * pad[...]``, one lag at a
  time, never a BLAS gemv).  Each element is therefore an independent chain of
  six multiply-subtract pairs, which we reproduce scalar-for-scalar in the
  same order.  The trailing ``/ sig`` is applied after the whole loop, exactly
  as batch does.
* **``np.log`` / ``np.log2`` / ``np.log10``** are length-independent in this
  numpy build (checked against 100k-element references at lengths 1, 2, 3 and
  13), so evaluating them on a length-1 array reproduces the batch float.  We
  never hand-roll them.
* **Ties.** ``_localise`` takes ``A.argmax(axis=0)`` over the ``-1``-filled
  score column.  We build the identical length-K column and call ``np.argmax``
  on it, so numpy's first-occurrence tie rule is inherited rather than
  re-implemented.
* **``np.cumsum`` accumulates sequentially**, so ``c[t+1] = c[t] + v[t]`` is
  the same float64 batch computes -- true both for the stream cumsums and for
  the three cumsums inside ``_roll_std`` (which we inline, because it is the
  one batch helper that is *not* elementwise in ``t``).

Cost per observation: O(sum of candidate counts) = at most 13 + 11 nodes per
stream per engine, i.e. <= 72 O(1) segment statistics plus one binary search
each into a <=1000-point historical null.  No rescan of the online prefix, no
re-sort, no refit.  See the timing test.

Speed note: the hot path deliberately works in *Python* floats rather than
0-d numpy scalars (~5x cheaper per operation).  This is not an approximation:
a Python float IS an IEEE-754 binary64, ``float(np.float64)`` is exact, and
``+ - * /`` on Python floats are the same correctly-rounded operations numpy's
scalar loops perform.  The two places where that equivalence would NOT hold --
``np.log`` inside ``_streams`` and ``np.log10`` in the max calibration -- are
still evaluated by numpy.  ``np.clip`` and ``np.maximum`` are replaced by the
equivalent comparisons, whose NaN/inf behaviour is identical (NaN fails both
comparisons and falls through unchanged, exactly as ``np.clip`` propagates it).
"""
from __future__ import annotations

from bisect import bisect_right
from math import sqrt

import numpy as np

from sbr.features.m06_loc import (
    AR_ORDER,
    CLIP,
    FIX_H,
    FIX_P,
    M_H,
    M_P,
    QFLOOR,
    STAB_W,
    STREAMS,
    _ar_apply,
    _fit_ar,
    _hist_null,
    _streams,
)

_ENGINES = (("h", "H", M_H, FIX_H), ("p", "P", M_P, FIX_P))
_SUFFIXES = ("q", "sz", "lm", "rel", "gap", "stab")
_CAP0 = 1024


class _Combo:
    """Per (engine, stream) fitted null + online scratch state."""

    __slots__ = ("etag", "kind", "st", "fixo", "nl", "M", "mult", "two",
                 "log2M", "K", "Nf", "SZ", "CUS", "fxj", "needs",
                 "Marr", "Mfl", "medarr", "sdarr", "A", "c1", "c2", "ck")

    def __init__(self, etag, kind, st, M, fixo, nl):
        self.etag, self.kind, self.st, self.fixo = etag, kind, st, fixo
        self.nl = nl
        self.mult = 1 if kind == "H" else 2
        if nl is None:
            self.fxj = list(fixo)
            return
        Ma = np.asarray(nl["M"], dtype=np.int64)
        self.M = [int(m) for m in Ma]
        self.K = len(self.M)
        self.two = self.mult == 2
        self.Nf = float(int(nl["N"]))
        # one contiguous ndarray per node so the binary search is a bare call
        self.SZ = [np.ascontiguousarray(nl["SZ"][j]) for j in range(self.K)]
        self.CUS = [np.ascontiguousarray(nl["CUS"][j]) for j in range(self.K)]
        # vectorised scan inputs (batch divides by the python int m, which numpy
        # promotes to the same float64 divisor)
        self.Marr = Ma
        self.Mfl = Ma.astype(np.float64)
        self.medarr = np.asarray(nl["med"], dtype=np.float64)
        self.sdarr = np.asarray(nl["sd"], dtype=np.float64)
        #: node j is usable at index t iff needs[j] <= t + 1  (batch's `need`)
        self.needs = [int(m) * self.mult for m in self.M]
        # batch: lm = np.log2(mhat) with mhat a float64 array.  np.log2 is
        # length-independent here, so the 13-node array gives the same floats.
        self.log2M = [float(x) for x in np.log2(self.Mfl)]
        pos = {m: j for j, m in enumerate(self.M)}
        self.fxj = [pos.get(int(m)) for m in fixo]
        # scratch reused every step (batch's `A` column of the score matrix)
        self.A = np.full(self.K, -1.0)


class StreamM06Loc:
    """Streaming (one row per observation) implementation of ``m06_loc``."""

    MODULE = "m06_loc"

    __slots__ = ("_cols", "_combos", "_coef", "_sig", "_ztail", "_e_prev0",
                 "_zo", "_eo", "_cs", "_cap", "_fitted")

    def __init__(self) -> None:
        self._cols: list[str] = []
        self._combos: list[_Combo] = []
        self._fitted = False

    # ----------------------------------------------------------------- fit
    def fit_historical(self, ctx) -> None:
        """Fit the module-local AR(6) and every historical null; reset online."""
        hp = ctx.hp
        zh = (np.asarray(ctx.hist, dtype=np.float64) - hp.mu) / hp.sd

        coef = _fit_ar(zh, AR_ORDER)
        eh_full = _ar_apply(zh, coef, np.zeros(AR_ORDER))
        eh = eh_full[AR_ORDER:]
        sig = float(np.std(eh, ddof=1)) if len(eh) > 1 else 1.0
        sig = max(sig, 1e-9)
        eh = eh / sig
        hs = _streams(eh, 0.0)

        self._coef = [float(x) for x in np.asarray(coef, dtype=np.float64)]
        self._sig = sig
        # `_ar_apply(zo, coef, zh)` pads with zh[-p:] when history is long
        # enough and with zeros otherwise -- replicate that choice exactly.
        if len(zh) >= AR_ORDER:
            self._ztail = [float(x) for x in zh[-AR_ORDER:]]
        else:
            self._ztail = [0.0] * AR_ORDER
        self._e_prev0 = float(eh[-1]) if len(eh) else 0.0

        cols: list[str] = []
        combos: list[_Combo] = []
        for etag, kind, M, fixo in _ENGINES:
            mult = 1 if kind == "H" else 2
            for st in STREAMS:
                # n_online is unknown while streaming; ask for the FULL grid.
                # Row t only ever reads nodes with need <= t+1 <= n_online, and
                # every per-node/CUS quantity is K-independent, so this is the
                # same null batch would have built (see module docstring).
                nl = _hist_null(hs[st], M, kind, int(M[-1]) * mult)
                combos.append(_Combo(etag, kind, st, M, fixo, nl))
                base = f"{etag}_{st}"
                for suf in _SUFFIXES:
                    cols.append(f"loc_{base}_{suf}")
                for m in fixo:
                    cols.append(f"fx_{base}_{m}")
        self._cols = cols
        self._combos = combos

        self._alloc()
        self._fitted = True

    def _alloc(self) -> None:
        """Reset every online buffer.

        The three stream cumsums live in float64 ndarrays because the candidate
        scan gathers ``len(grid)`` of their entries at once; the ``_roll_std``
        cumsums live in Python lists because they are only ever read by scalar
        index.  Either way ``c[t+1] = c[t] + v[t]`` is the same IEEE binary64
        addition ``np.cumsum`` performs, which is what parity rests on.
        """
        self._zo = []
        self._eo = []
        self._cap = _CAP0
        self._cs = {st: np.zeros(_CAP0 + 1, dtype=np.float64) for st in STREAMS}
        for cb in self._combos:
            cb.c1 = [0.0]
            cb.c2 = [0.0]
            cb.ck = [0.0]

    def _grow(self) -> None:
        self._cap *= 2
        for st in STREAMS:
            a = self._cs[st]
            b = np.zeros(self._cap + 1, dtype=np.float64)
            b[:len(a)] = a
            self._cs[st] = b

    # ---------------------------------------------------------------- step
    def step(self, ctx) -> np.ndarray:
        """Row ``t`` of the batch output, float64, non-finite mapped to NaN."""
        t = ctx.t
        n = t + 1
        nf = float(n)
        if n >= self._cap:
            self._grow()

        # --- standardised point.  build_transforms["mean"] IS (x - mu) / sd,
        #     which is exactly the batch module's `zo`.
        z = float(ctx.tr["mean"][t])
        zo = self._zo
        zo.append(z)

        # --- AR(6) residual, one lag at a time, in the batch's loop order.
        coef = self._coef
        ztail = self._ztail
        e = z
        for k in range(AR_ORDER):
            i = t - k - 1
            e = e - coef[k] * (zo[i] if i >= 0 else ztail[AR_ORDER + i])
        e = e / self._sig          # batch divides after the whole lag loop
        eo = self._eo

        # --- the three monitored streams (batch code, length-1 window) so the
        #     np.log inside rlab is numpy's, not libm's.
        prev = eo[t - 1] if t > 0 else self._e_prev0
        eo.append(e)
        sv = _streams(np.array([e], dtype=np.float64), prev)
        cs = self._cs
        for st in STREAMS:
            c = cs[st]
            c[n] = c[t] + sv[st][0]

        vals: list[float] = []
        for cb in self._combos:
            if cb.nl is None:
                vals.extend([np.nan] * (len(_SUFFIXES) + len(cb.fixo)))
                continue
            vals.extend(self._combo_row(cb, cs[cb.st], t, n, nf))

        a = np.asarray(vals, dtype=np.float64)
        # batch scrubs non-finites AFTER the float32 cast
        a[~np.isfinite(a.astype(np.float32))] = np.nan
        return a

    # ------------------------------------------------------------- per combo
    def _combo_row(self, cb, c, t, n, nf):
        Nf = cb.Nf
        M = cb.M

        # ---- candidate scan -------------------------------------------------
        # k == batch's np.searchsorted(needs, t+1, side="right"); nodes >= k are
        # batch's `if need > n: continue` rows, i.e. NaN -> -1.0 in the score
        # column.  Every candidate statistic is a cumsum difference, gathered
        # for all k live nodes in one shot.
        k = bisect_right(cb.needs, n)
        A = cb.A
        A.fill(-1.0)
        Zc = ()
        if k:
            Ms = cb.Marr[:k]
            Mf = cb.Mfl[:k]
            lo = n - Ms
            cm = c[lo]
            v = (c[n] - cm) / Mf
            if cb.two:
                v = v - (cm - c[lo - Ms]) / Mf
            zz = (v - cb.medarr[:k]) / cb.sdarr[:k]
            Zc = zz.tolist()
            az = np.abs(zz).tolist()
            SZ = cb.SZ
            for j in range(k):
                # (int + 0.5) / N is finite by construction, so batch's
                # `np.where(np.isfinite(U), U, -1.0)` can never fire here.
                A[j] = (SZ[j].searchsorted(az[j], "left") + 0.5) / Nf

        # ---- localise: max / argmax / runner-up (numpy's tie rule) ---------
        i1 = int(A.argmax())
        u1 = A[i1]
        A[i1] = -2.0                           # batch: A2 = A.copy(); A2[i1] = -2
        i2 = int(A.argmax())
        u2 = float(A[i2])
        ok = u1 >= 0.0

        if ok:
            lm = cb.log2M[i1]
            rel = 1.0 - float(M[i1]) / nf
            gap = (float(u1) - u2) if u2 >= 0.0 else np.nan
            zc = Zc[i1]
            sz = -CLIP if zc < -CLIP else (CLIP if zc > CLIP else zc)
            u1v = float(u1)
            # ---- multiplicity-matched calibration of the max (k >= 1 here)
            p = cb.CUS[k - 1].searchsorted(u1v, "left") / Nf
            p = 1.0 - p
            q = float(-np.log10(p if p > QFLOOR else QFLOOR))
        else:
            lm = np.nan
            rel = np.nan
            gap = np.nan
            sz = np.nan
            q = np.nan

        # ---- trailing-16 dispersion of log2(m_hat) (inlined _roll_std) ----
        xf = lm if ok else 0.0
        c1, c2, ck = cb.c1, cb.c2, cb.ck
        c1.append(c1[t] + xf)
        c2.append(c2[t] + xf * xf)
        ck.append(ck[t] + (1.0 if ok else 0.0))
        w = STAB_W
        if n < w:
            stab = np.nan
        elif ck[t + 1] - ck[t + 1 - w] >= w:
            s1 = c1[t + 1] - c1[t + 1 - w]
            s2 = c2[t + 1] - c2[t + 1 - w]
            # `mw * mw`, never `mw ** 2`: numpy's `**2` on an array is a squaring
            # multiply, but a Python scalar `**2` goes through libm `pow` and
            # rounds differently (688/500000 random doubles).  This subtraction
            # cancels to ~1e-14 on a near-constant window, so a single ULP there
            # moves the emitted float32 -- see STREAM_PARITY_REPRO.md.
            # (Do NOT name this `q`: that is the calibrated-max output below.)
            mw = s1 / w
            var = s2 / w - mw * mw
            # == np.sqrt(np.maximum(var, 0.0)); the NaN arm cannot fire here
            # (lm is bounded in [3, 9] and w == 16) but is kept faithful.
            stab = sqrt(var) if var > 0.0 else (0.0 if var == var else np.nan)
        else:
            stab = np.nan

        out = [q, sz, lm, rel, gap, stab]

        # ---- the FIXED-offset control family ------------------------------
        for j in cb.fxj:
            if j is None or j >= k:
                out.append(np.nan)
            else:
                zc = Zc[j]
                out.append(-CLIP if zc < -CLIP else (CLIP if zc > CLIP else zc))
        return out

    # ---------------------------------------------------------------- cols
    @property
    def cols(self) -> list[str]:
        if not self._fitted:
            raise RuntimeError("fit_historical() must be called before cols")
        return self._cols
