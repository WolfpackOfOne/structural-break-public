"""Exact incremental port of ``sbr.features.m07_bayes`` (absorbing-state
posterior, BOCPD, e-processes, sequential Bayes factors).

Everything in the batch module is already a forward recursion, so the streaming
form is the natural one and bitwise parity is attainable.  The port rests on
four rules:

1.  **Reuse the batch code for everything history-only.**  Every null object
    (``_PosNull``, ``_MargNull``, ``_AddNull``), the historical likelihood
    matrix, the historical BOCPD path and the block-restart null are produced
    by calling the *batch* functions on the *batch* inputs at
    ``fit_historical`` time.  There is no second implementation to drift.

2.  **Reuse the batch code per step on a one-row array** wherever the batch
    computation is a row-wise reduction (``_fam_lse``, the mixture posterior
    ``P``, ``_signed_surprise``).  Calling them with a length-1 input
    reproduces row ``t`` of the batch call exactly.

3.  **Carry the sequential reductions.**  ``np.cumsum``,
    ``np.maximum.accumulate`` and the absorbing / BOCPD loops accumulate in
    index order, so a carried accumulator is bit-for-bit what batch computes.

4.  **Scalarise only pure +-*/ arithmetic.**  IEEE +, -, *, / are exactly
    reproducible in Python floats, so the per-point likelihood ratios, the
    robust-z and the clips are inlined as scalars.  Anything transcendental
    stays on a numpy array, because ``math.log`` differs from ``np.log`` in the
    last ulp for ~0.3 % of arguments and ``math.log10`` from ``np.log10`` for
    ~20 % (measured).  Conversely the batch kernels ``_absorb_path`` /
    ``_bocpd`` use ``math.exp``/``math.log``/``math.log1p``/``math.lgamma``, so
    the streaming kernels use exactly those.

Three *shape* effects had to be handled explicitly; none of them is an
algorithmic difference:

*   ``ar_filter_causal`` over the whole online array is a BLAS ``dgemv``.  A
    constant-height lag matrix (``_AR_WIN`` rows, maintained by shifting one
    row per step) reproduces the full-array result bitwise, because the row we
    read sits inside a complete 4-row gemv block -- the same trick, and the
    same reason, as ``StreamCtx``.
*   ``X.sum(axis=0)`` / ``X.max(axis=0)`` over a matrix with **>= 8 rows** is
    NOT the same float when the matrix has one column as when it has many:
    numpy collapses the trailing unit axis and switches to pairwise summation
    with 8 partial accumulators.  Every axis-0 reduction here is therefore done
    on a *two-column* buffer (the current column, duplicated), which restores
    the sequential outer-axis reduction batch performs.  Verified for
    R = 4, 6, 8, 10, 16, 18.
*   ``P @ meta[:, 0]`` is a ``dgemv`` whose per-row accumulation order depends
    on the number of rows in the call: rows inside a complete 4-row block use
    the blocked kernel, the trailing ``n % 4`` rows use a scalar remainder
    path.  We evaluate it on a 4-row tile, i.e. the blocked-kernel value, which
    is what batch produces for every row except possibly the last ``n % 4``.
    (That makes ``ab_post_lvr`` / ``ab_post_rho`` the one place where the batch
    module is itself not prefix-invariant in float64 -- by one ulp.  It has
    never been observed to survive the float32 cast; see the tests.)

Cost per observation is O(K + R) with K = 18 mixture components and R = 80
BOCPD run-length buckets: bounded, and independent of t.

``fit_historical`` is dominated by three loops that ARE the batch algorithm --
the block-restarted absorbing null (``n_blocks x 256 x K`` log-add-exp
updates), the historical BOCPD path (``n_hist x R``) and two historical
absorbing paths (``2 x n_hist x K``).  They are re-expressed here in pure
Python floats rather than ``np.float64``, with loop-invariant subexpressions
hoisted and the k-loop lifted outside the t-loop; every one of those
re-expressions was fuzz-verified against the batch kernel at atol=0 BEFORE
adoption.  numpy vectorisation of these recursions is NOT available: ``np.exp``
disagrees with ``math.exp`` in the last ulp on ~4.8 % of arguments in the
relevant domain (measured), and the batch kernels use ``math``.
"""
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from itertools import islice

import numpy as np
from scipy.special import ndtri

from sbr.features.m07_bayes import (
    AR_ORDER,
    ARCH_GRID,
    BET_C,
    BLK_STRIDE,
    BO_AL0,
    BO_HAZ,
    BO_HIST_MAX,
    BO_KAP0,
    F_ARCH,
    F_DEP,
    F_LOC,
    F_VAR,
    GCLIP,
    GRAPA_K0,
    HAZ_FAST,
    HAZ_SLOW,
    MU_GRID,
    POS_GRID,
    POW_EPS,
    R_MAX,
    RHO_GRID,
    VAR_GRID,
    ZCAP,
    _AddNull,
    _bocpd_ct,
    _fam_lse,
    _fam_norm,
    _llr_matrix,
    _MargNull,
    _mix_bet,
    _PosNull,
    _signed_surprise,
)
from sbr.transforms import _ar_resid, _fit_ar

#: rows of the incrementally maintained AR(6) lag matrix.  Any value that
#: keeps the row inside a complete 4-row BLAS gemv block reproduces the
#: full-array result bitwise (verified for 4, 6, 8, 12, 16).
_AR_WIN = 4
#: z-buffer length; must be >= _AR_WIN + AR_ORDER.
_AR_PAD = _AR_WIN + AR_ORDER + 2

_HICAP = {"tail": 1.0, "disp": 0.25, "dep": 1.0, "vc": 1.0}
_KEYS = ("tail", "disp", "dep", "vc")

#: precomputed per-component constants of ``_llr_matrix`` (data independent).
_C_VAR = tuple((-0.5 * math.log(v), 1.0 - 1.0 / v) for v in VAR_GRID)
_C_MU = tuple((m, 0.5 * m * m) for m in MU_GRID)
_C_RHO = tuple((r, 1.0 - r * r, -0.5 * math.log(1.0 - r * r)) for r in RHO_GRID)
_C_ARCH = tuple(ARCH_GRID)
_NK = len(_C_VAR) + len(_C_MU) + len(_C_RHO) + len(_C_ARCH)

COLS = [
    "ab_lpo", "ab_lpo_z", "ab_lpo_sur", "ab_var", "ab_var_z", "ab_dep",
    "ab_dep_z", "ab_loc_z", "ab_post_lvr", "ab_post_rho", "ab_pfam_var",
    "ab_pfam_dep", "ab_lpo_rel", "ab_lpo_slp", "ab_lpo_pk_z", "ab_fast",
    "ab_fast_z", "ab_fast_slow", "abz_lpo", "abz_lpo_z", "abz_var_z",
    "bo_p_lt10", "bo_p_lt25", "bo_mean_rel", "bo_ent", "bo_nochange",
    "bo_lo_change", "bo_lo_change_z", "bo_p_lt25_z", "bo_ent_z",
    "bo_lo_change_pk", "ev_tail_mix", "ev_tail_z", "ev_tail_ad", "ev_disp_mix",
    "ev_disp_z", "ev_disp_ad", "ev_dep_mix", "ev_dep_z", "ev_vc_z",
    "ev_pow_mix", "ev_pow_z", "ev_pow_pk", "bf0_all", "bf0_all_z", "bf0_rate",
    "bf0_var", "bf0_var_z", "xb_max_z", "xb_n_hot",
]


# ------------------------------------------------------------------- scalars
def _runmax(prev, x, t):
    """``np.maximum.accumulate`` semantics (NaN absorbing), incrementally."""
    if t == 0:
        return x
    if x != x or prev != prev:
        return float("nan")
    return x if x > prev else prev


def _rz_s(v, med, sd):
    """Scalar form of ``m07_bayes._rz`` (exact: only -, /, max, clip)."""
    x = (v - med) / (sd if sd > 1e-9 or sd != sd else 1e-9)
    if x < -ZCAP:
        return -ZCAP
    if x > ZCAP:
        return ZCAP
    return x


def _argmin_grid(lt, lg):
    """``np.argmin(np.abs(lt - lg))`` -- first minimum wins, as numpy does."""
    best = abs(lt - lg[0])
    bj = 0
    for j in range(1, len(lg)):
        d = abs(lt - lg[j])
        if d < best:
            best = d
            bj = j
    return bj


def _llr_row(g, gp, out, arch_buf):  # noqa: C901
    """Row ``t`` of ``_llr_matrix``: bitwise identical, scalar arithmetic.

    Only the two ARCH components need a logarithm of a data-dependent value;
    they are done with a single ``np.log`` on a length-2 array, which numpy
    evaluates elementwise and therefore identically to the batch call on the
    length-n array.
    """
    g2 = g * g
    h = 0.5 * g2
    i = 0
    for c, k in _C_VAR:
        v = c + h * k
        out[i] = -40.0 if v < -40.0 else (40.0 if v > 40.0 else v)
        i += 1
    for m, c in _C_MU:
        v = m * g - c
        out[i] = -40.0 if v < -40.0 else (40.0 if v > 40.0 else v)
        i += 1
    for r, s2, c in _C_RHO:
        d = g - r * gp
        v = c - 0.5 * d * d / s2 + h
        out[i] = -40.0 if v < -40.0 else (40.0 if v > 40.0 else v)
        i += 1
    gpm = gp * gp - 1.0
    for j, a in enumerate(_C_ARCH):
        w = 1.0 + a * gpm
        arch_buf[j] = w if w > 0.05 else 0.05
    lg = np.log(arch_buf)
    for j in range(len(_C_ARCH)):
        w = arch_buf[j]
        v = -0.5 * lg[j] + h * (1.0 - 1.0 / w)
        out[i] = -40.0 if v < -40.0 else (40.0 if v > 40.0 else v)
        i += 1
    return out


def _absorb_path_fast(llr, log_h, log1mh):
    """Bitwise re-expression of ``_absorb_path`` in pure Python floats.

    Same arithmetic, same order, same ``math`` calls; only the container
    changes (numpy scalar indexing + ``np.float64`` ops -> list indexing +
    Python floats, which are the identical IEEE doubles).  The k-loop is
    hoisted outside the t-loop, which is legal because the recursion is
    independent across k.  Fuzz-verified against ``_absorb_path`` at atol=0.
    """
    K = llr.shape[1]
    llrT = llr.T.tolist()
    exp = math.exp
    log1p = math.log1p
    outT = []
    for k in range(K):
        a = -1.0e300
        col = []
        ap = col.append
        for x in llrT[k]:
            if a > log_h:
                m = a
                d = log_h - a
            else:
                m = log_h
                d = a - m
            s = m if d < -700.0 else m + log1p(exp(d))
            a = s + x - log1mh
            ap(a)
        outT.append(col)
    return np.array(outT).T.copy()


def _absorb_blocks_fast(llr, log_h, log1mh, logw, fam, famnorm, offs, pos, fams):
    """Bitwise re-expression of ``_absorb_blocks``; returns {family: (nb, npos)}.

    Identical to ``_absorb_blocks[:, :, f]`` for every requested family.  Three
    changes, none of them arithmetic: Python floats instead of ``np.float64``;
    the k-loop hoisted outermost so the accumulator lives in a local instead of
    a numpy slot; and the family membership test hoisted out of the innermost
    loop into a precomputed ascending index list (the k order, and therefore
    the summation order, is unchanged).  Only the families actually queried are
    materialised -- family ``F_ARCH`` is computed by the batch module but never
    read by it.  Fuzz-verified against ``_absorb_blocks`` at atol=0.
    """
    K = llr.shape[1]
    llrT = llr.T.tolist()
    logwl = logw.tolist()
    faml = fam.tolist()
    fnl = famnorm.tolist()
    posl = [int(v) for v in pos]
    offsl = [int(v) for v in offs]
    nb = len(offsl)
    npos = len(posl)
    lmax = posl[-1]
    exp = math.exp
    log1p = math.log1p
    log = math.log
    snap = [[[0.0] * K for _ in range(npos)] for _ in range(nb)]
    for k in range(K):
        ck = llrT[k]
        for b in range(nb):
            seg = ck[offsl[b]:offsl[b] + lmax]
            sb = snap[b]
            a = -1.0e300
            idx = 0
            for pi in range(npos):
                p = posl[pi]
                for x in seg[idx:p]:
                    if a > log_h:
                        m = a
                        d = log_h - a
                    else:
                        m = log_h
                        d = a - m
                    s = m if d < -700.0 else m + log1p(exp(d))
                    a = s + x - log1mh
                idx = p
                sb[pi][k] = a
    out = {}
    for f in fams:
        sel = [k for k in range(K) if f == 0 or faml[k] == f]
        wsel = [logwl[k] for k in sel]
        fn = fnl[f]
        arr = np.empty((nb, npos), dtype=np.float64)
        for b in range(nb):
            sb = snap[b]
            vals = []
            for pi in range(npos):
                pr = sb[pi]
                mx = -1.0e300
                vs = []
                va = vs.append
                for k, w in zip(sel, wsel):
                    v = w + pr[k]
                    va(v)
                    if v > mx:
                        mx = v
                ss = 0.0
                for v in vs:
                    ss += exp(v - mx)
                vals.append(mx + log(ss) - fn)
            arr[b] = vals
        out[f] = arr
    return out


def _absorb_step2(pa, pb, llr_row, log_ha, l1a, log_hb, l1b):
    """Two independent ``_absorb_path`` steps sharing one k-loop.

    The recursion is independent across k and across hazard, so folding the
    slow- and fast-hazard updates into a single pass changes only loop
    control.  Both accumulators are updated in place.
    """
    for k in range(len(pa)):
        x = llr_row[k]
        a = pa[k]
        if a > log_ha:
            m = a
            d = log_ha - a
        else:
            m = log_ha
            d = a - m
        s = m if d < -700.0 else m + math.log1p(math.exp(d))
        pa[k] = s + x - l1a
        a = pb[k]
        if a > log_hb:
            m = a
            d = log_hb - a
        else:
            m = log_hb
            d = a - m
        s = m if d < -700.0 else m + math.log1p(math.exp(d))
        pb[k] = s + x - l1b
    return pa


def _absorb_step(prev, llr_row, log_h, log1mh):
    """One step of ``_absorb_path``; ``prev`` is updated in place."""
    for k in range(len(prev)):
        a = prev[k]
        if a > log_h:
            m = a
            d = log_h - a
        else:
            m = log_h
            d = a - m
        if d < -700.0:
            s = m
        else:
            s = m + math.log1p(math.exp(d))
        prev[k] = s + llr_row[k] - log1mh
    return prev


def _fam_lse_row(L, sel, logw_sel, fnorm_f):
    """``_fam_lse`` for a single row, with the mask work hoisted to fit time.

    Identical numpy expression on an identically shaped row: for ``f == 0`` the
    batch mask is all-True, so ``L[:, sel]`` is a copy of ``L`` and dropping it
    changes nothing.
    """
    A = L + logw_sel[None, :] if sel is None else L[:, sel] + logw_sel[None, :]
    m = A.max(axis=1)
    return float((m + np.log(np.exp(A - m[:, None]).sum(axis=1)) - fnorm_f)[0])


class _BocpdStream:
    """Incremental Adams-MacKay run-length posterior; bitwise == ``_bocpd``.

    Same scalar ``math`` arithmetic in the same order as the batch kernel, with
    four purely mechanical changes (all fuzz-verified at atol=0 against
    ``_bocpd`` before adoption):

    * Python floats and lists instead of numpy arrays and ``np.float64``.
    * ``0.5 * (nu[r] + 1.0)``, ``kap[r] + 1.0`` and ``2.0 * (kap[r] + 1.0)``
      are loop invariants, hoisted into constant tables.
    * ``lp[r] + lpred[r]`` is computed once and reused: the batch recomputes it
      in the normaliser and again as ``nlp[r] = lp[src] + lpred[src] + log1mh``,
      and ``a + b + c`` is left-associative, so the cached sum is the same
      float.
    * the "renormalise" and "summarise" passes are merged; pass 2 reads only
      ``lp[r]`` which pass 1 has already written for that same r.

    The live lists are exactly ``top + 1`` long: the batch keeps full-length
    R arrays whose tail entries are stale and never read.
    """

    __slots__ = ("R", "kap", "nu", "ct", "sf", "hnu", "kp1", "kp2",
                 "mu", "be", "lp", "t", "mu0", "be0", "log_h", "log1mh")

    def __init__(self, mu0, be0, kap0, al0, log_h, log1mh, R):
        self.R = R
        self.mu0 = float(mu0)
        self.be0 = float(be0)
        self.log_h = log_h
        self.log1mh = log1mh
        self.kap = [0.0] * R
        self.nu = [0.0] * R
        self.ct = [0.0] * R
        self.sf = [0.0] * R
        self.hnu = [0.0] * R
        self.kp1 = [0.0] * R
        self.kp2 = [0.0] * R
        # shared with the batch kernel: numba's lgamma and CPython's disagree by
        # up to 512 ULP on this grid, so the table must come from one source or
        # the batch-trained and stream-served features sit on different constants
        ct = _bocpd_ct(al0, R)
        for r in range(R):
            kp = kap0 + r
            al = al0 + 0.5 * r
            nu = 2.0 * al
            self.kap[r] = kp
            self.nu[r] = nu
            self.ct[r] = float(ct[r])
            self.sf[r] = (kp + 1.0) / (al * kp)
            self.hnu[r] = 0.5 * (nu + 1.0)
            self.kp1[r] = kp + 1.0
            self.kp2[r] = 2.0 * (kp + 1.0)
        self.reset()

    def reset(self):
        self.lp = [0.0]
        self.mu = [self.mu0]
        self.be = [self.be0]
        self.t = -1

    def step(self, xt):
        R = self.R
        log_h = self.log_h
        log1mh = self.log1mh
        lp, mu, be = self.lp, self.mu, self.be
        kap, nu, ct, sf = self.kap, self.nu, self.ct, self.sf
        hnu, kp1, kp2 = self.hnu, self.kp1, self.kp2
        exp = math.exp
        log = math.log
        log1p = math.log1p
        self.t += 1
        t = self.t
        top = len(lp) - 1

        # predictive likelihood of every live run length, and its max
        vs = []
        va = vs.append
        mx = -1.0e300
        for lpr, ctr, sfr, nur, hnur, mur, ber in zip(lp, ct, sf, nu, hnu, mu, be):
            s2 = ber * sfr
            d = xt - mur
            v = lpr + (ctr - 0.5 * log(s2) - hnur * log1p(d * d / (nur * s2)))
            va(v)
            if v > mx:
                mx = v
        ss = 0.0
        for v in vs:
            ss += exp(v - mx)
        lcp = mx + log(ss) + log_h

        rm1 = R - 1
        newtop = top + 1
        if newtop > rm1:
            newtop = rm1
        nlp = [lcp]
        nmu = [self.mu0]
        nbe = [self.be0]
        an = nlp.append
        am = nmu.append
        ab = nbe.append
        for vv, kr, mur, ber, k1, k2 in islice(
                zip(vs, kap, mu, be, kp1, kp2), newtop):
            an(vv + log1mh)
            dm = xt - mur
            ab(ber + kr * dm * dm / k2)
            am((kr * mur + xt) / k1)
        if newtop == rm1 and top == rm1:
            # the truncated bucket absorbs the mass that would reach r = R
            v = nlp[rm1]
            b = vs[rm1] + log1mh
            if b > v:
                v = b + log1p(exp(v - b))
            else:
                v = v + log1p(exp(b - v))
            nlp[rm1] = v

        mx = -1.0e300
        for v in nlp:
            if v > mx:
                mx = v
        ss = 0.0
        for v in nlp:
            ss += exp(v - mx)
        lse = mx + log(ss)

        p10 = p25 = p50 = mean = ent = 0.0
        best = -1.0
        mode = 0
        r = 0
        p = 0.0
        newlp = []
        ap = newlp.append
        for v in nlp:
            lpr = v - lse
            ap(lpr)
            p = exp(lpr)
            if r < 10:
                p10 += p
            if r < 25:
                p25 += p
            if r < 50:
                p50 += p
            mean += r * p
            if p > 1e-300:
                ent -= p * lpr
            if p > best:
                best = p
                mode = r
            r += 1
        self.lp = newlp
        self.mu = nmu
        self.be = nbe

        pnc = p                                   # == exp(lp[top])
        rel = 1.0 / (t + 1.0)
        lo = log(max(1.0 - pnc, 1e-12)) - log(max(pnc, 1e-12))
        if lo > 30.0:
            lo = 30.0
        elif lo < -30.0:
            lo = -30.0
        return (p10, p25, p50, mode * rel, mean * rel, ent, pnc, lo)


class _LenNull:
    """Scalar view of a length-matched null (``_PosNull`` or ``_AddNull``).

    Holds the same ``med`` / ``sd`` / sorted-sample tables the batch object
    holds, as Python lists so the per-step query is pure scalar work; ``gi``
    indexes the shared list of distinct log-length grids.
    """

    __slots__ = ("gi", "med", "sd", "srt", "n", "sur")

    def __init__(self, obj, gi, pos):
        self.gi = gi
        self.med = np.asarray(obj.med, dtype=np.float64).tolist()
        self.sd = np.asarray(obj.sd, dtype=np.float64).tolist()
        if pos:
            self.srt = [obj.srt[:, k].tolist() for k in range(obj.srt.shape[1])]
            self.n = obj.n
            # pct can only take the 2n+1 values 0.5*k/n, so the surprise of
            # every reachable percentile is tabulated once with the batch
            # helper itself (this caches its RESULT, it does not re-derive it).
            self.sur = _signed_surprise(
                0.5 * np.arange(0, 2 * obj.n + 1, dtype=np.float64) / obj.n).tolist()
        else:
            self.srt = None
            self.n = 0

    def z(self, v, j):
        return _rz_s(v, self.med[j], self.sd[j])

    def pct_k(self, v, j):
        """``2 * n * _PosNull.pct``; bisect == np.searchsorted (fuzzed)."""
        col = self.srt[j]
        if v != v:                       # np.searchsorted puts NaN last
            return 2 * self.n
        return bisect_left(col, v) + bisect_right(col, v)


def _fam_from_A(A, sel, fn):
    """``_fam_lse`` for one row, given ``A = L + logw`` already materialised.

    ``A[:, sel] == L[:, sel] + logw[sel][None, :]`` elementwise, so this is the
    same numpy expression on the same floats.  ``sel is None`` is the batch's
    ``f == 0`` case, whose mask is all-True and whose fancy-index copy is
    therefore a no-op.  ``keepdims`` removes the ``m[:, None]`` view and the
    final ``m + log(s) - fn`` is done in Python floats -- the same three IEEE
    operations in the same order.  Fuzz-verified against ``_fam_lse`` at
    atol=0.
    """
    Af = A if sel is None else A[:, sel]
    m = Af.max(axis=1, keepdims=True)
    s = np.exp(Af - m).sum(axis=1)
    return float(m[0, 0]) + float(np.log(s)[0]) - fn


# ---------------------------------------------------------------------- module
class StreamM07Bayes:
    """Streaming twin of ``sbr.features.m07_bayes``."""

    MODULE = "m07_bayes"

    def __init__(self):
        self._fitted = False

    # ------------------------------------------------------------------- fit
    def fit_historical(self, ctx) -> None:
        hp = ctx.hp
        zh = (np.asarray(ctx.hist, dtype=np.float64) - hp.mu) / hp.sd
        self._zh_last = float(zh[-1]) if zh.shape[0] else 0.0

        coef = _fit_ar(zh, AR_ORDER)
        eh = _ar_resid(zh, coef)
        if eh.shape[0] < 200:
            eh = zh
            coef = np.zeros(AR_ORDER)
        sig = max(float(eh.std(ddof=1)), 1e-9)
        eh = eh / sig
        self.coef = coef
        self.sig = sig

        self._srt = np.sort(eh)
        self._H = self._srt.shape[0]

        gh = self._ns(eh)
        gh_prev = np.concatenate([[0.0], gh[:-1]])
        self._gh_last = float(gh[-1])

        llr_h, logw, fam, meta = _llr_matrix(gh, gh_prev)
        assert llr_h.shape[1] == _NK
        self.logw, self.fam, self.meta = logw, fam, meta
        self._meta0 = meta[:, 0]
        self._meta1 = meta[:, 1]
        self._sel_var = fam == F_VAR
        self._sel_pfdep = (fam == F_DEP) | (fam == F_ARCH)
        self.fnorm = _fam_norm(logw, fam)
        self._fsel = {}
        for f in (0, F_VAR, F_LOC, F_DEP, F_ARCH):
            sel = None if f == 0 else (fam == f)
            self._fsel[f] = (sel, logw if sel is None else logw[sel],
                             self.fnorm[f])
        self.K = llr_h.shape[1]
        self.lh_s, self.l1_s = math.log(HAZ_SLOW), math.log1p(-HAZ_SLOW)
        self.lh_f, self.l1_f = math.log(HAZ_FAST), math.log1p(-HAZ_FAST)

        self._grids = []          # distinct log-length grids, shared argmin

        # length-matched null for the absorbing recursion
        Hh = llr_h.shape[0]
        pos = POS_GRID[POS_GRID <= max(Hh // 3, 8)]
        if pos.size < 3:
            pos = np.array([1, 2, 4])
        lmax = int(pos[-1])
        offs = np.arange(0, Hh - lmax + 1, BLK_STRIDE, dtype=np.int64)
        if offs.shape[0] < 12:
            offs = np.arange(0, max(Hh - lmax + 1, 1), 1, dtype=np.int64)
        used = (0, F_VAR, F_LOC, F_DEP)       # F_ARCH is never queried
        BK = _absorb_blocks_fast(llr_h, self.lh_s, self.l1_s, logw, fam,
                                 self.fnorm, offs, pos.astype(np.int64), used)
        pn = {f: _PosNull(BK[f], pos) for f in used}
        self._slope_scale = float(np.median(pn[0].sd))
        self.pn = {f: _LenNull(pn[f], self._grid_id(pn[f].lpos), True)
                   for f in used}

        # fast-hazard marginal null
        mn_f = _MargNull(_fam_lse(_absorb_path_fast(llr_h, self.lh_f, self.l1_f),
                                  logw, fam, self.fnorm, 0))
        self._mn_f = (mn_f.med, mn_f.sd)

        # raw standardised channel
        zprev_h = np.concatenate([[0.0], zh[:-1]])
        llr_zh, _, _, _ = _llr_matrix(zh, zprev_h)
        Lzh = _absorb_path_fast(llr_zh, self.lh_s, self.l1_s)
        m = _MargNull(_fam_lse(Lzh, logw, fam, self.fnorm, 0))
        self._mn_z = (m.med, m.sd)
        m = _MargNull(_fam_lse(Lzh, logw, fam, self.fnorm, F_VAR))
        self._mn_zv = (m.med, m.sd)

        # BOCPD prior and historical path
        lbh, lb1 = math.log(BO_HAZ), math.log1p(-BO_HAZ)
        v_g = max(float(gh.var()), 1e-6)
        be0 = (BO_AL0 - 1.0) * v_g
        mu0 = float(gh.mean())
        self._bo_args = (mu0, be0, lbh, lb1)
        ghh = gh[-BO_HIST_MAX:] if gh.shape[0] > BO_HIST_MAX else gh
        bo = _BocpdStream(mu0, be0, BO_KAP0, BO_AL0, lbh, lb1, R_MAX)
        BH = np.array([bo.step(v) for v in ghh.tolist()], dtype=np.float64)
        m = _MargNull(BH[:, 7])
        self._mn_bo7 = (m.med, m.sd)
        m = _MargNull(BH[:, 1])
        self._mn_bo1 = (m.med, m.sd)
        m = _MargNull(BH[:, 5])
        self._mn_bo5 = (m.med, m.sd)

        # e-processes
        u_h = np.asarray(ctx.hist_tr["u"], dtype=np.float64)
        cu_h = u_h - 0.5
        up_h = np.concatenate([[0.0], cu_h[:-1]])
        self._cu_h_last = float(cu_h[-1])
        payoffs_h = {
            "tail": (np.abs(cu_h) > 0.4).astype(np.float64),
            "disp": cu_h * cu_h,
            "dep": (cu_h * up_h > 0).astype(np.float64),
            "vc": ((np.abs(cu_h) > 0.3) & (np.abs(up_h) > 0.3)).astype(np.float64),
        }
        nb = BET_C.shape[0]
        self.lw_bet = np.full(nb, -math.log(nb))
        # all four betting channels share ``lw_bet`` and the same 6 bets, so
        # they are evaluated as one (nb, 8) block: columns 2i, 2i+1 are key i
        # (duplicated, because a single-column axis-0 reduction is not the same
        # float -- see the module docstring).
        self._lam8 = np.empty((nb, 8), dtype=np.float64)
        self._m0 = []
        self._bet_null = []
        self._grapa = []
        for i, key in enumerate(_KEYS):
            h_h = payoffs_h[key]
            hi = _HICAP[key]
            m0 = float(np.clip(h_h.mean(), 1e-3, hi - 1e-3))
            inc_h, lam = _mix_bet(h_h, m0, hi)
            an = _AddNull(inc_h, self.lw_bet, h_h.shape[0])
            self._lam8[:, 2 * i] = lam
            self._lam8[:, 2 * i + 1] = lam
            self._m0.append(m0)
            self._bet_null.append(_LenNull(an, self._grid_id(an.lwg), False))
            if key in ("tail", "disp"):
                self._grapa.append((i, m0, max(float(h_h.var()), 1e-6),
                                    -0.9 / max(hi - m0, 1e-6),
                                    0.9 / max(m0, 1e-6)))

        p_h = np.clip(2.0 * np.minimum(u_h, 1.0 - u_h), 1e-6, 1.0)
        rows_h = []
        for e in POW_EPS:
            rows_h.append(math.log(e) + (e - 1.0) * np.log(p_h))
            rows_h.append(math.log(e) + (e - 1.0) * np.log(np.clip(1.0 - p_h, 1e-6, 1.0)))
        Rh = np.asarray(rows_h)
        self.lw_p = np.full(Rh.shape[0], -math.log(Rh.shape[0]))
        self._npow = Rh.shape[0]
        an = _AddNull(Rh, self.lw_p, p_h.shape[0])
        self._an_p = _LenNull(an, self._grid_id(an.lwg), False)
        self._pow_c = tuple((math.log(e), e - 1.0) for e in POW_EPS)

        # sequential Bayes-factor nulls
        an = _AddNull(llr_h.T, logw, Hh)
        self._an_bf = _LenNull(an, self._grid_id(an.lwg), False)
        an = _AddNull(llr_h[:, self._sel_var].T, logw[self._sel_var], Hh)
        self._an_bfv = _LenNull(an, self._grid_id(an.lwg), False)

        # ---- scalar-path tables: caches of the batch expressions' results
        self._srtl = self._srt.tolist()
        self._Hp1 = self._H + 1.0
        self._logw2 = np.ascontiguousarray(logw[None, :])
        self._fn0 = float(self.fnorm[0])
        self._famq = tuple((fam == f, float(self.fnorm[f]))
                           for f in (F_VAR, F_DEP, F_LOC))
        self._fvar_sel = fam == F_VAR
        self._fvar_fn = float(self.fnorm[F_VAR])
        self._lwbet2 = np.ascontiguousarray(self.lw_bet[:, None])
        self._lwp2 = np.ascontiguousarray(self.lw_p[:, None])
        self._slope_scale_c = max(self._slope_scale, 1e-6)
        # nearest-log-length grid index, tabulated with the identical
        # expression `_PosNull._idx` / `_AddNull._idx` evaluate per step
        # (exhaustively verified equal for every t in 1..4096).
        Ls = np.arange(1, 4097, dtype=np.float64)
        lL = np.log(np.maximum(Ls, 1.0))[:, None]
        self._jtab = [np.argmin(np.abs(lL - gg[None, :]), axis=1).tolist()
                      for gg in self._grids]

        self._reset_online(zh)
        self._fitted = True

    def _grid_id(self, lg):
        key = np.asarray(lg, dtype=np.float64)
        for i, g in enumerate(self._grids):
            if g.shape == key.shape and np.array_equal(g, key):
                return i
        self._grids.append(key)
        return len(self._grids) - 1

    def _reset_online(self, zh):
        buf = np.zeros(_AR_PAD, dtype=np.float64)
        m = min(_AR_PAD, zh.shape[0])
        if m:
            buf[_AR_PAD - m:] = zh[-m:]
        self._zbuf = buf
        # AR(6) lag matrix, maintained by shifting one row per step.  Only its
        # last row feeds the emitted residual; the leading rows exist solely to
        # keep the gemv inside a complete 4-row BLAS block.
        X = np.empty((_AR_WIN, AR_ORDER), dtype=np.float64)
        for r in range(_AR_WIN):
            c = _AR_PAD - _AR_WIN + r
            X[r] = buf[c - 1:c - 1 - AR_ORDER:-1]
        self._X = X

        K = self.K
        self._L = [-1.0e300] * K
        self._Lf = [-1.0e300] * K
        self._Lz = [-1.0e300] * K
        self._acc_llr = np.zeros((1, K), dtype=np.float64)

        self._prev_g = self._gh_last
        self._prev_z = self._zh_last
        self._prev_cu = self._cu_h_last

        self._pk_lpo = -np.inf
        self._pk_pow = -np.inf
        self._pk_bo7 = -np.inf
        self._lpo_hist = []

        mu0, be0, lbh, lb1 = self._bo_args
        self._bo = _BocpdStream(mu0, be0, BO_KAP0, BO_AL0, lbh, lb1, R_MAX)

        nb = BET_C.shape[0]
        self._cs_bet = np.zeros((nb, 8), dtype=np.float64)
        self._cs_pow = np.zeros((self._npow, 2), dtype=np.float64)
        self._gr_c = [0.0, 0.0]
        self._gr_s = [0.0, 0.0]

        # scratch buffers (allocated once)
        self._one = np.empty(1, dtype=np.float64)
        self._two = np.empty(2, dtype=np.float64)
        self._arch = np.empty(len(_C_ARCH), dtype=np.float64)
        self._llr1 = np.empty((1, K), dtype=np.float64)
        self._llrl = [0.0] * K
        self._llrzl = [0.0] * K
        self._outl = [0.0] * 50
        self._Lbuf = np.empty((1, K), dtype=np.float64)
        self._tile4 = np.empty((4, K), dtype=np.float64)
        self._bet_in = np.empty((nb, 8), dtype=np.float64)
        self._hm8 = np.empty(8, dtype=np.float64)
        self._gr2 = np.empty(2, dtype=np.float64)
        self._pow_in = np.empty((self._npow, 2), dtype=np.float64)
        self._Z = [0.0] * 7

    # ---------------------------------------------------------------- helpers
    def _ns(self, x):
        srt, H = self._srt, self._H
        lo = np.searchsorted(srt, x, side="left")
        hi = np.searchsorted(srt, x, side="right")
        return np.clip(ndtri((0.5 * (lo + hi) + 0.5) / (H + 1.0)), -GCLIP, GCLIP)

    @property
    def cols(self):
        return list(COLS)

    # ------------------------------------------------------------------ step
    def step(self, ctx) -> np.ndarray:
        t = ctx.t
        jt = self._jtab
        if t < 4096:
            js = [tab[t] for tab in jt]
        else:                                    # pragma: no cover
            one = self._one
            one[0] = t + 1.0
            lt = float(np.log(np.maximum(one, 1.0))[0])
            js = [_argmin_grid(lt, gg) for gg in self._grids]
        o = self._outl

        # ---- streams -----------------------------------------------------
        z = ctx.tr["mean"][t]
        buf = self._zbuf
        buf[:-1] = buf[1:]
        buf[-1] = z
        X = self._X
        X[:-1] = X[1:]
        X[-1] = buf[_AR_PAD - 2:_AR_PAD - 2 - AR_ORDER:-1]
        zf = float(z)
        ev = (zf - float((X @ self.coef)[-1])) / self.sig
        if ev != ev:                             # np.searchsorted puts NaN last
            lo = hi = self._H
        else:
            srtl = self._srtl
            lo = bisect_left(srtl, ev)
            hi = bisect_right(srtl, ev)
        g = float(ndtri((0.5 * (lo + hi) + 0.5) / self._Hp1))
        if g < -GCLIP:
            g = -GCLIP
        elif g > GCLIP:
            g = GCLIP

        gl = self._llrl
        _llr_row(g, self._prev_g, gl, self._arch)

        # ---- absorbing posterior, slow hazard ----------------------------
        Lrow = self._Lbuf
        Lrow[0] = _absorb_step2(self._L, self._Lf, gl,
                                self.lh_s, self.l1_s, self.lh_f, self.l1_f)
        # A, its row max and the exponentials are shared between the mixture
        # log Bayes factor and the component posterior: batch computes both
        # from the same `L + logw`, so this is one evaluation instead of two.
        A = Lrow + self._logw2
        mk = A.max(axis=1, keepdims=True)
        ex = np.exp(A - mk)
        sk = ex.sum(axis=1, keepdims=True)
        v = float(mk[0, 0]) + float(np.log(sk)[0, 0]) - self._fn0
        P = ex / sk

        tile = self._tile4
        tile[:] = P
        o[8] = float((tile @ self._meta0)[-1])
        o[9] = float((tile @ self._meta1)[-1])
        o[10] = float(P[:, self._sel_var].sum(axis=1)[0])
        o[11] = float(P[:, self._sel_pfdep].sum(axis=1)[0])

        (sv, fv), (sd_, fd), (sl, fl) = self._famq
        f_var = _fam_from_A(A, sv, fv)
        f_dep = _fam_from_A(A, sd_, fd)
        f_loc = _fam_from_A(A, sl, fl)

        pn = self.pn
        pn0 = pn[0]
        j0 = js[pn0.gi]
        med0, sd0 = pn0.med, pn0.sd
        o[0] = v
        z_lpo0 = _rz_s(v, med0[j0], sd0[j0])
        o[1] = z_lpo0
        o[2] = pn0.sur[pn0.pct_k(v, j0)]
        o[3] = f_var
        q = pn[F_VAR]
        o[4] = _rz_s(f_var, q.med[js[q.gi]], q.sd[js[q.gi]])
        o[5] = f_dep
        q = pn[F_DEP]
        o[6] = _rz_s(f_dep, q.med[js[q.gi]], q.sd[js[q.gi]])
        q = pn[F_LOC]
        o[7] = _rz_s(f_loc, q.med[js[q.gi]], q.sd[js[q.gi]])

        pk = self._pk_lpo = _runmax(self._pk_lpo, v, t)
        av = -pk if pk < 0.0 else pk
        o[12] = (v - pk) / (av if av > 1.0 else 1.0)
        hb = self._lpo_hist
        hb.append(v)
        if len(hb) > 17:
            hb.pop(0)
        if t >= 16:
            sl_ = (v - hb[0]) / (16 * self._slope_scale_c)
            o[13] = -ZCAP if sl_ < -ZCAP else (ZCAP if sl_ > ZCAP else sl_)
        else:
            o[13] = np.nan
        o[14] = _rz_s(pk, med0[j0], sd0[j0])

        # ---- absorbing posterior, fast hazard ----------------------------
        Lrow[0] = self._Lf
        lpo_f = _fam_from_A(Lrow + self._logw2, None, self._fn0)
        o[15] = lpo_f
        o[16] = _rz_s(lpo_f, *self._mn_f)
        o[17] = lpo_f - v

        # ---- absorbing posterior on the raw standardised stream ----------
        gz = self._llrzl
        _llr_row(zf, self._prev_z, gz, self._arch)
        Lrow[0] = _absorb_step(self._Lz, gz, self.lh_s, self.l1_s)
        Az = Lrow + self._logw2
        zl = _fam_from_A(Az, None, self._fn0)
        o[18] = zl
        o[19] = _rz_s(zl, *self._mn_z)
        o[20] = _rz_s(_fam_from_A(Az, self._fvar_sel, self._fvar_fn), *self._mn_zv)

        # ---- BOCPD -------------------------------------------------------
        bo = self._bo.step(g)
        b7 = bo[7]
        o[21] = bo[0]
        o[22] = bo[1]
        o[23] = bo[4]
        o[24] = bo[5]
        o[25] = bo[6]
        o[26] = b7
        o[27] = _rz_s(b7, *self._mn_bo7)
        o[28] = _rz_s(bo[1], *self._mn_bo1)
        o[29] = _rz_s(bo[5], *self._mn_bo5)
        self._pk_bo7 = _runmax(self._pk_bo7, b7, t)
        o[30] = self._pk_bo7

        # ---- e-processes -------------------------------------------------
        u = float(ctx.tr["u"][t])
        cu = u - 0.5
        up = self._prev_cu
        acu = -cu if cu < 0.0 else cu
        aup = -up if up < 0.0 else up
        payoff = (float(acu > 0.4), cu * cu, float(cu * up > 0.0),
                  float(acu > 0.3 and aup > 0.3))
        ev_z = self._Z
        hm = self._hm8
        m0s = self._m0
        for i in range(4):
            d = payoff[i] - m0s[i]
            hm[2 * i] = d
            hm[2 * i + 1] = d
        inbuf = self._bet_in
        np.multiply(self._lam8, hm[None, :], out=inbuf)
        inbuf += 1.0
        np.maximum(inbuf, 1e-12, out=inbuf)
        cs = self._cs_bet
        cs += np.log(inbuf)
        S = cs + self._lwbet2
        mx = S.max(axis=0)
        mix = mx + np.log(np.exp(S - mx[None, :]).sum(axis=0))
        mixl = mix.tolist()
        for i, i_mix, i_z in ((0, 31, 32), (1, 34, 35), (2, 37, 38), (3, -1, 39)):
            nl = self._bet_null[i]
            jj = js[nl.gi]
            mv = mixl[2 * i]
            zz = _rz_s(mv, nl.med[jj], nl.sd[jj])
            if i_mix >= 0:
                o[i_mix] = mv
            o[i_z] = zz
            ev_z[i] = zz

        # predictable (plug-in GRAPA) betting fraction, tail and disp channels
        gr2 = self._gr2
        gr_c, gr_s = self._gr_c, self._gr_s
        for j, (i, m0, vhat, lo_b, up_b) in enumerate(self._grapa):
            h = payoff[i]
            c_prev = gr_c[j]
            lam_g = ((GRAPA_K0 * m0 + c_prev) / (GRAPA_K0 + t) - m0) / vhat
            if lam_g < lo_b:
                lam_g = lo_b
            elif lam_g > up_b:
                lam_g = up_b
            w = 1.0 + lam_g * (h - m0)
            gr2[j] = w if w > 1e-12 else 1e-12
            gr_c[j] = c_prev + h
        lgr = np.log(gr2)
        gr_s[0] += float(lgr[0])
        gr_s[1] += float(lgr[1])
        o[33] = gr_s[0]
        o[36] = gr_s[1]

        # Vovk simple-mixture power martingale
        pp = 2.0 * (u if u < 1.0 - u else 1.0 - u)
        pp = 1e-6 if pp < 1e-6 else (1.0 if pp > 1.0 else pp)
        qq = 1.0 - pp
        qq = 1e-6 if qq < 1e-6 else (1.0 if qq > 1.0 else qq)
        two_ = self._two
        two_[0] = pp
        two_[1] = qq
        lpq = np.log(two_)
        lp_ = float(lpq[0])
        lq_ = float(lpq[1])
        Ro = self._pow_in
        r = 0
        for le, em1 in self._pow_c:
            w = le + em1 * lp_
            Ro[r, 0] = w
            Ro[r, 1] = w
            r += 1
            w = le + em1 * lq_
            Ro[r, 0] = w
            Ro[r, 1] = w
            r += 1
        cs = self._cs_pow
        cs += Ro
        S = cs + self._lwp2
        mx = S.max(axis=0)
        powmix = float((mx + np.log(np.exp(S - mx[None, :]).sum(axis=0)))[0])
        nl = self._an_p
        jj = js[nl.gi]
        pz = _rz_s(powmix, nl.med[jj], nl.sd[jj])
        o[40] = powmix
        o[41] = pz
        self._pk_pow = _runmax(self._pk_pow, powmix, t)
        o[42] = self._pk_pow
        ev_z[4] = pz

        # ---- sequential Bayes factor ------------------------------------
        llr = self._llr1
        llr[0] = gl
        acc = self._acc_llr
        acc += llr
        Ab = acc + self._logw2
        bf0 = _fam_from_A(Ab, None, self._fn0)
        nl = self._an_bf
        jj = js[nl.gi]
        zbf = _rz_s(bf0, nl.med[jj], nl.sd[jj])
        o[43] = bf0
        o[44] = zbf
        o[45] = bf0 / (t + 1.0)
        bf0v = _fam_from_A(Ab, self._fvar_sel, self._fvar_fn)
        o[46] = bf0v
        nl = self._an_bfv
        jj = js[nl.gi]
        o[47] = _rz_s(bf0v, nl.med[jj], nl.sd[jj])

        # ---- cross-channel agreement ------------------------------------
        ev_z[5] = z_lpo0
        ev_z[6] = zbf
        mxz = ev_z[0]
        hot = 0
        for x in ev_z:
            if x != x:                   # np.max propagates NaN; Python max does not
                mxz = x
                break
            if x > mxz:
                mxz = x
        for x in ev_z:
            if x > 2.0:
                hot += 1
        o[48] = mxz
        o[49] = float(hot)

        # carry
        self._prev_g = g
        self._prev_z = zf
        self._prev_cu = cu

        # batch casts to float32 and nan-fills the non-finite entries
        r32 = np.array(o, dtype=np.float32)
        r32[~np.isfinite(r32)] = np.nan
        return r32.astype(np.float64)
