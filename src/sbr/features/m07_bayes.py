"""m07_bayes -- generative / sequential Bayesian evidence for an ABSORBING break.

WHAT THIS IS (and why it is not m01_seq)
    m01_seq runs the classical frequentist detector bank (CUSUM, Page-Hinkley,
    GLR, Shiryaev-Roberts on a mean shift) and exposes the *shape* of each
    detector path.  This module instead writes down a GENERATIVE model of the
    problem and reports posterior quantities from it:

      * the latent state NOT-BROKEN -> BROKEN is ABSORBING, so the exact
        filtering recursion for P(broken at t | x_1:t) collapses to one
        log-space accumulator per alternative parameter value:

            L_t(th) = logaddexp(L_{t-1}(th), log h) + llr_t(th) - log(1-h)

        with L_t(th) = log [ P(broken, params th, x_1:t) / P(not broken, x_1:t) ]
        and llr_t(th) = log p1(x_t|th) - log p0(x_t).  Marginalising th with the
        prior weights gives an exact log posterior odds of BROKEN.  There is no
        dyadic approximation and no candidate-changepoint scan: the absorbing
        structure integrates over every changepoint exactly, in O(1) per step
        per mixture component.
      * Bayesian Online Change Point Detection (Adams & MacKay) with a
        Normal-Inverse-Gamma observation model whose prior IS the historical
        segment.  Its run-length posterior is a *different* object from a
        detector path: it is a distribution over "how long has the current
        regime lasted", and its shape (mass below k, mode, entropy, changed-vs-
        never-changed odds) is what we emit.  Unlike the absorbing model it
        allows repeated resets, so the two disagree exactly on transients.
      * E-VALUES / test martingales: betting capital processes against the
        per-series historical null, in log space, both with a predictable
        (plug-in GRAPA) betting fraction and as a mixture over a fixed grid of
        fractions, plus Vovk's simple-mixture power martingale on conformal
        p-values.  These are anytime-valid by Ville's inequality -- the right
        object for a stream whose stopping time is unknown -- and they are
        constructed from bounded payoff functions, so they never blow up on a
        single heavy-tailed outlier the way a likelihood ratio does.
      * Sequential Bayes factors for the "break at the first online point"
        alternative, i.e. the mixture likelihood ratio accumulated from t=0.

THE OBSERVATION STREAM (this is load-bearing)
    Data forensics says breaks are SCALE- and DEPENDENCE-dominant; location
    breaks essentially do not exist; and the *raw* variance ratio is biased by
    tail heaviness (a length-mismatch artifact), so an uncalibrated Gaussian
    likelihood would mis-rank the cross-section.  The primary stream is
    therefore

        raw -> standardise by historical (mu, sd)
            -> AR(6) whitening with historical coefficients (causal, warm
               started from the tail of history)
            -> NORMAL SCORES against the historical residual ECDF.

    After the last step the null distribution of the stream is exactly N(0,1)
    *per series, by construction*.  That is what makes a log Bayes factor
    comparable across series with wildly different volatility and tail weight,
    which is the whole game under TS-AUC.  It also kills the tail-heaviness bias
    of the raw variance ratio outright, because the transform is rank based.
    A second, deliberately un-Gaussianised channel runs the same absorbing
    recursion on the raw standardised series, so genuine tail/scale information
    that the rank transform compresses is not thrown away.

THE MIXTURE (where the prior mass goes)
    variance ratio grid 0.5x .. 2.0x                          weight 0.45
    AR(1) coefficient change (dependence)                     weight 0.20
    ARCH-type conditional-variance dependence                 weight 0.15
    mean shift  (present, deliberately NOT dominant)          weight 0.20

CALIBRATION (the other load-bearing part)
    A raw log Bayes factor is not comparable across series, so every headline
    quantity is emitted BOTH raw and calibrated against a per-series historical
    null built by running the IDENTICAL recursion over the break-free
    historical segment:
      * additive channels (sequential BF from t=0, mixture e-processes) admit an
        EXACT length-matched null: the statistic over a length-L window is a
        window sum of the same increments, so every length-L historical window
        gives one null draw.  ``_AddNull``.
      * the absorbing posterior is path dependent, so its null is built by
        RESTARTING the recursion at a dense grid of historical offsets and
        recording the path at a log-spaced grid of elapsed positions.
        ``_PosNull``.  This is length matched: online step t is scored against
        historical runs that have also been going for ~t steps, which matters
        because the posterior odds drift with elapsed time under the null.
      * BOCPD summaries are already probabilities and roughly comparable; they
        get a marginal historical-path null (``_MargNull``).

SIGNAL / ALPHA
    absorbing posterior + sequential BF: a genuine regime change makes the
      post-break likelihood ratio drift upward at a constant rate, so the log
      posterior odds grow ~linearly in elapsed-since-break and never come back
      -- the absorbing prior is exactly the right shape for a target that is
      itself absorbing (y_t = 1 for all t >= tau).
    BOCPD: after a break the run-length posterior collapses onto short run
      lengths and the never-changed mass decays to zero.
    e-processes: capital compounds only while the alternative keeps paying, and
      by Ville's inequality a large value is evidence at ANY stopping time, so
      the value is meaningful at every online index without a multiple-testing
      correction -- which is what TS-AUC scores.

FALSE SIGNAL
    An isolated outlier or a transient volatility burst also pushes the mixture
    likelihood ratio up.  Three companion channels separate them:
      1. the absorbing posterior cannot decay (it is a running integral), so on
         its own it confuses transient with permanent -- but ``ab_lpo_rel``
         (current / running max) and ``ab_lpo_slp`` (recent slope) do not: a
         transient leaves rel < 1 and slope ~ 0, a break keeps rel ~ 1 and the
         slope positive.
      2. BOCPD *does* decay: after a transient the run-length posterior
         re-concentrates on long runs, so ``bo_lo_change`` falls back.  The
         disagreement between the absorbing posterior (high) and BOCPD
         (recovered) is the transient signature, and both are emitted.
      3. the betting e-processes use BOUNDED payoffs, so one enormous
         observation can win at most one bet, whereas the Gaussian likelihood
         ratio would score it as overwhelming evidence.  ``ev_*`` high with
         ``ab_var`` low = a genuine sustained scale change; the reverse = a
         fat-tail artifact.
    A slowly drifting (non-absorbing) volatility level moves everything the same
    way; nothing here separates that, and it is a known limitation.

CAUSALITY
    Every historical object (AR coefficients, residual ECDF, NIG prior, all
    three nulls) is a function of ``hist`` alone.  Every online quantity is a
    strictly forward recursion / cumulative sum over ``online[:t+1]`` plus those
    constants; lagged inputs at t=0 are warm started from the last historical
    point.  No online reduction ever looks forward, and no constant depends on
    ``n_online``.  Verified bitwise by ``check_prefix_invariance`` at atol=0.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtri

from sbr.features.base import register
from sbr.transforms import _ar_resid, _fit_ar, ar_filter_causal

# ------------------------------------------------------------------ constants
AR_ORDER = 6                 # forensics: AR(6) whitening beats AR(2) for scale
HAZ_SLOW = 1.0 / 250.0       # ~1 / mean online length (NOT a function of n)
HAZ_FAST = 1.0 / 40.0
GCLIP = 6.0
ZCAP = 12.0
SURCAP = 6.0
EPS = 1e-12

# mixture over post-break parameter values -----------------------------------
VAR_GRID = (0.50, 0.63, 0.79, 1.26, 1.58, 2.00)      # variance ratio
MU_GRID = (-0.8, -0.4, -0.2, 0.2, 0.4, 0.8)          # mean shift (in sd)
RHO_GRID = (-0.4, -0.2, 0.2, 0.4)                    # AR(1) coefficient change
ARCH_GRID = (0.15, 0.35)                             # ARCH(1) coefficient
W_VAR, W_DEP, W_ARCH, W_LOC = 0.45, 0.20, 0.15, 0.20
F_VAR, F_LOC, F_DEP, F_ARCH = 1, 2, 3, 4             # family ids (0 = combined)

# BOCPD
R_MAX = 80
BO_KAP0, BO_AL0 = 25.0, 25.0
BO_HAZ = 1.0 / 250.0
BO_HIST_MAX = 800

# calibration grids
POS_GRID = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256])
BLK_STRIDE = 32
WIN_GRID = np.array([6, 18, 54, 162, 486])
NULL_MAXPTS = 500

# e-process betting
BET_C = np.array([-0.9, -0.6, -0.3, 0.3, 0.6, 0.9])
POW_EPS = np.array([0.15, 0.3, 0.5, 0.7, 0.9])
GRAPA_K0 = 20.0


# ------------------------------------------------------------------ kernels
try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:                                                # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not a or not callable(a[0]) else a[0]


@njit(cache=True, fastmath=False)
def _absorb_path(llr, log_h, log1mh):
    """L_t(k) = logaddexp(L_{t-1}(k), log h) + llr_t(k) - log(1-h), L_{-1} = -inf."""
    n, K = llr.shape
    L = np.empty((n, K))
    prev = np.empty(K)
    for k in range(K):
        prev[k] = -1.0e300
    for t in range(n):
        for k in range(K):
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
            v = s + llr[t, k] - log1mh
            L[t, k] = v
            prev[k] = v
    return L


@njit(cache=True, fastmath=False)
def _absorb_blocks(llr, log_h, log1mh, logw, fam, famnorm, offs, pos, nfam):
    """Restart the absorbing recursion at every offset in ``offs``.

    Returns (n_blocks, n_pos, n_fam): the family log Bayes factors after
    ``pos[j]`` steps of a run that started at ``offs[b]``.  Family 0 is the
    full mixture.  This is the length-matched historical null.
    """
    nb = offs.shape[0]
    npos = pos.shape[0]
    K = llr.shape[1]
    lmax = pos[npos - 1]
    out = np.empty((nb, npos, nfam))
    prev = np.empty(K)
    for b in range(nb):
        o = offs[b]
        for k in range(K):
            prev[k] = -1.0e300
        pi = 0
        for j in range(lmax):
            t = o + j
            for k in range(K):
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
                prev[k] = s + llr[t, k] - log1mh
            if pi < npos and (j + 1) == pos[pi]:
                for f in range(nfam):
                    mx = -1.0e300
                    for k in range(K):
                        if f == 0 or fam[k] == f:
                            v = logw[k] + prev[k]
                            if v > mx:
                                mx = v
                    ss = 0.0
                    for k in range(K):
                        if f == 0 or fam[k] == f:
                            ss += math.exp(logw[k] + prev[k] - mx)
                    out[b, pi, f] = mx + math.log(ss) - famnorm[f]
                pi += 1
    return out


@njit(cache=True, fastmath=False)
def _bocpd_ct(al0, R):
    """Student-t log normalising constant per run length: the ONLY source of it.

    ``ct[r] = lgamma((nu+1)/2) - lgamma(nu/2) - log(nu*pi)/2`` for
    ``nu = 2*(al0 + r/2)``.  Eighty data-independent constants.

    This lives in its own kernel because ``math.lgamma`` is the one primitive in
    this module whose numba and CPython implementations disagree (up to 512 ULP
    on this grid; ``log``/``exp``/``log1p`` agree bitwise).  The streaming
    mirror ``sbr.stream.s_m07_bayes._BocpdStream`` imports this function rather
    than recomputing the table, so both sides get the same bits under whichever
    of the two ``lgamma`` implementations is actually in play -- including the
    numba-absent fallback, where ``njit`` above degrades to a no-op and both
    sides equally use CPython's.  Recomputing it independently is what put
    batch-trained features and stream-served features on different constants.
    """
    ct = np.empty(R)
    for r in range(R):
        al = al0 + 0.5 * r
        nu = 2.0 * al
        ct[r] = (math.lgamma(0.5 * (nu + 1.0)) - math.lgamma(0.5 * nu)
                 - 0.5 * math.log(nu * math.pi))
    return ct


@njit(cache=True, fastmath=False)
def _bocpd(x, mu0, be0, kap0, al0, log_h, log1mh, R):
    """Adams-MacKay run-length posterior with a Normal-Inverse-Gamma model.

    Bucket ``r`` after step t holds the sufficient statistics of the last r
    online observations on top of the historical prior; ``r = t+1`` is the
    "no change since the online segment started" hypothesis.  The grid is
    truncated at R: mass that would reach r = R is folded into r = R-1, which
    turns the longest-run hypothesis into a trailing-window one.

    Returns (n, 8): P(r<10), P(r<25), P(r<50), mode/(t+1), mean/(t+1),
    entropy, P(no change), log-odds(changed vs never changed).
    """
    n = x.shape[0]
    kap = np.empty(R)
    al = np.empty(R)
    nu = np.empty(R)
    ct = _bocpd_ct(al0, R)
    sf = np.empty(R)
    mu = np.empty(R)
    be = np.empty(R)
    lp = np.empty(R)
    for r in range(R):
        kap[r] = kap0 + r
        al[r] = al0 + 0.5 * r
        nu[r] = 2.0 * al[r]
        sf[r] = (kap[r] + 1.0) / (al[r] * kap[r])
        mu[r] = mu0
        be[r] = be0
        lp[r] = -1.0e300
    lp[0] = 0.0
    lpred = np.empty(R)
    nmu = np.empty(R)
    nbe = np.empty(R)
    nlp = np.empty(R)
    out = np.zeros((n, 8))
    top = 0
    for t in range(n):
        xt = x[t]
        for r in range(top + 1):
            s2 = be[r] * sf[r]
            d = xt - mu[r]
            lpred[r] = (ct[r] - 0.5 * math.log(s2)
                        - 0.5 * (nu[r] + 1.0) * math.log1p(d * d / (nu[r] * s2)))
        mx = -1.0e300
        for r in range(top + 1):
            v = lp[r] + lpred[r]
            if v > mx:
                mx = v
        ss = 0.0
        for r in range(top + 1):
            ss += math.exp(lp[r] + lpred[r] - mx)
        lcp = mx + math.log(ss) + log_h

        newtop = top + 1
        if newtop > R - 1:
            newtop = R - 1
        for r in range(newtop, 0, -1):
            src = r - 1
            if src > top:
                src = top
            v = lp[src] + lpred[src] + log1mh
            if r == R - 1 and top == R - 1:
                b = lp[R - 1] + lpred[R - 1] + log1mh
                if b > v:
                    v = b + math.log1p(math.exp(lp[src] + lpred[src] + log1mh - b))
                else:
                    v = v + math.log1p(math.exp(b - v))
            nlp[r] = v
            k = kap[src]
            dm = xt - mu[src]
            nbe[r] = be[src] + k * dm * dm / (2.0 * (k + 1.0))
            nmu[r] = (k * mu[src] + xt) / (k + 1.0)
        nlp[0] = lcp
        nmu[0] = mu0
        nbe[0] = be0

        mx = -1.0e300
        for r in range(newtop + 1):
            if nlp[r] > mx:
                mx = nlp[r]
        ss = 0.0
        for r in range(newtop + 1):
            ss += math.exp(nlp[r] - mx)
        lse = mx + math.log(ss)
        for r in range(newtop + 1):
            lp[r] = nlp[r] - lse
            mu[r] = nmu[r]
            be[r] = nbe[r]
        top = newtop

        p10 = 0.0
        p25 = 0.0
        p50 = 0.0
        mean = 0.0
        ent = 0.0
        best = -1.0
        mode = 0
        for r in range(top + 1):
            p = math.exp(lp[r])
            if r < 10:
                p10 += p
            if r < 25:
                p25 += p
            if r < 50:
                p50 += p
            mean += r * p
            if p > 1e-300:
                ent -= p * lp[r]
            if p > best:
                best = p
                mode = r
        pnc = math.exp(lp[top])
        rel = 1.0 / (t + 1.0)
        out[t, 0] = p10
        out[t, 1] = p25
        out[t, 2] = p50
        out[t, 3] = mode * rel
        out[t, 4] = mean * rel
        out[t, 5] = ent
        out[t, 6] = pnc
        lo = math.log(max(1.0 - pnc, 1e-12)) - math.log(max(pnc, 1e-12))
        if lo > 30.0:
            lo = 30.0
        if lo < -30.0:
            lo = -30.0
        out[t, 7] = lo
    return out


# ------------------------------------------------------------------ nulls
def _rz(v, med, sd):
    return np.clip((v - med) / np.maximum(sd, 1e-9), -ZCAP, ZCAP)


def _signed_surprise(p):
    two = 2.0 * np.minimum(p, 1.0 - p)
    return np.sign(p - 0.5) * np.minimum(-np.log10(np.maximum(two, 1e-7)), SURCAP)


class _MargNull:
    """Marginal null of a historical detector path (after burn-in)."""

    __slots__ = ("srt", "n", "med", "sd", "hi")

    def __init__(self, path):
        a = np.asarray(path, dtype=np.float64)
        burn = min(64, a.shape[0] // 4)
        a = a[burn:]
        if a.shape[0] < 32:
            a = np.asarray(path, dtype=np.float64)
        if a.shape[0] > NULL_MAXPTS:
            a = a[:: max(1, a.shape[0] // NULL_MAXPTS)]
        self.srt = np.sort(a)
        self.n = self.srt.shape[0]
        self.med = float(self.srt[self.n // 2])
        q1 = float(self.srt[self.n // 4])
        q3 = float(self.srt[min(self.n - 1, (3 * self.n) // 4)])
        self.sd = max((q3 - q1) / 1.349, 1e-6)
        self.hi = float(self.srt[-1])

    def z(self, v):
        return _rz(v, self.med, self.sd)

    def pct(self, v):
        lo = np.searchsorted(self.srt, v, side="left")
        hi = np.searchsorted(self.srt, v, side="right")
        return 0.5 * (lo + hi) / self.n


class _PosNull:
    """Length-matched null: statistic after L steps of a restarted run."""

    __slots__ = ("lpos", "srt", "med", "sd", "n")

    def __init__(self, vals, pos):
        # vals (n_blocks, n_pos)
        self.lpos = np.log(pos.astype(np.float64))
        self.srt = np.sort(vals, axis=0)
        nb = vals.shape[0]
        self.n = nb
        self.med = self.srt[nb // 2]
        q1 = self.srt[nb // 4]
        q3 = self.srt[min(nb - 1, (3 * nb) // 4)]
        self.sd = np.maximum((q3 - q1) / 1.349, 1e-6)

    def _idx(self, L):
        return np.argmin(np.abs(np.log(np.maximum(L, 1.0))[:, None] - self.lpos[None, :]), axis=1)

    def z(self, v, L):
        j = self._idx(L)
        return _rz(v, self.med[j], self.sd[j])

    def pct(self, v, L):
        j = self._idx(L)
        out = np.empty(v.shape[0])
        for jj in np.unique(j):
            m = j == jj
            s = self.srt[:, jj]
            lo = np.searchsorted(s, v[m], side="left")
            hi = np.searchsorted(s, v[m], side="right")
            out[m] = 0.5 * (lo + hi) / self.n
        return out


class _AddNull:
    """Exact length-matched null for logsumexp_c(logw_c + window-sum of inc_c).

    With a single component and zero weight this is a plain cumulative sum, so
    it covers both the sequential Bayes factor and the mixture e-processes.
    """

    __slots__ = ("wg", "lwg", "srt", "med", "sd", "n")

    def __init__(self, inc, logw, hlen):
        inc = np.atleast_2d(np.asarray(inc, dtype=np.float64))
        logw = np.asarray(logw, dtype=np.float64)
        C = np.concatenate([np.zeros((inc.shape[0], 1)), np.cumsum(inc, axis=1)], axis=1)
        wg = WIN_GRID[WIN_GRID <= max(hlen // 2, 4)]
        if wg.size == 0:
            wg = np.array([4])
        self.wg = wg
        self.lwg = np.log(wg.astype(np.float64))
        self.srt, self.med, self.sd = [], [], []
        nc, Hn = inc.shape
        for L in wg:
            L = int(L)
            no = Hn - L + 1
            step = max(1, no // NULL_MAXPTS)
            o = np.arange(0, no, step)
            S = C[:, o + L] - C[:, o] + logw[:, None]
            m = S.max(axis=0)
            v = m + np.log(np.exp(S - m[None, :]).sum(axis=0))
            s = np.sort(v)
            k = s.shape[0]
            self.srt.append(s)
            self.med.append(float(s[k // 2]))
            q1 = float(s[k // 4])
            q3 = float(s[min(k - 1, (3 * k) // 4)])
            self.sd.append(max((q3 - q1) / 1.349, 1e-6))
        self.med = np.array(self.med)
        self.sd = np.array(self.sd)
        self.n = np.array([len(s) for s in self.srt])

    def _idx(self, L):
        return np.argmin(np.abs(np.log(np.maximum(L, 1.0))[:, None] - self.lwg[None, :]), axis=1)

    def z(self, v, L):
        j = self._idx(L)
        return _rz(v, self.med[j], self.sd[j])

    def pct(self, v, L):
        j = self._idx(L)
        out = np.empty(v.shape[0])
        for jj in np.unique(j):
            m = j == jj
            s = self.srt[jj]
            lo = np.searchsorted(s, v[m], side="left")
            hi = np.searchsorted(s, v[m], side="right")
            out[m] = 0.5 * (lo + hi) / len(s)
        return out


# ------------------------------------------------------------------ helpers
def _llr_matrix(g, g_prev):
    """Per-point log likelihood ratios of every mixture component vs N(0,1).

    ``g_prev`` is the lag-1 stream (warm started from history at position 0),
    which is what makes the dependence components causal at t = 0.
    """
    n = g.shape[0]
    cols, logw, fam, meta = [], [], [], []
    g2 = g * g

    for v in VAR_GRID:                                          # scale change
        cols.append(-0.5 * math.log(v) + 0.5 * g2 * (1.0 - 1.0 / v))
        logw.append(W_VAR / len(VAR_GRID))
        fam.append(F_VAR)
        meta.append((math.log(v), 0.0, 0.0))
    for m in MU_GRID:                                           # location change
        cols.append(m * g - 0.5 * m * m)
        logw.append(W_LOC / len(MU_GRID))
        fam.append(F_LOC)
        meta.append((0.0, 0.0, m))
    for r in RHO_GRID:                                          # dependence change
        s2 = 1.0 - r * r
        d = g - r * g_prev
        cols.append(-0.5 * math.log(s2) - 0.5 * d * d / s2 + 0.5 * g2)
        logw.append(W_DEP / len(RHO_GRID))
        fam.append(F_DEP)
        meta.append((0.0, r, 0.0))
    for a in ARCH_GRID:                                         # ARCH-type change
        v = 1.0 + a * (g_prev * g_prev - 1.0)
        v = np.maximum(v, 0.05)
        cols.append(-0.5 * np.log(v) + 0.5 * g2 * (1.0 - 1.0 / v))
        logw.append(W_ARCH / len(ARCH_GRID))
        fam.append(F_ARCH)
        meta.append((0.0, a, 0.0))

    M = np.ascontiguousarray(np.column_stack(cols))
    M = np.clip(M, -40.0, 40.0)
    return M, np.log(np.array(logw)), np.array(fam, dtype=np.int64), np.array(meta)


def _fam_norm(logw, fam, nfam=5):
    out = np.zeros(nfam)
    w = np.exp(logw)
    out[0] = math.log(w.sum())
    for f in range(1, nfam):
        s = w[fam == f].sum()
        out[f] = math.log(s) if s > 0 else 0.0
    return out


def _fam_lse(L, logw, fam, famnorm, f):
    """Row-wise family log Bayes factor from the per-component accumulators."""
    sel = np.ones(L.shape[1], bool) if f == 0 else (fam == f)
    A = L[:, sel] + logw[sel][None, :]
    m = A.max(axis=1)
    return m + np.log(np.exp(A - m[:, None]).sum(axis=1)) - famnorm[f]


def _bet_lambdas(m0, hi):
    lam = np.empty(BET_C.shape[0])
    for i, c in enumerate(BET_C):
        lam[i] = c / max(m0, 1e-6) if c > 0 else c / max(hi - m0, 1e-6)
    return lam


def _mix_bet(h, m0, hi):
    """Increments of the fixed-lambda mixture betting e-process, one row per bet."""
    lam = _bet_lambdas(m0, hi)
    return np.log(np.maximum(1.0 + lam[:, None] * (h[None, :] - m0), 1e-12)), lam


def _grapa(h, m0, hi, vhat):
    """Predictable (plug-in) betting fraction; lambda_t uses only h[:t]."""
    n = h.shape[0]
    c = np.concatenate([[0.0], np.cumsum(h)])
    t = np.arange(n, dtype=np.float64)
    mhat = (GRAPA_K0 * m0 + c[:-1]) / (GRAPA_K0 + t)
    lo = -0.9 / max(hi - m0, 1e-6)
    up = 0.9 / max(m0, 1e-6)
    lam = np.clip((mhat - m0) / max(vhat, 1e-6), lo, up)
    return np.cumsum(np.log(np.maximum(1.0 + lam * (h - m0), 1e-12)))


def _slope(d, k, scale):
    n = d.shape[0]
    out = np.full(n, np.nan)
    if n > k:
        out[k:] = np.clip((d[k:] - d[:-k]) / (k * max(scale, 1e-6)), -ZCAP, ZCAP)
    return out


# ------------------------------------------------------------------ module
@register("m07_bayes", version="1", owner="agent12")
def build(ctx):
    n = ctx.n
    tt = np.arange(1, n + 1, dtype=np.float64)
    cols: list[str] = []
    out: list[np.ndarray] = []

    def col(name, arr):
        cols.append(name)
        out.append(np.asarray(arr, dtype=np.float64))

    # ---------------- streams -------------------------------------------------
    hp = ctx.hp
    zh = (np.asarray(ctx.hist, dtype=np.float64) - hp.mu) / hp.sd
    zo = np.asarray(ctx.tr["mean"], dtype=np.float64)

    coef = _fit_ar(zh, AR_ORDER)
    eh = _ar_resid(zh, coef)                       # historical residual stream
    if eh.shape[0] < 200:                          # degenerate history guard
        eh = zh
        coef = np.zeros(AR_ORDER)
    sig = max(float(eh.std(ddof=1)), 1e-9)
    eh = eh / sig
    eo = ar_filter_causal(zo, coef, zh) / sig

    # normal scores against the historical residual ECDF -> exactly N(0,1) null
    srt = np.sort(eh)
    H = srt.shape[0]

    def _ns(x):
        lo = np.searchsorted(srt, x, side="left")
        hi = np.searchsorted(srt, x, side="right")
        return np.clip(ndtri((0.5 * (lo + hi) + 0.5) / (H + 1.0)), -GCLIP, GCLIP)

    g = _ns(eo)
    gh = _ns(eh)
    g_prev = np.concatenate([[gh[-1]], g[:-1]])
    gh_prev = np.concatenate([[0.0], gh[:-1]])

    # ---------------- absorbing posterior on the Gaussianised stream ----------
    llr_o, logw, fam, meta = _llr_matrix(g, g_prev)
    llr_h, _, _, _ = _llr_matrix(gh, gh_prev)
    fnorm = _fam_norm(logw, fam)
    lh_s, l1_s = math.log(HAZ_SLOW), math.log1p(-HAZ_SLOW)
    lh_f, l1_f = math.log(HAZ_FAST), math.log1p(-HAZ_FAST)

    L = _absorb_path(llr_o, lh_s, l1_s)
    lpo = _fam_lse(L, logw, fam, fnorm, 0)
    f_var = _fam_lse(L, logw, fam, fnorm, F_VAR)
    f_dep = _fam_lse(L, logw, fam, fnorm, F_DEP)
    f_arch = _fam_lse(L, logw, fam, fnorm, F_ARCH)
    f_loc = _fam_lse(L, logw, fam, fnorm, F_LOC)

    # posterior over the mixture component GIVEN broken -> "what kind of break"
    A = L + logw[None, :]
    P = np.exp(A - A.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    post_lvr = P @ meta[:, 0]
    post_rho = P @ meta[:, 1]
    pf_var = P[:, fam == F_VAR].sum(axis=1)
    pf_dep = P[:, (fam == F_DEP) | (fam == F_ARCH)].sum(axis=1)

    # length-matched historical null for the absorbing recursion
    Hh = llr_h.shape[0]
    pos = POS_GRID[POS_GRID <= max(Hh // 3, 8)]
    if pos.size < 3:
        pos = np.array([1, 2, 4])
    lmax = int(pos[-1])
    offs = np.arange(0, Hh - lmax + 1, BLK_STRIDE, dtype=np.int64)
    if offs.shape[0] < 12:
        offs = np.arange(0, max(Hh - lmax + 1, 1), 1, dtype=np.int64)
    BK = _absorb_blocks(llr_h, lh_s, l1_s, logw, fam, fnorm, offs,
                        pos.astype(np.int64), 5)
    pn = {f: _PosNull(BK[:, :, f], pos) for f in range(5)}

    col("ab_lpo", lpo)
    col("ab_lpo_z", pn[0].z(lpo, tt))
    col("ab_lpo_sur", _signed_surprise(pn[0].pct(lpo, tt)))
    col("ab_var", f_var)
    col("ab_var_z", pn[F_VAR].z(f_var, tt))
    col("ab_dep", f_dep)
    col("ab_dep_z", pn[F_DEP].z(f_dep, tt))
    col("ab_loc_z", pn[F_LOC].z(f_loc, tt))
    col("ab_post_lvr", post_lvr)
    col("ab_post_rho", post_rho)
    col("ab_pfam_var", pf_var)
    col("ab_pfam_dep", pf_dep)

    pk = np.maximum.accumulate(lpo)
    col("ab_lpo_rel", (lpo - pk) / np.maximum(np.abs(pk), 1.0))
    col("ab_lpo_slp", _slope(lpo, 16, float(np.median(pn[0].sd))))
    col("ab_lpo_pk_z", pn[0].z(pk, tt))

    # fast hazard: same likelihoods, shorter memory -> reacts to recent evidence
    Lf = _absorb_path(llr_o, lh_f, l1_f)
    lpo_f = _fam_lse(Lf, logw, fam, fnorm, 0)
    mn_f = _MargNull(_fam_lse(_absorb_path(llr_h, lh_f, l1_f), logw, fam, fnorm, 0))
    col("ab_fast", lpo_f)
    col("ab_fast_z", mn_f.z(lpo_f))
    col("ab_fast_slow", lpo_f - lpo)

    # ---------------- absorbing posterior on the RAW standardised stream ------
    zprev_o = np.concatenate([[zh[-1]], zo[:-1]])
    zprev_h = np.concatenate([[0.0], zh[:-1]])
    llr_zo, _, _, _ = _llr_matrix(zo, zprev_o)
    llr_zh, _, _, _ = _llr_matrix(zh, zprev_h)
    Lz = _absorb_path(llr_zo, lh_s, l1_s)
    Lzh = _absorb_path(llr_zh, lh_s, l1_s)
    z_lpo = _fam_lse(Lz, logw, fam, fnorm, 0)
    z_var = _fam_lse(Lz, logw, fam, fnorm, F_VAR)
    col("abz_lpo", z_lpo)
    col("abz_lpo_z", _MargNull(_fam_lse(Lzh, logw, fam, fnorm, 0)).z(z_lpo))
    col("abz_var_z", _MargNull(_fam_lse(Lzh, logw, fam, fnorm, F_VAR)).z(z_var))

    # ---------------- BOCPD ---------------------------------------------------
    lbh, lb1 = math.log(BO_HAZ), math.log1p(-BO_HAZ)
    v_g = max(float(gh.var()), 1e-6)
    be0 = (BO_AL0 - 1.0) * v_g
    mu0 = float(gh.mean())
    BO = _bocpd(np.ascontiguousarray(g), mu0, be0, BO_KAP0, BO_AL0, lbh, lb1, R_MAX)
    ghh = gh[-BO_HIST_MAX:] if gh.shape[0] > BO_HIST_MAX else gh
    BH = _bocpd(np.ascontiguousarray(ghh), mu0, be0, BO_KAP0, BO_AL0, lbh, lb1, R_MAX)

    col("bo_p_lt10", BO[:, 0])
    col("bo_p_lt25", BO[:, 1])
    col("bo_mean_rel", BO[:, 4])
    col("bo_ent", BO[:, 5])
    col("bo_nochange", BO[:, 6])
    col("bo_lo_change", BO[:, 7])
    col("bo_lo_change_z", _MargNull(BH[:, 7]).z(BO[:, 7]))
    col("bo_p_lt25_z", _MargNull(BH[:, 1]).z(BO[:, 1]))
    col("bo_ent_z", _MargNull(BH[:, 5]).z(BO[:, 5]))
    col("bo_lo_change_pk", np.maximum.accumulate(BO[:, 7]))

    # ---------------- e-processes / test martingales --------------------------
    u_o = np.asarray(ctx.tr["u"], dtype=np.float64)
    u_h = np.asarray(ctx.hist_tr["u"], dtype=np.float64)
    cu_o, cu_h = u_o - 0.5, u_h - 0.5
    up_o = np.concatenate([[cu_h[-1]], cu_o[:-1]])
    up_h = np.concatenate([[0.0], cu_h[:-1]])

    payoffs = {
        "tail": (np.abs(cu_o) > 0.4).astype(np.float64),
        "disp": cu_o * cu_o,
        "dep": (cu_o * up_o > 0).astype(np.float64),
        "vc": ((np.abs(cu_o) > 0.3) & (np.abs(up_o) > 0.3)).astype(np.float64),
    }
    payoffs_h = {
        "tail": (np.abs(cu_h) > 0.4).astype(np.float64),
        "disp": cu_h * cu_h,
        "dep": (cu_h * up_h > 0).astype(np.float64),
        "vc": ((np.abs(cu_h) > 0.3) & (np.abs(up_h) > 0.3)).astype(np.float64),
    }
    hicap = {"tail": 1.0, "disp": 0.25, "dep": 1.0, "vc": 1.0}
    lw_bet = np.full(BET_C.shape[0], -math.log(BET_C.shape[0]))
    ev_z = []
    for key in ("tail", "disp", "dep", "vc"):
        h_o, h_h = payoffs[key], payoffs_h[key]
        m0 = float(np.clip(h_h.mean(), 1e-3, hicap[key] - 1e-3))
        inc_o, _ = _mix_bet(h_o, m0, hicap[key])
        inc_h, _ = _mix_bet(h_h, m0, hicap[key])
        S = np.cumsum(inc_o, axis=1) + lw_bet[:, None]
        m = S.max(axis=0)
        mix = m + np.log(np.exp(S - m[None, :]).sum(axis=0))
        an = _AddNull(inc_h, lw_bet, h_h.shape[0])
        zz = an.z(mix, tt)
        if key != "vc":
            col(f"ev_{key}_mix", mix)
        col(f"ev_{key}_z", zz)
        ev_z.append(zz)
        if key in ("tail", "disp"):
            col(f"ev_{key}_ad", _grapa(h_o, m0, hicap[key], max(float(h_h.var()), 1e-6)))

    # Vovk simple-mixture power martingale on two-sided conformal p-values
    p_o = np.clip(2.0 * np.minimum(u_o, 1.0 - u_o), 1e-6, 1.0)
    p_h = np.clip(2.0 * np.minimum(u_h, 1.0 - u_h), 1e-6, 1.0)
    rows_o, rows_h = [], []
    for e in POW_EPS:
        rows_o.append(math.log(e) + (e - 1.0) * np.log(p_o))
        rows_h.append(math.log(e) + (e - 1.0) * np.log(p_h))
        rows_o.append(math.log(e) + (e - 1.0) * np.log(np.clip(1.0 - p_o, 1e-6, 1.0)))
        rows_h.append(math.log(e) + (e - 1.0) * np.log(np.clip(1.0 - p_h, 1e-6, 1.0)))
    Ro = np.asarray(rows_o)
    Rh = np.asarray(rows_h)
    lw_p = np.full(Ro.shape[0], -math.log(Ro.shape[0]))
    S = np.cumsum(Ro, axis=1) + lw_p[:, None]
    m = S.max(axis=0)
    powmix = m + np.log(np.exp(S - m[None, :]).sum(axis=0))
    an_p = _AddNull(Rh, lw_p, p_h.shape[0])
    pz = an_p.z(powmix, tt)
    col("ev_pow_mix", powmix)
    col("ev_pow_z", pz)
    col("ev_pow_pk", np.maximum.accumulate(powmix))
    ev_z.append(pz)

    # ---------------- sequential Bayes factor, break at the first online point
    bf_all_c = np.cumsum(llr_o, axis=0) + logw[None, :]
    m = bf_all_c.max(axis=1)
    bf0 = m + np.log(np.exp(bf_all_c - m[:, None]).sum(axis=1)) - fnorm[0]
    an_bf = _AddNull(llr_h.T, logw, Hh)
    col("bf0_all", bf0)
    col("bf0_all_z", an_bf.z(bf0, tt))
    col("bf0_rate", bf0 / tt)
    sel = fam == F_VAR
    bv = np.cumsum(llr_o[:, sel], axis=0) + logw[sel][None, :]
    m = bv.max(axis=1)
    bf0v = m + np.log(np.exp(bv - m[:, None]).sum(axis=1)) - fnorm[F_VAR]
    col("bf0_var", bf0v)
    col("bf0_var_z", _AddNull(llr_h[:, sel].T, logw[sel], Hh).z(bf0v, tt))

    # ---------------- cross-channel agreement --------------------------------
    Z = np.column_stack(ev_z + [pn[0].z(lpo, tt), an_bf.z(bf0, tt)])
    col("xb_max_z", Z.max(axis=1))
    col("xb_n_hot", (Z > 2.0).sum(axis=1).astype(np.float64))

    Aout = np.column_stack(out).astype(np.float32)
    Aout[~np.isfinite(Aout)] = np.nan
    assert len(cols) <= 50, f"m07_bayes column budget exceeded: {len(cols)}"
    return cols, Aout
