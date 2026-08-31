"""m06_loc -- causal online CHANGE-POINT LOCALISATION and localised segment evidence.

SIGNAL / ALPHA
    Every prefix-anchored statistic averages the post-break evidence together
    with every pre-break online point, so its effect size is diluted by
    (t - tau) / t.  Forensics on this data measured the cost directly: an ORACLE
    window ``online[tau:tau+h]`` separates break from no-break ~3 AUC points
    better than the realistic prefix ``online[0:tau+h]`` at elapsed 160-320,
    which is where most positive rows live.  This module tries to recover part
    of that gap *causally*: at each online index ``t`` it estimates a change
    point ``tau_hat <= t`` from a geometric bank of candidate segment lengths
    and emits statistics of the estimated post-break segment ``(tau_hat, t]``
    rather than of the whole prefix.

    Two localisation engines, because the break population is SCALE-dominant
    and DEPENDENCE-dominant (location breaks are dead, AUC 0.4998):

      H  "segment vs history":  the trailing-m mean of a stream, calibrated
         against the empirical distribution of the SAME statistic over
         length-m windows of the break-free historical segment.  Matched
         length is essential -- an uncalibrated segment statistic mostly
         encodes m (i.e. tau), not the break.
      P  "segment vs its own past": trailing-m mean minus the preceding-m
         mean, calibrated against the historical null of that same
         difference at the same m.  This is the PRE-referenced contrast,
         which forensics found beats the HIST reference when enough pre
         data exists, because it cancels the heavy-tail length bias.

    Three streams, all built on AR(p=6) residuals fitted inside this module
    (the shared context defaults to AR(2); forensics measured AR(6)-residual
    log-sd ratio at 0.596-0.603 series AUC vs 0.549-0.559 raw and ~0.9 points
    better than AR(2)):
      rsq   e_t^2                 innovation scale  (the strongest family)
      rlab  log(|e_t| + 1e-3)     robust / outlier-resistant scale
      rl1   e_t * e_{t-1}         dependence: if the AR coefficients change,
                                  residuals filtered with the HISTORICAL
                                  coefficients acquire autocorrelation.

FALSE SIGNAL
    The maximum over a bank of candidate change points is a maximum over
    correlated noise: a single fat-tailed observation makes the SHORT windows
    fire, so tau_hat snaps to t-8 and the "localised" statistic reports a large
    deviation with a small estimated elapsed.  Heavy tails are endemic here
    (historical kurtosis up to 4,782), and 17 % of no-break series contain a
    break-lookalike transient.

    DISAMBIGUATORS shipped alongside every maximum:
      * ``_q``   the max is calibrated against the historical null of the max
        over the SAME number of candidate windows, so the multiplicity of the
        scan is priced in and a transient that history also produces scores low;
      * ``_lm`` / ``_rel``  where the localisation landed -- a transient snaps to
        the shortest window and to rel ~ 1, a real break settles at growing m;
      * ``_gap`` peak sharpness (best minus runner-up on the calibrated scale):
        a genuine change makes a broad plateau of windows agree, an outlier
        makes one spike;
      * ``_stab`` the trailing-16 dispersion of log2(m_hat): under a real break
        tau_hat stays put while t grows, so log2(m_hat) drifts smoothly and its
        short-window dispersion is small; under noise tau_hat jumps around;
      * the ``fx_`` family (identical statistics at FIXED trailing offsets) is
        the control that says whether localisation adds anything at all beyond
        owning more window lengths.

APPROXIMATION (documented, per the mission)
    A full O(t) scan over candidate change points at every t is O(n^2) per
    series.  We evaluate the candidate set on a GEOMETRIC grid of segment
    lengths (ratio ~1.5), so the scan is O(log t) per step and every segment
    statistic is O(1) from a cumulative sum.  The relative resolution of
    tau_hat is therefore ~ +-20 % of the elapsed time, and the maximised
    statistic is a lower bound on the true max over all split points, tight to
    within the variation of the statistic between adjacent grid nodes.

    CAUSALITY: the grid is a module-level constant of *segment lengths*; at
    index t only the nodes with enough points inside online[:t+1] are used, so
    the candidate set is a function of t alone and never of n_online.  All
    nulls come from the historical segment only.
"""
from __future__ import annotations

import numpy as np

from sbr.features.base import register

AR_ORDER = 6

# geometric candidate grid of segment lengths (ratio ~1.5)
M_H = np.array([8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512], dtype=np.int64)
# engine P needs 2m points, so its grid stops earlier (2*256 = 512 <= min n_hist/2)
M_P = np.array([8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256], dtype=np.int64)

# fixed trailing offsets -- the CONTROL family.  Chosen from the grids so they
# reuse the same nulls (no extra cost) and are exactly the same statistic.
FIX_H = (16, 48, 128, 384)
FIX_P = (16, 48, 128, 256)

STREAMS = ("rsq", "rlab", "rl1")
STAB_W = 16
MIN_NULL = 200
MAX_NULL = 1000
CLIP = 12.0
QFLOOR = 1e-3


# ----------------------------------------------------------------- AR(6) fit
def _fit_ar(z: np.ndarray, p: int) -> np.ndarray:
    if p <= 0 or len(z) < 10 * p + 10:
        return np.zeros(p)
    X = np.column_stack([z[p - k - 1: len(z) - k - 1] for k in range(p)])
    y = z[p:]
    XtX = X.T @ X + 1e-6 * np.eye(p) * len(y)
    try:
        return np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        return np.zeros(p)


def _ar_apply(z: np.ndarray, coef: np.ndarray, warm: np.ndarray) -> np.ndarray:
    """Causal AR residuals of ``z`` with lags warm-started from ``warm``.

    Written as an explicit accumulation over lags (never a BLAS gemv) so the
    floating-point summation order is identical for a prefix and for the full
    array -- prefix invariance is checked bitwise.
    """
    p = len(coef)
    if p == 0:
        return z.copy()
    pad = np.concatenate([warm[-p:], z]) if len(warm) >= p else np.concatenate([np.zeros(p), z])
    out = z.astype(np.float64, copy=True)
    n = len(z)
    for k in range(p):
        out -= coef[k] * pad[p - k - 1: p - k - 1 + n]
    return out


def _streams(e: np.ndarray, e_prev0: float) -> dict:
    """The three monitored per-point transforms of a residual stream."""
    lag = np.empty_like(e)
    lag[0] = e_prev0
    lag[1:] = e[:-1]
    return {
        "rsq": e * e,
        "rlab": np.log(np.abs(e) + 1e-3),
        "rl1": e * lag,
    }


# ------------------------------------------------------------- historical null
def _hist_null(a: np.ndarray, M: np.ndarray, kind: str, n_online: int):
    """Empirical null of the segment statistic at every usable grid length.

    Returns per-window location/scale, the sorted null of |robust z| (used to
    convert each window's statistic to an exactly-uniform rank score so that
    windows are exchangeable under the null and the argmax is meaningful), and
    the sorted null of ``max over the first k windows`` for every k (used to
    price in the multiplicity of the scan at the matched candidate count).

    Grid nodes that can never fit inside the online segment are dropped.  This
    is prefix-safe: row t only ever consults nodes with ``need <= t+1`` and the
    k-matched null of the max over the first k nodes involves only those same
    nodes, so dropping unreachable tail nodes cannot change any emitted row.
    """
    mult = 1 if kind == "H" else 2
    La = len(a)
    # the null sample (end-point set, stride, size) is pinned to the FULL grid
    # constant, never to the truncated one -- otherwise the null itself would
    # depend on n_online and every row would silently change with the series
    # length.  Only the *number of nodes evaluated* is truncated.
    e0 = int(mult * int(M[-1]) - 1)
    if La - e0 < MIN_NULL:
        return None
    K = int(np.searchsorted(M * mult, n_online, side="right"))
    if K == 0:
        return None
    M = M[:K]
    ends = np.arange(e0, La, dtype=np.int64)
    stride = max(1, len(ends) // MAX_NULL)
    ends = ends[::stride]
    N = len(ends)
    c = np.concatenate(([0.0], np.cumsum(a)))
    med = np.empty(K)
    sd = np.empty(K)
    SZ = np.empty((K, N))
    U = np.empty((K, N))
    ranks = (np.arange(N) + 0.5) / N
    for j in range(K):
        m = int(M[j])
        v = (c[ends + 1] - c[ends + 1 - m]) / m
        if kind == "P":
            v = v - (c[ends + 1 - m] - c[ends + 1 - 2 * m]) / m
        sv = np.sort(v)
        mu = float(sv[N // 2])
        q1 = float(sv[int(0.25 * N)])
        q3 = float(sv[int(0.75 * N)])
        s = (q3 - q1) / 1.349
        if not np.isfinite(s) or s <= 0:
            s = max(float(sv[-1] - sv[0]) * 1e-3, 1e-12)
        med[j] = mu
        sd[j] = s
        z = np.abs((v - mu) / s)
        order = np.argsort(z, kind="stable")
        SZ[j] = z[order]
        U[j, order] = ranks
    CUS = np.sort(np.maximum.accumulate(U, axis=0), axis=1)
    return {"M": M, "kind": kind, "med": med, "sd": sd, "SZ": SZ, "CUS": CUS, "N": N}


# ------------------------------------------------------------------- online
def _online(a: np.ndarray, nl: dict, n: int):
    """Per-index localisation over the candidate grid.

    Returns (Z, U, needs) with Z/U of shape (K, n); NaN where the candidate
    does not fit inside online[:t+1].
    """
    M = nl["M"]
    K = len(M)
    kind = nl["kind"]
    med, sd, SZ, N = nl["med"], nl["sd"], nl["SZ"], nl["N"]
    c = np.concatenate(([0.0], np.cumsum(a)))
    Z = np.full((K, n), np.nan)
    U = np.full((K, n), np.nan)
    needs = np.empty(K, dtype=np.int64)
    for j in range(K):
        m = int(M[j])
        need = m if kind == "H" else 2 * m
        needs[j] = need
        if need > n:
            continue
        idx = np.arange(need - 1, n, dtype=np.int64)
        v = (c[idx + 1] - c[idx + 1 - m]) / m
        if kind == "P":
            v = v - (c[idx + 1 - m] - c[idx + 1 - 2 * m]) / m
        z = (v - med[j]) / sd[j]
        Z[j, idx] = z
        U[j, idx] = (np.searchsorted(SZ[j], np.abs(z), side="left") + 0.5) / N
    return Z, U, needs


def _roll_std(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing-w standard deviation of x, NaN unless all w values are finite."""
    n = len(x)
    out = np.full(n, np.nan)
    if w > n:
        return out
    ok = np.isfinite(x).astype(np.float64)
    xf = np.where(np.isfinite(x), x, 0.0)
    c1 = np.concatenate(([0.0], np.cumsum(xf)))
    c2 = np.concatenate(([0.0], np.cumsum(xf * xf)))
    ck = np.concatenate(([0.0], np.cumsum(ok)))
    s1 = c1[w:] - c1[:-w]
    s2 = c2[w:] - c2[:-w]
    sk = ck[w:] - ck[:-w]
    var = s2 / w - (s1 / w) ** 2
    val = np.where(sk >= w, np.sqrt(np.maximum(var, 0.0)), np.nan)
    out[w - 1:] = val
    return out


def _localise(Z, U, needs, M, n):
    """max / argmax / runner-up over the available candidates at every index."""
    len(M)
    ar = np.arange(n)
    A = np.where(np.isfinite(U), U, -1.0)
    i1 = A.argmax(axis=0)
    u1 = A[i1, ar]
    A2 = A.copy()
    A2[i1, ar] = -2.0
    i2 = A2.argmax(axis=0)
    u2 = A2[i2, ar]
    ok = u1 >= 0.0
    mhat = M[i1].astype(np.float64)
    lm = np.where(ok, np.log2(mhat), np.nan)
    rel = np.where(ok, 1.0 - mhat / (ar + 1.0), np.nan)
    gap = np.where(ok & (u2 >= 0.0), u1 - u2, np.nan)
    sz = np.where(ok, np.clip(Z[i1, ar], -CLIP, CLIP), np.nan)
    u1 = np.where(ok, u1, np.nan)
    k = np.searchsorted(needs, ar + 1, side="right")
    return u1, sz, lm, rel, gap, k


def _calibrate_max(u1, k, CUS, N):
    """Upper-tail surprise of the maximised score against the k-matched null."""
    n = len(u1)
    q = np.full(n, np.nan)
    kk = np.unique(k[np.isfinite(u1) & (k > 0)])
    for kv in kk:
        sel = (k == kv) & np.isfinite(u1)
        if not sel.any():
            continue
        row = CUS[int(kv) - 1]
        p = np.searchsorted(row, u1[sel], side="left") / float(N)
        q[sel] = -np.log10(np.maximum(1.0 - p, QFLOOR))
    return q


# --------------------------------------------------------------------- build
@register("m06_loc", version="1", owner="agent_loc")
def build(ctx):
    n = ctx.n
    cols: list[str] = []
    out: list[np.ndarray] = []
    nanv = np.full(n, np.nan)

    hp = ctx.hp
    zh = (np.asarray(ctx.hist, dtype=np.float64) - hp.mu) / hp.sd
    zo = (np.asarray(ctx.online, dtype=np.float64) - hp.mu) / hp.sd

    coef = _fit_ar(zh, AR_ORDER)
    eh_full = _ar_apply(zh, coef, np.zeros(AR_ORDER))
    eh = eh_full[AR_ORDER:]                      # drop the un-warmed head
    sig = float(np.std(eh, ddof=1)) if len(eh) > 1 else 1.0
    sig = max(sig, 1e-9)
    eh = eh / sig
    eo = _ar_apply(zo, coef, zh) / sig

    hs = _streams(eh, 0.0)
    os_ = _streams(eo, float(eh[-1]) if len(eh) else 0.0)

    engines = (("h", "H", M_H, FIX_H), ("p", "P", M_P, FIX_P))

    for etag, kind, M, fixo in engines:
        for st in STREAMS:
            nl = _hist_null(hs[st], M, kind, n)
            base = f"{etag}_{st}"
            if nl is None:
                for suf in ("q", "sz", "lm", "rel", "gap", "stab"):
                    cols.append(f"loc_{base}_{suf}")
                    out.append(nanv.copy())
                for m in fixo:
                    cols.append(f"fx_{base}_{m}")
                    out.append(nanv.copy())
                continue

            Ma = nl["M"]
            Z, U, needs = _online(os_[st], nl, n)
            u1, sz, lm, rel, gap, k = _localise(Z, U, needs, Ma, n)
            q = _calibrate_max(u1, k, nl["CUS"], nl["N"])
            stab = _roll_std(lm, STAB_W)

            cols.append(f"loc_{base}_q")
            out.append(q)
            cols.append(f"loc_{base}_sz")
            out.append(sz)
            cols.append(f"loc_{base}_lm")
            out.append(lm)
            cols.append(f"loc_{base}_rel")
            out.append(rel)
            cols.append(f"loc_{base}_gap")
            out.append(gap)
            cols.append(f"loc_{base}_stab")
            out.append(stab)

            pos = {int(mm): j for j, mm in enumerate(Ma)}
            for m in fixo:
                cols.append(f"fx_{base}_{m}")
                j = pos.get(int(m))
                out.append(nanv.copy() if j is None else np.clip(Z[j], -CLIP, CLIP))

    A = np.column_stack(out).astype(np.float32)
    A[~np.isfinite(A)] = np.nan
    return cols, A
