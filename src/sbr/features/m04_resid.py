"""m04_resid -- residual / innovation representations and the over-whitening test.

QUESTION
    Does removing predictable dynamics (AR filtering, volatility normalisation)
    make a structural break easier or harder to see?

CONSTRUCTION
    Every representation is a map  x -> e  fitted on the HISTORICAL segment only
    and applied causally forward over the online segment:

      raw   z = (x - mu_h)/sd_h                      (constant historical sd)
      ar1/ar2/ar5   AR(p) residual, fixed order, OLS on standardised history,
                    filtered forward with ``ar_filter_causal`` warmed by the
                    historical tail
      arR   ridge-regularised AR(3)   (lambda = 0.05 * n on a unit-variance Gram)
      arH   Huber-IRLS AR(2), 3 fixed reweighting steps (outlier-robust coefs)
      vol   z / sqrt(EWMA_pred(z^2, halflife 22))    -- predictive, state from history
      volM  z / EWMA_pred(min(|z|,4), halflife 63)   -- robust (winsorised) scale
      volG  z / sqrt(GARCH(1,1)_pred)  -- variance-targeted, 12-point deterministic
                                          QMLE grid fitted on history
      cmb   AR(2) residual, then EWMA(22) volatility normalisation of the residual

    Each stream is divided by the standard deviation of the *historical* stream,
    so all representations are on a common "1 = historical innovation sd" scale.

MONITORS  (all null-calibrated per series)
    mean, var, abs, tail-rate, ACF(1), ACF(1) of squared, and a Gaussian
    generalised-log-likelihood-ratio channel
        LLR/w = mean(e^2) - log(var(e)) - 1
    which is the exact 2*LLR per point of N(m,s^2) against the null N(0,1) and
    therefore reacts to a mean shift and a variance change together.

NULL
    Own calibration engine: for a window length w the null is the empirical
    distribution of the SAME statistic over every contiguous length-w window of
    the residual stream computed ON HISTORY.  Median and IQR-sigma are stored on
    a log-spaced grid and log-interpolated to any length, exactly as
    ``sbr.nullcal`` does for the raw transforms.  Only historical data enters
    the null; online values only ever enter as the query.

SIGNAL / FALSE SIGNAL / DISAMBIGUATOR
    see research/reports/agent04_residual.md -- one story per representation.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from sbr.features.base import register
from sbr.transforms import _ar_resid, _fit_ar, ar_filter_causal

# ---------------------------------------------------------------- configuration
GRID = np.array([6, 16, 32, 80, 200, 512])   # log-spaced null-interpolation nodes
W_TR = 32                 # single trailing scale for the composite LLR channel
MAXNULL = 600             # cap on null sample size (deterministic stride)
CLIP = 12.0
ECLIP = 8.0
BURN_MAX = 200            # history burn-in for recursive filters

TIER_A = ("raw", "ar1", "ar2", "ar5", "vol", "cmb")
TIER_B = ("arR", "arH", "volG", "volM")

A_MON = ("mean", "var", "abs", "tail", "acf1", "acf1sq")   # + llr, + w32 llr
B_MON = ("var", "acf1sq")                                   # + llr

_HL22 = 1.0 - 0.5 ** (1.0 / 22.0)      # EWMA alpha, halflife 22
_HL63 = 1.0 - 0.5 ** (1.0 / 63.0)      # EWMA alpha, halflife 63
_GA = (0.03, 0.06, 0.10, 0.16)
_GB = (0.80, 0.88, 0.94)


# ------------------------------------------------------------------- utilities
def _ewma_pred(x: np.ndarray, alpha: float, x0: float) -> np.ndarray:
    """Predictive EWMA: element t uses x[:t] only (never x[t])."""
    y = lfilter([alpha], [1.0, -(1.0 - alpha)], x, zi=np.array([(1.0 - alpha) * x0]))[0]
    out = np.empty_like(y)
    out[0] = x0
    out[1:] = y[:-1]
    return out


def _garch_pred(x: np.ndarray, a: float, b: float, v0: float = 1.0) -> np.ndarray:
    """Predictive GARCH(1,1) variance with variance targeting to 1."""
    om = (1.0 - a - b) * 1.0
    u = np.empty_like(x)
    u[0] = om + a * v0
    u[1:] = om + a * x[:-1]
    return lfilter([1.0], [1.0, -b], u, zi=np.array([b * v0]))[0]


def _fit_ar_ridge(z: np.ndarray, p: int, lam: float) -> np.ndarray:
    if p <= 0 or len(z) < 10 * p + 10:
        return np.zeros(p)
    X = np.column_stack([z[p - k - 1: len(z) - k - 1] for k in range(p)])
    y = z[p:]
    XtX = X.T @ X + lam * len(y) * np.eye(p)
    try:
        return np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        return np.zeros(p)


def _fit_ar_huber(z: np.ndarray, p: int, iters: int = 3) -> np.ndarray:
    """Huber IRLS AR fit, fixed iteration count (deterministic, cheap)."""
    if p <= 0 or len(z) < 10 * p + 10:
        return np.zeros(p)
    X = np.column_stack([z[p - k - 1: len(z) - k - 1] for k in range(p)])
    y = z[p:]
    n = len(y)
    eye = np.eye(p)
    coef = _fit_ar(z, p)
    for _ in range(iters):
        r = y - X @ coef
        s = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 1e-9)
        w = np.minimum(1.0, 1.345 * s / np.maximum(np.abs(r), 1e-12))
        Xw = X * w[:, None]
        try:
            coef = np.linalg.solve(Xw.T @ X + 1e-6 * n * eye, Xw.T @ y)
        except np.linalg.LinAlgError:
            break
    return coef


def _std(a: np.ndarray) -> float:
    return max(float(a.std(ddof=1)) if len(a) > 1 else 1.0, 1e-9)


# ------------------------------------------------------------- null calibration
class _Null:
    """median / IQR-sigma of a statistic over contiguous historical windows."""

    __slots__ = ("lg", "mu", "sd")

    def __init__(self, grid: np.ndarray, mu: np.ndarray, sd: np.ndarray):
        self.lg = np.log(grid.astype(np.float64))
        self.mu = mu
        self.sd = sd

    def z(self, L, v) -> np.ndarray:
        x = np.log(np.clip(np.atleast_1d(np.asarray(L, dtype=np.float64)), 1.0, None))
        mu = np.interp(x, self.lg, self.mu)
        sd = np.exp(np.interp(x, self.lg, np.log(self.sd)))
        return (np.atleast_1d(np.asarray(v, dtype=np.float64)) - mu) / np.maximum(sd, 1e-12)


_KQ = np.array([0.25, 0.5, 0.75])


def _summ(rm: np.ndarray) -> tuple[float, float]:
    """Median and IQR-sigma of a null sample -- cheap, deterministic.

    Uses a fixed stride subsample (never random) and a single partition call;
    np.quantile's generic machinery is ~5x more expensive and this runs ~330
    times per series.
    """
    m = len(rm)
    if m > MAXNULL:
        rm = rm[:: (m + MAXNULL - 1) // MAXNULL]
        m = len(rm)
    if m < 8:
        mu = float(rm.mean())
        return mu, max(float(rm.std()), 1e-12)
    k = ((m - 1) * _KQ).astype(np.intp)
    p = np.partition(rm, k)
    q1, med, q3 = float(p[k[0]]), float(p[k[1]]), float(p[k[2]])
    mu = float(rm.sum()) / m
    var = max(float(rm @ rm) / m - mu * mu, 0.0)
    return med, max((q3 - q1) / 1.349, var ** 0.5, 1e-12)


def _null_mean(ah: np.ndarray, grid: np.ndarray) -> _Null:
    """Null of the ROLLING MEAN of a per-point transform."""
    H = len(ah)
    c = np.concatenate([[0.0], np.cumsum(ah)])
    mus = np.empty(len(grid))
    sds = np.empty(len(grid))
    for j, w in enumerate(grid):
        w = int(w)
        rm = np.array([ah.mean()]) if w >= H else (c[w:] - c[:-w]) / w
        mus[j], sds[j] = _summ(rm)
    return _Null(grid, mus, sds)


def _llr_stat(m: np.ndarray, q: np.ndarray) -> np.ndarray:
    s2 = np.maximum(q - m * m, 1e-6)
    return q - np.log(s2) - 1.0


def _null_llr(eh: np.ndarray, e2h: np.ndarray, grid: np.ndarray) -> _Null:
    """Null of the Gaussian GLR statistic (a non-linear function of two means)."""
    H = len(eh)
    c1 = np.concatenate([[0.0], np.cumsum(eh)])
    c2 = np.concatenate([[0.0], np.cumsum(e2h)])
    mus = np.empty(len(grid))
    sds = np.empty(len(grid))
    for j, w in enumerate(grid):
        w = int(w)
        if w >= H:
            m = np.array([eh.mean()])
            q = np.array([e2h.mean()])
        else:
            m = (c1[w:] - c1[:-w]) / w
            q = (c2[w:] - c2[:-w]) / w
        mus[j], sds[j] = _summ(_llr_stat(m, q))
    return _Null(grid, mus, sds)


# ------------------------------------------------------- representation builder
def _reps(ctx) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {name: (hist_stream, online_stream)} -- both unit-sd on history."""
    hp = ctx.hp
    zh = (ctx.hist - hp.mu) / hp.sd
    zo = (ctx.online - hp.mu) / hp.sd
    h = len(zh)
    burn = min(BURN_MAX, max(h // 5, 1))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # ---- constant historical sd (the control)
    out["raw"] = (zh, zo)

    # ---- fixed-order AR residuals
    ar_cache: dict[str, np.ndarray] = {}
    for tag, p in (("ar1", 1), ("ar2", 2), ("ar5", 5)):
        coef = hp.ar_coef if p == 2 else _fit_ar(zh, p)
        rh = _ar_resid(zh, coef)
        ro = ar_filter_causal(zo, coef, zh)
        s = _std(rh)
        out[tag] = (rh / s, ro / s)
        ar_cache[tag] = coef

    # ---- regularised / robust AR coefficient estimates
    for tag, coef in (("arR", _fit_ar_ridge(zh, 3, 0.05)),
                      ("arH", _fit_ar_huber(zh, 2))):
        rh = _ar_resid(zh, coef)
        ro = ar_filter_causal(zo, coef, zh)
        s = _std(rh)
        out[tag] = (rh / s, ro / s)

    # ---- volatility normalisation (state initialised from history, run forward)
    zc = np.concatenate([zh, zo])
    x2 = zc * zc

    v = _ewma_pred(x2, _HL22, 1.0)
    e = zc / np.sqrt(np.maximum(v, 1e-9))
    s = _std(e[burn:h])
    out["vol"] = (e[burn:h] / s, e[h:] / s)

    sm = _ewma_pred(np.minimum(np.abs(zc), 4.0), _HL63, float(np.mean(np.minimum(np.abs(zh), 4.0))))
    e = zc / np.maximum(sm, 1e-9)
    s = _std(e[burn:h])
    out["volM"] = (e[burn:h] / s, e[h:] / s)

    # GARCH(1,1): deterministic 12-point QMLE grid, fitted on HISTORY ONLY
    xh = x2[:h]
    best, ba, bb = np.inf, _GA[0], _GB[0]
    for a in _GA:
        for b in _GB:
            if a + b > 0.995:
                continue
            vv = _garch_pred(xh, a, b)[burn:]
            vv = np.maximum(vv, 1e-9)
            loss = float(np.sum(np.log(vv) + xh[burn:] / vv))
            if loss < best:
                best, ba, bb = loss, a, b
    v = np.maximum(_garch_pred(x2, ba, bb), 1e-9)
    e = zc / np.sqrt(v)
    s = _std(e[burn:h])
    out["volG"] = (e[burn:h] / s, e[h:] / s)

    # ---- combination: AR(2) residual then EWMA volatility normalisation
    rh2, ro2 = out["ar2"]
    rc = np.concatenate([rh2, ro2])
    vr = _ewma_pred(rc * rc, _HL22, 1.0)
    e = rc / np.sqrt(np.maximum(vr, 1e-9))
    hb = len(rh2)
    b2 = min(BURN_MAX, max(hb // 5, 1))
    s = _std(e[b2:hb])
    out["cmb"] = (e[b2:hb] / s, e[hb:] / s)

    return out


def _point_transforms(e: np.ndarray, prev: float, thr: float, want) -> dict[str, np.ndarray]:
    """Per-point transforms whose rolling mean is the monitored statistic."""
    ec = np.clip(e, -ECLIP, ECLIP)
    e2 = ec * ec
    d: dict[str, np.ndarray] = {"mean": e, "var": e2}
    if "abs" in want:
        d["abs"] = np.abs(ec)
    if "tail" in want:
        d["tail"] = (np.abs(ec) > thr).astype(np.float64)
    if "acf1" in want or "acf1sq" in want:
        pc = np.clip(prev, -ECLIP, ECLIP)
        lag = np.concatenate([[pc], ec[:-1]])
        if "acf1" in want:
            d["acf1"] = ec * lag
        if "acf1sq" in want:
            d["acf1sq"] = (e2 - 1.0) * (lag * lag - 1.0)
    return d


# ------------------------------------------------------------------ the module
@register("m04_resid", version="1", owner="agent4")
def build(ctx):
    n = ctx.n
    h = len(ctx.hist)
    grid = GRID[GRID <= max(h // 2, 5)]
    if len(grid) == 0:
        grid = np.array([5])
    L = np.arange(1, n + 1, dtype=np.float64)
    reps = _reps(ctx)

    cols: list[str] = []
    out: list[np.ndarray] = []
    nan = np.full(n, np.nan)

    for tag in TIER_A + TIER_B:
        eh, eo = reps[tag]
        tier_a = tag in TIER_A
        want = A_MON if tier_a else B_MON
        ec_h = np.clip(eh, -ECLIP, ECLIP)
        thr = float(np.quantile(np.abs(ec_h), 0.95)) if len(ec_h) else 3.0

        th = _point_transforms(eh, eh[0] if len(eh) else 0.0, thr, want)
        to = _point_transforms(eo, eh[-1] if len(eh) else 0.0, thr, want)

        # ---- rolling-mean monitors, expanding window, null-calibrated
        for m in want:
            ah = th[m][1:] if m in ("acf1", "acf1sq") else th[m]
            nl = _null_mean(ah, grid)
            v = np.cumsum(to[m]) / L if n else np.zeros(0)
            cols.append(f"{tag}_e_{m}")
            out.append(np.clip(nl.z(L, v), -CLIP, CLIP) if n else nan)

        # ---- Gaussian GLR channel: expanding + one trailing scale
        nll = _null_llr(eh, np.clip(eh, -ECLIP, ECLIP) ** 2, grid)
        c1 = np.concatenate([[0.0], np.cumsum(to["mean"])])
        c2 = np.concatenate([[0.0], np.cumsum(to["var"])])
        st = _llr_stat(c1[1:] / L, c2[1:] / L)
        z = np.clip(nll.z(L, st), -CLIP, CLIP)
        z[L < 8] = np.nan
        cols.append(f"{tag}_e_llr")
        out.append(z)

        if tier_a:
            zz = np.full(n, np.nan)
            if n >= W_TR:
                m = (c1[W_TR:] - c1[:-W_TR]) / W_TR
                q = (c2[W_TR:] - c2[:-W_TR]) / W_TR
                zz[W_TR - 1:] = np.clip(nll.z(W_TR, _llr_stat(m, q)), -CLIP, CLIP)
            cols.append(f"{tag}_w{W_TR}_llr")
            out.append(zz)

    A = np.column_stack(out).astype(np.float32) if n else np.zeros((0, len(cols)), np.float32)
    A[~np.isfinite(A)] = np.nan
    return cols, A
