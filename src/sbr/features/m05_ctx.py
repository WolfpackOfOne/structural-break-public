"""m05_ctx -- historical-context / DGP fingerprint of the break-free segment.

The historical segment is guaranteed break-free and is fully visible from the
first online step, so every column here is CONSTANT along a series' online
trajectory.  That has a sharp consequence for TS-AUC, which only ever ranks
series against each other at a fixed online index t:

  * a context column can NEVER act as a within-series detector -- it does not
    move when the break happens;
  * it can still change the ranking in exactly two ways --
      H1 (LEVEL / PRIOR)  it shifts a whole series' score up or down because the
          generator that produced that history is a priori more or less likely
          to break;
      H2 (INTERACTION)    it tells the model how to *read* the online evidence,
          e.g. "a 3-sigma excursion in a heavy-tailed, strongly
          vol-clustered DGP is unremarkable; in a thin-tailed IID one it is a
          break".
    H1 is a generator artifact and will not survive a well-built private set;
    H2 is real modelling.  The permutation control in the accompanying report
    is what separates them.

SIGNAL / ALPHA
    Per family:
    - MOMENTS/TAILS/ENTROPY: pin down the null's shape so that the calibrated
      online evidence in m00_core can be discounted (heavy tails, high kurtosis
      -> exceedances are cheap) or amplified (near-Gaussian -> exceedances are
      expensive).
    - AR / PACF / VOL-CLUSTERING: the effective sample size of the online
      window.  A strongly persistent series has far fewer independent online
      observations than its length suggests, so the same z-score is much weaker
      evidence.
    - SPECTRAL / WAVELET / DFA: long-memory and low-frequency power make slow
      drifts look like level shifts; they are the main confounder of a
      mean-break detector.
    - RECENT-vs-WHOLE: the last 100/250/500 historical points are the immediate
      pre-online regime.  If the tail of history already differs from the bulk
      of history (vol ratio, kurtosis diff, AR diff), then the null that
      m00_core calibrates against is a mixture and the online segment starts
      off-centre through no fault of any break.

FALSE SIGNAL
    Every column here is by construction blind to the break, so it cannot
    produce a "false detection" in the usual sense.  Its failure mode is worse
    and quieter: if the data generator correlates DGP family with break
    probability (e.g. only the heavy-tailed regime ever gets a break injected),
    the model learns a prior over generators and reports a fine dev TS-AUC that
    is pure leakage of the simulation design.  n_hist is the loudest such
    suspect -- it is metadata, not physics.

DISAMBIGUATOR
    The permuted-context control: rebuild the identical column block but
    reassign each series' context vector to a different series.  A permuted
    block preserves the marginal distribution (so any pure level/prior effect
    that LightGBM can exploit from the marginal survives) and destroys the
    series match (so any interaction with that series' own online evidence
    dies).  true - permuted is the honest H2 gain.

Owner: agent8.  49 columns, all series-constant.
"""
from __future__ import annotations

import numpy as np

from sbr.features.base import register

RECENT = (100, 250, 500)
_EPS = 1e-12


# ----------------------------------------------------------------- utilities
def _safe(v, lo=-50.0, hi=50.0):
    v = float(v)
    if not np.isfinite(v):
        return np.nan
    return float(np.clip(v, lo, hi))


def _acf(x: np.ndarray, lags) -> np.ndarray:
    """Sample autocorrelations at the requested lags (x need not be centred)."""
    n = len(x)
    xc = x - x.mean()
    d = float(xc @ xc)
    out = np.zeros(len(lags))
    if d <= _EPS or n < 20:
        return out
    for j, k in enumerate(lags):
        if k >= n:
            continue
        out[j] = float(xc[k:] @ xc[:-k]) / d
    return out


def _pacf_from_acf(r: np.ndarray, p: int) -> np.ndarray:
    """Durbin-Levinson partial autocorrelations from acf lags 1..p."""
    phi = np.zeros((p + 1, p + 1))
    pac = np.zeros(p)
    if p == 0:
        return pac
    phi[1, 1] = r[0]
    pac[0] = r[0]
    v = 1.0 - r[0] ** 2
    for k in range(2, p + 1):
        if v <= _EPS:
            break
        num = r[k - 1] - sum(phi[k - 1, j] * r[k - j - 1] for j in range(1, k))
        a = num / v
        phi[k, k] = a
        pac[k - 1] = a
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - a * phi[k - 1, k - j]
        v *= (1.0 - a * a)
    return pac


def _ols_ar(z: np.ndarray, p: int):
    """OLS AR(p) on ``z``; returns (coefs, residual variance ratio)."""
    n = len(z)
    if n < 10 * p + 10:
        return np.zeros(p), 1.0
    X = np.column_stack([z[p - k - 1: n - k - 1] for k in range(p)])
    y = z[p:]
    XtX = X.T @ X + 1e-8 * np.eye(p) * len(y)
    try:
        c = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        return np.zeros(p), 1.0
    r = y - X @ c
    vy = float(y.var())
    ratio = float(r.var()) / vy if vy > _EPS else 1.0
    return c, ratio


def _hill(y: np.ndarray, frac: float = 0.05) -> float:
    """Hill tail index (alpha) of the positive exceedances of ``y``."""
    n = len(y)
    k = max(15, int(frac * n))
    ys = np.sort(y)
    if k >= n:
        k = n - 1
    u = ys[n - k - 1]
    if u <= _EPS:
        return np.nan
    ex = ys[n - k:]
    m = float(np.mean(np.log(np.maximum(ex, u * (1 + 1e-9)) / u)))
    if m <= _EPS:
        return np.nan
    return 1.0 / m


def _spacing_entropy(z: np.ndarray) -> float:
    """Vasicek m-spacing differential entropy of the standardised history."""
    n = len(z)
    m = max(2, int(np.sqrt(n)))
    zs = np.sort(z)
    d = zs[m:] - zs[:-m]
    d = np.maximum(d, 1e-9)
    return float(np.mean(np.log(d * n / (2.0 * m))))


def _var_ratio(x: np.ndarray, q: int) -> float:
    """Overlapping variance ratio VR(q): 1 under a random walk of increments."""
    n = len(x)
    if n < 5 * q:
        return np.nan
    v1 = float(x.var())
    if v1 <= _EPS:
        return np.nan
    c = np.concatenate([[0.0], np.cumsum(x)])
    s = c[q:] - c[:-q]
    return float(s.var()) / (q * v1)


def _dfa(x: np.ndarray) -> float:
    """DFA-1 scaling exponent (0.5 = white noise, >0.5 = long memory)."""
    n = len(x)
    y = np.cumsum(x - x.mean())
    scales = [s for s in (16, 32, 64, 128, 256, 512) if 4 * s <= n]
    if len(scales) < 3:
        return np.nan
    fs = []
    for s in scales:
        nb = n // s
        seg = y[: nb * s].reshape(nb, s)
        t = np.arange(s, dtype=np.float64)
        t = t - t.mean()
        tt = float(t @ t)
        b = (seg @ t) / tt
        a = seg.mean(axis=1)
        resid = seg - (a[:, None] + b[:, None] * t[None, :])
        fs.append(np.sqrt(float(np.mean(resid ** 2))))
    fs = np.asarray(fs)
    if not np.all(np.isfinite(fs)) or np.any(fs <= 0):
        return np.nan
    ls = np.log(np.asarray(scales, dtype=np.float64))
    lf = np.log(fs)
    return float(np.polyfit(ls, lf, 1)[0])


def _spectral(z: np.ndarray):
    """Band powers, spectral entropy, dominant frequency, spectral centroid."""
    n = len(z)
    m = 1 << int(np.floor(np.log2(n)))          # power-of-two tail, deterministic
    seg = z[-m:] - z[-m:].mean()
    w = np.hanning(m)
    P = np.abs(np.fft.rfft(seg * w)) ** 2
    P = P[1:]                                   # drop DC
    tot = float(P.sum())
    if tot <= _EPS or len(P) < 8:
        return [np.nan] * 4, np.nan, np.nan, np.nan
    p = P / tot
    f = np.arange(1, len(P) + 1) / float(m)     # cycles per sample, (0, 0.5]
    edges = (0.0, 0.03125, 0.0625, 0.25, 0.5 + 1e-9)
    bands = []
    for i in range(4):
        sel = (f > edges[i]) & (f <= edges[i + 1])
        bands.append(float(p[sel].sum()) if sel.any() else 0.0)
    ent = float(-(p * np.log(np.maximum(p, 1e-15))).sum() / np.log(len(p)))
    dom = float(f[int(np.argmax(p))])
    cen = float((f * p).sum())
    return [np.log(b + 1e-6) for b in bands], ent, dom, cen


def _wavelet(z: np.ndarray):
    """Haar pyramid energy fractions per level + log-log energy slope."""
    a = z - z.mean()
    levels = 6
    en = []
    for _ in range(levels):
        if len(a) < 4:
            break
        if len(a) % 2:
            a = a[:-1]
        d = (a[0::2] - a[1::2]) / np.sqrt(2.0)
        a = (a[0::2] + a[1::2]) / np.sqrt(2.0)
        en.append(float(d @ d))
    if len(en) < 4:
        return np.nan, np.nan, np.nan
    e = np.asarray(en, dtype=np.float64)
    tot = float(e.sum()) + float(a @ a)
    if tot <= _EPS:
        return np.nan, np.nan, np.nan
    frac = e / tot
    j = np.arange(1, len(e) + 1, dtype=np.float64)
    pos = e > 0
    slope = float(np.polyfit(j[pos], np.log(e[pos]), 1)[0]) if pos.sum() >= 3 else np.nan
    return float(frac[0]), float(frac[1]), slope


def _kurt(z: np.ndarray) -> float:
    s = float(z.std())
    if s <= _EPS:
        return np.nan
    return float(np.mean(((z - z.mean()) / s) ** 4))


# ------------------------------------------------------------------- module
@register("m05_ctx", version="1", owner="agent8")
def build(ctx):
    h = np.asarray(ctx.hist, dtype=np.float64)
    hp = ctx.hp
    nh = len(h)
    z = (h - hp.mu) / hp.sd

    cols: list[str] = []
    vals: list[float] = []

    def add(name, v, lo=-50.0, hi=50.0):
        cols.append(name)
        vals.append(_safe(v, lo, hi))

    # ---------------- A. moments / tails / entropy (whole history), 12
    add("h_log_sd", np.log(hp.sd + 1e-12))
    add("h_log_sd_over_mad", np.log((hp.sd + 1e-12) / (hp.mad + 1e-12)))
    add("h_skew", float(np.mean(z ** 3)) if nh > 2 else np.nan, -30, 30)
    add("h_log_kurt", np.log(max(_kurt(z), 1e-3)))
    add("h_bowley", (hp.q75 + hp.q25 - 2.0 * hp.med) / hp.iqr)
    q = np.quantile(h, [0.125, 0.25, 0.375, 0.625, 0.75, 0.875])
    den = q[4] - q[1]
    add("h_moors", ((q[5] - q[3]) + (q[2] - q[0])) / den if den > _EPS else np.nan)
    add("h_hill_hi", _hill(np.maximum(h - hp.med, 0.0) + 1e-300), 0, 30)
    add("h_hill_lo", _hill(np.maximum(hp.med - h, 0.0) + 1e-300), 0, 30)
    add("h_qr_99_95", (hp.q99 - hp.q01) / max(hp.q95 - hp.q05, 1e-12), 0, 50)
    add("h_qr_95_iqr", (hp.q95 - hp.q05) / hp.iqr, 0, 50)
    add("h_entropy", _spacing_entropy((h - hp.med) / hp.mad))
    add("h_frac_gt3", float(np.mean(np.abs(z) > 3.0)))

    # ---------------- B. linear + nonlinear dependence, 11
    r = _acf(z, (1, 2, 3, 5, 10))
    add("h_acf1", r[0], -1.5, 1.5)
    add("h_acf2", r[1], -1.5, 1.5)
    add("h_acf3", r[2], -1.5, 1.5)
    pac = _pacf_from_acf(r[:3], 3)
    add("h_pacf1", pac[0], -1.5, 1.5)
    add("h_pacf2", pac[1], -1.5, 1.5)
    add("h_pacf3", pac[2], -1.5, 1.5)
    c3, ratio = _ols_ar(z, 3)
    add("h_ar3_log_unexpl", np.log(max(ratio, 1e-6)))
    za = np.abs(z)
    zs = z * z
    ra = _acf(za, (1,))
    rs = _acf(zs, (1, 5, 10))
    add("h_acf_abs1", ra[0], -1.5, 1.5)
    add("h_acf_sq1", rs[0], -1.5, 1.5)
    add("h_acf_sq5", rs[1], -1.5, 1.5)
    add("h_lb_sq_log", np.log1p(nh * float(rs[0] ** 2 + rs[1] ** 2 + rs[2] ** 2)))

    # ---------------- C. stationarity / long memory, 4
    add("h_vr10", _var_ratio(z, 10), 0, 50)
    add("h_vr50", _var_ratio(z, 50), 0, 50)
    rho = _acf(z, (1,))[0]
    add("h_adf_nrho", nh * (rho - 1.0), -5000, 100)
    add("h_dfa", _dfa(z), -1, 3)

    # ---------------- D. spectral / wavelet, 9
    bands, sent, dom, cen = _spectral(z)
    for i, b in enumerate(bands):
        add(f"h_logband{i}", b, -20, 5)
    add("h_spec_entropy", sent)
    add("h_spec_dom", dom, 0, 1)
    add("h_spec_centroid", cen, 0, 1)
    w1, w2, wsl = _wavelet(z)
    add("h_wav_l1", w1, 0, 1)
    add("h_wav_l2", w2, 0, 1)
    add("h_wav_slope", wsl, -20, 20)

    # ---------------- E. recent history vs whole history, 12
    kur_all = _kurt(z)
    for W in RECENT:
        seg = z[-W:] if nh >= W else z
        add(f"h_r{W}_log_vol_ratio", np.log((float(seg.std()) + 1e-9) / (float(z.std()) + 1e-9)))
        add(f"h_r{W}_mean_z", float(seg.mean()) * np.sqrt(len(seg)), -30, 30)
    s500 = z[-500:] if nh >= 500 else z
    s250 = z[-250:] if nh >= 250 else z
    s100 = z[-100:] if nh >= 100 else z
    add("h_r500_kurt_diff", (_kurt(s500) - kur_all) if np.isfinite(kur_all) else np.nan, -50, 50)
    add("h_r500_acf1_diff", _acf(s500, (1,))[0] - r[0], -2, 2)
    add("h_r250_tail_diff", float(np.mean(np.abs(s250) > 2.0)) - float(np.mean(np.abs(z) > 2.0)))
    qs = np.quantile(s500, [0.25, 0.75])
    add("h_r500_log_iqr_ratio",
        np.log((qs[1] - qs[0] + 1e-9) / (hp.iqr / hp.sd + 1e-9)))
    t = np.arange(len(s500), dtype=np.float64)
    t = t - t.mean()
    tt = float(t @ t)
    add("h_r500_trend", float((s500 - s500.mean()) @ t) / tt * len(s500) if tt > _EPS else np.nan,
        -50, 50)
    add("h_r100_max_absz", float(np.max(np.abs(s100))), 0, 60)

    # ---------------- F. structural metadata, 1  (SHORTCUT SUSPECT)
    add("h_n_hist", float(nh), 0, 1e6)

    v = np.asarray(vals, dtype=np.float32)
    A = np.repeat(v[None, :], ctx.n, axis=0)
    A[~np.isfinite(A)] = np.nan
    return cols, A
