"""m14_spectral_impulse -- spectral impulsiveness contrasts for Pilot 6."""
from __future__ import annotations

import numpy as np

from sbr.features.base import register

FREQS = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625], dtype=np.float64)
BANDS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("nyq", (0,)),
    ("high", (1,)),
    ("mid", (2, 3)),
    ("low", (4, 5)),
)
BAND_NAMES = [b for b, _ in BANDS]
SEG = 32
MIN_L = 16
GRID = np.array([16, 64, 256], dtype=np.int64)
NULL_N = 700
CLIP_Z = 10.0
EPS = 1e-12

CONTRAST_COLS = [f"contrast_sk_{b}" for b in BAND_NAMES] + [f"contrast_negent_{b}" for b in BAND_NAMES]
ENERGY_COLS = [f"energy_z_{b}" for b in BAND_NAMES]


def _cum(x: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64))])


def band_envelopes(z: np.ndarray) -> dict[str, np.ndarray]:
    """Length-SEG trailing Goertzel band energies at every endpoint."""
    z = np.asarray(z, dtype=np.float64)
    n = len(z)
    out = {name: np.full(n, np.nan, dtype=np.float64) for name, _ in BANDS}
    if n < SEG:
        return out
    j = np.arange(n, dtype=np.float64)
    by_freq = []
    for f in FREQS:
        c = _cum(z * np.cos(2.0 * np.pi * f * j))
        s = _cum(z * np.sin(2.0 * np.pi * f * j))
        C = (c[SEG:] - c[:-SEG]) / float(SEG)
        S = (s[SEG:] - s[:-SEG]) / float(SEG)
        by_freq.append(C * C + S * S)
    for name, idxs in BANDS:
        v = np.zeros(n - SEG + 1, dtype=np.float64)
        for idx in idxs:
            v += by_freq[idx]
        out[name][SEG - 1 :] = v
    return out


def _block_stats_from_sums(L, s1, s2, s3, s4, sc, sclog):
    Lf = np.asarray(L, dtype=np.float64)
    mean = s1 / np.maximum(Lf, 1.0)
    e2 = s2 / np.maximum(Lf, 1.0)
    e3 = s3 / np.maximum(Lf, 1.0)
    e4 = s4 / np.maximum(Lf, 1.0)
    var = np.maximum(e2 - mean * mean, EPS)
    m4 = np.maximum(e4 - 4.0 * mean * e3 + 6.0 * mean * mean * e2 - 3.0 * mean ** 4, 0.0)
    sk = m4 / np.maximum(var * var, EPS) - 3.0
    energy = np.log(np.maximum(mean, EPS))

    denom = np.maximum(sc, EPS)
    H = np.log(denom) - sclog / denom
    negent = 1.0 - H / np.maximum(np.log(np.maximum(Lf, 2.0)), EPS)
    negent = np.clip(negent, 0.0, 1.0)
    return {"energy": energy, "sk": sk, "negent": negent}


def _rolling_stats(env: np.ndarray, L: int, cap: float) -> dict[str, np.ndarray]:
    e = np.asarray(env, dtype=np.float64)
    e = e[np.isfinite(e)]
    if len(e) < L:
        return {k: np.empty(0, dtype=np.float64) for k in ("energy", "sk", "negent")}
    ec = np.minimum(e, cap)
    c1, c2, c3, c4 = (_cum(e ** p) for p in (1, 2, 3, 4))
    cc = _cum(ec)
    cclog = _cum(ec * np.log(np.maximum(ec, EPS)))
    sl = slice(L, None)
    s1 = c1[sl] - c1[:-L]
    s2 = c2[sl] - c2[:-L]
    s3 = c3[sl] - c3[:-L]
    s4 = c4[sl] - c4[:-L]
    sc = cc[sl] - cc[:-L]
    sclog = cclog[sl] - cclog[:-L]
    return _block_stats_from_sums(L, s1, s2, s3, s4, sc, sclog)


def _summ(v: np.ndarray) -> tuple[float, float]:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) > NULL_N:
        v = v[:: max(1, len(v) // NULL_N)]
    if len(v) < 8:
        return 0.0, 1.0
    q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
    sd = max(float((q[3] - q[1]) / 1.349), float((q[4] - q[0]) / 3.29), EPS)
    return float(q[2]), sd


def _null_params(hist_env: np.ndarray, cap: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    vals = {k: (np.empty(len(GRID)), np.empty(len(GRID))) for k in ("energy", "sk", "negent")}
    for i, L in enumerate(GRID):
        stats = _rolling_stats(hist_env, int(L), cap)
        for key in vals:
            med, sd = _summ(stats[key])
            vals[key][0][i] = med
            vals[key][1][i] = np.log(sd)
    return vals


def _loglin(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    lg = np.log(GRID.astype(np.float64))
    out = np.interp(x, lg, y)
    lo = x < lg[0]
    if np.any(lo):
        out[lo] = y[0] + (y[1] - y[0]) / (lg[1] - lg[0]) * (x[lo] - lg[0])
    hi = x > lg[-1]
    if np.any(hi):
        out[hi] = y[-1] + (y[-1] - y[-2]) / (lg[-1] - lg[-2]) * (x[hi] - lg[-1])
    return out


def _z(raw: np.ndarray, L: np.ndarray, params: dict[str, tuple[np.ndarray, np.ndarray]], key: str) -> np.ndarray:
    out = np.full(len(raw), np.nan, dtype=np.float64)
    ok = np.isfinite(raw) & (L >= MIN_L)
    if not np.any(ok):
        return out
    med, lsd = params[key]
    x = np.log(np.maximum(L[ok].astype(np.float64), 1.0))
    mu = _loglin(med, x)
    sd = np.exp(_loglin(lsd, x))
    out[ok] = np.clip((raw[ok] - mu) / np.maximum(sd, EPS), -CLIP_Z, CLIP_Z)
    return out


def _online_stats(env: np.ndarray, cap: float) -> tuple[dict[str, np.ndarray], np.ndarray]:
    e = np.asarray(env, dtype=np.float64)
    finite = np.isfinite(e)
    ez = np.where(finite, e, 0.0)
    ec = np.minimum(ez, cap)
    cN = np.concatenate([[0], np.cumsum(finite.astype(np.int64))])
    c1, c2, c3, c4 = (_cum(ez ** p) for p in (1, 2, 3, 4))
    cc = _cum(np.where(finite, ec, 0.0))
    cclog = _cum(np.where(finite, ec * np.log(np.maximum(ec, EPS)), 0.0))

    t = np.arange(len(e), dtype=np.int64)
    W = np.maximum((t + 1) // 2, 1)
    lo = np.maximum(t - W + 1, 0)
    hi = t + 1
    L = cN[hi] - cN[lo]
    s1 = c1[hi] - c1[lo]
    s2 = c2[hi] - c2[lo]
    s3 = c3[hi] - c3[lo]
    s4 = c4[hi] - c4[lo]
    sc = cc[hi] - cc[lo]
    sclog = cclog[hi] - cclog[lo]
    stats = _block_stats_from_sums(np.maximum(L, 1), s1, s2, s3, s4, sc, sclog)
    for key in stats:
        stats[key] = np.where(L >= MIN_L, stats[key], np.nan)
    return stats, L


def spectral_impulse_features(ctx) -> tuple[np.ndarray, np.ndarray]:
    hist_z = np.asarray(ctx.hist_tr["mean"], dtype=np.float64)
    online_z = np.asarray(ctx.tr["mean"], dtype=np.float64)
    hist_env = band_envelopes(hist_z)
    online_env = band_envelopes(online_z)

    contrast = np.full((ctx.n, len(CONTRAST_COLS)), np.nan, dtype=np.float64)
    energy = np.full((ctx.n, len(ENERGY_COLS)), np.nan, dtype=np.float64)

    for j, name in enumerate(BAND_NAMES):
        h = hist_env[name]
        hf = h[np.isfinite(h)]
        cap = float(np.quantile(hf, 0.99)) if len(hf) else 1.0
        cap = max(cap, EPS)
        params = _null_params(h, cap)
        stats, L = _online_stats(online_env[name], cap)
        ez = _z(stats["energy"], L, params, "energy")
        skz = _z(stats["sk"], L, params, "sk")
        nz = _z(stats["negent"], L, params, "negent")
        energy[:, j] = ez
        contrast[:, j] = ez * (-skz)
        contrast[:, j + len(BAND_NAMES)] = ez * (-nz)

    return contrast.astype(np.float32), energy.astype(np.float32)


@register("m14_spectral_impulse", version="1", owner="codex-new-avenues")
def build_contrast(ctx):
    """Eight spectral impulsiveness product contrasts for Pilot 6."""
    contrast, _ = spectral_impulse_features(ctx)
    return CONTRAST_COLS, contrast


@register("m14_spectral_energy", version="1", owner="codex-new-avenues")
def build_energy(ctx):
    """Four matched plain band-energy controls for Pilot 6."""
    _, energy = spectral_impulse_features(ctx)
    return ENERGY_COLS, energy
