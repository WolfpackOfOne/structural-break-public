"""m17_observers -- frozen Kalman/NIS and Hankel-DMD observer residuals."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sbr.features.base import register
from sbr.transforms import HistParams

KALMAN_Q_GRID = (0.01, 0.05, 0.20)
KALMAN_R_GRID = (0.05, 0.20, 1.00)
KALMAN_WINDOWS = (32, 64)
DMD_DELAY = 16
DMD_RANK = 4
DMD_WINDOWS = (32, 64)
EPS = 1e-12

KALMAN_COLS = [
    "nis_log",
    "nis_cum_z",
    "nis_w32_z",
    "nis_w64_z",
    "innov_acf_w32_z",
    "innov_acf_w64_z",
]

DMD_COLS = [
    "dmd_resid_h1_z",
    "dmd_resid_h5_z",
    "dmd_subspace_resid_z",
    "dmd_effrank_w32_z",
    "dmd_effrank_w64_z",
    "dmd_resid_ratio_h5_h1",
]


@dataclass
class KalmanObserverState:
    phi: np.ndarray
    q: float
    r: float
    x0: np.ndarray
    p0: np.ndarray
    history_nll: float


@dataclass
class DmdObserverState:
    operator: np.ndarray
    operator_h5: np.ndarray
    subspace: np.ndarray
    fitted_rank: int
    nulls: dict[str, tuple[float, float]]


def _hp(hist: np.ndarray, hp: HistParams | None = None) -> HistParams:
    return hp if hp is not None else HistParams(np.asarray(hist, dtype=np.float64), ar_order=2)


def _standardized(x: np.ndarray, hp: HistParams) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - hp.mu) / hp.sd


def _ar2_phi(hp: HistParams) -> np.ndarray:
    phi = np.zeros(2, dtype=np.float64)
    coef = np.asarray(hp.ar_coef, dtype=np.float64)
    phi[: min(2, len(coef))] = coef[:2]
    return phi


def _kalman_pass(z: np.ndarray, phi: np.ndarray, q: float, r: float) -> tuple[np.ndarray, np.ndarray, float]:
    f = np.array([[phi[0], phi[1]], [1.0, 0.0]], dtype=np.float64)
    h = np.array([1.0, 0.0], dtype=np.float64)
    qmat = np.array([[q, 0.0], [0.0, 0.0]], dtype=np.float64)
    eye = np.eye(2, dtype=np.float64)

    if len(z) >= 2:
        x = np.array([z[1], z[0]], dtype=np.float64)
        start = 2
    elif len(z) == 1:
        x = np.array([z[0], 0.0], dtype=np.float64)
        start = 1
    else:
        x = np.zeros(2, dtype=np.float64)
        start = 0
    p = np.eye(2, dtype=np.float64)
    nll = 0.0
    used = 0

    for i in range(start, len(z)):
        x_pred = f @ x
        p_pred = f @ p @ f.T + qmat
        s = max(float(p_pred[0, 0] + r), EPS)
        v = float(z[i] - x_pred[0])
        if i >= 32:
            nll += 0.5 * (np.log(s) + (v * v) / s)
            used += 1
        k = p_pred[:, 0] / s
        x = x_pred + k * v
        p = (eye - np.outer(k, h)) @ p_pred
        p = 0.5 * (p + p.T)
    if used == 0:
        nll = 0.0
    return x, p, float(nll)


def fit_kalman_observer(hist: np.ndarray, hp: HistParams | None = None) -> KalmanObserverState:
    """Fit the fixed AR(2)-state Kalman observer from history only."""
    params = _hp(hist, hp)
    z = _standardized(hist, params)
    phi = _ar2_phi(params)

    best: tuple[float, float, np.ndarray, np.ndarray, float] | None = None
    for q in KALMAN_Q_GRID:
        for r in KALMAN_R_GRID:
            x, p, nll = _kalman_pass(z, phi, float(q), float(r))
            key = (nll, float(q), float(r))
            if best is None or key < (best[4], best[0], best[1]):
                best = (float(q), float(r), x, p, nll)
    assert best is not None
    return KalmanObserverState(phi=phi, q=best[0], r=best[1], x0=best[2], p0=best[3], history_nll=best[4])


def _acf1_scaled(x: np.ndarray, w: int) -> float:
    if len(x) < w:
        return np.nan
    v = np.asarray(x[-w:], dtype=np.float64)
    a = v[:-1] - float(np.mean(v[:-1]))
    b = v[1:] - float(np.mean(v[1:]))
    den = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if den <= EPS:
        return 0.0
    return float(np.sqrt(float(w)) * np.dot(a, b) / den)


def kalman_nis_features(ctx) -> np.ndarray:
    """Frozen AR(2)-state Kalman NIS and innovation-whiteness features."""
    state = fit_kalman_observer(ctx.hist, ctx.hp)
    z = _standardized(ctx.online, ctx.hp)
    n = len(z)
    out = np.full((n, len(KALMAN_COLS)), np.nan, dtype=np.float64)
    if n == 0:
        return out.astype(np.float32)

    f = np.array([[state.phi[0], state.phi[1]], [1.0, 0.0]], dtype=np.float64)
    h = np.array([1.0, 0.0], dtype=np.float64)
    qmat = np.array([[state.q, 0.0], [0.0, 0.0]], dtype=np.float64)
    eye = np.eye(2, dtype=np.float64)
    x = np.array(state.x0, copy=True, dtype=np.float64)
    p = np.array(state.p0, copy=True, dtype=np.float64)
    nis = np.empty(n, dtype=np.float64)
    innov = np.empty(n, dtype=np.float64)
    csum_nis = 0.0

    for t, z_t in enumerate(z):
        x_pred = f @ x
        p_pred = f @ p @ f.T + qmat
        s = max(float(p_pred[0, 0] + state.r), EPS)
        v = float(z_t - x_pred[0])
        e = v / np.sqrt(s)
        nis_t = e * e
        nis[t] = nis_t
        innov[t] = e
        csum_nis += nis_t

        out[t, 0] = np.log1p(nis_t)
        out[t, 1] = (csum_nis - float(t + 1)) / np.sqrt(2.0 * float(t + 1))
        if t + 1 >= 32:
            out[t, 2] = (float(np.sum(nis[t - 31 : t + 1])) - 32.0) / np.sqrt(64.0)
            out[t, 4] = _acf1_scaled(innov[: t + 1], 32)
        if t + 1 >= 64:
            out[t, 3] = (float(np.sum(nis[t - 63 : t + 1])) - 64.0) / np.sqrt(128.0)
            out[t, 5] = _acf1_scaled(innov[: t + 1], 64)

        k = p_pred[:, 0] / s
        x = x_pred + k * v
        p = (eye - np.outer(k, h)) @ p_pred
        p = 0.5 * (p + p.T)

    return out.astype(np.float32)


def _delay_matrix(z: np.ndarray, d: int = DMD_DELAY) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if len(z) < d:
        return np.empty((0, d), dtype=np.float64)
    return np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(z, d))


def _robust_null(x: np.ndarray) -> tuple[float, float]:
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 0.0, 1.0
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    scale = mad if mad > EPS else sd
    if scale <= EPS:
        scale = 1.0
    return med, scale


def _zscore(x: np.ndarray, null: tuple[float, float]) -> np.ndarray:
    med, scale = null
    return (np.asarray(x, dtype=np.float64) - med) / scale


def _subspace_fraction(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        v = v[None, :]
    if u.shape[1] == 0:
        return np.ones(v.shape[0], dtype=np.float64)
    proj = (v @ u) @ u.T
    den = np.linalg.norm(v, axis=1)
    den = np.maximum(den, EPS)
    return np.linalg.norm(v - proj, axis=1) / den


def _effective_rank_series(v: np.ndarray, w: int) -> np.ndarray:
    n = len(v)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < w:
        return out
    outer = np.einsum("ni,nj->nij", v, v, optimize=True)
    csum = np.concatenate([np.zeros((1, v.shape[1], v.shape[1]), dtype=np.float64), np.cumsum(outer, axis=0)], axis=0)
    grams = csum[w:] - csum[:-w]
    eig = np.linalg.eigvalsh(grams)
    sig = np.sqrt(np.clip(eig, 0.0, None))
    ss = np.sum(sig, axis=1)
    ok = ss > EPS
    ranks = np.ones(len(grams), dtype=np.float64)
    if np.any(ok):
        p = sig[ok] / ss[ok, None]
        ent = -np.sum(np.where(p > 0.0, p * np.log(p), 0.0), axis=1)
        ranks[ok] = np.exp(ent)
    out[w - 1 :] = ranks
    return out


def fit_hankel_dmd_observer(hist: np.ndarray, hp: HistParams | None = None) -> DmdObserverState:
    """Fit the frozen rank-4 Hankel-DMD observer from history only."""
    params = _hp(hist, hp)
    z = _standardized(hist, params)
    v = _delay_matrix(z, DMD_DELAY)
    d = DMD_DELAY
    a = np.zeros((d, d), dtype=np.float64)
    u_r = np.empty((d, 0), dtype=np.float64)
    fitted_rank = 0

    if len(v) >= max(DMD_RANK + 2, 8):
        x = v[:-1].T
        y = v[1:].T
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        fitted_rank = int(min(DMD_RANK, np.sum(s > EPS)))
        if fitted_rank > 0:
            u_r = u[:, :fitted_rank]
            v_r = vt[:fitted_rank].T
            a = (y @ (v_r / s[:fitted_rank])) @ u_r.T

    a5 = np.linalg.matrix_power(a, 5)
    h1 = np.full(len(v), np.nan, dtype=np.float64)
    h5 = np.full(len(v), np.nan, dtype=np.float64)
    if len(v) > 1:
        h1[1:] = np.linalg.norm(v[:-1] @ a.T - v[1:], axis=1) / np.sqrt(float(d))
    if len(v) > 5:
        h5[5:] = np.linalg.norm(v[:-5] @ a5.T - v[5:], axis=1) / np.sqrt(float(d))
    sub = _subspace_fraction(v, u_r) if len(v) else np.full(0, np.nan, dtype=np.float64)
    er32 = _effective_rank_series(v, 32)
    er64 = _effective_rank_series(v, 64)
    nulls = {
        "h1": _robust_null(h1),
        "h5": _robust_null(h5),
        "sub": _robust_null(sub),
        "er32": _robust_null(er32),
        "er64": _robust_null(er64),
    }
    return DmdObserverState(operator=a, operator_h5=a5, subspace=u_r, fitted_rank=fitted_rank, nulls=nulls)


def hankel_dmd_features(ctx) -> np.ndarray:
    """Frozen Hankel-DMD residual, subspace, and effective-rank features."""
    state = fit_hankel_dmd_observer(ctx.hist, ctx.hp)
    z_hist = _standardized(ctx.hist, ctx.hp)
    z_online = _standardized(ctx.online, ctx.hp)
    n = len(z_online)
    out = np.full((n, len(DMD_COLS)), np.nan, dtype=np.float64)
    if n == 0:
        return out.astype(np.float32)

    z_all = np.concatenate([z_hist, z_online])
    v_all = _delay_matrix(z_all, DMD_DELAY)
    if len(v_all) == 0:
        return out.astype(np.float32)

    endpoints = len(z_hist) + np.arange(n, dtype=np.int64)
    idx = endpoints - (DMD_DELAY - 1)
    valid = (idx >= 0) & (idx < len(v_all))
    if not np.any(valid):
        return out.astype(np.float32)
    rows = np.flatnonzero(valid)
    j = idx[valid]
    cur = v_all[j]
    d = float(DMD_DELAY)

    h1 = np.full(n, np.nan, dtype=np.float64)
    ok1 = j >= 1
    if np.any(ok1):
        h1[rows[ok1]] = np.linalg.norm(v_all[j[ok1] - 1] @ state.operator.T - cur[ok1], axis=1) / np.sqrt(d)

    h5 = np.full(n, np.nan, dtype=np.float64)
    ok5 = j >= 5
    if np.any(ok5):
        h5[rows[ok5]] = np.linalg.norm(v_all[j[ok5] - 5] @ state.operator_h5.T - cur[ok5], axis=1) / np.sqrt(d)

    sub = np.full(n, np.nan, dtype=np.float64)
    sub[rows] = _subspace_fraction(cur, state.subspace)
    er32_all = _effective_rank_series(v_all, 32)
    er64_all = _effective_rank_series(v_all, 64)
    er32 = np.full(n, np.nan, dtype=np.float64)
    er64 = np.full(n, np.nan, dtype=np.float64)
    er32[rows] = er32_all[j]
    er64[rows] = er64_all[j]

    out[:, 0] = _zscore(h1, state.nulls["h1"])
    out[:, 1] = _zscore(h5, state.nulls["h5"])
    out[:, 2] = _zscore(sub, state.nulls["sub"])
    out[:, 3] = _zscore(er32, state.nulls["er32"])
    out[:, 4] = _zscore(er64, state.nulls["er64"])
    ratio_ok = np.isfinite(h5) & np.isfinite(h1)
    out[ratio_ok, 5] = h5[ratio_ok] / np.maximum(h1[ratio_ok], EPS)
    return out.astype(np.float32)


def observer_features(ctx) -> tuple[np.ndarray, np.ndarray]:
    """Return Kalman/NIS and Hankel-DMD observer feature blocks."""
    return kalman_nis_features(ctx), hankel_dmd_features(ctx)


@register("m17_kalman_nis", version="1", owner="codex-new-avenues")
def build_kalman(ctx):
    """Six frozen AR(2)-state Kalman NIS and innovation-whiteness features."""
    return KALMAN_COLS, kalman_nis_features(ctx)


@register("m17_hankel_dmd", version="1", owner="codex-new-avenues")
def build_dmd(ctx):
    """Six frozen Hankel-DMD residual/effective-rank observer features."""
    return DMD_COLS, hankel_dmd_features(ctx)
