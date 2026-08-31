"""m18_weighted_ctm -- weighted and matched unweighted conformal test martingales."""
from __future__ import annotations

import math

import numpy as np

from sbr.features.base import register
from sbr.features.m07_bayes import BET_C, POW_EPS

TAIL_PAYOFF_THRESHOLD = 0.4
TAIL_WEIGHT_THRESHOLD = 0.45
VC_THRESHOLD = 0.3
OMEGA_MIN = 0.25
HIST_TAIL_RATE_MIN = 0.02
HIST_TAIL_RATE_MAX = 0.30
EPS = 1e-12

WEIGHTED_COLS = [
    "wctm_tail_log",
    "wctm_tail_pk",
    "wctm_disp_log",
    "wctm_disp_pk",
    "wctm_dep_log",
    "wctm_vc_log",
    "wctm_pow_log",
    "wctm_pow_pk",
]

UNWEIGHTED_COLS = [
    "uctm_tail_log",
    "uctm_tail_pk",
    "uctm_disp_log",
    "uctm_disp_pk",
    "uctm_dep_log",
    "uctm_vc_log",
    "uctm_pow_log",
    "uctm_pow_pk",
]

H2_EVALUE_COLS = [
    "ev_tail_mix",
    "ev_tail_ad",
    "ev_disp_mix",
    "ev_disp_ad",
    "ev_pow_mix",
]


def benign_tail_weight(cu_h: np.ndarray, cu_o: np.ndarray) -> np.ndarray:
    """Predictable tail-rate weight; row t excludes the current online row."""
    cu_h = np.asarray(cu_h, dtype=np.float64)
    cu_o = np.asarray(cu_o, dtype=np.float64)
    n = len(cu_o)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    hist_tail = np.abs(cu_h) > TAIL_WEIGHT_THRESHOLD
    r0 = float(np.mean(hist_tail)) if len(hist_tail) else HIST_TAIL_RATE_MIN
    r0 = float(np.clip(r0, HIST_TAIL_RATE_MIN, HIST_TAIL_RATE_MAX))
    online_tail = (np.abs(cu_o) > TAIL_WEIGHT_THRESHOLD).astype(np.float64)
    csum = np.concatenate([[0.0], np.cumsum(online_tail)])
    idx = np.arange(n, dtype=np.int64)

    def prior_rate(w: int) -> np.ndarray:
        lo = np.maximum(0, idx - int(w))
        count = csum[idx] - csum[lo]
        denom = idx - lo
        out = np.full(n, r0, dtype=np.float64)
        np.divide(count, denom, out=out, where=denom > 0)
        return out

    r32 = prior_rate(32)
    r128 = prior_rate(128)
    recent = 0.7 * r32 + 0.3 * r128
    excess = np.maximum(0.0, recent / r0 - 1.0)
    return np.clip(1.0 / (1.0 + excess), OMEGA_MIN, 1.0)


def _bet_lambdas(m0: float, hi: float) -> np.ndarray:
    lam = np.empty(BET_C.shape[0], dtype=np.float64)
    for i, c in enumerate(BET_C):
        lam[i] = c / max(m0, 1e-6) if c > 0 else c / max(hi - m0, 1e-6)
    return lam


def _mix_bet_log(h: np.ndarray, h_h: np.ndarray, hi: float, omega: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    h_h = np.asarray(h_h, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    if len(h) == 0:
        return np.empty(0, dtype=np.float64)
    m0 = float(np.clip(np.mean(h_h) if len(h_h) else 0.5 * hi, 1e-3, hi - 1e-3))
    lam = _bet_lambdas(m0, hi)
    inc = np.log(np.maximum(1.0 + (lam[:, None] * omega[None, :]) * (h[None, :] - m0), EPS))
    logw = -math.log(float(len(lam)))
    s = np.cumsum(inc, axis=1) + logw
    m = np.max(s, axis=0)
    return m + np.log(np.exp(s - m[None, :]).sum(axis=0))


def _power_mix_log(u: np.ndarray, omega: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    if len(u) == 0:
        return np.empty(0, dtype=np.float64)
    p = np.clip(2.0 * np.minimum(u, 1.0 - u), 1e-6, 1.0)
    rows = []
    for eps in POW_EPS:
        e1 = float(eps) * np.power(p, float(eps) - 1.0)
        e2 = float(eps) * np.power(np.clip(1.0 - p, 1e-6, 1.0), float(eps) - 1.0)
        rows.append(np.log(np.maximum(1.0 + omega * (e1 - 1.0), EPS)))
        rows.append(np.log(np.maximum(1.0 + omega * (e2 - 1.0), EPS)))
    r = np.asarray(rows, dtype=np.float64)
    logw = -math.log(float(r.shape[0]))
    s = np.cumsum(r, axis=1) + logw
    m = np.max(s, axis=0)
    return m + np.log(np.exp(s - m[None, :]).sum(axis=0))


def _ctm_features(ctx, weighted: bool) -> np.ndarray:
    u_o = np.asarray(ctx.tr["u"], dtype=np.float64)
    u_h = np.asarray(ctx.hist_tr["u"], dtype=np.float64)
    cu_o = u_o - 0.5
    cu_h = u_h - 0.5
    n = len(u_o)
    out = np.full((n, len(WEIGHTED_COLS)), np.nan, dtype=np.float64)
    if n == 0:
        return out.astype(np.float32)

    up_o = np.concatenate([[cu_h[-1] if len(cu_h) else 0.0], cu_o[:-1]])
    up_h = np.concatenate([[0.0], cu_h[:-1]]) if len(cu_h) else np.empty(0, dtype=np.float64)
    omega = benign_tail_weight(cu_h, cu_o) if weighted else np.ones(n, dtype=np.float64)
    pay_o = {
        "tail": (np.abs(cu_o) > TAIL_PAYOFF_THRESHOLD).astype(np.float64),
        "disp": cu_o * cu_o,
        "dep": (cu_o * up_o > 0.0).astype(np.float64),
        "vc": ((np.abs(cu_o) > VC_THRESHOLD) & (np.abs(up_o) > VC_THRESHOLD)).astype(np.float64),
    }
    pay_h = {
        "tail": (np.abs(cu_h) > TAIL_PAYOFF_THRESHOLD).astype(np.float64),
        "disp": cu_h * cu_h,
        "dep": (cu_h * up_h > 0.0).astype(np.float64),
        "vc": ((np.abs(cu_h) > VC_THRESHOLD) & (np.abs(up_h) > VC_THRESHOLD)).astype(np.float64),
    }
    caps = {"tail": 1.0, "disp": 0.25, "dep": 1.0, "vc": 1.0}
    tail = _mix_bet_log(pay_o["tail"], pay_h["tail"], caps["tail"], omega)
    disp = _mix_bet_log(pay_o["disp"], pay_h["disp"], caps["disp"], omega)
    dep = _mix_bet_log(pay_o["dep"], pay_h["dep"], caps["dep"], omega)
    vc = _mix_bet_log(pay_o["vc"], pay_h["vc"], caps["vc"], omega)
    powmix = _power_mix_log(u_o, omega)

    out[:, 0] = tail
    out[:, 1] = np.maximum.accumulate(tail)
    out[:, 2] = disp
    out[:, 3] = np.maximum.accumulate(disp)
    out[:, 4] = dep
    out[:, 5] = vc
    out[:, 6] = powmix
    out[:, 7] = np.maximum.accumulate(powmix)
    return out.astype(np.float32)


def weighted_ctm_features(ctx) -> np.ndarray:
    """Weighted conformal test martingale channels for Pilot 8."""
    return _ctm_features(ctx, weighted=True)


def unweighted_ctm_features(ctx) -> np.ndarray:
    """Matched unweighted conformal test martingale channels for Pilot 8."""
    return _ctm_features(ctx, weighted=False)


def h2_log_mean_evalue(m07_values: np.ndarray) -> np.ndarray:
    """Parameter-free log-average of preregistered m07 e-process log-capitals."""
    x = np.asarray(m07_values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(H2_EVALUE_COLS):
        raise ValueError(f"expected (n,{len(H2_EVALUE_COLS)}) m07 e-process matrix")
    x = np.clip(x, -40.0, 40.0)
    m = np.max(x, axis=1)
    return (m + np.log(np.exp(x - m[:, None]).mean(axis=1))).astype(np.float32)


@register("m18_wctm", version="1", owner="codex-new-avenues")
def build_weighted(ctx):
    """Eight weighted conformal test martingale features for Pilot 8."""
    return WEIGHTED_COLS, weighted_ctm_features(ctx)


@register("m18_uctm", version="1", owner="codex-new-avenues")
def build_unweighted(ctx):
    """Eight matched unweighted conformal test martingale control features."""
    return UNWEIGHTED_COLS, unweighted_ctm_features(ctx)
