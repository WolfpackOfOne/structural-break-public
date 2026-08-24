"""m15_ordinal_irrev -- ordinal transition and irreversibility features."""
from __future__ import annotations

import numpy as np

from sbr.features.base import register

PCODES = np.array([0, 1, 3, 4, 6, 7], dtype=np.int64)
CODE_TO_IDX = np.full(8, -1, dtype=np.int64)
for _i, _c in enumerate(PCODES):
    CODE_TO_IDX[_c] = _i

N_PATTERNS = 6
N_TRANS = N_PATTERNS * N_PATTERNS
ALPHA = 0.5
GRID = np.array([16, 64, 256], dtype=np.int64)
NULL_N = 700
MIN_COUNT = 16
CLIP_Z = 10.0
EPS = 1e-12

CANDIDATE_COLS = [
    "trans_kl_exp",
    "trans_kl_exp_z",
    "trans_kl_half",
    "trans_kl_half_z",
    "rr_asym_exp",
    "rr_asym_exp_z",
    "rr_asym_half",
    "rr_asym_half_z",
]
CONTROL_COLS = ["pe_exp", "pe_exp_z", "pe_half", "pe_half_z"]


def _cum(x: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64))])


def ordinal_codes(z: np.ndarray) -> np.ndarray:
    """Order-3 ordinal pattern indices with the m03_dyn tie convention."""
    z = np.asarray(z, dtype=np.float64)
    if len(z) < 3:
        return np.empty(0, dtype=np.int64)
    a = z[:-2]
    b = z[1:-1]
    c = z[2:]
    code = 4 * (a > b).astype(np.int64) + 2 * (a > c).astype(np.int64) + (b > c).astype(np.int64)
    idx = CODE_TO_IDX[code]
    if np.any(idx < 0):
        raise ValueError("unreachable ordinal code produced")
    return idx.astype(np.int64)


def _transition_indices(codes: np.ndarray) -> np.ndarray:
    if len(codes) < 2:
        return np.empty(0, dtype=np.int64)
    return (codes[:-1] * N_PATTERNS + codes[1:]).astype(np.int64)


def _counts_cum(idx: np.ndarray, n_cat: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    out = np.zeros((len(idx) + 1, n_cat), dtype=np.float64)
    if len(idx):
        out[1:] = np.cumsum(np.eye(n_cat, dtype=np.float64)[idx], axis=0)
    return out


def _window_counts(idx: np.ndarray, n_cat: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    n = len(idx)
    c = _counts_cum(idx, n_cat)
    t = np.arange(n, dtype=np.int64)
    if mode == "exp":
        L = t + 1
        lo = np.zeros(n, dtype=np.int64)
    elif mode == "half":
        L = np.maximum((t + 1) // 2, 1)
        lo = t - L + 1
    else:
        raise ValueError(mode)
    hi = t + 1
    return c[hi] - c[lo], L


def _block_counts(idx: np.ndarray, n_cat: int, L: int) -> np.ndarray:
    if len(idx) < L:
        return np.empty((0, n_cat), dtype=np.float64)
    c = _counts_cum(idx, n_cat)
    return c[L:] - c[:-L]


def _smooth_probs(counts: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    return (counts + alpha) / np.maximum(counts.sum(axis=-1, keepdims=True) + alpha * counts.shape[-1], EPS)


def _transition_ref(hist_trans: np.ndarray) -> np.ndarray:
    counts = np.bincount(hist_trans, minlength=N_TRANS).astype(np.float64)
    return _smooth_probs(counts)


def _kl_from_counts(counts: np.ndarray, ref: np.ndarray) -> np.ndarray:
    p = _smooth_probs(counts)
    return np.sum(p * (np.log(np.maximum(p, EPS)) - np.log(np.maximum(ref, EPS))), axis=-1)


def _pe_from_counts(counts: np.ndarray) -> np.ndarray:
    p = _smooth_probs(counts)
    return -np.sum(p * np.log(np.maximum(p, EPS)), axis=-1) / np.log(float(N_PATTERNS))


def _rr_from_sums(L, s2, s3) -> np.ndarray:
    Lf = np.asarray(L, dtype=np.float64)
    m2 = s2 / np.maximum(Lf, 1.0)
    m3 = s3 / np.maximum(Lf, 1.0)
    return m3 / np.maximum(m2, EPS) ** 1.5


def _rolling_rr(inc: np.ndarray, L: int) -> np.ndarray:
    inc = np.asarray(inc, dtype=np.float64)
    if len(inc) < L:
        return np.empty(0, dtype=np.float64)
    c2 = _cum(inc * inc)
    c3 = _cum(inc * inc * inc)
    return _rr_from_sums(L, c2[L:] - c2[:-L], c3[L:] - c3[:-L])


def _online_rr(inc: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    inc = np.asarray(inc, dtype=np.float64)
    n = len(inc)
    c2 = _cum(inc * inc)
    c3 = _cum(inc * inc * inc)
    t = np.arange(n, dtype=np.int64)
    if mode == "exp":
        L = t + 1
        lo = np.zeros(n, dtype=np.int64)
    elif mode == "half":
        L = np.maximum((t + 1) // 2, 1)
        lo = t - L + 1
    else:
        raise ValueError(mode)
    hi = t + 1
    return _rr_from_sums(L, c2[hi] - c2[lo], c3[hi] - c3[lo]), L


def _sample(v: np.ndarray) -> np.ndarray:
    if len(v) > NULL_N:
        v = v[:: max(1, len(v) // NULL_N)]
    return v


def _summ(v: np.ndarray) -> tuple[float, float]:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    v = _sample(v)
    if len(v) < 8:
        return 0.0, 1.0
    q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
    sd = max(float((q[3] - q[1]) / 1.349), float((q[4] - q[0]) / 3.29), 1e-9)
    return float(q[2]), sd


def _null_count_params(idx: np.ndarray, n_cat: int, stat_fn) -> tuple[np.ndarray, np.ndarray]:
    med = np.empty(len(GRID), dtype=np.float64)
    lsd = np.empty(len(GRID), dtype=np.float64)
    for i, L in enumerate(GRID):
        counts = _block_counts(idx, n_cat, int(L))
        vals = stat_fn(counts) if len(counts) else np.empty(0, dtype=np.float64)
        m, s = _summ(vals)
        med[i] = m
        lsd[i] = np.log(s)
    return med, lsd


def _null_rr_params(inc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.empty(len(GRID), dtype=np.float64)
    lsd = np.empty(len(GRID), dtype=np.float64)
    for i, L in enumerate(GRID):
        vals = _rolling_rr(inc, int(L))
        m, s = _summ(vals)
        med[i] = m
        lsd[i] = np.log(s)
    return med, lsd


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


def _z(raw: np.ndarray, L: np.ndarray, params: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    out = np.full(len(raw), np.nan, dtype=np.float64)
    ok = np.isfinite(raw) & (L >= MIN_COUNT)
    if not np.any(ok):
        return out
    med, lsd = params
    x = np.log(np.maximum(L[ok].astype(np.float64), 1.0))
    mu = _loglin(med, x)
    sd = np.exp(_loglin(lsd, x))
    out[ok] = np.clip((raw[ok] - mu) / np.maximum(sd, EPS), -CLIP_Z, CLIP_Z)
    return out


def _masked(raw: np.ndarray, L: np.ndarray) -> np.ndarray:
    return np.where(L >= MIN_COUNT, raw, np.nan)


def _online_increments(hist_z: np.ndarray, online_z: np.ndarray) -> np.ndarray:
    inc = np.empty(len(online_z), dtype=np.float64)
    if len(online_z) == 0:
        return inc
    inc[0] = online_z[0] - hist_z[-1]
    if len(online_z) > 1:
        inc[1:] = online_z[1:] - online_z[:-1]
    return inc


def ordinal_irreversibility_features(ctx) -> tuple[np.ndarray, np.ndarray]:
    hist_z = np.asarray(ctx.hist_tr["mean"], dtype=np.float64)
    online_z = np.asarray(ctx.tr["mean"], dtype=np.float64)
    n = len(online_z)

    hist_codes = ordinal_codes(hist_z)
    online_codes = ordinal_codes(np.concatenate([hist_z[-2:], online_z]))
    if len(online_codes) != n:
        raise ValueError("online ordinal code length mismatch")

    hist_trans = _transition_indices(hist_codes)
    if len(hist_trans) == 0:
        hist_trans = np.array([0], dtype=np.int64)
    ref_trans = _transition_ref(hist_trans)

    prev = np.empty(n, dtype=np.int64)
    if n:
        prev[0] = hist_codes[-1] if len(hist_codes) else online_codes[0]
        prev[1:] = online_codes[:-1]
    online_trans = (prev * N_PATTERNS + online_codes).astype(np.int64)

    trans_params = _null_count_params(hist_trans, N_TRANS, lambda c: _kl_from_counts(c, ref_trans))
    pe_params = _null_count_params(hist_codes, N_PATTERNS, _pe_from_counts)
    rr_params = _null_rr_params(np.diff(hist_z))
    online_inc = _online_increments(hist_z, online_z)

    candidate = np.full((n, len(CANDIDATE_COLS)), np.nan, dtype=np.float64)
    control = np.full((n, len(CONTROL_COLS)), np.nan, dtype=np.float64)

    for mode, raw_col, z_col in (("exp", 0, 1), ("half", 2, 3)):
        counts, L = _window_counts(online_trans, N_TRANS, mode)
        raw = _kl_from_counts(counts, ref_trans)
        candidate[:, raw_col] = _masked(raw, L)
        candidate[:, z_col] = _z(raw, L, trans_params)

    for mode, raw_col, z_col in (("exp", 4, 5), ("half", 6, 7)):
        raw, L = _online_rr(online_inc, mode)
        candidate[:, raw_col] = _masked(raw, L)
        candidate[:, z_col] = _z(raw, L, rr_params)

    for mode, raw_col, z_col in (("exp", 0, 1), ("half", 2, 3)):
        counts, L = _window_counts(online_codes, N_PATTERNS, mode)
        raw = _pe_from_counts(counts)
        control[:, raw_col] = _masked(raw, L)
        control[:, z_col] = _z(raw, L, pe_params)

    return candidate.astype(np.float32), control.astype(np.float32)


@register("m15_ordinal_irrev", version="1", owner="codex-new-avenues")
def build_candidate(ctx):
    """Ordinal transition KL and Ramsey-Rothman irreversibility for Pilot 7."""
    candidate, _ = ordinal_irreversibility_features(ctx)
    return CANDIDATE_COLS, candidate


@register("m15_ordinal_entropy", version="1", owner="codex-new-avenues")
def build_entropy_control(ctx):
    """Permutation-entropy-only control for Pilot 7."""
    _, control = ordinal_irreversibility_features(ctx)
    return CONTROL_COLS, control
