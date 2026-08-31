"""m16_joint_rarity -- joint size-duration rarity on residual excursions."""
from __future__ import annotations

import numpy as np

from sbr.features.base import register
from sbr.features.m13_scale_survival import residual_square_streams, trailing_mean

WINDOWS = (32, 64, 128)
Q = 0.90
MIN_ENDPOINTS = 16
EPS = 1e-12

JOINT_COLS = [
    "joint_live_w32",
    "joint_max_w32",
    "joint_live_w64",
    "joint_max_w64",
    "joint_live_w128",
    "joint_max_w128",
]
DWELL_COLS = [
    "dwell_live_w32",
    "dwell_max_w32",
    "dwell_live_w64",
    "dwell_max_w64",
    "dwell_live_w128",
    "dwell_max_w128",
]


def endpoint_states(stat: np.ndarray, median: float, band: float) -> tuple[np.ndarray, np.ndarray]:
    """Live hot-run duration and peak excess at each endpoint."""
    stat = np.asarray(stat, dtype=np.float64)
    dev = np.abs(stat - median)
    excess = np.maximum(dev - band, 0.0)
    hot = dev > band
    run = np.zeros(len(stat), dtype=np.int64)
    peak = np.zeros(len(stat), dtype=np.float64)
    cur_run = 0
    cur_peak = 0.0
    for i, is_hot in enumerate(hot):
        if bool(is_hot):
            cur_run += 1
            cur_peak = max(cur_peak, float(excess[i]))
        else:
            cur_run = 0
            cur_peak = 0.0
        run[i] = cur_run
        peak[i] = cur_peak
    return run, peak


def _dwell_surprise(hist_run: np.ndarray, query_run: np.ndarray) -> np.ndarray:
    hist_run = np.asarray(hist_run, dtype=np.int64)
    query_run = np.asarray(query_run, dtype=np.int64)
    out = np.zeros(len(query_run), dtype=np.float64)
    n = len(hist_run)
    if n == 0:
        out[:] = np.nan
        return out
    floor = 1.0 / (2.0 * n)
    sorted_run = np.sort(hist_run)
    active = query_run > 0
    if not np.any(active):
        return out
    q = query_run[active]
    counts = n - np.searchsorted(sorted_run, q, side="left")
    p = np.maximum(counts.astype(np.float64) / float(n), floor)
    out[active] = -np.log10(p)
    return out


def _joint_surprise(
    hist_run: np.ndarray, hist_peak: np.ndarray, query_run: np.ndarray, query_peak: np.ndarray
) -> np.ndarray:
    hist_run = np.asarray(hist_run, dtype=np.int64)
    hist_peak = np.asarray(hist_peak, dtype=np.float64)
    query_run = np.asarray(query_run, dtype=np.int64)
    query_peak = np.asarray(query_peak, dtype=np.float64)
    out = np.zeros(len(query_run), dtype=np.float64)
    n = len(hist_run)
    if n == 0:
        out[:] = np.nan
        return out
    floor = 1.0 / (2.0 * n)
    active = query_run > 0
    if not np.any(active):
        return out
    active_idx = np.flatnonzero(active)
    for d in np.unique(query_run[active_idx]):
        rows = active_idx[query_run[active_idx] == d]
        eligible = hist_peak[hist_run >= int(d)]
        if len(eligible) == 0:
            p = np.full(len(rows), floor, dtype=np.float64)
        else:
            sorted_peak = np.sort(eligible)
            counts = len(sorted_peak) - np.searchsorted(sorted_peak, query_peak[rows], side="left")
            p = np.maximum(counts.astype(np.float64) / float(n), floor)
        out[rows] = -np.log10(p)
    return out


def _window_features(
    hist_sq: np.ndarray, online_sq: np.ndarray, w: int
) -> tuple[np.ndarray, np.ndarray]:
    n = len(online_sq)
    candidate = np.full((n, 2), np.nan, dtype=np.float64)
    control = np.full((n, 2), np.nan, dtype=np.float64)
    hist_stat = trailing_mean(hist_sq, w)
    hist_stat = hist_stat[np.isfinite(hist_stat)]
    if len(hist_stat) < MIN_ENDPOINTS or n < w:
        return candidate, control

    median = float(np.median(hist_stat))
    band = float(np.quantile(np.abs(hist_stat - median), Q))
    hist_run, hist_peak = endpoint_states(hist_stat, median, band)

    online_stat = trailing_mean(online_sq, w)
    valid = np.isfinite(online_stat)
    if not np.any(valid):
        return candidate, control
    online_run, online_peak = endpoint_states(online_stat[valid], median, band)

    joint_live = _joint_surprise(hist_run, hist_peak, online_run, online_peak)
    dwell_live = _dwell_surprise(hist_run, online_run)
    joint_max = np.maximum.accumulate(np.nan_to_num(joint_live, nan=0.0, posinf=0.0, neginf=0.0))
    dwell_max = np.maximum.accumulate(np.nan_to_num(dwell_live, nan=0.0, posinf=0.0, neginf=0.0))

    candidate[valid, 0] = joint_live
    candidate[valid, 1] = joint_max
    control[valid, 0] = dwell_live
    control[valid, 1] = dwell_max
    return candidate, control


def joint_rarity_features(ctx) -> tuple[np.ndarray, np.ndarray]:
    """Return Pilot 10 joint-rarity candidate and dwell-only control features."""
    hist_sq, online_sq = residual_square_streams(ctx)
    n = len(online_sq)
    candidate = np.full((n, len(JOINT_COLS)), np.nan, dtype=np.float64)
    control = np.full((n, len(DWELL_COLS)), np.nan, dtype=np.float64)
    col = 0
    for w in WINDOWS:
        joint, dwell = _window_features(hist_sq, online_sq, int(w))
        candidate[:, col : col + 2] = joint
        control[:, col : col + 2] = dwell
        col += 2
    return candidate.astype(np.float32), control.astype(np.float32)


@register("m16_joint_rarity", version="1", owner="codex-new-avenues")
def build_joint(ctx):
    """Joint size-duration rarity features for Pilot 10."""
    candidate, _ = joint_rarity_features(ctx)
    return JOINT_COLS, candidate


@register("m16_dwell_rarity", version="1", owner="codex-new-avenues")
def build_dwell(ctx):
    """Matched dwell-only rarity control features for Pilot 10."""
    _, control = joint_rarity_features(ctx)
    return DWELL_COLS, control
