"""m13_scale_survival -- dyadic residual-scale survival summaries.

Pilot 5 asks whether a persistent residual-scale break stays abnormal under
causal temporal coarse-graining. The control arm exposes the six per-scale
surprises directly; the candidate exposes only cross-scale functionals.
"""
from __future__ import annotations

import numpy as np

from sbr.features.base import register
from sbr.transforms import _ar_resid, ar_filter_causal

SCALES = (1, 2, 4, 8, 16, 32)
LOG2_SCALES = np.log2(np.asarray(SCALES, dtype=np.float64))
Q05 = 1.301029995664
Q01 = 2.0

SUMMARY_COLS = [
    "survival_count_q05",
    "survival_count_q01",
    "largest_surviving_scale_q05",
    "largest_surviving_scale_q01",
    "slope_logscale",
]
INDIV_COLS = [
    "surprise_b1",
    "surprise_b2",
    "surprise_b4",
    "surprise_b8",
    "surprise_b16",
    "surprise_b32",
]


def _historical_ar_residual(z: np.ndarray, coef: np.ndarray) -> np.ndarray:
    p = len(coef)
    if p == 0:
        return z.copy()
    out = np.zeros(len(z), dtype=np.float64)
    if len(z) > p:
        out[p:] = _ar_resid(z, coef)
    elif len(z):
        out[:] = z
    return out


def residual_square_streams(ctx) -> tuple[np.ndarray, np.ndarray]:
    """Return history/online AR(2) residual-square streams.

    Residuals are fitted on history only and standardized by the historical
    residual sigma frozen in ``HistParams``.
    """
    hist = np.asarray(ctx.hist, dtype=np.float64)
    online = np.asarray(ctx.online, dtype=np.float64)
    hp = ctx.hp
    zh = (hist - hp.mu) / hp.sd
    zo = (online - hp.mu) / hp.sd
    eh = _historical_ar_residual(zh, hp.ar_coef) / hp.ar_sigma
    eo = ar_filter_causal(zo, hp.ar_coef, zh) / hp.ar_sigma
    return eh * eh, eo * eo


def trailing_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=np.float64)
    if w <= 0 or len(x) < w:
        return out
    c = np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64))])
    out[w - 1 :] = (c[w:] - c[:-w]) / float(w)
    return out


def scale_surprises(hist_sq: np.ndarray, online_sq: np.ndarray) -> np.ndarray:
    """Per-scale two-sided empirical-null surprises for one online series."""
    n = len(online_sq)
    out = np.full((n, len(SCALES)), np.nan, dtype=np.float64)
    for j, b in enumerate(SCALES):
        hist_stat = trailing_mean(hist_sq, b)
        hist_stat = hist_stat[np.isfinite(hist_stat)]
        if len(hist_stat) == 0:
            continue
        null = np.sort(hist_stat)
        m = float(len(null))
        online_stat = trailing_mean(online_sq, b)
        rows = np.flatnonzero(np.isfinite(online_stat))
        if len(rows) == 0:
            continue
        ranks = np.searchsorted(null, online_stat[rows], side="right").astype(np.float64)
        pct = (ranks + 0.5) / (m + 1.0)
        tail2 = 2.0 * np.minimum(pct, 1.0 - pct)
        tail2 = np.maximum(tail2, 1.0 / (m + 1.0))
        out[rows, j] = -np.log10(tail2)
    return out


def summary_from_surprises(surprises: np.ndarray) -> np.ndarray:
    n = surprises.shape[0]
    out = np.full((n, len(SUMMARY_COLS)), np.nan, dtype=np.float64)
    finite = np.isfinite(surprises)
    codes = np.arange(1, len(SCALES) + 1, dtype=np.float64)

    surv05 = finite & (surprises > Q05)
    surv01 = finite & (surprises > Q01)
    out[:, 0] = surv05.sum(axis=1)
    out[:, 1] = surv01.sum(axis=1)
    out[:, 2] = np.max(np.where(surv05, codes, 0.0), axis=1)
    out[:, 3] = np.max(np.where(surv01, codes, 0.0), axis=1)

    for i in range(n):
        mask = finite[i]
        if int(mask.sum()) < 2:
            continue
        x = LOG2_SCALES[mask]
        y = surprises[i, mask]
        xc = x - float(x.mean())
        den = float(np.dot(xc, xc))
        if den > 0.0:
            out[i, 4] = float(np.dot(xc, y - float(y.mean())) / den)
    return out.astype(np.float32)


def scale_survival_features(ctx) -> tuple[np.ndarray, np.ndarray]:
    hist_sq, online_sq = residual_square_streams(ctx)
    surprises = scale_surprises(hist_sq, online_sq)
    return summary_from_surprises(surprises), surprises.astype(np.float32)


@register("m13_scale_surv", version="1", owner="codex-new-avenues")
def build_summary(ctx):
    """Five cross-scale survival summaries from dyadic residual-square evidence."""
    summary, _ = scale_survival_features(ctx)
    return SUMMARY_COLS, summary


@register("m13_scale_indiv", version="1", owner="codex-new-avenues")
def build_individual(ctx):
    """Six individual dyadic residual-square surprise values for Pilot 5 control."""
    _, indiv = scale_survival_features(ctx)
    return INDIV_COLS, indiv
