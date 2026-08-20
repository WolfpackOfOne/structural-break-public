#!/usr/bin/env python3
"""ORACLE / DIAGNOSTIC information-frontier study for structural breaks.

This script intentionally builds non-deployable, known-boundary/offline
diagnostics.  It never reads 2026 reduced test data and never writes production
features.  The goal is to measure information content, not to create a candidate
submission.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import struct
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - exercised only without research deps
    lgb = None


HORIZONS: tuple[int | str, ...] = (
    1,
    2,
    3,
    5,
    10,
    20,
    30,
    50,
    75,
    100,
    150,
    200,
    300,
    500,
    "FULL",
)
PSEUDO_SEEDS = (0, 1, 7, 42, 2026)
BOOTSTRAP_HORIZONS = {"5", "20", "50", "100", "200", "FULL"}
KEY_CONTROL_HORIZONS = {"5", "20", "50", "100", "200", "500", "FULL"}
FOLDS = (0, 1, 2, 3, 4)
PRE_WINDOWS: tuple[int | str, ...] = (20, 50, 100, 200, 500, "all")
MODEL_EXCLUDE_COLUMNS = {
    "id",
    "fold",
    "target",
    "boundary",
    "rel_boundary",
    "n_hist",
    "n_online",
    "post_len",
    "horizon",
    "pseudo_seed",
    "break_family",
}
BASIC_TOKENS = (
    "hist__mean_z",
    "hist__abs_mean_z",
    "hist__welch_t",
    "hist__median_z",
    "hist__log_sd_ratio",
    "hist__log_var_ratio",
    "hist__log_mad_ratio",
    "hist__bf_t",
    "hist__ks",
    "hist__w1",
    "hist__energy",
    "hist__cvm",
    "hist__q25_z",
    "hist__q50_z",
    "hist__q75_z",
    "hist__acf1_diff",
    "hist__acf2_diff",
    "hist__acf5_diff",
    "hist__abs_acf1_diff",
    "resid_hist__mean_z",
    "resid_hist__log_sd_ratio",
    "resid_hist__ks",
    "pre50__mean_z",
    "pre50__abs_mean_z",
    "pre50__log_sd_ratio",
    "pre50__ks",
    "pre100__mean_z",
    "pre100__log_sd_ratio",
    "preall__mean_z",
    "preall__log_sd_ratio",
)


@dataclass
class PreparedSeries:
    id: int
    fold: int
    target: int
    n_hist: int
    n_online: int
    tau_index: int
    hist: np.ndarray
    online: np.ndarray
    hist_resid: np.ndarray
    online_resid: np.ndarray
    break_family: str = "none"
    hist_sorted: np.ndarray | None = None
    hist_iqr: float = float("nan")


def horizon_label(horizon: int | str) -> str:
    return str(horizon)


def horizon_order(horizon: int | str) -> int:
    return 10_000 if horizon == "FULL" else int(horizon)


def parse_horizon(value: str) -> int | str:
    if value.upper() == "FULL":
        return "FULL"
    out = int(value)
    if out <= 0:
        raise ValueError("horizon must be positive")
    return out


def post_slice(online: np.ndarray, boundary: int, horizon: int | str) -> np.ndarray:
    """First h observations after boundary, or all remaining observations."""
    if boundary < 0 or boundary >= len(online):
        return online[:0]
    if horizon == "FULL":
        return online[boundary:]
    return online[boundary : min(len(online), boundary + int(horizon))]


def residual_post_slice(
    online_resid: np.ndarray, boundary: int, horizon: int | str
) -> np.ndarray:
    return post_slice(online_resid, boundary, horizon)


def pre_slice(source: np.ndarray, boundary: int, window: int | str) -> np.ndarray:
    """Immediate pre-boundary reference."""
    if window == "all":
        return source[: max(boundary, 0)]
    w = int(window)
    return source[max(0, boundary - w) : max(boundary, 0)]


def finite_auc(y: np.ndarray, score: np.ndarray) -> float:
    ok = np.isfinite(score)
    y2 = np.asarray(y)[ok]
    s2 = np.asarray(score)[ok]
    if len(y2) == 0 or len(np.unique(y2)) < 2:
        return float("nan")
    return float(roc_auc_score(y2, s2))


def auc_with_folds(y: np.ndarray, score: np.ndarray, folds: np.ndarray) -> tuple[float, list[float]]:
    overall = finite_auc(y, score)
    per_fold = []
    for f in FOLDS:
        mask = folds == f
        per_fold.append(finite_auc(y[mask], score[mask]))
    return overall, per_fold


def safe_std(x: np.ndarray, ddof: int = 1) -> float:
    if len(x) <= ddof:
        return float("nan")
    out = float(np.nanstd(x, ddof=ddof))
    return out if out > 0 else float("nan")


def mad(x: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    med = float(np.nanmedian(x))
    return float(np.nanmedian(np.abs(x - med)) * 1.4826)


def trimmed_mean(x: np.ndarray, prop: float = 0.1) -> float:
    if len(x) == 0:
        return float("nan")
    if len(x) < 10:
        return float(np.nanmean(x))
    xs = np.sort(x[np.isfinite(x)])
    k = int(len(xs) * prop)
    if len(xs) <= 2 * k:
        return float(np.nanmean(xs))
    return float(np.nanmean(xs[k:-k]))


def skew_kurt(x: np.ndarray) -> tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    sk = float(stats.skew(x, bias=False, nan_policy="omit"))
    ku = float(stats.kurtosis(x, fisher=True, bias=False, nan_policy="omit")) if len(x) >= 4 else np.nan
    return sk, ku


def acf(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 2:
        return float("nan")
    z = np.asarray(x, dtype=np.float64)
    z = z - np.nanmean(z)
    den = float(np.dot(z, z))
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(z[lag:], z[:-lag]) / den)


def slope(x: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    y = np.asarray(x, dtype=np.float64)
    t = np.arange(len(y), dtype=np.float64)
    t -= t.mean()
    den = float(np.dot(t, t))
    if den <= 0:
        return float("nan")
    return float(np.dot(t, y - y.mean()) / den)


def ecdf_distances(a: np.ndarray, b: np.ndarray, scale: float) -> dict[str, float]:
    """Fast 1-D ECDF distances; works for very short post segments."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 1 or len(b) < 2:
        return {"ks": np.nan, "w1": np.nan, "energy": np.nan, "cvm": np.nan}
    aa = np.sort(a)
    bb = np.sort(b)
    v = np.concatenate([aa, bb])
    v.sort()
    if len(v) < 2:
        return {"ks": 0.0, "w1": 0.0, "energy": 0.0, "cvm": 0.0}
    left = v[:-1]
    d = np.diff(v)
    ca = np.searchsorted(aa, left, side="right") / len(aa)
    cb = np.searchsorted(bb, left, side="right") / len(bb)
    delta = ca - cb
    sc = scale if np.isfinite(scale) and scale > 1e-12 else 1.0
    return {
        "ks": float(np.max(np.abs(delta))) if len(delta) else 0.0,
        "w1": float(np.sum(np.abs(delta) * d) / sc),
        "energy": float(2.0 * np.sum(delta * delta * d) / sc),
        "cvm": float(len(aa) * len(bb) / (len(aa) + len(bb)) ** 2 * np.sum(delta * delta)),
    }


def ecdf_distances_against_sorted(
    a: np.ndarray, b_sorted: np.ndarray, scale: float
) -> dict[str, float]:
    """ECDF distances when the reference is already sorted."""
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    bb = np.asarray(b_sorted, dtype=np.float64)
    if len(a) < 1 or len(bb) < 2:
        return {"ks": np.nan, "w1": np.nan, "energy": np.nan, "cvm": np.nan}
    aa = np.sort(a)
    v = np.concatenate([aa, bb])
    v.sort()
    if len(v) < 2:
        return {"ks": 0.0, "w1": 0.0, "energy": 0.0, "cvm": 0.0}
    left = v[:-1]
    d = np.diff(v)
    ca = np.searchsorted(aa, left, side="right") / len(aa)
    cb = np.searchsorted(bb, left, side="right") / len(bb)
    delta = ca - cb
    sc = scale if np.isfinite(scale) and scale > 1e-12 else 1.0
    return {
        "ks": float(np.max(np.abs(delta))) if len(delta) else 0.0,
        "w1": float(np.sum(np.abs(delta) * d) / sc),
        "energy": float(2.0 * np.sum(delta * delta * d) / sc),
        "cvm": float(len(aa) * len(bb) / (len(aa) + len(bb)) ** 2 * np.sum(delta * delta)),
    }


def compare_segment(post: np.ndarray, ref: np.ndarray, prefix: str) -> dict[str, float]:
    """Classical two-sample feature bank for one post/ref comparison."""
    out: dict[str, float] = {}
    post = np.asarray(post, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    post = post[np.isfinite(post)]
    ref = ref[np.isfinite(ref)]
    if len(post) == 0 or len(ref) < 2:
        return out

    mu_p = float(np.mean(post))
    mu_r = float(np.mean(ref))
    sd_p = safe_std(post)
    sd_r = safe_std(ref)
    med_p = float(np.median(post))
    med_r = float(np.median(ref))
    mad_p = mad(post)
    mad_r = mad(ref)
    scale = sd_r if np.isfinite(sd_r) and sd_r > 1e-12 else 1.0
    iqr_r = float(np.subtract(*np.quantile(ref, [0.75, 0.25])))
    if not np.isfinite(iqr_r) or iqr_r <= 1e-12:
        iqr_r = scale

    out[f"{prefix}__mean_z"] = (mu_p - mu_r) / scale
    out[f"{prefix}__abs_mean_z"] = abs(out[f"{prefix}__mean_z"])
    out[f"{prefix}__median_z"] = (med_p - med_r) / scale
    out[f"{prefix}__trim_mean_z"] = (trimmed_mean(post) - trimmed_mean(ref)) / scale
    if np.isfinite(sd_p) and np.isfinite(sd_r):
        denom = math.sqrt(sd_p * sd_p / max(len(post), 1) + sd_r * sd_r / len(ref))
        out[f"{prefix}__welch_t"] = (mu_p - mu_r) / denom if denom > 1e-12 else np.nan
        out[f"{prefix}__std_mean_diff"] = (mu_p - mu_r) / math.sqrt((sd_p * sd_p + sd_r * sd_r) / 2.0)
        out[f"{prefix}__log_sd_ratio"] = math.log(max(sd_p, 1e-12) / max(sd_r, 1e-12))
        out[f"{prefix}__log_var_ratio"] = math.log(max(sd_p * sd_p, 1e-12) / max(sd_r * sd_r, 1e-12))
    else:
        out[f"{prefix}__welch_t"] = np.nan
        out[f"{prefix}__std_mean_diff"] = np.nan
        out[f"{prefix}__log_sd_ratio"] = np.nan
        out[f"{prefix}__log_var_ratio"] = np.nan
    out[f"{prefix}__log_mad_ratio"] = math.log(max(mad_p, 1e-12) / max(mad_r, 1e-12))
    abs_p = np.abs(post - med_r)
    abs_r = np.abs(ref - med_r)
    out[f"{prefix}__abs_mean_ratio"] = math.log(
        max(float(np.mean(abs_p)), 1e-12) / max(float(np.mean(abs_r)), 1e-12)
    )
    sd_abs_p = safe_std(abs_p)
    sd_abs_r = safe_std(abs_r)
    if np.isfinite(sd_abs_p) and np.isfinite(sd_abs_r):
        denom = math.sqrt(sd_abs_p * sd_abs_p / len(abs_p) + sd_abs_r * sd_abs_r / len(abs_r))
        out[f"{prefix}__bf_t"] = (
            (float(np.mean(abs_p)) - float(np.mean(abs_r))) / denom if denom > 1e-12 else np.nan
        )

    dists = ecdf_distances(post, ref, iqr_r)
    for key, val in dists.items():
        out[f"{prefix}__{key}"] = val

    q_post = np.quantile(post, [0.1, 0.25, 0.5, 0.75, 0.9])
    q_ref = np.quantile(ref, [0.1, 0.25, 0.5, 0.75, 0.9])
    for q, a, b in zip(("q10", "q25", "q50", "q75", "q90"), q_post, q_ref, strict=True):
        out[f"{prefix}__{q}_z"] = float((a - b) / scale)

    lo, hi = np.quantile(ref, [0.05, 0.95])
    q25, q75 = np.quantile(ref, [0.25, 0.75])
    out[f"{prefix}__tail_lo_delta"] = float(np.mean(post < lo) - 0.05)
    out[f"{prefix}__tail_hi_delta"] = float(np.mean(post > hi) - 0.05)
    out[f"{prefix}__tail_abs_delta"] = abs(out[f"{prefix}__tail_lo_delta"]) + abs(
        out[f"{prefix}__tail_hi_delta"]
    )
    out[f"{prefix}__center_delta"] = float(np.mean((post >= q25) & (post <= q75)) - 0.5)

    sk_p, ku_p = skew_kurt(post)
    sk_r, ku_r = skew_kurt(ref)
    out[f"{prefix}__skew_diff"] = sk_p - sk_r if np.isfinite([sk_p, sk_r]).all() else np.nan
    out[f"{prefix}__kurt_diff"] = ku_p - ku_r if np.isfinite([ku_p, ku_r]).all() else np.nan

    for lag in (1, 2, 5):
        ap = acf(post, lag)
        ar = acf(ref, lag)
        out[f"{prefix}__acf{lag}_diff"] = ap - ar if np.isfinite([ap, ar]).all() else np.nan
        out[f"{prefix}__abs_acf{lag}_diff"] = (
            abs(ap) - abs(ar) if np.isfinite([ap, ar]).all() else np.nan
        )
    acfs = [out.get(f"{prefix}__acf{lag}_diff", np.nan) for lag in (1, 2, 5)]
    out[f"{prefix}__ljung_proxy"] = float(np.nansum(np.square(acfs)))

    sl_p = slope(post)
    sl_r = slope(ref)
    out[f"{prefix}__slope_diff"] = sl_p - sl_r if np.isfinite([sl_p, sl_r]).all() else np.nan
    if len(post) >= 32 and len(ref) >= 32:
        out[f"{prefix}__fft_hi_power"] = spectral_hi_power(post) - spectral_hi_power(ref)
    else:
        out[f"{prefix}__fft_hi_power"] = np.nan
    return out


def spectral_hi_power(x: np.ndarray) -> float:
    y = np.asarray(x, dtype=np.float64)
    y = y - y.mean()
    if len(y) < 32:
        return float("nan")
    p = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    if p.sum() <= 0:
        return float("nan")
    f = np.fft.rfftfreq(len(y))
    return float(p[f >= 0.25].sum() / p.sum())


def fit_ar_residuals(hist: np.ndarray, online: np.ndarray, order: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Fit AR(order) on history only, then filter history and online causally."""
    if len(hist) <= order + 8:
        return hist - np.mean(hist), online - np.mean(hist)
    y = hist[order:]
    X = [np.ones(len(y))]
    for lag in range(1, order + 1):
        X.append(hist[order - lag : len(hist) - lag])
    Xmat = np.column_stack(X)
    try:
        beta, *_ = np.linalg.lstsq(Xmat, y, rcond=None)
    except np.linalg.LinAlgError:
        beta = np.r_[float(np.mean(hist)), np.zeros(order)]

    combo = np.concatenate([hist, online]).astype(np.float64)
    resid = np.full(len(combo), np.nan, dtype=np.float64)
    for t in range(order, len(combo)):
        pred = beta[0]
        for lag in range(1, order + 1):
            pred += beta[lag] * combo[t - lag]
        resid[t] = combo[t] - pred
    hist_resid = resid[order : len(hist)]
    online_resid = resid[len(hist) :]
    if len(hist_resid) < 2:
        hist_resid = hist - np.mean(hist)
    return hist_resid[np.isfinite(hist_resid)], online_resid


def extract_oracle_features(
    hist: np.ndarray,
    online: np.ndarray,
    hist_resid: np.ndarray,
    online_resid: np.ndarray,
    boundary: int,
    horizon: int | str,
    pre_reference_source: np.ndarray | None = None,
) -> dict[str, float]:
    """Known-boundary feature bank for a single series."""
    post = post_slice(online, boundary, horizon)
    post_resid = residual_post_slice(online_resid, boundary, horizon)
    features: dict[str, float] = {}
    features.update(compare_segment(post, hist, "hist"))
    features.update(compare_segment(post_resid, hist_resid, "resid_hist"))

    pre_source = online if pre_reference_source is None else pre_reference_source
    pre_boundary = boundary if pre_reference_source is None else len(pre_reference_source)
    for window in PRE_WINDOWS:
        ref = pre_slice(pre_source, pre_boundary, window)
        if len(ref) >= 5:
            name = f"pre{window}" if isinstance(window, int) else "preall"
            features.update(compare_segment(post, ref, name))
    return features


def extract_unknown_boundary_features(
    hist: np.ndarray,
    online: np.ndarray,
    boundary: int,
    horizon: int | str,
    max_candidates: int = 17,
    hist_sorted: np.ndarray | None = None,
    hist_iqr: float | None = None,
) -> dict[str, float]:
    """Boundary-search oracle: scan a fixed grid without revealing tau."""
    if horizon == "FULL":
        observed_end = len(online)
        min_post = 5
    else:
        observed_end = min(len(online), boundary + int(horizon))
        min_post = int(horizon)
    latest = observed_end - min_post
    if observed_end <= 0 or latest < 0:
        return {}
    if latest <= max_candidates - 1:
        candidates = np.arange(latest + 1, dtype=int)
    else:
        candidates = np.unique(np.linspace(0, latest, max_candidates).round().astype(int))

    best = {
        "unknown__max_abs_mean_z": np.nan,
        "unknown__max_abs_log_sd": np.nan,
        "unknown__max_ks": np.nan,
        "unknown__max_w1": np.nan,
        "unknown__max_combined": np.nan,
        "unknown__argmax_rel": np.nan,
    }
    hist_mu = float(np.mean(hist))
    hist_sd = safe_std(hist)
    ref_sorted = np.sort(hist) if hist_sorted is None else hist_sorted
    ref_iqr = (
        float(np.subtract(*np.quantile(hist, [0.75, 0.25])))
        if hist_iqr is None or not np.isfinite(hist_iqr)
        else float(hist_iqr)
    )
    if not np.isfinite(ref_iqr) or ref_iqr <= 1e-12:
        ref_iqr = hist_sd if np.isfinite(hist_sd) else 1.0
    vals: list[tuple[float, int, float, float, float, float]] = []
    for c in candidates:
        seg = online[c:observed_end]
        if len(seg) < 1:
            continue
        mean_z = (float(np.mean(seg)) - hist_mu) / (hist_sd if np.isfinite(hist_sd) else 1.0)
        sd_seg = safe_std(seg)
        log_sd = (
            math.log(max(sd_seg, 1e-12) / max(hist_sd, 1e-12))
            if np.isfinite([sd_seg, hist_sd]).all()
            else np.nan
        )
        d = ecdf_distances_against_sorted(seg, ref_sorted, ref_iqr)
        combined = np.nansum([abs(mean_z), abs(log_sd), d["ks"], d["w1"]])
        vals.append((combined, int(c), abs(mean_z), abs(log_sd), d["ks"], d["w1"]))
    if not vals:
        return best
    arr = np.array(vals, dtype=float)
    idx = int(np.nanargmax(arr[:, 0]))
    best["unknown__max_combined"] = float(arr[idx, 0])
    best["unknown__argmax_rel"] = float(arr[idx, 1] / max(observed_end, 1))
    best["unknown__max_abs_mean_z"] = float(np.nanmax(arr[:, 2]))
    best["unknown__max_abs_log_sd"] = float(np.nanmax(arr[:, 3]))
    best["unknown__max_ks"] = float(np.nanmax(arr[:, 4]))
    best["unknown__max_w1"] = float(np.nanmax(arr[:, 5]))
    return best


def assign_pseudo_taus(
    meta: pd.DataFrame,
    horizon: int | str,
    seed: int,
    n_strata: int = 8,
) -> pd.Series:
    """Matched no-break pseudo-boundaries.

    Positives contribute the empirical tau distribution.  Negatives draw from
    positives in the same n_online stratum when possible, subject to
    pseudo_tau + h <= n_online for finite h, or pseudo_tau < n_online for FULL.
    """
    m = meta.copy()
    if "id" not in m:
        m = m.reset_index().rename(columns={"index": "id"})
    h_need = 1 if horizon == "FULL" else int(horizon)
    rng = np.random.default_rng(seed)
    dev = m[m["fold"].isin(FOLDS)].copy()
    try:
        dev["len_stratum"] = pd.qcut(dev["n_online"], q=n_strata, labels=False, duplicates="drop")
    except ValueError:
        dev["len_stratum"] = 0
    if dev["len_stratum"].isna().any():
        dev["len_stratum"] = dev["len_stratum"].fillna(0)
    positives = dev[(dev["target"] == 1) & (dev["tau_index"] >= 0)].copy()
    negatives = dev[dev["target"] == 0].copy()
    by_stratum: dict[int, np.ndarray] = {}
    for s, g in positives.groupby("len_stratum"):
        by_stratum[int(s)] = g["tau_index"].to_numpy(dtype=int)
    all_tau = positives["tau_index"].to_numpy(dtype=int)

    out = pd.Series(-1, index=dev["id"].to_numpy(dtype=int), dtype=int)
    for r in negatives.itertuples(index=False):
        max_tau = int(r.n_online) - h_need
        if max_tau < 0:
            continue
        local = by_stratum.get(int(r.len_stratum), all_tau)
        cand = local[local <= max_tau]
        if len(cand) == 0:
            cand = all_tau[all_tau <= max_tau]
        if len(cand) == 0:
            continue
        out.loc[int(r.id)] = int(rng.choice(cand))
    return out


def one_row_per_series(records: pd.DataFrame) -> bool:
    return records["id"].is_unique


def fold_separation_ok(records: pd.DataFrame) -> bool:
    return records.groupby("id")["fold"].nunique().max() == 1


def model_feature_columns(df: pd.DataFrame, *, prefix: str | None = None) -> list[str]:
    cols = []
    for c in df.columns:
        if c in MODEL_EXCLUDE_COLUMNS:
            continue
        if prefix and not c.startswith(prefix):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def basic_feature_columns(cols: Iterable[str]) -> list[str]:
    available = list(cols)
    selected = [c for c in available if c in BASIC_TOKENS]
    if selected:
        return selected
    return [c for c in available if c.startswith(("hist__", "resid_hist__"))]


def best_single_oof(
    X: pd.DataFrame, y: np.ndarray, folds: np.ndarray
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Fold-pure best single statistic with sign selected on training folds."""
    out = np.full(len(y), np.nan, dtype=float)
    selections: list[dict[str, object]] = []
    cols = list(X.columns)
    arr = X.to_numpy(dtype=float)
    for f in FOLDS:
        tr = folds != f
        va = folds == f
        best = (-np.inf, None, 1)
        for j, c in enumerate(cols):
            col = arr[:, j]
            ok = tr & np.isfinite(col)
            if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
                continue
            auc = finite_auc(y[ok], col[ok])
            if not np.isfinite(auc):
                continue
            signed_auc = max(auc, 1.0 - auc)
            sign = 1 if auc >= 0.5 else -1
            if signed_auc > best[0]:
                best = (signed_auc, c, sign)
        _, col_name, sign = best
        if col_name is None:
            out[va] = 0.0
            selections.append({"fold": int(f), "feature": None, "train_auc": np.nan, "sign": 1})
        else:
            col = arr[:, cols.index(col_name)]
            median = float(np.nanmedian(col[tr])) if np.isfinite(col[tr]).any() else 0.0
            vals = np.where(np.isfinite(col[va]), col[va], median)
            out[va] = sign * vals
            selections.append(
                {
                    "fold": int(f),
                    "feature": col_name,
                    "train_auc_abs": float(best[0]),
                    "sign": int(sign),
                }
            )
    return out, selections


def crossfit_model(
    X: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    kind: str,
    seed: int,
    *,
    n_estimators: int = 500,
) -> np.ndarray:
    out = np.full(len(y), np.nan, dtype=float)
    for f in FOLDS:
        tr = folds != f
        va = folds == f
        if tr.sum() == 0 or va.sum() == 0:
            continue
        if kind == "logistic":
            model = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                StandardScaler(),
                LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    class_weight="balanced",
                    random_state=seed,
                ),
            )
        elif kind == "lgbm":
            if lgb is not None:
                model = make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True),
                    lgb.LGBMClassifier(
                        n_estimators=n_estimators,
                        learning_rate=0.04,
                        num_leaves=31,
                        min_child_samples=20,
                        subsample=0.85,
                        subsample_freq=1,
                        colsample_bytree=0.85,
                        reg_lambda=2.0,
                        objective="binary",
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=2,
                        verbose=-1,
                    ),
                )
            else:
                model = make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True),
                    ExtraTreesClassifier(
                        n_estimators=300,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=2,
                    ),
                )
        else:
            raise ValueError(f"unknown model kind {kind}")
        model.fit(X.iloc[tr], y[tr])
        out[va] = model.predict_proba(X.iloc[va])[:, 1]
    return out


def bootstrap_auc_ci(
    y: np.ndarray,
    score: np.ndarray,
    ids: np.ndarray,
    *,
    score_b: np.ndarray | None = None,
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, float | list[float]]:
    rng = np.random.default_rng(seed)
    unique = np.unique(ids)
    row_by_id = {int(s): np.flatnonzero(ids == s) for s in unique}
    vals = []
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([row_by_id[int(s)] for s in pick])
        vals.append(finite_auc(y[idx], score[idx]))
        if score_b is not None:
            diffs.append(finite_auc(y[idx], score[idx]) - finite_auc(y[idx], score_b[idx]))
    arr = np.asarray(vals, dtype=float)
    out: dict[str, float | list[float]] = {
        "mean": float(np.nanmean(arr)),
        "ci95": [float(np.nanquantile(arr, 0.025)), float(np.nanquantile(arr, 0.975))],
    }
    if score_b is not None:
        d = np.asarray(diffs, dtype=float)
        out["difference_mean"] = float(np.nanmean(d))
        out["difference_ci95"] = [
            float(np.nanquantile(d, 0.025)),
            float(np.nanquantile(d, 0.975)),
        ]
        out["difference_fraction_positive"] = float(np.nanmean(d > 0))
    return out


def load_2026_series(store_path: str, folds_path: str, ids: Iterable[int] | None = None) -> dict[int, PreparedSeries]:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    from sbr.store import load_store

    store = load_store(store_path)
    folds = pd.read_parquet(folds_path).copy()
    folds = folds.rename(columns={"has_break": "target"})
    folds["target"] = folds["target"].astype(int)
    if ids is not None:
        folds = folds[folds["id"].isin(list(ids))]
    folds = folds[folds["fold"].isin(FOLDS)].copy()
    fold_by_id = folds.set_index("id")

    prepared: dict[int, PreparedSeries] = {}
    meta_by_id = store.meta.set_index("id")
    for sid, fr in fold_by_id.iterrows():
        mr = meta_by_id.loc[sid]
        off = int(mr.off)
        n_hist = int(mr.n_hist)
        n_online = int(mr.n_online)
        hist = np.asarray(store.values[off : off + n_hist], dtype=np.float64)
        online = np.asarray(store.values[off + n_hist : off + n_hist + n_online], dtype=np.float64)
        hist_resid, online_resid = fit_ar_residuals(hist, online)
        hist_iqr = float(np.subtract(*np.quantile(hist, [0.75, 0.25])))
        prepared[int(sid)] = PreparedSeries(
            id=int(sid),
            fold=int(fr.fold),
            target=int(fr.target),
            n_hist=n_hist,
            n_online=n_online,
            tau_index=int(mr.tau_index),
            hist=hist,
            online=online,
            hist_resid=hist_resid,
            online_resid=online_resid,
            hist_sorted=np.sort(hist),
            hist_iqr=hist_iqr,
        )
    return prepared


def load_2025_series(data_dir: str) -> dict[int, PreparedSeries]:
    data = Path(data_dir)
    X = pd.read_parquet(data / "X_train.parquet")
    y = pd.read_parquet(data / "y_train.parquet").iloc[:, 0].astype(int)
    ids = np.asarray(y.index)
    folds = np.full(len(ids), -1, dtype=int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    for f, (_, va) in enumerate(skf.split(np.zeros(len(ids)), y.to_numpy())):
        folds[va] = f
    prepared: dict[int, PreparedSeries] = {}
    grouped = X.groupby(level="id", sort=False)
    fold_by_id = dict(zip(ids.tolist(), folds.tolist(), strict=True))
    for sid, g in grouped:
        if sid not in y.index:
            continue
        values = g["value"].to_numpy(dtype=np.float64)
        period = g["period"].to_numpy()
        hist = values[period == 0]
        post = values[period == 1]
        hist_resid, online_resid = fit_ar_residuals(hist, post)
        hist_iqr = float(np.subtract(*np.quantile(hist, [0.75, 0.25])))
        prepared[int(sid)] = PreparedSeries(
            id=int(sid),
            fold=int(fold_by_id[int(sid)]),
            target=int(y.loc[sid]),
            n_hist=len(hist),
            n_online=len(post),
            tau_index=0,
            hist=hist,
            online=post,
            hist_resid=hist_resid,
            online_resid=online_resid,
            hist_sorted=np.sort(hist),
            hist_iqr=hist_iqr,
        )
    return prepared


def build_2026_records(
    series: dict[int, PreparedSeries],
    horizon: int | str,
    pseudo_seed: int,
) -> pd.DataFrame:
    meta = pd.DataFrame(
        [
            {
                "id": s.id,
                "fold": s.fold,
                "target": s.target,
                "tau_index": s.tau_index,
                "n_hist": s.n_hist,
                "n_online": s.n_online,
            }
            for s in series.values()
        ]
    )
    pseudo = assign_pseudo_taus(meta, horizon, pseudo_seed)
    rows = []
    h_need = 1 if horizon == "FULL" else int(horizon)
    for s in series.values():
        if s.target == 1:
            boundary = s.tau_index
        else:
            boundary = int(pseudo.get(s.id, -1))
        if boundary < 0 or boundary + h_need > s.n_online:
            continue
        post = post_slice(s.online, boundary, horizon)
        if len(post) == 0:
            continue
        rec = {
            "id": s.id,
            "fold": s.fold,
            "target": s.target,
            "boundary": boundary,
            "rel_boundary": boundary / max(s.n_online, 1),
            "n_hist": s.n_hist,
            "n_online": s.n_online,
            "post_len": len(post),
            "horizon": horizon_label(horizon),
            "pseudo_seed": pseudo_seed,
            "break_family": s.break_family,
        }
        rec.update(extract_oracle_features(s.hist, s.online, s.hist_resid, s.online_resid, boundary, horizon))
        rows.append(rec)
    return pd.DataFrame(rows)


def build_unknown_records(
    known_records: pd.DataFrame,
    series: dict[int, PreparedSeries],
    horizon: int | str,
) -> pd.DataFrame:
    rows = []
    for r in known_records.itertuples(index=False):
        s = series[int(r.id)]
        rec = {
            "id": s.id,
            "fold": s.fold,
            "target": s.target,
            "boundary": int(r.boundary),
            "rel_boundary": float(r.rel_boundary),
            "n_hist": s.n_hist,
            "n_online": s.n_online,
            "post_len": int(r.post_len),
            "horizon": horizon_label(horizon),
            "pseudo_seed": int(r.pseudo_seed),
            "break_family": s.break_family,
        }
        rec.update(
            extract_unknown_boundary_features(
                s.hist,
                s.online,
                int(r.boundary),
                horizon,
                hist_sorted=s.hist_sorted,
                hist_iqr=s.hist_iqr,
            )
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def build_2025_records(series: dict[int, PreparedSeries]) -> pd.DataFrame:
    rows = []
    for s in series.values():
        rec = {
            "id": s.id,
            "fold": s.fold,
            "target": s.target,
            "boundary": 0,
            "rel_boundary": 0.0,
            "n_hist": s.n_hist,
            "n_online": s.n_online,
            "post_len": s.n_online,
            "horizon": "FULL",
            "pseudo_seed": -1,
            "break_family": "actual_2025",
        }
        rec.update(
            extract_oracle_features(
                s.hist,
                s.online,
                s.hist_resid,
                s.online_resid,
                0,
                "FULL",
                pre_reference_source=s.hist,
            )
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def score_models(
    df: pd.DataFrame,
    seed: int,
    *,
    include_unknown: bool = False,
    n_estimators: int = 500,
) -> dict[str, object]:
    y = df["target"].to_numpy(dtype=int)
    folds = df["fold"].to_numpy(dtype=int)
    feature_cols = model_feature_columns(df)
    if include_unknown:
        feature_cols = model_feature_columns(df, prefix="unknown__")
    X_all = df[feature_cols].copy()
    X_basic = df[basic_feature_columns(feature_cols)].copy() if not include_unknown else X_all
    out: dict[str, object] = {"n_features_rich": len(X_all.columns), "n_features_basic": len(X_basic.columns)}

    single_score, single_sel = best_single_oof(X_all, y, folds)
    auc, per = auc_with_folds(y, single_score, folds)
    out["best_single"] = {
        "auc": auc,
        "per_fold": per,
        "selections": single_sel,
        "oof": single_score,
    }

    logit = crossfit_model(X_basic, y, folds, "logistic", seed, n_estimators=n_estimators)
    auc, per = auc_with_folds(y, logit, folds)
    out["logistic"] = {"auc": auc, "per_fold": per, "oof": logit}

    lgbm_basic = crossfit_model(X_basic, y, folds, "lgbm", seed, n_estimators=n_estimators)
    auc, per = auc_with_folds(y, lgbm_basic, folds)
    out["lgbm_basic"] = {"auc": auc, "per_fold": per, "oof": lgbm_basic}

    if include_unknown:
        rich = lgbm_basic.copy()
        auc, per = auc_with_folds(y, rich, folds)
    else:
        rich = crossfit_model(X_all, y, folds, "lgbm", seed + 17, n_estimators=n_estimators)
        auc, per = auc_with_folds(y, rich, folds)
    out["lgbm_rich"] = {"auc": auc, "per_fold": per, "oof": rich}
    return out


def current_model_scores(
    records: pd.DataFrame,
    store_meta_path: str,
    oof_path: str,
    horizon: int | str,
) -> np.ndarray | None:
    if not oof_path or not os.path.exists(oof_path):
        return None
    meta = pd.read_parquet(store_meta_path).copy()
    n_on = meta["n_online"].to_numpy(dtype=int)
    off = np.r_[0, np.cumsum(n_on)[:-1]].astype(np.int64)
    off_by_id = dict(zip(meta["id"].to_numpy(dtype=int), off, strict=True))
    n_by_id = dict(zip(meta["id"].to_numpy(dtype=int), n_on, strict=True))
    pred = np.load(oof_path, mmap_mode="r")
    out = []
    for r in records.itertuples(index=False):
        sid = int(r.id)
        boundary = int(r.boundary)
        n = int(n_by_id[sid])
        if horizon == "FULL":
            t = n - 1
        else:
            t = min(boundary + int(horizon) - 1, n - 1)
        out.append(float(pred[off_by_id[sid] + t]))
    return np.asarray(out, dtype=float)


def derive_break_families(series: dict[int, PreparedSeries], seed: int = 0) -> dict[str, object]:
    """Simple transparent full-post family assignment for positives."""
    rows = build_2026_records(series, "FULL", seed)
    families = {
        "location": ["hist__abs_mean_z", "hist__median_z", "hist__trim_mean_z"],
        "scale": ["hist__log_sd_ratio", "hist__log_var_ratio", "hist__log_mad_ratio", "hist__bf_t"],
        "distribution": ["hist__ks", "hist__w1", "hist__energy", "hist__cvm", "hist__tail_abs_delta"],
        "dependence": ["hist__abs_acf1_diff", "hist__abs_acf2_diff", "hist__abs_acf5_diff"],
        "trend": ["hist__slope_diff"],
    }
    neg = rows[rows["target"] == 0]
    pos = rows[rows["target"] == 1]
    med_mad: dict[str, tuple[float, float]] = {}
    for fam, cols in families.items():
        vals = []
        for c in cols:
            if c in neg:
                vals.append(np.abs(neg[c].to_numpy(dtype=float)))
        allv = np.concatenate(vals) if vals else np.array([0.0])
        allv = allv[np.isfinite(allv)]
        med = float(np.median(allv)) if len(allv) else 0.0
        spread = mad(allv)
        if not np.isfinite(spread) or spread <= 1e-9:
            spread = float(np.std(allv)) if len(allv) else 1.0
        if not np.isfinite(spread) or spread <= 1e-9:
            spread = 1.0
        med_mad[fam] = (med, spread)

    counts: dict[str, int] = {}
    for r in pos.itertuples(index=False):
        scores = {}
        for fam, cols in families.items():
            vals = [abs(float(getattr(r, c))) for c in cols if hasattr(r, c) and np.isfinite(getattr(r, c))]
            raw = max(vals) if vals else 0.0
            med, spread = med_mad[fam]
            scores[fam] = (raw - med) / spread
        best = max(scores, key=scores.get)
        cls = best if scores[best] >= 2.0 else "weak_unclassified"
        series[int(r.id)].break_family = cls
        counts[cls] = counts.get(cls, 0) + 1
    return {"method": "fixed full-post effect family with no-break placebo MAD threshold 2.0", "counts": counts}


def summarize_numeric(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "q25": np.nan, "median": np.nan, "q75": np.nan, "max": np.nan}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "q75": float(np.quantile(x, 0.75)),
        "max": float(np.max(x)),
    }


def run_2026_frontier(
    store: str,
    folds: str,
    current_oof: str | None,
    output_prefix: Path,
    *,
    n_estimators: int = 500,
    limit_series: int | None = None,
    horizons: tuple[int | str, ...] = HORIZONS,
    seeds: tuple[int, ...] = PSEUDO_SEEDS,
    controls_all_seeds: bool = False,
) -> dict[str, object]:
    t0 = time.time()
    all_folds = pd.read_parquet(folds)
    ids = None
    if limit_series:
        ids = all_folds[all_folds["fold"].isin(FOLDS)]["id"].head(limit_series).tolist()
    series = load_2026_series(store, folds, ids=ids)
    family_info = derive_break_families(series, seed=0)

    store_meta_path = str(Path(store) / "meta.parquet")
    result: dict[str, object] = {
        "label": "ORACLE / DIAGNOSTIC -- NOT DEPLOYABLE",
        "base_sha": git_sha(),
        "store": store,
        "folds": folds,
        "current_oof": current_oof,
        "n_estimators": n_estimators,
        "controls_all_seeds": controls_all_seeds,
        "key_control_horizons": sorted(KEY_CONTROL_HORIZONS, key=lambda x: horizon_order(parse_horizon(x))),
        "horizons": [horizon_label(h) for h in horizons],
        "pseudo_seeds": list(seeds),
        "family_assignment": family_info,
        "runs": {},
        "summary": {},
        "controls": {},
        "bootstrap": {},
        "headroom": {},
        "break_family_frontier": {},
        "unknown_boundary": {},
    }
    wide_rows = []

    for horizon in horizons:
        hkey = horizon_label(horizon)
        result["runs"][hkey] = {}
        print(f"\n=== 2026 horizon {hkey} ===", flush=True)
        per_seed_summary = []
        for seed in seeds:
            seed_t0 = time.time()
            df = build_2026_records(series, horizon, seed)
            if len(df) == 0:
                continue
            assert one_row_per_series(df)
            assert fold_separation_ok(df)
            metrics = score_models(df, seed, n_estimators=n_estimators)
            y = df["target"].to_numpy(dtype=int)
            fold_arr = df["fold"].to_numpy(dtype=int)
            run_controls = (controls_all_seeds or seed == seeds[0]) and hkey in KEY_CONTROL_HORIZONS

            meta_auc = np.nan
            meta_pf = [np.nan] * 5
            perm_auc = np.nan
            perm_pf = [np.nan] * 5
            random_boundary_rich = {"auc": np.nan, "per_fold": [np.nan] * 5}
            unknown_rich = {"auc": np.nan, "per_fold": [np.nan] * 5}
            known_minus_unknown = np.nan
            if run_controls:
                meta_cols = ["boundary", "rel_boundary", "n_hist", "n_online", "post_len"]
                meta_score = crossfit_model(
                    df[meta_cols], y, fold_arr, "logistic", seed, n_estimators=n_estimators
                )
                meta_auc, meta_pf = auc_with_folds(y, meta_score, fold_arr)

                perm_y = np.random.default_rng(seed + 999).permutation(y)
                perm_score = crossfit_model(
                    df[model_feature_columns(df)],
                    perm_y,
                    fold_arr,
                    "logistic",
                    seed,
                    n_estimators=n_estimators,
                )
                perm_auc, perm_pf = auc_with_folds(perm_y, perm_score, fold_arr)

                both_records = random_boundary_records(series, df, horizon, seed)
                both_metrics = score_models(
                    both_records, seed, n_estimators=max(100, min(n_estimators, 300))
                )
                random_boundary_rich = strip_oof(both_metrics)["lgbm_rich"]

                unknown_df = build_unknown_records(df, series, horizon)
                unknown_metrics = score_models(
                    unknown_df,
                    seed,
                    include_unknown=True,
                    n_estimators=max(100, min(n_estimators, 300)),
                )
                unknown_rich = strip_oof(unknown_metrics)["lgbm_rich"]
                known_minus_unknown = float(metrics["lgbm_rich"]["auc"] - unknown_rich["auc"])

            current_scores = (
                current_model_scores(df, store_meta_path, current_oof, horizon) if current_oof else None
            )
            current_auc = np.nan
            current_pf = [np.nan] * 5
            if current_scores is not None:
                current_auc, current_pf = auc_with_folds(y, current_scores, fold_arr)

            seed_detail: dict[str, object] = {
                "n_series": int(len(df)),
                "n_pos": int(df["target"].sum()),
                "n_neg": int((df["target"] == 0).sum()),
                "eligibility_rate": float(len(df) / len(series)),
                "n_online": summarize_numeric(df["n_online"].to_numpy()),
                "boundary": summarize_numeric(df["boundary"].to_numpy()),
                "tau_positive": summarize_numeric(df.loc[df["target"] == 1, "boundary"].to_numpy()),
                "pseudo_tau_negative": summarize_numeric(
                    df.loc[df["target"] == 0, "boundary"].to_numpy()
                ),
                "models": strip_oof(metrics),
                "controls": {
                    "metadata_only": {"auc": meta_auc, "per_fold": meta_pf},
                    "permuted_labels_logistic": {"auc": perm_auc, "per_fold": perm_pf},
                    "random_boundary_both_lgbm_rich": random_boundary_rich,
                },
                "current_legal_rt300": {"auc": current_auc, "per_fold": current_pf},
                "unknown_boundary_lgbm": unknown_rich,
                "known_minus_unknown": known_minus_unknown,
                "oracle_minus_current_rt300": float(metrics["lgbm_rich"]["auc"] - current_auc)
                if np.isfinite(current_auc)
                else np.nan,
            }

            if hkey in BOOTSTRAP_HORIZONS and seed == seeds[0]:
                result["bootstrap"][hkey] = {
                    "lgbm_rich": bootstrap_auc_ci(y, metrics["lgbm_rich"]["oof"], df["id"].to_numpy()),
                }
                if current_scores is not None:
                    result["bootstrap"][hkey]["headroom_vs_current_rt300"] = bootstrap_auc_ci(
                        y,
                        metrics["lgbm_rich"]["oof"],
                        df["id"].to_numpy(),
                        score_b=current_scores,
                    )

            if hkey in {"5", "20", "50", "100", "200", "FULL"} and seed == seeds[0]:
                fam_rows = []
                rich = metrics["lgbm_rich"]["oof"]
                for fam in sorted(set(df["break_family"]) - {"none"}):
                    mask = (df["target"].to_numpy() == 0) | (
                        (df["target"].to_numpy() == 1) & (df["break_family"].to_numpy() == fam)
                    )
                    if mask.sum() > 10 and len(np.unique(y[mask])) == 2:
                        fam_rows.append(
                            {
                                "family": fam,
                                "n_pos": int(((y == 1) & (df["break_family"].to_numpy() == fam)).sum()),
                                "n_neg": int((y[mask] == 0).sum()),
                                "lgbm_rich_auc": finite_auc(y[mask], rich[mask]),
                            }
                        )
                result["break_family_frontier"][hkey] = fam_rows

            result["runs"][hkey][str(seed)] = seed_detail
            per_seed_summary.append(seed_detail)
            print(
                f"seed {seed}: n={len(df)} pos={seed_detail['n_pos']} "
                f"single={metrics['best_single']['auc']:.4f} "
                f"logit={metrics['logistic']['auc']:.4f} "
                f"lgbm_basic={metrics['lgbm_basic']['auc']:.4f} "
                f"lgbm_rich={metrics['lgbm_rich']['auc']:.4f} "
                f"current={current_auc:.4f} unknown={unknown_rich['auc']:.4f} "
                f"elapsed={time.time() - seed_t0:.1f}s",
                flush=True,
            )

        result["summary"][hkey] = summarize_horizon(per_seed_summary)
        wide_rows.append(summary_csv_row(hkey, result["summary"][hkey]))

    result["thresholds"] = threshold_crossings(result["summary"])
    result["runtime_s"] = round(time.time() - t0, 1)
    write_outputs_2026(result, wide_rows, output_prefix)
    return result


def random_boundary_records(
    series: dict[int, PreparedSeries],
    base_records: pd.DataFrame,
    horizon: int | str,
    seed: int,
) -> pd.DataFrame:
    """Negative control C: random pseudo boundary for positives and negatives."""
    rng = np.random.default_rng(seed + 3001)
    h_need = 1 if horizon == "FULL" else int(horizon)
    rows = []
    for r in base_records.itertuples(index=False):
        s = series[int(r.id)]
        max_tau = s.n_online - h_need
        if max_tau < 0:
            continue
        boundary = int(rng.integers(0, max_tau + 1))
        post = post_slice(s.online, boundary, horizon)
        if len(post) == 0:
            continue
        rec = {
            "id": s.id,
            "fold": s.fold,
            "target": s.target,
            "boundary": boundary,
            "rel_boundary": boundary / max(s.n_online, 1),
            "n_hist": s.n_hist,
            "n_online": s.n_online,
            "post_len": len(post),
            "horizon": horizon_label(horizon),
            "pseudo_seed": seed,
            "break_family": s.break_family,
        }
        rec.update(extract_oracle_features(s.hist, s.online, s.hist_resid, s.online_resid, boundary, horizon))
        rows.append(rec)
    return pd.DataFrame(rows)


def strip_oof(metrics: dict[str, object]) -> dict[str, object]:
    out = {}
    for key, val in metrics.items():
        if not isinstance(val, dict):
            out[key] = val
            continue
        out[key] = {k: v for k, v in val.items() if k != "oof"}
    return out


def summarize_horizon(seed_details: list[dict[str, object]]) -> dict[str, object]:
    if not seed_details:
        return {}
    models = ["best_single", "logistic", "lgbm_basic", "lgbm_rich"]
    out: dict[str, object] = {
        "n_pos_mean": float(np.mean([d["n_pos"] for d in seed_details])),
        "n_neg_mean": float(np.mean([d["n_neg"] for d in seed_details])),
        "eligibility_rate_mean": float(np.mean([d["eligibility_rate"] for d in seed_details])),
    }
    for model in models:
        aucs = np.array([d["models"][model]["auc"] for d in seed_details], dtype=float)
        out[model] = {
            "mean": float(np.nanmean(aucs)),
            "std": float(np.nanstd(aucs)),
            "min": float(np.nanmin(aucs)),
            "max": float(np.nanmax(aucs)),
        }
    for key in ("current_legal_rt300", "unknown_boundary_lgbm"):
        aucs = np.array([d[key]["auc"] for d in seed_details], dtype=float)
        out[key] = {
            "mean": float(np.nanmean(aucs)),
            "std": float(np.nanstd(aucs)),
            "min": float(np.nanmin(aucs)),
            "max": float(np.nanmax(aucs)),
        }
    out["known_minus_unknown_mean"] = float(
        np.nanmean([d["known_minus_unknown"] for d in seed_details])
    )
    out["oracle_minus_current_rt300_mean"] = float(
        np.nanmean([d["oracle_minus_current_rt300"] for d in seed_details])
    )
    return out


def summary_csv_row(hkey: str, summary: dict[str, object]) -> dict[str, object]:
    row: dict[str, object] = {
        "h": hkey,
        "h_order": horizon_order(parse_horizon(hkey)),
        "n_pos": summary.get("n_pos_mean"),
        "n_neg": summary.get("n_neg_mean"),
        "eligibility_rate": summary.get("eligibility_rate_mean"),
    }
    for model in ("best_single", "logistic", "lgbm_basic", "lgbm_rich", "current_legal_rt300", "unknown_boundary_lgbm"):
        info = summary.get(model, {})
        row[f"{model}_auc_mean"] = info.get("mean") if isinstance(info, dict) else np.nan
        row[f"{model}_auc_std"] = info.get("std") if isinstance(info, dict) else np.nan
    row["known_minus_unknown_mean"] = summary.get("known_minus_unknown_mean")
    row["oracle_minus_current_rt300_mean"] = summary.get("oracle_minus_current_rt300_mean")
    return row


def threshold_crossings(summary: dict[str, object]) -> dict[str, str | None]:
    thresholds = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
    out: dict[str, str | None] = {}
    ordered = sorted(summary.items(), key=lambda kv: horizon_order(parse_horizon(kv[0])))
    for th in thresholds:
        hit = None
        for h, s in ordered:
            auc = s.get("lgbm_rich", {}).get("mean") if isinstance(s, dict) else None
            if auc is not None and np.isfinite(auc) and auc >= th:
                hit = h
                break
        out[f"{th:.2f}"] = hit
    out["exceeds_0.90"] = any(
        (s.get("lgbm_rich", {}).get("mean", np.nan) > 0.90)
        for s in summary.values()
        if isinstance(s, dict)
    )
    return out


def run_2025_benchmark(data_dir: str, output_prefix: Path, *, n_estimators: int = 500) -> dict[str, object]:
    t0 = time.time()
    series = load_2025_series(data_dir)
    df = build_2025_records(series)
    metrics = score_models(df, 2026, n_estimators=n_estimators)
    y = df["target"].to_numpy(dtype=int)
    folds = df["fold"].to_numpy(dtype=int)
    meta = {
        "path": data_dir,
        "files": dataset_file_manifest(data_dir),
        "n_series": int(len(df)),
        "n_pos": int(df["target"].sum()),
        "n_neg": int((df["target"] == 0).sum()),
        "schema": {
            "X_train": "MultiIndex(id,time), columns value, period; period 0 is pre-boundary and period 1 is post-boundary",
            "y_train": "Index(id), one boolean column structural_breakpoint",
        },
    }
    result = {
        "label": "2025 actual offline task benchmark; separate from 2026 oracle diagnostics",
        "data": meta,
        "folds": "generated deterministic StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)",
        "models": strip_oof(metrics),
        "bootstrap": {
            "lgbm_rich": bootstrap_auc_ci(y, metrics["lgbm_rich"]["oof"], df["id"].to_numpy())
        },
        "runtime_s": round(time.time() - t0, 1),
    }
    write_outputs_2025(result, output_prefix)
    print(
        f"2025 benchmark: single={metrics['best_single']['auc']:.4f} "
        f"logit={metrics['logistic']['auc']:.4f} "
        f"lgbm_basic={metrics['lgbm_basic']['auc']:.4f} "
        f"lgbm_rich={metrics['lgbm_rich']['auc']:.4f}",
        flush=True,
    )
    return result


def dataset_file_manifest(data_dir: str) -> list[dict[str, object]]:
    out = []
    for path in sorted(Path(data_dir).glob("*")):
        if path.is_file():
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
            out.append({"name": path.name, "bytes": path.stat().st_size, "sha256": h.hexdigest()})
    return out


def write_outputs_2026(result: dict[str, object], rows: list[dict[str, object]], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    png_path = output_prefix.with_suffix(".png")
    md_path = output_prefix.with_suffix(".md")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, allow_nan=True)
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    write_frontier_png(rows, png_path)
    write_2026_markdown(result, rows, md_path)


def write_outputs_2025(result: dict[str, object], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(output_prefix.with_suffix(".json"), "w") as f:
        json.dump(result, f, indent=2, allow_nan=True)


def write_2026_markdown(result: dict[str, object], rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Oracle Information Frontier 2026",
        "",
        "**ORACLE / DIAGNOSTIC -- NOT DEPLOYABLE.** Uses true/pseudo boundaries and future post-boundary observations. No production model or submission code is modified.",
        "",
        f"Runtime: {result.get('runtime_s')} seconds.",
        f"Store: `{result.get('store')}`.",
        f"Folds: `{result.get('folds')}`.",
        f"Current legal OOF vector: `{result.get('current_oof')}`.",
        "",
        "## Frontier Summary",
        "",
        "| h | n_pos | n_neg | best single-stat AUC | logistic | LGBM basic | LGBM rich | current RT-300 | unknown-boundary LGBM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['h']} | {r['n_pos']:.1f} | {r['n_neg']:.1f} | "
            f"{r['best_single_auc_mean']:.4f} | {r['logistic_auc_mean']:.4f} | "
            f"{r['lgbm_basic_auc_mean']:.4f} | {r['lgbm_rich_auc_mean']:.4f} | "
            f"{r['current_legal_rt300_auc_mean']:.4f} | {r['unknown_boundary_lgbm_auc_mean']:.4f} |"
        )
    lines += [
        "",
        "## Threshold Crossings",
        "",
    ]
    for k, v in result.get("thresholds", {}).items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Pseudo-Boundary Construction",
        "",
        "No-break pseudo boundaries are drawn deterministically for seeds 0, 1, 7, 42, and 2026 from the empirical positive tau distribution within n_online strata, subject to `pseudo_tau + h <= n_online` (or `pseudo_tau < n_online` for FULL). The main table reports seed means; JSON includes per-seed fold scores, tau distributions, metadata controls, and sensitivity.",
        "",
        "## Controls",
        "",
        "Controls in the JSON include fold-pure metadata-only logistic AUC, permuted-label logistic AUC, and a random-boundary-for-both-groups LGBM control for every horizon/seed.",
        "",
        "## Break Families",
        "",
        "Family labels are simple full-post effect categories derived before horizon scoring using a no-break placebo MAD threshold. They are diagnostic, not a new validated taxonomy.",
    ]
    for h, fam_rows in result.get("break_family_frontier", {}).items():
        lines.append("")
        lines.append(f"### h={h}")
        lines.append("")
        lines.append("| family | n_pos | n_neg | LGBM rich AUC |")
        lines.append("|---|---:|---:|---:|")
        for row in fam_rows:
            lines.append(
                f"| {row['family']} | {row['n_pos']} | {row['n_neg']} | {row['lgbm_rich_auc']:.4f} |"
            )
    path.write_text("\n".join(lines) + "\n")


def write_frontier_png(rows: list[dict[str, object]], path: Path) -> None:
    """Small dependency-free PNG line plot."""
    width, height = 1100, 700
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    left, right, top, bottom = 90, 1040, 70, 610
    def put(x: int, y: int, color: tuple[int, int, int], r: int = 1) -> None:
        x0, x1 = max(0, x - r), min(width, x + r + 1)
        y0, y1 = max(0, y - r), min(height, y + r + 1)
        img[y0:y1, x0:x1] = color
    def line(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], r: int = 1) -> None:
        n = max(abs(x1 - x0), abs(y1 - y0), 1)
        for t in range(n + 1):
            x = int(round(x0 + (x1 - x0) * t / n))
            y = int(round(y0 + (y1 - y0) * t / n))
            put(x, y, color, r)

    for x in range(left, right + 1):
        put(x, bottom, (30, 30, 30))
    for y in range(top, bottom + 1):
        put(left, y, (30, 30, 30))
    if not rows:
        save_png(img, path)
        return
    xs_raw = np.array([horizon_order(parse_horizon(str(r["h"]))) for r in rows], dtype=float)
    xs = np.log10(xs_raw)
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = 0.45, 0.95
    def px(v: float) -> int:
        return int(left + (np.log10(v) - xmin) / max(xmax - xmin, 1e-9) * (right - left))
    def py(v: float) -> int:
        return int(bottom - (v - ymin) / (ymax - ymin) * (bottom - top))
    for yv in np.linspace(0.5, 0.9, 5):
        yy = py(float(yv))
        line(left, yy, right, yy, (230, 230, 230))
        draw_text(img, 22, yy - 5, f"{yv:.2f}", (60, 60, 60))
    for r in rows:
        xv = horizon_order(parse_horizon(str(r["h"])))
        xx = px(xv)
        line(xx, bottom, xx, bottom + 5, (30, 30, 30))
        draw_text(img, xx - 12, bottom + 14, str(r["h"]), (60, 60, 60))
    series = [
        ("best_single_auc_mean", (40, 120, 200), "single"),
        ("logistic_auc_mean", (20, 150, 80), "logit"),
        ("lgbm_basic_auc_mean", (220, 120, 30), "lgbm basic"),
        ("lgbm_rich_auc_mean", (190, 40, 50), "lgbm rich"),
        ("current_legal_rt300_auc_mean", (100, 100, 100), "current"),
    ]
    for key, color, label in series:
        pts = []
        for r in rows:
            val = r.get(key)
            if val is None or not np.isfinite(float(val)):
                continue
            pts.append((px(horizon_order(parse_horizon(str(r["h"])))), py(float(val))))
        for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
            line(x0, y0, x1, y1, color, 2)
        for x, y in pts:
            put(x, y, color, 4)
        yleg = 95 + 22 * series.index((key, color, label))
        line(780, yleg + 5, 830, yleg + 5, color, 2)
        draw_text(img, 840, yleg, label, color)
    draw_text(img, 90, 25, "ORACLE INFORMATION FRONTIER 2026 - NOT DEPLOYABLE", (20, 20, 20))
    draw_text(img, 430, 660, "post-boundary observations h (log scale)", (20, 20, 20))
    draw_text(img, 10, 48, "AUC", (20, 20, 20))
    save_png(img, path)


FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}


def draw_text(img: np.ndarray, x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
    xx = x
    for ch in text.upper():
        glyph = FONT.get(ch, FONT[" "])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    yy = y + gy
                    px = xx + gx
                    if 0 <= yy < img.shape[0] and 0 <= px < img.shape[1]:
                        img[yy, px] = color
        xx += 6


def save_png(img: np.ndarray, path: Path) -> None:
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def write_comparison_report(
    result_2026: dict[str, object] | None,
    result_2025: dict[str, object] | None,
    path: Path,
) -> None:
    lines = [
        "# 2025 vs 2026 Information-Structure Benchmark",
        "",
        "**ORACLE / DIAGNOSTIC.** This report compares offline/known-boundary information structures and must not be read as deployable 2026 performance.",
        "",
    ]
    if result_2025:
        m25 = result_2025["models"]["lgbm_rich"]
        lines += [
            "## 2025 Actual Task",
            "",
            f"- Path: `{result_2025['data']['path']}`",
            f"- Series: {result_2025['data']['n_series']} ({result_2025['data']['n_pos']} break / {result_2025['data']['n_neg']} no-break)",
            f"- Rich LGBM CV AUC: {m25['auc']:.4f}",
            f"- Fold AUCs: {' / '.join(f'{x:.4f}' for x in m25['per_fold'])}",
            "",
        ]
    if result_2026:
        full = result_2026["summary"].get("FULL", {})
        h100 = result_2026["summary"].get("100", {})
        h200 = result_2026["summary"].get("200", {})
        lines += [
            "## 2026 Training Data Oracle",
            "",
            f"- FULL known-boundary rich LGBM AUC: {full.get('lgbm_rich', {}).get('mean', np.nan):.4f}",
            f"- h=200 known-boundary rich LGBM AUC: {h200.get('lgbm_rich', {}).get('mean', np.nan):.4f}",
            f"- h=100 known-boundary rich LGBM AUC: {h100.get('lgbm_rich', {}).get('mean', np.nan):.4f}",
            f"- FULL current RT-300 matched series AUC: {full.get('current_legal_rt300', {}).get('mean', np.nan):.4f}",
            "",
        ]
    path.write_text("\n".join(lines) + "\n")


def git_sha() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def default_store_path(root: Path) -> str:
    candidates = [
        os.environ.get("SBR_STORE"),
        str(root / "cache/store"),
        str(root.parent / "structural-break-claude-wave3/cache/store"),
        str(root.parent / "structural-break-codex-wave3/cache/store"),
        str(root.parent / "structural-break/cache/store"),
    ]
    for c in candidates:
        if c and (Path(c) / "meta.parquet").exists() and (Path(c) / "values.npy").exists():
            return c
    raise FileNotFoundError("Could not find 2026 cache/store; pass --store")


def default_folds_path(root: Path) -> str:
    candidates = [
        str(root / "research/folds/folds.parquet"),
        str(root.parent / "structural-break-claude-wave3/research/folds/folds.parquet"),
        str(root.parent / "structural-break-codex-wave3/research/folds/folds.parquet"),
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    raise FileNotFoundError("Could not find folds.parquet; pass --folds")


def default_oof_path(root: Path) -> str | None:
    candidates = [
        str(root / "research/oof/RT-300.npy"),
        str(root.parent / "structural-break-claude-wave3/research/oof/RT-300.npy"),
        str(root.parent / "structural-break-codex-wave3/research/oof/RT-300.npy"),
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def default_2025_data_path() -> str | None:
    candidates = [
        "/home/user/Documents - Graham’s MacBook Pro/structural-break-project/competitions/structural-break/quickstarters/baseline/data",
    ]
    for c in candidates:
        p = Path(c)
        if (p / "X_train.parquet").exists() and (p / "y_train.parquet").exists():
            return c
    return None


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=None, help="2026 cache/store directory")
    parser.add_argument("--folds", default=None, help="2026 folds parquet")
    parser.add_argument("--current-oof", default=None, help="Current legal OOF npy, e.g. RT-300.npy")
    parser.add_argument("--data2025", default=None, help="2025 offline task data directory")
    parser.add_argument("--skip-2026", action="store_true")
    parser.add_argument("--skip-2025", action="store_true")
    parser.add_argument("--limit-series", type=int, default=None, help="Debug only")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    parser.add_argument("--seeds", default=",".join(map(str, PSEUDO_SEEDS)))
    parser.add_argument(
        "--controls-all-seeds",
        action="store_true",
        help="Run expensive leakage controls and unknown-boundary scans for every pseudo seed",
    )
    args = parser.parse_args(argv)

    output_2026 = root / "research/reports/oracle_information_frontier_2026"
    output_2025 = root / "research/reports/benchmark_2025_vs_2026"
    result_2026 = None
    result_2025 = None

    horizons = tuple(parse_horizon(x.strip()) for x in args.horizons.split(",") if x.strip())
    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())

    if not args.skip_2026:
        store = args.store or default_store_path(root)
        folds = args.folds or default_folds_path(root)
        current = args.current_oof
        if current is None:
            current = default_oof_path(root)
        result_2026 = run_2026_frontier(
            store,
            folds,
            current,
            output_2026,
            n_estimators=args.n_estimators,
            limit_series=args.limit_series,
            horizons=horizons,
            seeds=seeds,
            controls_all_seeds=args.controls_all_seeds,
        )
    if not args.skip_2025:
        data2025 = args.data2025 or default_2025_data_path()
        if data2025:
            result_2025 = run_2025_benchmark(data2025, output_2025, n_estimators=args.n_estimators)
        else:
            print("2025 data not found; skipping 2025 benchmark", flush=True)

    if result_2026 or result_2025:
        write_comparison_report(result_2026, result_2025, output_2025.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
