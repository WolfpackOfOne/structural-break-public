"""Per-series feature engineering for the competition (id, time)-indexed format.

The competition task is: for each independent series (identified by ``id``),
decide whether a structural break occurred at that series' ``period`` boundary
(``period == 0`` before, ``period == 1`` after). This module turns one series'
rows into a fixed-length feature vector comparing its "before" and "after"
segments, plus optional features derived from the existing change-point
detectors in :mod:`structural_break.detectors`.

``build_feature_matrix`` applies this to every series in a competition-format
``X`` frame and returns one row per series id, ready to feed into any
scikit-learn-style classifier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .detectors import BreakDetector, CusumDetector, PeltDetector, RollingZScoreDetector

__all__ = [
    "STAT_FEATURE_COLUMNS",
    "extract_series_features",
    "build_feature_matrix",
    "detector_scores_for_series",
    "build_detector_feature_matrix",
]


def _safe(value: float) -> float:
    """Coerce NaN/inf (from degenerate segments, e.g. n<2) to 0.0."""
    return float(value) if np.isfinite(value) else 0.0


def _slope(y: np.ndarray) -> float:
    """OLS slope of ``y`` against its own index; 0.0 if too short to fit."""
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    return _safe(float(np.polyfit(x, y, 1)[0]))


def _autocorr_lag1(y: np.ndarray) -> float:
    if len(y) < 3:
        return 0.0
    a, b = y[:-1], y[1:]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return _safe(float(np.corrcoef(a, b)[0, 1]))


def extract_series_features(group: pd.DataFrame) -> dict[str, float]:
    """Compute before/after comparison features for one series.

    Parameters
    ----------
    group:
        Rows for a single series id, with ``value`` and ``period`` columns, in
        their original (time) order.

    Returns
    -------
    dict[str, float]
        A fixed set of features describing how the "after" segment
        (``period == 1``) differs from the "before" segment (``period == 0``).
    """
    before = group.loc[group["period"] == 0, "value"].to_numpy(dtype=float)
    after = group.loc[group["period"] == 1, "value"].to_numpy(dtype=float)

    features: dict[str, float] = {
        "n_before": float(len(before)),
        "n_after": float(len(after)),
        "length_ratio": _safe(len(after) / len(before)) if len(before) else 0.0,
        "mean_before": _safe(before.mean()) if len(before) else 0.0,
        "mean_after": _safe(after.mean()) if len(after) else 0.0,
        "std_before": _safe(before.std(ddof=1)) if len(before) > 1 else 0.0,
        "std_after": _safe(after.std(ddof=1)) if len(after) > 1 else 0.0,
        "median_before": _safe(np.median(before)) if len(before) else 0.0,
        "median_after": _safe(np.median(after)) if len(after) else 0.0,
        "min_before": _safe(before.min()) if len(before) else 0.0,
        "min_after": _safe(after.min()) if len(after) else 0.0,
        "max_before": _safe(before.max()) if len(before) else 0.0,
        "max_after": _safe(after.max()) if len(after) else 0.0,
        "skew_before": _safe(stats.skew(before)) if len(before) > 2 else 0.0,
        "skew_after": _safe(stats.skew(after)) if len(after) > 2 else 0.0,
        "kurtosis_before": _safe(stats.kurtosis(before)) if len(before) > 3 else 0.0,
        "kurtosis_after": _safe(stats.kurtosis(after)) if len(after) > 3 else 0.0,
        "slope_before": _slope(before),
        "slope_after": _slope(after),
        "autocorr1_before": _autocorr_lag1(before),
        "autocorr1_after": _autocorr_lag1(after),
        "edge_jump": _safe(after[0] - before[-1]) if len(before) and len(after) else 0.0,
    }

    features["mean_diff"] = features["mean_after"] - features["mean_before"]
    features["mean_abs_diff"] = abs(features["mean_diff"])
    features["median_diff"] = features["median_after"] - features["median_before"]
    features["std_ratio"] = _safe(
        features["std_after"] / features["std_before"] if features["std_before"] else 0.0
    )
    features["range_before"] = features["max_before"] - features["min_before"]
    features["range_after"] = features["max_after"] - features["min_after"]
    features["skew_diff"] = features["skew_after"] - features["skew_before"]
    features["kurtosis_diff"] = features["kurtosis_after"] - features["kurtosis_before"]
    features["slope_diff"] = features["slope_after"] - features["slope_before"]
    features["autocorr1_diff"] = features["autocorr1_after"] - features["autocorr1_before"]

    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        q_before = _safe(np.quantile(before, q)) if len(before) else 0.0
        q_after = _safe(np.quantile(after, q)) if len(after) else 0.0
        features[f"quantile_diff_{int(q * 100)}"] = q_after - q_before

    if len(before) > 1 and len(after) > 1:
        ks_stat, ks_p = stats.ks_2samp(before, after)
        t_stat, t_p = stats.ttest_ind(before, after, equal_var=False)
        levene_stat, levene_p = stats.levene(before, after)
        try:
            mw_stat, mw_p = stats.mannwhitneyu(before, after, alternative="two-sided")
        except ValueError:
            mw_stat, mw_p = 0.0, 1.0
    else:
        ks_stat = ks_p = t_stat = t_p = levene_stat = levene_p = mw_stat = mw_p = 0.0

    features["ks_stat"] = _safe(ks_stat)
    features["ks_pvalue"] = _safe(ks_p)
    features["t_stat"] = _safe(t_stat)
    features["t_pvalue"] = _safe(t_p)
    features["levene_stat"] = _safe(levene_stat)
    features["levene_pvalue"] = _safe(levene_p)
    features["mannwhitney_stat"] = _safe(mw_stat)
    features["mannwhitney_pvalue"] = _safe(mw_p)

    return features


#: Stable column order produced by :func:`extract_series_features`.
STAT_FEATURE_COLUMNS: list[str] = list(
    extract_series_features(
        pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0], "period": [0, 0, 1, 1]})
    ).keys()
)


def build_feature_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`extract_series_features` to every series id in ``X``.

    Parameters
    ----------
    X:
        Competition-format frame (2-level MultiIndex: id, time; columns
        ``value``, ``period``).

    Returns
    -------
    pandas.DataFrame
        One row per series id (indexed by id), columns = :data:`STAT_FEATURE_COLUMNS`.
    """
    id_level = X.index.names[0] or "id"
    rows = {
        series_id: extract_series_features(group)
        for series_id, group in X.groupby(level=0, sort=False)
    }
    matrix = pd.DataFrame.from_dict(rows, orient="index", columns=STAT_FEATURE_COLUMNS)
    matrix.index.name = id_level
    return matrix


def _as_detector_input(group: pd.DataFrame) -> pd.DataFrame:
    """Turn one series' (value, period) rows into a synthetic (timestamp, value) frame.

    The detectors only need a consistent row order, not real dates, so a
    day-per-row synthetic timestamp preserves ordering without claiming any
    real calendar meaning.
    """
    n = len(group)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2000-01-01", periods=n, freq="D"),
            "value": group["value"].to_numpy(dtype=float),
        }
    )


def detector_scores_for_series(
    group: pd.DataFrame, detectors: dict[str, BreakDetector] | None = None
) -> dict[str, float]:
    """Run each change-point detector on one series and summarise its output.

    For each detector, extracts: the maximum break score anywhere in the
    series, the break score at the labelled boundary (first ``period == 1``
    row), and whether the detector flagged any break in the "after" segment.
    These are cheap, complementary signals to the pure before/after stats in
    :func:`extract_series_features` — pass them alongside as extra features.
    """
    if detectors is None:
        detectors = {
            "cusum": CusumDetector(),
            "rolling_zscore": RollingZScoreDetector(),
            "pelt": PeltDetector(),
        }

    detector_input = _as_detector_input(group)
    period = group["period"].to_numpy()
    boundary_idx = int(np.argmax(period == 1)) if (period == 1).any() else len(period) - 1

    features: dict[str, float] = {}
    for name, detector in detectors.items():
        try:
            result = detector.detect(detector_input)
            score = result["break_score"].to_numpy()
            flags = result["has_structural_break"].to_numpy()
        except Exception:  # noqa: BLE001 - a detector failing on edge-case input shouldn't crash the run
            score = np.zeros(len(detector_input))
            flags = np.zeros(len(detector_input), dtype=int)

        features[f"{name}_max_score"] = _safe(float(score.max())) if len(score) else 0.0
        features[f"{name}_score_at_boundary"] = (
            _safe(float(score[boundary_idx])) if boundary_idx < len(score) else 0.0
        )
        features[f"{name}_flagged_after_boundary"] = float(flags[boundary_idx:].sum() > 0)

    return features


def build_detector_feature_matrix(
    X: pd.DataFrame, detectors: dict[str, BreakDetector] | None = None
) -> pd.DataFrame:
    """Apply :func:`detector_scores_for_series` to every series id in ``X``."""
    id_level = X.index.names[0] or "id"
    rows = {
        series_id: detector_scores_for_series(group, detectors)
        for series_id, group in X.groupby(level=0, sort=False)
    }
    matrix = pd.DataFrame.from_dict(rows, orient="index")
    matrix.index.name = id_level
    return matrix
