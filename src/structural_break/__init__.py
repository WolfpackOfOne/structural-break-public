"""Structural-break detection toolkit.

A small, importable research package for detecting structural breaks (regime
shifts) in time-series data. This module re-exports the most commonly used
helpers so callers can do, for example::

    from structural_break import create_baseline_features, build_random_forest_baseline
"""

from __future__ import annotations

from .competition_data import (
    id_train_val_split,
    is_competition_format,
    load_feature_parts,
    load_labels,
    select_ids,
    series_ids,
)
from .data import load_csv, validate_required_columns
from .detectors import (
    BreakDetector,
    CusumDetector,
    PeltDetector,
    RollingZScoreDetector,
)
from .evaluation import (
    evaluate_predictions,
    extract_break_points,
    point_based_metrics,
)
from .features import (
    FEATURE_COLUMNS,
    REQUIRED_INPUT_COLUMNS,
    TARGET_COLUMN,
    create_baseline_features,
)
from .models import build_random_forest_baseline
from .predict import SUBMISSION_COLUMNS, build_submission
from .series_features import (
    STAT_FEATURE_COLUMNS,
    build_detector_feature_matrix,
    build_feature_matrix,
    detector_scores_for_series,
    extract_series_features,
)
from .synthetic import (
    make_mean_shift,
    make_multiple_breaks,
    make_synthetic_competition_dataset,
    make_variance_shift,
)

__version__ = "0.1.0"

__all__ = [
    "FEATURE_COLUMNS",
    "REQUIRED_INPUT_COLUMNS",
    "STAT_FEATURE_COLUMNS",
    "SUBMISSION_COLUMNS",
    "TARGET_COLUMN",
    "BreakDetector",
    "CusumDetector",
    "PeltDetector",
    "RollingZScoreDetector",
    "build_detector_feature_matrix",
    "build_feature_matrix",
    "build_random_forest_baseline",
    "build_submission",
    "create_baseline_features",
    "detector_scores_for_series",
    "evaluate_predictions",
    "extract_break_points",
    "extract_series_features",
    "id_train_val_split",
    "is_competition_format",
    "load_csv",
    "load_feature_parts",
    "load_labels",
    "make_mean_shift",
    "make_multiple_breaks",
    "make_synthetic_competition_dataset",
    "make_variance_shift",
    "point_based_metrics",
    "select_ids",
    "series_ids",
    "validate_required_columns",
]
