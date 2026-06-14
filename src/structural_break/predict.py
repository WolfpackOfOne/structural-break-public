"""Prediction and submission helpers for the structural-break baseline."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.pipeline import Pipeline

from .features import FEATURE_COLUMNS, create_baseline_features

#: Columns expected in a submission file.
SUBMISSION_COLUMNS: list[str] = ["timestamp", "has_structural_break"]


def build_submission(
    model: Pipeline,
    test_df: pd.DataFrame,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Generate a submission-style prediction table for ``test_df``.

    Features are engineered from ``test_df`` with
    :func:`~structural_break.features.create_baseline_features`, the model predicts a
    binary label per row, and the result is returned with the columns in
    :data:`SUBMISSION_COLUMNS`.

    Parameters
    ----------
    model:
        A fitted pipeline/estimator exposing ``predict``.
    test_df:
        Input series with at least ``timestamp`` and ``value`` columns.
    feature_columns:
        Columns to feed into the model. Defaults to :data:`FEATURE_COLUMNS`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with ``timestamp`` and ``has_structural_break`` columns. The row
        count equals that of ``test_df``.
    """
    featured = create_baseline_features(test_df)
    predictions = model.predict(featured[list(feature_columns)])

    submission = featured[["timestamp"]].copy()
    submission["has_structural_break"] = predictions
    return submission
