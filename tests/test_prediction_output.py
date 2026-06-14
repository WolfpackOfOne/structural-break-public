"""Tests for the submission/prediction output schema."""

from __future__ import annotations

import pandas as pd

from structural_break.features import FEATURE_COLUMNS, TARGET_COLUMN, create_baseline_features
from structural_break.models import build_random_forest_baseline
from structural_break.predict import SUBMISSION_COLUMNS, build_submission


def _fit_model(train_df: pd.DataFrame):
    features = create_baseline_features(train_df)
    model = build_random_forest_baseline()
    model.fit(features[FEATURE_COLUMNS], features[TARGET_COLUMN])
    return model


def test_output_is_dataframe(mean_shift_df: pd.DataFrame, unlabelled_df: pd.DataFrame) -> None:
    model = _fit_model(mean_shift_df)
    submission = build_submission(model, unlabelled_df)
    assert isinstance(submission, pd.DataFrame)


def test_output_has_expected_columns(
    mean_shift_df: pd.DataFrame, unlabelled_df: pd.DataFrame
) -> None:
    model = _fit_model(mean_shift_df)
    submission = build_submission(model, unlabelled_df)
    assert list(submission.columns) == SUBMISSION_COLUMNS
    assert "timestamp" in submission.columns
    assert "has_structural_break" in submission.columns


def test_output_row_count_matches_input(
    mean_shift_df: pd.DataFrame, unlabelled_df: pd.DataFrame
) -> None:
    model = _fit_model(mean_shift_df)
    submission = build_submission(model, unlabelled_df)
    assert len(submission) == len(unlabelled_df)


def test_predictions_are_binary(
    mean_shift_df: pd.DataFrame, unlabelled_df: pd.DataFrame
) -> None:
    model = _fit_model(mean_shift_df)
    submission = build_submission(model, unlabelled_df)
    assert set(submission["has_structural_break"].unique()).issubset({0, 1})
