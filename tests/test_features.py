"""Tests for baseline feature engineering."""

from __future__ import annotations

import pandas as pd
import pytest

from structural_break.features import (
    FEATURE_COLUMNS,
    create_baseline_features,
)

ENGINEERED_COLUMNS = [
    "rolling_mean_3",
    "rolling_std_3",
    "lag_1",
    "lag_2",
    "diff_1",
    "diff_2",
]


def test_returns_expected_engineered_columns(mean_shift_df: pd.DataFrame) -> None:
    result = create_baseline_features(mean_shift_df)
    for column in ENGINEERED_COLUMNS:
        assert column in result.columns
    # All modelling features should be present.
    assert set(FEATURE_COLUMNS).issubset(result.columns)


def test_preserves_row_count(mean_shift_df: pd.DataFrame) -> None:
    result = create_baseline_features(mean_shift_df)
    assert len(result) == len(mean_shift_df)


def test_no_missing_values_after_fill(mean_shift_df: pd.DataFrame) -> None:
    result = create_baseline_features(mean_shift_df)
    assert not result[FEATURE_COLUMNS].isna().any().any()


def test_does_not_mutate_input(mean_shift_df: pd.DataFrame) -> None:
    before = mean_shift_df.copy()
    create_baseline_features(mean_shift_df)
    pd.testing.assert_frame_equal(mean_shift_df, before)


def test_sorts_by_timestamp() -> None:
    df = pd.DataFrame(
        {
            "timestamp": ["2023-01-03", "2023-01-01", "2023-01-02"],
            "value": [3.0, 1.0, 2.0],
        }
    )
    result = create_baseline_features(df)
    assert list(result["value"]) == [1.0, 2.0, 3.0]
    assert result["timestamp"].is_monotonic_increasing


def test_raises_when_timestamp_missing() -> None:
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="timestamp"):
        create_baseline_features(df)


def test_raises_when_value_missing() -> None:
    df = pd.DataFrame({"timestamp": ["2023-01-01", "2023-01-02"]})
    with pytest.raises(ValueError, match="value"):
        create_baseline_features(df)
