"""Tests for the structural-break detectors."""

from __future__ import annotations

import pandas as pd
import pytest

from structural_break.detectors import (
    RESULT_COLUMNS,
    BreakDetector,
    CusumDetector,
    PeltDetector,
    RollingZScoreDetector,
)
from structural_break.evaluation import extract_break_points, point_based_metrics
from structural_break.synthetic import make_mean_shift, make_multiple_breaks

ALL_DETECTORS = [CusumDetector, RollingZScoreDetector, PeltDetector]


@pytest.fixture
def mean_shift():
    return make_mean_shift(n_pre=60, n_post=60, seed=0)


@pytest.mark.parametrize("detector_cls", ALL_DETECTORS)
def test_returns_result_columns(detector_cls, mean_shift) -> None:
    df, _ = mean_shift
    result = detector_cls().detect(df)
    assert list(result.columns) == RESULT_COLUMNS


@pytest.mark.parametrize("detector_cls", ALL_DETECTORS)
def test_preserves_row_count(detector_cls, mean_shift) -> None:
    df, _ = mean_shift
    result = detector_cls().detect(df)
    assert len(result) == len(df)


@pytest.mark.parametrize("detector_cls", ALL_DETECTORS)
def test_flags_are_binary(detector_cls, mean_shift) -> None:
    df, _ = mean_shift
    result = detector_cls().detect(df)
    assert set(result["has_structural_break"].unique()).issubset({0, 1})


@pytest.mark.parametrize("detector_cls", ALL_DETECTORS)
def test_method_name_is_recorded(detector_cls, mean_shift) -> None:
    df, _ = mean_shift
    result = detector_cls().detect(df)
    assert (result["method"] == detector_cls.name).all()


@pytest.mark.parametrize("detector_cls", ALL_DETECTORS)
def test_raises_when_value_missing(detector_cls) -> None:
    df = pd.DataFrame({"timestamp": pd.date_range("2023-01-01", periods=5, freq="D")})
    with pytest.raises(ValueError, match="value"):
        detector_cls().detect(df)


@pytest.mark.parametrize("detector_cls", ALL_DETECTORS)
def test_raises_when_timestamp_missing(detector_cls) -> None:
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    with pytest.raises(ValueError, match="timestamp"):
        detector_cls().detect(df)


@pytest.mark.parametrize("detector_cls", ALL_DETECTORS)
def test_works_on_small_series(detector_cls) -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=8, freq="D"),
            "value": [0.0, 0.1, -0.1, 0.0, 3.0, 3.1, 2.9, 3.0],
        }
    )
    result = detector_cls().detect(df)
    assert len(result) == len(df)


@pytest.mark.parametrize("detector_cls", [CusumDetector, PeltDetector])
def test_detects_mean_shift_near_truth(detector_cls, mean_shift) -> None:
    df, true_points = mean_shift
    result = detector_cls().detect(df)
    predicted = extract_break_points(result["has_structural_break"].to_numpy())
    metrics = point_based_metrics(true_points, predicted, tolerance=5)
    assert metrics["recall"] == 1.0


def test_pelt_finds_multiple_breaks() -> None:
    df, true_points = make_multiple_breaks(segment_length=40, seed=0)
    result = PeltDetector().detect(df)
    predicted = extract_break_points(result["has_structural_break"].to_numpy())
    metrics = point_based_metrics(true_points, predicted, tolerance=5)
    assert metrics["recall"] == 1.0


def test_detectors_subclass_base(mean_shift) -> None:
    df, _ = mean_shift
    for cls in ALL_DETECTORS:
        assert issubclass(cls, BreakDetector)
        assert isinstance(cls().detect(df), pd.DataFrame)
