"""Tests for change-point evaluation utilities and synthetic generators."""

from __future__ import annotations

from structural_break.evaluation import extract_break_points, point_based_metrics
from structural_break.synthetic import (
    make_mean_shift,
    make_multiple_breaks,
    make_variance_shift,
)


def test_extract_break_points_rising_edges() -> None:
    flags = [0, 0, 1, 1, 0, 0, 1, 0]
    assert extract_break_points(flags) == [2, 6]


def test_extract_break_points_empty() -> None:
    assert extract_break_points([0, 0, 0]) == []


def test_point_based_metrics_perfect_match() -> None:
    metrics = point_based_metrics([50], [51], tolerance=5)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_point_based_metrics_outside_tolerance() -> None:
    metrics = point_based_metrics([50], [70], tolerance=5)
    assert metrics["recall"] == 0.0
    assert metrics["fp"] == 1.0
    assert metrics["fn"] == 1.0


def test_point_based_metrics_each_true_matched_once() -> None:
    # Two predictions near one true point should yield one TP and one FP.
    metrics = point_based_metrics([50], [49, 51], tolerance=5)
    assert metrics["tp"] == 1.0
    assert metrics["fp"] == 1.0


def test_point_based_metrics_no_predictions() -> None:
    metrics = point_based_metrics([50], [], tolerance=5)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_make_mean_shift_shape_and_break() -> None:
    df, breaks = make_mean_shift(n_pre=30, n_post=30)
    assert len(df) == 60
    assert breaks == [30]
    assert list(df.columns) == ["timestamp", "value", "has_structural_break"]
    assert df["has_structural_break"].tolist() == [0] * 30 + [1] * 30


def test_make_multiple_breaks_break_points() -> None:
    df, breaks = make_multiple_breaks(segment_length=20, means=(0.0, 2.0, -2.0))
    assert len(df) == 60
    assert breaks == [20, 40]


def test_make_variance_shift_is_reproducible() -> None:
    a, _ = make_variance_shift(seed=3)
    b, _ = make_variance_shift(seed=3)
    assert a["value"].tolist() == b["value"].tolist()
