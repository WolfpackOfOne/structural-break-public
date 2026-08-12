"""Tests for the Real-Time Edition streaming detector (structural_break.realtime)."""

from __future__ import annotations

import numpy as np
import pytest

from structural_break.realtime import (
    DetectorParams,
    StreamingBreakDetector,
    infer,
    make_realtime_dataset,
    make_realtime_series,
    time_stratified_auc,
    train,
)

_DEFAULT_PARAMS = DetectorParams()


def _run_series(x_historical, x_online, params: DetectorParams = _DEFAULT_PARAMS):
    detector = StreamingBreakDetector(x_historical, params)
    return [detector.update(x) for x in x_online]


def test_scores_are_bounded() -> None:
    x_h, x_o, _ = make_realtime_series(
        n_historical=500, n_online=200, tau=100, mean_shift=3.0, seed=1
    )
    scores = _run_series(x_h, x_o)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_mean_shift_raises_score_after_break() -> None:
    x_h, x_o, tau = make_realtime_series(
        n_historical=1000, n_online=300, tau=150, mean_shift=4.0, seed=2
    )
    scores = np.array(_run_series(x_h, x_o))
    pre_break = scores[:tau]
    post_break = scores[tau + 20 :]  # allow a short detection lag
    assert post_break.mean() > pre_break.mean()
    assert post_break.mean() > 0.5


def test_no_break_series_stays_mostly_low() -> None:
    x_h, x_o, tau = make_realtime_series(n_historical=1000, n_online=300, tau=None, seed=3)
    assert tau is None
    scores = np.array(_run_series(x_h, x_o))
    assert scores.mean() < 0.3


def test_variance_shift_is_detected() -> None:
    x_h, x_o, tau = make_realtime_series(
        n_historical=1000, n_online=300, tau=150, mean_shift=0.0, sd_ratio=3.0, seed=4
    )
    scores = np.array(_run_series(x_h, x_o))
    pre_break = scores[:tau]
    post_break = scores[tau + 20 :]
    assert post_break.mean() > pre_break.mean()


def test_handles_degenerate_historical_segment() -> None:
    # A single-point (or empty) historical segment must not raise or divide by zero.
    detector = StreamingBreakDetector([5.0])
    scores = [detector.update(x) for x in [5.0, 5.0, 100.0]]
    assert all(0.0 <= s <= 1.0 for s in scores)

    detector_empty = StreamingBreakDetector([])
    scores_empty = [detector_empty.update(x) for x in [0.0, 1.0]]
    assert all(0.0 <= s <= 1.0 for s in scores_empty)


def test_train_infer_round_trip(tmp_path) -> None:
    dataset = make_realtime_dataset(n_series=4, n_historical=200, n_online=50, seed=5)
    train_tuples = [(i, x_h, x_o, tau) for i, x_h, x_o, tau in dataset]

    train(train_tuples, str(tmp_path))
    assert (tmp_path / "model.joblib").exists()

    test_iterable = ((x_h, iter(x_o)) for _, x_h, x_o, _ in dataset)
    gen = infer(test_iterable, str(tmp_path))

    ready = next(gen)
    assert ready is None

    all_scores = list(gen)
    expected_count = sum(len(x_o) for _, _, x_o, _ in dataset)
    assert len(all_scores) == expected_count
    assert all(0.0 <= s <= 1.0 for s in all_scores)


def test_time_stratified_auc_perfect_separation() -> None:
    # Three series, online length 4: A breaks at t=2, B breaks at t=1, C never
    # breaks. Scores rank correctly within every step that has both classes.
    scores = [
        [0.10, 0.15, 0.60, 0.80],  # A
        [0.05, 0.30, 0.55, 0.70],  # B
        [0.02, 0.10, 0.20, 0.40],  # C
    ]
    labels = [
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 0, 0],
    ]
    assert time_stratified_auc(scores, labels) == pytest.approx(1.0)


def test_time_stratified_auc_ignores_pure_steps() -> None:
    # Step 0 has only negatives across both series; it must not raise and
    # must not contribute to the weighted average.
    scores = [[0.1, 0.9], [0.2, 0.2]]
    labels = [[0, 1], [0, 0]]
    result = time_stratified_auc(scores, labels)
    assert 0.0 <= result <= 1.0


def test_time_stratified_auc_empty_returns_half() -> None:
    assert time_stratified_auc([], []) == pytest.approx(0.5)


def test_detector_beats_random_on_synthetic_mix() -> None:
    dataset = make_realtime_dataset(n_series=30, n_historical=800, n_online=250, seed=7)

    all_scores = []
    all_labels = []
    for _, x_h, x_o, tau in dataset:
        scores = _run_series(x_h, x_o)
        labels = (
            [1 if t >= tau else 0 for t in range(len(x_o))]
            if tau is not None
            else [0] * len(x_o)
        )
        all_scores.append(scores)
        all_labels.append(labels)

    ts_auc = time_stratified_auc(all_scores, all_labels)
    assert ts_auc > 0.6
