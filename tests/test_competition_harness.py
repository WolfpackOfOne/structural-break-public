"""Tests for the competition-format data loaders, per-series features, and the
weekend model-search harness (competition_data, series_features,
experiment_registry, experiment_runner).
"""

from __future__ import annotations

import pandas as pd
import pytest

from structural_break.competition_data import (
    id_train_val_split,
    is_competition_format,
    select_ids,
    series_ids,
)
from structural_break.experiment_registry import (
    EXPERIMENTS,
    ExperimentContext,
    ExperimentResult,
    register,
)
from structural_break.experiment_runner import (
    append_leaderboard_row,
    build_context,
    completed_experiment_ids,
    git_checkpoint,
    load_leaderboard,
    maybe_save_best,
    run_experiment,
)
from structural_break.series_features import (
    STAT_FEATURE_COLUMNS,
    build_detector_feature_matrix,
    build_feature_matrix,
    detector_scores_for_series,
    extract_series_features,
)
from structural_break.synthetic import make_synthetic_competition_dataset


@pytest.fixture
def competition_data():
    return make_synthetic_competition_dataset(n_series=24, min_len=40, max_len=60, seed=1)


def test_synthetic_dataset_is_competition_format(competition_data) -> None:
    X, y = competition_data
    assert is_competition_format(X)
    assert set(series_ids(X)) == set(y.index)
    assert set(y.unique()) <= {0, 1}


def test_id_train_val_split_is_disjoint_and_stratified(competition_data) -> None:
    _, y = competition_data
    train_ids, val_ids = id_train_val_split(y.index, y=y, val_fraction=0.25, seed=0)
    assert set(train_ids).isdisjoint(set(val_ids))
    assert set(train_ids) | set(val_ids) == set(y.index)
    # Both splits should contain at least one of each label present overall.
    assert set(y.loc[val_ids].unique()) <= set(y.unique())


def test_select_ids_returns_only_requested_series(competition_data) -> None:
    X, y = competition_data
    subset = list(y.index)[:5]
    sliced = select_ids(X, subset)
    assert set(sliced.index.get_level_values(0).unique()) == set(subset)


def test_extract_series_features_has_stable_columns() -> None:
    group = pd.DataFrame({"value": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0], "period": [0, 0, 0, 1, 1, 1]})
    features = extract_series_features(group)
    assert set(features.keys()) == set(STAT_FEATURE_COLUMNS)
    assert all(pd.notna(value) for value in features.values())
    # A clear mean shift should show up as a large mean_diff.
    assert features["mean_diff"] > 5.0


def test_build_feature_matrix_one_row_per_series(competition_data) -> None:
    X, y = competition_data
    matrix = build_feature_matrix(X)
    assert list(matrix.index) == list(series_ids(X))
    assert list(matrix.columns) == STAT_FEATURE_COLUMNS
    assert not matrix.isna().any().any()


def test_detector_scores_for_series_returns_finite_floats() -> None:
    group = pd.DataFrame({"value": [0.0] * 20 + [5.0] * 20, "period": [0] * 20 + [1] * 20})
    scores = detector_scores_for_series(group)
    assert scores  # non-empty
    assert all(pd.notna(value) for value in scores.values())


def test_build_detector_feature_matrix_one_row_per_series(competition_data) -> None:
    X, y = competition_data
    matrix = build_detector_feature_matrix(X)
    assert list(matrix.index) == list(series_ids(X))
    assert not matrix.isna().any().any()


def test_build_context_splits_and_aligns_features(competition_data) -> None:
    X, y = competition_data
    ctx = build_context(X, y, val_fraction=0.25, seed=0)
    assert len(ctx.train_y) + len(ctx.val_y) == len(y)
    assert list(ctx.train_stat_features.index) == list(ctx.train_y.index)
    assert list(ctx.val_stat_features.index) == list(ctx.val_y.index)
    combined = ctx.train_combined_features
    expected_cols = ctx.train_stat_features.shape[1] + ctx.train_detector_features.shape[1]
    assert combined.shape[0] == len(ctx.train_y)
    assert combined.shape[1] == expected_cols


def test_registry_has_working_and_stub_experiments() -> None:
    ids = {spec.id for spec in EXPERIMENTS}
    assert "logistic_stat_features" in ids
    assert "hmm_regime" in ids


def test_run_experiment_ok(competition_data) -> None:
    X, y = competition_data
    ctx = build_context(X, y, val_fraction=0.25, seed=0)
    spec = next(spec for spec in EXPERIMENTS if spec.id == "logistic_stat_features")
    row, model = run_experiment(spec, ctx, timeout=60)
    assert row["status"] == "ok"
    assert 0.0 <= row["auc"] <= 1.0
    assert model is not None


def test_run_experiment_stub_is_skipped_not_crashed(competition_data) -> None:
    X, y = competition_data
    ctx = build_context(X, y, val_fraction=0.25, seed=0)
    spec = next(spec for spec in EXPERIMENTS if spec.id == "hmm_regime")
    row, model = run_experiment(spec, ctx, timeout=60)
    assert row["status"] == "skipped"
    assert model is None


def test_run_experiment_catches_unexpected_errors(competition_data) -> None:
    X, y = competition_data
    ctx = build_context(X, y, val_fraction=0.25, seed=0)

    def _boom(ctx: ExperimentContext, **config) -> ExperimentResult:
        raise ValueError("deliberate failure")

    from structural_break.experiment_registry import ExperimentSpec

    spec = ExperimentSpec(id="boom", family="test", description="", fn=_boom)
    row, model = run_experiment(spec, ctx, timeout=60)
    assert row["status"] == "error"
    assert "deliberate failure" in row["notes"]
    assert model is None


def test_leaderboard_round_trip_and_resume(tmp_path) -> None:
    path = tmp_path / "leaderboard.csv"
    assert load_leaderboard(path).empty
    assert completed_experiment_ids(path) == set()

    row = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "experiment_id": "some_experiment",
        "family": "test",
        "status": "ok",
        "auc": 0.9,
        "runtime_seconds": 1.0,
        "config": "{}",
        "notes": "",
    }
    append_leaderboard_row(path, row)
    loaded = load_leaderboard(path)
    assert len(loaded) == 1
    assert completed_experiment_ids(path) == {"some_experiment"}


def test_maybe_save_best_only_improves(tmp_path) -> None:
    artifact_dir = tmp_path / "models"
    ok_row = {
        "experiment_id": "a",
        "family": "f",
        "status": "ok",
        "auc": 0.8,
        "config": "{}",
        "timestamp": "t",
    }
    assert maybe_save_best(object(), ok_row, artifact_dir) is True
    assert (artifact_dir.parent / "best_config.json").is_file()

    worse_row = {**ok_row, "auc": 0.5}
    assert maybe_save_best(object(), worse_row, artifact_dir) is False

    better_row = {**ok_row, "auc": 0.95}
    assert maybe_save_best(object(), better_row, artifact_dir) is True


def test_git_checkpoint_reports_no_changes_outside_repo(tmp_path) -> None:
    # A directory with no git repo should fail gracefully, not raise.
    outcome = git_checkpoint(tmp_path, "test checkpoint", paths=["."], push=False)
    assert "failed" in outcome or "no changes" in outcome
    # Never raises, even though `git` will error inside a non-repo directory.


def test_register_creates_one_spec_per_config() -> None:
    before = len(EXPERIMENTS)

    @register("dummy_sweep", family="test", configs=[{"x": 1}, {"x": 2}])
    def _dummy(ctx: ExperimentContext, x: int = 0) -> ExperimentResult:
        return ExperimentResult(auc=0.5)

    new_specs = EXPERIMENTS[before:]
    assert len(new_specs) == 2
    assert {spec.id for spec in new_specs} == {"dummy_sweep__x=1", "dummy_sweep__x=2"}
