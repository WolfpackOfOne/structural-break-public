"""Tests for the baseline model builder."""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from structural_break.features import FEATURE_COLUMNS, TARGET_COLUMN, create_baseline_features
from structural_break.models import build_random_forest_baseline


def test_returns_pipeline() -> None:
    model = build_random_forest_baseline()
    assert isinstance(model, Pipeline)


def test_is_reproducible_via_random_state() -> None:
    # Two builders with the same seed should expose the same configured seed.
    a = build_random_forest_baseline(random_state=7)
    b = build_random_forest_baseline(random_state=7)
    assert a.named_steps["classifier"].random_state == b.named_steps["classifier"].random_state == 7


def test_fit_and_predict_length(mean_shift_df: pd.DataFrame) -> None:
    features = create_baseline_features(mean_shift_df)
    model = build_random_forest_baseline()
    model.fit(features[FEATURE_COLUMNS], features[TARGET_COLUMN])

    predictions = model.predict(features[FEATURE_COLUMNS])
    assert len(predictions) == len(features)


def test_fits_on_tiny_dataset() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=6, freq="D"),
            "value": [0.0, 0.1, 0.0, 2.0, 2.1, 1.9],
            "has_structural_break": [0, 0, 0, 1, 1, 1],
        }
    )
    features = create_baseline_features(df)
    model = build_random_forest_baseline()
    model.fit(features[FEATURE_COLUMNS], features[TARGET_COLUMN])
    assert len(model.predict(features[FEATURE_COLUMNS])) == len(df)
