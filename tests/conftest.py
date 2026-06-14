"""Shared pytest fixtures built on tiny synthetic data (no competition data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_mean_shift(
    n_pre: int = 50,
    n_post: int = 50,
    pre_mean: float = 0.0,
    post_mean: float = 2.0,
    sd: float = 0.3,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a single ordered series with one injected mean shift."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_pre + n_post, freq="D")
    value = np.concatenate(
        [rng.normal(pre_mean, sd, n_pre), rng.normal(post_mean, sd, n_post)]
    )
    label = np.concatenate([np.zeros(n_pre, dtype=int), np.ones(n_post, dtype=int)])
    return pd.DataFrame(
        {"timestamp": dates, "value": value, "has_structural_break": label}
    )


@pytest.fixture
def mean_shift_df() -> pd.DataFrame:
    """A labelled training-style series with a mean shift at index 50."""
    return _make_mean_shift()


@pytest.fixture
def break_index() -> int:
    """The index at which the mean shift in ``mean_shift_df`` occurs."""
    return 50


@pytest.fixture
def unlabelled_df(mean_shift_df: pd.DataFrame) -> pd.DataFrame:
    """A test-style series without the label column."""
    return mean_shift_df[["timestamp", "value"]].copy()
