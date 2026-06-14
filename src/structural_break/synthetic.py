"""Synthetic time-series generators with known structural breaks.

These helpers produce small, reproducible datasets used for tests, examples, and
method comparison. Every generator returns a tuple ``(df, break_points)`` where:

- ``df`` is a DataFrame with ``timestamp``, ``value``, and ``has_structural_break``
  columns (``has_structural_break`` marks rows in a post-break regime), and
- ``break_points`` is the list of integer row indices at which a break occurs
  (the first row of each new regime).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "make_mean_shift",
    "make_variance_shift",
    "make_multiple_breaks",
]


def _timestamps(n: int, start: str = "2023-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="D")


def _assemble(value: np.ndarray, break_points: list[int]) -> pd.DataFrame:
    """Build the standard output frame, labelling post-(first-)break rows as 1."""
    n = len(value)
    label = np.zeros(n, dtype=int)
    if break_points:
        label[break_points[0] :] = 1
    return pd.DataFrame(
        {
            "timestamp": _timestamps(n),
            "value": np.round(value, 4),
            "has_structural_break": label,
        }
    )


def make_mean_shift(
    n_pre: int = 60,
    n_post: int = 60,
    pre_mean: float = 0.0,
    post_mean: float = 3.0,
    sd: float = 0.5,
    seed: int = 0,
) -> tuple[pd.DataFrame, list[int]]:
    """A single series with one shift in mean.

    Returns ``(df, [n_pre])`` — the break occurs at row ``n_pre``.
    """
    rng = np.random.default_rng(seed)
    value = np.concatenate(
        [rng.normal(pre_mean, sd, n_pre), rng.normal(post_mean, sd, n_post)]
    )
    return _assemble(value, [n_pre]), [n_pre]


def make_variance_shift(
    n_pre: int = 60,
    n_post: int = 60,
    mean: float = 0.0,
    pre_sd: float = 0.3,
    post_sd: float = 1.5,
    seed: int = 0,
) -> tuple[pd.DataFrame, list[int]]:
    """A single series with a constant mean but a shift in volatility.

    Returns ``(df, [n_pre])``. This case is deliberately hard for mean-based
    detectors and useful for illustrating their limitations.
    """
    rng = np.random.default_rng(seed)
    value = np.concatenate(
        [rng.normal(mean, pre_sd, n_pre), rng.normal(mean, post_sd, n_post)]
    )
    return _assemble(value, [n_pre]), [n_pre]


def make_multiple_breaks(
    segment_length: int = 40,
    means: tuple[float, ...] = (0.0, 3.0, -1.0),
    sd: float = 0.5,
    seed: int = 0,
) -> tuple[pd.DataFrame, list[int]]:
    """A series with several mean regimes.

    Returns ``(df, break_points)`` where ``break_points`` lists the start index of
    every regime after the first.
    """
    rng = np.random.default_rng(seed)
    segments = [rng.normal(mean, sd, segment_length) for mean in means]
    value = np.concatenate(segments)
    break_points = [segment_length * i for i in range(1, len(means))]
    return _assemble(value, break_points), break_points
