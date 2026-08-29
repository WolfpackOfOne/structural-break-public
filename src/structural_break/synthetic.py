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
    "make_synthetic_competition_dataset",
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


def make_synthetic_competition_dataset(
    n_series: int = 300,
    min_len: int = 100,
    max_len: int = 300,
    break_fraction: float = 0.5,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Many independent series in the *competition* (id, time)-indexed shape.

    Unlike the other generators here (one long series, per-row labels), this
    mirrors the real ADIA Lab / CrunchDAO format used by
    :mod:`structural_break.competition_data`: each series has its own id, a
    ``period`` column (``0`` before its boundary point, ``1`` after), and a
    single break/no-break label for the whole series. Break types are mixed
    (mean shift, variance shift, or none) so the harness has something
    non-trivial to learn and evaluate against before real competition data is
    dropped in.

    Returns
    -------
    (X, y)
        ``X`` — MultiIndex (``id``, ``time``) frame with ``value``, ``period``.
        ``y`` — Series indexed by ``id``, ``1`` if that series has a break.
    """
    rng = np.random.default_rng(seed)
    rows = []
    labels = {}

    for series_id in range(n_series):
        length = int(rng.integers(min_len, max_len + 1))
        boundary = int(rng.integers(max(10, length // 4), max(11, 3 * length // 4)))
        has_break = bool(rng.random() < break_fraction)

        pre_mean = float(rng.normal(0.0, 1.0))
        pre_sd = float(rng.uniform(0.3, 1.0))
        if has_break:
            break_kind = rng.choice(["mean", "variance", "both"])
            mean_shift = float(rng.normal(0.0, 1.0)) * (3.0 if break_kind != "variance" else 0.0)
            post_mean = pre_mean + mean_shift
            post_sd = pre_sd * float(rng.uniform(1.8, 3.0)) if break_kind != "mean" else pre_sd
        else:
            post_mean, post_sd = pre_mean, pre_sd

        pre_values = rng.normal(pre_mean, pre_sd, boundary)
        post_values = rng.normal(post_mean, post_sd, length - boundary)
        values = np.concatenate([pre_values, post_values])
        post_len = length - boundary
        period = np.concatenate([np.zeros(boundary, dtype=int), np.ones(post_len, dtype=int)])

        for t, (value, per) in enumerate(zip(values, period)):
            rows.append((series_id, t, float(value), int(per)))
        labels[series_id] = int(has_break)

    X = pd.DataFrame(rows, columns=["id", "time", "value", "period"]).set_index(["id", "time"])
    y = pd.Series(labels, name="has_structural_break")
    y.index.name = "id"
    return X, y
