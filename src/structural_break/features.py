"""Feature engineering for the baseline structural-break model.

The baseline treats the input as a *single ordered time series*. Rows are sorted
by ``timestamp`` and lag/rolling/difference features are computed across the whole
series. If you ever feed in multiple independent series concatenated together,
group by the series identifier *before* calling these helpers, otherwise features
will leak across series boundaries.
"""

from __future__ import annotations

import pandas as pd

from .data import validate_required_columns

#: Numeric columns consumed by the baseline model, in a stable order.
FEATURE_COLUMNS: list[str] = [
    "value",
    "rolling_mean_3",
    "rolling_std_3",
    "lag_1",
    "lag_2",
    "diff_1",
    "diff_2",
]

#: Binary target column expected in labelled training data.
TARGET_COLUMN: str = "has_structural_break"

#: Columns every input series must provide.
REQUIRED_INPUT_COLUMNS: set[str] = {"timestamp", "value"}


def create_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag, rolling, and difference features for the baseline model.

    The input must contain ``timestamp`` and ``value`` columns. Rows are sorted by
    ``timestamp`` (ascending) before features are computed, and missing values
    introduced by the rolling/lag/difference windows at the start of the series are
    back-filled so every row remains usable.

    Parameters
    ----------
    df:
        A DataFrame with at least ``timestamp`` and ``value`` columns. Any label or
        identifier columns are passed through unchanged.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` (sorted by timestamp, index reset) with the engineered
        feature columns in :data:`FEATURE_COLUMNS` added. The row count is preserved.

    Raises
    ------
    ValueError
        If ``timestamp`` or ``value`` is missing.
    """
    validate_required_columns(df, REQUIRED_INPUT_COLUMNS, "Input series")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    df["rolling_mean_3"] = df["value"].rolling(window=3).mean()
    df["rolling_std_3"] = df["value"].rolling(window=3).std()
    df["lag_1"] = df["value"].shift(1)
    df["lag_2"] = df["value"].shift(2)
    df["diff_1"] = df["value"].diff(1)
    df["diff_2"] = df["value"].diff(2)

    # Back-fill the NaNs produced by the leading windows so no row is dropped.
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].bfill()

    return df
