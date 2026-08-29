"""Loaders for the real ADIA Lab / CrunchDAO Structural Break Challenge data.

The official competition data has a different shape from the single-series CSV
samples bundled in ``data/``:

- ``X`` is a :class:`pandas.DataFrame` with a two-level ``MultiIndex`` — ``id``
  (one value per independent time series) and a per-row time index — and two
  columns: ``value`` (the observation) and ``period`` (``0`` before the series'
  boundary point, ``1`` after it).
- ``y`` is indexed by ``id`` and holds a single binary label per series: whether
  a structural break occurred at that series' boundary point.
- Because of GitHub's 100 MB file limit, the official ``X_train`` is typically
  distributed as several Parquet parts (``X_train.part1.parquet``,
  ``X_train.part2.parquet``, ...); :func:`load_feature_parts` concatenates them.

This module is deliberately independent from :mod:`structural_break.data` (which
handles the single-series CSV baseline) — the two formats should not be mixed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "REQUIRED_X_COLUMNS",
    "load_feature_parts",
    "load_labels",
    "is_competition_format",
    "series_ids",
    "id_train_val_split",
    "select_ids",
]

#: Columns every competition-format X frame must provide.
REQUIRED_X_COLUMNS: set[str] = {"value", "period"}


def load_feature_parts(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Load and concatenate one or more Parquet feature files.

    Parameters
    ----------
    paths:
        One or more Parquet file paths (e.g. the ``X_train.partN.parquet``
        split files, or a single ``X_test.parquet``). Order does not matter —
        rows are concatenated and the result is not re-sorted, since the
        MultiIndex already carries series/time ordering.

    Returns
    -------
    pandas.DataFrame
        The concatenated frame. Raises ``FileNotFoundError`` if any path is
        missing, and ``ValueError`` if the result doesn't look like the
        competition format (see :func:`is_competition_format`).
    """
    frames = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Competition data file not found: '{path}'. "
                "Place the official X_train/X_test Parquet file(s) there."
            )
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, axis=0) if len(frames) > 1 else frames[0]

    if not is_competition_format(df):
        raise ValueError(
            "Loaded frame does not look like the competition format: expected a "
            "2-level MultiIndex (series id, time) and columns "
            f"{sorted(REQUIRED_X_COLUMNS)}, got index nlevels="
            f"{df.index.nlevels} and columns={list(df.columns)}."
        )
    return df


def load_labels(path: str | Path) -> pd.Series:
    """Load the per-series ``y`` labels (Parquet or CSV), indexed by series id."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Label file not found: '{path}'. Place the official y_train file there."
        )
    if path.suffix == ".parquet":
        loaded = pd.read_parquet(path)
    else:
        loaded = pd.read_csv(path, index_col=0)
    if isinstance(loaded, pd.DataFrame):
        loaded = loaded.iloc[:, 0]
    return loaded.astype(int).rename("has_structural_break")


def is_competition_format(df: pd.DataFrame) -> bool:
    """True if ``df`` looks like the official (id, time)-indexed X frame."""
    return df.index.nlevels == 2 and REQUIRED_X_COLUMNS.issubset(set(df.columns))


def series_ids(X: pd.DataFrame) -> np.ndarray:
    """Return the unique series ids (index level 0), in first-seen order."""
    return X.index.get_level_values(0).unique().to_numpy()


def id_train_val_split(
    ids: Iterable,
    y: pd.Series | None = None,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split series ids into train/validation sets.

    This is a *fast, simple* split for ranking experiments against each other —
    not a rigorous nested cross-validation. Stratifies by label when ``y`` is
    given, so both splits keep a similar break/no-break ratio.

    Returns
    -------
    (train_ids, val_ids)
    """
    ids = np.asarray(list(ids))
    rng = np.random.default_rng(seed)

    if y is None:
        shuffled = ids.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_fraction))
        return shuffled[n_val:], shuffled[:n_val]

    labels = y.loc[ids].to_numpy()
    train_parts, val_parts = [], []
    for label_value in np.unique(labels):
        group_ids = ids[labels == label_value].copy()
        rng.shuffle(group_ids)
        n_val = max(1, int(len(group_ids) * val_fraction))
        val_parts.append(group_ids[:n_val])
        train_parts.append(group_ids[n_val:])
    return np.concatenate(train_parts), np.concatenate(val_parts)


def select_ids(X: pd.DataFrame, ids: Iterable) -> pd.DataFrame:
    """Slice a competition-format frame down to a subset of series ids."""
    id_level = X.index.get_level_values(0)
    return X[id_level.isin(list(ids))]
