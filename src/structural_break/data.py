"""Data loading and validation helpers for the structural-break baseline."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a :class:`pandas.DataFrame`.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        The loaded data.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist, with a message pointing at the missing file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Data file not found: '{path}'. "
            "Place the dataset there, or pass a different path via the CLI."
        )
    return pd.read_csv(path)


def validate_required_columns(
    df: pd.DataFrame, required: Iterable[str], dataset_name: str
) -> None:
    """Ensure ``df`` contains every column in ``required``.

    Parameters
    ----------
    df:
        The DataFrame to validate.
    required:
        Column names that must be present.
    dataset_name:
        Human-readable name used in the error message (for example ``"Training data"``).

    Raises
    ------
    ValueError
        If any required column is missing, listing the missing names.
    """
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {', '.join(sorted(missing))}. "
            f"Found columns: {', '.join(map(str, df.columns))}."
        )
