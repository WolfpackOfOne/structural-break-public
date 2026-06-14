"""Plotting helpers for structural-break detection.

These are optional conveniences: ``matplotlib`` is imported lazily so the core
package does not require it for tests or the baseline workflow.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .evaluation import extract_break_points

__all__ = ["plot_detection"]


def plot_detection(
    df: pd.DataFrame,
    result: pd.DataFrame,
    true_break_points: Sequence[int] | None = None,
    title: str | None = None,
    output_path: str | Path | None = None,
):
    """Plot a series with true and predicted break points.

    Parameters
    ----------
    df:
        Input series with ``timestamp`` and ``value`` columns.
    result:
        A detector result frame (see :mod:`structural_break.detectors`) with a
        ``has_structural_break`` column aligned to ``df``.
    true_break_points:
        Optional ground-truth break indices, drawn as solid reference lines.
    title:
        Optional plot title. Defaults to the detector's method name if available.
    output_path:
        If given, the figure is saved there (parent directories are created).

    Returns
    -------
    matplotlib.axes.Axes
        The axis the series was drawn on.

    Raises
    ------
    ImportError
        If ``matplotlib`` is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - exercised only without dep
        raise ImportError(
            "plot_detection requires matplotlib. Install it with `pip install matplotlib`."
        ) from error

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["timestamp"], df["value"], color="#1f77b4", lw=1.2, label="value")

    timestamps = pd.to_datetime(df["timestamp"]).reset_index(drop=True)

    if true_break_points:
        for i, point in enumerate(true_break_points):
            ax.axvline(
                timestamps.iloc[point],
                color="#2ca02c",
                ls="-",
                lw=1.5,
                alpha=0.7,
                label="true break" if i == 0 else None,
            )

    predicted = extract_break_points(result["has_structural_break"].to_numpy())
    for i, point in enumerate(predicted):
        ax.axvline(
            timestamps.iloc[point],
            color="#d62728",
            ls="--",
            lw=1.5,
            alpha=0.8,
            label="predicted break" if i == 0 else None,
        )

    method = result["method"].iloc[0] if "method" in result.columns and len(result) else ""
    ax.set_title(title or (f"Detected breaks — {method}" if method else "Detected breaks"))
    ax.set_xlabel("timestamp")
    ax.set_ylabel("value")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120)

    return ax
