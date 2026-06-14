"""Structural-break / change-point detectors with a shared interface.

Every detector implements :class:`BreakDetector` and exposes a single ``detect``
method:

    detector.detect(df) -> DataFrame[timestamp, break_score, has_structural_break, method]

Input contract
--------------
The input is treated as a *single ordered series*. It must contain ``timestamp``
and ``value`` columns; rows are sorted by ``timestamp`` before detection.

Output contract
---------------
- ``timestamp`` — copied from the (sorted) input.
- ``break_score`` — a continuous, non-negative score; larger means stronger
  evidence of a break at that row (semantics differ per method, see each class).
- ``has_structural_break`` — a binary flag (0/1) marking the row(s) where the
  detector locates a break. The row count always equals the input.
- ``method`` — the detector's name.

Row-level vs. point-level
-------------------------
Change-point methods naturally produce *break locations*, not a label per row.
Here, ``has_structural_break`` marks the located break row(s); contiguous flagged
rows are treated as a single detected break by :func:`extract_break_points`, which
the evaluation utilities use for point-based scoring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .data import validate_required_columns

__all__ = [
    "BreakDetector",
    "CusumDetector",
    "RollingZScoreDetector",
    "PeltDetector",
    "RESULT_COLUMNS",
]

#: Columns produced by every detector's ``detect`` method.
RESULT_COLUMNS: list[str] = [
    "timestamp",
    "break_score",
    "has_structural_break",
    "method",
]

_REQUIRED_INPUT: set[str] = {"timestamp", "value"}


class BreakDetector(ABC):
    """Abstract base class for structural-break detectors."""

    #: Short identifier used in the ``method`` output column.
    name: str = "base"

    @abstractmethod
    def _score(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Score a 1-D array of values.

        Returns a tuple ``(break_score, flags)`` of arrays the same length as
        ``values``: a continuous score and a 0/1 break flag.
        """

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the detector on ``df`` and return the standard result frame.

        Raises
        ------
        ValueError
            If ``timestamp`` or ``value`` is missing.
        """
        validate_required_columns(df, _REQUIRED_INPUT, f"{self.name} input")

        ordered = df.copy()
        ordered["timestamp"] = pd.to_datetime(ordered["timestamp"])
        ordered = ordered.sort_values("timestamp", kind="stable").reset_index(drop=True)

        values = ordered["value"].to_numpy(dtype=float)
        score, flags = self._score(values)

        return pd.DataFrame(
            {
                "timestamp": ordered["timestamp"],
                "break_score": np.asarray(score, dtype=float),
                "has_structural_break": np.asarray(flags, dtype=int),
                "method": self.name,
            }
        )


class CusumDetector(BreakDetector):
    """CUSUM mean-shift detector (single dominant change point).

    Uses the cumulative sum of deviations from the global mean,
    ``S_k = sum_{i<=k}(x_i - mean)``. The estimated change point is ``argmax|S_k|``;
    a break is reported there when the normalised statistic
    ``max|S_k| / (sigma * sqrt(n))`` exceeds ``threshold``.

    ``break_score`` is the normalised ``|S_k|`` at each row (peaks at the change
    point). Only the single most prominent mean shift is located; for several
    breaks, prefer :class:`PeltDetector`.

    Parameters
    ----------
    threshold:
        Significance threshold on the normalised CUSUM statistic. Higher values are
        more conservative. Default ``1.1``.
    """

    name = "cusum"

    def __init__(self, threshold: float = 1.1) -> None:
        self.threshold = threshold

    def _score(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(values)
        flags = np.zeros(n, dtype=int)
        if n == 0:
            return np.zeros(0), flags

        sigma = float(values.std(ddof=0)) or 1.0
        cumulative = np.cumsum(values - values.mean())
        score = np.abs(cumulative) / (sigma * np.sqrt(n))

        change_point = int(np.argmax(np.abs(cumulative)))
        if score[change_point] >= self.threshold:
            flags[change_point] = 1
        return score, flags


class RollingZScoreDetector(BreakDetector):
    """Rolling z-score / local-deviation detector.

    Flags rows whose value deviates from the trailing rolling mean by more than
    ``threshold`` standard deviations. This is a transparent local-anomaly detector:
    it highlights the *transition* into a new regime (and isolated outliers) rather
    than labelling a whole regime.

    ``break_score`` is the absolute rolling z-score per row.

    Parameters
    ----------
    window:
        Trailing window length for the rolling mean/std. Default ``20``.
    threshold:
        Absolute z-score above which a row is flagged. Default ``3.0``.
    """

    name = "rolling_zscore"

    def __init__(self, window: int = 20, threshold: float = 3.0) -> None:
        if window < 2:
            raise ValueError("window must be at least 2.")
        self.window = window
        self.threshold = threshold

    def _score(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        series = pd.Series(values)
        rolling = series.rolling(self.window, min_periods=2)
        mean = rolling.mean()
        std = rolling.std(ddof=0)

        # Avoid division by zero on flat windows; such rows get a score of 0.
        std = std.replace(0.0, np.nan)
        z = ((series - mean) / std).abs().fillna(0.0)

        score = z.to_numpy()
        flags = (score > self.threshold).astype(int)
        return score, flags


class PeltDetector(BreakDetector):
    """PELT change-point detector backed by the ``ruptures`` library.

    Detects one or more change points by minimising a segmentation cost plus a
    penalty (the PELT algorithm). All located change points are flagged.

    ``break_score`` for each row is the absolute difference between its segment mean
    and the global mean (a continuous "how different is this segment" signal).

    Parameters
    ----------
    model:
        ``ruptures`` cost model. ``"l2"`` (default) detects mean shifts.
    penalty:
        Penalty controlling the number of change points. If ``None``, a BIC-style
        default ``2 * log(n) * var(values)`` is used.
    min_size:
        Minimum segment length. Default ``2``.
    """

    name = "pelt"

    def __init__(
        self,
        model: str = "l2",
        penalty: float | None = None,
        min_size: int = 2,
    ) -> None:
        self.model = model
        self.penalty = penalty
        self.min_size = min_size

    def _score(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        try:
            import ruptures
        except ImportError as error:  # pragma: no cover - exercised only without dep
            raise ImportError(
                "PeltDetector requires the 'ruptures' package. "
                "Install it with `pip install ruptures`."
            ) from error

        n = len(values)
        flags = np.zeros(n, dtype=int)
        if n < 2 * self.min_size:
            return np.zeros(n), flags

        signal = values.reshape(-1, 1)
        penalty = self.penalty
        if penalty is None:
            penalty = 2.0 * np.log(n) * max(float(values.var()), 1e-8)

        algo = ruptures.Pelt(model=self.model, min_size=self.min_size).fit(signal)
        # ruptures returns segment end indices; the last one is n (not a break).
        boundaries = algo.predict(pen=penalty)
        change_points = [b for b in boundaries if 0 < b < n]
        for cp in change_points:
            flags[cp] = 1

        # Segment-mean deviation as a continuous score.
        score = np.zeros(n)
        global_mean = float(values.mean())
        starts = [0, *change_points]
        ends = [*change_points, n]
        for start, end in zip(starts, ends):
            segment_mean = float(values[start:end].mean())
            score[start:end] = abs(segment_mean - global_mean)
        return score, flags
