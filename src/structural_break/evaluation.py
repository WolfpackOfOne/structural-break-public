"""Evaluation helpers for structural-break detection.

Two evaluation styles live here:

- :func:`evaluate_predictions` — per-row classification metrics for the ML baseline.
- :func:`point_based_metrics` — windowed precision/recall/F1 for change-point
  detectors, where a predicted break counts as correct if it falls within a
  tolerance of a true break.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import classification_report, f1_score


def extract_break_points(flags: Sequence[int]) -> list[int]:
    """Convert a per-row 0/1 flag array into a list of break-point indices.

    Each contiguous run of ``1``s counts as a single detected break, located at the
    index where the run starts (a rising edge from 0 to 1).
    """
    points: list[int] = []
    previous = 0
    for index, flag in enumerate(flags):
        value = int(flag)
        if value and not previous:
            points.append(index)
        previous = value
    return points


def point_based_metrics(
    true_points: Sequence[int],
    predicted_points: Sequence[int],
    tolerance: int = 5,
) -> dict[str, float]:
    """Windowed precision/recall/F1 for change-point locations.

    A predicted point is a true positive if it lies within ``tolerance`` indices of
    a true point; each true point can be matched at most once (greedy nearest match).

    Parameters
    ----------
    true_points:
        Ground-truth break indices.
    predicted_points:
        Detected break indices.
    tolerance:
        Maximum index distance for a predicted point to count as correct.

    Returns
    -------
    dict
        ``precision``, ``recall``, ``f1``, plus the raw ``tp``, ``fp``, ``fn`` counts.
    """
    true_sorted = sorted(true_points)
    matched: set[int] = set()
    true_positives = 0

    for predicted in sorted(predicted_points):
        best_index: int | None = None
        best_distance: int | None = None
        for j, true_point in enumerate(true_sorted):
            if j in matched:
                continue
            distance = abs(predicted - true_point)
            if distance <= tolerance and (best_distance is None or distance < best_distance):
                best_index, best_distance = j, distance
        if best_index is not None:
            matched.add(best_index)
            true_positives += 1

    false_positives = len(list(predicted_points)) - true_positives
    false_negatives = len(true_sorted) - true_positives

    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, true_positives + false_negatives)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(true_positives),
        "fp": float(false_positives),
        "fn": float(false_negatives),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator``, or 0.0 when the denominator is 0."""
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def evaluate_predictions(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> dict[str, object]:
    """Compute baseline classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_pred:
        Predicted binary labels.

    Returns
    -------
    dict
        A dictionary with:

        - ``"f1"``: the binary F1 score (``float``).
        - ``"report"``: a formatted ``classification_report`` string.
    """
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }
