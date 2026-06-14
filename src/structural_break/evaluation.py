"""Evaluation helpers for the structural-break baseline."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import classification_report, f1_score


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
