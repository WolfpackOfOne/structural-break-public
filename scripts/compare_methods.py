#!/usr/bin/env python
"""Compare structural-break detectors on a synthetic dataset with known breaks.

Generates a labelled synthetic series, runs the change-point detectors (and,
optionally, the supervised ML baseline trained on a separate series), scores each
method with windowed point-based precision/recall/F1, and writes a comparison CSV.

Example
-------
::

    python scripts/compare_methods.py \\
        --dataset mean_shift \\
        --output outputs/method_comparison.csv \\
        --figure outputs/figures/example_break_detection.png

All results are computed on *synthetic* data with known break points — no
competition data is required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from structural_break import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    CusumDetector,
    PeltDetector,
    RollingZScoreDetector,
    build_random_forest_baseline,
    create_baseline_features,
    extract_break_points,
    make_mean_shift,
    make_multiple_breaks,
    make_variance_shift,
    point_based_metrics,
)

_DATASETS = {
    "mean_shift": make_mean_shift,
    "multiple": make_multiple_breaks,
    "variance": make_variance_shift,
}

_NOTES = {
    "cusum": "Single dominant mean shift; misses secondary breaks.",
    "rolling_zscore": "Flags transitions/outliers; not whole regimes.",
    "pelt": "Multiple mean shifts via penalised segmentation.",
    "ml_baseline": "Supervised Random Forest; needs labelled training data.",
}


def _baseline_break_points(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[int]:
    """Train the ML baseline on ``train_df`` and return predicted break points on ``test_df``."""
    train_features = create_baseline_features(train_df)
    model = build_random_forest_baseline()
    model.fit(train_features[FEATURE_COLUMNS], train_features[TARGET_COLUMN])

    test_features = create_baseline_features(test_df)
    predictions = model.predict(test_features[FEATURE_COLUMNS])
    return extract_break_points(predictions)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        choices=sorted(_DATASETS),
        default="mean_shift",
        help="Synthetic dataset to use (default: mean_shift).",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=5,
        help="Index tolerance for matching predicted to true breaks (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/method_comparison.csv"),
        help="Where to write the comparison CSV (default: outputs/method_comparison.csv).",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=None,
        help="Optional path to save a PELT detection figure.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the comparison workflow."""
    args = parse_args(argv)

    generator = _DATASETS[args.dataset]
    test_df, true_break_points = generator(seed=0)
    train_df, _ = generator(seed=1)  # independent series for the supervised baseline

    detectors = [CusumDetector(), RollingZScoreDetector(), PeltDetector()]
    rows = []

    for detector in detectors:
        result = detector.detect(test_df)
        predicted = extract_break_points(result["has_structural_break"].to_numpy())
        metrics = point_based_metrics(true_break_points, predicted, tolerance=args.tolerance)
        rows.append(
            {
                "method": detector.name,
                "precision": round(metrics["precision"], 3),
                "recall": round(metrics["recall"], 3),
                "f1": round(metrics["f1"], 3),
                "n_true_breaks": len(true_break_points),
                "n_predicted_breaks": len(predicted),
                "notes": _NOTES.get(detector.name, ""),
            }
        )

    baseline_points = _baseline_break_points(train_df, test_df)
    baseline_metrics = point_based_metrics(
        true_break_points, baseline_points, tolerance=args.tolerance
    )
    rows.append(
        {
            "method": "ml_baseline",
            "precision": round(baseline_metrics["precision"], 3),
            "recall": round(baseline_metrics["recall"], 3),
            "f1": round(baseline_metrics["f1"], 3),
            "n_true_breaks": len(true_break_points),
            "n_predicted_breaks": len(baseline_points),
            "notes": _NOTES["ml_baseline"],
        }
    )

    comparison = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output, index=False)

    print(
        f"Dataset: {args.dataset} | true breaks at {true_break_points} "
        f"| tolerance ±{args.tolerance}"
    )
    print(comparison.to_string(index=False))
    print(f"\nWrote comparison to '{args.output}'.")

    if args.figure is not None:
        from structural_break.visualization import plot_detection

        pelt_result = PeltDetector().detect(test_df)
        plot_detection(
            test_df,
            pelt_result,
            true_break_points=true_break_points,
            title=f"PELT detection — {args.dataset}",
            output_path=args.figure,
        )
        print(f"Wrote figure to '{args.figure}'.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
