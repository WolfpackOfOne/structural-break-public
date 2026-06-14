#!/usr/bin/env python
"""Command-line wrapper around the structural-break baseline workflow.

Example
-------
Run from the repository root::

    python scripts/baseline.py \\
        --train data/train.csv \\
        --test data/test.csv \\
        --output outputs/submission.csv

The script reads the training and test data, engineers baseline features, trains a
Random Forest classifier, prints training metrics, and writes a submission CSV with
``timestamp`` and ``has_structural_break`` columns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running straight from a fresh clone without `pip install -e .`.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from structural_break import (  # noqa: E402  (import after sys.path bootstrap)
    FEATURE_COLUMNS,
    REQUIRED_INPUT_COLUMNS,
    TARGET_COLUMN,
    build_random_forest_baseline,
    build_submission,
    create_baseline_features,
    evaluate_predictions,
    load_csv,
    validate_required_columns,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/train.csv"),
        help="Path to the training CSV (default: data/train.csv).",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/test.csv"),
        help="Path to the test CSV (default: data/test.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/submission.csv"),
        help="Where to write the submission CSV (default: outputs/submission.csv).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the model (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the baseline workflow end-to-end.

    Returns a process exit code: ``0`` on success, ``1`` if the input data is
    missing or malformed (reported as a clean message rather than a traceback).
    """
    args = parse_args(argv)

    try:
        train_df = load_csv(args.train)
        validate_required_columns(
            train_df, REQUIRED_INPUT_COLUMNS | {TARGET_COLUMN}, "Training data"
        )
        test_df = load_csv(args.test)
        validate_required_columns(test_df, REQUIRED_INPUT_COLUMNS, "Test data")
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    train_features = create_baseline_features(train_df)
    model = build_random_forest_baseline(random_state=args.random_state)
    model.fit(train_features[FEATURE_COLUMNS], train_features[TARGET_COLUMN])

    metrics = evaluate_predictions(
        train_features[TARGET_COLUMN], model.predict(train_features[FEATURE_COLUMNS])
    )
    print(f"Training F1 score: {metrics['f1']:.4f}")
    print(metrics["report"])

    submission = build_submission(model, test_df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {len(submission)} predictions to '{args.output}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
