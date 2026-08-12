#!/usr/bin/env python
"""Evaluate the Real-Time Edition streaming detector on synthetic data.

Runs `structural_break.realtime.StreamingBreakDetector` on a synthetic mix of
break / no-break series in the challenge's `(id, x_historical, x_online, tau)`
shape, then scores it with the competition's own metric, Time-Stratified AUC.

No competition data is required or used — see `realtime_submission.ipynb` for
the notebook used to actually submit to the live competition.

Example
-------
::

    python scripts/realtime_baseline.py --n-series 40 --seed 0

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from structural_break.realtime import (  # noqa: E402
    DetectorParams,
    StreamingBreakDetector,
    make_realtime_dataset,
    time_stratified_auc,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-series", type=int, default=40, help="Number of synthetic series.")
    parser.add_argument(
        "--n-historical", type=int, default=1000, help="Historical segment length."
    )
    parser.add_argument("--n-online", type=int, default=300, help="Online segment length.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the synthetic mix.")
    parser.add_argument("--alpha", type=float, default=0.05, help="EWMA decay.")
    parser.add_argument("--cusum-slack", type=float, default=0.5, help="CUSUM slack.")
    parser.add_argument(
        "--kappa-mean", type=float, default=3.0, help="Mean-evidence tanh scale."
    )
    parser.add_argument(
        "--kappa-var", type=float, default=1.5, help="Variance-evidence tanh scale."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the streaming detector over a synthetic dataset and print TS-AUC."""
    args = parse_args(argv)

    params = DetectorParams(
        alpha=args.alpha,
        cusum_slack=args.cusum_slack,
        kappa_mean=args.kappa_mean,
        kappa_var=args.kappa_var,
    )
    dataset = make_realtime_dataset(
        n_series=args.n_series,
        n_historical=args.n_historical,
        n_online=args.n_online,
        seed=args.seed,
    )

    all_scores = []
    all_labels = []
    n_with_break = 0
    for _, x_historical, x_online, tau in dataset:
        detector = StreamingBreakDetector(x_historical, params)
        scores = [detector.update(x) for x in x_online]

        if tau is not None:
            n_with_break += 1
            labels = [1 if t >= tau else 0 for t in range(len(x_online))]
        else:
            labels = [0] * len(x_online)

        all_scores.append(scores)
        all_labels.append(labels)

    ts_auc = time_stratified_auc(all_scores, all_labels)

    print(f"Series: {len(dataset)} ({n_with_break} with a break, seed={args.seed})")
    print(f"Params: {params}")
    print(f"Time-Stratified AUC: {ts_auc:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
