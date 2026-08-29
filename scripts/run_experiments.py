#!/usr/bin/env python
"""Weekend model-search driver loop.

Runs every registered experiment (see
``src/structural_break/experiment_registry.py``) against the competition data,
recording each result to a resumable leaderboard and checkpointing (committing,
optionally pushing) progress on an interval. Designed to be started, left
running unattended for hours, and safely restarted if interrupted — it always
picks up where the leaderboard left off.

Examples
--------
Smoke-test the harness against a synthetic dataset shaped like the real
competition data (no downloads required)::

    python scripts/run_experiments.py --synthetic

Run against the real competition data once it's placed under ``data/``
(supports the multi-part Parquet split, e.g. ``X_train.part1.parquet``)::

    python scripts/run_experiments.py \\
        --x-train data/X_train.part1.parquet data/X_train.part2.parquet \\
        --y-train data/y_train.parquet

Repeat the same command any time over the weekend — already-succeeded
experiments are skipped automatically.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from structural_break.competition_data import load_feature_parts, load_labels  # noqa: E402
from structural_break.experiment_registry import EXPERIMENTS  # noqa: E402
from structural_break.experiment_runner import (  # noqa: E402
    append_leaderboard_row,
    build_context,
    completed_experiment_ids,
    git_checkpoint,
    maybe_save_best,
    run_experiment,
)
from structural_break.synthetic import make_synthetic_competition_dataset  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--x-train",
        nargs="+",
        default=None,
        help="One or more X_train Parquet files (parts are concatenated).",
    )
    parser.add_argument("--y-train", default=None, help="y_train Parquet/CSV file.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a synthetic competition-shaped dataset instead of real data "
        "(default when --x-train is omitted).",
    )
    parser.add_argument("--n-synthetic-series", type=int, default=400)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--timeout-per-experiment", type=int, default=900, help="Seconds; 0 disables."
    )
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Git-checkpoint after this many newly-run experiments.",
    )
    parser.add_argument(
        "--checkpoint-minutes",
        type=float,
        default=30.0,
        help="Also git-checkpoint if this many minutes pass, even mid-batch.",
    )
    parser.add_argument(
        "--no-push", action="store_true", help="Commit checkpoints locally but don't push."
    )
    parser.add_argument("--leaderboard", default=str(_ROOT / "experiments" / "leaderboard.csv"))
    parser.add_argument("--progress", default=str(_ROOT / "experiments" / "progress.md"))
    parser.add_argument("--artifact-dir", default=str(_ROOT / "experiments" / "models"))
    return parser.parse_args()


def _load_data(args: argparse.Namespace):
    if args.x_train:
        if not args.y_train:
            raise SystemExit("--y-train is required when --x-train is given.")
        print(f"Loading real competition data from {args.x_train} / {args.y_train}")
        X = load_feature_parts(args.x_train)
        y = load_labels(args.y_train)
        return X, y

    print(
        f"No --x-train given: using a synthetic competition-shaped dataset "
        f"(n={args.n_synthetic_series}, seed={args.seed}) to smoke-test the harness."
    )
    return make_synthetic_competition_dataset(n_series=args.n_synthetic_series, seed=args.seed)


def _log_progress(progress_path: Path, message: str) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_path, "a") as handle:
        handle.write(message.rstrip("\n") + "\n")


def main() -> None:
    args = _parse_args()
    leaderboard_path = Path(args.leaderboard)
    progress_path = Path(args.progress)
    artifact_dir = Path(args.artifact_dir)

    X, y = _load_data(args)
    print(f"{len(y)} series total, {int(y.sum())} positive ({y.mean():.1%}).")

    print("Building train/validation split and precomputing features (one-time cost)...")
    ctx = build_context(
        X, y, val_fraction=args.val_fraction, seed=args.seed, artifact_dir=artifact_dir
    )
    print(f"train={len(ctx.train_y)} series, val={len(ctx.val_y)} series.")

    already_done = completed_experiment_ids(leaderboard_path)
    pending = [spec for spec in EXPERIMENTS if spec.id not in already_done]
    print(
        f"{len(EXPERIMENTS)} registered experiments, {len(already_done)} already 'ok', "
        f"{len(pending)} to run."
    )

    timeout = args.timeout_per_experiment or None
    run_start = time.time()
    since_checkpoint = time.time()
    run_count = 0
    best_seen = None

    for spec in pending:
        if args.max_experiments is not None and run_count >= args.max_experiments:
            print(f"Reached --max-experiments={args.max_experiments}, stopping.")
            break
        elapsed_minutes = (time.time() - run_start) / 60.0
        if args.max_minutes is not None and elapsed_minutes >= args.max_minutes:
            print(f"Reached --max-minutes={args.max_minutes}, stopping.")
            break

        print(
            f"[{run_count + 1}/{len(pending)}] running {spec.id} ({spec.family})...",
            end=" ",
            flush=True,
        )
        row, model = run_experiment(spec, ctx, timeout=timeout)
        print(f"{row['status']} auc={row['auc']} ({row['runtime_seconds']}s) {row['notes'][:120]}")

        append_leaderboard_row(leaderboard_path, row)
        if maybe_save_best(model, row, artifact_dir):
            best_seen = row
            _log_progress(
                progress_path,
                f"- New best: `{row['experiment_id']}` (family={row['family']}) "
                f"AUC={row['auc']} at {row['timestamp']}.",
            )
            print(f"  -> new best model saved (AUC={row['auc']}).")

        run_count += 1
        due_by_count = run_count % max(args.checkpoint_every, 1) == 0
        due_by_time = (time.time() - since_checkpoint) >= args.checkpoint_minutes * 60
        if due_by_count or due_by_time:
            message = (
                f"Experiment checkpoint: {run_count} run this session, "
                f"last={spec.id} ({row['status']})"
            )
            outcome = git_checkpoint(_ROOT, message, push=not args.no_push)
            print(f"  -> checkpoint: {outcome}")
            since_checkpoint = time.time()

    total_minutes = (time.time() - run_start) / 60.0
    best_line = (
        "(no new best)"
        if best_seen is None
        else f"{best_seen['experiment_id']} AUC={best_seen['auc']}"
    )
    summary = (
        f"\n## Run finished at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
        f"- Ran {run_count} experiment(s) in {total_minutes:.1f} minutes.\n"
        f"- Best this session: {best_line}.\n"
        f"- See `{leaderboard_path}` for the full ranked history.\n"
    )
    _log_progress(progress_path, summary)
    final_outcome = git_checkpoint(
        _ROOT, f"Final checkpoint: {run_count} experiments this session", push=not args.no_push
    )
    print(f"Final checkpoint: {final_outcome}")
    print(summary)


if __name__ == "__main__":
    main()
