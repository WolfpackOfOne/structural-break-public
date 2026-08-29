"""Core harness: build contexts, run experiments safely, track a leaderboard.

Designed to run **unattended for hours**, so every piece here is built to
survive a bad experiment rather than take the whole run down with it:

- :func:`run_experiment` wraps a single experiment in a timeout + broad
  exception handling, always returning a result row instead of raising.
- :func:`append_leaderboard_row` / :func:`load_leaderboard` /
  :func:`completed_experiment_ids` make the loop resumable: restart it and it
  picks up where it left off, skipping experiments already recorded as "ok".
- :func:`maybe_save_best` keeps the single best model + its config on disk,
  updated only when a new experiment beats it.
- :func:`git_checkpoint` commits (and pushes) progress on an interval, so a
  crash or a killed process never loses more than one checkpoint's worth of
  work.
"""

from __future__ import annotations

import csv
import json
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .competition_data import id_train_val_split, select_ids
from .experiment_registry import ExperimentContext, ExperimentResult, ExperimentSpec
from .series_features import build_detector_feature_matrix, build_feature_matrix

__all__ = [
    "LEADERBOARD_COLUMNS",
    "build_context",
    "run_experiment",
    "load_leaderboard",
    "completed_experiment_ids",
    "append_leaderboard_row",
    "maybe_save_best",
    "git_checkpoint",
]

LEADERBOARD_COLUMNS: list[str] = [
    "timestamp",
    "experiment_id",
    "family",
    "status",
    "auc",
    "runtime_seconds",
    "config",
    "notes",
]


class _ExperimentTimeout(Exception):
    """Raised internally when an experiment exceeds its time budget."""


def _alarm_handler(signum: int, frame: Any) -> None:
    raise _ExperimentTimeout()


def build_context(
    X: pd.DataFrame,
    y: pd.Series,
    val_fraction: float = 0.2,
    seed: int = 42,
    artifact_dir: str | Path | None = None,
) -> ExperimentContext:
    """Split by series id and precompute every experiment's shared features once.

    This is the expensive, one-time setup step — feature extraction runs once
    per run, not once per experiment, so a weekend sweep spends its time
    trying models, not re-deriving the same statistics.
    """
    ids = X.index.get_level_values(0).unique()
    train_ids, val_ids = id_train_val_split(ids, y=y, val_fraction=val_fraction, seed=seed)

    train_X, val_X = select_ids(X, train_ids), select_ids(X, val_ids)
    train_y, val_y = y.loc[train_ids], y.loc[val_ids]

    train_stat = build_feature_matrix(train_X).loc[train_ids]
    val_stat = build_feature_matrix(val_X).loc[val_ids]
    train_det = build_detector_feature_matrix(train_X).loc[train_ids]
    val_det = build_detector_feature_matrix(val_X).loc[val_ids]

    return ExperimentContext(
        train_X=train_X,
        train_y=train_y,
        val_X=val_X,
        val_y=val_y,
        train_stat_features=train_stat,
        val_stat_features=val_stat,
        train_detector_features=train_det,
        val_detector_features=val_det,
        seed=seed,
        artifact_dir=Path(artifact_dir) if artifact_dir else None,
    )


def run_experiment(
    spec: ExperimentSpec, ctx: ExperimentContext, timeout: int | None = 900
) -> tuple[dict[str, Any], Any]:
    """Run one experiment, never raising — always returns ``(row, model)``.

    ``row["status"]`` is one of:

    - ``"ok"`` — succeeded; ``row["auc"]`` and the returned model are valid.
    - ``"skipped"`` — the experiment raised ``NotImplementedError`` (a stub not
      yet filled in) or ``ImportError`` (an optional dependency isn't
      installed). Not a failure of the harness — just nothing to record yet.
    - ``"timeout"`` — exceeded ``timeout`` seconds (best-effort via
      ``SIGALRM``; unavailable on non-Unix platforms, in which case no timeout
      is enforced).
    - ``"error"`` — an unexpected exception; the message is captured in
      ``row["notes"]`` so the run keeps going instead of crashing.
    """
    start = time.time()
    row: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment_id": spec.id,
        "family": spec.family,
        "status": "error",
        "auc": None,
        "runtime_seconds": None,
        "config": json.dumps(spec.config, sort_keys=True),
        "notes": "",
    }
    model = None

    use_alarm = bool(timeout) and hasattr(signal, "alarm")
    previous_handler = None
    try:
        if use_alarm:
            previous_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(int(timeout))

        result: ExperimentResult = spec.fn(ctx, **spec.config)
        row["status"] = "ok"
        row["auc"] = round(float(result.auc), 6)
        row["notes"] = result.notes or spec.description
        model = result.model
    except (NotImplementedError, ImportError) as exc:
        row["status"] = "skipped"
        row["notes"] = str(exc)
    except _ExperimentTimeout:
        row["status"] = "timeout"
        row["notes"] = f"exceeded {timeout}s budget"
    except Exception as exc:  # noqa: BLE001 - one bad experiment must not end the run
        row["status"] = "error"
        row["notes"] = f"{type(exc).__name__}: {exc}"
    finally:
        if use_alarm:
            signal.alarm(0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)
        row["runtime_seconds"] = round(time.time() - start, 2)

    return row, model


def load_leaderboard(path: str | Path) -> pd.DataFrame:
    """Read the leaderboard CSV, or an empty frame with the right columns if absent."""
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    return pd.read_csv(path)


def completed_experiment_ids(path: str | Path) -> set[str]:
    """Experiment ids already recorded with status "ok" — safe to skip on resume.

    Non-"ok" outcomes (error/timeout/skipped) are retried on the next run,
    since the cause (a missing dependency, a transient resource limit) may
    have been fixed since the last attempt.
    """
    df = load_leaderboard(path)
    if df.empty:
        return set()
    return set(df.loc[df["status"] == "ok", "experiment_id"])


def append_leaderboard_row(path: str | Path, row: dict[str, Any]) -> None:
    """Append one result row to the leaderboard CSV, creating it if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEADERBOARD_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def maybe_save_best(model: Any, row: dict[str, Any], artifact_dir: str | Path) -> bool:
    """Persist ``model`` as the new best if ``row`` beats the recorded best AUC.

    Writes ``best_model.joblib`` under ``artifact_dir`` (git-ignored — it's a
    binary and reproducible from the config) and a small, human-readable
    ``best_config.json`` one level up (``artifact_dir.parent`` — tracked in
    git, so the leaderboard's "current best" survives even without the binary).
    Returns True if this row became the new best.
    """
    if model is None or row.get("status") != "ok" or row.get("auc") is None:
        return False

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    best_info_path = artifact_dir.parent / "best_config.json"

    current_best_auc = None
    if best_info_path.is_file():
        current_best_auc = json.loads(best_info_path.read_text()).get("auc")

    if current_best_auc is not None and row["auc"] <= current_best_auc:
        return False

    import joblib

    joblib.dump(model, artifact_dir / "best_model.joblib")
    best_info_path.write_text(
        json.dumps(
            {
                "experiment_id": row["experiment_id"],
                "family": row["family"],
                "auc": row["auc"],
                "config": row["config"],
                "timestamp": row["timestamp"],
            },
            indent=2,
        )
    )
    return True


def git_checkpoint(
    repo_dir: str | Path, message: str, paths: list[str] | None = None, push: bool = True
) -> str:
    """Stage, commit, and push ``paths`` (default: ``experiments/``). Never raises.

    Safe to call often: a no-op commit (nothing changed) is detected and
    skipped rather than producing an empty commit. Push failures (offline,
    no remote configured yet, etc.) are reported in the return value but do
    not stop the run — the commit is still made locally.
    """
    repo_dir = Path(repo_dir)
    paths = paths or ["experiments"]
    try:
        subprocess.run(
            ["git", "add", *paths], cwd=repo_dir, check=True, capture_output=True, text=True
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", *paths],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        if not status.stdout.strip():
            return "no changes to commit"

        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        if not push:
            return "committed locally (push disabled)"
        push_result = subprocess.run(["git", "push"], cwd=repo_dir, capture_output=True, text=True)
        if push_result.returncode != 0:
            return f"committed locally; push failed: {push_result.stderr.strip()[:500]}"
        return "committed and pushed"
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()[:500] if isinstance(exc.stderr, str) else str(exc)
        return f"git checkpoint failed: {stderr}"
