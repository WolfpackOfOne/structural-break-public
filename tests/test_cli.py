"""End-to-end test for the baseline CLI wrapper."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "baseline.py"


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def _toy_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=10, freq="D"),
            "value": [0.0, 0.1, -0.1, 0.0, 0.1, 3.0, 3.1, 2.9, 3.0, 3.2],
            "has_structural_break": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )
    test = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-02-01", periods=6, freq="D"),
            "value": [0.0, 0.1, 0.0, 3.0, 3.1, 2.9],
        }
    )
    return train, test


def test_cli_runs_and_writes_output(tmp_path: Path) -> None:
    train, test = _toy_frames()
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    output_path = tmp_path / "out" / "submission.csv"
    _write_csv(train_path, train)
    _write_csv(test_path, test)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--train", str(train_path),
            "--test", str(test_path),
            "--output", str(output_path),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.is_file()

    submission = pd.read_csv(output_path)
    assert list(submission.columns) == ["timestamp", "has_structural_break"]
    assert len(submission) == len(test)


def test_cli_reports_missing_file(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--train", str(tmp_path / "missing.csv")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "not found" in result.stderr.lower()
