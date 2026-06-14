"""Tests for the optional plotting helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from structural_break.detectors import CusumDetector
from structural_break.synthetic import make_mean_shift

# Skip entirely if matplotlib is not installed (the helper is optional).
pytest.importorskip("matplotlib")


@pytest.fixture(autouse=True)
def _use_agg_backend():
    import matplotlib

    matplotlib.use("Agg")


def test_plot_detection_saves_figure(tmp_path: Path) -> None:
    from structural_break.visualization import plot_detection

    df, true_points = make_mean_shift(n_pre=30, n_post=30)
    result = CusumDetector().detect(df)
    output = tmp_path / "fig.png"

    ax = plot_detection(df, result, true_break_points=true_points, output_path=output)

    assert output.is_file()
    assert ax is not None
