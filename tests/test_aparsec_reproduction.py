from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research/scripts/reproduce_aparsec_2025.py"
SPEC = importlib.util.spec_from_file_location("reproduce_aparsec_2025", SCRIPT)
assert SPEC is not None
aparsec = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = aparsec
SPEC.loader.exec_module(aparsec)


def test_source_derived_feature_manifest_counts() -> None:
    rows = aparsec.feature_manifest_rows()
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["stage"]] = counts.get(row["stage"], 0) + 1

    assert len(rows) == 2171
    assert counts["FirstFeatureGenerator"] == 8
    assert counts["SecondFeatureGenerator"] == 56
    assert counts["ThirdFeatureGenerator:pivot"] == 1260
    assert counts["ThirdFeatureGenerator:diff"] == 840
    assert counts["FourthFeatureGenerator"] == 6
    assert counts["TabPFN OOF meta-feature"] == 1


def test_public_selection_uses_tail_of_ascending_importance() -> None:
    ordered = [f"f{i}" for i in range(10)]
    assert aparsec.public_selected_tail(ordered, 3) == ["f7", "f8", "f9"]
    assert aparsec.public_selected_tail(ordered, 20) == ordered


def test_public_ensemble_probability_average() -> None:
    probabilities = [
        [0.0, 0.2, 1.0],
        [0.2, 0.4, 0.8],
        [0.4, 0.6, 0.6],
        [0.6, 0.8, 0.4],
    ]
    assert aparsec.average_probability_columns(probabilities) == pytest.approx([0.3, 0.5, 0.7])


def test_public_ensemble_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same length"):
        aparsec.average_probability_columns([[0.1, 0.2], [0.3]])


def test_dependency_gate_requires_public_solution_stack() -> None:
    versions = {
        "polars": "NOT_INSTALLED",
        "lightgbm": "4.6.0",
        "shap": "NOT_INSTALLED",
        "tabpfn": "2.1.3",
    }
    assert aparsec.missing_required_reproduction_deps(versions) == ["polars", "shap"]
