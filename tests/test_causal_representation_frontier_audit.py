"""Lightweight guards for the CAUSAL REPRESENTATION FRONTIER audit artifacts.

These tests read committed CSVs and run the static audit script's own checks.
They train nothing, score nothing and touch no store, so they run in any
checkout with no environment set up and no data caches present.
"""

from __future__ import annotations

import csv
import importlib.util
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT_DIR = os.path.join(ROOT, "research", "reports", "causal_representation_frontier")
SCRIPT = os.path.join(ROOT, "research", "scripts",
                      "audit_causal_representation_frontier.py")


def _load_module():
    """Import the audit script as a module without requiring PYTHONPATH.

    Returns
    -------
    module
        The imported audit module.
    """
    spec = importlib.util.spec_from_file_location("crf_audit", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(name: str) -> list:
    """Read one report CSV.

    Parameters
    ----------
    name : str
        File name inside the report directory.

    Returns
    -------
    list
        One dict per data row.
    """
    with open(os.path.join(REPORT_DIR, name), newline="") as fh:
        return list(csv.DictReader(fh))


def test_audit_artifacts_exist_and_are_non_empty() -> None:
    """Every artifact the program's design cites must exist and carry rows."""
    for name in ("prior_representation_audit.csv",
                 "representation_collision_matrix.csv",
                 "crf_candidate_priority.csv"):
        rows = _rows(name)
        assert rows, f"{name} is empty"
    for name in ("CAUSAL_REPRESENTATION_FRONTIER.md", "CRF_PROGRAM_PREREG.md",
                 "SOURCES.md"):
        path = os.path.join(REPORT_DIR, name)
        assert os.path.getsize(path) > 2000, f"{name} looks truncated"


def test_results_csv_is_untouched_by_the_design_task() -> None:
    """The CRF design task must not modify the experiment ledger."""
    mod = _load_module()
    out = mod.check_results_untouched()
    assert out["identical"], (
        f"RESULTS.csv changed: {out['observed_sha256']} != "
        f"{out['expected_sha256']}")
    assert out["no_crf_rows"], f"CRF rows appeared in RESULTS.csv: {out['crf_rows']}"


def test_every_proposed_candidate_differs_from_its_closest_prior() -> None:
    """A proposed mechanism must differ load-bearingly, or be declared conditional."""
    mod = _load_module()
    out = mod.check_collisions()
    assert out["all_pass"], out["per_candidate"]


def test_audit_rows_carry_a_falsification_reading() -> None:
    """Each prior attempt records both what it closes and what it does not."""
    for row in _rows("prior_representation_audit.csv"):
        assert row["what_it_falsifies"].strip(), row["experiment_id"]
        assert row["what_it_does_NOT_falsify"].strip(), row["experiment_id"]
        assert row["source"].strip(), row["experiment_id"]


def test_no_proposed_candidate_uses_rt600_as_an_input() -> None:
    """Section 16: the primary new mechanisms must be RT600-independent."""
    for row in _rows("representation_collision_matrix.csv"):
        if not row["status"].startswith("PROPOSED"):
            continue
        assert row["uses_rt600"].strip().upper() == "NO", row["mechanism"]
        assert row["independent_of_rt600"].strip().upper() == "YES", row["mechanism"]


def test_frontier_recomputes_and_stays_descriptive() -> None:
    """The abandon gate's basis must be reproducible from recorded arms."""
    pytest.importorskip("numpy")
    mod = _load_module()
    f = mod.frontier()
    assert f["n_arms"] == 17
    # Signal and redundancy have been near-perfectly confounded historically;
    # this is the fact the program is built around, so guard it.
    assert f["corr_standalone_rho"] > 0.95
    # The best low-redundancy arm ever measured is RT-1201 at 0.58358.
    assert f["best_arm_at_rho_le_0.60"] == "RT-1201"
    assert abs(f["best_standalone_at_rho_le_0.60"] - 0.58358) < 1e-9
