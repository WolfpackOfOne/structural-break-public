from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "scripts"))

import rt1320_promotion_prep as prep  # noqa: E402


def test_deployment_contract_is_an_eight_member_addition():
    contract = prep.build_deployment_contract()
    assert contract["experiment_id"] == "RT-1320"
    assert contract["n_members"] == 8
    assert [m["id"] for m in contract["members"]][-1] == "RT-1320"
    assert contract["members"][-1]["objective"] == "regression"
    assert contract["members"][-1]["path"] == "model.txt.7"
    assert contract["expected_provenance"]["n_boosters"] == 8
    assert contract["expected_provenance"]["partition"] == "folds_final10k"


def test_audit_keeps_gates_without_evidence_blocking():
    """A gate blocks exactly while its evidence is absent.

    This pins the *mechanism*, not a snapshot of which gates happen to be open.
    The original version hardcoded six gates as blocking and went stale three
    times over as evidence landed, so it is now split: gates whose evidence is
    committed must be closed, and gates that depend on a model directory must
    still block when build_audit() is called without one.
    """
    report = prep.build_audit()  # deliberately no model_dir
    assert report["status"] == "PROMOTION_BLOCKED"

    # Evidence committed to the repo -- these must NOT block any more.
    for gate in (
        "champion_relative_endpoint",
        "nested_teacher_student_report",
        "student_artifact_causality",   # ARTIFACT_CAUSALITY.json
        "crunch_test",                  # CRUNCH_TEST.json
        "alternate_partition_leg",      # four E2_E1_addition_contract.json records
    ):
        assert gate not in report["promotion_blockers"], gate

    # No model directory was supplied, so the artifact-dependent gates block,
    # and external_score cannot close without a Crunch score.
    for gate in (
        "final10k_target_and_fit",
        "production_artifact_manifest",
        "external_score",
    ):
        assert gate in report["promotion_blockers"], gate


def test_alternate_partition_leg_reads_all_four_partitions():
    """The leg gate must evaluate evidence, not assert a hardcoded status.

    It was previously pinned to status="missing" with no reader, so it could
    never close however much was run.
    """
    ev = prep.alt_partition_evidence()
    assert ev["status"] == "passed"
    assert ev["rule_verdict"] == "PASS"
    assert sorted(ev["per_partition"]) == ["alt1", "alt2", "alt3", "canonical"]
    assert ev["mean_E2_minus_E1"] >= prep.ALT_NOISE_FLOOR
    assert ev["n_partitions_negative"] == 0
    assert all(ev["checks"].values())


def test_corrected_endpoint_contract_is_the_primary_evidence():
    contract = json.loads(prep.CONTRACT_JSON.read_text())
    evidence = prep.endpoint_evidence(contract)
    assert evidence["status"] == "passed"
    assert evidence["primary_E2_minus_E1"] > evidence["noise_floor_paired_bootstrap"]
    assert evidence["primary_folds_positive"] == 4


def test_manifest_probe_rejects_non_rt1320_manifest(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "RT-1257",
                "feature_manifest_sha256": prep.RT600_FEATURE_MANIFEST_SHA256,
                "model_files": [],
                "booster_columns": [],
                "calibration": {"models": []},
                "expected_provenance": {"n_boosters": 7},
                "trained_on": {"partition": "folds_final10k", "n_series": 10000},
                "folds_sha256": prep.FOLDS_FINAL10K_SHA256,
                "streams": [],
            }
        )
    )
    probe = prep.inspect_model_manifest(model_dir)
    assert probe["status"] == "failed"
    assert not probe["checks"]["experiment_id_is_rt1320"]
    assert not probe["checks"]["has_8_model_files"]


def test_manifest_probe_accepts_static_rt1320_shape(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    members = prep.RT1257_MEMBERS + [prep.RT1320_STUDENT_MEMBER]
    for member in members:
        (model_dir / member["path"]).write_text("placeholder\n")
    (model_dir / "manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "RT-1320",
                "feature_manifest_sha256": prep.RT600_FEATURE_MANIFEST_SHA256,
                "model_files": [{"kind": m["kind"], "path": m["path"]} for m in members],
                "booster_columns": [[0] for _ in members],
                "calibration": {
                    "kind": "scdf",
                    "time_coord": "log_n_seen",
                    "models": [{} for _ in members],
                },
                "expected_provenance": {
                    "n_boosters": 8,
                    "partition": "folds_final10k",
                    "n_series": 10000,
                    "folds_sha256": prep.FOLDS_FINAL10K_SHA256,
                    "calibration_kind": "scdf",
                    "calibration_time_coord": "log_n_seen",
                },
                "trained_on": {"partition": "folds_final10k", "n_series": 10000},
                "folds_sha256": prep.FOLDS_FINAL10K_SHA256,
                "streams": [{"id": m["id"]} for m in members],
            }
        )
    )
    probe = prep.inspect_model_manifest(model_dir)
    assert probe["status"] == "passed"
    assert all(probe["checks"].values())
