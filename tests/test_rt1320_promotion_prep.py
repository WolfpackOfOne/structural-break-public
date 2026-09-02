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


def test_current_audit_keeps_missing_promotion_gates_blocking():
    report = prep.build_audit()
    assert report["status"] == "PROMOTION_BLOCKED"
    assert "champion_relative_endpoint" not in report["promotion_blockers"]
    assert "nested_teacher_student_report" not in report["promotion_blockers"]
    for gate in (
        "alternate_partition_leg",
        "student_artifact_causality",
        "final10k_target_and_fit",
        "production_artifact_manifest",
        "crunch_test",
        "external_score",
    ):
        assert gate in report["promotion_blockers"]


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
