#!/usr/bin/env python
"""RT-1320 promotion-prep and deployability audit.

This script is intentionally light by default: it reads committed evidence and
optionally validates a finished model artifact. It does not train, score a new
candidate, append ``research/RESULTS.csv``, touch the lockbox, or allocate a new
RT ID.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = REPO / "engineering" / "reports" / "rt1320_promotion_prep"
CONTRACT_JSON = (
    REPO
    / "research"
    / "reports"
    / "armc_residual_student_confirm_s20260901"
    / "E2_E1_addition_contract.json"
)
STUDENT_REPORT_JSON = (
    REPO
    / "research"
    / "reports"
    / "armc_residual_student_confirm_s20260901"
    / "armc_residual_student.json"
)
DEFAULT_ARTIFACT_CAUSALITY_JSON = DEFAULT_REPORT_DIR / "ARTIFACT_CAUSALITY.json"
DEFAULT_CRUNCH_TEST_JSON = DEFAULT_REPORT_DIR / "CRUNCH_TEST.json"

RT600_FEATURE_MANIFEST_SHA256 = "1646c3b9e09d8a7fb3c564483a1d1d999680caeefe848b092f907a6cac80cced"
FOLDS_FINAL10K_SHA256 = "bf0cdf642bde018a632663ae7d211714a173fb64ef824b15416caf2649e0c716"
SCDF_TIME_COORD = "log_n_seen"

RT1257_MEMBERS = [
    {"slot": 0, "name": "CAT-300", "id": "RT-1255", "kind": "catboost", "path": "model.cbm.0"},
    {"slot": 1, "name": "RT-410", "id": "RT-410", "kind": "lightgbm", "path": "model.txt.1"},
    {"slot": 2, "name": "RT-411", "id": "RT-411", "kind": "lightgbm", "path": "model.txt.2"},
    {"slot": 3, "name": "RT-412", "id": "RT-412", "kind": "lightgbm", "path": "model.txt.3"},
    {"slot": 4, "name": "CAT-413", "id": "RT-1254", "kind": "catboost", "path": "model.cbm.4"},
    {"slot": 5, "name": "RT-414", "id": "RT-414", "kind": "lightgbm", "path": "model.txt.5"},
    {"slot": 6, "name": "RT-415", "id": "RT-415", "kind": "lightgbm", "path": "model.txt.6"},
]
RT1320_STUDENT_MEMBER = {
    "slot": 7,
    "name": "M1 Arm-C residual student",
    "id": "RT-1320",
    "kind": "lightgbm",
    "objective": "regression",
    "path": "model.txt.7",
    "calibration": "SCDF_NSEEN over raw regression output",
}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()
    except Exception:
        return ""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_deployment_contract() -> dict[str, Any]:
    """Return the intended RT-1320 production artifact contract.

    ``src/sbr/production/model.py`` is already member-count agnostic and allows
    the manifest to override the default provenance pin. RT-1320 therefore needs
    an 8-member manifest/artifact, not a new inference architecture.
    """
    members = [dict(x) for x in RT1257_MEMBERS] + [dict(RT1320_STUDENT_MEMBER)]
    return {
        "experiment_id": "RT-1320",
        "what": "RT-1257 plus one Arm-C residual-student member",
        "member_contract": "addition of an 8th exchangeable member to RT-1257",
        "members": members,
        "n_members": len(members),
        "feature_bank": {
            "modules": [
                "m00_core",
                "m01_seq",
                "m02_dist",
                "m03_dyn",
                "m04_resid",
                "m06_loc",
                "m07_bayes",
            ],
            "n_columns": 500,
            "feature_manifest_sha256": RT600_FEATURE_MANIFEST_SHA256,
        },
        "student": {
            "id": "RT-1320",
            "input_features": "same 500 causal columns as RT-1257",
            "target": (
                "nested fold-pure Arm-C teacher residual, regenerated on "
                "folds_final10k before final fit"
            ),
            "model_kind": "LightGBM booster",
            "objective": "regression",
            "score_scale": (
                "raw regression output is converted by the member's own "
                "fold-pure SmoothTimeCDFCal payload before blending"
            ),
            "model_file": "model.txt.7",
            "new_runtime_dependency": "none beyond RT-1257's LightGBM/CatBoost stack",
        },
        "calibration": {
            "kind": "scdf",
            "time_coord": SCDF_TIME_COORD,
            "integration": "equal_weight_mean_of_8_member_scdf_scores",
            "student_payload_required": True,
        },
        "expected_provenance": {
            "n_series": 10000,
            "partition": "folds_final10k",
            "folds_sha256": FOLDS_FINAL10K_SHA256,
            "n_boosters": 8,
            "calibration_kind": "scdf",
            "calibration_time_coord": SCDF_TIME_COORD,
        },
        "production_code_assumption": (
            "ProductionModel.load already accepts manifest model_files of mixed "
            "kind and uses manifest expected_provenance to override n_boosters."
        ),
    }


def endpoint_evidence(contract: dict[str, Any] | None) -> dict[str, Any]:
    if contract is None:
        return {
            "status": "missing",
            "path": str(CONTRACT_JSON),
            "message": "corrected E2-E1 addition contract JSON is absent",
        }
    primary = float(contract.get("PRIMARY_E2_minus_E1", 0.0))
    secondary = float(contract.get("SECONDARY_E2_minus_E0", 0.0))
    noise = float(contract.get("noise_floor_paired_bootstrap", 0.0011))
    folds = int(contract.get("primary_folds_positive", 0))
    passed = primary >= noise and folds >= 4
    return {
        "status": "passed" if passed else "failed",
        "path": str(CONTRACT_JSON),
        "primary_E2_minus_E1": primary,
        "secondary_E2_minus_E0": secondary,
        "primary_folds_positive": folds,
        "noise_floor_paired_bootstrap": noise,
        "primary_clears_noise_floor": bool(contract.get("primary_clears_noise_floor")),
        "message": (
            "champion-relative addition endpoint clears the current noise floor"
            if passed
            else "champion-relative addition endpoint does not clear the gate"
        ),
    }


def student_report_evidence(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {
            "status": "missing",
            "path": str(STUDENT_REPORT_JSON),
            "message": "RT-1320 confirmation report JSON is absent",
        }
    gates = report.get("gates") or {}
    purity = ((report.get("causality_contract") or {}).get("fold_purity") or {})
    passed = bool(
        gates.get("prebreak_damage_gate_passed")
        and gates.get("neverbreak_net_positive")
        and purity.get("new_nested_passed")
        and purity.get("old_global_oof_sentinel_catches_defect")
    )
    return {
        "status": "passed" if passed else "failed",
        "path": str(STUDENT_REPORT_JSON),
        "prebreak_damage_rate_vs_rt600": gates.get("observed_prebreak_damage_rate_vs_rt600"),
        "max_prebreak_damage_rate": gates.get("max_prebreak_damage_rate"),
        "neverbreak_net_rate_vs_rt600": gates.get("observed_neverbreak_net_rate_vs_rt600"),
        "nested_fold_purity_passed": purity.get("new_nested_passed"),
        "old_scheme_sentinel_catches_defect": purity.get("old_global_oof_sentinel_catches_defect"),
        "message": (
            "nested teacher/student report gates are present and internally clean"
            if passed
            else "nested teacher/student report gates are incomplete or failed"
        ),
    }


def inspect_model_manifest(model_dir: Path | None) -> dict[str, Any]:
    if model_dir is None:
        return {
            "status": "missing",
            "message": "no RT-1320 model directory supplied",
            "checks": {},
        }
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        return {
            "status": "missing",
            "model_dir": str(model_dir),
            "message": "model directory has no manifest.json",
            "checks": {"manifest_exists": False},
        }

    manifest = read_json(manifest_path) or {}
    model_files = manifest.get("model_files") or []
    booster_columns = manifest.get("booster_columns") or []
    cal_models = (manifest.get("calibration") or {}).get("models") or []
    expected = manifest.get("expected_provenance") or {}
    trained_on = manifest.get("trained_on") or {}
    streams = manifest.get("streams") or []
    ids = {str(s.get("id")) for s in streams if isinstance(s, dict)}

    expected_files = {m["path"] for m in RT1257_MEMBERS + [RT1320_STUDENT_MEMBER]}
    present_files = {
        spec.get("path")
        for spec in model_files
        if isinstance(spec, dict) and isinstance(spec.get("path"), str)
    }
    file_paths_exist = {
        path: (model_dir / path).exists()
        for path in sorted(expected_files)
    }
    checks = {
        "manifest_exists": True,
        "experiment_id_is_rt1320": manifest.get("experiment_id") == "RT-1320",
        "feature_manifest_is_rt600_bank": (
            manifest.get("feature_manifest_sha256") == RT600_FEATURE_MANIFEST_SHA256
        ),
        "has_8_model_files": len(model_files) == 8,
        "has_8_booster_slices": len(booster_columns) == 8,
        "has_8_calibration_models": len(cal_models) == 8,
        "expected_provenance_n_boosters_is_8": expected.get("n_boosters") == 8,
        "trained_on_final10k_10000": (
            trained_on.get("partition") == "folds_final10k"
            and trained_on.get("n_series") == 10000
        ),
        "folds_sha256_final10k": manifest.get("folds_sha256") == FOLDS_FINAL10K_SHA256,
        "member_ids_include_rt1257_and_rt1320": {
            "RT-1255",
            "RT-1254",
            "RT-1320",
        }.issubset(ids),
        "expected_model_file_paths_present_in_manifest": expected_files.issubset(present_files),
        "expected_model_file_paths_exist": all(file_paths_exist.values()),
    }
    status = "passed" if all(checks.values()) else "failed"
    return {
        "status": status,
        "model_dir": str(model_dir),
        "manifest": str(manifest_path),
        "experiment_id": manifest.get("experiment_id"),
        "n_model_files": len(model_files),
        "n_booster_slices": len(booster_columns),
        "n_calibration_models": len(cal_models),
        "stream_ids": sorted(ids),
        "file_paths_exist": file_paths_exist,
        "checks": checks,
        "message": (
            "model manifest matches the RT-1320 8-member production contract"
            if status == "passed"
            else "model manifest is not yet a valid RT-1320 production artifact"
        ),
    }


def artifact_validation_evidence(path: Path) -> dict[str, Any]:
    data = read_json(path)
    if data is None:
        return {
            "status": "missing",
            "path": str(path),
            "message": "artifact causality validation has not been run",
        }
    passed = bool(data.get("PASS"))
    return {
        "status": "passed" if passed else "failed",
        "path": str(path),
        "pytest_exit_code": data.get("pytest_exit_code"),
        "manifest_status": (data.get("manifest_probe") or {}).get("status"),
        "message": (
            "artifact-level causality tests ran and passed"
            if passed
            else "artifact-level causality validation is absent or failed"
        ),
    }


def crunch_test_evidence(path: Path) -> dict[str, Any]:
    data = read_json(path)
    if data is None:
        return {
            "status": "missing",
            "path": str(path),
            "message": "no RT-1320 Crunch-test record has been filed",
        }
    result = str(data.get("result", "")).upper()
    determinism = str(data.get("determinism_check", "")).lower()
    passed = result == "PASSED" and "pass" in determinism
    return {
        "status": "passed" if passed else "failed",
        "path": str(path),
        "result": data.get("result"),
        "determinism_check": data.get("determinism_check"),
        "message": "Crunch test passed" if passed else "Crunch test is absent or failed",
    }


def build_audit(
    model_dir: Path | None = None,
    artifact_causality_path: Path = DEFAULT_ARTIFACT_CAUSALITY_JSON,
    crunch_test_path: Path = DEFAULT_CRUNCH_TEST_JSON,
) -> dict[str, Any]:
    contract_json = read_json(CONTRACT_JSON)
    student_json = read_json(STUDENT_REPORT_JSON)
    manifest_probe = inspect_model_manifest(model_dir)
    artifact_causality = artifact_validation_evidence(artifact_causality_path)
    crunch_test = crunch_test_evidence(crunch_test_path)

    gates = [
        {
            "id": "champion_relative_endpoint",
            "required_for": "research_survival",
            **endpoint_evidence(contract_json),
        },
        {
            "id": "nested_teacher_student_report",
            "required_for": "research_survival",
            **student_report_evidence(student_json),
        },
        {
            "id": "alternate_partition_leg",
            "required_for": "promotion",
            "status": "missing",
            "message": (
                "must refit comparable RT-1257 and RT-1320 OOF under predeclared "
                "folds_alt*.parquet partitions; canonical alone is insufficient"
            ),
        },
        {
            "id": "student_artifact_causality",
            "required_for": "promotion",
            **artifact_causality,
        },
        {
            "id": "final10k_target_and_fit",
            "required_for": "production_artifact",
            "status": "passed" if manifest_probe["status"] == "passed" else "missing",
            "message": (
                "model manifest records an 8-member folds_final10k fit"
                if manifest_probe["status"] == "passed"
                else "nested Arm-C residual target must be regenerated on folds_final10k and fit"
            ),
        },
        {
            "id": "production_artifact_manifest",
            "required_for": "production_artifact",
            **manifest_probe,
        },
        {
            "id": "crunch_test",
            "required_for": "external_submission",
            **crunch_test,
        },
        {
            "id": "external_score",
            "required_for": "active_component_status",
            "status": "missing",
            "message": "RT-1320 has no official external score as an 8-member system",
        },
    ]
    missing = [g["id"] for g in gates if g["status"] != "passed"]
    promotion_blockers = [
        g["id"]
        for g in gates
        if g["status"] != "passed" and g["required_for"] != "research_survival"
    ]
    return {
        "schema": "sbr.rt1320_promotion_prep/1",
        "generated_at": utc_now(),
        "git_sha": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "status": "PROMOTION_BLOCKED" if promotion_blockers else "READY_FOR_OWNER_REVIEW",
        "summary": (
            "RT-1320 remains the best live post-RT-1257 engineering target, but "
            "promotion is blocked until the missing gates are closed."
            if promotion_blockers
            else "RT-1320 has no recorded promotion blockers in this audit."
        ),
        "recommended_next_action": (
            "Do the student artifact causality harness first if a model artifact exists; "
            "otherwise prepare the folds_final10k target/fit job for the competition server."
        ),
        "missing_or_failed_gates": missing,
        "promotion_blockers": promotion_blockers,
        "evidence_paths": {
            "endpoint_contract": str(CONTRACT_JSON),
            "student_report": str(STUDENT_REPORT_JSON),
            "artifact_causality": str(artifact_causality_path),
            "crunch_test": str(crunch_test_path),
        },
        "deployment_contract": build_deployment_contract(),
        "gates": gates,
    }


def make_markdown(report: dict[str, Any]) -> str:
    contract = report["deployment_contract"]
    gates = report["gates"]
    lines = [
        "# RT-1320 Promotion Prep",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Branch: `{report['git_branch']}`",
        f"Git SHA: `{report['git_sha']}`",
        "",
        f"Status: **{report['status']}**",
        "",
        report["summary"],
        "",
        "## Current Decision",
        "",
        "Work on RT-1320, not another GPU tabular family. RT-1320 is the only "
        "current research-alive candidate with a champion-relative addition "
        "endpoint above the measured noise floor. It is not deployable yet.",
        "",
        "## Artifact Contract",
        "",
        f"- Experiment ID: `{contract['experiment_id']}`",
        f"- Members: `{contract['n_members']}` total, RT-1257 plus one residual student",
        "- Student model file: `model.txt.7`",
        "- Student objective: `regression`",
        "- Student score scale: member-local `SCDF_NSEEN` calibration before "
        "the equal-weight blend",
        "- Feature bank: existing 500 causal columns, unchanged RT-600 manifest SHA",
        "- Final fit partition: `folds_final10k`, 10,000 series",
        "",
        "## Gate Status",
        "",
        "| gate | status | note |",
        "|---|---|---|",
    ]
    for gate in gates:
        lines.append(f"| `{gate['id']}` | `{gate['status']}` | {gate['message']} |")
    lines.extend(
        [
            "",
            "## Next Server Work",
            "",
            "Prepare the folds-final10k Arm-C residual target and train the 8th-member "
            "student. The target must be regenerated under the final 10,000-series "
            "population; the existing nested teacher labels only cover the 8,000 dev "
            "series and cannot be reused as a final artifact.",
            "",
            "After a model directory exists, run:",
            "",
            "```bash",
            "python research/scripts/rt1320_promotion_prep.py validate-artifact "
            "--model-dir models/rt1320_final",
            "python research/scripts/rt1320_promotion_prep.py audit "
            "--model-dir models/rt1320_final",
            "```",
            "",
            "Promotion remains blocked until the artifact causality report, alternate "
            "partition leg, Crunch test, and external score are all recorded.",
            "",
        ]
    )
    return "\n".join(lines)


def write_audit(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "DEPLOYABILITY_AUDIT.json"
    md_path = out_dir / "DEPLOYABILITY_AUDIT.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(make_markdown(report))
    return {"json": str(json_path), "markdown": str(md_path)}


def run_artifact_validation(model_dir: Path, out_dir: Path) -> dict[str, Any]:
    manifest_probe = inspect_model_manifest(model_dir)
    tests = [
        "tests/test_no_n_online_leakage.py",
        "tests/test_production_contract.py::test_infer_contract_single_pass_and_range",
        "tests/test_production_contract.py::test_infer_is_deterministic_and_order_independent",
        "tests/test_production_contract.py::test_frozen_model_passes_its_own_provenance_gate",
    ]
    env = dict(os.environ)
    env["SBR_ROOT"] = str(REPO)
    env["SBR_MODEL_DIR"] = str(model_dir)
    env["PYTHONPATH"] = f"{REPO / 'src'}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *tests, "-q", "-rs", "--tb=short"],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    passed = proc.returncode == 0 and manifest_probe["status"] == "passed"
    report = {
        "schema": "sbr.rt1320_artifact_causality/1",
        "generated_at": utc_now(),
        "model_dir": str(model_dir),
        "manifest_probe": manifest_probe,
        "tests": tests,
        "pytest_exit_code": proc.returncode,
        "pytest_stdout_tail": proc.stdout[-4000:],
        "pytest_stderr_tail": proc.stderr[-4000:],
        "PASS": passed,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ARTIFACT_CAUSALITY.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    audit = sub.add_parser("audit", help="write the current RT-1320 deployability audit")
    audit.add_argument("--model-dir", type=Path, default=None)
    audit.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT_DIR)
    audit.add_argument("--artifact-causality", type=Path, default=DEFAULT_ARTIFACT_CAUSALITY_JSON)
    audit.add_argument("--crunch-test", type=Path, default=DEFAULT_CRUNCH_TEST_JSON)
    audit.add_argument("--no-write", action="store_true")

    validate = sub.add_parser("validate-artifact", help="run artifact-level causality tests")
    validate.add_argument("--model-dir", type=Path, required=True)
    validate.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT_DIR)

    parser.set_defaults(
        command="audit",
        model_dir=None,
        out_dir=DEFAULT_REPORT_DIR,
        artifact_causality=DEFAULT_ARTIFACT_CAUSALITY_JSON,
        crunch_test=DEFAULT_CRUNCH_TEST_JSON,
        no_write=False,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "validate-artifact":
        report = run_artifact_validation(args.model_dir.resolve(), args.out_dir.resolve())
        print(
            json.dumps(
                {
                    "PASS": report["PASS"],
                    "out": str(args.out_dir / "ARTIFACT_CAUSALITY.json"),
                },
                indent=2,
            )
        )
        return 0 if report["PASS"] else 1

    report = build_audit(
        model_dir=args.model_dir.resolve() if args.model_dir else None,
        artifact_causality_path=args.artifact_causality.resolve(),
        crunch_test_path=args.crunch_test.resolve(),
    )
    paths = None if args.no_write else write_audit(report, args.out_dir.resolve())
    print(
        json.dumps(
            {
                "status": report["status"],
                "promotion_blockers": report["promotion_blockers"],
                "recommended_next_action": report["recommended_next_action"],
                "written": paths,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
