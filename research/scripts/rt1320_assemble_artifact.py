"""Assemble the 8-member RT-1320 production artifact.

``rt1320_promotion_prep.py`` VALIDATES ``models/rt1320_final``; nothing built it.
This script is that missing step: it takes the finished 7-member RT-1257 artifact
and the two files the RT-1320 fit produces, and writes the 8-member artifact that
``inspect_model_manifest`` expects.

    RT-1257 artifact (7 members)          RT-1320 fit output
      model.cbm.0  model.txt.1..3           model.txt.7
      model.cbm.4  model.txt.5..6           RT-1320_student_scdf.json
      manifest.json                                |
              \\_________________  ______________/
                                \\/
                        models/rt1320_final  (8 members)

The 8th member is an addition, not a replacement: the seven RT-1257 boosters and
their calibration models are copied through byte-for-byte, and slot 7 is appended.
Every check ``inspect_model_manifest`` performs is re-run here before writing, so
a bad artifact fails at build time rather than at audit time.

Usage:
    python research/scripts/rt1320_assemble_artifact.py --check-base
    python research/scripts/rt1320_assemble_artifact.py \\
        --fit-dir ~/rt1320_local_fit/rt1320_final10k_fit
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
CRUNCH_ROOT = REPO.parent

DEFAULT_RT1257_DIR = (
    CRUNCH_ROOT / "structural-break-rt1257-deployment" / "models" / "rt1257_final"
)
DEFAULT_FIT_DIR = Path.home() / "rt1320_local_fit" / "rt1320_final10k_fit"
DEFAULT_OUT_DIR = REPO / "models" / "rt1320_final"

RT600_FEATURE_MANIFEST_SHA256 = (
    "1646c3b9e09d8a7fb3c564483a1d1d999680caeefe848b092f907a6cac80cced"
)
FOLDS_FINAL10K_SHA256 = (
    "bf0cdf642bde018a632663ae7d211714a173fb64ef824b15416caf2649e0c716"
)
SCDF_TIME_COORD = "log_n_seen"
N_FEATURES = 500

RT1257_FILES = [
    "model.cbm.0",
    "model.txt.1",
    "model.txt.2",
    "model.txt.3",
    "model.cbm.4",
    "model.txt.5",
    "model.txt.6",
]
STUDENT_MODEL = "model.txt.7"
STUDENT_CAL = "RT-1320_student_scdf.json"

STUDENT_STREAM = {
    "slot": 7,
    "member": "M1 Arm-C residual student",
    "id": "RT-1320",
    "kind": "lightgbm",
    "objective": "regression",
    "replaces": None,
    "modules": [
        "m00_core", "m01_seq", "m02_dist", "m03_dyn",
        "m04_resid", "m06_loc", "m07_bayes",
    ],
    "target": "fold-pure Arm-C teacher residual on folds_final10k",
    "calibration": "SCDF_NSEEN over raw regression output",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rt1257-dir", type=Path, default=DEFAULT_RT1257_DIR)
    p.add_argument("--fit-dir", type=Path, default=DEFAULT_FIT_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--check-base",
        action="store_true",
        help="validate the 7-member base and report what the fit still owes, "
        "then exit without writing anything",
    )
    p.add_argument("--force", action="store_true", help="overwrite an existing --out")
    return p.parse_args()


def load_base(rt1257_dir: Path) -> dict[str, Any]:
    manifest_path = rt1257_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"no RT-1257 manifest at {manifest_path}")
    base = json.loads(manifest_path.read_text(encoding="utf-8"))

    problems: list[str] = []
    if base.get("feature_manifest_sha256") != RT600_FEATURE_MANIFEST_SHA256:
        problems.append("base feature_manifest_sha256 is not the RT-600 bank")
    if base.get("folds_sha256") != FOLDS_FINAL10K_SHA256:
        problems.append("base folds_sha256 is not folds_final10k")
    for key, want in (("model_files", 7), ("booster_columns", 7), ("streams", 7)):
        if len(base.get(key) or []) != want:
            problems.append(f"base {key} has {len(base.get(key) or [])}, expected {want}")
    if len((base.get("calibration") or {}).get("models") or []) != 7:
        problems.append("base calibration.models is not 7 long")
    if (base.get("calibration") or {}).get("time_coord") != SCDF_TIME_COORD:
        problems.append(f"base calibration time_coord is not {SCDF_TIME_COORD}")

    # The base model files must match the hashes the base manifest recorded, or
    # we would be extending an artifact that has already drifted.
    hashes = base.get("model_hashes") or {}
    for name in RT1257_FILES:
        src = rt1257_dir / name
        if not src.exists():
            problems.append(f"missing base member file {name}")
            continue
        recorded = hashes.get(name)
        if recorded and recorded != sha256_file(src):
            problems.append(f"{name} does not match manifest model_hashes")

    if problems:
        raise SystemExit("RT-1257 base is not usable:\n  - " + "\n  - ".join(problems))
    return base


def load_fit_outputs(fit_dir: Path) -> tuple[Path, dict[str, Any], list[str]]:
    missing: list[str] = []
    model_path = fit_dir / STUDENT_MODEL
    cal_path = fit_dir / STUDENT_CAL
    if not model_path.exists():
        missing.append(f"{STUDENT_MODEL}   (produced by --stage final)")
    if not cal_path.exists():
        missing.append(f"{STUDENT_CAL}   (produced by --stage calib)")
    if missing:
        return model_path, {}, missing

    payload = json.loads(cal_path.read_text(encoding="utf-8"))
    if payload.get("kind") != "scdf":
        raise SystemExit(f"{cal_path}: calibration kind is {payload.get('kind')!r}")
    if payload.get("time_coord") != SCDF_TIME_COORD:
        raise SystemExit(f"{cal_path}: time_coord is {payload.get('time_coord')!r}")
    model = payload.get("model") or {}
    if sorted(model.keys()) != ["anchors", "grids", "time_coord"]:
        raise SystemExit(f"{cal_path}: unexpected SCDF payload keys {sorted(model)}")
    return model_path, payload, []


def verify_student_booster(model_path: Path) -> int:
    """The 8th member must consume the same 500-column bank as the other seven."""
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(model_path))
    n = int(booster.num_feature())
    if n != N_FEATURES:
        raise SystemExit(
            f"{model_path} uses {n} features, expected {N_FEATURES} -- it was not "
            "trained on the frozen causal bank"
        )
    return n


def build_manifest(base: dict[str, Any], cal_payload: dict[str, Any],
                   out_dir: Path) -> dict[str, Any]:
    m = copy.deepcopy(base)

    m["experiment_id"] = "RT-1320"
    m["what"] = "RT-1257 plus one Arm-C residual-student member"
    m["member_contract"] = "addition of an 8th exchangeable member to RT-1257"

    m["model_files"] = list(base["model_files"]) + [
        {"slot": 7, "kind": "lightgbm", "path": STUDENT_MODEL, "format": "txt"}
    ]
    # The student reads the whole 500-column bank, same as the RT-1257 members.
    m["booster_columns"] = list(base["booster_columns"]) + [list(range(N_FEATURES))]
    m["streams"] = list(base["streams"]) + [copy.deepcopy(STUDENT_STREAM)]
    m["composition"] = list(base["composition"]) + ["RT-1320"]

    cal = copy.deepcopy(base["calibration"])
    cal["models"] = list(cal["models"]) + [copy.deepcopy(cal_payload["model"])]
    m["calibration"] = cal

    prov = dict(base.get("expected_provenance") or {})
    prov["n_boosters"] = 8
    prov["folds_sha256"] = FOLDS_FINAL10K_SHA256
    m["expected_provenance"] = prov

    m["model_hashes"] = {
        name: sha256_file(out_dir / name) for name in RT1257_FILES + [STUDENT_MODEL]
    }
    m["rt1320_student"] = {
        "source_scdf_sha256": cal_payload.get("sha256"),
        "source_oof_sha256": cal_payload.get("source_oof_sha256"),
        "fitted_rows": cal_payload.get("fitted_rows"),
    }
    m["built_by"] = "research/scripts/rt1320_assemble_artifact.py"
    return m


def validate_final(m: dict[str, Any], out_dir: Path) -> None:
    """Re-run every check rt1320_promotion_prep.inspect_model_manifest performs."""
    ids = {str(s.get("id")) for s in m["streams"] if isinstance(s, dict)}
    expected_paths = {spec["path"] for spec in m["model_files"]}
    checks = {
        "experiment_id_is_rt1320": m.get("experiment_id") == "RT-1320",
        "feature_manifest_is_rt600_bank":
            m.get("feature_manifest_sha256") == RT600_FEATURE_MANIFEST_SHA256,
        "has_8_model_files": len(m["model_files"]) == 8,
        "has_8_booster_slices": len(m["booster_columns"]) == 8,
        "has_8_calibration_models": len(m["calibration"]["models"]) == 8,
        "expected_provenance_n_boosters_is_8":
            (m.get("expected_provenance") or {}).get("n_boosters") == 8,
        "trained_on_final10k_10000":
            (m.get("trained_on") or {}).get("partition") == "folds_final10k"
            and (m.get("trained_on") or {}).get("n_series") == 10000,
        "folds_sha256_final10k": m.get("folds_sha256") == FOLDS_FINAL10K_SHA256,
        "member_ids_include_rt1257_and_rt1320":
            {"RT-1255", "RT-1254", "RT-1320"}.issubset(ids),
        "expected_model_file_paths_exist":
            all((out_dir / p).exists() for p in expected_paths),
    }
    failed = [k for k, v in checks.items() if not v]
    for name, ok in checks.items():
        print(f"  {'OK ' if ok else 'FAIL'} {name}")
    if failed:
        raise SystemExit(f"\nartifact is not valid; failed: {failed}")


def main() -> int:
    args = parse_args()
    rt1257_dir = args.rt1257_dir.expanduser().resolve()
    fit_dir = args.fit_dir.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()

    print(f"base (7 members) : {rt1257_dir}")
    print(f"RT-1320 fit      : {fit_dir}")
    print(f"output           : {out_dir}\n")

    base = load_base(rt1257_dir)
    print(f"base OK -- RT-1257, {len(base['model_files'])} members, "
          f"{len(base['calibration']['models'])} calibration models")

    model_path, cal_payload, missing = load_fit_outputs(fit_dir)
    if missing:
        print("\nthe RT-1320 fit has not produced everything yet:")
        for item in missing:
            print(f"  -- {item}")
        if args.check_base:
            print("\n--check-base: the 7-member base is ready; waiting on the above.")
            return 0
        raise SystemExit("\ncannot assemble until both files exist.")

    if args.check_base:
        print("\n--check-base: base and fit outputs both present; run without the "
              "flag to assemble.")
        return 0

    if out_dir.exists():
        if not args.force:
            raise SystemExit(f"{out_dir} exists; pass --force to overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    for name in RT1257_FILES:
        shutil.copy2(rt1257_dir / name, out_dir / name)
    shutil.copy2(model_path, out_dir / STUDENT_MODEL)
    shutil.copy2(fit_dir / STUDENT_CAL, out_dir / STUDENT_CAL)
    print(f"copied {len(RT1257_FILES)} base members + {STUDENT_MODEL}")

    n_feat = verify_student_booster(out_dir / STUDENT_MODEL)
    print(f"{STUDENT_MODEL} loads, num_feature={n_feat}")

    manifest = build_manifest(base, cal_payload, out_dir)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\nvalidating against inspect_model_manifest's contract:")
    validate_final(manifest, out_dir)

    print(f"\nwrote {out_dir}/manifest.json")
    print(f"  model.txt.7 sha256 {manifest['model_hashes'][STUDENT_MODEL]}")
    print("\nnext:")
    print("  # causality gate against the student's own inference path")
    print(f"  python research/scripts/rt1320_promotion_prep.py validate-artifact "
          f"--model-dir {out_dir}")
    print("  # then the deployability audit, which reads that result")
    print(f"  python research/scripts/rt1320_promotion_prep.py audit "
          f"--model-dir {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
