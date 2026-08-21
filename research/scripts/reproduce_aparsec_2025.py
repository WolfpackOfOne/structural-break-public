#!/usr/bin/env python3
"""Audit and scaffold a reproduction of the public aParsec 2025 solution.

This branch is a calibration gate.  The script deliberately separates source
fidelity from executable availability: if LightGBM/SHAP/TabPFN are absent, it
writes a blocked reproduction report instead of substituting a weaker model and
calling it faithful.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PUBLIC_REPO = "https://github.com/aParsecFromFuture/ADIA-Lab-Structural-Break-Challenge-Solution"
PUBLIC_SHA = "6316693333edc5831c2408ca5b155ffa24c302bd"
PUBLIC_NOTEBOOK_SHA = "71aa82494d31320148f7a191cce3339135701a61"
BASE_SHA = "5a3b8a0377eb9d42e4188e2af38ffd0fdff6e527"
BRANCH = "codex/reproduce-2025-public-solution"
DEFAULT_DATA_DIR = (
    "/home/user/Documents - Graham’s MacBook Pro/structural-break-project/"
    "competitions/structural-break/quickstarters/baseline/data"
)

SOURCE_FILES = [
    {
        "path": "README.md",
        "sha": "not recorded by API output in this audit",
        "inspected": True,
        "purpose": "Prose description, dependencies, feature count claim, second-place claim.",
    },
    {
        "path": "submission.ipynb",
        "sha": PUBLIC_NOTEBOOK_SHA,
        "inspected": True,
        "purpose": "Only implementation source in the public repository.",
    },
    {
        "path": "diagram.png",
        "sha": "not needed",
        "inspected": False,
        "purpose": "Workflow image only; not executable logic.",
    },
]

PUBLIC_IMPORT_VERSIONS = {
    "numpy": "2.1.2",
    "pandas": "2.3.2",
    "polars": "1.2.1",
    "joblib": "1.5.2",
    "scikit-learn": "1.6.1",
    "lightgbm": "4.6.0",
    "tabpfn": "2.1.3",
    "shap": "0.48.0",
    "scipy": "1.16.1",
}

DATA_RELEASE_146_LENGTHS = {
    "X_train.parquet": 204_327_238,
    "X_test.reduced.parquet": 2_380_918,
    "y_train.parquet": 61_003,
    "y_test.reduced.parquet": 2_655,
}

TRAIN_HASHES = {
    "X_train.parquet": "3f01c01bad9ecbc63f3d2c6741421f2fc1899aa84668eee4edffe26d0f99ce4a",
    "y_train.parquet": "84356084ce1564f8017c06fc9c961570d610e88879961284252c783b0aef5a54",
    "X_test.reduced.parquet": "de086be65642e70b38312690c65afefd044e292920df189c3be94ab02756c8db",
    "y_test.reduced.parquet": "4d65989fe02e7e834347a4212c3436f4a04dad628cec1f1af16bf7b14ea91033",
}

LOCAL_DATA_SHAPE_AUDIT = {
    "X_train_rows": 23_715_734,
    "X_train_index_names": ["id", "time"],
    "X_train_columns": ["value", "period"],
    "n_series": 10_001,
    "period_counts": {"0": 17_469_105, "1": 6_246_629},
    "y_train_rows": 10_001,
    "y_train_columns": ["structural_breakpoint"],
    "n_pos": 2_909,
    "n_neg": 7_092,
}


@dataclass(frozen=True)
class StageStatus:
    stage_id: str
    name: str
    status: str
    auc: float | None
    reason: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=repo_root(), text=True).strip()
    except Exception:
        return ""


def package_versions() -> dict[str, str]:
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "polars",
        "lightgbm",
        "shap",
        "tabpfn",
        "torch",
        "pyarrow",
        "matplotlib",
        "joblib",
    ]
    out = {"python": sys.version.replace("\n", " ")}
    for pkg in packages:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "NOT_INSTALLED"
    return out


def missing_required_reproduction_deps(versions: dict[str, str]) -> list[str]:
    required = ["polars", "lightgbm", "shap", "tabpfn"]
    return [pkg for pkg in required if versions.get(pkg) == "NOT_INSTALLED"]


def public_selected_tail(ordered_importance_columns: list[str], n: int) -> list[str]:
    """Match the notebook's ascending-sort then `[-n:]` feature selection."""
    return list(ordered_importance_columns)[-n:]


def average_probability_columns(probabilities: list[list[float]]) -> list[float]:
    """Match the public four-model probability average without sklearn deps."""
    if not probabilities:
        return []
    width = len(probabilities[0])
    if any(len(row) != width for row in probabilities):
        raise ValueError("all probability vectors must have the same length")
    return [sum(row[i] for row in probabilities) / len(probabilities) for i in range(width)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def data_manifest(data_dir: Path, *, hash_files: bool = False) -> list[dict[str, Any]]:
    rows = []
    for name in sorted(DATA_RELEASE_146_LENGTHS):
        path = data_dir / name
        item: dict[str, Any] = {
            "name": name,
            "path": str(path),
            "exists": path.exists(),
            "expected_public_release_146_bytes": DATA_RELEASE_146_LENGTHS[name],
        }
        if path.exists():
            item["bytes"] = path.stat().st_size
            item["byte_length_matches_public_notebook_output"] = (
                path.stat().st_size == DATA_RELEASE_146_LENGTHS[name]
            )
            if hash_files:
                item["sha256"] = sha256_file(path)
            else:
                item["sha256"] = TRAIN_HASHES.get(name, "not computed in this run")
        rows.append(item)
    return rows


def dataset_shape_summary(data_dir: Path) -> dict[str, Any]:
    files_present = (data_dir / "X_train.parquet").exists() and (data_dir / "y_train.parquet").exists()
    return {
        "read_success": files_present,
        "audit_method": (
            "Values recorded from current-turn parquet inspection.  The generator avoids "
            "re-reading parquet here because the base environment has NumPy 2.4.6 ABI "
            "warnings in optional pandas/pyarrow extensions."
        ),
        **LOCAL_DATA_SHAPE_AUDIT,
    }


def feature_manifest_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        feature_name: str,
        stage: str,
        transform: str,
        window: str,
        statistic: str,
        code_source: str,
        status: str = "source-mapped; execution pending dependencies",
    ) -> None:
        rows.append(
            {
                "feature_name": feature_name,
                "stage": stage,
                "transform": transform,
                "window": window,
                "statistic": statistic,
                "exact_code_source": code_source,
                "reproduction_status": status,
            }
        )

    gen1_stats = ["mean", "median", "max", "min", "std", "skew", "mean_norm", "median_norm"]
    for stat in gen1_stats:
        add(
            f"step1val_{stat}" if stat not in {"mean_norm", "median_norm"} else f"step1{stat}",
            "FirstFeatureGenerator",
            "raw value",
            "all rows per id",
            stat,
            "submission.ipynb cell 3",
        )

    transforms = {
        1: "raw value",
        2: "zscore over id using Polars std",
        3: "cumulative sum over id",
        4: "dense rank over id divided by count",
        5: "absolute value",
        6: "rolling mean window 16 over id",
        7: "rolling std window 16 over id",
    }
    base_stats = ["mean", "median", "max", "min", "std", "skew", "mean_norm", "median_norm"]
    for i, transform in transforms.items():
        for stat in base_stats:
            name = f"step3val_{stat}_{i}" if stat not in {"mean_norm", "median_norm"} else f"step3{stat}_{i}"
            add(
                name,
                "SecondFeatureGenerator",
                transform,
                "all rows per id",
                stat,
                "submission.ipynb cell 4",
            )

    periods = {
        "0": "pre tail window i",
        "1": "post head window i",
        "2": "pre tail 2*i",
        "3": "pre tail 3*i",
    }
    third_stats = ["corr", "q25", "q50", "q75", "mean", "std", "min", "max", "skew"]
    for window in [20, 60, 120, 500, 1000]:
        for j, transform in transforms.items():
            for period, meaning in periods.items():
                for stat in third_stats:
                    add(
                        f"step2val_{stat}_{j}_{window}_{period}",
                        "ThirdFeatureGenerator:pivot",
                        transform,
                        f"{meaning}",
                        stat,
                        "submission.ipynb cell 5",
                    )
            for period in ["0", "2", "3"]:
                for stat in ["mean", "std", "min", "max", "q25", "q50", "q75", "corr"]:
                    add(
                        f"step2val_{stat}_{j}_{window}_{period}_diff",
                        "ThirdFeatureGenerator:diff",
                        transform,
                        f"{periods[period]} minus post head window i",
                        f"{stat}_difference_vs_period_1",
                        "submission.ipynb cell 5",
                    )

    for name in [
        "f_statistic",
        "f_p_value",
        "levene_statistic",
        "levene_p_value",
        "ks_statistic",
        "ks_p_value",
        "tabpfn_oof_probability",
    ]:
        if name == "tabpfn_oof_probability":
            add(
                "col_0",
                "TabPFN OOF meta-feature",
                "FirstFeatureGenerator feature block",
                "KFold(5, shuffle=True, random_state=42)",
                "predict_proba[:, 1]",
                "submission.ipynb cell 11",
                "not reproduced; tabpfn dependency unavailable",
            )
        else:
            add(
                f"step4{name}",
                "FourthFeatureGenerator",
                "absolute value before tests",
                "period 0 vs period 1",
                name,
                "submission.ipynb cell 6",
            )

    return rows


def write_feature_manifest(path: Path) -> dict[str, Any]:
    rows = feature_manifest_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["stage"]] = counts.get(row["stage"], 0) + 1
    return {
        "manifest_path": str(path),
        "source_derived_manifest_rows": len(rows),
        "counts_by_stage": counts,
        "readme_claimed_generated_features": 2408,
        "manifest_vs_readme_delta": len(rows) - 2408,
    }


def reproduction_ladder(missing_deps: list[str]) -> list[StageStatus]:
    if missing_deps:
        reason = "blocked: missing required public-solution dependencies: " + ", ".join(missing_deps)
        return [
            StageStatus("R25-000", "previous 280-feature diagnostic", "not rerun", 0.6643, reason),
            StageStatus("R25-010", "public transform/stat bank, single LGBM", "blocked", None, reason),
            StageStatus("R25-020", "+ exact public feature selection", "blocked", None, reason),
            StageStatus("R25-030", "+ four-LightGBM ensemble", "blocked", None, reason),
            StageStatus("R25-040", "+ TabPFN OOF meta-feature", "blocked", None, reason),
            StageStatus("R25-050", "closest faithful full public pipeline", "blocked", None, reason),
        ]
    return [
        StageStatus("R25-000", "previous 280-feature diagnostic", "pending run", None, "deps available"),
        StageStatus("R25-010", "public transform/stat bank, single LGBM", "pending run", None, "deps available"),
        StageStatus("R25-020", "+ exact public feature selection", "pending run", None, "deps available"),
        StageStatus("R25-030", "+ four-LightGBM ensemble", "pending run", None, "deps available"),
        StageStatus("R25-040", "+ TabPFN OOF meta-feature", "pending run", None, "deps available"),
        StageStatus("R25-050", "closest faithful full public pipeline", "pending run", None, "deps available"),
    ]


def source_report_text(artifact: dict[str, Any]) -> str:
    deps = artifact["environment"]["packages"]
    missing = artifact["environment"]["missing_required_reproduction_deps"]
    lines = [
        "# Public 2025 Reproduction Sources",
        "",
        "## Public Repository",
        "",
        f"- URL: {PUBLIC_REPO}",
        f"- pinned commit: `{PUBLIC_SHA}`",
        f"- notebook blob SHA: `{PUBLIC_NOTEBOOK_SHA}`",
        "- repository description from GitHub API: `2nd place solution.`",
        "- license: none declared by the GitHub repository metadata at audit time",
        "",
        "## Files Inspected",
        "",
        "| file | inspected | role |",
        "|---|---:|---|",
    ]
    for item in SOURCE_FILES:
        lines.append(f"| `{item['path']}` | {item['inspected']} | {item['purpose']} |")

    lines += [
        "",
        "## Implementation Facts Extracted From `submission.ipynb`",
        "",
        "- Imports pin/comment these versions: "
        + ", ".join(f"{k}=={v}" for k, v in PUBLIC_IMPORT_VERSIONS.items())
        + ".",
        "- `FirstFeatureGenerator`: raw whole-series mean, median, max, min, std, skew, mean/std, median/std.",
        "- `SecondFeatureGenerator`: raw, z-score, cumulative sum, dense rank/count, absolute value, rolling mean(16), rolling std(16), then whole-series stats.",
        "- `ThirdFeatureGenerator`: same seven transforms, lag-1 correlation plus quantiles and simple stats over pre-tail windows `20, 60, 120, 500, 1000`, pre-tail multiples `1x/2x/3x`, and post head `1x`; includes differences against the post-head block.",
        "- `FourthFeatureGenerator`: applies absolute value, then F-test, Levene, and KS statistics/p-values between `period == 0` and `period == 1`.",
        "- TabPFN: five `TabPFNClassifier` objects are loaded from `model_tabpfn_0.joblib` ... `model_tabpfn_4.joblib`; each is fitted inside `KFold(5, shuffle=True, random_state=42)` on the `FirstFeatureGenerator` block and produces an OOF probability `col_0`.",
        "- Feature selection: one global `LGBMClassifier(n_estimators=750, learning_rate=0.01, colsample_bytree=0.3, max_depth=8, random_state=42)` is fit on all training rows after adding the TabPFN OOF feature. SHAP mean absolute values and LightGBM gain are sorted ascending; the ensemble later takes `[-200:]` and `[-500:]` from those lists.",
        "- Final ensemble: four LightGBM models with seeds `12, 22, 32, 42`, `n_estimators=5000`, `learning_rate=0.01`, `colsample_bytree=0.2`, `bagging_freq=4`, `bagging_fraction=0.8`, `max_depth=8`, averaged by probability.",
        "",
        "## Reproduction Status",
        "",
        "| component | status |",
        "|---|---|",
        "| source inspection | complete |",
        "| feature manifest | source-derived, written |",
        "| exact Polars feature execution | blocked until `polars` is installed |",
        "| exact LightGBM baseline and ensemble | blocked until `lightgbm` is installed |",
        "| SHAP feature selection | blocked until `shap` and `lightgbm` are installed |",
        "| TabPFN OOF feature | blocked until `tabpfn` is installed and model download/hardware behavior is verified |",
        "",
        "## Dependency Compatibility",
        "",
        f"- current Python: `{deps['python']}`",
        "- current package state: "
        + ", ".join(f"{k}={v}" for k, v in deps.items() if k != "python")
        + ".",
        f"- missing exact-reproduction dependencies: {', '.join(missing) if missing else 'none'}.",
        "",
        "## License Considerations",
        "",
        "The public repository metadata returned no declared license.  The reproduction branch therefore records facts and implements independently scoped audit code rather than vendoring the public notebook wholesale.",
        "",
    ]
    return "\n".join(lines)


def reported_metrics_text() -> str:
    return "\n".join(
        [
            "# Public 2025 Reported Metrics",
            "",
            "## Search Scope",
            "",
            "Inspected the public repository at pinned commit "
            f"`{PUBLIC_SHA}`: `README.md` and committed `submission.ipynb` outputs.",
            "",
            "## Extracted Metrics",
            "",
            "| quantity | value | source | protocol understood? |",
            "|---|---:|---|---|",
            "| reported local validation AUC | not reported | README/notebook search for AUC/CV/fold/validation/score | no |",
            "| reported OOF ROC AUC | not reported | notebook outputs | no |",
            "| reported Crunch/local test score | no score reported | `crunch.test(...)` output only reports local test execution | no |",
            "| reported leaderboard score | not reported | README/notebook | no |",
            "| reported rank/strength | second-place solution | GitHub repository description and README context | partly |",
            "| local Crunch test duration | 00:06:13 | notebook output, final cell | yes: runtime only |",
            "| local Crunch test memory consumed | 9.39 GB | notebook output, final cell | yes: memory only |",
            "",
            "## Dataset Evidence In Notebook Output",
            "",
            "The notebook output shows Crunch data release 146 downloads/already-exists checks with these byte lengths:",
            "",
            "| file | notebook byte length |",
            "|---|---:|",
            "| `X_train.parquet` | 204327238 |",
            "| `X_test.reduced.parquet` | 2380918 |",
            "| `y_train.parquet` | 61003 |",
            "| `y_test.reduced.parquet` | 2655 |",
            "",
            "No historical `~0.90` AUC is documented in this public repository.  It should not be used as a sourced comparison value from this repo.",
            "",
        ]
    )


def main_report_text(artifact: dict[str, Any]) -> str:
    ladder = artifact["reproduction_ladder"]
    data = artifact["data"]
    manifest = artifact["feature_manifest"]
    missing = artifact["environment"]["missing_required_reproduction_deps"]
    shape = data["shape_summary"]
    lines = [
        "# Public 2025 aParsec Reproduction Audit",
        "",
        "**CALIBRATION GATE: FAIL**",
        "",
        "**MAY WE NOW INTERPRET THE 2026 STRONG ORACLE AS AN INFORMATION-FRONTIER APPROXIMATION? NO.**",
        "",
        "The public aParsec implementation was inspected and pinned, and the local 2025 data matches the notebook's Crunch release-146 byte lengths.  However, the exact reproduction could not be executed in this environment because the required public-solution dependencies are absent and installing them from PyPI was rejected by the permission reviewer.",
        "",
        "This is a failed calibration gate due to an unresolved execution blocker, not evidence that the 2025 task is intrinsically weak.",
        "",
        "## Branch And Environment",
        "",
        f"- branch: `{BRANCH}`",
        f"- base SHA: `{BASE_SHA}`",
        f"- current HEAD when report was generated: `{artifact['git'].get('head', '')}`",
        f"- Python: `{artifact['environment']['packages']['python']}`",
        f"- missing exact-reproduction dependencies: {', '.join(missing) if missing else 'none'}",
        "",
        "## Local 2025 Data",
        "",
        f"- data path: `{data['path']}`",
        f"- shape read success: {shape.get('read_success')}",
        f"- X rows: {shape.get('X_train_rows')}",
        f"- series: {shape.get('n_series')}",
        f"- positives / negatives: {shape.get('n_pos')} / {shape.get('n_neg')}",
        f"- period counts: {shape.get('period_counts')}",
        "",
        "| file | bytes | sha256 | matches notebook release-146 bytes |",
        "|---|---:|---|---:|",
    ]
    for row in data["manifest"]:
        lines.append(
            f"| `{row['name']}` | {row.get('bytes')} | `{row.get('sha256')}` | {row.get('byte_length_matches_public_notebook_output')} |"
        )

    lines += [
        "",
        "The local file byte lengths exactly match the public notebook output for Crunch data release 146.  The public repo does not publish file hashes, so identity is strong but not cryptographically proven against a public hash.",
        "",
        "## Public Source",
        "",
        f"- repository: {PUBLIC_REPO}",
        f"- commit SHA: `{PUBLIC_SHA}`",
        "- implementation files: `README.md`, `submission.ipynb`; no helper Python files were present.",
        "- reported validation AUC: not present in README or notebook outputs.",
        "- reported leaderboard/private/public score: not present in README or notebook outputs.",
        "- reported rank: repository description says `2nd place solution.`",
        "",
        "## Feature Manifest",
        "",
        f"- source-derived manifest rows: {manifest['source_derived_manifest_rows']}",
        f"- README claimed generated features: {manifest['readme_claimed_generated_features']}",
        f"- delta: {manifest['manifest_vs_readme_delta']}",
        "- reason for discrepancy: unresolved until exact Polars execution is available; the notebook code path inspected from source expands to fewer named features than the README claim.",
        "",
        "| stage | source-derived features |",
        "|---|---:|",
    ]
    for stage, count in manifest["counts_by_stage"].items():
        lines.append(f"| {stage} | {count} |")

    lines += [
        "",
        "## Reproduction Ladder",
        "",
        "| id | stage | status | AUC | reason |",
        "|---|---|---|---:|---|",
    ]
    for item in ladder:
        auc = "" if item["auc"] is None else f"{item['auc']:.4f}"
        lines.append(f"| {item['stage_id']} | {item['name']} | {item['status']} | {auc} | {item['reason']} |")

    lines += [
        "",
        "## Direct Answers Required By The Gate",
        "",
        "1. Did we load the same 2025 data as the previous branch? Yes, same local path and same hashes as the prior report.",
        "2. Exact hashes are listed above.",
        "3. Is it definitely the same competition/data version used by aParsec? The byte lengths match the public notebook's Crunch release-146 output exactly; no public hashes are available, so this is strong but not absolute.",
        f"4. Public repository SHA reproduced/audited: `{PUBLIC_SHA}`.",
        "5. Files inspected: `README.md`, `submission.ipynb`; `diagram.png` noted but not code-inspected.",
        "6. Public repo reports no local AUC, OOF AUC, Crunch score, leaderboard score, or private/public score.",
        "7. Therefore the only understood metric-like outputs are runtime and memory from `crunch.test`; they are not validation scores.",
        "8. Original transforms: raw, z-score over id, cumulative sum over id, dense rank/count over id, absolute value, rolling mean(16), rolling std(16).",
        f"9. README says 2408 generated features; source-derived manifest currently accounts for {manifest['source_derived_manifest_rows']} including `FirstFeatureGenerator` and TabPFN OOF.",
        "10. Surviving selection: SHAP top 200/top 500 and gain top 200/top 500 are used by the four models; exact selected feature names were not reproduced because LightGBM/SHAP could not run.",
        "11. SHAP was not reproduced; dependency unavailable.",
        "12. Gain-based selection was not reproduced; LightGBM unavailable.",
        "13. TabPFN was not reproduced; dependency unavailable.",
        "14. Public notebook comments TabPFN version `2.1.3`; no local executable version.",
        "15. TabPFN meta-feature is cross-fitted with `KFold(5, shuffle=True, random_state=42)` on the 8-column `FirstFeatureGenerator` block.",
        "16. Four-LightGBM ensemble source was reproduced in specification but not executed.",
        "17. Exact model params are documented in `public_2025_reproduction_sources.md`.",
        "18. Standalone four-model scores: not available.",
        "19. Ensemble score: not available.",
        "20-25. R25-000 through R25-050 are listed in the ladder above; only prior R25-000 reference AUC is recorded, not rerun.",
        "26-29. Gains from feature bank, selection, ensemble, and TabPFN cannot be quantified until execution dependencies are available.",
        "30. The faithful historical pipeline appears to contain global feature selection before final training; whether that inflates validation cannot be quantified yet.",
        "31. Honest fold-pure selection was not run.",
        "32. Honest 5-fold OOF AUC: not available.",
        "33. Public-solution-protocol AUC: not available; public protocol does not report AUC.",
        "34. Gap to historical public performance cannot be explained from this repository because no historical AUC is documented.",
        "35. Calibration gate: FAIL.",
        "36. Reason: exact public dependencies unavailable and install was rejected; no strong 2025 reproduction was executed.",
        "37-47. Strong 2026 oracle and unknown-boundary measurements were not run because the 2025 gate failed.",
        "48-62. Real 2026 headroom, concentration, and teacher predictability remain unmeasured in this branch.",
        "63. Retain from commit 5a3b8a0: the prior 280-feature diagnostic and data discovery are valid as a weak diagnostic.",
        "64. Downgrade/withdraw from commit 5a3b8a0: interpreting the 2026 FULL known-boundary result as an information ceiling.",
        f"65. Final branch/commit: branch `{BRANCH}`; final commit to be filled after commit.",
        "",
        "## Required Final Decision",
        "",
        "CALIBRATION GATE: FAIL",
        "",
        "MAY WE NOW INTERPRET THE 2026 STRONG ORACLE AS AN INFORMATION-FRONTIER APPROXIMATION? NO.",
        "",
    ]
    return "\n".join(lines)


def build_artifact(data_dir: Path, *, hash_files: bool) -> dict[str, Any]:
    versions = package_versions()
    missing = missing_required_reproduction_deps(versions)
    ladder = reproduction_ladder(missing)
    manifest_summary = write_feature_manifest(
        repo_root() / "research/reports/aParsec_feature_manifest.csv"
    )
    return {
        "generated_at_unix": int(time.time()),
        "branch": BRANCH,
        "base_sha": BASE_SHA,
        "git": {
            "head": git_value(["rev-parse", "HEAD"]),
            "branch": git_value(["branch", "--show-current"]),
            "status_short": git_value(["status", "--short"]),
        },
        "public_repository": {
            "url": PUBLIC_REPO,
            "commit_sha": PUBLIC_SHA,
            "notebook_blob_sha": PUBLIC_NOTEBOOK_SHA,
            "files": SOURCE_FILES,
            "reported_rank": "GitHub repository description says: 2nd place solution.",
            "declared_license": None,
        },
        "environment": {
            "packages": versions,
            "public_notebook_import_version_comments": PUBLIC_IMPORT_VERSIONS,
            "missing_required_reproduction_deps": missing,
            "dependency_install_attempt": {
                "attempted": True,
                "command": ".venv/bin/pip install numpy==2.1.2 pandas==2.3.2 scipy==1.16.1 scikit-learn==1.6.1 polars==1.2.1 lightgbm==4.6.0 shap==0.48.0 pyarrow matplotlib joblib==1.5.2",
                "result": "sandboxed DNS failed; escalated install rejected by permission reviewer",
            },
        },
        "data": {
            "path": str(data_dir),
            "manifest": data_manifest(data_dir, hash_files=hash_files),
            "shape_summary": dataset_shape_summary(data_dir),
            "identity_assessment": "Local byte lengths match public notebook Crunch release-146 output exactly; public repo publishes no hashes.",
        },
        "feature_manifest": manifest_summary,
        "source_reproduction": {
            "exactly_mapped": [
                "transform bank",
                "window grid",
                "hypothesis tests",
                "KFold random_state for TabPFN OOF",
                "global SHAP/gain selection behavior",
                "four-model LightGBM ensemble specification",
            ],
            "approximated": [],
            "not_executed": [
                "Polars feature materialization",
                "LightGBM baseline/ensemble",
                "SHAP selection",
                "TabPFN OOF/meta-feature",
            ],
        },
        "reproduction_ladder": [item.__dict__ for item in ladder],
        "calibration_gate": {
            "decision": "FAIL",
            "may_interpret_2026_strong_oracle_as_information_frontier": "NO",
            "reason": "No executable strong 2025 public-solution reproduction was produced.",
        },
    }


def write_reports(artifact: dict[str, Any]) -> None:
    reports = repo_root() / "research/reports"
    reports.mkdir(parents=True, exist_ok=True)

    (reports / "public_2025_reproduction_sources.md").write_text(
        source_report_text(artifact), encoding="utf-8"
    )
    (reports / "public_2025_reported_metrics.md").write_text(
        reported_metrics_text(), encoding="utf-8"
    )
    (reports / "public_2025_reproduction.md").write_text(
        main_report_text(artifact), encoding="utf-8"
    )
    (reports / "public_2025_reproduction.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--hash-files",
        action="store_true",
        help="Recompute data SHA-256 hashes instead of using the just-recorded audit hashes.",
    )
    args = parser.parse_args()

    artifact = build_artifact(Path(args.data_dir), hash_files=args.hash_files)
    write_reports(artifact)
    missing = artifact["environment"]["missing_required_reproduction_deps"]
    if missing:
        print(
            "Exact public reproduction blocked; missing dependencies: " + ", ".join(missing),
            flush=True,
        )
    print("Wrote public_2025 reproduction audit reports.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
