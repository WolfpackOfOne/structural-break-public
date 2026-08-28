#!/usr/bin/env python3
"""LOCAL lane execution for Deep Ensemble Frontier 2026.

This script intentionally keeps CSA-04 training separate from CSA-04 scoring:
all four new CatBoost OOF vectors must exist before any single-slot result is
computed. It reuses the prior CatBoost Specialist utilities for calibration and
pair-flow evaluation, but writes all new artifacts under this worktree.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = REPO.parent / "structural-break-learner-diversity-2026"
DEFAULT_CRF_ROOT = REPO.parent / "structural-break-causal-representation-frontier"
LOCAL_OOF_DIR = REPO / "research" / "oof"
REPORT_DIR = REPO / "research" / "reports" / "deep_ensemble_frontier_2026" / "local"
RESULTS_CSV = REPO / "research" / "RESULTS.csv"

for _p in (REPO / "src", REPO / "research" / "scripts"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

FULL = ["m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes"]
SPECIALISTS = ["RT-300", "RT-410", "RT-411", "RT-412", "RT-413", "RT-414", "RT-415"]
SEED_CLONES = ["RT-401", "RT-402", "RT-403", "RT-404", "RT-405", "RT-406"]
MATCHED_CLONE = "RT-401"
FOLDS = (0, 1, 2, 3, 4)
PAIR_SEED = 20260827
PAIRS_PER_T = 64

OLD_SINGLE_ORDER = ("CAT-413", "CAT-300", "CAT-410")
NEW_SINGLE_ORDER = ("CAT-411", "CAT-412", "CAT-414", "CAT-415")
ALL_SINGLE_ORDER = OLD_SINGLE_ORDER + NEW_SINGLE_ORDER

IDS = {
    "CAT-413": "RT-1254",
    "CAT-300": "RT-1255",
    "CAT-410": "RT-1256",
    "CAT-411": "RT-1260",
    "CAT-412": "RT-1261",
    "CAT-414": "RT-1262",
    "CAT-415": "RT-1263",
    "HYBRID": "RT-1264",
}

SPECS = {
    "CAT-413": {
        "replaced": "RT-413",
        "modules": FULL,
        "seed": 0,
        "max_train_rows": 700_000,
        "sample_mode": "uniform",
        "incumbent_objective": "pairwise_t",
    },
    "CAT-300": {
        "replaced": "RT-300",
        "modules": FULL,
        "seed": 0,
        "max_train_rows": 1_000_000,
        "sample_mode": "uniform",
        "incumbent_objective": "binary",
    },
    "CAT-410": {
        "replaced": "RT-410",
        "modules": ["m00_core", "m01_seq", "m07_bayes"],
        "seed": 0,
        "max_train_rows": 900_000,
        "sample_mode": "uniform",
        "incumbent_objective": "binary",
    },
    "CAT-411": {
        "replaced": "RT-411",
        "modules": ["m02_dist", "m03_dyn", "m04_resid", "m06_loc"],
        "seed": 1,
        "max_train_rows": 900_000,
        "sample_mode": "uniform",
        "incumbent_objective": "binary",
    },
    "CAT-412": {
        "replaced": "RT-412",
        "modules": FULL,
        "seed": 7,
        "max_train_rows": 900_000,
        "sample_mode": "per_series",
        "incumbent_objective": "binary_extra_trees_per_series",
    },
    "CAT-414": {
        "replaced": "RT-414",
        "modules": ["m07_bayes", "m06_loc", "m01_seq"],
        "seed": 3,
        "max_train_rows": 700_000,
        "sample_mode": "uniform",
        "incumbent_objective": "binary",
    },
    "CAT-415": {
        "replaced": "RT-415",
        "modules": FULL,
        "seed": 11,
        "max_train_rows": 700_000,
        "sample_mode": "uniform",
        "incumbent_objective": "binary_goss",
        "caveat": "GOSS has no CatBoost analogue; frozen Bernoulli bootstrap is used.",
    },
}

EXPECTED_RT1257_MARGIN = 0.002407204670070273


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", choices=("all", "train", "evaluate", "l3"), default="all")
    p.add_argument("--artifact-root", default=os.environ.get("SBR_ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT)))
    p.add_argument("--crf-root", default=os.environ.get("SBR_CRF_ROOT", str(DEFAULT_CRF_ROOT)))
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--no-ledger", action="store_true")
    return p.parse_args()


def import_csa(artifact_root: Path):
    """Import the prior CSA helpers without letting their CLI parse our flags."""
    old_argv = sys.argv[:]
    old_root = os.environ.get("SBR_ROOT")
    try:
        sys.argv = [old_argv[0], "--artifact-root", str(artifact_root), "--no-ledger"]
        os.environ["SBR_ROOT"] = str(artifact_root)
        import catboost_specialist_2026 as csa
    finally:
        sys.argv = old_argv
        if old_root is None:
            os.environ.pop("SBR_ROOT", None)
        else:
            os.environ["SBR_ROOT"] = old_root

    csa.IDS = dict(IDS)
    csa.SPECS = dict(SPECS)
    csa.REPORT_DIR = REPORT_DIR
    csa.OOF_DIR = LOCAL_OOF_DIR
    csa.SPECIALISTS = list(SPECIALISTS)
    csa.SEED_CLONES = list(SEED_CLONES)
    csa.MATCHED_CLONE = MATCHED_CLONE
    return csa


def git_sha(short: bool = True) -> str:
    args = ["git", "-C", str(REPO), "rev-parse", "--short" if short else "HEAD", "HEAD"]
    return subprocess.check_output(args, text=True).strip()


def copy_oof_if_missing(source_root: Path, name: str) -> dict:
    src = source_root / "research" / "oof" / f"{name}.npy"
    dst = LOCAL_OOF_DIR / f"{name}.npy"
    if not src.exists():
        raise SystemExit(f"missing source OOF {src}")
    LOCAL_OOF_DIR.mkdir(parents=True, exist_ok=True)
    copied = False
    if not dst.exists():
        shutil.copy2(src, dst, follow_symlinks=True)
        copied = True
    return {
        "id": name,
        "source": str(src),
        "resolved_source": str(src.resolve(strict=True)),
        "local": str(dst),
        "bytes": int(dst.stat().st_size),
        "copied": copied,
    }


def ensure_local_oof_inputs(artifact_root: Path) -> list[dict]:
    required = list(dict.fromkeys(SPECIALISTS + SEED_CLONES + ["RT-1254", "RT-1255", "RT-1256"]))
    return [copy_oof_if_missing(artifact_root, name) for name in required]


def ensure_l3_inputs(crf_root: Path) -> list[dict]:
    return [copy_oof_if_missing(crf_root, name) for name in ("RT-1234", "RT-1235")]


def validate_oof(csa, names: list[str], expected_dev_finite: int | None = 4_032_524) -> dict:
    c = csa.make_ctx()
    lockbox = c.d.rows_for([-1])
    out: dict[str, dict] = {}
    for name in names:
        path = LOCAL_OOF_DIR / f"{name}.npy"
        if not path.exists():
            raise SystemExit(f"missing local OOF {path}")
        arr = np.load(path, mmap_mode="r")
        rec = {
            "path": str(path),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "finite_dev_rows": int(np.isfinite(arr[c.dev]).sum()),
            "finite_lockbox_rows": int(np.isfinite(arr[lockbox]).sum()),
        }
        if arr.shape != (len(c.d.y),):
            raise SystemExit(f"{name} wrong shape: {rec}")
        if str(arr.dtype) != "float32":
            raise SystemExit(f"{name} wrong dtype: {rec}")
        if expected_dev_finite is not None and rec["finite_dev_rows"] != expected_dev_finite:
            raise SystemExit(f"{name} wrong finite dev count: {rec}")
        if rec["finite_lockbox_rows"] != 0:
            raise SystemExit(f"{name} fills lockbox rows: {rec}")
        out[name] = rec
    return out


def train_new_arms(csa, force_train: bool) -> dict:
    from catboost import CatBoostClassifier

    LOCAL_OOF_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = REPORT_DIR / "CSA04_TRAIN_LOG.json"
    prior = {}
    if log_path.exists():
        prior = json.loads(log_path.read_text())
    log = {
        "program": "DEEP_ENSEMBLE_FRONTIER_2026",
        "stage": "CSA-04 train",
        "git_sha": git_sha(),
        "score_free_log": True,
        "note": "No TS-AUC, pair-flow, replacement gain, or rank correlation is computed during training.",
        "arms": prior.get("arms", {}),
    }

    for name in NEW_SINGLE_ORDER:
        spec = SPECS[name]
        exp_id = IDS[name]
        out_path = LOCAL_OOF_DIR / f"{exp_id}.npy"
        if out_path.exists() and not force_train:
            print(f"{name}: reusing existing {out_path}", flush=True)
            log["arms"].setdefault(name, {
                "id": exp_id,
                "status": "REUSED",
                "oof_path": str(out_path),
                "folds": [],
                "runtime_s": 0.0,
            })
            log_path.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
            continue

        print(
            f"{name}: start id={exp_id} replaced={spec['replaced']} "
            f"modules={len(spec['modules'])} max_train_rows={spec['max_train_rows']} "
            f"seed={spec['seed']} sample_mode={spec['sample_mode']}",
            flush=True,
        )
        arm_t0 = time.time()
        d = csa.Data()
        mats, names_all = csa.load_features(spec["modules"])
        keep = np.arange(len(names_all), dtype=np.int64)
        oof = np.full(len(d.y), np.nan, dtype=np.float32)
        params = dict(csa.CATBOOST_BASE_PARAMS, random_seed=int(spec["seed"]))
        arm_log = {
            "id": exp_id,
            "name": name,
            "status": "TRAINING",
            "oof_path": str(out_path),
            "training_spec": dict(spec),
            "catboost_params": params,
            "folds": [],
        }
        for fold in FOLDS:
            fold_t0 = time.time()
            train_rows = csa.sample_train_rows(
                d, int(fold), int(spec["max_train_rows"]), int(spec["seed"]), str(spec["sample_mode"])
            )
            valid_rows = d.rows_for([fold])
            x_train = csa._stack(mats, names_all, train_rows, keep)
            y_train = d.y[train_rows]
            model = CatBoostClassifier(**params)
            model.fit(x_train, y_train)
            del x_train, y_train
            gc.collect()

            x_valid = csa._stack(mats, names_all, valid_rows, keep)
            pred = model.predict_proba(x_valid)[:, 1].astype(np.float32)
            del x_valid, model
            gc.collect()
            oof[valid_rows] = pred
            fold_log = {
                "fold": int(fold),
                "train_rows": int(len(train_rows)),
                "valid_rows": int(len(valid_rows)),
                "runtime_s": float(time.time() - fold_t0),
            }
            arm_log["folds"].append(fold_log)
            print(
                f"{name} fold {fold}: train_rows={fold_log['train_rows']} "
                f"valid_rows={fold_log['valid_rows']} runtime_s={fold_log['runtime_s']:.1f}",
                flush=True,
            )
            log["arms"][name] = arm_log
            log_path.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")

        np.save(out_path, oof)
        arm_log["status"] = "TRAINED_NOT_SCORED"
        arm_log["runtime_s"] = float(time.time() - arm_t0)
        log["arms"][name] = arm_log
        log_path.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
        print(f"{name}: wrote {out_path} runtime_s={arm_log['runtime_s']:.1f}", flush=True)
        del mats, names_all, oof, d
        gc.collect()

    return log


def verdict_ladder(marginal: float, positive_folds: int, pair_vs_e0: dict) -> str:
    dom = pair_vs_e0["dominant_cell"]["net_pair_lift"]
    mn = pair_vs_e0["mature_vs_never"]["net_pair_lift"]
    if marginal >= 0.0050:
        return "MAJOR"
    if marginal >= 0.0030 and positive_folds >= 4 and dom > 0 and mn > 0:
        return "SERIOUS"
    if marginal >= 0.0015 and positive_folds >= 4 and dom > 0:
        return "PROMOTION_WORTHY"
    if marginal >= 0.0010:
        return "INTERESTING"
    return "KILL"


def apply_local_verdict(result: dict) -> dict:
    ens = result["ensemble"]
    result["verdict"] = verdict_ladder(
        float(ens["marginal_vs_clone"]),
        int(ens["positive_folds_vs_clone"]),
        result["pair_flow_vs_E0"],
    )
    return result


def evaluate_hybrid_names(csa, c, names: list[str], hybrid_id: str | None = None) -> dict:
    h = csa.evaluate_hybrid(c, names)
    h["k"] = len(names)
    h["survivors_ordered"] = list(names)
    h["id"] = hybrid_id
    return apply_local_verdict(h)


def append_results(rows: list[dict]) -> None:
    if not rows:
        return
    with RESULTS_CSV.open("r+", newline="") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise RuntimeError("RESULTS.csv has no header")
        existing = {r["experiment_id"] for r in reader}
        dup = [r["experiment_id"] for r in rows if r["experiment_id"] in existing]
        if dup:
            raise RuntimeError(f"RESULTS.csv already contains {dup}")
        fh.seek(0, os.SEEK_END)
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def result_rows(result: dict) -> list[dict]:
    base = {
        "date": time.strftime("%Y-%m-%d"),
        "git_sha": result["git_sha"],
        "agent": "codex-deep-ensemble-local",
        "hypothesis": (
            "CSA-04 tests whether frozen RT-1251 CatBoost replacement extends to the "
            "four untested RT600 specialist slots and where the best-k hybrid peaks."
        ),
        "falsification_condition": (
            "KILL if single-slot marginal_vs_clone < +0.0010; hybrid gates follow "
            "the frozen Deep Ensemble Frontier ladder."
        ),
        "folds": "0,1,2,3,4",
        "train_series": "8000",
        "persistence": "none",
        "causal_verified": "inherited frozen causal feature bank; no RT600 score input; no lockbox/test",
        "test_reduced_touched": "no",
        "lockbox_touched": "no",
        "protocol": "deep_ensemble_frontier_2026_csa04_full5",
    }
    rows: list[dict] = []
    for name in NEW_SINGLE_ORDER:
        r = result["specialists"][name]
        spec = r["training_spec"]
        standalone = r["standalone"]
        ens = r["ensemble"]
        row = dict(base)
        row.update({
            "experiment_id": r["id"],
            "feature_set": ",".join(spec["modules"]),
            "n_features": str(len(r["training_spec"]["modules"]) if False else len_feature_count(spec)),
            "model": "catboost",
            "objective": "binary_logloss",
            "random_seed": str(spec["seed"]),
            "train_rows": str(spec["max_train_rows"]),
            "mean_oof_ts_auc": repr(standalone["mean_ts_auc"]),
            "pooled_oof_ts_auc": repr(standalone["pooled_ts_auc"]),
            "per_fold_ts_auc": ";".join(f"{x:.5f}" for x in standalone["per_fold_ts_auc"]),
            "fold_std": repr(standalone["fold_std"]),
            "sample_mode": spec["sample_mode"],
            "training_runtime_s": repr(r.get("runtime_s", 0.0)),
            "status": r["verdict"],
            "notes": (
                f"{name}; replaced={r['replaced']}; incumbent={r['incumbent_standalone']['mean_ts_auc']:.9f}; "
                f"standalone_delta={r['standalone_delta_vs_incumbent']:+.9f}; "
                f"rho={r['rho_vs_incumbent']:+.6f}; E0={ens['E0_original_rt600']['mean_ts_auc']:.9f}; "
                f"E1={ens['E1_clone_replacement']['mean_ts_auc']:.9f}; "
                f"E2={ens['E2_cat_replacement']['mean_ts_auc']:.9f}; "
                f"marginal_vs_clone={ens['marginal_vs_clone']:+.9f}; "
                f"E2-E0={ens['E2_minus_E0']:+.9f}; positive_folds={ens['positive_folds_vs_clone']}/5; "
                f"dominant_net={r['pair_flow_vs_E0']['dominant_cell']['net_pair_lift']}; "
                f"mature_vs_never={r['pair_flow_vs_E0']['mature_vs_never']['net_pair_lift']}; "
                f"mature_vs_prebreak={r['pair_flow_vs_E0']['mature_vs_prebreak']['net_pair_lift']}."
            ),
        })
        rows.append(row)

    best = result.get("best_hybrid")
    if best:
        ens = best["ensemble"]
        row = dict(base)
        row.update({
            "experiment_id": "RT-1264",
            "feature_set": ",".join(best["cat_streams"]),
            "n_features": "7",
            "model": "frozen_oof_equal_scdf_hybrid",
            "objective": "artifact_rescore",
            "random_seed": str(PAIR_SEED),
            "train_rows": "",
            "mean_oof_ts_auc": repr(ens["E2_cat_hybrid"]["mean_ts_auc"]),
            "pooled_oof_ts_auc": "",
            "per_fold_ts_auc": ";".join(f"{x:.5f}" for x in ens["E2_cat_hybrid"]["per_fold_ts_auc"]),
            "fold_std": repr(float(np.std(ens["E2_cat_hybrid"]["per_fold_ts_auc"]))),
            "sample_mode": "frozen_oof",
            "training_runtime_s": "0.0",
            "status": best["verdict"],
            "notes": (
                f"best_k={best['k']}; survivors={','.join(best['survivors_ordered'])}; "
                f"E0={ens['E0_original_rt600']['mean_ts_auc']:.9f}; "
                f"E1={ens['E1_clone_replacement']['mean_ts_auc']:.9f}; "
                f"E2={ens['E2_cat_hybrid']['mean_ts_auc']:.9f}; "
                f"marginal_vs_clone={ens['marginal_vs_clone']:+.9f}; "
                f"E2-E0={ens['E2_minus_E0']:+.9f}; positive_folds={ens['positive_folds_vs_clone']}/5; "
                f"dominant_net={best['pair_flow_vs_E0']['dominant_cell']['net_pair_lift']}; "
                f"mature_vs_never={best['pair_flow_vs_E0']['mature_vs_never']['net_pair_lift']}."
            ),
        })
        rows.append(row)
    return rows


def len_feature_count(spec: dict) -> int:
    modules = spec["modules"]
    if modules == FULL:
        return 500
    counts = {
        "m00_core": 151,
        "m01_seq": 60,
        "m02_dist": 59,
        "m03_dyn": 60,
        "m04_resid": 60,
        "m06_loc": 60,
        "m07_bayes": 50,
    }
    return int(sum(counts[m] for m in modules))


def write_csa_outputs(result: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "CSA04_RESULTS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (REPORT_DIR / "CSA04_RESULTS.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "arm", "rt_id", "replaced", "standalone", "rho_vs_incumbent",
            "marginal_vs_clone", "E2_minus_E0", "folds_positive",
            "dominant_net", "mature_vs_never_net", "runtime_s", "verdict",
        ])
        for name in ALL_SINGLE_ORDER:
            r = result["specialists"][name]
            ens = r["ensemble"]
            w.writerow([
                name, r["id"], r["replaced"],
                f"{r['standalone']['mean_ts_auc']:.12f}",
                f"{r['rho_vs_incumbent']:.12f}",
                f"{ens['marginal_vs_clone']:.12f}",
                f"{ens['E2_minus_E0']:.12f}",
                ens["positive_folds_vs_clone"],
                r["pair_flow_vs_E0"]["dominant_cell"]["net_pair_lift"],
                r["pair_flow_vs_E0"]["mature_vs_never"]["net_pair_lift"],
                f"{r.get('runtime_s', 0.0):.1f}",
                r["verdict"],
            ])
        for h in result["hybrid_curve"]:
            ens = h["ensemble"]
            w.writerow([
                f"HYBRID-k{h['k']}", h.get("id") or "", ",".join(h["survivors_ordered"]),
                "", "", f"{ens['marginal_vs_clone']:.12f}", f"{ens['E2_minus_E0']:.12f}",
                ens["positive_folds_vs_clone"],
                h["pair_flow_vs_E0"]["dominant_cell"]["net_pair_lift"],
                h["pair_flow_vs_E0"]["mature_vs_never"]["net_pair_lift"],
                "0.0", h["verdict"],
            ])

    lines = [
        "# CSA-04 LOCAL CATBOOST SLOT SWEEP -- FINAL",
        "",
        f"Date: 2026-08-28",
        f"Branch: `research/deep-ensemble-frontier-local-2026`",
        f"Scoring SHA: `{result['git_sha']}`",
        "",
        "## Single-Slot Replacements",
        "",
        "| arm | RT ID | slot | standalone | rho | marginal_vs_clone | E2-E0 | folds positive | dominant net | mature-never net | verdict |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in ALL_SINGLE_ORDER:
        r = result["specialists"][name]
        ens = r["ensemble"]
        lines.append(
            f"| {name} | `{r['id']}` | `{r['replaced']}` | "
            f"{r['standalone']['mean_ts_auc']:.9f} | {r['rho_vs_incumbent']:.6f} | "
            f"{ens['marginal_vs_clone']:+.9f} | {ens['E2_minus_E0']:+.9f} | "
            f"{ens['positive_folds_vs_clone']}/5 | "
            f"{r['pair_flow_vs_E0']['dominant_cell']['net_pair_lift']} | "
            f"{r['pair_flow_vs_E0']['mature_vs_never']['net_pair_lift']} | {r['verdict']} |"
        )
    lines += [
        "",
        "## Hybrid Curve",
        "",
        f"RT-1257 regression check: observed `{result['rt1257_regression_check']['observed']:+.15f}`, "
        f"expected `{EXPECTED_RT1257_MARGIN:+.15f}`, "
        f"diff `{result['rt1257_regression_check']['diff']:+.3e}`, "
        f"passed `{result['rt1257_regression_check']['passed']}`.",
        "",
        "| k | RT ID | ordered survivor slots | marginal_vs_clone | E2-E0 | folds positive | dominant net | mature-never net | verdict |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for h in result["hybrid_curve"]:
        ens = h["ensemble"]
        lines.append(
            f"| {h['k']} | `{h.get('id') or ''}` | {', '.join(h['survivors_ordered'])} | "
            f"{ens['marginal_vs_clone']:+.9f} | {ens['E2_minus_E0']:+.9f} | "
            f"{ens['positive_folds_vs_clone']}/5 | "
            f"{h['pair_flow_vs_E0']['dominant_cell']['net_pair_lift']} | "
            f"{h['pair_flow_vs_E0']['mature_vs_never']['net_pair_lift']} | {h['verdict']} |"
        )
    best = result["best_hybrid"]
    lines += [
        "",
        f"Best `k`: `{best['k']}` with survivors `{', '.join(best['survivors_ordered'])}`.",
        f"H2 composition for CRUNCH: `{', '.join(best['cat_streams'])}`.",
        "",
        "## Prediction Adjudication",
        "",
        result["prediction_adjudication"]["P1"],
        "",
        result["prediction_adjudication"]["P2"],
        "",
        result["prediction_adjudication"]["P3"],
        "",
        "No lockbox, test, production, feature, router, stacker, hyperparameter, seed, or blend-weight change was made.",
    ]
    (REPORT_DIR / "CSA04_FINAL.md").write_text("\n".join(lines) + "\n")

    h2 = [
        "# H2 -- CSA-04 BEST-K HYBRID HANDOFF",
        "",
        f"Date: 2026-08-28",
        f"Source branch: `research/deep-ensemble-frontier-local-2026`",
        f"Result file: `research/reports/deep_ensemble_frontier_2026/local/CSA04_RESULTS.json`",
        "",
        f"Best k: `{best['k']}`",
        f"Ordered survivors: `{', '.join(best['survivors_ordered'])}`",
        f"Seven-stream composition: `{', '.join(best['cat_streams'])}`",
        f"Matched clone composition: `{', '.join(best['clone_streams'])}`",
        f"marginal_vs_clone: `{best['ensemble']['marginal_vs_clone']:+.9f}`",
        f"E2_minus_E0: `{best['ensemble']['E2_minus_E0']:+.9f}`",
        f"verdict: `{best['verdict']}`",
    ]
    (REPORT_DIR / "H2_BEST_K_HYBRID.md").write_text("\n".join(h2) + "\n")


def adjudicate_predictions(result: dict) -> dict:
    singles = result["specialists"]
    wide = [singles["CAT-412"]["ensemble"]["marginal_vs_clone"], singles["CAT-415"]["ensemble"]["marginal_vs_clone"]]
    narrow = [singles["CAT-411"]["ensemble"]["marginal_vs_clone"], singles["CAT-414"]["ensemble"]["marginal_vs_clone"]]
    p1_ok = all(x >= 0.0010 for x in wide) and all(x < 0.0010 for x in narrow)
    p2_ok = all(x < 0.0010 for x in wide)
    best_k = result["best_hybrid"]["k"]
    max_k = len(result["hybrid_curve"][-1]["survivors_ordered"]) if result["hybrid_curve"] else 0
    p3_ok = best_k < max_k if max_k else False
    return {
        "P1": (
            "P1 breadth: "
            + ("supported" if p1_ok else "not supported")
            + f". CAT-412/CAT-415 margins were {wide[0]:+.9f}/{wide[1]:+.9f}; "
            + f"CAT-411/CAT-414 margins were {narrow[0]:+.9f}/{narrow[1]:+.9f}."
        ),
        "P2": (
            "P2 idiosyncrasy: "
            + ("supported" if p2_ok else "not supported")
            + f". The wide idiosyncratic slots CAT-412 and CAT-415 "
            + f"{'both failed' if p2_ok else 'did not both fail'} the +0.0010 gate."
        ),
        "P3": (
            "P3 interior maximum: "
            + ("supported" if p3_ok else "not supported")
            + f". Best k was {best_k}; largest evaluated k was {max_k}."
        ),
    }


def evaluate_csa04(csa, artifact_root: Path, no_ledger: bool) -> dict:
    ensure_local_oof_inputs(artifact_root)
    names_to_validate = list(dict.fromkeys(SPECIALISTS + SEED_CLONES + ["RT-1254", "RT-1255", "RT-1256"] + [IDS[n] for n in NEW_SINGLE_ORDER]))
    validation = validate_oof(csa, names_to_validate)
    c = csa.make_ctx()
    train_log_path = REPORT_DIR / "CSA04_TRAIN_LOG.json"
    train_log = json.loads(train_log_path.read_text()) if train_log_path.exists() else {}
    result = {
        "program": "DEEP_ENSEMBLE_FRONTIER_2026_LOCAL_CSA04",
        "git_sha": git_sha(),
        "artifact_root": str(artifact_root),
        "local_oof_dir": str(LOCAL_OOF_DIR),
        "validation": validation,
        "calibration": "SCDF_NSEEN",
        "pair_seed": PAIR_SEED,
        "pairs_per_t": PAIRS_PER_T,
        "training_log": train_log,
        "specialists": {},
        "hybrid_curve": [],
    }
    for name in ALL_SINGLE_ORDER:
        r = csa.evaluate_specialist(c, name)
        if name in train_log.get("arms", {}):
            r["runtime_s"] = float(train_log["arms"][name].get("runtime_s", 0.0))
            r["fold_runtime_s"] = [float(x["runtime_s"]) for x in train_log["arms"][name].get("folds", [])]
        result["specialists"][name] = apply_local_verdict(r)

    rt1257 = evaluate_hybrid_names(csa, c, ["CAT-413", "CAT-300"], hybrid_id=None)
    observed = float(rt1257["ensemble"]["marginal_vs_clone"])
    diff = observed - EXPECTED_RT1257_MARGIN
    result["rt1257_regression_check"] = {
        "observed": observed,
        "expected": EXPECTED_RT1257_MARGIN,
        "diff": diff,
        "passed": bool(abs(diff) <= 5e-12),
    }
    if not result["rt1257_regression_check"]["passed"]:
        (REPORT_DIR / "CSA04_RESULTS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        raise SystemExit(f"RT-1257 k=2 regression failed: observed={observed} expected={EXPECTED_RT1257_MARGIN} diff={diff}")

    survivors = sorted(
        [name for name, r in result["specialists"].items() if r["ensemble"]["marginal_vs_clone"] >= 0.0010],
        key=lambda n: result["specialists"][n]["ensemble"]["marginal_vs_clone"],
        reverse=True,
    )
    result["survivors_ordered_by_single_slot_marginal"] = survivors
    if len(survivors) < 2:
        raise SystemExit(f"CSA-04 expected at least the prior two survivors; got {survivors}")

    for k in range(2, len(survivors) + 1):
        result["hybrid_curve"].append(evaluate_hybrid_names(csa, c, survivors[:k], hybrid_id=None))

    best_idx = max(
        range(len(result["hybrid_curve"])),
        key=lambda i: result["hybrid_curve"][i]["ensemble"]["marginal_vs_clone"],
    )
    result["hybrid_curve"][best_idx]["id"] = "RT-1264"
    result["best_hybrid"] = result["hybrid_curve"][best_idx]
    result["prediction_adjudication"] = adjudicate_predictions(result)
    write_csa_outputs(result)
    if not no_ledger:
        append_results(result_rows(result))
    return result


def fold0_rank(values: np.ndarray, t: np.ndarray, rows: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    order = np.argsort(t[rows], kind="stable")
    sorted_rows = rows[order]
    sorted_t = t[sorted_rows]
    starts = np.flatnonzero(np.r_[True, sorted_t[1:] != sorted_t[:-1]])
    ends = np.r_[starts[1:], len(sorted_t)]
    for lo, hi in zip(starts, ends):
        rr = sorted_rows[lo:hi]
        out[rr] = rankdata(values[rr], method="average") / max(len(rr), 1)
    return out


def l3_pair_stats(c, base: np.ndarray, cand: np.ndarray, rows: np.ndarray) -> dict:
    rng = np.random.default_rng(PAIR_SEED)
    yy = c.d.y
    tt = c.d.t
    order = np.argsort(tt[rows], kind="stable")
    sorted_rows = rows[order]
    sorted_t = tt[sorted_rows]
    starts = np.flatnonzero(np.r_[True, sorted_t[1:] != sorted_t[:-1]])
    ends = np.r_[starts[1:], len(sorted_t)]
    repairs = damage = total = rt600_wrong = rt600_right = 0
    for lo, hi in zip(starts, ends):
        idx = sorted_rows[lo:hi]
        pos = idx[yy[idx] == 1]
        neg = idx[yy[idx] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        k = min(PAIRS_PER_T, len(pos), len(neg))
        pp = rng.choice(pos, k, replace=False)
        nn = rng.choice(neg, k, replace=False)
        base_right = base[pp] > base[nn]
        cand_right = cand[pp] > cand[nn]
        repairs += int((~base_right & cand_right).sum())
        damage += int((base_right & ~cand_right).sum())
        rt600_wrong += int((~base_right).sum())
        rt600_right += int(base_right.sum())
        total += int(k)
    return {
        "total_pairs_sampled": int(total),
        "rt600_wrong_pairs": int(rt600_wrong),
        "rt600_right_pairs": int(rt600_right),
        "repairs": int(repairs),
        "damage": int(damage),
        "net_pair_lift": int(repairs - damage),
        "repair_rate_of_rt600_wrong": float(repairs / rt600_wrong) if rt600_wrong else None,
        "damage_rate_of_rt600_right": float(damage / rt600_right) if rt600_right else None,
    }


def l3_split_rows(c, fold_rows: np.ndarray, split: str) -> np.ndarray:
    y = c.d.y[fold_rows]
    t = c.d.t[fold_rows]
    age = c.age[fold_rows]
    dominant = (t >= 200) & ((y == 0) | (age >= 100))
    if split == "whole":
        return fold_rows
    if split == "dominant_cell":
        return fold_rows[dominant]
    hb = c.has_break[c.d.sidx[fold_rows]]
    if split == "mature_vs_never":
        return fold_rows[dominant & ((y == 1) | ((y == 0) & ~hb))]
    if split == "mature_vs_prebreak":
        return fold_rows[dominant & ((y == 1) | ((y == 0) & hb))]
    raise KeyError(split)


def l3_rule_output(
    rule: str,
    q: float | None,
    base_rank: np.ndarray,
    candidate_rank: np.ndarray,
    dominant_rows_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    out = base_rank.copy()
    rows = np.flatnonzero(np.isfinite(base_rank) & np.isfinite(candidate_rank))
    mask = np.zeros(len(base_rank), dtype=bool)
    if rule == "unconditional_equal":
        mask[rows] = True
        out[mask] = 0.5 * base_rank[mask] + 0.5 * candidate_rank[mask]
        return out, mask
    if rule == "confidence":
        assert q is not None
        mask[rows] = (candidate_rank[rows] <= q) | (candidate_rank[rows] >= 1.0 - q)
        out[mask] = 0.5 * base_rank[mask] + 0.5 * candidate_rank[mask]
        return out, mask
    if rule == "rt600_boundary":
        assert q is not None
        mask[rows] = np.abs(base_rank[rows] - 0.5) <= (q / 2.0)
        out[mask] = 0.5 * base_rank[mask] + 0.5 * candidate_rank[mask]
        return out, mask
    if rule == "agreement_direction":
        mask[rows] = ((base_rank[rows] - 0.5) * (candidate_rank[rows] - 0.5)) > 0.0
        out[mask] = 0.5 * base_rank[mask] + 0.5 * candidate_rank[mask]
        return out, mask
    if rule == "dominant_cell_only":
        mask[rows] = dominant_rows_mask[rows]
        out[mask] = 0.5 * base_rank[mask] + 0.5 * candidate_rank[mask]
        return out, mask
    if rule == "dominant_confidence":
        assert q is not None
        mask[rows] = dominant_rows_mask[rows] & ((candidate_rank[rows] <= q) | (candidate_rank[rows] >= 1.0 - q))
        out[mask] = 0.5 * base_rank[mask] + 0.5 * candidate_rank[mask]
        return out, mask
    if rule == "three_way_abstain":
        assert q is not None
        tail = np.zeros(len(base_rank), dtype=bool)
        tail[rows] = (candidate_rank[rows] <= q) | (candidate_rank[rows] >= 1.0 - q)
        disagree = np.zeros(len(base_rank), dtype=bool)
        disagree[rows] = ((base_rank[rows] - 0.5) * (candidate_rank[rows] - 0.5)) < 0.0
        out[tail] = 0.5 * base_rank[tail] + 0.5 * candidate_rank[tail]
        abstain = (~tail) & disagree & np.isfinite(out)
        out[abstain] = 0.5
        mask = tail | abstain
        return out, mask
    raise KeyError(rule)


def l3_rules() -> list[tuple[str, str, float | None]]:
    return [
        ("unconditional_equal", "unconditional_equal", None),
        ("confidence_q05", "confidence", 0.05),
        ("confidence_q10", "confidence", 0.10),
        ("confidence_q20", "confidence", 0.20),
        ("confidence_q30", "confidence", 0.30),
        ("rt600_boundary_q10", "rt600_boundary", 0.10),
        ("rt600_boundary_q20", "rt600_boundary", 0.20),
        ("rt600_boundary_q30", "rt600_boundary", 0.30),
        ("agreement_direction", "agreement_direction", None),
        ("dominant_cell_only", "dominant_cell_only", None),
        ("dominant_confidence_q10", "dominant_confidence", 0.10),
        ("dominant_confidence_q20", "dominant_confidence", 0.20),
        ("three_way_abstain_q10", "three_way_abstain", 0.10),
        ("three_way_abstain_q20", "three_way_abstain", 0.20),
    ]


def run_l3(csa, artifact_root: Path, crf_root: Path) -> dict:
    from sbr.metric import ts_auc_flat

    ensure_local_oof_inputs(artifact_root)
    ensure_l3_inputs(crf_root)
    c = csa.make_ctx()
    fold_rows = c.rows[0]
    lockbox = c.d.rows_for([-1])
    raw_candidate = np.load(LOCAL_OOF_DIR / "RT-1234.npy")
    raw_control = np.load(LOCAL_OOF_DIR / "RT-1235.npy")
    checks = {}
    for name, arr in (("RT-1234", raw_candidate), ("RT-1235", raw_control)):
        checks[name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "finite_fold0_rows": int(np.isfinite(arr[fold_rows]).sum()),
            "finite_dev_rows": int(np.isfinite(arr[c.dev]).sum()),
            "finite_lockbox_rows": int(np.isfinite(arr[lockbox]).sum()),
        }
        if checks[name]["shape"] != [len(c.d.y)]:
            raise SystemExit(f"{name} wrong shape {checks[name]}")
        if checks[name]["dtype"] != "float32":
            raise SystemExit(f"{name} wrong dtype {checks[name]}")
        if checks[name]["finite_fold0_rows"] != len(fold_rows):
            raise SystemExit(f"{name} not finite on fold 0 {checks[name]}")
        if checks[name]["finite_dev_rows"] != len(fold_rows):
            raise SystemExit(f"{name} unexpectedly finite beyond fold 0 {checks[name]}")
        if checks[name]["finite_lockbox_rows"] != 0:
            raise SystemExit(f"{name} fills lockbox {checks[name]}")

    controls = csa.load_oof(list(SPECIALISTS))
    cache = csa.CalibratedFoldCache(c, controls)
    rt600, rt600_per = csa.blend_fixed(c, cache, list(SPECIALISTS))
    base_rank = fold0_rank(rt600, c.d.t, fold_rows)
    cand_rank = fold0_rank(raw_candidate, c.d.t, fold_rows)
    ctrl_rank = fold0_rank(raw_control, c.d.t, fold_rows)
    dominant_mask = np.zeros(len(c.d.y), dtype=bool)
    dominant_mask[l3_split_rows(c, fold_rows, "dominant_cell")] = True

    splits = ("whole", "dominant_cell", "mature_vs_never", "mature_vs_prebreak")
    rule_results = {}
    for label, rule, q in l3_rules():
        cand_out, cand_action = l3_rule_output(rule, q, base_rank, cand_rank, dominant_mask)
        ctrl_out, ctrl_action = l3_rule_output(rule, q, base_rank, ctrl_rank, dominant_mask)
        per_split = {
            split: l3_pair_stats(c, base_rank, cand_out, l3_split_rows(c, fold_rows, split))
            for split in splits
        }
        candidate_auc = float(ts_auc_flat(cand_out[fold_rows], c.d.y[fold_rows], c.d.t[fold_rows]))
        control_auc = float(ts_auc_flat(ctrl_out[fold_rows], c.d.y[fold_rows], c.d.t[fold_rows]))
        rule_results[label] = {
            "rule": rule,
            "q": q,
            "candidate_action_rows": int(cand_action[fold_rows].sum()),
            "control_action_rows": int(ctrl_action[fold_rows].sum()),
            "candidate_fold0_ts_auc": candidate_auc,
            "control_fold0_ts_auc": control_auc,
            "fold0_E2_minus_E1_descriptive": float(candidate_auc - control_auc),
            "pair_flow_vs_rt600": per_split,
        }

    unconditional_repairs = rule_results["unconditional_equal"]["pair_flow_vs_rt600"]["dominant_cell"]["repairs"]
    for r in rule_results.values():
        dom_repairs = r["pair_flow_vs_rt600"]["dominant_cell"]["repairs"]
        prebreak_damage = r["pair_flow_vs_rt600"]["mature_vs_prebreak"]["damage_rate_of_rt600_right"]
        r["retained_fraction_of_unconditional_dominant_repairs"] = float(dom_repairs / unconditional_repairs) if unconditional_repairs else None
        r["success"] = bool(
            unconditional_repairs
            and dom_repairs >= 0.5 * unconditional_repairs
            and prebreak_damage is not None
            and prebreak_damage < 0.05
        )

    successful = [name for name, r in rule_results.items() if r["success"]]
    best_by_net = max(rule_results, key=lambda name: rule_results[name]["pair_flow_vs_rt600"]["dominant_cell"]["net_pair_lift"])
    result = {
        "program": "DEEP_ENSEMBLE_FRONTIER_2026_LOCAL_L3",
        "git_sha": git_sha(),
        "date": "2026-08-28",
        "fold": 0,
        "fold0_rows": int(len(fold_rows)),
        "candidate": "RT-1234",
        "control": "RT-1235",
        "input_checks": checks,
        "rt600_fold0_ts_auc": float(rt600_per[0]),
        "coordinate": "within-t rank percentile for RT600, RT-1234, and RT-1235 on fold 0",
        "success_criterion": (
            "retain >=50% of unconditional dominant-cell repairs and keep "
            "mature-vs-prebreak damage_rate_of_rt600_right < 0.05"
        ),
        "rules": rule_results,
        "successful_rules": successful,
        "best_by_dominant_net": best_by_net,
        "verdict": "PASS_RETENTION_MECHANISM_FOUND" if successful else "FAIL_NO_RETENTION_MECHANISM",
        "limitation": "fold-0-only descriptive diagnostic; cannot promote any model.",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "L3_ARBITRATION_PROBE.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_l3_md(result)
    return result


def write_l3_md(result: dict) -> None:
    lines = [
        "# L3 ARBITRATION PROBE -- H3 VERDICT",
        "",
        f"Date: {result['date']}",
        f"Branch: `research/deep-ensemble-frontier-local-2026`",
        f"Scoring SHA: `{result['git_sha']}`",
        "",
        f"Verdict: `{result['verdict']}`.",
        "",
        "This is fold-0 only. `RT-1234` and `RT-1235` each have 806,334 finite rows, so this is a descriptive gate and cannot promote a model.",
        "",
        "| rule | action rows | dominant repairs | dominant damage | dominant net | retained repairs | mature-prebreak damage rate | fold0 E2-E1 | success |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, r in result["rules"].items():
        dom = r["pair_flow_vs_rt600"]["dominant_cell"]
        mp = r["pair_flow_vs_rt600"]["mature_vs_prebreak"]
        retained = r["retained_fraction_of_unconditional_dominant_repairs"]
        lines.append(
            f"| `{name}` | {r['candidate_action_rows']} | {dom['repairs']} | {dom['damage']} | "
            f"{dom['net_pair_lift']} | {retained:.3f} | "
            f"{mp['damage_rate_of_rt600_right']:.4f} | "
            f"{r['fold0_E2_minus_E1_descriptive']:+.9f} | {r['success']} |"
        )
    lines += [
        "",
        f"Successful rules: `{', '.join(result['successful_rules']) if result['successful_rules'] else 'none'}`.",
        f"Best dominant net rule: `{result['best_by_dominant_net']}`.",
        "",
        "The prior was poor: SS-01 through SS-04 were all KILL. The only reason to run this probe was the much lower RT-1234 rho and its distinct repair set. A negative result here is direct evidence against opening another neural detector without a new retention mechanism.",
    ]
    (REPORT_DIR / "L3_ARBITRATION_PROBE.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    crf_root = Path(args.crf_root).resolve()
    csa = import_csa(artifact_root)
    if args.run in ("all", "train"):
        ensure_local_oof_inputs(artifact_root)
        train_new_arms(csa, args.force_train)
        validate_oof(csa, [IDS[n] for n in NEW_SINGLE_ORDER])
    if args.run in ("all", "evaluate"):
        result = evaluate_csa04(csa, artifact_root, args.no_ledger)
        print(json.dumps({
            "stage": "CSA04_EVALUATE",
            "results": str(REPORT_DIR / "CSA04_RESULTS.json"),
            "final": str(REPORT_DIR / "CSA04_FINAL.md"),
            "best_k": result["best_hybrid"]["k"],
            "best_margin": result["best_hybrid"]["ensemble"]["marginal_vs_clone"],
            "best_verdict": result["best_hybrid"]["verdict"],
        }, indent=2, sort_keys=True))
    if args.run == "l3":
        result = run_l3(csa, artifact_root, crf_root)
        print(json.dumps({
            "stage": "L3",
            "result": str(REPORT_DIR / "L3_ARBITRATION_PROBE.json"),
            "report": str(REPORT_DIR / "L3_ARBITRATION_PROBE.md"),
            "verdict": result["verdict"],
            "successful_rules": result["successful_rules"],
            "best_by_dominant_net": result["best_by_dominant_net"],
        }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
