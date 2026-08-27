#!/usr/bin/env python
"""Build and validate the frozen RT-1257 deployment artifact.

This is engineering-only deployment work for RT-1257. It does not append
research/RESULTS.csv, score alternatives, tune parameters, or mint a new RT ID.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = REPO.parent / "structural-break-claude-wave3"
DEFAULT_RT600_MODEL = DEFAULT_ARTIFACT_ROOT / "models" / "final10k_ensemble"
DEFAULT_OUT = REPO / "models" / "rt1257_final"
REPORT_DIR = REPO / "engineering" / "reports" / "rt1257_deployment"

FULL = ("m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes")
FOLDS = (0, 1, 2, 3, 4)
RESULTS_SHA256_FROZEN = "f257ab77aeb170e34811f73268827a0f0e4ba4011bc8111f485e1c81d896d114"
RT600_MODEL_ZIP_SHA256 = "6c8960ddc7331afecfc6980974f2879401a1eb583f15f5cad34c2ea03299ea8c"

CATBOOST_BASE_PARAMS = {
    "iterations": 600,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "bootstrap_type": "Bernoulli",
    "subsample": 0.7,
    "rsm": 0.7,
    "border_count": 127,
    "thread_count": 2,
    "allow_writing_files": False,
    "verbose": False,
}

CAT_SPECS = {
    "CAT-300": {
        "id": "RT-1255",
        "slot": 0,
        "replaces": "RT-300",
        "modules": list(FULL),
        "seed": 0,
        "dev_max_train_rows": 1_000_000,
        "final_max_train_rows": 1_250_000,
        "sample_mode": "uniform",
        "model_file": "model.cbm.0",
    },
    "CAT-413": {
        "id": "RT-1254",
        "slot": 4,
        "replaces": "RT-413",
        "modules": list(FULL),
        "seed": 0,
        "dev_max_train_rows": 700_000,
        "final_max_train_rows": 875_000,
        "sample_mode": "uniform",
        "model_file": "model.cbm.4",
    },
}

MEMBERS = [
    {"slot": 0, "name": "CAT-300", "id": "RT-1255", "kind": "catboost", "path": "model.cbm.0"},
    {"slot": 1, "name": "RT-410", "id": "RT-410", "kind": "lightgbm", "path": "model.txt.1", "source_index": 1},
    {"slot": 2, "name": "RT-411", "id": "RT-411", "kind": "lightgbm", "path": "model.txt.2", "source_index": 2},
    {"slot": 3, "name": "RT-412", "id": "RT-412", "kind": "lightgbm", "path": "model.txt.3", "source_index": 3},
    {"slot": 4, "name": "CAT-413", "id": "RT-1254", "kind": "catboost", "path": "model.cbm.4"},
    {"slot": 5, "name": "RT-414", "id": "RT-414", "kind": "lightgbm", "path": "model.txt.5", "source_index": 5},
    {"slot": 6, "name": "RT-415", "id": "RT-415", "kind": "lightgbm", "path": "model.txt.6", "source_index": 6},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("oof", "fit", "validate", "benchmark", "benchmark-one", "all"))
    p.add_argument("--artifact-root", default=os.environ.get("SBR_ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT)))
    p.add_argument("--rt600-model", default=os.environ.get("SBR_RT600_MODEL", str(DEFAULT_RT600_MODEL)))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--only", default="", help="comma-separated CAT-300,CAT-413")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def configure_runtime(artifact_root: Path):
    os.environ["SBR_ROOT"] = str(artifact_root)
    for p in (REPO / "src", REPO / "research" / "scripts"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)

    import numpy as np
    import sbr.pipeline as PL

    PL.ROOT = str(artifact_root)
    PL.FEAT = str(artifact_root / "cache" / "features")
    PL.FEAT_SCREEN = str(artifact_root / "cache" / "features_screen")
    PL.FOLDS = str(artifact_root / "research" / "folds" / "folds_final10k.parquet")
    PL.STORE = str(artifact_root / "cache" / "store")
    PL.OOF = str(artifact_root / "research" / "oof")
    return np, PL


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def git_dirty(paths: list[str] | None = None) -> list[str]:
    cmd = ["git", "-C", str(REPO), "status", "--porcelain"]
    if paths:
        cmd += ["--", *paths]
    out = subprocess.check_output(cmd, text=True)
    return [x for x in out.splitlines() if x]


def env_versions() -> dict[str, str]:
    import importlib.metadata as md

    packages = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "catboost", "numba", "pyarrow"]
    out = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine()}
    for package in packages:
        try:
            out[package] = md.version(package)
        except Exception:
            out[package] = "missing"
    return out


def source_zip_sha() -> str:
    from build_submission import pack_dir

    _, sha = pack_dir(str(REPO / "src" / "sbr"), "sbr")
    return sha


def sample_rows(np, d, rows, budget: int, seed: int, sample_mode: str):
    rows = np.asarray(rows)
    if len(rows) <= budget:
        return rows
    rng = np.random.default_rng(seed)
    if sample_mode == "uniform":
        return np.sort(rng.choice(rows, budget, replace=False))
    if sample_mode == "per_series":
        s = d.sidx[rows]
        order = np.argsort(s, kind="stable")
        rows = rows[order]
        s = s[order]
        bnd = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
        ends = np.r_[bnd[1:], len(s)]
        per = budget // len(bnd)
        pick = [np.arange(b, e) if e - b <= per else rng.choice(np.arange(b, e), per, replace=False)
                for b, e in zip(bnd, ends)]
        return np.sort(rows[np.concatenate(pick)])
    raise KeyError(sample_mode)


def predict_catboost_rows(np, PL, model, mats, names, rows, keep, chunk_rows=150_000):
    out = np.empty(len(rows), dtype=np.float32)
    for i in range(0, len(rows), chunk_rows):
        j = min(i + chunk_rows, len(rows))
        X = PL._stack(mats, names, rows[i:j], keep)
        out[i:j] = model.predict_proba(X, thread_count=1, task_type="CPU")[:, 1].astype(np.float32)
        del X
        gc.collect()
    return out


def selected_specs(only: str) -> list[str]:
    wanted = [x for x in only.split(",") if x]
    if not wanted:
        return ["CAT-300", "CAT-413"]
    bad = sorted(set(wanted) - set(CAT_SPECS))
    if bad:
        raise SystemExit(f"unknown CAT member(s): {bad}")
    return wanted


def phase_oof(args: argparse.Namespace) -> dict:
    from catboost import CatBoostClassifier

    artifact_root = Path(args.artifact_root).resolve()
    np, PL = configure_runtime(artifact_root)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    oof_dir = REPORT_DIR / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)

    d = PL.Data()
    mats, names = PL.load_features(FULL)
    keep = np.arange(len(names), dtype=np.int64)
    result = {"phase": "oof", "purpose": "SCDF calibration only; no CV score computed", "members": {}}

    for name in selected_specs(args.only):
        spec = CAT_SPECS[name]
        oof_path = oof_dir / f"{spec['id']}.final10k.npy"
        sidecar = oof_dir / f"{spec['id']}.final10k.json"
        if oof_path.exists() and sidecar.exists() and not args.force:
            result["members"][name] = json.load(sidecar.open())
            result["members"][name]["status"] = "reused"
            continue

        params = dict(CATBOOST_BASE_PARAMS, random_seed=int(spec["seed"]))
        oof = np.full(len(d.y), np.nan, dtype=np.float32)
        folds = []
        t_member = time.time()
        for fold in FOLDS:
            t0 = time.time()
            tr_rows = d.rows_for([f for f in FOLDS if f != fold])
            tr_rows = sample_rows(np, d, tr_rows, int(spec["final_max_train_rows"]),
                                  int(spec["seed"]), spec["sample_mode"])
            va_rows = d.rows_for([fold])
            X = PL._stack(mats, names, tr_rows, keep)
            y = d.y[tr_rows]
            model = CatBoostClassifier(**params)
            model.fit(X, y)
            del X, y
            gc.collect()
            oof[va_rows] = predict_catboost_rows(np, PL, model, mats, names, va_rows, keep)
            del model
            gc.collect()
            folds.append({"fold": int(fold), "train_rows": int(len(tr_rows)),
                          "valid_rows": int(len(va_rows)), "runtime_s": time.time() - t0})
            print(f"{name} final10k OOF fold {fold}: {len(tr_rows)} train rows", flush=True)

        all_rows = d.rows_for(FOLDS)
        if not np.isfinite(oof[all_rows]).all():
            raise RuntimeError(f"{name} OOF has non-finite or missing predictions")
        np.save(oof_path, oof)
        meta = {
            "member": name,
            "id": spec["id"],
            "replaces": spec["replaces"],
            "params": params,
            "training_spec": spec,
            "folds": folds,
            "runtime_s": time.time() - t_member,
            "oof_path": str(oof_path),
            "oof_sha256": sha_file(oof_path),
            "no_cv_score_computed": True,
        }
        json.dump(meta, sidecar.open("w"), indent=2)
        result["members"][name] = meta
    json.dump(result, (REPORT_DIR / "RT1257_OOF.json").open("w"), indent=2)
    return result


def copy_retained_lightgbm(rt600_model: Path, out: Path, force: bool) -> list[dict]:
    copied = []
    for member in MEMBERS:
        if member["kind"] != "lightgbm":
            continue
        src = rt600_model / f"model.txt.{member['source_index']}"
        dst = out / member["path"]
        if not src.exists():
            raise FileNotFoundError(src)
        if force or not dst.exists():
            shutil.copy2(src, dst)
        copied.append({"member": member["name"], "from": str(src), "to": str(dst), "sha256": sha_file(dst)})
    return copied


def train_final_catboost(args: argparse.Namespace, name: str, d, mats, names, keep) -> dict:
    from catboost import CatBoostClassifier

    np, PL = configure_runtime(Path(args.artifact_root).resolve())
    spec = CAT_SPECS[name]
    out = Path(args.out).resolve()
    path = out / spec["model_file"]
    sidecar = REPORT_DIR / f"{spec['model_file']}.json"
    if path.exists() and sidecar.exists() and not args.force:
        meta = json.load(sidecar.open())
        meta["status"] = "reused"
        return meta

    all_rows = d.rows_for(FOLDS)
    rows = sample_rows(np, d, all_rows, int(spec["final_max_train_rows"]),
                       int(spec["seed"]), spec["sample_mode"])
    params = dict(CATBOOST_BASE_PARAMS, random_seed=int(spec["seed"]))
    t0 = time.time()
    X = PL._stack(mats, names, rows, keep)
    y = d.y[rows]
    model = CatBoostClassifier(**params)
    model.fit(X, y)
    del X, y
    gc.collect()
    model.save_model(path, format="cbm")

    sample = rows[:min(4096, len(rows))]
    Xs = PL._stack(mats, names, sample, keep)
    before = model.predict_proba(Xs, thread_count=1, task_type="CPU")[:, 1]
    loaded = CatBoostClassifier()
    loaded.load_model(path, format="cbm")
    after = loaded.predict_proba(Xs, thread_count=1, task_type="CPU")[:, 1]
    max_abs = float(np.max(np.abs(before - after))) if len(before) else 0.0
    del Xs, model, loaded
    gc.collect()

    meta = {
        "member": name,
        "id": spec["id"],
        "replaces": spec["replaces"],
        "path": str(path),
        "sha256": sha_file(path),
        "params": params,
        "training_spec": spec,
        "train_rows": int(len(rows)),
        "n_features": int(len(keep)),
        "runtime_s": time.time() - t0,
        "save_load_repro_max_abs": max_abs,
        "status": "trained",
    }
    json.dump(meta, sidecar.open("w"), indent=2)
    return meta


def build_manifest(args: argparse.Namespace, d, names, keep, final_meta: dict) -> dict:
    np, _ = configure_runtime(Path(args.artifact_root).resolve())
    from sbr.production.calibration import DEFAULT_COORD, SmoothTimeCDFCal
    from sbr.stream.engine import StreamEngine

    out = Path(args.out).resolve()
    rt600_model = Path(args.rt600_model).resolve()
    rt600_manifest = json.load((rt600_model / "manifest.json").open())
    eng = StreamEngine(FULL).fit_historical(np.arange(1200, dtype=np.float64) % 7 - 3.0)
    man_eng = eng.manifest()
    if man_eng["columns"] != list(names):
        raise RuntimeError("streaming column order != batch feature order")

    all_rows = d.rows_for(FOLDS)
    cals = []
    slices = []
    streams = []
    model_files = []
    for member in MEMBERS:
        slot = int(member["slot"])
        if member["kind"] == "catboost":
            spec = CAT_SPECS[member["name"]]
            oof_path = REPORT_DIR / "oof" / f"{spec['id']}.final10k.npy"
            if not oof_path.exists():
                raise FileNotFoundError(f"run phase 'oof' first: {oof_path}")
            oof = np.load(oof_path, mmap_mode="r")
            cals.append(SmoothTimeCDFCal.fit(oof[all_rows], d.t[all_rows], time_coord=DEFAULT_COORD))
            slices.append(list(range(len(keep))))
            streams.append({
                "slot": slot,
                "member": member["name"],
                "id": member["id"],
                "kind": "catboost",
                "replaces": spec["replaces"],
                "modules": list(FULL),
                "seed": spec["seed"],
                "n_columns": int(len(keep)),
                "dev_max_train_rows": spec["dev_max_train_rows"],
                "max_train_rows": spec["final_max_train_rows"],
                "sample_mode": spec["sample_mode"],
                "params": dict(CATBOOST_BASE_PARAMS, random_seed=int(spec["seed"])),
            })
            model_files.append({"slot": slot, "kind": "catboost", "path": member["path"], "format": "cbm"})
        else:
            source_index = int(member["source_index"])
            cals.append(SmoothTimeCDFCal.from_json(rt600_manifest["calibration"]["models"][source_index]))
            slices.append(rt600_manifest["booster_columns"][source_index])
            src_stream = dict(rt600_manifest["streams"][source_index])
            src_stream.update({"slot": slot, "member": member["name"], "id": member["id"],
                               "kind": "lightgbm", "source_rt600_slot": source_index})
            streams.append(src_stream)
            model_files.append({"slot": slot, "kind": "lightgbm", "path": member["path"], "format": "txt"})

    model_hashes = {m["path"]: sha_file(out / m["path"]) for m in model_files}
    return {
        "experiment_id": "RT-1257",
        "what": "Hybrid deployment: CAT-300 + CAT-413 replacing RT-300 + RT-413 in RT600",
        "scientific_tip": "0ca415daa38214f834044a71475e1cbd97218f40",
        "research_branch": "research/catboost-specialist-2026",
        "modules": list(man_eng["modules"]),
        "columns": list(names),
        "n_features": int(len(names)),
        "feature_manifest_sha256": man_eng["feature_manifest_sha256"],
        "booster_columns": slices,
        "model_files": model_files,
        "model_hashes": model_hashes,
        "calibration": {
            "kind": "scdf",
            "models": [c.to_json() for c in cals],
            "time_coord": DEFAULT_COORD,
            "n_anchor": 12,
            "grid": 256,
            "min_n": 400,
            "integration": "equal_weight_mean_of_member_scdf_scores",
            "fitted_on": "cross-fitted OOF over folds_final10k, all 10,000 series",
        },
        "streams": streams,
        "composition": [m["name"] for m in MEMBERS],
        "retained_rt600_members": ["RT-410", "RT-411", "RT-412", "RT-414", "RT-415"],
        "replaced_rt600_members": {"RT-300": "CAT-300/RT-1255", "RT-413": "CAT-413/RT-1254"},
        "excluded": ["CAT-410/RT-1256"],
        "code_git_sha": git_sha(),
        "code_git_clean": git_dirty([
            "src/sbr",
            "research/scripts/build_submission.py",
            "research/scripts/rt1257_deployment.py",
            "requirements-rt1257.txt",
        ]) == [],
        "source_zip_sha256": source_zip_sha(),
        "trained_on": {
            "folds": list(FOLDS),
            "partition": "folds_final10k",
            "n_series": 10000,
            "includes_former_lockbox": True,
        },
        "expected_provenance": {
            "n_series": 10000,
            "partition": "folds_final10k",
            "folds_sha256": rt600_manifest["folds_sha256"],
            "n_boosters": 7,
            "calibration_kind": "scdf",
            "calibration_time_coord": DEFAULT_COORD,
        },
        "folds_sha256": rt600_manifest["folds_sha256"],
        "rt600_source_model": str(rt600_model),
        "rt600_model_zip_sha256": RT600_MODEL_ZIP_SHA256,
        "catboost_final_training": final_meta,
        "dependencies": env_versions(),
        "lockbox_touched": False,
        "lockbox_note": "Former lockbox is final-fit training data only; no evaluation score is computed here.",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def phase_fit(args: argparse.Namespace) -> dict:
    artifact_root = Path(args.artifact_root).resolve()
    rt600_model = Path(args.rt600_model).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    np, PL = configure_runtime(artifact_root)
    d = PL.Data()
    mats, names = PL.load_features(FULL)
    keep = np.arange(len(names), dtype=np.int64)
    all_rows = d.rows_for(FOLDS)
    if len(np.unique(d.sidx[all_rows])) != 10000:
        raise RuntimeError("final fit must see all 10,000 labelled series")

    copied = copy_retained_lightgbm(rt600_model, out, args.force)
    final_meta = {"copied_lightgbm": copied, "catboost": {}}
    for name in selected_specs(args.only):
        final_meta["catboost"][name] = train_final_catboost(args, name, d, mats, names, keep)
    for name in CAT_SPECS:
        path = out / CAT_SPECS[name]["model_file"]
        if not path.exists():
            raise FileNotFoundError(f"missing final CatBoost model: {path}")

    manifest = build_manifest(args, d, names, keep, final_meta)
    json.dump(manifest, (out / "manifest.json").open("w"), indent=2)
    final_meta["manifest_sha256"] = sha_file(out / "manifest.json")
    final_meta["artifact_files"] = sorted(p.name for p in out.iterdir() if p.is_file())
    json.dump(final_meta, (REPORT_DIR / "RT1257_FIT.json").open("w"), indent=2)
    return final_meta


def phase_validate(args: argparse.Namespace) -> dict:
    artifact_root = Path(args.artifact_root).resolve()
    out = Path(args.out).resolve()
    np, PL = configure_runtime(artifact_root)
    for p in (REPO / "research" / "scripts",):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from local_runner import check_all, run_infer
    from sbr.production.model import ProductionModel
    from sbr.production.submission import infer
    from sbr.stream.engine import StreamEngine

    model = ProductionModel.load(str(out))
    manifest = model.manifest
    d = PL.Data()
    store = d.st

    rng = np.random.default_rng(1257)
    idx = [0, 1, 2, 137, 999, 2025, 4242, 7777]
    idx += [int(x) for x in rng.choice(store.n_series, 12, replace=False)]
    series = []
    labels = []
    for i in idx:
        h, o, _ = store.series(i)
        series.append((h, o))
        labels.append(store.labels(i))

    t0 = time.time()
    local = check_all(infer, series, str(out), labels=labels, verbose=False)
    wall = time.time() - t0

    a = run_infer(infer, series, str(out))
    b = run_infer(infer, series, str(out))
    flat = np.concatenate(a)
    key_mapping = {
        "series_ids": [int(store.meta.id.iloc[i]) for i in idx],
        "output_rows": int(sum(len(x) for x in a)),
        "expected_rows": int(sum(len(o) for _, o in series)),
        "row_mapping_ok": int(sum(len(x) for x in a)) == int(sum(len(o) for _, o in series)),
    }

    h, o, _ = store.series(idx[0])
    engine = StreamEngine(FULL).fit_historical(h)
    prefix_rows = [engine.step(x) for x in o[:100]]
    poisoned = o.copy()
    poisoned[100:] = rng.standard_normal(max(len(o) - 100, 0)) * 1e6
    engine2 = StreamEngine(FULL).fit_historical(h)
    prefix_rows_2 = [engine2.step(x) for x in poisoned[:100]]
    prefix_invariant = all(np.array_equal(x, y, equal_nan=True) for x, y in zip(prefix_rows, prefix_rows_2))

    results_hash = sha_file(REPO / "research" / "RESULTS.csv")
    report = {
        "phase": "validate",
        "artifact": str(out),
        "manifest_sha256": sha_file(out / "manifest.json"),
        "model_hashes": manifest.get("model_hashes"),
        "source_zip_sha256": manifest.get("source_zip_sha256"),
        "local_harness": local,
        "validation_wall_s": wall,
        "determinism_exact": all(np.array_equal(x, y) for x, y in zip(a, b)),
        "prefix_invariance_sample": prefix_invariant,
        "finite": bool(np.isfinite(flat).all()),
        "in_range": bool(((flat >= 0.0) & (flat <= 1.0)).all()),
        "score_min": float(flat.min()),
        "score_max": float(flat.max()),
        "key_mapping": key_mapping,
        "catboost_save_load_max_abs": {
            k: v.get("save_load_repro_max_abs")
            for k, v in (manifest.get("catboost_final_training", {}).get("catboost") or {}).items()
        },
        "results_sha256": results_hash,
        "results_unchanged": results_hash == RESULTS_SHA256_FROZEN,
    }
    json.dump(report, (REPORT_DIR / "RT1257_VALIDATE.json").open("w"), indent=2)
    return report


def maxrss_mb() -> float:
    import resource

    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw / (1024.0 * 1024.0) if raw > 10_000_000 else raw / 1024.0


def benchmark_one(args: argparse.Namespace) -> dict:
    out = Path(args.out).resolve()
    np, PL = configure_runtime(Path(args.artifact_root).resolve())
    from sbr.production.model import ProductionModel

    d = PL.Data()
    store = d.st
    rng = np.random.default_rng(600 if out.name == "final10k_ensemble" else 1257)
    idx = [int(x) for x in rng.choice(store.n_series, 40, replace=False)]
    series = [store.series(i)[:2] for i in idx]
    total_points = int(sum(len(o) for _, o in series))

    t0 = time.perf_counter()
    model = ProductionModel.load(str(out))
    load_s = time.perf_counter() - t0

    fixed_s = 0.0
    point_s = 0.0
    n_points = 0
    score_sum = 0.0
    for h, o in series:
        a = time.perf_counter()
        model.start_series(h)
        fixed_s += time.perf_counter() - a
        b = time.perf_counter()
        for x in o:
            v = model.step(float(x))
            score_sum += float(v)
        point_s += time.perf_counter() - b
        n_points += len(o)

    fixed_ms = 1000.0 * fixed_s / max(len(series), 1)
    point_ms = 1000.0 * point_s / max(n_points, 1)
    mean_online = float(np.mean(store.meta.n_online.to_numpy()))
    projected_s = 10_000.0 * (fixed_ms / 1000.0) + 10_000.0 * mean_online * (point_ms / 1000.0)
    return {
        "artifact": str(out),
        "n_series": len(series),
        "n_points": total_points,
        "load_s": load_s,
        "fixed_ms_per_series": fixed_ms,
        "ms_per_online_point": point_ms,
        "peak_rss_mb": maxrss_mb(),
        "mean_online_horizon": mean_online,
        "projected_10000_series_s": projected_s,
        "projected_10000_series_h": projected_s / 3600.0,
        "score_sum_guard": score_sum,
    }


def phase_benchmark(args: argparse.Namespace) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve()
    py = sys.executable
    common = ["--artifact-root", str(Path(args.artifact_root).resolve())]
    rt1257_cmd = [py, str(here), "benchmark-one", "--out", str(Path(args.out).resolve()), *common]
    rt600_cmd = [py, str(here), "benchmark-one", "--out", str(Path(args.rt600_model).resolve()), *common]

    def run(cmd):
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{REPO / 'src'}:{REPO / 'research' / 'scripts'}:{env.get('PYTHONPATH', '')}"
        p = subprocess.run(cmd, text=True, capture_output=True, check=True, env=env)
        return json.loads(p.stdout)

    res = {
        "phase": "benchmark",
        "rt1257": run(rt1257_cmd),
        "rt600": run(rt600_cmd),
        "quota_h": 15.0,
    }
    res["rt1257"]["pass_15h_quota"] = res["rt1257"]["projected_10000_series_h"] < res["quota_h"]
    res["rt600"]["pass_15h_quota"] = res["rt600"]["projected_10000_series_h"] < res["quota_h"]
    res["ratio_vs_rt600"] = (
        res["rt1257"]["projected_10000_series_h"] /
        max(res["rt600"]["projected_10000_series_h"], 1e-12))
    json.dump(res, (REPORT_DIR / "RT1257_BENCHMARK.json").open("w"), indent=2)
    return res


def main() -> None:
    args = parse_args()
    if args.phase == "benchmark-one":
        print(json.dumps(benchmark_one(args)))
        return
    if args.phase in ("oof", "all"):
        phase_oof(args)
    if args.phase in ("fit", "all"):
        phase_fit(args)
    if args.phase in ("validate", "all"):
        phase_validate(args)
    if args.phase in ("benchmark", "all"):
        phase_benchmark(args)


if __name__ == "__main__":
    main()
