"""Run the RT-1320 final10k fit locally, reusing the existing feature cache.

The Crunch submission (``submissions/I_rt1320_final10k_fit.py``) is written for
the cloud harness: ``train()`` consumes Crunch's stream of 10,000
``(id, x_historical, x_online, tau_index)`` tuples, materialises a store from
it, and rebuilds the 500-column feature bank (~15 min, ~10 GB) into a temp
root. Locally that rebuild is pure waste -- the final10k cache already exists.

This driver skips exactly those two steps and calls the submission's own
training functions against the existing cache, so the artifacts, the gates and
the checkpoint/resume logic are the submission's, not a reimplementation.

Why local is worth doing at all: Crunch does NOT persist the model directory
between runs, so a killed run loses everything (submission #18 died 33 minutes
short of the calibration and left nothing). Locally the checkpoints persist, so
this can be interrupted and resumed freely.

Thread count is a one-way door: ``lightgbm_num_threads`` is part of
``_checkpoint_contract()``, so changing it between runs quarantines every
checkpoint and restarts from zero. Pick a value and keep it.

Usage:
    python research/scripts/rt1320_local_fit.py --check
    python research/scripts/rt1320_local_fit.py --model-dir ~/rt1320_local_fit
    python research/scripts/rt1320_local_fit.py --stage final   # model.txt.7 only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
# The final10k feature cache and store live in the promotion worktree, not here,
# so this tool is cross-worktree by nature. Override with --artifact-root.
DEFAULT_ROOT = REPO.parent / "structural-break-rt1320-promotion-2026"
DEFAULT_SUBMISSION = REPO / "submissions" / "I_rt1320_final10k_fit.py"
ICLOUD_HINTS = ("Documents", "Desktop")

STAGES = ("final", "nested", "student", "calib")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ROOT,
        help="tree holding cache/store, cache/features and research/folds "
        "(default: this repo)",
    )
    p.add_argument(
        "--submission",
        type=Path,
        default=DEFAULT_SUBMISSION,
        help="the Crunch submission whose functions are reused",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=Path.home() / "rt1320_local_fit",
        help="where checkpoints and artifacts go; keep OFF iCloud-synced paths",
    )
    p.add_argument(
        "--num-threads",
        type=int,
        default=6,
        help="LightGBM threads. Part of the checkpoint contract -- never change "
        "it between runs of the same model-dir (default: 6 = M2 Pro perf cores)",
    )
    p.add_argument(
        "--stage",
        choices=("all",) + STAGES,
        default="all",
        help="'final' stops after model.txt.7 (~35 min); 'all' runs everything",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="verify the cache satisfies the contract, then exit without training",
    )
    p.add_argument("--allow-icloud", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def load_submission(path: Path, num_threads: int):
    """Import the submission with NUM_THREADS already set.

    NUM_THREADS is bound at module scope from the environment, and it feeds both
    LightGBM's params and the OMP thread caps, so it has to be set before import.
    """
    os.environ["SBR_RT1320_NUM_THREADS"] = str(int(num_threads))
    spec = importlib.util.spec_from_file_location("rt1320_submission", path)
    if not spec or not spec.loader:
        raise SystemExit(f"cannot load submission from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if mod.NUM_THREADS != int(num_threads):
        raise SystemExit(
            f"NUM_THREADS is {mod.NUM_THREADS}, expected {num_threads}; "
            "something else set SBR_RT1320_NUM_THREADS first"
        )
    return mod


def verify_contract(mod, artifact_root: Path) -> dict:
    """Re-check what the skipped store/feature build would have guaranteed.

    _materialize_final10k_store verified the folds hash and the 10,000-series
    population; _build_feature_cache verified the 500-column count and the
    RT-600 stream manifest hash. Skipping them must not skip their gates.
    """
    import numpy as np

    folds_src = artifact_root / "research" / "folds" / "folds_final10k.parquet"
    if not folds_src.exists():
        raise SystemExit(f"missing {folds_src}")
    folds_sha = mod._sha256_file(folds_src)
    if folds_sha != mod.FOLDS_FINAL10K_SHA256:
        raise SystemExit(
            f"folds_final10k sha mismatch: {folds_sha} != {mod.FOLDS_FINAL10K_SHA256}"
        )

    for sub in ("cache/store/values.npy", "cache/store/meta.parquet"):
        if not (artifact_root / sub).exists():
            raise SystemExit(f"missing {artifact_root / sub}")

    PL = mod._configure_runtime(artifact_root)

    from sbr.stream.engine import StreamEngine

    engine = StreamEngine(mod.FULL_MODULES).fit_historical(
        np.arange(1200, dtype=np.float64) % 7 - 3.0
    )
    stream_sha = engine.manifest().get("feature_manifest_sha256")
    if stream_sha != mod.RT600_FEATURE_MANIFEST_SHA256:
        raise SystemExit(
            f"feature manifest sha mismatch: {stream_sha} != "
            f"{mod.RT600_FEATURE_MANIFEST_SHA256}"
        )

    d = PL.Data()
    mats, names, keep_idx = mod._load_features(PL)  # 500 cols + forbidden-token gate
    all_rows = d.rows_for(mod.FOLDS)
    n_series = int(len(np.unique(d.sidx[all_rows])))
    if n_series != 10000:
        raise SystemExit(f"final10k fit needs all 10,000 series, cache has {n_series}")

    return {
        "PL": PL,
        "d": d,
        "mats": mats,
        "names": names,
        "keep_idx": keep_idx,
        "folds_sha256": folds_sha,
        "feature_manifest_sha256": stream_sha,
        "n_series": n_series,
        "n_rows": int(len(d.y)),
        "n_features": len(names),
    }


def main() -> int:
    args = parse_args()
    artifact_root = args.artifact_root.resolve()
    model_dir = args.model_dir.expanduser().resolve()

    home = Path.home()
    if not args.allow_icloud:
        for hint in ICLOUD_HINTS:
            if str(model_dir).startswith(str(home / hint)):
                raise SystemExit(
                    f"--model-dir {model_dir} is under ~/{hint}, which is "
                    "iCloud-synced; ~1 GB of checkpoints would sync while the job "
                    "runs. Pick a path outside it (e.g. ~/rt1320_local_fit)."
                )

    mod = load_submission(args.submission.resolve(), args.num_threads)
    print(f"submission : {args.submission}")
    print(f"artifact   : {artifact_root}")
    print(f"model dir  : {model_dir}")
    print(f"threads    : {mod.NUM_THREADS}  (fixed by the checkpoint contract)")

    ctx = verify_contract(mod, artifact_root)
    print(
        f"contract   : OK -- {ctx['n_series']} series, {ctx['n_rows']} rows, "
        f"{ctx['n_features']} features"
    )
    print(f"  folds    {ctx['folds_sha256']}")
    print(f"  features {ctx['feature_manifest_sha256']}")
    if args.check:
        print("\n--check given; nothing trained.")
        return 0

    PL, d = ctx["PL"], ctx["d"]
    mats, names, keep_idx = ctx["mats"], ctx["names"], ctx["keep_idx"]

    model_dir.mkdir(parents=True, exist_ok=True)
    out_root = mod._prepare_output_root(str(model_dir))
    final_of_row = mod._last_row_lookup(d)[d.sidx]
    purity = mod._fold_purity_report()
    if not purity["new_nested_passed"]:
        raise SystemExit(f"fold-purity plan failed: {purity}")

    # A single stage runs alone: the checkpoint logic makes the others resumable
    # independently, so there is no reason to force them into one invocation.
    want = STAGES if args.stage == "all" else (args.stage,)
    t0 = time.time()
    report: dict = {
        "schema": "sbr.rt1320_final10k_fit.local/1",
        "label": mod.LABEL,
        "timestamp": mod._utc_timestamp(),
        "where": "local",
        "artifact_root": str(artifact_root),
        "artifact_contract": mod._artifact_contract(),
        "package_versions": mod._package_versions(),
        "runtime_config": {"lightgbm_num_threads": mod.NUM_THREADS},
        "fold_purity": purity,
        "stages_run": list(want),
        "score_computed": False,
    }

    if "final" in want:
        print("\n=== 5 final teachers -> final student (produces model.txt.7) ===")
        report["teacher_final"] = [
            mod._train_final_teacher_checkpoint(
                PL, d, mats, names, keep_idx, final_of_row, out_root, fold
            )
            for fold in mod.FOLDS
        ]
        report["final_student"] = mod._train_final_student(
            PL, d, mats, names, keep_idx, out_root
        )
        print(f"model.txt.7 -> {out_root / 'model.txt.7'}")

    if "nested" in want:
        print("\n=== 20 nested teachers (the expensive half) ===")
        nested = []
        for outer in mod.FOLDS:
            for inner in mod.FOLDS:
                if inner == outer:
                    continue
                nested.append(
                    mod._train_teacher_checkpoint(
                        PL, d, mats, names, keep_idx, final_of_row,
                        out_root, outer, inner,
                    )
                )
        report["teacher_nested"] = nested

    if "student" in want:
        print("\n=== 5 student OOF folds ===")
        report["student_oof"] = [
            mod._train_student_oof_fold(PL, d, mats, names, keep_idx, out_root, outer)
            for outer in mod.FOLDS
        ]
        report["assembled_oof"] = mod._assemble_student_oof(out_root, d)

    if "calib" in want:
        print("\n=== student calibration (produces RT-1320_student_scdf.json) ===")
        cal = mod._fit_student_calibration(out_root, d)
        report["student_calibration"] = {
            "path": cal["path"],
            "sha256": cal["sha256"],
            "kind": cal["kind"],
            "time_coord": cal["time_coord"],
        }

    report["total_runtime_s"] = round(time.time() - t0, 1)
    report["artifact_files"] = sorted(p.name for p in out_root.iterdir() if p.is_file())
    mod._dump_json(out_root / "RT1320_LOCAL_FIT_RESULT.json", report)

    print("\n=== RT1320_LOCAL_FIT_RESULT ===")
    print(json.dumps(report, indent=2, sort_keys=True, default=mod._json_default))

    need = {"model.txt.7": "final student", "RT-1320_student_scdf.json": "calibration"}
    print("\ndeliverables:")
    for fname, what in need.items():
        p = out_root / fname
        mark = "OK " if p.exists() else "-- "
        extra = f"  sha256={mod._sha256_file(p)}" if p.exists() else "  (not yet built)"
        print(f"  {mark}{fname:32s} {what:16s}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
