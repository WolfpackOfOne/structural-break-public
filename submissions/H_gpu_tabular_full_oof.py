"""Crunch cloud full 5-fold OOF for GPU-01 TabM (RT-1258) and GPU-02 RealMLP
(RT-1259).

RESEARCH OOF ONLY - NOT A LEADERBOARD CANDIDATE.

This file intentionally exposes the same Real-Time Crunch public interface as
the deployed RT600 submission and the GPU hardware benchmark submission
(`submissions/G_gpu_tabular_benchmark.py`):

    train(datasets, model_directory_path)
    infer(datasets, model_directory_path)
    INFER_PARALLELISM

train() verifies RTX/CUDA, enforces the frozen config
(`research/reports/gpu_tabular_2026/FROZEN_GPU_CONFIG.json`) by content hash
and (when available) git commit/push purity, materializes the legal 500
causal-feature cache once over the full canonical dev population (folds
0..4, never the lockbox), trains TabM and RealMLP sequentially over all five
folds each via the existing checkpointed runners
(`research/scripts/gpu_tabular/{tabm,realmlp}_runner.py`), computes only
what is derivable from the candidates' own OOF predictions plus labels
already present in the materialized dev population (standalone TS-AUC,
dominant-cell AUC) after BOTH candidates have complete 5-fold OOF, and
prints a machine-readable result block. It never inspects a partial-fold or
single-learner predictive score.

The binding nested-replacement test (E0/E1/E2 vs the RT600 specialists and
the RT-401 matched clone) needs `research/oof/*.npy` control artifacts that
are `.gitignore`d and not present in a cold Crunch-cloud checkout; if they
happen to be present (e.g. a persistent research box) train() computes and
prints that too, otherwise it reports the binding test as deferred to a
local post-run step (`research/scripts/gpu_tabular/evaluate_gpu_oof.py`) --
see `research/reports/gpu_tabular_2026/FULL_OOF_PREREG.md` "Infrastructure
note".

infer() returns deterministic dummy probabilities only, exactly like the
benchmark submission, because this is research OOF, not a leaderboard
candidate.
"""

from __future__ import annotations

# ruff: noqa: UP006, UP035, UP045
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_k, "16")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GPU_SCRIPTS_DIR = _REPO_ROOT / "research" / "scripts" / "gpu_tabular"
for _p in (
    _REPO_ROOT / "src",
    _REPO_ROOT / "research" / "scripts",
    _GPU_SCRIPTS_DIR,
):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

# @crunch/keep:on
INFER_PARALLELISM = 1

LABEL = "RESEARCH OOF ONLY - NOT A LEADERBOARD CANDIDATE"
SEED = 1
DEV_FOLDS = (0, 1, 2, 3, 4)
MAX_TRAIN_ROWS = 1_000_000
RT_IDS = {"tabm": "RT-1258", "realmlp": "RT-1259"}
LEARNER_LABELS = {"tabm": "GPU-01 TabM", "realmlp": "GPU-02 RealMLP"}

# Mirrors common.FULL_MODULES. Duplicated as a module-level constant so the
# cache fingerprint can be built without importing repo modules (which
# require SBR_ROOT to already point at the artifact root).
FULL_MODULE_NAMES = (
    "m00_core",
    "m01_seq",
    "m02_dist",
    "m03_dyn",
    "m04_resid",
    "m06_loc",
    "m07_bayes",
)

# Provenance recorded and frozen alongside FROZEN_GPU_CONFIG.json. See
# research/reports/gpu_tabular_2026/FROZEN_GPU_CONFIG.json for the full
# distinction between these two SHAs.
_BENCHMARK_SUBMISSION_GIT_SHA = "0bcb7ae119e138ba3e47a8c0d807e6c3e3dfa560"
_CLOUD_EMBEDDED_FALLBACK_GIT_SHA = "e39a69c9eca8d35cbbf42211151f8e77ee3a3a96"
_CRUNCH_PROVENANCE = {"submission_id": 76357, "task_id": "run-3e834e0f"}

_FROZEN_CONFIG_REL = "research/reports/gpu_tabular_2026/FROZEN_GPU_CONFIG.json"
# sha256 of the committed FROZEN_GPU_CONFIG.json bytes as of commit d506aa6
# ("Freeze RTX 4090 TabM RealMLP configs"). This is the "enforce frozen config
# hash" check: it is verified independently of git metadata, which may be
# unavailable if the Crunch-uploaded tree has no .git directory.
_FROZEN_CONFIG_SHA256 = "1d28f1acfb8f524e1dcb8967a4605c19f48e99d152b4c57f01256cf90f17bb45"

FEATURE_WORKERS = int(os.environ.get("SBR_GPU_FEATURE_WORKERS", "16"))
FEATURE_CHUNK = int(os.environ.get("SBR_GPU_FEATURE_CHUNK", "64"))

SMOKE_ENV = "SBR_GPU_FULL_OOF_SMOKE"
KEEP_ARTIFACT_ENV = "SBR_GPU_KEEP_ARTIFACT_ROOT"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _torch_gpu_report_subprocess() -> dict[str, Any]:
    code = r"""
import json
import torch
if not torch.cuda.is_available():
    raise SystemExit("GPU REQUIRED: torch.cuda.is_available() is False")
props = torch.cuda.get_device_properties(0)
print(json.dumps({
    "cuda_available": True,
    "device": "cuda:0",
    "device_name": torch.cuda.get_device_name(0),
    "cuda_device_capability": torch.cuda.get_device_capability(0),
    "cuda_runtime_version": getattr(torch.version, "cuda", None),
    "torch_version": torch.__version__,
    "total_vram_gb": props.total_memory / 1e9,
}, sort_keys=True))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(msg or "GPU REQUIRED: torch CUDA preflight failed")
    return json.loads(proc.stdout)


def _print_device_report(label: str, report: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "step": label,
                "cuda_available": report.get("cuda_available"),
                "gpu_name": report.get("device_name"),
                "cuda_version": report.get("cuda_runtime_version"),
                "torch_version": report.get("torch_version"),
                "total_vram_gb": report.get("total_vram_gb"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _series_values(obj: Any) -> Any:
    import numpy as np

    if hasattr(obj, "columns") and "value" in obj.columns:
        return obj["value"].to_numpy(dtype=np.float32)
    if isinstance(obj, dict) and "value" in obj:
        return np.asarray(obj["value"], dtype=np.float32).reshape(-1)
    return np.asarray(obj, dtype=np.float32).reshape(-1)


def _normalise_tau(value: Any, n_online: int) -> int:
    import math

    import numpy as np

    if value is None:
        return -1
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return -1
    tau = int(value)
    if tau < -1 or tau >= n_online:
        raise RuntimeError(f"Invalid tau_index={tau}; expected -1 or [0, {n_online})")
    return tau


def _materialize_full_dev_store(
    datasets: Iterable[Tuple[int, List[float], List[float], Optional[int]]],
    artifact_root: Path,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Materialize ALL FIVE canonical dev folds (fold != -1), never the lockbox.

    Full 5-fold OOF needs every dev fold's series available so that each
    outer fold k can train on folds != k and predict on fold k -- unlike the
    hardware benchmark, which only needed a single held-out fold's
    outer-training population.
    """
    import numpy as np
    import pandas as pd

    folds_src = _REPO_ROOT / "research" / "folds" / "folds.parquet"
    canonical = pd.read_parquet(folds_src).sort_values("id").reset_index(drop=True)
    selected_folds = canonical[canonical["fold"] != -1].copy()
    selected_ids = set(int(x) for x in selected_folds["id"].to_numpy())
    lockbox_ids = set(
        int(x) for x in canonical.loc[canonical["fold"] == -1, "id"].to_numpy()
    )

    seen: dict[int, tuple[Any, Any, int]] = {}
    n_total_items = 0
    lockbox_touched = 0
    for item in datasets:
        n_total_items += 1
        if len(item) != 4:
            raise RuntimeError(
                "Expected Crunch real-time train item "
                "(id, x_historical, x_online, tau_index)"
            )
        sid_raw, x_historical, x_online, tau_raw = item
        sid = int(sid_raw)
        if sid in lockbox_ids:
            lockbox_touched += 1
            continue
        if sid not in selected_ids:
            continue
        hist = _series_values(x_historical)
        online = _series_values(x_online)
        tau = _normalise_tau(tau_raw, len(online))
        seen[sid] = (hist, online, tau)

    # `datasets` is Crunch's full real-time training stream; it necessarily
    # contains lockbox-fold ids (Crunch owns the split, not this code). Not
    # touching the lockbox means never storing, featurizing, selecting, or
    # training on those series -- which the `continue` above already
    # guarantees -- not that their ids may never flow past in this loop.
    # G_gpu_tabular_benchmark.py (proven on the real cloud, submission 76357,
    # lockbox_touched=false) applies the same skip-without-raising pattern.

    missing = [int(x) for x in selected_folds["id"].to_numpy() if int(x) not in seen]
    if require_complete and missing:
        raise RuntimeError(
            "Crunch train data is missing canonical dev-fold ids; "
            f"first missing ids={missing[:10]} total_missing={len(missing)}"
        )
    if not require_complete:
        selected_folds = selected_folds[selected_folds["id"].isin(seen.keys())].copy()

    store_dir = artifact_root / "cache" / "store"
    folds_dir = artifact_root / "research" / "folds"
    store_dir.mkdir(parents=True, exist_ok=True)
    folds_dir.mkdir(parents=True, exist_ok=True)

    total_values = int(
        sum(
            len(seen[int(sid)][0]) + len(seen[int(sid)][1])
            for sid in selected_folds["id"]
        )
    )
    values = np.lib.format.open_memmap(
        store_dir / "values.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_values,),
    )

    rows = []
    pos = 0
    for sid_raw in selected_folds["id"].to_numpy():
        sid = int(sid_raw)
        hist, online, tau = seen[sid]
        n_hist = int(len(hist))
        n_online = int(len(online))
        values[pos : pos + n_hist] = hist
        values[pos + n_hist : pos + n_hist + n_online] = online
        rows.append((sid, pos, n_hist, n_online, tau, int(tau >= 0)))
        pos += n_hist + n_online
    values.flush()

    meta = pd.DataFrame(
        rows,
        columns=["id", "off", "n_hist", "n_online", "tau_index", "has_break"],
    )
    meta.to_parquet(store_dir / "meta.parquet", index=False)

    fold_cols = [
        c
        for c in selected_folds.columns
        if c not in {"n_hist", "n_online", "tau_index", "has_break"}
    ]
    folds_out = selected_folds[fold_cols].merge(
        meta[["id", "n_hist", "n_online", "tau_index", "has_break"]],
        on="id",
        how="left",
    )
    folds_out.to_parquet(folds_dir / "folds.parquet", index=False)

    return {
        "n_train_items_seen": int(n_total_items),
        "store_dir": str(store_dir),
        "folds_path": str(folds_dir / "folds.parquet"),
        "selected_folds": list(DEV_FOLDS),
        "selected_series": int(len(meta)),
        "selected_online_rows": int(meta["n_online"].sum()),
        "selected_total_values": int(total_values),
        "lockbox_series_materialized": 0,
        "lockbox_ids_seen_in_datasets": int(lockbox_touched),
    }


def _build_feature_cache(artifact_root: Path, store_manifest: dict[str, Any]) -> dict[str, Any]:
    from common import FULL_MODULES

    from sbr.features.driver import build_features

    # Guard against drift between the local mirror used for cache
    # fingerprinting and the canonical list the features are actually built
    # from; a silent divergence would let a stale cache be reused.
    if tuple(FULL_MODULES) != FULL_MODULE_NAMES:
        raise RuntimeError(
            f"Module list drift: common.FULL_MODULES={tuple(FULL_MODULES)} but "
            f"FULL_MODULE_NAMES={FULL_MODULE_NAMES}"
        )

    features_dir = artifact_root / "cache" / "features"
    t0 = time.time()
    manifest = build_features(
        list(FULL_MODULES),
        store=store_manifest["store_dir"],
        out=str(features_dir),
        workers=FEATURE_WORKERS,
        chunk=FEATURE_CHUNK,
        limit=0,
    )
    manifest["wall_seconds"] = float(time.time() - t0)
    manifest["feature_modules"] = list(FULL_MODULES)
    manifest["feature_count"] = int(sum(v["columns"] for v in manifest["modules"].values()))
    if manifest["feature_count"] != 500:
        raise RuntimeError(f"Expected 500 causal features, built {manifest['feature_count']}")
    return manifest


_PREPARED_MANIFEST_NAME = "PREPARED_MANIFEST.json"


def _persist_root(model_directory_path: str) -> Path:
    """Durable working root inside model_directory_path.

    Crunch hands train() a model directory that (unlike a per-run temp dir)
    can be carried between runs of the same submission, so putting the
    materialized store, the 500-feature cache, and the per-fold prediction
    checkpoints here lets a re-run skip work that already completed instead
    of rebuilding features and retraining folds 0..k-1 from scratch.
    """
    return Path(model_directory_path).resolve() / "gpu_tabular_work"


def _preparation_fingerprint(store_manifest: dict[str, Any]) -> dict[str, Any]:
    """Identity of a materialized store+feature cache. A cache is reusable
    only if every field matches the run that would otherwise build it."""
    return {
        "frozen_config_sha256": _FROZEN_CONFIG_SHA256,
        "feature_modules": list(FULL_MODULE_NAMES),
        "feature_count": 500,
        "dev_folds": list(DEV_FOLDS),
        "series": store_manifest["selected_series"],
        "online_rows": store_manifest["selected_online_rows"],
        "total_values": store_manifest["selected_total_values"],
        "max_train_rows": MAX_TRAIN_ROWS,
        "seed": SEED,
    }


def _valid_cached_preparation(
    persist_root: Path, expected: dict[str, Any]
) -> dict[str, Any] | None:
    """Return the cached preparation manifest iff it is present, complete,
    and describes exactly the population/config this run needs."""
    manifest_path = persist_root / _PREPARED_MANIFEST_NAME
    if not manifest_path.exists():
        return None
    try:
        cached = json.loads(manifest_path.read_text())
    except Exception:
        return None
    if cached.get("fingerprint") != expected:
        print(
            "Cached feature/store manifest does not match this run's contract; rebuilding.",
            flush=True,
        )
        return None
    store_dir = persist_root / "cache" / "store"
    features_dir = persist_root / "cache" / "features"
    required = [
        store_dir / "values.npy",
        store_dir / "meta.parquet",
        persist_root / "research" / "folds" / "folds.parquet",
    ]
    required += [features_dir / f"{m}.npy" for m in FULL_MODULE_NAMES]
    required += [features_dir / f"{m}.cols.json" for m in FULL_MODULE_NAMES]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print(
            f"Cached preparation incomplete ({len(missing)} missing artifacts); rebuilding.",
            flush=True,
        )
        return None
    return cached


def _prepare_artifact_root(
    datasets: Iterable[Tuple[int, List[float], List[float], Optional[int]]],
    model_directory_path: str,
) -> tuple[Path, dict[str, Any]]:
    """Materialize the store + 500-feature cache into a durable root, reusing
    a valid cache from a previous run of this submission when one exists.

    Falls back to a fresh temp dir if the model directory is not usable, so a
    read-only/absent model dir degrades to the previous behaviour rather than
    failing the run.
    """
    persist_root = _persist_root(model_directory_path)
    try:
        persist_root.mkdir(parents=True, exist_ok=True)
        probe = persist_root / ".writable"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        artifact_root = persist_root
        durable = True
    except Exception as exc:
        artifact_root = Path(tempfile.mkdtemp(prefix="sbr_gpu_tabular_full_oof_")).resolve()
        durable = False
        print(
            f"Model directory not usable as a durable work root ({exc}); "
            f"falling back to ephemeral {artifact_root}.",
            flush=True,
        )

    os.environ["SBR_ROOT"] = str(artifact_root)
    os.environ["SBR_STORE"] = str(artifact_root / "cache" / "store")
    os.environ["SBR_FEATURES"] = str(artifact_root / "cache" / "features")

    t0 = time.time()
    # The store scan is cheap relative to the feature build, and it is what
    # produces the fingerprint the cache is validated against, so it always
    # runs; only the expensive feature build is skipped on a cache hit.
    store_manifest = _materialize_full_dev_store(datasets, artifact_root)
    expected = _preparation_fingerprint(store_manifest)

    cached = _valid_cached_preparation(artifact_root, expected) if durable else None
    if cached is not None:
        print(
            "Reusing cached 500-feature matrix from a previous run "
            f"({expected['series']} series, {expected['online_rows']} online rows); "
            "skipping feature build.",
            flush=True,
        )
        feature_manifest = cached["features"]
        feature_manifest["reused_from_previous_run"] = True
    else:
        feature_manifest = _build_feature_cache(artifact_root, store_manifest)
        feature_manifest["reused_from_previous_run"] = False

    prepared = {
        "artifact_root": str(artifact_root),
        "durable_work_root": durable,
        "store": store_manifest,
        "features": feature_manifest,
        "seconds": float(time.time() - t0),
    }
    if durable:
        (artifact_root / _PREPARED_MANIFEST_NAME).write_text(
            json.dumps(
                {"fingerprint": expected, "features": feature_manifest},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return artifact_root, prepared


def _report_existing_checkpoints(output_root: Path) -> None:
    """Log which fold checkpoints already exist so a resumed run's skipped
    work is visible. Reports fold identity only -- never a predictive score."""
    for learner in ("tabm", "realmlp"):
        present = [
            f for f in DEV_FOLDS if (output_root / learner / f"fold_{f}_pred.npy").exists()
        ]
        if present:
            print(
                f"Resume: {learner} already has checkpoints for folds {present}; "
                "these will be skipped by --resume.",
                flush=True,
            )
        else:
            print(f"Resume: {learner} has no existing fold checkpoints.", flush=True)


def _frozen_config_path() -> Path:
    return _REPO_ROOT / _FROZEN_CONFIG_REL


def _verify_frozen_config_hash() -> str:
    path = _frozen_config_path()
    if not path.exists():
        raise RuntimeError(f"MISSING FROZEN CONFIG: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != _FROZEN_CONFIG_SHA256:
        raise RuntimeError(
            "FROZEN CONFIG HASH MISMATCH: "
            f"expected {_FROZEN_CONFIG_SHA256}, got {digest}. "
            "The frozen TabM/RealMLP configuration in the uploaded tree does not match "
            "the committed, benchmarked configuration. Refusing to train."
        )
    return digest


def _git_purity_flags() -> tuple[list[str], str]:
    """Strict git commit/push purity when a genuinely usable git checkout is
    present; content-hash-only fallback otherwise (the hash check in
    _verify_frozen_config_hash always runs regardless of this branch).

    A bare `.git` path check is not enough: a git *worktree* checkout has a
    `.git` file (not a directory) containing a `gitdir:` pointer to an
    absolute path on the machine the worktree was created on. Crunch uploads
    that file verbatim, so on the cloud `.git` "exists" but points nowhere,
    and `git ls-files`/`rev-parse` inside `require_committed_and_pushed` then
    fails -- the exact false-strict-mode bug that broke the first cloud run.
    Actually invoke git and require it to work, not just check for a path.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--is-inside-work-tree"],
            text=True,
            capture_output=True,
            timeout=10,
        )
    except Exception:
        proc = None
    if proc is not None and proc.returncode == 0 and proc.stdout.strip() == "true":
        return [], "git_purity_strict"
    return (
        ["--allow-uncommitted-config", "--allow-unpushed-config"],
        "content_hash_only_no_usable_git_tree",
    )


# Exit codes subprocess.run reports as -N when the child was killed by signal
# N (SIGSEGV=11, SIGABRT=6, SIGBUS=7, SIGKILL=9 -- the last typically an OS/
# cgroup OOM kill, not a Python-catchable CUDA OOM). These are consistent
# with transient GPU-driver/allocator crashes on cloud boxes rather than a
# deterministic bug in the frozen model code; retry a few times before
# treating as a genuine TECHNICAL_FAILURE (spec section 16).
_SIGNAL_KILL_RETRY_LIMIT = 2


def _run_runner(
    learner: str,
    artifact_root: Path,
    output_root: Path,
    extra_flags: list[str],
    folds: list[int],
) -> dict[str, Any]:
    runner_path = _GPU_SCRIPTS_DIR / f"{learner}_runner.py"
    args = [
        sys.executable,
        str(runner_path),
        "--config",
        str(_frozen_config_path()),
        "--output-root",
        str(output_root),
        "--artifact-root",
        str(artifact_root),
        "--folds",
        ",".join(str(f) for f in folds),
        "--resume",
        *extra_flags,
    ]
    env = dict(os.environ)
    pythonpath_parts = [
        str(_GPU_SCRIPTS_DIR),
        str(_REPO_ROOT / "research" / "scripts"),
        str(_REPO_ROOT / "src"),
    ]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    attempt = 0
    while True:
        attempt += 1
        print(
            f"Running {learner} fold(s) {folds} (attempt {attempt}): {' '.join(args)}",
            flush=True,
        )
        proc = subprocess.run(args, env=env, text=True, capture_output=True, check=False)
        for line in (proc.stdout or "").splitlines():
            print(f"[{learner}] {line}", flush=True)
        if proc.returncode == 0:
            break
        signal_killed = proc.returncode < 0
        if signal_killed and attempt <= _SIGNAL_KILL_RETRY_LIMIT:
            print(
                f"[{learner}] fold(s) {folds} killed by signal {-proc.returncode} "
                f"(attempt {attempt}/{_SIGNAL_KILL_RETRY_LIMIT + 1}); retrying -- "
                "no hyperparameter or config change, same frozen invocation.",
                flush=True,
            )
            continue
        raise RuntimeError(
            f"{learner} runner failed on fold(s) {folds} (exit {proc.returncode}) after "
            f"{attempt} attempt(s). stderr tail:\n"
            + "\n".join((proc.stderr or "").splitlines()[-40:])
        )
    # The runner's final line is a JSON summary: {"learner","folds","assembly","score_computed"}.
    summary = None
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                summary = json.loads(_last_json_block(proc.stdout))
            except Exception:
                summary = None
            break
    if summary is None:
        raise RuntimeError(f"{learner} runner produced no parseable summary JSON")
    return summary


def _run_all_folds(
    learner: str,
    artifact_root: Path,
    output_root: Path,
    extra_flags: list[str],
) -> dict[str, Any]:
    """Run each dev fold in its own subprocess rather than one process for
    all five. A single long-lived process training TabM/RealMLP through five
    sequential ~44min (TabM) / ~11min (RealMLP) fold runs risks accumulating
    CUDA-context/allocator state across folds; per-fold process isolation
    with --resume (each fold checkpoints to disk before the next starts, so a
    crash mid-fold N only repeats fold N) is a standard, hyperparameter-free
    mitigation for exactly this failure shape. `assemble_oof` (called at the
    end of every runner invocation) reflects cumulative on-disk state across
    all folds regardless of which fold(s) this particular call requested, so
    the final fold's summary is the authoritative one to return.
    """
    all_fold_entries: list[Any] = []
    summary: dict[str, Any] | None = None
    for fold in DEV_FOLDS:
        summary = _run_runner(learner, artifact_root, output_root, extra_flags, [fold])
        all_fold_entries.extend(summary.get("folds") or [])
    assert summary is not None
    # `assembly` from the final call already reflects cumulative on-disk state
    # across all folds; `folds` is per-invocation, so stitch the per-fold
    # status entries back together for spec section 15's per-fold reporting.
    summary = dict(summary)
    summary["folds"] = all_fold_entries
    return summary


def _last_json_block(stdout: str) -> str:
    # main() prints one indented JSON object as the final stdout write; find
    # the last top-level '{' and take everything from there.
    idx = stdout.rfind("\n{\n")
    if idx == -1:
        idx = stdout.find("{")
    else:
        idx += 1
    return stdout[idx:]


def _standalone_and_diagnostics(
    learner: str, artifact_root: Path, output_root: Path
) -> dict[str, Any]:
    import evaluate_gpu_oof
    from common import configure_roots, load_eval_context

    configure_roots(artifact_root)
    c = load_eval_context(artifact_root)
    oof_path = output_root / f"{learner}_oof.npy"
    candidate = evaluate_gpu_oof.ensure_complete_oof(oof_path, c, learner)
    standalone = evaluate_gpu_oof.score_vec(c, candidate)
    dom_rows = evaluate_gpu_oof.cell_rows(c, c.dev, "dominant_cell")
    from sbr.metric import ts_auc_flat

    dominant_cell_auc = float(
        ts_auc_flat(candidate[dom_rows], c.d.y[dom_rows], c.d.t[dom_rows])
    )
    return {
        "standalone": standalone,
        "dominant_cell_auc": dominant_cell_auc,
    }


def _try_binding_replacement_test(artifact_root: Path, output_root: Path) -> dict[str, Any]:
    """Attempt the full E0/E1/E2 nested replacement test. Requires
    research/oof/*.npy RT600 specialist + RT-401 clone control artifacts,
    which are .gitignore'd and normally absent from a cold Crunch checkout.
    Returns {"computed": False, "reason": ...} when unavailable rather than
    failing the whole run."""
    import evaluate_gpu_oof
    from common import configure_roots, load_control_oof, load_eval_context

    try:
        configure_roots(_REPO_ROOT)
        controls = load_control_oof(_REPO_ROOT)
    except SystemExit as exc:
        return {
            "computed": False,
            "reason": (
                "RT600 specialist / RT-401 control OOF artifacts are not present in this "
                "environment (they are .gitignore'd and not part of the Crunch-uploaded "
                f"tree). {exc}. Run research/scripts/gpu_tabular/evaluate_gpu_oof.py "
                "locally after copying tabm_oof.npy/realmlp_oof.npy back into "
                "research/oof/gpu_tabular_2026/ to compute the binding replacement test."
            ),
        }

    configure_roots(artifact_root)
    c = load_eval_context(artifact_root)
    out: dict[str, Any] = {"computed": True, "learners": {}}
    for learner in ("tabm", "realmlp"):
        out["learners"][learner] = evaluate_gpu_oof.evaluate_learner(
            learner, c, controls, output_root
        )
    return out


def _finalize_result(
    tabm_summary: dict[str, Any],
    realmlp_summary: dict[str, Any],
    tabm_diag: dict[str, Any],
    realmlp_diag: dict[str, Any],
    binding: dict[str, Any],
    preparation: dict[str, Any],
    total_seconds: float,
) -> dict[str, Any]:
    from common import env_versions

    def learner_block(key: str, summary: dict[str, Any], diag: dict[str, Any]) -> dict[str, Any]:
        block = {
            "rt_id": RT_IDS[key],
            "label": LEARNER_LABELS[key],
            "folds": summary.get("folds"),
            "assembly": summary.get("assembly"),
            "standalone": diag["standalone"],
            "dominant_cell_auc": diag["dominant_cell_auc"],
            "score_computed": True,
        }
        if binding.get("computed"):
            bl = binding["learners"][key]
            block.update(
                {
                    "rho_vs_rt401": bl["rho_vs_rt401"],
                    "pair_flow_vs_E0": bl["pair_flow_vs_E0"],
                    "pair_flow_vs_E1_clone": bl["pair_flow_vs_E1_clone"],
                    "ensemble": bl["ensemble"],
                    "verdict": bl["verdict"],
                }
            )
        else:
            block.update(
                {
                    "binding_replacement_test": {
                        "computed": False,
                        "reason": binding.get("reason"),
                    },
                    "verdict": "PENDING_LOCAL_BINDING_EVALUATION",
                }
            )
        return block

    return {
        "label": LABEL,
        "timestamp": _utc_timestamp(),
        "benchmark_submission_git_sha": _BENCHMARK_SUBMISSION_GIT_SHA,
        "cloud_reported_embedded_fallback_git_sha": _CLOUD_EMBEDDED_FALLBACK_GIT_SHA,
        "crunch_provenance": _CRUNCH_PROVENANCE,
        "frozen_config_path": _FROZEN_CONFIG_REL,
        "frozen_config_sha256": _FROZEN_CONFIG_SHA256,
        "package_versions": env_versions(),
        "feature_preparation": {
            "series": preparation["store"]["selected_series"],
            "online_rows": preparation["store"]["selected_online_rows"],
            "wall_seconds": preparation["features"]["wall_seconds"],
            "feature_count": preparation["features"]["feature_count"],
        },
        "total_runtime_seconds": total_seconds,
        "no_combination_this_program": True,
        "combination_with_catboost_or_rt1257": {
            "run": False,
            "reason": (
                "not authorized in this program; separate preregistered experiment "
                "only if a neural learner survives"
            ),
        },
        "learners": {
            "tabm": learner_block("tabm", tabm_summary, tabm_diag),
            "realmlp": learner_block("realmlp", realmlp_summary, realmlp_diag),
        },
    }


def _print_result_block(result: dict[str, Any]) -> None:
    print("=== GPU_TABULAR_OOF_RESULTS_BEGIN ===", flush=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("=== GPU_TABULAR_OOF_RESULTS_END ===", flush=True)


def _write_model_artifacts(model_directory_path: str, result: dict[str, Any]) -> None:
    model_dir = Path(model_directory_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "RESEARCH_OOF_ONLY_NOT_LEADERBOARD.txt").write_text(
        "This run produces research OOF predictions only. infer() returns dummy probabilities.\n",
        encoding="utf-8",
    )
    (model_dir / "GPU_TABULAR_OOF_RESULTS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy_oof_predictions(output_root: Path, model_directory_path: str) -> None:
    model_dir = Path(model_directory_path) / "oof"
    model_dir.mkdir(parents=True, exist_ok=True)
    for learner in ("tabm", "realmlp"):
        src = output_root / f"{learner}_oof.npy"
        if src.exists():
            shutil.copy2(src, model_dir / f"{learner}_oof.npy")


def _run_smoke() -> None:
    """Exercise the Crunch interface, frozen-config enforcement, fold purity,
    checkpoint metadata, and the evaluator's incomplete-OOF refusal, all
    without CUDA or real neural training. See
    research/reports/gpu_tabular_2026/FULL_OOF_PREREG.md for the authorized
    smoke-mode contract."""
    import numpy as np
    import pandas as pd

    print("SMOKE MODE: no GPU, no real training", flush=True)

    # 1. Frozen-config hash + git purity enforcement (real check against the
    #    actual committed file -- this validates the real integrity chain).
    digest = _verify_frozen_config_hash()
    flags, mode = _git_purity_flags()
    print(
        json.dumps({"smoke_check": "frozen_config_hash", "sha256": digest, "status": "PASS"})
    )
    print(
        json.dumps(
            {"smoke_check": "git_purity_mode", "mode": mode, "extra_flags": flags, "status": "PASS"}
        )
    )

    # 2. Fold purity on a tiny synthetic slice of real canonical ids spanning
    #    two distinct dev folds (never the lockbox).
    canonical = pd.read_parquet(_REPO_ROOT / "research" / "folds" / "folds.parquet")
    dev = canonical[canonical["fold"] != -1]
    lockbox_ids = set(int(x) for x in canonical.loc[canonical["fold"] == -1, "id"].to_numpy())
    fold_ids = {}
    for f in (0, 1):
        ids = dev.loc[dev["fold"] == f, "id"].to_numpy()[:5]
        fold_ids[f] = [int(x) for x in ids]
    smoke_ids = set(fold_ids[0]) | set(fold_ids[1])
    assert not smoke_ids & lockbox_ids, "lockbox id leaked into smoke sample"
    train_ids_for_fold0 = set(fold_ids[1])  # "train" = the other fold, in this 2-fold slice
    val_ids_for_fold0 = set(fold_ids[0])
    assert not (train_ids_for_fold0 & val_ids_for_fold0), "fold purity violated"
    print(
        json.dumps(
            {
                "smoke_check": "fold_purity",
                "fold0_val": fold_ids[0],
                "fold1_train": fold_ids[1],
                "status": "PASS",
            }
        )
    )

    # 3. Checkpoint write/read round trip via the real helpers.
    from common import assemble_oof, completed_fold_valid, save_completed_fold

    tmp_out = Path(tempfile.mkdtemp(prefix="sbr_gpu_tabular_full_oof_smoke_"))
    try:
        fake_pred = np.full(5, 0.5, dtype=np.float32)
        fake_meta = {"fold": 0, "smoke": True, "score_computed": False}
        save_completed_fold("tabm", 0, fake_pred, fake_meta, tmp_out)
        valid = completed_fold_valid("tabm", 0, 5, tmp_out)
        assert valid, "checkpoint round-trip failed"
        print(json.dumps({"smoke_check": "checkpoint_roundtrip", "status": "PASS"}))

        # 4. Evaluator refuses an incomplete OOF (only fold 0 of 5 present).
        import evaluate_gpu_oof

        class _FakeData:
            y = np.zeros(25, dtype=np.int64)

            def rows_for(self, folds):
                return np.concatenate([np.arange(f * 5, f * 5 + 5) for f in folds])

        class _FakeCtx:
            d = _FakeData()
            rows = {f: np.arange(f * 5, f * 5 + 5) for f in range(5)}

        # assemble_oof(learner, d, output_root) wants the Data-shaped object
        # directly (.y, .rows_for); ensure_complete_oof(path, c, learner) wants
        # the Ctx-shaped wrapper (.d.y, .rows[f]) -- deliberately different
        # shapes, matching the real common.py / evaluate_gpu_oof.py contracts.
        assembly = assemble_oof("tabm", _FakeData(), tmp_out)
        assert assembly["completed_folds"] == [0]
        assert not assembly["all_folds_complete"]
        raised = False
        try:
            evaluate_gpu_oof.ensure_complete_oof(tmp_out / "tabm_oof.npy", _FakeCtx(), "tabm")
        except SystemExit:
            raised = True
        assert raised, "evaluator did not refuse a partial OOF vector"
        print(json.dumps({"smoke_check": "evaluator_refuses_partial_oof", "status": "PASS"}))
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)

    # 5. infer() is deterministic dummy output and never touches lockbox/test.
    dummy_datasets = [([1.0, 2.0], [3.0, 4.0, 5.0])]
    gen = infer(dummy_datasets, tempfile.mkdtemp(prefix="sbr_gpu_tabular_full_oof_smoke_infer_"))
    values = list(gen)
    assert values == [None, 0.5, 0.5, 0.5], f"infer() smoke output unexpected: {values}"
    print(json.dumps({"smoke_check": "infer_deterministic_dummy", "status": "PASS"}))

    print(
        json.dumps(
            {"smoke_mode": "ALL_CHECKS_PASSED", "score_computed": False},
            indent=2,
            sort_keys=True,
        )
    )


def train(
    datasets: List[Tuple[int, List[float], List[float], Optional[int]]],
    model_directory_path: str,
) -> None:
    print(LABEL, flush=True)

    if os.environ.get(SMOKE_ENV) == "1":
        _run_smoke()
        return

    t_start = time.time()
    digest = _verify_frozen_config_hash()
    extra_flags, purity_mode = _git_purity_flags()
    print(
        json.dumps(
            {
                "frozen_config_sha256": digest,
                "git_purity_mode": purity_mode,
                "extra_flags": extra_flags,
            }
        ),
        flush=True,
    )

    preflight = _torch_gpu_report_subprocess()
    _print_device_report("preflight", preflight)

    artifact_root: Path | None = None
    durable_root = False
    try:
        artifact_root, preparation = _prepare_artifact_root(datasets, model_directory_path)
        durable_root = bool(preparation.get("durable_work_root"))
        # Fold checkpoints live under the same root, so on a durable root a
        # re-run resumes: already-complete folds are skipped by --resume
        # instead of retrained.
        output_root = artifact_root / "research" / "oof" / "gpu_tabular_2026"
        _report_existing_checkpoints(output_root)

        print("Running GPU-01 TabM full 5-fold OOF (RT-1258)", flush=True)
        tabm_summary = _run_all_folds("tabm", artifact_root, output_root, extra_flags)
        if not tabm_summary.get("assembly", {}).get("all_folds_complete"):
            raise RuntimeError("TabM did not complete all 5 folds")

        print("Running GPU-02 RealMLP full 5-fold OOF (RT-1259)", flush=True)
        realmlp_summary = _run_all_folds("realmlp", artifact_root, output_root, extra_flags)
        if not realmlp_summary.get("assembly", {}).get("all_folds_complete"):
            raise RuntimeError("RealMLP did not complete all 5 folds")

        # Only after BOTH candidates have complete 5-fold OOF is anything
        # predictive computed.
        print("Both learners complete; computing standalone diagnostics", flush=True)
        tabm_diag = _standalone_and_diagnostics("tabm", artifact_root, output_root)
        realmlp_diag = _standalone_and_diagnostics("realmlp", artifact_root, output_root)
        binding = _try_binding_replacement_test(artifact_root, output_root)

        total_seconds = float(time.time() - t_start)
        result = _finalize_result(
            tabm_summary,
            realmlp_summary,
            tabm_diag,
            realmlp_diag,
            binding,
            preparation,
            total_seconds,
        )
        _copy_oof_predictions(output_root, model_directory_path)
        _write_model_artifacts(model_directory_path, result)
        _print_result_block(result)
    finally:
        # A durable root is the resume cache -- deleting it would defeat the
        # point. Only an ephemeral temp root is cleaned up.
        if (
            artifact_root is not None
            and not durable_root
            and os.environ.get(KEEP_ARTIFACT_ENV) != "1"
        ):
            shutil.rmtree(artifact_root, ignore_errors=True)


def infer(
    datasets: Iterable[Tuple[List[float], Iterable[float]]],
    model_directory_path: str,
):
    yield

    for _x_historical, x_online in datasets:
        for _point in x_online:
            yield 0.5
