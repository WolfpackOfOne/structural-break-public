"""Build research/REPRODUCIBILITY_MANIFEST.json.

Records everything needed to say what produced a number: code identity,
environment identity, data identity, fold identity, feature identity.
Run it whenever the checkpoint moves.  Never hand-edit the output.
"""
from __future__ import annotations

import hashlib, json, os, platform, subprocess, sys, time
import numpy as np

ROOT = "/home/claude/sb"
RAW = "/mnt/user-data/uploads/crunch 2026/structural-break/data"


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_dir_of(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(os.path.basename(p).encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest()


def git(*a):
    return subprocess.check_output(["git", "-C", ROOT, *a]).decode().strip()


def main():
    out = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    out["code"] = {
        "git_sha": git("rev-parse", "HEAD"),
        "git_sha_short": git("rev-parse", "--short", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "checkpoint_tag": "research-checkpoint-20260818-1",
        "checkpoint_sha": git("rev-parse", "research-checkpoint-20260818-1^{commit}"),
        "dirty": bool(git("status", "--porcelain")),
    }

    import lightgbm, pandas, pyarrow, scipy, sklearn
    out["environment"] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "n_cpu": os.cpu_count(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "pyarrow": pyarrow.__version__,
        "numpy_blas": str(np.__config__.show(mode="dicts").get("Build Dependencies", {}).get("blas", {}).get("name", "unknown")),
    }

    raw = {n: sha256_file(f"{RAW}/{n}") for n in
           ("X_train.parquet", "y_train.parquet", "y_train_index.parquet")
           if os.path.exists(f"{RAW}/{n}")}
    out["data"] = {"raw_sha256": raw,
                   "store_meta_sha256": sha256_file(f"{ROOT}/cache/store/meta.parquet"),
                   "store_values_sha256": sha256_file(f"{ROOT}/cache/store/values.npy"),
                   "store_values_bytes": os.path.getsize(f"{ROOT}/cache/store/values.npy")}

    fold_files = [f"{ROOT}/research/folds/{f}" for f in sorted(os.listdir(f"{ROOT}/research/folds"))
                  if f.endswith(".parquet")]
    out["folds"] = {os.path.basename(p): sha256_file(p) for p in fold_files}
    out["folds"]["canonical_id_fold_sha256"] = json.load(
        open(f"{ROOT}/research/folds/folds_summary.json"))["sha256"]

    featdir = f"{ROOT}/cache/features"
    if os.path.isdir(featdir):
        cols = {}
        for f in sorted(os.listdir(featdir)):
            if f.endswith(".cols.json"):
                m = f[:-len(".cols.json")]
                cols[m] = json.load(open(f"{featdir}/{f}"))["cols"]
        flat = [f"{m}::{c}" for m in sorted(cols) for c in cols[m]]
        out["features"] = {
            "modules": sorted(cols),
            "n_columns_per_module": {m: len(v) for m, v in cols.items()},
            "n_columns_total": len(flat),
            "batch_feature_manifest_sha256": hashlib.sha256("\n".join(flat).encode()).hexdigest(),
        }

    try:
        from sbr.stream.engine import StreamEngine
        from sbr.store import load_store
        st = load_store()
        h, _, _ = st.series(0)
        m = StreamEngine().fit_historical(h).manifest()
        out["streaming"] = {"n_features": m["n_features"],
                            "stream_feature_manifest_sha256": m["feature_manifest_sha256"],
                            "module_order": m["modules"]}
    except Exception as e:            # pragma: no cover
        out["streaming"] = {"error": repr(e)}

    src = []
    for d, _, fs in os.walk(f"{ROOT}/src/sbr"):
        if "__pycache__" in d:
            continue
        src += [os.path.join(d, f) for f in fs if f.endswith(".py")]
    out["source"] = {"n_files": len(src), "sbr_source_sha256": sha256_dir_of(src)}

    out["seeds"] = {"fold_seed": 20260818, "alt_fold_seeds": [20260901, 20260902, 20260903],
                    "model_seeds_protocol": [0, 1, 7, 42, 2026]}

    p = f"{ROOT}/research/REPRODUCIBILITY_MANIFEST.json"
    json.dump(out, open(p, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "features"}, indent=2)[:2600])
    print("\nwrote", p)


if __name__ == "__main__":
    main()
