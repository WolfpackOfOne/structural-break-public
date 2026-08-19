"""Train the DEPLOYABLE model and write a self-describing model directory.

Trains on dev folds only (the original lockbox stays untouched), on the SAME
cached features the research pipeline uses, and saves:
    model.txt.0 ...   LightGBM booster(s), text format
    manifest.json     modules, ordered columns, feature-manifest SHA, code SHA,
                      per-booster column slice, calibration payload

The manifest SHA is computed from the STREAMING engine's column list, so the
model can only ever load against an engine that produces exactly those columns.
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time
import numpy as np

sys.path.insert(0, "/home/claude/sb/src")
import lightgbm as lgb
from sbr.pipeline import Data, load_features, _stack
from sbr.stream.engine import StreamEngine

FULL = ["m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes"]
CHAMP_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63,
                    min_data_in_leaf=300, feature_fraction=0.5, bagging_fraction=0.7,
                    bagging_freq=1, lambda_l2=5.0, num_threads=2, verbose=-1, max_bin=127)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/claude/sb/models/rt100_stream")
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--trees", type=int, default=600)
    ap.add_argument("--modules", default=",".join(FULL))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    mods = a.modules.split(",")
    os.makedirs(a.out, exist_ok=True)

    t0 = time.time()
    d = Data()
    mats, names = load_features(mods)
    keep = np.arange(len(names))
    rows = d.rows_for([0, 1, 2, 3, 4])
    rng = np.random.default_rng(a.seed)
    if len(rows) > a.rows:
        rows = np.sort(rng.choice(rows, a.rows, replace=False))
    X = _stack(mats, names, rows, keep)
    y = d.y[rows]
    print(f"train matrix {X.shape}  ({time.time()-t0:.0f}s)", flush=True)

    p = dict(CHAMP_PARAMS)
    ds = lgb.Dataset(X, label=y, params=p)
    booster = lgb.train(p, ds, num_boost_round=a.trees)
    del X, ds
    booster.save_model(os.path.join(a.out, "model.txt.0"))

    eng = StreamEngine(mods).fit_historical(np.arange(1200, dtype=np.float64) % 7 - 3.0)
    m = eng.manifest()
    if m["columns"] != names:
        bad = [i for i, (x, z) in enumerate(zip(m["columns"], names)) if x != z][:5]
        raise SystemExit(f"streaming columns != batch columns at {bad}")
    sha = subprocess.check_output(["git", "-C", "/home/claude/sb", "rev-parse", "HEAD"]).decode().strip()
    man = {
        "modules": list(eng.modules),
        "columns": m["columns"],
        "n_features": m["n_features"],
        "feature_manifest_sha256": m["feature_manifest_sha256"],
        "booster_columns": [list(range(len(names)))],
        "calibration": None,
        "code_git_sha": sha,
        "trained_on": {"folds": [0, 1, 2, 3, 4], "n_rows": int(len(rows)),
                       "n_series": int((d.series_fold >= 0).sum()), "seed": a.seed},
        "params": {k: v for k, v in p.items() if k != "num_threads"},
        "n_estimators": a.trees,
        "lockbox_touched": False,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print("model dir:", a.out)
    print("feature manifest sha:", m["feature_manifest_sha256"])
    print("model.txt.0 bytes:", os.path.getsize(os.path.join(a.out, "model.txt.0")))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
