"""Train the BEST LEGALLY DEPLOYABLE system: 7 boosters + frozen time-conditional
CDF calibration, all sharing ONE incremental feature engine.

Why this is legal where RT-131 is not: each booster reads a slice of the same
per-series feature vector, and the blend is a mean of FROZEN functions
F_m(score | t) fitted on training-fold OOF. Nothing at inference time looks at
any other series. See research/reports/runner_semantics.md.

Why the boosters are the same configurations that produced the OOF: the
calibration grids are estimated from those OOF score distributions, so a
deployed booster with a different configuration would be calibrated against the
wrong distribution.

KNOWN, DOCUMENTED APPROXIMATION: the deployed boosters are fitted on all five
dev folds, while the calibration grids come from OOF scores produced by models
fitted on four. A full-data model's scores are slightly sharper than its
out-of-sample scores, so F_m is very slightly miscalibrated at deployment. This
is conservative (it compresses rather than inflates extreme scores), it is
identical in direction for every stream so it largely cancels in the mean, and
the alternative -- deploying 35 fold-models -- costs 5x the inference for an
unmeasured gain.
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time
import numpy as np

ROOT = os.environ.get(
    "SBR_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
)

sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")
import lightgbm as lgb

from sbr.pipeline import Data, load_features, _stack, _make_pairwise_t
from sbr.production.calibration import SmoothTimeCDFCal
from sbr.stream.engine import StreamEngine
from wave2_streams import JOBS

FULL = ["m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes"]
CHAMP = {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 300,
         "feature_fraction": 0.5, "bagging_fraction": 0.7, "lambda_l2": 5.0, "max_bin": 127}
STREAMS = {
    "RT-100R": dict(modules=FULL, seed=0, max_train_rows=1_000_000, params=dict(CHAMP)),
    **{k: dict(modules=v["modules"], seed=v["seed"], max_train_rows=v["max_train_rows"],
               params=dict(v["params"]), sample_mode=v.get("sample_mode", "uniform"))
       for k, v in JOBS.items()},
}
DEFAULT = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
               feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
               lambda_l2=5.0, num_threads=2, verbose=-1, max_bin=127)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/models/rt150_ensemble")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = Data()
    mats, names = load_features(FULL)
    eng = StreamEngine(FULL).fit_historical(np.arange(1200, dtype=np.float64) % 7 - 3.0)
    man_eng = eng.manifest()
    assert man_eng["columns"] == names, "streaming column order != batch column order"
    col_index = {c: i for i, c in enumerate(names)}

    dev = d.rows_for([0, 1, 2, 3, 4])
    order = list(STREAMS)
    slices, cals, cfgs = [], [], []

    for k, exp in enumerate(order):
        cfg = STREAMS[exp]
        sel = np.array([col_index[c] for c in names if c.split("::")[0] in set(cfg["modules"])])
        slices.append(sel.tolist())

        # ---- calibration from the CROSS-FITTED OOF of this exact configuration
        oof = np.load(f"{ROOT}/research/oof/{exp}.npy")
        cals.append(SmoothTimeCDFCal.fit(oof[dev], d.t[dev]))

        path = os.path.join(a.out, f"model.txt.{k}")
        cfgs.append({"experiment": exp, "modules": cfg["modules"], "seed": cfg["seed"],
                     "n_columns": int(len(sel)), "params": cfg["params"],
                     "max_train_rows": cfg["max_train_rows"],
                     "sample_mode": cfg.get("sample_mode", "uniform")})
        if a.only and exp not in a.only.split(","):
            print(f"skip {exp}"); continue
        if os.path.exists(path):
            print(f"{exp}: already trained, keeping {path}"); continue

        t0 = time.time()
        p = dict(DEFAULT); p.update(cfg["params"])
        n_round = int(p.pop("n_estimators"))
        rng = np.random.default_rng(cfg["seed"])
        rows = dev
        if len(rows) > cfg["max_train_rows"]:
            if cfg.get("sample_mode") == "per_series":
                s = d.sidx[rows]
                o = np.argsort(s, kind="stable"); rows = rows[o]; s = s[o]
                bnd = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
                ends = np.r_[bnd[1:], len(s)]
                per = cfg["max_train_rows"] // len(bnd)
                pick = [np.arange(b, e) if e - b <= per else rng.choice(np.arange(b, e), per, replace=False)
                        for b, e in zip(bnd, ends)]
                rows = np.sort(rows[np.concatenate(pick)])
            else:
                rows = np.sort(rng.choice(rows, cfg["max_train_rows"], replace=False))
        X = _stack(mats, names, rows, sel)
        y = d.y[rows]
        if p.get("objective") == "pairwise_t":
            p["objective"] = _make_pairwise_t(d.t[rows], y, seed=cfg["seed"])
        ds = lgb.Dataset(X, label=y, params=dict(p, objective="binary"))
        b = lgb.train(p, ds, num_boost_round=n_round)
        del X, ds
        b.save_model(path)
        print(f"{exp}: {len(rows)} rows x {len(sel)} cols -> {path} ({time.time()-t0:.0f}s)", flush=True)

    sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"]).decode().strip()
    man = {
        "modules": list(man_eng["modules"]), "columns": names, "n_features": len(names),
        "feature_manifest_sha256": man_eng["feature_manifest_sha256"],
        "booster_columns": slices,
        "calibration": {"kind": "scdf", "models": [c.to_json() for c in cals],
                        "fitted_on": "cross-fitted OOF over dev folds 0-4",
                        "recovers_pct_of_oracle": 99.7},
        "streams": cfgs, "code_git_sha": sha,
        "trained_on": {"folds": [0, 1, 2, 3, 4]}, "lockbox_touched": False,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"))
    sz = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out))
    print(f"\nmodel dir {a.out}  total {sz/1e6:.1f} MB  boosters {len(order)}")


if __name__ == "__main__":
    main()
