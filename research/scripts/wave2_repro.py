"""A1 -- independent reproduction of RT-000 / RT-101 / RT-100 from a CLEAN
checkout of research-checkpoint-20260818-1, in a fresh container, from raw data.

Nothing here reuses a fitted model object: the store, the folds, the feature
cache and every booster are rebuilt from `data/X_train.parquet`.
"""
import sys, os, json, time
CLEAN = "/home/claude/repro/src"
sys.path.insert(0, CLEAN)          # CODE from the clean checkout
import numpy as np
from sbr.pipeline import run

assert os.path.dirname(os.path.dirname(sys.modules["sbr"].__file__)) == CLEAN, sys.modules["sbr"].__file__

FULL = ["m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes"]
P = dict(agent="A1-repro", folds=(0, 1, 2, 3, 4), max_train_rows=1_000_000,
         params={"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63,
                 "min_data_in_leaf": 300, "feature_fraction": 0.5,
                 "bagging_fraction": 0.7, "lambda_l2": 5.0, "max_bin": 127})

which = sys.argv[1] if len(sys.argv) > 1 else "all"
if which in ("all", "100"):
    run(exp_id="RT-100R", modules=FULL,
        hypothesis="RT-100 (0.61510 mean OOF) reproduces from a clean checkout of "
                   "research-checkpoint-20260818-1 in a fresh container, rebuilt from raw parquet",
        falsification="mean OOF TS-AUC differs from 0.61510 by more than 0.002, or any "
                      "per-fold score differs by more than 0.003, or n_features != 500",
        notes="A1 independent reproduction; code from clean checkout e98f5b9; "
              "store/folds/features rebuilt from raw data in a fresh container", **P)
if which in ("all", "101"):
    run(exp_id="RT-101R", modules=["m00_core"],
        hypothesis="RT-101 (0.563491) reproduces from a clean checkout",
        falsification="mean OOF TS-AUC differs from 0.563491 by more than 0.002",
        notes="A1 independent reproduction", **P)
