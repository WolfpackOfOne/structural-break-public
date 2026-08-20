"""Wave-2 experiment helpers.  Never edits sbr core files."""
from __future__ import annotations

import contextlib, json, os, sys
import numpy as np

ROOT = os.environ.get("SBR_ROOT", "/home/claude/sb")
sys.path.insert(0, f"{ROOT}/src")
import sbr.pipeline as PL

FULL = ["m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes"]

#: Reduced-cost protocol used for ABLATIONS and STABILITY studies.  Every arm of
#: a comparison uses the SAME settings, so paired deltas are valid; absolute
#: levels sit slightly below the 1M-row champion protocol and must never be
#: quoted as a champion score.
ABL = dict(folds=(0, 1, 2, 3, 4), max_train_rows=400_000,
           params={"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63,
                   "min_data_in_leaf": 300, "feature_fraction": 0.5,
                   "bagging_fraction": 0.7, "lambda_l2": 5.0, "max_bin": 127})

#: The champion protocol (RT-100).
CHAMP = dict(folds=(0, 1, 2, 3, 4), max_train_rows=1_000_000,
             params={"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63,
                     "min_data_in_leaf": 300, "feature_fraction": 0.5,
                     "bagging_fraction": 0.7, "lambda_l2": 5.0, "max_bin": 127})


@contextlib.contextmanager
def alt_folds(name: str):
    """Temporarily point the pipeline at an ALTERNATIVE fold partition.

    Robustness testing only -- see research/VALIDATION_V2.md.  The alt files hold
    only the 8,000 dev series, so we materialise a full-length folds table with
    the lockbox series still marked -1 and untouched.
    """
    base = PL.pd.read_parquet(f"{ROOT}/research/folds/folds.parquet")
    alt = PL.pd.read_parquet(f"{ROOT}/research/folds/folds_{name}.parquet")
    mp = dict(zip(alt.id.tolist(), alt.fold.tolist()))
    new = base.copy()
    new["fold"] = [mp.get(i, -1) for i in base.id.tolist()]
    assert (new.fold.to_numpy() >= 0).sum() == len(alt)
    tmp = f"{ROOT}/cache/_folds_{name}.parquet"
    new.to_parquet(tmp, index=False)
    old = PL.FOLDS
    PL.FOLDS = tmp
    try:
        yield
    finally:
        PL.FOLDS = old


def cols_of(modules):
    names = []
    for m in modules:
        meta = json.load(open(os.path.join(os.environ.get("SBR_FEATURES", f"{ROOT}/cache/features"), f"{m}.cols.json")))
        names += [f"{m}::{c}" for c in meta["cols"]]
    return names
