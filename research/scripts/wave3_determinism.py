"""Is a wave-3 training run reproducible ON THIS MACHINE?

RT-300 reproduced RT-100 on 4 of 5 folds to the last printed digit and moved
+0.00476 on fold 3.  RT-301 reproduced RT-200 on 4 of 5 and moved -0.00124 on
fold 1.  Two readings are possible and they have opposite consequences:

  (a) the runs are deterministic here and the single-fold gaps are x86-vs-arm64
      floating point tipping a borderline split -- an environment difference,
      bounded and harmless; or
  (b) the runs are NOT deterministic here, in which case every paired delta this
      wave carries an unmeasured run-to-run component, and the competition's
      determinism REWARD-ELIGIBILITY condition (a 10% re-run must match to 1e-8)
      is at risk for the submission itself.

`sbr.pipeline.run` sets neither `deterministic` nor `force_row_wise`, so
LightGBM picks its histogram construction mode from a startup timing test.  That
choice is load-dependent, which would produce exactly this signature: mostly
identical, occasionally one fold apart.

This script re-runs ONE fold three times: twice with the shipped parameters, once
with `deterministic=True, force_row_wise=True`.  It writes no ledger row -- it is
an instrument check, not an experiment.
"""
from __future__ import annotations

import os, sys, json, time
import numpy as np

ROOT = os.environ.get("SBR_ROOT", "/home/claude/sb")
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

import lightgbm as lgb
from sbr.metric import ts_auc_flat
from sbr.pipeline import Data, load_features, _stack
from wave2_lib import CHAMP, FULL

FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def one_run(d, mats, names, keep_idx, extra_params=None, seed=0):
    p = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
             feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
             lambda_l2=5.0, num_threads=2, verbose=-1, max_bin=127)
    p.update(CHAMP["params"])
    if extra_params:
        p.update(extra_params)
    n_round = int(p.pop("n_estimators", 400))

    rng = np.random.default_rng(seed)
    # reproduce pipeline.run's draw order exactly: folds are consumed in order,
    # so to land on FOLD's sample we must burn the earlier folds' draws first.
    for f in (0, 1, 2, 3, 4):
        tr_folds = [x for x in (0, 1, 2, 3, 4) if x != f]
        tr_rows = d.rows_for(tr_folds)
        if len(tr_rows) > CHAMP["max_train_rows"]:
            sample = np.sort(rng.choice(tr_rows, CHAMP["max_train_rows"], replace=False))
        else:
            sample = tr_rows
        if f == FOLD:
            tr_rows = sample
            break

    va_rows = d.rows_for([FOLD])
    Xtr = _stack(mats, names, tr_rows, keep_idx)
    ytr = d.y[tr_rows]
    ds = lgb.Dataset(Xtr, label=ytr, params=p, feature_name=[f"f{i}" for i in range(len(keep_idx))])
    booster = lgb.train(p, ds, num_boost_round=n_round)
    del Xtr, ds
    Xva = _stack(mats, names, va_rows, keep_idx)
    pred = booster.predict(Xva).astype(np.float32)
    del Xva
    return float(ts_auc_flat(pred, d.y[va_rows], d.t[va_rows])), pred


def main():
    d = Data()
    mats, names = load_features(FULL)
    keep_idx = np.arange(len(names))
    out = {"fold": FOLD}

    t0 = time.time()
    a, pa = one_run(d, mats, names, keep_idx)
    print(f"run A (shipped params)                 fold {FOLD}: {a:.8f}   [{time.time()-t0:.0f}s]", flush=True)
    b, pb = one_run(d, mats, names, keep_idx)
    print(f"run B (shipped params, identical)      fold {FOLD}: {b:.8f}", flush=True)
    c, pc = one_run(d, mats, names, keep_idx,
                    extra_params={"deterministic": True, "force_row_wise": True})
    print(f"run C (deterministic+force_row_wise)   fold {FOLD}: {c:.8f}", flush=True)

    out["run_A"] = a
    out["run_B"] = b
    out["run_C_deterministic"] = c
    out["A_vs_B_score_delta"] = b - a
    out["A_vs_B_predictions_bitwise_identical"] = bool(np.array_equal(pa, pb))
    out["max_abs_pred_diff_A_B"] = float(np.max(np.abs(pa.astype(np.float64) - pb.astype(np.float64))))
    out["verdict"] = ("DETERMINISTIC on this machine: the ledger gap is an x86/arm64 difference"
                      if np.array_equal(pa, pb) else
                      "NOT DETERMINISTIC on this machine: run-to-run variation exists")
    path = f"{ROOT}/research/reports/wave3_determinism_fold{FOLD}.json"
    json.dump(out, open(path, "w"), indent=2)
    print("\n" + json.dumps(out, indent=2))
    print(f"\nwritten -> {path}")


if __name__ == "__main__":
    main()
