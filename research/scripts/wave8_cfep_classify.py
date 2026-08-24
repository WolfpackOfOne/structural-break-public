"""CFEP downstream classifier -- 500 causal cols + 16-dim frozen embedding,
LightGBM, fold-0 pilot. Run as a SEPARATE PROCESS from wave8_cfep.py's torch
training/extraction (PROTOCOL.md / commit 1068b96: torch and LightGBM
segfault sharing a process). The embedding itself was frozen and saved to
disk by wave8_cfep.py --extract before this script ever runs.

Usage:
    python wave8_cfep_classify.py --head bce      # RT-1041
    python wave8_cfep_classify.py --head future   # RT-1042
"""
from __future__ import annotations

import argparse, time

import numpy as np

import wave8_common as W8
import sbr.pipeline as PL
from wave5_lib import FOLDS, OOFDIR, ts_auc_flat
from wave7_d3r import FULL, ARM_B_PARAMS

OUTER_F = 0
MAX_TRAIN_ROWS = 1_000_000
CACHE = f"{OOFDIR}/wave8_cfep"


def run(head):
    exp_id = "RT-1041" if head == "bce" else "RT-1042"
    d = PL.Data()
    mats, names = PL.load_features(FULL)
    keep_idx = np.arange(len(names))
    W8.assert_no_forbidden_columns(names)
    Z = np.load(f"{CACHE}/embedding_{head}.npy")
    assert Z.shape == (len(d.y), 16)

    tr_rows = d.rows_for([g for g in FOLDS if g != OUTER_F])
    va_rows = d.rows_for([OUTER_F])
    rng = np.random.default_rng(0)
    if len(tr_rows) > MAX_TRAIN_ROWS:
        tr_rows = np.sort(rng.choice(tr_rows, MAX_TRAIN_ROWS, replace=False))

    import lightgbm as lgb
    p = dict(ARM_B_PARAMS)
    n_round = p.pop("n_estimators")
    t0 = time.time()
    Xtr = np.concatenate([PL._stack(mats, names, tr_rows, keep_idx), Z[tr_rows]], axis=1)
    ytr = d.y[tr_rows]
    ds = lgb.Dataset(Xtr, label=ytr, params=dict(p, objective="binary"),
                     feature_name=[f"f{i}" for i in range(Xtr.shape[1])])
    booster = lgb.train(dict(p), ds, num_boost_round=n_round)
    del Xtr, ds
    Xva = np.concatenate([PL._stack(mats, names, va_rows, keep_idx), Z[va_rows]], axis=1)
    pred = booster.predict(Xva).astype(np.float32)
    del Xva
    s = float(ts_auc_flat(pred, d.y[va_rows], d.t[va_rows]))
    oof = np.full(len(d.y), np.nan, dtype=np.float32)
    oof[va_rows] = pred
    np.save(f"{OOFDIR}/{exp_id}.npy", oof)
    print(f"[CFEP-{'B' if head=='bce' else 'C'}] TS-AUC (fold0) {s:.5f}  "
          f"runtime {time.time()-t0:.1f}s -> {exp_id}")

    label = "CFEP-B matched-capacity BCE-embedding control" if head == "bce" else \
        "CFEP-C future-predictive embedding (primary candidate)"
    res = dict(experiment_id=exp_id, date=time.strftime("%Y-%m-%d %H:%M"), git_sha=PL.git_sha(),
               agent="agent0",
               hypothesis=f"{label}: 500 causal cols + frozen 16-dim causal-TCN embedding "
                          f"(hidden 32, kernel 3, dilations 1/2/4/8/16/32, matched RT-970 shell).",
               falsification_condition="see research/WAVE8_FUTURE_AWARE_PREREG.md section 5.4",
               feature_set=",".join(FULL) + f"+cfep_{head}_embedding16", n_features=516,
               model="lgbm", objective="binary", folds=str(OUTER_F), random_seed=0,
               train_series=int((~np.isin(d.series_fold, [-1])).sum()), train_rows=len(tr_rows),
               mean_oof_ts_auc=s, pooled_oof_ts_auc=s, per_fold_ts_auc=f"{s:.5f}", fold_std=0.0,
               persistence="none", sample_mode="uniform", training_runtime_s=round(time.time() - t0, 1),
               causal_verified="embedding frozen before this LightGBM step; outer-fold-0 rows never "
                               "seen by the embedding's own training (see wave8_cfep.py train() "
                               "outer_train restriction)",
               test_reduced_touched="no", lockbox_touched="no", status="recorded",
               notes="Wave 8 CFEP pilot. Pre-registered research/WAVE8_FUTURE_AWARE_PREREG.md.",
               protocol="pilot_fold0_only")
    PL.append_result(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=["bce", "future"], required=True)
    run(ap.parse_args().head)


if __name__ == "__main__":
    main()
