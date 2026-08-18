"""AGENT 12 -- prediction-correlation + ensemble-delta analysis for m07_bayes.

Reproduces the screening protocol's fold-0 training exactly (same store, same
folds, same seed, same row subsample, same LightGBM params) for three feature
sets, keeps the fold-0 validation predictions, and reports

    * TS-AUC of each set
    * Pearson / Spearman correlation between the m07-alone and m00-alone
      prediction streams (globally and within-timestep, which is the space the
      metric actually lives in)
    * ensemble delta of an UNWEIGHTED rank average (no fitting) and of a
      weight sweep (reported as a curve; the weight is not selected here)

`sbr.pipeline.run` does not persist OOF under screen=True, which is why this
script exists; the training code path is copied from it verbatim so the numbers
are directly comparable to the ledger.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import lightgbm as lgb

from sbr.metric import ts_auc_flat
from sbr.pipeline import Data, load_features, _stack

OUT = "/home/claude/sb/research/artifacts"
SEED = 0
FOLD = 0
MAX_TRAIN = 300_000
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
              lambda_l2=5.0, num_threads=2, verbose=-1, max_bin=127)
N_ROUND = 300


def fit_predict(d, modules):
    mats, names = load_features(modules, screen=True)
    keep_idx = np.arange(sum(m.shape[1] for m in mats))
    rng = np.random.default_rng(SEED)
    tr_folds = [x for x in (0, 1, 2, 3, 4) if x != FOLD]
    tr_rows = d.rows_for(tr_folds)
    va_rows = d.rows_for([FOLD])
    if len(tr_rows) > MAX_TRAIN:
        tr_rows = np.sort(rng.choice(tr_rows, MAX_TRAIN, replace=False))
    Xtr = _stack(mats, names, tr_rows, keep_idx)
    ds = lgb.Dataset(Xtr, label=d.y[tr_rows],
                     feature_name=[f"f{i}" for i in range(len(keep_idx))])
    booster = lgb.train(PARAMS, ds, num_boost_round=N_ROUND)
    del Xtr, ds
    Xva = _stack(mats, names, va_rows, keep_idx)
    pred = booster.predict(Xva).astype(np.float64)
    del Xva
    return va_rows, pred


def rank01(v):
    o = np.argsort(v, kind="stable")
    r = np.empty(len(v))
    r[o] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def rank_within_t(v, t):
    """Rank-normalise inside each online index -- the space TS-AUC compares in."""
    out = np.empty(len(v))
    order = np.lexsort((v, t))
    ts = t[order]
    b = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]])
    e = np.r_[b[1:], len(ts)]
    pos = np.arange(len(ts)) - np.repeat(b, e - b)
    ln = np.repeat(e - b, e - b)
    out[order] = pos / np.maximum(ln - 1, 1)
    return out


def spearman(a, b):
    return float(np.corrcoef(rank01(a), rank01(b))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="agent12")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    d = Data(screen=True)

    sets = {
        "m00": ["m00_core"],
        "m07": ["m07_bayes"],
    }
    preds, res = {}, {}
    va_rows = None
    for k, mods in sets.items():
        va_rows, p = fit_predict(d, mods)
        preds[k] = p
        s = ts_auc_flat(p, d.y[va_rows], d.t[va_rows])
        res[k] = float(s)
        print(f"{k:10s} TS-AUC(fold0,screen) = {s:.5f}", flush=True)

    y = d.y[va_rows]
    t = d.t[va_rows].astype(np.int64)
    np.save(f"{OUT}/{a.tag}_screen_fold0_preds.npy",
            np.column_stack([preds["m00"], preds["m07"]]).astype(np.float32))

    p0, p7 = preds["m00"], preds["m07"]
    out = {"ts_auc": res}
    out["pearson_raw"] = float(np.corrcoef(p0, p7)[0, 1])
    out["pearson_logit"] = float(np.corrcoef(np.log(p0 / (1 - p0)), np.log(p7 / (1 - p7)))[0, 1])
    out["spearman_global"] = spearman(p0, p7)
    r0t, r7t = rank_within_t(p0, t), rank_within_t(p7, t)
    out["pearson_within_t_rank"] = float(np.corrcoef(r0t, r7t)[0, 1])

    # ---- ensembles
    out["ens_rank_avg_global"] = float(ts_auc_flat(0.5 * rank01(p0) + 0.5 * rank01(p7), y, t))
    out["ens_rank_avg_within_t"] = float(ts_auc_flat(0.5 * r0t + 0.5 * r7t, y, t))
    lo0 = np.log(p0 / (1 - p0)); lo7 = np.log(p7 / (1 - p7))
    out["ens_logit_avg"] = float(ts_auc_flat(0.5 * lo0 + 0.5 * lo7, y, t))
    sweep = {}
    for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        sweep[f"{w:.1f}"] = float(ts_auc_flat((1 - w) * rank01(p0) + w * rank01(p7), y, t))
    out["rank_blend_sweep_w_m07"] = sweep
    sweep_t = {}
    for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        sweep_t[f"{w:.1f}"] = float(ts_auc_flat((1 - w) * r0t + w * r7t, y, t))
    out["rank_blend_sweep_within_t"] = sweep_t

    best = max(res["m00"], res["m07"])
    out["delta_stacked_model_vs_m00"] = 0.61352161697193 - res["m00"]  # RT-120B - RT-120C
    out["delta_rank_avg_vs_best_alone"] = out["ens_rank_avg_global"] - best
    out["delta_rank_avg_within_t_vs_best_alone"] = out["ens_rank_avg_within_t"] - best

    # ---- per-elapsed-time breakdown of the standalone module
    json.dump(out, open(f"{OUT}/{a.tag}_blend.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
