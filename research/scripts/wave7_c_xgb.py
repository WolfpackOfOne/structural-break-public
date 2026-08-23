"""W7-C1 -- LANE C SCREEN: XGBoost on the identical causal state.

SAME 500 features, SAME rows, SAME folds, SAME labels, SAME scorer.  The ONLY
change is LightGBM -> XGBoost.  A tiny pre-declared configuration, no Optuna, no
grid (brief section 20).

Reports standalone TS-AUC, per-fold scores, age buckets, within-timestep rank
correlation against the LightGBM control, and the blend delta -- against the
SAME-STRENGTH LIGHTGBM SEED CLONE, which is the bar wave 6 established when the
TCN's 0.21 rank correlation turned out to be worth +0.0001.

Success: >= +0.003 beyond the matched seed-clone control, or a standalone score
that materially beats LightGBM.  A slower LightGBM is killed.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wave7_lib as W  # noqa: E402

sys.path.insert(0, f"{W.ROOT}/src")

#: Declared once.  Depth/eta/subsample chosen to MATCH the LightGBM control's
#: capacity (num_leaves 63 ~ depth 6, same lr, same column and row subsampling,
#: same L2, same round count), not to win a tuning contest.
XGB_PARAMS = dict(objective="binary:logistic", eta=0.05, max_depth=6,
                  min_child_weight=200, subsample=0.7, colsample_bytree=0.7,
                  reg_lambda=5.0, tree_method="hist", max_bin=127,
                  nthread=2, eval_metric="logloss")
N_ROUND = 400


def within_t_rank_corr(a, b, t) -> float:
    """Spearman correlation computed INSIDE each timestep and pair-weighted.

    A global rank correlation is the wrong statistic here: the metric only ever
    compares series at the same t, so agreement across timesteps is not agreement
    the metric can see.
    """
    from scipy.stats import rankdata
    order = np.argsort(t, kind="stable")
    a, b, t = a[order], b[order], t[order]
    gs = np.flatnonzero(np.r_[True, t[1:] != t[:-1]])
    ge = np.r_[gs[1:], len(t)]
    num = den = 0.0
    for s, e in zip(gs, ge):
        if e - s < 3:
            continue
        ra, rb = rankdata(a[s:e]), rankdata(b[s:e])
        if ra.std() == 0 or rb.std() == 0:
            continue
        c = float(np.corrcoef(ra, rb)[0, 1])
        w = float(e - s)
        num += w * c
        den += w
    return num / den if den else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--max-train-rows", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clone-seed", type=int, default=8,
                    help="the same-strength LightGBM seed clone -- the real control")
    a = ap.parse_args()

    W.require_store()
    import lightgbm as lgb
    import xgboost as xgb
    from sbr.metric import ts_auc_flat
    from sbr.pipeline import Data, _stack, load_features

    folds = [int(x) for x in a.folds.split(",")]
    d = Data()
    mats, names = load_features(W.PROD_MODULES)
    W.assert_no_teacher_in_features(names)
    keep_idx = np.arange(len(names))

    oof = {k: np.full(len(d.y), np.nan, np.float32) for k in ("xgb", "lgb", "clone")}
    per_fold = {k: [] for k in oof}
    rng = np.random.default_rng(a.seed)

    for f in folds:
        tr = d.rows_for([x for x in (0, 1, 2, 3, 4) if x != f])
        va = d.rows_for([f])
        if len(tr) > a.max_train_rows:
            tr = np.sort(rng.choice(tr, a.max_train_rows, replace=False))
        Xtr, ytr = _stack(mats, names, tr, keep_idx), d.y[tr]
        Xva = _stack(mats, names, va, keep_idx)

        dtr = xgb.DMatrix(Xtr, label=ytr, missing=np.nan, nthread=XGB_PARAMS["nthread"])
        bx = xgb.train(dict(XGB_PARAMS, seed=a.seed), dtr, num_boost_round=N_ROUND)
        oof["xgb"][va] = bx.predict(xgb.DMatrix(Xva, missing=np.nan)).astype(np.float32)
        del dtr

        for key, sd in (("lgb", a.seed), ("clone", a.clone_seed)):
            p = dict(W.CHAMP_PARAMS)
            n_round = int(p.pop("n_estimators"))
            p.update(seed=sd, bagging_seed=sd, feature_fraction_seed=sd)
            ds = lgb.Dataset(Xtr, label=ytr, params=p,
                             feature_name=[f"f{i}" for i in range(len(names))])
            oof[key][va] = lgb.train(p, ds, num_boost_round=n_round).predict(Xva).astype(np.float32)
        del Xtr, Xva

        for k in oof:
            s = float(ts_auc_flat(oof[k][va], d.y[va], d.t[va]))
            per_fold[k].append(s)
        print(f"fold {f}: " + "  ".join(f"{k} {per_fold[k][-1]:.5f}" for k in oof), flush=True)

    dev = d.rows_for(folds)
    y, t = d.y[dev], d.t[dev]

    def sc(v):
        return float(ts_auc_flat(v[dev], y, t))

    # Equal-weight rank blends -- no weight is searched (brief section 27).
    from scipy.stats import rankdata
    def blend(*keys):
        r = sum(rankdata(oof[k][dev]) for k in keys) / len(keys)
        return float(ts_auc_flat(r.astype(np.float32), y, t))

    out = {"schema": "wave7_c_xgb/1", "folds": folds, "xgb_params": XGB_PARAMS,
           "n_round": N_ROUND, "seed": a.seed, "clone_seed": a.clone_seed,
           "standalone": {k: sc(oof[k]) for k in oof},
           "per_fold": {k: per_fold[k] for k in oof},
           "mean_per_fold": {k: float(np.mean(per_fold[k])) for k in oof},
           "within_t_rank_corr_xgb_vs_lgb": within_t_rank_corr(oof["xgb"][dev], oof["lgb"][dev], t),
           "within_t_rank_corr_clone_vs_lgb": within_t_rank_corr(oof["clone"][dev], oof["lgb"][dev], t),
           "blend_lgb_xgb": blend("lgb", "xgb"),
           "blend_lgb_clone": blend("lgb", "clone")}
    out["marginal_over_seed_clone"] = out["blend_lgb_xgb"] - out["blend_lgb_clone"]
    out["verdict"] = ("CONTINUE" if out["marginal_over_seed_clone"] >= 0.003
                      or out["standalone"]["xgb"] - out["standalone"]["lgb"] >= 0.003
                      else "KILL")
    for k, v in out.items():
        if k not in ("xgb_params", "per_fold"):
            print(f"{k}: {v}")
    print("written:", W.write_report("wave7_c_xgb", out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
