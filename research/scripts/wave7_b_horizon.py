"""W7-B1 -- LANE B PILOT: a horizon specialist routed by CURRENT t.

ONE CANONICAL FOLD.  Global champion-configuration LightGBM against a specialist
trained only on rows inside one pre-registered t regime, evaluated ON THAT REGION.

PILOT REGION: H4, t in [101, 251).  The wave-7 brief proposed piloting the
early/mid regime "where current TS-AUC is weakest".  W7-D1 measured where the
LOSS is instead, and the early regime is not it: t < 50 owns 4.7% of remaining
weighted loss with a perfect-repair ceiling of +0.018, while t >= 100 owns 86%.
H4 is the earliest regime with material mass (20.3% of loss) and enough rows for
a stable fit; H5 (t 251-501) and H6 are the follow-ups if H4 clears.  The choice
is recorded in research/WAVE7_ALPHA_BUDGET.md and was made before any score.

BAR (pre-registered): >= +0.005 conditional TS-AUC inside the pilot region.
Below that the full six-regime bank is not built.

LEGALITY.  Routing reads `d.t`, the current online index: observed, identical for
every series being compared at that step, and independent of tau.  No estimated
break age, no inferred tau, no future horizon.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wave7_lib as W  # noqa: E402

sys.path.insert(0, f"{W.ROOT}/src")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--regime", default="H4", choices=W.HORIZON_NAMES)
    ap.add_argument("--max-train-rows", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--screen", action="store_true",
                    help="reduced protocol / synthetic smoke store -- NEVER a research number")
    a = ap.parse_args()

    W.require_store(a.screen)
    import lightgbm as lgb
    from sbr.metric import ts_auc_flat
    from sbr.pipeline import Data, _stack, load_features

    ri = W.HORIZON_NAMES.index(a.regime)
    lo, hi = W.HORIZON_EDGES[ri], W.HORIZON_EDGES[ri + 1]

    d = Data(screen=a.screen)
    mats, names = load_features(W.PROD_MODULES, screen=a.screen)
    W.assert_no_teacher_in_features(names)
    keep_idx = np.arange(len(names))

    tr_folds = [x for x in (0, 1, 2, 3, 4) if x != a.fold]
    tr_all = d.rows_for(tr_folds)
    va_rows = d.rows_for([a.fold])
    rng = np.random.default_rng(a.seed)

    def sample(rows, n):
        return np.sort(rng.choice(rows, n, replace=False)) if len(rows) > n else rows

    tr_glob = sample(tr_all, a.max_train_rows)
    tr_spec_pool = tr_all[(d.t[tr_all] >= lo) & (d.t[tr_all] < hi)]
    # The specialist gets the SAME ROW BUDGET, so any gain is specialisation and
    # not extra data.  If the regime has fewer rows than the budget it takes all
    # of them and the row count is reported.
    tr_spec = sample(tr_spec_pool, a.max_train_rows)

    va_reg = va_rows[(d.t[va_rows] >= lo) & (d.t[va_rows] < hi)]
    print(f"regime {a.regime} t in [{lo},{hi})  train pool {len(tr_spec_pool)} "
          f"used {len(tr_spec)}  valid-in-region {len(va_reg)}")

    Xva_reg = _stack(mats, names, va_reg, keep_idx)
    y_reg, t_reg = d.y[va_reg], d.t[va_reg]

    out = {"schema": "wave7_b_horizon/1", "protocol": "screen" if a.screen else "full", "fold": a.fold, "regime": a.regime,
           "t_lo": lo, "t_hi": hi, "n_train_specialist": int(len(tr_spec)),
           "n_train_global": int(len(tr_glob)), "n_valid_region": int(len(va_reg)),
           "params": W.CHAMP_PARAMS, "arms": {}}

    for arm, rows in (("global", tr_glob), ("specialist", tr_spec)):
        p = dict(W.CHAMP_PARAMS)
        n_round = int(p.pop("n_estimators"))
        X = _stack(mats, names, rows, keep_idx)
        ds = lgb.Dataset(X, label=d.y[rows], params=p,
                         feature_name=[f"f{i}" for i in range(len(names))])
        b = lgb.train(p, ds, num_boost_round=n_round)
        del X
        pr = b.predict(Xva_reg).astype(np.float32)
        s = float(ts_auc_flat(pr, y_reg, t_reg))
        out["arms"][arm] = {"conditional_ts_auc": s}
        print(f"  {arm:11s} conditional TS-AUC in {a.regime}: {s:.5f}", flush=True)

    dl = out["arms"]["specialist"]["conditional_ts_auc"] - out["arms"]["global"]["conditional_ts_auc"]
    out["delta_conditional"] = dl
    out["bar"] = 0.005
    out["verdict"] = "BUILD THE BANK" if dl >= 0.005 else "KILL"
    print(f"  delta {dl:+.5f}  bar +0.00500  -> {out['verdict']}")
    print("written:", W.write_report(f"wave7_b_horizon_{a.regime}_fold{a.fold}{'_screen' if a.screen else ''}", out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
