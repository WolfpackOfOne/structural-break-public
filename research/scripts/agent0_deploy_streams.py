"""RT-18x -- diverse streams over the DEPLOYABLE six-module set (no m02_dist).

All streams share one feature computation at inference, so the cost is the union
of modules, not the stream count.  Restricting every stream to the six-module set
buys the ensemble for free once the features are paid for.
"""
import sys; sys.path.insert(0, "/home/claude/sb/src")
from sbr.pipeline import run

SIX = ["m00_core", "m01_seq", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes"]
BASE = dict(agent="agent0", folds=(0, 1, 2, 3, 4), max_train_rows=700_000)

run(exp_id="RT-180", modules=SIX,
    hypothesis="Extra-trees variant over the deployable module set is a distinct stream",
    falsification="within-timestep rank correlation with RT-172 > 0.97",
    params={"n_estimators": 500, "learning_rate": 0.06, "num_leaves": 255, "min_data_in_leaf": 400,
            "feature_fraction": 0.3, "bagging_fraction": 0.6, "lambda_l2": 20.0, "max_bin": 63,
            "extra_trees": True},
    notes="deployable stream B: extra-trees, per-series sampling", seed=7, sample_mode="per_series", **BASE)

run(exp_id="RT-181", modules=SIX,
    hypothesis="Pairwise-t ranking over the deployable module set adds ensemble value",
    falsification="blend does not improve on RT-172",
    params={"n_estimators": 600, "objective": "pairwise_t", "learning_rate": 0.05, "num_leaves": 63,
            "min_data_in_leaf": 300, "feature_fraction": 0.5, "bagging_fraction": 0.7,
            "lambda_l2": 5.0, "max_bin": 127},
    notes="deployable stream C: pairwise logistic, groups = online index t", seed=0, **BASE)

run(exp_id="RT-182", modules=["m00_core", "m01_seq", "m07_bayes", "m06_loc"],
    hypothesis="A recursion-heavy subset of the deployable set is a decorrelated stream",
    falsification="within-timestep rank correlation with RT-172 > 0.95",
    params={"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
            "feature_fraction": 0.6, "bagging_fraction": 0.8, "lambda_l2": 3.0, "max_bin": 255},
    notes="deployable stream D: recursion-heavy, 321 cols", seed=3, **BASE)
