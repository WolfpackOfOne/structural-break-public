"""Wave 5 -- build DIVERSE full-scale OOF streams for the portfolio.

The champion is one point in (feature subset x algorithm bias) space.  These
three deliberately differ along both axes, so the blend has something to work
with: a model that is 0.99 correlated with the champion cannot help it however
good it is.
"""
import sys; sys.path.insert(0, "/home/claude/sb/src")
from sbr.pipeline import run

FULL = ["m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes"]
BASE = dict(agent="agent0", folds=(0, 1, 2, 3, 4), max_train_rows=900_000)

run(exp_id="RT-120", modules=["m00_core", "m01_seq", "m07_bayes"],
    hypothesis="A sequential/Bayesian-evidence-only model is a distinct alpha stream from the full bank",
    falsification="within-timestep rank correlation with RT-100 > 0.95 (i.e. a clone)",
    params={"n_estimators": 700, "learning_rate": 0.04, "num_leaves": 127, "min_data_in_leaf": 150,
            "feature_fraction": 0.35, "bagging_fraction": 0.7, "lambda_l2": 5.0, "max_bin": 127},
    notes="stream A: evidence-recursion heavy, deep trees", seed=0, **BASE)

run(exp_id="RT-121", modules=["m02_dist", "m03_dyn", "m04_resid", "m06_loc"],
    hypothesis="A distribution/dynamics/representation model is a distinct alpha stream",
    falsification="within-timestep rank correlation with RT-100 > 0.95",
    params={"n_estimators": 500, "learning_rate": 0.08, "num_leaves": 31, "min_data_in_leaf": 500,
            "feature_fraction": 0.7, "bagging_fraction": 0.7, "lambda_l2": 10.0, "max_bin": 127},
    notes="stream B: shape/dynamics heavy, shallow trees", seed=1, **BASE)

run(exp_id="RT-122", modules=FULL,
    hypothesis="An extremely-randomised, per-series-sampled variant decorrelates from the champion",
    falsification="within-timestep rank correlation with RT-100 > 0.97",
    params={"n_estimators": 500, "learning_rate": 0.06, "num_leaves": 255, "min_data_in_leaf": 400,
            "feature_fraction": 0.3, "bagging_fraction": 0.6, "lambda_l2": 20.0, "max_bin": 63,
            "extra_trees": True},
    notes="stream C: extra-trees, per-series row sampling, different seed",
    seed=7, sample_mode="per_series", **BASE)
