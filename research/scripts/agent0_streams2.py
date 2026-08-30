"""Wave 6 -- three more alpha streams for the portfolio.

The ensemble is the only thing that has transferred cleanly from development to
the lockbox (+0.0087 OOF, +0.0062 lockbox), so the highest-value alpha left is
more streams that are WRONG IN DIFFERENT WAYS -- not better streams.
"""
import sys; sys.path.insert(0, "/home/claude/sb/src")
from sbr.pipeline import run

FULL = ["m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes"]
BASE = dict(agent="agent0", folds=(0, 1, 2, 3, 4), max_train_rows=700_000)

# D: metric-aligned objective. It loses head-to-head against binary logloss
# (-0.0033, RT-111), but it is wrong in a different direction, which is the only
# thing that matters for a blend member.
run(exp_id="RT-123", modules=FULL,
    hypothesis="The pairwise-t ranking model adds ensemble value despite losing head-to-head",
    falsification="blend of RT-130 + RT-123 does not beat RT-130",
    params={"n_estimators": 600, "objective": "pairwise_t", "learning_rate": 0.05,
            "num_leaves": 63, "min_data_in_leaf": 300, "feature_fraction": 0.5,
            "bagging_fraction": 0.7, "lambda_l2": 5.0, "max_bin": 127},
    notes="stream D: pairwise logistic, groups = online index t", seed=0, **BASE)

# E: generative/sequential evidence only -- the family with the lowest
# within-timestep correlation to everything else.
run(exp_id="RT-124", modules=["m07_bayes", "m06_loc", "m01_seq"],
    hypothesis="A recursion-only model (Bayesian + localisation + sequential) is a distinct stream",
    falsification="within-timestep rank correlation with RT-100 > 0.95",
    params={"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
            "feature_fraction": 0.6, "bagging_fraction": 0.8, "lambda_l2": 3.0, "max_bin": 255},
    notes="stream E: 170 cols, no window-bank features at all", seed=3, **BASE)

# F: same features, different gradient estimator (GOSS keeps large-gradient rows
# instead of sampling uniformly), so it sees a different effective sample.
run(exp_id="RT-125", modules=FULL,
    hypothesis="GOSS boosting on the full bank decorrelates from the uniformly-bagged champion",
    falsification="within-timestep rank correlation with RT-100 > 0.97",
    params={"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 95, "min_data_in_leaf": 250,
            "feature_fraction": 0.45, "boosting": "goss", "top_rate": 0.25, "other_rate": 0.15,
            "lambda_l2": 8.0, "max_bin": 127},
    notes="stream F: GOSS gradient sampling", seed=11, **BASE)
