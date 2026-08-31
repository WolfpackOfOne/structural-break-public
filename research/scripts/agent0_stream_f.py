"""Stream F -- GOSS gradient sampling (bagging disabled; LightGBM forbids both)."""
import sys; sys.path.insert(0, "/home/claude/sb/src")
from sbr.pipeline import run

FULL = ["m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes"]
run(exp_id="RT-125", modules=FULL, agent="agent0", folds=(0, 1, 2, 3, 4), max_train_rows=700_000,
    hypothesis="GOSS boosting on the full bank decorrelates from the uniformly-bagged champion",
    falsification="within-timestep rank correlation with RT-100 > 0.97",
    params={"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 95, "min_data_in_leaf": 250,
            "feature_fraction": 0.45, "boosting": "goss", "top_rate": 0.25, "other_rate": 0.15,
            "lambda_l2": 8.0, "max_bin": 127, "bagging_fraction": 1.0, "bagging_freq": 0},
    notes="stream F: GOSS keeps large-gradient rows instead of sampling uniformly", seed=11)
