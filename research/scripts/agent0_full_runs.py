import sys; sys.path.insert(0,'/home/claude/sb/src')
from sbr.pipeline import run
FULL=["m00_core","m01_seq","m02_dist","m03_dyn","m04_resid","m06_loc","m07_bayes"]
P=dict(agent="agent0", folds=(0,1,2,3,4), max_train_rows=1_000_000,
       params={"n_estimators":600,"learning_rate":0.05,"num_leaves":63,"min_data_in_leaf":300,
               "feature_fraction":0.5,"bagging_fraction":0.7,"lambda_l2":5.0,"max_bin":127})
run(exp_id="RT-100", modules=FULL, hypothesis="Full causal feature bank (7 modules, 500 cols) + LightGBM is the champion",
    falsification="mean OOF TS-AUC <= 0.575 (public-method territory)", notes="CHAMPION candidate, full 5-fold", **P)
run(exp_id="RT-101", modules=["m00_core"], hypothesis="Calibrated multi-scale null evidence alone, full 5-fold reference",
    falsification="mean OOF TS-AUC <= 0.55", notes="single-module reference", **P)
run(exp_id="RT-102", modules=FULL+["m05_ctx"], hypothesis="Series-constant historical context block adds value on full data",
    falsification="delta vs RT-100 <= 0", notes="context block test at full scale", **P)
