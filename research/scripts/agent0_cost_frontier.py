"""RT-17x -- accuracy/inference-cost Pareto frontier over MODULE SUBSETS.

All seven ensemble streams read the same feature bank, so the feature work at
inference is the UNION of modules used, computed once, plus seven cheap booster
calls.  Inference cost is therefore a function of the module set, not the stream
count -- which means the real deployment lever is which modules we keep, and
rewriting an expensive module into streaming form is only worth it if its
accuracy contribution justifies its milliseconds.

Measured reference-streamer cost, ms/point (416-point series, 7 modules = 92.9):
    m02_dist 27.59 | m07_bayes 17.63 | m04_resid 16.60 | m03_dyn 11.46
    m06_loc   9.14 | m00_core   5.29 | m01_seq    4.78
"""
import sys; sys.path.insert(0, "/home/claude/sb/src")
from sbr.pipeline import run

COST = {"m00_core": 5.29, "m01_seq": 4.78, "m02_dist": 27.59, "m03_dyn": 11.46,
        "m04_resid": 16.60, "m06_loc": 9.14, "m07_bayes": 17.63}
P = dict(agent="agent0", folds=(0, 1, 2, 3, 4), max_train_rows=700_000, seed=0,
         params={"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63,
                 "min_data_in_leaf": 300, "feature_fraction": 0.5,
                 "bagging_fraction": 0.7, "lambda_l2": 5.0, "max_bin": 127})

SUBSETS = [
    ("RT-170", ["m00_core", "m01_seq", "m03_dyn"]),                                   # 21.5 ms
    ("RT-171", ["m00_core", "m01_seq", "m03_dyn", "m06_loc"]),                        # 30.7 ms
    ("RT-172", ["m00_core", "m01_seq", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes"]),  # 65.3 ms
]
for eid, mods in SUBSETS:
    c = sum(COST[m] for m in mods)
    print(f"=== {eid} {mods} -- {c:.1f} ms/point", flush=True)
    run(exp_id=eid, modules=mods,
        hypothesis=f"A {c:.0f} ms/point module subset retains most of the full bank's 0.61510",
        falsification=f"mean OOF below 0.605, i.e. the dropped modules are load-bearing",
        notes=f"cost frontier: {c:.1f} ms/point at reference-streamer speed", **P)
