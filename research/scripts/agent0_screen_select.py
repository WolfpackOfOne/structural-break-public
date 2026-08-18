import sys; sys.path.insert(0,'/home/claude/sb/src')
from sbr.pipeline import run
P=dict(screen=True, folds=(0,), max_train_rows=300_000, params={'n_estimators':300}, agent='agent0',
       falsification='delta vs m00_core control <= 0')
runs=[
 ("RT-060L", ["m00_core","m06_loc"], "Online change-point localisation adds signal beyond prefix-anchored windows"),
 ("RT-090A", ["m00_core","m01_seq","m02_dist","m03_dyn","m04_resid","m06_loc","m07_bayes"], "Full causal feature bank (no context block) is the champion feature set"),
 ("RT-090B", ["m00_core","m01_seq","m02_dist","m03_dyn","m04_resid","m06_loc","m07_bayes","m05_ctx"], "Adding the series-constant context block on top of the full bank helps"),
]
for eid, mods, hyp in runs:
    print("===", eid, mods, flush=True)
    run(exp_id=eid, modules=mods, hypothesis=hyp, notes="agent0 screen selection", **P)
