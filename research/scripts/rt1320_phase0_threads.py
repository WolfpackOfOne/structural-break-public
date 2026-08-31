"""Does num_threads move the teacher's predictions?

ARM_B_PARAMS pins num_threads=2, which matches the 2-CPU Linux research box.
On a 10-core machine that leaves ~5x unused -- but PROTOCOL_CHAMPION_2026.md's
determinism gate means a thread change must be shown not to move predictions
before it may feed a decision. This is that demonstration, at reduced scale.
"""
import os, sys, time
ROOT = "/path/to/workspace/structural-break-rt1320-promotion-2026"
os.environ["SBR_ROOT"] = ROOT
sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{ROOT}/research/scripts")
import numpy as np, lightgbm as lgb, sbr.pipeline as PL
from wave7_d3r import FULL, ARM_B_PARAMS, augmented_stack, last_row_lookup
from wave5_lib import FOLDS

ROWS, ROUNDS = 120_000, 25
d = PL.Data(); mats, names = PL.load_features(FULL)
keep = np.arange(len(names)); final_of_row = last_row_lookup(d)[d.sidx]
tr = d.rows_for([2, 3, 4])
tr = np.sort(np.random.default_rng(0).choice(tr, ROWS, replace=False))
X = augmented_stack(mats, names, keep, tr, final_of_row); y = d.y[tr]
probe = np.sort(np.random.default_rng(7).choice(d.rows_for([0]), 20_000, replace=False))
Xp = augmented_stack(mats, names, keep, probe, final_of_row)

out = {}
for nt in (2, 4, 8, 10):
    p = dict(ARM_B_PARAMS); p.pop("n_estimators"); p["num_threads"] = nt
    t = time.time()
    ds = lgb.Dataset(X, label=y, params=dict(p, objective="binary"),
                     feature_name=[f"f{i}" for i in range(X.shape[1])])
    bst = lgb.train(dict(p), ds, num_boost_round=ROUNDS)
    dt = time.time() - t
    out[nt] = (bst.model_to_string(), bst.predict(Xp), dt)
    print(f"num_threads={nt:2d}  train {dt:6.1f}s  speedup vs 2: {out[2][2]/dt:4.2f}x", flush=True)

print("\n=== determinism vs the frozen num_threads=2 ===")
base_s, base_p, _ = out[2]
for nt in (4, 8, 10):
    s, pr, _ = out[nt]
    print(f"  nt={nt:2d}: model identical {s==base_s!s:5s}  "
          f"preds bitwise equal {np.array_equal(pr, base_p)!s:5s}  "
          f"max|diff| {np.abs(pr-base_p).max():.3e}")
