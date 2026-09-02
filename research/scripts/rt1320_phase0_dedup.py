"""Is the (outer=f, inner=g) teacher identical to the (outer=g, inner=f) teacher?

If yes, build_nested_Q fits each distinct booster TWICE across the five outer
folds: 20 ordered pairs but only C(5,2)=10 distinct training sets. Halving that
is exact, not an approximation.

Reduced scale (rows/rounds) purely to make the check cheap. Writes nothing.
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

def build(outer_f, inner_g):
    """Mirrors train_inner_teacher exactly, at reduced scale."""
    train_folds = [x for x in FOLDS if x not in (outer_f, inner_g)]
    tr_rows = d.rows_for(train_folds)
    rng = np.random.default_rng(0)
    tr_rows = np.sort(rng.choice(tr_rows, ROWS, replace=False))
    X = augmented_stack(mats, names, keep, tr_rows, final_of_row)
    p = dict(ARM_B_PARAMS); p.pop("n_estimators")
    ds = lgb.Dataset(X, label=d.y[tr_rows], params=dict(p, objective="binary"),
                     feature_name=[f"f{i}" for i in range(X.shape[1])])
    bst = lgb.train(dict(p), ds, num_boost_round=ROUNDS)
    return train_folds, tr_rows, bst

t=time.time(); tf_a, tr_a, bst_a = build(0, 1); print(f"teacher(outer=0,inner=1) train_folds={tf_a}  {time.time()-t:.0f}s")
t=time.time(); tf_b, tr_b, bst_b = build(1, 0); print(f"teacher(outer=1,inner=0) train_folds={tf_b}  {time.time()-t:.0f}s")

print(f"\ntrain_folds identical : {tf_a == tf_b}")
print(f"train rows identical  : {np.array_equal(tr_a, tr_b)}  ({len(tr_a):,} rows)")
print(f"model string identical: {bst_a.model_to_string() == bst_b.model_to_string()}")

# predict both on a common held-out slab and compare bitwise
probe = np.sort(np.random.default_rng(7).choice(d.rows_for([0]), 20_000, replace=False))
Xp = augmented_stack(mats, names, keep, probe, final_of_row)
pa, pb = bst_a.predict(Xp), bst_b.predict(Xp)
print(f"predictions bitwise equal: {np.array_equal(pa, pb)}   max|diff| = {np.abs(pa-pb).max():.3e}")

print("\n=== distinct training sets across the full nested scheme ===")
seen = {}
for f in FOLDS:
    for g in [x for x in FOLDS if x != f]:
        seen.setdefault(frozenset((f, g)), []).append((f, g))
print(f"ordered (outer,inner) pairs fitted by build_nested_Q : {sum(len(v) for v in seen.values())}")
print(f"distinct training sets among them                    : {len(seen)}")
for k, v in sorted(seen.items(), key=lambda kv: sorted(kv[0])):
    print(f"  train on {sorted(set(FOLDS)-k)} <- fitted for {v}")
