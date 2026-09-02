"""Plan 0.5 -- scaled timing probe for one inner teacher fit.

Runs the FROZEN ARM_B_PARAMS architecture at reduced rows/rounds, measures the
per-round cost, and extrapolates to the real fit (1,000,000 rows x 900 rounds).

NOT a result. Produces no OOF vector, writes nothing to research/oof, and must
never be quoted as a score. Timing and memory only.
"""
import json, os, resource, sys, time
ROOT = "/path/to/workspace/structural-break-rt1320-promotion-2026"
OUT = os.path.dirname(os.path.abspath(__file__))
os.environ["SBR_ROOT"] = ROOT
sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{ROOT}/research/scripts")
import numpy as np, lightgbm as lgb, sbr.pipeline as PL
from wave7_d3r import FULL, ARM_B_PARAMS, augmented_stack, last_row_lookup
from wave5_lib import FOLDS

REAL_ROWS, REAL_ROUNDS = 1_000_000, ARM_B_PARAMS["n_estimators"]
PROBE_ROWS, PROBE_ROUNDS = 250_000, 60
OUTER, INNER = 0, 1

def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30

d = PL.Data(); mats, names = PL.load_features(FULL)
keep = np.arange(len(names)); final_of_row = last_row_lookup(d)[d.sidx]
train_folds = [x for x in FOLDS if x not in (OUTER, INNER)]
tr_all = d.rows_for(train_folds)
rows = np.sort(np.random.default_rng(0).choice(tr_all, PROBE_ROWS, replace=False))
print(f"frozen params: {ARM_B_PARAMS}")
print(f"real fit  = {REAL_ROWS:,} rows x {REAL_ROUNDS} rounds")
print(f"probe fit = {PROBE_ROWS:,} rows x {PROBE_ROUNDS} rounds  (teacher outer={OUTER} inner={INNER})\n")

t = time.time(); X = augmented_stack(mats, names, keep, rows, final_of_row)
t_stack = time.time() - t
y = d.y[rows]
print(f"augmented_stack {t_stack:6.1f}s  shape {X.shape}  {X.nbytes/2**30:.2f} GB  rss {rss():.2f} GB")

p = dict(ARM_B_PARAMS); p.pop("n_estimators")
t = time.time()
ds = lgb.Dataset(X, label=y, params=dict(p, objective="binary"),
                 feature_name=[f"f{i}" for i in range(X.shape[1])])
bst = lgb.train(dict(p), ds, num_boost_round=PROBE_ROUNDS)
t_train = time.time() - t
per_round = t_train / PROBE_ROUNDS
print(f"train {PROBE_ROUNDS} rounds  {t_train:6.1f}s  = {per_round:.3f} s/round  rss {rss():.2f} GB")

row_scale = REAL_ROWS / PROBE_ROWS
proj_train = per_round * REAL_ROUNDS * row_scale
proj_stack = t_stack * row_scale
proj_fit = proj_stack + proj_train
res = {
  "probe": {"rows": PROBE_ROWS, "rounds": PROBE_ROUNDS, "stack_s": round(t_stack,1),
            "train_s": round(t_train,1), "s_per_round": round(per_round,4),
            "peak_rss_gb": round(rss(),2)},
  "projected_one_inner_teacher": {
      "stack_s": round(proj_stack,1), "train_s": round(proj_train,1),
      "total_s": round(proj_fit,1), "total_min": round(proj_fit/60,1),
      "matrix_gb": round(REAL_ROWS*1000*4/2**30,2),
      "matrix_peak_gb": round(REAL_ROWS*1000*4/2**30 + REAL_ROWS*500*4*2/2**30,2)},
}
# 20 inner teachers + 5 outer folds (student fits) per partition
res["projected_phase1"] = {
  "one_partition_20_inner_h": round(proj_fit*20/3600,1),
  "three_alt_partitions_60_inner_h": round(proj_fit*60/3600,1),
}
print("\n=== PROJECTION (linear in rows and rounds) ===")
print(f"  one inner teacher : {proj_fit/60:.1f} min  (stack {proj_stack/60:.1f} + train {proj_train/60:.1f})")
print(f"  20 inner / partition : {proj_fit*20/3600:.1f} h")
print(f"  60 inner / 3 alt partitions : {proj_fit*60/3600:.1f} h")
print(f"  matrix at full scale : {res['projected_one_inner_teacher']['matrix_gb']} GB "
      f"(peak ~{res['projected_one_inner_teacher']['matrix_peak_gb']} GB)")
json.dump(res, open(os.path.join(OUT, "probe_timing.json"), "w"), indent=2)
