"""Plan 0.8 -- full-scale memory-lean inner teacher pilot.

Implements the two memory changes and confirms both extrapolations at real scale:
  change 1: free the float32 matrix once lgb.Dataset has binned it
  change 2: chunk the prediction instead of materialising a second 3 GB matrix
plus the chunked in-place augmented_stack from 0.7 lever 3, and lever 2's
num_threads=8 (verified bitwise-neutral at reduced scale).

It also demonstrates 0.7 lever 1 in situ: ONE booster serves both the
(outer=0,inner=1) and (outer=1,inner=0) roles, predicting onto folds 1 and 0.

Writes NOTHING to research/oof. Timing and memory only; no score, no OOF vector.
"""
import gc, json, os, resource, sys, time
ROOT = "/path/to/workspace/structural-break-rt1320-promotion-2026"
OUT = os.path.dirname(os.path.abspath(__file__))
os.environ["SBR_ROOT"] = ROOT
sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{ROOT}/research/scripts")
import numpy as np, lightgbm as lgb, sbr.pipeline as PL
from wave7_d3r import FULL, ARM_B_PARAMS, augmented_stack, last_row_lookup
from wave5_lib import FOLDS

ROWS, ROUNDS, THREADS = 1_000_000, 900, 8
STACK_CHUNK, PRED_CHUNK = 100_000, 200_000
OUTER, INNER = 0, 1

def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
def now(): return time.strftime("%H:%M:%S")

d = PL.Data(); mats, names = PL.load_features(FULL)
keep = np.arange(len(names)); final_of_row = last_row_lookup(d)[d.sidx]

def stack_chunked(rows, chunk=STACK_CHUNK):
    n, k = len(rows), len(keep)
    X = np.empty((n, 2 * k), dtype=np.float32)
    uniq, inv = np.unique(final_of_row[rows], return_inverse=True)
    fut_u = PL._stack(mats, names, uniq, keep)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        X[s:e, :k] = PL._stack(mats, names, rows[s:e], keep)
        X[s:e, k:] = fut_u[inv[s:e]]
    return X

def predict_chunked(bst, rows, chunk=PRED_CHUNK):
    out = np.empty(len(rows), dtype=np.float64)
    for s in range(0, len(rows), chunk):
        e = min(s + chunk, len(rows))
        Xc = stack_chunked(rows[s:e])
        out[s:e] = bst.predict(Xc)
        del Xc; gc.collect()
    return out

# ---------------------------------------------------------------- equivalence
print(f"[{now()}] EQUIVALENCE CHECK: chunked predict == whole-matrix predict", flush=True)
sm_tr = np.sort(np.random.default_rng(0).choice(d.rows_for([2, 3, 4]), 60_000, replace=False))
Xs = stack_chunked(sm_tr)
p0 = dict(ARM_B_PARAMS); p0.pop("n_estimators"); p0["num_threads"] = THREADS
ds_s = lgb.Dataset(Xs, label=d.y[sm_tr], params=dict(p0, objective="binary"),
                   feature_name=[f"f{i}" for i in range(Xs.shape[1])])
bst_s = lgb.train(dict(p0), ds_s, num_boost_round=15)
sm_va = np.sort(np.random.default_rng(7).choice(d.rows_for([0]), 40_000, replace=False))
whole = bst_s.predict(stack_chunked(sm_va))
chunked = predict_chunked(bst_s, sm_va, chunk=9_000)
ok = np.array_equal(whole, chunked)
print(f"[{now()}]   bitwise equal: {ok}   max|diff| {np.abs(whole-chunked).max():.3e}", flush=True)
if not ok:
    sys.exit("ABORT: chunked prediction is not bitwise identical; do not proceed")
del Xs, ds_s, bst_s, whole, chunked; gc.collect()

# --------------------------------------------------------------- full pilot
print(f"\n[{now()}] FULL-SCALE PILOT  rows={ROWS:,} rounds={ROUNDS} threads={THREADS}", flush=True)
t_all = time.time()
train_folds = sorted(set(FOLDS) - {OUTER, INNER})
tr = d.rows_for(train_folds)
rng = np.random.default_rng(0)
if len(tr) > ROWS:
    tr = np.sort(rng.choice(tr, ROWS, replace=False))
print(f"[{now()}] train_folds={train_folds}  rows={len(tr):,}", flush=True)

t = time.time(); X = stack_chunked(tr); t_stack = time.time() - t
print(f"[{now()}] stack        {t_stack:7.1f}s  {X.nbytes/2**30:.2f} GB  peak_rss {rss():.2f} GB", flush=True)

p = dict(ARM_B_PARAMS); p.pop("n_estimators"); p["num_threads"] = THREADS
t = time.time()
ds = lgb.Dataset(X, label=d.y[tr], params=dict(p, objective="binary"),
                 feature_name=[f"f{i}" for i in range(X.shape[1])])
ds.construct()                       # change 1: bin now ...
del X; gc.collect()                  # ... then free the float32 matrix
t_ds = time.time() - t
print(f"[{now()}] dataset+free {t_ds:7.1f}s  peak_rss {rss():.2f} GB", flush=True)

t = time.time(); bst = lgb.train(dict(p), ds, num_boost_round=ROUNDS); t_train = time.time() - t
print(f"[{now()}] train        {t_train:7.1f}s  = {t_train/ROUNDS:.3f} s/round  peak_rss {rss():.2f} GB", flush=True)

# lever 1 in situ: the SAME booster serves (0,1) and (1,0)
preds = {}
for g in (INNER, OUTER):
    t = time.time(); va = d.rows_for([g]); pr = predict_chunked(bst, va)
    preds[g] = (len(va), float(pr.mean()), time.time() - t)
    print(f"[{now()}] predict fold {g}: {len(va):,} rows  {time.time()-t:6.1f}s  "
          f"mean {pr.mean():.4f}  peak_rss {rss():.2f} GB", flush=True)

total = time.time() - t_all
res = {"rows": ROWS, "rounds": ROUNDS, "threads": THREADS, "train_folds": train_folds,
       "stack_s": round(t_stack,1), "dataset_s": round(t_ds,1), "train_s": round(t_train,1),
       "s_per_round": round(t_train/ROUNDS,4), "peak_rss_gb": round(rss(),2),
       "predict": {str(k): {"rows": v[0], "mean": v[1], "s": round(v[2],1)} for k,v in preds.items()},
       "total_s": round(total,1), "total_min": round(total/60,1)}
print(f"\n[{now()}] TOTAL {total/60:.1f} min   peak_rss {rss():.2f} GB")
print(f"  one teacher serving BOTH roles: {total/60:.1f} min")
print(f"  10 distinct teachers / partition: {total*10/3600:.1f} h")
print(f"  three alt partitions:            {total*30/3600:.1f} h")
json.dump(res, open(os.path.join(OUT, "pilot.json"), "w"), indent=2)
