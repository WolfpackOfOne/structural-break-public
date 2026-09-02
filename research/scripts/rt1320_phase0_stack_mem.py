"""Chunked in-place augmented_stack: is it bitwise identical, and what does it save?

Original: np.concatenate([own, fut]) holds own + fut + result simultaneously.
Chunked : preallocate the result, fill both halves in row blocks, so only a
          block-sized temporary ever exists alongside it.

Pure data movement -- no arithmetic -- so equality should be exact, not close.
Run as: probe_stack_mem.py <orig|chunked> <n_rows>
"""
import hashlib, os, resource, sys, time
ROOT = "/path/to/workspace/structural-break-rt1320-promotion-2026"
os.environ["SBR_ROOT"] = ROOT
sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{ROOT}/research/scripts")
import numpy as np, sbr.pipeline as PL
from wave7_d3r import FULL, augmented_stack, last_row_lookup
from wave5_lib import FOLDS

MODE, N = sys.argv[1], int(sys.argv[2])
CHUNK = 100_000

def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30

def augmented_stack_chunked(mats, names, keep_idx, rows, final_of_row, chunk=CHUNK):
    n, k = len(rows), len(keep_idx)
    X = np.empty((n, 2 * k), dtype=np.float32)
    frows = final_of_row[rows]
    uniq, inv = np.unique(frows, return_inverse=True)
    fut_u = PL._stack(mats, names, uniq, keep_idx)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        X[s:e, :k] = PL._stack(mats, names, rows[s:e], keep_idx)
        X[s:e, k:] = fut_u[inv[s:e]]
    return X

d = PL.Data(); mats, names = PL.load_features(FULL)
keep = np.arange(len(names)); final_of_row = last_row_lookup(d)[d.sidx]
tr = d.rows_for([2, 3, 4])
rows = np.sort(np.random.default_rng(0).choice(tr, N, replace=False))
base = rss()
t = time.time()
X = augmented_stack(mats, names, keep, rows, final_of_row) if MODE == "orig" \
    else augmented_stack_chunked(mats, names, keep, rows, final_of_row)
dt = time.time() - t
h = hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()
print(f"mode={MODE:8s} n={N:,} shape={X.shape} {X.dtype}")
print(f"  time {dt:6.1f}s   array {X.nbytes/2**30:.2f} GB   rss_before {base:.2f} GB   PEAK_RSS {rss():.2f} GB")
print(f"  sha256 {h}")
