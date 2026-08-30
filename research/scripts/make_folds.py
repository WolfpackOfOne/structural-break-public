"""PERMANENT series-level folds + lockbox.  Run once; never regenerate.

10,000 training series are split into:
  * LOCKBOX  : 2,000 series, never used for any model/feature/hyperparameter
               selection.  Touched only for a final confirmation of a decision
               already made on the dev folds.
  * DEV      : 8,000 series in 5 stratified folds.

Stratification is on has_break x tau_index quartile x n_hist tertile x
n_online tertile, so every fold sees the same mix of break/no-break, early/late
breaks, and short/long histories and online segments.
"""
import hashlib, json, os
import numpy as np, pandas as pd

STORE = "/home/claude/sb/cache/store"
OUT = "/home/claude/sb/research/folds"
SEED = 20260818
N_FOLDS = 5
N_LOCKBOX = 2000

meta = pd.read_parquet(f"{STORE}/meta.parquet").copy()
rng = np.random.default_rng(SEED)

hb = meta.has_break.to_numpy()
tau = meta.tau_index.to_numpy().astype(float)
tau_rel = np.where(hb == 1, tau / meta.n_online.to_numpy(), -1.0)

def qbucket(x, q, mask=None):
    b = np.full(len(x), -1, dtype=int)
    m = np.ones(len(x), bool) if mask is None else mask
    if m.sum() == 0:
        return b
    edges = np.quantile(x[m], np.linspace(0, 1, q + 1)[1:-1])
    b[m] = np.searchsorted(edges, x[m])
    return b

meta["b_tau"] = qbucket(tau_rel, 4, hb == 1)
meta["b_hist"] = qbucket(meta.n_hist.to_numpy().astype(float), 3)
meta["b_online"] = qbucket(meta.n_online.to_numpy().astype(float), 3)
meta["stratum"] = (
    meta.has_break.astype(str) + "_" + meta.b_tau.astype(str) + "_"
    + meta.b_hist.astype(str) + "_" + meta.b_online.astype(str)
)

# --- lockbox: proportional draw from every stratum ---
is_lock = np.zeros(len(meta), bool)
frac = N_LOCKBOX / len(meta)
for s, idx in meta.groupby("stratum").indices.items():
    idx = np.asarray(idx)
    idx = idx[rng.permutation(len(idx))]
    k = int(round(len(idx) * frac))
    is_lock[idx[:k]] = True
# fix rounding drift
diff = N_LOCKBOX - is_lock.sum()
pool = np.flatnonzero(~is_lock) if diff > 0 else np.flatnonzero(is_lock)
pick = rng.choice(pool, abs(diff), replace=False)
is_lock[pick] = diff > 0
assert is_lock.sum() == N_LOCKBOX

fold = np.full(len(meta), -1, dtype=int)   # -1 == lockbox
dev = np.flatnonzero(~is_lock)
devmeta = meta.iloc[dev]
for s, idx in devmeta.groupby("stratum").indices.items():
    idx = dev[np.asarray(idx)]
    idx = idx[rng.permutation(len(idx))]
    fold[idx] = np.arange(len(idx)) % N_FOLDS

meta["fold"] = fold
meta["split"] = np.where(is_lock, "lockbox", "dev")

os.makedirs(OUT, exist_ok=True)
cols = ["id", "fold", "split", "has_break", "tau_index", "n_hist", "n_online", "stratum"]
meta[cols].to_parquet(f"{OUT}/folds.parquet", index=False)

h = hashlib.sha256(meta[["id", "fold"]].to_csv(index=False).encode()).hexdigest()
summary = {
    "seed": SEED, "n_folds": N_FOLDS, "n_lockbox": int(N_LOCKBOX),
    "sha256": h,
    "per_fold": {
        str(f): {
            "n_series": int((meta.fold == f).sum()),
            "break_rate": float(meta.loc[meta.fold == f, "has_break"].mean()),
            "mean_n_online": float(meta.loc[meta.fold == f, "n_online"].mean()),
            "mean_n_hist": float(meta.loc[meta.fold == f, "n_hist"].mean()),
            "mean_tau_rel": float(np.mean(tau_rel[(meta.fold == f) & (hb == 1)])),
            "n_online_rows": int(meta.loc[meta.fold == f, "n_online"].sum()),
        } for f in list(range(N_FOLDS)) + [-1]
    },
}
json.dump(summary, open(f"{OUT}/folds_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
