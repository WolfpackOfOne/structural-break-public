"""FINAL-FIT fold partition over ALL 10,000 labelled series.  Run once.

This is NOT a research partition and must never be used to select anything.
It exists for one purpose: after the architecture is frozen, the final
calibration payload needs cross-fitted OOF scores, and every OOF vector the
project owns covers only the 8,000 dev series.  Scoring the final boosters'
calibration off an 8,000-series distribution while they are fitted on 10,000
would compound the training-regime mismatch that the deployable stack already
carries (see wave2_train_ensemble.py's "KNOWN, DOCUMENTED APPROXIMATION").

The old 2,000-series lockbox is folded in here as ORDINARY TRAINING DATA.  It
was spent for selection long ago -- two inspections -- and after freeze there is
no scientific reason to discard 20% of the labelled series.  Nothing scored
under this partition is evidence for any model choice; by the time it runs,
there are no model choices left to make.

Stratification is identical to make_folds.py -- has_break x tau quartile x
n_hist tertile x n_online tertile -- so the folds are comparable in composition
to the canonical ones.  The seed differs deliberately: reusing 20260818 would
reproduce the dev/lockbox permutation and correlate the two partitions for no
reason.

canonical folds.parquet is NEVER touched.
"""
import hashlib, json, os
import numpy as np, pandas as pd

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
STORE = os.environ.get("SBR_STORE", f"{ROOT}/cache/store")
OUT = f"{ROOT}/research/folds"
SEED = 20260920
N_FOLDS = 5

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
meta["stratum"] = (meta.has_break.astype(str) + "_" + meta.b_tau.astype(str) + "_"
                   + meta.b_hist.astype(str) + "_" + meta.b_online.astype(str))

fold = np.full(len(meta), -1, dtype=int)
for s, idx in meta.groupby("stratum").indices.items():
    idx = np.asarray(idx)
    idx = idx[rng.permutation(len(idx))]
    fold[idx] = np.arange(len(idx)) % N_FOLDS
assert (fold >= 0).all(), "every one of the 10,000 series must be in a fold"

meta["fold"] = fold
meta["split"] = "final10k"

os.makedirs(OUT, exist_ok=True)
cols = ["id", "fold", "split", "has_break", "tau_index", "n_hist", "n_online", "stratum"]
path = f"{OUT}/folds_final10k.parquet"
assert not os.path.exists(path), f"{path} exists -- refusing to regenerate a fold partition"
meta[cols].to_parquet(path, index=False)

h = hashlib.sha256(meta[["id", "fold"]].to_csv(index=False).encode()).hexdigest()
summary = {
    "purpose": "FINAL-FIT ONLY. Never used for model selection.",
    "seed": SEED, "n_folds": N_FOLDS, "n_series": int(len(meta)),
    "includes_former_lockbox": True,
    "canonical_folds_untouched": True,
    "sha256_id_fold": h,
    "per_fold": {str(f): {
        "n_series": int((meta.fold == f).sum()),
        "break_rate": float(meta.loc[meta.fold == f, "has_break"].mean()),
        "mean_n_online": float(meta.loc[meta.fold == f, "n_online"].mean()),
        "n_online_rows": int(meta.loc[meta.fold == f, "n_online"].sum()),
    } for f in range(N_FOLDS)},
}
json.dump(summary, open(f"{OUT}/folds_final10k_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
