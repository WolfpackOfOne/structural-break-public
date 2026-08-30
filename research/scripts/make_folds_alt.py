"""ALTERNATIVE grouped fold partitions -- ROBUSTNESS TESTING ONLY.

These are NOT a replacement for research/folds/folds.parquet, which is permanent
and never regenerated.  They exist to answer one question: does a conclusion
drawn on the canonical partition survive a different, equally valid, partition
of the same dev series?

Rules of use (binding, see research/VALIDATION_V2.md):
  * The SAME 8,000 dev series are repartitioned.  The 2,000-series original
    lockbox (fold == -1) is excluded and stays excluded.
  * Alternative partitions may be used to REPORT A DISTRIBUTION.  They may never
    be used to select a model, a feature set, a hyperparameter or a seed, and a
    result may not be re-run on further partitions until it looks better.
  * Each partition preserves the canonical stratification: has_break x tau
    quartile x n_hist tertile x n_online tertile.
"""
import hashlib, json, os, sys
import numpy as np, pandas as pd

FOLDS = "/home/claude/sb/research/folds"
N_FOLDS = 5
SEEDS = {"alt1": 20260901, "alt2": 20260902, "alt3": 20260903}

base = pd.read_parquet(f"{FOLDS}/folds.parquet")
dev = base[base.split == "dev"].reset_index(drop=True)
out = {}
for name, seed in SEEDS.items():
    rng = np.random.default_rng(seed)
    fold = np.full(len(dev), -1, dtype=int)
    for s, idx in dev.groupby("stratum").indices.items():
        idx = np.asarray(idx)
        idx = idx[rng.permutation(len(idx))]
        fold[idx] = np.arange(len(idx)) % N_FOLDS
    df = dev[["id"]].copy()
    df["fold"] = fold
    df.to_parquet(f"{FOLDS}/folds_{name}.parquet", index=False)
    sha = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
    # agreement with canonical: fraction of series in the same fold label
    agree = float((fold == dev.fold.to_numpy()).mean())
    out[name] = {
        "seed": seed, "sha256": sha, "n_series": int(len(df)),
        "label_agreement_with_canonical": agree,
        "per_fold": {str(f): {
            "n_series": int((fold == f).sum()),
            "break_rate": float(dev.has_break.to_numpy()[fold == f].mean()),
            "mean_n_online": float(dev.n_online.to_numpy()[fold == f].mean()),
            "mean_n_hist": float(dev.n_hist.to_numpy()[fold == f].mean()),
            "n_online_rows": int(dev.n_online.to_numpy()[fold == f].sum()),
        } for f in range(N_FOLDS)},
    }
json.dump(out, open(f"{FOLDS}/folds_alt_summary.json", "w"), indent=2)
print(json.dumps({k: {"sha256": v["sha256"], "agreement": round(v["label_agreement_with_canonical"], 4),
                      "break_rates": [round(v["per_fold"][str(f)]["break_rate"], 4) for f in range(N_FOLDS)]}
                  for k, v in out.items()}, indent=2))
