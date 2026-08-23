"""Synthetic store in the exact production layout.  EXECUTION SMOKE TEST ONLY.

Its scores are meaningless -- the data is not the competition's -- and no number
produced against it may appear in a research table.  What it proves is that the
wave-7 runners execute end to end against a real store, a real feature build and
the real scorer, so that the first thing the 2026 store meets is a working path.
"""
import os, sys
import numpy as np, pandas as pd

OUT = sys.argv[1]
rng = np.random.default_rng(2026)
N = 250
rows, vals = [], []
off = 0
for i in range(N):
    n_hist = int(rng.integers(1200, 4700))   # the real store spans 1140-4709; m00_core widens at ~900
    n_online = int(rng.integers(60, 400))
    has = int(rng.random() < 0.5)
    tau = int(rng.integers(0, n_online)) if has else -1
    hist = rng.normal(0.0, 1.0, n_hist)
    on = rng.normal(0.0, 1.0, n_online)
    if has:
        kind = rng.integers(0, 3)
        if kind == 0:
            on[tau:] += rng.normal(0, 1) * rng.uniform(0.2, 1.5)
        elif kind == 1:
            on[tau:] *= rng.uniform(1.3, 2.5)
        else:
            for k in range(tau + 1, n_online):
                on[k] += 0.6 * on[k - 1]
    v = np.r_[hist, on].astype(np.float32)
    vals.append(v)
    rows.append((i, off, n_hist, n_online, tau, has))
    off += len(v)

flat = np.concatenate(vals)
os.makedirs(f"{OUT}/cache/store_screen", exist_ok=True)
m = np.lib.format.open_memmap(f"{OUT}/cache/store_screen/values.npy", mode="w+",
                              dtype=np.float32, shape=flat.shape)
m[:] = flat; m.flush()
meta = pd.DataFrame(rows, columns=["id", "off", "n_hist", "n_online", "tau_index", "has_break"])
meta.to_parquet(f"{OUT}/cache/store_screen/meta.parquet", index=False)

folds = meta[["id", "n_hist", "n_online", "tau_index", "has_break"]].copy()
folds["fold"] = np.arange(N) % 5
folds["split"] = "dev"
folds["stratum"] = "synthetic"
folds = folds[["id", "fold", "split", "has_break", "tau_index", "n_hist", "n_online", "stratum"]]
folds.to_parquet(f"{OUT}/research/folds/folds_screen.parquet", index=False)
print(f"synthetic store: {N} series, {int(meta.n_online.sum())} online rows, "
      f"break rate {meta.has_break.mean():.2f}, max n_online {meta.n_online.max()}")
