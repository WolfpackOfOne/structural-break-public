"""Screen store: 500 dev series per fold (2,500 total), for cheap iteration.

Same layout as the full store, plus a folds file with identical fold labels, so
a screening run is structurally the same experiment as a full run -- just on
fewer series.  Series are drawn with a fixed seed and never redrawn.
"""
import numpy as np, pandas as pd, os
SEED = 987654321
SRC = "/home/claude/sb/cache/store"; DST = "/home/claude/sb/cache/store_screen"
os.makedirs(DST, exist_ok=True)
meta = pd.read_parquet(f"{SRC}/meta.parquet")
folds = pd.read_parquet("/home/claude/sb/research/folds/folds.parquet")
assert (meta.id.to_numpy() == folds.id.to_numpy()).all()
rng = np.random.default_rng(SEED)
sel = []
for f in range(5):
    idx = np.flatnonzero(folds.fold.to_numpy() == f)
    sel.append(rng.choice(idx, 500, replace=False))
sel = np.sort(np.concatenate(sel))
vals = np.load(f"{SRC}/values.npy", mmap_mode="r")
tot = int((meta.n_hist.to_numpy()[sel] + meta.n_online.to_numpy()[sel]).sum())
out = np.lib.format.open_memmap(f"{DST}/values.npy", mode="w+", dtype=np.float32, shape=(tot,))
pos = 0; rows = []
for i in sel:
    r = meta.iloc[i]; n = int(r.n_hist) + int(r.n_online)
    out[pos:pos+n] = vals[int(r.off):int(r.off)+n]
    rows.append((int(r.id), pos, int(r.n_hist), int(r.n_online), int(r.tau_index), int(r.has_break)))
    pos += n
out.flush()
m2 = pd.DataFrame(rows, columns=["id","off","n_hist","n_online","tau_index","has_break"])
m2.to_parquet(f"{DST}/meta.parquet", index=False)
folds.iloc[sel].reset_index(drop=True).to_parquet("/home/claude/sb/research/folds/folds_screen.parquet", index=False)
print("screen store", len(m2), "series", tot, "points", int(m2.n_online.sum()), "online rows")
print(m2.groupby(folds.iloc[sel].fold.to_numpy()).size())
