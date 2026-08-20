"""Convert official parquet -> compact per-series binary store (float32 memmap).

Layout in cache/store/:
  values.f32            concatenated series values, all series in ascending id order
  meta.parquet          id, off, n_hist, n_online, tau_index (-1 = no break)
Series i occupies values[off : off+n_hist+n_online]; first n_hist are the
break-free historical segment, the rest the online segment.
"""
import numpy as np, pandas as pd, pyarrow.parquet as pq, os, sys

ROOT = os.environ.get(
    "SBR_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
)
RAW = os.environ.get("SBR_RAW_DIR", f"{ROOT}/data")
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/cache/store"
XF = sys.argv[2] if len(sys.argv) > 2 else "X_train.parquet"
YIF = sys.argv[3] if len(sys.argv) > 3 else "y_train_index.parquet"

os.makedirs(OUT, exist_ok=True)
yi = pd.read_parquet(f"{RAW}/{YIF}").reset_index()  # id, tau_index, tau

pf = pq.ParquetFile(f"{RAW}/{XF}")
total = pf.metadata.num_rows
vals = np.lib.format.open_memmap(f"{OUT}/values.npy", mode="w+", dtype=np.float32, shape=(total,))

rows = []
pos = 0
carry = None  # (id, list_of_arrays_value, list_period)
def flush(sid, v, p):
    global pos
    v = np.concatenate(v); p = np.concatenate(p)
    n_h = int((p == 1).sum()); n_o = int((p == 2).sum())
    assert n_h + n_o == len(v), (sid, n_h, n_o, len(v))
    # period must be sorted 1s then 2s
    assert (p[:n_h] == 1).all() and (p[n_h:] == 2).all(), sid
    vals[pos:pos+len(v)] = v
    rows.append((sid, pos, n_h, n_o))
    pos += len(v)

for rg in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(rg, columns=["id","time","value","period"]).to_pandas().reset_index()
    ids = t["id"].to_numpy()
    v = t["value"].to_numpy(np.float32); p = t["period"].to_numpy(np.int8)
    # boundaries
    b = np.flatnonzero(np.diff(ids)) + 1
    starts = np.r_[0, b]; ends = np.r_[b, len(ids)]
    for s, e in zip(starts, ends):
        sid = int(ids[s])
        if carry is not None and carry[0] == sid:
            carry[1].append(v[s:e]); carry[2].append(p[s:e])
        else:
            if carry is not None:
                flush(carry[0], carry[1], carry[2])
            carry = (sid, [v[s:e]], [p[s:e]])
    print(f"rg {rg} done, series so far {len(rows)}", flush=True)
if carry is not None:
    flush(carry[0], carry[1], carry[2])

assert pos == total, (pos, total)
vals.flush()
meta = pd.DataFrame(rows, columns=["id", "off", "n_hist", "n_online"])
meta = meta.merge(yi[["id", "tau_index"]], on="id", how="left")
assert meta["tau_index"].notna().all()
meta["tau_index"] = meta["tau_index"].astype(int)
meta["has_break"] = (meta["tau_index"] >= 0).astype(int)
meta.to_parquet(f"{OUT}/meta.parquet", index=False)
print(meta.head())
print("series", len(meta), "rows", total)
print(meta[["n_hist", "n_online"]].describe())
print("break rate", meta.has_break.mean())
