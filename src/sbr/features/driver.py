"""Compute feature modules over the whole store and cache them to disk.

One pass over the series builds the shared per-series context (historical
params, null-calibration engine, transforms) once and hands it to every
requested module, so adding modules is cheap relative to the shared precompute.

Output per module:  cache/features/<name>.npy   (n_online_rows, k) float32
                    cache/features/<name>.cols.json
Rows are in series-major order matching Store.orow_off, so any subset of
modules can be hstacked without a join.
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np

from sbr.features.base import load_all, make_ctx, REGISTRY
from sbr.store import load_store

OUT = "/home/claude/sb/cache/features"


def _worker(args):
    lo, hi, mods, store_path = args
    from sbr.store import load_store as _ls
    st = _ls(store_path)
    res = {m: [] for m in mods}
    for i in range(lo, hi):
        h, o, _ = st.series(i)
        ctx = make_ctx(h, o)
        for m in mods:
            names, A = REGISTRY[m].fn(ctx)
            res[m].append(A)
    return lo, hi, {m: np.vstack(v) for m, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", required=True)
    ap.add_argument("--store", default="/home/claude/sb/cache/store")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    load_all()
    mods = a.modules.split(",")
    for m in mods:
        if m not in REGISTRY:
            raise SystemExit(f"unknown module {m}; have {sorted(REGISTRY)}")

    st = load_store(a.store)
    n_series = st.n_series if a.limit <= 0 else min(a.limit, st.n_series)
    n_rows = int(st.meta.n_online.to_numpy()[:n_series].sum())
    os.makedirs(a.out, exist_ok=True)

    h, o, _ = st.series(0)
    ctx = make_ctx(h, o)
    colmap = {}
    mm = {}
    for m in mods:
        names, A = REGISTRY[m].fn(ctx)
        colmap[m] = names
        json.dump({"cols": names, "version": REGISTRY[m].version, "owner": REGISTRY[m].owner},
                  open(os.path.join(a.out, f"{m}.cols.json"), "w"))
        mm[m] = np.lib.format.open_memmap(os.path.join(a.out, f"{m}.npy"), mode="w+",
                                          dtype=np.float32, shape=(n_rows, len(names)))

    tasks = [(lo, min(lo + a.chunk, n_series), mods, a.store) for lo in range(0, n_series, a.chunk)]
    t0 = time.time()
    done = 0
    if a.workers > 1:
        import multiprocessing as mp
        with mp.Pool(a.workers) as pool:
            for lo, hi, res in pool.imap(_worker, tasks):
                s = int(st.orow_off[lo]); e = s + int(st.meta.n_online.to_numpy()[lo:hi].sum())
                for m in mods:
                    mm[m][s:e] = res[m]
                done += hi - lo
                if done % 500 < a.chunk:
                    el = time.time() - t0
                    print(f"{done}/{n_series} series  {el:.0f}s  eta {el/done*(n_series-done):.0f}s", flush=True)
    else:
        for t in tasks:
            lo, hi, res = _worker(t)
            s = int(st.orow_off[lo]); e = s + int(st.meta.n_online.to_numpy()[lo:hi].sum())
            for m in mods:
                mm[m][s:e] = res[m]
            done += hi - lo
    for m in mods:
        mm[m].flush()
        print(m, "->", mm[m].shape)
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
