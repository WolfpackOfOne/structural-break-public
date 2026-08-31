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

import argparse
import json
import os
import time

import numpy as np

from sbr.features.base import REGISTRY, load_all, make_ctx
from sbr.store import load_store

OUT = os.environ.get(
    "SBR_FEATURES",
    os.environ.get("SBR_ROOT", "/home/claude/sb") + "/cache/features",
)


def _worker(args):
    lo, hi, mods, store_path = args
    from sbr.store import load_store as _ls

    # Spawn-start workers begin with an empty REGISTRY; fork-start workers inherit it.
    load_all()
    st = _ls(store_path)
    res = {m: [] for m in mods}
    for i in range(lo, hi):
        h, o, _ = st.series(i)
        ctx = make_ctx(h, o)
        for m in mods:
            names, A = REGISTRY[m].fn(ctx)
            res[m].append(A)
    return lo, hi, {m: np.vstack(v) for m, v in res.items()}


def build_features(modules, store, out=OUT, workers=2, chunk=100, limit=0):
    """Build feature memmaps for ``modules`` and return a compact manifest."""
    load_all()
    mods = list(modules)
    for m in mods:
        if m not in REGISTRY:
            raise SystemExit(f"unknown module {m}; have {sorted(REGISTRY)}")

    st = load_store(store)
    n_series = st.n_series if limit <= 0 else min(limit, st.n_series)
    n_rows = int(st.meta.n_online.to_numpy()[:n_series].sum())
    os.makedirs(out, exist_ok=True)

    h, o, _ = st.series(0)
    ctx = make_ctx(h, o)
    colmap = {}
    mm = {}
    for m in mods:
        names, A = REGISTRY[m].fn(ctx)
        colmap[m] = names
        json.dump({"cols": names, "version": REGISTRY[m].version, "owner": REGISTRY[m].owner},
                  open(os.path.join(out, f"{m}.cols.json"), "w"))
        mm[m] = np.lib.format.open_memmap(os.path.join(out, f"{m}.npy"), mode="w+",
                                          dtype=np.float32, shape=(n_rows, len(names)))

    tasks = [(lo, min(lo + chunk, n_series), mods, store) for lo in range(0, n_series, chunk)]
    t0 = time.time()
    done = 0
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for lo, hi, res in pool.imap(_worker, tasks):
                s = int(st.orow_off[lo])
                e = s + int(st.meta.n_online.to_numpy()[lo:hi].sum())
                for m in mods:
                    mm[m][s:e] = res[m]
                done += hi - lo
                if done % 500 < chunk:
                    el = time.time() - t0
                    eta = el / done * (n_series - done)
                    print(f"{done}/{n_series} series  {el:.0f}s  eta {eta:.0f}s", flush=True)
    else:
        for t in tasks:
            lo, hi, res = _worker(t)
            s = int(st.orow_off[lo])
            e = s + int(st.meta.n_online.to_numpy()[lo:hi].sum())
            for m in mods:
                mm[m][s:e] = res[m]
            done += hi - lo
    for m in mods:
        mm[m].flush()
        print(m, "->", mm[m].shape)
    seconds = float(time.time() - t0)
    print(f"total {seconds:.0f}s")
    return {
        "store": store,
        "out": out,
        "modules": {m: {"shape": list(mm[m].shape), "columns": len(colmap[m])} for m in mods},
        "n_series": int(n_series),
        "n_online_rows": int(n_rows),
        "seconds": seconds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", required=True)
    ap.add_argument(
        "--store",
        default=os.environ.get(
            "SBR_STORE",
            os.environ.get("SBR_ROOT", "/home/claude/sb") + "/cache/store",
        ),
    )
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    build_features(
        a.modules.split(","),
        store=a.store,
        out=a.out,
        workers=int(a.workers),
        chunk=int(a.chunk),
        limit=int(a.limit),
    )


if __name__ == "__main__":
    main()
