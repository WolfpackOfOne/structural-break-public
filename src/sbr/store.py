"""Per-series data store for the 2026 Real-Time structural break challenge.

The store is a flat float32 memmap of concatenated series plus a metadata
table.  Series ``i`` (row ``i`` of ``meta``) occupies
``values[off : off + n_hist + n_online]``; the first ``n_hist`` points are the
break-free historical segment and the remaining ``n_online`` are the online
segment revealed one point at a time at inference.

``tau_index`` is the 0-based index *within the online segment* of the first
post-break observation, or ``-1`` for a no-break series.  The online-step
target is ``y[t] = 1 if 0 <= tau_index <= t else 0``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_STORE = os.environ.get("SBR_STORE", "/home/claude/sb/cache/store")


@dataclass
class Store:
    values: np.ndarray          # (N,) float32 memmap
    meta: pd.DataFrame          # id, off, n_hist, n_online, tau_index, has_break
    orow_off: np.ndarray        # (S,) int64 start of each series' online rows in flat online space
    n_online_rows: int

    @property
    def n_series(self) -> int:
        return len(self.meta)

    def hist(self, i: int) -> np.ndarray:
        r = self.meta.iloc[i]
        o = int(r.off)
        return np.asarray(self.values[o : o + int(r.n_hist)], dtype=np.float64)

    def online(self, i: int) -> np.ndarray:
        r = self.meta.iloc[i]
        o = int(r.off) + int(r.n_hist)
        return np.asarray(self.values[o : o + int(r.n_online)], dtype=np.float64)

    def series(self, i: int):
        return self.hist(i), self.online(i), int(self.meta.iloc[i].tau_index)

    def labels(self, i: int) -> np.ndarray:
        r = self.meta.iloc[i]
        n = int(r.n_online)
        tau = int(r.tau_index)
        y = np.zeros(n, dtype=np.int8)
        if tau >= 0:
            y[tau:] = 1
        return y

    def all_labels(self) -> np.ndarray:
        """Flat int8 label vector over all online rows (series-major)."""
        out = np.zeros(self.n_online_rows, dtype=np.int8)
        n_on = self.meta.n_online.to_numpy()
        tau = self.meta.tau_index.to_numpy()
        for i in range(self.n_series):
            if tau[i] >= 0:
                s = self.orow_off[i]
                out[s + tau[i] : s + n_on[i]] = 1
        return out

    def all_t_index(self) -> np.ndarray:
        """Flat int32 vector giving the online step index t of each online row."""
        n_on = self.meta.n_online.to_numpy()
        return np.concatenate([np.arange(n, dtype=np.int32) for n in n_on])

    def all_series_index(self) -> np.ndarray:
        n_on = self.meta.n_online.to_numpy()
        return np.repeat(np.arange(self.n_series, dtype=np.int32), n_on)


def load_store(path: str = DEFAULT_STORE) -> Store:
    meta = pd.read_parquet(os.path.join(path, "meta.parquet"))
    values = np.load(os.path.join(path, "values.npy"), mmap_mode="r")
    n_on = meta.n_online.to_numpy()
    orow_off = np.r_[0, np.cumsum(n_on)[:-1]].astype(np.int64)
    return Store(values=values, meta=meta, orow_off=orow_off, n_online_rows=int(n_on.sum()))
