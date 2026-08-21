"""WAVE-5 shared machinery.  Never edits sbr core files.

Everything here operates on the CANONICAL 8,000-series development framework:
research/folds/folds.parquet (folds 0..4) plus the three alternate partitions.

FORBIDDEN AND NOT REACHABLE FROM THIS FILE
  * the RT-500..RT-506 all-10k OOF vectors (final calibration only, post-freeze)
  * folds_final10k.parquet
  * X_test.reduced / y_test.reduced / y_test_index.reduced
  * fold -1 (the spent lockbox) as an evaluation surface

The two reference arms are the W4-E1 arms, unchanged:

  S  specialist set   RT-300, RT-410..RT-415   (the RT-600 architecture)
  B  seed-clone set   RT-300, RT-401..RT-406   (the matched diversity control)
"""
from __future__ import annotations

import os, sys

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

import numpy as np
import pandas as pd

from sbr.metric import ts_auc_flat
from sbr.pipeline import Data
from wave4_cal import SCDF_T, SCDF_NSEEN, GlobalCDF, logit, sigmoid

FOLDS = (0, 1, 2, 3, 4)
OOFDIR = f"{ROOT}/research/oof"
REPORTS = f"{ROOT}/research/reports"

#: the RT-600 architecture's seven streams, in freeze order
SPECIALISTS = ["RT-300", "RT-410", "RT-411", "RT-412", "RT-413", "RT-414", "RT-415"]
#: the matched seed-clone control, seeds 0/1/7/42/2026/31415/271828
SEEDCLONES = ["RT-300", "RT-401", "RT-402", "RT-403", "RT-404", "RT-405", "RT-406"]

#: CANONICAL calibration coordinate per research/FINAL_ARCHITECTURE_FREEZE.md
CANON_CAL = SCDF_NSEEN

AGE_BUCKETS = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 10 ** 9)]


# ---------------------------------------------------------------------------
class Ctx:
    """Row-space bookkeeping + break metadata, loaded once."""

    def __init__(self, folds_file=None):
        import sbr.pipeline as PL
        self._old_folds = None
        if folds_file:
            self._old_folds, PL.FOLDS = PL.FOLDS, folds_file
        try:
            self.d = Data()
        finally:
            if self._old_folds:
                PL.FOLDS = self._old_folds
        self.rows = {k: self.d.rows_for([k]) for k in FOLDS}
        self.dev = self.d.rows_for(list(FOLDS))
        meta = pd.read_parquet(f"{ROOT}/cache/store/meta.parquet")
        self.tau = meta.tau_index.to_numpy()
        self.has_break = meta.has_break.to_numpy().astype(bool)
        self.n_online = meta.n_online.to_numpy()
        self.n_hist = meta.n_hist.to_numpy()
        # post-break age per online row; -1 for rows that are not post-break
        tau_row = self.tau[self.d.sidx]
        self.age = np.where(self.d.y == 1, self.d.t - tau_row, -1)

    # -- scoring ------------------------------------------------------------
    def score(self, v):
        """(mean over the five canonical folds, per-fold list)."""
        per = [float(ts_auc_flat(v[self.rows[k]], self.d.y[self.rows[k]],
                                 self.d.t[self.rows[k]])) for k in FOLDS]
        return float(np.mean(per)), per

    def pooled(self, v):
        return float(ts_auc_flat(v[self.dev], self.d.y[self.dev], self.d.t[self.dev]))

    def score_by_age(self, v, buckets=AGE_BUCKETS):
        """TS-AUC with positives restricted to one post-break age bucket.

        Negatives are held fixed (every no-break row), so the buckets share
        their comparison set and the numbers are commensurable.
        """
        out = {}
        r = self.dev
        y, t, a = self.d.y[r], self.d.t[r], self.age[r]
        neg = y == 0
        for lo, hi in buckets:
            m = neg | ((y == 1) & (a >= lo) & (a < hi))
            key = f"{lo}-{hi if hi < 10 ** 9 else ''}"
            out[key] = {"ts_auc": float(ts_auc_flat(v[r][m], y[m], t[m])),
                        "n_pos": int(((y == 1) & (a >= lo) & (a < hi)).sum())}
        return out

    # -- calibration --------------------------------------------------------
    def crossfit_blend(self, P, streams, cal=None, weights=None):
        """Equal- or given-weight mean of cross-fitted calibrated streams.

        Fold k's calibration map is fitted on folds != k, so a fold never
        informs its own ranking.
        """
        cal = cal or CANON_CAL
        w = np.ones(len(streams)) if weights is None else np.asarray(weights, float)
        w = w / w.sum()
        out = np.full(len(self.d.y), np.nan)
        for k in FOLDS:
            tr = np.concatenate([self.rows[g] for g in FOLDS if g != k])
            va = self.rows[k]
            fs = [cal(P[s][tr], self.d.t[tr]) for s in streams]
            cols = np.column_stack([f(P[s][va], self.d.t[va]) for f, s in zip(fs, streams)])
            out[va] = cols @ w
        return out

    def crossfit_single(self, p, cal=None):
        """Cross-fitted calibrated version of ONE stream (for fair mixing)."""
        return self.crossfit_blend({"x": p}, ["x"], cal=cal)

    # -- inference ----------------------------------------------------------
    def bootstrap(self, vecs, contrasts, n=200, seed=0):
        """Paired series-level bootstrap with common random numbers."""
        r = self.dev
        sid = self.d.sidx[r]
        order = np.argsort(sid, kind="stable")
        sid_s = sid[order]
        b = np.flatnonzero(np.r_[True, sid_s[1:] != sid_s[:-1]])
        e = np.r_[b[1:], len(sid_s)]
        srows = {int(sid_s[i]): r[order[i:j]] for i, j in zip(b, e)}
        uniq = np.unique(sid)
        rng = np.random.default_rng(seed)
        picks = [rng.choice(uniq, len(uniq), replace=True) for _ in range(n)]
        acc = {c: [] for c in contrasts}
        for pk in picks:
            rr = np.concatenate([srows[int(s)] for s in pk])
            y, tt = self.d.y[rr], self.d.t[rr]
            a = {k: ts_auc_flat(v[rr], y, tt) for k, v in vecs.items()}
            for c, (x, z) in contrasts.items():
                acc[c].append(float(a[x] - a[z]))
        out = {}
        for c, v in acc.items():
            v = np.asarray(v)
            out[c] = {"mean": float(v.mean()), "median": float(np.median(v)),
                      "ci95": [float(np.quantile(v, .025)), float(np.quantile(v, .975))],
                      "fraction_positive": float((v > 0).mean()), "n": int(n)}
        return out


def load_oof(names, suffix=""):
    miss = [e for e in names if not os.path.exists(f"{OOFDIR}/{e}{suffix}.npy")]
    if miss:
        raise SystemExit(f"MISSING OOF: {miss}")
    return {e: np.load(f"{OOFDIR}/{e}{suffix}.npy") for e in names}


def fmt(per):
    return " ".join(f"{x:.5f}" for x in per)
