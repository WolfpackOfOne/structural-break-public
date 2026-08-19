"""B1/B2/B3/B4 -- can any of RT-131's ensemble gain be recovered LEGALLY?

RT-131 averages WITHIN-TIMESTEP CROSS-SECTIONAL RANK PERCENTILES.  Under the
Crunch streaming interface that object does not exist (see
research/reports/runner_semantics.md), so RT-131 is an ORACLE/DIAGNOSTIC upper
bound, not a score.

This script measures, on identical folds:
    RT-100                              the legal single-model floor
    ORACLE rank average                 the illegal ceiling
    and every DEPLOYABLE blend we can think of, each one a FROZEN per-series
    transform fitted only on training folds and applied to the held-out fold.

recoverable fraction = (deployable - RT-100) / (oracle - RT-100)

Every calibration map is CROSS-FITTED: for validation fold k the map is built
from OOF scores on folds != k only.  Fold k's own score distribution never
informs fold k's ranking.  That is the whole discipline here.
"""
from __future__ import annotations

import itertools, json, os, sys
import numpy as np

sys.path.insert(0, "/home/claude/sb/src")
from sbr.metric import ts_auc_flat
from sbr.pipeline import Data

OOF = "/home/claude/sb/research/oof"
OUT = "/home/claude/sb/research/reports/deployable_ensemble.json"
FOLDS = (0, 1, 2, 3, 4)
#: time buckets for the conditional CDF (B1 candidate 4)
EDGES = np.array([10, 20, 50, 100, 200, 400, 700])


def within_t_rank(scores, t):
    """Percentile rank of each row inside its own timestep. ORACLE ONLY."""
    order = np.lexsort((scores, t))
    s, tt = scores[order], t[order]
    n = len(s)
    gstart = np.flatnonzero(np.r_[True, tt[1:] != tt[:-1]])
    glen = np.r_[gstart[1:], n] - gstart
    pos = np.arange(n) - np.repeat(gstart, glen)
    new = np.r_[True, (tt[1:] != tt[:-1]) | (s[1:] != s[:-1])]
    rs = np.flatnonzero(new)
    re_ = np.r_[rs[1:], n]
    rl = re_ - rs
    avg = pos[rs] + (rl - 1) / 2.0
    r = np.repeat(avg, rl) / np.maximum(np.repeat(glen, 1)[np.repeat(np.arange(len(glen)), glen)] - 1, 1)
    out = np.empty(n)
    out[order] = r
    return out


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


class GlobalCDF:
    """F_m(s) from training-fold OOF scores only."""

    def __init__(self, s):
        self.grid = np.sort(np.asarray(s, dtype=np.float64))

    def __call__(self, s):
        return np.searchsorted(self.grid, s, side="left") / len(self.grid)


class TimeBucketCDF:
    """F_m(s | time bucket), buckets from EDGES, training folds only."""

    def __init__(self, s, t, edges=EDGES, min_n=200):
        self.edges = edges
        b = np.searchsorted(edges, t, side="right")
        self.grids = []
        allg = np.sort(s)
        for k in range(len(edges) + 1):
            g = np.sort(s[b == k])
            self.grids.append(g if len(g) >= min_n else allg)

    def __call__(self, s, t):
        b = np.searchsorted(self.edges, t, side="right")
        out = np.empty(len(s))
        for k in range(len(self.grids)):
            m = b == k
            if m.any():
                g = self.grids[k]
                out[m] = np.searchsorted(g, s[m], side="left") / len(g)
        return out


class SmoothTimeCDF:
    """F_m(s | t) on a log-spaced t grid with linear interpolation between the
    two neighbouring anchors -- the continuous version of TimeBucketCDF."""

    def __init__(self, s, t, n_anchor=12, min_n=400):
        tt = np.asarray(t)
        self.anchors = np.unique(np.round(np.exp(
            np.linspace(np.log(1), np.log(max(tt.max(), 2)), n_anchor))).astype(int))
        self.grids = []
        allg = np.sort(s)
        half = np.diff(np.log(np.maximum(self.anchors, 1))).mean() if len(self.anchors) > 1 else 1.0
        for a in self.anchors:
            lo, hi = a * np.exp(-half), a * np.exp(half)
            m = (tt >= lo) & (tt <= hi)
            g = np.sort(s[m])
            self.grids.append(g if len(g) >= min_n else allg)

    def __call__(self, s, t):
        la = np.log(np.maximum(self.anchors, 1)).astype(np.float64)
        lt = np.log(np.maximum(np.asarray(t, dtype=np.float64), 1))
        j = np.clip(np.searchsorted(la, lt, side="right"), 1, len(la) - 1)
        w = np.clip((lt - la[j - 1]) / np.maximum(la[j] - la[j - 1], 1e-9), 0, 1)
        out = np.empty(len(s))
        for k in (j - 1, j):
            pass
        lo = np.empty(len(s)); hi = np.empty(len(s))
        for idx in np.unique(j):
            m = j == idx
            glo, ghi = self.grids[idx - 1], self.grids[idx]
            lo[m] = np.searchsorted(glo, s[m], side="left") / len(glo)
            hi[m] = np.searchsorted(ghi, s[m], side="left") / len(ghi)
        return (1 - w) * lo + w * hi


def main(exp_ids):
    d = Data()
    preds = {}
    for e in exp_ids:
        p = f"{OOF}/{e}.npy"
        if not os.path.exists(p):
            print(f"  MISSING {e}, skipping"); continue
        preds[e] = np.load(p)
    names = list(preds)
    print("streams:", names)

    rows = {f: d.rows_for([f]) for f in FOLDS}
    res = {"streams": names, "per_stream": {}, "blends": {}}

    for e in names:
        pf = [float(ts_auc_flat(preds[e][rows[f]], d.y[rows[f]], d.t[rows[f]])) for f in FOLDS]
        res["per_stream"][e] = {"per_fold": pf, "mean": float(np.mean(pf))}
        print(f"  {e}: {np.mean(pf):.5f}  {['%.5f'%x for x in pf]}")

    def score(vecs_by_fold, label):
        pf = []
        for f in FOLDS:
            r = rows[f]
            pf.append(float(ts_auc_flat(vecs_by_fold[f], d.y[r], d.t[r])))
        m = float(np.mean(pf))
        res["blends"][label] = {"per_fold": pf, "mean": m}
        print(f"  {label:34s} {m:.5f}  {['%.5f'%x for x in pf]}")
        return m

    # ---------------- ORACLE (illegal): within-timestep rank average ----------
    orc = {}
    for f in FOLDS:
        r = rows[f]
        R = np.column_stack([within_t_rank(preds[e][r], d.t[r]) for e in names])
        orc[f] = R.mean(axis=1)
    m_oracle = score(orc, "ORACLE within-t rank avg [ILLEGAL]")

    # ---------------- deployable candidates, all cross-fitted ----------------
    def deployable(fn, label):
        out = {}
        for f in FOLDS:
            tr = np.concatenate([rows[g] for g in FOLDS if g != f])
            out[f] = fn(tr, rows[f])
        return score(out, label)

    m_raw = deployable(lambda tr, va: np.column_stack([preds[e][va] for e in names]).mean(axis=1),
                       "B1.1 raw probability mean")
    m_logit = deployable(lambda tr, va: 1 / (1 + np.exp(-np.column_stack(
        [logit(preds[e][va]) for e in names]).mean(axis=1))), "B1.2 logit mean")

    def _gcdf(tr, va):
        return np.column_stack([GlobalCDF(preds[e][tr])(preds[e][va]) for e in names]).mean(axis=1)
    m_gcdf = deployable(_gcdf, "B1.3 global OOF CDF (cross-fit)")

    def _bcdf(tr, va):
        return np.column_stack([TimeBucketCDF(preds[e][tr], d.t[tr])(preds[e][va], d.t[va])
                                for e in names]).mean(axis=1)
    m_bcdf = deployable(_bcdf, "B1.4 time-bucket CDF (cross-fit)")

    def _scdf(tr, va):
        return np.column_stack([SmoothTimeCDF(preds[e][tr], d.t[tr])(preds[e][va], d.t[va])
                                for e in names]).mean(axis=1)
    m_scdf = deployable(_scdf, "B1.5 smooth time-conditional CDF")

    base = res["per_stream"].get("RT-100R", {}).get("mean") or res["per_stream"][names[0]]["mean"]
    gap = m_oracle - base
    res["recoverable_fraction"] = {k: (v["mean"] - base) / gap for k, v in res["blends"].items()}
    res["single_model_floor"] = base
    res["oracle_gap"] = gap
    print("\nsingle-model floor", round(base, 5), " oracle gap", round(gap, 5))
    for k, v in res["recoverable_fraction"].items():
        print(f"  recoverable {k:34s} {100*v:6.1f}%")

    json.dump(res, open(OUT, "w"), indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main(sys.argv[1:] or ["RT-100R", "RT-120R", "RT-121R", "RT-122R",
                          "RT-123R", "RT-124R", "RT-125R"])
