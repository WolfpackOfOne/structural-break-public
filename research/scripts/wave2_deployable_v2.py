"""Follow-ups to the deployable-ensemble result, all on saved OOF arrays.

1. NESTED SELECTION. The five calibrations in wave2_deployable_ensemble.py were
   compared on all five folds, so picking the best of them is selection on the
   evaluation data. Here the calibration FAMILY is chosen on inner folds and
   scored once on the held-out outer fold, which is the honest number.
2. GRID QUANTISATION. Deployment cannot ship a 4M-value empirical CDF per stream
   per time anchor. Does a q-point quantile grid reproduce the full-grid score?
3. B4 DIVERSITY. Within-timestep rank correlations, and the marginal
   contribution of each stream under the DEPLOYABLE blend (not the oracle).
4. PAIRED BOOTSTRAP. Series-level resampling for deployable-blend minus RT-100R.
"""
from __future__ import annotations

import itertools, json, sys
import numpy as np

sys.path.insert(0, "/home/claude/sb/src")
sys.path.insert(0, "/home/claude/sb/research/scripts")
from sbr.metric import ts_auc_flat
from sbr.pipeline import Data
from wave2_deployable_ensemble import (GlobalCDF, SmoothTimeCDF, TimeBucketCDF,
                                       logit, within_t_rank)

STREAMS = ["RT-100R", "RT-120R", "RT-121R", "RT-122R", "RT-123R", "RT-124R", "RT-125R"]
FOLDS = (0, 1, 2, 3, 4)
OUT = "/home/claude/sb/research/reports/deployable_ensemble_v2.json"

d = Data()
P = {s: np.load(f"/home/claude/sb/research/oof/{s}.npy") for s in STREAMS}
rows = {f: d.rows_for([f]) for f in FOLDS}
res = {}


# ------------------------------------------------------------------ builders
def make(kind, streams, tr, q=None):
    """Fit one calibration family on training rows; return a transform fn."""
    if kind == "raw":
        return lambda va: np.column_stack([P[s][va] for s in streams]).mean(axis=1)
    if kind == "logitmean":
        return lambda va: 1 / (1 + np.exp(-np.column_stack([logit(P[s][va]) for s in streams]).mean(axis=1)))
    if kind == "gcdf":
        fs = [GlobalCDF(_q(P[s][tr], q)) for s in streams]
        return lambda va: np.column_stack([f(P[s][va]) for f, s in zip(fs, streams)]).mean(axis=1)
    if kind == "bcdf":
        fs = [TimeBucketCDF(P[s][tr], d.t[tr]) for s in streams]
        if q:
            for f in fs:
                f.grids = [_q(g, q) for g in f.grids]
        return lambda va: np.column_stack([f(P[s][va], d.t[va]) for f, s in zip(fs, streams)]).mean(axis=1)
    if kind == "scdf":
        fs = [SmoothTimeCDF(P[s][tr], d.t[tr]) for s in streams]
        if q:
            for f in fs:
                f.grids = [_q(g, q) for g in f.grids]
        return lambda va: np.column_stack([f(P[s][va], d.t[va]) for f, s in zip(fs, streams)]).mean(axis=1)
    raise ValueError(kind)


def _q(g, q):
    """Subsample a sorted empirical grid to q points, preserving the CDF shape."""
    if q is None or len(g) <= q:
        return np.sort(g)
    g = np.sort(g)
    idx = np.linspace(0, len(g) - 1, q).astype(np.int64)
    return g[idx]


def sc(v, f):
    r = rows[f]
    return float(ts_auc_flat(v, d.y[r], d.t[r]))


KINDS = ["raw", "logitmean", "gcdf", "bcdf", "scdf"]

# ------------------------------------------------ 1. nested family selection
print("=== 1. nested selection of the calibration family ===")
picked, outer = [], []
for k in FOLDS:
    inner = [f for f in FOLDS if f != k]
    best, bestv = None, -1
    for kind in KINDS:
        vals = []
        for j in inner:                       # inner CV inside folds != k
            tr = np.concatenate([rows[g] for g in inner if g != j])
            vals.append(sc(make(kind, STREAMS, tr)(rows[j]), j))
        m = float(np.mean(vals))
        if m > bestv:
            best, bestv = kind, m
    tr = np.concatenate([rows[g] for g in inner])
    s = sc(make(best, STREAMS, tr)(rows[k]), k)
    picked.append(best); outer.append(s)
    print(f"  outer fold {k}: inner picks {best:9s} -> held-out {s:.5f}")
res["nested_selection"] = {"picked_per_fold": picked, "per_fold": outer,
                           "mean": float(np.mean(outer))}
print(f"  NESTED CONFIRMATION MEAN: {np.mean(outer):.5f}")

# --------------------------------------------------- 2. grid quantisation
print("\n=== 2. deployable grid size ===")
res["grid_quantisation"] = {}
for q in (256, 1024, 4096, 16384, None):
    pf = []
    for k in FOLDS:
        tr = np.concatenate([rows[g] for g in FOLDS if g != k])
        pf.append(sc(make("scdf", STREAMS, tr, q=q)(rows[k]), k))
    res["grid_quantisation"][str(q)] = {"per_fold": pf, "mean": float(np.mean(pf))}
    print(f"  scdf grid q={str(q):6s} mean {np.mean(pf):.5f}")

# ------------------------------------------------------------ 3. diversity
print("\n=== 3. diversity and marginal contribution under the DEPLOYABLE blend ===")
dev = d.rows_for(list(FOLDS))
rk = {s: within_t_rank(P[s][dev], d.t[dev]) for s in STREAMS}
corr = {}
for a, b in itertools.combinations(STREAMS, 2):
    corr[f"{a}|{b}"] = {"within_t_rank": float(np.corrcoef(rk[a], rk[b])[0, 1]),
                        "global_pearson": float(np.corrcoef(P[a][dev], P[b][dev])[0, 1])}
res["correlations"] = corr

full = []
for k in FOLDS:
    tr = np.concatenate([rows[g] for g in FOLDS if g != k])
    full.append(sc(make("scdf", STREAMS, tr)(rows[k]), k))
res["deployable_full"] = {"per_fold": full, "mean": float(np.mean(full))}
loo = {}
for s in STREAMS:
    sub = [x for x in STREAMS if x != s]
    pf = []
    for k in FOLDS:
        tr = np.concatenate([rows[g] for g in FOLDS if g != k])
        pf.append(sc(make("scdf", sub, tr)(rows[k]), k))
    loo[s] = {"mean_without": float(np.mean(pf)),
              "marginal": float(np.mean(full) - np.mean(pf)),
              "n_folds_hurt_by_removal": int(sum(a > b for a, b in zip(full, pf)))}
    print(f"  drop {s}: {np.mean(pf):.5f}  marginal {np.mean(full)-np.mean(pf):+.5f} "
          f"({loo[s]['n_folds_hurt_by_removal']}/5 folds worse without it)")
res["leave_one_stream_out_deployable"] = loo

# --------------------------------------------------- 4. paired bootstrap
print("\n=== 4. paired series-level bootstrap: deployable blend - RT-100R ===")
blend = np.full(len(d.y), np.nan)
for k in FOLDS:
    tr = np.concatenate([rows[g] for g in FOLDS if g != k])
    blend[rows[k]] = make("scdf", STREAMS, tr)(rows[k])
sid = d.sidx[dev]
uniq = np.unique(sid)
srows = {}
order = np.argsort(sid, kind="stable")
sid_s = sid[order]
bnd = np.flatnonzero(np.r_[True, sid_s[1:] != sid_s[:-1]])
ends = np.r_[bnd[1:], len(sid_s)]
for b, e in zip(bnd, ends):
    srows[sid_s[b]] = dev[order[b:e]]
rng = np.random.default_rng(0)
deltas = []
for _ in range(200):
    pick = rng.choice(uniq, len(uniq), replace=True)
    r = np.concatenate([srows[s] for s in pick])
    a = ts_auc_flat(blend[r], d.y[r], d.t[r])
    b = ts_auc_flat(P["RT-100R"][r], d.y[r], d.t[r])
    deltas.append(float(a - b))
deltas = np.array(deltas)
res["paired_bootstrap_vs_RT100"] = {
    "n_replicates": len(deltas), "mean_delta": float(deltas.mean()),
    "ci95": [float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))],
    "fraction_positive": float((deltas > 0).mean())}
print(f"  mean {deltas.mean():+.5f}  95% CI [{np.quantile(deltas,.025):+.5f}, "
      f"{np.quantile(deltas,.975):+.5f}]  {100*(deltas>0).mean():.0f}% positive")

json.dump(res, open(OUT, "w"), indent=2)
print("\nwrote", OUT)
