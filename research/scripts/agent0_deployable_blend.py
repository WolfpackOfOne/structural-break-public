"""RT-160 -- can the ensemble actually be computed at inference time?

The crunch runner is single-pass and SERIES-SEQUENTIAL: `infer` walks the
datasets one series at a time and must emit a score for online step t of series
i before it has seen series i+1 at all.  Our champion blend averages
WITHIN-TIMESTEP RANK PERCENTILES, which is a function of the cross-section at
step t.  That information does not exist at inference.  RT-131 as specified is
therefore NOT DEPLOYABLE.

This measures what the deployable alternatives cost.  A blend is deployable iff
it is a fixed per-series function of one series' own scores:
  raw       mean of the streams' probabilities
  logit     mean of log(p/(1-p))
  qnorm     mean after mapping each stream through ITS OWN score distribution,
            estimated offline and frozen into a lookup table.  This is the
            closest deployable relative of rank-averaging, and it is the only
            option that also handles the pairwise stream, whose output is an
            unbounded margin rather than a probability.

The qnorm lookup is fitted on the OTHER folds' OOF scores and applied to the
held-out fold, so the estimate stays leakage-safe.
"""
import sys, json; sys.path.insert(0, "/home/claude/sb/src")
import numpy as np
from sbr.pipeline import Data
from sbr.metric import ts_auc_flat

STREAMS = ["RT-100", "RT-120", "RT-121", "RT-122", "RT-123", "RT-124", "RT-125"]
d = Data()
dev = d.rows_for([0, 1, 2, 3, 4])
y, t, rf = d.y[dev], d.t[dev].astype(np.int64), d.row_fold[dev]

raw = {}
for s in STREAMS:
    v = np.load(f"/home/claude/sb/research/oof/{s}.npy")[dev].astype(np.float64)
    if np.isfinite(v).all():
        raw[s] = v
names = list(raw)
print("streams:", names, flush=True)
print("score ranges:", {k: (round(v.min(), 3), round(v.max(), 3)) for k, v in raw.items()}, flush=True)


def within_t_rank(s):
    order = np.lexsort((s, t)); n = len(s); ts = t[order]
    g = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]]); glen = np.r_[g[1:], n] - g
    pos = np.arange(n) - np.repeat(g, glen)
    new = np.r_[True, (ts[1:] != ts[:-1]) | (s[order][1:] != s[order][:-1])]
    rs = np.flatnonzero(new); re = np.r_[rs[1:], n]
    avg = np.repeat(pos[rs] + (re - rs - 1) / 2.0, re - rs)
    out = np.empty(n); out[order] = avg / np.maximum(np.repeat(glen, glen) - 1, 1)
    return out


def sc(v):
    per = [float(ts_auc_flat(v[rf == f], y[rf == f], t[rf == f])) for f in range(5)]
    return float(ts_auc_flat(v, y, t)), per


out = {"per_stream": {k: sc(v)[0] for k, v in raw.items()}}

# --- NOT deployable: the current champion, for reference
M_rank = np.column_stack([within_t_rank(raw[k]) for k in names])
p, pf = sc(M_rank.mean(axis=1))
out["rank_average_NOT_DEPLOYABLE"] = {"ts_auc": p, "per_fold": pf}

# --- deployable option 1: mean of raw scores
p, pf = sc(np.column_stack([raw[k] for k in names]).mean(axis=1))
out["deployable_raw_mean"] = {"ts_auc": p, "per_fold": pf}

# --- deployable option 2: mean of logits (clip probabilities; leave margins as-is)
def to_logit(v):
    if v.min() >= 0.0 and v.max() <= 1.0:
        q = np.clip(v, 1e-6, 1 - 1e-6)
        return np.log(q / (1 - q))
    return v          # already an unbounded margin (the pairwise stream)

p, pf = sc(np.column_stack([to_logit(raw[k]) for k in names]).mean(axis=1))
out["deployable_logit_mean"] = {"ts_auc": p, "per_fold": pf}

# --- deployable option 3: frozen per-stream quantile normalisation
GRID = 4096
qn = np.empty((len(y), len(names)))
for f in range(5):
    tr, te = rf != f, rf == f
    for j, k in enumerate(names):
        knots = np.quantile(raw[k][tr], np.linspace(0, 1, GRID))
        knots = np.maximum.accumulate(knots)
        qn[te, j] = np.searchsorted(knots, raw[k][te], side="left") / GRID
p, pf = sc(qn.mean(axis=1))
out["deployable_qnorm_mean"] = {"ts_auc": p, "per_fold": pf}

best_dep = max(("deployable_raw_mean", "deployable_logit_mean", "deployable_qnorm_mean"),
               key=lambda k: out[k]["ts_auc"])
out["summary"] = {
    "best_single_stream": max(out["per_stream"].values()),
    "rank_average_not_deployable": out["rank_average_NOT_DEPLOYABLE"]["ts_auc"],
    "best_deployable_blend": best_dep,
    "best_deployable_ts_auc": out[best_dep]["ts_auc"],
    "cost_of_deployability": out[best_dep]["ts_auc"] - out["rank_average_NOT_DEPLOYABLE"]["ts_auc"],
    "still_beats_best_single_by": out[best_dep]["ts_auc"] - max(out["per_stream"].values()),
}
json.dump(out, open("/home/claude/sb/research/reports/deployable_blend.json", "w"), indent=2)
print(json.dumps(out, indent=2))
