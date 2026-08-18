"""Wave 5 -- leakage-safe portfolio construction over full-scale OOF streams.

Two design points that matter:

1. TS-AUC is invariant to any monotone transform applied identically to every
   score WITHIN a timestep.  So the canonical representation of a stream is its
   WITHIN-TIMESTEP RANK PERCENTILE, not its raw probability.  Correlations
   between streams are measured there too -- a global Pearson is dominated by a
   shared time trend the metric never scores, which makes independent streams
   look like clones.

2. Stacking is leave-one-fold-out.  The blend applied to fold f is fitted only
   on the OOF rows of the other folds, and every base OOF prediction was itself
   made by a model that never saw its own fold.  No model in the chain has seen
   fold f.
"""
import sys, json, itertools; sys.path.insert(0, "/home/claude/sb/src")
import numpy as np
from sbr.pipeline import Data
from sbr.metric import ts_auc_flat

STREAMS = ["RT-100", "RT-120", "RT-121", "RT-122", "RT-123", "RT-124", "RT-125"]
d = Data()
dev = d.rows_for([0, 1, 2, 3, 4])
y = d.y[dev]; t = d.t[dev].astype(np.int64); rf = d.row_fold[dev]


def within_t_rank(s):
    """Rank percentile of each score among the series alive at the same t."""
    order = np.lexsort((s, t))
    n = len(s)
    ts = t[order]
    g = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]])
    glen = np.r_[g[1:], n] - g
    pos = np.arange(n) - np.repeat(g, glen)
    # mid-ranks for ties so the transform is exactly the metric's own view
    new = np.r_[True, (ts[1:] != ts[:-1]) | (s[order][1:] != s[order][:-1])]
    rs = np.flatnonzero(new); re = np.r_[rs[1:], n]
    avg = np.repeat(pos[rs] + (re - rs - 1) / 2.0, re - rs)
    out = np.empty(n)
    out[order] = avg / np.maximum(np.repeat(glen, glen) - 1, 1)
    return out


raw, rk = {}, {}
for s in STREAMS:
    try:
        v = np.load(f"/home/claude/sb/research/oof/{s}.npy")[dev].astype(np.float64)
    except FileNotFoundError:
        print("missing", s); continue
    if not np.isfinite(v).all():
        print("skip (incomplete)", s); continue
    raw[s] = v; rk[s] = within_t_rank(v)

names = list(raw)
out = {"streams": {}, "correlations": {}, "blends": {}}
for s in names:
    out["streams"][s] = {"ts_auc": float(ts_auc_flat(raw[s], y, t)),
                         "per_fold": [float(ts_auc_flat(raw[s][rf == f], y[rf == f], t[rf == f])) for f in range(5)]}

for a, b in itertools.combinations(names, 2):
    out["correlations"][f"{a}|{b}"] = {
        "global_pearson": float(np.corrcoef(raw[a], raw[b])[0, 1]),
        "within_t_rank_pearson": float(np.corrcoef(rk[a], rk[b])[0, 1]),
    }

M = np.column_stack([rk[s] for s in names])


def score_blend(v):
    return float(ts_auc_flat(v, y, t)), [float(ts_auc_flat(v[rf == f], y[rf == f], t[rf == f])) for f in range(5)]


# --- equal rank average over every subset
for r in range(1, len(names) + 1):
    for sub in itertools.combinations(range(len(names)), r):
        if r == 1:
            continue
        v = M[:, list(sub)].mean(axis=1)
        sc, pf = score_blend(v)
        out["blends"]["eq_" + "+".join(names[i] for i in sub)] = {"ts_auc": sc, "per_fold": pf}

# --- leave-one-fold-out logistic stack on within-timestep ranks
from sklearn.linear_model import LogisticRegression

blend = np.empty(len(y))
coefs = {}
for f in range(5):
    tr = rf != f; te = rf == f
    idx = np.flatnonzero(tr)
    sub = idx[np.random.default_rng(0).choice(len(idx), min(600_000, len(idx)), replace=False)]
    lr = LogisticRegression(max_iter=200, C=1.0)
    lr.fit(M[sub], y[sub])
    blend[te] = lr.decision_function(M[te])
    coefs[str(f)] = dict(zip(names, np.round(lr.coef_[0], 4).tolist()))
sc, pf = score_blend(blend)
out["blends"]["lofo_logistic_stack"] = {"ts_auc": sc, "per_fold": pf, "coefs_per_fold": coefs}

# --- leave-one-fold-out LightGBM stack on ranks + online index
import lightgbm as lgb
Ms = np.column_stack([M, t.astype(np.float32)])
blend2 = np.empty(len(y))
for f in range(5):
    tr = np.flatnonzero(rf != f); te = rf == f
    sub = tr[np.random.default_rng(1).choice(len(tr), min(600_000, len(tr)), replace=False)]
    p = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=2000,
             feature_fraction=1.0, num_threads=2, verbose=-1, max_bin=63)
    ds = lgb.Dataset(Ms[sub], label=y[sub], params=p)
    b = lgb.train(p, ds, num_boost_round=150)
    blend2[te] = b.predict(Ms[te])
sc, pf = score_blend(blend2)
out["blends"]["lofo_lgbm_stack"] = {"ts_auc": sc, "per_fold": pf}

# --- is SUBSET SELECTION worth anything, honestly measured?
# Greedy forward selection fitted on four folds, scored on the fifth.  Picking
# the best of 120 subsets on the pooled OOF and reporting that number would be
# selecting on the score we report.
lofo_sel = {}
lofo_pred = np.empty(len(y))
for f in range(5):
    tr = rf != f
    chosen, cur = [], -1.0
    while True:
        cand = [(float(ts_auc_flat(M[tr][:, chosen + [i]].mean(axis=1), y[tr], t[tr])), i)
                for i in range(len(names)) if i not in chosen]
        sc_i, i_best = max(cand)
        if sc_i <= cur + 1e-6:
            break
        cur, _ = sc_i, chosen.append(i_best)
    lofo_sel[str(f)] = [names[i] for i in chosen]
    lofo_pred[rf == f] = M[rf == f][:, chosen].mean(axis=1)
sc, pf = score_blend(lofo_pred)
out["blends"]["lofo_greedy_subset"] = {"ts_auc": sc, "per_fold": pf, "subset_per_fold": lofo_sel}

best = max(out["blends"].items(), key=lambda kv: kv[1]["ts_auc"])
champ = out["streams"].get("RT-100", {}).get("ts_auc", float("nan"))
all_eq = out["blends"]["eq_" + "+".join(names)]["ts_auc"]
out["summary"] = {
    "champion_RT-100": champ,
    "all_streams_equal_average": all_eq,          # parameter-free, no selection
    "all_streams_delta": all_eq - champ,
    "best_blend_in_hindsight": best[0],           # diagnostic only -- selected on the reported score
    "best_blend_ts_auc": best[1]["ts_auc"],
    "honest_subset_selection_lofo": out["blends"]["lofo_greedy_subset"]["ts_auc"],
    "selection_premium_over_all_streams":
        out["blends"]["lofo_greedy_subset"]["ts_auc"] - all_eq,
}
np.save("/home/claude/sb/research/oof/RT-130.blend.npy", blend)
json.dump(out, open("/home/claude/sb/research/reports/portfolio.json", "w"), indent=2)
print(json.dumps(out["streams"], indent=2))
print(json.dumps(out["correlations"], indent=2))
print(json.dumps({k: v["ts_auc"] for k, v in out["blends"].items()}, indent=2))
print(json.dumps(out["summary"], indent=2))
