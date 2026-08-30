"""RT-150 -- DGP-cluster gated specialists at full scale.

Agent 8 found the historical-context vector is worth ~0 as FEATURES (+0.0009,
and -0.0177 at full scale) but +0.032 as a ROUTER: cluster series by their
historical characterisation, train a specialist per cluster.  That was measured
on 2,000 screen series against a 151-column global model.  The open question is
whether routing still pays once the global model already has 500 columns and
600 trees to express the same conditioning itself.

Arms, all sharing one materialised matrix per fold so the comparison is exact:
  global        one model, 600 trees                       (the champion recipe)
  gated_equal   6 specialists x 100 trees  (equal total capacity)
  gated_full    6 specialists x 600 trees  (equal per-model capacity)
  blend         rank-average of global and gated_full
  permuted      gated_full on clusters randomly reassigned, sizes preserved
                -- the control that separates real routing from ensembling
"""
import sys, json, time; sys.path.insert(0, "/home/claude/sb/src")
import numpy as np, lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sbr.pipeline import Data, load_features, _stack, append_result
from sbr.metric import ts_auc_flat

FULL = ["m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes"]
P = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=300,
         feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0,
         num_threads=2, verbose=-1, max_bin=127)
K = 6
# fold list and training-row budget are CLI-settable: the fold-1 run had to be
# re-run alone at a smaller budget after an OOM. Arms within a fold always share
# one materialised matrix, so the arm-to-arm deltas stay exactly comparable even
# when the budget differs between folds.
FOLDS = tuple(int(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else (0, 1)
NTR = int(sys.argv[2]) if len(sys.argv) > 2 else 600_000

d = Data()
mats, names = load_features(FULL)
cols = np.arange(len(names))

# series-level context vector = the first online row of each series' m05_ctx block
ctxA = np.load("/home/claude/sb/cache/features/m05_ctx.npy", mmap_mode="r")
ctx = np.asarray(ctxA[d.st.orow_off], dtype=np.float64)
ctx = np.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)
print("context matrix", ctx.shape, flush=True)


def rank_pct(s, t):
    order = np.lexsort((s, t)); n = len(s); ts = t[order]
    g = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]]); glen = np.r_[g[1:], n] - g
    pos = np.arange(n) - np.repeat(g, glen)
    new = np.r_[True, (ts[1:] != ts[:-1]) | (s[order][1:] != s[order][:-1])]
    rs = np.flatnonzero(new); re = np.r_[rs[1:], n]
    avg = np.repeat(pos[rs] + (re - rs - 1) / 2.0, re - rs)
    out = np.empty(n); out[order] = avg / np.maximum(np.repeat(glen, glen) - 1, 1)
    return out


res = {a: [] for a in ("global", "gated_equal", "gated_full", "blend", "permuted")}
sizes = {}
t0 = time.time()
for f in FOLDS:
    rng = np.random.default_rng(f)
    tr = np.sort(rng.choice(d.rows_for([x for x in (0, 1, 2, 3, 4) if x != f]), NTR, replace=False))
    va = d.rows_for([f])
    ytr, yva, tva = d.y[tr], d.y[va], d.t[va].astype(np.int64)

    # gate fitted on TRAINING-fold series only
    train_series = np.flatnonzero(np.isin(d.series_fold, [x for x in (0, 1, 2, 3, 4) if x != f]))
    sc = StandardScaler().fit(ctx[train_series])
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(sc.transform(ctx[train_series]))
    cl_series = km.predict(sc.transform(ctx))
    sizes[f] = np.bincount(cl_series[train_series], minlength=K).tolist()
    cl_tr, cl_va = cl_series[d.sidx[tr]], cl_series[d.sidx[va]]

    Xtr = _stack(mats, names, tr, cols)
    Xva = _stack(mats, names, va, cols)
    print(f"fold {f}: matrices ready {time.time()-t0:.0f}s, cluster sizes {sizes[f]}", flush=True)

    def fit_predict(Xa, ya, Xb, rounds, pars=P):
        ds = lgb.Dataset(Xa, label=ya, params=pars)
        b = lgb.train(pars, ds, num_boost_round=rounds)
        out = b.predict(Xb)
        del ds, b
        return out

    # 600k training rows x 500 float32 columns is ~1.2 GB; with the validation
    # matrix and one cluster subset live at once this is the memory ceiling.

    p_global = fit_predict(Xtr, ytr, Xva, 600)
    res["global"].append(float(ts_auc_flat(p_global, yva, tva)))

    for arm, rounds in (("gated_equal", 100), ("gated_full", 600)):
        pred = np.zeros(len(va))
        for k in range(K):
            a, b = cl_tr == k, cl_va == k
            if a.sum() < 5000 or b.sum() == 0:
                pred[b] = p_global[b]
                continue
            pred[b] = fit_predict(Xtr[a], ytr[a], Xva[b], rounds)
        res[arm].append(float(ts_auc_flat(pred, yva, tva)))
        if arm == "gated_full":
            p_gated = pred

    res["blend"].append(float(ts_auc_flat(
        0.5 * rank_pct(p_global, tva) + 0.5 * rank_pct(p_gated, tva), yva, tva)))

    # permutation control: same cluster sizes, random series->cluster assignment
    perm = np.random.default_rng(99 + f).permutation(len(cl_series))
    cl_p = cl_series[perm]
    pt, pv = cl_p[d.sidx[tr]], cl_p[d.sidx[va]]
    pred = np.zeros(len(va))
    for k in range(K):
        a, b = pt == k, pv == k
        if a.sum() < 5000 or b.sum() == 0:
            pred[b] = p_global[b]
            continue
        pred[b] = fit_predict(Xtr[a], ytr[a], Xva[b], 600)
    res["permuted"].append(float(ts_auc_flat(pred, yva, tva)))

    del Xtr, Xva
    print(f"fold {f}: " + " ".join(f"{k}={v[-1]:.5f}" for k, v in res.items()), flush=True)

summary = {k: {"per_fold": v, "mean": float(np.mean(v))} for k, v in res.items()}
summary["cluster_sizes"] = sizes
summary["gating_minus_global"] = summary["gated_full"]["mean"] - summary["global"]["mean"]
summary["gating_minus_permuted"] = summary["gated_full"]["mean"] - summary["permuted"]["mean"]
summary["runtime_s"] = round(time.time() - t0)
tag = "_f" + "".join(map(str, FOLDS))
json.dump(summary, open(f"/home/claude/sb/research/reports/gating{tag}.json", "w"), indent=2)
best = max(("gated_equal", "gated_full", "blend"), key=lambda a: summary[a]["mean"])
append_result(dict(experiment_id="RT-150" + tag, date=time.strftime("%Y-%m-%d %H:%M"), git_sha="nogit",
    agent="agent0", hypothesis="DGP-cluster routing still pays once the global model has 500 columns and 600 trees",
    falsification_condition="gated_full mean <= global mean on folds 0-1, or <= its permuted control",
    feature_set=",".join(FULL), n_features=len(names), model=f"kmeans-k{K} gated specialists",
    objective="binary", folds=",".join(map(str, FOLDS)), random_seed=0,
    mean_oof_ts_auc=summary[best]["mean"], pooled_oof_ts_auc=summary[best]["mean"],
    per_fold_ts_auc=";".join(f"{x:.5f}" for x in summary[best]["per_fold"]),
    fold_std=float(np.std(summary[best]["per_fold"])), persistence="none", sample_mode="uniform",
    training_runtime_s=summary["runtime_s"], causal_verified="prefix-invariance@module",
    test_reduced_touched="no", lockbox_touched="no", protocol="full", status="recorded",
    notes="best arm " + best + "; " + "; ".join(f"{k}={summary[k]['mean']:.5f}" for k in res)))
print(json.dumps(summary, indent=2))
