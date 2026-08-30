"""Wave-2 confirmation battery: ablations, negative controls, stability.

ONE protocol for every arm (`wave2_lib.ABL`: 5 canonical folds, 400,000 training
rows, champion hyperparameters), so every comparison is paired and valid.  The
row cap is 400k rather than the champion's 1M purely for compute: on two cores
the full battery at 1M rows is ~14 CPU-hours.  Absolute levels therefore sit
BELOW the champion's 0.61510 and must never be quoted as a champion score --
only the deltas within the battery are meaningful.

Sections
  lomo    leave-one-module-out over all 7 production modules + the all-modules
          control, identical folds/rows/seed/params
  nc      negative controls (label permutation, series-id shuffle, time-only,
          random feature block)
  seeds   seed stability of the full bank: 0, 1, 7, 42, 2026
  altf    alternative grouped fold partitions alt1/alt2/alt3
  fcount  cross-fitted feature-count reduction, selection inside each outer fold
"""
from __future__ import annotations

import json, sys, time
sys.path.insert(0, "/home/claude/sb/src")
sys.path.insert(0, "/home/claude/sb/research/scripts")
import numpy as np

from sbr.pipeline import run, Data, load_features, _stack, append_result
from sbr.metric import ts_auc_flat
from wave2_lib import ABL, FULL, alt_folds

SECTION = sys.argv[1] if len(sys.argv) > 1 else "all"


# ------------------------------------------------------------------- LOMO
def lomo():
    run(exp_id="RT-200", modules=FULL, agent="ablation",
        hypothesis="Battery control: all 7 modules at the battery protocol",
        falsification="n/a (control)", notes="BATTERY control, 400k rows", **ABL)
    for m in FULL:
        keep = [x for x in FULL if x != m]
        run(exp_id=f"RT-2{FULL.index(m)+1:02d}", modules=keep, agent="ablation",
            hypothesis=f"Removing {m} costs TS-AUC at full scale (identical folds/rows/seed/params)",
            falsification=f"delta vs RT-200 >= 0, i.e. {m} contributes nothing or hurts",
            notes=f"BATTERY leave-one-module-out: minus {m}", **ABL)


# --------------------------------------------------------- NEGATIVE CONTROLS
def negative_controls():
    """Four controls.  Any of them scoring materially above 0.5 is a red flag."""
    d = Data()
    out = {}

    # (1) label permutation at SERIES level: the model must land at ~0.5.
    #     Permuting whole series preserves the within-series label trajectory
    #     shape, which is the honest version of this control -- permuting rows
    #     would destroy the time structure and make the test trivially easy.
    rng = np.random.default_rng(0)
    perm = rng.permutation(d.st.n_series)
    n_on = d.st.meta.n_online.to_numpy()
    tau = d.st.meta.tau_index.to_numpy()
    y_perm = np.zeros_like(d.y)
    for i in range(d.st.n_series):
        s = d.st.orow_off[i]
        j = perm[i]
        tj = tau[j]
        if tj >= 0:
            rel = tj / max(n_on[j], 1)
            k = int(rel * n_on[i])
            y_perm[s + k: s + n_on[i]] = 1
    out["label_permutation"] = _train_eval(d, FULL, y=y_perm, tag="NC label permutation")

    # (2) time-only model: t and log t are the ONLY inputs.  Under TS-AUC, which
    #     compares within a timestep, a function of t alone is constant inside
    #     every comparison and must score exactly 0.5.
    out["time_only"] = _train_eval(d, FULL, keep_regex=r"::(log_)?t_online$", tag="NC time-only")

    # (3) random feature block: 60 columns of noise added to the full bank.
    out["random_block"] = _random_block(d)

    # (4) series-id / historical-context shuffle is covered by wave 1's m05_ctx
    #     derangement control and is not re-run here; see FAILED_EXPERIMENTS.md.
    json.dump(out, open("/home/claude/sb/research/reports/negative_controls.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


def _train_eval(d, modules, y=None, keep_regex=None, folds=(0, 1), tag=""):
    import lightgbm as lgb
    mats, names = load_features(modules)
    if keep_regex:
        import re
        rx = re.compile(keep_regex)
        keep = np.flatnonzero([bool(rx.search(n)) for n in names])
    else:
        keep = np.arange(len(names))
    yy = d.y if y is None else y
    p = dict(ABL["params"]); n_round = p.pop("n_estimators")
    p.update(objective="binary", num_threads=2, verbose=-1, bagging_freq=1)
    per = []
    rng = np.random.default_rng(0)
    for f in folds:
        tr = d.rows_for([x for x in (0, 1, 2, 3, 4) if x != f])
        va = d.rows_for([f])
        tr = np.sort(rng.choice(tr, min(ABL["max_train_rows"], len(tr)), replace=False))
        X = _stack(mats, names, tr, keep)
        b = lgb.train(p, lgb.Dataset(X, label=yy[tr], params=p), num_boost_round=n_round)
        del X
        Xv = _stack(mats, names, va, keep)
        pr = b.predict(Xv); del Xv
        # scored against the TRUE labels: a control that beats 0.5 here is real
        per.append(float(ts_auc_flat(pr, d.y[va], d.t[va])))
        print(f"  {tag} fold {f}: {per[-1]:.5f}", flush=True)
    return {"per_fold": per, "mean": float(np.mean(per)), "n_features": int(len(keep))}


def _random_block(d, folds=(0, 1)):
    import lightgbm as lgb
    mats, names = load_features(FULL)
    p = dict(ABL["params"]); n_round = p.pop("n_estimators")
    p.update(objective="binary", num_threads=2, verbose=-1, bagging_freq=1)
    rng = np.random.default_rng(0)
    per_with, per_without = [], []
    for f in folds:
        tr = d.rows_for([x for x in (0, 1, 2, 3, 4) if x != f])
        va = d.rows_for([f])
        tr = np.sort(rng.choice(tr, min(ABL["max_train_rows"], len(tr)), replace=False))
        keep = np.arange(len(names))
        X = _stack(mats, names, tr, keep); Xv = _stack(mats, names, va, keep)
        for label, extra in (("without", 0), ("with", 60)):
            if extra:
                g = np.random.default_rng(1000 + f)
                Xa = np.hstack([X, g.standard_normal((len(X), extra)).astype(np.float32)])
                Xva = np.hstack([Xv, g.standard_normal((len(Xv), extra)).astype(np.float32)])
            else:
                Xa, Xva = X, Xv
            b = lgb.train(p, lgb.Dataset(Xa, label=d.y[tr], params=p), num_boost_round=n_round)
            s = float(ts_auc_flat(b.predict(Xva), d.y[va], d.t[va]))
            (per_with if extra else per_without).append(s)
            print(f"  NC random-block {label} fold {f}: {s:.5f}", flush=True)
            if extra:
                del Xa, Xva
        del X, Xv
    return {"with_random_block": per_with, "without": per_without,
            "delta": float(np.mean(per_with) - np.mean(per_without))}


# ------------------------------------------------------------------- SEEDS
def seeds():
    for s in (1, 7, 42, 2026):
        cfg = dict(ABL); cfg["seed"] = s
        run(exp_id=f"RT-21{[1,7,42,2026].index(s)}", modules=FULL, agent="stability",
            hypothesis=f"The full-bank architecture is stable under seed {s}",
            falsification="seed-to-seed spread exceeds the fold-to-fold spread (0.0109)",
            notes=f"BATTERY seed stability, seed={s}", **cfg)


# ---------------------------------------------------------- ALTERNATIVE FOLDS
def altfolds():
    for name in ("alt1", "alt2", "alt3"):
        with alt_folds(name):
            run(exp_id=f"RT-22{name[-1]}", modules=FULL, agent="stability",
                hypothesis=f"RT-100's level survives an alternative grouped partition ({name})",
                falsification="mean TS-AUC on the alternative partition differs from the "
                              "canonical battery control by more than the fold spread",
                notes=f"BATTERY alternative fold partition {name} -- ROBUSTNESS ONLY, "
                      f"never used for selection", **ABL)


# ------------------------------------------------- CROSS-FITTED FEATURE COUNT
def fcount():
    """For every outer fold k: rank features using folds != k, refit on folds
    != k at each size, score fold k once.  No universal top-K list is ever
    chosen using all folds -- that is what made RT-140 a discovery number."""
    import lightgbm as lgb
    d = Data()
    mats, names = load_features(FULL)
    sizes = [500, 400, 350, 300, 250, 200]
    p0 = dict(ABL["params"]); n_round = p0.pop("n_estimators")
    p0.update(objective="binary", num_threads=2, verbose=-1, bagging_freq=1)
    res = {str(k): [] for k in sizes}
    rt = {str(k): [] for k in sizes}
    for f in (0, 1, 2, 3, 4):
        rng = np.random.default_rng(0)
        tr = d.rows_for([x for x in (0, 1, 2, 3, 4) if x != f])
        va = d.rows_for([f])
        tr = np.sort(rng.choice(tr, min(ABL["max_train_rows"], len(tr)), replace=False))
        X = _stack(mats, names, tr, np.arange(len(names)))
        # ranking model: folds != k ONLY
        rank_b = lgb.train(p0, lgb.Dataset(X, label=d.y[tr], params=p0), num_boost_round=n_round)
        order = np.argsort(-rank_b.feature_importance("gain"))
        Xv = _stack(mats, names, va, np.arange(len(names)))
        for k in sizes:
            sel = np.sort(order[:k])
            t0 = time.time()
            b = lgb.train(p0, lgb.Dataset(X[:, sel], label=d.y[tr], params=p0), num_boost_round=n_round)
            pr = b.predict(Xv[:, sel])
            s = float(ts_auc_flat(pr, d.y[va], d.t[va]))
            res[str(k)].append(s)
            rt[str(k)].append(time.time() - t0)
            print(f"  fcount fold {f} k={k}: {s:.5f}  ({time.time()-t0:.0f}s)", flush=True)
        del X, Xv
    out = {"per_fold": res, "mean": {k: float(np.mean(v)) for k, v in res.items()},
           "fit_predict_seconds": {k: float(np.mean(v)) for k, v in rt.items()},
           "protocol": "cross-fitted: ranking and refit use folds != k only"}
    json.dump(out, open("/home/claude/sb/research/reports/feature_count.json", "w"), indent=2)
    print(json.dumps(out["mean"], indent=2))


if __name__ == "__main__":
    {"lomo": lomo, "nc": negative_controls, "seeds": seeds,
     "altf": altfolds, "fcount": fcount}[SECTION]()
