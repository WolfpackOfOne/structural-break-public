"""Agent 09 -- master sweep: objective x target x row-weighting x model family.

One process, one seed, one set of materialised matrices -> every arm is a paired
control of every other arm by construction.  Validation predictions are saved so
that cross-family decorrelation can be measured afterwards.
"""
from __future__ import annotations
import argparse, os, sys, time, json
import numpy as np
import lightgbm as lgb

sys.path.insert(0, "/home/claude/sb/scripts")
from agent09_lib import Bench, BASE, log, make_pairwise_obj, make_focal_obj, _sig

PRED = "/home/claude/sb/research/oof/agent09"
os.makedirs(PRED, exist_ok=True)


def lgb_train(pack, params, n_round, label=None, weight=None, fobj=None, rows=None):
    p = dict(BASE); p.update(params)
    X = pack["Xtr"]; y = pack["ytr"] if label is None else label
    w = weight
    if rows is not None:
        X = X[rows]; y = y[rows]
        w = None if w is None else w[rows]
    ds = lgb.Dataset(X, label=y, weight=w, free_raw_data=False,
                     feature_name=[f"f{i}" for i in range(X.shape[1])])
    if fobj is not None:
        p["objective"] = fobj
    b = lgb.train(p, ds, num_boost_round=n_round)
    return b.predict(pack["Xva"]).astype(np.float64), b


TAG = "L1"


def run_arm(bench, name, cfg, folds, rounds, store):
    t0 = time.time(); per_fold = []; meta = {}
    for f in folds:
        sm = cfg.get("sample_mode", "uniform")
        pack = bench.get(f, sample_mode=sm)
        h = pack["ttr"] - pack["tautr"]          # steps since break (garbage if tau<0)
        isb = pack["tautr"] >= 0
        kind = cfg["kind"]
        if kind == "std":
            p = {k: v for k, v in cfg.items() if k not in ("kind", "desc", "sample_mode")}
            pred, _ = lgb_train(pack, p, rounds)
        elif kind == "weighted":
            w = cfg["wfun"](pack)
            pred, _ = lgb_train(pack, cfg.get("params", {}), rounds, weight=w)
            meta = {"w_mean": float(w.mean()), "w_max": float(w.max())}
        elif kind == "subset":
            r = cfg["rowfun"](pack)
            pred, _ = lgb_train(pack, cfg.get("params", {}), rounds, rows=r)
            meta = {"n_rows": int(r.sum() if r.dtype == bool else len(r))}
        elif kind == "target":
            lab = cfg["labfun"](pack, h, isb)
            pred, _ = lgb_train(pack, cfg.get("params", {}), rounds, label=lab)
            meta = {"lab_mean": float(np.mean(lab))}
        elif kind == "focal":
            pred, _ = lgb_train(pack, {}, rounds, fobj=make_focal_obj(cfg["alpha"], cfg["gamma"]))
        elif kind == "pairwise":
            key = pack["ttr"] if cfg.get("group", "t") == "t" else pack["str_"]
            fobj, meta = make_pairwise_obj(key, pack["ytr"], m_neg=cfg["m"], seed=0,
                                           resample_every=cfg.get("resample", 1))
            pred, _ = lgb_train(pack, cfg.get("params", {}), rounds, fobj=fobj)
        elif kind == "rank":
            p = {k: v for k, v in cfg.items() if k not in ("kind", "desc", "sample_mode")}
            o = np.argsort(pack["ttr"], kind="stable")
            X = np.ascontiguousarray(pack["Xtr"][o]); y = pack["ytr"][o].astype(int)
            grp = np.bincount(pack["ttr"][o]); grp = grp[grp > 0]
            pp = dict(BASE); pp.update(p)
            ds = lgb.Dataset(X, label=y, group=grp, free_raw_data=False,
                             feature_name=[f"f{i}" for i in range(X.shape[1])])
            b = lgb.train(pp, ds, num_boost_round=rounds)
            pred = b.predict(pack["Xva"]).astype(np.float64); del X, ds
            meta = {"n_groups": int(len(grp))}
        elif kind == "xgb":
            import xgboost as xgb
            dtr = xgb.DMatrix(pack["Xtr"], label=pack["ytr"], nthread=2)
            dva = xgb.DMatrix(pack["Xva"], nthread=2)
            p = dict(objective="binary:logistic", eta=BASE["learning_rate"], max_depth=7, min_child_weight=20,
                     subsample=0.7, colsample_bytree=0.7, reg_lambda=5.0, nthread=2,
                     tree_method="hist", max_bin=127, eval_metric="logloss")
            p.update(cfg.get("params", {}))
            b = xgb.train(p, dtr, num_boost_round=rounds)
            pred = b.predict(dva).astype(np.float64); del dtr, dva
        elif kind == "cat":
            from catboost import CatBoostClassifier, Pool
            p = dict(iterations=rounds, learning_rate=BASE["learning_rate"], depth=7, l2_leaf_reg=5.0,
                     rsm=0.7, thread_count=2, verbose=0, border_count=127,
                     bootstrap_type="Bernoulli", subsample=0.7, allow_writing_files=False)
            p.update(cfg.get("params", {}))
            m = CatBoostClassifier(**p)
            m.fit(pack["Xtr"], pack["ytr"].astype(int))
            pred = m.predict_proba(pack["Xva"])[:, 1].astype(np.float64)
        else:
            raise ValueError(kind)
        s = bench.score(pred, pack)
        per_fold.append(s)
        store.setdefault(name, {})[f] = pred.astype(np.float32)
        print(f"  [{name}] fold {f}: TS-AUC {s:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    rt = time.time() - t0
    log(f"RT-A09-{name}-{TAG}", bench, per_fold,
        hypothesis=cfg.get("hyp", "supervision variant beats binary logloss control"),
        falsification="does not beat the paired binary-logloss control by >0.002 TS-AUC",
        model=cfg.get("model", "lgbm"), objective=name,
        notes=f"agent9 :: {cfg['desc']} :: {meta}",
        sample_mode=cfg.get("sample_mode", "uniform"), runtime=rt)
    print(f"== {name:22s} {np.mean(per_fold):.5f}  [{' '.join(f'{x:.5f}' for x in per_fold)}]  {rt:.0f}s  {cfg['desc']}", flush=True)
    return per_fold


# ------------------------------------------------------------------ arm defs
def build_arms():
    A = {}
    A["OBJ-binary"] = dict(kind="std", objective="binary", desc="CONTROL binary logloss")
    A["OBJ-pairwise-t-m8"] = dict(kind="pairwise", m=8, group="t",
                                  desc="pairwise logistic, groups = online index t, 8 neg/pos")
    A["OBJ-pairwise-t-m2"] = dict(kind="pairwise", m=2, group="t",
                                  desc="pairwise logistic, groups = t, 2 neg/pos")
    A["OBJ-pairwise-series"] = dict(kind="pairwise", m=8, group="series",
                                    desc="NEGATIVE CONTROL pairwise grouped by series")
    A["OBJ-lambdarank"] = dict(kind="rank", objective="lambdarank", label_gain=[0, 1],
                               lambdarank_truncation_level=2000,
                               desc="lambdarank, group = online index t, trunc 2000")
    A["OBJ-xendcg"] = dict(kind="rank", objective="rank_xendcg", label_gain=[0, 1],
                           desc="XE-NDCG, group = online index t")
    A["OBJ-focal-g2"] = dict(kind="focal", gamma=2.0, alpha=0.5, desc="focal loss gamma=2")
    A["OBJ-spw4"] = dict(kind="std", objective="binary", scale_pos_weight=4.0,
                         desc="binary, scale_pos_weight=4")
    A["OBJ-xentropy"] = dict(kind="std", objective="cross_entropy",
                             desc="cross-entropy objective")

    # ---- targets
    A["TGT-ramp20"] = dict(kind="target", params=dict(objective="cross_entropy"),
                           labfun=lambda p, h, b: np.where(b & (h >= 0), np.clip(h / 20.0, 0, 1), 0.0),
                           desc="soft ramp target min(h/20,1), cross-entropy")
    A["TGT-loghz"] = dict(kind="target", params=dict(objective="regression"),
                          labfun=lambda p, h, b: np.where(b & (h >= 0), np.log1p(np.maximum(h, 0)), 0.0),
                          desc="log1p(time-since-break) regression, 0 for pre-break/no-break")
    A["TGT-hazard"] = dict(kind="target", params=dict(objective="binary"),
                           labfun=lambda p, h, b: (b & (h == 0)).astype(float),
                           desc="hazard target 1[t==tau] (break-instant detector)")

    # ---- row weighting / sampling
    A["W-persseries"] = dict(kind="std", objective="binary", sample_mode="per_series",
                             desc="per-series equal sampling (binary)")
    A["W-pairprop"] = dict(kind="weighted", objective="binary",
                           wfun=lambda p: _pairw(p), desc="row weight = metric pair count at t")
    A["W-nearbreak"] = dict(kind="weighted", objective="binary",
                            wfun=lambda p: _nearw(p), desc="ramp-down weight on positives with h<10")
    A["W-tpairsamp"] = dict(kind="std", objective="binary", sample_mode="t_pairprop",
                            desc="row SAMPLING proportional to metric pair count")

    # ---- model family
    A["FAM-xgb"] = dict(kind="xgb", model="xgboost", desc="XGBoost hist binary:logistic")
    A["FAM-cat"] = dict(kind="cat", model="catboost", desc="CatBoost Logloss")
    return A


def _pairw(p):
    t = p["ttr"]; y = p["ytr"].astype(np.int64)
    nb = int(t.max()) + 1
    npos = np.bincount(t, weights=y, minlength=nb)
    ntot = np.bincount(t, minlength=nb)
    nneg = ntot - npos
    w = np.where(y == 1, nneg[t], npos[t]).astype(np.float64)
    w = np.maximum(w, 1e-6)
    return w / w.mean()


def _nearw(p):
    h = p["ttr"] - p["tautr"]; isb = p["tautr"] >= 0
    w = np.ones(len(h))
    m = isb & (h >= 0)
    w[m] = np.clip((h[m] + 1) / 10.0, 0.05, 1.0)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="all")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--modules", default="m00_core,m03_dyn")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--out", default="sweepA")
    a = ap.parse_args()
    folds = tuple(int(x) for x in a.folds.split(","))
    BASE["learning_rate"] = a.lr
    bench = Bench(a.modules.split(","), folds=folds, max_train_rows=300_000, seed=0, screen=True)
    global TAG
    TAG = a.out
    A = build_arms()
    want = list(A) if a.arms == "all" else a.arms.split(",")
    store, res = {}, {}
    for n in want:
        try:
            res[n] = run_arm(bench, n, A[n], folds, a.rounds, store)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"!! {n} FAILED: {e}", flush=True)
        np.savez_compressed(f"{PRED}/{a.out}.npz",
                            **{f"{k}|f{f}": v for k, d in store.items() for f, v in d.items()})
        json.dump({k: list(map(float, v)) for k, v in res.items()},
                  open(f"{PRED}/{a.out}.json", "w"), indent=1)
    print("\n=== RANKED ===")
    for n, pf in sorted(res.items(), key=lambda z: -np.mean(z[1])):
        print(f"{np.mean(pf):.5f}  {n:22s} [{' '.join(f'{x:.5f}' for x in pf)}]  {A[n]['desc']}")


if __name__ == "__main__":
    main()
