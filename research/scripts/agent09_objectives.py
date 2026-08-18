"""Agent 09 -- experiment 1: THE OBJECTIVE.  Paired controls, identical seed,
identical (Xtr, Xva) matrices for every arm."""
from __future__ import annotations
import argparse, json, sys, time
import numpy as np
import lightgbm as lgb

sys.path.insert(0, "/home/claude/sb/scripts")
from agent09_lib import Bench, BASE, log, make_pairwise_obj, make_focal_obj, _sig

MODULES = ["m00_core", "m03_dyn"]


def train_lgb(pack, params, n_round, label=None, weight=None, fobj=None, init_score=None):
    p = dict(BASE); p.update(params)
    y = pack["ytr"] if label is None else label
    ds = lgb.Dataset(pack["Xtr"], label=y, weight=weight, free_raw_data=False,
                     feature_name=[f"f{i}" for i in range(pack["Xtr"].shape[1])])
    if fobj is not None:
        p["objective"] = fobj
    b = lgb.train(p, ds, num_boost_round=n_round)
    return b.predict(pack["Xva"]).astype(np.float64), b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="all")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--modules", default=",".join(MODULES))
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    folds = tuple(int(x) for x in a.folds.split(","))
    mods = a.modules.split(",")
    bench = Bench(mods, folds=folds, max_train_rows=300_000, seed=0, screen=True)

    arms = {}

    # ---- 1. binary logloss (CONTROL)
    arms["binary"] = dict(objective="binary", kind="std", desc="binary logloss (control)")
    # ---- 2. within-t pairwise logistic (RankNet), groups = online index t
    for m in (4, 8, 16):
        arms[f"pairwise_m{m}"] = dict(kind="pairwise", m=m,
                                      desc=f"pairwise logistic, group=t, {m} negs/pos")
    # ---- 2b. LightGBM lambdarank / xendcg with group = online index t
    arms["lambdarank"] = dict(kind="rank", objective="lambdarank",
                              lambdarank_truncation_level=2000, label_gain=[0, 1],
                              desc="lambdarank group=t trunc=2000")
    arms["lambdarank_t30"] = dict(kind="rank", objective="lambdarank",
                                  lambdarank_truncation_level=30, label_gain=[0, 1],
                                  desc="lambdarank group=t trunc=30")
    arms["rank_xendcg"] = dict(kind="rank", objective="rank_xendcg", label_gain=[0, 1],
                               desc="XE-NDCG group=t")
    # ---- 2c. pairwise but grouped by SERIES (deliberate wrong grouping, negative control)
    arms["pairwise_series"] = dict(kind="pairwise_series", m=8,
                                   desc="pairwise logistic grouped by SERIES (wrong grouping)")
    # ---- 3. class weighting / focal
    for spw in (0.5, 2.0, 4.0):
        arms[f"spw{spw}"] = dict(kind="std", objective="binary", scale_pos_weight=spw,
                                 desc=f"binary, scale_pos_weight={spw}")
    for g in (1.0, 2.0):
        arms[f"focal_g{g}"] = dict(kind="focal", gamma=g, alpha=0.5,
                                   desc=f"focal loss gamma={g}")
    # ---- 4. rank-transformed / continuous targets on the SAME binary information
    arms["xentropy"] = dict(kind="std", objective="cross_entropy",
                            desc="cross-entropy objective on binary target")
    arms["l2"] = dict(kind="std", objective="regression",
                      desc="plain L2 regression on 0/1 target")

    want = list(arms) if a.arms == "all" else a.arms.split(",")

    out = []
    for name in want:
        cfg = arms[name]
        t0 = time.time()
        per_fold, meta = [], {}
        for f in folds:
            pack = bench.get(f)
            kind = cfg["kind"]
            if kind == "std":
                p = {k: v for k, v in cfg.items() if k not in ("kind", "desc")}
                pred, _ = train_lgb(pack, p, a.rounds)
            elif kind == "focal":
                fobj = make_focal_obj(cfg["alpha"], cfg["gamma"])
                pred, _ = train_lgb(pack, {}, a.rounds, fobj=fobj)
            elif kind == "pairwise":
                fobj, meta = make_pairwise_obj(pack["ttr"], pack["ytr"], m_neg=cfg["m"], seed=0)
                pred, _ = train_lgb(pack, {}, a.rounds, fobj=fobj)
            elif kind == "pairwise_series":
                fobj, meta = make_pairwise_obj(pack["str_"], pack["ytr"], m_neg=cfg["m"], seed=0)
                pred, _ = train_lgb(pack, {}, a.rounds, fobj=fobj)
            elif kind == "rank":
                p = {k: v for k, v in cfg.items() if k not in ("kind", "desc")}
                # lambdarank needs contiguous groups
                o = np.argsort(pack["ttr"], kind="stable")
                X = pack["Xtr"][o]; y = pack["ytr"][o].astype(int); ts = pack["ttr"][o]
                grp = np.bincount(ts)
                grp = grp[grp > 0]
                pp = dict(BASE); pp.update(p)
                ds = lgb.Dataset(X, label=y, group=grp, free_raw_data=False,
                                 feature_name=[f"f{i}" for i in range(X.shape[1])])
                b = lgb.train(pp, ds, num_boost_round=a.rounds)
                pred = b.predict(pack["Xva"]).astype(np.float64)
                del X
            else:
                raise ValueError(kind)
            s = bench.score(pred, pack)
            per_fold.append(s)
            print(f"  [{name}] fold {f}: TS-AUC {s:.5f}", flush=True)
        rt = time.time() - t0
        eid = f"RT-A09-OBJ-{name}{a.tag}"
        log(eid, bench, per_fold, hypothesis="within-t ranking supervision beats binary logloss",
            falsification="no objective beats binary logloss control by >0.002 TS-AUC on fold 0",
            model="lgbm", objective=name, notes=f"agent9 objective sweep :: {cfg['desc']} :: {meta}",
            sample_mode="uniform", runtime=rt)
        out.append((name, np.mean(per_fold), per_fold, rt, cfg["desc"]))
        print(f"== {name:20s} {np.mean(per_fold):.5f}  [{' '.join(f'{x:.5f}' for x in per_fold)}]  {rt:.0f}s", flush=True)

    print("\n=== RANKED ===")
    for name, m, pf, rt, desc in sorted(out, key=lambda z: -z[1]):
        print(f"{m:.5f}  {name:20s} {rt:6.0f}s  {desc}")


if __name__ == "__main__":
    main()
