#!/usr/bin/env python
"""RT-1321 -- LS-KD lag-space characteristic-kernel discrepancy.

Preregistration: research/reports/rt1321_lskd/RT1321_PREREG.md (frozen before
any label was evaluated).

Stages
    screen    Stage 1: standalone information screen of the candidate block
              (`m19_lskd`, joint lag law) against its mechanism-matched control
              (`m19_lskm`, coordinate-separable / marginal), anchored on the
              frozen RT-1320 eight-member blend, canonical fold 0, plus the
              fixed-perturbation pair-flow grid.
    member    Stage 2: train the matched ninth member on {500 bank columns +
              one kernel block} and write its OOF vector.
    contract  Stage 2 endpoint: E0 = RT-1320, E1 = +control ninth member,
              E2 = +candidate ninth member.  Primary endpoint E2-E1.

Nothing here edits RT-1320's stored artifacts; the incumbent is loaded read-only
from the frozen OOF vectors and rebuilt with the addition contract's own
calibration and blending so the baseline is bit-for-bit the recorded one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("SBR_ROOT", Path(__file__).resolve().parents[2]))
REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "research" / "scripts", ROOT / "src",
          ROOT / "research" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import armc_residual_student as A  # noqa: E402
from sbr.metric import ts_auc_flat  # noqa: E402

FOLDS = (0, 1, 2, 3, 4)
BANK = ("m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes")
CANDIDATE_BLOCK = "m19_lskd"
CONTROL_BLOCK = "m19_lskm"
OUT = ROOT / "research" / "reports" / "rt1321_lskd"
OOFDIR = ROOT / "research" / "oof"
FEATDIR = ROOT / "cache" / "features"
STUDENT_DIR = "research/reports/armc_residual_student_confirm_s20260901"

#: The repository's canonical production pairwise_t specialist configuration
#: (`research/scripts/wave2_streams.py` RT-123R, the RT-413 slot).  Frozen in
#: the preregistration.  E1 and E2 differ ONLY in which kernel block is stacked
#: onto the 500-column bank.
MEMBER_PARAMS = {
    "objective": "pairwise_t",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "max_bin": 127,
    "num_threads": 8,
    "verbose": -1,
}
MEMBER_ROUNDS = 600
MEMBER_MAX_ROWS = 700_000
MEMBER_SEED = 0

#: Stage-1 standalone screen: the same learner on the kernel block ALONE, so the
#: number measures the block's own information and nothing else.
SCREEN_PARAMS = dict(MEMBER_PARAMS)
SCREEN_ROUNDS = 400
SCREEN_MAX_ROWS = 400_000

#: Preregistered perturbation grid.  Reported in full; never selected after.
EPSILONS = (0.01, 0.02, 0.04)
PAIRS_PER_T = 20
PAIR_SEED = 0


# --------------------------------------------------------------------- loading
def student_oof_dir() -> Path:
    for base in (ROOT, REPO, REPO.parent / "structural-break-multi-agent-frontier-20260829"):
        p = base / STUDENT_DIR
        if (p / "armc_residual_student_oof.npy").exists():
            return p
    raise SystemExit("cannot locate the RT-1320 student OOF; looked under "
                     f"{[str(b / STUDENT_DIR) for b in (ROOT, REPO)]}")


def rt1320_members(rows) -> list[np.ndarray]:
    """The eight frozen RT-1320 members, calibrated exactly as the contract does."""
    oof_dir = A.default_data_root() / "research" / "oof"
    cat_dir = A.default_catboost_oof_dir()
    if cat_dir is None:
        raise SystemExit("cannot locate the CatBoost OOF vectors RT-1254 / RT-1255")
    spec = [A.load_calibrated_stream(oof_dir, s, rows) for s in A.SPECIALISTS]
    cat300 = A.crossfit_calibrate(np.load(cat_dir / "RT-1255.npy", mmap_mode="r"), rows)
    cat413 = A.crossfit_calibrate(np.load(cat_dir / "RT-1254.npy", mmap_mode="r"), rows)
    student_raw, _ = A.merge_student_oof(student_oof_dir(), rows, force=False)
    student = A.crossfit_calibrate(student_raw, rows)
    return [cat300, spec[1], spec[2], spec[3], cat413, spec[5], spec[6], student]


def load_block(modules):
    mats, names = [], []
    for m in modules:
        meta = json.loads((FEATDIR / f"{m}.cols.json").read_text())
        mats.append(np.load(FEATDIR / f"{m}.npy", mmap_mode="r"))
        names += [f"{m}::{c}" for c in meta["cols"]]
    return mats, names


# ---------------------------------------------------------------------- masks
def diagnostic_masks(rows):
    """Cuts used for REPORTING ONLY.  tau never reaches a feature or a model."""
    y, t, age, dev = rows["y"], rows["t"], rows["age"], rows["dev"]
    hb = rows["has_break_by_row"]
    dom = rows["dominant_cell"]
    return {
        "whole_dev": dev,
        "dominant_cell": dom,
        "dominant_never_break_only": dom & ((y == 1) | ~hb),
        "dominant_pre_break_only": dom & ((y == 1) | hb),
    }


def auc_on(vec, rows, mask, fold=None):
    m = mask.copy()
    if fold is not None:
        m &= rows["row_fold"] == fold
    r = np.flatnonzero(m & np.isfinite(vec))
    if len(r) == 0:
        return float("nan")
    return float(ts_auc_flat(np.asarray(vec)[r], rows["y"][r], rows["t"][r]))


# ------------------------------------------------------------------- training
def train_block_fold0(block: str, rows) -> np.ndarray:
    """Fold-0 standalone score of ONE kernel block, trained on folds 1-4."""
    import lightgbm as lgb
    from sbr.pipeline import _stack

    mats, names = load_block([block])
    keep = np.arange(len(names), dtype=np.int64)
    row_fold = rows["row_fold"]
    tr = A.rows_for(row_fold, [1, 2, 3, 4])
    va = A.rows_for(row_fold, [0])
    rng = np.random.default_rng(MEMBER_SEED)
    if len(tr) > SCREEN_MAX_ROWS:
        tr = np.sort(rng.choice(tr, SCREEN_MAX_ROWS, replace=False))

    Xtr = _stack(mats, names, tr, keep)
    ytr = rows["y"][tr]
    p = dict(SCREEN_PARAMS)
    p["objective"] = A_pairwise(rows["t"][tr], ytr, MEMBER_SEED)
    ds = lgb.Dataset(Xtr, label=ytr, params=dict(p, objective="binary"),
                     feature_name=[f"f{i}" for i in range(len(keep))])
    bst = lgb.train(p, ds, num_boost_round=SCREEN_ROUNDS)
    del Xtr, ds
    out = np.full(len(rows["y"]), np.nan, dtype=np.float64)
    out[va] = bst.predict(_stack(mats, names, va, keep))
    return out


def A_pairwise(t, y, seed):
    from sbr.pipeline import _make_pairwise_t
    return _make_pairwise_t(t, y, seed=seed)


def train_member(block: str, rows, folds=FOLDS, out_path: Path | None = None):
    """The matched ninth member: 500 bank columns + one kernel block."""
    import lightgbm as lgb
    from sbr.pipeline import _stack

    modules = list(BANK) + [block]
    mats, names = load_block(modules)
    if len(names) != 528:
        raise SystemExit(f"expected 500 bank + 28 kernel columns, got {len(names)}")
    keep = np.arange(len(names), dtype=np.int64)
    row_fold = rows["row_fold"]
    oof = np.full(len(rows["y"]), np.nan, dtype=np.float32)
    per_fold = {}
    t0 = time.time()
    for f in folds:
        rng = np.random.default_rng(MEMBER_SEED)      # identical row draw in E1/E2
        tr = A.rows_for(row_fold, [x for x in FOLDS if x != f])
        va = A.rows_for(row_fold, [f])
        if len(tr) > MEMBER_MAX_ROWS:
            tr = np.sort(rng.choice(tr, MEMBER_MAX_ROWS, replace=False))
        Xtr = _stack(mats, names, tr, keep)
        ytr = rows["y"][tr]
        p = dict(MEMBER_PARAMS)
        p["objective"] = A_pairwise(rows["t"][tr], ytr, MEMBER_SEED)
        ds = lgb.Dataset(Xtr, label=ytr, params=dict(p, objective="binary"),
                         feature_name=[f"f{i}" for i in range(len(keep))])
        bst = lgb.train(p, ds, num_boost_round=MEMBER_ROUNDS)
        del Xtr, ds
        pred = bst.predict(_stack(mats, names, va, keep)).astype(np.float32)
        oof[va] = pred
        per_fold[str(f)] = float(ts_auc_flat(pred, rows["y"][va], rows["t"][va]))
        print(f"  {block} fold {f}: standalone TS-AUC {per_fold[str(f)]:.6f} "
              f"({len(tr)} train rows, {time.time()-t0:.0f}s)", flush=True)
    if out_path is not None:
        np.save(out_path, oof)
    return oof, {"per_fold_standalone_ts_auc": per_fold, "runtime_s": round(time.time() - t0, 1)}


# ------------------------------------------------------------------- reporting
def perturbation_grid(base, cand, rows, masks, fold=0):
    """RT-1320's within-t rank plus epsilon * (block's within-t rank - 0.5).

    An information diagnostic, NOT the production ensemble.  The full
    preregistered grid is reported; no epsilon is chosen afterwards.
    """
    dev_fold = np.flatnonzero((rows["row_fold"] == fold) & np.isfinite(base) & np.isfinite(cand))
    rb = A.within_t_rank_vector(base, dev_fold, rows["t"])
    rc = A.within_t_rank_vector(cand, dev_fold, rows["t"])
    out = {}
    for eps in EPSILONS:
        pert = rb + eps * (rc - 0.5)
        row = {}
        for name, mask in masks.items():
            rr = np.flatnonzero(mask & (rows["row_fold"] == fold)
                                & np.isfinite(rb) & np.isfinite(pert))
            st = A.pair_repair_stats(rb, pert, rows["y"], rows["t"], rr,
                                     n_pairs_per_t=PAIRS_PER_T, seed=PAIR_SEED)
            row[name] = {"repairs": st["repairs"], "damage": st["damage"],
                         "net": st["net_pair_lift"], "net_rate": st["net_rate"],
                         "ts_auc_base": auc_on(rb, rows, mask, fold),
                         "ts_auc_perturbed": auc_on(pert, rows, mask, fold)}
        out[f"eps_{eps}"] = row
    return out


def cmd_screen(args):
    rows = A.build_row_arrays(ROOT / "research" / "folds" / "folds.parquet")
    masks = diagnostic_masks(rows)
    base = A.blend(rt1320_members(rows))
    print(f"E0 RT-1320 rebuilt: mean fold TS-AUC "
          f"{A.mean_fold_ts_auc(base, rows)['mean_ts_auc']:.9f}", flush=True)

    scores, report = {}, {"experiment_id": "RT-1321", "stage": "1_information_screen",
                          "fold": 0, "epsilons": list(EPSILONS)}
    for label, block in (("candidate_joint", CANDIDATE_BLOCK),
                         ("control_marginal", CONTROL_BLOCK)):
        t0 = time.time()
        v = train_block_fold0(block, rows)
        np.save(OOFDIR / f"RT-1321_screen_{label}.npy", v)
        scores[label] = v
        print(f"{label} ({block}) trained in {time.time()-t0:.0f}s", flush=True)

    fold0 = rows["row_fold"] == 0
    report["standalone"] = {
        label: {name: auc_on(v, rows, mask, 0) for name, mask in masks.items()}
        for label, v in scores.items()}
    report["rt1320_reference"] = {name: auc_on(base, rows, mask, 0) for name, mask in masks.items()}

    dev0 = np.flatnonzero(fold0 & np.isfinite(base))
    rb = A.within_t_rank_vector(base, dev0, rows["t"])
    report["within_t_rank_rho_vs_rt1320"] = {}
    for label, v in scores.items():
        rv = A.within_t_rank_vector(v, dev0, rows["t"])
        report["within_t_rank_rho_vs_rt1320"][label] = {
            name: A.pearson(rb, rv, mask & fold0) for name, mask in masks.items()}
    rc = A.within_t_rank_vector(scores["candidate_joint"], dev0, rows["t"])
    rm = A.within_t_rank_vector(scores["control_marginal"], dev0, rows["t"])
    report["candidate_vs_control_rho"] = {
        name: A.pearson(rc, rm, mask & fold0) for name, mask in masks.items()}

    report["perturbation_grid"] = {
        label: perturbation_grid(base, v, rows, masks, fold=0) for label, v in scores.items()}

    # conditional performance on the pairs RT-1320 gets wrong
    report["hard_pair_conditional"] = hard_pair_conditional(base, scores, rows, masks)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "STAGE1_SCREEN.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def hard_pair_conditional(base, scores, rows, masks, fold=0, n_pairs_per_t=20, seed=0):
    """AUC of each block restricted to the same-t pairs RT-1320 inverts."""
    rng = np.random.default_rng(seed)
    y, t = rows["y"], rows["t"]
    out = {}
    for name, mask in masks.items():
        rr = np.flatnonzero(mask & (rows["row_fold"] == fold) & np.isfinite(base))
        order = np.argsort(t[rr], kind="stable")
        rr = rr[order]
        tt = t[rr]
        st = np.flatnonzero(np.r_[True, tt[1:] != tt[:-1]])
        en = np.r_[st[1:], len(rr)]
        pos_l, neg_l = [], []
        for lo, hi in zip(st, en):
            idx = rr[lo:hi]
            p = idx[y[idx] == 1]
            n = idx[y[idx] == 0]
            if len(p) == 0 or len(n) == 0:
                continue
            k = min(n_pairs_per_t, len(p), len(n))
            pp = rng.choice(p, k, replace=False)
            nn = rng.choice(n, k, replace=False)
            wrong = base[pp] <= base[nn]
            pos_l.append(pp[wrong])
            neg_l.append(nn[wrong])
        if not pos_l:
            continue
        pp = np.concatenate(pos_l)
        nn = np.concatenate(neg_l)
        row = {"n_inverted_pairs": int(len(pp))}
        for label, v in scores.items():
            ok = np.isfinite(v[pp]) & np.isfinite(v[nn])
            row[label] = float(np.mean((v[pp][ok] > v[nn][ok]).astype(np.float64)
                                       + 0.5 * (v[pp][ok] == v[nn][ok]))) if ok.any() else float("nan")
        out[name] = row
    return out


def cmd_member(args):
    rows = A.build_row_arrays(ROOT / "research" / "folds" / "folds.parquet")
    block = {"candidate": CANDIDATE_BLOCK, "control": CONTROL_BLOCK}[args.arm]
    exp = f"RT-1321_member_{args.arm}"
    folds = tuple(int(x) for x in args.folds.split(",")) if args.folds else FOLDS
    _, meta = train_member(block, rows, folds=folds, out_path=OOFDIR / f"{exp}.npy")
    meta.update({"arm": args.arm, "block": block, "folds": list(folds),
                 "params": {k: v for k, v in MEMBER_PARAMS.items()},
                 "n_estimators": MEMBER_ROUNDS, "max_train_rows": MEMBER_MAX_ROWS,
                 "seed": MEMBER_SEED})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"MEMBER_{args.arm}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


def cmd_contract(args):
    rows = A.build_row_arrays(ROOT / "research" / "folds" / "folds.parquet")
    masks = diagnostic_masks(rows)
    members = rt1320_members(rows)
    folds = tuple(int(x) for x in args.folds.split(",")) if args.folds else FOLDS

    ctrl = A.crossfit_calibrate(np.load(OOFDIR / "RT-1321_member_control.npy", mmap_mode="r"), rows)
    cand = A.crossfit_calibrate(np.load(OOFDIR / "RT-1321_member_candidate.npy", mmap_mode="r"), rows)
    packs = {"E0": A.blend(members),
             "E1": A.blend(members + [ctrl]),
             "E2": A.blend(members + [cand])}
    res = {"experiment_id": "RT-1321", "stage": "2_ninth_member_contract",
           "contract": "addition of a 9th exchangeable member to RT-1320",
           "E1_control": "coordinate-separable marginal kernel block m19_lskm",
           "E2_candidate": "joint lag-space kernel block m19_lskd",
           "folds_evaluated": list(folds)}
    pf = {}
    for k, v in packs.items():
        pf[k] = {str(f): auc_on(v, rows, masks["whole_dev"], f) for f in folds}
        res[f"{k}_per_fold"] = pf[k]
        res[f"{k}_mean"] = float(np.mean([pf[k][str(f)] for f in folds]))
    res["PRIMARY_E2_minus_E1_per_fold"] = {str(f): pf["E2"][str(f)] - pf["E1"][str(f)] for f in folds}
    res["PRIMARY_E2_minus_E1"] = float(np.mean(list(res["PRIMARY_E2_minus_E1_per_fold"].values())))
    res["primary_folds_positive"] = int(sum(x > 0 for x in res["PRIMARY_E2_minus_E1_per_fold"].values()))
    res["SECONDARY_E2_minus_E0_per_fold"] = {str(f): pf["E2"][str(f)] - pf["E0"][str(f)] for f in folds}
    res["SECONDARY_E2_minus_E0"] = float(np.mean(list(res["SECONDARY_E2_minus_E0_per_fold"].values())))
    res["control_lift_E1_minus_E0"] = float(np.mean(
        [pf["E1"][str(f)] - pf["E0"][str(f)] for f in folds]))

    fold_mask = np.isin(rows["row_fold"], np.asarray(folds))
    res["pair_flow_E2_vs_E1"] = {}
    res["pair_flow_E2_vs_E0"] = {}
    for name, mask in masks.items():
        rr = np.flatnonzero(mask & fold_mask)
        res["pair_flow_E2_vs_E1"][name] = A.pair_repair_stats(
            packs["E1"], packs["E2"], rows["y"], rows["t"], rr,
            n_pairs_per_t=PAIRS_PER_T, seed=PAIR_SEED)
        res["pair_flow_E2_vs_E0"][name] = A.pair_repair_stats(
            packs["E0"], packs["E2"], rows["y"], rows["t"], rr,
            n_pairs_per_t=PAIRS_PER_T, seed=PAIR_SEED)
    res["cell_ts_auc"] = {k: {n: auc_on(v, rows, m) for n, m in masks.items()}
                          for k, v in packs.items()}
    if len(folds) == 5:
        res["paired_series_bootstrap_E2_minus_E1"] = paired_bootstrap(
            packs["E1"], packs["E2"], rows)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "" if len(folds) == 5 else f"_fold{'-'.join(map(str, folds))}"
    (OUT / f"CONTRACT{suffix}.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


def paired_bootstrap(e1, e2, rows, n_boot=400, seed=1321):
    """Series-level paired bootstrap of the mean-fold TS-AUC difference."""
    rng = np.random.default_rng(seed)
    folds_df = rows["folds"]
    sidx = rows["sidx"]
    row_fold = rows["row_fold"]
    y, t = rows["y"], rows["t"]
    deltas = []
    per_fold_series = {f: np.flatnonzero(folds_df["fold"].to_numpy() == f) for f in FOLDS}
    row_start = np.r_[0, np.cumsum(folds_df["n_online"].to_numpy())[:-1]]
    n_on = folds_df["n_online"].to_numpy()
    for _ in range(n_boot):
        per = []
        for f in FOLDS:
            s = per_fold_series[f]
            pick = rng.choice(s, len(s), replace=True)
            rr = np.concatenate([np.arange(row_start[i], row_start[i] + n_on[i]) for i in pick])
            per.append(float(ts_auc_flat(e2[rr], y[rr], t[rr]))
                       - float(ts_auc_flat(e1[rr], y[rr], t[rr])))
        deltas.append(float(np.mean(per)))
    d = np.asarray(deltas)
    return {"n_boot": n_boot, "mean": float(d.mean()),
            "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            "fraction_positive": float((d > 0).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("screen").set_defaults(fn=cmd_screen)
    m = sub.add_parser("member")
    m.add_argument("--arm", choices=("candidate", "control"), required=True)
    m.add_argument("--folds", default="")
    m.set_defaults(fn=cmd_member)
    c = sub.add_parser("contract")
    c.add_argument("--folds", default="")
    c.set_defaults(fn=cmd_contract)
    args = ap.parse_args()
    OOFDIR.mkdir(parents=True, exist_ok=True)
    args.fn(args)


if __name__ == "__main__":
    main()
