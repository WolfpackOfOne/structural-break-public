#!/usr/bin/env python
"""CatBoost Specialist Activation 2026.

Preregistered full canonical-fold test of whether the frozen RT-1251 CatBoost
learner becomes more useful when applied to selected RT600 specialist setups.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = REPO.parent / "structural-break-learner-diversity-2026"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-root", default=os.environ.get("SBR_ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT)))
    p.add_argument("--no-ledger", action="store_true")
    p.add_argument("--force-train", action="store_true")
    return p.parse_args()


ARGS = parse_args()
ARTIFACT_ROOT = Path(ARGS.artifact_root).resolve()
os.environ["SBR_ROOT"] = str(ARTIFACT_ROOT)

for _p in (REPO / "src", REPO / "research" / "scripts"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from sbr.metric import ts_auc_flat  # noqa: E402
from sbr.pipeline import Data, _stack, load_features  # noqa: E402
from wave4_cal import SCDF_NSEEN  # noqa: E402

REPORT_DIR = REPO / "research" / "reports" / "catboost_specialist_2026"
RESULTS_CSV = REPO / "research" / "RESULTS.csv"
OOF_DIR = ARTIFACT_ROOT / "research" / "oof"

FOLDS = (0, 1, 2, 3, 4)
SPECIALISTS = ["RT-300", "RT-410", "RT-411", "RT-412", "RT-413", "RT-414", "RT-415"]
SEED_CLONES = ["RT-401", "RT-402", "RT-403", "RT-404", "RT-405", "RT-406"]
MATCHED_CLONE = "RT-401"
PAIR_SEED = 20260827
PAIRS_PER_T = 64

IDS = {
    "CAT-413": "RT-1254",
    "CAT-300": "RT-1255",
    "CAT-410": "RT-1256",
    "HYBRID": "RT-1257",
}

FULL = ["m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes"]
SPECS = {
    "CAT-413": {
        "replaced": "RT-413",
        "modules": FULL,
        "seed": 0,
        "max_train_rows": 700_000,
        "sample_mode": "uniform",
        "incumbent_objective": "pairwise_t",
    },
    "CAT-300": {
        "replaced": "RT-300",
        "modules": FULL,
        "seed": 0,
        "max_train_rows": 1_000_000,
        "sample_mode": "uniform",
        "incumbent_objective": "binary",
    },
    "CAT-410": {
        "replaced": "RT-410",
        "modules": ["m00_core", "m01_seq", "m07_bayes"],
        "seed": 0,
        "max_train_rows": 900_000,
        "sample_mode": "uniform",
        "incumbent_objective": "binary",
    },
}

CATBOOST_BASE_PARAMS = {
    "iterations": 600,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "bootstrap_type": "Bernoulli",
    "subsample": 0.7,
    "rsm": 0.7,
    "border_count": 127,
    "thread_count": 2,
    "allow_writing_files": False,
    "verbose": False,
}


@dataclass
class Ctx:
    d: Data
    rows: dict[int, np.ndarray]
    dev: np.ndarray
    tau: np.ndarray
    has_break: np.ndarray
    age: np.ndarray


def git_sha() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()


def env_versions() -> dict[str, str]:
    import importlib.metadata as md

    pkgs = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "catboost"]
    out = {}
    for p in pkgs:
        try:
            out[p] = md.version(p)
        except Exception:
            out[p] = "missing"
    return out


def make_ctx() -> Ctx:
    import pandas as pd

    d = Data()
    rows = {k: d.rows_for([k]) for k in FOLDS}
    dev = d.rows_for(list(FOLDS))
    meta = pd.read_parquet(ARTIFACT_ROOT / "cache" / "store" / "meta.parquet")
    tau = meta.tau_index.to_numpy()
    has_break = meta.has_break.to_numpy().astype(bool)
    age = np.where(d.y == 1, d.t - tau[d.sidx], -1)
    return Ctx(d=d, rows=rows, dev=dev, tau=tau, has_break=has_break, age=age)


def load_oof(names: list[str]) -> dict[str, np.ndarray]:
    missing = [n for n in names if not (OOF_DIR / f"{n}.npy").exists()]
    if missing:
        raise SystemExit(f"MISSING OOF: {missing} in {OOF_DIR}")
    return {n: np.load(OOF_DIR / f"{n}.npy") for n in names}


class CalibratedFoldCache:
    def __init__(self, c: Ctx, P: dict[str, np.ndarray]):
        self.c = c
        self.P = P
        self._cache: dict[tuple[str, tuple[int, ...], int], np.ndarray] = {}

    def stream(self, name: str, train_folds, eval_fold: int) -> np.ndarray:
        key = (name, tuple(sorted(int(x) for x in train_folds)), int(eval_fold))
        if key not in self._cache:
            train_rows = np.concatenate([self.c.rows[g] for g in key[1]])
            eval_rows = self.c.rows[int(eval_fold)]
            f = SCDF_NSEEN(self.P[name][train_rows], self.c.d.t[train_rows])
            self._cache[key] = f(self.P[name][eval_rows], self.c.d.t[eval_rows])
        return self._cache[key]

    def blend(self, streams: list[str], train_folds, eval_fold: int) -> np.ndarray:
        cols = [self.stream(s, train_folds, eval_fold) for s in streams]
        return np.column_stack(cols).mean(axis=1)


def blend_fixed(c: Ctx, cache: CalibratedFoldCache, streams: list[str]) -> tuple[np.ndarray, list[float]]:
    out = np.full(len(c.d.y), np.nan, dtype=np.float64)
    per = []
    for f in FOLDS:
        rows = c.rows[f]
        pred = cache.blend(streams, [g for g in FOLDS if g != f], f)
        out[rows] = pred
        per.append(float(ts_auc_flat(pred, c.d.y[rows], c.d.t[rows])))
    return out, per


def score_vec(c: Ctx, v: np.ndarray) -> dict:
    per = [float(ts_auc_flat(v[c.rows[f]], c.d.y[c.rows[f]], c.d.t[c.rows[f]])) for f in FOLDS]
    return {
        "mean_ts_auc": float(np.mean(per)),
        "pooled_ts_auc": float(ts_auc_flat(v[c.dev], c.d.y[c.dev], c.d.t[c.dev])),
        "per_fold_ts_auc": per,
        "fold_std": float(np.std(per)),
    }


def cell_rows(c: Ctx, rows: np.ndarray, split: str) -> np.ndarray:
    y = c.d.y[rows]
    t = c.d.t[rows]
    age = c.age[rows]
    dominant = (t >= 200) & ((y == 0) | (age >= 100))
    if split == "whole":
        return rows
    if split == "dominant_cell":
        return rows[dominant]
    hb = c.has_break[c.d.sidx[rows]]
    if split == "mature_vs_never":
        return rows[dominant & ((y == 1) | ((y == 0) & ~hb))]
    if split == "mature_vs_prebreak":
        return rows[dominant & ((y == 1) | ((y == 0) & hb))]
    raise KeyError(split)


def pair_repair_stats(base, cand, c: Ctx, rows: np.ndarray, pairs_per_t: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    yy = c.d.y
    tt = c.d.t
    order = np.argsort(tt[rows], kind="stable")
    sorted_rows = rows[order]
    sorted_t = tt[sorted_rows]
    starts = np.flatnonzero(np.r_[True, sorted_t[1:] != sorted_t[:-1]])
    ends = np.r_[starts[1:], len(sorted_t)]
    repairs = damage = total = 0
    for lo, hi in zip(starts, ends):
        idx = sorted_rows[lo:hi]
        pos = idx[yy[idx] == 1]
        neg = idx[yy[idx] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        k = min(pairs_per_t, len(pos), len(neg))
        pp = rng.choice(pos, k, replace=False)
        nn = rng.choice(neg, k, replace=False)
        base_right = base[pp] > base[nn]
        cand_right = cand[pp] > cand[nn]
        repairs += int((~base_right & cand_right).sum())
        damage += int((base_right & ~cand_right).sum())
        total += int(k)
    return {
        "total_pairs_sampled": int(total),
        "repairs": int(repairs),
        "damage": int(damage),
        "net_pair_lift": int(repairs - damage),
        "damage_rate": float(damage / max(total, 1)),
    }


def pair_flows(base, cand, c: Ctx) -> dict:
    splits = ("whole", "dominant_cell", "mature_vs_never", "mature_vs_prebreak")
    return {
        s: pair_repair_stats(base, cand, c, cell_rows(c, c.dev, s), PAIRS_PER_T, PAIR_SEED)
        for s in splits
    }


def within_t_rank_corr(a, b, t, rows) -> float:
    rr = rows[np.isfinite(a[rows]) & np.isfinite(b[rows])]
    ra = np.empty(len(rr), dtype=np.float64)
    rb = np.empty(len(rr), dtype=np.float64)
    tt = t[rr]
    order = np.argsort(tt, kind="stable")
    sorted_t = tt[order]
    starts = np.flatnonzero(np.r_[True, sorted_t[1:] != sorted_t[:-1]])
    ends = np.r_[starts[1:], len(sorted_t)]
    for lo, hi in zip(starts, ends):
        loc = order[lo:hi]
        src = rr[loc]
        denom = max(hi - lo, 1)
        ra[loc] = rankdata(a[src], method="average") / denom
        rb[loc] = rankdata(b[src], method="average") / denom
    return float(np.corrcoef(ra, rb)[0, 1])


def sample_train_rows(d: Data, fold: int, max_train_rows: int, seed: int, sample_mode: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = {}
    for f in FOLDS:
        tr_rows = d.rows_for([g for g in FOLDS if g != f])
        if len(tr_rows) > max_train_rows:
            if sample_mode == "uniform":
                tr_rows = np.sort(rng.choice(tr_rows, max_train_rows, replace=False))
            elif sample_mode == "per_series":
                s = d.sidx[tr_rows]
                order = np.argsort(s, kind="stable")
                tr_rows = tr_rows[order]
                s = s[order]
                bnd = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
                per = max_train_rows // len(bnd)
                ends = np.r_[bnd[1:], len(s)]
                sel = []
                for b, e in zip(bnd, ends):
                    idx = np.arange(b, e)
                    sel.append(idx if e - b <= per else rng.choice(idx, per, replace=False))
                tr_rows = np.sort(tr_rows[np.concatenate(sel)])
            else:
                raise KeyError(sample_mode)
        out[int(f)] = tr_rows
    return out[int(fold)]


def train_cat_specialist(name: str) -> dict:
    from catboost import CatBoostClassifier

    spec = SPECS[name]
    exp_id = IDS[name]
    out_path = OOF_DIR / f"{exp_id}.npy"
    if out_path.exists() and not ARGS.force_train:
        print(f"{name}: reusing {out_path}", flush=True)
        c = make_ctx()
        return {
            "id": exp_id,
            "name": name,
            "status": "REUSED",
            "runtime_s": 0.0,
            "fold_runtime_s": [],
            "standalone": score_vec(c, np.load(out_path)),
        }

    t0 = time.time()
    d = Data()
    mats, names_all = load_features(spec["modules"])
    keep = np.arange(len(names_all), dtype=np.int64)
    oof = np.full(len(d.y), np.nan, dtype=np.float32)
    per = []
    fold_runtime = []
    params = dict(CATBOOST_BASE_PARAMS, random_seed=int(spec["seed"]))
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    for f in FOLDS:
        ft = time.time()
        tr_rows = sample_train_rows(d, f, int(spec["max_train_rows"]), int(spec["seed"]), spec["sample_mode"])
        va_rows = d.rows_for([f])
        Xtr = _stack(mats, names_all, tr_rows, keep)
        ytr = d.y[tr_rows]
        model = CatBoostClassifier(**params)
        model.fit(Xtr, ytr)
        del Xtr, ytr
        gc.collect()
        Xva = _stack(mats, names_all, va_rows, keep)
        pred = model.predict_proba(Xva)[:, 1].astype(np.float32)
        del Xva, model
        gc.collect()
        oof[va_rows] = pred
        auc = float(ts_auc_flat(pred, d.y[va_rows], d.t[va_rows]))
        per.append(auc)
        fold_runtime.append(float(time.time() - ft))
        print(f"{name} fold {f}: {auc:.6f} runtime_s={fold_runtime[-1]:.1f}", flush=True)
    np.save(out_path, oof)
    dev = d.rows_for(list(FOLDS))
    return {
        "id": exp_id,
        "name": name,
        "status": "SCORED",
        "oof_path": str(out_path),
        "runtime_s": float(time.time() - t0),
        "fold_runtime_s": fold_runtime,
        "standalone": {
            "mean_ts_auc": float(np.mean(per)),
            "pooled_ts_auc": float(ts_auc_flat(oof[dev], d.y[dev], d.t[dev])),
            "per_fold_ts_auc": per,
            "fold_std": float(np.std(per)),
        },
    }


def verdict(marginal: float, positive_folds: int, pair_vs_e0: dict) -> str:
    dom = pair_vs_e0["dominant_cell"]["net_pair_lift"]
    mn = pair_vs_e0["mature_vs_never"]["net_pair_lift"]
    mp = pair_vs_e0["mature_vs_prebreak"]["net_pair_lift"]
    if marginal >= 0.0030 and positive_folds >= 4 and dom > 0 and mn > 0 and mp > 0:
        return "SERIOUS"
    if marginal >= 0.0015 and positive_folds >= 4 and dom > 0:
        return "PROMISING"
    if marginal >= 0.0010:
        return "INTERESTING"
    return "KILL"


def evaluate_zero_training(c: Ctx) -> dict:
    need = list(dict.fromkeys(SPECIALISTS + [MATCHED_CLONE, "RT-1251"]))
    P = load_oof(need)
    cache = CalibratedFoldCache(c, P)
    e0_vec, e0_per = blend_fixed(c, cache, list(SPECIALISTS))
    e1_vec, e1_per = blend_fixed(c, cache, list(SPECIALISTS) + [MATCHED_CLONE])
    e2_vec, e2_per = blend_fixed(c, cache, list(SPECIALISTS) + ["RT-1251"])
    return {
        "E0_original_rt600": {"mean_ts_auc": float(np.mean(e0_per)), "per_fold_ts_auc": e0_per},
        "E1_rt600_plus_rt401": {"mean_ts_auc": float(np.mean(e1_per)), "per_fold_ts_auc": e1_per},
        "E2_rt600_plus_rt1251": {"mean_ts_auc": float(np.mean(e2_per)), "per_fold_ts_auc": e2_per},
        "E2_minus_E1": float(np.mean(e2_per) - np.mean(e1_per)),
        "E2_minus_E0": float(np.mean(e2_per) - np.mean(e0_per)),
        "fold_deltas_E2_minus_E1": [float(a - b) for a, b in zip(e2_per, e1_per)],
        "pair_flow_vs_E1": pair_flows(e1_vec, e2_vec, c),
        "pair_flow_vs_E0": pair_flows(e0_vec, e2_vec, c),
    }


def composition(replaced: str, replacement: str) -> list[str]:
    return [replacement if s == replaced else s for s in SPECIALISTS]


def evaluate_specialist(c: Ctx, name: str, trained: dict | None = None) -> dict:
    spec = SPECS[name]
    cat_id = IDS[name]
    replaced = spec["replaced"]
    need = list(dict.fromkeys(SPECIALISTS + [MATCHED_CLONE, cat_id]))
    P = load_oof(need)
    cache = CalibratedFoldCache(c, P)
    e0_vec, e0_per = blend_fixed(c, cache, list(SPECIALISTS))
    e1_vec, e1_per = blend_fixed(c, cache, composition(replaced, MATCHED_CLONE))
    e2_vec, e2_per = blend_fixed(c, cache, composition(replaced, cat_id))
    standalone = trained["standalone"] if trained else score_vec(c, P[cat_id])
    incumbent = score_vec(c, P[replaced])
    fold_deltas = [float(a - b) for a, b in zip(e2_per, e1_per)]
    marginal = float(np.mean(e2_per) - np.mean(e1_per))
    pair_vs_e0 = pair_flows(e0_vec, e2_vec, c)
    pair_vs_clone = pair_flows(e1_vec, e2_vec, c)
    positive = int(sum(x > 0 for x in fold_deltas))
    return {
        "id": cat_id,
        "name": name,
        "replaced": replaced,
        "status": "SCORED",
        "runtime_s": float((trained or {}).get("runtime_s", 0.0)),
        "fold_runtime_s": (trained or {}).get("fold_runtime_s", []),
        "catboost_params": dict(CATBOOST_BASE_PARAMS, random_seed=int(spec["seed"])),
        "training_spec": dict(spec),
        "standalone": standalone,
        "incumbent_standalone": incumbent,
        "standalone_delta_vs_incumbent": float(standalone["mean_ts_auc"] - incumbent["mean_ts_auc"]),
        "rho_vs_incumbent": within_t_rank_corr(P[cat_id], P[replaced], c.d.t, c.dev),
        "ensemble": {
            "E0_original_rt600": {"mean_ts_auc": float(np.mean(e0_per)), "per_fold_ts_auc": e0_per},
            "E1_clone_replacement": {"mean_ts_auc": float(np.mean(e1_per)), "per_fold_ts_auc": e1_per},
            "E2_cat_replacement": {"mean_ts_auc": float(np.mean(e2_per)), "per_fold_ts_auc": e2_per},
            "marginal_vs_clone": marginal,
            "E2_minus_E0": float(np.mean(e2_per) - np.mean(e0_per)),
            "fold_deltas_vs_clone": fold_deltas,
            "positive_folds_vs_clone": positive,
        },
        "pair_flow_vs_E0": pair_vs_e0,
        "pair_flow_vs_clone": pair_vs_clone,
        "verdict": verdict(marginal, positive, pair_vs_e0),
    }


def evaluate_hybrid(c: Ctx, survivors: list[str]) -> dict:
    cat_by_replaced = {SPECS[n]["replaced"]: IDS[n] for n in survivors}
    clone_by_replaced = {}
    clone_iter = iter(SEED_CLONES)
    for s in SPECIALISTS:
        if s in cat_by_replaced:
            clone_by_replaced[s] = next(clone_iter)
    cat_streams = [cat_by_replaced.get(s, s) for s in SPECIALISTS]
    clone_streams = [clone_by_replaced.get(s, s) for s in SPECIALISTS]
    need = list(dict.fromkeys(SPECIALISTS + cat_streams + clone_streams))
    P = load_oof(need)
    cache = CalibratedFoldCache(c, P)
    e0_vec, e0_per = blend_fixed(c, cache, list(SPECIALISTS))
    e1_vec, e1_per = blend_fixed(c, cache, clone_streams)
    e2_vec, e2_per = blend_fixed(c, cache, cat_streams)
    fold_deltas = [float(a - b) for a, b in zip(e2_per, e1_per)]
    marginal = float(np.mean(e2_per) - np.mean(e1_per))
    pair_vs_e0 = pair_flows(e0_vec, e2_vec, c)
    positive = int(sum(x > 0 for x in fold_deltas))
    if marginal >= 0.0050 and positive >= 4 and pair_vs_e0["dominant_cell"]["net_pair_lift"] > 0 and pair_vs_e0["mature_vs_never"]["net_pair_lift"] > 0:
        v = "MAJOR"
    elif marginal >= 0.0030 and positive >= 4 and pair_vs_e0["dominant_cell"]["net_pair_lift"] > 0 and pair_vs_e0["mature_vs_never"]["net_pair_lift"] > 0:
        v = "SERIOUS"
    elif marginal >= 0.0015 and positive >= 4 and pair_vs_e0["dominant_cell"]["net_pair_lift"] > 0 and pair_vs_e0["mature_vs_never"]["net_pair_lift"] > 0:
        v = "PROMOTION_WORTHY"
    else:
        v = "KILL"
    return {
        "id": IDS["HYBRID"],
        "name": "HYBRID",
        "survivors": survivors,
        "cat_streams": cat_streams,
        "clone_streams": clone_streams,
        "runtime_s": 0.0,
        "ensemble": {
            "E0_original_rt600": {"mean_ts_auc": float(np.mean(e0_per)), "per_fold_ts_auc": e0_per},
            "E1_clone_replacement": {"mean_ts_auc": float(np.mean(e1_per)), "per_fold_ts_auc": e1_per},
            "E2_cat_hybrid": {"mean_ts_auc": float(np.mean(e2_per)), "per_fold_ts_auc": e2_per},
            "marginal_vs_clone": marginal,
            "E2_minus_E0": float(np.mean(e2_per) - np.mean(e0_per)),
            "fold_deltas_vs_clone": fold_deltas,
            "positive_folds_vs_clone": positive,
        },
        "pair_flow_vs_E0": pair_vs_e0,
        "pair_flow_vs_clone": pair_flows(e1_vec, e2_vec, c),
        "verdict": v,
    }


def result_rows(result: dict) -> list[dict]:
    base = {
        "date": time.strftime("%Y-%m-%d"),
        "git_sha": result["git_sha"],
        "agent": "codex-catboost-specialist",
        "hypothesis": "Frozen RT-1251 CatBoost may amplify non-LightGBM ensemble alpha when reimplementing RT600 specialist configurations.",
        "falsification_condition": "KILL if fixed-slot marginal_vs_clone < +0.0010; hybrid promotion-worthy requires >=+0.0015, >=4/5 positive folds, dominant and mature pair-flow gates positive.",
        "folds": "0,1,2,3,4",
        "train_series": "8000",
        "persistence": "none",
        "causal_verified": "inherited frozen causal feature bank; no RT600 score input; no lockbox/test",
        "test_reduced_touched": "no",
        "lockbox_touched": "no",
        "protocol": "catboost_specialist_2026_full5",
    }
    rows = []
    for name in ("CAT-413", "CAT-300", "CAT-410"):
        r = result["specialists"].get(name)
        if not r or r.get("status") != "SCORED":
            continue
        spec = r["training_spec"]
        s = r["standalone"]
        ens = r["ensemble"]
        row = dict(base)
        row.update({
            "experiment_id": r["id"],
            "feature_set": ",".join(spec["modules"]),
            "n_features": "500" if spec["modules"] == FULL else "261",
            "model": "catboost",
            "objective": "binary_logloss",
            "random_seed": str(spec["seed"]),
            "train_rows": str(spec["max_train_rows"]),
            "mean_oof_ts_auc": repr(s["mean_ts_auc"]),
            "pooled_oof_ts_auc": repr(s["pooled_ts_auc"]),
            "per_fold_ts_auc": ";".join(f"{x:.5f}" for x in s["per_fold_ts_auc"]),
            "fold_std": repr(s["fold_std"]),
            "sample_mode": spec["sample_mode"],
            "training_runtime_s": repr(r.get("runtime_s", 0.0)),
            "status": r["verdict"],
            "notes": (
                f"{name}; replaced={r['replaced']}; incumbent={r['incumbent_standalone']['mean_ts_auc']:.9f}; "
                f"standalone_delta={r['standalone_delta_vs_incumbent']:+.9f}; "
                f"rho={r['rho_vs_incumbent']:+.6f}; E0={ens['E0_original_rt600']['mean_ts_auc']:.9f}; "
                f"E1={ens['E1_clone_replacement']['mean_ts_auc']:.9f}; "
                f"E2={ens['E2_cat_replacement']['mean_ts_auc']:.9f}; "
                f"marginal_vs_clone={ens['marginal_vs_clone']:+.9f}; "
                f"E2-E0={ens['E2_minus_E0']:+.9f}; positive_folds={ens['positive_folds_vs_clone']}/5; "
                f"dominant_net={r['pair_flow_vs_E0']['dominant_cell']['net_pair_lift']}; "
                f"mature_vs_never={r['pair_flow_vs_E0']['mature_vs_never']['net_pair_lift']}; "
                f"mature_vs_prebreak={r['pair_flow_vs_E0']['mature_vs_prebreak']['net_pair_lift']}."
            ),
        })
        rows.append(row)
    h = result.get("hybrid")
    if h and h.get("ran"):
        ens = h["ensemble"]
        row = dict(base)
        row.update({
            "experiment_id": h["id"],
            "feature_set": ",".join(h["cat_streams"]),
            "n_features": "7",
            "model": "frozen_oof_equal_scdf_hybrid",
            "objective": "artifact_rescore",
            "random_seed": "20260827",
            "train_rows": "",
            "mean_oof_ts_auc": repr(ens["E2_cat_hybrid"]["mean_ts_auc"]),
            "pooled_oof_ts_auc": "",
            "per_fold_ts_auc": ";".join(f"{x:.5f}" for x in ens["E2_cat_hybrid"]["per_fold_ts_auc"]),
            "fold_std": repr(float(np.std(ens["E2_cat_hybrid"]["per_fold_ts_auc"]))),
            "sample_mode": "frozen_oof",
            "training_runtime_s": "0.0",
            "status": h["verdict"],
            "notes": (
                f"hybrid survivors={','.join(h['survivors'])}; E0={ens['E0_original_rt600']['mean_ts_auc']:.9f}; "
                f"E1={ens['E1_clone_replacement']['mean_ts_auc']:.9f}; E2={ens['E2_cat_hybrid']['mean_ts_auc']:.9f}; "
                f"marginal_vs_clone={ens['marginal_vs_clone']:+.9f}; E2-E0={ens['E2_minus_E0']:+.9f}; "
                f"positive_folds={ens['positive_folds_vs_clone']}/5; "
                f"dominant_net={h['pair_flow_vs_E0']['dominant_cell']['net_pair_lift']}; "
                f"mature_vs_never={h['pair_flow_vs_E0']['mature_vs_never']['net_pair_lift']}."
            ),
        })
        rows.append(row)
    return rows


def append_results(result: dict) -> None:
    rows = result_rows(result)
    if not rows:
        return
    with RESULTS_CSV.open("r+", newline="") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise RuntimeError("RESULTS.csv has no header")
        existing = {r["experiment_id"] for r in reader}
        dup = [r["experiment_id"] for r in rows if r["experiment_id"] in existing]
        if dup:
            raise RuntimeError(f"RESULTS.csv already contains {dup}")
        fh.seek(0, os.SEEK_END)
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def write_report_csv(result: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORT_DIR / "results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "rt_id", "standalone", "rho", "marginal_vs_clone", "E2_minus_E0", "folds_positive", "dominant_net", "runtime_s", "verdict"])
        for name in ("CSA-00", "CAT-413", "CAT-300", "CAT-410", "HYBRID"):
            if name == "CSA-00":
                z = result["csa00"]
                w.writerow([name, "", "", "", z["E2_minus_E1"], z["E2_minus_E0"], "", z["pair_flow_vs_E0"]["dominant_cell"]["net_pair_lift"], "0.0", "DESCRIPTIVE"])
                continue
            r = result["specialists"].get(name) if name != "HYBRID" else result.get("hybrid")
            if not r or (name == "HYBRID" and not r.get("ran")):
                continue
            ens = r["ensemble"]
            if name == "HYBRID":
                w.writerow([name, r["id"], "", "", ens["marginal_vs_clone"], ens["E2_minus_E0"], ens["positive_folds_vs_clone"], r["pair_flow_vs_E0"]["dominant_cell"]["net_pair_lift"], r.get("runtime_s", 0.0), r["verdict"]])
            else:
                w.writerow([name, r["id"], r["standalone"]["mean_ts_auc"], r["rho_vs_incumbent"], ens["marginal_vs_clone"], ens["E2_minus_E0"], ens["positive_folds_vs_clone"], r["pair_flow_vs_E0"]["dominant_cell"]["net_pair_lift"], r.get("runtime_s", 0.0), r["verdict"]])


def fmt(x, n=9) -> str:
    if x == "" or x is None:
        return ""
    return f"{float(x):.{n}f}"


def write_final_md(result: dict) -> None:
    rows = []
    for name in ("CAT-413", "CAT-300", "CAT-410"):
        r = result["specialists"].get(name)
        if r:
            rows.append((name, r["id"], fmt(r["standalone"]["mean_ts_auc"]), f"{r['rho_vs_incumbent']:.6f}", f"{r['ensemble']['marginal_vs_clone']:+.9f}", f"{r['ensemble']['E2_minus_E0']:+.9f}", f"{r['ensemble']['positive_folds_vs_clone']}/5", str(r["pair_flow_vs_E0"]["dominant_cell"]["net_pair_lift"]), f"{r['runtime_s']:.1f}s", r["verdict"]))
    h = result.get("hybrid")
    if h and h.get("ran"):
        rows.append(("HYBRID", h["id"], "", "", f"{h['ensemble']['marginal_vs_clone']:+.9f}", f"{h['ensemble']['E2_minus_E0']:+.9f}", f"{h['ensemble']['positive_folds_vs_clone']}/5", str(h["pair_flow_vs_E0"]["dominant_cell"]["net_pair_lift"]), "0.0s", h["verdict"]))
    table = ["| arm | RT ID | standalone | rho | marginal_vs_clone | E2-E0 | folds positive | dominant net | runtime | verdict |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    table += [f"| {a} | `{rid}` | {st} | {rho} | {mvc} | {e20} | {fp} | {dn} | {rt} | {v} |" for a, rid, st, rho, mvc, e20, fp, dn, rt, v in rows]
    csa00 = result["csa00"]
    scored = [r for r in result["specialists"].values() if r.get("status") == "SCORED"]
    best_spec = max(scored, key=lambda r: r["ensemble"]["marginal_vs_clone"]) if scored else None
    best_gain = max([r["ensemble"]["E2_minus_E0"] for r in scored] + ([h["ensemble"]["E2_minus_E0"]] if h and h.get("ran") else []))
    best_marginal = max([r["ensemble"]["marginal_vs_clone"] for r in scored] + ([h["ensemble"]["marginal_vs_clone"]] if h and h.get("ran") else []))
    prod = "Training is expensive but finite; inference adds one CatBoost model per surviving replacement and needs separate deployment benchmarking."
    if best_spec and best_spec["verdict"] == "KILL":
        prod = "Not production-feasible as a promotion: CSA-01 failed the activation gate, so no CatBoost-heavy ensemble is justified."
    if h and h.get("ran"):
        hybrid_line = (
            f"Hybrid: run, verdict `{h['verdict']}`, "
            f"marginal_vs_clone `{h['ensemble']['marginal_vs_clone']:+.9f}`."
        )
    else:
        hybrid_line = "Hybrid: not run."
    lines = [
        "# CATBOOST SPECIALIST ACTIVATION 2026 — COMPLETE",
        "",
        f"Date: 2026-08-27",
        f"Branch: `research/catboost-specialist-2026`",
        f"Scoring SHA: `{result['git_sha']}`",
        "",
        *table,
        "",
        f"CSA-00: E2-E1 `{csa00['E2_minus_E1']:+.9f}`; E2-E0 `{csa00['E2_minus_E0']:+.9f}`; dominant net vs E0 `{csa00['pair_flow_vs_E0']['dominant_cell']['net_pair_lift']}`.",
        f"Best specialist: `{best_spec['name']}` with `marginal_vs_clone = {best_spec['ensemble']['marginal_vs_clone']:+.9f}`." if best_spec else "Best specialist: none scored.",
        hybrid_line,
        f"Best measured gain over original RT600: `{best_gain:+.9f}`.",
        f"Best marginal_vs_clone: `{best_marginal:+.9f}`.",
        f"Production feasibility: {prod}",
        f"Next action: {'close CatBoost specialist activation; do not tune CatBoost.' if best_spec and best_spec['verdict'] == 'KILL' else 'review surviving CatBoost specialist(s) against deployment cost before any production work.'}",
        "",
        "No lockbox, test, production, feature, router, stacker, hyperparameter, seed, or blend-weight change was made.",
    ]
    (REPORT_DIR / "FINAL.md").write_text("\n".join(lines) + "\n")


def write_outputs(result: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_report_csv(result)
    write_final_md(result)


def main() -> None:
    t0 = time.time()
    c = make_ctx()
    result = {
        "program": "CATBOOST SPECIALIST ACTIVATION 2026",
        "git_sha": git_sha(),
        "artifact_root": str(ARTIFACT_ROOT),
        "oof_dir": str(OOF_DIR),
        "versions": env_versions(),
        "calibration": "SCDF_NSEEN",
        "pair_seed": PAIR_SEED,
        "pairs_per_t": PAIRS_PER_T,
        "csa00": evaluate_zero_training(c),
        "specialists": {},
        "hybrid": {"ran": False, "reason": "requires at least two specialists with marginal_vs_clone >= +0.0010"},
    }
    trained = train_cat_specialist("CAT-413")
    result["specialists"]["CAT-413"] = evaluate_specialist(c, "CAT-413", trained)
    if result["specialists"]["CAT-413"]["ensemble"]["marginal_vs_clone"] >= 0.0010:
        for name in ("CAT-300", "CAT-410"):
            trained = train_cat_specialist(name)
            result["specialists"][name] = evaluate_specialist(c, name, trained)
        survivors = [n for n, r in result["specialists"].items() if r["ensemble"]["marginal_vs_clone"] >= 0.0010]
        if len(survivors) >= 2:
            result["hybrid"] = dict(evaluate_hybrid(c, survivors), ran=True)
    result["runtime_s"] = float(time.time() - t0)
    write_outputs(result)
    if not ARGS.no_ledger:
        append_results(result)
    print(json.dumps({
        "results": str(REPORT_DIR / "results.json"),
        "final": str(REPORT_DIR / "FINAL.md"),
        "csa01_verdict": result["specialists"]["CAT-413"]["verdict"],
        "csa01_marginal_vs_clone": result["specialists"]["CAT-413"]["ensemble"]["marginal_vs_clone"],
        "hybrid_ran": bool(result["hybrid"].get("ran")),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
