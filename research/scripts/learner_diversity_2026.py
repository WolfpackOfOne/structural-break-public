#!/usr/bin/env python
"""Learner Diversity 2026.

Full canonical-fold test of non-LightGBM learner-family diversity against the
RT-600 seven-specialist ensemble and an exchangeable RT-401 LightGBM clone.
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
os.environ.setdefault("SBR_ROOT", str(REPO))
for _p in (REPO / "src", REPO / "research" / "scripts"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from sbr.metric import ts_auc_flat  # noqa: E402
from sbr.pipeline import Data, _stack, load_features  # noqa: E402
from wave2_train_ensemble import FULL  # noqa: E402
from wave5_lib import CANON_CAL, FOLDS, SPECIALISTS, Ctx, load_oof  # noqa: E402

REPORT_DIR = REPO / "research" / "reports" / "learner_diversity_2026"
OOF_DIR = REPO / "research" / "oof"
RESULTS_CSV = REPO / "research" / "RESULTS.csv"

IDS = {"tabm": "RT-1250", "catboost": "RT-1251", "realmlp": "RT-1252"}
MATCHED_LGBM = "RT-401"
SEED = 1
MAX_TRAIN_ROWS = 1_000_000
PAIR_SEED = 20260827
PAIRS_PER_T = 64

CATBOOST_PARAMS = {
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
    "random_seed": SEED,
    "thread_count": 2,
    "allow_writing_files": False,
    "verbose": False,
}

INFEASIBLE = {
    "tabm": (
        "tabm 0.0.3 / pytabkit TabM_D defaults require k=32, d_block=512, "
        "batch_size=256, quantile_tabr preprocessing, patience=16 and up to "
        "1e9 epochs. At 1,000,000 rows/fold this is at least 3,907 batches per "
        "epoch and >=17 validation epochs per fold, five folds. Full-scale run "
        "is not feasible without shrinking or retuning, both forbidden."
    ),
    "realmlp": (
        "pytabkit 1.7.3 RealMLP_TD defaults require hidden_sizes=[256]*3, "
        "256 epochs, batch_size=256 and full matrix preprocessing/copies. "
        "At 1,000,000 rows/fold this is about 3,907 batches/epoch/fold and "
        "roughly five million batch steps over five folds. Full-scale run is "
        "not feasible without shrinking or retuning, both forbidden."
    ),
}


@dataclass(frozen=True)
class FoldChoice:
    outer_fold: int
    replaced: str
    replacement: str
    inner_mean_ts_auc: float
    inner_fold_ts_auc: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", choices=("all", "catboost", "analyze"), default="all")
    p.add_argument("--no-ledger", action="store_true")
    return p.parse_args()


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True
    ).strip()


def env_versions() -> dict[str, str]:
    import importlib.metadata as md

    pkgs = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "catboost",
        "torch",
        "tabm",
        "rtdl_num_embeddings",
        "pytabkit",
    ]
    out = {}
    for p in pkgs:
        try:
            out[p] = md.version(p)
        except Exception:
            out[p] = "missing"
    return out


def champ_fold_rows(d: Data) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(SEED)
    out = {}
    for f in FOLDS:
        tr_rows = d.rows_for([g for g in FOLDS if g != f])
        va_rows = d.rows_for([f])
        if len(tr_rows) > MAX_TRAIN_ROWS:
            tr_rows = np.sort(rng.choice(tr_rows, MAX_TRAIN_ROWS, replace=False))
        out[int(f)] = (tr_rows, va_rows)
    return out


def train_catboost() -> dict:
    from catboost import CatBoostClassifier

    t0 = time.time()
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OOF_DIR / f"{IDS['catboost']}.npy"
    d = Data()
    mats, names = load_features(FULL)
    keep = np.arange(len(names), dtype=np.int64)
    rows = champ_fold_rows(d)
    oof = np.full(len(d.y), np.nan, dtype=np.float32)
    per = []
    fold_runtime = []
    for f in FOLDS:
        ft = time.time()
        tr_rows, va_rows = rows[int(f)]
        Xtr = _stack(mats, names, tr_rows, keep)
        ytr = d.y[tr_rows]
        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(Xtr, ytr)
        del Xtr, ytr
        gc.collect()
        Xva = _stack(mats, names, va_rows, keep)
        pred = model.predict_proba(Xva)[:, 1].astype(np.float32)
        del Xva, model
        gc.collect()
        oof[va_rows] = pred
        auc = float(ts_auc_flat(pred, d.y[va_rows], d.t[va_rows]))
        per.append(auc)
        fold_runtime.append(float(time.time() - ft))
        print(f"catboost fold {f}: {auc:.6f} runtime_s={fold_runtime[-1]:.1f}", flush=True)
    np.save(out_path, oof)
    dev = d.rows_for(list(FOLDS))
    return {
        "id": IDS["catboost"],
        "learner": "CatBoost",
        "status": "scored",
        "oof_path": str(out_path),
        "standalone": {
            "mean_ts_auc": float(np.mean(per)),
            "pooled_ts_auc": float(ts_auc_flat(oof[dev], d.y[dev], d.t[dev])),
            "per_fold_ts_auc": per,
            "fold_std": float(np.std(per)),
        },
        "runtime_s": float(time.time() - t0),
        "fold_runtime_s": fold_runtime,
    }


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
            f = CANON_CAL(self.P[name][train_rows], self.c.d.t[train_rows])
            self._cache[key] = f(self.P[name][eval_rows], self.c.d.t[eval_rows])
        return self._cache[key]

    def blend(self, streams: list[str], train_folds, eval_fold: int) -> np.ndarray:
        cols = [self.stream(s, train_folds, eval_fold) for s in streams]
        return np.column_stack(cols).mean(axis=1)


def composition(replaced: str, replacement: str) -> list[str]:
    return [replacement if s == replaced else s for s in SPECIALISTS]


def score_inner(c: Ctx, cache: CalibratedFoldCache, streams: list[str], outer_train_folds):
    per = []
    for g in outer_train_folds:
        train_folds = [x for x in outer_train_folds if x != g]
        rows = c.rows[g]
        pred = cache.blend(streams, train_folds, g)
        per.append(float(ts_auc_flat(pred, c.d.y[rows], c.d.t[rows])))
    return float(np.mean(per)), tuple(per)


def select_nested(c: Ctx, cache: CalibratedFoldCache, outer_fold: int, replacement: str) -> FoldChoice:
    outer_train = tuple(g for g in FOLDS if g != outer_fold)
    scored = []
    for replaced in SPECIALISTS:
        mean_auc, per = score_inner(c, cache, composition(replaced, replacement), outer_train)
        scored.append((mean_auc, replacement, replaced, per))
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    mean_auc, repl, replaced, per = scored[0]
    return FoldChoice(int(outer_fold), replaced, repl, mean_auc, per)


def evaluate_fixed(c: Ctx, cache: CalibratedFoldCache, streams: list[str]):
    out = np.full(len(c.d.y), np.nan, dtype=np.float64)
    per = []
    for f in FOLDS:
        rows = c.rows[f]
        pred = cache.blend(streams, [g for g in FOLDS if g != f], int(f))
        out[rows] = pred
        per.append(float(ts_auc_flat(pred, c.d.y[rows], c.d.t[rows])))
    return out, per


def evaluate_choices(c: Ctx, cache: CalibratedFoldCache, choices: dict[int, FoldChoice]):
    out = np.full(len(c.d.y), np.nan, dtype=np.float64)
    per = []
    for f in FOLDS:
        rows = c.rows[f]
        ch = choices[int(f)]
        pred = cache.blend(composition(ch.replaced, ch.replacement), [g for g in FOLDS if g != f], int(f))
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


def cell_rows(c: Ctx, rows, split: str) -> np.ndarray:
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


def pair_repair_stats(base, cand, c: Ctx, rows, pairs_per_t: int, seed: int) -> dict:
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


def verdict(marginal: float, positive_folds: int, dominant_net: int, standalone_delta: float, rho: float, unique_net: int) -> str:
    if standalone_delta < -0.010 and rho > 0.95 and unique_net <= 0:
        return "KILL"
    if marginal >= 0.005:
        return "MAJOR"
    if marginal >= 0.003 and positive_folds >= 4 and dominant_net > 0:
        return "SERIOUS"
    if marginal >= 0.001:
        return "INTERESTING"
    return "KILL"


def analyze(scored: dict | None = None) -> dict:
    t0 = time.time()
    c = Ctx()
    needed = list(SPECIALISTS) + [MATCHED_LGBM]
    if (OOF_DIR / f"{IDS['catboost']}.npy").exists():
        needed.append(IDS["catboost"])
    P = load_oof(needed)
    result = {
        "program": "LEARNER DIVERSITY 2026",
        "git_sha": git_sha(),
        "versions": env_versions(),
        "matched_lgbm_control": MATCHED_LGBM,
        "calibration": CANON_CAL.__name__,
        "seed": SEED,
        "max_train_rows": MAX_TRAIN_ROWS,
        "pairs_per_t": PAIRS_PER_T,
        "pair_seed": PAIR_SEED,
        "learners": {},
        "combination": {"run": False, "reason": "requires at least two learners with marginal_vs_clone >= +0.0010"},
    }
    for key in ("tabm", "realmlp"):
        result["learners"][key] = {
            "id": IDS[key],
            "learner": "TabM" if key == "tabm" else "RealMLP",
            "status": "INFEASIBLE",
            "runtime_s": 0.0,
            "reason": INFEASIBLE[key],
            "verdict": "INFEASIBLE",
        }
    if IDS["catboost"] in P:
        cache = CalibratedFoldCache(c, P)
        e0_vec, e0_per = evaluate_fixed(c, cache, list(SPECIALISTS))
        clone_choices = {int(f): select_nested(c, cache, int(f), MATCHED_LGBM) for f in FOLDS}
        cand_choices = {int(f): select_nested(c, cache, int(f), IDS["catboost"]) for f in FOLDS}
        clone_vec, clone_per = evaluate_choices(c, cache, clone_choices)
        cand_vec, cand_per = evaluate_choices(c, cache, cand_choices)
        rows_dev = c.dev
        dom_rows = cell_rows(c, rows_dev, "dominant_cell")
        splits = ("whole", "dominant_cell", "mature_vs_never", "mature_vs_prebreak")
        candidate_raw = P[IDS["catboost"]]
        matched_raw = P[MATCHED_LGBM]
        standalone = scored["standalone"] if scored else score_vec(c, candidate_raw)
        matched = score_vec(c, matched_raw)
        standalone_delta = float(standalone["mean_ts_auc"] - matched["mean_ts_auc"])
        rho = within_t_rank_corr(candidate_raw, matched_raw, c.d.t, rows_dev)
        pair_vs_lgbm = {
            s: pair_repair_stats(matched_raw, candidate_raw, c, cell_rows(c, rows_dev, s), PAIRS_PER_T, PAIR_SEED)
            for s in splits
        }
        pair_vs_e0 = {
            s: pair_repair_stats(e0_vec, cand_vec, c, cell_rows(c, rows_dev, s), PAIRS_PER_T, PAIR_SEED)
            for s in splits
        }
        fold_deltas = [float(a - b) for a, b in zip(cand_per, clone_per)]
        marginal = float(np.mean(cand_per) - np.mean(clone_per))
        positive = int(sum(x > 0 for x in fold_deltas))
        dom_net = int(pair_vs_e0["dominant_cell"]["net_pair_lift"])
        v = verdict(marginal, positive, dom_net, standalone_delta, rho, pair_vs_lgbm["whole"]["net_pair_lift"])
        result["learners"]["catboost"] = {
            "id": IDS["catboost"],
            "learner": "CatBoost",
            "status": "SCORED",
            "runtime_s": float((scored or {}).get("runtime_s", 0.0)),
            "standalone": standalone,
            "matched_lgbm_standalone": matched,
            "standalone_delta_vs_matched_lgbm": standalone_delta,
            "dominant_cell_auc": float(ts_auc_flat(candidate_raw[dom_rows], c.d.y[dom_rows], c.d.t[dom_rows])),
            "rho_vs_matched_lgbm": rho,
            "pair_flow_vs_matched_lgbm": pair_vs_lgbm,
            "ensemble": {
                "E0_original_rt600": {"scores": {"mean_ts_auc": float(np.mean(e0_per)), "per_fold_ts_auc": e0_per}},
                "E1_clone_replacement": {"scores": {"mean_ts_auc": float(np.mean(clone_per)), "per_fold_ts_auc": clone_per}},
                "E2_candidate_replacement": {"scores": {"mean_ts_auc": float(np.mean(cand_per)), "per_fold_ts_auc": cand_per}},
                "marginal_vs_clone": marginal,
                "fold_deltas_vs_clone": fold_deltas,
                "positive_folds_vs_clone": positive,
                "candidate_choices": [ch.__dict__ for ch in cand_choices.values()],
                "clone_choices": [ch.__dict__ for ch in clone_choices.values()],
            },
            "pair_flow_vs_e0": pair_vs_e0,
            "verdict": v,
        }
    result["runtime_s"] = float(time.time() - t0)
    write_outputs(result)
    return result


def write_outputs(result: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "learner_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (REPORT_DIR / "learner_results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "learner",
            "id",
            "standalone",
            "rho_vs_lgbm",
            "marginal_vs_clone",
            "folds_positive",
            "dominant_pair_net",
            "runtime_s",
            "verdict",
        ])
        for key in ("tabm", "catboost", "realmlp"):
            row = result["learners"].get(key, {})
            ens = row.get("ensemble", {})
            w.writerow([
                row.get("learner"),
                row.get("id"),
                row.get("standalone", {}).get("mean_ts_auc", ""),
                row.get("rho_vs_matched_lgbm", ""),
                ens.get("marginal_vs_clone", ""),
                ens.get("positive_folds_vs_clone", ""),
                row.get("pair_flow_vs_e0", {}).get("dominant_cell", {}).get("net_pair_lift", ""),
                row.get("runtime_s", ""),
                row.get("verdict"),
            ])


def result_rows(result: dict) -> list[dict]:
    rows = []
    base = {
        "date": time.strftime("%Y-%m-%d"),
        "git_sha": result["git_sha"],
        "agent": "codex-learner-diversity",
        "hypothesis": "Non-LightGBM tabular learners may produce useful same-t pair orderings beyond an exchangeable LightGBM RT-401 clone.",
        "falsification_condition": "KILL if marginal_vs_clone < +0.0010; SERIOUS requires >=+0.0030, >=4/5 positive folds, and dominant pair net >0.",
        "feature_set": ",".join(FULL),
        "n_features": "500",
        "folds": "0,1,2,3,4",
        "random_seed": str(SEED),
        "train_series": "8000",
        "train_rows": str(MAX_TRAIN_ROWS),
        "persistence": "none",
        "sample_mode": "uniform_rt401_matched_rows",
        "causal_verified": "inherited frozen 500-column causal feature bank; no RT600 score input; no lockbox/test",
        "test_reduced_touched": "no",
        "lockbox_touched": "no",
        "protocol": "learner_diversity_2026_full5",
    }
    for key in ("tabm", "catboost", "realmlp"):
        r = result["learners"][key]
        row = dict(base)
        row["experiment_id"] = r["id"]
        row["model"] = r["learner"].lower()
        row["objective"] = "binary_logloss"
        row["training_runtime_s"] = repr(r.get("runtime_s", 0.0))
        row["status"] = r["verdict"]
        if r["status"] == "SCORED":
            s = r["standalone"]
            ens = r["ensemble"]
            row["mean_oof_ts_auc"] = repr(s["mean_ts_auc"])
            row["pooled_oof_ts_auc"] = repr(s["pooled_ts_auc"])
            row["per_fold_ts_auc"] = ";".join(f"{x:.5f}" for x in s["per_fold_ts_auc"])
            row["fold_std"] = repr(s["fold_std"])
            row["notes"] = (
                f"standalone={s['mean_ts_auc']:.9f}; matched_lgbm={r['matched_lgbm_standalone']['mean_ts_auc']:.9f}; "
                f"rho={r['rho_vs_matched_lgbm']:+.6f}; E0={ens['E0_original_rt600']['scores']['mean_ts_auc']:.9f}; "
                f"E1={ens['E1_clone_replacement']['scores']['mean_ts_auc']:.9f}; "
                f"E2={ens['E2_candidate_replacement']['scores']['mean_ts_auc']:.9f}; "
                f"marginal_vs_clone={ens['marginal_vs_clone']:+.9f}; "
                f"positive_folds={ens['positive_folds_vs_clone']}/5; "
                f"dominant_net={r['pair_flow_vs_e0']['dominant_cell']['net_pair_lift']}."
            )
        else:
            row["mean_oof_ts_auc"] = ""
            row["pooled_oof_ts_auc"] = ""
            row["per_fold_ts_auc"] = ""
            row["fold_std"] = ""
            row["notes"] = r["reason"]
        rows.append(row)
    return rows


def append_results(result: dict) -> None:
    rows = result_rows(result)
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


def main() -> None:
    args = parse_args()
    scored = None
    if args.run in ("all", "catboost"):
        scored = train_catboost()
    result = analyze(scored)
    if not args.no_ledger:
        append_results(result)
    print(json.dumps({
        "json": str(REPORT_DIR / "learner_results.json"),
        "csv": str(REPORT_DIR / "learner_results.csv"),
        "catboost_verdict": result["learners"].get("catboost", {}).get("verdict"),
        "catboost_marginal_vs_clone": result["learners"].get("catboost", {}).get("ensemble", {}).get("marginal_vs_clone"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
