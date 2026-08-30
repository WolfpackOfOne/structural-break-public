#!/usr/bin/env python
"""Evaluate completed GPU-tabular OOF files.

This is the only script in the package that computes TS-AUC. It refuses to run
until all five folds for a learner are complete.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from common import (
    FINAL_MD,
    FOLDS,
    LOCAL_OOF_ROOT,
    MATCHED_LGBM,
    PAIR_SEED,
    PAIRS_PER_T,
    REPORT_DIR,
    RESULTS_CSV,
    RESULTS_JSON,
    SPECIALISTS,
    configure_roots,
    env_versions,
    git_branch,
    git_sha,
    load_control_oof,
    load_eval_context,
)
from scipy.stats import rankdata

LEARNER_LABELS = {"tabm": "GPU-01 TabM", "realmlp": "GPU-02 RealMLP"}
SPLITS = ("whole", "dominant_cell", "mature_vs_never", "mature_vs_prebreak")


@dataclass(frozen=True)
class FoldChoice:
    outer_fold: int
    replaced: str
    replacement: str
    inner_mean_ts_auc: float
    inner_fold_ts_auc: tuple[float, ...]


class CalibratedFoldCache:
    def __init__(self, c: Any, P: dict[str, np.ndarray], cal: Any):
        self.c = c
        self.P = P
        self.cal = cal
        self._cache: dict[tuple[str, tuple[int, ...], int], np.ndarray] = {}

    def stream(self, name: str, train_folds, eval_fold: int) -> np.ndarray:
        key = (name, tuple(sorted(int(x) for x in train_folds)), int(eval_fold))
        if key not in self._cache:
            train_rows = np.concatenate([self.c.rows[g] for g in key[1]])
            eval_rows = self.c.rows[int(eval_fold)]
            f = self.cal(self.P[name][train_rows], self.c.d.t[train_rows])
            self._cache[key] = f(self.P[name][eval_rows], self.c.d.t[eval_rows])
        return self._cache[key]

    def blend(self, streams: list[str], train_folds, eval_fold: int) -> np.ndarray:
        cols = [self.stream(s, train_folds, eval_fold) for s in streams]
        return np.column_stack(cols).mean(axis=1)


def composition(replaced: str, replacement: str) -> list[str]:
    return [replacement if s == replaced else s for s in SPECIALISTS]


def score_inner(
    c: Any, cache: CalibratedFoldCache, streams: list[str], outer_train_folds
) -> tuple[float, tuple[float, ...]]:
    from sbr.metric import ts_auc_flat

    per = []
    for g in outer_train_folds:
        train_folds = [x for x in outer_train_folds if x != g]
        rows = c.rows[g]
        pred = cache.blend(streams, train_folds, g)
        per.append(float(ts_auc_flat(pred, c.d.y[rows], c.d.t[rows])))
    return float(np.mean(per)), tuple(per)


def select_nested(
    c: Any, cache: CalibratedFoldCache, outer_fold: int, replacement: str
) -> FoldChoice:
    outer_train = tuple(g for g in FOLDS if g != outer_fold)
    scored = []
    for replaced in SPECIALISTS:
        mean_auc, per = score_inner(c, cache, composition(replaced, replacement), outer_train)
        scored.append((mean_auc, replacement, replaced, per))
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    mean_auc, repl, replaced, per = scored[0]
    return FoldChoice(int(outer_fold), replaced, repl, mean_auc, per)


def evaluate_fixed(
    c: Any, cache: CalibratedFoldCache, streams: list[str]
) -> tuple[np.ndarray, list[float]]:
    from sbr.metric import ts_auc_flat

    out = np.full(len(c.d.y), np.nan, dtype=np.float64)
    per = []
    for f in FOLDS:
        rows = c.rows[f]
        pred = cache.blend(streams, [g for g in FOLDS if g != f], int(f))
        out[rows] = pred
        per.append(float(ts_auc_flat(pred, c.d.y[rows], c.d.t[rows])))
    return out, per


def evaluate_choices(
    c: Any, cache: CalibratedFoldCache, choices: dict[int, FoldChoice]
) -> tuple[np.ndarray, list[float]]:
    from sbr.metric import ts_auc_flat

    out = np.full(len(c.d.y), np.nan, dtype=np.float64)
    per = []
    for f in FOLDS:
        rows = c.rows[f]
        ch = choices[int(f)]
        pred = cache.blend(
            composition(ch.replaced, ch.replacement), [g for g in FOLDS if g != f], int(f)
        )
        out[rows] = pred
        per.append(float(ts_auc_flat(pred, c.d.y[rows], c.d.t[rows])))
    return out, per


def score_vec(c: Any, v: np.ndarray) -> dict[str, Any]:
    from sbr.metric import ts_auc_flat

    per = [float(ts_auc_flat(v[c.rows[f]], c.d.y[c.rows[f]], c.d.t[c.rows[f]])) for f in FOLDS]
    return {
        "mean_ts_auc": float(np.mean(per)),
        "pooled_ts_auc": float(ts_auc_flat(v[c.dev], c.d.y[c.dev], c.d.t[c.dev])),
        "per_fold_ts_auc": per,
        "fold_std": float(np.std(per)),
    }


def cell_rows(c: Any, rows: np.ndarray, split: str) -> np.ndarray:
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


def pair_repair_stats(
    base: np.ndarray, cand: np.ndarray, c: Any, rows: np.ndarray, pairs_per_t: int, seed: int
) -> dict[str, Any]:
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


def pair_flows(base: np.ndarray, cand: np.ndarray, c: Any) -> dict[str, Any]:
    return {
        s: pair_repair_stats(base, cand, c, cell_rows(c, c.dev, s), PAIRS_PER_T, PAIR_SEED)
        for s in SPLITS
    }


def within_t_rank_corr(a: np.ndarray, b: np.ndarray, t: np.ndarray, rows: np.ndarray) -> float:
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


def verdict(marginal: float, positive_folds: int, pair_vs_e0: dict[str, Any]) -> str:
    dom = int(pair_vs_e0["dominant_cell"]["net_pair_lift"])
    mn = int(pair_vs_e0["mature_vs_never"]["net_pair_lift"])
    if marginal >= 0.0050:
        return "MAJOR"
    if marginal >= 0.0030 and positive_folds >= 4 and dom > 0 and mn > 0:
        return "SERIOUS"
    if marginal >= 0.0015 and positive_folds >= 4 and dom > 0:
        return "PROMOTION_WORTHY"
    if marginal >= 0.0010:
        return "INTERESTING"
    return "KILL"


def ensure_complete_oof(path: Path, c: Any, learner: str) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"MISSING {learner} OOF: {path}")
    v = np.load(path)
    if v.shape != (len(c.d.y),):
        raise SystemExit(f"{learner} OOF shape {v.shape}, expected {(len(c.d.y),)}")
    missing = [int(f) for f in FOLDS if not np.isfinite(v[c.rows[f]]).all()]
    if missing:
        raise SystemExit(
            f"{learner} OOF is incomplete on folds {missing}; resume training before evaluation."
        )
    return v


def evaluate_learner(
    learner: str, c: Any, controls: dict[str, np.ndarray], output_root: Path
) -> dict[str, Any]:
    from wave5_lib import CANON_CAL

    from sbr.metric import ts_auc_flat

    candidate = ensure_complete_oof(output_root / f"{learner}_oof.npy", c, learner)
    P = dict(controls)
    P["__candidate__"] = candidate
    cache = CalibratedFoldCache(c, P, CANON_CAL)
    e0_vec, e0_per = evaluate_fixed(c, cache, list(SPECIALISTS))
    clone_choices = {int(f): select_nested(c, cache, int(f), MATCHED_LGBM) for f in FOLDS}
    cand_choices = {int(f): select_nested(c, cache, int(f), "__candidate__") for f in FOLDS}
    e1_vec, e1_per = evaluate_choices(c, cache, clone_choices)
    e2_vec, e2_per = evaluate_choices(c, cache, cand_choices)
    standalone = score_vec(c, candidate)
    matched = score_vec(c, controls[MATCHED_LGBM])
    rows_dev = c.dev
    dom_rows = cell_rows(c, rows_dev, "dominant_cell")
    fold_deltas = [float(a - b) for a, b in zip(e2_per, e1_per)]
    marginal = float(np.mean(e2_per) - np.mean(e1_per))
    positive = int(sum(x > 0 for x in fold_deltas))
    pair_vs_e0 = pair_flows(e0_vec, e2_vec, c)
    pair_vs_e1 = pair_flows(e1_vec, e2_vec, c)
    out = {
        "learner": LEARNER_LABELS[learner],
        "learner_key": learner,
        "status": "SCORED_AFTER_ALL_FOLDS_COMPLETE",
        "standalone": standalone,
        "matched_lgbm_standalone": matched,
        "standalone_delta_vs_matched_lgbm": float(
            standalone["mean_ts_auc"] - matched["mean_ts_auc"]
        ),
        "dominant_cell_auc": float(
            ts_auc_flat(candidate[dom_rows], c.d.y[dom_rows], c.d.t[dom_rows])
        ),
        "rho_vs_rt401": within_t_rank_corr(candidate, controls[MATCHED_LGBM], c.d.t, rows_dev),
        "pair_flow_vs_rt401_raw": pair_flows(controls[MATCHED_LGBM], candidate, c),
        "ensemble": {
            "E0_original_RT600": {"mean_ts_auc": float(np.mean(e0_per)), "per_fold_ts_auc": e0_per},
            "E1_six_RT600_plus_RT401_clone": {
                "mean_ts_auc": float(np.mean(e1_per)),
                "per_fold_ts_auc": e1_per,
            },
            "E2_six_RT600_plus_neural_candidate": {
                "mean_ts_auc": float(np.mean(e2_per)),
                "per_fold_ts_auc": e2_per,
            },
            "marginal_vs_clone": marginal,
            "E2_minus_E0": float(np.mean(e2_per) - np.mean(e0_per)),
            "fold_deltas_vs_clone": fold_deltas,
            "positive_folds_vs_clone": positive,
            "candidate_choices": [ch.__dict__ for ch in cand_choices.values()],
            "clone_choices": [ch.__dict__ for ch in clone_choices.values()],
            "selection_rule": (
                "outer-fold-pure nested inner-fold mean TS-AUC; "
                "lexicographic tie-break"
            ),
        },
        "pair_flow_vs_E0": pair_vs_e0,
        "pair_flow_vs_E1_clone": pair_vs_e1,
        "verdict": verdict(marginal, positive, pair_vs_e0),
    }
    return out


def write_outputs(result: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    rows = []
    for key, r in result["learners"].items():
        ens = r["ensemble"]
        rows.append(
            {
                "learner": key,
                "label": r["learner"],
                "standalone_mean_ts_auc": r["standalone"]["mean_ts_auc"],
                "standalone_pooled_ts_auc": r["standalone"]["pooled_ts_auc"],
                "rho_vs_rt401": r["rho_vs_rt401"],
                "dominant_cell_auc": r["dominant_cell_auc"],
                "marginal_vs_clone": ens["marginal_vs_clone"],
                "E2_minus_E0": ens["E2_minus_E0"],
                "positive_folds_vs_clone": ens["positive_folds_vs_clone"],
                "dominant_pair_net": r["pair_flow_vs_E0"]["dominant_cell"]["net_pair_lift"],
                "mature_vs_never_net": r["pair_flow_vs_E0"]["mature_vs_never"]["net_pair_lift"],
                "mature_vs_prebreak_net": r["pair_flow_vs_E0"]["mature_vs_prebreak"][
                    "net_pair_lift"
                ],
                "verdict": r["verdict"],
            }
        )
    with RESULTS_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    lines = [
        "# GPU TABULAR 2026 RESULTS",
        "",
        f"Date: {result['date']}",
        f"Branch: `{result['branch']}`",
        f"SHA: `{result['git_sha']}`",
        "",
        "| learner | standalone | rho vs RT-401 | marginal_vs_clone | E2-E0 | "
        "folds+ | dominant net | mature-never net | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in result["learners"].values():
        ens = r["ensemble"]
        lines.append(
            f"| {r['learner']} | {r['standalone']['mean_ts_auc']:.9f} | "
            f"{r['rho_vs_rt401']:+.6f} | {ens['marginal_vs_clone']:+.9f} | "
            f"{ens['E2_minus_E0']:+.9f} | {ens['positive_folds_vs_clone']}/5 | "
            f"{r['pair_flow_vs_E0']['dominant_cell']['net_pair_lift']} | "
            f"{r['pair_flow_vs_E0']['mature_vs_never']['net_pair_lift']} | {r['verdict']} |"
        )
    lines += [
        "",
        "Combination with CatBoost/RT1257 was not run. If either neural learner "
        "survives, that becomes a separate preregistered experiment.",
        "No test, lockbox, production branch, RT ID allocation, or "
        "research/RESULTS.csv update is performed by this evaluator.",
    ]
    FINAL_MD.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--learner", choices=("all", "tabm", "realmlp"), default="all")
    p.add_argument("--artifact-root", default=None)
    p.add_argument("--output-root", default=str(LOCAL_OOF_ROOT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_roots(args.artifact_root)
    c = load_eval_context(args.artifact_root)
    controls = load_control_oof(args.artifact_root)
    learners = ("tabm", "realmlp") if args.learner == "all" else (args.learner,)
    output_root = Path(args.output_root).resolve()
    result = {
        "program": "GPU TABULAR 2026",
        "date": time.strftime("%Y-%m-%d"),
        "branch": git_branch(),
        "git_sha": git_sha(short=False),
        "versions": env_versions(),
        "artifact_root": str(Path(args.artifact_root).resolve()) if args.artifact_root else None,
        "output_root": str(output_root),
        "calibration": "SCDF_NSEEN",
        "matched_lgbm_control": MATCHED_LGBM,
        "rt600_specialists": list(SPECIALISTS),
        "pair_seed": PAIR_SEED,
        "pairs_per_t": PAIRS_PER_T,
        "learners": {},
        "combination_with_catboost_or_rt1257": {
            "run": False,
            "reason": "separate preregistered experiment only if a neural learner survives",
        },
    }
    for learner in learners:
        result["learners"][learner] = evaluate_learner(learner, c, controls, output_root)
    write_outputs(result)
    print(
        json.dumps(
            {
                "results_json": str(RESULTS_JSON),
                "results_csv": str(RESULTS_CSV),
                "final_md": str(FINAL_MD),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
