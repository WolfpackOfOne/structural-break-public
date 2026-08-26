#!/usr/bin/env python
"""LA-01 specialist replacement salvage.

Artifact-only full five-fold evaluation. The scored arms are:

* RT-1243: nested replacement by RT-731/RT-751.
* RT-1244: nested replacement by RT-401 seed clone.

No model training happens here. Existing OOF streams are calibrated with the
repository's canonical fold-pure SCDF convention.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--artifact-root",
        default=os.environ.get("SBR_ROOT", str(REPO)),
        help="Root containing cache/store, research/folds, and research/oof.",
    )
    p.add_argument("--candidate-id", default="RT-1243")
    p.add_argument("--clone-id", default="RT-1244")
    p.add_argument("--pairs-per-t", type=int, default=64)
    p.add_argument("--pair-seed", type=int, default=20260826)
    p.add_argument("--no-ledger", action="store_true")
    return p.parse_args()


ARGS = parse_args()
os.environ["SBR_ROOT"] = ARGS.artifact_root

for _p in (REPO / "src", REPO / "research" / "scripts"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from sbr.metric import ts_auc_flat  # noqa: E402
from wave5_lib import Ctx, FOLDS, SPECIALISTS, load_oof  # noqa: E402
from wave5_lib import CANON_CAL  # noqa: E402

REPORT_DIR = REPO / "research" / "reports" / "leaderboard_alpha_2026"
RESULTS_CSV = REPO / "research" / "RESULTS.csv"

SEEDCLONE = "RT-401"
CANDIDATES = ("RT-731", "RT-751")
CANDIDATE_LABEL = {"RT-731": "m11_focus", "RT-751": "m12_rdep"}


@dataclass(frozen=True)
class FoldChoice:
    outer_fold: int
    replaced: str
    replacement: str
    inner_mean_ts_auc: float
    inner_fold_ts_auc: tuple[float, ...]


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()


def composition(replaced: str, replacement: str) -> list[str]:
    return [replacement if s == replaced else s for s in SPECIALISTS]


def calibrate_apply(P: dict[str, np.ndarray], streams: list[str], train_rows, eval_rows, t):
    cols = []
    for s in streams:
        f = CANON_CAL(P[s][train_rows], t[train_rows])
        cols.append(f(P[s][eval_rows], t[eval_rows]))
    return np.column_stack(cols).mean(axis=1)


def score_composition_inner(c: Ctx, P: dict[str, np.ndarray], streams: list[str], outer_train_folds):
    per = []
    for g in outer_train_folds:
        train_folds = [x for x in outer_train_folds if x != g]
        train_rows = np.concatenate([c.rows[x] for x in train_folds])
        eval_rows = c.rows[g]
        pred = calibrate_apply(P, streams, train_rows, eval_rows, c.d.t)
        per.append(float(ts_auc_flat(pred, c.d.y[eval_rows], c.d.t[eval_rows])))
    return float(np.mean(per)), tuple(per)


def select_nested(c: Ctx, P: dict[str, np.ndarray], outer_fold: int, replacements: tuple[str, ...]):
    outer_train = tuple(g for g in FOLDS if g != outer_fold)
    scored = []
    for repl in replacements:
        for replaced in SPECIALISTS:
            streams = composition(replaced, repl)
            mean_auc, per = score_composition_inner(c, P, streams, outer_train)
            scored.append((mean_auc, repl, replaced, per))
    # Max score, then deterministic non-discretionary tie resolution.
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    mean_auc, repl, replaced, per = scored[0]
    return FoldChoice(outer_fold, replaced, repl, mean_auc, per)


def evaluate_outer(c: Ctx, P: dict[str, np.ndarray], choices: dict[int, FoldChoice]):
    out = np.full(len(c.d.y), np.nan, dtype=np.float64)
    per = []
    for f in FOLDS:
        train_folds = [g for g in FOLDS if g != f]
        train_rows = np.concatenate([c.rows[g] for g in train_folds])
        eval_rows = c.rows[f]
        ch = choices[f]
        pred = calibrate_apply(P, composition(ch.replaced, ch.replacement), train_rows, eval_rows, c.d.t)
        out[eval_rows] = pred
        per.append(float(ts_auc_flat(pred, c.d.y[eval_rows], c.d.t[eval_rows])))
    return out, per


def evaluate_fixed(c: Ctx, P: dict[str, np.ndarray], streams: list[str]):
    out = np.full(len(c.d.y), np.nan, dtype=np.float64)
    per = []
    for f in FOLDS:
        train_folds = [g for g in FOLDS if g != f]
        train_rows = np.concatenate([c.rows[g] for g in train_folds])
        eval_rows = c.rows[f]
        pred = calibrate_apply(P, streams, train_rows, eval_rows, c.d.t)
        out[eval_rows] = pred
        per.append(float(ts_auc_flat(pred, c.d.y[eval_rows], c.d.t[eval_rows])))
    return out, per


def cell_rows(c: Ctx, rows, split: str):
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


def pair_repair_stats(base, cand, c: Ctx, rows, pairs_per_t: int, seed: int):
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
    rate = float(damage / max(total, 1))
    return {
        "total_pairs_sampled": int(total),
        "repairs": int(repairs),
        "damage": int(damage),
        "net_pair_lift": int(repairs - damage),
        "damage_rate": rate,
    }


def within_t_rank_corr(a, b, t, rows):
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


def score_vec(c: Ctx, v: np.ndarray):
    per = [
        float(ts_auc_flat(v[c.rows[f]], c.d.y[c.rows[f]], c.d.t[c.rows[f]]))
        for f in FOLDS
    ]
    dev = c.dev
    return {
        "mean_ts_auc": float(np.mean(per)),
        "pooled_ts_auc": float(ts_auc_flat(v[dev], c.d.y[dev], c.d.t[dev])),
        "per_fold_ts_auc": per,
        "fold_std": float(np.std(per)),
    }


def append_result_rows(result: dict):
    fieldnames = next(csv.DictReader(open(RESULTS_CSV))).fieldnames
    if fieldnames is None:
        raise RuntimeError("RESULTS.csv has no header")

    def row(exp_id: str, arm_key: str, status: str, notes: str):
        arm = result["arms"][arm_key]
        selected = result["nested_selection"][arm_key]
        return {
            "experiment_id": exp_id,
            "date": "2026-08-26",
            "git_sha": result["git_sha"],
            "agent": "codex-leaderboard-alpha",
            "hypothesis": "LA-01 specialist replacement salvage: useful m11_focus/m12_rdep specialization may have been erased by adding streams instead of replacing redundant RT600 specialists.",
            "falsification_condition": "KILL if nested candidate marginal_vs_clone < +0.0015, positive on <4/5 folds, or dominant pair net <= 0.",
            "feature_set": arm["feature_set"],
            "n_features": "7",
            "model": "frozen_oof_equal_scdf_replacement_blend",
            "objective": "artifact_rescore",
            "folds": "0,1,2,3,4",
            "random_seed": str(result["pair_seed"]),
            "train_series": "",
            "train_rows": "",
            "mean_oof_ts_auc": repr(arm["scores"]["mean_ts_auc"]),
            "pooled_oof_ts_auc": repr(arm["scores"]["pooled_ts_auc"]),
            "per_fold_ts_auc": ";".join(f"{x:.5f}" for x in arm["scores"]["per_fold_ts_auc"]),
            "fold_std": repr(arm["scores"]["fold_std"]),
            "persistence": "none",
            "sample_mode": f"{result['pairs_per_t']}_pairs_per_t_diagnostics",
            "training_runtime_s": repr(result["runtime_s"]),
            "causal_verified": "frozen causal OOF streams; nested outer-fold-pure SCDF replacement selection; no retraining",
            "test_reduced_touched": "no",
            "lockbox_touched": "no",
            "status": status,
            "notes": notes + " Nested choices: " + json.dumps(selected, sort_keys=True),
            "protocol": "leaderboard_alpha_2026_la01_full5_nested_replacement",
        }

    candidate_status = result["verdict"]
    candidate_notes = (
        f"RT-1243 candidate. E0={result['arms']['e0_rt600']['scores']['mean_ts_auc']:.9f}; "
        f"E1={result['arms']['rt1244_seedclone_replacement']['scores']['mean_ts_auc']:.9f}; "
        f"E2={result['arms']['rt1243_candidate_replacement']['scores']['mean_ts_auc']:.9f}; "
        f"marginal_vs_clone={result['primary']['marginal_vs_clone']:+.9f}; "
        f"fold_deltas={result['primary']['fold_deltas_vs_clone']}; "
        f"dominant_net={result['pair_flow']['rt1243_vs_e0']['dominant_cell']['net_pair_lift']}; "
        f"mature_vs_never_net={result['pair_flow']['rt1243_vs_e0']['mature_vs_never']['net_pair_lift']}; "
        f"mature_vs_prebreak_net={result['pair_flow']['rt1243_vs_e0']['mature_vs_prebreak']['net_pair_lift']}."
    )
    clone_notes = (
        "RT-1244 secondary control: nested replacement by exchangeable seed clone RT-401. "
        f"gain_vs_e0={result['arms']['rt1244_seedclone_replacement']['mean_delta_vs_e0']:+.9f}; "
        f"dominant_net={result['pair_flow']['rt1244_vs_e0']['dominant_cell']['net_pair_lift']}."
    )
    rows = [
        row(ARGS.candidate_id, "rt1243_candidate_replacement", candidate_status, candidate_notes),
        row(ARGS.clone_id, "rt1244_seedclone_replacement", "control", clone_notes),
    ]

    with open(RESULTS_CSV, "r+", newline="") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        existing = {r["experiment_id"] for r in csv.DictReader(fh)}
        dup = [r["experiment_id"] for r in rows if r["experiment_id"] in existing]
        if dup:
            raise RuntimeError(f"RESULTS.csv already contains {dup}")
        fh.seek(0, os.SEEK_END)
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def main():
    t0 = time.time()
    c = Ctx()
    needed = list(SPECIALISTS) + [SEEDCLONE] + list(CANDIDATES)
    P = load_oof(needed)
    for name, arr in P.items():
        finite_dev = np.isfinite(arr[c.dev]).sum()
        if finite_dev != len(c.dev):
            raise RuntimeError(f"{name} is not finite on all dev rows: {finite_dev}/{len(c.dev)}")
        lock_rows = c.d.rows_for([-1])
        if lock_rows.size and np.isfinite(arr[lock_rows]).any():
            raise RuntimeError(f"{name} has finite lockbox predictions")

    e0_vec, e0_per = evaluate_fixed(c, P, list(SPECIALISTS))

    cand_choices = {f: select_nested(c, P, f, CANDIDATES) for f in FOLDS}
    clone_choices = {f: select_nested(c, P, f, (SEEDCLONE,)) for f in FOLDS}
    cand_vec, cand_per = evaluate_outer(c, P, cand_choices)
    clone_vec, clone_per = evaluate_outer(c, P, clone_choices)

    arms = {
        "e0_rt600": {
            "feature_set": ",".join(SPECIALISTS),
            "scores": score_vec(c, e0_vec),
        },
        "rt1244_seedclone_replacement": {
            "feature_set": ",".join(sorted(set(SPECIALISTS + [SEEDCLONE]))),
            "scores": score_vec(c, clone_vec),
        },
        "rt1243_candidate_replacement": {
            "feature_set": ",".join(sorted(set(SPECIALISTS + list(CANDIDATES)))),
            "scores": score_vec(c, cand_vec),
        },
    }
    # Preserve the direct per-fold arrays from the nested evaluation.
    arms["e0_rt600"]["scores"]["per_fold_ts_auc"] = e0_per
    arms["rt1244_seedclone_replacement"]["scores"]["per_fold_ts_auc"] = clone_per
    arms["rt1243_candidate_replacement"]["scores"]["per_fold_ts_auc"] = cand_per
    for k in arms:
        s = arms[k]["scores"]
        s["mean_ts_auc"] = float(np.mean(s["per_fold_ts_auc"]))
        s["fold_std"] = float(np.std(s["per_fold_ts_auc"]))

    e0_mean = arms["e0_rt600"]["scores"]["mean_ts_auc"]
    clone_mean = arms["rt1244_seedclone_replacement"]["scores"]["mean_ts_auc"]
    cand_mean = arms["rt1243_candidate_replacement"]["scores"]["mean_ts_auc"]
    arms["rt1244_seedclone_replacement"]["mean_delta_vs_e0"] = float(clone_mean - e0_mean)
    arms["rt1243_candidate_replacement"]["mean_delta_vs_e0"] = float(cand_mean - e0_mean)

    fold_deltas_vs_clone = [float(a - b) for a, b in zip(cand_per, clone_per)]
    fold_deltas_vs_e0 = [float(a - b) for a, b in zip(cand_per, e0_per)]
    clone_fold_deltas_vs_e0 = [float(a - b) for a, b in zip(clone_per, e0_per)]

    rows_dev = c.dev
    split_names = ("whole", "dominant_cell", "mature_vs_never", "mature_vs_prebreak")
    pair_flow_cand = {
        nm: pair_repair_stats(
            e0_vec, cand_vec, c, cell_rows(c, rows_dev, nm), ARGS.pairs_per_t, ARGS.pair_seed
        )
        for nm in split_names
    }
    pair_flow_clone = {
        nm: pair_repair_stats(
            e0_vec, clone_vec, c, cell_rows(c, rows_dev, nm), ARGS.pairs_per_t, ARGS.pair_seed
        )
        for nm in split_names
    }

    dominant_rows = cell_rows(c, rows_dev, "dominant_cell")
    corr = {
        "rt1243_vs_e0_dev": within_t_rank_corr(cand_vec, e0_vec, c.d.t, rows_dev),
        "rt1243_vs_e0_dominant": within_t_rank_corr(cand_vec, e0_vec, c.d.t, dominant_rows),
        "rt1244_vs_e0_dev": within_t_rank_corr(clone_vec, e0_vec, c.d.t, rows_dev),
        "rt1244_vs_e0_dominant": within_t_rank_corr(clone_vec, e0_vec, c.d.t, dominant_rows),
    }

    positive_folds = int(sum(x > 0 for x in fold_deltas_vs_clone))
    dominant_net = pair_flow_cand["dominant_cell"]["net_pair_lift"]
    marginal = float(cand_mean - clone_mean)
    if marginal >= 0.0030 and positive_folds >= 4:
        verdict = "SERIOUS"
    elif marginal >= 0.0015 and positive_folds >= 4 and dominant_net > 0:
        verdict = "WEAK"
    else:
        verdict = "KILL"

    result = {
        "program": "LEADERBOARD ALPHA 2026",
        "experiment": "LA-01 specialist replacement salvage",
        "git_sha": git_sha(),
        "artifact_root": ARGS.artifact_root,
        "candidate_id": ARGS.candidate_id,
        "clone_id": ARGS.clone_id,
        "calibration": CANON_CAL.__name__,
        "pairs_per_t": ARGS.pairs_per_t,
        "pair_seed": ARGS.pair_seed,
        "runtime_s": float(time.time() - t0),
        "arms": arms,
        "nested_selection": {
            "rt1243_candidate_replacement": [
                {
                    "outer_fold": ch.outer_fold,
                    "replaced": ch.replaced,
                    "replacement": ch.replacement,
                    "replacement_label": CANDIDATE_LABEL.get(ch.replacement, "seed_clone"),
                    "inner_mean_ts_auc": ch.inner_mean_ts_auc,
                    "inner_fold_ts_auc": list(ch.inner_fold_ts_auc),
                }
                for ch in cand_choices.values()
            ],
            "rt1244_seedclone_replacement": [
                {
                    "outer_fold": ch.outer_fold,
                    "replaced": ch.replaced,
                    "replacement": ch.replacement,
                    "inner_mean_ts_auc": ch.inner_mean_ts_auc,
                    "inner_fold_ts_auc": list(ch.inner_fold_ts_auc),
                }
                for ch in clone_choices.values()
            ],
        },
        "primary": {
            "marginal_vs_clone": marginal,
            "positive_folds_vs_clone": positive_folds,
            "fold_deltas_vs_clone": fold_deltas_vs_clone,
            "mean_delta_vs_e0": float(cand_mean - e0_mean),
            "fold_deltas_vs_e0": fold_deltas_vs_e0,
            "clone_mean_delta_vs_e0": float(clone_mean - e0_mean),
            "clone_fold_deltas_vs_e0": clone_fold_deltas_vs_e0,
        },
        "pair_flow": {
            "rt1243_vs_e0": pair_flow_cand,
            "rt1244_vs_e0": pair_flow_clone,
        },
        "within_t_rank_corr": corr,
        "verdict": verdict,
        "gate": {
            "marginal_vs_clone_ge_0_0015": bool(marginal >= 0.0015),
            "positive_folds_ge_4": bool(positive_folds >= 4),
            "dominant_pair_net_positive": bool(dominant_net > 0),
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "la01_nested_replacement.json"
    csv_path = REPORT_DIR / "la01_nested_replacement_summary.csv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "mean_ts_auc", "pooled_ts_auc", "per_fold_ts_auc", "delta_vs_e0", "marginal_vs_clone"])
        w.writerow([
            "E0_RT600",
            arms["e0_rt600"]["scores"]["mean_ts_auc"],
            arms["e0_rt600"]["scores"]["pooled_ts_auc"],
            ";".join(f"{x:.9f}" for x in arms["e0_rt600"]["scores"]["per_fold_ts_auc"]),
            "",
            "",
        ])
        w.writerow([
            "RT-1244_seedclone_replacement",
            arms["rt1244_seedclone_replacement"]["scores"]["mean_ts_auc"],
            arms["rt1244_seedclone_replacement"]["scores"]["pooled_ts_auc"],
            ";".join(f"{x:.9f}" for x in arms["rt1244_seedclone_replacement"]["scores"]["per_fold_ts_auc"]),
            arms["rt1244_seedclone_replacement"]["mean_delta_vs_e0"],
            "",
        ])
        w.writerow([
            "RT-1243_candidate_replacement",
            arms["rt1243_candidate_replacement"]["scores"]["mean_ts_auc"],
            arms["rt1243_candidate_replacement"]["scores"]["pooled_ts_auc"],
            ";".join(f"{x:.9f}" for x in arms["rt1243_candidate_replacement"]["scores"]["per_fold_ts_auc"]),
            arms["rt1243_candidate_replacement"]["mean_delta_vs_e0"],
            result["primary"]["marginal_vs_clone"],
        ])

    if not ARGS.no_ledger:
        append_result_rows(result)
    print(json.dumps({
        "json": str(json_path),
        "csv": str(csv_path),
        "verdict": verdict,
        "marginal_vs_clone": marginal,
        "candidate_mean_ts_auc": cand_mean,
        "clone_mean_ts_auc": clone_mean,
        "rt600_mean_ts_auc": e0_mean,
        "runtime_s": result["runtime_s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

