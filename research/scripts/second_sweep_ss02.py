#!/usr/bin/env python3
"""SS-02 dominant-cell residual ranker execution.

Implements the frozen protocol in
research/reports/new_avenues_2026/second_sweep/SS02_EXECUTION_PREREG.md.
No core sbr files are edited by this script.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("SBR_ROOT", Path(__file__).resolve().parents[2]))
os.environ.setdefault("SBR_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(ROOT / "research" / "scripts" / "novel_streams"))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import sbr.pipeline as PL  # noqa: E402
from harness import rt600_blend  # noqa: E402
from second_sweep_ss01 import (  # noqa: E402
    append_result_append_only,
    base_from_cal,
    calibrated_for_outer,
    concat_rows,
    evaluate_stream,
    finite_float,
    git_sha,
    logit01,
    pair_flow_pack,
    peak_rss,
    rt600_sentinel,
)
from sbr.metric import ts_auc_flat  # noqa: E402
from wave5_lib import Ctx, FOLDS, SPECIALISTS, load_oof  # noqa: E402

PREREG = ROOT / "research" / "reports" / "new_avenues_2026" / "second_sweep" / "SS02_EXECUTION_PREREG.md"
OUTDIR = ROOT / "research" / "reports" / "new_avenues_2026" / "second_sweep"
OOFDIR = ROOT / "research" / "oof"

EXP_ID = "RT-1223"
SHUFFLED_ID = "RT-1224"
CONTROL_IDS = (SHUFFLED_ID,)
SEEDCLONE = "RT-401"
EXECUTION_PREREG_SHA = "3b39954"

FULL_MODULES = ["m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes"]
SCORE_STATE_FEATURE_COUNT = 25
FORBIDDEN_TOKENS = (
    "tau",
    "cut",
    "boundary",
    "break_at",
    "changepoint_true",
    "has_break",
    "n_online",
    "n_hist",
    "final_row",
    "eligible",
    "availability",
    "pad",
)

TRAIN_ROW_SEED = 2026082505
TRAIN_PAIR_SEED = 2026082504
SHUFFLE_SEED = 2026082506
MAX_TRAIN_ROWS = 800_000
TRAIN_PAIRS_PER_T = 128
EVAL_PAIRS_PER_T = 64
DOMINANT_T_MIN = 200
DOMINANT_AGE_MIN = 100
DAMAGE_PENALTY = 2.0
DOMINANT_PAIR_WEIGHT = 3.0
CORRECTION_SCALE = 0.10
RAW_CLIP = 0.50
SCREEN_GATE_MARGIN = 0.0015
SCREEN_GATE_DOMINANT_NET = 300
CONTROL_MATCH_GAP = 0.0005

LGB_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "num_threads": 2,
    "max_bin": 127,
    "seed": 20260825,
    "verbose": -1,
}
NUM_BOOST_ROUND = 250


def assert_no_forbidden_columns(names: list[str]) -> None:
    bad = [n for n in names if any(tok in n.lower() for tok in FORBIDDEN_TOKENS)]
    if bad:
        raise SystemExit(f"forbidden column(s) reachable by SS-02: {bad[:20]}")


def load_feature_bank() -> tuple[list[np.ndarray], list[str], np.ndarray]:
    mats, names = PL.load_features(FULL_MODULES)
    if len(names) != 500:
        raise SystemExit(f"expected 500 feature-bank columns, found {len(names)}")
    assert_no_forbidden_columns(names)
    return mats, names, np.arange(len(names), dtype=np.int32)


def load_scores(c: Ctx) -> tuple[dict[str, np.ndarray], dict]:
    names = list(SPECIALISTS) + [SEEDCLONE]
    scores = load_oof(names)
    dev = c.dev
    lockbox = c.d.rows_for([-1])
    checks = {}
    for name in names:
        x = scores[name]
        checks[name] = {
            "finite_dev_rows": int(np.isfinite(x[dev]).sum()),
            "expected_dev_rows": int(len(dev)),
            "finite_lockbox_rows": int(np.isfinite(x[lockbox]).sum()),
        }
        if checks[name]["finite_dev_rows"] != len(dev):
            raise SystemExit(f"{name} is not finite on all dev rows: {checks[name]}")
        if checks[name]["finite_lockbox_rows"] != 0:
            raise SystemExit(f"{name} unexpectedly fills lockbox rows: {checks[name]}")
    return scores, checks


def sample_outer_train_rows(c: Ctx, outer_fold: int) -> np.ndarray:
    rows = concat_rows(c, [g for g in FOLDS if int(g) != int(outer_fold)])
    if len(rows) <= MAX_TRAIN_ROWS:
        return np.sort(rows)
    rng = np.random.default_rng(TRAIN_ROW_SEED + int(outer_fold))
    return np.sort(rng.choice(rows, MAX_TRAIN_ROWS, replace=False))


def add_score_state_features(c: Ctx, cal: dict[str, np.ndarray], rows: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
    spec = np.column_stack([cal[s][rows] for s in SPECIALISTS]).astype(np.float32, copy=False)
    seed = cal[SEEDCLONE][rows].astype(np.float32, copy=False)
    base = spec.mean(axis=1).astype(np.float32, copy=False)
    t = c.d.t[rows].astype(np.float32, copy=False)
    cols: list[np.ndarray] = []
    names: list[str] = []

    def add(name: str, arr: np.ndarray) -> None:
        names.append(name)
        cols.append(np.asarray(arr, dtype=np.float32))

    for sid, values in zip(SPECIALISTS, spec.T):
        add(f"specialist_{sid.replace('-', '')}_cal", values)
    add("seedclone_RT401_cal", seed)
    add("rt600_cal", base)
    add("rt600_margin_abs", np.abs(base - 0.5))
    add("rt600_logit", logit01(base))
    add("seed_minus_rt600", seed - base)
    add("abs_seed_minus_rt600", np.abs(seed - base))
    add("specialist_std", spec.std(axis=1))
    add("specialist_range", spec.max(axis=1) - spec.min(axis=1))
    add("specialist_min", spec.min(axis=1))
    add("specialist_max", spec.max(axis=1))
    add("specialists_above_rt600_frac", (spec > base[:, None]).mean(axis=1))
    add("specialists_above_half_frac", (spec > 0.5).mean(axis=1))
    add("log1p_t", np.log1p(t))
    for cut in (20, 50, 100, 200, 400):
        add(f"t_ge_{cut}", (t >= cut).astype(np.float32))

    extra = np.column_stack(cols).astype(np.float32, copy=False)
    if not np.isfinite(extra).all():
        raise SystemExit("non-finite SS-02 score-state feature matrix")
    assert_no_forbidden_columns(names)
    return extra, names, base


def make_features(
    c: Ctx,
    mats: list[np.ndarray],
    bank_names: list[str],
    keep_idx: np.ndarray,
    cal: dict[str, np.ndarray],
    rows: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    bank = PL._stack(mats, bank_names, rows, keep_idx)
    extra, extra_names, base = add_score_state_features(c, cal, rows)
    X = np.empty((len(rows), bank.shape[1] + extra.shape[1]), dtype=np.float32)
    X[:, : bank.shape[1]] = bank
    X[:, bank.shape[1] :] = extra
    del bank, extra
    return X, bank_names + extra_names, base


def time_groups(rows: np.ndarray, t: np.ndarray):
    order = np.argsort(t[rows], kind="stable")
    sorted_rows = rows[order]
    sorted_t = t[sorted_rows]
    starts = np.flatnonzero(np.r_[True, sorted_t[1:] != sorted_t[:-1]])
    ends = np.r_[starts[1:], len(sorted_t)]
    for lo, hi in zip(starts, ends):
        yield int(sorted_t[lo]), sorted_rows[lo:hi]


def sample_training_pairs(c: Ctx, cal: dict[str, np.ndarray], train_rows: np.ndarray, outer_fold: int) -> dict:
    row_to_local = np.full(len(c.d.y), -1, dtype=np.int32)
    row_to_local[train_rows] = np.arange(len(train_rows), dtype=np.int32)
    pp_all: list[np.ndarray] = []
    nn_all: list[np.ndarray] = []
    for tt, idx in time_groups(train_rows, c.d.t):
        pos = idx[c.d.y[idx] == 1]
        neg = idx[c.d.y[idx] == 0]
        k = min(TRAIN_PAIRS_PER_T, len(pos), len(neg))
        if k <= 0:
            continue
        rng = np.random.default_rng(TRAIN_PAIR_SEED + 1000 * int(outer_fold) + int(tt))
        pp = rng.choice(pos, k, replace=False)
        nn = rng.choice(neg, k, replace=False)
        pp_all.append(pp.astype(np.int64, copy=False))
        nn_all.append(nn.astype(np.int64, copy=False))
    if not pp_all:
        raise SystemExit(f"outer fold {outer_fold}: no train pairs sampled")
    pp = np.concatenate(pp_all)
    nn = np.concatenate(nn_all)
    pos_local = row_to_local[pp]
    neg_local = row_to_local[nn]
    if (pos_local < 0).any() or (neg_local < 0).any():
        raise SystemExit("pair sampler selected a row outside the training matrix")

    base_p = base_from_cal(cal, pp)
    base_n = base_from_cal(cal, nn)
    base_diff = (base_p - base_n).astype(np.float32, copy=False)
    base_wrong = base_diff <= 0.0
    base_right = ~base_wrong
    weights = np.where(base_wrong, 1.0, DAMAGE_PENALTY).astype(np.float32)
    dominant = (c.d.t[pp] >= DOMINANT_T_MIN) & (c.age[pp] >= DOMINANT_AGE_MIN)
    weights[dominant] *= DOMINANT_PAIR_WEIGHT
    weights = weights / max(float(weights.mean()), 1e-12)
    return {
        "pos_local": pos_local.astype(np.int32, copy=False),
        "neg_local": neg_local.astype(np.int32, copy=False),
        "base_diff": base_diff,
        "weights": weights.astype(np.float32, copy=False),
        "summary": {
            "n_pairs": int(len(pp)),
            "rt600_wrong_pairs": int(base_wrong.sum()),
            "rt600_right_pairs": int(base_right.sum()),
            "dominant_pairs": int(dominant.sum()),
            "mean_pair_weight_after_norm": float(weights.mean()),
            "train_pairs_per_t": TRAIN_PAIRS_PER_T,
            "pair_seed_base": TRAIN_PAIR_SEED,
        },
    }


def make_residual_pair_objective(
    n_rows: int,
    pos: np.ndarray,
    neg: np.ndarray,
    base_diff: np.ndarray,
    weights: np.ndarray,
):
    base_scaled = np.asarray(base_diff, dtype=np.float64) / CORRECTION_SCALE
    w = np.asarray(weights, dtype=np.float64)
    w = w * (float(n_rows) / max(float(w.sum()), 1e-12))

    def obj(preds, dset):
        margin = base_scaled + preds[pos] - preds[neg]
        pr = 1.0 / (1.0 + np.exp(-np.clip(margin, -60, 60)))
        g = -(1.0 - pr) * w
        h = np.maximum(pr * (1.0 - pr), 1e-6) * w
        grad = np.bincount(pos, weights=g, minlength=n_rows) + np.bincount(neg, weights=-g, minlength=n_rows)
        hess = np.bincount(pos, weights=h, minlength=n_rows) + np.bincount(neg, weights=h, minlength=n_rows)
        return grad, np.maximum(hess, 1e-6)

    return obj


def corrected_score(base: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, dict]:
    z = np.clip(np.asarray(raw, dtype=np.float32), -RAW_CLIP, RAW_CLIP)
    score = base.astype(np.float32, copy=True) + CORRECTION_SCALE * z
    clip = np.abs(raw) >= RAW_CLIP
    return score.astype(np.float32, copy=False), {
        "raw_min": float(np.min(raw)),
        "raw_max": float(np.max(raw)),
        "raw_mean": float(np.mean(raw)),
        "raw_std": float(np.std(raw)),
        "clip_fraction": float(clip.mean()),
    }


def fit_one_outer(
    c: Ctx,
    mats: list[np.ndarray],
    bank_names: list[str],
    keep_idx: np.ndarray,
    scores: dict[str, np.ndarray],
    outer_fold: int,
    shuffled: bool = False,
) -> dict:
    cal = calibrated_for_outer(c, scores, outer_fold, list(SPECIALISTS) + [SEEDCLONE])
    train_rows = sample_outer_train_rows(c, outer_fold)
    val_rows = c.rows[int(outer_fold)]
    pair = sample_training_pairs(c, cal, train_rows, outer_fold)
    base_diff = pair["base_diff"]
    weights = pair["weights"]
    pair_summary = dict(pair["summary"])
    if shuffled:
        rng = np.random.default_rng(SHUFFLE_SEED + int(outer_fold))
        perm = rng.permutation(len(base_diff))
        base_diff = base_diff[perm]
        weights = weights[perm]
        pair_summary["shuffled_residual_seed"] = SHUFFLE_SEED + int(outer_fold)

    Xtr, feature_names, _ = make_features(c, mats, bank_names, keep_idx, cal, train_rows)
    params = dict(LGB_PARAMS)
    params["objective"] = make_residual_pair_objective(
        len(train_rows),
        pair["pos_local"],
        pair["neg_local"],
        base_diff,
        weights,
    )
    ds = lgb.Dataset(
        Xtr,
        label=c.d.y[train_rows],
        params=dict(LGB_PARAMS, objective="binary"),
        feature_name=[f"f{i}" for i in range(len(feature_names))],
    )
    booster = lgb.train(params, ds, num_boost_round=NUM_BOOST_ROUND)
    raw_tr = booster.predict(Xtr).astype(np.float32)
    _, train_raw_summary = corrected_score(np.zeros_like(raw_tr), raw_tr)
    del Xtr, ds

    Xva, _, base_va = make_features(c, mats, bank_names, keep_idx, cal, val_rows)
    raw_va = booster.predict(Xva).astype(np.float32)
    score, val_raw_summary = corrected_score(base_va, raw_va)
    del Xva

    imp = np.asarray(booster.feature_importance("gain"), dtype=np.float64)
    return {
        "fold": int(outer_fold),
        "val_rows": val_rows,
        "score": score,
        "pair_summary": pair_summary,
        "train_rows": int(len(train_rows)),
        "train_row_seed": TRAIN_ROW_SEED + int(outer_fold),
        "feature_names": feature_names,
        "feature_importance_gain": imp,
        "train_raw_summary": train_raw_summary,
        "val_raw_summary": val_raw_summary,
    }


def train_residual_oof(
    c: Ctx,
    mats: list[np.ndarray],
    bank_names: list[str],
    keep_idx: np.ndarray,
    scores: dict[str, np.ndarray],
    shuffled: bool = False,
    label: str = "ss02",
) -> dict:
    t0 = time.time()
    oof = np.full(len(c.d.y), np.nan, dtype=np.float32)
    fold_summaries = []
    imp_sum = None
    feature_names = None
    for f in FOLDS:
        fold_start = time.time()
        res = fit_one_outer(c, mats, bank_names, keep_idx, scores, int(f), shuffled=shuffled)
        oof[res["val_rows"]] = res["score"]
        imp_sum = res["feature_importance_gain"] if imp_sum is None else imp_sum + res["feature_importance_gain"]
        feature_names = res["feature_names"]
        fold_summaries.append(
            {
                "fold": int(f),
                "train_rows": res["train_rows"],
                "train_row_seed": res["train_row_seed"],
                "pair_summary": res["pair_summary"],
                "train_raw_summary": res["train_raw_summary"],
                "val_raw_summary": res["val_raw_summary"],
                "runtime_s": float(time.time() - fold_start),
            }
        )
        print(
            f"{label}: fold {f} trained, val clip {res['val_raw_summary']['clip_fraction']:.4f} "
            f"({time.time() - fold_start:.1f}s)",
            flush=True,
        )
    if np.isfinite(oof[c.d.rows_for([-1])]).any():
        raise SystemExit(f"{label}: lockbox rows were filled")
    order = np.argsort(imp_sum)[::-1][:30]
    top = [{"feature": feature_names[int(i)], "gain": float(imp_sum[int(i)])} for i in order if imp_sum[int(i)] > 0]
    return {"oof": oof, "fold_summaries": fold_summaries, "top_feature_importance": top, "runtime_s": float(time.time() - t0)}


def write_oof(exp_id: str, score: np.ndarray) -> str:
    OOFDIR.mkdir(parents=True, exist_ok=True)
    path = OOFDIR / f"{exp_id}.npy"
    np.save(path, score.astype(np.float32, copy=False))
    return str(path)


def control_screen_pass(ev: dict) -> bool:
    pf = ev["pair_flow"]
    margin = ev["ensemble_marginal"]["marginal_vs_clone"]
    dom = pf["dominant_cell"]["net_pair_lift"]
    never = pf["mature_vs_never"]["net_pair_lift"]
    pre = pf["mature_vs_prebreak"]["net_pair_lift"]
    never_ok = (never > 0) or (never == 0 and pre >= SCREEN_GATE_DOMINANT_NET)
    return bool(margin >= SCREEN_GATE_MARGIN and dom >= SCREEN_GATE_DOMINANT_NET and never_ok)


def gate_verdict(candidate: dict, control: dict) -> tuple[str, list[dict]]:
    failures = []
    pf = candidate["pair_flow"]
    margin = candidate["ensemble_marginal"]["marginal_vs_clone"]
    dom = pf["dominant_cell"]
    never = pf["mature_vs_never"]["net_pair_lift"]
    pre = pf["mature_vs_prebreak"]["net_pair_lift"]
    if margin < SCREEN_GATE_MARGIN:
        failures.append({"gate": "marginal_vs_clone", "threshold": SCREEN_GATE_MARGIN, "observed": margin})
    if dom["net_pair_lift"] < SCREEN_GATE_DOMINANT_NET:
        failures.append({"gate": "dominant_net_pair_lift", "threshold": SCREEN_GATE_DOMINANT_NET, "observed": dom["net_pair_lift"]})
    if not ((never > 0) or (never == 0 and pre >= SCREEN_GATE_DOMINANT_NET)):
        failures.append(
            {
                "gate": "mature_vs_never_or_prebreak_pair_flow",
                "threshold": "mature_vs_never > 0 OR mature_vs_never == 0 and mature_vs_prebreak >= 300",
                "observed": {"mature_vs_never": never, "mature_vs_prebreak": pre},
            }
        )
    cm = control["ensemble_marginal"]["marginal_vs_clone"]
    if margin - cm < CONTROL_MATCH_GAP:
        failures.append(
            {
                "gate": f"{SHUFFLED_ID}_control_gap",
                "threshold": f"candidate - control >= {CONTROL_MATCH_GAP}",
                "observed_candidate_minus_control": margin - cm,
                "control_marginal_vs_clone": cm,
            }
        )
    if control_screen_pass(control):
        failures.append({"gate": f"{SHUFFLED_ID}_independent_screen", "threshold": "control must not pass SS-02 screen", "observed": True})
    return ("KILL" if failures else "PASS"), failures


def write_json(result: dict) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with (OUTDIR / "ss02_residual_ranker.json").open("w") as f:
        json.dump(finite_float(result), f, indent=2, sort_keys=True)
        f.write("\n")


def markdown_pair_table(ev: dict) -> list[str]:
    rows = [
        "| split | pairs | RT600 wrong | RT600 right | repairs | damage | net | damage rate on RT600-right |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in ev["pair_flow"].items():
        dr = row["damage_rate_of_rt600_right"]
        rows.append(
            f"| `{name}` | {row['total_pairs_sampled']} | {row['rt600_wrong']} | {row['rt600_right']} | "
            f"{row['repairs']} | {row['damage']} | {row['net_pair_lift']} | "
            f"{dr:.4f} |" if dr is not None else
            f"| `{name}` | {row['total_pairs_sampled']} | {row['rt600_wrong']} | {row['rt600_right']} | "
            f"{row['repairs']} | {row['damage']} | {row['net_pair_lift']} | n/a |"
        )
    return rows


def write_markdown(result: dict) -> None:
    cand = result[EXP_ID]
    ctrl = result["controls"][SHUFFLED_ID]
    cm = cand["ensemble_marginal"]
    sm = ctrl["ensemble_marginal"]
    dom = cand["pair_flow"]["dominant_cell"]
    md = [
        "# SS-02 -- Dominant-Cell Residual Ranker",
        "",
        f"Execution preregistration: `research/reports/new_avenues_2026/second_sweep/SS02_EXECUTION_PREREG.md` at `{result['execution_prereg_sha']}`.",
        f"Experiment IDs: `{EXP_ID}` candidate, `{SHUFFLED_ID}` shuffled residual-offset/weight control.",
        "",
        "## RT-600 Sentinel",
        "",
        f"* Dev mean TS-AUC: `{result['rt600_sentinel']['mean']:.6f}`.",
        f"* Dev pooled TS-AUC: `{result['rt600_sentinel']['pooled']:.6f}`.",
        f"* Dev dominant-cell TS-AUC: `{result['rt600_sentinel']['dominant_cell']:.6f}`.",
        f"* Fold-0 E0 RT600: `{result['rt600_sentinel']['fold0_e0']:.6f}`.",
        f"* Fold-0 E1 RT600 + RT-401: `{result['rt600_sentinel']['fold0_e1_seedclone']:.6f}`.",
        "",
        "## Binding Result",
        "",
        "| arm | standalone fold-0 TS-AUC | E2 TS-AUC | marginal vs clone | gain vs RT600 |",
        "|---|---:|---:|---:|---:|",
        f"| RT600 |  | {cm['rt600_7stream']:.6f} |  |  |",
        f"| RT600 + RT-401 seed clone |  | {cm['rt600_plus_seedclone']:.6f} |  |  |",
        f"| RT600 + {EXP_ID} | {cand['fold0_standalone_ts_auc']:.6f} | {cm[f'rt600_plus_{EXP_ID}']:.6f} | {cm['marginal_vs_clone']:+.6f} | {cm['gain_vs_base']:+.6f} |",
        f"| RT600 + {SHUFFLED_ID} | {ctrl['fold0_standalone_ts_auc']:.6f} | {sm[f'rt600_plus_{SHUFFLED_ID}']:.6f} | {sm['marginal_vs_clone']:+.6f} | {sm['gain_vs_base']:+.6f} |",
        "",
        f"Verdict: **{result['verdict']}**.",
        "Gate failures: " + "; ".join(
            f"`{x['gate']}` observed `{x.get('observed', x.get('observed_candidate_minus_control'))}`"
            for x in result["gate_failures"]
        ),
        "",
        "## Candidate Pair Flow",
        "",
        *markdown_pair_table(cand),
        "",
        "## Shuffled Control Pair Flow",
        "",
        *markdown_pair_table(ctrl),
        "",
        "## Training Summary",
        "",
        "| arm | fold | train rows | pairs | RT600 wrong | RT600 right | dominant pairs | val clip frac |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for eid, fit in ((EXP_ID, cand["fit"]), (SHUFFLED_ID, ctrl["fit"])):
        for row in fit["fold_summaries"]:
            ps = row["pair_summary"]
            md.append(
                f"| `{eid}` | {row['fold']} | {row['train_rows']} | {ps['n_pairs']} | "
                f"{ps['rt600_wrong_pairs']} | {ps['rt600_right_pairs']} | {ps['dominant_pairs']} | "
                f"{row['val_raw_summary']['clip_fraction']:.4f} |"
            )
    md += [
        "",
        "## Top Feature Importance",
        "",
        "| rank | feature | gain |",
        "|---:|---|---:|",
    ]
    for i, row in enumerate(cand["fit"]["top_feature_importance"][:20], start=1):
        md.append(f"| {i} | `{row['feature']}` | {row['gain']:.1f} |")
    md += [
        "",
        "## Implementation Audits",
        "",
        f"* Feature-bank columns: `{result['feature_bank']['n_features']}`.",
        f"* Score-state covariates: `{result['feature_bank']['score_state_features']}`.",
        f"* Pair-sample reproduction: `{result['pair_sample_reproduction']}`.",
        "* Specialist and seed-clone coverage: all frozen OOF scores finite on dev rows and zero finite on lockbox rows.",
        f"* New OOF lockbox finite counts: `{result['new_oof_lockbox_finite_counts']}`.",
        f"* Peak RSS: `{result['peak_rss']}`.",
        f"* Runtime: `{result['runtime_s']:.1f}s`.",
        "",
        "## Interpretation",
        "",
        result["interpretation"],
    ]
    (OUTDIR / "ss02_residual_ranker.md").write_text("\n".join(md) + "\n")


def result_row(
    exp_id: str,
    status: str,
    notes: str,
    runtime_s: float,
    n_features: int,
    train_rows: int,
) -> dict:
    return {
        "experiment_id": exp_id,
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "git_sha": git_sha(),
        "agent": "codex-second-sweep",
        "hypothesis": "SS-02 dominant-cell residual pair ranker over the incumbent 500-feature bank.",
        "falsification_condition": "KILL if any SS-02 mandatory fold-0 screen gate fails; shuffled residual control must lose.",
        "feature_set": ",".join(FULL_MODULES) + ",RT600_specialists,RT-401",
        "n_features": n_features,
        "model": "lgbm_bounded_residual_pair_correction",
        "objective": "residual_pair_logistic" if exp_id == EXP_ID else "shuffled_residual_pair_logistic",
        "folds": "0",
        "random_seed": 20260825,
        "train_series": 8000,
        "train_rows": train_rows,
        "mean_oof_ts_auc": "",
        "pooled_oof_ts_auc": "",
        "per_fold_ts_auc": "",
        "fold_std": 0.0,
        "persistence": "none",
        "sample_mode": f"{TRAIN_PAIRS_PER_T}_train_pairs_per_t_{EVAL_PAIRS_PER_T}_eval_pairs_per_t",
        "training_runtime_s": round(runtime_s, 1),
        "causal_verified": "existing 500 causal feature bank + frozen OOF score covariates; SCDF_NSEEN outer/inner fold-pure calibration; no first-sweep sensors; no validation ranks",
        "test_reduced_touched": "no",
        "lockbox_touched": "no",
        "status": status,
        "notes": notes,
        "protocol": "second_sweep_ss02_fold0_screen",
    }


def append_rows(c: Ctx, rows: list[dict], scores: dict[str, np.ndarray]) -> None:
    existing = set()
    with (ROOT / "research" / "RESULTS.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            existing.add((row.get("experiment_id") or "").strip())
    for row in rows:
        eid = row["experiment_id"]
        if eid in existing:
            raise SystemExit(f"RESULTS.csv already contains {eid}")
        sc = scores[eid]
        r = c.rows[0]
        auc = float(ts_auc_flat(sc[r], c.d.y[r], c.d.t[r]))
        row["mean_oof_ts_auc"] = auc
        row["pooled_oof_ts_auc"] = auc
        row["per_fold_ts_auc"] = f"{auc:.5f}"
        append_result_append_only(row)
        existing.add(eid)


def preflight() -> None:
    c = Ctx()
    base = rt600_blend(c)
    sentinel = rt600_sentinel(c, base)
    scores, score_checks = load_scores(c)
    mats, names, _ = load_feature_bank()
    pair_repro = pair_flow_pack(c, base, base, fold=0)["dominant_cell"]
    out = {
        "git_sha": git_sha(),
        "branch": "research/new-avenues-pilots-2026",
        "prereg_exists": PREREG.exists(),
        "rt600_sentinel": sentinel,
        "score_coverage": score_checks,
        "feature_bank": {"modules": FULL_MODULES, "n_features": len(names), "matrix_shapes": [list(m.shape) for m in mats]},
        "forbidden_first_sweep_ids_loaded": [],
        "dominant_pair_sample_self": pair_repro,
    }
    print(json.dumps(finite_float(out), indent=2, sort_keys=True))


def run_score() -> None:
    t0 = time.time()
    if not PREREG.exists():
        raise SystemExit(f"missing execution preregistration: {PREREG}")
    c = Ctx()
    base = rt600_blend(c)
    sentinel = rt600_sentinel(c, base)
    scores, score_checks = load_scores(c)
    mats, names, keep_idx = load_feature_bank()

    print(f"training {EXP_ID} residual ranker", flush=True)
    candidate_fit = train_residual_oof(c, mats, names, keep_idx, scores, shuffled=False, label=EXP_ID)
    write_oof(EXP_ID, candidate_fit["oof"])

    print(f"training {SHUFFLED_ID} shuffled residual control", flush=True)
    shuffled_fit = train_residual_oof(c, mats, names, keep_idx, scores, shuffled=True, label=SHUFFLED_ID)
    write_oof(SHUFFLED_ID, shuffled_fit["oof"])

    candidate_eval = evaluate_stream(c, base, candidate_fit["oof"], EXP_ID, candidate_fit["runtime_s"])
    control_eval = evaluate_stream(c, base, shuffled_fit["oof"], SHUFFLED_ID, shuffled_fit["runtime_s"])
    verdict, failures = gate_verdict(candidate_eval, control_eval)

    if verdict == "KILL":
        interpretation = (
            "SS-02 is KILL under the preregistered screen. A bounded correction "
            "trained against RT600 residual same-t pair errors over the incumbent "
            "500-feature bank did not satisfy all mandatory marginal, pair-flow, "
            "and shuffled-control gates."
        )
    else:
        interpretation = (
            "SS-02 passes the fold-0 screen. Per the program preregistration, "
            "breadth execution must stop and the next step is the SS-02 5-fold "
            "confirmation."
        )

    # Feature count for the ledger is the declared 500 bank plus frozen score-state covariates.
    ledger_n_features = len(names) + SCORE_STATE_FEATURE_COUNT
    result = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "branch": "research/new-avenues-pilots-2026",
        "execution_prereg_sha": EXECUTION_PREREG_SHA,
        "head_sha": git_sha(),
        "exp_ids": [EXP_ID, SHUFFLED_ID],
        "constants": {
            "max_train_rows": MAX_TRAIN_ROWS,
            "train_row_seed": TRAIN_ROW_SEED,
            "train_pair_seed": TRAIN_PAIR_SEED,
            "shuffle_seed": SHUFFLE_SEED,
            "train_pairs_per_t": TRAIN_PAIRS_PER_T,
            "eval_pairs_per_t": EVAL_PAIRS_PER_T,
            "damage_penalty": DAMAGE_PENALTY,
            "dominant_pair_weight": DOMINANT_PAIR_WEIGHT,
            "correction_scale": CORRECTION_SCALE,
            "raw_clip": RAW_CLIP,
            "screen_gate_margin": SCREEN_GATE_MARGIN,
            "screen_gate_dominant_net": SCREEN_GATE_DOMINANT_NET,
            "control_match_gap": CONTROL_MATCH_GAP,
        },
        "rt600_sentinel": sentinel,
        "score_coverage": score_checks,
        "feature_bank": {
            "modules": FULL_MODULES,
            "n_features": len(names),
            "score_state_features": SCORE_STATE_FEATURE_COUNT,
            "ledger_n_features": ledger_n_features,
            "forbidden_first_sweep_ids_loaded": [],
        },
        "pair_sample_reproduction": {
            "dominant_pairs": pair_flow_pack(c, base, base, fold=0)["dominant_cell"]["total_pairs_sampled"],
            "dominant_rt600_wrong": pair_flow_pack(c, base, base, fold=0)["dominant_cell"]["rt600_wrong"],
            "dominant_rt600_right": pair_flow_pack(c, base, base, fold=0)["dominant_cell"]["rt600_right"],
        },
        EXP_ID: {
            **candidate_eval,
            "fit": {
                "runtime_s": candidate_fit["runtime_s"],
                "fold_summaries": candidate_fit["fold_summaries"],
                "top_feature_importance": candidate_fit["top_feature_importance"],
                "oof_artifact": str(OOFDIR / f"{EXP_ID}.npy"),
            },
        },
        "controls": {
            SHUFFLED_ID: {
                **control_eval,
                "fit": {
                    "runtime_s": shuffled_fit["runtime_s"],
                    "fold_summaries": shuffled_fit["fold_summaries"],
                    "top_feature_importance": shuffled_fit["top_feature_importance"],
                    "oof_artifact": str(OOFDIR / f"{SHUFFLED_ID}.npy"),
                },
            }
        },
        "verdict": verdict,
        "gate_failures": failures,
        "interpretation": interpretation,
        "new_oof_lockbox_finite_counts": {
            EXP_ID: int(np.isfinite(candidate_fit["oof"][c.d.rows_for([-1])]).sum()),
            SHUFFLED_ID: int(np.isfinite(shuffled_fit["oof"][c.d.rows_for([-1])]).sum()),
        },
        "peak_rss": peak_rss(),
        "runtime_s": float(time.time() - t0),
    }
    write_json(result)
    write_markdown(result)

    rows = [
        result_row(EXP_ID, "recorded", f"SS-02 candidate verdict {verdict}.", candidate_fit["runtime_s"], ledger_n_features, candidate_fit["fold_summaries"][0]["train_rows"]),
        result_row(SHUFFLED_ID, "control", "SS-02 shuffled residual-offset/weight control.", shuffled_fit["runtime_s"], ledger_n_features, shuffled_fit["fold_summaries"][0]["train_rows"]),
    ]
    append_rows(c, rows, {EXP_ID: candidate_fit["oof"], SHUFFLED_ID: shuffled_fit["oof"]})
    print(
        json.dumps(
            finite_float(
                {
                    "verdict": verdict,
                    "gate_failures": failures,
                    "marginal_vs_clone": candidate_eval["ensemble_marginal"]["marginal_vs_clone"],
                    "dominant_pair_flow": candidate_eval["pair_flow"]["dominant_cell"],
                    "mature_vs_never": candidate_eval["pair_flow"]["mature_vs_never"],
                    "mature_vs_prebreak": candidate_eval["pair_flow"]["mature_vs_prebreak"],
                    "control_marginal_vs_clone": control_eval["ensemble_marginal"]["marginal_vs_clone"],
                    "runtime_s": result["runtime_s"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("preflight", "score"), default="preflight")
    args = ap.parse_args()
    if args.mode == "preflight":
        preflight()
    else:
        run_score()


if __name__ == "__main__":
    main()
