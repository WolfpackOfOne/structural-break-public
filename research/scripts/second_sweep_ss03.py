#!/usr/bin/env python3
"""SS-03 negative-side null calibrator execution.

Implements the frozen protocol in
research/reports/new_avenues_2026/second_sweep/SS03_EXECUTION_PREREG.md.
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

import numpy as np  # noqa: E402

from second_sweep_ss01 import (  # noqa: E402
    append_result_append_only,
    calibrated_for_outer,
    concat_rows,
    evaluate_stream,
    finite_float,
    git_sha,
    pair_flow_pack,
    peak_rss,
    rt600_sentinel,
)
from sbr.metric import ts_auc_flat  # noqa: E402
from wave4_cal import SCDF_NSEEN  # noqa: E402
from wave5_lib import Ctx, FOLDS, SPECIALISTS, load_oof  # noqa: E402
from harness import rt600_blend  # noqa: E402

PREREG = ROOT / "research" / "reports" / "new_avenues_2026" / "second_sweep" / "SS03_EXECUTION_PREREG.md"
OUTDIR = ROOT / "research" / "reports" / "new_avenues_2026" / "second_sweep"
OOFDIR = ROOT / "research" / "oof"
FEATDIR = ROOT / "cache" / "features"

EXP_ID = "RT-1225"
GLOBAL_ID = "RT-1226"
DERANGED_ID = "RT-1227"
UNWEIGHTED_ID = "RT-1228"
SCALAR_ID = "RT-1229"
CONTROL_IDS = (GLOBAL_ID, DERANGED_ID, UNWEIGHTED_ID, SCALAR_ID)
PILOT9_SCALAR_ID = "RT-1212"

EXECUTION_PREREG_SHA = "8a5f41a"
SEEDCLONE = "RT-401"
MATURE_T_MIN = 200
MIN_STATE_NEG_ROWS = 5_000
CORRECTION_SCALE = 0.20
OFFSET_CLIP = 0.25
PAIR_EVAL_SEED = 20260825
DERANGE_SEED = 2026082507

SCREEN_GATE_MARGIN = 0.0010
DERANGED_CONTROL_GAP = 0.0005
PREBREAK_NET_CAP = -150
PREBREAK_DAMAGE_RATE_CAP = 0.0150

CTM_FEATURES = {
    "weighted": ("m18_wctm", "wctm_tail_log"),
    "unweighted": ("m18_uctm", "uctm_tail_log"),
}

FEATURE_SETS = {
    EXP_ID: "RT600_specialists,m18_uctm::uctm_tail_log,m18_wctm::wctm_tail_log,t",
    GLOBAL_ID: "RT600_specialists,t",
    DERANGED_ID: "RT600_specialists,m18_uctm::uctm_tail_log,m18_wctm::wctm_tail_log,t",
    UNWEIGHTED_ID: "RT600_specialists,m18_uctm::uctm_tail_log,t",
    SCALAR_ID: "RT600_specialists,RT-1212,t",
}
FEATURE_COUNTS = {
    EXP_ID: 10,
    GLOBAL_ID: 8,
    DERANGED_ID: 10,
    UNWEIGHTED_ID: 9,
    SCALAR_ID: 9,
}


def load_required_scores(c: Ctx) -> tuple[dict[str, np.ndarray], dict]:
    names = list(SPECIALISTS) + [SEEDCLONE, PILOT9_SCALAR_ID]
    scores = load_oof(names)
    dev = c.dev
    lockbox = c.d.rows_for([-1])
    checks = {}
    for name, x in scores.items():
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


def load_ctm_features(c: Ctx) -> tuple[dict[str, np.ndarray], dict]:
    dev = c.dev
    lockbox = c.d.rows_for([-1])
    out: dict[str, np.ndarray] = {}
    checks: dict[str, dict] = {}
    for key, (module, col) in CTM_FEATURES.items():
        cols_path = FEATDIR / f"{module}.cols.json"
        npy_path = FEATDIR / f"{module}.npy"
        if not cols_path.exists() or not npy_path.exists():
            raise SystemExit(f"missing CTM feature cache for {module}")
        meta = json.loads(cols_path.read_text())
        cols = list(meta["cols"])
        if col not in cols:
            raise SystemExit(f"{module} cache does not contain {col}")
        arr = np.load(npy_path, mmap_mode="r")
        idx = cols.index(col)
        vec = np.asarray(arr[:, idx], dtype=np.float32)
        checks[key] = {
            "module": module,
            "column": col,
            "shape": [int(arr.shape[0]), int(arr.shape[1])],
            "finite_dev_rows": int(np.isfinite(vec[dev]).sum()),
            "expected_dev_rows": int(len(dev)),
            "finite_lockbox_rows": int(np.isfinite(vec[lockbox]).sum()),
        }
        if checks[key]["finite_dev_rows"] != len(dev):
            raise SystemExit(f"{module}::{col} is not finite on all dev rows: {checks[key]}")
        if checks[key]["finite_lockbox_rows"] != 0:
            raise SystemExit(f"{module}::{col} unexpectedly fills lockbox rows: {checks[key]}")
        out[key] = vec
    out["weighted_suppression"] = out["unweighted"] - out["weighted"]
    return out, checks


def score_state(cal: dict[str, np.ndarray], rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = np.column_stack([cal[s][rows] for s in SPECIALISTS]).astype(np.float32, copy=False)
    base = spec.mean(axis=1).astype(np.float32, copy=False)
    dispersion = spec.std(axis=1).astype(np.float32, copy=False)
    scalar = cal[PILOT9_SCALAR_ID][rows].astype(np.float32, copy=False)
    if not np.isfinite(base).all() or not np.isfinite(dispersion).all() or not np.isfinite(scalar).all():
        raise SystemExit("non-finite SS-03 score state")
    return base, dispersion, scalar


def fold_thresholds(
    c: Ctx,
    train_rows: np.ndarray,
    dispersion: np.ndarray,
    scalar: np.ndarray,
    ctm: dict[str, np.ndarray],
) -> dict:
    mature = c.d.t[train_rows] >= MATURE_T_MIN
    if int(mature.sum()) < 100:
        raise SystemExit("too few mature training rows for SS-03 thresholds")
    return {
        "dispersion_median": float(np.median(dispersion[mature])),
        "weighted_ctm_suppression_median": float(np.median(ctm["weighted_suppression"][train_rows][mature])),
        "unweighted_ctm_tail_median": float(np.median(ctm["unweighted"][train_rows][mature])),
        "pilot9_scalar_median": float(np.median(scalar[mature])),
        "mature_train_rows": int(mature.sum()),
    }


def partition_ids(
    c: Ctx,
    rows: np.ndarray,
    base: np.ndarray,
    dispersion: np.ndarray,
    scalar: np.ndarray,
    ctm: dict[str, np.ndarray],
    thresholds: dict,
    kind: str,
) -> np.ndarray:
    t = c.d.t[rows]
    mature = t >= MATURE_T_MIN
    pid = np.zeros(len(rows), dtype=np.int16)
    if not np.any(mature):
        return pid
    high_rt600 = base >= 0.50
    high_dispersion = dispersion >= thresholds["dispersion_median"]
    if kind == "weighted":
        state_value = ctm["weighted_suppression"][rows]
        high_state = state_value >= thresholds["weighted_ctm_suppression_median"]
    elif kind == "unweighted":
        state_value = ctm["unweighted"][rows]
        high_state = state_value >= thresholds["unweighted_ctm_tail_median"]
    elif kind == "scalar":
        high_state = scalar >= thresholds["pilot9_scalar_median"]
    else:
        raise ValueError(kind)
    pid[mature] = (
        1
        + high_rt600[mature].astype(np.int16)
        + 2 * high_dispersion[mature].astype(np.int16)
        + 4 * high_state[mature].astype(np.int16)
    )
    return pid


def derange_within_t(c: Ctx, rows: np.ndarray, pid: np.ndarray) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(DERANGE_SEED)
    out = np.array(pid, copy=True)
    audit = {
        "seed": DERANGE_SEED,
        "groups": 0,
        "groups_shifted": 0,
        "groups_single_state": 0,
        "rows": int(len(rows)),
        "rows_changed": 0,
        "fixed_rows_after_shift": 0,
        "same_time_multisets_preserved": True,
    }
    tt = c.d.t[rows]
    for val in np.unique(tt):
        idx = np.flatnonzero(tt == val)
        audit["groups"] += 1
        if len(np.unique(pid[idx])) <= 1:
            audit["groups_single_state"] += 1
            continue
        order = np.lexsort((rows[idx], pid[idx]))
        sorted_idx = idx[order]
        shift = int(rng.integers(1, len(sorted_idx)))
        shifted_vals = np.roll(pid[sorted_idx], shift)
        if not np.array_equal(np.sort(shifted_vals), np.sort(pid[idx])):
            audit["same_time_multisets_preserved"] = False
        out[sorted_idx] = shifted_vals
        audit["groups_shifted"] += 1
    audit["rows_changed"] = int((out != pid).sum())
    audit["fixed_rows_after_shift"] = int((out == pid).sum())
    if not audit["same_time_multisets_preserved"]:
        raise SystemExit("SS-03 derangement failed to preserve same-time state multisets")
    return out, audit


def fit_common_cdfs(base: np.ndarray, t: np.ndarray, y: np.ndarray) -> dict:
    neg = y == 0
    mature_neg = neg & (t >= MATURE_T_MIN)
    if int(neg.sum()) < MIN_STATE_NEG_ROWS:
        raise SystemExit("too few negative rows for global null calibration")
    common = {
        "all": SCDF_NSEEN(base, t),
        "global_null": SCDF_NSEEN(base[neg], t[neg]),
        "mature_null": SCDF_NSEEN(base[mature_neg], t[mature_neg]) if int(mature_neg.sum()) >= MIN_STATE_NEG_ROWS else None,
        "counts": {
            "all_rows": int(len(base)),
            "negative_rows": int(neg.sum()),
            "mature_negative_rows": int(mature_neg.sum()),
        },
    }
    return common


def fit_state_cdfs(base: np.ndarray, t: np.ndarray, y: np.ndarray, pid: np.ndarray, common: dict) -> dict:
    neg = y == 0
    leaf = {}
    leaf_counts = {}
    for p in range(9):
        m = neg & (pid == p)
        leaf_counts[str(p)] = int(m.sum())
        if int(m.sum()) >= MIN_STATE_NEG_ROWS:
            leaf[p] = SCDF_NSEEN(base[m], t[m])

    parents = {}
    parent_counts = {}
    mature = t >= MATURE_T_MIN
    for bit in (0, 1):
        high_rt = np.zeros(len(pid), dtype=bool)
        mp = pid >= 1
        high_rt[mp] = (((pid[mp] - 1) & 1) == bit)
        m = neg & mature & high_rt
        parent_counts[str(bit)] = int(m.sum())
        if int(m.sum()) >= MIN_STATE_NEG_ROWS:
            parents[bit] = SCDF_NSEEN(base[m], t[m])

    return {
        "leaf": leaf,
        "parents": parents,
        "common": common,
        "counts": {
            "leaf_negative_rows": leaf_counts,
            "parent_high_rt600_negative_rows": parent_counts,
            "min_state_negative_rows": MIN_STATE_NEG_ROWS,
        },
    }


def select_state_cdf(maps: dict, pid: int):
    if pid in maps["leaf"]:
        return maps["leaf"][pid], f"leaf_{pid}"
    common = maps["common"]
    if pid == 0:
        return common["global_null"], "global_null"
    bit = int((pid - 1) & 1)
    if bit in maps["parents"]:
        return maps["parents"][bit], f"parent_high_rt600_{bit}"
    if common["mature_null"] is not None:
        return common["mature_null"], "mature_null"
    return common["global_null"], "global_null"


def corrected_score(base: np.ndarray, all_cdf: np.ndarray, null_cdf: np.ndarray) -> tuple[np.ndarray, dict]:
    offset_raw = null_cdf - all_cdf
    offset = np.clip(offset_raw, -OFFSET_CLIP, OFFSET_CLIP)
    score = np.clip(base + CORRECTION_SCALE * offset, 0.0, 1.0).astype(np.float32)
    return score, {
        "offset_raw_min": float(np.min(offset_raw)),
        "offset_raw_max": float(np.max(offset_raw)),
        "offset_raw_mean": float(np.mean(offset_raw)),
        "offset_raw_std": float(np.std(offset_raw)),
        "offset_clip_fraction": float((np.abs(offset_raw) >= OFFSET_CLIP).mean()),
        "score_min": float(np.min(score)),
        "score_max": float(np.max(score)),
        "score_mean": float(np.mean(score)),
    }


def state_calibrated_score(base: np.ndarray, t: np.ndarray, pid: np.ndarray, maps: dict) -> tuple[np.ndarray, dict]:
    all_cdf = maps["common"]["all"](base, t)
    null_cdf = np.empty(len(base), dtype=np.float64)
    source_counts: dict[str, int] = {}
    for p in np.unique(pid):
        m = pid == int(p)
        cdf, source = select_state_cdf(maps, int(p))
        null_cdf[m] = cdf(base[m], t[m])
        source_counts[source] = source_counts.get(source, 0) + int(m.sum())
    score, summary = corrected_score(base, all_cdf, null_cdf)
    summary["fallback_source_counts"] = source_counts
    return score, summary


def global_calibrated_score(base: np.ndarray, t: np.ndarray, common: dict) -> tuple[np.ndarray, dict]:
    all_cdf = common["all"](base, t)
    null_cdf = common["global_null"](base, t)
    score, summary = corrected_score(base, all_cdf, null_cdf)
    summary["fallback_source_counts"] = {"global_null": int(len(base))}
    return score, summary


def bincount_dict(x: np.ndarray) -> dict:
    counts = np.bincount(np.asarray(x, dtype=np.int16), minlength=9)
    return {str(i): int(counts[i]) for i in range(len(counts))}


def build_all_oofs(c: Ctx, scores: dict[str, np.ndarray], ctm: dict[str, np.ndarray]) -> dict:
    t0 = time.time()
    arms = [EXP_ID, GLOBAL_ID, DERANGED_ID, UNWEIGHTED_ID, SCALAR_ID]
    oofs = {eid: np.full(len(c.d.y), np.nan, dtype=np.float32) for eid in arms}
    summaries = {eid: [] for eid in arms}
    names = list(SPECIALISTS) + [PILOT9_SCALAR_ID]
    for f in FOLDS:
        fold_start = time.time()
        train_folds = [int(g) for g in FOLDS if int(g) != int(f)]
        train_rows = concat_rows(c, train_folds)
        val_rows = c.rows[int(f)]
        cal = calibrated_for_outer(c, scores, int(f), names)

        base_tr, disp_tr, scalar_tr = score_state(cal, train_rows)
        base_va, disp_va, scalar_va = score_state(cal, val_rows)
        thresholds = fold_thresholds(c, train_rows, disp_tr, scalar_tr, ctm)
        t_train = c.d.t[train_rows]
        y_train = c.d.y[train_rows]
        t_val = c.d.t[val_rows]
        common = fit_common_cdfs(base_tr, t_train, y_train)

        global_score, global_summary = global_calibrated_score(base_va, t_val, common)
        oofs[GLOBAL_ID][val_rows] = global_score
        summaries[GLOBAL_ID].append(
            {
                "fold": int(f),
                "train_rows": int(len(train_rows)),
                "valid_rows": int(len(val_rows)),
                "thresholds": thresholds,
                "common_counts": common["counts"],
                "score_summary": global_summary,
                "runtime_s": float(time.time() - fold_start),
            }
        )

        for eid, kind, deranged in (
            (EXP_ID, "weighted", False),
            (DERANGED_ID, "weighted", True),
            (UNWEIGHTED_ID, "unweighted", False),
            (SCALAR_ID, "scalar", False),
        ):
            p_train = partition_ids(c, train_rows, base_tr, disp_tr, scalar_tr, ctm, thresholds, kind)
            p_val = partition_ids(c, val_rows, base_va, disp_va, scalar_va, ctm, thresholds, kind)
            derange_audit = None
            p_eval = p_val
            if deranged:
                p_eval, derange_audit = derange_within_t(c, val_rows, p_val)
            maps = fit_state_cdfs(base_tr, t_train, y_train, p_train, common)
            score, score_summary = state_calibrated_score(base_va, t_val, p_eval, maps)
            oofs[eid][val_rows] = score
            summaries[eid].append(
                {
                    "fold": int(f),
                    "kind": kind,
                    "deranged": bool(deranged),
                    "train_rows": int(len(train_rows)),
                    "valid_rows": int(len(val_rows)),
                    "thresholds": thresholds,
                    "train_state_counts": bincount_dict(p_train),
                    "valid_state_counts": bincount_dict(p_val),
                    "eval_state_counts": bincount_dict(p_eval),
                    "map_counts": maps["counts"],
                    "common_counts": common["counts"],
                    "score_summary": score_summary,
                    "derangement_audit": derange_audit,
                    "runtime_s": float(time.time() - fold_start),
                }
            )
        print(f"SS-03 fold {f} emitted all arms ({time.time() - fold_start:.1f}s)", flush=True)

    lockbox = c.d.rows_for([-1])
    for eid, oof in oofs.items():
        if np.isfinite(oof[lockbox]).any():
            raise SystemExit(f"{eid} produced lockbox predictions")
        if int(np.isfinite(oof[c.dev]).sum()) != int(len(c.dev)):
            raise SystemExit(f"{eid} did not fill all dev rows")
    return {
        "oofs": oofs,
        "summaries": summaries,
        "runtime_s": float(time.time() - t0),
    }


def write_oof(exp_id: str, score: np.ndarray) -> str:
    OOFDIR.mkdir(parents=True, exist_ok=True)
    path = OOFDIR / f"{exp_id}.npy"
    np.save(path, score.astype(np.float32, copy=False))
    return str(path)


def gate_verdict(candidate: dict, controls: dict, audits_ok: bool) -> tuple[str, list[dict]]:
    failures = []
    margin = candidate["ensemble_marginal"]["marginal_vs_clone"]
    never_net = candidate["pair_flow"]["mature_vs_never"]["net_pair_lift"]
    pre = candidate["pair_flow"]["mature_vs_prebreak"]
    pre_net = pre["net_pair_lift"]
    pre_damage_rate = pre["damage_rate_of_rt600_right"]
    if never_net <= 0:
        failures.append({"gate": "never_break_dominant_pair_net", "threshold": ">0", "observed": never_net})
    if margin < SCREEN_GATE_MARGIN:
        failures.append({"gate": "marginal_vs_clone", "threshold": SCREEN_GATE_MARGIN, "observed": margin})
    if pre_net < PREBREAK_NET_CAP:
        failures.append({"gate": "prebreak_net_pair_cap", "threshold": PREBREAK_NET_CAP, "observed": pre_net})
    if pre_damage_rate is None or pre_damage_rate > PREBREAK_DAMAGE_RATE_CAP:
        failures.append({"gate": "prebreak_damage_rate_cap", "threshold": PREBREAK_DAMAGE_RATE_CAP, "observed": pre_damage_rate})

    der = controls[DERANGED_ID]
    der_margin = der["ensemble_marginal"]["marginal_vs_clone"]
    der_never_net = der["pair_flow"]["mature_vs_never"]["net_pair_lift"]
    if margin - der_margin < DERANGED_CONTROL_GAP:
        failures.append(
            {
                "gate": "deranged_control_marginal_gap",
                "threshold": DERANGED_CONTROL_GAP,
                "observed_candidate_minus_control": margin - der_margin,
                "control_marginal_vs_clone": der_margin,
            }
        )
    if der_never_net >= never_net:
        failures.append(
            {
                "gate": "deranged_control_never_break_net",
                "threshold": "deranged mature-vs-never net < candidate",
                "observed_candidate": never_net,
                "observed_control": der_never_net,
            }
        )

    for eid in (GLOBAL_ID, UNWEIGHTED_ID, SCALAR_ID):
        cm = controls[eid]["ensemble_marginal"]["marginal_vs_clone"]
        if cm >= margin:
            failures.append(
                {
                    "gate": f"{eid}_control_margin_not_lower",
                    "threshold": "control marginal_vs_clone < candidate",
                    "observed_candidate_minus_control": margin - cm,
                    "control_marginal_vs_clone": cm,
                }
            )
    if not audits_ok:
        failures.append({"gate": "causal_fold_purity_lockbox_audit", "threshold": True, "observed": False})
    return ("KILL" if failures else "PASS_SCREEN"), failures


def markdown_pair_table(ev: dict) -> list[str]:
    rows = [
        "| split | pairs | RT600 wrong | RT600 right | repairs | damage | net | damage rate on RT600-right |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in ev["pair_flow"].items():
        dr = row["damage_rate_of_rt600_right"]
        dr_text = f"{dr:.4f}" if dr is not None else "n/a"
        rows.append(
            f"| `{name}` | {row['total_pairs_sampled']} | {row['rt600_wrong']} | {row['rt600_right']} | "
            f"{row['repairs']} | {row['damage']} | {row['net_pair_lift']} | {dr_text} |"
        )
    return rows


def write_json(result: dict) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with (OUTDIR / "ss03_null_calibrator.json").open("w") as f:
        json.dump(finite_float(result), f, indent=2, sort_keys=True)
        f.write("\n")


def write_markdown(result: dict) -> None:
    cand = result[EXP_ID]
    controls = result["controls"]
    cm = cand["ensemble_marginal"]
    md = [
        "# SS-03 -- Negative-Side Null Calibrator",
        "",
        f"Execution preregistration: `research/reports/new_avenues_2026/second_sweep/SS03_EXECUTION_PREREG.md` at `{result['execution_prereg_sha']}`.",
        f"Experiment IDs: `{EXP_ID}` candidate, `{GLOBAL_ID}` global control, `{DERANGED_ID}` deranged partition control, `{UNWEIGHTED_ID}` unweighted CTM control, `{SCALAR_ID}` Pilot-9 scalar control.",
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
        "| arm | standalone fold-0 TS-AUC | E2 TS-AUC | marginal vs clone | gain vs RT600 | mature-vs-never net | mature-vs-prebreak net |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| RT600 |  | {cm['rt600_7stream']:.6f} |  |  |  |  |",
        f"| RT600 + RT-401 seed clone |  | {cm['rt600_plus_seedclone']:.6f} |  |  |  |  |",
        (
            f"| RT600 + {EXP_ID} | {cand['fold0_standalone_ts_auc']:.6f} | "
            f"{cm[f'rt600_plus_{EXP_ID}']:.6f} | {cm['marginal_vs_clone']:+.6f} | "
            f"{cm['gain_vs_base']:+.6f} | {cand['pair_flow']['mature_vs_never']['net_pair_lift']} | "
            f"{cand['pair_flow']['mature_vs_prebreak']['net_pair_lift']} |"
        ),
    ]
    for eid, ctrl in controls.items():
        em = ctrl["ensemble_marginal"]
        md.append(
            f"| RT600 + {eid} | {ctrl['fold0_standalone_ts_auc']:.6f} | "
            f"{em[f'rt600_plus_{eid}']:.6f} | {em['marginal_vs_clone']:+.6f} | "
            f"{em['gain_vs_base']:+.6f} | {ctrl['pair_flow']['mature_vs_never']['net_pair_lift']} | "
            f"{ctrl['pair_flow']['mature_vs_prebreak']['net_pair_lift']} |"
        )
    md += [
        "",
        f"Verdict: **{result['verdict']}**.",
        "Gate failures: "
        + ("; ".join(f"`{x['gate']}` observed `{x.get('observed', x.get('observed_candidate_minus_control', x.get('observed_control')) )}`" for x in result["gate_failures"]) if result["gate_failures"] else "none"),
        "",
        "## Candidate Pair Flow",
        "",
        *markdown_pair_table(cand),
        "",
        "## Control Summary",
        "",
        "| control | marginal vs clone | candidate minus control | mature-vs-never net | dominant net |",
        "|---|---:|---:|---:|---:|",
    ]
    cand_margin = cm["marginal_vs_clone"]
    for eid, ctrl in controls.items():
        em = ctrl["ensemble_marginal"]["marginal_vs_clone"]
        md.append(
            f"| `{eid}` | {em:+.6f} | {cand_margin - em:+.6f} | "
            f"{ctrl['pair_flow']['mature_vs_never']['net_pair_lift']} | "
            f"{ctrl['pair_flow']['dominant_cell']['net_pair_lift']} |"
        )
    md += [
        "",
        "## Fold-0 State Audits",
        "",
        f"* Candidate valid state counts: `{cand['fit']['fold_summaries'][0]['valid_state_counts']}`.",
        f"* Candidate eval fallback counts: `{cand['fit']['fold_summaries'][0]['score_summary']['fallback_source_counts']}`.",
        f"* Candidate thresholds: `{cand['fit']['fold_summaries'][0]['thresholds']}`.",
        f"* Deranged audit: `{controls[DERANGED_ID]['fit']['fold_summaries'][0]['derangement_audit']}`.",
        "",
        "## Implementation Audits",
        "",
        f"* Score coverage: all required frozen OOF scores finite on dev rows and zero finite on lockbox rows.",
        f"* CTM feature coverage: `{result['ctm_feature_coverage']}`.",
        f"* Pair-sample reproduction: `{result['pair_sample_reproduction']}`.",
        f"* New OOF lockbox finite counts: `{result['new_oof_lockbox_finite_counts']}`.",
        f"* Peak RSS: `{result['peak_rss']}`.",
        f"* Runtime: `{result['runtime_s']:.1f}s`.",
        "",
        "## Interpretation",
        "",
        result["interpretation"],
    ]
    (OUTDIR / "ss03_null_calibrator.md").write_text("\n".join(md) + "\n")


def result_row(exp_id: str, status: str, notes: str, runtime_s: float, train_rows: int) -> dict:
    return {
        "experiment_id": exp_id,
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "git_sha": git_sha(),
        "agent": "codex-second-sweep",
        "hypothesis": "SS-03 tests whether fixed null-state conditional calibration of RT600 reduces never-break false positives without pre-break damage.",
        "falsification_condition": "KILL if any SS-03 mandatory fold-0 screen gate fails; deranged partition and matched controls must lose.",
        "feature_set": FEATURE_SETS[exp_id],
        "n_features": FEATURE_COUNTS[exp_id],
        "model": "bounded_conditional_null_scdf",
        "objective": "negative_null_calibration",
        "folds": "0",
        "random_seed": DERANGE_SEED if exp_id == DERANGED_ID else 20260825,
        "train_series": 8000,
        "train_rows": train_rows,
        "mean_oof_ts_auc": "",
        "pooled_oof_ts_auc": "",
        "per_fold_ts_auc": "",
        "fold_std": 0.0,
        "persistence": "none",
        "sample_mode": "64_eval_pairs_per_t",
        "training_runtime_s": round(runtime_s, 1),
        "causal_verified": "frozen causal OOF/CTM covariates + outer-fold negative-null SCDF maps + no validation labels in fold maps + zero lockbox fill",
        "test_reduced_touched": "no",
        "lockbox_touched": "no",
        "status": status,
        "notes": notes,
        "protocol": "second_sweep_ss03_fold0_screen",
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
    if not PREREG.exists():
        raise SystemExit(f"missing execution preregistration: {PREREG}")
    c = Ctx()
    scores, score_checks = load_required_scores(c)
    ctm, ctm_checks = load_ctm_features(c)
    base = rt600_blend(c)
    sentinel = rt600_sentinel(c, base)
    cal = calibrated_for_outer(c, scores, 0, list(SPECIALISTS) + [PILOT9_SCALAR_ID])
    train_rows = concat_rows(c, [g for g in FOLDS if int(g) != 0])
    base_tr, disp_tr, scalar_tr = score_state(cal, train_rows)
    thresholds = fold_thresholds(c, train_rows, disp_tr, scalar_tr, ctm)
    p_weighted = partition_ids(c, train_rows, base_tr, disp_tr, scalar_tr, ctm, thresholds, "weighted")
    common = fit_common_cdfs(base_tr, c.d.t[train_rows], c.d.y[train_rows])
    maps = fit_state_cdfs(base_tr, c.d.t[train_rows], c.d.y[train_rows], p_weighted, common)
    pair_repro = pair_flow_pack(c, base, base, fold=0)["dominant_cell"]
    out = {
        "git_sha": git_sha(),
        "branch": "research/new-avenues-pilots-2026",
        "prereg_exists": PREREG.exists(),
        "rt600_sentinel": sentinel,
        "score_coverage": score_checks,
        "ctm_feature_coverage": ctm_checks,
        "fold0_thresholds": thresholds,
        "fold0_weighted_train_state_counts": bincount_dict(p_weighted),
        "fold0_weighted_map_counts": maps["counts"],
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
    scores, score_checks = load_required_scores(c)
    ctm, ctm_checks = load_ctm_features(c)
    fit = build_all_oofs(c, scores, ctm)
    oofs = fit["oofs"]
    for eid, score in oofs.items():
        write_oof(eid, score)

    evals = {eid: evaluate_stream(c, base, score, eid, fit["runtime_s"]) for eid, score in oofs.items()}
    for eid in evals:
        evals[eid]["fit"] = {
            "runtime_s": fit["runtime_s"],
            "fold_summaries": fit["summaries"][eid],
            "oof_artifact": str(OOFDIR / f"{eid}.npy"),
        }
    controls = {eid: evals[eid] for eid in CONTROL_IDS}
    lock_counts = {eid: int(np.isfinite(score[c.d.rows_for([-1])]).sum()) for eid, score in oofs.items()}
    audits_ok = bool(all(v == 0 for v in lock_counts.values()))
    verdict, failures = gate_verdict(evals[EXP_ID], controls, audits_ok)
    if verdict == "KILL":
        interpretation = (
            "SS-03 is KILL under the preregistered fold-0 screen. The fixed "
            "weighted-CTM null-state conditional SCDF correction did not satisfy "
            "all mandatory never-break, marginal, pre-break-damage, and control "
            "gates. No SS-03b, threshold adjustment, correction-scale change, CTM "
            "retuning, or Pilot-9 scalar variant is authorized."
        )
    else:
        interpretation = (
            "SS-03 passes the fold-0 screen. Per preregistration, breadth execution "
            "must stop and a separate confirmation execution note is required before "
            "any 5-fold confirmation."
        )

    result = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "branch": "research/new-avenues-pilots-2026",
        "execution_prereg_sha": EXECUTION_PREREG_SHA,
        "head_sha": git_sha(),
        "exp_ids": [EXP_ID, *CONTROL_IDS],
        "constants": {
            "mature_t_min": MATURE_T_MIN,
            "min_state_negative_rows": MIN_STATE_NEG_ROWS,
            "correction_scale": CORRECTION_SCALE,
            "offset_clip": OFFSET_CLIP,
            "pair_eval_seed": PAIR_EVAL_SEED,
            "derange_seed": DERANGE_SEED,
            "screen_gate_margin": SCREEN_GATE_MARGIN,
            "deranged_control_gap": DERANGED_CONTROL_GAP,
            "prebreak_net_cap": PREBREAK_NET_CAP,
            "prebreak_damage_rate_cap": PREBREAK_DAMAGE_RATE_CAP,
        },
        "rt600_sentinel": sentinel,
        "score_coverage": score_checks,
        "ctm_feature_coverage": ctm_checks,
        "pair_sample_reproduction": {
            "dominant_pairs": pair_flow_pack(c, base, base, fold=0)["dominant_cell"]["total_pairs_sampled"],
            "dominant_rt600_wrong": pair_flow_pack(c, base, base, fold=0)["dominant_cell"]["rt600_wrong"],
            "dominant_rt600_right": pair_flow_pack(c, base, base, fold=0)["dominant_cell"]["rt600_right"],
        },
        EXP_ID: evals[EXP_ID],
        "controls": controls,
        "verdict": verdict,
        "gate_failures": failures,
        "interpretation": interpretation,
        "new_oof_lockbox_finite_counts": lock_counts,
        "peak_rss": peak_rss(),
        "runtime_s": float(time.time() - t0),
    }
    write_json(result)
    write_markdown(result)

    fold0_train_rows = fit["summaries"][EXP_ID][0]["train_rows"]
    rows = [
        result_row(EXP_ID, "recorded", f"SS-03 candidate verdict {verdict}.", fit["runtime_s"], fold0_train_rows),
        result_row(GLOBAL_ID, "control", "SS-03 global RT600 null-SCDF calibration control.", fit["runtime_s"], fold0_train_rows),
        result_row(DERANGED_ID, "control", "SS-03 deranged weighted-CTM null-state partition control.", fit["runtime_s"], fold0_train_rows),
        result_row(UNWEIGHTED_ID, "control", "SS-03 unweighted CTM state partition control.", fit["runtime_s"], fold0_train_rows),
        result_row(SCALAR_ID, "control", "SS-03 Pilot-9 scalar difficulty state partition control.", fit["runtime_s"], fold0_train_rows),
    ]
    append_rows(c, rows, oofs)
    print(
        json.dumps(
            finite_float(
                {
                    "verdict": verdict,
                    "gate_failures": failures,
                    "marginal_vs_clone": evals[EXP_ID]["ensemble_marginal"]["marginal_vs_clone"],
                    "mature_vs_never_pair_flow": evals[EXP_ID]["pair_flow"]["mature_vs_never"],
                    "mature_vs_prebreak_pair_flow": evals[EXP_ID]["pair_flow"]["mature_vs_prebreak"],
                    "control_margins": {
                        eid: controls[eid]["ensemble_marginal"]["marginal_vs_clone"]
                        for eid in CONTROL_IDS
                    },
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
