#!/usr/bin/env python3
"""Post-hoc synthesis diagnostics for the New Avenues first sweep.

This script is intentionally descriptive.  It reads existing report JSON,
RESULTS.csv rows, and already-generated OOF arrays.  It does not train, score,
allocate an RT id, append RESULTS.csv, or touch lockbox/test artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# The local environment has optional pandas accelerators compiled against an
# older NumPy ABI.  Disabling them keeps this read-only report script quiet.
sys.modules["numexpr"] = None
sys.modules["bottleneck"] = None
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "research" / "reports" / "new_avenues_2026"
OUT_DIR = REPORT_DIR
OOF_DIR = ROOT / "research" / "oof"
RESULTS_CSV = ROOT / "research" / "RESULTS.csv"
RT600_BLEND = ROOT / "cache" / "novel_streams" / "rt600_blend.npy"

NOTICE = "POSTHOC_DESCRIPTIVE_NOT_PROMOTION_EVIDENCE"
FIRST_SWEEP_IDS = [f"RT-{i}" for i in range(1200, 1219)]
SPECIALISTS = ["RT-300", "RT-410", "RT-411", "RT-412", "RT-413", "RT-414", "RT-415"]
CONTEXT_SERIES = SPECIALISTS + ["RT-401", "RT-995"]


ARM_META: dict[str, dict[str, str]] = {
    "PILOT-1": {
        "report": "pilot01_failure_manifolds.json",
        "pilot": "Pilot 1",
        "family": "specialist/error-manifold diagnostic",
        "mechanism": "failure manifolds and specialist routing headroom",
        "candidate_or_control": "diagnostic",
        "mode": "diagnostic_only",
        "primary_gate": "routing/fingerprint headroom",
        "secondary_gate": "specialist pair-flow decomposition",
        "verdict": "WEAK_DIAGNOSTIC_NO_PROMOTION",
        "failure_mode": "D. NARROW COVERAGE; F. INTEGRATION FAILURE",
    },
    "RT-1200": {
        "report": "pilot02_relay_logic.json",
        "pilot": "Pilot 2",
        "family": "score-state relay",
        "mechanism": "relay score-state transform of RT600 evidence",
        "candidate_or_control": "candidate",
        "mode": "score_transform",
        "primary_gate": "marginal_vs_clone >= +0.0010",
        "secondary_gate": "positive dominant-cell pair flow",
        "verdict": "KILLED",
        "failure_mode": "B. REDUNDANT SIGNAL; C. WRONG PAIR FLOW; F. INTEGRATION FAILURE",
    },
    "RT-1201": {
        "report": "pilot03_im2_dwell.json",
        "pilot": "Pilot 3",
        "family": "IM2/dwell",
        "mechanism": "matched-length run null plus excursion dwell bank",
        "candidate_or_control": "candidate",
        "mode": "direct_score",
        "primary_gate": "marginal_vs_clone >= +0.0010",
        "secondary_gate": "dominant-cell repair without broad damage",
        "verdict": "KILLED",
        "failure_mode": "A. WEAK RAW SIGNAL; C. WRONG PAIR FLOW; D. NARROW COVERAGE",
    },
    "RT-1202": {
        "report": "pilot04_trajectory_geometry.json",
        "pilot": "Pilot 4",
        "family": "trajectory geometry",
        "mechanism": "real path-geometry summaries",
        "candidate_or_control": "candidate",
        "mode": "direct_score",
        "primary_gate": "marginal_vs_clone >= +0.0010 and beat shuffled control",
        "secondary_gate": "standalone same-t signal",
        "verdict": "KILLED",
        "failure_mode": "A. NO RAW SIGNAL; C. WRONG PAIR FLOW; I. REPRESENTATION FAILURE",
    },
    "RT-1203": {
        "report": "pilot04_trajectory_geometry.json",
        "pilot": "Pilot 4",
        "family": "trajectory geometry",
        "mechanism": "shuffled trajectory-geometry control",
        "candidate_or_control": "control",
        "mode": "matched_control",
        "primary_gate": "control should not match candidate",
        "secondary_gate": "negative control sanity check",
        "verdict": "CONTROL_NEGATIVE",
        "failure_mode": "A. NO RAW SIGNAL; C. WRONG PAIR FLOW",
    },
    "RT-1204": {
        "report": "pilot05_scale_survival.json",
        "pilot": "Pilot 5",
        "family": "scale survival",
        "mechanism": "dyadic cross-scale survival summaries",
        "candidate_or_control": "candidate",
        "mode": "feature_block",
        "primary_gate": "marginal_vs_clone >= +0.0010",
        "secondary_gate": "candidate-control gap >= +0.0005",
        "verdict": "KILLED",
        "failure_mode": "B. REDUNDANT SIGNAL; C. WRONG PAIR FLOW; E. CONTROL FAILURE",
    },
    "RT-1205": {
        "report": "pilot05_scale_survival.json",
        "pilot": "Pilot 5",
        "family": "scale survival",
        "mechanism": "individual-scale surprise control",
        "candidate_or_control": "control",
        "mode": "matched_control",
        "primary_gate": "control should trail summary arm",
        "secondary_gate": "negative control sanity check",
        "verdict": "CONTROL_MATCHED",
        "failure_mode": "B. REDUNDANT SIGNAL; C. WRONG PAIR FLOW",
    },
    "RT-1206": {
        "report": "pilot06_spectral_impulse.json",
        "pilot": "Pilot 6",
        "family": "spectral impulse",
        "mechanism": "localized spectral impulse contrast",
        "candidate_or_control": "candidate",
        "mode": "feature_block",
        "primary_gate": "marginal_vs_clone >= +0.0010",
        "secondary_gate": "beat plain-energy control",
        "verdict": "KILLED",
        "failure_mode": "B. REDUNDANT SIGNAL; C. WRONG PAIR FLOW; E. CONTROL FAILURE",
    },
    "RT-1207": {
        "report": "pilot06_spectral_impulse.json",
        "pilot": "Pilot 6",
        "family": "spectral impulse",
        "mechanism": "plain spectral-energy control",
        "candidate_or_control": "control",
        "mode": "matched_control",
        "primary_gate": "control should trail contrast arm",
        "secondary_gate": "negative control sanity check",
        "verdict": "CONTROL_BEAT_CANDIDATE",
        "failure_mode": "B. REDUNDANT SIGNAL; F. INTEGRATION FAILURE",
    },
    "RT-1208": {
        "report": "pilot07_ordinal_irreversibility.json",
        "pilot": "Pilot 7",
        "family": "ordinal irreversibility",
        "mechanism": "ordinal transition irreversibility",
        "candidate_or_control": "candidate",
        "mode": "feature_block",
        "primary_gate": "marginal_vs_clone >= +0.0010",
        "secondary_gate": "candidate-control gap >= +0.0005",
        "verdict": "KILLED",
        "failure_mode": "B. REDUNDANT SIGNAL; C. WRONG PAIR FLOW; E. CONTROL FAILURE",
    },
    "RT-1209": {
        "report": "pilot07_ordinal_irreversibility.json",
        "pilot": "Pilot 7",
        "family": "ordinal irreversibility",
        "mechanism": "ordinal entropy control",
        "candidate_or_control": "control",
        "mode": "matched_control",
        "primary_gate": "control should trail irreversibility arm",
        "secondary_gate": "negative control sanity check",
        "verdict": "CONTROL_CLOSE",
        "failure_mode": "B. REDUNDANT SIGNAL; C. WRONG PAIR FLOW",
    },
    "RT-1210": {
        "report": "pilot10_joint_rarity.json",
        "pilot": "Pilot 10",
        "family": "joint rarity",
        "mechanism": "joint rarity of large residuals and dwell",
        "candidate_or_control": "candidate",
        "mode": "feature_block",
        "primary_gate": "marginal_vs_clone >= +0.0010",
        "secondary_gate": "beat dwell-only control",
        "verdict": "KILLED",
        "failure_mode": "B. REDUNDANT SIGNAL; C. WRONG PAIR FLOW; E. CONTROL FAILURE",
    },
    "RT-1211": {
        "report": "pilot10_joint_rarity.json",
        "pilot": "Pilot 10",
        "family": "joint rarity",
        "mechanism": "dwell-only control",
        "candidate_or_control": "control",
        "mode": "matched_control",
        "primary_gate": "control should trail joint-rarity arm",
        "secondary_gate": "negative control sanity check",
        "verdict": "CONTROL_BEAT_CANDIDATE",
        "failure_mode": "B. REDUNDANT SIGNAL; C. WRONG PAIR FLOW",
    },
    "RT-1212": {
        "report": "pilot09_difficulty_gate.json",
        "pilot": "Pilot 9",
        "family": "scalar difficulty",
        "mechanism": "real scalar difficulty gate",
        "candidate_or_control": "candidate",
        "mode": "scalar_gate",
        "primary_gate": "marginal_vs_clone >= +0.0010",
        "secondary_gate": "beat deranged scalar control",
        "verdict": "KILLED",
        "failure_mode": "B. REDUNDANT SIGNAL; E. CONTROL FAILURE; F. INTEGRATION FAILURE",
    },
    "RT-1213": {
        "report": "pilot09_difficulty_gate.json",
        "pilot": "Pilot 9",
        "family": "scalar difficulty",
        "mechanism": "deranged scalar control",
        "candidate_or_control": "control",
        "mode": "matched_control",
        "primary_gate": "control should trail real scalar",
        "secondary_gate": "negative control sanity check",
        "verdict": "CONTROL_BEAT_CANDIDATE",
        "failure_mode": "B. REDUNDANT SIGNAL; E. CONTROL FAILURE",
    },
    "RT-1214": {
        "report": "pilot03_observers.json",
        "pilot": "Pilot 3B",
        "family": "state observers",
        "mechanism": "Kalman/NIS observer block",
        "candidate_or_control": "candidate",
        "mode": "feature_block",
        "primary_gate": "marginal_vs_clone >= +0.0010",
        "secondary_gate": "dominant-cell pair flow",
        "verdict": "KILLED",
        "failure_mode": "B. REDUNDANT SIGNAL; C. WRONG PAIR FLOW; F. INTEGRATION FAILURE",
    },
    "RT-1215": {
        "report": "pilot03_observers.json",
        "pilot": "Pilot 3B",
        "family": "state observers",
        "mechanism": "Hankel-DMD observer block",
        "candidate_or_control": "candidate",
        "mode": "feature_block",
        "primary_gate": "marginal_vs_clone >= +0.0010",
        "secondary_gate": "within-t rho <= 0.85 redundancy gate",
        "verdict": "KILLED",
        "failure_mode": "B. REDUNDANT SIGNAL; F. INTEGRATION FAILURE",
    },
    "RT-1216": {
        "report": "pilot08_weighted_ctm.json",
        "pilot": "Pilot 8",
        "family": "weighted CTM",
        "mechanism": "tail-weighted conformal test martingale",
        "candidate_or_control": "candidate",
        "mode": "feature_block",
        "primary_gate": "marginal_vs_clone >= +0.0010",
        "secondary_gate": "beat unweighted CTM on mature-vs-never",
        "verdict": "KILLED_CLOSE_MISS",
        "failure_mode": "B. REDUNDANT SIGNAL; C. WRONG PAIR FLOW; F. INTEGRATION FAILURE",
    },
    "RT-1217": {
        "report": "pilot08_weighted_ctm.json",
        "pilot": "Pilot 8",
        "family": "weighted CTM",
        "mechanism": "unweighted conformal test martingale control",
        "candidate_or_control": "control",
        "mode": "matched_control",
        "primary_gate": "control should trail weighted CTM",
        "secondary_gate": "mature-vs-never contrast",
        "verdict": "CONTROL_CLOSE",
        "failure_mode": "B. REDUNDANT SIGNAL; C. WRONG PAIR FLOW",
    },
    "RT-1218": {
        "report": "pilot08_weighted_ctm.json",
        "pilot": "Pilot 8",
        "family": "weighted CTM",
        "mechanism": "direct e-value aggregation",
        "candidate_or_control": "candidate",
        "mode": "direct_score",
        "primary_gate": "marginal_vs_clone >= +0.0010",
        "secondary_gate": "standalone same-t signal",
        "verdict": "KILLED",
        "failure_mode": "A. NO RAW SIGNAL; C. WRONG PAIR FLOW; I. REPRESENTATION FAILURE",
    },
}

MATCHED_CONTROLS = {
    "RT-1202": "RT-1203",
    "RT-1204": "RT-1205",
    "RT-1206": "RT-1207",
    "RT-1208": "RT-1209",
    "RT-1210": "RT-1211",
    "RT-1212": "RT-1213",
    "RT-1216": "RT-1217",
}

CANDIDATE_IDS = [
    "RT-1200",
    "RT-1201",
    "RT-1202",
    "RT-1204",
    "RT-1206",
    "RT-1208",
    "RT-1210",
    "RT-1212",
    "RT-1214",
    "RT-1215",
    "RT-1216",
    "RT-1218",
]


MATRIX_COLUMNS = [
    "experiment_id",
    "pilot",
    "family",
    "mechanism",
    "candidate_or_control",
    "mode",
    "feature_count",
    "model_type",
    "standalone_whole_auc",
    "dominant_cell_auc",
    "mature_vs_never_auc",
    "mature_vs_prebreak_auc",
    "within_t_rank_corr_rt600",
    "E0",
    "E1",
    "E2",
    "gain_vs_rt600",
    "marginal_vs_clone",
    "candidate_vs_matched_control_marginal",
    "whole_pair_repairs",
    "whole_pair_damage",
    "whole_pair_net",
    "dominant_pair_repairs",
    "dominant_pair_damage",
    "dominant_pair_net",
    "neverbreak_repairs",
    "neverbreak_damage",
    "neverbreak_net",
    "prebreak_repairs",
    "prebreak_damage",
    "prebreak_net",
    "training_runtime_s",
    "primary_gate",
    "secondary_gate",
    "verdict",
    "failure_mode",
    "notes",
    "source_report",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def fmt_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return repr(value)
    return value


def read_results_rows() -> dict[str, dict[str, str]]:
    with RESULTS_CSV.open(newline="") as f:
        return {row["experiment_id"]: row for row in csv.DictReader(f)}


def result_row_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    training = payload.get("training")
    if isinstance(training, dict) and isinstance(training.get("result_row"), dict):
        return training["result_row"]
    return {}


def payload_for_arm(report: dict[str, Any], exp_id: str) -> dict[str, Any]:
    if exp_id in report and isinstance(report[exp_id], dict):
        return report[exp_id]
    return report


def metric_value(payload: dict[str, Any], group: str, side: str = "candidate") -> float | None:
    pack = payload.get("diagnostic_pack") or {}
    value = pack.get(group)
    if isinstance(value, dict):
        return as_float(value.get(side))
    return as_float(value)


def marginal_value(payload: dict[str, Any], key: str) -> float | None:
    ens = payload.get("ensemble_marginal") or {}
    return as_float(ens.get(key))


def e2_value(exp_id: str, payload: dict[str, Any]) -> float | None:
    ens = payload.get("ensemble_marginal") or {}
    return as_float(ens.get(f"rt600_plus_{exp_id}"))


def pair_stat(payload: dict[str, Any], group: str, field: str) -> int | None:
    flow = payload.get("pair_flow") or {}
    value = flow.get(group)
    if isinstance(value, dict):
        if field == "net":
            return as_int(value.get("net_pair_lift"))
        return as_int(value.get(field))
    return None


def pilot1_row() -> dict[str, Any]:
    meta = ARM_META["PILOT-1"]
    report = read_json(REPORT_DIR / meta["report"])
    gates = report.get("gates", {})
    routing = report.get("top_selector_oof_headroom", [])
    top_routing = routing[0] if isinstance(routing, list) and routing else {}
    disagreement = report.get("specialist_disagreement", {})
    pair_flow = report.get("specialist_pair_flow_vs_rt600_dominant_cell", {})
    notes = [
        f"verdict={report.get('verdict', 'WEAK')}",
        f"selector_delta={gates.get('selector_delta', top_routing.get('delta_vs_rt600', ''))}",
        f"disagreement_loss_spearman={disagreement.get('spearman_series_disagreement_vs_rt600_loss', '')}",
        f"specialist_pair_flow_all_net_negative={bool(pair_flow)}",
        NOTICE,
    ]
    row = {col: None for col in MATRIX_COLUMNS}
    row.update(
        {
            "experiment_id": "PILOT-1",
            "pilot": meta["pilot"],
            "family": meta["family"],
            "mechanism": meta["mechanism"],
            "candidate_or_control": meta["candidate_or_control"],
            "mode": meta["mode"],
            "primary_gate": meta["primary_gate"],
            "secondary_gate": meta["secondary_gate"],
            "verdict": meta["verdict"],
            "failure_mode": meta["failure_mode"],
            "notes": "; ".join(str(x) for x in notes),
            "source_report": meta["report"],
        }
    )
    return row


def build_matrix() -> list[dict[str, Any]]:
    results = read_results_rows()
    rows = [pilot1_row()]
    payload_by_id: dict[str, dict[str, Any]] = {}

    for exp_id in FIRST_SWEEP_IDS:
        meta = ARM_META[exp_id]
        report = read_json(REPORT_DIR / meta["report"])
        payload = payload_for_arm(report, exp_id)
        payload_by_id[exp_id] = payload
        result = results.get(exp_id, {})
        embedded = result_row_from_payload(payload)
        row = {col: None for col in MATRIX_COLUMNS}
        rt600 = marginal_value(payload, "rt600_7stream")
        seedclone = marginal_value(payload, "rt600_plus_seedclone")
        row.update(
            {
                "experiment_id": exp_id,
                "pilot": meta["pilot"],
                "family": meta["family"],
                "mechanism": meta["mechanism"],
                "candidate_or_control": meta["candidate_or_control"],
                "mode": meta["mode"],
                "feature_count": as_int(result.get("n_features") or embedded.get("n_features")),
                "model_type": result.get("model") or embedded.get("model"),
                "standalone_whole_auc": metric_value(payload, "whole_fold"),
                "dominant_cell_auc": metric_value(payload, "dominant_cell"),
                "mature_vs_never_auc": metric_value(payload, "mature_vs_neverbreak"),
                "mature_vs_prebreak_auc": metric_value(payload, "mature_vs_prebreak"),
                "within_t_rank_corr_rt600": metric_value(payload, "within_t_rank_corr_rt600"),
                "E0": rt600,
                "E1": seedclone,
                "E2": e2_value(exp_id, payload),
                "gain_vs_rt600": marginal_value(payload, "gain_vs_base"),
                "marginal_vs_clone": marginal_value(payload, "marginal_vs_clone"),
                "whole_pair_repairs": pair_stat(payload, "whole_fold", "repairs"),
                "whole_pair_damage": pair_stat(payload, "whole_fold", "damage"),
                "whole_pair_net": pair_stat(payload, "whole_fold", "net"),
                "dominant_pair_repairs": pair_stat(payload, "dominant_cell", "repairs"),
                "dominant_pair_damage": pair_stat(payload, "dominant_cell", "damage"),
                "dominant_pair_net": pair_stat(payload, "dominant_cell", "net"),
                "neverbreak_repairs": pair_stat(payload, "cell_never_break_neg", "repairs"),
                "neverbreak_damage": pair_stat(payload, "cell_never_break_neg", "damage"),
                "neverbreak_net": pair_stat(payload, "cell_never_break_neg", "net"),
                "prebreak_repairs": pair_stat(payload, "cell_pre_break_neg", "repairs"),
                "prebreak_damage": pair_stat(payload, "cell_pre_break_neg", "damage"),
                "prebreak_net": pair_stat(payload, "cell_pre_break_neg", "net"),
                "training_runtime_s": as_float(
                    result.get("training_runtime_s") or embedded.get("training_runtime_s")
                ),
                "primary_gate": meta["primary_gate"],
                "secondary_gate": meta["secondary_gate"],
                "verdict": meta["verdict"],
                "failure_mode": meta["failure_mode"],
                "notes": f"{result.get('notes') or embedded.get('notes', '')}; {NOTICE}",
                "source_report": meta["report"],
            }
        )
        rows.append(row)

    by_id = {r["experiment_id"]: r for r in rows}
    for cand, ctrl in MATCHED_CONTROLS.items():
        cm = as_float(by_id[cand].get("marginal_vs_clone"))
        xm = as_float(by_id[ctrl].get("marginal_vs_clone"))
        if cm is not None and xm is not None:
            by_id[cand]["candidate_vs_matched_control_marginal"] = cm - xm
            by_id[ctrl]["candidate_vs_matched_control_marginal"] = xm - cm

    missing = [exp_id for exp_id in FIRST_SWEEP_IDS if exp_id not in results]
    if missing:
        raise SystemExit(f"Missing first-sweep ids from RESULTS.csv: {missing}")
    return rows


def write_matrix(rows: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "first_sweep_matrix.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MATRIX_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: fmt_cell(row.get(col)) for col in MATRIX_COLUMNS})
    write_json(
        OUT_DIR / "first_sweep_matrix.json",
        {
            "notice": NOTICE,
            "rows": rows,
            "source_results_sha256": sha256_file(RESULTS_CSV),
        },
    )


def rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def correlation(x: list[float], y: list[float]) -> dict[str, Any]:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[mask]
    ya = ya[mask]
    if len(xa) < 3 or np.std(xa) == 0 or np.std(ya) == 0:
        return {"n": int(len(xa)), "pearson": None, "spearman": None}
    pearson = float(np.corrcoef(xa, ya)[0, 1])
    rx = rank_average(xa)
    ry = rank_average(ya)
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    return {"n": int(len(xa)), "pearson": pearson, "spearman": spearman}


def numeric(row: dict[str, Any], col: str) -> float | None:
    return as_float(row.get(col))


def build_meta_relationships(rows: list[dict[str, Any]]) -> None:
    scored = [r for r in rows if str(r["experiment_id"]).startswith("RT-")]
    candidates = [r for r in scored if r["candidate_or_control"] == "candidate"]
    groups = {"all_scored_arms": scored, "candidate_arms_only": candidates}
    predictors = [
        "standalone_whole_auc",
        "dominant_cell_auc",
        "mature_vs_never_auc",
        "mature_vs_prebreak_auc",
        "within_t_rank_corr_rt600",
        "gain_vs_rt600",
        "candidate_vs_matched_control_marginal",
        "whole_pair_net",
        "dominant_pair_net",
        "neverbreak_net",
        "prebreak_net",
        "feature_count",
        "training_runtime_s",
    ]
    relationships: list[dict[str, Any]] = []
    for group_name, group_rows in groups.items():
        y = [numeric(r, "marginal_vs_clone") for r in group_rows]
        for predictor in predictors:
            x = [numeric(r, predictor) for r in group_rows]
            stats = correlation(x, y)
            relationships.append(
                {
                    "notice": NOTICE,
                    "group": group_name,
                    "predictor": predictor,
                    "target": "marginal_vs_clone",
                    **stats,
                }
            )

    with (OUT_DIR / "first_sweep_meta_relationships.csv").open("w", newline="") as f:
        fieldnames = ["notice", "group", "predictor", "target", "n", "pearson", "spearman"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in relationships:
            writer.writerow({k: fmt_cell(row.get(k)) for k in fieldnames})
    write_json(
        OUT_DIR / "first_sweep_meta_relationships.json",
        {
            "notice": NOTICE,
            "relationships": relationships,
            "interpretation_guardrail": "Small-n descriptive correlations only; no p-values and no promotion evidence.",
        },
    )


def read_store_index() -> dict[str, np.ndarray]:
    meta = pd.read_parquet(ROOT / "cache" / "store" / "meta.parquet").sort_values("id")
    folds = pd.read_parquet(ROOT / "research" / "folds" / "folds.parquet").sort_values("id")
    if not np.array_equal(meta["id"].to_numpy(), folds["id"].to_numpy()):
        raise SystemExit("meta.parquet and folds.parquet id orders differ after sorting")

    n_online = meta["n_online"].to_numpy(dtype=np.int64)
    tau = meta["tau_index"].to_numpy(dtype=np.int64)
    has_break = meta["has_break"].to_numpy(dtype=np.int8)
    fold = folds["fold"].to_numpy(dtype=np.int16)
    split = folds["split"].to_numpy()
    starts = np.concatenate([[0], np.cumsum(n_online[:-1])]).astype(np.int64)
    series = np.where((fold == 0) & (split == "dev"))[0]

    row_chunks: list[np.ndarray] = []
    t_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    age_chunks: list[np.ndarray] = []
    has_break_chunks: list[np.ndarray] = []
    sid_chunks: list[np.ndarray] = []
    for sid in series:
        n = int(n_online[sid])
        t = np.arange(n, dtype=np.int32)
        row_chunks.append(np.arange(starts[sid], starts[sid] + n, dtype=np.int64))
        t_chunks.append(t)
        if tau[sid] >= 0:
            y = (t >= tau[sid]).astype(np.int8)
            age = np.where(y == 1, t - tau[sid], -1).astype(np.int32)
        else:
            y = np.zeros(n, dtype=np.int8)
            age = np.full(n, -1, dtype=np.int32)
        y_chunks.append(y)
        age_chunks.append(age)
        has_break_chunks.append(np.full(n, has_break[sid], dtype=np.int8))
        sid_chunks.append(np.full(n, sid, dtype=np.int32))

    return {
        "row": np.concatenate(row_chunks),
        "t": np.concatenate(t_chunks),
        "y": np.concatenate(y_chunks),
        "age": np.concatenate(age_chunks),
        "has_break": np.concatenate(has_break_chunks),
        "series_id": np.concatenate(sid_chunks),
    }


def oof_path(label: str) -> Path:
    if label == "RT600_BLEND":
        return RT600_BLEND
    return OOF_DIR / f"{label}.npy"


def available_score_labels(include_context: bool = True) -> list[str]:
    labels = ["RT600_BLEND"] + FIRST_SWEEP_IDS
    if include_context:
        labels += [x for x in CONTEXT_SERIES if oof_path(x).exists()]
    missing = [x for x in labels if not oof_path(x).exists()]
    if missing:
        raise SystemExit(f"Missing required OOF arrays: {missing}")
    return labels


def load_score_matrix(labels: list[str], rows: np.ndarray) -> np.ndarray:
    cols = []
    for label in labels:
        arr = np.load(oof_path(label), mmap_mode="r")
        cols.append(np.asarray(arr[rows], dtype=np.float64))
    return np.column_stack(cols)


def build_rank_correlations(index: dict[str, np.ndarray]) -> None:
    labels = available_score_labels(include_context=True)
    rows = index["row"]
    t = index["t"]
    scores = load_score_matrix(labels, rows)
    finite = np.all(np.isfinite(scores), axis=1)
    scores = scores[finite]
    t = t[finite]
    order = np.argsort(t, kind="mergesort")
    scores = scores[order]
    t = t[order]

    m = len(labels)
    cross = np.zeros((m, m), dtype=np.float64)
    ss = np.zeros(m, dtype=np.float64)
    n_used = 0
    start = 0
    while start < len(t):
        end = start + 1
        while end < len(t) and t[end] == t[start]:
            end += 1
        group = scores[start:end]
        if len(group) >= 3:
            ranks = np.column_stack([rank_average(group[:, j]) for j in range(m)])
            centered = ranks - ranks.mean(axis=0, keepdims=True)
            cross += centered.T @ centered
            ss += np.sum(centered * centered, axis=0)
            n_used += len(group)
        start = end

    corr = cross / np.sqrt(np.outer(ss, ss))
    records = []
    for i, left in enumerate(labels):
        for j, right in enumerate(labels):
            records.append(
                {
                    "notice": NOTICE,
                    "sample": "fold0_dev_same_t_ranked",
                    "left": left,
                    "right": right,
                    "within_t_rank_corr": float(corr[i, j]),
                    "rows_used": int(n_used),
                }
            )

    with (OUT_DIR / "first_sweep_rank_corr.csv").open("w", newline="") as f:
        fieldnames = ["notice", "sample", "left", "right", "within_t_rank_corr", "rows_used"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({k: fmt_cell(record[k]) for k in fieldnames})
    write_json(
        OUT_DIR / "first_sweep_rank_corr.json",
        {
            "notice": NOTICE,
            "sample": "fold0 dev rows only; ranks recomputed within each t",
            "labels": labels,
            "rows_used": int(n_used),
            "matrix": corr.tolist(),
        },
    )


def stable_seed(label: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).hexdigest()
    return int(digest[:16], 16) % (2**32)


def sample_pairs_for_label(
    label: str,
    index: dict[str, np.ndarray],
    pairs_per_t: int,
    seed: int,
) -> dict[str, np.ndarray]:
    t = index["t"]
    y = index["y"]
    age = index["age"]
    has_break = index["has_break"]
    row = index["row"]

    if label == "whole_fold":
        pos_mask = y == 1
        neg_mask = y == 0
    elif label == "dominant_cell":
        pos_mask = (y == 1) & (age >= 100) & (t >= 200)
        neg_mask = (y == 0) & (t >= 200)
    elif label == "cell_never_break_neg":
        pos_mask = (y == 1) & (age >= 100) & (t >= 200)
        neg_mask = (y == 0) & (t >= 200) & (has_break == 0)
    elif label == "cell_pre_break_neg":
        pos_mask = (y == 1) & (age >= 100) & (t >= 200)
        neg_mask = (y == 0) & (t >= 200) & (has_break == 1)
    else:
        raise ValueError(label)

    rng = np.random.default_rng(stable_seed(label, seed))
    pos_by_t: dict[int, np.ndarray] = {}
    neg_by_t: dict[int, np.ndarray] = {}
    for tt in np.unique(t[pos_mask | neg_mask]):
        pos = np.flatnonzero(pos_mask & (t == tt))
        neg = np.flatnonzero(neg_mask & (t == tt))
        if len(pos) and len(neg):
            pos_by_t[int(tt)] = pos
            neg_by_t[int(tt)] = neg

    pos_rows: list[int] = []
    neg_rows: list[int] = []
    pair_t: list[int] = []
    neg_kind: list[str] = []
    for tt in sorted(pos_by_t):
        pos = pos_by_t[tt]
        neg = neg_by_t[tt]
        total = len(pos) * len(neg)
        k = min(pairs_per_t, total)
        if total <= pairs_per_t:
            chosen = [(int(p), int(n)) for p in pos for n in neg]
        else:
            chosen_set: set[tuple[int, int]] = set()
            max_trials = max(k * 20, 200)
            trials = 0
            while len(chosen_set) < k and trials < max_trials:
                pp = int(pos[rng.integers(0, len(pos))])
                nn = int(neg[rng.integers(0, len(neg))])
                chosen_set.add((pp, nn))
                trials += 1
            if len(chosen_set) < k:
                # Extremely unlikely; fall back to deterministic lexicographic fill.
                for pp in pos:
                    for nn in neg:
                        chosen_set.add((int(pp), int(nn)))
                        if len(chosen_set) >= k:
                            break
                    if len(chosen_set) >= k:
                        break
            chosen = sorted(chosen_set)
        for pp, nn in chosen:
            pos_rows.append(int(row[pp]))
            neg_rows.append(int(row[nn]))
            pair_t.append(tt)
            if y[nn] != 0:
                neg_kind.append("not_negative")
            elif has_break[nn] == 0:
                neg_kind.append("never_break")
            else:
                neg_kind.append("pre_break")

    return {
        "pos_row": np.asarray(pos_rows, dtype=np.int64),
        "neg_row": np.asarray(neg_rows, dtype=np.int64),
        "t": np.asarray(pair_t, dtype=np.int32),
        "neg_kind": np.asarray(neg_kind, dtype=object),
    }


def pair_correct(score: np.ndarray, pairs: dict[str, np.ndarray]) -> np.ndarray:
    return score[pairs["pos_row"]] > score[pairs["neg_row"]]


def jaccard(a: np.ndarray, b: np.ndarray) -> float | None:
    union = np.count_nonzero(a | b)
    if union == 0:
        return None
    return float(np.count_nonzero(a & b) / union)


def build_repair_diagnostics(index: dict[str, np.ndarray], pairs_per_t: int, seed: int) -> None:
    labels = FIRST_SWEEP_IDS
    candidate_labels = CANDIDATE_IDS
    score_labels = ["RT600_BLEND"] + labels + SPECIALISTS
    arrays = {label: np.load(oof_path(label), mmap_mode="r") for label in score_labels}

    sample_names = ["whole_fold", "dominant_cell", "cell_never_break_neg", "cell_pre_break_neg"]
    samples = {
        name: sample_pairs_for_label(name, index, pairs_per_t=pairs_per_t, seed=seed)
        for name in sample_names
    }
    sample_summary: dict[str, Any] = {}
    arm_summary: dict[str, Any] = {label: {} for label in labels}
    candidate_coverage: dict[str, Any] = {}
    per_sample_masks: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for sample_name, pairs in samples.items():
        base_correct = pair_correct(arrays["RT600_BLEND"], pairs)
        base_wrong = ~base_correct
        sample_summary[sample_name] = {
            "pairs_sampled": int(len(base_correct)),
            "rt600_wrong_pairs": int(np.count_nonzero(base_wrong)),
            "rt600_right_pairs": int(np.count_nonzero(base_correct)),
            "pairs_per_t": int(pairs_per_t),
            "seed": int(seed),
        }
        per_sample_masks[sample_name] = {}
        candidate_repairs = []
        candidate_damages = []
        for label in labels:
            correct = pair_correct(arrays[label], pairs)
            repair = base_wrong & correct
            damage = base_correct & (~correct)
            per_sample_masks[sample_name][label] = {"repair": repair, "damage": damage}
            arm_summary[label][sample_name] = {
                "repairs": int(np.count_nonzero(repair)),
                "damage": int(np.count_nonzero(damage)),
                "net_pair_lift": int(np.count_nonzero(repair) - np.count_nonzero(damage)),
                "repair_rate_of_rt600_wrong": float(np.count_nonzero(repair) / max(np.count_nonzero(base_wrong), 1)),
                "damage_rate_of_rt600_right": float(np.count_nonzero(damage) / max(np.count_nonzero(base_correct), 1)),
            }
            if label in candidate_labels:
                candidate_repairs.append(repair)
                candidate_damages.append(damage)
        repair_counts = np.sum(np.column_stack(candidate_repairs), axis=1) if candidate_repairs else np.zeros(len(base_correct))
        damage_counts = np.sum(np.column_stack(candidate_damages), axis=1) if candidate_damages else np.zeros(len(base_correct))
        candidate_coverage[sample_name] = {
            "candidate_set": candidate_labels,
            "rt600_wrong_pairs": int(np.count_nonzero(base_wrong)),
            "rt600_wrong_repaired_by_any_candidate": int(np.count_nonzero(base_wrong & (repair_counts > 0))),
            "rt600_wrong_repaired_by_any_candidate_rate": float(
                np.count_nonzero(base_wrong & (repair_counts > 0)) / max(np.count_nonzero(base_wrong), 1)
            ),
            "rt600_wrong_repaired_by_two_plus_candidates": int(np.count_nonzero(base_wrong & (repair_counts >= 2))),
            "rt600_wrong_repaired_by_two_plus_candidates_rate": float(
                np.count_nonzero(base_wrong & (repair_counts >= 2)) / max(np.count_nonzero(base_wrong), 1)
            ),
            "rt600_right_damaged_by_any_candidate": int(np.count_nonzero(base_correct & (damage_counts > 0))),
            "rt600_right_damaged_by_any_candidate_rate": float(
                np.count_nonzero(base_correct & (damage_counts > 0)) / max(np.count_nonzero(base_correct), 1)
            ),
            "mean_candidate_repairs_per_rt600_wrong_pair": float(np.mean(repair_counts[base_wrong])) if np.any(base_wrong) else None,
        }

    overlap_rows = []
    for sample_name in sample_names:
        for i, left in enumerate(labels):
            for right in labels[i + 1 :]:
                lm = per_sample_masks[sample_name][left]
                rm = per_sample_masks[sample_name][right]
                overlap_rows.append(
                    {
                        "notice": NOTICE,
                        "sample": sample_name,
                        "left": left,
                        "right": right,
                        "repair_left": int(np.count_nonzero(lm["repair"])),
                        "repair_right": int(np.count_nonzero(rm["repair"])),
                        "repair_intersection": int(np.count_nonzero(lm["repair"] & rm["repair"])),
                        "repair_jaccard": jaccard(lm["repair"], rm["repair"]),
                        "damage_left": int(np.count_nonzero(lm["damage"])),
                        "damage_right": int(np.count_nonzero(rm["damage"])),
                        "damage_intersection": int(np.count_nonzero(lm["damage"] & rm["damage"])),
                        "damage_jaccard": jaccard(lm["damage"], rm["damage"]),
                    }
                )
    with (OUT_DIR / "first_sweep_repair_overlap.csv").open("w", newline="") as f:
        fieldnames = [
            "notice",
            "sample",
            "left",
            "right",
            "repair_left",
            "repair_right",
            "repair_intersection",
            "repair_jaccard",
            "damage_left",
            "damage_right",
            "damage_intersection",
            "damage_jaccard",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in overlap_rows:
            writer.writerow({k: fmt_cell(row[k]) for k in fieldnames})

    write_json(
        OUT_DIR / "first_sweep_repair_summary.json",
        {
            "notice": NOTICE,
            "sampling": {
                "scope": "fold0 dev same-t positive/negative pairs only",
                "pairs_per_t": int(pairs_per_t),
                "seed": int(seed),
                "strict_pair_correct_rule": "score_positive > score_negative",
                "not_a_score": True,
            },
            "samples": sample_summary,
            "arm_summary": arm_summary,
            "candidate_coverage": candidate_coverage,
        },
    )

    build_specialist_decomposition(samples["dominant_cell"], arrays, per_sample_masks["dominant_cell"], candidate_labels)


def build_specialist_decomposition(
    pairs: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    dominant_masks: dict[str, dict[str, np.ndarray]],
    candidate_labels: list[str],
) -> None:
    base_correct = pair_correct(arrays["RT600_BLEND"], pairs)
    base_wrong = ~base_correct
    if not np.any(base_wrong):
        raise SystemExit("No sampled dominant-cell RT600 errors; cannot decompose specialist space")
    specialist_correct = np.column_stack([pair_correct(arrays[label], pairs) for label in SPECIALISTS])
    correct_count = np.sum(specialist_correct, axis=1)
    candidate_repair_count = np.sum(
        np.column_stack([dominant_masks[label]["repair"] for label in candidate_labels]), axis=1
    )
    rt1216_repair = dominant_masks["RT-1216"]["repair"]

    def summarize_mask(mask: np.ndarray) -> dict[str, Any]:
        denom = int(np.count_nonzero(mask))
        if denom == 0:
            return {
                "pairs": 0,
                "share_of_rt600_wrong": 0.0,
                "repaired_by_any_candidate": 0,
                "repaired_by_any_candidate_rate": None,
                "repaired_by_rt1216": 0,
                "repaired_by_rt1216_rate": None,
                "mean_candidate_repairs": None,
            }
        return {
            "pairs": denom,
            "share_of_rt600_wrong": float(denom / np.count_nonzero(base_wrong)),
            "repaired_by_any_candidate": int(np.count_nonzero(mask & (candidate_repair_count > 0))),
            "repaired_by_any_candidate_rate": float(np.count_nonzero(mask & (candidate_repair_count > 0)) / denom),
            "repaired_by_rt1216": int(np.count_nonzero(mask & rt1216_repair)),
            "repaired_by_rt1216_rate": float(np.count_nonzero(mask & rt1216_repair) / denom),
            "mean_candidate_repairs": float(np.mean(candidate_repair_count[mask])),
        }

    wrong = base_wrong
    histogram = {
        str(k): int(np.count_nonzero(wrong & (correct_count == k))) for k in range(len(SPECIALISTS) + 1)
    }
    partitions = {
        "all_specialists_wrong": wrong & (correct_count == 0),
        "minority_specialists_correct_1_to_3": wrong & (correct_count >= 1) & (correct_count <= 3),
        "majority_specialists_correct_4_to_7": wrong & (correct_count >= 4),
    }
    supplementary = {
        "near_split_3_or_4_specialists_correct": wrong & ((correct_count == 3) | (correct_count == 4)),
        "at_least_one_specialist_correct": wrong & (correct_count >= 1),
    }

    write_json(
        OUT_DIR / "first_sweep_specialist_decomposition.json",
        {
            "notice": NOTICE,
            "sample": "dominant_cell deterministic same-t pairs; RT600 wrong subset only",
            "specialists": SPECIALISTS,
            "rt600_wrong_pairs": int(np.count_nonzero(wrong)),
            "specialist_correct_count_histogram": histogram,
            "partitions": {name: summarize_mask(mask) for name, mask in partitions.items()},
            "supplementary": {name: summarize_mask(mask) for name, mask in supplementary.items()},
        },
    )


def source_manifest(skip_oof: bool, pairs_per_t: int, seed: int) -> None:
    report_files = sorted({ARM_META[k]["report"] for k in ARM_META})
    manifest = {
        "notice": NOTICE,
        "script": str(Path(__file__).relative_to(ROOT)),
        "results_csv_sha256": sha256_file(RESULTS_CSV),
        "reports": {name: sha256_file(REPORT_DIR / name) for name in report_files},
        "oof_used": None
        if skip_oof
        else {label: sha256_file(oof_path(label)) for label in available_score_labels(include_context=True)},
        "repair_sampling": None if skip_oof else {"pairs_per_t": int(pairs_per_t), "seed": int(seed)},
        "forbidden_actions": [
            "no training",
            "no new OOF score",
            "no TS-AUC calculation for a new candidate",
            "no RESULTS.csv edit",
            "no RT id allocation",
            "no lockbox/test/submission touch",
        ],
    }
    write_json(OUT_DIR / "first_sweep_source_manifest.json", manifest)


def validate_outputs(rows: list[dict[str, Any]], skip_oof: bool) -> None:
    ids = [r["experiment_id"] for r in rows if str(r["experiment_id"]).startswith("RT-")]
    if ids != FIRST_SWEEP_IDS:
        raise SystemExit(f"Unexpected RT id order in matrix: {ids}")
    if not all((OOF_DIR / f"{exp_id}.npy").exists() for exp_id in FIRST_SWEEP_IDS):
        raise SystemExit("At least one first-sweep OOF artifact is missing")
    for path in [
        OUT_DIR / "first_sweep_matrix.csv",
        OUT_DIR / "first_sweep_matrix.json",
        OUT_DIR / "first_sweep_meta_relationships.json",
        OUT_DIR / "first_sweep_meta_relationships.csv",
        OUT_DIR / "first_sweep_source_manifest.json",
    ]:
        if not path.exists():
            raise SystemExit(f"Missing expected output: {path}")
    if not skip_oof:
        for path in [
            OUT_DIR / "first_sweep_rank_corr.csv",
            OUT_DIR / "first_sweep_rank_corr.json",
            OUT_DIR / "first_sweep_repair_overlap.csv",
            OUT_DIR / "first_sweep_repair_summary.json",
            OUT_DIR / "first_sweep_specialist_decomposition.json",
        ]:
            if not path.exists():
                raise SystemExit(f"Missing expected OOF-derived output: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-oof", action="store_true", help="only normalize report/RESULTS artifacts")
    parser.add_argument("--pairs-per-t", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    before_hash = sha256_file(RESULTS_CSV)
    rows = build_matrix()
    write_matrix(rows)
    build_meta_relationships(rows)
    if not args.skip_oof:
        index = read_store_index()
        build_rank_correlations(index)
        build_repair_diagnostics(index, pairs_per_t=args.pairs_per_t, seed=args.seed)
    source_manifest(args.skip_oof, args.pairs_per_t, args.seed)
    after_hash = sha256_file(RESULTS_CSV)
    if after_hash != before_hash:
        raise SystemExit("RESULTS.csv hash changed during descriptive analysis")
    validate_outputs(rows, args.skip_oof)
    print(
        json.dumps(
            {
                "status": "ok",
                "notice": NOTICE,
                "matrix_rows": len(rows),
                "skip_oof": bool(args.skip_oof),
                "results_csv_sha256": after_hash,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
