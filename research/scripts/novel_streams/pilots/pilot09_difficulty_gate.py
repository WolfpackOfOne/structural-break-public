"""Pilot 9(i): nested scalar historical difficulty gate.

Scored candidates:
  RT-1212: nested OOF scalar difficulty conditioner.
  RT-1213: within-fold deranged scalar control.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("SBR_ROOT", Path(__file__).resolve().parents[4]))
os.environ.setdefault("SBR_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(ROOT / "research" / "scripts" / "novel_streams"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import sbr.pipeline as PL
from harness import rt600_blend
from pilot01_failure_manifolds import (
    MIN_SERIES_WEIGHT,
    history_fingerprints,
    per_series_loss,
)
from pilot06_spectral_impulse import (
    append_result_append_only,
    candidate_result,
    finite_float,
    git_sha,
    rt600_sentinel,
)
from sbr.metric import ts_auc_flat
from wave2_lib import ABL, FULL
from wave5_lib import Ctx, SPECIALISTS, load_oof

PREREG_SHA = "3275ffd"
CANDIDATE_ID = "RT-1212"
CONTROL_ID = "RT-1213"
CANDIDATE_COL = "pilot09_difficulty_scalar"
CONTROL_COL = "pilot09_difficulty_scalar_deranged"
OUTDIR = ROOT / "research" / "reports" / "new_avenues_2026"
OOFDIR = ROOT / "research" / "oof"
OUTDIR.mkdir(parents=True, exist_ok=True)
OOFDIR.mkdir(parents=True, exist_ok=True)
SCORED_FOLD = 0
DEV_FOLDS = (0, 1, 2, 3, 4)

EXPECTED_FP_NAMES = [
    "n_hist",
    "kurt",
    "skew",
    "hill",
    "ar1",
    "ar2",
    "ar3",
    "ar4",
    "ar5",
    "ar_sum",
    "acf1_sq",
    "acf1_abs",
    "vr10",
    "vr50",
    "spec_slope",
    "perm_ent",
    "turn_rate",
    "max_absz64",
    "exc_max_run64",
    "exc_n64",
    "max_logvr128",
    "q_ratio",
    "zerocross",
]

SCALAR_PARAMS = dict(
    objective="regression",
    learning_rate=0.05,
    num_leaves=15,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    num_threads=2,
    seed=0,
    verbose=-1,
)
SCALAR_ROUNDS = 250

PILOT_PARAMS = dict(objective="binary", num_threads=2, verbose=-1)
PILOT_PARAMS.update(ABL["params"])
PILOT_PARAMS.setdefault("bagging_freq", 1)


def robust_standardize(x: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    train = np.asarray(train_mask, dtype=bool)
    med = np.nanmedian(x[train], axis=0)
    mad = np.nanmedian(np.abs(x[train] - med), axis=0) * 1.4826
    sd = np.nanstd(x[train], axis=0)
    scale = np.where(mad > 1e-12, mad, sd)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (x - med) / scale
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return z, {"center": med.tolist(), "scale": scale.tolist()}


def nested_plan(folds: np.ndarray, outer_fold: int, dev_folds: tuple[int, ...] = DEV_FOLDS) -> list[dict]:
    plan = []
    dev = np.isin(folds, dev_folds)
    train_folds = [int(f) for f in dev_folds if int(f) != int(outer_fold)]
    plan.append(
        {
            "outer_fold": int(outer_fold),
            "pred_fold": int(outer_fold),
            "train_folds": train_folds,
            "train_mask": dev & np.isin(folds, train_folds),
            "predict_mask": folds == int(outer_fold),
        }
    )
    for pred_fold in train_folds:
        inner_train_folds = [int(f) for f in train_folds if int(f) != int(pred_fold)]
        plan.append(
            {
                "outer_fold": int(outer_fold),
                "pred_fold": int(pred_fold),
                "train_folds": inner_train_folds,
                "train_mask": dev & np.isin(folds, inner_train_folds),
                "predict_mask": folds == int(pred_fold),
            }
        )
    return plan


def derangement_source(folds: np.ndarray, seed: int = 0, dev_folds: tuple[int, ...] = DEV_FOLDS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    source = np.arange(len(folds), dtype=np.int64)
    for f in dev_folds:
        idx = np.flatnonzero(folds == int(f))
        if len(idx) <= 1:
            continue
        perm = idx.copy()
        for _ in range(200):
            perm = idx[rng.permutation(len(idx))]
            if not np.any(perm == idx):
                break
        if np.any(perm == idx):
            perm = np.roll(idx, 1)
        source[idx] = perm
    return source


def apply_derangement(scalar: np.ndarray, source: np.ndarray, folds: np.ndarray) -> np.ndarray:
    out = np.array(scalar, copy=True)
    dev = np.isin(folds, DEV_FOLDS)
    out[dev] = scalar[source[dev]]
    return out


def fit_scalar_predictions(
    fp: np.ndarray,
    target: np.ndarray,
    eligible: np.ndarray,
    folds: np.ndarray,
    outer_fold: int,
) -> tuple[np.ndarray, list[dict]]:
    scalar = np.full(len(folds), np.nan, dtype=np.float64)
    provenance = []
    for item in nested_plan(folds, outer_fold):
        train_mask = item["train_mask"] & eligible & np.isfinite(target)
        predict_mask = item["predict_mask"]
        if int(train_mask.sum()) < 200:
            raise SystemExit(f"too few scalar training series for outer fold {outer_fold}: {int(train_mask.sum())}")
        xz, constants = robust_standardize(fp, train_mask)
        ds = lgb.Dataset(xz[train_mask], label=target[train_mask], feature_name=EXPECTED_FP_NAMES)
        booster = lgb.train(SCALAR_PARAMS, ds, num_boost_round=SCALAR_ROUNDS)
        pred = booster.predict(xz[predict_mask])
        scalar[predict_mask] = pred
        provenance.append(
            {
                "outer_fold": int(item["outer_fold"]),
                "pred_fold": int(item["pred_fold"]),
                "train_folds": [int(x) for x in item["train_folds"]],
                "train_series": int(train_mask.sum()),
                "pred_series": int(predict_mask.sum()),
                "standardization_center": constants["center"],
                "standardization_scale": constants["scale"],
            }
        )
    return scalar, provenance


def fold_purity_ok(provenance: list[dict]) -> bool:
    for item in provenance:
        train_folds = set(int(x) for x in item["train_folds"])
        if int(item["outer_fold"]) in train_folds:
            return False
        if int(item["pred_fold"]) in train_folds:
            return False
    return True


def derangement_audit(real_scalar: np.ndarray, deranged: np.ndarray, source: np.ndarray, folds: np.ndarray) -> dict:
    out = {}
    for f in DEV_FOLDS:
        idx = np.flatnonzero(folds == int(f))
        no_fixed = bool(len(idx) <= 1 or not np.any(source[idx] == idx))
        same_multiset = bool(np.allclose(np.sort(real_scalar[idx]), np.sort(deranged[idx]), rtol=0.0, atol=0.0, equal_nan=True))
        out[str(f)] = {"n": int(len(idx)), "no_fixed_points": no_fixed, "same_multiset": same_multiset}
    out["ok"] = bool(all(v["no_fixed_points"] and v["same_multiset"] for v in out.values() if isinstance(v, dict)))
    return out


def row_scalar_from_series(c: Ctx, series_scalar: np.ndarray) -> np.ndarray:
    row_scalar = series_scalar[c.d.sidx].astype(np.float32)
    return row_scalar


def append_column(x: np.ndarray, col: np.ndarray) -> np.ndarray:
    out = np.empty((x.shape[0], x.shape[1] + 1), dtype=np.float32)
    out[:, :-1] = x.astype(np.float32, copy=False)
    out[:, -1] = col.astype(np.float32, copy=False)
    return out


def train_nested_oof(
    exp_id: str,
    col_name: str,
    deranged: bool,
    fp: np.ndarray,
    target: np.ndarray,
    eligible: np.ndarray,
    folds_series: np.ndarray,
    derange_source: np.ndarray,
    mats: list[np.ndarray],
    names: list[str],
    c: Ctx,
    hypothesis: str,
    falsification: str,
    notes: str,
) -> tuple[np.ndarray, dict]:
    t0 = time.time()
    d = PL.Data(screen=False)
    keep_idx = np.arange(len(names), dtype=np.int64)
    used_names = [names[i] for i in keep_idx] + [col_name]
    p = dict(PILOT_PARAMS)
    n_round = int(p.pop("n_estimators", 600))
    rng = np.random.default_rng(0)
    oof = np.full(len(d.y), np.nan, dtype=np.float32)
    imp_sum = np.zeros(len(used_names), dtype=np.float64)
    fold0_auc = None
    fit_summaries = []
    scalar_summaries = []

    for f in DEV_FOLDS:
        scalar, provenance = fit_scalar_predictions(fp, target, eligible, folds_series, int(f))
        if not fold_purity_ok(provenance):
            raise SystemExit(f"fold-purity audit failed for outer fold {f}")
        if deranged:
            scalar = apply_derangement(scalar, derange_source, folds_series)
        row_scalar = row_scalar_from_series(c, scalar)
        tr_folds = [x for x in DEV_FOLDS if x != f]
        tr_rows = d.rows_for(tr_folds)
        va_rows = d.rows_for([f])
        if np.isfinite(row_scalar[d.rows_for([-1])]).any():
            raise SystemExit("pilot09 scalar unexpectedly filled lockbox rows")
        if not np.isfinite(row_scalar[tr_rows]).all() or not np.isfinite(row_scalar[va_rows]).all():
            raise SystemExit(f"pilot09 scalar has NaN/Inf on fold {f} train/valid rows")
        if len(tr_rows) > ABL["max_train_rows"]:
            tr_rows = np.sort(rng.choice(tr_rows, ABL["max_train_rows"], replace=False))

        xtr_base = PL._stack(mats, names, tr_rows, keep_idx)
        xtr = append_column(xtr_base, row_scalar[tr_rows])
        del xtr_base
        ds = lgb.Dataset(xtr, label=d.y[tr_rows], params=dict(p, objective="binary"), feature_name=[f"f{i}" for i in range(len(used_names))])
        booster = lgb.train(p, ds, num_boost_round=n_round)
        del xtr, ds

        xva_base = PL._stack(mats, names, va_rows, keep_idx)
        xva = append_column(xva_base, row_scalar[va_rows])
        del xva_base
        pred = booster.predict(xva).astype(np.float32)
        del xva
        oof[va_rows] = pred
        imp_sum += booster.feature_importance("gain")

        item = {"fold": int(f), "train_rows": int(len(tr_rows)), "valid_rows": int(len(va_rows)), "runtime_s": round(time.time() - t0, 1)}
        if f == SCORED_FOLD:
            fold0_auc = float(ts_auc_flat(pred, d.y[va_rows], d.t[va_rows]))
            item["ts_auc_scored"] = fold0_auc
            print(f"{exp_id}: scored fold 0 TS-AUC {fold0_auc:.6f}", flush=True)
        else:
            print(f"{exp_id}: generated fold-pure calibration OOF for fold {f}", flush=True)
        fit_summaries.append(item)
        scalar_summaries.append(
            {
                "outer_fold": int(f),
                "deranged": bool(deranged),
                "series_scalar_min": float(np.nanmin(scalar[np.isin(folds_series, DEV_FOLDS)])),
                "series_scalar_max": float(np.nanmax(scalar[np.isin(folds_series, DEV_FOLDS)])),
                "series_scalar_mean": float(np.nanmean(scalar[np.isin(folds_series, DEV_FOLDS)])),
                "provenance": provenance,
            }
        )

    if fold0_auc is None:
        raise SystemExit(f"{exp_id} did not produce fold-0 predictions")

    np.save(OOFDIR / f"{exp_id}.npy", oof)
    pd.DataFrame({"feature": used_names, "gain": imp_sum}).sort_values("gain", ascending=False).to_csv(
        OOFDIR / f"{exp_id}.importance.csv",
        index=False,
    )

    train_rows_reference = len(d.rows_for([1, 2, 3, 4]))
    res = {
        "experiment_id": exp_id,
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "git_sha": git_sha(),
        "agent": "codex-new-avenues",
        "hypothesis": hypothesis,
        "falsification_condition": falsification,
        "feature_set": ",".join(FULL + [col_name]),
        "n_features": len(used_names),
        "model": "lgbm",
        "objective": "binary",
        "folds": str(SCORED_FOLD),
        "random_seed": 0,
        "train_series": int((~np.isin(d.series_fold, [-1])).sum()),
        "train_rows": int(min(ABL["max_train_rows"], train_rows_reference)),
        "mean_oof_ts_auc": fold0_auc,
        "pooled_oof_ts_auc": fold0_auc,
        "per_fold_ts_auc": f"{fold0_auc:.5f}",
        "fold_std": 0.0,
        "persistence": "none",
        "sample_mode": "uniform",
        "training_runtime_s": round(time.time() - t0, 1),
        "causal_verified": "nested fold-purity audit + within-fold derangement + no online inputs",
        "test_reduced_touched": "no",
        "lockbox_touched": "no",
        "status": "recorded",
        "notes": notes,
        "protocol": "pilot_fold0_only",
    }
    append_result_append_only(res)
    return oof.astype(np.float64), {
        "result_row": res,
        "fit_summaries": fit_summaries,
        "scalar_summaries": scalar_summaries,
        "oof_artifact": str(OOFDIR / f"{exp_id}.npy"),
        "importance_artifact": str(OOFDIR / f"{exp_id}.importance.csv"),
        "foldpure_oof_support_note": (
            "Folds 1-4 were predicted only to provide fold-pure SCDF calibration "
            "support for the existing ensemble marginal path; no TS-AUC was "
            "computed or used for those folds."
        ),
    }


def gate_verdict(candidate_margin: float, control_margin: float) -> tuple[str, str, str, list[dict]]:
    gap = candidate_margin - control_margin
    failures = []
    if control_margin >= candidate_margin:
        failures.append({"gate": "deranged_control_not_worse", "threshold": "control < candidate", "observed_candidate_minus_control": gap, "message": "deranged scalar control matches or exceeds the real scalar"})
    elif gap < 0.0005:
        failures.append({"gate": "derangement_gap", "threshold": 0.0005, "observed_candidate_minus_control": gap, "message": "real scalar does not beat deranged control by +0.0005"})
    if candidate_margin < 0.0010:
        failures.append({"gate": "primary_marginal_vs_clone", "threshold": 0.0010, "observed": candidate_margin, "message": "candidate marginal_vs_clone is below +0.0010"})
    if control_margin >= candidate_margin:
        return "KILL", "KILL", "deranged scalar control matches or exceeds the real scalar", failures
    if gap < 0.0005:
        return "KILL", "KILL", "real scalar does not beat deranged control by +0.0005", failures
    if candidate_margin < 0.0010:
        return "KILL", "KILL", "candidate marginal_vs_clone is below +0.0010", failures
    if candidate_margin < 0.0020:
        return "WEAK", "NO_5FOLD", "candidate clears kill floor but remains in the weak +0.001 to +0.002 band", failures
    if candidate_margin < 0.0030:
        return "INTERESTING", "CONTINUE", "candidate clears the preregistered 5-fold continuation gate", failures
    if candidate_margin < 0.0050:
        return "SERIOUS", "CONTINUE", "candidate clears the serious screen band; confirm before any next pilot", failures
    if candidate_margin < 0.0080:
        return "MAJOR", "CONTINUE", "candidate clears the major screen band; confirm before any next pilot", failures
    return "BREAKTHROUGH", "CONTINUE", "candidate clears the breakthrough screen band; confirm before any next pilot", failures


def fold0_scalar_spearman(fp: np.ndarray, target: np.ndarray, eligible: np.ndarray, folds: np.ndarray) -> dict:
    scalar, provenance = fit_scalar_predictions(fp, target, eligible, folds, SCORED_FOLD)
    m = (folds == SCORED_FOLD) & eligible & np.isfinite(target) & np.isfinite(scalar)
    rho, p = spearmanr(scalar[m], target[m]) if int(m.sum()) >= 5 else (np.nan, np.nan)
    return {
        "fold": SCORED_FOLD,
        "n_series": int(m.sum()),
        "spearman": float(rho) if np.isfinite(rho) else None,
        "p": float(p) if np.isfinite(p) else None,
        "provenance_ok": fold_purity_ok(provenance),
    }


def write_report(result: dict) -> None:
    json_path = OUTDIR / "pilot09_difficulty_gate.json"
    md_path = OUTDIR / "pilot09_difficulty_gate.md"
    json_path.write_text(json.dumps(finite_float(result), indent=2, sort_keys=True) + "\n")

    cand = result[CANDIDATE_ID]
    ctrl = result[CONTROL_ID]
    cm = cand["ensemble_marginal"]
    em = ctrl["ensemble_marginal"]
    cp = cand["diagnostic_pack"]
    ep = ctrl["diagnostic_pack"]
    sentinel = result["rt600_sentinel"]
    checks = result["fold_purity_checks"]

    md = [
        "# PILOT 9(i) -- SCALAR HISTORICAL DIFFICULTY GATE",
        "",
        f"Pre-registration: `research/reports/new_avenues_2026/PILOT09_PREREG.md` at `{PREREG_SHA}`.",
        f"Experiment IDs: `{CANDIDATE_ID}` nested scalar candidate, `{CONTROL_ID}` deranged scalar control.",
        "",
        "## RT-600 Anchor",
        "",
        f"* Dev mean TS-AUC: `{sentinel['mean']:.6f}`.",
        f"* Dev pooled TS-AUC: `{sentinel['pooled']:.6f}`.",
        f"* Dev dominant-cell TS-AUC: `{sentinel['dominant_cell']:.6f}`.",
        f"* Fold-0 E0 RT600: `{sentinel['fold0_e0']:.6f}`.",
        f"* Fold-0 E1 RT600 + RT-401: `{sentinel['fold0_e1_seedclone']:.6f}`.",
        "",
        "## Candidate",
        "",
        "One series-constant scalar from the 23 history-only fingerprints, trained "
        "nested/fold-pure to predict RT-600 dominant-cell pair loss rate. The "
        "control deranges that scalar within each permanent fold.",
        "",
        "## Fold-Purity Checks",
        "",
        f"* Fingerprint names match preregistration: `{checks['fingerprint_names_ok']}`.",
        f"* Candidate fold-purity audit: `{checks['candidate_fold_purity_ok']}`.",
        f"* Control fold-purity audit: `{checks['control_fold_purity_ok']}`.",
        f"* Derangement audit: `{checks['derangement_audit']}`.",
        f"* Row scalar constant within checked series: `{checks['row_scalar_constant_check']}`.",
        f"* Lockbox scalar finite count: `{checks['lockbox_scalar_finite_count']}`.",
        f"* m07 parity note: {checks['m07_bayes_parity_note']}",
        "",
        "## Binding Marginal Result",
        "",
        "| arm | fold-0 standalone TS-AUC | E2 TS-AUC | marginal vs clone | gain vs RT600 |",
        "|---|---:|---:|---:|---:|",
        f"| RT600 |  | {cm['rt600_7stream']:.6f} |  |  |",
        f"| RT600 + RT-401 seed clone |  | {cm['rt600_plus_seedclone']:.6f} |  |  |",
        (
            f"| RT600 + {CANDIDATE_ID} | {cp['whole_fold']['candidate']:.6f} | "
            f"{cm[f'rt600_plus_{CANDIDATE_ID}']:.6f} | {cm['marginal_vs_clone']:+.6f} | "
            f"{cm['gain_vs_base']:+.6f} |"
        ),
        (
            f"| RT600 + {CONTROL_ID} | {ep['whole_fold']['candidate']:.6f} | "
            f"{em[f'rt600_plus_{CONTROL_ID}']:.6f} | {em['marginal_vs_clone']:+.6f} | "
            f"{em['gain_vs_base']:+.6f} |"
        ),
        "",
        f"Candidate minus deranged-control marginal: `{result['candidate_minus_control_marginal']:+.6f}`.",
        f"Fold-0 scalar target Spearman diagnostic: `{result['fold0_scalar_target_spearman']}`.",
        f"Verdict: **{result['verdict']}** ({result['continuation_status']}) -- {result['verdict_reason']}.",
        "Failed gates: "
        + ("; ".join(f"`{x['gate']}` ({x['message']})" for x in result["gate_failures"]) if result["gate_failures"] else "none"),
        "",
        "## Diagnostic Pack",
        "",
        "| candidate | whole-fold AUC | dominant-cell AUC | mature vs never | mature vs pre | rho vs RT600 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| `{CANDIDATE_ID}` | {cp['whole_fold']['candidate']:.6f} | "
            f"{cp['dominant_cell']['candidate']:.6f} | "
            f"{cp['mature_vs_neverbreak']['candidate']:.6f} | "
            f"{cp['mature_vs_prebreak']['candidate']:.6f} | "
            f"{cp['within_t_rank_corr_rt600']:+.4f} |"
        ),
        (
            f"| `{CONTROL_ID}` | {ep['whole_fold']['candidate']:.6f} | "
            f"{ep['dominant_cell']['candidate']:.6f} | "
            f"{ep['mature_vs_neverbreak']['candidate']:.6f} | "
            f"{ep['mature_vs_prebreak']['candidate']:.6f} | "
            f"{ep['within_t_rank_corr_rt600']:+.4f} |"
        ),
        "",
        "## Pair Flow",
        "",
        f"### {CANDIDATE_ID}",
        "",
        "| split | repairs | damage | net | sampled pairs |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in cand["pair_flow"].items():
        md.append(f"| `{name}` | {row['repairs']} | {row['damage']} | {row['net_pair_lift']} | {row['total_pairs_sampled']} |")
    md += ["", f"### {CONTROL_ID}", "", "| split | repairs | damage | net | sampled pairs |", "|---|---:|---:|---:|---:|"]
    for name, row in ctrl["pair_flow"].items():
        md.append(f"| `{name}` | {row['repairs']} | {row['damage']} | {row['net_pair_lift']} | {row['total_pairs_sampled']} |")
    md += [
        "",
        "## Interpretation",
        "",
        result["interpretation"],
        "",
        f"Total runtime: `{result['runtime_s']:.1f}s`.",
    ]
    md_path.write_text("\n".join(md) + "\n")


def main() -> None:
    t0 = time.time()
    c = Ctx()
    _ = load_oof(SPECIALISTS + ["RT-401"])
    base = rt600_blend(c)
    sentinel = rt600_sentinel(c, base)

    fp, fp_names = history_fingerprints()
    if fp.shape[1] != len(EXPECTED_FP_NAMES) or list(fp_names) != EXPECTED_FP_NAMES:
        raise SystemExit(f"fingerprint names/shape mismatch: {fp.shape}, {fp_names}")
    folds_series = pd.read_parquet(ROOT / "research" / "folds" / "folds.parquet").fold.to_numpy()
    if len(folds_series) != fp.shape[0]:
        raise SystemExit("fold/fingerprint series length mismatch")
    sl, sw = per_series_loss(base, c, c.dev)
    target = sl / np.maximum(sw, 1.0)
    eligible = np.isin(folds_series, DEV_FOLDS) & (sw > MIN_SERIES_WEIGHT) & np.isfinite(target)

    dsrc = derangement_source(folds_series, seed=0)
    scalar0, prov0 = fit_scalar_predictions(fp, target, eligible, folds_series, SCORED_FOLD)
    dscalar0 = apply_derangement(scalar0, dsrc, folds_series)
    row0 = row_scalar_from_series(c, scalar0)
    checks = {
        "fingerprint_names_ok": True,
        "candidate_fold_purity_ok": fold_purity_ok(prov0),
        "control_fold_purity_ok": fold_purity_ok(prov0),
        "derangement_audit": derangement_audit(scalar0, dscalar0, dsrc, folds_series),
        "row_scalar_constant_check": {},
        "lockbox_scalar_finite_count": int(np.isfinite(row0[c.d.rows_for([-1])]).sum()),
        "min_series_weight": MIN_SERIES_WEIGHT,
        "eligible_scalar_targets": int(eligible.sum()),
        "m07_bayes_parity_note": (
            "Pilot 9(i) does not recompute m07_bayes::bo_p_lt25_z; it uses the "
            "existing cached base-bank features and one nested history-fingerprint scalar."
        ),
    }
    for sid in np.linspace(0, fp.shape[0] - 1, 8).astype(int):
        rows = np.flatnonzero(c.d.sidx == int(sid))
        vals = row0[rows]
        finite = vals[np.isfinite(vals)]
        checks["row_scalar_constant_check"][str(int(sid))] = bool(len(finite) == 0 or np.all(finite == finite[0]))
    if not checks["candidate_fold_purity_ok"] or not checks["derangement_audit"]["ok"]:
        raise SystemExit(f"Pilot 9 fold-purity/derangement checks failed: {checks}")
    if checks["lockbox_scalar_finite_count"] != 0:
        raise SystemExit("Pilot 9 scalar filled lockbox rows")

    mats, names = PL.load_features(FULL, screen=False)
    common_hypothesis = (
        "Pilot 9(i) tests whether one nested history-derived scalar predicting RT-600 "
        "dominant-cell loss propensity carries useful interaction information."
    )
    candidate_oof, candidate_train = train_nested_oof(
        CANDIDATE_ID,
        CANDIDATE_COL,
        False,
        fp,
        target,
        eligible,
        folds_series,
        dsrc,
        mats,
        names,
        c,
        common_hypothesis + " The real scalar should add ensemble alpha beyond an exchangeable seed clone.",
        "KILL if candidate marginal_vs_clone < +0.0010 or if the deranged scalar control matches within +0.0005.",
        "Pilot 9(i) nested scalar difficulty arm. Fold-0 screen only; folds 1-4 generated only as fold-pure SCDF calibration support.",
    )
    control_oof, control_train = train_nested_oof(
        CONTROL_ID,
        CONTROL_COL,
        True,
        fp,
        target,
        eligible,
        folds_series,
        dsrc,
        mats,
        names,
        c,
        common_hypothesis + " Within-fold derangement is the binding memorization/prior control.",
        "Control must be clearly worse than the real scalar; if not, J1 is killed as non-load-bearing or memorization-prone.",
        "Pilot 9(i) within-fold deranged scalar control. Fold-0 screen only; folds 1-4 generated only as fold-pure SCDF calibration support.",
    )

    candidate = candidate_result(CANDIDATE_ID, candidate_oof, base, c)
    control = candidate_result(CONTROL_ID, control_oof, base, c)
    candidate["training"] = candidate_train
    control["training"] = control_train
    candidate_margin = candidate["ensemble_marginal"]["marginal_vs_clone"]
    control_margin = control["ensemble_marginal"]["marginal_vs_clone"]
    verdict, continuation_status, reason, gate_failures = gate_verdict(candidate_margin, control_margin)

    if verdict == "KILL":
        interpretation = (
            "The preregistered scalar historical-difficulty gate failed a Pilot 9(i) binding gate. "
            "This falsifies the J1 nested one-scalar construction under the fixed 23 history-only "
            "fingerprints, RT-600 dominant-cell loss-rate target, within-fold derangement control, "
            "and Mode-A fold-0 ABL screen; it does not falsify all historical-DGP conditioning."
        )
    elif continuation_status == "CONTINUE":
        interpretation = (
            "The real scalar cleared the preregistered fold-0 continuation gate and beat the "
            "deranged scalar by the required gap. The next step is formal confirmation."
        )
    else:
        interpretation = (
            "The real scalar cleared the kill floor but remained in the weak band. It is not "
            "promoted without an explicit continuation decision."
        )

    result = {
        "generated": "2026-08-24",
        "branch": "research/new-avenues-pilots-2026",
        "prereg_sha": PREREG_SHA,
        "head_sha": git_sha(),
        "exp_ids": [CANDIDATE_ID, CONTROL_ID],
        "feature_columns": [CANDIDATE_COL, CONTROL_COL],
        "scalar_definitions": {
            "fingerprint_names": EXPECTED_FP_NAMES,
            "target": "RT-600 dominant-cell pair inversion loss rate",
            "min_series_weight": MIN_SERIES_WEIGHT,
            "scalar_params": SCALAR_PARAMS,
            "scalar_rounds": SCALAR_ROUNDS,
        },
        "fold_purity_checks": checks,
        "rt600_sentinel": sentinel,
        "fold0_scalar_target_spearman": fold0_scalar_spearman(fp, target, eligible, folds_series),
        CANDIDATE_ID: candidate,
        CONTROL_ID: control,
        "candidate_minus_control_marginal": candidate_margin - control_margin,
        "gate_failures": gate_failures,
        "verdict": verdict,
        "continuation_status": continuation_status,
        "verdict_reason": reason,
        "interpretation": interpretation,
        "runtime_s": time.time() - t0,
    }
    write_report(result)
    print(
        json.dumps(
            finite_float(
                {
                    "verdict": verdict,
                    "continuation_status": continuation_status,
                    "reason": reason,
                    CANDIDATE_ID: candidate["ensemble_marginal"],
                    CONTROL_ID: control["ensemble_marginal"],
                    "candidate_minus_control": candidate_margin - control_margin,
                    "runtime_s": result["runtime_s"],
                }
            ),
            indent=2,
        )
    )
    print(f"wrote {OUTDIR / 'pilot09_difficulty_gate.json'}")
    print(f"wrote {OUTDIR / 'pilot09_difficulty_gate.md'}")


if __name__ == "__main__":
    main()
