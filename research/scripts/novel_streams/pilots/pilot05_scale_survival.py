"""Pilot 5: scale-survival count plus coarse-graining exponent.

Scored candidates:
  RT-1204: cross-scale summary features.
  RT-1205: individual per-scale surprise control.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("SBR_ROOT", Path(__file__).resolve().parents[4]))
os.environ.setdefault("SBR_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(ROOT / "research" / "scripts" / "novel_streams"))

import lightgbm as lgb
import numpy as np
import pandas as pd

import sbr.pipeline as PL
from harness import (
    StreamingMechanism,
    cell_rows,
    diagnostic_pack,
    marginal,
    pair_flow_by_cell,
    rt600_blend,
    verify as harness_verify,
)
from sbr.features.base import REGISTRY, check_prefix_invariance, load_all, make_ctx
from sbr.features.m13_scale_survival import (
    INDIV_COLS,
    Q01,
    Q05,
    SCALES,
    SUMMARY_COLS,
    scale_survival_features,
)
from sbr.metric import ts_auc_flat
from sbr.store import load_store
from wave2_lib import ABL, FULL
from wave5_lib import Ctx, SPECIALISTS, load_oof

PREREG_SHA = "094c50f"
SUMMARY_ID = "RT-1204"
CONTROL_ID = "RT-1205"
SUMMARY_MODULE = "m13_scale_surv"
CONTROL_MODULE = "m13_scale_indiv"
OUTDIR = ROOT / "research" / "reports" / "new_avenues_2026"
OOFDIR = ROOT / "research" / "oof"
FEATDIR = ROOT / "cache" / "features"
OUTDIR.mkdir(parents=True, exist_ok=True)
OOFDIR.mkdir(parents=True, exist_ok=True)
FEATDIR.mkdir(parents=True, exist_ok=True)

PARAMS = dict(objective="binary", num_threads=2, verbose=-1)
PARAMS.update(ABL["params"])
PARAMS.setdefault("bagging_freq", 1)
FOLDS_FOR_FOLDPURE_OOF = (0, 1, 2, 3, 4)
SCORED_FOLD = 0


class ScaleSummaryMech(StreamingMechanism):
    name = "pilot05_scale_summary"
    cols = SUMMARY_COLS

    def emit(self, hist, online):
        summary, _ = scale_survival_features(make_ctx(hist, online))
        return summary


class ScaleIndividualMech(StreamingMechanism):
    name = "pilot05_scale_individual"
    cols = INDIV_COLS

    def emit(self, hist, online):
        _, indiv = scale_survival_features(make_ctx(hist, online))
        return indiv


def finite_float(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if np.isfinite(x) else None
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, dict):
        return {str(k): finite_float(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite_float(v) for v in x]
    return x


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "nogit"


def run_causal_checks(c: Ctx) -> dict:
    load_all()
    checks: dict[str, object] = {
        "thresholds": {"q05": Q05, "q01": Q01},
        "scales": list(SCALES),
    }

    for name, mech in (("summary", ScaleSummaryMech()), ("individual", ScaleIndividualMech())):
        ok, msg = harness_verify(mech)
        checks[f"harness_verify_{name}"] = {"ok": ok, "message": msg}
        if not ok:
            raise SystemExit(f"harness.verify failed for {name}: {msg}")

    st = load_store(str(ROOT / "cache" / "store"))
    dev_series = np.unique(c.d.sidx[c.dev])
    lens = c.n_online[dev_series]
    pick = dev_series[np.argsort(lens)[np.linspace(0, len(lens) - 1, 8).astype(int)]]
    prefix_results = {}
    for module in (SUMMARY_MODULE, CONTROL_MODULE):
        module_results = []
        for sid in pick:
            hist = st.hist(int(sid))
            online = st.online(int(sid))
            ok, msg = check_prefix_invariance(module, hist, online, cuts=(3, 10, 37, 111), atol=0.0)
            module_results.append({"series": int(sid), "ok": ok, "message": msg})
            if not ok:
                raise SystemExit(f"{module} prefix check failed on series {int(sid)}: {msg}")
        prefix_results[module] = module_results
    checks["module_prefix_invariance"] = prefix_results

    sid = int(pick[-1])
    hist = st.hist(sid)
    online = st.online(sid)
    summary, indiv = scale_survival_features(make_ctx(hist, online))
    first_valid = {
        "b1_row0_finite": bool(np.isfinite(indiv[0, 0])),
        "b2_row0_nan": bool(np.isnan(indiv[0, 1])),
        "b32_row30_nan": bool(np.isnan(indiv[30, -1])) if len(indiv) > 30 else None,
        "b32_row31_finite": bool(np.isfinite(indiv[31, -1])) if len(indiv) > 31 else None,
        "slope_row0_nan": bool(np.isnan(summary[0, SUMMARY_COLS.index("slope_logscale")])),
        "slope_row1_finite": bool(np.isfinite(summary[1, SUMMARY_COLS.index("slope_logscale")])) if len(summary) > 1 else None,
    }
    if not all(v for v in first_valid.values() if v is not None):
        raise SystemExit(f"first-valid check failed: {first_valid}")
    checks["first_valid_nan_semantics"] = first_valid

    cut = min(37, max(2, len(online) // 2))
    mutated = online.copy()
    mutated[cut:] = mutated[cut:][::-1] * -5.0 + 2.0
    m_summary, m_indiv = scale_survival_features(make_ctx(hist, mutated))
    future_ok = bool(
        np.allclose(summary[:cut], m_summary[:cut], rtol=0.0, atol=0.0, equal_nan=True)
        and np.allclose(indiv[:cut], m_indiv[:cut], rtol=0.0, atol=0.0, equal_nan=True)
    )
    if not future_ok:
        raise SystemExit("future-mutation check failed")
    checks["future_mutation_prefix_ok"] = future_ok

    r_summary, r_indiv = scale_survival_features(make_ctx(hist, online))
    replay_ok = bool(
        np.array_equal(summary, r_summary, equal_nan=True)
        and np.array_equal(indiv, r_indiv, equal_nan=True)
    )
    if not replay_ok:
        raise SystemExit("deterministic replay check failed")
    checks["deterministic_replay_ok"] = replay_ok
    return checks


def rt600_sentinel(c: Ctx, base: np.ndarray) -> dict:
    mean_auc, per_fold = c.score(base)
    pooled = c.pooled(base)
    rr = cell_rows(c, c.dev)
    dominant = float(ts_auc_flat(base[rr], c.d.y[rr], c.d.t[rr]))
    marg = marginal(base, c, fold=SCORED_FOLD, label="rt600_self")
    out = {
        "mean": mean_auc,
        "per_fold": per_fold,
        "pooled": pooled,
        "dominant_cell": dominant,
        "fold0_e0": marg["rt600_7stream"],
        "fold0_e1_seedclone": marg["rt600_plus_seedclone"],
    }
    expected = {
        "mean": 0.62581,
        "pooled": 0.625627,
        "dominant_cell": 0.66428,
        "fold0_e0": 0.63828,
        "fold0_e1_seedclone": 0.63859,
    }
    diffs = {k: out[k] - expected[k] for k in expected}
    out["expected"] = expected
    out["diffs"] = diffs
    out["ok"] = bool(all(abs(d) < 0.005 for d in diffs.values()))
    if not out["ok"]:
        raise SystemExit(f"RT600 sentinel materially differs: {out}")
    return out


def build_dev_feature_cache(c: Ctx, module_name: str) -> dict:
    load_all()
    if module_name not in REGISTRY:
        raise SystemExit(f"unknown feature module {module_name}")
    mod = REGISTRY[module_name]
    t0 = time.time()
    st = load_store(str(ROOT / "cache" / "store"))
    cols, _ = mod.fn(make_ctx(st.hist(0), st.online(0)))
    A = np.full((len(c.d.y), len(cols)), np.nan, dtype=np.float32)
    dev_series = np.unique(c.d.sidx[c.dev])
    for count, sid in enumerate(dev_series):
        rows = c.dev[c.d.sidx[c.dev] == sid]
        rows = rows[np.argsort(c.d.t[rows], kind="stable")]
        names, feats = mod.fn(make_ctx(st.hist(int(sid)), st.online(int(sid))))
        if names != cols:
            raise SystemExit(f"{module_name} column mismatch on series {int(sid)}")
        if len(feats) != len(rows):
            raise SystemExit(f"{module_name} row mismatch on series {int(sid)}")
        A[rows] = feats
        if count and count % 1000 == 0:
            print(f"{module_name}: built {count}/{len(dev_series)} series {time.time() - t0:.0f}s", flush=True)

    np.save(FEATDIR / f"{module_name}.npy", A)
    (FEATDIR / f"{module_name}.cols.json").write_text(
        json.dumps({"cols": cols, "version": mod.version, "owner": mod.owner}, sort_keys=True) + "\n"
    )
    return {
        "module": module_name,
        "cols": cols,
        "shape": [int(A.shape[0]), int(A.shape[1])],
        "dev_series": int(len(dev_series)),
        "lockbox_rows_filled": int(np.isfinite(A[c.d.rows_for([-1])]).sum()) if hasattr(c.d, "rows_for") else 0,
        "runtime_s": time.time() - t0,
    }


def load_feature_bank(modules: list[str]):
    mats, names = PL.load_features(modules, screen=False)
    keep_idx = np.arange(len(names), dtype=np.int64)
    return mats, names, keep_idx


def train_foldpure_oof(exp_id: str, module_name: str, hypothesis: str, falsification: str, notes: str) -> tuple[np.ndarray, dict]:
    t0 = time.time()
    d = PL.Data(screen=False)
    modules = FULL + [module_name]
    mats, names, keep_idx = load_feature_bank(modules)
    used_names = [names[i] for i in keep_idx]
    p = dict(PARAMS)
    n_round = int(p.pop("n_estimators", 600))
    rng = np.random.default_rng(0)
    oof = np.full(len(d.y), np.nan, dtype=np.float32)
    imp_sum = np.zeros(len(keep_idx), dtype=np.float64)
    fold0_auc = None
    fit_summaries = []

    for f in FOLDS_FOR_FOLDPURE_OOF:
        tr_folds = [x for x in (0, 1, 2, 3, 4) if x != f]
        tr_rows = d.rows_for(tr_folds)
        va_rows = d.rows_for([f])
        if len(tr_rows) > ABL["max_train_rows"]:
            tr_rows = np.sort(rng.choice(tr_rows, ABL["max_train_rows"], replace=False))

        Xtr = PL._stack(mats, names, tr_rows, keep_idx)
        ytr = d.y[tr_rows]
        ds = lgb.Dataset(Xtr, label=ytr, params=dict(p, objective="binary"), feature_name=[f"f{i}" for i in range(len(keep_idx))])
        booster = lgb.train(p, ds, num_boost_round=n_round)
        del Xtr, ds

        Xva = PL._stack(mats, names, va_rows, keep_idx)
        pred = booster.predict(Xva).astype(np.float32)
        del Xva
        oof[va_rows] = pred
        imp_sum += booster.feature_importance("gain")
        item = {
            "fold": int(f),
            "train_rows": int(len(tr_rows)),
            "valid_rows": int(len(va_rows)),
            "runtime_s": round(time.time() - t0, 1),
        }
        if f == SCORED_FOLD:
            fold0_auc = float(ts_auc_flat(pred, d.y[va_rows], d.t[va_rows]))
            item["ts_auc_scored"] = fold0_auc
            print(f"{exp_id}: scored fold 0 TS-AUC {fold0_auc:.6f}", flush=True)
        else:
            print(f"{exp_id}: generated fold-pure calibration OOF for fold {f}", flush=True)
        fit_summaries.append(item)

    if fold0_auc is None:
        raise SystemExit(f"{exp_id} did not produce fold-0 predictions")

    np.save(OOFDIR / f"{exp_id}.npy", oof)
    imp = pd.DataFrame({"feature": used_names, "gain": imp_sum}).sort_values("gain", ascending=False)
    imp.to_csv(OOFDIR / f"{exp_id}.importance.csv", index=False)

    train_rows_reference = len(d.rows_for([1, 2, 3, 4]))
    res = {
        "experiment_id": exp_id,
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "git_sha": git_sha(),
        "agent": "codex-new-avenues",
        "hypothesis": hypothesis,
        "falsification_condition": falsification,
        "feature_set": ",".join(modules),
        "n_features": len(keep_idx),
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
        "causal_verified": "harness.verify + prefix-invariance@module + future-mutation",
        "test_reduced_touched": "no",
        "lockbox_touched": "no",
        "status": "recorded",
        "notes": notes,
        "protocol": "pilot_fold0_only",
    }
    PL.append_result(res)
    return oof.astype(np.float64), {
        "result_row": res,
        "fit_summaries": fit_summaries,
        "oof_artifact": str(OOFDIR / f"{exp_id}.npy"),
        "importance_artifact": str(OOFDIR / f"{exp_id}.importance.csv"),
        "foldpure_oof_support_note": (
            "Folds 1-4 were predicted only to provide fold-pure SCDF calibration "
            "support for the existing ensemble marginal path; no TS-AUC was "
            "computed or used for those folds."
        ),
    }


def candidate_result(exp_id: str, cand: np.ndarray, base: np.ndarray, c: Ctx) -> dict:
    pack = diagnostic_pack(c, cand, base, fold=SCORED_FOLD, label=exp_id)
    pair_flow = pair_flow_by_cell(base, cand, c, fold=SCORED_FOLD, n_pairs_per_t=20, seed=0)
    marg = marginal(cand, c, fold=SCORED_FOLD, label=exp_id)
    return {"diagnostic_pack": pack, "pair_flow": pair_flow, "ensemble_marginal": marg}


def gate_verdict(summary_margin: float, control_margin: float) -> tuple[str, str, str, list[dict]]:
    gap = summary_margin - control_margin
    failures = []
    if summary_margin < 0.0010:
        failures.append(
            {
                "gate": "primary_marginal_vs_clone",
                "threshold": 0.0010,
                "observed": summary_margin,
                "message": "summary marginal_vs_clone is below +0.0010",
            }
        )
    if control_margin >= summary_margin:
        failures.append(
            {
                "gate": "individual_control_not_worse",
                "threshold": "control < summary",
                "observed_summary_minus_control": gap,
                "message": "individual-scale control matches or exceeds summary",
            }
        )
    elif gap < 0.0005:
        failures.append(
            {
                "gate": "summary_control_distinguishability",
                "threshold": 0.0005,
                "observed_summary_minus_control": gap,
                "message": "summary-control gap is below the preregistered +0.0005 distinguishability floor",
            }
        )
    if summary_margin < 0.0010:
        return "KILL", "KILL", failures[0]["message"], failures
    if control_margin >= summary_margin:
        return "KILL", "KILL", "individual-scale control matches or exceeds summary", failures
    if gap < 0.0005:
        return "KILL", "KILL", failures[0]["message"], failures
    if summary_margin < 0.0020:
        return "WEAK", "NO_5FOLD", "summary clears kill floor but remains in the weak +0.001 to +0.002 band", failures
    if summary_margin < 0.0030:
        return "INTERESTING", "CONTINUE", "summary clears the preregistered 5-fold continuation gate", failures
    if summary_margin < 0.0050:
        return "SERIOUS", "CONTINUE", "summary clears the serious screen band; confirm before any next pilot", failures
    if summary_margin < 0.0080:
        return "MAJOR", "CONTINUE", "summary clears the major screen band; confirm before any next pilot", failures
    return "BREAKTHROUGH", "CONTINUE", "summary clears the breakthrough screen band; confirm before any next pilot", failures


def write_report(result: dict) -> None:
    json_path = OUTDIR / "pilot05_scale_survival.json"
    md_path = OUTDIR / "pilot05_scale_survival.md"
    json_path.write_text(json.dumps(finite_float(result), indent=2, sort_keys=True) + "\n")

    summary = result[SUMMARY_ID]
    control = result[CONTROL_ID]
    sm = summary["ensemble_marginal"]
    cm = control["ensemble_marginal"]
    sp = summary["diagnostic_pack"]
    cp = control["diagnostic_pack"]
    checks = result["causal_checks"]
    sentinel = result["rt600_sentinel"]

    md = [
        "# PILOT 5 -- SCALE-SURVIVAL COARSE-GRAINING",
        "",
        f"Pre-registration: `research/reports/new_avenues_2026/PILOT05_PREREG.md` at `{PREREG_SHA}`.",
        f"Experiment IDs: `{SUMMARY_ID}` summary candidate, `{CONTROL_ID}` individual-scale control.",
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
        "AR(2) residual-square stream, causally coarse-grained at "
        "`{1,2,4,8,16,32}`. Each scale is compared against its own historical "
        "same-scale block-mean null. Surprise is two-sided empirical tail "
        "evidence, with fixed thresholds q05=`1.301029995664` and q01=`2.0`.",
        "",
        f"`{SUMMARY_ID}` emits five summary columns: survival counts at q05/q01, "
        "largest surviving scale code at q05/q01, and log-scale surprise slope.",
        f"`{CONTROL_ID}` emits the six individual per-scale surprises only.",
        "",
        "## Causality Checks",
        "",
        f"* Harness summary: `{checks['harness_verify_summary']['message']}`.",
        f"* Harness individual: `{checks['harness_verify_individual']['message']}`.",
        f"* Future mutation prefix check: `{checks['future_mutation_prefix_ok']}`.",
        f"* Deterministic replay: `{checks['deterministic_replay_ok']}`.",
        f"* First-valid semantics: `{checks['first_valid_nan_semantics']}`.",
        "",
        "## Binding Marginal Result",
        "",
        "| arm | fold-0 standalone TS-AUC | E2 TS-AUC | marginal vs clone | gain vs RT600 |",
        "|---|---:|---:|---:|---:|",
        f"| RT600 |  | {sm['rt600_7stream']:.6f} |  |  |",
        f"| RT600 + RT-401 seed clone |  | {sm['rt600_plus_seedclone']:.6f} |  |  |",
        (
            f"| RT600 + {SUMMARY_ID} | {summary['diagnostic_pack']['whole_fold']['candidate']:.6f} | "
            f"{sm[f'rt600_plus_{SUMMARY_ID}']:.6f} | {sm['marginal_vs_clone']:+.6f} | "
            f"{sm['gain_vs_base']:+.6f} |"
        ),
        (
            f"| RT600 + {CONTROL_ID} | {control['diagnostic_pack']['whole_fold']['candidate']:.6f} | "
            f"{cm[f'rt600_plus_{CONTROL_ID}']:.6f} | {cm['marginal_vs_clone']:+.6f} | "
            f"{cm['gain_vs_base']:+.6f} |"
        ),
        "",
        f"Summary minus individual-scale marginal: `{result['summary_minus_control_marginal']:+.6f}`.",
        f"Verdict: **{result['verdict']}** ({result['continuation_status']}) -- {result['verdict_reason']}.",
        "Failed gates: "
        + ("; ".join(f"`{x['gate']}` ({x['message']})" for x in result["gate_failures"]) if result["gate_failures"] else "none"),
        "",
        "## Diagnostic Pack",
        "",
        "| candidate | whole-fold AUC | dominant-cell AUC | mature vs never | mature vs pre | rho vs RT600 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| `{SUMMARY_ID}` | {sp['whole_fold']['candidate']:.6f} | "
            f"{sp['dominant_cell']['candidate']:.6f} | "
            f"{sp['mature_vs_neverbreak']['candidate']:.6f} | "
            f"{sp['mature_vs_prebreak']['candidate']:.6f} | "
            f"{sp['within_t_rank_corr_rt600']:+.4f} |"
        ),
        (
            f"| `{CONTROL_ID}` | {cp['whole_fold']['candidate']:.6f} | "
            f"{cp['dominant_cell']['candidate']:.6f} | "
            f"{cp['mature_vs_neverbreak']['candidate']:.6f} | "
            f"{cp['mature_vs_prebreak']['candidate']:.6f} | "
            f"{cp['within_t_rank_corr_rt600']:+.4f} |"
        ),
        "",
        "## Pair Flow",
        "",
        f"### {SUMMARY_ID}",
        "",
        "| split | repairs | damage | net | sampled pairs |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in summary["pair_flow"].items():
        md.append(
            f"| `{name}` | {row['repairs']} | {row['damage']} | "
            f"{row['net_pair_lift']} | {row['total_pairs_sampled']} |"
        )
    md += ["", f"### {CONTROL_ID}", "", "| split | repairs | damage | net | sampled pairs |", "|---|---:|---:|---:|---:|"]
    for name, row in control["pair_flow"].items():
        md.append(
            f"| `{name}` | {row['repairs']} | {row['damage']} | "
            f"{row['net_pair_lift']} | {row['total_pairs_sampled']} |"
        )
    md += [
        "",
        "## Interpretation",
        "",
        result["interpretation"],
        "",
        f"Feature build runtime: `{result['feature_build_runtime_s']:.1f}s`.",
        f"Total runtime: `{result['runtime_s']:.1f}s`.",
    ]
    md_path.write_text("\n".join(md) + "\n")


def main() -> None:
    t0 = time.time()
    c = Ctx()
    _ = load_oof(SPECIALISTS + ["RT-401"])
    base = rt600_blend(c)

    checks = run_causal_checks(c)
    sentinel = rt600_sentinel(c, base)

    build_summary = build_dev_feature_cache(c, SUMMARY_MODULE)
    build_control = build_dev_feature_cache(c, CONTROL_MODULE)
    feature_build_runtime = build_summary["runtime_s"] + build_control["runtime_s"]

    common_hypothesis = (
        "Pilot 5 tests whether persistent breaks survive dyadic temporal coarse-graining "
        "on an AR(2) residual-square stream."
    )
    summary_oof, summary_train = train_foldpure_oof(
        SUMMARY_ID,
        SUMMARY_MODULE,
        common_hypothesis + " Cross-scale count/largest-scale/slope summaries should add ensemble alpha.",
        "KILL if summary marginal_vs_clone < +0.0010, or if individual-scale control matches within +0.0005.",
        "Pilot 5 summary arm. Fold-0 screen only; folds 1-4 generated only as fold-pure SCDF calibration support.",
    )
    control_oof, control_train = train_foldpure_oof(
        CONTROL_ID,
        CONTROL_MODULE,
        common_hypothesis + " Individual per-scale surprises are the binding control for generic extra-window alpha.",
        "Control must not match or exceed the summary arm; if it does, scale-survival hypothesis is killed.",
        "Pilot 5 individual-scale control. Fold-0 screen only; folds 1-4 generated only as fold-pure SCDF calibration support.",
    )

    summary_result = candidate_result(SUMMARY_ID, summary_oof, base, c)
    control_result = candidate_result(CONTROL_ID, control_oof, base, c)
    summary_result["training"] = summary_train
    control_result["training"] = control_train

    summary_margin = summary_result["ensemble_marginal"]["marginal_vs_clone"]
    control_margin = control_result["ensemble_marginal"]["marginal_vs_clone"]
    verdict, continuation_status, reason, gate_failures = gate_verdict(summary_margin, control_margin)

    if verdict == "KILL":
        interpretation = (
            "Explicit scale-survival count / largest-scale / log-scale decay summaries "
            "on the frozen dyadic AR(2) residual-square coarse-grained stream failed "
            "a binding Pilot 5 gate. This falsifies this E1/E4 functional under the "
            "preregistered fold-0 screen; it does not falsify all multiscale "
            "representations or all residual-scale detectors."
        )
    elif continuation_status == "CONTINUE":
        interpretation = (
            "The summary arm cleared the preregistered fold-0 continuation gate and "
            "beat the individual-scale control by the required gap. The next step is "
            "5-fold confirmation before moving to Pilot 6."
        )
    else:
        interpretation = (
            "The summary arm cleared the kill floor but remained in the weak band. "
            "It is not promoted without an explicit continuation decision."
        )

    result = {
        "generated": "2026-08-24",
        "branch": "research/new-avenues-pilots-2026",
        "prereg_sha": PREREG_SHA,
        "head_sha": git_sha(),
        "exp_ids": [SUMMARY_ID, CONTROL_ID],
        "feature_modules": [SUMMARY_MODULE, CONTROL_MODULE],
        "feature_definitions": {
            "scales": list(SCALES),
            "thresholds": {"q05": Q05, "q01": Q01},
            "summary_cols": SUMMARY_COLS,
            "individual_cols": INDIV_COLS,
            "residual_stream": "AR(2) residual squared, standardized by historical residual sigma",
            "null": "per-scale two-sided empirical historical block-mean null",
        },
        "causal_checks": checks,
        "rt600_sentinel": sentinel,
        "feature_cache": {SUMMARY_MODULE: build_summary, CONTROL_MODULE: build_control},
        SUMMARY_ID: summary_result,
        CONTROL_ID: control_result,
        "summary_minus_control_marginal": summary_margin - control_margin,
        "gate_failures": gate_failures,
        "verdict": verdict,
        "continuation_status": continuation_status,
        "verdict_reason": reason,
        "interpretation": interpretation,
        "feature_build_runtime_s": feature_build_runtime,
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
                    SUMMARY_ID: summary_result["ensemble_marginal"],
                    CONTROL_ID: control_result["ensemble_marginal"],
                    "summary_minus_control": summary_margin - control_margin,
                    "runtime_s": result["runtime_s"],
                }
            ),
            indent=2,
        )
    )
    print(f"wrote {OUTDIR / 'pilot05_scale_survival.json'}")
    print(f"wrote {OUTDIR / 'pilot05_scale_survival.md'}")


if __name__ == "__main__":
    main()
