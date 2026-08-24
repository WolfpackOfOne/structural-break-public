"""Pilot 8: weighted conformal test martingale and e-value aggregation.

Scored arms:
  RT-1216: weighted conformal test martingale features.
  RT-1217: matched unweighted conformal test martingale control.
  RT-1218: parameter-free e-value aggregation direct-score arm.
"""
from __future__ import annotations

import csv
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

import numpy as np

from harness import StreamingMechanism, rt600_blend
from pilot06_spectral_impulse import (
    append_result_append_only,
    candidate_result,
    finite_float,
    git_sha,
    rt600_sentinel,
    train_foldpure_oof,
)
from sbr.features.base import REGISTRY, check_prefix_invariance, load_all, make_ctx
from sbr.features.m18_weighted_ctm import (
    H2_EVALUE_COLS,
    HIST_TAIL_RATE_MAX,
    HIST_TAIL_RATE_MIN,
    OMEGA_MIN,
    TAIL_WEIGHT_THRESHOLD,
    UNWEIGHTED_COLS,
    WEIGHTED_COLS,
    benign_tail_weight,
    h2_log_mean_evalue,
    unweighted_ctm_features,
    weighted_ctm_features,
)
from sbr.metric import ts_auc_flat
from sbr.store import load_store
from wave5_lib import Ctx, SPECIALISTS, load_oof

PREREG_SHA = "83c3994"
WEIGHTED_ID = "RT-1216"
UNWEIGHTED_ID = "RT-1217"
H2_ID = "RT-1218"
WEIGHTED_MODULE = "m18_wctm"
UNWEIGHTED_MODULE = "m18_uctm"
OUTDIR = ROOT / "research" / "reports" / "new_avenues_2026"
OOFDIR = ROOT / "research" / "oof"
FEATDIR = ROOT / "cache" / "features"
OUTDIR.mkdir(parents=True, exist_ok=True)
OOFDIR.mkdir(parents=True, exist_ok=True)
FEATDIR.mkdir(parents=True, exist_ok=True)
SCORED_FOLD = 0


class WeightedMech(StreamingMechanism):
    name = "pilot08_weighted_ctm"
    cols = WEIGHTED_COLS

    def emit(self, hist, online):
        return weighted_ctm_features(make_ctx(hist, online))


class UnweightedMech(StreamingMechanism):
    name = "pilot08_unweighted_ctm"
    cols = UNWEIGHTED_COLS

    def emit(self, hist, online):
        return unweighted_ctm_features(make_ctx(hist, online))


def _m07_column_indices() -> tuple[list[int], list[str]]:
    meta = json.loads((FEATDIR / "m07_bayes.cols.json").read_text())
    cols = list(meta["cols"])
    missing = [c for c in H2_EVALUE_COLS if c not in cols]
    if missing:
        raise SystemExit(f"missing preregistered m07 e-process columns: {missing}")
    return [cols.index(c) for c in H2_EVALUE_COLS], cols


def run_causal_checks(c: Ctx) -> dict:
    load_all()
    checks: dict[str, object] = {
        "weighted_cols": WEIGHTED_COLS,
        "unweighted_cols": UNWEIGHTED_COLS,
        "h2_evalue_cols": H2_EVALUE_COLS,
        "tail_weight_threshold": TAIL_WEIGHT_THRESHOLD,
        "hist_tail_rate_clip": [HIST_TAIL_RATE_MIN, HIST_TAIL_RATE_MAX],
        "omega_min": OMEGA_MIN,
        "m07_bayes_parity_note": (
            "Pilot 8 H1 does not recompute m07_bayes; H2 reads only existing "
            "e-process log-capital columns and excludes the known BOCPD z/parity path."
        ),
    }
    for name, mech in (("weighted", WeightedMech()), ("unweighted", UnweightedMech())):
        from harness import verify as harness_verify

        ok, msg = harness_verify(mech)
        checks[f"harness_verify_{name}"] = {"ok": ok, "message": msg}
        if not ok:
            raise SystemExit(f"harness.verify failed for {name}: {msg}")

    st = load_store(str(ROOT / "cache" / "store"))
    dev_series = np.unique(c.d.sidx[c.dev])
    lens = c.n_online[dev_series]
    pick = dev_series[np.argsort(lens)[np.linspace(0, len(lens) - 1, 8).astype(int)]]
    prefix_results = {}
    for module in (WEIGHTED_MODULE, UNWEIGHTED_MODULE):
        module_results = []
        for sid in pick:
            hist = st.hist(int(sid))
            online = st.online(int(sid))
            ok, msg = check_prefix_invariance(module, hist, online, cuts=(10, 37, 73, 111), atol=0.0)
            module_results.append({"series": int(sid), "ok": ok, "message": msg})
            if not ok:
                raise SystemExit(f"{module} prefix check failed on series {int(sid)}: {msg}")
        prefix_results[module] = module_results
    checks["module_prefix_invariance"] = prefix_results

    sid = int(pick[-1])
    hist = st.hist(sid)
    online = st.online(sid)
    ctx = make_ctx(hist, online)
    weighted = weighted_ctm_features(ctx)
    unweighted = unweighted_ctm_features(ctx)
    finite = {
        "weighted_all_finite": bool(np.isfinite(weighted).all()),
        "unweighted_all_finite": bool(np.isfinite(unweighted).all()),
        "weighted_peaks_dominate_current": bool(
            np.all(weighted[:, 1] >= weighted[:, 0])
            and np.all(weighted[:, 3] >= weighted[:, 2])
            and np.all(weighted[:, 7] >= weighted[:, 6])
        ),
        "unweighted_peaks_dominate_current": bool(
            np.all(unweighted[:, 1] >= unweighted[:, 0])
            and np.all(unweighted[:, 3] >= unweighted[:, 2])
            and np.all(unweighted[:, 7] >= unweighted[:, 6])
        ),
    }
    if not all(finite.values()):
        raise SystemExit(f"finite/peak check failed: {finite}")
    checks["finite_and_peak_checks"] = finite

    cu_h = np.r_[np.linspace(-0.49, -0.46, 20), np.linspace(-0.2, 0.2, 180)]
    cu_o = np.zeros(80)
    cu_o[0:40] = 0.49
    omega = benign_tail_weight(cu_h, cu_o)
    cu_o_changed = cu_o.copy()
    cu_o_changed[0] = 0.0
    omega_changed = benign_tail_weight(cu_h, cu_o_changed)
    predictable = {
        "omega_row0_excludes_current": bool(omega[0] == 1.0 and omega_changed[0] == omega[0]),
        "omega_row1_uses_prior_row": bool(omega[1] < 1.0 and omega_changed[1] > omega[1]),
        "omega_min_observed": float(np.min(omega)),
        "omega_max_observed": float(np.max(omega)),
    }
    if not predictable["omega_row0_excludes_current"] or not predictable["omega_row1_uses_prior_row"]:
        raise SystemExit(f"predictability check failed: {predictable}")
    checks["omega_predictability"] = predictable

    cut = min(111, max(2, len(online) // 2))
    mutated = online.copy()
    mutated[cut:] = mutated[cut:][::-1] * -2.0 + 4.0
    m_weighted = weighted_ctm_features(make_ctx(hist, mutated))
    m_unweighted = unweighted_ctm_features(make_ctx(hist, mutated))
    future_ok = bool(
        np.allclose(weighted[:cut], m_weighted[:cut], rtol=0.0, atol=0.0, equal_nan=True)
        and np.allclose(unweighted[:cut], m_unweighted[:cut], rtol=0.0, atol=0.0, equal_nan=True)
    )
    if not future_ok:
        raise SystemExit("future-mutation check failed")
    checks["future_mutation_prefix_ok"] = future_ok

    r_weighted = weighted_ctm_features(make_ctx(hist, online))
    r_unweighted = unweighted_ctm_features(make_ctx(hist, online))
    replay_ok = bool(
        np.array_equal(weighted, r_weighted, equal_nan=True)
        and np.array_equal(unweighted, r_unweighted, equal_nan=True)
    )
    if not replay_ok:
        raise SystemExit("deterministic replay check failed")
    checks["deterministic_replay_ok"] = replay_ok

    idx, m07_cols = _m07_column_indices()
    checks["h2_column_indices"] = [int(i) for i in idx]
    checks["h2_uses_only_preregistered_columns"] = bool([m07_cols[i] for i in idx] == H2_EVALUE_COLS)
    if not checks["h2_uses_only_preregistered_columns"]:
        raise SystemExit("H2 e-value columns differ from preregistration")
    return checks


def build_dev_feature_caches(c: Ctx) -> dict:
    load_all()
    for module_name in (WEIGHTED_MODULE, UNWEIGHTED_MODULE):
        if module_name not in REGISTRY:
            raise SystemExit(f"unknown feature module {module_name}")

    t0 = time.time()
    st = load_store(str(ROOT / "cache" / "store"))
    weighted = np.full((len(c.d.y), len(WEIGHTED_COLS)), np.nan, dtype=np.float32)
    unweighted = np.full((len(c.d.y), len(UNWEIGHTED_COLS)), np.nan, dtype=np.float32)
    dev_series = np.unique(c.d.sidx[c.dev])
    for count, sid in enumerate(dev_series):
        rows = c.dev[c.d.sidx[c.dev] == sid]
        rows = rows[np.argsort(c.d.t[rows], kind="stable")]
        ctx = make_ctx(st.hist(int(sid)), st.online(int(sid)))
        w_feat = weighted_ctm_features(ctx)
        u_feat = unweighted_ctm_features(ctx)
        if len(w_feat) != len(rows) or len(u_feat) != len(rows):
            raise SystemExit(f"Pilot 8 row mismatch on series {int(sid)}")
        weighted[rows] = w_feat
        unweighted[rows] = u_feat
        if count and count % 1000 == 0:
            print(f"pilot08 weighted CTM: built {count}/{len(dev_series)} series {time.time() - t0:.0f}s", flush=True)

    np.save(FEATDIR / f"{WEIGHTED_MODULE}.npy", weighted)
    np.save(FEATDIR / f"{UNWEIGHTED_MODULE}.npy", unweighted)
    (FEATDIR / f"{WEIGHTED_MODULE}.cols.json").write_text(
        json.dumps({"cols": WEIGHTED_COLS, "version": REGISTRY[WEIGHTED_MODULE].version, "owner": REGISTRY[WEIGHTED_MODULE].owner}, sort_keys=True) + "\n"
    )
    (FEATDIR / f"{UNWEIGHTED_MODULE}.cols.json").write_text(
        json.dumps({"cols": UNWEIGHTED_COLS, "version": REGISTRY[UNWEIGHTED_MODULE].version, "owner": REGISTRY[UNWEIGHTED_MODULE].owner}, sort_keys=True) + "\n"
    )
    return {
        WEIGHTED_MODULE: {
            "module": WEIGHTED_MODULE,
            "cols": WEIGHTED_COLS,
            "shape": [int(weighted.shape[0]), int(weighted.shape[1])],
            "dev_series": int(len(dev_series)),
            "lockbox_rows_filled": int(np.isfinite(weighted[c.d.rows_for([-1])]).sum()),
        },
        UNWEIGHTED_MODULE: {
            "module": UNWEIGHTED_MODULE,
            "cols": UNWEIGHTED_COLS,
            "shape": [int(unweighted.shape[0]), int(unweighted.shape[1])],
            "dev_series": int(len(dev_series)),
            "lockbox_rows_filled": int(np.isfinite(unweighted[c.d.rows_for([-1])]).sum()),
        },
        "runtime_s": time.time() - t0,
    }


def build_h2_score(c: Ctx) -> tuple[np.ndarray, dict]:
    t0 = time.time()
    idx, m07_cols = _m07_column_indices()
    m07 = np.load(FEATDIR / "m07_bayes.npy", mmap_mode="r")
    score = np.full(len(c.d.y), np.nan, dtype=np.float32)
    vals = np.asarray(m07[c.dev][:, idx], dtype=np.float32)
    score[c.dev] = h2_log_mean_evalue(vals)
    if np.isfinite(score[c.d.rows_for([-1])]).any():
        raise SystemExit("H2 direct score unexpectedly filled lockbox rows")
    np.save(OOFDIR / f"{H2_ID}.npy", score)
    fold_rows = c.rows[SCORED_FOLD]
    fold_auc = float(ts_auc_flat(score[fold_rows], c.d.y[fold_rows], c.d.t[fold_rows]))
    train_rows_reference = len(c.d.rows_for([1, 2, 3, 4]))
    res = {
        "experiment_id": H2_ID,
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "git_sha": git_sha(),
        "agent": "codex-new-avenues",
        "hypothesis": (
            "Pilot 8 H2 tests whether a parameter-free log-average of existing "
            "m07_bayes e-process log-capitals is useful as an eighth stream."
        ),
        "falsification_condition": "KILL if marginal_vs_clone < +0.0010 against RT600 + RT-401 seed clone.",
        "feature_set": ",".join(f"m07_bayes::{c}" for c in H2_EVALUE_COLS),
        "n_features": len(H2_EVALUE_COLS),
        "model": "log_mean_evalue",
        "objective": "none",
        "folds": str(SCORED_FOLD),
        "random_seed": 0,
        "train_series": int((~np.isin(c.d.series_fold, [-1])).sum()),
        "train_rows": int(train_rows_reference),
        "mean_oof_ts_auc": fold_auc,
        "pooled_oof_ts_auc": fold_auc,
        "per_fold_ts_auc": f"{fold_auc:.5f}",
        "fold_std": 0.0,
        "persistence": "none",
        "sample_mode": "none",
        "training_runtime_s": round(time.time() - t0, 1),
        "causal_verified": "existing m07 e-process columns only + no lockbox rows filled",
        "test_reduced_touched": "no",
        "lockbox_touched": "no",
        "status": "recorded",
        "notes": "Pilot 8 H2 parameter-free e-value aggregation direct-score arm. Fold-0 screen only.",
        "protocol": "pilot_fold0_only",
    }
    append_result_append_only(res)
    return score.astype(np.float64), {
        "result_row": res,
        "oof_artifact": str(OOFDIR / f"{H2_ID}.npy"),
        "m07_cols": [m07_cols[i] for i in idx],
        "runtime_s": time.time() - t0,
    }


def h1_verdict(weighted_margin: float, weighted_mvn: float, unweighted_mvn: float) -> tuple[str, str, str, list[dict]]:
    failures = []
    if weighted_margin < 0.0010:
        failures.append({"gate": "primary_marginal_vs_clone", "threshold": 0.0010, "observed": weighted_margin, "message": "weighted CTM marginal_vs_clone is below +0.0010"})
    if weighted_mvn <= unweighted_mvn:
        failures.append({"gate": "never_break_split_vs_unweighted", "threshold": "weighted > unweighted", "observed_weighted_minus_unweighted": weighted_mvn - unweighted_mvn, "message": "weighted CTM does not beat unweighted CTM on mature-vs-never-break AUC"})
    if failures:
        return "KILL", "KILL", failures[0]["message"], failures
    if weighted_margin < 0.0020:
        return "WEAK", "NO_5FOLD", "weighted CTM clears kill floor but remains in the weak +0.001 to +0.002 band", failures
    if weighted_margin < 0.0030:
        return "INTERESTING", "CONTINUE", "weighted CTM clears the preregistered 5-fold continuation gate", failures
    if weighted_margin < 0.0050:
        return "SERIOUS", "CONTINUE", "weighted CTM clears the serious screen band; confirm before any next pilot", failures
    if weighted_margin < 0.0080:
        return "MAJOR", "CONTINUE", "weighted CTM clears the major screen band; confirm before any next pilot", failures
    return "BREAKTHROUGH", "CONTINUE", "weighted CTM clears the breakthrough screen band; confirm before any next pilot", failures


def h2_verdict(margin: float) -> tuple[str, str, str, list[dict]]:
    failures = []
    if margin < 0.0010:
        failures.append({"gate": "primary_marginal_vs_clone", "threshold": 0.0010, "observed": margin, "message": "H2 marginal_vs_clone is below +0.0010"})
        return "KILL", "KILL", failures[0]["message"], failures
    if margin < 0.0020:
        return "WEAK", "NO_5FOLD", "H2 clears kill floor but remains in the weak +0.001 to +0.002 band", failures
    if margin < 0.0030:
        return "INTERESTING", "CONTINUE", "H2 clears the preregistered 5-fold continuation gate", failures
    if margin < 0.0050:
        return "SERIOUS", "CONTINUE", "H2 clears the serious screen band; confirm before any next pilot", failures
    if margin < 0.0080:
        return "MAJOR", "CONTINUE", "H2 clears the major screen band; confirm before any next pilot", failures
    return "BREAKTHROUGH", "CONTINUE", "H2 clears the breakthrough screen band; confirm before any next pilot", failures


def write_report(result: dict) -> None:
    json_path = OUTDIR / "pilot08_weighted_ctm.json"
    md_path = OUTDIR / "pilot08_weighted_ctm.md"
    json_path.write_text(json.dumps(finite_float(result), indent=2, sort_keys=True) + "\n")

    weighted = result[WEIGHTED_ID]
    unweighted = result[UNWEIGHTED_ID]
    h2 = result[H2_ID]
    wm = weighted["ensemble_marginal"]
    um = unweighted["ensemble_marginal"]
    hm = h2["ensemble_marginal"]
    wp = weighted["diagnostic_pack"]
    up = unweighted["diagnostic_pack"]
    hp = h2["diagnostic_pack"]
    checks = result["causal_checks"]
    sentinel = result["rt600_sentinel"]

    md = [
        "# PILOT 8 -- WEIGHTED CONFORMAL TEST MARTINGALE",
        "",
        f"Pre-registration: `research/reports/new_avenues_2026/PILOT08_PREREG.md` at `{PREREG_SHA}`.",
        f"Experiment IDs: `{WEIGHTED_ID}` weighted CTM, `{UNWEIGHTED_ID}` unweighted CTM control, `{H2_ID}` H2 e-value aggregation.",
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
        "The weighted arm scales the CTM betting fraction by a predictable recent "
        "tail-rate weight. The unweighted arm fixes the same weight to 1. H2 is "
        "a direct log-average of five existing `m07_bayes` e-process log-capitals.",
        "",
        "## Causality Checks",
        "",
        f"* Harness weighted: `{checks['harness_verify_weighted']['message']}`.",
        f"* Harness unweighted: `{checks['harness_verify_unweighted']['message']}`.",
        f"* Future mutation prefix check: `{checks['future_mutation_prefix_ok']}`.",
        f"* Deterministic replay: `{checks['deterministic_replay_ok']}`.",
        f"* Omega predictability: `{checks['omega_predictability']}`.",
        f"* H2 columns: `{checks['h2_evalue_cols']}`.",
        f"* m07 parity note: {checks['m07_bayes_parity_note']}",
        "",
        "## Binding Marginal Result",
        "",
        "| arm | fold-0 standalone TS-AUC | E2 TS-AUC | marginal vs clone | gain vs RT600 | mature-vs-never | verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| RT600 |  | {wm['rt600_7stream']:.6f} |  |  |  |  |",
        f"| RT600 + RT-401 seed clone |  | {wm['rt600_plus_seedclone']:.6f} |  |  |  |  |",
        (
            f"| RT600 + {WEIGHTED_ID} | {wp['whole_fold']['candidate']:.6f} | "
            f"{wm[f'rt600_plus_{WEIGHTED_ID}']:.6f} | {wm['marginal_vs_clone']:+.6f} | "
            f"{wm['gain_vs_base']:+.6f} | {wp['mature_vs_neverbreak']['candidate']:.6f} | "
            f"{result['verdicts'][WEIGHTED_ID]['verdict']} ({result['verdicts'][WEIGHTED_ID]['continuation_status']}) |"
        ),
        (
            f"| RT600 + {UNWEIGHTED_ID} | {up['whole_fold']['candidate']:.6f} | "
            f"{um[f'rt600_plus_{UNWEIGHTED_ID}']:.6f} | {um['marginal_vs_clone']:+.6f} | "
            f"{um['gain_vs_base']:+.6f} | {up['mature_vs_neverbreak']['candidate']:.6f} | control |"
        ),
        (
            f"| RT600 + {H2_ID} | {hp['whole_fold']['candidate']:.6f} | "
            f"{hm[f'rt600_plus_{H2_ID}']:.6f} | {hm['marginal_vs_clone']:+.6f} | "
            f"{hm['gain_vs_base']:+.6f} | {hp['mature_vs_neverbreak']['candidate']:.6f} | "
            f"{result['verdicts'][H2_ID]['verdict']} ({result['verdicts'][H2_ID]['continuation_status']}) |"
        ),
        "",
        f"Weighted minus unweighted mature-vs-never AUC: `{result['weighted_minus_unweighted_mature_vs_never']:+.6f}`.",
        f"Overall continuation status: **{result['overall_continuation_status']}**.",
        f"Weighted failed gates: {result['verdicts'][WEIGHTED_ID]['gate_failures']}.",
        f"H2 failed gates: {result['verdicts'][H2_ID]['gate_failures']}.",
        "",
        "## Diagnostic Pack",
        "",
        "| arm | whole-fold AUC | dominant-cell AUC | mature vs never | mature vs pre | rho vs RT600 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for eid, pack in ((WEIGHTED_ID, wp), (UNWEIGHTED_ID, up), (H2_ID, hp)):
        md.append(
            f"| `{eid}` | {pack['whole_fold']['candidate']:.6f} | "
            f"{pack['dominant_cell']['candidate']:.6f} | "
            f"{pack['mature_vs_neverbreak']['candidate']:.6f} | "
            f"{pack['mature_vs_prebreak']['candidate']:.6f} | "
            f"{pack['within_t_rank_corr_rt600']:+.4f} |"
        )
    md += [
        "",
        "## Pair Flow",
        "",
    ]
    for eid, row in ((WEIGHTED_ID, weighted), (UNWEIGHTED_ID, unweighted), (H2_ID, h2)):
        md += [
            f"### {eid}",
            "",
            "| split | repairs | damage | net | sampled pairs |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, pf in row["pair_flow"].items():
            md.append(f"| `{name}` | {pf['repairs']} | {pf['damage']} | {pf['net_pair_lift']} | {pf['total_pairs_sampled']} |")
        md.append("")
    md += [
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
    feature_cache = build_dev_feature_caches(c)

    common_hypothesis = (
        "Pilot 8 tests whether conformal test martingale weighting by a predictable "
        "benign-tail estimate improves never-break discrimination."
    )
    weighted_oof, weighted_train = train_foldpure_oof(
        WEIGHTED_ID,
        WEIGHTED_MODULE,
        common_hypothesis + " The weighted CTM arm should add ensemble alpha beyond the seed clone and incumbent m07 e-processes.",
        "KILL if marginal_vs_clone < +0.0010, or if weighted CTM does not beat unweighted CTM on mature-vs-never-break AUC.",
        "Pilot 8 weighted conformal test martingale arm. Fold-0 screen only; folds 1-4 generated only as fold-pure SCDF calibration support.",
    )
    unweighted_oof, unweighted_train = train_foldpure_oof(
        UNWEIGHTED_ID,
        UNWEIGHTED_MODULE,
        common_hypothesis + " The unweighted CTM block is the binding control isolating the weighting rule.",
        "Control must not match or exceed weighted CTM on the never-break split; if it does, H1 weighting is killed.",
        "Pilot 8 matched unweighted CTM control. Fold-0 screen only; folds 1-4 generated only as fold-pure SCDF calibration support.",
    )
    h2_oof, h2_build = build_h2_score(c)

    weighted = candidate_result(WEIGHTED_ID, weighted_oof, base, c)
    unweighted = candidate_result(UNWEIGHTED_ID, unweighted_oof, base, c)
    h2 = candidate_result(H2_ID, h2_oof, base, c)
    weighted["training"] = weighted_train
    unweighted["training"] = unweighted_train
    h2["training"] = h2_build

    w_mvn = weighted["diagnostic_pack"]["mature_vs_neverbreak"]["candidate"]
    u_mvn = unweighted["diagnostic_pack"]["mature_vs_neverbreak"]["candidate"]
    wv, wc, wr, wf = h1_verdict(weighted["ensemble_marginal"]["marginal_vs_clone"], w_mvn, u_mvn)
    hv, hc, hr, hf = h2_verdict(h2["ensemble_marginal"]["marginal_vs_clone"])
    verdicts = {
        WEIGHTED_ID: {"verdict": wv, "continuation_status": wc, "reason": wr, "gate_failures": wf},
        UNWEIGHTED_ID: {"verdict": "CONTROL", "continuation_status": "CONTROL", "reason": "matched unweighted control", "gate_failures": []},
        H2_ID: {"verdict": hv, "continuation_status": hc, "reason": hr, "gate_failures": hf},
    }
    continuation_arms = [eid for eid in (WEIGHTED_ID, H2_ID) if verdicts[eid]["continuation_status"] == "CONTINUE"]
    overall = "STOP_FOR_5FOLD_CONFIRMATION" if continuation_arms else "FIRST_SWEEP_EXHAUSTED"
    if continuation_arms:
        interpretation = (
            "At least one Pilot 8 arm cleared the preregistered continuation gate. "
            "The broad first sweep stops here for confirmation before any new mechanism."
        )
    else:
        interpretation = (
            "No Pilot 8 arm cleared the preregistered 5-fold continuation gate. "
            "This kills or shelves the exact weighted CTM and parameter-free e-value "
            "aggregation constructions under the fold-0 screen and exhausts the planned "
            "first-sweep queue."
        )

    result = {
        "generated": "2026-08-24",
        "branch": "research/new-avenues-pilots-2026",
        "prereg_sha": PREREG_SHA,
        "head_sha": git_sha(),
        "exp_ids": [WEIGHTED_ID, UNWEIGHTED_ID, H2_ID],
        "feature_modules": [WEIGHTED_MODULE, UNWEIGHTED_MODULE],
        "feature_definitions": {
            "weighted_cols": WEIGHTED_COLS,
            "unweighted_cols": UNWEIGHTED_COLS,
            "h2_evalue_cols": H2_EVALUE_COLS,
            "tail_weight_threshold": TAIL_WEIGHT_THRESHOLD,
            "omega_min": OMEGA_MIN,
            "hist_tail_rate_clip": [HIST_TAIL_RATE_MIN, HIST_TAIL_RATE_MAX],
        },
        "causal_checks": checks,
        "rt600_sentinel": sentinel,
        "feature_cache": feature_cache,
        WEIGHTED_ID: weighted,
        UNWEIGHTED_ID: unweighted,
        H2_ID: h2,
        "weighted_minus_unweighted_mature_vs_never": w_mvn - u_mvn,
        "verdicts": verdicts,
        "continuation_arms": continuation_arms,
        "overall_continuation_status": overall,
        "interpretation": interpretation,
        "feature_build_runtime_s": feature_cache["runtime_s"],
        "runtime_s": time.time() - t0,
    }
    write_report(result)
    print(
        json.dumps(
            finite_float(
                {
                    "overall_continuation_status": overall,
                    WEIGHTED_ID: {
                        "ensemble_marginal": weighted["ensemble_marginal"],
                        "mature_vs_never": w_mvn,
                        "verdict": verdicts[WEIGHTED_ID],
                    },
                    UNWEIGHTED_ID: {
                        "ensemble_marginal": unweighted["ensemble_marginal"],
                        "mature_vs_never": u_mvn,
                    },
                    H2_ID: {
                        "ensemble_marginal": h2["ensemble_marginal"],
                        "verdict": verdicts[H2_ID],
                    },
                    "weighted_minus_unweighted_mature_vs_never": w_mvn - u_mvn,
                    "runtime_s": result["runtime_s"],
                }
            ),
            indent=2,
        )
    )
    print(f"wrote {OUTDIR / 'pilot08_weighted_ctm.json'}")
    print(f"wrote {OUTDIR / 'pilot08_weighted_ctm.md'}")


if __name__ == "__main__":
    main()
