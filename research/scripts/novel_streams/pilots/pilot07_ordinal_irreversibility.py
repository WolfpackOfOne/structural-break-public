"""Pilot 7: ordinal transition divergence and time irreversibility.

Scored candidates:
  RT-1208: ordinal transition/asymmetry features.
  RT-1209: matched permutation-entropy-only control.
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

import numpy as np

from harness import StreamingMechanism, rt600_blend
from pilot06_spectral_impulse import (
    candidate_result,
    finite_float,
    git_sha,
    rt600_sentinel,
    train_foldpure_oof,
)
from sbr.features.base import REGISTRY, check_prefix_invariance, load_all, make_ctx
from sbr.features.m15_ordinal_irrev import (
    CANDIDATE_COLS,
    CONTROL_COLS,
    GRID,
    MIN_COUNT,
    ordinal_irreversibility_features,
)
from sbr.store import load_store
from wave5_lib import Ctx, SPECIALISTS, load_oof

PREREG_SHA = "1e7e42c"
CANDIDATE_ID = "RT-1208"
CONTROL_ID = "RT-1209"
CANDIDATE_MODULE = "m15_ordinal_irrev"
CONTROL_MODULE = "m15_ordinal_entropy"
OUTDIR = ROOT / "research" / "reports" / "new_avenues_2026"
FEATDIR = ROOT / "cache" / "features"
OUTDIR.mkdir(parents=True, exist_ok=True)
FEATDIR.mkdir(parents=True, exist_ok=True)
SCORED_FOLD = 0


class CandidateMech(StreamingMechanism):
    name = "pilot07_ordinal_irrev"
    cols = CANDIDATE_COLS

    def emit(self, hist, online):
        candidate, _ = ordinal_irreversibility_features(make_ctx(hist, online))
        return candidate


class EntropyMech(StreamingMechanism):
    name = "pilot07_ordinal_entropy"
    cols = CONTROL_COLS

    def emit(self, hist, online):
        _, control = ordinal_irreversibility_features(make_ctx(hist, online))
        return control


def run_causal_checks(c: Ctx) -> dict:
    load_all()
    checks: dict[str, object] = {
        "min_count": MIN_COUNT,
        "null_grid": [int(x) for x in GRID],
        "candidate_cols": CANDIDATE_COLS,
        "control_cols": CONTROL_COLS,
        "m07_bayes_parity_note": (
            "Pilot 7 does not recompute m07_bayes::bo_p_lt25_z; it uses the "
            "existing cached base-bank features and independent new ordinal streams."
        ),
    }
    for name, mech in (("candidate", CandidateMech()), ("entropy", EntropyMech())):
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
    for module in (CANDIDATE_MODULE, CONTROL_MODULE):
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
    candidate, control = ordinal_irreversibility_features(make_ctx(hist, online))
    first_valid = {
        "candidate_rows_before_15_all_nan": bool(np.isnan(candidate[:15]).all()),
        "control_rows_before_15_all_nan": bool(np.isnan(control[:15]).all()),
        "candidate_row15_exp_finite": bool(np.isfinite(candidate[15, [0, 1, 4, 5]]).all()) if len(candidate) > 15 else None,
        "control_row15_exp_finite": bool(np.isfinite(control[15, [0, 1]]).all()) if len(control) > 15 else None,
        "candidate_row30_half_all_nan": bool(np.isnan(candidate[30, [2, 3, 6, 7]]).all()) if len(candidate) > 30 else None,
        "control_row30_half_all_nan": bool(np.isnan(control[30, [2, 3]]).all()) if len(control) > 30 else None,
        "candidate_row31_all_finite": bool(np.isfinite(candidate[31]).all()) if len(candidate) > 31 else None,
        "control_row31_all_finite": bool(np.isfinite(control[31]).all()) if len(control) > 31 else None,
    }
    if not all(v for v in first_valid.values() if v is not None):
        raise SystemExit(f"first-valid check failed: {first_valid}")
    checks["first_valid_nan_semantics"] = first_valid

    cut = min(91, max(2, len(online) // 2))
    mutated = online.copy()
    mutated[cut:] = mutated[cut:][::-1] * -3.0 + 2.0
    m_candidate, m_control = ordinal_irreversibility_features(make_ctx(hist, mutated))
    future_ok = bool(
        np.allclose(candidate[:cut], m_candidate[:cut], rtol=0.0, atol=0.0, equal_nan=True)
        and np.allclose(control[:cut], m_control[:cut], rtol=0.0, atol=0.0, equal_nan=True)
    )
    if not future_ok:
        raise SystemExit("future-mutation check failed")
    checks["future_mutation_prefix_ok"] = future_ok

    r_candidate, r_control = ordinal_irreversibility_features(make_ctx(hist, online))
    replay_ok = bool(
        np.array_equal(candidate, r_candidate, equal_nan=True)
        and np.array_equal(control, r_control, equal_nan=True)
    )
    if not replay_ok:
        raise SystemExit("deterministic replay check failed")
    checks["deterministic_replay_ok"] = replay_ok
    return checks


def build_dev_feature_caches(c: Ctx) -> dict:
    load_all()
    for module_name in (CANDIDATE_MODULE, CONTROL_MODULE):
        if module_name not in REGISTRY:
            raise SystemExit(f"unknown feature module {module_name}")

    t0 = time.time()
    st = load_store(str(ROOT / "cache" / "store"))
    candidate = np.full((len(c.d.y), len(CANDIDATE_COLS)), np.nan, dtype=np.float32)
    control = np.full((len(c.d.y), len(CONTROL_COLS)), np.nan, dtype=np.float32)
    dev_series = np.unique(c.d.sidx[c.dev])
    for count, sid in enumerate(dev_series):
        rows = c.dev[c.d.sidx[c.dev] == sid]
        rows = rows[np.argsort(c.d.t[rows], kind="stable")]
        cand, ctrl = ordinal_irreversibility_features(make_ctx(st.hist(int(sid)), st.online(int(sid))))
        if len(cand) != len(rows) or len(ctrl) != len(rows):
            raise SystemExit(f"Pilot 7 row mismatch on series {int(sid)}")
        candidate[rows] = cand
        control[rows] = ctrl
        if count and count % 1000 == 0:
            print(f"pilot07 features: built {count}/{len(dev_series)} series {time.time() - t0:.0f}s", flush=True)

    np.save(FEATDIR / f"{CANDIDATE_MODULE}.npy", candidate)
    np.save(FEATDIR / f"{CONTROL_MODULE}.npy", control)
    (FEATDIR / f"{CANDIDATE_MODULE}.cols.json").write_text(
        json.dumps({"cols": CANDIDATE_COLS, "version": REGISTRY[CANDIDATE_MODULE].version, "owner": REGISTRY[CANDIDATE_MODULE].owner}, sort_keys=True) + "\n"
    )
    (FEATDIR / f"{CONTROL_MODULE}.cols.json").write_text(
        json.dumps({"cols": CONTROL_COLS, "version": REGISTRY[CONTROL_MODULE].version, "owner": REGISTRY[CONTROL_MODULE].owner}, sort_keys=True) + "\n"
    )
    return {
        CANDIDATE_MODULE: {
            "module": CANDIDATE_MODULE,
            "cols": CANDIDATE_COLS,
            "shape": [int(candidate.shape[0]), int(candidate.shape[1])],
            "dev_series": int(len(dev_series)),
            "lockbox_rows_filled": int(np.isfinite(candidate[c.d.rows_for([-1])]).sum()),
        },
        CONTROL_MODULE: {
            "module": CONTROL_MODULE,
            "cols": CONTROL_COLS,
            "shape": [int(control.shape[0]), int(control.shape[1])],
            "dev_series": int(len(dev_series)),
            "lockbox_rows_filled": int(np.isfinite(control[c.d.rows_for([-1])]).sum()),
        },
        "runtime_s": time.time() - t0,
    }


def gate_verdict(candidate_margin: float, control_margin: float) -> tuple[str, str, str, list[dict]]:
    gap = candidate_margin - control_margin
    failures = []
    if candidate_margin < 0.0010:
        failures.append({"gate": "primary_marginal_vs_clone", "threshold": 0.0010, "observed": candidate_margin, "message": "candidate marginal_vs_clone is below +0.0010"})
    if control_margin >= candidate_margin:
        failures.append({"gate": "entropy_control_not_worse", "threshold": "control < candidate", "observed_candidate_minus_control": gap, "message": "entropy-only control matches or exceeds transition/asymmetry candidate"})
    elif gap < 0.0005:
        failures.append({"gate": "candidate_control_distinguishability", "threshold": 0.0005, "observed_candidate_minus_control": gap, "message": "candidate-control gap is below the preregistered +0.0005 distinguishability floor"})
    if candidate_margin < 0.0010:
        return "KILL", "KILL", failures[0]["message"], failures
    if control_margin >= candidate_margin:
        return "KILL", "KILL", "entropy-only control matches or exceeds transition/asymmetry candidate", failures
    if gap < 0.0005:
        return "KILL", "KILL", failures[0]["message"], failures
    if candidate_margin < 0.0020:
        return "WEAK", "NO_5FOLD", "candidate clears kill floor but remains in the weak +0.001 to +0.002 band", failures
    if candidate_margin < 0.0030:
        return "INTERESTING", "CONTINUE", "candidate clears the preregistered 5-fold continuation gate", failures
    if candidate_margin < 0.0050:
        return "SERIOUS", "CONTINUE", "candidate clears the serious screen band; confirm before any next pilot", failures
    if candidate_margin < 0.0080:
        return "MAJOR", "CONTINUE", "candidate clears the major screen band; confirm before any next pilot", failures
    return "BREAKTHROUGH", "CONTINUE", "candidate clears the breakthrough screen band; confirm before any next pilot", failures


def write_report(result: dict) -> None:
    json_path = OUTDIR / "pilot07_ordinal_irreversibility.json"
    md_path = OUTDIR / "pilot07_ordinal_irreversibility.md"
    json_path.write_text(json.dumps(finite_float(result), indent=2, sort_keys=True) + "\n")

    cand = result[CANDIDATE_ID]
    ctrl = result[CONTROL_ID]
    cm = cand["ensemble_marginal"]
    em = ctrl["ensemble_marginal"]
    cp = cand["diagnostic_pack"]
    ep = ctrl["diagnostic_pack"]
    checks = result["causal_checks"]
    sentinel = result["rt600_sentinel"]

    md = [
        "# PILOT 7 -- ORDINAL TRANSITION DIVERGENCE AND TIME IRREVERSIBILITY",
        "",
        f"Pre-registration: `research/reports/new_avenues_2026/PILOT07_PREREG.md` at `{PREREG_SHA}`.",
        f"Experiment IDs: `{CANDIDATE_ID}` transition/asymmetry candidate, `{CONTROL_ID}` entropy-only control.",
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
        "Order-3 ordinal codes use the `m03_dyn` tie convention. The candidate emits "
        "expanding and trailing-half-prefix transition KL z/raw features plus signed "
        "Ramsey-Rothman increment-asymmetry z/raw features. The control emits only "
        "matching permutation-entropy features.",
        "",
        "## Causality Checks",
        "",
        f"* Harness candidate: `{checks['harness_verify_candidate']['message']}`.",
        f"* Harness entropy control: `{checks['harness_verify_entropy']['message']}`.",
        f"* Future mutation prefix check: `{checks['future_mutation_prefix_ok']}`.",
        f"* Deterministic replay: `{checks['deterministic_replay_ok']}`.",
        f"* First-valid semantics: `{checks['first_valid_nan_semantics']}`.",
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
        f"Candidate minus entropy-control marginal: `{result['candidate_minus_control_marginal']:+.6f}`.",
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
        "Pilot 7 tests whether order-3 ordinal transition structure and time irreversibility "
        "carry dependence-break information beyond permutation entropy."
    )
    candidate_oof, candidate_train = train_foldpure_oof(
        CANDIDATE_ID,
        CANDIDATE_MODULE,
        common_hypothesis + " Transition KL and Ramsey-Rothman asymmetry should add ensemble alpha.",
        "KILL if candidate marginal_vs_clone < +0.0010, or if entropy-only control matches within +0.0005.",
        "Pilot 7 ordinal transition/asymmetry arm. Fold-0 screen only; folds 1-4 generated only as fold-pure SCDF calibration support.",
    )
    control_oof, control_train = train_foldpure_oof(
        CONTROL_ID,
        CONTROL_MODULE,
        common_hypothesis + " Permutation entropy alone is the binding control for existing ordinal information.",
        "Control must not match or exceed the transition/asymmetry arm; if it does, new ordinal structure is killed.",
        "Pilot 7 permutation-entropy-only control. Fold-0 screen only; folds 1-4 generated only as fold-pure SCDF calibration support.",
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
            "The preregistered order-3 ordinal transition divergence and time-irreversibility "
            "block failed a Pilot 7 binding gate. This falsifies this L3/L4 construction "
            "under the fixed tie convention, transition KL, Ramsey-Rothman asymmetry, "
            "matched-count historical-null calibration, and Mode-A fold-0 ABL screen; "
            "it does not falsify all ordinal or nonlinear dynamics representations."
        )
    elif continuation_status == "CONTINUE":
        interpretation = (
            "The transition/asymmetry arm cleared the preregistered fold-0 continuation gate "
            "and beat the entropy-only control by the required gap. The next step is 5-fold confirmation."
        )
    else:
        interpretation = (
            "The transition/asymmetry arm cleared the kill floor but remained in the weak band. "
            "It is not promoted without an explicit continuation decision."
        )

    result = {
        "generated": "2026-08-24",
        "branch": "research/new-avenues-pilots-2026",
        "prereg_sha": PREREG_SHA,
        "head_sha": git_sha(),
        "exp_ids": [CANDIDATE_ID, CONTROL_ID],
        "feature_modules": [CANDIDATE_MODULE, CONTROL_MODULE],
        "feature_definitions": {
            "min_count": MIN_COUNT,
            "null_grid": [int(x) for x in GRID],
            "candidate_cols": CANDIDATE_COLS,
            "control_cols": CONTROL_COLS,
        },
        "causal_checks": checks,
        "rt600_sentinel": sentinel,
        "feature_cache": feature_cache,
        CANDIDATE_ID: candidate,
        CONTROL_ID: control,
        "candidate_minus_control_marginal": candidate_margin - control_margin,
        "gate_failures": gate_failures,
        "verdict": verdict,
        "continuation_status": continuation_status,
        "verdict_reason": reason,
        "interpretation": interpretation,
        "feature_build_runtime_s": feature_cache["runtime_s"],
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
    print(f"wrote {OUTDIR / 'pilot07_ordinal_irreversibility.json'}")
    print(f"wrote {OUTDIR / 'pilot07_ordinal_irreversibility.md'}")


if __name__ == "__main__":
    main()
