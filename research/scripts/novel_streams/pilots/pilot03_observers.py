"""Pilot 3 / IM3: frozen individualized observer residuals.

Scored arms:
  RT-1214: frozen AR(2)-state Kalman/NIS observer features.
  RT-1215: frozen Hankel-DMD observer features.
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
from sbr.features.m17_observers import (
    DMD_COLS,
    DMD_DELAY,
    DMD_RANK,
    DMD_WINDOWS,
    KALMAN_COLS,
    KALMAN_Q_GRID,
    KALMAN_R_GRID,
    KALMAN_WINDOWS,
    fit_hankel_dmd_observer,
    fit_kalman_observer,
    hankel_dmd_features,
    kalman_nis_features,
    observer_features,
)
from sbr.store import load_store
from wave5_lib import Ctx, SPECIALISTS, load_oof

PREREG_SHA = "0a9d97e"
KALMAN_ID = "RT-1214"
DMD_ID = "RT-1215"
KALMAN_MODULE = "m17_kalman_nis"
DMD_MODULE = "m17_hankel_dmd"
OUTDIR = ROOT / "research" / "reports" / "new_avenues_2026"
FEATDIR = ROOT / "cache" / "features"
OUTDIR.mkdir(parents=True, exist_ok=True)
FEATDIR.mkdir(parents=True, exist_ok=True)
SCORED_FOLD = 0


class KalmanMech(StreamingMechanism):
    name = "pilot03_kalman_nis"
    cols = KALMAN_COLS

    def emit(self, hist, online):
        return kalman_nis_features(make_ctx(hist, online))


class DmdMech(StreamingMechanism):
    name = "pilot03_hankel_dmd"
    cols = DMD_COLS

    def emit(self, hist, online):
        return hankel_dmd_features(make_ctx(hist, online))


def run_causal_checks(c: Ctx) -> dict:
    load_all()
    checks: dict[str, object] = {
        "kalman_q_grid": [float(x) for x in KALMAN_Q_GRID],
        "kalman_r_grid": [float(x) for x in KALMAN_R_GRID],
        "kalman_windows": [int(x) for x in KALMAN_WINDOWS],
        "dmd_delay": int(DMD_DELAY),
        "dmd_rank": int(DMD_RANK),
        "dmd_windows": [int(x) for x in DMD_WINDOWS],
        "kalman_cols": KALMAN_COLS,
        "dmd_cols": DMD_COLS,
        "m04_boundary_note": (
            "Mode A already includes m04_resid in the base bank; these arms test "
            "incremental observer residual value beyond that existing residual block."
        ),
        "m07_bayes_parity_note": (
            "Pilot 3 observers do not recompute m07_bayes::bo_p_lt25_z; they use "
            "the existing cached base-bank features and independent observer blocks."
        ),
    }
    for name, mech in (("kalman", KalmanMech()), ("dmd", DmdMech())):
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
    for module in (KALMAN_MODULE, DMD_MODULE):
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
    kalman, dmd = observer_features(make_ctx(hist, online))
    first_valid = {
        "kalman_nis_log_and_cum_finite": bool(np.isfinite(kalman[:, [0, 1]]).all()),
        "kalman_rows_before_31_w32_nan": bool(np.isnan(kalman[:31, [2, 4]]).all()) if len(kalman) > 31 else None,
        "kalman_row31_w32_finite": bool(np.isfinite(kalman[31, [2, 4]]).all()) if len(kalman) > 31 else None,
        "kalman_rows_before_63_w64_nan": bool(np.isnan(kalman[:63, [3, 5]]).all()) if len(kalman) > 63 else None,
        "kalman_row63_w64_finite": bool(np.isfinite(kalman[63, [3, 5]]).all()) if len(kalman) > 63 else None,
        "dmd_row0_all_finite_with_history_warmup": bool(np.isfinite(dmd[0]).all()) if len(dmd) else None,
    }
    if not all(v for v in first_valid.values() if v is not None):
        raise SystemExit(f"first-valid check failed: {first_valid}")
    checks["first_valid_nan_semantics"] = first_valid

    k_state_1 = fit_kalman_observer(hist)
    k_state_2 = fit_kalman_observer(hist.copy())
    d_state_1 = fit_hankel_dmd_observer(hist)
    d_state_2 = fit_hankel_dmd_observer(hist.copy())
    history_only = {
        "kalman_phi_replay_equal": bool(np.allclose(k_state_1.phi, k_state_2.phi, rtol=0.0, atol=0.0)),
        "kalman_grid_choice_replay_equal": bool(k_state_1.q == k_state_2.q and k_state_1.r == k_state_2.r),
        "dmd_operator_replay_equal": bool(np.allclose(d_state_1.operator, d_state_2.operator, rtol=0.0, atol=0.0)),
        "dmd_subspace_replay_equal": bool(np.allclose(d_state_1.subspace, d_state_2.subspace, rtol=0.0, atol=0.0)),
        "dmd_nulls_replay_equal": bool(d_state_1.nulls == d_state_2.nulls),
        "dmd_fitted_rank": int(d_state_1.fitted_rank),
    }
    if not all(v for k, v in history_only.items() if k != "dmd_fitted_rank"):
        raise SystemExit(f"history-only replay check failed: {history_only}")
    checks["history_only_fits"] = history_only

    cut = min(111, max(2, len(online) // 2))
    mutated = online.copy()
    mutated[cut:] = mutated[cut:][::-1] * -3.0 + 2.0
    m_kalman, m_dmd = observer_features(make_ctx(hist, mutated))
    future_ok = bool(
        np.allclose(kalman[:cut], m_kalman[:cut], rtol=0.0, atol=0.0, equal_nan=True)
        and np.allclose(dmd[:cut], m_dmd[:cut], rtol=0.0, atol=0.0, equal_nan=True)
    )
    if not future_ok:
        raise SystemExit("future-mutation check failed")
    checks["future_mutation_prefix_ok"] = future_ok

    r_kalman, r_dmd = observer_features(make_ctx(hist, online))
    replay_ok = bool(np.array_equal(kalman, r_kalman, equal_nan=True) and np.array_equal(dmd, r_dmd, equal_nan=True))
    if not replay_ok:
        raise SystemExit("deterministic replay check failed")
    checks["deterministic_replay_ok"] = replay_ok
    return checks


def build_dev_feature_caches(c: Ctx) -> dict:
    load_all()
    for module_name in (KALMAN_MODULE, DMD_MODULE):
        if module_name not in REGISTRY:
            raise SystemExit(f"unknown feature module {module_name}")

    t0 = time.time()
    st = load_store(str(ROOT / "cache" / "store"))
    kalman = np.full((len(c.d.y), len(KALMAN_COLS)), np.nan, dtype=np.float32)
    dmd = np.full((len(c.d.y), len(DMD_COLS)), np.nan, dtype=np.float32)
    dev_series = np.unique(c.d.sidx[c.dev])
    for count, sid in enumerate(dev_series):
        rows = c.dev[c.d.sidx[c.dev] == sid]
        rows = rows[np.argsort(c.d.t[rows], kind="stable")]
        k_feat, d_feat = observer_features(make_ctx(st.hist(int(sid)), st.online(int(sid))))
        if len(k_feat) != len(rows) or len(d_feat) != len(rows):
            raise SystemExit(f"Pilot 3 observer row mismatch on series {int(sid)}")
        kalman[rows] = k_feat
        dmd[rows] = d_feat
        if count and count % 1000 == 0:
            print(f"pilot03 observers: built {count}/{len(dev_series)} series {time.time() - t0:.0f}s", flush=True)

    np.save(FEATDIR / f"{KALMAN_MODULE}.npy", kalman)
    np.save(FEATDIR / f"{DMD_MODULE}.npy", dmd)
    (FEATDIR / f"{KALMAN_MODULE}.cols.json").write_text(
        json.dumps({"cols": KALMAN_COLS, "version": REGISTRY[KALMAN_MODULE].version, "owner": REGISTRY[KALMAN_MODULE].owner}, sort_keys=True) + "\n"
    )
    (FEATDIR / f"{DMD_MODULE}.cols.json").write_text(
        json.dumps({"cols": DMD_COLS, "version": REGISTRY[DMD_MODULE].version, "owner": REGISTRY[DMD_MODULE].owner}, sort_keys=True) + "\n"
    )
    return {
        KALMAN_MODULE: {
            "module": KALMAN_MODULE,
            "cols": KALMAN_COLS,
            "shape": [int(kalman.shape[0]), int(kalman.shape[1])],
            "dev_series": int(len(dev_series)),
            "lockbox_rows_filled": int(np.isfinite(kalman[c.d.rows_for([-1])]).sum()),
        },
        DMD_MODULE: {
            "module": DMD_MODULE,
            "cols": DMD_COLS,
            "shape": [int(dmd.shape[0]), int(dmd.shape[1])],
            "dev_series": int(len(dev_series)),
            "lockbox_rows_filled": int(np.isfinite(dmd[c.d.rows_for([-1])]).sum()),
        },
        "runtime_s": time.time() - t0,
    }


def arm_verdict(arm: str, marginal_vs_clone: float, within_t_rho: float) -> tuple[str, str, str, list[dict]]:
    failures = []
    if marginal_vs_clone < 0.0010:
        failures.append(
            {
                "gate": "primary_marginal_vs_clone",
                "threshold": 0.0010,
                "observed": marginal_vs_clone,
                "message": f"{arm} marginal_vs_clone is below +0.0010",
            }
        )
    if arm == "hankel_dmd" and within_t_rho > 0.85:
        failures.append(
            {
                "gate": "hankel_redundancy_rho",
                "threshold": 0.85,
                "observed": within_t_rho,
                "message": "Hankel-DMD within-t rank correlation with RT600 exceeds 0.85",
            }
        )
    if failures:
        return "KILL", "KILL", failures[0]["message"], failures
    if marginal_vs_clone < 0.0020:
        return "WEAK", "NO_5FOLD", f"{arm} clears kill floor but remains in the weak +0.001 to +0.002 band", failures
    if marginal_vs_clone < 0.0030:
        return "INTERESTING", "CONTINUE", f"{arm} clears the preregistered 5-fold continuation gate", failures
    if marginal_vs_clone < 0.0050:
        return "SERIOUS", "CONTINUE", f"{arm} clears the serious screen band; confirm before any next pilot", failures
    if marginal_vs_clone < 0.0080:
        return "MAJOR", "CONTINUE", f"{arm} clears the major screen band; confirm before any next pilot", failures
    return "BREAKTHROUGH", "CONTINUE", f"{arm} clears the breakthrough screen band; confirm before any next pilot", failures


def write_report(result: dict) -> None:
    json_path = OUTDIR / "pilot03_observers.json"
    md_path = OUTDIR / "pilot03_observers.md"
    json_path.write_text(json.dumps(finite_float(result), indent=2, sort_keys=True) + "\n")

    kalman = result[KALMAN_ID]
    dmd = result[DMD_ID]
    km = kalman["ensemble_marginal"]
    dm = dmd["ensemble_marginal"]
    kp = kalman["diagnostic_pack"]
    dp = dmd["diagnostic_pack"]
    checks = result["causal_checks"]
    sentinel = result["rt600_sentinel"]
    kv = result["verdicts"][KALMAN_ID]
    dv = result["verdicts"][DMD_ID]

    md = [
        "# PILOT 3 / IM3 -- INDIVIDUAL OBSERVER RESIDUALS",
        "",
        f"Pre-registration: `research/reports/new_avenues_2026/PILOT03_OBSERVERS_PREREG.md` at `{PREREG_SHA}`.",
        f"Experiment IDs: `{KALMAN_ID}` Kalman/NIS arm, `{DMD_ID}` Hankel-DMD arm.",
        "",
        "## RT-600 Anchor",
        "",
        f"* Dev mean TS-AUC: `{sentinel['mean']:.6f}`.",
        f"* Dev pooled TS-AUC: `{sentinel['pooled']:.6f}`.",
        f"* Dev dominant-cell TS-AUC: `{sentinel['dominant_cell']:.6f}`.",
        f"* Fold-0 E0 RT600: `{sentinel['fold0_e0']:.6f}`.",
        f"* Fold-0 E1 RT600 + RT-401: `{sentinel['fold0_e1_seedclone']:.6f}`.",
        "",
        "## Candidate Arms",
        "",
        "`RT-1214` is a per-series frozen AR(2)-state Kalman observer with fixed "
        "history-only `q x r` noise-grid selection. It emits NIS accumulation, "
        "windowed NIS excess, and normalized-innovation whiteness.",
        "",
        "`RT-1215` is a per-series frozen Hankel-DMD observer with delay `16`, "
        "rank `4`, horizons `1` and `5`, historical-subspace residual, and "
        "effective-rank monitors.",
        "",
        "The Mode-A base bank already includes `m04_resid`; the binding marginal "
        "therefore asks whether either observer adds value beyond existing scalar "
        "residual monitors and beyond the `RT-401` seed clone.",
        "",
        "## Causality Checks",
        "",
        f"* Harness Kalman/NIS: `{checks['harness_verify_kalman']['message']}`.",
        f"* Harness Hankel-DMD: `{checks['harness_verify_dmd']['message']}`.",
        f"* Future mutation prefix check: `{checks['future_mutation_prefix_ok']}`.",
        f"* Deterministic replay: `{checks['deterministic_replay_ok']}`.",
        f"* First-valid semantics: `{checks['first_valid_nan_semantics']}`.",
        f"* History-only fit replay: `{checks['history_only_fits']}`.",
        f"* m04 boundary note: {checks['m04_boundary_note']}",
        f"* m07 parity note: {checks['m07_bayes_parity_note']}",
        "",
        "## Binding Marginal Result",
        "",
        "| arm | fold-0 standalone TS-AUC | E2 TS-AUC | marginal vs clone | gain vs RT600 | rho vs RT600 | verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| RT600 |  | {km['rt600_7stream']:.6f} |  |  |  |  |",
        f"| RT600 + RT-401 seed clone |  | {km['rt600_plus_seedclone']:.6f} |  |  |  |  |",
        (
            f"| RT600 + {KALMAN_ID} | {kp['whole_fold']['candidate']:.6f} | "
            f"{km[f'rt600_plus_{KALMAN_ID}']:.6f} | {km['marginal_vs_clone']:+.6f} | "
            f"{km['gain_vs_base']:+.6f} | {kp['within_t_rank_corr_rt600']:+.4f} | "
            f"{kv['verdict']} ({kv['continuation_status']}) |"
        ),
        (
            f"| RT600 + {DMD_ID} | {dp['whole_fold']['candidate']:.6f} | "
            f"{dm[f'rt600_plus_{DMD_ID}']:.6f} | {dm['marginal_vs_clone']:+.6f} | "
            f"{dm['gain_vs_base']:+.6f} | {dp['within_t_rank_corr_rt600']:+.4f} | "
            f"{dv['verdict']} ({dv['continuation_status']}) |"
        ),
        "",
        f"Overall continuation status: **{result['overall_continuation_status']}**.",
        f"Kalman failed gates: {result['verdicts'][KALMAN_ID]['gate_failures']}.",
        f"Hankel-DMD failed gates: {result['verdicts'][DMD_ID]['gate_failures']}.",
        "",
        "## Diagnostic Pack",
        "",
        "| arm | whole-fold AUC | dominant-cell AUC | mature vs never | mature vs pre | rho vs RT600 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| `{KALMAN_ID}` | {kp['whole_fold']['candidate']:.6f} | "
            f"{kp['dominant_cell']['candidate']:.6f} | "
            f"{kp['mature_vs_neverbreak']['candidate']:.6f} | "
            f"{kp['mature_vs_prebreak']['candidate']:.6f} | "
            f"{kp['within_t_rank_corr_rt600']:+.4f} |"
        ),
        (
            f"| `{DMD_ID}` | {dp['whole_fold']['candidate']:.6f} | "
            f"{dp['dominant_cell']['candidate']:.6f} | "
            f"{dp['mature_vs_neverbreak']['candidate']:.6f} | "
            f"{dp['mature_vs_prebreak']['candidate']:.6f} | "
            f"{dp['within_t_rank_corr_rt600']:+.4f} |"
        ),
        "",
        "## Pair Flow",
        "",
        f"### {KALMAN_ID}",
        "",
        "| split | repairs | damage | net | sampled pairs |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in kalman["pair_flow"].items():
        md.append(f"| `{name}` | {row['repairs']} | {row['damage']} | {row['net_pair_lift']} | {row['total_pairs_sampled']} |")
    md += ["", f"### {DMD_ID}", "", "| split | repairs | damage | net | sampled pairs |", "|---|---:|---:|---:|---:|"]
    for name, row in dmd["pair_flow"].items():
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
        "Pilot 3 / IM3 tests whether individualized frozen historical observers expose "
        "break information through innovation or dynamical-subspace residuals."
    )
    kalman_oof, kalman_train = train_foldpure_oof(
        KALMAN_ID,
        KALMAN_MODULE,
        common_hypothesis + " The Kalman/NIS arm should add ensemble alpha through filter-consistency residuals.",
        "KILL if marginal_vs_clone < +0.0010 against RT600 + RT-401 seed clone.",
        "Pilot 3 frozen AR(2)-state Kalman/NIS observer arm. Fold-0 screen only; folds 1-4 generated only as fold-pure SCDF calibration support.",
    )
    dmd_oof, dmd_train = train_foldpure_oof(
        DMD_ID,
        DMD_MODULE,
        common_hypothesis + " The Hankel-DMD arm should add ensemble alpha through delay-embedded reconstruction and rank dynamics.",
        "KILL if marginal_vs_clone < +0.0010 or dominant-cell within-t rho with RT600 exceeds 0.85.",
        "Pilot 3 frozen Hankel-DMD observer arm. Fold-0 screen only; folds 1-4 generated only as fold-pure SCDF calibration support.",
    )

    kalman = candidate_result(KALMAN_ID, kalman_oof, base, c)
    dmd = candidate_result(DMD_ID, dmd_oof, base, c)
    kalman["training"] = kalman_train
    dmd["training"] = dmd_train

    kv, kc, kr, kf = arm_verdict(
        "kalman_nis",
        kalman["ensemble_marginal"]["marginal_vs_clone"],
        kalman["diagnostic_pack"]["within_t_rank_corr_rt600"],
    )
    dv, dc, dr, df = arm_verdict(
        "hankel_dmd",
        dmd["ensemble_marginal"]["marginal_vs_clone"],
        dmd["diagnostic_pack"]["within_t_rank_corr_rt600"],
    )
    verdicts = {
        KALMAN_ID: {"verdict": kv, "continuation_status": kc, "reason": kr, "gate_failures": kf},
        DMD_ID: {"verdict": dv, "continuation_status": dc, "reason": dr, "gate_failures": df},
    }
    continuation_arms = [eid for eid, v in verdicts.items() if v["continuation_status"] == "CONTINUE"]
    overall = "STOP_FOR_5FOLD_CONFIRMATION" if continuation_arms else "CONTINUE_TO_PILOT8"

    if continuation_arms:
        interpretation = (
            "At least one preregistered observer arm cleared the fold-0 continuation gate. "
            "The broad sweep stops here for 5-fold confirmation before any Pilot 8 scoring."
        )
    else:
        interpretation = (
            "Neither observer arm cleared the preregistered 5-fold continuation gate. "
            "This kills or shelves these exact frozen-observer constructions under the "
            "Mode-A fold-0 ABL screen, with m04_resid already present in the base bank; "
            "it does not falsify all state-space or delay-embedding observer ideas."
        )

    result = {
        "generated": "2026-08-24",
        "branch": "research/new-avenues-pilots-2026",
        "prereg_sha": PREREG_SHA,
        "head_sha": git_sha(),
        "exp_ids": [KALMAN_ID, DMD_ID],
        "feature_modules": [KALMAN_MODULE, DMD_MODULE],
        "feature_definitions": {
            "kalman_q_grid": [float(x) for x in KALMAN_Q_GRID],
            "kalman_r_grid": [float(x) for x in KALMAN_R_GRID],
            "kalman_windows": [int(x) for x in KALMAN_WINDOWS],
            "kalman_cols": KALMAN_COLS,
            "dmd_delay": int(DMD_DELAY),
            "dmd_rank": int(DMD_RANK),
            "dmd_windows": [int(x) for x in DMD_WINDOWS],
            "dmd_cols": DMD_COLS,
        },
        "causal_checks": checks,
        "rt600_sentinel": sentinel,
        "feature_cache": feature_cache,
        KALMAN_ID: kalman,
        DMD_ID: dmd,
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
                    KALMAN_ID: {
                        "ensemble_marginal": kalman["ensemble_marginal"],
                        "verdict": verdicts[KALMAN_ID],
                    },
                    DMD_ID: {
                        "ensemble_marginal": dmd["ensemble_marginal"],
                        "verdict": verdicts[DMD_ID],
                    },
                    "runtime_s": result["runtime_s"],
                }
            ),
            indent=2,
        )
    )
    print(f"wrote {OUTDIR / 'pilot03_observers.json'}")
    print(f"wrote {OUTDIR / 'pilot03_observers.md'}")


if __name__ == "__main__":
    main()
