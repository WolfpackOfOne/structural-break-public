"""Pilot 6: spectral impulsiveness contrast.

Scored candidates:
  RT-1206: spectral impulsiveness contrast features.
  RT-1207: matched plain band-energy control.
"""
from __future__ import annotations

import csv
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
from sbr.features.m14_spectral_impulse import (
    BAND_NAMES,
    CONTRAST_COLS,
    ENERGY_COLS,
    GRID,
    MIN_L,
    SEG,
    spectral_impulse_features,
)
from sbr.metric import ts_auc_flat
from sbr.store import load_store
from wave2_lib import ABL, FULL
from wave5_lib import Ctx, SPECIALISTS, load_oof

PREREG_SHA = "dddc2d9"
CONTRAST_ID = "RT-1206"
CONTROL_ID = "RT-1207"
CONTRAST_MODULE = "m14_spectral_impulse"
CONTROL_MODULE = "m14_spectral_energy"
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


class ContrastMech(StreamingMechanism):
    name = "pilot06_spectral_contrast"
    cols = CONTRAST_COLS

    def emit(self, hist, online):
        contrast, _ = spectral_impulse_features(make_ctx(hist, online))
        return contrast


class EnergyMech(StreamingMechanism):
    name = "pilot06_spectral_energy"
    cols = ENERGY_COLS

    def emit(self, hist, online):
        _, energy = spectral_impulse_features(make_ctx(hist, online))
        return energy


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


def append_result_append_only(res: dict) -> None:
    """Append one RESULTS.csv row without rewriting previous rows."""
    import fcntl

    path = ROOT / "research" / "RESULTS.csv"
    with open(str(path) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with path.open(newline="") as f:
            fieldnames = next(csv.reader(f))
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
            writer.writerow({k: res.get(k, "") for k in fieldnames})
        fcntl.flock(lk, fcntl.LOCK_UN)


def run_causal_checks(c: Ctx) -> dict:
    load_all()
    checks: dict[str, object] = {
        "segment_length": SEG,
        "min_endpoint_count": MIN_L,
        "null_grid": [int(x) for x in GRID],
        "bands": list(BAND_NAMES),
    }
    for name, mech in (("contrast", ContrastMech()), ("energy", EnergyMech())):
        ok, msg = harness_verify(mech)
        checks[f"harness_verify_{name}"] = {"ok": ok, "message": msg}
        if not ok:
            raise SystemExit(f"harness.verify failed for {name}: {msg}")

    st = load_store(str(ROOT / "cache" / "store"))
    dev_series = np.unique(c.d.sidx[c.dev])
    lens = c.n_online[dev_series]
    pick = dev_series[np.argsort(lens)[np.linspace(0, len(lens) - 1, 8).astype(int)]]
    prefix_results = {}
    for module in (CONTRAST_MODULE, CONTROL_MODULE):
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
    contrast, energy = spectral_impulse_features(make_ctx(hist, online))
    first_valid = {
        "contrast_rows_before_46_all_nan": bool(np.isnan(contrast[:46]).all()),
        "energy_rows_before_46_all_nan": bool(np.isnan(energy[:46]).all()),
        "contrast_row46_any_finite": bool(np.isfinite(contrast[46]).any()) if len(contrast) > 46 else None,
        "energy_row46_any_finite": bool(np.isfinite(energy[46]).any()) if len(energy) > 46 else None,
    }
    if not all(v for v in first_valid.values() if v is not None):
        raise SystemExit(f"first-valid check failed: {first_valid}")
    checks["first_valid_nan_semantics"] = first_valid

    cut = min(91, max(2, len(online) // 2))
    mutated = online.copy()
    mutated[cut:] = mutated[cut:][::-1] * -4.0 + 1.5
    m_contrast, m_energy = spectral_impulse_features(make_ctx(hist, mutated))
    future_ok = bool(
        np.allclose(contrast[:cut], m_contrast[:cut], rtol=0.0, atol=0.0, equal_nan=True)
        and np.allclose(energy[:cut], m_energy[:cut], rtol=0.0, atol=0.0, equal_nan=True)
    )
    if not future_ok:
        raise SystemExit("future-mutation check failed")
    checks["future_mutation_prefix_ok"] = future_ok

    r_contrast, r_energy = spectral_impulse_features(make_ctx(hist, online))
    replay_ok = bool(
        np.array_equal(contrast, r_contrast, equal_nan=True)
        and np.array_equal(energy, r_energy, equal_nan=True)
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


def build_dev_feature_caches(c: Ctx) -> dict:
    load_all()
    for module_name in (CONTRAST_MODULE, CONTROL_MODULE):
        if module_name not in REGISTRY:
            raise SystemExit(f"unknown feature module {module_name}")

    t0 = time.time()
    st = load_store(str(ROOT / "cache" / "store"))
    contrast = np.full((len(c.d.y), len(CONTRAST_COLS)), np.nan, dtype=np.float32)
    energy = np.full((len(c.d.y), len(ENERGY_COLS)), np.nan, dtype=np.float32)
    dev_series = np.unique(c.d.sidx[c.dev])
    for count, sid in enumerate(dev_series):
        rows = c.dev[c.d.sidx[c.dev] == sid]
        rows = rows[np.argsort(c.d.t[rows], kind="stable")]
        con, eng = spectral_impulse_features(make_ctx(st.hist(int(sid)), st.online(int(sid))))
        if len(con) != len(rows) or len(eng) != len(rows):
            raise SystemExit(f"Pilot 6 row mismatch on series {int(sid)}")
        contrast[rows] = con
        energy[rows] = eng
        if count and count % 1000 == 0:
            print(f"pilot06 features: built {count}/{len(dev_series)} series {time.time() - t0:.0f}s", flush=True)

    np.save(FEATDIR / f"{CONTRAST_MODULE}.npy", contrast)
    np.save(FEATDIR / f"{CONTROL_MODULE}.npy", energy)
    (FEATDIR / f"{CONTRAST_MODULE}.cols.json").write_text(
        json.dumps({"cols": CONTRAST_COLS, "version": REGISTRY[CONTRAST_MODULE].version, "owner": REGISTRY[CONTRAST_MODULE].owner}, sort_keys=True) + "\n"
    )
    (FEATDIR / f"{CONTROL_MODULE}.cols.json").write_text(
        json.dumps({"cols": ENERGY_COLS, "version": REGISTRY[CONTROL_MODULE].version, "owner": REGISTRY[CONTROL_MODULE].owner}, sort_keys=True) + "\n"
    )
    return {
        CONTRAST_MODULE: {
            "module": CONTRAST_MODULE,
            "cols": CONTRAST_COLS,
            "shape": [int(contrast.shape[0]), int(contrast.shape[1])],
            "dev_series": int(len(dev_series)),
            "lockbox_rows_filled": int(np.isfinite(contrast[c.d.rows_for([-1])]).sum()),
        },
        CONTROL_MODULE: {
            "module": CONTROL_MODULE,
            "cols": ENERGY_COLS,
            "shape": [int(energy.shape[0]), int(energy.shape[1])],
            "dev_series": int(len(dev_series)),
            "lockbox_rows_filled": int(np.isfinite(energy[c.d.rows_for([-1])]).sum()),
        },
        "runtime_s": time.time() - t0,
    }


def train_foldpure_oof(exp_id: str, module_name: str, hypothesis: str, falsification: str, notes: str) -> tuple[np.ndarray, dict]:
    t0 = time.time()
    d = PL.Data(screen=False)
    modules = FULL + [module_name]
    mats, names = PL.load_features(modules, screen=False)
    keep_idx = np.arange(len(names), dtype=np.int64)
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
        item = {"fold": int(f), "train_rows": int(len(tr_rows)), "valid_rows": int(len(va_rows)), "runtime_s": round(time.time() - t0, 1)}
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
    append_result_append_only(res)
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


def gate_verdict(contrast_margin: float, control_margin: float) -> tuple[str, str, str, list[dict]]:
    gap = contrast_margin - control_margin
    failures = []
    if contrast_margin < 0.0010:
        failures.append({"gate": "primary_marginal_vs_clone", "threshold": 0.0010, "observed": contrast_margin, "message": "contrast marginal_vs_clone is below +0.0010"})
    if control_margin >= contrast_margin:
        failures.append({"gate": "plain_energy_control_not_worse", "threshold": "control < contrast", "observed_contrast_minus_control": gap, "message": "plain-energy control matches or exceeds contrast"})
    elif gap < 0.0005:
        failures.append({"gate": "contrast_control_distinguishability", "threshold": 0.0005, "observed_contrast_minus_control": gap, "message": "contrast-control gap is below the preregistered +0.0005 distinguishability floor"})
    if contrast_margin < 0.0010:
        return "KILL", "KILL", failures[0]["message"], failures
    if control_margin >= contrast_margin:
        return "KILL", "KILL", "plain-energy control matches or exceeds contrast", failures
    if gap < 0.0005:
        return "KILL", "KILL", failures[0]["message"], failures
    if contrast_margin < 0.0020:
        return "WEAK", "NO_5FOLD", "contrast clears kill floor but remains in the weak +0.001 to +0.002 band", failures
    if contrast_margin < 0.0030:
        return "INTERESTING", "CONTINUE", "contrast clears the preregistered 5-fold continuation gate", failures
    if contrast_margin < 0.0050:
        return "SERIOUS", "CONTINUE", "contrast clears the serious screen band; confirm before any next pilot", failures
    if contrast_margin < 0.0080:
        return "MAJOR", "CONTINUE", "contrast clears the major screen band; confirm before any next pilot", failures
    return "BREAKTHROUGH", "CONTINUE", "contrast clears the breakthrough screen band; confirm before any next pilot", failures


def write_report(result: dict) -> None:
    json_path = OUTDIR / "pilot06_spectral_impulse.json"
    md_path = OUTDIR / "pilot06_spectral_impulse.md"
    json_path.write_text(json.dumps(finite_float(result), indent=2, sort_keys=True) + "\n")

    contrast = result[CONTRAST_ID]
    control = result[CONTROL_ID]
    cm = contrast["ensemble_marginal"]
    em = control["ensemble_marginal"]
    cp = contrast["diagnostic_pack"]
    ep = control["diagnostic_pack"]
    checks = result["causal_checks"]
    sentinel = result["rt600_sentinel"]

    md = [
        "# PILOT 6 -- SPECTRAL IMPULSIVENESS CONTRAST",
        "",
        f"Pre-registration: `research/reports/new_avenues_2026/PILOT06_PREREG.md` at `{PREREG_SHA}`.",
        f"Experiment IDs: `{CONTRAST_ID}` contrast candidate, `{CONTROL_ID}` plain-energy control.",
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
        "Four Goertzel bands from the `m03_dyn` dyadic frequency bank. Segment "
        "length is 32; adaptive online window is the trailing half-prefix; "
        "features are NaN until at least 16 finite segment-energy endpoints exist.",
        "",
        f"`{CONTRAST_ID}` emits eight contrast columns: energy_z times negative "
        "spectral-kurtosis z and negative robust-negentropy z for each band.",
        f"`{CONTROL_ID}` emits the four matched plain `energy_z` columns only.",
        "",
        "## Causality Checks",
        "",
        f"* Harness contrast: `{checks['harness_verify_contrast']['message']}`.",
        f"* Harness energy: `{checks['harness_verify_energy']['message']}`.",
        f"* Future mutation prefix check: `{checks['future_mutation_prefix_ok']}`.",
        f"* Deterministic replay: `{checks['deterministic_replay_ok']}`.",
        f"* First-valid semantics: `{checks['first_valid_nan_semantics']}`.",
        "",
        "## Binding Marginal Result",
        "",
        "| arm | fold-0 standalone TS-AUC | E2 TS-AUC | marginal vs clone | gain vs RT600 |",
        "|---|---:|---:|---:|---:|",
        f"| RT600 |  | {cm['rt600_7stream']:.6f} |  |  |",
        f"| RT600 + RT-401 seed clone |  | {cm['rt600_plus_seedclone']:.6f} |  |  |",
        (
            f"| RT600 + {CONTRAST_ID} | {cp['whole_fold']['candidate']:.6f} | "
            f"{cm[f'rt600_plus_{CONTRAST_ID}']:.6f} | {cm['marginal_vs_clone']:+.6f} | "
            f"{cm['gain_vs_base']:+.6f} |"
        ),
        (
            f"| RT600 + {CONTROL_ID} | {ep['whole_fold']['candidate']:.6f} | "
            f"{em[f'rt600_plus_{CONTROL_ID}']:.6f} | {em['marginal_vs_clone']:+.6f} | "
            f"{em['gain_vs_base']:+.6f} |"
        ),
        "",
        f"Contrast minus plain-energy marginal: `{result['contrast_minus_control_marginal']:+.6f}`.",
        f"Verdict: **{result['verdict']}** ({result['continuation_status']}) -- {result['verdict_reason']}.",
        "Failed gates: "
        + ("; ".join(f"`{x['gate']}` ({x['message']})" for x in result["gate_failures"]) if result["gate_failures"] else "none"),
        "",
        "## Diagnostic Pack",
        "",
        "| candidate | whole-fold AUC | dominant-cell AUC | mature vs never | mature vs pre | rho vs RT600 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| `{CONTRAST_ID}` | {cp['whole_fold']['candidate']:.6f} | "
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
        f"### {CONTRAST_ID}",
        "",
        "| split | repairs | damage | net | sampled pairs |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in contrast["pair_flow"].items():
        md.append(f"| `{name}` | {row['repairs']} | {row['damage']} | {row['net_pair_lift']} | {row['total_pairs_sampled']} |")
    md += ["", f"### {CONTROL_ID}", "", "| split | repairs | damage | net | sampled pairs |", "|---|---:|---:|---:|---:|"]
    for name, row in control["pair_flow"].items():
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
        "Pilot 6 tests whether high band energy with low impulsiveness marks sustained structural breaks "
        "rather than isolated outlier bursts."
    )
    contrast_oof, contrast_train = train_foldpure_oof(
        CONTRAST_ID,
        CONTRAST_MODULE,
        common_hypothesis + " Spectral-kurtosis and robust-negentropy product contrasts should add ensemble alpha.",
        "KILL if contrast marginal_vs_clone < +0.0010, or if the plain-energy control matches within +0.0005.",
        "Pilot 6 spectral impulsiveness contrast arm. Fold-0 screen only; folds 1-4 generated only as fold-pure SCDF calibration support.",
    )
    control_oof, control_train = train_foldpure_oof(
        CONTROL_ID,
        CONTROL_MODULE,
        common_hypothesis + " Plain band energy is the binding control for generic extra spectral power.",
        "Control must not match or exceed the contrast arm; if it does, spectral impulsiveness hypothesis is killed.",
        "Pilot 6 plain band-energy control. Fold-0 screen only; folds 1-4 generated only as fold-pure SCDF calibration support.",
    )

    contrast_result = candidate_result(CONTRAST_ID, contrast_oof, base, c)
    control_result = candidate_result(CONTROL_ID, control_oof, base, c)
    contrast_result["training"] = contrast_train
    control_result["training"] = control_train

    contrast_margin = contrast_result["ensemble_marginal"]["marginal_vs_clone"]
    control_margin = control_result["ensemble_marginal"]["marginal_vs_clone"]
    verdict, continuation_status, reason, gate_failures = gate_verdict(contrast_margin, control_margin)

    if verdict == "KILL":
        interpretation = (
            "The preregistered spectral impulsiveness contrast failed a Pilot 6 binding gate. "
            "This falsifies the F1/F6 product-contrast construction under the fixed dyadic "
            "Goertzel bands, SEG=32 envelope, robust historical-null calibration, and Mode-A "
            "fold-0 ABL screen; it does not falsify all spectral representations."
        )
    elif continuation_status == "CONTINUE":
        interpretation = (
            "The contrast arm cleared the preregistered fold-0 continuation gate and beat "
            "the plain-energy control by the required gap. The next step is 5-fold confirmation."
        )
    else:
        interpretation = (
            "The contrast arm cleared the kill floor but remained in the weak band. It is not "
            "promoted without an explicit continuation decision."
        )

    result = {
        "generated": "2026-08-24",
        "branch": "research/new-avenues-pilots-2026",
        "prereg_sha": PREREG_SHA,
        "head_sha": git_sha(),
        "exp_ids": [CONTRAST_ID, CONTROL_ID],
        "feature_modules": [CONTRAST_MODULE, CONTROL_MODULE],
        "feature_definitions": {
            "bands": list(BAND_NAMES),
            "segment_length": SEG,
            "min_endpoint_count": MIN_L,
            "null_grid": [int(x) for x in GRID],
            "contrast_cols": CONTRAST_COLS,
            "energy_cols": ENERGY_COLS,
        },
        "causal_checks": checks,
        "rt600_sentinel": sentinel,
        "feature_cache": feature_cache,
        CONTRAST_ID: contrast_result,
        CONTROL_ID: control_result,
        "contrast_minus_control_marginal": contrast_margin - control_margin,
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
                    CONTRAST_ID: contrast_result["ensemble_marginal"],
                    CONTROL_ID: control_result["ensemble_marginal"],
                    "contrast_minus_control": contrast_margin - control_margin,
                    "runtime_s": result["runtime_s"],
                }
            ),
            indent=2,
        )
    )
    print(f"wrote {OUTDIR / 'pilot06_spectral_impulse.json'}")
    print(f"wrote {OUTDIR / 'pilot06_spectral_impulse.md'}")


if __name__ == "__main__":
    main()
