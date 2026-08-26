#!/usr/bin/env python3
"""CRF-01 integration and evaluation.  NO-TORCH PROCESS.

Loads the frozen raw score arrays emitted by research/scripts/crf01_nncsr.py,
applies the canonical cross-fitted SCDF calibration, computes E0/E1/E2 and
marginal_vs_clone, computes the canonical pair flow, and writes the experiment
report.  Implements the frozen protocol in
research/reports/causal_representation_frontier/CRF01_EXECUTION_PREREG.md.

THIS FILE IMPORTS NO TORCH.  torch and lightgbm segfault sharing a process on
macOS/arm64 (Wave-6 finding), so the training process and this one are kept
separate.  KMP_DUPLICATE_LIB_OK and every equivalent hack are forbidden.

Stages, in the frozen order of CRF01_EXECUTION_PREREG section 8:

    --stage abandon    fold-0 standalone TS-AUC + within-t rho -> the section 7
                       cheap abandon gate.  Needs the fold-0 model only.
    --stage evaluate   full fold-0 evaluation: SCDF, E0/E1/E2, pair flow, gates.
                       Needs the complete five-fold OOF vector.
    --stage confirm    five-fold confirmation from the same OOF vectors, no new
                       training.  Only if the fold-0 screen passed.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time

import numpy as np

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
os.environ.setdefault("SBR_ROOT", ROOT)
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")
sys.path.insert(0, f"{ROOT}/research/scripts/novel_streams")

assert "torch" not in sys.modules, "crf01_integrate must not share a process with torch"

from harness import diagnostic_pack, rt600_blend                      # noqa: E402
from second_sweep_ss01 import (                                        # noqa: E402
    append_result_append_only, extended_pair_stats, finite_float, git_sha,
    pair_flow_pack, sample_eval_pairs_for_label,
)
from sbr.metric import ts_auc_flat                                     # noqa: E402
from wave5_lib import Ctx, FOLDS, SPECIALISTS                          # noqa: E402
from wave8_common import ensemble_marginal                             # noqa: E402

CACHE = os.environ.get("CRF01_CACHE", f"{ROOT}/cache/crf01")
OUTDIR = f"{ROOT}/research/reports/causal_representation_frontier"
OOFDIR = f"{ROOT}/research/oof"
SEEDCLONE = "RT-401"
EXECUTION_PREREG_SHA = "afba958"

ARMS = {"candidate": "RT-1234", "bce_control": "RT-1235", "shuffle_ctrl": "RT-1236"}

# --- frozen gates, CRF01_EXECUTION_PREREG sections 7 and 13 ------------------
ABANDON_STANDALONE = 0.600
ABANDON_RHO = 0.60
GATE_PRIMARY = 0.0015
GATE_C1_GAP = 0.0010
GATE_DAMAGE_CAP = 0.0150
SERIOUS_MEAN = 0.0030
SERIOUS_FOLDS = 4

RT600_EXPECTED = {"mean": 0.625811, "pooled": 0.625627, "dominant_cell": 0.664277,
                  "fold0_e0": 0.638276, "fold0_e1_seedclone": 0.638586}


# ---------------------------------------------------------------- assembly
def load_arm(arm: str, folds=FOLDS, n_rows: int | None = None):
    """Assemble one arm's fold-pure OOF vector from the frozen per-fold arrays."""
    exp_id = ARMS[arm]
    if n_rows is None:
        raise ValueError("n_rows required")
    oof = np.full(n_rows, np.nan, dtype=np.float32)
    meta = {}
    for f in folds:
        tag = f"{CACHE}/scores/{exp_id}_fold{f}"
        if not os.path.exists(f"{tag}.scores.npy"):
            return None, {}
        rows = np.load(f"{tag}.rows.npy")
        vals = np.load(f"{tag}.scores.npy")
        assert np.isnan(oof[rows]).all(), f"{exp_id} fold {f} overlaps an earlier fold"
        oof[rows] = vals
        meta[str(f)] = json.load(open(f"{tag}.json"))
    return oof, meta


def rt600_sentinel(c: Ctx, base: np.ndarray) -> dict:
    """Refuse to interpret any CRF-01 number if the RT600 baseline has moved."""
    mean_auc, per_fold = c.score(base)
    dom = c.dev[(c.d.t[c.dev] >= 200) & ((c.d.y[c.dev] == 0) | (c.age[c.dev] >= 100))]
    m = ensemble_marginal(base, c=c, fold=0, label="rt600_self")
    out = {"mean": mean_auc, "per_fold": per_fold, "pooled": c.pooled(base),
           "dominant_cell": float(ts_auc_flat(base[dom], c.d.y[dom], c.d.t[dom])),
           "fold0_e0": m["rt600_7stream"], "fold0_e1_seedclone": m["rt600_plus_seedclone"]}
    out["expected"] = RT600_EXPECTED
    out["diffs"] = {k: out[k] - RT600_EXPECTED[k] for k in RT600_EXPECTED}
    out["ok"] = bool(all(abs(v) < 0.005 for v in out["diffs"].values()))
    if not out["ok"]:
        raise SystemExit(f"RT600 sentinel materially differs: {out}")
    return out


def lockbox_check(name: str, v: np.ndarray, c: Ctx) -> int:
    lb = c.d.rows_for([-1])
    n = int(np.isfinite(v[lb]).sum())
    if n != 0:
        raise SystemExit(f"{name} produced {n} finite lockbox rows")
    return n


# ------------------------------------------------------------- pair diagnostics
def unique_repair_coverage(c: Ctx, base: np.ndarray, cand: np.ndarray,
                           clone: np.ndarray, label="dominant_cell", fold=0) -> dict:
    """Repairs the candidate makes that the matched seed clone does NOT make.

    The whole program is measured against E1 = E0 + RT-401, so the comparator
    for "unique" is the exchangeable eighth member, not another candidate.
    Same canonical deterministic pair sample as every other pair statistic.
    """
    pp, nn = sample_eval_pairs_for_label(c, label, fold=fold)
    if len(pp) == 0:
        return {"rt600_wrong": 0, "candidate_repairs": 0, "clone_repairs": 0,
                "unique_repairs": 0, "unique_repair_coverage": None,
                "repair_jaccard_vs_clone": None}
    wrong = ~(base[pp] > base[nn])
    r_cand = wrong & (cand[pp] > cand[nn])
    r_clone = wrong & (clone[pp] > clone[nn])
    inter = int((r_cand & r_clone).sum())
    union = int((r_cand | r_clone).sum())
    w = int(wrong.sum())
    return {
        "rt600_wrong": w,
        "candidate_repairs": int(r_cand.sum()),
        "clone_repairs": int(r_clone.sum()),
        "unique_repairs": int((r_cand & ~r_clone).sum()),
        "unique_repair_coverage": float((r_cand & ~r_clone).sum() / w) if w else None,
        "repair_jaccard_vs_clone": float(inter / union) if union else None,
    }


def prebreak_damage_rate(c: Ctx, base: np.ndarray, cand: np.ndarray, fold=0) -> float | None:
    """Damage rate on RT600-CORRECT pre-break pairs -- the SS-03 cap (0.0199 failed)."""
    s = extended_pair_stats(c, base, cand, "cell_pre_break_neg", fold=fold)
    return s["damage_rate_of_rt600_right"]


def arm_pack(c: Ctx, base: np.ndarray, clone_cal: np.ndarray, score: np.ndarray,
             label: str, fold: int = 0) -> dict:
    """The mandatory report of CRF_PROGRAM_PREREG section 0.3, for one arm."""
    r = c.rows[fold]
    dom = r[(c.d.t[r] >= 200) & ((c.d.y[r] == 0) | (c.age[r] >= 100))]
    pack = diagnostic_pack(c, score, base, fold=fold, label=label)
    flow = pair_flow_pack(c, base, score, fold=fold)
    marg = ensemble_marginal(score, c=c, fold=fold, label=label)
    return {
        "label": label,
        "standalone_whole_fold_ts_auc": float(
            ts_auc_flat(score[r], c.d.y[r], c.d.t[r])),
        "standalone_dominant_cell_ts_auc": float(
            ts_auc_flat(score[dom], c.d.y[dom], c.d.t[dom])),
        "within_t_rho_vs_rt600": pack["within_t_rank_corr_rt600"],
        "diagnostic_pack": pack,
        "pair_flow": flow,
        "unique_repair": unique_repair_coverage(c, base, score, clone_cal, fold=fold),
        "prebreak_damage_rate_on_rt600_correct": prebreak_damage_rate(
            c, base, score, fold=fold),
        "ensemble": marg,
        "marginal_vs_clone": marg["marginal_vs_clone"],
    }


# --------------------------------------------------------------------- stages
def stage_abandon(c: Ctx, base: np.ndarray) -> dict:
    """Section 7.  Fold-0 model only -- this is the genuinely cheap first read."""
    tag = f"{CACHE}/scores/{ARMS['candidate']}_fold0"
    rows = np.load(f"{tag}.rows.npy")
    vals = np.load(f"{tag}.scores.npy")
    score = np.full(len(c.d.y), np.nan, dtype=np.float32)
    score[rows] = vals
    lockbox_check("candidate fold-0", score, c)
    r = c.rows[0]
    assert np.isfinite(score[r]).all(), "candidate is not finite on every fold-0 row"
    dom = r[(c.d.t[r] >= 200) & ((c.d.y[r] == 0) | (c.age[r] >= 100))]
    pack = diagnostic_pack(c, score, base, fold=0, label="RT-1234")
    standalone = float(ts_auc_flat(score[r], c.d.y[r], c.d.t[r]))
    rho = float(pack["within_t_rank_corr_rt600"])
    fired = bool(standalone < ABANDON_STANDALONE and rho <= ABANDON_RHO)
    return {
        "stage": "abandon_gate", "experiment_id": ARMS["candidate"],
        "standalone_whole_fold_ts_auc": standalone,
        "standalone_dominant_cell_ts_auc": float(
            ts_auc_flat(score[dom], c.d.y[dom], c.d.t[dom])),
        "within_t_rho_vs_rt600": rho,
        "thresholds": {"standalone_min": ABANDON_STANDALONE, "rho_max": ABANDON_RHO},
        "abandon_gate_fired": fired,
        "verdict": "ABANDON" if fired else "CONTINUE",
        "diagnostic_pack": pack,
    }


def stage_evaluate(c: Ctx, base: np.ndarray, arms: list[str]) -> dict:
    n_rows = len(c.d.y)
    clone_cal = c.crossfit_blend({SEEDCLONE: np.load(f"{OOFDIR}/{SEEDCLONE}.npy")},
                                 [SEEDCLONE])
    out = {"stage": "fold0_evaluate", "arms": {}}
    oofs = {}
    for arm in arms:
        oof, meta = load_arm(arm, n_rows=n_rows)
        if oof is None:
            print(f"  {arm}: incomplete five-fold OOF, skipped", flush=True)
            continue
        lockbox_check(ARMS[arm], oof, c)
        assert np.isfinite(oof[c.dev]).all(), f"{ARMS[arm]} is not finite on every dev row"
        np.save(f"{OOFDIR}/{ARMS[arm]}.npy", oof)
        oofs[arm] = oof
        out["arms"][arm] = arm_pack(c, base, clone_cal, oof, ARMS[arm], fold=0)
        out["arms"][arm]["fold_meta"] = {
            k: {kk: vv for kk, vv in v.items()
                if kk in ("state_sha256", "runtime_s", "n_train_series",
                          "n_val_series", "n_parameters", "empty_pair_steps")}
            for k, v in meta.items()}
        m = out["arms"][arm]["marginal_vs_clone"]
        print(f"  {arm} ({ARMS[arm]}): marginal_vs_clone {m:+.6f}", flush=True)

    if "candidate" in out["arms"]:
        cand = out["arms"]["candidate"]
        g = {}
        g["primary_marginal"] = {
            "value": cand["marginal_vs_clone"], "threshold": GATE_PRIMARY,
            "pass": bool(cand["marginal_vs_clone"] >= GATE_PRIMARY)}
        if "bce_control" in out["arms"]:
            gap = cand["marginal_vs_clone"] - out["arms"]["bce_control"]["marginal_vs_clone"]
            g["objective_isolation_vs_c1"] = {
                "value": gap, "threshold": GATE_C1_GAP, "pass": bool(gap >= GATE_C1_GAP)}
        dn = cand["pair_flow"]["dominant_cell"]["net_pair_lift"]
        mn = cand["pair_flow"]["mature_vs_never"]["net_pair_lift"]
        g["dominant_pair_net"] = {"value": dn, "threshold": 0, "pass": bool(dn > 0)}
        g["mature_vs_never_net"] = {"value": mn, "threshold": 0, "pass": bool(mn > 0)}
        dr = cand["prebreak_damage_rate_on_rt600_correct"]
        g["prebreak_damage_cap"] = {
            "value": dr, "threshold": GATE_DAMAGE_CAP,
            "pass": bool(dr is not None and dr <= GATE_DAMAGE_CAP)}
        out["gates"] = g
        out["all_gates_pass"] = bool(all(v["pass"] for v in g.values()))
        m = cand["marginal_vs_clone"]
        out["band"] = ("MAJOR" if m >= 0.0050 else "SERIOUS" if m >= 0.0030
                       else "WEAK" if m >= 0.0015 else "KILL")
        out["verdict"] = "SCREEN_PASS" if out["all_gates_pass"] else "KILL"
    return out


def stage_confirm(c: Ctx, base: np.ndarray, arms: list[str]) -> dict:
    """Five-fold confirmation.  No new training: the OOF vectors already exist."""
    n_rows = len(c.d.y)
    out = {"stage": "five_fold_confirm", "arms": {}}
    for arm in arms:
        oof, _ = load_arm(arm, n_rows=n_rows)
        if oof is None:
            continue
        per = []
        for f in FOLDS:
            per.append(ensemble_marginal(oof, c=c, fold=f,
                                         label=ARMS[arm])["marginal_vs_clone"])
        out["arms"][arm] = {
            "per_fold_marginal_vs_clone": [float(x) for x in per],
            "mean_marginal_vs_clone": float(np.mean(per)),
            "n_folds_positive": int(sum(1 for x in per if x > 0)),
        }
    if "candidate" in out["arms"]:
        a = out["arms"]["candidate"]
        out["serious"] = bool(a["mean_marginal_vs_clone"] >= SERIOUS_MEAN
                              and a["n_folds_positive"] >= SERIOUS_FOLDS)
    return out


# ------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("abandon", "evaluate", "confirm", "sentinel"))
    ap.add_argument("--arms", default="candidate,bce_control")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    t0 = time.time()
    c = Ctx()
    base = rt600_blend(c)
    sent = rt600_sentinel(c, base)
    print(f"RT600 sentinel OK: fold0 E0 {sent['fold0_e0']:.6f} "
          f"E1 {sent['fold0_e1_seedclone']:.6f}", flush=True)
    if a.stage == "sentinel":
        res = {"stage": "sentinel", "rt600_sentinel": sent}
    elif a.stage == "abandon":
        res = stage_abandon(c, base)
    elif a.stage == "evaluate":
        res = stage_evaluate(c, base, a.arms.split(","))
    else:
        res = stage_confirm(c, base, a.arms.split(","))
    res["rt600_sentinel"] = sent
    res["git_sha"] = git_sha()
    res["execution_prereg_sha"] = EXECUTION_PREREG_SHA
    res["runtime_s"] = round(time.time() - t0, 1)
    res["lockbox_touched"] = "no"
    res["test_reduced_touched"] = "no"
    os.makedirs(OUTDIR, exist_ok=True)
    path = a.out or f"{OUTDIR}/crf01_{a.stage}.json"
    json.dump(finite_float(res), open(path, "w"), indent=1)
    print(json.dumps(finite_float(
        {k: v for k, v in res.items()
         if k not in ("diagnostic_pack", "rt600_sentinel", "arms")}), indent=1))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
