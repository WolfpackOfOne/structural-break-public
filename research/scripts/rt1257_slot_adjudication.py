#!/usr/bin/env python3
"""RT-1257-relative adjudication for residual CatBoost slot candidates.

This is a zero-training artifact re-score. It evaluates the remaining CSA-04
CatBoost slot candidates after RT-1257's CAT-300 and CAT-413 swaps are already
installed, so the question is incremental value over the current champion.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = REPO.parent / "structural-break-deep-ensemble-frontier-local-2026"
REPORT_DIR = REPO / "research" / "reports" / "deep_ensemble_frontier_2026" / "local"
RESULT_JSON = REPORT_DIR / "RT1257_SLOT_ADJUDICATION.json"
RESULT_MD = REPORT_DIR / "RT1257_SLOT_ADJUDICATION.md"

for _p in (REPO / "src", REPO / "research" / "scripts"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

import csa04r_reanalysis_2026 as bootlib  # noqa: E402
import deep_ensemble_local_2026 as local  # noqa: E402

from sbr.metric import ts_auc_flat  # noqa: E402

RUN_DATE = "2026-08-31"
FOLDS = (0, 1, 2, 3, 4)
NOISE_FLOOR = 0.0011
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260831
EXPECTED_RT600 = 0.6258113418832281
EXPECTED_RT1257 = 0.6278376636118985
TOL = 1e-9

RT1257_STREAMS = (
    "RT-1255",
    "RT-410",
    "RT-411",
    "RT-412",
    "RT-1254",
    "RT-414",
    "RT-415",
)

CANDIDATES: dict[str, dict[str, Any]] = {
    "CAT-412": {
        "id": "RT-1261",
        "ticket": "RT-1261 / CAT-412",
        "priority": 1,
        "note": "Strongest remaining residual slot candidate.",
    },
    "CAT-415": {
        "id": "RT-1263",
        "ticket": "RT-1263 / CAT-415",
        "priority": 2,
        "note": "Positive individually; no corrected multi-slot lift.",
    },
    "CAT-414": {
        "id": "RT-1262",
        "ticket": "RT-1262 / CAT-414",
        "priority": 3,
        "note": "Positive individually; no corrected multi-slot lift.",
    },
    "CAT-411": {
        "id": "RT-1260",
        "ticket": "RT-1260 / CAT-411",
        "priority": 4,
        "note": "Weakest residual slot; passed only the original individual gate.",
    },
}

SPLITS = ("whole", "dominant_cell", "mature_vs_never", "mature_vs_prebreak")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--artifact-root",
        default=os.environ.get("SBR_ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT)),
        help="Root containing cache/store, folds, and frozen OOF artifacts.",
    )
    p.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    p.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return p.parse_args()


def git_sha(short: bool = True) -> str:
    args = ["git", "-C", str(REPO), "rev-parse"]
    if short:
        args.append("--short")
    args.append("HEAD")
    return subprocess.check_output(args, text=True).strip()


def fmt(x: float, digits: int = 9, signed: bool = False) -> str:
    sign = "+" if signed else ""
    return f"{float(x):{sign}.{digits}f}"


def unique(items: list[str] | tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(items))


def replace_stream(streams: tuple[str, ...], old: str, new: str) -> list[str]:
    if old not in streams:
        raise KeyError(f"{old} is not in stream composition {streams}")
    return [new if s == old else s for s in streams]


def copy_oof_if_missing(source_root: Path, local_oof_dir: Path, name: str) -> dict[str, Any]:
    src = source_root / "research" / "oof" / f"{name}.npy"
    dst = local_oof_dir / f"{name}.npy"
    if not src.exists():
        raise SystemExit(f"missing source OOF {src}")
    local_oof_dir.mkdir(parents=True, exist_ok=True)
    copied = False
    if not dst.exists():
        shutil.copy2(src, dst, follow_symlinks=True)
        copied = True
    return {
        "id": name,
        "source": str(src),
        "resolved_source": str(src.resolve(strict=True)),
        "local": str(dst),
        "bytes": int(dst.stat().st_size),
        "copied": copied,
    }


def ensure_local_oof_inputs(
    source_root: Path, local_oof_dir: Path, names: list[str]
) -> list[dict[str, Any]]:
    return [copy_oof_if_missing(source_root, local_oof_dir, name) for name in names]


def validate_oof_inputs(c: Any, local_oof_dir: Path, names: list[str]) -> dict[str, dict[str, Any]]:
    lockbox = c.d.rows_for([-1])
    out: dict[str, dict[str, Any]] = {}
    expected_dev_rows = int(len(c.dev))
    for name in names:
        path = local_oof_dir / f"{name}.npy"
        arr = np.load(path, mmap_mode="r")
        rec = {
            "path": str(path),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "finite_dev_rows": int(np.isfinite(arr[c.dev]).sum()),
            "finite_lockbox_rows": int(np.isfinite(arr[lockbox]).sum()),
        }
        if arr.shape != (len(c.d.y),):
            raise SystemExit(f"{name} wrong shape: {rec}")
        if str(arr.dtype) != "float32":
            raise SystemExit(f"{name} wrong dtype: {rec}")
        if rec["finite_dev_rows"] != expected_dev_rows:
            raise SystemExit(f"{name} wrong finite dev count: {rec}")
        if rec["finite_lockbox_rows"] != 0:
            raise SystemExit(f"{name} fills lockbox rows: {rec}")
        out[name] = rec
    return out


def score_per_fold(c: Any, vec: np.ndarray) -> list[float]:
    return [float(ts_auc_flat(vec[c.rows[f]], c.d.y[c.rows[f]], c.d.t[c.rows[f]])) for f in FOLDS]


def score_summary(c: Any, vec: np.ndarray) -> dict[str, Any]:
    per = score_per_fold(c, vec)
    return {
        "mean_ts_auc": float(np.mean(per)),
        "pooled_ts_auc": float(ts_auc_flat(vec[c.dev], c.d.y[c.dev], c.d.t[c.dev])),
        "per_fold_ts_auc": per,
        "fold_std": float(np.std(per)),
    }


def score_pack(csa: Any, c: Any, vec: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for split in SPLITS:
        rows = csa.cell_rows(c, c.dev, split)
        out[split] = float(ts_auc_flat(vec[rows], c.d.y[rows], c.d.t[rows]))
    return out


def diff_pack(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {k: float(a[k] - b[k]) for k in a}


def compare_value(label: str, observed: float, expected: float) -> dict[str, Any]:
    diff = float(observed - expected)
    return {
        "label": label,
        "observed": float(observed),
        "expected": float(expected),
        "diff": diff,
        "passed": bool(abs(diff) <= TOL),
    }


def bootstrap_summary(samples: np.ndarray) -> dict[str, Any]:
    rec = bootlib.summarize_samples(samples)
    rec["prob_gt_zero"] = float(np.mean(samples > 0.0))
    rec["prob_ge_noise_floor"] = float(np.mean(samples >= NOISE_FLOOR))
    return rec


def recommendation(rec: dict[str, Any]) -> str:
    primary = rec["delta_E2_minus_E1"]
    secondary = rec["delta_E2_minus_E0"]
    pboot = rec["bootstrap"]["E2_minus_E1"]
    sboot = rec["bootstrap"]["E2_minus_E0"]
    primary_bar = max(NOISE_FLOOR, float(pboot["se"]))
    secondary_bar = max(NOISE_FLOOR, float(sboot["se"]))
    dom_net = int(rec["pair_flow_vs_E0"]["dominant_cell"]["net_pair_lift"])
    mn_net = int(rec["pair_flow_vs_E0"]["mature_vs_never"]["net_pair_lift"])

    if (
        primary >= primary_bar
        and secondary >= secondary_bar
        and rec["positive_folds_E2_minus_E1"] >= 4
        and rec["positive_folds_E2_minus_E0"] >= 4
        and dom_net > 0
        and mn_net > 0
    ):
        return "CONTINUE_TO_ALT_PARTITIONS"
    if primary > 0.0 and secondary > 0.0 and rec["positive_folds_E2_minus_E0"] >= 3:
        return "RESEARCH_ALIVE_WEAK"
    if secondary > 0.0:
        return "PARK_NO_PROMOTION"
    return "KILL_OR_SUPERSEDED"


def build_candidate_record(
    csa: Any,
    c: Any,
    predictions: dict[str, np.ndarray],
    cache: Any,
    e0_vec: np.ndarray,
    e0_per: list[float],
    e0_pack: dict[str, float],
    name: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    spec = local.SPECS[name]
    cat_id = CANDIDATES[name]["id"]
    replaced = spec["replaced"]
    e1_streams = replace_stream(RT1257_STREAMS, replaced, local.MATCHED_CLONE)
    e2_streams = replace_stream(RT1257_STREAMS, replaced, cat_id)
    e1_vec, e1_per = csa.blend_fixed(c, cache, e1_streams)
    e2_vec, e2_per = csa.blend_fixed(c, cache, e2_streams)
    e1_pack = score_pack(csa, c, e1_vec)
    e2_pack = score_pack(csa, c, e2_vec)
    incumbent_standalone = score_summary(c, predictions[replaced])
    candidate_standalone = score_summary(c, predictions[cat_id])
    primary_per = [float(a - b) for a, b in zip(e2_per, e1_per)]
    secondary_per = [float(a - b) for a, b in zip(e2_per, e0_per)]
    control_per = [float(a - b) for a, b in zip(e1_per, e0_per)]

    rec: dict[str, Any] = {
        "name": name,
        "ticket": CANDIDATES[name]["ticket"],
        "id": cat_id,
        "priority": CANDIDATES[name]["priority"],
        "note": CANDIDATES[name]["note"],
        "replaced": replaced,
        "clone_control": local.MATCHED_CLONE,
        "training_spec": dict(spec),
        "E1_streams": e1_streams,
        "E2_streams": e2_streams,
        "incumbent_standalone": incumbent_standalone,
        "candidate_standalone": candidate_standalone,
        "standalone_delta_vs_incumbent": float(
            candidate_standalone["mean_ts_auc"] - incumbent_standalone["mean_ts_auc"]
        ),
        "rho_candidate_vs_incumbent": float(
            csa.within_t_rank_corr(predictions[cat_id], predictions[replaced], c.d.t, c.dev)
        ),
        "E1": {
            "mean_ts_auc": float(np.mean(e1_per)),
            "per_fold_ts_auc": [float(x) for x in e1_per],
            "score_pack": e1_pack,
        },
        "E2": {
            "mean_ts_auc": float(np.mean(e2_per)),
            "per_fold_ts_auc": [float(x) for x in e2_per],
            "score_pack": e2_pack,
        },
        "delta_E1_minus_E0": float(np.mean(e1_per) - np.mean(e0_per)),
        "delta_E1_minus_E0_per_fold": control_per,
        "delta_E2_minus_E1": float(np.mean(e2_per) - np.mean(e1_per)),
        "delta_E2_minus_E1_per_fold": primary_per,
        "positive_folds_E2_minus_E1": int(sum(x > 0.0 for x in primary_per)),
        "delta_E2_minus_E0": float(np.mean(e2_per) - np.mean(e0_per)),
        "delta_E2_minus_E0_per_fold": secondary_per,
        "positive_folds_E2_minus_E0": int(sum(x > 0.0 for x in secondary_per)),
        "score_pack_delta_vs_E1": diff_pack(e2_pack, e1_pack),
        "score_pack_delta_vs_E0": diff_pack(e2_pack, e0_pack),
        "pair_flow_vs_E1": csa.pair_flows(e1_vec, e2_vec, c),
        "pair_flow_vs_E0": csa.pair_flows(e0_vec, e2_vec, c),
    }
    scores = {
        f"E1_{name}": e1_vec,
        f"E2_{name}": e2_vec,
    }
    return rec, scores


def to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [to_jsonable(v) for v in x.tolist()]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def write_report(result: dict[str, Any]) -> None:
    rt600_check = result["checks"]["RT600"]
    rt1257_check = result["checks"]["RT1257"]
    lines = [
        "# RT-1257 Residual Slot Adjudication",
        "",
        f"Date: {RUN_DATE}",
        "Branch: `research/rt1257-slot-adjudication`",
        f"Git SHA: `{result['git_sha']}`",
        "",
        "This is a zero-training adjudication of the remaining CSA-04 CatBoost "
        "slot candidates after",
        "`RT-1257` has already installed `CAT-300` (`RT-1255`) and `CAT-413` (`RT-1254`).",
        "It reads frozen OOF vectors only; no lockbox/test rows are filled or evaluated.",
        "",
        "## Arms",
        "",
        "- `E0`: RT-1257 fixed champion composition.",
        "- `E1`: RT-1257 with the target residual slot replaced by matched seed clone `RT-401`.",
        "- `E2`: RT-1257 with the target residual slot replaced by the CatBoost candidate.",
        "",
        "Primary endpoint: `E2-E1`. Secondary deployment endpoint: `E2-E0`.",
        f"Continuation requires both endpoints to clear `max(SE, {NOISE_FLOOR:.4f})`, "
        "at least 4/5 positive folds",
        "against RT-1257, and positive dominant-cell plus mature-vs-never pair-flow nets.",
        "",
        "## Harness Checks",
        "",
        f"- RT-600 mean TS-AUC: {fmt(rt600_check['observed'])} "
        f"(expected {fmt(rt600_check['expected'])}; passed={rt600_check['passed']})",
        f"- RT-1257 mean TS-AUC: {fmt(rt1257_check['observed'])} "
        f"(expected {fmt(rt1257_check['expected'])}; passed={rt1257_check['passed']})",
        "",
        "## Summary",
        "",
        "| priority | candidate | slot | primary E2-E1 | secondary E2-E0 | "
        "folds E2>E0 | boot 95% CI E2-E0 | dominant net | mature-never net | "
        "recommendation |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rec in result["candidates"]:
        sboot = rec["bootstrap"]["E2_minus_E0"]
        ci = sboot["ci95_percentile"]
        dom = rec["pair_flow_vs_E0"]["dominant_cell"]["net_pair_lift"]
        mn = rec["pair_flow_vs_E0"]["mature_vs_never"]["net_pair_lift"]
        lines.append(
            f"| {rec['priority']} | {rec['ticket']} | {rec['replaced']} | "
            f"{fmt(rec['delta_E2_minus_E1'], signed=True)} | "
            f"{fmt(rec['delta_E2_minus_E0'], signed=True)} | "
            f"{rec['positive_folds_E2_minus_E0']}/5 | "
            f"[{fmt(ci[0], signed=True)}, {fmt(ci[1], signed=True)}] | "
            f"{dom} | {mn} | {rec['recommendation']} |"
        )

    lines += [
        "",
        "## Details",
        "",
    ]
    for rec in result["candidates"]:
        pboot = rec["bootstrap"]["E2_minus_E1"]
        sboot = rec["bootstrap"]["E2_minus_E0"]
        cboot = rec["bootstrap"]["E1_minus_E0"]
        standalone_delta = fmt(rec["standalone_delta_vs_incumbent"], signed=True)
        rho = rec["rho_candidate_vs_incumbent"]
        ctrl_lo, ctrl_hi = cboot["ci95_percentile"]
        primary_lo, primary_hi = pboot["ci95_percentile"]
        secondary_lo, secondary_hi = sboot["ci95_percentile"]
        pf = rec["pair_flow_vs_E0"]
        dom_net = pf["dominant_cell"]["net_pair_lift"]
        mn_net = pf["mature_vs_never"]["net_pair_lift"]
        mp_net = pf["mature_vs_prebreak"]["net_pair_lift"]
        lines += [
            f"### {rec['ticket']}",
            "",
            f"- Slot: `{rec['replaced']}` replaced by `{rec['id']}`.",
            f"- Standalone delta vs incumbent: {standalone_delta}; "
            f"within-t rho vs incumbent: {rho:.6f}.",
            f"- Control movement (`E1-E0`): {fmt(rec['delta_E1_minus_E0'], signed=True)} "
            f"(bootstrap SE {fmt(cboot['se'])}, 95% CI [{fmt(ctrl_lo, signed=True)}, "
            f"{fmt(ctrl_hi, signed=True)}]).",
            f"- Primary (`E2-E1`): {fmt(rec['delta_E2_minus_E1'], signed=True)} "
            f"({rec['positive_folds_E2_minus_E1']}/5 folds; bootstrap SE {fmt(pboot['se'])}, "
            f"95% CI [{fmt(primary_lo, signed=True)}, "
            f"{fmt(primary_hi, signed=True)}]).",
            f"- Secondary (`E2-E0`): {fmt(rec['delta_E2_minus_E0'], signed=True)} "
            f"({rec['positive_folds_E2_minus_E0']}/5 folds; bootstrap SE {fmt(sboot['se'])}, "
            f"95% CI [{fmt(secondary_lo, signed=True)}, "
            f"{fmt(secondary_hi, signed=True)}]).",
            f"- Pair flow vs RT-1257: dominant net {dom_net}; "
            f"mature-vs-never net {mn_net}; mature-vs-prebreak net {mp_net}.",
            "",
            "| fold | E1-E0 | E2-E1 | E2-E0 |",
            "|---:|---:|---:|---:|",
        ]
        for f, ctrl, primary, secondary in zip(
            FOLDS,
            rec["delta_E1_minus_E0_per_fold"],
            rec["delta_E2_minus_E1_per_fold"],
            rec["delta_E2_minus_E0_per_fold"],
        ):
            lines.append(
                f"| {f} | {fmt(ctrl, signed=True)} | {fmt(primary, signed=True)} | "
                f"{fmt(secondary, signed=True)} |"
            )
        lines.append("")

    lines += [
        "## Interpretation",
        "",
        "None of the residual slots earns promotion from this champion-relative check unless "
        "its secondary",
        "`E2-E0` clears the measured uncertainty/noise bar. Positive standalone or "
        "clone-relative movement is",
        "treated as insufficient when the deployment composition does not improve RT-1257 "
        "by a distinguishable amount.",
        "",
        "Recommended next work:",
        "",
        "1. Do not open new training lanes for CAT-411, CAT-414, or CAT-415 from these "
        "frozen OOF results.",
        "2. Keep CAT-412 as the only residual slot worth a narrow follow-up, and only if "
        "a preregistered alt-partition",
        "   adjudication is desired despite the sub-noise RT-1257 lift.",
        "3. If CAT-412 is followed up, run alt partitions with the same RT-1257-relative "
        "`E0/E1/E2` contract and",
        "   require a deployment endpoint above the noise floor before changing the champion.",
        "",
        "## Repro",
        "",
        "```bash",
        "/home/user/anaconda3/bin/python research/scripts/rt1257_slot_adjudication.py "
        "--artifact-root ../structural-break-deep-ensemble-frontier-local-2026 "
        "--bootstrap-reps 2000",
        "```",
        "",
    ]
    RESULT_MD.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    if args.bootstrap_reps <= 1:
        raise SystemExit("--bootstrap-reps must be greater than 1")

    csa = local.import_csa(artifact_root)
    needed = unique(
        list(local.SPECIALISTS)
        + [local.MATCHED_CLONE]
        + list(RT1257_STREAMS)
        + [rec["id"] for rec in CANDIDATES.values()]
    )
    copy_records = ensure_local_oof_inputs(artifact_root, local.LOCAL_OOF_DIR, needed)
    c = csa.make_ctx()
    validation = validate_oof_inputs(c, local.LOCAL_OOF_DIR, needed)
    predictions = csa.load_oof(needed)
    cache = csa.CalibratedFoldCache(c, predictions)

    rt600_vec, rt600_per = csa.blend_fixed(c, cache, list(local.SPECIALISTS))
    e0_vec, e0_per = csa.blend_fixed(c, cache, list(RT1257_STREAMS))
    e0_pack = score_pack(csa, c, e0_vec)
    checks = {
        "RT600": compare_value("RT600.mean_ts_auc", float(np.mean(rt600_per)), EXPECTED_RT600),
        "RT1257": compare_value("RT1257.mean_ts_auc", float(np.mean(e0_per)), EXPECTED_RT1257),
    }
    failed = [v for v in checks.values() if not v["passed"]]
    if failed:
        raise SystemExit(f"harness checks failed: {failed}")

    candidate_records: list[dict[str, Any]] = []
    scores_for_bootstrap: dict[str, np.ndarray] = {"E0_RT1257": e0_vec}
    for name in sorted(CANDIDATES, key=lambda n: CANDIDATES[n]["priority"]):
        rec, scores = build_candidate_record(
            csa=csa,
            c=c,
            predictions=predictions,
            cache=cache,
            e0_vec=e0_vec,
            e0_per=e0_per,
            e0_pack=e0_pack,
            name=name,
        )
        candidate_records.append(rec)
        scores_for_bootstrap.update(scores)

    print(
        f"running paired series bootstrap: labels={len(scores_for_bootstrap)} "
        f"reps={args.bootstrap_reps} seed={args.bootstrap_seed}",
        flush=True,
    )
    boot_auc, boot_meta = bootlib.run_bootstrap(
        c, scores_for_bootstrap, args.bootstrap_reps, args.bootstrap_seed
    )

    for rec in candidate_records:
        e1_key = f"E1_{rec['name']}"
        e2_key = f"E2_{rec['name']}"
        rec["bootstrap"] = {
            "E1_minus_E0": bootstrap_summary(boot_auc[e1_key] - boot_auc["E0_RT1257"]),
            "E2_minus_E1": bootstrap_summary(boot_auc[e2_key] - boot_auc[e1_key]),
            "E2_minus_E0": bootstrap_summary(boot_auc[e2_key] - boot_auc["E0_RT1257"]),
        }
        rec["decision_bars"] = {
            "E2_minus_E1": float(max(NOISE_FLOOR, rec["bootstrap"]["E2_minus_E1"]["se"])),
            "E2_minus_E0": float(max(NOISE_FLOOR, rec["bootstrap"]["E2_minus_E0"]["se"])),
        }
        rec["recommendation"] = recommendation(rec)

    result = {
        "date": RUN_DATE,
        "git_sha": git_sha(short=False),
        "artifact_root": str(artifact_root),
        "protocol": {
            "name": "rt1257_slot_adjudication",
            "training": "none; frozen OOF artifact re-score only",
            "E0": list(RT1257_STREAMS),
            "E1": f"RT-1257 with each target residual slot replaced by {local.MATCHED_CLONE}",
            "E2": "RT-1257 with each target residual slot replaced by the CatBoost candidate",
            "primary_endpoint": "E2_minus_E1",
            "secondary_endpoint": "E2_minus_E0",
            "noise_floor": NOISE_FLOOR,
            "continuation_rule": (
                "Both endpoints must clear max(bootstrap SE, noise floor), with >=4/5 "
                "positive folds "
                "against RT-1257 and positive dominant-cell plus mature-vs-never pair-flow nets."
            ),
        },
        "checks": checks,
        "bootstrap": boot_meta,
        "copy_records": copy_records,
        "validation": validation,
        "E0_RT1257": {
            "mean_ts_auc": float(np.mean(e0_per)),
            "per_fold_ts_auc": [float(x) for x in e0_per],
            "score_pack": e0_pack,
        },
        "RT600": {
            "mean_ts_auc": float(np.mean(rt600_per)),
            "per_fold_ts_auc": [float(x) for x in rt600_per],
        },
        "candidates": candidate_records,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(to_jsonable(result), indent=2) + "\n")
    write_report(to_jsonable(result))
    print(f"wrote {RESULT_JSON}", flush=True)
    print(f"wrote {RESULT_MD}", flush=True)


if __name__ == "__main__":
    main()
