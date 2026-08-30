#!/usr/bin/env python3
"""CSA-04R reanalysis over existing CSA-04 OOF vectors.

No model is trained here. The script recomputes the CSA-04 harness values,
builds the preregistered E2-E0 greedy curve, runs a paired series bootstrap,
and writes the CSA-04R report artifacts.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO / "research" / "reports" / "deep_ensemble_frontier_2026" / "local"
PREREG = REPO / "research" / "reports" / "deep_ensemble_frontier_2026" / "CSA04R_REANALYSIS_PREREG.md"
CSA04_JSON = REPORT_DIR / "CSA04_RESULTS.json"
CSA04_FINAL = REPORT_DIR / "CSA04_FINAL.md"
RESULT_JSON = REPORT_DIR / "CSA04R_REANALYSIS.json"
RESULT_MD = REPORT_DIR / "CSA04R_REANALYSIS.md"
RESULTS_CSV = REPO / "research" / "RESULTS.csv"
RDOF_LEDGER = REPO / "research" / "RDOF_LEDGER.md"
ID_MAP = REPO / "research" / "EXPERIMENT_ID_MAP.md"

for _p in (REPO / "src", REPO / "research" / "scripts"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from sbr.metric import ts_auc_flat  # noqa: E402
import deep_ensemble_local_2026 as local  # noqa: E402

FOLDS = (0, 1, 2, 3, 4)
FIXED_ORDER = ("CAT-413", "CAT-300", "CAT-412", "CAT-415", "CAT-414", "CAT-411")
RT1264_COMPOSITION = ("CAT-412", "CAT-414", "CAT-411", "CAT-413", "CAT-300")
RT1257_COMPOSITION = ("CAT-413", "CAT-300")
EXPECTED_RT1257_MARGIN = 0.002407204670070273
EXPECTED_RT1257_E2_MINUS_E0 = 0.002026321728670382
NOISE_FLOOR = 0.0011
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260828
TOL = 1e-9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-root", default=os.environ.get("SBR_ARTIFACT_ROOT", str(local.DEFAULT_ARTIFACT_ROOT)))
    p.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    p.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    p.add_argument("--no-ledger", action="store_true", help="Write report/json only; do not append ledgers.")
    return p.parse_args()


def git_sha(short: bool = True) -> str:
    args = ["git", "-C", str(REPO), "rev-parse", "--short" if short else "HEAD", "HEAD"]
    return subprocess.check_output(args, text=True).strip()


def as_float(x: Any) -> float:
    return float(x)


def fmt(x: float | None, digits: int = 9, signed: bool = False) -> str:
    if x is None:
        return ""
    flag = "+" if signed else ""
    return f"{float(x):{flag}.{digits}f}"


def unique(items: list[str] | tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(items))


def cat_and_clone_streams(names: tuple[str, ...] | list[str]) -> tuple[list[str], list[str]]:
    cat_by_replaced = {local.SPECS[n]["replaced"]: local.IDS[n] for n in names}
    clone_by_replaced: dict[str, str] = {}
    clone_iter = iter(local.SEED_CLONES)
    for s in local.SPECIALISTS:
        if s in cat_by_replaced:
            clone_by_replaced[s] = next(clone_iter)
    cat_streams = [cat_by_replaced.get(s, s) for s in local.SPECIALISTS]
    clone_streams = [clone_by_replaced.get(s, s) for s in local.SPECIALISTS]
    return cat_streams, clone_streams


def score_per_fold(c: Any, vec: np.ndarray) -> list[float]:
    return [
        float(ts_auc_flat(vec[c.rows[f]], c.d.y[c.rows[f]], c.d.t[c.rows[f]]))
        for f in FOLDS
    ]


def load_inputs(csa: Any) -> dict[str, np.ndarray]:
    names = unique(
        list(local.SPECIALISTS)
        + list(local.SEED_CLONES)
        + [local.IDS[n] for n in local.ALL_SINGLE_ORDER]
    )
    return csa.load_oof(names)


def validate_oof_inputs(c: Any, names: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    expected_dev_rows = int(len(c.dev))
    for name in names:
        path = local.LOCAL_OOF_DIR / f"{name}.npy"
        if not path.exists():
            raise SystemExit(f"missing local OOF {path}")
        arr = np.load(path, mmap_mode="r")
        rec = {
            "path": str(path),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "bytes": int(path.stat().st_size),
            "finite_dev_rows": int(np.isfinite(arr[c.dev]).sum()),
        }
        if arr.shape != (len(c.d.y),):
            raise SystemExit(f"{name} wrong shape: {rec}")
        if str(arr.dtype) != "float32":
            raise SystemExit(f"{name} wrong dtype: {rec}")
        if rec["finite_dev_rows"] != expected_dev_rows:
            raise SystemExit(f"{name} wrong dev finite count: {rec}")
        out[name] = rec
    return out


class BlendBook:
    def __init__(self, csa: Any, c: Any, predictions: dict[str, np.ndarray]):
        self.csa = csa
        self.c = c
        self.cache = csa.CalibratedFoldCache(c, predictions)
        self._blend_cache: dict[tuple[str, ...], tuple[np.ndarray, list[float]]] = {}
        self.e0_vec, self.e0_per = self.blend(tuple(local.SPECIALISTS))

    def blend(self, streams: tuple[str, ...] | list[str]) -> tuple[np.ndarray, list[float]]:
        key = tuple(streams)
        if key not in self._blend_cache:
            self._blend_cache[key] = self.csa.blend_fixed(self.c, self.cache, list(key))
        return self._blend_cache[key]

    def hybrid(self, names: tuple[str, ...] | list[str], pairflow: bool = True) -> dict[str, Any]:
        names_t = tuple(names)
        cat_streams, clone_streams = cat_and_clone_streams(names_t)
        e1_vec, e1_per = self.blend(tuple(clone_streams))
        e2_vec, e2_per = self.blend(tuple(cat_streams))
        e2_minus_e0_per_fold = [float(a - b) for a, b in zip(e2_per, self.e0_per)]
        fold_deltas_vs_clone = [float(a - b) for a, b in zip(e2_per, e1_per)]
        rec: dict[str, Any] = {
            "k": int(len(names_t)),
            "survivors_ordered": list(names_t),
            "cat_streams": cat_streams,
            "clone_streams": clone_streams,
            "E0": float(np.mean(self.e0_per)),
            "E1": float(np.mean(e1_per)),
            "E2": float(np.mean(e2_per)),
            "E0_per_fold": [float(x) for x in self.e0_per],
            "E1_per_fold": [float(x) for x in e1_per],
            "E2_per_fold": [float(x) for x in e2_per],
            "marginal_vs_clone": float(np.mean(e2_per) - np.mean(e1_per)),
            "E2_minus_E0": float(np.mean(e2_per) - np.mean(self.e0_per)),
            "E2_minus_E0_per_fold": e2_minus_e0_per_fold,
            "positive_folds_vs_E0": int(sum(x > 0 for x in e2_minus_e0_per_fold)),
            "fold_deltas_vs_clone": fold_deltas_vs_clone,
            "positive_folds_vs_clone": int(sum(x > 0 for x in fold_deltas_vs_clone)),
        }
        if pairflow:
            rec["pair_flow_vs_E0"] = self.csa.pair_flows(self.e0_vec, e2_vec, self.c)
        return rec


def compare_value(label: str, observed: float, expected: float) -> dict[str, Any]:
    diff = float(observed - expected)
    return {
        "label": label,
        "observed": float(observed),
        "expected": float(expected),
        "diff": diff,
        "passed": bool(abs(diff) <= TOL),
    }


def run_harness_checks(book: BlendBook, committed: dict[str, Any]) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []

    for name in local.ALL_SINGLE_ORDER:
        observed = book.hybrid((name,), pairflow=False)
        expected = committed["specialists"][name]["ensemble"]
        comparisons.append(compare_value(f"{name}.marginal_vs_clone", observed["marginal_vs_clone"], expected["marginal_vs_clone"]))
        comparisons.append(compare_value(f"{name}.E2_minus_E0", observed["E2_minus_E0"], expected["E2_minus_E0"]))

    for row in committed["hybrid_curve"]:
        observed = book.hybrid(tuple(row["survivors_ordered"]), pairflow=False)
        expected = row["ensemble"]
        label = f"HYBRID-k{row['k']}:{','.join(row['survivors_ordered'])}"
        comparisons.append(compare_value(f"{label}.marginal_vs_clone", observed["marginal_vs_clone"], expected["marginal_vs_clone"]))
        comparisons.append(compare_value(f"{label}.E2_minus_E0", observed["E2_minus_E0"], expected["E2_minus_E0"]))

    failed = [r for r in comparisons if not r["passed"]]
    return {
        "source": str(CSA04_JSON),
        "tolerance": TOL,
        "comparisons": comparisons,
        "failed": failed,
        "passed": not failed,
    }


def fold_series_indices(c: Any) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for f in FOLDS:
        rows = c.rows[f]
        series = np.unique(c.d.sidx[rows])
        local_idx = np.searchsorted(series, c.d.sidx[rows]).astype(np.int32)
        out[int(f)] = (series, local_idx)
    return out


def bootstrap_counts(c: Any, seed: int, reps: int) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    out: dict[int, np.ndarray] = {}
    for f in FOLDS:
        rows = c.rows[f]
        n_series = int(len(np.unique(c.d.sidx[rows])))
        counts = np.zeros((reps, n_series), dtype=np.float64)
        draws = rng.integers(0, n_series, size=(reps, n_series), dtype=np.int32)
        for b in range(reps):
            counts[b] = np.bincount(draws[b], minlength=n_series)
        out[int(f)] = counts
    return out


def weighted_ts_auc_boot_fold(
    c: Any,
    fold: int,
    scores: np.ndarray,
    counts: np.ndarray,
    local_series_by_fold: dict[int, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    rows = c.rows[fold]
    _, row_series_local = local_series_by_fold[int(fold)]
    labels = c.d.y[rows].astype(np.int8)
    t_index = c.d.t[rows].astype(np.int32)
    score = scores[rows].astype(np.float64)
    if not np.isfinite(score).all():
        raise ValueError(f"scores for fold {fold} contain non-finite values")

    order = np.lexsort((score, t_index))
    row_series = row_series_local[order]
    y = labels[order]
    t = t_index[order]
    s = score[order]
    gstart = np.flatnonzero(np.r_[True, t[1:] != t[:-1]])
    gend = np.r_[gstart[1:], len(t)]

    reps = counts.shape[0]
    num = np.zeros(reps, dtype=np.float64)
    den = np.zeros(reps, dtype=np.float64)
    for lo, hi in zip(gstart, gend):
        rs = row_series[lo:hi]
        lab = y[lo:hi].astype(np.float64)
        w = counts[:, rs]
        pos = w * lab
        neg = w * (1.0 - lab)
        total_pos = pos.sum(axis=1)
        total_neg = neg.sum(axis=1)
        den += total_pos * total_neg

        sg = s[lo:hi]
        tie_starts = np.flatnonzero(np.r_[True, sg[1:] != sg[:-1]])
        if len(tie_starts) == hi - lo:
            neg_before = np.cumsum(neg, axis=1) - neg
            num += (pos * neg_before).sum(axis=1)
        else:
            pos_run = np.add.reduceat(pos, tie_starts, axis=1)
            neg_run = np.add.reduceat(neg, tie_starts, axis=1)
            neg_before_run = np.cumsum(neg_run, axis=1) - neg_run
            num += (pos_run * (neg_before_run + 0.5 * neg_run)).sum(axis=1)

    out = np.full(reps, 0.5, dtype=np.float64)
    valid = den > 0.0
    out[valid] = num[valid] / den[valid]
    return out


def exact_weighted_auc_checks(
    c: Any,
    scores_by_label: dict[str, np.ndarray],
    local_series_by_fold: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    checks: dict[str, float] = {}
    for label, scores in scores_by_label.items():
        max_abs = 0.0
        for f in FOLDS:
            n_series = len(local_series_by_fold[int(f)][0])
            counts = np.ones((1, n_series), dtype=np.float64)
            weighted = float(weighted_ts_auc_boot_fold(c, int(f), scores, counts, local_series_by_fold)[0])
            direct = float(ts_auc_flat(scores[c.rows[f]], c.d.y[c.rows[f]], c.d.t[c.rows[f]]))
            max_abs = max(max_abs, abs(weighted - direct))
        checks[label] = float(max_abs)
    return checks


def run_bootstrap(
    c: Any,
    scores_by_label: dict[str, np.ndarray],
    reps: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    local_series_by_fold = fold_series_indices(c)
    counts_by_fold = bootstrap_counts(c, seed, reps)
    exact_checks = exact_weighted_auc_checks(c, scores_by_label, local_series_by_fold)
    boot_auc = {label: np.zeros(reps, dtype=np.float64) for label in scores_by_label}
    for label, scores in scores_by_label.items():
        print(f"bootstrap weighted TS-AUC: {label}", flush=True)
        for f in FOLDS:
            boot_auc[label] += weighted_ts_auc_boot_fold(c, int(f), scores, counts_by_fold[int(f)], local_series_by_fold) / len(FOLDS)
    meta = {
        "method": "paired series bootstrap with exact weighted TS-AUC recomputation under multinomial series counts",
        "reps": int(reps),
        "seed": int(seed),
        "series_resampled_within_fold": True,
        "weighted_auc_exact_check_max_abs_by_label": exact_checks,
    }
    return boot_auc, meta


def summarize_samples(samples: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(samples)),
        "se": float(np.std(samples, ddof=1)),
        "ci95_percentile": [float(x) for x in np.percentile(samples, [2.5, 97.5])],
    }


def add_rt1257_contrasts(curve: list[dict[str, Any]], rt1257: dict[str, Any]) -> None:
    for row in curve:
        row["delta_vs_RT1257"] = float(row["E2_minus_E0"] - rt1257["E2_minus_E0"])
        row["delta_vs_RT1257_per_fold"] = [
            float(a - b) for a, b in zip(row["E2_per_fold"], rt1257["E2_per_fold"])
        ]
        row["positive_folds_vs_RT1257"] = int(sum(x > 0 for x in row["delta_vs_RT1257_per_fold"]))


def subset_appendix(book: BlendBook) -> dict[str, Any]:
    rows = []
    for size in range(1, len(FIXED_ORDER) + 1):
        for subset in itertools.combinations(FIXED_ORDER, size):
            rec = book.hybrid(subset, pairflow=False)
            rows.append({
                "subset": list(subset),
                "k": int(size),
                "cat_streams": rec["cat_streams"],
                "E2": rec["E2"],
                "E2_minus_E0": rec["E2_minus_E0"],
                "E2_minus_E0_per_fold": rec["E2_minus_E0_per_fold"],
                "non_selecting": True,
            })
    best = max(rows, key=lambda r: r["E2_minus_E0"])
    return {
        "non_selecting": True,
        "warning": "Descriptive only; barred by preregistration from selecting k* or supporting promotion.",
        "rows": rows,
        "max_subset": best,
    }


def selection_and_verdict(
    curve: list[dict[str, Any]],
    boot_deltas: dict[int, np.ndarray],
    rt1257: dict[str, Any],
) -> dict[str, Any]:
    max_row = max(curve, key=lambda r: r["E2_minus_E0"])
    max_k = int(max_row["k"])
    reference_k = 2
    selection_contrast = boot_deltas[max_k] - boot_deltas[reference_k]
    selection_se = float(np.std(selection_contrast, ddof=1))
    delta_noise = float(max(selection_se, NOISE_FLOOR))
    m = float(max_row["E2_minus_E0"])
    tied = [int(r["k"]) for r in curve if m - float(r["E2_minus_E0"]) < delta_noise]
    kstar = min(tied)
    selected = next(r for r in curve if int(r["k"]) == kstar)
    diff_vs_rt1257 = float(selected["E2_minus_E0"] - rt1257["E2_minus_E0"])
    dom_net = int(selected["pair_flow_vs_E0"]["dominant_cell"]["net_pair_lift"])
    mn_net = int(selected["pair_flow_vs_E0"]["mature_vs_never"]["net_pair_lift"])
    if diff_vs_rt1257 < 0:
        verdict = "INFERIOR"
    elif diff_vs_rt1257 >= delta_noise and selected["positive_folds_vs_RT1257"] >= 4 and dom_net > 0 and mn_net > 0:
        verdict = "SUPERSEDES_RT1257"
    else:
        verdict = "NOT_DISTINGUISHABLE"
    return {
        "max_k": max_k,
        "M_E2_minus_E0": m,
        "selection_contrast": f"k{max_k}_minus_k{reference_k}_RT1257",
        "selection_contrast_bootstrap": summarize_samples(selection_contrast),
        "delta_noise": delta_noise,
        "noise_floor": NOISE_FLOOR,
        "K_tied": tied,
        "kstar": kstar,
        "selected": selected,
        "diff_vs_RT1257": diff_vs_rt1257,
        "dominant_cell_net_vs_E0": dom_net,
        "mature_vs_never_net_vs_E0": mn_net,
        "verdict": verdict,
    }


def prediction_adjudication(selection: dict[str, Any], curve: list[dict[str, Any]], rt1257: dict[str, Any]) -> dict[str, Any]:
    k2 = next(r for r in curve if r["k"] == 2)
    csa04_k5 = tuple(RT1264_COMPOSITION)
    selected_comp = tuple(selection["selected"]["survivors_ordered"])
    p4_ok = abs(k2["E2_minus_E0"] - EXPECTED_RT1257_E2_MINUS_E0) <= TOL
    p5_ok = selection["kstar"] <= 2 and selection["verdict"] == "NOT_DISTINGUISHABLE"
    p6_ok = selected_comp != csa04_k5
    return {
        "P4": {
            "prediction": "The reordered curve starts at RT-1257; k=2 is CAT-413+CAT-300 and E2-E0 is +0.002026322.",
            "observed": k2["E2_minus_E0"],
            "passed": bool(p4_ok),
        },
        "P5": {
            "prediction": "k* will be small and the verdict will be NOT_DISTINGUISHABLE.",
            "observed": f"k*={selection['kstar']}; verdict={selection['verdict']}",
            "passed": bool(p5_ok),
        },
        "P6": {
            "prediction": "CSA-04's k=5 composition will not survive.",
            "observed": f"selected={','.join(selected_comp)}; CSA04_k5={','.join(csa04_k5)}",
            "passed": bool(p6_ok),
        },
    }


def bootstrap_decision_summaries(
    curve: list[dict[str, Any]],
    boot_auc: dict[str, np.ndarray],
    selection: dict[str, Any],
    rt1257: dict[str, Any],
) -> dict[str, Any]:
    boot_deltas = {int(r["k"]): boot_auc[f"k{r['k']}"] - boot_auc["E0"] for r in curve}
    per_k = {
        f"k{k}": summarize_samples(samples)
        for k, samples in sorted(boot_deltas.items())
    }
    kstar = int(selection["kstar"])
    kstar_vs_rt1257 = boot_deltas[kstar] - boot_deltas[2]
    max_k = int(selection["max_k"])
    adjacent = {}
    if max_k > 1:
        adjacent[f"k{max_k}_minus_k{max_k - 1}"] = summarize_samples(boot_deltas[max_k] - boot_deltas[max_k - 1])
    if max_k < len(curve):
        adjacent[f"k{max_k}_minus_k{max_k + 1}"] = summarize_samples(boot_deltas[max_k] - boot_deltas[max_k + 1])
    return {
        "per_k_E2_minus_E0": per_k,
        "kstar_vs_RT1257": summarize_samples(kstar_vs_rt1257),
        "adjacent_around_max": adjacent,
    }


def results_row(result: dict[str, Any]) -> dict[str, Any]:
    selected = result["selection"]["selected"]
    boot = result["bootstrap"]["decision_summaries"]["kstar_vs_RT1257"]
    return {
        "experiment_id": "RT-1265",
        "date": "2026-08-28",
        "git_sha": result["git_sha"],
        "agent": "codex-deep-ensemble-local",
        "hypothesis": "CSA-04R reanalysis selects hybrid size by fixed E2-E0 endpoint instead of E2-E1 control-collapse endpoint.",
        "falsification_condition": "SUPERSEDES_RT1257 only if selected E2-E0 exceeds RT-1257 by delta_noise with >=4/5 folds positive vs RT-1257 and positive dominant-cell and mature-vs-never pair nets.",
        "feature_set": ",".join(selected["cat_streams"]),
        "n_features": "7",
        "model": "frozen_oof_equal_scdf_hybrid_reanalysis",
        "objective": "artifact_rescore",
        "folds": "0,1,2,3,4",
        "random_seed": str(BOOTSTRAP_SEED),
        "train_series": "8000",
        "train_rows": "",
        "mean_oof_ts_auc": repr(selected["E2"]),
        "pooled_oof_ts_auc": "",
        "per_fold_ts_auc": ";".join(f"{x:.5f}" for x in selected["E2_per_fold"]),
        "fold_std": repr(float(np.std(selected["E2_per_fold"]))),
        "persistence": "none",
        "sample_mode": "frozen_oof",
        "training_runtime_s": "0.0",
        "causal_verified": "existing causal OOF vectors only; fold-pure SCDF_NSEEN; no training",
        "test_reduced_touched": "no",
        "lockbox_touched": "no",
        "status": result["selection"]["verdict"],
        "notes": (
            f"CSA-04R selected k*={selected['k']} survivors={','.join(selected['survivors_ordered'])}; "
            f"identical composition to RT-1257; E0={selected['E0']:.9f}; E1={selected['E1']:.9f}; "
            f"E2={selected['E2']:.9f}; E2-E0={selected['E2_minus_E0']:+.9f}; "
            f"delta_vs_RT1257={selected['delta_vs_RT1257']:+.9f}; delta_noise={result['selection']['delta_noise']:.9f}; "
            f"kstar_vs_RT1257_bootstrap_se={boot['se']:.9f}; "
            f"dominant_net={selected['pair_flow_vs_E0']['dominant_cell']['net_pair_lift']}; "
            f"mature_vs_never_net={selected['pair_flow_vs_E0']['mature_vs_never']['net_pair_lift']}; "
            "No new OOF vector was written."
        ),
        "protocol": "deep_ensemble_frontier_2026_csa04r_reanalysis",
    }


def append_results_csv(row: dict[str, Any]) -> bool:
    with RESULTS_CSV.open("r+", newline="") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise RuntimeError("RESULTS.csv has no header")
        existing = {r["experiment_id"] for r in reader}
        if row["experiment_id"] in existing:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            return False
        fh.seek(0, os.SEEK_END)
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writerow({k: row.get(k, "") for k in fieldnames})
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return True


def update_id_map_if_needed() -> bool:
    marker = "## Deep Ensemble Frontier 2026 -- LOCAL Lane CSA-04R Reanalysis"
    text = ID_MAP.read_text()
    if marker in text:
        return False
    old = "`RT-1265` through `RT-1269` remain unallocated LOCAL-lane contingency IDs."
    new = "`RT-1266` through `RT-1269` remain unallocated LOCAL-lane contingency IDs."
    if old not in text:
        raise RuntimeError("could not find RT-1265 allocation sentence in EXPERIMENT_ID_MAP.md")
    section = """

## Deep Ensemble Frontier 2026 -- LOCAL Lane CSA-04R Reanalysis

Registered as a separate post-CSA-04 reanalysis on 2026-08-28 because the
CSA-04 `E2-E1` endpoint was discovered to inflate as `k` grew. This consumes no
new model and writes no OOF vector; it reorders the already-admitted CSA-04
survivors by fixed single-slot `E2-E0` and selects `k` by the deployment
endpoint `E2-E0`.

| ID | arm |
|---|---|
| `RT-1265` | CSA-04R selected `k*` hybrid under the fixed `E2-E0` endpoint. The selected composition is the two-slot CAT-413 + CAT-300 hybrid, identical to `RT-1257`; no new OOF vector is written. |
"""
    ID_MAP.write_text(text.replace(old, new + section))
    return True


def append_rdof(result: dict[str, Any]) -> bool:
    marker = "## Deep Ensemble Frontier 2026 -- LOCAL Lane CSA-04R Reanalysis"
    text = RDOF_LEDGER.read_text()
    if marker in text:
        return False
    selection = result["selection"]
    block = f"""

## Deep Ensemble Frontier 2026 -- LOCAL Lane CSA-04R Reanalysis  (PRE-REGISTERED 2026-08-28; EXECUTED 2026-08-28)

Execution preregistration:
`research/reports/deep_ensemble_frontier_2026/CSA04R_REANALYSIS_PREREG.md`.

**Scientific question.** CSA-04 selected hybrid size by `E2-E1`, but that control
collapses as `k` grows. CSA-04R asks which already-admitted CatBoost survivor
composition is selected when the curve is ordered and judged by fixed
deployment endpoint `E2-E0`.

| Item | Degrees of freedom | Frozen before CSA-04R execution |
|---|---:|---|
| endpoint change | 1 | `E2-E0` replaces `E2-E1` for selecting `k*`; `E1` retained only for continuity |
| ordering change | 1 | fixed order CAT-413, CAT-300, CAT-412, CAT-415, CAT-414, CAT-411 |
| per-slot admission | 0 | unchanged CSA-04 `marginal_vs_clone >= +0.0010` at `k=1`; CAT-410 not re-admitted |
| model training | 0 | no training, no retuning, no new OOF vector |
| bootstrap | 0 | series bootstrap seed `20260828`, B = 2000, fixed in preregistration |
| descriptive subsets | 63 non-selecting looks | all non-empty subsets enumerated and explicitly barred from selecting `k*` or supporting promotion |

Result: `k*={selection['kstar']}` (`{', '.join(selection['selected']['survivors_ordered'])}`),
`E2-E0={selection['selected']['E2_minus_E0']:+.9f}`,
`delta_noise={selection['delta_noise']:.9f}`, verdict `{selection['verdict']}`.
"""
    RDOF_LEDGER.write_text(text.rstrip() + block + "\n")
    return True


def append_csa04_pointer(result: dict[str, Any]) -> bool:
    marker = "## CSA-04R Reanalysis Pointer"
    text = CSA04_FINAL.read_text()
    if marker in text:
        return False
    selection = result["selection"]
    block = f"""

## CSA-04R Reanalysis Pointer

CSA-04R is filed in
`research/reports/deep_ensemble_frontier_2026/local/CSA04R_REANALYSIS.md`.
It does not rewrite any CSA-04 numbers or verdicts. It reanalyzes the same OOF
vectors under the fixed `E2-E0` endpoint, selects `k*={selection['kstar']}`
(`{', '.join(selection['selected']['survivors_ordered'])}`), and returns
verdict `{selection['verdict']}` with `delta_noise={selection['delta_noise']:.9f}`.
"""
    CSA04_FINAL.write_text(text.rstrip() + block + "\n")
    return True


def write_report(result: dict[str, Any]) -> None:
    selection = result["selection"]
    boot = result["bootstrap"]["decision_summaries"]
    lines = [
        "# CSA-04R Reanalysis -- Final",
        "",
        "Date: 2026-08-28",
        "Branch: `research/deep-ensemble-frontier-local-2026`",
        f"Analysis SHA: `{result['git_sha']}`",
        f"Preregistration: `{PREREG.relative_to(REPO)}`",
        "",
        "## Scope",
        "",
        "This is a re-analysis of existing CSA-04 OOF vectors. It trained no model, tuned no weight, wrote no OOF vector, and did not use lockbox or test data.",
        "",
        "## Harness Checks",
        "",
        f"CSA-04 exact reproduction: `{result['harness_checks']['passed']}` across `{len(result['harness_checks']['comparisons'])}` committed values at tolerance `{TOL}`.",
        f"RT-1257 marginal regression: observed `{result['rt1257_regression_check']['observed']:+.15f}`, expected `{EXPECTED_RT1257_MARGIN:+.15f}`, diff `{result['rt1257_regression_check']['diff']:+.3e}`, passed `{result['rt1257_regression_check']['passed']}`.",
        f"P4 k=2 E2-E0 regression: observed `{result['p4_regression_check']['observed']:+.15f}`, expected `{EXPECTED_RT1257_E2_MINUS_E0:+.15f}`, diff `{result['p4_regression_check']['diff']:+.3e}`, passed `{result['p4_regression_check']['passed']}`.",
        "",
        "## Fixed Ordering",
        "",
        "| rank | slot | single-slot E2-E0 | single-slot marginal_vs_clone |",
        "|---:|---|---:|---:|",
    ]
    for i, name in enumerate(FIXED_ORDER, 1):
        s = result["single_slot_recomputed"][name]
        lines.append(f"| {i} | `{name}` (`{local.IDS[name]}`) | {s['E2_minus_E0']:+.9f} | {s['marginal_vs_clone']:+.9f} |")

    lines += [
        "",
        "## Greedy Curve",
        "",
        "| k | composition | E0 | E1 | E2 | E2-E0 | fold deltas E2-E0 | folds + vs E0 | delta vs RT-1257 | folds + vs RT-1257 | whole net | dominant net | mature-never net | mature-prebreak net |",
        "|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["greedy_curve"]:
        pf = row["pair_flow_vs_E0"]
        lines.append(
            f"| {row['k']} | `{', '.join(row['survivors_ordered'])}` | "
            f"{row['E0']:.9f} | {row['E1']:.9f} | {row['E2']:.9f} | {row['E2_minus_E0']:+.9f} | "
            f"`{';'.join(fmt(x, 6, True) for x in row['E2_minus_E0_per_fold'])}` | "
            f"{row['positive_folds_vs_E0']}/5 | {row['delta_vs_RT1257']:+.9f} | "
            f"{row['positive_folds_vs_RT1257']}/5 | "
            f"{pf['whole']['net_pair_lift']} | {pf['dominant_cell']['net_pair_lift']} | "
            f"{pf['mature_vs_never']['net_pair_lift']} | {pf['mature_vs_prebreak']['net_pair_lift']} |"
        )

    lines += [
        "",
        "## Bootstrap",
        "",
        f"Bootstrap method: {result['bootstrap']['meta']['method']}. B = `{result['bootstrap']['meta']['reps']}`, seed = `{result['bootstrap']['meta']['seed']}`. Series were resampled within fold.",
        "",
        "| k | observed E2-E0 | bootstrap SE | 95% percentile CI |",
        "|---:|---:|---:|---:|",
    ]
    for row in result["greedy_curve"]:
        s = boot["per_k_E2_minus_E0"][f"k{row['k']}"]
        lo, hi = s["ci95_percentile"]
        lines.append(f"| {row['k']} | {row['E2_minus_E0']:+.9f} | {s['se']:.9f} | [{lo:+.9f}, {hi:+.9f}] |")

    kv = boot["kstar_vs_RT1257"]
    lo, hi = kv["ci95_percentile"]
    lines += [
        "",
        f"k* vs RT-1257 bootstrap: observed `{selection['diff_vs_RT1257']:+.9f}`, SE `{kv['se']:.9f}`, 95% CI `[{lo:+.9f}, {hi:+.9f}]`.",
        "",
        "| adjacent contrast around max | observed | bootstrap SE | 95% percentile CI |",
        "|---|---:|---:|---:|",
    ]
    for label, s in boot["adjacent_around_max"].items():
        left, right = label.replace("k", "").split("_minus_")
        observed = next(r for r in result["greedy_curve"] if r["k"] == int(left))["E2_minus_E0"] - next(r for r in result["greedy_curve"] if r["k"] == int(right))["E2_minus_E0"]
        lo, hi = s["ci95_percentile"]
        lines.append(f"| `{label}` | {observed:+.9f} | {s['se']:.9f} | [{lo:+.9f}, {hi:+.9f}] |")

    lines += [
        "",
        "## Selection And Verdict",
        "",
        f"Maximum M occurs at `k={selection['max_k']}` with `E2-E0={selection['M_E2_minus_E0']:+.9f}`.",
        f"Selection contrast `{selection['selection_contrast']}` bootstrap SE is `{selection['selection_contrast_bootstrap']['se']:.9f}`; with floor `{NOISE_FLOOR:.7f}`, `delta_noise={selection['delta_noise']:.9f}`.",
        f"`K_tied = {{{', '.join(str(k) for k in selection['K_tied'])}}}`; preregistered parsimony selects `k*={selection['kstar']}`.",
        f"Selected composition: `{', '.join(selection['selected']['survivors_ordered'])}`.",
        f"Verdict: `{selection['verdict']}`.",
        "",
    ]
    if selection["verdict"] == "NOT_DISTINGUISHABLE":
        lines.append("Plain reading: no CSA-04R multi-slot composition beats the two-slot RT-1257 by a distinguishable margin. CSA-04's durable result is the broad single-slot finding: six of seven specialist slots responded to CatBoost replacement at the preregistered k=1 gate.")
    elif selection["verdict"] == "INFERIOR":
        lines.append("Plain reading: the selected parsimonious composition is below RT-1257 on E2-E0, so CSA-04R does not support a promotion claim.")
    else:
        lines.append("Plain reading: the selected composition clears the preregistered superiority path against RT-1257.")

    lines += [
        "",
        "## Prediction Adjudication",
        "",
        "| prediction | passed | observed |",
        "|---|---:|---|",
    ]
    for name, p in result["prediction_adjudication"].items():
        lines.append(f"| `{name}` | `{p['passed']}` | {p['observed']} |")

    subset = result["subset_appendix"]
    max_subset = subset["max_subset"]
    greedy_kstar = selection["selected"]["E2_minus_E0"]
    lines += [
        "",
        "## Descriptive 63-Subset Appendix",
        "",
        "This appendix is non-selecting by preregistration. It cannot select k* or support a promotion claim.",
        f"Best descriptive subset: `{', '.join(max_subset['subset'])}` with `E2-E0={max_subset['E2_minus_E0']:+.9f}`; gap vs greedy k* is `{max_subset['E2_minus_E0'] - greedy_kstar:+.9f}`.",
        "",
        "| subset | k | E2-E0 |",
        "|---|---:|---:|",
    ]
    for row in subset["rows"]:
        lines.append(f"| `{', '.join(row['subset'])}` | {row['k']} | {row['E2_minus_E0']:+.9f} |")

    RESULT_MD.write_text("\n".join(lines) + "\n")


def write_json(result: dict[str, Any]) -> None:
    RESULT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    csa = local.import_csa(artifact_root)
    c = csa.make_ctx()
    predictions = load_inputs(csa)
    validation = validate_oof_inputs(c, list(predictions))
    book = BlendBook(csa, c, predictions)
    committed = json.loads(CSA04_JSON.read_text())

    harness = run_harness_checks(book, committed)
    if not harness["passed"]:
        raise SystemExit(f"CSA-04 reproduction failed: {harness['failed'][:3]}")

    rt1257 = book.hybrid(RT1257_COMPOSITION, pairflow=True)
    rt_check = compare_value("RT-1257.marginal_vs_clone", rt1257["marginal_vs_clone"], EXPECTED_RT1257_MARGIN)
    if not rt_check["passed"]:
        raise SystemExit(f"RT-1257 marginal regression failed: {rt_check}")

    curve = [book.hybrid(FIXED_ORDER[:k], pairflow=True) for k in range(1, len(FIXED_ORDER) + 1)]
    add_rt1257_contrasts(curve, rt1257)
    p4_check = compare_value("P4.k2.E2_minus_E0", curve[1]["E2_minus_E0"], EXPECTED_RT1257_E2_MINUS_E0)
    if not p4_check["passed"]:
        raise SystemExit(f"P4 k=2 E2-E0 regression failed: {p4_check}")

    single_recomputed = {
        name: {
            "marginal_vs_clone": book.hybrid((name,), pairflow=False)["marginal_vs_clone"],
            "E2_minus_E0": book.hybrid((name,), pairflow=False)["E2_minus_E0"],
        }
        for name in local.ALL_SINGLE_ORDER
    }

    scores_by_label = {"E0": book.e0_vec}
    for row in curve:
        cat_streams, _ = cat_and_clone_streams(tuple(row["survivors_ordered"]))
        scores_by_label[f"k{row['k']}"] = book.blend(tuple(cat_streams))[0]
    boot_auc, boot_meta = run_bootstrap(c, scores_by_label, int(args.bootstrap_reps), int(args.bootstrap_seed))
    boot_deltas = {int(row["k"]): boot_auc[f"k{row['k']}"] - boot_auc["E0"] for row in curve}

    selection = selection_and_verdict(curve, boot_deltas, rt1257)
    boot_decisions = bootstrap_decision_summaries(curve, boot_auc, selection, rt1257)
    subsets = subset_appendix(book)
    predictions_adjudicated = prediction_adjudication(selection, curve, rt1257)

    selected_comp = tuple(selection["selected"]["survivors_ordered"])
    consume_rt1265 = selected_comp != RT1264_COMPOSITION
    result = {
        "program": "CSA04R_REANALYSIS",
        "date": "2026-08-28",
        "git_sha": git_sha(),
        "artifact_root": str(artifact_root),
        "preregistration": str(PREREG),
        "no_training": True,
        "no_new_oof": True,
        "lockbox_touched": False,
        "test_reduced_touched": False,
        "validation": validation,
        "fixed_order": list(FIXED_ORDER),
        "rt1257_reference": rt1257,
        "harness_checks": harness,
        "rt1257_regression_check": rt_check,
        "p4_regression_check": p4_check,
        "single_slot_recomputed": single_recomputed,
        "greedy_curve": curve,
        "bootstrap": {
            "meta": boot_meta,
            "decision_summaries": boot_decisions,
        },
        "selection": selection,
        "verdict": selection["verdict"],
        "prediction_adjudication": predictions_adjudicated,
        "subset_appendix": subsets,
        "rt1265_consumed": bool(consume_rt1265),
        "results_csv_row_written": False,
        "id_map_updated": False,
        "rdof_ledger_updated": False,
        "csa04_final_pointer_appended": False,
        "status_updated": False,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_report(result)
    if not args.no_ledger:
        result["rdof_ledger_updated"] = append_rdof(result)
        result["csa04_final_pointer_appended"] = append_csa04_pointer(result)
        if consume_rt1265:
            result["results_csv_row_written"] = append_results_csv(results_row(result))
            result["id_map_updated"] = update_id_map_if_needed()
        write_report(result)
    write_json(result)

    print(json.dumps({
        "program": result["program"],
        "report": str(RESULT_MD),
        "json": str(RESULT_JSON),
        "kstar": selection["kstar"],
        "delta_noise": selection["delta_noise"],
        "verdict": selection["verdict"],
        "rt1265_consumed": consume_rt1265,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
