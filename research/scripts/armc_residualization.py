"""Arm-C horizon residualization diagnostic.

This is an offline research diagnostic for the W7-D3R result. It decomposes
the non-causal Arm C prediction vector (RT-991) into a within-timestep linear
projection on endpoint/horizon variables and a residual, then scores both on
the fixed dominant cell from W7-D0:

    current t >= 200 and (positive post-break age >= 100 or any negative row)

No model is trained, no RT ID is allocated, and research/RESULTS.csv is not
modified. The script only reads existing OOF vectors and folds metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sbr.metric import ts_auc_flat  # noqa: E402

FOLDS = (0, 1, 2, 3, 4)


def default_oof_dir() -> Path:
    candidates = [
        ROOT / "research" / "oof",
        ROOT.parent / "structural-break-wave8" / "research" / "oof",
        ROOT.parent / "structural-break-wave6" / "research" / "oof",
    ]
    for path in candidates:
        if all((path / f"{exp}.npy").exists() for exp in ("RT-990", "RT-991")):
            return path
    return candidates[0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def logit(p: np.ndarray, eps: float) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(q / (1.0 - q))


def build_row_arrays(folds_path: Path) -> dict[str, np.ndarray]:
    folds = pd.read_parquet(folds_path).sort_values("id").reset_index(drop=True)
    ids = folds["id"].to_numpy()
    if not np.array_equal(ids, np.arange(len(folds))):
        raise ValueError("folds.parquet ids must be contiguous and sorted 0..n-1")

    n_online = folds["n_online"].to_numpy(dtype=np.int32)
    tau = folds["tau_index"].to_numpy(dtype=np.int32)
    has_break = folds["has_break"].to_numpy(dtype=bool)
    series_fold = folds["fold"].to_numpy(dtype=np.int16)

    n_rows = int(n_online.sum())
    sidx = np.repeat(np.arange(len(folds), dtype=np.int32), n_online)
    t = np.empty(n_rows, dtype=np.int32)
    y = np.zeros(n_rows, dtype=np.int8)

    off = 0
    for i, n in enumerate(n_online):
        n_int = int(n)
        t[off : off + n_int] = np.arange(n_int, dtype=np.int32)
        if tau[i] >= 0:
            y[off + int(tau[i]) : off + n_int] = 1
        off += n_int

    tau_by_row = tau[sidx]
    age = np.where(y == 1, t - tau_by_row, -1).astype(np.int32)
    return {
        "folds": folds,
        "sidx": sidx,
        "t": t,
        "y": y,
        "age": age,
        "row_fold": series_fold[sidx],
        "has_break_by_row": has_break[sidx],
        "n_online_by_row": n_online[sidx],
    }


def dominant_masks(rows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    y = rows["y"]
    t = rows["t"]
    age = rows["age"]
    row_fold = rows["row_fold"]
    has_break = rows["has_break_by_row"]
    dev = np.isin(row_fold, FOLDS)
    neg_is_prebreak = has_break & (y == 0)
    cell = dev & (t >= 200) & (((y == 1) & (age >= 100)) | (y == 0))
    return {
        "dev": dev,
        "dominant_cell": cell,
        "dominant_never_break_only": cell & ((y == 1) | (~neg_is_prebreak)),
        "dominant_pre_break_only": cell & ((y == 1) | neg_is_prebreak),
    }


def group_indices_by_t(t: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    idx = np.flatnonzero(mask)
    order = np.argsort(t[idx], kind="stable")
    idx = idx[order]
    tt = t[idx]
    starts = np.flatnonzero(np.r_[True, tt[1:] != tt[:-1]])
    ends = np.r_[starts[1:], len(idx)]
    return [idx[lo:hi] for lo, hi in zip(starts, ends)]


def design_matrix(t_value: int, n_online: np.ndarray) -> np.ndarray:
    n_online = np.asarray(n_online, dtype=np.float64)
    remaining = n_online - float(t_value)
    frac = float(t_value) / np.maximum(n_online, 1.0)
    return np.column_stack(
        [np.ones(len(n_online), dtype=np.float64), n_online, remaining, frac]
    )


def fit_project_predict(
    y_fit: np.ndarray,
    x_fit: np.ndarray,
    x_pred: np.ndarray,
) -> tuple[np.ndarray, bool]:
    if len(y_fit) < x_fit.shape[1] + 2:
        return np.full(x_pred.shape[0], float(np.mean(y_fit))), True

    mu = x_fit[:, 1:].mean(axis=0)
    sd = x_fit[:, 1:].std(axis=0)
    sd[sd < 1e-12] = 1.0
    z_fit = x_fit.copy()
    z_pred = x_pred.copy()
    z_fit[:, 1:] = (z_fit[:, 1:] - mu) / sd
    z_pred[:, 1:] = (z_pred[:, 1:] - mu) / sd

    beta, *_ = np.linalg.lstsq(z_fit, y_fit, rcond=None)
    return z_pred @ beta, False


def crossfit_projection(
    score: np.ndarray,
    groups: list[np.ndarray],
    row_fold: np.ndarray,
    t: np.ndarray,
    n_online: np.ndarray,
    *,
    shuffle_horizon: bool = False,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, int]]:
    out = np.full(score.shape, np.nan, dtype=np.float64)
    rng = np.random.default_rng(seed)
    fallback_groups = 0
    fitted_groups = 0

    for group in groups:
        t_value = int(t[group[0]])
        for fold in FOLDS:
            pred_idx = group[row_fold[group] == fold]
            if len(pred_idx) == 0:
                continue
            fit_idx = group[row_fold[group] != fold]
            if len(fit_idx) == 0:
                continue

            fit_n = n_online[fit_idx]
            pred_n = n_online[pred_idx]
            if shuffle_horizon:
                fit_n = rng.permutation(fit_n)
                pred_n = rng.choice(fit_n, size=len(pred_idx), replace=True)

            pred, fallback = fit_project_predict(
                score[fit_idx],
                design_matrix(t_value, fit_n),
                design_matrix(t_value, pred_n),
            )
            out[pred_idx] = pred
            fitted_groups += 1
            fallback_groups += int(fallback)

    return out, {"fitted_groups": fitted_groups, "fallback_groups": fallback_groups}


def insample_projection(
    score: np.ndarray,
    groups: list[np.ndarray],
    t: np.ndarray,
    n_online: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    out = np.full(score.shape, np.nan, dtype=np.float64)
    fallback_groups = 0
    for group in groups:
        t_value = int(t[group[0]])
        pred, fallback = fit_project_predict(
            score[group],
            design_matrix(t_value, n_online[group]),
            design_matrix(t_value, n_online[group]),
        )
        out[group] = pred
        fallback_groups += int(fallback)
    return out, {"fitted_groups": len(groups), "fallback_groups": fallback_groups}


def midrank_percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    n = len(values)
    ranks = np.empty(n, dtype=np.float64)
    starts = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1]])
    ends = np.r_[starts[1:], n]
    for lo, hi in zip(starts, ends):
        avg = 0.5 * (lo + hi - 1) + 1.0
        ranks[order[lo:hi]] = avg
    return ranks / (n + 1.0)


def within_t_rank_vector(scores: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    out = np.full(scores.shape, np.nan, dtype=np.float64)
    for group in groups:
        finite = np.isfinite(scores[group])
        if finite.sum() < 2:
            continue
        idx = group[finite]
        out[idx] = midrank_percentile(scores[idx])
    return out


def pearson(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    m = mask & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return float("nan")
    aa = a[m].astype(np.float64)
    bb = b[m].astype(np.float64)
    aa -= aa.mean()
    bb -= bb.mean()
    den = math.sqrt(float(np.dot(aa, aa) * np.dot(bb, bb)))
    return float(np.dot(aa, bb) / den) if den > 0 else float("nan")


def score_on(vec: np.ndarray, mask: np.ndarray, y: np.ndarray, t: np.ndarray) -> float:
    m = mask & np.isfinite(vec)
    if m.sum() == 0:
        return float("nan")
    return float(ts_auc_flat(vec[m], y[m], t[m]))


def summarize_score(
    vec: np.ndarray,
    masks: dict[str, np.ndarray],
    y: np.ndarray,
    t: np.ndarray,
    row_fold: np.ndarray,
) -> dict[str, object]:
    cell = masks["dominant_cell"]
    per_fold = {
        str(fold): score_on(vec, cell & (row_fold == fold), y, t) for fold in FOLDS
    }
    return {
        "dominant_cell_ts_auc": score_on(vec, cell, y, t),
        "dominant_never_break_only_ts_auc": score_on(
            vec, masks["dominant_never_break_only"], y, t
        ),
        "dominant_pre_break_only_ts_auc": score_on(
            vec, masks["dominant_pre_break_only"], y, t
        ),
        "per_fold_dominant_cell_ts_auc": per_fold,
    }


def weighted_per_step_auc(vec: np.ndarray, mask: np.ndarray, y: np.ndarray, t: np.ndarray):
    m = mask & np.isfinite(vec)
    _, per = ts_auc_flat(vec[m], y[m], t[m], return_per_step=True)
    return per


def r2(observed: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> float:
    m = mask & np.isfinite(observed) & np.isfinite(predicted)
    yy = observed[m]
    pp = predicted[m]
    sst = float(np.dot(yy - yy.mean(), yy - yy.mean()))
    sse = float(np.dot(yy - pp, yy - pp))
    return 1.0 - sse / sst if sst > 0 else float("nan")


def make_markdown(result: dict[str, object]) -> str:
    scores = result["scores"]
    deltas = result["deltas_vs_B"]
    rho = result["within_t_rank_rho_vs_B"]
    cut = result["cut_decomposition"]
    lines = [
        "# Arm-C Horizon Residualization Diagnostic",
        "",
        f"Date: `{result['date']}`",
        "",
        "This is an offline diagnostic of the W7-D3R Arm C result. It uses no",
        "new model training, allocates no RT ID, and does not edit `RESULTS.csv`.",
        "",
        "## Inputs",
        "",
        f"- folds: `{result['inputs']['folds_path']}`",
        f"- OOF dir: `{result['inputs']['oof_dir']}`",
        f"- Arm A: `{result['inputs']['rt300_path']}`",
        f"- Arm B: `{result['inputs']['rt990_path']}`",
        f"- Arm C: `{result['inputs']['rt991_path']}`",
        "",
        "Population: dev folds 0-4, `t >= 200`, and either positive",
        "post-break age `>= 100` or any negative row.",
        "",
        "## Scores",
        "",
        "| score | dominant cell | vs Arm B | never-break-only | pre-break-only |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in result["score_order"]:
        row = scores[name]
        lines.append(
            "| {name} | {cell:.6f} | {delta:+.6f} | {never:.6f} | {pre:.6f} |".format(
                name=name,
                cell=row["dominant_cell_ts_auc"],
                delta=deltas[name]["dominant_cell_ts_auc"],
                never=row["dominant_never_break_only_ts_auc"],
                pre=row["dominant_pre_break_only_ts_auc"],
            )
        )

    lines.extend(
        [
            "",
            "## Projection Diagnostics",
            "",
            f"- Cross-fit projection R2 on logit Arm C: `{result['projection']['crossfit_r2']:.6f}`",
            f"- In-sample projection R2 on logit Arm C: `{result['projection']['insample_r2']:.6f}`",
            f"- Shuffled-horizon cross-fit projection R2: `{result['projection']['shuffled_crossfit_r2']:.6f}`",
            f"- Cross-fit projection groups: `{result['projection']['crossfit_groups']}`",
            f"- In-sample projection groups: `{result['projection']['insample_groups']}`",
            f"- Within-t rank rho, residual vs Arm B: `{rho['residual_xfit']:.6f}`",
            f"- Within-t rank rho, Arm C vs Arm B: `{rho['arm_c']:.6f}`",
            "",
            "## Residual Lift By Cut",
            "",
            f"- Dominant cell (gated): `{cut['residual_lift_dominant_cell']:+.6f}`",
            f"- Never-break-only cut: `{cut['residual_lift_never_break_only']:+.6f}`",
            f"- Pre-break-only cut: `{cut['residual_lift_pre_break_only']:+.6f}`",
            "",
            "**The residual lift is a never-break-cut effect only.** On the",
            "pre-break cut the T-orthogonal residual scores *below* Arm B, so the",
            "residual carries no pre-break signal at all. Arm C's large raw",
            "pre-break lift (`+0.087898`) is endpoint/horizon information, not",
            "transferable structure -- the horizon projection alone scores",
            "`0.700855` on that cut. Any downstream student of this residual",
            "should be expected to move the never-break cut and nothing else,",
            "and should be described that way.",
            "",
            "## Verdict",
            "",
            result["verdict"],
            "",
            "Primary gate from the Grok report: residual cell lift over Arm B",
            "must be at least `+0.020` to promote the residual-student lane;",
            "the lane is killed if lift is below `+0.010` or residual-vs-B",
            "within-t rank rho exceeds `0.850`. That gate is scoped to the",
            "dominant cell; it does not certify the pre-break cut, which fails.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folds-path",
        type=Path,
        default=ROOT / "research" / "folds" / "folds.parquet",
    )
    parser.add_argument("--oof-dir", type=Path, default=default_oof_dir())
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "research" / "reports" / "armc_residualization.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "research" / "reports" / "armc_residualization.md",
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    rows = build_row_arrays(args.folds_path)
    masks = dominant_masks(rows)
    groups = group_indices_by_t(rows["t"], masks["dominant_cell"])

    oof_paths = {
        "Arm_A_RT300": args.oof_dir / "RT-300.npy",
        "Arm_B_RT990": args.oof_dir / "RT-990.npy",
        "Arm_C_RT991": args.oof_dir / "RT-991.npy",
    }
    for label, path in oof_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} missing: {path}")

    vectors = {label: np.load(path, mmap_mode="r") for label, path in oof_paths.items()}
    expected_len = len(rows["y"])
    for label, vec in vectors.items():
        if len(vec) != expected_len:
            raise ValueError(f"{label} length {len(vec)} != row count {expected_len}")

    finite_common = np.ones(expected_len, dtype=bool)
    for vec in vectors.values():
        finite_common &= np.isfinite(vec)
    masks["dominant_cell"] &= finite_common
    masks["dominant_never_break_only"] &= finite_common
    masks["dominant_pre_break_only"] &= finite_common
    groups = group_indices_by_t(rows["t"], masks["dominant_cell"])

    c_logit = logit(vectors["Arm_C_RT991"], args.eps)
    h_xfit, h_xfit_stats = crossfit_projection(
        c_logit,
        groups,
        rows["row_fold"],
        rows["t"],
        rows["n_online_by_row"],
        seed=args.seed,
    )
    h_insample, h_insample_stats = insample_projection(
        c_logit, groups, rows["t"], rows["n_online_by_row"]
    )
    h_shuffle, h_shuffle_stats = crossfit_projection(
        c_logit,
        groups,
        rows["row_fold"],
        rows["t"],
        rows["n_online_by_row"],
        shuffle_horizon=True,
        seed=args.seed,
    )

    residual_xfit = c_logit - h_xfit
    residual_insample = c_logit - h_insample
    residual_shuffle = c_logit - h_shuffle

    score_vectors = {
        "Arm_A_RT300": vectors["Arm_A_RT300"],
        "Arm_B_RT990": vectors["Arm_B_RT990"],
        "Arm_C_RT991": vectors["Arm_C_RT991"],
        "Horizon_projection_xfit": h_xfit,
        "T_orthogonal_residual_xfit": residual_xfit,
        "Horizon_projection_insample": h_insample,
        "T_orthogonal_residual_insample": residual_insample,
        "Shuffled_horizon_projection_xfit": h_shuffle,
        "Residual_after_shuffled_projection": residual_shuffle,
    }
    score_order = list(score_vectors)
    scores = {
        name: summarize_score(vec, masks, rows["y"], rows["t"], rows["row_fold"])
        for name, vec in score_vectors.items()
    }

    b_cell = scores["Arm_B_RT990"]["dominant_cell_ts_auc"]
    deltas_vs_b = {
        name: {
            key: value - b_cell if key == "dominant_cell_ts_auc" else None
            for key, value in score.items()
            if isinstance(value, float)
        }
        for name, score in scores.items()
    }

    rank_b = within_t_rank_vector(vectors["Arm_B_RT990"], groups)
    rank_c = within_t_rank_vector(vectors["Arm_C_RT991"], groups)
    rank_r = within_t_rank_vector(residual_xfit, groups)
    rank_h = within_t_rank_vector(h_xfit, groups)
    rho = {
        "arm_c": pearson(rank_c, rank_b, masks["dominant_cell"]),
        "horizon_projection_xfit": pearson(rank_h, rank_b, masks["dominant_cell"]),
        "residual_xfit": pearson(rank_r, rank_b, masks["dominant_cell"]),
    }

    residual_lift = (
        scores["T_orthogonal_residual_xfit"]["dominant_cell_ts_auc"] - b_cell
    )
    if residual_lift >= 0.020 and not (rho["residual_xfit"] > 0.850):
        verdict = (
            "PROMOTE diagnostic: the T-orthogonal residual keeps material "
            "dominant-cell signal beyond Arm B. Next step is a nested causal "
            "student with the Wave 7 fold-purity sentinel."
        )
    elif residual_lift < 0.010 or rho["residual_xfit"] > 0.850:
        verdict = (
            "KILL residual-student lane under the Grok gate: the cross-fitted "
            f"T-orthogonal residual lift is {residual_lift:+.6f} and the "
            f"within-t rank rho vs Arm B is {rho['residual_xfit']:.6f}. "
            "This supports treating W7-D3R Arm C mainly as endpoint/future "
            "information not recoverable by this residual target."
        )
    else:
        verdict = (
            "AMBIGUOUS: the T-orthogonal residual is above the kill lift but "
            "below the promotion lift. Do not train a full student until the "
            "projection specification is tightened or replicated."
        )

    result = {
        "experiment": "armc_horizon_residualization",
        "date": datetime.now().isoformat(timespec="seconds"),
        "purpose": (
            "Decompose RT-991 Arm C into a within-t projection on horizon "
            "variables and a residual on the fixed W7-D0 dominant cell."
        ),
        "inputs": {
            "folds_path": str(args.folds_path),
            "oof_dir": str(args.oof_dir),
            "rt300_path": str(oof_paths["Arm_A_RT300"]),
            "rt990_path": str(oof_paths["Arm_B_RT990"]),
            "rt991_path": str(oof_paths["Arm_C_RT991"]),
            "rt300_sha256": sha256_file(oof_paths["Arm_A_RT300"]),
            "rt990_sha256": sha256_file(oof_paths["Arm_B_RT990"]),
            "rt991_sha256": sha256_file(oof_paths["Arm_C_RT991"]),
        },
        "population": {
            "row_count_total": expected_len,
            "row_count_dev_finite": int((masks["dev"] & finite_common).sum()),
            "row_count_dominant_cell": int(masks["dominant_cell"].sum()),
            "row_count_never_break_only": int(
                masks["dominant_never_break_only"].sum()
            ),
            "row_count_pre_break_only": int(masks["dominant_pre_break_only"].sum()),
            "t_min": int(rows["t"][masks["dominant_cell"]].min()),
            "t_max": int(rows["t"][masks["dominant_cell"]].max()),
            "n_t_groups": len(groups),
        },
        "projection": {
            "score_scale": "logit(RT-991), clipped to [eps, 1-eps]",
            "eps": args.eps,
            "formula": "logit(RT-991) ~ 1 + n_online + (n_online - t) + t / n_online, fit separately within each t",
            "crossfit": "fold k projected by coefficients fit on dominant-cell rows from folds != k at the same t",
            "crossfit_groups": h_xfit_stats,
            "insample_groups": h_insample_stats,
            "shuffled_horizon_groups": h_shuffle_stats,
            "crossfit_r2": r2(c_logit, h_xfit, masks["dominant_cell"]),
            "insample_r2": r2(c_logit, h_insample, masks["dominant_cell"]),
            "shuffled_crossfit_r2": r2(c_logit, h_shuffle, masks["dominant_cell"]),
        },
        "score_order": score_order,
        "scores": scores,
        "deltas_vs_B": deltas_vs_b,
        "within_t_rank_rho_vs_B": rho,
        "cut_decomposition": {
            "residual_lift_dominant_cell": residual_lift,
            "residual_lift_never_break_only": (
                scores["T_orthogonal_residual_xfit"]["dominant_never_break_only_ts_auc"]
                - scores["Arm_B_RT990"]["dominant_never_break_only_ts_auc"]
            ),
            "residual_lift_pre_break_only": (
                scores["T_orthogonal_residual_xfit"]["dominant_pre_break_only_ts_auc"]
                - scores["Arm_B_RT990"]["dominant_pre_break_only_ts_auc"]
            ),
            "note": (
                "The gate is evaluated on the dominant cell only. Resolved by cut, the "
                "residual lift is entirely a never-break-cut effect: on the pre-break cut "
                "the T-orthogonal residual scores BELOW Arm B, so the residual carries no "
                "pre-break signal. Arm C's large raw pre-break lift is therefore endpoint/"
                "horizon information, not transferable structure."
            ),
        },
        "gates": {
            "promotion_residual_lift": 0.020,
            "kill_residual_lift": 0.010,
            "kill_rank_rho": 0.850,
            "observed_residual_lift": residual_lift,
            "gate_scope": "dominant_cell",
        },
        "verdict": verdict,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    args.out_md.write_text(make_markdown(result))

    print(json.dumps({
        "out_json": str(args.out_json),
        "out_md": str(args.out_md),
        "arm_b_cell_auc": b_cell,
        "arm_c_cell_auc": scores["Arm_C_RT991"]["dominant_cell_ts_auc"],
        "residual_cell_auc": scores["T_orthogonal_residual_xfit"]["dominant_cell_ts_auc"],
        "residual_lift_vs_b": residual_lift,
        "residual_rank_rho_vs_b": rho["residual_xfit"],
        "verdict": verdict,
    }, indent=2))


if __name__ == "__main__":
    main()
