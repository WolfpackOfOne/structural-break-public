"""Pilot 4: trajectory geometry via NN provenance and arc-crossing proxy.

Scored candidates:
  RT-1202: real trajectory-geometry scalar.
  RT-1203: shuffled-history arc-order control scalar.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("SBR_ROOT", Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(ROOT / "research" / "scripts" / "novel_streams"))

import numpy as np

from harness import diagnostic_pack, marginal, pair_flow_by_cell, rt600_blend
from sbr.store import load_store
from wave5_lib import Ctx, SPECIALISTS, load_oof

OUTDIR = ROOT / "research" / "reports" / "new_avenues_2026"
OOFDIR = ROOT / "research" / "oof"
OUTDIR.mkdir(parents=True, exist_ok=True)
OOFDIR.mkdir(parents=True, exist_ok=True)

REAL_ID = "RT-1202"
CONTROL_ID = "RT-1203"
WINDOWS = (16, 64)
MAX_HIST_REFS = 2000
SHUFFLE_SEED = 0


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


def robust_history_normalize(hist: np.ndarray, online: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    h = np.asarray(hist, dtype=np.float64)
    o = np.asarray(online, dtype=np.float64)
    med = float(np.median(h))
    mad = float(np.median(np.abs(h - med)) * 1.4826)
    sd = float(np.std(h))
    scale = mad if mad > 1e-12 else sd
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return (h - med) / scale, (o - med) / scale, {"median": med, "scale": scale}


def z_windows(x: np.ndarray, m: int, indices: np.ndarray | None = None) -> np.ndarray:
    if len(x) < m:
        return np.empty((0, m), dtype=np.float32)
    w = np.lib.stride_tricks.sliding_window_view(np.asarray(x, dtype=np.float64), m)
    if indices is not None:
        w = w[indices]
    mu = w.mean(axis=1, keepdims=True)
    sd = w.std(axis=1, keepdims=True)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return ((w - mu) / sd).astype(np.float32, copy=False)


def sampled_reference_indices(n_hist_windows: int) -> np.ndarray:
    if n_hist_windows <= MAX_HIST_REFS:
        return np.arange(n_hist_windows, dtype=np.int64)
    return np.linspace(0, n_hist_windows - 1, MAX_HIST_REFS).round().astype(np.int64)


def nearest_hist_distance(
    q: np.ndarray,
    r: np.ndarray,
    ref_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(q) == 0 or len(r) == 0:
        return (
            np.full(len(q), np.nan, dtype=np.float64),
            np.full(len(q), -1, dtype=np.int64),
        )
    qn = np.einsum("ij,ij->i", q, q, dtype=np.float64)
    rn = np.einsum("ij,ij->i", r, r, dtype=np.float64)
    dots = q @ r.T
    d2 = qn[:, None] + rn[None, :] - 2.0 * dots.astype(np.float64, copy=False)
    np.maximum(d2, 0.0, out=d2)
    pos = np.argmin(d2, axis=1)
    dist = np.sqrt(d2[np.arange(len(q)), pos])
    return dist.astype(np.float64, copy=False), ref_idx[pos].astype(np.int64, copy=False)


def nearest_prior_online_distance(q: np.ndarray, m: int) -> np.ndarray:
    n = len(q)
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= m:
        return out
    qn = np.einsum("ij,ij->i", q, q, dtype=np.float64)
    for i in range(m, n):
        prior = q[: i - m + 1]
        dots = prior @ q[i]
        d2 = qn[i] + qn[: i - m + 1] - 2.0 * dots.astype(np.float64, copy=False)
        best = float(np.min(np.maximum(d2, 0.0)))
        out[i] = np.sqrt(best)
    return out


def arc_rates(nearest_hidx: np.ndarray, n_hist_windows: int, shuffled: bool) -> np.ndarray:
    if len(nearest_hidx) == 0:
        return np.empty(0, dtype=np.float64)
    hidx = nearest_hidx
    if shuffled:
        perm = np.random.default_rng(SHUFFLE_SEED).permutation(n_hist_windows)
        hidx = perm[np.clip(hidx, 0, n_hist_windows - 1)]
    boundary_start = int(np.ceil(0.75 * n_hist_windows))
    near_boundary = hidx >= boundary_start
    elapsed = np.arange(1, len(hidx) + 1, dtype=np.float64)
    return np.cumsum(near_boundary, dtype=np.float64) / elapsed


def components_for_length(hist_z: np.ndarray, online_z: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(online_z)
    real = np.full((n, 2), np.nan, dtype=np.float32)
    control = np.full((n, 2), np.nan, dtype=np.float32)
    if len(hist_z) < m or n < m:
        return real, control

    n_hist_windows = len(hist_z) - m + 1
    ref_idx = sampled_reference_indices(n_hist_windows)
    refs = z_windows(hist_z, m, ref_idx)
    queries = z_windows(online_z, m)

    d_hist, nearest_hidx = nearest_hist_distance(queries, refs, ref_idx)
    d_online = nearest_prior_online_distance(queries, m)

    with np.errstate(invalid="ignore"):
        prov = np.log1p(d_hist) - np.log1p(d_online)
    arc = arc_rates(nearest_hidx, n_hist_windows, shuffled=False)
    arc_control = arc_rates(nearest_hidx, n_hist_windows, shuffled=True)

    rows = np.arange(m - 1, n, dtype=np.int64)
    real[rows, 0] = prov.astype(np.float32, copy=False)
    real[rows, 1] = arc.astype(np.float32, copy=False)
    control[rows, 0] = prov.astype(np.float32, copy=False)
    control[rows, 1] = arc_control.astype(np.float32, copy=False)
    return real, control


def trajectory_features_for_series(hist: np.ndarray, online: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hist_z, online_z, _ = robust_history_normalize(hist, online)
    n = len(online_z)
    real = np.full((n, len(WINDOWS) * 2), np.nan, dtype=np.float32)
    control = np.full_like(real, np.nan)
    for j, m in enumerate(WINDOWS):
        r, c = components_for_length(hist_z, online_z, m)
        real[:, 2 * j : 2 * j + 2] = r
        control[:, 2 * j : 2 * j + 2] = c
    return real, control


def robust_center_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0) * 1.4826
    sd = np.nanstd(x, axis=0)
    scale = np.where(mad > 1e-12, mad, sd)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return med, scale


def build_feature_matrices(c: Ctx) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    t0 = time.time()
    st = load_store(str(ROOT / "cache" / "store"))
    cols = []
    for m in WINDOWS:
        cols += [f"prov{m}", f"arc{m}"]
    real = np.full((len(c.d.y), len(cols)), np.nan, dtype=np.float32)
    control = np.full_like(real, np.nan)

    dev_series = np.unique(c.d.sidx[c.dev])
    for count, sid in enumerate(dev_series):
        rows = c.dev[c.d.sidx[c.dev] == sid]
        rows = rows[np.argsort(c.d.t[rows], kind="stable")]
        r, s = trajectory_features_for_series(st.hist(int(sid)), st.online(int(sid)))
        real[rows] = r
        control[rows] = s
        if count % 500 == 0:
            print(f"series {count}/{len(dev_series)} {time.time() - t0:.0f}s", flush=True)
    return real, control, cols, time.time() - t0


def scalar_candidate(c: Ctx, feats: np.ndarray) -> tuple[np.ndarray, dict]:
    train_rows = np.concatenate([c.rows[k] for k in (1, 2, 3, 4)])
    center, scale = robust_center_scale(feats[train_rows])
    z = (feats - center) / scale
    cand = np.full(len(c.d.y), np.nan, dtype=np.float64)
    z_dev = z[c.dev]
    finite = np.isfinite(z_dev)
    row_sum = np.where(finite, z_dev, 0.0).sum(axis=1)
    row_count = finite.sum(axis=1)
    row_mean = np.divide(row_sum, row_count, out=np.zeros_like(row_sum), where=row_count > 0)
    cand[c.dev] = np.nan_to_num(row_mean, nan=0.0, posinf=0.0, neginf=0.0)
    return cand, {"feature_center": center.tolist(), "feature_scale": scale.tolist()}


def verify_prefix(c: Ctx, n_series: int = 8) -> dict:
    st = load_store(str(ROOT / "cache" / "store"))
    lens = c.n_online.copy()
    dev_series = np.unique(c.d.sidx[c.dev])
    dev_lens = lens[dev_series]
    pick = dev_series[np.argsort(dev_lens)[np.linspace(0, len(dev_lens) - 1, n_series).astype(int)]]
    cuts = (3, 10, 37, 111)
    checked = 0
    for sid in pick:
        hist = st.hist(int(sid))
        online = st.online(int(sid))
        full_real, full_control = trajectory_features_for_series(hist, online)
        for cut in cuts:
            if cut >= len(online):
                continue
            part_real, part_control = trajectory_features_for_series(hist, online[:cut])
            for name, a, b in (
                ("real", full_real[:cut], part_real),
                ("control", full_control[:cut], part_control),
            ):
                if not np.allclose(a, b, rtol=0.0, atol=0.0, equal_nan=True):
                    diff = np.argwhere(~np.isclose(a, b, rtol=0.0, atol=0.0, equal_nan=True))[0]
                    return {
                        "ok": False,
                        "message": (
                            f"{name} series {int(sid)} prefix {cut} differs at "
                            f"row {int(diff[0])} col {int(diff[1])}"
                        ),
                    }
            checked += 1
    return {"ok": True, "message": "ok", "series_checked": int(len(pick)), "prefixes_checked": int(checked)}


def gate_verdict(real_margin: float, control_margin: float) -> tuple[str, str]:
    diff = real_margin - control_margin
    if real_margin < 0.0010:
        return "KILL", "real marginal_vs_clone is below +0.0010"
    if control_margin >= real_margin - 0.0002:
        return "KILL", "shuffled-history control matches/exceeds real within 0.0002"
    if real_margin >= 0.0020 and diff >= 0.0005:
        return "CONTINUE", "real clears continuation gate and beats shuffled control"
    return "NO_5FOLD", "real clears kill floor but not the preregistered continuation gate"


def candidate_result(exp_id: str, cand: np.ndarray, base: np.ndarray, c: Ctx) -> dict:
    pack = diagnostic_pack(c, cand, base, fold=0, label=exp_id)
    pair_flow = pair_flow_by_cell(base, cand, c, fold=0, n_pairs_per_t=20, seed=0)
    marg = marginal(cand, c, fold=0, label=exp_id)
    return {"diagnostic_pack": pack, "pair_flow": pair_flow, "ensemble_marginal": marg}


def main() -> None:
    t0 = time.time()
    c = Ctx()
    _ = load_oof(SPECIALISTS + ["RT-401"])
    base = rt600_blend(c)
    mean_auc, per_fold = c.score(base)
    pooled = c.pooled(base)

    verify = verify_prefix(c)
    if not verify["ok"]:
        raise SystemExit(f"prefix verification failed: {verify['message']}")

    feats_real, feats_control, cols, build_runtime = build_feature_matrices(c)
    cand_real, constants_real = scalar_candidate(c, feats_real)
    cand_control, constants_control = scalar_candidate(c, feats_control)
    np.save(OOFDIR / f"{REAL_ID}.npy", cand_real.astype(np.float32))
    np.save(OOFDIR / f"{CONTROL_ID}.npy", cand_control.astype(np.float32))

    real_result = candidate_result(REAL_ID, cand_real, base, c)
    control_result = candidate_result(CONTROL_ID, cand_control, base, c)
    runtime = time.time() - t0

    real_margin = real_result["ensemble_marginal"]["marginal_vs_clone"]
    control_margin = control_result["ensemble_marginal"]["marginal_vs_clone"]
    verdict, verdict_reason = gate_verdict(real_margin, control_margin)

    constants = {
        "windows": list(WINDOWS),
        "max_hist_refs": MAX_HIST_REFS,
        "history_reference_sampling": "evenly_spaced",
        "prior_online_rule": "nearest prior online subsequence ending at or before t-m",
        "boundary_band": "last quarter of historical subsequence indices",
        "shuffle_seed": SHUFFLE_SEED,
        "component_order": cols,
        "real_scalar": constants_real,
        "control_scalar": constants_control,
    }

    result = {
        "generated": "2026-08-24",
        "exp_ids": [REAL_ID, CONTROL_ID],
        "branch": "research/new-avenues-pilots-2026",
        "candidate": "trajectory geometry NN provenance plus history-boundary arc-rate",
        "control": "same provenance with shuffled historical arc-order",
        "rt600_reproduction": {"mean": mean_auc, "per_fold": per_fold, "pooled": pooled},
        "feature_columns": cols,
        "constants": constants,
        "prefix_verification": verify,
        REAL_ID: real_result,
        CONTROL_ID: control_result,
        "marginal_difference_real_minus_control": real_margin - control_margin,
        "feature_build_runtime_s": build_runtime,
        "runtime_s": runtime,
        "cleanup_incorporated": False,
        "result_status": "PRE-CLEANUP / PROVISIONAL",
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "oof_artifacts": {
            REAL_ID: str(OOFDIR / f"{REAL_ID}.npy"),
            CONTROL_ID: str(OOFDIR / f"{CONTROL_ID}.npy"),
        },
    }

    json_path = OUTDIR / "pilot04_trajectory_geometry.json"
    md_path = OUTDIR / "pilot04_trajectory_geometry.md"
    json_path.write_text(json.dumps(finite_float(result), indent=2, sort_keys=True) + "\n")

    real_pack = real_result["diagnostic_pack"]
    control_pack = control_result["diagnostic_pack"]
    real_marg = real_result["ensemble_marginal"]
    control_marg = control_result["ensemble_marginal"]
    md = [
        "# PILOT 4 -- TRAJECTORY GEOMETRY",
        "",
        f"Experiment IDs: `{REAL_ID}` real scalar, `{CONTROL_ID}` shuffled-history control.",
        "",
        "## RT-600 Anchor",
        "",
        f"* Dev mean TS-AUC: `{mean_auc:.6f}`.",
        f"* Dev pooled TS-AUC: `{pooled:.6f}`.",
        f"* Fold-0 RT-600 TS-AUC in integration: `{real_marg['rt600_7stream']:.6f}`.",
        "",
        "## Candidate",
        "",
        "Four-feature scalar from z-normalized subsequence windows `16,64`: "
        "nearest-neighbour provenance and history-boundary arc-rate per window. "
        "The shuffled control preserves distance/provenance and permutes the "
        "historical order used by the arc-rate component.",
        "",
        "**Status:** PRE-CLEANUP / PROVISIONAL. The expected novel-stream cleanup "
        "commit has not landed on `origin/research/current` yet.",
        "",
        f"* Prefix verification: `{verify['message']}` over `{verify['prefixes_checked']}` prefixes.",
        f"* Feature build runtime: `{build_runtime:.1f}s`.",
        "",
        "## Binding Marginal Result",
        "",
        "| arm | fold-0 TS-AUC | marginal vs clone | gain vs RT600 |",
        "|---|---:|---:|---:|",
        f"| RT600 | {real_marg['rt600_7stream']:.6f} |  |  |",
        f"| RT600 + RT-401 seed clone | {real_marg['rt600_plus_seedclone']:.6f} |  |  |",
        (
            f"| RT600 + {REAL_ID} | {real_marg[f'rt600_plus_{REAL_ID}']:.6f} | "
            f"{real_marg['marginal_vs_clone']:+.6f} | {real_marg['gain_vs_base']:+.6f} |"
        ),
        (
            f"| RT600 + {CONTROL_ID} | {control_marg[f'rt600_plus_{CONTROL_ID}']:.6f} | "
            f"{control_marg['marginal_vs_clone']:+.6f} | {control_marg['gain_vs_base']:+.6f} |"
        ),
        "",
        f"Real minus shuffled marginal: `{real_margin - control_margin:+.6f}`.",
        f"Verdict: **{verdict}** -- {verdict_reason}.",
        "",
        "## Diagnostic Pack",
        "",
        "| candidate | whole-fold AUC | dominant-cell AUC | mature vs never | mature vs pre | rho vs RT600 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| `{REAL_ID}` | {real_pack['whole_fold']['candidate']:.6f} | "
            f"{real_pack['dominant_cell']['candidate']:.6f} | "
            f"{real_pack['mature_vs_neverbreak']['candidate']:.6f} | "
            f"{real_pack['mature_vs_prebreak']['candidate']:.6f} | "
            f"{real_pack['within_t_rank_corr_rt600']:+.4f} |"
        ),
        (
            f"| `{CONTROL_ID}` | {control_pack['whole_fold']['candidate']:.6f} | "
            f"{control_pack['dominant_cell']['candidate']:.6f} | "
            f"{control_pack['mature_vs_neverbreak']['candidate']:.6f} | "
            f"{control_pack['mature_vs_prebreak']['candidate']:.6f} | "
            f"{control_pack['within_t_rank_corr_rt600']:+.4f} |"
        ),
        "",
        "## Pair Flow",
        "",
        f"### {REAL_ID}",
        "",
        "| split | repairs | damage | net | sampled pairs |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in real_result["pair_flow"].items():
        md.append(
            f"| `{name}` | {row['repairs']} | {row['damage']} | "
            f"{row['net_pair_lift']} | {row['total_pairs_sampled']} |"
        )
    md += ["", f"### {CONTROL_ID}", "", "| split | repairs | damage | net | sampled pairs |", "|---|---:|---:|---:|---:|"]
    for name, row in control_result["pair_flow"].items():
        md.append(
            f"| `{name}` | {row['repairs']} | {row['damage']} | "
            f"{row['net_pair_lift']} | {row['total_pairs_sampled']} |"
        )
    md += [
        "",
        "## Interpretation",
        "",
        "The binding decision is the marginal-vs-clone gate plus the shuffled-order "
        "control. Standalone separation and low RT-600 correlation are diagnostic "
        "only unless the ensemble marginal clears the preregistered thresholds.",
        "",
        f"Runtime: `{runtime:.1f}s`.",
    ]
    md_path.write_text("\n".join(md) + "\n")
    print(
        json.dumps(
            finite_float(
                {
                    "verdict": verdict,
                    "reason": verdict_reason,
                    REAL_ID: real_marg,
                    CONTROL_ID: control_marg,
                    "runtime_s": runtime,
                }
            ),
            indent=2,
        )
    )
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {OOFDIR / f'{REAL_ID}.npy'}")
    print(f"wrote {OOFDIR / f'{CONTROL_ID}.npy'}")


if __name__ == "__main__":
    main()
