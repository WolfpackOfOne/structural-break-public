"""Pilot 3: IM2 matched-length empirical run null + dwell bank.

Scored candidate: RT-1201.
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
from sbr.transforms import HistParams, _ar_resid, ar_filter_causal
from wave5_lib import Ctx, SPECIALISTS, load_oof

OUTDIR = ROOT / "research" / "reports" / "new_avenues_2026"
OOFDIR = ROOT / "research" / "oof"
OUTDIR.mkdir(parents=True, exist_ok=True)
OOFDIR.mkdir(parents=True, exist_ok=True)

EXP_ID = "RT-1201"
WINDOWS = (32, 64, 128)
Q = 0.90


def finite_float(x):
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


def rollmean(csum_ready: np.ndarray, w: int) -> np.ndarray:
    return (csum_ready[w:] - csum_ready[:-w]) / w


def hot_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not mask.any():
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    ch = np.diff(np.r_[0, mask.astype(np.int8), 0])
    starts = np.flatnonzero(ch == 1)
    ends = np.flatnonzero(ch == -1)
    return starts.astype(np.int64), ends.astype(np.int64)


def matched_run_percentile(
    run_starts: np.ndarray,
    run_ends: np.ndarray,
    hist_len: int,
    segment_len: int,
    observed_maxrun: float,
) -> float:
    """Exact P(history segment max run <= observed_maxrun) for fixed segment length.

    Counts historical segment starts whose length-L segment contains at least
    one hot sub-run of length observed_maxrun+1 using interval union. History
    only; no online future or final online length.
    """
    if hist_len <= 0 or segment_len <= 0 or segment_len > hist_len:
        return float("nan")
    nstarts = hist_len - segment_len + 1
    q = int(np.floor(observed_maxrun)) + 1
    if q <= 0:
        return 1.0
    intervals = []
    last_start = hist_len - segment_len
    for a, b in zip(run_starts, run_ends):
        if b - a < q:
            continue
        lo = int(a + q - segment_len)
        hi = int(b - q)
        lo = max(0, lo)
        hi = min(last_start, hi)
        if lo <= hi:
            intervals.append((lo, hi))
    if not intervals:
        return 1.0
    intervals.sort()
    covered = 0
    cur_lo, cur_hi = intervals[0]
    for lo, hi in intervals[1:]:
        if lo <= cur_hi + 1:
            cur_hi = max(cur_hi, hi)
        else:
            covered += cur_hi - cur_lo + 1
            cur_lo, cur_hi = lo, hi
    covered += cur_hi - cur_lo + 1
    return float(1.0 - covered / nstarts)


def episode_mass_null(excess_h: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lengths = ends - starts
    masses = np.array([float(excess_h[a:b].sum()) for a, b in zip(starts, ends)], dtype=np.float64)
    if len(lengths) == 0:
        return np.array([0], dtype=np.int64), np.array([0.0], dtype=np.float64)
    order = np.argsort(lengths, kind="stable")
    return lengths[order], masses[order]


def mass_percentile(lengths: np.ndarray, masses: np.ndarray, segment_len: int, observed_mass: float) -> float:
    """History-only episode-length empirical mass null.

    This is the cheap mass companion to the exact matched-length run null:
    compare the online running-max excursion mass to historical complete
    excursion masses whose dwell length fits inside the same elapsed
    opportunity length.
    """
    if not np.isfinite(observed_mass) or segment_len <= 0:
        return float("nan")
    hi = np.searchsorted(lengths, segment_len, side="right")
    if hi <= 0:
        return float("nan")
    return float(np.mean(masses[:hi] <= observed_mass))


def dwell_features_for_series(hist: np.ndarray, online: np.ndarray) -> np.ndarray:
    hp = HistParams(np.asarray(hist, dtype=np.float64), ar_order=2)
    zh = (np.asarray(hist, dtype=np.float64) - hp.mu) / hp.sd
    zo = (np.asarray(online, dtype=np.float64) - hp.mu) / hp.sd
    eh = np.concatenate([np.zeros(2), _ar_resid(zh, hp.ar_coef)]) / hp.ar_sigma
    eo = ar_filter_causal(zo, hp.ar_coef, zh) / hp.ar_sigma
    sh = eh * eh
    so = eo * eo
    n = len(online)
    out = np.full((n, len(WINDOWS) * 3), np.nan, dtype=np.float32)
    csh = np.concatenate([[0.0], np.cumsum(sh, dtype=np.float64)])
    cso = np.concatenate([[0.0], np.cumsum(so, dtype=np.float64)])

    col = 0
    for w in WINDOWS:
        if len(sh) < 2 * w or n < w:
            col += 3
            continue
        hist_stat = rollmean(csh, w)
        med = float(np.median(hist_stat))
        band = float(np.quantile(np.abs(hist_stat - med), Q))
        hot_h = np.abs(hist_stat - med) > band
        starts, ends = hot_runs(hot_h)

        dev_h = np.abs(hist_stat - med)
        excess_h = np.where(hot_h, dev_h - band, 0.0)
        mass_lengths, mass_values = episode_mass_null(excess_h, starts, ends)

        online_stat = np.full(n, np.nan, dtype=np.float64)
        online_stat[w - 1 :] = rollmean(cso, w)
        dev_o = np.abs(online_stat - med)
        hot_o = np.zeros(n, dtype=bool)
        hot_o[w - 1 :] = dev_o[w - 1 :] > band

        cur_run = 0.0
        max_run = 0.0
        cur_mass = 0.0
        max_mass = 0.0
        total_exceed = 0.0
        run_pct = np.full(n, np.nan, dtype=np.float64)
        mass_pct = np.full(n, np.nan, dtype=np.float64)
        growth = np.full(n, np.nan, dtype=np.float64)
        for t in range(w - 1, n):
            if hot_o[t]:
                cur_run += 1.0
                total_exceed += 1.0
                cur_mass += max(float(dev_o[t] - band), 0.0)
            else:
                cur_run = 0.0
                cur_mass = 0.0
            max_run = max(max_run, cur_run)
            max_mass = max(max_mass, cur_mass)
            # Number of rolling-stat observations available online through t.
            seg_len = t - w + 2
            run_pct[t] = matched_run_percentile(starts, ends, len(hot_h), seg_len, max_run)
            mass_pct[t] = mass_percentile(mass_lengths, mass_values, seg_len, max_mass)
            growth[t] = np.log1p(max_run) / max(np.log1p(seg_len), 1e-12)
            _ = total_exceed  # kept as explicit online state; not scored in primary scalar.

        out[:, col] = run_pct
        out[:, col + 1] = mass_pct
        out[:, col + 2] = growth
        col += 3
    return out


def robust_center_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0) * 1.4826
    sd = np.nanstd(x, axis=0)
    scale = np.where(mad > 1e-12, mad, sd)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return med, scale


def build_feature_matrix(c: Ctx) -> tuple[np.ndarray, list[str], float]:
    t0 = time.time()
    st = load_store(str(ROOT / "cache" / "store"))
    cols = []
    for w in WINDOWS:
        cols += [f"res{w}_run_pct90", f"res{w}_mass_pct90", f"res{w}_growth90"]
    feats = np.full((len(c.d.y), len(cols)), np.nan, dtype=np.float32)

    dev_series = np.unique(c.d.sidx[c.dev])
    for count, sid in enumerate(dev_series):
        rows = c.dev[c.d.sidx[c.dev] == sid]
        rows = rows[np.argsort(c.d.t[rows], kind="stable")]
        feats[rows] = dwell_features_for_series(st.hist(int(sid)), st.online(int(sid)))
        if count % 1000 == 0:
            print(f"series {count}/{len(dev_series)} {time.time() - t0:.0f}s", flush=True)
    return feats, cols, time.time() - t0


def scalar_candidate(c: Ctx, feats: np.ndarray) -> tuple[np.ndarray, dict]:
    train_rows = np.concatenate([c.rows[k] for k in (1, 2, 3, 4)])
    center, scale = robust_center_scale(feats[train_rows])
    z = (feats - center) / scale
    cand = np.full(len(c.d.y), np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        row_mean = np.nanmean(z[c.dev], axis=1)
    # Before the first registered window fills, no dwell evidence is available.
    # Use the neutral scalar 0.0 rather than letting NaN enter TS-AUC.
    cand[c.dev] = np.nan_to_num(row_mean, nan=0.0, posinf=0.0, neginf=0.0)
    return cand, {
        "feature_center": center.tolist(),
        "feature_scale": scale.tolist(),
        "windows": list(WINDOWS),
        "threshold_quantile": Q,
        "history_channel": "AR(2) residual-square rolling mean",
    }


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
        full = dwell_features_for_series(hist, online).astype(np.float64)
        for cut in cuts:
            if cut >= len(online):
                continue
            part = dwell_features_for_series(hist, online[:cut]).astype(np.float64)
            a = full[:cut]
            b = part
            if not np.allclose(a, b, rtol=0.0, atol=0.0, equal_nan=True):
                diff = np.argwhere(~np.isclose(a, b, rtol=0.0, atol=0.0, equal_nan=True))[0]
                return {
                    "ok": False,
                    "message": f"series {int(sid)} prefix {cut} differs at row {int(diff[0])} col {int(diff[1])}",
                }
            checked += 1
    return {"ok": True, "message": "ok", "series_checked": int(len(pick)), "prefixes_checked": int(checked)}


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

    feats, cols, build_runtime = build_feature_matrix(c)
    cand, constants = scalar_candidate(c, feats)
    np.save(OOFDIR / f"{EXP_ID}.npy", cand.astype(np.float32))

    pack = diagnostic_pack(c, cand, base, fold=0, label=EXP_ID)
    pair_flow = pair_flow_by_cell(base, cand, c, fold=0, n_pairs_per_t=20, seed=0)
    marg = marginal(cand, c, fold=0, label=EXP_ID)
    runtime = time.time() - t0
    verdict = "KILL" if marg["marginal_vs_clone"] < 0.0010 else "CONTINUE"

    result = {
        "generated": "2026-08-24",
        "exp_id": EXP_ID,
        "branch": "research/new-avenues-pilots-2026",
        "candidate": "IM2 matched-length run null plus excursion dwell bank",
        "rt600_reproduction": {"mean": mean_auc, "per_fold": per_fold, "pooled": pooled},
        "feature_columns": cols,
        "constants": constants,
        "prefix_verification": verify,
        "diagnostic_pack": pack,
        "pair_flow": pair_flow,
        "ensemble_marginal": marg,
        "feature_build_runtime_s": build_runtime,
        "runtime_s": runtime,
        "cleanup_incorporated": False,
        "result_status": "PRE-CLEANUP / PROVISIONAL",
        "verdict": verdict,
        "oof_artifact": str(OOFDIR / f"{EXP_ID}.npy"),
        "implementation_note": (
            "Run-length percentile uses exact interval-union matched-length history null. "
            "Mass percentile uses a history-only episode-length empirical null as the cheap "
            "mass companion."
        ),
    }
    json_path = OUTDIR / "pilot03_im2_dwell.json"
    md_path = OUTDIR / "pilot03_im2_dwell.md"
    json_path.write_text(json.dumps(finite_float(result), indent=2, sort_keys=True) + "\n")

    md = [
        "# PILOT 3 -- IM2 MATCHED-LENGTH RUN NULL AND DWELL BANK",
        "",
        f"Experiment ID: `{EXP_ID}`.",
        "",
        "## RT-600 Anchor",
        "",
        f"* Dev mean TS-AUC: `{mean_auc:.6f}`.",
        f"* Dev pooled TS-AUC: `{pooled:.6f}`.",
        f"* Fold-0 RT-600 TS-AUC in integration: `{marg['rt600_7stream']:.6f}`.",
        "",
        "## Candidate",
        "",
        "Nine-feature scalar from AR(2)-residual-square dwell state over windows "
        "`32,64,128`: matched-length run percentile, episode-mass percentile, "
        "and max-run growth proxy per window.",
        "",
        "**Status:** PRE-CLEANUP / PROVISIONAL. The expected novel-stream cleanup "
        "commit has not landed on `origin/research/current` yet.",
        "",
        f"* Prefix verification: `{verify['message']}` over `{verify['prefixes_checked']}` prefixes.",
        f"* Feature build runtime: `{build_runtime:.1f}s`.",
        "",
        "## Binding Marginal Result",
        "",
        "| arm | fold-0 TS-AUC |",
        "|---|---:|",
        f"| RT600 | {marg['rt600_7stream']:.6f} |",
        f"| RT600 + RT-401 seed clone | {marg['rt600_plus_seedclone']:.6f} |",
        f"| RT600 + {EXP_ID} | {marg[f'rt600_plus_{EXP_ID}']:.6f} |",
        "",
        f"Marginal vs clone: `{marg['marginal_vs_clone']:+.6f}`.",
        f"Verdict: **{verdict}**.",
        "",
        "## Diagnostic Pack",
        "",
        f"* Whole fold candidate TS-AUC: `{pack['whole_fold']['candidate']:.6f}` "
        f"(RT-600 `{pack['whole_fold']['rt600']:.6f}`).",
        f"* Dominant-cell candidate AUC: `{pack['dominant_cell']['candidate']:.6f}` "
        f"(RT-600 `{pack['dominant_cell']['rt600']:.6f}`).",
        f"* Mature vs never-break: `{pack['mature_vs_neverbreak']['candidate']:.6f}`.",
        f"* Mature vs pre-break: `{pack['mature_vs_prebreak']['candidate']:.6f}`.",
        f"* Within-t correlation with RT-600: `{pack['within_t_rank_corr_rt600']:+.4f}`.",
        "",
        "## Pair Flow",
        "",
        "| split | repairs | damage | net | sampled pairs |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in pair_flow.items():
        md.append(
            f"| `{name}` | {row['repairs']} | {row['damage']} | "
            f"{row['net_pair_lift']} | {row['total_pairs_sampled']} |"
        )
    md += [
        "",
        "## Interpretation",
        "",
        "The binding decision is the marginal-vs-clone comparison. Standalone dwell "
        "separation and low correlation are necessary diagnostics, not promotion criteria.",
        "",
        result["implementation_note"],
        "",
        f"Runtime: `{runtime:.1f}s`.",
    ]
    md_path.write_text("\n".join(md) + "\n")
    print(json.dumps(finite_float({"verdict": verdict, "marginal": marg, "runtime_s": runtime}), indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {OOFDIR / f'{EXP_ID}.npy'}")


if __name__ == "__main__":
    main()
