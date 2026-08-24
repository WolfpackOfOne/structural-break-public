"""Pilot 2: protective-relay score-state transform on RT-600 evidence.

Scored candidate: RT-1200.
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
from wave5_lib import Ctx, SPECIALISTS, load_oof

OUTDIR = ROOT / "research" / "reports" / "new_avenues_2026"
OOFDIR = ROOT / "research" / "oof"
OUTDIR.mkdir(parents=True, exist_ok=True)
OOFDIR.mkdir(parents=True, exist_ok=True)

EXP_ID = "RT-1200"
TAU_CHARGE = 8.0
TAU_COOL = 32.0
RESET_RATE = 0.25


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


def robust_center_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0) * 1.4826
    sd = np.nanstd(x, axis=0)
    scale = np.where(mad > 1e-12, mad, sd)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return med, scale


def relay_components_for_series(
    score: np.ndarray,
    pickup: float,
    dropout: float,
    tau_charge: float = TAU_CHARGE,
    tau_cool: float = TAU_COOL,
    reset_rate: float = RESET_RATE,
) -> np.ndarray:
    """Causal relay state. Row t reads only score[:t+1]."""
    n = len(score)
    out = np.zeros((n, 4), dtype=np.float64)
    picked = False
    operate = 0.0
    thermal = 0.0
    dwell = 0.0
    cycles = 0.0
    eps = 1e-9
    for i, s in enumerate(score):
        was_picked = picked
        if picked:
            if s <= dropout:
                picked = False
        elif s >= pickup:
            picked = True

        if picked:
            mult = max(s - pickup, 0.0) / max(1.0 - pickup, eps)
            operate += mult
            thermal += (s - thermal) / tau_charge
            dwell += 1.0
        else:
            operate = max(0.0, operate - reset_rate)
            thermal *= np.exp(-1.0 / tau_cool)
            dwell = 0.0
            if was_picked:
                cycles = min(5.0, cycles + 1.0)

        out[i] = [operate, thermal, dwell, cycles]
    return out


def build_relay_candidate(c: Ctx, rt600: np.ndarray) -> tuple[np.ndarray, dict]:
    train_rows = np.concatenate([c.rows[k] for k in (1, 2, 3, 4)])
    train_rows = train_rows[c.d.t[train_rows] >= 20]
    train_scores = rt600[train_rows]
    pickup = float(np.nanquantile(train_scores, 0.80))
    dropout = float(np.nanquantile(train_scores, 0.65))
    high_pickup = float(np.nanquantile(train_scores, 0.90))

    raw = np.full((len(c.d.y), 4), np.nan, dtype=np.float64)
    dev_rows = c.dev
    sid = c.d.sidx[dev_rows]
    order = np.argsort(sid, kind="stable")
    sorted_rows = dev_rows[order]
    sorted_sid = sid[order]
    starts = np.flatnonzero(np.r_[True, sorted_sid[1:] != sorted_sid[:-1]])
    ends = np.r_[starts[1:], len(sorted_sid)]
    for lo, hi in zip(starts, ends):
        rows = sorted_rows[lo:hi]
        # Rows are series-major in the repository, but sort by t defensively.
        rows = rows[np.argsort(c.d.t[rows], kind="stable")]
        raw[rows] = relay_components_for_series(rt600[rows], pickup, dropout)

    center, scale = robust_center_scale(raw[train_rows])
    z = (raw - center) / scale
    cand = np.full(len(c.d.y), np.nan, dtype=np.float64)
    cand[c.dev] = np.nanmean(z[c.dev], axis=1)
    constants = {
        "pickup_q80": pickup,
        "dropout_q65": dropout,
        "high_pickup_q90": high_pickup,
        "tau_charge": TAU_CHARGE,
        "tau_cool": TAU_COOL,
        "reset_rate": RESET_RATE,
        "component_center": center.tolist(),
        "component_scale": scale.tolist(),
        "components": ["operate_integral", "thermal_replica", "picked_dwell", "recloser_cycles"],
    }
    return cand, constants


def verify_prefix(c: Ctx, rt600: np.ndarray, constants: dict, n_series: int = 8) -> dict:
    lens = c.n_online.copy()
    dev_series = np.unique(c.d.sidx[c.dev])
    dev_lens = lens[dev_series]
    pick = dev_series[np.argsort(dev_lens)[np.linspace(0, len(dev_lens) - 1, n_series).astype(int)]]
    cuts = (3, 10, 37, 111)
    checked = 0
    for sid in pick:
        rows = c.dev[c.d.sidx[c.dev] == sid]
        rows = rows[np.argsort(c.d.t[rows], kind="stable")]
        full = relay_components_for_series(rt600[rows], constants["pickup_q80"], constants["dropout_q65"])
        for cut in cuts:
            if cut >= len(rows):
                continue
            part = relay_components_for_series(rt600[rows[:cut]], constants["pickup_q80"], constants["dropout_q65"])
            if not np.array_equal(full[:cut], part):
                return {"ok": False, "message": f"series {int(sid)} prefix {cut} differs"}
            checked += 1
    return {"ok": True, "message": "ok", "series_checked": int(len(pick)), "prefixes_checked": int(checked)}


def main() -> None:
    t0 = time.time()
    c = Ctx()
    oof = load_oof(SPECIALISTS + ["RT-401"])
    base = rt600_blend(c)
    mean_auc, per_fold = c.score(base)
    pooled = c.pooled(base)
    cand, constants = build_relay_candidate(c, base)
    verify = verify_prefix(c, base, constants)
    if not verify["ok"]:
        raise SystemExit(f"prefix verification failed: {verify['message']}")

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
        "candidate": "relay score-state transform of RT-600 evidence",
        "rt600_reproduction": {"mean": mean_auc, "per_fold": per_fold, "pooled": pooled},
        "constants": constants,
        "prefix_verification": verify,
        "diagnostic_pack": pack,
        "pair_flow": pair_flow,
        "ensemble_marginal": marg,
        "runtime_s": runtime,
        "verdict": verdict,
        "oof_artifact": str(OOFDIR / f"{EXP_ID}.npy"),
    }
    json_path = OUTDIR / "pilot02_relay_logic.json"
    md_path = OUTDIR / "pilot02_relay_logic.md"
    json_path.write_text(json.dumps(finite_float(result), indent=2, sort_keys=True) + "\n")

    md = [
        "# PILOT 2 -- RELAY SCORE-STATE LOGIC",
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
        "Causal protective-relay state applied to the RT-600 score path with fixed "
        "pickup/dropout quantiles from folds 1..4 and no label-tuned thresholds.",
        "",
        f"* Pickup q80: `{constants['pickup_q80']:.6f}`.",
        f"* Dropout q65: `{constants['dropout_q65']:.6f}`.",
        f"* Prefix verification: `{verify['message']}` over `{verify['prefixes_checked']}` prefixes.",
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
        "The kill/continue decision is based only on the marginal-vs-clone result. "
        "Standalone candidate behavior is diagnostic.",
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
