"""W7-D0: exact RT-600 pairwise-inversion loss cube.

Purpose: reproduce RT-600's development architecture OOF vector exactly
(research/oof/wave5_S_specialist.npy, the cross-fitted seven-specialist SCDF
blend documented as 0.62581 in research/HANDOFF_WAVE6.md and
research/RDOF_LEDGER.md), then decompose the *exact* remaining pairwise loss
by (current t bucket) x (positive post-break age bucket) x (negative type),
using the official metric's own rank/pair machinery -- no marginal-AUC
approximation.

No training. No new model. Pure recombination + exact scoring of an OOF
vector that already exists on disk. Same category as wave5_e1_mixture.py.
"""
from __future__ import annotations

import json
import time

import numpy as np

from wave5_lib import Ctx, FOLDS, REPORTS, OOFDIR, ts_auc_flat

T_BUCKETS = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 400), (400, 10**9)]
T_LABELS = ["0-20", "20-50", "50-100", "100-200", "200-400", "400+"]
AGE_BUCKETS = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 10**9)]
AGE_LABELS = ["0-5", "5-10", "10-20", "20-50", "50-100", "100+"]
NEG_TYPES = ["never_break", "pre_break"]

EXPECTED_MEAN = 0.62581
EXPECTED_PER_FOLD = [0.63828, 0.62040, 0.63392, 0.61750, 0.61894]
TOL = 1e-4


def bucketize(tvals, num, den, buckets):
    """Sum (num, den) into t-buckets."""
    out_num = np.zeros(len(buckets))
    out_den = np.zeros(len(buckets))
    for i, (lo, hi) in enumerate(buckets):
        m = (tvals >= lo) & (tvals < hi)
        out_num[i] = num[m].sum()
        out_den[i] = den[m].sum()
    return out_num, out_den


def main():
    t0 = time.time()
    c = Ctx()
    Sv = np.load(f"{OOFDIR}/wave5_S_specialist.npy")

    y = c.d.y
    t = c.d.t
    age = c.age
    sidx = c.d.sidx
    neg_is_prebreak = c.has_break[sidx] & (y == 0)  # False for never-break negs

    # ---- A. reproduction check -----------------------------------------
    mean, per_fold = c.score(Sv)
    delta_mean = mean - EXPECTED_MEAN
    delta_fold = [a - b for a, b in zip(per_fold, EXPECTED_PER_FOLD)]
    passed = bool(abs(delta_mean) <= TOL and all(abs(d) <= TOL for d in delta_fold))
    print("=== A. RT-600 REPRODUCTION ===")
    print(f"  expected mean {EXPECTED_MEAN:.5f}  observed {mean:.5f}  delta {delta_mean:+.6f}")
    print(f"  expected fold {EXPECTED_PER_FOLD}")
    print(f"  observed fold {[round(x,5) for x in per_fold]}")
    print(f"  PASS={passed}")
    if not passed:
        raise SystemExit("REPRODUCTION FAILED -- stop, do not build the cube on a mismatched vector")

    # ---- B. total loss, exact -------------------------------------------
    dev = c.dev
    total_ts_auc, dev_per = ts_auc_flat(Sv[dev], y[dev], t[dev], return_per_step=True)
    total_num = float((dev_per["auc"] * dev_per["w"]).sum())
    total_den = float(dev_per["w"].sum())
    total_loss = total_den - total_num
    print(f"\n=== B. EXACT REMAINING LOSS (pooled dev) ===")
    print(f"  pooled TS-AUC {total_ts_auc:.6f}  total pair weight {total_den:,.0f}  "
          f"total inversion-equivalent loss {total_loss:,.1f}  loss fraction {total_loss/total_den:.6f}")

    # ---- C. build the cube: t-bucket x age-bucket x neg-type -------------
    # cube[negtype][age][tbucket] = (num, den)
    dev_mask = np.zeros(len(y), dtype=bool)
    dev_mask[dev] = True
    cube = {}
    for nt_i, nt_name in enumerate(NEG_TYPES):
        want_prebreak = bool(nt_i)  # 0=never_break(False), 1=pre_break(True)
        cube[nt_name] = {}
        for (alo, ahi), alabel in zip(AGE_BUCKETS, AGE_LABELS):
            pos_mask = (y == 1) & (age >= alo) & (age < ahi)
            neg_mask = (y == 0) & (neg_is_prebreak == want_prebreak)
            m = (pos_mask | neg_mask) & dev_mask
            if not m.any():
                cube[nt_name][alabel] = {"num": [0.0]*len(T_BUCKETS), "den": [0.0]*len(T_BUCKETS)}
                continue
            _, per = ts_auc_flat(Sv[m], y[m], t[m], return_per_step=True)
            num_t = per["auc"] * per["w"]
            den_t = per["w"]
            bn, bd = bucketize(per["t"], num_t, den_t, T_BUCKETS)
            cube[nt_name][alabel] = {"num": bn.tolist(), "den": bd.tolist()}

    # ---- D. summarise the cube -------------------------------------------
    print("\n=== D. CUBE SUMMARY (pair weight share / loss share / cell AUC) ===")
    print(f"{'negtype':12s} {'age':8s} " + " ".join(f"{l:>10s}" for l in T_LABELS))
    grand_den = 0.0
    grand_loss = 0.0
    cell_rows = []
    for nt_name in NEG_TYPES:
        for alabel in AGE_LABELS:
            bn = np.array(cube[nt_name][alabel]["num"])
            bd = np.array(cube[nt_name][alabel]["den"])
            bl = bd - bn
            grand_den += bd.sum()
            grand_loss += bl.sum()
            for i, tl in enumerate(T_LABELS):
                cell_rows.append({
                    "neg_type": nt_name, "age_bucket": alabel, "t_bucket": tl,
                    "pair_weight": float(bd[i]), "inversion_loss": float(bl[i]),
                    "cell_auc": float(bn[i] / bd[i]) if bd[i] > 0 else None,
                })
            line = " ".join(f"{bd[i]:10.0f}" for i in range(len(T_LABELS)))
            print(f"{nt_name:12s} {alabel:8s} {line}")

    print(f"\n  grand total pair weight (cube) {grand_den:,.0f} vs pooled scorer {total_den:,.0f}  "
          f"(should match exactly: every dev pos/neg pair falls in exactly one cell)")
    print(f"  grand total loss (cube) {grand_loss:,.1f} vs pooled scorer {total_loss:,.1f}")

    # ---- E. the dominant-cell claim: t>=200 AND age>=100 -----------------
    dom_t = [i for i, (lo, hi) in enumerate(T_BUCKETS) if lo >= 200]
    dom_a = [i for i, l in enumerate(AGE_LABELS) if l == "100+"]
    dom_den = dom_loss = 0.0
    dom_den_never = dom_den_pre = dom_loss_never = dom_loss_pre = 0.0
    for nt_name in NEG_TYPES:
        alabel = AGE_LABELS[dom_a[0]]
        bn = np.array(cube[nt_name][alabel]["num"])
        bd = np.array(cube[nt_name][alabel]["den"])
        bl = bd - bn
        sub_den = bd[dom_t].sum()
        sub_loss = bl[dom_t].sum()
        dom_den += sub_den
        dom_loss += sub_loss
        if nt_name == "never_break":
            dom_den_never, dom_loss_never = sub_den, sub_loss
        else:
            dom_den_pre, dom_loss_pre = sub_den, sub_loss

    dom_auc = 1.0 - (dom_loss / dom_den) if dom_den > 0 else None
    frac_weight = dom_den / grand_den
    frac_loss = dom_loss / grand_loss
    # perfect-repair ceiling: zero out this cell's loss, recompute pooled TS-AUC
    ceiling_ts_auc = (total_num + dom_loss) / total_den

    print(f"\n=== E. DOMINANT CELL: t>=200 AND age>=100 ===")
    print(f"  fraction of pair weight: {frac_weight:.4f}")
    print(f"  fraction of inversion loss: {frac_loss:.4f}")
    print(f"  cell AUC: {dom_auc:.5f}" if dom_auc is not None else "  cell AUC: n/a")
    print(f"  perfect-repair ceiling TS-AUC: {ceiling_ts_auc:.5f}  (+{ceiling_ts_auc-total_ts_auc:.5f} over observed pooled)")
    print(f"  never-break share of cell loss: {dom_loss_never/dom_loss:.4f}  "
          f"(weight {dom_den_never/dom_den:.4f})")
    print(f"  pre-break share of cell loss:   {dom_loss_pre/dom_loss:.4f}  "
          f"(weight {dom_den_pre/dom_den:.4f})")

    # ---- F. capture requirement for leaderboard targets -------------------
    EXTERNAL_ANCHOR = 0.6268
    targets = [0.630, 0.635, 0.640, 0.645, 0.650]
    print(f"\n=== F. CAPTURE REQUIRED (external anchor {EXTERNAL_ANCHOR}, dev pooled {total_ts_auc:.5f}) ===")
    cap_rows = {}
    for target in targets:
        # required TS-AUC gain on the EXTERNAL scale, translated 1:1 onto the dev pool
        # (internal->external transfer measured flat-to-slightly-positive per HANDOFF_WAVE6)
        needed_gain = target - EXTERNAL_ANCHOR
        needed_dev_ts_auc = total_ts_auc + needed_gain
        needed_num = needed_dev_ts_auc * total_den
        needed_repair = needed_num - total_num  # additional "num" needed, in pair units
        frac_of_cell = needed_repair / dom_loss if dom_loss > 0 else float("inf")
        cap_rows[f"{target:.3f}"] = {
            "needed_gain": needed_gain, "needed_dev_ts_auc": needed_dev_ts_auc,
            "needed_repair_pairunits": needed_repair,
            "fraction_of_dominant_cell_inversions": frac_of_cell,
            "feasible_within_dominant_cell": bool(0 <= frac_of_cell <= 1),
        }
        print(f"  {target:.3f}: needs +{needed_gain:.5f} TS-AUC = repairing "
              f"{100*frac_of_cell:.2f}% of the dominant cell's inversions"
              f"{'  [EXCEEDS CELL CEILING]' if frac_of_cell > 1 else ''}")

    # ---- write outputs ------------------------------------------------
    out = {
        "experiment": "W7-D0", "purpose": "exact RT-600 pairwise inversion loss cube",
        "source_oof": "research/oof/wave5_S_specialist.npy",
        "reproduction": {
            "expected_mean": EXPECTED_MEAN, "observed_mean": mean, "delta": delta_mean,
            "expected_per_fold": EXPECTED_PER_FOLD, "observed_per_fold": per_fold,
            "delta_per_fold": delta_fold, "tolerance": TOL, "pass": passed,
        },
        "pooled": {
            "ts_auc": total_ts_auc, "total_pair_weight": total_den,
            "total_inversion_loss": total_loss, "loss_fraction": total_loss/total_den,
        },
        "t_buckets": T_LABELS, "age_buckets": AGE_LABELS, "neg_types": NEG_TYPES,
        "cube": cube,
        "cube_check": {"grand_pair_weight": grand_den, "grand_loss": grand_loss,
                        "matches_pooled_weight": bool(abs(grand_den-total_den) < 1.0),
                        "matches_pooled_loss": bool(abs(grand_loss-total_loss) < 1.0)},
        "dominant_cell_t200_age100": {
            "fraction_of_pair_weight": frac_weight, "fraction_of_inversion_loss": frac_loss,
            "cell_auc": dom_auc, "perfect_repair_ceiling_ts_auc": ceiling_ts_auc,
            "never_break": {"pair_weight": dom_den_never, "loss": dom_loss_never,
                             "loss_share": dom_loss_never/dom_loss, "weight_share": dom_den_never/dom_den},
            "pre_break": {"pair_weight": dom_den_pre, "loss": dom_loss_pre,
                          "loss_share": dom_loss_pre/dom_loss, "weight_share": dom_den_pre/dom_den},
        },
        "claim_45_7_pct": {
            "prior_inferred_value": 0.457, "exact_measured_value": frac_loss,
            "verdict": ("CONFIRMED" if abs(frac_loss-0.457) < 0.02 else
                        "APPROXIMATELY CONFIRMED" if abs(frac_loss-0.457) < 0.08 else
                        "REJECTED"),
        },
        "capture_requirements": {"external_anchor": EXTERNAL_ANCHOR, "targets": cap_rows},
        "cell_rows": cell_rows,
        "runtime_s": time.time() - t0,
    }
    with open(f"{REPORTS}/wave7_rt600_exact_alpha_budget.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote research/reports/wave7_rt600_exact_alpha_budget.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
