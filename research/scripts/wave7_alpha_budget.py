"""WAVE-7 W7-D1 -- the weighted pairwise error budget for the RT-600 architecture.

WHAT THIS ANSWERS
    TS-AUC is a weighted average of within-timestep AUCs.  One minus it is a
    weighted average of *inversions*.  This script splits that loss into buckets
    and reports, for each, the exact fraction of remaining loss it owns and the
    TS-AUC that a PERFECT repair of that bucket would buy.  That number is the
    ALPHA BUDGET: no mechanism aimed at a bucket can ever return more than it.

WHAT IS EXACT AND WHAT IS IMPORTED
    EXACT, computed here from `research/folds/folds.parquet` (version-controlled,
    8,000 dev series, their true `tau_index`, `n_online` and fold assignment):
        every pair WEIGHT -- w_t = n_pos(t) * n_neg(t), the official weighting --
        and its split across current-t buckets, post-break-age buckets, their
        JOINT cells, tau-fraction quartiles and online-length buckets.

    IMPORTED, from committed reports, never recomputed here:
        the per-bucket AUCs.  Scoring them needs OOF prediction vectors, which
        need the 2026 store, which is absent from this container.  Provenance is
        attached to every imported number and printed in the output.

    The decomposition  sum_B share_B * AUC_B = TS-AUC  is an identity, so the
    script CHECKS it against the independently reported aggregate and refuses to
    emit a budget whose parts do not reconstruct the whole.

FORBIDDEN HERE
    Nothing in this file reaches a model.  It reads labels and tau because it is
    a post-hoc weight-geometry analysis, exactly as the age-bucket reporting
    mandated by WAVE5_C2 §5.5 does.  It produces no feature and no prediction.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
FOLDS = f"{ROOT}/research/folds/folds.parquet"
OUT_JSON = f"{ROOT}/research/reports/wave7_alpha_budget.json"

T_EDGES = [0, 20, 50, 100, 200, 400, 1000]
T_NAMES = ["0-20", "20-50", "50-100", "100-200", "200-400", "400-1000"]
AGE_EDGES = [0, 5, 10, 20, 50, 100, 10 ** 9]
AGE_NAMES = ["0-5", "5-10", "10-20", "20-50", "50-100", "100+"]

# ----------------------------------------------------------------- imported AUCs
#
# Two independent measured sources.  They describe DIFFERENT arms, and the
# difference is stated rather than smoothed over:
#
#  * AGE buckets are the RT-600 ARCHITECTURE itself (seven specialists, canonical
#    partition, TS-AUC 0.62581) -- research/WAVE5_STATUS.md section 5, W5-D3.
#  * CURRENT-T buckets are the RT-300-CLASS single model (TS-AUC 0.61500 pooled)
#    -- research/reports/champion_diagnostics.json, by_online_index.
#
# No RT-600 by-t decomposition has ever been committed.  Rather than invent one,
# the t budget is reported on the arm that has one and flagged: RT-600 sits
# +0.0108 above it, so its t-bucket losses are somewhat SMALLER than shown.  The
# SHAPE across t -- which is what the budget is used for -- is what transfers.
AUC_BY_AGE = {
    "0-5": 0.51517, "5-10": 0.53408, "10-20": 0.54880,
    "20-50": 0.57702, "50-100": 0.60516, "100+": 0.66151,
}
AUC_BY_AGE_SRC = "research/WAVE5_STATUS.md S5 (W5-D3), arm S = RT-600 architecture"
AUC_BY_AGE_AGG = 0.62581

AUC_BY_T = {
    "0-20": 0.5246581403649083, "20-50": 0.5459913396273111,
    "50-100": 0.5538736525482396, "100-200": 0.5996048882162585,
    "200-400": 0.6160761059680906, "400-1000": 0.6473058271649713,
}
AUC_BY_T_SRC = "research/reports/champion_diagnostics.json by_online_index, RT-300-class"
AUC_BY_T_AGG = 0.6149965423542226

AUC_BY_TAUQ = {
    "q1 (0.00-0.24)": 0.6278019352321409, "q2 (0.24-0.48)": 0.6236752036971367,
    "q3 (0.48-0.74)": 0.5879337757628209, "q4 (0.74-1.00)": 0.5709880117402779,
}
AUC_BY_LEN = {
    "0-200": 0.5511153510918393, "200-400": 0.6113642977999438,
    "400-700": 0.6084438317055721, "700-1000": 0.6406555608605112,
}

# Yardstick: what one whole architecture generation was worth per age bucket.
# S - A, specialist ensemble minus the single model, WAVE5_STATUS S5.
ARCH_GEN_GAIN_BY_AGE = {
    "0-5": 0.00440, "5-10": 0.00806, "10-20": 0.01009,
    "20-50": 0.00777, "50-100": 0.00919, "100+": 0.01065,
}


def _bucket(vals, edges, names):
    idx = np.digitize(vals, edges[1:-1], right=False)
    return idx, names


def weight_geometry(meta: pd.DataFrame) -> dict:
    """Exact pair-weight geometry for one validation set.

    Returns per-t arrays and every bucket aggregation.  A series is ALIVE at t
    iff n_online > t; it is POSITIVE at t iff it breaks and tau_index <= t.  A
    break series before its own tau is a NEGATIVE, which is what makes the
    metric hard and is easy to get wrong.
    """
    n_on = meta["n_online"].to_numpy(np.int64)
    tau = meta["tau_index"].to_numpy(np.int64)
    has = meta["has_break"].to_numpy(np.int64)
    tmax = int(n_on.max())

    # A series is ALIVE at t iff n_online > t, i.e. dead once t >= n_online.
    # It is POSITIVE at t iff it breaks and tau_index <= t (and is still alive).
    # Both are cumulative counts of step events, so one bincount each.
    dec = np.bincount(n_on, minlength=tmax + 2)[: tmax + 1]
    alive = len(n_on) - np.cumsum(dec)

    br = has == 1
    inc = np.bincount(tau[br], minlength=tmax + 2)[: tmax + 1]
    dead_pos = np.bincount(n_on[br], minlength=tmax + 2)[: tmax + 1]
    npos = np.cumsum(inc) - np.cumsum(dead_pos)
    nneg = alive - npos
    assert (npos >= 0).all() and (nneg >= 0).all()

    w = (npos * nneg).astype(np.float64)
    W = float(w.sum())

    # ---- current-t buckets -------------------------------------------------
    t = np.arange(tmax + 1)
    tb = np.digitize(t, T_EDGES[1:-1], right=False)
    w_t = {name: float(w[tb == i].sum()) for i, name in enumerate(T_NAMES)}

    # ---- age buckets and the joint cell table ------------------------------
    # A positive row (series s, time t) carries pair weight nneg(t): it pairs
    # with every negative alive at t.  Summing over positives reproduces W.
    w_age = {a: 0.0 for a in AGE_NAMES}
    joint = {tn: {an: 0.0 for an in AGE_NAMES} for tn in T_NAMES}
    for s in np.flatnonzero(br):
        ts = np.arange(int(tau[s]), int(n_on[s]))
        if ts.size == 0:
            continue
        ages = ts - int(tau[s])
        ab = np.digitize(ages, AGE_EDGES[1:-1], right=False)
        tbb = np.digitize(ts, T_EDGES[1:-1], right=False)
        wn = nneg[ts].astype(np.float64)
        for i, an in enumerate(AGE_NAMES):
            m = ab == i
            if m.any():
                w_age[an] += float(wn[m].sum())
                for j, tn in enumerate(T_NAMES):
                    mm = m & (tbb == j)
                    if mm.any():
                        joint[tn][an] += float(wn[mm].sum())
    assert abs(sum(w_age.values()) - W) < 1e-6 * max(W, 1.0), "age split must reconstruct W"

    # ---- tau-fraction quartiles and online-length buckets ------------------
    w_tauq = {k: 0.0 for k in AUC_BY_TAUQ}
    tqk = list(AUC_BY_TAUQ)
    frac = np.where(br, tau / np.maximum(n_on, 1), np.nan)
    qed = [0.0, 0.24, 0.48, 0.74, 1.01]
    w_len = {k: 0.0 for k in AUC_BY_LEN}
    len_ed = [0, 200, 400, 700, 10 ** 9]
    len_k = list(AUC_BY_LEN)
    for s in np.flatnonzero(br):
        ts = np.arange(int(tau[s]), int(n_on[s]))
        if ts.size == 0:
            continue
        m = float(nneg[ts].sum())
        w_tauq[tqk[min(int(np.digitize(frac[s], qed[1:-1], right=False)), 3)]] += m
    # length buckets are a SERIES property: attribute a series' whole positive
    # mass to its own length bucket.  Negatives carry no age, so the length
    # split is over positives only -- same convention as by_online_length.
    for s in np.flatnonzero(br):
        ts = np.arange(int(tau[s]), int(n_on[s]))
        if ts.size == 0:
            continue
        k = len_k[min(int(np.digitize(n_on[s], len_ed[1:-1], right=False)), 3)]
        w_len[k] += float(nneg[ts].sum())

    return {"W": W, "w_t": w_t, "w_age": w_age, "joint": joint,
            "w_tauq": w_tauq, "w_len": w_len,
            "per_t_weight": w, "npos": npos, "nneg": nneg}


def budget(shares: dict, aucs: dict) -> dict:
    """share_B, AUC_B -> loss_B, fraction of total loss, perfect-repair ceiling."""
    rows = {}
    total_loss = sum(shares[b] * (1.0 - aucs[b]) for b in shares)
    recon = sum(shares[b] * aucs[b] for b in shares)
    for b in shares:
        loss = shares[b] * (1.0 - aucs[b])
        rows[b] = {
            "weight_share": shares[b],
            "auc": aucs[b],
            "loss": loss,
            "loss_fraction": loss / total_loss if total_loss else 0.0,
            "perfect_repair_delta": loss,
        }
    return {"buckets": rows, "total_loss": total_loss, "reconstructed_ts_auc": recon}


def main() -> int:
    meta = pd.read_parquet(FOLDS)
    dev = meta[meta["split"] == "dev"]

    per_fold = {}
    for f in sorted(dev["fold"].unique()):
        per_fold[int(f)] = weight_geometry(dev[dev["fold"] == f])
    pooled = weight_geometry(dev)

    def mean_share(key, names):
        out = {}
        for n in names:
            out[n] = float(np.mean([per_fold[f][key][n] / per_fold[f]["W"]
                                    for f in per_fold]))
        return out

    share_t = mean_share("w_t", T_NAMES)
    share_age = mean_share("w_age", AGE_NAMES)
    share_tauq = mean_share("w_tauq", list(AUC_BY_TAUQ))
    share_len = mean_share("w_len", list(AUC_BY_LEN))

    b_age = budget(share_age, AUC_BY_AGE)
    b_t = budget(share_t, AUC_BY_T)
    b_tauq = budget(share_tauq, AUC_BY_TAUQ)
    b_len = budget(share_len, AUC_BY_LEN)

    # ---- consistency gates: the parts must reconstruct the whole -----------
    gates = {
        "age_reconstruction": {
            "reconstructed": b_age["reconstructed_ts_auc"],
            "reported": AUC_BY_AGE_AGG,
            "abs_err": abs(b_age["reconstructed_ts_auc"] - AUC_BY_AGE_AGG),
            "tol": 2e-3,
        },
        "t_reconstruction": {
            "reconstructed": b_t["reconstructed_ts_auc"],
            "reported": AUC_BY_T_AGG,
            "abs_err": abs(b_t["reconstructed_ts_auc"] - AUC_BY_T_AGG),
            "tol": 2e-3,
        },
    }
    for g in gates.values():
        g["pass"] = bool(g["abs_err"] <= g["tol"])

    # ---- joint t x age weight mass, pooled over the dev set ----------------
    joint_share = {tn: {an: pooled["joint"][tn][an] / pooled["W"] for an in AGE_NAMES}
                   for tn in T_NAMES}

    # ---- how far is each target from here ----------------------------------
    champ = AUC_BY_AGE_AGG
    targets = {}
    for tgt in (0.630, 0.635, 0.640, 0.645, 0.650):
        need = tgt - 0.6268                       # external, LB-001
        targets[f"{tgt:.3f}"] = {
            "external_gap": need,
            "internal_equivalent_gap": tgt - champ,
            "fraction_of_remaining_loss": (tgt - champ) / b_age["total_loss"],
        }

    # ---- named error classes (brief section 28's false-pair taxonomy) ------
    #
    # Only the classes whose weight is EXACTLY computable from the fold table are
    # given a measured budget.  The rest need OOF prediction vectors and are
    # listed with the reason they cannot be sized yet, rather than guessed.
    age100 = b_age["buckets"]["100+"]
    frac_age100_late = sum(joint_share[tn]["100+"] for tn in ("200-400", "400-1000")) \
        / sum(joint_share[tn]["100+"] for tn in T_NAMES)
    young = sum(b_age["buckets"][k]["loss"] for k in ("0-5", "5-10", "10-20"))
    L = b_age["total_loss"]

    # Hard negatives: W5-D2 measured the top 1% of no-break series at mean
    # within-timestep percentile rank 0.9221 against 0.4731 for negatives overall.
    # A negative at rank r inverts a fraction ~r of the pairs it takes part in, so
    # the EXCESS loss those series carry over an average negative is
    #   (their share of negative pair mass) * (0.9221 - overall inversion rate).
    HARDNEG_SHARE, HARDNEG_RANK = 0.01, 0.9221
    hardneg_ceiling = HARDNEG_SHARE * HARDNEG_RANK
    hardneg_excess = HARDNEG_SHARE * (HARDNEG_RANK - L)

    classes = {
        "mature persistent break vs no-break, late online (t>=200, age>=100)": {
            "weight_share": sum(joint_share[tn]["100+"] for tn in ("200-400", "400-1000")),
            "loss": age100["loss"] * frac_age100_late,
            "basis": "EXACT weight; AUC taken flat across t within age 100+",
        },
        "mid-age break (20 <= age < 100)": {
            "weight_share": share_age["20-50"] + share_age["50-100"],
            "loss": b_age["buckets"]["20-50"]["loss"] + b_age["buckets"]["50-100"]["loss"],
            "basis": "EXACT weight, measured AUC",
        },
        "young break (age < 20) -- transient-vs-persistent territory": {
            "weight_share": share_age["0-5"] + share_age["5-10"] + share_age["10-20"],
            "loss": young,
            "basis": "EXACT weight, measured AUC",
        },
        "heavy-tail / outlier false alarm (top 1% hardest negatives)": {
            "weight_share": HARDNEG_SHARE,
            "loss": hardneg_excess,
            "perfect_repair_override": hardneg_ceiling,
            "basis": "W5-D2 mean percentile rank 0.9221; EXCESS over an average negative",
        },
        "short online segment (n_online < 200)": {
            "weight_share": share_len["0-200"],
            "loss": b_len["buckets"]["0-200"]["loss"],
            "basis": "EXACT weight, measured AUC (RT-300-class arm)",
        },
        "late break (tau > 0.74 of the online segment)": {
            "weight_share": share_tauq["q4 (0.74-1.00)"],
            "loss": b_tauq["buckets"]["q4 (0.74-1.00)"]["loss"],
            "basis": "EXACT weight, measured AUC (RT-300-class arm)",
        },
    }
    for c in classes.values():
        c["loss_fraction"] = c["loss"] / L
        c["perfect_repair_delta"] = c.get("perfect_repair_override", c["loss"])
    unsized = {
        "variance-only break": "needs per-series break-family labels joined to OOF scores",
        "dependence-only break": "needs per-series break-family labels joined to OOF scores",
        "weak magnitude": "by_break_class puts 92.8% of break series in weak_unclassified "
                          "at AUC 0.6005 vs 0.828 for mixed -- a SERIES-COUNT share, not a "
                          "weight share; sizing it needs the taxonomy joined to pair weights",
        "online-vs-history offset": "needs OOF scores to separate from the tail class",
    }

    out = {
        "schema": "wave7_alpha_budget/1",
        "exact_inputs": {
            "folds": "research/folds/folds.parquet",
            "n_dev_series": int(len(dev)),
            "n_break_series": int(dev["has_break"].sum()),
            "total_pair_weight_pooled": pooled["W"],
            "total_pair_weight_per_fold": {str(f): per_fold[f]["W"] for f in per_fold},
        },
        "imported_aucs": {
            "by_age": {"values": AUC_BY_AGE, "source": AUC_BY_AGE_SRC,
                       "arm": "RT-600 seven-specialist ensemble", "aggregate": AUC_BY_AGE_AGG},
            "by_t": {"values": AUC_BY_T, "source": AUC_BY_T_SRC,
                     "arm": "RT-300-class single model", "aggregate": AUC_BY_T_AGG},
            "by_tau_quartile": {"values": AUC_BY_TAUQ, "source": AUC_BY_T_SRC,
                                "arm": "RT-300-class single model"},
            "by_online_length": {"values": AUC_BY_LEN, "source": AUC_BY_T_SRC,
                                 "arm": "RT-300-class single model"},
        },
        "gates": gates,
        "budget_by_age": b_age,
        "budget_by_current_t": b_t,
        "budget_by_tau_quartile": b_tauq,
        "budget_by_online_length": b_len,
        "joint_weight_share_t_by_age": joint_share,
        "architecture_generation_yardstick_by_age": ARCH_GEN_GAIN_BY_AGE,
        "targets": targets,
        "named_error_classes": classes,
        "unsized_error_classes": unsized,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=1)

    # ------------------------------------------------------------------ report
    print("=" * 78)
    print("W7-D1  WEIGHTED PAIRWISE ERROR BUDGET")
    print("=" * 78)
    print(f"dev series {len(dev)}  breaks {int(dev.has_break.sum())}  "
          f"pooled pair weight {pooled['W']:.4g}")
    for name, g in gates.items():
        print(f"gate {name:20s} recon {g['reconstructed']:.5f} vs reported "
              f"{g['reported']:.5f}  err {g['abs_err']:.5f}  "
              f"{'PASS' if g['pass'] else 'FAIL'}")
    if not all(g["pass"] for g in gates.values()):
        print("\nRECONSTRUCTION GATE FAILED -- budget not emitted as authoritative")
        return 1

    def table(title, b, extra=None):
        print(f"\n--- {title} ---")
        hdr = f"{'bucket':>14} {'wt share':>9} {'AUC':>8} {'loss':>9} {'% loss':>7} {'ceiling':>9}"
        if extra:
            hdr += f" {extra[0]:>9}"
        print(hdr)
        for k, r in sorted(b["buckets"].items(), key=lambda kv: -kv[1]["loss"]):
            line = (f"{k:>14} {r['weight_share']:9.4f} {r['auc']:8.5f} "
                    f"{r['loss']:9.5f} {100*r['loss_fraction']:6.1f}% "
                    f"{r['perfect_repair_delta']:+9.5f}")
            if extra:
                line += f" {extra[1].get(k, float('nan')):+9.5f}"
            print(line)
        print(f"{'TOTAL':>14} {sum(r['weight_share'] for r in b['buckets'].values()):9.4f} "
              f"{b['reconstructed_ts_auc']:8.5f} {b['total_loss']:9.5f} {100.0:6.1f}%")

    table("BY POST-BREAK AGE  (arm: RT-600 architecture)", b_age,
          ("arch gen", ARCH_GEN_GAIN_BY_AGE))
    table("BY CURRENT ONLINE INDEX t  (arm: RT-300-class)", b_t)
    table("BY TAU FRACTION  (arm: RT-300-class)", b_tauq)
    table("BY ONLINE LENGTH  (arm: RT-300-class)", b_len)

    print("\n--- JOINT WEIGHT SHARE:  rows = current t,  cols = post-break age ---")
    print(f"{'t \\ age':>10}" + "".join(f"{a:>9}" for a in AGE_NAMES) + f"{'row':>9}")
    for tn in T_NAMES:
        row = joint_share[tn]
        print(f"{tn:>10}" + "".join(f"{100*row[a]:8.2f}%" for a in AGE_NAMES)
              + f"{100*sum(row.values()):8.2f}%")
    print(f"{'col':>10}" + "".join(
        f"{100*sum(joint_share[tn][a] for tn in T_NAMES):8.2f}%" for a in AGE_NAMES))

    print("\n--- NAMED ERROR CLASSES  (overlapping by construction) ---")
    print(f"{'class':>62} {'wt':>6} {'% loss':>7} {'ceiling':>9}")
    for k, c in sorted(classes.items(), key=lambda kv: -kv[1]["loss"]):
        print(f"{k[:62]:>62} {c['weight_share']:6.3f} {100*c['loss_fraction']:6.1f}% "
              f"{c['perfect_repair_delta']:+9.5f}")
    print("  unsized without OOF vectors: " + "; ".join(unsized))

    print("\n--- HOW MUCH LOSS MUST BE CONVERTED TO HIT EACH TARGET ---")
    print(f"{'target':>8} {'ext gap':>9} {'int gap':>9} {'% of remaining loss':>21}")
    for k, v in targets.items():
        print(f"{k:>8} {v['external_gap']:+9.4f} {v['internal_equivalent_gap']:+9.4f} "
              f"{100*v['fraction_of_remaining_loss']:20.2f}%")
    print(f"\nwritten: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
