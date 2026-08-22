"""W6-E1: is the calibration's resolution in the wrong place?

Pre-registered in research/WAVE6_PREREG.md section 3 BEFORE this ran.

The shipped calibration puts 12 log-spaced anchors at
1, 2, 4, 7, 12, 23, 43, 81, 152, 285, 533, 999.  Measured against the official
n_pos(t)*n_neg(t) weight: the first SEVEN span 1.05% of it, NINE of twelve sit
at t <= 168 spanning the first 25%, and exactly ONE lies inside t in (168, 451],
which carries 50%.

Three schemes, declared in advance and not extended, plus the NULL-ANCHOR
control -- five random re-placements whose MEAN (not best) is the control value.

LEAKAGE.  The pair-weight distribution is a function of the labels, so anchor
placement is model fitting and must be cross-fitted exactly as the grids already
are: fold k's anchors come from the pair weights of folds != k only.  A single
global placement would leak fold k's labels into fold k's ranking.
`_assert_fold_pure` checks it rather than trusting it.

No training.  Runs on OOF vectors already on disk.
"""
from __future__ import annotations

import json, sys, time
import numpy as np

from wave5_lib import Ctx, load_oof, SPECIALISTS, FOLDS, REPORTS, fmt
from wave4_cal import SCDF_NSEEN
from wave6_cal import (SCDF_ANCH, anchors_log, anchors_weight, anchors_hybrid,
                       anchors_null)

NULL_SEEDS = [0, 1, 7, 42, 2026]          # fixed in the pre-registration
SHIPPED = 0.62581                          # S under the shipped calibration


def _assert_fold_pure(rows_used, k, c):
    """Fold k must not contribute a single row to its own anchor placement."""
    own = c.rows[k]
    if np.intersect1d(rows_used, own, assume_unique=False).size:
        raise RuntimeError(f"ANCHOR LEAK: fold {k}'s rows entered its own placement")


def blend_under(c, P, streams, anchor_fn, tag):
    """Equal-weight blend of cross-fitted SCDF_ANCH calibrations."""
    out = np.full(len(c.d.y), np.nan)
    fallbacks, anchors_by_fold = 0, {}
    for k in FOLDS:
        tr = np.concatenate([c.rows[g] for g in FOLDS if g != k])
        va = c.rows[k]
        _assert_fold_pure(tr, k, c)
        a = anchor_fn(c.d.t[tr], c.d.y[tr])
        anchors_by_fold[k] = [int(x) for x in a]
        cols = []
        for s in streams:
            f = SCDF_ANCH(P[s][tr], c.d.t[tr], a)
            fallbacks += f.n_fallback
            cols.append(f(P[s][va], c.d.t[va]))
        out[va] = np.column_stack(cols).mean(1)
    return out, anchors_by_fold, fallbacks


def main():
    t0 = time.time()
    c = Ctx()
    P = load_oof(SPECIALISTS)
    res = {"experiment": "W6-E1", "streams": SPECIALISTS,
           "null_seeds": NULL_SEEDS, "schemes": {}}

    # ---- parity: the shipped calibration, as the comparison's zero ---------
    ship = c.crossfit_blend(P, SPECIALISTS, cal=SCDF_NSEEN)
    ship_m, ship_pf = c.score(ship)
    print(f"shipped SCDF_NSEEN                {ship_m:.5f}   {fmt(ship_pf)}")
    res["shipped"] = {"mean": ship_m, "per_fold": ship_pf}

    schemes = [("CAL-LOG", anchors_log), ("CAL-WT", anchors_weight),
               ("CAL-HYB", anchors_hybrid)]
    for i, sd in enumerate(NULL_SEEDS):
        schemes.append((f"NULL-{sd}", lambda t, y, s=sd: anchors_null(t, y, s)))

    vecs = {}
    for name, fn in schemes:
        v, anch, fb = blend_under(c, P, SPECIALISTS, fn, name)
        m, pf = c.score(v)
        vecs[name] = v
        res["schemes"][name] = {"mean": m, "per_fold": pf,
                                "anchors_fold0": anch[0], "grid_fallbacks": fb}
        print(f"  {name:10s} {m:.5f}   {fmt(pf)}   fallback grids {fb:3d}   "
              f"anchors(f0) {anch[0]}")

    # ---- the parity check the design depends on ---------------------------
    log_m = res["schemes"]["CAL-LOG"]["mean"]
    res["parity"] = {"cal_log": log_m, "shipped": ship_m, "diff": log_m - ship_m}
    print(f"\nPARITY  CAL-LOG {log_m:.5f} vs shipped {ship_m:.5f}  "
          f"diff {log_m-ship_m:+.6f}")
    if abs(log_m - ship_m) > 5e-5:
        print("  WARNING: the generalisation is NOT neutral; deltas below mix "
              "placement with windowing and must be read as such.")

    # ---- verdict against the pre-registered bar ---------------------------
    base = res["schemes"]["CAL-LOG"]
    nulls = [res["schemes"][f"NULL-{s}"]["mean"] for s in NULL_SEEDS]
    null_mean = float(np.mean(nulls))
    print(f"\nNULL-ANCHOR draws: " + " ".join(f"{x:.5f}" for x in nulls))
    print(f"NULL-ANCHOR mean (the control value): {null_mean:.5f}  "
          f"({null_mean-base['mean']:+.5f} vs CAL-LOG)")

    best, best_d = None, -9
    print()
    for name in ("CAL-WT", "CAL-HYB"):
        a = res["schemes"][name]
        d = [x - y for x, y in zip(a["per_fold"], base["per_fold"])]
        nb = int(sum(x > 0 for x in d))
        a["delta_vs_CAL_LOG"] = a["mean"] - base["mean"]
        a["n_folds_better"] = nb
        a["delta_vs_null_mean"] = a["mean"] - null_mean
        print(f"  {name}: vs CAL-LOG {a['mean']-base['mean']:+.5f} ({nb}/5)   "
              f"vs NULL mean {a['mean']-null_mean:+.5f}")
        if a["mean"] - base["mean"] > best_d:
            best, best_d = name, a["mean"] - base["mean"]

    a = res["schemes"][best]
    passed = (best_d >= 0.0030 and a["n_folds_better"] >= 4
              and a["delta_vs_null_mean"] >= 0.0020)
    res["verdict"] = {
        "best_scheme": best, "delta_vs_CAL_LOG": best_d,
        "n_folds_better": a["n_folds_better"],
        "null_anchor_mean": null_mean,
        "delta_vs_null_mean": a["delta_vs_null_mean"],
        "prereg_bar": {"delta": 0.0030, "min_folds": 4, "over_null": 0.0020},
        "screening_threshold": 0.0010,
        "inside_noise": bool(best_d < 0.0010),
        "outcome": ("PROMOTE" if passed else
                    "EXPLORATORY" if best_d >= 0.0010 else
                    "INSIDE NOISE -- counts toward the section-0 stopping rule"),
    }
    print(f"\nBEST {best}: {best_d:+.5f} vs CAL-LOG on {a['n_folds_better']}/5  "
          f"-> {res['verdict']['outcome']}")

    if best_d >= 0.0010:
        print("\n-- paired series bootstrap (200 reps, common random numbers) --")
        res["bootstrap"] = c.bootstrap(
            {"best": vecs[best], "log": vecs["CAL-LOG"], "null": vecs[f"NULL-{NULL_SEEDS[0]}"]},
            {"best_minus_log": ("best", "log"), "best_minus_null0": ("best", "null")},
            n=200, seed=0)
        for k, v in res["bootstrap"].items():
            print(f"    {k:18s} {v['mean']:+.5f}  CI [{v['ci95'][0]:+.5f}, "
                  f"{v['ci95'][1]:+.5f}]  {100*v['fraction_positive']:.0f}% positive")

    json.dump(res, open(f"{REPORTS}/wave6_e1_anchors.json", "w"), indent=2, default=float)
    print(f"\nwrote research/reports/wave6_e1_anchors.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
