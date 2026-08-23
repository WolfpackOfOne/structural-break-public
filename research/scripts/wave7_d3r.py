"""W7-D3R: same-prefix vs full-sequence three-arm diagnostic on the W7-D0
dominant cell (current t >= 200, positive post-break age >= 100).

Pre-registered in research/WAVE7_D3R_PREREG.md BEFORE this ran. Population,
arm definitions, thresholds and the interpretation matrix are fixed there and
are not re-derived here.

Arm A -- RT-300, reused, not retrained (legal causal prefix baseline).
Arm B -- RT-990, new training: same 500 legal causal columns/rows, stronger
          tree capacity only.
Arm C -- RT-991, new training: Arm B's capacity plus 500 more columns holding
          each row's own series' feature vector at that series' FINAL online
          row (full training-sequence information, no true tau, no explicit
          boundary, no true age). OFFLINE DIAGNOSTIC ONLY -- not causal, never
          a production candidate.

Usage:
    python wave7_d3r.py --arm b        # train RT-990
    python wave7_d3r.py --arm c        # train RT-991
    python wave7_d3r.py --analyze      # score A/B/C on the fixed cell, verdict
"""
from __future__ import annotations

import argparse, json, time

import numpy as np

from wave5_lib import Ctx, FOLDS, REPORTS, OOFDIR, ts_auc_flat
import sbr.pipeline as PL

FULL = ["m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes"]

ARM_B_PARAMS = dict(objective="binary", n_estimators=900, learning_rate=0.05,
                     num_leaves=127, min_data_in_leaf=150, feature_fraction=1.0,
                     bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                     max_bin=127, num_threads=2)

MAX_TRAIN_ROWS = 1_000_000


def last_row_lookup(d):
    """row index of each series' final online observation, robust to storage order."""
    sidx = d.sidx
    n_series = int(sidx.max()) + 1
    order = np.argsort(sidx, kind="stable")
    s_sorted = sidx[order]
    b = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])
    e = np.r_[b[1:], len(s_sorted)]
    last_row = np.zeros(n_series, dtype=np.int64)
    for lo, hi in zip(b, e):
        grp = order[lo:hi]
        last_row[s_sorted[lo]] = grp[np.argmax(d.t[grp])]
    assert np.all(d.t[last_row] == np.array(
        [d.t[order[lo:hi]].max() for lo, hi in zip(b, e)]))
    return last_row


def augmented_stack(mats, names, keep_idx, rows, final_of_row):
    """final_of_row: length-n_rows array, final_of_row[r] = row index of r's
    series' last online observation."""
    own = PL._stack(mats, names, rows, keep_idx)
    frows = final_of_row[rows]
    uniq, inv = np.unique(frows, return_inverse=True)
    fut_u = PL._stack(mats, names, uniq, keep_idx)
    fut = fut_u[inv]
    return np.concatenate([own, fut], axis=1)


def train_arm_c(exp_id="RT-991"):
    d = PL.Data()
    mats, names = PL.load_features(FULL)
    keep_idx = np.arange(len(names))
    last_row_by_series = last_row_lookup(d)
    final_of_row = last_row_by_series[d.sidx]

    rng = np.random.default_rng(0)
    p = dict(ARM_B_PARAMS)
    n_round = p.pop("n_estimators")
    oof = np.full(len(d.y), np.nan, dtype=np.float32)
    per_fold = []
    t_start = time.time()

    import lightgbm as lgb
    for f in FOLDS:
        tr_folds = [x for x in (0, 1, 2, 3, 4) if x != f]
        tr_rows = d.rows_for(tr_folds)
        va_rows = d.rows_for([f])
        if len(tr_rows) > MAX_TRAIN_ROWS:
            tr_rows = np.sort(rng.choice(tr_rows, MAX_TRAIN_ROWS, replace=False))

        Xtr = augmented_stack(mats, names, keep_idx, tr_rows, final_of_row)
        ytr = d.y[tr_rows]
        ds = lgb.Dataset(Xtr, label=ytr, params=dict(p, objective="binary"),
                         feature_name=[f"f{i}" for i in range(Xtr.shape[1])])
        booster = lgb.train(dict(p), ds, num_boost_round=n_round)
        del Xtr, ds

        Xva = augmented_stack(mats, names, keep_idx, va_rows, final_of_row)
        n_cols = Xva.shape[1]
        pred = booster.predict(Xva).astype(np.float32)
        del Xva
        oof[va_rows] = pred
        s = ts_auc_flat(pred, d.y[va_rows], d.t[va_rows])
        per_fold.append(float(s))
        print(f"  [C] fold {f}: TS-AUC {s:.5f}  ({len(tr_rows)} train rows, "
              f"{len(va_rows)} valid rows, {n_cols} cols)", flush=True)

    dev_rows = d.rows_for(list(FOLDS))
    overall = ts_auc_flat(oof[dev_rows], d.y[dev_rows], d.t[dev_rows])
    np.save(f"{OOFDIR}/{exp_id}.npy", oof)

    res = dict(
        experiment_id=exp_id, date=time.strftime("%Y-%m-%d %H:%M"), git_sha=PL.git_sha(),
        agent="agent0",
        hypothesis="W7-D3R Arm C: Arm B capacity + own-series final-row broadcast (full "
                   "training-sequence info, no true tau, no explicit boundary, no true age)",
        falsification_condition="see research/WAVE7_D3R_PREREG.md",
        feature_set=",".join(FULL) + "+final_row_broadcast", n_features=1000,
        model="lgbm", objective="binary", folds=",".join(map(str, FOLDS)), random_seed=0,
        train_series=int((~np.isin(d.series_fold, [-1])).sum()), train_rows=MAX_TRAIN_ROWS,
        mean_oof_ts_auc=float(np.mean(per_fold)), pooled_oof_ts_auc=float(overall),
        per_fold_ts_auc=";".join(f"{x:.5f}" for x in per_fold), fold_std=float(np.std(per_fold)),
        persistence="none", sample_mode="uniform",
        training_runtime_s=round(time.time() - t_start, 1),
        causal_verified="OFFLINE DIAGNOSTIC ONLY -- final-row broadcast is NOT causal, NOT deployable",
        test_reduced_touched="no", lockbox_touched="no", status="recorded",
        notes="W7-D3R Arm C. Pre-registered research/WAVE7_D3R_PREREG.md. NEVER a production candidate.",
        protocol="full",
    )
    PL.append_result(res)
    print(json.dumps({k: res[k] for k in
                      ("experiment_id", "mean_oof_ts_auc", "pooled_oof_ts_auc", "per_fold_ts_auc")},
                     indent=2))
    return res


def train_arm_b(exp_id="RT-990"):
    return PL.run(
        exp_id=exp_id, modules=FULL,
        hypothesis="W7-D3R Arm B: same info as RT-300 (500 legal causal cols, same rows), "
                   "stronger tree extraction only (leaves 127, ff 1.0, 900 trees)",
        falsification="see research/WAVE7_D3R_PREREG.md",
        agent="agent0", model="lgbm", params=dict(ARM_B_PARAMS), folds=FOLDS,
        max_train_rows=MAX_TRAIN_ROWS, seed=0,
        notes="W7-D3R Arm B. Pre-registered research/WAVE7_D3R_PREREG.md.",
    )


def cell_mask(y, t, age):
    return (t >= 200) & (((y == 1) & (age >= 100)) | (y == 0))


def score_on(v, mask, y, t):
    if not mask.any():
        return None
    return float(ts_auc_flat(v[mask], y[mask], t[mask]))


def bucket(delta):
    a = abs(delta)
    if a < 0.003:
        return "negligible"
    if a < 0.010:
        return "small/moderate"
    if a < 0.020:
        return "meaningful"
    return "large"


def analyze():
    c = Ctx()
    y, t, age, sidx = c.d.y, c.d.t, c.age, c.d.sidx
    neg_is_prebreak = c.has_break[sidx] & (y == 0)

    dev_mask = np.zeros(len(y), dtype=bool)
    dev_mask[c.dev] = True
    cellmask = cell_mask(y, t, age) & dev_mask
    cellmask_never = cellmask & ((y == 1) | (~neg_is_prebreak))
    cellmask_pre = cellmask & ((y == 1) | neg_is_prebreak)

    fold_bool = {}
    for k in FOLDS:
        fb = np.zeros(len(y), dtype=bool)
        fb[c.rows[k]] = True
        fold_bool[k] = fb

    # exact pair-weight fractions, scored with any finite vector (weight is
    # score-independent, only t/y grouping matters)
    dummy = np.zeros(cellmask.sum())
    _, cell_per = ts_auc_flat(dummy, y[cellmask], t[cellmask], return_per_step=True)
    cell_w = float(cell_per["w"].sum())
    _, dev_per = ts_auc_flat(np.zeros(dev_mask.sum()), y[dev_mask], t[dev_mask], return_per_step=True)
    total_w = float(dev_per["w"].sum())
    frac_weight = cell_w / total_w

    arms = {"A_RT300": "RT-300", "B_RT990": "RT-990", "C_RT991": "RT-991"}
    results = {}
    for label, exp_id in arms.items():
        v = np.load(f"{OOFDIR}/{exp_id}.npy")
        per_fold = {k: score_on(v, cellmask & fold_bool[k], y, t) for k in FOLDS}
        results[label] = {
            "experiment_id": exp_id,
            "pooled_cell_ts_auc": score_on(v, cellmask, y, t),
            "per_fold_cell_ts_auc": per_fold,
            "pooled_cell_ts_auc_never_break_only": score_on(v, cellmask_never, y, t),
            "pooled_cell_ts_auc_pre_break_only": score_on(v, cellmask_pre, y, t),
        }
        print(f"  {label} ({exp_id}): cell TS-AUC {results[label]['pooled_cell_ts_auc']:.5f}  "
              f"(never-break-only {results[label]['pooled_cell_ts_auc_never_break_only']:.5f}, "
              f"pre-break-only {results[label]['pooled_cell_ts_auc_pre_break_only']:.5f})")

    delta_BA = results["B_RT990"]["pooled_cell_ts_auc"] - results["A_RT300"]["pooled_cell_ts_auc"]
    delta_CB = results["C_RT991"]["pooled_cell_ts_auc"] - results["B_RT990"]["pooled_cell_ts_auc"]
    translated_BA = frac_weight * delta_BA
    translated_CB = frac_weight * delta_CB
    size_BA, size_CB = bucket(delta_BA), bucket(delta_CB)

    # Sign-aware: the brief's ">>"/"~=" reading is directional, not just a
    # magnitude bucket -- a NEGATIVE delta (stronger lever, same or worse
    # result) is "no headroom found", i.e. flat, regardless of |delta|. Only
    # a delta clearing +0.010 (meaningful/large) on the POSITIVE side counts
    # as "beats". A positive delta in [+0.003, +0.010) is the genuinely
    # ambiguous small/moderate zone the brief's own scale leaves open.
    def beats(delta):
        return delta >= 0.010

    def flat(delta):
        return delta < 0.003

    if beats(delta_BA) and flat(delta_CB):
        case = "CASE 1 -- representation/extraction limit"
    elif flat(delta_BA) and beats(delta_CB):
        case = "CASE 2 -- future-information limit"
    elif beats(delta_BA) and beats(delta_CB):
        case = "CASE 3 -- both"
    elif flat(delta_BA) and flat(delta_CB):
        case = "CASE 4 -- near-noise / irreducible"
    else:
        case = f"AMBIGUOUS -- B-A is {delta_BA:+.5f} ({size_BA}), C-B is {delta_CB:+.5f} ({size_CB}); no forced case"

    print(f"\n  cell pair-weight fraction of dev: {frac_weight:.4f}")
    print(f"  B - A: {delta_BA:+.5f} cell AUC ({size_BA})  -> translated pooled Δ {translated_BA:+.5f}")
    print(f"  C - B: {delta_CB:+.5f} cell AUC ({size_CB})  -> translated pooled Δ {translated_CB:+.5f}")
    print(f"  VERDICT: {case}")

    out = {
        "experiment": "W7-D3R", "prereg": "research/WAVE7_D3R_PREREG.md",
        "cell_pair_weight_fraction": frac_weight,
        "results": results,
        "delta_B_minus_A": {"cell_auc": delta_BA, "bucket": size_BA, "translated_pooled": translated_BA},
        "delta_C_minus_B": {"cell_auc": delta_CB, "bucket": size_CB, "translated_pooled": translated_CB},
        "verdict": case,
    }
    with open(f"{REPORTS}/wave7_d3r.json", "w") as f:
        json.dump(out, f, indent=2)

    md = [f"# WAVE 7 — W7-D3R RESULTS\n",
          f"Pre-registered: `research/WAVE7_D3R_PREREG.md`. Population: t≥200, "
          f"positive age≥100, all negatives (fixed by W7-D0, not re-derived here).\n",
          f"Cell pair-weight fraction of dev: **{frac_weight:.4f}**\n",
          "## Arms\n",
          "| arm | exp id | pooled cell TS-AUC | never-break-only | pre-break-only |",
          "|---|---|---:|---:|---:|"]
    for label, exp_id in arms.items():
        r = results[label]
        md.append(f"| {label} | `{exp_id}` | {r['pooled_cell_ts_auc']:.5f} | "
                  f"{r['pooled_cell_ts_auc_never_break_only']:.5f} | "
                  f"{r['pooled_cell_ts_auc_pre_break_only']:.5f} |")
    md += ["\n## Deltas\n",
           "| contrast | cell AUC Δ | size | translated pooled Δ |",
           "|---|---:|---|---:|",
           f"| B − A | {delta_BA:+.5f} | {size_BA} | {translated_BA:+.5f} |",
           f"| C − B | {delta_CB:+.5f} | {size_CB} | {translated_CB:+.5f} |",
           f"\n## Verdict\n\n**{case}**\n"]
    with open(f"{REPORTS}/wave7_d3r.md", "w") as f:
        f.write("\n".join(md))
    print(f"\nwrote research/reports/wave7_d3r.json and wave7_d3r.md")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["b", "c"])
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.arm == "b":
        train_arm_b()
    elif args.arm == "c":
        train_arm_c()
    elif args.analyze:
        analyze()
    else:
        raise SystemExit("pass --arm b, --arm c, or --analyze")


if __name__ == "__main__":
    main()
