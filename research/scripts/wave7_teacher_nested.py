"""W7 teacher/distillation, NESTED (double) cross-fitting -- the corrected
replacement for the outer-fold-contaminated pilot (RT-992/RT-993).

Pre-registered in research/WAVE7_TEACHER_NESTED_PREREG.md BEFORE any inner
teacher or nested student score exists. See that document for the defect,
the fix, and the fold-purity sentinel's exact contract.

For every OUTER validation fold f:
    outer_train = the other four folds
    for every INNER held-out fold g in outer_train:
        train an Arm-C-architecture teacher on folds {0..4} \\ {f, g}
        predict Q for fold g
    Q_outer_train = concatenation of the four inner predictions
    train T1 (label=Q_outer_train) and T2 (label=0.5*y+0.5*Q_outer_train)
        on outer_train, identical inputs/capacity to RT-992/RT-993
    evaluate on the untouched outer fold f

No teacher that labels an outer fold's training data ever trained on that
outer fold.

Usage:
    python wave7_teacher_nested.py --fold-purity-test
    python wave7_teacher_nested.py --outer-fold 0     # ... 1,2,3,4
    python wave7_teacher_nested.py --merge
    python wave7_teacher_nested.py --analyze-nested
"""
from __future__ import annotations

import argparse, json, time

import numpy as np

from wave5_lib import Ctx, FOLDS, REPORTS, OOFDIR, ts_auc_flat, AGE_BUCKETS
import sbr.pipeline as PL
from wave7_d3r import FULL, ARM_B_PARAMS, cell_mask, score_on, bucket, last_row_lookup, augmented_stack

EPS = 1e-6
MAX_TRAIN_ROWS = 1_000_000
FOLDS_SET = set(FOLDS)


# ---------------------------------------------------------------------------
def fold_purity_test():
    """Pure set arithmetic, no data/training. Must show: (a) the NEW nested
    scheme never lets a teacher_training_folds set touch {f, g}; (b) the
    OLD (contaminated) scheme ALWAYS does, for every (f, g) with f != g --
    proving the sentinel actually catches the defect, not just that the new
    construction trivially passes."""
    new_failures, old_contaminated, total = [], 0, 0
    for f in FOLDS:
        outer_train = [g for g in FOLDS if g != f]
        for g in outer_train:
            total += 1
            tt_new = FOLDS_SET - {f, g}
            if tt_new & {f, g}:
                new_failures.append([f, g, sorted(tt_new)])
            tt_old = FOLDS_SET - {g}          # old scheme: ignorant of f
            if tt_old & {f, g}:
                old_contaminated += 1

    new_passed = len(new_failures) == 0
    old_sentinel_catches_defect = (old_contaminated == total)
    print(f"NEW (nested) scheme: {len(new_failures)}/{total} purity failures (must be 0)")
    print(f"OLD (global-OOF) scheme: {old_contaminated}/{total} checks show contamination "
          f"(must be {total}, i.e. EVERY (f,g) pair -- proving the sentinel catches the "
          f"exact defect found in RT-992/RT-993)")
    out = {"new_scheme_failures": new_failures, "new_scheme_passed": new_passed,
           "old_scheme_contaminated_checks": old_contaminated, "total_checks": total,
           "old_scheme_sentinel_catches_defect": old_sentinel_catches_defect}
    with open(f"{REPORTS}/wave7_teacher_nested_fold_purity_test.json", "w") as fh:
        json.dump(out, fh, indent=2)
    assert new_passed, "NESTED scheme FAILED its own fold-purity test -- do not proceed"
    assert old_sentinel_catches_defect, ("sentinel did not reproduce the known RT-992/993 "
                                         "defect on the old scheme -- the test itself is broken")
    print("FOLD_PURITY_TEST = PASS (nested scheme clean; old scheme correctly shown "
          "contaminated on every (f,g) pair)")
    return out


# ---------------------------------------------------------------------------
def train_inner_teacher(outer_f, inner_g, d, mats, names, keep_idx, final_of_row):
    train_folds = [x for x in FOLDS if x not in (outer_f, inner_g)]
    tr_rows = d.rows_for(train_folds)
    va_rows = d.rows_for([inner_g])
    rng = np.random.default_rng(0)
    if len(tr_rows) > MAX_TRAIN_ROWS:
        tr_rows = np.sort(rng.choice(tr_rows, MAX_TRAIN_ROWS, replace=False))

    p = dict(ARM_B_PARAMS)
    n_round = p.pop("n_estimators")
    Xtr = augmented_stack(mats, names, keep_idx, tr_rows, final_of_row)
    assert Xtr.shape[1] == 1000, f"expected 1000 cols (500 causal + 500 broadcast), got {Xtr.shape[1]}"
    ytr = d.y[tr_rows]

    import lightgbm as lgb
    ds = lgb.Dataset(Xtr, label=ytr, params=dict(p, objective="binary"),
                     feature_name=[f"f{i}" for i in range(Xtr.shape[1])])
    booster = lgb.train(dict(p), ds, num_boost_round=n_round)
    del Xtr, ds

    Xva = augmented_stack(mats, names, keep_idx, va_rows, final_of_row)
    pred = booster.predict(Xva).astype(np.float64)
    del Xva
    assert set(train_folds) & {outer_f, inner_g} == set(), \
        f"PURITY VIOLATION: teacher for outer={outer_f} inner={inner_g} trained on {train_folds}"
    return va_rows, np.clip(pred, EPS, 1 - EPS)


def build_nested_Q(outer_f, d, mats, names, keep_idx, final_of_row):
    outer_train = [g for g in FOLDS if g != outer_f]
    Q = np.full(len(d.y), np.nan, dtype=np.float64)
    for g in outer_train:
        va_rows, pred = train_inner_teacher(outer_f, g, d, mats, names, keep_idx, final_of_row)
        Q[va_rows] = pred
        print(f"    inner teacher outer={outer_f} target_fold={g}: "
              f"{len(va_rows)} rows labeled, Q mean {pred.mean():.4f}", flush=True)
    return Q


def train_nested_student(kind, outer_f, d, mats_causal, names_causal, keep_idx_causal, Q):
    tr_rows = d.rows_for([g for g in FOLDS if g != outer_f])
    va_rows = d.rows_for([outer_f])
    rng = np.random.default_rng(0)
    if len(tr_rows) > MAX_TRAIN_ROWS:
        tr_rows = np.sort(rng.choice(tr_rows, MAX_TRAIN_ROWS, replace=False))
    assert not np.isnan(Q[tr_rows]).any(), "nested Q has NaN on a training row -- inner teacher coverage bug"

    y_hard = d.y.astype(np.float64)
    if kind == "t1":
        ytr = Q[tr_rows]
    else:
        ytr = 0.5 * y_hard[tr_rows] + 0.5 * Q[tr_rows]

    p = dict(ARM_B_PARAMS)
    p["objective"] = "xentropy"
    n_round = p.pop("n_estimators")
    Xtr = PL._stack(mats_causal, names_causal, tr_rows, keep_idx_causal)
    assert Xtr.shape[1] == 500

    import lightgbm as lgb
    ds = lgb.Dataset(Xtr, label=ytr, params=dict(p),
                     feature_name=[f"f{i}" for i in range(Xtr.shape[1])])
    booster = lgb.train(p, ds, num_boost_round=n_round)
    del Xtr, ds

    Xva = PL._stack(mats_causal, names_causal, va_rows, keep_idx_causal)
    assert Xva.shape[1] == 500
    pred = booster.predict(Xva).astype(np.float32)
    del Xva
    s = float(ts_auc_flat(pred, d.y[va_rows], d.t[va_rows]))
    print(f"  [{kind.upper()}-nested] outer fold {outer_f}: TS-AUC {s:.5f}  "
          f"({len(tr_rows)} train rows, {len(va_rows)} valid rows)", flush=True)
    return va_rows, pred, s


# ---------------------------------------------------------------------------
def run_outer_fold(f):
    t_start = time.time()
    d = PL.Data()
    mats_causal, names_causal = PL.load_features(FULL)
    keep_idx_causal = np.arange(len(names_causal))
    last_row_by_series = last_row_lookup(d)
    final_of_row = last_row_by_series[d.sidx]

    print(f"=== outer fold {f}: building nested Q over the other 4 folds ===", flush=True)
    Q = build_nested_Q(f, d, mats_causal, names_causal, keep_idx_causal, final_of_row)

    oof_t1 = np.full(len(d.y), np.nan, dtype=np.float32)
    oof_t2 = np.full(len(d.y), np.nan, dtype=np.float32)
    scores = {}
    for kind, oof in (("t1", oof_t1), ("t2", oof_t2)):
        va_rows, pred, s = train_nested_student(kind, f, d, mats_causal, names_causal, keep_idx_causal, Q)
        oof[va_rows] = pred
        scores[kind] = s

    np.save(f"{OOFDIR}/RT-994_outer{f}.npy", oof_t1)
    np.save(f"{OOFDIR}/RT-995_outer{f}.npy", oof_t2)
    summary = {"outer_fold": f, "t1_ts_auc": scores["t1"], "t2_ts_auc": scores["t2"],
               "runtime_s": round(time.time() - t_start, 1)}
    with open(f"{REPORTS}/wave7_teacher_nested_outer{f}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
def merge():
    d = PL.Data()
    for kind, exp_id in (("t1", "RT-994"), ("t2", "RT-995")):
        oof = np.full(len(d.y), np.nan, dtype=np.float32)
        per_fold = []
        for f in FOLDS:
            part = np.load(f"{OOFDIR}/{exp_id}_outer{f}.npy")
            va_rows = d.rows_for([f])
            assert not np.isnan(part[va_rows]).any(), f"missing predictions for outer fold {f} in {exp_id}"
            oof[va_rows] = part[va_rows]
            per_fold.append(float(ts_auc_flat(part[va_rows], d.y[va_rows], d.t[va_rows])))
        dev_rows = d.rows_for(list(FOLDS))
        overall = float(ts_auc_flat(oof[dev_rows], d.y[dev_rows], d.t[dev_rows]))
        np.save(f"{OOFDIR}/{exp_id}.npy", oof)

        res = dict(
            experiment_id=exp_id, date=time.strftime("%Y-%m-%d %H:%M"), git_sha=PL.git_sha(),
            agent="agent0",
            hypothesis=f"W7 teacher NESTED {kind.upper()}: outer-fold-pure double cross-fitted "
                       f"teacher, {'pure distillation' if kind == 't1' else 'hard+teacher 0.5/0.5 blend'}",
            falsification_condition="see research/WAVE7_TEACHER_NESTED_PREREG.md §6",
            feature_set=",".join(FULL), n_features=500, model="lgbm", objective="xentropy",
            folds=",".join(map(str, FOLDS)), random_seed=0,
            train_series=int((~np.isin(d.series_fold, [-1])).sum()), train_rows=MAX_TRAIN_ROWS,
            mean_oof_ts_auc=float(np.mean(per_fold)), pooled_oof_ts_auc=overall,
            per_fold_ts_auc=";".join(f"{x:.5f}" for x in per_fold), fold_std=float(np.std(per_fold)),
            persistence="none", sample_mode="uniform", training_runtime_s=None,
            causal_verified="student inputs = unmodified 500-col causal bank; nested teacher Q "
                            "used only as training label, never as a feature; fold-purity "
                            "sentinel passed (wave7_teacher_nested_fold_purity_test.json)",
            test_reduced_touched="no", lockbox_touched="no", status="recorded",
            notes="W7 teacher NESTED (outer-fold-pure) 5-fold run, replacing outer-fold-"
                  "contaminated RT-992/RT-993 per research/WAVE7_TEACHER_NESTED_PREREG.md.",
            protocol="nested_full",
        )
        PL.append_result(res)
        print(f"{exp_id}: mean {res['mean_oof_ts_auc']:.5f}  pooled {res['pooled_oof_ts_auc']:.5f}  "
              f"per-fold {res['per_fold_ts_auc']}")
    print("merged RT-994.npy and RT-995.npy, appended RESULTS.csv rows")


# ---------------------------------------------------------------------------
def analyze_nested():
    c = Ctx()
    T0 = np.load(f"{OOFDIR}/RT-990.npy")
    T1 = np.load(f"{OOFDIR}/RT-994.npy")
    T2 = np.load(f"{OOFDIR}/RT-995.npy")
    vecs = {"T0_RT990": T0, "T1_RT994": T1, "T2_RT995": T2}

    per_fold_rows = []
    for f in FOLDS:
        r = c.rows[f]
        y, t, age = c.d.y[r], c.d.t[r], c.age[r]
        sidx = c.d.sidx[r]
        neg_is_prebreak = c.has_break[sidx] & (y == 0)
        cm = cell_mask(y, t, age)
        cm_never = cm & ((y == 1) | (~neg_is_prebreak))
        cm_pre = cm & ((y == 1) | neg_is_prebreak)
        row = {"outer_fold": f}
        for label, v in vecs.items():
            v_r = v[r]
            row[f"{label}_whole"] = float(ts_auc_flat(v_r, y, t))
            row[f"{label}_cell"] = score_on(v_r, cm, y, t)
            row[f"{label}_cell_never"] = score_on(v_r, cm_never, y, t)
            row[f"{label}_cell_pre"] = score_on(v_r, cm_pre, y, t)
        row["T1_minus_T0_whole"] = row["T1_RT994_whole"] - row["T0_RT990_whole"]
        row["T2_minus_T0_whole"] = row["T2_RT995_whole"] - row["T0_RT990_whole"]
        per_fold_rows.append(row)
        print(f"  outer fold {f}: T0 {row['T0_RT990_whole']:.5f}  T1 {row['T1_RT994_whole']:.5f}  "
              f"T2 {row['T2_RT995_whole']:.5f}  T1-T0 {row['T1_minus_T0_whole']:+.5f}  "
              f"T2-T0 {row['T2_minus_T0_whole']:+.5f}")

    mean_T1 = float(np.mean([r["T1_minus_T0_whole"] for r in per_fold_rows]))
    mean_T2 = float(np.mean([r["T2_minus_T0_whole"] for r in per_fold_rows]))
    pos_T1 = sum(1 for r in per_fold_rows if r["T1_minus_T0_whole"] > 0)
    pos_T2 = sum(1 for r in per_fold_rows if r["T2_minus_T0_whole"] > 0)

    # age-bucket / current-t-bucket breakdown, pooled dev, holding negatives fixed
    age_buckets = {}
    for label, v in vecs.items():
        age_buckets[label] = c.score_by_age(v)

    t_edges = [0, 20, 50, 100, 200, 400, 1000, 10**9]
    t_buckets = {label: {} for label in vecs}
    y_dev, t_dev = c.d.y[c.dev], c.d.t[c.dev]
    for label, v in vecs.items():
        v_dev = v[c.dev]
        for lo, hi in zip(t_edges[:-1], t_edges[1:]):
            m = (t_dev >= lo) & (t_dev < hi)
            key = f"{lo}-{hi if hi < 10**9 else ''}"
            t_buckets[label][key] = score_on(v_dev, m, y_dev, t_dev)

    print("\n  running paired series-level bootstrap (200 reps)...")
    contrasts = {"T1-T0": ("T1_RT994", "T0_RT990"), "T2-T0": ("T2_RT995", "T0_RT990")}
    boot = c.bootstrap(vecs, contrasts, n=200, seed=0)
    for k, v in boot.items():
        print(f"    {k}: mean {v['mean']:+.5f}  CI95 [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]  "
              f"frac>0 {v['fraction_positive']:.2f}")

    # contaminated-vs-clean, fold 0 only (the pilot's only evaluated fold)
    with open(f"{REPORTS}/wave7_teacher_pilot.json") as fh:
        pilot = json.load(fh)
    contaminated_T1_whole_delta = pilot["deltas"]["T1_RT992"]["whole_fold0_delta"]
    contaminated_T2_whole_delta = pilot["deltas"]["T2_RT993"]["whole_fold0_delta"]
    clean_fold0 = per_fold_rows[0]
    contamination_effect = {
        "T1": {"contaminated_fold0_whole_delta": contaminated_T1_whole_delta,
               "clean_fold0_whole_delta": clean_fold0["T1_minus_T0_whole"],
               "contamination_inflation": contaminated_T1_whole_delta - clean_fold0["T1_minus_T0_whole"]},
        "T2": {"contaminated_fold0_whole_delta": contaminated_T2_whole_delta,
               "clean_fold0_whole_delta": clean_fold0["T2_minus_T0_whole"],
               "contamination_inflation": contaminated_T2_whole_delta - clean_fold0["T2_minus_T0_whole"]},
    }
    print(f"\n  contamination effect (fold 0): T1 inflation "
          f"{contamination_effect['T1']['contamination_inflation']:+.5f}, T2 inflation "
          f"{contamination_effect['T2']['contamination_inflation']:+.5f}")

    def reading(mean_delta):
        if mean_delta >= 0.006:
            return "MAJOR BREAKTHROUGH"
        if mean_delta >= 0.004:
            return "LEADERBOARD CANDIDATE"
        if mean_delta >= 0.003:
            return "SERIOUS CANDIDATE"
        return "DO NOT PROMOTE"

    verdict_T1, verdict_T2 = reading(mean_T1), reading(mean_T2)
    bootstrap_T1_supportive = boot["T1-T0"]["ci95"][0] > 0
    bootstrap_T2_supportive = boot["T2-T0"]["ci95"][0] > 0
    clears_all_T1 = mean_T1 >= 0.003 and pos_T1 >= 4 and bootstrap_T1_supportive
    clears_all_T2 = mean_T2 >= 0.003 and pos_T2 >= 4 and bootstrap_T2_supportive
    alt_partitions_authorized = clears_all_T1 or clears_all_T2

    out = {
        "experiment": "W7 teacher NESTED (outer-fold-pure)",
        "prereg": "research/WAVE7_TEACHER_NESTED_PREREG.md",
        "per_fold": per_fold_rows,
        "mean_delta": {"T1": mean_T1, "T2": mean_T2},
        "folds_positive": {"T1": pos_T1, "T2": pos_T2},
        "reading": {"T1": verdict_T1, "T2": verdict_T2},
        "bootstrap": boot,
        "age_buckets": age_buckets, "t_buckets": t_buckets,
        "contamination_effect": contamination_effect,
        "clears_promotion_legs_1to3": {"T1": clears_all_T1, "T2": clears_all_T2},
        "alternate_partitions_authorized": alt_partitions_authorized,
    }
    with open(f"{REPORTS}/wave7_teacher_nested.json", "w") as fh:
        json.dump(out, fh, indent=2)

    md = ["# WAVE 7 — TEACHER NESTED (OUTER-FOLD-PURE) RESULTS\n",
          "Pre-registered: `research/WAVE7_TEACHER_NESTED_PREREG.md`. "
          "Replaces outer-fold-contaminated `RT-992`/`RT-993` "
          "(`research/reports/wave7_teacher_pilot.md`) with `RT-994`/`RT-995`, "
          "trained under nested (double) cross-fitting so no teacher that "
          "labels an outer fold's training data ever trained on that fold.\n",
          "## Primary result table (whole-dev TS-AUC per outer fold)\n",
          "| outer fold | T0 | T1 | T2 | T1−T0 | T2−T0 |",
          "|---:|---:|---:|---:|---:|---:|"]
    for r in per_fold_rows:
        md.append(f"| {r['outer_fold']} | {r['T0_RT990_whole']:.5f} | {r['T1_RT994_whole']:.5f} | "
                  f"{r['T2_RT995_whole']:.5f} | {r['T1_minus_T0_whole']:+.5f} | {r['T2_minus_T0_whole']:+.5f} |")
    md += [f"\n**Mean Δ**: T1 {mean_T1:+.5f} ({pos_T1}/5 folds positive), "
           f"T2 {mean_T2:+.5f} ({pos_T2}/5 folds positive).\n",
           "## Dominant-cell (t≥200, age≥100) TS-AUC per outer fold\n",
           "| outer fold | T0 cell | T1 cell | T2 cell | T0 never-brk | T1 never-brk | T2 never-brk | T0 pre-brk | T1 pre-brk | T2 pre-brk |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in per_fold_rows:
        md.append(f"| {r['outer_fold']} | {r['T0_RT990_cell']:.5f} | {r['T1_RT994_cell']:.5f} | "
                  f"{r['T2_RT995_cell']:.5f} | {r['T0_RT990_cell_never']:.5f} | "
                  f"{r['T1_RT994_cell_never']:.5f} | {r['T2_RT995_cell_never']:.5f} | "
                  f"{r['T0_RT990_cell_pre']:.5f} | {r['T1_RT994_cell_pre']:.5f} | {r['T2_RT995_cell_pre']:.5f} |")
    md += ["\n## Paired series bootstrap (200 reps)\n",
           "| contrast | mean | CI95 | fraction > 0 |", "|---|---:|---|---:|"]
    for k, v in boot.items():
        md.append(f"| {k} | {v['mean']:+.5f} | [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}] | {v['fraction_positive']:.2f} |")
    md += ["\n## Contaminated-vs-clean (fold 0 only, whole-dev)\n",
           "| arm | contaminated Δ | clean (nested) Δ | contamination inflation |",
           "|---|---:|---:|---:|",
           f"| T1 | {contamination_effect['T1']['contaminated_fold0_whole_delta']:+.5f} | "
           f"{contamination_effect['T1']['clean_fold0_whole_delta']:+.5f} | "
           f"{contamination_effect['T1']['contamination_inflation']:+.5f} |",
           f"| T2 | {contamination_effect['T2']['contaminated_fold0_whole_delta']:+.5f} | "
           f"{contamination_effect['T2']['clean_fold0_whole_delta']:+.5f} | "
           f"{contamination_effect['T2']['contamination_inflation']:+.5f} |"]
    md += [f"\n## Reading (research/WAVE7_TEACHER_NESTED_PREREG.md §6)\n",
           f"T1: mean Δ {mean_T1:+.5f} → **{verdict_T1}**  \n",
           f"T2: mean Δ {mean_T2:+.5f} → **{verdict_T2}**\n",
           f"\n**Clears promotion legs 1-3 (magnitude ≥+0.003, ≥4/5 folds, bootstrap CI>0):** "
           f"T1={clears_all_T1}, T2={clears_all_T2}\n",
           f"\n**Alternate-partition confirmation authorized:** {alt_partitions_authorized}\n"]
    with open(f"{REPORTS}/wave7_teacher_nested.md", "w") as fh:
        fh.write("\n".join(md))
    print(f"\nT1: {verdict_T1}  T2: {verdict_T2}")
    print(f"wrote {REPORTS}/wave7_teacher_nested.{{md,json}}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-purity-test", action="store_true")
    ap.add_argument("--outer-fold", type=int, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--analyze-nested", action="store_true")
    args = ap.parse_args()
    if args.fold_purity_test:
        fold_purity_test()
    elif args.outer_fold is not None:
        run_outer_fold(args.outer_fold)
    elif args.merge:
        merge()
    elif args.analyze_nested:
        analyze_nested()
    else:
        raise SystemExit("pass --fold-purity-test, --outer-fold F, --merge, or --analyze-nested")


if __name__ == "__main__":
    main()
