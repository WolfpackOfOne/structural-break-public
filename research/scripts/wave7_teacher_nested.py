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

The 20 ordered (outer, inner) pairs contain only C(5,2)=10 distinct training
sets, because train_folds depends on the SET {outer, inner} and the subsample
RNG is fixed.  `--inner-teacher-pair F G` fits each distinct teacher once and
writes both nested_Q vectors it supports, halving the leg.  Verified bitwise in
research/reports/rt1320_promotion/PHASE0_COST_REDUCTION.md.

Usage:
    python wave7_teacher_nested.py --fold-purity-test
    python wave7_teacher_nested.py --list-inner-pairs        # the ten jobs
    python wave7_teacher_nested.py --inner-teacher-pair 0 1  # ... one per process
    python wave7_teacher_nested.py --outer-fold 0     # ... 1,2,3,4
    python wave7_teacher_nested.py --merge
    python wave7_teacher_nested.py --analyze-nested
"""
from __future__ import annotations

import argparse, gc, json, os, time

import numpy as np

from wave5_lib import Ctx, FOLDS, REPORTS, OOFDIR, ts_auc_flat, AGE_BUCKETS
import sbr.pipeline as PL
from wave7_d3r import FULL, ARM_B_PARAMS, cell_mask, score_on, bucket, last_row_lookup

EPS = 1e-6
MAX_TRAIN_ROWS = 1_000_000
FOLDS_SET = set(FOLDS)

#: Row-block sizes for the memory-lean paths below.  Verified in
#: research/reports/rt1320_promotion/PHASE0_COST_REDUCTION.md to leave output
#: bitwise unchanged; they exist only to keep peak memory down.
STACK_CHUNK = 100_000
PRED_CHUNK = 200_000


def _refuse_if_exists(path, force):
    """A nested_Q vector is gitignored and has no version-control copy.

    `cmd_inner_teacher` used to np.save over one without asking.  In a worktree
    whose research/oof is a symlink into a sibling checkout that silently
    destroys an input to an already-published result, which is exactly the
    class of failure `ProductionModel._check_provenance` exists to prevent.
    """
    if os.path.exists(path) and not force:
        raise SystemExit(
            f"REFUSING TO OVERWRITE {path}\n"
            f"  It already exists and *.npy is gitignored, so there is no copy to\n"
            f"  restore from. Move it aside, or pass --force if you truly mean to\n"
            f"  replace it.")


def augmented_stack_chunked(mats, names, keep_idx, rows, final_of_row, chunk=STACK_CHUNK):
    """`augmented_stack` without the np.concatenate transient.

    The original holds own (n x 500) + fut (n x 500) + the concatenated result
    (n x 1000) at once -- 7.45 GB at n=1e6.  Preallocating and filling in row
    blocks holds only the result plus one block.  Pure data movement, so the
    output is bitwise identical (verified by sha256 at n=250k and n=1e6).
    """
    n, k = len(rows), len(keep_idx)
    X = np.empty((n, 2 * k), dtype=np.float32)
    uniq, inv = np.unique(final_of_row[rows], return_inverse=True)
    fut_u = PL._stack(mats, names, uniq, keep_idx)
    for s0 in range(0, n, chunk):
        e0 = min(s0 + chunk, n)
        X[s0:e0, :k] = PL._stack(mats, names, rows[s0:e0], keep_idx)
        X[s0:e0, k:] = fut_u[inv[s0:e0]]
    return X


def _predict_chunked(booster, rows, mats, names, keep_idx, final_of_row, chunk=PRED_CHUNK):
    """Predict without materialising a second full-size feature matrix.

    Per-row independent, so bitwise identical to predicting the whole matrix
    (verified, max|diff| = 0.000e+00).
    """
    out = np.empty(len(rows), dtype=np.float64)
    for s0 in range(0, len(rows), chunk):
        e0 = min(s0 + chunk, len(rows))
        Xc = augmented_stack_chunked(mats, names, keep_idx, rows[s0:e0], final_of_row)
        out[s0:e0] = booster.predict(Xc)
        del Xc
        gc.collect()
    return out


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
def _fit_pair_booster(f, g, d, mats, names, keep_idx, final_of_row, threads=None):
    """Fit the teacher held out from BOTH fold f and fold g.

    `train_folds` depends only on the SET {f, g}, and the subsample RNG is a
    fixed default_rng(0) with no per-fold seed, so the booster for (outer=f,
    inner=g) and the one for (outer=g, inner=f) are the SAME model -- verified
    bitwise (identical model string, predictions equal at max|diff| = 0.000e+00)
    in research/reports/rt1320_promotion/PHASE0_COST_REDUCTION.md.

    Returns (booster, train_folds).  Callers may predict it onto either or both
    of the two held-out folds; doing both halves the nested scheme's cost.
    """
    train_folds = [x for x in FOLDS if x not in (f, g)]
    assert set(train_folds) & {f, g} == set(), \
        f"PURITY VIOLATION: teacher for pair {{{f}, {g}}} trained on {train_folds}"
    tr_rows = d.rows_for(train_folds)
    rng = np.random.default_rng(0)
    if len(tr_rows) > MAX_TRAIN_ROWS:
        tr_rows = np.sort(rng.choice(tr_rows, MAX_TRAIN_ROWS, replace=False))

    p = dict(ARM_B_PARAMS)
    n_round = p.pop("n_estimators")
    if threads is not None:
        p["num_threads"] = int(threads)
    Xtr = augmented_stack_chunked(mats, names, keep_idx, tr_rows, final_of_row)
    assert Xtr.shape[1] == 1000, f"expected 1000 cols (500 causal + 500 broadcast), got {Xtr.shape[1]}"

    import lightgbm as lgb
    ds = lgb.Dataset(Xtr, label=d.y[tr_rows], params=dict(p, objective="binary"),
                     feature_name=[f"f{i}" for i in range(Xtr.shape[1])])
    ds.construct()          # bin now, so the float32 matrix can go
    del Xtr
    gc.collect()
    booster = lgb.train(dict(p), ds, num_boost_round=n_round)
    del ds
    gc.collect()
    return booster, train_folds


def _predict_teacher_on(booster, target_fold, d, mats, names, keep_idx, final_of_row):
    va_rows = d.rows_for([target_fold])
    pred = _predict_chunked(booster, va_rows, mats, names, keep_idx, final_of_row)
    return va_rows, np.clip(pred.astype(np.float64), EPS, 1 - EPS)


def train_inner_teacher(outer_f, inner_g, d, mats, names, keep_idx, final_of_row, threads=None):
    """One teacher, predicting only its inner fold. Unchanged semantics."""
    booster, train_folds = _fit_pair_booster(
        outer_f, inner_g, d, mats, names, keep_idx, final_of_row, threads)
    assert set(train_folds) & {outer_f, inner_g} == set(), \
        f"PURITY VIOLATION: teacher for outer={outer_f} inner={inner_g} trained on {train_folds}"
    return _predict_teacher_on(booster, inner_g, d, mats, names, keep_idx, final_of_row)


def build_nested_Q(outer_f, d, mats, names, keep_idx, final_of_row, threads=None):
    """Assemble Q over the four outer-training folds.

    Prefers an already-computed `nested_Q_outer{f}_inner{g}.npy` on disk.  Every
    such vector is produced by the {f, g} pair teacher, so running the ten
    `--inner-teacher-pair` jobs first means this refits nothing.
    """
    outer_train = [g for g in FOLDS if g != outer_f]
    Q = np.full(len(d.y), np.nan, dtype=np.float64)
    for g in outer_train:
        cached = f"{OOFDIR}/nested_Q_outer{outer_f}_inner{g}.npy"
        if os.path.exists(cached):
            Qg = np.load(cached)
            m = ~np.isnan(Qg)
            va_rows = np.flatnonzero(m)
            expect = d.rows_for([g])
            assert np.array_equal(va_rows, expect), \
                f"cached {cached} covers {len(va_rows)} rows, fold {g} has {len(expect)}"
            Q[va_rows] = Qg[va_rows]
            print(f"    inner teacher outer={outer_f} target_fold={g}: "
                  f"{len(va_rows)} rows from cache, Q mean {Qg[va_rows].mean():.4f}", flush=True)
            continue
        va_rows, pred = train_inner_teacher(
            outer_f, g, d, mats, names, keep_idx, final_of_row, threads)
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
# Granular, disk-checkpointed entry points -- each is a SHORT-LIVED, standalone
# process that trains exactly one model, saves to disk, and exits (freeing all
# memory back to the OS before the next call starts). Introduced after two
# whole-outer-fold attempts (4 teachers + 2 students in one long-lived process,
# ~40-50 min) were killed partway through the second inner teacher -- once
# running concurrently (2 processes), once alone. Splitting into one call per
# model removes both plausible causes at once: each call is short (matches the
# successful pilot precedent, ~5-10 min) and starts with a clean process, so
# no cross-call memory fragmentation or accumulated feature-memmap page-cache
# pressure from four back-to-back 1000-column trainings in one process.
def _save_Q(path, n, va_rows, pred, force):
    _refuse_if_exists(path, force)
    Q = np.full(n, np.nan, dtype=np.float64)
    Q[va_rows] = pred
    np.save(path, Q)
    return path


def cmd_inner_teacher(outer_f, inner_g, force=False, threads=None):
    d = PL.Data()
    mats, names = PL.load_features(FULL)
    keep_idx = np.arange(len(names))
    final_of_row = last_row_lookup(d)[d.sidx]
    path = f"{OOFDIR}/nested_Q_outer{outer_f}_inner{inner_g}.npy"
    _refuse_if_exists(path, force)          # fail BEFORE spending the fit
    t0 = time.time()
    va_rows, pred = train_inner_teacher(
        outer_f, inner_g, d, mats, names, keep_idx, final_of_row, threads)
    _save_Q(path, len(d.y), va_rows, pred, force)
    print(f"  inner teacher outer={outer_f} target_fold={inner_g}: {len(va_rows)} rows, "
          f"Q mean {pred.mean():.4f}, runtime {time.time()-t0:.1f}s -> {path}")


def cmd_inner_teacher_pair(f, g, force=False, threads=None):
    """Fit the {f, g} teacher ONCE and write both nested_Q vectors it supports.

    The nested scheme runs 20 ordered (outer, inner) pairs but they contain only
    C(5,2)=10 distinct training sets, each fitted twice.  This is the same work
    as two `--inner-teacher` calls, for one fit.
    """
    if f == g:
        raise SystemExit("--inner-teacher-pair needs two DIFFERENT folds")
    d = PL.Data()
    mats, names = PL.load_features(FULL)
    keep_idx = np.arange(len(names))
    final_of_row = last_row_lookup(d)[d.sidx]
    # (outer=f, inner=g) predicts fold g; (outer=g, inner=f) predicts fold f.
    targets = [(f, g), (g, f)]
    paths = {t: f"{OOFDIR}/nested_Q_outer{t[0]}_inner{t[1]}.npy" for t in targets}
    for t in targets:
        _refuse_if_exists(paths[t], force)   # fail BEFORE spending the fit
    t0 = time.time()
    booster, train_folds = _fit_pair_booster(
        f, g, d, mats, names, keep_idx, final_of_row, threads)
    t_fit = time.time() - t0
    print(f"  pair {{{f}, {g}}} teacher trained on {train_folds} in {t_fit:.1f}s", flush=True)
    for outer_f, inner_g in targets:
        va_rows, pred = _predict_teacher_on(
            booster, inner_g, d, mats, names, keep_idx, final_of_row)
        _save_Q(paths[(outer_f, inner_g)], len(d.y), va_rows, pred, force)
        print(f"    -> outer={outer_f} target_fold={inner_g}: {len(va_rows)} rows, "
              f"Q mean {pred.mean():.4f} -> {paths[(outer_f, inner_g)]}", flush=True)
    print(f"  pair {{{f}, {g}}} total {time.time()-t0:.1f}s")


def cmd_list_inner_pairs():
    """Print the ten distinct pair jobs, one per line.

    Deliberately prints rather than runs them: the comment above
    cmd_inner_teacher records that long-lived multi-model processes were killed
    partway, so one process per fit remains the operational precedent.
    """
    seen = []
    for f in FOLDS:
        for g in FOLDS:
            if f < g:
                seen.append((f, g))
    for f, g in seen:
        print(f"--inner-teacher-pair {f} {g}")
    return seen


def cmd_train_nested_student(outer_f, kind):
    d = PL.Data()
    mats_causal, names_causal = PL.load_features(FULL)
    keep_idx_causal = np.arange(len(names_causal))
    outer_train = [g for g in FOLDS if g != outer_f]
    Q = np.full(len(d.y), np.nan, dtype=np.float64)
    for g in outer_train:
        path = f"{OOFDIR}/nested_Q_outer{outer_f}_inner{g}.npy"
        Qg = np.load(path)
        va_rows_g = d.rows_for([g])
        assert not np.isnan(Qg[va_rows_g]).any(), f"missing inner-teacher checkpoint {path}"
        Q[va_rows_g] = Qg[va_rows_g]
    t0 = time.time()
    va_rows, pred, s = train_nested_student(kind, outer_f, d, mats_causal, names_causal, keep_idx_causal, Q)
    exp_id = "RT-994" if kind == "t1" else "RT-995"
    oof = np.full(len(d.y), np.nan, dtype=np.float32)
    oof[va_rows] = pred
    np.save(f"{OOFDIR}/{exp_id}_outer{outer_f}.npy", oof)
    summary = {"outer_fold": outer_f, "kind": kind, "ts_auc": s, "runtime_s": round(time.time() - t0, 1)}
    with open(f"{REPORTS}/wave7_teacher_nested_outer{outer_f}_{kind}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
def run_outer_fold(f, threads=None):
    t_start = time.time()
    d = PL.Data()
    mats_causal, names_causal = PL.load_features(FULL)
    keep_idx_causal = np.arange(len(names_causal))
    last_row_by_series = last_row_lookup(d)
    final_of_row = last_row_by_series[d.sidx]

    print(f"=== outer fold {f}: building nested Q over the other 4 folds ===", flush=True)
    Q = build_nested_Q(f, d, mats_causal, names_causal, keep_idx_causal, final_of_row, threads)

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
    ap.add_argument("--inner-teacher", type=int, nargs=2, metavar=("OUTER", "INNER"))
    ap.add_argument("--train-nested-student", type=int, metavar="OUTER")
    ap.add_argument("--kind", choices=["t1", "t2"])
    ap.add_argument("--inner-teacher-pair", type=int, nargs=2, metavar=("F", "G"),
                    help="fit the {F,G} teacher ONCE and write both nested_Q vectors")
    ap.add_argument("--list-inner-pairs", action="store_true",
                    help="print the ten distinct pair jobs, one per line")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--analyze-nested", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing nested_Q vector (it is gitignored "
                         "and has no version-control copy -- be sure)")
    ap.add_argument("--threads", type=int, default=None,
                    help=f"override num_threads (frozen default "
                         f"{ARM_B_PARAMS['num_threads']}); shown bitwise-neutral in "
                         f"PHASE0_COST_REDUCTION.md, but opt in explicitly")
    args = ap.parse_args()
    if args.fold_purity_test:
        fold_purity_test()
    elif args.list_inner_pairs:
        cmd_list_inner_pairs()
    elif args.inner_teacher_pair is not None:
        cmd_inner_teacher_pair(*args.inner_teacher_pair, force=args.force,
                               threads=args.threads)
    elif args.inner_teacher is not None:
        cmd_inner_teacher(*args.inner_teacher, force=args.force, threads=args.threads)
    elif args.train_nested_student is not None:
        assert args.kind, "--train-nested-student requires --kind t1|t2"
        cmd_train_nested_student(args.train_nested_student, args.kind)
    elif args.outer_fold is not None:
        run_outer_fold(args.outer_fold, threads=args.threads)
    elif args.merge:
        merge()
    elif args.analyze_nested:
        analyze_nested()
    else:
        raise SystemExit("pass --fold-purity-test, --list-inner-pairs, "
                         "--inner-teacher-pair F G, --outer-fold F, --merge, "
                         "or --analyze-nested")


if __name__ == "__main__":
    main()
