"""Wave 8 shared infrastructure: future-row lookup, nested (double) cross-fit
regression, forbidden-column guard, and the one ensemble contract every
mechanism uses. Pre-registered in research/WAVE8_FUTURE_AWARE_PREREG.md
section 3 BEFORE any Wave-8 score exists. Never edits sbr core files.
"""
from __future__ import annotations

import os, sys

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

import numpy as np

import sbr.pipeline as PL
from wave5_lib import Ctx, FOLDS, OOFDIR, REPORTS, ts_auc_flat
from wave7_d3r import FULL

FORBIDDEN_TOKENS = ("tau", "cut", "boundary", "break_at", "changepoint_true",
                     "has_break", "n_online", "n_hist", "final_row",
                     "eligible", "availability", "pad")

SST_CHANNELS = {
    "cz100_cur": "m01_seq",       # 1. long-window location displacement
    "loc_h_rl1_q": "m06_loc",     # 2. robust/median location displacement
    "vr_qh_z": "m03_dyn",         # 3. log variance ratio
    "qd_exp_qiqr": "m02_dist",    # 4. robust scale / IQR ratio
    "ar2_e_acf1": "m04_resid",    # 5. dependence / AR displacement
    "glz_cur": "m01_seq",         # 6. cumulative evidence / GLR-like state
    "sl_mean_16_128": "m00_core", # 7. long-vs-short evidence persistence
    "bo_nochange": "m07_bayes",   # 8. historical no-break compatibility
}
SST_HORIZONS = (50, 100, 200)

REGRESSOR_PARAMS = dict(objective="regression", learning_rate=0.08, num_leaves=31,
                         min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                         bagging_freq=1, lambda_l2=5.0, num_threads=8, verbose=-1,
                         max_bin=127)
REGRESSOR_ROUNDS = 200


# ---------------------------------------------------------------------------
def row_layout(d):
    """Verify + return (first_row_of_series[n_series], series_len[n_series]).

    Physical row order is series-major with t contiguous 0..n-1 per series --
    verified empirically (see WAVE8_FUTURE_AWARE_PREREG.md section 3) before
    this file was written. Re-asserted here on 16 series every call (cheap)
    so a future data-layout change fails loudly instead of silently mis-
    indexing every future-row lookup in this file.
    """
    sidx = d.sidx
    n_series = int(sidx.max()) + 1
    assert np.array_equal(sidx, np.sort(sidx)), "store is not series-major; future_row_index invalid"
    order = np.argsort(sidx, kind="stable")
    s_sorted = sidx[order]
    b = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])
    e = np.r_[b[1:], len(s_sorted)]
    first_row = np.zeros(n_series, dtype=np.int64)
    series_len = np.zeros(n_series, dtype=np.int64)
    for i, (lo, hi) in enumerate(zip(b[:16], e[:16])):
        t_here = d.t[lo:hi]
        assert np.array_equal(t_here, np.arange(len(t_here))), \
            f"series {i} rows are not t=0..n-1 contiguous -- layout assumption broken"
    first_row[s_sorted[b]] = b
    series_len[s_sorted[b]] = e - b
    return first_row, series_len


def future_row_index(d, h, first_row=None, series_len=None):
    """row index of (same series, t+h) for every row, or -1 if unsupported."""
    if first_row is None:
        first_row, series_len = row_layout(d)
    fut = d.sidx.astype(np.int64) * 0  # placeholder dtype anchor
    fut = np.arange(len(d.y), dtype=np.int64) + h
    valid = (d.t + h) < series_len[d.sidx]
    fut = np.where(valid, fut, -1)
    return fut, valid


def eligibility_diagnostic(d, h, t_edges=(0, 20, 50, 100, 200, 400, 1000, 10**9)):
    """P(t+h eligible | y=1, t-bucket) vs P(t+h eligible | y=0, t-bucket).

    Run BEFORE any target is trained on (WAVE8_FUTURE_AWARE_PREREG.md section
    3/13): eligibility must never differ materially by class at matched t, or
    it is itself a tau-adjacent leak channel even though it never becomes a
    feature.
    """
    _, valid = future_row_index(d, h)
    out = {"horizon": h, "buckets": []}
    max_abs_gap = 0.0
    for lo, hi in zip(t_edges[:-1], t_edges[1:]):
        m = (d.t >= lo) & (d.t < hi)
        if not m.any():
            continue
        p1 = float(valid[m & (d.y == 1)].mean()) if (m & (d.y == 1)).any() else float("nan")
        p0 = float(valid[m & (d.y == 0)].mean()) if (m & (d.y == 0)).any() else float("nan")
        gap = abs(p1 - p0) if not (np.isnan(p1) or np.isnan(p0)) else 0.0
        max_abs_gap = max(max_abs_gap, gap)
        out["buckets"].append({"t_lo": lo, "t_hi": hi if hi < 10**9 else None,
                                "p_eligible_y1": p1, "p_eligible_y0": p0, "abs_gap": gap})
    out["max_abs_gap"] = max_abs_gap
    out["flagged"] = max_abs_gap > 0.05
    return out


def assert_no_forbidden_columns(names):
    bad = [n for n in names if any(tok in n.lower() for tok in FORBIDDEN_TOKENS)]
    assert not bad, f"forbidden column(s) reachable by a student: {bad}"


# ---------------------------------------------------------------------------
def nested_oof_regressor(outer_f, d, mats, names, keep_idx, y_full, train_mask,
                          params=None, rounds=REGRESSOR_ROUNDS, max_rows=400_000, seed=0):
    """Double-cross-fitted OOF regression prediction for every row of
    outer_train = FOLDS \\ {outer_f}.

    For each inner held-out fold g in outer_train: train a regressor on
    {outer_train \\ {g}} restricted to train_mask (e.g. rows with a defined
    future target), predict for EVERY row of fold g (whether or not that
    row itself is in train_mask -- a row missing its own true future target
    can still receive a predicted one). No teacher/regressor that labels an
    outer fold's training data ever trains on that outer fold (asserted).
    """
    import lightgbm as lgb
    p = dict(REGRESSOR_PARAMS if params is None else params)
    n_round = rounds
    rng = np.random.default_rng(seed)
    outer_train = [g for g in FOLDS if g != outer_f]
    Zhat = np.full(len(d.y), np.nan, dtype=np.float64)

    for g in outer_train:
        fit_folds = [x for x in outer_train if x != g]
        assert outer_f not in fit_folds and g not in fit_folds, \
            f"PURITY VIOLATION: inner regressor for outer={outer_f} inner={g} would fit on {fit_folds}"
        fit_rows = d.rows_for(fit_folds)
        fit_rows = fit_rows[train_mask[fit_rows]]
        if len(fit_rows) > max_rows:
            fit_rows = np.sort(rng.choice(fit_rows, max_rows, replace=False))
        va_rows = d.rows_for([g])

        Xtr = PL._stack(mats, names, fit_rows, keep_idx)
        ytr = y_full[fit_rows]
        ds = lgb.Dataset(Xtr, label=ytr, params=p, feature_name=[f"f{i}" for i in range(Xtr.shape[1])])
        booster = lgb.train(p, ds, num_boost_round=n_round)
        del Xtr, ds

        Xva = PL._stack(mats, names, va_rows, keep_idx)
        Zhat[va_rows] = booster.predict(Xva).astype(np.float64)
        del Xva

    tr_rows_all = d.rows_for(outer_train)
    assert not np.isnan(Zhat[tr_rows_all]).any(), "nested regressor left NaN on an outer-train row"
    return Zhat


def full_predict_for_outer_val(outer_f, d, mats, names, keep_idx, y_full, train_mask,
                                params=None, rounds=REGRESSOR_ROUNDS, max_rows=400_000, seed=0):
    """Retrain on ALL of outer_train (restricted to train_mask), predict the
    untouched outer fold f. Second-level-OOF contract, stage 2: the model the
    final classifier's OWN validation fold sees is fit on every outer_train
    row, not held back the way the inner nested-Q construction is.
    """
    import lightgbm as lgb
    p = dict(REGRESSOR_PARAMS if params is None else params)
    rng = np.random.default_rng(seed)
    outer_train = [g for g in FOLDS if g != outer_f]
    fit_rows = d.rows_for(outer_train)
    fit_rows = fit_rows[train_mask[fit_rows]]
    if len(fit_rows) > max_rows:
        fit_rows = np.sort(rng.choice(fit_rows, max_rows, replace=False))
    va_rows = d.rows_for([outer_f])

    Xtr = PL._stack(mats, names, fit_rows, keep_idx)
    ytr = y_full[fit_rows]
    ds = lgb.Dataset(Xtr, label=ytr, params=p, feature_name=[f"f{i}" for i in range(Xtr.shape[1])])
    booster = lgb.train(p, ds, num_boost_round=rounds)
    del Xtr, ds

    Xva = PL._stack(mats, names, va_rows, keep_idx)
    pred = booster.predict(Xva).astype(np.float64)
    del Xva
    out = np.full(len(d.y), np.nan, dtype=np.float64)
    out[va_rows] = pred
    return out


# ---------------------------------------------------------------------------
def fold_purity_test_generic():
    """Same set-arithmetic sentinel as wave7_teacher_nested.fold_purity_test,
    re-run here to confirm the identical contract holds for Wave 8's
    real-valued nested regressors, not just the binary teacher case."""
    from wave7_teacher_nested import fold_purity_test
    return fold_purity_test()


# ---------------------------------------------------------------------------
SPECIALISTS = ["RT-300", "RT-410", "RT-411", "RT-412", "RT-413", "RT-414", "RT-415"]
SEEDCLONE_ID = "RT-401"


def synthetic_features(hist, online, modules=FULL):
    """Compute the SAME production feature modules directly from raw (hist,
    online) arrays -- used by TGMC to score synthetic series with the real
    causal feature engine (guaranteed prefix-invariant, same code path
    production rows go through)."""
    from sbr.features.base import load_all, make_ctx, REGISTRY
    load_all()
    ctx = make_ctx(np.asarray(hist, dtype=np.float64), np.asarray(online, dtype=np.float64))
    mats, names = [], []
    for m in modules:
        cn, arr = REGISTRY[m].fn(ctx)
        mats.append(np.asarray(arr, dtype=np.float32))
        names += [f"{m}::{c}" for c in cn]
    return np.concatenate(mats, axis=1), names


def ensemble_marginal(candidate_oof, c=None, fold=0, label="candidate"):
    """RT600 vs RT600+seed-clone vs RT600+candidate, fold-0 pilot, legal
    cross-fitted SCDF calibration (wave5_lib.Ctx.crossfit_blend) -- never a
    global rank transform. candidate_oof: full-length array, NaN outside the
    rows it covers (must cover at least the pilot fold's validation rows).
    """
    if c is None:
        c = Ctx()
    P = {s: np.load(f"{OOFDIR}/{s}.npy") for s in SPECIALISTS + [SEEDCLONE_ID]}
    P["__candidate__"] = candidate_oof

    def blend_fold0(names):
        v = c.crossfit_blend(P, names)
        r = c.rows[fold]
        return float(ts_auc_flat(v[r], c.d.y[r], c.d.t[r]))

    base = blend_fold0(SPECIALISTS)
    clone = blend_fold0(SPECIALISTS + [SEEDCLONE_ID])
    cand = blend_fold0(SPECIALISTS + ["__candidate__"])
    return {
        "fold": fold, "rt600_7stream": base, "rt600_plus_seedclone": clone,
        f"rt600_plus_{label}": cand,
        "marginal_vs_clone": cand - clone, "gain_vs_base": cand - base,
    }
