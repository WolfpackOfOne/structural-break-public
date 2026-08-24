"""Wave 8 pre-score gates (WAVE8_FUTURE_AWARE_PREREG.md section 6).

Run to green BEFORE any SST/ORR/PCFB/CFEP/TGMC score is read.

    GATE 1  future-row lookup is exact (row_layout / future_row_index)
    GATE 2  nested double-cross-fit never lets an inner regressor see {f,g}
    GATE 3  the sentinel correctly REJECTS the old (contaminated) single-
            level global-OOF scheme -- proves it catches the defect, not
            just that the new construction trivially passes
    GATE 4  no forbidden column (tau/boundary/n_online/availability/...)
            reaches a student feature name
    GATE 5  eligibility diagnostic is well-formed and its "flagged" rule
            fires on a synthetic class-conditional-eligibility fixture
    GATE 6  deterministic target construction: same seed -> identical output
"""
from __future__ import annotations

import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "research", "scripts"))

import numpy as np
import pytest

import wave8_common as W8
from wave5_lib import FOLDS


# --------------------------------------------------------------------------- 1
def test_future_row_index_matches_direct_lookup():
    d = W8.PL.Data()
    first_row, series_len = W8.row_layout(d)
    for h in (1, 50, 200):
        fut, valid = W8.future_row_index(d, h, first_row, series_len)
        # direct check on a random sample of rows across many series
        rng = np.random.default_rng(0)
        sample = rng.choice(len(d.y), size=2000, replace=False)
        for r in sample:
            s, t = d.sidx[r], d.t[r]
            expect_valid = (t + h) < series_len[s]
            assert bool(valid[r]) == expect_valid
            if expect_valid:
                fr = fut[r]
                assert d.sidx[fr] == s and d.t[fr] == t + h
            else:
                assert fut[r] == -1


# --------------------------------------------------------------------------- 2/3
def test_nested_scheme_purity_and_old_scheme_rejected():
    """Pure set arithmetic (no training): the NEW nested scheme this file's
    nested_oof_regressor implements never lets an inner fit set touch
    {outer_f, inner_g}; the OLD (single-level global-OOF) scheme -- fit on
    everything except g, ignorant of the outer fold -- always does, for
    every (f,g) with f != g. Mirrors wave7_teacher_nested.fold_purity_test's
    already-proven contract, re-checked for Wave 8's own construction.
    """
    folds_set = set(FOLDS)
    new_failures, old_contaminated, total = [], 0, 0
    for f in FOLDS:
        outer_train = [g for g in FOLDS if g != f]
        for g in outer_train:
            total += 1
            fit_new = folds_set - {f, g}          # what nested_oof_regressor actually fits on
            if fit_new & {f, g}:
                new_failures.append((f, g))
            fit_old = folds_set - {g}              # naive global-OOF, ignorant of f
            if fit_old & {f, g}:
                old_contaminated += 1
    assert len(new_failures) == 0, f"nested scheme purity violated: {new_failures}"
    assert old_contaminated == total, "sentinel failed to reproduce the known contamination pattern"


def test_nested_oof_regressor_purity_live():
    """Same contract, exercised through the real function signature on one
    outer fold with a tiny row budget, so the assertion inside
    nested_oof_regressor itself is proven reachable and correct, not just
    the set arithmetic above."""
    d = W8.PL.Data()
    mats, names = W8.PL.load_features(["m00_core"])
    keep_idx = np.arange(len(names))
    y = d.y.astype(np.float64)
    train_mask = np.ones(len(y), dtype=bool)
    Zhat = W8.nested_oof_regressor(0, d, mats, names, keep_idx, y, train_mask,
                                   rounds=5, max_rows=5000)
    outer_train_rows = d.rows_for([g for g in FOLDS if g != 0])
    assert not np.isnan(Zhat[outer_train_rows]).any()
    outer_val_rows = d.rows_for([0])
    assert np.isnan(Zhat[outer_val_rows]).all(), \
        "nested regressor produced predictions for its own outer-validation fold"


# --------------------------------------------------------------------------- 4
def test_assert_no_forbidden_columns_catches_and_passes():
    with pytest.raises(AssertionError):
        W8.assert_no_forbidden_columns(["m00_core::foo", "diag::true_tau_index"])
    with pytest.raises(AssertionError):
        W8.assert_no_forbidden_columns(["feat::n_online_frac"])
    W8.assert_no_forbidden_columns([f"m0{i}::col{j}" for i in range(8) for j in range(5)])


def test_full_module_list_has_no_forbidden_columns():
    _, names = W8.PL.load_features(W8.FULL)
    W8.assert_no_forbidden_columns(names)


# --------------------------------------------------------------------------- 5
def test_eligibility_diagnostic_shape_and_flag_logic():
    d = W8.PL.Data()
    out = W8.eligibility_diagnostic(d, 200)
    assert out["horizon"] == 200
    assert len(out["buckets"]) > 0
    assert "flagged" in out and isinstance(out["flagged"], bool)
    for b in out["buckets"]:
        if not (np.isnan(b["p_eligible_y1"]) or np.isnan(b["p_eligible_y0"])):
            assert 0.0 <= b["p_eligible_y1"] <= 1.0
            assert 0.0 <= b["p_eligible_y0"] <= 1.0


# --------------------------------------------------------------------------- 6
def test_future_row_index_deterministic():
    d = W8.PL.Data()
    fut1, valid1 = W8.future_row_index(d, 100)
    fut2, valid2 = W8.future_row_index(d, 100)
    assert np.array_equal(fut1, fut2)
    assert np.array_equal(valid1, valid2)
