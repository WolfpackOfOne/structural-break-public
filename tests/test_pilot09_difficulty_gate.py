from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOTS = os.path.join(REPO, "research", "scripts", "novel_streams", "pilots")
NOVEL = os.path.join(REPO, "research", "scripts", "novel_streams")
SCRIPTS = os.path.join(REPO, "research", "scripts")
SRC = os.path.join(REPO, "src")

sys.path.insert(0, PILOTS)
sys.path.insert(0, NOVEL)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, SRC)

from pilot09_difficulty_gate import (  # noqa: E402
    DEV_FOLDS,
    apply_derangement,
    derangement_audit,
    derangement_source,
    fold_purity_ok,
    nested_plan,
    robust_standardize,
)


def test_nested_plan_excludes_outer_and_predicted_folds():
    folds = np.repeat(np.arange(5), 4)
    for outer in DEV_FOLDS:
        plan = nested_plan(folds, outer)
        assert len(plan) == 5
        for item in plan:
            train_folds = set(item["train_folds"])
            assert outer not in train_folds
            assert item["pred_fold"] not in train_folds
            assert item["predict_mask"].sum() == 4
        assert fold_purity_ok([
            {k: v for k, v in item.items() if k in ("outer_fold", "pred_fold", "train_folds")}
            for item in plan
        ])


def test_derangement_has_no_fixed_points_and_preserves_fold_multiset():
    folds = np.repeat(np.arange(5), 6)
    scalar = np.arange(len(folds), dtype=np.float64) + 0.25
    source = derangement_source(folds, seed=0)
    deranged = apply_derangement(scalar, source, folds)
    audit = derangement_audit(scalar, deranged, source, folds)
    assert audit["ok"]
    for f in DEV_FOLDS:
        idx = np.flatnonzero(folds == f)
        assert np.all(source[idx] != idx)
        assert np.array_equal(np.sort(deranged[idx]), np.sort(scalar[idx]))


def test_robust_standardize_uses_training_mask_only_and_cleans_nan_inf():
    x = np.array(
        [
            [1.0, 2.0, np.nan],
            [2.0, 4.0, 1.0],
            [100.0, 8.0, np.inf],
            [4.0, 16.0, -np.inf],
        ]
    )
    train = np.array([True, True, False, False])
    z, constants = robust_standardize(x, train)
    assert z.shape == x.shape
    assert np.isfinite(z).all()
    assert constants["center"][0] == 1.5
    assert constants["center"][1] == 3.0


def test_derangement_singleton_fold_is_left_in_place():
    folds = np.array([0, 1, 1, 2, 2, 2])
    scalar = np.arange(len(folds), dtype=np.float64)
    source = derangement_source(folds, seed=0, dev_folds=(0, 1, 2))
    deranged = apply_derangement(scalar, source, folds)
    assert source[0] == 0
    assert deranged[0] == scalar[0]
    for f in (1, 2):
        idx = np.flatnonzero(folds == f)
        assert np.all(source[idx] != idx)
