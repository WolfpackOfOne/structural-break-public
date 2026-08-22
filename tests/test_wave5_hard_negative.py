"""Fold-purity and behaviour gates for the Wave-5 hard-negative protocol.

Runs on synthetic arrays; no competition data required. The central test is
`test_mining_is_fold_pure`: it corrupts fold k arbitrarily and requires the
mined weights to be bitwise unchanged. A leak here would invalidate every
number produced under this protocol, and it would be invisible in a score.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

sys.path.insert(0, "research/scripts")

from wave5_hard_negative import (  # noqa: E402
    BUCKET_NAMES,
    LAMBDA,
    MAX_W,
    MIN_W,
    TOP_Q,
    age_buckets,
    hardness_from_oof,
    mine_fold_pure,
    oversample_index,
    weights_from_hardness,
)


def _synth(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    series_id = np.repeat(np.arange(n // 20), 20)
    fold = series_id % 5
    labels = (rng.uniform(size=n) < 0.45).astype(np.int8)
    oof = rng.uniform(size=n)
    return oof, labels, series_id, fold


def test_mining_is_fold_pure():
    """Corrupting fold k must not move a single mined weight."""
    oof, labels, sid, fold = _synth()
    for k in range(5):
        w0, h0 = mine_fold_pure(oof, labels, sid, fold, k)
        rng = np.random.default_rng(99 + k)
        oof2, labels2 = oof.copy(), labels.copy()
        m = fold == k
        oof2[m] = rng.uniform(size=m.sum())          # arbitrary new scores
        labels2[m] = 1 - labels2[m]                  # flip every fold-k label
        w1, h1 = mine_fold_pure(oof2, labels2, sid, fold, k)
        assert np.array_equal(w0[~m], w1[~m]), f"fold {k}: weights moved"
        assert np.array_equal(h0[~m], h1[~m]), f"fold {k}: hardness moved"


def test_fold_k_rows_are_not_reweighted():
    oof, labels, sid, fold = _synth()
    for k in range(5):
        w, h = mine_fold_pure(oof, labels, sid, fold, k)
        m = fold == k
        assert np.all(w[m] == 1.0)
        assert np.all(h[m] == 0.0)


def test_positives_are_never_reweighted():
    """Protecting young breaks starts here: a positive row always has weight 1."""
    oof, labels, sid, fold = _synth()
    w, _ = mine_fold_pure(oof, labels, sid, fold, 0)
    assert np.all(w[labels == 1] == 1.0)


def test_weights_are_bounded():
    oof, labels, sid, fold = _synth()
    w, _ = mine_fold_pure(oof, labels, sid, fold, 2)
    assert w.min() >= MIN_W - 1e-12
    assert w.max() <= MAX_W + 1e-12


def test_hardness_ranks_the_right_direction():
    """A high score on a NEGATIVE row is the mistake we want more of."""
    oof = np.array([0.9, 0.1, 0.5, 0.99])
    neg = np.array([True, True, True, False])
    h = hardness_from_oof(oof, neg)
    assert h[0] > h[2] > h[1], "hardness must increase with the score on negatives"
    assert h[3] == 0.0, "positives carry no hardness"


def test_oversample_is_deterministic_and_targets_the_hardest():
    oof, labels, sid, fold = _synth()
    _, h = mine_fold_pure(oof, labels, sid, fold, 1)
    neg = (labels == 0) & (fold != 1)
    a = oversample_index(h, neg)
    b = oversample_index(h, neg)
    assert np.array_equal(a, b), "oversampling must be deterministic"
    extra = a[len(h):]
    assert np.all(h[extra] >= 1.0 - TOP_Q - 1e-12)


def test_age_buckets_match_the_mandatory_reporting_grid():
    t = np.array([0, 3, 5, 9, 10, 19, 20, 49, 50, 99, 100, 500, 7])
    tau = np.zeros_like(t)
    tau[-1] = 50                      # a negative row: t < tau
    b = age_buckets(t, tau)
    assert list(b[:-1]) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert b[-1] == -1, "pre-break rows are negatives, not an age bucket"
    assert len(BUCKET_NAMES) == 6


def test_lambda_zero_is_the_uniform_control():
    """The control arm must be exactly uniform, so the comparison is clean."""
    oof, labels, sid, fold = _synth()
    w, _ = mine_fold_pure(oof, labels, sid, fold, 0, lam=0.0)
    assert np.all(w == 1.0)


@pytest.mark.parametrize("lam", [0.5, LAMBDA, 2.0])
def test_weights_increase_monotonically_with_hardness(lam):
    h = np.linspace(0, 1, 50)
    neg = np.ones(50, dtype=bool)
    w = weights_from_hardness(h, neg, lam)
    assert np.all(np.diff(w) >= -1e-12)
