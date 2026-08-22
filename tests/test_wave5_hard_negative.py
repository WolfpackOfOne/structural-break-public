"""Fold-purity, metric-alignment and behaviour gates for the Wave-5 hard-negative
protocol. Synthetic arrays only; no competition data required.

Two tests carry the weight here:
  * `test_mining_is_fold_pure` corrupts fold k arbitrarily -- new scores, every
    label flipped, rows reordered -- and requires the mined inner weights to be
    bitwise unchanged. A leak would be invisible in a score.
  * `test_time_conditional_disagrees_with_global_ranking` demonstrates that the
    PRE-C3 metric-alignment correction is not cosmetic: the two definitions
    disagree, and the old one disagrees with the metric.
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
    MIN_POS_AT_T,
    MIN_W,
    TOP_Q,
    _dyadic_bucket,
    age_buckets,
    hardness_from_oof,
    hardness_time_conditional,
    mine_fold_pure,
    oversample_index,
    weights_from_hardness,
)

N_T = 25          # distinct online indices in the synthetic panel
PER_T = 200       # rows per timestep -- comfortably above MIN_POS_AT_T


def _synth(n_series=400, n_t=N_T, seed=0):
    rng = np.random.default_rng(seed)
    series_id = np.repeat(np.arange(n_series), n_t)
    t_index = np.tile(np.arange(n_t), n_series)
    fold = series_id % 5
    labels = np.repeat((rng.uniform(size=n_series) < 0.5).astype(np.int8), n_t)
    oof = rng.uniform(size=n_series * n_t)
    return oof, labels, series_id, fold, t_index


# --------------------------------------------------------------- fold purity
def test_mining_is_fold_pure():
    """Corrupting fold k -- scores, labels AND row order -- must not move a weight."""
    oof, labels, sid, fold, t = _synth()
    for k in range(5):
        w0, h0 = mine_fold_pure(oof, labels, sid, fold, k, t_index=t)
        rng = np.random.default_rng(99 + k)
        oof2, labels2 = oof.copy(), labels.copy()
        m = fold == k
        oof2[m] = rng.uniform(size=int(m.sum()))
        labels2[m] = 1 - labels2[m]
        # also permute the fold-k rows among themselves
        idx = np.flatnonzero(m)
        perm = rng.permutation(idx)
        oof2[idx] = oof2[perm]
        w1, h1 = mine_fold_pure(oof2, labels2, sid, fold, k, t_index=t)
        assert np.array_equal(w0[~m], w1[~m]), f"fold {k}: weights moved"
        assert np.array_equal(h0[~m], h1[~m]), f"fold {k}: hardness moved"


def test_fold_k_rows_are_not_reweighted():
    oof, labels, sid, fold, t = _synth()
    for k in range(5):
        w, h = mine_fold_pure(oof, labels, sid, fold, k, t_index=t)
        m = fold == k
        assert np.all(w[m] == 1.0)
        assert np.all(h[m] == 0.0)


def test_missing_t_index_raises_rather_than_falling_back():
    """A silent fallback to global ranking would measure the wrong thing."""
    oof, labels, sid, fold, _ = _synth()
    with pytest.raises(ValueError, match="t_index is required"):
        mine_fold_pure(oof, labels, sid, fold, 0)


# ------------------------------------------------------- metric alignment
def test_same_time_ranking_is_correct():
    """Hardness is the fraction of same-t positives the negative outranks."""
    # one timestep, 5 positives at 0.1..0.5, negatives probing known positions
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.05, 0.35, 0.99])
    labels = np.array([1, 1, 1, 1, 1, 0, 0, 0])
    t = np.zeros(8, dtype=np.int64)
    h = hardness_time_conditional(scores, labels, t, np.ones(8, bool))
    assert h[5] == pytest.approx(0.0)   # below every positive -> no inversions
    assert h[6] == pytest.approx(0.6)   # beats 0.1,0.2,0.3 of 5
    assert h[7] == pytest.approx(1.0)   # beats all 5
    assert np.all(h[:5] == 0.0)         # positives carry no hardness


def test_tie_handling_is_midrank_and_deterministic():
    """Ties count as half an inversion, matching sbr.metric's mid-ranks."""
    scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    labels = np.array([1, 1, 1, 1, 1, 0])
    t = np.zeros(6, dtype=np.int64)
    h1 = hardness_time_conditional(scores, labels, t, np.ones(6, bool))
    h2 = hardness_time_conditional(scores, labels, t, np.ones(6, bool))
    assert h1[5] == pytest.approx(0.5), "a full tie is exactly half an inversion"
    assert np.array_equal(h1, h2), "tie handling must be deterministic"


def test_time_conditional_disagrees_with_global_ranking():
    """The correction is substantive, not cosmetic.

    Construct a panel where the global ordering is exactly backwards: at t=0 all
    scores are high, at t=1 all are low. A globally-ranked negative from t=0
    looks 'hard' even though it beats no positive at its own timestep.
    """
    scores = np.array([0.90, 0.95, 0.96, 0.97, 0.98, 0.99,      # t=0 positives
                       0.91,                                     # t=0 negative
                       0.10, 0.11, 0.12, 0.13, 0.14, 0.15,      # t=1 positives
                       0.20])                                    # t=1 negative
    labels = np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0])
    t = np.array([0] * 7 + [1] * 7, dtype=np.int64)
    elig = np.ones(14, bool)
    ht = hardness_time_conditional(scores, labels, t, elig)
    hg = hardness_from_oof(scores, labels == 0)
    # time-conditional: the t=0 negative beats almost no same-t positive; the
    # t=1 negative beats all of them
    assert ht[6] < ht[13], "time-conditional must rank the t=1 negative harder"
    # global ranking says the opposite, because 0.91 > 0.20 in absolute terms
    assert hg[6] > hg[13], "global ranking is backwards here -- that is the bug"


def test_fallback_to_dyadic_bucket_when_t_is_sparse():
    """Below MIN_POS_AT_T positives at a timestep, the dyadic bucket is used."""
    # t=8 and t=9 share dyadic bucket 3 (floor(log2(9))=3, floor(log2(10))=3)
    assert _dyadic_bucket(np.array([8]))[0] == _dyadic_bucket(np.array([9]))[0]
    n_pos_far = MIN_POS_AT_T + 3
    scores = np.concatenate([np.linspace(0.1, 0.5, n_pos_far), [0.2], [0.6]])
    labels = np.array([1] * n_pos_far + [1, 0])
    t = np.array([9] * n_pos_far + [8, 8], dtype=np.int64)
    h = hardness_time_conditional(scores, labels, t, np.ones(len(scores), bool))
    # t=8 has only 1 positive (< MIN_POS_AT_T), so the bucket reference is used
    # and the negative at 0.6 outranks every positive in the bucket
    assert h[-1] == pytest.approx(1.0)


def test_no_positive_anywhere_gives_zero_hardness():
    """With nothing to invert against, a negative causes no inversions."""
    scores = np.array([0.3, 0.9, 0.5])
    labels = np.array([0, 0, 0])
    t = np.array([0, 0, 1], dtype=np.int64)
    h = hardness_time_conditional(scores, labels, t, np.ones(3, bool))
    assert np.all(h == 0.0)


def test_no_negative_at_t_is_handled():
    """A timestep of pure positives contributes no hardness and does not crash."""
    scores = np.array([0.2, 0.4, 0.6, 0.8, 0.9, 0.5])
    labels = np.array([1, 1, 1, 1, 1, 0])
    t = np.array([0, 0, 0, 0, 0, 1], dtype=np.int64)
    h = hardness_time_conditional(scores, labels, t, np.ones(6, bool))
    assert np.all(h[:5] == 0.0)
    assert np.isfinite(h).all()


# ------------------------------------------------------------- behaviour
def test_positives_are_never_reweighted():
    """Protecting young breaks starts here: a positive row always has weight 1."""
    oof, labels, sid, fold, t = _synth()
    w, _ = mine_fold_pure(oof, labels, sid, fold, 0, t_index=t)
    assert np.all(w[labels == 1] == 1.0)


def test_weights_are_bounded():
    oof, labels, sid, fold, t = _synth()
    w, _ = mine_fold_pure(oof, labels, sid, fold, 2, t_index=t)
    assert w.min() >= MIN_W - 1e-12
    assert w.max() <= MAX_W + 1e-12


def test_oversample_is_deterministic_and_targets_the_hardest():
    oof, labels, sid, fold, t = _synth()
    _, h = mine_fold_pure(oof, labels, sid, fold, 1, t_index=t)
    neg = (labels == 0) & (fold != 1)
    a = oversample_index(h, neg)
    b = oversample_index(h, neg)
    assert np.array_equal(a, b), "oversampling must be deterministic"
    extra = a[len(h):]
    assert np.all(h[extra] >= 1.0 - TOP_Q - 1e-12)


def test_age_buckets_match_the_mandatory_reporting_grid():
    t = np.array([0, 3, 5, 9, 10, 19, 20, 49, 50, 99, 100, 500, 7])
    tau = np.zeros_like(t)
    tau[-1] = 50
    b = age_buckets(t, tau)
    assert list(b[:-1]) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert b[-1] == -1, "pre-break rows are negatives, not an age bucket"
    assert len(BUCKET_NAMES) == 6


def test_lambda_zero_is_the_uniform_control():
    oof, labels, sid, fold, t = _synth()
    w, _ = mine_fold_pure(oof, labels, sid, fold, 0, t_index=t, lam=0.0)
    assert np.all(w == 1.0)


@pytest.mark.parametrize("lam", [0.5, LAMBDA, 2.0])
def test_weights_increase_monotonically_with_hardness(lam):
    h = np.linspace(0, 1, 50)
    neg = np.ones(50, dtype=bool)
    w = weights_from_hardness(h, neg, lam)
    assert np.all(np.diff(w) >= -1e-12)
