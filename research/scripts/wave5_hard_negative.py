"""Fold-pure hard-negative mining for Wave 5. Protocol:
research/reports/wave5_hard_negative_protocol.md

Hard-negative mining is NOT a feature module and there is no `mXX_hardnegative`.
It is a change to the TRAINING DISTRIBUTION, and the only thing that makes it
legitimate is that the mining never sees the fold it will be evaluated on.

The label at row (series, t) is `t >= tau`, so NEGATIVES ARE OF TWO KINDS:
  1. every row of a series that never breaks, and
  2. the PRE-BREAK rows (`t < tau`) of a series that does break.
The second kind is structurally interesting: a model that anticipates a break it
cannot yet see scores those rows high, and the metric punishes it.

Everything here operates on plain arrays so it is unit-testable with no
competition data present.
"""
from __future__ import annotations

import numpy as np

# --- PRE-REGISTERED MINING CONSTANTS -----------------------------------------
# Fixed 2026-08-22, before any Wave-5 5-fold outcome exists and before the
# competition store is even present in the environment. Not tunable on a score.
LAMBDA = 1.0        # weight arm:      w = 1 + LAMBDA * hardness
TOP_Q = 0.20        # oversample arm:  duplicate the hardest 20% of negatives
MIN_W, MAX_W = 1.0, 2.0   # weights are bounded so no row can dominate a split

# --- PRE-C3 METRIC-ALIGNMENT CORRECTION, 2026-08-22 --------------------------
# Added BEFORE any Wave-5 TS-AUC was observed (training runs = 0 at the time).
#
# Minimum number of positives at an exact timestep for the same-t estimate to be
# used. Below this the dyadic online-index bucket below is used instead. Chosen
# for estimator stability, never from a score -- no score existed.
MIN_POS_AT_T = 5

# Fallback bucketing when a timestep has too few positives: dyadic buckets
# floor(log2(t+1)). Declared a priori; not derived from any observed outcome.
def _dyadic_bucket(t_index: np.ndarray) -> np.ndarray:
    """Deterministic coarse online-index bucket: floor(log2(t+1))."""
    return np.floor(np.log2(np.asarray(t_index, dtype=np.float64) + 1.0)).astype(np.int64)


def hardness_from_oof(oof_score: np.ndarray, is_negative: np.ndarray) -> np.ndarray:
    """SUPERSEDED by `hardness_time_conditional`. Retained for audit only.

    This ranks negatives GLOBALLY, across all timesteps at once. That is not what
    the metric measures: TS-AUC only ever compares a positive against a negative
    **at the same online index**, so a negative scoring 0.9 at a timestep where
    every positive scores 0.95 causes no inversions at all, while a negative
    scoring 0.4 where the positives score 0.3 causes many. Global ranking calls
    the first one hard and the second easy, i.e. exactly backwards.

    `mine_fold_pure` no longer uses this. It is kept so the PRE-C3
    METRIC-ALIGNMENT CORRECTION is auditable and so a test can demonstrate that
    the two definitions genuinely disagree.

    Rank-percentile hardness of each NEGATIVE row, 0 (easy) .. 1 (hardest).

    Hardness is the model's own out-of-fold score on a row whose true label is
    negative: a high score on a negative row is, by definition, the mistake we
    want more of in training. Positives get 0.0 and are never reweighted -- see
    the protocol's §"protecting young breaks".
    """
    oof_score = np.asarray(oof_score, dtype=np.float64)
    is_negative = np.asarray(is_negative, dtype=bool)
    out = np.zeros(len(oof_score), dtype=np.float64)
    n = int(is_negative.sum())
    if n == 0:
        return out
    s = oof_score[is_negative]
    order = np.argsort(s, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    out[is_negative] = ranks / max(n - 1, 1)
    return out


def _pair_fraction(neg_scores: np.ndarray, pos_scores_sorted: np.ndarray) -> np.ndarray:
    """Fraction of positives a negative outranks, with MID-RANK ties.

        h = ( #{pos < neg}  +  0.5 * #{pos == neg} ) / n_pos

    The 0.5 is not a detail. `sbr.metric` computes TS-AUC with mid-ranks, so a
    tie contributes exactly half a pairwise inversion there; counting ties any
    other way would make hardness disagree with the quantity it is meant to
    estimate. Deterministic: no RNG, no ordering dependence.
    """
    lo = np.searchsorted(pos_scores_sorted, neg_scores, side="left")
    hi = np.searchsorted(pos_scores_sorted, neg_scores, side="right")
    n = len(pos_scores_sorted)
    return (lo + 0.5 * (hi - lo)) / max(n, 1)


def hardness_time_conditional(oof_score: np.ndarray, labels: np.ndarray,
                              t_index: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """METRIC-ALIGNED hardness: pairwise inversions caused AT THE SAME TIMESTEP.

    For a negative row `i` observed at online index `t`,

        hardness(i) = P( score of a positive at t  <  score of negative i )

    estimated over the positives present at that same `t`. This is, up to a
    constant, the negative's own contribution to the numerator TS-AUC is built
    from -- the thing the competition actually penalises.

    `eligible` masks the rows that may participate at all (fold purity is
    enforced by the caller passing an inner-only mask). Positives receive 0.0 and
    are never reweighted.

    Support rules, both declared a priori:
      * exact same-`t` estimate whenever that timestep has >= MIN_POS_AT_T
        eligible positives;
      * otherwise the dyadic online-index bucket floor(log2(t+1));
      * if that bucket also has no eligible positives, hardness is 0.0 -- with no
        positive to be inverted against, the negative causes no inversions and is
        by definition not hard. That is a consequence of the definition, not a
        fallback choice.
    """
    oof_score = np.asarray(oof_score, dtype=np.float64)
    labels = np.asarray(labels)
    t_index = np.asarray(t_index, dtype=np.int64)
    eligible = np.asarray(eligible, dtype=bool)

    out = np.zeros(len(oof_score), dtype=np.float64)
    is_pos = eligible & (labels == 1)
    is_neg = eligible & (labels == 0)
    if not is_neg.any():
        return out

    # positives grouped by exact timestep, and by dyadic bucket
    pos_t = t_index[is_pos]
    pos_s = oof_score[is_pos]
    by_t: dict[int, np.ndarray] = {}
    for key in np.unique(pos_t):
        by_t[int(key)] = np.sort(pos_s[pos_t == key])

    pos_b = _dyadic_bucket(pos_t)
    by_b: dict[int, np.ndarray] = {}
    for key in np.unique(pos_b):
        by_b[int(key)] = np.sort(pos_s[pos_b == key])

    neg_idx = np.flatnonzero(is_neg)
    neg_t = t_index[neg_idx]
    neg_b = _dyadic_bucket(neg_t)
    neg_s = oof_score[neg_idx]

    for key in np.unique(neg_t):
        sel = neg_t == key
        ref = by_t.get(int(key))
        if ref is not None and len(ref) >= MIN_POS_AT_T:
            out[neg_idx[sel]] = _pair_fraction(neg_s[sel], ref)
        else:
            bkey = int(neg_b[sel][0])
            bref = by_b.get(bkey)
            if bref is not None and len(bref) > 0:
                out[neg_idx[sel]] = _pair_fraction(neg_s[sel], bref)
            # else: stays 0.0 -- nothing at this t or bucket to invert against
    return out


def weights_from_hardness(hardness: np.ndarray, is_negative: np.ndarray,
                          lam: float = LAMBDA) -> np.ndarray:
    """Bounded per-row training weights. Positives keep weight 1.0 exactly."""
    w = np.ones(len(hardness), dtype=np.float64)
    neg = np.asarray(is_negative, dtype=bool)
    w[neg] = np.clip(1.0 + lam * np.asarray(hardness)[neg], MIN_W, MAX_W)
    return w


def oversample_index(hardness: np.ndarray, is_negative: np.ndarray,
                     top_q: float = TOP_Q, rng: np.random.Generator | None = None):
    """Row index duplicating the hardest `top_q` fraction of negatives once.

    Deterministic: no RNG is consulted, the argument exists only so callers can
    pass one without special-casing.
    """
    neg = np.asarray(is_negative, dtype=bool)
    idx = np.arange(len(hardness))
    hard = neg & (np.asarray(hardness) >= (1.0 - top_q))
    return np.concatenate([idx, idx[hard]])


def mine_fold_pure(oof_score: np.ndarray, labels: np.ndarray, series_id: np.ndarray,
                   fold: np.ndarray, k: int, t_index: np.ndarray | None = None,
                   lam: float = LAMBDA):
    """Weights for training a candidate that will be evaluated on outer fold `k`.

    Uses the METRIC-ALIGNED time-conditional hardness (PRE-C3 correction,
    2026-08-22, made before any Wave-5 TS-AUC existed). `t_index` is the online
    index of each row and is required for that; if it is omitted the function
    raises rather than silently falling back to the superseded global ranking,
    because a silent fallback would produce plausible weights that measure the
    wrong thing.

    FOLD PURITY, which is the whole point:
      * rows in fold `k` are masked out of the eligible set before hardness is
        computed, so no fold-k score, label, timestep or ordering can influence
        any weight -- including through the same-t positive reference, which is
        built from inner positives only;
      * fold-k rows are returned with weight 1.0 and must not be trained on.

    `tests/test_wave5_hard_negative.py` asserts this by corrupting fold k
    entirely -- new scores, every label flipped, rows reordered -- and requiring
    the returned inner weights to be bitwise identical.
    """
    fold = np.asarray(fold)
    labels = np.asarray(labels)
    inner = fold != k
    w = np.ones(len(labels), dtype=np.float64)
    h = np.zeros(len(labels), dtype=np.float64)
    if not inner.any():
        return w, h
    if t_index is None:
        raise ValueError(
            "t_index is required: hardness is time-conditional after the "
            "2026-08-22 metric-alignment correction. Pass the online index.")

    h = hardness_time_conditional(oof_score, labels, t_index, inner)
    is_neg_inner = inner & (np.asarray(labels) == 0)
    w[inner] = weights_from_hardness(h, is_neg_inner, lam)[inner]
    return w, h


def age_buckets(t_index: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Post-break age bucket id for every row; -1 for negatives.

    Buckets are the six the protocol makes mandatory:
    0-5, 5-10, 10-20, 20-50, 50-100, 100+.
    """
    age = np.asarray(t_index) - np.asarray(tau)
    out = np.full(len(age), -1, dtype=np.int8)
    edges = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100)]
    ok = age >= 0
    for b, (lo, hi) in enumerate(edges):
        out[ok & (age >= lo) & (age < hi)] = b
    out[ok & (age >= 100)] = 5
    return out


BUCKET_NAMES = ("0-5", "5-10", "10-20", "20-50", "50-100", "100+")
