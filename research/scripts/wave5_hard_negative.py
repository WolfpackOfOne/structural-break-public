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
LAMBDA = 1.0        # weight arm:      w = 1 + LAMBDA * hardness_rank_pct
TOP_Q = 0.20        # oversample arm:  duplicate the hardest 20% of negatives
MIN_W, MAX_W = 1.0, 2.0   # weights are bounded so no row can dominate a split


def hardness_from_oof(oof_score: np.ndarray, is_negative: np.ndarray) -> np.ndarray:
    """Rank-percentile hardness of each NEGATIVE row, 0 (easy) .. 1 (hardest).

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
                   fold: np.ndarray, k: int, lam: float = LAMBDA):
    """Weights for training a candidate that will be evaluated on outer fold `k`.

    FOLD PURITY, which is the whole point:
      * rows in fold `k` are masked out before hardness is computed, so no fold-k
        score, label or rank can influence any weight;
      * fold-k rows are returned with weight 1.0 and must not be trained on;
      * hardness ranks are computed among inner negatives ONLY, so adding,
        removing or relabelling fold-k rows cannot move an inner row's rank.

    `tests/test_wave5_hard_negative.py` asserts this by permuting fold-k labels
    and scores and requiring the returned weights to be bitwise identical.
    """
    fold = np.asarray(fold)
    labels = np.asarray(labels)
    inner = fold != k
    w = np.ones(len(labels), dtype=np.float64)
    if not inner.any():
        return w, np.zeros(len(labels))
    is_neg_inner = np.zeros(len(labels), dtype=bool)
    is_neg_inner[inner] = labels[inner] == 0
    h = np.zeros(len(labels), dtype=np.float64)
    sub = hardness_from_oof(np.where(inner, oof_score, -np.inf), is_neg_inner)
    h[inner] = sub[inner]
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
