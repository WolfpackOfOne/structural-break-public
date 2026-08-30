"""Parity and causality tests for the 2026 Real-Time research library.

These are the two properties every result in ``research/`` rests on:

1. our fast Time-Stratified AUC equals an obviously-correct sklearn reference,
   including under ties;
2. feature modules are causal -- rebuilding on a truncated online segment
   reproduces the surviving rows bitwise.

If either breaks, every number in the experiment ledger is void.
"""
from __future__ import annotations

import numpy as np
import pytest

sbr_metric = pytest.importorskip("sbr.metric")
ts_auc_ragged = sbr_metric.ts_auc_ragged
ts_auc_reference = sbr_metric.ts_auc_reference
ts_auc_flat = sbr_metric.ts_auc_flat


def _random_trajectories(rng, n_series=40, round_scores=False):
    scores, labels = [], []
    for _ in range(n_series):
        n = int(rng.integers(3, 25))
        tau = int(rng.integers(0, n)) if rng.random() < 0.5 else -1
        y = np.zeros(n, dtype=np.int8)
        if tau >= 0:
            y[tau:] = 1
        s = rng.normal(size=n) + 0.3 * y
        scores.append(np.round(s, 1) if round_scores else s)
        labels.append(y)
    return scores, labels


@pytest.mark.parametrize("round_scores", [False, True])
def test_ts_auc_matches_sklearn_reference(round_scores):
    """Exact agreement with the slow reference, ties included."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        scores, labels = _random_trajectories(rng, round_scores=round_scores)
        fast = ts_auc_ragged(scores, labels)
        slow = ts_auc_reference(scores, labels)
        assert fast == pytest.approx(slow, abs=1e-12)


def test_ts_auc_is_invariant_to_within_timestep_monotone_transforms():
    """The metric only reads within-timestep order, so this must not move it."""
    rng = np.random.default_rng(1)
    scores, labels = _random_trajectories(rng)
    flat = np.concatenate(scores)
    y = np.concatenate(labels)
    t = np.concatenate([np.arange(len(s)) for s in scores])
    base = ts_auc_flat(flat, y, t)
    for f in (lambda x: 3.0 * x + 1.0, lambda x: np.expm1(x / 2.0)):
        assert ts_auc_flat(f(flat), y, t) == pytest.approx(base, abs=1e-12)


def test_single_class_timesteps_carry_no_weight():
    """A timestep with only one class present is undefined and must be skipped."""
    scores = [np.array([0.9, 0.1, 0.5]), np.array([0.2, 0.8, 0.4])]
    labels = [np.array([0, 0, 1], dtype=np.int8), np.array([0, 1, 1], dtype=np.int8)]
    # step 0 is all-negative; the answer must equal the metric over steps 1-2 alone
    trimmed = ts_auc_ragged([s[1:] for s in scores], [lab[1:] for lab in labels])
    assert ts_auc_ragged(scores, labels) == pytest.approx(trimmed, abs=1e-12)


def test_feature_modules_are_prefix_invariant():
    """Row t of a feature may depend only on history and online[:t+1]."""
    base = pytest.importorskip("sbr.features.base")
    base.load_all()
    rng = np.random.default_rng(7)
    hist = rng.normal(size=1200)
    online = np.concatenate([rng.normal(size=90), rng.normal(0.4, 1.6, size=110)])
    assert base.REGISTRY, "no feature modules registered"
    for name in sorted(base.REGISTRY):
        ok, msg = base.check_prefix_invariance(name, hist, online, cuts=(3, 17, 64))
        assert ok, msg
