"""§30 scorer parity: `ts_auc_flat` against a literal implementation of the
documented metric, on the REAL series geometry.

Needs no feature store. Labels, tau and online lengths come from
`research/folds/folds.parquet`, which is committed, so this validates the
measuring instrument on the actual data shape before any candidate is scored.

`sbr.metric` collapses the per-timestep AUC into a single global rank ratio for
speed. That is an optimisation of the documented metric, and this test is what
makes it an optimisation rather than an approximation.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from sbr.metric import ts_auc_flat

FOLDS = os.path.join(os.path.dirname(__file__), "..", "research", "folds", "folds.parquet")


@pytest.fixture(scope="module")
def flat():
    if not os.path.exists(FOLDS):
        pytest.skip("canonical folds.parquet not present")
    f = pd.read_parquet(FOLDS)
    dev = f[f.fold >= 0].sample(200, random_state=0)
    sid, tt, lab = [], [], []
    for r in dev.itertuples():
        n_on = int(r.n_online)
        t = np.arange(n_on)
        y = (t >= int(r.tau_index)).astype(np.int8) if r.has_break else np.zeros(n_on, np.int8)
        sid.append(np.full(n_on, r.id))
        tt.append(t)
        lab.append(y)
    return np.concatenate(sid), np.concatenate(tt), np.concatenate(lab)


def _brute(scores, labels, t_index):
    """Literal per-timestep metric: ROC AUC across eligible series, weight n_pos*n_neg."""
    num = den = 0.0
    for t in np.unique(t_index):
        m = t_index == t
        y = labels[m]
        npos, nneg = int(y.sum()), int((1 - y).sum())
        if npos == 0 or nneg == 0:
            continue
        w = npos * nneg
        num += w * roc_auc_score(y, scores[m])
        den += w
    return num / den


@pytest.mark.parametrize("kind", ["random", "informative", "heavy_ties", "degenerate"])
def test_ts_auc_matches_the_documented_metric(flat, kind):
    _, tt, lab = flat
    rng = np.random.default_rng(0)
    if kind == "random":
        sc = rng.uniform(size=len(lab))
    elif kind == "informative":
        sc = lab * 0.6 + rng.uniform(size=len(lab))
    elif kind == "heavy_ties":
        sc = np.round(lab * 0.5 + rng.uniform(size=len(lab)), 1)
    else:
        sc = np.full(len(lab), 0.5)
    assert ts_auc_flat(sc, lab, tt) == pytest.approx(_brute(sc, lab, tt), abs=1e-12)


def test_single_class_timesteps_carry_zero_weight(flat):
    """A timestep with only one class must not contribute, not silently count 0.5."""
    _, tt, lab = flat
    rng = np.random.default_rng(1)
    sc = rng.uniform(size=len(lab))
    late = tt >= np.percentile(tt, 99)     # sparse tail where classes often collapse
    keep = ~late
    assert ts_auc_flat(sc[keep], lab[keep], tt[keep]) == pytest.approx(
        _brute(sc[keep], lab[keep], tt[keep]), abs=1e-12)


def test_perfect_and_inverted_scores_bracket_the_range(flat):
    _, tt, lab = flat
    assert ts_auc_flat(lab.astype(float), lab, tt) == pytest.approx(1.0, abs=1e-12)
    assert ts_auc_flat(-lab.astype(float), lab, tt) == pytest.approx(0.0, abs=1e-12)
