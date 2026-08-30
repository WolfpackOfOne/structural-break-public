"""The n_online gate.  This is the test that protects the whole competition.

A feature built from `(t+1) / n_online` scores ~0.6295 by itself -- BETTER than
the legitimate seven-stream champion.  So any accidental sight of the online
segment's total length does not merely inflate a development number, it produces
a system that cannot be reproduced on the private set and invalidates the entry.

tests/test_production_contract.py already checks the FEATURES for length
dependence, by comparing a short online segment against a longer one sharing its
prefix.  That is necessary and not sufficient: it cannot see a `len(x_online)`
call whose result is used for anything other than a feature value -- a grid, a
buffer size, a normalisation, a branch.

These tests make the length physically unavailable.  `infer` is handed iterables
that RAISE on `len()`, on indexing, and on a second pass, and a counter records
how many online points had been pulled at the moment each score was emitted.
A causal single-pass detector pulls exactly k+1 points before emitting score k.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

MODEL_DIR = os.environ.get(
    "SBR_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "models", "rt150_ensemble"))
needs_model = pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_DIR, "manifest.json")),
    reason=f"no trained model directory at {MODEL_DIR}; set SBR_MODEL_DIR")


class Hostile:
    """A single-pass stream that refuses every question except 'next'."""

    def __init__(self, values, name="online"):
        self._values = list(values)
        self._name = name
        self.pulled = 0
        self._iterated = False

    def __iter__(self):
        if self._iterated:
            raise AssertionError(f"{self._name} iterable was consumed twice")
        self._iterated = True
        for v in self._values:
            self.pulled += 1
            yield v

    def __len__(self):
        raise AssertionError(f"infer called len() on the {self._name} stream -- n_online leak")

    def __getitem__(self, k):
        raise AssertionError(f"infer indexed the {self._name} stream -- n_online leak")

    def __reversed__(self):
        raise AssertionError(f"infer reversed the {self._name} stream -- future access")


def _hist(rng, n=1200):
    return rng.standard_normal(n)


@needs_model
def test_infer_never_measures_the_online_length():
    from sbr.production.submission import infer

    rng = np.random.default_rng(0)
    streams = [Hostile(rng.standard_normal(n)) for n in (10, 137, 400)]
    hists = [_hist(rng) for _ in streams]
    ds = Hostile([(h, s) for h, s in zip(hists, streams)], name="datasets")

    g = infer(ds, MODEL_DIR)
    assert next(g) is None, "the readiness handshake must yield nothing"
    out = list(g)

    assert len(out) == sum(len(s._values) for s in streams)
    assert all(np.isfinite(v) and 0.0 <= v <= 1.0 for v in out)


@needs_model
def test_score_k_is_emitted_after_exactly_k_plus_one_points():
    """The causality gate, stated as an accounting identity.

    If the detector ever read ahead -- buffered, looked at a window of future
    points, or drained the iterator to size something -- the pull count at the
    moment score k comes out would exceed k+1.
    """
    from sbr.production.submission import infer

    rng = np.random.default_rng(1)
    stream = Hostile(rng.standard_normal(60))
    ds = Hostile([(_hist(rng), stream)])

    g = infer(ds, MODEL_DIR)
    next(g)
    for k, _score in enumerate(g):
        assert stream.pulled == k + 1, (
            f"score {k} was emitted after {stream.pulled} points had been read; "
            f"a causal single-pass detector reads exactly {k + 1}")


@needs_model
def test_a_prefix_scores_identically_however_the_series_continues():
    """Two series sharing a prefix must score identically on that prefix.

    Run through `infer`, not through the engine, so the boosters and the
    calibration are in the path too -- a calibrator keyed on anything
    length-derived would show up here and nowhere else.
    """
    from sbr.production.submission import infer

    rng = np.random.default_rng(2)
    h = _hist(rng)
    prefix = rng.standard_normal(80)
    tail = rng.standard_normal(300) * 4.0        # a violent regime change AFTER the prefix

    def scores(online):
        g = infer(Hostile([(h, Hostile(online))]), MODEL_DIR)
        next(g)
        return np.array(list(g))

    short = scores(prefix)
    long = scores(np.r_[prefix, tail])[:len(prefix)]
    assert np.array_equal(short, long), (
        "the same prefix scored differently depending on what came after it")


@needs_model
def test_the_gate_itself_catches_a_length_peek():
    """A test that cannot fail is not a gate.  Prove Hostile actually bites."""
    stream = Hostile([1.0, 2.0, 3.0])
    with pytest.raises(AssertionError, match="len"):
        len(stream)
    with pytest.raises(AssertionError, match="indexed"):
        stream[0]
    list(iter(stream))
    with pytest.raises(AssertionError, match="consumed twice"):
        list(iter(stream))
