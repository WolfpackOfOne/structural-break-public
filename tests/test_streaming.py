"""Batch/stream parity — the property the submission depends on.

Training saw the batch feature matrix. Inference will see whatever the streamer
emits. If those differ by even a little, the model is being served inputs it was
never fitted on, and no offline score means anything.

Prefix invariance (tested in test_sbr_metric.py) gives us a free oracle here:
recomputing the batch builder on the growing prefix and taking the last row is by
definition what the batch pipeline would have produced. So the reference streamer
is correct by construction, and every faster streamer is checked against the
batch matrix directly, bitwise.
"""
from __future__ import annotations

import numpy as np
import pytest

base = pytest.importorskip("sbr.features.base")
streaming = pytest.importorskip("sbr.streaming")


def _series(seed=0, n_hist=1200, n_pre=90, n_post=110):
    rng = np.random.default_rng(seed)
    hist = rng.normal(size=n_hist)
    online = np.concatenate([rng.normal(size=n_pre), rng.normal(0.3, 1.7, size=n_post)])
    return hist, online


@pytest.mark.parametrize("n_online", [1, 2, 5, 40])
def test_reference_streamer_is_bitwise_identical_to_batch(n_online):
    base.load_all()
    hist, online = _series()
    online = online[:n_online]
    r = streaming.check_stream_parity(streaming.ReferenceStreamer, hist, online,
                                      ["m00_core"], atol=0.0)
    assert r.ok, r.detail


def test_incremental_streamer_is_bitwise_identical_to_batch():
    """The O(1) path must match the batch matrix exactly, not approximately."""
    base.load_all()
    hist, online = _series(seed=3)
    for m in streaming.STATELESS_MODULES:
        r = streaming.check_stream_parity(streaming.IncrementalStreamer, hist, online,
                                          [m], atol=0.0)
        assert r.ok, f"{m}: {r.detail}"


def test_single_online_point_does_not_crash():
    """n_online == 1 is legal and was a real crash: a lag-2 product built by
    concatenation returned length 2 instead of length 1."""
    base.load_all()
    hist, online = _series()
    ctx = base.make_ctx(hist, online[:1])
    for name, mod in base.REGISTRY.items():
        cols, A = mod.fn(ctx)
        assert A.shape == (1, len(cols)), f"{name} emitted {A.shape} for a single online point"
