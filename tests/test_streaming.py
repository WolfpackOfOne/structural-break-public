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


def test_streaming_is_deterministic():
    """The platform re-runs inference on 10 % of the data and requires agreement
    to 1e-8; non-deterministic solutions are ineligible for rewards.

    Our inference path has no RNG and carries no cross-series state, so this
    should hold by construction -- which is exactly the kind of claim worth a
    test rather than an assertion.
    """
    base.load_all()
    hist, online = _series(seed=11)
    mods = ["m00_core"]
    runs = []
    for _ in range(2):
        st = streaming.IncrementalStreamer(hist, mods)
        runs.append(np.array([st.update(x) for x in online]))
    assert np.array_equal(runs[0], runs[1], equal_nan=True), "streaming output is not reproducible"


def test_streaming_carries_no_cross_series_state():
    """Scoring another series first must not change this series' scores.

    Cross-series state is what would be needed to approximate within-timestep
    rank normalisation at inference. The organisers warn it breaks determinism
    under their parallelism, so the property we want is its absence.
    """
    base.load_all()
    h1, o1 = _series(seed=1)
    h2, o2 = _series(seed=2)
    mods = ["m00_core"]

    st = streaming.IncrementalStreamer(h1, mods)
    alone = np.array([st.update(x) for x in o1])

    warm = streaming.IncrementalStreamer(h2, mods)
    for x in o2:
        warm.update(x)
    st2 = streaming.IncrementalStreamer(h1, mods)
    after = np.array([st2.update(x) for x in o1])

    assert np.array_equal(alone, after, equal_nan=True), "a prior series changed this series' output"
