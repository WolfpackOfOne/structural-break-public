"""Gates for the three wave-5 feature modules.

Three properties are asserted, because each one has already been violated once
during development and none of them is visible in a score:

1.  CAUSALITY, bitwise.  Rebuilding on a truncated online segment must reproduce
    the surviving rows exactly.  The first `m12_rdep` sized its expanding nulls
    by `n_online` -- the single forbidden input in this competition -- and this
    is the check that caught it.

2.  NO LENGTH DEPENDENCE.  A stronger form of the same thing: two online
    segments sharing a prefix, of different total lengths, must agree on the
    shared rows.  This is what `tests/test_production_contract.py` does for the
    production modules.

3.  m11_focus's PRUNING IS EXACT.  The hull kernel must reproduce a brute-force
    maximisation over every candidate tau, bit for bit, on the maximum, its
    inferred age and the anchored statistic.  If it ever stops doing that, the
    module's central claim -- an exact statistic where the incumbent bank uses a
    dyadic approximation -- is false.
"""
from __future__ import annotations

import numpy as np
import pytest

from sbr.features.base import check_prefix_invariance, load_all, make_ctx

MODULES = ("m10_persist", "m11_focus", "m12_rdep")


def _series(rng, n_hist, n_online, kind="plain"):
    h = rng.normal(size=n_hist)
    o = rng.normal(size=n_online)
    if kind == "shift":
        o[n_online // 2:] += 1.5
    elif kind == "scale":
        o[n_online // 2:] *= 2.5
    elif kind == "outlier":
        o[n_online // 3] += 12.0
    elif kind == "dep":
        for i in range(1, n_online):
            o[i] += 0.7 * o[i - 1]
    return h, o


@pytest.fixture(scope="module", autouse=True)
def _registry():
    load_all()


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("n_online,kind", [
    (10, "plain"), (37, "outlier"), (120, "shift"), (200, "scale"), (300, "dep"),
])
def test_prefix_invariant_bitwise(module, n_online, kind):
    rng = np.random.default_rng(hash((module, n_online, kind)) % 2 ** 32)
    h, o = _series(rng, 1500, n_online, kind)
    ok, msg = check_prefix_invariance(module, h, o, cuts=(1, 3, 10, 37, 113), atol=0.0)
    assert ok, msg


@pytest.mark.parametrize("module", MODULES)
def test_no_online_length_dependence(module):
    """Rows shared by a short and a long online segment must be identical.

    A module that reads `len(online)` anywhere -- for a grid, a null length, a
    buffer -- fails here even if it never emits the length as a feature.
    """
    from sbr.features.base import REGISTRY
    rng = np.random.default_rng(7)
    h = rng.normal(size=2000)
    long_o = rng.normal(size=400)
    short_o = long_o[:150]
    _, A = REGISTRY[module].fn(make_ctx(h, long_o))
    _, B = REGISTRY[module].fn(make_ctx(h, short_o))
    a = np.asarray(A[:150], dtype=np.float64)
    b = np.asarray(B, dtype=np.float64)
    bad = ~np.isclose(a, b, rtol=0, atol=0.0, equal_nan=True)
    assert not bad.any(), f"{module}: {int(bad.any(axis=0).sum())} columns depend on n_online"


@pytest.mark.parametrize("n,kind", [
    (25, "plain"), (60, "shift"), (150, "outlier"), (340, "scale"), (400, "dep"),
])
def test_focus_hull_pruning_is_exact(n, kind):
    """The convex-hull candidate set must not cost a single bit of the maximum."""
    import sbr.features.m11_focus as M
    rng = np.random.default_rng(hash((n, kind)) % 2 ** 32)
    _, x = _series(rng, 10, n, kind)
    brute = M._focus_brute(x)
    hull = M._focus(x)
    for j, name in ((0, "stat"), (1, "age"), (3, "anchored")):
        assert np.array_equal(np.asarray(brute[j]), np.asarray(hull[j])), \
            f"hull pruning changed {name}: max diff " \
            f"{np.max(np.abs(np.asarray(brute[j]) - np.asarray(hull[j])))}"


def test_focus_detects_a_mean_shift_it_should():
    """A sanity floor: the statistic must actually respond to what it is for.

    Not a performance claim -- a wiring check.  A module can be perfectly causal,
    perfectly fast and completely inert.
    """
    import sbr.features.m11_focus as M
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    y = x.copy()
    y[200:] += 1.2
    s_flat = M._focus(x)[0][-1]
    s_break = M._focus(y)[0][-1]
    assert s_break > 5 * s_flat, f"flat {s_flat:.2f} vs shifted {s_break:.2f}"
    # and it should localise near the true change point
    age = M._focus(y)[1][-1]
    assert 120 <= age <= 260, f"inferred age {age} does not bracket the true 200"
