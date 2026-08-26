"""The streaming ports' floating-point primitives.

``sbr.stream._fp.fma`` exists because ``scipy.signal.lfilter`` contracts its
recursion's multiply-add into a hardware FMA on arm64, while a Python
``a * b + c`` rounds twice.  The batch feature pipeline wrote the cache the
frozen RT-600 model was trained on, so the streaming ports must round the same
way -- see engineering/reports/rt600_final_reliability/STREAM_PARITY_REPRO.md.
"""
from fractions import Fraction

import numpy as np
import pytest
from scipy.signal import lfilter

from sbr.stream._fp import fma


def _exact(a, b, c):
    """The correctly-rounded value of a*b + c, via exact rational arithmetic."""
    return float(Fraction(a) * Fraction(b) + Fraction(c))


def test_fma_is_correctly_rounded():
    rng = np.random.default_rng(20260826)
    n = 50_000
    scales = np.array([1e-3, 1e-2, 1.0, 1e2, 1e3])
    a = rng.standard_normal(n) * rng.choice(scales, n)
    b = rng.standard_normal(n) * rng.choice(scales, n)
    c = rng.standard_normal(n) * rng.choice(scales, n)
    bad = 0
    for x, y, z in zip(a.tolist(), b.tolist(), c.tolist()):
        if fma(x, y, z) != _exact(x, y, z):
            bad += 1
    assert bad == 0, f"{bad}/{n} triples were not correctly rounded"


@pytest.mark.parametrize("a,b,c", [
    (0.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.1, 0.1, -0.01),          # exactly cancelling to a tiny residue
    (1.0, 0.0, -0.0),
    (-1.0, 1.0, 1.0),
])
def test_fma_edge_cases(a, b, c):
    assert fma(a, b, c) == _exact(a, b, c)


def test_fma_reproduces_lfilter_recursion():
    """The reason the primitive exists: a one-pole IIR must match lfilter."""
    alpha = 1.0 / 22.0
    rng = np.random.default_rng(7)
    x = np.abs(rng.standard_normal(500)) * 2.0
    x0 = 1.0

    ref = lfilter([alpha], [1.0, -(1.0 - alpha)], x,
                  zi=np.array([(1.0 - alpha) * x0]))[0]

    got = np.empty_like(ref)
    z = (1.0 - alpha) * x0
    for i, v in enumerate(x.tolist()):
        y = fma(alpha, v, z)
        got[i] = y
        z = (1.0 - alpha) * y
    assert np.array_equal(got, ref), "carried-state FMA recursion != lfilter"

    # and the two-rounding form really does drift, i.e. this test has teeth
    naive = np.empty_like(ref)
    z = (1.0 - alpha) * x0
    for i, v in enumerate(x.tolist()):
        y = alpha * v + z
        naive[i] = y
        z = (1.0 - alpha) * y
    assert not np.array_equal(naive, ref), (
        "the unfused form no longer drifts -- this platform may not contract "
        "lfilter's multiply-add, and the FMA emulation should be re-justified")


def test_scalar_square_matches_numpy():
    """`x * x`, not `x ** 2`: the latter goes through libm pow and differs."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal(200_000) * rng.choice([1e-2, 1.0, 1e2], 200_000)
    np_sq = a ** 2
    assert int((np.array([v * v for v in a.tolist()]) != np_sq).sum()) == 0
    # the wrong spelling is genuinely different, so the distinction matters
    assert int((np.array([v ** 2 for v in a.tolist()]) != np_sq).sum()) > 0
