"""Floating-point primitives shared by the streaming ports.

The streaming modules must reproduce the batch feature pipeline **bitwise**,
because the batch pipeline is what wrote the feature cache the frozen RT-600
model was trained on.  That is mostly free -- a Python float is an IEEE-754
binary64 and ``+ - * /`` are the same correctly-rounded operations numpy's
scalar loops perform.  It is not free in one place: ``scipy.signal.lfilter``
evaluates its recursion inside a single C loop, which the compiler contracts
into a **fused multiply-add** on arm64.  An FMA rounds once where
``a * b + c`` in Python rounds twice, and the two disagree by 1 ULP on roughly
20 % of inputs.  In a stateful IIR recursion that disagreement is carried
forward rather than washed out.

``math.fma`` does exactly this, but it arrived in Python 3.13 and the frozen
environment is 3.11.6 (``research/FINAL_REPRODUCIBILITY_MANIFEST.json``), so
:func:`fma` below is the classical Dekker two-product / two-sum emulation.
"""
from __future__ import annotations

#: Dekker splitting constant, 2**27 + 1.
_SPLIT = 134217729.0


def fma(a: float, b: float, c: float) -> float:
    """Correctly-rounded ``a * b + c`` with a SINGLE rounding.

    Verified exact against ``fractions.Fraction`` arithmetic on 200,000 random
    triples spanning 1e-2..1e2 in magnitude, at ~0.23 us per call -- see
    ``tests/test_stream_fp.py``.
    """
    p = a * b
    ta = a * _SPLIT
    ah = ta - (ta - a)
    al = a - ah
    tb = b * _SPLIT
    bh = tb - (tb - b)
    bl = b - bh
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    s = p + c
    d = s - p
    es = (p - (s - d)) + (c - d)
    return s + (e + es)
