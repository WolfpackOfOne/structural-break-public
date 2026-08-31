"""Floating-point primitives shared by the streaming ports.

The streaming modules must reproduce the batch feature pipeline **bitwise**,
because the batch pipeline is what wrote the feature cache the frozen RT-600
model was trained on.  That is mostly free -- a Python float is an IEEE-754
binary64 and ``+ - * /`` are the same correctly-rounded operations numpy's
scalar loops perform.  It is not free in one place: ``scipy.signal.lfilter``
evaluates its recursion inside a single C loop, and the compiler/architecture
decide whether ``a * b + c`` is contracted into a fused multiply-add.  The
frozen macOS/arm64 RT-600 environment contracts it; GitHub's x86_64 Linux CI
does not.  An FMA rounds once where ``a * b + c`` in Python rounds twice, and
the two disagree by 1 ULP on roughly 20 % of inputs.  In a stateful IIR
recursion that disagreement is carried forward rather than washed out.

``math.fma`` does exactly this, but it arrived in Python 3.13 and the frozen
environment is 3.11.6 (``research/FINAL_REPRODUCIBILITY_MANIFEST.json``), so
:func:`fma` below is the classical Dekker two-product / two-sum emulation.

The stream-vs-batch contract is therefore architecture-scoped: stream code that
mirrors ``lfilter`` calls :func:`lfilter_madd`, which uses single rounding only
on architectures where this repository's supported ``lfilter`` builds do too.
"""
from __future__ import annotations

import platform

#: Dekker splitting constant, 2**27 + 1.
_SPLIT = 134217729.0

_LFILTER_FMA_MACHINES = frozenset({"arm64", "aarch64"})
_LFILTER_USES_FMA = platform.machine().lower() in _LFILTER_FMA_MACHINES


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


def lfilter_uses_fma() -> bool:
    """Whether ``scipy.signal.lfilter`` uses FMA semantics in this environment."""
    return _LFILTER_USES_FMA


def lfilter_madd(a: float, b: float, c: float) -> float:
    """One scalar multiply-add with this host's ``lfilter`` rounding semantics."""
    if _LFILTER_USES_FMA:
        return fma(a, b, c)
    return a * b + c
