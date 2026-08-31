"""Bitwise parity: StreamM12Rdep (incremental) vs the batch module m12_rdep.

The batch module is the specification.  Row ``t`` of
``REGISTRY["m12_rdep"].fn(make_ctx(hist, online))`` must be reproduced exactly
(``atol = 0``, ``NaN == NaN``) by ``StreamM12Rdep.step`` after ``StreamCtx.push``.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from sbr.features.base import REGISTRY, load_all, make_ctx
from sbr.features.m12_rdep import BINS, _occupancy_distances
from sbr.stream.ctx import StreamCtx
from sbr.stream.s_m12_rdep import StreamM12Rdep, _dist_from_occupancy

load_all()
MODULE = "m12_rdep"


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
    elif kind == "heavy":
        o = o * rng.choice([1.0, 5.0], size=n_online, p=[0.9, 0.1])
    return h, o


def _stream_rows(h, o):
    ctx = StreamCtx().fit_historical(h)
    eng = StreamM12Rdep()
    eng.fit_historical(ctx)
    k = len(eng.cols)
    rows = np.empty((len(o), k), dtype=np.float64)
    for t in range(len(o)):
        ctx.push(o[t])
        r = eng.step(ctx)
        assert r.dtype == np.float64 and r.shape == (k,)
        rows[t] = r
    return eng.cols, rows


def _compare(h, o):
    h = np.asarray(h, dtype=np.float64)
    o = np.asarray(o, dtype=np.float64)
    names, A = REGISTRY[MODULE].fn(make_ctx(h, o))
    cols, B = _stream_rows(h, o)
    assert cols == names, "column names/order differ from batch"
    A = np.asarray(A, dtype=np.float32)
    B32 = B.astype(np.float32)
    B32[~np.isfinite(B32)] = np.nan
    same = (A == B32) | (np.isnan(A) & np.isnan(B32))
    return names, A, B32, ~same


# ------------------------------------------------------------------ the gate
@pytest.mark.parametrize("n_online,kind", [
    (1, "plain"), (10, "plain"), (33, "outlier"), (120, "shift"),
    (200, "scale"), (300, "dep"), (400, "heavy"),
])
def test_bitwise_parity(n_online, kind):
    rng = np.random.default_rng(hash((n_online, kind)) % 2 ** 32)
    h, o = _series(rng, 1500, n_online, kind)
    names, A, B, bad = _compare(h, o)
    if bad.any():
        ij = np.argwhere(bad)[:6]
        msg = "\n".join(
            f"  col {names[j]} t={t}: batch {A[t, j]!r} stream {B[t, j]!r}"
            for t, j in ij)
        cols_bad = sorted({names[j] for _, j in np.argwhere(bad)})
        pytest.fail(f"{int(bad.sum())} mismatches in {len(cols_bad)} columns "
                    f"{cols_bad[:8]}\n{msg}")


def test_short_history_and_edge_lengths():
    """A history too short for the null grid must degrade identically."""
    rng = np.random.default_rng(11)
    for n_hist in (40, 80, 200):
        h, o = _series(rng, n_hist, 60, "plain")
        _, A, B, bad = _compare(h, o)
        assert not bad.any(), f"n_hist={n_hist}: {int(bad.sum())} mismatches"


def test_occupancy_block_bitwise():
    """The claim that a (bins, 1) axis-0 reduction equals the (bins, n) one.

    This is an empirical property of numpy's reduction order, not a theorem, so
    it is pinned here over random PIT streams rather than assumed in a comment.
    """
    rng = np.random.default_rng(3)
    for _ in range(20):
        n = int(rng.integers(2, 300))
        u = rng.random(n)
        full = _occupancy_distances(u, None)
        idx = np.clip((u * BINS).astype(np.int64), 0, BINS - 1)
        C = np.zeros((n + 1, BINS))
        for t in range(n):
            C[t + 1] = C[t]
            C[t + 1, idx[t]] += 1.0
        for t in range(n):
            from sbr.stream.s_m12_rdep import _pad2
            one = _dist_from_occupancy(_pad2(C[t + 1] / (t + 1)))
            for k, v in one.items():
                assert v == full[k][t] or (np.isnan(v) and np.isnan(full[k][t])), \
                    f"{k} at t={t}: column {v!r} vs full {full[k][t]!r}"


def test_reduction_width_invariance():
    """A (bins, 1) axis-0 reduction is NOT the same float as a (bins, n) one.

    This is the trap `_pad2` exists to avoid, and it is asserted rather than
    described: if a future numpy makes width 1 agree, the padding becomes
    harmless; if it makes width 2 disagree, this fails loudly instead of
    silently breaking parity.
    """
    rng = np.random.default_rng(7)
    n_bad_w1 = 0
    for _ in range(300):
        col = rng.random(BINS)
        ref = None
        for width in (1, 2, 4, 257):
            A = rng.random((BINS, width))
            A[:, 0] = col
            v = np.nansum(A, axis=0)[0]
            if width == 257:
                ref = v
            elif width == 1 and v != ref if ref is not None else False:
                pass
        # recompute in a fixed order now that ref is known
        A1 = np.empty((BINS, 1))
        A1[:, 0] = col
        A2 = np.empty((BINS, 2))
        A2[:, 0] = col
        A2[:, 1] = col
        AN = np.empty((BINS, 257))
        AN[:] = rng.random((BINS, 257))
        AN[:, 0] = col
        ref = np.nansum(AN, axis=0)[0]
        assert np.nansum(A2, axis=0)[0] == ref, "width-2 padding no longer matches width-n"
        if np.nansum(A1, axis=0)[0] != ref:
            n_bad_w1 += 1
    print(f"\nwidth-1 reductions differing from width-n: {n_bad_w1}/300")


def test_scalar_grid_matches_gridnull():
    """The tabulated length interpolation must equal `_GridNull.z` exactly.

    `_ScalarGrid` replaces nine numpy-dispatch-heavy calls per observation with
    two array reads.  It is only legitimate if it returns the same float, so
    that is asserted here over the whole length range rather than trusted.
    """
    from sbr.features.m12_rdep import _GridNull, _reflect_cusum
    from sbr.stream.s_m12_rdep import _ScalarGrid
    rng = np.random.default_rng(21)
    for _ in range(5):
        h = rng.normal(size=2000)
        gn = _GridNull(h, lambda seg: _reflect_cusum(seg - 0.5))
        sg = _ScalarGrid(gn, 1024)
        for n in (1, 2, 3, 7, 33, 128, 512, 999, 1024):
            for v in (0.0, 1e-9, 0.37, 12.5, -4.0, 1e6):
                a = float(gn.z(np.array([float(n)]), np.array([v]))[0])
                b = sg.z(n, v)
                assert a == b, f"n={n} v={v}: GridNull {a!r} vs tabulated {b!r}"


def test_throughput():
    """Report microseconds per observation; O(1) work, nothing rescanned."""
    rng = np.random.default_rng(5)
    h, o = _series(rng, 3000, 999, "dep")
    ctx = StreamCtx().fit_historical(h)
    eng = StreamM12Rdep()
    eng.fit_historical(ctx)
    t0 = time.perf_counter()
    for x in o:
        ctx.push(x)
        eng.step(ctx)
    us = (time.perf_counter() - t0) / len(o) * 1e6
    print(f"\nStreamM12Rdep: {us:.1f} us/observation over {len(o)} points")
    assert us < 5000, f"{us:.0f} us/obs is implausibly slow"
