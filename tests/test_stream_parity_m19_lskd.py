"""Bitwise parity: StreamM19Lskd / StreamM19Lskm vs the batch RT-1321 modules.

The batch modules are the specification.  Row ``t`` of
``REGISTRY["m19_lskd"].fn(make_ctx(hist, online))`` must be reproduced exactly
(``atol = 0``, ``NaN == NaN``) by ``step`` after ``StreamCtx.push``.

Two numerical facts make this reachable and are pinned here rather than assumed,
because both were measured to fail before the batch module was written the way
it is:

* the joint RFF projection must NOT be a ``Z @ W.T`` matmul -- BLAS ``gemm``
  regroups the length-``d`` inner sum differently for one row than for many, and
  disagreed in the last ulp on ~25% of entries;
* the Page accumulation must be carried as batch's own ``cumsum`` and running
  minimum, not as the algebraically equivalent ``max(0, S + a - k)`` recursion.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from sbr.features import m19_lskd as M
from sbr.features.base import REGISTRY, load_all, make_ctx
from sbr.stream.ctx import StreamCtx
from sbr.stream.s_m19_lskd import StreamM19Lskd, StreamM19Lskm

load_all()
PAIRS = (("m19_lskd", StreamM19Lskd), ("m19_lskm", StreamM19Lskm))


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
    elif kind == "const_hist":
        h = np.zeros(n_hist)
    elif kind == "tinyvar_hist":
        h = h * 1e-12
    elif kind == "t3_hist":
        h = rng.standard_t(3, size=n_hist)
    elif kind == "ar_hist":
        for i in range(1, n_hist):
            h[i] += 0.85 * h[i - 1]
    elif kind == "volclust":
        v = np.ones(n_online)
        for i in range(1, n_online):
            v[i] = 0.9 * v[i - 1] + 0.1 * abs(o[i - 1])
        o = o * v
    elif kind == "break_at_0":
        o += 3.0
    elif kind == "break_at_last":
        o[-1] += 20.0
    return h, o


def _stream_rows(cls, h, o):
    ctx = StreamCtx().fit_historical(np.asarray(h, dtype=np.float64))
    eng = cls()
    eng.fit_historical(ctx)
    k = len(eng.cols)
    rows = np.empty((len(o), k), dtype=np.float64)
    for t in range(len(o)):
        ctx.push(float(o[t]))
        r = eng.step(ctx)
        assert r.dtype == np.float64 and r.shape == (k,)
        rows[t] = r
    return eng.cols, rows


def _compare(module, cls, h, o):
    h = np.asarray(h, dtype=np.float64)
    o = np.asarray(o, dtype=np.float64)
    names, A = REGISTRY[module].fn(make_ctx(h, o))
    cols, B = _stream_rows(cls, h, o)
    assert cols == names, "column names/order differ from batch"
    A = np.asarray(A, dtype=np.float32)
    B32 = B.astype(np.float32)
    B32[~np.isfinite(B32)] = np.nan
    same = (A == B32) | (np.isnan(A) & np.isnan(B32))
    return names, A, B32, ~same


@pytest.mark.parametrize("module,cls", PAIRS)
@pytest.mark.parametrize("n_hist,n_online,kind", [
    (1500, 1, "plain"), (1500, 12, "plain"), (1500, 18, "outlier"),
    (1500, 120, "shift"), (1500, 250, "scale"), (1500, 300, "dep"),
    (1500, 400, "heavy"), (1500, 260, "volclust"),
    (900, 999, "plain"), (2400, 950, "ar_hist"), (1500, 300, "t3_hist"),
    (600, 200, "const_hist"), (600, 200, "tinyvar_hist"),
    (1500, 200, "break_at_0"), (1500, 200, "break_at_last"),
])
def test_bitwise_parity_synthetic(module, cls, n_hist, n_online, kind):
    rng = np.random.default_rng(abs(hash((module, n_hist, n_online, kind))) % 2 ** 32)
    h, o = _series(rng, n_hist, n_online, kind)
    names, A, B, bad = _compare(module, cls, h, o)
    if bad.any():
        ij = np.argwhere(bad)[:8]
        msg = "\n".join(f"  col {names[j]} t={t}: batch {A[t, j]!r} stream {B[t, j]!r}"
                        for t, j in ij)
        pytest.fail(f"{module}: {int(bad.sum())} of {bad.size} cells differ\n{msg}")


@pytest.mark.parametrize("module,cls", PAIRS)
def test_bitwise_parity_real_series(module, cls, store):
    rng = np.random.default_rng(20260904)
    idx = rng.choice(store.n_series, 30, replace=False)
    for i in idx:
        h, o, _ = store.series(int(i))
        names, A, B, bad = _compare(module, cls, h, o)
        if bad.any():
            j = int(np.argwhere(bad)[0][1])
            pytest.fail(f"{module}: real series {int(i)} differs, first column {names[j]}")


@pytest.mark.parametrize("module,cls", PAIRS)
def test_stream_cannot_see_the_future(module, cls):
    """Record row t, then poison every later observation: row t must not move."""
    rng = np.random.default_rng(4)
    h, o = _series(rng, 1200, 300, "shift")
    _, full = _stream_rows(cls, h, o)
    for cut in (1, 40, 199):
        poisoned = o.copy()
        poisoned[cut:] = 1e6
        _, other = _stream_rows(cls, h, poisoned)
        assert np.array_equal(np.nan_to_num(full[:cut], nan=-7.7),
                              np.nan_to_num(other[:cut], nan=-7.7))


@pytest.mark.parametrize("module,cls", PAIRS)
def test_timing(module, cls, capsys):
    rng = np.random.default_rng(9)
    h, o = _series(rng, 2000, 504, "dep")
    _stream_rows(cls, h, o[:20])                       # warm numba
    t0 = time.perf_counter()
    _stream_rows(cls, h, o)
    us = (time.perf_counter() - t0) / len(o) * 1e6
    with capsys.disabled():
        print(f"\n{module} streaming: {us:.1f} us/observation")
    assert us < 2000.0


def test_row_batch_projection_is_bitwise_identical():
    """The property the whole parity contract rests on."""
    rng = np.random.default_rng(0)
    for d in M.DEPTHS:
        Z = rng.uniform(0.0, 1.0, size=(400, d))
        for f in (M.phi_joint, M.phi_marginal):
            full = f(Z, d, 0.63)
            rowwise = np.vstack([f(Z[i:i + 1], d, 0.63) for i in range(len(Z))])
            assert np.array_equal(full, rowwise), (
                f"{f.__name__} d={d} is not row-count independent; batch/stream "
                "parity is unreachable and the batch module must be rewritten, "
                "not the streaming one")


def test_ewma_run_is_the_same_recursion_stepwise():
    rng = np.random.default_rng(2)
    P = rng.normal(size=(250, M.R))
    mu_H = rng.normal(size=M.R)
    lm = M.LAM[128]
    batch = M._ewma_discrepancy(P, mu_H, lm)
    mu = mu_H.copy()
    one = np.empty(1)
    step = np.empty(250)
    for i in range(250):
        M._ewma_run(np.ascontiguousarray(P[i:i + 1]), mu, mu_H, lm, one)
        step[i] = one[0]
    assert np.array_equal(batch, step)
