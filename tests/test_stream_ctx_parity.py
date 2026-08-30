"""Bitwise parity: StreamCtx (incremental) vs make_ctx (batch).

The whole streaming engine rests on this file.  If the shared context is not
bitwise identical, no module built on it can be.
"""
import numpy as np
import pytest

from sbr.features.base import make_ctx
from sbr.stream.ctx import TRANSFORM_NAMES, StreamCtx


def _synthetic(rng, n_hist, n_online, kind="mix"):
    h = rng.standard_normal(n_hist)
    if kind == "ar":
        for i in range(1, n_hist):
            h[i] += 0.6 * h[i - 1]
    elif kind == "t":
        h = rng.standard_t(3, n_hist)
    o = rng.standard_normal(n_online) * (2.0 if kind == "scale" else 1.0)
    return h, o


def _check(h, o):
    bctx = make_ctx(h, o)
    sctx = StreamCtx().fit_historical(h)
    bad = []
    for t in range(len(o)):
        sctx.push(o[t])
        for k in TRANSFORM_NAMES:
            a = bctx.tr[k][t] if k in bctx.tr else 0.0
            b = sctx.tr[k][t]
            if not (a == b or (np.isnan(a) and np.isnan(b))):
                bad.append(("tr", k, t, a, b))
            ca = bctx.cum[k][t + 1] if k in bctx.cum else 0.0
            cb = sctx.cum[k][t + 1]
            if not (ca == cb or (np.isnan(ca) and np.isnan(cb))):
                bad.append(("cum", k, t, ca, cb))
        for w in (8, 16, 32, 64, 128, 256):
            a = bctx.roll("sq", w)[t]
            b = sctx.roll("sq", w)
            if not (a == b or (np.isnan(a) and np.isnan(b))):
                bad.append(("roll", w, t, a, b))
        a = bctx.expand("abs")[t]
        b = sctx.expand("abs")
        if not (a == b or (np.isnan(a) and np.isnan(b))):
            bad.append(("expand", "abs", t, a, b))
    return bad


@pytest.mark.parametrize("kind", ["mix", "ar", "t", "scale"])
def test_ctx_parity_bitwise(kind):
    rng = np.random.default_rng(hash(kind) % 2**31)
    for _ in range(6):
        n_hist = int(rng.integers(1000, 5000))
        n_online = int(rng.integers(10, 400))
        h, o = _synthetic(rng, n_hist, n_online, kind)
        bad = _check(h, o)
        assert not bad, f"{kind}: {len(bad)} mismatches, first={bad[:3]}"


def test_ctx_parity_real_store():
    from sbr.store import load_store
    st = load_store()
    rng = np.random.default_rng(7)
    for i in rng.integers(0, st.n_series, 12):
        h, o, _ = st.series(int(i))
        bad = _check(h, o)
        assert not bad, f"series {i}: {len(bad)} mismatches, first={bad[:3]}"


def test_hist_side_identical():
    rng = np.random.default_rng(0)
    h = rng.standard_normal(3000)
    o = rng.standard_normal(50)
    b = make_ctx(h, o)
    s = StreamCtx().fit_historical(h)
    assert b.hp.mu == s.hp.mu and b.hp.sd == s.hp.sd and b.hp.ar_sigma == s.hp.ar_sigma
    assert np.array_equal(b.hp.ar_coef, s.hp.ar_coef)
    assert np.array_equal(b.hp.ecdf_x, s.hp.ecdf_x)
    for k, v in b.hist_tr.items():
        assert np.array_equal(v, s.hist_tr[k]), k
    for key, v in b.nc.null_sorted.items():
        assert np.array_equal(v, s.nc.null_sorted[key]), key
