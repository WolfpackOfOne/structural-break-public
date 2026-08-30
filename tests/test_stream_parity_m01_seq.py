"""Bitwise parity: StreamM01Seq (incremental) vs the batch ``m01_seq`` module.

The batch module is the specification.  Every assertion here is ``atol=0``,
NaN==NaN -- there is no tolerance anywhere, because every statistic in
``m01_seq`` is a sequential recursion and a sequential recursion is bitwise
reproducible one element at a time.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from conftest import skip_on_missing_store

from sbr.features.base import REGISTRY, load_all, make_ctx
from sbr.stream.ctx import StreamCtx
from sbr.stream.s_m01_seq import StreamM01Seq

load_all()
MODULE = "m01_seq"


# --------------------------------------------------------------------- helpers
def _stream(h, o):
    """Run the streaming engine over ``o`` and return the (n, k) row stack."""
    ctx = StreamCtx()
    ctx.fit_historical(h)
    eng = StreamM01Seq()
    eng.fit_historical(ctx)
    rows = []
    for x in o:
        ctx.push(x)
        r = eng.step(ctx)
        assert r.dtype == np.float64
        rows.append(r)
    return np.asarray(rows, dtype=np.float64), eng


def _batch(h, o):
    names, A = REGISTRY[MODULE].fn(make_ctx(h, o))
    return list(names), np.asarray(A, dtype=np.float64)


def _assert_parity(h, o, tag):
    names, A = _batch(h, o)
    B, eng = _stream(h, o)
    assert eng.cols == names, f"{tag}: column names/order differ"
    assert B.shape == A.shape, f"{tag}: shape {B.shape} vs {A.shape}"
    same = (A == B) | (np.isnan(A) & np.isnan(B))
    if not same.all():
        bad = ~same
        jj = np.where(bad.any(axis=0))[0]
        i, j = np.argwhere(bad)[0]
        raise AssertionError(
            f"{tag}: {int(bad.sum())} mismatched cells in columns "
            f"{[names[k] for k in jj][:8]}; first at row {i} col {names[j]}: "
            f"batch={A[i, j]!r} stream={B[i, j]!r}"
        )


# ------------------------------------------------------------------- synthetic
def _vol_cluster(rng, n):
    """GARCH-ish volatility clustering."""
    s = np.ones(n)
    x = np.empty(n)
    for i in range(n):
        x[i] = rng.standard_normal() * s[i]
        if i + 1 < n:
            s[i + 1] = np.sqrt(0.02 + 0.10 * x[i] ** 2 + 0.88 * s[i] ** 2)
    return x


def _ar1(rng, n, phi):
    x = rng.standard_normal(n)
    for i in range(1, n):
        x[i] += phi * x[i - 1]
    return x


def _cases():
    """>= 40 synthetic (hist, online, tag) triples spanning the hard regimes."""
    rng = np.random.default_rng(20260819)
    out = []

    # --- short online (10-20) --------------------------------------------
    for k in range(5):
        n = int(rng.integers(10, 21))
        out.append((rng.standard_normal(1500), rng.standard_normal(n), f"short{k}_n{n}"))

    # --- long online (900-999) -------------------------------------------
    for k in range(4):
        n = int(rng.integers(900, 1000))
        out.append((rng.standard_normal(4000), rng.standard_normal(n), f"long{k}_n{n}"))

    # --- constant history -------------------------------------------------
    out.append((np.full(1200, 3.0), rng.standard_normal(120), "consthist_normalonline"))
    out.append((np.full(1200, 3.0), np.full(60, 3.0), "consthist_constonline"))
    out.append((np.zeros(2000), rng.standard_normal(200) * 5.0, "zerohist"))

    # --- near-zero-variance history --------------------------------------
    out.append((rng.standard_normal(1500) * 1e-12, rng.standard_normal(150) * 1e-12,
                "tinyvar_both"))
    out.append((rng.standard_normal(1500) * 1e-12, rng.standard_normal(150),
                "tinyvar_hist_bigonline"))
    out.append((np.full(1000, 1.0) + rng.standard_normal(1000) * 1e-15,
                rng.standard_normal(80) * 1e-15 + 1.0, "nearconst"))

    # --- heavy-tailed (t3) history ---------------------------------------
    for k in range(4):
        out.append((rng.standard_t(3, int(rng.integers(1000, 3500))),
                    rng.standard_t(3, int(rng.integers(50, 700))), f"t3_{k}"))
    out.append((rng.standard_t(2, 2000), rng.standard_t(2, 300), "t2"))

    # --- strong AR(1) history --------------------------------------------
    for phi in (0.9, -0.85, 0.98):
        out.append((_ar1(rng, 3000, phi), _ar1(rng, 400, phi), f"ar1_{phi}"))
    out.append((_ar1(rng, 2500, 0.95), rng.standard_normal(300), "ar1hist_iidonline"))

    # --- volatility clustered --------------------------------------------
    for k in range(3):
        out.append((_vol_cluster(rng, 2500), _vol_cluster(rng, 400), f"volclust{k}"))
    out.append((_vol_cluster(rng, 2000), rng.standard_normal(250) * 6.0, "volclust_varbreak"))

    # --- break at tau = 0 --------------------------------------------------
    for _k, shift in enumerate((0.5, 3.0, 20.0)):
        o = rng.standard_normal(300) + shift
        out.append((rng.standard_normal(2500), o, f"tau0_shift{shift}"))
    o = rng.standard_normal(300) * 8.0
    out.append((rng.standard_normal(2500), o, "tau0_varshift"))

    # --- break at the LAST point ------------------------------------------
    for _k, shift in enumerate((3.0, 50.0)):
        o = rng.standard_normal(400)
        o[-1] += shift
        out.append((rng.standard_normal(2500), o, f"lastbreak{shift}"))
    o = rng.standard_normal(12)
    o[-1] += 30.0
    out.append((rng.standard_normal(1500), o, "lastbreak_shortonline"))

    # --- huge single outliers ---------------------------------------------
    for _k, mag in enumerate((1e6, 1e12, 1e30, -1e30)):
        o = rng.standard_normal(300)
        o[137] = mag
        out.append((rng.standard_normal(2500), o, f"outlier{mag:g}"))
    h = rng.standard_normal(2500)
    h[10] = 1e20
    out.append((h, rng.standard_normal(200), "hist_outlier"))

    # --- mid-series break + assorted mixtures ------------------------------
    for k in range(6):
        n = int(rng.integers(40, 800))
        o = rng.standard_normal(n)
        tau = n // 2
        o[tau:] += float(rng.uniform(-4, 4))
        o[tau:] *= float(rng.uniform(0.2, 5.0))
        out.append((rng.standard_normal(int(rng.integers(1000, 5000))), o, f"mid{k}_n{n}"))

    # --- exactly the SLOPE_K boundary (16/17 online points) ----------------
    out.append((rng.standard_normal(1200), rng.standard_normal(16), "slopek_16"))
    out.append((rng.standard_normal(1200), rng.standard_normal(17), "slopek_17"))
    out.append((rng.standard_normal(1200), rng.standard_normal(1), "n1"))

    return out


CASES = _cases()


def test_case_count():
    assert len(CASES) >= 40, f"only {len(CASES)} synthetic cases"


@pytest.mark.parametrize("i", range(len(CASES)))
def test_bitwise_parity_synthetic(i):
    h, o, tag = CASES[i]
    _assert_parity(np.asarray(h, float), np.asarray(o, float), tag)


# ------------------------------------------------------------------- real data
def _real_indices(k=30):
    from sbr.store import load_store

    st = skip_on_missing_store(load_store)
    rng = np.random.default_rng(4242)
    return st, sorted({int(i) for i in rng.integers(0, st.n_series, k + 8)})[:k]


def test_bitwise_parity_real_store():
    st, idx = _real_indices(30)
    assert len(idx) >= 30
    for i in idx:
        h, o, _tau = st.series(i)
        _assert_parity(h, o, f"real[{i}]")


# ----------------------------------------------------------------- column shape
def test_columns_match_batch():
    rng = np.random.default_rng(1)
    h, o = rng.standard_normal(2000), rng.standard_normal(50)
    names, A = _batch(h, o)
    eng = StreamM01Seq()
    assert eng.cols == names
    assert A.shape[1] == len(names) == 60
    assert len(set(names)) == len(names), "duplicate column names"


# ------------------------------------------------------------- prefix / poison
def test_prefix_and_poison():
    """A row emitted at t can never be changed by anything that arrives later.

    Structurally guaranteed (the engine only ever reads ``ctx`` at the current
    index and its own carried scalars), but tested: we run the stream on a
    prefix, record every row, then continue with deliberately poisonous data
    (huge outliers, NaN, inf) and re-run the whole thing.  The first ``k`` rows
    must be bit-identical, and identical to batch on the truncated series.
    """
    rng = np.random.default_rng(99)
    h = rng.standard_normal(2500)
    o = rng.standard_normal(300)
    k = 137

    ref, _ = _stream(h, o[:k])
    _, batch_pref = _batch(h, o[:k])
    assert np.array_equal(ref, batch_pref, equal_nan=True)

    for poison in (
        np.full(163, 1e30),
        np.full(163, -np.inf),
        np.full(163, np.nan),
        rng.standard_normal(163) * 1e9,
    ):
        o2 = np.concatenate([o[:k], poison])
        full, _ = _stream(h, o2)
        assert np.array_equal(full[:k], ref, equal_nan=True), "later data changed row <= k"

    # and the streaming rows equal batch rows on the *untruncated* series too
    full, _ = _stream(h, o)
    _, A = _batch(h, o)
    assert np.array_equal(full, A, equal_nan=True)
    assert np.array_equal(full[:k], ref, equal_nan=True)


def test_step_is_pure_wrt_future():
    """Interleaved check: batch row t equals the stream row emitted at t."""
    rng = np.random.default_rng(5)
    h = rng.standard_normal(2000)
    o = rng.standard_normal(220)
    _, A = _batch(h, o)
    ctx = StreamCtx()
    ctx.fit_historical(h)
    eng = StreamM01Seq()
    eng.fit_historical(ctx)
    for t, x in enumerate(o):
        ctx.push(x)
        r = eng.step(ctx)
        assert np.array_equal(r, A[t], equal_nan=True), f"row {t}"


# ------------------------------------------------------------------- timing
def test_timing_microseconds_per_obs(capsys):
    rng = np.random.default_rng(7)
    h = rng.standard_normal(3000)
    o = rng.standard_normal(999)

    # warm up (import/compile caches)
    ctx = StreamCtx()
    ctx.fit_historical(h)
    eng = StreamM01Seq()
    eng.fit_historical(ctx)
    for x in o[:20]:
        ctx.push(x)
        eng.step(ctx)

    best_step = float("inf")
    best_push = float("inf")
    best_fit = float("inf")
    for _ in range(5):
        ctx = StreamCtx()
        ctx.fit_historical(h)
        eng = StreamM01Seq()
        t0 = time.perf_counter()
        eng.fit_historical(ctx)
        best_fit = min(best_fit, time.perf_counter() - t0)
        t0 = time.perf_counter()
        for x in o:
            ctx.push(x)
            eng.step(ctx)
        best_step = min(best_step, time.perf_counter() - t0)

        ctx2 = StreamCtx()
        ctx2.fit_historical(h)
        t0 = time.perf_counter()
        for x in o:
            ctx2.push(x)
        best_push = min(best_push, time.perf_counter() - t0)

    us_total = best_step / len(o) * 1e6
    us_push = best_push / len(o) * 1e6
    with capsys.disabled():
        print(
            f"\n[m01_seq] fit_historical = {best_fit * 1e3:.2f} ms/series | "
            f"StreamCtx.push + StreamM01Seq.step = {us_total:.1f} us/obs "
            f"(of which shared ctx.push = {us_push:.1f}) | "
            f"StreamM01Seq.step alone = {us_total - us_push:.1f} us/obs"
        )
    # generous ceiling: this only guards against an accidental O(t) rescan
    assert us_total < 3000.0
