"""Bitwise parity: StreamM04Resid (incremental) vs the batch module m04_resid.

The batch module is the specification.  Row ``t`` of
``REGISTRY["m04_resid"].fn(make_ctx(hist, online))`` must be reproduced exactly
(``atol=0``, ``NaN == NaN``) by ``StreamM04Resid.step`` after ``StreamCtx.push``.
"""
import time

import numpy as np
import pytest

from sbr.features.base import REGISTRY, load_all, make_ctx
from sbr.stream.ctx import StreamCtx
from sbr.stream.s_m04_resid import StreamM04Resid

load_all()
MODULE = "m04_resid"


# --------------------------------------------------------------------- helpers
def _stream_rows(h, o):
    """Run the incremental engine and return (cols, rows as float32)."""
    ctx = StreamCtx().fit_historical(h)
    eng = StreamM04Resid()
    eng.fit_historical(ctx)
    cols = eng.cols
    k = len(cols)
    rows = np.empty((len(o), k), dtype=np.float32)
    for t in range(len(o)):
        ctx.push(o[t])
        r = eng.step(ctx)
        assert r.dtype == np.float64 and r.shape == (k,)
        rows[t] = r.astype(np.float32)
    rows[~np.isfinite(rows)] = np.nan
    return eng.cols, rows


def _mismatches(h, o):
    """Return (n_bad, worst) where worst is a list of (col, t, batch, stream)."""
    h = np.asarray(h, dtype=np.float64)
    o = np.asarray(o, dtype=np.float64)
    names, A = REGISTRY[MODULE].fn(make_ctx(h, o))
    cols, B = _stream_rows(h, o)
    assert cols == names, "column names/order differ from batch"
    assert B.shape == A.shape
    same = (A == B) | (np.isnan(A) & np.isnan(B))
    bad = ~same
    worst = []
    if bad.any():
        ij = np.argwhere(bad)[:5]
        worst = [(names[j], int(i), float(A[i, j]), float(B[i, j])) for i, j in ij]
    return int(bad.sum()), worst


# ------------------------------------------------------------------- synthetic
def _synthetic(seed):
    """44 series spanning every regime the contract asks for."""
    rng = np.random.default_rng(seed)
    out = []

    def add(name, h, o):
        out.append((name, np.asarray(h, np.float64), np.asarray(o, np.float64)))

    # plain gaussian, short and long online
    for k, n_on in enumerate((10, 13, 17, 20)):
        add(f"gauss_short{k}", rng.standard_normal(1500), rng.standard_normal(n_on))
    for k, n_on in enumerate((900, 950, 999)):
        add(f"gauss_long{k}", rng.standard_normal(2000), rng.standard_normal(n_on))

    # constant history / near-zero variance history
    add("const_hist", np.full(1500, 3.25), rng.standard_normal(120))
    add("const_hist2", np.full(800, -1.0), rng.standard_normal(40) * 5)
    add("tinyvar_hist", 2.0 + rng.standard_normal(1500) * 1e-13, rng.standard_normal(120))
    add("tinyvar_hist2", 2.0 + rng.standard_normal(900) * 1e-9, rng.standard_normal(80) * 1e-9)

    # heavy tailed (t3) history
    for k in range(3):
        add(f"t3_{k}", rng.standard_t(3, 1200 + 200 * k), rng.standard_t(3, 200))

    # strong AR(1) history
    for k, phi in enumerate((0.6, 0.85, -0.7, 0.95)):
        h = rng.standard_normal(2000)
        for i in range(1, len(h)):
            h[i] += phi * h[i - 1]
        o = rng.standard_normal(300)
        for i in range(1, len(o)):
            o[i] += phi * o[i - 1]
        add(f"ar1_{k}", h, o)

    # AR(5)-ish history (exercises the p=5 causal window)
    for k in range(2):
        h = rng.standard_normal(2500)
        for i in range(5, len(h)):
            h[i] += 0.3 * h[i - 1] + 0.2 * h[i - 3] - 0.15 * h[i - 5]
        add(f"ar5_{k}", h, rng.standard_normal(400))

    # volatility clustered (GARCH-like) history and online
    for k in range(3):
        h = np.empty(2000)
        v = 1.0
        for i in range(2000):
            v = 0.02 + 0.12 * (h[i - 1] ** 2 if i else 1.0) + 0.86 * v
            h[i] = np.sqrt(v) * rng.standard_normal()
        o = np.empty(400)
        for i in range(400):
            v = 0.02 + 0.12 * (o[i - 1] ** 2 if i else 1.0) + 0.86 * v
            o[i] = np.sqrt(v) * rng.standard_normal()
        add(f"garch_{k}", h, o)

    # break at tau = 0 (mean, variance, and both)
    add("break_tau0_mean", rng.standard_normal(1500), rng.standard_normal(250) + 4.0)
    add("break_tau0_var", rng.standard_normal(1500), rng.standard_normal(250) * 7.0)
    add("break_tau0_both", rng.standard_normal(1500), rng.standard_normal(250) * 0.05 - 3.0)

    # break at the very last point
    o = rng.standard_normal(200)
    o[-1] += 25.0
    add("break_last", rng.standard_normal(1500), o)
    o = rng.standard_normal(200)
    o[-1] *= 60.0
    add("break_last_var", rng.standard_normal(1500), o)

    # mid-series breaks
    for k, tau in enumerate((1, 31, 32, 199)):
        o = rng.standard_normal(200)
        o[tau:] += 3.0
        add(f"break_tau{tau}_{k}", rng.standard_normal(1500), o)

    # huge single outliers (history and online)
    h = rng.standard_normal(1500)
    h[700] = 1e7
    add("outlier_hist", h, rng.standard_normal(150))
    o = rng.standard_normal(150)
    o[70] = -1e8
    add("outlier_online", rng.standard_normal(1500), o)
    o = rng.standard_normal(150)
    o[0] = 1e9
    add("outlier_first", rng.standard_normal(1500), o)

    # discrete / degenerate online, drift, and short histories
    add("discrete", rng.integers(-2, 3, 1500).astype(float), rng.integers(-2, 3, 90).astype(float))
    add("const_online", rng.standard_normal(1500), np.full(90, 0.5))
    add("drift", rng.standard_normal(1500), np.linspace(0, 10, 200) + rng.standard_normal(200))
    add("hist_60", rng.standard_normal(60), rng.standard_normal(40))
    add("hist_25", rng.standard_normal(25), rng.standard_normal(30))
    add("hist_12", rng.standard_normal(12), rng.standard_normal(20))
    add("hist_400", rng.standard_normal(400), rng.standard_normal(120))
    add("scale_shift", rng.standard_normal(1500) * 100 + 50, rng.standard_normal(200) * 100 + 50)
    add("lognormal", np.exp(rng.standard_normal(1500)), np.exp(rng.standard_normal(200)))
    return out


CASES = _synthetic(20260819)


def test_case_count():
    assert len(CASES) >= 40, len(CASES)


@pytest.mark.parametrize("idx", range(len(CASES)))
def test_parity_synthetic_bitwise(idx):
    name, h, o = CASES[idx]
    nbad, worst = _mismatches(h, o)
    assert nbad == 0, f"{name}: {nbad} non-bitwise cells, first={worst}"


# ------------------------------------------------------------------ real store
def _real_indices(k=30, seed=4):
    from sbr.store import load_store

    st = load_store()
    rng = np.random.default_rng(seed)
    return st, [int(i) for i in rng.choice(st.n_series, size=k, replace=False)]


def test_parity_real_store_bitwise():
    st, idx = _real_indices(30)
    assert len(idx) >= 30
    for i in idx:
        h, o, _ = st.series(i)
        nbad, worst = _mismatches(h, o)
        assert nbad == 0, f"series {i}: {nbad} non-bitwise cells, first={worst}"


# ------------------------------------------------------------------- structure
def test_ncols_matches_batch():
    h = np.random.default_rng(0).standard_normal(1500)
    o = np.random.default_rng(1).standard_normal(50)
    names, A = REGISTRY[MODULE].fn(make_ctx(h, o))
    ctx = StreamCtx().fit_historical(h)
    eng = StreamM04Resid()
    eng.fit_historical(ctx)
    assert len(eng.cols) == A.shape[1] == len(names) == 60
    assert eng.cols == names
    assert eng.MODULE == MODULE


def test_nonfinite_is_nan():
    """Every emitted value is either NaN or a finite float in [-CLIP, CLIP]."""
    from sbr.features.m04_resid import CLIP

    rng = np.random.default_rng(9)
    h = rng.standard_normal(1200)
    o = np.concatenate([rng.standard_normal(80), np.full(20, 1e12)])
    _, B = _stream_rows(h, o)
    fin = np.isfinite(B)
    assert np.all(np.isnan(B[~fin]))
    assert np.all(np.abs(B[fin]) <= CLIP + 1e-6)


# ------------------------------------------------------------ prefix / poison
def test_prefix_invariance_and_poison():
    """Rows already emitted cannot be changed by anything that arrives later."""
    rng = np.random.default_rng(77)
    h = rng.standard_normal(1500)
    o = rng.standard_normal(200)

    ctx = StreamCtx().fit_historical(h)
    eng = StreamM04Resid()
    eng.fit_historical(ctx)
    ref = []
    for x in o:
        ctx.push(x)
        ref.append(eng.step(ctx).copy())
    ref = np.array(ref)

    for cut in (1, 7, 31, 32, 33, 120):
        # poison every observation after `cut` with wild values and re-run
        poisoned = o.copy()
        poisoned[cut:] = rng.standard_normal(len(o) - cut) * 1e6 + 1e5
        ctx2 = StreamCtx().fit_historical(h)
        eng2 = StreamM04Resid()
        eng2.fit_historical(ctx2)
        got = []
        for x in poisoned:
            ctx2.push(x)
            got.append(eng2.step(ctx2).copy())
        got = np.array(got)
        a, b = ref[:cut], got[:cut]
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        assert same.all(), f"poison after t={cut} changed an earlier row"

    # and the batch module agrees on truncated online segments too
    for cut in (10, 37, 200):
        _, part = REGISTRY[MODULE].fn(make_ctx(h, o[:cut]))
        a = ref[:cut].astype(np.float32)
        a[~np.isfinite(a)] = np.nan
        same = (a == part) | (np.isnan(a) & np.isnan(part))
        assert same.all(), f"stream vs batch on prefix {cut}"


# ----------------------------------------------------------------------- speed
def test_timing_microseconds_per_observation(capsys):
    from sbr.store import load_store

    st = load_store()
    idx = [int(i) for i in np.random.default_rng(3).choice(st.n_series, 6, replace=False)]

    # warm-up (imports, BLAS handles, code paths)
    h, o, _ = st.series(idx[0])
    c = StreamCtx().fit_historical(h)
    e = StreamM04Resid()
    e.fit_historical(c)
    for x in o[:25]:
        c.push(x)
        e.step(c)

    best_step = np.inf
    best_both = np.inf
    fit_s = 0.0
    for _ in range(3):
        t_both = t_push = 0.0
        n = 0
        for i in idx:
            h, o, _ = st.series(i)
            c = StreamCtx().fit_historical(h)
            e = StreamM04Resid()
            e.fit_historical(c)
            t0 = time.process_time()
            for x in o:
                c.push(x)
                e.step(c)
            t_both += time.process_time() - t0

            c = StreamCtx().fit_historical(h)
            t0 = time.process_time()
            for x in o:
                c.push(x)
            t_push += time.process_time() - t0
            n += len(o)
        best_both = min(best_both, 1e6 * t_both / n)
        best_step = min(best_step, 1e6 * (t_both - t_push) / n)

    ctxs = [StreamCtx().fit_historical(st.series(i)[0]) for i in idx]
    t0 = time.process_time()
    for c in ctxs:
        e = StreamM04Resid()
        e.fit_historical(c)
    fit_s = (time.process_time() - t0) / len(idx)

    with capsys.disabled():
        print(f"\n[m04_resid] step()          : {best_step:8.1f} us/observation (CPU)")
        print(f"[m04_resid] push()+step()   : {best_both:8.1f} us/observation (CPU)")
        print(f"[m04_resid] fit_historical(): {1e3 * fit_s:8.1f} ms/series (once)")
        print(f"[m04_resid] columns         : {len(e.cols)}")

    # generous ceiling: this is a 60-column module doing 5 AR filters per point
    assert best_step < 1500.0
