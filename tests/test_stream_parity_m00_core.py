"""Bitwise parity: StreamM00Core (incremental) vs the batch ``m00_core`` module.

Style follows ``tests/test_stream_ctx_parity.py``: build both sides on the same
(hist, online) pair, walk the online segment one point at a time, and compare
row t of the batch matrix against ``step()`` with ``atol=0`` and NaN == NaN.

The batch module returns float32 (``np.column_stack(out).astype(np.float32)``
followed by ``A[~np.isfinite(A)] = np.nan``).  Per the streaming contract
``step()`` returns float64 and the *caller* does the final cast, so the
comparison casts the streamed row to float32 first.  Nothing else is relaxed:
no tolerance, no rtol.
"""
import time

import numpy as np
import pytest

from sbr.features.base import REGISTRY, load_all, make_ctx
from sbr.stream.ctx import StreamCtx
from sbr.stream.s_m00_core import StreamM00Core

MODULE = "m00_core"

load_all()


# --------------------------------------------------------------------------
# synthetic series covering the regimes the contract asks for
# --------------------------------------------------------------------------
def _garch(rng, n, omega=0.05, alpha=0.12, beta=0.85):
    x = np.empty(n)
    v = omega / max(1e-9, 1.0 - alpha - beta)
    for i in range(n):
        e = rng.standard_normal()
        x[i] = np.sqrt(v) * e
        v = omega + alpha * x[i] ** 2 + beta * v
    return x


def _ar1(rng, n, phi=0.85):
    x = rng.standard_normal(n)
    for i in range(1, n):
        x[i] += phi * x[i - 1]
    return x


def _cases():
    """(label, hist, online) pairs -- >= 40 synthetic series."""
    rng = np.random.default_rng(20260819)
    out = []

    # 1. short online segments (10-20 points)
    for k in range(4):
        h = rng.standard_normal(int(rng.integers(1000, 3000)))
        o = rng.standard_normal(int(rng.integers(10, 21)))
        out.append((f"short_online_{k}", h, o))

    # 2. long online segments (900-999 points)
    for k in range(4):
        h = rng.standard_normal(int(rng.integers(1000, 4000)))
        o = rng.standard_normal(int(rng.integers(900, 1000)))
        out.append((f"long_online_{k}", h, o))

    # 3. constant history (sd/mad/iqr all collapse to their 1e-9 floors)
    for k in range(3):
        c = float(rng.standard_normal())
        h = np.full(int(rng.integers(1000, 2000)), c)
        o = c + rng.standard_normal(int(rng.integers(20, 200))) * (10.0 ** -k)
        out.append((f"const_hist_{k}", h, o))

    # 4. near-zero-variance history
    for k in range(3):
        n = int(rng.integers(1000, 2000))
        h = 3.0 + rng.standard_normal(n) * (10.0 ** -(9 + k))
        o = 3.0 + rng.standard_normal(int(rng.integers(20, 300))) * 1e-9
        out.append((f"tiny_var_hist_{k}", h, o))

    # 5. heavy-tailed (t3) history
    for k in range(4):
        h = rng.standard_t(3, int(rng.integers(1000, 4000)))
        o = rng.standard_t(3, int(rng.integers(50, 600)))
        out.append((f"t3_hist_{k}", h, o))

    # 6. strong AR(1) history
    for k in range(4):
        h = _ar1(rng, int(rng.integers(1000, 4000)), phi=0.7 + 0.07 * k)
        o = _ar1(rng, int(rng.integers(50, 600)), phi=0.7 + 0.07 * k)
        out.append((f"ar1_hist_{k}", h, o))

    # 7. volatility clustered
    for k in range(4):
        h = _garch(rng, int(rng.integers(1000, 3000)))
        o = _garch(rng, int(rng.integers(50, 500)))
        out.append((f"garch_{k}", h, o))

    # 8. break at tau = 0 (whole online segment is post-break)
    for k in range(4):
        h = rng.standard_normal(int(rng.integers(1000, 3000)))
        n = int(rng.integers(30, 500))
        o = rng.standard_normal(n) * (1.0 + 2.0 * k) + (2.0 + k)
        out.append((f"break_tau0_{k}", h, o))

    # 9. break at the very last online point
    for k in range(4):
        h = rng.standard_normal(int(rng.integers(1000, 3000)))
        n = int(rng.integers(30, 500))
        o = rng.standard_normal(n)
        o[-1] = 6.0 + 3.0 * k
        out.append((f"break_last_{k}", h, o))

    # 10. huge single outliers (float32 overflow territory)
    for k in range(4):
        h = rng.standard_normal(int(rng.integers(1000, 3000)))
        n = int(rng.integers(30, 400))
        o = rng.standard_normal(n)
        o[n // 3] = 10.0 ** (6 + 10 * k)
        out.append((f"huge_outlier_{k}", h, o))

    # 11. short history -> the null grid is capped, so long windows and the
    #     cross-scale block legitimately disappear from the column set
    for k, nh in enumerate((200, 300, 60)):
        h = rng.standard_normal(nh)
        o = rng.standard_normal(int(rng.integers(30, 200)))
        out.append((f"short_hist_{nh}_{k}", h, o))

    # 12. plain mixed / scale-shifted
    for k in range(6):
        h = rng.standard_normal(int(rng.integers(1000, 5000))) * (0.1 + k)
        o = rng.standard_normal(int(rng.integers(10, 700))) * (0.1 + k)
        out.append((f"mix_{k}", h, o))

    # 13. constant online segment on top of ordinary history
    for k in range(2):
        h = rng.standard_normal(int(rng.integers(1000, 3000)))
        o = np.full(int(rng.integers(20, 150)), float(rng.standard_normal()))
        out.append((f"const_online_{k}", h, o))

    return out


SYNTH = _cases()


# --------------------------------------------------------------------------
# comparison helper
# --------------------------------------------------------------------------
def _compare(h, o):
    """Return (n_mismatch, first_mismatch_description, n_cols)."""
    names, A = REGISTRY[MODULE].fn(make_ctx(h, o))
    ctx = StreamCtx().fit_historical(h)
    eng = StreamM00Core()
    eng.fit_historical(ctx)
    assert eng.cols == list(names), (
        f"column layout differs: stream {len(eng.cols)} vs batch {len(names)}")

    bad = 0
    first = None
    for t in range(len(o)):
        ctx.push(o[t])
        r = eng.step(ctx)
        assert r.dtype == np.float64 and r.shape == (len(names),)
        r32 = r.astype(np.float32)
        a = A[t]
        m = ~((r32 == a) | (np.isnan(r32) & np.isnan(a)))
        if m.any():
            bad += int(m.sum())
            if first is None:
                j = int(np.argmax(m))
                first = (t, names[j], float(r32[j]), float(a[j]))
    return bad, first, len(names)


@pytest.mark.parametrize("label,h,o", SYNTH, ids=[c[0] for c in SYNTH])
def test_parity_synthetic_bitwise(label, h, o):
    bad, first, k = _compare(h, o)
    assert bad == 0, f"{label}: {bad} mismatching cells, first={first}"
    assert k > 0


def test_synthetic_coverage():
    """The contract asks for >= 40 synthetic series spanning the listed regimes."""
    assert len(SYNTH) >= 40, len(SYNTH)
    labels = " ".join(c[0] for c in SYNTH)
    for regime in ("short_online", "long_online", "const_hist", "tiny_var_hist",
                   "t3_hist", "ar1_hist", "garch", "break_tau0", "break_last",
                   "huge_outlier"):
        assert regime in labels, regime


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------
def _real_ids(n=30, seed=11):
    from sbr.store import load_store
    st = load_store()
    rng = np.random.default_rng(seed)
    return st, [int(i) for i in rng.choice(st.n_series, size=n, replace=False)]


def test_parity_real_store_bitwise():
    st, ids = _real_ids(30)
    assert len(ids) >= 30
    for i in ids:
        h, o, _tau = st.series(i)
        bad, first, _k = _compare(h, o)
        assert bad == 0, f"series {i}: {bad} mismatching cells, first={first}"


# --------------------------------------------------------------------------
# causality: no lookahead, and later observations cannot rewrite earlier rows
# --------------------------------------------------------------------------
def test_prefix_poison_invariance():
    rng = np.random.default_rng(4242)
    h = rng.standard_normal(2500)
    o = rng.standard_normal(300)

    ctx = StreamCtx().fit_historical(h)
    eng = StreamM00Core()
    eng.fit_historical(ctx)
    rows = []
    for t in range(len(o)):
        ctx.push(o[t])
        rows.append(eng.step(ctx).copy())

    for cut in (1, 7, 40, 137, 299):
        poisoned = o.copy()
        poisoned[cut + 1:] = 1e9 * rng.standard_normal(len(o) - cut - 1)
        c2 = StreamCtx().fit_historical(h)
        e2 = StreamM00Core()
        e2.fit_historical(c2)
        for t in range(cut + 1):
            c2.push(poisoned[t])
            r = e2.step(c2)
        ref = rows[cut]
        assert np.array_equal(r, ref, equal_nan=True), f"row {cut} changed under poisoning"

    # and a truncated run must reproduce the surviving rows exactly
    for cut in (3, 10, 37):
        c2 = StreamCtx().fit_historical(h)
        e2 = StreamM00Core()
        e2.fit_historical(c2)
        for t in range(cut):
            c2.push(o[t])
            r = e2.step(c2)
        assert np.array_equal(r, rows[cut - 1], equal_nan=True), f"prefix {cut} differs"


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------
def test_timing_microseconds_per_observation(capsys):
    rng = np.random.default_rng(99)
    h = rng.standard_normal(3000)
    o = rng.standard_normal(1000)

    # warm-up (import/JIT-free, but numpy caches and branch prediction matter)
    ctx = StreamCtx().fit_historical(h)
    eng = StreamM00Core()
    eng.fit_historical(ctx)
    for x in o[:100]:
        ctx.push(x)
        eng.step(ctx)

    n_rep, tot_step, tot_all = 3, 0.0, 0.0
    for _ in range(n_rep):
        ctx = StreamCtx().fit_historical(h)
        eng = StreamM00Core()
        eng.fit_historical(ctx)
        t0 = time.perf_counter()
        for x in o:
            ctx.push(x)
        t1 = time.perf_counter()
        push_only = t1 - t0

        ctx = StreamCtx().fit_historical(h)
        eng = StreamM00Core()
        eng.fit_historical(ctx)
        t0 = time.perf_counter()
        for x in o:
            ctx.push(x)
            eng.step(ctx)
        t1 = time.perf_counter()
        tot_all += t1 - t0
        tot_step += (t1 - t0) - push_only

    n = n_rep * len(o)
    with capsys.disabled():
        print(f"\n[m00_core] step()          : {tot_step / n * 1e6:9.1f} us/obs")
        print(f"[m00_core] ctx.push+step() : {tot_all / n * 1e6:9.1f} us/obs")
    assert tot_all / n < 0.05, "sanity ceiling: 50 ms/obs"
