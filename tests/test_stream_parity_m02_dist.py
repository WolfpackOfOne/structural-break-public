"""Bitwise parity: StreamM02Dist (incremental) vs the batch module ``m02_dist``.

Target is `atol=0`, NaN == NaN, on the float32 array the batch module actually
returns (the batch build ends with ``.astype(np.float32)`` followed by
``A[~np.isfinite(A)] = np.nan``; the streaming ``step`` reproduces both).
"""
import time

import numpy as np
import pytest

from sbr.features.base import REGISTRY, load_all, make_ctx
from sbr.features.m02_dist import _sg, _up
from sbr.stream.ctx import StreamCtx
from sbr.stream.s_m02_dist import COLS, StreamM02Dist, _sg1, _up1

MODULE = "m02_dist"

load_all()


# --------------------------------------------------------------------- runner
def _batch(h, o):
    return REGISTRY[MODULE].fn(make_ctx(h, o))


def _stream(h, o):
    ctx = StreamCtx().fit_historical(h)
    eng = StreamM02Dist()
    eng.fit_historical(ctx)
    out = np.empty((len(o), len(COLS)), dtype=np.float64)
    for t, x in enumerate(o):
        ctx.push(x)
        out[t] = eng.step(ctx)
    return out


def _diff_report(names, A, S):
    """Return '' if bitwise identical, else a detailed message."""
    same = (A == S) | (np.isnan(A) & np.isnan(S))
    if same.all():
        return ""
    bad = ~same
    cols = [names[j] for j in np.nonzero(bad.any(axis=0))[0]]
    j = int(np.nonzero(bad.any(axis=0))[0][0])
    r = int(np.nonzero(bad[:, j])[0][0])
    a = np.asarray(A, dtype=np.float64)[bad]
    b = np.asarray(S, dtype=np.float64)[bad]
    with np.errstate(invalid="ignore", divide="ignore"):
        mad = np.nanmax(np.abs(a - b)) if a.size else 0.0
        mrd = np.nanmax(np.abs(a - b) / np.maximum(np.abs(a), 1e-300)) if a.size else 0.0
    return (f"{int(bad.sum())} cells differ in {len(cols)} cols {cols[:8]}; "
            f"first at row {r} col {names[j]}: batch={A[r, j]!r} stream={S[r, j]!r}; "
            f"max|abs dev|={mad} max|rel dev|={mrd}")


def _assert_bitwise(h, o, tag):
    names, A = _batch(h, o)
    assert names == COLS, f"{tag}: column names/order differ"
    assert A.shape[1] == len(COLS)
    S = _stream(h, o).astype(np.float32)
    msg = _diff_report(names, A, S)
    assert not msg, f"{tag}: {msg}"


# ------------------------------------------------------------------ scenarios
def _vol_cluster(rng, n):
    x = np.empty(n)
    s = 1.0
    for i in range(n):
        s = np.sqrt(0.02 + 0.1 * (x[i - 1] ** 2 if i else 1.0) + 0.87 * s * s)
        x[i] = s * rng.standard_normal()
    return x


def _ar1(rng, n, phi):
    x = rng.standard_normal(n)
    for i in range(1, n):
        x[i] += phi * x[i - 1]
    return x


def _cases():
    """>= 40 synthetic (hist, online) pairs covering the required regimes."""
    rng = np.random.default_rng(20260819)
    C = []

    def add(tag, h, o):
        C.append((tag, np.asarray(h, float), np.asarray(o, float)))

    # --- plain gaussian, assorted lengths ------------------------------
    for _k, no in enumerate((10, 13, 17, 20, 31, 64, 127, 200, 512, 900, 950, 999)):
        add(f"gauss_n{no}", rng.standard_normal(2000), rng.standard_normal(no))

    # --- constant / near-zero-variance history -------------------------
    add("const_hist", np.ones(1500), rng.standard_normal(200))
    add("const_hist_long", np.full(1500, -3.25), rng.standard_normal(950))
    add("nearzero_hist", 1.0 + 1e-13 * rng.standard_normal(1500), rng.standard_normal(300))
    add("nearzero_hist_short", 1.0 + 1e-15 * rng.standard_normal(1200), rng.standard_normal(12))
    add("const_online", rng.standard_normal(1500), np.full(300, 0.5))
    add("two_atom_hist", np.repeat([0.0, 1.0], 750), rng.standard_normal(300))

    # --- heavy tails ----------------------------------------------------
    add("t3_hist", rng.standard_t(3, 2500), rng.standard_t(3, 400))
    add("t3_hist_long", rng.standard_t(3, 3000), rng.standard_t(3, 999))
    add("t3_hist_short", rng.standard_t(3, 1200), rng.standard_t(3, 15))
    add("cauchy_hist", rng.standard_cauchy(2000), rng.standard_cauchy(300))

    # --- strong AR(1) ---------------------------------------------------
    add("ar1_p90", _ar1(rng, 3000, 0.9), _ar1(rng, 400, 0.9))
    add("ar1_m85", _ar1(rng, 3000, -0.85), _ar1(rng, 950, -0.85))
    add("ar1_p99", _ar1(rng, 2500, 0.99), _ar1(rng, 300, 0.99))
    add("ar1_short_online", _ar1(rng, 2000, 0.7), _ar1(rng, 11, 0.7))

    # --- volatility clustering -----------------------------------------
    add("garch", _vol_cluster(rng, 2500), _vol_cluster(rng, 400))
    add("garch_long", _vol_cluster(rng, 2000), _vol_cluster(rng, 999))
    add("garch_hist_gauss_online", _vol_cluster(rng, 2000), rng.standard_normal(300))

    # --- breaks at tau = 0 ----------------------------------------------
    add("break0_scale", rng.standard_normal(2000), rng.standard_normal(400) * 6.0)
    add("break0_mean", rng.standard_normal(2000), rng.standard_normal(400) + 5.0)
    add("break0_skew", rng.standard_normal(2000), rng.gamma(1.5, 1.0, 400))
    add("break0_short", rng.standard_normal(2000), rng.standard_normal(14) * 4.0)
    add("break0_long", rng.standard_normal(2000), rng.standard_normal(999) * 0.15)

    # --- break at the last point ----------------------------------------
    o = rng.standard_normal(300)
    o[-1] += 40.0
    add("break_last_point", rng.standard_normal(2000), o)
    o = rng.standard_normal(999)
    o[-1] = -1e6
    add("break_last_point_long", rng.standard_normal(2000), o)
    o = rng.standard_normal(10)
    o[-1] = 25.0
    add("break_last_point_tiny", rng.standard_normal(2000), o)

    # --- breaks in the middle -------------------------------------------
    o = rng.standard_normal(600)
    o[300:] *= 5.0
    add("break_mid_scale", rng.standard_normal(2000), o)
    o = rng.standard_normal(600)
    o[128:] += 3.0
    add("break_at_w128", rng.standard_normal(2000), o)
    o = rng.standard_normal(400)
    o[32:] = np.clip(o[32:], -0.4, 0.4)
    add("break_censoring", rng.standard_normal(2000), o)

    # --- huge single outliers -------------------------------------------
    o = rng.standard_normal(400)
    o[150] = 1e9
    add("outlier_1e9", rng.standard_normal(2000), o)
    o = rng.standard_normal(400)
    o[3] = -1e18
    add("outlier_early", rng.standard_normal(2000), o)
    add("outlier_all", rng.standard_normal(2000), np.full(300, 1e300))
    h = rng.standard_normal(2000)
    h[1000] = 1e12
    add("outlier_in_hist", h, rng.standard_normal(300))

    # --- histories too short for some/all nulls --------------------------
    add("hist55", rng.standard_normal(55), rng.standard_normal(200))
    add("hist70", rng.standard_normal(70), rng.standard_normal(300))
    add("hist120", rng.standard_normal(120), rng.standard_normal(300))
    add("hist200", rng.standard_normal(200), rng.standard_normal(400))
    add("hist300", rng.standard_normal(300), rng.standard_normal(500))
    add("hist_exact_296", rng.standard_normal(296), rng.standard_normal(300))

    # --- misc -------------------------------------------------------------
    add("bimodal_hist", np.concatenate([rng.standard_normal(1000) - 4,
                                        rng.standard_normal(1000) + 4]),
        rng.standard_normal(400))
    add("uniform_hist", rng.random(2000), rng.random(400))
    add("discrete_hist", rng.integers(0, 5, 2000).astype(float),
        rng.integers(0, 5, 300).astype(float))
    add("online_n1", rng.standard_normal(2000), rng.standard_normal(1))
    return C


CASES = _cases()
assert len(CASES) >= 40, len(CASES)


# --------------------------------------------------------------------- tests
def test_column_names_and_count_match_batch():
    rng = np.random.default_rng(0)
    names, A = _batch(rng.standard_normal(1500), rng.standard_normal(200))
    assert names == COLS
    assert A.shape[1] == len(COLS) == 59
    assert StreamM02Dist().cols == names


@pytest.mark.parametrize("idx", range(len(CASES)), ids=[c[0] for c in CASES])
def test_parity_synthetic_bitwise(idx):
    tag, h, o = CASES[idx]
    _assert_bitwise(h, o, tag)


def test_parity_real_store_bitwise():
    """>= 30 random real series from the competition store."""
    from sbr.store import load_store

    st = load_store()
    rng = np.random.default_rng(2026)
    idx = rng.choice(st.n_series, size=30, replace=False)
    for i in idx:
        h, o, _ = st.series(int(i))
        _assert_bitwise(h, o, f"store[{int(i)}]")


def test_prefix_poison_invariance():
    """A row emitted at t can never be changed by anything after t."""
    rng = np.random.default_rng(5)
    h = rng.standard_normal(2000)
    o = rng.standard_normal(400)

    ctx = StreamCtx().fit_historical(h)
    eng = StreamM02Dist()
    eng.fit_historical(ctx)
    rows = []
    for x in o:
        ctx.push(x)
        rows.append(eng.step(ctx).copy())
    rows = np.asarray(rows)

    for cut in (1, 5, 32, 128, 129, 333):
        poisoned = o.copy()
        poisoned[cut:] = 1e9 * rng.standard_normal(len(o) - cut)
        p = _stream(h, poisoned)[:cut]
        a, b = rows[:cut], p
        assert ((a == b) | (np.isnan(a) & np.isnan(b))).all(), f"cut={cut}"

    # ... and the batch module agrees on the truncated prefix, too.
    for cut in (17, 128, 200):
        _, part = _batch(h, o[:cut])
        a = rows[:cut].astype(np.float32)
        assert ((a == part) | (np.isnan(a) & np.isnan(part))).all(), f"batch cut={cut}"


def test_null_lookup_kernels_bitwise():
    """The scalar `_up1`/`_sg1` fast paths equal the batch `_up`/`_sg` exactly."""
    rng = np.random.default_rng(11)
    for m in (40, 97, 500):
        rs = np.sort(rng.standard_normal(m))
        v = np.concatenate([rng.standard_normal(20000), rs,
                            np.array([-np.inf, np.inf, rs[0], rs[-1], 0.0])])
        ru, rg = _up(rs, v), _sg(rs, v)
        for i in range(len(v)):
            assert _up1(rs, v[i]) == ru[i], (m, i, v[i])
            assert _sg1(rs, v[i]) == rg[i], (m, i, v[i])
    # ties (duplicated nulls) exercise the left/right searchsorted split
    rs = np.sort(np.round(rng.standard_normal(200), 1))
    v = np.round(rng.standard_normal(20000), 1)
    ru, rg = _up(rs, v), _sg(rs, v)
    for i in range(len(v)):
        assert _up1(rs, v[i]) == ru[i]
        assert _sg1(rs, v[i]) == rg[i]


def test_bisect_matches_searchsorted():
    """The bisect fast path returns numpy's indices for every non-NaN key."""
    from bisect import bisect_left, bisect_right

    rng = np.random.default_rng(31)
    for rs in (np.sort(rng.standard_normal(40)),
               np.sort(rng.standard_normal(500)),
               np.sort(np.round(rng.standard_normal(300), 1)),   # heavy ties
               np.sort(np.repeat(rng.standard_normal(5), 40))):  # few distinct
        lst = rs.tolist()
        v = np.concatenate([rng.standard_normal(30000), rs,
                            np.round(rng.standard_normal(5000), 1),
                            np.array([np.inf, -np.inf, 0.0, -0.0])])
        lo = np.searchsorted(rs, v, side="left")
        hi = np.searchsorted(rs, v, side="right")
        for i in range(len(v)):
            x = float(v[i])
            assert bisect_left(lst, x) == lo[i], (i, x)
            assert bisect_right(lst, x) == hi[i], (i, x)


def test_null_tables_match_kernels():
    """`_NT.up_of`/`sg_of` (tabulated) == `_up1`/`_sg1` == batch `_up`/`_sg`."""
    from sbr.stream.s_m02_dist import _NT

    rng = np.random.default_rng(41)
    for m in (40, 97, 500):
        rs = np.sort(rng.standard_normal(m))
        nt = _NT(rs, {})
        v = np.concatenate([rng.standard_normal(20000), rs,
                            np.array([np.inf, -np.inf, np.nan])])
        ru, rg = _up(rs, v), _sg(rs, v)
        for i in range(len(v)):
            x = float(v[i])
            assert nt.up_of(x) == _up1(rs, x)
            assert nt.sg_of(x) == _sg1(rs, x)
            if not np.isnan(v[i]):
                assert nt.up_of(x) == ru[i]
                assert nt.sg_of(x) == rg[i]
    # a null containing NaN must fall back to the numpy path
    rs = np.sort(np.array([1.0, 2.0, np.nan, 3.0]))
    nt = _NT(rs, {})
    assert nt.lst is None
    assert nt.up_of(2.5) == _up1(rs, 2.5)


def test_quant_block_bitwise():
    """The scalar quantile block == `_stats(..., quant=True)`, bit for bit."""
    from sbr.features.m02_dist import _stats
    from sbr.stream.s_m02_dist import StreamM02Dist as S

    rng = np.random.default_rng(51)
    rows, wls = [], []
    for _ in range(4000):
        B, w = 20, float(rng.choice([1, 2, 7, 32, 128, 257, 999]))
        style = rng.integers(0, 5)
        if style == 0:                       # all mass in one bin
            c = np.zeros(B)
            c[rng.integers(B)] = w
        elif style == 1:                     # empty (the degenerate dummy row)
            c = np.zeros(B)
        elif style == 2:                     # two-point support
            c = np.zeros(B)
            c[[0, B - 1]] = [w // 2, w - w // 2]
        elif style == 3:                     # multinomial
            c = rng.multinomial(int(w), np.full(B, 1.0 / B)).astype(float)
        else:                                # skewed multinomial
            p = rng.dirichlet(np.full(B, 0.3))
            c = rng.multinomial(int(w), p).astype(float)
        rows.append(c)
        wls.append(w)
    C = np.asarray(rows)
    wl = np.asarray(wls, dtype=np.float64)
    mu = np.full(len(C), 0.5)
    ref = _stats(C, wl, 20, mu, mu, True)
    for i in range(len(C)):
        got = S._quant(C[i], wl[i])
        exp = (ref["q50"][i], ref["qiqr"][i], ref["q9010"][i])
        for a, b in zip(got, exp):
            assert a == b or (np.isnan(a) and np.isnan(b)), (i, wl[i], a, b)


def test_pit_table_bitwise():
    """The tabulated PIT == `_pit_against`, bit for bit."""
    from sbr.features.m02_dist import _pit_against
    from sbr.stream.s_m02_dist import _pit1, _pit_table

    rng = np.random.default_rng(61)
    for ref in (np.sort(rng.standard_normal(500)),
                np.sort(np.round(rng.standard_normal(2000), 2)),
                np.sort(np.abs(rng.standard_normal(3000)))):
        pt = _pit_table(ref)
        v = np.concatenate([rng.standard_normal(20000), ref,
                            np.array([np.inf, -np.inf, 0.0])])
        exp = _pit_against(ref, v)
        for i in range(len(v)):
            assert _pit1(pt, float(v[i])) == exp[i], (i, v[i])


def test_math_sqrt_matches_numpy():
    """sqrt is correctly rounded per IEEE-754, so math.sqrt == np.sqrt."""
    import math

    rng = np.random.default_rng(71)
    v = np.abs(rng.standard_normal(200000)) * rng.choice([1e-8, 1.0, 1e8], 200000)
    ref = np.sqrt(v)
    assert all(math.sqrt(float(v[i])) == ref[i] for i in range(len(v)))


def test_nonfinite_becomes_nan():
    """`step` reproduces the batch `A[~np.isfinite(A)] = np.nan` tail step."""
    rng = np.random.default_rng(9)
    h = rng.standard_normal(1500)
    o = np.full(300, 1e300)
    S = _stream(h, o)
    assert np.isfinite(S[~np.isnan(S)]).all()
    _assert_bitwise(h, o, "nonfinite")


def test_timing_microseconds_per_observation(capsys):
    rng = np.random.default_rng(3)
    h = rng.standard_normal(2000)
    o = rng.standard_normal(1000)

    ctx = StreamCtx().fit_historical(h)
    eng = StreamM02Dist()
    t0 = time.perf_counter()
    eng.fit_historical(ctx)
    fit_ms = (time.perf_counter() - t0) * 1e3

    push_us = step_us = 0.0
    for x in o:
        a = time.perf_counter()
        ctx.push(x)
        b = time.perf_counter()
        eng.step(ctx)
        push_us += b - a
        step_us += time.perf_counter() - b
    n = len(o)
    with capsys.disabled():
        print(f"\n[m02_dist] fit_historical: {fit_ms:.1f} ms/series | "
              f"StreamCtx.push: {push_us / n * 1e6:.1f} us/obs | "
              f"StreamM02Dist.step: {step_us / n * 1e6:.1f} us/obs | "
              f"total {(push_us + step_us) / n * 1e6:.1f} us/obs")
    assert step_us / n * 1e6 < 20000.0
