"""Bitwise parity: StreamM03Dyn (incremental) vs the batch ``m03_dyn`` module.

The batch module is the specification.  Every test here compares the streaming
row produced *before the next observation exists* against row ``t`` of
``REGISTRY["m03_dyn"].fn(make_ctx(hist, online))``, with ``atol=0`` and
``NaN == NaN``.  Nothing in this file is allowed to use a tolerance.

The batch module casts to float32 as its last act, so the comparison is done on
the float32 view of the streaming row (which is what the caller stores); the
float64 row itself is compared too, wherever the batch float64 is recoverable
(see ``_batch64``), so that float32 rounding cannot hide a real deviation.
"""
import time

import numpy as np
import pytest
from conftest import skip_on_missing_store

import sbr.stream.s_m03_dyn as S
from sbr.features import m03_dyn as B
from sbr.features.base import REGISTRY, load_all, make_ctx
from sbr.stream.ctx import StreamCtx
from sbr.stream.s_m03_dyn import StreamM03Dyn

MODULE = "m03_dyn"
load_all()


# --------------------------------------------------------------------- helpers
def _batch(h, o):
    return REGISTRY[MODULE].fn(make_ctx(h, o))


def _batch64(h, o):
    """Batch names + the float64 matrix *before* the module's float32 cast.

    ``build`` ends with ``np.column_stack(out).astype(np.float32)``; capturing
    the ``column_stack`` result gives the pre-cast float64 block, which lets us
    check parity at full precision instead of only after rounding to float32.
    """
    grabbed = {}
    orig = B.np.column_stack

    def spy(arrs):
        r = orig(arrs)
        grabbed["A"] = r.copy()
        return r

    B.np.column_stack = spy
    try:
        names, _ = _batch(h, o)
    finally:
        B.np.column_stack = orig
    A = grabbed["A"]
    # batch scrubs non-finites after the float32 cast; mirror that on float64
    A[~np.isfinite(A.astype(np.float32))] = np.nan
    return names, A


def _stream(h, o):
    ctx = StreamCtx().fit_historical(h)
    eng = StreamM03Dyn()
    eng.fit_historical(ctx)
    rows = np.empty((len(o), len(eng.cols)), dtype=np.float64)
    for t in range(len(o)):
        ctx.push(o[t])
        rows[t] = eng.step(ctx)
    return eng.cols, rows


def _mismatch(h, o, limit=4):
    """Every (t, column, batch, stream) pair that is not bit-identical."""
    names, A64 = _batch64(h, o)
    cols, B64 = _stream(h, o)
    assert cols == names, "column names/order differ from batch"
    bad = []
    for M, N in ((A64, B64), (A64.astype(np.float32), B64.astype(np.float32))):
        eq = (M == N) | (np.isnan(M) & np.isnan(N))
        for t, j in np.argwhere(~eq)[:limit]:
            bad.append((int(t), names[j], M[t, j], N[t, j]))
    return bad


# ------------------------------------------------------------------- fixtures
def _garch(rng, n):
    x = np.empty(n)
    s = 1.0
    for i in range(n):
        e = rng.standard_normal()
        x[i] = s * e
        s = np.sqrt(0.02 + 0.12 * x[i] ** 2 + 0.85 * s * s)
    return x


def _ar1(rng, n, phi):
    x = rng.standard_normal(n)
    for i in range(1, n):
        x[i] += phi * x[i - 1]
    return x


def _cases():
    """>= 40 synthetic (label, hist, online) triples spanning the hard shapes."""
    rng = np.random.default_rng(20260819)
    C = []
    # -- short online (10..20) ------------------------------------------------
    for n in (10, 11, 13, 17, 20):
        C.append((f"short_online_{n}", rng.standard_normal(2000), rng.standard_normal(n)))
    # -- long online (900..999) -----------------------------------------------
    for n in (900, 951, 999):
        C.append((f"long_online_{n}", rng.standard_normal(3000), rng.standard_normal(n)))
    # -- constant history -----------------------------------------------------
    C.append(("const_hist", np.zeros(1500), rng.standard_normal(120)))
    C.append(("const_hist_nonzero", np.full(1200, 3.25), rng.standard_normal(200)))
    C.append(("const_hist_const_online", np.zeros(1100), np.zeros(60)))
    # -- near-zero-variance history -------------------------------------------
    C.append(("tiny_var_hist", 1e-9 * rng.standard_normal(1500), rng.standard_normal(180)))
    C.append(("tiny_var_hist_step", 1e-12 * rng.standard_normal(1000), rng.standard_normal(40)))
    C.append(("tiny_var_both", 1e-10 * rng.standard_normal(1000),
              1e-10 * rng.standard_normal(90)))
    # -- heavy tails ----------------------------------------------------------
    for n in (1000, 2500, 4000):
        C.append((f"t3_hist_{n}", rng.standard_t(3, n), rng.standard_t(3, 220)))
    # -- strong AR(1) ---------------------------------------------------------
    for phi in (0.6, 0.85, 0.97, -0.8):
        C.append((f"ar1_{phi}", _ar1(rng, 2500, phi), _ar1(rng, 300, phi)))
    # -- volatility clustering -------------------------------------------------
    for n in (1500, 3000):
        C.append((f"garch_{n}", _garch(rng, n), _garch(rng, 260)))
    C.append(("garch_break_vol", _garch(rng, 2000), 6.0 * _garch(rng, 300)))
    # -- break at tau = 0 ------------------------------------------------------
    for shift in (4.0, -9.0):
        C.append((f"break_tau0_mean_{shift}", rng.standard_normal(2000),
                  rng.standard_normal(300) + shift))
    C.append(("break_tau0_var", rng.standard_normal(2000), 5.0 * rng.standard_normal(300)))
    C.append(("break_tau0_ar", _ar1(rng, 2000, 0.9), _ar1(rng, 250, -0.9)))
    # -- break at the last point ------------------------------------------------
    for n in (12, 130, 640):
        o = rng.standard_normal(n)
        o[-1] += 40.0
        C.append((f"break_last_{n}", rng.standard_normal(2000), o))
    # -- huge single outliers ---------------------------------------------------
    for pos, mag in ((0, 1e7), (7, -1e6), (63, 1e9)):
        o = rng.standard_normal(200)
        o[pos] = mag
        C.append((f"outlier_{pos}", rng.standard_normal(2000), o))
    o = rng.standard_normal(300)
    o[100] = 1e30
    C.append(("outlier_float32_overflow", rng.standard_normal(2000), o))
    # -- mid-series regime switch ------------------------------------------------
    o = np.concatenate([rng.standard_normal(150), _ar1(rng, 150, 0.95)])
    C.append(("break_mid_ar", rng.standard_normal(2500), o))
    o = np.concatenate([rng.standard_normal(80), rng.standard_normal(80) * 0.05])
    C.append(("break_mid_shrink", rng.standard_normal(2500), o))
    # -- short / awkward histories (exercise the _local warm-up branches) --------
    for nh in (20, 40, 41, 104, 105, 106, 300):
        C.append((f"short_hist_{nh}", rng.standard_normal(nh), rng.standard_normal(55)))
    # -- assorted lengths --------------------------------------------------------
    for i in range(6):
        nh = int(rng.integers(1000, 5000))
        no = int(rng.integers(21, 900))
        C.append((f"mixed_{i}", rng.standard_normal(nh), rng.standard_normal(no)))
    # -- trend / drift ------------------------------------------------------------
    C.append(("linear_drift", rng.standard_normal(2000),
              rng.standard_normal(400) + np.linspace(0, 6, 400)))
    C.append(("sawtooth", rng.standard_normal(2000),
              np.sin(np.arange(300) * 0.7) * 3 + rng.standard_normal(300)))
    return C


CASES = _cases()
assert len(CASES) >= 40, len(CASES)


# ---------------------------------------------------------------------- tests
@pytest.mark.parametrize("label,h,o", CASES, ids=[c[0] for c in CASES])
def test_bitwise_parity_synthetic(label, h, o):
    bad = _mismatch(h, o)
    assert not bad, f"{label}: {len(bad)} mismatches, first={bad[:3]}"


def test_bitwise_parity_real_store():
    from sbr.store import load_store

    st = skip_on_missing_store(load_store)
    rng = np.random.default_rng(2026)
    idx = rng.choice(st.n_series, size=30, replace=False)
    for i in idx:
        h, o, _ = st.series(int(i))
        bad = _mismatch(h, o)
        assert not bad, f"series {i}: {len(bad)} mismatches, first={bad[:3]}"


def test_column_names_and_count():
    h = np.random.default_rng(0).standard_normal(2000)
    o = np.random.default_rng(1).standard_normal(50)
    names, A = _batch(h, o)
    ctx = StreamCtx().fit_historical(h)
    eng = StreamM03Dyn()
    eng.fit_historical(ctx)
    assert eng.cols == names
    assert len(eng.cols) == A.shape[1] == 60


def test_prefix_poison():
    """Row t cannot depend on anything after t.

    Two runs share the first ``cut`` observations and then diverge completely
    (different values, different length).  The first ``cut`` rows must be
    bit-identical -- and identical to the rows the batch module produces from
    the truncated prefix.
    """
    rng = np.random.default_rng(4242)
    h = rng.standard_normal(2200)
    o = rng.standard_normal(320)
    _, ref = _stream(h, o)
    for cut in (1, 5, 17, 64, 200):
        poison = o.copy()
        poison[cut:] = rng.standard_normal(len(o) - cut) * 50.0 + 100.0
        _, got = _stream(h, poison)
        a, b = ref[:cut], got[:cut]
        assert np.array_equal(a, b, equal_nan=True), f"poisoned tail changed row < {cut}"
        # and truncating the stream entirely reproduces the same prefix
        _, cutrun = _stream(h, o[:cut])
        assert np.array_equal(ref[:cut], cutrun, equal_nan=True), f"cut {cut}"


def test_state_is_not_shared_between_series():
    """A reused engine object must be fully reset by fit_historical."""
    rng = np.random.default_rng(11)
    h1, o1 = rng.standard_normal(1500), rng.standard_normal(90)
    h2, o2 = rng.standard_t(3, 1800), rng.standard_normal(140)
    _, ref2 = _stream(h2, o2)
    eng = StreamM03Dyn()
    ctx = StreamCtx().fit_historical(h1)
    eng.fit_historical(ctx)
    for x in o1:
        ctx.push(x)
        eng.step(ctx)
    ctx = StreamCtx().fit_historical(h2)
    eng.fit_historical(ctx)
    got = np.stack([(ctx.push(x), eng.step(ctx))[1] for x in o2])
    assert np.array_equal(ref2, got, equal_nan=True)


# ---- differential fuzz: every scalar re-expression vs the batch expression ---
#
# The streaming module evaluates each statistic at ONE index, in scalar
# arithmetic, instead of on a length-n vector.  For an elementwise expression
# that is the same IEEE operations on the same operands in the same order -- but
# "should be" is not evidence, so every substitution is pinned here against the
# batch original over random inputs that include +-inf, NaN, +-0.0, 1e300,
# 1e-300 and denormals.  These ran at 10x these counts before the scalar forms
# were adopted; the counts here are trimmed to keep CI honest but quick.
_FUZZ_N = 20000


def _nasty(rng, n):
    a = rng.standard_normal(n)
    k = max(n // 8, 1)
    a[:k] *= 1e300
    a[k:2 * k] *= 1e-300
    a[2 * k:3 * k] = 0.0
    a[3 * k:3 * k + k // 4] = -0.0
    a[4 * k:4 * k + k // 8] = np.inf
    a[5 * k:5 * k + k // 8] = -np.inf
    a[6 * k:6 * k + k // 8] = np.nan
    a[7 * k:] = rng.standard_normal(n - 7 * k) * rng.choice([1e-9, 1.0, 1e9], n - 7 * k)
    rng.shuffle(a)
    return a


def _same(a, b):
    a, b = float(a), float(b)
    return a == b or (a != a and b != b)


class _RA:
    """Batch-shaped accessor: R(name, mult) -> length-1 array."""

    def __init__(self, d):
        self.d = d

    def __call__(self, name, mult=1.0):
        return self.d[(name, mult)]


class _RS:
    """Streaming accessor surface used by the scalar forms."""

    ic, isn, ih, ip = 0, 6, 12, 16

    def __init__(self, d, V=None):
        self.d = d
        self.V = V
        self.specp = None
        self.hwlog = None

    def s(self, name, mult=1.0):
        return float(self.d[(name, mult)][0])

    def band(self, base, k):
        return self.V[base:base + k]


def test_fuzz_np_log_is_elementwise_across_shapes():
    """Scalar forms call np.log on a python float; batch calls it on an array."""
    rng = np.random.default_rng(5150)
    x = np.abs(_nasty(rng, _FUZZ_N))
    with np.errstate(all="ignore"):
        ref = np.log(x)
        for i in range(0, _FUZZ_N, 3):
            assert _same(np.log(float(x[i])), ref[i]), float(x[i])
            assert _same(np.log(np.array([x[i]]))[0], ref[i]), float(x[i])


def test_fuzz_acf_scalar_matches_batch():
    rng = np.random.default_rng(1)
    a, b, c = _nasty(rng, _FUZZ_N), _nasty(rng, _FUZZ_N), _nasty(rng, _FUZZ_N)
    fb = B._acf("m", "d", "m", "l")
    fs = S._acf_s("m", "d", "m", "l")
    with np.errstate(all="ignore"):
        for i in range(_FUZZ_N):
            d = {("m", 1.0): a[i:i + 1], ("d", 1.0): b[i:i + 1], ("l", 1.0): c[i:i + 1]}
            assert _same(fb(_RA(d), np.array([37.0]), np.array([36]))[0],
                         fs(_RS(d), 37, 36)), i


def test_fuzz_drift_and_ratio_scalar_match_batch():
    rng = np.random.default_rng(2)
    n = _FUZZ_N // 2
    a, b = _nasty(rng, n), _nasty(rng, n)
    ws = rng.integers(1, 1000, n)
    ts = rng.integers(0, 1000, n)
    fbr = B._ratio("u", 0.5, "v", 1.0)
    fsr = S._ratio_s("u", 0.5, "v", 1.0)
    with np.errstate(all="ignore"):
        for i in range(n):
            d = {("x", 1.0): a[i:i + 1], ("tx", 1.0): b[i:i + 1]}
            assert _same(B._drift(_RA(d), np.array([float(ws[i])]), np.array([int(ts[i])]))[0],
                         S._drift_s(_RS(d), int(ws[i]), int(ts[i]))), i
            d = {("u", 0.5): a[i:i + 1], ("v", 1.0): b[i:i + 1]}
            assert _same(fbr(_RA(d), None, None)[0], fsr(_RS(d), None, None)), i


def test_fuzz_hjorth_scalar_matches_batch():
    rng = np.random.default_rng(3)
    n = _FUZZ_N // 2
    cols = {k: _nasty(rng, n) for k in ("x", "x2", "d1", "d1_2", "d2", "d2_2")}
    with np.errstate(all="ignore"):
        for i in range(n):
            d = {(k, 1.0): v[i:i + 1] for k, v in cols.items()}
            assert _same(B._hj_mob(_RA(d), None, None)[0], S._hj_mob_s(_RS(d), None, None)), i
            assert _same(B._hj_comp(_RA(d), None, None)[0], S._hj_comp_s(_RS(d), None, None)), i


def test_fuzz_band_blocks_match_batch():
    """spectral / Haar / permutation-entropy read-outs built from a slice.

    These are the only statistics with a reduction whose summation order numpy
    chooses, so the reduction is left to numpy on an identically shaped (6, 1)
    block; what is fuzzed here is that building that block from a contiguous
    slice of the rolling-mean vector gives the same result as batch's
    ``np.stack`` of six length-1 arrays.
    """
    rng = np.random.default_rng(4)
    n = _FUZZ_N // 4
    with np.errstate(all="ignore"):
        for _ in range(n):
            V = _nasty(rng, 22)
            V[16:22] = np.abs(V[16:22])          # p* are occupancy counts >= 0
            d = {}
            for q in range(6):
                d[(f"c{q}", 1.0)] = V[q:q + 1]
                d[(f"s{q}", 1.0)] = V[6 + q:6 + q + 1]
                d[(f"p{q}", 1.0)] = V[16 + q:16 + q + 1]
            for j, sc in enumerate(B.HSCALES):
                d[(f"h{sc}", 1.0)] = V[12 + j:12 + j + 1]
            ra, rs = _RA(d), _RS(d, V)
            assert np.array_equal(B._spec_p(ra), S._specp_s(rs), equal_nan=True)
            rs.specp = None
            for fb, fs in ((B._sp_low, S._sp_low_s), (B._sp_high, S._sp_high_s),
                           (B._sp_ent, S._sp_ent_s), (B._sp_cen, S._sp_cen_s),
                           (S._sp_dom_batch, S._sp_dom_s), (B._pe, S._pe_s),
                           (B._hw_r(0, 1), S._hw_r_s(0, 1)),
                           (B._hw_r(1, 2), S._hw_r_s(1, 2)),
                           (B._hw_r(2, 3), S._hw_r_s(2, 3)),
                           (B._hw_hurst, S._hw_hurst_s)):
                assert _same(fb(ra, None, None)[0], fs(rs, None, None)), fs


# ---- the two numerical assumptions the port rests on ------------------------
def test_cos_sin_are_elementwise_across_shapes():
    """The DFT quadratures are formed on a length-6 vector, batch on (6, n)."""
    j = np.arange(1200, dtype=np.float64) - float(B.PAD)
    ang = 2.0 * np.pi * np.outer(B.FREQS, j)
    C, S = np.cos(ang), np.sin(ang)
    for t in range(0, 1200, 7):
        one = 2.0 * np.pi * (B.FREQS * j[t])
        assert np.array_equal(np.cos(one), C[:, t])
        assert np.array_equal(np.sin(one), S[:, t])


def test_loglin_is_length_independent():
    """The null mu(W)/sd(W) tables are built with one long _loglin call."""
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 8.0, size=4000)
    for _ in range(40):
        y = rng.standard_normal(3)
        full = B._loglin(y, x)
        for i in rng.integers(0, len(x), size=20):
            assert B._loglin(y, np.array([x[i]]))[0] == full[i]


# ---------------------------------------------------------------------- timing
def test_timing_microseconds_per_observation(capsys):
    h = np.random.default_rng(0).standard_normal(3000)
    o = np.random.default_rng(1).standard_normal(999)
    n = len(o)

    ctx = StreamCtx().fit_historical(h)
    c0 = time.process_time()
    for x in o:
        ctx.push(x)
    push_us = (time.process_time() - c0) / n * 1e6

    ctx = StreamCtx().fit_historical(h)
    eng = StreamM03Dyn()
    c0 = time.process_time()
    eng.fit_historical(ctx)
    fit_ms = (time.process_time() - c0) * 1e3

    c0 = time.process_time()
    for x in o:
        ctx.push(x)
        eng.step(ctx)
    both_us = (time.process_time() - c0) / n * 1e6
    step_us = both_us - push_us

    # O(1) per observation: the last decile must not be slower than the first
    # by more than a constant factor (a prefix rescan would be ~10x here).
    ctx = StreamCtx().fit_historical(h)
    eng = StreamM03Dyn()
    eng.fit_historical(ctx)
    early = late = 0.0
    for t, x in enumerate(o):
        ctx.push(x)
        c0 = time.process_time()
        eng.step(ctx)
        dt = time.process_time() - c0
        if 20 <= t < 120:
            early += dt
        elif t >= n - 100:
            late += dt

    with capsys.disabled():
        print(f"\n[m03_dyn] fit_historical {fit_ms:8.2f} ms/series")
        print(f"[m03_dyn] ctx.push       {push_us:8.1f} us/obs")
        print(f"[m03_dyn] step           {step_us:8.1f} us/obs   "
              f"(push+step {both_us:.1f})")
        print(f"[m03_dyn] step t=20..120 {early / 100 * 1e6:8.1f} us/obs   "
              f"t=n-100..n {late / 100 * 1e6:8.1f} us/obs")

    assert late < 4.0 * early + 5e-3, "per-step cost grows with t: prefix rescan?"
    assert step_us < 20000.0
