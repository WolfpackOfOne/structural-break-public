"""Bitwise parity: StreamM07Bayes (incremental) vs m07_bayes (batch).

The batch module returns float32 (``np.column_stack(out).astype(np.float32)``
followed by ``A[~np.isfinite(A)] = np.nan``), so *its output* is the parity
target: every streamed row is compared to the corresponding batch row with
``atol=0``, NaN == NaN, no tolerance anywhere.
"""
import time

import numpy as np
import pytest

from sbr.features.base import REGISTRY, load_all, make_ctx
from sbr.stream.ctx import StreamCtx
from sbr.stream.s_m07_bayes import COLS, StreamM07Bayes

load_all()
MODULE = "m07_bayes"


# ---------------------------------------------------------------- machinery
def _stream_rows(h, o):
    ctx = StreamCtx().fit_historical(h)
    eng = StreamM07Bayes()
    eng.fit_historical(ctx)
    B = np.empty((len(o), len(COLS)), dtype=np.float64)
    for t, x in enumerate(o):
        ctx.push(x)
        B[t] = eng.step(ctx)
    return B


def _batch(h, o):
    return REGISTRY[MODULE].fn(make_ctx(h, o))


def _assert_parity(h, o, tag):
    names, A = _batch(h, o)
    assert names == COLS, "column order/name drift"
    B = _stream_rows(h, o).astype(np.float32)
    bad = ~((A == B) | (np.isnan(A) & np.isnan(B)))
    if bad.any():
        c = int(np.argmax(bad.any(axis=0)))
        r = int(np.argmax(bad[:, c]))
        raise AssertionError(
            f"{tag}: {int(bad.sum())} non-identical values in "
            f"{int(bad.any(axis=0).sum())} columns; first = "
            f"{names[c]} row {r}: batch={A[r, c]!r} stream={B[r, c]!r}")


# ------------------------------------------------------------ synthetic bank
def _ar1(rng, n, rho, scale=1.0):
    x = rng.standard_normal(n) * scale
    for i in range(1, n):
        x[i] += rho * x[i - 1]
    return x


def _garch(rng, n):
    x = np.empty(n)
    s2 = 1.0
    for i in range(n):
        s2 = 0.02 + 0.12 * (x[i - 1] ** 2 if i else 1.0) + 0.85 * s2
        x[i] = rng.standard_normal() * np.sqrt(s2)
    return x


def _bank():
    """>= 40 synthetic (hist, online) pairs spanning the required regimes."""
    rng = np.random.default_rng(20260819)
    S = []
    def add(name, h, o):
        return S.append((name, np.asarray(h, float), np.asarray(o, float)))

    # --- short online (10-20) --------------------------------------------
    for _i, n_o in enumerate((10, 11, 13, 16, 20)):
        add(f"short{n_o}", rng.standard_normal(1500), rng.standard_normal(n_o))
    # --- long online (900-999) -------------------------------------------
    for n_o in (900, 941, 997, 998, 999):
        h = rng.standard_normal(1800)
        o = rng.standard_normal(n_o)
        o[n_o // 2:] *= 2.2                       # break in the middle
        add(f"long{n_o}", h, o)
    # --- constant / near-constant history --------------------------------
    add("const_hist", np.ones(1400), rng.standard_normal(60))
    add("const_hist2", np.full(1400, -3.25), rng.standard_normal(120) * 0.1)
    add("near_zero_var", rng.standard_normal(1400) * 1e-9,
        rng.standard_normal(60) * 1e-9)
    add("near_zero_var2", rng.standard_normal(1400) * 1e-12,
        rng.standard_normal(80) * 1e-6)
    add("stepped_hist", np.repeat(rng.standard_normal(70), 20),
        rng.standard_normal(90))
    # --- heavy tails ------------------------------------------------------
    for i, df in enumerate((2, 3, 3, 4)):
        add(f"t{df}_{i}", rng.standard_t(df, 2000), rng.standard_t(df, 150 + 30 * i))
    add("t3_break", rng.standard_t(3, 2000),
        np.concatenate([rng.standard_t(3, 100), rng.standard_t(3, 100) * 3.0]))
    # --- strong AR(1) -----------------------------------------------------
    for rho in (0.9, 0.75, -0.8, 0.95):
        add(f"ar{rho}", _ar1(rng, 2200, rho), _ar1(rng, 220, rho))
    add("ar_break_dep", _ar1(rng, 2200, 0.8),
        np.concatenate([_ar1(rng, 100, 0.8), _ar1(rng, 100, -0.2)]))
    # --- volatility clustering -------------------------------------------
    add("garch", _garch(rng, 2000), _garch(rng, 200))
    add("garch_break", _garch(rng, 2000),
        np.concatenate([_garch(rng, 100), _garch(rng, 100) * 3.0]))
    add("garch_long", _garch(rng, 1500), _garch(rng, 400))
    # --- break at tau = 0 -------------------------------------------------
    add("break_tau0_scale", rng.standard_normal(1800), rng.standard_normal(180) * 4.0)
    add("break_tau0_loc", rng.standard_normal(1800), rng.standard_normal(180) + 3.0)
    add("break_tau0_dep", rng.standard_normal(1800), _ar1(rng, 180, 0.85))
    # --- break at the last point -----------------------------------------
    for n_o in (40, 150):
        o = rng.standard_normal(n_o)
        o[-1] *= 12.0
        add(f"break_last{n_o}", rng.standard_normal(1600), o)
    o = rng.standard_normal(60)
    o[-2:] += 9.0
    add("break_last2", rng.standard_normal(1600), o)
    # --- huge single outliers --------------------------------------------
    for k, mag in ((30, 1e3), (77, 1e6), (5, 1e9)):
        o = rng.standard_normal(160)
        o[k] = mag
        add(f"outlier{mag:g}", rng.standard_normal(1700), o)
    o = rng.standard_normal(160)
    o[80] = -1e8
    add("outlier_neg", rng.standard_normal(1700), o)
    h = rng.standard_normal(1700)
    h[900] = 5e5
    add("outlier_in_hist", h, rng.standard_normal(120))
    # --- short / degenerate history (drops into the AR fallback path) ----
    add("hist150", rng.standard_normal(150), rng.standard_normal(40))
    add("hist205", rng.standard_normal(205), rng.standard_normal(40))
    add("hist210", rng.standard_normal(210), rng.standard_normal(40))
    add("hist_400", rng.standard_normal(400), rng.standard_normal(200))
    # --- misc mixtures ----------------------------------------------------
    add("skewed", rng.exponential(1.0, 2000), rng.exponential(1.0, 130))
    add("skew_break", rng.exponential(1.0, 2000),
        np.concatenate([rng.exponential(1.0, 60), rng.exponential(4.0, 60)]))
    add("discrete", rng.integers(-3, 4, 2000).astype(float),
        rng.integers(-3, 4, 100).astype(float))
    add("bimodal", np.concatenate([rng.standard_normal(1000) - 3,
                                   rng.standard_normal(1000) + 3]),
        rng.standard_normal(100) * 2)
    add("drift", rng.standard_normal(2000),
        rng.standard_normal(200) + np.linspace(0, 2, 200))
    add("mean_shift_small", rng.standard_normal(2000),
        np.concatenate([rng.standard_normal(100), rng.standard_normal(100) + 0.6]))
    add("var_shrink", rng.standard_normal(2000),
        np.concatenate([rng.standard_normal(100), rng.standard_normal(100) * 0.4]))
    add("single_online", rng.standard_normal(1500), rng.standard_normal(1))
    add("two_online", rng.standard_normal(1500), rng.standard_normal(2))
    return S


BANK = _bank()
assert len(BANK) >= 40, len(BANK)


@pytest.mark.parametrize("name,h,o", BANK, ids=[s[0] for s in BANK])
def test_parity_synthetic_bitwise(name, h, o):
    _assert_parity(h, o, name)


# ----------------------------------------------------------------- real data
def _real_ids(k=30, seed=20260819):
    from sbr.store import load_store

    st = load_store()
    rng = np.random.default_rng(seed)
    return st, sorted({int(i) for i in rng.integers(0, st.n_series, k + 10)})[:k]


try:
    _STORE, _IDS = _real_ids()
except Exception:                                          # pragma: no cover
    _STORE, _IDS = None, []


@pytest.mark.skipif(_STORE is None, reason="store unavailable")
@pytest.mark.parametrize("i", _IDS)
def test_parity_real_bitwise(i):
    h, o, _ = _STORE.series(i)
    _assert_parity(h, o, f"series{i}")


def test_real_count():
    assert len(_IDS) >= 30


# ------------------------------------------------------------ causality
def test_prefix_poison():
    """Rows already emitted cannot be changed by any later observation."""
    rng = np.random.default_rng(4242)
    h = rng.standard_normal(2000)
    o = rng.standard_normal(240)
    rows = _stream_rows(h, o)

    for cut in (1, 7, 33, 100, 239):
        poisoned = o.copy()
        poisoned[cut:] = rng.standard_normal(len(o) - cut) * 50.0 + 100.0
        got = _stream_rows(h, poisoned)[:cut]
        assert np.array_equal(np.nan_to_num(got, nan=-7.7),
                              np.nan_to_num(rows[:cut], nan=-7.7)), \
            f"row < {cut} moved when the future was poisoned"

    # and the same rows must equal the batch module built on the prefix only
    for cut in (3, 10, 37, 240):
        _, A = _batch(h, o[:cut])
        B = rows[:cut].astype(np.float32)
        bad = ~((A == B) | (np.isnan(A) & np.isnan(B)))
        assert not bad.any(), f"prefix {cut}: {int(bad.sum())} values differ"


def test_cols_match_batch():
    rng = np.random.default_rng(1)
    names, A = _batch(rng.standard_normal(1200), rng.standard_normal(12))
    assert names == COLS
    assert A.shape[1] == len(COLS) == 50
    assert StreamM07Bayes().cols == names


def test_nonfinite_are_nan():
    """The batch module nan-fills non-finite entries; so must the stream."""
    rng = np.random.default_rng(9)
    h = rng.standard_normal(1500)
    o = rng.standard_normal(80)
    o[40] = 1e300
    B = _stream_rows(h, o)
    assert not np.isinf(B).any()


# ---------------------------------------------------------------- timing
def test_timing_us_per_obs(capsys):
    rng = np.random.default_rng(11)
    h = rng.standard_normal(3000)
    o = rng.standard_normal(1000)

    ctx = StreamCtx().fit_historical(h)
    eng = StreamM07Bayes()
    t0 = time.perf_counter()
    eng.fit_historical(ctx)
    fit_ms = (time.perf_counter() - t0) * 1e3

    t_push = t_step = 0.0
    for x in o:
        a = time.perf_counter()
        ctx.push(x)
        b = time.perf_counter()
        eng.step(ctx)
        t_step += time.perf_counter() - b
        t_push += b - a
    n = len(o)
    with capsys.disabled():
        print(f"\n[m07_bayes] fit_historical {fit_ms:8.1f} ms/series")
        print(f"[m07_bayes] StreamCtx.push {t_push / n * 1e6:8.1f} us/obs (shared)")
        print(f"[m07_bayes] step           {t_step / n * 1e6:8.1f} us/obs")
    assert t_step / n < 5e-3      # generous ceiling: must stay O(1)


# ---------------------------------------------------------------------------
# Regression guard for the batch/stream BOCPD constant-table divergence fixed on
# engineering/rt600-final-reliability-2026.  See
# engineering/reports/rt600_final_reliability/STREAM_PARITY_REPRO.md.
def test_bocpd_ct_table_is_single_sourced():
    """The Student-t constant table must come from one implementation.

    ``math.lgamma`` is the one primitive in this module where numba and CPython
    disagree (up to 512 ULP on this grid; log/exp/log1p agree bitwise).  It is
    used only to build ``ct``, which enters every run length's predictive
    likelihood, so two independently computed tables put the batch-trained and
    stream-served features on permanently different constants -- a train/serve
    skew, not float noise.  Both sides must read the same table.
    """
    import math

    from sbr.features.m07_bayes import BO_AL0, BO_KAP0, R_MAX, _bocpd_ct
    from sbr.stream.s_m07_bayes import _BocpdStream

    shared = _bocpd_ct(BO_AL0, R_MAX)
    stream = _BocpdStream(0.0, 1.0, BO_KAP0, BO_AL0,
                          math.log(0.004), math.log1p(-0.004), R_MAX).ct
    assert len(stream) == R_MAX
    for r in range(R_MAX):
        assert stream[r] == float(shared[r]), (
            f"ct[{r}] diverged: stream {stream[r]!r} vs shared {shared[r]!r}; "
            "the streaming kernel is recomputing the table instead of importing it")


def test_bocpd_stream_matches_batch_kernel_bitwise():
    """``_BocpdStream`` must reproduce ``_bocpd`` exactly, JIT or not.

    Compares against the compiled kernel actually used to build the feature
    cache, not against its Python source -- the two differ, and the compiled one
    is what the frozen model was trained on.
    """
    import math

    from sbr.features.m07_bayes import BO_AL0, BO_HAZ, BO_KAP0, R_MAX, _bocpd
    from sbr.stream.s_m07_bayes import _BocpdStream

    lbh, lb1 = math.log(BO_HAZ), math.log1p(-BO_HAZ)
    rng = np.random.default_rng(20260826)
    for tag, x in (("gauss", rng.standard_normal(400)),
                   ("shift", np.r_[rng.standard_normal(200),
                                   rng.standard_normal(200) + 3.0]),
                   ("vol", np.r_[rng.standard_normal(200),
                                 rng.standard_normal(200) * 5.0])):
        batch = _bocpd(np.ascontiguousarray(x), 0.0, 24.0, BO_KAP0, BO_AL0,
                       lbh, lb1, R_MAX)
        bo = _BocpdStream(0.0, 24.0, BO_KAP0, BO_AL0, lbh, lb1, R_MAX)
        stream = np.array([bo.step(v) for v in x.tolist()], dtype=np.float64)
        bad = int((~((batch == stream)
                     | (np.isnan(batch) & np.isnan(stream)))).sum())
        assert bad == 0, f"{tag}: {bad}/{batch.size} float64 cells differ"
