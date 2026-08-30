"""Bitwise parity: StreamM06Loc (incremental) vs the batch ``m06_loc`` module.

The batch module is the specification.  Every assertion here is ``atol=0`` with
``NaN == NaN``; there is no tolerance anywhere in this file, because none is
needed (see the module docstring of ``sbr.stream.s_m06_loc`` for why bitwise is
reachable).

Coverage, per the streaming contract:
  * >= 40 synthetic series spanning short/long online segments, constant and
    near-zero-variance history, t3 tails, strong AR(1), volatility clustering,
    a break at tau = 0, a break at the very last point, and huge single
    outliers -- in both the history and the online segment;
  * >= 30 random real series from ``sbr.store.load_store()``;
  * a prefix/poison test (later observations cannot move an earlier row);
  * a timing test that prints microseconds per observation.
"""
import time

import numpy as np
import pytest
from conftest import skip_on_missing_store

from sbr.features.base import REGISTRY, load_all, make_ctx
from sbr.stream.ctx import StreamCtx
from sbr.stream.s_m06_loc import StreamM06Loc

MODULE = "m06_loc"

load_all()


# --------------------------------------------------------------------- utils
def _run_stream(h, o, upto=None):
    """Drive the streaming engine over ``o`` and stack the rows (float32)."""
    ctx = StreamCtx().fit_historical(h)
    eng = StreamM06Loc()
    eng.fit_historical(ctx)
    n = len(o) if upto is None else upto
    rows = np.empty((n, len(eng.cols)), dtype=np.float32)
    for t in range(n):
        ctx.push(o[t])
        rows[t] = eng.step(ctx).astype(np.float32)
    return eng.cols, rows


def _batch(h, o):
    return REGISTRY[MODULE].fn(make_ctx(np.asarray(h, dtype=np.float64),
                                        np.asarray(o, dtype=np.float64)))


def _diff(names, A, B):
    """Bitwise comparison; returns a human-readable report ('' when equal)."""
    a = np.asarray(A, dtype=np.float64)
    b = np.asarray(B, dtype=np.float64)
    if a.shape != b.shape:
        return f"shape {a.shape} != {b.shape}"
    bad = ~((a == b) | (np.isnan(a) & np.isnan(b)))
    if not bad.any():
        return ""
    msgs = []
    for j in np.where(bad.any(axis=0))[0][:6]:
        rows = np.where(bad[:, j])[0]
        t0 = int(rows[0])
        msgs.append(f"{names[j]}: {len(rows)} rows, first t={t0} "
                    f"batch={a[t0, j]!r} stream={b[t0, j]!r}")
    return f"{int(bad.sum())} cells differ | " + " ; ".join(msgs)


def _assert_parity(h, o, tag):
    names, A = _batch(h, o)
    cols, B = _run_stream(h, o)
    assert cols == names, f"{tag}: column order/name mismatch"
    d = _diff(names, A, B)
    assert not d, f"{tag}: {d}"


# ---------------------------------------------------------------- synthetics
def _vol_cluster(rng, n):
    """GARCH-ish volatility clustering."""
    v = np.ones(n)
    x = np.zeros(n)
    for i in range(1, n):
        v[i] = 0.05 + 0.85 * v[i - 1] + 0.10 * x[i - 1] ** 2
        x[i] = rng.standard_normal() * np.sqrt(v[i])
    return x


def _hist(rng, kind, n):
    if kind == "gauss":
        return rng.standard_normal(n)
    if kind == "const":
        return np.full(n, 1.25)
    if kind == "nearzero":
        return np.full(n, -0.5) + rng.standard_normal(n) * 1e-13
    if kind == "t3":
        return rng.standard_t(3, n)
    if kind == "ar1":
        h = rng.standard_normal(n)
        for i in range(1, n):
            h[i] += 0.9 * h[i - 1]
        return h
    if kind == "ar6":
        h = rng.standard_normal(n)
        for i in range(6, n):
            h[i] += 0.4 * h[i - 1] - 0.3 * h[i - 2] + 0.2 * h[i - 6]
        return h
    if kind == "vol":
        return _vol_cluster(rng, n)
    if kind == "outlier":
        h = rng.standard_normal(n)
        h[n // 3] = 5e3
        return h
    if kind == "skew":
        return rng.exponential(1.0, n) - 1.0
    raise ValueError(kind)


def _online(rng, kind, n, tau):
    o = rng.standard_normal(n)
    if kind == "none":
        pass
    elif kind == "scale":
        o[tau:] *= 3.5
    elif kind == "loc":
        o[tau:] += 2.0
    elif kind == "ar":
        for i in range(max(tau, 1), n):
            o[i] += 0.8 * o[i - 1]
    elif kind == "vol":
        o[tau:] = _vol_cluster(rng, n - tau) * 2.0
    elif kind == "outlier":
        o[min(tau, n - 1)] = 1e6
    elif kind == "t3":
        o[tau:] = rng.standard_t(3, n - tau)
    else:
        raise ValueError(kind)
    return o


def _cases():
    """>= 40 (hist, online, tag) synthetic series covering the required span."""
    rng = np.random.default_rng(20260819)
    out = []

    # --- short online segments (10-20) --------------------------------------
    for i, n_on in enumerate((10, 11, 15, 16, 17, 20)):
        h = _hist(rng, "gauss", 1500 + 37 * i)
        out.append((h, _online(rng, "scale", n_on, 0), f"short{n_on}"))

    # --- long online segments (900-999) -------------------------------------
    for i, n_on in enumerate((900, 947, 999)):
        h = _hist(rng, "gauss", 2000 + 101 * i)
        out.append((h, _online(rng, "scale", n_on, n_on // 2), f"long{n_on}"))
    out.append((_hist(rng, "t3", 3000), _online(rng, "none", 990, 0), "long_t3_nobreak"))

    # --- one series per history regime --------------------------------------
    for kind in ("gauss", "const", "nearzero", "t3", "ar1", "ar6", "vol",
                 "outlier", "skew"):
        h = _hist(rng, kind, 2500)
        out.append((h, _online(rng, "scale", 400, 150), f"hist_{kind}"))

    # --- break at tau = 0 and at the very last point ------------------------
    for kind in ("gauss", "t3", "ar1", "vol"):
        h = _hist(rng, kind, 2200)
        out.append((h, _online(rng, "scale", 300, 0), f"tau0_{kind}"))
        o = _online(rng, "none", 300, 0)
        o[-1] *= 40.0
        out.append((h, o, f"taulast_{kind}"))

    # --- huge single outliers, in history and online ------------------------
    h = _hist(rng, "gauss", 2000)
    for pos in (0, 1, 199, 250, 399):
        o = _online(rng, "none", 400, 0)
        o[pos] = 1e8
        out.append((h, o, f"spike{pos}"))
    ho = _hist(rng, "gauss", 2000)
    ho[1000] = 1e9
    out.append((ho, _online(rng, "none", 300, 0), "hist_spike"))

    # --- assorted break flavours -------------------------------------------
    for kind in ("none", "scale", "loc", "ar", "vol", "outlier", "t3"):
        h = _hist(rng, "gauss", 1800)
        out.append((h, _online(rng, kind, 350, 120), f"break_{kind}"))

    # --- short history: both engines fall back to an all-NaN block ----------
    out.append((_hist(rng, "gauss", 700), _online(rng, "scale", 200, 50), "hist700"))
    out.append((_hist(rng, "gauss", 716), _online(rng, "scale", 200, 50), "hist716"))
    out.append((_hist(rng, "gauss", 717), _online(rng, "scale", 200, 50), "hist717"))

    # --- online exactly at grid boundaries (candidate set turns over) -------
    h = _hist(rng, "gauss", 2600)
    for n_on in (8, 12, 24, 32, 48, 64, 128, 256, 512):
        out.append((h, _online(rng, "scale", n_on, 0), f"grid{n_on}"))
    return out


CASES = _cases()


def test_case_count():
    assert len(CASES) >= 40, len(CASES)


@pytest.mark.parametrize("idx", range(len(CASES)))
def test_synthetic_bitwise(idx):
    h, o, tag = CASES[idx]
    _assert_parity(h, o, tag)


# -------------------------------------------------------------------- real
def test_real_store_bitwise():
    from sbr.store import load_store

    st = skip_on_missing_store(load_store)
    rng = np.random.default_rng(606)
    ids = rng.choice(st.n_series, size=30, replace=False)
    for i in ids:
        h, o, _tau = st.series(int(i))
        _assert_parity(h, o, f"series{int(i)}")


def test_real_store_extremes_bitwise():
    """The shortest and the longest real online segments in the store."""
    from sbr.store import load_store

    st = skip_on_missing_store(load_store)
    n_on = st.meta.n_online.to_numpy()
    for i in list(np.argsort(n_on)[:3]) + list(np.argsort(n_on)[-3:]):
        h, o, _tau = st.series(int(i))
        _assert_parity(h, o, f"series{int(i)}(n={int(n_on[i])})")


# --------------------------------------------------------------- causality
def test_prefix_poison():
    """Rows already emitted cannot be moved by anything that arrives later.

    Structurally guaranteed (``step`` reads only ``ctx`` at the current index
    and carried state), but asserted anyway: two streams fed an identical
    prefix and then wildly different futures must agree on the shared prefix,
    and the batch module rebuilt on the truncated prefix must agree too.
    """
    rng = np.random.default_rng(4242)
    h = rng.standard_normal(2400)
    o = rng.standard_normal(400)
    cut = 137

    poisoned = o.copy()
    poisoned[cut:] = rng.standard_normal(400 - cut) * 250.0 + 900.0
    poisoned[cut + 3] = 1e12
    poisoned[cut + 9] = -np.inf

    names, A = _run_stream(h, o)
    _, B = _run_stream(h, poisoned)
    assert not _diff(names, A[:cut], B[:cut]), "future observations moved a past row"

    # ... and the batch module, rebuilt on the truncated online segment, agrees
    bnames, C = _batch(h, o[:cut])
    assert bnames == names
    assert not _diff(names, C, A[:cut]), "streamed prefix != batch on the prefix"


def test_buffer_growth_bitwise():
    """Online segments longer than the initial 1024-point buffers still match."""
    rng = np.random.default_rng(31337)
    h = rng.standard_normal(3000)
    o = rng.standard_normal(2100)
    o[1500:] *= 2.5
    _assert_parity(h, o, "grow2100")


def test_cols_before_fit_raises():
    eng = StreamM06Loc()
    with pytest.raises(RuntimeError):
        _ = eng.cols


def test_n_cols_matches_batch():
    rng = np.random.default_rng(1)
    h = rng.standard_normal(2000)
    o = rng.standard_normal(50)
    names, A = _batch(h, o)
    cols, B = _run_stream(h, o)
    assert len(names) == A.shape[1] == len(cols) == B.shape[1] == 60


# ------------------------------------------------------------------ timing
def test_timing_microseconds_per_observation(capsys):
    from sbr.store import load_store

    st = skip_on_missing_store(load_store)
    rng = np.random.default_rng(9)
    ids = rng.choice(st.n_series, size=5, replace=False)

    n_obs = 0
    t_step = 0.0
    t_all = 0.0
    for i in ids:
        h, o, _tau = st.series(int(i))
        ctx = StreamCtx().fit_historical(h)
        eng = StreamM06Loc()
        eng.fit_historical(ctx)
        t0 = time.perf_counter()
        for x in o:
            ctx.push(x)
            t1 = time.perf_counter()
            eng.step(ctx)
            t_step += time.perf_counter() - t1
        t_all += time.perf_counter() - t0
        n_obs += len(o)

    us_step = 1e6 * t_step / n_obs
    us_all = 1e6 * t_all / n_obs
    with capsys.disabled():
        print(f"\n[m06_loc] {n_obs} observations: "
              f"{us_step:.1f} us/obs in StreamM06Loc.step, "
              f"{us_all:.1f} us/obs including StreamCtx.push")
    # generous: the CI box is shared, this only guards against an O(n) blow-up
    assert us_step < 5000.0
