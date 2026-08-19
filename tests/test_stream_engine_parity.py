"""END-TO-END parity: the shared StreamEngine vs the batch feature pipeline.

This is the Research Director's own check.  It does not trust the per-module
test files: it rebuilds the full 500-column production vector from the streaming
engine and compares it, bit for bit, against the concatenation the batch driver
would have written into cache/features.
"""
import json

import numpy as np
import pytest

from sbr.features.base import REGISTRY, load_all, make_ctx
from sbr.store import load_store
from sbr.stream.engine import MODULE_ORDER, StreamEngine

load_all()


def _batch_row_block(hist, online):
    names, mats = [], []
    ctx = make_ctx(hist, online)
    for m in MODULE_ORDER:
        c, A = REGISTRY[m].fn(ctx)
        names += [f"{m}::{x}" for x in c]
        mats.append(np.asarray(A))
    return names, np.hstack(mats)


def _compare(hist, online):
    names, B = _batch_row_block(hist, online)
    eng = StreamEngine().fit_historical(hist)
    assert eng.cols == names, "column names/order differ from batch"
    S = np.vstack([eng.step(x) for x in online])
    assert S.shape == B.shape
    eq = (S == B) | (np.isnan(S) & np.isnan(B))
    return eq, S, B, names


@pytest.mark.parametrize("i0", [0, 137, 999, 4242, 7777])
def test_engine_parity_real(i0):
    st = load_store()
    rng = np.random.default_rng(i0)
    for i in rng.integers(0, st.n_series, 3):
        h, o, _ = st.series(int(i))
        eq, S, B, names = _compare(h, o)
        if not eq.all():
            bad = np.flatnonzero(~eq.all(axis=0))
            raise AssertionError(
                f"series {i}: {(~eq).sum()} / {eq.size} cells differ; "
                f"columns {[names[j] for j in bad[:6]]}")


def test_engine_parity_edge_cases():
    rng = np.random.default_rng(0)
    cases = {
        "short_online": (rng.standard_normal(3000), rng.standard_normal(10)),
        "long_online": (rng.standard_normal(5000), rng.standard_normal(999)),
        "constant_hist": (np.full(2000, 1.5), rng.standard_normal(120)),
        "tiny_var_hist": (rng.standard_normal(2000) * 1e-9, rng.standard_normal(80)),
        "heavy_tail": (rng.standard_t(3, 3000), rng.standard_t(3, 200)),
        "big_outlier": (rng.standard_normal(2000),
                        np.r_[rng.standard_normal(50), 1e9, rng.standard_normal(50)]),
        "break_tau0": (rng.standard_normal(2000), rng.standard_normal(150) * 4.0),
    }
    for name, (h, o) in cases.items():
        eq, S, B, names = _compare(h, o)
        bad = np.flatnonzero(~eq.all(axis=0))
        assert eq.all(), f"{name}: {(~eq).sum()} cells differ, cols {[names[j] for j in bad[:6]]}"


def test_manifest_stable():
    st = load_store()
    h, o, _ = st.series(0)
    a = StreamEngine().fit_historical(h).manifest()
    h2, o2, _ = st.series(500)
    b = StreamEngine().fit_historical(h2).manifest()
    assert a["feature_manifest_sha256"] == b["feature_manifest_sha256"]
    assert a["n_features"] == 500


def test_future_poison_cannot_change_an_emitted_row():
    """Emitting row t then rewriting every later observation must not change it."""
    rng = np.random.default_rng(1)
    h = rng.standard_normal(2500)
    o = rng.standard_normal(300)
    eng = StreamEngine().fit_historical(h)
    rows = [eng.step(x) for x in o[:120]]
    o2 = o.copy()
    o2[120:] = rng.standard_normal(180) * 1e6
    eng2 = StreamEngine().fit_historical(h)
    rows2 = [eng2.step(x) for x in o2[:120]]
    for t, (a, b) in enumerate(zip(rows, rows2)):
        assert np.array_equal(a, b, equal_nan=True), f"row {t} changed under future poisoning"
