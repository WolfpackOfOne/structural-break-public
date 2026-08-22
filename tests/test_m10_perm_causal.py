"""Causal certification gates for m10_perm.

These are the gates PROTOCOL.md §3 makes non-negotiable, wired into the suite so
they run without a feature store or any competition data. FAILED_EXPERIMENTS N6
records a causality bug that produced plausible-looking features and would have
silently inflated every downstream number; N7 records a garbage column that only
a per-column distribution audit caught. Both classes are covered here.
"""
from __future__ import annotations

import numpy as np
import pytest

from sbr.features.base import REGISTRY, check_prefix_invariance, load_all, make_ctx

MOD = "m10_perm"

PROFILES = [(300, 120), (500, 300), (800, 451), (1200, 800), (150, 40), (1000, 17)]
KINDS = ("null", "mean_shift", "var_shift", "burst")


@pytest.fixture(scope="module", autouse=True)
def _registry():
    load_all()


def _series(rng, nh, no, kind):
    hist = rng.normal(size=nh)
    on = rng.normal(size=no)
    tau = no // 2
    if kind == "mean_shift":
        on[tau:] += 1.1
    elif kind == "var_shift":
        on[tau:] *= 2.3
    elif kind == "burst":
        on[tau:tau + 3] += 7.0
    return hist, on


@pytest.mark.parametrize("nh,no", PROFILES)
@pytest.mark.parametrize("kind", KINDS)
def test_bitwise_prefix_invariant(nh, no, kind):
    """Row t may depend only on online[:t+1]. Checked bitwise, atol=0.0."""
    rng = np.random.default_rng(hash((nh, no, kind)) % 2**32)
    hist, on = _series(rng, nh, no, kind)
    ok, msg = check_prefix_invariance(MOD, hist, on, cuts=(1, 2, 3, 10, 37, 64, 113))
    assert ok, msg


def test_no_column_explodes_or_is_dead():
    """Per-column audit: no exploded scale, no constant column.

    The dead-column half of this test is not hypothetical. The module's first
    design emitted a mean-vs-robust-mean contrast that is identically zero for
    any input, because (x-mu)/sd and (x-med)/mad are affine images of each other
    and null-standardising removes any affine map exactly.
    """
    rng = np.random.default_rng(3)
    mats = []
    for nh, no in PROFILES:
        for kind in KINDS:
            hist, on = _series(rng, nh, no, kind)
            names, m = REGISTRY[MOD].fn(make_ctx(hist, on))
            mats.append(m)
    a = np.concatenate(mats, axis=0).astype(np.float64)
    assert a.shape[1] == len(names)
    for j, nm in enumerate(names):
        fin = a[:, j][np.isfinite(a[:, j])]
        assert fin.size > 0, f"{nm}: all non-finite"
        assert np.abs(fin).max() < 1e4, f"{nm}: exploded, max={np.abs(fin).max():.3g}"
        assert fin.std() > 1e-9, f"{nm}: constant column, carries no information"


def test_affine_robust_contrast_would_be_dead():
    """Guards the negative result, so nobody re-adds a mean-vs-rmean contrast.

    Null-standardising two affine images of the same statistic yields the
    identical vector. Any future 'robust vs plain' contrast must use a
    non-affine robust statistic (a median, a trimmed mean) or it is zeros.
    """
    from numpy.lib.stride_tricks import sliding_window_view as sw

    rng = np.random.default_rng(0)
    x = rng.standard_t(3, size=4000)
    w = 32
    mu, sd = x.mean(), x.std()
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    a = sw((x - mu) / sd, w).mean(axis=1)
    b = sw((x - med) / mad, w).mean(axis=1)
    za = (a - a.mean()) / a.std()
    zb = (b - b.mean()) / b.std()
    assert np.abs(za - zb).max() < 1e-10, "affine contrast unexpectedly non-degenerate"

    m = np.median(sw((x - mu) / sd, w), axis=1)
    zm = (m - m.mean()) / m.std()
    assert np.abs(za - zm).max() > 0.5, "median contrast must survive standardisation"


def test_separates_transient_from_permanent():
    """The mechanism claim, made falsifiable.

    A burst and a sustained shift carrying the SAME cumulative displacement are
    indistinguishable to a window-mean bank by construction. If this module
    cannot separate them where ground truth is known, it cannot work on real
    data. This is a mechanism diagnostic and NOT model selection: it produces no
    TS-AUC and ranks no candidate (PROTOCOL.md §1 forbids selecting on synthetic
    performance).
    """
    rng = np.random.default_rng(11)
    nh, no, tau, ns = 800, 400, 200, 60
    got = {}
    for kind in ("shift", "burst"):
        rows = []
        for _ in range(ns):
            hist = rng.normal(size=nh)
            on = rng.normal(size=no)
            if kind == "shift":
                on[tau:] += 0.8
            else:
                on[tau:tau + 5] += 0.8 * (no - tau) / 5.0
            names, m = REGISTRY[MOD].fn(make_ctx(hist, on))
            rows.append(m[tau + 60])
        got[kind] = np.array(rows, dtype=np.float64)

    strong = 0
    for j in range(len(names)):
        a, b = got["shift"][:, j], got["burst"][:, j]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 20 or len(b) < 20:
            continue
        d = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-12)
        if abs(d) > 1.0:
            strong += 1
    assert strong >= 6, f"only {strong} columns separate transient from permanent"
