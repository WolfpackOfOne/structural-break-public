"""Causal certification and exactness gates for m11_focus.

Runs without a feature store or any competition data.
"""
from __future__ import annotations

import numpy as np
import pytest

from sbr.features.base import REGISTRY, check_prefix_invariance, load_all, make_ctx
from sbr.features.m11_focus import LMAX, MIN_PRE, _focus

MOD = "m11_focus"
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
    rng = np.random.default_rng(hash((nh, no, kind, "f")) % 2**32)
    hist, on = _series(rng, nh, no, kind)
    ok, msg = check_prefix_invariance(MOD, hist, on, cuts=(1, 2, 3, 10, 37, 64, 113))
    assert ok, msg


@pytest.mark.parametrize("trial", range(5))
def test_vectorised_maximisation_is_exact(trial):
    """The lag loop must equal brute force. It is an optimisation, not an approximation."""
    rng = np.random.default_rng(100 + trial)
    n = int(rng.integers(20, 160))
    x = rng.normal(size=n)
    x[n // 2:] += 1.0
    hg, _, ag, _ = _focus(x)
    s = np.concatenate([[0.0], np.cumsum(x)])
    bh = np.zeros(n)
    ba = np.zeros(n)
    for t in range(n):
        best_h = best_a = 0.0
        for lag in range(1, min(LMAX, t + 1) + 1):
            d = s[t + 1] - s[t + 1 - lag]
            best_h = max(best_h, d * d / lag)
            n1 = t + 1 - lag
            if n1 >= MIN_PRE:
                best_a = max(best_a, n1 * lag / (n1 + lag) * (d / lag - s[t + 1 - lag] / n1) ** 2)
        bh[t], ba[t] = best_h, best_a
    assert np.abs(hg - np.sqrt(bh)).max() < 1e-9
    assert np.abs(ag - np.sqrt(ba)).max() < 1e-9


def test_no_column_explodes_or_is_dead():
    rng = np.random.default_rng(5)
    mats = []
    for nh, no in PROFILES:
        for kind in KINDS:
            hist, on = _series(rng, nh, no, kind)
            names, m = REGISTRY[MOD].fn(make_ctx(hist, on))
            mats.append(m)
    a = np.concatenate(mats, axis=0).astype(np.float64)
    for j, nm in enumerate(names):
        fin = a[:, j][np.isfinite(a[:, j])]
        assert fin.size > 0, f"{nm}: all non-finite"
        assert np.abs(fin).max() < 1e4, f"{nm}: exploded"
        assert fin.std() > 1e-9, f"{nm}: constant column"


def test_adaptive_baseline_rejects_a_pure_offset():
    """The module's whole hypothesis, made falsifiable.

    A series whose online segment merely sits somewhere history did not predict
    looks broken to a historical-baseline detector. If the adaptive arm cannot
    tell that apart from a real break, the module adds nothing over m01_seq's
    existing max-GLR and should not be run. Mechanism diagnostic only: no TS-AUC,
    no ranking (PROTOCOL.md §1).
    """
    rng = np.random.default_rng(21)
    nh, no, tau, ns = 800, 400, 200, 40
    got = {}
    for arm in ("offset", "break"):
        rows = []
        for _ in range(ns):
            h = rng.normal(size=nh)
            o = rng.normal(size=no)
            if arm == "offset":
                o += 0.9
            else:
                o[tau:] += 0.9
            names, m = REGISTRY[MOD].fn(make_ctx(h, o))
            rows.append(m[tau + 120])
        got[arm] = np.array(rows, dtype=np.float64)
    idx = {n: j for j, n in enumerate(names)}

    # the historical-baseline arm fires on a pure offset at least as hard as on
    # a real break -- this is the deficiency the module exists to address
    assert got["offset"][:, idx["hg_lv"]].mean() > got["break"][:, idx["hg_lv"]].mean()
    # the adaptive arm must not
    assert got["offset"][:, idx["ag_lv"]].mean() < got["break"][:, idx["ag_lv"]].mean()
    # and the contrast must separate the two arms strongly
    a = got["offset"][:, idx["dg_lv"]]
    b = got["break"][:, idx["dg_lv"]]
    d = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-12)
    assert d > 2.0, f"dg_lv separation only d={d:.2f}"
