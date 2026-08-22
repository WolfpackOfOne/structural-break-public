"""Causal certification, exactness and internal-control gates for m12_deplr.

Runs without a feature store or any competition data.
"""
from __future__ import annotations

import numpy as np
import pytest

from sbr.features.base import REGISTRY, check_prefix_invariance, load_all, make_ctx
from sbr.features.m12_deplr import LMAX, MIN_PRE, MIN_SS, _dep_scan

MOD = "m12_deplr"
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


def _ar(rng, n, phi, s=1.0):
    e = rng.normal(size=n) * s
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


@pytest.mark.parametrize("nh,no", PROFILES)
@pytest.mark.parametrize("kind", KINDS)
def test_bitwise_prefix_invariant(nh, no, kind):
    rng = np.random.default_rng(hash((nh, no, kind, "d")) % 2**32)
    hist, on = _series(rng, nh, no, kind)
    ok, msg = check_prefix_invariance(MOD, hist, on, cuts=(1, 2, 3, 10, 37, 64, 113))
    assert ok, msg


@pytest.mark.parametrize("trial", range(4))
def test_scan_is_exact(trial):
    """The vectorised changepoint scan must equal brute force, not approximate it."""
    rng = np.random.default_rng(200 + trial)
    n = int(rng.integers(40, 140))
    e = rng.normal(size=n)
    e[n // 2:] *= 1.5
    d, _, p = _dep_scan(e, 1)
    prev = np.concatenate([[0.0], e[:-1]])
    s = np.concatenate([[0.0], np.cumsum(e * prev)])
    x = np.concatenate([[0.0], np.cumsum(prev * prev)])
    bd = np.zeros(n)
    bp = np.zeros(n)
    for t in range(n):
        for lag in range(1, min(LMAX, t + 1) + 1):
            a = t + 1 - lag
            if a < MIN_PRE:
                continue
            w = a * lag / (a + lag)
            xxp, xxq = x[a], x[t + 1] - x[a]
            if xxp > MIN_SS and xxq > MIN_SS:
                bd[t] = max(bd[t], w * ((s[t + 1] - s[a]) / xxq - s[a] / xxp) ** 2)
            bp[t] = max(bp[t], w * ((s[t + 1] - s[a]) / lag - s[a] / a) ** 2)
    assert np.abs(d - np.sqrt(bd)).max() < 1e-9
    assert np.abs(p - np.sqrt(bp)).max() < 1e-9


def test_no_column_explodes_or_is_dead():
    rng = np.random.default_rng(9)
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


def test_coefficient_arm_ignores_a_pure_variance_change():
    """The mechanism, and the internal control, in one test.

    A variance change with a CONSTANT AR coefficient must move the lag-product
    arm (`plr`, an autocovariance -- the kind of statistic m04's `e_acf1` is) and
    must NOT move the coefficient arm (`dlr`), because the variance cancels in
    the ratio. If `dlr` also fires here it is not measuring what it claims and
    the module adds nothing over the incumbent.

    The assertion is on the SELECTIVITY RATIO dlr(dependence)/dlr(variance)
    rather than on an absolute effect size. A raw |d| near zero is unstable at
    small n -- an earlier draft asserted |d| < 0.6 at ns=25 and saw 0.05 on one
    draw and 0.66 on another, which is noise around a true value measured at
    0.15-0.23 with ns=120 across three seeds. The ratio was 13.6x, 14.3x and
    20.2x on those same seeds, so a 4x floor here carries real margin.

    Mechanism diagnostic only: no TS-AUC, ranks nothing (PROTOCOL.md §1).
    """
    rng = np.random.default_rng(31)
    ns, nh, no, tau = 60, 700, 450, 225

    def arm_means(kind):
        out = {"dlr": [], "plr": []}
        for _ in range(ns):
            phi = float(rng.uniform(-0.4, 0.5))
            h = _ar(rng, nh, phi)
            o = _ar(rng, no, phi)
            if kind == "var":
                o[tau:] = _ar(rng, no - tau, phi, 2.2)
            elif kind == "dep":
                o[tau:] = _ar(rng, no - tau, float(np.clip(phi + 0.55, -0.9, 0.9)), 1.0)
            names, m = REGISTRY[MOD].fn(make_ctx(h, o))
            i = {n: j for j, n in enumerate(names)}
            out["dlr"].append(np.nanmean(m[260:, i["dlr_rz"]]))
            out["plr"].append(np.nanmean(m[260:, i["plr_rz"]]))
        return out

    base, var, dep = arm_means("none"), arm_means("var"), arm_means("dep")

    def d(a, b):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        return abs((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-12))

    d_dlr_var = d(var["dlr"], base["dlr"])
    d_plr_var = d(var["plr"], base["plr"])
    d_dlr_dep = d(dep["dlr"], base["dlr"])

    # the control arm IS fooled by a pure variance change -- that is the point
    assert d_plr_var > 1.5, f"plr should be fooled by a variance change, d={d_plr_var:.2f}"
    # the hypothesis arm responds to dependence
    assert d_dlr_dep > 1.5, f"dlr should fire on a dependence change, d={d_dlr_dep:.2f}"
    # and it is far more selective for dependence than for variance
    ratio = d_dlr_dep / max(d_dlr_var, 1e-9)
    assert ratio > 4.0, (
        f"dlr selectivity only {ratio:.1f}x "
        f"(dependence d={d_dlr_dep:.2f}, variance d={d_dlr_var:.2f})")
    # and it is more selective for dependence than the control arm is
    assert ratio > d_plr_var / max(d(dep["plr"], base["plr"]), 1e-9)
