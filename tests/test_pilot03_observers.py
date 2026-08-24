from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOVEL = os.path.join(REPO, "research", "scripts", "novel_streams")
SCRIPTS = os.path.join(REPO, "research", "scripts")
SRC = os.path.join(REPO, "src")

sys.path.insert(0, NOVEL)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, SRC)

import harness
from sbr.features.base import REGISTRY, check_prefix_invariance, load_all, make_ctx
from sbr.features.m17_observers import (
    DMD_COLS,
    DMD_DELAY,
    DMD_RANK,
    KALMAN_COLS,
    fit_hankel_dmd_observer,
    fit_kalman_observer,
    hankel_dmd_features,
    kalman_nis_features,
    observer_features,
)


def _synthetic(seed: int = 0, n_hist: int = 720, n_online: int = 180):
    rng = np.random.default_rng(seed)
    h = rng.normal(0.0, 1.0, n_hist)
    o = rng.normal(0.0, 1.0, n_online)
    for t in range(2, n_hist):
        h[t] += 0.45 * h[t - 1] - 0.18 * h[t - 2] + 0.08 * np.sin(t / 11.0)
    warm = np.r_[h[-2:], o]
    for t in range(2, len(warm)):
        warm[t] += 0.45 * warm[t - 1] - 0.18 * warm[t - 2] + 0.08 * np.sin((n_hist + t) / 11.0)
    o = warm[2:]
    o[n_online // 2 : n_online // 2 + 30] += 1.7
    return h, o


class KalmanMech(harness.StreamingMechanism):
    name = "pilot03_kalman_nis"
    cols = KALMAN_COLS

    def emit(self, hist, online):
        return kalman_nis_features(make_ctx(hist, online))


class DmdMech(harness.StreamingMechanism):
    name = "pilot03_hankel_dmd"
    cols = DMD_COLS

    def emit(self, hist, online):
        return hankel_dmd_features(make_ctx(hist, online))


def test_harness_verify_accepts_pilot03_observer_mechanisms():
    series = [_synthetic(seed=s) + (f"s{s}",) for s in (0, 1)]
    for mech in (KalmanMech(), DmdMech()):
        ok, msg = harness.verify(mech, series=series, cuts=(10, 37, 73, 111))
        assert ok, msg


def test_registered_modules_are_prefix_invariant_at_zero_tolerance():
    load_all()
    assert "m17_kalman_nis" in REGISTRY
    assert "m17_hankel_dmd" in REGISTRY
    hist, online = _synthetic(seed=11, n_online=180)
    for module in ("m17_kalman_nis", "m17_hankel_dmd"):
        ok, msg = check_prefix_invariance(module, hist, online, cuts=(10, 37, 73, 111), atol=0.0)
        assert ok, msg


def test_shapes_columns_and_first_valid_semantics():
    hist, online = _synthetic(seed=21, n_online=160)
    kalman, dmd = observer_features(make_ctx(hist, online))
    assert kalman.shape == (160, len(KALMAN_COLS))
    assert dmd.shape == (160, len(DMD_COLS))
    assert np.isfinite(kalman[:, [0, 1]]).all()
    assert np.isnan(kalman[:31, [2, 4]]).all()
    assert np.isfinite(kalman[31, [2, 4]]).all()
    assert np.isnan(kalman[:63, [3, 5]]).all()
    assert np.isfinite(kalman[63, [3, 5]]).all()
    assert np.isfinite(dmd[0]).all()
    assert DMD_DELAY == 16
    assert DMD_RANK == 4


def test_future_online_mutation_does_not_change_past_rows():
    hist, online = _synthetic(seed=31, n_online=180)
    cut = 111
    full_kalman, full_dmd = observer_features(make_ctx(hist, online))
    mutated = online.copy()
    mutated[cut:] = mutated[cut:][::-1] * -3.0 + 2.0
    mut_kalman, mut_dmd = observer_features(make_ctx(hist, mutated))
    assert np.allclose(full_kalman[:cut], mut_kalman[:cut], rtol=0.0, atol=0.0, equal_nan=True)
    assert np.allclose(full_dmd[:cut], mut_dmd[:cut], rtol=0.0, atol=0.0, equal_nan=True)


def test_deterministic_replay():
    hist, online = _synthetic(seed=41, n_online=160)
    a_kalman, a_dmd = observer_features(make_ctx(hist, online))
    b_kalman, b_dmd = observer_features(make_ctx(hist, online))
    assert np.array_equal(a_kalman, b_kalman, equal_nan=True)
    assert np.array_equal(a_dmd, b_dmd, equal_nan=True)


def test_fitted_observer_state_uses_history_only_inputs():
    hist, online = _synthetic(seed=51, n_online=160)
    k1 = fit_kalman_observer(hist)
    d1 = fit_hankel_dmd_observer(hist)
    mutated_online = online[::-1] * 5.0 - 1.0
    k2 = fit_kalman_observer(hist)
    d2 = fit_hankel_dmd_observer(hist)
    assert mutated_online.shape == online.shape
    assert np.allclose(k1.phi, k2.phi, rtol=0.0, atol=0.0)
    assert k1.q == k2.q
    assert k1.r == k2.r
    assert np.allclose(k1.x0, k2.x0, rtol=0.0, atol=0.0)
    assert np.allclose(d1.operator, d2.operator, rtol=0.0, atol=0.0)
    assert np.allclose(d1.subspace, d2.subspace, rtol=0.0, atol=0.0)
    assert d1.nulls == d2.nulls
