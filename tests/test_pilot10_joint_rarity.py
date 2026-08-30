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
from sbr.features.m16_joint_rarity import (
    DWELL_COLS,
    JOINT_COLS,
    MIN_ENDPOINTS,
    WINDOWS,
    endpoint_states,
    joint_rarity_features,
)


def _synthetic(seed: int = 0, n_hist: int = 640, n_online: int = 180):
    rng = np.random.default_rng(seed)
    h = rng.normal(0.0, 1.0, n_hist)
    o = rng.normal(0.0, 1.0, n_online)
    for t in range(2, n_hist):
        h[t] += 0.25 * h[t - 1] - 0.12 * h[t - 2]
    for t in range(2, n_online):
        o[t] += 0.25 * o[t - 1] - 0.12 * o[t - 2]
    o[n_online // 2 : n_online // 2 + 35] += 2.0
    return h, o


class JointMech(harness.StreamingMechanism):
    name = "pilot10_joint_rarity"
    cols = JOINT_COLS

    def emit(self, hist, online):
        candidate, _ = joint_rarity_features(make_ctx(hist, online))
        return candidate


class DwellMech(harness.StreamingMechanism):
    name = "pilot10_dwell_rarity"
    cols = DWELL_COLS

    def emit(self, hist, online):
        _, control = joint_rarity_features(make_ctx(hist, online))
        return control


def test_harness_verify_accepts_pilot10_mechanisms():
    series = [_synthetic(seed=s) + (f"s{s}",) for s in (0, 1)]
    for mech in (JointMech(), DwellMech()):
        ok, msg = harness.verify(mech, series=series, cuts=(10, 37, 73, 111))
        assert ok, msg


def test_registered_modules_are_prefix_invariant_at_zero_tolerance():
    load_all()
    assert "m16_joint_rarity" in REGISTRY
    assert "m16_dwell_rarity" in REGISTRY
    hist, online = _synthetic(seed=11, n_online=180)
    for module in ("m16_joint_rarity", "m16_dwell_rarity"):
        ok, msg = check_prefix_invariance(module, hist, online, cuts=(10, 37, 73, 111), atol=0.0)
        assert ok, msg


def test_first_valid_nan_semantics():
    hist, online = _synthetic(seed=21, n_online=160)
    candidate, control = joint_rarity_features(make_ctx(hist, online))
    assert candidate.shape == (160, len(JOINT_COLS))
    assert control.shape == (160, len(DWELL_COLS))
    assert np.isnan(candidate[:31]).all()
    assert np.isnan(control[:31]).all()
    assert np.isfinite(candidate[31, [0, 1]]).all()
    assert np.isfinite(control[31, [0, 1]]).all()
    assert np.isnan(candidate[62, [2, 3]]).all()
    assert np.isnan(control[62, [2, 3]]).all()
    assert np.isfinite(candidate[63, [0, 1, 2, 3]]).all()
    assert np.isfinite(control[63, [0, 1, 2, 3]]).all()
    assert np.isnan(candidate[126, [4, 5]]).all()
    assert np.isnan(control[126, [4, 5]]).all()
    assert np.isfinite(candidate[127]).all()
    assert np.isfinite(control[127]).all()
    assert WINDOWS == (32, 64, 128)
    assert MIN_ENDPOINTS == 16


def test_future_online_mutation_does_not_change_past_rows():
    hist, online = _synthetic(seed=31, n_online=180)
    cut = 111
    full_candidate, full_control = joint_rarity_features(make_ctx(hist, online))
    mutated = online.copy()
    mutated[cut:] = mutated[cut:][::-1] * -2.5 + 3.0
    mut_candidate, mut_control = joint_rarity_features(make_ctx(hist, mutated))
    assert np.allclose(full_candidate[:cut], mut_candidate[:cut], rtol=0.0, atol=0.0, equal_nan=True)
    assert np.allclose(full_control[:cut], mut_control[:cut], rtol=0.0, atol=0.0, equal_nan=True)


def test_deterministic_replay():
    hist, online = _synthetic(seed=41, n_online=160)
    a_candidate, a_control = joint_rarity_features(make_ctx(hist, online))
    b_candidate, b_control = joint_rarity_features(make_ctx(hist, online))
    assert np.array_equal(a_candidate, b_candidate, equal_nan=True)
    assert np.array_equal(a_control, b_control, equal_nan=True)


def test_joint_surprise_dominates_dwell_surprise_for_same_state():
    hist, online = _synthetic(seed=51, n_online=180)
    candidate, control = joint_rarity_features(make_ctx(hist, online))
    ok = np.isfinite(candidate) & np.isfinite(control)
    assert ok.any()
    assert np.all(candidate[ok] + 1e-6 >= control[ok])


def test_endpoint_state_peak_resets_between_runs():
    stat = np.array([0.0, 3.0, 4.0, 0.0, -5.0, -6.0, -1.0])
    run, peak = endpoint_states(stat, median=0.0, band=2.0)
    assert run.tolist() == [0, 1, 2, 0, 1, 2, 0]
    assert peak.tolist() == [0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0]
