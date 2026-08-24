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
from sbr.features.m14_spectral_impulse import (
    CONTRAST_COLS,
    ENERGY_COLS,
    MIN_L,
    SEG,
    spectral_impulse_features,
)


def _synthetic(seed: int = 0, n_hist: int = 512, n_online: int = 160):
    rng = np.random.default_rng(seed)
    h_t = np.arange(n_hist)
    o_t = np.arange(n_online)
    hist = rng.normal(0.0, 1.0, n_hist) + 0.2 * np.sin(2 * np.pi * 0.125 * h_t)
    online = rng.normal(0.0, 1.0, n_online) + 0.2 * np.sin(2 * np.pi * 0.125 * o_t)
    online[n_online // 2 :] += 0.8 * np.sin(2 * np.pi * 0.25 * o_t[n_online // 2 :])
    return hist, online


class ContrastMech(harness.StreamingMechanism):
    name = "pilot06_spectral_contrast"
    cols = CONTRAST_COLS

    def emit(self, hist, online):
        contrast, _ = spectral_impulse_features(make_ctx(hist, online))
        return contrast


class EnergyMech(harness.StreamingMechanism):
    name = "pilot06_spectral_energy"
    cols = ENERGY_COLS

    def emit(self, hist, online):
        _, energy = spectral_impulse_features(make_ctx(hist, online))
        return energy


def test_harness_verify_accepts_pilot06_mechanisms():
    series = [_synthetic(seed=s) + (f"s{s}",) for s in (0, 1)]
    for mech in (ContrastMech(), EnergyMech()):
        ok, msg = harness.verify(mech, series=series, cuts=(10, 37, 73, 111))
        assert ok, msg


def test_registered_modules_are_prefix_invariant_at_zero_tolerance():
    load_all()
    assert "m14_spectral_impulse" in REGISTRY
    assert "m14_spectral_energy" in REGISTRY
    hist, online = _synthetic(seed=11, n_online=180)
    for module in ("m14_spectral_impulse", "m14_spectral_energy"):
        ok, msg = check_prefix_invariance(module, hist, online, cuts=(10, 37, 73, 111), atol=0.0)
        assert ok, msg


def test_first_valid_nan_semantics():
    hist, online = _synthetic(seed=21, n_online=96)
    contrast, energy = spectral_impulse_features(make_ctx(hist, online))
    assert contrast.shape == (96, len(CONTRAST_COLS))
    assert energy.shape == (96, len(ENERGY_COLS))
    assert np.isnan(contrast[:46]).all()
    assert np.isnan(energy[:46]).all()
    assert np.isfinite(contrast[46:]).any()
    assert np.isfinite(energy[46:]).any()
    assert SEG == 32
    assert MIN_L == 16


def test_future_online_mutation_does_not_change_past_rows():
    hist, online = _synthetic(seed=31, n_online=160)
    cut = 91
    full_contrast, full_energy = spectral_impulse_features(make_ctx(hist, online))
    mutated = online.copy()
    mutated[cut:] = mutated[cut:][::-1] * -4.0 + 1.5
    mut_contrast, mut_energy = spectral_impulse_features(make_ctx(hist, mutated))
    assert np.allclose(full_contrast[:cut], mut_contrast[:cut], rtol=0.0, atol=0.0, equal_nan=True)
    assert np.allclose(full_energy[:cut], mut_energy[:cut], rtol=0.0, atol=0.0, equal_nan=True)


def test_deterministic_replay():
    hist, online = _synthetic(seed=41, n_online=144)
    a_contrast, a_energy = spectral_impulse_features(make_ctx(hist, online))
    b_contrast, b_energy = spectral_impulse_features(make_ctx(hist, online))
    assert np.array_equal(a_contrast, b_contrast, equal_nan=True)
    assert np.array_equal(a_energy, b_energy, equal_nan=True)
