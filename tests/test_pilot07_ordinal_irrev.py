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

import harness  # noqa: E402

from sbr.features.base import REGISTRY, check_prefix_invariance, load_all, make_ctx  # noqa: E402
from sbr.features.m15_ordinal_irrev import (  # noqa: E402
    CANDIDATE_COLS,
    CONTROL_COLS,
    MIN_COUNT,
    ordinal_codes,
    ordinal_irreversibility_features,
)


def _synthetic(seed: int = 0, n_hist: int = 512, n_online: int = 160):
    rng = np.random.default_rng(seed)
    h = rng.normal(0.0, 1.0, n_hist)
    o = rng.normal(0.0, 1.0, n_online)
    for t in range(2, n_online):
        o[t] += 0.35 * o[t - 1] - 0.25 * o[t - 2]
    o[n_online // 2 :] += 0.6 * np.sign(np.sin(np.arange(n_online // 2) * 0.7))
    return h, o


class CandidateMech(harness.StreamingMechanism):
    name = "pilot07_ordinal_irrev"
    cols = CANDIDATE_COLS

    def emit(self, hist, online):
        cand, _ = ordinal_irreversibility_features(make_ctx(hist, online))
        return cand


class EntropyMech(harness.StreamingMechanism):
    name = "pilot07_ordinal_entropy"
    cols = CONTROL_COLS

    def emit(self, hist, online):
        _, control = ordinal_irreversibility_features(make_ctx(hist, online))
        return control


def test_harness_verify_accepts_pilot07_mechanisms():
    series = [_synthetic(seed=s) + (f"s{s}",) for s in (0, 1)]
    for mech in (CandidateMech(), EntropyMech()):
        ok, msg = harness.verify(mech, series=series, cuts=(10, 37, 73, 111))
        assert ok, msg


def test_registered_modules_are_prefix_invariant_at_zero_tolerance():
    load_all()
    assert "m15_ordinal_irrev" in REGISTRY
    assert "m15_ordinal_entropy" in REGISTRY
    hist, online = _synthetic(seed=11, n_online=180)
    for module in ("m15_ordinal_irrev", "m15_ordinal_entropy"):
        ok, msg = check_prefix_invariance(module, hist, online, cuts=(10, 37, 73, 111), atol=0.0)
        assert ok, msg


def test_first_valid_nan_semantics():
    hist, online = _synthetic(seed=21, n_online=96)
    candidate, control = ordinal_irreversibility_features(make_ctx(hist, online))
    assert candidate.shape == (96, len(CANDIDATE_COLS))
    assert control.shape == (96, len(CONTROL_COLS))
    assert np.isnan(candidate[:15]).all()
    assert np.isnan(control[:15]).all()
    assert np.isfinite(candidate[15, [0, 1, 4, 5]]).all()
    assert np.isfinite(control[15, [0, 1]]).all()
    assert np.isnan(candidate[30, [2, 3, 6, 7]]).all()
    assert np.isnan(control[30, [2, 3]]).all()
    assert np.isfinite(candidate[31]).all()
    assert np.isfinite(control[31]).all()
    assert MIN_COUNT == 16


def test_future_online_mutation_does_not_change_past_rows():
    hist, online = _synthetic(seed=31, n_online=160)
    cut = 91
    full_candidate, full_control = ordinal_irreversibility_features(make_ctx(hist, online))
    mutated = online.copy()
    mutated[cut:] = mutated[cut:][::-1] * -3.0 + 2.0
    mut_candidate, mut_control = ordinal_irreversibility_features(make_ctx(hist, mutated))
    assert np.allclose(
        full_candidate[:cut], mut_candidate[:cut], rtol=0.0, atol=0.0, equal_nan=True
    )
    assert np.allclose(full_control[:cut], mut_control[:cut], rtol=0.0, atol=0.0, equal_nan=True)


def test_deterministic_replay():
    hist, online = _synthetic(seed=41, n_online=144)
    a_candidate, a_control = ordinal_irreversibility_features(make_ctx(hist, online))
    b_candidate, b_control = ordinal_irreversibility_features(make_ctx(hist, online))
    assert np.array_equal(a_candidate, b_candidate, equal_nan=True)
    assert np.array_equal(a_control, b_control, equal_nan=True)


def test_tie_handling_uses_m03_style_non_strict_comparisons():
    codes = ordinal_codes(np.array([1.0, 1.0, 1.0, 2.0, 1.0, 1.0]))
    assert codes.tolist() == [0, 0, 1, 4]
