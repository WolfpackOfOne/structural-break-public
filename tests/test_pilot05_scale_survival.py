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
from sbr.features.m13_scale_survival import (  # noqa: E402
    INDIV_COLS,
    Q01,
    Q05,
    SCALES,
    SUMMARY_COLS,
    scale_survival_features,
)


def _synthetic(seed: int = 0, n_hist: int = 320, n_online: int = 96):
    rng = np.random.default_rng(seed)
    hist = rng.normal(0.0, 1.0, n_hist)
    online = rng.normal(0.0, 1.0, n_online)
    online[n_online // 2 :] *= 1.8
    return hist, online


class ScaleSummaryMech(harness.StreamingMechanism):
    name = "pilot05_scale_summary"
    cols = SUMMARY_COLS

    def emit(self, hist, online):
        summary, _ = scale_survival_features(make_ctx(hist, online))
        return summary


class ScaleIndividualMech(harness.StreamingMechanism):
    name = "pilot05_scale_individual"
    cols = INDIV_COLS

    def emit(self, hist, online):
        _, indiv = scale_survival_features(make_ctx(hist, online))
        return indiv


def test_harness_verify_accepts_pilot05_mechanisms():
    series = [_synthetic(seed=s) + (f"s{s}",) for s in (0, 1)]
    for mech in (ScaleSummaryMech(), ScaleIndividualMech()):
        ok, msg = harness.verify(mech, series=series)
        assert ok, msg


def test_registered_modules_are_prefix_invariant_at_zero_tolerance():
    load_all()
    assert "m13_scale_surv" in REGISTRY
    assert "m13_scale_indiv" in REGISTRY
    hist, online = _synthetic(seed=11, n_online=120)
    for module in ("m13_scale_surv", "m13_scale_indiv"):
        ok, msg = check_prefix_invariance(module, hist, online, cuts=(3, 10, 37, 73), atol=0.0)
        assert ok, msg


def test_first_valid_nan_and_summary_semantics():
    hist, online = _synthetic(seed=21, n_online=40)
    summary, indiv = scale_survival_features(make_ctx(hist, online))
    assert list(summary.shape) == [40, len(SUMMARY_COLS)]
    assert list(indiv.shape) == [40, len(SCALES)]

    assert np.isfinite(indiv[0, 0])
    assert np.isnan(indiv[0, 1:]).all()
    assert np.isfinite(indiv[1, 1])
    assert np.isnan(indiv[30, -1])
    assert np.isfinite(indiv[31, -1])

    assert np.isnan(summary[0, SUMMARY_COLS.index("slope_logscale")])
    assert np.isfinite(summary[1, SUMMARY_COLS.index("slope_logscale")])
    assert np.all(summary[:, 0] >= 0)
    assert np.all(summary[:, 1] >= 0)
    assert np.all(summary[:, 2] >= 0)
    assert np.all(summary[:, 3] >= 0)
    assert Q05 < Q01


def test_future_online_mutation_does_not_change_past_rows():
    hist, online = _synthetic(seed=31, n_online=96)
    cut = 37
    full_summary, full_indiv = scale_survival_features(make_ctx(hist, online))
    mutated = online.copy()
    mutated[cut:] = mutated[cut:][::-1] * -7.0 + 3.0
    mut_summary, mut_indiv = scale_survival_features(make_ctx(hist, mutated))
    assert np.allclose(full_summary[:cut], mut_summary[:cut], rtol=0.0, atol=0.0, equal_nan=True)
    assert np.allclose(full_indiv[:cut], mut_indiv[:cut], rtol=0.0, atol=0.0, equal_nan=True)


def test_deterministic_replay():
    hist, online = _synthetic(seed=41, n_online=88)
    a_summary, a_indiv = scale_survival_features(make_ctx(hist, online))
    b_summary, b_indiv = scale_survival_features(make_ctx(hist, online))
    assert np.array_equal(a_summary, b_summary, equal_nan=True)
    assert np.array_equal(a_indiv, b_indiv, equal_nan=True)
