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
from sbr.features.m18_weighted_ctm import (  # noqa: E402
    H2_EVALUE_COLS,
    UNWEIGHTED_COLS,
    WEIGHTED_COLS,
    benign_tail_weight,
    h2_log_mean_evalue,
    unweighted_ctm_features,
    weighted_ctm_features,
)


def _synthetic(seed: int = 0, n_hist: int = 720, n_online: int = 180):
    rng = np.random.default_rng(seed)
    h = rng.standard_t(df=5, size=n_hist)
    o = rng.standard_t(df=5, size=n_online)
    o[70:95] += 2.0
    o[120:128] += rng.normal(0.0, 5.0, 8)
    return h, o


class WeightedMech(harness.StreamingMechanism):
    name = "pilot08_weighted_ctm"
    cols = WEIGHTED_COLS

    def emit(self, hist, online):
        return weighted_ctm_features(make_ctx(hist, online))


class UnweightedMech(harness.StreamingMechanism):
    name = "pilot08_unweighted_ctm"
    cols = UNWEIGHTED_COLS

    def emit(self, hist, online):
        return unweighted_ctm_features(make_ctx(hist, online))


def test_harness_verify_accepts_pilot08_mechanisms():
    series = [_synthetic(seed=s) + (f"s{s}",) for s in (0, 1)]
    for mech in (WeightedMech(), UnweightedMech()):
        ok, msg = harness.verify(mech, series=series, cuts=(10, 37, 73, 111))
        assert ok, msg


def test_registered_modules_are_prefix_invariant_at_zero_tolerance():
    load_all()
    assert "m18_wctm" in REGISTRY
    assert "m18_uctm" in REGISTRY
    hist, online = _synthetic(seed=11, n_online=180)
    for module in ("m18_wctm", "m18_uctm"):
        ok, msg = check_prefix_invariance(module, hist, online, cuts=(10, 37, 73, 111), atol=0.0)
        assert ok, msg


def test_shapes_columns_and_finite_outputs():
    hist, online = _synthetic(seed=21, n_online=160)
    weighted = weighted_ctm_features(make_ctx(hist, online))
    unweighted = unweighted_ctm_features(make_ctx(hist, online))
    assert weighted.shape == (160, len(WEIGHTED_COLS))
    assert unweighted.shape == (160, len(UNWEIGHTED_COLS))
    assert np.isfinite(weighted).all()
    assert np.isfinite(unweighted).all()
    assert np.all(weighted[:, 1] >= weighted[:, 0])
    assert np.all(weighted[:, 3] >= weighted[:, 2])
    assert np.all(weighted[:, 7] >= weighted[:, 6])


def test_tail_weight_is_predictable_and_excludes_current_row():
    cu_h = np.r_[np.linspace(-0.49, -0.46, 20), np.linspace(-0.2, 0.2, 180)]
    cu_o = np.zeros(80)
    cu_o[0] = 0.49
    cu_o[1:40] = 0.49
    omega = benign_tail_weight(cu_h, cu_o)
    assert omega[0] == 1.0
    assert omega[1] < 1.0
    changed_current = cu_o.copy()
    changed_current[0] = 0.0
    omega2 = benign_tail_weight(cu_h, changed_current)
    assert omega2[0] == omega[0]
    assert omega2[1] > omega[1]


def test_future_online_mutation_does_not_change_past_rows():
    hist, online = _synthetic(seed=31, n_online=180)
    cut = 111
    full_w = weighted_ctm_features(make_ctx(hist, online))
    full_u = unweighted_ctm_features(make_ctx(hist, online))
    mutated = online.copy()
    mutated[cut:] = mutated[cut:][::-1] * -2.0 + 4.0
    mut_w = weighted_ctm_features(make_ctx(hist, mutated))
    mut_u = unweighted_ctm_features(make_ctx(hist, mutated))
    assert np.allclose(full_w[:cut], mut_w[:cut], rtol=0.0, atol=0.0, equal_nan=True)
    assert np.allclose(full_u[:cut], mut_u[:cut], rtol=0.0, atol=0.0, equal_nan=True)


def test_deterministic_replay():
    hist, online = _synthetic(seed=41, n_online=160)
    a_w = weighted_ctm_features(make_ctx(hist, online))
    a_u = unweighted_ctm_features(make_ctx(hist, online))
    b_w = weighted_ctm_features(make_ctx(hist, online))
    b_u = unweighted_ctm_features(make_ctx(hist, online))
    assert np.array_equal(a_w, b_w, equal_nan=True)
    assert np.array_equal(a_u, b_u, equal_nan=True)


def test_h2_log_mean_evalue_uses_exact_five_columns_shape():
    assert H2_EVALUE_COLS == [
        "ev_tail_mix", "ev_tail_ad", "ev_disp_mix", "ev_disp_ad", "ev_pow_mix",
    ]
    x = np.array([[0.0, 1.0, -1.0, 0.5, -0.5], [2.0, 2.0, 2.0, 2.0, 2.0]], dtype=np.float64)
    got = h2_log_mean_evalue(x)
    expected0 = np.log(np.exp(x[0]).mean())
    assert np.allclose(got[0], expected0)
    assert got[1] == np.float32(2.0)
