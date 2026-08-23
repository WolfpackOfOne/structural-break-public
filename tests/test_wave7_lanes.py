"""WAVE-7 gates.  All synthetic -- these run with no competition data present.

They exist to make three claims mechanically checkable BEFORE any pilot scores:

    1. a teacher quantity cannot reach a student input;
    2. lane B's routing is a function of the CURRENT online index and nothing
       else, so a prefix routes identically however the series continues;
    3. the bucket report used to gate every candidate is the exact
       decomposition of the official metric, not an approximation of it.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

import wave7_lib as W  # noqa: E402
from sbr.metric import ts_auc_flat  # noqa: E402


def _series(rng, n_hist=400, n_pre=120, n_post=250, shift=1.2, scale=1.0):
    hist = rng.normal(0, 1, n_hist)
    online = np.r_[rng.normal(0, 1, n_pre), rng.normal(shift, scale, n_post)]
    return hist, online, n_pre


# --------------------------------------------------------------- leak gates
def test_teacher_columns_are_refused_by_the_student_gate():
    W.assert_no_teacher_in_features(["m00_core::mean", "m01_seq::glr"])
    with pytest.raises(W.TeacherLeak):
        W.assert_no_teacher_in_features(["m00_core::mean", W.TEACHER_PREFIX + "evidence"])


def test_causal_name_gate_refuses_a_tau_column():
    with pytest.raises(W.TeacherLeak):
        W.assert_causal_names(["w7oracle::tau_frac"])


def test_the_gate_itself_catches_a_planted_leak():
    """The gate's own self-test: it must fail on a matrix that really does leak."""
    names = ["m00_core::mean"] + [W.TEACHER_PREFIX + "mag_total"]
    with pytest.raises(W.TeacherLeak):
        W.assert_no_teacher_in_features(names)


# ------------------------------------------------------------ teacher targets
def test_evidence_is_zero_for_every_no_break_series():
    rng = np.random.default_rng(0)
    for _ in range(20):
        online = rng.normal(0, 1, rng.integers(10, 400))
        e = W.teacher_evidence_path(rng.normal(0, 1, 300), online, -1)
        assert e.shape == online.shape
        assert (e == 0).all()


def test_evidence_is_monotone_and_bounded():
    rng = np.random.default_rng(1)
    for _ in range(20):
        hist, online, tau = _series(rng, shift=rng.uniform(0.2, 2.0))
        e = W.teacher_evidence_path(hist, online, tau)
        assert np.all(np.diff(e) >= -1e-7)
        assert e.min() >= 0.0 and e.max() <= 1.0
        assert (e[:tau] == 0).all()


def test_evidence_orders_series_by_true_severity():
    """The mechanism claim: a bigger true break earns more evidence at the same age."""
    rng = np.random.default_rng(2)
    small = W.teacher_evidence_path(*_series(rng, shift=0.25))
    rng = np.random.default_rng(2)
    big = W.teacher_evidence_path(*_series(rng, shift=2.5))
    assert big[200] > small[200]


def test_teacher_targets_are_deliberately_NOT_prefix_invariant():
    """This is the asymmetry the whole lane rests on, so it is asserted, not assumed.

    A FEATURE must be prefix-invariant; a teacher LABEL must not be, or it would
    carry no privileged information.  If this test ever passes, the teacher has
    become a causal statistic and lane A has no mechanism left.
    """
    rng = np.random.default_rng(3)
    hist, online, tau = _series(rng, n_post=250, shift=1.5)
    full = W.teacher_evidence_path(hist, online, tau)
    truncated = W.teacher_evidence_path(hist, online[:tau + 30], tau)
    assert not np.allclose(full[:tau + 30], truncated, atol=1e-6)


def test_permanence_separates_a_persistent_shift_from_a_transient_one():
    rng = np.random.default_rng(4)
    hist = rng.normal(0, 1, 400)
    persistent = np.r_[rng.normal(0, 1, 100), rng.normal(2.0, 1, 300)]
    transient = np.r_[rng.normal(0, 1, 100), rng.normal(2.0, 1, 40), rng.normal(0, 1, 260)]
    p = W.teacher_permanence(hist, persistent, 100)
    q = W.teacher_permanence(hist, transient, 100)
    # measured just after the apparent change, where the two look alike causally
    assert p[130] > q[130]


def test_undeclared_teacher_target_is_refused():
    class _FakeStore:
        meta = None

        def series(self, i):
            rng = np.random.default_rng(i)
            return _series(rng)

    with pytest.raises(KeyError):
        W.build_teacher_row_targets(_FakeStore(), [0], which=("not_declared",))


# ------------------------------------------------------------------- lane B
def test_horizon_routing_uses_only_the_current_index():
    t = np.arange(0, 1000)
    h = W.horizon_of(t)
    assert h.min() == 0 and h.max() == len(W.HORIZON_NAMES) - 1
    assert np.all(np.diff(h) >= 0)                     # monotone in t
    for lo, hi, i in zip(W.HORIZON_EDGES[:-1], W.HORIZON_EDGES[1:],
                         range(len(W.HORIZON_NAMES))):
        m = (t >= lo) & (t < hi)
        if m.any():
            assert (h[m] == i).all()


def test_a_prefix_routes_identically_however_the_series_continues():
    """Lane B's causality gate, in the same shape as the production one."""
    short = W.horizon_of(np.arange(0, 120))
    long_ = W.horizon_of(np.arange(0, 900))
    assert np.array_equal(short, long_[:120])
    ws = W.horizon_weights(np.arange(0, 120))
    wl = W.horizon_weights(np.arange(0, 900))
    assert np.allclose(ws, wl[:120], atol=0.0)


def test_horizon_weights_are_a_normalised_deterministic_function_of_t():
    w1 = W.horizon_weights(np.array([3, 60, 300, 900]))
    w2 = W.horizon_weights(np.array([3, 60, 300, 900]))
    assert np.array_equal(w1, w2)
    assert np.allclose(w1.sum(axis=1), 1.0)
    assert (w1 >= 0).all()
    # a later t must put more mass on later regimes
    assert w1[3].argmax() > w1[0].argmax()


# ------------------------------------------------------------------ reporting
def test_bucket_report_is_the_exact_decomposition_of_the_metric():
    """sum_A weight_A * AUC_A / W must equal the aggregate, to floating point.

    If this drifts, every promotion decision gated on an age bucket is being
    made against a statistic that is not the metric.
    """
    rng = np.random.default_rng(7)
    n_series, rows = 300, []
    for s in range(n_series):
        n = int(rng.integers(20, 400))
        tau = int(rng.integers(0, n)) if rng.random() < 0.5 else -1
        for t in range(n):
            y = int(tau >= 0 and t >= tau)
            rows.append((t, y, (t - tau) if y else -1,
                         rng.normal(0.4 * y + 0.002 * t, 1.0)))
    t = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows], np.int8)
    age = np.array([r[2] for r in rows])
    sc = np.array([r[3] for r in rows], np.float32)

    rep = W.bucket_report(sc, y, t, age)

    # exact pair weights per age bucket
    tmax = t.max()
    nneg = np.array([int(((y == 0) & (t == k)).sum()) for k in range(tmax + 1)])
    total = 0.0
    recon = 0.0
    for (lo, hi), key in zip(W.AGE_BUCKETS, list(rep["by_age"])):
        m = (y == 1) & (age >= lo) & (age < hi)
        w = float(nneg[t[m]].sum())
        total += w
        recon += w * rep["by_age"][key]
    assert total > 0
    assert abs(recon / total - rep["aggregate"]) < 1e-9


def test_bucket_report_aggregate_is_the_official_scorer():
    rng = np.random.default_rng(8)
    t = rng.integers(0, 50, 5000)
    y = (rng.random(5000) < 0.4).astype(np.int8)
    sc = rng.normal(0.3 * y, 1).astype(np.float32)
    age = np.where(y == 1, rng.integers(0, 200, 5000), -1)
    rep = W.bucket_report(sc, y, t, age)
    assert rep["aggregate"] == pytest.approx(float(ts_auc_flat(sc, y, t)), abs=0.0)


# ------------------------------------------------------------- champion config
def test_lane_arms_hold_the_champion_learner_fixed():
    """Every lane must vary one thing.  If this drifts, deltas stop being attributable.

    Read out of the pipeline's AST rather than matched as text, so quoting style
    cannot make the gate pass or fail for the wrong reason.
    """
    import ast
    import inspect
    import textwrap

    from sbr import pipeline

    tree = ast.parse(textwrap.dedent(inspect.getsource(pipeline.run)))
    found = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", None) == "default"
                and isinstance(node.value, ast.Call)):
            found = {kw.arg: ast.literal_eval(kw.value) for kw in node.value.keywords}
            break
    assert found is not None, "pipeline.run no longer defines its default params inline"
    for k, v in W.CHAMP_PARAMS.items():
        if k == "n_estimators":
            continue
        assert found.get(k) == v, f"{k}: wave 7 has {v!r}, the pipeline default is {found.get(k)!r}"
