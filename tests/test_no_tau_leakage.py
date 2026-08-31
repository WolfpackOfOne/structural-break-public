"""Regression gates for the RT-900 failure.

W6-E2 / `RT-900` scored 0.86552 TS-AUC against a 0.61605 champion because the
oracle feature block was `NaN` exactly when `t < cut`, and for a break series
`cut = tau`.  The block's MISSINGNESS MASK was the row-level target
`y[t] = 1[t >= tau]`, and LightGBM splits on missingness natively.  The bare
indicator `1[t >= cut]` scores 0.81442 by itself.

The standing rule that came out of it is in `research/PROTOCOL.md` §1:

    for the row-level real-time target, any experiment that gives the learner
    true tau -- directly OR through the availability, support, length,
    missingness, denominator, calibration window or segment boundary of a
    feature -- is invalid for predictive-performance measurement.

These tests pin the three properties that make the rule enforceable rather than
aspirational:

1.  A registered module CANNOT see tau.  The context it is handed is built from
    `(hist, online)` and carries no boundary, cut, tau or label anywhere.
2.  `w6oracle` is not, and never becomes, a registered module.
3.  No production column's MISSINGNESS is informative about the row-level label.
    This is the specific signature RT-900 had, tested directly rather than
    inferred from the absence of a tau argument -- a module could recreate the
    mask from something else and this is what would catch it.

They are cheap and they run without the feature cache.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from sbr.features.base import REGISTRY, load_all, make_ctx

PRODUCTION_MODULES = ("m00_core", "m01_seq", "m02_dist", "m03_dyn",
                      "m04_resid", "m06_loc", "m07_bayes")
#: anything on a context or in a column name that would smell of the boundary
FORBIDDEN_TOKENS = ("tau", "cut", "boundary", "break_at", "changepoint_true",
                    "has_break", "label", "target", "y_true", "n_online")


@pytest.fixture(scope="module")
def loaded():
    load_all()
    return REGISTRY


def _series(rng, n_hist=300, n_online=200, tau=90):
    h = rng.normal(size=n_hist)
    o = rng.normal(size=n_online)
    o[tau:] *= 2.2          # a real scale break, so the mask has something to hide behind
    o[tau:] += 0.8
    return h, o


# --------------------------------------------------------------------------- 1
def test_make_ctx_takes_only_hist_and_online():
    """The engine's entry point cannot be handed tau even by accident."""
    sig = inspect.signature(make_ctx)
    required = [n for n, p in sig.parameters.items()
                if p.default is inspect.Parameter.empty]
    assert required == ["hist", "online"], (
        f"make_ctx's required arguments changed to {required}; if a boundary "
        "argument was added, PROTOCOL.md §1 has been broken")


def test_context_carries_no_boundary_or_label(loaded):
    """No attribute of the per-series context may name tau, the cut or the label."""
    rng = np.random.default_rng(0)
    h, o = _series(rng)
    ctx = make_ctx(h, o)
    bad = []
    for name in dir(ctx):
        if name.startswith("_"):
            continue
        low = name.lower()
        for tok in FORBIDDEN_TOKENS:
            if tok in low:
                bad.append(name)
    assert not bad, f"context exposes boundary/label-ish attributes: {bad}"


def test_production_column_names_carry_no_boundary_token(loaded):
    rng = np.random.default_rng(1)
    h, o = _series(rng)
    ctx = make_ctx(h, o)
    bad = []
    for m in PRODUCTION_MODULES:
        names, _ = loaded[m].fn(ctx)
        for c in names:
            low = c.lower()
            # t_online / log_t_online are elapsed-online COUNTERS, which are
            # causal and legal; the forbidden thing is the FINAL length.
            if low in ("t_online", "log_t_online"):
                continue
            for tok in FORBIDDEN_TOKENS:
                if tok in low:
                    bad.append(f"{m}.{c}")
    assert not bad, f"production columns name a boundary/label concept: {bad}"


# --------------------------------------------------------------------------- 2
def test_w6oracle_is_not_registered(loaded):
    """The voided RT-900 block must never be reachable from load_all()."""
    assert "w6oracle" not in loaded, (
        "w6oracle is registered. It is the VOID RT-900 true-tau block "
        "(research/FAILED_EXPERIMENTS.md) and must never reach a manifest.")
    for name in loaded:
        assert "oracle" not in name.lower(), f"oracle-ish module registered: {name}"


def test_production_module_set_is_unchanged(loaded):
    for m in PRODUCTION_MODULES:
        assert m in loaded, f"production module {m} vanished from the registry"


# --------------------------------------------------------------------------- 3
def _flat_mask_panel(loaded, n_series=24, seed=2026):
    """A little panel of series with DIFFERENT taus, flattened row-wise.

    This is the shape the metric actually sees: at a fixed online index `t`,
    TS-AUC compares series against each other.  A mask that is a deterministic
    function of `t` alone -- a fixed warm-up, say -- is constant within every
    timestep and therefore carries EXACTLY ZERO ranking information, however
    strongly it correlates with `y` when the rows are pooled.  A mask that
    depends on `tau` is the RT-900 leak.  Only the within-timestep view
    separates the two, so that is the view this test uses.
    """
    rng = np.random.default_rng(seed)
    masks = {m: [] for m in PRODUCTION_MODULES}
    ys, ts = [], []
    for k in range(n_series):
        n_online = int(rng.integers(150, 400))
        n_hist = 600      # the shipped store's n_hist floor is 512; below it the
                          # m00_core window bank narrows and widths stop matching
        broke = k % 3 != 0                      # a third of the panel never breaks
        tau = int(rng.integers(20, n_online - 20)) if broke else n_online + 1
        h = rng.normal(size=n_hist)
        o = rng.normal(size=n_online)
        if broke:
            o[tau:] = o[tau:] * 2.2 + 0.8
        ctx = make_ctx(h, o)
        for m in PRODUCTION_MODULES:
            _, A = loaded[m].fn(ctx)
            masks[m].append(~np.isfinite(np.asarray(A, dtype=np.float64)))
        ys.append((np.arange(n_online) >= tau).astype(np.int8))
        ts.append(np.arange(n_online, dtype=np.int64))
    names = {}
    for m in PRODUCTION_MODULES:
        names[m] = loaded[m].fn(make_ctx(rng.normal(size=600), rng.normal(size=120)))[0]
        masks[m] = np.vstack(masks[m])
    return masks, names, np.concatenate(ys), np.concatenate(ts)


def test_no_production_column_missingness_carries_within_timestep_information(loaded):
    """The RT-900 signature, tested with the official scorer.

    RT-900's oracle block was NaN exactly when `t < cut`, and for a break series
    `cut = tau`.  Scoring that mask as if it were a model gives a degenerate
    TS-AUC -- the bare indicator `1[t >= cut]` scored **0.81442** on the real
    dev set.  No production column's AVAILABILITY may behave that way.
    """
    from sbr.metric import ts_auc_flat

    masks, names, y, t = _flat_mask_panel(loaded)
    worst = ("none", 0.5)
    for m in PRODUCTION_MODULES:
        M = masks[m]
        for j, c in enumerate(names[m]):
            col = M[:, j].astype(np.float64)
            if col.min() == col.max():
                continue                      # constant mask carries nothing
            a = float(ts_auc_flat(col, y, t))
            if abs(a - 0.5) > abs(worst[1] - 0.5):
                worst = (f"{m}.{c}", a)
    assert abs(worst[1] - 0.5) <= 0.10, (
        f"missingness of {worst[0]} scores TS-AUC {worst[1]:.5f} as a standalone "
        "predictor. This is the RT-900 failure mode: a column whose AVAILABILITY "
        "tracks whether the break has happened, which under the row-level target "
        "IS the label. See research/PROTOCOL.md §1 and "
        "research/FAILED_EXPERIMENTS.md.")


def test_missingness_pattern_does_not_depend_on_where_the_break_is(loaded):
    """Two series identical except for tau must have the same NaN pattern.

    If the set of available columns at row t depended on tau, that dependence
    IS the label under the row-level target -- regardless of how the values
    were computed.
    """
    rng = np.random.default_rng(7)
    n_hist, n_online = 300, 240
    h = rng.normal(size=n_hist)
    base = rng.normal(size=n_online)

    def pattern(tau):
        o = base.copy()
        o[tau:] = o[tau:] * 2.2 + 0.8
        ctx = make_ctx(h, o)
        return {m: ~np.isfinite(np.asarray(loaded[m].fn(ctx)[1], dtype=np.float64))
                for m in PRODUCTION_MODULES}

    a = pattern(60)
    b = pattern(170)
    for m in PRODUCTION_MODULES:
        # column-wise: a column that is entirely present in one and entirely
        # present in the other is fine; what is forbidden is a mask that
        # switches at the break.
        for j in range(a[m].shape[1]):
            ca, cb = a[m][:, j], b[m][:, j]
            if ca.all() and cb.all():
                continue
            if not ca.any() and not cb.any():
                continue
            # the mask may legitimately depend on the DATA (a degenerate window
            # of constant values, say); it may not switch AT tau in both.
            sa = np.flatnonzero(np.diff(ca.astype(int)) != 0)
            sb = np.flatnonzero(np.diff(cb.astype(int)) != 0)
            if len(sa) == 1 and len(sb) == 1:
                assert not (abs(sa[0] + 1 - 60) <= 1 and abs(sb[0] + 1 - 170) <= 1), (
                    f"{m} column {j}: its NaN mask switches exactly at tau in "
                    "both series. That mask is the label.")
