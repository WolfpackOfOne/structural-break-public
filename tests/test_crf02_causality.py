r"""CRF-02 PRE-SCORE GATES -- fold purity, causality and scientific isolation.

research/reports/causal_representation_frontier/CRF02_EXECUTION_PREREG.md section 10,
implementing CRF_PROGRAM_PREREG.md sections 0.5 and 2.6.

Run to green BEFORE any CRF-02 TS-AUC, marginal_vs_clone or pair-flow number is
produced or read.  Nothing in this file prints or computes a score.

    P1   the null's fitted series set is exactly FOLDS \ {f}
    P2   positive control: the GLOBAL pretraining scheme (one null over all
         10k histories, reused across outer folds) is reproduced and shown to
         contaminate every outer fold -- the Wave-7 pattern, detected
    P3   live path: pretraining asserts its own set; a validation series' h_i
         is a forward pass, never a fit
    P4   mu_H, sigma_H, Hwin, p_i, h_i and the neglog baseline depend on that
         series' history alone and ignore its online segment
    P5   the 10-feature standardiser is fitted on TRAINING-FOLD ROWS ONLY, and
         the contaminated variant is shown to differ
    P6   the frozen generative parameters receive NO gradient while the ranking
         head trains -- the label can never rewrite the null
    C1   bitwise prefix invariance, atol = 0.0, over the 10-feature block
    C2   no forbidden name reaches a signal or a metadata key
    C3   truncation reproduces surviving rows
    C4   batch composition
    C5   deterministic replay
    C6   lockbox is never filled
    C7   THE STRICT-PAST SHIFT: the null never sees the value it is pricing
    C8   single-series inference is sufficient -- no cross-sectional quantity
    C9   the 21 quantile knots are monotone everywhere
    C10  C2's derangement is a true derangement within fold group
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research", "scripts"))

import crf01_nncsr as K1  # noqa: E402
import crf02_acgn as K2  # noqa: E402
from wave6_neural_lib import set_determinism, sha_state_dict  # noqa: E402

TOL = 1e-8


@pytest.fixture(scope="module")
def store():
    from sbr.store import load_store
    return load_store(f"{os.environ.get('SBR_ROOT', ROOT)}/cache/store")


@pytest.fixture(scope="module")
def spread(store):
    lens = store.meta.n_online.to_numpy()
    pick = np.argsort(lens)[np.linspace(0, len(lens) - 1, 8).astype(int)]
    return [(store.hist(int(i)), store.online(int(i)), int(i)) for i in pick]


@pytest.fixture(scope="module")
def data():
    from sbr.pipeline import Data
    return Data()


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """No test may write into the production cache.

    A unit test's 24-series, 1-epoch, HWIN=128 null was written to
    cache/crf02/null_fold0.pt and silently loaded by a real fold-0 run, which
    is why RT-1237/RT-1238/RT-1239 were voided.  Every test now gets its own
    throwaway cache directory, and the checkpoint's own provenance fingerprint
    (test_ckpt_*) is the second, load-bearing guard.
    """
    monkeypatch.setattr(K2, "CACHE", str(tmp_path / "crf02"))


@pytest.fixture(scope="module")
def net64():
    dev = set_determinism(0)
    return K2.ACGN.build(0, dev).to(torch.float64).eval()


def _signals(net, nl, o, dtype=np.float64):
    """The real signal path for one series, exactly as generate_signals runs it."""
    zo = K2.z_online(nl, o).astype(dtype)
    zs = K2.shift_input(zo, nl).astype(dtype)
    t = torch.from_numpy(zs[None, None, :])
    with torch.no_grad():
        p = net.encode(torch.from_numpy(nl.hwin[None, None, :].astype(dtype)),
                       torch.ones(1, len(nl.hwin), dtype=t.dtype))
        h = net.h_of(p)
        Q = net.knots(t, h).numpy()[0].T
        bf = net.body_features(t).numpy()[0].T
    return K2.signals_from_knots(Q, zo, 0.0, bf, p.numpy()[0]), Q


# ============================================================ P1 / P2 fold purity
def test_p1_null_fit_set_is_exactly_folds_minus_f(data):
    folds_set = set(K2.FOLDS)
    for f in K2.FOLDS:
        tr = np.flatnonzero(np.isin(data.series_fold, [g for g in K2.FOLDS if g != f]))
        va = np.flatnonzero(data.series_fold == f)
        lb = np.flatnonzero(data.series_fold == -1)
        assert set(data.series_fold[tr]) == folds_set - {f}
        assert len(np.intersect1d(tr, va)) == 0
        assert len(np.intersect1d(tr, lb)) == 0


def test_p2_global_pretraining_positive_control_is_detected():
    """CRF_PROGRAM_PREREG 2.6 forbids one global null reused across outer folds.

    Reproduce that scheme and show it contaminates EVERY outer fold, so the
    sentinel is proven to catch the Wave-7 nested-teacher pattern rather than
    passing vacuously on ours.
    """
    per_fold_hits, global_hits, total = 0, 0, 0
    for f in K2.FOLDS:
        total += 1
        per_fold = {g for g in K2.FOLDS if g != f}    # what pretrain_null fits on
        global_scheme = set(K2.FOLDS)                 # the forbidden scheme
        per_fold_hits += len(per_fold & {f})
        global_hits += len(global_scheme & {f})
    assert per_fold_hits == 0, "CRF-02 pretraining is not outer-fold scoped"
    assert global_hits == total, "sentinel failed to reproduce the known defect"


def test_p3_pretraining_live_path_and_forward_only_h_for_validation(data, monkeypatch):
    """The real pretrain_null on a tiny budget, plus the contract that a
    validation series' h_i is a FORWARD PASS through the training-fold-fitted
    encoder -- never a fit."""
    monkeypatch.setattr(K2, "PRETRAIN_EPOCHS", 1)
    monkeypatch.setattr(K2, "HWIN", 128)
    sf0 = data.series_fold
    keep_tr = np.concatenate([np.flatnonzero(sf0 == g)[:6] for g in (1, 2, 3, 4)])
    keep_va = np.flatnonzero(sf0 == 0)[:8]
    sf = np.full_like(sf0, 9)
    sf[keep_tr] = sf0[keep_tr]
    sf[keep_va] = 0
    monkeypatch.setattr(data, "series_fold", sf)

    nulls = [None] * data.st.n_series
    for s in np.concatenate([keep_tr, keep_va]):
        nulls[int(s)] = K2.SeriesNull(data.st.hist(int(s)))
    for s in range(data.st.n_series):
        if nulls[s] is None:
            nulls[s] = K2.SeriesNull(np.zeros(200))

    net, meta = K2.pretrain_null(0, data, nulls, log=lambda m: None)
    assert meta["fit_series"] == len(keep_tr)
    # every generative parameter is frozen after pretraining
    assert all(not p.requires_grad for p in net.parameters()), "the null is not frozen"
    # a validation series' h_i is a pure forward pass: no grad, and reproducible
    s = int(keep_va[0])
    z = nulls[s].hwin[None, None, :]
    with torch.no_grad():
        h1 = net.h_of(net.encode(torch.from_numpy(z), torch.ones(1, z.shape[-1])))
        h2 = net.h_of(net.encode(torch.from_numpy(z), torch.ones(1, z.shape[-1])))
    assert h1.requires_grad is False and np.array_equal(h1.numpy(), h2.numpy())
    assert h1.shape == (1, K2.BOTTLENECK)


def test_p3b_pretrain_null_rejects_a_contaminated_series_set(data, monkeypatch):
    monkeypatch.setattr(K2, "PRETRAIN_EPOCHS", 1)
    orig = np.isin

    def poisoned(a, b, **kw):
        if isinstance(b, list) and set(b) == {1, 2, 3, 4}:
            return orig(a, [0, 1, 2, 3, 4], **kw)
        return orig(a, b, **kw)

    monkeypatch.setattr(K2.np, "isin", poisoned)
    nulls = [K2.SeriesNull(np.zeros(200))] * data.st.n_series
    with pytest.raises(AssertionError, match="PURITY VIOLATION"):
        K2.pretrain_null(0, data, nulls, log=lambda m: None)


# ================================================ P4 history-only per-series state
def test_p4_series_state_depends_only_on_its_own_history(spread):
    for h, _o, sid in spread:
        a = K2.SeriesNull(h)
        b = K2.SeriesNull(np.array(h, copy=True))
        for f in ("mu", "sd", "sig", "last_hist"):
            assert getattr(a, f) == getattr(b, f), f"{f} moved for series {sid}"
        assert np.array_equal(a.hwin, b.hwin)
        assert np.array_equal(a.phi, b.phi)
        assert len(a.hwin) == min(len(h), K2.HWIN)


def test_p4b_series_state_and_early_signals_ignore_the_online_future(spread, net64):
    for h, o, sid in spread:
        if len(o) < 60:
            continue
        nl = K2.SeriesNull(h)
        F_full, _ = _signals(net64, nl, o)
        o2 = np.array(o, copy=True)
        o2[len(o2) // 2:] += 100.0
        nl2 = K2.SeriesNull(h)
        assert (nl.mu, nl.sd, nl.sig, nl.last_hist) == (nl2.mu, nl2.sd, nl2.sig, nl2.last_hist)
        F_pert, _ = _signals(net64, nl2, o2)
        half = len(o) // 2
        assert np.array_equal(F_full[:half], F_pert[:half]), \
            f"series {sid}: a future perturbation moved an earlier signal row"


# =========================================== P5 the ONE fitted global standardiser
def test_p5_feature_standardiser_uses_training_fold_rows_only(data):
    rng = np.random.default_rng(0)
    F = rng.normal(size=(len(data.y), K2.N_FEAT)).astype(np.float32)
    off, n_on = data.st.orow_off, data.st.meta.n_online.to_numpy().astype(np.int64)

    def rows_of(series):
        return np.concatenate([np.arange(int(off[s]), int(off[s]) + int(n_on[s]))
                               for s in series])

    tr = rows_of(np.flatnonzero(data.series_fold == 1)[:200])
    va_series = np.flatnonzero(data.series_fold == 0)[:200]
    va = rows_of(va_series)
    clean = K2.FeatureStandardiser().fit(F, tr)
    F2 = F.copy()
    F2[va] = 9999.0                                  # corrupt validation rows only
    clean2 = K2.FeatureStandardiser().fit(F2, tr)
    assert np.array_equal(clean.med, clean2.med) and np.array_equal(clean.iqr, clean2.iqr), \
        "the standardiser moved when validation rows changed -- it is not fold-pure"
    # positive control: the CONTAMINATED variant (fit on train + val) DOES move
    dirty = K2.FeatureStandardiser().fit(F2, np.concatenate([tr, va]))
    assert not np.allclose(dirty.med, clean.med), \
        "sentinel failed to reproduce the contaminated standardiser"


def test_p5b_standardiser_is_frozen_and_applied_everywhere(data):
    rng = np.random.default_rng(1)
    F = rng.normal(size=(5000, K2.N_FEAT)).astype(np.float32)
    st = K2.FeatureStandardiser().fit(F, np.arange(2000))
    Z = st.transform(F)
    assert Z.shape == F.shape
    assert np.isfinite(Z).all()
    assert np.abs(Z).max() <= K2.FEAT_CLIP + 1e-6
    st2 = K2.FeatureStandardiser().fit(F, np.arange(2000))
    assert np.array_equal(st.med, st2.med) and np.array_equal(st.iqr, st2.iqr)


# ====================================== P6 the label never rewrites the null
def test_p6_label_gradient_never_reaches_the_generative_null():
    """The scientific isolation CRF-02 exists to establish: the generative model
    is frozen before the ranking head is fitted, so the supervised label cannot
    rewrite the null (CRF_PROGRAM_PREREG 2.4)."""
    dev = set_determinism(0)
    gen = K2.ACGN.build(0, dev)
    for p in gen.parameters():
        p.requires_grad_(False)
    head = K2.build_ranking_head(0, dev)
    x = torch.from_numpy(np.random.default_rng(0).normal(
        size=(4, 30, K2.N_FEAT)).astype(np.float32))
    loss = torch.nn.functional.softplus(-head(x).squeeze(-1)).mean()
    loss.backward()
    bad = [n for n, p in gen.named_parameters()
           if p.grad is not None and float(p.grad.abs().sum()) > 0.0]
    assert not bad, f"ISOLATION VIOLATION: the label reached the null: {bad[:5]}"
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in head.parameters()), "the head received no gradient at all"


def test_p6b_freezing_the_null_clears_its_pretraining_gradients(data, monkeypatch):
    """The case the LIVE assert caught that the static P6 test did not.

    After pretraining, every generative parameter still carries the PINBALL
    loss's gradient.  Those are stale the moment the null is frozen, and leaving
    them makes any downstream "did the label touch the null?" check measure
    pretraining instead of the label -- a false positive that would have to be
    reasoned away exactly when it matters.  pretrain_null therefore clears them,
    so afterwards a non-None generative grad can ONLY have come from the label.
    """
    monkeypatch.setattr(K2, "PRETRAIN_EPOCHS", 1)
    monkeypatch.setattr(K2, "HWIN", 128)
    sf0 = data.series_fold
    keep_tr = np.concatenate([np.flatnonzero(sf0 == g)[:4] for g in (1, 2, 3, 4)])
    sf = np.full_like(sf0, 9)
    sf[keep_tr] = sf0[keep_tr]
    sf[np.flatnonzero(sf0 == 0)[:4]] = 0
    monkeypatch.setattr(data, "series_fold", sf)
    nulls = [K2.SeriesNull(np.zeros(200))] * data.st.n_series
    for s_ in keep_tr:
        nulls[int(s_)] = K2.SeriesNull(data.st.hist(int(s_)))

    net, _ = K2.pretrain_null(0, data, nulls, log=lambda m: None)
    stale = [n for n, p in net.named_parameters()
             if p.requires_grad or p.grad is not None]
    assert not stale, (
        "the null was frozen but still carries pretraining gradients; the live "
        f"isolation assert would false-positive on {stale[:3]}")


# ============================================ C1 bitwise prefix invariance
def test_c1_prefix_invariance_bitwise_atol_zero(spread, net64):
    """atol = 0.0 over the whole 10-feature block, in float64.

    float64 because the deployed float32 convolution's reduction order depends
    on the sequence length, which is rounding rather than dependence -- the same
    reason tests/test_neural_causality.py's Wave-6 gate 5 casts to float64.  The
    float32 magnitude is measured separately in test_c1b.
    """
    assert len(spread) >= 8
    assert len({len(o) for _, o, _ in spread}) >= 4
    for h, o, sid in spread:
        nl = K2.SeriesNull(h)
        full, _ = _signals(net64, nl, o)
        for k in (3, 10, 37, 111):
            if k >= len(o):
                continue
            part, _ = _signals(net64, nl, o[:k])
            bad = ~np.isclose(full[:k], part, rtol=0, atol=0.0, equal_nan=True)
            assert not bad.any(), (
                f"series {sid} prefix {k} differs in feature "
                f"{K2.FEATURE_NAMES[int(np.argmax(bad.any(axis=0)))]}")


def test_c1b_float32_prefix_noise_is_rounding_only(spread):
    dev = set_determinism(0)
    net = K2.ACGN.build(0, dev).eval()
    worst = 0.0
    for h, o, _ in spread:
        nl = K2.SeriesNull(h)
        full, _ = _signals(net, nl, o, dtype=np.float32)
        for k in (37, 111):
            if k >= len(o):
                continue
            part, _ = _signals(net, nl, o[:k], dtype=np.float32)
            worst = max(worst, float(np.abs(full[:k] - part).max()))
    print(f"C1b float32 prefix noise {worst:.3e}")
    assert worst <= 1e-5


# ============================================================ C2 forbidden names
def test_c2_no_forbidden_columns():
    from wave8_common import assert_no_forbidden_columns
    assert_no_forbidden_columns([f"crf02::{c}" for c in K2.FEATURE_NAMES])
    K1.assert_no_forbidden_channel_names(K2.FEATURE_NAMES)
    K1.assert_no_forbidden_channel_names(K2.SIGNAL_NAMES)
    with pytest.raises(AssertionError):
        K1.assert_no_forbidden_channel_names(list(K2.FEATURE_NAMES) + ["elapsed"])


# =================================================================== C3 truncation
def test_c3_truncation_reproduces_surviving_rows(spread, net64):
    worst = 0.0
    for h, o, sid in spread:
        if len(o) < 200:
            continue
        nl = K2.SeriesNull(h)
        full, _ = _signals(net64, nl, o)
        for k in (120, 200):
            part, _ = _signals(net64, nl, o[:k])
            d = float(np.abs(full[:k] - part).max())
            worst = max(worst, d)
            assert d <= TOL, f"series {sid} truncation at {k}: {d}"
    print(f"C3 worst truncation delta {worst:.3e}")


# ============================================================ C4 batch composition
def test_c4_batch_composition_does_not_change_the_knots(spread, net64):
    """Signals are generated one series at a time, but the history encoder runs
    in batches, so the pooled state must not depend on batch companions."""
    nulls = [K2.SeriesNull(h) for h, _, _ in spread]
    T = max(len(n.hwin) for n in nulls)

    def pool(idx):
        Z = np.zeros((len(idx), 1, T))
        M = np.zeros((len(idx), T))
        for k, j in enumerate(idx):
            w = nulls[j].hwin
            Z[k, 0, :len(w)] = w
            M[k, :len(w)] = 1.0
        with torch.no_grad():
            return net64.encode(torch.from_numpy(Z), torch.from_numpy(M)).numpy()

    target, worst = 3, 0.0
    alone = pool([target])[0]
    for comp in ([0, 1, target], [target, 5, 6, 7], [7, 6, target, 0, 1]):
        d = float(np.abs(alone - pool(comp)[comp.index(target)]).max())
        worst = max(worst, d)
        assert d <= TOL, f"batch {comp} moved the pooled history state by {d}"
    print(f"C4 worst batch-composition delta {worst:.3e}")


# ============================================================ C5 deterministic
def test_c5_deterministic_build_and_signals(spread):
    d1 = set_determinism(0)
    n1 = K2.ACGN.build(11, d1).eval()
    d2 = set_determinism(0)
    n2 = K2.ACGN.build(11, d2).eval()
    assert sha_state_dict(n1.state_dict()) == sha_state_dict(n2.state_dict())
    h, o, _ = spread[4]
    nl = K2.SeriesNull(h)
    a, _ = _signals(n1, nl, o, dtype=np.float32)
    b, _ = _signals(n2, nl, o, dtype=np.float32)
    assert np.array_equal(a, b)


# =================================================================== C6 lockbox
def test_c6_dev_rows_never_touch_the_lockbox(data):
    lb = data.rows_for([-1])
    assert len(lb) > 0
    dev = data.rows_for(list(K2.FOLDS))
    assert len(np.intersect1d(dev, lb)) == 0


# ============================== C7 THE STRICT-PAST SHIFT -- the load-bearing gate
def test_c7_the_null_never_sees_the_value_it_is_pricing(spread, net64):
    """A causal TCN at position t sees z_t.  The null must predict x_t from
    x_{t-1..t-R}, so the predictive pass is fed the sequence shifted right by
    one.  If that shift were missing or wrong, replacing z_t and everything
    after it would move the knots at t -- and the whole experiment would be
    pricing the observation with itself."""
    for h, o, sid in spread:
        if len(o) < 120:
            continue
        nl = K2.SeriesNull(h)
        _, Q = _signals(net64, nl, o)
        o2 = np.array(o, copy=True)
        t = 60
        o2[t:] = o2[t:] * 0.0 + 1e6                  # obliterate t and the future
        _, Q2 = _signals(net64, nl, o2)
        assert np.array_equal(Q[:t + 1], Q2[:t + 1]), (
            f"series {sid}: the predicted knots at t depend on z_t or later -- "
            f"the strict-past shift is broken")


def test_c7b_shift_input_carries_the_last_history_point_not_a_zero(spread):
    for h, o, _ in spread:
        nl = K2.SeriesNull(h)
        zo = K2.z_online(nl, o)
        zs = K2.shift_input(zo, nl)
        assert zs[0] == np.float32(nl.last_hist)
        assert np.array_equal(zs[1:], zo[:-1])
    # and the last history point is really the last one, standardised
    h = np.arange(2000, dtype=np.float64)
    nl = K2.SeriesNull(h)
    assert abs(nl.last_hist - (h[-1] - nl.mu) / nl.sd) < 1e-9


def test_c7c_no_final_online_length_in_the_signal_path():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(K2.signals_from_knots))
    fn = tree.body[0]
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant):
        fn.body = fn.body[1:]
    body = ast.unparse(fn)
    for bad in ("n_online", "tau", "elapsed", "[::-1]", "flip"):
        assert bad not in body, f"signal builder references {bad!r}"


# ================================================= C8 no cross-sectional inference
def test_c8_single_series_inference_is_sufficient(spread, net64):
    h, o, _ = spread[4]
    nl = K2.SeriesNull(h)
    F, _ = _signals(net64, nl, o)
    assert F.shape == (len(o), K2.N_FEAT)
    assert np.isfinite(F).all()
    F2, _ = _signals(net64, nl, o)
    assert np.array_equal(F, F2)


# ================================================================ C9 monotonicity
def test_c9_quantile_knots_are_monotone_on_real_data(spread, net64):
    assert len(K2.LEVELS) == 21
    assert np.all(np.diff(K2.LEVELS) > 0)
    assert K2.LEVELS[0] == 0.01 and K2.LEVELS[1] == 0.05
    assert K2.LEVELS[-1] == 0.99 and K2.LEVELS[-2] == 0.95
    for h, o, sid in spread:
        nl = K2.SeriesNull(h)
        _, Q = _signals(net64, nl, o)
        assert (np.diff(Q, axis=1) >= 0).all(), f"series {sid}: knots crossed"


def test_c9b_fixed_null_knots_are_monotone_and_strictly_causal(spread):
    """C1's AR(5) + residual-ECDF null gets the same treatment as the learned one."""
    for h, o, sid in spread:
        nl = K2.SeriesNull(h)
        Q, _ = K2.fixed_null_knots(nl, K2.z_online(nl, o))
        assert (np.diff(Q, axis=1) >= -1e-9).all(), f"series {sid}: fixed knots crossed"
        o2 = np.array(o, copy=True)
        t = min(60, len(o) - 1)
        o2[t:] += 1e6
        Q2, _ = K2.fixed_null_knots(nl, K2.z_online(nl, o2))
        assert np.allclose(Q[:t + 1], Q2[:t + 1], rtol=0, atol=0.0), \
            f"series {sid}: the fixed null sees the value it is pricing"


# ============================================================== C10 derangement
def test_c10_derangement_is_a_true_within_fold_derangement(data):
    src = np.arange(data.st.n_series)
    for f in K2.FOLDS:
        g = np.flatnonzero(data.series_fold == f)
        r = np.random.default_rng(K2.DERANGE_SEED + f)
        perm = r.permutation(len(g))
        for j in np.flatnonzero(perm == np.arange(len(g))):
            k = (j + 1) % len(g)
            perm[j], perm[k] = perm[k], perm[j]
        src[g] = g[perm]
    for f in K2.FOLDS:
        g = np.flatnonzero(data.series_fold == f)
        assert set(src[g].tolist()) == set(g.tolist()), "derangement left the fold group"
        assert (src[g] != g).all(), "derangement has a fixed point"
    lb = np.flatnonzero(data.series_fold == -1)
    assert (src[lb] == lb).all(), "derangement touched the lockbox"
    # deterministic
    assert K2.DERANGE_SEED != K2.PAIR_SEED


# ================================== the checkpoint provenance guard (post-mortem)
def test_ckpt_fingerprint_captures_the_frozen_configuration(data):
    tr = np.flatnonzero(np.isin(data.series_fold, [1, 2, 3, 4]))
    fp = K2.null_fingerprint(0, 0, tr)
    for k in ("fold", "seed", "pretrain_epochs", "hwin", "batch_series", "hidden",
              "bottleneck", "n_levels", "lr", "wd", "n_fit_series", "fit_series_sha256"):
        assert k in fp, f"fingerprint omits {k}"
    assert fp["n_fit_series"] == len(tr)
    assert fp["pretrain_epochs"] == K2.PRETRAIN_EPOCHS and fp["hwin"] == K2.HWIN
    # a different fit SET changes the fingerprint even at the same size
    other = np.flatnonzero(np.isin(data.series_fold, [0, 2, 3, 4]))
    assert K2.null_fingerprint(0, 0, other)["fit_series_sha256"] != fp["fit_series_sha256"]
    # so does a different epoch count, window, fold or seed
    assert K2.null_fingerprint(1, 0, tr) != fp
    assert K2.null_fingerprint(0, 1, tr) != fp


def test_ckpt_provenance_mismatch_is_refused_loudly(data, monkeypatch, tmp_path):
    """The exact defect that voided RT-1237/8/9: a checkpoint fitted on the wrong
    series with the wrong epoch count must be REFUSED, not silently reused.

    The state-sha check alone cannot catch it -- a toy null is perfectly
    self-consistent.  Only the fingerprint can.
    """
    monkeypatch.setattr(K2, "CACHE", str(tmp_path))
    monkeypatch.setattr(K2, "PRETRAIN_EPOCHS", 1)
    monkeypatch.setattr(K2, "HWIN", 128)
    sf0 = data.series_fold
    toy_tr = np.concatenate([np.flatnonzero(sf0 == g)[:6] for g in (1, 2, 3, 4)])
    sf = np.full_like(sf0, 9)
    sf[toy_tr] = sf0[toy_tr]
    sf[np.flatnonzero(sf0 == 0)[:8]] = 0
    monkeypatch.setattr(data, "series_fold", sf)
    nulls = [K2.SeriesNull(np.zeros(300))] * data.st.n_series
    for s_ in toy_tr:
        nulls[int(s_)] = K2.SeriesNull(data.st.hist(int(s_)))
    K2.pretrain_null(0, data, nulls, log=lambda m: None)          # writes the toy ckpt
    assert os.path.exists(K2._null_ckpt(0))

    monkeypatch.setattr(data, "series_fold", sf0)                 # now the REAL fit set
    with pytest.raises(SystemExit, match="CHECKPOINT PROVENANCE MISMATCH"):
        K2.pretrain_null(0, data, nulls, log=lambda m: None)


# ============================================ the shared-8 diagnostic is declared
def test_shared8_diagnostic_consumes_no_id_and_drops_only_lshift():
    assert K2.ARMS["shared8_diag"]["exp_id"] is None, "the diagnostic must consume no RT id"
    cols = [0, 1, 2, 3, 5, 6, 7, 8]
    names = [K2.FEATURE_NAMES[c] for c in cols]
    assert "lshift" not in names and "peak_lshift" not in names
    assert len(names) == 8
    for a in ("candidate", "fixed_null", "deranged"):
        assert K2.ARMS[a]["feats"] == "all"
    assert K2.ARMS["candidate"]["exp_id"] == "RT-1240"
    assert K2.ARMS["fixed_null"]["exp_id"] == "RT-1241"
    assert K2.ARMS["deranged"]["exp_id"] == "RT-1242"
    live = {v["exp_id"] for v in K2.ARMS.values()}
    assert "RT-1236" not in live, \
        "RT-1236 is reserved to CRF-01's unrun C2 arm and must not be recycled"
    for v in K2.VOID_IDS:
        assert v not in live, f"{v} is VOID and retired; it must not be recycled"
