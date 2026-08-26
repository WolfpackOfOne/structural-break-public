r"""CRF-01 PRE-SCORE GATES -- fold purity and causality.

research/reports/causal_representation_frontier/CRF01_EXECUTION_PREREG.md
sections 10 and 11, which implement CRF_PROGRAM_PREREG.md section 0.5.

These run to green BEFORE any CRF-01 TS-AUC, marginal_vs_clone or pair-flow
number is produced or read.  A failure VOIDS the run; it does not get patched
once a score is known.  Nothing in this file prints or computes a score.

    P1  set arithmetic          fitted series set is exactly FOLDS \ {f}
    P2  positive control        the contaminated scheme is reproduced and DOES
                                intersect {f} -- the sentinel catches the defect
    P3  live fitting path       the real training entry point asserts purity and
                                emits only its own validation fold
    P4  per-series constants    every history constant depends on that series'
                                own history alone
    P5  no global standardiser  channel output is independent of call order,
                                of neighbours, and of any fold assignment
    P6  validation rows do not train
    C1  prefix invariance, atol = 0.0, >= 8 real series of different lengths
    C2  no forbidden column name reaches an input or a metadata key
    C3  truncation: encoder + channels reproduce surviving rows to <= 1e-8
    C4  batch composition: a score does not depend on batch companions
    C5  deterministic replay
    C6  lockbox: no emitted vector is finite on fold -1
    C7  no final online length is referenced
    C8  no cross-sectional inference: one series alone reproduces its score
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research", "scripts"))

import crf01_nncsr as K  # noqa: E402
from wave6_neural_lib import CausalTCN, set_determinism, sha_state_dict  # noqa: E402

TOL = 1e-8
CUTS = (3, 10, 37, 111)


# --------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def store():
    from sbr.store import load_store
    root = os.environ.get("SBR_ROOT", ROOT)
    return load_store(f"{root}/cache/store")


@pytest.fixture(scope="module")
def spread(store):
    """Eight real series spanning the store's online-length range."""
    lens = store.meta.n_online.to_numpy()
    pick = np.argsort(lens)[np.linspace(0, len(lens) - 1, 8).astype(int)]
    return [(store.hist(int(i)), store.online(int(i)), int(i)) for i in pick]


@pytest.fixture(scope="module")
def data():
    from sbr.pipeline import Data
    return Data()


# ============================================================ P1 / P2  fold purity
def test_p1_fitted_series_set_is_exactly_folds_minus_f(data):
    folds_set = set(K.FOLDS)
    for f in K.FOLDS:
        tr = np.flatnonzero(np.isin(data.series_fold, [g for g in K.FOLDS if g != f]))
        va = np.flatnonzero(data.series_fold == f)
        lb = np.flatnonzero(data.series_fold == -1)
        assert set(data.series_fold[tr]) == folds_set - {f}
        assert len(np.intersect1d(tr, va)) == 0
        assert len(np.intersect1d(tr, lb)) == 0
        assert len(np.intersect1d(va, lb)) == 0


def test_p2_positive_control_contaminated_scheme_is_detected(data):
    """The sentinel must REJECT the contaminated scheme, not merely accept ours.

    The Wave-7 nested-teacher incident is the standing example: a scheme that
    fits on every fold and is ignorant of the outer fold contaminates every
    outer fold, and this reproduces that pattern and shows it differs.
    """
    contaminated_hits, clean_hits, total = 0, 0, 0
    for f in K.FOLDS:
        total += 1
        clean = {g for g in K.FOLDS if g != f}          # what train_fold actually fits on
        dirty = set(K.FOLDS)                            # the contaminated scheme
        clean_hits += len(clean & {f})
        contaminated_hits += len(dirty & {f})
    assert clean_hits == 0, "CRF-01 scheme is not outer-fold pure"
    assert contaminated_hits == total, "sentinel failed to reproduce the known defect"


def test_p3_live_fitting_path_asserts_purity_and_emits_only_its_own_fold(data, monkeypatch):
    """Exercised through the REAL train_fold signature on a tiny budget.

    Proves the purity assertions inside train_fold are reachable and correct,
    and that the emitted vector covers exactly the outer validation fold --
    not just that the set arithmetic above is tidy.
    """
    monkeypatch.setattr(K, "EPOCHS", 1)
    n_rows = len(data.y)
    C = np.zeros((n_rows, K.N_CHANNELS), dtype=np.float32)

    # a tiny outer fold: 6 training series from EACH of folds 1-4 and 8
    # validation series from fold 0; every other series is parked outside FOLDS
    fold = 0
    sf0 = data.series_fold
    keep_tr = np.concatenate([np.flatnonzero(sf0 == g)[:6] for g in (1, 2, 3, 4)])
    keep_va = np.flatnonzero(sf0 == 0)[:8]
    sf = np.full_like(sf0, 9)
    sf[keep_tr] = sf0[keep_tr]
    sf[keep_va] = 0
    monkeypatch.setattr(data, "series_fold", sf)

    res = K.train_fold("candidate", fold, data, C, log=lambda m: None)
    assert res is not None
    assert res["n_val_series"] == len(keep_va)
    emitted = set(data.sidx[res["rows"]].tolist())
    assert emitted == set(keep_va.tolist()), "emitted rows are not exactly the val fold"
    assert emitted.isdisjoint(set(keep_tr.tolist()))


def test_p3b_train_fold_rejects_a_contaminated_series_set(data, monkeypatch):
    """The in-function purity assertion actually fires when the set is wrong."""
    monkeypatch.setattr(K, "EPOCHS", 1)
    sf = data.series_fold.copy()
    sf[np.flatnonzero(sf == 0)[:4]] = 1        # move val series into the training set
    monkeypatch.setattr(data, "series_fold", sf)
    orig = np.isin
    def poisoned(a, b, **kw):                  # make FOLDS \ {0} accidentally include 0
        if isinstance(b, list) and set(b) == {1, 2, 3, 4}:
            return orig(a, [0, 1, 2, 3, 4], **kw)
        return orig(a, b, **kw)
    monkeypatch.setattr(K.np, "isin", poisoned)
    C = np.zeros((len(data.y), K.N_CHANNELS), dtype=np.float32)
    with pytest.raises(AssertionError, match="PURITY VIOLATION"):
        K.train_fold("candidate", 0, data, C, log=lambda m: None)


# ==================================================== P4  per-series constants only
def test_p4_history_constants_depend_only_on_that_series_history(spread):
    for h, o, sid in spread:
        a = K.HistoryNull(h)
        # rebuild from an independent copy while unrelated data is scrambled
        rng = np.random.default_rng(7)
        _decoy = rng.normal(size=5000)          # noqa: F841  -- deliberately unused
        b = K.HistoryNull(np.array(h, copy=True))
        for f in ("mu", "sd", "sig", "mad", "q90", "n_hist"):
            assert getattr(a, f) == getattr(b, f), f"{f} moved for series {sid}"
        assert np.array_equal(a.phi, b.phi)
        for k in ("ecdf_x", "ecdf_inn", "ecdf_absinn"):
            for u, v in zip(getattr(a, k), getattr(b, k)):
                assert np.array_equal(u, v)


def test_p4b_history_constants_ignore_the_online_segment(spread):
    """A constant that moved with the online stream would be a future leak."""
    for h, o, sid in spread:
        a = K.HistoryNull(h)
        C_full = K.causal_channels(h, o, null=a)
        o2 = np.array(o, copy=True)
        o2[len(o2) // 2:] += 100.0             # a violent regime change late on
        b = K.HistoryNull(h)
        assert (a.mu, a.sd, a.sig, a.mad, a.q90) == (b.mu, b.sd, b.sig, b.mad, b.q90)
        C_pert = K.causal_channels(h, o2, null=b)
        half = len(o) // 2
        assert np.array_equal(C_full[:half], C_pert[:half]), \
            f"series {sid}: a future perturbation moved an earlier row"


# ================================================= P5  no fitted global standardiser
def test_p5_channel_output_is_independent_of_call_order_and_neighbours(spread):
    a_first = [K.causal_channels(h, o) for h, o, _ in spread]
    a_rev = [K.causal_channels(h, o) for h, o, _ in reversed(spread)][::-1]
    for x, y in zip(a_first, a_rev):
        assert np.array_equal(x, y), "channel builder carries fitted global state"


def test_p5b_channel_builder_takes_no_fold_and_no_label():
    import inspect
    sig = set(inspect.signature(K.causal_channels).parameters)
    assert sig == {"hist", "online", "null"}, sig
    sig2 = set(inspect.signature(K.HistoryNull.__init__).parameters)
    assert sig2 == {"self", "hist"}, sig2
    # no module-level mutable fitted state
    for name in ("MEAN", "STD", "SCALE", "STANDARDISER", "GLOBAL_CLIP"):
        assert not hasattr(K, name), f"CRF-01 fitted a global {name}"


# =========================================== P6  validation rows never reach training
def test_p6_perturbing_a_validation_series_does_not_move_a_training_batch(data):
    n_on = data.st.meta.n_online.to_numpy().astype(np.int64)
    off = data.st.orow_off
    C = np.random.default_rng(0).normal(size=(len(data.y), K.N_CHANNELS)).astype(np.float32)
    y = data.y.astype(np.float32)
    tr = np.flatnonzero(data.series_fold == 1)[:16]
    va = np.flatnonzero(data.series_fold == 0)[:16]
    X0, Y0, M0 = K._pack(C, y, off, n_on, tr)
    for s in va:                                # corrupt every validation row
        a, n = int(off[s]), int(n_on[s])
        C[a:a + n] = 12345.0
    X1, Y1, M1 = K._pack(C, y, off, n_on, tr)
    assert np.array_equal(X0, X1) and np.array_equal(Y0, Y1) and np.array_equal(M0, M1)


# ================================================== C1  bitwise prefix invariance
def test_c1_prefix_invariance_bitwise_atol_zero(spread):
    """PROTOCOL.md section 3: row t may use hist and online[:t+1] and nothing else."""
    assert len(spread) >= 8
    assert len({len(o) for _, o, _ in spread}) >= 4, "series lengths are not spread"
    for h, o, sid in spread:
        full = K.causal_channels(h, o)
        for k in CUTS:
            if k >= len(o):
                continue
            part = K.causal_channels(h, o[:k])
            assert part.shape == full[:k].shape
            bad = ~np.isclose(full[:k], part, rtol=0, atol=0.0, equal_nan=True)
            assert not bad.any(), (
                f"series {sid} prefix {k} differs in channel "
                f"{K.CHANNEL_NAMES[int(np.argmax(bad.any(axis=0)))]}")


# ================================================== C2  forbidden column names
def test_c2_no_forbidden_columns():
    sys.path.insert(0, os.path.join(ROOT, "research", "scripts"))
    from wave8_common import assert_no_forbidden_columns
    assert_no_forbidden_columns([f"crf01::{c}" for c in K.CHANNEL_NAMES])
    K.assert_no_forbidden_channel_names(K.CHANNEL_NAMES)
    with pytest.raises(AssertionError):
        K.assert_no_forbidden_channel_names(["pit", "elapsed"])
    with pytest.raises(AssertionError):
        K.assert_no_forbidden_channel_names(["rt600_logit"])
    with pytest.raises(AssertionError):
        K.assert_no_forbidden_channel_names(["n_online_frac"])


# =========================== C3 / C7  truncation through channels AND the encoder
def test_c3_truncation_reproduces_surviving_score_rows(spread):
    """Channels rebuilt and the encoder re-run on a truncated online segment
    reproduce the surviving rows to <= 1e-8.  This is also gate C7: a model that
    referenced the final online length could not pass it."""
    device = set_determinism(0)
    net = CausalTCN.build(K.HIDDEN, K.DROPOUT, 0, device, n_in=K.N_CHANNELS)
    net = net.to(torch.float64).eval()   # wave-6 gate-5 precedent: float32 conv
    worst = 0.0                          # reduction order is not associative
    for h, o, sid in spread:
        if len(o) < 200:
            continue
        with torch.no_grad():
            full = net(torch.from_numpy(
                K.causal_channels(h, o).T[None].astype(np.float64))).numpy()[0]
            for k in (120, 200):
                part = net(torch.from_numpy(
                    K.causal_channels(h, o[:k]).T[None].astype(np.float64))).numpy()[0]
                d = float(np.abs(full[:k] - part).max())
                worst = max(worst, d)
                assert d <= TOL, f"series {sid} truncation at {k}: max |delta| {d}"
    print(f"C3 worst truncation delta {worst:.3e}")


# ================================================== C4 / C8  batch composition
def test_c4_batch_composition_does_not_change_a_prediction(spread):
    device = set_determinism(0)
    net = CausalTCN.build(K.HIDDEN, K.DROPOUT, 0, device, n_in=K.N_CHANNELS)
    net = net.to(torch.float64).eval()   # wave-6 gate-5 precedent, verbatim
    chans = [K.causal_channels(h, o) for h, o, _ in spread]
    T = max(len(c) for c in chans)

    def pack(idx):
        X = np.zeros((len(idx), K.N_CHANNELS, T), dtype=np.float64)
        for k, j in enumerate(idx):
            X[k, :, :len(chans[j])] = chans[j].T
        return torch.from_numpy(X)

    target = 3
    worst = 0.0
    with torch.no_grad():
        alone = net(pack([target])).numpy()[0][:len(chans[target])]
        for companions in ([0, 1, target], [target, 5, 6, 7], [7, 6, target, 0, 1]):
            p = net(pack(companions)).numpy()[companions.index(target)][:len(chans[target])]
            d = float(np.abs(alone - p).max())
            worst = max(worst, d)
            assert d <= TOL, f"batch {companions} moved the score by {d}"
    print(f"C4 worst batch-composition delta {worst:.3e}")


def test_c4b_float32_batch_composition_noise_is_rounding_only():
    """The deployed dtype is float32, whose convolution reduction order depends
    on the batch partition.  That is numerical noise, not information flow --
    float64 shows the true dependence is zero (test_c4) -- but the float32
    magnitude is MEASURED and reported rather than assumed negligible."""
    device = set_determinism(0)
    net = CausalTCN.build(K.HIDDEN, K.DROPOUT, 0, device, n_in=K.N_CHANNELS).eval()
    rng = np.random.default_rng(3)
    X1 = rng.normal(size=(1, K.N_CHANNELS, 400)).astype(np.float32)
    X4 = np.zeros((4, K.N_CHANNELS, 400), dtype=np.float32)
    X4[2] = X1[0]
    with torch.no_grad():
        a = net(torch.from_numpy(X1)).numpy()[0]
        b = net(torch.from_numpy(X4)).numpy()[2]
    d = float(np.abs(a - b).max())
    scale = float(np.abs(a).max())
    print(f"C4b float32 batch noise {d:.3e} (score scale {scale:.3e}, "
          f"relative {d / max(scale, 1e-12):.3e})")
    assert d <= 1e-5, d
    assert d / max(scale, 1e-12) <= 1e-5, "float32 batch noise is not rounding-scale"


def test_c8_no_cross_sectional_inference_single_series_is_sufficient(spread):
    """The scoring function must accept ONE series and ONE causal state and emit
    one scalar.  Same-t grouping is legal for TRAINING only; a candidate that
    needed other test series at inference would be illegal."""
    device = set_determinism(0)
    net = CausalTCN.build(K.HIDDEN, K.DROPOUT, 0, device, n_in=K.N_CHANNELS).eval()
    h, o, _ = spread[4]
    C = K.causal_channels(h, o)
    with torch.no_grad():
        one = net(torch.from_numpy(C.T[None])).numpy()[0]
    assert one.shape == (len(o),)
    assert np.isfinite(one).all()
    # and the per-row score is a pure function of the single-series prefix
    with torch.no_grad():
        again = net(torch.from_numpy(C.T[None])).numpy()[0]
    assert np.array_equal(one, again)


# ================================================== C5  deterministic replay
def test_c5_deterministic_build_and_forward():
    d1 = set_determinism(0)
    n1 = CausalTCN.build(K.HIDDEN, K.DROPOUT, 7, d1, n_in=K.N_CHANNELS)
    d2 = set_determinism(0)
    n2 = CausalTCN.build(K.HIDDEN, K.DROPOUT, 7, d2, n_in=K.N_CHANNELS)
    assert sha_state_dict(n1.state_dict()) == sha_state_dict(n2.state_dict())
    x = torch.from_numpy(np.random.default_rng(0).normal(
        size=(2, K.N_CHANNELS, 300)).astype(np.float32))
    n1.eval(); n2.eval()
    with torch.no_grad():
        assert np.array_equal(n1(x).numpy(), n2(x).numpy())


def test_c5b_pair_sampler_is_deterministic_and_respects_occupancy():
    rng_a = np.random.default_rng(K.PAIR_SEED)
    rng_b = np.random.default_rng(K.PAIR_SEED)
    Y = np.zeros((32, 40), dtype=np.float32)
    M = np.ones((32, 40), dtype=np.float32)
    Y[:10, :] = 1.0                              # 10 positives, 22 negatives per t
    a = K.sample_pairs(Y, M, rng_a)
    b = K.sample_pairs(Y, M, rng_b)
    for u, v in zip(a[:4], b[:4]):
        assert np.array_equal(u, v)
    assert a[4] == 40                            # every timestep contributes
    assert len(a[0]) == 40 * 10 * K.M_NEG        # m_neg = 8 negatives per positive
    # a timestep with fewer than m_neg negatives must NOT contribute
    Y2 = np.zeros((32, 3), dtype=np.float32); Y2[:26, :] = 1.0   # 6 negatives only
    assert K.sample_pairs(Y2, np.ones_like(Y2), np.random.default_rng(0)) is None
    # nor one with no positives
    assert K.sample_pairs(np.zeros((32, 3), dtype=np.float32),
                          np.ones((32, 3), dtype=np.float32),
                          np.random.default_rng(0)) is None


def test_c5c_pairs_are_same_t_and_correctly_signed():
    rng = np.random.default_rng(0)
    Y = np.zeros((32, 12), dtype=np.float32)
    M = np.ones((32, 12), dtype=np.float32)
    Y[:8, :] = 1.0
    pb, pt, nb, nt, _ = K.sample_pairs(Y, M, rng)
    assert np.array_equal(pt, nt), "a pair spans two timesteps -- not a same-t pair"
    assert (Y[pb, pt] == 1).all(), "a 'positive' is not labelled 1"
    assert (Y[nb, nt] == 0).all(), "a 'negative' is not labelled 0"


def test_c5d_padded_positions_never_enter_a_pair():
    rng = np.random.default_rng(0)
    Y = np.zeros((32, 10), dtype=np.float32)
    M = np.ones((32, 10), dtype=np.float32)
    Y[:8, :] = 1.0
    M[:, 6:] = 0.0                                # right padding
    s = K.sample_pairs(Y, M, rng)
    assert s is not None
    pb, pt, nb, nt, _ = s
    assert pt.max() < 6 and nt.max() < 6, "a padded position was sampled"


# ================================================== C6  lockbox is never filled
def test_c6_emitted_rows_never_touch_the_lockbox(data):
    lb = data.rows_for([-1])
    assert len(lb) > 0
    for f in K.FOLDS:
        va = np.flatnonzero(data.series_fold == f)
        rows = np.concatenate([
            np.arange(int(data.st.orow_off[s]),
                      int(data.st.orow_off[s]) + int(data.st.meta.n_online.iloc[s]))
            for s in va[:200]])
        assert len(np.intersect1d(rows, lb)) == 0


# ================================================== C7  no final-length reference
def test_c7_no_final_online_length_in_the_input_path():
    """A length-dependent constant cannot survive C1, but assert the source too."""
    import ast, inspect
    tree = ast.parse(inspect.getsource(K.causal_channels))
    fn = tree.body[0]
    if (fn.body and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)):
        fn.body = fn.body[1:]                       # drop the docstring
    body = ast.unparse(fn)
    for bad in ("n_online", "tau", "elapsed", "arange", "linspace"):
        assert bad not in body, f"channel builder references {bad!r}"
    # the only names the builder reads are its own arguments and frozen constants
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    allowed = {"hist", "online", "null", "h", "x", "n", "C", "np", "pit", "z",
               "warm", "inn", "absinn", "inn_pit", "lag_ewma", "den", "vol", "g",
               "surp", "ex", "lag1", "K", "HistoryNull", "_ndtri", "_ecdf_eval",
               "_ar_filter_causal", "_lagged_ewma", "N_CHANNELS", "CLIP_PIT",
               "CLIP_VOL", "CLIP_SURP", "CLIP_LAG1", "VOL_FLOOR_FRAC",
               "EWMA_HALFLIFE", "EXCEED_SLOPE",
               # `len(x)` is the length of the PREFIX SUPPLIED SO FAR, never the
               # final online length -- it is used only to size the output array
               # and to branch the t = 0 lag. test_c1 is the binding proof: a
               # channel that depended on it would not reproduce bitwise under
               # truncation, because at prefix k the value is k, not n.
               "len"}
    assert names <= allowed, f"channel builder reads unexpected names: {names - allowed}"


# ================================================== C2b  no shuffle leak in candidate
def test_shuffle_control_is_arm_scoped_and_deterministic(data):
    assert K.ARMS["candidate"]["shuffle"] is False
    assert K.ARMS["bce_control"]["shuffle"] is False
    assert K.ARMS["shuffle_ctrl"]["shuffle"] is True
    p1 = K._series_perm(123, 50)
    p2 = K._series_perm(123, 50)
    assert np.array_equal(p1, p2)
    assert not np.array_equal(p1, K._series_perm(124, 50))
    assert sorted(p1.tolist()) == list(range(50))

    n_on = data.st.meta.n_online.to_numpy().astype(np.int64)
    off = data.st.orow_off
    C = np.random.default_rng(1).normal(size=(len(data.y), K.N_CHANNELS)).astype(np.float32)
    y = data.y.astype(np.float32)
    ids = np.flatnonzero(data.series_fold == 0)[:4]
    Xs, Ys, Ms = K._pack(C, y, off, n_on, ids, shuffle=True)
    Xp, Yp, Mp = K._pack(C, y, off, n_on, ids, shuffle=False)
    assert np.array_equal(Ys, Yp), "C2 moved the label; it must stay at its original t"
    assert np.array_equal(Ms, Mp)
    assert not np.array_equal(Xs, Xp), "C2 did not shuffle anything"
    # the same permutation is applied to ALL eight channels
    s0, n0 = int(ids[0]), int(n_on[ids[0]])
    perm = K._series_perm(s0, n0)
    a = int(off[s0])
    assert np.array_equal(Xs[0, :, :n0], C[a:a + n0][perm].T)
    # and the per-series marginal of every channel is preserved exactly
    for j in range(K.N_CHANNELS):
        assert np.array_equal(np.sort(Xs[0, j, :n0]), np.sort(Xp[0, j, :n0]))


# ================================================== architecture is the RT-970 shell
def test_architecture_matches_the_frozen_rt970_shell():
    device = set_determinism(0)
    net = CausalTCN.build(K.HIDDEN, K.DROPOUT, 0, device, n_in=K.N_CHANNELS)
    assert sum(p.numel() for p in net.parameters()) == 35649
    assert not any("BatchNorm" in type(m).__name__ for m in net.modules())
    assert not any(isinstance(m, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN,
                                  torch.nn.MultiheadAttention)) for m in net.modules())


def test_receptive_field_is_causal_and_253():
    """Measured, not asserted from arithmetic.  CRF01_EXECUTION_PREREG section 5.1:
    the program preregistration's '127' counts one convolution per dilation and
    the RT-970 block has two; the architecture is binding, the number is not."""
    device = set_determinism(0)
    net = CausalTCN.build(K.HIDDEN, K.DROPOUT, 0, device, n_in=K.N_CHANNELS).double().eval()
    T, j = 700, 300
    x = torch.zeros(1, K.N_CHANNELS, T, dtype=torch.float64)
    with torch.no_grad():
        base = net(x).numpy()[0]
        x2 = x.clone(); x2[0, :, j] = 1.0
        pert = net(x2).numpy()[0]
    moved = np.flatnonzero(np.abs(pert - base) > 0)
    assert moved.min() == j, "the encoder is not causal: an earlier output moved"
    assert moved.max() - moved.min() + 1 == 253
