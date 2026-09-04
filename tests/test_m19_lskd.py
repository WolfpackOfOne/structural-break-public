"""RT-1321 LS-KD: causality, determinism and matched-control gates.

The candidate block ``m19_lskd`` and its mechanism-matched control ``m19_lskm``
differ in exactly one thing -- the random-Fourier feature map -- so anything the
two share cannot explain a difference between them.  These tests pin that
claim, and pin the three properties that make the block deployable at all:
bitwise prefix invariance, a frozen RFF basis, and no sight of ``tau`` or
``n_online``.

No competition data required; everything here is synthetic.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from sbr.features.base import REGISTRY, check_prefix_invariance, load_all, make_ctx
from sbr.features import m19_lskd as M

MODULES = ("m19_lskd", "m19_lskm")

#: Pinned SHA-256 over the concatenated RFF directions and phases.  If this
#: moves, every LS-KD feature value in every recorded RT-1321 result moves with
#: it, so a change here invalidates the experiment rather than merely failing a
#: test.  `RandomState` (NEP 19) is used precisely so this cannot drift with a
#: NumPy upgrade the way `default_rng` would.
RFF_BASIS_SHA256 = "ef89c1cc0834ce13442a321fafcb853a8cbf67205f63456bd37b933e6efd0580"


@pytest.fixture(scope="module", autouse=True)
def _registry():
    load_all()
    return REGISTRY


def _series(rng, n_hist=600, n_online=220, kind="plain", tau=90):
    h = rng.normal(size=n_hist)
    o = rng.normal(size=n_online)
    if kind == "shift":
        o[tau:] += 1.5
    elif kind == "scale":
        o[tau:] *= 2.5
    elif kind == "dep":                       # transition-law change, marginal ~fixed
        for i in range(1, n_online):
            o[i] += (0.8 if i >= tau else 0.0) * o[i - 1]
        o /= max(o.std(), 1e-9)
    elif kind == "outlier":
        o[n_online // 3] += 14.0
    elif kind == "heavy":
        o = o * rng.choice([1.0, 5.0], size=n_online, p=[0.9, 0.1])
    elif kind == "const_hist":
        h = np.zeros(n_hist)
    elif kind == "tiny_var_hist":
        h = h * 1e-12
    return h, o


# ------------------------------------------------------------------ 1. frozen
def test_rff_basis_is_frozen():
    assert M.rff_basis_sha256() == RFF_BASIS_SHA256, (
        "the RFF basis moved; every recorded RT-1321 number was computed against "
        f"{RFF_BASIS_SHA256} and is no longer reproducible")


def test_frozen_configuration():
    assert (M.SEED, M.R, M.DEPTHS, M.HALFLIVES, M.STREAMS) == (
        1321, 32, (3, 5, 8), (32, 128), ("u", "r"))
    # lambda_h = 1 - 2^(-1/h)
    for h in M.HALFLIVES:
        assert M.LAM[h] == pytest.approx(1.0 - 2.0 ** (-1.0 / h), rel=0, abs=0)
    assert len(M.COLS) == 28
    assert len(set(M.COLS)) == 28


# -------------------------------------------------------------- 2. causality
@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("n_online,kind", [
    (1, "plain"), (2, "plain"), (9, "plain"), (40, "outlier"), (120, "shift"),
    (220, "scale"), (300, "dep"), (260, "heavy"), (150, "const_hist"),
    (150, "tiny_var_hist"),
])
def test_bitwise_prefix_invariance(module, n_online, kind):
    """Row t may depend only on hist and online[:t+1] -- checked at atol=0."""
    rng = np.random.default_rng(abs(hash((module, n_online, kind))) % 2 ** 32)
    h, o = _series(rng, n_online=n_online, kind=kind, tau=min(90, n_online // 2))
    ok, msg = check_prefix_invariance(module, h, o, cuts=(1, 2, 5, 17, 63), atol=0.0)
    assert ok, msg


@pytest.mark.parametrize("module", MODULES)
def test_future_mutation_cannot_move_an_earlier_row(module):
    """Poison every observation after t and require rows 0..t to be identical."""
    rng = np.random.default_rng(7)
    h, o = _series(rng, n_online=200, kind="shift")
    _, full = REGISTRY[module].fn(make_ctx(h, o))
    for cut in (1, 13, 77, 150):
        poisoned = o.copy()
        poisoned[cut:] = 1e6 * rng.normal(size=len(o) - cut)
        _, other = REGISTRY[module].fn(make_ctx(h, poisoned))
        a, b = full[:cut], other[:cut]
        assert np.array_equal(a, b) or np.array_equal(
            np.nan_to_num(a, nan=-7.7), np.nan_to_num(b, nan=-7.7)), (
            f"{module}: future observations moved row < {cut}")


@pytest.mark.parametrize("module", MODULES)
def test_no_tau_or_n_online_tokens_reachable(module):
    """The module signature and its column names carry no boundary information."""
    fn = REGISTRY[module].fn
    assert list(inspect.signature(fn).parameters) == ["ctx"]
    forbidden = ("tau", "cut", "boundary", "break_at", "has_break", "label",
                 "target", "n_online", "n_hist", "final", "eligible")
    names, _ = fn(make_ctx(*_series(np.random.default_rng(1))))
    bad = [n for n in names if any(tok in n.lower() for tok in forbidden)]
    assert not bad, bad
    src = inspect.getsource(M)
    for tok in ("tau", "has_break", "n_online"):
        assert f"ctx.{tok}" not in src


@pytest.mark.parametrize("module", MODULES)
def test_length_of_online_does_not_change_shared_rows(module):
    """A longer online segment sharing a prefix must not move the shared rows.

    This is the `n_online` gate in feature space: a normalisation, grid or
    buffer sized by the total online length would show up here.
    """
    rng = np.random.default_rng(11)
    h, o = _series(rng, n_online=400, kind="scale", tau=120)
    _, short = REGISTRY[module].fn(make_ctx(h, o[:150]))
    _, long_ = REGISTRY[module].fn(make_ctx(h, o))
    assert np.array_equal(np.nan_to_num(short, nan=-7.7),
                          np.nan_to_num(long_[:150], nan=-7.7))


@pytest.mark.parametrize("module", MODULES)
def test_deterministic_replay(module):
    rng = np.random.default_rng(3)
    h, o = _series(rng, n_online=180, kind="dep")
    _, a = REGISTRY[module].fn(make_ctx(h, o))
    _, b = REGISTRY[module].fn(make_ctx(h, o))
    assert np.array_equal(np.nan_to_num(a, nan=-7.7), np.nan_to_num(b, nan=-7.7))


@pytest.mark.parametrize("module", MODULES)
def test_series_order_independence(module):
    """Building series B after A must give what building B alone gives."""
    rng = np.random.default_rng(5)
    ha, oa = _series(rng, n_online=140, kind="shift")
    hb, ob = _series(rng, n_online=210, kind="scale")
    _, solo = REGISTRY[module].fn(make_ctx(hb, ob))
    REGISTRY[module].fn(make_ctx(ha, oa))
    _, after = REGISTRY[module].fn(make_ctx(hb, ob))
    assert np.array_equal(np.nan_to_num(solo, nan=-7.7), np.nan_to_num(after, nan=-7.7))


@pytest.mark.parametrize("module", MODULES)
def test_no_nan_columns_on_ordinary_series(module):
    rng = np.random.default_rng(13)
    for kind in ("plain", "shift", "scale", "dep", "heavy"):
        h, o = _series(rng, n_online=250, kind=kind)
        _, A = REGISTRY[module].fn(make_ctx(h, o))
        assert A.shape == (250, 28)
        assert np.isfinite(A).all(), f"{module}/{kind}: non-finite output"


# --------------------------------------------------- 3. the matched control
def test_candidate_and_control_are_shape_matched():
    rng = np.random.default_rng(17)
    h, o = _series(rng, n_online=200, kind="dep")
    ctx = make_ctx(h, o)
    nc, A = REGISTRY["m19_lskd"].fn(ctx)
    nm, B = REGISTRY["m19_lskm"].fn(ctx)
    assert nc == nm and A.shape == B.shape


def test_control_is_separable_and_candidate_is_not():
    """The control's map is additive across lag coordinates; the candidate's is not.

    Permuting the lag coordinates of a delay vector leaves an additive map
    invariant and moves a joint one.  That is the whole mechanism difference,
    tested directly rather than asserted in a docstring.
    """
    rng = np.random.default_rng(19)
    d = 5
    Z = rng.uniform(0.0, 1.0, size=(64, d))
    Zp = Z[:, ::-1].copy()
    sigma = 0.7
    # additive: the per-coordinate contributions are a plain sum, so a coordinate
    # permutation applied jointly to directions and data is exactly recovered.
    A = M.phi_marginal(Z, d, sigma)
    A_perm = (M.SQRT2R / np.sqrt(d)) * np.cos(
        Zp[:, None, :] * (M.W_BASE[d][:, ::-1][None, :, :] / sigma)
        + M.B_COORD[d][:, ::-1][None, :, :]).sum(axis=2)
    assert np.allclose(A, A_perm), "control map is not coordinate-separable"

    J = M.phi_joint(Z, d, sigma)
    J_perm = M.phi_joint(Zp, d, sigma)
    assert not np.allclose(J, J_perm), "joint map ignored lag order -- it is not joint"


def test_candidate_and_control_differ_on_a_transition_law_change():
    """A dependence-only break must move the candidate more than the control.

    The series' marginal is held fixed by construction (the post-break half is
    rescaled to unit variance), so a purely marginal channel has little to see
    and a joint one does.
    """
    rng = np.random.default_rng(23)
    n_hist, n_online, tau = 1500, 400, 150
    h = rng.normal(size=n_hist)
    o = rng.normal(size=n_online)
    for i in range(tau, n_online):
        o[i] += 0.85 * o[i - 1]
    o[tau:] = (o[tau:] - o[tau:].mean()) / o[tau:].std()
    ctx = make_ctx(h, o)
    names, A = REGISTRY["m19_lskd"].fn(ctx)
    _, B = REGISTRY["m19_lskm"].fn(ctx)
    j = names.index("kd_u_d8_h128")
    late = slice(tau + 120, n_online)
    assert np.nanmean(A[late, j]) > np.nanmean(B[late, j]), (
        "joint LS-KD did not exceed its marginal control on a dependence-only break; "
        "the mechanism distinction is not doing what the hypothesis claims")
