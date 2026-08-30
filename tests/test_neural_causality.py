"""THE FIVE PRE-SCORE GATES for the wave-6 neural track.

research/WAVE6_NEURAL_PREREG.md section 4.1.  These run BEFORE any neural
TS-AUC is read, and a failure VOIDS the run rather than getting patched once the
score is known.

    GATE 1  prefix invariance          shared prefix, different futures -> same
    GATE 2  no n_online / final length  truncation cannot move a surviving row
    GATE 3  no true tau                 moving the break cannot move a pre-break row
    GATE 4  fold purity                 normalisation sees training rows only
    GATE 5  determinism                 same seed, same input -> same output

Plus the section-5 normalisation audit, which is the specific way this track
would fake a result: "standardise the sequence" is the default habit in deep
learning and it is a future leak here.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import os  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "research", "scripts"))
from wave6_neural_lib import (  # noqa: E402
    CHANNEL_NAMES,
    N_CHANNELS,
    PROD_MODULES,
    CausalTCN,
    FoldStandardiser,
    build_mlp,
    causal_channels,
    set_determinism,
)

from sbr.features.base import REGISTRY, load_all, make_ctx  # noqa: E402

TOL = 1e-8


@pytest.fixture(scope="module")
def loaded():
    load_all()
    return REGISTRY


def _series(rng, n_hist=1200, n_online=300, tau=None, scale=2.2, shift=0.8):
    h = rng.normal(size=n_hist)
    o = rng.normal(size=n_online)
    if tau is not None:
        o[tau:] = o[tau:] * scale + shift
    return h, o


def _tcn(hidden=32, seed=0):
    dev = set_determinism(seed)
    net = CausalTCN.build(hidden, 0.1, seed, dev)
    net.eval()
    return net


def _tcn_predict(net, h, o):
    C = causal_channels(h, o)
    with torch.no_grad():
        x = torch.from_numpy(C.T[None]).to(torch.float64)
        net64 = net.to(torch.float64)
        return net64(x)[0].numpy()


def _mlp_predict(net, std, feats):
    with torch.no_grad():
        X = torch.from_numpy(std.transform(feats)).to(torch.float64)
        return net.to(torch.float64)(X)[:, 0].numpy()


def _features(loaded, h, o):
    ctx = make_ctx(h, o)
    return np.hstack([np.asarray(loaded[m].fn(ctx)[1], dtype=np.float64)
                      for m in PROD_MODULES])


def _fitted_mlp(loaded, seed=0):
    rng = np.random.default_rng(99)
    h, o = _series(rng, n_online=400, tau=150)
    F = _features(loaded, h, o)
    sizes = [len(loaded[m].fn(make_ctx(h, o))[0]) for m in PROD_MODULES]
    std = FoldStandardiser(sizes).fit(F)
    dev = set_determinism(seed)
    net = build_mlp(F.shape[1] + len(sizes), 0.1, seed, dev)
    net.eval()
    return net, std


# ------------------------------------------------------------------- GATE 1
def test_gate1_prefix_invariance_channels():
    """The TCN's inputs must not move when the future changes."""
    rng = np.random.default_rng(1)
    h = rng.normal(size=1200)
    base = rng.normal(size=400)
    a = base.copy()
    b = base.copy()
    b[250:] = b[250:] * 4.0 - 3.0           # a completely different future
    Ca, Cb = causal_channels(h, a), causal_channels(h, b)
    assert np.array_equal(Ca[:250], Cb[:250]), "channel values depend on the future"


def test_gate1_prefix_invariance_tcn():
    net = _tcn()
    rng = np.random.default_rng(2)
    h = rng.normal(size=1200)
    base = rng.normal(size=400)
    a = base.copy()
    b = base.copy()
    b[250:] = b[250:] * 4.0 - 3.0
    pa, pb = _tcn_predict(net, h, a), _tcn_predict(net, h, b)
    d = np.abs(pa[:250] - pb[:250]).max()
    assert d <= TOL, f"TCN prefix invariance violated by {d:.3e}"


def test_gate1_prefix_invariance_mlp(loaded):
    net, std = _fitted_mlp(loaded)
    rng = np.random.default_rng(3)
    h = rng.normal(size=1200)
    base = rng.normal(size=400)
    a = base.copy()
    b = base.copy()
    b[250:] = b[250:] * 4.0 - 3.0
    pa = _mlp_predict(net, std, _features(loaded, h, a))
    pb = _mlp_predict(net, std, _features(loaded, h, b))
    d = np.abs(pa[:250] - pb[:250]).max()
    assert d <= TOL, f"MLP prefix invariance violated by {d:.3e}"


# ------------------------------------------------------------------- GATE 2
def test_gate2_no_n_online_dependence_tcn():
    """Truncating the stream cannot move a row that survives the truncation."""
    net = _tcn()
    rng = np.random.default_rng(4)
    h = rng.normal(size=1200)
    o = rng.normal(size=500)
    full = _tcn_predict(net, h, o)
    for cut in (37, 120, 301, 499):
        part = _tcn_predict(net, h, o[:cut])
        d = np.abs(full[:cut] - part).max()
        assert d <= TOL, f"TCN saw the final length: cut={cut}, delta={d:.3e}"


def test_gate2_no_n_online_dependence_mlp(loaded):
    net, std = _fitted_mlp(loaded)
    rng = np.random.default_rng(5)
    h = rng.normal(size=1200)
    o = rng.normal(size=400)
    full = _mlp_predict(net, std, _features(loaded, h, o))
    for cut in (40, 199, 399):
        part = _mlp_predict(net, std, _features(loaded, h, o[:cut]))
        d = np.abs(full[:cut] - part).max()
        assert d <= TOL, f"MLP saw the final length: cut={cut}, delta={d:.3e}"


def test_gate2_elapsed_channel_is_a_counter_not_a_length():
    """`elapsed` must be log1p(t)/7 -- identical whatever the series length is."""
    rng = np.random.default_rng(6)
    h = rng.normal(size=1200)
    o = rng.normal(size=600)
    short = causal_channels(h, o[:100])[:, CHANNEL_NAMES.index("elapsed")]
    long = causal_channels(h, o)[:, CHANNEL_NAMES.index("elapsed")]
    assert np.array_equal(short, long[:100])
    assert np.allclose(long, np.log1p(np.arange(600)) / 7.0, atol=1e-6)


# ------------------------------------------------------------------- GATE 3
def test_gate3_no_tau_dependence_tcn():
    """Two series identical up to t, breaking at different places, agree up to t."""
    net = _tcn()
    rng = np.random.default_rng(7)
    h = rng.normal(size=1200)
    base = rng.normal(size=400)

    def with_tau(tau):
        o = base.copy()
        o[tau:] = o[tau:] * 2.5 + 1.0
        return _tcn_predict(net, h, o)

    pa, pb = with_tau(180), with_tau(320)
    d = np.abs(pa[:180] - pb[:180]).max()
    assert d <= TOL, f"TCN prediction depends on where the break is: {d:.3e}"


def test_gate3_channels_contain_no_tau_named_thing():
    assert len(CHANNEL_NAMES) == N_CHANNELS
    for c in CHANNEL_NAMES:
        low = c.lower()
        for tok in ("tau", "cut", "boundary", "post", "label", "target", "n_online"):
            assert tok not in low, f"channel {c} names a forbidden concept"


def test_gate3_group_indicators_carry_no_within_timestep_information(loaded):
    """The MLP's 7 missingness indicators, scored with the OFFICIAL scorer.

    This is the RT-900 signature applied to the one channel the prereg adds on
    top of the 500 columns.  Pooled correlation is not the test -- a mask that
    is a function of t alone is constant within a timestep and carries nothing.
    """
    from sbr.metric import ts_auc_flat
    rng = np.random.default_rng(2026)
    _sizes, IND, ys, ts = None, [], [], []
    for k in range(18):
        n_online = int(rng.integers(150, 400))
        broke = k % 3 != 0
        tau = int(rng.integers(20, n_online - 20)) if broke else n_online + 1
        h, o = _series(rng, n_hist=1200, n_online=n_online,
                       tau=tau if broke else None)
        ctx = make_ctx(h, o)
        parts, sz = [], []
        for m in PROD_MODULES:
            names, A = loaded[m].fn(ctx)
            parts.append(np.asarray(A, dtype=np.float64))
            sz.append(len(names))
        F = np.hstack(parts)
        bad = ~np.isfinite(F)
        off, ind = 0, np.empty((len(F), len(sz)))
        for g, w in enumerate(sz):
            ind[:, g] = bad[:, off:off + w].any(1)
            off += w
        IND.append(ind)
        ys.append((np.arange(n_online) >= tau).astype(np.int8))
        ts.append(np.arange(n_online, dtype=np.int64))
    IND = np.vstack(IND)
    y = np.concatenate(ys)
    t = np.concatenate(ts)
    worst = (None, 0.5)
    for g in range(IND.shape[1]):
        col = IND[:, g]
        if col.min() == col.max():
            continue
        a = float(ts_auc_flat(col, y, t))
        if abs(a - 0.5) > abs(worst[1] - 0.5):
            worst = (PROD_MODULES[g], a)
    assert abs(worst[1] - 0.5) <= 0.10, (
        f"group indicator for {worst[0]} scores TS-AUC {worst[1]:.5f} alone -- "
        "this is the RT-900 failure mode")


# ------------------------------------------------------------------- GATE 4
def test_gate4_standardiser_is_fold_pure():
    """Constants must be a function of the fitted rows and nothing else."""
    rng = np.random.default_rng(8)
    A = rng.normal(size=(4000, 20)).astype(np.float32)
    B = rng.normal(size=(4000, 20)).astype(np.float32) * 9.0 + 40.0
    s1 = FoldStandardiser([20]).fit(A)
    s2 = FoldStandardiser([20]).fit(A)
    assert np.array_equal(s1.med, s2.med) and np.array_equal(s1.iqr, s2.iqr)
    s3 = FoldStandardiser([20]).fit(np.vstack([A, B]))
    assert not np.allclose(s1.med, s3.med), "fitting on more rows changed nothing -- suspicious"
    # transforming a held-out block must not alter the fitted constants
    med, iqr = s1.med.copy(), s1.iqr.copy()
    s1.transform(B)
    assert np.array_equal(s1.med, med) and np.array_equal(s1.iqr, iqr)


def test_gate4_standardiser_survives_degenerate_columns():
    X = np.zeros((500, 6), dtype=np.float32)
    X[:, 1] = np.nan
    X[:, 2] = np.inf
    s = FoldStandardiser([6]).fit(X)
    Z = s.transform(X)
    assert np.isfinite(Z).all()
    assert Z.shape == (500, 7)


# ------------------------------------------------------------------- GATE 5
def test_gate5_determinism_tcn():
    a = _tcn_predict(_tcn(hidden=32, seed=0), *_series(np.random.default_rng(9)))
    b = _tcn_predict(_tcn(hidden=32, seed=0), *_series(np.random.default_rng(9)))
    assert np.abs(a - b).max() <= TOL


def test_gate5_determinism_mlp(loaded):
    n1, s1 = _fitted_mlp(loaded, seed=0)
    n2, s2 = _fitted_mlp(loaded, seed=0)
    rng = np.random.default_rng(10)
    h, o = _series(rng, n_online=250, tau=100)
    F = _features(loaded, h, o)
    assert np.abs(_mlp_predict(n1, s1, F) - _mlp_predict(n2, s2, F)).max() <= TOL


def test_gate5_batch_composition_does_not_change_a_prediction():
    """Sequences are batched together; a series' output must not depend on its
    neighbours.  This is what makes length-bucketed batching legal instead of a
    covert length channel."""
    net = _tcn()
    rng = np.random.default_rng(11)
    h = rng.normal(size=1200)
    seqs = [rng.normal(size=n) for n in (120, 300, 460)]
    alone = [_tcn_predict(net, h, s) for s in seqs]
    T = max(len(s) for s in seqs)
    X = np.zeros((3, N_CHANNELS, T))
    for i, s in enumerate(seqs):
        X[i, :, :len(s)] = causal_channels(h, s).T
    with torch.no_grad():
        P = net.to(torch.float64)(torch.from_numpy(X)).numpy()
    for i, s in enumerate(seqs):
        d = np.abs(P[i, :len(s)] - alone[i]).max()
        assert d <= TOL, f"batching moved series {i} by {d:.3e}"


# --------------------------------------------------- section 5: normalisation
def test_normalisation_uses_no_future_statistic():
    """Scaling a series' tail must not change any earlier channel value.

    A full-online mean/std, a final min/max or a sequence LayerNorm across time
    would all fail this. It is the single most likely way this track fakes a
    result, so it is tested directly rather than argued.
    """
    rng = np.random.default_rng(12)
    h = rng.normal(size=1200)
    o = rng.normal(size=500)
    C0 = causal_channels(h, o)
    o2 = o.copy()
    o2[300:] *= 100.0
    C1 = causal_channels(h, o2)
    assert np.array_equal(C0[:300], C1[:300])


def test_normalisation_constants_come_from_history_only():
    """Shifting the ONLINE segment changes z; shifting HISTORY rescales it.
    If the reverse were true the channels would be self-normalising over the
    online window, which is the leak."""
    rng = np.random.default_rng(13)
    h = rng.normal(size=1200)
    o = rng.normal(size=200)
    z0 = causal_channels(h, o)[:, 0]
    z1 = causal_channels(h, o + 5.0)[:, 0]
    assert not np.allclose(z0, z1), "online shift did not move z -- self-normalising"
    z2 = causal_channels(h + 5.0, o + 5.0)[:, 0]
    assert np.allclose(z0, z2, atol=1e-9), "history is not the normalisation source"


def test_tcn_has_no_time_axis_normalisation_layer():
    net = _tcn()
    from torch import nn
    for m in net.modules():
        assert not isinstance(m, (nn.BatchNorm1d, nn.InstanceNorm1d)), type(m)
        if isinstance(m, nn.LayerNorm):
            pytest.fail("LayerNorm inside the TCN would normalise across time")


def test_tcn_convolutions_are_all_causal():
    net = _tcn()
    from torch import nn
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            assert m.padding in (0, (0,)), f"conv has built-in padding {m.padding}"
        assert not isinstance(m, (nn.GRU, nn.LSTM, nn.RNN)), "no recurrent layer here"
