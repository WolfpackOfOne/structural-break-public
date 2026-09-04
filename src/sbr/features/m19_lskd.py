"""m19_lskd / m19_lskm -- lag-space characteristic-kernel discrepancy (RT-1321).

WHAT IS ACTUALLY MISSING FROM THE INCUMBENT BANK
    `m02_dist` computes chi2, JS, Hellinger, TV, Cramer-von Mises, KS,
    1-D Wasserstein and energy distance, and `m12_rdep` repeats the family on
    the AR-residual stream.  Energy distance IS a maximum-mean-discrepancy with
    the distance kernel -- but every one of those distances is taken on the
    MARGINAL, one-dimensional PIT.  Nothing in the bank compares the JOINT law
    of a short consecutive state

        P(U_t, U_{t-1}, ..., U_{t-d+1})

    against the corresponding historical joint law.  `research/new_avenues_2026.csv`
    row G5 says exactly this ("every incumbent distance is on the MARGINAL (1-D);
    the delay-embedded cloud is joint over d lags"), rates it HIGH, and it was
    never executed.  This module is that execution.

SIGNAL / ALPHA
    Breaks that move the transition law while leaving the one-dimensional
    marginal nearly intact: nonlinear transition structure, conditional
    heteroskedasticity, tail clustering, multimodal transition behaviour,
    nonlinear lag interaction, a change of dependence invisible to a handful of
    ACF/PACF coefficients.  A characteristic (Gaussian) kernel embeds the whole
    joint law, so the discrepancy is zero if and only if the joint laws agree.

WHY THIS IS NOT RT-1215 (Hankel-DMD, KILL)
    RT-1215 fits a LINEAR delay-subspace operator on history and monitors
    reconstruction error, subspace angle and effective rank -- it asks whether
    the online delay cloud still lies in a linear subspace.  This module asks a
    different question: whether the DISTRIBUTION of the delay cloud, in an RKHS,
    still matches history.  Two laws can share a linear subspace and a spectrum
    and still differ.  `NEGATIVE_RESULTS_INDEX.md` records for the RT-1215 row
    that kernel / explicitly nonlinear variants are "genuinely NOT covered".

THE MATCHED CONTROL LIVES IN THIS FILE ON PURPOSE
    `m19_lskm` shares the inputs, the depths, the half-lives, the RFF dimension,
    the frozen bandwidth, the historical reference construction, the history-only
    normalization, the summary geometry and the column count.  The ONLY
    difference is the feature map: `m19_lskd` projects the JOINT lag vector
    (cross-coordinate interaction terms present), `m19_lskm` is an equal
    combination of independent 1-D coordinate maps, whose induced kernel is
    additive across lag coordinates and therefore carries only coordinatewise
    MARGINAL distribution information.  Both are causal.  Anything the two
    blocks share cannot explain a difference between them.

CAUSALITY
    Bandwidth, historical kernel mean and normalization constants are fitted on
    the HISTORICAL segment only and frozen.  The online recursion is an EWMA, so
    row t is a function of `hist` and `online[:t+1]` and of nothing else.  There
    is no dependence on `tau`, on `n_online`, on the final state or on any other
    series.  Bitwise prefix-invariant, checked at atol=0.
"""
from __future__ import annotations

import hashlib

import numpy as np

from sbr.features.base import register
from sbr.features.m02_dist import _pit_against

# --------------------------------------------------------------- frozen config
#: RT-1321.  Never tuned, never searched.
SEED = 1321
R = 32                                   #: random Fourier features per map
DEPTHS = (3, 5, 8)                       #: embedding depths d
HALFLIVES = (32, 128)                    #: online EWMA half-lives h
STREAMS = ("u", "r")                     #: raw PIT, AR-residual PIT
BW_SUBSAMPLE = 192                       #: history vectors used for the median heuristic
BW_LO, BW_HI = 0.05, 2.0                 #: bandwidth clamp, in units of sqrt(d)
PAGE_DRIFT = 1.0                         #: Page accumulation drift k
PERSIST_LEVEL = 2.0                      #: "above a history-calibrated level"
CLIP = 8.0
EPS = 1e-9

DMAX = max(DEPTHS)


def _draw_bases():
    """Deterministic RFF bases.

    `RandomState` rather than `default_rng`: NEP 19 guarantees the legacy
    stream is stable forever across NumPy versions, and `Generator` explicitly
    does not.  A submission whose feature values depend on the host's NumPy
    minor version is not deployable.  `test_m19_lskd.py` pins a SHA over the
    concatenated arrays.
    """
    rs = np.random.RandomState(SEED)
    W, bj, Bc = {}, {}, {}
    for d in DEPTHS:                                   # fixed order: 3, 5, 8
        W[d] = rs.standard_normal((R, d))              # base directions
        bj[d] = rs.uniform(0.0, 2.0 * np.pi, R)        # joint phases   (candidate)
        Bc[d] = rs.uniform(0.0, 2.0 * np.pi, (R, d))   # per-coord phases (control)
    return W, bj, Bc


W_BASE, B_JOINT, B_COORD = _draw_bases()
SQRT2R = np.sqrt(2.0 / R)


def rff_basis_sha256() -> str:
    h = hashlib.sha256()
    for d in DEPTHS:
        for a in (W_BASE[d], B_JOINT[d], B_COORD[d]):
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()


def lam(h: int) -> float:
    """EWMA coefficient for half-life h:  lambda = 1 - 2^(-1/h)."""
    return 1.0 - 2.0 ** (-1.0 / float(h))


LAM = {h: lam(h) for h in HALFLIVES}


# ------------------------------------------------------------------- primitives
def delay_matrix(u: np.ndarray, warm: np.ndarray, d: int) -> np.ndarray:
    """(len(u), d) causal delay vectors [u_t, u_{t-1}, ..., u_{t-d+1}].

    ``warm`` is the tail of the stream that physically precedes ``u`` (the
    historical PIT tail when ``u`` is the online PIT).  A store series is one
    contiguous series split into hist | online, so warm-starting from history is
    exactly as legal as the AR filter's `ar_filter_causal` warm start, and it
    means row 0 exists rather than being NaN.
    """
    u = np.asarray(u, dtype=np.float64)
    n = len(u)
    pad = d - 1
    if pad == 0:
        return u[:, None]
    if len(warm) >= pad:
        head = np.asarray(warm[-pad:], dtype=np.float64)
    else:
        head = np.full(pad, 0.5)
        if len(warm):
            head[pad - len(warm):] = warm
    ext = np.concatenate([head, u])
    out = np.empty((n, d), dtype=np.float64)
    for k in range(d):                       # column k is lag k
        out[:, k] = ext[pad - k: pad - k + n]
    return out


def history_delay_matrix(uh: np.ndarray, d: int) -> np.ndarray:
    """Every FULLY OBSERVED history delay vector (no warm start, no padding)."""
    uh = np.asarray(uh, dtype=np.float64)
    m = len(uh) - d + 1
    if m <= 0:
        return np.zeros((0, d), dtype=np.float64)
    out = np.empty((m, d), dtype=np.float64)
    for k in range(d):
        out[:, k] = uh[d - 1 - k: d - 1 - k + m]
    return out


def bandwidth(Zh: np.ndarray, d: int) -> float:
    """History-only deterministic median-distance heuristic, clamped."""
    lo, hi = BW_LO * np.sqrt(d), BW_HI * np.sqrt(d)
    m = Zh.shape[0]
    if m < 2:
        return float(lo)
    if m > BW_SUBSAMPLE:
        idx = np.linspace(0, m - 1, BW_SUBSAMPLE).astype(np.intp)
        S = Zh[idx]
    else:
        S = Zh
    g = S @ S.T
    sq = np.diag(g)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * g, 0.0)
    iu = np.triu_indices(len(S), k=1)
    med = float(np.median(np.sqrt(d2[iu])))
    return float(min(max(med, lo), hi))


def phi_joint(Z: np.ndarray, d: int, sigma: float) -> np.ndarray:
    """sqrt(2/R) cos(omega_j' z + b_j) over the JOINT lag vector.  (n, R)

    The projection is accumulated as ``d`` explicit rank-1 updates rather than
    written ``Z @ W.T``.  That is deliberate and load-bearing: BLAS ``gemm``
    regroups the length-``d`` inner sum differently depending on how many rows
    it is given, so the matrix form disagrees with a one-row call in the last
    ulp on ~25% of entries (measured: `test_m19_lskd.py::
    test_row_batch_projection_is_bitwise_identical`).  A streaming module sees
    exactly one row per step, so with the matrix form batch/stream parity would
    be unreachable, and this repository has already had a single-ulp feature
    defect move a shipped prediction.  The explicit accumulation is elementwise,
    therefore independent of the row count, therefore bitwise reproducible from
    a stream.
    """
    acc = np.repeat(B_JOINT[d][None, :], Z.shape[0], axis=0)
    Wt = W_BASE[d].T / sigma                      # (d, R)
    for k in range(d):
        acc += Z[:, k:k + 1] * Wt[k]
    return SQRT2R * np.cos(acc)


def phi_marginal(Z: np.ndarray, d: int, sigma: float) -> np.ndarray:
    """Coordinate-separable map: equal combination of independent 1-D maps.

    phi_j(z) = sqrt(2/R) * (1/sqrt(d)) * sum_c cos(w_{j,c} z_c + b_{j,c}).
    The induced kernel is (1/d) sum_c k_1(z_c, z'_c) -- additive across lag
    coordinates, so it sees coordinatewise marginal distribution change and no
    cross-coordinate interaction.
    """
    A = Z[:, None, :] * (W_BASE[d][None, :, :] / sigma) + B_COORD[d][None, :, :]
    # sum over the last (contiguous, length-d) axis: numpy uses the same
    # sequential order regardless of the leading dimension, so this is already
    # row-count independent.  Pinned by the same parity test as `phi_joint`.
    return (SQRT2R / np.sqrt(d)) * np.cos(A).sum(axis=2)


def _ewma_run(P: np.ndarray, mu: np.ndarray, mu_H: np.ndarray, lm: float,
              out: np.ndarray) -> None:
    """Advance the EWMA over the rows of ``P``, filling ``out`` with ||mu-mu_H||^2.

    ``mu`` is updated IN PLACE, so the batch path (one call over the whole
    online segment) and the streaming path (one call per observation, carrying
    ``mu`` across calls) execute the identical arithmetic in the identical
    order.  There is one definition of the recursion and no second
    implementation to drift.  The squared norm is accumulated with an explicit
    scalar loop rather than ``e @ e`` for the same reason ``phi_joint`` avoids
    ``gemm``: BLAS ``ddot`` may regroup the sum.
    """
    n = P.shape[0]
    k = mu.shape[0]
    one = 1.0 - lm
    for i in range(n):
        s = 0.0
        for j in range(k):
            mu[j] = one * mu[j] + lm * P[i, j]
            e = mu[j] - mu_H[j]
            s += e * e
        out[i] = s


try:                                                     # pragma: no cover
    from numba import njit

    _ewma_run = njit(cache=True, fastmath=False)(_ewma_run)
except Exception:                                        # pragma: no cover
    pass


def _ewma_discrepancy(P: np.ndarray, mu_H: np.ndarray, lm: float) -> np.ndarray:
    """Batch convenience wrapper: run the recursion from mu = mu_H."""
    out = np.empty(P.shape[0], dtype=np.float64)
    _ewma_run(np.ascontiguousarray(P), mu_H.copy(), mu_H, lm, out)
    return out


def robust_center_scale(D_hist: np.ndarray, h: int) -> tuple[float, float]:
    """Median / MAD of log1p(D) over a history replay, after a burn-in cut.

    The online path starts at mu_H too, so its transient matches the replay's;
    the burn-in is dropped anyway so the frozen null describes the STATIONARY
    regime the dominant cell (t >= 200) actually lives in, instead of being
    pulled down by a transient whose length would otherwise vary with n_hist.
    """
    L = len(D_hist)
    if L == 0:
        return 0.0, 1.0
    burn = min(4 * h, L // 4)
    g = np.log1p(D_hist[burn:]) if L - burn >= 16 else np.log1p(D_hist)
    m = float(np.median(g))
    s = 1.4826 * float(np.median(np.abs(g - m)))
    if not np.isfinite(s) or s < EPS:
        s = float(np.std(g))
    return m, max(s, EPS)


def _page(v: np.ndarray, k: float = PAGE_DRIFT) -> np.ndarray:
    """S_t = max(0, S_{t-1} + v_t - k): one cumsum plus a running minimum.

    The running minimum INCLUDES index t, which is what makes the identity
    ``S_t = c_t - min(0, c_0..c_t)`` hold and keeps the output non-negative.
    `m12_rdep._reflect_cusum` excludes t and can therefore go negative; that is
    harmless there because the result is fed straight into a robust z, but here
    it is passed through ``log1p`` and a negative value below -1 is NaN.
    """
    c = np.cumsum(v - k)
    return c - np.minimum(np.minimum.accumulate(c), 0.0)


# ----------------------------------------------------------------- the streams
def hist_streams(ctx) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """Historical PIT of each input stream, plus the residual PIT reference.

    Split out of `_streams` so the STREAMING module can fit its history without
    also evaluating the (not yet populated) online buffers -- there is still one
    definition of each historical object, which is what parity depends on.

    Both constructions are the repository's own: `HistParams.pit` via
    `ctx.hist_tr["u"]`, and `m02_dist`'s AR-residual PIT block reproduced by
    importing that module's `_pit_against`.  Availability is decided by
    `nh - p_ar > 50`, a property of the HISTORY alone, so it can never depend on
    the online segment.
    """
    uh = np.asarray(ctx.hist_tr["u"], dtype=np.float64)
    p_ar = len(ctx.hp.ar_coef)
    nh = len(ctx.hist)
    if "res_mean" in ctx.hist_tr and nh - p_ar > 50:
        rh = np.asarray(ctx.hist_tr["res_mean"], dtype=np.float64)[p_ar:]
        sref = np.sort(rh)
        return {"u": uh, "r": _pit_against(sref, rh)}, sref
    return {"u": uh, "r": uh}, None


def _streams(ctx) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(historical PIT, online PIT) for each input stream."""
    H, sref = hist_streams(ctx)
    uo = np.asarray(ctx.tr["u"], dtype=np.float64)
    if sref is None:
        return {"u": (H["u"], uo), "r": (H["r"], uo)}
    uro = _pit_against(sref, np.asarray(ctx.tr["res_mean"], dtype=np.float64))
    return {"u": (H["u"], uo), "r": (H["r"], uro)}


def _cols() -> list[str]:
    cols = [f"kd_{s}_d{d}_h{h}" for s in STREAMS for d in DEPTHS for h in HALFLIVES]
    for s in STREAMS:
        for h in HALFLIVES:
            for tag in ("page", "pk", "dd", "per"):
                cols.append(f"ag_{s}_h{h}_{tag}")
    return cols


COLS = _cols()


def lskd_features(ctx, joint: bool) -> np.ndarray:
    """(n_online, 28) float32.  ``joint`` selects candidate vs matched control."""
    n = ctx.n
    L = np.arange(1, n + 1, dtype=np.float64)
    phi = phi_joint if joint else phi_marginal
    st = _streams(ctx)

    core: dict[tuple[str, int, int], np.ndarray] = {}
    for s in STREAMS:
        uh, uo = st[s]
        for d in DEPTHS:
            Zh = history_delay_matrix(uh, d)
            sigma = bandwidth(Zh, d)
            if Zh.shape[0] == 0:
                for h in HALFLIVES:
                    core[(s, d, h)] = np.zeros(n, dtype=np.float64)
                continue
            Ph = phi(Zh, d, sigma)
            mu_H = Ph.mean(axis=0)
            Zo = delay_matrix(uo, uh, d)
            Po = phi(Zo, d, sigma)
            for h in HALFLIVES:
                lm = LAM[h]
                m, sc = robust_center_scale(_ewma_discrepancy(Ph, mu_H, lm), h)
                Do = _ewma_discrepancy(Po, mu_H, lm)
                core[(s, d, h)] = np.clip((np.log1p(Do) - m) / sc, -CLIP, CLIP)

    out = [core[(s, d, h)] for s in STREAMS for d in DEPTHS for h in HALFLIVES]
    for s in STREAMS:
        for h in HALFLIVES:
            A = np.mean([core[(s, d, h)] for d in DEPTHS], axis=0)
            pk = np.maximum.accumulate(A)
            out.append(np.log1p(_page(A)))
            out.append(pk)
            out.append(pk - A)
            out.append(np.cumsum(A > PERSIST_LEVEL) / L)

    M = np.column_stack(out).astype(np.float32)
    M[~np.isfinite(M)] = np.nan
    return M


@register("m19_lskd", version="1", owner="rt1321")
def build_lskd(ctx):
    """CANDIDATE: RFF discrepancy of the JOINT lag-vector law."""
    return list(COLS), lskd_features(ctx, joint=True)


@register("m19_lskm", version="1", owner="rt1321")
def build_lskm(ctx):
    """CONTROL: coordinate-separable (marginal) RFF discrepancy, same everything else."""
    return list(COLS), lskd_features(ctx, joint=False)
