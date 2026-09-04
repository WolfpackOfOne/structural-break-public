"""StreamM19Lskd / StreamM19Lskm -- incremental twins of the RT-1321 blocks.

The batch modules `sbr.features.m19_lskd` are the specification.  Row ``t`` of
``REGISTRY["m19_lskd"].fn(make_ctx(hist, online))`` must be reproduced exactly
(``atol = 0``, ``NaN == NaN``) by ``step`` after ``StreamCtx.push``.

WHY BITWISE IS REACHABLE HERE
-----------------------------
1. **The historical half is literally the same code.**  `fit_historical` calls
   the batch module's own `_streams`, `history_delay_matrix`, `bandwidth`,
   `phi_*`, `_ewma_discrepancy` and `robust_center_scale` on the historical
   arrays.  There is one definition of the bandwidth, the frozen kernel mean and
   the frozen normalization constants, and no second implementation to drift.

2. **The feature map is row-count independent by construction.**  The batch
   `phi_joint` accumulates the projection as ``d`` explicit rank-1 updates
   instead of ``Z @ W.T``, precisely because BLAS ``gemm`` regroups the inner sum
   differently for one row than for many.  Calling the same function on a
   ``(1, d)`` row therefore returns the same floats batch computed.

3. **The EWMA is ONE function, called with a carried state.**  `_ewma_run`
   updates ``mu`` in place; batch calls it once over the whole online segment,
   this module calls it once per observation carrying ``mu`` forward.  Same
   arithmetic, same order.

4. **The summaries are running recursions, not window scans.**  Page
   accumulation, running peak, drawdown and persistence are all expressible from
   the current value plus O(1) carried state, and are written here as the
   identity batch's cumulative form evaluates to.

STATE AND COST PER OBSERVATION
------------------------------
Carried state: the frozen bandwidths, RFF bases (module-level constants), frozen
kernel means (2 streams x 3 depths x R), frozen (median, MAD) pairs
(2 x 3 x 2), the online EWMA means (2 x 3 x 2 x R), the last ``d_max - 1`` PIT
values per stream, and four scalars per aggregate path.  Everything is bounded;
nothing grows with ``t`` and the online prefix is never rescanned.

Work per observation: ``2 x 3`` RFF maps of ``R = 32`` cosines plus ``2 x 3 x 2``
EWMA updates of length ``R`` -- O(sum over depths of R*d), as specified.
"""
from __future__ import annotations

import numpy as np

from sbr.features.m19_lskd import (
    CLIP,
    COLS,
    DEPTHS,
    DMAX,
    HALFLIVES,
    LAM,
    PAGE_DRIFT,
    PERSIST_LEVEL,
    R,
    STREAMS,
    _ewma_discrepancy,
    _ewma_run,
    hist_streams,
    bandwidth,
    history_delay_matrix,
    phi_joint,
    phi_marginal,
    robust_center_scale,
)


class _StreamLskdBase:
    """Shared machinery; the subclasses differ only in the feature map."""

    MODULE = "m19_lskd"
    JOINT = True

    def __init__(self):
        self._reset()

    def _reset(self):
        self.sigma: dict[tuple[str, int], float] = {}
        self.mu_H: dict[tuple[str, int], np.ndarray] = {}
        self.norm: dict[tuple[str, int, int], tuple[float, float]] = {}
        self.mu: dict[tuple[str, int, int], np.ndarray] = {}
        self.tail: dict[str, np.ndarray] = {}
        # Page state is carried as batch's OWN intermediates -- the running
        # cumulative sum and its running minimum -- not as the algebraically
        # equivalent max(0, S+a-k) recursion.  The two agree mathematically and
        # disagree in floating point once the accumulator has been clipped, and
        # batch is the specification.
        self.page_c: dict[tuple[str, int], float] = {}
        self.page_m: dict[tuple[str, int], float] = {}
        self.peak: dict[tuple[str, int], float] = {}
        self.hits: dict[tuple[str, int], int] = {}
        self.sref: np.ndarray | None = None
        self.use_resid = False
        self._t = -1
        self._scratch = np.empty(1, dtype=np.float64)

    @property
    def cols(self) -> list[str]:
        return list(COLS)

    def _phi(self, Z, d, sigma):
        return phi_joint(Z, d, sigma) if self.JOINT else phi_marginal(Z, d, sigma)

    # ------------------------------------------------------------------ fit
    def fit_historical(self, ctx) -> None:
        self._reset()
        st, sref = hist_streams(ctx)
        self.use_resid = sref is not None
        self.sref = sref
        self.n_sref = len(sref) if sref is not None else 0

        for s in STREAMS:
            uh = st[s]
            self.tail[s] = np.asarray(uh[-(DMAX - 1):], dtype=np.float64).copy() \
                if len(uh) >= DMAX - 1 else np.concatenate(
                    [np.full(DMAX - 1 - len(uh), 0.5), np.asarray(uh, dtype=np.float64)])
            for d in DEPTHS:
                Zh = history_delay_matrix(uh, d)
                sigma = bandwidth(Zh, d)
                self.sigma[(s, d)] = sigma
                if Zh.shape[0] == 0:
                    self.mu_H[(s, d)] = np.zeros(R, dtype=np.float64)
                    for h in HALFLIVES:
                        self.norm[(s, d, h)] = None          # degenerate -> emit 0.0
                        self.mu[(s, d, h)] = np.zeros(R, dtype=np.float64)
                    continue
                Ph = self._phi(Zh, d, sigma)
                mu_H = Ph.mean(axis=0)
                self.mu_H[(s, d)] = mu_H
                for h in HALFLIVES:
                    self.norm[(s, d, h)] = robust_center_scale(
                        _ewma_discrepancy(Ph, mu_H, LAM[h]), h)
                    self.mu[(s, d, h)] = mu_H.copy()
            for h in HALFLIVES:
                self.page_c[(s, h)] = 0.0
                self.page_m[(s, h)] = 0.0
                self.peak[(s, h)] = -np.inf
                self.hits[(s, h)] = 0
        self._t = -1

    # ----------------------------------------------------------------- step
    def _pit_now(self, ctx, s: str) -> float:
        """The current online PIT value of stream ``s``.

        Both branches read the value the batch module reads: `ctx.last("u")` is
        `ctx.tr["u"][t]`, and the residual branch reproduces
        `m02_dist._pit_against` for a single scalar.
        """
        if s == "u" or not self.use_resid:
            return float(ctx.last("u"))
        x = float(ctx.last("res_mean"))
        lo = int(np.searchsorted(self.sref, x, side="left"))
        hi = int(np.searchsorted(self.sref, x, side="right"))
        return (0.5 * (lo + hi) + 0.5) / (self.n_sref + 1.0)

    def step(self, ctx) -> np.ndarray:
        t = ctx.t
        self._t = t
        n_seen = float(t + 1)
        core: dict[tuple[str, int, int], float] = {}

        for s in STREAMS:
            u_t = self._pit_now(ctx, s)
            tail = self.tail[s]
            for d in DEPTHS:
                # z_t = [u_t, u_{t-1}, ..., u_{t-d+1}]; lags come off the tail
                Z = np.empty((1, d), dtype=np.float64)
                Z[0, 0] = u_t
                for k in range(1, d):
                    Z[0, k] = tail[DMAX - 1 - k]
                P = self._phi(Z, d, self.sigma[(s, d)])
                for h in HALFLIVES:
                    nz = self.norm[(s, d, h)]
                    if nz is None:
                        core[(s, d, h)] = 0.0
                        continue
                    _ewma_run(P, self.mu[(s, d, h)], self.mu_H[(s, d)],
                              LAM[h], self._scratch)
                    m, sc = nz
                    core[(s, d, h)] = float(
                        np.clip((np.log1p(self._scratch[0]) - m) / sc, -CLIP, CLIP))
            # shift the lag buffer AFTER every depth has consumed it
            tail[:-1] = tail[1:]
            tail[-1] = u_t

        out = [core[(s, d, h)] for s in STREAMS for d in DEPTHS for h in HALFLIVES]
        for s in STREAMS:
            for h in HALFLIVES:
                a = float(np.mean(np.asarray([core[(s, d, h)] for d in DEPTHS])))
                # batch: c = np.cumsum(A - k); S = c - min(minimum.accumulate(c), 0)
                c = self.page_c[(s, h)] + (a - PAGE_DRIFT)
                self.page_c[(s, h)] = c
                m = min(self.page_m[(s, h)], c)
                self.page_m[(s, h)] = m
                self.peak[(s, h)] = max(self.peak[(s, h)], a)
                self.hits[(s, h)] += int(a > PERSIST_LEVEL)
                pk = self.peak[(s, h)]
                out.append(float(np.log1p(c - min(m, 0.0))))
                out.append(pk)
                out.append(pk - a)
                out.append(self.hits[(s, h)] / n_seen)

        row = np.asarray(out, dtype=np.float64)
        row[~np.isfinite(row)] = np.nan
        return row


class StreamM19Lskd(_StreamLskdBase):
    """CANDIDATE: joint lag-vector RFF discrepancy."""

    MODULE = "m19_lskd"
    JOINT = True


class StreamM19Lskm(_StreamLskdBase):
    """CONTROL: coordinate-separable (marginal) RFF discrepancy."""

    MODULE = "m19_lskm"
    JOINT = False
