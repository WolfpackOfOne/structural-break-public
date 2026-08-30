"""StreamCtx -- the ONE shared incremental feature state.

Design contract
---------------
`StreamCtx` reproduces, incrementally and exactly, the state that
`sbr.features.base.make_ctx` builds in batch:

    fit_historical(hist)          <->  HistParams + hist transforms + NullCal
    push(x_t)  for t = 0,1,2,...  <->  ctx.tr[name][t], ctx.cum[name][t+1]

and exposes the same read API (`roll`, `expand`, `nc`, `hp`) evaluated *at the
current time index only*.  Feature modules then have a `step()` that computes
row `t` using the same scalar arithmetic the batch module applies to row `t`.

Bitwise identity is the target, and it is achievable because:
  * the historical half is literally the same code on the same input;
  * every per-point transform is elementwise, so evaluating it on a 3-point
    window and taking the last element gives the same float as evaluating it on
    the whole array (we reuse `build_transforms` itself, so there is no second
    implementation to drift);
  * `np.cumsum` accumulates sequentially, so `c[t+1] = c[t] + v[t]` is exactly
    what batch computes;
  * `roll`/`expand` are the same two expressions, evaluated at one index.

The one place that needs care is the AR residual: batch multiplies by
`ar_sigma` and then `build_transforms` divides by it again, which is NOT the
identity in floating point.  We replicate the multiply-then-divide exactly.

Cost per observation: O(1) arithmetic + O(log n_hist) per null-calibration
query.  Memory per series: O(n_online) float64 per transform (23 transforms x
1000 points x 8 B ~ 184 kB) -- the cumulative-sum arrays are kept in full
because trailing windows of length up to 256 and expanding windows both read
them, and 1000 points is the competition's hard cap.
"""
from __future__ import annotations

import numpy as np

from sbr.nullcal import NullCal
from sbr.transforms import HistParams, _ar_resid, ar_filter_causal, build_transforms

#: Immutable transform ordering.  Never reorder; append only.
TRANSFORM_NAMES = (
    "mean", "sq", "abs", "rmean", "rabs", "tail_hi", "tail_lo", "tail_x",
    "center", "sign", "u", "u2", "cube", "quart", "logabs",
    "res_mean", "res_sq", "res_abs", "res_lag1", "res_sq_lag1",
    "lag1", "lag2", "abslag1",
)

_MAX_ONLINE = 1024
#: history z values kept in front of the online z buffer, and the constant AR
#: window length.  _AR_PAD must exceed _AR_WIN + ar_order.
_AR_PAD = 10
_AR_WIN = 6


class StreamCtx:
    """Incremental per-series feature state.

    Usage::

        ctx = StreamCtx(); ctx.fit_historical(hist)
        for x in online:
            ctx.push(x)
            v = engine.step(ctx)      # features for THIS observation
    """

    __slots__ = ("hp", "nc", "hist_tr", "ar_order", "_warm", "_raw", "_z",
                 "_e", "tr", "cum", "t", "cap", "hist", "_zf")

    def __init__(self, ar_order: int = 2, cap: int = _MAX_ONLINE):
        self.ar_order = ar_order
        self.cap = cap
        self.t = -1

    # ------------------------------------------------------------------ fit
    def fit_historical(self, hist: np.ndarray) -> StreamCtx:
        """One-off O(n_hist log n_hist) historical summary. Same code as batch."""
        hist = np.asarray(hist, dtype=np.float64)
        p = self.ar_order
        self.hist = hist
        self.hp = HistParams(hist, ar_order=p)
        zh = (hist - self.hp.mu) / self.hp.sd
        rh = np.concatenate([np.zeros(p), _ar_resid(zh, self.hp.ar_coef)]) if p else zh
        self.hist_tr = build_transforms(
            hist, self.hp, ar_resid=rh * self.hp.ar_sigma if p else None
        )
        self.nc = NullCal(self.hist_tr)
        self._warm = zh[-p:].copy() if p and len(zh) >= p else np.zeros(p)
        # AR-residual scratch: 10 trailing historical z values followed by the
        # online z values.  A CONSTANT-LENGTH window into this buffer makes the
        # streaming `ar_filter_causal` call take the same BLAS path as batch,
        # which is what buys bitwise identity (see tests).
        self._zf = np.zeros(_AR_PAD + self.cap, dtype=np.float64)
        if len(zh) >= _AR_PAD:
            self._zf[:_AR_PAD] = zh[-_AR_PAD:]
        else:
            self._zf[_AR_PAD - len(zh):_AR_PAD] = zh

        self._raw = np.empty(self.cap, dtype=np.float64)
        self._z = np.empty(self.cap, dtype=np.float64)
        self._e = np.empty(self.cap, dtype=np.float64)      # AR residual, z-units
        self.tr = {k: np.empty(self.cap, dtype=np.float64) for k in TRANSFORM_NAMES}
        self.cum = {k: np.zeros(self.cap + 1, dtype=np.float64) for k in TRANSFORM_NAMES}
        self.t = -1
        return self

    # ----------------------------------------------------------------- push
    def push(self, x: float) -> int:
        """Fold in one online observation. Returns the new online index t."""
        t = self.t + 1
        if t >= self.cap:                      # grow rather than fail
            self._grow()
        hp = self.hp
        self._raw[t] = x
        self._z[t] = (x - hp.mu) / hp.sd
        self._zf[_AR_PAD + t] = self._z[t]

        # --- elementwise transforms: reuse the batch implementation on a
        #     3-point window so there is exactly one definition of each column.
        lo = max(0, t - 2)
        xw = self._raw[lo:t + 1]

        p = self.ar_order
        if p:
            # AR residual: call the SAME function batch calls, on a short causal
            # window.  A hand-rolled dot product differs from batch's BLAS gemv
            # in the last ulp on ~1 % of points; reusing `ar_filter_causal` with
            # a correctly warm-started window is bitwise identical (verified in
            # tests/test_stream_ctx_parity.py).
            i = _AR_PAD + t
            self._e[t] = ar_filter_causal(
                self._zf[i - _AR_WIN + 1:i + 1], self.hp.ar_coef,
                self._zf[i - _AR_WIN + 1 - p:i - _AR_WIN + 1])[-1]
            # batch does: ar_on = raw_e * ar_sigma ; then e = ar_on / ar_sigma
            ew = (self._e[lo:t + 1] * hp.ar_sigma)
            mini = build_transforms(xw, hp, ar_resid=ew)
        else:
            mini = build_transforms(xw, hp, ar_resid=None)

        for k in TRANSFORM_NAMES:
            v = mini[k][-1] if k in mini else 0.0
            self.tr[k][t] = v
            self.cum[k][t + 1] = self.cum[k][t] + v
        self.t = t
        return t

    def _grow(self):
        self.cap *= 2
        for name in ("_raw", "_z", "_e"):
            a = getattr(self, name)
            b = np.empty(self.cap, dtype=np.float64)
            b[:len(a)] = a
            setattr(self, name, b)
        a = self._zf
        b = np.zeros(_AR_PAD + self.cap, dtype=np.float64)
        b[:len(a)] = a
        self._zf = b
        for k in TRANSFORM_NAMES:
            a = self.tr[k]
            b = np.empty(self.cap, dtype=np.float64)
            b[:len(a)] = a
            self.tr[k] = b
            a = self.cum[k]
            b = np.zeros(self.cap + 1, dtype=np.float64)
            b[:len(a)] = a
            self.cum[k] = b

    # ------------------------------------------------------------- read API
    @property
    def n(self) -> int:
        """Number of online points seen so far (== t + 1)."""
        return self.t + 1

    def roll(self, name: str, w: int) -> float:
        """Trailing-window mean at the CURRENT index; NaN if fewer than w points."""
        t = self.t
        if w > t + 1:
            return np.nan
        c = self.cum[name]
        return (c[t + 1] - c[t + 1 - w]) / w

    def expand(self, name: str) -> float:
        """Expanding mean over online[0..t] at the CURRENT index."""
        t = self.t
        return self.cum[name][t + 1] / (t + 1)

    def last(self, name: str) -> float:
        return self.tr[name][self.t]

    def window(self, name: str, w: int) -> np.ndarray:
        """The last min(w, n) values of a transform (view; do not mutate)."""
        t = self.t
        lo = max(0, t + 1 - w)
        return self.tr[name][lo:t + 1]

    def z_window(self, w: int) -> np.ndarray:
        t = self.t
        lo = max(0, t + 1 - w)
        return self._z[lo:t + 1]
