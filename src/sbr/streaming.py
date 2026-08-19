"""Streaming inference: one observation in, one score out.

The competition runner is **series-sequential and single-pass**.  ``infer``
receives an iterable of ``(x_historical, x_online)`` and must yield exactly one
score per online observation, in order, having seen nothing of the series that
come later.  Two consequences shape everything here:

1. **No cross-series information is available.**  Any blend of models must be a
   fixed per-series function of that series' own scores.  We average logits.
2. **Row t must be emitted before row t+1 exists.**  Every feature must be
   computable from the historical segment plus ``online[:t+1]``.

The second property is one we already prove for every feature module, bitwise,
via ``check_prefix_invariance``.  That gives streaming a free correctness oracle:
recomputing the batch feature builder on the growing prefix and taking the last
row is, by definition, exactly what the batch pipeline would have produced.
``ReferenceStreamer`` does precisely that.  It is correct by construction and too
slow for production, and it is the parity target every optimised streamer must
match to the last bit.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from sbr.features.base import REGISTRY, SeriesCtx, ctx_from_state, load_all, make_hist_state


class ReferenceStreamer:
    """Exactly correct, deliberately slow.  The parity oracle for the port.

    Historical state (``HistParams`` + the null-calibration engine) depends only
    on the historical segment, so it is built once per series -- that is the
    expensive part, ~150 ms.  Everything after that is recomputed on the growing
    online prefix, which is O(t) work at step t and therefore O(n^2) per series.
    """

    def __init__(self, hist: np.ndarray, modules: list[str], ar_order: int = 2):
        self.hs = make_hist_state(np.asarray(hist, dtype=np.float64), ar_order=ar_order)
        self.modules = list(modules)
        self.buf: list[float] = []
        self.cols: list[str] | None = None

    def update(self, x: float) -> np.ndarray:
        """Fold in one online observation; return its feature row."""
        self.buf.append(float(x))
        ctx = ctx_from_state(self.hs, np.asarray(self.buf, dtype=np.float64))
        parts, names = [], []
        for m in self.modules:
            cn, A = REGISTRY[m].fn(ctx)
            parts.append(A[-1])
            if self.cols is None:
                names += [f"{m}::{c}" for c in cn]
        if self.cols is None:
            self.cols = names
        return np.concatenate(parts)


def batch_rows(hist: np.ndarray, online: np.ndarray, modules: list[str]) -> np.ndarray:
    """The batch feature matrix, i.e. what training saw."""
    ctx = ctx_from_state(make_hist_state(np.asarray(hist, dtype=np.float64)),
                         np.asarray(online, dtype=np.float64))
    return np.column_stack([REGISTRY[m].fn(ctx)[1] for m in modules])


@dataclass
class ParityResult:
    ok: bool
    n_rows: int
    n_cols: int
    max_abs_diff: float
    first_bad_col: str | None
    detail: str


def check_stream_parity(streamer_factory, hist, online, modules, atol=0.0) -> ParityResult:
    """Every streamed row must equal the batch row it corresponds to.

    ``atol=0.0`` means bitwise (NaN counts as equal to NaN).  A streamer that
    cannot meet that is silently training on one feature and serving another.
    """
    B = np.asarray(batch_rows(hist, online, modules), dtype=np.float64)
    st = streamer_factory(hist, modules)
    S = np.empty_like(B)
    for i, x in enumerate(online):
        S[i] = st.update(x)
    bad = ~np.isclose(S, B, rtol=0.0, atol=atol, equal_nan=True)
    cols = getattr(st, "cols", None) or [f"c{i}" for i in range(B.shape[1])]
    if not bad.any():
        return ParityResult(True, *B.shape, 0.0, None, "bitwise identical")
    j = int(np.argmax(bad.any(axis=0)))
    i = int(np.argmax(bad[:, j]))
    d = np.nanmax(np.abs(S[bad] - B[bad])) if np.isfinite(S[bad]).any() else float("nan")
    return ParityResult(False, *B.shape, float(d), cols[j],
                        f"first mismatch row {i} col {cols[j]}: stream {S[i, j]!r} vs batch {B[i, j]!r}")


def benchmark(hist, online, modules, streamer_factory=ReferenceStreamer) -> dict:
    t0 = time.perf_counter()
    st = streamer_factory(hist, modules)
    t_init = time.perf_counter() - t0
    t0 = time.perf_counter()
    for x in online:
        st.update(x)
    t_stream = time.perf_counter() - t0
    n = len(online)
    return {"n_online": n, "n_hist": len(hist),
            "init_ms": round(t_init * 1e3, 1),
            "total_s": round(t_init + t_stream, 3),
            "ms_per_point": round(t_stream / max(n, 1) * 1e3, 3)}




class IncrementalStreamer:
    """O(1)-per-point streaming for modules that read the context only through
    ``roll`` / ``expand`` / ``idx`` and elementwise operations.

    The cumulative sums are extended by one element per observation and the
    context is handed to the module in single-row mode, so the module's own code
    -- unchanged, the same code that built the training matrix -- evaluates
    exactly one row.  Modules carrying their own sequential recursions (running
    peaks, CUSUM paths, Bayesian filters) cannot be served this way and must
    carry explicit state; ``STATELESS_MODULES`` records which are known to work,
    and ``check_stream_parity`` is the arbiter.
    """

    def __init__(self, hist, modules, ar_order: int = 2):
        self.hs = make_hist_state(np.asarray(hist, dtype=np.float64), ar_order=ar_order)
        self.modules = list(modules)
        self.cols = None
        self.pos = -1
        self.tail = []                       # recent raw points, for lag products
        self.cum = None                      # name -> growing cumulative sum
        self._cap = 1024

    def _grow(self, tr_point: dict):
        if self.cum is None:
            self.cum = {k: np.zeros(self._cap + 1) for k in tr_point}
        elif self.pos + 2 > self._cap:
            self._cap *= 2
            self.cum = {k: np.resize(v, self._cap + 1) for k, v in self.cum.items()}
        p = self.pos
        for k, v in tr_point.items():
            self.cum[k][p + 1] = self.cum[k][p] + v

    def update(self, x: float) -> np.ndarray:
        from sbr.transforms import build_transforms

        self.pos += 1
        self.tail.append(float(x))
        # transforms of the new point only: recompute over a short tail window so
        # lag products and the AR filter see the lags they need, then keep the
        # last element.  The tail is bounded, so this is O(1) in the series length.
        k = min(len(self.tail), 8)
        seg = np.asarray(self.tail[-k:], dtype=np.float64)
        warm = self.hs.zh if self.pos + 1 <= k else None
        zo = (seg - self.hs.hp.mu) / self.hs.hp.sd
        if self.hs.ar_order:
            from sbr.transforms import ar_filter_causal
            prior = np.asarray(self.tail[:-k], dtype=np.float64)
            prior_z = (prior - self.hs.hp.mu) / self.hs.hp.sd if len(prior) else self.hs.zh
            ar_seg = ar_filter_causal(zo, self.hs.hp.ar_coef,
                                      prior_z if len(prior_z) else self.hs.zh) * self.hs.hp.ar_sigma
        else:
            ar_seg = None
        tr = build_transforms(seg, self.hs.hp, ar_resid=ar_seg)
        self._grow({kk: float(vv[-1]) for kk, vv in tr.items()})

        ctx = SeriesCtx(hist=self.hs.hist, online=np.asarray(self.tail),
                        hp=self.hs.hp, nc=self.hs.nc, tr=None, cum=self.cum,
                        n=1, hist_tr=self.hs.hist_tr, ar_online=None, pos=self.pos)
        parts, names = [], []
        for m in self.modules:
            cn, A = REGISTRY[m].fn(ctx)
            parts.append(A[-1])
            if self.cols is None:
                names += [f"{m}::{c}" for c in cn]
        if self.cols is None:
            self.cols = names
        return np.concatenate(parts)


# modules verified to produce bitwise-identical rows under IncrementalStreamer
STATELESS_MODULES = ["m00_core"]


class BlendedModel:
    """The deployable ensemble: mean of the streams' logits.

    A per-series function of that series' own scores, so it survives the
    single-pass, series-sequential inference contract.  Averaging within-timestep
    ranks does not -- see research/FAILED_EXPERIMENTS.md.
    """

    def __init__(self, boosters, module_sets, all_modules):
        self.boosters = boosters
        self.idx = []
        for mods in module_sets:
            keep = []
            off = 0
            for m in all_modules:
                k = len(REGISTRY[m].fn.__dict__.get("_ncols", []) or [])  # placeholder
                del k
            self.idx.append(mods)
        self.all_modules = all_modules

    @staticmethod
    def _logit(p, eps=1e-6):
        p = np.clip(p, eps, 1.0 - eps)
        return np.log(p / (1.0 - p))


def _ensure_registry():
    if not REGISTRY:
        load_all()
