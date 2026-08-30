"""Incremental port of the ``m00_core`` batch feature module.

The batch module (``sbr.features.m00_core``) computes, for every online index
``t``, a row of calibrated evidence built out of

  * expanding-window robust z's of 18 transforms,
  * trailing-window signed surprise + robust z of 7 transforms at 6 scales,
  * cross-scale max/mean/min of the |z| columns,
  * short-vs-long and short-vs-expanding contrasts,
  * the two "how much online evidence exists" conditioning columns.

Every one of those is a *scalar* function of quantities the shared
:class:`~sbr.stream.ctx.StreamCtx` already carries at index ``t``
(``roll`` / ``expand`` / ``nc``), so the port is mechanical.

Bitwise parity, and how the fast path keeps it
----------------------------------------------
The first working version of this module evaluated every batch expression on a
length-1 array so the identical numpy/nullcal code path ran.  That is correct
but costs ~2.0 ms/observation, almost all of it numpy dispatch overhead on
one-element arrays.  The fast path removes the per-row numpy calls whose result
is either (a) precomputable at fit time or (b) reproducible exactly in scalar
float64.  Each substitution below was validated by a differential fuzz against
the original expression before being adopted:

``NullCal.z``
    ``z`` is ``(v - mu) / np.maximum(sd, 1e-12)`` where ``(mu, sd)`` come from
    ``NullCal._interp``.  We do NOT reimplement the interpolation -- crucially,
    ``sd = exp(interp(log(null_sd)))`` is not the tabulated ``null_sd`` even at
    an exact grid node, so recomputing it any other way breaks parity.  Instead
    we call ``nc._interp`` itself at fit time and cache its *result*:
    ``np.interp`` is strictly elementwise, so one vectorised call on
    ``arange(1, cap+1)`` yields a table whose entry ``L-1`` is bit-identical to
    the scalar call ``nc._interp(name, L)``.  The trailing block needs only
    ``L == w`` (6 values); the expanding block needs ``L == t+1``.
    ``np.maximum(sd, 1e-12)`` is folded into the table too.

``NullCal.pct``
    exact rational arithmetic over two integer binary searches; the scalar
    ``rs.searchsorted(v, side=...)`` returns the same integers as the array
    form, so ``(0.5*(lo+hi))/len(rs)`` is identical.

``_signed_surprise``
    reproduced scalar-wise, but the ``log10`` MUST stay ``np.log10``:
    ``math.log10`` disagrees with ``np.log10`` by 1 ulp on ~22 % of the
    realizable p-grid (4064 / 18k values), so it is not usable here.

``np.clip`` / ``np.abs``
    scalar branches; verified NaN-, inf- and signed-zero-safe.

``np.nanmax/nanmean/nanmin`` over the cross-scale row
    reproduced scalar-wise.  ``np.nanmean`` sums the NaN->0 substituted row and
    divides by the non-NaN count; for a row of k < 8 elements numpy's pairwise
    summation degenerates to a sequential accumulation from ``0.0``, which is
    what the Python loop does.  ``WINDOWS`` has 6 entries so k <= 6 always.
    An all-NaN row yields NaN rather than raising, as in batch.

``xs_*`` column selection
    the batch module locates the cross-scale inputs by string matching over the
    accumulating ``cols`` list.  The column layout is fixed once the null grid
    is known, so the same string matching is replayed at fit time and collapses
    to fixed integer indices.

Finally, batch does ``A = np.column_stack(out).astype(np.float32)`` and only
*then* ``A[~np.isfinite(A)] = np.nan``, so the finiteness test is applied to the
float32 values.  We reproduce that exactly (it matters: some columns are finite
in float64 but overflow float32) while returning float64, as the contract
requires the caller to do the final cast.

Cost per observation: no rescan of the online prefix, no re-sort, no refit, and
after the fit-time caching no interpolation either -- ~7 numpy calls per row
plus scalar arithmetic.  See the timing test.
"""
from __future__ import annotations

import numpy as np

from sbr.features.m00_core import CLIP, TR_EXP, TR_MULTI, WINDOWS

#: contrasts emitted by the short-vs-long block, in batch order
SL_NAMES = ("mean", "sq", "abs", "u2")

_NEG_CLIP = -CLIP
_LOG10 = np.log10          # bound once; must be numpy's, not math's (see docstring)


class StreamM00Core:
    """Streaming (one row per observation) implementation of ``m00_core``."""

    MODULE = "m00_core"

    __slots__ = ("_cols", "_exp_plan", "_w_plan", "_xs", "_sl_plan",
                 "_cap", "_nc", "_fitted")

    def __init__(self) -> None:
        self._cols: list[str] = []
        self._exp_plan: list = []
        self._w_plan: list = []
        self._xs: list = []
        self._sl_plan: list = []
        self._cap = 0
        self._nc = None
        self._fitted = False

    # ----------------------------------------------------------------- fit
    def fit_historical(self, ctx) -> None:
        """Resolve the column layout and cache every null-calibration lookup.

        The batch module's column set depends on the historical segment through
        ``ctx.nc.grid`` (the null grid is capped at ``n_hist // 2``), so short
        histories legitimately drop the longer windows -- and with them the
        cross-scale block if fewer than two scales survive.  We replay the exact
        same construction here, on names only.
        """
        nc = ctx.nc
        self._nc = nc
        cols: list[str] = []
        exp_names: list[str] = []
        wn: list[tuple[int, str]] = []
        self._xs = []

        for name in TR_EXP:
            if name not in ctx.cum:
                continue
            exp_names.append(name)
            cols.append(f"exp_z_{name}")
            cols.append(f"exp_absz_{name}")

        grid = set(int(g) for g in nc.grid)
        for w in WINDOWS:
            if w not in grid:
                continue
            for name in TR_MULTI:
                if name not in ctx.cum:
                    continue
                wn.append((int(w), name))
                cols.append(f"w{w}_sur_{name}")
                cols.append(f"w{w}_z_{name}")

        for name in TR_MULTI:
            idx = [i for i, c in enumerate(cols)
                   if c.startswith("w") and c.endswith(f"_z_{name}")]
            if len(idx) >= 2:
                # batch's string matching, resolved once to fixed indices
                self._xs.append(tuple(idx))
                cols.append(f"xs_max_{name}")
                cols.append(f"xs_mean_{name}")
                cols.append(f"xs_min_{name}")

        for name in SL_NAMES:
            cols.append(f"sl_{name}_16_128")
            cols.append(f"se_{name}_16_exp")

        cols.append("t_online")
        cols.append("log_t_online")

        self._cols = cols
        self._sl_plan = [n for n in SL_NAMES]

        # --- fit-time null-calibration tables (results of nc._interp itself) --
        self._build_tables(exp_names, wn, int(getattr(ctx, "cap", 1024)))
        self._fitted = True

    def _build_tables(self, exp_names, wn, cap: int) -> None:
        """Cache ``nc._interp`` *results* for every L this series can need.

        ``np.interp`` is elementwise, so evaluating it once on ``arange(1,cap+1)``
        gives a table that is bit-identical, entry by entry, to the per-L scalar
        calls it replaces (verified by fuzz).
        """
        nc = self._nc
        cap = max(int(cap), 1)
        L = np.arange(1, cap + 1, dtype=np.float64)
        self._cap = cap

        exp_plan = []
        for name in exp_names:
            mu, sd = nc._interp(name, L)
            exp_plan.append((name, mu, np.maximum(sd, 1e-12)))
        self._exp_plan = exp_plan

        w_plan = []
        for w, name in wn:
            mu, sd = nc._interp(name, w)
            rs = nc.null_sorted[(name, int(w))]
            w_plan.append((w, name, float(mu[0]),
                           float(np.maximum(sd, 1e-12)[0]), rs, len(rs)))
        self._w_plan = w_plan

    def _grow_tables(self, need: int) -> None:
        exp_names = [p[0] for p in self._exp_plan]
        wn = [(p[0], p[1]) for p in self._w_plan]
        self._build_tables(exp_names, wn, max(need, self._cap * 2))

    # ---------------------------------------------------------------- step
    def step(self, ctx) -> np.ndarray:
        """Row ``t`` of the batch output, float64, non-finite mapped to NaN."""
        t = ctx.t
        t1 = t + 1
        if t1 > self._cap:
            self._grow_tables(t1)
        cum = ctx.cum
        vals: list = []
        ap = vals.append

        # ---- expanding-window evidence -------------------------------------
        # batch: z = clip(nc.z(name, L, expand(name)), -CLIP, CLIP); then |z|
        j = t                                   # table index for L = t+1
        for name, mu_tab, den_tab in self._exp_plan:
            v = cum[name][t1] / t1              # == ctx.expand(name)
            z = (v - mu_tab[j]) / den_tab[j]
            if z < _NEG_CLIP:
                z = _NEG_CLIP
            elif z > CLIP:
                z = CLIP
            ap(z)
            ap(abs(z))

        # ---- trailing multi-scale evidence ---------------------------------
        for w, name, mu, den, rs, nrs in self._w_plan:
            if w > t1:
                # batch: ss/zz keep their np.full(n, nan) initialisation
                ap(np.nan)
                ap(np.nan)
                continue
            c = cum[name]
            r = (c[t1] - c[t1 - w]) / w         # == ctx.roll(name, w)
            if r != r:                          # NaN (batch's ~np.isnan(v) mask)
                ap(np.nan)
                ap(np.nan)
                continue
            # --- signed surprise on the exact empirical percentile
            # searchsorted(side="right") can only differ from side="left" when
            # `r` is actually present in the null, so probe rs[lo] first and
            # skip the second binary search otherwise.  Exact by construction.
            lo = rs.searchsorted(r, side="left")
            hi = rs.searchsorted(r, side="right") if (lo < nrs and rs[lo] == r) else lo
            p = (0.5 * (lo + hi)) / nrs
            q = 1.0 - p
            two = 2.0 * (p if p <= q else q)
            s = -_LOG10(two if two > 1e-6 else 1e-6)
            d = p - 0.5
            ss = (0.0 if d == 0.0 else (1.0 if d > 0.0 else -1.0)) * s
            if ss < -6.0:
                ss = -6.0
            elif ss > 6.0:
                ss = 6.0
            ap(ss)
            # --- robust z
            zz = (r - mu) / den
            if zz < _NEG_CLIP:
                zz = _NEG_CLIP
            elif zz > CLIP:
                zz = CLIP
            ap(zz)

        # ---- cross-scale aggregation ---------------------------------------
        # batch: nanmax/nanmean/nanmin over |z| of this transform's scales
        for idx in self._xs:
            tot = 0.0
            cnt = 0
            mx = 0.0
            mn = 0.0
            for i in idx:
                x = vals[i]
                if x != x:                      # NaN contributes 0.0 to the sum
                    tot += 0.0
                    continue
                if x < 0.0:
                    x = -x
                tot += x
                if cnt == 0 or x > mx:
                    mx = x
                if cnt == 0 or x < mn:
                    mn = x
                cnt += 1
            if cnt == 0:                        # all-NaN row -> NaN, not an error
                ap(np.nan)
                ap(np.nan)
                ap(np.nan)
            else:
                ap(mx)
                ap(tot / cnt)
                ap(mn)

        # ---- short-vs-long contrast ----------------------------------------
        for name in self._sl_plan:
            c = cum[name]
            s = (c[t1] - c[t1 - 16]) / 16 if t1 >= 16 else np.nan
            l = (c[t1] - c[t1 - 128]) / 128 if t1 >= 128 else np.nan
            e = c[t1] / t1
            ap(s - l)
            ap(s - e)

        # ---- evidence-quantity conditioning columns -------------------------
        Lt = float(t1)
        ap(Lt)
        ap(np.log1p(Lt))

        a = np.asarray(vals, dtype=np.float64)
        # batch does the finiteness scrub *after* the float32 cast, so values
        # that overflow float32 become NaN.  `over="ignore"` only silences the
        # RuntimeWarning batch also raises (once per matrix instead of once per
        # row); it does not change a single bit of the result.
        # NOTE: a single reusable np.errstate instance would save ~2 us/row but
        # numpy 2.x raises "Cannot enter `np.errstate` twice", so it is built
        # per row.  `over="ignore"` silences the float32-cast overflow warning
        # batch also raises (once per matrix there, once per row here); it does
        # not change a single bit of the result.
        with np.errstate(over="ignore"):
            a[~np.isfinite(a.astype(np.float32))] = np.nan
        return a

    # ---------------------------------------------------------------- cols
    @property
    def cols(self) -> list[str]:
        if not self._fitted:
            raise RuntimeError("fit_historical() must be called before cols")
        return self._cols
