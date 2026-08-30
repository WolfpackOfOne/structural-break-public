"""StreamM12Rdep -- incremental twin of the batch module ``m12_rdep``.

The batch module is the specification.  Row ``t`` of
``REGISTRY["m12_rdep"].fn(make_ctx(hist, online))`` must be reproduced exactly
(``atol = 0``, ``NaN == NaN``) by ``step`` after ``StreamCtx.push``.

WHY BITWISE IS REACHABLE HERE
-----------------------------
1. **Every null is built by calling the batch helpers.**  ``_GridNull``,
   ``_WinNull`` and ``sbr.nullcal.NullCal`` are imported from the batch module
   and constructed at ``fit_historical`` on the historical arrays, exactly as
   batch constructs them.  There is one definition of each null and no chance
   of drift.

2. **Cumulative arrays are kept, not running scalars.**  Batch forms every
   window as a difference of a ``np.cumsum``, and ``np.cumsum`` accumulates
   sequentially, so ``c[t+1] = c[t] + v[t]`` reproduces it bit for bit.  We keep
   the arrays and difference them with the *identical expression* -- including
   the deliberate ``((c[t+1] - c[t+1-w]) / w) * w`` round trip in the dependence
   block, which is NOT the identity in floating point and must be reproduced,
   not simplified.

3. **The AR residual is read off the context, never recomputed.**  Batch uses
   ``e = ctx.ar_online / hp.ar_sigma``, and ``build_transforms`` defines
   ``res_mean`` by exactly that expression on exactly that input, so
   ``ctx.tr["res_mean"][t]`` IS batch's ``e[t]``.  Recomputing the AR filter
   here would differ in the last ulp on ~1% of points.

4. **The occupancy reductions are along axis 0.**  Batch reduces a
   ``(bins, n)`` array along axis 0 -- the non-contiguous axis -- which numpy
   evaluates as a strictly sequential accumulation per column, with no pairwise
   regrouping.  A ``(bins, 1)`` array therefore gives the same float.  This is
   an empirical claim about numpy and it is pinned by
   ``test_stream_parity_m12_rdep.py::test_occupancy_block_bitwise`` over random
   occupancy columns rather than assumed.

COST PER OBSERVATION
--------------------
O(bins) plus a bounded number of scalar expressions and two ``np.interp``
lookups per calibrated column.  Nothing is rescanned: no sorting, no re-nulling,
no revisiting of the online prefix.  Measured microseconds/observation are
reported by the parity test.
"""
from __future__ import annotations

import numpy as np

from sbr.features.m12_rdep import (
    BINS,
    CLIP,
    EPS,
    WINDOWS,
    _dep_lr_path,
    _GridNull,
    _occupancy_distances,
    _reflect_cusum,
    _WinNull,
)
from sbr.nullcal import NullCal

DIST = ("chi2", "js", "ks", "w1", "cvm", "ene")
#: (name, sign) of the four CUSUM paths, in batch emission order
PATHS = (("rc", 1.0, "up"), ("rc", -1.0, "dn"), ("rq", 1.0, "up"), ("rq", -1.0, "dn"))
K_DRIFT = 0.5
_MAX_ONLINE = 1024


def _dist_from_occupancy(occ, bins=BINS):
    """The batch divergence block, applied to ONE occupancy column.

    ``occ`` must be ``(bins, 2)`` float64 with the occupancy in column 0; column 1
    is padding and its value is discarded.

    WHY THE PADDING IS LOAD-BEARING.  Batch reduces a ``(bins, n)`` array along
    axis 0 -- the NON-contiguous axis -- and numpy takes a strided outer-loop
    path for that.  A ``(bins, 1)`` array is contiguous, so numpy takes the
    pairwise path instead and the two disagree in the last ulp on ~1/3 of random
    columns.  Widths 2 and above all agree with each other and with width n, so
    one padding column restores the batch reduction exactly.  Measured, not
    assumed: ``test_reduction_width_invariance``.

    Every expression below is copied verbatim from ``_occupancy_distances`` so
    that the arithmetic, the operation order and the axis of every reduction
    match.  Do not simplify it.
    """
    q = 1.0 / bins
    with np.errstate(invalid="ignore", divide="ignore"):
        chi2 = np.nansum((occ - q) ** 2, axis=0) / q
        M = 0.5 * (occ + q)
        js = 0.5 * (
            np.nansum(
                np.where(occ > 0, occ * np.log(np.maximum(occ, EPS) / np.maximum(M, EPS)), 0.0),
                axis=0,
            )
            + np.nansum(np.where(M > 0, q * np.log(q / np.maximum(M, EPS)), 0.0), axis=0)
        )
        d = np.cumsum(occ, axis=0) - np.cumsum(np.full(bins, q))[:, None]
        ks = np.nanmax(np.abs(d), axis=0)
        w1 = np.nansum(np.abs(d), axis=0) / bins
        cvm = np.nansum(d ** 2, axis=0) / bins
        ene = 2.0 * cvm
    return {"chi2": chi2[0], "js": js[0], "ks": ks[0],
            "w1": w1[0], "cvm": cvm[0], "ene": ene[0]}


class _ScalarGrid:
    """A ``_GridNull`` with its length interpolation precomputed for every L.

    ``_GridNull.z`` costs a ``np.log``, two ``np.interp``, a ``np.exp`` and a
    ``np.clip`` per call, all on one-element arrays -- ~10 us of numpy dispatch
    for ~10 ns of arithmetic, and the module makes nine such calls per
    observation.  But the length it interpolates at is ``L = t + 1``, a function
    of the index alone, so the whole curve can be tabulated once at fit time and
    the per-observation work becomes two array reads and three scalar ops.

    Bitwise: ``np.interp`` is elementwise, so evaluating it over ``[1..cap]``
    gives each entry the same float as evaluating it at that scalar; the
    remaining arithmetic is the same IEEE operations in the same order.  Pinned
    by ``test_scalar_grid_matches_gridnull``.
    """

    __slots__ = ("mu", "sg", "cap")

    def __init__(self, gn, cap):
        L = np.arange(1, cap + 1, dtype=np.float64)
        x = np.log(np.maximum(L, 1.0))
        self.mu = np.interp(x, gn.lg, gn.med)
        self.sg = np.exp(np.interp(x, gn.lg, np.log(gn.sd)))
        self.cap = cap

    def z(self, n, v):
        """``n`` is the 1-based length (== t + 1); ``v`` a python float."""
        i = n - 1
        r = (v - self.mu[i]) / max(self.sg[i], EPS)
        if r != r:
            return r
        return -CLIP if r < -CLIP else (CLIP if r > CLIP else r)


class _ScalarNullCal:
    """``NullCal.z`` for one transform, tabulated over L and over fixed windows."""

    __slots__ = ("mu", "sg", "wmu", "wsg")

    def __init__(self, nc, name, cap, windows):
        L = np.arange(1, cap + 1, dtype=np.float64)
        mu, sd = nc._interp(name, L)
        self.mu, self.sg = mu, sd
        self.wmu, self.wsg = {}, {}
        for w in windows:
            m, s = nc._interp(name, float(w))
            self.wmu[w], self.wsg[w] = float(m[0]), float(s[0])

    def z_exp(self, n, v):
        i = n - 1
        return (v - self.mu[i]) / max(self.sg[i], EPS)

    def z_win(self, w, v):
        return (v - self.wmu[w]) / max(self.wsg[w], EPS)


def _clip(r):
    """``np.clip(r, -CLIP, CLIP)`` for a python float, NaN-preserving."""
    if r != r:
        return r
    return -CLIP if r < -CLIP else (CLIP if r > CLIP else r)


def _pad2(col):
    """A (bins, 2) array whose column 0 is ``col``.  See _dist_from_occupancy."""
    occ = np.empty((len(col), 2), dtype=np.float64)
    occ[:, 0] = col
    occ[:, 1] = col
    return occ


class StreamM12Rdep:
    MODULE = "m12_rdep"

    def __init__(self, cap: int = _MAX_ONLINE):
        self.cap = cap
        self._cols = None

    # ------------------------------------------------------------------ fit
    def fit_historical(self, ctx) -> None:
        hp = ctx.hp
        self.hp = hp
        zh = (ctx.hist - hp.mu) / hp.sd
        eh = np.asarray(ctx.hist_tr.get("res_mean", zh), dtype=np.float64)
        self.zh, self.eh = zh, eh

        # ---- residual PIT against the historical residual ECDF -------------
        self.hs = np.sort(eh)
        self.nh = len(self.hs)
        uh = (np.searchsorted(self.hs, eh, side="left") + 0.5) / (self.nh + 1.0)

        # ---- W5-E4 nulls: one _GridNull per distance, 2 _WinNulls each -----
        self.gn_exp = {k: _GridNull(uh, lambda seg, kk=k: _occupancy_distances(seg, None)[kk])
                       for k in DIST}
        self.wn = {(w, k): _WinNull(uh, w, lambda a, ww, kk=k: _occupancy_distances(a, ww)[kk])
                   for w in WINDOWS for k in DIST}

        # ---- W5-E4b robust-scale nulls: the shared NullCal engine ----------
        self.med_h = float(np.median(eh))
        self.q99 = float(np.quantile(np.abs(eh - self.med_h), 0.99))
        tr_h = {"bf": np.abs(eh - self.med_h), "lv": (eh - self.med_h) ** 2,
                "tail": (np.abs(eh - self.med_h) > self.q99).astype(np.float64),
                "e2": eh * eh}
        self.nc = NullCal(tr_h)

        # ---- W5-E5 CUSUM path nulls ---------------------------------------
        self.gn_path = {}
        for nm, sgn, sfx in PATHS:
            y_h = eh if nm == "rc" else eh * eh - 1.0
            self.gn_path[(nm, sfx)] = _GridNull(
                y_h, lambda seg, sg=sgn: _reflect_cusum(sg * seg - K_DRIFT))

        # ---- W5-E6 dependence nulls ---------------------------------------
        self.phi0 = float(hp.ar_coef[0]) if len(hp.ar_coef) else 0.0
        self.gn_lr = _GridNull(zh, lambda seg: _dep_lr_path(seg, None, self.phi0)[0])
        self.gn_v = _GridNull(zh, lambda seg: _dep_lr_path(seg, None, self.phi0)[2])
        self.wn_lr = {w: _WinNull(zh, w, lambda a, ww: _dep_lr_path(a, ww, self.phi0)[0])
                      for w in WINDOWS}
        self.wn_v = {w: _WinNull(zh, w, lambda a, ww: _dep_lr_path(a, ww, self.phi0)[2])
                     for w in WINDOWS}

        # tabulate every length-interpolated null over L = 1..cap
        self.sg_exp = {k: _ScalarGrid(self.gn_exp[k], self.cap) for k in DIST}
        self.sg_path = {key: _ScalarGrid(g, self.cap) for key, g in self.gn_path.items()}
        self.sg_lr = _ScalarGrid(self.gn_lr, self.cap)
        self.sg_v = _ScalarGrid(self.gn_v, self.cap)
        self.snc = {k: _ScalarNullCal(self.nc, k, self.cap, WINDOWS)
                    for k in ("bf", "lv", "tail")}
        # window nulls are already scalar: (v - med) / sd, or NaN when unusable
        self.wn_s = {key: (wn.ok, getattr(wn, "med", 0.0), getattr(wn, "sd", 1.0))
                     for key, wn in self.wn.items()}
        self.wn_lr_s = {w: (g.ok, getattr(g, "med", 0.0), getattr(g, "sd", 1.0))
                        for w, g in self.wn_lr.items()}
        self.wn_v_s = {w: (g.ok, getattr(g, "med", 0.0), getattr(g, "sd", 1.0))
                       for w, g in self.wn_v.items()}

        self._reset_online()
        self._cols = self._build_cols()

    def _reset_online(self):
        c = self.cap
        self.Cbin = np.zeros((c + 1, BINS), dtype=np.float64)
        self.cums = {k: np.zeros(c + 1, dtype=np.float64) for k in ("bf", "lv", "tail")}
        # dependence cumulative sums, in the order _dep_lr_path forms them
        self.c_xx = np.zeros(c + 1, dtype=np.float64)
        self.c_xy = np.zeros(c + 1, dtype=np.float64)
        self.c_yy = np.zeros(c + 1, dtype=np.float64)
        # CUSUM state: cumulative sum of the drifted stream and its running min
        self.pc = {(nm, sfx): np.zeros(c + 1, dtype=np.float64) for nm, _, sfx in PATHS}
        self.pmin = {(nm, sfx): 0.0 for nm, _, sfx in PATHS}
        self.ppk = {(nm, sfx): -np.inf for nm, _, sfx in PATHS}
        self.pdpk = {(nm, sfx): -1.0e18 for nm, _, sfx in PATHS}
        self.parg = {(nm, sfx): 0.0 for nm, _, sfx in PATHS}
        self.pcnt = {(nm, sfx): 0 for nm, _, sfx in PATHS}
        self._t = -1

    def _build_cols(self):
        cols = []
        for k in DIST:
            cols.append(f"rd_exp_{k}")
        for w in WINDOWS:
            for k in DIST:
                cols.append(f"rd_w{w}_{k}")
        for k in ("bf", "lv", "tail"):
            cols.append(f"rs_exp_{k}")
            for w in WINDOWS:
                cols.append(f"rs_w{w}_{k}")
        for nm, _, sfx in PATHS:
            tag = f"{nm}{sfx}"
            cols += [f"{tag}_cur", f"{tag}_pk", f"{tag}_dpk", f"{tag}_tsp", f"{tag}_per"]
        cols += ["dl_exp_lr", "dl_exp_phi", "dl_exp_vlr"]
        for w in WINDOWS:
            cols += [f"dl_w{w}_lr", f"dl_w{w}_phi", f"dl_w{w}_vlr"]
        cols.append("dl_exp_sep")
        return cols

    @property
    def cols(self):
        return list(self._cols)

    @staticmethod
    def _L1(L):
        """Length as a 1-element array.

        `_GridNull.z` returns a numpy SCALAR when handed a scalar length and an
        ARRAY when handed an array, and batch always hands it an array.  Passing
        a 1-element array keeps the `np.interp` call on the same code path as
        batch's, which is the point.
        """
        return np.array([L], dtype=np.float64)

    # ----------------------------------------------------------------- step
    def step(self, ctx) -> np.ndarray:
        t = ctx.t
        if t + 1 > self.cap:
            self._grow()
        n = t + 1
        L = float(n)
        t1 = t + 1
        out = np.empty(len(self._cols), dtype=np.float64)
        i = 0

        z_t = float(ctx.tr["mean"][t])
        e_t = float(ctx.tr["res_mean"][t])

        # ---------- residual PIT and its occupancy ------------------------
        lo = np.searchsorted(self.hs, e_t, side="left")
        hi = np.searchsorted(self.hs, e_t, side="right")
        u_t = (0.5 * (lo + hi) + 0.5) / (self.nh + 1.0)
        b = int(u_t * BINS)
        if b > BINS - 1:
            b = BINS - 1
        if b < 0:
            b = 0
        self.Cbin[t1] = self.Cbin[t]
        self.Cbin[t1, b] += 1.0

        Oexp = _pad2(self.Cbin[t1] / L)
        Dexp = _dist_from_occupancy(Oexp)
        for k in DIST:
            out[i] = self.sg_exp[k].z(n, Dexp[k])
            i += 1
        for w in WINDOWS:
            if w <= n:
                Ow = _pad2((self.Cbin[t1] - self.Cbin[t1 - w]) / w)
                Dw = _dist_from_occupancy(Ow)
                for k in DIST:
                    ok, med, sd = self.wn_s[(w, k)]
                    out[i] = _clip((Dw[k] - med) / sd) if ok else np.nan
                    i += 1
            else:
                for _k in DIST:
                    out[i] = np.nan
                    i += 1

        # ---------- robust two-sample scale on the residual ----------------
        bf_t = abs(e_t - self.med_h)
        lv_t = (e_t - self.med_h) ** 2
        tail_t = float(bf_t > self.q99)
        for k, v in (("bf", bf_t), ("lv", lv_t), ("tail", tail_t)):
            c = self.cums[k]
            c[t1] = c[t] + v
            out[i] = _clip(self.snc[k].z_exp(n, c[t1] / L))
            i += 1
            for w in WINDOWS:
                if w <= n:
                    rv = (c[t1] - c[t1 - w]) / w
                    out[i] = _clip(self.snc[k].z_win(w, rv))
                else:
                    out[i] = np.nan
                i += 1

        # ---------- residual CUSUM / CUSUMSQ paths -------------------------
        for nm, sgn, sfx in PATHS:
            y = e_t if nm == "rc" else e_t * e_t - 1.0
            key = (nm, sfx)
            c = self.pc[key]
            c[t1] = c[t] + (sgn * y - K_DRIFT)
            p = c[t1] - self.pmin[key]
            # running minimum of concat([[0.0], cumsum))[:t+1]  == min(0, c_0..c_{t-1})
            if c[t1] < self.pmin[key]:
                self.pmin[key] = c[t1]
            zp = self.sg_path[key].z(n, p)
            pk = zp if zp > self.ppk[key] else self.ppk[key]
            self.ppk[key] = pk
            a = self.pdpk[key] * 0.98
            dpk = zp if zp > a else a
            self.pdpk[key] = dpk
            if zp >= pk - 1e-12:
                self.parg[key] = L
            if zp > 2.0:
                self.pcnt[key] += 1
            out[i] = zp
            i += 1
            out[i] = pk
            i += 1
            out[i] = dpk
            i += 1
            out[i] = np.log1p(L - self.parg[key])
            i += 1
            out[i] = self.pcnt[key] / L
            i += 1

        # ---------- dependence LR, variance profiled out -------------------
        zl1 = float(ctx.tr["mean"][t - 1]) if t >= 1 else 0.0
        self.c_xx[t1] = self.c_xx[t] + zl1 * zl1
        self.c_xy[t1] = self.c_xy[t] + zl1 * z_t
        self.c_yy[t1] = self.c_yy[t] + z_t * z_t

        sxx, sxy, syy, m = self.c_xx[t1], self.c_xy[t1], self.c_yy[t1], L
        phi = sxy / max(sxx, EPS)
        free = max((syy - 2 * phi * sxy + phi * phi * sxx) / m, EPS)
        base = max((syy - 2 * self.phi0 * sxy + self.phi0 * self.phi0 * sxx) / m, EPS)
        lr = m * np.log(base / free)
        dl_exp_lr = self.sg_lr.z(n, lr)
        out[i] = dl_exp_lr
        i += 1
        pe = phi - self.phi0
        out[i] = -3.0 if pe < -3.0 else (3.0 if pe > 3.0 else pe)
        i += 1
        dl_exp_vlr = self.sg_v.z(n, free)
        out[i] = dl_exp_vlr
        i += 1

        for w in WINDOWS:
            if w <= n:
                # the /w then *w round trip is deliberate: batch's _roll divides
                # and _dep_lr_path multiplies back, which is not the identity
                sxxw = ((self.c_xx[t1] - self.c_xx[t1 - w]) / w) * w
                sxyw = ((self.c_xy[t1] - self.c_xy[t1 - w]) / w) * w
                syyw = ((self.c_yy[t1] - self.c_yy[t1 - w]) / w) * w
                mw = float(w)
                phiw = sxyw / max(sxxw, EPS)
                freew = max((syyw - 2 * phiw * sxyw + phiw * phiw * sxxw) / mw, EPS)
                basew = max((syyw - 2 * self.phi0 * sxyw
                             + self.phi0 * self.phi0 * sxxw) / mw, EPS)
                lrw = mw * np.log(basew / freew)
                ok, med, sd = self.wn_lr_s[w]
                out[i] = _clip((lrw - med) / sd) if ok else np.nan
                i += 1
                pw = phiw - self.phi0
                out[i] = -3.0 if pw < -3.0 else (3.0 if pw > 3.0 else pw)
                i += 1
                ok, med, sd = self.wn_v_s[w]
                out[i] = _clip((freew - med) / sd) if ok else np.nan
                i += 1
            else:
                out[i] = np.nan
                i += 1
                out[i] = np.nan
                i += 1
                out[i] = np.nan
                i += 1

        out[i] = dl_exp_lr - abs(dl_exp_vlr)
        i += 1

        assert i == len(self._cols), (i, len(self._cols))
        out[~np.isfinite(out)] = np.nan
        self._t = t
        return out

    def _grow(self):
        self.cap *= 2
        C = np.zeros((self.cap + 1, BINS), dtype=np.float64)
        C[:len(self.Cbin)] = self.Cbin
        self.Cbin = C
        for k in self.cums:
            a = self.cums[k]
            b = np.zeros(self.cap + 1, dtype=np.float64)
            b[:len(a)] = a
            self.cums[k] = b
        for name in ("c_xx", "c_xy", "c_yy"):
            a = getattr(self, name)
            b = np.zeros(self.cap + 1, dtype=np.float64)
            b[:len(a)] = a
            setattr(self, name, b)
        for key in list(self.pc):
            a = self.pc[key]
            b = np.zeros(self.cap + 1, dtype=np.float64)
            b[:len(a)] = a
            self.pc[key] = b
