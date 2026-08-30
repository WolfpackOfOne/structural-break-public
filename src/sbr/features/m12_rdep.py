"""m12_rdep -- residualised distribution distances, residual CUSUM/CUSUMSQ paths,
and a dependence likelihood ratio that isolates phi from scale.

WHAT IS ACTUALLY MISSING FROM THE INCUMBENT BANK
    This module exists to close three specific gaps, each identified by reading
    what the seven production modules already do rather than by assuming:

    W5-E4 / RT-906.  m02_dist computes chi2, JS, Hellinger, TV, Cramer-von Mises,
        KS, 1-D Wasserstein and energy distance -- but all of them on the RAW
        historical-ECDF PIT.  The 2025 public solutions computed the same
        distances on AR-RESIDUAL streams as well, and nothing in the bank does.
        Removing predictable dynamics first changes what the distance sees: a
        series whose marginal law is unchanged but whose innovation law is not
        is invisible to the raw PIT and visible to the residual PIT.

    W5-E5 / RT-907.  m01_seq runs CUSUM and CUSUM-SQ as full detector paths with
        peak/decay/persistence geometry, but on the raw standardised series.
        m04_resid monitors residual MOMENTS against a null but runs no
        recursion on them.  The residual CUSUM/CUSUMSQ PATH is in neither.

    W5-E6.  m03_dyn emits ACF/PACF-style z-scores and m07_bayes a dependence
        posterior channel, but no likelihood ratio for a change in the AR
        coefficient itself.  A variance change and a dependence change move
        every moment-based dependence statistic together; the LR below profiles
        the innovation variance out, so it responds to phi and not to sigma.

SIGNAL / ALPHA
    Breaks that leave the marginal law nearly intact and rewire the dynamics:
    an AR coefficient flip, a persistence change, a switch between random walk
    and mean reversion.  Forensics already found this population to be
    scale- and dependence-dominant and location breaks to be dead.

FALSE SIGNAL
    Residualisation is a double-edged tool.  If the historical AR fit is
    misspecified, the residuals are serially correlated under the null too, and
    a two-sample residual test inflates exactly like the raw one -- worse,
    because the inflation is invisible.  Serial correlation also breaks the
    nominal null of every distance statistic here.

DISAMBIGUATOR
    The nulls are never nominal.  Every statistic is calibrated against the
    empirical distribution of the IDENTICAL statistic computed over
    length-matched windows of the break-free HISTORICAL residual stream -- so a
    misspecified AR fit inflates the null and the online value together, and
    cancels.  The path statistics get the same treatment m01_seq gives its own:
    the recursion is re-run over history and the online path is scored against
    that.  Alongside each dependence LR we emit the innovation-variance LR, so
    a model can tell "phi moved" from "sigma moved" instead of guessing.

CAUSALITY
    AR coefficients are fitted on the HISTORICAL segment only and applied
    forward with `ar_filter_causal`; no online data ever re-fits them.  Every
    online quantity is a cumulative sum over online[:t+1].  Bitwise
    prefix-invariant, checked at atol=0.
"""
from __future__ import annotations

import numpy as np

from sbr.features.base import register
from sbr.nullcal import NullCal

WINDOWS = (32, 128)
BINS = 10


# ------------------------------------------------------------------ nulls
#
# Every null below is length-matched and built from HISTORY only.  The first
# version of this module sized its expanding nulls by `n`, the online length --
# which is both a causality violation and the single forbidden input in this
# competition.  `check_prefix_invariance` caught it at atol=0 on every series.
# The fix is the construction `sbr.nullcal` already uses for rolling means and
# `m01_seq` for detector paths: a log-spaced grid of reference LENGTHS, and
# log-interpolation to whatever elapsed length a row actually has.

LEN_GRID = np.array([8, 16, 32, 64, 128, 256, 512, 1024], dtype=np.int64)
N_RESTART = 16
CLIP = 12.0
EPS = 1e-12


def _cum(a):
    return np.concatenate([[0.0], np.cumsum(np.asarray(a, dtype=np.float64))])


def _roll(c, w, n):
    out = np.full(n, np.nan)
    if w <= n:
        out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


class _GridNull:
    """Median/IQR-sigma of a statistic over history, on a log-spaced length grid.

    ``paths(seg)`` must return the statistic evaluated at EVERY prefix length of
    ``seg`` -- i.e. the same expanding recursion the online side runs -- so the
    null at reference length L is the statistic after L observations, which is
    exactly what a row with elapsed length L should be scored against.
    """

    def __init__(self, h, paths, n_restart=N_RESTART):
        H = len(h)
        self.grid = LEN_GRID[LEN_GRID <= max(H // 2, 8)]
        if len(self.grid) == 0:
            self.grid = LEN_GRID[:1]
        span = int(min(self.grid[-1], H))
        starts = np.unique(np.linspace(0, max(H - span, 0), n_restart).astype(np.int64))
        acc = []
        for s in starts:
            p = np.asarray(paths(h[s:s + span]), dtype=np.float64)
            idx = np.minimum(self.grid, len(p)) - 1
            acc.append(p[idx])
        A = np.array(acc) if acc else np.zeros((1, len(self.grid)))
        with np.errstate(invalid="ignore"):
            self.med = np.nanmedian(A, axis=0)
            q1, q3 = np.nanquantile(A, 0.25, axis=0), np.nanquantile(A, 0.75, axis=0)
            self.sd = np.maximum(np.maximum((q3 - q1) / 1.349, np.nanstd(A, axis=0)), EPS)
        self.med = np.nan_to_num(self.med)
        self.lg = np.log(self.grid.astype(np.float64))

    def z(self, L, v):
        x = np.log(np.maximum(np.asarray(L, dtype=np.float64), 1.0))
        mu = np.interp(x, self.lg, self.med)
        sg = np.exp(np.interp(x, self.lg, np.log(self.sd)))
        return np.clip((np.asarray(v, dtype=np.float64) - mu) / np.maximum(sg, EPS), -CLIP, CLIP)


class _WinNull:
    """Distribution of a WINDOW statistic over every length-w window of history.

    ``path(a, w)`` must be vectorised over the whole historical array, so the
    null costs one pass, not one evaluation per start position.
    """

    def __init__(self, h, w, path):
        v = np.asarray(path(h, w), dtype=np.float64)
        v = v[np.isfinite(v)]
        self.ok = len(v) >= 8
        if self.ok:
            self.sorted = np.sort(v)
            self.med = float(np.median(v))
            q1, q3 = np.quantile(v, [0.25, 0.75])
            self.sd = max((q3 - q1) / 1.349, float(v.std()), EPS)

    def z(self, v):
        v = np.asarray(v, dtype=np.float64)
        if not self.ok:
            return np.full(len(v), np.nan)
        return np.clip((v - self.med) / self.sd, -CLIP, CLIP)


# --------------------------------------------------------------- statistics
def _reflect_cusum(y):
    """S_t = max(0, S_{t-1} + y_t) as one cumsum plus a running minimum."""
    c = np.cumsum(np.asarray(y, dtype=np.float64))
    return c - np.minimum.accumulate(np.concatenate([[0.0], c]))[:-1]


def _occupancy_distances(u, w, bins=BINS):
    """Closed-form divergences of the PIT occupancy vector against uniform.

    ``w=None`` gives the EXPANDING occupancy at every prefix length; an integer
    ``w`` gives the trailing-window version at every position.  One cumsum per
    bin, so the whole family is O(bins * n) and history is never re-sorted.
    """
    u = np.asarray(u, dtype=np.float64)
    n = len(u)
    if n == 0:
        return {k: np.zeros(0) for k in ("chi2", "js", "ks", "w1", "cvm", "ene")}
    idx = np.clip((u * bins).astype(np.int64), 0, bins - 1)
    C = np.stack([_cum((idx == b).astype(np.float64)) for b in range(bins)])
    if w is None:
        occ = C[:, 1:] / np.arange(1, n + 1)
        valid = np.ones(n, bool)
    else:
        occ = np.full((bins, n), np.nan)
        if w <= n:
            occ[:, w - 1:] = (C[:, w:] - C[:, :-w]) / w
        valid = ~np.isnan(occ[0])
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
    out = {"chi2": chi2, "js": js, "ks": ks, "w1": w1, "cvm": cvm, "ene": ene}
    return {k: np.where(valid, v, np.nan) for k, v in out.items()}


def _dep_lr_path(x, w, phi0):
    """2*LLR for a free AR(1) coefficient vs ``phi0``, variance profiled out.

    ``w=None`` expanding, integer ``w`` trailing.  Returns (lr, phi, s2_free).
    Vectorised over the whole array, so it doubles as the historical null path.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    xl = np.concatenate([[0.0], x[:-1]])[:n]
    c_xx, c_xy, c_yy = _cum(xl * xl), _cum(xl * x), _cum(x * x)
    if w is None:
        sxx, sxy, syy = c_xx[1:], c_xy[1:], c_yy[1:]
        m = np.arange(1, n + 1, dtype=np.float64)
    else:
        sxx = _roll(c_xx, w, n) * w
        sxy = _roll(c_xy, w, n) * w
        syy = _roll(c_yy, w, n) * w
        m = np.full(n, float(w))
    phi = sxy / np.maximum(sxx, EPS)
    free = np.maximum((syy - 2 * phi * sxy + phi * phi * sxx) / m, EPS)
    base = np.maximum((syy - 2 * phi0 * sxy + phi0 * phi0 * sxx) / m, EPS)
    return m * np.log(base / free), phi, free

def _decay_max(v, rho):
    out = np.empty(len(v), dtype=np.float64)
    acc = -1.0e18
    for i in range(len(v)):
        a = acc * rho
        acc = v[i] if v[i] > a else a
        out[i] = acc
    return out

try:
    from numba import njit
    _decay_max = njit(cache=True, fastmath=False)(_decay_max)
except Exception:                                            # pragma: no cover
    pass


# ------------------------------------------------------------------ module
@register("m12_rdep", version="2", owner="claude-wave5")
def build(ctx):
    n = ctx.n
    L = np.arange(1, n + 1, dtype=np.float64)
    hp = ctx.hp
    cols: list[str] = []
    out: list[np.ndarray] = []

    def add(name, v):
        v = np.asarray(v, dtype=np.float64)
        assert v.shape == (n,), f"{name}: {v.shape} != ({n},)"
        cols.append(name)
        out.append(v)

    z = (ctx.online - hp.mu) / hp.sd
    zh = (ctx.hist - hp.mu) / hp.sd
    e = (ctx.ar_online / hp.ar_sigma) if ctx.ar_online is not None else z
    eh = np.asarray(ctx.hist_tr.get("res_mean", zh), dtype=np.float64)

    # ------- W5-E4: distribution distances on the AR-RESIDUAL PIT ----------
    hs = np.sort(eh)
    nh = len(hs)
    lo = np.searchsorted(hs, e, side="left")
    hi = np.searchsorted(hs, e, side="right")
    u = (0.5 * (lo + hi) + 0.5) / (nh + 1.0)
    uh = (np.searchsorted(hs, eh, side="left") + 0.5) / (nh + 1.0)

    DIST = ("chi2", "js", "ks", "w1", "cvm", "ene")
    Dexp = _occupancy_distances(u, None)
    gn_exp = {k: _GridNull(uh, lambda seg, kk=k: _occupancy_distances(seg, None)[kk])
              for k in DIST}
    for k in DIST:
        add(f"rd_exp_{k}", gn_exp[k].z(L, Dexp[k]))
    for w in WINDOWS:
        Dw = _occupancy_distances(u, w)
        for k in DIST:
            wn = _WinNull(uh, w, lambda a, ww, kk=k: _occupancy_distances(a, ww)[kk])
            add(f"rd_w{w}_{k}", wn.z(Dw[k]))

    # ------- W5-E4b: robust two-sample SCALE on the residual ---------------
    # Rolling means of transforms, so `sbr.nullcal` is the right engine and is
    # reused verbatim rather than reimplemented: it already does length-matched
    # log-interpolated nulls and is covered by the wave-1 tests.
    med_h = float(np.median(eh))
    q99 = float(np.quantile(np.abs(eh - med_h), 0.99))
    tr_on = {"bf": np.abs(e - med_h), "lv": (e - med_h) ** 2,
             "tail": (np.abs(e - med_h) > q99).astype(np.float64), "e2": e * e}
    tr_h = {"bf": np.abs(eh - med_h), "lv": (eh - med_h) ** 2,
            "tail": (np.abs(eh - med_h) > q99).astype(np.float64), "e2": eh * eh}
    nc = NullCal(tr_h)
    cums = {k: _cum(v) for k, v in tr_on.items()}
    for k in ("bf", "lv", "tail"):
        add(f"rs_exp_{k}", np.clip(nc.z(k, L, cums[k][1:] / L), -CLIP, CLIP))
        for w in WINDOWS:
            v = _roll(cums[k], w, n)
            zz = np.full(n, np.nan)
            ok = ~np.isnan(v)
            if ok.any():
                zz[ok] = np.clip(nc.z(k, w, v[ok]), -CLIP, CLIP)
            add(f"rs_w{w}_{k}", zz)

    # ------- W5-E5: residual CUSUM and CUSUMSQ PATHS -----------------------
    k_drift = 0.5
    for nm, y_on, y_h in (("rc", e, eh), ("rq", e * e - 1.0, eh * eh - 1.0)):
        for sgn, sfx in ((1.0, "up"), (-1.0, "dn")):
            p_on = _reflect_cusum(sgn * y_on - k_drift)
            gn = _GridNull(y_h, lambda seg, sg=sgn: _reflect_cusum(sg * seg - k_drift))
            zp = gn.z(L, p_on)
            pk = np.maximum.accumulate(zp)
            dpk = _decay_max(zp, 0.98)
            arg = np.maximum.accumulate(np.where(zp >= pk - 1e-12, L, 0.0))
            tag = f"{nm}{sfx}"
            add(f"{tag}_cur", zp)
            add(f"{tag}_pk", pk)
            add(f"{tag}_dpk", dpk)
            add(f"{tag}_tsp", np.log1p(L - arg))
            add(f"{tag}_per", np.cumsum(zp > 2.0) / L)

    # ------- W5-E6: dependence LR, innovation variance profiled out --------
    phi0 = float(hp.ar_coef[0]) if len(hp.ar_coef) else 0.0
    lr_e, phi_e, s2_e = _dep_lr_path(z, None, phi0)
    gn_lr = _GridNull(zh, lambda seg: _dep_lr_path(seg, None, phi0)[0])
    add("dl_exp_lr", gn_lr.z(L, lr_e))
    add("dl_exp_phi", np.clip(phi_e - phi0, -3, 3))
    gn_v = _GridNull(zh, lambda seg: _dep_lr_path(seg, None, phi0)[2])
    add("dl_exp_vlr", gn_v.z(L, s2_e))
    for w in WINDOWS:
        lr_w, phi_w, s2_w = _dep_lr_path(z, w, phi0)
        wn = _WinNull(zh, w, lambda a, ww: _dep_lr_path(a, ww, phi0)[0])
        add(f"dl_w{w}_lr", wn.z(lr_w))
        add(f"dl_w{w}_phi", np.clip(np.nan_to_num(phi_w - phi0, nan=np.nan), -3, 3))
        wnv = _WinNull(zh, w, lambda a, ww: _dep_lr_path(a, ww, phi0)[2])
        add(f"dl_w{w}_vlr", wnv.z(s2_w))

    # dependence-vs-scale disambiguator: the LR that says phi moved, minus the
    # one that says sigma moved.  Either alone is confounded; the pair is not.
    add("dl_exp_sep", out[cols.index("dl_exp_lr")] - np.abs(out[cols.index("dl_exp_vlr")]))

    A = np.column_stack(out).astype(np.float32)
    A[~np.isfinite(A)] = np.nan
    return cols, A
