"""m02_dist -- distributional / PIT / rank break detection.

SIGNAL / ALPHA
    Under stability the historical-ECDF PIT  u_t = F_hist(x_t)  of the online
    segment is Uniform(0,1).  A structural break of *any* kind that changes the
    marginal law -- location, scale, skew, kurtosis, tail weight, bimodality,
    a censoring/clipping change, a regime that only moves the 10th percentile --
    destroys that uniformity.  Moment monitoring (m00_core) sees only the first
    two or four moments of that departure; occupancy of the historical quantile
    bins sees the *whole shape*, and rank statistics see it without any
    dependence on how heavy the tails are.

    Everything here is therefore built on one object: the trailing / expanding
    occupancy vector of the online PIT over B equal-probability historical
    bins.  Every classical two-sample divergence (chi-square, Jensen-Shannon,
    Hellinger, total variation, Cramer-von Mises, Kolmogorov-Smirnov,
    1-D Wasserstein, energy distance) is a closed-form function of that vector
    against the uniform reference, so a whole divergence family costs one
    cumulative sum per bin and no re-sorting of history per step.

CALIBRATION
    A raw JS divergence of 0.04 means "broken" for a Gaussian series and
    "Tuesday" for a heavy-tailed one, and TS-AUC compares series against each
    other at a fixed online index, so raw divergences are close to useless.
    Every statistic here is therefore ALSO emitted as a per-series
    historical-null surprise: the identical statistic evaluated over
    contiguous windows of the (break-free) historical segment of the *same
    length*, sorted once, and the online value looked up by searchsorted.
    Nulls are subsampled to <= 500 start positions per length, which costs
    nothing in resolution (we floor p at 1/2m anyway) and 5-10x in time.

FALSE SIGNAL
    See the per-family stories in research/reports/agent06_distribution.md.
    The short version: a single outlier lands in exactly one extreme bin and
    moves chi-square/max-deviation hard at short windows while leaving the
    interior occupancy, the entropy and the rank CUSUM untouched; a transient
    volatility burst inflates both tails symmetrically and reverts, which is
    why tail *asymmetry* and the long/expanding windows are emitted alongside
    the short ones.
"""
from __future__ import annotations

import numpy as np

from sbr.features.base import register

B_FINE = 20
B_COARSE = 10
W_FIX = (32, 128)
W_TAIL = 64
G_EXP = (16, 64, 256)          # null lengths used to calibrate expanding windows
NMAX = 500                     # max null start positions per length
MIN_NULL = 40                  # fewer than this -> null unusable
CLIP = 6.0
QLEV = (0.1, 0.25, 0.5, 0.75, 0.9)


# ----------------------------------------------------------------- utilities
def _starts(nh: int, w: int) -> np.ndarray | None:
    """Deterministic (strided) start positions of length-w historical windows."""
    m = nh - w + 1
    if m < MIN_NULL:
        return None
    if m > NMAX:
        return np.linspace(0, m - 1, NMAX).astype(np.intp)
    return np.arange(m, dtype=np.intp)


def _pct(rs: np.ndarray, v: np.ndarray) -> np.ndarray:
    lo = np.searchsorted(rs, v, side="left")
    hi = np.searchsorted(rs, v, side="right")
    return 0.5 * (lo + hi) / len(rs)


def _up(rs: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Upper-tail surprise -log10 P(null >= v); for "bigger = more different"."""
    p = 1.0 - _pct(rs, v)
    return np.clip(-np.log10(np.maximum(p, 1.0 / (2.0 * len(rs)))), 0.0, CLIP)


def _sg(rs: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Signed two-sided surprise; for statistics with a meaningful direction."""
    p = _pct(rs, v)
    two = 2.0 * np.minimum(p, 1.0 - p)
    s = np.clip(-np.log10(np.maximum(two, 1.0 / (2.0 * len(rs)))), 0.0, CLIP)
    return np.sign(p - 0.5) * s


def _pit_against(sorted_ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = len(sorted_ref)
    lo = np.searchsorted(sorted_ref, x, side="left")
    hi = np.searchsorted(sorted_ref, x, side="right")
    return (0.5 * (lo + hi) + 0.5) / (n + 1.0)


def _cum(a: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.asarray(a, dtype=np.float64))])


def _onehot_cum(b: np.ndarray, B: int) -> np.ndarray:
    """(len(b)+1, B) cumulative bin counts with a leading zero row."""
    oh = np.zeros((len(b), B))
    oh[np.arange(len(b)), b] = 1.0
    out = np.zeros((len(b) + 1, B))
    np.cumsum(oh, axis=0, out=out[1:])
    return out


def _stats(C: np.ndarray, wl, B: int, mu=None, mu2=None, quant: bool = False) -> dict:
    """All occupancy-derived divergences.  C is (m, B) counts, wl the window
    length of each row (scalar or (m,)).  Reference is uniform-over-bins, which
    is exactly the historical distribution by construction of the PIT."""
    w = np.asarray(wl, dtype=np.float64)
    if w.ndim == 0:
        w = np.full(C.shape[0], float(w))
    wc = w[:, None]
    e = wc / B
    q = C / wc
    dev = C - e
    F = np.cumsum(q, axis=1)
    Fi = F[:, :-1]
    tgt = (np.arange(1, B) / B)[None, :]
    d = Fi - tgt
    ub = 1.0 / B
    mm = 0.5 * (q + ub)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_q = np.where(q > 0, q * np.log(np.maximum(q, 1e-300) / mm), 0.0).sum(1)
        kl_u = (ub * np.log(ub / mm)).sum(1)
        ent = -np.where(q > 0, q * np.log(np.maximum(q, 1e-300)), 0.0).sum(1)
    ad = np.abs(dev)
    out = {
        "chi2": (dev * dev).sum(1) / e[:, 0],
        "ks": np.abs(d).max(1) * np.sqrt(w),
        "cvm": (d * d).sum(1) * (w / B),
        "w1": np.abs(d).sum(1) / B,
        "js": 0.5 * (kl_q + kl_u),
        "hell": 1.0 - np.sqrt(q).sum(1) / np.sqrt(B),
        "tv": 0.5 * np.abs(q - ub).sum(1),
        "ent": ent,
        "maxdev": ad.max(1) / np.sqrt(e[:, 0] * (1.0 - ub)),
        "amax": ad.argmax(1) / (B - 1.0),
    }
    if mu is not None:
        # energy distance to Uniform(0,1):  2E|U-V| - E|U-U'| - E|V-V'|
        #   E|U-V| = mean(u^2 - u + 1/2),  E|V-V'| = 1/3,
        #   E|U-U'| = 2 int F(1-F)  (from the binned CDF)
        euu = 2.0 * (Fi * (1.0 - Fi)).sum(1) / B
        out["ene"] = 2.0 * (mu2 - mu + 0.5) - euu - (1.0 / 3.0)
    if quant:
        Q = {}
        for p in QLEV:
            k = np.minimum((F < p).sum(1), B - 1)
            Fk = np.take_along_axis(F, k[:, None], 1)[:, 0]
            Fkm = np.where(
                k > 0, np.take_along_axis(F, np.maximum(k - 1, 0)[:, None], 1)[:, 0], 0.0
            )
            Q[p] = (k + (p - Fkm) / np.maximum(Fk - Fkm, 1e-12)) / B
        out["q50"] = Q[0.5] - 0.5
        out["qiqr"] = (Q[0.75] - Q[0.25]) - 0.5
        out["q9010"] = (Q[0.9] - Q[0.1]) - 0.8
    return out


class _Stream:
    """One PIT stream (raw / |x-med| / AR residual) at one bin resolution."""

    def __init__(self, uh, uo, B, quant=False, ene=False):
        self.B = B
        self.nh = len(uh)
        self.n = len(uo)
        self.quant = quant
        self.ene = ene
        bh = np.minimum((uh * B).astype(np.intp), B - 1)
        bo = np.minimum((uo * B).astype(np.intp), B - 1)
        self.Ch = _onehot_cum(bh, B)
        self.Co = _onehot_cum(bo, B)
        if ene:
            self.ch_u, self.ch_u2 = _cum(uh), _cum(uh * uh)
            self.co_u, self.co_u2 = _cum(uo), _cum(uo * uo)
        self._null: dict[int, dict] = {}

    def null(self, w: int) -> dict | None:
        w = int(w)
        if w in self._null:
            return self._null[w]
        idx = _starts(self.nh, w)
        if idx is None:
            self._null[w] = None
            return None
        C = self.Ch[idx + w] - self.Ch[idx]
        mu = mu2 = None
        if self.ene:
            mu = (self.ch_u[idx + w] - self.ch_u[idx]) / w
            mu2 = (self.ch_u2[idx + w] - self.ch_u2[idx]) / w
        st = _stats(C, float(w), self.B, mu, mu2, self.quant)
        self._null[w] = {k: np.sort(v) for k, v in st.items()}
        return self._null[w]

    def roll(self, w: int) -> dict | None:
        """Statistics over the trailing window w, rows t = w-1 .. n-1."""
        if w > self.n:
            return None
        C = self.Co[w:] - self.Co[:-w]
        mu = mu2 = None
        if self.ene:
            mu = (self.co_u[w:] - self.co_u[:-w]) / w
            mu2 = (self.co_u2[w:] - self.co_u2[:-w]) / w
        return _stats(C, float(w), self.B, mu, mu2, self.quant)

    def expand(self) -> dict:
        L = np.arange(1, self.n + 1, dtype=np.float64)
        C = self.Co[1:]
        mu = mu2 = None
        if self.ene:
            mu = self.co_u[1:] / L
            mu2 = self.co_u2[1:] / L
        return _stats(C, L, self.B, mu, mu2, self.quant)


class _Emit:
    """Column accumulator with fixed-window placement and null lookup."""

    def __init__(self, n):
        self.n = n
        self.cols: list[str] = []
        self.out: list[np.ndarray] = []

    def add(self, name, v):
        self.cols.append(name)
        self.out.append(np.asarray(v, dtype=np.float64))

    def nan(self, name):
        self.add(name, np.full(self.n, np.nan))

    def fixed(self, name, w, st, key, null, mode="up"):
        """Calibrated trailing-window statistic (NaN before w online points)."""
        if st is None or null is None or key not in null:
            return self.nan(name)
        v = np.full(self.n, np.nan)
        f = _up if mode == "up" else _sg
        v[w - 1:] = f(null[key], st[key])
        self.add(name, v)

    def fixed_raw(self, name, w, st, key):
        if st is None:
            return self.nan(name)
        v = np.full(self.n, np.nan)
        v[w - 1:] = st[key]
        self.add(name, v)

    def exp(self, name, st, key, nulls, L, mode="up"):
        """Calibrated expanding-window statistic, bucketed to the nearest
        null length in log space (a function of t only -> causal)."""
        gs = sorted(g for g, d in nulls.items() if d is not None and key in d)
        if not gs:
            return self.nan(name)
        ga = np.asarray(gs, dtype=np.float64)
        j = np.argmin(np.abs(np.log(L)[:, None] - np.log(ga)[None, :]), axis=1)
        f = _up if mode == "up" else _sg
        v = np.full(self.n, np.nan)
        for k, g in enumerate(gs):
            m = j == k
            if m.any():
                v[m] = f(nulls[g][key], st[key][m])
        self.add(name, v)


def _roll_null(ch: np.ndarray, nh: int, w: int):
    idx = _starts(nh, w)
    if idx is None:
        return None
    return np.sort((ch[idx + w] - ch[idx]) / w)


def _cusum_online(u: np.ndarray) -> np.ndarray:
    """Running max |partial sum of (u-1/2)| normalised by sqrt(L/12)."""
    S = np.cumsum(u - 0.5)
    M = np.maximum.accumulate(np.abs(S))
    L = np.arange(1, len(u) + 1, dtype=np.float64)
    return M / np.sqrt(L / 12.0)


def _cusum_null(u: np.ndarray, w: int):
    idx = _starts(len(u), w)
    if idx is None:
        return None
    c = _cum(u - 0.5)
    try:
        V = np.lib.stride_tricks.sliding_window_view(c, w + 1)[idx]
    except ValueError:
        return None
    base = V[:, :1]
    M = np.maximum((V[:, 1:] - base).max(1), (base - V[:, 1:]).max(1))
    return np.sort(M / np.sqrt(w / 12.0))


@register("m02_dist", version="1", owner="agent6")
def build(ctx):
    n = ctx.n
    nh = len(ctx.hist)
    hp = ctx.hp
    L = np.arange(1, n + 1, dtype=np.float64)
    E = _Emit(n)

    uo = np.asarray(ctx.tr["u"], dtype=np.float64)
    uh = np.asarray(ctx.hist_tr["u"], dtype=np.float64)

    lens_fine = sorted(set(W_FIX) | set(G_EXP))
    lens_alt = sorted({W_FIX[1]} | set(G_EXP))

    # ---------------------------------------------------------------- stream u
    s20 = _Stream(uh, uo, B_FINE, quant=True, ene=True)
    n20 = {w: s20.null(w) for w in lens_fine}
    r20 = {w: s20.roll(w) for w in W_FIX}
    e20 = s20.expand()
    ne20 = {g: n20[g] for g in G_EXP}

    # --- divergence family (dv) ------------------------------------------
    for k in ("chi2", "js", "ks", "ene"):
        E.fixed(f"dv_u_w32_{k}", 32, r20[32], k, n20[32])
    for k in ("chi2", "js", "ks", "cvm", "w1", "ene"):
        E.fixed(f"dv_u_w128_{k}", 128, r20[128], k, n20[128])
    for k in ("chi2", "js", "ks", "ene"):
        E.exp(f"dv_u_exp_{k}", e20, k, ne20, L)
    E.fixed_raw("dv_u_w128_js_raw", 128, r20[128], "js")
    E.add("dv_u_exp_js_raw", e20["js"])

    # --- occupancy-shape family (oc), coarse bins ------------------------
    s10 = _Stream(uh, uo, B_COARSE)
    n10 = {w: s10.null(w) for w in lens_fine}
    r10 = {w: s10.roll(w) for w in W_FIX}
    e10 = s10.expand()
    ne10 = {g: n10[g] for g in G_EXP}
    E.fixed("oc_u_w32_ent", 32, r10[32], "ent", n10[32], mode="sg")
    E.fixed("oc_u_w32_maxdev", 32, r10[32], "maxdev", n10[32])
    E.fixed("oc_u_w128_ent", 128, r10[128], "ent", n10[128], mode="sg")
    E.fixed("oc_u_w128_maxdev", 128, r10[128], "maxdev", n10[128])
    E.fixed_raw("oc_u_w128_amax", 128, r10[128], "amax")
    E.fixed("oc_u_w128_hell", 128, r10[128], "hell", n10[128])
    E.exp("oc_u_exp_ent", e10, "ent", ne10, L, mode="sg")
    E.exp("oc_u_exp_maxdev", e10, "maxdev", ne10, L)
    E.exp("oc_u_exp_tv", e10, "tv", ne10, L)
    E.add("oc_u_exp_amax", e10["amax"])

    # --- quantile-deviation family (qd), PIT space -----------------------
    #     in u-space the historical quantile at level p is p itself, so the
    #     deviation is already scaled by the historical spread.
    for k in ("q50", "qiqr", "q9010"):
        E.fixed_raw(f"qd_w128_{k}_raw", 128, r20[128], k)
    for k in ("q50", "qiqr", "q9010"):
        E.add(f"qd_exp_{k}_raw", e20[k])
    for k in ("q50", "qiqr", "q9010"):
        E.fixed(f"qd_w128_{k}", 128, r20[128], k, n20[128], mode="sg")
    for k in ("q50", "qiqr"):
        E.exp(f"qd_exp_{k}", e20, k, ne20, L, mode="sg")

    # --- tail family (tl) -------------------------------------------------
    hi05, lo05 = (uh > 0.95).astype(float), (uh < 0.05).astype(float)
    hi01, lo01 = (uh > 0.99).astype(float), (uh < 0.01).astype(float)
    ctr_h = ((uh > 0.25) & (uh < 0.75)).astype(float)
    tail_defs = {
        "asym05": (hi05 - lo05, (uo > 0.95).astype(float) - (uo < 0.05).astype(float)),
        "asym01": (hi01 - lo01, (uo > 0.99).astype(float) - (uo < 0.01).astype(float)),
        "ctr": (ctr_h, ((uo > 0.25) & (uo < 0.75)).astype(float)),
    }
    tcum = {k: (_cum(a), _cum(b)) for k, (a, b) in tail_defs.items()}
    for k in ("asym05", "asym01", "ctr"):
        ch, co = tcum[k]
        rs = _roll_null(ch, nh, W_TAIL)
        v = np.full(n, np.nan)
        if rs is not None and W_TAIL <= n:
            v[W_TAIL - 1:] = _sg(rs, (co[W_TAIL:] - co[:-W_TAIL]) / W_TAIL)
        E.add(f"tl_w{W_TAIL}_{k}", v)
    for k in ("asym05", "asym01"):
        ch, co = tcum[k]
        nulls = {g: _roll_null(ch, nh, g) for g in G_EXP}
        gs = sorted(g for g in G_EXP if nulls[g] is not None)
        v = np.full(n, np.nan)
        if gs:
            ga = np.asarray(gs, dtype=np.float64)
            j = np.argmin(np.abs(np.log(L)[:, None] - np.log(ga)[None, :]), axis=1)
            ev = co[1:] / L
            for kk, g in enumerate(gs):
                m = j == kk
                if m.any():
                    v[m] = _sg(nulls[g], ev[m])
        E.add(f"tl_exp_{k}", v)
    co = tcum["asym05"][1]
    E.add("tl_exp_asym05_raw", co[1:] / L)

    # --- rank family (rk) -------------------------------------------------
    chu, chu2 = _cum(uh), _cum(uh * uh)
    cou, cou2 = _cum(uo), _cum(uo * uo)

    def _var_null(w):
        idx = _starts(nh, w)
        if idx is None:
            return None
        m1 = (chu[idx + w] - chu[idx]) / w
        m2 = (chu2[idx + w] - chu2[idx]) / w
        return np.sort(m2 - m1 * m1)

    w = W_FIX[1]
    rs = _var_null(w)
    v = np.full(n, np.nan)
    if rs is not None and w <= n:
        m1 = (cou[w:] - cou[:-w]) / w
        m2 = (cou2[w:] - cou2[:-w]) / w
        v[w - 1:] = _sg(rs, m2 - m1 * m1)
    E.add("rk_w128_var", v)
    vnulls = {g: _var_null(g) for g in G_EXP}
    gs = sorted(g for g in G_EXP if vnulls[g] is not None)
    v = np.full(n, np.nan)
    if gs:
        ga = np.asarray(gs, dtype=np.float64)
        j = np.argmin(np.abs(np.log(L)[:, None] - np.log(ga)[None, :]), axis=1)
        ev = cou2[1:] / L - (cou[1:] / L) ** 2
        for kk, g in enumerate(gs):
            m = j == kk
            if m.any():
                v[m] = _sg(vnulls[g], ev[m])
    E.add("rk_exp_var", v)

    # ---- AR-residual PIT stream (built here from ctx.hist_tr) ------------
    p_ar = len(hp.ar_coef)
    if "res_mean" in ctx.tr and nh - p_ar > 50:
        rh = np.asarray(ctx.hist_tr["res_mean"], dtype=np.float64)[p_ar:]
        ro = np.asarray(ctx.tr["res_mean"], dtype=np.float64)
        sref = np.sort(rh)
        urh = _pit_against(sref, rh)
        uro = _pit_against(sref, ro)
    else:
        urh, uro = uh, uo

    def _cusum_block(uh_, uo_, tag):
        cu = _cusum_online(uo_)
        E.add(f"rk_cusum_{tag}_raw", np.clip(cu, 0, 20))
        nulls = {g: _cusum_null(uh_, g) for g in G_EXP}
        gs_ = sorted(g for g in G_EXP if nulls[g] is not None)
        v_ = np.full(n, np.nan)
        if gs_:
            ga_ = np.asarray(gs_, dtype=np.float64)
            j_ = np.argmin(np.abs(np.log(L)[:, None] - np.log(ga_)[None, :]), axis=1)
            for kk, g in enumerate(gs_):
                m = j_ == kk
                if m.any():
                    v_[m] = _up(nulls[g], cu[m])
        E.add(f"rk_cusum_{tag}", v_)

    _cusum_block(uh, uo, "u")
    _cusum_block(urh, uro, "r")

    # --- |x - med| PIT stream (ab) ----------------------------------------
    svh = np.sort(np.abs(np.asarray(ctx.hist, dtype=np.float64) - hp.med))
    uah = _pit_against(svh, np.abs(np.asarray(ctx.hist, dtype=np.float64) - hp.med))
    uao = _pit_against(svh, np.abs(np.asarray(ctx.online, dtype=np.float64) - hp.med))
    sab = _Stream(uah, uao, B_COARSE)
    nab = {w_: sab.null(w_) for w_ in lens_alt}
    rab = sab.roll(128)
    eab = sab.expand()
    neab = {g: nab[g] for g in G_EXP}
    E.fixed("ab_w128_chi2", 128, rab, "chi2", nab[128])
    E.fixed("ab_w128_ks", 128, rab, "ks", nab[128])
    E.exp("ab_exp_chi2", eab, "chi2", neab, L)
    E.exp("ab_exp_js", eab, "js", neab, L)
    cah, cao = _cum(uah), _cum(uao)
    rsn = _roll_null(cah, len(uah), 128)
    v = np.full(n, np.nan)
    if rsn is not None and 128 <= n:
        v[127:] = _sg(rsn, (cao[128:] - cao[:-128]) / 128.0)
    E.add("ab_w128_mean", v)

    # --- AR-residual PIT stream (rs) --------------------------------------
    srs = _Stream(urh, uro, B_COARSE)
    nrs = {w_: srs.null(w_) for w_ in lens_alt}
    rrs = srs.roll(128)
    ers = srs.expand()
    ners = {g: nrs[g] for g in G_EXP}
    E.fixed("rs_w128_chi2", 128, rrs, "chi2", nrs[128])
    E.fixed("rs_w128_ks", 128, rrs, "ks", nrs[128])
    E.fixed("rs_w128_js", 128, rrs, "js", nrs[128])
    E.exp("rs_exp_chi2", ers, "chi2", ners, L)
    E.exp("rs_exp_js", ers, "js", ners, L)

    A = np.column_stack(E.out).astype(np.float32)
    A[~np.isfinite(A)] = np.nan
    return E.cols, A
