#!/usr/bin/env python3
"""AGENT 02 -- 2026 data forensics / break taxonomy.

READ-ONLY analysis over the DEV folds of cache/store (fold >= 0 in
research/folds/folds.parquet).  The lockbox (fold == -1) and X_test.reduced are
never touched.

Stage `compute`  : per-series segment characterisation + effect sizes +
                   horizon statistics -> research/artifacts/_taxo_raw.parquet
Stage `analyze`  : calibration against the no-break placebo null, taxonomy
                   assignment, detectability curves, DGP / tau interaction ->
                   research/artifacts/break_taxonomy.parquet (+ tables)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/sb/src")
from sbr.store import load_store  # noqa: E402

ROOT = "/home/claude/sb"
ART = os.path.join(ROOT, "research/artifacts")
RAW_PQ = os.path.join(ART, "_taxo_raw.parquet")
OUT_PQ = os.path.join(ART, "break_taxonomy.parquet")
TAB_JSON = os.path.join(ART, "_taxo_tables.json")

SEED = 20260818
MIN_SEG = 24          # minimum points to characterise a segment at all
MIN_PRE = 60          # minimum pre-tau length to use PRE as the reference
HORIZONS = (5, 10, 20, 40, 80, 160, 320, 640)
LAGS = (1, 2, 3, 5, 10, 20)


# --------------------------------------------------------------------------
# primitive statistics
# --------------------------------------------------------------------------
def _mad(x, med=None):
    if med is None:
        med = np.median(x)
    return float(np.median(np.abs(x - med)) * 1.4826)


def acf_at(z, lags, mean=None):
    """Sample ACF at given lags (biased-denominator, standard)."""
    n = len(z)
    zc = z - (z.mean() if mean is None else mean)
    den = float(np.dot(zc, zc))
    out = {}
    for L in lags:
        out[L] = float(np.dot(zc[L:], zc[:-L]) / den) if (n > L + 8 and den > 0) else np.nan
    return out


def pacf_from_acf(r):
    """Levinson-Durbin PACF at lags 1..3 from acf dict keyed by lag."""
    out = [np.nan, np.nan, np.nan]
    try:
        r1, r2, r3 = r[1], r[2], r[3]
        if not np.isfinite([r1, r2, r3]).all():
            return out
        phi = [r1]
        out[0] = r1
        v = 1 - r1 * r1
        if abs(v) < 1e-12:
            return out
        p2 = (r2 - r1 * r1) / v
        out[1] = p2
        phi = [r1 - p2 * r1, p2]
        v2 = v * (1 - p2 * p2)
        if abs(v2) < 1e-12:
            return out
        p3 = (r3 - phi[0] * r2 - phi[1] * r1) / v2
        out[2] = p3
    except Exception:
        pass
    return out


def perm_entropy(x, m=3):
    n = len(x)
    if n < m + 30:
        return np.nan
    W = np.lib.stride_tricks.sliding_window_view(x, m)
    idx = np.argsort(W, axis=1, kind="stable")
    codes = idx @ (m ** np.arange(m))
    cnt = np.bincount(codes, minlength=m ** m).astype(np.float64)
    p = cnt[cnt > 0] / cnt.sum()
    return float(-(p * np.log(p)).sum() / math.log(math.factorial(m)))


def hjorth(x):
    if len(x) < 12:
        return np.nan, np.nan
    d1 = np.diff(x)
    d2 = np.diff(d1)
    s0 = x.std()
    s1 = d1.std()
    s2 = d2.std()
    if s0 <= 0 or s1 <= 0:
        return np.nan, np.nan
    mob = s1 / s0
    comp = (s2 / s1) / mob if s1 > 0 else np.nan
    return float(mob), float(comp)


def spectral_stats(x):
    n = len(x)
    if n < 48:
        return dict(bp_lo=np.nan, bp_mid=np.nan, bp_hi=np.nan, dom_f=np.nan, spec_ent=np.nan)
    z = x - x.mean()
    w = np.hanning(n)
    P = np.abs(np.fft.rfft(z * w)) ** 2
    f = np.fft.rfftfreq(n)
    P = P[1:]
    f = f[1:]
    tot = float(P.sum())
    if tot <= 0:
        return dict(bp_lo=np.nan, bp_mid=np.nan, bp_hi=np.nan, dom_f=np.nan, spec_ent=np.nan)
    p = P / tot
    return dict(
        bp_lo=float(p[f < 1 / 6].sum()),
        bp_mid=float(p[(f >= 1 / 6) & (f < 1 / 3)].sum()),
        bp_hi=float(p[f >= 1 / 3].sum()),
        dom_f=float(f[np.argmax(P)]),
        spec_ent=float(-(p * np.log(p + 1e-300)).sum() / math.log(len(p))),
    )


def theil_sen(y, max_pairs=200, rng=None):
    n = len(y)
    if n < 12:
        return np.nan
    t = np.arange(n, dtype=np.float64)
    i = rng.integers(0, n, max_pairs)
    j = rng.integers(0, n, max_pairs)
    ok = i != j
    i, j = i[ok], j[ok]
    if len(i) < 5:
        return np.nan
    s = (y[i] - y[j]) / (t[i] - t[j])
    return float(np.median(s))


def ols_slope_quad(y):
    n = len(y)
    if n < 12:
        return np.nan, np.nan
    t = (np.arange(n) - (n - 1) / 2.0) / max(n - 1, 1)   # in [-0.5, 0.5]
    X = np.column_stack([np.ones(n), t, t * t - t.var()])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    return float(beta[1]), float(beta[2])


# --------------------------------------------------------------------------
# two-sample distances (1-D, ECDF based, O((n+m) log))
# --------------------------------------------------------------------------
def ecdf_dists(a_sorted, b_sorted, scale, qedges):
    na, nb = len(a_sorted), len(b_sorted)
    if na < 8 or nb < 8:
        return dict(w1=np.nan, energy=np.nan, ks=np.nan, cvm=np.nan, js=np.nan, hell=np.nan)
    v = np.concatenate([a_sorted, b_sorted])
    v.sort()
    d = np.diff(v)
    ca = np.searchsorted(a_sorted, v[:-1], side="right") / na
    cb = np.searchsorted(b_sorted, v[:-1], side="right") / nb
    df = ca - cb
    w1 = float(np.abs(df * d).sum()) / scale
    en = float(2.0 * np.sum(df * df * d)) / scale
    ks = float(np.abs(df).max())
    cvm = float(na * nb / (na + nb) ** 2 * np.sum(df * df) * 2.0)
    # binned divergences on historical-quantile bins
    pa = np.histogram(a_sorted, bins=qedges)[0].astype(np.float64)
    pb = np.histogram(b_sorted, bins=qedges)[0].astype(np.float64)
    pa = (pa + 0.5) / (pa.sum() + 0.5 * len(pa))
    pb = (pb + 0.5) / (pb.sum() + 0.5 * len(pb))
    m = 0.5 * (pa + pb)
    js = float(0.5 * np.sum(pa * np.log2(pa / m)) + 0.5 * np.sum(pb * np.log2(pb / m)))
    hell = float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(pa * pb)))))
    return dict(w1=w1, energy=en, ks=ks, cvm=cvm, js=js, hell=hell)


# --------------------------------------------------------------------------
# reference (historical) parameters
# --------------------------------------------------------------------------
def href(h):
    med = float(np.median(h))
    q = np.quantile(h, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    hs = np.sort(h)
    qe = np.quantile(hs, np.linspace(0, 1, 21))
    qe[0] = -np.inf
    qe[-1] = np.inf
    qe = np.maximum.accumulate(qe)
    # strictly increasing for np.histogram
    for k in range(1, len(qe)):
        if qe[k] <= qe[k - 1]:
            qe[k] = qe[k - 1] + 1e-12
    return dict(
        mu=float(h.mean()),
        sd=max(float(h.std(ddof=1)), 1e-9),
        med=med,
        mad=max(_mad(h, med), 1e-9),
        q01=float(q[0]), q05=float(q[1]), q25=float(q[2]), q50=float(q[3]),
        q75=float(q[4]), q95=float(q[5]), q99=float(q[6]),
        iqr=max(float(q[4] - q[2]), 1e-9),
        absmu=float(np.mean(np.abs(h - med))),
        sorted=hs, qedges=qe,
    )


# --------------------------------------------------------------------------
# full segment characterisation
# --------------------------------------------------------------------------
SEG_KEYS = None


def seg_stats(x, R, rng):
    """Characterise segment ``x`` against historical reference ``R``."""
    n = len(x)
    o = {}
    o["n"] = float(n)
    if n < 5:
        return o
    sd_h, mad_h = R["sd"], R["mad"]
    mu = float(x.mean())
    med = float(np.median(x))
    sd = float(x.std(ddof=1)) if n > 1 else np.nan
    mad = _mad(x, med)
    # LOCATION (scaled by historical sd and MAD)
    o["mean_z"] = (mu - R["mu"]) / sd_h
    o["med_zmad"] = (med - R["med"]) / mad_h
    o["med_zsd"] = (med - R["med"]) / sd_h
    if n >= 10:
        k = int(0.1 * n)
        xs = np.sort(x)
        o["trim_z"] = (float(xs[k:n - k].mean()) - R["mu"]) / sd_h
    else:
        o["trim_z"] = np.nan
    if n >= MIN_SEG:
        q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
        o["dq05"] = (q[0] - R["q05"]) / sd_h
        o["dq25"] = (q[1] - R["q25"]) / sd_h
        o["dq75"] = (q[3] - R["q75"]) / sd_h
        o["dq95"] = (q[4] - R["q95"]) / sd_h
        iqr = max(float(q[3] - q[1]), 1e-12)
        spread = max(float(q[4] - q[0]), 1e-12)
        o["log_iqr_r"] = math.log(iqr / R["iqr"])
        o["qspacing"] = spread / iqr
    else:
        for k_ in ("dq05", "dq25", "dq75", "dq95", "log_iqr_r", "qspacing"):
            o[k_] = np.nan
    # SCALE
    o["log_sd_r"] = math.log(max(sd, 1e-12) / sd_h) if n > 1 else np.nan
    o["log_mad_r"] = math.log(max(mad, 1e-12) / mad_h)
    o["log_absdev_r"] = math.log(max(float(np.mean(np.abs(x - med))), 1e-12) / max(R["absmu"], 1e-12))
    o["var_ratio"] = (sd / sd_h) ** 2 if n > 1 else np.nan
    if n >= MIN_SEG:
        dz = np.abs(x - R["med"]) / sd_h
        o["absdev_m1"] = float(dz.mean())
        o["absdev_m2"] = float((dz ** 2).mean())
    else:
        o["absdev_m1"] = o["absdev_m2"] = np.nan
    # SHAPE (scale/location free where possible)
    if n >= MIN_SEG and sd > 0:
        zs = (x - mu) / sd
        o["skew"] = float((zs ** 3).mean())
        o["kurt"] = float((zs ** 4).mean()) - 3.0
        zr = (x - med) / max(mad, 1e-12)
        o["skew_r"] = float(np.clip(zr, -8, 8).__pow__(3).mean())
        o["kurt_r"] = float(np.clip(zr, -8, 8).__pow__(4).mean()) - 3.0
    else:
        o["skew"] = o["kurt"] = o["skew_r"] = o["kurt_r"] = np.nan
    o["occ_center"] = float(((x > R["q25"]) & (x < R["q75"])).mean())
    o["occ_tail05"] = float(((x < R["q05"]) | (x > R["q95"])).mean())
    o["occ_tail01"] = float(((x < R["q01"]) | (x > R["q99"])).mean())
    o["occ_hi"] = float((x > R["q95"]).mean())
    o["occ_lo"] = float((x < R["q05"]).mean())
    if n >= MIN_SEG:
        cnt = np.histogram(x, bins=R["qedges"])[0].astype(np.float64)
        p = (cnt + 0.5) / (cnt.sum() + 0.5 * len(cnt))
        o["pit_entropy"] = float(-(p * np.log(p)).sum() / math.log(len(p)))
    else:
        o["pit_entropy"] = np.nan
    # DEPENDENCE
    r = acf_at(x, LAGS) if n >= MIN_SEG else {L: np.nan for L in LAGS}
    for L in LAGS:
        o[f"acf{L}"] = r[L]
    r3 = acf_at(x, (1, 2, 3)) if n >= MIN_SEG else {1: np.nan, 2: np.nan, 3: np.nan}
    pa = pacf_from_acf(r3)
    o["pacf1"], o["pacf2"], o["pacf3"] = pa
    if n >= MIN_SEG:
        a = np.abs(x - med)
        s2 = (x - med) ** 2
        ra = acf_at(a, (1, 2, 5))
        rs = acf_at(s2, (1, 2, 5))
        o["aacf1"], o["aacf2"], o["aacf5"] = ra[1], ra[2], ra[5]
        o["sacf1"], o["sacf2"], o["sacf5"] = rs[1], rs[2], rs[5]
        # volatility clustering: dispersion of local sd across blocks of 10
        nb = n // 10
        if nb >= 4:
            bl = x[: nb * 10].reshape(nb, 10)
            bs = bl.std(axis=1)
            o["volvol"] = float(bs.std() / max(bs.mean(), 1e-12))
        else:
            o["volvol"] = np.nan
    else:
        for k_ in ("aacf1", "aacf2", "aacf5", "sacf1", "sacf2", "sacf5", "volvol"):
            o[k_] = np.nan
    # TREND (in historical-sd units across the whole segment)
    if n >= MIN_SEG:
        sl, cu = ols_slope_quad(x)
        o["slope_sd"] = sl / sd_h if np.isfinite(sl) else np.nan
        o["curv_sd"] = cu / sd_h if np.isfinite(cu) else np.nan
        ts = theil_sen(x, 200, rng)
        o["rslope_sd"] = ts * n / sd_h if np.isfinite(ts) else np.nan
        h1 = x[: n // 2].mean()
        h2 = x[n // 2:].mean()
        o["halfdiff_sd"] = float(h2 - h1) / sd_h
    else:
        for k_ in ("slope_sd", "curv_sd", "rslope_sd", "halfdiff_sd"):
            o[k_] = np.nan
    # SPECTRAL / COMPLEXITY
    sp = spectral_stats(x)
    o.update(sp)
    mo, co = hjorth(x)
    o["hj_mob"], o["hj_comp"] = mo, co
    o["pe3"] = perm_entropy(x, 3)
    o["pe4"] = perm_entropy(x, 4) if n >= 120 else np.nan
    return o


# --------------------------------------------------------------------------
# horizon statistics (calibrated near-z-scores against the historical null)
# --------------------------------------------------------------------------
HSTATS = ("tmean", "atmean", "trmean", "tvar", "tmad", "tks", "ttail", "tacf1", "tabsacf1", "tslope")


def horizon_stats(seg, R):
    n = len(seg)
    out = np.full(len(HSTATS), np.nan)
    if n < 4:
        return out
    sn = math.sqrt(n)
    z = (seg - R["mu"]) / R["sd"]
    m = float(z.mean())
    out[0] = m * sn
    out[1] = abs(m) * sn
    out[2] = (float(np.median(seg)) - R["med"]) / R["mad"] * sn / 1.2533
    if n > 2:
        v = float(seg.var(ddof=1)) / R["sd"] ** 2
        out[3] = math.log(max(v, 1e-12)) * math.sqrt(n / 2.0)
        md = _mad(seg)
        out[4] = math.log(max(md, 1e-12) / R["mad"]) * sn
    hs = R["sorted"]
    cd = np.searchsorted(hs, np.sort(seg), side="right") / len(hs)
    emp = np.arange(1, n + 1) / n
    out[5] = float(np.abs(cd - emp).max()) * sn
    tail = float(((seg < R["q05"]) | (seg > R["q95"])).mean())
    out[6] = (tail - 0.10) * sn / 0.3
    if n >= 20:
        r = acf_at(seg, (1,))[1]
        a = np.abs(seg - R["med"])
        ra = acf_at(a, (1,))[1]
        out[7] = r * sn if np.isfinite(r) else np.nan
        out[8] = ra * sn if np.isfinite(ra) else np.nan
        sl, _ = ols_slope_quad(seg)
        out[9] = sl / R["sd"] * math.sqrt(n / 12.0) if np.isfinite(sl) else np.nan
    return out


# --------------------------------------------------------------------------
# transient detection on a segment (used for no-break characterisation)
# --------------------------------------------------------------------------
def roll_mean(x, w):
    c = np.concatenate([[0.0], np.cumsum(x)])
    return (c[w:] - c[:-w]) / w


def transients(x, R):
    """Detect break-lookalike transients; also return the naive-detector peaks."""
    n = len(x)
    o = dict(t_out=0, t_vol=0, t_exc=0, n_out5=0, det_mean=np.nan, det_rmean=np.nan,
             det_vol=np.nan, det_cusum=np.nan)
    if n < 30:
        return o
    zr = (x - R["med"]) / R["mad"]
    az = np.abs(zr)
    o["n_out5"] = int((az > 5).sum())
    # T1 isolated outlier: |z|>5 with both neighbours quiet
    idx = np.flatnonzero(az > 5)
    for i in idx:
        lo = az[i - 1] if i > 0 else 0.0
        hi = az[i + 1] if i < n - 1 else 0.0
        if lo < 3 and hi < 3:
            o["t_out"] = 1
            break
    # T2 transient volatility burst: rolling-20 sd > 2x hist sd for 10..80 pts,
    #     then back under 1.25x for >= 30 pts
    w = 20
    if n >= 2 * w + 40:
        c1 = np.concatenate([[0.0], np.cumsum(x)])
        c2 = np.concatenate([[0.0], np.cumsum(x * x)])
        m = (c1[w:] - c1[:-w]) / w
        v = np.maximum((c2[w:] - c2[:-w]) / w - m * m, 0.0)
        s = np.sqrt(v) / R["sd"]
        hot = s > 2.0
        cool = s < 1.25
        d = np.diff(np.concatenate([[0], hot.view(np.int8), [0]]))
        st = np.flatnonzero(d == 1)
        en = np.flatnonzero(d == -1)
        for a, b in zip(st, en):
            L = b - a
            if 10 <= L <= 80 and b + 30 <= len(s) and cool[b:b + 30].mean() > 0.8:
                o["t_vol"] = 1
                break
    # T3 mean-reverting excursion: rolling-30 |mean| z > 3.5/sqrt(30) for >= 15,
    #     then |z| < 1.5 for >= 30
    w = 30
    if n >= 2 * w + 40:
        rm = roll_mean((x - R["mu"]) / R["sd"], w) * math.sqrt(w)
        hot = np.abs(rm) > 3.5
        cool = np.abs(rm) < 1.5
        d = np.diff(np.concatenate([[0], hot.view(np.int8), [0]]))
        st = np.flatnonzero(d == 1)
        en = np.flatnonzero(d == -1)
        for a, b in zip(st, en):
            if (b - a) >= 15 and b + 30 <= len(rm) and cool[b:b + 30].mean() > 0.8:
                o["t_exc"] = 1
                break
    # naive online detectors on expanding prefixes (t >= 20)
    z = (x - R["mu"]) / R["sd"]
    cz = np.cumsum(z)
    t = np.arange(1, n + 1)
    pm = np.abs(cz / np.sqrt(t))
    o["det_mean"] = float(pm[19:].max())
    cz2 = np.cumsum(z * z)
    vv = cz2 / t
    with np.errstate(divide="ignore", invalid="ignore"):
        pv = np.abs(np.log(np.maximum(vv, 1e-12))) * np.sqrt(t / 2.0)
    o["det_vol"] = float(pv[19:].max())
    # CUSUM over all candidate change points (max standardised mean split)
    if n >= 40:
        k = np.arange(10, n - 10)
        tot = cz[-1]
        left = cz[k - 1]
        right = tot - left
        stat = np.abs(left / k - right / (n - k)) / np.sqrt(1.0 / k + 1.0 / (n - k))
        o["det_cusum"] = float(np.nanmax(stat))
    zr_c = np.cumsum(zr)
    o["det_rmean"] = float((np.abs(zr_c / np.sqrt(t)) / 1.2533)[19:].max())
    return o


# --------------------------------------------------------------------------
# compute stage
# --------------------------------------------------------------------------
def compute(limit=None, log_every=250):
    folds = pd.read_parquet(os.path.join(ROOT, "research/folds/folds.parquet"))
    st = load_store()
    assert (st.meta.id.to_numpy() == folds.id.to_numpy()).all()
    dev = folds.fold.to_numpy() >= 0
    idx = np.flatnonzero(dev)
    if limit:
        idx = idx[:limit]
    rng = np.random.default_rng(SEED)

    # placebo pseudo-tau for no-break series, drawn from the empirical
    # relative-tau distribution of the break series (dev folds only)
    m = st.meta
    br = idx[m.tau_index.to_numpy()[idx] >= 0]
    rel_pool = (m.tau_index.to_numpy()[br] / m.n_online.to_numpy()[br])
    pseudo = {}
    for i in idx:
        if m.tau_index.iloc[i] < 0:
            n_on = int(m.n_online.iloc[i])
            r = rel_pool[rng.integers(0, len(rel_pool))]
            pseudo[i] = int(min(max(int(round(r * n_on)), 0), n_on - 1))

    rows = []
    t0 = time.time()
    for c, i in enumerate(idx):
        r = m.iloc[i]
        n_on = int(r.n_online)
        tau = int(r.tau_index)
        hb = int(r.has_break)
        h = st.hist(i)
        on = st.online(i)
        R = href(h)
        cut = tau if hb else pseudo[i]
        pre = on[:cut]
        post = on[cut:]
        rec = dict(
            id=int(r.id), fold=int(folds.fold.iloc[i]), has_break=hb,
            tau_index=tau, n_hist=int(r.n_hist), n_online=n_on,
            cut=cut, n_pre=len(pre), n_post=len(post),
            rel_tau=cut / n_on,
            hist_mu=R["mu"], hist_sd=R["sd"], hist_mad=R["mad"],
        )
        # segment stats
        sh = seg_stats(h, R, rng)
        sp = seg_stats(pre, R, rng) if len(pre) >= 5 else {"n": float(len(pre))}
        so = seg_stats(post, R, rng) if len(post) >= 5 else {"n": float(len(post))}
        sf = seg_stats(on, R, rng)
        for pfx, d in (("H_", sh), ("P_", sp), ("O_", so), ("F_", sf)):
            for k, v in d.items():
                rec[pfx + k] = v
        # reference choice
        ref_pre = len(pre) >= MIN_PRE
        rec["ref_used"] = "pre" if ref_pre else "hist"
        # distances
        ps = np.sort(post) if len(post) >= 8 else None
        prs = np.sort(pre) if len(pre) >= 8 else None
        if ps is not None:
            for k, v in ecdf_dists(ps, R["sorted"], R["sd"], R["qedges"]).items():
                rec["dOH_" + k] = v
            if prs is not None:
                for k, v in ecdf_dists(ps, prs, R["sd"], R["qedges"]).items():
                    rec["dOP_" + k] = v
            # shape-only distance: both samples standardised by own median/MAD
            zo = (post - np.median(post)) / max(_mad(post), 1e-9)
            zh = (h - R["med"]) / R["mad"]
            dd = ecdf_dists(np.sort(zo), np.sort(zh), 1.0,
                            np.quantile(np.sort(zh), np.linspace(0, 1, 21)) + np.linspace(0, 1e-9, 21))
            rec["dOHs_ks"] = dd["ks"]
            rec["dOHs_w1"] = dd["w1"]
        if prs is not None:
            for k, v in ecdf_dists(prs, R["sorted"], R["sd"], R["qedges"]).items():
                rec["dPH_" + k] = v
        # transients on the pre-cut region and on the whole online segment
        for pfx, seg in (("tr_on_", on), ("tr_pre_", pre)):
            td = transients(seg, R) if len(seg) >= 30 else {}
            for k, v in td.items():
                rec[pfx + k] = v
        # transient baseline on a matched-length random historical window
        wlen = min(len(on), len(h))
        s0 = int(rng.integers(0, len(h) - wlen + 1))
        td = transients(h[s0:s0 + wlen], R)
        for k, v in td.items():
            rec["tr_hw_" + k] = v
        # horizon statistics
        for hz in HORIZONS:
            if n_on - cut >= hz:
                wv = horizon_stats(on[cut:cut + hz], R)
                pv = horizon_stats(on[: cut + hz], R)
            else:
                wv = np.full(len(HSTATS), np.nan)
                pv = np.full(len(HSTATS), np.nan)
            for k, name in enumerate(HSTATS):
                rec[f"w{hz}_{name}"] = wv[k]
                rec[f"p{hz}_{name}"] = pv[k]
        rows.append(rec)
        if (c + 1) % log_every == 0:
            el = time.time() - t0
            print(f"{c+1}/{len(idx)}  {el:.1f}s  eta {el/(c+1)*(len(idx)-c-1):.0f}s", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(ART, exist_ok=True)
    df.to_parquet(RAW_PQ, index=False)
    print("wrote", RAW_PQ, df.shape, f"{time.time()-t0:.1f}s")
    return df


# ==========================================================================
# ANALYZE STAGE
# ==========================================================================
from scipy.stats import rankdata  # noqa: E402

FAMILY_MEMBERS = {
    "loc":   [("d", "mean_z"), ("d", "med_zmad"), ("d", "trim_z")],
    "scale": [("d", "log_sd_r"), ("d", "log_mad_r"), ("d", "log_iqr_r")],
    "dep":   [("d", "acf1"), ("d", "acf2"), ("d", "acf3"), ("d", "acf5"),
              ("d", "acf10"), ("d", "acf20"), ("d", "aacf1"), ("d", "sacf1"),
              ("d", "pacf1")],
    "shape": [("d", "skew"), ("d", "kurt"), ("d", "qspacing"), ("d", "occ_center"),
              ("d", "occ_tail05"), ("d", "pit_entropy"), ("d", "pe3"),
              ("d", "hj_comp"), ("a", "dOHs_ks")],
    "trend": [("o", "slope_sd"), ("o", "halfdiff_sd"), ("o", "rslope_sd"),
              ("o", "curv_sd")],
}
NBUCKETS = [0, 50, 100, 200, 400, 10 ** 9]
THR = 2.0          # -log10 p ; 2.0 == p<0.01 against the placebo null
MIX = 0.60         # runner-up / leader ratio above which we call it mixed


def auc_np(score, y):
    m = np.isfinite(score)
    s, yy = score[m], y[m]
    n1 = int(yy.sum())
    n0 = len(yy) - n1
    if n1 < 5 or n0 < 5:
        return np.nan, n1 + n0
    r = rankdata(s)
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)), n1 + n0


def surprise_col(vals, null_mask, buckets, cap=3.2):
    """-log10(1-F) of |vals| against the placebo null, within n_post buckets."""
    v = np.abs(vals.astype(np.float64))
    out = np.full(len(v), np.nan)
    for b in np.unique(buckets):
        m = buckets == b
        nul = v[m & null_mask]
        nul = nul[np.isfinite(nul)]
        if len(nul) < 50:
            continue
        ns = np.sort(nul)
        tgt = m & np.isfinite(v)
        f = np.searchsorted(ns, v[tgt], side="right") / (len(ns) + 1.0)
        out[tgt] = np.minimum(-np.log10(np.maximum(1.0 - f, 1.0 / (len(ns) + 1.0))), cap)
    return out


def md_table(df, floatfmt="%.3f"):
    df = df.copy()
    cols = list(df.columns)
    lines = ["| " + " | ".join([str(df.index.name or "")] + [str(c) for c in cols]) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for ix, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else floatfmt % v)
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join([str(ix)] + cells) + " |")
    return "\n".join(lines)


def fam_members_eh(mem):
    """hist-referenced twin of a family member list."""
    return [("dh" if k == "d" else k, n) for k, n in mem]


def main():
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    out = []
    d = pd.read_parquet(RAW_PQ)
    nb = d.has_break.to_numpy() == 0
    bk = ~nb
    y = bk.astype(int)
    n_post = d.n_post.to_numpy()
    buckets = np.digitize(n_post, NBUCKETS[1:-1])
    ref_pre = (d.ref_used.to_numpy() == "pre")

    # ---- effect sizes: POST vs REF (mission rule) and POST vs HIST ------
    eff = {}
    for fam, mem in FAMILY_MEMBERS.items():
        for kind, k in mem:
            if kind == "d":
                o = d["O_" + k].to_numpy(np.float64)
                rp = d["P_" + k].to_numpy(np.float64)
                rh = d["H_" + k].to_numpy(np.float64)
                eff["e_" + k] = o - np.where(ref_pre & np.isfinite(rp), rp, rh)
                eff["eh_" + k] = o - rh
            elif kind == "o":
                eff["e_" + k] = d["O_" + k].to_numpy(np.float64)
                eff["eh_" + k] = d["O_" + k].to_numpy(np.float64)
            else:
                eff["e_" + k] = d[k].to_numpy(np.float64)
                eff["eh_" + k] = d[k].to_numpy(np.float64)
    E = pd.DataFrame(eff)

    # ---- stage-1 member surprise, stage-2 family surprise --------------
    S = {}
    for fam, mem in FAMILY_MEMBERS.items():
        cols = []
        for kind, k in mem:
            S["s_" + k] = surprise_col(E["e_" + k].to_numpy(), nb, buckets)
            cols.append(S["s_" + k])
        raw = np.nanmax(np.vstack(cols), axis=0)
        S["S1_" + fam] = raw
        S["S_" + fam] = surprise_col(raw, nb, buckets)      # equalised across families
    SD = pd.DataFrame(S)
    fam_names = list(FAMILY_MEMBERS)
    FS = np.nan_to_num(SD[["S_" + f for f in fam_names]].to_numpy(), nan=-1.0)

    order = np.argsort(-FS, axis=1)
    lead = FS[np.arange(len(FS)), order[:, 0]]
    run2 = FS[np.arange(len(FS)), order[:, 1]]
    lead_name = np.array(fam_names)[order[:, 0]]
    trend_s = FS[:, fam_names.index("trend")]
    cls = np.where(lead < THR, "weak_unclassified",
                   np.where((trend_s >= THR) & (trend_s >= 0.8 * lead), "trend_dominant",
                            np.where(run2 >= MIX * lead, "mixed",
                                     np.char.add(lead_name, "_dominant"))))
    d = pd.concat([d.reset_index(drop=True), E, SD], axis=1)
    d["break_class"] = cls
    d["lead_family"] = lead_name          # always assigned, no threshold
    d["lead_score"] = lead
    d["runner_score"] = run2
    d["n_post_bucket"] = buckets

    # =================================================================
    out.append("## A. Sample, reference availability, comparison used\n")
    out.append(f"- dev series **{len(d)}** (folds 0-4): break {int(bk.sum())} "
               f"({bk.mean()*100:.2f}%), no-break {int(nb.sum())}.\n")
    out.append(f"- no-break series get a PLACEBO cut drawn from the empirical rel-tau "
               f"distribution of break series (seed {SEED}); every statistic below is "
               f"computed identically for both groups.\n")
    tab = pd.DataFrame({
        "n_pre==0": [(d.n_pre[bk] == 0).mean(), (d.n_pre[nb] == 0).mean()],
        "n_pre<24": [(d.n_pre[bk] < 24).mean(), (d.n_pre[nb] < 24).mean()],
        "n_pre<60 -> ref=HIST": [(d.n_pre[bk] < MIN_PRE).mean(), (d.n_pre[nb] < MIN_PRE).mean()],
        "n_post<24": [(d.n_post[bk] < 24).mean(), (d.n_post[nb] < 24).mean()],
        "n_post<60": [(d.n_post[bk] < 60).mean(), (d.n_post[nb] < 60).mean()],
    }, index=["break", "no-break (placebo cut)"])
    tab.index.name = "group"
    out.append(md_table(tab) + "\n")

    out.append("\n## B. The historical segment is pre-standardised (generator artifact)\n")
    hs = pd.DataFrame({
        "hist_mu": d.hist_mu.describe(), "hist_sd(ddof=1)": d.hist_sd.describe(),
        "hist_mad/sd": (d.hist_mad / d.hist_sd).describe(), "H_kurt": d.H_kurt.describe(),
        "H_acf1": d.H_acf1.describe(), "H_aacf1": d.H_aacf1.describe(),
        "H_volvol": d.H_volvol.describe(), "H_pe3": d.H_pe3.describe(),
    })
    out.append(md_table(hs, "%.5f") + "\n")
    nbd = pd.DataFrame({
        "no-break online": [d.loc[nb, c].mean() for c in
                            ["F_mean_z", "F_log_sd_r", "F_log_mad_r", "F_acf1", "F_kurt",
                             "F_skew", "F_occ_tail05", "F_occ_center", "F_slope_sd"]],
        "no-break online (median)": [d.loc[nb, c].median() for c in
                                     ["F_mean_z", "F_log_sd_r", "F_log_mad_r", "F_acf1", "F_kurt",
                                      "F_skew", "F_occ_tail05", "F_occ_center", "F_slope_sd"]],
        "break online (median)": [d.loc[bk, c].median() for c in
                                  ["F_mean_z", "F_log_sd_r", "F_log_mad_r", "F_acf1", "F_kurt",
                                   "F_skew", "F_occ_tail05", "F_occ_center", "F_slope_sd"]],
        "historical value": [0.0, 0.0, 0.0, d.H_acf1.median(), d.H_kurt.median(),
                             d.H_skew.median(), 0.10, 0.50, d.H_slope_sd.median()],
    }, index=["F_mean_z", "F_log_sd_r", "F_log_mad_r", "F_acf1", "F_kurt", "F_skew",
              "F_occ_tail05", "F_occ_center", "F_slope_sd"])
    nbd.index.name = "whole online segment vs hist"
    out.append("\n**Is a no-break online segment distributionally identical to its own history?**\n")
    out.append(md_table(nbd, "%.4f") + "\n")

    # ---- which family carries the signal (full post segment) ----------
    out.append("\n## C. Which family carries the signal (whole post-cut segment)\n")
    fam_rep = {"loc": "mean_z", "scale": "log_sd_r", "scale2": "log_mad_r",
               "dep": "acf1", "dep2": "aacf1", "shape": "kurt", "shape2": "skew",
               "shape3": "dOHs_ks", "trend": "slope_sd", "trend2": "halfdiff_sd"}
    rows = {}
    for nm, k in fam_rep.items():
        r = {}
        for lbl, pfx in (("|POST vs REF|", "e_"), ("|POST vs HIST|", "eh_")):
            v = np.abs(d[pfx + k].to_numpy(np.float64))
            r[lbl], _ = auc_np(v, y)
        rows[f"{nm}: {k}"] = r
    for k in ["dOH_ks", "dOH_w1", "dOH_energy", "dOH_js", "dOH_hell", "dOH_cvm",
              "dOP_ks", "dOP_w1", "dOP_energy", "dOP_js", "dOP_cvm", "dPH_ks"]:
        v = np.abs(d[k].to_numpy(np.float64))
        a, _ = auc_np(v, y)
        rows["dist: " + k] = {"|POST vs REF|": np.nan, "|POST vs HIST|": a} \
            if k.startswith("dOH") or k.startswith("dPH") else {"|POST vs REF|": a, "|POST vs HIST|": np.nan}
    FA = pd.DataFrame(rows).T
    FA.index.name = "statistic"
    out.append("\n**Series-level AUC (break vs no-break placebo), whole post segment:**\n")
    out.append(md_table(FA, "%.4f") + "\n")
    # stratified by n_post
    blab = ["n_post<50", "50-99", "100-199", "200-399", ">=400"]
    rows = {}
    for nm, k in [("loc", "mean_z"), ("scale", "log_sd_r"), ("scale-mad", "log_mad_r"),
                  ("dep", "acf1"), ("shape-kurt", "kurt"), ("shape-ks", "dOHs_ks"),
                  ("trend", "slope_sd"), ("dist-w1", "dOH_w1"), ("dist-ks", "dOH_ks")]:
        col = ("eh_" + k) if ("eh_" + k) in d else k
        v = np.abs(d[col].to_numpy(np.float64))
        r = {}
        for b in range(5):
            m = buckets == b
            a, _ = auc_np(np.where(m, v, np.nan), y)
            r[blab[b]] = a
        rows[nm] = r
    FB = pd.DataFrame(rows).T
    FB.index.name = "family (vs HIST)"
    out.append("\n**Same AUC stratified by post-segment length:**\n")
    out.append(md_table(FB, "%.4f") + "\n")

    # ---- taxonomy -----------------------------------------------------
    out.append("\n## D. Taxonomy\n")
    cf = pd.crosstab(d.break_class, np.where(bk, "break", "no-break"))
    for c in ("break", "no-break"):
        if c not in cf:
            cf[c] = 0
    cf["break_%"] = cf["break"] / int(bk.sum()) * 100
    cf["placebo_%"] = cf["no-break"] / int(nb.sum()) * 100
    cf["excess_%"] = cf["break_%"] - cf["placebo_%"]
    cf.index.name = "class (threshold rule, -log10p >= 2.0)"
    out.append(md_table(cf, "%.2f") + "\n")
    lf = pd.crosstab(d.lead_family, np.where(bk, "break", "no-break"), normalize="columns") * 100
    lf["excess_pp"] = lf["break"] - lf["no-break"]
    lf.index.name = "leading family (argmax, no threshold)"
    out.append("\n**Leading family for every series (no threshold) - the placebo column is "
               "the null composition:**\n")
    out.append(md_table(lf, "%.2f") + "\n")
    tot_ex = cf.loc[cf.index != "weak_unclassified", "excess_%"].clip(lower=0).sum()
    sh = (cf.loc[cf.index != "weak_unclassified", "excess_%"].clip(lower=0) / max(tot_ex, 1e-9) * 100)
    sh.index.name = "class"
    out.append("\n**Excess-over-placebo composition of the DETECTABLE breaks "
               f"(total detectable excess = {tot_ex:.1f}% of break series):**\n")
    out.append(md_table(sh.to_frame("share_of_detectable_%"), "%.1f") + "\n")

    prof = d[bk].groupby("break_class")[["S_" + f for f in fam_names]].mean()
    prof["n"] = d[bk].groupby("break_class").size()
    prof.index.name = "class"
    out.append("\n**Validation - family surprise profile by class (break series, "
               "mean stage-2 -log10 p, cap 3.2):**\n")
    out.append(md_table(prof) + "\n")
    rawprof = d[bk].groupby("break_class")[
        ["e_mean_z", "e_log_sd_r", "e_log_mad_r", "e_acf1", "e_aacf1", "e_kurt", "e_skew",
         "e_dOHs_ks", "e_slope_sd", "e_halfdiff_sd", "dOH_ks", "dOH_w1", "dOH_js"]].agg(
        lambda s: np.nanmedian(np.abs(s)))
    rawprof["n_post_med"] = d[bk].groupby("break_class")["n_post"].median()
    rawprof.index.name = "class"
    out.append("\n**Validation - median |raw effect size| by class (break series):**\n")
    out.append(md_table(rawprof, "%.4f") + "\n")
    rawprof_nb = d[nb].groupby("break_class")[
        ["e_mean_z", "e_log_sd_r", "e_acf1", "e_kurt", "e_slope_sd", "dOH_w1"]].agg(
        lambda s: np.nanmedian(np.abs(s)))
    rawprof_nb.index.name = "class (placebo/no-break)"
    out.append("\n**Same for the PLACEBO (no-break) series assigned to each class - "
               "this is what the class looks like when there is no break:**\n")
    out.append(md_table(rawprof_nb, "%.4f") + "\n")

    # ---- detectability ------------------------------------------------
    out.append("\n## E. Detectability curve\n")
    rows_w, rows_p, nn = {}, {}, {}
    for hz in HORIZONS:
        rw, rp = {}, {}
        for st in HSTATS:
            for mode, dst in (("w", rw), ("p", rp)):
                v = d[f"{mode}{hz}_{st}"].to_numpy(np.float64)
                v = np.abs(v) if st != "tmean" else v
                a, n = auc_np(v, y)
                dst[st] = a
                if st == "atmean":
                    nn[(hz, mode)] = n
        rw["BEST"] = max([x for x in rw.values() if np.isfinite(x)], default=np.nan)
        rp["BEST"] = max([x for x in rp.values() if np.isfinite(x)], default=np.nan)
        rows_w[hz] = rw
        rows_p[hz] = rp
    AW = pd.DataFrame(rows_w).T
    AP = pd.DataFrame(rows_p).T
    AW.index.name = "horizon h"
    AP.index.name = "horizon h"
    AW["n_series"] = [nn[(h, "w")] for h in AW.index]
    AP["n_series"] = [nn[(h, "p")] for h in AP.index]
    out.append("\n**AUC, ORACLE WINDOW: statistic on online[tau : tau+h] only "
               "(needs to know tau; upper bound):**\n")
    out.append(md_table(AW, "%.4f") + "\n")
    out.append("\n**AUC, REALISTIC PREFIX: statistic on online[0 : tau+h], i.e. what an "
               "online detector actually sees at elapsed h:**\n")
    out.append(md_table(AP, "%.4f") + "\n")
    # fixed cohort (n_post >= 160) removes cohort composition drift
    coh = n_post >= 160
    rows = {}
    for hz in [h for h in HORIZONS if h <= 160]:
        r = {}
        for st in HSTATS:
            v = np.where(coh, d[f"w{hz}_{st}"].to_numpy(np.float64), np.nan)
            v = np.abs(v) if st != "tmean" else v
            r[st], _ = auc_np(v, y)
        r["BEST"] = max([x for x in r.values() if np.isfinite(x)], default=np.nan)
        rows[hz] = r
    AC = pd.DataFrame(rows).T
    AC.index.name = "horizon h"
    out.append(f"\n**Oracle-window AUC inside a FIXED cohort (n_post >= 160, "
               f"{int((coh & bk).sum())} break / {int((coh & nb).sum())} no-break):**\n")
    out.append(md_table(AC, "%.4f") + "\n")

    # per class, oracle window: vs ALL no-break (biased) and vs class-matched placebo
    fam_stat = {"loc_dominant": "atmean", "scale_dominant": "tvar", "dep_dominant": "tacf1",
                "shape_dominant": "tks", "trend_dominant": "tslope", "mixed": "tks",
                "weak_unclassified": "tks"}
    for label, matched in (("vs ALL no-break (oracle class assignment -> optimistic)", False),
                           ("vs class-MATCHED placebo (honest)", True)):
        rows = {}
        for cl in sorted(d.break_class.unique()):
            mpos = (d.break_class.to_numpy() == cl) & bk
            mneg = ((d.break_class.to_numpy() == cl) & nb) if matched else nb
            if mpos.sum() < 20 or mneg.sum() < 20:
                continue
            r = {}
            for hz in HORIZONS:
                best = np.nan
                for st in HSTATS:
                    v = d[f"w{hz}_{st}"].to_numpy(np.float64)
                    v = np.abs(v) if st != "tmean" else v
                    sc = np.concatenate([v[mpos], v[mneg]])
                    yy = np.concatenate([np.ones(mpos.sum()), np.zeros(mneg.sum())])
                    a, _ = auc_np(sc, yy)
                    if np.isfinite(a) and (not np.isfinite(best) or a > best):
                        best = a
                r[hz] = best
            r["n_pos"] = int(mpos.sum())
            r["n_neg"] = int(mneg.sum())
            rows[cl] = r
        CA = pd.DataFrame(rows).T
        CA.index.name = "class"
        out.append(f"\n**Best-single-statistic oracle-window AUC by class and horizon, {label}:**\n")
        out.append(md_table(CA, "%.3f") + "\n")

    st_meta = d[bk]
    el = [float(np.minimum(np.maximum(st_meta.n_online - st_meta.tau_index, 0), h).sum())
          for h in HORIZONS]
    tot_pos = float((st_meta.n_online - st_meta.tau_index).clip(lower=0).sum())
    ER = pd.DataFrame({"cum_share_of_positive_rows": [e / tot_pos for e in el],
                       "prefix_BEST_AUC": [AP.loc[h, "BEST"] for h in HORIZONS]},
                      index=list(HORIZONS))
    ER.index.name = "elapsed <= h"
    out.append("\n**Where the metric is decided: share of all post-break online rows within "
               "h steps of tau, next to the prefix AUC available there:**\n")
    out.append(md_table(ER, "%.4f") + "\n")

    # ---- no-break transients ------------------------------------------
    out.append("\n## F. No-break series: transients that look like breaks\n")
    tr = pd.DataFrame({
        "no-break online": [d.tr_on_t_out[nb].mean(), d.tr_on_t_vol[nb].mean(),
                            d.tr_on_t_exc[nb].mean(), (d.tr_on_n_out5[nb] > 0).mean(),
                            (d.tr_on_t_out[nb] + d.tr_on_t_vol[nb] + d.tr_on_t_exc[nb] > 0).mean()],
        "matched-length hist window (no-break)": [
            d.tr_hw_t_out[nb].mean(), d.tr_hw_t_vol[nb].mean(), d.tr_hw_t_exc[nb].mean(),
            (d.tr_hw_n_out5[nb] > 0).mean(),
            (d.tr_hw_t_out[nb] + d.tr_hw_t_vol[nb] + d.tr_hw_t_exc[nb] > 0).mean()],
        "pre-tau region (break series)": [
            d.tr_pre_t_out[bk].mean(), d.tr_pre_t_vol[bk].mean(), d.tr_pre_t_exc[bk].mean(),
            (d.tr_pre_n_out5[bk] > 0).mean(),
            (d.tr_pre_t_out[bk] + d.tr_pre_t_vol[bk] + d.tr_pre_t_exc[bk] > 0).mean()],
        "whole online (break series)": [
            d.tr_on_t_out[bk].mean(), d.tr_on_t_vol[bk].mean(), d.tr_on_t_exc[bk].mean(),
            (d.tr_on_n_out5[bk] > 0).mean(),
            (d.tr_on_t_out[bk] + d.tr_on_t_vol[bk] + d.tr_on_t_exc[bk] > 0).mean()],
    }, index=["isolated outlier |z_rob|>5 with quiet neighbours", "transient vol burst",
              "mean-reverting excursion", "any |z_rob|>5 point", "ANY of the three"])
    tr.index.name = "transient (rate per series)"
    out.append(md_table(tr, "%.4f") + "\n")
    q = [.5, .9, .95, .99]
    cols = ["tr_on_det_mean", "tr_on_det_rmean", "tr_on_det_vol", "tr_on_det_cusum"]
    pcols = ["tr_pre_det_mean", "tr_pre_det_rmean", "tr_pre_det_vol", "tr_pre_det_cusum"]
    detn = d.loc[nb, cols].quantile(q); detn.index = [f"no-break q{int(x*100)}" for x in q]
    detb = d.loc[bk, cols].quantile(q); detb.index = [f"break q{int(x*100)}" for x in q]
    detp = d.loc[bk, pcols].quantile(q); detp.columns = cols
    detp.index = [f"break pre-tau q{int(x*100)}" for x in q]
    DD = pd.concat([detn, detb, detp]); DD.index.name = "quantile"
    out.append("\n**Naive online detector peaks (max over t of the expanding-prefix statistic):**\n")
    out.append(md_table(DD) + "\n")
    fp = {}
    for c in cols:
        thr = float(np.nanmedian(d.loc[bk, c]))
        fp[c] = dict(thr_at_50pct_break_recall=thr,
                     nobreak_fire_rate=float((d.loc[nb, c] > thr).mean()),
                     series_auc=auc_np(d[c].to_numpy(np.float64), y)[0])
        thr9 = float(np.nanquantile(d.loc[nb, c], 0.95))
        fp[c]["break_recall_at_5pct_FPR"] = float((d.loc[bk, c] > thr9).mean())
    FP = pd.DataFrame(fp).T
    FP.index.name = "naive detector"
    out.append("\n**Naive series-level detectors: operating points:**\n")
    out.append(md_table(FP, "%.4f") + "\n")

    # ---- DGP interaction ----------------------------------------------
    out.append("\n## G. Interaction with the historical DGP\n")
    hp_cols = ["H_kurt", "H_acf1", "H_aacf1", "H_volvol", "H_pe3", "H_spec_ent",
               "H_qspacing", "H_bp_lo", "H_hj_mob"]
    fl, fn = {}, {}
    for c in hp_cols:
        ter = pd.qcut(d[c], 3, labels=["low", "mid", "high"], duplicates="drop")
        g = pd.DataFrame({"t": ter, "lead": d.lead_score, "bk": bk})
        fl[c] = g[g.bk].groupby("t", observed=True)["lead"].mean().to_dict()
        fn[c] = g[~g.bk].groupby("t", observed=True)["lead"].mean().to_dict()
    FL = pd.DataFrame(fl).T
    FN = pd.DataFrame(fn).T
    FL.columns = [f"break {c}" for c in FL.columns]
    FN.columns = [f"placebo {c}" for c in FN.columns]
    FLN = pd.concat([FL, FN], axis=1)
    FLN["break-placebo (high)"] = FLN["break high"] - FLN["placebo high"]
    FLN["break-placebo (low)"] = FLN["break low"] - FLN["placebo low"]
    FLN.index.name = "hist property"
    out.append("\n**Mean leading-family surprise by historical-property tertile "
               "(break vs placebo - the difference is the real effect):**\n")
    out.append(md_table(FLN) + "\n")
    for c in ["H_kurt", "H_acf1", "H_aacf1", "H_volvol"]:
        ter = pd.qcut(d[c], 3, labels=["low", "mid", "high"], duplicates="drop")
        ct = pd.crosstab(ter[bk], d.lead_family[bk], normalize="index") * 100
        ct2 = pd.crosstab(ter[nb], d.lead_family[nb], normalize="index") * 100
        diff = ct - ct2
        diff.index.name = c + " tertile"
        out.append(f"\n**Leading-family composition EXCESS over placebo (pp) by {c} tertile:**\n")
        out.append(md_table(diff, "%.1f") + "\n")
    br_rate = {}
    for c in hp_cols + ["H_n"]:
        ter = pd.qcut(d[c], 3, labels=["low", "mid", "high"], duplicates="drop")
        br_rate[c] = pd.Series(bk).groupby(ter.values, observed=True).mean().to_dict()
    BR = pd.DataFrame(br_rate).T
    BR.index.name = "hist property"
    out.append("\n**Break rate by historical-property tertile (does hist alone leak the label?):**\n")
    out.append(md_table(BR, "%.4f") + "\n")
    aucs = {}
    for c in hp_cols:
        a, _ = auc_np(d[c].to_numpy(np.float64), y)
        aucs[c] = {"AUC(hist stat -> has_break)": a}
    out.append("\n**Series-level AUC of purely historical statistics against has_break "
               "(0.50 = no leakage):**\n")
    out.append(md_table(pd.DataFrame(aucs).T, "%.4f") + "\n")

    # ---- tau / generator artifacts -------------------------------------
    out.append("\n## H. tau, lengths, generator artifacts\n")
    from scipy.stats import kstest, spearmanr
    rt = d.rel_tau[bk].to_numpy()
    ks = kstest(rt, "uniform")
    out.append(f"- rel_tau = tau/n_online on break series: mean {rt.mean():.4f}, sd {rt.std():.4f}, "
               f"KS vs U(0,1) D={ks.statistic:.4f} p={ks.pvalue:.3g}\n")
    dec = pd.Series(np.histogram(rt, bins=10, range=(0, 1))[0] / len(rt) * 100,
                    index=[f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)])
    out.append(f"- rel_tau decile shares (%): {dec.round(2).to_dict()}\n")
    ti = d.tau_index[bk].to_numpy(); no_ = d.n_online[bk].to_numpy()
    out.append(f"- Spearman rho: (tau_index, n_online) {spearmanr(ti, no_).statistic:.4f}; "
               f"(rel_tau, n_online) {spearmanr(rt, no_).statistic:.4f}; "
               f"(rel_tau, n_hist) {spearmanr(rt, d.n_hist[bk]).statistic:.4f}; "
               f"(n_hist, n_online) all-series {spearmanr(d.n_hist, d.n_online).statistic:.4f}\n")
    out.append(f"- break rate by n_online tertile: "
               f"{pd.Series(bk).groupby(pd.qcut(d.n_online,3,labels=['low','mid','high']).values, observed=True).mean().round(4).to_dict()}\n")
    out.append(f"- break rate by n_hist tertile: "
               f"{pd.Series(bk).groupby(pd.qcut(d.n_hist,3,labels=['low','mid','high']).values, observed=True).mean().round(4).to_dict()}\n")
    corr_rows = {}
    for c in ["eh_mean_z", "eh_log_sd_r", "eh_acf1", "eh_kurt", "e_slope_sd", "dOH_w1", "lead_score"]:
        v = np.abs(d[c].to_numpy(np.float64))
        r = {}
        for k in ["tau_index", "rel_tau", "n_online", "n_hist", "n_post", "n_pre"]:
            m = np.isfinite(v) & bk
            r["break: " + k] = spearmanr(v[m], d[k].to_numpy()[m]).statistic
        m2 = np.isfinite(v) & nb
        r["placebo: n_post"] = spearmanr(v[m2], d["n_post"].to_numpy()[m2]).statistic
        r["placebo: rel_tau"] = spearmanr(v[m2], d["rel_tau"].to_numpy()[m2]).statistic
        corr_rows[c] = r
    CC = pd.DataFrame(corr_rows).T
    CC.index.name = "|effect size| vs"
    out.append("\n**Spearman correlation of effect size with tau and lengths:**\n")
    out.append(md_table(CC, "%.4f") + "\n")
    for k, lab in (("n_hist", "n_hist"), ("n_online", "n_online")):
        ter = pd.qcut(d[k], 3, labels=["low", "mid", "high"])
        ct = pd.crosstab(ter[bk], d.lead_family[bk], normalize="index") * 100
        ct2 = pd.crosstab(ter[nb], d.lead_family[nb], normalize="index") * 100
        diff = ct - ct2
        diff.index.name = lab + " tertile"
        out.append(f"\n**Leading-family EXCESS over placebo (pp) by {lab} tertile:**\n")
        out.append(md_table(diff, "%.1f") + "\n")

    d.to_parquet(OUT_PQ, index=False)
    with open("/tmp/a2_tables.md", "w") as f:
        f.write("\n".join(out))
    print("\n".join(out))
    print("\nwrote", OUT_PQ, d.shape)


# ==========================================================================
# RESIDUAL PROBE -- is the break in the innovations rather than the raw series?
# ==========================================================================
def residual_probe(limit=None):
    from sbr.transforms import HistParams, ar_filter_causal, _ar_resid, _fit_ar
    folds = pd.read_parquet(os.path.join(ROOT, "research/folds/folds.parquet"))
    st = load_store()
    base = pd.read_parquet(OUT_PQ, columns=["id", "cut", "has_break", "n_post"])
    cut_by_id = dict(zip(base.id, base.cut))
    idx = np.flatnonzero(folds.fold.to_numpy() >= 0)
    if limit:
        idx = idx[:limit]
    rows = []
    t0 = time.time()
    for c, i in enumerate(idx):
        r = st.meta.iloc[i]
        sid = int(r.id)
        h = st.hist(i)
        on = st.online(i)
        cut = int(cut_by_id[sid])
        rec = {"id": sid}
        for p in (2, 6):
            z = (h - h.mean()) / max(h.std(ddof=1), 1e-9)
            coef = _fit_ar(z, p)
            rh = _ar_resid(z, coef)
            s_h = max(float(rh.std(ddof=1)), 1e-9)
            zo = (on - h.mean()) / max(h.std(ddof=1), 1e-9)
            ro = ar_filter_causal(zo, coef, z)
            post = ro[cut:]
            pre = ro[:cut]
            if len(post) >= 12:
                rec[f"r{p}_logsd"] = math.log(max(float(post.std(ddof=1)), 1e-9) / s_h)
                rec[f"r{p}_mean"] = float(post.mean()) / s_h
                rec[f"r{p}_logmad"] = math.log(max(_mad(post), 1e-9) /
                                               max(_mad(rh), 1e-9))
                if len(pre) >= 60:
                    rec[f"r{p}_logsd_pre"] = math.log(max(float(post.std(ddof=1)), 1e-9) /
                                                      max(float(pre.std(ddof=1)), 1e-9))
                if len(post) >= 30:
                    rec[f"r{p}_acf1"] = acf_at(post, (1,))[1] - (acf_at(rh, (1,))[1] or 0.0)
                    rec[f"r{p}_absacf1"] = (acf_at(np.abs(post), (1,))[1] -
                                            (acf_at(np.abs(rh), (1,))[1] or 0.0))
                    zs = (post - post.mean()) / max(post.std(ddof=1), 1e-9)
                    zhs = (rh - rh.mean()) / max(rh.std(ddof=1), 1e-9)
                    rec[f"r{p}_kurt"] = float((zs ** 4).mean()) - float((zhs ** 4).mean())
                    dd = ecdf_dists(np.sort(post), np.sort(rh), s_h,
                                    np.quantile(np.sort(rh), np.linspace(0, 1, 21)) +
                                    np.linspace(0, 1e-9, 21))
                    rec[f"r{p}_ks"] = dd["ks"]
                    rec[f"r{p}_w1"] = dd["w1"]
        dh = np.diff(h)
        dpost = np.diff(on[cut:]) if len(on) - cut >= 12 else None
        if dpost is not None and len(dpost) > 4:
            rec["diff_logsd"] = math.log(max(float(dpost.std(ddof=1)), 1e-9) /
                                         max(float(dh.std(ddof=1)), 1e-9))
        rows.append(rec)
        if (c + 1) % 1000 == 0:
            print(f"{c+1}/{len(idx)} {time.time()-t0:.0f}s", flush=True)
    R = pd.DataFrame(rows)
    d = pd.read_parquet(OUT_PQ)
    d = d.drop(columns=[c for c in R.columns if c != "id" and c in d.columns])
    d = d.merge(R, on="id", how="left")
    d.to_parquet(OUT_PQ, index=False)
    y = (d.has_break == 1).to_numpy().astype(int)
    res = {}
    for c in R.columns:
        if c == "id":
            continue
        a, n = auc_np(np.abs(d[c].to_numpy(np.float64)), y)
        res[c] = {"AUC": a, "n": n}
    for c in ["eh_log_sd_r", "eh_log_mad_r", "e_log_sd_r", "dOH_w1", "eh_acf1"]:
        a, n = auc_np(np.abs(d[c].to_numpy(np.float64)), y)
        res["(raw) " + c] = {"AUC": a, "n": n}
    cmp_cols = ["e_log_sd_r", "eh_log_sd_r", "r2_logsd", "r6_logsd", "r2_logsd_pre",
                "r6_logsd_pre", "r6_w1", "dOH_w1", "diff_logsd", "r6_acf1"]
    M = np.all([np.isfinite(d[c].to_numpy(np.float64)) for c in cmp_cols], axis=0)
    for c in cmp_cols:
        v = np.where(M, np.abs(d[c].to_numpy(np.float64)), np.nan)
        res["[matched n=%d] %s" % (M.sum(), c)] = {"AUC": auc_np(v, y)[0], "n": int(M.sum())}
    T = pd.DataFrame(res).T
    T.index.name = "residual statistic"
    txt = ("\n## I. AR-residual probe: is the break in the innovations?\n\n" +
           md_table(T, "%.4f") + "\n")
    print(txt)
    with open("/tmp/a2_resid.md", "w") as f:
        f.write(txt)
    print("merged residual columns into", OUT_PQ, d.shape, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["compute", "analyze", "resid", "all"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.stage in ("compute", "all"):
        compute(limit=a.limit or None)
    if a.stage in ("analyze", "all"):
        main()
    if a.stage in ("resid", "all"):
        residual_probe(limit=a.limit or None)
