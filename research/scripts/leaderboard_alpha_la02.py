#!/usr/bin/env python
"""LA-02 counterfactual synthetic augmentation.

Full five-fold, real-validation-only training-data intervention:

* RT-1245: same-count fixed-null synthetic control.
* RT-1246: paired persistent-positive vs transient-hard-negative augmentation.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-root", default=os.environ.get("SBR_ROOT", str(REPO)))
    p.add_argument("--cache-dir", default=str(REPO / "cache" / "leaderboard_alpha_2026" / "la02"))
    p.add_argument("--arms", default="c1,candidate")
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--ratio", type=float, default=0.33)
    p.add_argument("--seed", type=int, default=2026082602)
    p.add_argument("--pairs-per-t", type=int, default=64)
    p.add_argument("--pair-seed", type=int, default=20260826)
    p.add_argument("--no-ledger", action="store_true")
    return p.parse_args()


ARGS = parse_args()
os.environ["SBR_ROOT"] = ARGS.artifact_root

for _p in (REPO / "src", REPO / "research" / "scripts"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

import sbr.pipeline as PL  # noqa: E402
from sbr.features.base import REGISTRY, load_all, make_ctx  # noqa: E402
from sbr.metric import ts_auc_flat  # noqa: E402
from wave2_train_ensemble import DEFAULT, FULL, STREAMS  # noqa: E402
from wave5_lib import Ctx, FOLDS, SPECIALISTS, load_oof  # noqa: E402

CACHE = Path(ARGS.cache_dir)
REPORT_DIR = REPO / "research" / "reports" / "leaderboard_alpha_2026"
OOF_DIR = REPO / "research" / "oof"
RESULTS_CSV = REPO / "research" / "RESULTS.csv"

C1_ID = "RT-1245"
CAND_ID = "RT-1246"
STREAM_MAP = {
    "RT-300": "RT-100R",
    "RT-410": "RT-120R",
    "RT-411": "RT-121R",
    "RT-412": "RT-122R",
    "RT-413": "RT-123R",
    "RT-414": "RT-124R",
    "RT-415": "RT-125R",
}
ARM_LABEL = {"c1": C1_ID, "candidate": CAND_ID}
TRANSIENT_TYPES = ("outlier", "shock", "variance_burst", "temporary_displacement")
MECHANISMS = ("location", "scale", "dependence")
AR_ORDER = 5
PAIRFLOW_SPLITS = ("whole", "dominant_cell", "mature_vs_never", "mature_vs_prebreak")


@dataclass
class FixedNull:
    mu: float
    sd: float
    med: float
    mad: float
    phi: np.ndarray
    resid: np.ndarray
    z_tail: np.ndarray


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()


def robust_mad(x):
    x = np.asarray(x, dtype=np.float64)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)) * 1.4826)
    return med, max(mad, 1e-9)


def ar1(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3:
        return 0.0
    z = x - x.mean()
    den = float(np.dot(z[:-1], z[:-1]))
    if den <= 1e-12:
        return 0.0
    return float(np.dot(z[1:], z[:-1]) / den)


def yule_walker(z, p=AR_ORDER):
    z = np.asarray(z, dtype=np.float64)
    n = len(z)
    if p <= 0 or n < 10 * p + 10:
        return np.zeros(p, dtype=np.float64)
    r = np.array([float(np.dot(z[:n - k], z[k:])) / n for k in range(p + 1)])
    if not np.isfinite(r).all() or r[0] <= 0.0:
        return np.zeros(p, dtype=np.float64)
    i = np.arange(p)
    T = r[np.abs(i[:, None] - i[None, :])] + np.eye(p) * (1e-10 * r[0])
    try:
        phi = np.linalg.solve(T, r[1:])
    except np.linalg.LinAlgError:
        return np.zeros(p, dtype=np.float64)
    return phi if np.isfinite(phi).all() else np.zeros(p, dtype=np.float64)


def ar_resid(z, phi):
    p = len(phi)
    if p == 0 or len(z) <= p:
        return np.asarray(z, dtype=np.float64).copy()
    X = np.column_stack([z[p - k - 1: len(z) - k - 1] for k in range(p)])
    return z[p:] - X @ phi


def fixed_null(hist) -> FixedNull:
    h = np.asarray(hist, dtype=np.float64)
    mu = float(h.mean()) if len(h) else 0.0
    sd = max(float(h.std(ddof=1)) if len(h) > 1 else 1.0, 1e-9)
    med, mad = robust_mad(h)
    z = (h - mu) / sd
    phi = yule_walker(z)
    resid = ar_resid(z, phi)
    if len(resid) == 0 or not np.isfinite(resid).all():
        resid = np.array([0.0], dtype=np.float64)
    z_tail = z[-AR_ORDER:] if len(z) >= AR_ORDER else np.r_[np.zeros(AR_ORDER - len(z)), z]
    return FixedNull(mu, sd, med, mad, phi, resid.astype(np.float64), z_tail.astype(np.float64))


def sample_null_path(nl: FixedNull, n: int, rng: np.random.Generator):
    z = np.empty(n, dtype=np.float64)
    eps = rng.choice(nl.resid, size=n, replace=True)
    hist = list(nl.z_tail.astype(float))
    for t in range(n):
        lags = np.array(hist[-AR_ORDER:][::-1], dtype=np.float64)
        z[t] = float(np.dot(nl.phi, lags) + eps[t])
        if not np.isfinite(z[t]):
            z[t] = eps[t]
        z[t] = float(np.clip(z[t], -12.0, 12.0))
        hist.append(z[t])
    x = nl.mu + nl.sd * z
    return x.astype(np.float64), z, eps


def estimate_distributions(d: PL.Data, c: Ctx, rt600, fold: int):
    st = d.st
    train_folds = [g for g in FOLDS if g != fold]
    train_series = np.flatnonzero(np.isin(d.series_fold, train_folds))
    meta = st.meta
    n_online_pool = meta.n_online.to_numpy()[train_series].astype(int)
    break_series = [int(s) for s in train_series if bool(meta.has_break.iloc[int(s)])]

    effects = []
    rel_tau = []
    for s in break_series:
        tau = int(meta.tau_index.iloc[s])
        n = int(meta.n_online.iloc[s])
        if tau < 10 or n - tau < 10:
            continue
        h = st.hist(s)
        o = st.online(s)
        post = o[tau:]
        hmed, hmad = robust_mad(h)
        pmed, pmad = robust_mad(post)
        loc = (pmed - hmed) / hmad
        scale = math.log(max(pmad, 1e-9) / max(hmad, 1e-9))
        dep = ar1(post) - ar1(h)
        if np.isfinite([loc, scale, dep]).all():
            effects.append((loc, scale, dep))
            rel_tau.append(float(tau / max(n - 1, 1)))
    eff = np.asarray(effects, dtype=np.float64)
    if len(eff) == 0:
        eff = np.array([[0.8, 0.4, 0.2], [-0.8, -0.4, -0.2]], dtype=np.float64)
        rel_tau = [0.5]
    rel_tau = np.asarray(rel_tau if rel_tau else [0.5], dtype=np.float64)
    scale = np.median(np.abs(eff), axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    dom = np.argmax(np.abs(eff) / scale[None, :], axis=1)
    weights = np.bincount(dom, minlength=3).astype(np.float64)
    weights = weights / weights.sum()
    pools = {}
    for j, name in enumerate(MECHANISMS):
        vals = eff[dom == j, j]
        if len(vals) == 0:
            vals = eff[:, j]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            vals = np.array([0.5], dtype=np.float64)
        pools[name] = vals.astype(float).tolist()

    # Fold-pure estimate of RT600 no-break false-positive run durations.
    train_rows = d.rows_for(train_folds)
    hb_row = meta.has_break.to_numpy().astype(bool)[d.sidx[train_rows]]
    nb_rows = train_rows[~hb_row]
    scores = rt600[nb_rows]
    thr = float(np.quantile(scores[np.isfinite(scores)], 0.95)) if len(scores) else 1.0
    durations = []
    for s in train_series:
        if bool(meta.has_break.iloc[int(s)]):
            continue
        off = int(st.orow_off[int(s)])
        n = int(meta.n_online.iloc[int(s)])
        rows = np.arange(off, off + n, dtype=np.int64)
        high = rt600[rows] >= thr
        if not high.any():
            continue
        starts = np.flatnonzero(np.r_[high[0], high[1:] & ~high[:-1]])
        ends = np.flatnonzero(np.r_[high[:-1] & ~high[1:], high[-1]]) + 1
        durations.extend([int(e - b) for b, e in zip(starts, ends) if e > b])
    if not durations:
        durations = [1]

    return {
        "train_folds": train_folds,
        "n_train_series": int(len(train_series)),
        "n_online_pool": n_online_pool.tolist(),
        "rel_tau_pool": rel_tau.tolist(),
        "mechanism_weights": {MECHANISMS[i]: float(weights[i]) for i in range(3)},
        "effect_pools": pools,
        "duration_pool": [int(max(1, min(200, x))) for x in durations],
        "rt600_no_break_q95": thr,
    }


def sample_specs(dist: dict, target_rows: int, rng: np.random.Generator):
    specs = []
    total = 0
    n_pool = np.asarray(dist["n_online_pool"], dtype=np.int64)
    rel_pool = np.asarray(dist["rel_tau_pool"], dtype=np.float64)
    weights = np.array([dist["mechanism_weights"][m] for m in MECHANISMS], dtype=np.float64)
    weights = weights / weights.sum()
    while total < target_rows:
        n = int(rng.choice(n_pool))
        n = int(np.clip(n, 10, 999))
        rel = float(rng.choice(rel_pool))
        tau = int(np.clip(round(rel * max(n - 1, 1)), 1, max(n - 2, 1)))
        mech = str(rng.choice(MECHANISMS, p=weights))
        tr = str(rng.choice(TRANSIENT_TYPES))
        specs.append({"n": n, "tau": tau, "mechanism": mech, "transient": tr})
        total += 2 * n
    return specs, total


def choose_effect(dist: dict, mechanism: str, rng: np.random.Generator):
    vals = np.asarray(dist["effect_pools"][mechanism], dtype=np.float64)
    return float(rng.choice(vals))


def apply_persistent(base_x, base_z, eps, nl: FixedNull, tau: int, mechanism: str, value: float):
    x = base_x.copy()
    if mechanism == "location":
        x[tau:] = x[tau:] + value * nl.mad
    elif mechanism == "scale":
        fac = float(np.exp(np.clip(value, -2.0, 2.0)))
        x[tau:] = nl.med + fac * (x[tau:] - nl.med)
    elif mechanism == "dependence":
        phi = nl.phi.copy()
        phi[0] = np.clip(phi[0] + value, -0.95, 0.95)
        z = base_z.copy()
        hist = list(np.r_[nl.z_tail, z[:tau]].astype(float))
        for t in range(tau, len(z)):
            lags = np.array(hist[-AR_ORDER:][::-1], dtype=np.float64)
            z[t] = np.clip(float(np.dot(phi, lags) + eps[t]), -12.0, 12.0)
            hist.append(z[t])
        x = nl.mu + nl.sd * z
    return x.astype(np.float64)


def apply_transient(base_x, nl: FixedNull, dist: dict, transient: str, rng: np.random.Generator):
    x = base_x.copy()
    n = len(x)
    dur = int(rng.choice(np.asarray(dist["duration_pool"], dtype=np.int64)))
    dur = int(np.clip(dur, 1, max(1, min(200, n))))
    start = int(rng.integers(0, max(1, n - dur + 1)))
    end = min(n, start + dur)
    sign = -1.0 if rng.random() < 0.5 else 1.0
    tail = np.abs(nl.resid)
    mag = float(np.quantile(tail, 0.995)) if len(tail) else 3.0
    mag = max(mag, 1e-6)
    if transient == "outlier":
        k = int(min(end - start, rng.integers(1, 4)))
        pos = rng.choice(np.arange(start, end), size=k, replace=False)
        x[pos] += sign * mag * nl.sd
    elif transient == "shock":
        a = np.arange(end - start, dtype=np.float64)
        decay = np.exp(-a / max(float(end - start) / 3.0, 1.0))
        x[start:end] += sign * mag * nl.sd * decay
    elif transient == "variance_burst":
        val = abs(choose_effect(dist, "scale", rng))
        fac = float(np.exp(np.clip(val, 0.0, 2.0)))
        x[start:end] = nl.med + fac * (x[start:end] - nl.med)
    elif transient == "temporary_displacement":
        val = choose_effect(dist, "location", rng)
        x[start:end] += val * nl.mad
    return x.astype(np.float64)


def synthetic_features(hist, online):
    ctx = make_ctx(np.asarray(hist, dtype=np.float64), np.asarray(online, dtype=np.float64))
    mats = []
    cols = []
    for m in FULL:
        cn, arr = REGISTRY[m].fn(ctx)
        mats.append(np.asarray(arr, dtype=np.float32))
        cols.extend([f"{m}::{c}" for c in cn])
    return np.concatenate(mats, axis=1), cols


def build_synthetic_fold(arm: str, fold: int, d: PL.Data, c: Ctx, real_names, rt600):
    outdir = CACHE / arm / f"fold{fold}"
    x_path = outdir / "X.npy"
    y_path = outdir / "y.npy"
    t_path = outdir / "t.npy"
    meta_path = outdir / "meta.json"
    if x_path.exists() and y_path.exists() and t_path.exists() and meta_path.exists():
        return np.load(x_path, mmap_mode="r"), np.load(y_path, mmap_mode="r"), np.load(t_path, mmap_mode="r"), json.load(open(meta_path))

    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(ARGS.seed + fold * 1000 + (0 if arm == "c1" else 500))
    dist = estimate_distributions(d, c, rt600, fold)
    target_rows = int(round(max(STREAMS[STREAM_MAP["RT-300"]]["max_train_rows"], 1) * ARGS.ratio))
    specs, total_rows = sample_specs(dist, target_rows, rng)

    X = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=(total_rows, len(real_names)))
    y = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.int8, shape=(total_rows,))
    tt = np.lib.format.open_memmap(t_path, mode="w+", dtype=np.int32, shape=(total_rows,))

    load_all()
    pos = 0
    st = d.st
    train_series = np.flatnonzero(np.isin(d.series_fold, dist["train_folds"]))
    null_cache: dict[int, FixedNull] = {}
    cols_checked = False
    for j, spec in enumerate(specs):
        sid = int(rng.choice(train_series))
        h = st.hist(sid)
        nl = null_cache.get(sid)
        if nl is None:
            nl = fixed_null(h)
            null_cache[sid] = nl
        n = int(spec["n"])
        if arm == "candidate":
            base_x, base_z, eps = sample_null_path(nl, n, rng)
            val = choose_effect(dist, spec["mechanism"], rng)
            pos_x = apply_persistent(base_x, base_z, eps, nl, int(spec["tau"]), spec["mechanism"], val)
            neg_x = apply_transient(base_x, nl, dist, spec["transient"], rng)
            ys = [
                np.r_[np.zeros(int(spec["tau"]), dtype=np.int8), np.ones(n - int(spec["tau"]), dtype=np.int8)],
                np.zeros(n, dtype=np.int8),
            ]
            ons = [pos_x, neg_x]
        else:
            ons = [sample_null_path(nl, n, rng)[0], sample_null_path(nl, n, rng)[0]]
            ys = [np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.int8)]
        for online, yy in zip(ons, ys):
            Fm, cn = synthetic_features(h, online)
            if not cols_checked:
                if cn != real_names:
                    raise RuntimeError("synthetic feature columns do not match real feature cache")
                cols_checked = True
            m = len(online)
            X[pos:pos + m] = Fm
            y[pos:pos + m] = yy
            tt[pos:pos + m] = np.arange(m, dtype=np.int32)
            pos += m
        if (j + 1) % 100 == 0:
            print(f"  synth {arm} fold {fold}: {j + 1}/{len(specs)} pairs, rows={pos}", flush=True)
    X.flush(); y.flush(); tt.flush()
    meta = {
        "arm": arm,
        "fold": fold,
        "target_rows": target_rows,
        "actual_rows": int(total_rows),
        "n_pairs": int(len(specs)),
        "ratio": float(ARGS.ratio),
        "seed": int(ARGS.seed + fold * 1000 + (0 if arm == "c1" else 500)),
        "distributions": {k: v for k, v in dist.items() if k != "n_online_pool"},
    }
    json.dump(meta, open(meta_path, "w"), indent=2, sort_keys=True)
    return np.load(x_path, mmap_mode="r"), np.load(y_path, mmap_mode="r"), np.load(t_path, mmap_mode="r"), meta


def sample_real_rows(d: PL.Data, folds, max_train_rows: int, sample_mode: str, rng):
    rows = d.rows_for(folds)
    if len(rows) <= max_train_rows:
        return rows
    if sample_mode == "uniform":
        return np.sort(rng.choice(rows, max_train_rows, replace=False))
    if sample_mode == "per_series":
        s = d.sidx[rows]
        order = np.argsort(s, kind="stable")
        rows = rows[order]
        s = s[order]
        bnd = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
        ends = np.r_[bnd[1:], len(s)]
        per = max(1, max_train_rows // len(bnd))
        pick = [
            np.arange(b, e) if e - b <= per else rng.choice(np.arange(b, e), per, replace=False)
            for b, e in zip(bnd, ends)
        ]
        return np.sort(rows[np.concatenate(pick)])
    raise ValueError(sample_mode)


def train_stream(arm: str, sid: str, d: PL.Data, mats, names, col_index, synth_cache):
    out_name = f"{ARM_LABEL[arm]}_{sid}"
    out_path = OOF_DIR / f"{out_name}.npy"
    if out_path.exists():
        arr = np.load(out_path)
        if np.isfinite(arr[d.rows_for(FOLDS)]).all():
            print(f"{out_name}: existing OOF, reuse", flush=True)
            return arr

    cfg = dict(STREAMS[STREAM_MAP[sid]])
    modules = set(cfg["modules"])
    sel = np.array([col_index[n] for n in names if n.split("::")[0] in modules], dtype=np.int64)
    params = dict(DEFAULT)
    params.update(cfg["params"])
    n_round = int(params.pop("n_estimators", 600))
    seed = int(cfg["seed"])
    sample_mode = cfg.get("sample_mode", "uniform")
    max_train_rows = int(cfg["max_train_rows"])

    oof = np.full(len(d.y), np.nan, dtype=np.float32)
    rng_real = np.random.default_rng(seed)
    for f in FOLDS:
        tr_folds = [g for g in FOLDS if g != f]
        va_rows = d.rows_for([f])
        real_rows = sample_real_rows(d, tr_folds, max_train_rows, sample_mode, rng_real)

        Xsyn, ysyn, tsyn, _ = synth_cache[(arm, f)]
        n_syn = int(round(len(real_rows) * ARGS.ratio))
        n_syn = min(n_syn, len(ysyn))
        rng_syn = np.random.default_rng(ARGS.seed + 10000 * list(SPECIALISTS).index(sid) + f)
        syn_rows = np.sort(rng_syn.choice(np.arange(len(ysyn)), n_syn, replace=False))

        Xr = PL._stack(mats, names, real_rows, sel)
        Xs = np.asarray(Xsyn[syn_rows][:, sel], dtype=np.float32)
        Xtr = np.vstack([Xr, Xs])
        ytr = np.concatenate([d.y[real_rows], np.asarray(ysyn[syn_rows], dtype=np.int8)])
        ttr = np.concatenate([d.t[real_rows], np.asarray(tsyn[syn_rows], dtype=np.int32)])
        del Xr, Xs

        p = dict(params)
        if p.get("objective") == "pairwise_t":
            p["objective"] = PL._make_pairwise_t(ttr, ytr, seed=seed)
        ds = lgb.Dataset(Xtr, label=ytr, params=dict(p, objective="binary"),
                         feature_name=[f"f{i}" for i in range(len(sel))])
        booster = lgb.train(p, ds, num_boost_round=n_round)
        del Xtr, ds

        Xva = PL._stack(mats, names, va_rows, sel)
        pred = booster.predict(Xva).astype(np.float32)
        del Xva
        oof[va_rows] = pred
        auc = float(ts_auc_flat(pred, d.y[va_rows], d.t[va_rows]))
        print(
            f"{out_name} fold {f}: {auc:.5f} real={len(real_rows)} synth={n_syn} cols={len(sel)}",
            flush=True,
        )

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    np.save(out_path, oof)
    return oof


def score_vec(c: Ctx, v):
    per = [
        float(ts_auc_flat(v[c.rows[f]], c.d.y[c.rows[f]], c.d.t[c.rows[f]]))
        for f in FOLDS
    ]
    return {
        "mean_ts_auc": float(np.mean(per)),
        "pooled_ts_auc": float(ts_auc_flat(v[c.dev], c.d.y[c.dev], c.d.t[c.dev])),
        "per_fold_ts_auc": per,
        "fold_std": float(np.std(per)),
    }


def load_rt600(c: Ctx):
    p = Path(ARGS.artifact_root) / "research" / "oof" / "wave5_S_specialist.npy"
    if p.exists():
        v = np.load(p)
        if np.isfinite(v[c.dev]).all():
            return v.astype(np.float64)
    return c.crossfit_blend(load_oof(SPECIALISTS), SPECIALISTS)


def cell_rows(c: Ctx, rows, split: str):
    y = c.d.y[rows]
    t = c.d.t[rows]
    age = c.age[rows]
    dominant = (t >= 200) & ((y == 0) | (age >= 100))
    if split == "whole":
        return rows
    if split == "dominant_cell":
        return rows[dominant]
    hb = c.has_break[c.d.sidx[rows]]
    if split == "mature_vs_never":
        return rows[dominant & ((y == 1) | ((y == 0) & ~hb))]
    if split == "mature_vs_prebreak":
        return rows[dominant & ((y == 1) | ((y == 0) & hb))]
    raise KeyError(split)


def pair_repair_stats(base, cand, c: Ctx, rows, pairs_per_t: int, seed: int):
    rng = np.random.default_rng(seed)
    yy = c.d.y
    tt = c.d.t
    order = np.argsort(tt[rows], kind="stable")
    sorted_rows = rows[order]
    sorted_t = tt[sorted_rows]
    starts = np.flatnonzero(np.r_[True, sorted_t[1:] != sorted_t[:-1]])
    ends = np.r_[starts[1:], len(sorted_t)]
    repairs = damage = total = 0
    for lo, hi in zip(starts, ends):
        idx = sorted_rows[lo:hi]
        pos = idx[yy[idx] == 1]
        neg = idx[yy[idx] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        k = min(pairs_per_t, len(pos), len(neg))
        pp = rng.choice(pos, k, replace=False)
        nn = rng.choice(neg, k, replace=False)
        base_right = base[pp] > base[nn]
        cand_right = cand[pp] > cand[nn]
        repairs += int((~base_right & cand_right).sum())
        damage += int((base_right & ~cand_right).sum())
        total += int(k)
    return {
        "total_pairs_sampled": int(total),
        "repairs": int(repairs),
        "damage": int(damage),
        "net_pair_lift": int(repairs - damage),
        "damage_rate": float(damage / max(total, 1)),
    }


def within_t_rank_corr(a, b, t, rows):
    rr = rows[np.isfinite(a[rows]) & np.isfinite(b[rows])]
    ra = np.empty(len(rr), dtype=np.float64)
    rb = np.empty(len(rr), dtype=np.float64)
    tt = t[rr]
    order = np.argsort(tt, kind="stable")
    sorted_t = tt[order]
    starts = np.flatnonzero(np.r_[True, sorted_t[1:] != sorted_t[:-1]])
    ends = np.r_[starts[1:], len(sorted_t)]
    for lo, hi in zip(starts, ends):
        loc = order[lo:hi]
        src = rr[loc]
        denom = max(hi - lo, 1)
        ra[loc] = rankdata(a[src], method="average") / denom
        rb[loc] = rankdata(b[src], method="average") / denom
    return float(np.corrcoef(ra, rb)[0, 1])


def append_result_rows(result):
    rows = []
    for key, exp_id, status in (("c1", C1_ID, "control"), ("candidate", CAND_ID, result["verdict"])):
        arm = result["arms"][key]
        notes = (
            f"LA-02 {key}. C0={result['arms']['c0_rt600']['scores']['mean_ts_auc']:.9f}; "
            f"C1={result['arms']['c1']['scores']['mean_ts_auc']:.9f}; "
            f"candidate={result['arms']['candidate']['scores']['mean_ts_auc']:.9f}; "
            f"marginal_vs_clone={result['primary']['marginal_vs_clone']:+.9f}; "
            f"delta_vs_c0={arm['mean_delta_vs_c0']:+.9f}."
        )
        if key == "candidate":
            notes += (
                f" dominant_net={result['pair_flow']['candidate_vs_c0']['dominant_cell']['net_pair_lift']}; "
                f"mature_vs_never_net={result['pair_flow']['candidate_vs_c0']['mature_vs_never']['net_pair_lift']}; "
                f"mature_vs_prebreak_net={result['pair_flow']['candidate_vs_c0']['mature_vs_prebreak']['net_pair_lift']}."
            )
        rows.append({
            "experiment_id": exp_id,
            "date": "2026-08-26",
            "git_sha": result["git_sha"],
            "agent": "codex-leaderboard-alpha",
            "hypothesis": "LA-02 counterfactual synthetic augmentation tests whether fold-pure history-generated persistent-vs-transient examples improve the RT600 architecture by changing training data only.",
            "falsification_condition": "KILL if candidate marginal_vs_clone < +0.0025, positive on <4/5 folds, dominant net <=0, mature-vs-never net <=0, or candidate-C1 < +0.0010.",
            "feature_set": "RT600 seven specialist architecture; real plus synthetic training rows",
            "n_features": "500",
            "model": "seven_lgbm_specialists_equal_scdf_blend",
            "objective": "binary_and_pairwise_t_as_rt600",
            "folds": "0,1,2,3,4",
            "random_seed": str(ARGS.seed),
            "train_series": "",
            "train_rows": str(result["synthetic"]["total_rows_by_arm"][key]),
            "mean_oof_ts_auc": repr(arm["scores"]["mean_ts_auc"]),
            "pooled_oof_ts_auc": repr(arm["scores"]["pooled_ts_auc"]),
            "per_fold_ts_auc": ";".join(f"{x:.5f}" for x in arm["scores"]["per_fold_ts_auc"]),
            "fold_std": repr(arm["scores"]["fold_std"]),
            "persistence": "none",
            "sample_mode": f"real_rt600_caps_plus_{ARGS.ratio:.2f}_synthetic",
            "training_runtime_s": repr(result["runtime_s"]),
            "causal_verified": "same RT600 causal feature modules; fixed per-series history-only null; real validation only; fold-pure generator distributions",
            "test_reduced_touched": "no",
            "lockbox_touched": "no",
            "status": status,
            "notes": notes,
            "protocol": "leaderboard_alpha_2026_la02_full5_counterfactual_augmentation",
        })

    with open(RESULTS_CSV, "r+", newline="") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        existing = {r["experiment_id"] for r in reader}
        dup = [r["experiment_id"] for r in rows if r["experiment_id"] in existing]
        if dup:
            raise RuntimeError(f"RESULTS.csv already contains {dup}")
        fh.seek(0, os.SEEK_END)
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def main():
    t0 = time.time()
    folds = tuple(int(x) for x in ARGS.folds.split(",") if x)
    if folds != FOLDS:
        raise SystemExit("LA-02 is preregistered for full folds 0,1,2,3,4 only")
    arms = tuple(x.strip() for x in ARGS.arms.split(",") if x.strip())
    if set(arms) != {"c1", "candidate"}:
        raise SystemExit("LA-02 run must include both c1 and candidate arms")

    d = PL.Data()
    c = Ctx()
    mats, names = PL.load_features(FULL)
    col_index = {n: i for i, n in enumerate(names)}
    rt600 = load_rt600(c)

    synth_cache = {}
    synth_meta = {"by_arm_fold": {}, "total_rows_by_arm": {"c1": 0, "candidate": 0}}
    for arm in arms:
        for f in FOLDS:
            Xs, ys, ts, meta = build_synthetic_fold(arm, f, d, c, names, rt600)
            synth_cache[(arm, f)] = (Xs, ys, ts, meta)
            synth_meta["by_arm_fold"][f"{arm}_fold{f}"] = meta
            synth_meta["total_rows_by_arm"][arm] += int(meta["actual_rows"])

    arm_oofs = {}
    for arm in arms:
        streams = {}
        for sid in SPECIALISTS:
            streams[f"{ARM_LABEL[arm]}_{sid}"] = train_stream(arm, sid, d, mats, names, col_index, synth_cache)
        arm_oofs[arm] = streams

    c1_blend = c.crossfit_blend(arm_oofs["c1"], list(arm_oofs["c1"]))
    cand_blend = c.crossfit_blend(arm_oofs["candidate"], list(arm_oofs["candidate"]))

    scores = {
        "c0_rt600": score_vec(c, rt600),
        "c1": score_vec(c, c1_blend),
        "candidate": score_vec(c, cand_blend),
    }
    scores["c1"]["mean_delta_vs_c0"] = scores["c1"]["mean_ts_auc"] - scores["c0_rt600"]["mean_ts_auc"]
    scores["candidate"]["mean_delta_vs_c0"] = scores["candidate"]["mean_ts_auc"] - scores["c0_rt600"]["mean_ts_auc"]

    fold_delta_vs_c1 = [
        float(a - b) for a, b in zip(scores["candidate"]["per_fold_ts_auc"], scores["c1"]["per_fold_ts_auc"])
    ]
    fold_delta_vs_c0 = [
        float(a - b) for a, b in zip(scores["candidate"]["per_fold_ts_auc"], scores["c0_rt600"]["per_fold_ts_auc"])
    ]
    marginal = scores["candidate"]["mean_ts_auc"] - scores["c1"]["mean_ts_auc"]
    positive = int(sum(x > 0 for x in fold_delta_vs_c1))

    pair_flow = {
        split: pair_repair_stats(
            rt600, cand_blend, c, cell_rows(c, c.dev, split), ARGS.pairs_per_t, ARGS.pair_seed
        )
        for split in PAIRFLOW_SPLITS
    }
    corr = {
        "candidate_vs_c0_dev": within_t_rank_corr(cand_blend, rt600, c.d.t, c.dev),
        "candidate_vs_c0_dominant": within_t_rank_corr(cand_blend, rt600, c.d.t, cell_rows(c, c.dev, "dominant_cell")),
        "c1_vs_c0_dev": within_t_rank_corr(c1_blend, rt600, c.d.t, c.dev),
        "c1_vs_c0_dominant": within_t_rank_corr(c1_blend, rt600, c.d.t, cell_rows(c, c.dev, "dominant_cell")),
    }

    if marginal >= 0.0050 and positive >= 4:
        verdict = "MAJOR"
    elif marginal >= 0.0030 and positive >= 4:
        verdict = "SERIOUS"
    elif (
        marginal >= 0.0025
        and positive >= 4
        and pair_flow["dominant_cell"]["net_pair_lift"] > 0
        and pair_flow["mature_vs_never"]["net_pair_lift"] > 0
        and marginal >= 0.0010
    ):
        verdict = "WEAK"
    else:
        verdict = "KILL"

    result = {
        "program": "LEADERBOARD ALPHA 2026",
        "experiment": "LA-02 counterfactual synthetic augmentation",
        "git_sha": git_sha(),
        "artifact_root": ARGS.artifact_root,
        "cache_dir": str(CACHE),
        "ratio": float(ARGS.ratio),
        "seed": int(ARGS.seed),
        "pairs_per_t": int(ARGS.pairs_per_t),
        "pair_seed": int(ARGS.pair_seed),
        "runtime_s": float(time.time() - t0),
        "synthetic": synth_meta,
        "arms": {
            "c0_rt600": {"scores": scores["c0_rt600"]},
            "c1": {"exp_id": C1_ID, "scores": scores["c1"], "mean_delta_vs_c0": scores["c1"]["mean_delta_vs_c0"]},
            "candidate": {
                "exp_id": CAND_ID,
                "scores": scores["candidate"],
                "mean_delta_vs_c0": scores["candidate"]["mean_delta_vs_c0"],
            },
        },
        "primary": {
            "marginal_vs_clone": float(marginal),
            "positive_folds_vs_c1": positive,
            "fold_deltas_vs_c1": fold_delta_vs_c1,
            "fold_deltas_vs_c0": fold_delta_vs_c0,
        },
        "pair_flow": {"candidate_vs_c0": pair_flow},
        "within_t_rank_corr": corr,
        "gate": {
            "marginal_vs_clone_ge_0_0025": bool(marginal >= 0.0025),
            "positive_folds_ge_4": bool(positive >= 4),
            "dominant_pair_net_positive": bool(pair_flow["dominant_cell"]["net_pair_lift"] > 0),
            "mature_vs_never_net_positive": bool(pair_flow["mature_vs_never"]["net_pair_lift"] > 0),
            "candidate_beats_c1_ge_0_0010": bool(marginal >= 0.0010),
        },
        "verdict": verdict,
        "oof_streams": {arm: [f"{ARM_LABEL[arm]}_{s}.npy" for s in SPECIALISTS] for arm in arms},
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "la02_counterfactual_augmentation.json"
    csv_path = REPORT_DIR / "la02_counterfactual_augmentation_summary.csv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["arm", "mean_ts_auc", "pooled_ts_auc", "per_fold_ts_auc", "delta_vs_c0", "marginal_vs_c1"])
        for key in ("c0_rt600", "c1", "candidate"):
            s = result["arms"][key]["scores"]
            w.writerow([
                key,
                s["mean_ts_auc"],
                s["pooled_ts_auc"],
                ";".join(f"{x:.9f}" for x in s["per_fold_ts_auc"]),
                result["arms"][key].get("mean_delta_vs_c0", ""),
                result["primary"]["marginal_vs_clone"] if key == "candidate" else "",
            ])
    if not ARGS.no_ledger:
        append_result_rows(result)
    print(json.dumps({
        "json": str(json_path),
        "csv": str(csv_path),
        "verdict": verdict,
        "marginal_vs_clone": result["primary"]["marginal_vs_clone"],
        "candidate_mean_ts_auc": scores["candidate"]["mean_ts_auc"],
        "c1_mean_ts_auc": scores["c1"]["mean_ts_auc"],
        "c0_mean_ts_auc": scores["c0_rt600"]["mean_ts_auc"],
        "runtime_s": result["runtime_s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

