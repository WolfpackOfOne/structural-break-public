#!/usr/bin/env python
"""LA-03 per-series history adaptation.

Full five-fold predictive-null adaptation test:

* RT-1247: global AR(5) no-adaptation control head.
* RT-1248: fixed per-series AR(5)+history-residual-ECDF control head.
* RT-1249: global AR(5) plus per-series affine-adapted candidate head.
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
    p.add_argument("--cache-dir", default=str(REPO / "cache" / "leaderboard_alpha_2026" / "la03"))
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--seed", type=int, default=2026082603)
    p.add_argument("--max-train-rows", type=int, default=1_000_000)
    p.add_argument("--rounds", type=int, default=600)
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
from scipy.special import ndtri  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

import sbr.pipeline as PL  # noqa: E402
from sbr.metric import ts_auc_flat  # noqa: E402
from wave2_train_ensemble import DEFAULT  # noqa: E402
from wave5_lib import Ctx, FOLDS, SPECIALISTS, load_oof  # noqa: E402

CACHE = Path(ARGS.cache_dir)
REPORT_DIR = REPO / "research" / "reports" / "leaderboard_alpha_2026"
OOF_DIR = REPO / "research" / "oof"
RESULTS_CSV = REPO / "research" / "RESULTS.csv"

C0_ID = "RT-1247"
C1_ID = "RT-1248"
CAND_ID = "RT-1249"
EXP_IDS = {"global": C0_ID, "fixed": C1_ID, "adapted": CAND_ID}
ARM_LABELS = {
    "global": "C0 global no-adaptation",
    "fixed": "C1 fixed per-series null",
    "adapted": "candidate affine-adapted",
}

AR_ORDER = 5
RIDGE_LAMBDA = 32.0
ADAPTER_CLIP = 3.0
FEATURE_VERSION = "la03_affine_global_ar5_v1"
FEATURE_NAMES = (
    "resid_z",
    "resid_pit",
    "abs_resid_pit",
    "surp",
    "run_mean_resid_z",
    "run_mean_abs_resid_z",
    "run_mean_surp",
    "run_max_abs_resid_pit",
    "exceed90",
    "lag1_pit",
)
PAIRFLOW_SPLITS = ("whole", "dominant_cell", "mature_vs_never", "mature_vs_prebreak")


@dataclass
class ReferenceDist:
    sig: float
    resid_sorted: np.ndarray
    abs_resid_sorted: np.ndarray
    q90: float


@dataclass
class GlobalState:
    phi: np.ndarray
    ref: ReferenceDist


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()


def robust_norm(hist: np.ndarray):
    h = np.asarray(hist, dtype=np.float64)
    mu = float(h.mean()) if len(h) else 0.0
    sd = float(h.std(ddof=1)) if len(h) > 1 else 1.0
    if not np.isfinite(sd) or sd <= 1e-9:
        sd = 1.0
    return mu, sd, (h - mu) / sd


def yule_walker(z: np.ndarray, p: int = AR_ORDER) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    n = len(z)
    if n < 10 * p + 10:
        return np.zeros(p, dtype=np.float64)
    r = np.array([float(np.dot(z[: n - k], z[k:])) / n for k in range(p + 1)])
    if not np.isfinite(r).all() or r[0] <= 0.0:
        return np.zeros(p, dtype=np.float64)
    i = np.arange(p)
    T = r[np.abs(i[:, None] - i[None, :])] + np.eye(p) * (1e-10 * r[0])
    try:
        phi = np.linalg.solve(T, r[1:])
    except np.linalg.LinAlgError:
        return np.zeros(p, dtype=np.float64)
    return phi if np.isfinite(phi).all() else np.zeros(p, dtype=np.float64)


def ar_predict_strict(z: np.ndarray, phi: np.ndarray, warm: np.ndarray | None = None) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    p = len(phi)
    if p == 0:
        return np.zeros(len(z), dtype=np.float64)
    if warm is None:
        warm_p = np.zeros(p, dtype=np.float64)
    else:
        w = np.asarray(warm, dtype=np.float64)
        warm_p = w[-p:] if len(w) >= p else np.r_[np.zeros(p - len(w)), w]
    base = np.r_[warm_p, z]
    win = np.lib.stride_tricks.sliding_window_view(base, p)[: len(z)]
    return win[:, ::-1] @ phi


def reference_from_resid(resid: np.ndarray) -> ReferenceDist:
    r = np.asarray(resid, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        r = np.array([0.0], dtype=np.float64)
    sig = float(r.std(ddof=1)) if len(r) > 1 else 1.0
    if not np.isfinite(sig) or sig <= 1e-9:
        sig = 1.0
    rz = r / sig
    rz = rz[np.isfinite(rz)]
    if len(rz) == 0:
        rz = np.array([0.0], dtype=np.float64)
    abs_rz = np.abs(rz)
    q90 = float(np.quantile(abs_rz, 0.90)) if len(abs_rz) else 1.0
    q90 = max(q90, 1e-6)
    return ReferenceDist(
        sig=sig,
        resid_sorted=np.sort(rz).astype(np.float64),
        abs_resid_sorted=np.sort(abs_rz).astype(np.float64),
        q90=q90,
    )


def ecdf_eval(sorted_x: np.ndarray, x: np.ndarray) -> np.ndarray:
    s = np.asarray(sorted_x, dtype=np.float64)
    if len(s) == 0:
        s = np.array([0.0], dtype=np.float64)
    ranks = np.searchsorted(s, x, side="right").astype(np.float64)
    return (ranks + 1.0) / (len(s) + 2.0)


def residual_features(resid_z: np.ndarray, ref: ReferenceDist) -> np.ndarray:
    r = np.asarray(resid_z, dtype=np.float64)
    n = len(r)
    out = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
    if n == 0:
        return out.astype(np.float32)

    u = np.clip(ecdf_eval(ref.resid_sorted, r), 1e-6, 1.0 - 1e-6)
    pit = np.clip(ndtri(u), -4.0, 4.0)
    abs_pit = np.abs(pit)
    abs_r = np.abs(r)
    ua = np.clip(ecdf_eval(ref.abs_resid_sorted, abs_r), 1e-6, 1.0 - 1e-6)
    surp = np.clip(-np.log(1.0 - ua + 1.0 / (len(ref.abs_resid_sorted) + 1.0)), 0.0, 12.0)
    k = np.arange(1, n + 1, dtype=np.float64)

    lag1 = np.zeros(n, dtype=np.float64)
    if n > 1:
        lag1[1:] = pit[1:] * pit[:-1]

    out[:, 0] = np.clip(r, -8.0, 8.0)
    out[:, 1] = pit
    out[:, 2] = abs_pit
    out[:, 3] = surp
    out[:, 4] = np.clip(np.cumsum(r) / k, -5.0, 5.0)
    out[:, 5] = np.clip(np.cumsum(abs_r) / k, 0.0, 10.0)
    out[:, 6] = np.clip(np.cumsum(surp) / k, 0.0, 12.0)
    out[:, 7] = np.maximum.accumulate(abs_pit)
    out[:, 8] = 1.0 / (1.0 + np.exp(-np.clip(4.0 * (abs_r - ref.q90), -60.0, 60.0)))
    out[:, 9] = np.clip(lag1, -16.0, 16.0)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def fit_global_state(d: PL.Data, fold: int) -> GlobalState:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"global_state_fold{fold}.npz"
    meta_path = CACHE / f"global_state_fold{fold}.json"
    train_folds = [g for g in FOLDS if g != fold]
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("version") == FEATURE_VERSION and meta.get("train_folds") == train_folds:
            z = np.load(path)
            ref = ReferenceDist(
                sig=float(z["sig"]),
                resid_sorted=z["resid_sorted"].astype(np.float64),
                abs_resid_sorted=z["abs_resid_sorted"].astype(np.float64),
                q90=float(z["q90"]),
            )
            return GlobalState(phi=z["phi"].astype(np.float64), ref=ref)

    train_series = np.flatnonzero(np.isin(d.series_fold, train_folds))
    ac = np.zeros(AR_ORDER + 1, dtype=np.float64)
    ct = np.zeros(AR_ORDER + 1, dtype=np.float64)
    z_cache: list[np.ndarray] = []
    for s in train_series:
        _, _, z = robust_norm(d.st.hist(int(s)))
        z = z.astype(np.float64)
        z_cache.append(z)
        n = len(z)
        if n <= AR_ORDER:
            continue
        for k in range(AR_ORDER + 1):
            ac[k] += float(np.dot(z[: n - k], z[k:]))
            ct[k] += float(n - k)
    r = np.divide(ac, np.maximum(ct, 1.0))
    if r[0] <= 0.0 or not np.isfinite(r).all():
        phi = np.zeros(AR_ORDER, dtype=np.float64)
    else:
        i = np.arange(AR_ORDER)
        T = r[np.abs(i[:, None] - i[None, :])] + np.eye(AR_ORDER) * (1e-10 * r[0])
        try:
            phi = np.linalg.solve(T, r[1:])
        except np.linalg.LinAlgError:
            phi = np.zeros(AR_ORDER, dtype=np.float64)
        if not np.isfinite(phi).all():
            phi = np.zeros(AR_ORDER, dtype=np.float64)

    resid_parts = []
    valid0 = AR_ORDER
    for z in z_cache:
        if len(z) <= valid0:
            continue
        pred = ar_predict_strict(z, phi)
        resid_parts.append(z[valid0:] - pred[valid0:])
    resid = np.concatenate(resid_parts) if resid_parts else np.array([0.0], dtype=np.float64)
    ref = reference_from_resid(resid)
    np.savez_compressed(
        path,
        phi=phi,
        sig=np.array(ref.sig),
        resid_sorted=ref.resid_sorted,
        abs_resid_sorted=ref.abs_resid_sorted,
        q90=np.array(ref.q90),
    )
    meta_path.write_text(json.dumps({
        "version": FEATURE_VERSION,
        "fold": fold,
        "train_folds": train_folds,
        "n_train_series": int(len(train_series)),
        "n_residuals": int(len(resid)),
    }, indent=2, sort_keys=True) + "\n")
    return GlobalState(phi=phi, ref=ref)


def local_fixed_state(z_hist: np.ndarray):
    phi = yule_walker(z_hist)
    pred = ar_predict_strict(z_hist, phi)
    valid = np.arange(len(z_hist)) >= AR_ORDER
    resid = z_hist[valid] - pred[valid] if valid.any() else np.array([0.0], dtype=np.float64)
    return phi, reference_from_resid(resid)


def local_affine_state(z_hist: np.ndarray, global_phi: np.ndarray):
    pred = ar_predict_strict(z_hist, global_phi)
    valid = np.arange(len(z_hist)) >= AR_ORDER
    if valid.sum() < 20:
        a, b = 1.0, 0.0
        resid = z_hist[valid] - pred[valid] if valid.any() else np.array([0.0], dtype=np.float64)
    else:
        X = np.column_stack([pred[valid], np.ones(int(valid.sum()), dtype=np.float64)])
        y = z_hist[valid]
        prior = np.array([1.0, 0.0], dtype=np.float64)
        lhs = X.T @ X + RIDGE_LAMBDA * np.eye(2)
        rhs = X.T @ y + RIDGE_LAMBDA * prior
        try:
            theta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            theta = prior
        theta = np.where(np.isfinite(theta), theta, prior)
        a = float(np.clip(theta[0], -ADAPTER_CLIP, ADAPTER_CLIP))
        b = float(np.clip(theta[1], -ADAPTER_CLIP, ADAPTER_CLIP))
        resid = y - (a * pred[valid] + b)
    return a, b, reference_from_resid(resid)


def features_for_series(hist: np.ndarray, online: np.ndarray, arm: str, global_state: GlobalState | None):
    mu, sd, z_hist = robust_norm(hist)
    z_online = (np.asarray(online, dtype=np.float64) - mu) / sd
    warm = z_hist[-AR_ORDER:] if len(z_hist) >= AR_ORDER else np.r_[np.zeros(AR_ORDER - len(z_hist)), z_hist]

    if arm == "global":
        assert global_state is not None
        pred = ar_predict_strict(z_online, global_state.phi, warm=warm)
        resid_z = (z_online - pred) / global_state.ref.sig
        return residual_features(resid_z, global_state.ref)
    if arm == "fixed":
        phi, ref = local_fixed_state(z_hist)
        pred = ar_predict_strict(z_online, phi, warm=warm)
        resid_z = (z_online - pred) / ref.sig
        return residual_features(resid_z, ref)
    if arm == "adapted":
        assert global_state is not None
        a, b, ref = local_affine_state(z_hist, global_state.phi)
        pred = ar_predict_strict(z_online, global_state.phi, warm=warm)
        resid_z = (z_online - (a * pred + b)) / ref.sig
        return residual_features(resid_z, ref)
    raise ValueError(arm)


def feature_cache_path(arm: str, fold: int):
    if arm == "fixed":
        return CACHE / "features_fixed.npy", CACHE / "features_fixed.json"
    return CACHE / f"features_{arm}_fold{fold}.npy", CACHE / f"features_{arm}_fold{fold}.json"


def build_features(arm: str, fold: int, d: PL.Data):
    path, meta_path = feature_cache_path(arm, fold)
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (
            meta.get("version") == FEATURE_VERSION
            and meta.get("arm") == arm
            and meta.get("n_rows") == len(d.y)
            and meta.get("feature_names") == list(FEATURE_NAMES)
            and (arm == "fixed" or meta.get("fold") == fold)
        ):
            return np.load(path, mmap_mode="r")

    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    global_state = fit_global_state(d, fold) if arm in {"global", "adapted"} else None
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(d.y), len(FEATURE_NAMES)))
    off = d.st.orow_off
    t0 = time.time()
    for s in range(d.st.n_series):
        h, o, _ = d.st.series(s)
        a = int(off[s])
        out[a: a + len(o)] = features_for_series(h, o, arm, global_state)
        if (s + 1) % 1000 == 0:
            print(f"features {arm} fold {fold}: {s + 1}/{d.st.n_series} {time.time() - t0:.0f}s", flush=True)
    out.flush()
    del out
    os.replace(tmp, path)
    meta_path.write_text(json.dumps({
        "version": FEATURE_VERSION,
        "arm": arm,
        "fold": None if arm == "fixed" else fold,
        "n_rows": len(d.y),
        "feature_names": list(FEATURE_NAMES),
        "built_s": round(time.time() - t0, 1),
    }, indent=2, sort_keys=True) + "\n")
    return np.load(path, mmap_mode="r")


def sample_train_rows(d: PL.Data, folds, max_rows: int, seed: int) -> np.ndarray:
    rows = d.rows_for(folds)
    if len(rows) <= max_rows:
        return rows
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(rows, max_rows, replace=False))


def train_arm(arm: str, d: PL.Data, c: Ctx) -> np.ndarray:
    exp_id = EXP_IDS[arm]
    out_path = OOF_DIR / f"{exp_id}.npy"
    if out_path.exists():
        arr = np.load(out_path)
        if np.isfinite(arr[c.dev]).all():
            print(f"{exp_id}: existing OOF, reuse", flush=True)
            return arr

    oof = np.full(len(d.y), np.nan, dtype=np.float32)
    for fold in FOLDS:
        feats = build_features(arm, fold, d)
        train_folds = [g for g in FOLDS if g != fold]
        tr_rows = sample_train_rows(d, train_folds, ARGS.max_train_rows, ARGS.seed + fold)
        va_rows = d.rows_for([fold])

        Xtr = np.asarray(feats[tr_rows], dtype=np.float32)
        ytr = d.y[tr_rows]
        ttr = d.t[tr_rows]
        params = dict(DEFAULT)
        params["objective"] = PL._make_pairwise_t(ttr, ytr, seed=ARGS.seed + 100 * fold)
        ds = lgb.Dataset(Xtr, label=ytr, params=dict(params, objective="binary"), feature_name=list(FEATURE_NAMES))
        booster = lgb.train(params, ds, num_boost_round=ARGS.rounds)
        del Xtr, ds

        Xva = np.asarray(feats[va_rows], dtype=np.float32)
        pred = booster.predict(Xva).astype(np.float32)
        del Xva
        oof[va_rows] = pred
        auc = float(ts_auc_flat(pred, d.y[va_rows], d.t[va_rows]))
        print(f"{exp_id} {arm} fold {fold}: {auc:.5f} rows={len(tr_rows)}", flush=True)

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
    if split == "mature_vs_never":
        return rows[(y == 0) | ((y == 1) & (age >= 100))]
    if split == "mature_vs_prebreak":
        hb = c.has_break[c.d.sidx[rows]]
        pre = hb & (y == 0)
        return rows[pre | ((y == 1) & (age >= 100))]
    raise ValueError(split)


def pair_repair_stats(base, cand, c: Ctx, rows, n_pairs_per_t: int, seed: int):
    rng = np.random.default_rng(seed)
    y = c.d.y
    t = c.d.t
    rr = rows[np.isfinite(base[rows]) & np.isfinite(cand[rows])]
    order = np.argsort(t[rr], kind="stable")
    sorted_t = t[rr][order]
    starts = np.flatnonzero(np.r_[True, sorted_t[1:] != sorted_t[:-1]])
    ends = np.r_[starts[1:], len(sorted_t)]
    repairs = damage = total = 0
    for lo, hi in zip(starts, ends):
        idx = rr[order[lo:hi]]
        pp = idx[y[idx] == 1]
        nn = idx[y[idx] == 0]
        if len(pp) == 0 or len(nn) == 0:
            continue
        k = min(n_pairs_per_t, len(pp), len(nn))
        psel = rng.choice(pp, k, replace=False)
        nsel = rng.choice(nn, k, replace=False)
        base_right = base[psel] > base[nsel]
        cand_right = cand[psel] > cand[nsel]
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


def write_summary_csv(result):
    path = REPORT_DIR / "la03_per_series_adaptation_summary.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["arm", "kind", "mean_ts_auc", "pooled_ts_auc", "per_fold_ts_auc", "delta"])
        for key in ("global", "fixed", "adapted"):
            s = result["standalone"][key]["scores"]
            delta = ""
            if key == "fixed":
                delta = result["isolation"]["adapted_minus_fixed"]
            elif key == "adapted":
                delta = result["isolation"]["adapted_minus_global"]
            w.writerow([
                EXP_IDS[key],
                f"standalone_{key}",
                s["mean_ts_auc"],
                s["pooled_ts_auc"],
                ";".join(f"{x:.9f}" for x in s["per_fold_ts_auc"]),
                delta,
            ])
        for key in ("e0_rt600", "e1_rt600_plus_global", "e1_fixed_diag", "e2_rt600_plus_adapted"):
            s = result["integration"][key]["scores"]
            delta = ""
            if key == "e1_rt600_plus_global":
                delta = result["integration"][key]["mean_delta_vs_e0"]
            elif key == "e1_fixed_diag":
                delta = result["integration"][key]["mean_delta_vs_e0"]
            elif key == "e2_rt600_plus_adapted":
                delta = result["primary"]["marginal_vs_clone"]
            w.writerow([
                key,
                "integrated",
                s["mean_ts_auc"],
                s["pooled_ts_auc"],
                ";".join(f"{x:.9f}" for x in s["per_fold_ts_auc"]),
                delta,
            ])
    return path


def append_result_rows(result):
    rows = []
    row_specs = (
        ("global", C0_ID, "control", "e1_rt600_plus_global"),
        ("fixed", C1_ID, "control", "e1_fixed_diag"),
        ("adapted", CAND_ID, result["verdict"], "e2_rt600_plus_adapted"),
    )
    for arm, exp_id, status, integ_key in row_specs:
        integ = result["integration"][integ_key]["scores"]
        stand = result["standalone"][arm]["scores"]
        notes = (
            f"LA-03 {arm}. standalone={stand['mean_ts_auc']:.9f}; "
            f"E0={result['integration']['e0_rt600']['scores']['mean_ts_auc']:.9f}; "
            f"E1_global={result['integration']['e1_rt600_plus_global']['scores']['mean_ts_auc']:.9f}; "
            f"E2_adapted={result['integration']['e2_rt600_plus_adapted']['scores']['mean_ts_auc']:.9f}; "
            f"marginal_vs_clone={result['primary']['marginal_vs_clone']:+.9f}; "
            f"metric_class={result['metric_class']}; final_verdict={result['verdict']}."
        )
        if arm == "adapted":
            notes += (
                f" isolation_vs_global={result['isolation']['adapted_minus_global']:+.9f}; "
                f"isolation_vs_fixed={result['isolation']['adapted_minus_fixed']:+.9f}; "
                f"dominant_net={result['pair_flow']['e2_vs_e0']['dominant_cell']['net_pair_lift']}; "
                f"mature_vs_never_net={result['pair_flow']['e2_vs_e0']['mature_vs_never']['net_pair_lift']}; "
                f"mature_vs_prebreak_net={result['pair_flow']['e2_vs_e0']['mature_vs_prebreak']['net_pair_lift']}."
            )
        rows.append({
            "experiment_id": exp_id,
            "date": "2026-08-26",
            "git_sha": result["git_sha"],
            "agent": "codex-leaderboard-alpha",
            "hypothesis": "LA-03 tests whether a tiny per-series history-fitted affine adapter on a global predictive null adds competition alpha beyond the no-adaptation clone and the fixed per-series null.",
            "falsification_condition": "KILL if standalone adapted head fails to beat both controls by +0.0005, or if integrated marginal_vs_clone < +0.0015, or if fewer than 4/5 folds are positive.",
            "feature_set": "RT600 seven specialists plus one LA03 10-feature predictive-null head",
            "n_features": "10_head_plus_rt600_scores_at_blend_only",
            "model": "single_lgbm_head_or_rt600_plus_head_equal_scdf_blend",
            "objective": "pairwise_t",
            "folds": "0,1,2,3,4",
            "random_seed": str(ARGS.seed),
            "train_series": "",
            "train_rows": str(ARGS.max_train_rows),
            "mean_oof_ts_auc": repr(integ["mean_ts_auc"]),
            "pooled_oof_ts_auc": repr(integ["pooled_ts_auc"]),
            "per_fold_ts_auc": ";".join(f"{x:.5f}" for x in integ["per_fold_ts_auc"]),
            "fold_std": repr(integ["fold_std"]),
            "persistence": "none",
            "sample_mode": "uniform_1m_rows_per_outer_fold",
            "training_runtime_s": repr(result["runtime_s"]),
            "causal_verified": "strict-past predictive residuals; per-series adapters fit on H_i only; no RT600 score input to head; real validation only",
            "test_reduced_touched": "no",
            "lockbox_touched": "no",
            "status": status,
            "notes": notes,
            "protocol": "leaderboard_alpha_2026_la03_full5_per_series_adaptation",
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
        raise SystemExit("LA-03 is preregistered for full folds 0,1,2,3,4 only")

    d = PL.Data()
    c = Ctx()
    arm_oof = {arm: train_arm(arm, d, c).astype(np.float64) for arm in ("global", "fixed", "adapted")}

    rt600 = load_rt600(c)
    P = load_oof(SPECIALISTS)
    P.update({EXP_IDS[k]: v for k, v in arm_oof.items()})
    e0 = rt600
    e1 = c.crossfit_blend(P, SPECIALISTS + [C0_ID])
    e_fixed = c.crossfit_blend(P, SPECIALISTS + [C1_ID])
    e2 = c.crossfit_blend(P, SPECIALISTS + [CAND_ID])

    standalone = {arm: {"exp_id": EXP_IDS[arm], "label": ARM_LABELS[arm], "scores": score_vec(c, v)}
                  for arm, v in arm_oof.items()}
    integration = {
        "e0_rt600": {"scores": score_vec(c, e0)},
        "e1_rt600_plus_global": {"exp_id": C0_ID, "scores": score_vec(c, e1)},
        "e1_fixed_diag": {"exp_id": C1_ID, "scores": score_vec(c, e_fixed)},
        "e2_rt600_plus_adapted": {"exp_id": CAND_ID, "scores": score_vec(c, e2)},
    }
    for key in ("e1_rt600_plus_global", "e1_fixed_diag", "e2_rt600_plus_adapted"):
        integration[key]["mean_delta_vs_e0"] = (
            integration[key]["scores"]["mean_ts_auc"] - integration["e0_rt600"]["scores"]["mean_ts_auc"]
        )

    iso_global = standalone["adapted"]["scores"]["mean_ts_auc"] - standalone["global"]["scores"]["mean_ts_auc"]
    iso_fixed = standalone["adapted"]["scores"]["mean_ts_auc"] - standalone["fixed"]["scores"]["mean_ts_auc"]
    fold_delta_vs_clone = [
        float(a - b) for a, b in zip(
            integration["e2_rt600_plus_adapted"]["scores"]["per_fold_ts_auc"],
            integration["e1_rt600_plus_global"]["scores"]["per_fold_ts_auc"],
        )
    ]
    marginal = (
        integration["e2_rt600_plus_adapted"]["scores"]["mean_ts_auc"]
        - integration["e1_rt600_plus_global"]["scores"]["mean_ts_auc"]
    )
    positive = int(sum(x > 0 for x in fold_delta_vs_clone))
    isolation = {
        "adapted_minus_global": float(iso_global),
        "adapted_minus_fixed": float(iso_fixed),
        "adapted_beats_global_ge_0_0005": bool(iso_global >= 0.0005),
        "adapted_beats_fixed_ge_0_0005": bool(iso_fixed >= 0.0005),
    }
    gate = {
        "isolation_vs_global": isolation["adapted_beats_global_ge_0_0005"],
        "isolation_vs_fixed": isolation["adapted_beats_fixed_ge_0_0005"],
        "marginal_vs_clone_ge_0_0015": bool(marginal >= 0.0015),
        "positive_folds_ge_4": bool(positive >= 4),
    }
    if marginal >= 0.0030 and positive >= 4:
        metric_class = "SERIOUS"
    elif marginal >= 0.0015 and positive >= 4:
        metric_class = "WEAK"
    else:
        metric_class = "KILL"
    verdict = metric_class if all(gate.values()) else "KILL"

    pair_flow = {
        split: pair_repair_stats(
            e0, e2, c, cell_rows(c, c.dev, split), ARGS.pairs_per_t, ARGS.pair_seed
        )
        for split in PAIRFLOW_SPLITS
    }
    corr = {
        "e1_global_vs_e0_dev": within_t_rank_corr(e1, e0, c.d.t, c.dev),
        "e1_global_vs_e0_dominant": within_t_rank_corr(e1, e0, c.d.t, cell_rows(c, c.dev, "dominant_cell")),
        "e2_adapted_vs_e0_dev": within_t_rank_corr(e2, e0, c.d.t, c.dev),
        "e2_adapted_vs_e0_dominant": within_t_rank_corr(e2, e0, c.d.t, cell_rows(c, c.dev, "dominant_cell")),
    }

    result = {
        "program": "LEADERBOARD ALPHA 2026",
        "experiment": "LA-03 per-series history adaptation",
        "git_sha": git_sha(),
        "artifact_root": ARGS.artifact_root,
        "cache_dir": str(CACHE),
        "seed": int(ARGS.seed),
        "max_train_rows": int(ARGS.max_train_rows),
        "rounds": int(ARGS.rounds),
        "pairs_per_t": int(ARGS.pairs_per_t),
        "pair_seed": int(ARGS.pair_seed),
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "adapter": {
            "ridge_lambda": RIDGE_LAMBDA,
            "clip": ADAPTER_CLIP,
            "form": "z_t = a_i * global_ar5_prediction_t + b_i + e_t",
        },
        "runtime_s": float(time.time() - t0),
        "standalone": standalone,
        "integration": integration,
        "isolation": isolation,
        "primary": {
            "marginal_vs_clone": float(marginal),
            "positive_folds_vs_clone": positive,
            "fold_deltas_vs_clone": fold_delta_vs_clone,
            "candidate_vs_e0": integration["e2_rt600_plus_adapted"]["mean_delta_vs_e0"],
            "clone_vs_e0": integration["e1_rt600_plus_global"]["mean_delta_vs_e0"],
            "fixed_diag_vs_e0": integration["e1_fixed_diag"]["mean_delta_vs_e0"],
        },
        "pair_flow": {"e2_vs_e0": pair_flow},
        "within_t_rank_corr": corr,
        "gate": gate,
        "metric_class": metric_class,
        "verdict": verdict,
        "oof_heads": {arm: f"{EXP_IDS[arm]}.npy" for arm in ("global", "fixed", "adapted")},
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "la03_per_series_adaptation.json"
    csv_path = write_summary_csv(result)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not ARGS.no_ledger:
        append_result_rows(result)
    print(json.dumps({
        "json": str(json_path),
        "csv": str(csv_path),
        "verdict": verdict,
        "metric_class": metric_class,
        "marginal_vs_clone": result["primary"]["marginal_vs_clone"],
        "positive_folds_vs_clone": positive,
        "isolation": isolation,
        "e0_mean_ts_auc": integration["e0_rt600"]["scores"]["mean_ts_auc"],
        "e1_global_mean_ts_auc": integration["e1_rt600_plus_global"]["scores"]["mean_ts_auc"],
        "e2_adapted_mean_ts_auc": integration["e2_rt600_plus_adapted"]["scores"]["mean_ts_auc"],
        "runtime_s": result["runtime_s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
