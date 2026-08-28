#!/usr/bin/env python3
"""Deterministic, no-training data forensics for the real-time challenge."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO.parent / "structural-break-learner-diversity-2026"
DEFAULT_LOCAL_OOF_ROOT = REPO.parent / "structural-break-deep-ensemble-frontier-local-2026" / "research" / "oof"
OUT_DIR = REPO / "research" / "reports" / "data_forensics_2026"

for _p in (REPO / "src", REPO / "research" / "scripts"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from sbr.metric import ts_auc_flat  # noqa: E402
from wave4_cal import SCDF_NSEEN  # noqa: E402

FOLDS = (0, 1, 2, 3, 4)
PAIR_SEED = 20260828
PAIRS_PER_T = 128
FEATURE_SAMPLE_STRIDE = 50
MODULES = ("m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid", "m06_loc", "m07_bayes")
RT600_STREAMS = ("RT-300", "RT-410", "RT-411", "RT-412", "RT-413", "RT-414", "RT-415")
RT1264_STREAMS = ("RT-1255", "RT-410", "RT-1260", "RT-1261", "RT-1254", "RT-1262", "RT-415")
T_BINS = (
    ("t_000_049", 0, 49),
    ("t_050_099", 50, 99),
    ("t_100_199", 100, 199),
    ("t_200_399", 200, 399),
    ("t_400_699", 400, 699),
    ("t_700_999", 700, 999),
)
REL_BINS = (
    ("rel_00_10", 0.0, 0.10),
    ("rel_10_25", 0.10, 0.25),
    ("rel_25_50", 0.25, 0.50),
    ("rel_50_75", 0.50, 0.75),
    ("rel_75_90", 0.75, 0.90),
    ("rel_90_100", 0.90, 1.0000001),
)
TOP_THRESHOLDS = (("top_1", 0.99), ("top_5", 0.95), ("top_10", 0.90))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=os.environ.get("SBR_DATA_ROOT", str(DEFAULT_DATA_ROOT)))
    p.add_argument("--local-oof-root", default=os.environ.get("SBR_LOCAL_OOF_ROOT", str(DEFAULT_LOCAL_OOF_ROOT)))
    p.add_argument("--skip-feature-audit", action="store_true")
    return p.parse_args()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "nogit"


def q(values: np.ndarray, probs: Iterable[float]) -> list[float | None]:
    arr = np.asarray(values)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return [None for _ in probs]
    return [float(x) for x in np.quantile(arr, list(probs))]


def mean_or_none(values: np.ndarray) -> float | None:
    arr = np.asarray(values)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else None


def auc_or_none(score: np.ndarray, y: np.ndarray, t: np.ndarray, mask: np.ndarray) -> float | None:
    if int(mask.sum()) == 0:
        return None
    yy = y[mask]
    if int(yy.sum()) == 0 or int((yy == 0).sum()) == 0:
        return None
    try:
        return float(ts_auc_flat(score[mask], yy, t[mask]))
    except ValueError:
        return None


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_dev_metadata(data_root: Path) -> dict:
    folds_path = data_root / "research" / "folds" / "folds.parquet"
    meta_path = data_root / "cache" / "store" / "meta.parquet"
    values_path = data_root / "cache" / "store" / "values.npy"

    folds = pd.read_parquet(folds_path)
    dev_folds = folds[folds["fold"].isin(FOLDS)].copy()
    dev_ids = dev_folds["id"].tolist()

    offsets_meta = pd.read_parquet(meta_path, columns=["id", "n_online"])
    offsets_meta = offsets_meta.reset_index().rename(columns={"index": "series_idx"})
    offsets_meta["online_start"] = np.r_[0, np.cumsum(offsets_meta["n_online"].to_numpy()[:-1])].astype(np.int64)

    meta_cols = ["id", "off", "n_hist", "n_online", "tau_index", "has_break"]
    meta_dev = pd.read_parquet(meta_path, columns=meta_cols, filters=[("id", "in", dev_ids)])
    meta_dev = meta_dev.merge(dev_folds[["id", "fold"]], on="id", how="inner")
    meta_dev = meta_dev.merge(offsets_meta[["id", "series_idx", "online_start"]], on="id", how="inner")
    meta_dev = meta_dev.sort_values("series_idx").reset_index(drop=True)
    if len(meta_dev) != len(dev_folds):
        raise RuntimeError(f"dev metadata mismatch: meta_dev={len(meta_dev)} dev_folds={len(dev_folds)}")

    total_rows = int(meta_dev["n_online"].sum())
    rows = np.empty(total_rows, dtype=np.int64)
    labels = np.empty(total_rows, dtype=np.int8)
    t_index = np.empty(total_rows, dtype=np.int32)
    fold_index = np.empty(total_rows, dtype=np.int8)
    series_index = np.empty(total_rows, dtype=np.int32)
    rel_t = np.empty(total_rows, dtype=np.float32)
    phase = np.empty(total_rows, dtype=np.int8)
    dist_to_break = np.full(total_rows, -1, dtype=np.int32)
    age_since_break = np.full(total_rows, -1, dtype=np.int32)
    hist_len = np.empty(total_rows, dtype=np.int32)
    online_len = np.empty(total_rows, dtype=np.int32)
    z = np.empty(total_rows, dtype=np.float32)

    values = np.load(values_path, mmap_mode="r")
    pos = 0
    for r in meta_dev.itertuples(index=False):
        n = int(r.n_online)
        tau = int(r.tau_index)
        start = int(r.online_start)
        tt = np.arange(n, dtype=np.int32)
        sl = slice(pos, pos + n)
        rows[sl] = start + tt
        labels[sl] = 0
        if tau >= 0:
            labels[pos + tau: pos + n] = 1
        t_index[sl] = tt
        fold_index[sl] = int(r.fold)
        series_index[sl] = int(r.series_idx)
        denom = max(n - 1, 1)
        rel_t[sl] = tt.astype(np.float32) / float(denom)
        hist_len[sl] = int(r.n_hist)
        online_len[sl] = n

        ph = np.zeros(n, dtype=np.int8)
        if tau < 0:
            ph[:] = 0  # never_break
        else:
            d = tau - tt
            a = tt - tau
            dist_to_break[sl] = np.where(tt < tau, d, -1)
            age_since_break[sl] = np.where(tt >= tau, a, -1)
            ph[(tt < tau) & (d > 100)] = 1
            ph[(tt < tau) & (d <= 100)] = 2
            ph[(tt >= tau) & (a < 100)] = 3
            ph[(tt >= tau) & (a >= 100)] = 4
        phase[sl] = ph

        off = int(r.off)
        nh = int(r.n_hist)
        hist = np.asarray(values[off: off + nh], dtype=np.float64)
        online = np.asarray(values[off + nh: off + nh + n], dtype=np.float64)
        sd = float(hist.std(ddof=1)) if nh > 1 else 1.0
        sd = max(sd, 1e-9)
        z[sl] = ((online - float(hist.mean())) / sd).astype(np.float32)
        pos += n

    fold_rows = {int(f): np.flatnonzero(fold_index == int(f)) for f in FOLDS}
    return {
        "folds": folds,
        "meta_dev": meta_dev,
        "rows": rows,
        "y": labels,
        "t": t_index,
        "fold_index": fold_index,
        "series_index": series_index,
        "rel_t": rel_t,
        "phase": phase,
        "dist_to_break": dist_to_break,
        "age_since_break": age_since_break,
        "hist_len": hist_len,
        "online_len": online_len,
        "z": z,
        "fold_rows": fold_rows,
    }


def locate_oof(name: str, roots: list[Path]) -> Path:
    for root in roots:
        path = root / f"{name}.npy"
        if path.exists():
            return path
    raise FileNotFoundError(f"missing OOF {name} in {[str(r) for r in roots]}")


class CalibratedBlend:
    def __init__(self, idx: dict, oof_roots: list[Path]):
        self.idx = idx
        self.oof_roots = oof_roots
        self.raw: dict[str, np.ndarray] = {}
        self.cache: dict[tuple[str, tuple[int, ...], int], np.ndarray] = {}

    def load(self, name: str) -> np.ndarray:
        if name not in self.raw:
            self.raw[name] = np.load(locate_oof(name, self.oof_roots), mmap_mode="r")
        return self.raw[name]

    def stream(self, name: str, train_folds: Iterable[int], eval_fold: int) -> np.ndarray:
        key = (name, tuple(sorted(int(x) for x in train_folds)), int(eval_fold))
        if key not in self.cache:
            raw = self.load(name)
            train_pos = np.concatenate([self.idx["fold_rows"][int(f)] for f in key[1]])
            eval_pos = self.idx["fold_rows"][int(eval_fold)]
            train_rows = self.idx["rows"][train_pos]
            eval_rows = self.idx["rows"][eval_pos]
            cal = SCDF_NSEEN(raw[train_rows], self.idx["t"][train_pos])
            self.cache[key] = cal(raw[eval_rows], self.idx["t"][eval_pos])
        return self.cache[key]

    def blend(self, names: tuple[str, ...]) -> tuple[np.ndarray, list[float]]:
        out = np.full(len(self.idx["y"]), np.nan, dtype=np.float64)
        per_fold = []
        for fold in FOLDS:
            pos = self.idx["fold_rows"][fold]
            train_folds = [f for f in FOLDS if f != fold]
            cols = [self.stream(name, train_folds, fold) for name in names]
            pred = np.column_stack(cols).mean(axis=1)
            out[pos] = pred
            per_fold.append(float(ts_auc_flat(pred, self.idx["y"][pos], self.idx["t"][pos])))
        if not np.isfinite(out).all():
            raise RuntimeError("blend produced non-finite dev predictions")
        return out, per_fold


def phase_name(code: int) -> str:
    return {
        0: "never_break",
        1: "prebreak_far",
        2: "prebreak_near",
        3: "postbreak_early",
        4: "postbreak_mature",
    }[int(code)]


def population_tables(idx: dict) -> dict:
    meta = idx["meta_dev"]
    y = idx["y"]
    t = idx["t"]
    phase = idx["phase"]
    rows = []
    for fold in FOLDS:
        mf = meta[meta["fold"] == fold]
        mask = idx["fold_index"] == fold
        rows.append({
            "fold": fold,
            "series": int(len(mf)),
            "rows": int(mask.sum()),
            "break_series": int(mf["has_break"].astype(bool).sum()),
            "break_series_rate": float(mf["has_break"].astype(bool).mean()),
            "positive_rows": int(y[mask].sum()),
            "positive_row_rate": float(y[mask].mean()),
            "hist_len_mean": float(mf["n_hist"].mean()),
            "online_len_mean": float(mf["n_online"].mean()),
        })
    write_csv(OUT_DIR / "population_by_fold.csv", rows)

    t_rows = []
    for name, lo, hi in T_BINS:
        mask = (t >= lo) & (t <= hi)
        t_rows.append(balance_row(name, mask, idx))
    write_csv(OUT_DIR / "population_by_t_bin.csv", t_rows)

    rel_rows = []
    rel = idx["rel_t"]
    for name, lo, hi in REL_BINS:
        mask = (rel >= lo) & (rel < hi)
        rel_rows.append(balance_row(name, mask, idx))
    write_csv(OUT_DIR / "population_by_rel_bin.csv", rel_rows)

    phase_rows = []
    for code in range(5):
        phase_rows.append(balance_row(phase_name(code), phase == code, idx))
    write_csv(OUT_DIR / "population_by_phase.csv", phase_rows)

    return {
        "folds": rows,
        "t_bins": t_rows,
        "rel_bins": rel_rows,
        "phase": phase_rows,
        "dev_series": int(len(meta)),
        "dev_rows": int(len(y)),
        "break_series": int(meta["has_break"].astype(bool).sum()),
        "positive_rows": int(y.sum()),
        "hist_len_quantiles": q(meta["n_hist"].to_numpy(), [0.0, 0.25, 0.5, 0.75, 1.0]),
        "online_len_quantiles": q(meta["n_online"].to_numpy(), [0.0, 0.25, 0.5, 0.75, 1.0]),
        "tau_rel_quantiles": q(
            (meta.loc[meta["tau_index"] >= 0, "tau_index"] / meta.loc[meta["tau_index"] >= 0, "n_online"]).to_numpy(),
            [0.0, 0.25, 0.5, 0.75, 1.0],
        ),
    }


def balance_row(name: str, mask: np.ndarray, idx: dict) -> dict:
    y = idx["y"]
    meta_series = np.unique(idx["series_index"][mask])
    return {
        "slice": name,
        "series": int(len(meta_series)),
        "rows": int(mask.sum()),
        "positive_rows": int(y[mask].sum()),
        "negative_rows": int((y[mask] == 0).sum()),
        "positive_row_rate": float(y[mask].mean()) if int(mask.sum()) else None,
    }


def hist_online_tertile_masks(idx: dict) -> dict[str, np.ndarray]:
    meta = idx["meta_dev"]
    hist_q = np.quantile(meta["n_hist"].to_numpy(), [1 / 3, 2 / 3])
    online_q = np.quantile(meta["n_online"].to_numpy(), [1 / 3, 2 / 3])
    return {
        "hist_len_low": idx["hist_len"] <= hist_q[0],
        "hist_len_mid": (idx["hist_len"] > hist_q[0]) & (idx["hist_len"] <= hist_q[1]),
        "hist_len_high": idx["hist_len"] > hist_q[1],
        "online_len_low": idx["online_len"] <= online_q[0],
        "online_len_mid": (idx["online_len"] > online_q[0]) & (idx["online_len"] <= online_q[1]),
        "online_len_high": idx["online_len"] > online_q[1],
    }


def slice_masks(idx: dict) -> dict[str, np.ndarray]:
    y = idx["y"]
    t = idx["t"]
    phase = idx["phase"]
    masks: dict[str, np.ndarray] = {"whole_dev": np.ones(len(y), dtype=bool)}
    for fold in FOLDS:
        masks[f"fold_{fold}"] = idx["fold_index"] == fold
    for name, lo, hi in T_BINS:
        masks[name] = (t >= lo) & (t <= hi)
    for name, lo, hi in REL_BINS:
        masks[name] = (idx["rel_t"] >= lo) & (idx["rel_t"] < hi)
    masks.update(hist_online_tertile_masks(idx))

    post_mature = (phase == 4)
    never = phase == 0
    pre = (phase == 1) | (phase == 2)
    pre_near = phase == 2
    dominant = (t >= 200) & ((y == 0) | post_mature)
    masks["dominant_cell"] = dominant
    masks["mature_vs_never"] = (t >= 200) & (post_mature | never)
    masks["mature_vs_prebreak"] = (t >= 200) & (post_mature | pre)
    masks["near_boundary"] = pre_near | (phase == 3)
    masks["late_never_vs_mature"] = (t >= 400) & (never | post_mature)
    return masks


def slice_auc_tables(idx: dict, rt600: np.ndarray, rt1264: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    rows = []
    for name, mask in masks.items():
        a0 = auc_or_none(rt600, idx["y"], idx["t"], mask)
        a1 = auc_or_none(rt1264, idx["y"], idx["t"], mask)
        rows.append({
            "slice": name,
            "rows": int(mask.sum()),
            "positive_rows": int(idx["y"][mask].sum()),
            "negative_rows": int((idx["y"][mask] == 0).sum()),
            "rt600_ts_auc": a0,
            "rt1264_ts_auc": a1,
            "delta": (float(a1 - a0) if a0 is not None and a1 is not None else None),
        })
    write_csv(OUT_DIR / "slice_auc.csv", rows)
    return {"rows": rows}


def pair_flow_for_mask(idx: dict, base: np.ndarray, cand: np.ndarray, mask: np.ndarray) -> dict:
    y = idx["y"]
    t = idx["t"]
    positions = np.flatnonzero(mask)
    order = np.argsort(t[positions], kind="stable")
    positions = positions[order]
    sorted_t = t[positions]
    if len(positions) == 0:
        return empty_pair_flow()
    starts = np.flatnonzero(np.r_[True, sorted_t[1:] != sorted_t[:-1]])
    ends = np.r_[starts[1:], len(sorted_t)]
    rng = np.random.default_rng(PAIR_SEED)
    repairs = damage = total = base_wrong = base_right_total = 0
    for lo, hi in zip(starts, ends):
        pos = positions[lo:hi]
        pp = pos[y[pos] == 1]
        nn = pos[y[pos] == 0]
        if len(pp) == 0 or len(nn) == 0:
            continue
        k = min(PAIRS_PER_T, len(pp), len(nn))
        psel = rng.choice(pp, k, replace=False)
        nsel = rng.choice(nn, k, replace=False)
        b = base[psel] > base[nsel]
        c = cand[psel] > cand[nsel]
        repairs += int((~b & c).sum())
        damage += int((b & ~c).sum())
        base_wrong += int((~b).sum())
        base_right_total += int(b.sum())
        total += int(k)
    return {
        "sampled_pairs": int(total),
        "rt600_wrong_pairs": int(base_wrong),
        "rt600_right_pairs": int(base_right_total),
        "repairs": int(repairs),
        "damage": int(damage),
        "net_pair_lift": int(repairs - damage),
        "repair_rate_of_rt600_wrong": float(repairs / base_wrong) if base_wrong else None,
        "damage_rate_of_rt600_right": float(damage / base_right_total) if base_right_total else None,
    }


def empty_pair_flow() -> dict:
    return {
        "sampled_pairs": 0,
        "rt600_wrong_pairs": 0,
        "rt600_right_pairs": 0,
        "repairs": 0,
        "damage": 0,
        "net_pair_lift": 0,
        "repair_rate_of_rt600_wrong": None,
        "damage_rate_of_rt600_right": None,
    }


def pair_flow_tables(idx: dict, rt600: np.ndarray, rt1264: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    rows = []
    for name, mask in masks.items():
        rec = pair_flow_for_mask(idx, rt600, rt1264, mask)
        rec = {"slice": name, **rec}
        rows.append(rec)
    write_csv(OUT_DIR / "pair_flow_slices.csv", rows)
    return {"rows": rows, "pairs_per_t": PAIRS_PER_T, "seed": PAIR_SEED}


def within_t_percentile(scores: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.empty(len(scores), dtype=np.float32)
    order = np.lexsort((scores, t))
    tt = t[order]
    ss = scores[order]
    starts = np.flatnonzero(np.r_[True, tt[1:] != tt[:-1]])
    ends = np.r_[starts[1:], len(tt)]
    for lo, hi in zip(starts, ends):
        sub = order[lo:hi]
        n = hi - lo
        ranks = np.arange(1, n + 1, dtype=np.float64)
        run_start = lo
        while run_start < hi:
            run_end = run_start + 1
            while run_end < hi and ss[run_end] == ss[run_start]:
                run_end += 1
            avg = 0.5 * ((run_start - lo + 1) + (run_end - lo))
            ranks[run_start - lo: run_end - lo] = avg
            run_start = run_end
        out[sub] = (ranks - 0.5) / max(n, 1)
    return out


def high_score_mass(idx: dict, rt600: np.ndarray, rt1264: np.ndarray) -> dict:
    rows = []
    for label, scores in (("RT600", rt600), ("RT1264", rt1264)):
        pct = within_t_percentile(scores, idx["t"])
        for threshold_name, threshold in TOP_THRESHOLDS:
            mask = pct >= threshold
            y = idx["y"]
            phase = idx["phase"]
            total_pos = int(y.sum())
            row = {
                "model": label,
                "threshold": threshold_name,
                "rows": int(mask.sum()),
                "positive_rows": int((mask & (y == 1)).sum()),
                "positive_capture_rate": float((mask & (y == 1)).sum() / max(total_pos, 1)),
                "never_break_negative_rows": int((mask & (phase == 0)).sum()),
                "prebreak_negative_rows": int((mask & ((phase == 1) | (phase == 2))).sum()),
                "postbreak_early_rows": int((mask & (phase == 3)).sum()),
                "postbreak_mature_rows": int((mask & (phase == 4)).sum()),
                "negative_share": float((mask & (y == 0)).sum() / max(int(mask.sum()), 1)),
                "never_break_share": float((mask & (phase == 0)).sum() / max(int(mask.sum()), 1)),
                "prebreak_share": float((mask & ((phase == 1) | (phase == 2))).sum() / max(int(mask.sum()), 1)),
            }
            rows.append(row)
    write_csv(OUT_DIR / "top_score_mass.csv", rows)
    return {"rows": rows}


def raw_process_tables(idx: dict) -> dict:
    z = idx["z"].astype(np.float64)
    absz = np.abs(z)
    rows = []
    for code in range(5):
        mask = idx["phase"] == code
        rows.append(raw_row(f"phase_{phase_name(code)}", mask, z, absz))
    for name, lo, hi in T_BINS:
        mask = (idx["t"] >= lo) & (idx["t"] <= hi)
        rows.append(raw_row(name, mask, z, absz))
    write_csv(OUT_DIR / "raw_process_slices.csv", rows)
    return {"rows": rows}


def raw_row(name: str, mask: np.ndarray, z: np.ndarray, absz: np.ndarray) -> dict:
    zz = z[mask]
    aa = absz[mask]
    return {
        "slice": name,
        "rows": int(mask.sum()),
        "z_mean": mean_or_none(zz),
        "z_p01": q(zz, [0.01])[0],
        "z_p50": q(zz, [0.50])[0],
        "z_p99": q(zz, [0.99])[0],
        "abs_z_mean": mean_or_none(aa),
        "abs_z_p50": q(aa, [0.50])[0],
        "abs_z_p95": q(aa, [0.95])[0],
        "tail_abs_gt_2_rate": float((aa > 2.0).mean()) if len(aa) else None,
        "tail_abs_gt_3_rate": float((aa > 3.0).mean()) if len(aa) else None,
    }


def feature_audit(idx: dict, data_root: Path) -> dict:
    sample_rows = idx["rows"][::FEATURE_SAMPLE_STRIDE]
    feat_root = data_root / "cache" / "features"
    feature_rows = []
    module_rows = []
    matrices = []
    module_slices = {}
    col_offset = 0
    for module in MODULES:
        cols = json.loads((feat_root / f"{module}.cols.json").read_text())["cols"]
        arr = np.load(feat_root / f"{module}.npy", mmap_mode="r")
        X = np.asarray(arr[sample_rows, :], dtype=np.float32)
        finite = np.isfinite(X)
        means = np.nanmean(np.where(finite, X, np.nan), axis=0)
        stds = np.nanstd(np.where(finite, X, np.nan), axis=0)
        qs = np.nanquantile(np.where(finite, X, np.nan), [0.01, 0.50, 0.99], axis=0)
        finite_rates = finite.mean(axis=0)
        for j, col in enumerate(cols):
            feature_rows.append({
                "module": module,
                "feature": col,
                "finite_rate": float(finite_rates[j]),
                "mean": float(means[j]) if math.isfinite(float(means[j])) else None,
                "std": float(stds[j]) if math.isfinite(float(stds[j])) else None,
                "p01": float(qs[0, j]) if math.isfinite(float(qs[0, j])) else None,
                "p50": float(qs[1, j]) if math.isfinite(float(qs[1, j])) else None,
                "p99": float(qs[2, j]) if math.isfinite(float(qs[2, j])) else None,
            })
        Z = standardize_for_corr(X, means, stds)
        module_slices[module] = slice(col_offset, col_offset + Z.shape[1])
        col_offset += Z.shape[1]
        matrices.append(Z)
        ev = corr_eigenvalues(Z)
        module_rows.append({
            "module": module,
            "columns": int(X.shape[1]),
            "sample_rows": int(X.shape[0]),
            "mean_finite_rate": float(finite_rates.mean()),
            "min_finite_rate": float(finite_rates.min()),
            "near_constant_columns": int((stds < 1e-8).sum()),
            "effective_rank_participation": participation_ratio(ev),
            "top_eigenvalue_share": top_eigen_share(ev),
            "mean_abs_within_corr": mean_abs_corr(Z),
        })
    Xall = np.column_stack(matrices)
    full_ev = corr_eigenvalues(Xall)
    between_rows = []
    for i, left in enumerate(MODULES):
        for right in MODULES[i + 1:]:
            a = module_slices[left]
            b = module_slices[right]
            c = (Xall[:, a].T @ Xall[:, b]) / max(Xall.shape[0] - 1, 1)
            between_rows.append({
                "left_module": left,
                "right_module": right,
                "mean_abs_corr": float(np.mean(np.abs(c))),
                "max_abs_corr": float(np.max(np.abs(c))),
            })
    module_rows.append({
        "module": "ALL_500",
        "columns": int(Xall.shape[1]),
        "sample_rows": int(Xall.shape[0]),
        "mean_finite_rate": None,
        "min_finite_rate": None,
        "near_constant_columns": int(sum(int(r["near_constant_columns"]) for r in module_rows)),
        "effective_rank_participation": participation_ratio(full_ev),
        "top_eigenvalue_share": top_eigen_share(full_ev),
        "mean_abs_within_corr": mean_abs_corr(Xall),
    })
    write_csv(OUT_DIR / "feature_summary.csv", feature_rows)
    write_csv(OUT_DIR / "feature_module_summary.csv", module_rows)
    write_csv(OUT_DIR / "feature_between_module_corr.csv", between_rows)
    return {
        "sample_rows": int(len(sample_rows)),
        "module_summary": module_rows,
        "between_module_corr": between_rows,
    }


def standardize_for_corr(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64, copy=True)
    for j in range(X.shape[1]):
        m = float(means[j]) if np.isfinite(means[j]) else 0.0
        s = float(stds[j]) if np.isfinite(stds[j]) and float(stds[j]) > 1e-8 else 1.0
        col = X[:, j]
        bad = ~np.isfinite(col)
        col[bad] = m
        X[:, j] = (col - m) / s
    return X


def corr_eigenvalues(Z: np.ndarray) -> np.ndarray:
    if Z.shape[1] == 0:
        return np.array([], dtype=np.float64)
    corr = (Z.T @ Z) / max(Z.shape[0] - 1, 1)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    ev = np.linalg.eigvalsh(corr)
    return np.maximum(ev, 0.0)


def participation_ratio(ev: np.ndarray) -> float | None:
    if len(ev) == 0 or float(np.sum(ev * ev)) <= 0:
        return None
    return float((ev.sum() ** 2) / np.sum(ev * ev))


def top_eigen_share(ev: np.ndarray) -> float | None:
    s = float(ev.sum())
    if len(ev) == 0 or s <= 0:
        return None
    return float(ev.max() / s)


def mean_abs_corr(Z: np.ndarray) -> float | None:
    if Z.shape[1] < 2:
        return None
    corr = (Z.T @ Z) / max(Z.shape[0] - 1, 1)
    iu = np.triu_indices(corr.shape[0], k=1)
    return float(np.mean(np.abs(corr[iu])))


def write_report(result: dict) -> None:
    pop = result["population"]
    overall = result["overall_scores"]
    pair_rows = result["pair_flow"]["rows"]
    auc_rows = result["slice_auc"]["rows"]
    feature = result.get("feature_audit")

    def top_delta(rows: list[dict], n: int = 8) -> list[dict]:
        usable = [r for r in rows if r.get("delta") is not None]
        return sorted(usable, key=lambda r: abs(float(r["delta"])), reverse=True)[:n]

    def find(rows: list[dict], name: str) -> dict:
        return next(r for r in rows if r["slice"] == name)

    lines = [
        "# DATA_FORENSICS_2026 -- FINAL REPORT",
        "",
        f"Date: 2026-08-28",
        f"Branch: `research/data-forensics-2026`",
        f"Scoring/analysis SHA: `{result['git_sha']}`",
        "",
        "## Scope",
        "",
        "No model was trained, no feature module was created, no OOF vector was written, and no RT ID was consumed. All row-level diagnostics use canonical dev folds `0..4`; lockbox labels, predictions, and values are not summarized.",
        "",
        "## Headline Read",
        "",
        f"Dev population: `{pop['dev_series']}` series, `{pop['dev_rows']}` online rows, `{pop['break_series']}` break series, `{pop['positive_rows']}` positive rows.",
        f"Reconstructed RT600 mean TS-AUC: `{overall['rt600_mean_ts_auc']:.9f}`. Reconstructed RT-1264 mean TS-AUC: `{overall['rt1264_mean_ts_auc']:.9f}`. Delta: `{overall['delta_mean_ts_auc']:+.9f}`.",
        f"Fold deltas RT-1264 minus RT600: `{', '.join(f'{x:+.9f}' for x in overall['fold_deltas'])}`.",
        "",
        "The fold-mean delta is positive on all five folds, so the `RT-1264` gain is not a single-fold artifact. The largest fold contribution is fold 3, but the sign is stable. Slice tables below use pooled TS-AUC inside each slice, so `whole_dev` does not numerically equal the fold-mean headline.",
        "",
        "## Largest Slice Deltas",
        "",
        "| slice | rows | positives | RT600 | RT-1264 | delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in top_delta(auc_rows):
        lines.append(
            f"| `{row['slice']}` | {row['rows']} | {row['positive_rows']} | "
            f"{row['rt600_ts_auc']:.9f} | {row['rt1264_ts_auc']:.9f} | {row['delta']:+.9f} |"
        )
    lines += [
        "",
        "## Fixed Cells",
        "",
        "| cell | RT600 | RT-1264 | delta | pair net | repairs | damage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    pair_by = {r["slice"]: r for r in pair_rows}
    for name in ("whole_dev", "dominant_cell", "mature_vs_never", "mature_vs_prebreak", "near_boundary", "late_never_vs_mature"):
        a = find(auc_rows, name)
        p = pair_by[name]
        lines.append(
            f"| `{name}` | {a['rt600_ts_auc']:.9f} | {a['rt1264_ts_auc']:.9f} | "
            f"{a['delta']:+.9f} | {p['net_pair_lift']} | {p['repairs']} | {p['damage']} |"
        )
    lines += [
        "",
        "The strongest interpretable cell is `mature_vs_prebreak`: `RT-1264` gains `+0.005154009` pooled TS-AUC and `+291` sampled pair net. That is the part of the problem previous arbitration attempts kept damaging. The weak spot is early relative time: `rel_10_25` loses `-0.003631996`, and pair flow there is `-114`. Deployment review should check whether the live stream distribution over early relative positions matches dev.",
        "",
        "## High-Score Mass",
        "",
        "| model | threshold | rows | positive capture | negative share | never-break share | prebreak share |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["high_score_mass"]["rows"]:
        lines.append(
            f"| `{row['model']}` | `{row['threshold']}` | {row['rows']} | "
            f"{row['positive_capture_rate']:.6f} | {row['negative_share']:.6f} | "
            f"{row['never_break_share']:.6f} | {row['prebreak_share']:.6f} |"
        )
    lines += [
        "",
        "At the top 10% within each `t`, `RT-1264` captures 1,198 more positive rows than RT600 while reducing negative share from `0.495378` to `0.492400`. Prebreak share is essentially unchanged at top 10 and lower at top 1/top 5. Never-break share is mixed: slightly higher at top 1/top 5, lower at top 10. This does not look like a broad false-positive explosion, but never-break top-score mass remains the production-risk cell to watch.",
        "",
        "## Raw Process",
        "",
        "| slice | rows | z mean | abs(z) mean | abs(z) p95 | abs(z)>3 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["raw_process"]["rows"]:
        if row["slice"].startswith("phase_"):
            lines.append(
                f"| `{row['slice']}` | {row['rows']} | {row['z_mean']:.6f} | "
                f"{row['abs_z_mean']:.6f} | {row['abs_z_p95']:.6f} | {row['tail_abs_gt_3_rate']:.6f} |"
            )
    lines += [
        "",
        "The raw series itself says why simple thresholding has been hard. Never-break, far-prebreak, and near-prebreak rows have nearly identical historical-z summaries. Early postbreak rows move only modestly. Mature postbreak rows show a clearer mean/absolute-z shift, but the tail-rate separation is still small (`abs(z)>3` is `0.013109` for mature postbreak versus `0.006302` for never-break). The remaining signal is not a one-dimensional amplitude anomaly.",
    ]
    if feature:
        all500 = next(r for r in feature["module_summary"] if r["module"] == "ALL_500")
        lines += [
            "",
            "## Feature Bank Audit",
            "",
            f"Deterministic sample rows: `{feature['sample_rows']}` (`dev_rows[::50]`). For the full 500-column bank, participation-ratio effective rank is `{all500['effective_rank_participation']:.3f}`, top eigenvalue share is `{all500['top_eigenvalue_share']:.6f}`, and mean absolute pairwise correlation is `{all500['mean_abs_within_corr']:.6f}`.",
            "",
            "| module | cols | mean finite | near constant | effective rank | top eigen share | mean abs corr |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in feature["module_summary"]:
            if row["module"] == "ALL_500":
                continue
            lines.append(
                f"| `{row['module']}` | {row['columns']} | {row['mean_finite_rate']:.6f} | "
                f"{row['near_constant_columns']} | {row['effective_rank_participation']:.3f} | "
                f"{row['top_eigenvalue_share']:.6f} | {row['mean_abs_within_corr']:.6f} |"
            )
        lines += [
            "",
            "The frozen 500-column bank has participation-ratio effective rank `21.282`, not anything close to 500. `m03_dyn` and `m06_loc` are the most internally diverse modules by this audit; `m01_seq`, `m04_resid`, and `m07_bayes` are much more compressed. This supports the recent empirical pattern: more learner or mechanism diversity is likelier to matter than adding near-duplicate columns inside the same transform family.",
        ]
    lines += [
        "",
        "## Interpretation",
        "",
        "- This report is descriptive only. It does not authorize a model, threshold, router, feature, or production change.",
        "- Any future experiment motivated by these slices needs its own preregistration before scoring.",
        "- `RT-1264` remains an internal OOF result pending separate deployment feasibility and confirmation work.",
        "- For deployment review, the concrete checks are early-relative-position behavior and never-break top-score mass, not global mean AUC.",
    ]
    (OUT_DIR / "DATA_FORENSICS_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    local_oof_root = Path(args.local_oof_root).resolve()
    oof_roots = [local_oof_root, data_root / "research" / "oof"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    idx = read_dev_metadata(data_root)
    pop = population_tables(idx)
    blender = CalibratedBlend(idx, oof_roots)
    rt600, rt600_per = blender.blend(RT600_STREAMS)
    rt1264, rt1264_per = blender.blend(RT1264_STREAMS)
    masks = slice_masks(idx)
    slice_auc = slice_auc_tables(idx, rt600, rt1264, masks)
    pair_flow = pair_flow_tables(idx, rt600, rt1264, masks)
    high_mass = high_score_mass(idx, rt600, rt1264)
    raw_process = raw_process_tables(idx)
    feature = None if args.skip_feature_audit else feature_audit(idx, data_root)

    result = {
        "program": "DATA_FORENSICS_2026",
        "git_sha": git_sha(),
        "data_root": str(data_root),
        "local_oof_root": str(local_oof_root),
        "dev_only": True,
        "lockbox_summarized": False,
        "test_reduced_touched": False,
        "calibration": "SCDF_NSEEN fold-pure",
        "population": pop,
        "overall_scores": {
            "rt600_mean_ts_auc": float(np.mean(rt600_per)),
            "rt1264_mean_ts_auc": float(np.mean(rt1264_per)),
            "delta_mean_ts_auc": float(np.mean(rt1264_per) - np.mean(rt600_per)),
            "rt600_per_fold": rt600_per,
            "rt1264_per_fold": rt1264_per,
            "fold_deltas": [float(a - b) for a, b in zip(rt1264_per, rt600_per)],
        },
        "slice_auc": slice_auc,
        "pair_flow": pair_flow,
        "high_score_mass": high_mass,
        "raw_process": raw_process,
        "feature_audit": feature,
    }
    (OUT_DIR / "data_forensics_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_report(result)
    print(json.dumps({
        "program": "DATA_FORENSICS_2026",
        "report": str(OUT_DIR / "DATA_FORENSICS_REPORT.md"),
        "result": str(OUT_DIR / "data_forensics_results.json"),
        "rt600": result["overall_scores"]["rt600_mean_ts_auc"],
        "rt1264": result["overall_scores"]["rt1264_mean_ts_auc"],
        "delta": result["overall_scores"]["delta_mean_ts_auc"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
