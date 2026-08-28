#!/usr/bin/env python3
"""Dev-only raw time-series atlas for structural-break data."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "research" / "reports" / "raw_data_atlas_2026"
FIG_DIR = OUT_DIR / "figures"

for _p in (REPO / "src", REPO / "research" / "scripts"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from data_forensics_2026 import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_LOCAL_OOF_ROOT,
    FOLDS,
    RT1264_STREAMS,
    RT600_STREAMS,
    CalibratedBlend,
    read_dev_metadata,
)

PHASE_NAMES = {
    0: "never_break",
    1: "prebreak_far",
    2: "prebreak_near",
    3: "postbreak_early",
    4: "postbreak_mature",
}
SHAPE_ORDER = ("level_up", "level_down", "scale_up", "scale_down", "tail_up", "tail_down", "subtle")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=os.environ.get("SBR_DATA_ROOT", str(DEFAULT_DATA_ROOT)))
    p.add_argument("--local-oof-root", default=os.environ.get("SBR_LOCAL_OOF_ROOT", str(DEFAULT_LOCAL_OOF_ROOT)))
    return p.parse_args()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "nogit"


def finite_mean(x: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def finite_std(x: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1)) if len(x) > 1 else None


def finite_rate(mask: np.ndarray) -> float | None:
    return float(np.asarray(mask, dtype=bool).mean()) if len(mask) else None


def as_float(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


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


def add_dev_offsets(idx: dict) -> pd.DataFrame:
    meta = idx["meta_dev"].copy()
    starts = np.r_[0, np.cumsum(meta["n_online"].to_numpy()[:-1])].astype(np.int64)
    meta["dev_start"] = starts
    meta["dev_end"] = starts + meta["n_online"].to_numpy().astype(np.int64)
    return meta


def shape_family(row: dict) -> str:
    pre_n = int(row.get("pre100_rows") or 0)
    post_n = int(row.get("post100_rows") or 0)
    if pre_n < 25 or post_n < 25:
        return "boundary_limited"
    level = as_float(row.get("level_shift_100"))
    scale = as_float(row.get("scale_log_100"))
    tail = as_float(row.get("tail2_shift_100"))
    if level is None or scale is None or tail is None:
        return "boundary_limited"
    if abs(level) < 0.20 and abs(scale) < 0.20 and abs(tail) < 0.03:
        return "subtle"
    scores = {
        "level": abs(level),
        "scale": abs(scale),
        "tail": 4.0 * abs(tail),
    }
    winner = max(scores, key=scores.get)
    if winner == "level":
        return "level_up" if level >= 0 else "level_down"
    if winner == "scale":
        return "scale_up" if scale >= 0 else "scale_down"
    return "tail_up" if tail >= 0 else "tail_down"


def shape_strength(row: dict) -> float | None:
    level = as_float(row.get("level_shift_100"))
    scale = as_float(row.get("scale_log_100"))
    tail = as_float(row.get("tail2_shift_100"))
    if level is None or scale is None or tail is None:
        return None
    return float(abs(level) + abs(scale) + 4.0 * abs(tail))


def build_series_metrics(idx: dict, rt600: np.ndarray, rt1264: np.ndarray) -> pd.DataFrame:
    meta = add_dev_offsets(idx)
    rows: list[dict] = []
    for r in meta.itertuples(index=False):
        lo = int(r.dev_start)
        hi = int(r.dev_end)
        z = idx["z"][lo:hi].astype(np.float64)
        phase = idx["phase"][lo:hi]
        rel = idx["rel_t"][lo:hi]
        delta = rt1264[lo:hi] - rt600[lo:hi]
        t = idx["t"][lo:hi]
        tau = int(r.tau_index)
        rec: dict = {
            "series_id": int(r.id),
            "series_idx": int(r.series_idx),
            "fold": int(r.fold),
            "has_break": bool(r.has_break),
            "tau_index": tau,
            "off": int(r.off),
            "n_hist": int(r.n_hist),
            "n_online": int(r.n_online),
            "dev_start": lo,
            "dev_end": hi,
            "online_mean_z": finite_mean(z),
            "online_abs_z_mean": finite_mean(np.abs(z)),
            "online_max_abs_z": float(np.nanmax(np.abs(z))) if len(z) else None,
            "online_tail2_rate": finite_rate(np.abs(z) > 2.0),
            "online_tail3_rate": finite_rate(np.abs(z) > 3.0),
            "score_delta_mean": finite_mean(delta),
            "rt600_score_mean": finite_mean(rt600[lo:hi]),
            "rt1264_score_mean": finite_mean(rt1264[lo:hi]),
        }
        for code, name in PHASE_NAMES.items():
            m = phase == code
            rec[f"{name}_rows"] = int(m.sum())
            rec[f"{name}_mean_z"] = finite_mean(z[m])
            rec[f"{name}_abs_z_mean"] = finite_mean(np.abs(z[m]))
            rec[f"{name}_tail2_rate"] = finite_rate(np.abs(z[m]) > 2.0)
            rec[f"{name}_tail3_rate"] = finite_rate(np.abs(z[m]) > 3.0)
            rec[f"{name}_score_delta"] = finite_mean(delta[m])

        early_rel = (rel >= 0.10) & (rel < 0.25)
        near_boundary = (phase == 2) | (phase == 3)
        rec["early_rel_rows"] = int(early_rel.sum())
        rec["early_rel_score_delta"] = finite_mean(delta[early_rel])
        rec["near_boundary_rows"] = int(near_boundary.sum())
        rec["near_boundary_score_delta"] = finite_mean(delta[near_boundary])

        if tau >= 0:
            pre = (t >= max(0, tau - 100)) & (t < tau)
            post = (t >= tau) & (t < min(int(r.n_online), tau + 100))
            pre_z = z[pre]
            post_z = z[post]
            pre_std = finite_std(pre_z)
            post_std = finite_std(post_z)
            rec.update({
                "pre100_rows": int(pre.sum()),
                "post100_rows": int(post.sum()),
                "pre100_mean_z": finite_mean(pre_z),
                "post100_mean_z": finite_mean(post_z),
                "pre100_std_z": pre_std,
                "post100_std_z": post_std,
                "pre100_tail2_rate": finite_rate(np.abs(pre_z) > 2.0),
                "post100_tail2_rate": finite_rate(np.abs(post_z) > 2.0),
            })
            if pre_std is not None and post_std is not None:
                rec["level_shift_100"] = as_float(rec["post100_mean_z"]) - as_float(rec["pre100_mean_z"])
                rec["scale_log_100"] = float(math.log((post_std + 1e-6) / (pre_std + 1e-6)))
                rec["tail2_shift_100"] = as_float(rec["post100_tail2_rate"]) - as_float(rec["pre100_tail2_rate"])
            else:
                rec["level_shift_100"] = None
                rec["scale_log_100"] = None
                rec["tail2_shift_100"] = None
        else:
            rec.update({
                "pre100_rows": 0,
                "post100_rows": 0,
                "pre100_mean_z": None,
                "post100_mean_z": None,
                "pre100_std_z": None,
                "post100_std_z": None,
                "pre100_tail2_rate": None,
                "post100_tail2_rate": None,
                "level_shift_100": None,
                "scale_log_100": None,
                "tail2_shift_100": None,
            })
        rec["shape_family"] = shape_family(rec)
        rec["shape_strength"] = shape_strength(rec)
        rows.append(rec)
    df = pd.DataFrame(rows)
    never = df[(~df["has_break"]) & (df["n_online"] >= 400)]
    med_abs = float(never["online_abs_z_mean"].median())
    med_mean = float(never["online_mean_z"].median())
    df["never_stable_score"] = (df["online_abs_z_mean"] - med_abs).abs() + (df["online_mean_z"] - med_mean).abs()
    df["abs_online_mean_z"] = df["online_mean_z"].abs()
    df["post_scale_tail_strength"] = df["postbreak_mature_abs_z_mean"] - df["prebreak_far_abs_z_mean"]
    return df


def choose_unique(df: pd.DataFrame, specs: list[dict], group: str) -> pd.DataFrame:
    used: set[int] = set()
    picks = []
    for spec in specs:
        sub = df[spec["mask"](df)].copy()
        sub = sub[~sub["series_id"].isin(used)]
        sub = sub[np.isfinite(sub[spec["key"]].to_numpy(dtype=float))]
        if sub.empty:
            continue
        sub = sub.sort_values([spec["key"], "series_id"], ascending=[spec["ascending"], True])
        row = sub.iloc[0].copy()
        used.add(int(row["series_id"]))
        row["atlas_group"] = group
        row["atlas_label"] = spec["label"]
        row["selection_rule"] = spec["rule"]
        picks.append(row)
    return pd.DataFrame(picks)


def select_examples(df: pd.DataFrame) -> pd.DataFrame:
    raw_specs = [
        {
            "label": "never_stable",
            "key": "never_stable_score",
            "ascending": True,
            "mask": lambda d: (~d["has_break"]) & (d["n_online"] >= 400),
            "rule": "never-break, n_online>=400, closest to never-break median online_abs_z_mean and online_mean_z",
        },
        {
            "label": "never_tail_outlier",
            "key": "online_max_abs_z",
            "ascending": False,
            "mask": lambda d: ~d["has_break"],
            "rule": "never-break, largest online_max_abs_z",
        },
        {
            "label": "never_level_drift",
            "key": "abs_online_mean_z",
            "ascending": False,
            "mask": lambda d: ~d["has_break"],
            "rule": "never-break, largest abs(online_mean_z)",
        },
        {
            "label": "prebreak_near_tail",
            "key": "prebreak_near_abs_z_mean",
            "ascending": False,
            "mask": lambda d: d["has_break"] & (d["prebreak_near_rows"] >= 25),
            "rule": "break series, at least 25 near-prebreak rows, largest prebreak_near_abs_z_mean",
        },
        {
            "label": "postbreak_level_up",
            "key": "postbreak_mature_mean_z",
            "ascending": False,
            "mask": lambda d: d["has_break"] & (d["postbreak_mature_rows"] >= 50),
            "rule": "break series, at least 50 mature-postbreak rows, largest postbreak_mature_mean_z",
        },
        {
            "label": "postbreak_level_down",
            "key": "postbreak_mature_mean_z",
            "ascending": True,
            "mask": lambda d: d["has_break"] & (d["postbreak_mature_rows"] >= 50),
            "rule": "break series, at least 50 mature-postbreak rows, smallest postbreak_mature_mean_z",
        },
        {
            "label": "postbreak_scale_tail",
            "key": "post_scale_tail_strength",
            "ascending": False,
            "mask": lambda d: d["has_break"] & (d["postbreak_mature_rows"] >= 50),
            "rule": "break series, at least 50 mature-postbreak rows, largest mature abs-z minus far-prebreak abs-z",
        },
        {
            "label": "postbreak_subtle",
            "key": "shape_strength",
            "ascending": True,
            "mask": lambda d: d["shape_family"] == "subtle",
            "rule": "break series classified subtle, smallest fixed shape_strength",
        },
    ]
    model_specs = [
        {
            "label": "rt1264_mature_lift",
            "key": "postbreak_mature_score_delta",
            "ascending": False,
            "mask": lambda d: d["has_break"] & (d["postbreak_mature_rows"] >= 50),
            "rule": "break series, largest mean RT1264-RT600 score delta on mature-postbreak rows",
        },
        {
            "label": "rt1264_mature_drop",
            "key": "postbreak_mature_score_delta",
            "ascending": True,
            "mask": lambda d: d["has_break"] & (d["postbreak_mature_rows"] >= 50),
            "rule": "break series, smallest mean RT1264-RT600 score delta on mature-postbreak rows",
        },
        {
            "label": "rt1264_never_score_reduction",
            "key": "never_break_score_delta",
            "ascending": True,
            "mask": lambda d: ~d["has_break"],
            "rule": "never-break series, smallest mean RT1264-RT600 score delta",
        },
        {
            "label": "rt1264_never_score_increase",
            "key": "never_break_score_delta",
            "ascending": False,
            "mask": lambda d: ~d["has_break"],
            "rule": "never-break series, largest mean RT1264-RT600 score delta",
        },
        {
            "label": "early_rel_loss",
            "key": "early_rel_score_delta",
            "ascending": True,
            "mask": lambda d: d["early_rel_rows"] >= 20,
            "rule": "at least 20 rows in relative position [.10,.25), smallest mean score delta",
        },
        {
            "label": "early_rel_gain",
            "key": "early_rel_score_delta",
            "ascending": False,
            "mask": lambda d: d["early_rel_rows"] >= 20,
            "rule": "at least 20 rows in relative position [.10,.25), largest mean score delta",
        },
        {
            "label": "near_boundary_lift",
            "key": "near_boundary_score_delta",
            "ascending": False,
            "mask": lambda d: d["near_boundary_rows"] >= 40,
            "rule": "at least 40 near-boundary rows, largest mean score delta",
        },
        {
            "label": "near_boundary_drop",
            "key": "near_boundary_score_delta",
            "ascending": True,
            "mask": lambda d: d["near_boundary_rows"] >= 40,
            "rule": "at least 40 near-boundary rows, smallest mean score delta",
        },
    ]
    return pd.concat(
        [choose_unique(df, raw_specs, "raw_archetype"), choose_unique(df, model_specs, "model_anchored")],
        ignore_index=True,
    )


def values_for_series(data_root: Path, row: pd.Series) -> tuple[np.ndarray, np.ndarray, float, float]:
    values = np.load(data_root / "cache" / "store" / "values.npy", mmap_mode="r")
    off = int(row["off"])
    nh = int(row["n_hist"])
    no = int(row["n_online"])
    hist = np.asarray(values[off: off + nh], dtype=np.float64)
    online = np.asarray(values[off + nh: off + nh + no], dtype=np.float64)
    mu = float(hist.mean())
    sd = max(float(hist.std(ddof=1)) if len(hist) > 1 else 1.0, 1e-9)
    return (hist - mu) / sd, (online - mu) / sd, mu, sd


def plot_trace_gallery(
    selected: pd.DataFrame,
    data_root: Path,
    rt600: np.ndarray,
    rt1264: np.ndarray,
    path: Path,
    title: str,
    include_scores: bool,
) -> None:
    rows = selected.reset_index(drop=True)
    n = len(rows)
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 16), constrained_layout=True)
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    for i, (_, row) in enumerate(rows.iterrows()):
        ax = axes[i]
        ax.axis("on")
        hist_z, online_z, _, _ = values_for_series(data_root, row)
        hist_tail = hist_z[-300:]
        xh = np.arange(-len(hist_tail), 0)
        xo = np.arange(len(online_z))
        ax.plot(xh, hist_tail, color="#9ca3af", linewidth=0.75, alpha=0.85)
        ax.plot(xo, online_z, color="#111827", linewidth=0.9)
        ax.axhline(0, color="#6b7280", linewidth=0.6)
        ax.axhline(2, color="#d1d5db", linewidth=0.5, linestyle=":")
        ax.axhline(-2, color="#d1d5db", linewidth=0.5, linestyle=":")
        if bool(row["has_break"]):
            tau = int(row["tau_index"])
            ax.axvline(tau, color="#dc2626", linewidth=1.0)
            ax.axvspan(tau, len(online_z), color="#fecaca", alpha=0.18)
        ax.set_xlim(-len(hist_tail), max(len(online_z), 1))
        ax.set_ylim(-6, 6)
        ax.set_title(
            f"{row['atlas_label']} | id={int(row['series_id'])} fold={int(row['fold'])} "
            f"tau={int(row['tau_index'])} n={int(row['n_online'])} {row['shape_family']}",
            fontsize=8,
        )
        ax.tick_params(labelsize=7)
        if include_scores:
            lo = int(row["dev_start"])
            hi = int(row["dev_end"])
            ax2 = ax.twinx()
            ax2.plot(xo, rt600[lo:hi], color="#2563eb", linewidth=0.8, alpha=0.8)
            ax2.plot(xo, rt1264[lo:hi], color="#059669", linewidth=0.8, alpha=0.8)
            ax2.set_ylim(0.0, 1.0)
            ax2.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_break_shape_bands(metrics: pd.DataFrame, idx: dict, data_root: Path, path: Path) -> None:
    values = np.load(data_root / "cache" / "store" / "values.npy", mmap_mode="r")
    families = [f for f in SHAPE_ORDER if int((metrics["shape_family"] == f).sum()) > 0]
    if not families:
        return
    offsets = np.arange(-100, 200)
    fig, axes = plt.subplots(nrows=math.ceil(len(families) / 2), ncols=2, figsize=(14, 3.2 * math.ceil(len(families) / 2)), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, family in zip(axes, families):
        ax.axis("on")
        traces = []
        sub = metrics[metrics["shape_family"] == family]
        for _, row in sub.iterrows():
            tau = int(row["tau_index"])
            n = int(row["n_online"])
            if tau < 0:
                continue
            off = int(row["off"])
            nh = int(row["n_hist"])
            hist = np.asarray(values[off: off + nh], dtype=np.float64)
            online = np.asarray(values[off + nh: off + nh + n], dtype=np.float64)
            mu = float(hist.mean())
            sd = max(float(hist.std(ddof=1)) if len(hist) > 1 else 1.0, 1e-9)
            online_z = (online - mu) / sd
            trace = np.full(len(offsets), np.nan, dtype=np.float64)
            pos = tau + offsets
            ok = (pos >= 0) & (pos < n)
            trace[ok] = online_z[pos[ok]]
            traces.append(trace)
        if not traces:
            continue
        M = np.vstack(traces)
        med = np.nanmedian(M, axis=0)
        q25 = np.nanquantile(M, 0.25, axis=0)
        q75 = np.nanquantile(M, 0.75, axis=0)
        ax.fill_between(offsets, q25, q75, color="#93c5fd", alpha=0.35, linewidth=0)
        ax.plot(offsets, med, color="#1d4ed8", linewidth=1.4)
        ax.axvline(0, color="#dc2626", linewidth=1.0)
        ax.axhline(0, color="#6b7280", linewidth=0.6)
        ax.set_ylim(-2.5, 2.5)
        ax.set_title(f"{family} (n={len(traces)})", fontsize=10)
        ax.tick_params(labelsize=8)
    fig.suptitle("Standardized raw online value around tau by fixed break-shape family", fontsize=14)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_phase_absz_box(idx: dict, path: Path) -> None:
    stride = 20
    pos = np.arange(0, len(idx["z"]), stride)
    absz = np.abs(idx["z"][pos])
    phase = idx["phase"][pos]
    data = [absz[phase == code] for code in range(5)]
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.boxplot(data, tick_labels=[PHASE_NAMES[c] for c in range(5)], showfliers=False)
    ax.set_ylabel("abs(z), z standardized by historical segment")
    ax.set_title("Raw amplitude distributions by offline phase, deterministic dev-row sample")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_raw_score_links(metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5.5), constrained_layout=True)
    br = metrics[metrics["has_break"] & np.isfinite(metrics["post_scale_tail_strength"]) & np.isfinite(metrics["postbreak_mature_score_delta"])]
    nb = metrics[(~metrics["has_break"]) & np.isfinite(metrics["online_abs_z_mean"]) & np.isfinite(metrics["never_break_score_delta"])]
    axes[0].scatter(br["post_scale_tail_strength"], br["postbreak_mature_score_delta"], s=8, alpha=0.35, color="#7c3aed")
    axes[0].axhline(0, color="#6b7280", linewidth=0.7)
    axes[0].axvline(0, color="#6b7280", linewidth=0.7)
    axes[0].set_xlabel("mature abs(z) mean - far-prebreak abs(z) mean")
    axes[0].set_ylabel("mean RT1264 - RT600 score delta on mature-postbreak rows")
    axes[0].set_title("Break series: raw scale/tail shift vs score lift")
    axes[1].scatter(nb["online_abs_z_mean"], nb["never_break_score_delta"], s=8, alpha=0.35, color="#0f766e")
    axes[1].axhline(0, color="#6b7280", linewidth=0.7)
    axes[1].set_xlabel("never-break online abs(z) mean")
    axes[1].set_ylabel("mean RT1264 - RT600 score delta")
    axes[1].set_title("Never-break series: raw amplitude vs score delta")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def shape_summary(metrics: pd.DataFrame) -> list[dict]:
    out = []
    break_df = metrics[metrics["has_break"]]
    for family, sub in break_df.groupby("shape_family"):
        out.append({
            "shape_family": family,
            "series": int(len(sub)),
            "share_of_break_series": float(len(sub) / max(len(break_df), 1)),
            "median_level_shift_100": as_float(sub["level_shift_100"].median()),
            "median_scale_log_100": as_float(sub["scale_log_100"].median()),
            "median_tail2_shift_100": as_float(sub["tail2_shift_100"].median()),
            "median_mature_score_delta": as_float(sub["postbreak_mature_score_delta"].median()),
        })
    order = {name: i for i, name in enumerate(("boundary_limited",) + SHAPE_ORDER)}
    return sorted(out, key=lambda r: order.get(r["shape_family"], 999))


def correlations(metrics: pd.DataFrame) -> list[dict]:
    specs = [
        ("break_post_scale_tail_vs_mature_score_delta", "post_scale_tail_strength", "postbreak_mature_score_delta", metrics["has_break"]),
        ("break_level_shift_abs_vs_mature_score_delta", "level_shift_abs", "postbreak_mature_score_delta", metrics["has_break"]),
        ("break_tail2_shift_vs_mature_score_delta", "tail2_shift_100", "postbreak_mature_score_delta", metrics["has_break"]),
        ("never_absz_mean_vs_score_delta", "online_abs_z_mean", "never_break_score_delta", ~metrics["has_break"]),
        ("never_tail3_rate_vs_score_delta", "online_tail3_rate", "never_break_score_delta", ~metrics["has_break"]),
        ("early_rel_absz_vs_early_score_delta", "online_abs_z_mean", "early_rel_score_delta", metrics["early_rel_rows"] >= 20),
    ]
    df = metrics.copy()
    df["level_shift_abs"] = df["level_shift_100"].abs()
    rows = []
    for name, left, right, mask in specs:
        sub = df.loc[mask, [left, right]].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({
            "diagnostic": name,
            "left": left,
            "right": right,
            "series": int(len(sub)),
            "pearson": float(sub[left].corr(sub[right], method="pearson")) if len(sub) > 2 else None,
            "spearman": float(sub[left].corr(sub[right], method="spearman")) if len(sub) > 2 else None,
        })
    return rows


def write_report(result: dict) -> None:
    shape_rows = result["shape_family_summary"]
    corr_rows = result["raw_score_correlations"]
    selected = result["selected_examples"]
    headline = result["score_reconstruction"]
    lines = [
        "# RAW_DATA_ATLAS_2026 -- FINAL REPORT",
        "",
        "Date: 2026-08-28",
        "Branch: `research/raw-data-atlas-2026`",
        f"Analysis SHA: `{result['git_sha']}`",
        "",
        "## Scope",
        "",
        "This atlas looks at the actual stored time-series values on dev folds only. It trains no model, writes no OOF vector, consumes no RT ID, appends no `RESULTS.csv` row, and does not summarize lockbox/test data.",
        "",
        "## Figures",
        "",
        "![Raw archetype gallery](figures/raw_archetype_gallery.png)",
        "",
        "![Model anchored gallery](figures/model_anchored_gallery.png)",
        "",
        "![Break shape bands](figures/break_shape_bands.png)",
        "",
        "![Phase abs-z box](figures/phase_absz_box.png)",
        "",
        "![Raw score links](figures/raw_score_links.png)",
        "",
        "## Score Context",
        "",
        f"The fold-pure reconstruction used only existing OOF streams. RT600 mean TS-AUC is `{headline['rt600_mean_ts_auc']:.9f}`; RT-1264 mean TS-AUC is `{headline['rt1264_mean_ts_auc']:.9f}`; delta is `{headline['delta_mean_ts_auc']:+.9f}`.",
        "",
        "## Break Shape Families",
        "",
        "| family | series | share | median level shift | median scale log | median tail2 shift | median mature score delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in shape_rows:
        lines.append(
            f"| `{row['shape_family']}` | {row['series']} | {row['share_of_break_series']:.4f} | "
            f"{fmt(row['median_level_shift_100'])} | {fmt(row['median_scale_log_100'])} | "
            f"{fmt(row['median_tail2_shift_100'])} | {fmt(row['median_mature_score_delta'])} |"
        )
    lines += [
        "",
        "The raw break families are heterogeneous. Boundary-limited cases are common because many breaks happen close to the start or end of the online segment. Among well-windowed breaks, the largest families are level and scale/tail changes, but the `subtle` bucket is material: some labelled breaks have only weak local standardized-amplitude evidence around `tau`.",
        "",
        "## Selected Series",
        "",
        "| group | label | id | fold | tau | n_online | shape | online abs-z | mature delta | never delta |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| `{row['atlas_group']}` | `{row['atlas_label']}` | {row['series_id']} | {row['fold']} | "
            f"{row['tau_index']} | {row['n_online']} | `{row['shape_family']}` | "
            f"{fmt(row.get('online_abs_z_mean'))} | {fmt(row.get('postbreak_mature_score_delta'))} | "
            f"{fmt(row.get('never_break_score_delta'))} |"
        )
    lines += [
        "",
        "The model-anchored examples make the same point as the numeric forensics: RT-1264's useful changes are not visually equivalent to a single raw threshold. Some large raw excursions are never-breaks; some true breaks are low-amplitude or delayed. The score overlays often separate regimes gradually rather than at an obvious point discontinuity.",
        "",
        "## Raw Metric / Score Links",
        "",
        "| diagnostic | series | Pearson | Spearman |",
        "|---|---:|---:|---:|",
    ]
    for row in corr_rows:
        lines.append(
            f"| `{row['diagnostic']}` | {row['series']} | {fmt(row['pearson'])} | {fmt(row['spearman'])} |"
        )
    lines += [
        "",
        "The correlations are descriptive, not model-selection evidence. They are useful mainly for ruling out simplistic stories: raw amplitude and local tail metrics explain some visible regimes, but they are too weak and mixed to become standalone hand rules without a new preregistered experiment.",
        "",
        "## Conclusion",
        "",
        "- The data itself is not a clean step-change detection problem; labelled breaks include level, scale, tail, delayed, and subtle regimes.",
        "- Never-break false positives can look like plausible shocks under raw historical z-scores.",
        "- RT-1264's dev gain does not come from an obvious visual threshold; deployment review should focus on early-relative-position behavior and never-break high-score mass.",
        "- This atlas authorizes no production, router, threshold, or feature change.",
    ]
    (OUT_DIR / "RAW_DATA_ATLAS_REPORT.md").write_text("\n".join(lines) + "\n")


def fmt(x) -> str:
    v = as_float(x)
    return "" if v is None else f"{v:.6f}"


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    local_oof_root = Path(args.local_oof_root).resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    idx = read_dev_metadata(data_root)
    blender = CalibratedBlend(idx, [local_oof_root, data_root / "research" / "oof"])
    rt600, rt600_per = blender.blend(RT600_STREAMS)
    rt1264, rt1264_per = blender.blend(RT1264_STREAMS)

    metrics = build_series_metrics(idx, rt600, rt1264)
    metrics.to_csv(OUT_DIR / "series_raw_metrics.csv", index=False)
    selected = select_examples(metrics)
    selected.to_csv(OUT_DIR / "selected_series.csv", index=False)
    shape_rows = shape_summary(metrics)
    write_csv(OUT_DIR / "shape_family_summary.csv", shape_rows)
    corr_rows = correlations(metrics)
    write_csv(OUT_DIR / "raw_score_correlations.csv", corr_rows)

    raw_selected = selected[selected["atlas_group"] == "raw_archetype"]
    model_selected = selected[selected["atlas_group"] == "model_anchored"]
    plot_trace_gallery(raw_selected, data_root, rt600, rt1264, FIG_DIR / "raw_archetype_gallery.png", "Raw trace archetypes, dev folds only", False)
    plot_trace_gallery(model_selected, data_root, rt600, rt1264, FIG_DIR / "model_anchored_gallery.png", "Model-anchored raw traces with calibrated score overlays", True)
    plot_break_shape_bands(metrics, idx, data_root, FIG_DIR / "break_shape_bands.png")
    plot_phase_absz_box(idx, FIG_DIR / "phase_absz_box.png")
    plot_raw_score_links(metrics, FIG_DIR / "raw_score_links.png")

    result = {
        "program": "RAW_DATA_ATLAS_2026",
        "git_sha": git_sha(),
        "data_root": str(data_root),
        "local_oof_root": str(local_oof_root),
        "dev_only": True,
        "lockbox_summarized": False,
        "test_reduced_touched": False,
        "score_reconstruction": {
            "rt600_mean_ts_auc": float(np.mean(rt600_per)),
            "rt1264_mean_ts_auc": float(np.mean(rt1264_per)),
            "delta_mean_ts_auc": float(np.mean(rt1264_per) - np.mean(rt600_per)),
            "rt600_per_fold": [float(x) for x in rt600_per],
            "rt1264_per_fold": [float(x) for x in rt1264_per],
        },
        "shape_family_summary": shape_rows,
        "raw_score_correlations": corr_rows,
        "selected_examples": json.loads(selected.to_json(orient="records")),
        "figures": [
            "figures/raw_archetype_gallery.png",
            "figures/model_anchored_gallery.png",
            "figures/break_shape_bands.png",
            "figures/phase_absz_box.png",
            "figures/raw_score_links.png",
        ],
    }
    (OUT_DIR / "raw_data_atlas_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_report(result)
    print(json.dumps({
        "program": "RAW_DATA_ATLAS_2026",
        "report": str(OUT_DIR / "RAW_DATA_ATLAS_REPORT.md"),
        "selected_series": len(selected),
        "rt1264_delta": result["score_reconstruction"]["delta_mean_ts_auc"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
