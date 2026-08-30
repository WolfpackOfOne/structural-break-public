#!/usr/bin/env python
"""Moonshot kill gate: is the reconstructed generator actually the DGP?

Grok (agent_05 section 5) and Kimi (agent_02 section 5) independently proposed
the SAME moonshot -- identify the generator family, then train an amortized
posterior on unlimited simulated series with known tau -- and independently
specified the same kill gate. This script runs it once and discharges both.

    Grok: "Do NOT train SNPE. First: fit a tiny parametric generator (AR-GARCH +
    rare variance jump + t-noise) by matching the taxonomy table. Draw 2,000
    simulated series. Train the existing RT-600 pipeline on simulated data, score
    on real fold-0. If transfer AUC is <0.55, the generator is not the DGP and
    the moonshot dies for ~1 h."

    Kimi: "(i) after a bounded search, no simulator family matches >= 4/5 of the
    artifact battery; (ii) a simulator-trained model scores < 0.55 on real
    fold-0."

Both halves are run: the artifact battery, and then the transfer test. The
transfer test is the decisive one, and it is deliberately generous to the
moonshot -- the simulator gets to choose its own best parameters against the
real battery before transfer is measured, so a failure here is not a failure of
tuning.

Nothing here trains an amortized posterior. That is the point: this gate exists
so that the expensive stage is never entered on an unvalidated generator.

No RT ID is allocated and RESULTS.csv is not touched.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = REPO.parent / "structural-break-wave6"


def _preparse_artifact_root() -> Path:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--artifact-root", default=os.environ.get("SBR_ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT)))
    args, _ = p.parse_known_args()
    return Path(args.artifact_root).resolve()


ARTIFACT_ROOT = _preparse_artifact_root()
os.environ["SBR_ROOT"] = str(ARTIFACT_ROOT)

for _p in (REPO / "src", REPO / "research" / "scripts"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from sbr.metric import ts_auc_flat  # noqa: E402
from sbr.store import load_store  # noqa: E402
from sbr.transforms import HistParams, _ar_resid, ar_filter_causal  # noqa: E402
from wave5_lib import Ctx, FOLDS  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", default=os.environ.get("SBR_ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT)))
    p.add_argument("--output-dir", default=str(REPO / "research" / "reports" / "moonshot_generator_kill_gate"))
    p.add_argument("--n-sim", type=int, default=2000)
    p.add_argument("--n-real-battery", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260830)
    return p.parse_args()


def git_sha() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()


# ------------------------------------------------------------- simulator ---


def simulate_series(rng, n_hist: int, n_online: int, params: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """AR(p) + GARCH-ish persistent variance + t-noise, with a rare permanent jump.

    This is the family both agents named. A break, when present, is a permanent
    change to the innovation scale and/or the AR dependence, placed at a
    near-uniform tau -- matching the measured tau law.
    """
    p_ar = params["ar_order"]
    phi = params["phi"] * (params["phi_decay"] ** np.arange(p_ar))
    nu = params["t_df"]
    n = n_hist + n_online
    burn = 200

    tau_rel = rng.uniform(0.05, 0.95)
    has_break = rng.random() < params["break_rate"]
    tau = int(tau_rel * n_online) if has_break else -1

    scale_mult = np.ones(n + burn)
    phi_mult = np.ones(n + burn)
    if has_break:
        cut = burn + n_hist + tau
        fam = rng.random()
        if fam < params["p_scale"]:
            lo, hi = params["scale_jump"]
            m = rng.uniform(lo, hi)
            scale_mult[cut:] = m if rng.random() < 0.5 else 1.0 / m
        elif fam < params["p_scale"] + params["p_dep"]:
            phi_mult[cut:] = rng.uniform(*params["dep_jump"])
        else:
            m = rng.uniform(*params["scale_jump"])
            scale_mult[cut:] = m if rng.random() < 0.5 else 1.0 / m
            phi_mult[cut:] = rng.uniform(*params["dep_jump"])

    # Persistent conditional variance produces the long on-attractor excursions
    # the repo measures as transients, without any break present.
    a, b, w = params["garch_a"], params["garch_b"], params["garch_w"]
    sig2 = np.empty(n + burn)
    sig2[0] = w / max(1.0 - a - b, 1e-3)
    x = np.zeros(n + burn)
    eps = rng.standard_t(nu, size=n + burn) / np.sqrt(nu / (nu - 2.0))
    for i in range(1, n + burn):
        sig2[i] = w + a * (x[i - 1] - np.dot(phi[: min(i - 1, p_ar)][::-1] * phi_mult[i - 1],
                                             x[max(i - 1 - p_ar, 0): i - 1]) if i > p_ar else 0.0) ** 2 \
            + b * sig2[i - 1]
        mean = 0.0
        if i > p_ar:
            mean = float(np.dot(phi[::-1] * phi_mult[i], x[i - p_ar: i]))
        x[i] = mean + np.sqrt(max(sig2[i], 1e-12)) * scale_mult[i] * eps[i]

    x = x[burn:]
    hist, online = x[:n_hist], x[n_hist:]
    # Histories in this competition are exactly standardised; match that.
    mu, sd = float(hist.mean()), float(hist.std(ddof=0)) or 1.0
    return (hist - mu) / sd, (online - mu) / sd, tau


# ---------------------------------------------------------- the battery ---


def series_battery_stats(hist: np.ndarray, online: np.ndarray, tau: int, split: int | None = None) -> dict:
    """Battery statistics for one series.

    `split` is the segmentation point used to form pre/post windows. For break
    series it is the true tau; for never-break series it MUST be a pseudo-tau
    drawn from the positive rel_tau law, never the midpoint. Splitting positives
    at their true change point and negatives at a fixed midpoint manufactures
    separation in every family statistic -- it reports location AUC near 0.57
    where the repo measures 0.4995. The sibling branch's 2025 compatibility
    audit already solved this with pseudo_taus_2026; the same construction is
    used here.
    """
    hp = HistParams(np.asarray(hist, dtype=np.float64), ar_order=6)
    zh = (hist - hp.mu) / hp.sd
    eh = np.concatenate([np.zeros(6), _ar_resid(zh, hp.ar_coef)]) / hp.ar_sigma
    eo = ar_filter_causal(np.asarray(online, dtype=np.float64), hp.ar_coef,
                          np.asarray(hist, dtype=np.float64)) / hp.ar_sigma
    cut = split if split is not None else (tau if tau >= 0 else len(online) // 2)
    cut = int(min(max(cut, 5), max(len(online) - 5, 5)))
    post = online[cut:]
    pre = online[:cut]
    ref_sd = float(np.std(hist, ddof=1)) or 1.0

    def safe(v, d=np.nan):
        return float(v) if np.isfinite(v) else d

    sm = None
    if len(eh) > 64:
        s = eh * eh
        cs = np.concatenate([[0.0], np.cumsum(s)])
        sm = (cs[64:] - cs[:-64]) / 64.0
    transient = np.nan
    if sm is not None and len(sm) > 10:
        med = float(np.median(sm))
        thr = med + float(np.quantile(np.abs(sm - med), 0.90))
        transient = float((sm > thr).mean())
    eo_f = eo[np.isfinite(eo)]
    eo_pre = eo[:cut][np.isfinite(eo[:cut])]
    eo_post = eo[cut:][np.isfinite(eo[cut:])]
    q = np.linspace(0.05, 0.95, 19)
    w1 = safe(np.mean(np.abs(np.quantile(post, q) - np.quantile(pre, q))) / ref_sd) if len(post) > 20 and len(pre) > 20 else np.nan
    return {
        "hist_mean": safe(np.mean(hist)),
        "hist_sd": safe(np.std(hist, ddof=0)),
        "ar6_resid_logsd": safe(np.log(np.std(eo_f, ddof=1) + 1e-12)),
        "loc_abs": safe(abs(np.mean(post) - np.mean(pre)) / ref_sd),
        "scale_abs": safe(abs(np.log((np.std(post, ddof=1) + 1e-12) / (np.std(pre, ddof=1) + 1e-12)))),
        "dep_abs": safe(abs(_acf1(post) - _acf1(pre))),
        "shape_abs": safe(abs(_kurt(post) - _kurt(pre))),
        "transient_rate": transient,
        "hist_kurt": safe(_kurt(hist)),
        # richer channels: the transfer test needs a real-trained reference well
        # above chance or it cannot distinguish "simulator is wrong" from
        # "features are weak".
        "resid_logsd_ratio": safe(np.log((np.std(eo_post, ddof=1) + 1e-12) / (np.std(eo_pre, ddof=1) + 1e-12))) if len(eo_post) > 20 and len(eo_pre) > 20 else np.nan,
        "resid_absmean_ratio": safe(np.log((np.mean(np.abs(eo_post)) + 1e-12) / (np.mean(np.abs(eo_pre)) + 1e-12))) if len(eo_post) > 20 and len(eo_pre) > 20 else np.nan,
        "resid_energy_max_post": safe(np.max(eo_post ** 2)) if len(eo_post) > 5 else np.nan,
        "wasserstein_pre_post": w1,
        "online_vs_hist_logsd": safe(np.log((np.std(online, ddof=1) + 1e-12) / (np.std(hist, ddof=1) + 1e-12))),
        "online_vs_hist_acf1": safe(abs(_acf1(online) - _acf1(hist))),
        "online_tail_rate": safe(np.mean(np.abs(eo_f) > 3.0)) if len(eo_f) else np.nan,
        "online_resid_kurt": safe(_kurt(eo_f)),
    }


def _acf1(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 5:
        return np.nan
    z = x - x.mean()
    den = float(np.dot(z, z))
    return float(np.dot(z[1:], z[:-1]) / den) if den > 0 else np.nan


def _kurt(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 5:
        return np.nan
    z = x - x.mean()
    v = float(np.mean(z * z))
    return float(np.mean(z**4) / (v * v) - 3.0) if v > 1e-12 else np.nan


def auc(y: np.ndarray, s: np.ndarray) -> float:
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if len(np.unique(y)) < 2:
        return float("nan")
    o = np.argsort(s, kind="stable")
    r = np.empty(len(s), dtype=np.float64)
    r[o] = np.arange(1, len(s) + 1)
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def battery(rows: list[dict], labels: np.ndarray) -> dict:
    def col(k):
        return np.array([r.get(k, np.nan) for r in rows], dtype=np.float64)

    return {
        "break_rate": float(np.mean(labels)),
        "hist_mean_abs": float(np.nanmean(np.abs(col("hist_mean")))),
        "hist_sd_mean": float(np.nanmean(col("hist_sd"))),
        "hist_kurt_median": float(np.nanmedian(col("hist_kurt"))),
        "transient_rate_mean": float(np.nanmean(col("transient_rate"))),
        "ar6_resid_logsd_auc": auc(labels, col("ar6_resid_logsd")),
        "location_auc": auc(labels, np.abs(col("loc_abs"))),
        "scale_auc": auc(labels, col("scale_abs")),
        "dependence_auc": auc(labels, col("dep_abs")),
        "shape_auc": auc(labels, col("shape_abs")),
    }


BATTERY_TOLERANCE = {
    "location_auc": 0.04,
    "scale_auc": 0.06,
    "dependence_auc": 0.06,
    "shape_auc": 0.06,
    "ar6_resid_logsd_auc": 0.06,
    "transient_rate_mean": 0.08,
    "hist_kurt_median": 1.50,
}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    c = Ctx()
    st = load_store(str(ARTIFACT_ROOT / "cache" / "store"))

    # ---- real battery -----------------------------------------------------
    dev_series = np.unique(c.d.sidx[c.dev])
    take = rng.choice(dev_series, min(args.n_real_battery, len(dev_series)), replace=False)
    pos_rel = np.array([
        c.tau[s] / max(c.n_online[s], 1) for s in dev_series
        if c.has_break[s] and c.tau[s] >= 0
    ], dtype=np.float64)
    real_rows, real_y, real_shapes = [], [], []
    for sid in take:
        h = np.asarray(st.hist(int(sid)), dtype=np.float64)
        o = np.asarray(st.online(int(sid)), dtype=np.float64)
        if len(h) < 200 or len(o) < 100:
            continue
        tau = int(c.tau[sid])
        split = tau if (c.has_break[sid] and tau >= 0) else int(rng.choice(pos_rel) * len(o))
        real_rows.append(series_battery_stats(h, o, tau, split=split))
        real_y.append(1 if c.has_break[sid] else 0)
        real_shapes.append((len(h), len(o)))
    real_y = np.array(real_y, dtype=np.int8)
    real_b = battery(real_rows, real_y)

    # ---- simulator search -------------------------------------------------
    # Bounded search, as Kimi's kill criterion (i) specifies. The simulator is
    # allowed to pick its best configuration against the REAL battery before the
    # transfer test, so transfer failure cannot be blamed on untuned parameters.
    grid = []
    for phi0 in (0.15, 0.30, 0.45):
        for a in (0.05, 0.12):
            for b in (0.80, 0.90):
                for df in (5.0, 8.0, 20.0):
                    grid.append(dict(
                        ar_order=6, phi=phi0, phi_decay=0.55, t_df=df,
                        garch_a=a, garch_b=b, garch_w=0.05,
                        break_rate=float(real_b["break_rate"]),
                        p_scale=0.45, p_dep=0.35,
                        scale_jump=(1.15, 2.0), dep_jump=(0.3, 2.2),
                    ))
    best = None
    for gi, params in enumerate(grid):
        srng = np.random.default_rng(args.seed + 1000 + gi)
        rows, ys = [], []
        for i in range(400):  # small draw for the search stage
            n_h, n_o = real_shapes[srng.integers(len(real_shapes))]
            h, o, tau = simulate_series(srng, n_h, n_o, params)
            split = tau if tau >= 0 else int(srng.choice(pos_rel) * len(o))
            rows.append(series_battery_stats(h, o, tau, split=split))
            ys.append(1 if tau >= 0 else 0)
        b = battery(rows, np.array(ys, dtype=np.int8))
        matched = {
            k: bool(np.isfinite(b.get(k, np.nan)) and np.isfinite(real_b.get(k, np.nan))
                    and abs(b[k] - real_b[k]) <= tol)
            for k, tol in BATTERY_TOLERANCE.items()
        }
        n_matched = int(sum(matched.values()))
        loss = float(np.nansum([
            abs(b.get(k, np.nan) - real_b.get(k, np.nan)) / tol for k, tol in BATTERY_TOLERANCE.items()
        ]))
        cand = {"params": params, "battery": b, "matched": matched,
                "n_matched": n_matched, "loss": loss}
        if best is None or (n_matched, -loss) > (best["n_matched"], -best["loss"]):
            best = cand
    n_criteria = len(BATTERY_TOLERANCE)
    battery_pass = bool(best["n_matched"] >= int(np.ceil(0.8 * n_criteria)))

    # ---- transfer test ----------------------------------------------------
    # Grok's gate says "train the existing RT-600 pipeline on simulated data,
    # score on real fold-0". Earlier revisions of this script used hand-rolled
    # summary statistics instead, and the real-trained reference on those topped
    # out at 0.5950 -- below the 0.55 threshold's useful range, so the test had
    # no power to separate "wrong simulator" from "weak features". This runs the
    # actual feature modules on simulated series, at row level, scored by TS-AUC,
    # which is the metric the project is judged on.
    import lightgbm as lgb
    from sbr.features.base import REGISTRY, load_all, make_ctx

    load_all()
    modules = ["m00_core", "m07_bayes"]

    def build_bank(hist, online):
        ctx = make_ctx(np.asarray(hist, dtype=np.float64), np.asarray(online, dtype=np.float64), ar_order=6)
        mats, names = [], []
        for m in modules:
            nm, arr = REGISTRY[m].fn(ctx)
            mats.append(np.asarray(arr, dtype=np.float32))
            names.extend(f"{m}::{x}" for x in nm)
        return np.column_stack(mats), names

    srng = np.random.default_rng(args.seed + 77)
    sim_x, sim_y, sim_names = [], [], None
    t_sim = time.time()
    for i in range(args.n_sim):
        n_h, n_o = real_shapes[srng.integers(len(real_shapes))]
        h, o, tau = simulate_series(srng, n_h, n_o, best["params"])
        try:
            xb, nm = build_bank(h, o)
        except Exception:
            continue
        if sim_names is None:
            sim_names = nm
        elif nm != sim_names:
            continue
        yb = np.zeros(len(o), dtype=np.float64)
        if tau >= 0:
            yb[tau:] = 1.0
        sim_x.append(xb)
        sim_y.append(yb)
        if i and i % 250 == 0:
            print(f"  simulated bank {i}/{args.n_sim} in {time.time() - t_sim:.0f}s", flush=True)
    xs = np.vstack(sim_x)
    ys = np.concatenate(sim_y)

    # Real rows, same modules, from the real cache -- identical column order.
    import sbr.pipeline as PL

    rmats, rnames = PL.load_features(modules)
    if rnames != sim_names:
        raise SystemExit("simulated and real feature column orders differ; refusing to compare")
    fold0_rows = c.rows[0]
    other_rows = np.concatenate([c.rows[f] for f in FOLDS if f != 0])
    lim = np.random.default_rng(args.seed).permutation(len(other_rows))[:600_000]
    # PL._stack walks rows as contiguous spans, so the index array must stay sorted.
    other_rows = np.sort(other_rows[lim])
    xr0 = PL._stack(rmats, rnames, fold0_rows, np.arange(len(rnames)))
    xro = PL._stack(rmats, rnames, other_rows, np.arange(len(rnames)))

    params_lgb = dict(objective="binary", learning_rate=0.05, num_leaves=63,
                      min_data_in_leaf=300, feature_fraction=0.8, bagging_fraction=0.8,
                      bagging_freq=1, verbose=-1, seed=args.seed)
    if len(xs) > 800_000:
        sel = np.random.default_rng(args.seed).permutation(len(xs))[:800_000]
        xs, ys = xs[sel], ys[sel]
    booster = lgb.train(params_lgb, lgb.Dataset(xs, label=ys), num_boost_round=300)
    pred_sim = booster.predict(np.nan_to_num(xr0, nan=0.0))
    transfer_auc = float(ts_auc_flat(pred_sim, c.d.y[fold0_rows], c.d.t[fold0_rows]))

    # Reference: identical features, learner and evaluation rows, trained on REAL
    # data. This is what separates "the simulator is not the DGP" from "these
    # features cannot see breaks".
    booster_real = lgb.train(params_lgb, lgb.Dataset(xro, label=c.d.y[other_rows].astype(np.float64)),
                             num_boost_round=300)
    real_ref_auc = float(ts_auc_flat(booster_real.predict(np.nan_to_num(xr0, nan=0.0)),
                                     c.d.y[fold0_rows], c.d.t[fold0_rows]))
    yr = c.d.y[fold0_rows]
    n_sim_used = len(sim_x)

    transfer_pass = bool(np.isfinite(transfer_auc) and transfer_auc >= 0.55)
    # A bare threshold cannot discriminate when the features saturate below it.
    # Retention against the real-trained reference on identical features and rows
    # is the statistic with power: it asks how much of the achievable signal the
    # simulator actually transfers.
    lift_real = real_ref_auc - 0.5
    retention = float((transfer_auc - 0.5) / lift_real) if np.isfinite(real_ref_auc) and lift_real > 0.02 else float("nan")
    power_ok = bool(np.isfinite(real_ref_auc) and real_ref_auc >= 0.60)
    retention_pass = bool(np.isfinite(retention) and retention >= 0.60)
    passed = bool(battery_pass and transfer_pass and power_ok and retention_pass)
    result = {
        "experiment": "moonshot_generator_kill_gate",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": git_sha(),
        "discharges": [
            "agent_05_grok46.md section 5 (SNPE / ABC-SMC on a reconstructed generator)",
            "agent_02_kimi_k3.md section 5 (DGP reverse-engineering -> simulation-based amortized inference)",
        ],
        "note": "Both agents proposed the same moonshot and the same kill gate, independently.",
        "n_real_battery": len(real_rows),
        "n_simulated_transfer": args.n_sim,
        "n_simulated_series_used": n_sim_used,
        "transfer_protocol": "row-level TS-AUC on real fold 0; real feature modules "
                             "m00_core + m07_bayes built on simulated series",
        "grid_size": len(grid),
        "real_battery": real_b,
        "best_simulator": {"params": best["params"], "battery": best["battery"],
                           "matched": best["matched"], "n_matched": best["n_matched"],
                           "n_criteria": n_criteria, "loss": best["loss"]},
        "battery_tolerance": BATTERY_TOLERANCE,
        "gates": {
            "battery_match": {"n_matched": best["n_matched"], "n_criteria": n_criteria,
                              "required": int(np.ceil(0.8 * n_criteria)), "passed": battery_pass},
            "transfer_auc_real_fold0": {"observed": transfer_auc, "threshold": 0.55,
                                        "passed": transfer_pass},
        },
        "real_trained_reference_auc": real_ref_auc,
        "transfer_retention_of_real_reference": retention,
        "gates_power": {
            "real_reference_auc": real_ref_auc,
            "required_for_test_to_have_power": 0.60,
            "passed": power_ok,
            "note": "If the real-trained reference is at or below the 0.55 transfer threshold, "
                    "the transfer test cannot distinguish a wrong simulator from weak features "
                    "and no verdict may be drawn from it.",
        },
        "gates_retention": {"retention": retention, "threshold": 0.60, "passed": retention_pass},
        "n_real_fold0_rows": int(len(yr)),
        "runtime_s": round(time.time() - t0, 1),
    }
    if not power_ok:
        result["verdict"] = (
            f"NO VERDICT: the test lacks power. The real-trained reference reaches only "
            f"{real_ref_auc:.4f} on these features, so the {0.55} transfer threshold cannot "
            "distinguish a wrong simulator from weak features. Enrich the feature set before "
            "reading anything into the transfer number."
        )
        (out_dir / "moonshot_generator_kill_gate.json").write_text(json.dumps(result, indent=2) + "\n")
        write_report(result, out_dir)
        print(json.dumps({"verdict": result["verdict"]}, indent=2))
        return
    result["verdict"] = (
        f"PROCEED to amortized inference: the generator matches {best['n_matched']}/{n_criteria} "
        f"battery criteria, a simulator-trained model reaches {transfer_auc:.4f} on real fold 0, "
        f"and that is {retention:.2f} of the real-trained reference's lift over chance."
        if passed
        else (
            f"KILL both moonshots. Battery match {best['n_matched']}/{n_criteria} "
            f"(required {int(np.ceil(0.8 * n_criteria))}); simulator-trained transfer AUC on real "
            f"fold 0 is {transfer_auc:.4f} against the {0.55} threshold both agents set, while the "
            f"identical features and learner trained on REAL data reach {real_ref_auc:.4f} on the "
            f"same rows, so the simulator transfers {retention:.2f} of the achievable lift "
            "against a 0.60 requirement. The features are not the problem; the generator is not "
            "the DGP. The expensive amortization stage is never entered."
        )
    )
    (out_dir / "moonshot_generator_kill_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(result, out_dir)
    print(json.dumps({"verdict": result["verdict"]}, indent=2))


def fmt(x, d: int = 4) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "nan" if not np.isfinite(v) else f"{v:.{d}f}"


def write_report(r: dict, out_dir: Path) -> None:
    md = [
        "# Moonshot Kill Gate — Reconstructed Generator",
        "",
        f"Date: `{r['generated']}`  ·  git `{r['git_sha']}`",
        "",
        "Grok and Kimi independently proposed the same moonshot and independently",
        "specified the same kill gate. This runs it once and discharges both:",
        "",
    ] + [f"- {d}" for d in r["discharges"]] + [
        "",
        "**No amortized posterior was trained.** That is the purpose of the gate: the",
        "expensive stage is entered only against a validated generator.",
        "",
        "## Artifact battery",
        "",
        f"Bounded search over `{r['grid_size']}` simulator configurations. The simulator is",
        "allowed to pick its best configuration against the real battery *before* the",
        "transfer test, so a transfer failure cannot be blamed on untuned parameters.",
        "",
        "| statistic | real | best simulator | tolerance | matched |",
        "|---|---:|---:|---:|---|",
    ]
    best = r["best_simulator"]
    for k, tol in r["battery_tolerance"].items():
        md.append(
            f"| {k} | {fmt(r['real_battery'].get(k))} | {fmt(best['battery'].get(k))} | "
            f"{tol} | {'yes' if best['matched'].get(k) else 'no'} |"
        )
    md += [
        "",
        f"Matched `{best['n_matched']}/{best['n_criteria']}`; required "
        f"`{r['gates']['battery_match']['required']}`.",
        "",
        "## Transfer test",
        "",
        "Train on simulated series only, score on real fold 0. Series-level, because a",
        "simulated corpus shares no row space with the real one — which is the protocol",
        "both agents' kill gates name.",
        "",
        "| model | AUC on real fold 0 |",
        "|---|---:|",
        f"| trained on {r.get('n_simulated_series_used', r['n_simulated_transfer'])} simulated series | "
        f"**{fmt(r['gates']['transfer_auc_real_fold0']['observed'])}** |",
        f"| identical features and learner, trained on real data | "
        f"{fmt(r['real_trained_reference_auc'])} |",
        "",
        f"Threshold `0.55`, row-level TS-AUC on `{r['n_real_fold0_rows']}` real fold-0 rows.",
        "",
        "The real-trained reference is the control that makes the result readable: it",
        "separates 'these summary features are too weak' from 'the simulator is not the",
        "DGP'. Only the second explanation kills the moonshot, and only the second is",
        "consistent with a real-trained model doing well on the same rows and features.",
        "",
        "## Verdict",
        "",
        r["verdict"],
        "",
    ]
    (out_dir / "moonshot_generator_kill_gate.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
