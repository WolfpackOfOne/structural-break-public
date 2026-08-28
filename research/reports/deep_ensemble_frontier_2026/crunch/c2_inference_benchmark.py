#!/usr/bin/env python
"""C2 -- paired online-inference benchmark for RT-600 vs RT-1257.

Engineering measurement only. Allocates no RT ID, appends no ledger, scores
nothing, tunes nothing, and touches neither test nor lockbox data. It reads
model artifacts and the feature store from other worktrees READ-ONLY and writes
only under this lane's own report directory.

Why this exists when engineering/reports/rt1257_deployment/RT1257_BENCHMARK.json
already has numbers:

  1. That benchmark draws a DIFFERENT random series sample per arm
     (`default_rng(600 if out.name == "final10k_ensemble" else 1257)`), so its
     headline `ratio_vs_rt600 = 0.912` compares two models over two different
     populations. Series composition and model cost are confounded.
  2. It is a single unreplicated shot per arm. Re-running the RT-1257 arm
     unchanged reproduced `score_sum_guard` bitwise but moved
     `ms_per_online_point` from 2.2775 to 2.4663 -- ~8% run-to-run drift on one
     machine, which is the same size as the ~9% effect being claimed.

This script fixes both: ONE series sample shared by both arms, and R repeats
with the arm order alternated so machine drift cancels rather than accumulating
into whichever arm ran second. It reports the PAIRED per-repeat ratio, which is
the estimator that actually isolates model cost.

It also decomposes `ProductionModel.step()` into
    engine.step (shared, model-independent) | per-member predict | blend
so the cost of a CatBoost slot and a LightGBM slot are measured separately. That
is what lets the k-scaling question be answered by measurement rather than
extrapolation, per the C2 brief.

Usage:
    python c2_inference_benchmark.py --repeats 3 --series 40
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

SB = Path("/path/to/workspace")
RT1257_WORKTREE = SB / "structural-break-rt1257-deployment"
ARTIFACT_ROOT = SB / "structural-break-claude-wave3"
RT600_MODEL = ARTIFACT_ROOT / "models" / "final10k_ensemble"
RT1257_MODEL = RT1257_WORKTREE / "models" / "rt1257_final"

QUOTA_H = 15.0
N_SERIES_TARGET = 10_000.0


def configure_runtime():
    """Mirror rt1257_deployment.configure_runtime, reading from the deployment
    worktree read-only. We do not import that module, to avoid its REPO-relative
    write paths."""
    os.environ["SBR_ROOT"] = str(ARTIFACT_ROOT)
    for p in (RT1257_WORKTREE / "src", RT1257_WORKTREE / "research" / "scripts"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)

    import numpy as np
    import sbr.pipeline as PL

    PL.ROOT = str(ARTIFACT_ROOT)
    PL.FEAT = str(ARTIFACT_ROOT / "cache" / "features")
    PL.FEAT_SCREEN = str(ARTIFACT_ROOT / "cache" / "features_screen")
    PL.FOLDS = str(ARTIFACT_ROOT / "research" / "folds" / "folds_final10k.parquet")
    PL.STORE = str(ARTIFACT_ROOT / "cache" / "store")
    PL.OOF = str(ARTIFACT_ROOT / "research" / "oof")
    return np, PL


def maxrss_mb() -> float:
    import resource

    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw / (1024.0 * 1024.0) if raw > 10_000_000 else raw / 1024.0


def time_arm(model, series) -> dict:
    """One full pass over the shared series sample.

    Records BOTH wall time and process CPU time. This machine is shared with a
    concurrent agent session, and wall time under contention is unreliable --
    the first paired run showed 36% wall spread across repeats. All member
    predicts are pinned single-threaded (`num_threads=1` / `thread_count=1`), so
    CPU time is the contention-robust estimator and is what the summary uses.
    Wall time is retained because the competition budget is a wall-clock budget.

    No predictive score is computed or reported beyond a sum guard used to prove
    both arms ran the same work.
    """
    fixed_s = fixed_c = 0.0
    point_s = point_c = 0.0
    n_points = 0
    score_sum = 0.0
    for h, o in series:
        a, ac = time.perf_counter(), time.process_time()
        model.start_series(h)
        fixed_s += time.perf_counter() - a
        fixed_c += time.process_time() - ac
        b, bc = time.perf_counter(), time.process_time()
        for x in o:
            score_sum += float(model.step(float(x)))
        point_s += time.perf_counter() - b
        point_c += time.process_time() - bc
        n_points += len(o)
    n = max(n_points, 1)
    return {
        "fixed_ms_per_series": 1000.0 * fixed_s / max(len(series), 1),
        "fixed_cpu_ms_per_series": 1000.0 * fixed_c / max(len(series), 1),
        "ms_per_online_point": 1000.0 * point_s / n,
        "cpu_ms_per_online_point": 1000.0 * point_c / n,
        "n_points": n_points,
        "score_sum_guard": score_sum,
    }


def decompose(model, series, np) -> dict:
    """Split step() into engine / per-member predict / blend.

    Re-implements step()'s body with timers around each part. The member loop is
    timed per member so LightGBM and CatBoost slots are separated. This changes
    timing slightly (timer overhead per member) so it is reported as a
    decomposition of SHARE, not as an absolute replacement for time_arm.
    """
    n_members = len(model.members)
    member_c = [0.0] * n_members
    engine_c = 0.0
    blend_c = 0.0
    n_points = 0

    for h, o in series:
        model.start_series(h)
        eng = model._engine
        for x in o:
            a = time.process_time()
            row = eng.step(float(x))[None, :]
            t = eng.ctx.t
            b = time.process_time()
            engine_c += b - a
            ps = []
            for i, (m, s) in enumerate(zip(model.members, model.slices)):
                c = time.process_time()
                ps.append(m.predict_one(row[:, s]))
                member_c[i] += time.process_time() - c
            d = time.process_time()
            model._blend(ps, t)
            blend_c += time.process_time() - d
            n_points += 1

    kinds = [m.kind for m in model.members]
    per_member_ms = [1000.0 * s / max(n_points, 1) for s in member_c]
    return {
        "n_points": n_points,
        "timer": "process_time (CPU), contention-robust",
        "engine_cpu_ms_per_point": 1000.0 * engine_c / max(n_points, 1),
        "blend_cpu_ms_per_point": 1000.0 * blend_c / max(n_points, 1),
        "member_kinds": kinds,
        "member_cpu_ms_per_point": per_member_ms,
        "total_member_cpu_ms_per_point": float(sum(per_member_ms)),
        "mean_cpu_ms_per_point_by_kind": {
            k: float(np.mean([v for v, kk in zip(per_member_ms, kinds) if kk == k]))
            for k in sorted(set(kinds))
        },
    }


def project_k(decomp_rt600: dict, decomp_rt1257: dict, mean_online: float,
              fixed_ms: float) -> dict:
    """Measured k-scaling, not extrapolated from a single point.

    The ensemble is SEVEN SLOTS. A CatBoost specialist REPLACES a LightGBM one
    rather than being added, so member count stays 7 for every k and only the
    mix changes:

        member_cost(k) = (7 - k) * lgb_slot + k * cat_slot

    Per-slot costs are measured directly by decompose(). `lgb_slot` is taken
    from the RT-600 arm (all seven slots LightGBM); `cat_slot` from RT-1257's
    two CatBoost slots. Shared per-point cost (feature engine + blend) is taken
    from RT-600 and held constant, which it is by construction -- neither
    depends on the model mix.

    Stated assumption: cost is additive and linear in the slot mix. That holds
    here because the members are independent single-row predicts in a Python
    loop with no shared state, and it is exactly what decompose() measures. It
    would NOT hold if slots were batched together.
    """
    lgb = decomp_rt600["mean_cpu_ms_per_point_by_kind"]["lightgbm"]
    cat = decomp_rt1257["mean_cpu_ms_per_point_by_kind"]["catboost"]
    shared = (
        decomp_rt600["engine_cpu_ms_per_point"] + decomp_rt600["blend_cpu_ms_per_point"]
    )
    rows = {}
    for k in range(0, 8):
        per_pt = shared + (7 - k) * lgb + k * cat
        secs = N_SERIES_TARGET * (fixed_ms / 1000.0) + N_SERIES_TARGET * mean_online * (
            per_pt / 1000.0
        )
        h = secs / 3600.0
        rows[f"k={k}"] = {
            "n_catboost_slots": k,
            "cpu_ms_per_point": per_pt,
            "projected_10000_series_h": h,
            "pass_quota": h < QUOTA_H,
            "headroom_x": QUOTA_H / max(h, 1e-12),
        }
    return {
        "model": "member_cost(k) = (7-k)*lgb_slot + k*cat_slot, + shared engine/blend",
        "lgb_slot_cpu_ms_per_point": lgb,
        "cat_slot_cpu_ms_per_point": cat,
        "cat_over_lgb_slot_cost_x": cat / max(lgb, 1e-12),
        "shared_engine_plus_blend_cpu_ms_per_point": shared,
        "fixed_ms_per_series_used": fixed_ms,
        "mean_online_horizon": mean_online,
        "quota_h": QUOTA_H,
        "by_k": rows,
    }


def project_hours(fixed_ms: float, point_ms: float, mean_online: float) -> float:
    secs = N_SERIES_TARGET * (fixed_ms / 1000.0) + N_SERIES_TARGET * mean_online * (
        point_ms / 1000.0
    )
    return secs / 3600.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--series", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    np, PL = configure_runtime()
    from sbr.production.model import ProductionModel

    d = PL.Data()
    store = d.st
    mean_online = float(np.mean(store.meta.n_online.to_numpy()))

    # ONE sample, shared by both arms. This is the fix for defect (1).
    rng = np.random.default_rng(args.seed)
    idx = [int(x) for x in rng.choice(store.n_series, args.series, replace=False)]
    series = [store.series(i)[:2] for i in idx]
    total_points = int(sum(len(o) for _, o in series))

    arms = {}
    for name, path in (("rt600", RT600_MODEL), ("rt1257", RT1257_MODEL)):
        t0 = time.perf_counter()
        arms[name] = ProductionModel.load(str(path))
        print(f"loaded {name} in {time.perf_counter() - t0:.3f}s", file=sys.stderr)

    # Warm-up on a small slice so first-touch/JIT costs land outside the
    # measured repeats and do not fall on whichever arm happens to run first.
    for name in ("rt600", "rt1257"):
        time_arm(arms[name], series[:2])

    reps = []
    for r in range(args.repeats):
        # Alternate arm order each repeat so monotone machine drift cancels
        # instead of loading onto one arm. This is the fix for defect (2).
        order = ("rt600", "rt1257") if r % 2 == 0 else ("rt1257", "rt600")
        rec = {"repeat": r, "order": list(order)}
        for name in order:
            rec[name] = time_arm(arms[name], series)
            print(
                f"  rep {r} {name}: {rec[name]['ms_per_online_point']:.4f} ms/pt",
                file=sys.stderr,
            )
        rec["paired_ratio_rt1257_over_rt600"] = (
            rec["rt1257"]["ms_per_online_point"] / rec["rt600"]["ms_per_online_point"]
        )
        rec["paired_cpu_ratio_rt1257_over_rt600"] = (
            rec["rt1257"]["cpu_ms_per_online_point"]
            / rec["rt600"]["cpu_ms_per_online_point"]
        )
        reps.append(rec)

    def agg(name, key):
        vals = [r[name][key] for r in reps]
        return {
            "values": vals,
            "median": float(statistics.median(vals)),
            "mean": float(statistics.fmean(vals)),
            "stdev": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
            "spread_pct": (
                100.0 * (max(vals) - min(vals)) / max(statistics.fmean(vals), 1e-12)
            ),
        }

    ratios = [r["paired_ratio_rt1257_over_rt600"] for r in reps]
    cpu_ratios = [r["paired_cpu_ratio_rt1257_over_rt600"] for r in reps]
    summary = {}
    for name in ("rt600", "rt1257"):
        pt = agg(name, "ms_per_online_point")
        cpt = agg(name, "cpu_ms_per_online_point")
        fx = agg(name, "fixed_ms_per_series")
        h = project_hours(fx["median"], pt["median"], mean_online)
        h_cpu = project_hours(
            agg(name, "fixed_cpu_ms_per_series")["median"], cpt["median"], mean_online
        )
        summary[name] = {
            "ms_per_online_point": pt,
            "cpu_ms_per_online_point": cpt,
            "fixed_ms_per_series": fx,
            "projected_10000_series_h_wall": h,
            "projected_10000_series_h_cpu": h_cpu,
            "quota_h": QUOTA_H,
            "pass_quota_wall": h < QUOTA_H,
            "pass_quota_cpu": h_cpu < QUOTA_H,
            "headroom_x_cpu": QUOTA_H / max(h_cpu, 1e-12),
            "score_sum_guard": reps[0][name]["score_sum_guard"],
        }

    print("decomposing step() per member ...", file=sys.stderr)
    decomp = {
        name: decompose(arms[name], series[: max(4, args.series // 8)], np)
        for name in ("rt600", "rt1257")
    }
    k_projection = project_k(
        decomp["rt600"],
        decomp["rt1257"],
        mean_online,
        summary["rt600"]["fixed_ms_per_series"]["median"],
    )

    result = {
        "task": "C2",
        "kind": "engineering_inference_benchmark",
        "note": (
            "Paired design: one shared series sample, arm order alternated across "
            "repeats. Supersedes the confounded per-arm sampling in "
            "engineering/reports/rt1257_deployment/RT1257_BENCHMARK.json."
        ),
        "allocates_rt_id": False,
        "appends_results_csv": False,
        "touched_test_or_lockbox": False,
        "host": {
            "platform": sys.platform,
            "python": sys.version.split()[0],
            "note": "LOCAL machine, NOT the Crunch platform runtime.",
        },
        "sample": {
            "n_series": len(series),
            "total_online_points": total_points,
            "seed": args.seed,
            "shared_across_arms": True,
            "mean_online_horizon": mean_online,
        },
        "repeats": reps,
        "summary": summary,
        "paired_ratio_rt1257_over_rt600": {
            "values": ratios,
            "median": float(statistics.median(ratios)),
            "mean": float(statistics.fmean(ratios)),
            "stdev": float(statistics.stdev(ratios)) if len(ratios) > 1 else 0.0,
        },
        "paired_cpu_ratio_rt1257_over_rt600": {
            "values": cpu_ratios,
            "median": float(statistics.median(cpu_ratios)),
            "mean": float(statistics.fmean(cpu_ratios)),
            "stdev": float(statistics.stdev(cpu_ratios)) if len(cpu_ratios) > 1 else 0.0,
        },
        "step_decomposition": decomp,
        "k_projection": k_projection,
        "peak_rss_mb": maxrss_mb(),
    }

    out = args.out or str(
        Path(__file__).resolve().parent / "C2_RT1257_INFERENCE_BENCHMARK.json"
    )
    Path(out).write_text(json.dumps(result, indent=2) + "\n")

    print("\n=== per-slot cost (CPU, measured) ===")
    print(f"  LightGBM slot : {k_projection['lgb_slot_cpu_ms_per_point']:.4f} ms/pt")
    print(f"  CatBoost slot : {k_projection['cat_slot_cpu_ms_per_point']:.4f} ms/pt")
    print(f"  ratio         : {k_projection['cat_over_lgb_slot_cost_x']:.2f}x")
    print("\n=== projected hours per 10,000 series vs 15 h quota ===")
    for key, row in k_projection["by_k"].items():
        flag = "PASS" if row["pass_quota"] else "FAIL"
        print(
            f"  {key}: {row['cpu_ms_per_point']:.4f} ms/pt -> "
            f"{row['projected_10000_series_h']:6.3f} h  {flag} "
            f"({row['headroom_x']:.1f}x headroom)"
        )
    print("\npaired CPU ratio rt1257/rt600:")
    print(json.dumps(result["paired_cpu_ratio_rt1257_over_rt600"], indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
