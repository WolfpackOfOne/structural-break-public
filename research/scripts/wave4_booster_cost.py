"""W4-E4: what does a booster actually cost at inference?

The wave-2 decision to deploy 7 full-data boosters rather than 35 fold-models
was justified by "costs 5x the inference".  That was asserted, not measured, and
the brief is right to distrust it: the architecture is ONE shared streaming
feature engine feeding N boosters a slice each, so the marginal cost of a
booster is one 500-column row -> one tree walk, not one feature pass.

This measures it.  1, 7, 14 and 35 boosters over the SAME feature stream, on
real series from the store.  For counts above 7 the seven real boosters are
CYCLED -- a duplicate booster has exactly the cost structure of a fold-model
(same trees, same depth, same slice), and this experiment measures COST ONLY.
It says nothing about the score of a fold-model ensemble and is never quoted as
if it did.

Run it on an otherwise idle machine.  Timings taken under training load are
meaningless.
"""
from __future__ import annotations

import json, os, sys, time

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")

import numpy as np
from sbr.production.model import ProductionModel
from sbr.store import load_store

MODEL_DIR = os.environ.get("SBR_MODEL_DIR",
                           "/path/to/workspace/structural-break-codex-wave3/models/rt150_ensemble")
N_SERIES = int(os.environ.get("W4_N_SERIES", 12))
OUT = f"{ROOT}/research/reports/wave4_booster_cost.json"


def replicate(base: ProductionModel, n: int) -> ProductionModel:
    """A model with n boosters, cycling the real ones.  COST ONLY."""
    k = len(base.boosters)
    m = ProductionModel(
        [base.boosters[i % k] for i in range(n)],
        dict(base.manifest, booster_columns=[base.manifest["booster_columns"][i % k]
                                             for i in range(n)]))
    if base._cals is not None:
        m._cals = [base._cals[i % k] for i in range(n)]
        m.calibration = dict(base.calibration,
                             models=[base.calibration["models"][i % k] for i in range(n)])
    return m


def main():
    t0 = time.time()
    st = load_store(os.environ.get("SBR_STORE", f"{ROOT}/cache/store"))
    rng = np.random.default_rng(0)
    ids = rng.choice(len(st.meta), N_SERIES, replace=False)

    print(f"loading {MODEL_DIR}")
    t = time.time()
    base = ProductionModel.load(MODEL_DIR)
    load_s = time.time() - t
    print(f"  {len(base.boosters)} boosters loaded in {load_s:.2f}s")

    series = []
    for i in ids:
        h, o, _tau = st.series(int(i))
        series.append((np.asarray(h, dtype=np.float64), np.asarray(o, dtype=np.float64)))
    n_points = sum(len(o) for _, o in series)
    print(f"  {len(series)} series, {n_points} online points")

    res = {"model_dir": MODEL_DIR, "n_series": len(series), "n_points": int(n_points),
           "base_boosters": len(base.boosters), "base_load_s": load_s, "counts": {}}

    for n in (1, 7, 14, 35):
        m = replicate(base, n)
        per_point = []
        t = time.time()
        for h, o in series:
            m.start_series(h)
            for x in o:
                a = time.perf_counter()
                m.step(float(x))
                per_point.append(time.perf_counter() - a)
        wall = time.time() - t
        p = np.array(per_point) * 1000.0
        res["counts"][str(n)] = {
            "wall_s": wall, "ms_per_point_wall": 1000.0 * wall / n_points,
            "p50_ms": float(np.percentile(p, 50)), "p95_ms": float(np.percentile(p, 95)),
            "p99_ms": float(np.percentile(p, 99)), "mean_ms": float(p.mean()),
        }
        print(f"  {n:3d} boosters: {1000.0*wall/n_points:7.3f} ms/pt wall   "
              f"p50 {np.percentile(p,50):6.3f}  p95 {np.percentile(p,95):6.3f}")

    b1 = res["counts"]["1"]["ms_per_point_wall"]
    b7 = res["counts"]["7"]["ms_per_point_wall"]
    b35 = res["counts"]["35"]["ms_per_point_wall"]
    marg = (b35 - b7) / 28.0
    res["analysis"] = {
        "engine_plus_one_booster_ms": b1,
        "marginal_ms_per_extra_booster": marg,
        "implied_shared_engine_ms": b1 - marg,
        "ratio_35_over_7": b35 / b7,
        "naive_5x_claim_ms": 5 * b7,
        "budget_hours_public_5.0M": {str(n): 5.0e6 * v["ms_per_point_wall"] / 3.6e6
                                     for n, v in res["counts"].items()},
        "budget_hours_private_10.1M": {str(n): 10.1e6 * v["ms_per_point_wall"] / 3.6e6
                                       for n, v in res["counts"].items()},
    }
    a = res["analysis"]
    print(f"\n  shared engine ~{a['implied_shared_engine_ms']:.3f} ms/pt, "
          f"marginal booster ~{marg:.4f} ms/pt")
    print(f"  35 vs 7 boosters: {b35/b7:.2f}x end-to-end (the asserted figure was 5x)")
    print(f"  private-set projection at 35 boosters: "
          f"{a['budget_hours_private_10.1M']['35']:.2f} h against a 15 h budget")

    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
