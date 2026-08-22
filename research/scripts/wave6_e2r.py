"""W6-E2R --- the CORRECTED oracle experiment.  SERIES-LEVEL.  NOT DEPLOYABLE.

W6-E2 / RT-900 was VOID: at the row level the target is y[t] = 1[t >= tau], so
handing the learner tau hands it the label.  See research/WAVE6_STATUS.md and
research/reports/wave6_corrected_oracle.md.

This script moves the same question to the level where it is well posed:

    ONE ROW PER SERIES.  SERIES LABEL.  SERIES ROC AUC.  NEVER TS-AUC.

At one row per series there is no "before vs after" target for the boundary to
encode, which is exactly why the prior oracle-information-frontier study is a
valid instrument and this one inherits its validity.

THE QUESTION
    At the FULL horizon -- the only horizon where the frontier found real
    headroom (+0.0396 over legal RT-300) -- does OUR 500-column causal bank,
    given the same boundary, beat the frontier's GENERIC bank at 0.6497?

    beats it materially   -> representation is still a live lever
    matches it            -> the frontier's bank was already saturated and the
                             remaining gap is about not knowing tau

ARMS (all: FULL horizon, canonical folds 0..4, LGBM 150 trees, series ROC AUC)
    A_rich    frontier generic oracle bank, rich          [REPRODUCTION, 0.6497]
    A_basic   frontier generic oracle bank, basic         [REPRODUCTION, 0.6418]
    B_causal  our 500-col causal bank, evaluated at the boundary split
    C_nobound our 500-col causal bank at the end of the series, NO boundary
    AB        A_rich ++ B_causal, are they complementary

HOW ARM B IS GIVEN THE BOUNDARY
    The engine is run unmodified on a RE-SPLIT series:
        hist'   = hist ++ online[:boundary]
        online' = online[boundary:]
    and the LAST row of its (n_online', 500) output is the series vector.  This
    is the exact analogue of the frontier's compare_segment(post, hist): the
    post-boundary segment against a pre-boundary reference.  No new features are
    invented, no module is modified, nothing is fitted to the label.

WHY t_online / log_t_online ARE DROPPED
    At the last row those two columns ARE post_len, which the frontier excludes
    from every one of its models (MODEL_EXCLUDE_COLUMNS).  Keeping them would
    hand arm B a metadata channel arm A does not have.  The undropped variant is
    still reported, as B_causal_withpos, as a diagnostic.

SENTINELS (primary seed, run and reported BEFORE the arms are interpreted)
    S1 boundary metadata only        must not predict
    S2 missingness pattern only      must not predict
    S3 support / length only         must not predict
    S4 permuted labels               must be ~0.5
    S5 placebo boundary, both classes  must not produce a large AUC

FORBIDDEN AND NOT REACHABLE FROM THIS FILE
    row-level TS-AUC, sbr.metric, the RT-500..RT-506 all-10k OOF vectors,
    folds_final10k.parquet, X_test.reduced / y_test.reduced, fold -1.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")

# import OUR sbr first so the frontier module's own sys.path.insert cannot
# shadow it -- the two copies are byte-identical for store/base/m00..m07, and
# this makes that fact irrelevant rather than load-bearing.
import sbr.store  # noqa: F401
from sbr.features.base import load_all, make_ctx, REGISTRY

PROD_MODULES = ("m00_core", "m01_seq", "m02_dist", "m03_dyn",
                "m04_resid", "m06_loc", "m07_bayes")
#: at the final row these two ARE post_len; the frontier excludes post_len
POSITION_COLS = ("t_online", "log_t_online")

WAVE3 = "/path/to/workspace/structural-break-claude-wave3"
FRONTIER_WT = "/path/to/workspace/structural-break-oracle"
STORE = os.environ.get("W6_STORE", f"{WAVE3}/cache/store")
FOLDS_PQ = os.environ.get("W6_FOLDS", f"{WAVE3}/research/folds/folds.parquet")
FEATCACHE = os.environ.get("W6_FEATCACHE", f"{WAVE3}/cache/features")
OOF_RT300 = f"{WAVE3}/research/oof/RT-300.npy"

PSEUDO_SEEDS = (0, 1, 7, 42, 2026)
FOLDS = (0, 1, 2, 3, 4)
N_ESTIMATORS = 150            # the 2026 frontier's capacity, unchanged
HORIZON = "FULL"

#: the engine's own floor: the shipped store's shortest online segment is 10
#: points, so a re-split with fewer post-boundary points is outside anything the
#: modules were ever built for (and several blocks are undefined there).  The
#: SAME eligibility filter is applied to every head-to-head arm; arm A is ALSO
#: scored on the unfiltered population, which is the reproduction check.
MIN_POST = 10

OUTDIR = f"{ROOT}/research/reports"
CACHE = os.environ.get("W6_CACHE", f"{ROOT}/cache/w6e2r")


# ---------------------------------------------------------------------------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frontier():
    """Import the ORIGINAL frontier script, unmodified, by path."""
    p = Path(FRONTIER_WT) / "research/scripts/oracle_information_frontier.py"
    spec = importlib.util.spec_from_file_location("oracle_frontier", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["oracle_frontier"] = mod
    spec.loader.exec_module(mod)
    return mod, str(p), sha256(p)


# --------------------------------------------------------------- arm B build
_W = {}


def _winit(store_path):
    from sbr.store import load_store
    load_all()
    _W["st"] = load_store(store_path)
    _W["meta"] = _W["st"].meta.set_index("id")


def _wfeat(job):
    """One (id, boundary) -> the 500-column causal state at the last online row."""
    sid, boundary = job
    st = _W["st"]; mr = _W["meta"].loc[sid]
    off = int(mr.off); n_hist = int(mr.n_hist); n_online = int(mr.n_online)
    v = st.values
    hist = np.asarray(v[off:off + n_hist], dtype=np.float64)
    online = np.asarray(v[off + n_hist:off + n_hist + n_online], dtype=np.float64)
    hp = np.concatenate([hist, online[:boundary]]) if boundary > 0 else hist
    op = online[boundary:]
    ctx = make_ctx(hp, op)
    parts = []
    for m in PROD_MODULES:
        _, A = REGISTRY[m].fn(ctx)
        parts.append(np.asarray(A[-1], dtype=np.float64))
    return sid, boundary, np.concatenate(parts)


def causal_bank_columns():
    """Column names in PROD_MODULES order, from the shipped cols.json files."""
    cols = []
    for m in PROD_MODULES:
        cols.extend(json.load(open(f"{FEATCACHE}/{m}.cols.json"))["cols"])
    return cols


def build_causal_bank(jobs, workers, tag):
    """jobs: iterable of (id, boundary).  Returns {(id,boundary): vec}, cached."""
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/bank_{tag}.npz"
    have = {}
    if os.path.exists(path):
        z = np.load(path)
        for sid, b, row in zip(z["ids"], z["bnd"], z["X"]):
            have[(int(sid), int(b))] = row
    todo = [j for j in dict.fromkeys(jobs) if j not in have]
    if todo:
        t0 = time.time()
        import multiprocessing as mp
        ctxmp = mp.get_context("spawn")
        with ctxmp.Pool(workers, initializer=_winit, initargs=(STORE,)) as pool:
            for k, (sid, b, vec) in enumerate(pool.imap_unordered(_wfeat, todo, chunksize=8), 1):
                have[(int(sid), int(b))] = vec.astype(np.float64)
                if k % 500 == 0 or k == len(todo):
                    el = time.time() - t0
                    print(f"  [{tag}] {k}/{len(todo)}  {el:.0f}s  eta {el/k*(len(todo)-k):.0f}s",
                          flush=True)
                if k % 5000 == 0:
                    _save_bank(path, have)
        _save_bank(path, have)
    return have


def _save_bank(path, have):
    keys = sorted(have)
    tmp = path + ".tmp.npz"
    np.savez(tmp,
             ids=np.array([k[0] for k in keys], dtype=np.int64),
             bnd=np.array([k[1] for k in keys], dtype=np.int64),
             X=np.vstack([have[k] for k in keys]))
    os.replace(tmp, path)


def bank_matrix(records, bank, cols, drop_position=True):
    keep = [i for i, c in enumerate(cols)
            if not (drop_position and c in POSITION_COLS)]
    names = [cols[i] for i in keep]
    X = np.vstack([bank[(int(r.id), int(r.boundary))][keep] for r in records.itertuples(index=False)])
    return pd.DataFrame(X, columns=names)


# ------------------------------------------------------------- arm C (cached)
def endstate_matrix(ids, cols, drop_position=True):
    """Last cached row per series: the LEGAL 500-col state, no boundary at all."""
    meta = pd.read_parquet(f"{STORE}/meta.parquet")
    n_on = meta["n_online"].to_numpy(dtype=np.int64)
    off = np.r_[0, np.cumsum(n_on)[:-1]]
    last = dict(zip(meta["id"].to_numpy(dtype=int), off + n_on - 1, strict=True))
    rows = np.array([last[int(i)] for i in ids], dtype=np.int64)
    order = np.argsort(rows); inv = np.argsort(order)
    parts = []
    for m in PROD_MODULES:
        A = np.load(f"{FEATCACHE}/{m}.npy", mmap_mode="r")
        parts.append(np.asarray(A[rows[order]], dtype=np.float64)[inv])
    X = np.hstack(parts)
    keep = [i for i, c in enumerate(cols)
            if not (drop_position and c in POSITION_COLS)]
    return pd.DataFrame(X[:, keep], columns=[cols[i] for i in keep])


# ---------------------------------------------------------------------- eval
def evaluate(F, X, y, folds, ids, seed, label, ref=None, boot=0):
    oof = F.crossfit_model(X, y, folds, "lgbm", seed, n_estimators=N_ESTIMATORS)
    auc, per = F.auc_with_folds(y, oof, folds)
    out = {"label": label, "auc": auc, "per_fold": per, "n_features": X.shape[1],
           "lgbm_seed": seed}
    if boot:
        out["bootstrap"] = F.bootstrap_auc_ci(y, oof, ids, score_b=ref, n_boot=boot, seed=0)
    print(f"    {label:<26s} {auc:.5f}   " + " ".join(f"{p:.5f}" for p in per), flush=True)
    return out, oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seeds", default=",".join(map(str, PSEUDO_SEEDS)))
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--skip-sentinels", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="SMOKE TEST ONLY")
    a = ap.parse_args()
    seeds = tuple(int(s) for s in a.seeds.split(",") if s.strip())

    global OUTJSON
    OUTJSON = f"{OUTDIR}/wave6_corrected_oracle" + (f"_smoke{a.limit}" if a.limit else "") + ".json"
    t_start = time.time()
    F, fpath, fsha = load_frontier()
    load_all()
    cols = causal_bank_columns()
    assert len(cols) == 500, len(cols)

    print("=== W6-E2R  corrected series-level known-boundary representation test ===")
    print(f"frontier module {fpath}\n  sha256 {fsha}")
    for p in (f"{STORE}/meta.parquet", f"{STORE}/values.npy", FOLDS_PQ, OOF_RT300):
        print(f"  {sha256(p)}  {p}")

    print("\nloading series ...", flush=True)
    ids0 = None
    if a.limit:
        af = pd.read_parquet(FOLDS_PQ)
        ids0 = af[af["fold"].isin(FOLDS)]["id"].head(a.limit).tolist()
    series = F.load_2026_series(STORE, FOLDS_PQ, ids=ids0)
    fam = F.derive_break_families(series, seed=0)
    print(f"  {len(series)} dev series", flush=True)

    # ---- every (id, boundary) any arm or sentinel will need, deduped ----
    recs = {}
    for s in seeds:
        df = F.build_2026_records(series, HORIZON, s)
        assert F.one_row_per_series(df) and F.fold_separation_ok(df)
        recs[s] = df
    placebo = F.random_boundary_records(series, recs[seeds[0]], HORIZON, seeds[0])
    elig = {s: (recs[s]["post_len"] >= MIN_POST).to_numpy() for s in seeds}
    elig_pl = (placebo["post_len"] >= MIN_POST).to_numpy()
    for s in seeds:
        d = recs[s]
        print(f"  seed {s}: {len(d)} series, eligible (post_len >= {MIN_POST}) "
              f"{int(elig[s].sum())}  dropped {int((~elig[s]).sum())} "
              f"(pos {int(((~elig[s]) & (d['target']==1)).sum())}, "
              f"neg {int(((~elig[s]) & (d['target']==0)).sum())})", flush=True)
    jobs = []
    for s in seeds:
        d = recs[s][elig[s]]
        jobs += list(zip(d["id"].astype(int), d["boundary"].astype(int)))
    dpl = placebo[elig_pl]
    jobs += list(zip(dpl["id"].astype(int), dpl["boundary"].astype(int)))
    jobs = list(dict.fromkeys(jobs))
    print(f"\ncausal bank: {len(jobs)} distinct (id, boundary) pairs", flush=True)
    bank = build_causal_bank(jobs, a.workers, "full" if not a.limit else f"smoke{a.limit}")

    result = {
        "label": "W6-E2R -- ORACLE / DIAGNOSTIC, SERIES-LEVEL, NOT DEPLOYABLE",
        "supersedes": "W6-E2 / RT-900 (VOID -- label leak via the missingness mask)",
        "metric": "series ROC AUC, one row per series -- NEVER row-level TS-AUC",
        "horizon": HORIZON, "n_estimators": N_ESTIMATORS,
        "pseudo_seeds": list(seeds), "folds": list(FOLDS),
        "frontier_module": {"path": fpath, "sha256": fsha},
        "inputs": {p: sha256(p) for p in
                   (f"{STORE}/meta.parquet", f"{STORE}/values.npy", FOLDS_PQ, OOF_RT300)},
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True).stdout.strip(),
        "position_cols_dropped": list(POSITION_COLS), "min_post": MIN_POST,
        "arms": {}, "sentinels": {}, "per_seed": [],
    }

    # ---------------------------------------------------------------- arms --
    for s in seeds:
        full = recs[s]
        yf = full["target"].to_numpy(dtype=int)
        ff = full["fold"].to_numpy(dtype=int)
        idf = full["id"].to_numpy(dtype=int)
        XAf = full[F.model_feature_columns(full)].copy()
        print(f"\n--- pseudo seed {s}  REPRODUCTION on the unfiltered population "
              f"n={len(full)} ---", flush=True)
        rep = evaluate(F, XAf, yf, ff, idf, s + 17, "A_rich full population")[0]

        df = recs[s][elig[s]].reset_index(drop=True)
        y = df["target"].to_numpy(dtype=int)
        folds = df["fold"].to_numpy(dtype=int)
        ids = df["id"].to_numpy(dtype=int)
        print(f"\n--- pseudo seed {s}   n={len(df)}  pos={int(y.sum())} ---", flush=True)

        fcols = F.model_feature_columns(df)
        XA = df[fcols].copy()
        XAb = df[F.basic_feature_columns(fcols)].copy()
        XB = bank_matrix(df, bank, cols, drop_position=True)
        XBp = bank_matrix(df, bank, cols, drop_position=False)
        XC = endstate_matrix(ids, cols, drop_position=True)
        XAB = pd.concat([XA.reset_index(drop=True), XB.reset_index(drop=True)], axis=1)

        row = {"pseudo_seed": s, "n": int(len(df)), "n_pos": int(y.sum()),
               "n_full": int(len(full)), "A_rich_full_population": rep}
        aA, oA = evaluate(F, XA, y, folds, ids, s + 17, "A_rich frontier")
        row["A_rich"] = aA
        row["A_basic"] = evaluate(F, XAb, y, folds, ids, s, "A_basic frontier")[0]
        row["B_causal"] = evaluate(F, XB, y, folds, ids, s + 17, "B_causal@boundary",
                                   ref=oA, boot=a.boot)[0]
        row["B_causal_withpos"] = evaluate(F, XBp, y, folds, ids, s + 17,
                                           "B_causal +position")[0]
        row["C_nobound"] = evaluate(F, XC, y, folds, ids, s + 17, "C_causal@end")[0]
        row["AB"] = evaluate(F, XAB, y, folds, ids, s + 17, "AB union", ref=oA,
                             boot=a.boot)[0]
        row["delta_B_minus_A"] = row["B_causal"]["auc"] - row["A_rich"]["auc"]
        row["delta_B_minus_C"] = row["B_causal"]["auc"] - row["C_nobound"]["auc"]
        result["per_seed"].append(row)
        json.dump(result, open(OUTJSON, "w"), indent=1)

    # ----------------------------------------------------------- sentinels --
    if not a.skip_sentinels:
        s = seeds[0]
        df = recs[s][elig[s]].reset_index(drop=True)
        y = df["target"].to_numpy(dtype=int)
        folds = df["fold"].to_numpy(dtype=int)
        ids = df["id"].to_numpy(dtype=int)
        XB = bank_matrix(df, bank, cols, drop_position=True)
        print(f"\n--- LEAKAGE SENTINELS (pseudo seed {s}) ---", flush=True)
        sen = {}

        # S1 boundary metadata only
        S1 = df[["boundary", "rel_boundary", "n_hist", "n_online", "post_len"]].copy()
        sen["S1_boundary_metadata"] = evaluate(F, S1, y, folds, ids, s, "S1 boundary meta")[0]

        # S2 missingness pattern only -- of arm B, and of arm A
        MB = pd.DataFrame((~np.isfinite(XB.to_numpy())).astype(float),
                          columns=[f"nan__{c}" for c in XB.columns])
        MB = MB.loc[:, MB.nunique() > 1]
        sen["S2_missingness_B"] = evaluate(
            F, MB if MB.shape[1] else pd.DataFrame({"z": np.zeros(len(y))}),
            y, folds, ids, s, "S2 missingness B")[0]
        XA = df[F.model_feature_columns(df)]
        MA = pd.DataFrame((~np.isfinite(XA.to_numpy())).astype(float),
                          columns=[f"nan__{c}" for c in XA.columns])
        MA = MA.loc[:, MA.nunique() > 1]
        sen["S2_missingness_A"] = evaluate(
            F, MA if MA.shape[1] else pd.DataFrame({"z": np.zeros(len(y))}),
            y, folds, ids, s, "S2 missingness A")[0]

        # S3 support / length only
        S3 = pd.DataFrame({
            "n_valid_B": np.isfinite(XB.to_numpy()).sum(1).astype(float),
            "n_valid_A": np.isfinite(XA.to_numpy()).sum(1).astype(float),
            "pre_len": (df["n_hist"] + df["boundary"]).to_numpy(dtype=float),
            "post_len": df["post_len"].to_numpy(dtype=float),
        })
        sen["S3_support_length"] = evaluate(F, S3, y, folds, ids, s, "S3 support/length")[0]

        # S4 permuted labels on arm B
        rng = np.random.default_rng(12345)
        yp = y[rng.permutation(len(y))]
        sen["S4_permuted_B"] = evaluate(F, XB, yp, folds, ids, s + 17, "S4 permuted B")[0]

        # S5 placebo boundary for BOTH classes
        pl = placebo[elig_pl].reset_index(drop=True)
        yb = pl["target"].to_numpy(dtype=int)
        fb = pl["fold"].to_numpy(dtype=int)
        ib = pl["id"].to_numpy(dtype=int)
        XBpl = bank_matrix(pl, bank, cols, drop_position=True)
        sen["S5_placebo_B"] = evaluate(F, XBpl, yb, fb, ib, s + 17, "S5 placebo B")[0]
        XApl = pl[F.model_feature_columns(pl)].copy()
        sen["S5_placebo_A"] = evaluate(F, XApl, yb, fb, ib, s + 17, "S5 placebo A")[0]

        result["sentinels"] = sen

    # ------------------------------------------------------------- summary --
    def agg(key):
        v = np.array([r[key]["auc"] for r in result["per_seed"]], dtype=float)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "min": float(v.min()), "max": float(v.max())}
    for k in ("A_rich", "A_basic", "B_causal", "B_causal_withpos", "C_nobound", "AB",
              "A_rich_full_population"):
        result["arms"][k] = agg(k)
    result["arms"]["delta_B_minus_A"] = {
        "mean": float(np.mean([r["delta_B_minus_A"] for r in result["per_seed"]])),
        "per_seed": [r["delta_B_minus_A"] for r in result["per_seed"]],
        "seeds_positive": int(sum(r["delta_B_minus_A"] > 0 for r in result["per_seed"])),
    }
    result["arms"]["delta_B_minus_C"] = {
        "mean": float(np.mean([r["delta_B_minus_C"] for r in result["per_seed"]])),
        "per_seed": [r["delta_B_minus_C"] for r in result["per_seed"]],
    }
    result["reference_prior_study"] = {
        "lgbm_rich_FULL": 0.6496853783350913, "lgbm_basic_FULL": 0.6418479346849486,
        "current_RT300_FULL": 0.6100486112738779,
        "source": "research/reports/oracle_information_frontier_2026.csv (codex/oracle-information-frontier-2026 @ 5a3b8a0)",
    }
    result["runtime_s"] = time.time() - t_start
    json.dump(result, open(OUTJSON, "w"), indent=1)

    print("\n=== SUMMARY (series ROC AUC, FULL horizon, mean over pseudo seeds) ===")
    for k in ("A_rich_full_population", "A_rich", "A_basic", "B_causal",
              "B_causal_withpos", "C_nobound", "AB"):
        g = result["arms"][k]
        print(f"  {k:<20s} {g['mean']:.5f} +/- {g['std']:.5f}   [{g['min']:.5f}, {g['max']:.5f}]")
    rr = result["arms"]["A_rich_full_population"]["mean"]
    print(f"  reproduction check: A_rich full population {rr:.5f} "
          f"vs prior 0.64969  delta {rr-0.6496853783350913:+.5f}")
    print(f"  B - A  {result['arms']['delta_B_minus_A']['mean']:+.5f}  "
          f"({result['arms']['delta_B_minus_A']['seeds_positive']}/{len(seeds)} seeds positive)")
    print(f"  B - C  {result['arms']['delta_B_minus_C']['mean']:+.5f}")
    print(f"\nwrote {OUTJSON}  ({result['runtime_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
