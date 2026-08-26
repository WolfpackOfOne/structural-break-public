"""Release gates run FROM the packaged deployment artifact, not from src/.

Carried onto release/rt600-reliability-2026 from
engineering/rt600-final-reliability-2026 (d4793771) as RELEASE-VALIDATION
tooling. It is not a runtime requirement: nothing in the shipped payload
imports it, and it is not packed into the source zip. ONE line differs from the
engineering copy -- ``parity_population`` reads the audited 64-series sample
from ``PARITY_SAMPLE.json`` beside this file instead of from
``engineering/reports/rt600_final_reliability/FULL_PARITY.json``, which does not
exist on this lineage and whose remaining contents are the engineering run's own
results. The series identities are identical, so the battery is the same one.

The source tree passing is not evidence that the shipped notebook passes: the
notebook ships a frozen zip of ``src/sbr`` and unpacks it at run time, so every
gate that matters has to be measured through that unpacked copy.  This script
executes the notebook's own code cells in a scratch directory with this
repository removed from ``sys.path``, asserts that ``import sbr`` resolves
inside the payload, and then runs one gate against it.

    OUT=<dir> python packaged_release_check.py --nb <notebook> --phase parity

Phases are separate processes on purpose: each one re-extracts the payload, so
a phase cannot contaminate the next through module state or numba caches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time

import numpy as np

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def load_payload(nb_path):
    """Execute the notebook's code cells with this repo off sys.path.

    Returns
    -------
    tuple
        ``(globals_dict, payload_dir, work_dir)``.
    """
    nb = json.load(open(nb_path))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    work = tempfile.mkdtemp(prefix="rc_")
    os.chdir(work)
    clean = [p for p in sys.path
             if os.path.abspath(p) != ROOT and not os.path.abspath(p).startswith(ROOT + "/")]
    for m in [m for m in list(sys.modules) if m == "sbr" or m.startswith("sbr.")]:
        del sys.modules[m]
    sys.path[:] = clean
    g = {"__name__": "__main__"}
    for i, src in enumerate(code):
        exec(compile(src, f"<cell {i}>", "exec"), g)
    import sbr as _s
    pdir = os.path.abspath(os.path.dirname(_s.__file__))
    if pdir == ROOT or pdir.startswith(ROOT + "/"):
        raise SystemExit(f"NOT ISOLATED: notebook imported the repo tree at {pdir}")
    return g, pdir, work


def load_store_only():
    """Load the store through the PAYLOAD's own reader."""
    from sbr.store import load_store
    return load_store()


def store_and_modules():
    """Load the store and the 7 frozen production modules from the PAYLOAD.

    The pre-fix artifact predates the MODULE_ORDER / PRODUCTION_MODULES split,
    where the frozen seven were simply MODULE_ORDER; fall back to that so the
    same battery can be pointed at either artifact.
    """
    from sbr.store import load_store
    from sbr.features.base import REGISTRY, load_all, make_ctx
    import sbr.stream.engine as _eng
    load_all()
    mods = getattr(_eng, "PRODUCTION_MODULES", None) or _eng.MODULE_ORDER
    return (load_store(), REGISTRY, make_ctx, list(mods), _eng.StreamEngine)


def parity_population(st, n_total):
    """The audited 64-series sample, extended deterministically to n_total."""
    audited = json.load(open(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "PARITY_SAMPLE.json")))["sample"]
    n_on = st.meta.n_online.to_numpy()
    pool = np.flatnonzero(n_on >= 50)
    rest = np.setdiff1d(pool, np.asarray(audited))
    extra = np.random.default_rng(0).choice(
        rest, min(n_total - len(audited), len(rest)), replace=False)
    return [int(x) for x in audited] + sorted(int(x) for x in extra)


def eq(a, b):
    """Bitwise-equal with NaN treated as equal to NaN."""
    return (a == b) | (np.isnan(a) & np.isnan(b))


def phase_parity(g, args):
    st, REGISTRY, make_ctx, MODS, StreamEngine = store_and_modules()
    sample = parity_population(st, args.n_parity)
    per = {m: {"cells": 0, "bad": 0, "bad_series": []} for m in MODS}
    t0 = time.time()
    for k, si in enumerate(sample):
        h, o, _ = st.series(si)
        e = StreamEngine().fit_historical(h)
        S = np.vstack([e.step(x) for x in o])
        ctx = make_ctx(h, o)
        col = 0
        for m in MODS:
            names, vals = REGISTRY[m].fn(ctx)
            B = np.asarray(vals)
            w = B.shape[1]
            good = eq(B, S[:, col:col + w])
            nbad = int((~good).sum())
            per[m]["cells"] += int(good.size)
            per[m]["bad"] += nbad
            if nbad:
                bc = sorted({names[j] for j in np.unique(np.flatnonzero(~good) % w)})
                per[m]["bad_series"].append({"series": si, "n_bad": nbad, "cols": bc})
            col += w
        if (k + 1) % 20 == 0:
            tot = sum(v["cells"] for v in per.values())
            bad = sum(v["bad"] for v in per.values())
            print(f"  {k+1}/{len(sample)}  {tot:,} cells  {bad} mismatches  "
                  f"{time.time()-t0:.0f}s", flush=True)
    total_cells = sum(v["cells"] for v in per.values())
    total_bad = sum(v["bad"] for v in per.values())
    for m in MODS:
        per[m]["series"] = len(sample)
    return {"n_series": len(sample), "modules": MODS, "per_module": per,
            "total_cells": total_cells, "total_mismatches": total_bad,
            "sample": sample, "wall_s": time.time() - t0,
            "PASS": total_bad == 0}


PREFIXES = [1, 2, 3, 5, 10, 16, 20, 32, 50, 64, 100, 128, 200, 256]


def phase_prefix(g, args):
    st, REGISTRY, make_ctx, MODS, StreamEngine = store_and_modules()
    n_on = st.meta.n_online.to_numpy()
    cand = np.flatnonzero((n_on >= 200) & (n_on <= 320))
    sample = [int(x) for x in cand[:14]]
    cells = bad = 0
    detail = []
    for si in sample:
        h, o, _ = st.series(si)
        n = len(o)
        full = np.vstack([StreamEngine().fit_historical(h).step(x) for x in o]) \
            if False else None
        e = StreamEngine().fit_historical(h)
        full = np.vstack([e.step(x) for x in o])
        for L in PREFIXES:
            if L > n:
                continue
            e2 = StreamEngine().fit_historical(h)
            pre = np.vstack([e2.step(x) for x in o[:L]])
            good = eq(pre, full[:L])
            cells += int(good.size)
            b = int((~good).sum())
            bad += b
            if b:
                detail.append({"series": si, "prefix": L, "n_bad": b,
                               "mode": "stream_truncated"})
        for L in (10, 50, 200):
            if L > n:
                continue
            ct = make_ctx(h, o[:L])
            Bt = np.hstack([np.asarray(REGISTRY[m].fn(ct)[1]) for m in MODS])
            good = eq(Bt, full[:L])
            cells += int(good.size)
            b = int((~good).sum())
            bad += b
            if b:
                detail.append({"series": si, "prefix": L, "n_bad": b,
                               "mode": "batch_recomputed_on_truncated_online"})
        print(f"  {si}: cumulative {cells:,} cells, {bad} mismatches", flush=True)
    return {"n_series": len(sample), "prefixes": PREFIXES, "cells": cells,
            "mismatches": bad, "detail": detail, "PASS": bad == 0}


def phase_indep(g, args):
    """Series independence at feature level and at prediction level."""
    st, REGISTRY, make_ctx, MODS, StreamEngine = store_and_modules()
    n_on = st.meta.n_online.to_numpy()
    cand = np.flatnonzero((n_on >= 200) & (n_on <= 320))
    tgt = [int(x) for x in cand[:8]]
    others = [int(x) for x in cand[20:24]]
    ref = {}
    for si in tgt:
        h, o, _ = st.series(si)
        e = StreamEngine().fit_historical(h)
        ref[si] = np.vstack([e.step(x) for x in o])
    cells = bad = 0
    for si in reversed(tgt):
        for oj in others:
            hh, oo, _ = st.series(oj)
            e = StreamEngine().fit_historical(hh)
            for x in oo[:40]:
                e.step(x)
        h, o, _ = st.series(si)
        e = StreamEngine().fit_historical(h)
        got = np.vstack([e.step(x) for x in o])
        good = eq(got, ref[si])
        cells += int(good.size)
        bad += int((~good).sum())
    feat = {"n_series": len(tgt), "cells": cells, "mismatches": bad}

    # prediction level: reorder + interleave foreign series through infer()
    sys.path.insert(0, f"{ROOT}/research/scripts")
    from local_runner import run_infer
    mdir = os.path.join(os.getcwd(), "model_out")
    os.makedirs(mdir, exist_ok=True)
    g["train"]([], mdir)
    series = [(st.series(i)[0], st.series(i)[1]) for i in tgt]
    base = run_infer(g["infer"], series, mdir)
    order = list(range(len(tgt)))[::-1]
    foreign = [(st.series(i)[0], st.series(i)[1][:40]) for i in others]
    mixed, pos = [], {}
    for k, i in enumerate(order):
        mixed.append(foreign[k % len(foreign)])
        pos[i] = len(mixed)
        mixed.append(series[i])
    got = run_infer(g["infer"], mixed, mdir)
    npred = sum(len(s) for s in base)
    mism = sum(int((base[i] != got[pos[i]]).sum()) for i in order)
    pred = {"n_predictions": npred, "mismatches": mism}
    return {"features": feat, "predictions": pred,
            "PASS": bad == 0 and mism == 0}


def _series_sample(st, n, seed=0):
    n_on = st.meta.n_online.to_numpy()
    pool = np.flatnonzero(n_on >= 50)
    return [int(x) for x in np.random.default_rng(seed).choice(pool, n, replace=False)]


def phase_contract(g, args):
    """Determinism, output contract and the full local Crunch-contract battery."""
    st, REGISTRY, make_ctx, MODS, StreamEngine = store_and_modules()
    sys.path.insert(0, f"{ROOT}/research/scripts")
    from local_runner import check_all, run_infer
    mdir = os.path.join(os.getcwd(), "model_out")
    os.makedirs(mdir, exist_ok=True)
    g["train"]([], mdir)
    print("train() installed:", sorted(os.listdir(mdir)), flush=True)
    idx = _series_sample(st, args.n_contract, seed=0)
    series = [(st.series(i)[0], st.series(i)[1]) for i in idx]
    # labels deliberately withheld: no TS-AUC of any kind is computed here
    res = check_all(g["infer"], series, mdir, labels=None, verbose=False)
    hashes = set()
    for _ in range(args.repeats):
        out = run_infer(g["infer"], series, mdir)
        h = hashlib.sha256()
        for s in out:
            h.update(np.asarray(s, dtype=np.float64).tobytes())
        hashes.add(h.hexdigest()[:16])
    flat = np.concatenate([np.asarray(s) for s in run_infer(g["infer"], series, mdir)])
    res["deterministic_replay"] = {"repeats": args.repeats,
                                   "distinct_hashes": len(hashes),
                                   "hash": sorted(hashes)[0]}
    res["output_contract"] = {
        "n": int(flat.size),
        "expected_n": int(sum(len(o) for _, o in series)),
        "row_count_matches": int(flat.size) == int(sum(len(o) for _, o in series)),
        "nan": int(np.isnan(flat).sum()), "inf": int(np.isinf(flat).sum()),
        "outside_0_1": int(((flat < 0) | (flat > 1)).sum()),
        "min": float(flat.min()), "max": float(flat.max()),
        "n_series": len(series),
        "per_series_rows_correct": all(len(a) == len(o) for a, (_, o)
                                       in zip(run_infer(g["infer"], series, mdir), series)),
    }
    res["series_ids"] = idx
    res["PASS"] = bool(res["PASS"] and len(hashes) == 1
                       and res["output_contract"]["nan"] == 0
                       and res["output_contract"]["inf"] == 0
                       and res["output_contract"]["outside_0_1"] == 0
                       and res["output_contract"]["row_count_matches"])
    return res


BENCH = {"hist~1000_online~1000": (800, 1200, 800, 1200),
         "hist~5000_online~1000": (4500, 5200, 800, 1200),
         "hist~3000_online~500": (2700, 3300, 400, 600),
         "short_online_10_20": (0, 10**9, 10, 20)}


def phase_runtime(g, args):
    """Benchmark the packaged artifact and fit marginal + fixed cost per series."""
    import resource
    st = load_store_only()
    sys.path.insert(0, f"{ROOT}/research/scripts")
    from local_runner import run_infer
    mdir = os.path.join(os.getcwd(), "model_out")
    os.makedirs(mdir, exist_ok=True)
    g["train"]([], mdir)
    nh = st.meta.n_hist.to_numpy()
    no = st.meta.n_online.to_numpy()
    # macOS reports ru_maxrss in BYTES (Linux in KiB); normalise to MB.
    def _rss_mb():
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / (1048576.0 if sys.platform == "darwin" else 1024.0)

    warm = int(np.flatnonzero(no >= 100)[0])
    run_infer(g["infer"], [(st.series(warm)[0], st.series(warm)[1])], mdir)
    rss0 = _rss_mb()
    bench, pairs = {}, []
    for name, (h0, h1, o0, o1) in BENCH.items():
        idx = np.flatnonzero((nh >= h0) & (nh <= h1) & (no >= o0) & (no <= o1))[:3]
        if not len(idx):
            continue
        tot_w = tot_p = 0
        for i in idx:
            h, o, _ = st.series(int(i))
            t0 = time.time()
            out = run_infer(g["infer"], [(h, o)], mdir)
            w = time.time() - t0
            pairs.append((len(o), w))
            tot_w += w
            tot_p += len(out[0])
        bench[name] = {"n_series": len(idx), "n_points": tot_p, "wall_s": tot_w,
                       "s_per_series": tot_w / len(idx),
                       "ms_per_point": 1000 * tot_w / tot_p}
        print(f"  {name}: {bench[name]['ms_per_point']:.3f} ms/pt", flush=True)
    X = np.array([[n, 1.0] for n, _ in pairs])
    y = np.array([t for _, t in pairs])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    ms_pt, fixed_ms = float(coef[0] * 1000), float(coef[1] * 1000)
    rss1 = _rss_mb()
    mean_online = float(no.mean())
    proj_h = (ms_pt * mean_online + fixed_ms) * 10000 / 1000 / 3600
    return {"benchmark": bench, "n_series_timed": len(pairs),
            "marginal_ms_per_point": ms_pt, "fixed_ms_per_series": fixed_ms,
            "mean_online_len": mean_online, "projected_10k_hours": proj_h,
            "peak_rss_mb": rss1, "rss_growth_mb": rss1 - rss0}


DIVERGENT = [2575, 4746, 5002, 5111, 7395, 8581]


def phase_predict(g, args):
    """Dump predictions for the equivalence battery. No TS-AUC is computed."""
    st = load_store_only()
    sys.path.insert(0, f"{ROOT}/research/scripts")
    from local_runner import run_infer
    mdir = os.path.join(os.getcwd(), "model_out")
    os.makedirs(mdir, exist_ok=True)
    g["train"]([], mdir)
    n_on = st.meta.n_online.to_numpy()
    pool = np.setdiff1d(np.flatnonzero(n_on >= 50), np.asarray(DIVERGENT))
    extra = np.random.default_rng(7).choice(pool, args.n_predict - len(DIVERGENT),
                                            replace=False)
    idx = DIVERGENT + sorted(int(x) for x in extra)
    series = [(st.series(i)[0], st.series(i)[1]) for i in idx]
    out = run_infer(g["infer"], series, mdir)
    np.savez_compressed(args.preds_out, ids=np.array(idx),
                        lens=np.array([len(s) for s in out]),
                        scores=np.concatenate([np.asarray(s, dtype=np.float64)
                                               for s in out]))
    return {"n_series": len(idx), "n_predictions": int(sum(len(s) for s in out)),
            "series_ids": idx, "preds": os.path.abspath(args.preds_out)}


PHASES = {"parity": phase_parity, "prefix": phase_prefix, "indep": phase_indep,
          "contract": phase_contract, "runtime": phase_runtime,
          "predict": phase_predict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", required=True)
    ap.add_argument("--phase", required=True, choices=sorted(PHASES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--preds-out", default=None)
    ap.add_argument("--n-parity", type=int, default=201)
    ap.add_argument("--n-contract", type=int, default=30)
    ap.add_argument("--n-predict", type=int, default=67)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    nb = os.path.abspath(a.nb)
    out = os.path.abspath(a.out) if a.out else None
    if a.preds_out:
        a.preds_out = os.path.abspath(a.preds_out)
    nb_sha = hashlib.sha256(open(nb, "rb").read()).hexdigest()
    g, pdir, work = load_payload(nb)
    print(f"payload isolation OK: {pdir}", flush=True)
    res = PHASES[a.phase](g, a)
    res["_meta"] = {"phase": a.phase, "notebook": nb, "notebook_sha256": nb_sha,
                    "payload_dir": pdir, "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                time.gmtime())}
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("sample", "series_ids", "detail")}, indent=2,
                     default=str))
    if out:
        json.dump(res, open(out, "w"), indent=2, default=str)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
