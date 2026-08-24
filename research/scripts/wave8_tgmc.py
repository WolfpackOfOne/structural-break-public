"""M5 -- Teacher-Guided Matched Counterfactuals (TGMC). Fold-0 pilot.

Pre-registered in research/WAVE8_FUTURE_AWARE_PREREG.md section 5.5. One
intervention family only: weak persistent location displacement. Base
processes and simulation parameters (shift magnitude, noise) come from
OUTER-TRAINING series only (folds 1-4) -- residual-bootstrapped from each
base series' own real historical segment, never fold 0.

For each sampled base series: two continuations sharing the same bootstrapped
innovation path --
    A (stable):  the path, unmodified, label 0 throughout
    B (broken):  the same path + a fixed-magnitude persistent shift from a
                 random split point onward, label 1 from that point
Both are scored with the SAME production causal feature engine
(wave8_common.synthetic_features) so features are guaranteed causal/legal.
Scored by:
    scorer_B (500 causal cols, trained on REAL outer_train only) -- the
        RT-990-style "current causal model" a pair might already fool or not
    scorer_T (500 + own-final-row broadcast, trained on REAL outer_train
        only) -- the offline future-aware "teacher" (same construction as
        wave7_d3r's Arm C, legitimate here because we KNOW each synthetic
        series' true final row)
Teacher-guided ACCEPTED pairs: scorer_T confidently ranks B > A while
scorer_B is ambiguous or wrong. Random accepted: same COUNT, unfiltered.

Usage:
    python wave8_tgmc.py --fit-scorers
    python wave8_tgmc.py --simulate
    python wave8_tgmc.py --train --arm random     # RT-1051
    python wave8_tgmc.py --train --arm teacher     # RT-1052
    python wave8_tgmc.py --analyze
"""
from __future__ import annotations

import argparse, json, os, time

import numpy as np

import wave8_common as W8
import sbr.pipeline as PL
from wave5_lib import Ctx, FOLDS, OOFDIR, REPORTS, ts_auc_flat
from wave7_d3r import FULL, ARM_B_PARAMS, augmented_stack, cell_mask, last_row_lookup, score_on

OUTER_F = 0
INNER_FOLDS = [g for g in FOLDS if g != OUTER_F]
CACHE = f"{OOFDIR}/wave8_tgmc"
os.makedirs(CACHE, exist_ok=True)

N_BASE_SERIES = 300
SHIFT_SIGMA_MULT = 0.75      # WEAK, fixed, not tuned
POST_T_SAMPLES = 5
CONFIDENT_MARGIN = 0.10      # teacher logit-space-free probability margin threshold
MAX_TRAIN_ROWS = 1_000_000


def cmd_fit_scorers():
    """Two LightGBM boosters trained on REAL outer_train (folds 1-4) rows
    only -- scorer_B (500 causal cols, RT-990-equivalent) and scorer_T (500 +
    own-final-row broadcast, RT-991-equivalent). Kept in memory / pickled for
    reuse scoring both real and synthetic series."""
    import lightgbm as lgb, pickle
    d = PL.Data()
    mats, names = PL.load_features(FULL)
    keep_idx = np.arange(len(names))
    tr_rows = d.rows_for(INNER_FOLDS)
    rng = np.random.default_rng(0)
    if len(tr_rows) > MAX_TRAIN_ROWS:
        tr_rows = np.sort(rng.choice(tr_rows, MAX_TRAIN_ROWS, replace=False))

    p = dict(ARM_B_PARAMS)
    n_round = p.pop("n_estimators")
    t0 = time.time()
    Xb = PL._stack(mats, names, tr_rows, keep_idx)
    ds = lgb.Dataset(Xb, label=d.y[tr_rows], params=dict(p, objective="binary"),
                     feature_name=[f"f{i}" for i in range(Xb.shape[1])])
    booster_B = lgb.train(dict(p), ds, num_boost_round=n_round)
    del Xb, ds
    print(f"scorer_B fit ({time.time()-t0:.1f}s)")

    last_row_by_series = last_row_lookup(d)
    final_of_row = last_row_by_series[d.sidx]
    Xt = augmented_stack(mats, names, keep_idx, tr_rows, final_of_row)
    ds2 = lgb.Dataset(Xt, label=d.y[tr_rows], params=dict(p, objective="binary"),
                      feature_name=[f"f{i}" for i in range(Xt.shape[1])])
    booster_T = lgb.train(dict(p), ds2, num_boost_round=n_round)
    del Xt, ds2
    print(f"scorer_T fit, total {time.time()-t0:.1f}s")

    with open(f"{CACHE}/scorer_B.pkl", "wb") as f:
        pickle.dump(booster_B, f)
    with open(f"{CACHE}/scorer_T.pkl", "wb") as f:
        pickle.dump(booster_T, f)
    print(f"scorers cached -> {CACHE}/")


# ---------------------------------------------------------------------------
def _bootstrap_path(hist, n, rng):
    """Block residual bootstrap from the series' own historical segment --
    preserves marginal scale and short-range dependence without importing a
    parametric noise model. Block length 8 (fixed)."""
    hist = np.asarray(hist, dtype=np.float64)
    bl = 8
    n_blocks = int(np.ceil(n / bl))
    starts = rng.integers(0, max(len(hist) - bl, 1), size=n_blocks)
    out = np.concatenate([hist[s:s + bl] for s in starts])[:n]
    return out.copy()


def cmd_simulate():
    import pickle
    d = PL.Data()
    with open(f"{CACHE}/scorer_B.pkl", "rb") as f:
        booster_B = pickle.load(f)
    with open(f"{CACHE}/scorer_T.pkl", "rb") as f:
        booster_T = pickle.load(f)

    tr_series = np.flatnonzero(np.isin(d.series_fold, INNER_FOLDS))
    rng = np.random.default_rng(0)
    base_series = rng.choice(tr_series, min(N_BASE_SERIES, len(tr_series)), replace=False)

    accepted_teacher, accepted_random_pool, rows_out = [], [], []
    t0 = time.time()
    sid_counter = 0
    series_rows = []   # list of dict: sid, hist, online, y (synthetic label vector)

    for si in base_series:
        h, o, _ = d.st.series(int(si))
        n = len(o)
        if n < 40:
            continue
        sigma = float(np.std(h)) if len(h) else 1.0
        path = _bootstrap_path(h, n, rng)
        split = int(rng.integers(int(0.2 * n), int(0.8 * n)))
        online_A = path.copy()
        online_B = path.copy()
        online_B[split:] += SHIFT_SIGMA_MULT * sigma
        y_A = np.zeros(n, dtype=np.int64)
        y_B = np.zeros(n, dtype=np.int64); y_B[split:] = 1

        Xa, namesA = W8.synthetic_features(h, online_A)
        Xb, namesB = W8.synthetic_features(h, online_B)
        assert namesA == namesB
        pa_B = booster_B.predict(Xa)
        pb_B = booster_B.predict(Xb)
        # own-final-row broadcast for the teacher scorer
        fa = np.tile(Xa[-1], (n, 1)); fb = np.tile(Xb[-1], (n, 1))
        pa_T = booster_T.predict(np.concatenate([Xa, fa], axis=1))
        pb_T = booster_T.predict(np.concatenate([Xb, fb], axis=1))

        post_ts = np.linspace(split, n - 1, min(POST_T_SAMPLES, n - split)).astype(int)
        for t in post_ts:
            m_R = float(pb_B[t] - pa_B[t])
            m_T = float(pb_T[t] - pa_T[t])
            pair = dict(sid=sid_counter, t=int(t), m_R=m_R, m_T=m_T)
            accepted_random_pool.append(pair)
            if m_R <= 0.0 and m_T > CONFIDENT_MARGIN:
                accepted_teacher.append(pair)

        series_rows.append(dict(sid=sid_counter, h=h, oA=online_A, oB=online_B, yA=y_A, yB=y_B))
        sid_counter += 1

    print(f"simulated {sid_counter} base series -> {len(accepted_random_pool)} candidate pairs, "
          f"{len(accepted_teacher)} teacher-confirmed, {time.time()-t0:.1f}s")

    rng.shuffle(accepted_random_pool)
    accepted_random = accepted_random_pool[:max(len(accepted_teacher), 1)]

    with open(f"{CACHE}/pairs.json", "w") as f:
        json.dump({"n_series": sid_counter, "n_candidates": len(accepted_random_pool),
                   "n_teacher_confirmed": len(accepted_teacher),
                   "teacher_confirmed_sids": sorted(set(p["sid"] for p in accepted_teacher)),
                   "random_sample_sids": sorted(set(p["sid"] for p in accepted_random))},
                  f, indent=2)

    import pickle as pk
    with open(f"{CACHE}/series_rows.pkl", "wb") as f:
        pk.dump(series_rows, f)
    print(f"cached synthetic series + pair sets -> {CACHE}/")


# ---------------------------------------------------------------------------
def build_synthetic_training_rows(sids):
    import pickle as pk
    with open(f"{CACHE}/series_rows.pkl", "rb") as f:
        series_rows = pk.load(f)
    by_sid = {r["sid"]: r for r in series_rows}
    Xs, ys = [], []
    for sid in sids:
        r = by_sid.get(sid)
        if r is None:
            continue
        Xa, _ = W8.synthetic_features(r["h"], r["oA"])
        Xb, _ = W8.synthetic_features(r["h"], r["oB"])
        Xs.append(Xa); ys.append(r["yA"])
        Xs.append(Xb); ys.append(r["yB"])
    if not Xs:
        return np.zeros((0, 500), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.concatenate(Xs, axis=0).astype(np.float32), np.concatenate(ys, axis=0)


def cmd_train(arm):
    assert arm in ("random", "teacher")
    with open(f"{CACHE}/pairs.json") as f:
        pairs = json.load(f)
    sids = pairs["teacher_confirmed_sids"] if arm == "teacher" else pairs["random_sample_sids"]
    exp_id = "RT-1052" if arm == "teacher" else "RT-1051"

    d = PL.Data()
    mats, names = PL.load_features(FULL)
    keep_idx = np.arange(len(names))
    W8.assert_no_forbidden_columns(names)
    tr_rows_real = d.rows_for(INNER_FOLDS)
    va_rows = d.rows_for([OUTER_F])
    rng = np.random.default_rng(0)
    if len(tr_rows_real) > MAX_TRAIN_ROWS:
        tr_rows_real = np.sort(rng.choice(tr_rows_real, MAX_TRAIN_ROWS, replace=False))

    t0 = time.time()
    Xreal = PL._stack(mats, names, tr_rows_real, keep_idx)
    yreal = d.y[tr_rows_real]
    Xsyn, ysyn = build_synthetic_training_rows(sids)
    print(f"[TGMC-{arm}] {len(tr_rows_real)} real rows + {len(ysyn)} synthetic rows "
          f"from {len(sids)} series ({time.time()-t0:.1f}s)")

    import lightgbm as lgb
    p = dict(ARM_B_PARAMS)
    n_round = p.pop("n_estimators")
    Xtr = np.concatenate([Xreal, Xsyn], axis=0)
    ytr = np.concatenate([yreal, ysyn], axis=0)
    ds = lgb.Dataset(Xtr, label=ytr, params=dict(p, objective="binary"),
                     feature_name=[f"f{i}" for i in range(Xtr.shape[1])])
    booster = lgb.train(dict(p), ds, num_boost_round=n_round)
    del Xtr, ds, Xreal, Xsyn

    Xva = PL._stack(mats, names, va_rows, keep_idx)
    pred = booster.predict(Xva).astype(np.float32)
    del Xva
    s = float(ts_auc_flat(pred, d.y[va_rows], d.t[va_rows]))
    oof = np.full(len(d.y), np.nan, dtype=np.float32)
    oof[va_rows] = pred
    np.save(f"{OOFDIR}/{exp_id}.npy", oof)
    print(f"[TGMC-{arm}] TS-AUC (real fold0 only) {s:.5f}  runtime {time.time()-t0:.1f}s -> {exp_id}")

    res = dict(experiment_id=exp_id, date=time.strftime("%Y-%m-%d %H:%M"), git_sha=PL.git_sha(),
               agent="agent0",
               hypothesis=f"TGMC-{'T' if arm=='teacher' else 'R'}: real outer-train rows + "
                          f"{'teacher-guided' if arm=='teacher' else 'random'} matched synthetic "
                          f"weak-persistent-location-shift pairs ({len(sids)} series). Evaluated "
                          f"on REAL fold-0 rows only.",
               falsification_condition="see research/WAVE8_FUTURE_AWARE_PREREG.md section 5.5",
               feature_set=",".join(FULL) + f"+{len(ysyn)}_synthetic_rows", n_features=500,
               model="lgbm", objective="binary", folds=str(OUTER_F), random_seed=0,
               train_series=None, train_rows=int(len(tr_rows_real) + len(ysyn)),
               mean_oof_ts_auc=s, pooled_oof_ts_auc=s, per_fold_ts_auc=f"{s:.5f}", fold_std=0.0,
               persistence="none", sample_mode="uniform", training_runtime_s=round(time.time() - t0, 1),
               causal_verified="synthetic features computed via the production feature engine "
                               "(wave8_common.synthetic_features); simulator fit on outer_train "
                               "only; evaluated on real fold0 only",
               test_reduced_touched="no", lockbox_touched="no", status="recorded",
               notes="Wave 8 TGMC pilot. Pre-registered research/WAVE8_FUTURE_AWARE_PREREG.md.",
               protocol="pilot_fold0_only")
    PL.append_result(res)


# ---------------------------------------------------------------------------
def analyze():
    c = Ctx()
    y, t, age = c.d.y, c.d.t, c.age
    r0 = c.rows[OUTER_F]
    cellmask0 = cell_mask(y[r0], t[r0], age[r0])
    A = np.load(f"{OOFDIR}/RT-990.npy")
    R = np.load(f"{OOFDIR}/RT-1051.npy")
    Tt = np.load(f"{OOFDIR}/RT-1052.npy")

    def whole(v):
        return score_on(v[r0], np.ones(len(r0), bool), y[r0], t[r0])

    def cell(v):
        rr = r0[cellmask0]
        return score_on(v[rr], np.ones(len(rr), bool), y[rr], t[rr])

    with open(f"{CACHE}/pairs.json") as f:
        pairs = json.load(f)

    ens = W8.ensemble_marginal(Tt, c=c, fold=OUTER_F, label="tgmc_t")
    out = {
        "n_candidate_pairs": pairs["n_candidates"], "n_teacher_confirmed": pairs["n_teacher_confirmed"],
        "control_A_whole": whole(A), "control_A_cell": cell(A),
        "TGMC_R_whole": whole(R), "TGMC_R_cell": cell(R),
        "TGMC_T_whole": whole(Tt), "TGMC_T_cell": cell(Tt),
        "T_minus_R_whole": whole(Tt) - whole(R), "T_minus_R_cell": cell(Tt) - cell(A) if cell(A) else None,
        "ensemble": ens,
    }
    gate_a = out["TGMC_T_whole"] > out["TGMC_R_whole"]
    gate_b = ens["marginal_vs_clone"] >= 0.0015
    out["continuation_gate_cleared"] = bool(gate_a and gate_b)
    print(json.dumps(out, indent=2))
    with open(f"{REPORTS}/wave8_tgmc.json", "w") as f:
        json.dump(out, f, indent=2)
    md = ["# WAVE 8 -- TGMC PILOT RESULT (fold 0)\n",
          f"Candidate synthetic pairs: {pairs['n_candidates']}, teacher-confirmed: "
          f"{pairs['n_teacher_confirmed']}\n",
          "| arm | whole fold0 (real) | dominant cell (real) |", "|---|---:|---:|",
          f"| control (`RT-990`) | {out['control_A_whole']:.5f} | {out['control_A_cell']:.5f} |",
          f"| TGMC-R (`RT-1051`) | {out['TGMC_R_whole']:.5f} | {out['TGMC_R_cell']:.5f} |",
          f"| TGMC-T (`RT-1052`) | {out['TGMC_T_whole']:.5f} | {out['TGMC_T_cell']:.5f} |",
          f"\n**T − R (whole):** {out['T_minus_R_whole']:+.5f}\n",
          "\n## Ensemble marginal (fold 0)\n", "```json", json.dumps(ens, indent=2), "```",
          f"\n## CONTINUATION GATE: {'CLEARED' if out['continuation_gate_cleared'] else 'NOT CLEARED'}\n"]
    with open(f"{REPORTS}/wave8_tgmc.md", "w") as f:
        f.write("\n".join(md))
    print(f"wrote {REPORTS}/wave8_tgmc.{{md,json}}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-scorers", action="store_true")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--arm", choices=["random", "teacher"])
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.fit_scorers:
        cmd_fit_scorers()
    elif args.simulate:
        cmd_simulate()
    elif args.train:
        cmd_train(args.arm)
    elif args.analyze:
        analyze()
    else:
        raise SystemExit("pass --fit-scorers, --simulate, --train --arm random|teacher, or --analyze")


if __name__ == "__main__":
    main()
