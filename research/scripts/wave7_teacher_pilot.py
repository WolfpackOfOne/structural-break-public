"""W7 teacher/distillation pilot: can a strictly causal LightGBM student
recover some of W7-D3R Arm C's future-information advantage from a
privileged training-time teacher target?

Pre-registered in research/WAVE7_TEACHER_PREREG.md BEFORE any diagnostic or
student score exists. Population, teacher construction, student arms and the
continuation gate are fixed there and are not re-derived here.

Teacher: Q = clip(research/oof/RT-991.npy, eps, 1-eps) -- W7-D3R Arm C's own
OOF score, reused (not retrained). Already series-level cross-fitted.

Students (fold 0 only, same 500-column causal bank, same rows, same capacity
as RT-990, LightGBM's built-in xentropy objective):
    T0 -- RT-990, reused (hard-label control, no retraining needed)
    T1 -- RT-992, new: label = Q
    T2 -- RT-993, new: label = 0.5*y + 0.5*Q

Usage:
    python wave7_teacher_pilot.py --diagnostics
    python wave7_teacher_pilot.py --parity-check
    python wave7_teacher_pilot.py --train t1
    python wave7_teacher_pilot.py --train t2
    python wave7_teacher_pilot.py --analyze
"""
from __future__ import annotations

import argparse, json, time

import numpy as np

from wave5_lib import Ctx, FOLDS, REPORTS, OOFDIR, ts_auc_flat
import sbr.pipeline as PL
from wave7_d3r import FULL, ARM_B_PARAMS, cell_mask, score_on, bucket

EPS = 1e-6
MAX_TRAIN_ROWS = 1_000_000
VAL_FOLD = 0
TRAIN_FOLDS = [1, 2, 3, 4]


def load_Q():
    Q = np.load(f"{OOFDIR}/RT-991.npy").astype(np.float64)
    return np.clip(Q, EPS, 1 - EPS)


# ---------------------------------------------------------------------------
def diagnostics():
    c = Ctx()
    y, t, sidx = c.d.y, c.d.t, c.d.sidx
    age = c.age
    Q = load_Q()

    dev_mask = np.zeros(len(y), dtype=bool)
    dev_mask[c.dev] = True
    cellmask = cell_mask(y, t, age) & dev_mask
    neg_is_prebreak = c.has_break[sidx] & (y == 0)

    def stats(mask):
        v = Q[mask]
        if len(v) == 0:
            return None
        qs = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
        return {"n": int(len(v)), "mean": float(v.mean()), "std": float(v.std()),
                "q05": float(qs[0]), "q25": float(qs[1]), "q50": float(qs[2]),
                "q75": float(qs[3]), "q95": float(qs[4])}

    groups = {
        "dev_all": dev_mask,
        "dev_pos": dev_mask & (y == 1),
        "dev_neg": dev_mask & (y == 0),
        "cell_all": cellmask,
        "cell_pos": cellmask & (y == 1),
        "cell_neg": cellmask & (y == 0),
        "outside_cell_pos": dev_mask & (y == 1) & ~cellmask,
        "outside_cell_neg": dev_mask & (y == 0) & ~cellmask,
    }
    group_stats = {k: stats(m) for k, m in groups.items()}

    def corr(mask, other):
        v, o = Q[mask], other[mask]
        if len(v) < 2 or v.std() == 0 or o.std() == 0:
            return None
        return float(np.corrcoef(v, o)[0, 1])

    correlations = {
        "corr_Q_y_dev": corr(dev_mask, y.astype(float)),
        "corr_Q_t_dev": corr(dev_mask, t.astype(float)),
        "corr_Q_age_positives": corr(dev_mask & (y == 1), age.astype(float)),
        "corr_Q_tau_positives": corr(dev_mask & (y == 1), c.tau[sidx].astype(float)),
        "corr_Q_n_online_dev": corr(dev_mask, c.n_online[sidx].astype(float)),
        "corr_Q_has_break_negatives": corr(dev_mask & (y == 0), neg_is_prebreak.astype(float)),
    }

    same_t_auc = {
        "pooled_dev": float(ts_auc_flat(Q[c.dev], y[c.dev], t[c.dev])),
        "dominant_cell": score_on(Q, cellmask, y, t),
    }

    pos_std = group_stats["dev_pos"]["std"] if group_stats["dev_pos"] else 0.0
    corr_y = correlations["corr_Q_y_dev"] or 0.0
    stop_and_redesign = abs(corr_y) > 0.98 and pos_std < 0.02

    out = {
        "teacher_source": "research/oof/RT-991.npy (W7-D3R Arm C, reused, not retrained)",
        "group_stats": group_stats,
        "correlations": correlations,
        "same_t_auc": same_t_auc,
        "stop_and_redesign_condition_met": bool(stop_and_redesign),
    }
    with open(f"{REPORTS}/wave7_teacher_diagnostics.json", "w") as f:
        json.dump(out, f, indent=2)

    md = ["# WAVE 7 — TEACHER TARGET DIAGNOSTICS (Q = RT-991, reused)\n",
          "Computed before any student score exists, per "
          "`research/WAVE7_TEACHER_PREREG.md` §2.\n",
          "## Same-t discriminatory power of Q itself\n",
          f"pooled dev TS-AUC: **{same_t_auc['pooled_dev']:.5f}**  "
          f"(cross-check vs W7-D3R Arm C pooled 0.71989)\n",
          f"dominant-cell TS-AUC: **{same_t_auc['dominant_cell']:.5f}**  "
          f"(cross-check vs W7-D3R Arm C cell 0.71859)\n",
          "## Group stats\n",
          "| group | n | mean | std | q05 | q25 | q50 | q75 | q95 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, s in group_stats.items():
        if s is None:
            continue
        md.append(f"| {k} | {s['n']} | {s['mean']:.4f} | {s['std']:.4f} | "
                  f"{s['q05']:.4f} | {s['q25']:.4f} | {s['q50']:.4f} | "
                  f"{s['q75']:.4f} | {s['q95']:.4f} |")
    md += ["\n## Correlations\n", "| pair | r |", "|---|---:|"]
    for k, v in correlations.items():
        md.append(f"| {k} | {v:.4f} |" if v is not None else f"| {k} | n/a |")
    md += [f"\n## Stop-and-redesign condition met: **{stop_and_redesign}**\n"]
    with open(f"{REPORTS}/wave7_teacher_diagnostics.md", "w") as f:
        f.write("\n".join(md))
    print(json.dumps(out, indent=2)[:4000])
    print(f"\nwrote {REPORTS}/wave7_teacher_diagnostics.{{json,md}}")
    print(f"STOP_AND_REDESIGN = {stop_and_redesign}")
    return out


# ---------------------------------------------------------------------------
def parity_check():
    d = PL.Data()
    mats, names = PL.load_features(FULL)
    keep_idx = np.arange(len(names))

    tr_rows_full = d.rows_for(TRAIN_FOLDS)
    va_rows = d.rows_for([VAL_FOLD])
    rng = np.random.default_rng(1)
    tr_rows = np.sort(rng.choice(tr_rows_full, min(100_000, len(tr_rows_full)), replace=False))

    Xtr = PL._stack(mats, names, tr_rows, keep_idx)
    ytr = d.y[tr_rows].astype(np.float64)
    Xva = PL._stack(mats, names, va_rows, keep_idx)

    import lightgbm as lgb
    base = dict(ARM_B_PARAMS)
    base.pop("objective")
    n_round = 150
    preds = {}
    for obj in ("binary", "xentropy"):
        p = dict(base, objective=obj)
        ds = lgb.Dataset(Xtr, label=ytr, params=dict(p),
                         feature_name=[f"f{i}" for i in range(Xtr.shape[1])])
        booster = lgb.train(p, ds, num_boost_round=n_round)
        preds[obj] = booster.predict(Xva).astype(np.float64)

    corr = float(np.corrcoef(preds["binary"], preds["xentropy"])[0, 1])
    auc_binary = float(ts_auc_flat(preds["binary"], d.y[va_rows], d.t[va_rows]))
    auc_xentropy = float(ts_auc_flat(preds["xentropy"], d.y[va_rows], d.t[va_rows]))
    auc_diff = abs(auc_binary - auc_xentropy)
    passed = corr >= 0.999 and auc_diff <= 0.0005

    out = {"corr_binary_xentropy": corr, "auc_binary": auc_binary,
           "auc_xentropy": auc_xentropy, "auc_diff": auc_diff, "passed": passed}
    print(json.dumps(out, indent=2))
    with open(f"{REPORTS}/wave7_teacher_parity_check.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"PARITY_CHECK_PASSED = {passed}")
    if not passed:
        raise SystemExit("xentropy/binary hard-label parity check FAILED -- "
                         "per prereg §3.1, T1/T2 must not run under this "
                         "pre-registration until this is resolved.")
    return out


# ---------------------------------------------------------------------------
def train_student(kind):
    assert kind in ("t1", "t2")
    exp_id = "RT-992" if kind == "t1" else "RT-993"
    d = PL.Data()
    mats, names = PL.load_features(FULL)
    keep_idx = np.arange(len(names))

    rng = np.random.default_rng(0)
    tr_rows = d.rows_for(TRAIN_FOLDS)
    va_rows = d.rows_for([VAL_FOLD])
    if len(tr_rows) > MAX_TRAIN_ROWS:
        tr_rows = np.sort(rng.choice(tr_rows, MAX_TRAIN_ROWS, replace=False))

    Q = load_Q()
    y_hard = d.y.astype(np.float64)
    if kind == "t1":
        ytr = Q[tr_rows]
        hyp = "W7 teacher pilot T1: pure distillation, label = Q (RT-991 OOF, reused)"
    else:
        ytr = 0.5 * y_hard[tr_rows] + 0.5 * Q[tr_rows]
        hyp = "W7 teacher pilot T2: hard+teacher fixed 0.5/0.5 blend, label = 0.5*y + 0.5*Q"

    t_start = time.time()
    Xtr = PL._stack(mats, names, tr_rows, keep_idx)
    assert Xtr.shape[1] == 500, f"expected 500 causal columns, got {Xtr.shape[1]}"

    p = dict(ARM_B_PARAMS)
    p["objective"] = "xentropy"
    n_round = p.pop("n_estimators")

    import lightgbm as lgb
    ds = lgb.Dataset(Xtr, label=ytr, params=dict(p),
                     feature_name=[f"f{i}" for i in range(Xtr.shape[1])])
    booster = lgb.train(p, ds, num_boost_round=n_round)
    del Xtr, ds

    Xva = PL._stack(mats, names, va_rows, keep_idx)
    assert Xva.shape[1] == 500
    pred = booster.predict(Xva).astype(np.float32)
    del Xva

    oof = np.full(len(d.y), np.nan, dtype=np.float32)
    oof[va_rows] = pred
    s = float(ts_auc_flat(pred, d.y[va_rows], d.t[va_rows]))
    np.save(f"{OOFDIR}/{exp_id}.npy", oof)

    res = dict(
        experiment_id=exp_id, date=time.strftime("%Y-%m-%d %H:%M"), git_sha=PL.git_sha(),
        agent="agent0", hypothesis=hyp,
        falsification_condition="see research/WAVE7_TEACHER_PREREG.md §6",
        feature_set=",".join(FULL), n_features=500, model="lgbm", objective="xentropy",
        folds=str(VAL_FOLD), random_seed=0,
        train_series=int((~np.isin(d.series_fold, [-1])).sum()), train_rows=int(len(tr_rows)),
        mean_oof_ts_auc=s, pooled_oof_ts_auc=s, per_fold_ts_auc=f"{s:.5f}", fold_std=0.0,
        persistence="none", sample_mode="uniform",
        training_runtime_s=round(time.time() - t_start, 1),
        causal_verified="student inputs = unmodified 500-col causal bank; teacher (Q) used "
                        "only as training label, never as a feature",
        test_reduced_touched="no", lockbox_touched="no", status="recorded",
        notes="W7 teacher pilot. Pre-registered research/WAVE7_TEACHER_PREREG.md. "
              "ONE-FOLD PILOT (fold 0 only) -- not a full 5-fold result.",
        protocol="pilot_fold0_only",
    )
    PL.append_result(res)
    print(f"  [{kind.upper()}] fold {VAL_FOLD}: TS-AUC {s:.5f}  "
          f"({len(tr_rows)} train rows, {len(va_rows)} valid rows)")
    print(json.dumps({k: res[k] for k in
                      ("experiment_id", "pooled_oof_ts_auc", "training_runtime_s")}, indent=2))
    return res


# ---------------------------------------------------------------------------
# DEPRECATED -- DO NOT USE. train_student_full()/analyze_full() reuse the
# global RT-991 OOF as Q, which is outer-fold contaminated (see
# research/WAVE7_TEACHER_NESTED_PREREG.md §0): the teacher for any given
# target fold was trained on every OTHER fold, including whatever fold a
# student run later treats as its held-out outer fold. Started once under
# the old scheme (--train-full t1/t2) and killed before any fold completed
# once this was found -- no RT-994/RT-995 artifact from this path exists.
# The corrected, outer-fold-pure replacement is
# research/scripts/wave7_teacher_nested.py. Left here, unused, only so the
# contaminated construction is legible in the historical record.
def train_student_full(kind):
    """Full 5-fold promotion-track run of T1/T2, per WAVE7_TEACHER_PREREG.md §6
    ("Continue to a full 5-fold run (new IDs, not RT-992/RT-993 re-used)").

    Row sampling mirrors sbr.pipeline.run()/wave7_d3r.train_arm_c(): ONE rng
    created before the fold loop, advancing per fold in order (0,1,2,3,4) --
    this reproduces RT-990's exact per-fold training-row sample bit-for-bit,
    since RT-990 was trained the same way (same MAX_TRAIN_ROWS, same seed).
    """
    assert kind in ("t1", "t2")
    exp_id = "RT-994" if kind == "t1" else "RT-995"
    d = PL.Data()
    mats, names = PL.load_features(FULL)
    keep_idx = np.arange(len(names))
    Q = load_Q()
    y_hard = d.y.astype(np.float64)

    rng = np.random.default_rng(0)
    oof = np.full(len(d.y), np.nan, dtype=np.float32)
    per_fold = []
    t_start = time.time()

    import lightgbm as lgb
    for f in FOLDS:
        tr_folds = [x for x in (0, 1, 2, 3, 4) if x != f]
        tr_rows = d.rows_for(tr_folds)
        va_rows = d.rows_for([f])
        if len(tr_rows) > MAX_TRAIN_ROWS:
            tr_rows = np.sort(rng.choice(tr_rows, MAX_TRAIN_ROWS, replace=False))

        if kind == "t1":
            ytr = Q[tr_rows]
        else:
            ytr = 0.5 * y_hard[tr_rows] + 0.5 * Q[tr_rows]

        Xtr = PL._stack(mats, names, tr_rows, keep_idx)
        assert Xtr.shape[1] == 500, f"expected 500 causal columns, got {Xtr.shape[1]}"
        p = dict(ARM_B_PARAMS)
        p["objective"] = "xentropy"
        n_round = p.pop("n_estimators")
        ds = lgb.Dataset(Xtr, label=ytr, params=dict(p),
                         feature_name=[f"f{i}" for i in range(Xtr.shape[1])])
        booster = lgb.train(p, ds, num_boost_round=n_round)
        del Xtr, ds

        Xva = PL._stack(mats, names, va_rows, keep_idx)
        assert Xva.shape[1] == 500
        pred = booster.predict(Xva).astype(np.float32)
        del Xva
        oof[va_rows] = pred
        s = float(ts_auc_flat(pred, d.y[va_rows], d.t[va_rows]))
        per_fold.append(s)
        print(f"  [{kind.upper()}-full] fold {f}: TS-AUC {s:.5f}  "
              f"({len(tr_rows)} train rows, {len(va_rows)} valid rows)", flush=True)

    dev_rows = d.rows_for(list(FOLDS))
    overall = float(ts_auc_flat(oof[dev_rows], d.y[dev_rows], d.t[dev_rows]))
    np.save(f"{OOFDIR}/{exp_id}.npy", oof)

    hyp = ("W7 teacher FULL T1: pure distillation, label = Q (RT-991 OOF, reused)"
           if kind == "t1" else
           "W7 teacher FULL T2: hard+teacher fixed 0.5/0.5 blend, label = 0.5*y + 0.5*Q")
    res = dict(
        experiment_id=exp_id, date=time.strftime("%Y-%m-%d %H:%M"), git_sha=PL.git_sha(),
        agent="agent0", hypothesis=hyp,
        falsification_condition="see research/WAVE7_TEACHER_PREREG.md §6, promotion bar "
                                "research/HANDOFF_WAVE6.md §3.3",
        feature_set=",".join(FULL), n_features=500, model="lgbm", objective="xentropy",
        folds=",".join(map(str, FOLDS)), random_seed=0,
        train_series=int((~np.isin(d.series_fold, [-1])).sum()), train_rows=MAX_TRAIN_ROWS,
        mean_oof_ts_auc=float(np.mean(per_fold)), pooled_oof_ts_auc=overall,
        per_fold_ts_auc=";".join(f"{x:.5f}" for x in per_fold), fold_std=float(np.std(per_fold)),
        persistence="none", sample_mode="uniform",
        training_runtime_s=round(time.time() - t_start, 1),
        causal_verified="student inputs = unmodified 500-col causal bank; teacher (Q) used "
                        "only as training label, never as a feature",
        test_reduced_touched="no", lockbox_touched="no", status="recorded",
        notes="W7 teacher FULL 5-fold run, promoted from one-fold pilot RT-992/RT-993 "
              "per research/WAVE7_TEACHER_PREREG.md §6 continuation gate (both cleared).",
        protocol="full",
    )
    PL.append_result(res)
    print(json.dumps({k: res[k] for k in
                      ("experiment_id", "mean_oof_ts_auc", "pooled_oof_ts_auc",
                       "per_fold_ts_auc", "fold_std")}, indent=2))
    return res


# ---------------------------------------------------------------------------
def analyze():
    c = Ctx()
    va_rows0 = c.rows[VAL_FOLD]
    y0, t0, age0 = c.d.y[va_rows0], c.d.t[va_rows0], c.age[va_rows0]
    sidx0 = c.d.sidx[va_rows0]
    neg_is_prebreak0 = c.has_break[sidx0] & (y0 == 0)
    cellmask0 = cell_mask(y0, t0, age0)
    cellmask0_never = cellmask0 & ((y0 == 1) | (~neg_is_prebreak0))
    cellmask0_pre = cellmask0 & ((y0 == 1) | neg_is_prebreak0)

    arms = {"T0_RT990": "RT-990", "T1_RT992": "RT-992", "T2_RT993": "RT-993"}
    results = {}
    for label, exp_id in arms.items():
        full = np.load(f"{OOFDIR}/{exp_id}.npy")
        v0 = full[va_rows0]
        results[label] = {
            "experiment_id": exp_id,
            "cell_ts_auc": score_on(v0, cellmask0, y0, t0),
            "cell_ts_auc_never_break_only": score_on(v0, cellmask0_never, y0, t0),
            "cell_ts_auc_pre_break_only": score_on(v0, cellmask0_pre, y0, t0),
            "whole_fold0_ts_auc": float(ts_auc_flat(v0, y0, t0)),
        }
        print(f"  {label} ({exp_id}): cell {results[label]['cell_ts_auc']:.5f}  "
              f"whole-fold0 {results[label]['whole_fold0_ts_auc']:.5f}")

    t0_cell = results["T0_RT990"]["cell_ts_auc"]
    t0_whole = results["T0_RT990"]["whole_fold0_ts_auc"]

    with open(f"{REPORTS}/wave7_d3r.json") as f:
        d3r = json.load(f)
    armB_fold0 = d3r["results"]["B_RT990"]["per_fold_cell_ts_auc"][str(VAL_FOLD)]
    armC_fold0 = d3r["results"]["C_RT991"]["per_fold_cell_ts_auc"][str(VAL_FOLD)]
    future_gap_fold0 = armC_fold0 - armB_fold0
    frac_weight = d3r["cell_pair_weight_fraction"]

    deltas = {}
    for label in ("T1_RT992", "T2_RT993"):
        cell_d = results[label]["cell_ts_auc"] - t0_cell
        whole_d = results[label]["whole_fold0_ts_auc"] - t0_whole
        translated = frac_weight * cell_d
        eff = cell_d / future_gap_fold0 if future_gap_fold0 else None
        deltas[label] = {
            "cell_delta": cell_d, "whole_fold0_delta": whole_d,
            "translated_aggregate_delta": translated,
            "distillation_efficiency": eff, "bucket": bucket(cell_d),
        }
        print(f"  {label} - T0: cell {cell_d:+.5f} ({bucket(cell_d)})  "
              f"whole {whole_d:+.5f}  translated {translated:+.5f}  "
              f"efficiency {eff:.1%}" if eff is not None else "")

    best_label = max(deltas, key=lambda k: deltas[k]["cell_delta"])
    best = deltas[best_label]

    gate_A = best["cell_delta"] >= 0.010 and best["whole_fold0_delta"] > -0.003
    gate_B = best["whole_fold0_delta"] >= 0.003
    gate_C = best["translated_aggregate_delta"] >= 0.004
    continue_gate = gate_A or gate_B or gate_C
    if best["translated_aggregate_delta"] <= 0.0005:
        verdict = "KILL"
    elif best["translated_aggregate_delta"] <= 0.001:
        verdict = "PROBABLY KILL"
    elif continue_gate:
        verdict = "CONTINUE"
    else:
        verdict = "AMBIGUOUS -- below continuation gate, above kill threshold"

    out = {
        "experiment": "W7 teacher pilot", "prereg": "research/WAVE7_TEACHER_PREREG.md",
        "val_fold": VAL_FOLD, "armB_fold0_cell_auc": armB_fold0, "armC_fold0_cell_auc": armC_fold0,
        "future_gap_fold0": future_gap_fold0, "cell_pair_weight_fraction": frac_weight,
        "results": results, "deltas": deltas,
        "best_arm": best_label, "gate_A": gate_A, "gate_B": gate_B, "gate_C": gate_C,
        "verdict": verdict,
    }
    with open(f"{REPORTS}/wave7_teacher_pilot.json", "w") as f:
        json.dump(out, f, indent=2)

    md = ["# WAVE 7 — TEACHER PILOT RESULTS (fold 0 only)\n",
          "Pre-registered: `research/WAVE7_TEACHER_PREREG.md`. Teacher = "
          "`RT-991` (W7-D3R Arm C, reused). Fold 0 only -- not a full "
          "5-fold result.\n",
          f"D3R fold-0 reference: Arm B cell AUC {armB_fold0:.5f}, Arm C cell "
          f"AUC {armC_fold0:.5f}, future gap {future_gap_fold0:+.5f}.\n",
          "## Arms\n",
          "| arm | exp id | cell TS-AUC | never-break-only | pre-break-only | whole-fold0 TS-AUC |",
          "|---|---|---:|---:|---:|---:|"]
    for label, exp_id in arms.items():
        r = results[label]
        md.append(f"| {label} | `{exp_id}` | {r['cell_ts_auc']:.5f} | "
                  f"{r['cell_ts_auc_never_break_only']:.5f} | "
                  f"{r['cell_ts_auc_pre_break_only']:.5f} | {r['whole_fold0_ts_auc']:.5f} |")
    md += ["\n## Deltas vs T0\n",
           "| arm | cell Δ | bucket | whole-fold0 Δ | translated aggregate Δ | distillation efficiency |",
           "|---|---:|---|---:|---:|---:|"]
    for label, dd in deltas.items():
        eff = f"{dd['distillation_efficiency']:.1%}" if dd["distillation_efficiency"] is not None else "n/a"
        md.append(f"| {label} | {dd['cell_delta']:+.5f} | {dd['bucket']} | "
                  f"{dd['whole_fold0_delta']:+.5f} | {dd['translated_aggregate_delta']:+.5f} | {eff} |")
    md += [f"\n## Gates (best arm: {best_label})\n",
           f"A (cell ≥+0.010, no material damage): {gate_A}  \n"
           f"B (whole-fold0 ≥+0.003): {gate_B}  \n"
           f"C (translated ≥+0.004): {gate_C}\n",
           f"\n## VERDICT: **{verdict}**\n"]
    with open(f"{REPORTS}/wave7_teacher_pilot.md", "w") as f:
        f.write("\n".join(md))
    print(f"\nVERDICT: {verdict}")
    print(f"wrote {REPORTS}/wave7_teacher_pilot.{{json,md}}")
    return out


PROMOTION_BAR_MEAN = 0.0030
PROMOTION_BAR_FOLDS = 4  # of 5


def analyze_full():
    """Full 5-fold promotion-track analysis: aggregate TS-AUC vs the matched
    control (RT-990: same rows/columns/capacity, hard label), per-fold count,
    and paired series-level bootstrap (research/HANDOFF_WAVE6.md §3.3 legs
    1-3). Leg 4 (alternate-partition stability) is NOT run here -- it needs
    separate retraining on research/folds/folds_alt{1,2,3}.parquet and is
    reported as outstanding, not silently skipped.
    """
    c = Ctx()
    arms = {"T0_RT990": "RT-990", "RT300_champion_single": "RT-300",
            "T1_RT994": "RT-994", "T2_RT995": "RT-995"}
    vecs, results = {}, {}
    for label, exp_id in arms.items():
        v = np.load(f"{OOFDIR}/{exp_id}.npy")
        vecs[label] = v
        mean_auc, per = c.score(v)
        pooled = c.pooled(v)
        cell_by_fold = {}
        for k in FOLDS:
            r = c.rows[k]
            y, t, age = c.d.y[r], c.d.t[r], c.age[r]
            cm = cell_mask(y, t, age)
            cell_by_fold[k] = score_on(v[r], cm, y, t)
        results[label] = {
            "experiment_id": exp_id, "mean_ts_auc": mean_auc, "pooled_ts_auc": pooled,
            "per_fold_ts_auc": per, "cell_ts_auc_by_fold": cell_by_fold,
        }
        print(f"  {label} ({exp_id}): mean {mean_auc:.5f}  pooled {pooled:.5f}  "
              f"per-fold {['%.5f' % x for x in per]}")

    t0 = results["T0_RT990"]
    deltas = {}
    for label in ("T1_RT994", "T2_RT995"):
        r = results[label]
        per_fold_delta = [r["per_fold_ts_auc"][k] - t0["per_fold_ts_auc"][k] for k in range(5)]
        n_pos = sum(1 for x in per_fold_delta if x > 0)
        mean_delta = r["mean_ts_auc"] - t0["mean_ts_auc"]
        pooled_delta = r["pooled_ts_auc"] - t0["pooled_ts_auc"]
        deltas[label] = {
            "mean_delta_vs_RT990": mean_delta, "pooled_delta_vs_RT990": pooled_delta,
            "per_fold_delta": per_fold_delta, "folds_positive": n_pos,
            "clears_magnitude_bar": mean_delta >= PROMOTION_BAR_MEAN,
            "clears_fold_bar": n_pos >= PROMOTION_BAR_FOLDS,
        }
        print(f"  {label} - T0: mean Δ {mean_delta:+.5f}  pooled Δ {pooled_delta:+.5f}  "
              f"folds positive {n_pos}/5")

    print("\n  running paired series-level bootstrap (200 reps)...")
    contrasts = {"T1-T0": ("T1_RT994", "T0_RT990"), "T2-T0": ("T2_RT995", "T0_RT990"),
                 "T1-RT300": ("T1_RT994", "RT300_champion_single"),
                 "T2-RT300": ("T2_RT995", "RT300_champion_single")}
    boot = c.bootstrap(vecs, contrasts, n=200, seed=0)
    for k, v in boot.items():
        print(f"    {k}: mean {v['mean']:+.5f}  CI95 [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]  "
              f"frac>0 {v['fraction_positive']:.2f}")

    verdicts = {}
    for label in ("T1_RT994", "T2_RT995"):
        dd = deltas[label]
        boot_key = "T1-T0" if label == "T1_RT994" else "T2-T0"
        bootstrap_supportive = boot[boot_key]["ci95"][0] > 0
        legs = {
            "1_magnitude_ge_0.0030": dd["clears_magnitude_bar"],
            "2_folds_ge_4of5": dd["clears_fold_bar"],
            "3_bootstrap_ci_above_zero": bootstrap_supportive,
            "4_alternate_partitions": "NOT RUN -- outstanding, see notes",
        }
        legs_1_3_clear = legs["1_magnitude_ge_0.0030"] and legs["2_folds_ge_4of5"] and legs["3_bootstrap_ci_above_zero"]
        verdicts[label] = {
            "legs": legs,
            "verdict": ("LEGS 1-3 CLEAR, LEG 4 (alternate partitions) OUTSTANDING -- "
                       "not yet a full promotion, no submission") if legs_1_3_clear else
                       "DOES NOT CLEAR THE PROMOTION BAR",
        }
        print(f"\n  {label}: {verdicts[label]['verdict']}")

    out = {
        "experiment": "W7 teacher FULL 5-fold", "prereg": "research/WAVE7_TEACHER_PREREG.md",
        "promotion_bar_source": "research/HANDOFF_WAVE6.md §3.3 / research/WAVE5_PREREG.md §4",
        "results": results, "deltas": deltas, "bootstrap": boot, "verdicts": verdicts,
    }
    with open(f"{REPORTS}/wave7_teacher_full.json", "w") as f:
        json.dump(out, f, indent=2)

    md = ["# WAVE 7 — TEACHER FULL 5-FOLD RESULTS\n",
          "Pre-registered: `research/WAVE7_TEACHER_PREREG.md` §6 (continuation "
          "of the one-fold pilot, which cleared its gate). Promotion bar: "
          "`research/HANDOFF_WAVE6.md` §3.3.\n",
          "## Aggregate TS-AUC (whole dev, all 5 folds)\n",
          "| arm | exp id | mean | pooled | fold0 | fold1 | fold2 | fold3 | fold4 |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, exp_id in arms.items():
        r = results[label]
        pf = r["per_fold_ts_auc"]
        md.append(f"| {label} | `{exp_id}` | {r['mean_ts_auc']:.5f} | {r['pooled_ts_auc']:.5f} | "
                  + " | ".join(f"{x:.5f}" for x in pf) + " |")
    md += ["\n## Deltas vs T0 (`RT-990`, matched control)\n",
           "| arm | mean Δ | pooled Δ | folds positive | magnitude bar (≥+0.0030) | fold bar (≥4/5) |",
           "|---|---:|---:|---:|---|---|"]
    for label, dd in deltas.items():
        md.append(f"| {label} | {dd['mean_delta_vs_RT990']:+.5f} | {dd['pooled_delta_vs_RT990']:+.5f} | "
                  f"{dd['folds_positive']}/5 | {dd['clears_magnitude_bar']} | {dd['clears_fold_bar']} |")
    md += ["\n## Paired series bootstrap (200 reps)\n",
           "| contrast | mean | CI95 | fraction > 0 |", "|---|---:|---|---:|"]
    for k, v in boot.items():
        md.append(f"| {k} | {v['mean']:+.5f} | [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}] | {v['fraction_positive']:.2f} |")
    md += ["\n## Promotion-bar verdicts\n"]
    for label, vv in verdicts.items():
        md.append(f"**{label}**: {vv['verdict']}\n")
        for k, val in vv["legs"].items():
            md.append(f"* {k}: {val}")
        md.append("")
    md += ["**Leg 4 (alternate-partition stability) has not been run** -- it "
           "requires retraining on `research/folds/folds_alt{1,2,3}.parquet` "
           "and is a separate, further compute commitment, not skipped "
           "silently. No submission until it is checked.\n"]
    with open(f"{REPORTS}/wave7_teacher_full.md", "w") as f:
        f.write("\n".join(md))
    print(f"\nwrote {REPORTS}/wave7_teacher_full.{{json,md}}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostics", action="store_true")
    ap.add_argument("--parity-check", action="store_true")
    ap.add_argument("--train", choices=["t1", "t2"])
    ap.add_argument("--train-full", choices=["t1", "t2"])
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--analyze-full", action="store_true")
    args = ap.parse_args()
    if args.diagnostics:
        diagnostics()
    elif args.parity_check:
        parity_check()
    elif args.train:
        train_student(args.train)
    elif args.train_full:
        train_student_full(args.train_full)
    elif args.analyze:
        analyze()
    elif args.analyze_full:
        analyze_full()
    else:
        raise SystemExit("pass --diagnostics, --parity-check, --train t1|t2, "
                         "--train-full t1|t2, --analyze, or --analyze-full")


if __name__ == "__main__":
    main()
