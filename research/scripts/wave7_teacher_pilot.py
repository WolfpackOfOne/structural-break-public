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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostics", action="store_true")
    ap.add_argument("--parity-check", action="store_true")
    ap.add_argument("--train", choices=["t1", "t2"])
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.diagnostics:
        diagnostics()
    elif args.parity_check:
        parity_check()
    elif args.train:
        train_student(args.train)
    elif args.analyze:
        analyze()
    else:
        raise SystemExit("pass --diagnostics, --parity-check, --train t1|t2, or --analyze")


if __name__ == "__main__":
    main()
