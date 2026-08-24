"""M3 -- Oracle Repair Ranker (ORR). Fold-0 pilot.

Pre-registered in research/WAVE8_FUTURE_AWARE_PREREG.md section 5.3.

Reuses Wave 7's already-computed nested inner-fold teacher Q
(research/oof/nested_Q_outer0_inner{1,2,3,4}.npy -- outer_f=0, so these cover
exactly outer_train = folds {1,2,3,4}, each fold's Q from a teacher that
never saw {0, that fold}) and RT-600's per-specialist OOF -- NO new teacher
training. Mines same-t (positive, negative) pairs within each of those four
folds, keeps ones where RT-600 is wrong (m_R<0) and the teacher is confident
(m_T above quantile threshold), builds a per-row "repair pressure" REGRESSION
target from confirmed pairs (not a pairwise loss -- this project's prior
pairwise-ranking rows, RT-111/RT-700/RT-701, were mildly negative), trains it
with the same nested double-cross-fit machinery as SST, and blends
score = RT600_score + beta*repair_pred with beta chosen on inner folds only.

Usage:
    python wave8_orr.py --mine            # build confirmed-pair sets + row targets, cache
    python wave8_orr.py --train           # RT-1020: nested repair regressor
    python wave8_orr.py --blend           # RT-1021: beta selection + blended fold-0 score
    python wave8_orr.py --analyze
"""
from __future__ import annotations

import argparse, json, os, time

import numpy as np

import wave8_common as W8
import sbr.pipeline as PL
from wave5_lib import Ctx, FOLDS, OOFDIR, REPORTS, ts_auc_flat
from wave7_d3r import FULL, cell_mask, score_on

OUTER_F = 0
INNER_FOLDS = [g for g in FOLDS if g != OUTER_F]     # 1,2,3,4
CACHE = f"{OOFDIR}/wave8_orr"
os.makedirs(CACHE, exist_ok=True)
EPS = 1e-6
PAIRS_PER_T = 12          # sampled (pos,neg) pairs per online index t per fold
CONFIDENT_Q = 0.75        # teacher-margin quantile for "confident"

SPECIALISTS = W8.SPECIALISTS


def rt600_score(c):
    P = {s: np.load(f"{OOFDIR}/{s}.npy") for s in SPECIALISTS}
    return c.crossfit_blend(P, SPECIALISTS)


def load_nested_q():
    Q = np.full(5_036_517, np.nan, dtype=np.float64)
    n = None
    for g in INNER_FOLDS:
        arr = np.load(f"{OOFDIR}/nested_Q_outer{OUTER_F}_inner{g}.npy")
        if n is None:
            n = len(arr)
            Q = np.full(n, np.nan, dtype=np.float64)
        Q = np.where(np.isnan(Q), arr, Q)
    return Q


# ---------------------------------------------------------------------------
def mine_pairs(c, R, Q):
    """Per fold g, sample pairs at matched t, keep 'confirmed' subset where
    RT-600 is wrong and the teacher is confident. Returns:
      confirmed: list of (fold, pos_row, neg_row, m_R, m_T)
      row_pressure: dict fold -> (n_rows_in_fold_local,) repair-pressure array
      row_index: dict fold -> global row indices for that local array
    """
    rng = np.random.default_rng(0)
    confirmed_all = []
    pressure_by_fold = {}
    m_T_pool = []
    for g in INNER_FOLDS:
        rows = c.rows[g]
        y, t = c.d.y[rows], c.d.t[rows]
        r_score, q_score = R[rows], Q[rows]
        assert not np.isnan(q_score).any(), f"fold {g} has un-nested Q rows"
        order = np.argsort(t, kind="stable")
        t_sorted = t[order]
        b = np.flatnonzero(np.r_[True, t_sorted[1:] != t_sorted[:-1]])
        e = np.r_[b[1:], len(t_sorted)]
        pos_local = y[order] == 1
        pressure = np.zeros(len(rows), dtype=np.float64)
        pair_buf = []
        for lo, hi in zip(b, e):
            idx = order[lo:hi]
            p_idx = idx[y[idx] == 1]
            n_idx = idx[y[idx] == 0]
            if len(p_idx) == 0 or len(n_idx) == 0:
                continue
            k = min(PAIRS_PER_T, len(p_idx), len(n_idx))
            pp = rng.choice(p_idx, k, replace=False)
            nn = rng.choice(n_idx, k, replace=False)
            for pi, ni in zip(pp, nn):
                m_R = float(r_score[pi] - r_score[ni])
                m_T = float(q_score[pi] - q_score[ni])
                pair_buf.append((pi, ni, m_R, m_T))
        m_T_pool += [x[3] for x in pair_buf]
        thresh = float(np.quantile(np.abs(m_T_pool), CONFIDENT_Q)) if m_T_pool else 0.0
        for pi, ni, m_R, m_T in pair_buf:
            if m_R < 0 and m_T > thresh:
                pressure[pi] += 1.0
                pressure[ni] -= 1.0
                confirmed_all.append((g, int(rows[pi]), int(rows[ni]), m_R, m_T))
        pressure_by_fold[g] = pressure
    return confirmed_all, pressure_by_fold


def cmd_mine():
    c = Ctx()
    R = rt600_score(c)
    Q = load_nested_q()
    t0 = time.time()
    confirmed, pressure_by_fold = mine_pairs(c, R, Q)
    print(f"mined {sum(len(c.rows[g]) for g in INNER_FOLDS)} candidate-t rows, "
          f"{len(confirmed)} confirmed teacher-corrected inversions, {time.time()-t0:.1f}s")
    n = len(c.d.y)
    target = np.full(n, np.nan, dtype=np.float64)
    train_mask = np.zeros(n, dtype=bool)
    for g in INNER_FOLDS:
        rows = c.rows[g]
        target[rows] = pressure_by_fold[g]
        train_mask[rows] = True
    np.save(f"{CACHE}/repair_target.npy", target)
    np.save(f"{CACHE}/repair_train_mask.npy", train_mask)
    with open(f"{CACHE}/confirmed_pairs.json", "w") as f:
        json.dump({"n_confirmed": len(confirmed), "confident_quantile": CONFIDENT_Q,
                   "sample": confirmed[:20]}, f, indent=2, default=float)
    print(f"cached repair target + {len(confirmed)} confirmed pairs -> {CACHE}/")


# ---------------------------------------------------------------------------
def cmd_train():
    d = PL.Data()
    mats, names = PL.load_features(FULL)
    keep_idx = np.arange(len(names))
    W8.assert_no_forbidden_columns(names)
    c = Ctx()
    R = rt600_score(c)

    target = np.load(f"{CACHE}/repair_target.npy")
    train_mask = np.load(f"{CACHE}/repair_train_mask.npy")

    # legal inputs = 500 causal cols + RT600 score + 7 specialist scores,
    # appended as extra columns via the same augmented-feature pattern SST uses.
    P = {s: np.load(f"{OOFDIR}/{s}.npy") for s in SPECIALISTS}
    extra_full = np.column_stack([R] + [P[s] for s in SPECIALISTS])  # (n, 8)

    t0 = time.time()
    import lightgbm as lgb
    p = dict(W8.REGRESSOR_PARAMS)
    outer_train_rows = d.rows_for(INNER_FOLDS)
    val_rows = d.rows_for([OUTER_F])

    def fit_predict(fit_rows, pred_rows):
        Xtr = np.concatenate([PL._stack(mats, names, fit_rows, keep_idx), extra_full[fit_rows]], axis=1)
        ytr = target[fit_rows]
        ds = lgb.Dataset(Xtr, label=ytr, params=p, feature_name=[f"f{i}" for i in range(Xtr.shape[1])])
        booster = lgb.train(p, ds, num_boost_round=200)
        del Xtr, ds
        Xp = np.concatenate([PL._stack(mats, names, pred_rows, keep_idx), extra_full[pred_rows]], axis=1)
        pred = booster.predict(Xp).astype(np.float64)
        del Xp
        return pred

    # inner nested OOF over the 4 outer-train folds (for recoverability + beta selection)
    inner_oof = np.full(len(d.y), np.nan, dtype=np.float64)
    for g in INNER_FOLDS:
        fit_folds = [x for x in INNER_FOLDS if x != g]
        fit_rows = d.rows_for(fit_folds)
        fit_rows = fit_rows[train_mask[fit_rows]]
        pred_rows = d.rows_for([g])
        inner_oof[pred_rows] = fit_predict(fit_rows, pred_rows)
        print(f"  inner repair regressor, held out fold {g}: {len(pred_rows)} rows predicted "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)
    np.save(f"{CACHE}/repair_inner_oof.npy", inner_oof)

    # refit on ALL outer_train, predict fold 0 (second-level-OOF contract)
    fit_rows_all = outer_train_rows[train_mask[outer_train_rows]]
    val_pred = fit_predict(fit_rows_all, val_rows)
    repair_val = np.full(len(d.y), np.nan, dtype=np.float64)
    repair_val[val_rows] = val_pred
    np.save(f"{OOFDIR}/RT-1020.npy", repair_val)
    print(f"[ORR train] fold-0 repair predictions saved -> RT-1020.npy, "
          f"total runtime {time.time()-t0:.1f}s")

    res = dict(experiment_id="RT-1020", date=time.strftime("%Y-%m-%d %H:%M"), git_sha=PL.git_sha(),
               agent="agent0",
               hypothesis="ORR repair-propensity regressor: predicts teacher-confirmed RT-600 "
                          "same-t inversion pressure from 500 causal cols + RT600 + 7 specialist "
                          "scores. Regression target, not a pairwise loss -- checked against "
                          "this project's prior negative pairwise-ranking rows (RT-111/700/701).",
               falsification_condition="see research/WAVE8_FUTURE_AWARE_PREREG.md section 5.3",
               feature_set=",".join(FULL) + "+rt600_score+7_specialist_scores",
               n_features=508, model="lgbm", objective="regression",
               folds=str(OUTER_F), random_seed=0,
               train_series=int((~np.isin(d.series_fold, [-1])).sum()), train_rows=len(fit_rows_all),
               mean_oof_ts_auc=None, pooled_oof_ts_auc=None, per_fold_ts_auc="", fold_std=None,
               persistence="none", sample_mode="uniform",
               training_runtime_s=round(time.time() - t0, 1),
               causal_verified="nested double-cross-fit; teacher Q used only to MINE the training "
                               "target, never as a feature; no true tau/age/n_online reachable",
               test_reduced_touched="no", lockbox_touched="no", status="recorded",
               notes="Wave 8 ORR pilot, repair-propensity model only (not yet blended -- see RT-1021).",
               protocol="pilot_fold0_only")
    PL.append_result(res)


# ---------------------------------------------------------------------------
def cmd_blend():
    c = Ctx()
    R = rt600_score(c)
    inner_oof = np.load(f"{CACHE}/repair_inner_oof.npy")
    repair_val = np.load(f"{OOFDIR}/RT-1020.npy")

    # beta selection on INNER folds only (never fold 0)
    inner_rows = np.concatenate([c.rows[g] for g in INNER_FOLDS])
    y_in, t_in = c.d.y[inner_rows], c.d.t[inner_rows]
    r_in, rep_in = R[inner_rows], inner_oof[inner_rows]
    betas = np.linspace(0, 2.0, 21)
    best_beta, best_s = 0.0, ts_auc_flat(r_in, y_in, t_in)
    for b in betas:
        s = float(ts_auc_flat(r_in + b * rep_in, y_in, t_in))
        if s > best_s:
            best_s, best_beta = s, b
    print(f"beta selected on inner folds only: {best_beta:.3f} (inner TS-AUC {best_s:.5f} "
          f"vs unblended {ts_auc_flat(r_in, y_in, t_in):.5f})")

    val_rows = c.rows[OUTER_F]
    blended = R.copy()
    blended[val_rows] = R[val_rows] + best_beta * repair_val[val_rows]
    np.save(f"{OOFDIR}/RT-1021.npy", blended)

    s_r = float(ts_auc_flat(R[val_rows], c.d.y[val_rows], c.d.t[val_rows]))
    s_blend = float(ts_auc_flat(blended[val_rows], c.d.y[val_rows], c.d.t[val_rows]))
    print(f"[ORR blend] fold-0 RT600-only {s_r:.5f}  blended {s_blend:.5f}  delta {s_blend-s_r:+.5f}")

    with open(f"{CACHE}/beta.json", "w") as f:
        json.dump({"beta": best_beta, "inner_ts_auc_unblended": float(ts_auc_flat(r_in, y_in, t_in)),
                   "inner_ts_auc_blended": best_s}, f, indent=2)

    res = dict(experiment_id="RT-1021", date=time.strftime("%Y-%m-%d %H:%M"), git_sha=PL.git_sha(),
               agent="agent0",
               hypothesis=f"ORR blend: score = RT600 + {best_beta:.3f}*repair_pred, beta chosen "
                          f"on inner folds only.",
               falsification_condition="see research/WAVE8_FUTURE_AWARE_PREREG.md section 5.3",
               feature_set="RT600_7stream+RT-1020_repair", n_features=None, model="blend",
               objective="binary", folds=str(OUTER_F), random_seed=0,
               train_series=None, train_rows=None,
               mean_oof_ts_auc=s_blend, pooled_oof_ts_auc=s_blend, per_fold_ts_auc=f"{s_blend:.5f}",
               fold_std=0.0, persistence="none", sample_mode="uniform", training_runtime_s=0.0,
               causal_verified="blend of two already-causal legal scores; beta selected on inner "
                               "folds only, never fold 0", test_reduced_touched="no",
               lockbox_touched="no", status="recorded",
               notes="Wave 8 ORR pilot blend. Pre-registered research/WAVE8_FUTURE_AWARE_PREREG.md.",
               protocol="pilot_fold0_only")
    PL.append_result(res)


# ---------------------------------------------------------------------------
def analyze():
    c = Ctx()
    y, t, age, sidx = c.d.y, c.d.t, c.age, c.d.sidx
    r0 = c.rows[OUTER_F]
    cellmask0 = cell_mask(y[r0], t[r0], age[r0])

    with open(f"{CACHE}/confirmed_pairs.json") as f:
        mined = json.load(f)
    inner_oof = np.load(f"{CACHE}/repair_inner_oof.npy")
    R = rt600_score(c)
    Q = load_nested_q()
    confirmed, _ = mine_pairs(c, R, Q)  # re-derive to score recoverability on the exact set

    hits = 0
    for g, pi, ni, m_R, m_T in confirmed:
        if inner_oof[pi] > inner_oof[ni]:
            hits += 1
    recoverability = hits / len(confirmed) if confirmed else None

    RT1021 = np.load(f"{OOFDIR}/RT-1021.npy")
    whole0_R = score_on(R[r0], np.ones(len(r0), bool), y[r0], t[r0])
    whole0_blend = score_on(RT1021[r0], np.ones(len(r0), bool), y[r0], t[r0])
    cell_R = score_on(R[r0][cellmask0], np.ones(cellmask0.sum(), bool), y[r0][cellmask0], t[r0][cellmask0])
    cell_blend = score_on(RT1021[r0][cellmask0], np.ones(cellmask0.sum(), bool), y[r0][cellmask0], t[r0][cellmask0])

    ens = W8.ensemble_marginal(RT1021, c=c, fold=OUTER_F, label="orr")

    out = {
        "n_confirmed_pairs": len(confirmed), "recoverability": recoverability,
        "whole_fold0_rt600": whole0_R, "whole_fold0_blend": whole0_blend,
        "delta_whole": whole0_blend - whole0_R,
        "cell_rt600": cell_R, "cell_blend": cell_blend,
        "delta_cell": (cell_blend - cell_R) if (cell_R is not None and cell_blend is not None) else None,
        "ensemble": ens,
    }
    gate_a = ens["marginal_vs_clone"] >= 0.002
    gate_b = (recoverability is not None and recoverability >= 0.55)
    out["continuation_gate_cleared"] = bool(gate_a and gate_b)
    print(json.dumps(out, indent=2))
    with open(f"{REPORTS}/wave8_orr.json", "w") as f:
        json.dump(out, f, indent=2)
    md = ["# WAVE 8 -- ORR PILOT RESULT (fold 0)\n",
          f"Confirmed teacher-corrected inversions mined (inner folds 1-4): **{len(confirmed)}**\n",
          f"Recoverability P(repair_pos>repair_neg | RT600 wrong, teacher confident): "
          f"**{recoverability}**\n",
          "| | RT-600 alone | + repair blend | delta |", "|---|---:|---:|---:|",
          f"| whole fold 0 | {whole0_R:.5f} | {whole0_blend:.5f} | {whole0_blend-whole0_R:+.5f} |",
          f"| dominant cell | {cell_R:.5f} | {cell_blend:.5f} | {(cell_blend-cell_R):+.5f} |",
          "\n## Ensemble marginal (fold 0)\n", "```json", json.dumps(ens, indent=2), "```",
          f"\n## CONTINUATION GATE: {'CLEARED' if out['continuation_gate_cleared'] else 'NOT CLEARED'}\n"]
    with open(f"{REPORTS}/wave8_orr.md", "w") as f:
        f.write("\n".join(md))
    print(f"wrote {REPORTS}/wave8_orr.{{md,json}}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--blend", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.mine:
        cmd_mine()
    elif args.train:
        cmd_train()
    elif args.blend:
        cmd_blend()
    elif args.analyze:
        analyze()
    else:
        raise SystemExit("pass --mine, --train, --blend, or --analyze")


if __name__ == "__main__":
    main()
