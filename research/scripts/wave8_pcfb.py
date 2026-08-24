"""M2 -- Predictability-Constrained Future Bottleneck (PCFB). Fold-0 pilot.

Pre-registered in research/WAVE8_FUTURE_AWARE_PREREG.md section 5.2. Frozen
to horizon t+200, d=16, methods {PCA16, PLS16} only.

Future source: the same 500 causal columns (wave7_d3r.FULL) evaluated at
future_row_index(d, 200), for rows with h=200 support (SST's eligibility
cache is reused, not recomputed).

PCA16: PCA of the future 500-col vectors, fit on a subsample of eligible
outer-train rows.
PLS16: PLS between current 500 cols (X) and future 500 cols (Y), same fit
population -- the shared-predictability decomposition the proposal asks for,
as opposed to PCA's information-only decomposition.

For each method: oracle Z (true future vector projected through the FITTED
transform) vs legal Zhat (16 small LightGBM regressors, current X_t ->
Z_component, nested double-cross-fit exactly like SST).

Usage:
    python wave8_pcfb.py --fit               # fit PCA16/PLS16 on a subsample, cache transforms
    python wave8_pcfb.py --arm b --method pca # RT-1030
    python wave8_pcfb.py --arm c --method pca # RT-1031
    python wave8_pcfb.py --arm b --method pls # RT-1032
    python wave8_pcfb.py --arm c --method pls # RT-1033
    python wave8_pcfb.py --analyze --method pls
"""
from __future__ import annotations

import argparse, json, os, time

import numpy as np

import wave8_common as W8
import sbr.pipeline as PL
from wave5_lib import Ctx, FOLDS, OOFDIR, REPORTS, ts_auc_flat
from wave7_d3r import FULL, ARM_B_PARAMS, cell_mask, score_on

OUTER_F = 0
H = 200
D = 16
FIT_SAMPLE_ROWS = 150_000
MAX_TRAIN_ROWS = 1_000_000
CACHE = f"{OOFDIR}/wave8_pcfb"
os.makedirs(CACHE, exist_ok=True)


def eligible_h200(d):
    yp, vp = (f"{OOFDIR}/wave8_sst/Y_h200.npy", f"{OOFDIR}/wave8_sst/valid_h200.npy")
    assert os.path.exists(vp), "run wave8_sst.py --targets first (shares the h=200 eligibility mask)"
    return np.load(vp)


def future_500(d, mats, names, rows, first_row, series_len, keep_idx):
    fut, valid = W8.future_row_index(d, H, first_row, series_len)
    assert valid[rows].all(), "requested future_500 for rows without h=200 support"
    return PL._stack(mats, names, fut[rows], keep_idx)


# ---------------------------------------------------------------------------
def cmd_fit():
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import PLSRegression

    d = PL.Data()
    mats, names = PL.load_features(FULL)
    keep_idx = np.arange(len(names))
    first_row, series_len = W8.row_layout(d)
    valid = eligible_h200(d)
    outer_train_rows = d.rows_for([g for g in FOLDS if g != OUTER_F])
    elig_tr = outer_train_rows[valid[outer_train_rows]]
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(elig_tr, min(FIT_SAMPLE_ROWS, len(elig_tr)), replace=False))

    t0 = time.time()
    Xcur = PL._stack(mats, names, sample, keep_idx)
    Xfut = future_500(d, mats, names, sample, first_row, series_len, keep_idx)
    print(f"fit sample: {len(sample)} eligible outer-train rows, current+future stacked "
          f"({time.time()-t0:.1f}s)")

    # sklearn's PCA/PLS reject NaN; LightGBM steps elsewhere handle it natively,
    # but these two need a column-median impute, fit on this same outer-train
    # sample and reused (never refit) at prediction time.
    cur_median = np.nanmedian(Xcur, axis=0)
    cur_median[~np.isfinite(cur_median)] = 0.0
    fut_median = np.nanmedian(Xfut, axis=0)
    fut_median[~np.isfinite(fut_median)] = 0.0
    Xcur_i = np.where(np.isnan(Xcur), cur_median, Xcur)
    Xfut_i = np.where(np.isnan(Xfut), fut_median, Xfut)

    pca = PCA(n_components=D, random_state=0).fit(Xfut_i)
    import pickle
    with open(f"{CACHE}/pca16.pkl", "wb") as f:
        pickle.dump({"model": pca, "fut_median": fut_median}, f)
    print(f"PCA16 fit: explained_variance_ratio sum = {pca.explained_variance_ratio_.sum():.4f}")

    pls = PLSRegression(n_components=D, scale=True).fit(Xcur_i, Xfut_i)
    with open(f"{CACHE}/pls16.pkl", "wb") as f:
        pickle.dump({"model": pls, "fut_median": fut_median}, f)
    print(f"PLS16 fit: {time.time()-t0:.1f}s total")


def project(method, Xfut):
    import pickle
    with open(f"{CACHE}/{method}16.pkl", "rb") as f:
        blob = pickle.load(f)
    m, fut_median = blob["model"], blob["fut_median"]
    Xfut_i = np.where(np.isnan(Xfut), fut_median, Xfut)
    if method == "pca":
        return m.transform(Xfut_i)
    else:
        # PLS16 was fit as PLSRegression(Xcur, Xfut); its Y-side latent scores
        # (y_scores_ during fit) are not directly re-derivable for new Y without
        # re-running the same decomposition -- use the Y loading/rotation the
        # fitted object exposes for transforming new Y (future) vectors into
        # the shared latent space.
        Xfut_c = Xfut_i - m._y_mean
        Xfut_c = Xfut_c / m._y_std
        return Xfut_c @ m.y_rotations_


def load_model(method):
    import pickle
    with open(f"{CACHE}/{method}16.pkl", "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
def run_arm_b(method):
    d = PL.Data()
    mats, names = PL.load_features(FULL)
    keep_idx = np.arange(len(names))
    first_row, series_len = W8.row_layout(d)
    valid = eligible_h200(d)
    W8.assert_no_forbidden_columns(names)

    tr_rows_all = d.rows_for([g for g in FOLDS if g != OUTER_F])
    va_rows_all = d.rows_for([OUTER_F])
    tr_rows = tr_rows_all[valid[tr_rows_all]]
    va_rows = va_rows_all[valid[va_rows_all]]
    rng = np.random.default_rng(0)
    if len(tr_rows) > MAX_TRAIN_ROWS:
        tr_rows = np.sort(rng.choice(tr_rows, MAX_TRAIN_ROWS, replace=False))

    t0 = time.time()
    Zt = project(method, future_500(d, mats, names, tr_rows, first_row, series_len, keep_idx))
    Zv = project(method, future_500(d, mats, names, va_rows, first_row, series_len, keep_idx))

    import lightgbm as lgb
    p = dict(ARM_B_PARAMS)
    n_round = p.pop("n_estimators")
    Xtr = np.concatenate([PL._stack(mats, names, tr_rows, keep_idx), Zt], axis=1)
    ytr = d.y[tr_rows]
    ds = lgb.Dataset(Xtr, label=ytr, params=dict(p, objective="binary"),
                     feature_name=[f"f{i}" for i in range(Xtr.shape[1])])
    booster = lgb.train(dict(p), ds, num_boost_round=n_round)
    del Xtr, ds
    Xva = np.concatenate([PL._stack(mats, names, va_rows, keep_idx), Zv], axis=1)
    pred = booster.predict(Xva).astype(np.float32)
    del Xva
    s = float(ts_auc_flat(pred, d.y[va_rows], d.t[va_rows]))
    oof = np.full(len(d.y), np.nan, dtype=np.float32)
    oof[va_rows] = pred
    exp_id = "RT-1030" if method == "pca" else "RT-1032"
    np.save(f"{OOFDIR}/{exp_id}.npy", oof)
    print(f"[PCFB-B {method}] TS-AUC (eligible fold0) {s:.5f}  runtime {time.time()-t0:.1f}s -> {exp_id}")

    res = dict(experiment_id=exp_id, date=time.strftime("%Y-%m-%d %H:%M"), git_sha=PL.git_sha(),
               agent="agent0",
               hypothesis=f"PCFB-B oracle ({method}16): 500 causal + TRUE {method}16 latent of the "
                          f"future 500-col state at h=200, eligible rows only. OFFLINE DIAGNOSTIC ONLY.",
               falsification_condition="see research/WAVE8_FUTURE_AWARE_PREREG.md section 5.2",
               feature_set=",".join(FULL) + f"+{method}16_true", n_features=500 + D,
               model="lgbm", objective="binary", folds=str(OUTER_F), random_seed=0,
               train_series=int((~np.isin(d.series_fold, [-1])).sum()), train_rows=len(tr_rows),
               mean_oof_ts_auc=s, pooled_oof_ts_auc=s, per_fold_ts_auc=f"{s:.5f}", fold_std=0.0,
               persistence="none", sample_mode="uniform", training_runtime_s=round(time.time() - t0, 1),
               causal_verified="OFFLINE DIAGNOSTIC ONLY -- true future lookup is NOT causal",
               test_reduced_touched="no", lockbox_touched="no", status="recorded",
               notes="Wave 8 PCFB-B pilot. Pre-registered research/WAVE8_FUTURE_AWARE_PREREG.md.",
               protocol="pilot_fold0_only")
    PL.append_result(res)


def run_arm_c(method):
    d = PL.Data()
    mats, names = PL.load_features(FULL)
    keep_idx = np.arange(len(names))
    first_row, series_len = W8.row_layout(d)
    valid = eligible_h200(d)
    W8.assert_no_forbidden_columns(names)
    m = load_model(method)

    # true Z needed only on eligible rows, to build the nested regressor's training target
    all_rows = np.arange(len(d.y))
    elig_rows = all_rows[valid]
    rng = np.random.default_rng(0)
    if len(elig_rows) > MAX_TRAIN_ROWS:
        elig_rows_for_target = np.sort(rng.choice(elig_rows, MAX_TRAIN_ROWS, replace=False))
    else:
        elig_rows_for_target = elig_rows
    Zfut = project(method, future_500(d, mats, names, elig_rows_for_target, first_row, series_len, keep_idx))
    Ztrue = np.full((len(d.y), D), np.nan, dtype=np.float64)
    Ztrue[elig_rows_for_target] = Zfut

    t0 = time.time()
    Zhat = np.full((len(d.y), D), np.nan, dtype=np.float32)
    tr_rows_all = d.rows_for([g for g in FOLDS if g != OUTER_F])
    va_rows_all = d.rows_for([OUTER_F])
    dev_rows = np.concatenate([tr_rows_all, va_rows_all])
    for k in range(D):
        ckpt = f"{CACHE}/zhat_{method}_dim{k}.npy"
        if os.path.exists(ckpt):
            Zhat[:, k] = np.load(ckpt)
            print(f"  [PCFB-C {method}] latent dim {k+1}/{D} loaded from checkpoint "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)
            continue
        y_full = np.nan_to_num(Ztrue[:, k], nan=0.0)
        train_mask = ~np.isnan(Ztrue[:, k])
        z_inner = W8.nested_oof_regressor(OUTER_F, d, mats, names, keep_idx, y_full, train_mask,
                                          rounds=150, max_rows=250_000, seed=0)
        z_val = W8.full_predict_for_outer_val(OUTER_F, d, mats, names, keep_idx, y_full, train_mask,
                                              rounds=150, max_rows=250_000, seed=0)
        Zhat[tr_rows_all, k] = z_inner[tr_rows_all]
        Zhat[va_rows_all, k] = z_val[va_rows_all]
        np.save(ckpt, Zhat[:, k])       # per-dim checkpoint: a crash costs one dim, not all 16
        print(f"  [PCFB-C {method}] latent dim {k+1}/{D} done ({time.time()-t0:.0f}s elapsed)", flush=True)
    assert not np.isnan(Zhat[dev_rows]).any(), \
        "PCFB-C predicted latent matrix has uncovered DEV rows (lockbox rows are legitimately untouched)"

    tr_rows = tr_rows_all
    if len(tr_rows) > MAX_TRAIN_ROWS:
        rng2 = np.random.default_rng(0)
        tr_rows = np.sort(rng2.choice(tr_rows, MAX_TRAIN_ROWS, replace=False))

    import lightgbm as lgb
    p = dict(ARM_B_PARAMS)
    n_round = p.pop("n_estimators")
    t1 = time.time()
    Xtr = np.concatenate([PL._stack(mats, names, tr_rows, keep_idx), Zhat[tr_rows]], axis=1)
    ytr = d.y[tr_rows]
    ds = lgb.Dataset(Xtr, label=ytr, params=dict(p, objective="binary"),
                     feature_name=[f"f{i}" for i in range(Xtr.shape[1])])
    booster = lgb.train(dict(p), ds, num_boost_round=n_round)
    del Xtr, ds
    Xva = np.concatenate([PL._stack(mats, names, va_rows_all, keep_idx), Zhat[va_rows_all]], axis=1)
    pred = booster.predict(Xva).astype(np.float32)
    del Xva
    s = float(ts_auc_flat(pred, d.y[va_rows_all], d.t[va_rows_all]))
    oof = np.full(len(d.y), np.nan, dtype=np.float32)
    oof[va_rows_all] = pred
    exp_id = "RT-1031" if method == "pca" else "RT-1033"
    np.save(f"{OOFDIR}/{exp_id}.npy", oof)
    print(f"[PCFB-C {method}] TS-AUC (full fold0) {s:.5f}  classifier {time.time()-t1:.1f}s  "
          f"total {time.time()-t0:.1f}s -> {exp_id}")

    res = dict(experiment_id=exp_id, date=time.strftime("%Y-%m-%d %H:%M"), git_sha=PL.git_sha(),
               agent="agent0",
               hypothesis=f"PCFB-C legal ({method}16): 500 causal + nested-OOF predicted {method}16 "
                          f"latent of the future 500-col state at h=200. Causal at inference.",
               falsification_condition="see research/WAVE8_FUTURE_AWARE_PREREG.md section 5.2",
               feature_set=",".join(FULL) + f"+{method}16_pred", n_features=500 + D,
               model="lgbm", objective="binary", folds=str(OUTER_F), random_seed=0,
               train_series=int((~np.isin(d.series_fold, [-1])).sum()), train_rows=len(tr_rows),
               mean_oof_ts_auc=s, pooled_oof_ts_auc=s, per_fold_ts_auc=f"{s:.5f}", fold_std=0.0,
               persistence="none", sample_mode="uniform", training_runtime_s=round(time.time() - t0, 1),
               causal_verified="nested double-cross-fit; no target-availability feature",
               test_reduced_touched="no", lockbox_touched="no", status="recorded",
               notes="Wave 8 PCFB-C pilot. Pre-registered research/WAVE8_FUTURE_AWARE_PREREG.md.",
               protocol="pilot_fold0_only")
    PL.append_result(res)


# ---------------------------------------------------------------------------
def analyze(method):
    c = Ctx()
    y, t, age, sidx = c.d.y, c.d.t, c.age, c.d.sidx
    r0 = c.rows[OUTER_F]
    cellmask0 = cell_mask(y[r0], t[r0], age[r0])

    A = np.load(f"{OOFDIR}/RT-990.npy")
    exp_b = "RT-1030" if method == "pca" else "RT-1032"
    exp_c = "RT-1031" if method == "pca" else "RT-1033"
    B = np.load(f"{OOFDIR}/{exp_b}.npy")
    Cc = np.load(f"{OOFDIR}/{exp_c}.npy")
    elig = ~np.isnan(B[r0])

    def cell(v, mask=None):
        m = cellmask0.copy() if mask is None else (cellmask0 & mask)
        rr = r0[m]
        return score_on(v[rr], np.ones(len(rr), bool), y[rr], t[rr]) if len(rr) else None

    dAB = cell(B, elig) - cell(A, elig)
    dAC_full = cell(Cc) - cell(A)
    dAC_elig = cell(Cc, elig) - cell(A, elig)
    ens = W8.ensemble_marginal(Cc, c=c, fold=OUTER_F, label=f"pcfb_{method}")

    out = {"method": method, "eligible_fraction_fold0": float(elig.mean()),
           "A_control_cell": cell(A), "A_control_cell_eligible": cell(A, elig),
           "B_oracle_cell_eligible": cell(B, elig),
           "C_legal_cell_full": cell(Cc), "C_legal_cell_eligible": cell(Cc, elig),
           "delta_oracle_minus_control": dAB, "delta_legal_minus_control_full": dAC_full,
           "delta_legal_minus_control_eligible": dAC_elig,
           "retention_R": (dAC_elig / dAB) if dAB else None, "ensemble": ens}
    print(json.dumps(out, indent=2))
    with open(f"{REPORTS}/wave8_pcfb_{method}.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def compare_methods():
    pca = analyze("pca")
    pls = analyze("pls")
    gate_a = pls["delta_legal_minus_control_eligible"] is not None and \
        (pls["C_legal_cell_eligible"] - pca["C_legal_cell_eligible"]) >= 0.004
    gate_b = pls["ensemble"]["marginal_vs_clone"] >= 0.002
    out = {"pca16": pca, "pls16": pls, "pls_minus_pca_legal_cell_eligible":
           pls["C_legal_cell_eligible"] - pca["C_legal_cell_eligible"],
           "continuation_gate_cleared": bool(gate_a and gate_b)}
    with open(f"{REPORTS}/wave8_pcfb.json", "w") as f:
        json.dump(out, f, indent=2)
    md = ["# WAVE 8 -- PCFB PILOT RESULT (fold 0, h=200, d=16)\n",
          "| | PCA16 | PLS16 |", "|---|---:|---:|",
          f"| oracle cell (eligible) | {pca['B_oracle_cell_eligible']:.5f} | {pls['B_oracle_cell_eligible']:.5f} |",
          f"| legal cell (eligible) | {pca['C_legal_cell_eligible']:.5f} | {pls['C_legal_cell_eligible']:.5f} |",
          f"| legal cell (full pop) | {pca['C_legal_cell_full']:.5f} | {pls['C_legal_cell_full']:.5f} |",
          f"| retention R | {pca['retention_R']} | {pls['retention_R']} |",
          f"| RT600+cand marginal vs clone | {pca['ensemble']['marginal_vs_clone']:+.5f} | "
          f"{pls['ensemble']['marginal_vs_clone']:+.5f} |",
          f"\nPLS16 legal − PCA16 legal (eligible cell): {out['pls_minus_pca_legal_cell_eligible']:+.5f}\n",
          f"\n## CONTINUATION GATE: {'CLEARED' if out['continuation_gate_cleared'] else 'NOT CLEARED'}\n"]
    with open(f"{REPORTS}/wave8_pcfb.md", "w") as f:
        f.write("\n".join(md))
    print(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--arm", choices=["b", "c"])
    ap.add_argument("--method", choices=["pca", "pls"])
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.fit:
        cmd_fit()
    elif args.arm == "b":
        run_arm_b(args.method)
    elif args.arm == "c":
        run_arm_c(args.method)
    elif args.compare:
        compare_methods()
    elif args.analyze:
        analyze(args.method)
    else:
        raise SystemExit("pass --fit, --arm b|c --method pca|pls, --analyze --method .., or --compare")


if __name__ == "__main__":
    main()
