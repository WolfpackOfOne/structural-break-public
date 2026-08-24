"""M1 -- Structural Successor Targets (SST). Fold-0 pilot.

Pre-registered in research/WAVE8_FUTURE_AWARE_PREREG.md section 5.1 BEFORE
this ran. Three arms, matched rows/capacity to RT-990/RT-991 (wave7_d3r):

    A  control:  500 causal columns alone            -- reused RT-990 fold-0 slice
    B  oracle:   500 + TRUE future structural targets  -- offline diagnostic only,
                 restricted to rows with t+h support (never imputed/masked)
    C  legal:    500 + nested-OOF PREDICTED structural targets -- causal at
                 inference, full row population

8 channels (wave8_common.SST_CHANNELS) x horizons {50,100,200}, combined (24
target columns) as the primary arms (RT-1006 legal / RT-1007 oracle);
per-horizon arms (RT-1000-1005) are the same machinery restricted to one h.

Usage:
    python wave8_sst.py --eligibility                # already run, see reports/
    python wave8_sst.py --targets                    # build+cache true target matrix
    python wave8_sst.py --arm b --horizons combined   # RT-1007
    python wave8_sst.py --arm c --horizons combined   # RT-1006
    python wave8_sst.py --arm b --horizons 50         # RT-1000
    python wave8_sst.py --arm c --horizons 50         # RT-1003
    python wave8_sst.py --analyze
"""
from __future__ import annotations

import argparse, json, time

import numpy as np

import wave8_common as W8
import sbr.pipeline as PL
from wave5_lib import Ctx, FOLDS, OOFDIR, REPORTS, ts_auc_flat
from wave7_d3r import FULL, ARM_B_PARAMS, cell_mask, score_on, bucket

OUTER_F = 0
MAX_TRAIN_ROWS = 1_000_000
CACHE = f"{OOFDIR}/wave8_sst"
import os
os.makedirs(CACHE, exist_ok=True)

CHANNELS = list(W8.SST_CHANNELS.items())  # [(col, module), ...] in fixed order


def channel_global_index(names):
    idx = []
    for col, module in CHANNELS:
        key = f"{module}::{col}"
        assert key in names, f"SST channel column missing from feature bank: {key}"
        idx.append(names.index(key))
    return np.array(idx, dtype=np.int64)


def horizon_set(spec):
    return list(W8.SST_HORIZONS) if spec == "combined" else [int(spec)]


def target_tag(spec):
    return "combined" if spec == "combined" else str(spec)


# ---------------------------------------------------------------------------
def build_true_targets(d, mats, names, first_row, series_len):
    """Cache Y[h] (n_rows, 8) NaN-padded + valid[h] (n_rows,) to disk once."""
    chan_idx = channel_global_index(names)
    for h in W8.SST_HORIZONS:
        yp, vp = f"{CACHE}/Y_h{h}.npy", f"{CACHE}/valid_h{h}.npy"
        if os.path.exists(yp) and os.path.exists(vp):
            continue
        fut, valid = W8.future_row_index(d, h, first_row, series_len)
        Y = np.full((len(d.y), len(CHANNELS)), np.nan, dtype=np.float32)
        valid_rows = np.flatnonzero(valid)
        block = PL._stack(mats, names, fut[valid_rows], chan_idx)
        Y[valid_rows] = block
        np.save(yp, Y)
        np.save(vp, valid)
        print(f"  h={h}: {valid.mean():.3f} rows eligible, target matrix cached -> {yp}")


def load_true_targets(horizons):
    Ys, Vs = [], []
    for h in horizons:
        Ys.append(np.load(f"{CACHE}/Y_h{h}.npy"))
        Vs.append(np.load(f"{CACHE}/valid_h{h}.npy"))
    Y = np.concatenate(Ys, axis=1)               # (n_rows, 8*len(horizons))
    valid_any_h = np.stack(Vs, axis=1)            # (n_rows, len(horizons)) per-horizon validity
    # a row is in the ORACLE population only if EVERY requested horizon is eligible
    valid_all = valid_any_h.all(axis=1)
    return Y, valid_all, Vs


# ---------------------------------------------------------------------------
def train_augmented(d, mats, names, keep_idx, tr_rows, va_rows, extra):
    import lightgbm as lgb
    p = dict(ARM_B_PARAMS)
    n_round = p.pop("n_estimators")
    Xtr = np.concatenate([PL._stack(mats, names, tr_rows, keep_idx), extra[tr_rows]], axis=1)
    ytr = d.y[tr_rows]
    ds = lgb.Dataset(Xtr, label=ytr, params=dict(p, objective="binary"),
                     feature_name=[f"f{i}" for i in range(Xtr.shape[1])])
    booster = lgb.train(dict(p), ds, num_boost_round=n_round)
    del Xtr, ds
    Xva = np.concatenate([PL._stack(mats, names, va_rows, keep_idx), extra[va_rows]], axis=1)
    pred = booster.predict(Xva).astype(np.float32)
    del Xva
    return pred


def run_arm_b(spec):
    """Oracle: 500 causal + TRUE targets, restricted to the eligible population."""
    d = PL.Data()
    mats, names = PL.load_features(FULL)
    keep_idx = np.arange(len(names))
    horizons = horizon_set(spec)
    Y, valid_all, _ = load_true_targets(horizons)
    W8.assert_no_forbidden_columns(names)

    tr_rows_all = d.rows_for([g for g in FOLDS if g != OUTER_F])
    va_rows_all = d.rows_for([OUTER_F])
    tr_rows = tr_rows_all[valid_all[tr_rows_all]]
    va_rows = va_rows_all[valid_all[va_rows_all]]
    rng = np.random.default_rng(0)
    if len(tr_rows) > MAX_TRAIN_ROWS:
        tr_rows = np.sort(rng.choice(tr_rows, MAX_TRAIN_ROWS, replace=False))
    print(f"[SST-B {spec}] eligible train rows {len(tr_rows)}/{len(tr_rows_all)}, "
          f"eligible val rows {len(va_rows)}/{len(va_rows_all)}")

    t0 = time.time()
    Yf = np.nan_to_num(Y, nan=0.0)
    pred = train_augmented(d, mats, names, keep_idx, tr_rows, va_rows, Yf)
    s = float(ts_auc_flat(pred, d.y[va_rows], d.t[va_rows]))
    oof = np.full(len(d.y), np.nan, dtype=np.float32)
    oof[va_rows] = pred
    exp_id = "RT-1007" if spec == "combined" else f"RT-100{ [50,100,200].index(int(spec)) }"
    np.save(f"{OOFDIR}/{exp_id}.npy", oof)
    print(f"[SST-B {spec}] TS-AUC (eligible-only fold0) {s:.5f}  runtime {time.time()-t0:.1f}s -> {exp_id}")

    res = dict(experiment_id=exp_id, date=time.strftime("%Y-%m-%d %H:%M"), git_sha=PL.git_sha(),
               agent="agent0",
               hypothesis=f"SST-B oracle ({spec}): 500 causal cols + TRUE future structural "
                          f"targets at h={horizons}, restricted to eligible rows. OFFLINE DIAGNOSTIC ONLY.",
               falsification_condition="see research/WAVE8_FUTURE_AWARE_PREREG.md section 5.1",
               feature_set=",".join(FULL) + f"+sst_true_{target_tag(spec)}",
               n_features=500 + Y.shape[1], model="lgbm", objective="binary",
               folds=str(OUTER_F), random_seed=0,
               train_series=int((~np.isin(d.series_fold, [-1])).sum()), train_rows=len(tr_rows),
               mean_oof_ts_auc=s, pooled_oof_ts_auc=s, per_fold_ts_auc=f"{s:.5f}", fold_std=0.0,
               persistence="none", sample_mode="uniform",
               training_runtime_s=round(time.time() - t0, 1),
               causal_verified="OFFLINE DIAGNOSTIC ONLY -- true future lookup is NOT causal, NOT deployable",
               test_reduced_touched="no", lockbox_touched="no", status="recorded",
               notes=f"Wave 8 SST-B pilot. eligible_train={len(tr_rows)}/{len(tr_rows_all)} "
                     f"eligible_val={len(va_rows)}/{len(va_rows_all)}. Pre-registered "
                     f"research/WAVE8_FUTURE_AWARE_PREREG.md.",
               protocol="pilot_fold0_only")
    PL.append_result(res)
    return s, va_rows


def run_arm_c(spec):
    """Legal: 500 causal + nested-OOF PREDICTED targets, full row population."""
    d = PL.Data()
    mats, names = PL.load_features(FULL)
    keep_idx = np.arange(len(names))
    horizons = horizon_set(spec)
    Y, valid_all, valid_per_h = load_true_targets(horizons)
    W8.assert_no_forbidden_columns(names)

    n_targets = Y.shape[1]
    Zhat = np.full((len(d.y), n_targets), np.nan, dtype=np.float32)
    tr_rows_all = d.rows_for([g for g in FOLDS if g != OUTER_F])
    va_rows_all = d.rows_for([OUTER_F])
    dev_rows = np.concatenate([tr_rows_all, va_rows_all])
    t0 = time.time()
    col = 0
    for hi, h in enumerate(horizons):
        vmask = valid_per_h[hi]
        for ci, (chan, module) in enumerate(CHANNELS):
            ckpt = f"{CACHE}/zhat_{spec}_h{h}_{chan}.npy"
            if os.path.exists(ckpt):
                Zhat[:, col] = np.load(ckpt)
                print(f"  [SST-C {spec}] h={h} channel={chan}: loaded from checkpoint "
                      f"({time.time()-t0:.0f}s elapsed)", flush=True)
                col += 1
                continue
            y_full = np.nan_to_num(Y[:, col], nan=0.0).astype(np.float64)
            z_inner = W8.nested_oof_regressor(OUTER_F, d, mats, names, keep_idx, y_full, vmask,
                                              rounds=150, max_rows=300_000, seed=0)
            z_val = W8.full_predict_for_outer_val(OUTER_F, d, mats, names, keep_idx, y_full, vmask,
                                                  rounds=150, max_rows=300_000, seed=0)
            Zhat[tr_rows_all, col] = z_inner[tr_rows_all]
            Zhat[va_rows_all, col] = z_val[va_rows_all]
            np.save(ckpt, Zhat[:, col])          # per-channel checkpoint: a crash costs one channel, not all 24
            print(f"  [SST-C {spec}] h={h} channel={chan}: nested+refit done "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)
            col += 1
    assert not np.isnan(Zhat[dev_rows]).any(), \
        "SST-C predicted target matrix has uncovered DEV rows (lockbox rows are legitimately untouched)"
    rng = np.random.default_rng(0)
    tr_rows = tr_rows_all
    if len(tr_rows) > MAX_TRAIN_ROWS:
        tr_rows = np.sort(rng.choice(tr_rows, MAX_TRAIN_ROWS, replace=False))

    t1 = time.time()
    pred = train_augmented(d, mats, names, keep_idx, tr_rows, va_rows_all, Zhat)
    s = float(ts_auc_flat(pred, d.y[va_rows_all], d.t[va_rows_all]))
    oof = np.full(len(d.y), np.nan, dtype=np.float32)
    oof[va_rows_all] = pred
    exp_id = "RT-1006" if spec == "combined" else f"RT-100{ [50,100,200].index(int(spec))+3 }"
    np.save(f"{OOFDIR}/{exp_id}.npy", oof)
    print(f"[SST-C {spec}] TS-AUC (full fold0) {s:.5f}  classifier runtime {time.time()-t1:.1f}s  "
          f"total runtime {time.time()-t0:.1f}s -> {exp_id}")

    res = dict(experiment_id=exp_id, date=time.strftime("%Y-%m-%d %H:%M"), git_sha=PL.git_sha(),
               agent="agent0",
               hypothesis=f"SST-C legal ({spec}): 500 causal cols + nested-OOF double-cross-fitted "
                          f"predicted future structural targets at h={horizons}. Causal at inference.",
               falsification_condition="see research/WAVE8_FUTURE_AWARE_PREREG.md section 5.1",
               feature_set=",".join(FULL) + f"+sst_pred_{target_tag(spec)}",
               n_features=500 + n_targets, model="lgbm", objective="binary",
               folds=str(OUTER_F), random_seed=0,
               train_series=int((~np.isin(d.series_fold, [-1])).sum()), train_rows=len(tr_rows),
               mean_oof_ts_auc=s, pooled_oof_ts_auc=s, per_fold_ts_auc=f"{s:.5f}", fold_std=0.0,
               persistence="none", sample_mode="uniform",
               training_runtime_s=round(time.time() - t0, 1),
               causal_verified="nested double-cross-fit (wave8_common.nested_oof_regressor); "
                               "no target-availability feature; fold-purity sentinel passed",
               test_reduced_touched="no", lockbox_touched="no", status="recorded",
               notes="Wave 8 SST-C pilot, fold 0. Pre-registered research/WAVE8_FUTURE_AWARE_PREREG.md.",
               protocol="pilot_fold0_only")
    PL.append_result(res)
    return s, va_rows_all


# ---------------------------------------------------------------------------
def analyze(spec="combined"):
    c = Ctx()
    y, t, age, sidx = c.d.y, c.d.t, c.age, c.d.sidx
    neg_is_prebreak = c.has_break[sidx] & (y == 0)
    r0 = c.rows[OUTER_F]
    cellmask0 = cell_mask(y[r0], t[r0], age[r0])

    A = np.load(f"{OOFDIR}/RT-990.npy")
    exp_b = "RT-1007" if spec == "combined" else f"RT-100{[50,100,200].index(int(spec))}"
    exp_c = "RT-1006" if spec == "combined" else f"RT-100{[50,100,200].index(int(spec))+3}"
    B = np.load(f"{OOFDIR}/{exp_b}.npy")
    Cc = np.load(f"{OOFDIR}/{exp_c}.npy")

    elig = ~np.isnan(B[r0])
    def whole(v, mask=None):
        m = np.ones(len(r0), bool) if mask is None else mask
        rr = r0[m]
        return score_on(v[rr], np.ones(len(rr), bool), y[rr], t[rr])
    def cell(v, mask=None):
        m = cellmask0.copy() if mask is None else (cellmask0 & mask)
        rr = r0[m]
        if len(rr) == 0:
            return None
        return score_on(v[rr], np.ones(len(rr), bool), y[rr], t[rr])

    out = {
        "spec": spec, "outer_fold": OUTER_F,
        "eligible_fraction_fold0": float(elig.mean()),
        "A_control_whole": whole(A), "A_control_whole_eligible_only": whole(A, elig),
        "A_control_cell": cell(A), "A_control_cell_eligible_only": cell(A, elig),
        "B_oracle_whole_eligible_only": whole(B, elig),
        "B_oracle_cell_eligible_only": cell(B, elig),
        "C_legal_whole": whole(Cc), "C_legal_whole_eligible_only": whole(Cc, elig),
        "C_legal_cell": cell(Cc), "C_legal_cell_eligible_only": cell(Cc, elig),
    }
    dAB_cell = out["B_oracle_cell_eligible_only"] - out["A_control_cell_eligible_only"]
    dAC_cell = out["C_legal_cell"] - out["A_control_cell"]
    dAC_cell_elig = out["C_legal_cell_eligible_only"] - out["A_control_cell_eligible_only"]
    out["delta_oracle_minus_control_cell_eligible_pop"] = dAB_cell
    out["delta_legal_minus_control_cell_full_pop"] = dAC_cell
    out["delta_legal_minus_control_cell_eligible_pop"] = dAC_cell_elig
    out["retention_R_eligible_pop"] = (dAC_cell_elig / dAB_cell) if dAB_cell not in (0, None) and dAB_cell != 0 else None

    ens = W8.ensemble_marginal(Cc, c=c, fold=OUTER_F, label=f"sst_{spec}")
    out["ensemble"] = ens

    gate_a = (dAC_cell is not None and dAC_cell >= 0.003)
    gate_b = ens["marginal_vs_clone"] >= 0.0015
    out["continuation_gate_cleared"] = bool(gate_a or gate_b)

    print(json.dumps(out, indent=2))
    with open(f"{REPORTS}/wave8_sst.json", "w") as f:
        json.dump(out, f, indent=2)
    md = [f"# WAVE 8 -- SST ({spec}) PILOT RESULT (fold 0)\n",
          f"Eligible-population fraction of fold 0: **{out['eligible_fraction_fold0']:.3f}** "
          f"(oracle/retention numbers below are computed ONLY on this subpopulation; "
          f"eligibility diagnostic showed material class-conditional gaps, see "
          f"research/reports/wave8_eligibility_diagnostic.json -- this is expected and handled "
          f"by restricting the population, never by adding an eligibility feature).\n",
          "| arm | whole fold0 | dominant cell | population |",
          "|---|---:|---:|---|",
          f"| A control (`RT-990`) | {out['A_control_whole']:.5f} | {out['A_control_cell']:.5f} | full |",
          f"| A control, eligible-only | {out['A_control_whole_eligible_only']:.5f} | {out['A_control_cell_eligible_only']:.5f} | eligible |",
          f"| B oracle (`{exp_b}`) | {out['B_oracle_whole_eligible_only']:.5f} | {out['B_oracle_cell_eligible_only']:.5f} | eligible |",
          f"| C legal (`{exp_c}`) | {out['C_legal_whole']:.5f} | {out['C_legal_cell']:.5f} | full |",
          f"| C legal, eligible-only | {out['C_legal_whole_eligible_only']:.5f} | {out['C_legal_cell_eligible_only']:.5f} | eligible |",
          f"\n**Oracle utility (B-A, eligible pop, dominant cell):** {dAB_cell:+.5f}\n",
          f"\n**Legal utility (C-A, dominant cell):** full pop {dAC_cell:+.5f}, eligible pop {dAC_cell_elig:+.5f}\n",
          f"\n**Retention R = legal/oracle (eligible pop):** {out['retention_R_eligible_pop']}\n",
          f"\n## Ensemble marginal (fold 0)\n", "```json", json.dumps(ens, indent=2), "```",
          f"\n## CONTINUATION GATE: {'CLEARED' if out['continuation_gate_cleared'] else 'NOT CLEARED'}\n"]
    with open(f"{REPORTS}/wave8_sst.md", "w") as f:
        f.write("\n".join(md))
    print(f"\nwrote {REPORTS}/wave8_sst.{{md,json}}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", action="store_true")
    ap.add_argument("--arm", choices=["b", "c"])
    ap.add_argument("--horizons", default="combined")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.targets:
        d = PL.Data()
        mats, names = PL.load_features(FULL)
        first_row, series_len = W8.row_layout(d)
        build_true_targets(d, mats, names, first_row, series_len)
    elif args.arm == "b":
        run_arm_b(args.horizons)
    elif args.arm == "c":
        run_arm_c(args.horizons)
    elif args.analyze:
        analyze(args.horizons)
    else:
        raise SystemExit("pass --targets, --arm b|c [--horizons combined|50|100|200], or --analyze")


if __name__ == "__main__":
    main()
