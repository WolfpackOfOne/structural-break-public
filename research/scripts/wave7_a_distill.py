"""W7-A1 -- LANE A PILOT: full-sequence privileged teacher -> causal student.

ONE CANONICAL FOLD.  Three arms, declared in research/WAVE7_PREREG.md before any
score.  No target-weight sweep (brief section 13).

    A0  control   ordinary binary LightGBM, champion params, 500 causal columns
    A1  student   identical capacity, target = teacher `evidence` path
    A2  student   identical capacity, target = 0.5*y + 0.5*evidence

Everything except the TARGET is held byte-identical across arms: same rows, same
columns, same folds, same seed, same learner configuration, same scorer.

CONTINUATION BAR (pre-registered): >= +0.003 TS-AUC over A0 on the pilot fold,
or >= +0.005 in ages 0-20 with credible aggregate potential.  +0.0015 kills it.

FOLD PURITY.  Teacher targets are computed from the FULL series of TRAINING
series only.  The validation fold's series never reach `build_teacher_row_targets`
-- asserted, not assumed.  At inference the teacher does not exist: `--prove-deletion`
trains a student, deletes every teacher artifact, and asserts the predictions are
bit-identical.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wave7_lib as W  # noqa: E402

sys.path.insert(0, f"{W.ROOT}/src")
sys.path.insert(0, f"{W.ROOT}/research/scripts")

#: EVERY arm trains with `cross_entropy`, not `binary`.  On a hard 0/1 label the
#: two are bit-identical, so A0 is still exactly the champion's objective; on a
#: soft label `binary` throws the target away.  Holding one objective across all
#: arms is also what makes the deltas attributable to the TARGET alone.
OBJECTIVE = W.SOFT_LABEL_OBJECTIVE

ARMS = {
    "A0": {"target": "binary", "hypothesis": "control"},
    "A1": {"target": "evidence", "hypothesis": "the graded oracle-evidence path is a "
           "better-conditioned ranking target than the hard step label"},
    "A2": {"target": "half", "hypothesis": "the hard label and the evidence path carry "
           "complementary supervision"},
}


def _targets(y, tea, kind):
    if kind == "binary":
        return y.astype(np.float32)
    if kind == "evidence":
        return tea["evidence"].astype(np.float32)
    if kind == "half":
        return (0.5 * y + 0.5 * tea["evidence"]).astype(np.float32)
    raise KeyError(kind)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max-train-rows", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="A0,A1,A2")
    ap.add_argument("--prove-deletion", action="store_true")
    ap.add_argument("--screen", action="store_true",
                    help="reduced protocol / synthetic smoke store -- NEVER a research number")
    a = ap.parse_args()

    store = W.require_store(a.screen)            # hard-stops here when the store is absent
    import lightgbm as lgb
    from sbr.pipeline import Data, _stack, load_features

    d = Data(screen=a.screen)
    mats, names = load_features(W.PROD_MODULES, screen=a.screen)
    W.assert_no_teacher_in_features(names)
    W.assert_causal_names(names)
    keep_idx = np.arange(len(names))

    tr_folds = [x for x in (0, 1, 2, 3, 4) if x != a.fold]
    tr_rows = d.rows_for(tr_folds)
    va_rows = d.rows_for([a.fold])
    rng = np.random.default_rng(a.seed)
    if len(tr_rows) > a.max_train_rows:
        tr_rows = np.sort(rng.choice(tr_rows, a.max_train_rows, replace=False))

    # ---- teacher: TRAINING SERIES ONLY, asserted -----------------------------
    tr_series = np.unique(d.sidx[tr_rows])
    va_series = np.unique(d.sidx[va_rows])
    assert not np.intersect1d(tr_series, va_series).size, "teacher touched a validation series"
    assert not np.isin(d.series_fold[tr_series], [-1, a.fold]).any(), "teacher touched lockbox/val fold"

    print(f"building teacher targets on {len(tr_series)} training series ...", flush=True)
    tea_by_series = W.build_teacher_row_targets(store, tr_series, which=("evidence",))
    # map flat per-series concatenation onto the sampled training rows
    off = {}
    p = 0
    for i in tr_series:
        n = int(store.meta.n_online.iloc[int(i)])
        off[int(i)] = (p, n)
        p += n
    ev = np.empty(len(tr_rows), np.float32)
    srow = d.t[tr_rows]
    for k, r in enumerate(tr_rows):
        s0, _ = off[int(d.sidx[r])]
        ev[k] = tea_by_series["evidence"][s0 + int(srow[k])]
    tea = {"evidence": ev}
    print(f"teacher evidence: mean {ev.mean():.4f}  on positives "
          f"{ev[d.y[tr_rows] == 1].mean():.4f}  on negatives {ev[d.y[tr_rows] == 0].mean():.4f}")

    Xtr = _stack(mats, names, tr_rows, keep_idx)
    Xva = _stack(mats, names, va_rows, keep_idx)
    yva, tva = d.y[va_rows], d.t[va_rows]
    tau = store.meta.tau_index.to_numpy()[d.sidx[va_rows]]
    age = np.where(yva == 1, tva - tau, -1)

    res = {"schema": "wave7_a_distill/2", "objective": OBJECTIVE,
           "protocol": "screen" if a.screen else "full", "fold": a.fold, "seed": a.seed,
           "n_train_rows": int(len(tr_rows)), "n_valid_rows": int(len(va_rows)),
           "n_features": len(names), "params": W.CHAMP_PARAMS, "arms": {}}

    preds = {}
    for arm in a.arms.split(","):
        spec = ARMS[arm]
        p = dict(W.CHAMP_PARAMS)
        n_round = int(p.pop("n_estimators"))
        lab = _targets(d.y[tr_rows], tea, spec["target"])
        p["objective"] = OBJECTIVE
        W.assert_objective_matches_label(OBJECTIVE, lab)
        ds = lgb.Dataset(Xtr, label=lab, params=p,
                         feature_name=[f"f{i}" for i in range(len(names))])
        b = lgb.train(p, ds, num_boost_round=n_round)
        pr = b.predict(Xva).astype(np.float32)
        preds[arm] = pr
        rep = W.bucket_report(pr, yva, tva, age)
        rep["hypothesis"] = spec["hypothesis"]
        rep["target"] = spec["target"]
        res["arms"][arm] = rep
        print(f"{arm}: TS-AUC {rep['aggregate']:.5f}   {spec['target']}", flush=True)

    if "A0" in preds:
        # Parity gate.  If `cross_entropy` ever stops reproducing `binary` on the
        # hard label, A0 has stopped being the champion's control and every delta
        # below is measuring an objective change instead of a target change.
        pb = dict(W.CHAMP_PARAMS)
        nb = int(pb.pop("n_estimators"))
        pb["objective"] = "binary"
        hard = d.y[tr_rows].astype(np.float32)
        bb = lgb.train(pb, lgb.Dataset(Xtr, label=hard, params=pb,
                                       feature_name=[f"f{i}" for i in range(len(names))]),
                       num_boost_round=nb)
        pbin = bb.predict(Xva).astype(np.float32)
        gap = float(abs(W.bucket_report(pbin, yva, tva, age)["aggregate"]
                        - res["arms"]["A0"]["aggregate"]))
        res["objective_parity"] = {"binary_vs_cross_entropy_abs_gap": gap,
                                   "tol": 1e-4, "pass": gap <= 1e-4}
        print(f"objective parity binary vs {OBJECTIVE}: {gap:.2e} "
              f"{'PASS' if gap <= 1e-4 else 'FAIL'}")

        base = res["arms"]["A0"]["aggregate"]
        for arm in res["arms"]:
            res["arms"][arm]["delta_vs_A0"] = res["arms"][arm]["aggregate"] - base
            res["arms"][arm]["verdict"] = (
                "CONTINUE" if res["arms"][arm]["delta_vs_A0"] >= 0.003 else "KILL")

    if a.prove_deletion:
        # The production test the brief's section 9 requires: the student must be
        # a pure function of causal state.  Delete every teacher artifact and
        # re-predict from the trained booster; the bytes must not move.
        h_before = W.sha_array(preds[a.arms.split(",")[-1]])
        del tea, tea_by_series, ev
        import gc
        gc.collect()
        pr2 = b.predict(Xva).astype(np.float32)
        h_after = W.sha_array(pr2)
        res["teacher_deletion_test"] = {"sha_before": h_before, "sha_after": h_after,
                                        "identical": h_before == h_after}
        print(f"teacher-deletion test: {'PASS' if h_before == h_after else 'FAIL'}")

    print("written:", W.write_report(f"wave7_a_distill_fold{a.fold}{'_screen' if a.screen else ''}", res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
