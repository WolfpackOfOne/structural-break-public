"""FINAL FIT — all 10,000 labelled series.  Runs only after the architecture freeze.

Two phases, in order:

  oof   cross-fitted OOF for each stream under research/folds/folds_final10k.parquet
        (5 folds over ALL 10,000 series).  These score distributions are what the
        calibration grids are estimated from.

  fit   the deployed boosters, each fitted on ALL 10,000 series, plus the
        calibration payload built from the phase-1 OOF, plus the manifest.

WHY THE OOF PASS EXISTS.  Every OOF vector the project owns covers only the
8,000 dev series.  Fitting the final calibration from those while the boosters
see 10,000 would widen the training-regime mismatch that the deployable stack
already carries.  Running the OOF pass under the same 10,000-series partition
narrows it to "grids from 8,000-series models, boosters from 10,000-series
models", which is strictly better than the wave-2 artifact's "grids from
6,400-series models, boosters from 8,000-series models".  It is not zero and
research/RDOF_LEDGER.md says so.

ROW BUDGET.  Pre-declared before this script existed: 1.25x the dev budget,
rounded to the nearest thousand, applied identically in both phases.  It is read
from FINAL_ROWS below and is not a command-line argument, because a row budget
that can be passed in is a row budget that can be tuned after freeze.

NOTHING HERE IS EVIDENCE FOR A MODEL CHOICE.  By the time it runs there are none
left to make.  No number it prints may be used to revisit the architecture.
"""
from __future__ import annotations

import argparse, contextlib, hashlib, json, os, subprocess, sys, time

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

import numpy as np
import sbr.pipeline as PL
from sbr.pipeline import Data, load_features, _stack, _make_pairwise_t, run
from sbr.production.calibration import DEFAULT_COORD, SmoothTimeCDFCal
from sbr.stream.engine import StreamEngine
from wave2_train_ensemble import DEFAULT, STREAMS

FINAL_FOLDS = f"{ROOT}/research/folds/folds_final10k.parquet"
SERIES_RATIO = 10000 / 8000

#: stream -> final-fit experiment id for its 10k cross-fitted OOF
FINAL_OOF_ID = {
    "RT-100R": "RT-500", "RT-120R": "RT-501", "RT-121R": "RT-502",
    "RT-122R": "RT-503", "RT-123R": "RT-504", "RT-124R": "RT-505",
    "RT-125R": "RT-506",
}
ORDER = list(STREAMS)


def final_rows(dev_rows: int) -> int:
    """The pre-declared budget rule.  1.25x, to the nearest thousand."""
    return int(round(dev_rows * SERIES_RATIO / 1000.0) * 1000)


@contextlib.contextmanager
def final_folds():
    """Point the pipeline at the 10,000-series partition, then put it back."""
    assert os.path.exists(FINAL_FOLDS), f"missing {FINAL_FOLDS}"
    old = PL.FOLDS
    PL.FOLDS = FINAL_FOLDS
    try:
        yield
    finally:
        PL.FOLDS = old


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def phase_oof(only):
    with final_folds():
        for exp in (only or ORDER):
            cfg = STREAMS[exp]
            eid = FINAL_OOF_ID[exp]
            if os.path.exists(f"{ROOT}/research/oof/{eid}.npy"):
                print(f"{eid}: already on disk, keeping"); continue
            run(exp_id=eid, modules=cfg["modules"], agent="claude-wave4-final",
                seed=cfg["seed"], params=dict(cfg["params"]),
                max_train_rows=final_rows(cfg["max_train_rows"]),
                sample_mode=cfg.get("sample_mode", "uniform"), folds=(0, 1, 2, 3, 4),
                hypothesis=f"FINAL-FIT cross-fitted OOF for {exp} over all 10,000 labelled "
                           f"series, to build the deployed calibration payload",
                falsification="n/a -- FINAL FIT, not a candidate. No model choice may be "
                              "revisited on the basis of this row.",
                notes=f"FINAL-FIT OOF, folds_final10k (all 10,000 series incl. the spent "
                      f"lockbox), row budget {final_rows(cfg['max_train_rows'])} "
                      f"(= 1.25x dev, pre-declared). alias of {exp}.")


def phase_fit(out, only):
    os.makedirs(out, exist_ok=True)
    with final_folds():
        d = Data()
    mats, names = load_features(list(STREAMS["RT-100R"]["modules"]))
    eng = StreamEngine(list(STREAMS["RT-100R"]["modules"])).fit_historical(
        np.arange(1200, dtype=np.float64) % 7 - 3.0)
    man_eng = eng.manifest()
    assert man_eng["columns"] == names, "streaming column order != batch column order"
    col_index = {c: i for i, c in enumerate(names)}

    allrows = d.rows_for([0, 1, 2, 3, 4])
    assert len(np.unique(d.sidx[allrows])) == 10000, "final fit must see all 10,000 series"
    print(f"final fit over {len(allrows)} online rows / 10,000 series")

    import lightgbm as lgb
    slices, cals, cfgs = [], [], []
    for k, exp in enumerate(ORDER):
        cfg = STREAMS[exp]
        sel = np.array([col_index[c] for c in names if c.split("::")[0] in set(cfg["modules"])])
        slices.append(sel.tolist())
        budget = final_rows(cfg["max_train_rows"])

        oof_path = f"{ROOT}/research/oof/{FINAL_OOF_ID[exp]}.npy"
        assert os.path.exists(oof_path), f"run phase 'oof' first: missing {oof_path}"
        oof = np.load(oof_path)
        cals.append(SmoothTimeCDFCal.fit(oof[allrows], d.t[allrows], time_coord=DEFAULT_COORD))

        cfgs.append({"experiment": exp, "final_oof_id": FINAL_OOF_ID[exp],
                     "modules": cfg["modules"], "seed": cfg["seed"],
                     "n_columns": int(len(sel)), "params": cfg["params"],
                     "max_train_rows": budget, "dev_max_train_rows": cfg["max_train_rows"],
                     "sample_mode": cfg.get("sample_mode", "uniform")})

        path = os.path.join(out, f"model.txt.{k}")
        if only and exp not in only:
            print(f"skip {exp}"); continue
        if os.path.exists(path):
            print(f"{exp}: already trained, keeping {path}"); continue

        t0 = time.time()
        p = dict(DEFAULT); p.update(cfg["params"])
        n_round = int(p.pop("n_estimators"))
        rng = np.random.default_rng(cfg["seed"])
        rows = allrows
        if len(rows) > budget:
            if cfg.get("sample_mode") == "per_series":
                s = d.sidx[rows]
                o = np.argsort(s, kind="stable"); rows = rows[o]; s = s[o]
                bnd = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
                ends = np.r_[bnd[1:], len(s)]
                per = budget // len(bnd)
                pick = [np.arange(b, e) if e - b <= per
                        else rng.choice(np.arange(b, e), per, replace=False)
                        for b, e in zip(bnd, ends)]
                rows = np.sort(rows[np.concatenate(pick)])
            else:
                rows = np.sort(rng.choice(rows, budget, replace=False))
        X = _stack(mats, names, rows, sel)
        y = d.y[rows]
        if p.get("objective") == "pairwise_t":
            p["objective"] = _make_pairwise_t(d.t[rows], y, seed=cfg["seed"])
        ds = lgb.Dataset(X, label=y, params=dict(p, objective="binary"))
        b = lgb.train(p, ds, num_boost_round=n_round)
        del X, ds
        b.save_model(path)
        print(f"{exp}: {len(rows)} rows x {len(sel)} cols -> {path} "
              f"({time.time()-t0:.0f}s)", flush=True)

    dirty = subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain"]).decode().strip()
    sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"]).decode().strip()
    man = {
        "modules": list(man_eng["modules"]), "columns": names, "n_features": len(names),
        "feature_manifest_sha256": man_eng["feature_manifest_sha256"],
        "booster_columns": slices,
        "calibration": {"kind": "scdf", "models": [c.to_json() for c in cals],
                        "time_coord": DEFAULT_COORD,
                        "fitted_on": "cross-fitted OOF over folds_final10k, all 10,000 series",
                        "training_regime_note":
                            "grids from models fitted on 8,000 of 10,000 series; deployed "
                            "boosters fitted on all 10,000. Stated, bounded, accepted -- "
                            "see research/RDOF_LEDGER.md."},
        "streams": cfgs,
        "code_git_sha": sha,
        "code_git_clean": dirty == "",
        "trained_on": {"folds": [0, 1, 2, 3, 4], "partition": "folds_final10k",
                       "n_series": 10000,
                       "includes_former_lockbox": True},
        "lockbox_touched": False,
        "lockbox_note": ("The 2,000-series lockbox is FINAL-FIT TRAINING DATA here, not an "
                         "evaluation set. It was spent for selection in waves 1-2 and no "
                         "score was taken from it in wave 4."),
        "folds_sha256": sha_file(FINAL_FOLDS),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"))
    if dirty:
        print("\n*** WARNING: working tree is DIRTY. This artifact's code_git_sha does not "
              "describe its source. Do not ship it. ***\n" + dirty)
    print(f"wrote {out}/manifest.json  code_git_sha={sha}  clean={dirty == ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("oof", "fit"))
    ap.add_argument("--out", default=f"{ROOT}/models/final10k_ensemble")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = [x for x in a.only.split(",") if x] or None
    if a.phase == "oof":
        phase_oof(only)
    else:
        phase_fit(a.out, only)


if __name__ == "__main__":
    main()
