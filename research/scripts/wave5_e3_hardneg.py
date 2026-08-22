"""W5-E3: does a hard-negative curriculum improve within-timestep ranking?

Pre-registered in research/WAVE5_PREREG.md section 6.

HYPOTHESIS
    TS-AUC is a ranking loss, and ranking losses are dominated by a minority of
    no-break series the champion scores high.  Concentrating training emphasis
    on those series should buy false-positive rejection.  The question is
    whether it costs more on young true breaks than it buys.

WHY THE MINING IS NESTED (this is the part that can silently fake a result)
    "Hardness" is a function of the LABELS.  The obvious implementation --
    score hardness from the existing RT-300 OOF vector -- leaks: a row in fold j
    gets its hardness from the model trained on every fold except j, and that
    model SAW fold k.  So fold k's labels reach fold k's training weights.  The
    effect is small and entirely capable of manufacturing a +0.002.

    Instead, for each OUTER fold k the hardness of the training rows (folds != k)
    is produced by an INNER 4-fold cross-fit run entirely inside folds != k.
    Fold k never enters any model that assigns any weight used to predict it.
    `_assert_fold_pure` checks this rather than trusting it.

    The mining model is deliberately cheap -- m00_core only, 250k rows, 200
    trees.  Hardness is a coarse ranking; it does not need the champion, and a
    cheap miner keeps the nested cost at 20 small models.

THE THREE ARMS (weights fixed here, before the first run, and never tuned)
    RT-710  uniform            CONTROL.  A custom objective starts from raw
                               score 0 instead of the label prior, so the
                               control MUST be a wbinary run, not RT-300.
    RT-711  smooth reweighting w_neg = 1 + 3 r^2, r = within-t percentile rank
                               of the nested hardness score.  Positives w = 1.
    RT-712  oversampling       genuine row duplication: negative rows in the top
                               HARD_FRAC of hardness enter the sampling pool
                               OVER_K times, at the same total row budget.

FALSIFICATION
    Neither treated arm beats RT-710 by the section-4 bar, or the gain is bought
    by degrading the 0-20 post-break age buckets.
"""
from __future__ import annotations

import json, os, sys, time

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

import numpy as np
import sbr.pipeline as PL
import wave5_obj
from sbr.pipeline import Data, run, load_features, _stack
from wave2_lib import FULL, CHAMP

FOLDS = (0, 1, 2, 3, 4)
MINE_MODULES = ["m00_core"]
MINE_ROWS = 250_000
MINE_TREES = 200
#: RT-711
ALPHA, POWER = 3.0, 2.0
#: RT-712
HARD_FRAC, OVER_K = 0.10, 4
HARD_FILE = f"{ROOT}/research/oof/wave5_hardness.npy"


# ---------------------------------------------------------------- mining
def _within_t_rank(scores, t):
    order = np.lexsort((scores, t))
    s, tt = scores[order], t[order]
    n = len(s)
    gs = np.flatnonzero(np.r_[True, tt[1:] != tt[:-1]])
    gl = np.r_[gs[1:], n] - gs
    pos = np.arange(n) - np.repeat(gs, gl)
    new = np.r_[True, (tt[1:] != tt[:-1]) | (s[1:] != s[:-1])]
    rs = np.flatnonzero(new)
    rl = np.r_[rs[1:], n] - rs
    avg = np.repeat(pos[rs] + (rl - 1) / 2.0, rl)
    out = np.empty(n)
    out[order] = avg / np.maximum(np.repeat(gl, gl) - 1, 1)
    return out


def _assert_fold_pure(seen_folds, outer_k):
    if outer_k in seen_folds:
        raise RuntimeError(
            f"FOLD LEAK: mining model for outer fold {outer_k} trained on {sorted(seen_folds)}")


def mine():
    """Nested hardness: HARD[k] holds hardness for folds != k, fold k unseen."""
    import lightgbm as lgb
    t0 = time.time()
    d = Data()
    mats, names = load_features(MINE_MODULES)
    keep = np.arange(len(names))
    H = np.full((len(FOLDS), len(d.y)), np.nan, dtype=np.float32)
    p = dict(objective="binary", learning_rate=0.08, num_leaves=63,
             min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.7,
             bagging_freq=1, lambda_l2=5.0, num_threads=2, verbose=-1, max_bin=127)
    for k in FOLDS:
        inner = [f for f in FOLDS if f != k]
        for j in inner:
            tr_folds = [f for f in inner if f != j]
            _assert_fold_pure(set(tr_folds) | {j}, k)
            tr = d.rows_for(tr_folds)
            va = d.rows_for([j])
            rng = np.random.default_rng(1000 * k + j)
            if len(tr) > MINE_ROWS:
                tr = np.sort(rng.choice(tr, MINE_ROWS, replace=False))
            X = _stack(mats, names, tr, keep)
            ds = lgb.Dataset(X, label=d.y[tr], params=p,
                             feature_name=[f"f{i}" for i in range(len(keep))])
            b = lgb.train(p, ds, num_boost_round=MINE_TREES)
            del X, ds
            Xv = _stack(mats, names, va, keep)
            H[k, va] = b.predict(Xv).astype(np.float32)
            del Xv
        print(f"  outer fold {k}: mined {int(np.isfinite(H[k]).sum())} rows "
              f"({time.time()-t0:.0f}s)", flush=True)
    np.save(HARD_FILE, H)
    print(f"wrote {HARD_FILE}  ({time.time()-t0:.0f}s)")
    return H


# ------------------------------------------------------- weights per outer fold
def build_weights(d, H, k, kind):
    """Weight vector over ALL rows; only folds != k are ever read by training."""
    w = np.ones(len(d.y), dtype=np.float64)
    m = np.isfinite(H[k]) & (d.y == 0)
    if not m.any():
        raise RuntimeError("no mined negative rows")
    r = np.zeros(len(d.y))
    r[m] = _within_t_rank(H[k][m].astype(np.float64), d.t[m])
    if kind == "reweight":
        w[m] = 1.0 + ALPHA * r[m] ** POWER
    elif kind != "uniform":
        raise ValueError(kind)
    return w, r


def oversample_pool(d, H, k):
    """Row pool for outer fold k with the hardest negatives repeated OVER_K x.

    Selection is by QUANTILE of the same-t inversion rank over the ELIGIBLE
    inner-fold negatives -- `np.quantile(r[m], 1 - HARD_FRAC)` -- not by a fixed
    threshold on the rank.  Those differ once hardness stopped being a global
    percentile: `r >= 0.90` would select whatever share of rows happens to sit
    above 0.90, while this selects the hardest HARD_FRAC.  The realised fraction
    is 0.10000 on every outer fold, recorded in
    research/reports/wave5_e3_conformance.json.

    FOLD EXCLUSION IS AN INVARIANT OF THIS FUNCTION, not a property inherited
    from the caller.  `H[k]` is NaN on fold k by construction of the nested
    mining, so `m` already excludes it; the explicit `row_fold != k` term makes
    that structural rather than incidental.  Verified identical to the executed
    RT-712 mask: 0 selected rows lay in the held-out fold on all five folds, so
    the added term changes no value and cannot have changed the observed result.
    """
    m = np.isfinite(H[k]) & (d.y == 0) & (d.row_fold != k)
    r = np.zeros(len(d.y))
    r[m] = _within_t_rank(H[k][m].astype(np.float64), d.t[m])
    thr = np.quantile(r[m], 1.0 - HARD_FRAC)
    sel = (r >= thr) & m
    assert not sel[d.row_fold == k].any(), "oversample pool leaked the held-out fold"
    assert not sel[d.y == 1].any(), "oversample pool selected a POSITIVE row"
    return sel


# ------------------------------------------------------------------- arms
_ORIG_ROWS_FOR = Data.rows_for


def install_oversampler(hard_mask_by_fold):
    """Duplicate hard negatives in the TRAINING pool only.

    `run` calls rows_for with 4 folds for training, 1 for validation and 5 for
    the pooled score.  Only the 4-fold call is touched, so no evaluation surface
    ever sees a duplicated row.
    """
    def patched(self, folds):
        rows = _ORIG_ROWS_FOR(self, folds)
        f = list(np.atleast_1d(folds))
        k = [x for x in FOLDS if x not in f]
        # "four folds means training" is a CONVENTION; everything below turns it
        # into a checked invariant.  The duplicated rows are drawn from `rows`,
        # never from arange(n_rows), so a held-out row cannot enter even once.
        if len(f) != 4 or len(k) != 1 or not set(f) <= set(FOLDS):
            return rows
        held = k[0]
        mask = hard_mask_by_fold[held]
        assert not mask[self.row_fold == held].any(), (
            f"hard mask for outer fold {held} contains held-out rows")
        extra = rows[mask[rows]]
        out = np.sort(np.concatenate([rows] + [extra] * (OVER_K - 1)))
        assert not (self.row_fold[out] == held).any(), (
            f"oversampled TRAINING index contains rows from held-out fold {held}")
        return out
    Data.rows_for = patched


def restore_oversampler():
    Data.rows_for = _ORIG_ROWS_FOR


ARMS = {
    "RT-710": "uniform",
    "RT-711": "reweight",
    "RT-712": "oversample",
}


def main(which):
    H = np.load(HARD_FILE) if os.path.exists(HARD_FILE) else mine()
    d = Data()
    hard_mask = {k: oversample_pool(d, H, k) for k in FOLDS}
    W = {k: {} for k in FOLDS}
    for k in FOLDS:
        for kind in ("uniform", "reweight"):
            W[k][kind], _ = build_weights(d, H, k, kind)

    for exp in which:
        kind = ARMS[exp]
        wkey = "uniform" if kind == "oversample" else kind
        wave5_obj.SPEC = {
            "kind": "wbinary",
            "weights_by_fold": {k: W[k][wkey] for k in FOLDS},
            "row_fold": d.row_fold,
        }
        wave5_obj.install()
        if kind == "oversample":
            install_oversampler(hard_mask)
        t0 = time.time()
        try:
            run(exp_id=exp, modules=FULL, agent="claude-wave5", seed=0,
                folds=FOLDS, params=dict(CHAMP["params"], objective="pairwise_t"),
                max_train_rows=CHAMP["max_train_rows"],
                hypothesis="Concentrating training emphasis on no-break rows that a NESTED, "
                           "fold-pure miner scores high improves within-timestep ranking by "
                           "rejecting false positives, without costing young true breaks.",
                falsification="fails to beat the RT-710 uniform-weight control by +0.0030 on "
                              ">=4/5 folds, or degrades the 0-20 post-break age buckets",
                notes=f"W5-E3 {kind}; nested miner m00_core/250k/200 trees; "
                      f"alpha={ALPHA} power={POWER} hard_frac={HARD_FRAC} over_k={OVER_K}",
                extra={"objective": f"wbinary_{kind}"})
        finally:
            wave5_obj.restore()
            restore_oversampler()
        print(f"[{exp}] {kind} {time.time()-t0:.0f}s\n", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "mine":
        mine()
    else:
        main(a or list(ARMS))
