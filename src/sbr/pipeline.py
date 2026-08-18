"""Leakage-safe OOF training/evaluation pipeline + experiment ledger.

Everything is series-level: a series is entirely in train or entirely in
validation, so no two adjacent prefixes of one series ever straddle the split.
Model selection uses held-out **trajectory TS-AUC** (the official metric on full
validation trajectories), never row-level AUC.

Row subsampling applies to TRAINING ROWS ONLY -- validation always scores every
online row of every validation series, exactly as the competition does.

The lockbox (fold == -1) is never loaded here unless --lockbox is passed, which
also stamps the ledger row so the touch is permanently visible.
"""
from __future__ import annotations

import argparse, json, os, subprocess, time
import numpy as np
import pandas as pd

from sbr.metric import ts_auc_flat
from sbr.store import load_store

ROOT = "/home/claude/sb"
FEAT = f"{ROOT}/cache/features"
FEAT_SCREEN = f"{ROOT}/cache/features_screen"
FOLDS = f"{ROOT}/research/folds/folds.parquet"
FOLDS_SCREEN = f"{ROOT}/research/folds/folds_screen.parquet"
STORE = f"{ROOT}/cache/store"
STORE_SCREEN = f"{ROOT}/cache/store_screen"
OOF = f"{ROOT}/research/oof"
RESULTS = f"{ROOT}/research/RESULTS.csv"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def load_features(modules, screen=False):
    feat = FEAT_SCREEN if screen else FEAT
    mats, names = [], []
    for m in modules:
        meta = json.load(open(f"{feat}/{m}.cols.json"))
        A = np.load(f"{feat}/{m}.npy", mmap_mode="r")
        cn = [f"{m}::{c}" for c in meta["cols"]]
        mats.append(A)
        names += cn
    return mats, names


class Data:
    """Row-space bookkeeping shared by every experiment."""

    def __init__(self, store=None, screen=False):
        self.screen = screen
        self.st = load_store(store or (STORE_SCREEN if screen else STORE))
        self.folds = pd.read_parquet(FOLDS_SCREEN if screen else FOLDS)
        assert (self.folds.id.to_numpy() == self.st.meta.id.to_numpy()).all()
        self.series_fold = self.folds.fold.to_numpy()
        self.y = self.st.all_labels()
        self.t = self.st.all_t_index()
        self.sidx = self.st.all_series_index()
        self.row_fold = self.series_fold[self.sidx]

    def rows_for(self, folds):
        return np.flatnonzero(np.isin(self.row_fold, np.atleast_1d(folds)))


def _stack(mats, names, rows, keep_idx, max_span=150_000):
    """Materialise selected rows/columns from the module memmaps.

    Rows are read in slices whose covered index range is bounded, then subset
    in memory.  The store is series-major and folds are series-level, so a
    fold's rows form long runs; slicing beats row-by-row fancy indexing on a
    5M-row memmap by an order of magnitude, and the span bound keeps peak
    memory flat when the rows are a subsample.
    """
    rows = np.asarray(rows)
    out = np.empty((len(rows), len(keep_idx)), dtype=np.float32)
    off = 0
    spans = []
    for A in mats:
        spans.append((off, off + A.shape[1]))
        off += A.shape[1]
    keep_idx = np.asarray(keep_idx)

    cuts = []
    i = 0
    while i < len(rows):
        j = i + int(np.searchsorted(rows[i:], rows[i] + max_span))
        j = max(j, i + 1)
        cuts.append((i, j))
        i = j

    pos = 0
    for A, (lo, hi) in zip(mats, spans):
        sel = keep_idx[(keep_idx >= lo) & (keep_idx < hi)] - lo
        if len(sel) == 0:
            continue
        whole = len(sel) == A.shape[1] and sel[0] == 0
        for i, j in cuts:
            a = rows[i]
            blk = np.asarray(A[a:rows[j - 1] + 1])
            idx = rows[i:j] - a
            out[i:j, pos:pos + len(sel)] = blk[idx] if whole else blk[idx][:, sel]
            del blk
        pos += len(sel)
    return out


def run(exp_id, modules, hypothesis, falsification, agent="agent0", model="lgbm",
        params=None, folds=(0, 1, 2, 3, 4), max_train_rows=800_000, seed=0,
        drop_cols=(), keep_regex=None, notes="", persistence="none", lockbox=False,
        save_oof=True, sample_mode="uniform", screen=False, extra=None):
    import lightgbm as lgb

    t_start = time.time()
    d = Data(screen=screen)
    mats, names = load_features(modules, screen=screen)
    names_arr = np.array(names)

    keep = np.ones(len(names), bool)
    for pat in drop_cols:
        keep &= ~np.char.find(names_arr.astype(str), pat).astype(bool).__ne__(False) if False else \
                ~np.array([pat in n for n in names])
    if keep_regex:
        import re
        rx = re.compile(keep_regex)
        keep &= np.array([bool(rx.search(n)) for n in names])
    keep_idx = np.flatnonzero(keep)
    used_names = [names[i] for i in keep_idx]

    default = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                   feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
                   lambda_l2=5.0, num_threads=2, verbose=-1, max_bin=127)
    p = dict(default)
    if params:
        p.update(params)
    n_round = int(p.pop("n_estimators", 400))

    rng = np.random.default_rng(seed)
    oof = np.full(len(d.y), np.nan, dtype=np.float32)
    per_fold, fold_rt = [], []
    imp_sum = np.zeros(len(keep_idx))

    for f in folds:
        tr_folds = [x for x in (0, 1, 2, 3, 4) if x != f]
        tr_rows = d.rows_for(tr_folds)
        va_rows = d.rows_for([f])
        if len(tr_rows) > max_train_rows:
            if sample_mode == "uniform":
                tr_rows = np.sort(rng.choice(tr_rows, max_train_rows, replace=False))
            elif sample_mode == "per_series":
                # equal number of rows per series: stops long series dominating
                s = d.sidx[tr_rows]
                order = np.argsort(s, kind="stable")
                tr_rows = tr_rows[order]; s = s[order]
                bnd = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
                per = max_train_rows // len(bnd)
                sel = []
                ends = np.r_[bnd[1:], len(s)]
                for b, e in zip(bnd, ends):
                    idx = np.arange(b, e)
                    sel.append(idx if e - b <= per else rng.choice(idx, per, replace=False))
                tr_rows = np.sort(tr_rows[np.concatenate(sel)])

        Xtr = _stack(mats, names, tr_rows, keep_idx)
        ytr = d.y[tr_rows]
        pf = dict(p)
        if pf.get("objective") == "pairwise_t":
            # RankNet-style pairwise logistic whose pairs are drawn WITHIN the
            # same online index -- the metric's own stratification.  Built here
            # because it needs the sampled training rows' t and y.
            pf["objective"] = _make_pairwise_t(d.t[tr_rows], ytr, seed=seed)
        ds = lgb.Dataset(Xtr, label=ytr, params=dict(pf, objective="binary"),
                         feature_name=[f"f{i}" for i in range(len(keep_idx))])
        booster = lgb.train(pf, ds, num_boost_round=n_round)
        del Xtr, ds

        Xva = _stack(mats, names, va_rows, keep_idx)
        pred = booster.predict(Xva).astype(np.float32)
        del Xva
        pred = apply_persistence(pred, d, va_rows, persistence)
        oof[va_rows] = pred
        s = ts_auc_flat(pred, d.y[va_rows], d.t[va_rows])
        per_fold.append(float(s))
        imp_sum += booster.feature_importance("gain")
        fold_rt.append(time.time() - t_start)
        print(f"  fold {f}: TS-AUC {s:.5f}  ({len(tr_rows)} train rows, {len(va_rows)} valid rows)", flush=True)

    dev_rows = d.rows_for(list(folds))
    overall = ts_auc_flat(oof[dev_rows], d.y[dev_rows], d.t[dev_rows])
    res = {
        "experiment_id": exp_id, "date": time.strftime("%Y-%m-%d %H:%M"), "git_sha": git_sha(),
        "agent": agent, "hypothesis": hypothesis, "falsification_condition": falsification,
        "feature_set": ",".join(modules), "n_features": len(keep_idx), "model": model,
        "objective": (params or {}).get("objective", "binary"), "folds": ",".join(map(str, folds)), "random_seed": seed,
        "train_series": int((~np.isin(d.series_fold, [-1])).sum()),
        "train_rows": int(min(max_train_rows, len(d.rows_for([x for x in (0,1,2,3,4) if x != folds[0]])))),
        "mean_oof_ts_auc": float(np.mean(per_fold)), "pooled_oof_ts_auc": float(overall),
        "per_fold_ts_auc": ";".join(f"{x:.5f}" for x in per_fold),
        "fold_std": float(np.std(per_fold)),
        "persistence": persistence, "sample_mode": sample_mode,
        "training_runtime_s": round(time.time() - t_start, 1),
        "causal_verified": "prefix-invariance@module", "test_reduced_touched": "no",
        "lockbox_touched": "yes" if lockbox else "no",
        "status": "recorded", "notes": notes, "protocol": "screen" if screen else "full",
    }
    if extra:
        res.update(extra)
    if save_oof and not screen:
        os.makedirs(OOF, exist_ok=True)
        np.save(f"{OOF}/{exp_id}.npy", oof)
        imp = pd.DataFrame({"feature": used_names, "gain": imp_sum}).sort_values("gain", ascending=False)
        imp.to_csv(f"{OOF}/{exp_id}.importance.csv", index=False)
    append_result(res)
    print(json.dumps({k: res[k] for k in ("experiment_id", "mean_oof_ts_auc", "pooled_oof_ts_auc",
                                          "per_fold_ts_auc", "fold_std", "n_features")}, indent=2))
    return res


def _make_pairwise_t(t, y, m_neg=8, seed=0):
    """Pairwise logistic loss over (positive, negative) pairs sharing an online index.

    TS-AUC only ever asks whether a broken series outranks a not-yet-broken one
    at the SAME t, so the groups here are online indices, not series.
    """
    t = np.asarray(t, np.int64)
    y = np.asarray(y).astype(np.int8)
    n = len(t)
    order = np.lexsort((y, t))          # negatives first inside each group
    ts, ys = t[order], y[order]
    gstart = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]])
    gend = np.r_[gstart[1:], n]
    gid = np.repeat(np.arange(len(gstart)), gend - gstart)
    nneg = np.array([int((ys[a:b] == 0).sum()) for a, b in zip(gstart, gend)])
    npos = (gend - gstart) - nneg
    ok = (nneg > 0) & (npos > 0)
    pm = np.flatnonzero((ys == 1) & ok[gid])
    pos_rows, pos_gid = order[pm], gid[pm]
    neg_pool = order[np.flatnonzero(ys == 0)]
    negoff = np.r_[0, np.cumsum(nneg)[:-1]]
    rng = np.random.default_rng(seed)
    scale = n / max(len(pos_rows) * m_neg, 1)

    def obj(preds, dset):
        j = (rng.random((len(pos_rows), m_neg)) * nneg[pos_gid][:, None]).astype(np.int64)
        jj = neg_pool[negoff[pos_gid][:, None] + j].ravel()
        ii = np.repeat(pos_rows, m_neg)
        pr = 1.0 / (1.0 + np.exp(-np.clip(preds[ii] - preds[jj], -60, 60)))
        g = -(1.0 - pr) * scale
        h = np.maximum(pr * (1.0 - pr), 1e-6) * scale
        grad = np.bincount(ii, weights=g, minlength=n) + np.bincount(jj, weights=-g, minlength=n)
        hess = np.bincount(ii, weights=h, minlength=n) + np.bincount(jj, weights=h, minlength=n)
        return grad, np.maximum(hess, 1e-6)

    return obj


def apply_persistence(pred, d, rows, mode):
    if mode == "none":
        return pred
    s = d.sidx[rows]
    out = pred.copy()
    bnd = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    ends = np.r_[bnd[1:], len(s)]
    for b, e in zip(bnd, ends):
        v = out[b:e]
        if mode == "runmax":
            out[b:e] = np.maximum.accumulate(v)
        elif mode.startswith("decaymax"):
            g = float(mode.split(":")[1]) if ":" in mode else 0.99
            m = v[0]; res = np.empty_like(v)
            for i, x in enumerate(v):
                m = max(x, m * g); res[i] = m
            out[b:e] = res
        elif mode.startswith("ewma"):
            a = float(mode.split(":")[1]) if ":" in mode else 0.3
            m = v[0]; res = np.empty_like(v)
            for i, x in enumerate(v):
                m = (1 - a) * m + a * x; res[i] = m
            out[b:e] = res
    return out


def append_result(res: dict):
    """Append one experiment row under a file lock, keeping the CSV rectangular."""
    import fcntl
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            df = pd.read_csv(RESULTS) if os.path.exists(RESULTS) else pd.DataFrame()
        except Exception:
            df = pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([res])], ignore_index=True)
        df.to_csv(RESULTS, index=False)
        fcntl.flock(lk, fcntl.LOCK_UN)


def evaluate_scores(scores: np.ndarray, folds=(0, 1, 2, 3, 4), d: "Data | None" = None):
    """TS-AUC of an arbitrary full-length score vector (NaN outside dev folds)."""
    d = d or Data()
    per = []
    for f in folds:
        r = d.rows_for([f])
        per.append(float(ts_auc_flat(scores[r], d.y[r], d.t[r])))
    r = d.rows_for(list(folds))
    return float(np.mean(per)), per, float(ts_auc_flat(scores[r], d.y[r], d.t[r]))
