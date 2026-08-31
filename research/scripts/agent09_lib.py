"""Agent 09 -- supervised objectives / targets / weighting harness.

Reuses sbr.pipeline building blocks but keeps its own training loop so that the
materialised (Xtr, Xva) matrices are built ONCE per fold and shared across every
objective variant -- that is the only way to fit a real objective sweep into the
compute budget.  Every run is logged through sbr.pipeline.append_result with the
same fields pipeline.run() writes.
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

sys.path.insert(0, "/home/claude/sb/src")

from sbr.pipeline import Data, load_features, _stack, append_result, git_sha
from sbr.metric import ts_auc_flat

ROOT = "/home/claude/sb"
DEV_FOLDS = (0, 1, 2, 3, 4)


class Bench:
    """Holds the row-space + materialised matrices for a set of folds."""

    def __init__(self, modules, folds=(0,), max_train_rows=300_000, seed=0,
                 sample_mode="uniform", screen=True):
        self.modules = list(modules)
        self.folds = tuple(folds)
        self.max_train_rows = max_train_rows
        self.seed = seed
        self.sample_mode = sample_mode
        self.screen = screen
        t0 = time.time()
        self.d = Data(screen=screen)
        self.mats, self.names = load_features(self.modules, screen=screen)
        self.n_feat = sum(m.shape[1] for m in self.mats)
        self.keep_idx = np.arange(self.n_feat)
        st = self.d.st
        # per-row tau (== -1 for no-break series)
        tau_series = st.meta.tau_index.to_numpy().astype(np.int64)
        self.tau = tau_series[self.d.sidx]
        self.cache = {}
        print(f"[bench] {self.modules} n_feat={self.n_feat} load={time.time()-t0:.1f}s", flush=True)

    # ---------------------------------------------------------------- rows
    def train_rows(self, f, sample_mode=None, seed=None):
        sample_mode = sample_mode or self.sample_mode
        seed = self.seed if seed is None else seed
        rng = np.random.default_rng(seed)
        tr_folds = [x for x in DEV_FOLDS if x != f]
        tr = self.d.rows_for(tr_folds)
        if len(tr) <= self.max_train_rows:
            return tr
        if sample_mode == "uniform":
            return np.sort(rng.choice(tr, self.max_train_rows, replace=False))
        if sample_mode == "per_series":
            s = self.d.sidx[tr]
            order = np.argsort(s, kind="stable")
            tr = tr[order]; s = s[order]
            bnd = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
            per = self.max_train_rows // len(bnd)
            ends = np.r_[bnd[1:], len(s)]
            sel = []
            for b, e in zip(bnd, ends):
                idx = np.arange(b, e)
                sel.append(idx if e - b <= per else rng.choice(idx, per, replace=False))
            return np.sort(tr[np.concatenate(sel)])
        if sample_mode == "t_pairprop":
            # sample rows with probability proportional to their metric pair count:
            # positive at t participates in n_neg(t) pairs, negative in n_pos(t).
            w = self.pair_weight(tr)
            p = w / w.sum()
            pick = rng.choice(len(tr), self.max_train_rows, replace=False, p=p)
            return np.sort(tr[pick])
        raise ValueError(sample_mode)

    def pair_weight(self, rows):
        """Metric-native row weight: #pairs the row takes part in at its step t."""
        t = self.d.t[rows]; y = self.d.y[rows].astype(np.int64)
        nb = int(t.max()) + 1
        npos = np.bincount(t, weights=y, minlength=nb)
        ntot = np.bincount(t, minlength=nb)
        nneg = ntot - npos
        w = np.where(y == 1, nneg[t], npos[t]).astype(np.float64)
        return np.maximum(w, 0.0)

    # ---------------------------------------------------------------- data
    def get(self, f, sample_mode=None, seed=None):
        key = (f, sample_mode or self.sample_mode, self.seed if seed is None else seed)
        if key in self.cache:
            return self.cache[key]
        t0 = time.time()
        tr = self.train_rows(f, key[1], key[2])
        va = self.d.rows_for([f])
        Xtr = _stack(self.mats, self.names, tr, self.keep_idx)
        Xva = _stack(self.mats, self.names, va, self.keep_idx)
        pack = dict(tr=tr, va=va, Xtr=Xtr, Xva=Xva,
                    ytr=self.d.y[tr].astype(np.float64), yva=self.d.y[va],
                    ttr=self.d.t[tr].astype(np.int64), tva=self.d.t[va].astype(np.int64),
                    str_=self.d.sidx[tr], tautr=self.tau[tr], tauva=self.tau[va])
        self.cache[key] = pack
        print(f"[bench] fold {f} {key[1]}: {len(tr)} train / {len(va)} valid rows "
              f"({time.time()-t0:.1f}s)", flush=True)
        return pack

    def score(self, pred, pack):
        return float(ts_auc_flat(np.asarray(pred, np.float64), pack["yva"], pack["tva"]))


# ------------------------------------------------------------------ logging
def log(exp_id, bench, per_fold, *, hypothesis, falsification, model, objective,
        notes, sample_mode, runtime, agent="agent09", oof=None, extra=None):
    res = {
        "experiment_id": exp_id, "date": time.strftime("%Y-%m-%d %H:%M"), "git_sha": git_sha(),
        "agent": agent, "hypothesis": hypothesis, "falsification_condition": falsification,
        "feature_set": ",".join(bench.modules), "n_features": bench.n_feat, "model": model,
        "objective": objective, "folds": ",".join(map(str, bench.folds)),
        "random_seed": bench.seed,
        "train_series": int((bench.d.series_fold >= 0).sum()),
        "train_rows": bench.max_train_rows,
        "mean_oof_ts_auc": float(np.mean(per_fold)),
        "pooled_oof_ts_auc": float(np.mean(per_fold)) if oof is None else float(oof),
        "per_fold_ts_auc": ";".join(f"{x:.5f}" for x in per_fold),
        "fold_std": float(np.std(per_fold)),
        "persistence": "none", "sample_mode": sample_mode,
        "training_runtime_s": round(runtime, 1),
        "causal_verified": "prefix-invariance@module(inherited)", "test_reduced_touched": "no",
        "lockbox_touched": "no", "status": "recorded", "notes": notes,
    }
    if extra:
        res.update(extra)
    append_result(res)
    return res


# --------------------------------------------------------------- objectives
BASE = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
            feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
            lambda_l2=5.0, num_threads=2, verbose=-1, max_bin=127)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def make_pairwise_obj(t, y, m_neg=8, seed=0, resample_every=1):
    """Pairwise logistic (RankNet) with pairs drawn WITHIN the same online index t.

    Groups are the online step t -- exactly the metric's stratification.
    """
    t = np.asarray(t, np.int64); y = np.asarray(y).astype(np.int8)
    n = len(t)
    order = np.lexsort((y, t))            # negatives first inside each t
    ts = t[order]; ys = y[order]
    gstart = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]])
    gend = np.r_[gstart[1:], n]
    gid = np.repeat(np.arange(len(gstart)), gend - gstart)
    nneg = np.array([int((ys[a:b] == 0).sum()) for a, b in zip(gstart, gend)])
    npos = (gend - gstart) - nneg
    ok = (nneg > 0) & (npos > 0)
    # positions (in ORIGINAL row space) of positives in usable groups
    pos_mask = (ys == 1) & ok[gid]
    pos_rows = order[np.flatnonzero(pos_mask)]
    pos_gid = gid[np.flatnonzero(pos_mask)]
    # flat negative pool laid out group-major
    neg_pool = order[np.flatnonzero((ys == 0))]
    negoff = np.r_[0, np.cumsum(nneg)[:-1]]
    rng = np.random.default_rng(seed)
    state = {"it": 0, "pairs": None}
    npairs = len(pos_rows) * m_neg
    scale = n / max(npairs, 1)

    def draw():
        u = rng.random((len(pos_rows), m_neg))
        j = (u * nneg[pos_gid][:, None]).astype(np.int64)
        jj = neg_pool[negoff[pos_gid][:, None] + j].ravel()
        ii = np.repeat(pos_rows, m_neg)
        return ii, jj

    def obj(preds, dset):
        if state["pairs"] is None or state["it"] % resample_every == 0:
            state["pairs"] = draw()
        state["it"] += 1
        ii, jj = state["pairs"]
        p = _sig(preds[ii] - preds[jj])       # P(pos ranked above neg)
        g = -(1.0 - p) * scale
        h = np.maximum(p * (1.0 - p), 1e-6) * scale
        grad = np.bincount(ii, weights=g, minlength=n) + np.bincount(jj, weights=-g, minlength=n)
        hess = np.bincount(ii, weights=h, minlength=n) + np.bincount(jj, weights=h, minlength=n)
        return grad, np.maximum(hess, 1e-6)

    return obj, dict(n_pos_used=int(len(pos_rows)), n_pairs=int(npairs),
                     n_groups=int(ok.sum()))


def make_focal_obj(alpha=0.5, gamma=1.5):
    def obj(preds, dset):
        y = dset.get_label()
        p = _sig(preds)
        a = np.where(y == 1, alpha, 1 - alpha)
        pt = np.where(y == 1, p, 1 - p)
        # d/dz of  -a (1-pt)^g log pt      (z = raw score)
        g_ = gamma
        common = a * (1 - pt) ** g_
        grad = common * (g_ * pt * np.log(np.maximum(pt, 1e-12)) / np.maximum(1 - pt, 1e-12) + pt - 1.0)
        grad = np.where(y == 1, grad, -grad)
        # numeric hessian (stable, cheap)
        eps = 1e-4
        def loss_grad(z):
            pp = _sig(z); ptt = np.where(y == 1, pp, 1 - pp)
            cc = a * (1 - ptt) ** g_
            gg = cc * (g_ * ptt * np.log(np.maximum(ptt, 1e-12)) / np.maximum(1 - ptt, 1e-12) + ptt - 1.0)
            return np.where(y == 1, gg, -gg)
        hess = (loss_grad(preds + eps) - loss_grad(preds - eps)) / (2 * eps)
        return grad, np.maximum(hess, 1e-6)
    return obj
