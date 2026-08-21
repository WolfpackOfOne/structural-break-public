"""W5-E9 / W5-E3 objective and sample-weight hooks.

Pre-registered in research/WAVE5_PREREG.md sections 6 (W5-E9, W5-E3).

WHY THIS IS A HOOK AND NOT A NEW TRAINING LOOP
    Every wave-1..wave-4 number came out of `sbr.pipeline.run`.  A candidate
    trained by a different loop is not paired with the incumbents, so the deltas
    stop being comparable.  `run` already dispatches one custom objective by
    name (`pairwise_t`, which builds a closure from the SAMPLED training rows'
    t and y -- information that only exists inside the fold loop).  We reuse
    exactly that dispatch point: `_make_pairwise_t` is replaced by a router that
    reads a module-global spec.  The fold loop, the row sampling, the feature
    stacking, the seeds and the ledger are untouched.

    Every run started through here records its REAL objective in the ledger via
    `extra=`, never the `pairwise_t` string that triggered the dispatch.

WHAT THE OBJECTIVES ARE

  pairwise_t   INCUMBENT (stream RT-123R/RT-413).  RankNet-style pairwise
               logistic over (positive, negative) pairs sharing an online index.
               Each POSITIVE ROW contributes m_neg pairs, regardless of how many
               negatives are alive at its timestep.

  pairwise_w   W5-E9a.  The same pairs, reweighted so that a positive at time t
               carries weight proportional to n_neg(t).  This is the official
               metric's own structure: TS-AUC pools concordant pairs over
               sum_t n_pos(t)*n_neg(t), so every (pos, neg) pair in the dataset
               counts EQUALLY -- which the incumbent's fixed m_neg does not
               reproduce.  Timesteps where many series are still alive are
               under-weighted by the incumbent and correctly weighted here.

  pairwise_h   W5-E9b.  The same pairs and the same weights, squared-hinge
               margin loss instead of logistic.  Isolates the loss SHAPE from
               the weighting: hinge stops pushing a pair once it is ranked
               correctly by a margin, so gradient goes to the pairs still
               inverted rather than to widening already-correct margins.

  wbinary      W5-E3 vehicle.  Plain binary logloss with PER-ROW WEIGHTS,
               computed through the same custom-objective path so that the
               hard-negative arms and their control differ ONLY in the weight
               vector.  A custom objective starts from raw score 0 rather than
               the label prior, so the W5-E3 control is a `wbinary` run with
               uniform weights -- never RT-300.
"""
from __future__ import annotations

import numpy as np
import sbr.pipeline as PL

#: set by the driver before PL.run(); read inside the fold loop
SPEC: dict = {"kind": "pairwise_t", "m_neg": 8}

_ORIG = PL._make_pairwise_t
_ORIG_STACK = PL._stack

#: gradient-scale match between the squared hinge and the logistic at d = 0
HINGE_LR_MATCH = 0.25

#: the training rows LightGBM is about to see, captured from the pipeline's own
#: call.  `run` does `Xtr = _stack(..., tr_rows, ...)` immediately before it
#: builds the objective closure, so this is exactly the sampled training set --
#: recovered rather than re-derived, which is why a per-row weight vector can be
#: aligned to it without duplicating the sampler.
LAST_ROWS: dict = {"rows": None}


def _stack_spy(mats, names, rows, keep_idx, max_span=150_000):
    LAST_ROWS["rows"] = np.asarray(rows)
    return _ORIG_STACK(mats, names, rows, keep_idx, max_span)


def _pair_index(t, y, m_neg, seed):
    """Shared pair machinery: positives, their group, and the negative pool."""
    t = np.asarray(t, np.int64)
    y = np.asarray(y).astype(np.int8)
    n = len(t)
    order = np.lexsort((y, t))              # negatives first inside each group
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
    return n, pos_rows, pos_gid, neg_pool, negoff, nneg


def _pair_weights(pos_gid, nneg, kind):
    """Per-positive pair weight.

    'pairwise_t'   1.0                 -- every positive contributes m_neg pairs
    'pairwise_w'   n_neg(t) / mean     -- every (pos, neg) PAIR contributes
    'pairwise_h'   n_neg(t) / mean
    """
    if kind == "pairwise_t":
        return np.ones(len(pos_gid))
    w = nneg[pos_gid].astype(np.float64)
    return w / max(w.mean(), 1e-12)


def _make_pairwise(t, y, m_neg=8, seed=0):
    """Router installed over PL._make_pairwise_t.  Returns an lgb objective."""
    kind = SPEC.get("kind", "pairwise_t")

    if kind == "wbinary":
        rows = LAST_ROWS["rows"]
        yv = np.asarray(y, dtype=np.float64)
        if rows is None or len(rows) != len(yv):
            raise RuntimeError("row capture failed; refusing to guess the weight alignment")
        w = np.asarray(SPEC["weights"], dtype=np.float64)[rows]
        if not np.isfinite(w).all() or (w < 0).any():
            raise ValueError("weights must be finite and non-negative")
        w = w * (len(w) / w.sum())          # mean weight 1: same effective lr

        def obj_wb(preds, dset):
            p = 1.0 / (1.0 + np.exp(-np.clip(preds, -60, 60)))
            return w * (p - yv), np.maximum(w * p * (1.0 - p), 1e-6)

        return obj_wb

    m_neg = int(SPEC.get("m_neg", m_neg))
    n, pos_rows, pos_gid, neg_pool, negoff, nneg = _pair_index(t, y, m_neg, seed)
    pw = _pair_weights(pos_gid, nneg, kind)
    rng = np.random.default_rng(seed)
    # keep the total gradient mass equal to the incumbent's, so learning_rate
    # means the same thing in every arm
    scale = n / max(pw.sum() * m_neg, 1e-12)
    wrep = np.repeat(pw, m_neg) * scale
    ii_base = np.repeat(pos_rows, m_neg)
    hinge = kind == "pairwise_h"

    def obj_pair(preds, dset):
        j = (rng.random((len(pos_rows), m_neg)) * nneg[pos_gid][:, None]).astype(np.int64)
        jj = neg_pool[negoff[pos_gid][:, None] + j].ravel()
        ii = ii_base
        dmar = preds[ii] - preds[jj]
        if hinge:
            # squared hinge on the margin: loss = max(0, 1 - d)^2.
            # HINGE_LR_MATCH is not a tuning knob: at d = 0 the squared hinge's
            # per-pair gradient is 2 while the logistic's is 1/2, so without it
            # the hinge arm would run at 4x the effective learning rate and the
            # comparison would measure step size, not loss shape.
            act = np.maximum(1.0 - dmar, 0.0)
            g = -2.0 * act * wrep * HINGE_LR_MATCH
            h = np.where(act > 0, 2.0, 0.0) * wrep * HINGE_LR_MATCH
        else:
            pr = 1.0 / (1.0 + np.exp(-np.clip(dmar, -60, 60)))
            g = -(1.0 - pr) * wrep
            h = np.maximum(pr * (1.0 - pr), 1e-6) * wrep
        grad = np.bincount(ii, weights=g, minlength=n) + np.bincount(jj, weights=-g, minlength=n)
        hess = np.bincount(ii, weights=h, minlength=n) + np.bincount(jj, weights=h, minlength=n)
        return grad, np.maximum(hess, 1e-6)

    return obj_pair


def install():
    PL._make_pairwise_t = _make_pairwise
    PL._stack = _stack_spy


def restore():
    PL._make_pairwise_t = _ORIG
    PL._stack = _ORIG_STACK
