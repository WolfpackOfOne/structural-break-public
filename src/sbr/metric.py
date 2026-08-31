"""Official Time-Stratified AUC, computed exactly and fast.

For each online index ``t`` we take every series still alive at ``t``, compute
the ordinary cross-sectional ROC AUC between series whose break has already
occurred (``t >= tau``) and series whose break has not (or which never break),
and weight that timestep by ``w_t = n_pos(t) * n_neg(t)``:

    TS_AUC = sum_t w_t * AUC_t / sum_t w_t

Timesteps with only one class present contribute zero weight.

Implementation note: with the pair-count weighting the whole thing collapses to
a single global ratio,

    TS_AUC = sum_t (R_pos(t) - n_pos(t)(n_pos(t)+1)/2) / sum_t n_pos(t) n_neg(t)

where ``R_pos(t)`` is the sum of the mid-ranks of the positive scores inside
timestep ``t``.  That is computed with one lexsort over all online rows, so a
full 5M-row trajectory set scores in a couple of seconds and ties are handled
exactly (mid-ranks), matching ``sklearn.metrics.roc_auc_score``.
"""
from __future__ import annotations

import numpy as np


def ts_auc_flat(
    scores: np.ndarray,
    labels: np.ndarray,
    t_index: np.ndarray,
    return_per_step: bool = False,
    weighting: str = "pairs",
):
    """Time-stratified AUC over flat (row-wise) arrays.

    scores/labels/t_index are parallel arrays over all online rows of the
    evaluated series (any series ordering; only ``t_index`` matters).

    weighting: 'pairs' (official: n_pos*n_neg), 'equal' (unweighted mean over
    timesteps), or 'alive' (n_pos+n_neg) -- the latter two exist purely as
    robustness checks, never for model selection.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    t_index = np.asarray(t_index, dtype=np.int64)
    if scores.shape != labels.shape or scores.shape != t_index.shape:
        raise ValueError("scores, labels and t_index must have identical shape")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or inf")

    order = np.lexsort((scores, t_index))
    s = scores[order]
    y = labels[order]
    t = t_index[order]

    n = len(s)
    # group boundaries for t
    gstart = np.flatnonzero(np.r_[True, t[1:] != t[:-1]])
    gend = np.r_[gstart[1:], n]
    glen = gend - gstart

    # rank within group, 1-based, mid-ranks for ties
    pos_in_group = np.arange(n) - np.repeat(gstart, glen)
    rank = pos_in_group + 1.0
    # tie handling: identical (t, score) runs get the average of their ranks
    new_run = np.r_[True, (t[1:] != t[:-1]) | (s[1:] != s[:-1])]
    run_start = np.flatnonzero(new_run)
    run_end = np.r_[run_start[1:], n]
    run_len = run_end - run_start
    run_first_rank = rank[run_start]
    avg = run_first_rank + (run_len - 1) / 2.0
    rank = np.repeat(avg, run_len)

    npos = np.add.reduceat(y.astype(np.int64), gstart)
    nneg = glen - npos
    rpos = np.add.reduceat(np.where(y == 1, rank, 0.0), gstart)

    valid = (npos > 0) & (nneg > 0)
    num_t = rpos - npos * (npos + 1) / 2.0          # concordant pair count per step
    den_t = (npos * nneg).astype(np.float64)

    if not valid.any():
        out = 0.5
        return (out, None) if return_per_step else out

    if weighting == "pairs":
        val = float(num_t[valid].sum() / den_t[valid].sum())
    else:
        auc_t = num_t[valid] / den_t[valid]
        if weighting == "equal":
            w = np.ones_like(auc_t)
        elif weighting == "alive":
            w = (npos + nneg)[valid].astype(np.float64)
        else:
            raise ValueError(f"unknown weighting {weighting!r}")
        val = float((w * auc_t).sum() / w.sum())

    if return_per_step:
        per = {
            "t": t[gstart][valid],
            "auc": num_t[valid] / den_t[valid],
            "w": den_t[valid],
            "n_pos": npos[valid],
            "n_neg": nneg[valid],
        }
        return val, per
    return val


def ts_auc_ragged(scores, labels, **kw):
    """Reference/parity entry point taking lists of per-series trajectories."""
    sc = np.concatenate([np.asarray(s, dtype=np.float64) for s in scores])
    la = np.concatenate([np.asarray(lab, dtype=np.int8) for lab in labels])
    ti = np.concatenate([np.arange(len(s), dtype=np.int64) for s in scores])
    return ts_auc_flat(sc, la, ti, **kw)


def ts_auc_reference(scores, labels) -> float:
    """Slow, obviously-correct reference implementation (sklearn per step)."""
    from sklearn.metrics import roc_auc_score

    max_len = max((len(s) for s in scores), default=0)
    ws = 0.0
    tw = 0.0
    for t in range(max_len):
        st = [s[t] for s in scores if t < len(s)]
        lt = [int(lab[t]) for lab in labels if t < len(lab)]
        npos = sum(lt)
        nneg = len(lt) - npos
        if npos == 0 or nneg == 0:
            continue
        ws += npos * nneg * float(roc_auc_score(lt, st))
        tw += npos * nneg
    return ws / tw if tw > 0 else 0.5
