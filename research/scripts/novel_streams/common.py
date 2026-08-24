"""Shared loader for the 2026 new-avenues diagnostics. READ-ONLY on the repo."""
import os, sys, json
ROOT = os.environ.get("SBR_ROOT")
sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{ROOT}/research/scripts")
import numpy as np
from wave5_lib import Ctx, SPECIALISTS, SEEDCLONES, load_oof, FOLDS, OOFDIR
from sbr.metric import ts_auc_flat

OUT = os.path.dirname(os.path.abspath(__file__))

def rt600(c=None, cache=True):
    """Cross-fitted SCDF equal-weight blend of the seven RT-600 specialists."""
    f = f"{OUT}/rt600_blend.npy"
    if cache and os.path.exists(f):
        return np.load(f)
    c = c or Ctx()
    P = load_oof(SPECIALISTS)
    v = c.crossfit_blend(P, SPECIALISTS)
    np.save(f, v)
    return v

def cell_mask(c, rows, t_min=200, age_min=100):
    """Dominant-cell restriction: t>=t_min, positives with age>=age_min, all negatives."""
    y, t, a = c.d.y[rows], c.d.t[rows], c.age[rows]
    keep = (t >= t_min) & ((y == 0) | (a >= age_min))
    return keep
