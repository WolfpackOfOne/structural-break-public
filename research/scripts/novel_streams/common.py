"""Shared loader for the 2026 new-avenues diagnostics. READ-ONLY on the repo."""
import os
import sys

#: The checkout this file belongs to. Its `src` and `research/scripts` are
#: ALWAYS importable, so `import harness` works with no environment set up and
#: keeps working when SBR_ROOT points at a data-only tree.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

#: Where the data lives: cache/store, cache/features, research/folds,
#: research/oof. Defaults to REPO; SBR_ROOT points it elsewhere when the
#: caches are shared across worktrees (the usual local setup).
ROOT = os.environ.get("SBR_ROOT", REPO)

for _p in (f"{REPO}/src", f"{REPO}/research/scripts",
           f"{ROOT}/src", f"{ROOT}/research/scripts"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402,F401
from wave5_lib import SPECIALISTS, Ctx, load_oof  # noqa: E402,F401

from sbr.metric import ts_auc_flat  # noqa: E402,F401

#: Diagnostic artifacts land under cache/, which .gitignore already covers.
#: They used to be written next to the source, which left .npz files untracked
#: in the working tree of whoever ran a diagnostic. These are runner scripts,
#: so the directory is created here rather than lazily at each save site.
OUT = os.path.join(ROOT, "cache", "novel_streams")
os.makedirs(OUT, exist_ok=True)

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
