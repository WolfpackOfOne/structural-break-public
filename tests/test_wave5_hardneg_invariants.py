"""W5-E3 hard-negative invariants -- conformance, not a new hypothesis.

RT-710 / RT-711 / RT-712 are already observed and recorded (W5-E3 was rejected:
reweighting -0.00701, oversampling -0.01339).  Nothing here changes a number.
These tests pin two properties that were previously true by construction and
are now true by assertion:

  1. the oversampling pool is the hardest HARD_FRAC of ELIGIBLE INNER-FOLD
     NEGATIVES, selected by quantile of the same-t inversion rank -- not every
     row whose rank happens to exceed a fixed constant.  Those two coincide only
     if hardness is a global percentile, and it stopped being one.

  2. no row of the held-out outer fold can enter the training index, not even
     once.  The duplicated rows are drawn from the training-fold row list, never
     from `arange(n_rows)`, and both the pool builder and the `rows_for` patch
     now assert it.

Realised measurements, all five outer folds:
  selected fraction 0.10000 exactly; 0 held-out rows in any pool; 0 positives
  ever duplicated; training index length exactly len(train) + 3 * len(pool).
  research/reports/wave5_e3_conformance.json
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "research", "scripts"))

ROOT = os.environ.get("SBR_ROOT", _REPO)
HARD = f"{ROOT}/research/oof/wave5_hardness.npy"
pytestmark = pytest.mark.skipif(
    not (os.path.exists(HARD) and os.path.exists(f"{ROOT}/cache/store/meta.parquet")),
    reason="needs the store and the wave-5 nested hardness vector")


@pytest.fixture(scope="module")
def env():
    from wave5_e3_hardneg import FOLDS, HARD_FRAC, OVER_K, oversample_pool

    from sbr.pipeline import Data
    d = Data()
    H = np.load(HARD, mmap_mode="r")
    return d, H, FOLDS, HARD_FRAC, OVER_K, oversample_pool


def test_pool_is_the_hardest_fraction_not_a_fixed_threshold(env):
    d, H, FOLDS, HARD_FRAC, _, pool = env
    for k in FOLDS:
        m = np.isfinite(H[k]) & (d.y == 0) & (d.row_fold != k)
        sel = pool(d, H, k)
        frac = sel.sum() / m.sum()
        assert abs(frac - HARD_FRAC) < 1e-3, (
            f"outer fold {k}: realised fraction {frac:.5f} != HARD_FRAC "
            f"{HARD_FRAC}. A fixed rank threshold instead of a quantile of the "
            "eligible negatives is exactly how this drifts.")


def test_pool_never_contains_the_held_out_fold_or_a_positive(env):
    d, H, FOLDS, _, _, pool = env
    for k in FOLDS:
        sel = pool(d, H, k)
        assert not sel[d.row_fold == k].any(), f"pool for outer fold {k} leaked fold {k}"
        assert not sel[d.y == 1].any(), f"pool for outer fold {k} duplicated a positive"


def test_training_index_excludes_the_held_out_fold(env):
    """The invariant the whole experiment rests on, asserted end to end."""
    d, H, FOLDS, _, OVER_K, pool = env
    from wave5_e3_hardneg import install_oversampler, restore_oversampler
    masks = {k: pool(d, H, k) for k in FOLDS}
    install_oversampler(masks)
    try:
        for k in FOLDS:
            tr = [f for f in FOLDS if f != k]
            idx = d.rows_for(tr)
            assert not (d.row_fold[idx] == k).any(), (
                f"outer fold {k}: the oversampled TRAINING index contains "
                "held-out rows")
            base = int(np.isin(d.row_fold, tr).sum())
            assert len(idx) == base + (OVER_K - 1) * int(masks[k].sum())
            # the validation and pooled calls must be untouched
            assert np.array_equal(d.rows_for([k]), np.flatnonzero(d.row_fold == k))
            five = d.rows_for(list(FOLDS))
            assert len(five) == int(np.isin(d.row_fold, list(FOLDS)).sum())
    finally:
        restore_oversampler()


def test_restore_is_clean(env):
    d, H, FOLDS, _, _, pool = env
    from wave5_e3_hardneg import install_oversampler, restore_oversampler
    before = {k: d.rows_for([f for f in FOLDS if f != k]).copy() for k in FOLDS}
    install_oversampler({k: pool(d, H, k) for k in FOLDS})
    restore_oversampler()
    for k in FOLDS:
        assert np.array_equal(d.rows_for([f for f in FOLDS if f != k]), before[k])
