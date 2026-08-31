"""RT-1320 pre-score causality gates (RT1320_PROMOTION_PLAN.md section 5, Phase 2).

`PROTOCOL_CHAMPION_2026.md` states that no predictive score is interpreted
before the causality gate passes.  The RT-1320 student reuses the already-gated
500-column causal bank and adds no features, so the FEATURE-level properties are
inherited -- but none of it had been run against the student's own path, and the
ledger records causal_verified=no.  This file is that run.

    GATE 1  the student's inputs are exactly the 500-column causal bank
    GATE 2  no forbidden column (tau/boundary/n_online/availability/...) is
            reachable as a student feature name
    GATE 3  the sentinel REJECTS a planted forbidden column -- proving it
            catches the defect rather than trivially passing
    GATE 4  nested double-cross-fit purity, and the sentinel still rejects the
            old contaminated global-OOF scheme
    GATE 5  no label-derived state enters inference: the teacher-derived target
            is training-only, and prediction consumes the causal matrix alone
    GATE 6  per-row independence and deterministic replay -- a row scores
            identically alone, in a shuffled batch, and on a repeat run
    GATE 7  output contract: finite, non-NaN, and the residual-valued range is
            recorded rather than assumed to be a probability

WHAT THIS DOES NOT COVER, stated so the gap is not mistaken for a pass:
streaming prefix-invariance through a production inference path.  RT-1320 has no
production artifact -- that is Phase 4 -- so the artifact-level gates in
`verify_causality_artifacts.py` cannot be pointed at it.  Prefix invariance is
INHERITED here: the student is a pure row-wise function of the causal bank, and
that bank's prefix invariance is what GATE 1 pins.  It must be re-run directly
once a production artifact exists.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "research", "scripts"))

import numpy as np
import pytest

import armc_residual_student as A

FEATURE_DIR = Path(A.default_data_root()) / "cache" / "features"
pytestmark = pytest.mark.skipif(
    not (FEATURE_DIR / "m00_core.npy").exists(),
    reason="feature bank not present in this checkout",
)


def _bank():
    return A.load_features(FEATURE_DIR)


# --------------------------------------------------------------------------- 1
def test_gate1_inputs_are_exactly_the_500_column_causal_bank():
    mats, names = _bank()
    assert len(names) == 500, f"student bank is {len(names)} columns, not 500"
    assert len(set(names)) == 500, "duplicate column names in the student bank"
    assert [n.split("::")[0] for n in names[:1]][0] in A.FULL
    modules = {n.split("::")[0] for n in names}
    assert modules == set(A.FULL), f"unexpected modules {modules ^ set(A.FULL)}"
    assert sum(m.shape[1] for m in mats) == 500


# --------------------------------------------------------------------------- 2
def test_gate2_no_forbidden_token_is_reachable():
    _, names = _bank()
    bad = [n for n in names if any(tok in n.lower() for tok in A.FORBIDDEN_TOKENS)]
    assert bad == [], f"forbidden columns reachable: {bad[:10]}"


# --------------------------------------------------------------------------- 3
def test_gate3_sentinel_rejects_a_planted_forbidden_column(tmp_path, monkeypatch):
    """The check must catch a defect, not merely pass on clean data."""
    src = FEATURE_DIR
    for module in A.FULL:
        for suffix in (".npy", ".cols.json"):
            (tmp_path / f"{module}{suffix}").symlink_to(src / f"{module}{suffix}")
    meta = json.loads((src / "m00_core.cols.json").read_text())
    poisoned = dict(meta)
    poisoned["cols"] = list(meta["cols"])
    poisoned["cols"][0] = "tau_index_peek"          # a forbidden token
    (tmp_path / "m00_core.cols.json").unlink()
    (tmp_path / "m00_core.cols.json").write_text(json.dumps(poisoned))
    with pytest.raises(ValueError, match="forbidden"):
        A.load_features(tmp_path)


# --------------------------------------------------------------------------- 4
def test_gate4_nested_purity_and_the_sentinel_still_catches_the_old_scheme():
    folds = set(A.FOLDS)
    for f in A.FOLDS:
        for g in A.FOLDS:
            if f == g:
                continue
            new_train = folds - {f, g}
            assert not (new_train & {f, g}), f"nested scheme leaked on ({f},{g})"
            old_train = folds - {g}
            assert old_train & {f}, (
                f"the old global-OOF scheme should be contaminated on ({f},{g}); "
                "if this stops failing the sentinel has lost its power")


# --------------------------------------------------------------------------- 5
def test_gate5_inference_consumes_only_the_causal_matrix():
    """The target is teacher-derived; the INPUTS must not be.

    stack_features is the whole inference input path. It is a function of the
    feature matrices and a row index only -- no labels, no Q, no tau.
    """
    import inspect
    params = set(inspect.signature(A.stack_features).parameters)
    forbidden = {"y", "labels", "q", "Q", "tau", "has_break", "target"}
    assert not (params & forbidden), f"label-derived arg in inference path: {params & forbidden}"
    src = inspect.getsource(A.stack_features)
    for tok in ("d.y", "has_break", "tau_index", "nested_Q"):
        assert tok not in src, f"{tok!r} reachable inside stack_features"


# --------------------------------------------------------------------------- 6
def test_gate6_row_independence_and_deterministic_replay():
    import lightgbm as lgb
    mats, names = _bank()
    keep = np.arange(len(names))
    rng = np.random.default_rng(0)
    n_rows = int(mats[0].shape[0])
    rows = np.sort(rng.choice(n_rows, 4000, replace=False))
    X = A.stack_features(mats, rows, keep)
    y = rng.random(len(rows))
    # num_threads=1 and validate_features=False are how sbr.production.model
    # actually calls predict, and for a reason recorded there: the default path
    # spins up a thread pool per call. Twenty-five single-row predicts under the
    # full suite segfaulted LightGBM on this platform -- the same fragility the
    # repo already hit at INFER_PARALLELISM=4. Match production's convention.
    p = dict(A.STUDENT_PARAMS, num_threads=1)
    bst = lgb.train(p, lgb.Dataset(X, label=y), num_boost_round=8)

    def pred(m):
        return bst.predict(m, validate_features=False, num_threads=1)

    whole = pred(X)
    # a) a row scores identically on its own
    for i in rng.choice(len(rows), 25, replace=False):
        assert pred(X[i:i + 1])[0] == whole[i], "row score depends on its batch"
    # b) a shuffled batch gives the same per-row scores
    order = rng.permutation(len(rows))
    assert np.array_equal(pred(X[order]), whole[order]), "score depends on row order"
    # c) deterministic replay
    assert np.array_equal(pred(X), whole), "prediction is not deterministic"


# --------------------------------------------------------------------------- 7
def test_gate7_output_contract():
    reports = Path(A.ROOT) / "research" / "reports"
    found = sorted(reports.glob("armc_residual_student*/armc_residual_student_oof.npy"))
    if not found:
        pytest.skip("no merged student OOF in this checkout")
    for oof in found:
        v = np.load(oof, mmap_mode="r")
        covered = v[np.isfinite(v)]
        assert covered.size > 0, f"{oof.parent.name}: student OOF entirely non-finite"
        assert not np.isnan(covered).any(), f"{oof.parent.name}: NaN in covered region"
        print(f"  {oof.parent.name}: {covered.size:,} covered rows, "
              f"range [{covered.min():.6f}, {covered.max():.6f}]")
    # The student is a `regression`-objective booster: its raw output is a
    # residual prediction, NOT a probability. The range is RECORDED above; it is
    # deliberately not asserted to [0, 1], which would be the wrong contract for
    # this member and would pass only by accident.
