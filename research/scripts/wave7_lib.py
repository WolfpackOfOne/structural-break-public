"""WAVE-7 LEADERBOARD ASSAULT -- shared machinery for lanes A, B and C.

Binding pre-registration: research/WAVE7_PREREG.md.  Nothing architectural here
may be tuned after a TS-AUC is seen; if it changes, the experiment gets a new ID.

THE ONE RULE THAT ORGANISES THIS FILE
    A TEACHER QUANTITY IS A LABEL, NEVER A FEATURE.
    Teacher targets are computed from the FULL training series -- future
    observations and true tau included -- because the teacher is never deployed.
    The student sees only the causal 500-column state at t.  `teacher_targets`
    and `student_matrix` therefore live in different functions with different
    inputs, and `assert_no_teacher_in_features` is called on every student
    matrix built by a runner in this wave.

FORBIDDEN IN ANY STUDENT INPUT
    tau, future observations, n_online, final online length, boundary-
    conditioned availability, cross-series live state, the lockbox fold (-1),
    folds_final10k.parquet, any teacher output.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")

from sbr.metric import ts_auc_flat  # noqa: E402

REPORTS = f"{ROOT}/research/reports"

#: The champion's learner configuration, verbatim from `sbr.pipeline.run`.  Every
#: lane in wave 7 holds this fixed so that a delta is attributable to the lane
#: and not to a hyperparameter.  `n_estimators` is popped by the caller.
CHAMP_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63,
                    min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.7,
                    bagging_freq=1, lambda_l2=5.0, num_threads=2, verbose=-1,
                    max_bin=127, n_estimators=400)

PROD_MODULES = ("m00_core", "m01_seq", "m02_dist", "m03_dyn",
                "m04_resid", "m06_loc", "m07_bayes")

#: Lane B regimes.  LOG-SPACED, declared here BEFORE any score, and never
#: re-cut against TS-AUC (brief section 15: "Do not optimize cutpoints on
#: TS-AUC").  Edges are right-open; the last is unbounded.
HORIZON_EDGES = (0, 16, 41, 101, 251, 501, 10 ** 9)
HORIZON_NAMES = ("H1", "H2", "H3", "H4", "H5", "H6")

TEACHER_PREFIX = "teacher::"


# ===========================================================================
# leakage enforcement
# ===========================================================================
class TeacherLeak(AssertionError):
    """Raised when a teacher quantity reaches a student input."""


def assert_no_teacher_in_features(names) -> None:
    """A student feature matrix may not carry a teacher column, by name or origin."""
    bad = [n for n in names if str(n).startswith(TEACHER_PREFIX)]
    if bad:
        raise TeacherLeak(f"teacher columns reached the student matrix: {bad[:5]}")


def assert_causal_names(names) -> None:
    """Second gate: no student column may name a forbidden quantity."""
    banned = ("tau", "n_online", "future", "oracle", "boundary", "final_")
    bad = [n for n in names
           for b in banned
           if b in str(n).lower() and not str(n).lower().startswith("m0")]
    if bad:
        raise TeacherLeak(f"suspicious student column names: {sorted(set(bad))[:5]}")


#: LightGBM objectives that silently BINARISE their label.  `binary` maps any
#: label above zero to a positive, so a soft target in [0, 1] is thrown away
#: without a warning and the arm quietly becomes its own control.  Caught by the
#: wave-7 synthetic smoke run before it cost a fold of real compute; `cross_entropy`
#: is bit-identical to `binary` on a hard 0/1 label and honours a soft one.
BINARISING_OBJECTIVES = ("binary", "binary:logistic")
SOFT_LABEL_OBJECTIVE = "cross_entropy"


class SoftLabelMisuse(AssertionError):
    """Raised when a soft target is handed to an objective that would binarise it."""


def assert_objective_matches_label(objective: str, label) -> None:
    lab = np.asarray(label)
    hard = np.isin(lab, (0.0, 1.0)).all()
    if not hard and str(objective) in BINARISING_OBJECTIVES:
        raise SoftLabelMisuse(
            f"objective {objective!r} binarises its label, but the target is soft "
            f"(min {lab.min():.4f}, max {lab.max():.4f}). Use {SOFT_LABEL_OBJECTIVE!r}, "
            "which is bit-identical to 'binary' on a hard label.")


def sha_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ===========================================================================
# LANE A -- the teacher target bank
# ===========================================================================
#
# Every function below takes the WHOLE series and the TRUE tau.  That is legal
# and it is the point: these are offline labels for training rows, computed on
# training series only, and no deployed object may depend on one.
#
# The bank is deliberately small.  The wave-7 error budget (W7-D1) says 51% of
# remaining weighted loss sits at post-break age 100+ and 86% at current index
# t >= 100 -- mature, long-observed breaks -- while the young-age cells that a
# PERMANENCE target addresses hold 13.7% of the loss in total.  The targets are
# therefore ordered magnitude/evidence FIRST and permanence LAST, which reverses
# the ordering the wave-7 brief proposed.  The reversal is recorded in
# research/WAVE7_ALPHA_BUDGET.md section 6 with the numbers behind it.

def _welford_ref(x: np.ndarray) -> tuple[float, float]:
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    return m, max(s, 1e-12)


def teacher_series_summary(hist: np.ndarray, online: np.ndarray, tau: int) -> dict:
    """Full-sequence, per-SERIES privileged summary.  Offline only.

    Returns the quantities that describe HOW BIG and WHAT KIND the break is,
    measured with the whole series in hand.  For a no-break series every
    magnitude is 0 by construction, which is what makes these usable as
    regression targets across both classes.
    """
    hm, hs = _welford_ref(hist)
    if tau < 0 or tau >= len(online):
        post = online[:0]
    else:
        post = online[tau:]
    pre = online[:tau] if tau > 0 else online[:0]
    base = np.concatenate([hist, pre]) if pre.size else hist
    bm, bs = _welford_ref(base)

    if post.size < 2:
        return {"mag_mean": 0.0, "mag_logvar": 0.0, "mag_acf1": 0.0,
                "mag_total": 0.0, "n_post": int(post.size)}
    pm, ps = _welford_ref(post)
    mag_mean = abs(pm - bm) / bs
    mag_logvar = abs(float(np.log(ps / bs)))

    def _acf1(z):
        if len(z) < 3:
            return 0.0
        z = z - z.mean()
        d = float(z @ z)
        return float(z[:-1] @ z[1:]) / d if d > 0 else 0.0

    mag_acf1 = abs(_acf1(post) - _acf1(base))
    # A single scalar severity.  Weights are 1/1/1 -- declared, not fitted.
    mag_total = float(np.sqrt(mag_mean ** 2 + mag_logvar ** 2 + mag_acf1 ** 2))
    return {"mag_mean": float(mag_mean), "mag_logvar": float(mag_logvar),
            "mag_acf1": float(mag_acf1), "mag_total": mag_total,
            "n_post": int(post.size), "hist_mean": hm, "hist_sd": hs}


def teacher_evidence_path(hist: np.ndarray, online: np.ndarray, tau: int,
                          summ: dict | None = None) -> np.ndarray:
    """`tgt_evidence[t]` -- how strongly the FULL series says a break has
    occurred by t, as a probability-like score in [0, 1].

    Mechanism.  The hard label `y[t] = 1[t >= tau]` says a break happened.  It
    does not say the evidence is visible.  A weak break at age 3 is labelled
    exactly like a huge break at age 300, and under TS-AUC the first is
    unrankable while the second is trivial.  Forcing one learner to fit both at
    full confidence spends capacity on pairs that cannot be won.

    The teacher knows the true break parameters from the whole series, so it can
    say how much evidence a PERFECT detector would have accumulated by t:

        z[t] = mag_total * sqrt(post-break points held at t)      (0 before tau)
        evidence[t] = 1 - exp(-z[t] / SCALE)

    SCALE = 4.0 is declared here and never fitted.  The path is monotone
    non-decreasing in t, 0 for every no-break series, and depends on the future
    only through `mag_total`.
    """
    n = len(online)
    out = np.zeros(n, dtype=np.float32)
    if tau < 0 or tau >= n:
        return out
    s = summ or teacher_series_summary(hist, online, tau)
    age = np.arange(n - tau, dtype=np.float64) + 1.0
    z = s["mag_total"] * np.sqrt(age)
    out[tau:] = (1.0 - np.exp(-z / 4.0)).astype(np.float32)
    return out


def teacher_permanence(hist: np.ndarray, online: np.ndarray, tau: int,
                       horizon: int = 50) -> np.ndarray:
    """`tgt_permanence[t]` -- does the level displacement visible at t SURVIVE?

    For each t this compares the mean of the `horizon` points ending at t with
    the mean of everything after t, in historical sd units, and returns
    exp(-|difference|): 1 when the shift at t is still there later, 0 when it
    reverts.  Defined for BOTH classes -- a no-break series with a transient
    excursion gets a low value, which is precisely the hard negative the brief's
    section 47 wants, discovered by the teacher rather than by score-thresholding.

    Deprioritised by the W7-D1 budget (see the note above) but kept because it is
    the only target that separates transient from persistent directly.
    """
    n = len(online)
    _, hs = _welford_ref(hist)
    out = np.zeros(n, dtype=np.float32)
    cs = np.concatenate([[0.0], np.cumsum(online, dtype=np.float64)])
    for t in range(n):
        lo = max(0, t - horizon + 1)
        w_now = (cs[t + 1] - cs[lo]) / (t + 1 - lo)
        rest = n - (t + 1)
        if rest < 5:
            out[t] = out[t - 1] if t else 0.0
            continue
        w_fut = (cs[n] - cs[t + 1]) / rest
        out[t] = np.float32(np.exp(-abs(w_now - w_fut) / hs))
    return out


#: The declared bank.  Adding a target after a score exists is a protocol
#: violation; adding one before is a new pre-registration line.
TEACHER_TARGETS = {
    "evidence": "graded break-by-now evidence from full-sequence magnitude",
    "mag_total": "full-sequence break severity, per series, 0 for no-break",
    "permanence": "does the displacement visible at t survive the rest of the series",
}


def build_teacher_row_targets(store, series_ids, which=("evidence",)) -> dict:
    """Row-aligned teacher targets for the given TRAINING series only.

    The caller is responsible for passing training-fold series.  Passing a
    validation series is a fold-purity violation, and `wave7_a_distill` asserts
    against it rather than trusting the caller.
    """
    out = {k: [] for k in which}
    for i in series_ids:
        hist, online, tau = store.series(int(i))
        summ = teacher_series_summary(hist, online, tau)
        for k in which:
            if k == "evidence":
                out[k].append(teacher_evidence_path(hist, online, tau, summ))
            elif k == "mag_total":
                out[k].append(np.full(len(online), summ["mag_total"], np.float32))
            elif k == "permanence":
                out[k].append(teacher_permanence(hist, online, tau))
            else:
                raise KeyError(f"undeclared teacher target {k!r}")
    return {k: np.concatenate(v) if v else np.zeros(0, np.float32) for k, v in out.items()}


# ===========================================================================
# LANE B -- horizon routing on CURRENT t only
# ===========================================================================
def horizon_of(t: np.ndarray) -> np.ndarray:
    """Regime index from the CURRENT online index alone.

    Legal: `t` is observed, identical for every series compared at that step, and
    carries no information about tau.  This is the whole of the routing rule --
    there is no estimated break age and no inferred tau anywhere in lane B.
    """
    return np.digitize(np.asarray(t), HORIZON_EDGES[1:-1], right=False)


def horizon_weights(t: np.ndarray, n_regimes: int = len(HORIZON_NAMES),
                    width: float = 0.5) -> np.ndarray:
    """Overlapping soft routing weights, an ANALYTIC function of current t only.

    Brief section 16: hard regime boundaries may be unstable, and blending
    weights may not be searched on validation.  These are therefore a fixed
    Gaussian kernel in log(1+t) centred on each regime's log-midpoint, with
    `width` declared here.  No validation quantity enters.
    """
    t = np.asarray(t, dtype=np.float64)
    lo = np.array(HORIZON_EDGES[:-1], dtype=np.float64)
    hi = np.array([min(e, 1000.0) for e in HORIZON_EDGES[1:]], dtype=np.float64)
    centres = (np.log1p(lo) + np.log1p(hi)) / 2.0
    d = np.log1p(t)[:, None] - centres[None, :n_regimes]
    w = np.exp(-0.5 * (d / width) ** 2)
    return w / w.sum(axis=1, keepdims=True)


# ===========================================================================
# reporting -- age buckets are mandatory for every candidate (W5 section 5.5)
# ===========================================================================
AGE_BUCKETS = ((0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 10 ** 9))
T_BUCKETS = ((0, 20), (20, 50), (50, 100), (100, 200), (200, 400), (400, 10 ** 9))


def bucket_report(scores, y, t, age) -> dict:
    """TS-AUC restricted to pairs whose POSITIVE sits in each bucket.

    Negatives are never bucketed -- they have no age -- so each bucket keeps the
    full negative pool at its timestep, which is what makes the bucket AUCs
    combine back to the aggregate under the pair weights.
    """
    out = {"aggregate": float(ts_auc_flat(scores, y, t)), "by_age": {}, "by_t": {}}
    for lo, hi in AGE_BUCKETS:
        m = (y == 0) | ((age >= lo) & (age < hi))
        if (y[m] == 1).any():
            out["by_age"][f"{lo}-{hi if hi < 10**9 else ''}"] = float(
                ts_auc_flat(scores[m], y[m], t[m]))
    for lo, hi in T_BUCKETS:
        m = (t >= lo) & (t < hi)
        if m.any() and len(np.unique(y[m])) == 2:
            out["by_t"][f"{lo}-{hi if hi < 10**9 else ''}"] = float(
                ts_auc_flat(scores[m], y[m], t[m]))
    return out


def write_report(name: str, payload: dict) -> str:
    os.makedirs(REPORTS, exist_ok=True)
    payload.setdefault("written_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    path = f"{REPORTS}/{name}.json"
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    return path


def require_store(screen: bool = False):
    """Fail loudly and early, with the remedy, rather than deep inside a fold.

    `screen=True` selects the project's existing reduced protocol, which is also
    what the synthetic execution smoke test runs against.  A screen store is a
    development convenience: no research number may be reported from one.
    """
    from sbr.pipeline import STORE, STORE_SCREEN
    from sbr.store import load_store
    p = os.environ.get("SBR_STORE") or (STORE_SCREEN if screen else STORE)
    if not os.path.exists(os.path.join(p, "meta.parquet")):
        raise SystemExit(
            f"HARD STOP: the 2026 store is absent at {p}.\n"
            "Wave 7's pilots are all data-bound.  Set SBR_STORE to a built store,\n"
            "or build one, then re-run this script unchanged.")
    return load_store(p)
