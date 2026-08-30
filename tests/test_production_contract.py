"""Production contract tests -- the things that must never regress.

These are deliberately NOT research checks.  They are the failure modes that
would cost the competition silently: a NaN reaching the scorer, a score outside
[0,1], a feature that peeks at the future, a model loaded against the wrong
feature bank, a series shape that crashes the engine mid-run.

Run: python3 -m pytest tests/test_production_contract.py -q
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from sbr.stream.engine import MODULE_ORDER, PRODUCTION_MODULES, StreamEngine

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.environ.get("SBR_MODEL_DIR", os.path.join(_REPO, "models", "rt150_ensemble"))
_HAS_MODEL = os.path.exists(os.path.join(MODEL_DIR, "manifest.json"))
needs_model = pytest.mark.skipif(not _HAS_MODEL, reason="no trained model directory")


# --------------------------------------------------------------- generators
def _hist(rng, n=2000, kind="normal"):
    if kind == "normal":
        return rng.standard_normal(n)
    if kind == "constant":
        return np.full(n, 2.5)
    if kind == "tiny_var":
        return rng.standard_normal(n) * 1e-12
    if kind == "heavy":
        return rng.standard_t(2.5, n)
    if kind == "ar":
        x = rng.standard_normal(n)
        for i in range(1, n):
            x[i] += 0.92 * x[i - 1]
        return x
    if kind == "vol":
        s, out = 1.0, np.empty(n)
        for i in range(n):
            s = 0.9 * s + 0.1 * (1 + 3 * (rng.random() < 0.05))
            out[i] = rng.standard_normal() * s
        return out
    raise ValueError(kind)


EDGE_ONLINE = {
    "len1": lambda r: r.standard_normal(1),
    "len10_minimum": lambda r: r.standard_normal(10),
    "len999_maximum": lambda r: r.standard_normal(999),
    "constant": lambda r: np.full(200, 0.7),
    "zeros": lambda r: np.zeros(150),
    "huge_outlier": lambda r: np.r_[r.standard_normal(60), 1e12, r.standard_normal(60)],
    "tiny_values": lambda r: r.standard_normal(120) * 1e-15,
    "tau0_scale_break": lambda r: r.standard_normal(180) * 5.0,
    "tau_last_point": lambda r: np.r_[r.standard_normal(179), 40.0],
    "very_late_tau": lambda r: np.r_[r.standard_normal(190), r.standard_normal(10) * 6],
    "monotone_ramp": lambda r: np.linspace(-3, 3, 250),
    "alternating": lambda r: np.tile([-2.0, 2.0], 100),
}


# ------------------------------------------------------------------- engine
@pytest.mark.parametrize("hkind", ["normal", "constant", "tiny_var", "heavy", "ar", "vol"])
@pytest.mark.parametrize("oname", list(EDGE_ONLINE))
def test_engine_never_crashes_and_never_emits_inf(hkind, oname):
    rng = np.random.default_rng(abs(hash((hkind, oname))) % 2**31)
    h = _hist(rng, 1500, hkind)
    o = EDGE_ONLINE[oname](rng)
    eng = StreamEngine().fit_historical(h)
    for t, x in enumerate(o):
        row = eng.step(x)
        assert row.shape == (500,), (hkind, oname, t, row.shape)
        assert row.dtype == np.float32
        assert not np.isinf(row).any(), f"inf at t={t} in {hkind}/{oname}"


def test_engine_handles_online_longer_than_the_buffer():
    """The competition caps online at 999, but the engine must not corrupt state
    if it ever sees more -- silent buffer wraparound is the worst failure mode."""
    rng = np.random.default_rng(3)
    h = _hist(rng, 1200)
    o = rng.standard_normal(1500)
    eng = StreamEngine().fit_historical(h)
    rows = [eng.step(x) for x in o]
    assert len(rows) == 1500
    assert not np.isinf(np.vstack(rows[-50:])).any()


#: the shipped RT-600 feature bank, and the sha256 over its 500 column names.
#: Both are frozen: `src/sbr/production/model.py::_check_manifest` refuses to
#: load a model whose engine does not reproduce this sha.
RT600_MODULES = ("m00_core", "m01_seq", "m02_dist", "m03_dyn",
                 "m04_resid", "m06_loc", "m07_bayes")
RT600_MANIFEST_SHA = "1646c3b9e09d8a7fb3c564483a1d1d999680caeefe848b092f907a6cac80cced"


def test_feature_order_is_immutable():
    """The PRODUCTION feature order, which is what the model manifest binds to.

    This test used to assert `MODULE_ORDER == the seven`, and it went red in
    wave 5 for a reason that was not a regression: wave 5 split one concept into
    two.  `MODULE_ORDER` is now the append-only column ORDERING over everything
    the streaming registry can build, and `PRODUCTION_MODULES` is the shipped
    default that `StreamEngine()` gives you when you ask for nothing.
    Registering `m12_rdep` appended to the former and deliberately left the
    latter alone, so no production column moved -- but the old assertion could
    not express that and failed on a true statement.

    The invariant is restated here in the three parts that actually matter.
    """
    # 1. the shipped default is, and stays, the seven RT-600 modules
    assert PRODUCTION_MODULES == RT600_MODULES

    # 2. MODULE_ORDER is APPEND-ONLY: research modules may be added after the
    #    production block but can never reorder or displace it
    assert MODULE_ORDER[:len(RT600_MODULES)] == RT600_MODULES
    assert len(set(MODULE_ORDER)) == len(MODULE_ORDER), "MODULE_ORDER has duplicates"

    # 3. registration is deterministic: every ordered module is buildable, and
    #    every buildable module is ordered -- otherwise `StreamEngine(modules)`
    #    would silently drop or reorder a block depending on set iteration
    from sbr.stream.engine import _CLASSES
    assert set(MODULE_ORDER) == set(_CLASSES), (
        f"MODULE_ORDER and _CLASSES disagree: "
        f"ordered-not-buildable {sorted(set(MODULE_ORDER) - set(_CLASSES))}, "
        f"buildable-not-ordered {sorted(set(_CLASSES) - set(MODULE_ORDER))}")

    rng = np.random.default_rng(0)
    eng = StreamEngine().fit_historical(_hist(rng))
    cols = eng.cols
    assert len(cols) == 500
    # NOTE: the columns are NOT sorted within a module -- each module emits its
    # own fixed order and only the BLOCKS are ordered.  An earlier version of
    # this test asserted `... or True`, which cannot fail; the real invariant is
    # below and it is the one the model manifest depends on.
    # the module blocks must be contiguous and in PRODUCTION_MODULES order
    seen = [c.split("::")[0] for c in cols]
    blocks = [k for i, k in enumerate(seen) if i == 0 or seen[i - 1] != k]
    assert tuple(blocks) == PRODUCTION_MODULES

    # 4. and the manifest the shipped model checks against is byte-identical
    import hashlib
    assert hashlib.sha256("\n".join(cols).encode()).hexdigest() == RT600_MANIFEST_SHA
    assert eng.manifest()["feature_manifest_sha256"] == RT600_MANIFEST_SHA


def test_appending_a_research_module_cannot_move_a_production_column():
    """The append-only claim, exercised rather than asserted.

    Build the engine on EVERY module MODULE_ORDER knows about and confirm the
    first 500 columns are exactly, and in the same order as, the RT-600 bank.
    """
    rng = np.random.default_rng(11)
    h = _hist(rng)
    seven = StreamEngine(RT600_MODULES).fit_historical(h).cols
    every = StreamEngine(MODULE_ORDER).fit_historical(h).cols
    assert every[:len(seven)] == seven
    assert len(every) >= len(seven)


def test_n_online_is_never_a_feature():
    """A feature that changes when only the FUTURE length changes is a leak.

    This is the exact defect that invalidated every pre-2026-06-08 submission.
    """
    rng = np.random.default_rng(11)
    h = _hist(rng, 2000)
    o_short = rng.standard_normal(120)
    o_long = np.r_[o_short, rng.standard_normal(500)]
    e1 = StreamEngine().fit_historical(h)
    r1 = np.vstack([e1.step(x) for x in o_short])
    e2 = StreamEngine().fit_historical(h)
    r2 = np.vstack([e2.step(x) for x in o_long])[:len(o_short)]
    assert np.array_equal(r1, r2, equal_nan=True), (
        "a feature depends on how long the series will be"
    )


# -------------------------------------------------------------------- model
@needs_model
def test_model_rejects_a_mismatched_feature_manifest(tmp_path):
    import shutil

    from sbr.production.model import ProductionModel
    shutil.copytree(MODEL_DIR, tmp_path / "m")
    man = json.load(open(tmp_path / "m" / "manifest.json"))
    man["feature_manifest_sha256"] = "0" * 64
    json.dump(man, open(tmp_path / "m" / "manifest.json", "w"))
    with pytest.raises(RuntimeError, match="FEATURE MANIFEST MISMATCH"):
        ProductionModel.load(str(tmp_path / "m"))


@needs_model
def test_infer_contract_single_pass_and_range():
    import sys
    sys.path.insert(0, os.path.join(_REPO, "research", "scripts"))
    from local_runner import run_infer

    from sbr.production.submission import infer
    rng = np.random.default_rng(5)
    series = [(_hist(rng, 1200, k), EDGE_ONLINE[o](rng))
              for k, o in [("normal", "len10_minimum"), ("constant", "constant"),
                           ("heavy", "huge_outlier"), ("ar", "tau0_scale_break")]]
    out = run_infer(infer, series, MODEL_DIR)
    assert len(out) == len(series)
    for s, (_, o) in zip(out, series):
        assert len(s) == len(o)
        assert np.isfinite(s).all()
        assert ((s >= 0) & (s <= 1)).all()


@needs_model
def test_infer_is_deterministic_and_order_independent():
    import sys
    sys.path.insert(0, os.path.join(_REPO, "research", "scripts"))
    from local_runner import run_infer

    from sbr.production.submission import infer
    rng = np.random.default_rng(6)
    series = [(_hist(rng, 1100), rng.standard_normal(60)) for _ in range(5)]
    a = run_infer(infer, series, MODEL_DIR)
    b = run_infer(infer, series, MODEL_DIR)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    order = [3, 0, 4, 1, 2]
    c = run_infer(infer, [series[i] for i in order], MODEL_DIR)
    assert all(np.array_equal(a[i], c[k]) for k, i in enumerate(order))


# --------------------------------------------------------------- provenance
# Added on engineering/rt600-final-reliability-2026 after applying the CRF-02
# void-run lesson to the production artifact: a valid checksum proves
# INTEGRITY, never PROVENANCE.  See
# engineering/reports/rt600_final_reliability/RT600_DEPLOYMENT_READINESS.md.
@needs_model
def test_frozen_model_passes_its_own_provenance_gate():
    from sbr.production.model import ProductionModel
    ProductionModel.load(MODEL_DIR)          # must not raise


@needs_model
@pytest.mark.parametrize("field,value", [
    (("trained_on", "n_series"), 24),            # the CRF-02 failure mode exactly
    (("trained_on", "n_series"), 8000),          # a dev-fold model, not the final fit
    (("trained_on", "partition"), "folds_dev"),  # the wrong partition
    (("folds_sha256",), "0" * 64),               # a different fold assignment
    (("calibration", "kind"), "logit_mean"),     # a rejected blend
    (("calibration", "time_coord"), "log_t"),    # the pre-W4-E3 coordinate
])
def test_model_with_wrong_provenance_is_refused(tmp_path, field, value):
    """The feature manifest still matches -- only the fit population is wrong."""
    import shutil

    from sbr.production.model import ProductionModel
    shutil.copytree(MODEL_DIR, tmp_path / "m")
    p = tmp_path / "m" / "manifest.json"
    man = json.load(open(p))
    node = man
    for k in field[:-1]:
        node = node[k]
    node[field[-1]] = value
    json.dump(man, open(p, "w"))

    with pytest.raises(RuntimeError, match="MODEL PROVENANCE MISMATCH"):
        ProductionModel.load(str(tmp_path / "m"))


@needs_model
def test_provenance_gate_has_a_documented_escape_hatch(tmp_path, monkeypatch):
    """Research must be able to load a different model; production must not."""
    import shutil

    from sbr.production.model import ProductionModel
    shutil.copytree(MODEL_DIR, tmp_path / "m")
    p = tmp_path / "m" / "manifest.json"
    man = json.load(open(p))
    man["trained_on"]["n_series"] = 24
    json.dump(man, open(p, "w"))

    monkeypatch.setenv("SBR_ALLOW_UNPINNED_MODEL", "1")
    ProductionModel.load(str(tmp_path / "m"))    # allowed, explicitly


@needs_model
def test_expected_provenance_matches_the_reproducibility_manifest():
    """The pin must agree with the frozen record, not drift from it."""
    from sbr.production.model import ProductionModel
    rec = json.load(open(os.path.join(
        _REPO, "research", "FINAL_REPRODUCIBILITY_MANIFEST.json")))
    exp = ProductionModel.EXPECTED_PROVENANCE
    assert exp["n_series"] == rec["training"]["n_series"]
    assert exp["partition"] == rec["training"]["partition"]
    assert exp["folds_sha256"] == rec["hashes"]["folds_final10k_sha256"]
    assert exp["n_boosters"] == len(rec["training"]["streams"])
    assert exp["calibration_kind"] == rec["calibration"]["kind"]
    assert exp["calibration_time_coord"] == rec["calibration"]["time_coord"]
