"""Novel-stream harness: portability and the batch/stream parity sentinel.

These tests exist because of two concrete defects, not as generic coverage:

  1. `research/scripts/novel_streams/harness.py` used to `raise SystemExit` at
     import time unless `SBR_ROOT` was set, and `common.py` silently built a
     "None/src" sys.path entry, so a fresh `research/current` checkout hit an
     opaque failure before any research could start.
  2. `verify()` -- the bitwise prefix-invariance check -- rejected the shipped
     reference mechanism on its first run because the full replay emitted 0.0
     where the truncated replay emitted NaN. That is not a look-ahead, but it
     is the batch/stream parity defect this repo treats as its largest
     deployment risk. The sentinel must keep catching it.

Everything here runs on synthetic series with NO competition data, so it is
valid on a clean CI checkout. Nothing here trains a model, scores a fold,
consumes an experiment ID, or touches RESULTS.csv.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOVEL = os.path.join(REPO, "research", "scripts", "novel_streams")
SCRIPTS = os.path.join(REPO, "research", "scripts")

sys.path.insert(0, NOVEL)
sys.path.insert(0, SCRIPTS)

harness = pytest.importorskip("harness", reason="novel_streams harness not on this branch")


# --------------------------------------------------------------------- A
def test_harness_imports_with_no_environment_at_all():
    """A fresh checkout must not need SBR_ROOT, PYTHONPATH, or any cache.

    Run in a subprocess with a scrubbed environment so the ambient pytest
    configuration (which puts `src` on the path) cannot mask a regression.
    """
    env = {k: v for k, v in os.environ.items()
           if k not in ("SBR_ROOT", "PYTHONPATH", "PYTHONSTARTUP")}
    code = (
        f"import sys; sys.path.insert(0, {NOVEL!r})\n"
        "import os\n"
        "import harness\n"
        "assert os.path.isdir(os.path.join(harness.REPO, 'src'))\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"import failed with a clean environment:\n{r.stderr}"
    assert "OK" in r.stdout


def test_harness_import_has_no_filesystem_side_effects(tmp_path):
    """Importing the harness must not create directories in the checkout."""
    env = {k: v for k, v in os.environ.items() if k not in ("SBR_ROOT", "PYTHONPATH")}
    env["SBR_ROOT"] = str(tmp_path)
    code = (
        f"import sys; sys.path.insert(0, {NOVEL!r})\n"
        "import os\n"
        "import harness\n"
        "print('EXISTS' if os.path.exists(harness.CACHE) else 'CLEAN')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "CLEAN" in r.stdout, "harness created its cache directory as an import side effect"


# --------------------------------------------------------------------- B
def test_wave8_shared_helpers_are_importable():
    """ensemble_marginal / pair_repair_stats must resolve from research/scripts."""
    import wave8_common as W8

    assert callable(W8.ensemble_marginal)
    assert callable(W8.pair_repair_stats)


def test_harness_lazy_imports_name_only_present_helpers():
    """Every deferred `from wave8_common import X` in the harness must exist."""
    import wave8_common as W8

    src = open(os.path.join(NOVEL, "harness.py")).read()
    wanted = set()
    for line in src.splitlines():
        line = line.strip()
        if line.startswith("from wave8_common import "):
            wanted.update(n.strip() for n in line.split("import", 1)[1].split(","))
    assert wanted, "harness no longer defers to wave8_common -- update this test"
    missing = sorted(n for n in wanted if not hasattr(W8, n))
    assert not missing, f"harness imports names absent from wave8_common: {missing}"


# --------------------------------------------------------------------- C
def test_no_sibling_worktree_dependency():
    """The harness's dependencies must live inside THIS checkout.

    Wave 8 stays unmerged; only its reusable helpers were ported into
    research/current. If wave8_common ever resolves to another worktree, a
    fresh clone would break in a way that is invisible on this machine.
    """
    import wave5_lib
    import wave8_common

    for mod in (harness, wave8_common, wave5_lib):
        path = os.path.realpath(mod.__file__)
        assert path.startswith(os.path.realpath(REPO)), (
            f"{mod.__name__} resolved to {path}, outside this checkout")


def test_novel_streams_has_no_stale_branch_setup_instructions():
    """No file may tell a future agent to fetch wave8_common from a branch."""
    banned = ("cherry-pick", "wave8-future-aware", "wave7-t2-promotion")
    offenders = []
    for name in sorted(os.listdir(NOVEL)):
        if not name.endswith((".py", ".md")):
            continue
        text = open(os.path.join(NOVEL, name)).read().lower()
        for token in banned:
            if token in text:
                offenders.append(f"{name}: {token}")
    assert not offenders, f"stale cross-branch setup instructions: {offenders}"


# --------------------------------------------------------------------- toys
def _synthetic_series(n_hist=600, n_online=240, seed=0):
    rng = np.random.default_rng(seed)
    hist = rng.normal(0.0, 1.0, n_hist)
    online = np.concatenate([rng.normal(0.0, 1.0, n_online // 2),
                             rng.normal(0.0, 2.0, n_online - n_online // 2)])
    return hist, online


SERIES = [_synthetic_series(seed=s) + (f"synthetic{s}",) for s in (0, 1, 2)]


class GoodMech(harness.StreamingMechanism):
    """Causal: row t sees only hist and online[:t+1]. NaN until the window fills."""

    name = "toy_good"
    cols = ["roll8"]
    W = 8

    def fit_history(self, hist):
        return float(np.mean(hist))

    def emit(self, hist, online):
        mu = self.fit_history(hist)
        n = len(online)
        out = np.full((n, 1), np.nan, dtype=np.float32)
        c = np.concatenate([[0.0], np.cumsum(online - mu)])
        if n >= self.W:
            out[self.W - 1:, 0] = (c[self.W:] - c[:-self.W]) / self.W
        return out


class ZeroFillMech(GoodMech):
    """THE REAL DEFECT: 0.0 before the window fills in the full replay, but an
    all-NaN block when the online segment is shorter than the window."""

    name = "toy_zero_fill"

    def emit(self, hist, online):
        mu = self.fit_history(hist)
        n = len(online)
        out = np.full((n, 1), np.nan, dtype=np.float32)
        if n < self.W:
            return out
        out[:, 0] = 0.0
        c = np.concatenate([[0.0], np.cumsum(online - mu)])
        out[self.W - 1:, 0] = (c[self.W:] - c[:-self.W]) / self.W
        return out


class LookAheadMech(GoodMech):
    """Non-causal: normalises by a statistic of the WHOLE online segment."""

    name = "toy_look_ahead"

    def emit(self, hist, online):
        out = super().emit(hist, online)
        return out / max(float(np.std(online)), 1e-9)


# --------------------------------------------------------------------- D
def test_verify_rejects_zero_versus_nan_disagreement():
    ok, msg = harness.verify(ZeroFillMech(), series=SERIES)
    assert not ok, "sentinel accepted the 0.0-vs-NaN batch/stream parity defect"
    assert "toy_zero_fill" in msg


def test_verify_rejects_look_ahead():
    ok, msg = harness.verify(LookAheadMech(), series=SERIES)
    assert not ok, "sentinel accepted a mechanism that reads the whole online segment"
    assert "toy_look_ahead" in msg


def test_verify_rejects_shape_disagreement():
    class RaggedMech(GoodMech):
        name = "toy_ragged"

        def emit(self, hist, online):
            out = super().emit(hist, online)
            return out[:-1] if len(online) < 100 else out

    ok, msg = harness.verify(RaggedMech(), series=SERIES)
    assert not ok and "toy_ragged" in msg


# --------------------------------------------------------------------- E
def test_verify_accepts_a_causal_mechanism():
    ok, msg = harness.verify(GoodMech(), series=SERIES)
    assert ok, f"sentinel rejected a causal mechanism: {msg}"
    assert msg == "ok"


def test_shipped_reference_mechanism_passes_verify():
    """m20_dwell_probe is the skeleton Codex starts the first pilot from."""
    import m20_dwell_probe

    ok, msg = harness.verify(m20_dwell_probe.DwellProbe(),
                             series=[_synthetic_series(n_hist=900, n_online=400, seed=s) + (s,)
                                     for s in (0, 1)])
    assert ok, f"reference mechanism no longer passes prefix invariance: {msg}"
