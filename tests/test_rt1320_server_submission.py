from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submissions" / "I_rt1320_final10k_fit.py"


def _load_submission():
    spec = importlib.util.spec_from_file_location("rt1320_final10k_fit", SUBMISSION)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rt1320_server_submission_smoke_train_and_dummy_infer(monkeypatch):
    mod = _load_submission()
    monkeypatch.setenv(mod.SMOKE_ENV, "1")
    with tempfile.TemporaryDirectory() as td:
        mod.train([], td)
        smoke = Path(td) / mod.OUTPUT_NAME / "SMOKE_RESULT.json"
        assert smoke.exists()

        online = [3.0, 4.0, 5.0]
        out = list(mod.infer([([1.0, 2.0], online)], td))
        assert out == [None, 0.5, 0.5, 0.5]


def test_rt1320_fold_purity_sentinel_catches_old_scheme():
    mod = _load_submission()
    report = mod._fold_purity_report()
    assert report["new_nested_passed"]
    assert report["checked_outer_inner_pairs"] == 20
    assert report["old_global_oof_contaminated_checks"] == 20
    assert report["old_global_oof_sentinel_catches_defect"]


def test_rt1320_submission_is_research_only():
    mod = _load_submission()
    assert mod.INFER_PARALLELISM == 1
    assert "RESEARCH" in mod.LABEL
    assert "LEADERBOARD" not in mod.LABEL or "ONLY" in mod.__doc__
    assert os.path.basename(str(SUBMISSION)) == "I_rt1320_final10k_fit.py"
