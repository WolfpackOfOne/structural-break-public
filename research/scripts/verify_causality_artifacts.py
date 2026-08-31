"""Verify the ARTIFACT-LEVEL causality gates actually RUN, and that RT-600 is unchanged.

Four tests in tests/test_no_n_online_leakage.py are guarded by
`@needs_model` and silently SKIP when SBR_MODEL_DIR is unset.  A skipped
causality test is worse than a missing one, because the suite still says green.
This script proves they executed against the frozen artifact and passed, and
records the manifest identity alongside.

    SBR_MODEL_DIR=.../models/final10k_ensemble python research/scripts/verify_causality_artifacts.py
"""
from __future__ import annotations

import hashlib, json, os, re, subprocess, sys, time

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")

#: the RT-600 feature bank identity. `sbr.production.model._check_manifest`
#: refuses to load a model whose live engine does not reproduce this.
RT600_MANIFEST_SHA = "1646c3b9e09d8a7fb3c564483a1d1d999680caeefe848b092f907a6cac80cced"
RT600_MODULES = ["m00_core", "m01_seq", "m02_dist", "m03_dyn",
                 "m04_resid", "m06_loc", "m07_bayes"]
TESTS = [
    "tests/test_no_n_online_leakage.py::test_infer_never_measures_the_online_length",
    "tests/test_no_n_online_leakage.py::test_score_k_is_emitted_after_exactly_k_plus_one_points",
    "tests/test_no_n_online_leakage.py::test_a_prefix_scores_identically_however_the_series_continues",
    "tests/test_no_n_online_leakage.py::test_the_gate_itself_catches_a_length_peek",
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    md = os.environ.get("SBR_MODEL_DIR")
    if not md or not os.path.exists(os.path.join(md, "manifest.json")):
        raise SystemExit(
            "SBR_MODEL_DIR must point at a trained model directory, or the four "
            "artifact-level causality tests SKIP and prove nothing")

    man = json.load(open(os.path.join(md, "manifest.json")))
    out = {"schema": "sbr.causality_artifacts/1",
           "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip(),
           "model_dir": md}

    # ---- 1. the artifact's own manifest -----------------------------------
    out["artifact_manifest"] = {
        "modules": man["modules"],
        "n_columns": len(man["columns"]),
        "feature_manifest_sha256": man["feature_manifest_sha256"],
        "matches_frozen_rt600_sha": man["feature_manifest_sha256"] == RT600_MANIFEST_SHA,
        "modules_are_the_rt600_seven": man["modules"] == RT600_MODULES,
    }
    assert man["modules"] == RT600_MODULES, man["modules"]
    assert man["feature_manifest_sha256"] == RT600_MANIFEST_SHA
    assert len(man["columns"]) == 500

    # ---- 2. the LIVE engine still reproduces it ---------------------------
    import numpy as np
    from sbr.stream.engine import PRODUCTION_MODULES, StreamEngine
    eng = StreamEngine().fit_historical(np.random.default_rng(0).standard_normal(2000))
    live = eng.manifest()
    out["live_engine"] = {
        "default_modules": list(PRODUCTION_MODULES),
        "n_columns": len(eng.cols),
        "feature_manifest_sha256": live["feature_manifest_sha256"],
        "matches_artifact": live["feature_manifest_sha256"] == man["feature_manifest_sha256"],
        "columns_identical_to_artifact": list(eng.cols) == list(man["columns"]),
    }
    assert live["feature_manifest_sha256"] == RT600_MANIFEST_SHA
    assert list(eng.cols) == list(man["columns"]), "column ORDER drifted from the artifact"

    # ---- 3. the artifact's files ------------------------------------------
    files = {}
    for f in sorted(os.listdir(md)):
        p = os.path.join(md, f)
        if os.path.isfile(p):
            files[f] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
    out["artifact_files"] = files

    # ---- 4. the four tests RAN, and did not skip --------------------------
    env = dict(os.environ); env.setdefault("SBR_ROOT", ROOT)
    p = subprocess.run([sys.executable, "-m", "pytest", *TESTS, "-v", "-rs",
                        "--tb=short"], capture_output=True, text=True,
                       env=env, cwd=ROOT)
    txt = p.stdout + p.stderr
    per = {}
    for t in TESTS:
        name = t.split("::")[1]
        m = re.search(rf"{re.escape(name)}\s+(PASSED|FAILED|SKIPPED|ERROR)", txt)
        per[name] = m.group(1) if m else "NOT REPORTED"
    tail = re.search(r"(\d+) passed(?:, (\d+) skipped)?", txt)
    out["causality_tests"] = {
        "requested": len(TESTS), "per_test": per,
        "passed": int(tail.group(1)) if tail else -1,
        "skipped": int(tail.group(2) or 0) if tail else -1,
        "exit_code": p.returncode,
    }
    ok = (p.returncode == 0 and all(v == "PASSED" for v in per.values())
          and out["causality_tests"]["skipped"] == 0)
    out["verdict"] = ("ALL FOUR ARTIFACT-LEVEL CAUSALITY TESTS RAN AND PASSED; "
                      "RT-600 MANIFEST BYTE-IDENTICAL") if ok else "FAILED"

    dst = f"{ROOT}/research/reports/wave6_causality_artifacts.json"
    json.dump(out, open(dst, "w"), indent=1)
    print(json.dumps({k: out[k] for k in
                      ("model_dir", "artifact_manifest", "live_engine",
                       "causality_tests", "verdict")}, indent=1))
    print(f"\nwrote {dst}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
