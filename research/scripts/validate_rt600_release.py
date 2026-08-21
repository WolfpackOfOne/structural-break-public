#!/usr/bin/env python3
"""Verify — never modify — that an RT-600 release checkout is submittable.

This script is READ-ONLY by construction.  It does not train, rebuild, repack,
recalibrate, or write anything anywhere.  It answers exactly one question:

    is the artifact on disk right now the frozen RT-600 artifact, in the right
    checkout, pointed at the right competition?

It exists because the LB-001 release found two traps worth failing loudly on:

  1. `crunch` walks *upward* for `.crunchdao/project.json`.  There is a config in
     the home directory for the 2025 `structural-break` competition.  If the
     nearest config is not the real-time one, the test you just "passed" was the
     wrong competition and proves nothing.
  2. `tests/test_no_n_online_leakage.py` SKIPS when it cannot find a model.  A
     skipped causality gate is not a release pass, it just looks like one.

Usage:
    python3 research/scripts/validate_rt600_release.py [--model-directory resources]
                                                       [--skip-tests]

Exit 0 = every gate passed.  Exit 1 = at least one gate failed; do not submit.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import zipfile

# The frozen RT-600 identity, from research/FINAL_REPRODUCIBILITY_MANIFEST.json.
# These are the hashes of the artifact submitted as LB-001 on 2026-08-21.
EXPECTED = {
    "source_zip": "199db8c9f5db7e1429f6ae09fa018d43cc58c5eb47b5c623a017799537a413a0",
    "model_zip": "6c8960ddc7331afecfc6980974f2879401a1eb583f15f5cad34c2ea03299ea8c",
    "python_entrypoint": "660c88c104c990626a5758f8d36af49c87273a587e399d86bbc353cc9bba2647",
    "notebook": "8332b698b6dbe91376c79d2051c5d294f515f77dea835a278b858b45bda73cc5",
    "feature_manifest": "1646c3b9e09d8a7fb3c564483a1d1d999680caeefe848b092f907a6cac80cced",
    "model_manifest": "1483a59a268ded18b6af804a5f0eb0b191376a1dc86526f286751bf1613ec940",
}
COMPETITION = "structural-break-real-time"
MODEL_CODE_SHA = "41ab0695a906834361298b4a61c3636909ceb79c"
ENTRYPOINT = "submissions/C_ensemble_deployable.py"
NOTEBOOK = "submissions/C_ensemble_deployable.ipynb"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(REPO)  # research/scripts -> research -> repo root

_failures: list[str] = []
_checks = 0


def check(ok: bool, label: str, detail: str = "") -> bool:
    global _checks
    _checks += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        _failures.append(label)
    return ok


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", REPO, *args],
                          capture_output=True, text=True).stdout.strip()


def section(name: str) -> None:
    print(f"\n{name}")


def read_payloads(py_path: str) -> tuple[dict, dict]:
    """Decode the artifact's embedded zips and return (hashes, model manifest)."""
    text = open(py_path, encoding="utf-8").read()
    hashes, manifest = {}, {}
    for var, key in (("_SRC_B64", "source_zip"), ("_MDL_B64", "model_zip")):
        m = re.search(var + r' = "([A-Za-z0-9+/=]+)"', text)
        if not m:
            return {}, {}
        raw = base64.b64decode(m.group(1))
        hashes[key] = hashlib.sha256(raw).hexdigest()
        if key == "model_zip":
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                name = next(n for n in z.namelist() if n.endswith("manifest.json"))
                blob = z.read(name)
                hashes["embedded_model_manifest"] = hashlib.sha256(blob).hexdigest()
                manifest = json.loads(blob)
        else:
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                hashes["source_members"] = sorted(
                    n for n in z.namelist() if not n.endswith("/"))
    return hashes, manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-directory", default="resources",
                    help="model/resource directory to validate (default: resources)")
    ap.add_argument("--skip-tests", action="store_true",
                    help="skip the release-critical pytest run (NOT a release pass)")
    a = ap.parse_args()

    model_dir = a.model_directory
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(REPO, model_dir)
    py_path = os.path.join(REPO, ENTRYPOINT)

    print(f"RT-600 release validation (read-only)\nrepo: {REPO}")

    # ---- 1. checkout -------------------------------------------------------
    section("1. checkout")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain")
    check(bool(head), "HEAD resolves", head)
    print(f"       branch: {branch}")
    check(git("cat-file", "-t", MODEL_CODE_SHA) == "commit",
          "model training SHA exists in git", MODEL_CODE_SHA[:10])
    check(subprocess.run(["git", "-C", REPO, "merge-base", "--is-ancestor",
                          MODEL_CODE_SHA, "HEAD"]).returncode == 0,
          "model training SHA is an ancestor of HEAD")
    tracked_dirty = [l for l in dirty.splitlines() if not l.startswith("??")]
    check(not tracked_dirty, "no modifications to tracked files",
          "; ".join(tracked_dirty[:3]) if tracked_dirty else "clean")

    # ---- 2. artifact hashes ------------------------------------------------
    section("2. artifact identity")
    if not check(os.path.exists(py_path), "entrypoint exists", ENTRYPOINT):
        return finish()
    py_sha = sha256_file(py_path)
    check(py_sha == EXPECTED["python_entrypoint"], "python entrypoint sha256",
          py_sha if py_sha != EXPECTED["python_entrypoint"] else "matches frozen RT-600")
    nb = os.path.join(REPO, NOTEBOOK)
    if os.path.exists(nb):
        nb_sha = sha256_file(nb)
        check(nb_sha == EXPECTED["notebook"], "notebook sha256",
              nb_sha if nb_sha != EXPECTED["notebook"] else "matches frozen RT-600")

    hashes, manifest = read_payloads(py_path)
    if not check(bool(manifest), "embedded payloads decode"):
        return finish()
    check(hashes["source_zip"] == EXPECTED["source_zip"], "embedded source zip sha256",
          hashes["source_zip"] if hashes["source_zip"] != EXPECTED["source_zip"] else "matches")
    check(hashes["model_zip"] == EXPECTED["model_zip"], "embedded model zip sha256",
          hashes["model_zip"] if hashes["model_zip"] != EXPECTED["model_zip"] else "matches")

    # ---- 3. model directory ------------------------------------------------
    section(f"3. model directory ({os.path.relpath(model_dir, REPO)})")
    man_path = os.path.join(model_dir, "manifest.json")
    if not check(os.path.exists(man_path), "model manifest present", man_path):
        return finish()
    disk_man_sha = sha256_file(man_path)
    check(disk_man_sha == EXPECTED["model_manifest"], "model manifest sha256",
          disk_man_sha if disk_man_sha != EXPECTED["model_manifest"] else "matches")
    check(disk_man_sha == hashes["embedded_model_manifest"],
          "on-disk model == model embedded in the artifact")

    # ---- 4. provenance -----------------------------------------------------
    section("4. provenance")
    trained = manifest.get("trained_on", {})
    check(trained.get("n_series") == 10000, "trained on 10,000 series",
          str(trained.get("n_series")))
    check(manifest.get("n_features") == 500, "feature count == 500",
          str(manifest.get("n_features")))
    check(manifest.get("feature_manifest_sha256") == EXPECTED["feature_manifest"],
          "feature manifest sha256")
    calib = manifest.get("calibration", {})
    check(calib.get("kind") == "scdf", "calibration kind == scdf", str(calib.get("kind")))
    check(calib.get("time_coord") == "log_n_seen",
          "calibration time_coord == log_n_seen", str(calib.get("time_coord")))
    coords = {m.get("time_coord") for m in calib.get("models", [])}
    check(coords == {"log_n_seen"}, "every calibrator uses log_n_seen", str(coords))
    streams = [s.get("experiment") for s in manifest.get("streams", [])]
    expected_streams = ["RT-100R", "RT-120R", "RT-121R", "RT-122R",
                        "RT-123R", "RT-124R", "RT-125R"]
    check(streams == expected_streams, "seven specialist streams", ", ".join(streams))
    check(manifest.get("code_git_sha") == MODEL_CODE_SHA, "manifest code_git_sha")

    text = open(py_path, encoding="utf-8").read()
    m = re.search(r"INFER_PARALLELISM\s*=\s*(\d+)", text)
    check(bool(m) and m.group(1) == "1", "INFER_PARALLELISM == 1",
          m.group(1) if m else "not found")

    # embedded src/sbr must equal the tree the model was trained from
    tree = git("ls-tree", "-r", "--name-only", MODEL_CODE_SHA, "--", "src/sbr").split()
    if tree:
        want = {}
        for f in tree:
            blob = subprocess.run(["git", "-C", REPO, "show", f"{MODEL_CODE_SHA}:{f}"],
                                  capture_output=True).stdout
            want[f[len("src/"):]] = hashlib.sha256(blob).hexdigest()
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(
                re.search(r'_SRC_B64 = "([A-Za-z0-9+/=]+)"', text).group(1)))) as z:
            got = {n: hashlib.sha256(z.read(n)).hexdigest()
                   for n in z.namelist() if not n.endswith("/")}
        diff = sorted(set(want) ^ set(got)) + sorted(
            k for k in set(want) & set(got) if want[k] != got[k])
        check(not diff, "embedded src/sbr identical to training SHA tree",
              f"{len(got)} files" if not diff else f"{len(diff)} differ: {diff[:3]}")

    # ---- 5. competition ----------------------------------------------------
    section("5. crunch competition")
    # replicate the CLI's upward walk; the NEAREST config is the one it uses
    found, d = None, REPO
    while True:
        cand = os.path.join(d, ".crunchdao", "project.json")
        if os.path.exists(cand):
            found = cand
            break
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    if check(found is not None, "a .crunchdao/project.json is reachable",
             found or "none found — run `crunch setup` in this checkout"):
        cfg = json.load(open(found))
        name = cfg.get("competitionName")
        check(name == COMPETITION, f"competitionName == {COMPETITION}",
              f"{name}  (from {found})")
        check(os.path.dirname(os.path.dirname(found)) == REPO,
              "nearest config belongs to THIS checkout",
              "otherwise the CLI walks up to a different competition")
    ignored = subprocess.run(["git", "-C", REPO, "check-ignore", "-q", ".crunchdao"])
    check(ignored.returncode == 0, ".crunchdao/ is gitignored (token never committed)")

    # ---- 6. release-critical tests -----------------------------------------
    section("6. release-critical tests")
    if a.skip_tests:
        check(False, "release-critical tests ran", "--skip-tests is NOT a release pass")
    else:
        env = dict(os.environ, SBR_MODEL_DIR=model_dir)
        r = subprocess.run(
            [sys.executable, "-m", "pytest",
             "tests/test_no_n_online_leakage.py",
             "tests/test_calibration_time_coord.py",
             "tests/test_production_contract.py", "-q", "-rs"],
            cwd=REPO, env=env, capture_output=True, text=True)
        tail = (r.stdout or "").strip().splitlines()
        summary = tail[-1] if tail else "(no output)"
        check(r.returncode == 0, "pytest exit code 0", summary)
        # a skipped causality gate is the failure mode this whole script exists for
        skipped = re.search(r"(\d+) skipped", r.stdout or "")
        check(skipped is None, "nothing skipped",
              f"{skipped.group(1)} skipped — a skipped gate is NOT a pass"
              if skipped else "0 skipped")
        check("passed" in summary and r.returncode == 0,
              "no-n_online causality gate executed",
              "verified by pytest exit status and zero skips")

    return finish()


def finish() -> int:
    print(f"\n{'=' * 68}")
    if _failures:
        print(f"RELEASE VALIDATION FAILED — {len(_failures)}/{_checks} gate(s) failed:")
        for f in _failures:
            print(f"  - {f}")
        print("\nDO NOT SUBMIT.  Report the blocker; do not work around it.")
        return 1
    print(f"RELEASE VALIDATION PASSED — {_checks}/{_checks} gates.")
    print("The artifact on disk is the frozen RT-600, in the right checkout,")
    print("pointed at the 2026 real-time competition.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
