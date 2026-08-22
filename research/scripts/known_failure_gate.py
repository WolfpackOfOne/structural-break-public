"""KNOWN-FAILURE FINGERPRINT AND GATE.

This suite is not green and must not be made green by changing what the shipped
engine emits.  Fifteen tests fail, they have failed since before wave 5, and
they reproduce name for name on the pre-wave-5 RT-600 submission baseline.  The
honest handling is to pin the exact set and fail loudly on ANY drift -- a new
failure, a disappearance, or a rename.

    python research/scripts/known_failure_gate.py --generate   # rebuild the file
    python research/scripts/known_failure_gate.py              # gate (CI use)

A DISAPPEARANCE IS AN ALERT, NOT A CELEBRATION.  Every one of the pinned
failures is a batch-versus-streaming float difference in a production module.
If one silently starts passing, the most likely cause is that somebody changed
what the streaming engine emits -- which changes RT-600's live predictions --
and the second most likely is that a test stopped asserting anything.  Both need
a human.

The gate REQUIRES the artifact environment (SBR_STORE / SBR_FEATURES /
SBR_MODEL_DIR).  Without it, dozens of tests fail or skip for reasons that have
nothing to do with the code, and the fingerprint would be meaningless.
"""
from __future__ import annotations

import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
FINGERPRINT = f"{ROOT}/research/known_failures.json"

BASELINE_SHA = "9aaa9b0"
BASELINE_BRANCH = "claude/rt600-baseline-submission"
BASELINE_NOTE = ("the RT-600 submission baseline, PRE-WAVE-5. Running the same "
                 "four parity files there reproduces the identical 15 node ids.")

#: why each family fails, and why it is not repaired
CLASSES = {
    "m07_bayes_stream_parity": {
        "modules": ["m07_bayes"],
        "reason": ("batch-vs-streaming float difference in the Bayesian online "
                   "changepoint block: bo_p_lt25_z, bo_lo_change_z, bo_ent. The "
                   "engine-level test reports 2 differing cells out of 185,000 "
                   "on series 5111."),
        "why_not_fixed": ("repairing the twin would change the values the "
                          "STREAMING engine emits, i.e. frozen production "
                          "semantics and therefore RT-600's live predictions. "
                          "RT-600 scored 0.6268 on the leaderboard through this "
                          "exact streaming path."),
    },
    "m06_loc_stream_parity": {
        "modules": ["m06_loc"],
        "reason": "same class: batch-vs-streaming float difference in the localisation block.",
        "why_not_fixed": "same: it would change frozen production streaming semantics.",
    },
    "m01_seq_stream_parity": {
        "modules": ["m01_seq"],
        "reason": "same class, one synthetic case.",
        "why_not_fixed": "same: it would change frozen production streaming semantics.",
    },
    "engine_parity_real": {
        "modules": ["m07_bayes"],
        "reason": ("the whole-engine parity test, which fails BECAUSE of the "
                   "m07_bayes cells above -- it is the same defect observed "
                   "through the assembled engine, not an independent one."),
        "why_not_fixed": "same root cause; same reason.",
    },
}


def classify(node_id: str) -> str:
    if "test_stream_parity_m07_bayes" in node_id:
        return "m07_bayes_stream_parity"
    if "test_stream_parity_m06_loc" in node_id:
        return "m06_loc_stream_parity"
    if "test_stream_parity_m01_seq" in node_id:
        return "m01_seq_stream_parity"
    if "test_stream_engine_parity" in node_id:
        return "engine_parity_real"
    return "UNCLASSIFIED"


def required_env() -> dict:
    missing = [k for k in ("SBR_STORE", "SBR_FEATURES", "SBR_MODEL_DIR")
               if not os.environ.get(k)]
    if missing:
        raise SystemExit(
            f"the gate needs {missing} set; without the artifacts the failure set "
            "is dominated by missing-file errors and the fingerprint is noise")
    return {k: os.environ[k] for k in ("SBR_STORE", "SBR_FEATURES", "SBR_MODEL_DIR")}


def run_suite() -> tuple[list[str], dict]:
    env = dict(os.environ)
    env.setdefault("SBR_ROOT", ROOT)
    p = subprocess.run(
        [sys.executable, "-m", "pytest", f"{ROOT}/tests", "-q", "--tb=no", "-rf",
         "-p", "no:randomly"],
        capture_output=True, text=True, env=env, cwd=ROOT)
    out = p.stdout + p.stderr
    failed = sorted({m.group(1).strip() for m in
                     re.finditer(r"^FAILED (\S+)", out, re.M)})
    tail = re.search(r"(\d+) failed, (\d+) passed(?:, (\d+) skipped)?", out)
    if tail is None:
        tail = re.search(r"(\d+) passed(?:, (\d+) skipped)?", out)
        counts = {"failed": 0, "passed": int(tail.group(1)) if tail else -1,
                  "skipped": int(tail.group(2) or 0) if tail else -1}
    else:
        counts = {"failed": int(tail.group(1)), "passed": int(tail.group(2)),
                  "skipped": int(tail.group(3) or 0)}
    return failed, counts


def generate():
    envs = required_env()
    failed, counts = run_suite()
    unclassified = [n for n in failed if classify(n) == "UNCLASSIFIED"]
    doc = {
        "schema": "sbr.known_failures/1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generated_on_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True).stdout.strip(),
        "baseline_sha": BASELINE_SHA, "baseline_branch": BASELINE_BRANCH,
        "baseline_note": BASELINE_NOTE,
        "required_env": {k: "(path)" for k in envs},
        "counts": counts,
        "classes": CLASSES,
        "known_failures": [{"node_id": n, "class": classify(n)} for n in failed],
        "policy": (
            "DO NOT modify production feature or streaming semantics to make "
            "these green. Any NEW failure, any DISAPPEARANCE, and any RENAME is "
            "a gate failure requiring a human decision."),
    }
    if unclassified:
        doc["WARNING_unclassified"] = unclassified
    json.dump(doc, open(FINGERPRINT, "w"), indent=1)
    print(f"wrote {FINGERPRINT}: {len(failed)} known failures, counts {counts}")
    for n in failed:
        print(f"  [{classify(n)}] {n}")
    return 1 if unclassified else 0


def gate():
    if not os.path.exists(FINGERPRINT):
        raise SystemExit(f"no fingerprint at {FINGERPRINT}; run --generate")
    doc = json.load(open(FINGERPRINT))
    required_env()
    known = {e["node_id"] for e in doc["known_failures"]}
    failed, counts = run_suite()
    now = set(failed)
    new = sorted(now - known)
    gone = sorted(known - now)
    print(f"known {len(known)}   observed {len(now)}   counts {counts}")
    rc = 0
    if new:
        rc = 1
        print("\nNEW FAILURES -- these are regressions and must be fixed:")
        for n in new:
            print(f"  + {n}")
    if gone:
        rc = 1
        print("\nKNOWN FAILURES THAT DISAPPEARED -- this is an ALERT, not good news.")
        print("  A pinned failure is a batch-vs-stream difference in a production")
        print("  module. If it now passes, either the STREAMING ENGINE'S OUTPUT")
        print("  CHANGED -- which changes RT-600's live predictions -- or a test")
        print("  stopped asserting. Both need a human.")
        for n in gone:
            print(f"  - {n}")
    if rc == 0:
        print("\nOK: the failure set is exactly the pinned fingerprint.")
    return rc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    a = ap.parse_args()
    raise SystemExit(generate() if a.generate else gate())
