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
a human.  That rule was applied, not waived: see below.

2026-08-26 -- THE FIFTEEN WERE RESOLVED AND THE SET IS NOW EMPTY.
--------------------------------------------------------------
The alert fired.  All fifteen disappeared at once, a human investigated, and the
cause was the third possibility the paragraph above does not list: the defect
they pinned was found and repaired.  The rationale recorded against them was
that repairing the streaming twin would change frozen production semantics.  It
would not, because the streaming twin was never the defect -- three primitives
underneath it were, and the batch path is the reference by construction because
it wrote the feature cache RT-600 was trained on:

  * ``math.lgamma`` disagrees between numba and CPython by up to 512 ULP, and
    the BOCPD Student-t constant table was built independently on each side.
    Now single-sourced from ``m07_bayes._bocpd_ct``   (b41da11)
  * ``scipy.signal.lfilter`` contracts ``b0*x + z`` into an arm64 FMA, rounding
    once where Python rounds twice.  Now matched by ``sbr.stream._fp.fma``
  * a Python scalar ``x ** 2`` goes through libm ``pow`` where numpy squares by
    multiplication.  Now matched                                     (03e4637)

No test was removed, xfailed or loosened, no tolerance was relaxed and no
expected output was edited to obtain green: the whole repair is in the shipped
engine, and the batch path is bit-identical before and after, so no refit is
implied.  Evidence is in ``engineering/reports/rt600_final_reliability/``
(STREAM_PARITY_REPRO.md for the root cause, BUGFIX_IMPACT.json for the measured
prediction impact) and the regeneration was authorised by the owner as an
engineering-only release action.

The previous fingerprint is preserved verbatim under ``ARCHIVE_DIR`` and is also
summarised inside the new one under ``previous_fingerprint``, so the historical
fact that these fifteen failed -- and why they were left failing -- is not lost.

The gate below is UNCHANGED, and against an empty pinned set it is strictly
stronger than it was: every failure is now a new failure.

The gate REQUIRES the artifact environment (SBR_STORE / SBR_FEATURES /
SBR_MODEL_DIR).  Without it, dozens of tests fail or skip for reasons that have
nothing to do with the code, and the fingerprint would be meaningless.
"""
from __future__ import annotations

import argparse, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
FINGERPRINT = f"{ROOT}/research/known_failures.json"

#: --generate copies the fingerprint it is about to replace in here, verbatim and
#: under a content-addressed name, before writing the new one.  A regeneration
#: must never be the only record that the previous failure set existed.
ARCHIVE_DIR = f"{ROOT}/research/archive/known_failures"

BASELINE_SHA = "9aaa9b0"
BASELINE_BRANCH = "claude/rt600-baseline-submission"
BASELINE_NOTE = ("the RT-600 submission baseline, PRE-WAVE-5. Running the same "
                 "four parity files there reproduces the identical 15 node ids.")

#: How each family was resolved, keyed by the class name it was pinned under.
#: Kept after the set went empty because deleting it would delete the record of
#: what the fifteen were.
RESOLVED = {
    "m07_bayes_stream_parity": {
        "resolved_by": "b41da11",
        "resolved_how": (
            "the BOCPD Student-t log-normalising constant table is now built "
            "once, by m07_bayes._bocpd_ct, and imported by _BocpdStream instead "
            "of recomputed. numba's math.lgamma and CPython's disagree by up to "
            "512 ULP on this grid, so the two sides had been sitting on "
            "different constants since the streaming port was written."),
    },
    "m06_loc_stream_parity": {
        "resolved_by": "03e4637",
        "resolved_how": (
            "scalar-square parity: a Python `x ** 2` goes through libm pow, "
            "numpy's array `** 2` is a squaring multiply, and they differ on "
            "688 of 500,000 doubles. m06's rolling variance is a cumsum "
            "difference that cancels to ~1e-14, so one ULP moved the emitted "
            "float32 by 15%."),
    },
    "m01_seq_stream_parity": {
        "resolved_by": "03e4637",
        "resolved_how": (
            "arm64 fused-multiply-add parity: scipy.signal.lfilter contracts "
            "b0*x + z into a single-rounding FMA, which sbr.stream._fp.fma now "
            "reproduces exactly (Dekker two-product/two-sum, since math.fma is "
            "3.13+ and the frozen environment is 3.11.6)."),
    },
    "engine_parity_real": {
        "resolved_by": "b41da11 and 03e4637",
        "resolved_how": (
            "no independent defect: this is the assembled-engine view of the "
            "three above, and went green when they did."),
    },
}

#: why each family failed, and the rationale under which it was left unrepaired.
#: PRESERVED VERBATIM. The `why_not_fixed` text is the position that was
#: overturned on 2026-08-26 -- see RESOLVED above and the module docstring --
#: and it is kept unedited so the overturned reasoning stays readable.
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


#: torch and LightGBM each ship their own libomp.dylib, and loading both into
#: one process SEGFAULTS on macOS/arm64 -- reproduced: the suite dies inside
#: lightgbm/basic.py once tests/test_neural_causality.py has imported torch.
#: The suite is therefore run in TWO processes and the results unioned. This is
#: an environment hazard, not a test failure, and papering over it with
#: KMP_DUPLICATE_LIB_OK would trade a loud crash for silent corruption.
#: EVERY test file that imports torch, directly or through a helper. This list
#: was "tests/test_neural_causality.py" alone until the CRF program added
#: test_crf01_causality.py and test_crf02_causality.py, which import torch via
#: wave6_neural_lib / crf01_nncsr / crf02_acgn. Leaving them in process 1 put
#: torch and LightGBM back in one process and the gate SEGFAULTED inside
#: lightgbm/basic.py rather than reporting -- i.e. the split silently stopped
#: doing its job. Anything added here must import torch; anything that imports
#: torch must be added here.
TORCH_TESTS = (
    "tests/test_neural_causality.py",
    "tests/test_crf01_causality.py",
    "tests/test_crf02_causality.py",
)


def _pytest(args) -> tuple[set, dict, int]:
    env = dict(os.environ)
    env.setdefault("SBR_ROOT", ROOT)
    p = subprocess.run([sys.executable, "-m", "pytest", *args, "-q", "--tb=no",
                        "-rf", "-p", "no:randomly"],
                       capture_output=True, text=True, env=env, cwd=ROOT)
    out = p.stdout + p.stderr
    if "Fatal Python error" in out or "Segmentation fault" in out:
        raise SystemExit(
            "pytest CRASHED rather than failed. If this is the torch/LightGBM "
            "libomp collision, the two-process split below has been broken.\n"
            + out[-2000:])
    failed = {m.group(1).strip() for m in re.finditer(r"^FAILED (\S+)", out, re.M)}
    tail = re.search(r"(\d+) failed, (\d+) passed(?:, (\d+) skipped)?", out)
    if tail is None:
        tail = re.search(r"(\d+) passed(?:, (\d+) skipped)?", out)
        counts = {"failed": 0, "passed": int(tail.group(1)) if tail else -1,
                  "skipped": int(tail.group(2) or 0) if tail else -1}
    else:
        counts = {"failed": int(tail.group(1)), "passed": int(tail.group(2)),
                  "skipped": int(tail.group(3) or 0)}
    if counts["passed"] < 0:
        raise SystemExit(f"could not parse a pytest summary from:\n{out[-2000:]}")
    return failed, counts, p.returncode


def run_suite() -> tuple[list[str], dict]:
    """Two processes, unioned. See TORCH_TESTS above for why."""
    ignores = [f"--ignore={ROOT}/{t}" for t in TORCH_TESTS]
    a_fail, a_cnt, _ = _pytest([f"{ROOT}/tests", *ignores])
    b_fail, b_cnt, _ = _pytest([f"{ROOT}/{t}" for t in TORCH_TESTS])
    counts = {k: a_cnt[k] + b_cnt[k] for k in ("failed", "passed", "skipped")}
    counts["processes"] = {"non_torch": a_cnt, "torch_only": b_cnt}
    return sorted(a_fail | b_fail), counts


def archive_previous() -> dict:
    """Copy the fingerprint about to be replaced into ARCHIVE_DIR, verbatim.

    Returns a summary of what was archived, for embedding in the new document,
    or ``None`` when there is no previous fingerprint to preserve.
    """
    if not os.path.exists(FINGERPRINT):
        return None
    raw = open(FINGERPRINT, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    old = json.loads(raw)
    stamp = (old.get("generated_at") or "unknown").split(" ")[0]
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    name = f"known_failures_{stamp}_{sha[:12]}.json"
    with open(os.path.join(ARCHIVE_DIR, name), "wb") as fh:
        fh.write(raw)
    return {
        "archived_to": f"research/archive/known_failures/{name}",
        "sha256": sha,
        "generated_at": old.get("generated_at"),
        "generated_on_sha": old.get("generated_on_sha"),
        "counts": old.get("counts"),
        "n_known_failures": len(old.get("known_failures") or []),
        "known_failures": old.get("known_failures"),
        "policy": old.get("policy"),
    }


def generate(reason=None, authorised_by=None):
    envs = required_env()
    previous = archive_previous()
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
        "execution": {
            "two_processes": True,
            "reason": ("torch and LightGBM each ship libomp.dylib and loading "
                       "both into one process segfaults on macOS/arm64 -- "
                       "reproduced inside lightgbm/basic.py. Process 1 runs the "
                       "suite excluding " + ", ".join(TORCH_TESTS) +
                       "; process 2 runs only those files. Results are unioned."),
            "do_not": ("do NOT set KMP_DUPLICATE_LIB_OK to run them together -- "
                       "that trades a loud crash for silent numerical corruption"),
        },
        "counts": counts,
        "classes": CLASSES,
        "resolved": RESOLVED,
        "previous_fingerprint": previous,
        "regeneration": {
            "reason": reason,
            "authorised_by": authorised_by,
            "previous_sha256": (previous or {}).get("sha256"),
            "previous_n_known_failures": (previous or {}).get("n_known_failures"),
            "new_n_known_failures": len(failed),
        } if (reason or authorised_by) else None,
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
    if previous:
        print(f"  previous fingerprint sha256 {previous['sha256']}")
        print(f"  archived verbatim to {previous['archived_to']} "
              f"({previous['n_known_failures']} pinned failures)")
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
    ap.add_argument("--reason", default=None,
                    help="why the fingerprint is being regenerated; recorded in it")
    ap.add_argument("--authorised-by", default=None,
                    help="who authorised the regeneration; recorded in it")
    a = ap.parse_args()
    raise SystemExit(generate(a.reason, a.authorised_by) if a.generate else gate())
