# WAVE-3 INTEGRATION AUDIT

**What was integrated, what conflicted, how each conflict was resolved, and
whether any research semantics changed.**

Date 2026-08-20. Auditor: the wave-4 session.

---

## 1. THE THREE COMMITS

| role | ref | SHA | title |
|---|---|---|---|
| common base | — | `8e76ad205c68a67b219ff6b781dd9d696533d984` | Wave 2: reproducibility gate, bitwise streaming engine, deployable ensemble |
| research parent | `research/wave2-2026` | `bfcb232de79555c59760c31b4fb11c8cd763a6ec` | Wave 3: backward suffix-vs-prefix contrast, rejected by its own controls |
| engineering parent | `codex/wave3-engineering` | `24675a63887b82512e1ffa60404ab72d738f12bc` | Add RT-150 crunch test artifacts |
| **integration** | `research/wave3-integration` | **`c4fb01e`** | Integrate codex/wave3-engineering into the wave-3 research branch |

`git merge-base bfcb232 24675a6` = `8e76ad2`, confirmed. They are **siblings**,
each a single commit off the shared base. Neither contains the other.

Both source branches are untouched: `research/wave2-2026` still resolves to
`bfcb232` and `codex/wave3-engineering` still resolves to `24675a6`. No history
was rewritten and no branch was deleted.

Method: `git checkout -b research/wave3-integration bfcb232`, then
`git cherry-pick -n 24675a6`. Starting from the research parent is deliberate —
it owns `RESULTS.csv`, `RDOF_LEDGER.md`, `FAILED_EXPERIMENTS.md` and
`STATE_OF_RESEARCH_V3.md`, the files where a bad merge would corrupt the record
rather than merely break a script.

## 2. FILES TOUCHED BY BOTH PARENTS

Only two.

### `src/sbr/pipeline.py` — **no conflict, identical change**

Both parents replaced `ROOT = "/home/claude/sb"` with
`ROOT = os.environ.get("SBR_ROOT", "/home/claude/sb")`, character for character.
Git merged it silently. Verified after the fact by diffing the integrated file
against both parents' versions: identical to both.

### `research/scripts/wave2_lib.py` — **1 file, 2 conflicting hunks**

**Hunk A — the `ROOT` definition.**

| side | value |
|---|---|
| research (`bfcb232`) | `os.environ.get("SBR_ROOT", "/home/claude/sb")` |
| engineering (`24675a6`) | `os.environ.get("SBR_ROOT", os.path.abspath(.../"../.."))` |

**Resolved to the engineering side.** Both honour `SBR_ROOT`, but the research
side's *fallback* is the path of a container that no longer exists, so an
unset-variable invocation fails with a confusing missing-file error instead of
working. The engineering fallback resolves to the checkout the file lives in,
which is correct in every worktree. This is the file-ownership rule from the
brief working as intended: shared infrastructure, merged on merit.

**Hunk B — `cols_of()`.**

| side | value |
|---|---|
| research (`bfcb232`) | reads `os.environ.get("SBR_FEATURES", f"{ROOT}/cache/features")` |
| engineering (`24675a6`) | reads `f"{ROOT}/cache/features"` |

**Resolved to the research side.** The `SBR_FEATURES` override is what lets a
worktree point at a *shared* 10 GB feature cache instead of duplicating it. The
engineering version silently drops that capability. Taking "theirs" wholesale
here would have cost 10 GB of disk on a machine with 66 GB free.

Neither hunk was resolved by `-X ours` or `-X theirs`. The integrated file is
the union of the two better halves, and carries a comment saying so.

## 3. DID ANY RESEARCH SEMANTICS CHANGE?

**No.** Every hunk the engineering parent contributed was inspected line by line:

| file | change | semantics? |
|---|---|---|
| `research/scripts/build_store.py` | `RAW`/`OUT` from `SBR_RAW_DIR`/`SBR_ROOT` instead of dead absolute paths | path only |
| `research/scripts/build_submission.py` | path/portability, deterministic `.py` emission | build only |
| `research/scripts/test_submission_notebook.py` | path/portability | harness only |
| `research/scripts/wave2_train_ensemble.py` | `sys.path`, `--out` default, OOF path, `git -C` target | path only — **the `STREAMS` dict, the calibration fit and the manifest fields are unchanged** |
| `research/PUBLIC_IDEA_MAP.md` | +27 lines, 2025 repo translation pointers | additive documentation |
| `research/reports/codex_2025_translation.md` | new | new documentation |
| `research/reports/crunch_test_rt150.md` | new | new evidence |
| `research/reports/submission_notebook_test.json` | re-measured on macOS | new evidence, supersedes a container measurement |
| `submissions/C_ensemble_deployable.{ipynb,py}` | new, 28 MB each | the Crunch-tested artifact |

The engineering parent did not touch `RESULTS.csv`, `RDOF_LEDGER.md`,
`FAILED_EXPERIMENTS.md`, `VALIDATION_V2.md`, any feature module, any fold
definition or any promotion decision. There was no semantic collision to
adjudicate; the ownership hierarchy in the brief was never actually tested,
because the two lanes stayed in their lanes.

**Verification.** `git diff 24675a6 HEAD --numstat` shows deletions in exactly
three files — `research/RESULTS.csv`, `research/scripts/wave2_lib.py` and
`src/sbr/features/driver.py` — all of them files the research parent
legitimately modified. No engineering content was dropped.

## 4. ONE PORTABILITY FIX ADDED ON TOP (`c6da270`)

`research/scripts/wave2_streams.py` still carried two dead
`sys.path.insert(0, "/home/claude/sb/...")` lines. Harmless on import but
misleading, and the file had to be importable here because wave 4 re-runs the
specialist configurations it defines. Replaced with the same repo-relative
pattern. The `JOBS` dict — the actual stream definitions — is byte-identical.

## 5. WHAT THE INTEGRATION REVEALED

Three things that neither branch's documentation stated:

1. **The Crunch-tested artifact's provenance chain is broken.**
   `manifest.code_git_sha = b5ea9d1d9cd87f06f574c5a78e4c850f41ef852f` does not
   resolve in this repository (`git log -1 b5ea9d1…` → `fatal: bad object`) and
   is not on `origin`. The report that records the passing test is now in the
   integrated tree; the commit it names is not. This has to be fixed by
   rebuilding from a reachable commit, not by editing the manifest.

2. **The fold partition is verifiably the tested one.** The canonical `id,fold`
   CSV hash of `research/folds/folds.parquet` in this worktree is
   `6e114f80240c0ff069266c6d65b660e500723ce9bf2ca7633596d9e392d6b9e9`, exactly
   the value in `crunch_test_rt150.md`. Anything measured here is measured
   against the same split the artifact was built on.

3. **The champion's OOF vectors are gone.** `wave2_train_ensemble.py` builds each
   stream's calibration from `research/oof/{exp}.npy`. Those seven files exist
   nowhere on this machine or in any commit. The 0.62589 could not be recomputed
   from the repository before wave 4 retrained the streams. The number is not
   wrong — `deployable_ensemble_v2.json` records it with per-fold detail — but it
   was, until now, unreproducible.
