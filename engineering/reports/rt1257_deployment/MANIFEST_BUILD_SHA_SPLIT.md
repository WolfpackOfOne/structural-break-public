# RT-1257: the `manifest_matches_build_sha: false` split, verified

Date: 2026-09-01. Closes the second of the two open items `SUBMISSION_16.md` §4
requires before promotion — *"the unverified `manifest_matches_build_sha: false`
split"*. The first (the stale `CRUNCH_TEST.json`) was closed by
`CONSOLIDATION_HYGIENE_REVIEW.md` gate item 4.

`SUBMISSION_16.md` put the standard plainly: **"An anchor whose local test record
describes a different artifact is not an anchor."** This verifies which artifact
the two SHAs describe.

## The split

`submissions/RT1257_deployable.build.json` records two different commits:

| field | commit | date | subject |
|---|---|---|---|
| `manifest_code_git_sha` | `6b4fafac` | 2026-08-27 | Qualify RT-1257 deployment artifact |
| `code_git_sha` | `e50098a4` | 2026-08-29 | Declare lightgbm and catboost as inference dependencies |

`6b4fafac` is an **ancestor** of `e50098a4`, three commits behind. The manifest was
written two days before the build, on the same lineage — not a fork, and not an
artifact built from unknown code.

## What changed across the split

```
d4c5c64  Commit RT-1257 deployment qualification evidence
733c727  Unpack the submission payload to a temp dir, not the code tree
e50098a  Declare lightgbm and catboost as inference dependencies
```

Every file touched:

- `engineering/reports/rt1257_deployment/` — 8 evidence JSONs and a log
- `requirements.txt`
- `research/scripts/build_submission.py`
- `submissions/RT1257_deployable.build.json` — the build record itself

## The finding

**`src/` is untouched across the entire split.** `git diff 6b4fafac e50098a4 --
src/sbr` is empty, so no model code, no feature code, and no calibration code
differs between the commit the manifest records and the commit the artifact was
built at. The manifest and the build describe the same model.

The two non-evidence changes are both deployment-hygiene fixes, and neither
alters inference:

**`requirements.txt`** — declares the inference dependencies the artifact
already loaded:

```
lightgbm==4.7.0
catboost==1.2.10
```

The commit notes "neither was declared here before, and catboost is not known to
be present in the runner image", with versions matched to
`requirements-rt1257.txt`, the environment the artifact was built and
benchmarked against. This makes an existing implicit requirement explicit; it
does not change what runs.

**`build_submission.py`** (`733c727`) — fixes where the submission payload
unpacks:

```diff
- _WORK = os.path.abspath(f"./_sbr_payload_{os.getpid()}")
+ # Unpack into a guaranteed-writable temp directory, NEVER into the code tree.
```

This is a real cloud bug fix, and its failure mode is on record: an RT-1257
deployable built before it died at import on the Crunch runner with

```
File "/context/code/submissions/RT1257_deployable.py", line 21, in <module>
    os.makedirs(_WORK, exist_ok=True)
PermissionError: [Errno 13] Permission denied: '/context/code/_sbr_payload_110'
```

because `/context/code` is read-only. Residue from the pre-fix behaviour is still
visible as untracked `_sbr_payload_3826/` and `_sbr_payload_75521/` directories in
the `structural-break-claude-wave3` worktree.

So the split spans a fix that makes the artifact *more* deployable, not less.

## Verdict

The `manifest_matches_build_sha: false` flag is **benign and now verified**. It
records that the model manifest was written three commits before the build, on
the same lineage, across changes that touch evidence, dependency declaration and
the builder — and no model code. `SUBMISSION_16.md` §4 item 2 is satisfied.

This is a verification, not a promotion. Formal promotion still needs
`CONSOLIDATION_HYGIENE_REVIEW.md` gate item 1 — a clean rebuild from a tracked
checkout, with the prediction-equivalence battery run against it — and then item
7, which is an owner action.

## Scope

Verified from git history and the committed build record only. Nothing was
rebuilt, retrained or re-scored to produce this document.
