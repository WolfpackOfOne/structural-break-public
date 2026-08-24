# Repository Cleanup Report — 2026-08-24

What was actually done, following the audit in
`docs/repository_cleanup_audit.md`. Every action below preserves history:
no `git filter-repo`/BFG, no force-push, no rewritten commits, no deleted
report or negative result.

## A. Start state

- Remote branches: 20
- Open PRs: 1 (#11)
- Worktrees: 9, all clean, all synced
- Tags: 1 (`lb-003-running`)
- `.git` size: 253.77 MiB (8 packs, ~1.88 MiB reclaimable local garbage)
- Two branches had genuine unpushed local commits (`research/multi-agent-2026`,
  `research/wave6-alpha`); one branch had diverged between two sessions
  (`research/wave5-alpha`); a substantial amount of research material sat
  untracked in a working tree with no copy anywhere in git history.

## B. Scientific state preserved

**15 annotated tags created and pushed** (full messages carry experiment
ID / conclusion / score / source path / date):

| Tag | Points at |
| --- | --- |
| `rt600-production-0.6268` | `claude/rt600-baseline-submission` tip — frozen RT-600, external score 0.6268 |
| `wave2-2026-final` | `research/wave2-2026` tip (branch since deleted, tag preserves it) |
| `wave3-integration-final` | `research/wave3-integration` tip (branch since deleted, tag preserves it) |
| `wave5-alpha-final` | The Wave-5 lineage that actually fed Wave 6/7/8 |
| `wave5-alpha-c1c2c3-superseded` | The diverged, earlier Wave-5 C1/C2/C3 attempt found on `origin/research/wave5-alpha` |
| `wave7-d3r-information-frontier` | W7-D3R diagnostic, the future-information-limit result |
| `wave7-t2-final` | `research/wave7-teacher-distillation` tip |
| `wave7-t2-promotion-mostly-redundant` | `research/wave7-t2-promotion` tip — final T2/RT-995 conclusion |
| `wave8-future-aware-final` | `research/wave8-future-aware-distillation` tip |
| `multi-agent-2026-final` | `research/multi-agent-2026` tip (pushed, see D) |
| `wave7-early-attempt-superseded` | `claude/project-setup-github-20g6pt` tip — unique, unreachable-elsewhere commits |
| `oracle-information-frontier-2026-study` | `codex/oracle-information-frontier-2026` tip |
| `reproduce-2025-public-solution-audit` | `codex/reproduce-2025-public-solution` tip |
| `wave3-engineering-rt150` | `codex/wave3-engineering` tip |

**Frozen production commit:** `9aaa9b0` on `claude/rt600-baseline-submission`
(also now `production/rt600`), external Crunch leaderboard score **0.6268**.

**Canonical branches retained/created:**

- `main` — clean, stable baseline package
- `production/rt600` (new) — frozen RT-600 deployment lineage
- `research/current` (new) — canonical active research (from `research/wave7-t2-promotion`)

## C. Branches retained (reason)

| Branch | Reason |
| --- | --- |
| `main` | The project's stable base |
| `production/rt600`, `research/current` | New permanent branches per the target model |
| `claude/rt600-baseline-submission` | Pinned by an active worktree; source of `production/rt600` |
| `research/wave5-alpha`, `research/wave7-teacher-distillation`, `research/wave7-t2-promotion`, `research/wave8-future-aware-distillation` | Each pinned by an active worktree, and/or explicitly load-bearing recent history |
| `research/multi-agent-2026` | Unpushed unique commits (now pushed) representing a genuinely separate, non-superseded research track with its own experiment-ID space |
| `research/wave6-alpha` | Unpushed unique commits (now pushed); content also reachable via `wave7-teacher-distillation` |
| `codex/wave3-engineering`, `codex/oracle-information-frontier-2026`, `codex/reproduce-2025-public-solution` | Each pinned by an active worktree |
| `claude/project-setup-github-20g6pt` | 4 unique commits not reachable from any other ref (an early, apparently-abandoned Wave-7 opening attempt); whether it's truly abandoned is a judgment call left to the research owner |
| `claude/structural-break-competition-entry-yjoj8r` | The current primary worktree's branch; source of closed PR #11, left untouched |
| `origin/research/wave5-alpha` (the branch ref itself, distinct from the `research/wave5-alpha` worktree) | **Diverged, not resolved** — see below |

**Unresolved item flagged for the repository owner:** `origin/research/wave5-alpha`
(tip `74f75d9`) and the local `research/wave5-alpha` worktree (tip `26b01f6`)
diverged from a common ancestor after two sessions apparently both started
a "Wave 5" from the same point. Neither side was force-pushed, merged, or
rebased. Both tips are preserved by tags (`wave5-alpha-final` for the
lineage that fed Wave 6+, `wave5-alpha-c1c2c3-superseded` for the other).
**This still needs a human decision** — e.g., rename the origin branch to
something like `research/wave5-alpha-c1c2c3` so the name stops colliding,
or intentionally reconcile the two. See
`docs/repository_cleanup_audit.md` section C for the full trace.

## D. Branches deleted

All seven verified safe by SHA ancestry before deletion (see audit section C):

| Branch | Why safe | Where it's reachable now |
| --- | --- | --- |
| `ci/add-tests` | Squash-merged, PR #8 | `main` (`deb8211`) |
| `cleanup/repo-hygiene` | Squash-merged, PR #6 | `main` (`a1510c0`) |
| `refactor/package-baseline` | Squash-merged, PR #7 | `main` (`6953e47`) |
| `research/add-change-point-models` | Squash-merged, PR #9 | `main` (`191bb13`) |
| `docs/portfolio-polish` | Squash-merged, PR #10 | `main` (`cef5458`) |
| `research/wave2-2026` | Literal git ancestor of the canonical chain, no worktree, no PR | `research/current`, tag `wave2-2026-final` |
| `research/wave3-integration` | Literal git ancestor of the canonical chain, no worktree, no PR | `research/current`, tag `wave3-integration-final` |

Before any deletion, `research/multi-agent-2026` (2 commits) and
`research/wave6-alpha` (3 commits) were pushed from local to origin —
neither had been at risk of loss (both were in a clean, synced local
worktree/branch, not a scratch environment), but per the "push before
cleanup" rule this closed the gap between local and origin.

## E. PRs closed

| PR | Reason |
| --- | --- |
| #11 "Add Real-Time Edition submission: streaming detector, tests, and notebook" | Superseded — this is the RT-000-equivalent baseline (EWMA/CUSUM/variance-ratio noisy-OR), long since surpassed by RT-600 (external score 0.6268) and the Wave 5–8 research ladder. Closed with an explanatory comment; not merged; branch and full diff preserved in git history. |

## F. Research directory changes (on `research/current`)

- Created `research/reports/{wave4,wave5,wave6,wave7}/` and moved the
  corresponding `WAVE*_PREREG.md` / `WAVE*_STATUS.md` files there via
  `git mv` (history preserved as renames).
- Created `research/archive/{briefs,handoffs,dated_updates}/` and moved
  `BRIEF_CLAUDE_wave3_alpha.md`, `BRIEF_CODEX_wave3_engineering.md`,
  `HANDOFF_WAVE3.md`, `HANDOFF_WAVE6.md`, `20260818_claude_update_1.md`,
  `STATE_OF_RESEARCH_V2.md`–`V5.md`, `VALIDATION_V2.md`,
  `PUBLIC_IDEA_MAP.md` there.
- Canonical control docs stayed at `research/` top level: `README.md`,
  `STATE_OF_RESEARCH.md`, `PROTOCOL.md`, `RESULTS.csv`,
  `EXPERIMENT_ID_MAP.md`, `RDOF_LEDGER.md`, `FAILED_EXPERIMENTS.md`,
  `FINAL_ARCHITECTURE_FREEZE.md`, `FINAL_REPRODUCIBILITY_MANIFEST.json`,
  `REPRODUCIBILITY_MANIFEST.json` (both manifest files kept — different
  scope, verified by diff, not duplicates).
- Added `research/STATUS.md` — the concise current-state pointer.
- Updated `research/README.md` with a "read these first" list and a note
  that old reports are evidence, not live instructions.
- Rescued genuinely-orphaned files (never committed anywhere, verified via
  `git log --all --diff-filter=A`) into
  `research/archive/legacy_untracked_2026-08-19/`:
  `A_rt100_streaming.ipynb`, and a `wave2_draft/` subfolder for five files
  that share a basename with already-committed `codex/wave3-engineering`
  content but are not byte-identical to it.
- Wave 8's final report was **not** copied in — it remains only on
  `research/wave8-future-aware-distillation`, since merging it here would
  be exactly the "merge research branches blindly" this cleanup was told
  not to do.

## G. New agent governance

- **`AGENTS.md`** (repo root, on `main` via this PR) — branch/worktree
  safety rules, preregistration-before-score, experiment-ID allocation,
  causal rules (no true-`tau` feature, future information teacher-only,
  nested cross-fitting), what counts as a valid promotion, artifact policy,
  how to update `STATUS.md`/`RESULTS.csv`/`FAILED_EXPERIMENTS.md`, and
  "old reports are evidence, not live instructions."
- **`research/README.md`** — "if you are a new research agent, read these
  first" ordered list; updated layout table.
- **`research/STATUS.md`** — concise, deliberately short current-state doc.

## H. Generated-artifact policy

- Root `.gitignore` gained `.crunchdao/` and `*.bundle`/`*.tar.gz` —
  exactly the categories of file found sitting untracked as session-
  transfer clutter during the audit.
- Nothing already tracked in git was removed from version control.
- Untracked local material confirmed redundant during the audit (stale
  `.bundle` files, superseded `RESULTS.csv`/`STATE_OF_RESEARCH.md`
  snapshots, a duplicate `platform_constraints.md`) was **left in place on
  disk** in the `structural-break` main worktree rather than deleted —
  deleting a user's local untracked files is outside the scope of a git
  history cleanup; the audit documents which ones are safe to remove by
  hand.
- Model binaries (`wave2/models_*.tar.gz`) were left untracked and
  un-committed, per policy.

## I. CI / hygiene changes

- `research/scripts/check_research_hygiene.py` +
  `.github/workflows/research-hygiene.yml` on `research/current`: a
  minimal duplicate-experiment-ID + malformed-header check on
  `research/RESULTS.csv`, triggered only on pushes to `research/**`
  branches that touch that file. Tested locally (211 rows, 0 duplicates)
  before committing.
- `main`'s existing `ci.yml` (pytest + ruff on push/PR) was left
  unchanged — it already does the two things asked for, and the brief
  explicitly warned against building an elaborate CI system.

## J. Large-file findings

`git count-objects -vH` showed 253.77 MiB in `.git`, dominated by dozens of
near-duplicate ~28 MB blobs of `submissions/C_ensemble_deployable.ipynb`
and `.py`, re-committed repeatedly with embedded notebook outputs across
history (see `docs/repository_cleanup_audit.md` section E for the exact
`git rev-list --objects` output). **History was not rewritten.**
Recommendation for future sessions: strip notebook outputs before
committing (`nbstripout` or equivalent); consider Git LFS if large model
artifacts need to be versioned going forward.

## K. Remaining manual GitHub settings (not configured by this pass)

Recommended, not applied (requires repository admin settings, not
exposed through the tooling available here):

- Protect `main`: require a pull request before merging.
- Require the `ci.yml` check to pass before merging into `main`.
- Prevent force-pushes to `main`.
- Prevent deletion of `main`, `production/rt600`, `research/current`.
- Optionally require branches to be up to date with `main` before merge.

## L. End state

- Remote branches: **15** (was 20)
- Open PRs: **0** (was 1)
- Tags: **15** (was 1)
- Worktrees: **9**, all clean, all synced (unchanged from start — none
  were touched)
- Local tree: clean, all intended commits pushed
- **Verified before finishing:** `main` history intact; RT-600 production
  commit (`9aaa9b0`) reachable via `production/rt600` and
  `rt600-production-0.6268`; Wave 7 T2 final result (`5093e0a`) reachable
  via `research/wave7-teacher-distillation` and `wave7-t2-final`; T2
  promotion result (`d5d78ed`) reachable via `research/current`,
  `research/wave7-t2-promotion`, and
  `wave7-t2-promotion-mostly-redundant`; Wave 8 final result (`589e1db`)
  reachable via `research/wave8-future-aware-distillation` and
  `wave8-future-aware-final`; no duplicate experiment IDs in the current
  `RESULTS.csv` (211 rows checked); no worktree was modified or removed
  except a temporary one this session created and cleaned up itself.

## One item this pass could not safely resolve

The `origin/research/wave5-alpha` divergence (section C) needs a human
decision. Everything is preserved either way — it is a naming/reconciliation
question, not a data-loss risk.
