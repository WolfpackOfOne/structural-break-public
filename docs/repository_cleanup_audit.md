# Repository Cleanup Audit — 2026-08-24

Full audit performed before any destructive action. Ground truth from `git
fetch --all --prune`, `git branch -vv`, `git log --all --graph`, `gh pr list`,
`gh issue list`, and direct inspection of every worktree and untracked file.
This document is the record of what was found; see
`docs/REPOSITORY_CLEANUP_REPORT.md` for what was actually done.

## A. Start state

- **Remote branches:** 20 (incl. `main`)
- **Local worktrees:** 9 (`main` checkout + 8 auxiliary), all clean, all
  tracking a remote branch with no uncommitted changes
- **Open PRs:** 1 (#11)
- **Merged PRs:** 5 (#6–#10, all squashed onto `main`)
- **Closed issues:** 5 (#1–#5, auto-closed by the merged PRs)
- **Tags:** 1 (`lb-003-running`)
- **`.git` size:** 253.77 MiB, 8 packs, ~1.88 MiB reclaimable garbage
  (stale `tmp_pack_*`/`tmp_idx_*` fragments from an interrupted local
  operation — harmless, local-only, not pushed to the remote)
- **Untracked material in the main worktree:** ~10 top-level docs, a
  `research/` dir (3 files), `wave2/`, `submissions/`, six `.bundle` files,
  one `.tar.gz`

## B. Worktree inventory (none touched)

| Worktree dir | Branch | Status |
|---|---|---|
| `structural-break` | `claude/structural-break-competition-entry-yjoj8r` | clean, PR #11 source |
| `structural-break-claude-wave3` | `claude/rt600-baseline-submission` | clean, synced |
| `structural-break-codex-wave3` | `codex/wave3-engineering` | clean, synced |
| `structural-break-oracle` | `codex/oracle-information-frontier-2026` | clean, synced |
| `structural-break-reproduce-2025` | `codex/reproduce-2025-public-solution` | clean, synced |
| `structural-break-wave5` | `research/wave5-alpha` | clean, **diverged from origin (see D)** |
| `structural-break-wave6` | `research/wave7-teacher-distillation` | clean, synced |
| `structural-break-wave7-promotion` | `research/wave7-t2-promotion` | clean, synced (T2 result already on origin) |
| `structural-break-wave8` | `research/wave8-future-aware-distillation` | clean, synced |

All nine were left exactly as found. No worktree was removed.

## C. Branch-by-branch disposition

### Squash-merged into `main` — verified by SHA ancestry, safe to delete

Each PR's merge-commit SHA was confirmed with
`git merge-base --is-ancestor <sha> origin/main`:

| Branch | PR | Merge commit | Ancestor of main? |
|---|---|---|---|
| `cleanup/repo-hygiene` | #6 | `a1510c0` | yes |
| `refactor/package-baseline` | #7 | `6953e47` | yes |
| `ci/add-tests` | #8 | `deb8211` | yes |
| `research/add-change-point-models` | #9 | `191bb13` | yes |
| `docs/portfolio-polish` | #10 | `cef5458` | yes |

### Fully subsumed by the canonical research chain — tag, then delete

`research/wave2-2026` (`bfcb232`) is a literal git ancestor of
`research/wave3-integration` (`17bb5df`), which is a literal ancestor of
`research/wave5-alpha` → `wave6-alpha` → `wave7-teacher-distillation` →
`wave7-t2-promotion`. Verified with `git merge-base --is-ancestor`. No
worktree uses either ref, no PR references them, local == origin for both
(no divergence). Every commit remains reachable from `research/current`
after it is created, and each gets a milestone tag before deletion.

### Diverged — DO NOT TOUCH, flagged for human review

`origin/research/wave5-alpha` (tip `74f75d9`, "C3 stage 1: prove the
scorer...") and the local `research/wave5-alpha` worktree (tip `26b01f6`,
"W6-E0 de-risks the ladder...") **diverged** from a common ancestor
(`17bb5df`, the wave3-integration tip): origin has 8 commits not on local,
local has 44 commits not on origin. Reading the commit messages, the
origin-only side (`58cef85`...`74f75d9`, all dated 2026-08-22, "Wave 5
C1/C2/C3... open the alpha branch") looks like an earlier, independently
-restarted Wave-5 attempt that a different session pushed to the branch
name, while the local worktree continued a separate "V5 restart" that is
the one which actually fed into Wave 6/7/8 (its tip is a verified ancestor
of `wave7-t2-promotion`).

**Action taken:** neither side was pushed, merged, rebased, or force-updated.
Both tips were preserved with their own milestone tags (see D) under
distinct names so neither is lost, and `origin/research/wave5-alpha` the
branch ref was left completely untouched. **This needs a human decision**
(rename one lineage, or intentionally merge/reconcile) — it was not resolved
here because guessing wrong would either discard research or fabricate a
merge that never happened.

### Unpushed local research — pushed before any cleanup

- `research/multi-agent-2026`: local tip `23cb4da` was 2 commits ahead of
  `origin/research/multi-agent-2026` (`e98f5b9`) — "Streaming port:
  correctness foundation..." and "Platform constraints found; deployable
  configuration is 0.62368 at 64.9 ms/point." Simple fast-forward, no
  divergence. Pushed.
- `research/wave6-alpha`: local tip `a084956` was 3 commits ahead of
  `origin/research/wave6-alpha` (`33fc210`) — W7-D0 and W7-D3R diagnostics.
  These commits were already reachable via `origin/research/wave7-teacher-distillation`
  (pushed separately), so nothing was at risk, but the branch ref itself
  was stale. Simple fast-forward, no divergence. Pushed for ref-completeness.

### Unique, unreachable, no PR, no worktree — kept, tagged for discoverability

- `claude/project-setup-github-20g6pt`: 4 commits (`f4de370`...`0463c5b`)
  represent an early, independent attempt to open Wave 7 (merging the
  wave-6 line and re-aiming at the metric) that is **not** an ancestor of
  either `research/wave7-teacher-distillation` or `research/wave7-t2-promotion`
  — a genuinely different, apparently-abandoned path. Kept as a branch
  (not deleted) since its status as "superseded" vs. "still wanted" is a
  judgment call better left to the research owner; tagged for
  discoverability.

### Named as currently important — kept as branches, tagged

Per repository ground truth (confirmed, not assumed): `research/wave7-teacher-distillation`,
`research/wave7-t2-promotion`, `research/wave8-future-aware-distillation`
are each pinned by an active worktree and represent the load-bearing recent
history of the canonical research ladder. All three are kept as permanent
branches in addition to the new `research/current` pointer.

### Standalone completed studies — kept, tagged (both have active worktrees)

`codex/oracle-information-frontier-2026` and
`codex/reproduce-2025-public-solution` are sibling, not-merged research
studies (2025-reproduction audit, oracle information-frontier study) that
informed the later canonical Wave 5+ work but were never merged in via git.
Both are pinned by active worktrees — not eligible for deletion regardless.
Tagged as milestones.

### Production lineage — new permanent branch + tag

`claude/rt600-baseline-submission` (tip `9aaa9b0`, 46 commits) is the real
frozen deployment lineage: environment build, LB-002/003/004 leaderboard
verification runs, and submission #5. External Crunch leaderboard score
(LB-001) is **0.6268**, confirmed in `research/EXPERIMENT_ID_MAP.md` on
`research/wave7-t2-promotion`. Pinned by an active worktree — kept as a
branch; also becomes the basis of the new `production/rt600` branch.

## D. Untracked working-tree material — traced individually

| Path | Disposition | Evidence |
|---|---|---|
| `sbr-research*.bundle` (6 files), `structural-break-research-2026.tar.gz` | Redundant — every bundle head SHA is already a reachable commit (`65b0e21`, `46583f5`, `e98f5b9`, `ea5e2f2`, `23cb4da`, `8e76ad2`) once `research/multi-agent-2026` is pushed. Safe to remove from the working tree. | `git bundle list-heads` on each |
| `RESULTS.csv`, `STATE_OF_RESEARCH.md` (top level) | Stale snapshots (79 rows / "79 experiments", up to `RT-131`) — both `codex/wave3-engineering` (up to `RT-213`) and `research/multi-agent-2026` (up to `RT-190`) already have newer committed copies. Safe to remove. | row counts + commit dates compared directly |
| `BRIEF_CLAUDE_wave3_alpha.md`, `BRIEF_CODEX_wave3_engineering.md`, `HANDOFF_WAVE3.md` (root) | Byte-identical duplicates of the copies already sitting in the untracked `research/` subdir. Neither copy is committed anywhere. Preserved by committing one copy into the archive. | `diff -q` root vs. `research/` |
| `20260818_claude_update_1.md` | Differs (19 diff lines) from the version committed on `codex/wave3-engineering` — genuinely distinct content, not a pure duplicate. Preserved into the archive. | `diff` |
| `platform_constraints.md` (root) | Byte-identical to `research/reports/platform_constraints.md` already committed on `research/multi-agent-2026`. Redundant once that branch is pushed. Safe to remove. | `diff -q` |
| `deep-research-report (3).md` | Not found committed anywhere under any branch. Preserved into the archive. | `git log --all` search |
| `wave2/PUBLIC_IDEA_MAP.md`, `redteam_generator_artifacts.md`, `runner_semantics.md`, `STATE_OF_RESEARCH_V2.md`, `VALIDATION_V2.md` | Same basenames exist committed on `codex/wave3-engineering` (commit `8e76ad2`) but content differs (not byte-identical — likely an earlier draft). Preserved into the archive rather than discarded, since "same name" is not proof of "same content." | `diff -q` (differ) |
| `wave2/RESULTS_wave2.csv` | **Never committed anywhere**, under any path, in `git log --all`. Preserved into the archive. | `git log --all --diff-filter=A` search, empty |
| `wave2/models_rt100_stream.tar.gz`, `wave2/models_rt150_ensemble.tar.gz` | Model binaries. Per the generated-artifact policy (section 16 of the operating brief), these should **not** be committed. Left on disk, untracked, `.gitignore`'d going forward — not deleted (deleting a user's local files is out of scope for a git-history cleanup). | policy |
| `submissions/A_rt100_streaming.ipynb` | **Never committed anywhere.** A real deliverable notebook sitting only in the working tree. Preserved into the archive. | `git log --all --diff-filter=A` search, empty |
| untracked `research/` (3 files) | Duplicates of the root `BRIEF_*`/`HANDOFF_WAVE3.md` files, see above. | `diff -q` |

## E. Large-file finding (no history rewrite performed)

`git rev-list --objects --all` + `git cat-file --batch-check` shows the
largest blobs in history are dozens of near-duplicate ~28 MB copies of
`submissions/C_ensemble_deployable.ipynb` and `.py`, re-committed repeatedly
with embedded outputs. This is the dominant contributor to the 253 MB
`.git` size. Per the absolute safety rules, history was **not** rewritten
and no `filter-repo`/BFG pass was run. Recommendation for future sessions:
strip notebook outputs before committing (`nbstripout` or equivalent), and
consider Git LFS if large model artifacts must be versioned going forward.

## F. PR #11

`gh pr diff 11 --name-only` and the PR body confirm it adds a self-contained
`StreamingBreakDetector` (EWMA z-score + two-sided CUSUM + EWMA
variance-ratio via noisy-OR) — this is the RT-000-equivalent baseline
described in the earliest row of `RESULTS.csv`. It predates and is
superseded by RT-600 (external score 0.6268) and the entire Wave 5–8
research ladder. Confirmed superseded; disposition recorded in
`docs/REPOSITORY_CLEANUP_REPORT.md`.
