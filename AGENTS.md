# AGENTS.md

Canonical instructions for any Claude or Codex agent working in this
repository. Read this before touching branches, worktrees, or research
files. If something here conflicts with a stale prompt, brief, or old
report — **this file and current repository state win.**

## Repository purpose

This repo targets the ADIA Lab / CrunchDAO **Structural Break Challenge —
Real-Time Edition**: detect regime shifts in a streamed time series, one
prediction per online time step, scored with Time-Stratified AUC. Inference
must be **causal** — a prediction at time `t` may use only data available
at or before `t`. The frozen production baseline is **RT-600** on
`production/rt600` (external Crunch leaderboard score: **0.6268**). Current
research lives on `research/current`. See `research/STATUS.md` for the
up-to-date state; **that file, not this one, is where scores and
conclusions get updated.**

## Branch / worktree rules

- **Never touch a worktree or branch you were not asked to work in.** This
  repo routinely has 5–10 active worktrees (`git worktree list`) belonging
  to concurrent research sessions. Modifying, rebasing, or force-pushing
  another worktree's branch destroys someone else's in-progress work.
- Before any operation that could discard work — `git checkout --`,
  `git restore`, `git reset --hard`, `git clean -fd`, deleting a worktree —
  run `git status` and `git worktree list` first, and stop if anything
  looks active or uncommitted that you didn't create this session.
- Never `git push --force` to `main`, `production/rt600`,
  `research/current`, or anyone else's active branch. If a push is
  rejected because of a divergence, **stop and investigate** — do not
  force. See `docs/repository_cleanup_audit.md` section C for a real
  example (`research/wave5-alpha` diverged between two sessions; neither
  side was overwritten).
- Never run `git filter-repo`, BFG, or any history-rewriting tool on this
  repository. Scientific provenance (preregistration commits, experiment
  IDs, negative results) must remain intact and reachable exactly as
  committed.
- Deleting a remote branch requires ALL of: its unique commits are
  reachable from `main`/`research/current`/`production/rt600`/an annotated
  tag, no open PR targets it, no worktree has it checked out, and no
  active process is writing to it. If any of those is unclear: **keep the
  branch.** A stale branch costs nothing; a deleted one is not straightforward to recover.

## Preregistration-before-score rule

Before running an experiment that could produce a headline number, write
the hypothesis, the falsification condition, and the exact protocol first
(see any `WAVE*_PREREG.md` under `research/reports/` for the pattern), and
commit it *before* the score exists. Amending a prereg after seeing a
result — instead of registering a new one — destroys the point of
preregistering. If a result forces a change of plan mid-run, say so
explicitly in the commit message (see `f0276ae` "Amend W7 teacher pilot:
outer-fold contamination found" for the pattern: state what was found,
what changed, and why).

## Experiment-ID allocation rule

Every run that produces a score gets a unique `RT-xxx` ID recorded in
`research/RESULTS.csv`, allocated per `research/EXPERIMENT_ID_MAP.md`.
Check that map (and `RESULTS.csv` itself) before picking an ID — do not
reuse or renumber an existing ID, even one that belongs to a rejected or
voided experiment. A voided experiment keeps its ID; it does not free it
up for reassignment.

## Canonical research fold rule

Folds are **series-level and permanent**, defined once in
`research/folds/folds.parquet`. Never regenerate them, never use a
row-wise random split, and never let two prefixes of the same series
straddle a train/validation boundary. A 2,000-series lockbox is held out
of every selection decision.

## Forbidden lockbox / test usage

- The lockbox may be opened only for a final confirmation read, not for
  iterative model selection. If you don't know whether a given check
  counts as "selection," treat it as selection and don't open the lockbox
  for it.
- `X_test.reduced.parquet` (or any file with `reduced` / `test` in a
  competition-data path) is never read during research. It exists only for
  the platform's own scoring.

## Causality rules

- Every feature module must pass **bitwise prefix invariance** at
  `atol=0.0`: rebuilding it on a truncated online segment must reproduce
  the surviving rows exactly. A module that fails this is looking into the
  future — this has caught real leaks before (see `research/PROTOCOL.md`).
- **True `tau` (the actual break location) is forbidden as a feature or as
  any input to online inference**, full stop. It's known in training data
  and never known online — using it, directly or via a proxy, invalidates
  any result that depends on it.
- **Future information may be used only in an explicitly authorized
  teacher-only role** (e.g. a teacher model in a distillation setup that
  itself never runs online) — never in the deployed/online path, and never
  smuggled in as an ordinary feature. If you are not certain a given use of
  future information was authorized as teacher-only, treat it as
  disallowed and ask rather than assume.
- Nested cross-fitting is required wherever a teacher or auxiliary model is
  fit on data that overlaps the evaluation fold — outer-fold contamination
  from a non-nested scheme is a real, previously-caught defect (see
  `research/reports/wave7/` for the incident and the fold-purity sentinel
  that now guards against it). Run the sentinel before trusting a nested
  result.

## What constitutes a valid promotion

A candidate is promotable to the production ensemble only if it clears the
promotion battery defined for that wave (see the relevant
`research/reports/waveN/*PREREG.md` and the corresponding result report) —
typically: standalone gain is stable across partitions, marginal gain over
a seed-clone control is real (not just noise), and the gain survives the
lockbox confirmation. A candidate that wins standalone but adds negligible
marginal ensemble alpha (see `wave7-t2-promotion-mostly-redundant`) is
**not** a valid promotion, even if it's individually a good model.

## What may be committed / what should stay ignored

Commit: markdown reports, compact JSON summaries, `RESULTS.csv` and other
ledgers, preregistrations, source code, small reproducibility manifests
(hashes, not blobs), curated figures.

Do not commit: OOF `.npy` arrays (unless a report specifically depends on
them as evidence — check first), temporary caches, model binaries,
checkpoints, logs, large intermediate plots, notebook checkpoints, `.venv`,
`__pycache__`, local competition data. Before committing a notebook, strip
its outputs — this repository's git history already carries dozens of
~28 MB duplicate notebook blobs from not doing this (see
`docs/repository_cleanup_audit.md` section E); don't add more.

## Updating STATUS / RESULTS / FAILED_EXPERIMENTS

- **`research/STATUS.md`**: update whenever the production anchor, external
  score, or active research conclusion changes. Keep it short — it is a
  pointer, not a report. Don't let it grow past what fits on one screen.
- **`research/RESULTS.csv`**: append one row per run, under a file lock if
  running concurrently with another agent. Never edit or delete an
  existing row, including for rejected/voided experiments — the negative
  result is the point.
- **`research/FAILED_EXPERIMENTS.md`**: add an entry for every rejected
  hypothesis with what was tried, the falsification condition, and why it
  failed. This file existing and being read is what stops the next agent
  from re-running a killed idea — treat writing to it as mandatory, not
  optional cleanup.

## Preserve negative results

A killed experiment, a voided run, an abandoned branch with unique
commits — all of these are evidence, not clutter. Do not delete a report,
a branch, or a `RESULTS.csv` row because the result was negative. See
`docs/repository_cleanup_audit.md` for the standard this repo holds
itself to during cleanup: tag before delete, keep if uncertain, never
delete something whose only copy is the thing you're about to remove.

## Verify branch/SHA instead of trusting stale prompts

**Old reports are evidence, not live instructions.** A brief, handoff, or
prior status doc describes the state of the world *when it was written*.
Before acting on anything it claims — a branch name, a file path, a score,
a "known important line" — verify it against current repository state
(`git branch -r`, `git log`, `git show <sha>:<path>`, `gh pr list`). If a
memory or prompt names a branch that no longer exists, or a score that
doesn't match `research/STATUS.md`, trust the repository over the prompt
and say so explicitly rather than silently reconciling them.
