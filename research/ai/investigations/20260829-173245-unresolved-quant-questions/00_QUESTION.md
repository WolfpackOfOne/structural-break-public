# 00_QUESTION.md

## Investigation ID
20260829-173245-unresolved-quant-questions

## Timestamp (UTC)
2026-08-29T17:32:45Z

## Research Question

Identify the three most important unresolved quantitative research questions
documented in this repository, based only on:

- current research reports,
- `RESULTS.csv`,
- recent Git history.

Do not propose new training runs.

## Scope

- Primary repository under investigation: `structural-break/` (the only
  directory under the workspace root that is an actual Git repository and the
  only one containing a tracked/untracked `RESULTS.csv` experiment ledger).
- The workspace root (`/path/to/workspace`) contains ~25
  sibling directories named `structural-break-*` that appear to be
  historical/parallel wave checkouts or worktrees of the same competition
  project (structural break detection). These are NOT separate Git
  repositories (no `.git` directory found in any of them) and should be
  treated as read-only historical snapshots/context, not as independent
  provenance sources, unless the scout finds explicit evidence they are
  authoritative for specific branches/waves.
- Evidence sources to prioritize, in order:
  1. `structural-break/RESULTS.csv` (experiment ledger, 79 rows incl. header,
     columns include experiment_id, git_sha, hypothesis, falsification_condition,
     mean_oof_ts_auc, pooled_oof_ts_auc, per_fold_ts_auc, status, notes, protocol).
  2. Current research reports/state docs in `structural-break/` (e.g.
     `STATE_OF_RESEARCH.md`, `README.md`, `experiments/README.md`,
     `experiments/progress.md`, `experiments/leaderboard.csv`,
     `experiments/best_config.json`, handoff/brief docs).
  3. Git history of `structural-break/` (commit log, branches, especially
     recent commits and any `research/*`, `engineering/*`, `production/*`,
     `release/*` branches that indicate ongoing or abandoned research
     threads).
- "Unresolved" means: a question that is explicitly flagged as open,
  contradicted, KILLED/NEGATIVE-but-ambiguous, UNVERIFIED, or where the
  ledger/report/history show disagreement, inconclusive falsification, or
  an explicitly stated open problem.

## Constraints

- Follow `.opencode/quant/RESEARCH_PROTOCOL.md`,
  `.opencode/quant/ARTIFACT_SCHEMA.md`, `.opencode/quant/WORKFLOW.md`.
- Use the evidence hierarchy labels; do not upgrade REPORT-CLAIM to VERIFIED
  without independent reproduction.
- Apply OOF discipline, metric discipline, and selection-bias scrutiny to any
  AUC claims drawn from `RESULTS.csv` or reports.
- Prefer cheap ($0) diagnostics over expensive retraining when adjudicating
  claims.

## Non-Goals

- Do NOT propose, design, or preregister any NEW training run or experiment
  that requires fresh model training. (The user has explicitly excluded
  this.) A preregistration should only be created if it can be satisfied by
  analysis of EXISTING artifacts (OOF vectors, predictions, RESULTS.csv rows,
  historical logs) -- i.e. a diagnostic/analysis experiment, not a training
  experiment. If no such zero-training-cost experiment is justified, the
  synthesizer should explicitly say no preregistration is warranted.
- Do NOT modify production code, model code, experiment code, or any file in
  `structural-break/` or any sibling directory.
- Do NOT run any training scripts.
- Do NOT proceed to implementation. Implementation requires an explicit
  separate user command (`/quant-implement`).

## Starting Git State (structural-break/, the canonical repo)

- Branch: `claude/local-model-results-review-lk0f7i`
- HEAD SHA: `33fa041921f2e29d3b7bfbffcfac565997f9789a`
- Worktree status: DIRTY (30 changed/untracked paths, including untracked
  `RESULTS.csv`, `STATE_OF_RESEARCH.md`, `research/`, `wave2/`, several
  `.bundle` files, and modified files under `experiments/`, `scripts/`,
  `src/structural_break/`, `tests/`).
- Remote: `origin` -> `https://github.com/WolfpackOfOne/structural-break.git`
- Many remote/local branches exist representing parallel research waves
  (`research/*`, `engineering/*`, `production/*`, `release/*`, `codex/*`).

## Workspace-Level Provenance (parent folder, not a Git repo)

- `/path/to/workspace` itself is NOT a Git repository.
- It contains the canonical repo `structural-break/` plus ~25 sibling
  directories that look like separate wave/agent checkouts
  (`structural-break-catboost-specialist-2026`,
  `structural-break-causal-representation-frontier`,
  `structural-break-claude-wave3`, `structural-break-codex-wave3`,
  `structural-break-data-forensics-2026`,
  `structural-break-deep-ensemble-frontier-2026`, etc.). None of these
  contain a `.git` directory. The scout should determine whether any of them
  contain reports/results materially different from `structural-break/`.

## User-Supplied Claims Requiring Verification

None supplied directly. The question itself presumes that the repository's
current reports, RESULTS.csv, and Git history already document unresolved
quantitative questions -- this presumption should be checked, not assumed.
