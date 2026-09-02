# Recovery note — filed 2026-09-01

This investigation was committed on 2026-08-29 as `75fd841` on branch
`research/investigation-20260829-unresolved-questions`, in the legacy
`structural-break` worktree. That branch was never merged and the worktree is a
disconnected lineage, so none of this was reachable from `main`. The files are
recovered here byte-for-byte from that commit; only this note is added.

## Why it was worth recovering

It is a structured multi-role audit — question, evidence manifest, primary
research, independent skeptic, statistical audit, synthesis, preregistration —
of the three most important unresolved quantitative questions, with every claim
tagged by evidence class (`[GIT-VERIFIED]`, `[ARTIFACT-VERIFIED]`,
`[REPRODUCED]`, `[REPORT-CLAIM]`).

It also diagnosed the repository pathology this session kept running into:

> **[GIT-VERIFIED]** Canonical HEAD `33fa041` is a disconnected 16-commit harness
> lineage; later research lives on shared-object-database branches, tags, and
> worktrees.

> **[ARTIFACT-VERIFIED]** Canonical `RESULTS.csv` is a wave-1 fossil: 78 rows, 77
> with `git_sha=nogit`; it still labels illegal-oracle RT-131 `NEW CHAMPION`.

## What has happened since, as of 2026-09-01

Its three questions have partly resolved, so read the conclusions as of
2026-08-29 rather than as current state:

- **Q2 (CatBoost hybrid robustness).** It noted "alternate-partition evidence
  are absent" and called for "one existing-artifact-only CatBoost audit". That
  audit was done (`RT1257_SLOT_ADJUDICATION.md`), and the residual CatBoost slot
  lane is now **closed** — all four of CAT-411/412/414/415 are
  `PARKED / NO_PROMOTION`, every deployment endpoint below the 0.0011 floor.
- **Q1 (what Arm C's +0.07110 dominant-cell advantage measures).** Still open as
  a mechanism question, but the causal student built from that teacher (RT-1320)
  has since passed a four-partition alternate-partition gate, mean E2−E1
  +0.0017678 with 4/4 partitions positive.
- **Q3 (distance to the ceiling of the legal causal information set).** Open.

The rest of that branch — the `src/structural_break` weekend model-search harness
in `33fa041`, `7564c7b`, `603f548` — is **not** recovered. It targets the legacy
`src/structural_break` package, superseded by `src/sbr`, and the investigation
itself established that its reported 0.598843 is "single-split series AUC, not
five-fold row-level TS-AUC". Deliberately left unmerged.
