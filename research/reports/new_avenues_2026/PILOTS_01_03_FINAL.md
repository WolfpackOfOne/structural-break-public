# FIRST THREE NEW-AVENUES PILOTS -- FINAL

## A. Git / Environment

Branch: `research/new-avenues-pilots-2026`.

Worktree:
`/path/to/workspace/structural-break-new-avenues-pilots`.

Starting `origin/research/current` SHA:
`6c37cb122f83012c39706f1f182ce8e3182461be`.

Scored-result HEAD before this final report:
`d1d6bd6`.

Cleanup incorporated in the initial scoring commit: **No**. Post-cleanup
revalidation merged `origin/research/current` at
`aca2c4f9b68ad6315956f7c499ebefb2ae311f7b` and reproduced Pilots 1-3 exactly;
see `post_cleanup_revalidation.{md,json}`.

Cache setup: local ignored `cache/store` and `cache/features` symlink to
`structural-break-claude-wave3/cache`; local ignored `research/oof` contains
symlinks to incumbent OOF streams plus local candidate OOF artifacts.

## B. Preregistration

Preregistration commit: `2dfde1d`.

File: `research/reports/new_avenues_2026/PILOTS_01_03_PREREG.md`.

Reserved IDs:

| ID | pilot | status |
|---|---|---|
| none | Pilot 1 diagnostics | no RT ID consumed |
| `RT-1200` | relay score-state | KILL |
| `RT-1201` | IM2+dwell scalar | KILL |

Degrees of freedom: one diagnostic family for Pilot 1, one relay score-state
variant, one IM2+dwell scalar. No validation-tuned thresholds, blend weights, or
post-fold0 variants.

## C. RT-600 Reproduction

RT-600 anchor reproduced from committed OOF streams:

| metric | value |
|---|---:|
| dev mean TS-AUC | 0.625811 |
| dev pooled TS-AUC | 0.625627 |
| dominant-cell AUC | 0.664277 |
| fold-0 integration E0 | 0.638276 |

## D. Pilot 1 -- Specialist Competence / Failure Manifolds

Hypothesis: existing RT-600 specialists may have routable competence by
history-only DGP fingerprints or failure manifolds.

Diagnostics:

| diagnostic | result |
|---|---:|
| best fold-held selector delta vs RT600 | -0.023559 |
| best fold-preserving permutation selector delta | -0.019020 |
| strongest specialist-advantage Spearman | +0.1529 (`kurt`, never-break) |
| specialist disagreement vs RT600 loss Spearman | +0.1876 |
| cluster eta2 loss max | 0.000057 |

Verdict: **WEAK**.

Implication: history fingerprints remain a weak monotone substrate, especially
for heavy-tail never-break failures, but the simple fold-held specialist selector
is not viable and does not justify building a legal gate now.

## E. Pilot 2 -- Relay Score-State

Candidate: `RT-1200`, causal operate/reset, thermal-replica, picked dwell, and
recloser-cycle state applied to RT-600 score evidence.

| metric | value |
|---|---:|
| standalone whole-fold TS-AUC | 0.619611 |
| dominant-cell AUC | 0.658230 |
| mature vs never-break | 0.655859 |
| mature vs pre-break | 0.664844 |
| within-t corr vs RT600 | +0.7746 |
| dominant-cell pair net | -318 |
| E0 RT600 | 0.638276 |
| E1 RT600 + seed clone | 0.638586 |
| E2 RT600 + RT-1200 | 0.638380 |
| marginal vs clone | **-0.000207** |
| runtime | 99.6s |

Verdict: **KILL**. The score-state transform adds essentially no ensemble value
and damages same-t pair flow in the dominant cell.

## F. Pilot 3 -- IM2 Matched-Length Run Null / Dwell Bank

Candidate: `RT-1201`, nine-feature direct scalar from AR(2)-residual-square
dwell state over windows 32, 64, and 128.

| metric | value |
|---|---:|
| standalone whole-fold TS-AUC | 0.583578 |
| dominant-cell AUC | 0.610158 |
| mature vs never-break | 0.613459 |
| mature vs pre-break | 0.600945 |
| within-t corr vs RT600 | +0.3817 |
| dominant-cell pair net | -1151 |
| E0 RT600 | 0.638276 |
| E1 RT600 + seed clone | 0.638586 |
| E2 RT600 + RT-1201 | 0.638888 |
| marginal vs clone | **+0.000301** |
| feature build runtime | 178.0s |
| total runtime | 254.4s |

Verdict: **KILL**. The channel is genuinely low-correlation, but as a direct
score it is far too weak to clear the seed-clone marginal gate.

## G. Comparative Table

| pilot | information channel | standalone | dominant | corr RT600 | marginal vs clone | pair net | runtime | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | history-DGP routing | n/a | selector delta -0.023559 | n/a | n/a | specialists all net negative | 81.8s | WEAK |
| 2 | relay reset dynamics | 0.619611 | 0.658230 | +0.7746 | -0.000207 | -318 | 99.6s | KILL |
| 3 | contiguity / run-null dwell | 0.583578 | 0.610158 | +0.3817 | +0.000301 | -1151 | 254.4s | KILL |

## H. Scientific Interpretation

New information appears most plausible in the dwell/run-null channel: its
within-t correlation with RT-600 is low, matching the D4 signal. However the
direct scalar is too weak in same-t ranking geometry, and weak standalone
channels do not beat the exchangeable seed-clone control.

Relay reset logic on the final RT-600 score is redundant with, or weaker than,
ordinary score-state smoothing. It should not be pursued further as an F-mode
post-processor without a new reason.

History-only DGP conditioning is real but weak. It may help explain failures,
especially heavy-tail never-break false positives, but the existing specialists
are not cleanly routable by a simple fingerprint selector.

Falsified here:

* final-score relay state as a cheap marginal-alpha post-processor;
* this direct IM2/dwell scalar as a promotion candidate;
* simple history-only specialist selection as an immediate gate.

Still open:

* dwell information as a trained feature block or as joint size-duration rarity;
* lower-level relay logic only if preregistered as a feature block, not as a
  score post-processor.

Pilot 4 trajectory shape / nearest-neighbour provenance was subsequently run
as `RT-1202` with shuffled control `RT-1203` and also killed.

## I. Next Action

**PILOT 4 COMPLETED SEPARATELY; CONTINUE THE BROAD SWEEP ONLY AFTER THE
POST-CLEANUP CHECKPOINT.**

No candidate deserves 5-fold confirmation. `RT-1200` and `RT-1201` both fail the
`+0.0010` marginal-vs-clone gate.

## I.1 Post-Cleanup Reproduction

Cleanup SHA: `aca2c4f9b68ad6315956f7c499ebefb2ae311f7b`.

`pytest -q tests/test_novel_streams_harness.py`: 11 passed.

`research/scripts/check_research_hygiene.py`: OK, 215 experiment rows, no
duplicate IDs.

| item | old marginal | clean marginal | changed | final verdict |
|---|---:|---:|---|---|
| Pilot 1 diagnostic | n/a | n/a | no affected path | WEAK |
| `RT-1200` | -0.000206633 | -0.000206633 | no | KILL |
| `RT-1201` | +0.000301470 | +0.000301470 | no | KILL |

Floating metrics were compared at tolerance `1e-12`; pair-flow counts matched
exactly. Existing `RESULTS.csv` rows were not duplicated or edited.

## J. GitHub / CI

Pushed commits:

| SHA | message |
|---|---|
| `2dfde1d` | Preregister first three new-avenues pilots |
| `0805df3` | Execute specialist competence diagnostics |
| `f73d5dc` | Add relay score-state pilot runner |
| `a0e53c4` | Evaluate causal protective relay score state |
| `f8f4c86` | Add IM2 dwell pilot runner |
| `82d9588` | Handle first-valid dwell scalar rows |
| `d1d6bd6` | Evaluate IM2 dwell run-null scalar |

Research-hygiene workflow:

| run | commit | result |
|---|---|---|
| `32753144718` | `a0e53c4` | success |
| `32754371146` | `d1d6bd6` | success |

Tests / verification:

* `research/scripts/check_research_hygiene.py`: passed locally after each
  RESULTS edit.
* Pilot scripts: `py_compile` passed.
* `RT-1200`: prefix-state verification passed on 8 series / 29 prefixes.
* `RT-1201`: prefix verification passed on 8 series / 29 prefixes.
* Post-cleanup reproduction: `RT-1200` and `RT-1201` matched exactly after
  merging `aca2c4f`; Pilot 1 had no affected harness path.
* Full pytest was not run; these pilots did not edit shared package code.

PR: not opened during the initial first-three-pilot run because the expected
cleanup commit had not landed on `origin/research/current` yet. The cleanup is
now merged into this pilot branch and the first-three-pilot results are final
screen results.

Lockbox/test usage: none. Production branch: untouched.
