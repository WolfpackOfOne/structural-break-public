# WAVE 8 — FUTURE-AWARE DISTILLATION — FINAL REPORT

Pre-registration: `research/WAVE8_FUTURE_AWARE_PREREG.md` (commit `78fa5cf`).
Pilot fold: 0. All numbers in this report are fold-0-pilot numbers unless
stated otherwise.

## A. START STATE

- Branch: `research/wave8-future-aware-distillation`, worktree
  `/path/to/workspace/structural-break-wave8`.
- Starting SHA: `902d232` ("Pre-register nothing yet: propose Wave 8
  future-aware distillation routes").
- Newer commits since `902d232`: none on this branch's own history at
  start; `research/wave7-teacher-distillation` had advanced past this
  branch's fork point to `5093e0a` with T2's final result — ingested, not
  merged (see B).
- No active training process found anywhere on the machine at start.

## B. WAVE-8 PRE-REGISTRATION

- Prereg SHA: `78fa5cf`.
- Experiment IDs: `RT-1000`–`RT-1099` reserved; used
  `RT-1006`/`RT-1007` (SST), `RT-1020`/`RT-1021` (ORR), `RT-1030`–`RT-1033`
  (PCFB), `RT-1041`/`RT-1042` (CFEP), `RT-1051`/`RT-1052` (TGMC).
- Pilot fold: 0 (fixed before any score existed, per the proposal's own
  choice).
- T2 gate: `RT-995` (`research/wave7-teacher-distillation@5093e0a`) is a
  **final** 5-outer-fold nested result, not the WIP checkpoint the original
  proposal cited — mean +0.00943 vs `RT-990`, 5/5 folds positive, bootstrap
  CI [+0.0043, +0.0137] entirely above zero. Clears every *measured*
  promotion leg (magnitude, folds, bootstrap); leg 4 (alternate-partition)
  and ensemble-integration are authorized but unrun. This licensed Wave 8 to
  proceed.

## C. LEAKAGE / CAUSALITY TESTS

`tests/test_wave8_causality.py`, 7 tests, all green before any Wave-8 score
was read: future-row-index correctness against direct lookup, nested-scheme
purity (set-arithmetic AND a live call through `nested_oof_regressor`),
old-scheme contamination correctly reproduced (20/20 `(f,g)` pairs), no
forbidden column reachable by a student, eligibility-diagnostic
well-formedness, determinism.

Pre-flight diagnostic (`research/reports/wave8_eligibility_diagnostic.json`):
target-availability (`t+h` within the series) is materially
class-conditional at matched `t` — e.g. `h=200, t<20`:
`P(eligible|y=1)=0.33` vs `P(eligible|y=0)=0.81`. Handled per prereg: never
a feature; oracle arms restricted to the eligible subpopulation, legal arms
use the full population, never mixed.

## D. SST

8 structural channels × {50,100,200}, combined.

| | value |
|---|---:|
| oracle utility (B−A, eligible-pop, dominant cell) | +0.00616 |
| legal utility (C−A, dominant cell, **full pop**) | **−0.00060** |
| legal utility (C−A, dominant cell, eligible-pop) | +0.00810 |
| retention R (eligible-pop) | 1.32 (legal ≈ oracle on this slice — small-fold noise, not a general claim) |
| RT600+SST marginal vs RT600+clone | −0.00031 |

**PASS/FAIL: FAIL.** Flat on the population that actually matters (the
whole dominant cell, not the eligible slice). KILL per gate.

## E. ORR

| | value |
|---|---:|
| teacher-confirmed inversions mined | 877 |
| recoverability P(repair right \| RT600 wrong, teacher confident) | **0.737** (>>0.55 bar) |
| repairs / damage (pair diagnostic) | 769 / 694 |
| RT600 alone vs RT600+repair (whole fold0) | 0.63828 → 0.63823 (−0.00005) |
| RT600+ORR marginal vs RT600+clone | −0.00032 |

**Verdict: FAIL, but the most informative negative result of the five.**
The repair mechanism is real and visible at the row level — it is not the
"blend moved but the mechanism is invisible" failure mode the prereg warned
about. It simply does not move the aggregate metric: the confirmed-pair
population is too small a share of the metric's pair weight.

## F. PCFB

h=200, d=16, PCA vs PLS.

| | PCA16 | PLS16 |
|---|---:|---:|
| oracle (eligible cell) | +0.01100 | +0.00566 |
| legal (full-pop cell) | −0.00148 | **−0.00270** |
| retention R | 0.071 | **−0.31** |
| RT600+cand marginal vs clone | −0.00031 | −0.00031 |

**Verdict: FAIL.** The core hypothesis (PLS's predictability-selected
subspace should beat PCA's pure-information subspace) is **contradicted** —
PLS legal is worse than PCA legal by −0.0025, and PLS's own retention is
negative.

## G. CFEP

Causal-TCN shell (RT-970/971 architecture) + 16-dim bottleneck, BCE vs
future-predictive (MSE against SST's true h=200 target) objectives.

| | whole fold0 | dominant cell |
|---|---:|---:|
| A control (`RT-990`) | 0.62656 | 0.66381 |
| B BCE embedding (`RT-1041`) | 0.62540 | 0.66647 |
| C future embedding (`RT-1042`) | 0.62166 | 0.66272 |
| **C − B (binding)** | **−0.00374** | **−0.00375** |

**Verdict: FAIL, decisively.** The future-predictive objective loses to its
own matched-capacity BCE control, not just to the bare column control.

Two real bugs were found and fixed en route (both documented in the commit
history): (1) an unstandardized SST target channel (range ±6000) exploded
MSE training to NaN loss on every epoch; (2) `NaN * mask ≠ 0` in floating
point, so eligibility-masked rows still poisoned the loss even after
standardization, until explicitly zeroed. The reported numbers are from the
corrected run.

## H. TGMC

Weak persistent location-displacement, 292 base series, 1460 candidate
pairs, 234 teacher-confirmed.

| | whole fold0 (real data) | dominant cell |
|---|---:|---:|
| control (`RT-990`) | 0.62656 | 0.66381 |
| TGMC-R (random, `RT-1051`) | 0.61738 | 0.64912 |
| TGMC-T (teacher, `RT-1052`) | 0.62240 | 0.66211 |

T − R = **+0.00502** (teacher-guided beats random, as hypothesized) but
**both sit below the no-synthetic-data control.** Adding either kind of
synthetic row to real training data made fold-0 worse.

**Verdict: FAIL.** Per the pre-registered kill condition: real outer TS-AUC
flat/negative regardless of synthetic-pair quality → kill without tuning
the simulator.

## I. FIVE-PILOT COMPARISON TABLE

See `research/reports/wave8_pilot_comparison.md` for the full table
(including the pair-repair diagnostic). Summary:

| mechanism | dominant-cell Δ (full pop) | RT600+cand marginal vs clone | verdict |
|---|---:|---:|---|
| SST | −0.00060 | −0.00031 | FAIL |
| ORR | −0.00004 (ensemble-scale, see note) | −0.00032 | FAIL |
| PCFB (PLS16) | −0.00270 | −0.00031 | FAIL |
| CFEP | −0.00375 (vs matched BCE control) | −0.00031 | FAIL |
| TGMC | −0.00170 | −0.00031 | FAIL |

## J–L. FULL-CV PROMOTIONS / BOOTSTRAP / ALTERNATE PARTITIONS

**N/A.** No mechanism cleared its pilot gate, so none advanced to the
5-fold protocol, paired bootstrap, or alternate-partition confirmation, per
the pre-registered promotion rule (execution-brief section 23).

## M. T2 INTEGRATION

Not attempted for combination with any Wave-8 mechanism, since none
qualified. T2 (`RT-995`) itself stands as evidence, cited by SHA (see B) —
it is not re-derived or re-scored on this branch.

## N. REDUNDANCY MATRIX

`research/reports/wave8_futureaware_redundancy.md`. T2 correlates 0.90–0.91
(within-t rank) with SST/PCFB/CFEP/TGMC's repair patterns but only 0.67 with
ORR — ORR's blend is nearly RT600 itself (its correction was negligible), so
this reflects RT600's own baseline correlation with T2, not a shared repair
signature. No pair shows the low-correlation, independent-repair-reservoir
signature that would matter here — moot, since nothing cleared its gate.

## O. BEST ACTUALLY MEASURED CAUSAL ENSEMBLE

Unchanged by Wave 8: the RT-600 7-specialist ensemble (`RT-250`/`RT-600`
production, external 0.6268). `T2` remains a promising but
**not fully validated** addition (clears 3 of 4 measured promotion legs;
alternate-partition and ensemble-integration legs are unrun) — it is not
"the ensemble" yet, it is the strongest pending candidate to become one.

## P. SCORE LADDER

- External RT-600 anchor: **0.6268**.
- No measured increment from Wave 8 — every candidate's marginal
  contribution over `RT600+seedclone` was negative or effectively zero
  (all ≈ −0.0003).
- T2's own measured dev-fold marginal (+0.00943, not yet an external or
  lockbox number, and not yet ensemble-integration-tested) is the only
  positive figure anywhere in this wave's evidence chain. It predates Wave
  8 and is not claimed as a Wave-8 result.

## Q. 0.650 ASSESSMENT

**NOT SUPPORTED** by measured marginal alpha. Wave 8 measured zero net
positive contribution across five distinct mechanisms and five real pilots.
Even crediting T2's un-integration-tested dev-fold marginal at full value
(which the project's own accounting rule, section 6, explicitly forbids
doing naively) would put the ladder near 0.636 — still well short of 0.650
— and that number itself needs its own remaining validation legs before it
can be trusted at all. Nothing in this wave's evidence supports "possible,"
let alone higher.

## R. NEXT SINGLE HIGHEST-VALUE EXPERIMENT

**Finish T2's own promotion battery** (leg 4 alternate-partition
confirmation + ensemble-integration test against the RT-600 7-specialist
set) before proposing a sixth future-aware mechanism. T2 is the only
component in the entire pipeline — Wave 8 included — with a real,
statistically supported positive marginal signal, and it has not yet been
tested the one way that would actually matter for the leaderboard: added to
the seven specialists and re-measured as a a full ensemble, on a partition
Wave 8 has not looked at. Inventing a sixth distillation mechanism now,
with five consecutive negative results in hand and the exact premise
(`T2`'s reliability) still only 3/4-validated, would not be a well-aimed
use of compute.

## S. GIT END STATE

- Ending SHA: this file's own commit (see below).
- Pushed: yes, every commit boundary pushed to
  `origin/research/wave8-future-aware-distillation`.
- Clean tree: yes, after this commit.
- Active training jobs: none.

## FINDING PER EXECUTION-BRIEF SECTION 29

> FUTURE-AWARE TRANSFER FAMILY DID NOT CAPTURE ENOUGH MARGINAL ALPHA.

All five mechanisms killed at the fold-0 pilot per their pre-registered
gates. This is not read as "the future contains no information" (W7-D3R
already proved otherwise, +0.07110 cell AUC oracle gap) — it is read as
"none of the five ways this wave tried to distill that information into a
causal, deployable feature survived contact with the full population and
the ensemble." The oracle-vs-legal gap was consistently large where
measured (SST +0.006 to +0.008 on its eligible slice, PCFB PCA +0.011) and
consistently did not survive to the full population or the RT600 ensemble.
That gap — not "is there information," but "why doesn't distillation
survive contact with the whole row population" — is the open question this
wave leaves for whoever picks it up next.
