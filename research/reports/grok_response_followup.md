# Grok Response Follow-Up

Date: 2026-08-30

Branch: `grok-response-issues-20260830`

This records the follow-up work for
`research/multi_agent_frontier_20260829/responses/agent_05_grok46.md`.

## Corrections Applied

| Grok issue | Status |
|---|---|
| `RT-991.npy` was treated as missing or unverified | Addressed. It exists in the sibling Wave 8 OOF artifacts and is now used by `research/scripts/armc_residualization.py`. |
| `STATE_OF_RESEARCH.md` still presented `RT-131` as current champion | Addressed. It is now marked archival; `STATUS.md` is the live current-state file. |
| `RT-600 pooled OOF ~= 0.63828` mixed fold-0 with whole-dev estimates | Addressed. `STATUS.md` and the prompt now distinguish fold-0 `0.638276`, mean OOF `0.625811342`, and pooled dev `0.625626926`. |
| CAT-413/CAT-300 fold concentration wording was contrast-dependent | Addressed in the evidence prompt as an estimand-specific fixed E2-E0 statement, not a universal fold claim. |
| W7-D3R CASE 2 was treated as settled without residualizing Arm C against endpoint/horizon variables | Addressed and overturned as a settled claim. CASE 2 is now only partially right. |

## Score-Relevant Work Completed

### 1. Arm-C Horizon Residualization

Report: `research/reports/armc_residualization.md`

Script: `research/scripts/armc_residualization.py`

The cross-fitted within-`t` projection of `logit(RT-991)` on `{n_online,
n_online-t, t/n_online}` separated the endpoint/horizon component from a real
residual:

| score | dominant cell | never-break cut | pre-break cut |
|---|---:|---:|---:|
| Arm B (`RT-990`) | 0.647491 | 0.649119 | 0.642799 |
| Arm C (`RT-991`) | 0.718588 | 0.714388 | 0.730697 |
| Horizon projection | 0.599887 | 0.564864 | 0.700855 |
| T-orthogonal residual | 0.685794 | 0.703074 | 0.635977 |

Residual lift vs Arm B is `+0.038303` on the dominant cell; within-`t` rank rho
vs Arm B is `0.457016`. The shuffled-horizon control is clean. This clears
Grok's residual-student gate, which is scoped to the dominant cell.

Resolved by cut, that lift is **entirely a never-break-cut effect**:

| cut | residual lift vs Arm B |
|---|---:|
| dominant cell (gated) | `+0.038303` |
| never-break-only | `+0.053955` |
| pre-break-only | `-0.006822` |

On the pre-break cut the residual scores *below* Arm B. Arm C's large raw
pre-break lift (`+0.087898`) is endpoint/horizon information, not transferable
structure — the horizon projection alone scores `0.700855` there. Kimi's
independent length-matched decomposition reaches the same conclusion by a
different route (pre-break lift survival `0.106` at K=10, negative on two of
five folds), so this is a two-method result, not a single-specification one.

### 2. Nested Causal Student Of The Residual

Report: `research/reports/armc_residual_student/armc_residual_student.md`

Script: `research/scripts/armc_residual_student.py`

Contract:

- Existing 500 causal features only.
- Full dev population, no `t+h` eligibility filter.
- Nested fold-pure Arm-C teacher checkpoints as labels only.
- T2-style purity sentinel passed: corrected nested scheme clean, old global
  OOF scheme caught as contaminated in `20/20` checks.
- No RT ID, no `RESULTS.csv` edit, no lockbox/test touch.

Main result:

| score | whole dev | dominant cell | never-break cut | pre-break cut |
|---|---:|---:|---:|---:|
| Residual student raw | 0.615486 | 0.656890 | 0.658733 | 0.651577 |
| `RT600 + residual_student` | 0.627600 | 0.667417 | 0.668697 | 0.663728 |

Key deltas:

- Raw student dominant-cell lift vs Arm B: `+0.009398` pooled — but see the
  per-fold caveat below; this contrast is not stable.
- `RT600 + residual_student` pooled whole-dev gain vs RT600: `+0.001973`.
- Marginal vs seed-clone blend: `+0.001951`.
- Dominant-cell gain vs RT600: `+0.003140`.
- Never-break net rate vs RT600: `+0.003409`.
- Pre-break damage rate vs RT600: `0.013672`, passing the hard `<0.015` gate
  on the dominant pre-break cut. The whole-dev damage rate is `0.015175`,
  above the same number; the gate is scoped to the pre-break cut and does not
  bound damage everywhere.
- Student retention of the oracle residual cell lift is reported as `0.245374`
  but is **not a like-for-like ratio** — the denominator uses the global
  (fold-contaminated) RT-991 residual, the numerator the nested fold-pure
  student. Indicative magnitude only.

Scope of the gain: the pre-break pair net is `+0.000363`, essentially zero,
against `+0.003409` never-break. Consistent with the residualization above,
**this is a never-break-cut gain**, not a broad improvement, and the blend's
positive pre-break delta is stream dilution rather than new pre-break
information.

Per-fold stability, the promotion-relevant view:

| contrast | f0 | f1 | f2 | f3 | f4 | positive | mean | t |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Student_raw vs Arm B (whole dev) | `-0.009894` | `+0.019941` | `-0.001370` | `+0.015058` | `-0.005354` | 2/5 | `+0.003676` | 0.63 |
| Student_blend vs RT600 (whole dev) | `+0.000535` | `+0.003261` | `+0.001313` | `+0.004277` | `+0.000439` | 5/5 | `+0.001965` | 2.56 |
| Student_blend vs clone (whole dev) | `+0.000225` | `+0.003177` | `+0.001084` | `+0.004140` | `+0.001065` | 5/5 | `+0.001938` | 2.64 |

The blend contrasts are 5/5 positive and clear the repo's `+0.0015`
marginal-vs-clone bar. The standalone student-vs-Arm-B contrast is not stable —
2/5 positive, `t = 0.63` — so its pooled value describes this fit, not the
mechanism. The blend gain is also concentrated: folds 1 and 3 are several times
folds 0 and 4, so the mean clears the bar with a modest margin relative to its
own fold spread. The confirmation run is load-bearing.

Light RT-1257 complementarity check, using original dev OOF arrays for
`RT-1254`/`RT-1255`, is also positive:

- `RT1257 + residual_student` minus `RT-1257`: `+0.001396` mean whole-dev
  TS-AUC, `+0.001403` pooled whole-dev TS-AUC.
- `RT1257 + residual_student` minus `RT-600`: `+0.003422` mean whole-dev
  TS-AUC.
- Pair flow vs RT-1257 remains positive in the targeted cuts, including
  never-break net rate `+0.003270` and pre-break damage rate `0.013376`.

Verdict: promote for confirmation only. This is a score candidate, not a
production change.

The OOF arrays behind these numbers (`armc_residual_student_oof.npy` and the
five per-fold checkpoints) are `*.npy` and gitignored, so they do not travel
with the commit. Regenerate with `--train-outer F` for each fold, then
`--merge-analyze`; the analysis step is deterministic and reproduces every
figure above.

## Cross-Branch Note

The sibling branch `kimi-response-issues-20260830` independently built the same
mechanism: `wave7_epod.py` (`RT-1258`) is also "strip the endpoint component
from the Arm-C oracle, then distill." The two lanes ran concurrently on the same
machine — this branch's `gate.log` records blocking on the Kimi PIDs at 09:19
and 09:24.

They corroborate each other on the decomposition: this branch's regression
residualization puts the endpoint-orthogonal Arm-C dominant-cell lift at
`+0.038303`, Kimi's length-matched pairing at `+0.042342`, and both find the
pre-break lift is an artifact. That agreement across two unrelated methods is
the strongest result on either branch.

That comparison has now been run. The Kimi candidate (since renamed
`W7EPOD-01`; its `RT-1258` label was never allocated) was trained on all five
outer folds and scored through an identical gate — cross-fitted `SCDF_NSEEN`
per stream, equal-weight mean, E1 = seven specialists + seed clone, E2 = seven
specialists + candidate — resolving against the same E0, `0.625811`:

| candidate | standalone whole-dev | marginal vs clone | positive folds | pre-break pair net | verdict |
|---|---:|---:|---:|---:|---|
| `W7EPOD-01` (Kimi branch) | `0.619053` | `+0.000514` | 4/5 | `-46` | KILL |
| Arm-C residual student (this branch) | `0.615486` | `+0.001938` | 5/5 | `+16` | PROMOTE for confirmation |

The ordering **inverts** between standalone and ensemble: `W7EPOD-01` is the
better standalone model by `+0.003567` whole-dev and the worse ensemble
contribution by `-0.001424`, falling below the `+0.0010` kill threshold. Only
this branch's residual student goes forward.

The reason is the point of the whole lane: EPOD purges the endpoint component
from the teacher and distils what remains, which leaves a model that still
ranks much like the incumbents; the residual student distils the *orthogonal
residual*, which is weaker alone but carries what the blend does not already
have. Orthogonality to the incumbent beats standalone strength here — the same
pattern as Wave 7 T2 (`+0.00943` standalone, `+0.00024` at ensemble). Full
record: `agent_02_kimi_audit.md` §D on the sibling branch.

## Remaining Grok Mechanisms

| mechanism | Status | Next action |
|---|---|---|
| M3 frozen cohort two-null atlas | Still open and now first in queue. It is the next lowest-cost score-relevant check because it uses existing feature cache plus OOF arrays before any training. | Run the cheap cross-fitted atlas disagreement probe, then train only if it beats the deranged/control screen and keeps pre-break damage `<0.015`. |
| M4 explicit-duration HSMM | Still open. Lower upside than M3 but high enough probability to probe. | First run the historical excursion survival-vs-geometric falsification; do not train the filter if historical dwell is geometric or redundant with `ab_fast`. |
| M2 per-series delay-cloud predictive null | Still open. Requires new causal feature computation and should follow M3/M4 probes. | Start with the 400-row occupancy-disagreement falsification before building a stream. |
| M5 per-series C2ST | Still open but lowest priority. | Only run after M3/M4/M2 unless a cheap length-matched linear C2ST probe beats Wasserstein by the declared margin. |
| Moonshot simulator posterior | Not a next branch item. | Only worth the one-hour generator-transfer kill gate after the concrete prefix/reference-measure lanes are exhausted. |

Operational rule for the remaining large jobs: check machine load before launch,
then retry every 15 minutes until idle enough to run. Completed residual-student
fold checkpoints are already saved and resumable.
