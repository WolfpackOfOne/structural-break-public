# Arm-C Residual Student

Date: `2026-08-30T13:04:58`

Nested causal student of the Arm-C residual. No RT ID was allocated and
`RESULTS.csv` was not edited.

## Contract

- Feature bank: `/path/to/workspace/structural-break-wave8/cache/features`
- Student inputs: existing 500 causal columns only
- Target: nested fold-pure Arm-C residual, used as label only
- Population: full dev folds 0-4; no t+h eligibility filter
- OOF arrays: `/path/to/workspace/structural-break-multi-agent-frontier-20260829/research/reports/armc_residual_student/armc_residual_student_oof.npy` and the five per-fold checkpoints beside it. These are `*.npy` and therefore gitignored, so they do not travel with the commit; regenerate with `--train-outer F` for each fold, then `--merge-analyze`.
- T2 purity sentinel: nested passed = `True`, old global-OOF contamination caught = `20/20`

## Main Scores

| score | whole dev | dominant cell | never-break cut | pre-break cut |
|---|---:|---:|---:|---:|
| Arm_B_RT990_raw | 0.611773 | 0.647491 | 0.649119 | 0.642799 |
| Arm_C_RT991_raw | 0.719893 | 0.718588 | 0.714388 | 0.730697 |
| Residual_student_raw | 0.615486 | 0.656890 | 0.658733 | 0.651577 |
| RT600_7stream | 0.625627 | 0.664277 | 0.665431 | 0.660951 |
| RT600_plus_seedclone | 0.625649 | 0.664309 | 0.665411 | 0.661132 |
| RT600_plus_residual_student | 0.627600 | 0.667417 | 0.668697 | 0.663728 |

## Deltas

| contrast | whole dev | dominant cell | never-break cut | pre-break cut |
|---|---:|---:|---:|---:|
| Arm_C_vs_Arm_B | +0.108120 | +0.071097 | +0.065269 | +0.087898 |
| Student_raw_vs_Arm_B | +0.003713 | +0.009398 | +0.009614 | +0.008778 |
| Student_blend_vs_RT600 | +0.001973 | +0.003140 | +0.003266 | +0.002777 |
| Student_blend_vs_seedclone_blend | +0.001951 | +0.003108 | +0.003286 | +0.002596 |

## Per-Fold Stability

Pooled TS-AUC hides fold heterogeneity. These are the same contrasts
resolved per fold.

| contrast | cut | f0 | f1 | f2 | f3 | f4 | positive | mean | sd | t |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Student_raw_vs_Arm_B | whole_dev | -0.009894 | +0.019941 | -0.001370 | +0.015058 | -0.005354 | 2/5 | +0.003676 | 0.013089 | 0.63 |
| Student_raw_vs_Arm_B | dominant_cell | -0.004847 | +0.033518 | +0.004015 | +0.017451 | -0.001718 | 3/5 | +0.009684 | 0.015825 | 1.37 |
| Student_blend_vs_RT600 | whole_dev | +0.000535 | +0.003261 | +0.001313 | +0.004277 | +0.000439 | 5/5 | +0.001965 | 0.001719 | 2.56 |
| Student_blend_vs_RT600 | dominant_cell | +0.001531 | +0.005520 | +0.002508 | +0.005118 | +0.001424 | 5/5 | +0.003220 | 0.001967 | 3.66 |
| Student_blend_vs_seedclone_blend | whole_dev | +0.000225 | +0.003177 | +0.001084 | +0.004140 | +0.001065 | 5/5 | +0.001938 | 0.001644 | 2.64 |
| Student_blend_vs_seedclone_blend | dominant_cell | +0.000788 | +0.006089 | +0.001962 | +0.004806 | +0.002218 | 5/5 | +0.003173 | 0.002193 | 3.23 |

The blend contrasts are 5/5 positive, which is the promotion-relevant
gate. The standalone `Student_raw_vs_Arm_B` contrast is not: its pooled
value is carried by two folds and is negative on others, so it describes
this fit rather than a stable property of the mechanism. The blend gain
is likewise concentrated -- folds 1 and 3 are several times the size of
folds 0 and 4 -- so the mean clears the bar with a small margin relative
to its own fold spread. Treat the confirmation run as load-bearing.

## Scope Of The Gain

- Never-break pair net vs RT-600: `+0.003409`
- Pre-break pair net vs RT-600: `+0.000363`

**This is a never-break-cut gain.** The pre-break pair net is
approximately zero, and in the upstream residualization the
T-orthogonal residual scores *below* Arm B on the pre-break cut --
the residual carries no pre-break signal. The blend's positive
pre-break delta comes from dilution of the seven incumbent streams,
not from new pre-break information. Do not describe this result as a
broad improvement.

- Whole-dev damage rate vs RT-600: `0.015175`

The damage-rate gate is applied to the `dominant_pre_break_only` cut only.
The whole-dev damage rate above sits over the same numeric threshold and
is deliberately not gated; it is shown so the gate's scope is not
mistaken for a claim that damage is bounded everywhere.

## Pair Flow Vs RT-600

| split | pairs | repairs | damage | net | damage rate | net rate |
|---|---:|---:|---:|---:|---:|---:|
| whole_dev | 63194 | 1139 | 959 | 180 | 0.015175 | +0.002848 |
| dominant_cell | 50458 | 790 | 657 | 133 | 0.013021 | +0.002636 |
| dominant_never_break_only | 50452 | 837 | 665 | 172 | 0.013181 | +0.003409 |
| dominant_pre_break_only | 44033 | 618 | 602 | 16 | 0.013672 | +0.000363 |

## Gates

- Pre-break damage rate: `0.013672` (gate `< 0.015000`) -> `True`
- Never-break net rate vs RT-600: `+0.003409`
- Whole-dev marginal vs seed-clone blend: `+0.001951`
- Whole-dev gain vs RT-600: `+0.001973`
- Dominant-cell gain vs RT-600: `+0.003140`
- Student raw retention of oracle residual cell lift: `0.245374` -- **not a like-for-like ratio.** The denominator is the oracle residual on the global (fold-contaminated) RT-991 over the dominant cell; the numerator is the nested fold-pure student on full-population labels. Indicative magnitude only; do not quote as a retention rate.

## Correlation

- Student raw vs Arm B, within-t dominant-cell rho: `0.605144`
- Student raw vs RT-600, within-t dominant-cell rho: `0.694757`

## RT-1257 Complementarity

Uses original dev OOF arrays RT-1254/RT-1255 from the CatBoost specialist research directory, not the deployment final10k artifacts.

| score | mean whole-dev | pooled whole-dev | dominant cell | never-break cut | pre-break cut |
|---|---:|---:|---:|---:|---:|
| RT600_7stream | 0.625811 | 0.625627 | 0.664277 | 0.665431 | 0.660951 |
| RT1257_catboost_hybrid | 0.627838 | 0.627578 | 0.666658 | 0.667492 | 0.664254 |
| RT1257_plus_residual_student | 0.629234 | 0.628981 | 0.669163 | 0.670153 | 0.666308 |
| clone_plus_seed_control | 0.625517 | 0.625310 | 0.663534 | 0.664477 | 0.660816 |

| contrast | mean whole-dev | pooled whole-dev | dominant cell |
|---|---:|---:|---:|
| RT1257_plus_student_vs_RT1257 | +0.001396 | +0.001403 | +0.002505 |
| RT1257_plus_student_vs_clone_plus_seed | +0.003717 | +0.003671 | +0.005628 |
| RT1257_plus_student_vs_RT600 | +0.003422 | +0.003355 | +0.004886 |

Pair flow of `RT1257 + residual_student` vs `RT1257`:

| split | pairs | repairs | damage | net | damage rate | net rate |
|---|---:|---:|---:|---:|---:|---:|
| whole_dev | 63194 | 1098 | 1016 | 82 | 0.016077 | +0.001298 |
| dominant_cell | 50458 | 811 | 650 | 161 | 0.012882 | +0.003191 |
| dominant_never_break_only | 50452 | 836 | 671 | 165 | 0.013300 | +0.003270 |
| dominant_pre_break_only | 44033 | 640 | 589 | 51 | 0.013376 | +0.001158 |

## Verdict

PROMOTE for confirmation only: the nested 500-feature residual student beats the seed-clone blend marginally, has positive never-break pair net, and clears the pre-break damage gate. The light RT-1257 complementarity check is also positive: +student is +0.001396 mean whole-dev TS-AUC over the CatBoost hybrid.
