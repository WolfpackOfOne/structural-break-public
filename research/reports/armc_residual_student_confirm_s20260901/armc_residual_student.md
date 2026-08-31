# Arm-C Residual Student

Date: `2026-08-30T22:06:27`

Nested causal student of the Arm-C residual. No RT ID was allocated and
`RESULTS.csv` was not edited.

## Contract

- Feature bank: `/path/to/workspace/structural-break-wave8/cache/features`
- Student inputs: existing 500 causal columns only
- Target: nested fold-pure Arm-C residual, used as label only
- Population: full dev folds 0-4; no t+h eligibility filter
- OOF arrays: `/path/to/workspace/structural-break-multi-agent-frontier-20260829/research/reports/armc_residual_student_confirm_s20260901/armc_residual_student_oof.npy` and the five per-fold checkpoints beside it. These are `*.npy` and therefore gitignored, so they do not travel with the commit; regenerate with `--train-outer F` for each fold, then `--merge-analyze`.
- T2 purity sentinel: nested passed = `True`, old global-OOF contamination caught = `20/20`

## Main Scores

| score | whole dev | dominant cell | never-break cut | pre-break cut |
|---|---:|---:|---:|---:|
| Arm_B_RT990_raw | 0.611773 | 0.647491 | 0.649119 | 0.642799 |
| Arm_C_RT991_raw | 0.719893 | 0.718588 | 0.714388 | 0.730697 |
| Residual_student_raw | 0.615473 | 0.657979 | 0.659757 | 0.652854 |
| RT600_7stream | 0.625627 | 0.664277 | 0.665431 | 0.660951 |
| RT600_plus_seedclone | 0.625649 | 0.664309 | 0.665411 | 0.661132 |
| RT600_plus_residual_student | 0.627614 | 0.667622 | 0.668890 | 0.663965 |

## Deltas

| contrast | whole dev | dominant cell | never-break cut | pre-break cut |
|---|---:|---:|---:|---:|
| Arm_C_vs_Arm_B | +0.108120 | +0.071097 | +0.065269 | +0.087898 |
| Student_raw_vs_Arm_B | +0.003700 | +0.010488 | +0.010638 | +0.010055 |
| Student_blend_vs_RT600 | +0.001987 | +0.003345 | +0.003460 | +0.003014 |
| Student_blend_vs_seedclone_blend | +0.001965 | +0.003313 | +0.003480 | +0.002833 |

## Per-Fold Stability

Pooled TS-AUC hides fold heterogeneity. These are the same contrasts
resolved per fold.

| contrast | cut | f0 | f1 | f2 | f3 | f4 | positive | mean | sd | t |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Student_raw_vs_Arm_B | whole_dev | -0.009757 | +0.016417 | -0.000703 | +0.016394 | -0.003572 | 2/5 | +0.003756 | 0.012002 | 0.70 |
| Student_raw_vs_Arm_B | dominant_cell | -0.007163 | +0.031675 | +0.006823 | +0.020306 | +0.002385 | 4/5 | +0.010805 | 0.015293 | 1.58 |
| Student_blend_vs_RT600 | whole_dev | +0.000545 | +0.002824 | +0.001429 | +0.004355 | +0.000745 | 5/5 | +0.001979 | 0.001600 | 2.77 |
| Student_blend_vs_RT600 | dominant_cell | +0.001100 | +0.005346 | +0.003056 | +0.005398 | +0.002242 | 5/5 | +0.003428 | 0.001906 | 4.02 |
| Student_blend_vs_seedclone_blend | whole_dev | +0.000234 | +0.002739 | +0.001200 | +0.004217 | +0.001371 | 5/5 | +0.001952 | 0.001550 | 2.82 |
| Student_blend_vs_seedclone_blend | dominant_cell | +0.000357 | +0.005915 | +0.002510 | +0.005086 | +0.003035 | 5/5 | +0.003381 | 0.002200 | 3.44 |

The blend contrasts are 5/5 positive, which is the promotion-relevant
gate. The standalone `Student_raw_vs_Arm_B` contrast is not: its pooled
value is carried by two folds and is negative on others, so it describes
this fit rather than a stable property of the mechanism. The blend gain
is likewise concentrated -- folds 1 and 3 are several times the size of
folds 0 and 4 -- so the mean clears the bar with a small margin relative
to its own fold spread. Treat the confirmation run as load-bearing.

## Scope Of The Gain

- Never-break pair net vs RT-600: `+0.003706`
- Pre-break pair net vs RT-600: `+0.001544`

**This is a never-break-cut gain.** The pre-break pair net is
approximately zero, and in the upstream residualization the
T-orthogonal residual scores *below* Arm B on the pre-break cut --
the residual carries no pre-break signal. The blend's positive
pre-break delta comes from dilution of the seven incumbent streams,
not from new pre-break information. Do not describe this result as a
broad improvement.

- Whole-dev damage rate vs RT-600: `0.014906`

The damage-rate gate is applied to the `dominant_pre_break_only` cut only.
The whole-dev damage rate above sits over the same numeric threshold and
is deliberately not gated; it is shown so the gate's scope is not
mistaken for a claim that damage is bounded everywhere.

## Pair Flow Vs RT-600

| split | pairs | repairs | damage | net | damage rate | net rate |
|---|---:|---:|---:|---:|---:|---:|
| whole_dev | 63194 | 1126 | 942 | 184 | 0.014906 | +0.002912 |
| dominant_cell | 50458 | 820 | 655 | 165 | 0.012981 | +0.003270 |
| dominant_never_break_only | 50452 | 843 | 656 | 187 | 0.013002 | +0.003706 |
| dominant_pre_break_only | 44033 | 645 | 577 | 68 | 0.013104 | +0.001544 |

## Gates

- Pre-break damage rate: `0.013104` (gate `< 0.015000`) -> `True`
- Never-break net rate vs RT-600: `+0.003706`
- Whole-dev marginal vs seed-clone blend: `+0.001965`
- Whole-dev gain vs RT-600: `+0.001987`
- Dominant-cell gain vs RT-600: `+0.003345`
- Student raw retention of oracle residual cell lift: `0.273811` -- **not a like-for-like ratio.** The denominator is the oracle residual on the global (fold-contaminated) RT-991 over the dominant cell; the numerator is the nested fold-pure student on full-population labels. Indicative magnitude only; do not quote as a retention rate.

## Correlation

- Student raw vs Arm B, within-t dominant-cell rho: `0.605488`
- Student raw vs RT-600, within-t dominant-cell rho: `0.693949`

## RT-1257 Complementarity

Uses original dev OOF arrays RT-1254/RT-1255 from the CatBoost specialist research directory, not the deployment final10k artifacts.

| score | mean whole-dev | pooled whole-dev | dominant cell | never-break cut | pre-break cut |
|---|---:|---:|---:|---:|---:|
| RT600_7stream | 0.625811 | 0.625627 | 0.664277 | 0.665431 | 0.660951 |
| RT1257_catboost_hybrid | 0.627838 | 0.627578 | 0.666658 | 0.667492 | 0.664254 |
| RT1257_plus_residual_student | 0.629254 | 0.628998 | 0.669369 | 0.670353 | 0.666535 |
| clone_plus_seed_control | 0.625517 | 0.625310 | 0.663534 | 0.664477 | 0.660816 |

| contrast | mean whole-dev | pooled whole-dev | dominant cell |
|---|---:|---:|---:|
| RT1257_plus_student_vs_RT1257 | +0.001416 | +0.001420 | +0.002711 |
| RT1257_plus_student_vs_clone_plus_seed | +0.003737 | +0.003688 | +0.005835 |
| RT1257_plus_student_vs_RT600 | +0.003442 | +0.003371 | +0.005092 |

Pair flow of `RT1257 + residual_student` vs `RT1257`:

| split | pairs | repairs | damage | net | damage rate | net rate |
|---|---:|---:|---:|---:|---:|---:|
| whole_dev | 63194 | 1124 | 1009 | 115 | 0.015967 | +0.001820 |
| dominant_cell | 50458 | 781 | 654 | 127 | 0.012961 | +0.002517 |
| dominant_never_break_only | 50452 | 845 | 664 | 181 | 0.013161 | +0.003588 |
| dominant_pre_break_only | 44033 | 678 | 570 | 108 | 0.012945 | +0.002453 |

## Verdict

PROMOTE for confirmation only: the nested 500-feature residual student beats the seed-clone blend marginally, has positive never-break pair net, and clears the pre-break damage gate. The light RT-1257 complementarity check is also positive: +student is +0.001416 mean whole-dev TS-AUC over the CatBoost hybrid.
