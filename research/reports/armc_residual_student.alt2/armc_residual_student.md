# Arm-C Residual Student

Date: `2026-09-01T12:07:39`

Nested causal student of the Arm-C residual. No RT ID was allocated and
`RESULTS.csv` was not edited.

## Contract

- Feature bank: `/path/to/workspace/structural-break-rt1320-promotion-2026/cache/features`
- Student inputs: existing 500 causal columns only
- Target: nested fold-pure Arm-C residual, used as label only
- Population: full dev folds 0-4; no t+h eligibility filter
- OOF arrays: `/path/to/workspace/structural-break-rt1320-promotion-2026/research/reports/armc_residual_student.alt2/armc_residual_student_oof.npy` and the five per-fold checkpoints beside it. These are `*.npy` and therefore gitignored, so they do not travel with the commit; regenerate with `--train-outer F` for each fold, then `--merge-analyze`.
- T2 purity sentinel: nested passed = `True`, old global-OOF contamination caught = `20/20`

## Main Scores

| score | whole dev | dominant cell | never-break cut | pre-break cut |
|---|---:|---:|---:|---:|
| Residual_student_raw | 0.615760 | 0.655338 | 0.656496 | 0.652000 |
| RT600_7stream | 0.626935 | 0.664212 | 0.665343 | 0.660952 |
| RT600_plus_seedclone | 0.626726 | 0.664145 | 0.665321 | 0.660756 |
| RT600_plus_residual_student | 0.628907 | 0.667101 | 0.668222 | 0.663868 |

## Deltas

| contrast | whole dev | dominant cell | never-break cut | pre-break cut |
|---|---:|---:|---:|---:|
| Student_blend_vs_RT600 | +0.001972 | +0.002888 | +0.002879 | +0.002916 |
| Student_blend_vs_seedclone_blend | +0.002181 | +0.002955 | +0.002901 | +0.003111 |

## Per-Fold Stability

Pooled TS-AUC hides fold heterogeneity. These are the same contrasts
resolved per fold.

| contrast | cut | f0 | f1 | f2 | f3 | f4 | positive | mean | sd | t |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Student_blend_vs_RT600 | whole_dev | +0.001304 | +0.003970 | +0.000712 | +0.001750 | +0.002333 | 5/5 | +0.002014 | 0.001245 | 3.62 |
| Student_blend_vs_RT600 | dominant_cell | +0.002533 | +0.005017 | +0.001664 | +0.002827 | +0.002680 | 5/5 | +0.002944 | 0.001244 | 5.29 |
| Student_blend_vs_seedclone_blend | whole_dev | +0.001661 | +0.004201 | +0.000652 | +0.001919 | +0.002777 | 5/5 | +0.002242 | 0.001332 | 3.76 |
| Student_blend_vs_seedclone_blend | dominant_cell | +0.002993 | +0.005407 | +0.001320 | +0.002354 | +0.003144 | 5/5 | +0.003043 | 0.001504 | 4.53 |

Read from this run's own per-fold numbers:
- Blend vs seed-clone blend: 5/5 folds positive, mean +0.002242, sd 0.001332, t 3.76.
- Largest fold effect is 6.4x the smallest in magnitude; the wider that ratio, the more the mean rests on a few folds.
- The blend contrasts, not the standalone one, are the promotion-relevant gate.

## Scope Of The Gain

- Never-break pair net vs RT-600: `+0.003607`
- Pre-break pair net vs RT-600: `+0.001272`

**The gain appears on both cuts here**: never-break pair net
+0.003607 against pre-break +0.001272
(ratio 2.8x). That differs from the canonical run, where the
pre-break net was approximately zero, and the difference is itself a
result worth reporting rather than smoothing over.

- Whole-dev damage rate vs RT-600: `0.015603`

The damage-rate gate is applied to the `dominant_pre_break_only` cut only.
The whole-dev damage rate above sits over the same numeric threshold and
is deliberately not gated; it is shown so the gate's scope is not
mistaken for a claim that damage is bounded everywhere.

## Pair Flow Vs RT-600

| split | pairs | repairs | damage | net | damage rate | net rate |
|---|---:|---:|---:|---:|---:|---:|
| whole_dev | 63194 | 1216 | 986 | 230 | 0.015603 | +0.003640 |
| dominant_cell | 50458 | 796 | 637 | 159 | 0.012624 | +0.003151 |
| dominant_never_break_only | 50452 | 840 | 658 | 182 | 0.013042 | +0.003607 |
| dominant_pre_break_only | 44033 | 656 | 600 | 56 | 0.013626 | +0.001272 |

## Gates

- Pre-break damage rate: `0.013626` (gate `< 0.015000`) -> `True`
- Never-break net rate vs RT-600: `+0.003607`
- Whole-dev marginal vs seed-clone blend: `+0.002181`
- Whole-dev gain vs RT-600: `+0.001972`
- Dominant-cell gain vs RT-600: `+0.002888`
- Student raw retention of oracle residual cell lift: `n/a` -- **not a like-for-like ratio.** The denominator is the oracle residual on the global (fold-contaminated) RT-991 over the dominant cell; the numerator is the nested fold-pure student on full-population labels. Indicative magnitude only; do not quote as a retention rate.

## Correlation

- Student raw vs RT-600, within-t dominant-cell rho: `0.691290`

## RT-1257 Complementarity

Uses original dev OOF arrays RT-1254/RT-1255 from the CatBoost specialist research directory, not the deployment final10k artifacts.

| score | mean whole-dev | pooled whole-dev | dominant cell | never-break cut | pre-break cut |
|---|---:|---:|---:|---:|---:|
| RT600_7stream | 0.627557 | 0.626935 | 0.664212 | 0.665343 | 0.660952 |
| RT1257_catboost_hybrid | 0.629943 | 0.629174 | 0.667216 | 0.667988 | 0.664989 |
| RT1257_plus_seedclone | 0.629746 | 0.628990 | 0.667029 | 0.667781 | 0.664864 |
| RT1257_plus_residual_student | 0.631228 | 0.630437 | 0.669275 | 0.670099 | 0.666900 |
| all_clone_control | 0.627863 | 0.627182 | 0.664581 | 0.665583 | 0.661693 |

| contrast | mean whole-dev | pooled whole-dev | dominant cell |
|---|---:|---:|---:|
| RT1257_plus_student_vs_RT1257 | +0.001285 | +0.001263 | +0.002060 |
| **PRIMARY** RT1257_plus_student_vs_RT1257_plus_seedclone | +0.001482 | +0.001447 | +0.002246 |
| RT1257_plus_student_vs_all_clone_control (NOT an E1) | +0.003366 | +0.003255 | +0.004694 |
| RT1257_plus_student_vs_RT600 | +0.003671 | +0.003502 | +0.005063 |


PRIMARY endpoint under PROTOCOL_CHAMPION_2026 is E2-E1 where E1 is RT1257_plus_seedclone -- RT-1257 with one matched exchangeable clone added, exactly one change from E0. all_clone_control is NOT an E1: it clones the CAT-300 and CAT-413 members as well as adding a clone, so it sits at RT-600 grade and a delta against it re-credits the two CatBoost slot swaps to the candidate. Reports before 2026-08-31 quoted that delta as if it were a champion-relative marginal; it is not. See research/scripts/armc_e2_e1_addition_contract.py.

Pair flow of `RT1257 + residual_student` vs `RT1257`:

| split | pairs | repairs | damage | net | damage rate | net rate |
|---|---:|---:|---:|---:|---:|---:|
| whole_dev | 63194 | 1130 | 1025 | 105 | 0.016220 | +0.001662 |
| dominant_cell | 50458 | 802 | 658 | 144 | 0.013041 | +0.002854 |
| dominant_never_break_only | 50452 | 834 | 683 | 151 | 0.013538 | +0.002993 |
| dominant_pre_break_only | 44033 | 617 | 605 | 12 | 0.013740 | +0.000273 |

## Verdict

PROMOTE for confirmation only: the nested 500-feature residual student beats the seed-clone blend marginally, has positive never-break pair net, and clears the pre-break damage gate. The light RT-1257 complementarity check is also positive: +student is +0.001285 mean whole-dev TS-AUC over the CatBoost hybrid.
