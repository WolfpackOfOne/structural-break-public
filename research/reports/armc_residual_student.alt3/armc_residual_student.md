# Arm-C Residual Student

Date: `2026-09-01T17:35:09`

Nested causal student of the Arm-C residual. No RT ID was allocated and
`RESULTS.csv` was not edited.

## Contract

- Feature bank: `/path/to/workspace/structural-break-rt1320-promotion-2026/cache/features`
- Student inputs: existing 500 causal columns only
- Target: nested fold-pure Arm-C residual, used as label only
- Population: full dev folds 0-4; no t+h eligibility filter
- OOF arrays: `/path/to/workspace/structural-break-rt1320-promotion-2026/research/reports/armc_residual_student.alt3/armc_residual_student_oof.npy` and the five per-fold checkpoints beside it. These are `*.npy` and therefore gitignored, so they do not travel with the commit; regenerate with `--train-outer F` for each fold, then `--merge-analyze`.
- T2 purity sentinel: nested passed = `True`, old global-OOF contamination caught = `20/20`

## Main Scores

| score | whole dev | dominant cell | never-break cut | pre-break cut |
|---|---:|---:|---:|---:|
| Residual_student_raw | 0.614119 | 0.655671 | 0.656873 | 0.652207 |
| RT600_7stream | 0.619665 | 0.655131 | 0.657397 | 0.648599 |
| RT600_plus_seedclone | 0.619586 | 0.655104 | 0.657518 | 0.648147 |
| RT600_plus_residual_student | 0.622066 | 0.659083 | 0.661185 | 0.653020 |

## Deltas

| contrast | whole dev | dominant cell | never-break cut | pre-break cut |
|---|---:|---:|---:|---:|
| Student_blend_vs_RT600 | +0.002400 | +0.003952 | +0.003789 | +0.004421 |
| Student_blend_vs_seedclone_blend | +0.002480 | +0.003978 | +0.003668 | +0.004873 |

## Per-Fold Stability

Pooled TS-AUC hides fold heterogeneity. These are the same contrasts
resolved per fold.

| contrast | cut | f0 | f1 | f2 | f3 | f4 | positive | mean | sd | t |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Student_blend_vs_RT600 | whole_dev | +0.001426 | +0.002825 | +0.003344 | +0.003772 | +0.000798 | 5/5 | +0.002433 | 0.001271 | 4.28 |
| Student_blend_vs_RT600 | dominant_cell | +0.001878 | +0.005107 | +0.005561 | +0.005695 | +0.002063 | 5/5 | +0.004061 | 0.001922 | 4.72 |
| Student_blend_vs_seedclone_blend | whole_dev | +0.001588 | +0.002964 | +0.003674 | +0.003849 | +0.000433 | 5/5 | +0.002502 | 0.001459 | 3.83 |
| Student_blend_vs_seedclone_blend | dominant_cell | +0.002299 | +0.004637 | +0.006208 | +0.005505 | +0.001702 | 5/5 | +0.004070 | 0.001981 | 4.59 |

Read from this run's own per-fold numbers:
- Blend vs seed-clone blend: 5/5 folds positive, mean +0.002502, sd 0.001459, t 3.83.
- Largest fold effect is 8.9x the smallest in magnitude; the wider that ratio, the more the mean rests on a few folds.
- The blend contrasts, not the standalone one, are the promotion-relevant gate.

## Scope Of The Gain

- Never-break pair net vs RT-600: `+0.003944`
- Pre-break pair net vs RT-600: `+0.003134`

**The gain appears on both cuts here**: never-break pair net
+0.003944 against pre-break +0.003134
(ratio 1.3x). That differs from the canonical run, where the
pre-break net was approximately zero, and the difference is itself a
result worth reporting rather than smoothing over.

- Whole-dev damage rate vs RT-600: `0.015998`

The damage-rate gate is applied to the `dominant_pre_break_only` cut only.
The whole-dev damage rate above sits over the same numeric threshold and
is deliberately not gated; it is shown so the gate's scope is not
mistaken for a claim that damage is bounded everywhere.

## Pair Flow Vs RT-600

| split | pairs | repairs | damage | net | damage rate | net rate |
|---|---:|---:|---:|---:|---:|---:|
| whole_dev | 63194 | 1104 | 1011 | 93 | 0.015998 | +0.001472 |
| dominant_cell | 50458 | 811 | 654 | 157 | 0.012961 | +0.003111 |
| dominant_never_break_only | 50452 | 824 | 625 | 199 | 0.012388 | +0.003944 |
| dominant_pre_break_only | 44033 | 729 | 591 | 138 | 0.013422 | +0.003134 |

## Gates

- Pre-break damage rate: `0.013422` (gate `< 0.015000`) -> `True`
- Never-break net rate vs RT-600: `+0.003944`
- Whole-dev marginal vs seed-clone blend: `+0.002480`
- Whole-dev gain vs RT-600: `+0.002400`
- Dominant-cell gain vs RT-600: `+0.003952`
- Student raw retention of oracle residual cell lift: `n/a` -- **not a like-for-like ratio.** The denominator is the oracle residual on the global (fold-contaminated) RT-991 over the dominant cell; the numerator is the nested fold-pure student on full-population labels. Indicative magnitude only; do not quote as a retention rate.

## Correlation

- Student raw vs RT-600, within-t dominant-cell rho: `0.695727`

## Verdict

PROMOTE for confirmation only: the nested 500-feature residual student beats the seed-clone blend marginally, has positive never-break pair net, and clears the pre-break damage gate.
