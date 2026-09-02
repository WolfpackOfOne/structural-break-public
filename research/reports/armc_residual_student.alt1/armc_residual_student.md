# Arm-C Residual Student

Date: `2026-08-31T13:07:53`

Nested causal student of the Arm-C residual. No RT ID was allocated and
`RESULTS.csv` was not edited.

## Contract

- Feature bank: `/path/to/workspace/structural-break-rt1320-promotion-2026/cache/features`
- Student inputs: existing 500 causal columns only
- Target: nested fold-pure Arm-C residual, used as label only
- Population: full dev folds 0-4; no t+h eligibility filter
- OOF arrays: `/path/to/workspace/structural-break-rt1320-promotion-2026/research/reports/armc_residual_student.alt1/armc_residual_student_oof.npy` and the five per-fold checkpoints beside it. These are `*.npy` and therefore gitignored, so they do not travel with the commit; regenerate with `--train-outer F` for each fold, then `--merge-analyze`.
- T2 purity sentinel: nested passed = `True`, old global-OOF contamination caught = `20/20`

## Main Scores

| score | whole dev | dominant cell | never-break cut | pre-break cut |
|---|---:|---:|---:|---:|
| Residual_student_raw | 0.612760 | 0.653051 | 0.654210 | 0.649709 |
| RT600_7stream | 0.617427 | 0.654145 | 0.654381 | 0.653462 |
| RT600_plus_seedclone | 0.617336 | 0.654614 | 0.654987 | 0.653538 |
| RT600_plus_residual_student | 0.620030 | 0.657842 | 0.658164 | 0.656912 |

## Deltas

| contrast | whole dev | dominant cell | never-break cut | pre-break cut |
|---|---:|---:|---:|---:|
| Student_blend_vs_RT600 | +0.002603 | +0.003697 | +0.003783 | +0.003450 |
| Student_blend_vs_seedclone_blend | +0.002694 | +0.003227 | +0.003177 | +0.003374 |

## Per-Fold Stability

Pooled TS-AUC hides fold heterogeneity. These are the same contrasts
resolved per fold.

| contrast | cut | f0 | f1 | f2 | f3 | f4 | positive | mean | sd | t |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Student_blend_vs_RT600 | whole_dev | +0.002922 | +0.003944 | +0.002124 | +0.001192 | +0.002938 | 5/5 | +0.002624 | 0.001028 | 5.71 |
| Student_blend_vs_RT600 | dominant_cell | +0.004419 | +0.005649 | +0.003997 | +0.001742 | +0.002658 | 5/5 | +0.003693 | 0.001526 | 5.41 |
| Student_blend_vs_seedclone_blend | whole_dev | +0.003148 | +0.003851 | +0.002050 | +0.001944 | +0.002520 | 5/5 | +0.002703 | 0.000799 | 7.57 |
| Student_blend_vs_seedclone_blend | dominant_cell | +0.004284 | +0.004593 | +0.003823 | +0.001459 | +0.001919 | 5/5 | +0.003216 | 0.001429 | 5.03 |

Read from this run's own per-fold numbers:
- Blend vs seed-clone blend: 5/5 folds positive, mean +0.002703, sd 0.000799, t 7.57.
- Largest fold effect is 2.0x the smallest in magnitude; the wider that ratio, the more the mean rests on a few folds.
- The blend contrasts, not the standalone one, are the promotion-relevant gate.

## Scope Of The Gain

- Never-break pair net vs RT-600: `+0.004519`
- Pre-break pair net vs RT-600: `+0.003452`

**The gain appears on both cuts here**: never-break pair net
+0.004519 against pre-break +0.003452
(ratio 1.3x). That differs from the canonical run, where the
pre-break net was approximately zero, and the difference is itself a
result worth reporting rather than smoothing over.

- Whole-dev damage rate vs RT-600: `0.015191`

The damage-rate gate is applied to the `dominant_pre_break_only` cut only.
The whole-dev damage rate above sits over the same numeric threshold and
is deliberately not gated; it is shown so the gate's scope is not
mistaken for a claim that damage is bounded everywhere.

## Pair Flow Vs RT-600

| split | pairs | repairs | damage | net | damage rate | net rate |
|---|---:|---:|---:|---:|---:|---:|
| whole_dev | 63194 | 1149 | 960 | 189 | 0.015191 | +0.002991 |
| dominant_cell | 50458 | 846 | 700 | 146 | 0.013873 | +0.002893 |
| dominant_never_break_only | 50452 | 861 | 633 | 228 | 0.012547 | +0.004519 |
| dominant_pre_break_only | 44033 | 682 | 530 | 152 | 0.012036 | +0.003452 |

## Gates

- Pre-break damage rate: `0.012036` (gate `< 0.015000`) -> `True`
- Never-break net rate vs RT-600: `+0.004519`
- Whole-dev marginal vs seed-clone blend: `+0.002694`
- Whole-dev gain vs RT-600: `+0.002603`
- Dominant-cell gain vs RT-600: `+0.003697`
- Student raw retention of oracle residual cell lift: `n/a` -- **not a like-for-like ratio.** The denominator is the oracle residual on the global (fold-contaminated) RT-991 over the dominant cell; the numerator is the nested fold-pure student on full-population labels. Indicative magnitude only; do not quote as a retention rate.

## Correlation

- Student raw vs RT-600, within-t dominant-cell rho: `0.688702`

## Verdict

PROMOTE for confirmation only: the nested 500-feature residual student beats the seed-clone blend marginally, has positive never-break pair net, and clears the pre-break damage gate.
