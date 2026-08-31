# Arm-C Horizon Residualization Diagnostic

Date: `2026-08-30T13:06:04`

This is an offline diagnostic of the W7-D3R Arm C result. It uses no
new model training, allocates no RT ID, and does not edit `RESULTS.csv`.

## Inputs

- folds: `/path/to/workspace/structural-break-multi-agent-frontier-20260829/research/folds/folds.parquet`
- OOF dir: `/path/to/workspace/structural-break-wave8/research/oof`
- Arm A: `/path/to/workspace/structural-break-wave8/research/oof/RT-300.npy`
- Arm B: `/path/to/workspace/structural-break-wave8/research/oof/RT-990.npy`
- Arm C: `/path/to/workspace/structural-break-wave8/research/oof/RT-991.npy`

Population: dev folds 0-4, `t >= 200`, and either positive
post-break age `>= 100` or any negative row.

## Scores

| score | dominant cell | vs Arm B | never-break-only | pre-break-only |
|---|---:|---:|---:|---:|
| Arm_A_RT300 | 0.653407 | +0.005915 | 0.654132 | 0.651314 |
| Arm_B_RT990 | 0.647491 | +0.000000 | 0.649119 | 0.642799 |
| Arm_C_RT991 | 0.718588 | +0.071097 | 0.714388 | 0.730697 |
| Horizon_projection_xfit | 0.599887 | -0.047605 | 0.564864 | 0.700855 |
| T_orthogonal_residual_xfit | 0.685794 | +0.038303 | 0.703074 | 0.635977 |
| Horizon_projection_insample | 0.600104 | -0.047388 | 0.565104 | 0.701002 |
| T_orthogonal_residual_insample | 0.686047 | +0.038555 | 0.703332 | 0.636214 |
| Shuffled_horizon_projection_xfit | 0.499300 | -0.148192 | 0.499274 | 0.499373 |
| Residual_after_shuffled_projection | 0.718422 | +0.070931 | 0.714209 | 0.730568 |

## Projection Diagnostics

- Cross-fit projection R2 on logit Arm C: `0.394188`
- In-sample projection R2 on logit Arm C: `0.396075`
- Shuffled-horizon cross-fit projection R2: `0.198522`
- Cross-fit projection groups: `{'fitted_groups': 3995, 'fallback_groups': 5}`
- In-sample projection groups: `{'fitted_groups': 799, 'fallback_groups': 0}`
- Within-t rank rho, residual vs Arm B: `0.457016`
- Within-t rank rho, Arm C vs Arm B: `0.413710`

## Residual Lift By Cut

- Dominant cell (gated): `+0.038303`
- Never-break-only cut: `+0.053955`
- Pre-break-only cut: `-0.006822`

**The residual lift is a never-break-cut effect only.** On the
pre-break cut the T-orthogonal residual scores *below* Arm B, so the
residual carries no pre-break signal at all. Arm C's large raw
pre-break lift (`+0.087898`) is endpoint/horizon information, not
transferable structure -- the horizon projection alone scores
`0.700855` on that cut. Any downstream student of this residual
should be expected to move the never-break cut and nothing else,
and should be described that way.

## Verdict

PROMOTE diagnostic: the T-orthogonal residual keeps material dominant-cell signal beyond Arm B. Next step is a nested causal student with the Wave 7 fold-purity sentinel.

Primary gate from the Grok report: residual cell lift over Arm B
must be at least `+0.020` to promote the residual-student lane;
the lane is killed if lift is below `+0.010` or residual-vs-B
within-t rank rho exceeds `0.850`. That gate is scoped to the
dominant cell; it does not certify the pre-break cut, which fails.
