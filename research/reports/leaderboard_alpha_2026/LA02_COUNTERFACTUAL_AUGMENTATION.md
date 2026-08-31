# LA-02 -- Counterfactual Synthetic Augmentation Result

Date: 2026-08-26

Program preregistration: `PROGRAM_PREREG.md` at `ed84d00`.
Execution preregistration: `LA02_EXECUTION_PREREG.md` at `8438d93`.

## Verdict

`RT-1246` is **KILL** under the preregistered survival gate.

The primary marginal signal is large enough to be **MAJOR** by magnitude
(`+0.005157709` vs the same-count synthetic clone), but the mandatory pair-flow
gates fail: the candidate damages the RT600 dominant residual cell on net and
also damages mature-vs-never pairs on net. LA-02 therefore does not become
SERIOUS for program-control purposes, and LA-03 opens.

## Scores

| arm | mean TS-AUC | pooled TS-AUC | fold TS-AUC |
|---|---:|---:|---|
| C0 RT600 | `0.625811264` | `0.625626926` | `0.638276303;0.620401850;0.633930229;0.617507505;0.618940432` |
| `RT-1245` C1 null-only synthetic control | `0.617350841` | `0.617173688` | `0.623062535;0.621246514;0.626302501;0.604153267;0.611989386` |
| `RT-1246` paired counterfactual candidate | `0.622508549` | `0.621966631` | `0.630046077;0.613145623;0.634367831;0.615112677;0.619870538` |

Primary `marginal_vs_clone = RT-1246 - RT-1245 = +0.005157709`.
Positive folds vs clone: `4/5`.

Candidate vs C0: `-0.003302715`.
C1 vs C0: `-0.008460423`.

Fold deltas vs clone:

`+0.006984; -0.008101; +0.008065; +0.010959; +0.007881`

Fold deltas vs C0:

`-0.008230; -0.007256; +0.000438; -0.002395; +0.000930`

## Pair Flow

Candidate vs C0, 64 same-`t` pairs per time point, seed `20260826`:

| split | repairs | damage | net | damage rate |
|---|---:|---:|---:|---:|
| whole dev | `3971` | `4312` | `-341` | `0.068234` |
| dominant cell | `2853` | `3179` | `-326` | `0.063003` |
| mature-vs-never | `2788` | `3154` | `-366` | `0.062515` |
| mature-vs-prebreak | `2428` | `2674` | `-246` | `0.060727` |

Within-`t` rank correlation vs C0:

| arm | dev | dominant cell |
|---|---:|---:|
| C1 | `0.848391` | `0.865745` |
| candidate | `0.865618` | `0.891756` |

## Gate Accounting

| gate | result |
|---|---|
| `marginal_vs_clone >= +0.0025` | PASS |
| at least 4/5 folds positive | PASS |
| candidate beats C1 by `>= +0.0010` | PASS |
| dominant-cell net positive | FAIL (`-326`) |
| mature-vs-never net positive | FAIL (`-366`) |

## Compute

Synthetic rows generated: `1,652,724` for C1 and `1,653,744` for candidate.
Runtime: `11,255.3 s` (`3.13 h`).

Metrics: `la02_counterfactual_augmentation.json` and
`la02_counterfactual_augmentation_summary.csv`.
