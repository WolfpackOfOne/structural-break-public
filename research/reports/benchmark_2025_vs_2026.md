# 2025 vs 2026 Information-Structure Benchmark

**ORACLE / DIAGNOSTIC -- NOT DEPLOYABLE.** This report compares offline and
known-boundary information structures. It does not create or alter a 2026
submission model.

## 2025 Data Found

Yes. A local 2025-style Structural Break dataset was found at:

`/home/user/Documents - Graham’s MacBook Pro/structural-break-project/competitions/structural-break/quickstarters/baseline/data`

Evidence:

- `.crunchdao/project.json`: `{"competitionName": "structural-break", ...}`.
- `X_train.parquet`: 23,715,734 rows, 10,001 series, MultiIndex
  `(id, time)`, columns `value`, `period`.
- `y_train.parquet`: 10,001 labels, index `id`, column
  `structural_breakpoint`.
- `period == 0` is the pre-boundary segment and `period == 1` is the
  post-boundary segment, so the candidate boundary is supplied for every series.
- Class balance: 2,909 break / 7,092 no-break.

Hashes:

| file | sha256 |
|---|---|
| `X_train.parquet` | `3f01c01bad9ecbc63f3d2c6741421f2fc1899aa84668eee4edffe26d0f99ce4a` |
| `y_train.parquet` | `84356084ce1564f8017c06fc9c961570d610e88879961284252c783b0aef5a54` |

The reduced 2025 test labels exist in that directory, but this diagnostic used
only `X_train.parquet` and `y_train.parquet` with deterministic 5-fold CV.

## 2025 Benchmark

Protocol: one row per series, known boundary from `period`, deterministic
`StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)`, fixed feature
bank shared with the 2026 oracle, and 500-tree LGBM for the rich model.

| model | CV AUC | fold AUCs |
|---|---:|---|
| best single statistic | 0.5716 | 0.6008 / 0.5918 / 0.5971 / 0.5938 / 0.5969 |
| logistic | 0.6091 | 0.6067 / 0.5951 / 0.6154 / 0.6127 / 0.6157 |
| LGBM basic | 0.6465 | 0.6306 / 0.6464 / 0.6555 / 0.6522 / 0.6483 |
| LGBM rich | 0.6643 | 0.6346 / 0.6662 / 0.6816 / 0.6642 / 0.6766 |

This does **not** reproduce the historical `~0.90` winning benchmark. The local
data discovery is real, but this diagnostic pipeline is not a 2025 public
solution reproduction.

## Apples-To-Apples Table

| Dataset/task | Boundary known? | Future post-boundary available? | Metric | AUC |
|---|---|---|---|---:|
| 2025 actual local task | yes | yes | series ROC AUC | 0.6643 |
| 2026 data, true boundary + FULL post | yes | yes | series ROC AUC | 0.6497 |
| 2026 data, true boundary + 200 | yes | 200 pts | series ROC AUC | 0.6640 |
| 2026 data, true boundary + 100 | yes | 100 pts | series ROC AUC | 0.6161 |
| 2026 data, true boundary + 20 | yes | 20 pts | series ROC AUC | 0.5552 |
| 2026 legal RT-300 at matched FULL | no true-boundary access | causal | series ROC AUC diagnostic | 0.6100 |
| 2026 official system | no | causal | TS-AUC | reported current range ~0.60-0.63 |

The final two rows are not the same metric as the offline series AUC rows.

## Decomposition

Diagnostic, not causal accounting:

| component | estimate |
|---|---:|
| 2025 local rich benchmark A | 0.6643 |
| 2026 FULL known-boundary B | 0.6497 |
| A - B, local apples-to-apples gap | +0.0146 |
| 2026 h=100 known-boundary C | 0.6161 |
| B - C, limited-post penalty vs h=100 | +0.0336 |
| 2026 FULL unknown-boundary D | 0.5154 |
| B - D, known-boundary advantage | +0.1289 |
| 2026 FULL current RT-300 E | 0.6100 |
| B - E, full-future model-extraction gap | +0.0396 |

Because the 2025 benchmark is only `0.6643`, the comparison cannot prove that
the 2026 generator lacks all `~0.90` information. It does show that this
classical/rich diagnostic stack does not reveal `~0.90`-class information in
the 2026 known-boundary/full-post analogue.
