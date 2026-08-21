# Public 2025 aParsec Reproduction Audit

**CALIBRATION GATE: FAIL**

**MAY WE NOW INTERPRET THE 2026 STRONG ORACLE AS AN INFORMATION-FRONTIER APPROXIMATION? NO.**

The public aParsec implementation was inspected and pinned, and the local 2025 data matches the notebook's Crunch release-146 byte lengths.  However, the exact reproduction could not be executed in this environment because the required public-solution dependencies are absent and installing them from PyPI was rejected by the permission reviewer.

This is a failed calibration gate due to an unresolved execution blocker, not evidence that the 2025 task is intrinsically weak.

## Branch And Environment

- branch: `codex/reproduce-2025-public-solution`
- base SHA: `5a3b8a0377eb9d42e4188e2af38ffd0fdff6e527`
- current HEAD when report was generated: `5a3b8a0377eb9d42e4188e2af38ffd0fdff6e527`
- Python: `3.11.6 | packaged by conda-forge | (main, Oct  3 2023, 10:37:07) [Clang 15.0.7 ]`
- missing exact-reproduction dependencies: polars, lightgbm, shap, tabpfn

## Local 2025 Data

- data path: `/home/user/Documents - Graham’s MacBook Pro/structural-break-project/competitions/structural-break/quickstarters/baseline/data`
- shape read success: True
- X rows: 23715734
- series: 10001
- positives / negatives: 2909 / 7092
- period counts: {'0': 17469105, '1': 6246629}

| file | bytes | sha256 | matches notebook release-146 bytes |
|---|---:|---|---:|
| `X_test.reduced.parquet` | 2380918 | `de086be65642e70b38312690c65afefd044e292920df189c3be94ab02756c8db` | True |
| `X_train.parquet` | 204327238 | `3f01c01bad9ecbc63f3d2c6741421f2fc1899aa84668eee4edffe26d0f99ce4a` | True |
| `y_test.reduced.parquet` | 2655 | `4d65989fe02e7e834347a4212c3436f4a04dad628cec1f1af16bf7b14ea91033` | True |
| `y_train.parquet` | 61003 | `84356084ce1564f8017c06fc9c961570d610e88879961284252c783b0aef5a54` | True |

The local file byte lengths exactly match the public notebook output for Crunch data release 146.  The public repo does not publish file hashes, so identity is strong but not cryptographically proven against a public hash.

## Public Source

- repository: https://github.com/aParsecFromFuture/ADIA-Lab-Structural-Break-Challenge-Solution
- commit SHA: `6316693333edc5831c2408ca5b155ffa24c302bd`
- implementation files: `README.md`, `submission.ipynb`; no helper Python files were present.
- reported validation AUC: not present in README or notebook outputs.
- reported leaderboard/private/public score: not present in README or notebook outputs.
- reported rank: repository description says `2nd place solution.`

## Feature Manifest

- source-derived manifest rows: 2171
- README claimed generated features: 2408
- delta: -237
- reason for discrepancy: unresolved until exact Polars execution is available; the notebook code path inspected from source expands to fewer named features than the README claim.

| stage | source-derived features |
|---|---:|
| FirstFeatureGenerator | 8 |
| SecondFeatureGenerator | 56 |
| ThirdFeatureGenerator:pivot | 1260 |
| ThirdFeatureGenerator:diff | 840 |
| FourthFeatureGenerator | 6 |
| TabPFN OOF meta-feature | 1 |

## Reproduction Ladder

| id | stage | status | AUC | reason |
|---|---|---|---:|---|
| R25-000 | previous 280-feature diagnostic | not rerun | 0.6643 | blocked: missing required public-solution dependencies: polars, lightgbm, shap, tabpfn |
| R25-010 | public transform/stat bank, single LGBM | blocked |  | blocked: missing required public-solution dependencies: polars, lightgbm, shap, tabpfn |
| R25-020 | + exact public feature selection | blocked |  | blocked: missing required public-solution dependencies: polars, lightgbm, shap, tabpfn |
| R25-030 | + four-LightGBM ensemble | blocked |  | blocked: missing required public-solution dependencies: polars, lightgbm, shap, tabpfn |
| R25-040 | + TabPFN OOF meta-feature | blocked |  | blocked: missing required public-solution dependencies: polars, lightgbm, shap, tabpfn |
| R25-050 | closest faithful full public pipeline | blocked |  | blocked: missing required public-solution dependencies: polars, lightgbm, shap, tabpfn |

## Direct Answers Required By The Gate

1. Did we load the same 2025 data as the previous branch? Yes, same local path and same hashes as the prior report.
2. Exact hashes are listed above.
3. Is it definitely the same competition/data version used by aParsec? The byte lengths match the public notebook's Crunch release-146 output exactly; no public hashes are available, so this is strong but not absolute.
4. Public repository SHA reproduced/audited: `6316693333edc5831c2408ca5b155ffa24c302bd`.
5. Files inspected: `README.md`, `submission.ipynb`; `diagram.png` noted but not code-inspected.
6. Public repo reports no local AUC, OOF AUC, Crunch score, leaderboard score, or private/public score.
7. Therefore the only understood metric-like outputs are runtime and memory from `crunch.test`; they are not validation scores.
8. Original transforms: raw, z-score over id, cumulative sum over id, dense rank/count over id, absolute value, rolling mean(16), rolling std(16).
9. README says 2408 generated features; source-derived manifest currently accounts for 2171 including `FirstFeatureGenerator` and TabPFN OOF.
10. Surviving selection: SHAP top 200/top 500 and gain top 200/top 500 are used by the four models; exact selected feature names were not reproduced because LightGBM/SHAP could not run.
11. SHAP was not reproduced; dependency unavailable.
12. Gain-based selection was not reproduced; LightGBM unavailable.
13. TabPFN was not reproduced; dependency unavailable.
14. Public notebook comments TabPFN version `2.1.3`; no local executable version.
15. TabPFN meta-feature is cross-fitted with `KFold(5, shuffle=True, random_state=42)` on the 8-column `FirstFeatureGenerator` block.
16. Four-LightGBM ensemble source was reproduced in specification but not executed.
17. Exact model params are documented in `public_2025_reproduction_sources.md`.
18. Standalone four-model scores: not available.
19. Ensemble score: not available.
20-25. R25-000 through R25-050 are listed in the ladder above; only prior R25-000 reference AUC is recorded, not rerun.
26-29. Gains from feature bank, selection, ensemble, and TabPFN cannot be quantified until execution dependencies are available.
30. The faithful historical pipeline appears to contain global feature selection before final training; whether that inflates validation cannot be quantified yet.
31. Honest fold-pure selection was not run.
32. Honest 5-fold OOF AUC: not available.
33. Public-solution-protocol AUC: not available; public protocol does not report AUC.
34. Gap to historical public performance cannot be explained from this repository because no historical AUC is documented.
35. Calibration gate: FAIL.
36. Reason: exact public dependencies unavailable and install was rejected; no strong 2025 reproduction was executed.
37-47. Strong 2026 oracle and unknown-boundary measurements were not run because the 2025 gate failed.
48-62. Real 2026 headroom, concentration, and teacher predictability remain unmeasured in this branch.
63. Retain from commit 5a3b8a0: the prior 280-feature diagnostic and data discovery are valid as a weak diagnostic.
64. Downgrade/withdraw from commit 5a3b8a0: interpreting the 2026 FULL known-boundary result as an information ceiling.
65. Final branch/commit: branch `codex/reproduce-2025-public-solution`; final commit to be filled after commit.

## Required Final Decision

CALIBRATION GATE: FAIL

MAY WE NOW INTERPRET THE 2026 STRONG ORACLE AS AN INFORMATION-FRONTIER APPROXIMATION? NO.
