# Public 2025 Reproduction Sources

## Public Repository

- URL: https://github.com/aParsecFromFuture/ADIA-Lab-Structural-Break-Challenge-Solution
- pinned commit: `6316693333edc5831c2408ca5b155ffa24c302bd`
- notebook blob SHA: `71aa82494d31320148f7a191cce3339135701a61`
- repository description from GitHub API: `2nd place solution.`
- license: none declared by the GitHub repository metadata at audit time

## Files Inspected

| file | inspected | role |
|---|---:|---|
| `README.md` | True | Prose description, dependencies, feature count claim, second-place claim. |
| `submission.ipynb` | True | Only implementation source in the public repository. |
| `diagram.png` | False | Workflow image only; not executable logic. |

## Implementation Facts Extracted From `submission.ipynb`

- Imports pin/comment these versions: numpy==2.1.2, pandas==2.3.2, polars==1.2.1, joblib==1.5.2, scikit-learn==1.6.1, lightgbm==4.6.0, tabpfn==2.1.3, shap==0.48.0, scipy==1.16.1.
- `FirstFeatureGenerator`: raw whole-series mean, median, max, min, std, skew, mean/std, median/std.
- `SecondFeatureGenerator`: raw, z-score, cumulative sum, dense rank/count, absolute value, rolling mean(16), rolling std(16), then whole-series stats.
- `ThirdFeatureGenerator`: same seven transforms, lag-1 correlation plus quantiles and simple stats over pre-tail windows `20, 60, 120, 500, 1000`, pre-tail multiples `1x/2x/3x`, and post head `1x`; includes differences against the post-head block.
- `FourthFeatureGenerator`: applies absolute value, then F-test, Levene, and KS statistics/p-values between `period == 0` and `period == 1`.
- TabPFN: five `TabPFNClassifier` objects are loaded from `model_tabpfn_0.joblib` ... `model_tabpfn_4.joblib`; each is fitted inside `KFold(5, shuffle=True, random_state=42)` on the `FirstFeatureGenerator` block and produces an OOF probability `col_0`.
- Feature selection: one global `LGBMClassifier(n_estimators=750, learning_rate=0.01, colsample_bytree=0.3, max_depth=8, random_state=42)` is fit on all training rows after adding the TabPFN OOF feature. SHAP mean absolute values and LightGBM gain are sorted ascending; the ensemble later takes `[-200:]` and `[-500:]` from those lists.
- Final ensemble: four LightGBM models with seeds `12, 22, 32, 42`, `n_estimators=5000`, `learning_rate=0.01`, `colsample_bytree=0.2`, `bagging_freq=4`, `bagging_fraction=0.8`, `max_depth=8`, averaged by probability.

## Reproduction Status

| component | status |
|---|---|
| source inspection | complete |
| feature manifest | source-derived, written |
| exact Polars feature execution | blocked until `polars` is installed |
| exact LightGBM baseline and ensemble | blocked until `lightgbm` is installed |
| SHAP feature selection | blocked until `shap` and `lightgbm` are installed |
| TabPFN OOF feature | blocked until `tabpfn` is installed and model download/hardware behavior is verified |

## Dependency Compatibility

- current Python: `3.11.6 | packaged by conda-forge | (main, Oct  3 2023, 10:37:07) [Clang 15.0.7 ]`
- current package state: numpy=2.4.6, pandas=2.3.3, scipy=1.17.1, scikit-learn=1.9.0, polars=NOT_INSTALLED, lightgbm=NOT_INSTALLED, shap=NOT_INSTALLED, tabpfn=NOT_INSTALLED, torch=2.5.1, pyarrow=24.0.0, matplotlib=3.11.0, joblib=1.5.3.
- missing exact-reproduction dependencies: polars, lightgbm, shap, tabpfn.

## License Considerations

The public repository metadata returned no declared license.  The reproduction branch therefore records facts and implements independently scoped audit code rather than vendoring the public notebook wholesale.
