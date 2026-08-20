# Codex 2025 public-repo translation report

Date: 2026-08-20 UTC

This report translates two reachable 2025 public solution repositories into
causal 2026 real-time ideas. It is a source/engineering report only: no
validation surfaces, lockbox, `research/folds/`, `research/RESULTS.csv`, or
feature modules were changed.

## Source access status

| Source | Status | Evidence read |
| --- | --- | --- |
| `aParsecFromFuture/ADIA-Lab-Structural-Break-Challenge-Solution` | `VERIFIED`; 2nd-place claim is `ATTESTED` by repo/README, not independently re-ranked here | repository listing, `README.md`, `submission.ipynb` |
| `StefanConstantin707/adia-lab-structural-break-challenge` | `VERIFIED` | repository listing, `README.md`, `main.py`, and high-signal modules under `src/features/` |

## Repository facts

### aParsecFromFuture

The README and notebook describe a 2025 offline/known-boundary solution using
LightGBM over a large transformed feature bank. The notebook builds two main
feature matrices, adds a TabPFN OOF/meta feature, selects features with SHAP and
LightGBM gain, and trains an averaged four-model LightGBM classifier. Its own
notebook output shows `crunch.test(force_first_train=False)` completed with
determinism check passed in `00:06:13` and memory after `10.11 GB`.

Verified implementation details:

- transformations: raw/z-score, cumulative sum, dense rank/count, absolute
  value, rolling mean(16), rolling std(16)
- simple stats: mean, median, std, min, max, skewness, autocorrelation,
  coefficient of variation, Q25, Q75
- multi-window stats/correlations over windows `20`, `60`, `120`, `500`, `1000`
- two-sample tests on absolute values: F-test, Levene, KS
- TabPFN: five fold-trained classifiers used to create one OOF/meta probability
  column
- feature selection: SHAP top 200/500 plus LightGBM-gain top 500
- final model: four LightGBM models over two selected feature sets and two
  feature-count cuts, averaged

### StefanConstantin707

The README and code describe a 2025 offline/known-boundary classifier with
classical statistics, distribution tests, time-series descriptors, and XGBoost
or RandomForest fallback. `main.py` imports a broad extractor set and logs
apparent single-module scores in comments; these comments are useful triage
signals but are not revalidated here.

Verified implementation details:

- distribution features: segment stats, t-test, Levene, Mann-Whitney, KS,
  Wasserstein, energy distance, Cramer-von Mises, Epps-Singleton, residualized
  variants
- CUSUM/CUSUMSQ features: regression residual CUSUM, CUSUMSQ, p-values,
  significant flags, PACF lag selection, Jarque-Bera/skew/kurtosis,
  Ljung-Box, variance-ratio diagnostics
- regression breakpoint features: Chow-style AR regression split, RSS/F-stat,
  p-value, coefficient-difference norm
- spectral features: periodogram centroid, rolloff, bandwidth, dominant
  frequency, cross-spectral Euclidean/KL distances
- volatility features: realized volatility, vol-of-vol, return skew/kurtosis,
  upside/downside volatility, asymmetry
- autocorrelation features: ACF/PACF at lags 1/2/5/10, Ljung-Box, first zero,
  significant-lag count, padded ACF/PACF distances

## Idea translations

| ID | Source/status | 2025 observation | Causal 2026 translation | Target signal | False-signal risk | Companion/control | Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RT-901 | aParsec `VERIFIED` | The transform bank crosses raw/z-score/cumsum/rank/abs/rolling views with a statistics bank. | Add only missing low-cost transforms that are not already covered by `m00_core`/`m02_dist`: causal dense-rank/PIT summaries and cumsum drift summaries, each updated from historical reference state plus online prefix. | distribution shape, monotone drift, rank spread | transform redundancy and time-since-start leakage | length-matched historical null calibration and LOFO importance check | screen as small add-on block, not a full 2,408-feature clone |
| RT-902 | aParsec `VERIFIED` | Windowed pre/post stats and correlations at 20/60/120/500/1000 exploit known boundary. | Replace known-boundary windows with causal trailing windows and soft candidate-tau summaries over the online prefix; never use final online length. | transient/localized scale, dependence, and mean shifts | candidate grid overfit; short-window noise | localization confidence and persistence features | prototype a tiny window bank over existing streaming primitives |
| RT-903 | aParsec `VERIFIED` | F-test, Levene, and KS on absolute values are explicit features. | Maintain online Levene/Brown-Forsythe, rank-based Fligner-like, and KS/PIT distances versus historical distribution and trailing historical windows. | robust scale and distribution shift | p-values drift with repeated online testing | e-value/null-calibrated z-scores; decayed peak plus persistence | promote only if it beats existing scale/divergence features in 5-fold OOF |
| RT-904 | aParsec `VERIFIED` | TabPFN OOF probability is used as one meta feature. | Do not ship TabPFN online; instead test whether offline TabPFN-like predictions can be distilled into a tiny causal LightGBM stream feature or used as a feature-selection oracle. | nonlinear feature interactions | dependency/runtime risk; non-causal batch assumptions | distillation into existing feature space; strict local harness | research only unless distilled artifact is dependency-free |
| RT-905 | aParsec `VERIFIED` | SHAP top 200/500 and LightGBM-gain top 500 define final feature subsets. | Run fold-pure feature selection for new blocks, using only training folds inside each evaluation split; compare gain/SHAP to time-bucket permutation importance. | feature-bank pruning and decorrelation | selection leakage and gain bias toward high-cardinality/time features | nested fold selection; time-bucketed permutation | use as a screening protocol, not as post-hoc validation |
| RT-906 | Stefan `VERIFIED` | DistributionCombined uses raw and AR-residual distances: Wasserstein, energy, KS, CvM, Epps-Singleton, Mann-Whitney, Levene, t-test. | Compute residualized distribution distances online by fitting historical AR residual models once, then comparing online residual prefix/trailing windows to historical residuals. | scale, tail, and dependence changes after removing predictable dynamics | AR misspecification; serial correlation inflates test evidence | residual diagnostics and length-matched null windows | add Wasserstein/energy/CvM residual channels if absent |
| RT-907 | Stefan `VERIFIED` | CUSUM/CUSUMSQ regression residual features include p-values, significance flags, PACF selection, JB/skew/kurtosis, Ljung-Box, variance ratio. | Use online CUSUM/CUSUMSQ recursions on historical-fit residuals; expose normalized level, decayed peak, elapsed-since-peak, and diagnostic flags. | mean/scale changes in residual process | heavy tails and volatility bursts create false alarms | JB/heavy-tail and Ljung-Box flags used as trust modifiers | implement only as calibrated evidence, not hard p-value thresholding |
| RT-908 | Stefan `VERIFIED` | Chow-style AR regression breakpoint features use full/pre/post known-boundary regressions. | Approximate with a causal candidate-tau grid over the observed online prefix: max/softmax split-RSS improvement and coefficient-difference norm. | AR coefficient/dependence breaks | grid search overfits noise; too expensive if dense | coarse grid, minimum segment length, null-calibrated max statistic | benchmark cost before adding to RT artifact |
| RT-909 | Stefan `VERIFIED` | Spectral extractor compares segment periodograms via centroid, rolloff, bandwidth, dominant frequency, Euclidean/KL distance. | Maintain trailing Welch/periodogram summaries for a few fixed windows and compare spectral shape against historical reference. | frequency/dependence breaks | variance changes masquerade as spectral change; edge effects | variance-normalized spectral shape and paired volatility controls | test as a small `m03_dyn` extension |
| RT-910 | Stefan `VERIFIED` | Volatility extractor includes realized vol, vol-of-vol, downside/upside vol, return skew/kurtosis, asymmetry. | Add robust realized-vol and vol-of-vol summaries using clipped returns and historical MAD normalization. | scale and volatility-regime changes | outliers and transient bursts | robust MAD, tail count, persistence/decayed peak | compare to existing `m00_core` scale and `m04_resid` residual-vol features |
| RT-911 | Stefan `VERIFIED` | Autocorrelation extractor uses ACF/PACF lags 1/2/5/10, Ljung-Box, first zero, significant-lag count, ACF/PACF distances. | Maintain causal lag-product moments and trailing-vs-historical ACF distances for a few lags; PACF only if cheap/stable. | dependence breaks | small-sample ACF noise; effective-n volatility | effective-sample and lag confidence features | add if it decorrelates from current `m03_dyn` lag features |
| RT-912 | Stefan `VERIFIED` from README/main imports; internals not fully read | README names CUSUM, PELT, Binary Segmentation, LSTM/CNN/attention as advanced techniques. | Treat offline changepoint algorithms as teachers/feature generators, not online runtime dependencies; distill their alarms into causal features if useful. | localized break timing | offline methods see both sides of the break | causal replay of teacher outputs only up to t | low priority until cheaper tests plateau |
| RT-913 | Stefan `VERIFIED` from imports/file listing; internals not fully read | Repo includes CNN autoencoder and nonlinear/factor/GARCH feature files. | Consider self-supervised/deep encoders only as offline teachers or distilled summary features; avoid shipping neural runtime. | nonlinear pattern changes | runtime, dependency, and non-causal training shortcuts | no-GPU distillation and local harness budget gate | defer unless classical blocks stall |

## Immediate recommendations

1. First screen the low-cost robust test/distance ideas: `RT-903`, `RT-906`,
   `RT-907`, and `RT-911`.
2. Treat TabPFN/deep/offline changepoint ideas as teacher/distillation work only.
3. Keep every candidate behind the same controls that protected RT-150:
   length-matched historical nulls, no `n_online`, no cross-series state, and
   official Crunch determinism testing before promotion.
