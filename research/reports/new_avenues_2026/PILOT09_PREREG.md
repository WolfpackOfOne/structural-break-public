# PILOT 9(i) -- SCALAR HISTORICAL DIFFICULTY GATE PREREGISTRATION

Written 2026-08-24 on branch `research/new-avenues-pilots-2026`, at
checkpoint `903b171`, after Pilot 10 was scored, killed, committed, pushed,
and Research hygiene passed. No Pilot 9(i) validation score exists at
preregistration time.

This preregistration covers only Pilot 9 part (i), the J1 scalar difficulty
conditioner. Pilot 1 already ran the specialist/failure-manifold diagnostic
material; this file does not rerun that diagnostic.

## Hypothesis

`m05_ctx` failed because it gave the model dozens of near-continuous
series-constant columns and allowed identity-like memorization. D3 nevertheless
showed that the 23 history-only fingerprints weakly predict RT-600's own
dominant-cell loss propensity. Pilot 9(i) tests whether one fold-pure scalar,
trained only to predict RT-600 loss propensity, can act as a legal interaction
conditioner when appended as a single column to the Mode-A base bank.

The target is RT-600's own dominant-cell pairwise inversion loss rate, not
`has_break`, not tau, and not any future class label.

## Reserved IDs

| ID | candidate |
|---|---|
| `RT-1212` | nested OOF scalar difficulty conditioner |
| `RT-1213` | within-fold deranged scalar control |

## Exact Inputs

* Seven RT-600 specialist OOF streams and the RT-600 equal blend rebuilt with
  `Ctx.crossfit_blend`.
* Canonical permanent folds from `research/folds/folds.parquet`.
* The 23-column history-only fingerprint bank from
  `pilot01_failure_manifolds.history_fingerprints()`.
* Row labels and RT-600 scores only for constructing training-fold RT-600 loss
  propensity targets. No validation-fold target enters a scalar used on that
  fold.
* No true tau, `has_break`, final online length, future row, lockbox/test row,
  or production branch artifact enters the scalar model.
* Pilot 9(i) does not recompute the production `m07_bayes` stream engine or its
  known `bo_p_lt25_z` batch/stream parity issue. The Mode-A screen uses the
  existing cached legal base bank plus one series-constant scalar.

## Target Construction

Use the same dominant-cell definition as the common harness:

* `t >= 200`
* positives restricted to post-break age `>= 100`
* all negatives retained

For RT-600 only, decompose same-time pairwise inversion loss by series using the
mid-rank pair machinery already used by Pilot 1. For each series:

`loss_rate = series_pair_loss / series_pair_weight`.

Only series with `series_pair_weight > 500` are eligible as scalar-regression
training targets. Predictions are still emitted for all dev-fold series with a
finite fingerprint vector.

## Scalar Regression Model

For every scalar fit:

* features: exactly the 23 history-only fingerprint columns;
* preprocessing: robust center and scale fitted on the scalar training series
  only, with NaN/Inf converted to zero after scaling;
* model: LightGBM regression;
* `num_boost_round=250`
* `learning_rate=0.05`
* `num_leaves=15`
* `min_data_in_leaf=100`
* `feature_fraction=0.8`
* `bagging_fraction=0.8`
* `bagging_freq=1`
* `lambda_l2=5.0`
* `num_threads=2`
* `seed=0`

The emitted scalar is the raw predicted RT-600 dominant-cell loss rate. There
is no binning, monotone transform, clipping, rank transform, interaction
precomputation, or post-score recalibration.

## Nested Fold-Pure Construction

Because the scalar target is label-derived, the feature must be nested inside
the outer fold used by the Mode-A candidate model.

For each outer validation fold `f`:

1. Define outer scalar-training folds as all permanent dev folds except `f`.
2. Train one scalar regressor on eligible series in those outer-training folds
   and predict the held-out fold `f`.
3. For each model-training fold `j != f`, train an inner scalar regressor on
   eligible series in folds excluding both `f` and `j`; predict fold `j`.
4. The outer Mode-A LightGBM for fold `f` uses these fold-pure scalar values
   for both its training rows and its validation rows.

Thus no scalar value used by the outer fold `f` model is fitted using fold `f`
targets, and every row's scalar is out-of-fold with respect to its own
permanent fold.

Folds 1..4 may be predicted only to provide fold-pure SCDF calibration support
for the existing ensemble marginal path. No TS-AUC from folds 1..4 may be
inspected or used in the Pilot 9(i) screen decision.

## Derangement Control

`RT-1213` uses the exact same nested scalar values, then applies a fixed
within-fold derangement:

* seed: `0`
* derange separately inside each permanent dev fold;
* no series may keep its own scalar where the fold has at least two series;
* the fold-wise scalar multiset is preserved exactly.

This control preserves any marginal distribution or fold-level prior carried
by the scalar and destroys the series-specific fingerprint match. If this
control is not clearly worse, the scalar is treated as memorization or marginal
prior leakage rather than useful interaction information.

## Candidate And Control Architecture

Pilot 9(i) is Mode A with a custom nested feature append because the scalar is
label-derived and fold-specific. Both arms use the existing legal feature bank

`m00_core,m01_seq,m02_dist,m03_dyn,m04_resid,m06_loc,m07_bayes`

plus exactly one scalar column:

* `RT-1212`: `pilot09_difficulty_scalar`
* `RT-1213`: `pilot09_difficulty_scalar_deranged`

Use the ABL fold-0 screen architecture:

* folds: `(0,)`
* train folds: 1..4, validation fold: 0
* max training rows: 400,000
* seed: 0
* objective: binary LightGBM
* `n_estimators=600`
* `learning_rate=0.05`
* `num_leaves=63`
* `min_data_in_leaf=300`
* `feature_fraction=0.5`
* `bagging_fraction=0.7`
* `bagging_freq=1`
* `lambda_l2=5.0`
* `max_bin=127`
* `num_threads=2`
* sample mode: uniform

No candidate-specific blend weight, global rank transform, or additional
context column is allowed.

## Causal And Fold-Purity Checks Before Training

Before the first Pilot 9(i) score:

* verify the fingerprint bank shape and exact 23 names;
* verify scalar provenance for every outer fold: no scalar model used in that
  outer fit trains on the outer validation fold;
* verify every emitted scalar is out-of-fold with respect to its own permanent
  fold;
* verify the derangement has no fixed points inside folds and preserves each
  fold's scalar multiset;
* verify the row-level scalar is constant within each series;
* verify no lockbox rows are filled;
* run the new Pilot 9(i) targeted unit tests;
* run the local research hygiene script before any scored push.

## RT600 Sentinel

Before interpreting Pilot 9(i), reproduce:

* RT600 mean approximately `0.62581`
* RT600 pooled approximately `0.625627`
* dominant cell approximately `0.66428`
* fold-0 E0 RT600 approximately `0.63828`
* fold-0 E1 RT600 + RT-401 seed clone approximately `0.63859`

If these materially differ, stop and debug the evaluation path before scoring
the pilot.

## Ensemble Integration

For both `RT-1212` and `RT-1213`, compute:

* E0: RT600
* E1: RT600 + exchangeable `RT-401` seed clone
* E2: RT600 + Pilot 9(i) candidate/control
* `marginal_vs_clone = E2 - E1`

Use the same `harness.marginal` equal-weight integration path as Pilots 2-10.

## Metrics

For both scored arms report:

* whole-fold TS-AUC
* dominant-cell AUC
* mature-break vs never-break AUC
* mature-break vs pre-break AUC
* t buckets
* age buckets
* within-t rank correlation with RT600
* sampled pair repairs, damage, and net lift
* E0, E1, E2, and `marginal_vs_clone`
* candidate-control marginal gap
* scalar target OOF Spearman on fold 0 as a diagnostic only
* feature/training/runtime diagnostics

## Binding Gates

First binding control gate:

* KILL immediately if `RT-1212 marginal_vs_clone - RT-1213 marginal_vs_clone
  < +0.0005`.
* KILL immediately if `RT-1213 marginal_vs_clone >= RT-1212 marginal_vs_clone`.

Primary candidate gate after the derangement control passes:

* `marginal_vs_clone < +0.0010`: KILL
* `+0.0010` to `< +0.0020`: WEAK
* `+0.0020` to `< +0.0030`: INTERESTING
* `+0.0030` to `< +0.0050`: SERIOUS
* `+0.0050` to `< +0.0080`: MAJOR
* `>= +0.0080`: BREAKTHROUGH

## Continuation Gate

Run 5-fold confirmation only if both conditions hold:

* `RT-1212 marginal_vs_clone >= +0.0020`
* `RT-1212 marginal_vs_clone - RT-1213 marginal_vs_clone >= +0.0005`

Run paired/series bootstrap only after a positive five-fold result with
`marginal_vs_clone >= +0.0030` and positive sign on at least 4/5 folds.

## Runtime Ceiling

2.5 hours for implementation checks, scalar construction, both fold-0 arms,
diagnostics, and Pilot 9(i) report.

## Code Paths

Create:

* `research/scripts/novel_streams/pilots/pilot09_difficulty_gate.py`
* `tests/test_pilot09_difficulty_gate.py`
* `research/reports/new_avenues_2026/pilot09_difficulty_gate.md`
* `research/reports/new_avenues_2026/pilot09_difficulty_gate.json`
