# SS-02 Execution Preregistration -- Dominant-Cell Residual Ranker

This file freezes the implementation choices for `SS-02` before any SS-02
headline validation score is generated. It follows the program-level
preregistration in
`research/reports/new_avenues_2026/SECOND_SWEEP_PREREG.md` after `SS-01` was
killed at commit `30abfeb`.

## Question

Can a correction model trained directly against RT600 residual same-t ranking
errors move broad dominant-cell pair flow using only the incumbent causal bank?

This is not a new anomaly statistic, not generic stacking, and not a
break-probability BCE classifier. The learned stream is a bounded correction to
the frozen RT600 score.

## Experiment IDs

- `RT-1223`: SS-02 bounded residual-pair correction candidate.
- `RT-1224`: shuffled residual-offset/weight control.

The existing `RT-401` seed clone remains the matched diversity comparator and
does not consume a new RT ID. The seven RT600 specialist streams
`RT-300`, `RT-410`, `RT-411`, `RT-412`, `RT-413`, `RT-414`, and `RT-415` are
reused only as frozen OOF covariates and as the RT600 base blend.

## Collision Audit

The old generic same-t pairwise objective (`RT-110`, `RT-111`, `RT-123`,
`RT-700`, `RT-701`, `RT-702`) optimized global break-vs-no-break ordering. Its
full-scale result was a dead heat with BCE and did not specifically model where
RT600 was already correct versus wrong.

W5-E3 hard-negative training (`RT-710`, `RT-711`, `RT-712`) changed row
emphasis inside a BCE-style learner using nested hardness, and failed by
suppressing high-evidence episodes generally.

SS-02 differs materially: the loss is defined on the final corrected margin
`RT600 + bounded correction`; every training pair carries the frozen RT600
residual offset; and RT600-correct pairs receive an explicit damage-preservation
penalty. No first-sweep killed sensor prediction is used.

## Data, Folds, And Population

- Use the canonical full development store only, via `sbr.pipeline.Data()` and
  `research/folds/folds.parquet`.
- Development folds are `0,1,2,3,4`.
- Fold `-1` lockbox rows are never used for training, calibration, scoring, or
  selection.
- No `X_test.reduced.parquet`, hidden/test rows, production rows, or submission
  data are touched.
- Train and predict all five development folds to create a fold-pure OOF vector
  for the candidate and shuffled control. The binding screen is fold `0` only.
  Folds `1`-`4` from this run are support for OOF/calibration provenance and are
  not a serious-result confirmation.

For outer validation fold `f`:

1. Validation rows are exactly fold `f`.
2. The training pool is all rows from folds `{0,1,2,3,4} \ {f}`.
3. The model training rows are a sorted uniform sample without replacement of at
   most `800000` rows from the training pool using seed `2026082505 + f`.
4. Pair training examples are sampled only from those selected training rows.

## Inputs

Allowed feature matrix:

- the existing 500-column causal feature bank:
  `m00_core`, `m01_seq`, `m02_dist`, `m03_dyn`, `m04_resid`, `m06_loc`,
  `m07_bayes`;
- fold-pure calibrated OOF scores for the seven RT600 specialists;
- fold-pure calibrated OOF score for `RT-401`;
- deterministic causal score-state summaries:
  `rt600_cal`, `abs(rt600_cal - 0.5)`, `logit(rt600_cal)`,
  `RT-401 - rt600_cal`, `abs(RT-401 - rt600_cal)`, specialist standard
  deviation, range, min, max, fraction above RT600, fraction above 0.5,
  `log1p(t)`, and indicators for `t >= {20,50,100,200,400}`.

Forbidden inputs:

- every first-sweep candidate or control prediction (`RT-1200` through
  `RT-1222`);
- `tau`, true break age, future rows, final online length, `n_online`,
  `n_hist`, lockbox/test information, labels at inference, or validation/test
  cross-sectional ranks.

Column names are checked against the existing forbidden-token guard before
scoring.

## Score Calibration And Fold Purity

All frozen OOF score inputs are transformed by `SCDF_NSEEN`.

For outer validation fold `f`:

- validation-fold score calibration maps are fit only on folds not equal to
  `f`;
- for training rows in inner fold `g`, score calibration maps are fit only on
  folds excluding both `f` and `g`;
- calibration uses only scores and legal time coordinates, not labels;
- no same-t validation ranks or test ranks are computed as features.

The RT600 base for training and validation is the mean of the seven calibrated
specialist streams under this outer/inner fold-pure calibration.

## Pair Target

For each outer fold `f`, after selecting the `800000` or fewer training rows:

1. Group selected training rows by online index `t`.
2. Let positives be rows with `y=1` and negatives rows with `y=0`.
3. For every `t`, sample
   `k = min(128, n_positive_at_t, n_negative_at_t)` positives without
   replacement and `k` negatives without replacement.
4. Pair sampled positives and negatives in sampled order.
5. The per-t RNG seed is `2026082504 + 1000*f + t`.

For every sampled pair `(p, n)`:

- `base_diff = rt600_cal[p] - rt600_cal[n]`;
- a repair-needed pair has `base_diff <= 0`;
- a preserve-needed pair has `base_diff > 0`;
- raw pair weight is `1.0` for repair-needed pairs;
- raw pair weight is `2.0` for preserve-needed pairs, the explicit damage
  penalty;
- multiply weight by `3.0` when `t >= 200` and the positive row has true
  post-break age at least `100`;
- normalize pair weights within each outer fold to mean `1.0`.

True age is used only for training-pair weighting, never as an inference input.

## Model And Loss

Model: LightGBM over the 500-bank plus score-state covariates.

Parameters:

- `learning_rate = 0.05`
- `num_leaves = 31`
- `max_depth = 6`
- `min_data_in_leaf = 300`
- `feature_fraction = 0.7`
- `bagging_fraction = 0.7`
- `bagging_freq = 1`
- `lambda_l2 = 10.0`
- `num_threads = 2`
- `max_bin = 127`
- `seed = 20260825`
- boosting rounds: `250`

Custom residual pair loss:

For model raw output `z`, pair margin is

`base_diff / 0.10 + z[p] - z[n]`.

The loss is weighted logistic pair loss:

`w * log(1 + exp(-margin))`.

The base offset makes the loss operate on RT600 residual ranking errors rather
than ordinary break classification. The higher preserve-needed pair weight is
the damage penalty.

## Candidate Score

For each outer validation row:

1. Predict raw correction `z`.
2. Clip `z` to `[-0.5, +0.5]`.
3. Candidate score is `rt600_cal + 0.10 * clipped_z`.

The maximum correction magnitude is therefore `0.05`. The output is a corrected
RT600 stream, not a standalone probability model.

## Shuffled Residual Control

`RT-1224` uses the same rows, features, pair endpoints, model parameters, and
candidate score formula as `RT-1223`.

Before fitting each outer fold, it permutes the tuple `(base_diff, pair_weight)`
across the sampled pair list using seed `2026082506 + f`. The positive/negative
endpoints are unchanged. This preserves generic same-t label orientation and
pair count but destroys the association between row features and RT600 residual
offset/damage state.

## Evaluation

Before scoring:

- reproduce the RT600 sentinel:
  dev mean `0.625811`, pooled `0.625627`, dominant-cell `0.664277`, fold-0 E0
  `0.638276`, fold-0 E1 `0.638586`;
- verify frozen specialist and seed-clone OOF coverage on dev rows and zero
  finite lockbox rows;
- verify no first-sweep RT IDs are loaded.

Report for `RT-1223` and `RT-1224`:

- fold-0 standalone corrected-stream TS-AUC;
- E0, E1, E2, `marginal_vs_clone`, and `gain_vs_RT600`;
- pair repairs, damage, net, and RT600-right damage rate for whole fold,
  dominant cell, mature-vs-never, and mature-vs-prebreak using the canonical
  deterministic pair sample with seed `20260825` and `64` pairs per `t`;
- train-pair counts and RT600 wrong/right counts;
- correction clipping rates;
- top feature importance;
- runtime and peak memory.

## Binding Fold-0 Screen Gate

`SS-02` passes the screen only if all of the following hold on fold `0`:

1. `RT-1223` marginal_vs_clone is at least `+0.0015`.
2. Dominant-cell net pair lift is at least `+300` on the canonical deterministic
   pair sample.
3. Mature-vs-never pair net is positive, or it is exactly neutral with
   mature-vs-prebreak net at least `+300`.
4. `RT-1223` beats `RT-1224` by at least `+0.0005` marginal_vs_clone.
5. `RT-1224` does not independently satisfy the candidate screen.
6. Legality and purity checks pass.

If any mandatory gate fails, `SS-02 = KILL`. No pair-loss retuning, no alternate
damage penalty, no row-budget change, no feature subset retry, and no SS-02b are
authorized.

If the screen passes, stop breadth execution and run the preregistered SS-02
5-fold confirmation gate from `SECOND_SWEEP_PREREG.md`.

## Runtime And Accounting

The SS-02 screen is expected to fit two five-fold OOF streams. If projected wall
clock exceeds six hours, finish the active fold safely, record the partial
state, and stop for inspection rather than changing the design.

Append exactly one `RESULTS.csv` row for `RT-1223` and one for `RT-1224` after
scoring. Do not modify old result rows.
