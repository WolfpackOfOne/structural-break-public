# SS-01 Execution Preregistration -- Repair-Damage Arbiter

Status: `PREREGISTERED_NOT_SCORED`

Program preregistration: `research/reports/new_avenues_2026/SECOND_SWEEP_PREREG.md`
at `2be1d11`.

This file freezes the implementation details for the first Second Sweep
experiment before any SS-01 headline validation score is generated.

## Objective

Core question: can a fold-pure, row-deployable arbiter keep useful first-sweep
repairs while rejecting first-sweep damage?

This is not a new anomaly statistic and not unrestricted stacking. First-sweep
arms are used only as frozen, prefix-computable score sensors.

## RT IDs

- `RT-1219`: SS-01 constrained repair-damage arbiter candidate.
- `RT-1220`: global average of the same frozen first-sweep sensors control.
- `RT-1221`: shuffled repair/damage target arbiter control.
- `RT-1222`: single-best killed-arm blend control selected using train folds only.

Family-holdout ablations are recorded inside the SS-01 report as diagnostic
controls, not as separately promotable RT experiments.

## Data And Folds

- Population: canonical 8,000-series development folds `0,1,2,3,4` from
  `research/folds/folds.parquet`.
- Screen fold: outer fold `0`.
- Fold-pure OOF support: `RT-1219`, `RT-1221`, and every family-holdout arbiter
  predict all five dev folds with one outer model per fold; only fold `0` is
  interpreted for the screen gate.
- Lockbox fold `-1`: not loaded for scoring, not filled in new OOF vectors, and
  not used for selection.
- Test/reduced/final/submission data: not read.

## Frozen Score Inputs

The action set uses the twelve first-sweep candidate sensors:

- `RT-1200` relay score-state
- `RT-1201` IM2/dwell
- `RT-1202` trajectory geometry
- `RT-1204` scale survival
- `RT-1206` spectral impulse
- `RT-1208` ordinal irreversibility
- `RT-1210` joint rarity
- `RT-1212` scalar difficulty
- `RT-1214` Kalman/NIS observer
- `RT-1215` Hankel-DMD observer
- `RT-1216` weighted CTM
- `RT-1218` direct e-value aggregation

The matched first-sweep controls (`RT-1203`, `RT-1205`, `RT-1207`, `RT-1209`,
`RT-1211`, `RT-1213`, `RT-1217`) are excluded from the candidate action set.

Additional allowed score state:

- RT-600 seven specialist OOF streams:
  `RT-300`, `RT-410`, `RT-411`, `RT-412`, `RT-413`, `RT-414`, `RT-415`
- `RT-401` seed clone
- causal row state: online index `t` transformed as `log1p(t)`, RT600 margin,
  specialist dispersion/range, specialist-vs-RT600 disagreement counts, and
  first-sweep sensor deltas from RT600.

Forbidden as features: `tau`, true break age, `n_online`, `n_hist`, final online
length, future rows, lockbox/test data, and labels at inference.

## Calibration

Every raw score input is transformed with the frozen `SCDF_NSEEN` calibration
from `research/scripts/wave4_cal.py`.

For outer validation fold `f`, calibration maps for validation features are fit
only on folds `{0,1,2,3,4} \ {f}`.

For rows used to train the arbiter inside outer fold `f`, score features are
inner-cross-fitted: a training row from fold `g` uses calibration maps fit only
on `{0,1,2,3,4} \ {f,g}`. This keeps the feature construction for arbiter
training row-pure with respect to both the outer validation fold and the row's
own fold.

No within-time validation rank, contemporaneous test rank, or validation-label
calibration is used as an input.

## Pair Sample And Target

Training target is derived from same-time positive/negative pairs, not from BCE.

For each outer fold `f`, training pairs are sampled from folds
`{0,1,2,3,4} \ {f}` only. For each online index `t`, sample up to `64`
positive rows and `64` negative rows without replacement and pair them in the
sampled order. Pair RNG seed is `2026082501 + 1000*f + t`.

Pair weights:

- `3.0` if the pair lies in the dominant cell (`t >= 200` and the positive row
  has true post-break age `>= 100`);
- `1.0` otherwise.

The true age condition is used only to weight training targets and report
diagnostics. It is never an inference feature.

For each sampled pair `(p, n)` and sensor family `j`, define:

- `base_diff = rt600_cal[p] - rt600_cal[n]`
- `delta_j = clip(sensor_j_cal - rt600_cal, -0.25, +0.25)`
- `action_diff_j = base_diff + 0.20 * (delta_j[p] - delta_j[n])`

Family utility contribution:

- if `base_diff <= 0` and `action_diff_j > 0`, add `+weight`
- if `base_diff > 0` and `action_diff_j <= 0`, add `-2.0 * weight`
- otherwise add `0`

The same family utility is accumulated for both rows in the pair. Rows with at
least one sampled exposure receive a class target:

- `0`: RT600 only, if every family utility is non-positive;
- `1..12`: the family with maximum positive utility, ties resolved by the
  sensor order listed above.

The shuffled-target control uses the same pair sample and features, but applies
a seed-`2026082502` within-outer-train permutation to the row class targets
before fitting.

## Model

Model class: one LightGBM multiclass classifier per outer fold.

Fixed parameters:

- `objective="multiclass"`
- `num_class=13`
- `learning_rate=0.05`
- `num_leaves=15`
- `max_depth=4`
- `min_data_in_leaf=500`
- `feature_fraction=0.8`
- `bagging_fraction=0.8`
- `bagging_freq=1`
- `lambda_l2=10.0`
- `num_threads=2`
- `max_bin=127`
- `seed=20260825`
- `num_boost_round=150`

No hyperparameter sweep, no fold-0 threshold search, and no feature selection.

## Candidate Score Formula

For each validation row, the arbiter chooses the highest-probability class.

- If class `0` is chosen: candidate score = `rt600_cal`.
- If family `j` is chosen: candidate score =
  `rt600_cal + 0.20 * clip(sensor_j_cal - rt600_cal, -0.25, +0.25)`.

Thus the maximum absolute correction is `0.05` on the calibrated score scale.
No learned continuous weight over first-sweep sensors is used.

The saved `RT-1219.npy` stream is the corrected calibrated score on dev folds
only, with non-dev rows left `NaN`.

## Controls

`RT-1220` global average control:

- score = mean of the twelve calibrated first-sweep sensor streams;
- no training;
- same fold-pure calibration path as validation candidate features.

`RT-1221` shuffled-target arbiter:

- identical features, pair sample, model parameters, and correction rule as
  `RT-1219`;
- row class targets permuted within the outer training population before fit.

`RT-1222` single-best killed-arm control:

- for each outer fold `f`, compute train-fold marginal integration for every
  one of the twelve sensor streams using only folds `{0,1,2,3,4} \ {f}`;
- choose the highest train-fold `marginal_vs_clone`, ties by sensor order;
- validation score for fold `f` is that chosen calibrated sensor stream.

Family-holdout controls:

- for each of the twelve families, refit the same arbiter after removing that
  family from the action set and feature deltas;
- evaluate with the same screen/evaluation functions;
- used to check whether one family explains the candidate.

## Evaluation Contract

For `RT-1219` and controls report:

- standalone/correction diagnostics;
- fold-0 E0, E1, E2;
- `marginal_vs_clone`;
- `gain_vs_RT600`;
- repairs, damage, and net pair flow for whole fold, dominant cell,
  mature-vs-never, and mature-vs-prebreak;
- RT600-right damage rate;
- contributing sensor-family counts and action shares;
- family-holdout results;
- shuffled-target result;
- single-best-arm control;
- runtime and peak memory if available.

Pair-flow evaluation uses the deterministic 64-pair-per-t sample with seed
`20260825`, matching the first-sweep repair-reservoir synthesis convention.

Repair-reservoir reporting uses the first-sweep synthesis baselines:

- dominant RT600-wrong repair reservoir: `14,868 / 16,193 = 91.8%`
- dominant RT600-right any-damage baseline: `28,505 / 34,347 = 83.0%`

## Binding SS-01 Screen Gate

`SS-01` passes the screen only if all of the following are true on fold `0`:

- `marginal_vs_clone >= +0.0015`;
- dominant-cell net pair lift is positive;
- damage rate on RT600-right dominant pairs is below half of `0.830`;
- at least two sensor families contribute in fold-pure validation;
- no control matches within `+0.0005`;
- causal-prefix legality and fold-purity audits pass.

If any mandatory gate fails, `SS-01 = KILL`. No SS-01b, no retuning, no
alternative threshold, no sensor subset search.

If the screen passes, stop breadth execution and run the preregistered SS-01
5-fold confirmation before SS-02.

## Runtime Cap

The SS-01 fold-0 screen family, including controls and family holdouts, is
capped at 4 wall-clock hours. If the cap would be exceeded before family
holdouts complete, finish candidate plus the three mandatory controls first and
record any omitted holdout as a runtime-limited diagnostic omission.

## Safety

This execution will not read lockbox rows for selection, will not read test or
Crunch hidden data, will not touch `production/rt600`, will not modify
`research/current`, will not submit to Crunch, and will not force-push.
