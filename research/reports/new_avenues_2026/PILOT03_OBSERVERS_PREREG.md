# PILOT 3 / IM3 -- INDIVIDUAL OBSERVER RESIDUALS PREREGISTRATION

Written 2026-08-24 on branch `research/new-avenues-pilots-2026`, at
checkpoint `410acc7`, after Pilot 9(i) was scored, killed, committed, pushed,
and Research hygiene passed. No Pilot 3 observer validation score exists at
preregistration time.

This file covers the remaining Pilot 3 / IM3 observer work requested after
Pilots 7, 10, and 9(i). It does not rerun the earlier `RT-1201` IM2 dwell
pilot.

## Hypothesis

The current bank contains scalar one-step residual monitors (`m04_resid`) and
AR-coefficient likelihood ratios (`m12_rdep` in prior work), but it does not
represent each series relative to a frozen multi-state historical observer.
Pilot 3 tests whether a per-series history-fitted model, never re-estimated
online, exposes break information through:

* Kalman normalized innovations and innovation whiteness (C2);
* Hankel-DMD reconstruction residuals, effective rank, and historical-subspace
  deviation (C1).

The arms are scored separately. No Kalman+DMD union is allowed before either
arm earns confirmation.

## Reserved IDs

| ID | candidate |
|---|---|
| `RT-1214` | frozen AR(2)-state Kalman/NIS observer arm |
| `RT-1215` | frozen Hankel-DMD observer arm |

No new scored m04 control ID is allocated. The Mode-A base bank already
contains `m04_resid`, so the binding marginal asks whether each observer adds
value beyond existing residual monitors and beyond the seed clone. Reports must
still discuss this boundary explicitly.

## Exact Inputs

* Raw history and online arrays from `cache/store`, dev folds only.
* History-fitted standardization and AR(2) parameters via `HistParams`.
* RT-600 and RT-401 OOF streams only for diagnostics and ensemble integration.
* No true tau, labels, final online length, future rows, lockbox/test rows, or
  production branch artifacts in feature construction.
* Pilot 3 observers do not directly recompute the production `m07_bayes` stream
  engine or its known `bo_p_lt25_z` batch/stream parity issue. The Mode-A
  screen uses the existing cached legal base bank plus independent observer
  features.

## Arm A: Frozen Kalman/NIS Observer (`RT-1214`)

Historical fit:

1. Standardize history using `HistParams`.
2. Fit fixed AR(2) coefficients on standardized history using the existing
   history-only AR fit.
3. Use a two-state AR companion model:

   `state_t = [z_t, z_{t-1}]`

   with transition matrix

   `F = [[phi1, phi2], [1, 0]]`.

4. Observation matrix is `H = [1, 0]`.
5. Choose process/measurement noise by a deterministic history-only grid:
   `q in {0.01, 0.05, 0.20}` and `r in {0.05, 0.20, 1.00}`. Select the pair
   minimizing one-step Gaussian negative log likelihood on historical
   standardized observations after the first 32 history points.
6. Freeze `phi`, `q`, and `r`. They are never updated online.

Online state:

Run a standard Kalman filter forward from the last historical state. At row
`t`, before updating on `z_t`, compute innovation `v_t`, innovation variance
`S_t`, normalized innovation `e_t = v_t / sqrt(S_t)`, and
`NIS_t = e_t^2`. Then update only the filter state/covariance with fixed
parameters.

`RT-1214` emits exactly six columns:

1. `nis_log`: `log1p(NIS_t)`
2. `nis_cum_z`: `(sum_{u<=t} NIS_u - (t+1)) / sqrt(2*(t+1))`
3. `nis_w32_z`: trailing-32 NIS excess z, NaN until 32 online rows exist
4. `nis_w64_z`: trailing-64 NIS excess z, NaN until 64 online rows exist
5. `innov_acf_w32_z`: `sqrt(32) * lag-1 acf` of normalized innovations, NaN
   until 32 online rows exist
6. `innov_acf_w64_z`: `sqrt(64) * lag-1 acf` of normalized innovations, NaN
   until 64 online rows exist

This arm tests filter consistency residuals; it is not an adaptive variance
model. The filter state updates online, but parameters and noise are frozen.

## Arm B: Frozen Hankel-DMD Observer (`RT-1215`)

Historical fit:

1. Standardize history using `HistParams`.
2. Build delay vectors of length `d = 16` from the standardized historical
   stream.
3. Fit one-step DMD from consecutive historical delay vectors using truncated
   SVD rank `r = 4`.
4. Store the full frozen operator `A`, the rank-4 historical dominant subspace,
   and historical null summaries for the preregistered residual/effective-rank
   statistics.

Online state:

For each online row, form the current length-16 delay vector using the
available history tail and online prefix. Compute:

* one-step reconstruction residual from `A * v_{t-1}` to `v_t`;
* five-step reconstruction residual from `A^5 * v_{t-5}` to `v_t`;
* residual fraction outside the historical rank-4 subspace;
* effective rank of the causal trailing Hankel window over the last 32 and 64
  delay vectors, with historical delay vectors used only as warm-up before
  enough online delay vectors exist.

Historical null calibration is median/MAD-sigma on the same statistic computed
over history only. There is no rank, delay, horizon, or window search.

`RT-1215` emits exactly six columns:

1. `dmd_resid_h1_z`
2. `dmd_resid_h5_z`
3. `dmd_subspace_resid_z`
4. `dmd_effrank_w32_z`
5. `dmd_effrank_w64_z`
6. `dmd_resid_ratio_h5_h1`

## Candidate Architecture

Pilot 3 observers are Mode A. Both arms use the existing legal feature bank:

`m00_core,m01_seq,m02_dist,m03_dyn,m04_resid,m06_loc,m07_bayes`

plus only the observer feature block under test.

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

For fold-0 ensemble integration, full fold-pure OOF support may be generated
for folds 1..4 only so the existing SCDF calibration path has candidate values
on calibration folds. No TS-AUC from folds 1..4 may be inspected or used in the
Pilot 3 observer screen decision.

## Causal Checks Before Training

Before the first observer score:

* run `harness.verify()` on both observer mechanisms;
* run registered feature-module prefix invariance at `atol=0.0`;
* explicitly test first-valid/NaN behavior for the Kalman 32/64 windows;
* explicitly test deterministic replay;
* explicitly test that mutating future online rows does not change earlier
  emitted rows;
* verify Kalman parameters are selected only from history;
* verify Hankel-DMD operator/subspace/nulls are fit only from history;
* run the new Pilot 3 observer targeted unit tests.

## RT600 Sentinel

Before interpreting Pilot 3 observers, reproduce:

* RT600 mean approximately `0.62581`
* RT600 pooled approximately `0.625627`
* dominant cell approximately `0.66428`
* fold-0 E0 RT600 approximately `0.63828`
* fold-0 E1 RT600 + RT-401 seed clone approximately `0.63859`

If these materially differ, stop and debug the evaluation path before scoring.

## Ensemble Integration

For both `RT-1214` and `RT-1215`, compute:

* E0: RT600
* E1: RT600 + exchangeable `RT-401` seed clone
* E2: RT600 + Pilot 3 observer arm
* `marginal_vs_clone = E2 - E1`

Use the same `harness.marginal` equal-weight integration path as the other
first-sweep pilots. No candidate-specific blend weight or global rank transform
is allowed.

## Metrics

For each scored arm report:

* whole-fold TS-AUC
* dominant-cell AUC
* mature-break vs never-break AUC
* mature-break vs pre-break AUC
* t buckets
* age buckets
* within-t rank correlation with RT600
* sampled pair repairs, damage, and net lift
* E0, E1, E2, and `marginal_vs_clone`
* feature-build runtime and training/runtime diagnostics

## Binding Gates

Primary gate for each arm:

* `marginal_vs_clone < +0.0010`: KILL
* `+0.0010` to `< +0.0020`: WEAK
* `+0.0020` to `< +0.0030`: INTERESTING
* `+0.0030` to `< +0.0050`: SERIOUS
* `+0.0050` to `< +0.0080`: MAJOR
* `>= +0.0080`: BREAKTHROUGH

Additional redundancy gate:

* For `RT-1215` Hankel-DMD, KILL if within-t rank correlation with RT600 in the
  dominant cell is greater than `0.85`, even if the marginal gate clears. This
  is the preregistered C1 redundancy guard from `new_avenues_2026.csv`.
* For `RT-1214` Kalman/NIS, within-t correlation is diagnostic, not binding,
  because C2 is expected to overlap more with existing innovation monitors.

## Continuation Gate

Run 5-fold confirmation for an arm only if:

* that arm's `marginal_vs_clone >= +0.0020`; and
* for `RT-1215`, the redundancy gate also passes.

If one arm clears continuation, stop the broad sweep and confirm that arm. If
both fail, file both and continue to Pilot 8.

## Runtime Ceiling

3 hours for implementation checks, feature build, both fold-0 arms,
diagnostics, and Pilot 3 observer report.

## Code Paths

Create:

* `src/sbr/features/m17_observers.py`
* `research/scripts/novel_streams/pilots/pilot03_observers.py`
* `tests/test_pilot03_observers.py`
* `research/reports/new_avenues_2026/pilot03_observers.md`
* `research/reports/new_avenues_2026/pilot03_observers.json`
