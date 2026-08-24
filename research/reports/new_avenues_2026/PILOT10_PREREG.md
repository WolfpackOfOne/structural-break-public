# PILOT 10 -- JOINT SIZE-DURATION RARITY PREREGISTRATION

Written 2026-08-24 on branch `research/new-avenues-pilots-2026`, at
checkpoint `7acd89b`, after Pilot 7 was scored, killed, committed, pushed, and
Research hygiene passed. No Pilot 10 validation score exists at
preregistration time.

## Hypothesis

Pilot 3 / `RT-1201` showed that marginal dwell/run-null information was not
sufficient to clear the ensemble gate. Pilot 10 tests the stronger I1 claim
from `NEW_AVENUES_2026.md`: the metric-relevant event may be the joint rarity
of deviation magnitude and persistence duration,

`P(|deviation| >= s AND duration >= d | historical null)`.

Existing features price magnitude at fixed lengths and dwell at fixed
thresholds. This pilot asks whether the non-factorized joint event carries
additional same-time ranking information.

## Reserved IDs

| ID | candidate |
|---|---|
| `RT-1210` | joint size-duration rarity candidate |
| `RT-1211` | matched dwell-only rarity control |

`RT-1211` is an official scored control row because `RT-1201` was a
direct-score scalar, not the same Mode-A LightGBM feature-addition architecture
used by Pilots 5-7. The existing `RT-1201` result remains a secondary reference.

## Exact Inputs

* Raw history and online arrays from `cache/store`, dev folds only.
* History-fitted AR(2) residual-square stream, using `HistParams` and
  `ar_filter_causal` as in Pilots 3 and 5.
* RT-600 and RT-401 OOF streams are used only for diagnostics and ensemble
  integration.
* No true tau, labels, final online length, future rows, or lockbox/test data
  in feature construction.
* Pilot 10 does not directly recompute the production `m07_bayes` stream engine
  or its known `bo_p_lt25_z` batch/stream parity issue. The Mode-A screen uses
  the existing cached legal base bank plus independent new residual-square
  streams.

## Residual-Square Excursion State

For each series independently:

1. Fit history-only robust normalization and AR(2) residual parameters using
   `HistParams`.
2. Compute standardized AR residuals on history and online causally.
3. Square the residuals.
4. For windows `w in {32, 64, 128}`, compute the trailing rolling mean on the
   residual-square stream.
5. On the historical rolling statistic, set the null center to the median and
   the excursion band to the q90 absolute deviation from that median.
6. A rolling-stat endpoint is hot when
   `abs(stat - median) > q90_abs_deviation`.
7. Excursion size `s` is the maximum positive excess
   `abs(stat - median) - q90_abs_deviation` inside the current contiguous hot
   run. Excursion duration `d` is the current contiguous hot-run length in
   rolling-stat endpoints.

Rows before the first rolling-stat endpoint for a window (`t < w - 1`) emit
NaN for that window. Once the endpoint exists, a non-hot state emits zero
surprise, not NaN.

## Historical Joint Null

Use guaranteed break-free history only.

For each window `w`, form all historical rolling-stat endpoints and their
causal endpoint states:

* live hot-run duration ending at that historical endpoint;
* live maximum excess size inside that ending run.

This endpoint table is the empirical joint null for the online live state.
It estimates the fraction of historical opportunity endpoints whose excursion
state was at least as large and at least as long as the online query:

`P_hat(s,d) = count(history_peak_excess >= s AND history_live_run >= d) / n_endpoints`.

The probability is floored at `1 / (2 * n_endpoints)`, matching the
`NullCal.surprise` floor convention. No independence approximation
`P(size) * P(duration)` is allowed.

For non-hot online rows, the query is `(s=0,d=0)` and the emitted live surprise
is exactly zero.

## Online State And Emitted Candidate Features

At online row `t`, compute:

* the live joint surprise for the current excursion state;
* the running-max joint surprise, defined prospectively as the maximum live
  joint surprise observed through row `t`.

Surprise is one-sided:

`joint_surprise = -log10(max(P_hat(s,d), 1/(2*n_endpoints)))`.

`RT-1210` emits exactly six columns:

1. `joint_live_w32`
2. `joint_max_w32`
3. `joint_live_w64`
4. `joint_max_w64`
5. `joint_live_w128`
6. `joint_max_w128`

No extra windows, thresholds, z-score calibrations, interactions, or
post-score transforms may be added after fold 0 is observed.

## Dwell-Only Control

The matched A1/dwell control uses the same residual-square stream, windows,
q90 excursion band, endpoint table, probability floor, LightGBM architecture,
fold, seed, row budget, and base bank. It removes size from the event:

`P_hat_dwell(d) = count(history_live_run >= d) / n_endpoints`.

The emitted dwell surprise is:

`dwell_surprise = -log10(max(P_hat_dwell(d), 1/(2*n_endpoints)))`.

`RT-1211` emits exactly six columns:

1. `dwell_live_w32`
2. `dwell_max_w32`
3. `dwell_live_w64`
4. `dwell_max_w64`
5. `dwell_live_w128`
6. `dwell_max_w128`

This is the binding control for generic dwell/run-length alpha under the same
Mode-A training path. Existing `RT-1201` diagnostics are reported as a
secondary reference, but are not substituted for this matched control.

## Candidate And Control Architecture

Pilot 10 is Mode A. Both arms use the existing legal feature bank

`m00_core,m01_seq,m02_dist,m03_dyn,m04_resid,m06_loc,m07_bayes`

plus only the Pilot 10 feature block under test.

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
Pilot 10 screen decision.

## Causal Checks Before Training

Before the first Pilot 10 score:

* run `harness.verify()` on the joint-rarity candidate and dwell-only control;
* run registered feature-module prefix invariance at `atol=0.0`;
* explicitly test first-valid/NaN behavior at windows `32,64,128`;
* explicitly test that mutating future online rows does not change earlier
  emitted rows;
* test deterministic replay;
* test that the joint surprise is never below the matched dwell surprise for
  the same online state;
* run `pytest -q tests/test_novel_streams_harness.py`;
* run the new Pilot 10 targeted unit tests.

Do not weaken existing tests.

## RT600 Sentinel

Before interpreting Pilot 10, reproduce:

* RT600 mean approximately `0.62581`
* RT600 pooled approximately `0.625627`
* dominant cell approximately `0.66428`
* fold-0 E0 RT600 approximately `0.63828`
* fold-0 E1 RT600 + RT-401 seed clone approximately `0.63859`

If these materially differ, stop and debug the evaluation path before scoring
the pilot.

## Ensemble Integration

For both `RT-1210` and `RT-1211`, compute:

* E0: RT600
* E1: RT600 + exchangeable `RT-401` seed clone
* E2: RT600 + Pilot 10 candidate/control
* `marginal_vs_clone = E2 - E1`

Use the same `harness.marginal` equal-weight integration path as Pilots 2-7.
No candidate-specific blend weight or global rank transform is allowed.

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
* candidate-control dominant-cell AUC gap
* feature-build runtime and training/runtime diagnostics

## Binding Gates

Primary gate for the joint-rarity arm:

* `marginal_vs_clone < +0.0010`: KILL
* `+0.0010` to `< +0.0020`: WEAK
* `+0.0020` to `< +0.0030`: INTERESTING
* `+0.0030` to `< +0.0050`: SERIOUS
* `+0.0050` to `< +0.0080`: MAJOR
* `>= +0.0080`: BREAKTHROUGH

Second binding scientific gate:

* KILL if `RT-1211 marginal_vs_clone >= RT-1210 marginal_vs_clone`.
* KILL if `RT-1210 dominant-cell AUC <= RT-1211 dominant-cell AUC`.
* KILL if
  `RT-1210 marginal_vs_clone - RT-1211 marginal_vs_clone < +0.0005`.

This means a generic dwell/run-length benefit is not evidence for I1 joint
size-duration rarity.

## Continuation Gate

Run 5-fold confirmation only if all conditions hold:

* `RT-1210 marginal_vs_clone >= +0.0020`
* `RT-1210 marginal_vs_clone - RT-1211 marginal_vs_clone >= +0.0005`
* `RT-1210 dominant-cell AUC > RT-1211 dominant-cell AUC`

Run paired/series bootstrap only after a positive five-fold result with
`marginal_vs_clone >= +0.0030` and positive sign on at least 4/5 folds.

## Runtime Ceiling

2.5 hours for implementation checks, feature build, both fold-0 arms,
diagnostics, and Pilot 10 report.

## Code Paths

Create:

* `src/sbr/features/m16_joint_rarity.py`
* `research/scripts/novel_streams/pilots/pilot10_joint_rarity.py`
* `tests/test_pilot10_joint_rarity.py`
* `research/reports/new_avenues_2026/pilot10_joint_rarity.md`
* `research/reports/new_avenues_2026/pilot10_joint_rarity.json`
