# PILOT 6 -- SPECTRAL IMPULSIVENESS CONTRAST PREREGISTRATION

Written 2026-08-24 on branch `research/new-avenues-pilots-2026`, after
Pilot 5 was committed and pushed at `42a5dc8` and the Research hygiene workflow
passed. No Pilot 6 validation score exists at preregistration time.

## Hypothesis

Excess band energy with low impulsiveness is more consistent with a sustained
regime change than with an isolated outlier burst. Excess band energy with high
spectral kurtosis / envelope concentration is more consistent with an impulse
that should not be promoted as a persistent structural break.

This tests F1/F6 from `NEW_AVENUES_2026.md`: spectral kurtosis plus a
heavy-tail-robust envelope-spectrum negentropy contrast. The existing bank
contains `m03_dyn` band powers, spectral entropy, centroid, and dominant band;
it does not encode whether the band energy arrives as one impulse or as a
sustained/repetitive structure.

## Reserved IDs

| ID | candidate |
|---|---|
| `RT-1206` | spectral impulsiveness contrast candidate |
| `RT-1207` | matched plain band-energy control |

## Exact Inputs

* Raw history and online arrays from `cache/store`, dev folds only.
* History-fitted standardization from `sbr.features.base.make_ctx`.
* Raw standardized stream `z = (x - historical_mean) / historical_sd`.
* No true tau, labels, final online length, future rows, or lockbox/test data in
  feature construction.
* RT-600 and RT-401 OOF streams are used only for diagnostics and ensemble
  integration.

## Frozen Frequency Bank

Use the same dyadic Goertzel frequencies as `m03_dyn`:

`{0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625}` cycles/sample.

Collapse them into exactly four preregistered band channels:

| band | frequencies |
|---|---|
| `nyq` | `{0.5}` |
| `high` | `{0.25}` |
| `mid` | `{0.125, 0.0625}` |
| `low` | `{0.03125, 0.015625}` |

No additional frequencies, wavelets, FFT bins, or tuned bands may be added
after seeing fold 0.

## Segment Envelope

For each band, build a causal segment-energy envelope:

1. Fixed segment length `SEG = 32`.
2. For every endpoint `u`, if `u + 1 < SEG`, envelope value is NaN.
3. Otherwise, for every frequency `f` in the band, compute the exact
   length-`SEG` trailing Goertzel quadratures over `z[u-SEG+1:u+1]`:
   `C_f = mean(z_j cos(2*pi*f*j))`, `S_f = mean(z_j sin(2*pi*f*j))`.
4. Band envelope energy at endpoint `u` is
   `sum_f (C_f^2 + S_f^2)`.

This is strictly one-sided and uses the same fixed-frequency quadrature idea as
`m03_dyn`.

## Online Window

At online row `t`, use the adaptive trailing half-prefix window from `m03_dyn`:

`W(t) = max(floor((t + 1) / 2), 1)`.

The statistic uses finite envelope endpoints in `[t - W(t) + 1, t]`. Let `L(t)`
be the number of finite segment-energy endpoints in that window. All Pilot 6
features are NaN until `L(t) >= 16`.

## Historical Null Calibration

For each series, band, statistic, and grid length `L in {16, 64, 256}`:

1. Build the same segment-energy envelope on the break-free history only.
2. Compute the statistic on contiguous blocks of `L` finite historical envelope
   endpoints.
3. If more than 700 historical blocks are available, use a deterministic fixed
   stride to cap the null sample at approximately 700.
4. Store median and robust scale
   `max(IQR/1.349, (p95-p05)/3.29, 1e-9)`.

For online row `t`, robust z is log-linear interpolation of the grid medians and
log-scales at `log(L(t))`, with linear extrapolation at the grid ends. This is
the same calibration style as `m03_dyn`; no fold-label calibration is used.

## Statistics

For each band and online row with `L(t) >= 16`:

* `energy_raw`: `log(mean(envelope) + 1e-12)`.
* `energy_z`: robust z of `energy_raw` against the historical null.
* `sk_raw`: excess kurtosis of the un-clipped envelope values in the window,
  `m4 / variance^2 - 3`, with variance floor `1e-12`.
* `sk_z`: robust z of `sk_raw` against the historical null.
* `negent_raw`: robust envelope-spectrum negentropy. Clip envelope values at
  that series/band's historical 99th percentile, then compute
  `1 - H(p) / log(L)`, where `p_i = clipped_env_i / sum(clipped_env)`.
* `negent_z`: robust z of `negent_raw` against the historical null.

High `sk_z` or high `negent_z` means impulsive/concentrated band energy.

## Emitted Features

`RT-1206` emits exactly eight impulsiveness-contrast columns:

* for each band: `contrast_sk_{band} = energy_z * (-sk_z)`
* for each band: `contrast_negent_{band} = energy_z * (-negent_z)`

Positive contrast means high band energy with unusually low impulsiveness.

`RT-1207` emits exactly four plain-energy control columns:

* for each band: `energy_z_{band}`

The control receives no spectral-kurtosis, negentropy, product contrast, or
interaction columns.

## Candidate And Control Architecture

Pilot 6 is Mode A. Both arms use the existing legal feature bank

`m00_core,m01_seq,m02_dist,m03_dyn,m04_resid,m06_loc,m07_bayes`

plus only the Pilot 6 feature block under test.

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
on calibration folds. No TS-AUC from folds 1..4 may be inspected or used in
the Pilot 6 screen decision.

## Causal Checks Before Training

Before the first Pilot 6 score:

* run `harness.verify()` on the contrast and energy-control mechanisms;
* run registered feature-module prefix invariance at `atol=0.0`;
* explicitly test first-valid/NaN behavior at `SEG=32` and `L>=16`;
* explicitly test that mutating future online rows does not change earlier
  emitted rows;
* test deterministic replay;
* run `pytest -q tests/test_novel_streams_harness.py`;
* run the new Pilot 6 targeted unit tests.

Do not weaken existing tests.

## RT600 Sentinel

Before interpreting Pilot 6, reproduce:

* RT600 mean approximately `0.62581`
* RT600 pooled approximately `0.625627`
* dominant cell approximately `0.66428`
* fold-0 E0 RT600 approximately `0.63828`
* fold-0 E1 RT600 + RT-401 seed clone approximately `0.63859`

If these materially differ, stop and debug the evaluation path before scoring
the pilot.

## Ensemble Integration

For both `RT-1206` and `RT-1207`, compute:

* E0: RT600
* E1: RT600 + exchangeable `RT-401` seed clone
* E2: RT600 + Pilot 6 candidate/control
* `marginal_vs_clone = E2 - E1`

Use the same `harness.marginal` equal-weight integration path as Pilots 2-5.
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
* feature-build runtime and training/runtime diagnostics

## Binding Gates

Primary gate for the contrast arm:

* `marginal_vs_clone < +0.0010`: KILL
* `+0.0010` to `< +0.0020`: WEAK
* `+0.0020` to `< +0.0030`: INTERESTING
* `+0.0030` to `< +0.0050`: SERIOUS
* `+0.0050` to `< +0.0080`: MAJOR
* `>= +0.0080`: BREAKTHROUGH

Second binding scientific gate:

* KILL if `RT-1207 marginal_vs_clone >= RT-1206 marginal_vs_clone`.
* Also KILL if
  `RT-1206 marginal_vs_clone - RT-1207 marginal_vs_clone < +0.0005`.

This means generic extra band-energy alpha is not evidence for F1/F6
impulsiveness.

## Continuation Gate

Run 5-fold confirmation only if both conditions hold:

* `RT-1206 marginal_vs_clone >= +0.0020`
* `RT-1206 marginal_vs_clone - RT-1207 marginal_vs_clone >= +0.0005`

Run paired/series bootstrap only after a positive five-fold result with
`marginal_vs_clone >= +0.0030` and positive sign on at least 4/5 folds.

## Runtime Ceiling

2.5 hours for implementation checks, feature build, both fold-0 arms,
diagnostics, and Pilot 6 report.

## Code Paths

Create:

* `src/sbr/features/m14_spectral_impulse.py`
* `research/scripts/novel_streams/pilots/pilot06_spectral_impulse.py`
* `tests/test_pilot06_spectral_impulse.py`
* `research/reports/new_avenues_2026/pilot06_spectral_impulse.md`
* `research/reports/new_avenues_2026/pilot06_spectral_impulse.json`
