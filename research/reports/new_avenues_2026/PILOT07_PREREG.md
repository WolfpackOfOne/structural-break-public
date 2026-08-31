# PILOT 7 -- ORDINAL TRANSITION DIVERGENCE AND TIME IRREVERSIBILITY PREREGISTRATION

Written 2026-08-24 on branch `research/new-avenues-pilots-2026`, at
checkpoint `03b1ace`, before any Pilot 7 validation score was produced.

Pilots 1-6 are final screen results and are not retuned here. The branch is
synced with `origin/research/new-avenues-pilots-2026`; `origin/research/current`
at `aca2c4f` is already an ancestor of this branch.

## Hypothesis

A dependence break can preserve the marginal distribution and even the
order-3 permutation-entropy level while changing either:

* the transition law between ordinal patterns, or
* the forward-vs-reversed temporal asymmetry of the stream.

This tests L3/L4 from `NEW_AVENUES_2026.md`. The existing bank contains
`m03_dyn` permutation entropy / ordinal complexity, but not the ordinal
transition matrix and not a time-reversal asymmetry statistic.

## Reserved IDs

| ID | candidate |
|---|---|
| `RT-1208` | ordinal transition divergence plus irreversibility candidate |
| `RT-1209` | matched permutation-entropy-only control |

## Exact Inputs

* Raw history and online arrays from `cache/store`, dev folds only.
* History-fitted standardized stream from `sbr.features.base.make_ctx`:
  `z = (x - historical_mean) / historical_sd`.
* No true tau, labels, final online length, future rows, or lockbox/test data
  in feature construction.
* RT-600 and RT-401 OOF streams are used only for diagnostics and ensemble
  integration.
* Pilot 7 does not recompute the production `m07_bayes` stream engine or its
  known `bo_p_lt25_z` batch/stream parity issue. The Mode-A LightGBM screen
  uses the existing cached legal base bank, as Pilots 5-6 did, plus independent
  new ordinal features.

## Order-3 Ordinal Pattern Stream

At online endpoint `t`, form the triple `(z[t-2], z[t-1], z[t])`, using the
last two historical standardized values as causal warm-up for `t < 2`.

Tie handling is exactly the `m03_dyn` convention:

`code = 4 * 1[a > b] + 2 * 1[a > c] + 1 * 1[b > c]`

for triple `(a,b,c)`. The six realizable codes are fixed as:

`(0, 1, 3, 4, 6, 7)`.

Ties resolve through the non-strict comparisons above; no random tie-breaking,
jitter, or rank averaging is allowed.

## Transition Divergence

Build a 6x6 ordinal transition-count matrix over consecutive pattern endpoints.

Historical reference:

1. Compute historical order-3 pattern codes on the guaranteed break-free
   history.
2. Compute all consecutive historical transitions.
3. Form the full-history joint transition distribution over 36 ordered pairs
   with fixed pseudocount `alpha = 0.5` on every cell.

Online state:

1. At row `t`, define the boundary-safe transition from the immediately prior
   pattern endpoint to the current online endpoint. For `t = 0`, the previous
   endpoint is the last historical pattern endpoint.
2. Maintain two causal transition windows:
   * expanding prefix: all online transitions `0..t`;
   * trailing half-prefix: last `max(floor((t + 1) / 2), 1)` online transitions.
3. For each window, form the smoothed joint transition distribution using the
   same `alpha = 0.5`.
4. The raw divergence is
   `KL(P_online || P_hist)` over the 36-cell joint transition distribution.

All transition-divergence features are NaN until at least 16 online transitions
are present in the corresponding window.

## Matched-Count Transition Null

For each series and grid count `L in {16, 64, 256}`:

1. Slide a contiguous length-`L` transition block across historical transitions.
2. Compute the same smoothed `KL(P_block || P_hist)`.
3. If more than 700 historical blocks are available, use a deterministic fixed
   stride to cap the null sample at approximately 700.
4. Store median and robust scale
   `max(IQR/1.349, (p95-p05)/3.29, 1e-9)`.

Online transition KL z-scores use log-linear interpolation of the grid medians
and log-scales at `log(L_online)`, with linear extrapolation at the grid ends.

## Time-Irreversibility Statistic

Use the signed Ramsey-Rothman increment-asymmetry statistic on standardized
increments:

`RR = mean(diff(z)^3) / (mean(diff(z)^2)^(3/2) + 1e-12)`.

Historical increments are `hist_z[i] - hist_z[i-1]`. The online increment at
row `t = 0` is `online_z[0] - hist_z[-1]`; later rows use
`online_z[t] - online_z[t-1]`.

Online state uses the same expanding-prefix and trailing-half-prefix windows as
the transition block. All irreversibility features are NaN until at least 16
online increments are present in the corresponding window.

The historical null uses contiguous increment blocks at `L in {16,64,256}` and
the same median/robust-scale interpolation rule as the transition KL null.

## Entropy-Only Control

The control arm computes only the existing ordinal information type:
permutation entropy of the order-3 pattern distribution.

For expanding and trailing-half-prefix pattern windows:

`PE = -sum_i p_i log(p_i) / log(6)`,

with pseudocount `alpha = 0.5` over the six pattern codes. Historical null
calibration uses matched-count contiguous historical pattern blocks at
`L in {16,64,256}`.

The control receives no transition-matrix features, no irreversibility
features, no interactions, and no extra ordinal orders.

## Emitted Features

`RT-1208` emits exactly eight columns:

1. `trans_kl_exp`
2. `trans_kl_exp_z`
3. `trans_kl_half`
4. `trans_kl_half_z`
5. `rr_asym_exp`
6. `rr_asym_exp_z`
7. `rr_asym_half`
8. `rr_asym_half_z`

`RT-1209` emits exactly four entropy-only control columns:

1. `pe_exp`
2. `pe_exp_z`
3. `pe_half`
4. `pe_half_z`

No additional windows, ordinal orders, visibility-graph features, or alternate
divergences may be added after fold 0 is observed.

## Candidate And Control Architecture

Pilot 7 is Mode A. Both arms use the existing legal feature bank

`m00_core,m01_seq,m02_dist,m03_dyn,m04_resid,m06_loc,m07_bayes`

plus only the Pilot 7 feature block under test.

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
Pilot 7 screen decision.

## Causal Checks Before Training

Before the first Pilot 7 score:

* run `harness.verify()` on the transition/asymmetry candidate and entropy
  control mechanisms;
* run registered feature-module prefix invariance at `atol=0.0`;
* explicitly test first-valid/NaN behavior for the 16-count minimum and the
  expanding/half-prefix windows;
* explicitly test that mutating future online rows does not change earlier
  emitted rows;
* test deterministic replay;
* run `pytest -q tests/test_novel_streams_harness.py`;
* run the new Pilot 7 targeted unit tests.

Do not weaken existing tests.

## RT600 Sentinel

Before interpreting Pilot 7, reproduce:

* RT600 mean approximately `0.62581`
* RT600 pooled approximately `0.625627`
* dominant cell approximately `0.66428`
* fold-0 E0 RT600 approximately `0.63828`
* fold-0 E1 RT600 + RT-401 seed clone approximately `0.63859`

If these materially differ, stop and debug the evaluation path before scoring
the pilot.

## Ensemble Integration

For both `RT-1208` and `RT-1209`, compute:

* E0: RT600
* E1: RT600 + exchangeable `RT-401` seed clone
* E2: RT600 + Pilot 7 candidate/control
* `marginal_vs_clone = E2 - E1`

Use the same `harness.marginal` equal-weight integration path as Pilots 2-6.
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

Primary gate for the transition/asymmetry arm:

* `marginal_vs_clone < +0.0010`: KILL
* `+0.0010` to `< +0.0020`: WEAK
* `+0.0020` to `< +0.0030`: INTERESTING
* `+0.0030` to `< +0.0050`: SERIOUS
* `+0.0050` to `< +0.0080`: MAJOR
* `>= +0.0080`: BREAKTHROUGH

Second binding scientific gate:

* KILL if `RT-1209 marginal_vs_clone >= RT-1208 marginal_vs_clone`.
* Also KILL if
  `RT-1208 marginal_vs_clone - RT-1209 marginal_vs_clone < +0.0005`.

This means permutation-entropy-only alpha is not evidence for new ordinal
transition or irreversibility information.

## Continuation Gate

Run 5-fold confirmation only if both conditions hold:

* `RT-1208 marginal_vs_clone >= +0.0020`
* `RT-1208 marginal_vs_clone - RT-1209 marginal_vs_clone >= +0.0005`

Run paired/series bootstrap only after a positive five-fold result with
`marginal_vs_clone >= +0.0030` and positive sign on at least 4/5 folds.

## Runtime Ceiling

2.5 hours for implementation checks, feature build, both fold-0 arms,
diagnostics, and Pilot 7 report.

## Code Paths

Create:

* `src/sbr/features/m15_ordinal_irrev.py`
* `research/scripts/novel_streams/pilots/pilot07_ordinal_irreversibility.py`
* `tests/test_pilot07_ordinal_irrev.py`
* `research/reports/new_avenues_2026/pilot07_ordinal_irreversibility.md`
* `research/reports/new_avenues_2026/pilot07_ordinal_irreversibility.json`
