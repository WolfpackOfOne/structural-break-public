# PILOT 5 -- SCALE-SURVIVAL COARSE-GRAINING PREREGISTRATION

Written 2026-08-24 on branch `research/new-avenues-pilots-2026`, at
checkpoint `c48bec7`, before any Pilot 5 validation score was produced.

The novel-stream harness cleanup is already merged on this branch. Pilots 1-4
are final screen results and are not retuned here.

## Hypothesis

A persistent break should remain abnormal after causal temporal
coarse-graining, while a transient disturbance should fade as the scale grows.
The existing RT-600 feature bank already contains many per-window statistics;
this pilot tests whether explicit cross-scale functionals add marginal
competition value beyond the individual scale values themselves.

## Information Channel

E1/E4 from `NEW_AVENUES_2026.md`: scale-survival count plus a coarse-graining
decay exponent on a residual evidence stream.

## Reserved IDs

| ID | candidate |
|---|---|
| `RT-1204` | cross-scale summary candidate |
| `RT-1205` | six individual per-scale surprise control |

## Exact Inputs

* Raw history and online arrays from `cache/store`, dev folds only.
* History-fitted AR(2) residual stream using `sbr.transforms.HistParams`,
  `_ar_resid`, and `ar_filter_causal`.
* Residual evidence stream is squared standardized AR(2) residual:
  `res_sq = (residual / historical_residual_sigma)^2`.
* No true tau, labels, final online length, future rows, or lockbox/test data in
  feature construction.
* RT-600 and RT-401 OOF streams are used only for diagnostics and ensemble
  integration.

## Frozen Scale Bank

Use exactly `b in {1, 2, 4, 8, 16, 32}`. No additional scales may be added
after seeing fold 0.

## Null Calibration

For each series and each scale `b`:

1. Build the historical residual-square stream from history only.
2. Form the historical same-scale null from all trailing length-`b`
   contiguous block means of that historical stream.
3. At online row `t`, form the causal trailing online block mean over
   `online[t-b+1:t+1]`. If `t + 1 < b`, that scale emits NaN.
4. Let `m_b` be the number of historical null values. Let `r_b(t)` be the count
   of historical null values `<=` the online block mean.
5. Define empirical percentile
   `p_b(t) = (r_b(t) + 0.5) / (m_b + 1.0)`.
6. Define two-sided tail probability
   `tail2_b(t) = max(2 * min(p_b(t), 1 - p_b(t)), 1 / (m_b + 1.0))`.
7. Define nonnegative surprise
   `surprise_b(t) = -log10(tail2_b(t))`.

Each scale has its own historical null. There is no shared all-scale null and
no label-fitted calibration.

## Thresholds

Thresholds are fixed prospectively:

* `q1 = 1.301029995664` (`-log10(0.05)`, two-sided 5% historical-null tail)
* `q2 = 2.0` (`-log10(0.01)`, two-sided 1% historical-null tail)

These thresholds are theoretical tail-evidence thresholds, not fold-0 tuned
quantiles.

## Online Semantics

All coarse-graining is one-sided. Row `t` may use only online rows `<= t`.

For `t + 1 < b`, scale `b` is NaN. Summary counts and largest-scale codes are
computed over finite scales only. `largest_surviving_scale_*` is an integer
code: 0 means no finite scale survives the threshold, 1 means `b=1`, 2 means
`b=2`, ..., 6 means `b=32`. `slope_logscale` is NaN until at least two scales
are finite.

The same online prefix must produce bit-identical rows whether the full online
sequence later has length 50 or 1000.

## Emitted Features

`RT-1204` emits exactly five cross-scale summary columns:

1. `survival_count_q05`: count of finite scales with
   `surprise_b > 1.301029995664`.
2. `survival_count_q01`: count of finite scales with `surprise_b > 2.0`.
3. `largest_surviving_scale_q05`: largest surviving scale code at q05.
4. `largest_surviving_scale_q01`: largest surviving scale code at q01.
5. `slope_logscale`: OLS slope of `surprise_b` against `log2(b)` over finite
   scales.

`RT-1205` emits exactly six individual-scale control columns:

`surprise_b1`, `surprise_b2`, `surprise_b4`, `surprise_b8`, `surprise_b16`,
`surprise_b32`.

The control receives no counts, largest-scale summaries, slopes, interactions,
or fitted thresholds.

## Candidate And Control Architecture

Pilot 5 is Mode A. Both arms use the existing legal feature bank

`m00_core,m01_seq,m02_dist,m03_dyn,m04_resid,m06_loc,m07_bayes`

plus only the Pilot 5 feature block under test.

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
* sample mode: uniform

Only the Pilot 5 added feature block differs between `RT-1204` and `RT-1205`.

## Causal Checks Before Training

Before the first score:

* run `harness.verify()` on the new summary and individual mechanisms;
* run prefix-invariance checks for the registered feature modules at
  `atol=0.0`;
* explicitly test first-valid/NaN behavior for all six scales;
* explicitly test that mutating future online rows does not change earlier
  emitted rows;
* test deterministic replay;
* run `pytest -q tests/test_novel_streams_harness.py`;
* run the new Pilot 5 targeted unit tests.

Do not weaken existing harness tests.

## RT600 Sentinel

Before interpreting Pilot 5, reproduce the evaluation anchor:

* RT600 mean approximately `0.62581`
* RT600 pooled approximately `0.625627`
* dominant cell approximately `0.66428`
* fold-0 E0 RT600 approximately `0.63828`
* fold-0 E1 RT600 + RT-401 seed clone approximately `0.63859`

If these materially differ, stop and debug the evaluation path before scoring
the pilot.

## Ensemble Integration

For both `RT-1204` and `RT-1205`, compute:

* E0: RT600
* E1: RT600 + exchangeable `RT-401` seed clone
* E2: RT600 + Pilot 5 candidate/control
* `marginal_vs_clone = E2 - E1`

Use the same `harness.marginal` equal-weight integration path as Pilots 2-4.
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

Primary gate for the summary arm:

* `marginal_vs_clone < +0.0010`: KILL
* `+0.0010` to `< +0.0020`: WEAK
* `+0.0020` to `< +0.0030`: INTERESTING
* `+0.0030` to `< +0.0050`: SERIOUS
* `+0.0050` to `< +0.0080`: MAJOR
* `>= +0.0080`: BREAKTHROUGH

Second binding scientific gate:

* KILL the scale-survival hypothesis if
  `RT-1205 marginal_vs_clone >= RT-1204 marginal_vs_clone`.
* Also KILL if the summary-control gap is effectively indistinguishable,
  defined prospectively as
  `RT-1204 marginal_vs_clone - RT-1205 marginal_vs_clone < +0.0005`.

This means generic per-scale alpha is not evidence for E1/E4 scale survival.

## Continuation Gate

Run 5-fold confirmation only if both conditions hold:

* `RT-1204 marginal_vs_clone >= +0.0020`
* `RT-1204 marginal_vs_clone - RT-1205 marginal_vs_clone >= +0.0005`

Run paired/series bootstrap only after a positive five-fold result with
`marginal_vs_clone >= +0.0030` and positive sign on at least 4/5 folds.

If `RT-1204 marginal_vs_clone < +0.0010`, kill immediately, document, commit,
push, verify CI, and continue to Pilot 6.

## Runtime Ceiling

2.5 hours for implementation checks, feature build, both fold-0 arms,
diagnostics, and Pilot 5 report.

## Code Paths

Create:

* `src/sbr/features/m13_scale_survival.py`
* `research/scripts/novel_streams/pilots/pilot05_scale_survival.py`
* `tests/test_pilot05_scale_survival.py`
* `research/reports/new_avenues_2026/pilot05_scale_survival.md`
* `research/reports/new_avenues_2026/pilot05_scale_survival.json`
