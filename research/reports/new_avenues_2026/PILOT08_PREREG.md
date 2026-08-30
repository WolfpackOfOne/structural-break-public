# PILOT 8 -- WEIGHTED CONFORMAL TEST MARTINGALE PREREGISTRATION

Written 2026-08-24 on branch `research/new-avenues-pilots-2026`, at
checkpoint `2429b6d`, after Pilot 3 / IM3 observers were scored, killed,
committed, pushed, and Research hygiene passed. No Pilot 8 H1/H2 validation
score exists at preregistration time.

## Hypothesis

The incumbent `m07_bayes` block already contains unweighted conformal
martingales on per-series historical PIT/conformity scores. Those martingales
can still compound on benign heavy-tail excursions, which is the false-positive
population called out by the New Avenues survey. Pilot 8 tests whether a
predictable weight, based only on the series' own recent tail rate relative to
its historical tail rate, reduces never-break damage while preserving useful
evidence for persistent shifts.

Pilot 8 also tests the H2 arm: a parameter-free e-value aggregation rule as an
eighth stream, without learned stacking or candidate-specific weights.

## Reserved IDs

| ID | candidate |
|---|---|
| `RT-1216` | weighted conformal test martingale feature arm |
| `RT-1217` | matched unweighted conformal test martingale feature control |
| `RT-1218` | parameter-free e-value aggregation direct-score arm |

## Exact Inputs

* Raw history and online arrays from `cache/store`, dev folds only.
* Per-series historical `HistParams` PIT values and online PIT values.
* RT-600 and `RT-401` OOF streams only for diagnostics and ensemble integration.
* Existing `m07_bayes` cached columns may be read by the H2 direct-score arm.
* No true tau, labels, final online length, future rows, lockbox/test rows, or
  production branch artifacts in feature construction.
* The Mode-A H1 screen uses the existing legal base bank, including
  `m07_bayes`; it therefore asks whether the new weighted martingale features
  add value beyond the incumbent unweighted martingales already in the base.

## H1 Weighted CTM Candidate (`RT-1216`)

Conformity stream:

1. Convert raw observations to historical PIT values with `HistParams.pit`.
2. Use centered PIT `c_t = u_t - 0.5`.
3. Define the same four bounded payoff families as the incumbent e-process
   block:
   * `tail`: `1[abs(c_t) > 0.4]`
   * `disp`: `c_t^2`
   * `dep`: `1[c_t * c_{t-1} > 0]`, warm-started from the last historical PIT
   * `vc`: `1[abs(c_t) > 0.3 and abs(c_{t-1}) > 0.3]`
4. Define the Vovk simple-mixture power martingale on two-sided conformal
   p-values `p_t = clip(2*min(u_t, 1-u_t), 1e-6, 1)`, with powers
   `{0.15,0.3,0.5,0.7,0.9}` and both `p_t` and `1-p_t`.

Predictable benign-tail weight:

* Tail-rate indicator for weighting is `1[abs(c_t) > 0.45]`.
* Historical tail rate `r0` is the historical mean of that indicator, clipped to
  `[0.02, 0.30]`.
* At online row `t`, compute only from prior online rows:
  `r32_t = mean(indicator over rows max(0,t-32)..t-1)` and
  `r128_t = mean(indicator over rows max(0,t-128)..t-1)`.
  Empty windows use `r0`.
* Recent benign-tail estimate is `r_t = 0.7*r32_t + 0.3*r128_t`.
* Betting weight is
  `omega_t = clip(1 / (1 + max(0, r_t/r0 - 1)), 0.25, 1.0)`.

Betting rule:

* For bounded payoff families, use the fixed lambda grid from `m07_bayes`
  (`BET_C = {-0.9,-0.6,-0.3,0.3,0.6,0.9}`) scaled by `omega_t` before the
  one-step bet: `log(1 + omega_t*lambda*(h_t-m0))`, where `m0` is the
  history-only payoff mean and the payoff cap is the same as `m07_bayes`.
* For power martingales, transform each base one-step e-value `e_t` into
  `1 + omega_t*(e_t - 1)`. This remains predictable because `omega_t` excludes
  row `t`.
* Emit current log-capital and running max channels. No online-estimated
  parameters, thresholds, powers, caps, or weights are tuned.

`RT-1216` emits exactly eight columns:

1. `wctm_tail_log`
2. `wctm_tail_pk`
3. `wctm_disp_log`
4. `wctm_disp_pk`
5. `wctm_dep_log`
6. `wctm_vc_log`
7. `wctm_pow_log`
8. `wctm_pow_pk`

## H1 Matched Unweighted Control (`RT-1217`)

The control uses the same conformity stream, payoffs, p-values, lambda grid,
power grid, caps, and output columns, but fixes `omega_t = 1` for every row.
It emits the same eight columns with `uctm_` prefixes. It is a binding control
for the weighting itself; the base bank still includes incumbent `m07_bayes`.

## H2 Parameter-Free E-Value Aggregation (`RT-1218`)

The H2 direct-score arm reads the existing cached `m07_bayes` e-process
log-capital columns only:

* `ev_tail_mix`
* `ev_tail_ad`
* `ev_disp_mix`
* `ev_disp_ad`
* `ev_pow_mix`

At each row it computes the parameter-free log-average e-value:

`score_t = log(mean(exp(clip(log_e_i,t, -40, 40))))`

No learned model is fit for `RT-1218`. The score vector is evaluated directly as
a candidate OOF stream and then passed through the same fold-pure SCDF marginal
integration path as other direct-score pilots. Running-max, z-scored, posterior,
and non-e-process columns are excluded from H2.

## Candidate Architecture

`RT-1216` and `RT-1217` are Mode A feature-addition screens using the existing
legal feature bank:

`m00_core,m01_seq,m02_dist,m03_dyn,m04_resid,m06_loc,m07_bayes`

plus only the H1 feature block under test.

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
Pilot 8 screen decision.

## Causal Checks Before Training

Before the first Pilot 8 score:

* run `harness.verify()` on weighted and unweighted H1 mechanisms;
* run registered feature-module prefix invariance at `atol=0.0`;
* explicitly test that `omega_t` excludes row `t`;
* explicitly test deterministic replay;
* explicitly test that mutating future online rows does not change earlier
  emitted rows;
* verify H2 uses only the five preregistered existing e-process columns;
* run the new Pilot 8 targeted unit tests.

## RT600 Sentinel

Before interpreting Pilot 8, reproduce:

* RT600 mean approximately `0.62581`
* RT600 pooled approximately `0.625627`
* dominant cell approximately `0.66428`
* fold-0 E0 RT600 approximately `0.63828`
* fold-0 E1 RT600 + RT-401 seed clone approximately `0.63859`

If these materially differ, stop and debug the evaluation path before scoring.

## Ensemble Integration

For all three IDs, compute:

* E0: RT600
* E1: RT600 + exchangeable `RT-401` seed clone
* E2: RT600 + Pilot 8 arm
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
* feature-build runtime and training/runtime diagnostics where applicable

## Binding Gates

Primary H1 weighted gate:

* `RT-1216 marginal_vs_clone < +0.0010`: KILL
* `+0.0010` to `< +0.0020`: WEAK
* `+0.0020` to `< +0.0030`: INTERESTING
* `+0.0030` to `< +0.0050`: SERIOUS
* `+0.0050` to `< +0.0080`: MAJOR
* `>= +0.0080`: BREAKTHROUGH

Additional H1 never-break gate:

* KILL `RT-1216` if its mature-break-vs-never-break AUC does not exceed
  `RT-1217` on fold 0. This is the preregistered target population where the
  weighting is predicted to help.

Control interpretation:

* `RT-1217` is not promoted. If it beats or matches `RT-1216` on the never-break
  split, the weighting hypothesis is killed even if the weighted marginal is
  positive.

H2 gate:

* `RT-1218 marginal_vs_clone < +0.0010`: KILL
* `+0.0010` to `< +0.0020`: WEAK
* `>= +0.0020`: CONTINUE to confirmation as a parameter-free e-value
  aggregation candidate.

## Continuation Gate

Run 5-fold confirmation only if:

* `RT-1216 marginal_vs_clone >= +0.0020` and `RT-1216` beats `RT-1217` on
  mature-break-vs-never-break AUC; or
* `RT-1218 marginal_vs_clone >= +0.0020`.

If either arm clears continuation, stop the broad sweep and confirm it. If no
arm clears continuation, file Pilot 8 and stop the planned first-sweep queue as
exhausted.

## Runtime Ceiling

2 hours for implementation checks, feature build, fold-0 H1 arms, H2 direct
score, diagnostics, and Pilot 8 report.

## Code Paths

Create:

* `src/sbr/features/m18_weighted_ctm.py`
* `research/scripts/novel_streams/pilots/pilot08_weighted_ctm.py`
* `tests/test_pilot08_weighted_ctm.py`
* `research/reports/new_avenues_2026/pilot08_weighted_ctm.md`
* `research/reports/new_avenues_2026/pilot08_weighted_ctm.json`
