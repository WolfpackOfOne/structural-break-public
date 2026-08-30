# SS-03 Execution Preregistration: Negative-Side Null Calibrator

Written 2026-08-25 on branch `research/new-avenues-pilots-2026`, at checkpoint
`1849b1f`, after SS-01 and SS-02 were scored, killed, committed, pushed, and
Research hygiene passed. No SS-03 score, OOF artifact, `RESULTS.csv` row,
lockbox/test result, production artifact, or submission artifact exists at this
preregistration checkpoint.

Program preregistration:
`research/reports/new_avenues_2026/SECOND_SWEEP_PREREG.md`, section
`SS-03 Negative-Side Null Calibrator`.

## Reserved IDs

| ID | role |
|---|---|
| `RT-1225` | SS-03 candidate: weighted-CTM null-state conditional calibration of RT600 |
| `RT-1226` | global RT600 null-SCDF calibration control |
| `RT-1227` | deranged weighted-CTM null-state partition control |
| `RT-1228` | unweighted-CTM state partition control |
| `RT-1229` | Pilot-9 scalar difficulty state partition control |

## Inputs

Allowed inputs:

- Seven frozen RT600 specialist OOF streams `RT-300`, `RT-410` through
  `RT-415`.
- Frozen seed clone `RT-401` only for the standard marginal-vs-clone
  integration denominator.
- Frozen Pilot-8 weighted and unweighted CTM feature caches
  `m18_wctm::wctm_tail_log` and `m18_uctm::uctm_tail_log`.
- Frozen Pilot-9 difficulty scalar OOF stream `RT-1212` only for the scalar
  control arm.
- Causal row state `t` for the fixed maturity split.
- Fold labels and row labels only to fit fold-pure calibration maps on
  outer-training rows.

Forbidden inputs:

- `RT-1216` parameter, threshold, weight, window, or feature tuning.
- Any first-sweep candidate/control prediction as a candidate input, except
  `RT-1212` in the declared scalar control arm.
- Deranged scalar or partition search.
- Raw context leakage, `n_online`, `n_hist`, target `tau`, future rows,
  lockbox/test rows, production artifacts, submission artifacts, or any
  validation-fold label in that fold's calibration map.

## Fold-Pure Construction

For each outer fold `f`:

1. Fit all SS-03 calibration maps on rows from folds `{0,1,2,3,4} \ {f}`.
2. Emit predictions only for fold `f`.
3. For training-fold score-state covariates used to fit maps for outer fold
   `f`, calibrate each frozen stream with `SCDF_NSEEN` using folds excluding
   both `f` and the row's own permanent fold. For fold `f` validation rows,
   calibrate frozen streams using all folds except `f`.
4. No lockbox/test row receives a finite SS-03 score.

The RT600 base score in SS-03 is the mean of the seven fold-pure calibrated
specialist streams, matching the RT600 ensemble contract for each validation
fold.

## Candidate Null-State Partition

The candidate partition is fixed before scoring.

Rows with `t < 200` use one early catch-all state. Rows with `t >= 200` use
three binary state bits:

1. `high_rt600 = rt600_cal >= 0.50`.
2. `high_specialist_dispersion = specialist_std >= median(specialist_std)` on
   outer-training mature rows.
3. `high_weighted_ctm_suppression =
   (m18_uctm::uctm_tail_log - m18_wctm::wctm_tail_log) >= median(...)` on
   outer-training mature rows.

Mature state id is
`1 + high_rt600 + 2 * high_specialist_dispersion + 4 * high_weighted_ctm_suppression`.
The candidate therefore has one early state plus eight mature states.

No threshold is searched. The two medians are recomputed separately inside each
outer fold using only outer-training rows and no labels.

## Calibration Formula

For each outer fold and each arm:

- `F_all` is `SCDF_NSEEN(rt600_cal, t)` fit on all outer-training rows.
- `F_null_global` is `SCDF_NSEEN(rt600_cal, t)` fit on outer-training negative
  rows.
- For each state `s`, `F_null_s` is `SCDF_NSEEN(rt600_cal, t)` fit on
  outer-training negative rows assigned to state `s`.
- A state-specific null map is used only if it has at least `5,000` negative
  rows; otherwise it falls back to the mature high-RT600 parent, then mature
  global null, then global null.

For a validation row:

`offset = clip(F_null_state(rt600_cal, t) - F_all(rt600_cal, t), -0.25, +0.25)`

`ss03_score = clip(rt600_cal + 0.20 * offset, 0, 1)`

The maximum absolute calibration correction is therefore `0.05`. This is a
bounded recalibration of incumbent evidence, not a new break detector and not a
new trained classifier.

## Controls

`RT-1226` uses the same formula but replaces `F_null_state` with
`F_null_global` for every validation row.

`RT-1227` fits the same state maps as `RT-1225`, then deranges validation
state ids within exact `t` groups using seed `2026082507`. Groups with more
than one distinct state are cyclically shifted after sorting by state id and
row id, preserving the exact same-time state multiset while destroying the
row-state match. Groups with one distinct state are unchanged and recorded in
the audit.

`RT-1228` replaces the weighted-CTM suppression bit with
`m18_uctm::uctm_tail_log >= median(...)` on outer-training mature rows.

`RT-1229` replaces the CTM bit with
`RT-1212_cal >= median(RT-1212_cal)` on outer-training mature rows. This is a
control replay of the killed Pilot-9 scalar as a partition covariate, not a new
scalar search or a new Pilot-9 mechanism.

All controls use the same `F_all`, correction scale, offset clip, fallback
hierarchy, folds, pair sampler, and reporting path as the candidate.

## Evaluation

Binding screen is fold 0 only. Folds 1-4 are emitted only to provide fold-pure
SCDF calibration support for the standard marginal-vs-clone integration path;
their TS-AUCs are not inspected for the SS-03 screen decision.

For every scored SS-03 ID report:

- fold-0 standalone TS-AUC;
- diagnostic pack: whole fold, dominant cell, mature-vs-never-break,
  mature-vs-prebreak, time buckets, age buckets, within-`t` rank correlation
  with RT600;
- canonical same-time pair-flow pack with 64 pairs per `t`, including
  whole-fold, dominant-cell, mature-vs-never, and mature-vs-prebreak splits;
- E0 RT600, E1 RT600 + `RT-401`, E2 RT600 + SS-03 arm, and
  `marginal_vs_clone`;
- state counts, fallback counts, medians, lockbox finite counts, runtime, and
  peak RSS.

Before interpreting the screen, reproduce the RT600 sentinel:

- dev mean approximately `0.625811`;
- dev pooled approximately `0.625627`;
- dominant cell approximately `0.664277`;
- fold-0 E0 RT600 approximately `0.638276`;
- fold-0 E1 RT600 + `RT-401` approximately `0.638586`.

## Binding Screen Gates

`RT-1225` is KILL if any mandatory gate fails:

1. Fold-0 mature-vs-never dominant-cell net pair lift must be `> 0`.
2. Fold-0 `marginal_vs_clone >= +0.0010`.
3. Mature-vs-prebreak net pair lift must be `>= -150`.
4. Mature-vs-prebreak RT600-right damage rate must be `<= 0.0150`.
5. Deranged partition control `RT-1227` must lose by at least `+0.0005`
   marginal-vs-clone and must not match or exceed candidate mature-vs-never net
   pair lift.
6. Global, unweighted-CTM, and scalar-difficulty controls must not match or
   exceed the candidate marginal-vs-clone value.
7. Causal/fold-purity/lockbox audits must pass.

If `RT-1225` passes, stop the broad sweep and write a separate confirmation
execution note before any 5-fold confirmation. If it fails, file the result as
KILL and continue to SS-04; no SS-03b, threshold adjustment, partition search,
correction-scale change, CTM retuning, or Pilot-9 scalar variant is authorized.

## Kill Interpretation

A KILL closes this fixed null-state calibration over the current RT600 score
state, weighted CTM suppression state, and bounded conditional-null correction.
It does not falsify a future materially different causal state representation
or external-data null model.
