# SS-04 Execution Preregistration: Specialist Disagreement Micro-Router

Written 2026-08-25 on branch `research/new-avenues-pilots-2026`, at checkpoint
`6a554ba`, after SS-01, SS-02, and SS-03 were scored, killed, committed,
pushed, and Research hygiene passed. No SS-04 score, OOF artifact,
`RESULTS.csv` row, lockbox/test result, production artifact, or submission
artifact exists at this preregistration checkpoint.

Program preregistration:
`research/reports/new_avenues_2026/SECOND_SWEEP_PREREG.md`, section
`SS-04 Specialist Disagreement Micro-Router`.

## Reserved IDs

| ID | role |
|---|---|
| `RT-1230` | SS-04 candidate: bounded row-level specialist action router |
| `RT-1231` | global logistic specialist reweighting control |
| `RT-1232` | Pilot-1 static history-fingerprint selector replay control |
| `RT-1233` | shuffled disagreement-target router control |

The equal RT600 blend is the E0 baseline and consumes no new RT ID.

## Inputs

Allowed candidate inputs:

- Seven frozen RT600 specialist OOF streams `RT-300`, `RT-410` through
  `RT-415`.
- Frozen seed clone `RT-401` only as row-state covariate and for the standard
  marginal-vs-clone denominator.
- Causal row-level score state derived from the specialist streams: RT600
  calibrated blend, specialist deltas from that blend, score margin, specialist
  dispersion/range/min/max, fraction above RT600, fraction above 0.5, seed-clone
  difference, and fixed time summaries.

Forbidden candidate inputs:

- Pilot-1 history fingerprints or static DGP selectors.
- Any global specialist weights/subsets as the candidate output.
- First-sweep killed arm scores `RT-1200` through `RT-1229`.
- Raw context leakage, `n_online`, `n_hist`, target `tau`, future rows,
  lockbox/test rows, production artifacts, submission artifacts, or validation
  labels in that fold's router fit.

## Candidate Action Set

For every outer fold `f`, all frozen streams are calibrated with `SCDF_NSEEN`
using folds excluding `f` for validation rows and excluding both `f` and the
row's own fold for training rows. The RT600 base is the mean of the seven
calibrated specialist streams.

The candidate has eight actions:

- action `0`: keep RT600 unchanged;
- action `1..7`: move toward one specialist `j`.

For action `j`, the emitted score is:

`score = rt600_cal + 0.25 * clip(specialist_j_cal - rt600_cal, -0.35, +0.35)`

The maximum absolute row movement is `0.0875`. There are no candidate-specific
global weights, blend weights, specialist subset weights, or monotone rank
transforms.

## Router Target

The router target is derived only from outer-training folds.

For each outer fold `f`, sample dominant-cell same-time training pairs:

- positives: `y=1`, `t >= 200`, post-break age `>= 100`;
- negatives: `y=0`, `t >= 200`;
- per `t`, sample `k=min(128, n_pos, n_neg)` positive rows and the same number
  of negative rows without replacement;
- seed `2026082510 + 1000*f + t`.

For each sampled pair `(p, n)` and each specialist action `j`, accumulate
row-local utility:

- positive endpoint utility is `+1` if action `j` repairs an RT600-wrong pair
  by making `score_j(p) > rt600(n)`;
- negative endpoint utility is `+1` if action `j` repairs an RT600-wrong pair
  by keeping `rt600(p) > score_j(n)`;
- if RT600 is right, an action that flips the pair wrong contributes `-2`;
- pairs with `t >= 200` and mature positive age `>= 100` have weight `3`.

For each exposed row, action label is the specialist action with maximum
positive normalized utility; otherwise action `0`. Row sample weight is total
pair exposure normalized to mean one.

## Router Model

The candidate and shuffled-target control use the same low-degree model:

- multinomial logistic regression;
- fixed robust median/MAD standardization fit on router-training rows only;
- `C=0.5`, `solver=lbfgs`, `max_iter=200`, `fit_intercept=True`;
- action is taken only if the best nonzero class has probability at least
  `0.20` and exceeds the keep-RT600 probability by at least `0.05`;
- otherwise emit action `0`.

Report coefficient norms by feature family and top signed coefficients for
audit. These audits are descriptive and do not create new scored arms.

## Controls

`RT-1231` global logistic specialist reweighting control:

- for each outer fold, fit one binary logistic regression on up to `400,000`
  uniformly sampled outer-training rows using only the seven calibrated
  specialist scores;
- seed `2026082511 + f`;
- emit validation probabilities. This is the forbidden candidate class
  "global specialist weighting/stacking" used only as a control.

`RT-1232` Pilot-1 static selector replay control:

- replay the Pilot-1 best history-only selector fingerprint `exc_max_run64`;
- split eligible dev series into five fold-pure quantile bins using
  outer-training series only;
- within each bin, choose the specialist with lowest dominant-cell
  pair-loss rate on outer-training series;
- validation series receive that bin's chosen specialist stream; empty bins
  fall back to RT600.

This is a static series-level selector control and is forbidden to the
candidate.

`RT-1233` shuffled disagreement-target control:

- same features, pairs, labels, weights, model, action gating, and bounded
  specialist moves as `RT-1230`;
- row action labels are permuted within the exposed training rows for each
  outer fold using seed `2026082512 + f`.

## Evaluation

Binding screen is fold 0 only. Folds 1-4 are emitted only to provide fold-pure
SCDF calibration support for the standard marginal-vs-clone integration path;
their TS-AUCs are not inspected for the SS-04 screen decision.

For every scored SS-04 ID report:

- fold-0 standalone TS-AUC;
- diagnostic pack: whole fold, dominant cell, mature-vs-never-break,
  mature-vs-prebreak, time buckets, age buckets, within-`t` rank correlation
  with RT600;
- canonical same-time pair-flow pack with 64 pairs per `t`, including
  whole-fold, dominant-cell, mature-vs-never, and mature-vs-prebreak splits;
- specialist-disagreement pair-flow on the canonical dominant-cell sample for
  `majority_correct` (`>=4` specialists rank the positive above the negative)
  and `near_split` (`3` or `4` specialists rank the positive above the
  negative);
- E0 RT600, E1 RT600 + `RT-401`, E2 RT600 + SS-04 arm, and
  `marginal_vs_clone`;
- action counts, selected-specialist counts, coefficient audits, lockbox finite
  counts, runtime, and peak RSS.

Before interpreting the screen, reproduce the RT600 sentinel:

- dev mean approximately `0.625811`;
- dev pooled approximately `0.625627`;
- dominant cell approximately `0.664277`;
- fold-0 E0 RT600 approximately `0.638276`;
- fold-0 E1 RT600 + `RT-401` approximately `0.638586`.

## Binding Screen Gates

`RT-1230` is KILL if any mandatory gate fails:

1. Fold-0 `marginal_vs_clone >= +0.0010`.
2. Dominant-cell net pair lift must be `>= 0`.
3. `majority_correct` specialist-disagreement pair-flow net must be `> 0`.
4. `near_split` specialist-disagreement pair-flow net must be `> 0`.
5. Candidate must beat the global, static-selector, and shuffled-target
   controls by at least `+0.0005` marginal-vs-clone.
6. The shuffled-target control must not independently satisfy gates 1-4.
7. Causal/fold-purity/lockbox audits must pass.

If `RT-1230` passes, stop broad execution and write a separate confirmation
execution note before any 5-fold confirmation. If it fails, file the result as
KILL and close specialist-disagreement routing; no SS-04b, action-threshold
tuning, alpha/cap tuning, feature addition, fingerprint selector promotion, or
global specialist weighting is authorized.

## Kill Interpretation

A KILL closes this fixed row-level specialist-disagreement micro-router over
the current RT600 specialist score state. With SS-01 through SS-04 killed, the
preregistered Second Sweep has no remaining authorized candidate in this
document.
