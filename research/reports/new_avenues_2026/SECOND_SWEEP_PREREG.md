# Second Sweep Preregistration

Status: `DESIGNED_NOT_EXECUTED`

This document preregisters the second New Avenues sweep after the first sweep was exhausted. No second-sweep experiment has been run. No RT id has been allocated. No OOF score, TS-AUC, `RESULTS.csv` row, lockbox/test result, production artifact, or submission artifact exists for any second-sweep candidate.

Design inputs:

- `research/reports/new_avenues_2026/FIRST_SWEEP_SYNTHESIS.md`
- `research/reports/new_avenues_2026/first_sweep_matrix.csv`
- `research/reports/new_avenues_2026/first_sweep_repair_summary.json`
- `research/reports/new_avenues_2026/first_sweep_specialist_decomposition.json`
- historical Wave4-Wave8 reports and RT600 freeze documents

Machine-readable priority table:

- `research/reports/new_avenues_2026/second_sweep_candidate_priority.csv`

## Global Rules

Second-sweep candidates must be materially different from first-sweep killed mechanisms. A candidate may reuse a killed arm only as a frozen diagnostic sensor inside a preregistered arbitration target; it may not be a tuned variant of the killed mechanism.

Every executed candidate must state its RT id before scoring, append `research/RESULTS.csv` only through the normal protocol after a scored experiment, and preserve negative results. This preregistration itself allocates no RT id.

Execution remains fold-pure and causal-prefix only. No lockbox/test/submission path is allowed during the screen. Full confirmation is allowed only after the fold-0 continuation gate is met and a separate execution note says why confirmation is warranted.

Default screen gate:

- fold-0 `marginal_vs_clone >= +0.0015`
- dominant-cell damage-adjusted pair flow positive
- matched controls do not match or beat the candidate
- no causal-prefix violation
- no hidden use of `tau`, `n_online`, lockbox, test, or future rows

Default serious-result gate:

- 5-fold `marginal_vs_clone >= +0.0030`
- positive marginal direction in at least 4 of 5 folds
- dominant-cell repair/damage balance positive in every fold or explained by a preregistered cell split
- production feasibility and runtime reviewed separately before any lockbox/test action

## SS-01 Repair-Damage Arbiter

Priority: 1

Recommended first second-sweep run: yes.

Core question: can fold-pure arbitration keep the first-sweep repairs while rejecting first-sweep damages?

Rationale: the first sweep contains a large repair reservoir. In the post-hoc dominant-cell sample, any first-sweep candidate repaired `91.8%` of RT600 wrong pairs, and two or more candidates repaired `71.8%`. The same set damaged `83.0%` of RT600-correct pairs. This makes an unfiltered new statistic unlikely to help, but it leaves a specific falsifiable question: are repair events separable from damage events?

Allowed inputs:

- frozen RT600 score state
- seven frozen RT600 specialist scores
- seed-clone score
- first-sweep scored arms `RT-1200` through `RT-1218` as fold-pure, prefix-computable sensors
- row state available causally at prediction time: `t`, score ranks/margins, specialist dispersion, score-state history, and preregistered maturity buckets

Forbidden inputs:

- target `tau`, future rows, `n_online`, final series length, lockbox/test rows
- any retraining or tuning of `RT-1216` or another killed arm
- any objective that simply stacks killed arms by global weights

Candidate form: a constrained correction/arbitration layer over frozen sensors. The layer outputs an additive correction to RT600 or a finite action over `RT600 only`, `small positive blend with sensor family j`, and `reject sensor family j`. It is trained with a same-time pair residual objective: reward RT600-wrong pair repairs and penalize RT600-right pair damages. It is not a standalone break classifier.

Controls:

- RT600 plus seed clone
- global average of the same frozen sensors
- shuffled repair/damage target
- family-holdout arbiter
- single-best killed-arm blend selected on train folds only

Screen gate:

- fold-0 `marginal_vs_clone >= +0.0015`
- dominant-cell net pair lift positive
- damage rate on RT600-right dominant pairs below half the candidate-set any-damage descriptive rate from the synthesis
- at least two sensor families contribute in fold-pure validation
- controls do not match within `+0.0005`

Full confirmation gate:

- 5-fold `marginal_vs_clone >= +0.0030`
- positive in at least 4 of 5 folds
- no single killed arm explains more than half the lift under family-holdout analysis

Kill interpretation: if SS-01 fails, the second sweep should stop treating first-sweep killed arms as useful production sensors. The failure would support the conclusion that repairs are not separable from damages with available causal state.

## SS-02 Dominant-Cell Residual Ranker

Priority: 2

Recommended first second-sweep run: no; run after SS-01 or if first-sweep sensors are ruled out for runtime reasons.

Core question: can a correction model trained on RT600 residual pair inversions move broad dominant-cell pair flow?

Rationale: first-sweep mechanisms repeatedly had plausible standalone signal but wrong pair flow. This candidate attacks the metric residual directly instead of adding another anomaly block.

Allowed inputs:

- existing 500-feature causal bank
- frozen RT600 score, fold-pure specialist scores, and seed-clone score
- causal row state: `t`, age proxy buckets available without `tau`, score ranks/margins, and specialist dispersion

Forbidden inputs:

- first-sweep killed candidate scores, unless SS-01 is killed specifically for non-separability and a new prereg reopens them
- target `tau`, future rows, `n_online`, lockbox/test rows

Candidate form: a ranker/correction stream trained on same-time RT600 residual pair examples from training folds, with higher weight on dominant-cell pairs and explicit damage penalty on RT600-correct pairs. The output is a correction to RT600, not a standalone BCE probability.

Collision check:

- Not the old generic pairwise objective: the target is RT600 residual repair/damage, not global break/non-break ordering.
- Not W5 hard-negative reweighting: it uses an explicit correction score and pair damage penalty, not row reweighting inside the same BCE learner.
- Not Wave7 T2: it has no future-label teacher and no nested future broadcast.

Screen gate:

- fold-0 `marginal_vs_clone >= +0.0015`
- dominant-cell pair net at least `+300` on the canonical repair sample used by the execution report
- never-break-negative pair net positive or neutral with clear pre-break gain

Full confirmation gate:

- 5-fold `marginal_vs_clone >= +0.0030`
- dominant-cell lift positive in all five folds
- seed-clone and shuffled-residual controls lose by at least `+0.0010`

Kill interpretation: if SS-02 fails after SS-01 fails, the remaining causal prefix information is likely too diffuse for a residual correction model using the incumbent bank.

## SS-03 Negative-Side Null Calibrator

Priority: 3

Recommended first second-sweep run: no.

Core question: can causal null-state calibration reduce never-break false positives without damaging pre-break ordering?

Rationale: dominant-cell loss is heavily negative-side, D3 fingerprints weakly but nonzero predict loss, and `RT-1216` suggests predictable benign-tail state can matter. The first sweep also warns that scalar DGP gates are fragile and controls can erase the mechanism.

Allowed inputs:

- RT600 score and specialist score state
- causal tail/dependence/maturity fingerprints already validated in the feature bank or first-sweep diagnostics
- optional CTM-derived state only as a calibration covariate, not as a tuned CTM feature family

Forbidden inputs:

- `RT-1216` parameter tuning
- deranged scalar search
- raw context leakage, `n_online`, future rows, or target `tau`

Candidate form: a hierarchical or split-conformal calibration of incumbent evidence conditioned on a small preregistered null-state partition. The output should mostly alter negative-side rank order and confidence, not discover a new break statistic.

Controls:

- global RT600 SCDF calibration
- deranged null-state partition
- unweighted CTM state
- scalar difficulty gate control from Pilot 9

Screen gate:

- fold-0 never-break dominant-cell pair net positive
- overall `marginal_vs_clone >= +0.0010`
- pre-break damage not worse than a preregistered cap
- deranged partition loses

Full confirmation gate:

- 5-fold `marginal_vs_clone >= +0.0030`
- never-break benefit survives in at least 4 of 5 folds
- no control failure

Kill interpretation: if SS-03 fails, null-state calibration should be considered exhausted for this bank unless new external data or a materially different causal state representation exists.

## SS-04 Specialist Disagreement Micro-Router

Priority: 4

Recommended first second-sweep run: no.

Core question: can row-level specialist disagreement rescue the `66.3%` of sampled dominant RT600 mistakes where at least one specialist is correct?

Rationale: the specialist decomposition shows raw disagreement signal, but simple routing is weak. Only `6.3%` of sampled RT600 dominant mistakes had four or more specialists correct, and Pilot 1 fingerprint selectors were far below RT600. This is a narrow, cheap falsifier, not a primary alpha path.

Allowed inputs:

- seven frozen specialist scores
- RT600 blend score
- seed-clone score
- causal row-level score margins, ranks, dispersion, and time-state summaries

Forbidden inputs:

- static DGP/fingerprint selectors from Pilot 1
- global specialist weights or subsets
- first-sweep killed arm scores

Candidate form: a low-degree row-level router that changes the blend only when disagreement/margin state predicts a repair with bounded damage. It must be constrained enough to be audited by coefficient/sign and family ablations.

Controls:

- equal RT600 blend
- global reweighted specialist blend
- Pilot 1 static selector replay
- shuffled disagreement target

Screen gate:

- fold-0 `marginal_vs_clone >= +0.0010`
- positive repair/damage balance on the majority-correct and near-split RT600-error subsets
- no dominant-cell net loss

Full confirmation gate:

- 5-fold `marginal_vs_clone >= +0.0030`, or `>= +0.0020` only if paired with an already-successful SS-01 family-holdout result
- positive damage-adjusted pair flow in at least 4 of 5 folds

Kill interpretation: if SS-04 fails, specialist-disagreement routing is closed. Continue only with non-specialist residual correction or stop.

## Explicitly Closed From This Prereg

The following are not second-sweep candidates:

- `RT-1216` weight/window/threshold tuning
- a new CTM variant as a standalone feature block
- another scalar difficulty gate
- another spectral, ordinal, trajectory, dwell, or joint-rarity feature block
- generic ensemble stacking/weighting/subset selection
- raw TCN/MLP capacity runs
- future-aware distillation repeats
- lockbox/test/prod/submission activity

## Exact First Run

Run exactly one first if the second sweep is executed:

`SS-01 Repair-Damage Arbiter`

Reason: it directly tests the largest new fact from the synthesis: first-sweep arms collectively repair most sampled RT600 mistakes but damage most RT600-correct pairs. A successful SS-01 can plausibly reach `+0.003` only by learning that separation. A failed SS-01 prevents wasting the second sweep on slightly modified killed mechanisms.
