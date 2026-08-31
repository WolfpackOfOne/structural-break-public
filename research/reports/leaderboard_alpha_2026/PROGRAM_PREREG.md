# LEADERBOARD ALPHA 2026 -- PROGRAM PREREGISTRATION

Date: 2026-08-26

Branch: `research/leaderboard-alpha-2026`

Base SHA: `d47937718ba91c2327aafa4528bdba39ee06661f`

Base rationale: this is the tip of `engineering/rt600-final-reliability-2026`,
which contains the completed CRF lineage and the audited RT600 runtime-fix
commits. The protected branches `production/rt600`,
`release/rt600-reliability-2026`, `main`, and `research/current` are not
modified.

## Program Scope

The only permitted scored experiments in this program are:

1. LA-01 specialist replacement salvage.
2. LA-02 counterfactual synthetic augmentation.
3. LA-03 per-series history adaptation, conditional on LA-02 not becoming
   SERIOUS.

No feature brainstorming, learner comparison, Optuna, routing, stacking,
calibration sweep, future distillation, repair/damage arbiter, generic sequence
model, or learned amortized-null lane is in scope.

LA-02 is the main bet. If LA-02 is SERIOUS, breadth work stops and LA-02 is
confirmed before LA-03 is opened.

## Standing Evaluation Rule

Every scored experiment uses the full five canonical development folds from the
start. No fold-0 architecture fishing is permitted.

Primary competition metric: `marginal_vs_clone` under the established E0/E1/E2
protocol.

Every scored report must include:

- mean five-fold TS-AUC
- fold deltas
- dominant-cell pair repairs, damage, and net
- mature-vs-never pair net
- mature-vs-prebreak pair net
- within-`t` correlation with RT600
- runtime

Classification:

- `< +0.0015 marginal_vs_clone`: KILL
- `+0.0015` to `< +0.0030`: WEAK
- `>= +0.0030`: SERIOUS
- `>= +0.0050`: MAJOR

SERIOUS also requires at least four of five folds positive.

## LA-00 Optional Old-Data Audit

LA-00 may consume at most 30 minutes and produces no score. A prior/public 2025
competition dataset may be used only if all of the following are unambiguous:

1. current competition rules allow it,
2. schema and semantics are compatible,
3. descriptive train-only distributions look usable.

If legality or compatibility is unclear, the dataset is marked NOT USED and the
program continues. No model score may depend on LA-00.

## LA-01 Specialist Replacement Salvage

Question: did `m11_focus` or `m12_rdep` contain useful specialization that was
erased because previous integration added models instead of replacing a
redundant RT600 specialist?

Use existing OOF vectors/models where available. Do not retrain a large family
unless a required artifact is missing.

Inputs:

- seven RT600 specialist OOF vectors
- candidate `m11_focus` specialist
- candidate `m12_rdep` specialist

Design:

- keep exactly seven ensemble members
- equal weighting
- existing SCDF/integration convention
- do not tune blend weights
- do not simply add an eighth or ninth member

Nested headline rule:

For each outer fold `f`, using only the other four folds, select the one RT600
specialist whose replacement by either candidate gives the best training-fold
ensemble result. Evaluate that frozen replacement on fold `f`.

Controls:

- primary: untouched seven-specialist RT600 ensemble
- secondary: nested replacement by an exchangeable seed clone
- fixed replacement diagnostics are allowed but are not headline results

Gate:

- `marginal_vs_clone >= +0.0015`
- at least four of five folds positive
- dominant pair net `> 0`

If `marginal_vs_clone >= +0.0030`, classify SERIOUS. If KILL, permanently close
specialist-replacement salvage.

## LA-02 Counterfactual Synthetic Augmentation

Hypothesis: the model has largely exhausted finite labelled trajectories, but
each series supplies 1k-5k guaranteed clean historical points that can generate
fold-pure paired counterfactual training examples. The experiment isolates
training data by using the existing RT600 feature/model architecture.

Null generator:

- use the already validated fixed per-series null corresponding to the CRF
  fixed-null control
- fit using `H_i` only
- no learned amortized null

Paired counterfactuals:

- from the same simulated null continuation create a persistent-positive
  trajectory and a hard-negative trajectory
- persistent positives cover training-supported location, scale, and dependence
  mechanisms
- transient negatives mimic heavy-tail/outlier, shock, variance burst, and
  temporary displacement false positives
- all magnitude, duration, online-length, and break-location distributions are
  estimated from outer-training series only and frozen before scoring

Time/data:

- generate full causal sequences and train normally
- do not directly metric-weight the loss
- sample online lengths and break locations from empirical outer-training
  distributions
- use one fixed augmentation ratio only: synthetic rows target 33 percent of real
  training rows, chosen as a conservative midpoint of the requested 25-50 percent
  range that should be large enough to affect training while keeping compute
  below the screening budget
- no validation-derived sampling adjustment

Model:

- train the same seven-specialist RT600 architecture on real plus synthetic
  training rows
- validation remains entirely real
- no synthetic validation
- no new model family or hyperparameter sweep

Controls:

- C0: normal RT600 training
- C1: same number of additional synthetic null-only, non-paired examples

Primary LA-02 is paired persistent-vs-transient counterfactual augmentation.

Gate:

- five-fold `marginal_vs_clone >= +0.0025`
- at least four of five folds positive
- dominant pair net positive
- mature-vs-never pair net positive
- candidate beats C1 by `>= +0.0010`

`>= +0.0030` is SERIOUS. `>= +0.0050` is MAJOR.

## LA-03 Per-Series History Adaptation

Run only if LA-02 does not become SERIOUS.

Hypothesis: CRF showed series-specific history contains information, but a small
global amortized bottleneck loses to the fixed per-series null. Test local
adaptation rather than a wider global bottleneck.

Allowed design:

- global small causal predictive model
- tiny per-series adaptation fitted only on `H_i`
- adaptation may be a last-layer/head, affine state calibration, or small
  low-rank adapter
- no full independent model per series
- unsupervised/predictive target only, such as next-step prediction or residual
  likelihood
- online inference is causal and emits predictive residual/state evidence into a
  small fixed break-scoring head
- no RT600 score as model input

Controls:

- C0: same global model without per-series adaptation
- C1: existing fixed per-series null

Mandatory isolation gate:

- adapted predictive/null representation must materially beat both controls
  before ensemble integration
- if it cannot beat the fixed null, KILL immediately

Final competition gate:

- `marginal_vs_clone >= +0.0015`
- SERIOUS requires `>= +0.0030` and at least four of five folds positive

No width sweep, adapter-size sweep, or alternate architecture is allowed.

## Combination Rule

Only if two mechanisms individually survive, test exactly one frozen-output
combination: the best surviving LA-01/02/03 combination.

The combination must beat the best component by `>= +0.0010
marginal_vs_clone`; otherwise reject it.

