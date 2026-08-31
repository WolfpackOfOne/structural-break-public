# FIRST THREE NEW-AVENUES PILOTS -- PREREGISTRATION

Written 2026-08-24 on branch `research/new-avenues-pilots-2026`, before any
new pilot headline score or marginal-vs-clone result was produced in this
branch.

Starting base: `origin/research/current` at
`6c37cb122f83012c39706f1f182ce8e3182461be`.

Local cleanup status at preregistration time: the sibling
`research/current` worktree contains an unpushed cleanup commit
`f236c62`, but `origin/research/current` has not advanced. This branch starts
from remote truth and will merge the cleanup later only if it lands on
`origin/research/current`.

No lockbox rows, `X_test.reduced.parquet`, production branch, or
`research/current` worktree files may be read for selection or modified by
these pilots.

## Repository Facts Verified Before This Preregistration

The following load-bearing prior facts were verified from repository files, not
from the driving prompt:

* `research/NEW_AVENUES_2026.md` and
  `research/reports/new_avenues_2026_diagnostics.json` report RT-600 dev
  reproduction at mean TS-AUC `0.625811`, pooled `0.625627`, dominant-cell
  AUC `0.664277`.
* D1: dominant-cell loss is diffuse; worst 10% of never-break series account
  for `0.277` of never-break loss.
* D3: history-only fingerprints predict dominant-cell loss rate OOF at
  Spearman `+0.1921` for never-break series; permutation control `-0.0313`.
* D4: `res64_maxrun90` has fold-0 dominant-cell AUC `0.60791` and within-t
  correlation `+0.367` with RT-600.
* D5/D6: raw excursion length is not enough; growth is the discriminator, and
  the naive Erdos-Renyi normalization was worse than raw max-run.

## Shared Evaluation Contract

Unless a pilot-specific section narrows the contract, the screen is fold 0 only,
using the permanent repository folds. The binding competition screen is:

`TS-AUC(RT600 + candidate) - TS-AUC(RT600 + RT-401 seed clone)`.

The integration path is `wave8_common.ensemble_marginal`, which applies the
canonical cross-fitted smooth time-conditional CDF calibration and equal-weight
blend. No global rank transform and no validation-tuned blend weight are
allowed.

Standard diagnostics for every scored candidate:

* candidate whole-fold TS-AUC
* dominant-cell AUC, with dominant cell fixed as `t >= 200` and positives
  restricted to age `>= 100`
* mature-break vs never-break AUC
* mature-break vs pre-break AUC
* t-bucket and age-bucket TS-AUC
* within-t rank correlation with RT-600
* pair repairs, damage, and net pair lift
* `RT600`, `RT600 + RT-401 seed clone`, `RT600 + candidate`
* runtime

Kill/continuation gates:

* `candidate - seed clone < +0.0010`: KILL.
* `+0.0010..+0.0020`: WEAK; continue only if pilot-specific cheap continuation
  criteria are also met.
* `+0.0020..+0.0030`: INTERESTING; run 5-fold confirmation after cleanup sync.
* `+0.0030..+0.0050`: SERIOUS; run 5 folds plus bootstrap.
* `> +0.0050`: MAJOR.

Before any expensive 5-fold confirmation, fetch `origin`, compare this starting
SHA with `origin/research/current`, and merge the expected cleanup commit if it
has landed. Do not rebase a pushed branch.

## RT ID Reservation

Pilot 1 is diagnostic-only and consumes no RT ID.

Reserved scored IDs:

| ID | pilot | candidate |
|---|---|---|
| `RT-1200` | Pilot 2 | relay score-state transform of RT-600 |
| `RT-1201` | Pilot 3 | IM2 matched-length empirical run-null plus dwell bank |

If a preregistered sensitivity candidate is scored later, it must receive a new
ID allocated before scoring.

## Pilot 1 -- Specialist Competence And Failure Manifolds

Hypothesis:
The seven RT-600 specialists have systematically different competence across
history-only DGP fingerprints and/or error manifolds. If this variation is
stronger than a permuted-fingerprint control, a future legal history-only gate
or repair model may provide marginal ensemble alpha without retraining
data-starved specialists.

Information channel:
Historical-DGP conditioning and ensemble failure geometry. This is the queued
J4/K1 diagnostic from `NEW_AVENUES_2026.md` section S.

Exact inputs:

* Seven OOF specialist streams: `RT-300`, `RT-410`, `RT-411`, `RT-412`,
  `RT-413`, `RT-414`, `RT-415`.
* RT-600 blend rebuilt with `Ctx.crossfit_blend`.
* Canonical fold assignments from `research/folds/folds.parquet`.
* History-only fingerprint bank from `d2_fingerprint.py`; if absent, rebuild it
  from history only.
* D1 per-series dominant-cell loss attribution from `d1_loss_attrib.py`; if
  absent, rebuild it from OOF scores and fold metadata.

Exact algorithm:

1. For each dev-fold series, compute dominant-cell same-t pair loss and pair
   weight for RT-600 and for each specialist independently.
2. For each series with adequate dominant-cell weight, define specialist
   competence as `specialist_loss_rate - RT600_loss_rate`; lower is better.
3. Bin each history-only fingerprint into fixed training-free quintiles over
   dev series. For each quintile, compute the winning specialist, AUC/loss-rate
   spread across specialists, and monotone Spearman association between the
   fingerprint and each specialist advantage.
4. Run a single permutation control with seed `0`: permute fingerprint rows
   within fold and repeat the spread/argmax-variation calculation.
5. Build a failure-manifold diagnostic matrix using history fingerprints plus
   per-series loss rates by RT-600 and by each specialist. Cluster with fixed
   `k in {2,3}` only, using z-scored columns and deterministic
   `random_state=0`; report silhouette and whether clusters separate loss type,
   never-break heaviness, and specialist advantage better than the permuted
   control.
6. Compute an oracle history-only selector headroom diagnostic: within each
   fingerprint quintile, select the historically best specialist by folds
   other than the held-out fold, then evaluate its dominant-cell loss/AUC on the
   held-out fold. This is diagnostic only and cannot be deployed until a future
   gate is preregistered.

Online state:
None. This pilot scores no deployable online candidate.

Parameter-selection rule:
No tuning. Fingerprint quintiles, `k in {2,3}`, min series dominant-cell weight
`500`, and permutation seed `0` are fixed here.

Exact variants:
Diagnostic J4 specialist-competence and K1 failure-manifold clustering only.
No learned gate is trained or scored in this pilot.

Primary candidate:
None. Primary artifact is a diagnostic verdict.

Sensitivity candidate:
None.

Population:
Canonical dev folds only. Never-break and break populations are reported
separately where relevant. Tau/age may be used only for post-hoc dominant-cell
and mature-vs-never/pre-break diagnostics.

Fold used for screen:
All five dev folds for descriptive OOF summaries; any fold-held selector
headroom uses permanent fold holdout. No lockbox.

Causal constraints:
Only history fingerprints are legal gate inputs. Specialist OOF/labels/tau are
diagnostic outcomes, not deployable gate inputs.

Control:
Fold-preserving permuted fingerprint rows with seed `0`; plus RT-600 equal blend
as the incumbent.

Primary metric:
Variation in specialist dominance by fingerprint quintile above permutation
control, and fold-held oracle selector improvement in dominant-cell AUC/loss.

Dominant-cell metric:
Fixed dominant-cell AUC and loss-rate difference.

Mature-break-vs-never metric:
Loss and AUC split by never-break negatives and pre-break negatives.

Pair-repair diagnostic:
For each specialist, sampled pair repairs/damage versus RT-600 in the dominant
cell, split by negative type.

Correlation diagnostic:
Within-t rank correlations among specialists and with RT-600 inside the
dominant cell; series-level Spearman between fingerprints and specialist
advantage.

Ensemble integration procedure:
None in this pilot.

Kill gate:
Family J/K diagnostic is CLOSED if fold-held oracle selector improvement is
`< +0.0010` dominant-cell AUC, no specialist argmax varies beyond permutation,
and clusters do not separate loss manifolds better than permutation.

Continuation gate:
WEAK if one of the three gates clears but selector headroom is `< +0.0020`.
OPEN if selector headroom is `>= +0.0020` or two diagnostic gates clear.
HIGH-PRIORITY if selector headroom is `>= +0.0030` and at least one stable
history fingerprint has monotone specialist advantage with absolute Spearman
`>= 0.10`.

Conditions for 5-fold expansion:
Not applicable; this pilot already uses the five dev folds descriptively.

Conditions for bootstrap:
Not applicable.

Runtime ceiling:
20 minutes.

Code paths to be created:
`research/scripts/novel_streams/pilots/pilot01_failure_manifolds.py` and
`research/reports/new_avenues_2026/pilot01_failure_manifolds.md/json`.

## Pilot 2 -- Causal Protective-Relay Score-State

Hypothesis:
A relay-style state machine with explicit operate/reset dynamics applied
causally to RT-600 evidence separates temporary score disturbances from
persistent faults and adds marginal ensemble alpha beyond an exchangeable seed
clone.

Information channel:
Reset dynamics and asymmetric time constants, the B1/B3/B5 arm-F route from
`NEW_AVENUES_2026.md`.

Exact inputs:

* RT-600 fold-0 OOF blend, rebuilt from seven specialists.
* No raw features, no tau, no final online length, no future observations.
* Optional calibration constants derived from the training rows of folds other
  than fold 0 only, using RT-600 score distribution at matched `t >= 20`.

Exact algorithm:

1. Rebuild RT-600 OOF blend via `Ctx.crossfit_blend`.
2. For fold 0 scoring, derive constants from rows in folds `1..4`:
   pickup = within-training RT-600 score q80, dropout = q65, high pickup = q90.
   These are fixed quantiles, not optimized.
3. For every row of every series in causal order, maintain:
   * picked-up flag with Schmitt hysteresis (`score >= pickup` to enter,
     `score <= dropout` to exit)
   * inverse-time operate integral:
     `op_t = clip(op_{t-1} + max(score - pickup, 0) / max(1 - pickup, eps), 0, inf)`
     while picked up, and reset by subtracting `0.25` per non-picked step
   * thermal replica:
     `thermal_t = thermal_{t-1} + (score - thermal_{t-1}) / tau_charge`
     while score exceeds pickup, else
     `thermal_t = thermal_{t-1} * exp(-1 / tau_cool)`
     with `tau_charge = 8`, `tau_cool = 32`
   * recloser count: number of completed pickup-to-reset cycles so far,
     capped at 5.
4. Emit one scalar candidate score, not a tuned blend:
   z-scored fixed arithmetic mean of operate integral, thermal state,
   picked-up dwell time, and capped recloser count, with z scales estimated on
   folds `1..4`.

Exact online state:
`picked`, `operate`, `thermal`, `picked_dwell`, `cycle_count`.

Parameter-selection rule:
All constants fixed above. No validation-label threshold tuning. Fold-0
training rows may define quantiles and z scales only.

Exact variants:
Primary only: relay score-state `RT-1200`. A sensitivity candidate may be
preregistered later only if the primary clears at least WEAK; it will use a new
ID.

Primary candidate:
`RT-1200`, scalar relay state score.

Sensitivity candidate:
None in this preregistered run.

Population:
Fold 0 screen rows for headline; training folds `1..4` only for constants.

Fold used for screen:
Fold 0.

Causal constraints:
At row `t`, the relay state may use only RT-600 scores from the same series at
times `<= t`. Constants are fitted from other folds' OOF rows only.

Control:
`RT600 + RT-401 seed clone`. Secondary context: previously measured score EWMA
at about `+0.0005` and decay/runmax failures; no rerun of that family is used
for selection.

Primary metric:
Fold-0 `marginal_vs_clone` from `wave8_common.ensemble_marginal`.

Dominant-cell metric:
Fold-0 dominant-cell candidate AUC and RT-600 AUC.

Mature-break-vs-never metric:
Fold-0 mature positives versus never-break negatives.

Pair-repair diagnostic:
Sampled same-t pair repairs/damage versus RT-600 in whole fold, dominant cell,
never-break-negative cell, and pre-break-negative cell.

Correlation diagnostic:
Within-t rank correlation with RT-600 in the dominant cell.

Ensemble integration procedure:
`wave8_common.ensemble_marginal`, equal-weight addition of `RT-1200` to the
seven RT-600 streams, compared to equal-weight addition of `RT-401`.

Kill gate:
KILL if `marginal_vs_clone < +0.0010`, or if result lies within `+/-0.0005` of
the previously measured EWMA score-state gain, indicating no new state.

Continuation gate:
WEAK/INTERESTING/SERIOUS gates follow the shared thresholds. No threshold
changes after fold 0.

Conditions for 5-fold expansion:
Run full five-fold relay constants and candidate scores only if fold-0
`marginal_vs_clone >= +0.0020`.

Conditions for bootstrap:
Run paired series bootstrap only if five-fold confirmation remains
`>= +0.0030` and positive on at least 4/5 folds.

Runtime ceiling:
35 minutes.

Code paths to be created:
`research/scripts/novel_streams/pilots/pilot02_relay_logic.py` and
`research/reports/new_avenues_2026/pilot02_relay_logic.md/json`.

## Pilot 3 -- IM2 Matched-Length Empirical Run Null And Dwell Bank

Hypothesis:
The prefix contains a contiguity/growth statistic absent from the 500-column
bank: under stationarity, length-matched historical pseudo-online runs price
the natural growth of maximum run length, while persistent breaks grow faster.
The empirical matched-length null adds marginal ensemble alpha where the naive
closed-form Erdos-Renyi normalization failed.

Information channel:
Path-functional null calibration of dwell/run state, using A1/A2/A5/A7 and IM2.

Exact inputs:

* Raw history and online arrays from `cache/store`, dev folds only.
* History-fitted `HistParams` with AR(2) residual scale.
* RT-600 OOF blend only for diagnostics/integration, not for constructing
  dwell state.
* No final online length, tau, label, future row, or future support.

Exact algorithm:

For each series independently:

1. Fit history-only robust normalization and AR(2) residual parameters using
   `HistParams`.
2. Compute AR-residual-square rolling mean at windows `w in {32, 64, 128}` on
   history and online causally.
3. For each `w`, define the null band by the historical rolling statistic's
   median and q90 absolute deviation. q99 is not scored in the primary to keep
   column count and degrees of freedom down.
4. From the historical hot/cold binary sequence at the same `w`, build an
   empirical matched-length table for each possible elapsed opportunity length
   `L`: all contiguous historical pseudo-online segments of length `L`, their
   maximum hot run length, and their maximum excursion mass.
5. Online at time `t`, with `L = max(t - w + 2, 1)`, compute current run,
   running max run, total exceedance count, contiguity ratio
   `maxrun / max(total_exceedances, 1)`, current excursion mass, running max
   mass, and growth proxy `log1p(maxrun) / log1p(L)`.
6. Convert running max run and running max mass to empirical percentiles against
   the matched-length historical table at the same `L`. If no comparable
   historical segment exists, emit NaN until it exists.
7. Primary scalar candidate `RT-1201` is a fixed z-scored arithmetic mean of
   the nine per-window components: run percentile, mass percentile, and growth
   proxy for `w in {32,64,128}`. Z scales are fitted on fold-0 training rows
   only.

Exact online state:
For each `w`: current run, running max run, total exceedance count, current
mass, running max mass. Matched-length null tables are history-only immutable
state.

Parameter-selection rule:
Windows `{32,64,128}`, threshold q90, AR(2) residual-square channel, and fixed
component mean are frozen here. No grid search, no q99 sensitivity unless a new
preregistration/ID is created.

Exact variants:
Primary only: `RT-1201`.

Primary candidate:
`RT-1201`, scalar IM2+dwell score.

Sensitivity candidate:
None.

Population:
Fold 0 screen rows for headline. Training folds `1..4` may be used only for
z-scale constants and LightGBM-free calibration of the scalar aggregate.

Fold used for screen:
Fold 0.

Causal constraints:
Online row `t` may use only history and online rows `<= t`. Matched-length
nulls are functions of history only and elapsed prefix length `L`, not final
online length.

Control:
`RT600 + RT-401 seed clone`. Secondary reference: `RT-740`/`m10_persist` and
D6 raw/naive normalization are reported as prior controls, not rerun-selected.

Primary metric:
Fold-0 `marginal_vs_clone`.

Dominant-cell metric:
Fold-0 dominant-cell candidate AUC and RT-600 AUC.

Mature-break-vs-never metric:
Fold-0 mature positives versus never-break negatives, plus pre-break split.

Pair-repair diagnostic:
Sampled same-t pair repairs/damage versus RT-600 in the standard splits.

Correlation diagnostic:
Within-t rank correlation with RT-600 in the dominant cell.

Ensemble integration procedure:
`wave8_common.ensemble_marginal`, equal-weight addition of `RT-1201` compared
with equal-weight addition of `RT-401`.

Kill gate:
KILL if `marginal_vs_clone < +0.0010`, or if the age profile shows all gain
concentrated below post-break age 50.

Continuation gate:
INTERESTING at `>= +0.0020`; SERIOUS at `>= +0.0030`, following the shared
thresholds.

Conditions for 5-fold expansion:
Run full five-fold confirmation only if fold-0 `marginal_vs_clone >= +0.0020`
and the dominant-cell mature-vs-never AUC is above the raw D4 `res64_maxrun90`
reference.

Conditions for bootstrap:
Run paired series bootstrap only after a positive five-fold result with
`marginal_vs_clone >= +0.0030` and positive sign on at least 4/5 folds.

Runtime ceiling:
90 minutes for fold-0 screen and diagnostics.

Code paths to be created:
`research/scripts/novel_streams/pilots/pilot03_im2_dwell.py` and
`research/reports/new_avenues_2026/pilot03_im2_dwell.md/json`.
