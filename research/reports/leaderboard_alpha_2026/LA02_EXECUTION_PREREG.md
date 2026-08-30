# LA-02 -- Counterfactual Synthetic Augmentation Execution Preregistration

Date: 2026-08-26

Program preregistration: `research/reports/leaderboard_alpha_2026/PROGRAM_PREREG.md`
at `ed84d00`.

LA-01 result: KILL at `b87759d`; specialist-replacement salvage is closed.

## Allocated IDs

| ID | arm |
|---|---|
| `RT-1245` | LA-02 C1 same-count synthetic null-only control |
| `RT-1246` | LA-02 paired persistent-positive vs transient-hard-negative counterfactual augmentation |

C0 is the existing RT600 seven-specialist ensemble (`RT-300`, `RT-410`-`RT-415`)
and receives no new ID.

## Frozen Design

Train the same seven stream configurations as RT600:

- `RT-300` / `RT-100R`
- `RT-410` / `RT-120R`
- `RT-411` / `RT-121R`
- `RT-412` / `RT-122R`
- `RT-413` / `RT-123R`
- `RT-414` / `RT-124R`
- `RT-415` / `RT-125R`

Every stream keeps its original module subset, seed, row cap, sampling mode,
LightGBM parameters, and objective. Real validation rows are unchanged and
entirely real.

Synthetic ratio: target synthetic rows equal to `33%` of the largest RT600
stream's selected real training rows per outer fold. With the `RT-300` cap this
targets approximately `330,000` synthetic rows per fold and arm. Smaller streams
sample from the same fold's synthetic pool in the same ratio as their selected
real training rows.

## Fold Purity

For each outer fold `f`, all generator distributions are estimated only from
series in folds `{0,1,2,3,4} \ {f}`. Validation fold histories, online values,
labels, and scores do not enter generator fitting.

No synthetic validation row is produced or scored.

## Fixed Null Generator

Each synthetic source series uses only its own break-free history `H_i`:

- standardize by history mean and standard deviation,
- fit fixed Yule-Walker AR(5) on standardized history,
- compute empirical AR residuals from history,
- simulate null continuations by AR(5) recursion plus sampled historical
  residual innovations.

This is the fixed per-series AR(5) plus history-residual ECDF null family used
as CRF-02's fixed-null control. No learned or amortized null is used.

## Candidate Paired Counterfactuals

For each sampled source history and online length, generate one base null
continuation. From that same base path produce:

- persistent positive: label `0` before synthetic break, `1` at and after it;
- hard negative: label all rows `0`, with a transient nonpersistent excursion.

Persistent mechanisms are exactly:

- location
- scale
- dependence

Outer-training labelled break series estimate the empirical online-length
distribution, break-location distribution, mechanism mixture, and signed effect
size pools. The mechanism mixture is estimated by robustly standardizing the
three effect families within the outer-training break series and counting which
family dominates each series.

Transient negative mechanisms are exactly:

- isolated heavy-tail/outlier
- shock/exponential excursion
- variance burst
- temporary displacement

Transient duration is estimated from contiguous high-RT600 false-positive runs
among outer-training no-break rows only. If a fold has no such run, duration
falls back to one row. Transient magnitudes use source-history residual tails
and the outer-training location/scale effect pools. The four transient mechanism
types are sampled uniformly because the historical labels identify persistence,
not a reliable subtype label for transient false positives.

## Controls and Metric

C0: existing RT600 seven-specialist ensemble.

C1 (`RT-1245`): train RT600's seven stream configs on real rows plus the same
number of synthetic fixed-null trajectories, all labelled non-break.

Candidate (`RT-1246`): train RT600's seven stream configs on real rows plus the
paired persistent-positive/transient-negative counterfactual trajectories.

Primary program metric for this data-intervention experiment:

`marginal_vs_clone = RT-1246 mean TS-AUC - RT-1245 mean TS-AUC`.

Also report candidate minus C0, C1 minus C0, fold deltas, dominant-cell pair
repairs/damage/net, mature-vs-never pair net, mature-vs-prebreak pair net,
within-`t` rank correlation with C0, and runtime.

Pair-flow sampling is fixed at 64 same-`t` pairs per time point, seed
`20260826`, candidate-vs-C0.

## Gate

LA-02 survives only if:

- five-fold `marginal_vs_clone >= +0.0025`,
- at least 4/5 folds have positive `RT-1246 - RT-1245`,
- dominant-cell net pair lift is positive,
- mature-vs-never net pair lift is positive,
- `RT-1246 - RT-1245 >= +0.0010`.

`>= +0.0030` is SERIOUS. `>= +0.0050` is MAJOR.

If SERIOUS, breadth stops and LA-02 is confirmed before LA-03. If KILL, proceed
to LA-03.

