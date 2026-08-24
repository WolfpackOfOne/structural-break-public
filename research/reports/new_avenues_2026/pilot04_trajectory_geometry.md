# PILOT 4 -- TRAJECTORY GEOMETRY

Experiment IDs: `RT-1202` real scalar, `RT-1203` shuffled-history control.

## RT-600 Anchor

* Dev mean TS-AUC: `0.625811`.
* Dev pooled TS-AUC: `0.625627`.
* Fold-0 RT-600 TS-AUC in integration: `0.638276`.

## Candidate

Four-feature scalar from z-normalized subsequence windows `16,64`: nearest-neighbour provenance and history-boundary arc-rate per window. The shuffled control preserves distance/provenance and permutes the historical order used by the arc-rate component.

**Status:** PRE-CLEANUP / PROVISIONAL. The expected novel-stream cleanup commit has not landed on `origin/research/current` yet.

* Prefix verification: `ok` over `29` prefixes.
* Feature build runtime: `169.5s`.

## Binding Marginal Result

| arm | fold-0 TS-AUC | marginal vs clone | gain vs RT600 |
|---|---:|---:|---:|
| RT600 | 0.638276 |  |  |
| RT600 + RT-401 seed clone | 0.638586 |  |  |
| RT600 + RT-1202 | 0.636000 | -0.002587 | -0.002277 |
| RT600 + RT-1203 | 0.635545 | -0.003042 | -0.002732 |

Real minus shuffled marginal: `+0.000455`.
Verdict: **KILL** -- real marginal_vs_clone is below +0.0010.

## Diagnostic Pack

| candidate | whole-fold AUC | dominant-cell AUC | mature vs never | mature vs pre | rho vs RT600 |
|---|---:|---:|---:|---:|---:|
| `RT-1202` | 0.499324 | 0.500308 | 0.501410 | 0.497232 | +0.0039 |
| `RT-1203` | 0.493824 | 0.489650 | 0.486356 | 0.498841 | +0.0074 |

## Pair Flow

### RT-1202

| split | repairs | damage | net | sampled pairs |
|---|---:|---:|---:|---:|
| `whole_fold` | 3287 | 6546 | -3259 | 19562 |
| `dominant_cell` | 2460 | 5434 | -2974 | 15598 |
| `cell_never_break_neg` | 2521 | 5393 | -2872 | 15582 |
| `cell_pre_break_neg` | 1998 | 4614 | -2616 | 13378 |

### RT-1203

| split | repairs | damage | net | sampled pairs |
|---|---:|---:|---:|---:|
| `whole_fold` | 3285 | 6471 | -3186 | 19562 |
| `dominant_cell` | 2357 | 5487 | -3130 | 15598 |
| `cell_never_break_neg` | 2408 | 5522 | -3114 | 15582 |
| `cell_pre_break_neg` | 2020 | 4572 | -2552 | 13378 |

## Interpretation

The binding decision is the marginal-vs-clone gate plus the shuffled-order control. Standalone separation and low RT-600 correlation are diagnostic only unless the ensemble marginal clears the preregistered thresholds.

Runtime: `328.3s`.
