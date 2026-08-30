# PILOT 4 -- TRAJECTORY GEOMETRY

Experiment IDs: `RT-1202` real scalar, `RT-1203` shuffled-history control.

## RT-600 Anchor

* Dev mean TS-AUC: `0.625811`.
* Dev pooled TS-AUC: `0.625627`.
* Fold-0 RT-600 TS-AUC in integration: `0.638276`.

## Candidate

Four-feature scalar from z-normalized subsequence windows `16,64`: nearest-neighbour provenance and history-boundary arc-rate per window. The shuffled control preserves distance/provenance and permutes the historical order used by the arc-rate component.

**Status:** FINAL SCREEN RESULT. Reproduced after the `aca2c4f` novel-stream
harness cleanup merge.

* Prefix verification: `ok` over `29` prefixes.
* Feature build runtime: `165.6s`.

## Binding Marginal Result

| arm | fold-0 TS-AUC | marginal vs clone | gain vs RT600 |
|---|---:|---:|---:|
| RT600 | 0.638276 |  |  |
| RT600 + RT-401 seed clone | 0.638586 |  |  |
| RT600 + RT-1202 | 0.636000 | -0.002587 | -0.002277 |
| RT600 + RT-1203 | 0.635545 | -0.003042 | -0.002732 |

Real minus shuffled marginal: `+0.000455`.
Verdict: **KILL** -- real marginal_vs_clone is below +0.0010.

## Post-Cleanup Reproduction

Cleanup SHA: `aca2c4f9b68ad6315956f7c499ebefb2ae311f7b`.

Verification: prefix check `ok` over `29` prefixes. Numeric tolerance was set
before comparison at `1e-12` for floating metrics; pair-flow counts were
required to match exactly.

| candidate | metric | old | clean | delta |
|---|---|---:|---:|---:|
| `RT-1202` | standalone whole-fold AUC | 0.499324115 | 0.499324115 | 0 |
| `RT-1202` | dominant-cell AUC | 0.500307812 | 0.500307812 | 0 |
| `RT-1202` | mature vs never-break | 0.501410131 | 0.501410131 | 0 |
| `RT-1202` | within-t rho vs RT600 | 0.003912508 | 0.003912508 | 0 |
| `RT-1202` | dominant repairs | 2460 | 2460 | 0 |
| `RT-1202` | dominant damage | 5434 | 5434 | 0 |
| `RT-1202` | E2 RT600 + candidate | 0.635999515 | 0.635999515 | 0 |
| `RT-1202` | marginal vs clone | -0.002586857 | -0.002586857 | 0 |
| `RT-1203` | standalone whole-fold AUC | 0.493823753 | 0.493823753 | 0 |
| `RT-1203` | dominant-cell AUC | 0.489650163 | 0.489650163 | 0 |
| `RT-1203` | mature vs never-break | 0.486356249 | 0.486356249 | 0 |
| `RT-1203` | within-t rho vs RT600 | 0.007449945 | 0.007449945 | 0 |
| `RT-1203` | dominant repairs | 2357 | 2357 | 0 |
| `RT-1203` | dominant damage | 5487 | 5487 | 0 |
| `RT-1203` | E2 RT600 + candidate | 0.635544744 | 0.635544744 | 0 |
| `RT-1203` | marginal vs clone | -0.003041629 | -0.003041629 | 0 |

Changed: **no**. Final verdict remains **KILL** for `RT-1202`; `RT-1203`
remains a negative shuffled-order control.

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

Runtime: `311.4s`.
