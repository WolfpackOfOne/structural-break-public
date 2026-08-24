# PILOT 3 -- IM2 MATCHED-LENGTH RUN NULL AND DWELL BANK

Experiment ID: `RT-1201`.

## RT-600 Anchor

* Dev mean TS-AUC: `0.625811`.
* Dev pooled TS-AUC: `0.625627`.
* Fold-0 RT-600 TS-AUC in integration: `0.638276`.

## Candidate

Nine-feature scalar from AR(2)-residual-square dwell state over windows `32,64,128`: matched-length run percentile, episode-mass percentile, and max-run growth proxy per window.

**Status:** PRE-CLEANUP / PROVISIONAL. The expected novel-stream cleanup commit has not landed on `origin/research/current` yet.

* Prefix verification: `ok` over `29` prefixes.
* Feature build runtime: `178.0s`.

## Binding Marginal Result

| arm | fold-0 TS-AUC |
|---|---:|
| RT600 | 0.638276 |
| RT600 + RT-401 seed clone | 0.638586 |
| RT600 + RT-1201 | 0.638888 |

Marginal vs clone: `+0.000301`.
Verdict: **KILL**.

## Diagnostic Pack

* Whole fold candidate TS-AUC: `0.583578` (RT-600 `0.638276`).
* Dominant-cell candidate AUC: `0.610158` (RT-600 `0.677711`).
* Mature vs never-break: `0.613459`.
* Mature vs pre-break: `0.600945`.
* Within-t correlation with RT-600: `+0.3817`.

## Pair Flow

| split | repairs | damage | net | sampled pairs |
|---|---:|---:|---:|---:|
| `whole_fold` | 2797 | 4244 | -1447 | 19562 |
| `dominant_cell` | 2021 | 3172 | -1151 | 15598 |
| `cell_never_break_neg` | 2053 | 3107 | -1054 | 15582 |
| `cell_pre_break_neg` | 1598 | 2840 | -1242 | 13378 |

## Interpretation

The binding decision is the marginal-vs-clone comparison. Standalone dwell separation and low correlation are necessary diagnostics, not promotion criteria.

Run-length percentile uses exact interval-union matched-length history null. Mass percentile uses a history-only episode-length empirical null as the cheap mass companion.

Runtime: `254.4s`.
