# PILOT 2 -- RELAY SCORE-STATE LOGIC

Experiment ID: `RT-1200`.

## RT-600 Anchor

* Dev mean TS-AUC: `0.625811`.
* Dev pooled TS-AUC: `0.625627`.
* Fold-0 RT-600 TS-AUC in integration: `0.638276`.

## Candidate

Causal protective-relay state applied to the RT-600 score path with fixed pickup/dropout quantiles from folds 1..4 and no label-tuned thresholds.

* Pickup q80: `0.710468`.
* Dropout q65: `0.589428`.
* Prefix verification: `ok` over `29` prefixes.

## Binding Marginal Result

**Status:** PRE-CLEANUP / PROVISIONAL. The expected novel-stream cleanup commit has not landed on `origin/research/current` yet.

| arm | fold-0 TS-AUC |
|---|---:|
| RT600 | 0.638276 |
| RT600 + RT-401 seed clone | 0.638586 |
| RT600 + RT-1200 | 0.638380 |

Marginal vs clone: `-0.000207`.
Verdict: **KILL**.

## Diagnostic Pack

* Whole fold candidate TS-AUC: `0.619611` (RT-600 `0.638276`).
* Dominant-cell candidate AUC: `0.658230` (RT-600 `0.677711`).
* Mature vs never-break: `0.655859`.
* Mature vs pre-break: `0.664844`.
* Within-t correlation with RT-600: `+0.7746`.

## Pair Flow

| split | repairs | damage | net | sampled pairs |
|---|---:|---:|---:|---:|
| `whole_fold` | 1443 | 2361 | -918 | 19562 |
| `dominant_cell` | 1158 | 1476 | -318 | 15598 |
| `cell_never_break_neg` | 1131 | 1481 | -350 | 15582 |
| `cell_pre_break_neg` | 923 | 1243 | -320 | 13378 |

## Interpretation

The kill/continue decision is based only on the marginal-vs-clone result. Standalone candidate behavior is diagnostic.

Runtime: `99.6s`.
