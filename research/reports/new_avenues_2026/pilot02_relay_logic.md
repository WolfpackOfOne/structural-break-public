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

**Status:** FINAL SCREEN RESULT. Reproduced after the `aca2c4f` novel-stream
harness cleanup merge.

| arm | fold-0 TS-AUC |
|---|---:|
| RT600 | 0.638276 |
| RT600 + RT-401 seed clone | 0.638586 |
| RT600 + RT-1200 | 0.638380 |

Marginal vs clone: `-0.000207`.
Verdict: **KILL**.

## Post-Cleanup Reproduction

Cleanup SHA: `aca2c4f9b68ad6315956f7c499ebefb2ae311f7b`.

Verification: prefix-state check `ok` over `29` prefixes. Numeric tolerance was
set before comparison at `1e-12` for floating metrics; pair-flow counts were
required to match exactly.

| metric | old | clean | delta |
|---|---:|---:|---:|
| standalone whole-fold AUC | 0.619610560 | 0.619610560 | 0 |
| dominant-cell AUC | 0.658229596 | 0.658229596 | 0 |
| mature vs never-break | 0.655859118 | 0.655859118 | 0 |
| within-t rho vs RT600 | 0.774581181 | 0.774581181 | 0 |
| dominant repairs | 1158 | 1158 | 0 |
| dominant damage | 1476 | 1476 | 0 |
| E0 RT600 | 0.638276303 | 0.638276303 | 0 |
| E1 RT600 + seed clone | 0.638586372 | 0.638586372 | 0 |
| E2 RT600 + RT-1200 | 0.638379739 | 0.638379739 | 0 |
| marginal vs clone | -0.000206633 | -0.000206633 | 0 |

Changed: **no**. Final verdict remains **KILL**.

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

Runtime: `64.3s`.
