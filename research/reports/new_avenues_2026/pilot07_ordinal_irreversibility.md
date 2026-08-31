# PILOT 7 -- ORDINAL TRANSITION DIVERGENCE AND TIME IRREVERSIBILITY

Pre-registration: `research/reports/new_avenues_2026/PILOT07_PREREG.md` at `1e7e42c`.
Experiment IDs: `RT-1208` transition/asymmetry candidate, `RT-1209` entropy-only control.

## RT-600 Anchor

* Dev mean TS-AUC: `0.625811`.
* Dev pooled TS-AUC: `0.625627`.
* Dev dominant-cell TS-AUC: `0.664277`.
* Fold-0 E0 RT600: `0.638276`.
* Fold-0 E1 RT600 + RT-401: `0.638586`.

## Candidate

Order-3 ordinal codes use the `m03_dyn` tie convention. The candidate emits expanding and trailing-half-prefix transition KL z/raw features plus signed Ramsey-Rothman increment-asymmetry z/raw features. The control emits only matching permutation-entropy features.

## Causality Checks

* Harness candidate: `ok`.
* Harness entropy control: `ok`.
* Future mutation prefix check: `True`.
* Deterministic replay: `True`.
* First-valid semantics: `{'candidate_rows_before_15_all_nan': True, 'control_rows_before_15_all_nan': True, 'candidate_row15_exp_finite': True, 'control_row15_exp_finite': True, 'candidate_row30_half_all_nan': True, 'control_row30_half_all_nan': True, 'candidate_row31_all_finite': True, 'control_row31_all_finite': True}`.
* m07 parity note: Pilot 7 does not recompute m07_bayes::bo_p_lt25_z; it uses the existing cached base-bank features and independent new ordinal streams.

## Binding Marginal Result

| arm | fold-0 standalone TS-AUC | E2 TS-AUC | marginal vs clone | gain vs RT600 |
|---|---:|---:|---:|---:|
| RT600 |  | 0.638276 |  |  |
| RT600 + RT-401 seed clone |  | 0.638586 |  |  |
| RT600 + RT-1208 | 0.629141 | 0.638551 | -0.000036 | +0.000274 |
| RT600 + RT-1209 | 0.626836 | 0.638079 | -0.000508 | -0.000198 |

Candidate minus entropy-control marginal: `+0.000472`.
Verdict: **KILL** (KILL) -- candidate marginal_vs_clone is below +0.0010.
Failed gates: `primary_marginal_vs_clone` (candidate marginal_vs_clone is below +0.0010); `candidate_control_distinguishability` (candidate-control gap is below the preregistered +0.0005 distinguishability floor)

## Diagnostic Pack

| candidate | whole-fold AUC | dominant-cell AUC | mature vs never | mature vs pre | rho vs RT600 |
|---|---:|---:|---:|---:|---:|
| `RT-1208` | 0.629141 | 0.668268 | 0.664570 | 0.678588 | +0.8780 |
| `RT-1209` | 0.626836 | 0.663077 | 0.663400 | 0.662175 | +0.8850 |

## Pair Flow

### RT-1208

| split | repairs | damage | net | sampled pairs |
|---|---:|---:|---:|---:|
| `whole_fold` | 1320 | 1492 | -172 | 19562 |
| `dominant_cell` | 925 | 1000 | -75 | 15598 |
| `cell_never_break_neg` | 923 | 1054 | -131 | 15582 |
| `cell_pre_break_neg` | 751 | 882 | -131 | 13378 |

### RT-1209

| split | repairs | damage | net | sampled pairs |
|---|---:|---:|---:|---:|
| `whole_fold` | 1270 | 1496 | -226 | 19562 |
| `dominant_cell` | 863 | 1139 | -276 | 15598 |
| `cell_never_break_neg` | 826 | 1057 | -231 | 15582 |
| `cell_pre_break_neg` | 636 | 1017 | -381 | 13378 |

## Interpretation

The preregistered order-3 ordinal transition divergence and time-irreversibility block failed a Pilot 7 binding gate. This falsifies this L3/L4 construction under the fixed tie convention, transition KL, Ramsey-Rothman asymmetry, matched-count historical-null calibration, and Mode-A fold-0 ABL screen; it does not falsify all ordinal or nonlinear dynamics representations.

Feature build runtime: `421.5s`.
Total runtime: `1627.8s`.
