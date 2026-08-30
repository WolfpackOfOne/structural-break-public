# PILOT 8 -- WEIGHTED CONFORMAL TEST MARTINGALE

Pre-registration: `research/reports/new_avenues_2026/PILOT08_PREREG.md` at `83c3994`.
Experiment IDs: `RT-1216` weighted CTM, `RT-1217` unweighted CTM control, `RT-1218` H2 e-value aggregation.

## RT-600 Anchor

* Dev mean TS-AUC: `0.625811`.
* Dev pooled TS-AUC: `0.625627`.
* Dev dominant-cell TS-AUC: `0.664277`.
* Fold-0 E0 RT600: `0.638276`.
* Fold-0 E1 RT600 + RT-401: `0.638586`.

## Candidate

The weighted arm scales the CTM betting fraction by a predictable recent tail-rate weight. The unweighted arm fixes the same weight to 1. H2 is a direct log-average of five existing `m07_bayes` e-process log-capitals.

## Causality Checks

* Harness weighted: `ok`.
* Harness unweighted: `ok`.
* Future mutation prefix check: `True`.
* Deterministic replay: `True`.
* Omega predictability: `{'omega_row0_excludes_current': True, 'omega_row1_uses_prior_row': True, 'omega_min_observed': 0.25, 'omega_max_observed': 1.0}`.
* H2 columns: `['ev_tail_mix', 'ev_tail_ad', 'ev_disp_mix', 'ev_disp_ad', 'ev_pow_mix']`.
* m07 parity note: Pilot 8 H1 does not recompute m07_bayes; H2 reads only existing e-process log-capital columns and excludes the known BOCPD z/parity path.

## Binding Marginal Result

| arm | fold-0 standalone TS-AUC | E2 TS-AUC | marginal vs clone | gain vs RT600 | mature-vs-never | verdict |
|---|---:|---:|---:|---:|---:|---|
| RT600 |  | 0.638276 |  |  |  |  |
| RT600 + RT-401 seed clone |  | 0.638586 |  |  |  |  |
| RT600 + RT-1216 | 0.634279 | 0.639523 | +0.000937 | +0.001247 | 0.670290 | KILL (KILL) |
| RT600 + RT-1217 | 0.629803 | 0.638799 | +0.000212 | +0.000523 | 0.668155 | control |
| RT600 + RT-1218 | 0.525678 | 0.636373 | -0.002214 | -0.001904 | 0.540505 | KILL (KILL) |

Weighted minus unweighted mature-vs-never AUC: `+0.002135`.
Overall continuation status: **FIRST_SWEEP_EXHAUSTED**.
Weighted failed gates: [{'gate': 'primary_marginal_vs_clone', 'threshold': 0.001, 'observed': 0.0009365678242838626, 'message': 'weighted CTM marginal_vs_clone is below +0.0010'}].
H2 failed gates: [{'gate': 'primary_marginal_vs_clone', 'threshold': 0.001, 'observed': -0.0022136261597894835, 'message': 'H2 marginal_vs_clone is below +0.0010'}].

## Diagnostic Pack

| arm | whole-fold AUC | dominant-cell AUC | mature vs never | mature vs pre | rho vs RT600 |
|---|---:|---:|---:|---:|---:|
| `RT-1216` | 0.634279 | 0.673986 | 0.670290 | 0.684301 | +0.8653 |
| `RT-1217` | 0.629803 | 0.670833 | 0.668155 | 0.678303 | +0.8683 |
| `RT-1218` | 0.525678 | 0.536757 | 0.540505 | 0.526296 | +0.1341 |

## Pair Flow

### RT-1216

| split | repairs | damage | net | sampled pairs |
|---|---:|---:|---:|---:|
| `whole_fold` | 1450 | 1517 | -67 | 19562 |
| `dominant_cell` | 992 | 1027 | -35 | 15598 |
| `cell_never_break_neg` | 928 | 1079 | -151 | 15582 |
| `cell_pre_break_neg` | 898 | 843 | 55 | 13378 |

### RT-1217

| split | repairs | damage | net | sampled pairs |
|---|---:|---:|---:|---:|
| `whole_fold` | 1380 | 1517 | -137 | 19562 |
| `dominant_cell` | 999 | 1044 | -45 | 15598 |
| `cell_never_break_neg` | 937 | 1030 | -93 | 15582 |
| `cell_pre_break_neg` | 803 | 901 | -98 | 13378 |

### RT-1218

| split | repairs | damage | net | sampled pairs |
|---|---:|---:|---:|---:|
| `whole_fold` | 3078 | 5375 | -2297 | 19562 |
| `dominant_cell` | 2202 | 4369 | -2167 | 15598 |
| `cell_never_break_neg` | 2193 | 4233 | -2040 | 15582 |
| `cell_pre_break_neg` | 1675 | 4093 | -2418 | 13378 |

## Interpretation

No Pilot 8 arm cleared the preregistered 5-fold continuation gate. This kills or shelves the exact weighted CTM and parameter-free e-value aggregation constructions under the fold-0 screen and exhausts the planned first-sweep queue.

Feature build runtime: `393.6s`.
Total runtime: `1651.5s`.
