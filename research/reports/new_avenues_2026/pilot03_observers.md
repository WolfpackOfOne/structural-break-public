# PILOT 3 / IM3 -- INDIVIDUAL OBSERVER RESIDUALS

Pre-registration: `research/reports/new_avenues_2026/PILOT03_OBSERVERS_PREREG.md` at `0a9d97e`.
Experiment IDs: `RT-1214` Kalman/NIS arm, `RT-1215` Hankel-DMD arm.

## RT-600 Anchor

* Dev mean TS-AUC: `0.625811`.
* Dev pooled TS-AUC: `0.625627`.
* Dev dominant-cell TS-AUC: `0.664277`.
* Fold-0 E0 RT600: `0.638276`.
* Fold-0 E1 RT600 + RT-401: `0.638586`.

## Candidate Arms

`RT-1214` is a per-series frozen AR(2)-state Kalman observer with fixed history-only `q x r` noise-grid selection. It emits NIS accumulation, windowed NIS excess, and normalized-innovation whiteness.

`RT-1215` is a per-series frozen Hankel-DMD observer with delay `16`, rank `4`, horizons `1` and `5`, historical-subspace residual, and effective-rank monitors.

The Mode-A base bank already includes `m04_resid`; the binding marginal therefore asks whether either observer adds value beyond existing scalar residual monitors and beyond the `RT-401` seed clone.

## Causality Checks

* Harness Kalman/NIS: `ok`.
* Harness Hankel-DMD: `ok`.
* Future mutation prefix check: `True`.
* Deterministic replay: `True`.
* First-valid semantics: `{'kalman_nis_log_and_cum_finite': True, 'kalman_rows_before_31_w32_nan': True, 'kalman_row31_w32_finite': True, 'kalman_rows_before_63_w64_nan': True, 'kalman_row63_w64_finite': True, 'dmd_row0_all_finite_with_history_warmup': True}`.
* History-only fit replay: `{'kalman_phi_replay_equal': True, 'kalman_grid_choice_replay_equal': True, 'dmd_operator_replay_equal': True, 'dmd_subspace_replay_equal': True, 'dmd_nulls_replay_equal': True, 'dmd_fitted_rank': 4}`.
* m04 boundary note: Mode A already includes m04_resid in the base bank; these arms test incremental observer residual value beyond that existing residual block.
* m07 parity note: Pilot 3 observers do not recompute m07_bayes::bo_p_lt25_z; they use the existing cached base-bank features and independent observer blocks.

## Binding Marginal Result

| arm | fold-0 standalone TS-AUC | E2 TS-AUC | marginal vs clone | gain vs RT600 | rho vs RT600 | verdict |
|---|---:|---:|---:|---:|---:|---|
| RT600 |  | 0.638276 |  |  |  |  |
| RT600 + RT-401 seed clone |  | 0.638586 |  |  |  |  |
| RT600 + RT-1214 | 0.630817 | 0.638722 | +0.000135 | +0.000446 | +0.8836 | KILL (KILL) |
| RT600 + RT-1215 | 0.631331 | 0.638812 | +0.000226 | +0.000536 | +0.8859 | KILL (KILL) |

Overall continuation status: **CONTINUE_TO_PILOT8**.
Kalman failed gates: [{'gate': 'primary_marginal_vs_clone', 'threshold': 0.001, 'observed': 0.00013548394103102268, 'message': 'kalman_nis marginal_vs_clone is below +0.0010'}].
Hankel-DMD failed gates: [{'gate': 'primary_marginal_vs_clone', 'threshold': 0.001, 'observed': 0.00022599709153225955, 'message': 'hankel_dmd marginal_vs_clone is below +0.0010'}, {'gate': 'hankel_redundancy_rho', 'threshold': 0.85, 'observed': 0.8858580997826474, 'message': 'Hankel-DMD within-t rank correlation with RT600 exceeds 0.85'}].

## Diagnostic Pack

| arm | whole-fold AUC | dominant-cell AUC | mature vs never | mature vs pre | rho vs RT600 |
|---|---:|---:|---:|---:|---:|
| `RT-1214` | 0.630817 | 0.670067 | 0.667936 | 0.676015 | +0.8836 |
| `RT-1215` | 0.631331 | 0.673238 | 0.669922 | 0.682489 | +0.8859 |

## Pair Flow

### RT-1214

| split | repairs | damage | net | sampled pairs |
|---|---:|---:|---:|---:|
| `whole_fold` | 1280 | 1475 | -195 | 19562 |
| `dominant_cell` | 927 | 1025 | -98 | 15598 |
| `cell_never_break_neg` | 898 | 1056 | -158 | 15582 |
| `cell_pre_break_neg` | 884 | 834 | 50 | 13378 |

### RT-1215

| split | repairs | damage | net | sampled pairs |
|---|---:|---:|---:|---:|
| `whole_fold` | 1308 | 1437 | -129 | 19562 |
| `dominant_cell` | 920 | 931 | -11 | 15598 |
| `cell_never_break_neg` | 853 | 983 | -130 | 15582 |
| `cell_pre_break_neg` | 839 | 790 | 49 | 13378 |

## Interpretation

Neither observer arm cleared the preregistered 5-fold continuation gate. This kills or shelves these exact frozen-observer constructions under the Mode-A fold-0 ABL screen, with m04_resid already present in the base bank; it does not falsify all state-space or delay-embedding observer ideas.

Feature build runtime: `2898.9s`.
Total runtime: `4120.5s`.
