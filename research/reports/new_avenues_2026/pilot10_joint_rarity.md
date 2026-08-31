# PILOT 10 -- JOINT SIZE-DURATION RARITY

Pre-registration: `research/reports/new_avenues_2026/PILOT10_PREREG.md` at `58ee792`.
Experiment IDs: `RT-1210` joint rarity candidate, `RT-1211` dwell-only control.

## RT-600 Anchor

* Dev mean TS-AUC: `0.625811`.
* Dev pooled TS-AUC: `0.625627`.
* Dev dominant-cell TS-AUC: `0.664277`.
* Fold-0 E0 RT600: `0.638276`.
* Fold-0 E1 RT600 + RT-401: `0.638586`.

## Candidate

AR(2) residual-square rolling means use windows `32,64,128` and the Pilot-3 q90 excursion band. The candidate emits live and running-max one-sided joint rarity of excursion peak excess and live duration. The control emits the matched duration-only rarity.

## Causality Checks

* Harness candidate: `ok`.
* Harness dwell control: `ok`.
* Future mutation prefix check: `True`.
* Deterministic replay: `True`.
* First-valid semantics: `{'candidate_rows_before_31_all_nan': True, 'control_rows_before_31_all_nan': True, 'candidate_row31_w32_finite': True, 'control_row31_w32_finite': True, 'candidate_row62_w64_all_nan': True, 'control_row62_w64_all_nan': True, 'candidate_row63_w32_w64_finite': True, 'control_row63_w32_w64_finite': True, 'candidate_row126_w128_all_nan': True, 'control_row126_w128_all_nan': True, 'candidate_row127_all_finite': True, 'control_row127_all_finite': True}`.
* Joint surprise >= dwell surprise: `True`.
* m07 parity note: Pilot 10 does not recompute m07_bayes::bo_p_lt25_z; it uses the existing cached base-bank features and independent AR(2) residual-square streams.

## Binding Marginal Result

| arm | fold-0 standalone TS-AUC | E2 TS-AUC | marginal vs clone | gain vs RT600 |
|---|---:|---:|---:|---:|
| RT600 |  | 0.638276 |  |  |
| RT600 + RT-401 seed clone |  | 0.638586 |  |  |
| RT600 + RT-1210 | 0.626121 | 0.638065 | -0.000522 | -0.000211 |
| RT600 + RT-1211 | 0.627141 | 0.638350 | -0.000237 | +0.000073 |

Candidate minus dwell-control marginal: `-0.000285`.
Candidate minus dwell-control dominant-cell AUC: `-0.000896`.
Verdict: **KILL** (KILL) -- candidate marginal_vs_clone is below +0.0010.
Failed gates: `primary_marginal_vs_clone` (candidate marginal_vs_clone is below +0.0010); `dwell_control_not_worse` (dwell-only control matches or exceeds joint-rarity candidate); `dominant_cell_i1_beats_a1` (joint-rarity candidate does not beat dwell-only control on the dominant cell)

## Diagnostic Pack

| candidate | whole-fold AUC | dominant-cell AUC | mature vs never | mature vs pre | rho vs RT600 |
|---|---:|---:|---:|---:|---:|
| `RT-1210` | 0.626121 | 0.663758 | 0.659593 | 0.675382 | +0.8779 |
| `RT-1211` | 0.627141 | 0.664654 | 0.662531 | 0.670579 | +0.8704 |

## Secondary Dwell Reference

`RT-1201`: `{'available': True, 'exp_id': 'RT-1201', 'marginal_vs_clone': 0.0003014699051322456, 'dominant_cell_auc': 0.6101579099132464, 'within_t_rank_corr_rt600': 0.38170592359325234, 'note': 'Secondary reference only; RT-1211 is the matched Mode-A dwell control.'}`.

## Pair Flow

### RT-1210

| split | repairs | damage | net | sampled pairs |
|---|---:|---:|---:|---:|
| `whole_fold` | 1260 | 1494 | -234 | 19562 |
| `dominant_cell` | 902 | 1139 | -237 | 15598 |
| `cell_never_break_neg` | 854 | 1188 | -334 | 15582 |
| `cell_pre_break_neg` | 737 | 949 | -212 | 13378 |

### RT-1211

| split | repairs | damage | net | sampled pairs |
|---|---:|---:|---:|---:|
| `whole_fold` | 1325 | 1639 | -314 | 19562 |
| `dominant_cell` | 941 | 1146 | -205 | 15598 |
| `cell_never_break_neg` | 926 | 1131 | -205 | 15582 |
| `cell_pre_break_neg` | 799 | 1052 | -253 | 13378 |

## Interpretation

The preregistered joint size-duration rarity block failed a Pilot 10 binding gate. This falsifies the I1 endpoint-null construction under the fixed AR(2) residual-square channel, windows 32/64/128, q90 excursion band, historical joint endpoint table, matched dwell-only control, and Mode-A fold-0 ABL screen; it does not falsify all large-deviation or scan-statistic approaches.

Feature build runtime: `390.7s`.
Total runtime: `1580.6s`.
