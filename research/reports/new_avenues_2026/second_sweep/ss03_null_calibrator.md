# SS-03 -- Negative-Side Null Calibrator

Execution preregistration: `research/reports/new_avenues_2026/second_sweep/SS03_EXECUTION_PREREG.md` at `8a5f41a`.
Experiment IDs: `RT-1225` candidate, `RT-1226` global control, `RT-1227` deranged partition control, `RT-1228` unweighted CTM control, `RT-1229` Pilot-9 scalar control.

## RT-600 Sentinel

* Dev mean TS-AUC: `0.625811`.
* Dev pooled TS-AUC: `0.625627`.
* Dev dominant-cell TS-AUC: `0.664277`.
* Fold-0 E0 RT600: `0.638276`.
* Fold-0 E1 RT600 + RT-401: `0.638586`.

## Binding Result

| arm | standalone fold-0 TS-AUC | E2 TS-AUC | marginal vs clone | gain vs RT600 | mature-vs-never net | mature-vs-prebreak net |
|---|---:|---:|---:|---:|---:|---:|
| RT600 |  | 0.638276 |  |  |  |  |
| RT600 + RT-401 seed clone |  | 0.638586 |  |  |  |  |
| RT600 + RT-1225 | 0.637193 | 0.638287 | -0.000299 | +0.000011 | -49 | -40 |
| RT600 + RT-1226 | 0.638275 | 0.638276 | -0.000310 | -0.000000 | -3 | 3 |
| RT600 + RT-1227 | 0.637208 | 0.638238 | -0.000348 | -0.000038 | -63 | -92 |
| RT600 + RT-1228 | 0.637247 | 0.638296 | -0.000290 | +0.000020 | -48 | -74 |
| RT600 + RT-1229 | 0.637111 | 0.638251 | -0.000335 | -0.000025 | -58 | -156 |

Verdict: **KILL**.
Gate failures: `never_break_dominant_pair_net` observed `-49`; `marginal_vs_clone` observed `-0.0002988941787396282`; `prebreak_damage_rate_cap` observed `0.019902604276942622`; `deranged_control_marginal_gap` observed `4.905971037938439e-05`; `RT-1228_control_margin_not_lower` observed `-8.841006806981078e-06`

## Candidate Pair Flow

| split | pairs | RT600 wrong | RT600 right | repairs | damage | net | damage rate on RT600-right |
|---|---:|---:|---:|---:|---:|---:|---:|
| `whole_fold` | 63380 | 22143 | 41237 | 722 | 866 | -144 | 0.0210 |
| `dominant_cell` | 50540 | 16193 | 34347 | 659 | 773 | -114 | 0.0225 |
| `mature_vs_never` | 50540 | 15804 | 34736 | 678 | 727 | -49 | 0.0209 |
| `mature_vs_prebreak` | 48677 | 15616 | 33061 | 618 | 658 | -40 | 0.0199 |

## Control Summary

| control | marginal vs clone | candidate minus control | mature-vs-never net | dominant net |
|---|---:|---:|---:|---:|
| `RT-1226` | -0.000310 | +0.000012 | -3 | -4 |
| `RT-1227` | -0.000348 | +0.000049 | -63 | -45 |
| `RT-1228` | -0.000290 | -0.000009 | -48 | -120 |
| `RT-1229` | -0.000335 | +0.000037 | -58 | -164 |

## Fold-0 State Audits

* Candidate valid state counts: `{'0': 290762, '1': 43792, '2': 92774, '3': 68360, '4': 49396, '5': 61075, '6': 66092, '7': 83034, '8': 51049}`.
* Candidate eval fallback counts: `{'leaf_0': 290762, 'leaf_1': 43792, 'leaf_2': 92774, 'leaf_3': 68360, 'leaf_4': 49396, 'leaf_5': 61075, 'leaf_6': 66092, 'leaf_7': 83034, 'leaf_8': 51049}`.
* Candidate thresholds: `{'dispersion_median': 0.13061124086380005, 'weighted_ctm_suppression_median': -1.0793898105621338, 'unweighted_ctm_tail_median': -5.644867897033691, 'pilot9_scalar_median': 0.5239032506942749, 'mature_train_rows': 2066655}`.
* Deranged audit: `{'seed': 2026082507, 'groups': 999, 'groups_shifted': 797, 'groups_single_state': 202, 'rows': 806334, 'rows_changed': 436126, 'fixed_rows_after_shift': 370208, 'same_time_multisets_preserved': True}`.

## Implementation Audits

* Score coverage: all required frozen OOF scores finite on dev rows and zero finite on lockbox rows.
* CTM feature coverage: `{'weighted': {'module': 'm18_wctm', 'column': 'wctm_tail_log', 'shape': [5036517, 8], 'finite_dev_rows': 4032524, 'expected_dev_rows': 4032524, 'finite_lockbox_rows': 0}, 'unweighted': {'module': 'm18_uctm', 'column': 'uctm_tail_log', 'shape': [5036517, 8], 'finite_dev_rows': 4032524, 'expected_dev_rows': 4032524, 'finite_lockbox_rows': 0}}`.
* Pair-sample reproduction: `{'dominant_pairs': 50540, 'dominant_rt600_wrong': 16193, 'dominant_rt600_right': 34347}`.
* New OOF lockbox finite counts: `{'RT-1225': 0, 'RT-1226': 0, 'RT-1227': 0, 'RT-1228': 0, 'RT-1229': 0}`.
* Peak RSS: `{'ru_maxrss': 2327412736, 'approx_mib': 2219.59375, 'platform': 'darwin'}`.
* Runtime: `513.9s`.

## Interpretation

SS-03 is KILL under the preregistered fold-0 screen. The fixed weighted-CTM null-state conditional SCDF correction did not satisfy all mandatory never-break, marginal, pre-break-damage, and control gates. No SS-03b, threshold adjustment, correction-scale change, CTM retuning, or Pilot-9 scalar variant is authorized.
