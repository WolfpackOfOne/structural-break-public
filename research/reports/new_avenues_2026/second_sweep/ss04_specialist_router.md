# SS-04 -- Specialist Disagreement Micro-Router

Execution preregistration: `research/reports/new_avenues_2026/second_sweep/SS04_EXECUTION_PREREG.md` at `25f40ac`.
Experiment IDs: `RT-1230` candidate, `RT-1231` global reweighting control, `RT-1232` Pilot-1 static selector control, `RT-1233` shuffled target control.

## RT-600 Sentinel

* Dev mean TS-AUC: `0.625811`.
* Dev pooled TS-AUC: `0.625627`.
* Dev dominant-cell TS-AUC: `0.664277`.
* Fold-0 E0 RT600: `0.638276`.
* Fold-0 E1 RT600 + RT-401: `0.638586`.

## Binding Result

| arm | standalone fold-0 TS-AUC | E2 TS-AUC | marginal vs clone | gain vs RT600 | dominant net | majority net | near-split net |
|---|---:|---:|---:|---:|---:|---:|---:|
| RT600 |  | 0.638276 |  |  |  |  |  |
| RT600 + RT-401 seed clone |  | 0.638586 |  |  |  |  |  |
| RT600 + RT-1230 | 0.638266 | 0.638275 | -0.000312 | -0.000002 | 0 | 0 | 0 |
| RT600 + RT-1231 | 0.632596 | 0.638145 | -0.000441 | -0.000131 | -178 | -1499 | -78 |
| RT600 + RT-1232 | 0.628751 | 0.638031 | -0.000555 | -0.000245 | -552 | -1749 | -235 |
| RT600 + RT-1233 | 0.638276 | 0.638276 | -0.000310 | -0.000000 | 0 | 0 | 0 |

Verdict: **KILL**.
Gate failures: `marginal_vs_clone` observed `-0.0003117947941426724`; `majority_correct_pair_net` observed `0`; `near_split_pair_net` observed `0`; `RT-1231_control_gap` observed `0.00012966932113767093`; `RT-1232_control_gap` observed `0.00024323577243845484`; `RT-1233_control_gap` observed `-1.6927248839282427e-06`

## Candidate Pair Flow

| split | pairs | RT600 wrong | RT600 right | repairs | damage | net | damage rate on RT600-right |
|---|---:|---:|---:|---:|---:|---:|---:|
| `whole_fold` | 63380 | 22143 | 41237 | 3 | 4 | -1 | 0.0001 |
| `dominant_cell` | 50540 | 16193 | 34347 | 0 | 0 | 0 | 0.0000 |
| `mature_vs_never` | 50540 | 15804 | 34736 | 0 | 0 | 0 | 0.0000 |
| `mature_vs_prebreak` | 48677 | 15616 | 33061 | 1 | 0 | 1 | 0.0000 |

## Specialist-Disagreement Pair Flow

| subset | pairs | RT600 wrong | RT600 right | repairs | damage | net |
|---|---:|---:|---:|---:|---:|---:|
| `majority_correct` | 34155 | 992 | 33163 | 0 | 0 | 0 |
| `near_split` | 6734 | 3100 | 3634 | 0 | 0 | 0 |
| `at_least_one_correct` | 45045 | 10698 | 34347 | 0 | 0 | 0 |

## Action And Coefficient Audits

* Fold-0 candidate action counts: `{'0': 805340, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 994}`.
* Fold-0 candidate selected specialists: `{'RT-300': 0, 'RT-410': 0, 'RT-411': 0, 'RT-412': 0, 'RT-413': 0, 'RT-414': 0, 'RT-415': 994}`.
* Fold-0 target counts: `{'0': 188209, '1': 2354, '2': 2683, '3': 2539, '4': 948, '5': 791, '6': 927, '7': 265}`.
* Fold-0 coefficient family L2: `{'rt600_seed_state': 1.0374330631568005, 'dispersion_state': 2.2547279509639098, 'time_state': 0.5488929179829255, 'specialist_delta': 3.534793723381226}`.

## Controls

| control | marginal vs clone | candidate minus control | dominant net |
|---|---:|---:|---:|
| `RT-1231` | -0.000441 | +0.000130 | -178 |
| `RT-1232` | -0.000555 | +0.000243 | -552 |
| `RT-1233` | -0.000310 | -0.000002 | 0 |

## Implementation Audits

* Score coverage: all required frozen specialist and seed-clone OOF scores finite on dev rows and zero finite on lockbox rows.
* Pair-sample reproduction: `{'dominant_pairs': 50540, 'dominant_rt600_wrong': 16193, 'dominant_rt600_right': 34347}`.
* New OOF lockbox finite counts: `{'RT-1230': 0, 'RT-1231': 0, 'RT-1232': 0, 'RT-1233': 0}`.
* Peak RSS: `{'ru_maxrss': 2178318336, 'approx_mib': 2077.40625, 'platform': 'darwin'}`.
* Runtime: `811.2s`.

## Interpretation

SS-04 is KILL under the preregistered fold-0 screen. The bounded row-level specialist action router did not satisfy all mandatory marginal, dominant-pair, targeted-disagreement, and control-gap gates. Specialist-disagreement routing is closed under this Second Sweep.
