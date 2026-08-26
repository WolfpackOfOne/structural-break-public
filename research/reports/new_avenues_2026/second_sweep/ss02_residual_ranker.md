# SS-02 -- Dominant-Cell Residual Ranker

Execution preregistration: `research/reports/new_avenues_2026/second_sweep/SS02_EXECUTION_PREREG.md` at `3b39954`.
Experiment IDs: `RT-1223` candidate, `RT-1224` shuffled residual-offset/weight control.

## RT-600 Sentinel

* Dev mean TS-AUC: `0.625811`.
* Dev pooled TS-AUC: `0.625627`.
* Dev dominant-cell TS-AUC: `0.664277`.
* Fold-0 E0 RT600: `0.638276`.
* Fold-0 E1 RT600 + RT-401: `0.638586`.

## Binding Result

| arm | standalone fold-0 TS-AUC | E2 TS-AUC | marginal vs clone | gain vs RT600 |
|---|---:|---:|---:|---:|
| RT600 |  | 0.638276 |  |  |
| RT600 + RT-401 seed clone |  | 0.638586 |  |  |
| RT600 + RT-1223 | 0.637896 | 0.638297 | -0.000290 | +0.000020 |
| RT600 + RT-1224 | 0.638583 | 0.638384 | -0.000203 | +0.000107 |

Verdict: **KILL**.
Gate failures: `marginal_vs_clone` observed `-0.000289694109078531`; `dominant_net_pair_lift` observed `-108`; `mature_vs_never_or_prebreak_pair_flow` observed `{'mature_vs_never': -77, 'mature_vs_prebreak': -102}`; `RT-1224_control_gap` observed `-8.697748055974674e-05`

## Candidate Pair Flow

| split | pairs | RT600 wrong | RT600 right | repairs | damage | net | damage rate on RT600-right |
|---|---:|---:|---:|---:|---:|---:|---:|
| `whole_fold` | 63380 | 22143 | 41237 | 494 | 587 | -93 | 0.0142 |
| `dominant_cell` | 50540 | 16193 | 34347 | 330 | 438 | -108 | 0.0128 |
| `mature_vs_never` | 50540 | 15804 | 34736 | 376 | 453 | -77 | 0.0130 |
| `mature_vs_prebreak` | 48677 | 15616 | 33061 | 328 | 430 | -102 | 0.0130 |

## Shuffled Control Pair Flow

| split | pairs | RT600 wrong | RT600 right | repairs | damage | net | damage rate on RT600-right |
|---|---:|---:|---:|---:|---:|---:|---:|
| `whole_fold` | 63380 | 22143 | 41237 | 595 | 613 | -18 | 0.0149 |
| `dominant_cell` | 50540 | 16193 | 34347 | 404 | 509 | -105 | 0.0148 |
| `mature_vs_never` | 50540 | 15804 | 34736 | 402 | 490 | -88 | 0.0141 |
| `mature_vs_prebreak` | 48677 | 15616 | 33061 | 439 | 466 | -27 | 0.0141 |

## Training Summary

| arm | fold | train rows | pairs | RT600 wrong | RT600 right | dominant pairs | val clip frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| `RT-1223` | 0 | 800000 | 113605 | 41564 | 72041 | 72146 | 0.7579 |
| `RT-1223` | 1 | 800000 | 113624 | 40981 | 72643 | 72111 | 0.7456 |
| `RT-1223` | 2 | 800000 | 113813 | 41532 | 72281 | 72252 | 0.7562 |
| `RT-1223` | 3 | 800000 | 113471 | 41166 | 72305 | 72019 | 0.7474 |
| `RT-1223` | 4 | 800000 | 113729 | 41177 | 72552 | 72222 | 0.7415 |
| `RT-1224` | 0 | 800000 | 113605 | 41564 | 72041 | 72146 | 0.5006 |
| `RT-1224` | 1 | 800000 | 113624 | 40981 | 72643 | 72111 | 0.5196 |
| `RT-1224` | 2 | 800000 | 113813 | 41532 | 72281 | 72252 | 0.4758 |
| `RT-1224` | 3 | 800000 | 113471 | 41166 | 72305 | 72019 | 0.5132 |
| `RT-1224` | 4 | 800000 | 113729 | 41177 | 72552 | 72222 | 0.4923 |

## Top Feature Importance

| rank | feature | gain |
|---:|---|---:|
| 1 | `rt600_cal` | 3071445.8 |
| 2 | `rt600_logit` | 2878167.9 |
| 3 | `rt600_margin_abs` | 918724.3 |
| 4 | `specialists_above_half_frac` | 574134.1 |
| 5 | `m07_bayes::bo_lo_change_pk` | 287023.3 |
| 6 | `m07_bayes::ev_pow_pk` | 257441.1 |
| 7 | `m01_seq::cz100_pkr` | 227289.5 |
| 8 | `m01_seq::phd_pkr` | 205883.7 |
| 9 | `m01_seq::glz_pk` | 194257.1 |
| 10 | `m01_seq::cz50_pk` | 180058.3 |
| 11 | `m01_seq::phu_pkr` | 179699.6 |
| 12 | `m01_seq::gle_pkr` | 178760.2 |
| 13 | `m01_seq::xc_max_pk` | 176396.7 |
| 14 | `m01_seq::ce50_pk` | 162071.0 |
| 15 | `m03_dyn::az_sg_exp_l1_z` | 158985.8 |
| 16 | `m00_core::w8_sur_tail_x` | 151008.2 |
| 17 | `m01_seq::srz_pkr` | 149174.2 |
| 18 | `m01_seq::cz25_pkr` | 143972.2 |
| 19 | `m01_seq::ce25_pkr` | 134565.6 |
| 20 | `m07_bayes::ab_lpo_pk_z` | 131017.5 |

## Implementation Audits

* Feature-bank columns: `500`.
* Score-state covariates: `25`.
* Pair-sample reproduction: `{'dominant_pairs': 50540, 'dominant_rt600_wrong': 16193, 'dominant_rt600_right': 34347}`.
* Specialist and seed-clone coverage: all frozen OOF scores finite on dev rows and zero finite on lockbox rows.
* New OOF lockbox finite counts: `{'RT-1223': 0, 'RT-1224': 0}`.
* Peak RSS: `{'ru_maxrss': 6840713216, 'approx_mib': 6523.8125, 'platform': 'darwin'}`.
* Runtime: `1242.3s`.

## Interpretation

SS-02 is KILL under the preregistered screen. A bounded correction trained against RT600 residual same-t pair errors over the incumbent 500-feature bank did not satisfy all mandatory marginal, pair-flow, and shuffled-control gates.
