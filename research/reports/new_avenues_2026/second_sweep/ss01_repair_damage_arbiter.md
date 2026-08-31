# SS-01 -- Repair-Damage Arbiter

Execution preregistration: `research/reports/new_avenues_2026/second_sweep/SS01_EXECUTION_PREREG.md` at `32b5427`.
Experiment IDs: `RT-1219` candidate, `RT-1220` global-average control, `RT-1221` shuffled-target control, `RT-1222` single-best-arm control.

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
| RT600 + RT-1219 | 0.638276 | 0.638276 | -0.000310 | -0.000000 |
| RT600 + RT-1220 | 0.635674 | 0.638900 | +0.000314 | +0.000624 |
| RT600 + RT-1221 | 0.638276 | 0.638276 | -0.000310 | -0.000000 |
| RT600 + RT-1222 | 0.583550 | 0.638904 | +0.000317 | +0.000627 |

Verdict: **KILL**.
Gate failures: `marginal_vs_clone` observed `-0.0003101020692587442`; `dominant_net_pair_lift` observed `0`; `contributing_sensor_families` observed `0`; `RT-1220_control_gap` observed `-0.000623849725659098`; `RT-1221_control_gap` observed `0.0`; `RT-1222_control_gap` observed `-0.000627337764813829`

## Pair Flow

### RT-1219

| split | pairs | RT600 wrong | RT600 right | repairs | damage | net | damage rate on RT600-right |
|---|---:|---:|---:|---:|---:|---:|---:|
| `whole_fold` | 63380 | 22143 | 41237 | 0 | 0 | 0 | 0.0000 |
| `dominant_cell` | 50540 | 16193 | 34347 | 0 | 0 | 0 | 0.0000 |
| `mature_vs_never` | 50540 | 15804 | 34736 | 0 | 0 | 0 | 0.0000 |
| `mature_vs_prebreak` | 48677 | 15616 | 33061 | 1 | 0 | 1 | 0.0000 |

## Repair Reservoir

* Original dominant repair reservoir: `14868 / 16193`.
* Original dominant any-damage baseline: `28505 / 34347`.
* SS-01 dominant repairs: `0`.
* SS-01 dominant damage: `0`.
* Repair reservoir retained: `0.0000`.
* Original candidate-union damage rejected: `1.0000`.

## Action Use

* Non-RT600 action rows: `0`.
* Contributing families for gate: `[]`.

| family | fold-0 rows | share of non-RT600 actions | counts for gate |
|---|---:|---:|---|
| `RT-1200` relay_score_state | 0 | 0.0000 | False |
| `RT-1201` im2_dwell | 0 | 0.0000 | False |
| `RT-1202` trajectory_geometry | 0 | 0.0000 | False |
| `RT-1204` scale_survival | 0 | 0.0000 | False |
| `RT-1206` spectral_impulse | 0 | 0.0000 | False |
| `RT-1208` ordinal_irreversibility | 0 | 0.0000 | False |
| `RT-1210` joint_rarity | 0 | 0.0000 | False |
| `RT-1212` scalar_difficulty | 0 | 0.0000 | False |
| `RT-1214` kalman_nis | 0 | 0.0000 | False |
| `RT-1215` hankel_dmd | 0 | 0.0000 | False |
| `RT-1216` weighted_ctm | 0 | 0.0000 | False |
| `RT-1218` direct_evalue | 0 | 0.0000 | False |

## Controls

| control | marginal vs clone | candidate minus control | dominant net |
|---|---:|---:|---:|
| `RT-1220` | +0.000314 | -0.000624 | -161 |
| `RT-1221` | -0.000310 | +0.000000 | 0 |
| `RT-1222` | +0.000317 | -0.000627 | -3311 |

## Family Holdouts

| removed family | marginal vs clone | candidate minus holdout | dominant net |
|---|---:|---:|---:|
| `RT-1200` relay_score_state | -0.000310 | +0.000000 | 0 |
| `RT-1201` im2_dwell | -0.000310 | +0.000000 | 0 |
| `RT-1202` trajectory_geometry | -0.000310 | +0.000000 | 0 |
| `RT-1204` scale_survival | -0.000310 | +0.000000 | 0 |
| `RT-1206` spectral_impulse | -0.000310 | +0.000000 | 0 |
| `RT-1208` ordinal_irreversibility | -0.000310 | +0.000000 | 0 |
| `RT-1210` joint_rarity | -0.000310 | +0.000000 | 0 |
| `RT-1212` scalar_difficulty | -0.000310 | +0.000000 | 0 |
| `RT-1214` kalman_nis | -0.000310 | +0.000000 | 0 |
| `RT-1215` hankel_dmd | -0.000310 | +0.000000 | 0 |
| `RT-1216` weighted_ctm | -0.000310 | +0.000000 | 0 |
| `RT-1218` direct_evalue | -0.000310 | +0.000000 | 0 |

## Implementation Audits

* Pair-sample reproduction: `{'dominant_pairs': 50540, 'dominant_rt600_wrong': 16193, 'dominant_rt600_right': 34347, 'matches_first_sweep_counts': True}`.
* Sensor coverage: all frozen sensors finite on dev rows and zero finite on lockbox rows.
* New OOF lockbox finite counts: `{'RT-1219': 0, 'RT-1220': 0, 'RT-1221': 0, 'RT-1222': 0}`.
* Peak RSS: `{'ru_maxrss': 2547433472, 'approx_mib': 2429.421875, 'platform': 'darwin'}`.
* Runtime: `5126.4s`.

## Interpretation

SS-01 is KILL under the preregistered screen. Fold-pure repair-vs-damage arbitration over frozen first-sweep sensors did not satisfy all mandatory same-t pair-flow and marginal-vs-clone gates. Under the frozen state representation, first-sweep killed arms should not be treated as useful production arbitration sensors.
