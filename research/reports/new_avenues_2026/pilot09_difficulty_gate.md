# PILOT 9(i) -- SCALAR HISTORICAL DIFFICULTY GATE

Pre-registration: `research/reports/new_avenues_2026/PILOT09_PREREG.md` at `3275ffd`.
Experiment IDs: `RT-1212` nested scalar candidate, `RT-1213` deranged scalar control.

## RT-600 Anchor

* Dev mean TS-AUC: `0.625811`.
* Dev pooled TS-AUC: `0.625627`.
* Dev dominant-cell TS-AUC: `0.664277`.
* Fold-0 E0 RT600: `0.638276`.
* Fold-0 E1 RT600 + RT-401: `0.638586`.

## Candidate

One series-constant scalar from the 23 history-only fingerprints, trained nested/fold-pure to predict RT-600 dominant-cell pair loss rate. The control deranges that scalar within each permanent fold.

## Fold-Purity Checks

* Fingerprint names match preregistration: `True`.
* Candidate fold-purity audit: `True`.
* Control fold-purity audit: `True`.
* Derangement audit: `{'0': {'n': 1617, 'no_fixed_points': True, 'same_multiset': True}, '1': {'n': 1609, 'no_fixed_points': True, 'same_multiset': True}, '2': {'n': 1601, 'no_fixed_points': True, 'same_multiset': True}, '3': {'n': 1592, 'no_fixed_points': True, 'same_multiset': True}, '4': {'n': 1581, 'no_fixed_points': True, 'same_multiset': True}, 'ok': True}`.
* Row scalar constant within checked series: `{'0': True, '1428': True, '2856': True, '4285': True, '5713': True, '7142': True, '8570': True, '9999': True}`.
* Lockbox scalar finite count: `0`.
* m07 parity note: Pilot 9(i) does not recompute m07_bayes::bo_p_lt25_z; it uses the existing cached base-bank features and one nested history-fingerprint scalar.

## Binding Marginal Result

| arm | fold-0 standalone TS-AUC | E2 TS-AUC | marginal vs clone | gain vs RT600 |
|---|---:|---:|---:|---:|
| RT600 |  | 0.638276 |  |  |
| RT600 + RT-401 seed clone |  | 0.638586 |  |  |
| RT600 + RT-1212 | 0.629133 | 0.638750 | +0.000164 | +0.000474 |
| RT600 + RT-1213 | 0.630245 | 0.638855 | +0.000269 | +0.000579 |

Candidate minus deranged-control marginal: `-0.000105`.
Fold-0 scalar target Spearman diagnostic: `{'fold': 0, 'n_series': 1271, 'spearman': 0.13738341604616944, 'p': 8.81602273739388e-07, 'provenance_ok': True}`.
Verdict: **KILL** (KILL) -- deranged scalar control matches or exceeds the real scalar.
Failed gates: `deranged_control_not_worse` (deranged scalar control matches or exceeds the real scalar); `primary_marginal_vs_clone` (candidate marginal_vs_clone is below +0.0010)

## Diagnostic Pack

| candidate | whole-fold AUC | dominant-cell AUC | mature vs never | mature vs pre | rho vs RT600 |
|---|---:|---:|---:|---:|---:|
| `RT-1212` | 0.629133 | 0.666635 | 0.663818 | 0.674497 | +0.8629 |
| `RT-1213` | 0.630245 | 0.666912 | 0.664874 | 0.672601 | +0.8712 |

## Pair Flow

### RT-1212

| split | repairs | damage | net | sampled pairs |
|---|---:|---:|---:|---:|
| `whole_fold` | 1365 | 1522 | -157 | 19562 |
| `dominant_cell` | 983 | 1054 | -71 | 15598 |
| `cell_never_break_neg` | 939 | 1115 | -176 | 15582 |
| `cell_pre_break_neg` | 890 | 912 | -22 | 13378 |

### RT-1213

| split | repairs | damage | net | sampled pairs |
|---|---:|---:|---:|---:|
| `whole_fold` | 1302 | 1608 | -306 | 19562 |
| `dominant_cell` | 902 | 1128 | -226 | 15598 |
| `cell_never_break_neg` | 862 | 1131 | -269 | 15582 |
| `cell_pre_break_neg` | 786 | 974 | -188 | 13378 |

## Interpretation

The preregistered scalar historical-difficulty gate failed a Pilot 9(i) binding gate. This falsifies the J1 nested one-scalar construction under the fixed 23 history-only fingerprints, RT-600 dominant-cell loss-rate target, within-fold derangement control, and Mode-A fold-0 ABL screen; it does not falsify all historical-DGP conditioning.

Total runtime: `1176.1s`.
