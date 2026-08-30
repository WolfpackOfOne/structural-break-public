# PILOT 6 -- SPECTRAL IMPULSIVENESS CONTRAST

Pre-registration: `research/reports/new_avenues_2026/PILOT06_PREREG.md` at `dddc2d9`.
Experiment IDs: `RT-1206` contrast candidate, `RT-1207` plain-energy control.

## RT-600 Anchor

* Dev mean TS-AUC: `0.625811`.
* Dev pooled TS-AUC: `0.625627`.
* Dev dominant-cell TS-AUC: `0.664277`.
* Fold-0 E0 RT600: `0.638276`.
* Fold-0 E1 RT600 + RT-401: `0.638586`.

## Candidate

Four Goertzel bands from the `m03_dyn` dyadic frequency bank. Segment length is 32; adaptive online window is the trailing half-prefix; features are NaN until at least 16 finite segment-energy endpoints exist.

`RT-1206` emits eight contrast columns: energy_z times negative spectral-kurtosis z and negative robust-negentropy z for each band.
`RT-1207` emits the four matched plain `energy_z` columns only.

## Causality Checks

* Harness contrast: `ok`.
* Harness energy: `ok`.
* Future mutation prefix check: `True`.
* Deterministic replay: `True`.
* First-valid semantics: `{'contrast_rows_before_46_all_nan': True, 'energy_rows_before_46_all_nan': True, 'contrast_row46_any_finite': True, 'energy_row46_any_finite': True}`.

## Binding Marginal Result

| arm | fold-0 standalone TS-AUC | E2 TS-AUC | marginal vs clone | gain vs RT600 |
|---|---:|---:|---:|---:|
| RT600 |  | 0.638276 |  |  |
| RT600 + RT-401 seed clone |  | 0.638586 |  |  |
| RT600 + RT-1206 | 0.627636 | 0.638233 | -0.000353 | -0.000043 |
| RT600 + RT-1207 | 0.631084 | 0.638775 | +0.000189 | +0.000499 |

Contrast minus plain-energy marginal: `-0.000542`.
Verdict: **KILL** (KILL) -- contrast marginal_vs_clone is below +0.0010.
Failed gates: `primary_marginal_vs_clone` (contrast marginal_vs_clone is below +0.0010); `plain_energy_control_not_worse` (plain-energy control matches or exceeds contrast)

## Diagnostic Pack

| candidate | whole-fold AUC | dominant-cell AUC | mature vs never | mature vs pre | rho vs RT600 |
|---|---:|---:|---:|---:|---:|
| `RT-1206` | 0.627636 | 0.667578 | 0.664795 | 0.675342 | +0.8809 |
| `RT-1207` | 0.631084 | 0.674571 | 0.673311 | 0.678088 | +0.8851 |

## Pair Flow

### RT-1206

| split | repairs | damage | net | sampled pairs |
|---|---:|---:|---:|---:|
| `whole_fold` | 1247 | 1541 | -294 | 19562 |
| `dominant_cell` | 892 | 1141 | -249 | 15598 |
| `cell_never_break_neg` | 849 | 1154 | -305 | 15582 |
| `cell_pre_break_neg` | 747 | 944 | -197 | 13378 |

### RT-1207

| split | repairs | damage | net | sampled pairs |
|---|---:|---:|---:|---:|
| `whole_fold` | 1280 | 1428 | -148 | 19562 |
| `dominant_cell` | 942 | 962 | -20 | 15598 |
| `cell_never_break_neg` | 898 | 941 | -43 | 15582 |
| `cell_pre_break_neg` | 778 | 911 | -133 | 13378 |

## Interpretation

The preregistered spectral impulsiveness contrast failed a Pilot 6 binding gate. This falsifies the F1/F6 product-contrast construction under the fixed dyadic Goertzel bands, SEG=32 envelope, robust historical-null calibration, and Mode-A fold-0 ABL screen; it does not falsify all spectral representations.

Feature build runtime: `435.0s`.
Total runtime: `1586.9s`.
