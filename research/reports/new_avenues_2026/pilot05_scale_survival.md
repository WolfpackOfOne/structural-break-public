# PILOT 5 -- SCALE-SURVIVAL COARSE-GRAINING

Pre-registration: `research/reports/new_avenues_2026/PILOT05_PREREG.md` at `094c50f`.
Experiment IDs: `RT-1204` summary candidate, `RT-1205` individual-scale control.

## RT-600 Anchor

* Dev mean TS-AUC: `0.625811`.
* Dev pooled TS-AUC: `0.625627`.
* Dev dominant-cell TS-AUC: `0.664277`.
* Fold-0 E0 RT600: `0.638276`.
* Fold-0 E1 RT600 + RT-401: `0.638586`.

## Candidate

AR(2) residual-square stream, causally coarse-grained at `{1,2,4,8,16,32}`. Each scale is compared against its own historical same-scale block-mean null. Surprise is two-sided empirical tail evidence, with fixed thresholds q05=`1.301029995664` and q01=`2.0`.

`RT-1204` emits five summary columns: survival counts at q05/q01, largest surviving scale code at q05/q01, and log-scale surprise slope.
`RT-1205` emits the six individual per-scale surprises only.

## Causality Checks

* Harness summary: `ok`.
* Harness individual: `ok`.
* Future mutation prefix check: `True`.
* Deterministic replay: `True`.
* First-valid semantics: `{'b1_row0_finite': True, 'b2_row0_nan': True, 'b32_row30_nan': True, 'b32_row31_finite': True, 'slope_row0_nan': True, 'slope_row1_finite': True}`.

## Binding Marginal Result

| arm | fold-0 standalone TS-AUC | E2 TS-AUC | marginal vs clone | gain vs RT600 |
|---|---:|---:|---:|---:|
| RT600 |  | 0.638276 |  |  |
| RT600 + RT-401 seed clone |  | 0.638586 |  |  |
| RT600 + RT-1204 | 0.629444 | 0.638350 | -0.000236 | +0.000074 |
| RT600 + RT-1205 | 0.625238 | 0.637880 | -0.000707 | -0.000396 |

Summary minus individual-scale marginal: `+0.000471`.
Verdict: **KILL** (KILL) -- summary marginal_vs_clone is below +0.0010.
Failed gates: `primary_marginal_vs_clone` (summary marginal_vs_clone is below +0.0010); `summary_control_distinguishability` (summary-control gap is below the preregistered +0.0005 distinguishability floor).

## Diagnostic Pack

| candidate | whole-fold AUC | dominant-cell AUC | mature vs never | mature vs pre | rho vs RT600 |
|---|---:|---:|---:|---:|---:|
| `RT-1204` | 0.629444 | 0.670218 | 0.666092 | 0.681732 | +0.8860 |
| `RT-1205` | 0.625238 | 0.664388 | 0.662921 | 0.668480 | +0.8870 |

## Pair Flow

### RT-1204

| split | repairs | damage | net | sampled pairs |
|---|---:|---:|---:|---:|
| `whole_fold` | 1229 | 1487 | -258 | 19562 |
| `dominant_cell` | 874 | 1005 | -131 | 15598 |
| `cell_never_break_neg` | 824 | 1022 | -198 | 15582 |
| `cell_pre_break_neg` | 846 | 751 | 95 | 13378 |

### RT-1205

| split | repairs | damage | net | sampled pairs |
|---|---:|---:|---:|---:|
| `whole_fold` | 1242 | 1475 | -233 | 19562 |
| `dominant_cell` | 871 | 1046 | -175 | 15598 |
| `cell_never_break_neg` | 818 | 1020 | -202 | 15582 |
| `cell_pre_break_neg` | 687 | 920 | -233 | 13378 |

## Interpretation

Explicit scale-survival count / largest-scale / log-scale decay summaries on the frozen dyadic AR(2) residual-square coarse-grained stream failed the primary Pilot 5 marginal gate and missed the preregistered summary-vs-individual distinguishability floor. This falsifies this E1/E4 functional under the preregistered fold-0 screen; it does not falsify all multiscale representations or all residual-scale detectors.

Feature build runtime: `845.0s`.
Total runtime: `1998.6s`.
