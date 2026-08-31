# CSA-04R Reanalysis -- Final

Date: 2026-08-28
Branch: `research/deep-ensemble-frontier-local-2026`
Analysis SHA: `98a1a9a`
Preregistration: `research/reports/deep_ensemble_frontier_2026/CSA04R_REANALYSIS_PREREG.md`

## Scope

This is a re-analysis of existing CSA-04 OOF vectors. It trained no model, tuned no weight, wrote no OOF vector, and did not use lockbox or test data.

## Harness Checks

CSA-04 exact reproduction: `True` across `24` committed values at tolerance `1e-09`.
RT-1257 marginal regression: observed `+0.002407204670070`, expected `+0.002407204670070`, diff `+0.000e+00`, passed `True`.
P4 k=2 E2-E0 regression: observed `+0.002026321728670`, expected `+0.002026321728670`, diff `+7.286e-17`, passed `True`.

## Fixed Ordering

| rank | slot | single-slot E2-E0 | single-slot marginal_vs_clone |
|---:|---|---:|---:|
| 1 | `CAT-413` (`RT-1254`) | +0.001244347 | +0.001087151 |
| 2 | `CAT-300` (`RT-1255`) | +0.001126876 | +0.001029459 |
| 3 | `CAT-412` (`RT-1261`) | +0.001029593 | +0.001630380 |
| 4 | `CAT-415` (`RT-1263`) | +0.000907324 | +0.001001378 |
| 5 | `CAT-414` (`RT-1262`) | +0.000552119 | +0.001588999 |
| 6 | `CAT-411` (`RT-1260`) | +0.000362939 | +0.001567468 |

## Greedy Curve

| k | composition | E0 | E1 | E2 | E2-E0 | fold deltas E2-E0 | folds + vs E0 | delta vs RT-1257 | folds + vs RT-1257 | whole net | dominant net | mature-never net | mature-prebreak net |
|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `CAT-413` | 0.625811342 | 0.625968538 | 0.627055689 | +0.001244347 | `+0.000348;-0.000140;+0.001079;+0.002921;+0.002014` | 4/5 | -0.000781975 | 0/5 | 9 | 93 | 80 | 71 |
| 2 | `CAT-413, CAT-300` | 0.625811342 | 0.625430459 | 0.627837664 | +0.002026322 | `+0.002154;+0.000291;+0.001208;+0.003238;+0.003241` | 5/5 | +0.000000000 | 0/5 | 192 | 158 | 73 | 146 |
| 3 | `CAT-413, CAT-300, CAT-412` | 0.625811342 | 0.624790485 | 0.628175898 | +0.002364556 | `+0.001538;+0.002298;+0.001751;+0.004297;+0.001940` | 5/5 | +0.000338234 | 3/5 | 245 | 127 | 112 | 151 |
| 4 | `CAT-413, CAT-300, CAT-412, CAT-415` | 0.625811342 | 0.624362745 | 0.628002886 | +0.002191544 | `+0.001555;+0.002544;+0.002127;+0.003636;+0.001096` | 5/5 | +0.000165222 | 3/5 | 254 | 125 | 83 | 206 |
| 5 | `CAT-413, CAT-300, CAT-412, CAT-415, CAT-414` | 0.625811342 | 0.623111491 | 0.627801636 | +0.001990295 | `+0.001512;+0.002802;+0.002145;+0.004055;-0.000563` | 4/5 | -0.000036027 | 3/5 | 294 | 156 | 77 | 139 |
| 6 | `CAT-413, CAT-300, CAT-412, CAT-415, CAT-414, CAT-411` | 0.625811342 | 0.621778591 | 0.627675484 | +0.001864143 | `+0.000756;+0.002490;+0.002114;+0.003729;+0.000232` | 5/5 | -0.000162179 | 3/5 | 228 | 100 | 41 | 152 |

## Bootstrap

Bootstrap method: paired series bootstrap with exact weighted TS-AUC recomputation under multinomial series counts. B = `2000`, seed = `20260828`. Series were resampled within fold.

| k | observed E2-E0 | bootstrap SE | 95% percentile CI |
|---:|---:|---:|---:|
| 1 | +0.001244347 | 0.000487628 | [+0.000288014, +0.002206501] |
| 2 | +0.002026322 | 0.000801945 | [+0.000430574, +0.003541292] |
| 3 | +0.002364556 | 0.001101832 | [+0.000216935, +0.004462796] |
| 4 | +0.002191544 | 0.001413819 | [-0.000574202, +0.004906035] |
| 5 | +0.001990295 | 0.001597549 | [-0.001163178, +0.005116132] |
| 6 | +0.001864143 | 0.001676207 | [-0.001517702, +0.005106774] |

k* vs RT-1257 bootstrap: observed `+0.000000000`, SE `0.000000000`, 95% CI `[+0.000000000, +0.000000000]`.

| adjacent contrast around max | observed | bootstrap SE | 95% percentile CI |
|---|---:|---:|---:|
| `k3_minus_k2` | +0.000338234 | 0.000511841 | [-0.000678422, +0.001317609] |
| `k3_minus_k4` | +0.000173012 | 0.000466674 | [-0.000776013, +0.001112353] |

## Selection And Verdict

Maximum M occurs at `k=3` with `E2-E0=+0.002364556`.
Selection contrast `k3_minus_k2_RT1257` bootstrap SE is `0.000511841`; with floor `0.0011000`, `delta_noise=0.001100000`.
`K_tied = {2, 3, 4, 5, 6}`; preregistered parsimony selects `k*=2`.
Selected composition: `CAT-413, CAT-300`.
Verdict: `NOT_DISTINGUISHABLE`.

Plain reading: no CSA-04R multi-slot composition beats the two-slot RT-1257 by a distinguishable margin. CSA-04's durable result is the broad single-slot finding: six of seven specialist slots responded to CatBoost replacement at the preregistered k=1 gate.

## Prediction Adjudication

| prediction | passed | observed |
|---|---:|---|
| `P4` | `True` | 0.002026321728670455 |
| `P5` | `True` | k*=2; verdict=NOT_DISTINGUISHABLE |
| `P6` | `True` | selected=CAT-413,CAT-300; CSA04_k5=CAT-412,CAT-414,CAT-411,CAT-413,CAT-300 |

## Descriptive 63-Subset Appendix

This appendix is non-selecting by preregistration. It cannot select k* or support a promotion claim.
Best descriptive subset: `CAT-413, CAT-300, CAT-412, CAT-411` with `E2-E0=+0.002401432`; gap vs greedy k* is `+0.000375110`.

| subset | k | E2-E0 |
|---|---:|---:|
| `CAT-413` | 1 | +0.001244347 |
| `CAT-300` | 1 | +0.001126876 |
| `CAT-412` | 1 | +0.001029593 |
| `CAT-415` | 1 | +0.000907324 |
| `CAT-414` | 1 | +0.000552119 |
| `CAT-411` | 1 | +0.000362939 |
| `CAT-413, CAT-300` | 2 | +0.002026322 |
| `CAT-413, CAT-412` | 2 | +0.001922106 |
| `CAT-413, CAT-415` | 2 | +0.001791362 |
| `CAT-413, CAT-414` | 2 | +0.001617247 |
| `CAT-413, CAT-411` | 2 | +0.001503745 |
| `CAT-300, CAT-412` | 2 | +0.001852317 |
| `CAT-300, CAT-415` | 2 | +0.001702036 |
| `CAT-300, CAT-414` | 2 | +0.001516295 |
| `CAT-300, CAT-411` | 2 | +0.001383278 |
| `CAT-412, CAT-415` | 2 | +0.001585804 |
| `CAT-412, CAT-414` | 2 | +0.001439569 |
| `CAT-412, CAT-411` | 2 | +0.001281977 |
| `CAT-415, CAT-414` | 2 | +0.001263667 |
| `CAT-415, CAT-411` | 2 | +0.001145259 |
| `CAT-414, CAT-411` | 2 | +0.000881366 |
| `CAT-413, CAT-300, CAT-412` | 3 | +0.002364556 |
| `CAT-413, CAT-300, CAT-415` | 3 | +0.002201550 |
| `CAT-413, CAT-300, CAT-414` | 3 | +0.002205180 |
| `CAT-413, CAT-300, CAT-411` | 3 | +0.002160264 |
| `CAT-413, CAT-412, CAT-415` | 3 | +0.002100325 |
| `CAT-413, CAT-412, CAT-414` | 3 | +0.002131089 |
| `CAT-413, CAT-412, CAT-411` | 3 | +0.002073455 |
| `CAT-413, CAT-415, CAT-414` | 3 | +0.001948577 |
| `CAT-413, CAT-415, CAT-411` | 3 | +0.001919301 |
| `CAT-413, CAT-414, CAT-411` | 3 | +0.001837442 |
| `CAT-300, CAT-412, CAT-415` | 3 | +0.002076472 |
| `CAT-300, CAT-412, CAT-414` | 3 | +0.002084611 |
| `CAT-300, CAT-412, CAT-411` | 3 | +0.002002437 |
| `CAT-300, CAT-415, CAT-414` | 3 | +0.001882982 |
| `CAT-300, CAT-415, CAT-411` | 3 | +0.001825068 |
| `CAT-300, CAT-414, CAT-411` | 3 | +0.001743327 |
| `CAT-412, CAT-415, CAT-414` | 3 | +0.001801699 |
| `CAT-412, CAT-415, CAT-411` | 3 | +0.001723700 |
| `CAT-412, CAT-414, CAT-411` | 3 | +0.001671713 |
| `CAT-415, CAT-414, CAT-411` | 3 | +0.001473064 |
| `CAT-413, CAT-300, CAT-412, CAT-415` | 4 | +0.002191544 |
| `CAT-413, CAT-300, CAT-412, CAT-414` | 4 | +0.002386652 |
| `CAT-413, CAT-300, CAT-412, CAT-411` | 4 | +0.002401432 |
| `CAT-413, CAT-300, CAT-415, CAT-414` | 4 | +0.002162529 |
| `CAT-413, CAT-300, CAT-415, CAT-411` | 4 | +0.002205380 |
| `CAT-413, CAT-300, CAT-414, CAT-411` | 4 | +0.002307052 |
| `CAT-413, CAT-412, CAT-415, CAT-414` | 4 | +0.002105514 |
| `CAT-413, CAT-412, CAT-415, CAT-411` | 4 | +0.002117702 |
| `CAT-413, CAT-412, CAT-414, CAT-411` | 4 | +0.002239698 |
| `CAT-413, CAT-415, CAT-414, CAT-411` | 4 | +0.002031666 |
| `CAT-300, CAT-412, CAT-415, CAT-414` | 4 | +0.002094520 |
| `CAT-300, CAT-412, CAT-415, CAT-411` | 4 | +0.002096669 |
| `CAT-300, CAT-412, CAT-414, CAT-411` | 4 | +0.002211510 |
| `CAT-300, CAT-415, CAT-414, CAT-411` | 4 | +0.001976268 |
| `CAT-412, CAT-415, CAT-414, CAT-411` | 4 | +0.001906645 |
| `CAT-413, CAT-300, CAT-412, CAT-415, CAT-414` | 5 | +0.001990295 |
| `CAT-413, CAT-300, CAT-412, CAT-415, CAT-411` | 5 | +0.002079040 |
| `CAT-413, CAT-300, CAT-412, CAT-414, CAT-411` | 5 | +0.002378478 |
| `CAT-413, CAT-300, CAT-415, CAT-414, CAT-411` | 5 | +0.002129357 |
| `CAT-413, CAT-412, CAT-415, CAT-414, CAT-411` | 5 | +0.002084519 |
| `CAT-300, CAT-412, CAT-415, CAT-414, CAT-411` | 5 | +0.002079297 |
| `CAT-413, CAT-300, CAT-412, CAT-415, CAT-414, CAT-411` | 6 | +0.001864143 |
