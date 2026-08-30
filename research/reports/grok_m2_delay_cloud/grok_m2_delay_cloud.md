# Grok M2 — Per-Series Delay-Cloud Predictive Null

Date: `2026-08-30 16:19:41`  ·  git `bb35a35`

The screen cleared this narrowly. This builds the streaming occupancy block
and puts it through the ensemble gate.

## Construction

- Delay dimension `3`, fixed, never chosen from series length
- `512` historical delay vectors per series, `k=5`
- Two clouds: raw standardised series and AR(6) residual stream
- `mu_H` fitted on history only and frozen before the first online step
- Dev coverage `0.8781`

## Univariate signal

| channel | dominant-cell AUC |
|---|---:|
| occ_surprise_raw | 0.517652 |
| occ_surprise_resid | 0.526353 |
| occ_ewma16_raw | 0.534624 |
| occ_ewma64_raw | 0.547076 |
| occ_conformal_p_raw | 0.520323 |
| occ_conformal_p_resid | 0.528551 |
| occ_dwell_off_cloud | 0.513911 |
| occ_max_dwell_off_cloud | 0.542766 |
| occ_frac_off_last64 | 0.543904 |
| occ_eprocess_log | 0.558370 |
| occ_minus_ar_resid_z | 0.504228 |
| occ_raw_minus_resid | 0.511799 |

## Redundancy gates

| gate | observed | requirement | passed |
|---|---:|---|---|
| rho vs RT-600, within-t cell | 0.103608 | <= 0.85 (kill), <= 0.75 to promote | True |

## Ensemble gate

Two fold-pure specialists on the same `m00_core` base, differing only by the
occupancy block. The increment is the contrast that identifies the mechanism;
the absolute marginal is reported because it is what the spec names, but it is
dominated by the base and is not the verdict.

| arm | marginal vs clone | positive folds | standalone cell |
|---|---:|---:|---:|
| control_m00 | -0.000672 | 1/5 | 0.587903 |
| occupancy | -0.000190 | 3/5 | 0.596834 |
| control_full_bank | +0.000621 | 4/5 | 0.640150 |
| occupancy_full_bank | +0.000251 | 4/5 | 0.637752 |

| contrast | value |
|---|---:|
| occupancy_minus_control | +0.000482 |
| increment_positive_folds | +5.000000 |
| dominant_net_pair_flow | -46.000000 |
| full_bank_occupancy_minus_control | -0.000370 |
| full_bank_increment_positive_folds | +2.000000 |

Per-fold increment over the `{m00_core}` control: +0.000049, +0.000420, +0.001076, +0.000593, +0.000270 (5/5 positive).

Per-fold increment over the **full 500-column bank**: +0.000785, -0.001045, -0.000713, +0.000606, -0.001485 (2/5 positive).

### The redundancy gate measured the wrong object

Occupancy correlates `0.1036` with RT-600 within-`t` on the dominant cell, which
clears the spec's `0.85` kill line with enormous room and reads as "this is a
genuinely new channel". Against the seven-module bank it is worth
`-0.000370` at
`2/5` folds.

Both are true, and the tension between them is the lesson. Low correlation with
the ensemble's blended **score** is not evidence of independence from the
ensemble's **inputs**. A channel can be nearly uncorrelated with RT-600's output
while lying inside the span of the 500 columns that produce it -- which is
exactly what the two increments show: `+0.000482` at 5/5 over `m00_core` alone,
and negative over the full bank. The six modules beyond `m00_core` already carry
what occupancy adds.

Any future redundancy gate on this ledger should be stated against the feature
bank, not against the blended score. RT-1215 was killed at `rho = 0.8859`
measured the same way; that kill happened to be right, but the statistic would
not have caught this mechanism.

Pair flow of the occupancy arm against E0:

| split | repairs | damage | net | damage rate |
|---|---:|---:|---:|---:|
| dominant_cell | 901 | 947 | -46 | 0.0188 |
| mature_vs_never | 965 | 1022 | -57 | 0.0203 |
| mature_vs_prebreak | 761 | 828 | -67 | 0.0188 |

## Verdict

KILL, on the mechanism's own criteria: marginal_vs_clone -0.000190 < +0.0010; dominant net -46 < 0. Grok's kill rule trips if ANY of marginal < +0.0010, rho > 0.85, or dominant net < 0; 2 of the three trip. What is real and worth carrying forward: occupancy is genuinely non-redundant with the incumbent (rho 0.1036 against an 0.85 line, the least redundant channel screened on this branch), and its increment over the specified m00_core base is +0.000482 at 5/5 positive folds -- consistent in sign on every fold, unlike M4's 2/5. On the full 500-column bank, which is the base that actually matters, the increment is -0.000370 at 2/5. The absolute marginal is negative because the m00_core base is itself negative (-0.000672); occupancy improves that base without rescuing it.

