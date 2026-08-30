# Grok Mechanisms M2-M5: Cheapest-Falsification Screens

Date: `2026-08-30 13:40:57`  ·  git `966bdd2`

**No model was trained.** Each screen is the falsification the mechanism's own
author specified, run at the threshold the author declared.

M3 and M4 are row-level and fold-pure. M2 and M5 are **series-level batch
diagnostics at the final online point**, which is the protocol their specs ask
for and the only protocol under which an end-of-series statistic is admissible
(PROTOCOL.md: series-level, one row per series, series ROC AUC). They screen
whether a family is worth building; they are not row-level predictive claims.

## Summary

| mechanism | gate | observed | verdict |
|---|---|---:|---|
| M3_frozen_cohort_two_null_atlas | >= 0.55 | 0.501655 | **KILL** |
| M4_explicit_duration_absorbing_filter | kill if the dwell hazard is flat (\|slope\| < 0.01) and survival tracks the matched geometric within 0.05; also kill if the pooled non-geometry is explained by per-series frailty (flat within-series hazard AND a mixture of per-series geometrics reproducing pooled survival within 0.05) | -0.013811 | **PROCEED** |
| M2_per_series_delay_cloud_predictive_null | >= 0.55 | 0.582653 | **PROCEED** |
| M5_per_series_classifier_two_sample_test | >= 0.53 | 0.503378 | **KILL** |

## M3 — Frozen Cohort Two-Null Atlas

*Spec:* Cheapest falsification: AUC of 1[own z > 2 and population rank < 0.5] on the dominant-cell mature-vs-never cut. Kill the family if < 0.55.

*Protocol:* row-level, fold-pure: atlas built on folds 1-4, scored on fold 0

| statistic | value |
|---|---:|
| disagreement_indicator | 0.501655 |
| continuous_two_null_residual | 0.533826 |
| atlas_rank_alone | 0.596503 |
| own_z_alone | 0.558822 |
| indicator_deranged_atlas | 0.513314 |
| indicator_contaminated_atlas | 0.501556 |

- Scored rows: `388252`; atlas rows: `2403027`; atlas coverage `0.9995`
- Indicator fires on `0.0072` of scored rows
- Real-minus-deranged atlas gap: `-0.011659`

Do the two reference measures actually disagree?

- Within-t rank correlation, own-history vs cohort atlas: `0.240387`
- Mean absolute rank disagreement: `0.282484`
- P(own-hot) `0.0800`, P(population-typical | own-hot) `0.0901`

They do disagree substantially, so the family does not die of the two nulls
being the same object. It dies because the *disagreement* carries nothing on
the target cut while the population *level* does: the atlas rank alone scores
`0.596503` against own-z alone at
`0.558822`. A within-t population level is
exactly what SmoothTimeCDFCal already applies to scores, which answers the
mechanism's own 'information even on failure' question: yes, a population
reference is already implicit in SCDF calibration.

**Verdict.** KILL the M3 family: the two-null disagreement indicator does not separate the target cut at the declared threshold.

## M4 — Explicit-Duration Absorbing Filter

*Spec:* Cheapest falsification: empirical survival of excursion length on never-break history windows. If geometrically tailed the HSMM collapses to m07 and we kill without training.

*Protocol:* history-only, never-break series only; no online data, no labels used. Excursions are runs of smoothed AR(6) residual energy above the historical q90 of its own absolute deviation from median.

- Never-break series sampled: `3600`, excursions: `42565`
- Mean run length `19.168`, matched geometric p `0.05217`
- Hazard slope in log-dwell: `-0.013811` (0 = memoryless)
- Max survival gap vs matched geometric: `0.304702`

| dwell k | at risk | hazard |
|---:|---:|---:|
| 1 | 42565 | 0.2280 |
| 2 | 32862 | 0.1616 |
| 3 | 27552 | 0.1252 |
| 4 | 24102 | 0.1063 |
| 5 | 21540 | 0.0904 |
| 6 | 19592 | 0.0781 |
| 7 | 18061 | 0.0733 |
| 8 | 16737 | 0.0630 |
| 9 | 15683 | 0.0557 |
| 10 | 14810 | 0.0533 |
| 11 | 14020 | 0.0511 |
| 12 | 13304 | 0.0446 |

Frailty control. A pooled decreasing hazard is also what a mixture of
per-series geometric dwells produces even when each series is individually
memoryless, so the pooled figure alone cannot carry the verdict.

| window | excursions | pooled slope | vs geometric | within-series slope | series | vs geometric mixture |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 94650 | -0.02703 | 0.1875 | -0.05987 | 1356 | 0.1656 |
| 64 | 62337 | -0.01867 | 0.2494 | -0.06400 | 416 | 0.2039 |
| 128 | 42565 | -0.01381 | 0.3047 | -0.06312 | 67 | 0.2229 |

The within-series hazard slope is steeper than the pooled one at every
window, and a mixture of per-series geometrics leaves a survival gap far
above the 0.05 tolerance, so the non-geometry is duration memory inside an
excursion rather than heterogeneity across series. The within-series
estimate is best powered at window 32 (211 series); windows 64 and 128 agree
on the sign and magnitude with far fewer series.

**Verdict.** PROCEED: at window 128 the dwell hazard is not flat (slope -0.0138 in log-dwell, survival gap 0.3047 vs the matched geometric), and the frailty control does not explain it (within-series hazard slope -0.0631, mixture survival gap 0.2229), so an explicit-duration prior is not a reparametrisation of m07's geometric hazard.

## M2 — Per-Series Delay-Cloud Predictive Null

*Spec:* Cheapest falsification: k-NN occupancy surprise at t=max on ~200 mature-break and ~200 never-break series. Kill if the occupancy disagreement with m00_core does not rank the rows above 0.55 AUC.

*Protocol:* SERIES-LEVEL batch diagnostic at the final online point, one row per series, series ROC AUC. Not a row-level predictive claim.

| statistic | value |
|---|---:|
| occupancy_disagreement_auc | 0.582653 |
| occupancy_disagreement_ci95_lo | 0.558890 |
| occupancy_disagreement_ci95_hi | 0.605693 |
| m00_scale_control_auc | 0.556306 |
| occupancy_residualised_on_m00_auc | 0.553682 |
| rho_occupancy_vs_m00 | 0.684349 |

- Bootstrap 95% CI on the occupancy AUC: `[0.5589, 0.6057]`; the gate is
  applied to the lower bound, not the point estimate.

Read this verdict narrowly. The screen clears the threshold its author set,
but the margin over the plain scale control is modest and the correlation
with it is `0.6843`: residualised on
that control the statistic falls to
`0.553682`. PROCEED here means
'not falsified, worth one fold-0 build', not 'carries independent signal'.

**Verdict.** PROCEED to a streaming delay-cloud likelihood

## M5 — Per-Series Classifier Two-Sample Test

*Spec:* Cheapest falsification: batch linear C2ST at the last online point on ~100 series. Kill if series-level AUC < 0.53, or if it does not beat Wasserstein on the same windows by +0.02.

*Protocol:* SERIES-LEVEL batch diagnostic at the final online point, one row per series, series ROC AUC. Not a row-level predictive claim.

| statistic | value |
|---|---:|
| c2st_auc | 0.503378 |
| c2st_ci95_lo | 0.500693 |
| c2st_ci95_hi | 0.541090 |
| wasserstein_control_auc | 0.591948 |
| margin_over_wasserstein | -0.088570 |
| rho_c2st_vs_wasserstein | 0.060355 |

- Bootstrap 95% CI on the C2ST AUC: `[0.5007, 0.5411]`

The kill is decisive rather than marginal: the learned test sits at chance
while the named distance it would have to displace scores
`0.591948` on the identical windows.
This is the linear C2ST the spec asked for; a nonlinear variant would have to
close a gap of `0.0886`, not
win a close contest.

**Verdict.** KILL the M5 family: a learned two-sample test does not beat the named distance it would have to displace, so it is m02 by another name.
