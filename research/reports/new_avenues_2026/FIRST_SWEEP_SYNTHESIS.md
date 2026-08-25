# First Sweep Synthesis

Status: `FIRST_SWEEP_EXHAUSTED`

Scope: post-hoc descriptive synthesis of the completed New Avenues first sweep. This report uses only existing pilot reports, existing OOF arrays, existing fold-0 diagnostics, and historical wave reports. It is `POSTHOC_DESCRIPTIVE_NOT_PROMOTION_EVIDENCE`.

Forbidden-action check: no new model was trained, no new candidate OOF score was generated, no TS-AUC was calculated for a new candidate, no RT id was allocated, `research/RESULTS.csv` was not edited, and no lockbox/test/production/submission artifact was touched.

## Bottom Line

The first New Avenues sweep did not find a promotable mechanism. All scored arms `RT-1200` through `RT-1218` failed their pre-registered continuation gates. The strongest arm, `RT-1216` weighted CTM, was a real clue but not a continuation result: fold-0 `marginal_vs_clone = +0.0009366`, below the `+0.0010` gate, with dominant-cell pair flow still negative in the pilot report (`repairs=992`, `damage=1027`, `net=-35`).

The common failure was not lack of locally repairable mistakes. In a deterministic post-hoc fold-0 sample of dominant-cell same-time pairs, at least one first-sweep candidate repaired `14,868 / 16,193 = 91.8%` of sampled RT600 wrong pairs. The same candidate set also damaged `28,505 / 34,347 = 83.0%` of sampled RT600 correct pairs. The bottleneck is therefore repair-vs-damage arbitration, not another unfiltered anomaly statistic.

## First-Sweep Matrix

The normalized matrix is written to:

- `research/reports/new_avenues_2026/first_sweep_matrix.csv`
- `research/reports/new_avenues_2026/first_sweep_matrix.json`

Condensed candidate readout:

| Arm | Mechanism | Whole AUC | rho vs RT600 | Marginal vs clone | Dominant net | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `RT-1200` | relay score-state | 0.619611 | 0.7746 | -0.000207 | -318 | killed |
| `RT-1201` | IM2/dwell | 0.583578 | 0.3817 | +0.000301 | -1151 | killed |
| `RT-1202` | trajectory geometry | 0.499324 | 0.0039 | -0.002587 | -2974 | killed |
| `RT-1204` | scale survival | 0.629444 | 0.8860 | -0.000236 | -131 | killed |
| `RT-1206` | spectral impulse | 0.627636 | 0.8809 | -0.000353 | -249 | killed |
| `RT-1208` | ordinal irreversibility | 0.629141 | 0.8780 | -0.000036 | -75 | killed |
| `RT-1210` | joint rarity | 0.626121 | 0.8779 | -0.000522 | -237 | killed |
| `RT-1212` | scalar difficulty | 0.629133 | 0.8629 | +0.000164 | -71 | killed |
| `RT-1214` | Kalman/NIS | 0.630817 | 0.8836 | +0.000135 | -98 | killed |
| `RT-1215` | Hankel-DMD | 0.631331 | 0.8859 | +0.000226 | -11 | killed |
| `RT-1216` | weighted CTM | 0.634279 | 0.8653 | +0.000937 | -35 | killed close miss |
| `RT-1218` | direct e-value aggregation | 0.525678 | 0.1341 | -0.002214 | -2167 | killed |

Matched-control outcomes were also damaging. Plain spectral energy beat spectral impulse by `+0.000542` marginal-vs-clone. Dwell-only beat joint rarity by `+0.000285`. Deranged scalar difficulty beat the real scalar by `+0.000105`. Scale survival and ordinal irreversibility beat their controls, but by `+0.000471` and `+0.000472`, just below their `+0.0005` control-separation gates.

## Meta Relationships

Generated files:

- `research/reports/new_avenues_2026/first_sweep_meta_relationships.csv`
- `research/reports/new_avenues_2026/first_sweep_meta_relationships.json`

These are small-n descriptive relationships only. No p-values are reported.

For candidate arms only (`n=12`), the strongest non-tautological relationships with `marginal_vs_clone` were:

| Predictor | Pearson | Spearman | Interpretation |
| --- | ---: | ---: | --- |
| whole-pair net | 0.856 | 0.678 | pair-flow direction was more informative than mechanism label |
| dominant-cell pair net | 0.879 | 0.678 | dominant-cell repair/damage balance was the main gate proxy |
| whole-fold standalone AUC | 0.885 | 0.643 | raw signal helped, but was not sufficient |
| never-break pair net | 0.885 | 0.629 | false-positive side mattered materially |
| within-time rho vs RT600 | 0.805 | 0.245 | redundancy was nonlinear: low rho usually meant no signal; high rho meant saturation |

Standalone AUC was not meaningless: `RT-1216`, `RT-1215`, `RT-1214`, `RT-1204`, `RT-1208`, and controls all had plausible standalone AUCs. But the metric rewards within-time pair ordering under the incumbent ensemble, and every candidate still had non-positive dominant-cell pair flow in its pilot report.

## Repair Coverage

Generated files:

- `research/reports/new_avenues_2026/first_sweep_repair_summary.json`
- `research/reports/new_avenues_2026/first_sweep_repair_overlap.csv`

Sampling convention: fold-0 dev only, same-time positive/negative pairs, strict `score_positive > score_negative`, deterministic seed `20260825`, `64` sampled pairs per time point. This is descriptive and not a score.

| Sample | Pairs | RT600 wrong | Repaired by any candidate | Damaged by any candidate |
| --- | ---: | ---: | ---: | ---: |
| whole fold | 63,380 | 22,143 | 20,219 / 91.3% | 34,914 / 84.7% of RT600-right |
| dominant cell | 50,540 | 16,193 | 14,868 / 91.8% | 28,505 / 83.0% of RT600-right |
| never-break negative cell | 50,540 | 15,804 | 14,557 / 92.1% | 28,804 / 82.9% of RT600-right |
| pre-break negative cell | 48,677 | 15,616 | 13,858 / 88.7% | 27,716 / 83.8% of RT600-right |

Mean repair overlap across all first-sweep scored arms was moderate, not zero: dominant-cell repair Jaccard mean `0.232`, median `0.263`; whole-fold repair Jaccard mean `0.236`, median `0.268`. The failed arms are not identical, but their repairs are not cleanly separable from their damages.

## Specialist Decomposition

Generated file:

- `research/reports/new_avenues_2026/first_sweep_specialist_decomposition.json`

Among sampled dominant-cell RT600 wrong pairs:

| Specialist state | Pairs | Share | Any candidate repairs | `RT-1216` repairs |
| --- | ---: | ---: | ---: | ---: |
| all seven specialists wrong | 5,457 | 33.7% | 83.4% | 4.4% |
| one to three specialists correct | 9,710 | 60.0% | 95.7% | 24.3% |
| four to seven specialists correct | 1,026 | 6.3% | 99.8% | 50.1% |

Histogram of correct specialists on RT600 wrong dominant-cell pairs: `0:5457`, `1:4113`, `2:3375`, `3:2222`, `4:904`, `5:118`, `6:4`, `7:0`.

This matters for the next program. There is raw disagreement signal: at least one specialist is correct on `66.3%` of sampled RT600 mistakes. But simple specialist routing is not enough: majority-correct blend failures are only `6.3%`, Pilot 1 static fingerprint selectors were far below RT600, and every individual specialist had negative direct dominant-cell pair flow versus RT600 in the Pilot 1 diagnostic.

## Why The Mechanisms Failed

The sweep covered real scientific variety: score-state transforms, excursion dwell, trajectory geometry, scale survival, spectral impulse, ordinal irreversibility, joint rarity, scalar difficulty, Kalman/NIS, Hankel-DMD, weighted CTM, and direct e-value aggregation. They converged to three practical failure modes.

First, mechanisms that were truly orthogonal usually did not have enough same-time break signal. `RT-1202` trajectory geometry had rho `0.0039` and whole AUC `0.4993`; `RT-1218` direct e-value aggregation had rho `0.1341` and whole AUC `0.5257`. Low correlation was novelty, not alpha.

Second, mechanisms with plausible signal mostly collapsed into the incumbent residual-scale/maturity direction. The feature-block candidates from `RT-1204` through `RT-1217` had rho values around `0.86-0.89`. They could rank some local errors correctly but did not create broad positive pair flow after the seven-stream ensemble already averaged related evidence.

Third, several claimed mechanisms did not beat simpler matched controls. This is the clearest evidence against mechanistic interpretation: plain energy beat spectral impulse, dwell-only beat joint rarity, deranged scalar beat real scalar difficulty, and unweighted CTM came close enough that weighted CTM is a clue rather than a result.

## RT-1216 Interpretation

`RT-1216` should be treated as a signal about the problem, not as a tunable near-miss. It had the best standalone result in the sweep (`0.634279`), the best marginal-vs-clone result (`+0.0009366`), and a real matched-control gap versus unweighted CTM (`+0.0007241` marginal-vs-clone; `+0.002135` mature-vs-never AUC). It still failed the continuation gate and had negative dominant-cell pair flow in the pilot report.

The structural reading is that predictable benign-tail weighting helps in some negative-side states, but the CTM feature family is already too aligned with incumbent residual/e-process evidence. A second sweep may use the lesson, not the mechanism: prioritize false-positive/null calibration and repair-damage arbitration. Do not tune `RT-1216` weights, thresholds, windows, or CTM variants as a second-sweep slot.

## Bottleneck Ranking

H1. Repair-vs-damage arbitration is the dominant bottleneck. Evidence: any candidate repairs `91.8%` of sampled dominant RT600 mistakes, but any candidate damages `83.0%` of sampled RT600-correct dominant pairs.

H2. Objective/integration mismatch is nearly as important. Evidence: every candidate had non-positive official dominant-cell pair net even when standalone AUC was plausible.

H3. Incumbent representation saturation is high. Evidence: most viable feature-block arms have rho `0.86-0.89` against RT600 and repeat known residual-scale/maturity/e-process directions.

H4. Static error-manifold routing is weak. Evidence: Pilot 1 selectors were far below RT600, and individual specialists had negative direct pair flow versus RT600. Specialist disagreement exists, but the route is not simple fingerprint selection.

H5. Negative-side/null calibration remains plausible but narrow. Evidence: dominant-cell loss is heavily never-break-negative, D3 fingerprints weakly predict loss, and `RT-1216`/unweighted CTM show tail-state signal. The scalar difficulty and deranged-control results make broad DGP gating suspect.

H6. Missing causal prefix information is not the primary explanation. Evidence: low-rho novel arms mostly lacked standalone signal, and future-aware/neural/raw-capacity sweeps did not translate to marginal alpha.

H7. Generic learner capacity is low priority. Evidence: stronger same-column learners, raw TCN/MLP paths, weighting/subset/stacking, and future-aware transfer have repeatedly failed or become redundant.

## Closed Lanes

Closed for the second sweep unless a separate prereg explicitly reopens them with a materially different target:

- CTM tuning or `RT-1216` variants.
- New scalar anomaly statistics without a repair-damage arbitration layer.
- Static DGP/fingerprint routers.
- Generic ensemble weighting, subset selection, or stacking over RT600 specialists.
- Raw neural learners on the same legal channels.
- Future-aware distillation variants that repeat SST/ORR/PCFB/CFEP/TGMC.
- Trajectory geometry, direct e-value aggregation, spectral impulse, joint rarity, ordinal irreversibility, and scalar difficulty as standalone mechanisms.

## Second-Sweep Implication

The next sweep should not ask "what new statistic detects breaks?" It should ask "can we distinguish when a non-incumbent signal repairs RT600 from when it damages RT600?" The first run should therefore be a repair-vs-damage arbiter, not another feature block and not a `RT-1216` tune.

The preregistered second-sweep plan is in `research/reports/new_avenues_2026/SECOND_SWEEP_PREREG.md`.
