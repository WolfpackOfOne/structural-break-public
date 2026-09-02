# Statistical Audit

## Questions Requiring Adjudication

The three highest-priority unresolved quantitative questions are:

1. **What does W7-D3R's large non-causal Arm-C advantage measure?** Its size and fold-sign claim are verified below, but the fraction due to final-row/series-length information versus future break-path information remains unknown.
2. **Is CatBoost a small, genuine complementary learner?** CSA-04R resolves the advertised `k=5`/`+0.00593` headline, but robustness of the parsimonious `k=2` effect to partition choice and a score-level `n_online` purity screen remains open.
3. **Is the seven-specialist ensemble saturated for existing causal information?** Many candidates are redundant or merely diverse, but `m12_rdep` and especially CatBoost rule out an absolute “nothing else can help” claim.

Future-information capture and saturation are related but distinct: Arm C is non-causal and outside the deployable information set; saturation concerns useful complementarity within the legal causal set.

## Data / Predictions Used

### Prediction-vector audit

| vectors | provenance | fold status | row count | target definition |
|---|---|---|---:|---|
| `RT-300`, `RT-401`, `RT-410`–`RT-415`, `RT-990`, `RT-995` | `structural-break-research-current/research/oof/*.npy` | Assembled five-fold dev OOF per tracked Wave-7 reports; `RT-995` is nested-pure, unlike contaminated pilots | 5,036,517 positions each; 4,032,524 common finite dev rows; 1,003,993 common NaNs | Row-level break state for TS-AUC: post-break positive; never-break and pre-break negative, compared within time |
| `RT-1254`, `RT-1255`, `RT-1260`–`RT-1263` and counterpart incumbents `RT-300`, `RT-411`–`RT-415` | `structural-break-deep-ensemble-frontier-local-2026/research/oof/*.npy` | Reported five-fold OOF with frozen SCDF downstream analysis; no lockbox/test predictions | 5,036,517 positions each; 4,032,524 pairwise-finite dev rows | Same target; dev has 8,000 series, 3,975 break series, and 1,033,242 positive rows (`DATA_FORENSICS_REPORT.md`) |

The common NaN mask makes reproduced raw correlations comparable. Raw Pearson/Spearman correlations pool time and are not substitutes for within-time rank correlation or incremental TS-AUC.

Aggregate artifacts used without prediction vectors:

- `research/reports/wave7_d3r.json`: dominant-cell aggregate and five-fold metrics for Arms A/B/C. `RT-991.npy` was not found, so row-level metric reconstruction and series bootstrap were impossible.
- `CSA04_FINAL.md`, `CSA04R_REANALYSIS.md`, `research/RESULTS.csv`, and `DATA_FORENSICS_REPORT.md`: frozen-OOF rescoring and series-bootstrap outputs. The bootstrap was not independently rerun.
- `wave7_t2_ensemble_integration.json`: fold-0 E0/E1/E2 metrics.
- Wave-5/6 reports for `m12_rdep`, neural controls, seed-clone controls, and alternate partitions.

No training, lockbox/test labels, or leaderboard probing was used.

## Metric Reconstruction

### W7-D3R `+0.071`

**Data:** `wave7_d3r.json`; dominant cell `t>=200, age>=100`, 50.5047% of dev pair weight. **N:** five outer folds; row/pair N absent from JSON. **Metric:** pair-weighted dominant-cell TS-AUC. **Procedure:** subtract Arm B from Arm C within fold and for pooled metrics.

| quantity | estimate |
|---|---:|
| pooled Arm B | 0.647491470 |
| pooled Arm C | 0.718588416 |
| pooled C-B | **+0.071096947** |
| fold C-B | +0.052767, +0.080459, +0.065884, +0.076037, +0.081801 |
| folds positive | **5/5** |
| unweighted fold-delta mean | +0.071390 |
| fold-delta sample SD | 0.012138 |
| pair-weight translation reported in JSON | +0.035907 pooled-equivalent |

**Ruling:** `[REPRODUCED]` arithmetic/fold signs from the stored metric artifact; `[ARTIFACT-VERIFIED]`, not raw-OOF-reproduced, underlying AUCs. The `+0.071` and `5/5` claims are correct. The fold mean differs slightly from pooled C-B because pooling is pair-weighted. No confidence interval is available. Sign consistency does not identify whether the gain is future-path evidence, endpoint/length information, or both.

### CatBoost: CSA-04 versus CSA-04R

The apparent contradiction mixes estimands and selection rules.

- CSA-04 `+0.005934643` is **E2-E1**, where E1 is a composition-dependent seed-clone replacement control. As `k` increases E1 deteriorates; this is not gain over fixed RT-600.
- For the same selected `k=5` vector, fixed-baseline **E2-E0 is `+0.002378478`**, with folds `+0.001171, +0.002433, +0.002120, +0.004772, +0.001397` (5/5; fold SD 0.001434).
- CSA-04R was later preregistered, fixed E2-E0, and exactly reproduced 24 committed values. Its maximum was `k=3`, `+0.002364556`; gain over `k=2` was `+0.000338234`, paired series-bootstrap 95% percentile CI `[-0.000678, +0.001318]`. Parsimony therefore selected `k*=2`.
- `k=2` (`RT-1257`, represented again by reanalysis row `RT-1265`) has E2-E0 **`+0.002026322`**, folds `+0.002154, +0.000291, +0.001208, +0.003238, +0.003241` (5/5; fold SD 0.001288). Reported paired series-bootstrap CI: `[+0.000431, +0.003541]`, B=2,000.

**Ruling:** CSA-04R supersedes the inference that five CatBoost slots deliver `+0.00593` production-relative alpha. `NOT_DISTINGUISHABLE` means larger compositions are not distinguishable from the two-slot incumbent; it does **not** mean k=2's `+0.00203` is indistinguishable from zero. The durable estimate is a small k=2 fixed-baseline gain. Its canonical-partition CI excludes zero, but alternate-partition evidence and a CatBoost `n_online` screen are absent. Extensive prior search prevents calling it externally confirmed.

## Uncertainty Analysis

| claim | uncertainty available | limitation |
|---|---|---|
| D3R C-B `+0.07110` | Five fold deltas; SD 0.01214; no CI | No series bootstrap or raw Arm-C vector |
| CatBoost k=2 E2-E0 | Reported series-bootstrap CI `[0.000431, 0.003541]`, B=2,000 | Not independently rerun; conditional on canonical partition/selection history |
| CatBoost k=3 minus k=2 | Reported CI `[-0.000678, 0.001318]` | No evidence extra slots improve k=2 |
| CatBoost k=5 E2-E0 | Reported CI `[-0.001163, 0.005116]` under CSA-04R ordering | Includes zero |
| T2 ensemble marginal | Fold-0 `+0.0002365`; no CI | Cannot establish nonzero ensemble alpha |
| `m12_rdep` over clone | Canonical +0.00141; alt1 -0.00025; alt2 +0.00190 | Partition sign change |

CatBoost's bootstrap addresses series sampling within folds, not partition-draw or model-selection uncertainty. The repository's reported partition-draw SD near 0.005 exceeds k=2's point estimate; this is a caution, not a CatBoost CI.

## Fold Consistency

- **D3R:** C-B positive in 5/5, minimum +0.05277—strong sign consistency for a non-causal cell advantage.
- **CatBoost k=2:** positive in 5/5, but fold 1 is +0.00029 and folds 3/4 contribute 64% of summed fold deltas. Not one-fold-only, but uneven.
- **CatBoost k=5:** positive in 5/5; fold 3 largest at +0.00477. This does not validate the selected `+0.00593` estimand.
- **`m12_rdep`:** standalone gain was 4/5 on canonical, alt1, and alt2; ensemble margin over clone was 4/5, 2/5, and 4/5. New information is more stable standalone than as ensemble complementarity.

## Conditional Analysis

For `RT-1264`, the existing forensic artifact uses 8,000 series/4,032,524 rows and reports:

- whole-dev pooled TS-AUC delta +0.002267;
- dominant-cell +0.002978;
- mature-vs-prebreak +0.005154;
- relative-time 10–25% **-0.003632**.

The effect is heterogeneous. These are post-hoc slices without multiplicity correction or CIs; they localize but do not confirm a mechanism.

D3R is conditional on the dominant cell. Calling `+0.071` overall OOF gain is incorrect. The +0.03591 translation is arithmetic, not a directly scored deployable model.

## Complementarity Analysis

On 4,032,524 finite dev rows, reproduced T2 (`RT-995`) Pearson correlations were 0.669–0.908 with seven specialists, 0.904 with `RT-990`, and 0.906 with seed clone `RT-401`. This confirms overlap, subject to pooled-time confounding.

For six CatBoost/incumbent pairs, reproduced pooled Pearson correlations were 0.657–0.942 and Spearman 0.433–0.943. CatBoost materially changes some rankings, especially `RT-1254` versus `RT-413`, while several replacements remain very similar. These are not the reports' within-time `rho`.

Evidence for practical saturation/diminishing returns:

- an eighth seed clone adds +0.00003 (another -0.00051);
- T2 falls from +0.00943 standalone to +0.0002365 versus clone in the ensemble;
- low-correlation neural models (`rho` about 0.21 within time) add only +0.00010 to +0.00019 over clone—diversity is not useful diversity;
- `m12_rdep` shrinks from stable standalone gain to partition-unstable ensemble value and -0.0027 on full architecture rebuild;
- many Wave-8/New-Avenues candidates are approximately zero/negative against clone controls.

Counterevidence:

- `m12_rdep` canonical +0.00141 over clone shows the bank was not literally devoid of new causal information, though instability/negative rebuild block promotion;
- CatBoost k=2 gives +0.00203 over fixed E0, 5/5, with a canonical-partition bootstrap CI excluding zero.

**Ruling:** `[SUPPORTED]` extensive redundancy and practical diminishing returns. `[REJECTED]` absolute saturation (“no legal same-bank learner can improve RT-600”). CatBoost is current counterevidence; `m12_rdep` is weaker counterevidence. The open issue is robustness of CatBoost's small gain, not whether old candidates were correlated.

## Leakage / Purity Checks

- D3R Arm C is intentionally non-causal: valid oracle diagnostic, not deployable OOF. Missing `RT-991.npy` blocks residualization against `n_online`, endpoint, or remaining length.
- Only nested-pure `RT-995`, not contaminated teacher pilots, was used in correlations.
- CatBoost rows report five-fold OOF, inherited causal features, fold-pure `SCDF_NSEEN`, and no lockbox/test use. This audit did not code-trace every fold fit.
- No CatBoost score-level `n_online` screen is reported. This is a purity gap, not evidence of leakage.
- Identical missingness (4,032,524 finite of 5,036,517) prevents differential masks from creating reproduced correlations.
- Selection bias is material: CSA-04 chose k after viewing a curve and used moving E1. CSA-04R's fixed endpoint/preregistered parsimony is preferred.

## Claims Supported

1. W7-D3R C-B is +0.0710969 dominant-cell TS-AUC, positive 5/5 folds (aggregate-artifact level).
2. T2 is mostly redundant to RT-600; its only ensemble estimate is +0.0002365 on fold 0.
3. The ensemble exhibits strong diminishing returns and overlapping predictions.
4. CatBoost has a small canonical-partition signal: k=2 E2-E0 +0.002026, 5/5, reported CI excluding zero.
5. RT-600 is the production anchor; later lineage documents supersede earlier research champions.

## Claims Rejected

1. **“RT-1264 adds +0.00593 over RT-600.”** +0.00593 is against composition-dependent E1; fixed-baseline gain is +0.00238, and larger k is not distinguishable from k=2.
2. **“CSA-04R found k=2's +0.002 indistinguishable from zero.”** `NOT_DISTINGUISHABLE` concerns improvement over RT-1257/k=2.
3. **“Five positive D3R folds prove causally harvestable future information.”** They prove a stable non-causal oracle advantage, not its interpretation.
4. **“Low correlation establishes useful diversity.”** Rejected by neural controls.
5. **“RT-131 remains champion.”** Later tracked documents classify its live cross-sectional rank as illegal/oracle.
6. **“Three champions remain scientifically unresolved.”** RT-160 is parallel corroboration, RT-250 canonicalizes former RT-150, and RT-600 is the later production anchor. This is documentation lineage, not scientific ambiguity.
7. **“0.63828 versus 0.6268 is an internal-to-LB collapse.”** 0.63828 is fold-0 E0; five-fold RT-600 mean is 0.625811. External 0.6268 is report-only but not in tension.
8. **“HEAD weekend-harness 0.598843 competes with wave results.”** It is single-split series AUC, not five-fold row-level TS-AUC.

## Claims Still Unresolved

1. Fraction of D3R Arm C's +0.07110 attributable to endpoint/`n_online` versus future break-path information; Arm-C predictions are unavailable.
2. Whether CatBoost k=2 remains positive under alternate partitions and a score-level `n_online` screen.
3. Whether legal existing predictions offer reproducible useful diversity beyond CatBoost's small effect. Evidence supports near-saturation, not a mathematical ceiling.
4. Exact external submission producing 0.6268 and independent verification of that score—provenance uncertainty, not top-three modeling science.

## Statistical Bottom Line

RT-131, the three-champion narrative, and both HEAD metric mismatches are resolved documentation/metric-discipline issues rather than unresolved science. The skeptic is correct that newer CatBoost evidence enters the top three. CSA-04R resolves its headline contradiction: retire `+0.00593 MAJOR` as production-relative and retain provisional **k=2, +0.00203 E2-E0**.

W7-D3R's `+0.07110`, 5/5 claim is numerically sound, but its causal meaning remains unresolved. Ensemble saturation is strongly supported as practical diminishing returns, yet contradicted as absolute by CatBoost and, more weakly, `m12_rdep`. The top three are therefore **D3R oracle-content attribution; CatBoost k=2 robustness/purity; and the attainable complementarity ceiling of the existing legal information set.** No new training run is proposed.
