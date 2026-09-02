# Scientific Synthesis

## Executive Conclusion

The three most important unresolved quantitative questions are:

1. **What does W7-D3R Arm C's +0.07110 dominant-cell TS-AUC advantage measure?** Its size is stable, but the fraction due to endpoint/`n_online` information versus future break-path evidence is unknown.
2. **Is the CatBoost two-slot hybrid a clean, robust complement to RT-600?** The defensible estimate is k=2 E2-E0 **+0.002026**, 5/5 positive folds, with reported series-bootstrap 95% CI `[+0.000431,+0.003541]`; score-level `n_online` and alternate-partition evidence are absent.
3. **How close is RT-600 to the attainable ceiling of the existing legal causal information set?** Strong diminishing returns are verified, but provisional CatBoost evidence and weaker `m12_rdep` evidence reject an absolute ceiling.

The best-fitting explanation is that the final-row oracle imports substantial non-shadowable endpoint/future information, while legal candidates usually recover signal already represented by RT-600 or signal too sparse to matter under pair-weighted TS-AUC. This is favored, not proved. No new training is warranted. One existing-artifact-only CatBoost audit is warranted.

## Verified Facts

- **[GIT-VERIFIED]** Canonical HEAD `33fa041` is a disconnected 16-commit harness lineage; later research lives on shared-object-database branches, tags, and worktrees.
- **[ARTIFACT-VERIFIED]** Canonical `RESULTS.csv` is a wave-1 fossil: 78 rows, 77 with `git_sha=nogit`; it still labels illegal-oracle RT-131 `NEW CHAMPION`.
- **[CODE/ARTIFACT-VERIFIED]** The HEAD harness's 0.598843 is single-split series AUC, not five-fold row-level TS-AUC.
- **[REPRODUCED from aggregate artifact]** D3R Arm C-B is +0.071096947 in the dominant cell, positive in 5/5 folds; the cell carries 50.5047% of dev pair weight. This proves a stable non-causal advantage, not its mechanism.
- **[REPRODUCED]** T2 substantially overlaps existing specialists; its standalone gain collapses to a fold-0 ensemble marginal of +0.0002365 versus clone.
- **[ARTIFACT-VERIFIED]** Five Wave-8 full-population ensemble marginals are approximately -0.00031 in fold-0 pilots. This establishes failure at their gates, not universal impossibility.
- **[ARTIFACT-VERIFIED / DERIVED]** CSA-04's +0.005934643 is E2-E1 against a deteriorating clone control. Fixed-baseline k=5 E2-E0 is +0.002378478.
- **[REPORT-CLAIM, statistically audited]** CSA-04R's parsimonious k=2 E2-E0 is +0.002026322, 5/5 positive folds, reported CI `[+0.000431,+0.003541]`. Its bootstrap was not independently rerun.
- **[REPORT-CLAIM]** RT-600 is the production anchor: five-fold development mean 0.625811 and external 0.6268. The external score was not independently verified.

## Consensus

All analyses agree that RT-131 status, RT-150's ID collision, “three champions,” RT-125R naming, 0.63828 versus 0.6268, and the weekend-harness score are documentation, provenance, or metric-discipline issues—not top unresolved science. They agree that D3R is numerically stable but causally uninterpreted, useful diversity requires incremental ensemble evidence, and no new training is justified.

## Important Disagreement

The primary researcher ranked production artifact screens third and the ensemble ceiling second. The skeptic correctly used post-2026-08-24 history to introduce CatBoost and split future-information uncertainty into oracle content and eligible/full-population collapse. The statistical audit adjudicated that CatBoost enters the top three; eligible/full collapse remains a mechanism nested within questions 1 and 3.

The primary report listed an `RT-991.npy` source, while the skeptic and statistician could not locate it. It is treated as missing unless provenance-verified, blocking direct Arm-C residualization.

## Primary Researcher: What Survived

- Correctly identified the future-information/distillation dissociation as central.
- Reproduced OOF dimensions, masks, and T2/specialist correlation structure.
- Correctly separated standalone alpha from ensemble complementarity and emphasized pair-weight dilution.
- Correctly resolved stale champion/ID contradictions as documentation.
- Correctly warned against a sixth distillation attempt.

Its feature-gain/`az` audit is not top-three science: gain is weakly related to metric value and no performance claim attaches to the unexplained gain mass. The six-stream `n_online` screen remains useful assurance work.

## Skeptic: What Survived

- Correctly incorporated executed-and-killed New Avenues work and later CatBoost/CSA-04R evidence.
- Correctly sharpened D3R into endpoint/remaining-length versus future-path attribution.
- Correctly identified eligibility and pair-weight coverage as explanations of Wave-8 collapse.
- Correctly exposed CSA-04's moving-control estimand and k-search bias.
- Correctly retained k=2 as provisional: `NOT_DISTINGUISHABLE` concerns extra slots over k=2, not k=2 versus zero.

## Statistical Adjudication

- D3R's +0.07110 and 5/5 signs are resolved; causal attribution is not. No CI or raw Arm-C reconstruction is available.
- Retire “RT-1264 adds +0.00593 over RT-600.” E0 is the fixed deployment counterfactual. CSA-04R supports provisional k=2 alpha of +0.00203; larger k is not distinguishable from k=2.
- CatBoost k=2 is positive in all folds but uneven: folds 3–4 supply 64% of summed deltas. Its bootstrap does not address partition-draw or prior-selection uncertainty.
- Practical diminishing returns are supported. Absolute saturation is rejected.
- Low correlation is insufficient: diverse neural candidates added only about +0.00010 to +0.00019 over clone.

## Literature Implications

No literature-review artifact exists; no external-literature inference is used.

## Rejected Explanations

- RT-131 remains champion; three champion lineages imply scientific ambiguity.
- 0.63828 to 0.6268 is leaderboard collapse.
- CSA-04 proves +0.00593 production-relative alpha.
- Five positive D3R folds prove causally harvestable information.
- RT-600 is absolutely saturated.
- Missing duration state alone explains Wave 8.
- Another future-aware training mechanism is warranted.

## Remaining Unknowns

1. Arm C's decomposition into endpoint/`n_online`, identity, and future-path evidence.
2. Whether CatBoost k=2 is score-level `n_online`-clean and positive beyond the canonical partition.
3. The attainable out-of-sample complementarity ceiling among existing legal predictors.
4. Whether Wave-8 eligible-slice gains survive remaining-length adjustment or reduce to coverage arithmetic.
5. Independent provenance for external 0.6268 and the exact successful submission.
6. The shared 19.93% OOF NaN mask's meaning and the missing Arm-C vector.

## Ranked Hypotheses

### 1. Endpoint information plus legal-signal redundancy explains the apparent future-information gap

- **mechanism:** Arm C imports final-row/remaining-horizon information unavailable at `t`; its shadowable component is already represented by RT-600.
- **evidence:** D3R +0.07110; little reported early-horizon known-boundary-oracle headroom; T2 +0.00943 standalone becomes +0.00024 in ensemble; Wave-8 full-population failures.
- **confidence:** Moderate.
- **falsifier:** length/endpoint residualization leaves nearly the full Arm-C gap and novel corrected pairs.
- **expected value of testing:** Very high, but blocked by missing `RT-991.npy`.

### 2. CatBoost supplies small genuine complementary learner bias

- **mechanism:** ordered boosting extracts different rankings from the same causal bank.
- **evidence:** k=2 E2-E0 +0.002026, 5/5 positive folds, reported CI excluding zero.
- **confidence:** Moderate-low pending purity and partition evidence.
- **falsifier:** material `n_online` predictability, failed fixed sensitivity checks, or nonpositive alternate-partition effects.
- **expected value of testing:** High; existing vectors allow purity and canonical-partition sensitivity checks.

### 3. RT-600 is near, but not exactly at, the practical legal-bank ceiling

- **mechanism:** seven specialists span most useful causal directions; remaining candidates duplicate errors or affect low-weight slices.
- **evidence:** clone +0.00003, T2 collapse, diverse neural failures, Wave-8/New-Avenues negatives, unstable `m12_rdep`; CatBoost is counterevidence.
- **confidence:** Moderate-high for diminishing returns; low for an exact ceiling.
- **falsifier:** a fixed fold-pure existing-vector combination gives stable material gain beyond CatBoost k=2 without post-hoc selection.
- **expected value of testing:** Moderate; unrestricted library search would amplify selection bias.

### 4. Pair-weight coverage explains Wave-8 eligible/full-population collapse

- **mechanism:** a valid effect occurs on a small class-conditioned slice and is diluted or reversed over all scored pairs.
- **evidence:** SST +0.00810 eligible versus -0.00060 full; ORR mechanism evidence with near-zero metric effect.
- **confidence:** Moderate.
- **falsifier:** eligible gain disappears after remaining-length adjustment, favoring confounding.
- **expected value of testing:** Moderate-high using existing SST vectors.

## Recommended Next Experiment

Run the existing-artifact-only CatBoost k=2 purity and sensitivity audit in `06_PREREGISTRATION.md`. It fixes CAT-413 (`RT-1254`) plus CAT-300 (`RT-1255`), prohibits search/tuning, reproduces E2-E0 and its bootstrap, tests score-level `n_online`, and reports fixed fold/slice sensitivities. It cannot supply alternate-partition evidence; success means only clean provisional canonical-partition alpha.

Direct Arm-C attribution has higher value but is not currently executable because `RT-991.npy` is unverified/missing.

## Experiments Not Worth Running

- Any new training, sixth distillation, or New-Avenues continuation.
- Any renewed CatBoost search over k, subsets, weights, or endpoints.
- Re-ranking champions with the stale ledger or mixing weekend-harness AUC with TS-AUC.
- Re-running RT-100 reproduction or re-litigating RT-131.
- Unrestricted OOF-library optimization presented as confirmation.
- Direct Arm-C residualization until its predictions are provenance-verified.

## Final Decision

The repository documents a stable privileged-information frontier and severe practical diminishing returns, not a demonstrated causal opportunity of size +0.071. CatBoost k=2 at approximately +0.002 is the only credible recent positive and remains provisional. Proceed only with the zero-training audit. Do not authorize new training.
