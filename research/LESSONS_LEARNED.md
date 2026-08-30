# Lessons Learned

Cross-program conclusions earned by the 2026 Real-Time structural-break research program. These are synthesis statements, not replacements for the underlying experiment reports.

## 1. The problem is primarily one of conditional ranking, not raw break probability

The competition score rewards correct ordering within time strata. This made time-aware calibration, matched controls, and within-t ranking behavior as important as standalone probabilistic fit. Several apparently stronger standalone models did not improve the ensemble because they did not repair the right pairwise ordering errors.

## 2. The 500-column causal bank became a strong common representation

The mature system converged on a 500-column streaming feature bank built from core historical-null evidence, sequential-memory features, distributional evidence, dynamics/dependence, residuals, location-related signals, and Bayesian evidence. Many later ideas failed because they re-expressed information already present in this bank rather than introducing a genuinely new state variable.

This is visible in repeated negative results: detector current levels, slope-of-evidence summaries, thresholded hot counts, short EWMAs, rank-CUSUM restatements, and simple trend families often looked sensible in isolation but added little or hurt conditional on the mature bank.

## 3. Specialist diversity mattered more than simply cloning more trees

The seven-specialist architecture was a major step because members saw different feature subsets, objectives, sampling rules, or inductive biases. Seed-clone controls became essential precisely because an eighth exchangeable tree can improve an ensemble slightly even when it adds no new mechanism. Promotion therefore cannot be judged against the incumbent ensemble alone; it must beat the value of an exchangeable matched control.

## 4. Calibration was part of the model, not presentation polish

Cross-fitted smooth time-conditional CDF calibration materially improved deployable ensemble ranking while preserving causality. This made raw model scores from different specialists comparable enough to combine without leaderboard-driven weights or a learned stacker.

## 5. Learner-family diversity produced the first externally validated move beyond RT-600

RT-1257 replaced exactly two RT-600 LightGBM slots with CatBoost implementations: CAT-300 and CAT-413. It scored 0.6290 externally versus RT-600's 0.6268, a +0.0022 move. The realized external delta fell between the internal E2-E0 (+0.002026) and marginal-vs-clone (+0.002407) estimates that licensed deployment.

This is evidence that the internal promotion battery identified a real improvement on this artifact. It is not enough evidence to infer a universal internal-to-external transfer coefficient, and the leaderboard must not become a hyperparameter oracle.

## 6. “Use more CatBoost” is not the lesson

CAT-413 and CAT-300 were the right two replacements. Later single-slot work found additional positive CatBoost specialists, but the corrected CSA-04R composition analysis did not establish that a larger CatBoost hybrid beats RT-1257.

The corrected fixed E2-E0 greedy curve peaks nominally at k=3 after adding CAT-412, but the +0.000338234 edge over RT-1257 is below the 0.0011 paired-bootstrap noise floor. Parsimony therefore selects k=2, exactly RT-1257.

The useful next question is not “how many CatBoost slots can we replace?” It is “does CAT-412 or another survivor repair a stable subset of RT-1257 mistakes that can be extracted without importing offsetting damage?”

## 7. Control behavior can manufacture a misleading marginal headline

RT-1264 is a particularly important methodological lesson. The original multi-slot analysis selected on E2-E1 while the matched clone control itself deteriorated as k increased. That made larger k look better partly because E1 got worse.

CSA-04R corrected this by using fixed E2-E0 as the deployment endpoint and a preregistered parsimony/noise rule. The correction selected RT-1257's two-slot composition. Future ensemble tests must inspect E0, E1, and E2 separately rather than trusting a single difference statistic.

## 8. Offline future information is real and large

Oracle/full-sequence studies established that future-state information contains substantial predictive power that a same-prefix causal learner does not fully recover. Those scores are diagnostics, not deployable leaderboard comparables, but they establish that the real-time problem has an information-representation gap rather than merely an under-tuned tree problem.

## 9. Distillation can transfer future information into a legal causal student

RT-995/T2 materially improved its matched standalone student while remaining causal at inference. That is a substantive success: privileged future information at training time can shape a better prefix-only model.

## 10. Strong standalone alpha is not enough

RT-995/T2 is the clearest example. Its standalone improvement was large, but its marginal contribution after insertion into the mature ensemble was only a few ten-thousandths over a matched clone. The information was largely already represented by RT-600.

This changed the research standard. The primary question for future models is not “is this model good?” but “does this model know something the champion does not?”

## 11. Low correlation is not enough either

TabM is a useful counterexample to a simplistic diversity thesis. RT-1258 reached standalone mean TS-AUC ~0.60065 and rho ~0.585 versus RT-401 — both reasonably competent and genuinely different — yet its marginal-vs-clone result was only +0.000040947 and E2-E0 was negative.

A model can be both different and decent while still repairing the wrong errors or losing its gains elsewhere.

## 12. Neural tabular models did not solve the diversity problem

RT-1258 TabM and RT-1259 RealMLP completed the GPU-authorized five-fold test and both failed the binding ensemble endpoint. RealMLP was clearly harmful; TabM was close to zero marginal value despite positive dominant-cell/mature-vs-never tendencies.

This closes the hypothesis that simply switching from boosted trees to a modern tabular neural architecture creates useful ensemble alpha on the same 500-column representation.

## 13. Hardware feasibility and scientific failure are different states

The original RT-1250 TabM and RT-1252 RealMLP arms were INFEASIBLE under their frozen local/default compute contract. They were not scored KILL results. RTX 4090 benchmarking later removed that blocker without shrinking the intended configurations, which justified fresh GPU arms RT-1258/RT-1259. Those later arms supplied the scientific KILL verdicts.

Preserving this distinction matters because “the model did not fit the budget” and “the model fit and did not add alpha” imply different things about the hypothesis.

## 14. Continuous evidence usually beat arbitrary discretization

Several sequential-feature failures had the same structure: hard counts and fixed thresholds discarded ordering that a tree could learn from continuous features directly. `xc_n_hot` is the cleanest example. When using flexible nonlinear downstream learners, discretization needs a mechanism-level reason, not just interpretability appeal.

## 15. Recursive memory mattered more than detector current level

Sequential detector current values were largely redundant with the multi-scale window bank. Peaks, persistence, decay-from-peak, and other stateful summaries were much more useful because they carried information about the path the detector had taken, not merely its present level.

That is an important design principle for future streaming features: add state, not another smoothed view of the current window.

## 16. Causality bugs can look completely plausible

The mixture-GLR normalizer once depended on total online series length and produced plausible features until prefix-invariance exposed it. Another oracle experiment leaked the break state through a missingness mask and produced an enormous false score.

Therefore every new mechanism needs causality/prefix tests before any predictive number is interpreted. A spectacular AUC from an unverified causal path is evidence of a possible bug first, alpha second.

## 17. Pair-flow and cell diagnostics are necessary complements to global AUC

Several candidates improved one headline metric while damaging important ordering subsets. Dominant-cell, mature-vs-never, and mature-vs-prebreak repair/damage counts exposed these cases. A future promotion should require not only aggregate marginal improvement but also an intelligible error-repair pattern without severe offsetting damage.

## 18. Negative results are reusable assets

The project has repeatedly tested intuitive ideas that future agents are likely to propose again: trend features, threshold counts, alternate objectives, future-aware transfer, neural representations, causal representation learning, per-series adaptation, and others. The value of `FAILED_EXPERIMENTS.md` is that it records not merely a KILL label but the mechanism of failure and whether a materially different retry could be justified.

## 19. The correct baseline has changed

Historically, the research question was “can this beat a single LightGBM?” and later “can this beat RT-600?” After the external 0.6290 result, serious new research should primarily ask whether a candidate adds marginal information to RT-1257.

RT-600 remains important as the homogeneous seven-LightGBM reference because it isolates the value of learner-family diversity. RT-1257 is the target a new production candidate must ultimately improve.

## 20. Repository provenance is part of scientific validity

The late audit found that the branch described as the canonical research trunk did not contain the later RT-1250+ champion lineage in its `RESULTS.csv`, while sibling branches contained different later generations of that ledger. It also found scored RT-1258/RT-1259 results in reports without corresponding rows in the inspected canonical ledger generation.

A research conclusion is not fully institutionalized until its ID, result, report, code/config, and current-status interpretation are all reachable from the canonical repository state. The consolidation therefore treats registry/index/ledger integrity as part of model validation, not housekeeping.
