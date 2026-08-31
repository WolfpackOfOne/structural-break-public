# Research Index

Single navigation point for the 2026 Real-Time structural-break research program. Historical wave reports are evidence, not current instructions. Start with `STATUS.md` and `MODEL_REGISTRY.md`, then use this index to find the underlying record.

## Current state

- `STATUS.md` — concise current-state pointer.
- `MODEL_REGISTRY.md` — models that matter now and their current status.
- `RESULTS.csv` — append-only quantitative experiment ledger.
- `EXPERIMENT_ID_MAP.md` — canonical mapping/provenance for RT identifiers.
- `RDOF_LEDGER.md` — degrees-of-freedom and preregistration accounting.
- `PROTOCOL.md` — historical binding research protocol.
- `NEGATIVE_RESULTS_INDEX.md` — searchable summary of major failures.
- `FAILED_EXPERIMENTS.md` — detailed negative evidence.
- `LESSONS_LEARNED.md` — cross-program scientific conclusions.

## Current external champion

**RT-1257** — official TS-AUC 0.6290, submission #16.

Evidence:
- `research/reports/catboost_specialist_2026/FINAL.md`
- `engineering/reports/rt1257_deployment/SUBMISSION_16.md`
- `submissions/RT1257_deployable.build.json`
- `engineering/reports/rt1257_deployment/CONSOLIDATION_HYGIENE_REVIEW.md`

RT-1257 is the RT-600 seven-slot architecture with exactly two learner swaps: RT-300 -> CAT-300/RT-1255 and RT-413 -> CAT-413/RT-1254.

## Production reference

**RT-600** — official TS-AUC 0.6268.

Retained permanently as the homogeneous seven-LightGBM reference and, until final RT-1257 production promotion is explicitly completed, the formal production anchor. A fresh local post-build RT-1257 Crunch test was captured on 2026-08-31 and PR #14 CI passed; formal RT-1257 production promotion is deferred to a separate owner tag/status action.

Key evidence:
- `research/FINAL_ARCHITECTURE_FREEZE.md`
- `research/FINAL_REPRODUCIBILITY_MANIFEST.json`
- RT-600 reliability/deployment reports imported through the engineering lineage.

## Current research survivors

- RT-1261 / CAT-412 — strongest remaining additional CatBoost slot question.
- RT-1263 / CAT-415 — research-alive.
- RT-1262 / CAT-414 — research-alive.
- RT-1260 / CAT-411 — research-alive with weaker corrected evidence.
- RT-1320 / M1 Arm-C residual student — research-alive; the only candidate that adds an 8th member. Primary E2-E1 +0.001516 at 4/5 folds against RT-1257 plus a matched seed clone. Not deployable yet.
- RT-995 / T2 — parked: real standalone teacher-distillation signal, mostly redundant in ensemble.

RT-1320 evidence:
- `research/reports/armc_residualization.md` — the endpoint/horizon decomposition of the Arm-C oracle.
- `research/reports/armc_residual_student/armc_residual_student.md` — original run, seed 20260830.
- `research/reports/armc_residual_student_confirm_s20260901/armc_residual_student.md` — confirmation run, seed 20260901.
- `research/reports/armc_residual_student_confirm_s20260901/E2_E1_addition_contract.json` — the protocol-correct primary endpoint.
- `research/reports/grok_response_followup.md` — roll-up, including the four sibling mechanisms that were killed.

Corrected CatBoost-composition evidence:
- `research/reports/deep_ensemble_frontier_2026/local/CSA04R_REANALYSIS.md`
- `research/reports/deep_ensemble_frontier_2026/local/CSA04R_REANALYSIS.json`
- `research/reports/deep_ensemble_frontier_2026/CSA04R_REANALYSIS_PREREG.md`

## Core causal representation and feature bank

The deployed program converged on a 500-column causal streaming bank assembled from:

- `m00_core` — multi-scale historical-null level/moment/tail evidence
- `m01_seq` — sequential detector memory / peaks / persistence
- `m02_dist` — distributional / PIT / divergence evidence
- `m03_dyn` — dependence and dynamic evidence
- `m04_resid` — residual / innovation evidence
- `m06_loc` — location-related causal evidence
- `m07_bayes` — Bayesian evidence

Code: `src/sbr/features/`.

Detailed historical feature wins and failures are indexed in `FAILED_EXPERIMENTS.md`, `NEGATIVE_RESULTS_INDEX.md`, and the wave/agent reports.

## Ensemble and calibration research

Topics:
- specialist diversity vs seed diversity
- smooth time-conditional CDF calibration
- fold-pure calibration
- exchangeable clone controls
- pair-flow and cell-repair diagnostics

Historical milestones include RT-250, RT-420, and RT-600. These are provenance/reference models, not alternative current champions.

## CatBoost specialist activation

Primary program:
- `research/reports/catboost_specialist_2026/PREREG.md`
- `research/reports/catboost_specialist_2026/FINAL.md`

Results:
- CAT-413 / RT-1254 — survives, component of RT-1257
- CAT-300 / RT-1255 — survives, component of RT-1257
- CAT-410 / RT-1256 — KILL
- RT-1257 — selected two-slot hybrid and current external champion

## Deep Ensemble Frontier / corrected composition analysis

- `research/reports/deep_ensemble_frontier_2026/PROGRAM_PLAN.md`
- `research/reports/deep_ensemble_frontier_2026/local/CSA04_PREREG.md`
- `research/reports/deep_ensemble_frontier_2026/local/CSA04_FINAL.md`
- `research/reports/deep_ensemble_frontier_2026/CSA04R_REANALYSIS_PREREG.md`
- `research/reports/deep_ensemble_frontier_2026/local/CSA04R_REANALYSIS.md`

Key conclusion: RT-1264's original five-slot best-k result is superseded as a current selection conclusion. Corrected CSA-04R / RT-1265 selects k=2, exactly RT-1257. CAT-412 is the third slot in the nominal k=3 maximum, but its edge is below the paired-bootstrap noise floor.

## Learner diversity and GPU tabular

Original Learner Diversity:
- RT-1250 TabM — INFEASIBLE under original frozen compute contract
- RT-1251 CatBoost — INTERESTING and became the learner used for specialist activation
- RT-1252 RealMLP — INFEASIBLE under original frozen compute contract

GPU re-authorization after hardware benchmark:
- `research/reports/gpu_tabular_2026/FULL_OOF_PREREG.md`
- `research/reports/gpu_tabular_2026/FROZEN_GPU_CONFIG.json`
- `research/reports/deep_ensemble_frontier_2026/crunch/H4_GPU_TABULAR_RESULT.md`
- reproduction code under `research/scripts/gpu_tabular/`
- optional dependencies: `research/requirements-gpu-tabular.txt`

Binding results:
- RT-1258 TabM — KILL
- RT-1259 RealMLP — KILL

These are distinct GPU arms, not second KILL labels attached to RT-1250/RT-1252.

## Teacher distillation / privileged information

Wave 7:
- RT-994 — pure distillation, insufficient fold consistency
- RT-995 / T2 — strong causal student standalone signal, but final RT-600 integration mostly redundant

Current status: RT-995 is PARKED, not KILL.

Evidence lives in the Wave-7 teacher-distillation and T2-promotion reports and the milestone tag `wave7-t2-promotion-mostly-redundant`.

## Offline oracle / information frontier

Purpose: determine how much information exists in future/full-sequence views and known-boundary diagnostics. These models are not deployable.

Key branch/tag lineage:
- `codex/oracle-information-frontier-2026`
- tag `oracle-information-frontier-2026-study`
- Wave-7 D3R information-frontier evidence

Never compare an offline/oracle score directly to a real-time leaderboard score as if they were the same model class.

## Causal Representation Frontier

Completed negative program. CRF-01 and corrected CRF-02 did not produce a promotable representation; pair-flow and isolation controls failed. No active CRF model survives.

Final evidence under `research/reports/causal_representation_frontier_2026/` and branch history `research/causal-representation-frontier-2026`.

## Leaderboard Alpha

Completed negative program. LA-03 showed some clone-relative marginal movement but failed the fixed-null isolation gate and pair-flow tests. No Leaderboard Alpha mechanism survives.

Evidence under `research/reports/leaderboard_alpha_2026/`.

## New Avenues

The broad mechanism survey and subsequent pilots are retained as evidence. Executed first/second sweeps did not yield a confirmation candidate. Do not treat the earlier design queue as live merely because it appears in an older status document.

Evidence:
- `research/NEW_AVENUES_2026.md`
- `research/new_avenues_2026.csv`
- `research/reports/new_avenues_2026/`

## Wave 8 future-aware transfer

ORR, TGMC, SST, PCFB, and CFEP all failed full-population promotion gates. Program closed without a surviving candidate.

Historical branch/tag:
- `research/wave8-future-aware-distillation`
- `wave8-future-aware-final`

## Negative-result navigation

Use:
1. `NEGATIVE_RESULTS_INDEX.md` for fast lookup.
2. `FAILED_EXPERIMENTS.md` for detailed hypothesis/result/why/retry reasoning.
3. The linked final program report for full controls and quantitative evidence.

## Historical architecture milestones

Older deployable or near-deployable systems such as RT-160, RT-190, RT-250, and RT-420 remain important for provenance and methodology but are superseded by RT-600/RT-1257 for current decisions.

## External submissions

- RT-600 — 0.6268 official external TS-AUC.
- RT-1257 / submission #16 — 0.6290 official external TS-AUC.

The leaderboard is used as external validation, not as a hyperparameter oracle.

## Reproducibility and consolidation

- `research/archive/2026-08-30/` — immutable pre-consolidation branch/model/results snapshot.
- `research/archive/2026-08-31/BRANCH_TAG_PRUNING_AUDIT.md` — final branch/tag pruning audit for PR #14.
- `docs/MAIN_CONSOLIDATION_REPORT_2026.md` — consolidation actions, verified findings, gate status, and unresolved items.
- `engineering/reports/rt1257_deployment/CONSOLIDATION_HYGIENE_REVIEW.md` — RT-1257 artifact/provenance review.

No historical branch should be deleted until its unique history is reachable from the consolidation/main history or preserved by an immutable milestone tag.
