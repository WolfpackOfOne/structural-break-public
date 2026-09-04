# Model Registry

Canonical current-state model registry for the 2026 ADIA Lab / CrunchDAO Structural Break Challenge — Real-Time Edition.

This file answers **what matters now**. It does not replace `RESULTS.csv` (the quantitative experiment ledger) or `EXPERIMENT_ID_MAP.md` (the immutable ID/provenance map). Historical experiment rows are never rewritten to make current conclusions look cleaner.

## Status vocabulary

- **EXTERNAL_CHAMPION** — best official external score, but not necessarily the formal production anchor.
- **REFERENCE** — retained canonical comparison model.
- **ACTIVE_COMPONENT** — a member of the current external champion.
- **RESEARCH_ALIVE** — passed meaningful individual evidence but has not demonstrated a material improvement over the current champion.
- **PARKED** — real signal, but insufficient marginal ensemble value for active promotion.
- **SUPERSEDED** — historical result replaced by a corrected analysis or stronger lineage; evidence remains valid as history.
- **KILL** — failed its preregistered promotion endpoint.
- **INFEASIBLE** — authorized arm did not reach a predictive score under its frozen execution contract.
- **DIAGNOSTIC / ORACLE_NONCAUSAL / INVALID** — evidence-only; never production candidates.

## Current model set

| Model | Alias / family | Role | Key evidence | Current status |
| --- | --- | --- | --- | --- |
| **RT-1257** | CAT-300 + five incumbent LightGBM specialists + CAT-413 | Seven-member base of the current champion; formal production anchor | Official submission #16 TS-AUC **0.6290**, vs RT-600 0.6268 (**+0.0022**). Internal E2-E0 +0.002026322; marginal_vs_clone +0.002407205; 5/5 positive. Fresh local post-build Crunch test and PR #14 CI passed on 2026-08-31. | **FORMAL_PRODUCTION_ANCHOR — promoted 2026-09-02. CEASED to be EXTERNAL_CHAMPION 2026-09-03, superseded by RT-1320 at 0.6303. Formal anchor promotion of RT-1320 is a separate explicit action and has NOT been taken.** |
| **RT-600** | Seven LightGBM specialists + fold-pure smooth time-conditional CDF calibration | Pure-LightGBM reference and current formal production anchor | Official external TS-AUC **0.6268**; frozen/reliability-tested lineage. | **REFERENCE** — permanent homogeneous-LightGBM comparison model; ceased to be the formal anchor 2026-09-02 |
| **RT-1254** | CAT-413 | Replaces RT-413 inside RT-1257 | standalone 0.620440244; fixed-slot E2-E0 +0.001244347; marginal_vs_clone +0.001087151; 5/5 positive; dominant net +93. | **ACTIVE_COMPONENT** |
| **RT-1255** | CAT-300 | Replaces RT-300 inside RT-1257 | standalone 0.620242061; fixed-slot E2-E0 +0.001126876; marginal_vs_clone +0.001029459; 5/5 positive; dominant net +75. | **ACTIVE_COMPONENT** |
| **RT-1261** | CAT-412 | Strongest remaining additional CatBoost slot candidate | RT-1257-relative adjudication: primary E2-E1 +0.001087276 (5/5; 95% CI [+0.000120550, +0.002001007]) but deployment E2-E0 only **+0.000338234** (3/5; 95% CI [-0.000631306, +0.001358457]), below the 0.0011 noise floor; dominant-cell net -31. | **PARKED / NO_PROMOTION — lane closed 2026-09-01; alt-partition leg scoped, costed and declined** |
| **RT-1263** | CAT-415 | Additional CatBoost slot replacement | RT-1257-relative adjudication: primary E2-E1 +0.000283602 (3/5) and deployment E2-E0 +0.000175228 (3/5; 95% CI [-0.000756519, +0.001139340]); dominant-cell net -16 and mature-vs-never net -3. | **PARKED / NO_PROMOTION — lane closed 2026-09-01** |
| **RT-1262** | CAT-414 | Additional CatBoost slot replacement | RT-1257-relative adjudication: primary E2-E1 +0.001057324 (4/5) but deployment E2-E0 +0.000178858 (4/5; 95% CI [-0.000729204, +0.001056890]); dominant-cell net -84. | **PARKED / NO_PROMOTION — lane closed 2026-09-01** |
| **RT-1260** | CAT-411 | Additional CatBoost slot replacement | RT-1257-relative adjudication: primary E2-E1 +0.001317358 (5/5) is driven by a degraded clone control; deployment E2-E0 is only +0.000133942 (2/5; 95% CI [-0.000455188, +0.000748957]). | **PARKED / NO_PROMOTION** |
| **RT-1320** | RT-1257 + M1 Arm-C residual student (8th member) | **Current external champion** | **Official submission #19 TS-AUC 0.6303 (2026-09-03), vs RT-1257 0.6290 — realized +0.0013; vs RT-600 0.6268 — +0.0035.** The secondary endpoint E2-E0 against RT-1257 predicted **+0.001416**; realized **+0.0013**. Second consecutive external calibration point matching its internal estimate to ~0.0001 (RT-1257 predicted +0.0020/+0.0024, realized +0.0022). **Phase 1 four-partition gate PASSED 2026-09-01.** Addition contract vs RT-1257, E1 = RT-1257 + one matched added RT-403 clone. Mean E2-E1 across canonical/alt1/alt2/alt3 **+0.0017678**, 4/4 partitions positive, worst +0.0014825, none negative (rule: mean >= +0.0011, >0 on >=3 of 4, none < -0.0011). Per partition: canonical +0.0015159 (4/5), alt1 +0.0021146 (5/5), alt2 +0.0014825 (5/5), alt3 +0.0019582 (4/5). E1 control flat-to-negative on all four, so the RT-1264/CSA-04 inflation shape is absent. Fold 0 negative on both endpoints. Gain is a **never-break-cut** gain (never-break pair net +0.003409; pre-break +0.000363, ~0) - a false-positive repair mechanism, not a broad improvement. Causality passes at research and artifact level. | **EXTERNAL_CHAMPION — promoted 2026-09-03 on submission #19. `external_score` blocker CLEARED. Formal production-anchor promotion is a separate explicit action, not yet taken.** |
| **M1 student** | Arm-C residual student, the 8th member of RT-1320 | Added member of the current external champion | Carried into the champion by submission #19 (0.6303). Its lift over the matched added clone is RT-1320's primary endpoint above. | **ACTIVE_COMPONENT — since 2026-09-03** |
| **RT-1321** | LS-KD joint lag-space characteristic-kernel discrepancy (`m19_lskd`), proposed 9th member of RT-1320 | Candidate ninth member; executes avenue G5 and the "kernel PCA / explicitly nonlinear" gap RT-1215 left open | Addition contract on RT-1320: E0 0.629253722, E1 0.628699976, E2 0.629219268. Primary **E2-E1 +0.000519292** (4/5) against a +0.0011 floor; deployment **E2-E0 -0.000034454**; paired bootstrap 95% CI **[-0.000116, +0.001069]** contains zero. Block standalone 0.542467 vs the incumbent's 0.640281; within-`t` rho **0.16**; conditional AUC on the pairs RT-1320 inverts **0.4712, below chance**. Pair flow E2 vs E0: dominant -5, mature-vs-never -54, mature-vs-pre -32. Full causality suite passes including batch/stream bitwise parity. | **KILL** — no final fit, no artifact, no submission |
| **RT-1322** | LS-KD matched marginal-kernel control (`m19_lskm`) | The `E1` arm of RT-1321's addition contract; also a standalone finding about ensemble saturation | **E1-E0 = -0.000553746**: an ordinary matched ninth LightGBM member *degrades* RT-1320, which is why RT-1321's positive primary endpoint must not be read as a near-miss (RT-1264/CSA-04 inflation mode). The control block also beat the joint candidate on every standalone cut (fold-0 whole 0.544396 vs 0.542467; dominant cell 0.567139 vs 0.557594). | **CONTROL** |
| **RT-995 / T2** | Teacher-distilled causal LightGBM | Privileged-information student | Strong standalone improvement (~+0.00943 vs matched T0) but final ensemble marginal vs matched seed clone only ~+0.000237; mostly redundant. | **PARKED** |
| **RT-1265** | CSA-04R corrected reanalysis identity | Corrected hybrid-selection result | Fixed E2-E0 + preregistered parsimony selects k*=2 = CAT-413 + CAT-300, exactly RT-1257. `delta_vs_RT1257=0`; no new OOF vector. | **SUPERSEDING_ANALYSIS / NOT_DISTINGUISHABLE** |
| **RT-1264** | Original CSA-04 five-slot hybrid | Historical best-k result under flawed selection endpoint | Original E2-E1 endpoint was contaminated by worsening clone control as k grew. Descriptive 63-subset appendix cannot select a champion. | **SUPERSEDED BY RT-1265** |
| **RT-1256** | CAT-410 | CatBoost RT-410 replacement | marginal_vs_clone +0.000753095, below +0.0010 gate. | **KILL** |
| **RT-1250** | TabM, Learner Diversity | Original local/default full-scale feasibility arm | Did not complete under its frozen compute contract; shrinking/retuning was not authorized. | **INFEASIBLE** |
| **RT-1252** | RealMLP, Learner Diversity | Original local/default full-scale feasibility arm | Did not complete under its frozen compute contract; shrinking/retuning was not authorized. | **INFEASIBLE** |
| **RT-1258** | GPU-01 TabM | Fresh separately-numbered GPU full-OOF arm after hardware removed RT-1250's feasibility blocker | Full 5-fold OOF completed. standalone 0.6006475014; marginal_vs_clone +0.000040947; E2-E0 -0.000789923; 3/5 positive. | **KILL** |
| **RT-1259** | GPU-02 RealMLP | Fresh separately-numbered GPU full-OOF arm after hardware removed RT-1252's feasibility blocker | Full 5-fold OOF completed. standalone 0.5599057458; marginal_vs_clone -0.003223744; 0/5 positive. | **KILL** |

## RT-1257 composition

RT-1257 is not an eight- or nine-model additive ensemble. It preserves the seven RT-600 slots and swaps exactly two learner implementations:

1. RT-300 -> **RT-1255 / CAT-300**
2. RT-410 -> RT-410 LightGBM
3. RT-411 -> RT-411 LightGBM
4. RT-412 -> RT-412 LightGBM
5. RT-413 -> **RT-1254 / CAT-413**
6. RT-414 -> RT-414 LightGBM
7. RT-415 -> RT-415 LightGBM

The two CatBoost models are components of RT-1257, not extra members added on top.

## RT-1264 / RT-1265 correction

The historical RT-1264 row remains in the experiment ledger. Current-state documents must not describe it as a live upside candidate. CSA-04R found that the original E2-E1 best-k endpoint inflated with k because the matched clone control degraded as more slots were replaced. The corrected fixed E2-E0 endpoint and preregistered parsimony/noise rule select k=2, exactly RT-1257. The nominal k=3 maximum adds CAT-412 but improves only +0.000338234 over RT-1257, less than the 0.0011 paired-bootstrap noise floor. Therefore RT-1264 is **SUPERSEDED**, RT-1265 is the corrected analysis, and CAT-412 is the remaining single-slot research question.

## TabM / RealMLP ID relationship

RT-1250 and RT-1252 are the original Learner Diversity feasibility records. They were INFEASIBLE and never received a binding predictive verdict. RT-1258 and RT-1259 are **distinct, freshly preregistered GPU arms**, created after an RTX 4090 benchmark showed that the same intended official/default-scale configurations could fit the compute budget without shrinking. The later H4 execution is a technical recovery of the already-preregistered RT-1258/RT-1259 arms after an alignment issue; it is not a third attempt and it does not make RT-1250/RT-1252 independent KILL results.

## Production status

RT-1257 is the current **external champion** at 0.6290. RT-600 remains the formal production anchor until final RT-1257 production promotion is explicitly completed. The repository has verified that the manifest/build-SHA gap spans packaging, dependency declaration, and engineering evidence changes rather than `src/sbr` model/feature source changes, and the PR #14 follow-up recorded a fresh local post-build Crunch test PASS plus passing CI. Formal RT-1257 production promotion is deferred to a separate owner tag/status action.

## Future comparison rule

Serious new candidates should be evaluated for **marginal information relative to RT-1257**, not just standalone AUC. RT-600 remains a permanent homogeneous LightGBM reference. For a candidate C, prefer a matched test of E0=RT-1257, E1=RT-1257 with an exchangeable matched control, and E2=RT-1257 with C, with E2-E1 as the primary endpoint and E2-E0, fold consistency, pair flow, dominant-cell repair, mature-vs-never repair, correlation, runtime, causality, and deployment complexity as supporting evidence.

First application of this rule to the four residual CSA-04 slot candidates is filed in `research/reports/deep_ensemble_frontier_2026/local/RT1257_SLOT_ADJUDICATION.md`.

**Amended 2026-09-04 after RT-1321.** The champion is now RT-1320, so the
addition contract for a new candidate is `E0 = RT-1320`, `E1 = RT-1320 + matched
control`, `E2 = RT-1320 + candidate`. Two requirements are now explicit:

1. **Report `E1 - E0` as a first-class number.** RT-1321 measured it at
   **-0.000554** — an ordinary matched ninth LightGBM member makes RT-1320
   *worse*. A primary endpoint of `+0.00052` against that control is "degrades
   the champion less than the control does", not a marginal gain, and `E2 - E0`
   said so directly (-0.000034). This is the third time in the programme that a
   degrading control has manufactured a positive-looking primary
   (RT-1260/CAT-411, RT-1264/CSA-04, RT-1321).
2. **A matched control should be mechanism-matched, not a seed clone, whenever
   the hypothesis names a mechanism.** RT-1321's control shared its inputs,
   depths, half-lives, RFF dimension, bandwidth, historical reference,
   normalization and column count and differed only in whether the kernel map was
   joint or coordinate-separable. That is what made "joint lag structure adds
   nothing beyond nonlinear marginal structure" a measurable statement rather
   than a rhetorical one.

## RT-1320 — what it would take to become an ACTIVE_COMPONENT

RT-1320 is the first candidate in this program that **adds** a member instead of
replacing one, so it is worth being explicit that `RESEARCH_ALIVE` is not
timidity. By the status vocabulary at the top of this file, `ACTIVE_COMPONENT`
means *a member of the current external champion*. RT-1257 scored 0.6290
externally without RT-1320 in it, so RT-1320 cannot be an active component today
no matter how good its internal evidence is.

Status as of 2026-09-01: **items 1–4 are done; item 5 is the sole remaining
blocker** and needs Crunch quota rather than local work. The original list is
kept below with each item marked, so the checklist reads as a record rather than
being rewritten.

1. **Alternate-partition leg. — DONE 2026-09-01, PASS.** Four-partition mean E2−E1 +0.0017678, 4/4 positive. See `reports/rt1320_promotion/PHASE1_STAGE2.md`. Note the premise below is stale: the seven specialists did *not* need refitting, their alt OOF vectors were already on disk from wave 4/5. All evidence is on canonical `folds.parquet`, and
   `FINAL_ARCHITECTURE_FREEZE.md` records that canonical was the most favourable
   of the four partitions for the RT-600 specialisation delta. The leg requires
   refitting the seven specialists under `folds_alt*.parquet` first, because
   their OOF is cross-fitted on the canonical partition.
2. **Causality gate. — DONE.** Research-stage PASS (`PHASE2_CAUSALITY.md`) and artifact-level PASS on the assembled 8-member artifact (`engineering/reports/rt1320_promotion_prep/ARTIFACT_CAUSALITY.json`). `PROTOCOL_CHAMPION_2026.md` requires prefix invariance,
   no total-horizon dependence, per-series independence, deterministic replay,
   and output-contract checks before a predictive score is interpreted. The
   student reuses the already-verified 500-column causal bank and adds no new
   features, so much of this is inherited — but none of it has been run against
   the student's own inference path. `causal_verified=no` in the ledger.
3. **Final-10k fit. — DONE 2026-09-01.** Regenerated at 10k locally after the Crunch run was lost to quota; `model.txt.7` and `RT-1320_student_scdf.json` built. Production fits use `folds_final10k.parquet` over 10,000
   series. The nested Arm-C teacher that generates RT-1320's training target
   exists only over the 8,000 dev series, so the target itself must be
   regenerated at 10k before an artifact can be built.
4. **Production artifact and manifest. — DONE.** `models/rt1320_final` built by `research/scripts/rt1320_assemble_artifact.py`; 8 model files, 8 booster slices, 8 calibration models. `src/sbr/production/model.py` is
   member-count agnostic and manifest-driven, so an 8th member is an artifact and
   calibration-payload change rather than a code redesign. The student is a
   `regression`-objective booster whose raw output is a residual prediction, not
   a probability; it reaches a common scale through the same SCDF calibration as
   every other member, which is rank-based and therefore tolerates that.
**Reporting defect, fixed 2026-08-31.** The original `rt1257_combo_analysis` block
compared RT-1257 + student against a control that cloned the CAT-300 and CAT-413
members as well as adding a clone, and reported the resulting +0.0037 as if it
were a champion-relative marginal. It is not an E1. The script now emits the
protocol's primary endpoint against RT-1257 + one matched clone, and labels the
old quantity `all_clone_control (NOT an E1)`. The two committed reports carry a
correction header; their numbers were left as produced.

5. **A fresh Crunch test and an external score. — HALF DONE; THE REMAINING BLOCKER.** Local Crunch test passed twice with bit-identical predictions (`CRUNCH_TEST.json`, `max_abs_prediction_delta=0.0`). The external score is not obtained and needs quota. RT-1257 itself is still not the
   formal production anchor pending its own promotion action, so an 8th member
   stacks on top of an unpromoted champion.
