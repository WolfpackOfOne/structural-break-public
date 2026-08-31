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
| **RT-1257** | CAT-300 + five incumbent LightGBM specialists + CAT-413 | Best measured deployable system | Official submission #16 TS-AUC **0.6290**, vs RT-600 0.6268 (**+0.0022**). Internal E2-E0 +0.002026322; marginal_vs_clone +0.002407205; 5/5 positive. Fresh local post-build Crunch test and PR #14 CI passed on 2026-08-31. | **EXTERNAL_CHAMPION — formal production promotion deferred** |
| **RT-600** | Seven LightGBM specialists + fold-pure smooth time-conditional CDF calibration | Pure-LightGBM reference and current formal production anchor | Official external TS-AUC **0.6268**; frozen/reliability-tested lineage. | **REFERENCE / FORMAL_PRODUCTION_ANCHOR** |
| **RT-1254** | CAT-413 | Replaces RT-413 inside RT-1257 | standalone 0.620440244; fixed-slot E2-E0 +0.001244347; marginal_vs_clone +0.001087151; 5/5 positive; dominant net +93. | **ACTIVE_COMPONENT** |
| **RT-1255** | CAT-300 | Replaces RT-300 inside RT-1257 | standalone 0.620242061; fixed-slot E2-E0 +0.001126876; marginal_vs_clone +0.001029459; 5/5 positive; dominant net +75. | **ACTIVE_COMPONENT** |
| **RT-1261** | CAT-412 | Strongest remaining additional CatBoost slot candidate | Corrected greedy k=3 = CAT-413 + CAT-300 + CAT-412 reaches E2-E0 +0.002364556, only **+0.000338234 vs RT-1257**, below the 0.0011 paired-bootstrap noise floor. | **RESEARCH_ALIVE — highest-priority residual slot question** |
| **RT-1263** | CAT-415 | Additional CatBoost slot replacement | Positive individual evidence; corrected multi-slot curve does not establish improvement over RT-1257. | **RESEARCH_ALIVE** |
| **RT-1262** | CAT-414 | Additional CatBoost slot replacement | Positive individual evidence; corrected multi-slot curve does not establish improvement over RT-1257. | **RESEARCH_ALIVE** |
| **RT-1260** | CAT-411 | Additional CatBoost slot replacement | Passed original clone-relative individual gate, but corrected E2-E0 is small and mature-vs-never evidence is weak. | **RESEARCH_ALIVE** |
| **RT-1320** | M1 Arm-C residual student | First candidate to *add* an 8th member rather than swap a slot | Addition contract vs RT-1257: primary E2-E1 **+0.001516** (4/5 folds, fold 0 negative); secondary E2-E0 +0.001416. On the RT-600 lane +0.001965 at 5/5. Confirmed at a second seed. | **RESEARCH_ALIVE — clears the primary endpoint; NOT deployable, NOT a member of any promoted system** |
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

## RT-1320 — what it would take to become an ACTIVE_COMPONENT

RT-1320 is the first candidate in this program that **adds** a member instead of
replacing one, so it is worth being explicit that `RESEARCH_ALIVE` is not
timidity. By the status vocabulary at the top of this file, `ACTIVE_COMPONENT`
means *a member of the current external champion*. RT-1257 scored 0.6290
externally without RT-1320 in it, so RT-1320 cannot be an active component today
no matter how good its internal evidence is.

Outstanding before promotion is even arguable:

1. **Alternate-partition leg.** All evidence is on canonical `folds.parquet`, and
   `FINAL_ARCHITECTURE_FREEZE.md` records that canonical was the most favourable
   of the four partitions for the RT-600 specialisation delta. The leg requires
   refitting the seven specialists under `folds_alt*.parquet` first, because
   their OOF is cross-fitted on the canonical partition.
2. **Causality gate.** `PROTOCOL_CHAMPION_2026.md` requires prefix invariance,
   no total-horizon dependence, per-series independence, deterministic replay,
   and output-contract checks before a predictive score is interpreted. The
   student reuses the already-verified 500-column causal bank and adds no new
   features, so much of this is inherited — but none of it has been run against
   the student's own inference path. `causal_verified=no` in the ledger.
3. **Final-10k fit.** Production fits use `folds_final10k.parquet` over 10,000
   series. The nested Arm-C teacher that generates RT-1320's training target
   exists only over the 8,000 dev series, so the target itself must be
   regenerated at 10k before an artifact can be built.
4. **Production artifact and manifest.** `src/sbr/production/model.py` is
   member-count agnostic and manifest-driven, so an 8th member is an artifact and
   calibration-payload change rather than a code redesign. The student is a
   `regression`-objective booster whose raw output is a residual prediction, not
   a probability; it reaches a common scale through the same SCDF calibration as
   every other member, which is rank-based and therefore tolerates that.
5. **A fresh Crunch test and an external score.** RT-1257 itself is still not the
   formal production anchor pending its own promotion action, so an 8th member
   stacks on top of an unpromoted champion.
