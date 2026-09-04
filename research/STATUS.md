# STATUS

Concise canonical pointer to the current state of the 2026 ADIA Lab / CrunchDAO Structural Break Challenge — Real-Time Edition research program.

## External champion

**RT-1320 — official TS-AUC 0.6303 (submission #19, 2026-09-03).**

RT-1320 is RT-1257 plus the M1 Arm-C residual student as an 8th member. It
improves on RT-1257's 0.6290 by **+0.0013** and on RT-600's 0.6268 by **+0.0035**.

The internal evidence predicted this closely. The secondary endpoint E2-E0
against RT-1257 — the estimate of what the student adds to the champion — was
**+0.001416**; the realized external gain is **+0.0013**. That is the second
consecutive external calibration point matching its internal estimate to about
0.0001, after RT-1257's (+0.0020/+0.0024 predicted, +0.0022 realized). The
dev-fold addition contract, the 0.0011 noise floor and the four-partition gate
are now externally validated twice. This is a reason to trust the internal
machinery, **not** a license to tune on the leaderboard.

**Scope.** RT-1320's gain is a never-break-cut gain (never-break pair net
+0.003409; pre-break +0.000363, approximately zero). It is a false-positive
repair mechanism, not a broad improvement, and should not be described as one.

**Formal production anchor is unchanged.** RT-1257 remains the anchor; promoting
RT-1320 to anchor is a separate explicit action that has not been taken.

> ## ⚠ OUTSTANDING ACTION — SELECT THE SUBMISSION
>
> **0.6303 is a PUBLIC-set score. It does not decide prizes.** At close, only each
> participant's **selected** submission is scored once on the unseen private set,
> and selection is **not** automatically the latest submission
> (`PLATFORM_CONSTRAINTS.md` §0).
>
> **Verify that submission #19 / RT-1320 is explicitly selected.** An unset or
> stale selection forfeits up to 0.0035 — more than every remaining modelling
> gain combined. This is the highest-leverage open item in the project.
>
> Expect the private *level* to come in below 0.6303 (`FINAL_ARCHITECTURE_FREEZE.md`
> §4 base case ~0.615). The **delta** is what transfers, and RT-1320's delta over
> RT-1257 is positive on all five populations measured: canonical +0.0015159,
> alt1 +0.0021146, alt2 +0.0014825, alt3 +0.0019582, public +0.0013.

### Prior external champion

**RT-1257 — official TS-AUC 0.6290 (submission #16, 2026-08-29). Formally promoted to production anchor 2026-09-02. Ceased to be external champion 2026-09-03.**

It improves on RT-600's 0.6268 by **+0.0022**.

RT-1257 preserves the seven RT-600 specialist slots and replaces exactly two learners:

- RT-300 LightGBM -> **RT-1255 / CAT-300**
- RT-410 LightGBM
- RT-411 LightGBM
- RT-412 LightGBM
- RT-413 LightGBM -> **RT-1254 / CAT-413**
- RT-414 LightGBM
- RT-415 LightGBM

The internal evidence that licensed RT-1257 was E2-E0 **+0.002026322**, marginal_vs_clone **+0.002407205**, and 5/5 positive folds. The realized external +0.0022 landed between those two internal estimates. This is one useful external calibration point, not a license to tune on the leaderboard.

## Formal production anchor

**RT-1257 — official TS-AUC 0.6290. Promoted 2026-09-02.**

RT-1257 is the formal production anchor. It is no longer the external champion —
RT-1320 took that on 2026-09-03 at 0.6303 — and moving the anchor to RT-1320 is a
separate explicit action that has not been taken. The
seven-item promotion gate in `engineering/reports/rt1257_deployment/CONSOLIDATION_HYGIENE_REVIEW.md`
is complete: items 2-6 closed on 2026-08-31, item 1 (clean rebuild) on 2026-09-02,
and item 7 — this status change and the `rt1257-production-0.6290` tag — is the
owner action recorded here.

The evidence that closed item 1 (`RT1257_REBUILD_2026.json`): a clean rebuild at
`f1912b6` with `embedded_source_clean=true`, whose `model_zip_sha256`,
`model_manifest_sha256` and `feature_manifest_sha256` all match the recorded
`e50098a4` build. Predictions were compared against a shipped-behaviour proxy
rebuilt from clean `e50098a4`, verified byte-identical to the recorded shipped
source zip: **0 changed out of 89,706 predictions** across two independent
samples (0/38,723 on the 67-series packaged battery, 0/50,983 on the 100-series
Crunch reduced set). That matters because the intervening `src/sbr` changes had
moved 1 prediction in 38,723 for RT-600's release candidate; they move none of
RT-1257's, measured on the same sample where RT-600's change was found.

**RT-600 — official TS-AUC 0.6268 — is retained as `REFERENCE`**, the permanent
homogeneous-LightGBM comparison model, at tag `rt600-production-0.6268` on branch
`production/rt600`. It is no longer the formal anchor.

**Correction, 2026-09-01.** This section previously said the intervening tracked changes were "packaging, dependency-declaration, and engineering-evidence changes rather than `src/sbr` model/feature changes". That is **not accurate**. Between the recorded RT-1257 build (`e50098a4`) and `f1912b6`, `src/sbr` changed by 34 files and ~843 non-comment lines, including the stream modules `s_m03_dyn`, `s_m04_resid`, `s_m07_bayes` and `s_m12_rdep`. The changes predate the 2026-09-01 merge, which did not touch `src/sbr`.

The changes are **legitimate and documented**, not a defect: `engineering/reports/rt600_final_reliability/release_candidate/PREDICTION_CHANGE_ROOTCAUSE.json` root-causes them to the `m06_loc` scalar-square ULP defect, measures the effect at **1 changed prediction in 38,723**, and finds the rebuilt artifact agrees with both the batch reference and the feature cache the frozen model was trained on while the shipped artifact does not — i.e. the change moves served features *toward* trained-on semantics.

Two consequences worth stating plainly:

- The unchanged RT-600 feature-manifest SHA (`1646c3b9…`) is **not** evidence that predictions are unchanged. It covers the declared column set, not the numerical implementation, and it did not move across this diff.
- Promotion-gate item 1 in `CONSOLIDATION_HYGIENE_REVIEW.md` — "build RT-1257 from a clean tracked checkout using the canonical build script" — remains **outstanding**, and it is precisely the step that would quantify this for RT-1257 rather than for RT-600's release candidate. Item 7 says "only then", so RT-1257 cannot be formally promoted on the current record.

See `engineering/reports/rt1257_deployment/CONSOLIDATION_HYGIENE_REVIEW.md`.

## Current research-alive models

The residual CatBoost slot lane (**RT-1260 / CAT-411**, **RT-1261 / CAT-412**, **RT-1262 / CAT-414**, **RT-1263 / CAT-415**) is **closed as of 2026-09-01** — all four are `PARKED / NO_PROMOTION`. See "Closed lanes" below. RT-1320 is no longer research-alive: it cleared its `external_score` blocker on 2026-09-03 and is the external champion. **There is now no research-alive model.**
1. **RT-1320 / M1 Arm-C residual student** — the first candidate that *adds* an 8th member instead of swapping a slot. Under the `PROTOCOL_CHAMPION_2026.md` addition contract the primary endpoint E2-E1 against a matched added seed clone (`RT-403`) is **+0.001516** at **4/5** positive folds, above the 0.0011 paired-bootstrap noise floor; secondary E2-E0 is +0.001416. Fold 0 is negative on both. The widely quoted **5/5** figure belongs to the RT-600 lane (+0.001965 marginal vs the seed-clone blend), **not** to the champion lane. Reproduced at a second seed inside ~1% of the noise floor. **Phase 1's four-partition gate PASSED on 2026-09-01**: mean E2-E1 across canonical/alt1/alt2/alt3 is **+0.0017678** with 4/4 partitions positive and none negative, against a rule frozen before alt1 was run. The leg found the opposite of what it was designed to expose — canonical, recorded in the freeze as the most favourable partition, is the weakest of the four and the only one with a negative fold 0 (1 of 4, so §9's majority criterion is not triggered). The alternate-partition leg, the causality gate and the final-10k fit are now all done, and the 8-member artifact is built and causality-verified. **RESOLVED 2026-09-03: submission #19 scored 0.6303**, beating RT-1257's 0.6290 by +0.0013 against a predicted E2-E0 of +0.001416. The `external_score` blocker is cleared and RT-1320 is the external champion; its student is an ACTIVE_COMPONENT. See `reports/rt1320_promotion/PHASE1_STAGE2.md` and `MODEL_REGISTRY.md`.

## Closed lanes

**Residual CatBoost slots — closed 2026-09-01.** `RT1257_SLOT_ADJUDICATION.md` already directed "do not open new training lanes for CAT-411, CAT-414, or CAT-415 from these frozen OOF results", leaving CAT-412 as the only open question and marking even that optional: a follow-up was to be run "only if a preregistered alt-partition adjudication is desired despite the sub-noise RT-1257 lift".

That alt-partition leg was scoped and costed on 2026-09-01 and **declined**. It would have needed partition-awareness added to `deep_ensemble_local_2026.py` (which trains CAT-412 and has none) and to `rt1257_slot_adjudication.py`, plus a single-specialist train path — roughly a day of plumbing before any compute — to test a candidate whose deployment endpoint is already **+0.000338234 against a 0.0011 noise floor**. Declining is permitted by the adjudication's own wording; the leg was never mandatory, and unlike RT-1320's Phase 1 Stage 2 there is no preregistered clause making an early stop invalid. The leg could only have rescued a sub-floor result, and nothing obliges funding a rescue.

If it is ever revived, the requirement is unchanged: the same RT-1257-relative `E0/E1/E2` contract on alt partitions, with a deployment endpoint above the noise floor before the champion changes.

## Parked model

**RT-995 / T2** — causal teacher-distilled LightGBM. It demonstrated substantial standalone teacher signal but only small marginal ensemble alpha over a matched seed clone. Preserve it as a scientific success about privileged-information transfer, but do not fund ordinary additive promotion work without a new complementarity mechanism.

## Important corrected classifications

- **RT-1264** — **SUPERSEDED**, not research-alive. Its original five-slot best-k conclusion used an E2-E1 selection endpoint whose clone control worsened with k.
- **RT-1265 / CSA-04R** — corrected reanalysis. Fixed E2-E0 + preregistered parsimony selects k*=2, CAT-413 + CAT-300, exactly RT-1257; verdict NOT_DISTINGUISHABLE; no new OOF vector.
- **RT-1250 TabM / RT-1252 RealMLP** — INFEASIBLE original Learner Diversity arms; no binding predictive verdict.
- **RT-1258 TabM / RT-1259 RealMLP** — distinct fresh GPU-authorized arms after hardware removed the feasibility blocker; both completed five folds and are KILL.
- **RT-1256 / CAT-410** — KILL.

## Canonical research records

Read these first:

- `MODEL_REGISTRY.md` — current model set and classifications.
- `INDEX.md` — navigation across the research program.
- `RESULTS.csv` — quantitative ledger.
- `EXPERIMENT_ID_MAP.md` — RT-ID mapping and provenance.
- `NEGATIVE_RESULTS_INDEX.md` — concise failed-idea lookup.
- `FAILED_EXPERIMENTS.md` — detailed negative evidence.
- `LESSONS_LEARNED.md` — cross-program synthesis.
- `RDOF_LEDGER.md` — degrees-of-freedom accounting.
- `PROTOCOL.md` — historical binding protocol.

## Ledger consolidation finding

The pre-consolidation `research/current` branch was not a complete canonical ledger: its `RESULTS.csv` omitted the later RT-1250+ champion lineage. The consolidation imported the later CatBoost / Deep Ensemble generation containing RT-1250..RT-1257 and RT-1260..RT-1265. A follow-up then filed the binding RT-1258/RT-1259 H4 results into canonical `RESULTS.csv` without inventing missing metadata.

PR #14 follow-up bookkeeping filed RT-1258/RT-1259 as append-only rows in canonical `RESULTS.csv` using recovered metadata from the reachable H4 JSON at commit `3f94d55`: `git_sha=b3a16bc`, `train_series=8000`, and both per-fold standalone TS-AUC vectors. The `persistence` field remains blank because no source artifact defines a persistence tag/category for these GPU arms.

The exact pre-consolidation `research/current` ledger is preserved at `archive/2026-08-30/RESULTS_SNAPSHOT.csv`.

## Current research standard

A serious new candidate must ultimately answer:

**Does it add complementary information to RT-1257?**

Use RT-1257 as the primary ensemble baseline and RT-600 as the permanent homogeneous LightGBM reference. Prefer matched-control ensemble marginal tests over standalone AUC, and inspect fold consistency, pair flow, dominant-cell repair, mature-vs-never behavior, causality, runtime, and deployment complexity before promotion.

## Consolidation state

PR #14 was merged and `main` now carries the consolidated tree. The 2026-08-31 branch/tag pruning audit is recorded in `archive/2026-08-31/BRANCH_TAG_PRUNING_AUDIT.md`. No historical branch should be deleted unless that audit classifies it as reachable/tag-protected, or a later owner-approved tag/import decision preserves it first.

_Last updated: 2026-09-03. **RT-1320 is the external champion at 0.6303** (submission #19), +0.0013 over RT-1257 against a predicted +0.001416 — the second consecutive external point matching its internal estimate to ~0.0001. Its `external_score` blocker is cleared; the formal production anchor is still RT-1257 pending an explicit promotion. The residual CatBoost slot lane (CAT-411 / 412 / 414 / 415) remains closed, all four PARKED / NO_PROMOTION._
