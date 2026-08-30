# Main Consolidation Report — 2026

Date: 2026-08-30
Branch: `release/2026-research-consolidation`
Target: `main` only after all required gates are evidenced.

## Executive summary

The repository has entered research-to-production consolidation. The purpose is to make `main` the single authoritative view of the 2026 Real-Time research program without destroying the historical evidence that produced it.

`main` has **not** been moved by this consolidation at the time of this report.

The current best external model is **RT-1257**, official TS-AUC **0.6290**, versus the RT-600 reference at **0.6268** (+0.0022). RT-1257 is not yet being called the formal production anchor because its tracked local Crunch-test record predates the final packaging build.

The consolidation has also corrected two important scientific/provenance misunderstandings:

1. RT-1264 is not a live five-slot upside candidate. CSA-04R / RT-1265 supersedes its current-selection conclusion and selects the two-slot RT-1257 composition under the corrected fixed E2-E0 endpoint and parsimony/noise rule.
2. RT-1250/RT-1252 and RT-1258/RT-1259 are not two independent scored attempts. The first pair are INFEASIBLE original Learner Diversity arms. The second pair are distinct freshly preregistered GPU arms after new hardware removed the feasibility blocker; those later arms completed and received the binding KILL verdicts.

## 1. Pre-consolidation preservation

### PASS — consolidation branch created from main

`release/2026-research-consolidation` was created directly from `main` at `cef5458da2d51fbb8513ad71177fca426616ee28` before importing advanced research.

### PASS — immutable pre-consolidation snapshot created

Commit `afd3553ef4c6831458c5220c9b42ee932d47239a` added:

- `research/archive/2026-08-30/BRANCH_SNAPSHOT.md`
- `research/archive/2026-08-30/BRANCH_HEADS.csv`
- `research/archive/2026-08-30/MODEL_SNAPSHOT.md`
- `research/archive/2026-08-30/RESULTS_SNAPSHOT.csv`

The results snapshot points to the exact pre-consolidation `research/current` RESULTS blob, preserving the before-state byte-for-byte.

### PASS — branch/tag inventory performed before deletion

32 pre-existing remote branch heads and 16 existing milestone tags were inventoried. No branch was deleted, rebased, force-pushed, or moved during the audit.

One pre-existing open PR was found: #12, the August-24 repository-cleanup PR from `chore/repo-cleanup-2026` to `main`. It predates the RT-1257/Deep Ensemble work and was left untouched during the initial inventory.

## 2. Controlled lineage import

### PASS — canonical RT-1257 lineage imported without moving main

The RT-1257 engineering lineage is a strict descendant of pre-consolidation `main`. It was imported into the consolidation branch through PR #13 and merge commit `710d84a6211fb5e1c8365b2a2019c7bd6444cb19`.

This brought forward the mature `src/sbr` pipeline, RT-600 reliability work, Learner Diversity, CatBoost specialist activation, and RT-1257 deployment record.

### PASS — diverged Deep Ensemble histories preserved through curated multi-parent import

The Deep Ensemble LOCAL lane diverged from the RT-1257 deployment lane after the CatBoost specialist commit. Rather than blindly merging the complete working trees, commit `b25ca755394bc16c786a9f64c53cdb7348abfcf1` curated the unique scientific assets into the consolidation tree and records both the LOCAL and CRUNCH heads as additional parents.

Imported evidence includes:

- RT-1260..RT-1265 ledger generation
- CSA-04 original evidence
- CSA-04R preregistration and corrected reanalysis
- GPU full-OOF preregistration/config
- H4 TabM/RealMLP binding result
- minimal reproduction scripts

This makes the sibling scientific histories reachable without replacing the RT-1257 deployment tree wholesale.

## 3. Independent-review finding: RT-1264 / RT-1265

### PASS — finding verified, with corrected interpretation

`research/reports/deep_ensemble_frontier_2026/local/CSA04R_REANALYSIS.md` establishes:

- corrected selection endpoint: fixed E2-E0
- corrected order begins CAT-413, CAT-300, CAT-412
- k=2: CAT-413 + CAT-300, E2-E0 +0.002026322, identical to RT-1257
- k=3: adds CAT-412, E2-E0 +0.002364556
- incremental edge over RT-1257: +0.000338234
- paired-bootstrap noise floor: 0.0011
- preregistered parsimony therefore selects k*=2
- verdict: NOT_DISTINGUISHABLE

The descriptive 63-subset appendix explicitly cannot select a champion or support a promotion claim.

Canonical conclusion:

- RT-1264 historical row: preserve unchanged.
- Current status of RT-1264: **SUPERSEDED BY RT-1265**.
- RT-1265: corrected selection analysis, identical composition to RT-1257, no new OOF vector.
- CAT-412 / RT-1261: strongest remaining additional single-slot research question.

## 4. Independent-review finding: stale canonical RESULTS ledger

### PASS — core defect verified

The pre-consolidation `research/current/research/RESULTS.csv` contains no RT-1250 row and therefore does not contain the later RT-1257 champion lineage.

### VERIFIED NUANCE — sibling files are not five identical complete RT-1250..RT-1265 ledgers

The second review's broader conclusion (the canonical trunk ledger is stale) is correct, but the exact branch-file description was too strong.

Observed ledger generations:

- `research/catboost-specialist-2026`, `research/deep-ensemble-frontier-crunch-2026`, and `research/multi-agent-frontier-20260829` share the same later ledger blob for the RT-1250/1251/1252 and RT-1254..RT-1257 generation.
- `research/deep-ensemble-frontier-local-2026` has a later ledger generation that adds RT-1260..RT-1265.
- the binding RT-1258/RT-1259 H4 result is documented in reports/commit history but no RT-1258/RT-1259 result rows were found in the inspected branch-head RESULTS ledgers.

The consolidation therefore imported the latest verified Deep Ensemble LOCAL ledger generation rather than silently treating `research/current` as canonical.

### OPEN — RT-1258/RT-1259 binding rows still need canonical RESULTS filing

`H4_GPU_TABULAR_RESULT.md` explicitly instructed the LOCAL lane to file RT-1258 and RT-1259 after the binding run. Those rows were not found at the inspected branch heads.

The numerical result is fully documented:

- RT-1258 TabM: marginal_vs_clone +0.000040946994511; 3/5 folds positive; E2-E0 -0.0007899234; standalone mean 0.6006475014; KILL.
- RT-1259 RealMLP: marginal_vs_clone -0.003223743533063; 0/5 positive; standalone mean 0.5599057458; KILL.

This consolidation will not invent missing ledger metadata merely to make the file look complete. Until the rows are filed with a documented provenance convention, this gate remains **OPEN** and is called out explicitly rather than silently repaired.

## 5. Independent-review finding: TabM / RealMLP ID pairs

### PASS — verified, but the proposed interpretation required correction

The authoritative `FULL_OOF_PREREG.md` states that RT-1258/RT-1259 are a **distinct authorization** from RT-1250/RT-1252.

- RT-1250 TabM and RT-1252 RealMLP: original Learner Diversity arms. They were recorded INFEASIBLE before predictive scoring because the frozen installed/default full-scale execution did not fit the authorized compute budget and shrinking/retuning was forbidden.
- An RTX 4090 benchmark later demonstrated that the same intended official/default-scale configurations fit the competition budget without shrinking.
- RT-1258 TabM and RT-1259 RealMLP were therefore freshly allocated and preregistered rather than reopening RT-1250/RT-1252.
- The later H4 run describes itself as a technical recovery of the already-existing **RT-1258/RT-1259 FULL_OOF_PREREG arms**, not as a new third set of arms.

Canonical wording must therefore avoid both mistakes:

1. do not say TabM/RealMLP were independently tested and killed twice;
2. do not say RT-1258/RT-1259 are merely alternate IDs for RT-1250/RT-1252.

The correct statement is: first pair = infeasible original authorization; second pair = distinct GPU re-authorization after the hardware blocker was removed; second pair = binding scored KILL results.

## 6. Independent-review finding: CI dependencies

### DISPROVED AS STATED — dependency import gap not present in the current advanced tree

The current workflow installs `requirements-dev.txt`; that file includes `requirements.txt`. The relevant runtime dependency declarations are therefore pulled into the CI environment rather than being omitted merely because `requirements-research.txt` is not installed explicitly.

A real GitHub Actions run triggered by consolidation PR #13 provides execution evidence:

- dependency installation: **PASS**
- Python checkout/setup: **PASS**
- Ruff: **FAIL**
- pytest: **SKIPPED** because Ruff failed first

Therefore the feared LightGBM/Numba/PyArrow import-install failure is not the current blocker.

### OPEN — full CI still not green

The actual blocker is repository-wide `ruff check .` against a tree containing large historical research surfaces. The consolidation workflow has been changed to:

- verify `lightgbm`, `numba`, `pyarrow`, `catboost`, and `sbr` imports explicitly;
- lint maintained package/test/script surfaces (`src tests scripts`) rather than treating archived research evidence as production lint debt;
- run `research/scripts/check_research_hygiene.py` separately;
- then run pytest.

This revised workflow must execute on GitHub Actions before any CI PASS claim is made.

## 7. RT-1257 deployment hygiene

### PASS — external score provenance

Submission #16 is recorded at 0.6290 official TS-AUC, versus RT-600 0.6268.

### PASS — manifest/build tracked-code difference classified

`submissions/RT1257_deployable.build.json` records:

- manifest code SHA `6b4fafacd30711bebf2bf977c3c390641f3ae522`
- build code SHA `e50098a42f28b1007746cb20bbabc157e285b036`
- `manifest_matches_build_sha: false`

A direct Git comparison shows three tracked commits and no `src/sbr` model/feature/calibration/inference changes. The changed tracked files are deployment evidence, dependency declaration, build-submission packaging logic, and generated build metadata. The source-code gap is therefore verified as packaging/deployment-only.

### UNVERIFIED — fresh post-build Crunch test

The tracked `engineering/reports/rt1257_deployment/CRUNCH_TEST.json` predates the final build and lacks the final entrypoint hash. It cannot be used as evidence that the exact shipped/post-packaging build passed the local Crunch test.

### BLOCKED — formal production promotion

Because the fresh artifact test is missing, no `rt1257-production-0.6290` production tag has been created and `STATUS.md` retains RT-600 as the formal production anchor.

## 8. Canonical current-state documents

### CREATED

- `research/MODEL_REGISTRY.md`
- `research/INDEX.md`
- `research/NEGATIVE_RESULTS_INDEX.md`
- `research/LESSONS_LEARNED.md`
- `research/PROTOCOL_CHAMPION_2026.md`
- `engineering/reports/rt1257_deployment/CONSOLIDATION_HYGIENE_REVIEW.md`

### REWRITTEN FOR CURRENT STATE

- `research/STATUS.md`
- root `README.md`

The new documents explicitly distinguish EXTERNAL_CHAMPION, REFERENCE, ACTIVE_COMPONENT, RESEARCH_ALIVE, PARKED, SUPERSEDED, KILL, INFEASIBLE, DIAGNOSTIC, ORACLE_NONCAUSAL, and INVALID.

## 9. Current model classification

| Model | Consolidated status |
| --- | --- |
| RT-1257 | EXTERNAL_CHAMPION; production promotion blocked pending fresh Crunch test |
| RT-600 | REFERENCE / FORMAL_PRODUCTION_ANCHOR |
| RT-1254 / CAT-413 | ACTIVE_COMPONENT |
| RT-1255 / CAT-300 | ACTIVE_COMPONENT |
| RT-1261 / CAT-412 | RESEARCH_ALIVE; strongest residual slot question |
| RT-1263 / CAT-415 | RESEARCH_ALIVE |
| RT-1262 / CAT-414 | RESEARCH_ALIVE |
| RT-1260 / CAT-411 | RESEARCH_ALIVE |
| RT-995 / T2 | PARKED |
| RT-1265 | SUPERSEDING_ANALYSIS / NOT_DISTINGUISHABLE; exact RT-1257 composition |
| RT-1264 | SUPERSEDED BY RT-1265 |
| RT-1256 / CAT-410 | KILL |
| RT-1250 TabM | INFEASIBLE |
| RT-1252 RealMLP | INFEASIBLE |
| RT-1258 TabM | KILL |
| RT-1259 RealMLP | KILL |

## 10. Branch/tag hygiene

### PASS — no premature branch deletion

No historical branch has been deleted during consolidation.

### PASS — important diverged Deep Ensemble histories are reachable from the consolidation commit graph

The curated import records the LOCAL and CRUNCH heads as parents, so their histories remain reachable from the consolidation lineage.

### OPEN — final milestone tagging / branch pruning

Historical branch deletion is intentionally deferred until:

1. the consolidation validation gates pass;
2. all required milestone tags exist;
3. the final PR to main is ready and reviewed;
4. each candidate branch is rechecked for unique unreachable history.

Do not delete branches merely because their science is classified KILL.

## 11. Merge-gate matrix

| Check | Status | Evidence / note |
| --- | --- | --- |
| Branch heads captured before cleanup | **PASS** | snapshot commit `afd3553` |
| Existing milestone tags inventoried | **PASS** | pre-consolidation snapshot |
| No force-push/history rewrite | **PASS** | consolidation uses ordinary branch/merge/commit operations |
| `main` unchanged during research import | **PASS** | work isolated to release branch |
| Pre-consolidation RESULTS preserved | **PASS** | byte-identical blob in archive snapshot |
| RT-1250+ stale-trunk ledger defect identified | **PASS** | `research/current` lacks RT-1250 |
| Later CatBoost / RT-1260..1265 ledger generation imported | **PASS** | curated Deep Ensemble import |
| RT-1258/RT-1259 binding results documented | **PASS** | H4 result imported |
| RT-1258/RT-1259 rows filed in canonical RESULTS.csv | **OPEN** | rows absent at inspected branch heads; do not invent metadata |
| RT-1264 corrected classification | **PASS** | CSA-04R / RT-1265 evidence |
| TabM/RealMLP ID relationship resolved | **PASS** | FULL_OOF_PREREG + H4 evidence |
| MODEL_REGISTRY created | **PASS** | current-state classifications |
| Research INDEX created | **PASS** | canonical navigation |
| Negative-results index created | **PASS** | failed/infeasible/superseded distinctions |
| Lessons learned created | **PASS** | cross-program synthesis |
| README rewritten around actual 2026 system | **PASS** | RT-1257 visible as external champion |
| STATUS rewritten | **PASS** | formal-anchor nuance retained |
| RT-600 preserved | **PASS** | reference/frozen lineage retained |
| RT-1257 external provenance | **PASS** | submission #16 record |
| Manifest/build tracked-code gap classified | **PASS** | direct Git compare, packaging-only tracked diff |
| Fresh post-build RT-1257 Crunch test | **UNVERIFIED** | tracked test predates final build |
| RT-1257 formal production promotion | **BLOCKED** | depends on fresh test and final validation |
| CI dependency installation | **PASS** | actual PR #13 GitHub Actions install step |
| CI runtime import smoke test | **PENDING RUN** | revised workflow authored, not yet executed |
| Maintained-code Ruff | **PENDING RUN** | revised scope authored |
| Research ledger hygiene script | **PENDING RUN** | revised workflow authored |
| pytest full intended suite | **UNVERIFIED** | prior run skipped at Ruff |
| production causality/prefix/independence/determinism gate | **PARTIAL / NEEDS FINAL RUN** | historical evidence exists; consolidation CI/clean artifact confirmation pending |
| branch tags created for new late milestones | **OPEN** | defer until validation / available safe tag operation |
| branch deletion | **NOT STARTED BY DESIGN** | must occur only after final merge and reachability check |

## 12. Items that block moving main

The consolidation branch must not be merged to `main` while any required item below is unresolved:

1. Revised GitHub CI has not yet executed to completion.
2. pytest has not yet run under the consolidated GitHub environment.
3. RT-1258/RT-1259 binding rows are still absent from canonical `RESULTS.csv` and need a provenance-safe filing operation.
4. RT-1257 still lacks a fresh tracked Crunch-test record tied to the final/post-packaging build and entrypoint hash.
5. Formal RT-1257 production promotion/tagging therefore remains blocked.
6. Final branch/tag pruning must wait until after validation and a last unique-history audit.

## 13. Scientific operating state after consolidation

The intended hierarchy is now explicit:

- external champion: RT-1257, 0.6290
- formal production/reference anchor pending final RT-1257 artifact gate: RT-600, 0.6268
- champion components: CAT-300 / RT-1255 and CAT-413 / RT-1254
- research-alive residual CatBoost slots: CAT-412, CAT-415, CAT-414, CAT-411
- parked signal: RT-995 / T2
- corrected hybrid selection: RT-1265 = RT-1257 composition
- superseded selection conclusion: RT-1264
- binding neural KILLs: RT-1258 TabM, RT-1259 RealMLP

Future serious research should optimize **marginal information relative to RT-1257**, not standalone AUC, while retaining RT-600 as the homogeneous LightGBM reference.

## 14. Integrity principle

No PASS in this report means “probably.” PASS is used only where repository/GitHub evidence was directly checked. Items that require a fresh local competition environment, unavailable generated artifacts, or an actual CI run remain UNVERIFIED, PENDING, OPEN, or BLOCKED.
