# Main Consolidation Report — 2026

Date: 2026-08-31
Branch: `release/2026-research-consolidation`
Target: `main` only after all required gates are evidenced.

## Executive summary

The repository has entered research-to-production consolidation. The purpose is to make `main` the single authoritative view of the 2026 Real-Time research program without destroying the historical evidence that produced it.

`main` has **not** been moved by this consolidation at the time of this report.

The current best external model is **RT-1257**, official TS-AUC **0.6290**, versus the RT-600 reference at **0.6268** (+0.0022). A fresh local post-build Crunch test now passes against the final entrypoint hash, but RT-1257 is not yet being called the formal production anchor until the latest consolidation head passes GitHub CI and the owner promotion/tag step is made deliberately.

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

### PASS — RT-1258/RT-1259 binding rows filed in canonical RESULTS.csv

`H4_GPU_TABULAR_RESULT.md` explicitly instructed the LOCAL lane to file RT-1258 and RT-1259 after the binding run. Those rows were not found at the inspected branch heads, so PR #14 follow-up appended them to canonical `research/RESULTS.csv` after `RT-1265`.

The numerical result is fully documented:

- RT-1258 TabM: marginal_vs_clone +0.000040946994511; 3/5 folds positive; E2-E0 -0.0007899234; standalone mean 0.6006475014; KILL.
- RT-1259 RealMLP: marginal_vs_clone -0.003223743533063; 0/5 positive; standalone mean 0.5599057458; KILL.

The filed rows use recovered metadata from `3f94d55:research/reports/deep_ensemble_frontier_2026/crunch/H4_GPU_TABULAR_RESULT.json`: `git_sha=b3a16bc`, `train_series=8000`, and the exact five per-fold standalone TS-AUC values. The `persistence` field remains blank because no source artifact defines a persistence tag/category for these GPU arms. Persisted model/OOF artifact hashes were not recovered because the H4 JSON says the authoritative artifacts require Crunch dashboard access.

The follow-up also restored the RT-1200/RT-1201 CSV quoting style to match the historical prefix rather than moving the causal-representation-frontier audit baseline.

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

### LOCAL PASS / GITHUB PENDING — full CI command set

The actual blocker is repository-wide `ruff check .` against a tree containing large historical research surfaces. The consolidation workflow has been changed to:

- verify `lightgbm`, `numba`, `pyarrow`, `catboost`, and `sbr` imports explicitly;
- lint maintained package/test/script surfaces (`src tests scripts`) rather than treating archived research evidence as production lint debt;
- run `research/scripts/check_research_hygiene.py` separately;
- then run pytest.

PR #14 follow-up also corrected the FMA/lfilter parity assumption that made x86_64 Linux CI stricter than macOS/arm64. The stream code now uses `sbr.stream._fp.lfilter_madd`, which follows the active host's `scipy.signal.lfilter` multiply-add contract: single-rounded FMA on the supported arm64 build, ordinary two-rounded `a * b + c` on x86_64 Linux.

Local CI-equivalent validation after the follow-up changes:

- import smoke for `lightgbm`, `numba`, `pyarrow`, `catboost`, and `sbr`: PASS
- `ruff check src tests scripts`: PASS
- `python research/scripts/check_research_hygiene.py`: PASS, 275 experiment rows and no duplicate IDs
- focused parity/audit subset: PASS, 63 passed / 2 skipped
- `pytest -q`: PASS, 692 passed / 79 skipped / 88 warnings

The revised workflow must still execute on GitHub Actions for the pushed PR head before any remote CI PASS claim is made.

## 7. RT-1257 deployment hygiene

### PASS — external score provenance

Submission #16 is recorded at 0.6290 official TS-AUC, versus RT-600 0.6268.

### PASS — manifest/build tracked-code difference classified

`submissions/RT1257_deployable.build.json` records:

- manifest code SHA `6b4fafacd30711bebf2bf977c3c390641f3ae522`
- build code SHA `e50098a42f28b1007746cb20bbabc157e285b036`
- `manifest_matches_build_sha: false`

A direct Git comparison shows three tracked commits and no `src/sbr` model/feature/calibration/inference changes. The changed tracked files are deployment evidence, dependency declaration, build-submission packaging logic, and generated build metadata. The source-code gap is therefore verified as packaging/deployment-only.

### PASS — fresh post-build Crunch test

PR #14 follow-up ran a fresh local Crunch test against the shipped/post-packaging RT-1257 entrypoint and regenerated `engineering/reports/rt1257_deployment/CRUNCH_TEST.json`.

Evidence captured:

- tested_at_utc: 2026-08-31T01:07:46Z
- Crunch CLI: 11.11.0
- entrypoint SHA-256: `05eafb66f4589f5b426f4af43779f428f7a90f2d64b423530d6f315a798e888b`
- build manifest SHA-256: `8e27b96c0b7cb4ac6998ff89c37ac180cf54a5253a3141d2e096e52feac6b344`
- result: PASSED
- determinism_check: passed
- prediction rows: 50,983
- prediction SHA-256: `cd6c5fe9f376ef4c5e620e77f5f94cf4bb77843d1e12820a02256e08e7a80b1b`

### PENDING — formal production promotion

No `rt1257-production-0.6290` production tag has been created and `STATUS.md` retains RT-600 as the formal production anchor. The fresh local test is now closed; formal promotion still depends on the latest PR head passing GitHub CI and the owner/tag/status promotion step being made explicitly.

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
| RT-1257 | EXTERNAL_CHAMPION; fresh local Crunch test passed; production promotion pending CI/tag/owner gate |
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
| RT-1258/RT-1259 rows filed in canonical RESULTS.csv | **PASS** | append-only follow-up rows; unrecovered metadata left blank |
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
| Fresh post-build RT-1257 Crunch test | **PASS** | `CRUNCH_TEST.json` regenerated from 2026-08-31 local run |
| RT-1257 formal production promotion | **PENDING** | depends on GitHub CI and explicit owner/tag/status promotion |
| CI dependency installation | **PASS** | actual PR #13 GitHub Actions install step |
| CI runtime import smoke test | **LOCAL PASS / GITHUB PENDING** | local command passed after editable install |
| Maintained-code Ruff | **LOCAL PASS / GITHUB PENDING** | `ruff check src tests scripts` |
| Research ledger hygiene script | **PASS** | local script reports 275 rows and no duplicate IDs |
| pytest full intended suite | **LOCAL PASS / GITHUB PENDING** | `pytest -q`: 692 passed, 79 skipped, 88 warnings |
| production causality/prefix/independence/determinism gate | **LOCAL PASS / GITHUB PENDING** | test suite passes locally; real-store CRF tests skip where competition store is absent |
| branch tags created for new late milestones | **OPEN** | defer until validation / available safe tag operation |
| branch deletion | **NOT STARTED BY DESIGN** | must occur only after final merge and reachability check |

## 12. Items that block moving main

The consolidation branch must not be merged to `main` while any required item below is unresolved:

1. Revised GitHub CI for the latest PR head has not yet executed to completion.
2. Formal RT-1257 production promotion/tagging/status change has not been made by the owner; RT-600 remains the formal production anchor until then.
3. Final branch/tag pruning must wait until after validation and a last unique-history audit.

## 13. Scientific operating state after consolidation

The intended hierarchy is now explicit:

- external champion: RT-1257, 0.6290
- formal production/reference anchor pending final RT-1257 promotion: RT-600, 0.6268
- champion components: CAT-300 / RT-1255 and CAT-413 / RT-1254
- research-alive residual CatBoost slots: CAT-412, CAT-415, CAT-414, CAT-411
- parked signal: RT-995 / T2
- corrected hybrid selection: RT-1265 = RT-1257 composition
- superseded selection conclusion: RT-1264
- binding neural KILLs: RT-1258 TabM, RT-1259 RealMLP

Future serious research should optimize **marginal information relative to RT-1257**, not standalone AUC, while retaining RT-600 as the homogeneous LightGBM reference.

## 14. Integrity principle

No PASS in this report means "probably." PASS is used only where repository, local command, or available GitHub evidence was directly checked. Items that require unavailable generated artifacts, a future GitHub Actions run, or an owner promotion step remain UNVERIFIED, PENDING, OPEN, or BLOCKED.
