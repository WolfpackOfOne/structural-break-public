# Evidence Manifest

## Question

> "Identify the three most important unresolved quantitative research questions documented in this repository, based only on current research reports, RESULTS.csv, and recent Git history. Do not propose new training runs."

Investigation: `20260829-173245-unresolved-quant-questions`

---

## Starting Repository State

- Canonical repo: `/path/to/workspace/structural-break/`
- Branch of record: `claude/local-model-results-review-lk0f7i`
- HEAD SHA: `33fa041921f2e29d3b7bfbffcfac565997f9789a` ("Add weekend model-search harness for the real competition data format")
- Worktree: DIRTY (untracked `RESULTS.csv`, `STATE_OF_RESEARCH.md`, `research/`, `wave2/`, `*.bundle`, `*.tar.gz`; modified `experiments/`, `scripts/`, `src/structural_break/`, `tests/`).
- Remote: `origin` -> `https://github.com/WolfpackOfOne/structural-break.git`

### CRITICAL PROVENANCE FINDING (GIT-VERIFIED)

The canonical worktree's `git log` contains **only 16 commits**, all about the baseline Python package (Phases 1-5) plus a "weekend model-search harness". **None of the wave 1-8 research is in the current branch's history.** The research lives in:

1. **Git tags** (present in object DB): `wave2-2026-final`, `multi-agent-2026-final`, `wave3-engineering-rt150`, `wave3-integration-final`, `wave5-alpha-final`, `wave7-t2-final`, `wave8-future-aware-final`, `rt600-production-0.6268`, `wave7-d3r-information-frontier`, `oracle-information-frontier-2026-study`, `reproduce-2025-public-solution-audit`, etc.
2. **Git branches** (61 total incl. remote): `research/current`, `research/wave8-future-aware-distillation`, `production/rt600`, `research/multi-agent-2026`, `codex/*`, `claude/*`, `research/wave5-alpha`, `research/wave6-alpha`, `research/wave7-*`.
3. **Sibling directories are git WORKTREES, not independent repos.** Each `structural-break-*/` has a `.git` **file** (not directory) pointing to `structural-break/.git/worktrees/<name>`. E.g. `structural-break-wave8/.git` -> `gitdir: /path/to/workspace/structural-break/.git/worktrees/structural-break-wave8`. This **contradicts the 00_QUESTION.md premise** that "None have a .git directory". They share the canonical repo's object database.

**Consequence:** the "current research reports" and "recent Git history" evidence base is NOT in the canonical worktree - it is spread across tags, branches, and sibling worktrees. The canonical worktree is a **stale/divergent checkout** that has been reset to a fresh "weekend harness" state.

---

## Highest-Priority Evidence

### E1. `research/current` branch - the canonical active research lineage (GIT-VERIFIED / REPORT-CLAIM)

- **Where:** branch `research/current`; worktree `structural-break-research-current/`; `research/STATUS.md`, `research/STATE_OF_RESEARCH.md`, `research/NEW_AVENUES_2026.md`, `research/RESULTS.csv` (212 rows).
- **Why relevant:** This is the authoritative "current" state. `STATUS.md` (updated 2026-08-24) states:
  - Production anchor = **RT-600** (seven-stream deployable ensemble), external Crunch leaderboard score **0.6268** (LB-001).
  - Best deployable internal = RT-600 seven specialists, pooled OOF ~ **0.63828** (fold-0).
  - Latest major negative: **Wave 8 future-aware transfer family - five pilots (ORR, TGMC, SST, PCFB, CFEP) all KILL** on full population.
  - Latest major positive: W7-D3R same-prefix-vs-full-sequence diagnostic established the future-information limit.
  - Next-wave direction: `NEW_AVENUES_2026.md` - "DESIGN COMPLETE, NOTHING EXECUTED".
- **Confidence:** HIGH for existence; MEDIUM for individual numbers (not independently recomputed here).
- **Limitations:** Numbers are REPORT-CLAIM unless recomputed; `NEW_AVENUES_2026.md` section A.2 claims to have re-derived them but that is itself a report.

### E2. `NEW_AVENUES_2026.md` - the single most explicit open-question statement (REPORT-CLAIM)

- **Where:** `structural-break-research-current/research/NEW_AVENUES_2026.md` (written 2026-08-24, `research/current` @ `a425798`), lines ~196-204.
- **Why relevant:** Verbatim open question: *"W7-D3R proved the information gap is real (+0.071 cell AUC from each series' own future, 5/5 folds) and Wave 8 proved that five different ways of distilling that future into a causal student all fail. The open question Wave 8 leaves is not 'is there information' but 'why does none of it survive contact with the full row population'."*
- **Confidence:** HIGH (direct quote).
- **Limitations:** This is the author's framing; the underlying +0.071 and the five KILLs are REPORT-CLAIM until recomputed from OOF vectors.

### E3. `RESULTS.csv` (canonical worktree, untracked) - 79 rows, ends at RT-131 (ARTIFACT-VERIFIED)

- **Where:** `structural-break/RESULTS.csv` (untracked, mtime 2026-08-18 16:29).
- **Why relevant:** This is the ledger the question names. It contains wave-1 rows RT-000 -> RT-131 only. **It does NOT contain RT-150, RT-160, RT-190, RT-600, or any wave 2-8 rows.**
- **Key rows / clusters:**
  - `RT-000` baseline 0.52051 (handcrafted noisy-OR).
  - `RT-100` champion single model 0.61510 (500 cols, 5-fold). `RT-100-LOCKBOX` 0.60791.
  - `RT-102` **STOPPED EARLY - falsified on fold 0** (context block -0.0177).
  - `RT-130` 4-stream rank average 0.62394, status **NEW CHAMPION**.
  - `RT-131` 7-stream rank average 0.62524, status **NEW CHAMPION** (see contradiction C1).
  - `RT-150_f1` = "DGP-cluster routing" gating experiment 0.60619 (NOT the deployable ensemble - see contradiction C2).
  - `RT-140` feature selection: all-500 = 0.62689 (fold-0 held-out).
- **Provenance flag:** 66 of 78 wave-1 rows have `git_sha = "nogit"` (unattributable).
- **Confidence:** HIGH (read directly).
- **Limitations:** Untracked; not in git history; ends at wave 1.

### E4. `wave2/RESULTS_wave2.csv` - 105 rows, wave-1 + wave-2 (ARTIFACT-VERIFIED)

- **Where:** `structural-break/wave2/RESULTS_wave2.csv` (untracked).
- **Why relevant:** Extends the ledger with wave-2 rows: `RT-100R` (reproduction, delta 0.0), `RT-121R/122R/123R/124R/120R/125R` (stream rebuilds), `RT-200..207` (leave-one-module-out battery), `RT-221/222/223` (alt partitions), `RT-210` (seed 1), `RT-231/232` (champion-protocol confirmations), `RT-2410/2411/2412`, `RT-2420/2421` (random-drop controls).
- **Key finding:** `RT-125R` recorded 0.617411 vs wave-1 `RT-125` 0.613557 (+0.0039), yet `STATE_OF_RESEARCH_V2.md` says "RT-125R could not be reproduced as configured" (LightGBM 4.7.0 rejects `boosting=goss` + bagging). See contradiction C4.
- **Confidence:** HIGH (read directly).
- **Limitations:** Untracked; wave-2 rows carry real SHAs (e98f5b9, 93bdb79, b5ea9d1, 94f77c5, 5101066) but **94f77c5, 5101066, 93bdb79, b5ea9d1 are NOT in the object DB** (verified `git cat-file -t` -> "Not a valid object name"). Only e98f5b9/8e76ad2/23cb4da/65b0e21/46583f5/ea5e2f2 exist.

### E5. `STATE_OF_RESEARCH.md` (canonical worktree, untracked) - describes RT-160/RT-190 (REPORT-CLAIM)

- **Where:** `structural-break/STATE_OF_RESEARCH.md` (mtime 2026-08-19 15:41).
- **Why relevant:** Describes "DEPLOYMENT CHAMPION RT-190" (four-stream logit average, 0.62368) and "RESEARCH CHAMPION RT-160" (seven-stream logit average, 0.62561). These IDs do NOT appear in the canonical `RESULTS.csv`. This doc belongs to the **multi-agent-2026 lineage**, not the wave2 lineage.
- **Confidence:** HIGH (read directly).
- **Limitations:** Untracked; describes experiments absent from the ledger; contradicts wave2's RT-150 champion (see C2/C3).

### E6. `wave2/STATE_OF_RESEARCH_V2.md` - RT-150 deployable champion (REPORT-CLAIM)

- **Where:** `structural-break/wave2/STATE_OF_RESEARCH_V2.md`.
- **Why relevant:** Declares **RT-150** (7 streams + frozen cross-fitted time-conditional CDF) = **0.62589** deployable OOF, +0.01057 over RT-100R (bootstrap CI [+0.00763,+0.01320], 200/200). Declares RT-131 an **ORACLE/DIAGNOSTIC** ("must never be called the champion"). Documents the noise hierarchy: fold-to-fold SD 0.0085, partition-draw SD 0.0050, seed SD 0.0012. Documents the leave-one-module-out confound (random-drop control).
- **Confidence:** HIGH (read directly).
- **Limitations:** REPORT-CLAIM; the submission notebook and OOF vectors it references are uncommitted/missing (see Missing Evidence).

### E7. `HANDOFF_WAVE3.md` + `BRIEF_*_wave3_*.md` - the fork and the ID collision (REPORT-CLAIM)

- **Where:** `structural-break/HANDOFF_WAVE3.md`, `BRIEF_CLAUDE_wave3_alpha.md`, `BRIEF_CODEX_wave3_engineering.md` (also duplicated in `research/`).
- **Why relevant:** Explicitly documents: (a) **two divergent branches both claim "RT-150"** - `research/wave2-2026` (RT-150 = deployable CDF ensemble 0.62589) vs `research/multi-agent-2026` (RT-150 = a REJECTED gating experiment -0.0219; RT-160/RT-190 = logit-average blends 0.62544/0.62368). (b) The submission notebook does not exist (gitignored). (c) `crunch test` never run (egress blocked). (d) Only 1 of 25 wave-3 alpha teams executed (W3-A1 `m08_chan`, REJECTED -0.00073).
- **Confidence:** HIGH (read directly).
- **Limitations:** REPORT-CLAIM; the referenced branch `research/wave2-2026` does NOT appear in the current branch list (only `research/multi-agent-2026` does), though the tag `wave2-2026-final` exists.

### E8. `experiments/leaderboard.csv` + `experiments/progress.md` - the HEAD "weekend harness" (ARTIFACT-VERIFIED)

- **Where:** `structural-break/experiments/leaderboard.csv` (143 rows), `experiments/progress.md`, `experiments/best_config.json`.
- **Why relevant:** This is what the canonical HEAD (33fa041) actually produced - a **completely different, metric-incompatible research thread**. 142 experiments, best = `seed_stability_check__feature_set=stat,split_seed=7` AUC **0.598843**; best fixed-split = `rf_stat_features` **0.579896**. These are **series-level AUC on a single 80/20 split** (not TS-AUC, not 5-fold), using statistical before/after features (not the 500-column causal bank). Scores (0.48-0.60) are far below the wave-1 champion (0.61510).
- **Confidence:** HIGH (read directly).
- **Limitations:** This thread appears unaware of / disconnected from waves 1-8. It is a fresh restart, not a continuation. See contradiction C5.

---

## Relevant Experiment Ledger Entries (statuses / flags)

| ID | status/note | signal | why it matters |
|---|---|---|---|
| RT-102 | STOPPED EARLY - falsified on fold 0 | context block -0.0177 | screen +0.0009 -> full -0.0177 reversal |
| RT-111 | recorded | pairwise -0.0033 fold-0, later dead heat | "one fold is not a result" |
| RT-130 | NEW CHAMPION | 4-stream rank avg 0.62394 | rank avg later ruled illegal |
| RT-131 | NEW CHAMPION | 7-stream rank avg 0.62524 | **illegal oracle** (C1) |
| RT-150_f1 | recorded | gating 0.60619 | ID collision with deployable RT-150 (C2) |
| RT-900 | **VOID** | oracle 0.86552 | label leak via missingness mask; "give the model tau is degenerate" |
| RT-991 | recorded (OFFLINE DIAGNOSTIC) | final-row broadcast 0.72003 | illegal future-info ceiling |
| RT-992/993 | pilot_fold0_only | teacher distillation 0.64277/0.64327 | wave-7 teacher pilot |
| RT-1007/1030/1032 | pilot_fold0_only (OFFLINE) | SST/PCFB oracles 0.63084/0.64697/0.63786 | wave-8 future-aware oracles |
| RT-1006/1031/1033/1042 | pilot_fold0_only | legal future-aware students 0.62326/0.62287/0.62438/0.62166 | wave-8 legal distillations - all ~ RT-600, no gain |
| RT-1020/1021 | pilot_fold0_only | ORR repair blend 0.63823 | wave-8 repair-propensity |
| RT-960/961/970/971 | recorded | MLP 0.5706/0.5806, TCN 0.5415/0.5432 | wave-6 neural learners - far below trees |
| RT-710/711/712 | recorded | hard-negative 0.61472 / 0.60770 / 0.60132 | curriculum path rejected |

---

## Relevant Source Code

- `src/structural_break/` (canonical worktree): `competition_data.py`, `experiment_registry.py`, `experiment_runner.py`, `series_features.py`, `synthetic.py` - the **weekend harness** code (modified, dirty). This is the HEAD lineage, NOT the wave research.
- `src/sbr/` (referenced by wave docs, e.g. `store.py`, `metric.py`, `nullcal.py`, `transforms.py`, `pipeline.py`, `features/`, `stream/`, `production/`) - **NOT present in the canonical worktree.** Lives on `research/wave2-2026` / `research/current` branches and in sibling worktrees (`structural-break-wave8/src/`, etc.).
- `scripts/run_experiments.py` (canonical, dirty) - weekend harness driver.
- `scripts/wave8_common.py`, `scripts/novel_streams/` - on `research/current` (referenced by NEW_AVENUES_2026.md).

---

## Relevant Evaluation Code

- `sbr/metric.py` - official TS-AUC implementation (referenced; not in canonical worktree). `platform_constraints.md` claims metric parity confirmed: `TS-AUC = sum_t w(t)*AUC(t)/sum_t w(t)`, `w(t)=n_pos(t)*n_neg(t)`.
- `tests/test_sbr_metric.py`, `tests/test_streaming.py`, `tests/test_neural_causality.py`, `tests/test_n_online_is_never_a_feature` - referenced in wave docs; present in sibling worktrees, not canonical.

---

## Relevant Prediction Artifacts

- `wave2/models_rt100_stream.tar.gz`, `wave2/models_rt150_ensemble.tar.gz` (untracked, canonical worktree) - model artifacts.
- `submissions/A_rt100_streaming.ipynb` (canonical, untracked) - the only submission notebook present.
- `submissions/C_ensemble_deployable.ipynb` - **MISSING** (gitignored build artifact; referenced as PASS at 4.57 ms/point but unverifiable).
- OOF `.npy` vectors for the 7 streams - **MISSING/uncommitted** (referenced by wave2 and HANDOFF).
- `research/folds/folds.parquet` (SHA256 `6e114f80...`) + alt1/alt2/alt3 - present in `wave2-2026-final` tag tree, NOT in canonical worktree.
- `research/reports/*.json` (champion_diagnostics, deployable_ensemble, lockbox_confirmation, negative_controls, etc.) - present in `wave2-2026-final` tag tree, NOT in canonical worktree.

---

## Relevant Reports (inventory)

**Canonical worktree (untracked):**
- `STATE_OF_RESEARCH.md` (multi-agent lineage, RT-160/RT-190)
- `20260818_claude_update_1.md` (wave-1 session summary, 78 experiments)
- `HANDOFF_WAVE3.md`, `BRIEF_CLAUDE_wave3_alpha.md`, `BRIEF_CODEX_wave3_engineering.md`
- `platform_constraints.md` (metric/runtime/determinism)
- `deep-research-report (3).md` (public-methods research)
- `wave2/STATE_OF_RESEARCH_V2.md`, `wave2/VALIDATION_V2.md`, `wave2/runner_semantics.md`, `wave2/redteam_generator_artifacts.md`, `wave2/PUBLIC_IDEA_MAP.md`, `wave2/RESULTS_wave2.csv`

**In `wave2-2026-final` tag / sibling worktrees (NOT in canonical worktree):**
- `research/FAILED_EXPERIMENTS.md`, `research/RDOF_LEDGER.md`, `research/VALIDATION_V2.md`, `research/REPRODUCIBILITY_MANIFEST.json`, `research/STATE_OF_RESEARCH_V2/V3/V4/V5.md`, `research/PROTOCOL.md`, `research/README.md`
- `research/reports/` (break_taxonomy, reproducibility_v2, streaming_port, runner_semantics, redteam_generator_artifacts, platform_constraints, agent04-12 reports, wave4/5/6/7 reports)
- `research/WAVE4_STATUS.md`, `WAVE5_PREREG/STATUS.md`, `WAVE6_*`, `WAVE7_*`, `WAVE8_*` (in `structural-break-wave8/research/`)
- `research/EXPERIMENT_ID_MAP.md`, `FINAL_ARCHITECTURE_FREEZE.md`, `FINAL_REPRODUCIBILITY_MANIFEST.json`, `LEADERBOARD_ASSAULT_STATUS.md`, `known_failures.json`

---

## Relevant Git History

- `git log --oneline -40` (canonical HEAD): 16 commits, all baseline package + weekend harness. Latest: `33fa041 Add weekend model-search harness...`, `7564c7b Final checkpoint: 12 experiments this session`, `603f548 Final checkpoint: 4 experiments this session`, then Phases 1-5.
- **The wave research is NOT in this log.** It is reachable via tags/branches:
  - `wave2-2026-final` -> `bfcb232` (full `research/` tree)
  - `multi-agent-2026-final` -> `23cb4da`
  - `wave3-integration-final` -> `17bb5df`
  - `wave5-alpha-final` -> `26b01f6`
  - `wave7-t2-final` -> `5093e0a`
  - `wave8-future-aware-final` -> `589e1db`
  - `rt600-production-0.6268` -> `9aaa9b0`
- Referenced-but-missing commits: `94f77c5`, `5101066`, `93bdb79`, `b5ea9d1` (wave-2 experiment SHAs in RESULTS_wave2.csv) - **not in object DB**.
- Referenced-but-missing branch: `research/wave2-2026` (HANDOFF says "USE THIS", but it is not in `git branch -a`; only `research/multi-agent-2026` is).

---

## Historical Branch Evidence

- `research/multi-agent-2026` (HEAD 23cb4da): the RT-160/RT-190 logit-average lineage; its RT-150 = rejected gating.
- `research/wave8-future-aware-distillation` (tag `wave8-future-aware-final`): five future-aware mechanisms, all KILL.
- `codex/oracle-information-frontier-2026` (tag `oracle-information-frontier-2026-study`): boundary-aware oracle vs RT-300 - legal model matches/beats oracle through h=150 (headroom -0.0058/-0.0019/-0.0025 at h=20/100/150; only FULL horizon +0.0396).
- `codex/reproduce-2025-public-solution` (tag `reproduce-2025-public-solution-audit`): 2025 reproduction **failed its own gate** (deps never installed, all rungs `blocked`).
- `production/rt600` (tag `rt600-production-0.6268`): frozen RT-600, external 0.6268.

---

## Metric Definitions

- **TS-AUC** (Time-Stratified AUC): `sum_t w(t)*AUC(t) / sum_t w(t)`, `w(t) = n_pos(t)*n_neg(t)`. Implemented in `sbr/metric.py`. Confirmed against official docs (platform_constraints.md, 2026-08-19).
- **Series-level AUC** (used by the weekend harness in `experiments/leaderboard.csv`): a DIFFERENT quantity - one label per series, single 80/20 split. **Not comparable to TS-AUC.** This is a metric-discipline hazard (RESEARCH_PROTOCOL section 4).
- **DISCOVERY / CONFIRMATION / DEPLOYABLE / ORACLE-DIAGNOSTIC** labels (VALIDATION_V2.md) - binding vocabulary for score provenance.
- **Noise hierarchy** (wave2): fold-to-fold SD 0.0085, partition-draw SD 0.0050, seed SD 0.0012. "Below +0.0005 is noise; +0.005 potentially meaningful; +0.010 material."

---

## Known Contradictions

**C1 - RT-131 "NEW CHAMPION" vs "ILLEGAL ORACLE".** Canonical `RESULTS.csv` row 79 marks RT-131 (7-stream within-timestep rank average, 0.62524) as `NEW CHAMPION`. `wave2/STATE_OF_RESEARCH_V2.md` and `wave2/runner_semantics.md` state RT-131 is **illegal** (the cross-section does not exist under the single-pass runner) and "must never be called the champion". The root `STATE_OF_RESEARCH.md` also acknowledges RT-131 "could never have been submitted". **The ledger and the reports disagree about whether RT-131 is the champion.**

**C2 - "RT-150" means three different things.** (a) wave2: deployable 7-stream frozen-CDF ensemble, 0.62589. (b) multi-agent-2026: a REJECTED gating experiment, -0.0219. (c) canonical `RESULTS.csv` `RT-150_f1`: "DGP-cluster routing" gating, 0.60619. HANDOFF_WAVE3.md section 0/2 explicitly warns about this collision.

**C3 - Three divergent "champions" across lineages.** RT-160 (logit average, 0.62544/0.62561, multi-agent lineage, root STATE_OF_RESEARCH.md) vs RT-150 (frozen CDF, 0.62589, wave2 lineage) vs RT-600 (seven specialists, 0.62581 dev / 0.6268 external, research/current lineage). No single authoritative champion exists in the canonical worktree.

**C4 - RT-125R reproducibility.** `wave2/STATE_OF_RESEARCH_V2.md` NEGATIVE section says "RT-125R could not be reproduced as configured" (LightGBM 4.7.0 rejects `boosting=goss` + bagging), yet `wave2/RESULTS_wave2.csv` records RT-125R = 0.617411 (vs wave-1 0.613557, +0.0039). The recorded value and the "could not reproduce" claim are in tension.

**C5 - HEAD "weekend harness" is metric-incompatible with the wave research.** The canonical HEAD (33fa041) produced `experiments/leaderboard.csv` (series-level AUC, best 0.598843) using statistical before/after features, while the wave research (tags/branches) uses TS-AUC and the 500-column causal bank (champion 0.61510). The HEAD thread appears to be a fresh restart that does not build on waves 1-8.

**C6 - Metric-weighting "CLOSED" vs "unconfirmed".** `20260818_claude_update_1.md` (Path E) warns the pair-count weighting "could not be confirmed from public material. If the weighting is wrong, every number in this project is measuring the wrong thing." `platform_constraints.md` (2026-08-19) and root `STATE_OF_RESEARCH.md` mark it "CLOSED" (confirmed `w(t)=n_pos*n_neg`). This is a resolved risk, but the two documents sit side by side in the worktree with opposite statuses.

**C7 - Early-detection premise vs metric reality.** `BRIEF_CLAUDE_wave3_alpha.md` states "we care enormously about early evidence because real-time TS-AUC weights every timestep." Wave 5 D1 (`STATE_OF_RESEARCH_V5.md`) found ages 0-20 carry only **11%** of pair weight and age 100+ carries **57%**, and that matching the training objective to the metric's pair weighting made the model **worse** (-0.00147). The brief's premise is contradicted by the measured weighting.

---

## Missing Evidence

1. **`research/` in the canonical worktree contains only 3 files** (BRIEF/HANDOFF). All referenced research artifacts (`FAILED_EXPERIMENTS.md`, `RDOF_LEDGER.md`, `VALIDATION_V2.md`, `STATE_OF_RESEARCH_V2/V3/V4/V5.md`, `folds/`, `reports/`, `scripts/`) are absent from the canonical worktree - they live only in tags/branches/sibling worktrees.
2. **Submission notebook `submissions/C_ensemble_deployable.ipynb`** - gitignored, missing; RT-150's 4.57 ms/point PASS is unverifiable from a clean checkout.
3. **Trained model directory + wave-2 OOF `.npy` vectors** - uncommitted/missing.
4. **Wave-2 experiment commits** `94f77c5`, `5101066`, `93bdb79`, `b5ea9d1` - not in object DB (provenance gap for RESULTS_wave2.csv rows).
5. **Branch `research/wave2-2026`** - referenced as "USE THIS" but absent from branch list.
6. **`crunch test` result** - never run (egress blocked); the only unverified link in the deployment chain.
7. **`n_online`-correlation gate for the other 6 streams** - red team only screened RT-100R; RT-121/122/123/124/125 were never screened (redteam_generator_artifacts.md section 9.2).
8. **`m03_dyn::az_*_exp_l*_z` ablation** - the "only genuinely unexplained mass" (~1.3M gain, no univariate signal, no clock explanation) was never ablated (redteam section 9.1, 10.2).

---

## Claims Requiring Reproduction

1. RT-600 external 0.6268 (LB-001) and dev pooled 0.63828 - REPORT-CLAIM; OOF vectors exist on `research/wave7-t2-promotion` worktree.
2. RT-150 deployable 0.62589 and +0.01057 bootstrap CI - REPORT-CLAIM; OOF vectors uncommitted.
3. W7-D3R future-information gap +0.071 cell AUC (5/5 folds) - REPORT-CLAIM.
4. Wave-8 five mechanisms "all KILL, marginals ~ -0.0003" - REPORT-CLAIM.
5. The `n_online` oracle = 0.62948 alone / 0.66625 blended - REPORT-CLAIM (red team measured; not independently reproduced).
6. Gain-importance vs metric-value Spearman 0.163 - REPORT-CLAIM.
7. Composition sensitivity +/-0.02-0.03 - REPORT-CLAIM.

---

## Recommended Reads for Research Agents

1. `structural-break-research-current/research/STATUS.md` - the authoritative current-state pointer.
2. `structural-break-research-current/research/NEW_AVENUES_2026.md` - the open-question framing + 9 functional classes + 11-experiment queue (esp. section D.6, E).
3. `structural-break-wave8/research/STATE_OF_RESEARCH_V5.md` - wave-5 metric-weighting finding + oracle-frontier implications.
4. `structural-break/HANDOFF_WAVE3.md` - the fork, ID collision, blockers, traps.
5. `structural-break/wave2/STATE_OF_RESEARCH_V2.md` + `VALIDATION_V2.md` + `runner_semantics.md` - deployability + validation discipline.
6. `structural-break/wave2/redteam_generator_artifacts.md` - n_online oracle + gain-importance misalignment.
7. `structural-break/RESULTS.csv` + `wave2/RESULTS_wave2.csv` + `structural-break-research-current/research/RESULTS.csv` (212 rows) + `structural-break-wave8/research/RESULTS.csv` (221 rows) - the four ledger versions.
8. `structural-break-research-current/research/EXPERIMENT_ID_MAP.md` - resolves the RT-xxx ID spaces across waves.

---

## Candidate Files Requiring Deeper Inspection (by primary researcher / skeptic)

1. `structural-break-research-current/research/RESULTS.csv` (212 rows) - full wave 1-7 ledger with statuses; needs row-by-row status/contradiction audit (KILLED/VOID/INVALID/UNVERIFIED).
2. `structural-break-wave8/research/RESULTS.csv` (221 rows) - includes wave-8 pilots (RT-1006..1052) and the RT-900 VOID oracle.
3. `structural-break-wave8/research/WAVE8_FUTURE_AWARE_PREREG.md` + `WAVE8_PROPOSAL_future_aware_distillation.md` - the five killed mechanisms and their falsification conditions.
4. `structural-break-research-current/research/reports/wave7/` - W7-D3R information-frontier numbers and T2 promotion battery.
5. `structural-break-research-current/research/FAILED_EXPERIMENTS.md` + `RDOF_LEDGER.md` - the cumulative negative-results and degrees-of-freedom ledger.
6. `structural-break-research-current/research/new_avenues_2026.csv` + `reports/new_avenues_2026_diagnostics.json` - the 79 source-backed mechanisms and D1-D6 diagnostics.
7. `structural-break-wave8/research/EXPERIMENT_ID_MAP.md` - to disambiguate RT-150/RT-160/RT-190/RT-600 across lineages.
8. The `wave2-2026-final` tag tree (`git ls-tree -r wave2-2026-final`) - the complete wave-2 `research/` including `REPRODUCIBILITY_MANIFEST.json` and `reports/*.json`.

---

## Sibling Directory Spot-Check Summary (budgeted archaeology)

Inspected (all are git worktrees of the canonical repo, `.git` file -> `structural-break/.git/worktrees/<name>`):

| sibling | research/ contents | materially different from canonical? |
|---|---|---|
| `structural-break-wave8` | 45 entries incl. STATE_OF_RESEARCH_V4/V5, WAVE4-8 status/prereg, RESULTS.csv (221 rows) | **YES** - most complete/recent wave history |
| `structural-break-research-current` | STATUS.md, NEW_AVENUES_2026.md, RESULTS.csv (212 rows), reports/wave4-7 | **YES** - canonical active lineage |
| `structural-break-oracle` | STATE_OF_RESEARCH_V2/V3, WAVE4_STATUS | YES (oracle-frontier study) |
| `structural-break-reproduce-2025` | STATE_OF_RESEARCH_V2/V3, WAVE4_STATUS | YES (2025 reproduction) |
| `structural-break-rt1257-deployment` | 22 entries, engineering/ | YES (deployment lane) |
| `structural-break-raw-data-atlas-2026`, `-data-forensics-2026`, `-deep-ensemble-frontier-crunch-2026`, `-gpu-tabular-2026` | 22-23 entries each | YES (parallel wave lanes) |

None of the siblings has a top-level `STATE_OF_RESEARCH.md` or `RESULTS.csv`; their research lives under `research/`. All share the canonical repo's object DB, so their branches/tags are the authoritative provenance for waves 2-8.

---

## Evidence Hierarchy Summary

- **[ARTIFACT-VERIFIED]**: `RESULTS.csv` (canonical), `wave2/RESULTS_wave2.csv`, `experiments/leaderboard.csv`, `experiments/best_config.json`, `experiments/progress.md`.
- **[GIT-VERIFIED]**: commit/tag/branch existence (16-commit HEAD log; tags `wave2-2026-final` etc.; missing commits 94f77c5/5101066/93bdb79/b5ea9d1; missing branch `research/wave2-2026`; sibling `.git` worktree pointers).
- **[REPORT-CLAIM]**: all STATE_OF_RESEARCH*.md, HANDOFF/BRIEF docs, VALIDATION_V2.md, redteam/PUBLIC_IDEA_MAP, NEW_AVENUES_2026.md, platform_constraints.md, 20260818_claude_update_1.md.
- **[CONTRADICTED]**: RT-131 "NEW CHAMPION" (C1); early-detection premise (C7); RT-125R reproducibility (C4).
- **[UNKNOWN]**: whether any of the three champion lineages is authoritative; whether the n_online oracle is fully avoided across all 7 streams; the m03_dyn::az_*_exp_l*_z unexplained mass.
