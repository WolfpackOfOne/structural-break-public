# Primary Research

Investigation: `20260829-173245-unresolved-quant-questions`
Primary researcher artifact (`02_PRIMARY_RESEARCH.md`), written blind of the independent skeptic.
Starting point: `structural-break/` @ `33fa041921f2e29d3b7bfbffcfac565997f9789a` (`claude/local-model-results-review-lk0f7i`), DIRTY worktree.

---

## Executive Conclusion

The repository's own documents name its central unresolved question twice, in almost the same words
(`research/reports/wave8_final.md` final section; `research/NEW_AVENUES_2026.md` §D.6), and my
independent inspection confirms it is real, quantitative, and open. My ranked top three:

1. **The future-information distillation gap.** An oracle arm that sees each series' own final
   online row gains **+0.07110 dominant-cell TS-AUC, 5/5 folds** over a matched-capacity causal arm
   [ARTIFACT-VERIFIED, `wave7_d3r.json`], yet six distinct attempts to transfer future information
   into causal students produced *one* weak ensemble marginal (T2: +0.00943 standalone but only
   **+0.00024** over an exchangeable seed clone, fold 0) and five pre-registered KILLS
   (marginals ≈ **−0.00031** each) [ARTIFACT-VERIFIED, `wave8_pilot_comparison.json`,
   `wave7_t2_ensemble_integration.json`]. Which of four mechanisms (non-shadowability, redundancy,
   pair-weight dilution, pilot underpower) explains the non-transfer is unresolved, and each implies
   a different research allocation.

2. **The ensemble diversity ceiling.** Since wave 4, *every* candidate measured against the binding
   control `(S + candidate) − (S + seed clone)` has returned ≈0 or negative: 8th exchangeable member
   +0.00003; 13-booster union rejected; `m09_back` lost to a seed clone; T2 +0.00024; five Wave-8
   mechanisms ≈ −0.0003. `wave7_t2_promotion_final.md` §N explicitly nominates "why the specialists
   correlate as highly as they do with everything tried against them" as possibly "the actual
   ceiling". Whether the 500-column causal bank is *already saturated* by the seven specialists —
   which would explain question 1(b) and a dozen negatives at once — has never been measured from
   the (extensive) existing OOF library. I reproduced the core redundancy fact from raw OOF vectors:
   T2's Pearson correlation with each specialist is 0.85–0.91, vs 0.936 for a seed clone
   [REPRODUCED].

3. **The unclosed artifact screens on the ledger's gain mass.** The red team priced the largest
   known illegal structure (`n_online` oracle = 0.62948 alone > champion 0.61510; 0.66625 blended)
   and verified the bank and *one* stream clean, but explicitly left: (a) the `n_online`
   correlation screen of the **other six production streams** unrun; (b) the
   `m03_dyn::az_*_exp_l*_z` family (~1.3 M gain, gain ranks 23–52, within-t AUC 0.484–0.510,
   clockR² < 0.015 — "the only genuinely unexplained mass") never ablated; (c) the consequence of
   Spearman(gain, |within-t AUC − 0.5|) = **0.163** — that RT-140's "all 500 columns are needed"
   conclusion was reached by pruning on gain, a quantity the red team showed is only weakly related
   to metric value, and the within-t-AUC-weighted alternative ranking was never computed
   [REPORT-CLAIM, `redteam_generator_artifacts.md` §5.2, §6, §9, §10; artifact-backed].

The headline *apparent* contradictions flagged for this investigation (RT-131 "NEW CHAMPION" vs
"illegal oracle"; three meanings of "RT-150"; three divergent "champions"; the HEAD harness's
incompatible metric) are **not unresolved scientific questions**: they are resolved by
`EXPERIMENT_ID_MAP.md` and the wave-2/5/7 documents, or are documentation/provenance defects.
I detail the resolutions — and the genuine residual verification gaps (lost RT-250 OOF vectors, a
single-observation external anchor with an in-repo labeling inconsistency) — below, because they
change what "the ledger says" but not what is scientifically open.

---

## Exact Question Reconstruction

The user asks: *identify the three most important unresolved quantitative research questions
documented in this repository, based only on current research reports, `RESULTS.csv`, and recent
Git history; do not propose new training runs.*

Operational reading, after inspection:

- "The repository" is effectively the shared object database of `structural-break/` plus its 23
  git worktrees [GIT-VERIFIED: `git worktree list`]. The canonical worktree's own branch contains
  only 16 commits of baseline-package history plus a "weekend model-search harness"; **all wave
  1–8 research lives on other branches/tags** (`research/current`, `research/wave8-…`, tags
  `wave2-2026-final` … `rt600-production-0.6268`) [GIT-VERIFIED]. Restricting attention to the
  canonical worktree would answer a different, mostly vacuous question.
- "RESULTS.csv" exists in (at least) four versions: canonical worktree (78 rows, wave 1 only,
  untracked), `wave2/RESULTS_wave2.csv` (105 rows, untracked), `research/current` (211 rows),
  `structural-break-wave8` (220 rows) [ARTIFACT-VERIFIED by direct read / `wc -l`].
- "Unresolved" per `00_QUESTION.md`: explicitly open, contradicted, KILLED-but-ambiguous,
  UNVERIFIED, or disagreement between ledger/report/history.
- "Important" I operationalize as: (stakes of the number at issue) × (degree to which the answer
  changes future research allocation) × (strength of evidence that the question is genuinely open
  rather than already answered).
- Non-goal honored throughout: every recommended next step is analysis of existing artifacts
  (OOF `.npy` vectors, importance CSVs, ledgers, JSON reports, model tarballs used for
  *inference-time* diagnostics only). No training run is proposed anywhere below.

---

## Verified Facts

### Repository structure and provenance

- **[GIT-VERIFIED]** Canonical worktree HEAD = `33fa041`, 16 commits total, all baseline package
  ("Phase 1–5") + "weekend model-search harness". No wave 1–8 research in this history.
- **[GIT-VERIFIED]** The wave research is reachable via branches (`research/current` @ `aca2c4f`,
  `research/wave8-future-aware-distillation` @ `589e1db`, `research/multi-agent-2026` @ `23cb4da`,
  `production/rt600`, …) and tags (`wave2-2026-final`, `wave5-alpha-final`, `wave7-t2-final`,
  `wave7-d3r-information-frontier`, `wave8-future-aware-final`, `rt600-production-0.6268`, …).
- **[GIT-VERIFIED]** All ~23 `structural-break-*/` siblings are **git worktrees** of the canonical
  repo (`.git` files → `structural-break/.git/worktrees/<name>`), contradicting the
  `00_QUESTION.md` premise; they share one object database.
- **[GIT-VERIFIED]** Wave-2 experiment SHAs `94f77c5`, `5101066`, `93bdb79`, `b5ea9d1` referenced
  by `wave2/RESULTS_wave2.csv` rows are **not in the object database** (`git cat-file -t` → fatal).
  `e98f5b9`, `8e76ad2`, `23cb4da`, `65b0e21`, `46583f5`, `ea5e2f2` do exist.
- **[GIT-VERIFIED]** Branch `research/wave2-2026` does not exist (only tag `wave2-2026-final`);
  `HANDOFF_WAVE3.md`'s "USE THIS" pointer is dangling.
- **[ARTIFACT-VERIFIED]** Canonical `RESULTS.csv` (untracked): 78 data rows; **77 carry
  `git_sha = "nogit"`**. Last rows: RT-130 (`NEW CHAMPION`, 0.62394), RT-131 (`NEW CHAMPION`,
  mean 0.625411023 / pooled 0.625244474). No RT-150 ensemble row; `RT-150_f1` = DGP-gating 0.60619.
- **[ARTIFACT-VERIFIED]** `experiments/leaderboard.csv` (HEAD harness): 142 rows, timestamps
  2026-08-29; best 0.598843 (`seed_stability_check…split_seed=7`); best fixed-split 0.579896.
- **[CODE-VERIFIED]** The HEAD harness metric is a **series-level AUC on a single 80/20 split**
  (`id_train_val_split`, `val_fraction=0.2`, `src/structural_break/competition_data.py:130-157`;
  `row["auc"]` in `experiment_runner.py`). It is a different quantity from TS-AUC.

### The metric and what it rewards

- **[REPORT-CLAIM]** Official TS-AUC = Σ_t w(t)·AUC(t) / Σ_t w(t), w(t) = n_pos(t)·n_neg(t);
  parity with organizers marked CLOSED (`platform_constraints.md`, 2026-08-19). The older warning
  (`20260818_claude_update_1.md`) that the weighting was unconfirmed is superseded — resolved, not
  open.
- **[REPORT-CLAIM]** Pair-weight distribution (W5-D1, `STATE_OF_RESEARCH_V5.md` §4): ages 0–20
  carry **11%** of pair weight; age 100+ carries **57%**; t∈[200,700] carries 64.3%.
- **[REPORT-CLAIM]** Noise hierarchy (wave 2): fold-to-fold SD 0.0085; partition-draw SD 0.0050;
  seed SD 0.0012. Ledger rule: below +0.001 "indistinguishable" after ~200 experiments
  (`RDOF_LEDGER.md`).

### Champions, resolved (the apparent contradictions)

- **C1 — RT-131 "NEW CHAMPION" vs "illegal oracle": resolved as documentation.**
  [ARTIFACT-VERIFIED] the canonical ledger row says `NEW CHAMPION`; [REPORT-CLAIM → ruling]
  `wave2/STATE_OF_RESEARCH_V2.md` §"ORACLE ENSEMBLE" and `EXPERIMENT_ID_MAP.md` §2 declare RT-131
  "ORACLE / ILLEGAL … not deployable, never a champion", because the within-timestep cross-section
  its rank average requires never exists under the single-pass runner (`runner_semantics.md`;
  production contract tests on `production/rt600` enforce single-pass, order-independence,
  prefix-invariance — 94–100 tests passing per the LB release record). The untracked canonical
  ledger is simply a stale wave-1 artifact that was never amended; the *tracked* ledgers on
  `research/current` / wave-8 never call RT-131 champion. **No open quantitative question remains
  here** — unless one doubts the runner semantics, which is falsified by the passing contract tests.
- **C2 — "RT-150" three meanings: resolved.** [REPORT-CLAIM, `EXPERIMENT_ID_MAP.md` §1–2]
  `RT-150_f1` (ledger) = rejected DGP gating, 0.60619; the deployable wave-2 ensemble had *no
  ledger row* and was **renamed RT-250** (0.62589, nested-confirmed scdf calibration, bootstrap CI
  [+0.00763, +0.01320] vs RT-100R). The map documents that "RT-150" never unambiguously named the
  ensemble.
- **C3 — three divergent champions: resolved by lineage, with two genuine residual gaps.**
  RT-250 (Linux, wave 2) 0.62589 → RT-420 / RT-600-development-architecture (macOS rebuild)
  **0.62581** [REPORT-CLAIM, `STATE_OF_RESEARCH_V5.md` §3; recomputed by the NEW_AVENUES author
  from the seven committed OOF vectors, §A.2] → RT-600 external **0.6268** (one public leaderboard
  observation). RT-160 (0.62544) is the independent corroborator on `research/multi-agent-2026`.
  Production anchor is unambiguously RT-600 (`STATUS.md`). Residual gaps:
  (i) **[REPORT-CLAIM, `EXPERIMENT_ID_MAP.md` §4 "Known defect"]** RT-250's original OOF vectors
  did not survive; 0.62589 is not recomputable from artifacts — only corroborated cross-platform
  by RT-420 = 0.62581.
  (ii) **[CONTRADICTED labeling]** the research docs attribute 0.6268 to "LB-001", but the
  authoritative release record (`production/rt600:research/reports/rt600_baseline_submission.md`
  §14–16) shows submission #1 **died at import**, #2 died in `infer` (missing lightgbm), and the
  runnable submissions were #3 (P=1) and #4 (P=4, predictions bitwise identical, sha256
  `dd082590…`, 50,983 rows). The 0.6268 therefore must have come from #3/#4; "LB-001" in
  `STATUS.md`/`EXPERIMENT_ID_MAP.md`/`RDOF_LEDGER.md` is a label error. The score itself is a
  leaderboard fact not verifiable from repo artifacts [REPORT-CLAIM].
  (iii) **[REPORT-CLAIM, release record §8]** the deployed final-10k model has *no* OOF estimate of
  its own (trained on all 10,000 series); 0.62581 is the development architecture's number.
- **C4 — RT-125R: resolved.** [REPORT-CLAIM, `RDOF_LEDGER.md` + ID map §4] The recorded 0.617411
  comes from a *reconstructed* config (LightGBM 4.7.0 rejects wave-1's `boosting=goss` + bagging);
  "could not be reproduced as configured" and the recorded value are consistent once the config
  change is known. Wave-1↔wave-2 stream deltas measure configuration, not reproducibility.
- **C5 — HEAD harness metric incompatibility: confirmed, but it is not an open research question.**
  [CODE-VERIFIED + ARTIFACT-VERIFIED] The HEAD thread optimizes series-level AUC on one 80/20
  split with 49–63 statistical columns; scores 0.48–0.60; it shares no code path with the wave
  lineage. The *metric-authority* question is resolved (TS-AUC is official; the ID map §"RT-940"
  even records a deliberate rule keeping series-ROC-AUC numbers out of `RESULTS.csv` to prevent
  exactly this confusion). The hazard is metric-discipline (comparing 0.5988 to 0.6151 would be a
  category error), and the ID map shows the project already learned this lesson once.

### The future-information chain (the core evidence)

- **[ARTIFACT-VERIFIED, `wave7_d3r.json` + ledger rows RT-990/RT-991]** W7-D3R (pre-registered,
  `WAVE7_D3R_PREREG.md`): dominant cell (t≥200, age≥100; 50.50% of dev pair weight):
  Arm A (RT-300) 0.65341; Arm B (RT-990, same 500 legal columns, more capacity) 0.64749;
  Arm C (RT-991, +500 columns of the series' own *final-row* feature vector — non-causal)
  **0.71859**. C−B = **+0.07110**, 5/5 folds (+0.05277…+0.08180); B−A = −0.00592 (4/5 negative).
  Verdict: CASE 2 — future-information limit; extraction capacity is not the bottleneck.
- **[ARTIFACT-VERIFIED, ledger + `wave7_teacher_nested.md`]** W7 nested-pure teacher distillation:
  T2 (RT-995) = +0.00943 over T0 (RT-990), 5/5 folds, bootstrap CI [+0.00425, +0.01373]. The
  earlier pilots RT-992/993 were caught as **outer-fold contaminated** (overstating T1 by +0.0235,
  T2 by +0.0122) and are kept in the ledger marked as such — the correction machinery worked.
- **[ARTIFACT-VERIFIED, `wave7_t2_ensemble_integration.json`]** Ensemble integration (fold 0):
  E0 = 0.6382763, E1 (+seed clone) = 0.6385864, E2 (+T2) = 0.6388229; **E2−E1 = +0.00024** →
  verdict MOSTLY REDUNDANT; standalone-to-ensemble overstatement ≈ 40×.
- **[ARTIFACT-VERIFIED, `wave8_pilot_comparison.json`]** Wave 8, five mechanisms, fold-0 pilots,
  all NOT CLEARED; marginals vs `RT600+clone`: SST −0.000310, ORR −0.000317, PCFB −0.000310,
  CFEP −0.000310, TGMC ≈ −0.0003. Notable internals: **ORR standalone +0.01167 / cell +0.01386
  yet negative ensemble marginal**; ORR recoverability 0.737 (bar 0.55) with repairs/damage
  769/694; CFEP's future-predictive embedding loses to its own matched BCE control (−0.00374);
  TGMC teacher-guided beats random (+0.00502) but both lose to the no-synthetic control; SST legal
  full-population −0.00060 vs eligible-slice +0.00810. Class-conditional eligibility at h=200,
  t<20: P(eligible|y=1)=0.33 vs P(eligible|y=0)=0.81 (`wave8_eligibility_diagnostic.json`,
  REPORT-CLAIM).
- **[REPORT-CLAIM, `oracle_information_frontier_2026.md`]** The standalone oracle-frontier study
  (series ROC-AUC, one row per series, pseudo-τ): RT-300 already matches/beats the known-boundary
  oracle at every horizon through h=150 (headroom −0.0058…+0.0037); only FULL horizon shows
  +0.0396. Superficially in tension with D3R's +0.071; they measure different objects (boundary
  *location* vs *eventual-state* feature vector; series-level vs within-t pair ranking).
- **[REPRODUCED]** T2 redundancy from raw OOF vectors (`research/current/research/oof/*.npy`;
  shapes 5,036,517 float32, shared 19.93% NaN mask): Pearson(T2, specialist) = 0.669 (RT-413) to
  0.908 (RT-300); Pearson(RT-300, RT-401 seed clone) = 0.936; Pearson(T2, its own control RT-990)
  = 0.904. The distillation moved the model very little in prediction space.

### The artifact/forensic surface

- **[REPORT-CLAIM, `redteam_generator_artifacts.md`, artifact-backed]** `n_online` oracle = 0.62948
  alone, 0.66625 blended with the champion; the champion itself is clean (AUC(score→n_online) ∈
  [0.488, 0.514] across six t-bands); placebo TS-AUC with fake τ on no-break series = 0.49904 ±
  0.00649; t=0 500-column multivariate has_break AUC = 0.4952; history-only = 0.5106; no
  cross-fold near-duplicates; bitwise causality at cuts (60,150,300,450).
- **[REPORT-CLAIM, same source]** Spearman(gain, |within-t AUC−0.5|) = **0.163**; four near-pure
  clocks hold 6.01% of total champion gain (`m00_core::t_online`, gain rank 2, has within-t AUC
  *exactly* 0.5000); 65 columns with no within-t signal hold 9.88% of gain; the
  `m03_dyn::az_*_exp_l*_z` family (~1.3 M gain, ranks 23–52) has no univariate signal and no clock
  explanation and **was never ablated**; the other six streams were **never screened** for
  `n_online` correlation; composition sensitivity: TS-AUC spread 0.058 across n_online tertiles,
  0.039 across kurtosis quartiles.
- **[ARTIFACT-VERIFIED]** RT-140 (fold-0 selection study): all_500 = 0.62689 > top_300 = 0.62645
  > top_200 = 0.62111 > … — concluded "keep all 500", via gain-ranked pruning.
- **[ARTIFACT-VERIFIED, ledger] + [REPRODUCED arithmetic]** W5-E9 objective arms: RT-700
  (metric-aligned `pairwise_w`) −0.00147 vs RT-702 control, per-fold deltas [+0.00085, −0.00502,
  −0.00107, +0.00008, −0.00217] → **2/5 folds positive**; RT-701 (hinge) −0.00476, 1/5 positive.
- **[REPORT-CLAIM]** Diversity-ceiling exhibits: W5-NULLTEST (8th exchangeable member) = +0.00003;
  W4-E6 13-booster union rejected (worse than best component); `m09_back` decorrelated *less* than
  a seed clone (0.8219 vs 0.7846) and lost to it; `m12_rdep` is the only block ever to beat a seed
  clone (+0.00141) and still failed the +0.0030 promotion bar.

---

## Reproduced Calculations

| # | computation | result | label |
|---|---|---|---|
| R1 | Canonical `RESULTS.csv` audit (row count, `nogit` share, RT-130/131/150_f1 statuses) | 78 rows; 77 `nogit`; RT-131 `NEW CHAMPION` 0.62541/0.62524; `RT-150_f1` = gating 0.60619 | [REPRODUCED] |
| R2 | Object-DB membership of wave-2 SHAs cited in `RESULTS_wave2.csv` | `94f77c5`, `5101066`, `93bdb79`, `b5ea9d1` absent; `e98f5b9` etc. present | [GIT-VERIFIED] |
| R3 | W5-E9 per-fold deltas from `structural-break-wave8/research/RESULTS.csv` rows | RT-700−RT-702: mean −0.001466, **2/5 positive**; RT-701−RT-702: −0.004760, 1/5 positive | [REPRODUCED] |
| R4 | OOF vector audit + correlations (`research/current/research/oof/`) | 10 vectors load, 5,036,517 float32 rows, shared 19.93% NaN mask; Pearson(T2, seven specialists) = 0.669–0.908; Pearson(RT-300, RT-401) = 0.936; Pearson(T2, RT-990) = 0.904 — matches the reports' diversity table (within-t 0.71–0.91; clone 0.91) | [REPRODUCED] |
| R5 | Wave-8 pilot marginals and T2 integration from JSON artifacts | marginals −0.000310…−0.000317, all NOT CLEARED; E2−E1 = +0.0002365 (fold 0) | [ARTIFACT-VERIFIED] |
| R6 | D3R deltas from JSON | C−B = +0.07110 (5/5), B−A = −0.00592; cell pair weight 0.5050 | [ARTIFACT-VERIFIED] |
| R7 | RT-131 ledger note arithmetic | pooled 0.62524 − RT-100 pooled 0.61500 = +0.01025 ✓ (matches note) | [DERIVED] |

Not reproduced (stated honestly): RT-600 external 0.6268 (leaderboard-side); the +0.071's raw pair
machinery (JSON internals consistent with the .md, but I did not recompute from rows — the row
layout requires the 10 GB store); the full SCDF recalibration of the seven-stream blend (machinery
exists on `research/current`, vectors exist; not run).

---

## What the Existing Result Actually Means

The repository contains **two disjoint research programmes sharing one working tree**:

1. The wave lineage (waves 1–8, on tags/branches/worktrees): a TS-AUC, 5-fold, permanently-folded,
   bootstrap-and-partition-controlled programme whose production anchor is RT-600 (external 0.6268,
   dev-architecture 0.62581). Its self-documented state is: representation = 500 causal columns in
   7 null-calibrated modules; combination = equal-weight cross-fitted SCDF blend of 7 specialists;
   every learned combination/weighting/gating loses; the one external measurement transferred
   flat-to-positive; the future-information frontier is measured (+0.071 cell AUC) and six attempts
   to harvest it causally failed.
2. The HEAD "weekend harness" (today's `experiments/leaderboard.csv`): a series-level, single-split
   sweep whose scores (0.48–0.60) are a different quantity and must not enter any comparison with
   the TS-AUC ledger. It duplicates, at a much weaker evidentiary standard, ground the wave lineage
   covered and abandoned. Nothing quantitative is *unresolved* by its existence; the risk is
   category error by future readers, and the repository's own ID map shows this exact confusion
   occurred before (RT-940–944 exclusion rule).

The most consequential *single* numbers in the repo, in my judgment: the D3R C−B gap (+0.07110),
the T2 standalone gain (+0.00943), the T2/ensemble marginal (+0.00024), the seed-clone bar
(+0.0030 promotion threshold vs +0.00003 measured value of an exchangeable member), and the
`n_online` oracle price (0.62948/0.66625). Every unresolved question I nominate is a question
about how these numbers relate to one another.

---

## Mechanism Analysis

The pattern to explain is not "five mechanisms failed"; it is the **three-layer dissociation**:

- (i) an oracle with the series' own future gains +0.071 cell AUC — the information exists;
- (ii) a nested-pure teacher distilled from that oracle gains +0.00943 standalone, 5/5 folds,
  CI clear of zero — *some* of the information is causally transferable to a single model;
- (iii) the same T2, and five other transfer routes, add ≤ +0.0003 over a *seed clone* inside the
  seven-specialist ensemble — none of the transferable information survives the ensemble.

Candidate global mechanisms:

- **M1: Redundancy / saturation.** The seven specialists already extract what the prefix offers;
  the teacher's transferable component is a *refinement of the same signal* (T2's Pearson with its
  own control is 0.904 [REPRODUCED]). The oracle's extra lift comes mostly from the part that
  cannot be transferred ("answer importation": the final row directly resolves "has this deviation
  persisted?"). Under M1, the ceiling is the bank, and no same-bank candidate can ever clear the
  seed-clone bar.
- **M2: Dilution / coverage.** The transferable signal is real but concentrated in a
  pair-weight-poor slice (SST: +0.0081 on the eligible slice, −0.0006 full-population; ORR: 877
  confirmed pairs, +0.737 recoverability, −0.00005 aggregate). Under M2, the metric's geometry —
  not the information — kills the mechanisms, and only channels applying to *most* of the dominant
  cell can matter (NEW_AVENUES §E.1's "diffuse loss" finding supports this reading).
- **M3: Non-shadowability.** The +0.071 is information that exists only *after* the prefix; no
  causal state variable summarizes it. CFEP's future-predictive embedding losing to its own matched
  BCE control (−0.00374) is the sharpest single exhibit: learning to predict the future did not
  even beat learning the task.
- **M4: Statistical fragility of the gates.** All Wave-8 verdicts are fold-0 pilots with
  un-bootstrapped marginals (acknowledged in `wave7_t2_promotion_final.md` §I); the marginals are
  suspiciously constant (−0.000310 ± 3e-6 across three mechanisms, and `rt600_plus_candidate`
  matches E0 to the 7th decimal for SST/PCFB/CFEP — consistent with the candidate contributing
  ~nothing to the blend rather than a measured negative). Under M4 some KILLs are premature and
  the true marginals are ~0 ± pilot noise.

M1 and M2 are not exclusive; M1 explains the ensemble layer, M2 the population layer, M3 the
oracle layer. M4 is about evidentiary strength, not the world.

---

## Competing Explanations

### For Q1 (the distillation gap)

- **H1a (non-shadowability / answer importation).** Supporting: CFEP −0.00374 vs its own BCE
  control [ARTIFACT-VERIFIED]; TGMC's synthetic-row dilution; the D3R interpretation ("the series'
  own eventual state carries almost all of the discriminating power this cell is missing, and none
  of it is legally visible at time t"); the oracle-frontier study showing headroom only at FULL
  horizon. Contradicting: T2 *did* transfer +0.00943 standalone through nested-pure labels — the
  future is not entirely inscrutable. Falsification: if the pair-overlap diagnostic (below) shows
  T2's corrected pairs are a large, *stable* subset of Arm C's corrected pairs and those pairs are
  lost by the prefix in principle (e.g., concentrated where prefix excursion lengths are
  statistically indistinguishable across classes), H1a is strengthened; if Arm C's gain
  decomposes onto channels whose prefix analogues demonstrably exist (excursion duration,
  persistence), H1a weakens.
- **H1b (redundancy with the bank).** Supporting: T2's correlation structure [REPRODUCED]; Arm B
  ≈ Arm A (more capacity buys nothing); W5-NULLTEST +0.00003; ORR's positive standalone but zero
  marginal. Contradicting: T2 is the *most* decorrelated successful candidate relative to the
  blend (0.674 within-t with E0 per `wave7_t2_promotion_final.md` §H) yet still adds nothing —
  under naive redundancy, lower correlation with real standalone alpha should add *something*.
  Falsification: effective-rank analysis of the existing OOF library; if rank ≈ 2–3 and a
  cross-fitted convex blend of existing vectors cannot beat E1, H1b is confirmed as binding.
- **H1c (pair-weight dilution / coverage).** Supporting: SST eligible-vs-full contrast
  (+0.0081 → −0.0006); eligibility class-conditioning (0.33 vs 0.81); ORR's tiny confirmed
  population; D1's diffuse-loss decomposition. Contradicting: T2's standalone +0.00943 is a
  *whole-dev* number, so its alpha is not confined to a tiny slice; the dilution story alone
  cannot explain why T2's ensemble marginal is 40× smaller than its standalone gain.
  Falsification: cell decomposition of T2's standalone gain (D2 below) — if T2's gain lives
  outside the dominant cell, dilution explains the ensemble failure but also indicts the pilot
  design (training signal not aimed at the cell); if inside, dilution is refuted for T2.
- **H1d (pilot underpower).** Supporting: fold-0-only, un-bootstrapped marginals; suspiciously
  constant −0.00031. Contradicting: 5/5 consistent sign across mechanisms, and T2's full-5-fold
  bootstrap exists for the standalone claim. Falsification: series-level bootstrap of the five
  existing fold-0 marginal computations from existing vectors (no training) — if CIs include the
  seed-clone bar, the KILL verdicts were statistically premature.

### For Q2 (the diversity ceiling)

- **H2a (bank saturation).** Supporting: W5-NULLTEST; W4-E6; `m09_back`; Arm B ≈ Arm A; my
  correlation reproduction. Contradicting: `m12_rdep` beat a seed clone (+0.00141) — saturation is
  not total; and the NEW_AVENUES D4 diagnostic shows a *standalone* excursion-run channel at 0.608
  cell AUC with ρ = 0.37 to RT-600 (far below any prior candidate), i.e., apparently-untapped
  causal information may exist. Falsification: the cross-fitted convex ceiling over existing
  vectors (D4 below); if it materially exceeds E1 with fold consistency, saturation is refuted.
- **H2b (combination-rule failure, not information absence).** Supporting: learned stacks lost to
  equal averages in wave 1 (RT-130 note); weighting/gating all measured and lose. Contradicting:
  the equal-weight + SCDF blend already recovers 99.7% of the illegal oracle rank average's gain
  (wave-2 B1.5) — the combination rule is demonstrably near the achievable frontier *for these
  streams*. Falsification: if a cross-fitted convex blend (analysis-only) beats the equal blend by
  > +0.001 consistently, the combination rule — not the information — is the constraint.
- **H2c (selection-exhausted search space).** Supporting: RDOF_LEDGER documents ~200 experiments;
  the "+0.001 is indistinguishable" rule; repeated reuse of the same folds. Contradicting: the
  promotion battery's controls (seed clones, alt partitions) are specifically designed against
  this, and the Wave-7/8 preregistrations were committed before scores. Falsification: hard to
  falsify with existing artifacts; a lower bound is the alt-partition stability of the existing
  key deltas (alt1/alt2/alt3 vectors exist for RT-300/401/41x), which *is* checkable for free.

### For Q3 (the artifact screens)

- **H3a (production is clean; the gaps are formalities).** Supporting: the bank-level screens are
  clean at t=0 (0.4952 multivariate), placebo-clean (0.49904), champion score-level clean across
  six t-bands; all streams use (subsets of) the same screened bank; production no-`n_online`
  contract tests pass. Contradicting: the red team explicitly warns "a feature is not safe because
  nobody wrote `n_online` in the code; it is safe when its correlation with `n_online` is measured
  at zero" — and that measurement exists for 1 of 7 streams. Falsification: the six missing
  screens themselves (D5 below).
- **H3b (the az family is fitted noise / non-transferring interaction mass).** Supporting: ~1.3 M
  gain with within-t AUC 0.484–0.510 and clockR² < 0.015; the project's own D.3 ("gains
  anti-stack") shows importance mass and metric value decouple. Contradicting: removing columns
  can hurt even when they look individually dead (LightGBM interaction value is real; the red
  team says "either pure interaction value or fitted noise" — genuinely unknown).
  Falsification: prediction-time knockout/permutation on existing model artifacts (D6 below).
- **H3c (RT-140's selection conclusion is an artifact of ranking on gain).** Supporting:
  Spearman 0.163; 6% of gain in pure clocks. Contradicting: RT-140's recorded subsets already show
  all_500 > top_300 > top_200 — the *measured* ordering favored keeping everything, so the
  conclusion has empirical support beyond the ranking statistic; the untested claim is only that a
  *within-t-AUC-weighted* top-k would do better. Falsification: compute the alternative ranking
  (D7) — if it coincides with gain ranks on the top ~300, the question collapses.

---

## Statistical Concerns

1. **Fold-0 pilots with un-bootstrapped gates.** Every Wave-8 marginal is a single-fold point
   estimate; the reports acknowledge this. The measured fold-to-fold SD is ~0.0085–0.012
   [REPORT-CLAIM + ledger per-fold SDs], so a ±0.0003 marginal is far below pilot-level noise —
   but the *sign consistency* across five mechanisms is the actual evidence, and it is negative.
2. **Selection pressure.** ~220 ledger rows across versions; RDOF_LEDGER's accounting is honest
   ("66 of 78 wave-1 rows `nogit`… may not be cited as confirmed results"). The promotion bar
   (+0.0030, 4/5 folds, bootstrap, alt partitions) is appropriately tighter than the noise
   hierarchy. The one place selection may have bitten: the *standalone* T2 +0.00943 was selected
   as the best of {T1, T2} before the ensemble leg; T1 failed its CI. Treat T2's standalone number
   as a screen, its ensemble marginal as the result.
3. **The "metric-alignment hurts" headline is underpowered.** My recomputation: −0.00147 with 2/5
   folds positive — sign-inconsistent, inside fold noise. The honest statement is "no evidence
   metric-aligned pair weighting helps"; `STATE_OF_RESEARCH_V5.md`'s "makes the model worse" is
   stronger than its own table supports. (The hinge arm, −0.00476 with 1/5 positive, does look
   genuinely negative.)
4. **External anchor = one observation.** 0.6268 vs dev 0.62581 — flat-to-positive transfer
   [REPORT-CLAIM]. Composition sensitivity (±0.02–0.03 across n_online tertiles) dwarfs most
   ledger deltas; one external point cannot pin transfer to better than that.
5. **Correlated errors / resampling unit.** The project's own bootstrap is series-level (correct
   for series dependence). My reproductions respect the same unit.
6. **Shared 19.93% NaN mask in the OOF vectors** [REPRODUCED]: all ten vectors share it, so
   cross-vector correlations are not distorted by differential masking; I did not verify *which*
   rows are masked (likely pre-window rows) — flagged as a minor UNKNOWN.

## Leakage / Purity Assessment

- **Caught and handled correctly:** RT-900 (label leak via missingness mask) → VOID, ID retired;
  RT-992/993 outer-fold contamination → caught, quantified (+0.0235/+0.0122 overstatement),
  corrected by nested RT-994/995, contaminated rows *kept and marked*; wave-8 eligibility
  class-conditioning handled per prereg (never a feature; oracles restricted to eligible
  subpopulations).
- **Standing gates:** production contract tests include four named no-`n_online` tests, prefix
  invariance, order independence — 94–100 passing at each release [REPORT-CLAIM, release record].
- **Open purity gaps (the substance of Q3):** six production streams never screened for
  `n_online` correlation at the score level; `m03_dyn::az_*` unexplained gain mass never ablated;
  RT-140's feature-selection conclusion never stress-tested against the within-t-AUC ranking.
- **Provenance purity:** 77/78 canonical-ledger rows unattributable (`nogit`); four wave-2 SHAs
  missing from the object DB; RT-250's OOF vectors lost. These do not contaminate numbers, but
  they cap the evidentiary grade of the wave-1/2 record at [REPORT-CLAIM] except where
  re-executed (RT-100R exact, RT-123R exact, RT-420 cross-platform).

## Falsification Tests

For each nominated question, what would show it is **already resolved or unimportant**:

- **Q1 resolved-downward:** (i) T2's pair corrections over T0 overlap Arm C's corrections over
  Arm B at no more than seed-clone rates, AND the ensemble already wins those pairs → the gap was
  never causally harvestable beyond the bank; question closes as "redundancy + importation",
  fully explained. (ii) A series-level bootstrap of the five existing fold-0 marginals shows the
  KILLs were noise → the question was never as settled as reported (re-opens, different answer).
- **Q1 unimportant:** the cross-fitted ceiling over *existing* vectors shows the ensemble is
  already at the bank's ceiling AND the D3R-translated pooled ceiling (+0.0359) is unreachable in
  principle for any causal student (e.g., Arm C's gain concentrates in final-row channels with no
  prefix analogue) → the +0.071 is trivia, not a research direction.
- **Q2 resolved:** the greedy/cross-fitted ceiling computation over the existing OOF library
  answers it directly in either direction (any vector or blend clears +0.0015 with 4/5-fold
  consistency → ceiling dead; nothing does → ceiling confirmed as binding for this bank).
- **Q3 resolved:** the six missing `n_online` screens land in [0.48, 0.52]; the az knockout moves
  fold-paired TS-AUC by ≲0.001; the within-t-AUC-weighted top-300 ≈ gain-ranked top-300.
- **Q3 unimportant:** even if a stream shows mild `n_online` correlation, the single external
  point (0.6268 vs 0.62581 dev) already bounds the total transfer gap to be non-disastrous —
  *unless* the correlation is large, in which case the external number becomes the anomaly to
  explain.

## Highest-Value Diagnostic Tests (existing artifacts only; $0, no training)

All artifacts below are enumerated and verified present:
`research/current/research/oof/` (RT-300, RT-401, RT-410–415 + alt1/2/3, RT-990, RT-995 +
per-outer-fold), `structural-break-wave6/research/oof/` (adds RT-300–303, RT-401–406 seed clones,
RT-430–434, RT-500–506, RT-700/701/702, RT-710–712, RT-730–760, RT-811–816, RT-900, RT-960/961,
RT-970/971, RT-991, RT-992/993, RT-994 + nested_Q inner-teacher vectors, `wave5_*` blend vectors,
`*.importance.csv` for most), `structural-break-wave8/research/oof/` (adds RT-1006–1052 pilot
vectors), helpers `research/scripts/wave8_common.py` (`ensemble_marginal`, `pair_repair_stats`,
`nested_oof_regressor`) on `research/current`, folds in `research/folds/`, store/features in
`structural-break-claude-wave3/cache/`, model tarballs `wave2/models_rt100_stream.tar.gz`,
`wave2/models_rt150_ensemble.tar.gz`, and `models/final10k_ensemble/` on `production/rt600`.

- **D1 — Triple pair-overlap of corrections (Q1, discriminates H1a/H1b/H1c).** Using
  `pair_repair_stats`-style machinery on existing vectors: compute the sets of weighted pairs that
  (a) Arm C fixes vs Arm B (`RT-991.npy` vs `RT-990.npy`), (b) T2 fixes vs T0 (`RT-995.npy` vs
  `RT-990.npy`), (c) seed clone fixes vs champion (`RT-401.npy` vs `RT-300.npy`), (d) E0 already
  wins. Report |b ∩ a| vs |c ∩ a| (chance) and |b ∩ a ∩ d|. Cost: hours of CPU, no training.
- **D2 — Cell decomposition of T2's standalone +0.00943 (Q1).** The D1 loss cube
  (`wave7_rt600_exact_alpha_budget.json` + decomposition scripts) already exists for RT-600;
  recompute the same decomposition for the RT-995−RT-990 delta by cell/age/break-family. If the
  gain lives outside the dominant cell, the distillation never touched the D3R information.
- **D3 — Bootstrap the five Wave-8 fold-0 marginals (Q1, H1d).** Series-level bootstrap of the
  existing marginal computations (vectors exist) to attach CIs to the five KILL verdicts.
- **D4 — Ensemble ceiling from the existing OOF library (Q2).** (i) greedy forward selection over
  all ~40 existing OOF vectors under the documented outer-fold protocol (weights/selection fitted
  on folds ≠ k, scored on k — analysis, not training); (ii) effective rank of the within-t
  rank-correlation matrix; (iii) leave-one-in marginals for every existing vector vs the
  seed-clone control. This directly answers "can anything already computed beat +0.0003?".
- **D5 — The six missing `n_online` screens (Q3, H3a).** AUC(stream score → n_online > median)
  at t ∈ {0,20,50,100,200,400} for RT-410–415 and the production blend, from existing vectors +
  store meta. The red team's own estimate: minutes.
- **D6 — `az` block prediction-time knockout (Q3, H3b).** Permute/zero the 10 `m03_dyn::az_*`
  columns in the *cached* features and re-score with the *existing* trained boosters
  (inference-time perturbation of existing artifacts — no training). Paired fold-delta on existing
  OOF rows. Caveat: permutation breaks interactions asymmetrically; report both zeroing and
  within-series permutation.
- **D7 — Alternative feature-ranking paper audit (Q3, H3c).** Compute within-t AUC per column from
  cached features + labels; form the within-t-AUC-weighted importance; compare the alternative
  top-k sets to RT-140's gain-ranked sets. Produces the exact column lists any future
  (out-of-scope) retrain would test, and bounds how different they are.
- **D8 — Objective-paradox bootstrap (secondary).** Paired series bootstrap of RT-700 vs RT-702
  from existing OOFs (wave-8 worktree) to replace the "makes it worse" headline with a CI.

## Candidate New Experiments (analysis-only; NO training, per constraints)

1. **"What did the teacher actually transfer?" (D1+D2 as a preregistered analysis).**
   Mechanism: discriminates H1a/H1b/H1c with a single pair-set algebra. Expected upside: converts
   the project's named open question from a framing into a measured decomposition; either kills
   the entire future-aware family with evidence or names the exact subpopulation where a legal
   harvest remains. Evidence for upside: all vectors exist; the helpers exist; the D1 cube exists.
   Cost: ~1 analyst-day, hours CPU. Major failure mode: pair-set overlap is underpowered in
   low-weight regions (the dominant cell is 50% of weight, so this is unlikely to bind there).
2. **"The bank's ceiling" (D4).** Mechanism: H2a vs H2b. Upside: a measured ceiling
   retroactively explains ~10 negative results and gates the entire NEW_AVENUES queue (most of
   whose 79 mechanisms are same-bank additions). Cost: ~1 day. Failure mode: cross-fitted convex
   weights on 5 folds can overfit the selection itself — mitigate by the existing nested/SCDF
   protocol and by reporting the *selection-free* greedy path only.
3. **"Full-coverage artifact screen" (D5+D6+D7).** Mechanism: H3a/H3b/H3c. Upside: closes the
   only documented purity gaps on the production anchor; either certifies the ledger or finds a
   retroactive correction. Cost: D5 minutes, D6 hours, D7 hours. Failure mode: D6's knockout is
   not a clean ablation (interactions), so a null result is informative but a positive result is
   ambiguous between "interaction value" and "artifact".
4. **"Oracle-gain anatomy" (D1 extended to channels).** Correlate Arm C's corrected pairs with
   the existing excursion-run diagnostics (NEW_AVENUES D4/D5 channels, computed by
   `scripts/novel_streams/` on the existing store) to test whether the +0.071 concentrates where
   "has this deviation lasted?" is prefix-visible. Failure mode: the RT-991 importance CSV was
   **not** saved (verified absent — only `RT-990.importance.csv` exists), so the column-level
   version is blocked by a missing artifact; the pair-level version survives.

## Experiments I Would NOT Run

- **Any sixth future-aware distillation mechanism**, even if training were allowed: five
  pre-registered kills plus a measured 40× standalone→ensemble decay is a completed experiment
  series, not an invitation.
- **Re-running RT-100/RT-100R reproduction**: already exact (delta 0.0).
- **Re-deriving the +0.071 or the five marginals from rows**: JSON artifacts verified internally
  consistent; the marginal value of a fourth restatement is zero.
- **Any analysis on the HEAD weekend-harness leaderboard** as if it bore on the TS-AUC programme:
  category error (different label, different split, different metric).
- **Alternate-partition confirmation of T2's *standalone* number** (the one leg the T2 report
  left unrun): the ensemble leg already gated T2 at MOSTLY REDUNDANT; as the report itself argues,
  alt partitions cannot raise the ensemble verdict. (Its own author left it as "flag it and it can
  be started" — I would not flag it.)
- **Leaderboard probing** to measure transfer: prohibited by the project's own standing rule
  (release record §12) and scientifically empty given composition variance.

## Remaining Unknowns

1. Whether the +0.071 decomposes into "importation" vs "shadowable persistence state" (Q1 core).
2. The effective dimensionality and attainable ceiling of the existing OOF library (Q2 core).
3. The six unscreened streams' `n_online` profiles; the az family's true contribution (Q3 core).
4. [UNKNOWN] Which submission (#3 or #4) returned 0.6268, and whether a scoreboard export exists
   anywhere; the docs' "LB-001" label is inconsistent with the release record.
5. [UNKNOWN] What the shared 19.93% NaN mask in the OOF vectors corresponds to (assumed warmup
   rows; not verified against the store layout).
6. [UNKNOWN, secondary] The true sign of the metric-aligned objective effect (RT-700):
   sign-inconsistent at 2/5 folds; the hinge variant genuinely negative. The claim "alignment
   hurts" is not supported; "alignment does not help" is supported.
7. [UNKNOWN] Whether the 2025 reproduction study's failure (all rungs `blocked`, dependencies
   never installed) ever got a corrected rerun — as recorded it is a failed calibration anchor.
8. [UNKNOWN] Missing artifacts that block specific diagnostics: RT-991 importance CSV; RT-250's
   original Linux OOF vectors; wave-2 `.npy` stream vectors; `submissions/C_ensemble_deployable.ipynb`
   in the canonical worktree (exists on `production/rt600`).

## Unresolved Questions — Ranked Top 3

### Q1 — Why does the measured +0.071 future-information gap not survive distillation into any causal student or the ensemble?

**Precise falsifiable statement.** Four mutually distinct explanations — (a) the information is
not causally shadowable (answer importation), (b) it is shadowable but already extracted by the
seven specialists (redundancy), (c) it is real and novel but confined to a pair-weight-negligible
slice (dilution), (d) the KILL verdicts are fold-0 pilot noise — each make different, testable
predictions about the *existing* OOF vectors (pair-set overlaps, cell decomposition of T2's gain,
bootstrap CIs on the five marginals). None has been falsified; the repository's two most recent
research documents both name exactly this as the open question (`wave8_final.md` final section;
`NEW_AVENUES_2026.md` §D.6: "why does none of it survive contact with the full row population").

**Strongest evidence it is unresolved.** [ARTIFACT-VERIFIED] D3R +0.07110 5/5 folds vs
[ARTIFACT-VERIFIED] five NOT-CLEARED pilots at −0.0003 and T2's +0.00943 → +0.00024 collapse.
[REPRODUCED] T2's 0.904 correlation with its own control shows the distillation barely moved
prediction space. [ARTIFACT-VERIFIED] ORR's standalone +0.0117 yet negative marginal shows the
standalone→ensemble dissociation is not unique to T2.

**$0 diagnostics.** D1 (triple pair-overlap), D2 (cell decomposition of T2's gain), D3 (bootstrap
the five marginals), D4-extended (channel anatomy of Arm C's corrected pairs).

**Falsification conditions.** Listed above ("Falsification tests", Q1 bullets): the question is
resolved if the pair-overlap shows clone-level transfer and pre-won pairs (→ redundancy/
importation, closed) or if the marginals' CIs swallow the seed-clone bar (→ verdicts premature,
re-opened with different content).

**Importance.** It is the difference between "the bank is done, only new causal state variables
matter" and "the future-aware direction was executed wrong". The entire 79-mechanism
NEW_AVENUES queue is premised on one answer to this question; the premise is untested.

### Q2 — Is the seven-specialist ensemble at the information ceiling of the 500-column causal bank?

**Precise falsifiable statement.** "No model derived from the same 500 columns / same rows can
beat the seed-clone control by ≥ +0.0015 with 4/5-fold consistency" — testable *today* against
the ~40 existing OOF vectors without training. `wave7_t2_promotion_final.md` §N nominates exactly
this ("that redundancy … may be the actual ceiling on ensemble improvement from this direction").

**Strongest evidence it is unresolved.** The ceiling has never been *measured* — only inferred
from ~10 consecutive failures (W5-NULLTEST +0.00003; W4-E6; m09_back; T2 +0.00024; five Wave-8
marginals; W6 neural 0.54–0.58 far below trees). Counter-evidence that the bank is *not*
saturated: `m12_rdep` beat its seed clone (+0.00141); NEW_AVENUES D4's excursion channel shows
0.608 cell AUC at ρ = 0.37 to RT-600 [REPORT-CLAIM, diagnostics JSON]. My [REPRODUCED]
correlations quantify the redundancy but cannot bound the *achievable* marginal — only the
ceiling computation can.

**$0 diagnostics.** D4 (greedy + cross-fitted convex ceiling + effective rank over existing
vectors).

**Falsification conditions.** Any existing vector or cross-fit blend clears the bar → ceiling
dead (and the whole post-wave-4 negative streak needs re-reading as combination failure, H2b).
Nothing clears it → every same-bank "eighth stream" proposal is dead on arrival, permanently.

**Importance.** It is the joint sufficient statistic for a dozen results and the gatekeeper for
the largest documented research queue in the repo.

### Q3 — Does the production ledger survive a full-coverage artifact screen? (six unscreened streams; the unexplained `az` gain mass; gain-ranked feature selection)

**Precise falsifiable statement.** (a) All seven production streams' scores satisfy
AUC(score → n_online>median) ∈ [0.48, 0.52] across t-bands — currently verified for 1 of 7.
(b) The `m03_dyn::az_*_exp_l*_z` block's contribution is metric-bearing (knockout delta ≳ fold
noise) rather than fitted noise — currently unknown for the only unexplained ~1.3 M gain mass in
the champion. (c) RT-140's "all 500" conclusion is robust to ranking columns by within-t AUC
rather than gain — currently untested even as a paper ranking, despite Spearman(gain, metric
value) = 0.163 [REPORT-CLAIM, artifact-backed red-team measurements].

**Strongest evidence it is unresolved.** The red team's own "could NOT test" list (§9) and "with
more compute" list (§10) name exactly these three items; the `n_online` oracle price (0.62948 /
0.66625) makes (a) the highest-stakes unmeasured gate in the project; the production contract
tests prove the *code* cannot read `n_online`, but the red team demonstrated the risk is
*acquired correlation through training data*, which code tests cannot see.

**$0 diagnostics.** D5 (minutes), D6 (hours, inference-time on existing artifacts), D7 (hours).

**Falsification conditions.** Clean screens + null knockout + coincident rankings → Q3 closes
with the ledger certified. A large `n_online` correlation in any stream → the external 0.6268
becomes the anomaly and the ledger's deltas involving that stream need re-measurement.

**Importance.** The only nominated question that can *retroactively* change published numbers,
and the cheapest to resolve.

## Final Research Judgment

The repository is unusually honest: it prices its own degrees of freedom, keeps its contaminated
rows marked, voids its leaks, and names its open question in plain language. My independent
inspection found that honesty largely warranted — every load-bearing number I checked
(D3R deltas, T2 battery, Wave-8 marginals, ledger arithmetic, OOF correlation structure) verified
against artifacts or reproduced from raw vectors. The apparent provenance collapse is real as
*hygiene* (a stale canonical worktree whose untracked ledger still crowns an illegal oracle; a
labeling bug around "LB-001"; four dangling SHAs; one lost set of OOF vectors) but it does not
translate into open scientific questions, because the tracked lineage on `research/current`
resolves each contradiction explicitly.

The science is stuck at a well-measured wall, and the wall has two bricks that have never been
measured separately: how much of the +0.071 future gap is in-principle causal (Q1), and whether
the causal bank itself is already exhausted by the current seven (Q2). Both are answerable to a
first approximation from artifacts that already exist on disk, and I verified that those artifacts
exist and are enumerable. Q3 is cheaper than either and protects everything else. If only one
analysis is funded, I would fund D1+D2 (the pair-overlap triple and T2 cell decomposition): it is
the only test that discriminates among all four live explanations of the repository's own named
open question, and it cannot produce a useless outcome — every cell of its outcome table kills or
redirects a documented line of work.

No new training run is justified by the current record; no preregistration beyond analysis-only
diagnostics is warranted.

---

### Evidence-label summary for this report

- **[REPRODUCED]**: R1, R3, R4, R7 (ledger audits; W5-E9 per-fold deltas; OOF shapes/NaN mask;
  T2/specialist/clone correlations; RT-131 note arithmetic).
- **[ARTIFACT-VERIFIED]**: D3R JSON numbers; wave-8 pilot JSON numbers; T2 integration JSON;
  ledger rows in all four RESULTS.csv versions; HEAD leaderboard values; importance-CSV and OOF
  file inventories (including the *absence* of `RT-991.importance.csv`).
- **[GIT-VERIFIED]**: 16-commit HEAD; branch/tag topology; worktree identity of siblings; missing
  SHAs; missing `research/wave2-2026` branch.
- **[CODE-VERIFIED]**: HEAD harness split/metric mechanics.
- **[REPORT-CLAIM]** (unreproduced here, artifact-backed): red-team numbers; W5-D1 pair weights;
  noise hierarchy; external 0.6268; release-record test counts; NEW_AVENUES D1–D6 diagnostics.
- **[CONTRADICTED]**: "RT-131 is champion" (ledger) vs the binding ruling (illegal); "LB-001 =
  0.6268" vs the release record (submission #1 died at import); "metric alignment makes the model
  worse" (headline) vs its own sign-inconsistent per-fold table [REPRODUCED].
- **[UNKNOWN]**: items listed under Remaining Unknowns.
