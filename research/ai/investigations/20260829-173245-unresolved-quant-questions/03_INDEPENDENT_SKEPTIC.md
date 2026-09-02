# Independent Skeptical Analysis

Investigation: `20260829-173245-unresolved-quant-questions`
Role: Independent Quantitative Skeptic
Starting SHA: `33fa041921f2e29d3b7bfbffcfac565997f9789a` (`claude/local-model-results-review-lk0f7i`)
Blind to: `02_PRIMARY_RESEARCH.md` (not read)

---

## Independent Conclusion

The three most important **unresolved quantitative** questions in this repository are not the documentation collisions the evidence manifest leads with (RT-131 "NEW CHAMPION" vs illegal oracle; three "divergent champions"; RT-125R reproducibility; HEAD weekend-harness vs wave research). Those are **stale-ledger / ID-collision / metric-discipline problems**, not live scientific puzzles.

After adversarial screening of current reports, all four RESULTS ledgers, tags, and sibling worktrees through 2026-08-28, three questions survive:

1. **What fraction of W7-D3R Arm C's +0.071 cell AUC is illegal length/endpoint information versus genuine future-path break evidence?** This is the load-bearing scientific uncertainty behind Wave 7/8 and New Avenues. It has never been decomposed. Arm C (`RT-991`) is a final-row broadcast; `n_online` alone scores 0.62948 TS-AUC; the known-boundary oracle has **no headroom through h=150**.
2. **Why do Wave 8 legal future-aware students retain signal on the eligible subpopulation and collapse on the full row population the metric actually scores?** Wave 8 named this as *the* leftover question. New Avenues' proposed answer (missing duration state) was subsequently **falsified**. Eligibility is class-conditional. Existing Wave 8 OOF vectors can test remaining-length confounding without training.
3. **Which CatBoost-hybrid number, if any, is real: CSA-04's `RT-1264` "MAJOR" +0.00593 vs clone, or CSA-04R's `RT-1265`/`RT-1257` k*=2 `NOT_DISTINGUISHABLE` +0.00203 E2-E0?** Same-day (2026-08-28) documents disagree. STATUS files still advertise the MAJOR headline. This is the only post-saturation internal positive that is not already killed, and it is entangled with k-search selection bias.

Everything else that looks "open" in older reports is either **resolved by later work**, **a quoting error**, or **not a quantitative research question**.

---

## Facts I Could Verify

### Provenance (GIT-VERIFIED)

- Canonical worktree HEAD is `33fa041` on `claude/local-model-results-review-lk0f7i`, 16-commit linear history, **no wave 1-8 research in that log**. Confirmed.
- Research lives on tags (`wave2-2026-final`, `wave7-d3r-information-frontier`, `wave8-future-aware-final`, `rt600-production-0.6268`, ...) and sibling **git worktrees** of the same object DB. Scout's correction of `00_QUESTION.md` (siblings have `.git` files, not independent repos) is correct.
- `research/wave2-2026` branch is **absent**; tag `wave2-2026-final` -> `3f8bbba` exists. Commits `94f77c5`, `5101066`, `93bdb79`, `b5ea9d1` are **not in the object DB**. `e98f5b9`, `8e76ad2`, `23cb4da` exist.
- This is **normal branch-based workflow plus a stale HEAD checkout**, not an epistemic collapse. The science is reconstructible from tags/worktrees. The practical hazard is **stale STATUS files**, not missing objects.

Lineage dates that matter:

| ref | SHA (short) | date | role |
|---|---|---|---|
| canonical HEAD | `33fa041` | weekend harness | disconnected restart |
| `research/current` | `aca2c4f` | 2026-08-24 | last "nothing executed" New Avenues design |
| `research/wave8-future-aware-distillation` | `589e1db` | 2026-08-23 | five future-aware KILLs |
| `research/new-avenues-pilots-2026` | `b47b22a` | 2026-08-25 | first+second sweep exhausted, all KILL |
| `research/catboost-specialist-2026` | (CSA scoring `a360dfd`) | 2026-08-27 | `RT-1257` PROMOTION_WORTHY |
| `research/deep-ensemble-frontier-local-2026` | CSA-04 `766ecf0` / CSA-04R `98a1a9a` | 2026-08-28 | `RT-1264` MAJOR vs `RT-1265` NOT_DISTINGUISHABLE |
| `production/rt600` | `6e59130` | 2026-08-26 | frozen 0.6268 |

**Consequence the scout under-weighted:** `research/current` `STATUS.md` (2026-08-24, "DESIGN COMPLETE, NOTHING EXECUTED") is **not** the latest research state. Treating it as canonical open-question text will mis-rank what is still unresolved.

### Ledgers (ARTIFACT-VERIFIED)

| ledger | n rows | last ID | notes |
|---|---:|---|---|
| `structural-break/RESULTS.csv` (untracked) | 78 | `RT-131` | 77/78 `git_sha=nogit`; statuses `recorded` 75, `NEW CHAMPION` 2, one STOPPED EARLY |
| `wave2/RESULTS_wave2.csv` | 104 | `RT-210` | still marks `RT-130`/`RT-131` NEW CHAMPION |
| `research/current` `RESULTS.csv` | 211 | `RT-1100` | adds `RT-250`, `RT-900` VOID, W7 teacher IDs |
| `wave8` `RESULTS.csv` | 220 | `RT-1041` family | +12 Wave-8 pilots, all `recorded` (kills live in the report, not the status field) |
| `new-avenues-pilots` `RESULTS.csv` | 245 | `RT-1233` | RT-1200-1233, all KILL/control |
| `catboost-specialist` `RESULTS.csv` | 267 | `RT-1257` | CatBoost slots |
| `deep-ensemble-local` `RESULTS.csv` | 273 | `RT-1265` | `RT-1264` status MAJOR; `RT-1265` NOT_DISTINGUISHABLE |

Canonical `RT-131`: mean 0.625411, pooled 0.625244, fold_std 0.00898, status **NEW CHAMPION**, sha `nogit`. [ARTIFACT-VERIFIED]

`RT-150_f1`: 0.60619, gating, **not** the deployable ensemble. [ARTIFACT-VERIFIED]

`research/current` `RT-250`: 0.625895, notes explicitly say this is the canonical ID for the ensemble previously called RT-150. [ARTIFACT-VERIFIED]

`RT-900`: status **VOID**, 0.86552, label leak via missingness mask. [ARTIFACT-VERIFIED]

`RT-990` Arm B: 0.61185 mean / 0.61177 pooled. `RT-991` Arm C: 0.72003 mean / 0.71989 pooled. [ARTIFACT-VERIFIED]

`RT-1100` notes: E0=0.63828, E1=0.63859, E2=0.63882 (fold-0 T2 battery). [ARTIFACT-VERIFIED]

`RT-1257`: 0.627838, PROMOTION_WORTHY, E0=0.625811, E2=0.627838. [ARTIFACT-VERIFIED]

`RT-1264`: 0.628190, MAJOR, best_k=5. `RT-1265`: 0.627838, NOT_DISTINGUISHABLE, identical composition to `RT-1257`. [ARTIFACT-VERIFIED]

### Production / metric facts (REPORT-CLAIM unless noted)

- Production anchor **RT-600**, external LB-001 **0.6268**. [REPORT-CLAIM: `STATUS.md` on every current lineage; tag `rt600-production-0.6268`]
- 5-fold development mean for the RT-600 architecture: **0.62581**; pooled **0.625627**; dominant-cell **0.66428**. [REPORT-CLAIM: `NEW_AVENUES_2026.md` §A.2, author claims recomputation from seven specialist OOF vectors; data-forensics independently reconstructs RT600 mean **0.625811342**. Consistent to 6 decimals across two later reports.]
- The oft-quoted "pooled OOF ≈ 0.63828" is **fold-0 of the T2 E0 battery**, not 5-fold pooled. [ARTIFACT-VERIFIED from `RT-1100` notes; REPORT-CLAIM from `NEW_AVENUES_2026.md` §A.2 which already flags this correction.] **LB 0.6268 vs 5-fold 0.62581 is not a gap.** It is flat-to-slightly-positive transfer. [REPORT-CLAIM: `STATE_OF_RESEARCH_V5.md` §1]
- Official metric: `TS-AUC = Σ_t w(t) AUC(t) / Σ_t w(t)`, `w(t)=n_pos(t) n_neg(t)`. Marked CLOSED 2026-08-19. [REPORT-CLAIM: `HANDOFF_WAVE3.md` §3, `platform_constraints.md`]
- Noise hierarchy: fold-to-fold SD ≈ 0.0085, partition-draw SD ≈ 0.0050, seed SD ≈ 0.0012. Below +0.0005 is noise; +0.005 potentially meaningful; +0.010 material. [REPORT-CLAIM: `HANDOFF_WAVE3.md` §1]
- `n_online` illegal oracle: `(t+1)/n_online` or `-n_online` scores **0.62948** TS-AUC; 50/50 rank blend with RT-100 reaches **0.66625**. RT-100 scores do **not** predict `n_online` (AUC 0.488-0.514 across t-bands). [REPORT-CLAIM: `redteam_generator_artifacts.md` §1]
- W7-D3R cell (t≥200, age≥100): Arm A 0.65341, Arm B 0.64749, Arm C **0.71859**; C-B **+0.07110**, 5/5 folds, translated pooled +0.03591. B-A **-0.00592**. Verdict CASE 2. [REPORT-CLAIM from `wave7_d3r.md`; numbers also in `wave7_d3r.json` — ARTIFACT-VERIFIED JSON, not recomputed from OOF]
- Wave 8: five mechanisms, all FAIL on full population; RT600+cand marginal vs clone all ≈ -0.0003. SST legal **+0.00810** on eligible-pop dominant cell, **-0.00060** on full-pop. Eligibility at h=200, t<20: P(eligible|y=1)=0.33 vs P(eligible|y=0)=0.81. [REPORT-CLAIM: `wave8_final.md`]
- New Avenues first sweep RT-1200-1218: all failed continuation gates. Strongest `RT-1216` weighted CTM marginal_vs_clone **+0.000937** < +0.0010 gate, dominant net -35. Second sweep SS-01..04 all KILL. [ARTIFACT-VERIFIED ledger notes + REPORT-CLAIM `STATUS.md` 2026-08-25]
- Oracle information frontier (series-level AUC, different quantity from TS-AUC): legal RT-300 matches/beats known-boundary LGBM-rich through h=150 (headroom -0.0058 / -0.0019 / -0.0025 at h=20/100/150); FULL +0.0396. [REPORT-CLAIM: `oracle_information_frontier_2026.md`]
- Feature bank participation-ratio effective rank **21.282** / 500 columns. [REPORT-CLAIM: `DATA_FORENSICS_REPORT.md`]
- Ages 0-20 carry **11%** of pair weight; age 100+ **57%**. Metric-aligned `pairwise_w` **-0.00147**. [REPORT-CLAIM: `STATE_OF_RESEARCH_V5.md`]
- CSA-04R: under fixed E2-E0, parsimony selects k*=2 = `RT-1257`; verdict **NOT_DISTINGUISHABLE** from adding more slots; k=2 E2-E0 +0.002026, bootstrap 95% CI [+0.00043, +0.00354], B=2000. k=5 E2-E0 CI **includes 0**. [REPORT-CLAIM: `CSA04R_REANALYSIS.md`; `RT-1265` ledger row ARTIFACT-VERIFIED]

### HEAD weekend harness (ARTIFACT-VERIFIED)

`experiments/leaderboard.csv`: series-level AUC, single 80/20 split, best 0.598843. **Not TS-AUC. Not comparable.** A disconnected restart, not a competing champion.

---

## Claims I Could Not Verify

- Any AUC recomputed from raw OOF against labels in this pass. I did not load prediction vectors or rerun `sbr/metric.py`. Numbers above that are not ledger cells remain **REPORT-CLAIM** (or JSON-artifact claims).
- `RT-991.npy` (Arm C OOF): **not found** in `structural-break-wave7-promotion/research/oof/` (which does contain `RT-990.npy`, `RT-995.npy`) nor by filename search. Wave 7 D3R JSON exists; the prediction vector needed to residualize Arm C against `n_online` may be missing. [UNKNOWN / missing artifact]
- Wave-2 OOF `.npy` streams and `submissions/C_ensemble_deployable.ipynb`: missing/gitignored, as HANDOFF said. [UNKNOWN]
- External 0.6268: not independently fetched from Crunch. [REPORT-CLAIM]
- `n_online` screen on streams other than RT-100R: still documented as unrun. [UNKNOWN]
- `m03_dyn::az_*_exp_l*_z` ablation: still unrun. Given gain-vs-metric Spearman 0.163, this is probably not a metric puzzle. [UNKNOWN, low value]
- Whether CatBoost `RT-1254`-`RT-1263` `.npy` in the deep-ensemble-local `research/oof/` are outer-fold-pure 5-fold OOF or something else: not code-verified here.
- CSA-04R bootstrap: not recomputed. [REPORT-CLAIM]
- NEW_AVENUES §A.2 "I reproduced the RT-600 blend... bit-consistent with W7-D0": not independently reproduced. Later data-forensics reconstruction of 0.625811342 is corroborating but still a report.

---

## Strongest Alternative Explanation

The attractive story in `NEW_AVENUES_2026.md` §D.6 and `wave8_final.md` §S/§29 is:

> W7-D3R proved a real future-information gap (+0.071 cell AUC, 5/5). Wave 8 proved five distillations fail. The open question is why none of that information survives contact with the full row population. The missing state variable is duration ("has this deviation lasted?").

**Mundane alternative that currently fits more of the evidence:**

Arm C's lift is largely **non-causal series-length / endpoint / remaining-horizon information**, not "future break evidence a causal prefix could approximate."

Why this is not a stretch:

1. Arm C's extra features are each series' **final online row** (`wave7_d3r.md`). The final row exists at `t = n_online - 1`. Time-calibrated features of that row encode length.
2. A one-integer illegal oracle `(t+1)/n_online` already scores **0.62948**, beating RT-100 and the illegal rank ensemble. [REPORT-CLAIM, redteam §1]
3. The known-boundary oracle — a *different* privileged teacher that knows tau but not the far future — has **zero or negative headroom through h=150**, the horizons that carry most real-time mass. Headroom appears at FULL (+0.0396 series AUC) and weakly at h=200. [REPORT-CLAIM, oracle frontier] W7-D3R Arm C is the FULL-horizon object, not "a bit more future."
4. Wave 8 SST: legal student **works on the eligible slice** (+0.00810) and **fails on the full population** (-0.00060). Eligibility is class-conditional. That is remaining-length confounding sitting in the open in `wave8_final.md` §C/§D.
5. New Avenues then tested the duration/state-variable story (dwell, joint rarity, scale survival, observers, CTM, ...) and **killed all of it**. Second sweep tested repair-vs-damage arbitration and **killed all of it**. The proposed mechanism of the attractive story has been falsified; the length/eligibility alternative has not.

If this alternative is right, Waves 7-8 and New Avenues were not "failing to extract a real causal gap." They were trying to distill an illegal or non-causal functional of series length. That would **resolve** question 1 as artifact, and **explain** question 2 without a sixth distillation.

It would **not** automatically kill question 3 (CatBoost): a different tree library on a rank-21 bank can still have small complementary bias even if the future-info program is a dead end.

---

## Fragile Assumptions

1. **`research/current` is the authoritative "current" open-question list.** Fragile. It stopped on 2026-08-24. New Avenues pilots, CatBoost, CSA-04/04R, and data-forensics all postdate it. Scout E1/E2 over-weights a superseded STATUS.
2. **Ledger `status=NEW CHAMPION` is a live scientific claim.** Fragile. Canonical RESULTS.csv is an untracked wave-1 fossil. Later `EXPERIMENT_ID_MAP.md` reclassifies RT-131 as ORACLE/ILLEGAL and allocates RT-250. The status field was never rewritten. [ARTIFACT-VERIFIED vs REPORT-CLAIM contradiction that is documentary, not empirical]
3. **"Pooled OOF ≈ 0.63828" is the internal champion score.** Fragile / **[CONTRADICTED]** by the same document family that quotes it. It is fold-0. 5-fold mean is 0.62581.
4. **Arm C's +0.071 is "future break evidence."** Fragile. The operational definition is "final-row feature vector," which is not the same object.
5. **Wave 8 "all KILL" on fold-0 generalizes.** Moderately robust *as kills* (fold-0 early-stop is conservative against false promotion) but fragile as a claim that *no* future-aware mechanism can work — only that these five failed on fold 0, full pop.
6. **CSA-04 `marginal_vs_clone` is the binding ensemble metric.** Fragile. Project's own rule (Wave 6/7) is E2 vs RT600+seed-clone **and** E2-E0. CSA-04R shows the two rankings disagree: `marginal_vs_clone` prefers k=5; E2-E0 prefers k=2-3 and cannot distinguish them.
7. **"best k" after seeing the hybrid curve is a confirmation.** Fragile. Wave 1 already measured honest LOFO subset selection **losing** to the equal average (-0.0008) while hindsight subset "won" +0.0003. CSA-04 repeats that pattern.
8. **Effective rank 21 => new features cannot help.** Overstated. It does explain anti-stacking of *same-family* columns. It does not forbid a different learner's split policy (CatBoost) from using the same 21-dimensional cloud differently.
9. **RT-100 being n_online-clean => the seven-stream ensemble is n_online-clean.** Fragile. Only RT-100R was screened. Other streams are different hyperparameters/subsets on the same bank; label-`n_online` correlation could still be exploited differently. Unrun, not contradicted.
10. **Sibling STATUS files are independently maintained.** Fragile. Several (`gpu-tabular`, `deep-ensemble-frontier-2026`) copy CatBoost STATUS verbatim, including "this worktree is `research/catboost-specialist-2026`." Copy-paste, not independent confirmation.

---

## Alternative Explanations

Covered in "Strongest Alternative Explanation" and "Alternative Mechanisms" below. The one-line version: most "open contradictions" in the manifest are **stale documents**; the remaining scientific story ("huge future-info gap, distillation just hasn't found the encoding") is **length/eligibility confounding plus selection on CatBoost k**.

---

## Metric / Population Risks

- **TS-AUC vs series-level AUC vs cell AUC vs fold-0 vs 5-fold mean vs LB.** The repo constantly juxtaposes these. Protocol §4 violations are the default, not the exception.
  - Weekend harness 0.5988 != wave champion 0.615. Different metric. [ARTIFACT-VERIFIED]
  - Oracle frontier 0.6497 FULL is **series ROC-AUC**, not TS-AUC. Do not subtract from 0.62581.
  - W7-D3R +0.071 is **dominant-cell TS-AUC**, ~50.5% of pair weight; translated pooled +0.036.
  - T2 +0.00943 is **standalone vs RT-990**, not vs RT-600 ensemble. Ensemble marginal was +0.00024. [REPORT-CLAIM]
  - CatBoost `marginal_vs_clone` +0.00593 != E2-E0 +0.00238. [ARTIFACT-VERIFIED / REPORT-CLAIM]
- **Dominant cell** (t>=200, age>=100, ~50.5% pair weight, ~45% remaining loss) is where W7-D3R and Wave 8 bind. Young ages (0-20) are 11% of weight. "Early detection" briefs are **[CONTRADICTED]** by W5-D1. That contradiction is **resolved**, not open.
- **Eligible vs full population** in Wave 8 is a first-class population split. Mixing them is how a +0.008 looks like a failed distillation.
- **Lockbox vs dev vs public LB vs private.** RT-100 lockbox 0.60791 vs 0.61510 (-0.007). 4-stream lockbox 0.61214 vs 0.62374 (-0.012). Public LB 0.6268 vs 5-fold 0.62581 is *not* a haircut. Private-set composition is untested; redteam strata say n_online tertiles move the champion **0.582 / 0.604 / 0.640** (+/-0.02-0.03). [REPORT-CLAIM] That dwarfs remaining modeling alpha (~0.002).

---

## OOF / Leakage Risks

- **RT-131** within-timestep rank average: illegal under series-sequential single-pass runner. Later reports agree. Ledger still says NEW CHAMPION. **Resolved scientifically; stale status field.**
- **RT-900 VOID**: label leak via missingness mask. Correctly retired. Do not recycle the ID.
- **RT-992/993**: outer-fold contaminated teacher. Nested correction `RT-994/995` exists. Contaminated T1 was **overstated by +0.0235** on fold 0 (true clean effect negative). [REPORT-CLAIM] Any "teacher works" claim that cites 992/993 is invalid.
- **RT-991 Arm C**: explicitly non-causal. The leakage question is not "did they ship it?" (they didn't). It is "does the diagnostic number smuggle `n_online`?" **Unanswered.** `RT-991.npy` appears missing, which itself is an artifact-quality problem.
- **Wave 8 students**: causality tests claimed green before scoring (`wave8_final.md` §C). Not code-verified here. Two real bugs (unstandardized target; `NaN * mask != 0`) were found and fixed *during* the wave. Reported numbers are from the corrected run — good — but this is a reminder that Wave 8 scores are not mechanically above suspicion.
- **n_online standing gate** exists as a recommendation, not as an automated test on every new stream. CatBoost/New Avenues streams were not shown to have been screened.
- **Calibration:** deployable path is frozen cross-fitted SmoothTimeCDF. Rank averaging was the illegal path. Logit average (RT-160) was a parallel-lineage legal substitute. Residual calibration mismatch (grids from 8k, boosters from 10k) is acknowledged in `FINAL_ARCHITECTURE_FREEZE.md` §5 as smaller than status quo, not zero. Not a top-3 question.
- **OOF assembly:** T2 E0/E1/E2 fold-0 numbers (0.63828/0.63859/0.63882) sit in the same ledger column as 5-fold means. `RT-1100` mean=pooled=0.63882. Anyone ranking by `mean_oof_ts_auc` will crown fold-0 blends. **OOF assembly / column semantics error**, not a new model.

---

## Selection-Bias Risks

Protocol §8 is the right lens here. Approximate experiment counts in ledgers: 78 (wave 1) -> 104 (wave 2) -> 211 (`research/current`) -> 220 (wave 8) -> 245 (new avenues) -> 267 (catboost) -> 273 (deep ensemble). Hundreds of variants, same folds reused, thresholds and blend rules changed after seeing outcomes.

Specific selection artifacts that **look like open questions but are not**:

- **RT-131 NEW CHAMPION**: selected as champion, later ruled illegal. Not unresolved.
- **Three champions**: RT-160/190 (multi-agent, 2026-08-19), RT-150/250 (wave 2), RT-600 (production). Sequential supersession plus an ID collision (`RT-150` = rejected gating in one lineage, deployable ensemble in another). `EXPERIMENT_ID_MAP.md` and `HANDOFF_WAVE3.md` already explain this. **Not a scientific disagreement.**
- **RT-125R "could not reproduce" vs ledger 0.617411:** `EXPERIMENT_ID_MAP.md` §4, the R-suffix trap. `RT-125R` is **not** a reproduction of `RT-125` (different `n_estimators`, lr, leaves, `lambda_l2`, `max_bin`). Only RT-100R and RT-123R carry wave-1 configs. The tension is a naming bug, not a failed reproduction of a locked config.
- **T2 +0.00943 standalone vs +0.00024 ensemble:** the project already applied the right control (seed clone). Resolved as MOSTLY REDUNDANT. `LEADERBOARD_ASSAULT_STATUS.md` saying ensemble-integration is "not yet run" is **stale**.
- **New Avenues "nothing executed":** contradicted by RT-1200-1233. Resolved negative.
- **CSA-04 best-k=5 MAJOR:** k searched after seeing single-slot results; ranking used `marginal_vs_clone`; STATUS files still quote +0.00593. CSA-04R, preregistered, using E2-E0 and parsimony, selects k=2 and says extra slots are not distinguishable. **This is the live selection-bias question (top-3 #3), not a reason to believe +0.006.**
- **Screen->full reversals** (context -0.0177, pairwise, DGP gating): already internalized as a rule (NEW_AVENUES D.6). Wave 8 and New Avenues mostly respected "full pop / ensemble marginal." CatBoost hybrids still need that discipline at confirmation.

Cherry-picking which failures to write up as "interesting open questions" **is** happening in STATUS copy-paste: RT-1264 MAJOR is advertised after CSA-04R already killed the extra-k claim. That is selective reporting of the more exciting number.

---

## Statistical Stability Risks

- Fold SD ~0.009 makes **single-fold** deltas under ~0.01 uninterpretable. Wave 8 pilots are fold-0. New Avenues first sweep is fold-0 screen. Kills at fold-0 with negative pair flow are still informative (they failed a conservative gate). **Positive** fold-0 results would not be.
- T2 nested: +0.00943, 5/5, CI [+0.0043, +0.0137] vs RT-990 — internally solid as a single-model effect, then **vanished** at ensemble (+0.00024). Magnitude without the seed-clone ensemble control is the trap. CatBoost reporting currently includes that control; good. CSA-04's `marginal_vs_clone` still inflates vs E2-E0.
- CSA-04R bootstrap: k=2 CI excludes 0; k=4,5,6 CIs include 0; k=3 minus k=2 CI **includes 0** [-0.00068, +0.00132]. Adjacent-k differences are inside noise. k=2 fold deltas: +0.00215, +0.00029, +0.00121, +0.00324, +0.00324 — fold 1 is +0.00029 (noise), fold 3/4 carry the mean. **Not a single-fold artifact, but concentrated.** Data-forensics: largest fold contribution is fold 3; `rel_10_25` **loses** -0.00363. Sign-stable overall, slice-fragile.
- Partition-draw SD 0.0050: any claimed +0.002 that has not seen alt partitions is inside one partition-draw SD. T2 leg 4 (alt partitions) was **never run** (resource-cost rationale in RDOF). CatBoost has no alt-partition result either.
- Paired series bootstrap is the right unit (redteam/CSA-04R use it). Row bootstrap would overstate precision. Not a current reporting failure for the later waves.
- Apparent "unresolved contradiction" between 0.63828 and 0.6268 is **within a fold-mean vs fold-0 confusion**, not statistical disagreement.

---

## Alternative Mechanisms

For the surviving questions, competing mechanisms:

**Q1 (Arm C gap)**

| ID | mechanism | currently favored? |
|---|---|---|
| M1 | Genuine future-path break evidence, causally unrecoverable from prefix | Attractive; not decomposed |
| M2 | `n_online` / endpoint / final-row time-calibration (illegal length oracle) | Strongest mundane |
| M3 | Series-identity fingerprint of the full sequence (legal at t=final, illegal at t) | Related to M2 |
| M4 | Teacher capacity + extra 500 columns, not "future" per se (Arm C vs B also changes representation size) | Weaker: B already raised capacity and lost |

**Q2 (Wave 8 eligible vs full)**

| ID | mechanism | currently favored? |
|---|---|---|
| N1 | Distillation works but eligible mass is too small a share of pair weight (ORR's own reading) | Partial, documented |
| N2 | Remaining-length / eligibility confounding; "success" on eligible pop is a length proxy | Compatible with M2 |
| N3 | Missing duration state variable | **Falsified** by first sweep |
| N4 | Missing repair-vs-damage arbiter | **Falsified** by SS-01..04 |
| N5 | Student capacity / objective mismatch (CFEP future-MSE lost to BCE control) | Local to CFEP; does not explain SST eligible success |

**Q3 (CatBoost)**

| ID | mechanism | currently favored? |
|---|---|---|
| P1 | Real complementary learner bias on a rank-21 bank (ordered boosting vs GOSS/leaf-wise LightGBM) | Possible at k=2, +0.002 |
| P2 | k-search + `marginal_vs_clone` selection artifact | Explains RT-1264 MAJOR; does **not** fully explain k=2 CI-excluding-0 |
| P3 | Fold-3 / mature-vs-prebreak slice luck, will not transfer | Open; forensics shows that slice is the gain location |
| P4 | Silent `n_online` correlation in CatBoost scores | Unrun |

---

## Falsification Tests

No new training. Existing artifacts only.

**T1 — Arm C vs length ($0, may be blocked).** If `RT-991.npy` can be located (or rebuilt from a stored checkpoint without a new research decision), compute within-t AUC of Arm C scores as a predictor of `n_online` and of `(t+1)/n_online`. Residualize Arm C against those, re-score dominant-cell TS-AUC vs Arm B.
- If residual cell delta ≈ 0: M1 dies, M2 wins, Q1 is **resolved as artifact**.
- If residual cell delta remains ~0.07: M2 dies, Q1 remains as a true information gap.
**Blocker:** `RT-991.npy` not found in this pass. JSON `wave7_d3r.json` cannot answer this.

**T2 — Wave 8 SST remaining-length ($0, artifacts present).** `structural-break-wave8/research/oof/` contains `RT-1006.npy` (legal SST) and `RT-1007.npy` (oracle SST). Correlate legal scores with remaining length `n_online - t` inside the eligible slice and on the full pop. Residualize; recompute the two cell deltas already reported (+0.00810 eligible, -0.00060 full).
- If eligible +0.008 dies after residualization: N2 wins, Q2 is **resolved as length confounding**.
- If eligible +0.008 survives and full-pop remains ~0: N1 (coverage) wins; Q2 becomes a pair-weight arithmetic fact, not an open mechanism.

**T3 — CatBoost k=2 vs k=5 using existing OOF ($0).** Deep-ensemble-local `research/oof/` has `RT-1254..1263.npy` plus specialist `RT-300/410-415.npy`. Reproduce CSA-04R E2-E0 curve (already claimed exact at 1e-9). Additional cheap checks that CSA-04R did not emphasize: (a) within-t AUC of CatBoost hybrid scores -> `n_online`; (b) drop fold 3 and recompute E2-E0; (c) `rel_10_25` damage persistence.
- If n_online AUC ~0.50 and fold-3-dropped delta still ≳ +0.001 with CI>0: P1 strengthened.
- If n_online AUC >> 0.50 or fold-3-dropped delta crosses 0: P3/P4, Q3 is not promotion-worthy.

**T4 — n_online screen of the other six LightGBM streams ($0 if their OOF exist).** Wave7-promotion `research/oof/` has `RT-300` and `RT-410-415` (and alt partitions). Repeat redteam §1 table. Completes a documented incomplete audit. A clean result does **not** answer Q1 (Arm C is a different object). A dirty result would invalidate production more than it would open a research question.

**T5 — composition metadata on `X_test.reduced` if legally observable without labels ($0).** Redteam already asked for this. n_online / n_hist / kurtosis mix vs dev. If the test mix sits in the 0.58 tertile, remaining +0.002 modeling alpha is irrelevant. This is decision-relevant, not a model-class question; I do **not** rank it top-3 as a *research* question.

Tests I will **not** call falsifiers of "unresolved":

- Re-litigating RT-131's ledger status.
- Retraining T2 alt partitions (forbidden here; also T2 ensemble is already redundant).
- A sixth future-aware distillation.
- Tuning `RT-1216` CTM weights (explicitly closed).

---

## Predictions That Distinguish Competing Explanations

| observation | favors |
|---|---|
| Arm C within-t AUC(score -> n_online) ≳ 0.70, residual cell delta ~0 | M2 (length oracle); Q1 not a real gap |
| Arm C n_online-AUC ~0.50, residual cell delta still ~0.07 | M1 (true future path); Q1 remains, but Wave 8/New Avenues still failed to capture it |
| SST legal eligible delta dies after remaining-length residualization | N2; Wave 8 "open question" is eligibility confounding |
| SST legal eligible delta survives; full-pop delta explained by eligible mass x pair weight | N1; coverage, not mechanism mystery |
| CatBoost k=2 E2-E0 survives drop-fold-3 and n_online screen | P1; small real learner diversity |
| CatBoost k=2 dies without fold 3, or predicts n_online | P3/P4; do not promote |
| k=5 beats k=2 outside noise on a **predeclared** endpoint | would revive CSA-04 MAJOR; currently **contradicted** by CSA-04R |
| Any new stream with n_online-AUC >> 0.50 | production number partly illegal; research pause |

---

## What Would Change My Mind

**I would drop Q1 from the top 3** if someone produces the Arm C vs `n_online` residualization and the residual is still large (then the gap is real, and the leftover question is only capture — which Waves 8 + New Avenues already failed, making further capture a new-training problem this investigation is forbidden to propose). I would also drop it if `RT-991.npy` is confirmed destroyed and no reconstruction is possible **and** Wave 8 SST residualization (T2) already shows the same length confounding, because Q1 and Q2 would have collapsed into one answered mechanism.

**I would drop Q2** if T2 (SST residualization) is run on existing OOF and clearly attributes the eligible/full split to remaining length or to pair-weight arithmetic. Wave 8's prose would then be a solved autopsy, not an open question.

**I would drop Q3** if one treats CSA-04R as already decisive (k*=2, extra k not distinguishable, RT-1264 MAJOR retired) **and** treats +0.002 E2-E0 with CI excluding 0 as a *resolved small positive* awaiting only deployment engineering. I keep it because (i) STATUS/forensics still advertise RT-1264 MAJOR, so the repo currently **disagrees with itself**, and (ii) fold-3 concentration + no alt-partition + no n_online screen on CatBoost scores means the +0.002 is not yet a confirmed scientific fact. Finding that drop-fold-3 still yields CI>0 and n_online-AUC~0.50 would move Q3 from "unresolved" to "small confirmed internal alpha, promotion is an engineering decision."

**I would not be moved** by: updating the RT-131 status field; finding another champion ID in an old STATE_OF_RESEARCH; the weekend-harness 0.5988; T2's standalone +0.009; NEW_AVENUES' nine untested functional classes (they were tested and killed on the pilots branch).

---

## Final Assessment

A naive reading of `01_EVIDENCE_MANIFEST.md` will nominate: (1) RT-131 champion vs illegal oracle, (2) three divergent champions / RT-150 collision, (3) Wave 8 "why doesn't future info survive," possibly with RT-125R and the 0.638 vs 0.6268 "gap" as runners-up.

**I reject (1) and (2) as unresolved quantitative research questions.** They are superseded documentation. The production champion is RT-600 at 0.6268 external / 0.62581 5-fold internal. ID collisions are mapped. Provenance "collapse" is a stale checkout, not missing science.

**I keep a sharpened form of (3), split into content-of-the-oracle vs eligible-pop collapse, and I add the CatBoost self-contradiction that the scout could not see because it postdates `research/current`.**

New Avenues' duration hypothesis does **not** survive: it was executed and killed. Treating `NEW_AVENUES_2026.md` as the current open-question list is itself a stale-artifact error.

If Q1 residualization shows Arm C ≈ length oracle, the correct research conclusion is **saturation of legal causal information**, not "a 0.07 hole waiting for the right student." Remaining decision risk is then private-set composition (+/-0.02-0.03) and whether CatBoost's +0.002 is worth deployment cost — not another representation wave.

The original Wave 8 capture question **survives scrutiny only as a question about what Arm C actually measured**, not as a license to keep distilling.

---

## Ranked top-3 unresolved quantitative research questions

### 1. Information content of W7-D3R Arm C: future-path evidence or length/endpoint oracle?

**Precise statement.** Of the dominant-cell TS-AUC gap C-B = **+0.07110** (`RT-991` vs `RT-990`, 5/5 folds, cell pair-weight 0.5050), what fraction is explained by non-causal series length / final-row existence / `(t+1)/n_online` versus residual future-path break evidence that a legal prefix could in principle approximate?

**Strongest surviving evidence.**
- [ARTIFACT-VERIFIED JSON] `research/reports/wave7_d3r.json` (on `research/current` @ `aca2c4f` and copies): C 0.71859 vs B 0.64749, per-fold C-B +0.053 to +0.082.
- [REPORT-CLAIM] `wave7_d3r.md`; tag `wave7-d3r-information-frontier`.
- [ARTIFACT-VERIFIED] `RESULTS.csv` `RT-991` whole-dev 0.72003, notes "NEVER a production candidate."
- [REPORT-CLAIM] redteam: `n_online` oracle 0.62948; RT-100 does not predict n_online.
- [REPORT-CLAIM] oracle frontier: legal model matches/beats known-boundary teacher through h=150; FULL +0.0396 series AUC.
- Negative capture: Wave 8 five KILLs (`wave8-future-aware-final`); New Avenues first+second sweep all KILL (`research/new-avenues-pilots-2026` @ `b47b22a`).

**Why it survived scrutiny.** The *existence* of a large Arm C number is not in doubt. The *interpretation* as causally relevant future information is. Later waves treated M1 as given and spent the next week falsifying capture mechanisms. Nobody residualized Arm C against the already-priced length oracle. That is the actual hole. It is not documentation staleness: EXPERIMENT_ID_MAP, STATUS, and Wave 8 final still bind decisions to this number.

**What would make it NOT unresolved.** T1 residualization ≈ 0 => resolved as artifact (M2). Residualization ≈ full gap => resolved as true information limit (M1); leftover work is capture, which this investigation cannot propose as training. Missing `RT-991.npy` with no substitute => Q1 stays open but may be **unanswerable** from current artifacts; then Q2's SST OOF becomes the proxy.

**$0 diagnostic.** Locate or confirm destruction of `RT-991.npy`. If present: within-t n_online AUC + residual cell TS-AUC vs `RT-990.npy` (present in wave7-promotion `research/oof/`). If absent: skip to Q2's SST vectors.

---

### 2. Wave 8 eligible-population retention vs full-population collapse

**Precise statement.** Why does the SST legal student show dominant-cell delta **+0.00810** on the eligible subpopulation and **-0.00060** on the full population (RT600+SST marginal vs clone **-0.00031**), given that eligibility itself is class-conditional (`P(eligible|y=1)=0.33` vs `0.81` at h=200, t<20)? Is the eligible "success" remaining-length confounding, or real future-signal that is too small a share of pair weight to move the metric (the ORR pattern: repair P=0.737, metric delta -0.00005)?

**Strongest surviving evidence.**
- [REPORT-CLAIM] `structural-break-wave8/research/reports/wave8_final.md` §C, §D, §E, §I, §29; SHA `589e1db`; tag `wave8-future-aware-final`.
- [ARTIFACT-VERIFIED] wave8 `RESULTS.csv` rows `RT-1006` 0.62326 (legal SST), `RT-1007` 0.63084 (oracle SST), `RT-1021` 0.63823 (ORR blend), and the other pilots.
- [ARTIFACT-VERIFIED] OOF files `RT-1006.npy`, `RT-1007.npy` present under `structural-break-wave8/research/oof/`.
- Subsequent falsification of the documented proposed answers: duration/state (first sweep) and arbitration (SS-01..04). [ARTIFACT-VERIFIED ledger + REPORT-CLAIM 2026-08-25 STATUS]

**Why it survived scrutiny.** Wave 8 *explicitly* left this as the open question. Unlike RT-131, this was not later reclassified. Unlike New Avenues' duration story, the eligible/full split was **not** itself tested as a length confound. ORR already showed "real row-level mechanism, zero metric." SST showed "works where future exists, fails where the metric must score rows with no future." Those are quantitative, competing, $0-testable mechanisms. Treating "try a sixth distillation" as the leftover question would ignore the eligible-pop number Wave 8 already printed.

**What would make it NOT unresolved.** T2 residualization of SST legal scores against remaining length. If eligible delta vanishes, Q2 is answered (N2). If eligible delta survives and full-pop failure is pair-weight arithmetic, Q2 is answered (N1). Either way it leaves the top 3.

**$0 diagnostic.** Using existing `RT-1006.npy` / `RT-1007.npy` plus store metadata (`n_online`, `t`, folds): remaining-length correlation; residual cell TS-AUC on eligible vs full pop; pair-weight share of the eligible slice. No retraining.

---

### 3. CSA-04 "MAJOR" `RT-1264` vs CSA-04R "NOT_DISTINGUISHABLE" `RT-1265`/`RT-1257`

**Precise statement.** After LightGBM-capacity, T2-ensemble, Wave 8, and New Avenues all returned ~0 ensemble alpha, CatBoost slot replacement produced two incompatible headlines on the same OOF: (a) CSA-04 best-k=5 `RT-1264` **MAJOR**, `marginal_vs_clone = +0.005934643`, E2-E0 `+0.002378478`, 5/5 folds; (b) CSA-04R greedy-on-E2-E0, parsimony `k*=2` = `RT-1257`/`RT-1265`, E2-E0 **+0.002026**, bootstrap CI [+0.00043, +0.00354], extra slots **NOT_DISTINGUISHABLE** (`delta_noise=0.0011`; k=5 E2-E0 CI includes 0). Which quantity is the confirmed complementary-learner effect, and does even k=2 survive n_online screening and leave-fold-3?

**Strongest surviving evidence.**
- [ARTIFACT-VERIFIED] `deep-ensemble-frontier-local` `RESULTS.csv`: `RT-1257` PROMOTION_WORTHY 0.627838; `RT-1264` MAJOR 0.628190; `RT-1265` NOT_DISTINGUISHABLE 0.627838 (same score as 1257).
- [REPORT-CLAIM] `CSA04_FINAL.md` (SHA `766ecf0`, 2026-08-28); `CSA04R_REANALYSIS.md` (SHA `98a1a9a`, same day, B=2000 series bootstrap).
- [REPORT-CLAIM] `DATA_FORENSICS_REPORT.md`: reconstructed RT600 0.625811342 vs RT-1264 0.628189820, fold deltas all positive, strongest `mature_vs_prebreak` +0.00515, **loss** on `rel_10_25` -0.00363; bank effective rank 21.282.
- [ARTIFACT-VERIFIED] specialist/CatBoost OOF `.npy` present in that worktree's `research/oof/`.
- Contrast class: T2 E2-E1 +0.00024 (MOSTLY REDUNDANT); New Avenues max +0.000937 killed; Wave 8 ~-0.0003. [ARTIFACT-VERIFIED / REPORT-CLAIM]

**Why it survived scrutiny.** This is not a stale ID collision. Two same-day, same-OOF analyses disagree, and current STATUS/forensics still lead with (a). k-search plus the wrong endpoint (`marginal_vs_clone`) is exactly the selection bias Wave 1 already paid for. CSA-04R is the more disciplined reading, but k=2's CI excluding 0 plus 5/5 folds means I cannot dismiss CatBoost as pure noise either. Fold-3 concentration and an unrun n_online screen keep it unresolved as *science*, not merely as a deploy/no-deploy ticket.

**What would make it NOT unresolved.** (i) Repo-wide retirement of the RT-1264 MAJOR headline in favor of CSA-04R, **and** T3 showing k=2 n_online-clean and drop-fold-3-stable => resolved small positive (promotion becomes engineering). (ii) T3 showing k=2 fails those checks => resolved negative / UNVERIFIED, do not promote. (iii) Treating deployment cost as the only leftover — that is not a quantitative research question.

**$0 diagnostic.** Recompute E2-E0 with existing `RT-1254/1255/300/413` OOF (k=2 is those two replacements). n_online screen; drop fold 3; `rel_10_25` pair flow. Do not search k again. Do not train.

---

### Explicitly rejected as top-3 (so a naive manifest reading does not sneak them back)

| candidate | why rejected |
|---|---|
| RT-131 NEW CHAMPION vs illegal oracle | Stale ledger status; mapped in `EXPERIMENT_ID_MAP.md`; not a live empirical dispute |
| Three divergent champions / RT-150 collision | Sequential supersession + ID collision; RT-600 is production |
| 0.63828 vs LB 0.6268 | Fold-0 vs 5-fold mean 0.62581; transfer is flat |
| RT-125R reproducibility | R-suffix trap; not a locked reproduction |
| Metric weighting unconfirmed | Closed 2026-08-19 |
| Early-detection vs pair weight | W5-D1 resolved (11% / 57%; pairwise_w -0.00147) |
| T2 ensemble integration "not run" | Run; +0.00024; MOSTLY REDUNDANT |
| NEW_AVENUES "nothing executed" / duration state | Executed; killed |
| Provenance collapse as a research question | Process/docs, not a quantitative claim |
| Weekend harness 0.5988 | Wrong metric |
| m03_dyn unexplained gain | Gain!=AUC (Spearman 0.163); no metric claim |
| Other-6-stream n_online screen | Incomplete audit, worth T4, not a top scientific question unless dirty |
| Private-set composition +/-0.02-0.03 | Decision risk; untestable without (legal) test metadata; not a model question |
