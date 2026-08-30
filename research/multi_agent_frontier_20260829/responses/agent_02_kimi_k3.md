I now have everything I need, including one genuinely new measurement no prior agent had. The write/execute lockdown means I'll deliver the full report inline. Here it is.

---

# Independent Frontier Analysis — 2026 ADIA Lab / CrunchDAO Structural Break Challenge (Real-Time)

**Agent identity:** Kimi (kimi-k3), one of five independent agents
**Intended output file:** `research/multi_agent_frontier_20260829/responses/agent_kimi_k3.md` — **write blocked by environment permissions; delivered inline in full**
**Independence confirmation:** `responses/` contained only `.gitkeep` when inspected; no other agent response, note, or conclusion was read. Both source prompts were read in full first. Repository forensics covered `research/current`, `wave7-teacher-distillation`, `wave7-t2-promotion`, `wave8-future-aware-distillation`, `catboost-specialist-2026`, `learner-diversity-2026`, `gpu-tabular-2026`, `deep-ensemble-frontier-{2026,local,crunch}`, `causal-representation-frontier-2026`, `engineering/rt1257-deployment-2026`, `codex/oracle-information-frontier-2026`, `research/investigation-20260829-unresolved-questions`, plus the on-branch ledgers and reports.
**Analysis only:** no model code modified, no training run, no RT IDs allocated, `RESULTS.csv`/prompts/`STATUS.md` untouched. I ran exactly one *read-only* rescoring of existing OOF vectors (same category as `wave7_d0_exact_loss_cube.py`: "no training, no new model, pure recombination + exact scoring of vectors already on disk"); it reproduced every recorded number it touched to 5 decimal places before producing one new result (§2, R9).

---

## 1. Executive diagnosis

**What the project does well.** The RT-600/RT-1257 system is the mature endpoint of a correctly-run program: a 500-column causal feature bank built on per-series historical-null calibration at matched window length (the single most valuable idea in the repo), seven LightGBM specialists, a parameter-free time-conditional (SCDF) blend that matches the illegal rank oracle (0.62581 vs 0.62580), bitwise prefix-invariance, and a disciplined preregistration culture that has caught two real leaks (RT-131's illegal cross-sectional rank average; the W7 pilot's outer-fold contamination, worth +0.0122–0.0235 of fake alpha). External: 0.6268 (RT-600), 0.6290 (RT-1257).

**The bottleneck, classified:**

| axis | verdict | evidence |
|---|---|---|
| Learner | **Not the bottleneck, but not zero.** CatBoost's ordered-boosting/oblivious trees on the *same 500 columns* gave the only promotion in two waves (RT-1257, E2−E0 +0.00203, externally +0.0022). More capacity on the same columns is negative (Arm B −0.00592). | measured |
| Feature-bank saturation | **Saturated within its own family.** 79 mechanisms/15 families across two preregistered sweeps: zero confirmation candidates. But effective rank 21.3/500 and Spearman(gain, metric value)=0.163 say the bank is *redundant*, not *complete*. | measured (rank figure: report-claim) |
| Representation | **Largely closed for this family.** The representation×objective factorial is empirically filled and empty (CRF: RT-970 0.52618 → RT-1235 0.57054 → RT-1234 0.59276; sum still short). Best low-ρ point moved and the answer didn't change. | measured |
| Objective | **Real but an order too small.** Isolated objective effect +0.0222 (whole), +0.0371 (cell). Not the lever. | measured |
| **Training target / teacher engineering** | **Least exhausted, most promising.** The only standalone alpha in two waves came from a target, not a feature or learner (T2 +0.00943, 5/5 folds, CI [+0.0043,+0.0137]) — and it shrank ~40× at ensemble because the teacher was contaminated with *untransmittable* information (my §2/R9 measurement). | measured + new inference |
| Arbitration/routing | **Dead.** SS-01, SS-04, L3 (14 gating rules), DGP-gating, all stacking/weighting — every attempt to know *when to trust* a non-incumbent signal failed. | measured |
| Historical-null estimation | **Settled, and the answer is the incumbent.** The fixed per-series null beat a learned amortized null (−0.0214) and an affine-adapted null (−0.0215). | measured |
| Finite training data | **Untested lever.** 2025→2026 transfer "not attempted at all" — the only axis that adds *data*. | measured absence |
| **Causal-information ceiling** | **Real for a large share of the oracle gap — and now quantified.** My new measurement (R9): **~40% of Arm C's +0.07110 cell lift is endpoint/`n_online` artifact; ~60% is genuine future-path information; on the never-break-negative side (74% of cell loss) ~82% is genuine.** | measured + arithmetic inference (R9) |

**One-paragraph synthesis.** The program ended on "CASE 2 — future-information limit" (W7-D3R) and CRF closed representation/objective/null. The unresolved-questions investigation correctly identified that nobody had decomposed Arm C's +0.07110 into endpoint vs genuine future information because `RT-991.npy` was "unverified or missing." I found the vector present and internally consistent in two sibling worktrees (sha256 `e0b726ea…`, exactly 4,032,524 finite values = the dev population) and ran the decomposition read-only. Result: the skeptic's strongest mundane hypothesis ("the oracle gap is mostly series-length/endpoint artifact") is **refuted for the loss-dominant population** (never-break negatives: ~82% of the oracle lift survives `n_online`-matching) but **confirmed for the pre-break-negative side** (~89% of that side's lift is endpoint). The consequence: the dominant remaining question is not "is there future information" (there is, genuinely) but "**is the future path of a weak excursion predictable from its prefix**" — a *shadowability* question — and the only two levers with unexhausted headroom are (a) teacher/target engineering that transmits the *genuine* component instead of the confounded whole, and (b) more training data.

---

## 2. Repository evidence

Grading: **MF** = measured fact verified against repo artifact or reproduced by me; **RC** = report-claim (read, not reverified); **INF** = my inference.

| # | Evidence | Grade | What it establishes |
|---|---|---|---|
| R1 | RT-600 dev OOF 0.62581 (pooled 0.625627); external 0.6268 (LB-001). I reproduced pooled 0.6256269256734708 from `wave5_S_specialist.npy`. | **MF** | Production anchor; my harness is exact. |
| R2 | RT-1257 (CatBoost CAT-413+CAT-300 replacing slots 413/300): `marginal_vs_clone` +0.002407, E2−E0 +0.002026, 5/5 folds, dominant net +158, mature-vs-never +73; external 0.6290 (submission #16). k=3–6 not distinguishable from k=2 (CSA-04R, δ_noise 0.0011). CatBoost per-slot inference 3.27× LightGBM; k=7 all-CatBoost projects 5.08 h/10k — in budget. | **MF** (external: single observation) | Learner diversity on the *same* representation pays a little; breadth beyond k=2 unproven. Folds 3–4 supply 64% of deltas (RC) — partition fragility unclosed. |
| R3 | Dominant cell (t≥200 ∧ age≥100): 45.29% of inversion loss, 50.50% of pair weight, cell AUC 0.66428, never-break negatives 74.0% of loss. Loss is *diffuse* across series (worst 5% of series carry 0.16–0.20 of loss — NEW_AVENUES §E.1). Reaching 0.640 requires repairing 7.79% of the cell's inversions. | **MF** (I re-verified cell AUC 0.664277 on the cell) | The metric is won in one cell, against never-break negatives, with no pathological-series shortcut. |
| R4 | W7-D3R: Arm B (more capacity, same 500 legal cols) −0.00592 cell (1/5 folds); Arm C (each series' own final-row features broadcast) **+0.07110 cell, 5/5 folds, every fold ≥ +0.053**. I reproduced all six arm AUCs to recorded precision (A 0.65341/B 0.64749/C 0.71859). | **MF** | Capacity is not the bottleneck; a full-sequence oracle knows a lot more. What it knows was never decomposed — until R9. |
| R5 | Wave 7 T2 (distill Arm-C teacher Q, 0.5y+0.5Q, nested-pure): standalone **+0.00943, 5/5, CI [+0.0043,+0.0137]**; ensemble marginal over a seed clone **+0.00024**; T2↔specialists ρ 0.71–0.91, ↔seed-clone 0.9099. Contaminated pilot overstated by +0.0122. | **MF** | Real single-model alpha that the ensemble already knew. Teacher transmits ~1/40th of itself. |
| R6 | Wave 8 (SST, ORR, PCFB, CFEP, TGMC): all KILL on full population (marginals ≈ −0.0003). PCFB retention R=0.071; ORR recoverability 0.737 but −0.00005 on the metric; SST +0.00810 on eligible slice / −0.00060 full-pop; eligibility class-conditional (P(eligible\|y=1)=0.33 vs 0.81). | **MF** | The future is real but barely predictable from the prefix as targeted by those five designs; "the future exists" is itself label-informative. |
| R7 | First sweep: any candidate repairs 91.8% of sampled dominant-cell RT-600 mistakes, damages 83.0% of correct pairs; dominant-net is the best marginal proxy (Pearson 0.879). Second sweep (SS-01…SS-04) all KILL. CRF-01 repairs 39.5% of cell mistakes, damages 26.8% of correct pairs. | **MF** | Repair-vs-damage symmetry is the binding constraint on every *integration*; detection is not scarce, reliability is. |
| R8 | Illegal oracles: `−n_online` alone 0.62948 dev (I reproduced 0.6294786959690457 exactly), 50/50 rank blend with champion 0.66625; champion clean (AUC 0.488–0.514 per t-band, but screened on only 1 of 7 streams — RC). Known-boundary oracle headroom ≈ 0 through h=150, +0.0396 only at FULL horizon. | **MF** (blend/champion-clean: RC) | Generator length structure is label-informative and unused; genuine future info needs *long* futures. |
| R9 | **NEW MEASUREMENT (this agent, read-only rescoring, 2026-08-29).** Within-(t × `n_online`-decile) matched AUC vs plain within-t AUC, dominant cell: C 0.71859→**0.68979** (Δ 0.0288); B 0.64749→0.64745 (Δ 0.00004); A Δ 0.00046; RT-600 Δ 0.00022; T2 Δ 0.00086; seed clone Δ 0.00074. **Arm C lift over B: raw +0.07110 → matched +0.04234 (survival 59.6%).** On the pre-break-negative cut: C 0.73070 (reproduces D3R exactly), matched 0.65345; B matched 0.64417 → lift +0.0879 raw → **+0.0093 matched (survival 10.6%)**. Whole-dev: C 0.71989→0.66973 (survival 54.0%). | **MF** (matched/unmatched numbers) + **INF** (never-break split below) | ~40% of the oracle gap is endpoint artifact (worthless causally); ~60% is genuine future-path info. The endpoint part lives almost entirely on the pre-break side — a uniform-τ conditioning artifact ("series that will break later are longer at fixed t"). **Inference by pair-weight arithmetic (weights 0.7425/0.2575, which reproduces the raw lift exactly: 0.7425×0.0653+0.2575×0.0879=0.0711 ✓): on the never-break side, matched lift ≈ +0.0538 of +0.0653 raw → ~82% genuine.** Caveat: decile-matching is an upper bound on removal of endpoint info (finer structure could explain more); genuine info correlated with length would bias the other way. A corrected direct never-break cut is a 3-line change (my first never-break mask erroneously excluded positives; discarded). |
| R10 | m05_ctx −0.0177 (memorization, permutation control −0.057); LA-03 adapted null −0.0215 vs fixed null; CRF-02 learned null −0.0214 vs fixed null with *passing* derangement control (+0.0034/+0.0084). E.2: history-only fingerprints predict never-break loss at ρ=+0.19 (permutation-controlled), R²<0. | **MF** | The fixed per-series null is the right null; history carries only a *weak monotone* difficulty signal. |
| R11 | m09_back (online suffix-vs-prefix contrast): mechanism falsified by its own age split; and the seed-clone control (+0.00496) beat the real module (+0.00371) in blend — "changing the seed decorrelates more than 51 new columns." | **MF** | Online-self nulls are dead; seed-clone controls are mandatory (I applied this standard to R9's controls). |
| R12 | Stacking/weighting/subset all lose to the plain average (LOFO logistic 0.62514, LOFO LGBM 0.62144, greedy −0.0008 vs 0.62544). Alphabot (2025) won *by* stacking — over four **disjoint feature philosophies**, not seven same-representation variants. | **MF** (ours) / method-document (theirs) | The stacking contradiction (evidence-brief Q-A) resolves as: stacking pays iff base learners are informationally disjoint. Ours aren't. |
| R13 | Lockbox haircut −0.0116 (4-stream, only clean dev→held-out read). Fold SD 0.0085; partition-draw SD 0.0050 (RC). | **MF** (lockbox) | Price every dev gain ×~0.98 and distrust <0.005. |

**Discrepancy notes (reporting as required):** (i) the canonical `RESULTS.csv` is a wave-1 fossil still labeling illegal RT-131 "NEW CHAMPION" — confirmed; use `EXPERIMENT_ID_MAP.md`. (ii) The 64.9/92.9 ms/point figures in the evidence brief are legacy *reference-streamer* numbers from `research/multi-agent-2026`; the shipped engine is ~1.6–3.6 ms/pt locally (RT-1257: 2.28 ms/pt, 3.69 h/10k projected — deployment is not compute-bound). (iii) `RT-991.npy` is **not missing**: present in `structural-break-wave6/research/oof/` (mtime matches its RESULTS.csv training timestamp) and `structural-break-wave8/…`, byte-identical, sha256 `e0b726ea…` (computed by my scout; no digest was ever pinned in git — a real provenance gap worth closing), finite mask exactly the 4,032,524 dev rows, values in [3.3e-7, 0.99996] consistent with the clipped teacher Q. The W7-D3R script can regenerate it but not bitwise-guaranteed (no LightGBM `seed` in ARM_B_PARAMS) — validate any regeneration against the recorded per-fold metrics, not bitwise.

---

## 3. Five high-upside mechanisms

### M1 — EPOD: Endpoint-Purged Oracle Distillation *(training-target surgery)*

- **Core hypothesis.** T2's teacher (Arm C) is, in the dominant cell, ~40% endpoint artifact and ~89% endpoint on the pre-break side (R9). A causal student *cannot* learn `n_online` from causal features — that component is pure label noise from the student's viewpoint, diluting the transmissible component. Purging it before distillation should transmit more per unit of teacher signal.
- **New information represented.** None — this is the only mechanism I propose that adds no information; it *removes confounding* from the best teacher the project has. It is justified as the direct, measured repair of Wave-7's diagnosed failure mode.
- **Why the 500-feature system doesn't contain it.** It's a training-target construction, orthogonal to the bank.
- **Why not falsified.** Wave 7 distilled the *raw* teacher; nobody measured what the teacher knows (R9 is the first decomposition). The hypothesis "purged teacher > raw teacher" is untested.
- **Causal inference state.** Student = unmodified 500 causal columns (assert `Xtr.shape[1]==500`). Teacher construction is offline-only. Inference identical to RT-600 specialists; deterministic; no cross-series state.
- **Training data / target.** Teacher Q′ = within-t residual of `clip(RT-991.npy)` on `n_online` (per-t decile-dummy projection, exactly my R9 construction), clipped back to (0,1). Student label = `0.5·y + 0.5·Q′`. Nested double cross-fitting exactly as `wave7_teacher_nested.py` (fold-purity sentinel mandatory).
- **Input/estimator.** 500 cols → LightGBM xentropy (parity-checked vs binary), T0 = RT-990 control.
- **Streaming inference.** Identical to a specialist.
- **Interaction with RT-600/1257.** If standalone beats T2, run the ensemble-integration battery (E0/E1/E2, seed-clone control).
- **Dominant-cell effect.** Positive by construction (purge concentrates signal on the cell's genuine component, which is never-break-dominated).
- **Early-break effect.** Neutral-to-positive (less horizon noise).
- **Pair repair / damage.** Repair via cleaner mature-vs-never separation; damage risk *lower* than T2's because the endpoint component was the least causal-transmissible part.
- **Compute.** ~1 h fold-0 pilot (nested Q recomputed once); ~5 h full nested if promoted. Fits easily.
- **Leakage risks.** The residualization uses only training-fold rows per inner fit; the purge uses no labels (unsupervised), so it cannot import label leakage — but the outer-fold-purity sentinel must pass.
- **Cheapest falsification.** Fold-0 nested pilot: EPOD standalone Δ vs T0, compared to T2's clean fold-0 (+0.00447).
- **Kill criterion.** Fold-0 Δ < +0.00447, or retention-decomposition shows the purged component no more prefix-predictable than the raw.
- **Promotion criterion.** Nested 5-fold ≥ T2's +0.00943 **and** ensemble marginal_vs_clone ≥ +0.0010.
- **Plausible upside / probability.** +0.002–0.006 dev if the genuine component transmits better; P(success) ≈ 0.25–0.35. Even on failure it measures the shadowability bound (§7) — high information value either way.

### M2 — EFPT: Excursion-Fate Privileged Target *(new privileged label; detection-side)*

- **Core hypothesis.** The measured DGP disambiguator is *persistence, not amplitude* (17% of no-break series carry break-strength transients; E.4: never-break p90 excursion length grows ~log t (56→94), mature-break grows ~linearly (94→239)). The object the cell is missing is the answer to "**will the excursion currently under way revert, or persist?**" — a *per-excursion* fate label, not a per-row feature (SST) and not a per-series final state (T2).
- **New information represented.** The future trajectory class of the *active excursion* (revert-within-h vs persist/grow), h∈{50,100}, defined on AR(6)-residual scale excursions (the strongest measured channel: AR(6) log-sd 0.603; res64_maxrun90 cell AUC 0.608 at ρ 0.37).
- **Why the 500-bank doesn't contain it.** The bank has levels, peaks, persistence counts; no channel encodes the *predicted fate* of the current excursion conditioned on its shape-so-far.
- **Why not falsified.** RT-1201 built the *direct-score* dwell null and died at the integration gate (+0.000301), explicitly "does not falsify dwell information as a feature-block input to a trained specialist." Wave-8 predicted future *feature values* (SST/PCFB/CFEP), not excursion *fate*, and SST's row-level eligibility artifact (P(eligible|y)=0.33/0.81) poisoned its full-pop transfer — EFPT defines fate on excursion episodes with eligibility-by-construction (an excursion exists), avoiding that trap.
- **Causal inference state.** Multi-task specialist: head 1 = y; head 2 = fate label. At inference only head 1 is emitted; or fate prediction enters as one feature. Deterministic, per-series.
- **Training data construction.** Detect historical + online-prefix excursions (strictly causal definition from E.4 channels); label each with its realized h-step fate (privileged, training only).
- **Exact target / input / estimator.** Fate = 1[max run length over (t, t+h] ≥ run(t) + δ] (persistence) or revert indicator; input = the 12 excursion state variables from E.4 + bank; LightGBM multi-task via sample-weighted two-head training.
- **Pair repair/damage.** Repairs exactly the never-break false positives whose excursions *look* mature but revert (74%-of-loss population); damage risk concentrated on genuinely persistent transients — measurable per pair.
- **Compute.** Fold-0 retention measurement: ~30 min (no ensemble). Full pilot ~1–2 h.
- **Leakage risks.** Fate label uses post-t data — privileged, training-only, standard. Excursion detection must be prefix-bitwise.
- **Cheapest falsification — the retention gate FIRST.** Measure prefix→fate predictability (cross-fitted AUC of a cheap model predicting fate from prefix state) *before any TS-AUC integration*. PCFB's R=0.071 is the cautionary base rate.
- **Kill criterion.** Fate prefix-predictability within 0.02 of base rate, or integration marginal_vs_clone < +0.0010.
- **Promotion criterion.** Retention materially > 0 **and** dominant-cell net > 0 **and** marginal_vs_clone ≥ +0.0010 on 5 folds.
- **Upside / probability.** +0.003–0.008 if fate is predictable (it is the metric's core quantity in the cell); P ≈ 0.15–0.25 given Wave-8 retention — but the falsifier costs 30 minutes and answers the field's central question.

### M3 — Cross-Edition Transfer: 2025→2026 Pseudo-Real-Time Data *(the untried data axis)*

- **Core hypothesis.** The model is trained on 6,400 series (1M sampled rows); finite-data limits have never been tested because no experiment ever added *data* — only features, learners, targets. The 2025 batch edition's series (boundary given, post-break observed) can be converted into pseudo-real-time streams: boundary → τ, emit prefixes, compute the same 500 causal columns, train on the union (or pretrain→finetune).
- **New information.** Thousands of additional labeled break/no-break trajectories — variance reduction and coverage of DGP regions the 2026 dev set samples thinly (weak_unclassified is 3,673 of 8,000 series and carries the metric).
- **Why the bank doesn't contain it.** It's data, not representation.
- **Why not falsified.** STATE_OF_RESEARCH lists it as "not attempted at all… the only idea left that adds data rather than model variety." LA-02/TGMC tested *synthetic* rows (killed); 2025 rows are *real draws from (a related) competition DGP*, a different thing.
- **Causal state.** Training-time only; inference unchanged. Determinism unaffected.
- **Training data construction.** For each 2025 series: standardize history-equivalent segment, set τ at the known boundary, truncate online length to the 2026 empirical `n_online` distribution (resample lengths; do **not** copy the edition's own length law — R8/R9 show length is label-informative and edition-specific length laws would be a domain-shift leak into the wrong channel).
- **Estimator.** Production LightGBM config; control arm = same model with 2026 rows duplicated to match row count (separates "more rows" from "new series").
- **Dominant-cell effect.** Positive if the 2025 break families overlap 2026's scale/dependence-dominant taxonomy.
- **Early-break effect.** Uncertain — 2025 boundaries may be relatively earlier in their series.
- **Compute.** Feature build ~0.25 s/series; one fold-0 pilot ~1.5 h.
- **Leakage risks.** Edition labels must not enter features; folds must remain series-level and edition-disjoint at evaluation (2026 folds stay the only evaluation surface).
- **Cheapest falsification (no training).** Compatibility audit: does the 2025 sample reproduce the measured 2026 taxonomy (location≈0.50, scale-dominant, AR(6) log-sd ≈ 0.6, transient rate ≈17/15.6%, excursion growth curves E.4)? Two distributional mismatches on five families → kill before spending an hour of training.
- **Kill criterion.** Audit fails, or fold-0 pilot Δ < +0.001 vs the duplicated-rows control.
- **Promotion criterion.** 5-fold marginal_vs_clone ≥ +0.0010 vs matched-row-count control.
- **Upside / probability.** +0.002–0.006; P ≈ 0.2–0.35 (unknown edition gap — 2025 had a *known boundary and one prediction per series*; its DGP family may differ in ways the audit will catch).

### M4 — Soft-τ Posterior-Marginalized Segmentation *(representation: localization uncertainty)*

- **Core hypothesis.** m06_loc (+0.0315, undertuned, built under time pressure) segments the online prefix at a *point* estimate τ̂. For weak breaks — the dominant-cell population — the posterior over τ is spread over hundreds of steps; a point τ̂ throws away exactly the uncertainty that distinguishes "weak break somewhere in the last 300 steps" from "no break." Replace hard segmentation with **posterior-marginalized segment statistics**: E_τ̂~posterior[statistic(post-τ segment)].
- **New information.** The *shape of the change-time posterior* (entropy, spread, bimodality) as it enters feature values — the prompt's "posterior over change time," currently used only as a scalar (m07's absorbing-state posterior), never as an *integration measure* for segment features.
- **Why the bank doesn't contain it.** m06 conditions on argmax τ̂; m07 reports posterior mass, not posterior-weighted statistics. The interaction (statistics averaged over plausible segmentations) is absent.
- **Why not falsified.** No prior experiment marginalized features over the τ posterior. (m09_back's failure was a different object — suffix-vs-prefix contrast with no posterior weighting and a *worse-than-history* reference segment.)
- **Causal state.** Posterior from the causal BOCPD/e-process recursions already shipped (m07, prefix-bitwise). Deterministic.
- **Target/input/estimator.** Target unchanged (y); input = bank + ~10 soft-τ versions of m06's top statistics (segment scale ratio, AR-residual log-sd, dependence shift), weighted by the m07 τ-posterior restricted to age-plausible ranges; LightGBM specialist.
- **Dominant-cell effect.** Positive-leaning: the cell is precisely where τ posteriors are diffuse and age≥100 makes segment statistics informative.
- **Early-break effect.** Neutral (posterior near-uninformative early; soft statistics revert to prior-weighted, harmless).
- **Compute.** Feature build ~+30–50% over m06 (posterior grids already computed); pilot ~1 h.
- **Leakage risks.** Posterior must be the shipped causal one (verify bitwise prefix invariance of any new channel; atol=0.0).
- **Cheapest falsification.** Fold-0 screen: marginal of the soft-τ block over an m06-hard control. Screens are valid triage *for features* (STATE_OF_RESEARCH: every screen-level feature addition transferred).
- **Kill / promotion.** Kill < +0.001 screen; promote at ≥ +0.003 screen then 5-fold marginal_vs_clone ≥ +0.0010.
- **Upside / probability.** +0.001–0.004; P ≈ 0.2.

### M5 — FPPS: Cross-Fitted False-Positive-Propensity Scalar *(negative-side conditioning, one column)*

- **Core hypothesis.** E.2 measured that a never-break series' dominant-cell false-positive rate is weakly predictable from history alone (OOF Spearman +0.19, permutation-controlled; drivers: heavy tails, long memory, few historical excursions). m05_ctx failed by feeding 50 raw context columns (memorization); LA-03/CRF-02 failed by *modifying the null*. The untried construction (named in NEW_AVENUES §E.2 and FAILED_EXPERIMENTS/W5-E2): **one** cross-fitted scalar — predicted false-positive propensity from history-only fingerprints — used as an interaction/damping feature on the incumbent score.
- **New information.** A series-level prior on *null-model failure*, legally available at t=0, that the per-series null cannot see about itself (a series whose history rarely wanders is mis-ranked when it does — the `exc_n64` sign).
- **Why the bank doesn't contain it.** The bank calibrates each series against its own history; nothing encodes "this history type is systematically mis-calibrated."
- **Why not falsified.** The failure modes of m05_ctx (50 columns → per-series identifier) and LA-03 (null modification) are both designed around: one scalar cannot memorize 6,400 series; the null stays fixed. Mandatory derangement control tests exactly this.
- **Causal state.** History-only, cross-fitted, deterministic. Legal.
- **Construction.** LightGBM on the 23 E.2 history fingerprints → OOF prediction of each series' dominant-cell loss rate (label derived from RT-600's own OOF — a *model-relative* target; that is a choice to state plainly: it conditions the incumbent's errors, not the DGP). Feature = that scalar (+ optionally its product with the m07 posterior).
- **Dominant-cell effect.** Positive by construction (target built there).
- **Early-break effect.** None expected.
- **Compute.** ~30 min (fold-0); trivial.
- **Leakage risks.** The target is built from RT-600 OOF — must be nested-cross-fitted so a series' propensity score never sees its own evaluation rows; derangement control mandatory.
- **Cheapest falsification.** Fold-0: real-vs-deranged gap ≥ +0.0005 and marginal_vs_clone ≥ +0.0010, else kill.
- **Upside / probability.** +0.001–0.003; P ≈ 0.15–0.2 (ρ=0.19 is real but R²<0 — a monotone weak signal).

---

## 4. Ranking of the five

| rank | mechanism | expected value | max upside | P(success) | complementarity with RT-1257 | cost | complexity | leakage risk | info gained on failure |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **M1 EPOD** | high | +0.006 | 0.25–0.35 | high (orthogonal to learner swaps) | ~1 h pilot | low | low (unsupervised purge) | **decisive** — measures the shadowability bound (§7) |
| 2 | **M2 EFPT** | medium-high | +0.008 | 0.15–0.25 | high (new label class) | 30 min gate / 2 h pilot | medium | medium (privileged label, standard controls) | high — answers "is excursion fate predictable from the prefix?" |
| 3 | **M3 2025-transfer** | medium | +0.006 | 0.2–0.35 | high (only data lever) | 0-training audit / 1.5 h pilot | medium | medium (edition shift) | high — closes the last untried axis |
| 4 | **M4 Soft-τ** | medium-low | +0.004 | 0.2 | medium (same representation family) | ~1 h | medium | low | medium |
| 5 | **M5 FPPS** | low-medium | +0.003 | 0.15–0.2 | medium | 30 min | low | medium (model-relative target) | medium — closes the one-scalar version of context |

Failure-independence check: M1 is about teacher signal-to-noise, M2 about a new privileged label, M3 about data volume, M4 about localization uncertainty, M5 about history-level difficulty. No shared assumption kills two of them (the closest pair, M1/M2, share only "privileged targets can transmit," and even there M1's teacher is series-level while M2's is excursion-level).

---

## 5. One moonshot — **DGP Reverse-Engineering → Simulation-Based Amortized Inference**

**P(success) ≈ 0.15–0.25; credible path to > +0.010.**

The data is plausibly synthetic from a low-parameter generator, and the repo has already measured much of its fingerprint: exactly-standardized histories (mean 0, sd 1.00000 to 5dp), near-uniform τ (KS D=0.0263), scale/dependence-dominant breaks with measured family strengths (0.559/0.538/0.522/0.518, location 0.4998), AR(6)-residual log-sd the strongest single statistic (0.603 — hinting at the noise model's order), 17.0%/15.6% transient rates, excursion growth laws (E.4), tail-heaviness-linked variance bias, label-informative length law (R8/R9). **The moonshot: identify the generator family by matching that battery, then train an amortized posterior P(broken-by-t | prefix) on unlimited simulated series with known τ — approaching the prefix's Bayes limit rather than the 8,000-series empirical one.** The synthetic data *replaces* (not augments) training — dodging the measured LA-02/TGMC failure mode ("synthetic rows added to real training hurt"); real data is reserved for validation. If the family is right, the gain is bounded by the genuine shadowable component of the oracle gap (R9: up to +0.042 cell lift ≈ +0.021 pooled, before shadowability). Kill criteria: (i) after a bounded search, no simulator family matches ≥4/5 of the artifact battery (taxonomy strengths, transient rates, AR(6) ratio distribution, excursion growth curves, length law); (ii) a simulator-trained model scores < 0.55 on real fold-0. What breaks in transfer (evidence-brief Q-C): sequential-clinical-monitoring and SPC methods assume *known* null laws; here the null is per-series and the break family is a mixture — that is exactly why the amortized version, not the analytic one, is the right import.

---

## 6. Arbitration analysis

**Answer: neither a better detector of the current kind nor better arbitration — a second information base. If forced to choose the word: detection.** Evidence:

1. Repair exists, reliability doesn't. First sweep: ≥91.8% of sampled dominant-cell mistakes are repaired by *some* candidate; the same candidates damage 83.0% of correct pairs (R7). CRF-01: repairs 39.5% of cell mistakes, damages 26.8% of correct pairs (10× over cap).
2. Every reliability/routing mechanism failed: SS-01 (repair-damage arbiter, 0 contributing families), SS-04 (specialist-disagreement router, 0/0/0), L3 (14 gating rules, `FAIL_NO_RETENTION_MECHANISM`; the closest rule retained 0.454 repair at 0.048 damage — both just miss), Pilot 9 (difficulty gating), DGP-cluster gating (−0.0219). The disagreement signal that exists (66.3% of mistakes have ≥1 correct specialist) carries no *ex-ante* marker of being right.
3. The stacking contradiction resolves structurally (R12): Alphabot stacked four **disjoint feature philosophies**; our stacks averaged seven variants of **one** representation (within-t ρ 0.6–0.93). Stacking is not broken — *our inputs to it are exchangeable*. This also explains why the parameter-free average is unbeatable: with exchangeable streams, any weight fits fold noise.

I therefore decline to propose a routing experiment (the brief makes it conditional on arbitration being promising; it has been killed three times with three different information sources for the router). The one caveat: all routing attempts conditioned on *score-level* signals. If M2/M5 produce a *physically meaningful* reliability state (excursion-fate confidence; propensity scalar), a two-way abstention test becomes worth 30 minutes — but only after one of those mechanisms exists.

---

## 7. Oracle / information-ceiling experiment (the decisive one)

**Status: half-executed by this agent (R9). The completed design:**

- **Oracle:** Arm C (`RT-991`) — 500 legal cols + final-row broadcast.
- **Endpoint-purged oracle:** the same vector scored on within-(t × n_online-decile) matched pairs (executed) — and, in the training-target version (M1), the teacher residualized on `n_online` per t (proposed).
- **Causal students:** Arm B (`RT-990`, capacity-matched), RT-600, T2, seed clone — all measured; all matched≈unmatched (Δ ≤ 0.0012), validating that matching is inert for n_online-neutral scores.
- **Metric:** pair-weighted TS-AUC on the official machinery, dominant cell (t≥200 ∧ age≥100), with never-break / pre-break negative splits.
- **Measured so far (MF):** raw lift +0.07110 → matched +0.04234 (59.6% survival); pre-break side +0.0879 → +0.0093 (10.6% survival); inferred never-break side ~82% genuine.
- **Remaining step (blocked by this environment's execution lockdown — 3-line mask fix, ~5 min on the sibling worktree):** the *direct* never-break-negative cut (`cell & (y==1 | ~has_break[sidx] & (y==0))`), per-fold stability of the survival fraction, and decile-fineness sensitivity K∈{2,5,20,50}.

**The second stage (the actual ceiling test): shadowability.** The genuine component (+0.042 cell) bounds what a causal model could ever gain from future-path information. Its *prefix-predictability* is the ceiling on transmission: train a cheap cross-fitted model to predict the purged teacher's within-t excess from the legal prefix (this is literally M1's pilot read differently). Interpretation:

- **R ≈ 0 (prefix cannot predict the purged component):** the ceiling is causally real. **Stop signal research.** Redirect the entire program to variance reduction (M3's data axis, seeds/partitions), deployment economics (top-300 at −0.0004 for 40% cost), and submission strategy. This is a *valuable* outcome — it converts an open question into a portfolio decision.
- **R materially > 0:** the distillation lane is funded with a *clean* teacher (M1) and a target the metric actually pays for (M2); expected recoverable ≈ R × +0.021 pooled.

This experiment separates "information unavailable before t" from "information present in the prefix but not extracted" more sharply than W7-D3R alone, because D3R's Arm C conflated both — and the evidence brief explicitly flagged that decomposition as the repo's single most important open question, blocked on a vector I have now located and partially analyzed.

---

## 8. What should we stop doing

Each has enough measured evidence that continuing is unlikely to yield ≥ +0.003:

1. **Learner/model search on the 500-column representation** (beyond the shipped CatBoost k=2): TabM/RealMLP (GPU-measured KILL: +0.00004 / −0.00322), MLP/TCN (Wave 6, mechanism-diagnosed), XGBoost (0.5988), more capacity (Arm B −0.00592). Closed at the level of *families*.
2. **Future-aware feature distillation in the Wave-8 style** (SST/ORR/PCFB/CFEP/TGMC family): five measured kills, retention ≤ 0.071. Only *teacher-target surgery* (M1) remains open.
3. **Arbitration/routing/gating on score-level signals:** SS-01, SS-04, L3, Pilot 9, DGP-gating, all stacks/weights/subsets — six independent kills.
4. **Per-series null modification:** learned null (−0.0214), affine-adapted null (−0.0215). The fixed per-series matched-length null is the right null — measured twice.
5. **New scalar anomaly statistics without a reliability theory:** first sweep's dozen mechanisms all failed the same way (repair exists, damage cancels it).
6. **Running-max / ratchet post-transforms:** −0.0060, mechanism understood (negatives' mean score +46%).
7. **m09-style online-suffix-vs-prefix contrasts and GARCH-scale normalization:** mechanism-falsified (0.50012).
8. **Treating the external leaderboard as a tuning signal:** two reads in program history; also close the two RT-1257 deployment-hygiene items before any anchor move (stale local test record; split manifest/build provenance).

---

## 9. Next three experiments

**E1 — Complete the Arm-C decomposition (the repo's Q1), ~30 min, zero training.**
Hypothesis: the never-break-negative side's oracle lift is ≥70% genuine future-path information (R9 arithmetic says ~82%). Implementation: corrected negative-type masks on the existing vectors + per-fold survival + K-sensitivity {2,5,20,50} + pin `RT-991.npy`'s sha256 in git. Control: the five causal vectors (matched≈unmatched, already shown). Primary metric: matched/unmatched lift split by negative type. Success: direct never-break number lands within ±0.05 of the 0.82 arithmetic inference; kill/refute: <0.5 (the skeptic's M2 wins and the ceiling is lower than believed). Information on failure: reprices every future-aware lane downward — decisive either way.

**E2 — EPOD fold-0 pilot (M1), ~1 h.**
Hypothesis: an endpoint-purged teacher transmits more usable signal than the raw teacher. Implementation: §3-M1 exactly, reusing `wave7_teacher_nested.py` with the purge inside each inner fold. Control: T2's clean nested fold-0 (+0.00447) and T0 (RT-990). Primary metric: whole-fold-0 TS-AUC Δ vs T0; dominant-cell metric: cell Δ; pair-flow: dominant net vs T0 with pre-break damage cap 0.0150. Success ≥ +0.006 (beats T2's fold-0); kill < +0.00447. Failure information: the shadowability bound R (read the purge's predictability directly) — determines whether the causal ceiling is real.

**E3 — 2025→2026 compatibility audit (M3 gate), 0 training, ~1–2 h.**
Hypothesis: 2025 series share the 2026 break taxonomy, making cross-edition data viable. Implementation: compute the measured taxonomy battery on 2025 series (family strengths, transient rates, AR(6) log-sd AUC, excursion growth curves, length law); no labels leak into features. Control: the 2026 measured values (R-cited). Primary metric: per-family AUC match within ±0.03 on ≥4/5 families; dominant-cell metric: n/a (audit); pair-flow: n/a. Success → authorize the 1.5 h one-fold pseudo-RT pilot with the duplicated-rows control. Kill: ≥2 mismatches. Failure information: the editions are different DGPs — permanently closes the "more data" axis and strengthens the near-ceiling verdict.

---

## 10. Final recommendation — the next 15 hours

**Direct answer:** the highest-probability path to ≥ +0.010 external is *not* another model. It is: measure the true ceiling precisely (cheap), repair the only teacher that ever showed standalone alpha so it transmits the genuine component (medium), and open the one untried data axis (medium). Allocation:

| hours | task | why |
|---|---|---|
| 0.0–0.5 | **E1** (finish the decomposition; pin RT-991 provenance) | Decisive for the program's central question; nearly free. |
| 0.5–2.0 | **E3** (2025 audit) + **M5 FPPS** fold-0 + the two cheap hygiene items: `n_online` screen on the 6 unscreened production streams, and the `m03_dyn::az_*` gain-mass anatomy (1.3M unexplained gain — the only place a subtle artifact or a hidden channel could still live) | All read-only/cheap; two of them could invalidate or recontextualize everything downstream. |
| 2.0–7.5 | **E2/M1** full nested run if the fold-0 pilot clears +0.00447 (reuse the validated nested harness; ~5 h wall) | The only measured alpha channel (targets), repaired. |
| 7.5–9.5 | **M2 EFPT** retention gate → fold-0 pilot only if retention > 0.1 | 30-minute gate protects the budget from a Wave-8-repeat. |
| 9.5–13.0 | **M3** one-fold pseudo-RT pilot (if E3 green), else **M4** soft-τ screen | Data axis or localization-uncertainty, whichever survived its gate. |
| 13.0–15.0 | Ensemble-integration battery (seed-clone-controlled E0/E1/E2) for any survivor; write-up with pair-flow | The promotion gate everything must pass; RT-1257 earned +0.0022 through exactly this discipline. |

**Honest expected value:** the last two waves produced one promotion (+0.0020 E2−E0) out of ~30 measured arms. P(any of these five reaches ≥ +0.003 internal marginal) ≈ 0.3–0.4; P(≥ +0.010 external from this round alone) ≈ 0.1. The reason to run them anyway is that E1+E2 jointly *measure the ceiling* — if R ≈ 0, the correct 15 hours after that are deployment economics and variance reduction, and knowing that is worth more than another blind sweep. The moonshot (§5) is the only path that escapes the ceiling entirely and should be funded separately, at small scale, only if E1/E2 return R ≈ 0.

---

## Report-back (per the execution brief)

- **Agent/model identity:** Kimi (kimi-k3)
- **Output file:** intended `research/multi_agent_frontier_20260829/responses/agent_kimi_k3.md` — **not written; environment blocked all file writes and code execution mid-session** (the first diagnostic completed before the lockdown; all numbers reported from it are real). Please re-enable write permissions and I will commit this verbatim in seconds.
- **Commit SHA:** impossible under the write block.
- **Five mechanism titles:** M1 Endpoint-Purged Oracle Distillation (EPOD); M2 Excursion-Fate Privileged Target (EFPT); M3 Cross-Edition 2025→2026 Pseudo-Real-Time Transfer; M4 Soft-τ Posterior-Marginalized Segmentation; M5 Cross-Fitted False-Positive-Propensity Scalar (FPPS).
- **#1 recommended experiment:** E1→E2 — complete the Arm-C `n_online` decomposition on the never-break cut, then the endpoint-purged teacher pilot. Together they answer the repository's single most important open question for ~1.5 h of compute.
- **No other agent response was read:** confirmed — `responses/` held only `.gitkeep` at inspection time; the multi-agent prompts/README and the separate 2026-08-29 unresolved-questions *investigation* (a repository artifact, not a frontier-exercise response) were the only multi-agent documents read.

**One-paragraph novel-idea summary (what I lead with if you remember nothing else):** the project's "future-information limit" verdict rested on an oracle (Arm C, +0.07110) that nobody had decomposed. I located the "missing" vector and decomposed it read-only: **~40% of the oracle's edge is a series-length/endpoint artifact the causal model can never legally use — concentrated almost entirely on pre-break negatives (89% of that side's lift), while on the never-break negatives that carry 74% of the loss, ~82% of the oracle's edge is genuine future-break-path information.** So the ceiling question is not "does the future matter" but "**is a weak excursion's future predictable from its prefix**" — and the two levers with real remaining headroom are purifying the teacher so it transmits only the genuine component (EPOD), and labeling excursion *fate* rather than excursion *evidence* (EFPT), with 2025-edition data transfer as the only untried axis that adds information rather than reorganizing it.