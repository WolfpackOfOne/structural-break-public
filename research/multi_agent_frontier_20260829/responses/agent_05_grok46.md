# Independent Frontier Analysis — agent_05_grok46

**Agent / model identity:** grok-4.6 (opencode)
**Date:** 2026-08-29
**Worktree:** `structural-break-multi-agent-frontier-20260829` @ `00f7ebd`
**Independence:** no file under `research/multi_agent_frontier_20260829/responses/` was read before this report was written. The directory was empty (0 files) at start.
**Scope:** analysis only. No model code, no training, no experiment IDs, no ledger edits.

Evidence labels used below:

- **MEASURED FACT** — recomputed or loaded from a primary artifact in this session, or a number I traced to a named JSON/report and treat as verified.
- **REPORT-CLAIM** — stated in a repository report; not independently recomputed here.
- **INFERENCE** — my interpretation of measured facts.
- **HYPOTHESIS** — a falsifiable claim I am proposing, not established.

---

## 1. Executive diagnosis

The production system is a strong, saturated *own-history surprise engine*. Seven causal modules (~500 columns) price the online prefix against a per-series, length-matched historical null; seven tree specialists are blended by a parameter-free logit/SCDF average. That architecture is the right answer to the problem the taxonomy actually contains: location breaks are absent (series-level AUC 0.4998, REPORT-CLAIM from `research/reports/break_taxonomy.md` via `STATE_OF_RESEARCH.md`), scale/dependence dominate, 92.4% of break series are individually indistinguishable from a placebo split at p<0.01, and transients exist at similar rates in break and no-break series. Persistence, not amplitude, is the disambiguator, and the top gain features are exactly peak / decayed-peak / absorbing-posterior channels.

What it does well:

- Per-series historical-null calibration makes scores comparable across series, which is the only thing TS-AUC scores.
- The 500-column bank is *collectively* live: nested pruning loses −0.0004 at top-300 and −0.0174 at top-60 (REPORT-CLAIM, `STATE_OF_RESEARCH.md` RT-140). This is not a redundant bag.
- Equal-weight averaging of same-representation specialists is the correct combiner for *this* representation. Every learned stack lost.
- CatBoost as a two-slot replacement (`RT-1257`) produced the first external movement since 0.6268: **0.6290**, +0.0022 (REPORT-CLAIM, `engineering/reports/rt1257_deployment/SUBMISSION_16.md`). Internal E2−E0 +0.002026, 5/5 folds (REPORT-CLAIM, `reports/catboost_specialist_2026/FINAL.md`). That is real, small, and a learner-family increment, not new information.

The true bottleneck is **not learner capacity, not blend weights, and not “more of the same 500 columns.”** Arm B of W7-D3R (same 500 columns, more trees) *lost* to Arm A: cell AUC −0.00592, 1/5 folds (MEASURED FACT from `structural-break-wave8/research/reports/wave7_d3r.md` / `wave7_d3r.json`). CRF-01’s representation×objective factorial moved the low-redundancy frontier to 0.59276 at ρ=0.446 and still damaged the incumbent (REPORT-CLAIM, `CRF_FINAL.md`). Wave 8 distilled five distinct functionals of the future and all five ensemble-marginals sat at ≈ −0.00031 (REPORT-CLAIM, `wave8_final.md`). New Avenues first and second sweeps, 79 catalogued mechanisms, SS-01…SS-04, all KILL.

Classification of the bottleneck:

| axis | status |
|---|---|
| Learner | Closed as a +0.003 path. CatBoost +0.002 is the residual. TabM/RealMLP INFEASIBLE. MLP/TCN on `y[t]` learn elapsed time. |
| Representation (500-bank) | **Extracted, not necessarily complete.** Arm B shows this subspace is fully used. It does not show that no other subspace of the prefix exists. |
| Objective | Live. `t_online` is a top-10 BCE gain feature and is *constant* inside each `AUC(t)`. Row-level BCE rewards a direction TS-AUC deletes. Pairwise-t ranking was tried and did not beat logloss on trees (`RT-123` 0.61444 vs 0.61510). The mismatch is real; it has not yet been shown to be the remaining alpha. |
| Target | Live, but Wave 7/8 show that naive future-score distillation reproduces RT-600. The untested object is a *T-orthogonal* future-path residual, not another teacher of `y`. |
| Arbitration | Dead for the current seven streams. Alive *after* a disjoint view exists. Alphabot stacked seven hyperparameter variants of one bank. That is the stacking contradiction. |
| Compute | Binding but not the scientific limit. RT-600 fits P=6; new streams must be streaming O(1) or cheap history-only precompute. |
| Information ceiling | **Unfalsified, not confirmed.** W7-D3R CASE 2 (“future-information limit”) treats Arm C’s +0.07110 cell AUC as post-`t` evidence. Arm C broadcasts each series’ *final-row 500-vector*, which includes `t_online = T` (series length) and every peak that occurs after `t`. Nobody has residualized Arm C against remaining horizon / `T`. The prior investigation’s claim that `RT-991.npy` is missing is **false**: the vector exists and loads. |

The latent quantity the current system does not represent well is not “more temporal information.” It is the **second reference measure**.

Every column in the bank asks: *is the online stream surprising relative to this series’ own break-free history?* Remaining dominant-cell loss is 45.29% of inversion loss, 50.50% of pair weight, cell AUC 0.66428, and **74.0% of that cell’s loss sits on never-break negatives** (MEASURED FACT, `research/reports/wave7_rt600_exact_alpha_budget.json`). Those never-breaks are surprising versus their own history — that is why they score high — and they are not breaks. The missing questions, which are mathematically distinct, are:

1. Is this surprise *on this DGP’s attractor* (stationary long excursion) or *off it* (regime change)?
2. Is this surprise unusual *among series of the same online age*, or only unusual versus self?
3. Is the *duration* of the surprise typical of this DGP’s transients under a non-memoryless dwell, or typical of an absorbing change?
4. Will the surprise *revert after t* (shadowable future path), after stripping remaining-horizon leakage?

The 500-bank measures calibrated amplitude, peak shape, and a geometric-hazard absorbing posterior. It does not represent (1)–(4). New Avenues tested crude *scalars* of related ideas (dwell, trajectory geometry, Hankel-DMD) and they either had no signal (RT-1202 AUC 0.4993, ρ=0.0039) or collapsed into the incumbent direction (ρ≈0.86–0.89). That falsifies those scalars. It does not falsify the four objects above.

If I had to solve the competition from scratch and were forbidden the 500-column bank, I would not start with a Transformer. I would build:

1. A **per-series nonparametric predictive null** in delay space (k-NN / kernel density on historical delay vectors; history-only fit, frozen online) plus the existing AR-residual ECDF, because CRF-02 showed a *per-series* null beats an amortized learned null by −0.021.
2. An **explicit-duration (semi-Markov) absorbing filter** on a 5–10 dimensional sufficient statistic (scale, dependence, tail occupancy, attractor occupancy, one CUSUM), not a geometric BOCPD.
3. A **frozen training-cohort atlas** that prices each online statistic against the empirical law of training series at the same `t` — a second null, legal at inference because it is a lookup table, not cross-series state.
4. Trees on ~80 columns from those three views, equal-weight blended. Disjoint reference measures, not disjoint hyperparameters.

**Frame (prompt §5B).** I do not believe we are at the legal Bayes limit. The perfect-repair ceiling if the dominant cell were fixed is 0.79518 (MEASURED FACT, same JSON). That ceiling uses future knowledge, so it does not refute a legal-prefix ceiling near 0.63. The cheapest observation that *does* separate “near the Bayes limit” from “stuck in a local frame” is the Arm-C residualization designed in §7. If that residual is ~0, I would reverse this diagnosis and spend the rest of the budget on RT-1257 hygiene, lockbox economics, and variance reduction. Until that residual is measured, treating CASE 2 as settled is the program’s most expensive assumption.

---

## 2. Repository evidence

### 2.1 Numbers that informed the diagnosis

| item | value | class | where | what it actually establishes |
|---|---|---|---|---|
| RT-600 external | 0.6268 | REPORT-CLAIM | `STATUS.md`; `EXPERIMENT_ID_MAP.md` | One public read of the frozen seven-specialist SCDF blend. |
| RT-1257 external | 0.6290 (+0.0022) | REPORT-CLAIM | `engineering/reports/rt1257_deployment/SUBMISSION_16.md:6,119-121` | First external movement. One observation, no error bar, not a transfer law. Realised delta sits between internal `marginal_vs_clone` +0.002407 and E2−E0 +0.002026. |
| RT-600 5-fold mean OOF | 0.625811 | MEASURED FACT | `wave7_rt600_exact_alpha_budget.json` `reproduction.observed_mean` | The development estimand. **Not** the 0.63828 figure. |
| RT-600 fold-0 | 0.63828 | MEASURED FACT | same JSON `expected_per_fold[0]` | Fold-0 only. Wave-7 “pooled OOF ≈ 0.63828” is this, not a 5-fold pool. |
| Dominant cell loss share | 0.452906 | MEASURED FACT | same JSON `dominant_cell_t200_age100.fraction_of_inversion_loss` | `t≥200` AND age`≥100` holds 45.29% of remaining inversion loss. |
| Dominant cell pair weight | 0.505047 | MEASURED FACT | same JSON | Half the metric lives in this cell. |
| Dominant cell AUC | 0.664277 | MEASURED FACT | same JSON | Mature-break ranking is already much stronger than overall 0.626. |
| Never-break share of cell loss | 0.739907 | MEASURED FACT | same JSON | The remaining problem is false-positive never-breaks, not early detection. |
| Perfect-repair ceiling | 0.79518 | MEASURED FACT | same JSON | Huge *oracle* headroom; not a legal-prefix ceiling. |
| D3R B−A | −0.00592 cell, 1/5 folds | REPORT-CLAIM (traced to `wave7_d3r.md`) | `structural-break-wave8/research/reports/wave7_d3r.md` | Same 500 columns, more capacity, no gain. Extraction of *this* bank is done. |
| D3R C−B | +0.07110 cell, 5/5 folds | REPORT-CLAIM | same | Final-row 500-vector is enormously informative. Mechanism not decomposed. |
| Arm C pooled cell AUC | 0.71859 | REPORT-CLAIM | same | |
| `RT-991.npy` | shape (5036517,) float32, 19.934% NaN, min 3.3e-7, max 0.99996 | MEASURED FACT | loaded `structural-break-wave8/research/oof/RT-991.npy` and the wave6 copy | **The “missing Arm-C vector” blocker is false.** Same NaN mask on RT-990 and RT-300: it is the non-dev-fold mask, not Arm-C corruption. |
| T2 standalone | +0.00943, 5/5, CI [+0.0043,+0.0137] | REPORT-CLAIM | `wave7_t2_promotion_final.md` | Real single-model alpha. |
| T2 ensemble E2−E1 | +0.00024 | REPORT-CLAIM | same | Learned something RT-600 already knew. |
| Wave 8 five pilots | all ≈ −0.00031 vs clone | REPORT-CLAIM | `wave8_final.md:144-150` | Unresidualized future distillation does not survive the full population. |
| SST eligibility | P(elig\|y=1)=0.33 vs 0.81 at h=200, t<20 | REPORT-CLAIM | `wave8_final.md:44-49` | Future-horizon targets are class-conditionally available. A structural confound Wave 8 documented and then still used as a teacher family. |
| CRF-01 | 0.592762 @ ρ 0.446; pre-break damage 0.2626 | REPORT-CLAIM | `CRF_FINAL.md` | New representation + ranking objective still too weak and too damaging. Representation effect +0.0444, objective effect +0.0222, sum still short. |
| CRF-02 | learned null 0.55914 vs fixed null 0.58059; derangement +0.00338 | REPORT-CLAIM | `CRF_FINAL.md` | Amortized 8-float null loses to per-series AR(5)+256-knot ECDF. Conditioning works; compression fails. |
| RT-1216 | marginal_vs_clone +0.000937 | REPORT-CLAIM | `FIRST_SWEEP_SYNTHESIS.md` | Closest New Avenues miss. Weighted conformal martingale; still KILL. |
| RT-1215 Hankel-DMD | +0.000226, ρ=0.8859 | REPORT-CLAIM | `pilot03_observers.md` | Linear delay-embedded operator is already in the bank’s span. |
| RT-1202 trajectory geometry | AUC 0.4993, ρ=0.0039, net −2974 | REPORT-CLAIM | `FIRST_SWEEP_SYNTHESIS.md` | That scalar contained nothing. Orthogonality without signal. |
| SS-01…04 | −0.00031, −0.00029, −0.00030, −0.00031 | REPORT-CLAIM | `STATUS.md` | Arbitration of existing sensors is KILL. SS-03 also failed pre-break damage cap 0.0199>0.0150. |
| Lockbox haircut | −0.0116 (0.61214 vs 0.62374) | REPORT-CLAIM | `STATE_OF_RESEARCH.md:31-32` | Only clean dev→held-out read. Price every claimed dev gain against it. |
| MLP vs LGBM | 0.57059/0.58061 vs 0.61605, 0/5 folds; train BCE 0.077 | REPORT-CLAIM | `wave6_neural_results.md` | Elapsed-time trap. Any sequence model on `y[t]` must say how it avoids this. |
| RT-900 | 0.86552 VOID | REPORT-CLAIM | `wave6_corrected_oracle.md` | Missingness mask = label. Oracle designs that encode `t≥τ` via NaN are illegal and misleading. |
| W6-E2R B−A | +0.0235 series AUC, 5/5 seeds, 25/25 folds | REPORT-CLAIM | same | Given a *known* boundary, the 500-bank beats a generic oracle bank. Representation is still a lever *when τ is given*. That does not license online copies of Alphabot two-sample features. |
| Stacking | LOFO logistic 0.62514, LGBM stack 0.62144, greedy −0.0008 vs logit avg 0.62544 | REPORT-CLAIM | `STATE_OF_RESEARCH.md` | Learned combination of same-representation streams loses to the parameter-free average. |
| Feature prune | top-300 −0.0004, top-60 −0.0174 | REPORT-CLAIM | `STATE_OF_RESEARCH.md` RT-140 | Bank is not a pile of duplicates. |
| Location family | AUC 0.4998 | REPORT-CLAIM | break taxonomy | Classical mean-shift literature is aimed at a break type this data does not contain. |
| AR(6) residual log-sd ratio | 0.603 | REPORT-CLAIM | taxonomy / `STATE_OF_RESEARCH.md` | Strongest single statistic in the program. Scale-after-whitening is the primitive. |
| m05_ctx | +0.0009 screen, −0.0177 full; deranged context −0.057 | REPORT-CLAIM | `STATE_OF_RESEARCH.md` | History does not predict *whether* a break occurs. Series-constant features are a memorisation attack. |

### 2.2 Discrepancies versus the evidence brief

1. **`RT-991.npy` is not missing.** Prompt §3 and the 2026-08-29 unresolved-questions investigation treat Arm-C residualization as blocked. I loaded the vector from `structural-break-wave8/research/oof/RT-991.npy` (and a byte-identical-shape copy under wave6). MEASURED FACT. This is the highest-value cheap diagnostic in the repository, and it is unblocked.
2. **`STATE_OF_RESEARCH.md` on this worktree is stale.** It still titles RT-131 (illegal rank average) as current champion. STATUS.md and the RT-1257 deployment record are current. Do not reason from `STATE_OF_RESEARCH.md` without checking STATUS.
3. **“RT-600 pooled OOF ≈ 0.63828”** is fold-0, not pooled. 5-fold mean is 0.62581; 5-fold pooled is 0.625627 (JSON `pooled.ts_auc`). The evidence brief already warns about estimands; the number is still quoted as pooled.
4. **RT-1257 fold concentration.** The evidence brief says folds 3–4 supply 64% of summed deltas. Learner-diversity E2−E1 fold deltas are +0.00191, +0.00066, +0.00133, +0.00052, +0.00113 (`learner_diversity_2026/FINAL.md:30-31`) — folds 0 and 2 dominate *that* contrast. I did not recompute the hybrid’s E2−E0 per-fold shares. Treat “64% in folds 3–4” as unverified here.
5. **Canonical vs worktree RESULTS.csv.** The evidence brief’s “77 of 78 rows at `git_sha=nogit`” describes the disconnected harness lineage, not this worktree’s research ledger.

### 2.3 What I am *not* treating as fact

- CASE 2 “the residual W7-D3R gap is predominantly post-`t` information” (CRF_FINAL / STATUS). That is an **INFERENCE** from an undecomposed Arm C.
- “Missing causal prefix information is not the primary explanation” (FIRST_SWEEP_SYNTHESIS H6). That is an **INFERENCE** from killed *scalars*, not from a prefix-vs-future decomposition.
- Any claimed path from +0.002 internal to +0.010 external. The lockbox haircut is −0.0116 on the only clean read.

---

## 3. Five high-upside mechanisms

These are scientifically distinct: failure of one does not imply failure of the others. At least one is from outside this repository’s lineage (C2ST / two-sample classification; HSMM duration from clinical monitoring). At least one lives outside {features}×{learners}×{blend} (T-orthogonal teacher residual; frozen cohort atlas as a *reference measure*, not a feature family in the usual sense).

---

### Mechanism 1 — T-orthogonal full-sequence residual (shadowable future-path)

**Core hypothesis.** Arm C’s +0.07110 cell AUC is a mixture of (a) remaining-horizon / series-length `T`, which is non-shadowable and worthless under an unknown horizon, and (b) the series’ *eventual evidence path* after `t` (does the excursion persist or revert?). Component (b) is a legal training target. Wave 8 distilled (a)+(b) entangled, with class-conditional horizon eligibility, and measured zero ensemble alpha.

**New information represented.** The component of `full_sequence_break_confidence` orthogonal to `{T, T−t, 1[t near T], n_online_final}`. Equivalently: *future reversion vs future accumulation*, not *how much series is left*.

**Why the 500-bank probably does not contain it.** The bank is a function of `x_{1:t}` only. Future reversion is a functional of `x_{t+1:T}`. Peak/persistence channels are causal *proxies* for permanence; Arm C says those proxies are far from sufficient in the dominant cell.

**Why previous experiments do not falsify it.** T2 distilled a nested full-sequence teacher and became redundant with RT-600 (E2−E1 +0.00024): that teacher was not residualized against `T`. Wave 8 SST/PCFB/CFEP used `t+h` targets whose availability is class-conditional (`wave8_final.md` §C). ORR repaired rows but not pair-weight. None of these is “Arm C score minus a regression on remaining horizon.”

**Causal inference state.** Student: any function of `(H_i, x_{1:t})`. Teacher: function of the full series, used only in training, nested outer-fold pure.

**Training data construction.** Using existing OOF: `RT-991` (Arm C), `RT-990` (Arm B), `RT-300` (Arm A), `folds.parquet`, per-series `T`. Residualize `logit(RT-991)` on `{T, T−t, t/T}` *within each t* (so we do not reintroduce elapsed time). The residual `R` is the teacher target.

**Exact target.** `R_i(t) = logit(s^C_{i,t}) − Π[logit(s^C) | T, T−t, t/T, t]`, where `Π` is a within-`t` isotonic or ridge projection. Secondary target: sign of future peak increment `peak(T)−peak(t)` after the same projection.

**Exact input representation.** Student inputs = current 500 columns (or a cheap subset). No future columns.

**Suggested estimator.** Nested OOF LightGBM/CatBoost regressor, same fold-purity sentinel as T2 (0/20 vs 20/20 is already a validated tool).

**Streaming inference.** One extra tree on the existing feature vector. Negligible ms/point.

**Interaction with RT-600 / RT-1257.** Add as an eighth stream only if dominant-cell pair net is positive and pre-break damage < 0.015. Do not average in a residual that is ρ>0.85 with the incumbent.

**Expected dominant-cell effect.** Primary. That is where Arm C’s lift lives.

**Expected early-break effect.** Small. Early cells have less future path and less pair weight.

**Expected pair repair.** Never-break FPs whose evidence later reverts should be down-weighted; mature breaks whose evidence continues to accumulate should be up-weighted. This is exactly the 74% never-break cell-loss mass.

**Expected pair damage.** If `R` still carries a monotone transform of the incumbent score, we replay T2’s redundancy. If the projection is incomplete and `T` leaks, we fit series-length, which TS-AUC may or may not reward and which is illegal at test (unknown horizon). The within-`t` projection is the guard.

**Compute.** Diagnostic on existing npy: minutes. Distillation fold-0: ~1–2 h. Full nested: ~4–5 h.

**Leakage risks.** Outer-fold contamination of the kind that inflated the first T2 pilot by +0.012 to +0.023. Using `T` as a *student* feature. Eligibility masking that correlates with `y` (Wave 8’s documented trap).

**Cheapest falsification.** $0. Load RT-991/990/300. Within each `t` in the dominant cell, regress Arm C on `{T, T−t}`. Report cell AUC of (i) Arm C, (ii) the projection, (iii) the residual, (iv) Arm B. No training of a new model.

**Kill criterion.** Residual cell AUC − Arm B cell AUC < +0.01, or residual vs Arm B within-`t` ρ > 0.85. Then CASE 2 is *confirmed* and this family closes.

**Promotion criterion.** Residual cell lift ≥ +0.02 over Arm B, 5/5 folds on the diagnostic; student retains ≥ 30% of that lift on a nested fold-0; ensemble `marginal_vs_clone` ≥ +0.0015 with dominant net > 0 and pre-break damage < 0.015.

**Plausible TS-AUC upside.** If 25–40% of +0.07110 is shadowable: +0.018 to +0.028 *cell* ≈ +0.009 to +0.014 pooled before lockbox. After −0.0116 haircut economics, **+0.003 to +0.007** external is the honest band if the student retains a third to a half. Not +0.010 unless retention is unusually high.

**Probability of success.** 0.22 that the residual is material; 0.40 that we can distill a material residual if it exists. Joint ≈ **0.09** for a deployable +0.003. Diagnostic itself has probability ~1 of producing a decisive scientific number.

---

### Mechanism 2 — Per-series delay-cloud predictive null (nonparametric attractor occupancy)

**Core hypothesis.** Never-break false positives are series whose *stationary* DGP produces long on-attractor excursions that a moment-calibrated null prices as breaks. A per-series nonparametric predictive density in delay space prices those excursions correctly; a permanent scale/dependence change leaves the historical delay cloud.

**New information represented.** Local likelihood of the current delay vector `z_t = (x_t, x_{t-1}, …, x_{t-d+1})` under the historical occupation measure `μ_H`. Objects: `−log μ̂_H(z_t)`, k-NN radius, conformal p-value vs historical delay vectors, and the *dwell of off-cloud occupancy* (not the dwell of a CUSUM).

**Why the 500-bank probably does not contain it.** `m04_resid` is scalar AR/GARCH. `m03_dyn` has permutation entropy and lag-2 products. NEW_AVENUES §C.4–C.5 correctly noted that temporal order beyond lag-2 and delay-cloud occupancy are absent. RT-1215 tested a *rank-4 linear Hankel operator*, not a nonparametric occupation measure, and it was already in the bank’s span (ρ=0.8859).

**Why previous experiments do not falsify it.** RT-1215 = linear DMD, killed for redundancy, explicitly “does not falsify all delay-embedding observer ideas” (`pilot03_observers.md`). RT-1202 trajectory-geometry *scalar* had AUC 0.4993 — that implementation contained no signal; it is not a k-NN likelihood. CRF-02 showed amortized 8-float nulls lose to per-series fits; this mechanism *is* a per-series fit, which is the direction CRF-02’s derangement control motivates, and it is not trying to beat the AR+ECDF null at the AR+ECDF’s own job — it prices a different object (occupation, not one-step Gaussian residual).

**Causal inference state.** `μ_H` fitted on the break-free historical sample only, frozen before the first online step. Online: O(1) or O(log n_H) query.

**Training data construction.** No extra labels. History window of each series → delay vectors. Subsample to ~256–512 historical delay vectors per series (fits IM3’s “per-series historical-model cache” philosophy, 7 s / 10k series REPORT-CLAIM).

**Exact target.** Downstream still `y[t]`. The new *features* are the occupancy scores. Optional auxiliary: predict whether the *next* delay vector remains inside the historical 90% highest-density region (self-supervised, no `y`).

**Exact input representation.** 8–12 streaming columns: current occupancy surprise, EWMA of surprise, dwell outside q90, max dwell, fraction of last 64 steps outside, k-NN residual vs AR residual (the disagreement is the point), conformal p, running product of p (e-process on occupancy).

**Suggested estimator.** LightGBM specialist on {m00_core + occupancy block}, not a neural net on `y[t]`.

**Streaming inference.** History-only k-d tree or 2-D PCA histogram of delay vectors (PCA fit on history only). Per-step: form `z_t`, lookup. Target ≪ 1 ms/point.

**Interaction with RT-600 / RT-1257.** Eighth stream, or replace a weak slot, only if ρ vs RT-600 ≤ 0.70 *and* mature-vs-never net > 0. If ρ>0.85, it is RT-1215 again.

**Expected dominant-cell effect.** Primary: never-break FPs with on-attractor excursions should fall.

**Expected early-break effect.** Weak. Cloud occupancy needs enough online points to leave the cloud.

**Expected pair repair.** Specifically RT-600-high never-breaks whose online delay vectors remain in `μ_H` despite large moment-z. That is a disagreement channel, not a re-statement of `w256_z`.

**Expected pair damage.** True weak scale breaks that stay *on* a rescaled version of the same attractor (self-similar DGPs). Mitigate by also emitting occupancy on AR-residual delay vectors.

**Compute.** Fold-0 screen on 2,500-series store: 1–2 h including feature build. Kill before 5-fold.

**Leakage risks.** Fitting the cloud on any online point. Using series length to choose `d`. Cross-series pooling of delay vectors (that becomes Mechanism 3).

**Cheapest falsification.** On 200 dominant-cell never-break FPs and 200 mature-break TNs (from existing RT-600 OOF), compute batch k-NN occupancy surprise at `t=max`. If the occupancy disagreement with `m00_core` does not rank the 400 rows above 0.55 AUC, kill. No model training.

**Kill criterion.** Fold-0 `marginal_vs_clone` < +0.0010, or ρ>0.85, or dominant net < 0.

**Promotion criterion.** `marginal_vs_clone` ≥ +0.0015, ρ≤0.75, dominant net > 50, pre-break damage < 0.015, then 5-fold.

**Plausible TS-AUC upside.** +0.002 to +0.005 if the never-break FP mechanism is occupancy misspecification. Unlikely >+0.006 as a single stream.

**Probability of success.** **0.18.** The scientific direction is right; previous scalars in the neighbourhood died; the implementation has to be a likelihood, not another z.

---

### Mechanism 3 — Frozen cohort two-null residual (legal population reference)

**Core hypothesis.** Own-history calibration makes a heavy-tailed series look “hot” whenever it does what it always does. A second null — the empirical law of *training series at the same online age t* — identifies “unusual for me but typical for the world at age t.” The *disagreement* of the two nulls is new information. This is cohort distillation without illegal inference-time cross-section: the cohort is frozen into an atlas at training.

**New information represented.** For a calibrated surprise channel `u_{i,t}` (already own-history priced), the population residual `u_{i,t} − Q_t(u)`, where `Q_t` is a quantile function of the training cohort at age `t`. Vector of such residuals, plus the rank of `u_{i,t}` in the frozen atlas (a legal analogue of the illegal RT-131 within-t rank).

**Why the 500-bank probably does not contain it.** The bank’s null is *per-series historical windows*, not *other series at age t*. SmoothTimeCDFCal maps *scores* through a time-conditional CDF; it does not re-price *features* against a population two-sample geometry, and it cannot express disagreement between two reference measures.

**Why previous experiments do not falsify it.** RT-131 used live within-t ranks at inference (illegal; cross-section does not exist). m05_ctx used *series-constant* historical descriptors to predict break *propensity* and was a memorisation attack. LA-03 per-series adaptation failed a fixed-null isolation gate. None of these is a frozen, time-indexed, training-only atlas of feature quantiles. SS-03 null-state calibration was a different object (negative-side score calibration) and damaged pre-break pairs.

**Causal inference state.** Atlas `Q_t` computed on training folds only, nested. Inference: lookup `Q_t` by `t` (or `log n_seen` bin). No other test series, no RNG, no carried cross-series state. Determinism holds on a 10% rerun because the atlas is a frozen file.

**Training data construction.** Cross-fitted: for outer fold f, atlas from series not in f. Bins of `t` (the same 12 log-spaced anchors the calibrator already uses). Store 9 quantiles × ~20 top channels ≈ 2 k floats. Tiny.

**Exact target.** Downstream `y[t]`. The mechanism is a reference-measure change, not a new label. Optional teacher: illegal live within-t rank of RT-600 (the RT-131 object) distilled into a student that sees only `(own features, atlas lookup)`. That distillation is outside {features}×{learners}×{blend}.

**Exact input representation.** 20–40 columns: population residual and atlas-rank of `ab_fast`, `gle_pkr`, `glz_pk`, AR-residual log-sd, and a handful of m00 scale channels; plus the *sign disagreement* `1[own-history z high ∧ population rank low]`.

**Suggested estimator.** LightGBM on the disagreement block + m00_core, as a specialist. Alternatively, replace SmoothTimeCDFCal’s score CDF with a two-null score.

**Streaming inference.** Binary search in a 12-bin atlas. Microseconds.

**Interaction with RT-600 / RT-1257.** Recalibration of existing streams is safer than a new stream. If used as recalibration, the control is frozen current SCDF. Pair-flow on never-breaks is the gate.

**Expected dominant-cell effect.** Primary, on never-break FPs that are cohort-typical.

**Expected early-break effect.** Atlas at small `t` is noisier (fewer series have long online prefixes — actually more series are alive at small t). Ambiguous.

**Expected pair repair.** Own-history-hot, population-typical never-breaks drop. Own-history-mild, population-extreme weak breaks rise.

**Expected pair damage.** If the training cohort’s age-t law does not match test (distribution shift), we mis-center everyone. Lockbox is the check. Also: true breaks that look like the median of a world that already contains many mature breaks at large t — the atlas is contaminated by positives. **Mitigation:** build the atlas from *pre-break rows and never-break series only* (labels used at training, as in any supervised model). Using y in atlas construction is legal; using other test series at inference is not.

**Compute.** Atlas build: minutes. Fold-0 specialist: <1 h.

**Leakage risks.** Building the atlas on the scored fold. Using test-series statistics. Using `T` as a bin key. Putting series-constant atlas IDs into features (m05_ctx mode).

**Cheapest falsification.** From existing feature cache and OOF, compute for one channel (`ab_fast`) the within-t training quantile rank. AUC of `1[own z > 2 ∧ population rank < 0.5]` on dominant-cell never-break FPs vs mature breaks. If this indicator does not separate at AUC≥0.55, kill the family.

**Kill criterion.** Fold-0 `marginal_vs_clone` < +0.0010 or mature-vs-never net < 0.

**Promotion criterion.** +0.0015 marginal, 5/5, lockbox haircut not worse than incumbent, pre-break damage < 0.015.

**Plausible TS-AUC upside.** +0.002 to +0.006. This is the Alphabot lesson applied legally: a *different reference measure* is a different feature philosophy.

**Probability of success.** **0.24.** Cheap, legal, untried, aimed at the exact remaining error type.

---

### Mechanism 4 — Explicit-duration absorbing filter (semi-Markov / HSMM)

**Core hypothesis.** `m07_bayes` is the best module (gain/col 124.8k, +0.0397 on m00_core, REPORT-CLAIM) and its absorbing model and BOCPD both use a **geometric, memoryless** hazard. Memoryless dwell says P(still broken | hot for 100 steps) does not depend on 100 except through current evidence. A Weibull/lognormal duration prior says: a 100-step excursion is either absorbing or an extremely atypical transient for this DGP. That is the mature-break vs never-break likelihood ratio, and it is not a GBM feature.

**New information represented.** The joint posterior `p(state, duration | x_{1:t})` under a non-memoryless dwell. Specifically: posterior mass on {absorbing, dwell ≥ d0} versus {transient, expected remaining dwell}. Field of origin: **sequential clinical monitoring and hidden semi-Markov models** (explicit-duration HSMM; also discrete-time survival with a cure fraction). Transfer breaks: no biomarker mean shift; we run the HSMM on already-whitened residual energy, not on raw `x`. Budget is fine if we keep state dimension tiny. Determinism is fine if the filter is closed-form / fixed-grid.

**Why the 500-bank probably does not contain it.** Geometric hazard is explicit in NEW_AVENUES §C.6 as absent. `m01_seq` has *fraction* of time above q99 and time-since-peak — those are GBM-side dwell *scalars*. An HSMM changes the *posterior computation*. RT-1201 (IM2+dwell scalar) dying does not kill a filter replacement any more than `xc_n_hot` dying killed continuous `xc_max`.

**Why previous experiments do not falsify it.** RT-1201 dwell scalar KILL. SS-03 null calibrator KILL. Neither replaced m07’s duration model. Running-max as a *score post-transform* costs −0.006 (`STATE_OF_RESEARCH.md`); that is a ratchet on the output, not an HSMM.

**Causal inference state.** Filter on `(H_i, x_{1:t})` only. Duration prior parameters fitted on historical transients of *this* series (empirical excursion lengths in history) — still per-series, still causal.

**Training data construction.** History → empirical distribution of contiguous exceedance lengths under the historical null, used as the transient-duration prior. Absorbing-duration prior: Weibull with increasing hazard, hyperparameters frozen globally (not learned from y on the scored fold).

**Exact target.** The HSMM posterior `p(absorbing | x_{1:t})` *is* the score candidate. Optionally feed it as 6 columns into the existing tree (posterior, expected dwell, posterior entropy, MAP duration, Bayes factor vs geometric m07, disagreement with `ab_fast`).

**Exact input representation.** Prefer 6 columns over a new 50-col module. Input to the filter: one or two whitened energy streams (AR(6) residual sq, robust scale).

**Suggested estimator.** Closed-form forward recursion on a duration grid `{1,2,4,…,256}`. Not an EM fit per step.

**Streaming inference.** Duration-grid HSMM is `O(|D|)` per step. `|D|≈16` is cheap relative to 500 features.

**Interaction with RT-600 / RT-1257.** Replace or augment `m07_bayes::ab_fast`, the single most important column. If the new posterior is monotone in `ab_fast`, kill. The useful object is the *disagreement* with geometric `ab_fast`.

**Expected dominant-cell effect.** Primary. Duration information accumulates at age ≥ 100.

**Expected early-break effect.** Near zero (duration not yet informative). Correct: the metric is not won there.

**Expected pair repair.** Never-breaks whose hot streak is long but *typical of that series’ historical excursion length distribution* get down-weighted. Mature breaks whose dwell exceeds every historical excursion get up-weighted.

**Expected pair damage.** Series with no historical excursions (quiet history) have an uninformative transient-duration prior; the filter may treat the first long excursion as absorbing. That could inflate quiet never-breaks — the opposite of Mechanism 2. Cap the Bayes factor when historical excursion counts are low.

**Compute.** Fold-0: 1–2 h.

**Leakage risks.** Fitting Weibull hyperparameters on online y. Using true τ to set duration. Variable `|D|` that depends on `T`.

**Cheapest falsification.** On historical windows of never-break series, measure the empirical survival function of excursion length. If it is geometrically tailed, the HSMM collapses to m07 and we kill without training. If it is heavier- or shorter-tailed than geometric in a way that differs between never-break FPs and mature breaks at age≥100, proceed.

**Kill criterion.** Disagreement with `ab_fast` has univariate dominant-cell AUC < 0.52, or ρ>0.90, or `marginal_vs_clone` < +0.0010.

**Promotion criterion.** +0.0015 marginal, mature-vs-never net > 0, 5/5.

**Plausible TS-AUC upside.** +0.001 to +0.004. This is a refinement of the best existing module, not a new philosophy. Worth doing because the cost is low and the assumption (memoryless dwell) is known-false for transients.

**Probability of success.** **0.28.** Highest P(success) of the five; lowest upside. Good EV as a cheap experiment, not as a moonshot.

---

### Mechanism 5 — Per-series classifier two-sample test (C2ST) on history vs online windows

**Core hypothesis.** The bank’s two-sample geometry is a catalogue of named distances (KS, CvM, Wasserstein, energy, JS, Hellinger, GLR, …) on moments, PITs, and residuals. A *learned* two-sample test on raw windows can detect persistent differences that no named functional captures, without training on `y[t]` (so it cannot learn elapsed time).

**New information represented.** The running accuracy / logit of a classifier trained to distinguish historical windows from online windows *of this series*. After a true break the two samples become separable; after a transient they are not, or the separability decays. Field of origin: **classifier two-sample tests** (Lopez-Paz & Oquab 2017; used in GAN evaluation and two-sample testing), not from this repository’s sequential-statistics lineage.

**Why the 500-bank probably does not contain it.** Named distances span a specific function class. A small MLP or extra-trees classifier on windowed `(x, x^2, |x|, lag1)` can implement tests outside that class. CRF-01 was a *sequence ranker trained on y*, not a per-series two-sample classifier trained on window origin.

**Why previous experiments do not falsify it.** CRF-01 NNCSR trained on labels, hit the elapsed-time-adjacent ranking objective, damaged pre-break pairs at rate 0.2626. Wave 6 MLP trained on `y[t]`. m02_dist’s named distances are the control, not this test. A C2ST that never sees `y` cannot learn `t` except insofar as window length differs — match window lengths.

**Causal inference state.** Classifier fitted on history vs *causal* online windows using only origin labels `{history, online}`, never `y`. Refit on a schedule (e.g. every 32 steps) using only data ≤ t, or — cheaper and safer — fit once on history vs the first 32 online points as a *probe* and then freeze (that version only detects early change). Prefer: frozen *feature map* (random Fourier features of windows, fit on history) and a streaming Fisher discriminant between history and online (closed form, deterministic).

**Training data construction.** No `y`. Per series, length-matched windows.

**Exact target.** C2ST statistic `s_t = Φ^{-1}(AUĈ_t)` of the window classifier, plus its EWMA and dwell above a historical permutation null (permute online/history labels on historical data to get a per-series null of the C2ST itself). Downstream `y[t]` only at the tree that consumes `s_t`.

**Exact input representation.** 6 columns: current C2ST z, peak, dwell, permutation p-value, C2ST-on-residuals, disagreement with `m02` Wasserstein z.

**Suggested estimator.** Streaming Fisher / linear C2ST first (cheap falsification). Extra-trees C2ST only if linear dies for lack of capacity, not for lack of signal.

**Streaming inference.** Linear C2ST: maintain means and covariances of random features, O(k^2) with k≈16. Fits the budget.

**Interaction with RT-600 / RT-1257.** New specialist only if ρ≤0.70. If it correlates with m02_dist, it is a slower Wasserstein and we kill.

**Expected dominant-cell effect.** Medium. C2ST needs sample size; at t≥200 it has it. Aimed at weak persistent changes that named distances miss, including some false negatives (the 26% of cell loss that is not never-break).

**Expected early-break effect.** Poor. Do not spend on early t.

**Expected pair repair.** Mature breaks whose change is in a direction no named distance watches. Complementary to Mechanisms 2–4, which target never-break FPs.

**Expected pair damage.** Over-sensitive C2ST will separate *any* long online sample from history because of estimation noise (two-sample tests at large n detect trivial differences). **Mandatory:** permutation / length-matched historical-vs-historical null, the project’s own calibration philosophy. Without it this mechanism is toxic at large t — exactly the dominant cell.

**Compute.** Linear version fold-0: ~2 h.

**Leakage risks.** Training the window classifier on y. Not length-matching. No permutation null (guaranteed FP inflation at large t).

**Cheapest falsification.** For 100 series, compute batch linear C2ST at the last online point, permutation-calibrated. Univariate series-level AUC. If <0.53, kill. Compare to Wasserstein on the same windows; require +0.02 over that control or kill (otherwise it is m02).

**Kill criterion.** Not beating Wasserstein by +0.02 univariate, or ρ vs m02 > 0.80, or `marginal_vs_clone` < +0.0010.

**Promotion criterion.** Beats Wasserstein control, ρ≤0.70 vs RT-600, +0.0015 marginal.

**Plausible TS-AUC upside.** +0.001 to +0.004, concentrated on false negatives rather than never-break FPs.

**Probability of success.** **0.12.** High risk of rediscovering m02. Included because it is the cleanest *outside-lineage* detector and it targets the minority of remaining loss that Mechanisms 2–4 do not.

---

## 4. Rank the five

| rank | mechanism | EV | max upside | P(success) | complementarity vs RT-1257 | cost | complexity | leakage risk | information if fail |
|---|---|---|---|---|---|---|---|---|---|
| 1 | M1 T-orthogonal Arm-C residual | highest | +0.007 (+0.014 if retention is wild) | 0.09 deployable; ~1 as diagnostic | High if residual is real (new target, not new learner) | $0 diagnostic, then 2–5 h | Low diagnostic / medium distill | High if nested purity slips | **Decisive** for CASE 2 vs local-frame |
| 2 | M3 frozen cohort two-null | high | +0.006 | 0.24 | High: different reference measure, not CatBoost | <1 h | Low | Medium (atlas fold purity; positive contamination) | Tells us whether population-at-t is already implicit in SCDF |
| 3 | M4 HSMM duration | medium | +0.004 | 0.28 | Medium: refines m07, may be redundant with CatBoost’s splits on dwell-like columns | 1–2 h | Medium | Low | Tells us whether geometric hazard is binding |
| 4 | M2 delay-cloud occupancy | medium | +0.005 | 0.18 | High if ρ is low | 1–2 h + cheap probe | Medium | Low | Tells us whether never-break FPs are on-attractor |
| 5 | M5 C2ST | low | +0.004 | 0.12 | Medium (FN-oriented) | 2 h | Medium | High without permutation null | Tells us whether named distances span the two-sample class |

Complementarity ranking with RT-1257 specifically: M1 > M3 > M2 > M5 > M4, because CatBoost already re-splits the existing bank (M4 lives closest to that bank).

---

## 5. One moonshot

**Name.** Simulation-based amortized posterior after DGP identification (SNPE / ABC-SMC on a reconstructed generator).

**Hypothesis.** The data are plausibly synthetic from a low-dimensional family (scale, dependence, shape, mixture, with rare absorbing switches and frequent transients). If that generator can be identified to the point that simulated series are exchangeable with real ones on the taxonomy diagnostics (location AUC≈0.50, scale≈0.56, 92% individually undetectable, transient rates matched), then unlimited labelled series with known `τ` let us amortize `p(broken_by_t, family, permanence | prefix)` far past 8,000 noisy labels.

**Why <30%.** LA-02 already showed synthetic pairs can have MAJOR signal versus a *synthetic* clone (+0.00516) and still fail pair-flow and sit below RT-600 on real data (`STATUS.md` / leaderboard-alpha final). TGMC (Wave 8) made real fold-0 *worse* by adding synthetic rows. DGP misspecification is the default outcome.

**Why >+0.010 if correct.** A well-specified simulator plus SNPE is the actual Bayes posterior for this DGP, not a 500-dimensional proxy. The legal-prefix Bayes score is exactly the object CASE 2 claims we cannot reach with the current bank. If the bank is a strict subspace of that posterior, the gap between 0.626 and the legal ceiling is the moonshot.

**Causal student.** At inference, the amortized network maps the causal prefix (or a streaming sufficient statistic) to the posterior. No future, no cross-section, no RNG if we freeze weights and use deterministic inference (NPE, not MCMC).

**What breaks in transfer.** Unknown true generator; 15 h/week does not include training a huge simulator-identification loop on the Crunch runner (that work is offline); neural amortization reintroduces the elapsed-time trap unless the target is `p(τ≤t | prefix)` trained on simulated `τ`, not BCE on real `y[t]` with `t` as a feature. The last point is solvable: train on simulated prefixes with `t` held fixed inside each minibatch, or drop `t` entirely.

**Cheapest falsification before the moonshot.** Do *not* train SNPE. First: fit a tiny parametric generator (AR-GARCH + rare variance jump + t-noise) by matching the taxonomy table. Draw 2,000 simulated series. Train the *existing* RT-600 pipeline on simulated data, score on real fold-0. If transfer AUC is <0.55, the generator is not the DGP and the moonshot dies for ~1 h. This is the kill gate.

---

## 6. Arbitration analysis

**We primarily need a better detector, not a better arbiter of the current detectors.**

Evidence:

- Every learned combination of the seven streams lost to a parameter-free average (REPORT-CLAIM). That is what arbitration looks like when experts share a representation.
- SS-01 repair-damage arbiter: 0/0/0 dominant repairs, `marginal_vs_clone` −0.00031.
- SS-04 specialist-disagreement micro-router: 0/0/0, −0.00031.
- FIRST_SWEEP_SYNTHESIS: at least one specialist is correct on 66.3% of sampled RT-600 dominant mistakes, but majority-correct blend failures are only 6.3%, and *every* individual specialist had negative direct dominant-cell pair flow versus RT-600. Disagreement exists; it is not usable by routing.
- ORR (Wave 8): recoverability 0.737 on teacher-confirmed inversions, and the pair diagnostic moved (769/694) — then the metric did not, because the confirmed-pair population is too small a share of pair weight.
- T2: real standalone alpha, zero ensemble alpha. A better *view of the same information* does not arbitrate.

**Stacking contradiction (prompt §5A).** Both findings are locally true.

- Alphabot stacked **four disjoint feature philosophies** (Humberto classical tests, Mario random-search interactions, Rafael multi-representation distances, João meta-features) in a *batch* setting with a *known boundary*. Stacking has parameters to estimate because the inputs are not already the same number.
- We stacked **seven hyperparameter / subset variants of one 500-column bank**, scored by within-t AUC, where monotone transforms are free and streams sit inside a 0.010 band with fold noise 0.011. The equal average has no parameter variance; a stacker does.

The implication is not “stacking never works here.” It is “stacking is what you do *after* you have Mechanism 3’s reference measure or Mechanism 2’s occupancy view, not to the current seven.” If M2 or M3 produces ρ≤0.70 and standalone ≥0.58, I would *then* run one causal LOFO stack. Not before.

**If we still ran one routing experiment:** a *two-null disagreement gate*, not a specialist router. Let `d_t = 1[own-history z high ∧ frozen-atlas rank low]`. When `d_t=1`, shrink the incumbent score toward the atlas-implied quantile (never-break-like). When `d_t=0`, leave RT-1257 untouched. Cross-fitted atlas, no test-series state. Control: shrink at random with the same rate (must beat it by ≥0.0005). Kill if pre-break damage ≥0.015. This is arbitration *conditional on a new detector bit*, which SS-01 never had.

---

## 7. Oracle / information-ceiling experiment

**Purpose.** Separate information unavailable before `t` from information present in the prefix but not extracted. Also the cheapest Bayes-limit vs local-frame separator (prompt §5B).

**Why existing oracles do not decide this.**

- RT-900: VOID, missingness mask = label.
- W6-E2R: series-level, known boundary — different estimand.
- W7-D3R Arm C: full-sequence *features including T*, undecomposed.
- Wave 8: legal students of class-conditionally eligible future targets.

**Design — Arm-C residualization (no new training required for the decision).**

| role | information | artifact |
|---|---|---|
| Oracle C | legal features at `t` **plus** final-row 500-vector (includes `T` and post-`t` peaks) | `RT-991.npy` |
| Horizon oracle H | a within-`t` regression of logit(C) on `{T, T−t, t/T}` | computed |
| Residual oracle R | C after subtracting H | computed |
| Causal student A/B | legal 500 at `t` | `RT-300.npy` / `RT-990.npy` |
| Incumbent | RT-600 OOF | `wave5_S_specialist.npy` |

**Metric.** Dominant-cell TS-AUC and never-break-only cell AUC, 5 folds, same cell definition as W7-D0 (`t≥200`, age≥100). Also within-`t` rank-ρ of R vs B.

**Interpretation.**

| outcome | meaning | action |
|---|---|---|
| H ≈ C, R ≈ 0 | Arm C was remaining-horizon leakage. CASE 2 confirmed. Legal-prefix ceiling ≈ current system. | Stop prefix-information search. Spend on RT-1257 robustness, lockbox, submission. |
| R ≫ B, ρ(R,B) low | Shadowable future-path exists in the full sequence *and is not in the 500-bank*. Local frame, not Bayes limit. | Distill R (Mechanism 1). |
| R ≫ B, ρ(R,B) high | Future path is already linearly in the 500-bank; Arm C is a better readout. | Objective/target problem, not representation. Try Mechanism 1 student anyway; do not add features. |
| H carries most of C on never-breaks but not on pre-breaks (or vice versa) | Different remaining error types; route research at the type R actually moves. | |

**Optional second stage (only if R ≫ B):** a *causal* student trained to predict R from the prefix. If the student gets none of R, the information is in the future observations themselves (true ceiling). If it gets a fraction, that fraction is the remaining legal alpha.

**Frame falsification (prompt §5B).** “Near the Bayes limit” is the first row. “Stuck in a local frame” is the second. This uses artifacts that already exist. I did not run it in this session because aligning `T` onto the 5,036,517-row OOF index is a script, not an inspection, and I was forbidden to launch experiments. It should be the first thing the next 15 hours does.

---

## 8. What should we stop doing

Directions with enough evidence that continuing them is unlikely to produce +0.003:

1. **More capacity / more trees / more seeds on the 500 columns.** Arm B lost. CatBoost already harvested the learner-family residual (+0.002). TabM/RealMLP INFEASIBLE.
2. **Neural nets trained on `y[t]`** (MLP, TCN, Transformer, GRU, SSM) without an explicit anti-elapsed-time design. Wave 6 diagnosed the mechanism off the training curve. CRF-01 with a ranking objective still damaged pre-break pairs at 0.26.
3. **Stacking, weighting, LOFO subsetting, SS-01/SS-04-style routing of the current seven streams.** Repeatedly worse than the average.
4. **Feature pruning as research.** top-300 is a deploy-cost trick (−0.0004), not alpha.
5. **New Avenues scalar add-ons in the incumbent residual-scale/maturity direction** (ρ 0.86–0.89 family: scale survival, spectral impulse, Kalman/NIS, Hankel-DMD, unweighted/weighted CTM tuning). Closed. Do not retune RT-1216.
6. **Orthogonal scalars with no standalone signal** (RT-1202-class). Diversity is not the currency (`wave6_neural_results.md`; FIRST_SWEEP_SYNTHESIS).
7. **Unresidualized future-score distillation** (SST/ORR/PCFB/CFEP/TGMC repeats; T2-like teachers that correlate 0.91 with the incumbent). Wave 8 left a precise open question — *why* distillation dies on the full population — which Mechanism 1 answers by changing the *target*, not by inventing a sixth acronym.
8. **Learned amortized generative nulls with a tiny bottleneck.** CRF-02: the shipped per-series null *is* the right null for that job.
9. **Series-constant historical context as features.** m05_ctx.
10. **Running-max as a score post-transform.** −0.006.
11. **Mean-shift / location detectors.** Taxonomy AUC 0.4998.
12. **Treating the external 0.6290 as a tuning signal.** One calibration point.

Do **not** stop: per-series historical-null calibration (the foundation); pair-flow accounting; fold-purity sentinels; the dominant-cell / never-break loss cube as the targeting instrument.

---

## 9. Next three experiments

### Experiment A — Arm-C residualization (the ceiling test)

- **Hypothesis.** A material fraction of Arm C’s +0.07110 cell AUC is orthogonal to `{T, T−t}` and therefore is a legal teacher target.
- **Implementation.** Script over existing `RT-991/990/300.npy` + `folds.parquet` + per-series lengths. Within-`t` projection. No new model.
- **Control.** Arm B (`RT-990`); also a shuffled-`T` projection (must *not* kill C if C is real future-path).
- **Required data.** OOF vectors already on disk in the wave8 worktree.
- **Required compute.** Minutes to one hour. CPU.
- **Primary metric.** Dominant-cell TS-AUC of residual R vs Arm B.
- **Dominant-cell metric.** Same (this *is* the cell).
- **Pair-flow metric.** Never-break-only cell AUC of R vs B; pre-break-only cell AUC of R vs B.
- **Success threshold.** R − B ≥ +0.02 cell AUC.
- **Kill threshold.** R − B < +0.01, or ρ(R,B) > 0.85.
- **Information even on failure.** Confirms CASE 2. Authorises stopping prefix-information search. That is worth more than a weak positive.

### Experiment B — Frozen two-null atlas, fold-0 screen

- **Hypothesis.** Disagreement between own-history z and a nested training-cohort age-t atlas repairs never-break FPs without damaging incumbent-correct pairs.
- **Implementation.** Cross-fitted quantile atlas on ~20 channels; 20–40 disagreement columns; LightGBM specialist vs RT-401 clone protocol already in the CatBoost/New Avenues harness.
- **Control.** (i) incumbent; (ii) atlas built from *all* rows including positives (contamination control); (iii) deranged atlas (permute series’ atlas ranks).
- **Required data.** Feature cache + folds. Already present via wave3/wave7 worktree symlinks (`NEW_AVENUES_2026.md` §A.1).
- **Required compute.** <1–2 h fold-0.
- **Primary metric.** `marginal_vs_clone`.
- **Dominant-cell metric.** Dominant net pair flow; never-break-only net.
- **Pair-flow metric.** Dominant repairs/damage/net; pre-break damage rate.
- **Success threshold.** `marginal_vs_clone` ≥ +0.0015, dominant net > 0, pre-break damage < 0.015, deranged atlas worse by ≥0.0005.
- **Kill threshold.** < +0.0010 or dominant net < 0 or deranged ≥ real.
- **Information even on failure.** Whether a population reference is already implicit in SCDF calibration.

### Experiment C — Geometric vs explicit-duration m07, fold-0

- **Hypothesis.** Replacing m07’s geometric hazard with a per-series empirical-transient-duration prior plus an increasing-hazard absorbing Weibull produces a posterior whose *disagreement* with `ab_fast` has dominant-cell signal.
- **Implementation.** Duration-grid forward filter on AR(6) residual energy; emit 6 columns; specialist on {m00 + m07_old + m07_hsmm} vs {m00 + m07_old}.
- **Control.** Geometric m07 on the identical energy stream (matched). Also a dwell-*scalar* control (RT-1201-class) to confirm any gain is from the posterior, not from “we added dwell again.”
- **Required data.** Raw series + existing m07 code path.
- **Required compute.** 1–2 h fold-0.
- **Primary metric.** Univariate dominant-cell AUC of disagreement; then `marginal_vs_clone`.
- **Dominant-cell metric.** Mature-vs-never net.
- **Pair-flow metric.** As above.
- **Success threshold.** Disagreement univariate cell AUC ≥ 0.55 *and* `marginal_vs_clone` ≥ +0.0010 *and* beats dwell-scalar control by ≥0.0005.
- **Kill threshold.** Disagreement AUC < 0.52, or ρ(`ab_fast`)>0.90, or does not beat dwell-scalar.
- **Information even on failure.** Closes NEW_AVENUES §C.6 rather than leaving “geometric hazard” as an untested assumption.

These three answer: (A) is the ceiling real, (B) is the missing object a second reference measure, (C) is the missing object duration memory. They are not leaderboard shots.

---

## 10. Final recommendation

**If I controlled the next 15 hours of research compute and wanted the highest probability of eventually improving the external score by at least 100 basis points, what would I do?**

100 basis points external is +0.010. Nothing in the measured record supports that as a *probable* 15-hour outcome. RT-1257 moved +0.0022 external on a +0.002 internal. The lockbox haircut is −0.0116. A honest 15-hour plan maximises P(eventually +0.010), which is not the same as P(+0.010 this week).

**Allocation:**

| hours | work |
|---|---|
| **2.0** | **Experiment A** (Arm-C residualization). Non-negotiable. Uses existing npy. Decides whether the rest of the 13 hours is information-seeking or robustness-seeking. |
| **1.0** | Cheap probes for B and C (atlas indicator AUC; historical excursion survival vs geometric). Kill either before training. |
| **If A says R ≈ 0 (ceiling confirmed):** 8 h RT-1257 deployment hygiene and alternate-partition confirmation of the CatBoost hybrid (the brief notes no alt-partition evidence); 2 h lockbox protocol for any future +0.002-class candidate; 2 h stop. Do not spend the 8 h on another detector. External +0.010 is then probably unavailable without more labelled data or a DGP crack, and the moonshot’s 1 h generator-transfer kill gate is the only remaining information bet. |
| **If A says R ≫ B (local frame):** **5 h** Mechanism 1 fold-0 nested distillation of R, with the T2 purity sentinel, eligibility *not* restricted to `t+h` (full population, residual defined everywhere). **3 h** Experiment B if the probe passed. **2 h** Experiment C if the probe passed. **1 h** slack for pair-flow / ρ gates. Promote nothing that fails dominant net or the 0.015 pre-break cap. |

I would **not** spend the 15 hours on: another CatBoost slot, a Transformer, Wave 8.1, SS-05, TabM, or reading this report’s mechanisms as five features to dump into the 500-bank.

The single highest-EV action in the entire repository right now is not a novel architecture. It is **residualizing `RT-991.npy` against series length**, a file that was incorrectly believed missing. Every subsequent claim about a causal-information ceiling, including mine, should be postponed until that number exists.

---

## Appendix — assumptions that may be wrong

Required by the execution brief Q4; folded here so they stay attached to evidence.

1. **Assumption:** W7-D3R CASE 2 (future-information limit). **Why wrong:** Arm C includes `T`. **Lost if wrong:** the entire shadowable-path research lane. **Cheap test:** Experiment A. **Upside if wrong:** +0.003 to +0.007 legal.

2. **Assumption:** feature-bank saturation = prefix-information saturation. **Why wrong:** Arm B saturates *this* subspace. CRF-01’s 0.593@ρ0.45 shows other subspaces exist and are currently too weak, not that they cannot be made strong. **Lost if wrong:** Mechanisms 2, 3, 5. **Cheap test:** occupancy / atlas probes. **Upside:** +0.002 to +0.006.

3. **Assumption:** geometric / memoryless duration is adequate because trees see time-since-peak. **Why wrong:** trees see a scalar; they do not compute a duration-conditioned posterior. RT-1201 dying is the same error as `xc_n_hot` dying. **Lost if wrong:** HSMM lane. **Cheap test:** historical excursion survival. **Upside:** +0.001 to +0.004.

4. **Assumption:** the only correct null is per-series historical. **Why wrong:** that null is why never-breaks that do what they always do look broken. A population-at-t null is a different question. **Lost if wrong:** Mechanism 3 and the legal reconstruction of Alphabot’s disjoint-philosophy stacking. **Cheap test:** atlas indicator. **Upside:** +0.002 to +0.006.

5. **Assumption:** row-level BCE is an adequate TS-AUC surrogate for trees. **Why wrong:** `t_online` is a top-10 gain feature and is constant in `AUC(t)`. Trees resist the trap via `min_data_in_leaf=300`; they still spend splits on a direction the metric deletes. **Lost if wrong:** capacity that could have gone to never-break disambiguation. **Cheap test:** drop all elapsed-time columns, retrain one specialist, compare TS-AUC (must not fall) and dominant-cell AUC (must rise if the assumption is wrong). **Upside:** +0.000 to +0.002, plus every future neural idea becomes viable if we train on a t-ablated target.

6. **Assumption:** distillation failure ⇒ future is unpredictable from the prefix. **Why wrong:** Wave 8 targets were class-conditionally eligible and not T-orthogonal; T2 was redundant, not empty. **Lost if wrong:** Mechanism 1. **Cheap test:** Experiment A then a single nested student. **Upside:** as M1.

7. **Assumption:** low correlation without standalone AUC is a research clue (keep hunting orthogonal views). **Why wrong:** RT-1202 AUC 0.4993; TCN ρ=0.21 worth +0.0001. **Lost if we keep doing it:** weeks. **Cheap test:** refuse any candidate with standalone <0.55 from 5-fold. **Upside of dropping it:** opportunity cost.

---

## Independence and process note

- I read both source prompts in full, then investigated this worktree, `STATUS.md`, `FAILED_EXPERIMENTS.md` (partial), `STATE_OF_RESEARCH.md` (stale), CatBoost/CRF/learner-diversity finals, New Avenues synthesis, RT-1257 submission record, W7-D3R, W6 corrected oracle, Wave 8 final, the exact alpha-budget JSON, and loaded `RT-991/990/300.npy`.
- I did not read any other agent’s response.
- I did not modify model code, features, RESULTS.csv, STATUS.md, or the source prompts.
- I did not allocate experiment IDs or train.

---

Local write is blocked on that worktree. Could not push a file to GitHub from here without a permitted path. Copy the markdown above into:

`research/multi_agent_frontier_20260829/responses/agent_05_grok46.md`

**Agent:** grok-4.6
**Five mechanisms:** T-orthogonal Arm-C residual; delay-cloud occupancy null; frozen cohort two-null; HSMM duration filter; C2ST
**#1 experiment:** residualize `RT-991.npy` against series length T
**No other agent response was read.**