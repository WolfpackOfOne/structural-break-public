# Independent Frontier Analysis — deepseek-v4-pro (opencode)

AGENT_ID: `deepseek_v4_pro`

This report was written from the execution brief (`chatgpt_research_prompt.md`), the evidence brief (`claude_code_research_prompt.md`), and direct inspection of: `STATE_OF_RESEARCH.md`, `STATUS.md`, `break_taxonomy.md`, `wave7_d3r.md` + `WAVE7_D3R_PREREG.md`, the Wave 8 final report (`research/wave8-future-aware-distillation`), `NEW_AVENUES_2026.md`, `FIRST_SWEEP_SYNTHESIS.md`, `pilot03_im2_dwell.md`, `SECOND_SWEEP_PREREG.md`, `LEADERBOARD_ALPHA_2026_FINAL.md`, and the CRF/Leaderboard Alpha accounts in `STATUS.md`. No other agent response was read.

---

## 1. Executive diagnosis

**What the project does well.** This is the most disciplined change-point program I have seen reconstruct. It correctly discovered, independently and by measurement, every structural fact about the data that matters: location breaks do not exist (`break_taxonomy.md`, series AUC 0.4998); scale and dependence are the only living families (best single statistic AR(6)-residual log-sd at 0.603); 92.4 % of break series are individually sub-threshold (`weak_unclassified`); and the false-positive population is a *DGP*, not a defect (17.0 % transients vs 15.6 % in matched break-free history). It built a 500-column bank whose most valuable inputs are "shape and persistence of accumulated evidence" — exactly the transient-vs-permanent disambiguator the forensics said was required. It answered the running-max question correctly (peak-as-feature is load-bearing, peak-as-post-transform is harmful, −0.0060). It prices everything against a `−0.0116` lockbox haircut and a seed-clone marginal. That last habit — the pervasive use of `marginal_vs_clone` rather than standalone AUC — is the single most important thing this project does correctly, and most of the field does not.

**Where it is wrong, and where I believe the true bottleneck is.** The program has concluded, at least three times (`H-D` in the CRF account, `CASE 2` in W7-D3R, and `FIRST_SWEEP_SYNTHESIS.md` H6), that it has hit a *causal information ceiling*. The conclusion is *directionally* right but rests on a number that has **never been decomposed**, and as long as that number is un-decomposed, the program cannot tell "near the Bayes limit" from "stuck in a frame." The number is Arm C's `+0.07110` cell AUC (`wave7_d3r.md`), the single largest measured signal gap in the repository, and the decomposition is explicitly **blocked** (`RT-991.npy` unverified/missing). I believe the true bottleneck is that this decomposition is the *only* open question that matters, and everything else — five future-aware distillation pilots, two preregistered sweeps, 79 catalogued mechanisms — has been aimed at "what future signal can we import," when the correct question is "**how much of the future signal is importable at all**."

**Classification across the requested axes:**

- **Learner capacity — NOT the bottleneck (measured).** W7-D3R Arm B (same 500 columns, 127 leaves / ff 1.0 / 900 trees) beat Arm A on 1/5 folds and was `−0.00592` pooled. More model on the same representation is a *closed lane* (`D.4`). This is not a prior; it is the cleanest extraction-ceiling measurement in the repo, and I accept it.
- **Feature-bank saturation — real but misdiagnosed.** The 500 columns are collectively non-redundant (top-300 costs −0.0004, top-60 costs −0.0174), so the bank is not "full" in the redundancy sense. But `FIRST_SWEEP_SYNTHESIS` H3 is right that every *viable* new feature collapses into the incumbent residual-scale / maturity / e-process direction (ρ 0.86–0.89). The bank is saturated *against the incumbent's information*, not against new information per se.
- **Representation limitation — partially, and more subtle than stated.** The bank is a bank of *level statistics and detector-path levels* (`NEW_AVENUES_2026.md` §C). The one functional that `§C` identifies as missing — contiguity / excursion *duration* — was then built *with the correct matched-length null* (`pilot03_im2_dwell.md`) and still produced `+0.000301` marginal (KILL). So the "missing state variable" hypothesis, which `NEW_AVENUES` was largely built on, **has already been tested at the feature level and failed.** This is the most important negative a naive reading of `NEW_AVENUES_2026.md` misses, because that document was written *before* the sweep ran.
- **Objective mismatch — partially.** `RT-123` pairwise and `RT-131` rank-average both failed to beat binary BCE / logit-average. Objective surgery is largely a measured dead end (`wave6` neural analysis raised the same mechanism: the row target is monotone in `t` and TS-AUC deletes `t`). I do not recommend more of it.
- **Target limitation — the ripest but still mostly refuted.** Privileged/auxiliary targets (soft ramp, log-hazard, scale_pos_weight, XE-NDCG) all lost to binary `1[t≥τ]`. T2 distillation *did* produce real single-model alpha (+0.00943, 5/5 folds, CI>0) — but +0.00024 ensemble marginal. So privileged targets buy single-model power that the ensemble already owns.
- **Arbitration/routing — exhausted, not under-explored.** SS-01 through SS-04 all KILL. The first-sweep meta-finding is the key fact: every candidate repairs ~92 % of RT600-wrong pairs *and* damages ~83 % of RT600-correct pairs; repair-vs-damage is **not separable** with available causal state (SS-01's own kill interpretation). Routing is closed.
- **Compute / historical-null estimation — healthy.** Streaming port is the only outstanding deployment risk and does not gate research. Null calibration is the project's crown and the CRF-02 result (fixed per-series null beats a learned amortized null) *confirms* the shipped null is right.
- **Genuine causal-information ceiling — the real bottleneck, but with a caveat that changes the strategy.** I accept H-D in spirit: the practical limit is the legal prefix. But the *quantitative* basis for that belief is Arm C's +0.071, which conflates two very different things (next section). The program has been treating the whole +0.071 as "future-break-path evidence that distillation failed to import," when a substantial and possibly dominant fraction of it is *remaining-horizon / endpoint* information that is **non-shadowable in principle** — which, if true, means the causal ceiling is lower than the program thinks *and* the remaining shadowable gap is smaller and should be spent differently.

**The one-sentence diagnosis:** this is not a feature problem, a learner problem, or an arbitration problem; it is an **estimand-confusion problem** — the program has been left chasing a +0.071 oracle gap whose decomposition was never completed, and the highest-value single action in the repository is to finish that decomposition, because its answer redirects the entire remaining budget.

---

## 2. Repository evidence

I distinguish **MEASURED FACT** (reproduced/consistent across ≥2 artifacts or a single pre-registered run) from **INFERENCE** and **HYPOTHESIS**.

1. **MEASURED FACT — Arm C ≫ Arm B on the dominant cell.** `wave7_d3r.md`: `C−B = +0.07110` cell AUC, 5/5 folds, per-fold range +0.05277…+0.08180; `B−A = −0.00592`. C's never-break-only cell AUC 0.71439, pre-break-only 0.73070 (vs A's 0.65413). Arm C broadcasts each series' *final-online-row* 500-feature vector to every row; it is explicitly non-causal and illegal. *Establishes:* adding the destination state of each series is decisively informative for ranking the *current* row. It does **not** establish what part of that destination is shadowable.

2. **MEASURED FACT — the eligibility diagnostic.** Wave 8 pre-flight (`wave8_eligibility_diagnostic.json`, cited in `wave8_final.md` §C): target-availability (`t+h` in-series) is class-conditional at matched `t` — e.g. `h=200, t<20`: `P(eligible | y=1) = 0.33` vs `P(eligible | y=0) = 0.81`. *Establishes:* oracle future-states are only *defined* where the series survives to `t+h`, and that survival is strongly correlated with the label. **This is remaining-horizon / `n_online` information, the very thing the program repeatedly calls worthless and non-shadowable.** I read this as direct evidence that Arm C's broadcast (which literally puts the final-row `t_online = n_online` into every row as a feature) is substantially contaminated by non-shadowable endpoint information.

3. **MEASURED FACT — oracle-teacher utility collapses on the full population.** Five Wave 8 pilots: oracle utility positive, legal utility ~0 or negative. SST `+0.00616` oracle → `−0.00060` legal (full pop) vs `+0.00810` legal (eligible slice only); PCFB `+0.01100` oracle → `−0.00148`/−0.00270 legal; every `RT600+candidate` marginal ≈ `−0.00031`. *Establishes:* the future-signal that works on the "eligible" slice does **not** survive contact with the full row population. The eligible/full split is exactly the remaining-horizon conditioning axis. **INFERENCE:** the oracle gain was financed by horizon leakage, not by shadowable future-break regularity.

4. **MEASURED FACT — ORR repairability is real but non-aggregating.** `wave8_final.md` §E: 877 teacher-confirmed inversions; `P(repair correct | RT600 wrong, teacher confident) = 0.737`; 769 repairs / 694 damage; aggregate `−0.00005`. *Establishes:* a genuine, high-probability repair mechanism exists and still moves the metric by nothing. **INFERENCE:** precision on a small population cannot move a metric whose loss is diffuse (§5, E.1).

5. **MEASURED FACT — the repair/damage asymmetry.** `FIRST_SWEEP_SYNTHESIS.md`: any first-sweep candidate repairs 91.8 % of sampled dominant RT600-wrong pairs *and* damages 83.0 % of RT600-correct pairs; dominant net is negative for every arm. `E.1`: loss carried by the worst 5 % of series is only 0.158 (never-break) / 0.202 (break) — **the loss is diffuse, not concentrated.** *Establishes:* there is no small repairable population, so any *targeted* repair is a dead end and any *broad* re-rank must be correlated with the true label near the boundary, not merely decorrelated from RT600.

6. **MEASURED FACT — contiguity/duration was built correctly and failed.** `pilot03_im2_dwell.md` (`RT-1201`): matched-length run percentile using an "exact interval-union matched-length history null" + episode-mass + growth proxy over 3 windows; within-t ρ 0.3817 vs RT600 (genuinely decorrelated), standalone 0.5835, dominant-cell 0.610, but **marginal vs clone `+0.000301`**, pair net `−1151`. *Establishes:* the excursion-duration signal is real and decorrelated but too weak to lift the ensemble, *even with the correct run-length null* (the infrastructure `D6` demanded). **INFERENCE:** "build the missing contiguity state variable" is now falsified *at the feature level*, not merely untried.

7. **MEASURED FACT — a privileged teacher gives real single-model alpha but no ensemble alpha.** `wave7_teacher_nested` / `STATUS.md`: T2 `+0.00943` standalone, 5/5 folds, CI [+0.0043,+0.0137]; ensemble marginal `+0.00024`. Original pilot was outer-fold contaminated (+0.0122 to +0.0235); corrected nested scheme 0/20 violations. *Establishes:* distillation transmits *something real*, but it is already priced by the incumbent blend.

8. **MEASURED FACT — the only external movement is a specialist *replacement*, not an *addition*.** `RT-1257` (`engineering/rt1257-deployment-2026`, submission #16): external **0.6290**, +0.0022 over the 0.6268 anchor; it replaces `RT-300` and `RT-413` with a two-slot CatBoost hybrid. Internal `marginal_vs_clone +0.002407`, `E2−E0 +0.002026`. *Establishes / INFERENCE:* the leverage point in 2026 is **replacing an ensemble member with a better one**, not adding a ninth stream. An additive stream's marginal is bounded by its redundancy with seven incumbents; a *replacement* can capture single-model alpha directly.

9. **MEASURED FACT — CRF-02 fixed null beats learned null.** `RT-1240` vs `RT-1241/42`: learned-null gate `−0.021446`; derangement control *passed* (+0.003377), so the negative is amortization, not memorization. *Establishes:* the shipped per-series historical null is the right null; learned generative nulls are closed.

10. **REPORT-CLAIM NOT YET VERIFIED — the canonical `RESULTS.csv` is a wave-1 fossil** (77/78 rows `git_sha=nogit`, still labels `RT-131` "NEW CHAMPION"). I did not rely on it for any number above; every number I cite came from the report files or the branch reports, and where a number is important I flag its estimand.

The single most load-bearing pair is (2)+(3): the future-aware program's oracle successes were *conditioned on remaining horizon*, and their legal failures were the same quantity scored on the whole population. I take this as the strongest evidence anywhere in the repo that **Arm C's +0.071 is dominated by non-shadowable endpoint information**, and that the correct target is smaller than the program believes.

---

## 3. Five high-upside mechanisms

### Mechanism 1 — Forward-horizon broadcast decomposition (unblock the ceiling number)

- **Name:** FHD — forward-horizon decomposition sweep.
- **Core hypothesis:** Arm C's `+0.07110` is a convex combination of (a) remaining-horizon / `n_online` / endpoint information (non-shadowable, worthless) and (b) genuine future-break-path regularity (partially shadowable). They have opposite strategic implications, and a single sweep separates them.
- **New information represented:** the *information-accumulation curve* of the future: how much of the oracle gap arrives at forward horizon Δ = 0, 20, 50, 100, 200, 400 vs the endpoint.
- **Why the 500-feature system does not already have it:** it is a diagnostic over the *training oracle*, not a runtime feature. Nothing in the bank or the sweeps computed a horizon-resolved oracle gap.
- **Why previous experiments do not falsify it:** W7-D3R broadcast **only the final row** (Δ = endpoint), the single most horizon-contaminated choice possible. Wave 8 consumed five *distillation mechanisms* but never re-measured the oracle gap as a function of Δ. Neither is a horizon sweep.
- **Causal inference state:** diagnostic only; no runtime causal claim.
- **Training data construction:** reuse the cached feature memmaps (`cache/features/m0*.npy`). For each series, build Arm-B-style rows but broadcast the feature vector evaluated at row `min(t+Δ, n_online−1)` (or emit NaN where `t+Δ` exceeds the series), for the Δ ladder above.
- **Exact target:** binary `1[t≥τ]`, unchanged; LightGBM, Arm B config, same 5 folds, same seed.
- **Exact input representation:** the same 500 causal columns + 500 broadcast columns at each Δ (final-row broadcast is Δ=∞ included as the ceiling reference).
- **Suggested estimator/model:** LightGBM, Arm-B hyperparameters, one training per Δ.
- **Streaming inference procedure:** none; offline.
- **How it interacts with RT600/RT1257:** it does not, directly — it *prices* every other mechanism.
- **Expected dominant-cell effect:** the outcome is the whole point; see kill/promotion.
- **Expected early-break effect:** weak (diagnostic is scored on the dominant cell).
- **Expected pair-repair mechanism:** none directly.
- **Expected pair-damage risk:** none (no deployment).
- **Compute requirements:** ~1 full run per Δ; roughly 6–8 × 6.5 min/fold on the fixed 2-core host. SMALL–MEDIUM.
- **Leakage risks:** it is deliberately non-causal; must never touch the lockbox or a submission path. Treated as a diagnostic artifact, exactly as `break_taxonomy.parquet`.
- **Cheapest falsification experiment:** this *is* the falsification experiment.
- **Kill criterion:** if the lift is flat across Δ, the ceiling is genuinely long-horizon.
- **Promotion criterion:** if Δ=50–100 already captures ≥ half of the endpoint lift, short-horizon persistence is shadowable and Mechanism 2 becomes high-EV.
- **Plausible TS-AUC upside:** 0 directly; it licenses or kills the *real* upside.
- **Probability of success (of a *decision-relevant* result):** ~0.8 — the two outcomes are both decisive.
- **Why this first:** it is the highest-information-per-dollar action available and it is the explicit blocker the evidence brief invites me to route around ("If you can design a way around that blocker, say so prominently").
- **Addendum — the decomposition does NOT need `RT-991.npy`.** Nobody needs to recover the missing Arm-C vector. Re-run Arm C *per Δ* on the cached memmaps, which are present. The endpoint (Δ=∞) reproduction is the equivalent of `RT-991`, produced from first principles rather than a possibly-corrupted vector. This is the "way around the blocker."

### Mechanism 2 — Reversible-transient latent filter (non-absorbing, explicit-duration)

- **Name:** RTLF — reversible-transient latent filter.
- **Core hypothesis:** the transient-vs-permanent decision is a *latent-state filtering* problem, and the incumbent's two Bayesian objects are both **absorbing / single-change-point** (`m07_bayes` absorbing posterior; BOCPD). Neither can represent a deviation that *reverts*. Therefore, on a never-break series with a 100-step transient, evidence for "break" accumulates *monotonically* and never corrects — which is precisely the 74 %-of-cell-loss population (never-break negatives). A reversible three-state filter (null ↔ transient → broken, broken absorbing, transient with an empirical dwell distribution ~20–60 steps taken from `break_taxonomy.md` §F) yields `P(broken | prefix)` that **saturates and then declines** when a deviation outlives the transient's characteristic lifetime without an absorbing signature.
- **New information represented:** the posterior mass split between "currently in a reverting transient" and "currently in the absorbing broken state" — a function of dwell time that no current column computes. `m07_bayes` gives `P(some change occurred)`, which is *not* the same object.
- **Why the 500-feature system does not already have it:** `m07_bayes` and BOCPD are both single-change-point; the taxonomy's transients are explicitly two-change-point (in, then back) events. The bank has peak/decay channels but no *reversion* channel.
- **Why previous experiments do not falsify it:** `D1` (explicit-duration BOCPD) was **queued but never executed** — the first sweep's killed arms were feature *scalars* (`RT-1201` dwell; `RT-1210` joint rarity), not a probabilistic filter with a reversion state. `pilot03` proved the *scalar* dwell is too weak; it did not test a *posterior* that can actively subtract evidence.
- **Causal inference state:** posterior at time `t` uses only data ≤ `t`.
- **Training data construction:** fit the transition/dwell/emission parameters on the break-free historical segment per series (legal; history is visible at t=0), exactly as `NullCal` and CRF-02's fixed null already do. Emit the causal filter posterior trajectory as new columns.
- **Exact target:** binary `1[t≥τ]` (compatible with the incumbent head) *and*, as an ablation, `1[t≥τ]` with the filter posterior's own rank already embedded.
- **Exact input representation:** add a small block (≈10–20 columns): the filter's `P(transient)` / `P(broken)` / dwell-exceeded indicator / posterior-odds slope, on the AR-residual scale, at the incumbent's matched-length null calibration.
- **Suggested estimator/model:** keep LightGBM; the filter is a feature generator, not a replacement learner. (If it shows margin, promote to an eighth stream.)
- **Streaming inference procedure:** O(1)/step absorbing-extension filter (the exact same order as `m07_bayes`); trivially streamable.
- **How it interacts with RT600/RT1257:** additive stream first; if the additive marginal clears, replacement candidate for a weak specialist (§2.8).
- **Expected dominant-cell effect:** directly targets the never-break 74 %; expected to *lower* false-positive ranks, which is the only dense, metric-relevant correction.
- **Expected early-break effect:** ~nil (mature cell is the target).
- **Expected pair-repair mechanism:** repairs RT600-over-ranked never-break series by a *dense* correction (subtract evidence on over-long transients), not a sparse one.
- **Expected pair-damage risk:** the historical risk for any new stream ≈ −0.0003 marginal; the deposit here is that the correction is *label-correlated* (never-break side), unlike the decorrelated candidates of the first sweep.
- **Compute requirements:** SMALL; reuses cached residuals.
- **Leakage risks:** none (causal, per-series, history-fitted).
- **Cheapest falsification experiment:** fold-0, dominant cell, `marginal_vs_clone` with the seed-clone control.
- **Kill criterion:** `marginal_vs_clone < +0.0015` OR pair net ≤ 0.
- **Promotion criterion:** `≥ +0.0030`, 5/5 or 4/5-positive folds, never-break pair net positive.
- **Plausible TS-AUC upside:** +0.002…+0.006 if the never-break mis-ranking is as dense as `E.1` suggests.
- **Probability of success:** ~0.25 — the honest caveat is that all feature-level persistence signals have failed; the *posterior-form* is the only untried degree of freedom.

### Mechanism 3 — Generative simulation-based amortized posterior (the weak-nudge mass)

- **Name:** GEN-SBI — reverse-engineer the DGP, amortize the posterior on unlimited simulated series.
- **Core hypothesis:** 92.4 % of breaks are a *tiny distributional nudge spread over almost every series* (`weak_unclassified`, excess −3.92 pp vs placebo). The only way to see a nudge like that is to aggregate over many rows *with the correct null*, and a hand-crafted bank can only enumerate finitely many nulls. A simulator that reproduces the DGP (heavy-tailed history, matched-length transients, scale/dependence nudge without mean shift) can label an *unlimited* number of series with known τ and train a *learned* evidence aggregator / amortized posterior sharper than any hand-typed prior.
- **New information represented:** a learned mapping from the residual/null-normalized prefix (and its shape/memory) to `P(break | prefix)`, fit on a distribution the hand bank only samples.
- **Why the 500-feature system does not already have it:** it is an empirical-statistic bank; every posterior (`m07`) is an analytical prior, not a learned one.
- **Why previous experiments do not falsify it:** LA-02 used *counterfactual synthetic augmentation of existence labels* on real rows; TGMC used a *location-displacement* simulator — the one break type this data does **not** contain (`break_taxonomy.md`: location AUC 0.4998). Neither reverse-engineered the actual scale/dependence-nudge DGP, and neither trained an *amortized* estimator.
- **Causal inference state:** student reads only prefix; τ used only to label the simulated set.
- **Training data construction:** fit the simulator to measured marginals (historical kurtosis/Hill index, transient rates 17 %/15.6 %, AR(6)-whitened scale nudge of ~0.6 AUC strength); generate e.g. 100k labeled series.
- **Exact target:** `1[t≥τ]` on simulated data; then transfer the encoder as features or finetune on real dev.
- **Exact input representation:** the incumbent null-normalized residual streams (not raw, per `NEW_AVENUES` O4 and the neural collapse in Wave 6).
- **Suggested estimator/model:** a small causal sequence head over the null-normalized stream; but the *transfer* may be as frozen embedding columns into LightGBM.
- **Streaming inference procedure:** causal encoder, O(1)/step state.
- **How it interacts with RT600/RT1257:** additive stream; also a possible replacement.
- **Expected dominant-cell effect:** targets the 74 % never-break side by a sharper null.
- **Expected early-break effect:** low.
- **Expected pair-repair mechanism:** dense, label-correlated re-ranking.
- **Expected pair-damage risk:** same as M2.
- **Compute requirements:** MEDIUM (simulation cheap; training the encoder needs CPU-only care given the torch/OpenMP segfault trap).
- **Leakage risks:** must prove the simulator does not memorize the label; use a derangement control as CRF did.
- **Cheapest falsification experiment:** does the learned posterior beat `m07_bayes::ab_fast` (the incumbent's best Bayesian feature) on the *real* dev fold-0 dominant cell, standalone, before any ensemble work.
- **Kill criterion:** does not beat `ab_fast` standalone.
- **Promotion criterion:** beats `ab_fast` and adds `≥ +0.0015` marginal vs clone.
- **Plausible TS-AUC upside:** +0.003…+0.010 if the nudge is real and learnable.
- **Probability of success:** ~0.15 — effectively the moonshot *dressed as a mechanism*; I separate a cheaper version (simulator-once) from the true moonshot (§5).

### Mechanism 4 — Cohort-conditional null reconstruction (cross-sectional calibration at training)

- **Name:** COHORT-CAL — learn the cohort-conditional distribution to shadow the illegal within-timestep rank.
- **Core hypothesis:** the metric is cross-sectional, and the one thing the deployed model cannot see is the cohort. But the cohort *distribution* at time `t` is a training-time observable. A per-series calibrator that learns `F_t(score | history-fingerprint)` — the within-timestep conditional CDF *given the series' own heavy-tail/long-memory fingerprint* — can reproduce part of the illegal rank transform with only per-series inputs.
- **New information represented:** the *fingerprint-conditional* cohort position, not the unconditional SCDF the incumbent ships (12 log-spaced anchors on `log_n_seen`).
- **Why the 500-feature system does not already have it:** the shipped SCDF is unconditional in fingerprint; `E.2`/`D3` measured that never-break false-positive rate is fingerprint-predictable (ρ +0.192 OOF) — heavy tails, long memory, calm history — but that signal was only ever tested as *raw columns* (`m05_ctx`, rejected) or a *single scalar* (`RT-1212`, killed), never as a **cohort-conditional CDF**.
- **Why previous experiments do not falsify it:** SS-03 (negative-side null calibrator) was killed on a narrow *pre-break damage cap* (0.0199 > 0.0150), not on the never-break side where `D3` says the signal lives; and it used a static partition, not a learned fingerprint-conditional map.
- **Causal inference state:** causal (fingerprint from history, t from elapsed).
- **Training data construction:** at training, compute the within-t cohort rank of each score; regress it on (score, t, fingerprint) cross-fitted.
- **Exact target:** the within-timestep rank percentile of the incumbent score.
- **Exact input representation:** score, `t`, and the ~23 history-fingerprint columns already validated in `E.2`.
- **Suggested estimator/model:** a low-depth regressor or monotone GAM (few parameters; keep it auditable).
- **Streaming inference procedure:** O(1) table/light-model lookup per row.
- **How it interacts with RT600/RT1257:** a *calibration* layer, not a new stream — the one place the project has not yet tried a learned fingerprint map.
- **Expected dominant-cell effect:** reduce never-break over-ranking (74 % side).
- **Expected early-break effect:** ~nil.
- **Expected pair-repair mechanism:** dense re-calibration of a large slice.
- **Expected pair-damage risk:** lowering; this is its draw vs a new stream.
- **Compute requirements:** TINY.
- **Leakage risks:** must be cross-fitted; the within-t rank is an illegal *inference-time* object and must be distilled, never emitted.
- **Cheapest falsification experiment:** fold-0, never-break dominant-cell pair net + `marginal_vs_clone`.
- **Kill criterion:** never-break pair net ≤ 0 or `marginal_vs_clone < +0.0010`.
- **Promotion criterion:** 5-fold `≥ +0.0030`, pre-break damage under cap.
- **Plausible TS-AUC upside:** +0.002…+0.005.
- **Probability of success:** ~0.20.

### Mechanism 5 — Decay/reversion asymmetry as an explicit evidence *channel* (not a scalar)

- **Name:** ASYM — asymmetric charge/discharge evidence with a counted reset.
- **Core hypothesis:** the incumbent evidence accumulators are all monotone or fixed-half-life (`m01_seq` decayed peak; `apply_persistence` measured). Nothing charges fast and discharges *fast with a counted reset*, which is the functional form that distinguishes a one-shot transient (charge, then a single reset event) from a persistent break (no reset, sustained charge). The *reset count* is the novel state variable, and `NEW_AVENUES` B5 names it but the sweep never shipped it as a *channel bank*.
- **New information represented:** number of complete excursion→re-entry cycles (auto-recloser count), asymmetric decay, dropout-vs-pickup hysteresis — on the AR-residual scale channel.
- **Why the 500-feature system does not already have it:** every accumulator is symmetric/monotone; `NEW_AVENUES` §C.3 already concedes this.
- **Why previous experiments do not falsify it:** `RT-1201` was dwell (duration), and the relay family (B1–B6) was rated TINY but **not among the executed pilots**; the executed relay variant was `RT-1200` score-state (a *post-transform* on the score, measured close to the known-harmful `apply_persistence` family), not an *evidence-channel* reset counter.
- **Causal inference state:** causal.
- **Training data construction:** history-fitted thresholds; emit cycle counts and hysteresis state per window.
- **Exact target:** binary `1[t≥τ]`.
- **Exact input representation:** ≈8–12 new columns on the residual-scale channel.
- **Suggested estimator/model:** LightGBM (existing head).
- **Streaming inference procedure:** O(1) accumulator.
- **How it interacts with RT600/RT1257:** additive; low priority vs M1–M4.
- **Expected dominant-cell effect:** target the transient-vs-permanent boundary.
- **Expected early-break effect:** low.
- **Expected pair-repair mechanism:** reset-count disambiguation of transients.
- **Expected pair-damage risk:** same as any additive stream (−0.0003 class).
- **Compute requirements:** TINY.
- **Leakage risks:** none.
- **Cheapest falsification experiment:** fold-0 dominant-cell pair net.
- **Kill criterion:** pair net ≤ 0.
- **Promotion criterion:** 5-fold `≥ +0.0030`.
- **Plausible TS-AUC upside:** +0.001…+0.003.
- **Probability of success:** ~0.12 — I include it because it is the cheapest way to get an out-of-repo idea (protective-relay logic, IEEE C37.112 reset dynamics) onto a *channel* rather than the already-measured score transform.

---

## 4. Rank the five

| rank | mechanism | EV | max upside | P(success) | complementary to RT1257 | cost | complexity | leak risk | info if it fails |
|---|---|---|---|---|---|---|---|---|---|
| 1 | M1 FHD decomposition | very high (decides everything) | 0 (routes) | 0.8 | — | small | low | guardrail | settles the #1 open question |
| 2 | M2 RTLF reversible-transient filter | high | +0.006 | 0.25 | high (new posterior object) | small | med | none | kills/confirms absorbing-model misspec |
| 3 | M3 GEN-SBI amortized posterior | med | +0.010 | 0.15 | high (learned, non-tabular) | med | high | med | kills/confirms nudge learnability |
| 4 | M4 COHORT-CAL fingerprint-conditional CDF | med | +0.005 | 0.20 | med (calibration layer) | tiny | low | low | settles SS-03's narrow kill |
| 5 | M5 ASYM reset-count channel | low | +0.003 | 0.12 | med | tiny | low | none | closes the relay family |

Ordering rationale: M1 is not a mechanism-to-ship but it is the highest-EV action in the entire repository, so I rank it first despite zero direct upside. M2 is the only remaining *information-hypothesis* that (i) targets the 74 %-never-break loss, (ii) is a **posterior-form** rather than a feature-scalar, and (iii) was never executed. M3 is the highest-ceiling but lowest-confidence. M4 is cheap and attacks a specific known narrow kill. M5 is a completeness item.

---

## 5. One moonshot

**Full DGP reverse-engineering + amortized likelihood-ratio estimator trained on unlimited simulated series with known τ** — the "neural likelihood-ratio between NOT-YET-BROKEN and ALREADY-BROKEN" that `STATE_OF_RESEARCH.md` already names as the theoretically correct object for this metric and never built.

Concretely: estimate the DGP from the measured marginals and dependence structure (heavy-tailed history with matched-length transients at the measured rates; a scale/dependence innovation nudge with **no mean shift**; the τ tilt; the historical length standardisation), then train a causal sequence encoder to estimate the likelihood ratio `p(prefix | broken) / p(prefix | not-broken)` directly (TReND/NRE style), on an arbitrarily large simulated corpus where τ is known exactly. The encoder is applied causally at inference; τ never appears as a runtime input.

- **Probability of success:** < 0.30 (I estimate ~0.10–0.15).
- **Path to > +0.010:** if the residual causal gap is in fact *representational* rather than *informational* for the weak-nudge mass — i.e., a learned, high-dimensional aggregator over the correct null sees a nudge the 500 hand statistics do not — this is the only mechanism in the program with the representational capacity to surface it, and it attacks precisely the 92.4 % that carries the metric.
- **What makes it a genuine moonshot and not just M3:** M3 is the cheap simulator-once + transfer; the moonshot is the full NRE amortization where the estimator *is* the model, with no hand bank at all — high-risk, high-ceiling, and the thing that would actually falsify the "information ceiling" rather than nibble at its edges.

---

## 6. Arbitration analysis

**We need a better detector, not a better arbiter — but only in one very specific sense, and "better detector" is itself the wrong frame.**

The evidence that arbitration is *not* the lever:

1. **Repair/damage is not separable (MEASURED).** Any first-sweep candidate repairs 91.8 % of RT600-wrong pairs *and* damages 83.0 % of correct ones (`FIRST_SWEEP_SYNTHESIS.md`). SS-01 (the repair-damage arbiter designed exactly to learn that separation) was **KILL** (`marginal_vs_clone −0.000310`, 0 contributing sensor families). SS-04 (specialist disagreement router) was **KILL** *despite* the specialist decomposition showing ≥1 specialist correct on 66.3 % of RT600 mistakes — because the correct-specialist signal is not locatable from causal state.
2. **The loss is diffuse (MEASURED).** Worst 5 % of series carry only 0.16 of the loss (`E.1`). ORR's genuinely-correct repairs (0.737) moved the metric by −0.00005 because the repairable population is too small. Routing a sparse repair cannot move a diffuse metric.

The evidence that "just add a detector" is *also* not the lever: every standalone detector with real signal (dwell 0.61, trajectory, spectral, ordinal…) collapses to ρ 0.86–0.89 with the incumbent or nets negative pair flow. So the honest answer is:

**Neither analysis, as the prompt phrases it, is the bottleneck. The bottleneck is that the incumbent is already near the causal-prefix Bayes limit, and the remaining headroom was mis-measured.** The correct next detector is *not* another anomaly statistic and *not* a router over existing ones — it is the **single horizon-free posterior** (M2) or the **single learned null-conditional aggregator** (M3) that asks a question none of the incumbents asks: *"is this deviation currently consistent with a permanent regime change, as opposed to a transient that happens to have lasted this long?"*

If I must name one concrete causal routing experiment (as the brief requires), it is **not** member-mixture routing but a **negative-side-only arbitration** over the never-break population: train a cross-fitted classifier on the 23 history-fingerprint columns (`E.2`/`D3`) plus the incumbent score to output a *shrinkage* factor applied to the score only when the fingerprint indicates heavy-tail/long-memory/calm-history (the false-positive-prone subpopulation). This is distinct from SS-03 (which used a static partition and was killed on the pre-break side, not the never-break side) and from `RT-1212` (which died to a deranged control because it was a *scalar*, not a fingerprint-conditional CDF). Promising but folded into M4, because the program's own evidence says routing is exhausted.

---

## 7. Oracle / information-ceiling experiment

**Question:** is the remaining gap fundamentally *causal* (information genuinely absent before `t`) or *representational* (present in the prefix, not extracted)?

**Design (this is M1 formalized as the decisive experiment):**

- **Oracle arms:** Arm-C style broadcast of the feature vector evaluated at forward horizon Δ ∈ {0, 20, 50, 100, 200, 400, ∞(endpoint)}, using the cached memmaps, LightGBM Arm-B config, same 5 folds, same seed. `Δ=0` is the legal-prefix baseline (Arm A/B); `Δ=∞` reproduces Arm C.
- **Causal student arms:** the incumbent (Arm A) and, as the representational test, Arm B (same info, more capacity).
- **Information available to each:** oracle arms see `t+Δ` (and, for Δ=∞, `n_online`); student arms see only the prefix through `t`.
- **Metric:** dominant-cell TS-AUC (t≥200, age≥100), plus the never-break vs pre-break split, per fold and pooled.
- **Comparison:** the *profile* `cell-AUC(Δ)`. The quantity that matters is the **fraction of the endpoint lift already captured at short Δ**, `g(Δ) = [AUC(Δ)−AUC(0)] / [AUC(∞)−AUC(0)]`, and its shape.

**Interpretation of each outcome:**

- **If `g(Δ)` rises steeply to Δ≈50–100 and then plateaus** (short-horizon dominant): the gap is *representational*, and the missing object is a *short-horizon persistence* quantity that **is** causal — M2/M3 become the main line, and the "future-information limit" verdict is *wrong in the way it matters* (Case 1/3, not Case 2).
- **If `g(Δ)` grows roughly linearly in Δ all the way to the endpoint** (long-horizon dominant): the gap is genuinely *causal/long-horizon*, and its main ingredient is the endpoint `n_online`/remaining-horizon leakage the eligibility diagnostic already implicated — then the ceiling is real and *lower than assumed*, and the correct remaining work is variance reduction / calibration / submission economics (§8), not signal.
- **Mixed** (steep to Δ≈100, then a second slow rise to endpoint): decompose into a short-horizon shadowable component (attack with M2) and a long-horizon non-shadowable component (stop chasing it).

**Why this is decisive where the prior answer was not:** W7-D3R proved `C≫B` (future matters) and Wave 8 proved "importing the endpoint fails," but neither measured *how much future is enough*, so the program could neither bound the shadowable fraction nor kill the endpoint-leakage explanation. This experiment does both. It requires no new data, no new model class, and no recovery of `RT-991.npy`; it runs on artifacts that already exist.

---

## 8. What should we stop doing?

1. **Future-aware distillation in the Wave-8 mold.** Five mechanisms, five KILLs, all with the same signature — oracle wins on the eligible slice, ~0 on the full population. The `SST/ORR/PCFB/CFEP/TGMC` family is measured closed (`wave8_final.md` §I). Do not fund a sixth "distill the future into a causal student" unless it explicitly conditions out remaining horizon (and M1 is the prerequisite that would even tell us that is possible).
2. **Additional scalar anomaly/contiguity feature blocks.** `RT-1201` built the correct matched-length run null and still got `+0.000301`; the dwell/spectral/ordinal/joint-rarity/trajectory families are closed (`FIRST_SWEEP_SYNTHESIS` "Closed Lanes"). The `NEW_AVENUES` "missing state variable" thesis has been tested and failed at the feature level; do not re-run it as one more column.
3. **Generic stacking / weighting / subset selection / learned combination.** Every learned combiner loses to the equal average; LOFO subsets and stacks are all negative. This is one of the most-replicated negatives in the repo.
4. **Raw neural learners on the 500 columns.** Wave 6 MLP/TCN failed with a *diagnosed mechanism* (the monotone-`t` trap that trees survive); CRF-01/CFEP repeated the failure on the same class. A sequence model is only justified as an *encoder over the null-normalized stream* (M3), never as a raw-channel regression.
5. **More tree capacity / hyperparameter search on the same bank.** Arm B's negativity plus the learner-diversity sweep (TabM/RealMLP infeasible, CatBoost marginal +0.001) make "same features, stronger learner" a closed lane.
6. **Pursuing the "detect the exhaustive break" framing at all.** Location breaks are AUC 0.4998; trend breaks collapse to 0.51 under honest class-matched comparison. Any classical change-point detector aimed at a mean shift is a dead family in this data.

These are not "we ran out of ideas" — they are individually falsified, most with a diagnosed mechanism, and continuing any of them has expected value below the +0.003 promotion bar.

---

## 9. Next three experiments

**Experiment 1 — Forward-horizon decomposition sweep (M1).**

- **Hypothesis:** Arm C's `+0.071` is dominated by non-shadowable endpoint / remaining-horizon information, not by short-horizon shadowable signal.
- **Implementation:** broadcast feature vectors at Δ ∈ {0,20,50,100,200,400,∞} on the cached memmaps; LightGBM Arm-B config; 5 folds; report `g(Δ)`.
- **Control:** Δ=0 (Arm A legal-prefix) and Δ=∞ (Arm C reproduction) bracket the curve.
- **Required data:** `cache/features/m0*.npy` (already present), folds 0–4.
- **Required compute:** ≈ 7 trainings ≈ 7 × 6.5 min/fold on 2 cores. MEDIUM.
- **Primary metric:** dominant-cell TS-AUC pooled + per fold.
- **Dominant-cell metric:** `g(Δ)` profile, never-break vs pre-break split.
- **Pair-flow metric:** repair/damage of the Δ=50 arm vs RT600.
- **Success threshold:** `g(50) ≥ 0.5` ⇒ short-horizon is shadowable.
- **Kill threshold:** `g(Δ)` linear in Δ to the endpoint ⇒ ceiling is long-horizon/endpoint.
- **Expected information on failure:** even a linear `g(Δ)` *settles the single most important open question* and redirects all remaining budget away from signal work.

**Experiment 2 — Reversible-transient latent filter (M2) prototype on the dominant cell.**

- **Hypothesis:** a non-absorbing three-state filter yields a never-break-side posterior that is label-correlated where the incumbent over-ranks, in a way the dwell *scalar* (`RT-1201`) could not express.
- **Implementation:** fit per-series transient dwell from the historical segment; emit `P(broken)/P(transient)/reset-count` columns on the AR-residual scale; add to the bank; run the incumbent LightGBM head; measure `marginal_vs_clone`.
- **Control:** seed clone (the binding control) + the `RT-1201` dwell columns as a lower-bound control.
- **Required data:** cached residual memmaps + historical segment (t=0-visible).
- **Required compute:** SMALL (fold-0 first).
- **Primary metric:** fold-0 `marginal_vs_clone`.
- **Dominant-cell metric:** never-break cell AUC + pair net.
- **Pair-flow metric:** never-break net (must be positive).
- **Success threshold:** `marginal_vs_clone ≥ +0.0015`, never-break net > 0.
- **Kill threshold:** never-break net ≤ 0 or marginal < +0.0015.
- **Expected information on failure:** closes the "absorbing-model misspecification" hypothesis — the last untried *posterior-form* degree of freedom — cleanly.

**Experiment 3 — Fingerprint-conditional cohort calibration (M4).**

- **Hypothesis:** never-break false-positive rate is predictable from the 23 history fingerprints (`D3` ρ +0.192), and a learned fingerprint-conditional CDF can shadow part of the illegal within-timestep rank, reducing never-break over-ranking.
- **Implementation:** cross-fitted regressor of within-t rank percentile on (incumbent score, t, 23 fingerprints); apply as a per-series monotone calibration.
- **Control:** the incumbent unconditional SCDF; a deranged-fingerprint control; SS-03's static partition.
- **Required data:** incumbent OOF scores + fingerprints (present).
- **Required compute:** TINY.
- **Primary metric:** fold-0 `marginal_vs_clone`.
- **Dominant-cell metric:** never-break cell AUC.
- **Pair-flow metric:** never-break net; pre-break damage rate (must stay < 0.015).
- **Success threshold:** `marginal_vs_clone ≥ +0.0010`, never-break net > 0, pre-break damage under cap.
- **Kill threshold:** deranged control matches candidate, or pre-break damage > 0.015.
- **Expected information on failure:** distinguishes "SS-03 died of a narrow pre-break cap artifact" from "the fingerprint signal is too weak to monetize at all."

These three answer the program's three unresolved scientific questions (is the ceiling causal; is the latent misspecification load-bearing; is the negative-side signal monetizable), rather than adding three more leaderboard attempts.

---

## 10. Final recommendation

"If I controlled the next 15 hours of research compute and wanted the highest probability of eventually improving the external score by ≥ 100 basis points, I would…"

**Hour 0–5 (settle the only number that matters):** run **Experiment 1**, the forward-horizon decomposition sweep. This is the highest-information-per-dollar action in the repository and costs ~7 trainings. It unblocks `RT-991` without needing the missing vector, and its outcome dictates everything below. It also directly serves the 300-basis-point ambition: if `g(50) ≥ 0.5`, there is a *shadowable* short-horizon component the program has never targeted; if not, no signal investment of any size will produce +0.003, and the budget should be reallocated to variance reduction.

**Hour 5–12 (the one remaining untried information hypothesis):** prototype **Experiment 2**, the reversible-transient latent filter, on fold-0. This is the only mechanism in my list that (a) targets the 74 %-never-break loss, (b) is a posterior-form rather than a feature scalar, and (c) was *queued but never executed*. If it gates (`≥ +0.0015`), promote to 5-fold and price it as a **specialist replacement** (the leverage point the external +0.0022 of `RT-1257` demonstrated), not an additive stream.

**Hour 12–15 (cheap, parallel, finish-safe):** run **Experiment 3** (fingerprint-conditional calibration) in parallel with M2's 5-fold confirmation. Both are cheap; both attack the never-break side; both are the kind of *calibration/replacement* work that the diffuse-loss finding (`E.1`) says is the only dense lever left.

**Allocation philosophy:** roughly 50 % of the 15 hours to the decomposition (M1), 35 % to the reversible filter (M2), 15 % to calibration (M4) — with **zero** hours to a new anomaly statistic, a sixth distillation mechanism, a ninth ensemble stream, or any neural learner on the raw channels, all of which are individually falsified with diagnosed mechanisms.

**The honest probability statement:** my best estimate is that the causal-prefix Bayes limit is near 0.63–0.65 and much of the +0.071 oracle gap is non-shadowable, which puts the *information* path to +0.003–0.010 at low-to-moderate probability (~0.25 across M2/M3/M4 combined). But the decomposition (M1) is near-certain to be *decision-relevant*, which is why it gets the largest share: in a program that has already mined the easy signal dry, the scarce resource is *knowing whether the remaining signal exists at all*, and no other single artifact in the repository answers that question.

---

## Return block

- **Agent/model identity:** `deepseek_v4_pro` (opencode).
- **Five mechanisms:** (1) Forward-horizon broadcast decomposition; (2) Reversible-transient latent filter; (3) Generative simulation-based amortized posterior; (4) Cohort-conditional null reconstruction; (5) Asymmetric reset-count evidence channel.
- **#1 recommended experiment:** the forward-horizon decomposition sweep (M1).
- **Confirmation:** no other agent response was read.