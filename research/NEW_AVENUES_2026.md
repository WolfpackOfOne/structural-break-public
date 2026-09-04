# DEEP RESEARCH — NEW AVENUES FOR REAL-TIME STRUCTURAL BREAK DETECTION

**2026 ADIA Lab / CrunchDAO Structural Break Challenge — Real-Time Edition**
**Written 2026-08-24 on `research/current` (`a425798`). Research, design and prioritisation only.**
**No experiment ID consumed. No promotion. No submission. No headline OOF claimed.**

Companion artifacts:
`research/new_avenues_2026.csv` (machine-readable candidate database),
`research/reports/new_avenues_2026_diagnostics.json` (every number below),
`research/scripts/novel_streams/` (harness + the diagnostic runners),
`research/scripts/novel_streams/PREDECLARE.md` (written before D4–D6 produced a number).

---

## A. REPOSITORY / EMPIRICAL GROUND TRUTH

### A.1 What the repository actually is

Verified by `git fetch --all --prune`, `git branch -vv`, `git worktree list`, `git tag -l`
and `git log --all --graph` on 2026-08-24. The three-branch model in the driving brief is
correct as far as it goes, and incomplete:

| ref | what it holds |
|---|---|
| `origin/main` | package / portfolio landing branch (Phases 1–5). No `research/`. |
| `origin/production/rt600` | frozen RT-600 lineage, tag `rt600-production-0.6268`, submission records, `crunch test` artifacts. |
| `research/current` | **canonical active research lineage.** wave2 → wave3-integration → wave5-alpha → wave6-alpha → wave7-teacher-distillation → wave7-t2-promotion, reorganised into `reports/waveN` + `archive/`. |
| `chore/repo-cleanup-2026` | in-flight cleanup PR; adds `AGENTS.md`, removes `research/`. This is what a fresh `structural-break` checkout lands on — **it is not the research branch.** |
| `research/wave8-future-aware-distillation` | **Wave 8 lives here and was never merged.** Tag `wave8-future-aware-final`. |
| `research/multi-agent-2026`, `codex/*`, `claude/*` | earlier parallel tracks and standalone studies. |

Two things the brief did not say and a new agent must know:

1. **Wave 8 is not on `research/current`, but its reusable harness already is.** The Wave-8
   *report* and its five killed mechanisms live only on the sibling branch and stay there.
   The *shared evaluation helpers* were carried forward long ago and sit at
   **`research/scripts/wave8_common.py`** on this branch — `ensemble_marginal`,
   `pair_repair_stats`, `nested_oof_regressor` and the forbidden-column guard, function-only,
   with no future-aware mechanism code (commits `4b04983` and `22e479d`, both ancestors of
   this branch). **Import them; do not copy them from a historical branch, and do not
   reimplement them.** Nothing about using them makes Wave 8's failed mechanisms part of the
   active research programme.
2. **The full research machine exists locally and runs.** The 10,000-series float32 store
   and all seven 500-column feature caches sit in
   `structural-break-claude-wave3/cache/`, and every worktree symlinks to them. The
   `research/wave7-t2-promotion` worktree already carries store + features + folds + all
   RT-300/RT-4xx/RT-990/RT-995 OOF vectors and is the cheapest place to start.

### A.2 Verification of the brief's stated numbers

Every load-bearing number in the driving brief was re-derived from the repository, not
copied. All confirmed:

| brief claim | verified value | source |
|---|---|---|
| RT-600 external 0.6268 | 0.6268 (LB-001) | `STATUS.md`, `EXPERIMENT_ID_MAP.md` |
| RT-600 pooled dev ≈ 0.63828 | **fold-0** = 0.638285; **dev mean** = 0.62581, **dev pooled** = 0.625627 | recomputed here from the seven specialists |
| dominant cell = t≥200, age≥100 | confirmed | `WAVE7_RT600_EXACT_ALPHA_BUDGET.md` |
| ~50.5 % of metric pair weight | 0.5050 | W7-D0 |
| ~45 % of remaining loss | 0.4529 | W7-D0 |
| cell AUC ≈ 0.664 | **0.66428** (recomputed) | this document |
| ~74 % of cell loss vs never-break | 0.7399 | W7-D0. **Independent cross-check:** my row-wise decomposition splits each pair's loss half to the positive and half to the negative, and gives never-break series 0.370 of the total — exactly 0.74 × 0.5, since every positive lives in a break series. The two decompositions agree to three decimals. |
| Arm C − Arm B ≈ +0.05…+0.08, 5/5 folds | +0.05277 … +0.08180, 5/5 | `wave7_d3r.md` |
| T2 = +0.00943 standalone, 5/5, CI>0 | confirmed | `wave7_teacher_nested`, `RESULTS.csv` |
| T2 ensemble E2−E1 = +0.00024 | confirmed (E0 0.63828 / E1 0.63859 / E2 0.63882) | `wave7_t2_promotion_final.md` |
| Wave 8: five mechanisms, all KILL | confirmed, all marginals ≈ −0.0003 | `wave8_final.md` |
| ~500 causal features, 7 LightGBM specialists, time-conditional cross-fitted calibration | confirmed | `FINAL_ARCHITECTURE_FREEZE.md` |

I reproduced the RT-600 development blend from the seven committed OOF vectors and got
**0.62581 mean / 0.625627 pooled / 0.66428 dominant-cell** — bit-consistent with W7-D0.
Everything below is computed on that exact array.

**One correction to the brief's framing.** The brief describes the Wave-8 acronyms
correctly (SST / ORR / PCFB / CFEP / TGMC) and warns against the NotebookLM report's
invented expansions. It is right to. But the brief also describes our current bank as
"approximately 500 causal streaming features" in a way that undersells what is in them —
see §B. Most of §18–§26 of the brief is **already built**. That is the single most
important fact for aiming the next wave, and it is why this document spends §I on a
collision check rather than a wish list.

---

## B. WHAT OUR EXISTING SYSTEM ACTUALLY KNOWS

Read from the source, not from the handoffs. The 500 columns are seven designed modules,
each with its own historical-null calibration engine:

| module | cols | what is genuinely inside it |
|---|---|---|
| `m00_core` | 151 | 7 transforms × 6 trailing windows + 18 expanding transforms, each as a robust z and an empirical percentile against **the distribution of the same statistic over every length-matched historical window**. |
| `m01_seq` | 60 | CUSUM, CUSUM-SQ, Page–Hinkley, Shiryaev–Roberts, dyadic GLR, mixture-GLR, multi-half-life EWMA — each exposed as *seven shape channels*: current, running peak, decayed peak, time-since-peak, **fraction of elapsed time above the historical q99**, local slope, current/peak ratio. Calibrated by re-running the identical recursion on history. |
| `m02_dist` | 59 | PIT occupancy over historical quantile bins → χ², JS, Hellinger, TV, Cramér–von Mises, KS, 1-D Wasserstein, energy distance, rank-CUSUM, Darling–Erdős — all null-calibrated at matched window length. |
| `m03_dyn` | 60 | ACF/PACF-style dependence, Goertzel-exact windowed DFT (band powers, spectral entropy, slope, dominant frequency), Haar wavelet energies, **permutation entropy / ordinal complexity**, variance ratios — on adaptive quarter/half/full windows. |
| `m04_resid` | 60 | Nine whitenings fitted on history and filtered causally forward: raw, AR(1)/AR(2)/AR(5), ridge-AR(3), Huber-IRLS AR(2), EWMA-vol, robust EWMA-vol, GARCH(1,1), AR+vol. Each monitored on mean/var/abs/tail/ACF(1)/ACF(1)-of-squares plus a Gaussian GLR. |
| `m06_loc` | 60 | Online change-point **localisation**: geometric bank of candidate segment lengths, statistics of the *estimated post-break segment*, both history-referenced and pre-referenced, on AR(6) residual squared / log-abs / lag-1 product. |
| `m07_bayes` | 50 | Exact absorbing-state filtering posterior (integrates over every changepoint in O(1)/step), BOCPD with NIG observation model and historical prior, **e-values / test martingales** (plug-in GRAPA + mixture-over-betting-fractions + Vovk's power martingale on conformal p-values), sequential Bayes factors. |

Two more modules exist and were measured: `m11_focus` (exact `max_τ (S_t−S_τ)²/2(t−τ)`)
and `m12_rdep` (residualised distances, residual CUSUM/CUSUMSQ paths, AR-coefficient LR).

**Therefore, before proposing anything, the following are NOT new here:**
per-series historical-null calibration; AR/robust-AR/ridge-AR/GARCH/EWMA residual streams;
CUSUM / Page–Hinkley / Shiryaev–Roberts / GLR / mixture-GLR; BOCPD; e-processes and
conformal test martingales; KS / CvM / Wasserstein / energy distance / JS on both raw and
residual PIT; spectral entropy, band powers, spectral slope, dominant frequency, Haar
wavelet energy; permutation entropy; exact maximisation over the candidate change point;
running peak / decayed peak / time-since-peak / current-over-peak; and the *fraction* of
elapsed time above a historical threshold.

The architecture on top: seven LightGBM specialists differing in feature subset, leaves,
sampling policy, row budget, objective and boosting type; each passed through a frozen
cross-fitted **smooth time-conditional CDF** calibration (`log_n_seen` coordinate, 12
log-spaced anchors); equal-weight arithmetic mean. Weighting, stacking, subset selection
and gating were all measured and all lose.

---

## C. WHAT IT APPEARS NOT TO KNOW

Reading the same source with the opposite question. The bank is a bank of **level
statistics and detector-path levels**, all calibrated against the distribution of
**rolling means over historical windows**. What has no representation anywhere in it:

1. **Contiguity.** `m01_seq` computes `per = cumsum(d > q99)/t` — the *fraction* of elapsed
   time above threshold. No column anywhere computes the **length of the current
   contiguous excursion**, the **running maximum of that length**, the **number of
   separate excursion episodes**, the **inter-episode gap**, or the **area accumulated
   since the last re-entry into the null band**. A heavy-tailed stable series and a broken
   series can have identical exceedance *fractions* and completely different *episode
   structure*. §E shows this is exactly the confusion that dominates the remaining loss.
2. **Run-length / excursion nulls.** `NullCal` calibrates rolling means. There is no
   per-series empirical null for any *path functional* — longest run, excursion area,
   first-passage time, number of episodes. So even where a run statistic were computed,
   the project's own calibration philosophy could not currently price it.
3. **Reset / hysteresis dynamics.** Every accumulator in the bank either grows
   monotonically, decays at a fixed half-life, or is a plain rolling window. Nothing has an
   **asymmetric charge/discharge time constant**, a **dropout threshold below its pickup
   threshold**, or a **counted reset event**.
4. **Trajectory similarity.** Nothing in 500 columns compares the online window to the
   history *as a shape*. No nearest-neighbour distance, no motif, no arc-crossing, no
   delay-embedded point cloud, no recurrence structure. Every statistic is a marginal or a
   low-order-moment functional; **temporal order beyond lag-2 products is essentially
   discarded**.
5. **Multi-dimensional dynamics.** `m04_resid` fits scalar AR predictors. Nothing fits a
   **Hankel / delay-embedded linear operator** and monitors its reconstruction residual or
   its **effective rank** — the object that distinguishes "transient exploring the full
   state space" from "settled on a lower-dimensional attractor".
6. **Explicit duration priors.** `m07_bayes`'s absorbing model uses a geometric hazard `h`;
   BOCPD uses a geometric hazard. Both impose a memoryless dwell time. There is no
   **explicit-duration (semi-Markov)** run-length prior anywhere.
7. **Impulsiveness vs repetitiveness in the frequency domain.** `m03_dyn` measures where
   the energy *is*. Nothing measures whether the online excess energy arrives as one
   impulse or as a sustained/repetitive structure — the spectral-kurtosis / cyclostationary
   distinction.
8. **Joint (size × duration) rarity.** Every surprise channel prices "how far" or "how
   long" separately. Nothing prices `P(deviation ≥ s lasting ≥ d)` jointly under the
   series' own null.
9. **A scalar difficulty conditioner.** `m05_ctx` gave the model 50 raw series-constant
   columns and was rejected for memorising series identity. There is no *single*
   history-derived scalar the model can use as an interaction term. §E.3 shows one exists.

---

## D. LESSONS FROM RT-600, T2, D3R AND WAVE 8

Six rules that must constrain every proposal below. They are the most valuable asset in
the repository and every one of them was paid for.

**D.1 — Standalone gain is not information.** T2 beat its matched single-model control by
+0.00943, 5/5 folds, bootstrap CI clear of zero, and added **+0.00024** to the
seven-specialist ensemble over an exchangeable seed clone — a 40× overstatement. *The only
binding number is the marginal against `RT600 + seed clone`.*

**D.2 — Low correlation is not a diversity credential.** `m09_back` (51 new statistical
columns) decorrelated *less* from the champion than a seed change did, and blended worse.
Any "my stream is decorrelated" argument must carry a seed-clone control.

**D.3 — Gains anti-stack.** W4-E6 (13 boosters) and W5-E10 (175 columns from three blocks)
both scored *worse* than their best component. A fixed feature-fraction budget spread over
more columns dilutes the block that works. New columns must be few and load-bearing.

**D.4 — Extraction capacity is not the bottleneck in the dominant cell.** W7-D3R Arm B —
same 500 columns, same rows, 127 leaves / ff 1.0 / 900 trees — was *negative* against Arm A
on 4/5 folds. More model on the same representation is a closed lane.

**D.5 — Coverage beats precision.** ORR's repair mechanism was real (P(repair correct |
RT-600 wrong, teacher confident) = 0.737, far above its 0.55 bar) and moved the aggregate
metric by −0.00005, because the confirmed population was too small a share of pair weight.
§E.1 measures why this generalises: **the loss is diffuse.**

**D.6 — The screen store is triage for features only.** Three screen-level *architectural*
wins reversed sign at full scale (context block, pairwise objective, DGP gating); every
screen-level *feature* addition transferred. Anything that repartitions training data must
be tested at full scale from the start.

And one that is not a rule but a boundary condition: **W7-D3R proved the information gap
is real** (+0.071 cell AUC from each series' own future, 5/5 folds) **and Wave 8 proved
that five different ways of distilling that future into a causal student all fail.** The
open question Wave 8 leaves is not "is there information" but "why does none of it survive
contact with the full row population". This document's answer, and the reason it does not
propose a sixth distillation mechanism: *the future-aware family kept trying to import the
answer instead of building the missing state variable.* Arm C's advantage is largely the
answer to a question the prefix can partly ask on its own — **has this deviation lasted?** —
and the prefix currently has no column that asks it.

---

## E. DOMINANT ERROR — SCIENTIFIC DIAGNOSIS

Four new descriptive diagnostics, run for this document on the real store, fold-pure,
τ used only post hoc (`PROTOCOL.md` §1). Pre-declared before any number existed.

### E.1 The remaining loss is DIFFUSE, not concentrated (D1)

Exact per-series decomposition of the dominant cell's inversion loss, using the official
mid-rank pair machinery decomposed row-wise (totals reconcile to W7-D0 exactly):

| population | share of cell loss | share of cell weight |
|---|---:|---:|
| never-break series | 0.370 | 0.371 |
| break series (mature positives + pre-break negatives) | 0.630 | 0.629 |

(Each pair's loss is split half to the positive and half to the negative, so never-break
series can carry at most 0.5. Their 0.370 is exactly W7-D0's 0.7399 × 0.5 — the two
decompositions agree independently.)

| loss carried by the worst … | never-break | break |
|---|---:|---:|
| 5 % of series | 0.158 | 0.202 |
| 10 % of series | 0.277 | 0.340 |
| 20 % of series | 0.466 | 0.537 |
| 50 % of series | 0.829 | 0.861 |

A perfectly concentrated failure would put ~1.0 in the 5 % row. We see 0.16. **There is no
small set of pathological series to repair.** This is the quantitative reason ORR failed
and the reason no targeted-repair mechanism should be first in the queue: to move the
metric you must move a channel that applies to *most* of the cell, weakly.

### E.2 Never-break false positives ARE null-model errors (D3) — the brief's §43, answered

Series-level, out-of-fold LightGBM predicting each series' own dominant-cell loss rate from
**23 history-only fingerprint columns** (history is fully visible at t=0, so this is legal),
five permanent folds, with a label-permutation control:

| population | OOF Spearman(pred, true loss rate) | permuted-label control | n |
|---|---:|---:|---:|
| never-break | **+0.192** | −0.031 | 3,239 |
| break | +0.116 | −0.030 | 3,137 |

Univariate directions for never-break series (all p < 1e-7):

| fingerprint | ρ with loss rate | reading |
|---|---:|---|
| `q_ratio` = (q99−q01)/(q75−q25) | **+0.143** | heavier tails → more false positives |
| kurtosis | **+0.142** | same |
| Hill tail index | **−0.128** | lower α (heavier tail) → more false positives |
| `exc_n64` = number of historical excursion episodes | **−0.129** | a history that *rarely* wanders is mis-ranked when the online segment does |
| `n_hist` | −0.095 | shorter history → noisier null |
| permutation entropy | +0.095 | less structured history |
| VR(50) | +0.082 | long memory |

**Answer to §43: yes.** The hard never-break negatives share heavy tails, long memory and
histories that themselves produce few excursions. Note the *sign* of `exc_n64`: it is not
that excursion-prone series fool the model — those are priced by the per-series null. It is
series whose null looks *calm* and then wanders.

**But note R² is negative** (−0.002 / −0.025). The signal is a weak monotone ordering, not a
level prediction. That is exactly the shape that fails as 50 raw columns (`m05_ctx`,
−0.0177 at full scale) and could work as *one scalar interaction column* — which has never
been tried. It also matches the retry that `FAILED_EXPERIMENTS.md` itself names for W5-E2:
*"conditioning the null on the series' own tail index rather than sharpening the online
statistic."*

### E.3 What "persistence" actually means, measured (D4) — the brief's §42, answered

Twelve causal excursion state variables (3 channels × 3 windows × {current run, run at a
stricter threshold, excursion mass, running-max run}), history-only nulls, whole store,
**26 seconds of compute**. Fold-0 dominant cell, against RT-600 at 0.67771:

| state variable | cell AUC | within-t ρ with RT-600 |
|---|---:|---:|
| `res64_maxrun90` — longest AR-residual-scale excursion so far | **0.60791** | +0.367 |
| `res32_maxrun90` | 0.59968 | +0.373 |
| `res128_mass90` — accumulated excursion area | 0.59170 | +0.415 |
| `res128_run90` — current run length | 0.59098 | +0.412 |
| `scale64_maxrun90` | 0.58819 | +0.268 |
| `loc128_mass90` — location channel | 0.51106 | +0.084 |

Three readings, all load-bearing:

* **Excursion duration is a real discriminator and it is not in the bank.** 0.608 standalone
  in the cell where RT-600 sits at 0.678 is a meaningful weak channel, and ρ = 0.37 within
  timestep is far below anything the project has previously measured against the blend
  (T2 was 0.674; the seed clone 0.910).
* **It is stronger exactly where the loss is.** `res64_maxrun90` scores **0.6132 against
  never-break negatives** and **0.5931 against pre-break negatives**, while RT-600 is
  flat across the two (0.6768 / 0.6803). The new channel's comparative advantage is
  concentrated in the 74 %-of-loss population.
* **The channel is the innovation scale, not the level.** The location channel is worth
  nothing (0.511) — consistent with the taxonomy's finding that location breaks do not
  exist here. Every proposal below is built on residual/scale channels for this reason.

**Honest negative from the same run:** a naive fixed-weight blend of any of these ranks
with the RT-600 rank *hurts* (−0.001 to −0.009 cell AUC). A 0.61 stream at 25 % weight
against a 0.678 blend is expected to hurt; this says nothing about the properly-harnessed
marginal, and it does say **do not ship this as a score post-transform**.

### E.4 Contiguity, and how it scales with elapsed time (D5) — the missing state variable

The same run, split by current online index, mature-break positives vs never-break negatives:

| current t | median longest run (pos) | (neg) | p90 (pos) | p90 (neg) |
|---|---:|---:|---:|---:|
| 200–300 | 13 | 3 | 94 | 56 |
| 400–500 | 30 | 16 | 127 | 66 |
| 650–800 | 51 | 27 | 226 | 81 |
| 800–1000 | 64 | 33 | **239** | **94** |

Share of series whose longest AR-residual-scale excursion ever reaches length L:

| L | never-break | break |
|---:|---:|---:|
| 25 | 0.458 | 0.592 |
| 50 | 0.278 | 0.415 |
| 100 | 0.062 | 0.145 |
| 200 | 0.030 | 0.067 |

**This is the answer to the brief's central question.** 27.8 % of never-break series carry
an excursion of ≥ 50 steps and 3.0 % carry one of ≥ 200 steps — long excursions in stable
series are common, which is why amplitude and duration alone cannot separate the classes.
What separates them is **growth**: the never-break p90 excursion length rises 56 → 94 as
t goes 200 → 1000 (roughly logarithmic), while the mature-break p90 rises 94 → 239 (roughly
linear). Under stationarity with mixing, the longest run of exceedances grows like
**log t** (Erdős–Rényi law of large numbers, with a Gumbel limit for the longest head run);
under a persistent regime change the current excursion never terminates and grows like **t**.

**The missing state variable is not "how extreme" or "how long" — it is "is the longest
excursion growing faster than log t, for this series' own dependence structure".**

### E.5 A negative that de-risks the pilot (D6)

I implemented the obvious Erdős–Rényi normalisation — divide the observed longest run by
`scale · log(t_eff)/log(1/p)`, where `scale` is fitted from the history's own longest run —
and it is **worse than the raw longest run**:

| variant | cell AUC | ρ with RT-600 |
|---|---:|---:|
| `maxrun_raw` | **0.60791** | +0.367 |
| `maxrun / ER-expectation` | 0.60028 | +0.355 |
| `maxrun − ER-expectation` (Gumbel z) | 0.58028 | +0.312 |

The single historical longest run is one order statistic and far too noisy a scale
estimate. **The correct construction is the project's own:** build the *empirical
distribution* of length-matched run maxima over the historical segment, exactly as
`NullCal` does for rolling means, and read the online run's percentile against it. That is
infrastructure (§O, IM2), and this negative is why the pilot in §L is specified that way
rather than the naive way.

---

## F. INTERDISCIPLINARY LITERATURE REVIEW

Sources, what the source actually shows, and — separated explicitly — what is our
extrapolation. Full URLs in §V.

**F.1 Probability — run and scan statistics.** The Erdős–Rényi law of large numbers gives
the longest run of ones in n observations as ~log n; Erdős–Révész refine it with iterated-
logarithm upper/lower classes, and Gordon–Schilling–Waterman give an extreme-value (Gumbel)
approximation for the longest head run. Discrete scan statistics extend this to dependent
models with sharp distributional bounds.
*Supported:* the log-order growth of the longest run and its extreme-value limit under
stationarity/mixing. *Our inference:* that the ratio of observed longest run to this
stationary growth law is a break statistic. Untested here beyond the descriptive D5/D6.

**F.2 Electrical protection — inverse-time relays.** IEEE C37.112 (1996/2018) standardises
inverse-time overcurrent characteristics as an **integral equation** so that coordination
holds "not only in the case of constant current input but for any current condition of
varying magnitude", and explicitly defines the **reset characteristic** matching
electromechanical disc behaviour, for microprocessor relay designers.
*Supported:* the operate integral, the reset integral, and the design intent — discriminate
a sustained overcurrent from a sequence of brief ones without tripping on the latter.
*Our inference:* the identical two-time-constant accumulator applied to a calibrated
break-evidence channel is a persistence-vs-transient discriminator. The standard says
nothing about structural breaks.

**F.3 Control / estimation — innovation monitoring.** Kalman-filter consistency testing
uses the **normalized innovation squared** (χ² under correct specification) and
**whiteness** of the innovation sequence; accumulating normalized innovations against an
accumulated threshold is standard for subtle-fault detection ("infinite-horizon innovations
monitor"). Modified whiteness tests are used for structural damage detection.
*Supported:* NIS/whiteness as fault detectors; the accumulation construction.
*Our inference:* that a *deliberately non-adaptive* history-fitted filter is the right one
here. This is actually supported by our own evidence, not the literature: `m04_resid`'s
GARCH(1,1) scored 0.50012 because an adaptive filter erases the break it is hunting.

**F.4 Nonlinear dynamics — DMD / Koopman.** Gottwald & Gugole show the **reconstruction
error of a dynamic mode decomposition** detects transient dynamics and regime change,
because transients explore the full state-space dimension with fast relaxation while
equilibrium evolves on a lower-dimensional attractor; online DMD variants for streaming
changepoint detection exist.
*Supported:* DMD reconstruction error and effective dimension as regime-change observables.
*Our inference:* the univariate Hankel-embedded version, fitted on history and frozen, is
causal and cheap. The papers are mostly multivariate.

**F.5 Data mining — matrix profile / FLOSS.** Gharghabi et al., *Matrix Profile VIII:
Domain Agnostic Online Semantic Segmentation at Superhuman Performance Levels* (ICDM 2017)
introduce the **arc curve**: count how many nearest-neighbour "arcs" cross each index; few
crossings ⇒ high probability of a semantic regime change. FLOSS is the streaming variant,
explicitly designed for real-time use and an order of magnitude faster.
*Supported:* arc-crossing as an online, domain-agnostic regime-change score.
*Our inference:* using the **historical segment as the reference half** of the arc count,
so the statistic becomes "does the online window still find its neighbours in history?".

**F.6 Recurrence analysis.** Marwan et al. introduced **laminarity** and **trapping time**
from vertical structures in recurrence plots, which detect chaos–chaos (laminar-phase)
transitions that diagonal-structure measures miss; trapping time estimates the average time
the system spends in a given state.
*Supported:* trapping time as a dwell-time observable and its use for transition detection.
*Our inference:* computing it causally against a recurrence threshold fixed on history.

**F.7 Vibration / signal processing — spectral kurtosis.** Antoni's spectral kurtosis and
the kurtogram find the frequency band where impulsive transients dominate. A documented
property: **kurtosis decreases as the transient repetition rate increases** — the kurtogram
cannot tell a repetitive series of transients from a single one. Robust/negentropy variants
exist for non-Gaussian backgrounds.
*Supported:* the impulsiveness measure, its band-selection use, and its blindness to
repetition rate. *Our inference:* that *this exact blindness, used as a contrast*, is the
discriminator — high band energy with **low** spectral kurtosis is a sustained regime
change; high band energy with **high** kurtosis is an outlier burst. The literature treats
the property as a limitation; we propose to use it as a feature.

**F.8 Game-theoretic statistics.** Ramdas, Grünwald, Vovk & Shafer's safe anytime-valid
inference; Vovk's conformal test martingale is powerful against changepoint alternatives.
**Weighted conformal test martingales** (2025) "continuously adapt to benign covariate
shifts without raising unnecessary alarms while detecting harmful shifts more rapidly".
Choe & Ramdas give adjusters for combining evidence across filtrations.
*Supported:* all of the above as stated. *Our inference:* that "benign shift" maps to our
heavy-tailed excursion population and "harmful shift" to a true break. `m07_bayes` already
ships plain conformal martingales; the **weighting** is the new part.

**F.9 Duration modelling.** Hidden semi-Markov models exist precisely because an HMM forces
a **geometric** (memoryless, monotonically decreasing) dwell time whose modal duration is
one step; HSMMs allow arbitrary dwell distributions, "decoupling state persistence from the
transition structure".
*Supported:* the limitation and the fix. *Our inference:* that our absorbing-state and
BOCPD hazards inherit exactly this limitation and that an explicit-duration prior changes
what the posterior can express about persistence.

**F.10 Physics / ecology — critical slowing down.** Scheffer et al. (2009) and Dakos et al.
(2012) establish rising variance and lag-1 autocorrelation as generic precursors of critical
transitions, because the potential well flattens and recovery from perturbation slows.
*Supported:* the relaxation-rate mechanism. *Explicit caveat and our inference:* the
original use is **pre**-transition warning, which under our target `y[t]=1[t≥τ]` would
*raise* pre-break negatives and hurt. Our proposed use is the opposite one — the
**post**-transition relaxation rate settling at a *new stable* value, versus spiking and
returning during a transient. That inversion is ours, not the literature's, and it is the
main reason this family is ranked below the others.

**F.11 Topology.** Perea & Harer, *Sliding Windows and Persistence*, show maximum
persistence of the sliding-window (delay) embedding quantifies periodicity, with structural
and convergence theorems and window-size dependence.
*Supported:* the construction and its convergence behaviour. *Our inference:* that a change
in maximum persistence between historical and online windows is a break statistic. Expensive
and speculative; ranked accordingly.

---

## G / H. SEVENTY-NINE NEW MECHANISMS, TAXONOMY BY INFORMATION CHANNEL

Implementation modes per the brief: **A** new streaming feature · **B** new specialist ·
**C** residual score on RT-600 · **D** historical gate · **E** null-normalisation transform ·
**F** score post-processor · **G** replacement representation.
Cost: TINY <10 min · SMALL 10–30 min · MEDIUM 30 min–2 h · LARGE 2–6 h · XL >6 h.

### FAMILY A — EXCURSION GEOMETRY AND RUN STATISTICS *(new channel: contiguity)*
| # | mechanism | mode | cost |
|---|---|---|---|
| A1 | Contiguous exceedance dwell bank: current run, running-max run, episode count, mean episode length, inter-episode gap — on AR-residual scale at 3 windows × 2 thresholds | A | SMALL |
| A2 | **Run-length null calibration**: empirical distribution of length-matched run maxima over history; emit the online run's percentile (fixes D6's failure) | A + infra | SMALL |
| A3 | Excursion mass: area above the null band accumulated since the last re-entry, and its running max | A | TINY |
| A4 | Re-entry / recovery statistics: number of returns to band, mean recovery time, empirical hazard of re-entry given current dwell | A | SMALL |
| A5 | **Contiguity ratio** `longest_run / total_exceedance_count` — the explicit scattered-vs-sustained discriminator at matched exceedance load | A | TINY |
| A6 | Multi-threshold time-over-threshold profile (q75/q90/q99 run vector) — the relay "time–current curve" analogue in evidence space | A | SMALL |
| A7 | Growth-exponent channel: regress log(running-max run) on log(t) online; slope ≈ 0 under stationarity, ≈ 1 under a persistent break | A | SMALL |

### FAMILY B — PROTECTIVE-RELAY / FAULT-PERSISTENCE LOGIC *(new channel: reset dynamics)*
| # | mechanism | mode | cost |
|---|---|---|---|
| B1 | IEEE C37.112 inverse-time operate integral on a calibrated evidence channel, with the standard **reset** integral | A/F | TINY |
| B2 | Two-stage coordination: low-pickup/long-delay + high-pickup/short-delay accumulators emitted jointly | A | TINY |
| B3 | Thermal-replica I²t accumulator with **asymmetric** charge/cool time constants (hysteresis by construction) | A/F | TINY |
| B4 | Pickup/dropout hysteresis (Schmitt trigger, dropout ratio < 1); state = time in picked-up state | A | TINY |
| B5 | **Auto-recloser count**: number of complete pickup→reset cycles. A permanent fault re-trips; a transient does not | A | TINY |
| B6 | Arc-fault analogue: fraction of recent windows whose high-band energy ratio exceeded its historical band, with reset | A | SMALL |

### FAMILY C — SYSTEM IDENTIFICATION / OBSERVER RESIDUALS *(new channel: multi-dim dynamics)*
| # | mechanism | mode | cost |
|---|---|---|---|
| C1 | **Hankel-DMD observer**: rank-r operator fitted on the historical Hankel matrix, frozen; monitor multi-step reconstruction residual **and effective rank** online | A/E | MEDIUM |
| C2 | Kalman NIS + innovation whiteness against a *non-adaptive* history-fitted local-level+AR state space; accumulate normalized innovations (infinite-horizon monitor) | A | SMALL |
| C3 | **Observer bank / MMAE**: fit K models on K disjoint historical blocks; online, track the posterior over which block explains the data. Drift of that posterior is the signal | A/B | MEDIUM |
| C4 | Unknown-input-observer analogue: residual generator decoupled from the series' dominant historical mode, so it responds only to new structure | A | MEDIUM |
| C5 | Conditional-quantile null: historical quantile-AR fits; monitor per-quantile online conditional coverage | A/E | MEDIUM |
| C6 | Reservoir / random-feature nonlinear AR fitted on history, residual monitored online (the cheap non-neural version of a sequence model) | A/E | MEDIUM |
| C7 | Block-bootstrap historical null for any statistic — repairs NullCal's implicit independent-window assumption for long-memory series | infra + A | SMALL |

### FAMILY D — PERSISTENCE / REVERSION / DURATION *(new channel: dwell time)*
| # | mechanism | mode | cost |
|---|---|---|---|
| D1 | **Explicit-duration (semi-Markov) run-length prior** in BOCPD, replacing the geometric hazard | A | MEDIUM |
| D2 | Trapping time / laminarity from a causal recurrence plot thresholded on history | A | MEDIUM |
| D3 | Reversion hazard: from history, estimate `P(return to band | deviation size s, dwell d)`; score the online state's survival | A | SMALL |
| D4 | Two-state dwell ratio over several lookbacks, with hysteresis so single points cannot flip the state | A | TINY |
| D5 | Metastability / escape rate: 1-D effective potential from the historical density; Kramers-style escape rate and online basin residence time | A | MEDIUM |

### FAMILY E — MULTISCALE / SCALE-SPACE PERSISTENCE
| # | mechanism | mode | cost |
|---|---|---|---|
| E1 | **Scale-survival count**: at how many increasing scales is the current state still outside the per-series null band? Plus the coarsest surviving scale | A | SMALL |
| E2 | Time-causal scale-space cascade (truncated exponentials); emit the scale at which evidence peaks and whether that scale is growing | A | MEDIUM |
| E3 | Wavelet-modulus-maxima line persistence across scales | A | MEDIUM |
| E4 | Renormalisation coarse-graining: block-average by factor b; emit the decay exponent of evidence in b | A | SMALL |
| E5 | MFDFA multifractal spectrum-width change | A | LARGE |

### FAMILY F — FREQUENCY / TIME-FREQUENCY
| # | mechanism | mode | cost |
|---|---|---|---|
| F1 | **Spectral kurtosis contrast**: band excess energy × (low spectral kurtosis) = sustained; × (high kurtosis) = impulsive burst | A | SMALL |
| F2 | Cyclostationarity: cyclic autocovariance / envelope-spectrum change vs the historical null | A | MEDIUM |
| F3 | **AR pole migration**: roots of the online-refit AR polynomial vs historical poles (modulus and argument drift) | A | SMALL |
| F4 | Split-window spectral coherence (first vs second half of the online window) against its historical null | A | SMALL |
| F5 | Cepstral / envelope statistics | A | SMALL |
| F6 | Negentropy of the envelope spectrum — the heavy-tail-robust alternative to kurtosis | A | SMALL |

### FAMILY G — TRAJECTORY GEOMETRY / MOTIFS / TOPOLOGY *(new channel: shape similarity)*
| # | mechanism | mode | cost |
|---|---|---|---|
| G1 | **FLOSS arc-crossing curve** with history as the reference half | A/B | MEDIUM |
| G2 | Matrix-profile distance from each online subsequence to the historical subsequence set (novelty), plus its running statistics | A | MEDIUM |
| G3 | **Nearest-neighbour provenance ratio**: fraction of the current window's k-NN that live in history vs in the recent online segment | A | MEDIUM |
| G4 | Sliding-window persistent homology max-persistence change (SW1PerS) | A | LARGE |
| G5 | Delay-embedded attractor separation: energy/Wasserstein distance between the historical and online delay point clouds | A | MEDIUM |
| G6 | RQA recurrence rate / determinism / divergence against a history-fixed threshold | A | MEDIUM |

### FAMILY H — DISTRIBUTION-FREE SEQUENTIAL EVIDENCE
| # | mechanism | mode | cost |
|---|---|---|---|
| H1 | **Weighted conformal test martingale** — betting function weighted to be insensitive to benign heavy-tail excursions | A/B | MEDIUM |
| H2 | E-value calculus aggregation (averaging / adjusters) as a **parameter-free** eighth specialist, instead of LightGBM combination | B | SMALL |
| H3 | Sequential kernel MMD vs the historical sample, scanned over window lengths | A | MEDIUM |
| H4 | Conformal prediction-interval coverage stream: coverage collapse and width inflation as two channels | A | SMALL |
| H5 | Rank CvM / Anderson–Darling with an explicit run-length component | A | SMALL |

### FAMILY I — LARGE DEVIATIONS / RARE EVENTS
| # | mechanism | mode | cost |
|---|---|---|---|
| I1 | **Joint (size × duration) rarity**: `−log P̂(deviation ≥ s AND duration ≥ d)` estimated by empirical scan over the historical segment | A | SMALL |
| I2 | Scan statistic over (start, length) with the exact historical null of the maximum (multiple testing priced per series) | A | MEDIUM |
| I3 | Cramér rate function of the block mean estimated from history; score = rate × block length | A | SMALL |
| I4 | Record statistics: number of new historical-scale records set online vs the log-growth expected under stationarity | A | TINY |

### FAMILY J — HISTORICAL-DGP CONDITIONING (gate/interaction, never raw columns)
| # | mechanism | mode | cost |
|---|---|---|---|
| J1 | **Scalar difficulty gate**: one OOF-trained history-only scalar predicting RT-600's own cell loss rate, used as a single interaction column | D/A | SMALL |
| J2 | **Tail-index-conditioned null**: GPD/Hill tail fit of history replaces empirical quantiles in the exceedance thresholds | E | SMALL |
| J3 | Effective-sample-size correction: divide every online z by √ESS from the series' own historical ACF | E | SMALL |
| J4 | **Specialist-competence gate** — measure first whether which of the seven specialists wins varies by fingerprint; gate only if it does | D | TINY |
| J5 | Per-series conformal p-value transform before ranking — a legal approximation of the illegal within-t rank | F | SMALL |

### FAMILY K — ENSEMBLE ERROR REPAIR / COMPLEMENTARITY
| # | mechanism | mode | cost |
|---|---|---|---|
| K1 | Error-manifold clustering on the D1 per-series loss cube × fingerprint — are there ≥2 distinct failure manifolds? | diagnostic | SMALL |
| K2 | **Density-ratio repair** over the *whole* cell (no confidence threshold) — the direct answer to ORR's coverage failure | C | MEDIUM |
| K3 | Negative-only specialist trained to rank negatives by break-likeness, applied subtractively | B/C | MEDIUM |
| K4 | Specialist within-t rank *spread* (disagreement) as an input channel | A | TINY |
| K5 | Two-population calibration: separate SCDF anchors for heavy- vs light-tailed series | F | SMALL |

### FAMILY L — INFORMATION-THEORETIC
| # | mechanism | mode | cost |
|---|---|---|---|
| L1 | Normalised compression distance between the online window and history (assumption-free universal divergence) | A | MEDIUM |
| L2 | Active information storage of the online window vs the historical AIS | A | MEDIUM |
| L3 | **Ordinal transition matrix divergence** — `m03_dyn` has permutation *entropy* but not the ordinal *transition* structure | A | SMALL |
| L4 | **Time irreversibility** (Ramsey–Rothman / visibility-graph asymmetry) — a dynamics change invisible to the marginal | A | SMALL |

### FAMILY M — SCORE-STATE TRANSFORMS ON RT-600 (MODE F)
| # | mechanism | mode | cost |
|---|---|---|---|
| M1 | Leaky integrator with asymmetric charge/discharge on the calibrated score | F | TINY |
| M2 | Time-above-baseline and area-above-baseline of the calibrated score (new *state*, unlike the already-tested level smoothing) | F | TINY |
| M3 | Monotone recalibration in (score, log t, time-above-threshold) | F | SMALL |
| M4 | Score velocity / acceleration in log-time | F | TINY |
| M5 | Score percentile against the series' *own* historical-null-implied score distribution | F | SMALL |

> **Standing caveat on Family M.** Post-transform running-max on the score costs −0.0060 and
> EWMA/decayed-max are worth +0.0005 (inside noise) — already measured. Family M is only
> interesting where it introduces genuinely *new state* (M2, M3), not where it smooths the level.

### FAMILY N — PHYSICS-INSPIRED STATE VARIABLES
| # | mechanism | mode | cost |
|---|---|---|---|
| N1 | Order parameter from the historical effective potential; online basin occupancy | A | MEDIUM |
| N2 | Post-transition relaxation rate (AR(1) recovery) settling at a new level vs spiking and returning | A | SMALL |
| N3 | Cross-scale variance transfer (intermittency analogue): does the online segment *redistribute* variance across scales or merely rescale it? | A | SMALL |
| N4 | First-passage-time statistics to historical quantile levels | A | SMALL |

### FAMILY O — RADICAL REDESIGNS (not to be implemented now)
| # | mechanism | precursor that would justify the cost |
|---|---|---|
| O1 | Hierarchical Bayesian per-series generative null with cross-series shrinkage, posterior-predictive p-value stream | J2 or C7 showing that null misspecification is the binding constraint |
| O2 | Replace the 500-column bank with per-series residual streams + a global model on residual-space features only (MODE E at full scale) | Pilot 3 (C1/C2) showing residual-space features beat their raw-space twins |
| O3 | Meta-learned detector (hypernetwork on the historical fingerprint) | J4 showing specialist competence varies materially by fingerprint |
| O4 | Causal sequence model on the *PIT/residual* stream with a within-t ranking loss — D3R's Case 2 attacked on a normalised representation rather than raw channels | G1/G3 showing trajectory shape carries information the 500-vector lacks |
| O5 | Learned within-t calibration that approximates the illegal cross-sectional rank from training data alone | K5 showing per-population calibration moves the metric |

---

## I. PRIOR-EXPERIMENT COLLISION CHECK

This is the section that matters most, because roughly half of the driving brief's §18–§26
suggestions are already in production. For every family, the closest prior experiment and
why the proposal is not it.

| proposal | closest prior work | why it is genuinely different |
|---|---|---|
| A1–A7 (run/dwell/contiguity) | `m01_seq` `_per` channel = **cumulative fraction** above q99; `m10_persist` exceedance run length | `m10_persist` was **rejected** (−0.00041 vs a seed clone) and computed run length only as one of 15 outlier-vs-bulk columns aimed at *amplitude concentration*, never as a **running-max dwell with a matched-length run null** and never as the growth exponent (A7). `_per` is a fraction and cannot distinguish scattered from contiguous. D4/D5 measure the gap directly. |
| A2 (run-length null) | `NullCal` (rolling-mean nulls), `m01_seq._Chan` (detector-path nulls, incl. dyadic running-peak nulls) | `_Chan` nulls the running **peak level**, never the running **duration**. No path-length functional has a null anywhere. |
| B1–B6 (relay logic) | `m01_seq` decayed peak (fixed half-life), `apply_persistence` (runmax / decaymax / EWMA on the final score) | Every incumbent accumulator is symmetric or monotone. None has an asymmetric charge/discharge pair, a dropout threshold below pickup, or a counted reset event. `apply_persistence` operates on the *final score* and was measured (runmax −0.0060); B applies to *evidence channels* with new state. |
| C1 (Hankel-DMD) | `m04_resid` AR(1/2/5), ridge, Huber; `m12_rdep` AR-coefficient LR | All are scalar one-step predictors. None forms a delay-embedded operator, and **effective rank has no analogue anywhere in the bank**. |
| C2 (Kalman NIS) | `m04_resid` GLR channel `mean(e²) − log var(e) − 1` | The GLR is a static Gaussian LR on a fixed whitening. NIS is normalised by the filter's *own* predicted covariance, and whiteness (Ljung–Box on innovations) is a separate functional not computed anywhere. |
| C3 (observer bank / MMAE) | `m05_ctx` (rejected as features), DGP-cluster gating (rejected, −0.0219) | Both prior attempts partitioned **series across the dataset**. C3 partitions **history within one series** and never splits the training data — D.6's failure mode does not apply. |
| D1 (semi-Markov BOCPD) | `m07_bayes` BOCPD + absorbing posterior | Both impose a geometric hazard. Explicit-duration priors change what the run-length posterior can express, which is the exact quantity §E.4 says is missing. |
| D2/G6 (RQA) | none | No recurrence structure exists in the repository. |
| E1–E4 (multiscale) | `m00_core` six trailing windows; `m03_dyn` Haar energies | The bank emits the statistic at each scale and lets LightGBM combine them. It never emits the **count of surviving scales** or the **decay exponent across scales** — a difference-of-features the boosters approximate badly (the same argument that motivated `m10_persist`'s `*_gap` column). |
| F1/F6 (spectral kurtosis / negentropy) | `m03_dyn` band powers, spectral entropy, slope | Impulsiveness is not measured anywhere. The kurtogram's documented blindness to repetition rate is the discriminator. |
| F3 (AR pole migration) | `m12_rdep` AR-coefficient likelihood ratio | The LR is a scalar test of coefficient change; pole modulus/argument separates *persistence* change from *frequency* change, which the LR cannot. |
| G1–G5 (trajectory geometry) | `m06_loc`, `m11_focus` (both localise via segment statistics) | Localisation ≠ similarity. Nothing compares shapes; §C.4. This is also the only family that directly tests the brief's §44 (is temporal order lost?). |
| H1 (weighted CTM) | `m07_bayes` e-processes incl. Vovk power martingale on conformal p-values | The incumbent bets against a **fixed** null. Weighting to be insensitive to benign shift while sensitive to harmful shift is the 2025 extension and is exactly aimed at §E.2's heavy-tail population. |
| H2 (e-value aggregation) | 7-specialist equal-weight SCDF mean | A parameter-free anytime-valid combination is a different *combination rule*, and the project has established that every *learned* combination loses to the equal mean. |
| I1–I4 (large deviations) | `m00_core` `surprise()` (two-sided −log10 p of a rolling mean), `m11_focus` max-over-τ | Every incumbent surprise prices magnitude at fixed length. None prices the **joint** (magnitude, duration) event, which is the object §E.4 identifies. |
| J1 (scalar gate) | `m05_ctx` 50 raw columns (−0.0177 full scale, permutation control −0.057) | The failure was **encoding**, not substrate: 50 near-continuous series-constant columns let LightGBM memorise series identity. One OOF-trained scalar with a derangement control cannot. §E.2 shows the substrate exists for *loss rate* even though it does not exist for *has_break* (which is what `m05_ctx` H1 tested and correctly rejected). |
| J2 (tail-conditioned null) | `m10_persist` (W5-E2, rejected) | `FAILED_EXPERIMENTS.md` names this exact retry as the untested alternative: "conditioning the null on the series' own tail index rather than sharpening the online statistic." |
| J4 (specialist competence) | DGP-cluster gating (rejected) | The rejected experiment **trained** specialists per cluster (data-starving them). J4 measures whether the *existing, fully-trained* seven have different competence profiles, and would reweight, not retrain. |
| K2 (density-ratio repair) | ORR (Wave 8, KILL) | ORR mined **teacher-confirmed** errors (877 pairs, 0.737 recoverability, flat aggregate). K2 deliberately abandons confidence thresholding for full-cell coverage — the direct response to D.5, and §E.1 is the measurement that says this is the right direction. |
| M1–M5 | `apply_persistence` (measured: runmax −0.0060, decaymax −0.0033/+0.0005, EWMA +0.0005) | Only M2/M3 introduce new state; M1/M4 are close enough to the measured family that they are ranked LOW. |
| N2 (relaxation rate) | `m03_dyn` ACF channels | ACF at a window is a level; the **rate of return to baseline after a perturbation** is a conditional functional not computed anywhere. |
| O4 | W6-N2 causal TCN (0.5432, failed both bars); CFEP (Wave 8, −0.00374 vs its own BCE control) | Both neural attempts read raw or lightly-processed channels. O4 reads the **null-normalised** stream. This is a real distinction but a weak one — hence "radical redesign, not now". |

---

## J. TOP 25 (plausible)

Ranked by expected research value = novelty vs *our repo* × independence from RT-600 ×
dominant-cell relevance × (1 / cost) × literature support. Per-mechanism 1–5 scores on all
ten of the brief's §34 axes are in `research/new_avenues_2026.csv`.

**Rated VERY HIGH (5):** A1, A2, A5, A7, I1.
**Rated HIGH (20):** A3, A4, A6, B1, B3, B4, B5, C1, C2, C3, C5, D1, D2, D3, D4, E1, E4,
F1, F3, F6, G1, G2, G3, G5, G6, H1, I3, I4, J1, J2, J4, K1, L3, L4, N3, N4 — of which the
25 that survive the cost-and-collision filter are carried into §K, and the rest are queued
behind them in the CSV's `priority` column.
**Explicitly rated LOW and not queued:** E5, F5, H3 (collapses into G5), I2 (m11_focus
already does most of it), J5, M1, M4, O5.

## K. TOP 15 (high value)

| # | mechanism | family | origin field | mode | non-ML? | infra-only? |
|---|---|---|---|---|---|---|
| 1 | A1+A2+A5+A7 excursion dwell bank with run-length null | A | probability / run statistics | A | ✅ | ✅ |
| 2 | B1+B3+B5 relay inverse-time, thermal replica, recloser count | B | electrical protection | A/F | ✅ | ✅ |
| 3 | C1 Hankel-DMD residual + effective rank | C | nonlinear dynamics / fluid mechanics | A/E | ✅ | ✅ |
| 4 | C2 Kalman NIS + innovation whiteness (non-adaptive) | C | control / estimation | A | ✅ | ✅ |
| 5 | G1+G3 FLOSS arc-crossing + NN provenance | G | data mining / time-series motifs | A/B | — | ✅ |
| 6 | I1 joint (size × duration) rarity | I | large deviations / scan statistics | A | ✅ | ✅ |
| 7 | E1+E4 scale-survival count and coarse-graining exponent | E | scale-space / renormalisation | A | ✅ | ✅ |
| 8 | F1+F6 spectral kurtosis / negentropy impulsiveness contrast | F | vibration diagnostics | A | ✅ | ✅ |
| 9 | J1+J4 scalar difficulty gate + specialist competence | J | meta-learning / our own diagnostics | D/A | — | ✅ |
| 10 | H1 weighted conformal test martingale | H | game-theoretic statistics | A/B | ✅ | ✅ |
| 11 | L3+L4 ordinal transition divergence + time irreversibility | L | information theory / nonlinear dynamics | A | ✅ | ✅ |
| 12 | D1 explicit-duration BOCPD | D | duration modelling | A | ✅ | — |
| 13 | K2 density-ratio repair over the whole cell | K | ML (our own ORR post-mortem) | C | — | ✅ |
| 14 | J2 tail-index-conditioned null | J | extreme value theory | E | ✅ | ✅ |
| 15 | D3+D4 reversion hazard and hysteretic dwell ratio | D | reliability / survival | A | ✅ | ✅ |

**Brief §49 satisfied:** 11 of 15 originate outside mainstream ML (probability, electrical
protection, control, fluid mechanics/dynamics, large deviations, scale-space, vibration
diagnostics, game-theoretic statistics, information theory, duration modelling, EVT).
**Brief §50 satisfied:** 14 of 15 plug into RT-600 as features, gates, post-processors or
one extra specialist. Only D1 needs meaningful new modelling code, and even that is an
edit to an existing recursion.
**Brief §51 satisfied:** three radical redesigns specified in Family O with explicit
precursor gates (O1←J2/C7, O2←Pilot 3, O3←J4, O4←G1/G3).

---

## L. TOP 10 CHEAP / MEDIUM PILOTS — FULL SPECIFICATIONS

Shared protocol for all ten (this **is** the standard cheap pilot the brief asks for):

* **Fold 0 only**, fixed in advance. The canonical 5 dev folds, `folds.parquet`, never regenerated.
* **Causality gate first:** `novel_streams.harness.verify()` — bitwise prefix invariance,
  `atol=0.0`, 8 series spanning the length range, cuts (3, 10, 37, 111). A mechanism that
  fails does not get scored.
* **No hyperparameter search. At most two theoretically-justified variants.**
* **Diagnostic pack** (`harness.diagnostic_pack`, seconds, no training): whole-fold TS-AUC,
  dominant-cell AUC, mature-vs-never-break, mature-vs-pre-break, six t-buckets, five
  age-buckets, within-t rank correlation with the RT-600 blend.
* **Pair flow** (`harness.pair_flow_by_cell`): repairs / damage / net, split by dominant
  cell and negative type.
* **Binding number**: `harness.marginal` = `wave8_common.ensemble_marginal` —
  `RT600` vs `RT600 + RT-401 seed clone` vs `RT600 + candidate`, cross-fitted SCDF, fold 0.
  **The reported result is `cand − clone`, never `cand − base`.**
* **Kill gate**: `cand − clone < +0.0010` → KILL and write it up.
  `+0.0010…+0.0020` WEAK (continue only if the mechanism cost is TINY/SMALL).
  `+0.0020…+0.0030` INTERESTING → 5 folds. `+0.0030…+0.0050` SERIOUS → 5 folds + bootstrap
  + alternate partitions. `> +0.0050` MAJOR. `> +0.0080` BREAKTHROUGH.
* **Continuation gate** for a MODE-A feature block: it must also beat the `m12_rdep`
  precedent — survive as an *architecture* addition, not only as an eighth stream.


### L.0 — TOP-10 SUMMARY TABLE

| rank | mechanism | information source | field | mode | closest prior experiment | why different | dominant-cell relevance | RT600 redundancy risk | pilot cost | literature confidence | priority |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Excursion dwell bank + run-length null (A1/A2/A5/A7) | contiguity and growth rate of deviations | probability, run/scan statistics | A | `m10_persist` (W5-E2, REJECTED); `m01_seq::_per` | `_per` is a *fraction*, never a contiguous run; no run-length null exists anywhere; A7's growth exponent is new | **VERY HIGH** — measured 0.608 cell AUC, 0.613 vs never-break negatives | **LOW** (ρ=0.367, lowest ever measured here) | SMALL | HIGH (Erdős–Rényi, Gordon–Schilling–Waterman, scan statistics) | **1** |
| 2 | Relay inverse-time + thermal replica + recloser count (B1/B3/B5) | reset dynamics and asymmetric time constants | electrical protection | A/F | `apply_persistence` runmax/decaymax/EWMA (measured) | every incumbent accumulator is symmetric or monotone; none has dropout<pickup or a counted reset | HIGH — the F arm acts directly on the cell's score | MODERATE (F arm partly overlaps measured smoothing) | TINY | HIGH (IEEE C37.112 operate + reset integrals) | **2** |
| 3 | Hankel-DMD residual + effective rank; Kalman NIS (C1/C2) | multi-dimensional dynamics, attractor dimensionality | nonlinear dynamics, control | A/E | `m04_resid` (scalar AR/GARCH); `m12_rdep` AR-coef LR | all incumbents are scalar one-step predictors; effective rank has no analogue | HIGH — targets never-break null misspecification | LOW–MODERATE | MEDIUM | HIGH (Gottwald & Gugole; Bar-Shalom NIS) | **3** |
| 4 | FLOSS arc-crossing + NN provenance (G1/G3) | trajectory shape similarity to history | data mining / time-series motifs | A/B | none — no similarity statistic exists | the bank discards temporal order beyond lag-2 products | HIGH; also the cheapest test of brief §44 | **LOW** | MEDIUM | HIGH (Matrix Profile VIII, ICDM 2017) | 4 |
| 5 | Scale-survival count + coarse-graining exponent (E1/E4) | cross-scale survival of evidence | scale-space / renormalisation | A | `m00_core` per-scale windows | bank emits each scale, never the count of surviving scales or the decay exponent | MODERATE–HIGH | MODERATE | SMALL | MODERATE (our own m00_core rationale; scale-space theory) | 5 |
| 6 | Joint (size × duration) rarity (I1) | joint rarity of magnitude AND persistence | large deviations / scan statistics | A | `m00_core::surprise` (magnitude at fixed length) | nothing prices the joint event; this is the formal form of §E.4 | **VERY HIGH** | MODERATE (overlaps Pilot 1) | SMALL | HIGH | 6 |
| 7 | Spectral impulsiveness contrast (F1/F6) | impulsive vs repetitive energy | vibration diagnostics | A | `m03_dyn` band powers / spectral entropy | impulsiveness is not measured; kurtogram's blindness to repetition rate is the discriminator | MODERATE | MODERATE | SMALL | HIGH (Antoni; robust negentropy variants) | 7 |
| 8 | Ordinal transition divergence + time irreversibility (L3/L4) | higher-order ordinal dependence, time asymmetry | information theory / nonlinear dynamics | A | `m03_dyn` permutation *entropy* | entropy is a scalar summary; the transition matrix and reversibility are different functionals | MODERATE | MODERATE | SMALL | MODERATE–HIGH (Bandt–Pompe; Ramsey–Rothman) | 8 |
| 9 | Scalar difficulty gate + specialist competence (J1/J4) | historical-DGP conditioning | meta-learning / our own diagnostics | D/A | `m05_ctx` (−0.0177); DGP gating (−0.0219) | the failure was *encoding* (50 series-constant columns) and *retraining*; one OOF scalar + reweighting is neither | MODERATE — measured ρ=+0.192 OOF for loss rate | **UNKNOWN** — worth measuring for that reason | TINY–SMALL | MODERATE (our own D3 measurement) | 9 |
| 10 | Weighted conformal test martingale (H1) | benign-shift-robust sequential evidence | game-theoretic statistics | A/B | `m07_bayes` e-processes incl. Vovk power martingale | incumbent bets against a *fixed* null; the weighting is the 2025 extension, aimed at heavy-tail negatives | HIGH for the never-break population specifically | MODERATE–HIGH | MEDIUM | HIGH (Ramdas et al.; WCTM 2025) | 10 |

### PILOT 1 — Excursion dwell bank with a run-length null *(A1+A2+A5+A7)*
* **Hypothesis.** After a persistent break the longest contiguous excursion of the
  AR-residual scale outside the per-series historical null band grows linearly in elapsed
  time; under stationarity it grows like log t (Erdős–Rényi). The *growth-normalised* dwell
  is therefore a break statistic that is absent from all 500 columns.
* **Sources.** Erdős–Rényi law; Gordon–Schilling–Waterman extreme-value approximation for
  the longest head run; discrete scan statistics for dependent models. §V.
* **Mode.** A (new streaming feature module, `m20_dwell`, ≤ 24 columns — D.3 forbids more).
* **Historical fit.** AR(2) coefficients + residual σ (already in `HistParams`); rolling
  statistic at w ∈ {32, 64, 128} on `e²`; band = median ± empirical q90/q99 of `|rolling − median|`
  over history; **and the empirical distribution of length-matched run maxima over the
  historical segment** (the D6 fix — this is the new calibration object).
* **Online state.** current run length; running-max run; episode count; inter-episode gap;
  excursion mass since last re-entry; running-max mass.
* **Emitted columns.** For each (channel, w): run percentile against the matched-length run
  null; running-max-run percentile; contiguity ratio `maxrun / total_exceedances`; growth
  exponent `d log(maxrun) / d log t` estimated online.
* **Downstream.** LightGBM: the seven-module 500-column bank + `m20_dwell` at the ABL
  protocol (400k rows, seed 0), and the matched control (`RT-301`) reused from wave 3.
* **Control.** The seed clone `RT-303`, already on disk. Plus the `m10_persist` arm
  (`RT-740`) as a *second* control, because it is the closest prior attempt.
* **Metrics.** Full pack + `cand − clone`.
* **Leakage tests.** Prefix invariance; plus a specific one: run length must be identical
  when the online segment is truncated mid-excursion.
* **Kill.** `cand − clone < +0.0010`, or the age profile shows the gain concentrated below
  age 50 (which would mean it is smoothing, the `m09_back` / `m10_persist` failure shape).
* **Cost.** SMALL. Stream build measured at **26 s for the whole 10,000-series store**;
  one ABL fold ≈ 8 min.
* **Prior evidence in hand.** Fold-0 dominant-cell AUC 0.608 standalone, ρ = 0.367 with the
  blend, 0.613 vs never-break negatives.

### PILOT 2 — Relay persistence logic *(B1+B3+B5)*
* **Hypothesis.** A two-time-constant accumulator with an explicit reset — the mechanism
  power systems use to avoid tripping on transient faults — separates persistent from
  transient evidence better than the symmetric decays already in `m01_seq`.
* **Sources.** IEEE C37.112-2018 operate and reset integral equations. §V.
* **Mode.** A on evidence channels **and** F on the calibrated RT-600 score (two arms; the
  F arm is nearly free and doubles as the answer to brief §31).
* **Historical fit.** Pickup = historical q95 of the channel; dropout = 0.9 × pickup;
  operate constant from the historical median crossing rate. Nothing tuned on labels.
* **Online state.** operate integral `∫dt/T(M)`, reset integral, thermal replica with
  charge τ_c and cool τ_h = 4τ_c, picked-up flag, pickup→reset cycle count.
* **Columns.** ≤ 12 (3 channels × 4 states).
* **Control.** Seed clone; plus `apply_persistence(mode="decaymax:0.99")` on the same score,
  which is already measured at −0.0033, as the "is this just smoothing?" control.
* **Kill.** `cand − clone < +0.0010`; or the F-arm result is inside ±0.0005 of the
  already-measured EWMA result (+0.0005), which would mean no new state was added.
* **Cost.** TINY (the F arm is pure post-processing on existing OOF vectors: minutes).

### PILOT 3 — Individualised observer residuals: Hankel-DMD and Kalman NIS *(C1+C2)*
* **Hypothesis.** A *frozen*, history-fitted multi-dimensional predictor produces an
  innovation stream whose (a) normalized-innovation-squared accumulation and (b) effective
  rank / reconstruction residual respond to a persistent regime change and *not* to a
  transient, because a transient explores the same attractor while a break moves it.
* **Sources.** Kalman consistency (NIS, whiteness, accumulated-innovation monitors);
  Gottwald & Gugole, *Detecting regime transitions in time series using DMD*. §V.
* **Mode.** A (and E if arm (b) wins — see O2).
* **Historical fit.** Arm (a): local-level + AR(2) state space, parameters by a fixed
  deterministic grid on history, **never re-estimated online**. Arm (b): Hankel matrix of
  history with delay d = 16, rank r = 4 via SVD; companion operator frozen.
* **Online state.** Arm (a): innovation, NIS, cumulative NIS vs its χ² expectation,
  Ljung–Box on the last L innovations. Arm (b): h-step reconstruction residual (h ∈ {1, 5}),
  running effective rank of the online Hankel window, subspace angle to the historical
  dominant subspace.
* **Why not adaptive.** `m04_resid`'s GARCH result (0.50012, exactly chance) is the
  in-repository proof that a filter adapting on the break's timescale erases it. This is a
  hard design constraint, stated in the module docstring.
* **Control.** Seed clone; plus an `m04_resid`-only ablation to show the new columns are not
  a re-statement of the existing AR residual monitors.
* **Kill.** `cand − clone < +0.0010`, or within-t ρ with the blend > 0.85 (redundancy signature).
* **Cost.** MEDIUM (~45 min build, ~10 min score).

### PILOT 4 — Trajectory geometry: FLOSS arc-crossing and NN provenance *(G1+G3)*
* **Hypothesis.** Nothing in 500 columns compares the online window to the history *as a
  shape*. If the online window's nearest neighbours migrate from history to the recent
  online segment, the generating shape changed — and that is a different functional from
  every moment/occupancy statistic in the bank. This is also the cheapest direct test of
  brief §44.
* **Sources.** Gharghabi et al., Matrix Profile VIII (FLUSS/FLOSS, arc curve); Yeh et al.,
  Matrix Profile I. §V.
* **Mode.** A first; B if it survives (it is naturally a whole-stream score).
* **Historical fit.** z-normalised subsequence index over the historical segment at
  m ∈ {16, 64}; a random projection or a sampled reference set (≤ 2,000 subsequences) to
  keep it O(n).
* **Online state.** For each online subsequence: distance to nearest historical neighbour;
  distance to nearest *recent-online* neighbour; provenance ratio; the FLOSS arc-crossing
  count at the history/online boundary, normalised by its history-only null.
* **Control.** Seed clone; plus a **shuffled-history control** — recompute with the
  historical subsequence order permuted. Distances are order-invariant, arc-crossings are
  not, so this isolates whether the gain is shape or ordering.
* **Kill.** `cand − clone < +0.0010`; or the shuffled-history control matches the real one
  (which would mean the channel is a disguised marginal statistic).
* **Cost.** MEDIUM (~1.5 h build; the dominant cost is the k-NN, bounded by the reference set).

### PILOT 5 — Scale-survival count *(E1+E4)*
* **Hypothesis.** A persistent break remains visible under temporal coarse-graining; a
  transient does not. The *count of surviving scales* and the *decay exponent of evidence in
  block size* are differences-of-features that boosters approximate badly and that the bank
  never emits explicitly.
* **Sources.** Scale-space / coarse-graining; supported directly by our own m00_core design
  rationale (which emits per-scale evidence but never the cross-scale summary).
* **Mode.** A. ≤ 8 columns.
* **Online state.** For b ∈ {1, 2, 4, 8, 16, 32}: block-average the online residual scale
  stream, score against the length-matched historical null at the *same* coarse-graining;
  emit `#{b : surprise_b > q}` for two q, the largest surviving b, and the OLS slope of
  surprise in log b.
* **Control.** Seed clone; plus an arm that adds the six per-scale surprises *individually*
  (which the bank effectively already has) to prove the summary is doing the work.
* **Kill.** `cand − clone < +0.0010`, or the individual-scale arm matches the summary arm.
* **Cost.** SMALL (~20 min).

### PILOT 6 — Spectral impulsiveness contrast *(F1+F6)*
* **Hypothesis.** Excess band energy accompanied by *low* spectral kurtosis is a sustained
  regime change; the same energy with *high* kurtosis is an outlier burst. The kurtogram's
  documented inability to distinguish repetition rates is precisely the contrast we want.
* **Sources.** Antoni, spectral kurtosis / kurtogram; negentropy and robust cyclostationarity
  for non-Gaussian backgrounds. §V.
* **Mode.** A. ≤ 10 columns.
* **Online state.** Four causal band-pass channels (from the existing Goertzel machinery in
  `m03_dyn`); per band: energy ratio vs historical null, spectral kurtosis of the band
  envelope, envelope-spectrum negentropy, and the **product contrast** `energy_z × (−SK_z)`.
* **Control.** Seed clone; plus an `m03_dyn`-only ablation (band powers without the
  impulsiveness channels).
* **Kill.** `cand − clone < +0.0010`, or the contrast column carries less gain than the
  plain energy columns (which would mean impulsiveness added nothing).
* **Cost.** SMALL (~25 min).

### PILOT 7 — Ordinal transition structure and time irreversibility *(L3+L4)*
* **Hypothesis.** A dependence break can leave the marginal law and even the permutation
  *entropy* unchanged while changing the ordinal *transition* matrix and the time-reversal
  asymmetry. `m03_dyn` has the entropy; nothing has the transition structure.
* **Sources.** Bandt–Pompe permutation entropy; time-irreversibility statistics
  (Ramsey–Rothman; visibility-graph asymmetry). §V.
* **Mode.** A. ≤ 10 columns.
* **Online state.** Order-3 ordinal pattern stream; running transition-count matrix;
  KL divergence of the online transition matrix from the historical one (null-calibrated at
  matched count); plus the Ramsey–Rothman asymmetry statistic and its historical null.
* **Control.** Seed clone; plus an arm with permutation entropy alone (already in the bank).
* **Kill.** `cand − clone < +0.0010`.
* **Cost.** SMALL (~20 min).

### PILOT 8 — Weighted conformal test martingale *(H1)*, with an e-value aggregation arm *(H2)*
* **Hypothesis.** The incumbent test martingales bet against a fixed null and are therefore
  driven up by benign heavy-tail excursions — exactly the population §E.2 shows dominates
  RT-600's false positives. A weighting that adapts to benign shift while remaining
  sensitive to persistent shift should improve the *negative* side specifically.
* **Sources.** Ramdas–Grünwald–Vovk–Shafer (safe anytime-valid inference); Vovk conformal
  test martingale; WCTM (2025); Choe & Ramdas adjusters. §V.
* **Mode.** A for the martingale channels; **B** for the H2 arm (a parameter-free combined
  e-process as an eighth specialist).
* **Online state.** Conformal p-values against the historical conformity scores; a
  predictable betting fraction weighted by a benign-shift estimate (the series' own recent
  tail-rate relative to its historical tail rate); log-capital and its running max.
* **Control.** Seed clone; plus the **unweighted** conformal martingale, which `m07_bayes`
  already ships — this is the matched control that isolates the weighting.
* **Kill.** `cand − clone < +0.0010`, or the weighted arm does not beat the unweighted arm
  in the never-break-negative split (which is the only place it is predicted to help).
* **Cost.** MEDIUM (~1 h).

### PILOT 9 — Historical-DGP conditioning: scalar gate and specialist competence *(J1+J4)*
* **Hypothesis (two parts, both measurable before any training).**
  (i) A *single* history-derived difficulty scalar carries usable interaction information
  even though 50 raw context columns do not. (ii) Which of the seven specialists wins varies
  by historical fingerprint, in which case a reweighting (not a retraining) is available.
* **Sources.** Our own D3 result (§E.2) plus the `m05_ctx` and DGP-gating post-mortems.
* **Mode.** D (gate) + A (one column).
* **Design.** Part (ii) is a **pure diagnostic and runs first**: for each fingerprint
  quintile, compute each of the seven specialists' cell AUC, and test whether the argmax
  varies more than a permuted-fingerprint control allows. If it does not, part (i) is the
  only arm.
  Part (i): nested, fold-pure OOF LightGBM on the 23 fingerprint columns predicting the
  series' own cell loss rate (never `has_break` — that substrate is confirmed absent), then
  the resulting scalar appended as one column.
* **Control.** Seed clone; **plus the within-fold derangement control** that caught
  `m05_ctx` — the scalar deranged among training series must score materially worse.
* **Kill.** Derangement control not clearly worse ⇒ memorisation ⇒ KILL immediately.
  Otherwise `cand − clone < +0.0010`.
* **Cost.** TINY for the diagnostic (minutes — the fingerprint bank builds in 7 s and is
  already on disk), SMALL for the gate arm.

### PILOT 10 — Joint size×duration rarity *(I1)*
* **Hypothesis.** Every surprise channel in the bank prices magnitude at fixed length or
  length at fixed magnitude. The metric-relevant event is the *joint* one:
  `P(|deviation| ≥ s AND duration ≥ d)` under the series' own null. That is the formal
  statement of §E.4's finding.
* **Sources.** Scan statistics; Erdős–Rényi/Erdős–Révész; Cramér large deviations. §V.
* **Mode.** A. ≤ 8 columns.
* **Historical fit.** Empirical 2-D table over the historical segment: for a grid of
  (threshold s, duration d), the fraction of history containing an excursion at least that
  extreme and that long. Floored at 1/(2·n_windows), same convention as `NullCal.surprise`.
* **Online state.** Current (s, d) of the live excursion and of the running-max excursion;
  emit `−log10 P̂` for both.
* **Control.** Seed clone; **plus Pilot 1's dwell columns** — I1 must beat A1 to justify
  the extra machinery, otherwise it is a re-parameterisation.
* **Kill.** `cand − clone < +0.0010`, or `I1 ≤ A1` on the dominant cell.
* **Cost.** SMALL (~25 min). Reuses Pilot 1's cached streams entirely.

---

## M. TOP 5 LARGE FOLLOW-UPS

| # | follow-up | trigger | cost |
|---|---|---|---|
| 1 | **Union block**, tested correctly: the two or three winning pilots combined, with `feature_fraction` adjusted to hold the effective per-tree column count constant (the fix W5-E10's post-mortem specifies) | ≥ 2 pilots reach INTERESTING | LARGE |
| 2 | **MODE E at full scale (O2)**: replace raw-space monitoring with observer-residual-space monitoring and rebuild the specialist set on it | Pilot 3 arm (b) reaches SERIOUS | XL |
| 3 | **Dwell-aware specialist set**: rebuild all seven specialist configurations with the winning persistence block, per the `m12_rdep`/W5-E11 precedent (architecture, not eighth member) | Pilot 1 or 2 reaches SERIOUS | LARGE |
| 4 | **Density-ratio repair (K2)** at full coverage, 5 folds, bootstrapped | Pilot-stage K1 finds ≥ 2 distinct failure manifolds | LARGE |
| 5 | **Causal sequence model on the normalised stream (O4)** with a within-t ranking loss | Pilot 4 shows trajectory shape carries independent information | XL |

---

## N. NOVEL-STREAM INFRASTRUCTURE DESIGN

Written and committed with this document: `research/scripts/novel_streams/harness.py`.
It is deliberately thin — every heavy component already existed and is *wired*, not rebuilt.

```python
class StreamingMechanism:
    name: str
    cols: list[str]
    def fit_history(self, hist):          ...   # per-series state, history ONLY
    def emit(self, hist, online):         ...   # (n_online, k) float32, causal
    # optional strictly-online form
    def initialize_online(self, state):   ...
    def update(self, state, x_t):         ...
    def current_features(self, state):    ...
```

```
mechanism plugin
   -> harness.verify()            bitwise prefix invariance, atol=0.0  [sbr contract]
   -> harness.build()             cached (n_rows, k) float32 memmap in cache/novel_streams/
   -> harness.diagnostic_pack()   whole-fold / dominant-cell / negtype / t- / age-buckets
                                  + within-t rank correlation with the RT-600 blend
   -> harness.pair_flow_by_cell() repairs / damage / net, split by cell and negative type
   -> [optional] one LightGBM specialist on 500 + new columns
   -> harness.marginal()          RT600 vs RT600+clone vs RT600+candidate   [BINDING]
   -> KILL or PROMOTE
```

**Verified working end-to-end while writing this document.**
`research/scripts/novel_streams/m20_dwell_probe.py` is a runnable three-column reference
mechanism (the Pilot 1 skeleton). `python m20_dwell_probe.py` runs
`verify → build → diagnostic_pack` in under a minute over all 10,000 series and prints:

```
prefix invariance (atol=0.0): True  ok
       res64_run90  cell 0.57014 (rt600 0.67771)  vs-neverbreak 0.57326  rho +0.366
    res64_maxrun90  cell 0.60791 (rt600 0.67771)  vs-neverbreak 0.61322  rho +0.367
      res64_mass90  cell 0.57009 (rt600 0.67771)  vs-neverbreak 0.57313  rho +0.368
```

**And `verify()` earned its place on its first call.** The first version of the probe
initialised the run and mass arrays to *zero* and returned an all-NaN block when
`n_online < W`. `verify()` rejected it immediately (series 7588, prefix 3): the full build
emitted `0.0` for rows before the window filled while the truncated build emitted `NaN` for
the same rows. Not a look-ahead — but exactly the class of inconsistency that would surface
as a batch/stream parity failure at deployment time, which
`STATE_OF_RESEARCH.md` names as the project's largest outstanding deployment risk. The
fix and the reason are documented in the probe's own docstring.

**Setup for a new agent (2 minutes, no data copying):**
**Importing needs no setup.** `harness.py` resolves its own checkout from `__file__`, so
`import harness` works in a fresh `research/current` clone with no environment variables,
no `PYTHONPATH` and no caches. That is covered by `tests/test_novel_streams_harness.py`,
which runs the import in a scrubbed subprocess.

**Running a mechanism needs the data caches**, which are not redistributed. Point `SBR_ROOT`
at a tree that carries them; the code still loads from the checkout `harness.py` lives in:

```bash
cd "<repo>/structural-break-research-current"
mkdir -p cache && ln -s .../structural-break-claude-wave3/cache/store cache/store
                  ln -s .../structural-break-claude-wave3/cache/features cache/features
mkdir -p research/oof && ln -s .../structural-break-wave7-promotion/research/oof/*.npy research/oof/
export SBR_ROOT="$PWD"
```
(`cache/` and `research/oof/` are already gitignored, as are `*.npy` and `*.npz`; cached
streams and diagnostic arrays land in `cache/novel_streams/`, never in the source tree.)

**Do not duplicate scoring code.** `sbr.metric.ts_auc_flat`, `wave5_lib.Ctx`,
`wave8_common.ensemble_marginal` and `wave8_common.pair_repair_stats` are the only
implementations that may be used, and all four are **already on this branch** —
`wave8_common.py` is at `research/scripts/wave8_common.py`. Nothing needs cherry-picking,
and no sibling worktree is required; `tests/test_novel_streams_harness.py` asserts that
every one of those dependencies resolves inside this checkout.

*Future refactor, deliberately not done now:* `wave8_common.py` is a permanent piece of
research infrastructure carrying a wave-specific name. A neutral name (`research_eval.py`,
`ensemble_eval.py`) would read better, but renaming touches imports in `wave7_d3r.py`,
`wave7_teacher_nested.py`, `tests/test_wave8_causality.py` and every Wave-8 script on the
sibling branch, and would break the historical reports that cite it by name. Stability
before the pilots wins; revisit only alongside a wider `research/scripts` tidy-up.

---

## O. INFRASTRUCTURE MULTIPLIERS

Five, each unlocking many mechanisms rather than one.

| # | addition | unlocks | cost |
|---|---|---|---|
| **IM1** | `novel_streams/harness.py` — the plugin API and standard pack **(written)** | every mechanism in this document | done |
| **IM2** | **Path-functional null calibration**: extend the `NullCal` idea from rolling means to run lengths, excursion areas, first-passage times and running maxima, cached per series | A1–A7, D3–D5, I1–I4, E1, N4 — 17 mechanisms | SMALL |
| **IM3** | **Per-series historical-model cache**: AR(p) coefficients, Hankel-DMD operator + subspace, GPD/Hill tail fit, run-length null, the 23-column fingerprint — one memmap, computed once (fingerprint bank measured at **7 s / 10,000 series**) | C1–C7, J1–J5, I1–I3, O1–O3 — 15 mechanisms | SMALL |
| **IM4** | **Cell-restricted scorer** as one reusable function with the dominant-cell, negative-type, t- and age-bucket splits **(written, in `harness.diagnostic_pack`)** | every pilot | done |
| **IM5** | **Pair-flow by cell** — `pair_repair_stats` with a cell mask and metric weighting **(written, in `harness.pair_flow_by_cell`)**; the Wave-7 T2 report explicitly declined to compute this split | every pilot; directly answers "does the candidate repair where the loss is?" | done |

IM2 and IM3 are the only ones still to build, both SMALL, and Pilot 1 needs IM2 anyway.
**Build IM2 as part of Pilot 1; build IM3 as part of Pilot 3.** Do not build them speculatively.

---

## P. HISTORICAL-DGP DIAGNOSTIC PLAN

Partly executed here; the rest specified.

| # | question | status |
|---|---|---|
| P1 | How heterogeneous are historical DGPs? | **Done.** 23-column fingerprint over all 10,000 series, `d2_fingerprint.py`, 7 s. |
| P2 | Does RT-600's dominant-cell loss concentrate by fingerprint? | **Done.** OOF Spearman +0.192 (never-break) / +0.116 (break) vs permutation controls ≈ −0.03. Weak, real, monotone-only. |
| P3 | Which fingerprints mark the hard never-break negatives? | **Done.** Heavy tails (kurtosis, q-ratio, low Hill α), long memory (VR(50)), few historical excursion episodes, short history. |
| P4 | Does *specialist* competence vary by fingerprint? | **Not run.** Pilot 9 part (ii). TINY — seven OOF vectors already on disk. |
| P5 | How predictable is the first online sample from history? | Not run. Cheap: one-step-ahead predictive log score at t=0. Feeds C5/C6. |
| P6 | How variable are AR coefficients and spectral profiles across series? | **Partly done** (fingerprint distribution). The decision-relevant version is the *conditional* one in P4. |
| P7 | Do the seven specialists' *errors* cluster into ≥ 2 manifolds? | Not run. K1, SMALL, uses the D1 per-series loss cube already on disk. |
| P8 | Is `has_break` predictable from history? | **Settled: NO.** Three independent tests (`m05_ctx`); break rate flat across DGP clusters. Do not retest. |

---

## Q. DOMINANT-CELL DIAGNOSTIC PLAN

| # | question | status |
|---|---|---|
| Q1 | Exact loss cube by t × age × negative type | **Done** (W7-D0), reproduced here. |
| Q2 | Is the loss concentrated in a few series? | **Done. No** — worst 10 % of never-break series hold 27.7 %. §E.1. |
| Q3 | How long do null excursions persist, and how does that scale with t? | **Done.** §E.4 — never-break p90 grows ~log t, mature-break p90 grows ~linearly. |
| Q4 | Does an explicit dwell state separate the cell? | **Done, partially.** 0.608 standalone, ρ = 0.367, better against never-break than pre-break negatives. §E.3. |
| Q5 | Does the naive extreme-value normalisation help? | **Done. No** — 0.580 vs 0.608. §E.5. The matched-length empirical run null is the correct construction. |
| Q6 | Is temporal order beyond lag-2 informative given the 500-vector? | **Not run.** Pilot 4's shuffled-history control is the cheapest decisive form of brief §44. |
| Q7 | Are there ≥ 2 failure manifolds? | Not run. K1. |
| Q8 | Does the candidate repair *where* the loss is? | Now answerable for every candidate via IM5. |

---

## R. EXPECTED REDUNDANCY WITH RT-600

The formal check every candidate must pass, and the calibration for reading it.

| reference | within-t rank ρ with the RT-600 blend | what it means |
|---|---:|---|
| exchangeable seed clone (`RT-401`) vs T2 | 0.910 | zero new information |
| T2 (`RT-995`) vs the blend | 0.674 | ~zero marginal (+0.00024) |
| `m09_back` vs the champion | 0.822 | *less* decorrelated than a seed change |
| **excursion dwell (`res64_maxrun90`) vs the blend** | **0.367** | lowest ever measured here — necessary, not sufficient |

**The rule (D.2 restated as a procedure):** ρ is a screening statistic only. The decision is
`cand − clone` from `ensemble_marginal`, supported by pair flow *inside the dominant cell,
split by negative type* (IM5). A candidate that repairs never-break inversions and damages
pre-break ones is telling you something even if its net is flat; a candidate whose repairs
and damage are both diffuse is a variance-reduction artifact.

Predicted redundancy per top-15 family (qualitative, with reasoning — no fabricated numbers):

* **LOW redundancy (different functional class):** A (contiguity), G (shape similarity),
  C1 (effective rank). Nothing in the bank computes these functionals at all.
* **MODERATE:** B, D, E, I, F1/F6, L3/L4. These read channels the bank already monitors but
  through summaries it does not form. Expect ρ in the 0.4–0.7 band.
* **HIGH:** M (score transforms — already measured near zero), H2 (a re-combination of
  information the ensemble has), J5, K4. Ranked accordingly.
* **UNKNOWN and worth measuring precisely because unknown:** J1/J2. The `m05_ctx` result
  says the *encoding* was fatal; it does not say the conditioning is redundant.

---

## S. RESEARCH EXECUTION QUEUE

Ordered by information gain per hour, **not** by sophistication. Failure of any one does not
imply failure of the next — they draw on eight different information channels.

| # | experiment | what it tests | wall clock |
|---|---|---|---|
| 1 | **J4 specialist-competence diagnostic** (Pilot 9 part ii) + **K1 failure-manifold clustering** | Is there routable structure at all? Uses only OOF vectors + the fingerprint bank already on disk. Decides whether Family J/K deserve a slot. | **20 min** |
| 2 | **Pilot 2, arm F** — relay integrator / thermal replica / recloser count on the RT-600 score | The cheapest possible test of the whole persistence-logic idea. Pure post-processing on existing OOF arrays. | **35 min** |
| 3 | **IM2 + Pilot 1** — run-length null calibration and the excursion dwell bank | The single best-evidenced new channel (§E.3–E.5). | **1 h 30** |
| 4 | **Pilot 10** — joint size×duration rarity, reusing Pilot 1's caches | Whether the *joint* rarity beats the marginal dwell. | **40 min** |
| 5 | **Pilot 5** — scale-survival count | Multiscale persistence, orthogonal to duration. | **45 min** |
| 6 | **Pilot 7** — ordinal transition divergence + time irreversibility | Nonlinear/ordinal channel, unrelated to 1–5. | **45 min** |
| 7 | **Pilot 6** — spectral impulsiveness contrast | Frequency channel, unrelated to 1–6. | **1 h** |
| 8 | **Pilot 9 part (i)** — scalar difficulty gate with derangement control | Historical-DGP conditioning, done the way `m05_ctx` was not. | **1 h** |
| 9 | **IM3 + Pilot 3** — Kalman NIS and Hankel-DMD observer residuals | Individualised null / system-ID channel; the precursor for radical redesign O2. | **2 h** |
| 10 | **Pilot 4** — FLOSS arc-crossing + NN provenance, with the shuffled-history control | Trajectory geometry; also the decisive cheap form of brief §44. | **2 h 30** |
| 11 | **Pilot 8** — weighted conformal test martingale vs its unweighted incumbent | Distribution-free sequential evidence, aimed at the never-break population. | **1 h 30** |
| — | *stop and re-plan* | Whatever cleared +0.0010 goes to 5 folds; everything else is written into `FAILED_EXPERIMENTS.md`. | — |

Total to a full first sweep: **≈ 13 hours** across eleven independent information channels.
That is less than one Wave-8 mechanism cost.

---

## T. THE THREE MOST IMPORTANT SCIENTIFIC QUESTIONS

**T1. Does the prefix contain a *duration* statistic that the 500 columns never form?**
Everything in the bank is a level, a level's peak, or a fraction of time above a level.
§E.4 measures that the class-separating quantity is how the *longest contiguous* deviation
scales with elapsed time. If Pilot 1 returns `cand − clone < +0.0010`, the answer is no and
the persistence story — which has motivated this project since wave 1 — is exhausted as a
feature-engineering direction.

**T2. Is the never-break false-positive population a null-misspecification population?**
§E.2 says the hard negatives are heavy-tailed, long-memory series whose own histories rarely
wander, and that RT-600's loss rate on them is weakly predictable from history alone
(ρ = +0.19 OOF). If Pilots 9 and Pilot 3 both fail, then the overlap between those series'
historical and online laws is genuine and irreducible — and the ceiling is a property of the
data, not of our representation. That would be the single most valuable negative result
available, because it would close Families C, J and O1 at once.

**T3. Is temporal order beyond lag-2 products recoverable, and does it carry label
information conditional on the 500-vector?**
W7-D3R showed the *future* resolves the cell. Wave 8 showed five ways of importing that
future all fail. The untested third possibility is that the *past trajectory shape* — not
its moments — already carries part of it. Pilot 4's shuffled-history control answers this
directly and cheaply, and its answer decides whether any sequence model (O4) is ever worth
building.

---

## U. TOP THREE EXPERIMENTS TO RUN FIRST

1. **Queue #1 — J4 + K1 (20 min).** Two diagnostics, zero new computation beyond what is
   cached. They decide whether two whole families are alive, and they cannot be contaminated
   by anything because they train nothing that gets promoted.
2. **Queue #2 — Pilot 2 arm F (35 min).** Relay logic on the RT-600 score. If a
   two-time-constant accumulator with an explicit reset does nothing on top of the already-
   measured EWMA (+0.0005), Family B is dead for 35 minutes of compute and Family M with it.
3. **Queue #3 — IM2 + Pilot 1 (1 h 30).** The best-evidenced new channel in this document,
   built the way §E.5's negative says it must be built.

---

## U-BIS. ROUTE TO 0.650 — WHAT IS MEASURED, WHAT IS NOT

Required, per brief §52. **No arithmetic that adds unmeasured deltas.**

| category | value | evidence |
|---|---|---|
| **Measured external anchor** | 0.6268 | LB-001, frozen |
| **Measured internal marginal alpha available today** | **+0.00024** (T2) | the only positive ensemble-integration number anywhere in the project |
| **Measured to be unavailable** | −0.0003 × 5 (Wave 8), −0.0060 (score runmax), −0.0219 (DGP gating), −0.0177 (`m05_ctx`), −0.00095 (13-booster union) | ledger |
| **Oracle headroom (not achievable)** | +0.1696 pooled if the dominant cell were perfectly repaired; +0.0359 pooled translated from Arm C's future-information gap | W7-D0, W7-D3R |
| **Required to reach 0.650 from the dev pooled level** | +0.0232, i.e. repair **13.68 %** of the dominant cell's inversions | W7-D0 §H |
| **Plausible but unmeasured** | every mechanism in this document | — |

**What this document does and does not claim.** It claims that nine specific functional
classes are absent from the current representation (§C), that four of them are measurably
relevant to the dominant cell (§E), and that eleven independent experiments totalling ~13
hours will falsify or support them. **It does not claim any of them will produce alpha, and
it assigns no numerical TS-AUC estimate to any of them.** The honest prior, given that the
last two waves produced one +0.00024 and five negatives, is that most of the eleven will
fail. The reason to run them anyway is that they are *cheap and independent*: the project's
last two waves spent ~20 hours to test one idea five ways, and this queue spends 13 hours to
test eleven ideas once each, in eight different information channels.

A route to 0.650 exists arithmetically — 13.68 % of one cell's inversions — and there is
**no evidence in this repository that any known mechanism captures that much.** Anyone who
writes "+0.010 here plus +0.008 there" is doing the thing the T2 result exists to forbid.

---

## V. SOURCES

**Run and scan statistics / large deviations**
1. P. Erdős, A. Rényi, "On a new law of large numbers", *J. Analyse Math.* 23 (1970).
2. L. Gordon, M. F. Schilling, M. S. Waterman, "An extreme value theory for long head runs", *Probab. Theory Relat. Fields* 72 (1986) 279–287. https://link.springer.com/article/10.1007/BF00699107 · http://www.csun.edu/~hcmth031/GSW.pdf
3. A. Amărioarei, C. Preda, "One-dimensional discrete scan statistics for dependent models and some related problems", *Mathematics* 8(4):576 (2020). https://doi.org/10.3390/math8040576
4. J. Glaz, J. Naus, S. Wallenstein, *Scan Statistics*, Springer (2001).
5. "Detection and estimation of multiple transient changes", arXiv:2112.06308. https://arxiv.org/pdf/2112.06308

**Electrical protection**
6. IEEE Std C37.112-2018, *IEEE Standard for Inverse-Time Characteristics Equations for Overcurrent Relays* (and C37.112-1996). https://standards.ieee.org/ieee/C37.112/7036/ · https://ieeexplore.ieee.org/document/8635630/

**Control / estimation / fault detection**
7. Y. Bar-Shalom, X.-R. Li, T. Kirubarajan, *Estimation with Applications to Tracking and Navigation*, Wiley (2001) — NIS and consistency testing.
8. "Online tests of Kalman filter consistency", *Int. J. Adaptive Control and Signal Processing*. https://www.researchgate.net/publication/276924230_Online_tests_of_Kalman_filter_consistency
9. "A modified whiteness test for damage detection using Kalman filter innovations", *Structural Control and Health Monitoring*. https://www.researchgate.net/publication/226975932_A_Modified_Whiteness_Test_for_Damage_Detection_Using_Kalman_Filter_Innovations
10. "Kalman filter damage detection in the presence of changing process and measurement noise", *MSSP* (2013). https://www.sciencedirect.com/science/article/abs/pii/S0888327013000915

**Koopman / DMD / nonlinear dynamics**
11. G. A. Gottwald, F. Gugole, "Detecting regime transitions in time series using dynamic mode decomposition", *J. Stat. Phys.* 176 (2019). https://link.springer.com/article/10.1007/s10955-019-02392-3 · https://arxiv.org/pdf/1904.09082
12. "Online changepoint detection via dynamic mode decomposition", arXiv:2405.15576. https://arxiv.org/pdf/2405.15576
13. "Change-point detection in industrial data streams based on online DMD with control", arXiv:2407.05976. https://arxiv.org/pdf/2407.05976

**Matrix profile / motifs**
14. S. Gharghabi, Y. Ding, C.-C. M. Yeh, K. Kamgar, L. Ulanova, E. Keogh, "Matrix Profile VIII: Domain agnostic online semantic segmentation at superhuman performance levels", *ICDM* (2017); extended in *Data Min. Knowl. Disc.* (2019). https://link.springer.com/article/10.1007/s10618-018-0589-3 · https://pmc.ncbi.nlm.nih.gov/articles/PMC6373324/
15. C.-C. M. Yeh et al., "Matrix Profile I: All pairs similarity joins for time series", *ICDM* (2016). https://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf
16. STUMPY documentation, semantic segmentation / FLOSS. https://stumpy.readthedocs.io/en/latest/Tutorial_Semantic_Segmentation.html

**Recurrence analysis**
17. N. Marwan, M. C. Romano, M. Thiel, J. Kurths, "Recurrence plots for the analysis of complex systems", *Physics Reports* 438 (2007) — laminarity, trapping time. https://www.recurrence-plot.tk/rqa.php
18. "Trends in recurrence analysis of dynamical systems", arXiv:2409.04110. https://arxiv.org/html/2409.04110v1
19. "Detection of dynamical regime transitions with lacunarity as a multiscale recurrence quantification measure", arXiv:2101.10136. https://arxiv.org/pdf/2101.10136

**Vibration / spectral kurtosis / cyclostationarity**
20. J. Antoni, "The spectral kurtosis: a useful tool for characterising non-stationary signals", *MSSP* 20 (2006); "The spectral kurtosis: application to the vibratory surveillance and diagnostics of rotating machines", *MSSP* 20 (2006).
21. Y. Wang, J. Xiang, R. Markert, M. Liang, "Spectral kurtosis for fault detection, diagnosis and prognostics of rotating machines: a review with applications", *MSSP* 66–67 (2015). https://www.sciencedirect.com/science/article/abs/pii/S0888327015002897
22. "Applications of robust statistics for cyclostationarity detection in non-Gaussian signals for local damage detection in bearings", arXiv:2502.07478. https://arxiv.org/pdf/2502.07478

**Game-theoretic statistics / anytime-valid inference**
23. A. Ramdas, P. Grünwald, V. Vovk, G. Shafer, "Game-theoretic statistics and safe anytime-valid inference", *Statistical Science* 38 (2023). https://arxiv.org/pdf/2210.01948
24. V. Vovk, "Testing randomness online" / conformal test martingales, *Statistical Science* 36 (2021).
25. V. Volkhonskiy, E. Burnaev, I. Nouretdinov, A. Gammerman, V. Vovk, "Inductive conformal martingales for change-point detection", arXiv:1706.03415. https://arxiv.org/pdf/1706.03415
26. "WATCH: Weighted adaptive testing for changepoint hypotheses via weighted-conformal martingales", arXiv:2505.04608. https://arxiv.org/html/2505.04608v1
27. Y. J. Choe, A. Ramdas, "Combining evidence across filtrations using adjusters". https://yjchoe.github.io/slides/ChoeRamdas2024-CombiningEvidenceAcrossFiltrations-Slides.pdf

**Duration modelling**
28. Y. Guédon / J. Bulla & I. Bulla, "hsmm — an R package for analyzing hidden semi-Markov models", *CSDA* (2010). https://www.sciencedirect.com/science/article/abs/pii/S016794730800426X
29. "Hidden semi-Markov models with inhomogeneous state dwell-time distributions", *CSDA* (2025). https://www.sciencedirect.com/science/article/pii/S0167947325000477
30. "Hidden Markov and semi-Markov models: when and why are these models useful for classifying states in time series data?", arXiv:2105.11490. https://arxiv.org/pdf/2105.11490

**Critical transitions**
31. M. Scheffer et al., "Early-warning signals for critical transitions", *Nature* 461 (2009). https://pdodds.w3.uvm.edu/files/papers/others/2009/scheffer2009a.pdf
32. V. Dakos, S. R. Carpenter, et al., "Robustness of variance and autocorrelation as indicators of critical slowing down", *Ecology* 93 (2012). https://esajournals.onlinelibrary.wiley.com/doi/10.1890/11-0889.1
33. T. M. Lenton et al., "Early warning of climate tipping points from critical slowing down: comparing methods to improve robustness", *Phil. Trans. R. Soc. A* 370 (2012). https://royalsocietypublishing.org/rsta/article/370/1962/1185/114587/

**Topology**
34. J. A. Perea, J. Harer, "Sliding windows and persistence: an application of topological methods to signal analysis", *Found. Comput. Math.* 15 (2015). https://link.springer.com/article/10.1007/s10208-014-9206-z · https://arxiv.org/pdf/1307.6188

**Ordinal / information-theoretic**
35. C. Bandt, B. Pompe, "Permutation entropy: a natural complexity measure for time series", *PRL* 88 (2002).
36. J. B. Ramsey, P. Rothman, "Time irreversibility and business cycle asymmetry", *J. Money Credit Banking* 28 (1996).

**Internal (this repository, verified)**
37. `research/reports/wave7/WAVE7_RT600_EXACT_ALPHA_BUDGET.md` — W7-D0 exact loss cube.
38. `research/reports/wave7_d3r.md` — W7-D3R three-arm information-frontier diagnostic.
39. `research/reports/wave7_t2_promotion_final.md`, `wave7_t2_pairflow.md` — T2 MOSTLY REDUNDANT.
40. `research/reports/wave8_final.md` (branch `research/wave8-future-aware-distillation`) — five KILLs.
41. `research/FAILED_EXPERIMENTS.md`, `research/RDOF_LEDGER.md`, `research/RESULTS.csv`, `research/FINAL_ARCHITECTURE_FREEZE.md`.
42. `research/reports/new_avenues_2026_diagnostics.json` — D1–D6, produced for this document.

---

## Closure note — G5 and H3, 2026-09-04 (`RT-1321`)

**G5 ("delay-embedded attractor separation", priority HIGH, `after Pilot 4`) and
H3 ("sequential kernel MMD", explicitly filed as collapsing into G5) are now
EXECUTED and CLOSED.**

G5's own gap analysis was correct and is worth quoting because it was the reason
the arm was funded: *"every incumbent distance is on the MARGINAL (1-D); the
delay-embedded cloud is joint over d lags."* `RT-1321` built exactly that — a
Gaussian characteristic-kernel (RFF) discrepancy between the online kernel mean
of causal delay vectors and a frozen history-only kernel mean — and tested it
against a **coordinate-separable kernel control** that isolates the joint-vs-
marginal distinction and nothing else.

The gap was real. The joint block is genuinely decorrelated from the champion
(within-`t` rho **0.16**) and measurably beats its own marginal control both as a
member and in dominant-cell pair flow. It is also worth nothing: block standalone
**0.5425** against the incumbent's 0.6403, conditional AUC of **0.4712** — below
chance — on exactly the pairs the incumbent inverts, and a ninth-member
deployment endpoint of **`E2-E0 = -0.000034`**.

G5's own preregistered falsification (`cand-clone < +0.0010`) fires. Its
predeclared risk column said `LOW-MODERATE` redundancy and `HIGH` novelty; both
were right, and both were irrelevant — the failure is not redundancy, it is that
decorrelated information does not repair pairs. That is the fourth confirmation
of the two-sided squeeze after `RT-1201` (rho 0.38), `RT-1258` (rho 0.585) and
`RT-1202` (rho 0.004).

See `research/reports/rt1321_lskd/RT1321_RESULT.md` and the `RT-1321` /
`RT-1322` rows in `NEGATIVE_RESULTS_INDEX.md` and `FAILED_EXPERIMENTS.md`.
