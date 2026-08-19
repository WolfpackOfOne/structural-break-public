# STATE OF RESEARCH — 2026 ADIA Lab / CrunchDAO Structural Break Challenge (Real-Time Edition)
**Research Director (Agent 0) · 2026-08-19 · 79 logged experiments · 8 agents · waves 1-6 complete**

---

## DEPLOYMENT CHAMPION — four-stream logit average, six modules (`RT-190`)

| | |
|---|---|
| **Architecture** | 6 feature modules (441 cols) → 4 LightGBM streams → mean of logits |
| **Pooled OOF TS-AUC** | **0.62368** |
| **Per-fold** | 0.63410 / 0.61730 / 0.63560 / 0.61470 / 0.61720 |
| **Inference cost** | **64.9 ms/point** (62.6 with the ported `m00_core`) |
| **Fits** | parallelism-6 budget (64.3 ms/point, 10,000-series public set) |

`m02_dist` is dropped: it is the most expensive module (29.7 % of inference cost)
and the model is **slightly better without it** (+0.0034 single-model). Deployability
costs 0.0018 against the research champion below. All streams share one feature
computation, so cost depends on the module set, not the stream count.

## RESEARCH CHAMPION — seven-stream **logit-average** ensemble (`RT-160`)

| | |
|---|---|
| **Architecture** | 7 LightGBM streams → equal-weight average of **logits** |
| **Mean OOF TS-AUC** | **0.62561** (pooled 0.62544) |
| **Per-fold** | 0.63653 / 0.61998 / 0.63622 / 0.61762 / 0.61772 (std 0.00868) |
| **Ensemble delta over the best single model** | **+0.01025** |
| **Weights** | none, and none is the right answer — see below |

**The blend must be a per-series function, and this nearly went wrong.** The
previous champion (`RT-131`) averaged *within-timestep rank percentiles* — the
rank of a series' score among all series alive at that online index. That is a
function of the cross-section, and the crunch runner is series-sequential and
single-pass: it must emit series `i`'s score at step `t` before it has seen series
`i+1`. **The cross-section does not exist at inference, so RT-131 could never have
been submitted.** Averaging logits is deployable, and it costs nothing —
0.62544 against the rank average's 0.62524, fractionally *better*. Frozen
per-stream quantile normalisation gives 0.62500; raw probability averaging gives
0.62288 and should be avoided, because the pairwise stream emits an unbounded
margin rather than a probability and a plain mean is scale-mismatched.

**Weighting and selection both lose.** Honest leave-one-fold-out greedy subset
selection scores 0.62442 (−0.0008 against simply using everything); a LOFO
logistic stack 0.62514; a LOFO LightGBM stack 0.62144. The best subset *chosen in
hindsight* beats the plain average by 0.0003 — and the gap between that 0.0003
and the honest −0.0008 is exactly the self-deception on offer. With seven streams
inside a 0.010 band and a per-fold spread of 0.011, any weighting scheme is
fitting fold noise. **The equal average has no parameters, so it has no parameter
variance.** The submission blend rule is therefore: average the rank percentiles
of every stream you have.

**Lockbox.** The 4-stream rank-average version scored **0.61214** on the 2,000
untouched series against a 0.62374 dev estimate (−0.0116). The 7-stream logit
blend has *not* been re-measured there and deliberately will not be: the
composition rule gained no free parameters, so there is nothing new to confirm,
and the lockbox is worth more unspent. Expect the same ~0.011 haircut, i.e.
roughly **0.614**.

### The seven streams

| stream | features | model bias | OOF | corr. with `RT-100` (within-t / global) |
|---|---|---|---|---|
| `RT-100` full bank | all 7 modules, 500 cols | 63 leaves, ff 0.5, 1M rows | **0.61500** | — |
| `RT-123` pairwise | 500 cols | pairwise-t ranking loss | 0.61444 | 0.780 / 0.717 |
| `RT-125` GOSS | 500 cols | gradient-based one-side sampling | 0.61355 | 0.780 / 0.927 |
| `RT-122` extra-trees | 500 cols | 255 leaves, per-series sampling | 0.61307 | 0.685 / 0.879 |
| `RT-124` recursion-only | `m01`+`m06`+`m07`, 170 cols | no window-bank features at all | 0.61064 | 0.613 / 0.862 |
| `RT-120` evidence-heavy | `m00`+`m01`+`m07`, 261 cols | 127 leaves, ff 0.35 | 0.60855 | 0.662 / 0.876 |
| `RT-121` shape/dynamics | `m02`+`m03`+`m04`+`m06`, 239 cols | 31 leaves, ff 0.7 | 0.60468 | 0.617 / 0.841 |

**Global Pearson correlations run 0.60–0.93; within-timestep rank correlations run
0.39–0.78.** The global figure is inflated by a shared time trend the metric never
scores. Measured in the space the metric actually compares, these are genuinely
different models — and the most decorrelated pair (`RT-121`/`RT-124`, ρ = 0.393)
are two of the three *weakest* streams. Note also `RT-123` and `RT-125`: nearly
identical within-timestep correlation to the champion (0.780 both) but wildly
different global correlation (0.717 vs 0.927), which is a clean demonstration that
the global number carries almost no information about blend value here.

### Best single model (`RT-100`)

| | |
|---|---|
| **Architecture** | 7 causal feature modules (500 columns) → LightGBM binary logloss |
| **Mean OOF TS-AUC** | 0.61510, pooled 0.61500 |
| **Per-fold** | 0.62903 / 0.61061 / 0.62688 / 0.60747 / 0.60152 (std 0.01091) |
| **LOCKBOX** | 0.60791 — gap to dev mean −0.0072, inside the fold spread |
| **Training** | 1,000,000 sampled rows/fold, 600 trees, lr 0.05, 63 leaves, ff 0.5 |
| **Runtime** | ~6.5 min/fold on 2 cores; feature build ~0.25 s/series |
| **Causality** | every module passes bitwise prefix invariance (atol = 0.0) |

**Components** — `m00_core` (calibrated multi-scale null evidence, 151), `m01_seq`
(sequential detector bank with peak/persistence shape channels, 60), `m02_dist`
(PIT / occupancy / divergence, 59), `m03_dyn` (dependence, spectral, wavelet,
complexity, 60), `m04_resid` (AR and volatility residual representations, 60),
`m06_loc` (online change-point localisation, 60), `m07_bayes` (absorbing-state
posterior, BOCPD, e-processes, 50).

### Feature selection: the bank is not redundant (`RT-140`)

Nested selection — importance ranking and every refit use folds 1–4 only, fold 0
is a held-out score per candidate k:

| columns | held-out TS-AUC |
|---|---|
| all 500 | 0.62689 |
| top 300 | 0.62645 (−0.0004) |
| top 200 | 0.62111 (−0.0058) |
| top 120 | 0.61885 (−0.0080) |
| top 60 | 0.60952 (−0.0174) |

There is no free lunch in pruning: the 500 columns earn their place collectively
even though most are individually weak. **top-300 is a 40 % inference-cost
reduction for −0.0004**, which is the right trade for deployment but not a
research gain.

## STARTING BASELINE

| | TS-AUC | Δ |
|---|---|---|
| `RT-000` shipped EWMA + CUSUM + variance noisy-OR | 0.52051 | — |
| `RT-101` `m00_core` alone (calibrated null evidence + LightGBM) | 0.56349 | +0.0430 |
| `RT-100` best single model | 0.61500 | +0.0945 |
| `RT-130` 4-stream ensemble | 0.62374 | +0.1032 |
| `RT-131` 7-stream rank average (not implementable) | 0.62524 | — |
| **`RT-160` 7-stream logit average (deployable)** | **0.62544** | **+0.1049** |

Paired series-level bootstrap (120 replicates, resampling whole series):
champion − baseline 0 = **+0.0951, 95 % CI [+0.0841, +0.1059], 120/120 replicates positive**;
champion − `m00_core`-only = **+0.0524, CI [+0.0424, +0.0610], 120/120 positive**.

## PUBLIC BENCHMARK REPLICATION

The strongest public 2026 numbers described in the deep-research report are
~0.575 (calibrated + AR/GARCH residual + LightGBM stack) and ~0.579 (648
engineered causal features → 200 → LightGBM, series-level CV). We do not have
their code, so this is a reconstruction on our folds, not a byte-level replication:

| Public approach | Our nearest reconstruction | Our score |
|---|---|---|
| raw cumulative statistic (~0.518) | `RT-000` handcrafted detector | 0.5205 |
| + per-series empirical null (~0.533) | `m00_core` expanding-null block | ~0.55 (screen) |
| + residual monitoring (~0.557) | `m00_core`+`m04_resid` | 0.6024 (screen fold 0) |
| + LightGBM stack (~0.575–0.579) | `RT-101` `m00_core` alone, full 5-fold | 0.5635 |
| — | `RT-100` best single model, full 5-fold | 0.6150 |
| — | **`RT-160` ensemble, full 5-fold** | **0.6254** |

We clear the public band by ~0.046 on 5-fold series-level OOF and by ~0.033 on
the untouched lockbox.

## BREAK TAXONOMY — what the competition actually contains

From 8,000 dev series (`research/reports/break_taxonomy.md`):

- **Location breaks essentially do not exist.** Median |mean shift| is 0.054 σ for
  break series versus 0.053 σ for placebo splits; series-level AUC 0.4998. Every
  mean-shift detector in the classical literature is aimed at a break type this
  data does not contain.
- **Scale dominates**, then dependence: post-vs-pre AUC 0.559 (scale), 0.558
  (Wasserstein), 0.538 (dependence), 0.522 (shape), 0.518 (trend).
- The strongest single statistic found anywhere in this project is the
  **AR(6)-residual log-sd ratio** at 0.603 — +0.035 over the raw variance ratio
  and +0.009 over AR(2).
- **92.4 % of break series are individually indistinguishable from a placebo split**
  at p < 0.01 on any family. This is a weak-signal aggregation problem, not a
  detection problem, which is why a 500-column learned combination beats every
  handcrafted rule by so much.
- **17.0 % of no-break series contain a break-lookalike transient** — but so do
  15.6 % of matched-length break-free historical windows. Transients are the DGP,
  not the break. **The disambiguator has to be persistence, not amplitude**, and the
  feature importances confirm the model found exactly that (see below).
- Champion performance by break class: `mixed` 0.828, `dep_dominant` 0.697,
  `shape_dominant` 0.691, `trend_dominant` 0.687, and the 3,673-series
  `weak_unclassified` mass 0.601 — that mass is where the metric is actually won.

## WHERE THE METRIC IS WON

| elapsed online index | rows | champion | baseline 0 |
|---|---|---|---|
| 0–20 | 159 k | 0.5247 | 0.4986 |
| 20–50 | 233 k | 0.5460 | 0.5062 |
| 50–100 | 373 k | 0.5539 | 0.5158 |
| 100–200 | 685 k | 0.5996 | 0.5181 |
| 200–400 | 1.13 M | 0.6161 | 0.5203 |
| 400–1000 | 1.45 M | **0.6473** | 0.5254 |

By tau quartile: early breaks 0.628 / 0.624, late breaks 0.588 / 0.571 — a late
break simply has fewer post-break observations to accumulate evidence from. By
online length: 0.551 (<200) rising to 0.641 (700–1000).

**Metric-weight sensitivity.** The official pair-count weighting gives 0.61500;
unweighted-over-timesteps gives 0.62652, alive-count weighting 0.61143. Our
conclusions do not depend on which weighting the organiser actually uses — worth
knowing, because the deep-research report flags that the exact weight formula was
not verifiable from public material.

## BEST SIGNAL FAMILIES

Screen-protocol marginal contribution on top of `m00_core` (fold 0, identical seed,
each agent reproduced the 0.57384 control to the last digit):

| module | cols | +Δ on `m00_core` | standalone | champion gain share | gain/col |
|---|---|---|---|---|---|
| `m07_bayes` absorbing-state / BOCPD / e-values | 50 | **+0.0397** | 0.5992 | 24.6 % | **124.8 k** |
| `m03_dyn` dependence + spectral + complexity | 60 | +0.0300 | 0.5578 | 14.8 % | 62.5 k |
| `m06_loc` online change-point localisation | 60 | +0.0315 | — | 3.7 % | 15.7 k |
| `m04_resid` AR / volatility representations | 60 | +0.0286 | 0.5814 | 13.9 % | 58.7 k |
| `m01_seq` sequential bank + peak/persistence | 60 | +0.0143 | 0.5439 | 18.0 % | 76.1 k |
| `m02_dist` PIT / occupancy / divergence | 59 | +0.0131 | 0.5467 | 8.9 % | 38.1 k |
| `m00_core` calibrated multi-scale null | 151 | — | 0.5738 | 16.3 % | 27.4 k |
| `m05_ctx` historical context (series-constant) | 50 | +0.0009 | 0.5014 | **rejected** | — |

**The single most reliable improvement is per-series historical-null calibration**,
which is not one module but the property every module shares: raw detector
magnitudes are not comparable across series, and TS-AUC only ever compares series
against each other. Everything downstream is built on it.

**Top 10 champion features** (gain): `m07_bayes::ab_fast` (absorbing-state posterior,
fast mixture), `m00_core::t_online`, `m07_bayes::ab_var_z`, `m07_bayes::bo_lo_change_pk`,
`m07_bayes::ev_pow_pk`, `m07_bayes::bo_mean_rel`, `m01_seq::gle_pkr`, `m01_seq::glz_pk`,
`m00_core::w8_sur_tail_x`, `m01_seq::phd_pkr`. Note how many are **peak / running-peak /
peak-ratio channels** — the model's most valuable inputs describe the *shape and
persistence* of accumulated evidence, exactly the transient-vs-permanent
disambiguation the forensics said was required.

## PERSISTENCE — the running-max question, resolved

Applied post-hoc to the champion's OOF (so the comparison is exact):

| transform | TS-AUC | mean score on negatives |
|---|---|---|
| raw (champion) | 0.61500 | 0.2026 |
| **running max** | **0.60896 (−0.0060)** | 0.2950 |
| decayed max γ=0.999 | 0.61167 (−0.0033) | 0.2750 |
| decayed max γ=0.99 | 0.61546 (+0.0005) | 0.2310 |
| decayed max γ=0.95 | 0.61553 (+0.0005) | 0.2094 |
| EWMA α=0.3 | 0.61552 (+0.0005) | 0.2012 |

**Verdict: do not ratchet.** Hard running max costs 0.006 because it permanently
inflates every no-break series that ever produced a transient spike — negatives'
mean score rises 46 % — and TS-AUC compares against those negatives at every later
timestep. Mild smoothing is worth +0.0005, which is inside noise. This settles the
contradiction between the two public sources: **running max as a FEATURE is one of
our most valuable inputs; running max as a POST-TRANSFORM on the final score is
harmful.** Both public claims are right about different objects.

## EXPERIMENTS THAT WON

1. Per-series historical-null calibration of every statistic at matched window length.
2. The absorbing-state Bayesian posterior with mixture mass on **variance and
   dependence** changes rather than mean shifts (`m07_bayes`, best gain/column).
3. Peak / decayed-peak / time-since-peak / persistence-count channels on every
   sequential recursion (`m01_seq`) — 41 % of that module's contribution.
4. Online change-point localisation: statistics of the estimated post-break segment
   instead of the whole online prefix (`m06_loc`, +0.0315).
5. Monitoring several *different* whitenings simultaneously (`m04_resid`) rather than
   picking one.
6. Dependence / spectral / complexity monitoring of the online stream (`m03_dyn`).
7. Normal-scoring the AR-whitened stream against the historical residual ECDF, which
   makes log Bayes factors genuinely comparable across series.

## EXPERIMENTS THAT FAILED

Full detail in `research/FAILED_EXPERIMENTS.md` (17 recorded negatives). Headlines:

- **`m05_ctx` series-constant historical context as features** — +0.0009 on screen,
  **−0.0177 at full scale** (`RT-102`, stopped after fold 0). The permutation control
  is the reason: within-fold-deranged context scores −0.057 *below* no context at all,
  i.e. 50 near-continuous series-constant columns give each training series a unique
  signature that LightGBM memorises. H1 ("history predicts whether a break occurs")
  is **rejected** on three independent tests.
- **Pairwise-t ranking objective** — +0.0072 on screen, −0.0033 in a paired fold-0
  test, and **0.61444 vs 0.61510 over five folds on 30 % fewer training rows**, i.e.
  a dead heat. *My* −0.0033 was as much noise as their +0.0072; see the amendment in
  `FAILED_EXPERIMENTS.md`. It is now a permanent ensemble member.
- **GARCH(1,1) volatility normalisation** — 0.50012 standalone. Literally zero signal:
  the filter adapts on the same timescale as the break it is meant to reveal.
- **Trend family** (cumulative slopes, Mann-Kendall) — −0.005; consistent with the
  forensics finding that trend breaks barely exist.
- **KS / tail-asymmetry / rank-CUSUM families** — no marginal value on top of the
  calibrated occupancy channels.
- **Alternative targets** — soft ramp −0.012, log-hazard regression −0.018,
  `scale_pos_weight=4` −0.009, XE-NDCG −0.014. Binary `1[t ≥ τ]` survived every attack.
- **Slope-of-evidence channels, mixture-GLR, EWMA level banks** — zero gain, dominated
  by the peak channels.
- **DGP-cluster gated specialists** — +0.032 on screen, **−0.0219 at full scale on
  both folds tested**. The clusters carry real information (gating beats its own
  permutation control by +0.0064) but partitioning 6,400 training series into six
  groups costs far more than the routing gains.
- **Ensemble weighting and subset selection** — honest LOFO subset selection −0.0008,
  logistic stack −0.0001, LightGBM stack −0.0038, all against the plain equal average.

## SURPRISING FINDINGS

1. **Location breaks do not exist in this data.** The entire classical mean-shift
   apparatus is aimed at the wrong target.
2. **The Bayesian module is both the best gain-per-column family and near-orthogonal
   to the rest** — within-timestep rank correlation with the `m00_core` stream is only
   **0.187**, against a global Pearson of 0.559. Global correlation is inflated by a
   shared time trend the metric does not score; the right correlation to measure is
   *within timestep*, and almost nobody would think to.
3. **Univariate feature AUCs are sign-inverted for most divergence channels** yet the
   same features are strongly positive jointly. Screening features by univariate AUC
   on this problem would discard the best ones.
4. **AR order matters in the opposite direction to the public folklore.** AR(5) beats
   AR(1) by +0.016 standalone and is the best single 8-column addition; the danger is
   not high order but *matching* the filter to the break.
5. **Three of three screen-level *architectural* wins reversed at full scale**
   (context block, pairwise objective, gating) while every screen-level *feature*
   addition transferred. All three reversals are the same mechanism: they help a
   data-starved model and stop helping once the model is not data-starved. The
   operating rule this bought us — **the screen store is valid triage for features
   and is not evidence for objectives, architectures, or anything that repartitions
   the training data** — is worth more than any single one of the results.
6. Standalone and incremental value are anti-correlated: raw representations are the
   best standalone and the worst incremental (−0.002); AR(2) is the worst standalone
   and among the best incremental (+0.021).

## VALIDATION RISKS

- **Selection intensity.** 67 logged experiments, most on the 2,500-series screen
  store, which is a subset of the dev folds. The champion feature set was chosen from
  screen results, so the dev-fold number carries some selection. **The lockbox says
  that selection cost us at most 0.007** (0.60791 vs 0.61510), which is inside the
  fold-to-fold spread. The lockbox has now been touched **twice** — once for the single
  model and once for the parameter-free ensemble — and should be considered spent.
  The ensemble's dev-to-lockbox gap is −0.0116, slightly wider than the single
  model's −0.0072, which is the price of having selected the blend on dev folds.
- **Screen-to-full transfer is not reliable.** Two screen findings reversed sign on
  promotion (context block, pairwise objective). Screen results are a triage filter,
  never evidence.
- ~~**Metric weighting** is unverified against the organiser's scorer.~~ **CLOSED
  2026-08-19.** The official docs state the metric as
  `TS-AUC = Σ_t w(t)·AUC(t) / Σ_t w(t)` with `w(t) = n_pos(t)·n_neg(t)`, which is
  exactly what `sbr/metric.py` implements. Every number in this project is measured
  against the right objective. See `research/reports/platform_constraints.md`.
- **Single seed.** The champion has not been re-run under a different seed or a
  different fold assignment. Fold std 0.011 is our only stability estimate.

## GENERATOR-ARTIFACT RISKS

- **Clean:** series id, store offset, row order and `n_hist` all score 0.489–0.499 for
  predicting `has_break`. Break rate is flat across DGP clusters (0.478–0.536). No
  history-only label leakage exists.
- **Real but handled:** every historical segment is exactly standardised (mean 0,
  sd 1.00000). The online/historical variance ratio carries a systematic negative bias
  scaling with tail heaviness (median −0.004 in the lightest historical-kurtosis
  quartile vs −0.063 in the heaviest) — **uncalibrated variance features mis-rank the
  cross-section**, which is precisely why every module calibrates per series.
- **Real and important:** any statistic *not* calibrated at matched window length
  encodes `n_post`, i.e. τ, rather than the break — the apparent effect-size/τ
  correlation (+0.23 to +0.33) is entirely a sample-size artifact reproduced by placebo
  series. Matched-length calibration is not a nicety, it is the difference between a
  feature and a τ-detector.
- τ is *almost* uniform but rejected (KS D = 0.0263, p = 0.008), tilted ~32.1 % / 28.9 %
  toward the first vs last 30 % of the online segment.
- `n_online` is **never used as a feature** — at online step t a competitor cannot know
  how much longer the series runs.

## COMPUTE / DEPLOYMENT RISKS

- **Streaming port is under way — see `research/reports/streaming_port.md`.** A
  provably-correct path now exists (`ReferenceStreamer`, bitwise identical to the
  batch matrix), the context is dual-mode so module code is shared between batch and
  stream, `m00_core` is ported with bitwise parity, and `tests/test_streaming.py`
  guards it. But the reference path costs **93 ms/point** across the seven champion
  modules, and six modules remain unported. Roughly the first fifth of the job.
- The remaining bottleneck is **per-column numpy call overhead**, not algorithmic
  cost: `m00_core` alone issues ~80 null-calibration calls and ~129 clips per emitted
  row, each on a length-1 array. Batching those into one vectorised call per step (or
  a numba row builder) is where the last order of magnitude lives.
- 500 columns × 600 trees is a large but deployable model; distillation has not been
  needed or tested.
- **Runtime budget is 15 hours/week** against a test set of 10,000 public +
  10,000 private series (~5.0M and ~10.1M scoring points). Measured cost today is
  92.9 ms/point on the correct-but-slow reference streamer, so the port needs
  **~2.2× at parallelism 4 on the public set, ~4.3× across both** — single digits,
  not the two orders of magnitude previously assumed. Parallelism is supported via
  `INFER_PARALLELISM`, with RAM scaling per process.
- **Determinism is a reward-eligibility condition**: re-running on 10 % of the data
  must reproduce predictions to 1e-8. Our inference path has no RNG and no
  cross-series state, so this should hold by construction, but it is untested.
- All research ran on **2 CPU cores and 7 GB RAM**, which shaped what was attempted:
  no deep sequence models, no Optuna at scale, no augmentation, no 2025 transfer.

## PUBLIC METHODS NOT YET REPRODUCED

- The ~648-feature / feature-selection-to-200 pipeline as such (we built 500 columns
  designed top-down instead; a selection pass over our bank has never been run).
- TabPFN-derived features and the 2025 stacking-of-independent-researchers architecture.
- 2025→2026 transfer (converting known-boundary 2025 series into pseudo-real-time
  examples). **Not attempted at all.**
- Synthetic augmentation and hard-negative training (transients that revert).

## NEXT EXPERIMENTS, RANKED (updated after wave 6)

| # | Experiment | Payoff | Uncertainty | Effort | Compute | Ensemble value |
|---|---|---|---|---|---|---|
| 1 | **Streaming port + batch/stream parity tests** — the submission blocker. Nothing here can be submitted today | — | low | high | low | — |
| 2 | More streams. Every stream added so far has paid: 4 streams +0.0087, 7 streams +0.0102. Cheapest next ones: a DART-boosted model, a different-seed champion, a soft-gated model (the gated model is bad alone but structurally different) | med | low | low | high | high |
| 3 | Raise the shared context AR order from 2 to 5–6 and rebuild. Forensics measured AR(6)-residual log-sd at 0.603, the strongest single statistic in the project. Partly redundant with `m04`/`m07`, which fit their own | med | med | low | high | — |
| 4 | Localisation v2: better τ̂, more segment statistics. `m06_loc` was built by an agent that ran out of budget before tuning it, and still delivered +0.0315 | med | med | med | high | med |
| 5 | 2025 data transfer as pseudo-real-time training series — never attempted, and the only idea left that adds *data* rather than model variety | high | high | high | med | med |
| 6 | Hard-negative augmentation: synthesised transients that revert | med | med | med | med | med |
| 7 | Seed and fold-assignment stability study of the ensemble | med | low | low | high | — |
| 8 | Explicit false-positive head trained on the 17 % transient-bearing no-break series | med | med | med | low | med |
| 9 | Cross-series conditioning: a series' rank *within the current timestep* as an input | med | high | med | med | high |
| 10 | Deploy the top-300 column subset (−0.0004 for a 40 % inference-cost cut) | — | low | low | low | — |
| 11 | Byte-level parity check of our TS-AUC against the live Crunch scorer | — | low | low | — | — |

**Closed by wave 6:** DGP-gated specialists (−0.0219 at full scale, rejected),
ensemble weighting and subset selection (both negative — use the plain average),
and the pairwise objective question (a dead heat; it is now a blend member).

**Closed by wave 5:** feature selection over the 500-column bank (no gain — the
bank is not redundant), the ensemble itself (built and confirmed on the lockbox),
and learned stacking (loses to the equal rank average).

## MOONSHOTS

1. **Neural likelihood-ratio estimation** between NOT-YET-BROKEN and ALREADY-BROKEN
   latent states, trained directly on the exact τ labels — the theoretically correct
   object for this metric, and nothing we built approximates it directly.
2. **Contrastive historical encoder**: pretrain a sequence encoder to embed a
   historical segment such that same-DGP windows are close, then use the embedding to
   *gate* online evidence. This is the H2 architecture the context experiments pointed
   at, done properly instead of as 50 raw columns.
3. **State-space / Mamba-style causal sequence model** on raw + engineered channels,
   trained with a within-timestep ranking loss, kept for ensemble diversity even if it
   loses standalone.

## CURRENT WINNING HYPOTHESIS

The 2026 problem is **weak-signal cross-sectional ranking under heterogeneous DGPs**,
not change-point detection. Three things carry the edge, in order:

1. **Per-series historical-null calibration at matched window length** — makes evidence
   comparable across series, which is the only thing TS-AUC rewards, and simultaneously
   removes the τ-encoding artifact.
2. **The shape and persistence of accumulated evidence** — peaks, decayed peaks, time
   since peak, persistence counts — because 17 % of no-break series contain transients
   that match a break in amplitude and differ only in duration.
3. **Diversity of monitored representations** — raw, several whitenings, PIT/rank,
   dependence, spectral, and an explicit absorbing-state posterior — because breaks here
   are scale- and dependence-dominant, individually tiny, and spread across almost every
   break series rather than concentrated in a detectable minority.

Everything that failed, failed for one of two reasons: it assumed a break type this
data does not contain (location, trend), or it assumed a series-level prior the data
does not support (historical context predicting break occurrence).
