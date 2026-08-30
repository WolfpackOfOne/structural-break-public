# PUBLIC IDEA MAP — external / public research
**ALPHA TEAM 16 · 2026-08-19 · 43 URLs successfully retrieved (~36 with substantive content), ~24 web searches**

Scope: web research only. No experiments run, nothing in `/home/claude/sb` touched except
reading `research/STATE_OF_RESEARCH.md`, `research/FAILED_EXPERIMENTS.md`, and writing this file.

---

## 0. SOURCING HEADLINE — READ THIS BEFORE THE IDEAS

Five things worth knowing before any of the numbered entries.

**(a) The 2025 winners' code is NOT public, except 2nd place.** The forum thread that is
literally titled "Winning Solutions of Structural Break Challenge"
(https://forum.crunchdao.com/t/winning-solutions-of-structural-break-challenge/1070)
contains *no* technical content — a user asks where the solutions are, staff member `enzo`
replies "Some of the winners shared their writeup on their solution" and points at the
leaderboard. The leaderboard page itself
(https://hub.crunchdao.com/competitions/structural-break/leaderboard) is a JS app and
returns no data to a fetcher. **I could not retrieve the leaderboard, the writeups it
allegedly hosts, or the results video.** The award-ceremony videos
(https://www.youtube.com/watch?v=LFWdWgAcOqU and .../watch?v=pMAGJHlImpE) exist but YouTube
returned HTTP 429 / robots blocks. The X/Twitter announcement
(https://x.com/adia_lab/status/1986732959710498829) is robots-disallowed.
The single richest public account of the winning solution is a **Medium post by a member of
the winning team**, which is what entries 1–5 below are built on.

**(b) The 2025 winner was Alphabot** (Brazilian prop-trading firm), per Humberto Brandão's
Medium post. Reported **90.14 % accuracy private / 90.65 % mean CV AUC public**, pushed to
**91.99 % AUC public** after absorbing three competitors' models post-hoc. **That is the
BATCH problem (known boundary, one label per series), not our metric.** Do not compare
0.90 AUC there to 0.61 TS-AUC here — they are different tasks. This is the single most
common way to misread the public record.

**(c) There is a hard public number for OUR task.** In
https://forum.crunchdao.com/t/leaderboard-comparability-after-the-june-8-real-time-data-access-fix-were-pre-fix-scores-rescored/1188
the poster states: *"none of the public post-fix baselines and write-ups I'm aware of exceed
~0.61, while the top of the leaderboard stands meaningfully higher."* Staff (`enzo`)
confirmed **all pre-June-8 predictions were invalidated** because the leak was inside
`infer()`. So: **~0.61 TS-AUC is the public state of the art we can verify**, and our
champion's 0.6252 dev / ~0.614 expected lockbox is at or slightly above it. The
"meaningfully higher" leaderboard tops are post-fix and unexplained.

**(d) Cross-series state is effectively BANNED by the determinism check.** Same forum, plus
https://forum.crunchdao.com/t/structural-break-real-time-may-infer-use-prior-completed-series-state/1186.
`enzo`: state persistence between series is mechanically possible but *"it isn't a good idea
because both parallelism support and the determinism check will work against you"* — the
runner splits 10,000 series across N workers, re-runs 10 % (1,000 series) across N workers,
and requires 1e-8 agreement. A model whose score at series *k* depends on series 1..k−1
cannot pass. He additionally warns: *"we will disqualify"* disk-based state tricks.
**This kills `NEXT EXPERIMENTS #9` (cross-series conditioning: a series' rank within the
current timestep as an input) as written.** Entry 27 below gives the only legal substitute.

**(e) The leak that was fixed was *series length*.** Competitors could recover `n_online`
during inference. Two consequences for us: (i) the ~0.61 ceiling is a *post-fix* number and
is honest; (ii) our decision never to emit `n_online` as a feature was not paranoia, it was
the exact exploit the organisers patched. Keep it that way.

**What I did NOT find, stated plainly so nobody re-searches it:**
- No public repo of *any* 2026 real-time solution. Zero. GitHub search is robots-blocked to
  fetchers, but every keyword search returned only 2025 batch repos.
- No writeup, blog, notebook or forum post describing a real-time feature set.
- No public TS-AUC leaderboard values other than the "~0.61" remark above.
- No "Inni Dynamics" anything. That team name returns nothing on the open web. Either it is
  not a public handle or it is spelled differently in public. **Unverified.**
- The Kaggle baseline notebook `crunchdao/structural-break-real-time-baseline` exists but
  Kaggle serves no content to fetchers; I could not read it.
- The 2025 leaderboard names beyond Alphabot (1st), `aParsecFromFuture` / Farukcan Sağlam
  (2nd, self-declared in repo description), and `abkimc` (52/490, self-declared).

---

## PART A — IDEAS DERIVED FROM PUBLIC COMPETITION SOLUTIONS

---

### 1. Alphabot's two-level stack of four *independently authored* feature blocks

**SOURCE** https://humbertobrandao.medium.com/how-far-can-we-push-the-winning-model-of-the-adia-lab-structural-break-challenge-87ebf3d0ff67 — Humberto Brandão (Alphabot, 1st place 2025).

**ORIGINAL METHOD** Eight level-0 models (XGBoost + RandomForest) over four feature blocks
built by four different people who did *not* coordinate: (i) Brandão — 58 features: local
t-tests, variance tests, Fligner tests, entropy differentials, CUSUM statistics,
Jensen–Shannon / Hellinger / Wasserstein divergences; (ii) Mário Augusto Filho —
statistical + autocorrelation + divergence descriptors recombined into non-linear composites
(log, tanh, trigonometric transforms of features); (iii) Rafael Alencar — wavelet denoising,
cumulative sums, percentage-change views, then distributional distances; (iv) João Pedro
Peinado — an integration layer of meta-features fusing predictions with statistical
descriptors. Level-1 meta-model merges level-0 predictions **plus raw statistical
descriptors**. 90.65 % CV AUC; +1.34 pp from adding three *outside* competitors' models.

**HYPOTHESIZED MECHANISM HERE** This is the same finding as our `RT-131`: independent
authorship is the cheapest source of decorrelation, and the gain from adding *foreign*
models (+1.34 pp) exceeded anything they got from tuning their own. Our 7-stream ensemble
gains +0.0102 and every stream added has paid. The unexploited half of their architecture is
that the meta-model sees **predictions AND raw descriptors together**, letting it learn
"trust stream A when the series looks like X". Our equal rank-average cannot do that — but
note our own LOFO stacking experiments (`RT-131`) say learned stacking loses here.

**CAUSAL 2026 TRANSLATION** Two separable pieces. (a) *More independently-authored streams*:
already our plan, trivially causal. (b) *Descriptor-conditioned blending*: at online index
`t`, blend stream ranks with weights that are a function of a small number of **online**
statistics (elapsed `t`, current calibrated evidence level), not of series-constant history.
Cost O(#streams) per point.

**ALREADY IMPLEMENTED?** (a) yes — `RT-131` is exactly this. (b) **no** — we only tested
*global* stacking, never *t-conditioned* blending.

**EXPERIMENT** Fit, leave-one-fold-out, a blend weight vector that is piecewise-constant in
elapsed-`t` bucket (0–20, 20–50, 50–100, 100–200, 200–400, 400+). Compare against the plain
equal rank average on the same OOF matrix. Falsifiable: if the honest LOFO number does not
beat 0.62524 by >0.001, close it permanently. Prior from `RT-131`: **likely negative**, but
this is a 6-parameter model rather than a 7-parameter one and `t` is the one covariate whose
effect on stream quality we have *measured* (the metric-by-elapsed-t table).

**FALSE SIGNAL** Fold-noise in the per-`t` slices; the 0–20 bucket has only 159 k rows and
TS-AUC there is 0.5247, so weights fitted there are close to unidentified.

**COST** cheap (post-hoc on stored OOF). **LEAKAGE RISK** low — `t` is legal, but do NOT let
the bucket boundaries be chosen by looking at the test score.

---

### 2. Divergence family: Jensen–Shannon / Hellinger / Wasserstein between pre- and post-segments

**SOURCE** Same Medium post (Brandão's 58-feature block); independently in
https://github.com/abkimc/ADIA-Lab-Structural-Break-Challenge-Solution (JS distance, KS) and
https://github.com/secabird/structural-break-challenge (energy statistic
`2*cross_dist − before_dist − after_dist`).

**ORIGINAL METHOD** Histogram/ECDF the "before" and "after" segments, compute JS, Hellinger,
Wasserstein-1, and the energy distance between them, feed as features.

**HYPOTHESIZED MECHANISM HERE** Our forensics puts **Wasserstein post-vs-pre AUC at 0.558**,
essentially tied with scale (0.559) and above dependence (0.538) — so this family is
measuring the second-strongest thing in the data. Wasserstein is scale-sensitive without
being *only* scale-sensitive, which matters because the breaks are scale-dominant but 92 %
of them are individually invisible.

**CAUSAL 2026 TRANSLATION** The historical segment is fixed and known at t=0, so its ECDF is
a **constant lookup table**. Precompute historical quantiles once (O(n_hist log n_hist), paid
once). Online, maintain a fixed-bin occupancy histogram of the online prefix against those
historical bin edges — that is O(1) per point (one bin increment). Wasserstein-1 between the
online empirical measure and the historical measure = Σ_bins |F_online(b) − F_hist(b)| · Δb,
recomputable in O(#bins) which is O(1) in `n`. Same trick gives JS and Hellinger from the
same occupancy vector. **Crucially, calibrate the resulting distance against the
distribution of the same distance computed over length-matched historical windows** —
otherwise it is an `n_post` detector, per our own generator-artifact finding.

**ALREADY IMPLEMENTED?** **Partly** — `m02_dist` has PIT / occupancy / divergence channels
(59 cols, +0.0131 on `m00_core`), and `RT-063-divall` measured the `(dv,oc,qd,ab,rs)_`
divergence slice at +0.01171. **Energy distance and Wasserstein-1-against-history are not
explicitly listed in our module inventory.** Worth confirming against the code.

**EXPERIMENT** Add exactly 3 columns — calibrated W1, calibrated energy distance, and their
running decayed peak — to `m02_dist` and screen. Falsifiable at the 0.005 level on 5 folds.

**FALSE SIGNAL** Heavy-tailed historical segments: a single outlier in the online prefix
moves W1 far more than it moves a rank statistic. Our own note that the online/historical
variance ratio is biased −0.063 in the heaviest-kurtosis quartile applies here with force.

**COST** cheap (O(1)/point after a one-off O(n_hist) setup). **LEAKAGE RISK** low if
calibration is length-matched; **high if not** — an uncalibrated W1 grows with `n_post` and
becomes a τ-detector.

---

### 3. Levene / Fligner–Killeen / Brown–Forsythe as *online-updatable* variance-equality tests

**SOURCE** Medium post (Brandão block: "variance tests, Fligner tests");
https://github.com/secabird/structural-break-challenge (Levene at ±7/±15/±30/±100 windows
around the boundary, Mood's median test, Fligner–Killeen);
https://github.com/gsoisson/adia-structural-break; and the 2nd-place repo
https://github.com/aParsecFromFuture/ADIA-Lab-Structural-Break-Challenge-Solution
(F-test, Levene, KS).

**ORIGINAL METHOD** Two-sample tests of equal spread on the absolute deviations from the
group centre (Levene uses the mean, Brown–Forsythe the median, Fligner the ranks of
|x−median|), computed at several window radii around the known boundary.

**HYPOTHESIZED MECHANISM HERE** **Scale is the dominant break type in this data (AUC 0.559,
the largest of any family).** Levene/Brown–Forsythe on |x − median| are *robust* scale tests
— they do not blow up on the heavy-tailed series where the plain variance ratio carries a
−0.063 bias. This is the well-sourced, unanimously-used public statistic that directly
targets our dominant break type, in a robust form.

**CAUSAL 2026 TRANSLATION** Everything needed is a running sum. Let `m_h` = historical
median (constant). Define `z_t = |x_t − m_h|`. Levene/BF is a one-way ANOVA F on `z`, so it
needs only Σz, Σz², and counts for the historical group (constants) and the online prefix
(two running scalars). **O(1) per point, two accumulators.** Fligner needs ranks of `z`
against the historical `z` distribution — also O(log n_hist) via binary search into a
precomputed sorted array, or O(1) with a fixed quantile grid. Emit both at the whole-prefix
scale and at trailing windows (w = 32/64/128), each calibrated against the length-matched
historical null the way `m00_core` already does.

**ALREADY IMPLEMENTED?** **Probably partly, but not by name.** `m00_core` monitors moments
at six trailing scales plus expanding, and `m02_dist` covers occupancy. But a *robust,
median-centred, rank-based* scale test on |x − median| is a different functional from a
moment ratio, and it is precisely the one that survives heavy tails. **Treat as "no" until
someone greps the code.**

**EXPERIMENT** 6 columns (BF-F and Fligner, each at expanding / w=64 / decayed-peak),
calibrated, screened on top of `m00_core`. Then the 5-fold promotion test. Predict +0.003 to
+0.008; the honest kill threshold is 0.0 on 5 folds.

**FALSE SIGNAL** Any transient volatility burst — and 17 % of no-break series contain one.
This is why the *decayed-peak / persistence* channel matters more than the level, exactly as
`m01_seq` found.

**COST** cheap. **LEAKAGE RISK** low.

---

### 4. Multi-representation transformation stack (z, cumsum, dense-rank, abs, MA, MSD)

**SOURCE** https://github.com/aParsecFromFuture/ADIA-Lab-Structural-Break-Challenge-Solution
— Farukcan Sağlam, self-declared 2nd place. 2,408 features generated by crossing six
transformations with a statistics bank, then SHAP-top-200/500 ∪ LightGBM-gain-top-500.

**ORIGINAL METHOD** Apply {z-score, cumulative sum, dense rank, absolute value, moving
average, moving standard deviation} to the raw series, then compute the *same* descriptive
statistics + hypothesis tests on every transformed version.

**HYPOTHESIZED MECHANISM HERE** This is a brute-force version of our own strongest structural
finding — *"monitoring several different whitenings simultaneously"* (`m04_resid`) and
*"diversity of monitored representations"* (winning hypothesis #3). Their **dense rank** is
our PIT channel; their **abs** is a scale channel; their **MSD** is a local-volatility
channel. The one they have that we do not obviously have is **cumsum**, whose statistics are
the Kolmogorov/Darling–Erdős family.

**CAUSAL 2026 TRANSLATION** All six are streaming-trivial: z from historical constants,
cumsum is one accumulator, dense-rank becomes "rank of x_t within the historical ECDF"
(binary search, O(log n_hist)), abs is pointwise, MA and MSD are ring-buffer O(1). The
*statistics* on top must be the causal ones we already use.

**ALREADY IMPLEMENTED?** **Mostly yes** — `m02_dist` (PIT ≈ dense rank), `m04_resid`
(whitenings), `m00_core` (moments at scales), `m03_dyn`. But note **`RT-063-rk` (rank CUSUM
on raw + AR-residual PIT) scored −0.00273** and was rejected as redundant with `m00_core`'s
expanding PIT channels. So the cumsum-of-PIT direction is *already closed*.

**EXPERIMENT** Do not re-run the family. The one open sub-question: cumsum **of the AR(6)
residual**, not of the raw series, since our strongest single statistic is AR(6)-residual
log-sd. 3 columns.

**FALSE SIGNAL** Cumsum statistics are dominated by low-frequency drift, and our trend family
(`RT-071C`) already scored −0.00487. Expect this to be near-dead.

**COST** cheap. **LEAKAGE RISK** low.

---

### 5. TabPFN-derived features as extra columns

**SOURCE** Same 2nd-place repo — "TabPFN Features: predictions from Tabular Prior-data Fitted
Networks", used as *inputs* to the final LightGBM, not as the final model.
Background: https://www.nature.com/articles/s41586-024-08328-6 (TabPFN, Nature 2025).

**ORIGINAL METHOD** Run TabPFN on the engineered feature table, take its predicted
probability, feed it as an additional feature to LightGBM alongside the raw features.

**HYPOTHESIZED MECHANISM HERE** TabPFN is an in-context Bayesian predictor over small tabular
tasks with a strong structural prior. On our problem it would be a *different inductive bias*
over the same 500 columns — precisely the axis on which our ensemble gains (+0.0102 from 7
streams, all LightGBM variants). A non-tree stream should decorrelate more than any of our
current seven do from each other.

**CAUSAL 2026 TRANSLATION** Causality is not the issue — the features are already causal, and
TabPFN just maps feature vector → probability. **The issue is inference cost.** TabPFN is a
transformer forward pass per row; we have ~10 M scoring rows and 2 CPU cores. This is only
viable if (a) we subsample training context hard, and (b) we distil TabPFN's OOF predictions
back into a LightGBM at inference time — i.e. use TabPFN purely as a *training-time teacher*.

**ALREADY IMPLEMENTED?** **no.** Listed in our own "PUBLIC METHODS NOT YET REPRODUCED".

**EXPERIMENT** Take the existing OOF feature matrix, subsample to ~10 k rows/fold, fit TabPFN
(v2, CPU) on the top-60 columns, score the OOF, add as an 8th stream to the rank average.
Falsifiable: does the 8-stream average beat 0.62524?

**FALSE SIGNAL** TabPFN's context-set choice is stochastic; a different subsample gives a
different score. Must fix the seed and check determinism to 1e-8 before ever shipping it.

**COST** **expensive** on 2 cores at inference; medium as a teacher-only. **LEAKAGE RISK**
medium-high — TabPFN sees the whole context set at once; if the context set contains rows
from the same *series* as the query row, that is within-series leakage across timesteps.
Context must be sampled at **series** granularity, matching our fold structure.

---

### 6. Variance-ratio *trajectory* over sliding windows, and the distance from the max-shift to the boundary

**SOURCE** https://github.com/secabird/structural-break-challenge — "Variance ratio dynamics
use sliding windows (size=100, step=50) to identify magnitude changes and **detect distance
between maximum variance shifts and the true boundary**".

**ORIGINAL METHOD** Slide a window across the whole series, compute the variance ratio
between adjacent windows, take the maximum, and record **how far that maximum is from the
declared boundary**. If the biggest variance jump is at the boundary, it is a break; if it is
elsewhere, it is DGP noise.

**HYPOTHESIZED MECHANISM HERE** **This is a localisation-quality feature and it is the
cleanest public statement of the transient-vs-break disambiguator we independently derived.**
In our setting there is no declared boundary, but there is an estimated τ̂ from `m06_loc`.
The analogue is: *how concentrated is the evidence around τ̂, and does the second-best
candidate change point come close to the best?* A genuine break gives a single dominant
candidate; a transient gives a spike that is dominant now and was not dominant 50 points ago.

**CAUSAL 2026 TRANSLATION** Maintain a dyadic grid of candidate change points (stride 32, cap
256 positions — the grid `m07_bayes` already uses). At each `t` track: (i) argmax candidate
τ̂_t, (ii) the gap between best and second-best score, (iii) **the stability of τ̂ over time**
— e.g. `|τ̂_t − τ̂_{t−64}|` and a running variance of τ̂. A real break pins τ̂; a transient
lets τ̂ wander. O(#candidates) per point on the existing grid, i.e. free if the grid exists.

**ALREADY IMPLEMENTED?** **Partly.** `m06_loc` computes τ̂ and post-segment statistics
(+0.0315) but our own notes say it *"was built by an agent that ran out of budget before
tuning it"* and NEXT-EXPERIMENTS #4 is "Localisation v2". **τ̂-stability-over-time and the
best-vs-second-best margin are not listed.** Treat as **no**.

**EXPERIMENT** Add to `m06_loc`: τ̂ drift over lags 32/128, running sd of τ̂, best−second
margin, and the fraction of the last 128 steps in which argmax stayed inside ±16 of the
current τ̂. 6 columns. Screen, then 5-fold.

**FALSE SIGNAL** τ̂ is mechanically pinned near `t` early in the online segment (few
candidates exist), so τ̂-stability is confounded with elapsed `t`. **Must be residualised
against `t` or the model will learn `t`, which it already has.**

**COST** cheap. **LEAKAGE RISK** low, but check prefix invariance bitwise — argmax over a
grid whose *size* depends on `n_online` is exactly the bug caught in `FAILED_EXPERIMENTS` N6.

---

### 7. Hurst exponent / DFA change, sample entropy, AR(5) coefficients, wavelet (db4)

**SOURCE** https://github.com/secabird/structural-break-challenge (Hurst, sample entropy,
AR(5), db4 wavelet, tsfresh selections); also the Alencar block in the Medium post (wavelet
denoising).

**ORIGINAL METHOD** Long-memory and complexity descriptors of before vs after.

**HYPOTHESIZED MECHANISM HERE** Dependence is our #3 break family (AUC 0.538) and `m03_dyn`
already captures a lot of it (+0.0300, 62.5 k gain/col). Note strongly: **our forensics found
AR(5)/AR(6) residual log-sd is the single strongest statistic in the whole project at 0.603,
and secabird independently landed on AR(5).** Two independent parties converging on order 5
is the most corroborated single number in this entire idea map.

**CAUSAL 2026 TRANSLATION** AR(5–6) coefficients are fitted **once on the historical segment**
(constant), then the residual is filtered forward causally. Everything downstream — residual
log-sd, residual ACF, residual ACF-of-squares — is a running sum. O(p) per point.

**ALREADY IMPLEMENTED?** `m04_resid` and `m07_bayes` fit their own AR filters; the **shared
context AR order is 2**. NEXT-EXPERIMENTS #3 is exactly "raise the shared context AR order
from 2 to 5–6". **This public source is independent corroboration that #3 is the right call.**

**EXPERIMENT** Already ranked #3 in our own plan. Run it. Falsifiable at 5 folds against the
0.61510 single-model control.

**FALSE SIGNAL** A higher-order AR fitted on a *short* historical segment (n_hist can be as
low as 1,000) over-fits and whitens away real signal — the over-whitening failure that killed
GARCH at 0.50012. Guard: compare AR(6) fitted on the first half of history and validated on
the second.

**COST** cheap. **LEAKAGE RISK** low (coefficients from history only, never updated).

---

### 8. Nested CV + Optuna + multi-seed averaging + logit-stacking (gsoisson pipeline)

**SOURCE** https://github.com/gsoisson/adia-structural-break.

**ORIGINAL METHOD** Robust standardisation by historical median/MAD, winsorisation at
historical thresholds, then ~200 features across blocks (quantile, ACF/PACF, hypothesis tests,
FFT bandpower, Δ/Δ², boundary-local, rolling, AR), each block computed on **five parallel
streams** (z, Δz, Δ²z, |z|, z²). Collinearity pruning at |ρ|>0.98, univariate-AUC filter.
Then XGB/LGBM/CatBoost × many seeds, nested CV with Optuna inner loop, meta-features = mean
and **sd** of base-learner logits, XGB meta-learner.

**HYPOTHESIZED MECHANISM HERE** Two transferable pieces. (a) **Winsorisation at historical
thresholds** — a cheap, causal, heavy-tail defence that directly attacks our measured
kurtosis-driven variance-ratio bias. (b) **The sd of base-learner logits as a meta-feature** —
disagreement among streams is itself informative, and we have never used it.

**CAUSAL 2026 TRANSLATION** (a) Clip `x_t` to historical q0.005/q0.995 before every scale
statistic — O(1), fully causal. Emit both clipped and unclipped versions so the model can
learn from the *difference* (a large gap = the evidence is one outlier). (b) At each `t`,
compute mean and sd of the seven streams' within-timestep rank percentiles, and use the sd as
an input to a tiny second-stage model — or, more honestly given `RT-131`, just check whether
sd correlates with error.

**ALREADY IMPLEMENTED?** (a) **unclear/likely no** as an explicit dual clipped/unclipped
channel. (b) **no.**

**EXPERIMENT** (a) 4 columns: clipped-variance z, clipped-minus-unclipped variance z, at
expanding and w=128. (b) post-hoc on stored OOF: does adding `sd_of_stream_ranks` to a logistic
on `mean_rank` beat the plain mean under LOFO? Our `RT-131` result says learned stacking
loses, so the honest prior is **negative**, but sd is one column and costs nothing to test.

**FALSE SIGNAL** (a) On light-tailed series clipping is a no-op, so the feature is a tail-index
proxy — i.e. a series-constant DGP fingerprint, which is exactly what `m05_ctx` proved is
harmful. **Emit only the online-vs-historical clipping-rate *difference*, never the level.**

**COST** cheap. **LEAKAGE RISK** (a) medium — clipping rate is close to a series-constant
signature; run the within-fold derangement control from the `m05_ctx` post-mortem.

---

### 9. Nonlinear composite features (log / tanh / trigonometric recombinations)

**SOURCE** Medium post, Mário Augusto Filho's block: "combining them through iterative
validation using non-linear composite features (log, tanh, trigonometric transformations)".

**ORIGINAL METHOD** Hand-built nonlinear recombinations of pairs of descriptors, kept or
dropped by iterative validation.

**HYPOTHESIZED MECHANISM HERE** **I think this transfers poorly and should be low priority.**
LightGBM is invariant to monotone transforms of a single feature, so `log` and `tanh` of one
column are no-ops for a tree. The only content is in *interactions* (ratios, products), which
trees approximate with depth. Our 500-column bank at 63 leaves already has the capacity.

**CAUSAL 2026 TRANSLATION** n/a — trivially causal, just pointwise maps.

**ALREADY IMPLEMENTED?** n/a.

**EXPERIMENT** Only worth one narrow test: explicit **ratios** between the strongest
cross-module pairs (e.g. `m07_bayes::ab_fast / m00_core::w8_sur_tail_x`), 10 columns.
Falsifiable, near-zero prior.

**FALSE SIGNAL** Ratios explode when the denominator is near zero; the unfloored-null-spread
bug (`FAILED_EXPERIMENTS` N7) is the exact failure mode.

**COST** cheap. **LEAKAGE RISK** low.

---

### 10. Boundary-local windows at several radii (±7, ±15, ±30, ±100)

**SOURCE** https://github.com/secabird/structural-break-challenge.

**ORIGINAL METHOD** Compute variance differences and Levene statistics in symmetric windows
of several radii around the known boundary.

**HYPOTHESIZED MECHANISM HERE** In 2026 there is no known boundary, but there IS a known
boundary at t=0: **the historical→online seam**. A break at τ=0 (or very early) is exactly the
regime where our champion is weakest (elapsed 0–20 → TS-AUC 0.5247, barely above chance) and
where the metric still assigns 159 k rows. Also relevant: the W23 changelog fixed *"29 bad
timeseries that reported no structural breaks when one was actually present in the very first
step"* — so **breaks at step 1 exist and the organisers care about them.**

**CAUSAL 2026 TRANSLATION** Asymmetric by necessity: at small `t`, compare the online prefix
[0..t] against the *last* w historical points for w ∈ {8,16,32,64}, rather than against the
whole history. The last-w-historical statistics are constants computed at t=0. This gives a
high-power, low-latency seam test that does not wait for the expanding window to accumulate.
O(1) per point.

**ALREADY IMPLEMENTED?** **Partly** — `m05_ctx` had `recent-100/250/500 vs whole-history`
columns, but `m05_ctx` was rejected wholesale (−0.0177 at full scale) because it was
*series-constant*. **The seam test proposed here is NOT series-constant** — it is
online-prefix vs recent-history, which varies with `t`. Distinct object. Treat as **no**.

**EXPERIMENT** 8 columns: robust-z of (online prefix mean, |dev| mean) against the
distribution of the same statistic over length-matched windows drawn from the **last 500**
historical points, at w=8/16/32/64. Score restricted to elapsed t<50 to see if the early
regime moves at all. Falsifiable: does TS-AUC in the 0–20 and 20–50 buckets rise?

**FALSE SIGNAL** The last-500 historical window is itself a small sample; its moments are
noisy, and the noise is series-constant → a DGP fingerprint again. Use the *matched-length
null* to normalise, and run the derangement control.

**COST** cheap. **LEAKAGE RISK** medium (series-constant contamination). Run the `m05_ctx`
derangement control before believing any positive.

---

### 11. The winners' own reported gain from absorbing *foreign* models (+1.34 pp AUC)

**SOURCE** Medium post: adding Farukcan Sağlam's, Julian Mukaj's and the DataTech Team's
models to Alphabot's stack took public AUC from 90.65 % → 91.99 %.

**HYPOTHESIZED MECHANISM HERE** The largest single post-hoc gain reported anywhere in the
public record for this problem came from **model diversity sourced from strangers**, not from
tuning. We cannot absorb strangers' models, but we can absorb the *public* 2025 method
families we have not reproduced: the 2,408-feature transformation cross-product (entry 4),
the TabPFN stream (entry 5), and the 200-feature 5-stream block design (entry 8), each as an
independent stream rather than as columns merged into the champion.

**CAUSAL 2026 TRANSLATION** Build each as a *separate LightGBM stream* on a *disjoint or
near-disjoint* column subset, then rank-average. This is precisely our `RT-131` rule and
requires no new machinery.

**ALREADY IMPLEMENTED?** The mechanism yes; these specific streams **no**.

**EXPERIMENT** Ranked #2 in our own NEXT EXPERIMENTS ("more streams"). The public record
raises its expected value: every source that reports ensemble deltas reports them as the
largest available gain.

**FALSE SIGNAL** Streams that differ only in seed give a smaller gain than streams that differ
in features; our own within-timestep correlations (0.39–0.78) are the right diagnostic, and
the global correlation (0.60–0.93) is the wrong one.

**COST** medium (one training run per stream). **LEAKAGE RISK** low.

---

### 12. Deliberately NOT persisting state across series (organiser-confirmed constraint)

**SOURCE** https://forum.crunchdao.com/t/structural-break-real-time-may-infer-use-prior-completed-series-state/1186 and
https://forum.crunchdao.com/t/leaderboard-comparability-after-the-june-8-real-time-data-access-fix-were-pre-fix-scores-rescored/1188 — staff `enzo`.

**ORIGINAL METHOD** n/a — this is a rule, not a method. Verbatim: *"trying to persist state
across time series will likely result in your code not being deterministic"*; the runner
splits 10,000 series across N workers and re-checks 10 % (1,000 series) across N workers at
1e-8; disk-based state *"we will disqualify"*.

**HYPOTHESIZED MECHANISM HERE** Directly deletes NEXT-EXPERIMENTS #9 ("cross-series
conditioning: a series' rank within the current timestep as an input"). Any rank-within-
timestep requires the other series, which requires either cross-series state or a second
pass. **Both are illegal.**

**CAUSAL 2026 TRANSLATION** The legal substitute is a **frozen cross-sectional calibrator
learned at TRAINING time**: for each elapsed index `t`, estimate on the training folds the
CDF of the model's raw score among alive series at that `t`, store it as a lookup table
(t-bucket × score-quantile), and at inference map raw score → its training-time percentile at
that `t`. This is a pure function of (t, score) — deterministic, per-series, O(1), no
cross-series state at inference.

**ALREADY IMPLEMENTED?** **no.**

**EXPERIMENT** Post-hoc on stored OOF: build the (t-bucket, score)→percentile table from four
folds, apply to the fifth, score. **Note the honest prior: TS-AUC is invariant to any monotone
transform applied identically within a timestep, so a table indexed only by `t` should be
EXACTLY neutral.** It becomes non-neutral only if the table is indexed by something else too —
which is the real experiment: does mapping through a per-(t, historical-kurtosis-quartile)
table help? That is no longer monotone-within-timestep and could genuinely move the metric.

**FALSE SIGNAL** Any measured gain from a pure `t`-indexed table is a bug in the scorer or in
the tie-handling, not a result. Use it as a **unit test**.

**COST** cheap. **LEAKAGE RISK** low, but the table must be fitted on training folds only.

---

### 13. Streaming-protocol conformance as an engineering risk (organiser-documented)

**SOURCE** https://docs.crunchdao.com/competitions/competitions/structural-break-real-time —
the `infer(datasets, model_directory_path)` generator signature, `model.consume(point)`,
"online segment cannot be read twice", "must yield a result before receiving the next point",
determinism 1e-8 on a 10 % re-run, 15 h/week budget, `INFER_PARALLELISM` process-level only,
"no file I/O recommended".

**HYPOTHESIZED MECHANISM HERE** Our own STATE_OF_RESEARCH names this the largest outstanding
risk: modules are batch-first, computing whole trajectories from cumulative sums. The docs
confirm the exact shape of the required port and — importantly — that **process-level
parallelism is available**, one series per process, which means our 0.25 s/series feature
build can be divided by the worker count. At 10,000 series × 0.25 s = ~42 min single-threaded,
we are comfortably inside 15 h even before parallelism.

**CAUSAL 2026 TRANSLATION** n/a — this is the deployment spec.

**ALREADY IMPLEMENTED?** **no** (explicitly: "has not been written or parity-tested").

**EXPERIMENT** Port, then assert bitwise equality (atol=0.0) between batch features and
streaming features on ≥100 series at every prefix. Then assert 1e-8 determinism across two
runs with different `INFER_PARALLELISM`. **This is a blocker, not an idea.**

**FALSE SIGNAL** n/a. **COST** high effort, low compute. **LEAKAGE RISK** — the port is where
leakage gets *introduced*: any statistic that touches `len(x_online)` is the exact exploit the
organisers patched on June 8.

---

## PART B — ACADEMIC IDEAS

---

### 14. e-detectors: the O(1) Shiryaev–Roberts / CUSUM recursion over e-processes

**SOURCE** https://arxiv.org/abs/2203.03532 / https://arxiv.org/pdf/2203.03532 — Shin,
Ramdas, Rinaldo, *E-detectors: a nonparametric framework for sequential change detection*.

**ORIGINAL METHOD** Build e^j-processes Λ^(j) started at each candidate change time j, with
`sup_P E[Λ^(j)_τ | F_{j−1}] ≤ 1`. Aggregate as **SR: M_n = Σ_j Λ^(j)_n** or
**CUSUM: M_n = max_j Λ^(j)_n**. When the increments are multiplicative,
`Λ^(j)_n = ∏_{i=j..n} L_i`, both collapse to **O(1) recursions**:
```
M^SR_n = L_n · (M^SR_{n-1} + 1)
M^CU_n = L_n · max{M^CU_{n-1}, 1}
```
with exponential baseline increments `L^λ_n = exp{λ·s(X_n) − ψ(λ)·v(X_n)}`. For unknown
post-change magnitude, mix K components with weights ω_k (paper's Algorithm 1 picks K≈69),
costing O(K)/step; with unknown bounds, an adaptive schedule `K(n) = K_L + ⌈m log_η n⌉` keeps
it O(m log n). Declaring at `M_n ≥ 1/α` gives ARL ≥ 1/α (Ville).

**HYPOTHESIZED MECHANISM HERE** This is the theoretically correct object for our problem and
it is *cheap*. Critically for us: **choosing `v(X_n) = (X_n − m)²` makes the baseline
increment automatically adapt to unknown variance without estimating it** — i.e. an e-detector
that is natively a scale detector, which is our dominant break family. And the score is
already a *calibrated* likelihood-ratio-like quantity, which is what makes evidence comparable
across series — the one thing our project says the metric rewards.

**CAUSAL 2026 TRANSLATION** Two accumulators per (λ, channel). Channels: raw z, |z|, z²,
AR(6)-residual, AR(6)-residual². λ-grid: 8–16 log-spaced values. Total ~80 scalars, all O(1).
Emit `log M^SR`, `log M^CU`, their decayed peaks, time-since-peak, and the **argmax j** from
the CUSUM version (free localisation, feeds entry 6).

**ALREADY IMPLEMENTED?** **Partly.** `m07_bayes` contains "e-processes" and is our best module
by gain-per-column (+0.0397, 124.8 k gain/col). But the agent's own post-mortem says: *"I have
no per-column or per-block attribution for the 0.599 ... So I cannot tell you whether the
absorbing posterior, BOCPD, the e-processes or simply the AR(6)+normal-scores stream is doing
the work."* **We do not know whether the e-process part works.** And the *variance-adaptive
`v(X)=(X−m)²` choice* is a specific construction that may or may not be what was built.

**EXPERIMENT** The ablation our own agent 12 asked for and never ran: four runs dropping one
`m07_bayes` block at a time, plus one run of the AR(6)+normal-scores stream through
`m00_core`-style calibration with **none** of the Bayesian machinery. If the last recovers
most of 0.599, **the finding is the stream, not the recursions, and it transfers to every
module for free.** This is the single highest-information cheap experiment in this document.

**FALSE SIGNAL** e-processes grow monotonically under any persistent mis-specification of the
"null", including a historical segment whose tail index differs from the online segment's by
chance. Per-series calibration against the historical null is mandatory.

**COST** cheap (O(K)/point, K≈16–80 scalars). **LEAKAGE RISK** low — the recursion is
manifestly prefix-measurable, which makes it one of the safest families to port to streaming.

---

### 15. e-processes for testing the VARIANCE specifically (E-GREE / e-mixture betting)

**SOURCE** https://academic.oup.com/biomet/article/112/1/asae049/7796539 — *Testing the mean
and variance by e-processes*, Biometrika 112(1), 2025.

**ORIGINAL METHOD** `M_t = ∏_{i=1..t} (1 − λ_i + λ_i E_i)` with per-observation e-variable
`E_0 = (X − μ)²_+ / σ²` for the composite null {mean ≤ μ, variance ≤ σ²}. Two ways to pick λ:
**e-mixture** (average over a fixed λ grid, e.g. {0.01,…,0.20}) and **E-GREE** (adaptive,
`λ_i = [Σ(E_j−1) / Σ(E_j−1)²]_+ ∧ ½` — a closed-form growth-rate-optimal step). Reject at
`M_t ≥ 1/α`.

**HYPOTHESIZED MECHANISM HERE** This is entry 14 specialised to exactly our dominant break
type, with a **composite null over the variance** — which matters because our per-series
historical σ is estimated, not known, and a test that must assume σ exactly will mis-rank the
cross-section. E-GREE's λ update is one running ratio of two accumulators: **O(1), no grid,
no tuning.**

**CAUSAL 2026 TRANSLATION** Set μ, σ from the historical segment (they are exactly 0 and 1 by
construction — the organisers z-score each series' history). Then `E_i = (x_i)²_+ / σ_h²` and
`M_t` updates multiplicatively. Keep two accumulators for E-GREE's λ. Run one instance
per direction (variance-up: `(x)²_+`; variance-down: an analogous e-variable) and one per
channel (raw, AR-residual). ~8 scalars total.

**ALREADY IMPLEMENTED?** **Unknown / probably no** — `m07_bayes` says "e-processes" without
specifying construction, and E-GREE is a 2025 Biometrika result. **Treat as no.**

**EXPERIMENT** 6 columns (log M for variance-up and variance-down on raw and AR(6)-residual,
plus decayed peaks), screened on `m00_core`, then 5-fold. Kill threshold 0.0 at 5 folds.

**FALSE SIGNAL** The historical segment is standardised to sd exactly 1.00000, so σ_h² = 1 is
*too* exact — any systematic online-vs-historical variance bias (we measured median −0.004 to
−0.063 by kurtosis quartile) is fed straight into `M_t` and compounds multiplicatively over
`t`. **This will produce a τ/`t`-encoding artifact unless the e-process is calibrated against
its own historical-null trajectory at matched length.**

**COST** cheap. **LEAKAGE RISK** low mechanically, **medium statistically** (the bias above).

---

### 16. AR(p)-FOCuS — exact GLR over ALL change points, for autocorrelated data, at O(log n)/step

**SOURCE** https://arxiv.org/abs/2607.16106 (*An Efficient Likelihood Ratio Test for Online
Changepoint Detection in the Presence of Autocorrelation*) building on
https://arxiv.org/abs/2110.08205 / https://www.jmlr.org/papers/v24/21-1230.html (Romano,
Eckley, Fearnhead, Rigaill, *Fast Online Changepoint Detection via Functional Pruning CUSUM
Statistics*, JMLR 24, 2023).

**ORIGINAL METHOD** FOCuS maintains the CUSUM likelihood-ratio as a *function* of the unknown
post-change parameter — a set of piecewise quadratics indexed by candidate change point —
and prunes candidates that can never be optimal. Result: *"equivalent to running these earlier
methods simultaneously for all sizes of window, or all possible values for the size of
change"*, at **cost logarithmic in the number of observations per iteration**. AR(p)-FOCuS
extends the GLR to AR(p) processes and keeps O(log n)/step; the paper reports *"greater
detection power than IID-based tests when the underlying data exhibit temporal correlation"*.

**HYPOTHESIZED MECHANISM HERE** This is the single best-matched algorithm I found. Our data is
(i) autocorrelated, (ii) breaks in scale and dependence, (iii) requires *all* window sizes
because τ is near-uniform on [0, n_online] and n_online spans 10–999, and (iv) is compute-
constrained. FOCuS is exactly "max-GLR over all τ and all magnitudes, cheaply and exactly".
Our `m01_seq` computes max-GLR over a **dyadic grid** — FOCuS computes it over **every**
candidate, exactly, for the same cost. And `glz_pk` / `gle_pkr` (peak GLR channels) are
already two of our top-10 features by gain, so the family is proven; FOCuS is a strictly
better estimator of the same quantity.

**CAUSAL 2026 TRANSLATION** Native — FOCuS *is* an online algorithm. Run it on the AR(6)
residual and on the squared AR(6) residual (the latter turns a variance break into a mean
break in the squares, which is what FOCuS detects optimally). Emit: current max-GLR, its
decayed peak, time-since-peak, the argmax τ̂, and the number of surviving candidates (a
cheap proxy for evidence concentration → feeds entry 6). Reference implementation exists
(`changepoint-online` on PyPI, https://pypi.org/project/changepoint-online/) — read it, do
not necessarily import it, since determinism and inference cost must be controlled.

**ALREADY IMPLEMENTED?** **no.** Our GLR is grid-based (`m01_seq::glz_pk`, `gle_pkr`) and our
`glz_mix` mixture variant was measured as dominated (0.008 % gain vs 3.56 %).

**EXPERIMENT** Replace the dyadic-grid max-GLR in `m01_seq` with exact FOCuS on the AR(6)
residual and on its square. Paired, same folds, same seeds. Two falsifiable claims:
(1) exact-max ≥ grid-max on TS-AUC; (2) the τ̂ from FOCuS improves `m06_loc`.
**Prior: this is the highest-expected-value unimplemented idea in this document.**

**FALSE SIGNAL** FOCuS on the *raw* series detects mean changes, which do not exist here
(AUC 0.4998) — running it on raw would be a null result that says nothing about the method.
**It must be run on squares / residual-squares.** Also, the candidate-set size grows with `t`,
so "number of surviving candidates" is confounded with `t`; residualise.

**COST** cheap (O(log n)/point; the JMLR paper's whole selling point is high-frequency
streams on limited hardware). **LEAKAGE RISK** low — the pruning is causal by construction.
But re-verify prefix invariance bitwise; pruning bugs are subtle.

---

### 17. NP-FOCuS — nonparametric sequential LR on empirical-CDF counts

**SOURCE** https://arxiv.org/abs/2302.02718 — *A Log-Linear Non-Parametric Online Changepoint
Detection Algorithm based on Functional Pruning*.

**ORIGINAL METHOD** *"A sequential likelihood ratio test for a change in a set of points of
the empirical cumulative density function"*, tracking *"the number of observations above or
below those points"*, at log-linear total cost. Reported to outperform existing nonparametric
online CPD methods.

**HYPOTHESIZED MECHANISM HERE** This is FOCuS made distribution-free by reducing each
observation to a vector of indicator counts at Q historical quantiles. That is **exactly our
`m00_core` occupancy channel, but with an exact max-GLR over all change points on top of it**
instead of a windowed z. And because the quantiles come from the historical ECDF, the
statistic is per-series calibrated *by construction* — the property our project identifies as
the single most reliable source of improvement.

**CAUSAL 2026 TRANSLATION** Pick Q ∈ {5, 9} historical quantiles (e.g. 5/25/50/75/95 %).
Each new point contributes a Q-vector of Bernoulli indicators. Run one FOCuS-style Bernoulli
GLR per quantile, aggregate by max or by sum-of-logs. O(Q log n)/point.

**ALREADY IMPLEMENTED?** **no.** We have occupancy *levels* calibrated by window
(`m00_core`), and rank-CUSUM was tried and rejected (`RT-063-rk`, −0.00273) — but rank-CUSUM
is a *fixed-window* statistic, whereas this maximises over all change points. Different object.

**EXPERIMENT** 5 columns (max over quantiles of the Bernoulli max-GLR, its decayed peak,
time-since-peak, argmax τ̂, and the tail-vs-centre contrast of which quantile fired). Screen,
then 5-fold. **Note the standing warning: `RT-063-rk` failed because it re-stated information
`m00_core` already had. The novelty claim here rests entirely on the max-over-τ, so if it
fails, that is a clean falsification of "exact max-GLR beats windowed z on occupancy".**

**FALSE SIGNAL** With Q=5 and a short online prefix, counts are tiny and the Bernoulli GLR is
dominated by discreteness; expect noise for t < 30.

**COST** cheap. **LEAKAGE RISK** low.

---

### 18. Backward CUSUM / stacked backward CUSUM — power for LATE breaks

**SOURCE** https://arxiv.org/pdf/2003.02682 — Otto & Breitung, *Backward CUSUM for Testing and
Monitoring Structural Change*, Econometric Theory.

**ORIGINAL METHOD** `BQ_{t,T} = Q_T(1) − Q_T((t−1)/T)` — cumulate recursive residuals
**backwards from the present**. Rationale (verbatim): pre-break residuals carry *"no useful
information about a subsequent break"*, so backward cumulation *"collects the informative
residuals first"*. The *stacked* backward CUSUM evaluates backward CUSUMs at every time point,
forming a triangular array, and is the monitoring (real-time) version. Boundary
`d_lin(r) = 1 + 2r`; infinite-horizon `d_inf(r,s) = √(r(1+2(r−s)))`. Reported: backward beats
forward for any break after 15 % of the sample, and in monitoring, *"the stacked backward
CUSUM maintains nearly constant detection delays across break locations, while forward
CUSUM's delays increase substantially with later breaks"*.

**HYPOTHESIZED MECHANISM HERE** **This targets a weakness we have measured precisely.** Our
champion scores **0.628/0.624 on early-τ quartiles but 0.588/0.571 on late-τ**, and our own
diagnosis is "a late break simply has fewer post-break observations". Backward CUSUM's whole
claim is that *forward* accumulation dilutes late-break evidence with a long pre-break prefix,
and that reversing the accumulation restores near-constant delay. If that claim holds on our
data, it is worth roughly the late-quartile gap.

**CAUSAL 2026 TRANSLATION** The stacked backward CUSUM at time t is
`max_{s≤t} |S_t − S_s| / boundary(t,s)` where `S` is the cumulative sum of calibrated
residuals — i.e. a *self-normalised max over suffixes*. Maintain the running max via the
standard Page recursion (O(1)) for a fixed boundary; for the full boundary-scaled version use
the same dyadic candidate grid `m07_bayes` already carries (O(#candidates), ~256). Run on
squared AR(6) residuals (scale breaks) and on the |z| stream.

**ALREADY IMPLEMENTED?** **no.** Our `m01_seq` bank is CUSUM/GLR/SR/Page-Hinkley — all
*forward*. Nothing backward.

**EXPERIMENT** 6 columns of stacked-backward CUSUM (on z², AR-residual², |z|; level and
decayed peak), calibrated. **Score the late-τ quartile separately** — this idea makes a
specific directional prediction (gain concentrated in late τ and in the 400–1000 elapsed
bucket) and should be judged on that, not only on the pooled number.

**FALSE SIGNAL** Backward CUSUM is maximally sensitive to the most recent points, which is
exactly what a transient looks like. Expect a *worse* false-positive profile on the 17 % of
no-break series with transients. **Pair it with the persistence channels or it will hurt.**

**COST** cheap-to-medium (O(1) with a fixed boundary; O(256) with the grid). **LEAKAGE RISK**
low, but the boundary function must not depend on the (unknown) horizon `T` — use the
**infinite-horizon** `d_inf`, since `n_online` is unknowable and using it is the patched
exploit.

---

### 19. NEWMA — difference of two EWMAs of random Fourier features

**SOURCE** https://arxiv.org/pdf/1805.08061 — Keriven, Garreau, Poli, *NEWMA: a new method for
scalable model-free online change-point detection*. Code: https://github.com/lightonai/newma.

**ORIGINAL METHOD** Two EWMAs of a feature map with different forgetting factors:
`z_t = (1−Λ)z_{t−1} + ΛΨ(x_t)`, `z'_t = (1−λ)z'_{t−1} + λΨ(x_t)`, λ < Λ. Flag when
`‖z_t − z'_t‖` crosses a threshold. Proposition 1: the difference equals a weighted
recent-vs-older comparison, so it emulates a sliding-window two-sample test **without storing
raw data**. Ψ = random Fourier features approximating an MMD kernel. Adaptive threshold
`τ²_t = μ_t + a·σ_t` from running estimates of the statistic's own mean and sd.

**HYPOTHESIZED MECHANISM HERE** MMD with a Gaussian kernel is sensitive to *any* distributional
change including scale and shape, not just mean — which is what we need. And the O(m) memory
/ O(m) per-step cost with **no raw-data storage** is a perfect fit for the streaming API.
The adaptive self-normalising threshold is a per-series calibration that comes for free.

**CAUSAL 2026 TRANSLATION** Draw m ≈ 32–64 random frequencies with a **fixed seed** (mandatory
for the 1e-8 determinism check). Ψ(x) = [cos(ω_j x), sin(ω_j x)]/√m — for univariate x this is
cheap. **Initialise both EWMAs from the historical segment** (run them over the history
first), so `‖z−z'‖` starts at its historical null level and the online drift is the signal.
Bandwidth from the historical median heuristic. Emit ‖z−z'‖, its calibrated z against the
historical trajectory of the same quantity, decayed peak, time-since-peak. Try 2–3 (λ, Λ)
pairs for multiscale.

**ALREADY IMPLEMENTED?** **no.** `m01_seq` has an EWMA bank (which was mostly dead —
`ew8_z` 0.0011 %, `ew32_z` 0.0022 % gain), but that bank is EWMAs of the *level*, not of a
kernel feature map, and it is a single EWMA not a difference of two. **The failure of the
level-EWMA bank does not falsify this**; the post-mortem's own diagnosis was that the
rectangular windows in `m00_core` dominated the *level* channel, which says nothing about a
kernel-MMD channel.

**EXPERIMENT** 8 columns from 2 (λ,Λ) pairs × {level-z, decayed peak, time-since-peak,
peak-ratio}. Screen on `m00_core`, then 5-fold. Kill at 0.0.

**FALSE SIGNAL** RFF bandwidth is estimated from the historical segment → a series-constant
quantity leaks in through the back door. Fix bandwidth from the *global* training median, not
per-series, or emit only the calibrated version. Also: heavy-tailed series produce large-|x|
points that saturate cos/sin and look like changes.

**COST** cheap (O(m)=O(64) per point; no raw storage). **LEAKAGE RISK** medium — the RFF seed
and the bandwidth choice must be frozen at training time or determinism fails.

---

### 20. Scan-B / online kernel CUSUM — MMD against reference blocks, maximised over block size

**SOURCE** https://arxiv.org/pdf/2211.15070 — Wei & Xie, *Online Kernel CUSUM for Change-Point
Detection*; and Li, Xie, Song, Chen, *Scan B-statistic for kernel change-point detection*,
Sequential Analysis 38(4).

**ORIGINAL METHOD** Scan-B: `Z_B(t) = D̂_B(t) / √Var_∞(D̂_B(t))`, a normalised MMD between the
last B observations and N reference blocks of size B; cost O(NB²)/step, memory O(NB). Online
kernel CUSUM improves it by **maximising over block size**:
`T_w = inf{t : max_{B ∈ [2, min(w,t)]} Z_B(t) ≥ b}`, cost O(Nw²)/step with recursive Gram
updates; thresholds from analytic ARL approximations using 1st and 3rd moments.

**HYPOTHESIZED MECHANISM HERE** The **reference blocks are free for us**: the historical
segment is guaranteed break-free, so we can draw N reference blocks from it once and reuse
them for the whole online pass. That converts a normally expensive method into a fixed-cost
one. Maximising over block size B is the same "all window sizes at once" idea as FOCuS but in
a kernel/MMD metric that is scale- and shape-sensitive rather than mean-sensitive.

**CAUSAL 2026 TRANSLATION** Precompute at t=0: N=5 reference blocks of size B_max=64 from the
historical segment, and their Gram matrices, and the null mean/variance of `Z_B` at each B
(this *is* the per-series historical-null calibration, obtained for free from the same
blocks). Online: maintain a ring buffer of the last B_max points; per step, incremental Gram
updates give `Z_B(t)` for B ∈ {8,16,32,64}. Cost ~O(N·B_max) = O(320) per point with
incremental updates — the honest ceiling; **this is the most expensive idea here that is
still plausibly affordable.**

**ALREADY IMPLEMENTED?** **no.** No kernel/MMD channel exists anywhere in the module list.

**EXPERIMENT** 8 columns (max-over-B `Z`, argmax B, decayed peak, time-since-peak, ×2 kernels).
Screen. If it wins, profile before promoting — the cost is real.

**FALSE SIGNAL** MMD with a median-heuristic bandwidth is dominated by scale changes, which is
good, but it means it partially re-measures `m00_core`'s variance channels. The marginal test
on top of `m00_core` is the only meaningful one.

**COST** **medium-to-expensive** (~320 kernel evals/point). Compare to our current 0.25 s/series
total budget — this could double it. Consider NEWMA (entry 19) as the cheap substitute; NEWMA
is explicitly designed as the O(m) approximation to exactly this.

**LEAKAGE RISK** low.

---

### 21. Conformal test martingales (Vovk) — calibrated betting on historical-ECDF p-values

**SOURCE** https://arxiv.org/pdf/2102.10439 / https://proceedings.mlr.press/v152/vovk21b.html
— Vovk, Petej, Nouretdinov, Ahlberg, Carlsson, Gammerman, *Retrain or not retrain: conformal
test martingales for change-point detection*.

**ORIGINAL METHOD** Conformity score α_i per observation; smoothed conformal p-value
`p_n = (|{i: α_i < α_n}| + θ_n|{i: α_i = α_n}|)/n` with θ_n ~ U[0,1]; under exchangeability
the p-values are i.i.d. U[0,1]. Then a betting martingale `S_n = ∏ f(p_i)` with the **Simple
Jumper** betting strategy: three channels ε ∈ {−1,0,1}, `f_ε(p) = 1 + (p − 0.5)ε`, equal
initial capital, stay in state with prob 1−J (J=0.01), jump uniformly otherwise, combine
multiplicatively. Ville's inequality gives the threshold. Empirically: **signed** errors
detect faster than **unsigned** (median delay 55 vs 88 on their benchmark).

**HYPOTHESIZED MECHANISM HERE** Three things make this fit unusually well.
(1) **The exchangeability null is exactly right for us**: the historical segment is
guaranteed break-free, so historical points give an exact calibration set.
(2) The martingale is a *capital process* — a genuinely calibrated, unit-free evidence
measure, comparable across series without any extra normalisation. That is the single
property our project says the metric rewards.
(3) The Simple Jumper adapts its bet size **online, per series, with no tuning** — which is
what we need given 10,000 heterogeneous DGPs.
The smoothing randomisation θ_n ~ U[0,1] must be replaced by a deterministic tie-break
(see risk).

**CAUSAL 2026 TRANSLATION** Conformity score = a nonconformity of `x_t` w.r.t. the historical
segment. Because scale, not location, is the break, use **α = |x_t|** and **α = |AR(6)
residual_t|** (and, for dependence, `α = |x_t · x_{t−1}|` — a lag-1 product, which turns an
autocorrelation change into a mean change, exactly the trick used in the deep-CPD paper,
entry 24). p-value = rank of α_t within the sorted historical α-array → **O(log n_hist)** via
binary search on a precomputed array. Then the Simple Jumper is 3 scalars, O(1). Emit
`log S_t`, decayed peak, time-since-peak, per channel.

**ALREADY IMPLEMENTED?** **Almost certainly no as a conformal martingale.** `m02_dist` has
PIT channels (the p-values) and `m07_bayes` has e-processes, but the *betting-martingale
composition of conformal p-values with an adaptive jumper* is a specific and different object.

**EXPERIMENT** 9 columns: 3 channels (|x|, |AR-resid|, |x·x_{t−1}|) × {log S, decayed peak,
time-since-peak}. Screen against `m00_core`, then 5-fold. Strong prior in favour: it is cheap,
it is calibrated by construction, and it is orthogonal to windowed z-statistics.

**FALSE SIGNAL** (a) **Determinism**: the θ_n ~ U[0,1] randomisation breaks the 1e-8 check.
Use θ_n = 0.5 deterministically (slightly conservative, still valid enough for a *feature*).
(b) Any historical/online distributional difference that is not a break — e.g. the measured
variance-ratio bias by kurtosis quartile — makes p-values non-uniform from t=1 and the
martingale drifts up on *every* series. **This is precisely the artifact that would make the
feature a τ-detector.** Calibrate `log S_t` against the trajectory of `log S` computed on
held-out historical windows of matched length.

**COST** cheap. **LEAKAGE RISK** low mechanically; **medium** via (b).

---

### 22. Robust / generalised-Bayes BOCPD (Dm-BOCD, β-BOCD)

**SOURCE** https://proceedings.mlr.press/v202/altamirano23a/altamirano23a.pdf — Altamirano,
Briol, Knoblauch, *Robust and Scalable Bayesian Online Changepoint Detection*, ICML 2023.

**ORIGINAL METHOD** Keep the standard BOCPD run-length recursion
`p(r_t, x_{1:t}) = Σ_{r_{t−1}} p(x_t | x^{(r_t)}_{t−1}) H(r_t|r_{t−1}) p(r_{t−1}, x_{1:t−1})`
but replace the Bayesian posterior with a **generalised-Bayes belief**
`π ∝ π(θ)exp{−ωT·D̂_m(θ)}` using a **diffusion score-matching divergence**, which stays
conjugate in closed form for exponential families. Complexity **O(T(d² + p²))**, same as
standard BOCPD, >10× faster than β-BOCD; run-length pruning to top k=50. Results: on the 2013
Twitter Flash Crash, standard BOCPD falsely flagged 3 changepoints in a market blip while
Dm-BOCD correctly flagged none. Synthetic with outliers: PPV 0.907 vs 0.600.

**HYPOTHESIZED MECHANISM HERE** **The Flash Crash result is our exact problem statement.**
17 % of our no-break series contain a break-like transient, and 15.6 % of matched break-free
historical windows do too — transients are the DGP. A robust BOCPD is a principled detector
that *does not fire on transients*, which is the disambiguation our forensics says is the
whole game. Our champion currently solves this with peak/persistence *shape* channels; a
robust likelihood solves it upstream, at the evidence level, and would therefore be
**structurally decorrelated** from the shape channels rather than redundant with them.

**CAUSAL 2026 TRANSLATION** BOCPD is natively online. Run-length posterior truncated to
R_MAX ≈ 50–80 (we already use 80), Gaussian model on the AR(6) residual with unknown mean AND
**unknown variance** (Normal-Inverse-Gamma), score-matching-weighted likelihood. Emit:
posterior mass on short run-lengths, `E[r_t]`, `P(r_t < r_t^max/2)`, the MAP run-length, and
the difference between the **robust** and **standard** posteriors (their disagreement is the
transient indicator, and is the most interesting single column here).

**ALREADY IMPLEMENTED?** **Standard BOCPD yes** (`m07_bayes::bo_lo_change_pk`,
`bo_mean_rel` are top-10 champion features). **Robust/generalised-Bayes BOCPD: no.**

**EXPERIMENT** Add a robust-BOCPD twin and, critically, the **robust-minus-standard**
posterior contrast — 6 columns. Falsifiable prediction with a specific sign: the gain should
concentrate on the 17 % of no-break series that carry transients. Measure that subgroup
separately (we have the taxonomy to do it).

**FALSE SIGNAL** The robustness weight ω is a tuning knob; picking it on dev folds is a fold-
noise trap. Fix it from theory or from the historical segment only.

**COST** medium (O(R_MAX) per point = O(50–80), same order as our existing BOCPD).
**LEAKAGE RISK** low.

---

### 23. PM-CuSum — mixture of predictive distributions over window lengths, Fixed-Share weights

**SOURCE** https://arxiv.org/html/2606.05072 — *Sequential Change Detection using Mixtures of
Predictive Distributions*.

**ORIGINAL METHOD** `S_n = max{S_{n−1}, 0} + log(p̂_n(X_n)/q(X_n))`, where the post-change
predictive `p̂_n = Σ_w π_n^{(w)} p̂_n^{(w)}` mixes predictors fitted on windows of several
lengths w, and the weights π update by **Fixed Share** (reallocate a fraction α uniformly,
otherwise proportional to recent predictive performance). Only O(log log γ) windows are
needed — a dyadic set {2,4,8,16,…}. First-order asymptotically optimal with delay
`log γ / D(p‖q) + O((log log γ)²)`.

**HYPOTHESIZED MECHANISM HERE** Our own `glz_mix` (log-mean-exp mixture over dyadic candidate
change points) **failed** — 0.008 % gain vs 3.56 % for the max — and the diagnosed reason was
that a static mixture is a smoothed copy of the max. **Fixed Share is the fix**: the weights
are *adaptive and non-stationary*, so the mixture tracks the best window instead of averaging
over all of them. That is a materially different estimator, and it is the natural repair for
one of our recorded negatives.

**CAUSAL 2026 TRANSLATION** `q` = the historical (null) predictive, known at t=0. Windows
w ∈ {4,8,16,32,64,128}. Each `p̂^{(w)}` is a Gaussian with the trailing-w mean and variance —
O(1) ring-buffer updates. Fixed-Share is 6 multiplications + a renormalisation per point.
Emit `S_n`, its decayed peak, and **the argmax weight window `w*`** (an implicit scale
estimate of the break, and a candidate feature in its own right).

**ALREADY IMPLEMENTED?** **no** — we have `glz_mix` (static log-mean-exp, dead) and the
windowed bank, but no adaptively-weighted mixture.

**EXPERIMENT** 5 columns. Explicitly framed as the retry of `FAILED_EXPERIMENTS` N5: does
adaptive weighting rescue the mixture that static weighting could not? Clean falsification
either way.

**FALSE SIGNAL** Fixed Share's α is a tuning knob and the weights are path-dependent, so the
statistic is sensitive to the first few online points — high variance for small `t`.

**COST** cheap. **LEAKAGE RISK** low.

---

### 24. Train the detector on simulated data instead of deriving it (deep-learning CPD)

**SOURCE** https://personal.lse.ac.uk/wangt60/publication/aichangepoint.pdf /
https://academic.oup.com/jrsssb/article/86/2/273/7517020 — Li, Fearnhead, Fryzlewicz, Wang,
*Automatic change-point detection in time series via deep learning*, JRSS-B 86(2), 2024.

**ORIGINAL METHOD** Simulate labelled sequences (τ drawn at random, parameters before/after,
varied SNR, balanced classes), train a classifier (they show a single-hidden-layer net with
O(log n) nodes can *represent* CUSUM and generalised CUSUM; also a 21-block residual CNN).
**Key preprocessing: square the input for variance changes, and take products of adjacent
values for autocorrelation changes.** Result: matches CUSUM on i.i.d. Gaussian and
**substantially outperforms it under autocorrelated or heavy-tailed noise**.

**HYPOTHESIZED MECHANISM HERE** Two separable contributions, and the *smaller* one is the more
useful to us.
(a) **The preprocessing rule is free and immediately actionable**: `x_t²` converts a variance
break into a mean break, and `x_t · x_{t−1}` converts an autocorrelation break into a mean
break. **Every mean-optimal detector we have — and we have a lot of them, all aimed at a break
type that does not exist in this data (AUC 0.4998) — becomes correctly aimed if you feed it
these two streams instead of the raw one.** This reframes our headline forensic finding:
location breaks do not exist, but *in the squared and lag-product streams they do*.
(b) The deep-learning-from-simulation architecture itself — our "moonshot 1".

**CAUSAL 2026 TRANSLATION** (a) is two extra channels, O(1), fed into the existing `m00_core`,
`m01_seq`, `m07_bayes` machinery. (b) requires a causal architecture (TCN/Mamba) and 2 cores.

**ALREADY IMPLEMENTED?** (a) **partly** — `m04_resid` monitors "ACF-of-squares" and
`m03_dyn` covers dependence, and the underlying quantity is present. But whether the **full
detector bank** (CUSUM/GLR/SR/Page-Hinkley/e-processes/BOCPD) is run **on the `x²` and
`x_t·x_{t−1}` streams as first-class monitored representations** is not stated anywhere in the
module descriptions. **Treat as no, and check the code — this is the cheapest possible large
win if it turns out we only computed ACF-of-squares and not the whole bank on squares.**
(b) **no.**

**EXPERIMENT** (a) Re-run the `m01_seq` sequential bank with the input stream set to `x²−1`
and to `x_t·x_{t−1}` (both centred on their historical means), producing ~24 columns. Screen,
then 5-fold. **Prediction: this is where the mean-shift machinery we already own starts
working, and the gain should be largest in `scale_dominant` and `dep_dominant` classes.**
(b) A 3-layer causal TCN on 8 engineered channels, trained with binary `1[t≥τ]`, kept as an
8th ensemble stream regardless of standalone score.

**FALSE SIGNAL** (a) `x²` is heavy-tailed by construction (χ²-like), so a single outlier
dominates. Use a **winsorised or rank-transformed** square, and calibrate against the
historical-null trajectory. (b) A neural net will memorise series-level idiosyncrasy exactly
the way `m05_ctx` did — run the derangement control.

**COST** (a) cheap. (b) expensive on 2 cores. **LEAKAGE RISK** (a) low. (b) medium.

---

### 25. Contrastive / neural likelihood-ratio between pre- and post-change segments

**SOURCE** https://proceedings.mlr.press/v206/puchkin23a/puchkin23a.pdf — Puchkin, Shcherbakova,
*A Contrastive Approach to Online Change Point Detection*, AISTATS 2023.

**ORIGINAL METHOD** For each candidate τ, maximise a reparametrised cross-entropy
`T_{τ,t}(f) = ((t−τ)/t)Σ_{s≤τ}[f(X_s) − ln(1+e^{f(X_s)})] − (τ/t)Σ_{s>τ} ln(1+e^{f(X_s)})`
over a function class F; statistic `S_t = max_{τ<t} T_{τ,t}(f̂_{τ,t})`; declare a change when
`S_t > z`. Detection-delay bounds scale as `1/JS(p,q)` — i.e. governed by the **Jensen–Shannon
divergence** between pre- and post-change laws. Corollary 3.1 gives the first non-asymptotic
delay bound for neural-network discriminators.

**HYPOTHESIZED MECHANISM HERE** This is the formal version of our own Moonshot 1 ("neural
likelihood-ratio estimation between NOT-YET-BROKEN and ALREADY-BROKEN states, the
theoretically correct object for this metric"). It also supplies the right *effect-size
currency*: **JS divergence**, which is what Brandão's winning block also used as a feature
(entry 2). Two independent lines — one theoretical, one empirical-competitive — converge on
JS as the quantity that governs detectability here.

**CAUSAL 2026 TRANSLATION** The full per-τ optimisation is far too expensive online. The
affordable reduction: **fit the discriminator ONCE, offline, on the training data** —
a small MLP taking a fixed-length feature summary of a candidate pre-segment and post-segment
and outputting a logit — then at inference evaluate it on the existing dyadic τ grid and take
the max. O(#candidates × MLP) per point, which is only affordable if the MLP is tiny and the
grid is coarse (say 16 candidates, 2-layer 16-unit MLP). Alternatively evaluate it every 16
points and hold, which is causal and 16× cheaper.

**ALREADY IMPLEMENTED?** **no.**

**EXPERIMENT** Offline: train the discriminator on (pre-summary, post-summary) pairs from the
training folds, where negatives are placebo splits of break-free history — **we already have
this exact placebo machinery from the forensics work.** Then emit max-over-τ logit and its
decayed peak. 4 columns. Falsifiable at 5 folds.

**FALSE SIGNAL** The discriminator will learn `n_post` (segment length) unless pre/post
summaries are length-matched and length is withheld — the same artifact that makes
uncalibrated statistics τ-detectors.

**COST** medium. **LEAKAGE RISK** medium-high — this is the idea in this document most likely
to silently learn τ instead of the break. Length-matching is non-negotiable.

---

### 26. Self-normalisation — ratio-type statistics that need no long-run-variance estimate

**SOURCE** https://www.tandfonline.com/doi/full/10.1080/07350015.2023.2231041 (*A General
Framework for Constructing Locally Self-Normalized Multiple-Change-Point Tests*, JBES 2023);
https://doi.org/10.1111/rssb.12552 (*Segmenting Time Series via Self-Normalisation*, JRSS-B);
https://www3.stat.sinica.edu.tw/statistica/oldpdf/A31n120.pdf (*A Self-Normalized Approach to
Sequential Change-point Detection*, Statistica Sinica); and the adjusted-range-based SN
approach (https://www.researchgate.net/publication/388823732).

**ORIGINAL METHOD** Divide a CUSUM-type numerator by a **normaliser built from the same
recursive partial sums** (e.g. Σ_{s≤t} (partial CUSUM at s)²) rather than by an estimated
long-run variance. Yields pivotal limits without bandwidth selection.

**HYPOTHESIZED MECHANISM HERE** Our entire architecture solves the same problem a different
way: we estimate the null by simulating length-matched historical windows. Self-normalisation
solves it **analytically and for free**, with no null simulation at all. Two consequences:
(i) a much cheaper calibration path for the streaming port (no per-series null grids —
recall `m07_bayes` spent 225 ms/series on dense nulls before cutting to 73 ms);
(ii) SN statistics are **robust to unconditional heteroskedasticity**, which the Boldea–Hall
review (https://arxiv.org/html/2507.22204) flags as the exact condition under which standard
critical values go non-pivotal — and our data is scale-break-dominated, i.e. always in that
regime.

**CAUSAL 2026 TRANSLATION** Everything is a running sum of running sums: maintain `S_t = Σx`,
`Q_t = Σ_s S_s²`, and the analogous quantities on `x²`. The SN ratio at time t is
`(numerator)²/Q_t`. **O(1) per point, no null draws.** Run on x, x², AR-residual², |x|.

**ALREADY IMPLEMENTED?** **no** as a named family. Everything of ours is calibrated by
simulated historical nulls instead.

**EXPERIMENT** Two questions, and the second is the more valuable.
(1) 6 SN columns on top of `m00_core` — do they add?
(2) **Replace** the simulated-null calibration of 3 existing channels with SN and compare
TS-AUC *and* runtime. If SN matches the simulated null at a fraction of the cost, that buys
back a large chunk of the streaming-port compute budget.

**FALSE SIGNAL** SN statistics have low power against *very late* breaks (the normaliser is
contaminated by post-break data) — the mirror image of backward CUSUM's strength. Expect a
τ-quartile-dependent result and check that slice.

**COST** cheap; potentially **cost-negative** (it removes null simulation). **LEAKAGE RISK**
low — a pure function of the prefix.

---

### 27. Frozen training-time cross-sectional calibration (the legal form of rank conditioning)

**SOURCE** Constraint from https://forum.crunchdao.com/t/structural-break-real-time-may-infer-use-prior-completed-series-state/1186;
method form is standard (isotonic / quantile mapping). **[INFERRED]** — I found no public
source proposing this specific construction for this competition.

**ORIGINAL METHOD** n/a — this is a synthesis, flagged as inferred.

**HYPOTHESIZED MECHANISM HERE** TS-AUC compares series **within** a timestep. Our raw scores
have a strong shared time trend (evidenced by global Pearson 0.60–0.93 between streams vs
within-timestep rank correlation 0.39–0.78 — a gap our own report calls out). A calibration
that is a function of `t` alone is exactly monotone-within-timestep and therefore provably
neutral. A calibration that is a function of `(t, s)` where `s` is a *series-varying, causal*
covariate is **not** neutral and can genuinely re-rank.

**CAUSAL 2026 TRANSLATION** At training time, build a table
`F(t-bucket, s-bucket) → percentile of raw score`, where `s` is a causal online statistic —
candidates: current historical-null-calibrated evidence magnitude, or the online-prefix
kurtosis. At inference, look up. O(1), deterministic, no cross-series state.

**ALREADY IMPLEMENTED?** **no.**

**EXPERIMENT** Post-hoc on OOF. First run the **null version** (table indexed by `t` only) and
confirm ΔTS-AUC = 0 to numerical precision — that is a scorer unit test. Then run the
`(t, s)` version for two choices of `s` under LOFO.

**FALSE SIGNAL** Any gain from the `t`-only table indicates a scorer bug or tie-handling
difference, not a modelling result.

**COST** cheap. **LEAKAGE RISK** low; table must come from training folds only.

---

### 28. Heavy-tailed p-value combination (Cauchy / harmonic mean) for aggregating weak channels

**SOURCE** https://www.pnas.org/doi/10.1073/pnas.1814092116 (harmonic mean p-value, Wilson);
https://pubmed.ncbi.nlm.nih.gov/33012899/ (Cauchy combination test, Liu & Xie);
https://pmc.ncbi.nlm.nih.gov/articles/PMC12570179/ (*Aggregating dependent signals with
heavy-tailed combination tests*, 2025).

**ORIGINAL METHOD** `T_CCT = Σ w_i tan((0.5 − p_i)π)`; the null tail of a sum of Cauchys is
Cauchy **regardless of the dependence structure** among the p_i, so a valid combined p-value
is available in closed form with no correlation estimate.

**HYPOTHESIZED MECHANISM HERE** Our headline forensic fact is *"92.4 % of break series are
individually indistinguishable from a placebo split at p<0.01 on any family"* — this is
textbook weak-signal aggregation across many **strongly dependent** tests. Cauchy combination
is the standard tool for exactly that, and it is analytically valid under arbitrary
dependence, which is the hard part.

**CAUSAL 2026 TRANSLATION** Each calibrated channel already produces (or can produce) a
per-series historical-null p-value at each `t`. Compute `Σ tan((0.5−p_i)π)` across channels,
O(#channels) per point. Emit the combined statistic and its decayed peak.

**ALREADY IMPLEMENTED?** **no.** We have `xc_max_pk` / `xc_max_cur` (max across channels,
3.13 % of `m01_seq`'s gain) and `xc_n_hot` (hard-threshold count, **dead, 0.0 gain**). The
Cauchy sum is the *principled* version of `xc_n_hot` — continuous, tail-weighted, and valid
under dependence, which is precisely the three reasons `xc_n_hot` failed.

**EXPERIMENT** 4 columns (CCT statistic, harmonic-mean-p statistic, decayed peaks of both).
Screen. **Note the honest counter-argument: LightGBM can build its own combination from the
continuous inputs, which is exactly why `xc_n_hot` added nothing.** The claim that CCT is
different rests on it being a *tail-weighted* aggregation that a tree needs many splits to
approximate. Falsifiable, cheap, and the negative is informative.

**FALSE SIGNAL** `tan` explodes as p→0, so one badly-calibrated channel dominates the sum
entirely. Floor p at 1/(n_null+1). This is `FAILED_EXPERIMENTS` N7 waiting to happen again.

**COST** cheap. **LEAKAGE RISK** low.

---

### 29. Spectral / linear-spectral-statistic monitoring of covariance

**SOURCE** https://arxiv.org/html/2601.22602 — *A spectral approach for online covariance change
point detection*.

**ORIGINAL METHOD** Fisher matrix `F_k = S_1^{-1} S_{2,k}` from a reference and a monitoring
covariance; standardised increment of a linear spectral statistic
`L̃_k(f) = [Tr f(F_k) − Tr f(F_{k−1}) − μ_k]/σ_k`; CUSUM
`T_p(n,i) = max_{j≤i} |Σ_{t=j..i} L̃_t(f)|`; stop when it exceeds `c_α`.

**HYPOTHESIZED MECHANISM HERE** The construction is high-dimensional and we are univariate,
so it does not transfer directly. **But the univariate reduction is meaningful**: embed the
series into a lag-vector (x_t, x_{t−1}, …, x_{t−p+1}) and the "covariance" becomes the
**autocovariance matrix**, so monitoring its spectrum is monitoring the dependence structure —
our #3 break family. The `S_1^{-1}S_2` Fisher-matrix form is *itself* the per-series
calibration: the reference covariance comes from the break-free history.

**CAUSAL 2026 TRANSLATION** p = 6 lag embedding. `S_1` = historical lag-6 autocovariance
(constant, invert once). `S_{2,k}` = running online lag-6 autocovariance (rank-1 updates,
O(p²)=36 per point). `Tr f(F_k)` for f = log or f = (·−1)² over 6 eigenvalues — a 6×6
eigendecomposition per point is affordable but not free; **evaluate every 8 points and hold**
to cut it 8×. Emit the CUSUM of standardised increments and its decayed peak.

**ALREADY IMPLEMENTED?** **no.** `m03_dyn` has spectral band powers, but not a
historical-referenced Fisher-matrix spectrum of the lag-embedded covariance.

**EXPERIMENT** 5 columns. Screen. Judge specifically on the `dep_dominant` class (champion
0.697) rather than only pooled.

**FALSE SIGNAL** `S_1^{-1}` is ill-conditioned when the historical series is near-singular in
lag space (strong AR roots). Ridge the inverse and emit the condition number so the model can
discount it.

**COST** medium (O(p²) per point + periodic eigendecomposition). **LEAKAGE RISK** low.

---

### 30. Ordinal-pattern / turning-rate statistics for dependence change

**SOURCE** https://www.arxiv.org/pdf/2502.03099 (*Ordinal Patterns Based Change Point
Detection*); https://www.mdpi.com/1099-4300/20/9/709 (conditional entropy of ordinal patterns).

**ORIGINAL METHOD** The **turning rate**: relative frequency of the four "local extremum"
ordinal patterns of length 3 — (0,2,1), (1,0,2), (1,2,0), (2,0,1) — computed over blocks, then
a CUSUM (and a self-normalised variant) on the block series. Key identity for Gaussian
processes: **`cos(π·E[q̂_m]) = ρ(1)`** — the turning rate is a monotone reparametrisation of
lag-1 autocorrelation. Applied retrospectively (EEG sleep staging, p = 7.29e−5).

**HYPOTHESIZED MECHANISM HERE** The turning rate is an **ordinal, hence scale-invariant and
outlier-immune, estimator of lag-1 dependence.** That is unusual and valuable here: every other
dependence channel we have is moment-based and therefore contaminated by the scale breaks that
dominate this data. A scale-*invariant* dependence channel is the clean way to separate our #2
family (scale) from our #3 family (dependence), which currently confound each other. It also
requires no distributional assumption and no calibration beyond a historical rate.

**CAUSAL 2026 TRANSLATION** Trivially online: each new point completes one length-3 ordinal
pattern from (x_{t−2}, x_{t−1}, x_t); increment a counter if it is a turning pattern.
`q̂_t` = running mean of that indicator; compare to the historical turning rate `q_h`
(constant) via a Bernoulli z or a Bernoulli CUSUM/GLR. **O(1) per point, two comparisons.**
Extend to length-4 patterns (24 categories) for a richer dependence signature, still O(1).
This is the cheapest idea in the entire document.

**ALREADY IMPLEMENTED?** **no.** `m03_dyn` has "complexity" features but ordinal-pattern
turning rate is not named, and the *sequential Bernoulli monitoring of it against the
historical rate* certainly is not.

**EXPERIMENT** 6 columns: turning-rate Bernoulli z (expanding, w=128), Bernoulli max-GLR,
decayed peak, plus length-4 permutation-entropy z. Screen against `m00_core`, then 5-fold.
Judge on the `dep_dominant` class.

**FALSE SIGNAL** Ties in continuous data are measure-zero so patterns are well defined, but
**quantisation in the stored data would create ties and break it** — check for repeated values
in the raw series before trusting this. Also, the turning rate saturates: it only resolves
ρ(1), not higher-order dependence, so it will be redundant with any good ACF(1) channel we
already have. The novelty is the *robustness*, not the information.

**COST** **cheapest available.** **LEAKAGE RISK** very low.

---

### 31. Deliberately slow / robust volatility normalisation (the anti-GARCH lesson, sourced)

**SOURCE** https://maheu.azurewebsites.net/csda2010-pub.pdf — He & Maheu, *Real time detection
of structural breaks in GARCH models*, CSDA 54(11), 2010.

**ORIGINAL METHOD** A Bayesian change-point GARCH (Chib-style states, restricted Markov
transition enforcing ordered break dates) estimated **sequentially** by an auxiliary particle
filter with kernel-smoothed parameter learning, sidestepping GARCH's path-dependence problem.
Finding on NASDAQ: the *partial* structural break specification — **allowing only the
intercept to change** — beat full-parameter breaks; and with t-innovations, previously
"detected" breaks partly dissolved into tail observations.

**HYPOTHESIZED MECHANISM HERE** This directly corroborates our `RT-044-volG` post-mortem
(GARCH normalisation → 0.50012, "the filter adapts on the same timescale as the break"). He &
Maheu's answer is not to *filter* with GARCH but to put the break **inside** the model as a
latent state, and to restrict which parameters may change (intercept only = the unconditional
variance level). Their t-innovation finding is also a direct warning about our 17 % transient
problem: heavy tails masquerade as breaks in variance models.

**CAUSAL 2026 TRANSLATION** Not the particle filter — far too expensive. The transferable
piece is: monitor the **GARCH intercept ω implied by the historical fit** against a running
estimate, i.e. `(realised variance over trailing w) − (GARCH-predicted variance)` where the
GARCH parameters are **frozen from history and never updated**. A frozen filter cannot adapt
to the break, which is the precise fix for why our GARCH attempt scored 0.500. O(1) per point.

**ALREADY IMPLEMENTED?** **no** in this form. `RT-044-volG` fitted GARCH on history and
"filtered causally forward" — which is close, but the emitted monitors were variance-flavoured
statistics *of the normalised residual*, i.e. the adaptive quantity. Emitting the **frozen-model
prediction error** rather than the normalised residual is the inversion.

**EXPERIMENT** 3 columns: cumulative `(x_t² − h_t)` where `h_t` is the frozen-GARCH conditional
variance, its calibrated z, and its decayed peak. This is a 1-hour experiment that directly
tests whether our GARCH negative was about GARCH or about *which functional we emitted*.

**FALSE SIGNAL** Heavy tails: one large x_t² swamps the cumulative sum. Winsorise. He & Maheu's
own t-innovation result is the warning.

**COST** cheap. **LEAKAGE RISK** low (parameters frozen from history).

---

### 32. Heavy-tailed online CPD via clipped-SGD with finite-sample FPR control

**SOURCE** https://arxiv.org/abs/2306.09548 — *Online Heavy-tailed Change-point detection*.

**ORIGINAL METHOD** Clipped-SGD estimates the mean and simultaneously produces confidence
intervals at *all* confidence levels; the change is declared when intervals separate. Claim:
*"the first OCPD algorithm that guarantees finite-sample FPR, even if the data is high
dimensional and the underlying distributions are heavy-tailed"*, under only bounded second
moments.

**HYPOTHESIZED MECHANISM HERE** Our measured artifact — variance-ratio bias of −0.004
(lightest kurtosis quartile) vs −0.063 (heaviest) — is a heavy-tail-induced miscalibration
that mis-ranks the cross-section. A detector whose false-positive rate is *uniform over tail
behaviour* is exactly the corrective: it makes the evidence comparable across DGPs without
per-series null simulation.

**CAUSAL 2026 TRANSLATION** Clipped-SGD on the stream `x_t² − 1` (variance break as a mean
break, per entry 24), with clipping radius set from the historical distribution. Two scalars
(iterate + step counter). Emit the running estimate and the width of the anytime-valid
confidence interval. O(1) per point.

**ALREADY IMPLEMENTED?** **no.**

**EXPERIMENT** 4 columns. Screen. The falsifiable, DGP-conditional prediction: gains should
concentrate in the **heaviest historical-kurtosis quartile**, where our current features are
most biased. Slice on that.

**FALSE SIGNAL** The clipping radius is a per-series historical quantity → series-constant
contamination. Emit only the calibrated statistic, never the radius.

**COST** cheap. **LEAKAGE RISK** medium (see above).

---

### 33. Boldea–Hall: wild bootstrap when covariance stationarity fails

**SOURCE** https://arxiv.org/html/2507.22204 — Boldea & Hall, *Testing for multiple
change-points in macroeconometrics*. (Note: the lead author, Otilia Boldea, is a professor of
econometrics specialising in break-point detection and **posted in the CrunchDAO forum
recruiting a team**, https://forum.crunchdao.com/t/structural-break-contest-tips/943 — but
that thread contains **no technical content whatsoever**, only team-formation replies. I
checked; there is nothing there.)

**ORIGINAL METHOD** sup-F / sup-Wald, UDmax / WDmax, sequential ℓ vs ℓ+1 tests, dynamic
programming at O(T³). **Critical warning**: when covariance stationarity fails under the null
(Great Moderation, crises), critical values are **non-pivotal** and tabled values are invalid;
use a wild bootstrap (Rademacher or Mammen multipliers, B = 400–1000). Explicitly **does not
cover online monitoring** — I asked and the review is entirely retrospective.

**HYPOTHESIZED MECHANISM HERE** The *warning* is the transferable content, and it validates our
architecture: our data is scale-break-dominated, i.e. permanently non-covariance-stationary,
so any tabled critical value is wrong. Our per-series simulated historical null **is** the
bootstrap, done right, and this is the strongest external justification for it I found. The
sup-Wald/UDmax machinery itself is aimed at mean/coefficient breaks and is therefore aimed at
the break type our data does not contain (AUC 0.4998).

**CAUSAL 2026 TRANSLATION** One concrete import: the **wild bootstrap for the historical null**.
Instead of drawing length-matched historical windows (which reuses the same data and
under-represents tail variation), multiply historical residuals by i.i.d. Rademacher signs and
recompute the statistic. Cheap and generates unlimited null draws. Must be seeded.

**ALREADY IMPLEMENTED?** **no** — our nulls come from length-matched historical windows.

**EXPERIMENT** Swap the null-generation mechanism in `m00_core` for a Rademacher wild bootstrap
at matched length; compare TS-AUC and null-draw cost. Note `m07_bayes` already found that
**denser null grids bought nothing** (225 ms → 73 ms with no loss), so the prior on
"better nulls help" is **negative**. Low priority; recorded for completeness.

**FALSE SIGNAL** Wild bootstrap preserves the historical dependence structure only under
specific conditions; on strongly autocorrelated series a sign-flip bootstrap destroys the
autocorrelation and produces a null that is too easy to reject.

**COST** cheap. **LEAKAGE RISK** low; must be seeded for the 1e-8 determinism check.

---

### 34. Multiscale MOSUM with gradual bandwidth adjustment

**SOURCE** https://arxiv.org/html/2103.01060 (*Multiscale change point detection via gradual
bandwidth adjustment in moving sum processes*, EJS 17); https://cran.r-project.org/web/packages/mosum/mosum.pdf.

**ORIGINAL METHOD** Moving-sum statistics at many bandwidths, with a gradual bandwidth
adjustment rather than a fixed dyadic set, and localised pruning to combine scales.

**HYPOTHESIZED MECHANISM HERE** We already run rectangular windows at w = 8…256 in `m00_core`,
which is the discrete version. The unexploited part is the **combination rule across scales**:
we hand the model all scales and let LightGBM pick; MOSUM theory says the right combination is
scale-penalised (larger bandwidths get a different threshold). A scale-penalised max is a
different feature from the raw per-scale bank.

**CAUSAL 2026 TRANSLATION** `max_w [ Z_w(t) − a(w) ] / b(w)` with the standard
`a(w) = √(2 log(T/w))`-type penalty, but with `T` **replaced by `t`** (we cannot know the
horizon). Plus the argmax `w*` as a feature (the implied break scale). O(#scales) per point,
free given the existing bank.

**ALREADY IMPLEMENTED?** **Partly** — the bank exists; the penalised cross-scale max and the
argmax-scale do not appear in the feature list (`xc_max_pk` is a cross-*channel* max, not a
cross-*scale* penalised max).

**EXPERIMENT** 4 columns: penalised cross-scale max, argmax scale `w*`, decayed peak,
`w*` stability. Cheap and purely derived from existing quantities.

**FALSE SIGNAL** `w*` is mechanically small when `t` is small; residualise against `t`.

**COST** cheap. **LEAKAGE RISK** low, provided the penalty uses `t` and never `n_online`.

---

### 35. Hard-negative augmentation: synthesise reverting transients

**SOURCE** No direct public source for this competition. **[INFERRED]** — synthesised from
our own forensics (17 % of no-break series contain break-like transients) plus the
simulate-and-train paradigm of Li et al. (entry 24) and the robust-BOCPD flash-crash result
(entry 22).

**ORIGINAL METHOD** n/a.

**HYPOTHESIZED MECHANISM HERE** Our champion's most valuable inputs are peak/persistence
channels — the model has *already* discovered that duration, not amplitude, is the
disambiguator. Training on synthetic negatives that are *maximally* transient-like would
sharpen exactly that boundary, and it is the only idea here that adds **data** rather than
model variety (alongside 2025 transfer).

**CAUSAL 2026 TRANSLATION** Generate negatives by taking a break-free historical segment,
splitting it, and injecting a variance/dependence excursion that **reverts** after
d ∈ {10,…,200} points. Label 0. Interleave with real training rows.

**ALREADY IMPLEMENTED?** **no** — listed in our NEXT EXPERIMENTS at #6.

**EXPERIMENT** Augment training rows by 20 % with synthetic reverting transients; 5-fold, same
features. Falsifiable: does TS-AUC on the *real* OOF rise, and specifically on the
`weak_unclassified` mass (3,673 series, currently 0.601, where the metric is won)?

**FALSE SIGNAL** If the synthetic transients are not distributionally matched to the real ones,
the model learns to detect *synthetic*, and OOF improves for the wrong reason. Validate by
checking the model's score distribution on the real 17 % transient-bearing no-break subgroup.

**COST** medium. **LEAKAGE RISK** low (synthetic data has no labels to leak), but the
augmentation must be regenerated per fold with a per-fold seed.

---

### 36. 2025 → 2026 transfer as pseudo-real-time training series

**SOURCE** No public source proposes this. **[INFERRED]** — from the fact that the 2025 data
is public at https://hub.crunchdao.com/competitions/structural-break with known boundaries.

**ORIGINAL METHOD** n/a.

**HYPOTHESIZED MECHANISM HERE** 2025 series have a **known** boundary and a label. Cutting each
2025 series at its boundary gives (history, online-with-known-τ) pairs — exactly our training
format, and it roughly doubles the labelled data. **Caveat that must be checked first**: the
2025 series are 2,000–5,000 points with a boundary somewhere inside, whereas 2026 has
n_hist ∈ [1000,5000] and n_online ∈ [10,999]. The geometry differs and the generator may
differ. If the DGP families differ, this adds noise, not data.

**CAUSAL 2026 TRANSLATION** Preprocessing only.

**ALREADY IMPLEMENTED?** **no** — our NEXT EXPERIMENTS #5, marked "never attempted".

**EXPERIMENT** Before any training run, do the **cheap diagnostic first**: build the `m05_ctx`
historical characterisation vector on 2025 and 2026 series and train a classifier to tell the
two corpora apart. If it separates at AUC > 0.9, the generators differ and transfer will hurt;
that is a 20-minute experiment that decides whether to spend the day.

**FALSE SIGNAL** Any gain could come from extra data volume rather than transfer; control with
an equal-size subsample of 2026-only data.

**COST** medium-high. **LEAKAGE RISK** low.

---

## RANKED SHORTLIST — 10 highest expected-value ideas we have NOT implemented

Ranked by (probability of a real gain) × (size of gain) ÷ (cost), with our own operating rule
applied: **feature additions screen reliably; architectures and objectives do not.** Eight of
the ten below are feature additions, deliberately.

1. **(#16) AR(p)-FOCuS — exact max-GLR over ALL change points on squared AR(6) residuals.**
   Replaces our dyadic-grid GLR (already a top-10 feature family) with the exact statistic at
   O(log n)/step; targets scale + dependence + autocorrelation simultaneously; independently
   validated in JMLR and extended to AR(p) in 2026. Best method-to-problem match found.

2. **(#24a) Run the ENTIRE existing detector bank on the `x²` and `x_t·x_{t−1}` streams.**
   Squaring turns a variance break into a mean break; lag-products turn an autocorrelation
   break into a mean break. We own a large, well-tuned mean-shift apparatus that our own
   forensics says is aimed at a break type that does not exist — this re-aims all of it for
   near-zero cost. Highest gain-per-hour on the list; check the code first, it may be partly done.

3. **(#14 ablation) Find out which part of `m07_bayes` actually works.** Our best module by
   gain-per-column (+0.0397) has **no attribution** and its own author flagged it. Four
   drop-one runs plus one AR(6)+normal-scores-through-`m00_core` run. If the stream is doing
   the work rather than the recursions, that finding transfers to every module for free —
   the highest-information experiment in the document, and it is not even a new feature.

4. **(#18) Stacked backward CUSUM on z² and AR-residual².** Attacks a weakness we have
   *measured*: late-τ TS-AUC 0.571–0.588 vs early-τ 0.624–0.628. Otto & Breitung report
   near-constant detection delay across break locations where forward CUSUM degrades. Every
   sequential statistic we own is forward. Use the infinite-horizon boundary — never `n_online`.

5. **(#30) Ordinal-pattern turning rate, monitored as a Bernoulli GLR against the historical
   rate.** The only **scale-invariant** dependence channel available; separates our #2 family
   (scale) from our #3 family (dependence), which currently contaminate each other in every
   moment-based feature. Genuinely O(1) — cheapest idea here by a wide margin.

6. **(#21) Conformal test martingale with a deterministic Simple Jumper, on |x|,
   |AR-residual| and |x·x_{t−1}|.** Exact exchangeability null from the guaranteed-break-free
   history; produces a calibrated, cross-series-comparable capital process — the exact property
   our project says TS-AUC rewards — with no tuning. Set θ = 0.5 for 1e-8 determinism.

7. **(#3) Robust scale tests (Brown–Forsythe / Fligner) on |x − historical median|.** Directly
   targets the dominant break family (scale, AUC 0.559) in a heavy-tail-robust form, using two
   running accumulators. Independently used by *four* separate public 2025 solutions —
   the most corroborated single feature family in the public record.

8. **(#6) Localisation v2: τ̂ stability, best-vs-second-best margin, argmax drift.** `m06_loc`
   delivered +0.0315 while explicitly under-built; secabird's "distance from max variance shift
   to the boundary" is the public analogue. τ̂ *wandering* is a transient; τ̂ *pinning* is a
   break — the disambiguation our forensics says is the whole problem. Residualise against `t`.

9. **(#22) Robust (generalised-Bayes) BOCPD twin, and the robust-minus-standard contrast.**
   The Flash-Crash result (standard BOCPD: 3 false alarms; robust: 0) is our 17 %-transient
   problem in miniature. The *contrast* column between the two posteriors is a direct transient
   indicator and is structurally decorrelated from our shape/persistence channels.

10. **(#26) Self-normalised statistics as a calibration path.** Two-sided value: possibly
    additive as features, and possibly **cost-negative** — it could replace per-series null
    simulation entirely, buying back compute for the streaming port, which is our stated
    largest deployment risk. Robust to unconditional heteroskedasticity, which the Boldea–Hall
    review identifies as exactly the regime our scale-dominated data sits in.

*Just below the line, and why:* **(#19) NEWMA** — cheap and orthogonal, but our level-EWMA bank
was mostly dead and the kernel version is unproven here. **(#5) TabPFN stream** — the public
record supports it, but the 2-core inference cost is a real obstacle and it must be
teacher-only. **(#36) 2025 transfer** — highest ceiling of anything on the list, but do the
20-minute corpus-discriminability diagnostic before committing a day to it.

---

## DEAD ENDS PUBLICLY REPORTED

Honest scope note: the public record for this competition is **thin on negatives**. Winners
publish what worked. Almost everything below is either (a) a negative *inferable* from what
public solutions conspicuously did not use, or (b) a negative from the academic literature.
Where I am inferring, I say so. I am not going to dress up absence of evidence as evidence.

1. **Mean/location-shift tests are the wrong target — and the public record shows the winners
   knew it.** Every public 2025 solution that lists its tests leads with **variance** tests
   (F-test, Levene, Fligner, Brown–Forsythe, Mood) and **distributional** distances (KS,
   JS, Wasserstein, energy, Anderson–Darling, Cramér–von Mises). The t-test appears only in the
   weakest public solutions (gsoisson's baseline description, StefanConstantin707's baseline).
   This independently corroborates our forensic finding (location AUC 0.4998) from four
   separate codebases. **[INFERRED from composition, not from a stated negative.]**

2. **The Boldea–Hall review states outright that tabled critical values for sup-F / sup-Wald /
   UDmax are invalid when covariance stationarity fails under the null**, requiring a wild
   bootstrap. Our data is scale-break-dominated, i.e. always in that regime. Anyone importing
   classical break-test *thresholds* here will get garbage. (Retrieved, explicit.)

3. **The Boldea–Hall review does not cover online monitoring at all.** The entire modern
   econometric multiple-break apparatus (dynamic programming, O(T³), sequential ℓ vs ℓ+1) is
   retrospective. There is no cheap import from that literature into a single-pass streaming
   detector. (Retrieved, explicit — I checked specifically for this.)

4. **Standard BOCPD false-alarms on transients.** Altamirano et al. show standard BOCPD
   flagging 3 spurious changepoints in the 2013 Twitter Flash Crash where the robust version
   flagged none, and PPV 0.600 vs 0.907 on outlier-contaminated synthetic data. Since our data
   contains transients in 17 % of no-break series *by construction*, a vanilla BOCPD block is a
   known false-positive generator. (Retrieved, explicit.)

5. **GARCH-type filters erase the break they are meant to detect, and this is confirmed
   externally.** Our `RT-044-volG` scored 0.50012. He & Maheu independently found that the
   *filtering* approach is the wrong frame and that breaks must be modelled as latent states,
   and further that with t-innovations previously "detected" GARCH breaks partly dissolve into
   tail observations. Two independent confirmations that adaptive volatility filtering is a
   dead end for detection. (Retrieved, explicit.)

6. **Conformal test martingales: unsigned scores detect ~60 % slower than signed scores**
   (median delay 88 vs 55 in Vovk et al.). Directional information matters. Since our breaks
   are in scale rather than location, the analogue is `x² − 1` (signed in the variance
   direction) rather than `|x|` (unsigned) — build the *signed* version. (Retrieved, explicit.)

7. **Static mixtures over candidate change points are dominated by the max.** Our `glz_mix`
   scored 0.008 % gain vs 3.56 % for `glz_pk`. The PM-CuSum paper's whole contribution is that
   you need **adaptive** (Fixed Share) weights, not static ones, and the FOCuS line of work
   argues for the exact max instead. Three lines of evidence agree: **do not average over
   candidate change points with fixed weights.**

8. **Cross-series state and cross-series ranking at inference are not viable.** Organiser
   `enzo`, on the record: parallelism plus the 1e-8 determinism re-check on a 10 % subset
   "will work against you", and disk-based state persistence "we will disqualify". Our
   NEXT-EXPERIMENTS #9 should be closed and replaced by entry 27. (Retrieved, explicit.)

9. **Every pre-June-8 leaderboard score is void, and any writeup based on them is worthless.**
   The leak let submissions recover series length inside `infer()`. All predictions were
   invalidated. Do not calibrate ambition against any number from that period.
   (Retrieved, explicit.)

10. **Scan-B / online kernel CUSUM is O(NB²)–O(Nw²) per step.** The authors state it plainly.
    On 2 cores with ~10 M scoring rows this is at or past our budget. NEWMA exists precisely as
    the O(m) approximation. **Do not reach for the exact kernel method first.** (Retrieved.)

11. **The ordinal-pattern change-point paper is retrospective, not online**, and it tests for a
    *mean shift in the turning-rate series* rather than for dependence change directly. The
    online Bernoulli-monitoring version proposed in entry 30 is my construction, not theirs.
    **[Marked so nobody cites the paper for a claim it does not make.]**

12. **Discretising continuous evidence for a GBM is self-harm** — our own `xc_n_hot` scored
    exactly 0.0 gain. Worth restating here because the Cauchy-combination idea (entry 28) looks
    superficially like a rescue of it and must be justified on the *tail-weighting* argument
    alone, not on "counting hot channels is a good idea".

13. **Not found, therefore not usable — stated so nobody re-searches:** no 2026 real-time
    solution repo, notebook, blog or forum writeup exists publicly as of 2026-08-19; the 2025
    winners' writeups referenced by staff on the forum are behind a JS leaderboard I could not
    retrieve; the results video and the X announcement are robots-blocked; the Kaggle real-time
    baseline serves no content to fetchers; and **"Inni Dynamics" returns nothing anywhere on
    the open web** — I could not verify that team name exists in public at all.

---

## APPENDIX — sources actually retrieved

**Competition (13):**
docs.crunchdao.com/competitions/competitions/structural-break-real-time ·
adialab.ae/adia-lab-x-crunch-the-structural-break-challenge-2026 ·
adialab.ae/structural-break-open-benchmark · structural-break.crunchdao.com ·
forum.crunchdao.com/t/1070 (empty) · /t/943 (empty) · /t/1156 · /t/1164 · /t/1186 · /t/1188
(+ .json full text) · hub.crunchdao.com/competitions/structural-break-real-time (metadata only) ·
hub.crunchdao.com/competitions/structural-break/leaderboard (no data) ·
hub.crunchdao.com/competitions/structural-break-open-benchmark (metadata only)

**2025 solutions (7):**
humbertobrandao.medium.com (Alphabot, 1st) · github.com/aParsecFromFuture/... (2nd, + raw
README) · github.com/secabird/structural-break-challenge · github.com/gsoisson/adia-structural-break ·
github.com/StefanConstantin707/adia-lab-structural-break-challenge ·
github.com/abkimc/ADIA-Lab-Structural-Break-Challenge-Solution (52/490) ·
github.com/humbertobrandao (profile) · github.com/aParsecFromFuture (profile)

**Academic (16):**
arxiv 2203.03532 (e-detectors, abstract + full PDF) · Biometrika 112(1) asae049 (e-processes,
mean & variance) · arxiv 2102.10439 (conformal test martingales) · arxiv 2110.08205 +
jmlr.org/papers/v24/21-1230 (FOCuS) · arxiv 2607.16106 (AR(p)-FOCuS) · arxiv 2302.02718
(NP-FOCuS) · eprints.lancs.ac.uk/174160 (NUNC) · arxiv 2606.05072 (PM-CuSum) ·
PMLR v202 altamirano23a (robust BOCPD) · arxiv 2003.02682 (backward CUSUM) · arxiv 1805.08061
(NEWMA) · arxiv 2211.15070 (online kernel CUSUM / Scan-B) · PMLR v206 puchkin23a (contrastive
OCPD) · personal.lse.ac.uk/wangt60 (deep-learning CPD, JRSS-B) · arxiv 2601.22602 (spectral
covariance CPD) · arxiv 2502.03099 (ordinal patterns) · arxiv 2306.09548 (heavy-tailed OCPD) ·
arxiv 2507.22204 (Boldea & Hall review) · maheu.azurewebsites.net/csda2010-pub.pdf (He & Maheu)

**Failed to retrieve (7):** youtube.com/watch?v=LFWdWgAcOqU and ?v=pMAGJHlImpE (429 /
robots) · x.com/adia_lab/status/1986732959710498829 (robots) ·
kaggle.com/code/crunchdao/structural-break-real-time-baseline (no content served) ·
github.com/crunchdao/competitions tree pages (robots) · github.com/search (robots) ·
sciencedirect NUNC full text (robots) · api.hub.crunchdao.com leaderboard (404)
