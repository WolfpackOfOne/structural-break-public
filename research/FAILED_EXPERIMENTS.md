
---

## agent5 / m01_seq — negative results (2026-08-18)

Context: `m01_seq` overall is a KEEP (+0.01425 screen fold-0 TS-AUC over an
exact paired control, RT-010C 0.57384 -> RT-010T 0.58809). These are the parts
of it that did NOT work, measured by LightGBM gain share in RT-010T
(211 features, screen fold 0, seed 0) and by the RT-010N ablation.

### N1. Slope-of-evidence channels — DEAD (zero gain)
- **Hypothesis** The local slope of a detector path over the last 16 steps
  distinguishes "still accumulating" (break) from "decaying" (transient).
- **What I did** Emitted `cz50_slp`, `ce50_slp`, `sq50_slp`, `glz_slp`,
  `ew32_slp`, each normalised by the historical-null spread of that detector.
- **Number** Gain = **exactly 0.0** for all five. LightGBM never split on them.
- **Why I think it failed** The slope is almost rank-equivalent to
  `current − decayed_peak`, which the model already has in a lower-variance
  form, and differencing a nonnegative accumulating path over 16 steps is a very
  noisy estimator of the drift. The information is real but already present.
- **Retry warranted?** Only as a *replacement* for `*_dpk`, not an addition, and
  with a much longer differencing lag (64–128). Low priority.

### N2. Hard-threshold cross-channel count (`xc_n_hot`) — DEAD (zero gain)
- **Hypothesis** "How many independent detectors are currently hot" is stronger
  evidence than any single detector.
- **What I did** `xc_n_hot = #{channels with calibrated surprise > 2.0}`.
- **Number** Gain = **0.0**, while the continuous versions `xc_max_pk` (3.13 %
  of total gain) and `xc_max_cur` were kept.
- **Why I think it failed** Thresholding at a fixed surprise destroys exactly
  the ordering a tree would otherwise exploit, and a tree can build its own
  threshold from the continuous max anyway. Discretising for a GBM is
  self-harm.
- **Retry warranted?** No.

### N3. Calibrated *current level* of the sequential statistics — REDUNDANT
- **Hypothesis** The calibrated current value of CUSUM / GLR / SR / PH carries
  information beyond `m00_core`'s trailing- and expanding-window bank.
- **What I did** `*_cur` columns for all 12 channels.
- **Number** Near-zero gain given `m00_core`: `cz50_cur` 0.0015 %,
  `glz_cur` 0.005 %, `ce50_cur` 0.006 %, `srz_cur` 0.003 %, `ce25_cur` 0.003 %.
  Their *peak* counterparts, by contrast, are 5 of the top 10 features overall.
- **Important nuance** They are not worthless in isolation: RT-010N (levels only,
  18 m01_seq cols) still scores 0.58229, +0.00845 over control. They are
  worthless *conditional on* `m00_core`.
- **Why I think it failed** A sequential detector's current value is a smoothed
  function of the recent window means, which is exactly what `m00_core` already
  supplies at six scales with better calibration. What the recursion adds is
  memory of its own past, not its present.
- **Retry warranted?** No — instead, drop most `*_cur` columns and spend the
  budget on more peak/persistence channels.

### N4. EWMA bank — mostly redundant with the window bank
- **Hypothesis** Exponentially-weighted views at half-lives 8/32/128 add
  bounded-memory level evidence over rectangular windows.
- **Number** `ew8_z` 0.0011 %, `ew32_z` 0.0022 % gain — effectively dead.
  `ew128_z` (0.072 %) and `ew32_pkrel` (0.067 %) survive but are marginal.
  `ewv32_z` (variance ratio) 0.057 %.
- **Why I think it failed** `m00_core` covers w = 8…256 rectangular windows plus
  an expanding window, all calibrated by `ctx.nc` against a *length-matched*
  historical null. My EWMA robust-z uses only the marginal historical EWMA
  distribution, which is a weaker calibration. The rectangular version wins.
- **Retry warranted?** Only the *variance-ratio* EWMA and the long half-life,
  and only if the calibration is upgraded to a length-matched null. The
  short-half-life EWMA level channels should be dropped.

### N5. Mixture-GLR — dominated by max-GLR
- **Hypothesis** Averaging the likelihood over dyadic candidate change points
  (log-mean-exp) is better conditioned than maximising over them.
- **Number** `glz_mix` 0.008 % gain vs `glz_pk` 3.56 % and `gle_pkr` 3.86 %.
- **Why I think it failed** The mixture is dominated by its largest term
  whenever any evidence exists, so it is a smoothed copy of the max, with the
  smoothing removing the tail resolution that actually discriminates.
- **Retry warranted?** No, not in this form.

### N6. (process) Mixture-GLR global normaliser — a real causality bug, caught
Not a modelling failure but worth recording: the first implementation normalised
the mixture by a count of dyadic windows that fit in `n_online`, making row `t`
depend on the total series length. `check_prefix_invariance` failed at prefix 1
on 12/12 series with `atol=0.0`. It produced perfectly plausible-looking
features and would have silently inflated every downstream number. Fixed by
making the mixture row-local. **Run the harness before you look at any score.**

### N7. (process) Unfloored null spread — a garbage column, caught by audit
The historical-null spread was floored at `1e-9`. For series whose AR-residual
CUSUM null is near-degenerate this made `ce50_slp` explode to ±3e7 (column mean
-2357, sd 2.4e5). Prefix invariance passed; nothing in the pipeline complained.
Only a per-column min/max/sd audit over the built feature matrix found it. Floor
robust scales by a fraction of the null range, and audit every column's
distribution after the first build.

## RT-063-tl / RT-063-rk — agent6 — tail-asymmetry and rank-CUSUM families add nothing on top of m00_core

**Hypothesis.** Within `m02_dist`, (a) tail occupancy *asymmetry* at the historical
q01/q05/q95/q99 thresholds and centre depletion, and (b) distribution-free rank
statistics (rolling variance of the PIT, Kolmogorov/Darling-Erdos rank CUSUM on the
raw and AR-residual PIT streams, all calibrated against a length-matched historical
null), each carry independent break evidence that `m00_core`'s moment/tail monitoring
misses.

**What I did.** Screen protocol, fold 0, seed 0, `max_train_rows=300_000`,
`n_estimators=300`, paired against my own re-run of the control in the same session.
Sliced my own module with `keep_regex` so only one family at a time sat on top of
`m00_core`.

| run | slice | n_feat | TS-AUC | Δ |
|---|---|---|---|---|
| RT-060C | `m00_core` control | 151 | 0.57384 | — |
| RT-063-tl | + `m02_dist::tl_` (6 cols) | 157 | 0.56974 | **−0.00410** |
| RT-063-rk | + `m02_dist::rk_` (6 cols) | 157 | 0.57111 | **−0.00273** |
| RT-063-divall | + `m02_dist::(dv,oc,qd,ab,rs)_` (47 cols) | 198 | 0.58555 | +0.01171 |
| RT-061 | + all of `m02_dist` (59 cols) | 210 | 0.58698 | +0.01313 |

**Why I think they failed.**
- `tl_`: `m00_core` already monitors `tail_hi`, `tail_lo`, `tail_x` and `center` at
  every window length through the same null engine. The only new content in `tl_` is
  the *asymmetry contrast* `P(u>0.95) − P(u<0.05)`, and tail rates are the sparsest,
  noisiest statistics available (at w=64 a single point is 1/64 of the statistic).
  Six extra sparse columns cost more in fit variance than the contrast is worth.
- `rk_`: rank CUSUM and rank variance are individually the *strongest* columns in the
  module univariately (`rk_cusum_u` TS-AUC 0.5413), but they are strongly redundant
  with `m00_core`'s expanding PIT-mean (`exp_z_u`) and PIT-dispersion (`exp_z_u2`)
  surprise channels, which are the same rank information read through a different
  functional. Adding a correlated re-statement of an existing channel is a variance
  cost with no information gain.

**Retry warranted?** Not as standalone families. Both are worth one retest *jointly
with* the divergence family at 5 folds, because the full module (+0.01313) beats the
divergence-only slice (+0.01171) by +0.00142 — consistent with the design story that
tail asymmetry and rank CUSUM are *disambiguators* that only mean something
conditional on a divergence firing, but also fully consistent with single-fold noise.
Not worth spending a 5-fold slot on unless the 47-column slice is being promoted
anyway. Do not resurrect `tl_` in a module that does not already carry `m00_core`'s
tail channels — it would then be measuring something new rather than re-measuring.


---

## agent7 / RT-071C — TREND family (`m03_dyn` cumsum slopes + Mann-Kendall) — NEGATIVE

**Hypothesis.** Recursive and multi-scale trailing-window slopes, slope changes,
acceleration and a robust sign-based (Mann-Kendall / Theil-Sen-substitute) trend
statistic, all calibrated against the per-series historical null, detect breaks
into a drifting or mean-reverting regime that a moment-based module misses.

**What I did.** 12 columns in `m03_dyn`: OLS slope over quarter / half / full
online prefix computed from cumulative sums of `x` and `t*x` (never a regression
per step), each emitted as total drift and as a robust z against the
distribution of the same slope over length-matched historical windows; the two
short-minus-long z contrasts (`dslope_qh`, `dslope_he`) as an
acceleration/slope-change channel; and rolling means of `sign(x_t - x_{t-h})`
for h = 1 and 8 as a cheap robust trend statistic, raw and calibrated.
Screened on `cache/store_screen`, fold 0, seed 0, `n_estimators=300`,
`max_train_rows=300_000`, against a paired `m00_core` control run in the same
session (RT-070C = 0.57384, which reproduces Agent 0's RT-001S exactly).

**The number.** RT-071C: `m00_core` + trend family only = **0.56897**, i.e.
**-0.00487** against the control. Corroborated by RT-072NT: removing the trend
family from the full module *raises* TS-AUC from 0.60381 to **0.60775**
(+0.00394) while removing 12 columns. Two independent experiments, same sign.

**Why I think it failed.** A trailing-window OLS slope is a linear functional of
the window with weights proportional to `(j - jbar)` — a smoothed first
difference. Every level shift, every autocorrelated wiggle and every
variance burst near a window edge moves it, and when the post-break mean differs
from the historical mean the slope is a strictly *weaker*, noisier version of the
mean shift that `m00_core` already measures optimally at six window lengths plus
an expanding window. So the family contributes almost no independent
information while adding 12 columns of noise that dilute LightGBM's
`feature_fraction=0.7` sampling. The slope-change channel that was supposed to
disambiguate transient from persistent (`dslope_qh`, `dslope_he`) turned out to
be the two weakest columns in the entire 60-column module (univariate TS-AUC
0.4986 and 0.4977 — nothing).

An additional mechanism showed up in the H1/H2 decomposition (see
`research/reports/agent07_dynamics.md`): for `mk1` and `mk1_z` the series-level
DGP-fingerprint component (AUC 0.5341 / 0.5440) and the online-movement
component (0.4612 / 0.4687) point in **opposite directions**, so the raw column
is a cancellation of two real but opposing effects. The model cannot exploit
either without a conditioning variable that identifies the DGP.

**Is a retry warranted?** Yes, but conditionally and not by me:
1. Re-test the trend family *jointly with Agent 8's historical-context module*.
   The opposite-sign const/resid structure predicts these columns become useful
   once the model can condition on DGP identity. This is a specific, falsifiable
   prediction.
2. Do not retry the family standalone, and do not retry it with more window
   lengths — the failure is structural (redundancy with the mean channel), not a
   tuning problem.
3. Until then the recommended production configuration is
   `drop_cols=("::drift_", "::dslope_", "::mk")`, which is +0.00394 and -12
   columns versus shipping the full module.

**Not deleted from the code**, so that Agent 0 can re-test on the full 5-fold
protocol before the columns are removed for good; one fold at one seed is not
enough to delete code over.

## agent4 / m04_resid — negative results (2026-08-18, screen protocol, fold 0, seed 0, n_estimators=300)

Paired control in every case: `RT-040C` = `m00_core` alone = **0.57384** (reproduces
Agent 0's number to the last digit). Full context: `research/reports/agent04_residual.md`.

### N1. Redundant raw representation — `RT-045-raw`
- **Hypothesis**: my mean/var/abs/tail/ACF/ACF-of-squares/GLR monitor set, computed on
  the plain robust-normalised series, adds evidence on top of `m00_core`.
- **Did**: `m00_core` + the 8 `m04_resid::raw_*` columns, `keep_regex=r"(m00_core::|m04_resid::raw_)"`.
- **Number**: 0.57169 vs control 0.57384 → **−0.00215**.
- **Why it failed**: `m00_core` already monitors the raw series at six trailing scales
  plus expanding, against the same historical null. My 8 raw columns are a strict subset
  of that information at one scale; the only genuinely new channel is the Gaussian GLR,
  and it is not enough to pay for 8 extra columns of split-search noise.
- **Retry?** No. The raw slice is kept in the module only as the *control arm of the
  ablation*, and it should be the first thing dropped if Agent 0 needs columns back.

### N2. Regularised and robust AR coefficient estimates — `RT-044-arR`, `RT-044-arH`
- **Hypothesis**: historical outliers bias the OLS AR coefficients, so a ridge or
  Huber-IRLS fit gives a cleaner residual and better break detection.
- **Did**: matched 3-column comparison (var, ACF(1)-of-squares, GLR) on identical
  monitors: OLS AR(2), ridge AR(3) with λ=0.05 on a unit-variance Gram, Huber-IRLS AR(2)
  with 3 fixed reweighting steps.
- **Numbers**: OLS AR(2) **0.53089**, ridge AR(3) 0.52702 (−0.0039), Huber AR(2) 0.52658
  (−0.0043). Both below plain OLS; the two are tied with each other.
- **Why it failed**: the premise is backwards for *detection*. A slightly worse-fitting
  AR filter whitens less and therefore leaves more break signal in the residual. Cleaning
  up the coefficients makes the filter better at removing exactly the structure we are
  hunting. Robustness helps estimation, not detection.
- **Retry?** Not as a power play. Both are worth keeping only as a 6-column robustness
  control, and are the first cut if the budget tightens.

### N3. GARCH(1,1) volatility normalisation — `RT-044-volG`
- **Hypothesis**: normalising by a properly fitted conditional-variance model gives the
  cleanest possible innovation stream and therefore the sharpest break evidence.
- **Did**: variance-targeted GARCH(1,1), deterministic 12-point (α,β) QMLE grid fitted on
  the historical segment, filtered causally forward; same 3 variance-flavoured monitors.
- **Number**: **0.50012** — no signal whatsoever. Compare robust EWMA halflife 63 at
  0.53490 and EWMA halflife 22 at 0.51534 on the identical 3 columns.
- **Why it failed**: this is the textbook over-whitening failure and it is total. A GARCH
  filter's whole purpose is to track a change in conditional variance. When the structural
  break *is* a variance change, the filter follows it within a few tens of points and the
  normalised residual returns to unit variance — the break is erased by construction.
  Power falls monotonically with the filter's adaptation speed.
- **Retry?** Not as a detector. Retained (3 columns) purely as a *contrast partner*:
  `raw_e_var` high with `volG_e_var` ≈ 0 identifies "the break was a conditional-variance
  break a GARCH can follow", while both high identifies a break in the innovation
  distribution itself. **General lesson for every agent: never let a filter adapt on the
  same timescale and in the same channel as the break you are hunting. Volatility filters
  used for break detection should be deliberately slow and robust.**

### N4. AR(1) / AR(2) residuals as standalone detectors — `RT-043-ar1`, `RT-043-ar2`
- **Number**: AR(1) 0.53360, AR(2) 0.53624 vs the raw 8-column slice 0.54305
  (−0.0095, −0.0068 on identical folds, seeds and monitors).
- **Why**: a level shift, which in raw space displaces every trailing and expanding
  window persistently, collapses in AR-residual space into a one-off spike of size
  (1−Σφ)·Δμ at t=τ and then reverts. The expanding-mean null-z decays like 1/√L after the
  break instead of growing like √L. AR-only detectors lose the entire mean-shift family.
- **IMPORTANT — this negative result does NOT generalise, and reading it as "don't
  whiten" would be a mistake.** Conditional on `m00_core` already covering the raw
  channel, the same 8 AR(2) columns are worth **+0.021** and the 8 AR(5) columns
  **+0.023** (`RT-045-ar2`, `RT-045-ar5`), while the 8 raw columns are worth −0.002.
  The standalone ablation measures marginal power; the incremental ablation measures
  information. They rank the representations in opposite orders. Screen incremental,
  not standalone.

---

## agent8 — `m05_ctx` historical-context block as raw features (RT-A08-H1 / RT-A08-H2 / RT-A08-PERM)

**Hypothesis (H1).** The break-free historical segment alone predicts break
occurrence — some DGP families are more break-prone, which would shift a whole
series' score level and therefore move TS-AUC.

**What I did.** Built `m05_ctx` (50 series-constant columns: robust moments,
skew/kurtosis, Hill tail indices, quantile spacings, spacing entropy, AR(3) +
PACF, ACF of squares/abs, variance ratios, ADF-flavoured statistic, DFA,
spectral band powers / entropy / dominant frequency / centroid, Haar wavelet
energies, recent-100/250/500 vs whole-history ratios and differences, `n_hist`).
Prefix invariance passes bitwise (atol=0.0) on 8 series. Screen store, fold 0,
300k train rows, seed 0, `n_estimators=300`, paired against my own
`m00_core`-only control run in the same session.

**Numbers.**
- `m05_ctx` alone: **TS-AUC 0.50143** (control 0.57384). Chance.
- Series-level LightGBM on the 2,500 context vectors → `has_break`, permanent
  series-level folds: **AUC 0.5068**, versus a label-permuted null of
  0.5063 / 0.5155 / 0.5163. Inside the null.
- Break rate per KMeans DGP cluster (k=6): 0.5055 / 0.5114 / 0.4785 / 0.5049 /
  0.4908 / 0.5357 vs global 0.4960. Flat.
- `m00_core + m05_ctx`: 0.57476, **delta +0.00092** — inside noise.
- `m00_core + m05_ctx_perm` (context deranged within fold): 0.51662,
  **delta −0.05723**.

**Why it failed.** Two distinct reasons and they matter separately.
1. *H1 has no substrate.* The 2026 generator appears to assign breaks
   independently of the historical DGP, so there is simply no series-level prior
   to learn. Three independent tests agree. This is a property of the data, not
   of the features — no encoding of history will fix it.
2. *The raw-column encoding is actively harmful.* Fifty near-continuous,
   series-constant columns give every training series a near-unique signature.
   LightGBM splits on it and memorises series-level idiosyncrasy; the
   permutation control exposes this by scoring −0.057 below the control, i.e.
   far worse than having no context at all. The +0.058 "true beats permuted"
   gap is therefore mostly *damage avoided*, not *value added*, and reading it
   as a win would have been a serious error.

**Also negative:** the "recent-history features are unexpectedly useful" claim
from public 2026 work did not reproduce as a standalone break prior. The best
recent-history column, `h_r500_acf1_diff`, reaches series-level AUC 0.4710
(|d| = 0.029) — one marginal hit out of 50 columns tested, which is what chance
predicts. `n_hist` is inert (AUC 0.4887 univariate, 0.4948 as a classifier), as
are series id, store offset and row order (all 0.4920, AUC 0.4988 jointly).
`n_online` was measured for diagnosis only (0.5132, also inside the null band)
and is NOT emitted.

**Retry warranted?** Not as a feature block — no. The same information used as a
6-way *gate* selecting per-DGP specialist models is worth **+0.03248 TS-AUC at
equal tree capacity** (+0.04578 over a permuted-cluster control); that positive
result is in `research/reports/agent08_context.md` and is what should be pursued.
Anyone tempted to add a large block of series-constant columns to any model
should run the within-fold derangement control first — it is cheap and it caught
this.

---

## AGENT 12 — `m07_bayes` (Bayesian / sequential): no negative headline, three honest negatives inside it

The module itself is a **positive** result (screen fold 0: standalone 0.59916 vs
the 0.57384 m00_core control, stacked 0.61352; `research/reports/agent12_bayesian.md`,
experiments `RT-120A/B/C`). What belongs here are the things that did **not**
work along the way and the claims I could **not** substantiate, so nobody spends
budget rediscovering them.

**1. Dense null grids bought nothing and cost 3x.** The first working version
calibrated against an 8-window `_AddNull` grid `(4,8,…,512)` with 1,400 null
draws per window, a 320-step restart-block grid at stride 20, and `R_MAX = 100`
with an 2,000-point historical BOCPD pass. That version cost **225 ms/series**,
nearly 3x the 80 ms budget. Cutting to a 5-window factor-3 grid, 500 draws,
stride 32 / 256 positions, `R_MAX = 80` and an 800-point historical pass took it
to **73 ms/series**. I did not screen the expensive version — the point is that
the cheap version already produces the headline number, so the fine calibration
grid is not where the signal is. Anyone tempted to spend compute on denser
historical nulls should assume it is wasted until shown otherwise.

**2. `np.quantile` / `np.median` inside a per-series feature module is a trap.**
Roughly 40 % of the original runtime was `np.quantile` calls computing a median
and an IQR on arrays that had *already been sorted* one line earlier. Indexing
the sorted array directly (`s[n//2]`, `s[n//4]`, `s[3n//4]`) is bitwise-identical
for the median at odd length and close enough for an IQR scale, and it removed
~30 ms/series. This is a general note for every module author on this box.

**3. Claims I could not substantiate and am not making.** (a) I have **no**
per-column or per-block attribution for the 0.599 — `sbr.pipeline.run` does not
persist feature importance under `screen=True`, and I chose not to spend a
sixth 300k-row training on it under a load-8 box. So I cannot tell you whether
the absorbing posterior, BOCPD, the e-processes or simply the AR(6)+normal-scores
stream is doing the work, and the report says so. (b) The location mixture
family carries 20 % of the prior mass on the strength of "keep it, but not
dominant"; the forensics puts location-break AUC at 0.4998, so this mass is
probably dead weight, but I did not measure it. (c) Everything is one fold. A
50-column block beating a 151-column block standalone is unusual enough that the
*magnitude* should not be quoted until it survives the 5-fold protocol.

**Retry warranted?** Not for the negatives above — (1) and (2) are settled.
For (3) the ablation is cheap and worth someone's next hour: four runs dropping
one block at a time, plus one run of the AR(6)+normal-scores stream fed through
`m00_core`-style windowed calibration with none of the Bayesian machinery. If
that last one recovers most of the 0.599, the interesting finding is the stream,
not the recursions, and it transfers to every other module for free.

## agent0 / RT-110, RT-111 — pairwise-t ranking objective FAILED TO PROMOTE

**Hypothesis.** TS-AUC is a within-timestep cross-sectional ranking metric, so a
pairwise logistic loss whose pairs are drawn within the same online index `t`
should beat binary logloss. Agent 9 measured exactly that on the screen store
(211 features, fold 0): pairwise 0.61395 vs binary control 0.60677, **+0.00718**.

**What happened on promotion.** On the champion feature set (500 columns) and the
full store, with the *identical* 800k-row materialised matrix for both losses so
that the only difference is the objective (RT-111, fold 0):

| objective | TS-AUC |
|---|---|
| binary logloss | 0.62631 |
| pairwise logistic, groups = online index `t`, 8 neg/pos | 0.62302 |
| **delta** | **−0.00330** |

**Why it probably failed.** The screen store has 2,000 training series; the full
store has ~6,400. The pairwise loss trades away calibration for ordering, which
is worth something when the ranking signal is scarce and the model is
data-starved, and worth nothing once there is enough data for the pointwise model
to learn the ordering anyway. The pairwise sampler also only ever uses positives
that share a timestep with a negative, discarding rows the pointwise loss keeps.

**Retry warranted?** Yes — and this entry has since been **partly retracted**.

### AMENDMENT (RT-123, five folds)

Running the pairwise objective properly over all five folds gives **0.61450**
against the binary champion's 0.61510 — on **700k training rows versus the
champion's 1M**. That is a dead heat, not a −0.0033 loss. The single paired
fold-0 comparison that produced the −0.0033 was itself inside the noise: the same
objective scored 0.62302 there and 0.63073 on the same fold in the five-fold run,
a 0.008 swing from nothing but the row sample and the pair draw.

**The honest conclusion is that the two objectives are equivalent within noise on
this data, and neither the screen-level +0.0072 nor my −0.0033 was real.** The
methodological lesson is aimed at me, not at the objective: **one fold is not a
result, however well paired it is.** A per-fold spread of 0.011 means any
single-fold delta below ~0.01 is unresolvable, and I promoted a conclusion from
one anyway.

The pairwise model is now a permanent ensemble member (`RT-123`); its
within-timestep rank correlation with the champion is 0.780.

**Process note.** The first attempt (RT-110) compared pairwise-at-800k against
the champion-at-1M and would have reported −0.006; that number was not a paired
test and was discarded in favour of RT-111. Screen-level objective results in
general did not survive: lambdarank +0.0005, XE-NDCG −0.014, scale_pos_weight=4
−0.009, soft-ramp target −0.012, log-hazard regression −0.018, all measured
against a binary control on the screen store.

## agent0 / RT-150 — DGP-cluster GATED SPECIALISTS failed to promote

**Hypothesis.** Agent 8 measured that the historical-context vector is worth ~0 as
features but **+0.032 as a router**: KMeans the series by their historical
characterisation, train a specialist per cluster. Routing is architectural, so it
should compose with everything else rather than compete with it.

**What happened at full scale.** k=6, gate fitted on training-fold series only,
every arm sharing one materialised matrix per fold so the comparison is exact:

| arm | fold 0 | fold 1 | mean |
|---|---|---|---|
| global (600 trees) | 0.62291 | 0.61012 | **0.61652** |
| gated, equal total capacity (6 x 100 trees) | 0.60617 | 0.59762 | 0.60190 |
| gated, equal per-model capacity (6 x 600) | 0.60080 | 0.58823 | 0.59452 |
| rank-blend of global + gated | 0.62224 | 0.60619 | 0.61422 |
| **permuted-cluster control** (6 x 600) | 0.58606 | 0.58186 | 0.58396 |

**gated − global = −0.0219** (−0.0219 and −0.0219 on the two folds independently).
**gated − permuted = +0.0064.**

**Why it failed, and why the screen result was not wrong.** The clusters do carry
real information — gating beats its own permutation control on both folds. But
partitioning 6,400 training series into six groups costs far more than the routing
gains. Agent 8's +0.032 was measured against a **151-column** global model on
**2,000** screen series; that global model could not express the conditioning
itself, so an explicit router supplied it. With 500 columns and 600 trees on
6,400 series, the global model already learns the same interactions, and the data
split is pure loss. Note the largest cluster holds ~47 % of series while the
smallest holds ~3 % — the small specialists are badly data-starved.

**Retry warranted?** Only as an ensemble member (the gated model makes structurally
different errors and was never measured for blend delta), or with soft gating that
shrinks each specialist toward the global model instead of replacing it. Not as an
architecture.

**THE PATTERN — worth stating plainly.** This is the **third** screen-level win to
reverse at full scale, after the context block (+0.0009 → −0.0177) and the
pairwise-t objective (+0.0072 → −0.0033). All three are the same failure mode:
they help a **data-starved** model and stop helping once the model is not
data-starved. Meanwhile every *feature-addition* screened on the same store has
transferred. Operating rule for the rest of this project: **the screen store is
valid triage for new features, and is not evidence for objectives, architectures,
or anything that changes how the training data is partitioned.** Those must be
tested at full scale from the start.

## agent0 / RT-131 — ENSEMBLE WEIGHTING AND SUBSET SELECTION both fail

With seven streams available, the obvious next move is to weight them, or to keep
only the good ones. Both lose to doing nothing:

| blend of 7 streams | pooled OOF TS-AUC |
|---|---|
| **equal-weight rank average (all seven)** | **0.62524** |
| best subset, chosen in hindsight on the reported score | 0.62556 (+0.0003, not a real option) |
| leave-one-fold-out greedy forward subset selection | 0.62442 (**−0.0008**) |
| leave-one-fold-out logistic stack | 0.62514 (−0.0001) |
| leave-one-fold-out LightGBM stack | 0.62144 (−0.0038) |

The hindsight-best subset beats the plain average by 0.0003 — and when subset
choice is made honestly (fitted on four folds, scored on the fifth) it *loses* by
0.0008. The gap between those two numbers is the size of the self-deception on
offer.

**Why.** Seven streams within 0.010 of each other, and a per-fold spread of 0.011,
means the fold-to-fold ranking of streams is unstable. Any weighting or selection
scheme is fitting that instability. The equal average is the estimator with no
variance in its parameters because it has no parameters.

**Consequence for the submission:** the blend rule is "average the within-timestep
rank percentiles of every stream you have" — nothing to tune, nothing to leak,
and one fewer thing that can silently overfit between now and the deadline.

---

## WAVE 3 / W3-A1 `m08_chan` — CANNOT BE VERIFIED FROM THIS REPOSITORY

`BRIEF_CLAUDE_wave3_alpha.md` section 4 and `HANDOFF_WAVE3.md` section 5 both
report a completed and rejected wave-3 experiment: a transformed detector bank
(`m08_chan`, CUSUM / Page-Hinkley / Shiryaev-Roberts over six channels, 72
columns) with a matched ABL delta of **−0.00073**, a deployable ensemble delta
of +0.00023, and a standalone of 0.61201, under experiment IDs `RT-301`/`RT-302`.

**None of it exists in the repository.**

| claimed artifact | actual state |
|---|---|
| `src/sbr/features/m08_chan.py` "is on disk" | absent from the working tree **and** from every commit of every branch |
| write-up "in `FAILED_EXPERIMENTS.md`" | no such section existed in this file before this one |
| `RT-301` / `RT-302` rows | absent from `research/RESULTS.csv`; the string `RT-30` appears nowhere in `research/` in any commit |
| pre-registration "in `RDOF_LEDGER.md`" | absent |

The result may well be exactly as described — the mechanism and the numbers are
plausible and internally consistent — but it is prose, not evidence, and under
`VALIDATION_V2.md` it cannot be cited. Two consequences, both acted on:

1. **The IDs `RT-301`/`RT-302` are treated as unallocated** and are used in wave 3
   for the backward-CUSUM experiment, pre-registered in `RDOF_LEDGER.md`.
2. **The derived advice is not inherited.** The brief's recommendation that "only
   `sgn·sgn` and `e_t e_{t-1}` are worth isolating" rests on an ablation nobody
   can inspect. Those two channels are neither privileged nor excluded; if a
   dependence-channel experiment is run it gets its own pre-registration and its
   own matched control.

**Lesson, and it is the same lesson as the 66 `nogit` rows.** A result that lives
only in a handoff document is not a result. The ledger row, the OOF artifact and
the code are what make a negative result reusable — without them the next agent
either repeats the work or, worse, trusts it.

---

## WAVE 3 / RT-302 — `m09_back` backward suffix-vs-prefix contrast — **NOT PROMOTED**

**Hypothesis (pre-registered in `RDOF_LEDGER.md` before the run).** The metric
loses its mass on late breaks. A *two-sample* contrast of the last `k` online
points against the earlier online points,

    D_k(t) = ( M_suffix(k) - M_prefix(t+1-k) ) / sqrt( sd_null(k)^2 + sd_null(t+1-k)^2 )

maximised over `k`, should see a late break that `m00_core`'s *one-sample*
trailing-window bank cannot, because the one-sample form carries the series'
persistent online-vs-historical offset in every window while the contrast
differences it out.

**What I built.** `src/sbr/features/m09_back.py`, 51 columns, six channels
(level, robust scale, PIT, AR-residual level, innovation scale, lag-1 product),
`k` grid (4, 8, 16, 32, 64, 128, 256) fixed a priori from `m00_core`'s grid with
no tuning. 64 ms/series. Bitwise prefix-invariant (atol=0.0) on 10 series
including the shortest in the dataset. Per-column audit clean: the `d8/d32/d128`
columns sit at median ~0, sd ~1, which is what a correctly studentised
two-sample statistic should look like.

**The numbers.** Matched ABL protocol, 5 canonical folds, 400k training rows,
seed 0, identical parameters, same session.

| arm | mean OOF | per fold |
|---|---|---|
| `RT-301` control, 7 modules | 0.61257 | 0.62567 / 0.60719 / 0.62018 / 0.60506 / 0.60475 |
| `RT-302` + `m09_back` | 0.61413 | 0.62880 / 0.61113 / 0.62178 / 0.60843 / 0.60049 |
| **delta** | **+0.00156** | +0.00313 / +0.00394 / +0.00160 / +0.00337 / **−0.00426** |

Positive on 4 of 5 folds, delta sd 0.00301 (1.9x the mean). It therefore
**survives its own falsification conditions** — and fails every promotion bar.

**1. The mechanism is falsified by its own diagnostic.** TS-AUC by post-break age,
the measurement the brief asks for precisely because that is where the mass is
lost:

| post-break age | RT-301 | RT-302 | delta |
|---|---|---|---|
| 0–5 | 0.51278 | 0.51186 | **−0.00092** |
| 5–10 | 0.52594 | 0.52560 | **−0.00035** |
| 10–20 | 0.54279 | 0.54063 | **−0.00215** |
| 20–50 | 0.56866 | 0.56564 | **−0.00302** |
| 50–100 | 0.59150 | 0.59252 | +0.00102 |
| 100+ | 0.64569 | 0.64917 | +0.00349 |

The module makes **young breaks worse and mature breaks better** — the exact
opposite of what it was built for. All of the aggregate gain comes from the 100+
bucket, which holds 706k of the ~1.03M post-break rows. Feature importance says
the same thing: `m09_back` takes 1.25 % of total gain from 9.3 % of the columns,
and six of its top eight columns are the `_pre` family — the prefix-vs-history
*nuisance* term included as a conditioning variable — not the suffix-vs-prefix
contrast that was the hypothesis. What got built is a lagged expanding-drift
channel, not a late-break detector.

**2. The bootstrap route to promotion is closed.** Paired series-level bootstrap,
300 replicates: CI **[−0.00190, +0.00494]**, median +0.00156, 81.3 % of replicates
favouring the treatment. The CI straddles zero, so it is not "materially
favourable" under VALIDATION_V2 section 7.

**3. The ensemble route is closed by its own negative control — and this is the
part worth remembering.** The deployable logit blend `RT-301`+`RT-302` scores
0.61627, **+0.00371** over the control, which is the alternative promotion route.
So `RT-303` was pre-registered and run: the identical 7 modules at the identical
protocol with **seed 1** — a stream containing no new information at all.

| deployable logit blend | mean | delta vs `RT-301` |
|---|---|---|
| `RT-301` + `RT-302` (+ m09_back, 51 new columns) | 0.61627 | +0.00371 |
| **`RT-301` + `RT-303` (seed clone, zero new information)** | **0.61753** | **+0.00496** |

**The null-information clone blends better than the new module does**, by 0.00126.
The ensemble delta was generic two-model variance reduction the whole time.

Worse for the diversity argument: within-timestep rank correlation with the
control is **0.8219** for `m09_back` and **0.7846** for the seed clone. *Changing
the random seed decorrelates more than adding 51 columns of new statistics did.*

**LESSON — the one to carry forward.** A deployable ensemble delta is not
evidence for a candidate unless it beats a same-strength stream that is known to
contain nothing new; and low within-timestep rank correlation with the champion
is **not** a diversity credential, because a seed change buys more of it than a
new feature family does. Every future "this stream is decorrelated and the blend
gains" claim in this project must carry a seed-clone control. Wave 2's seven
streams were never measured against one.

**Retry warranted?** Not as a late-break specialist — that hypothesis is dead in
this form. The `_pre` family (a lagged expanding mean, calibrated at matched
length) is the only part that earned its gain and is worth **6 columns**, not 51,
tested against a control that already contains `m00_core`'s expanding block. The
suffix-vs-prefix contrast itself should not be retried by widening the `k` grid:
the failure is that the pre-break online prefix is a *worse* reference than
history for exactly the young breaks it was meant to help, because at small
post-break age the suffix is short and the contrast is dominated by prefix noise.

**Artifacts.** `research/reports/wave3_RT-301_vs_RT-302.json`,
`research/reports/wave3_ensemble_control.json`, OOF vectors
`research/oof/RT-30{0,1,2,3}.npy` (gitignored by repo policy — regenerate with
`research/scripts/wave3_queue_a.py`).

---

## W4-E6 — UNION OF SPECIALISTS AND SEED CLONES — **REJECTED**

**Pre-registered** in `RDOF_LEDGER.md` before scoring, with the provenance
stated: the hypothesis was chosen *after* W4-E1 reported, so the bar carried an
extra bootstrap condition.

**Hypothesis.** W4-E1 decomposed the seven-stream ensemble's gain into +0.00559
from ordinary bagging and +0.00417 from specialist diversity. The deployed
champion harvests the second and only incidentally the first — each
*configuration* appears exactly once. If the two effects are even partly
additive, blending both arms should beat either.

**Implementation.** Zero new training. The union of the two W4-E1 arms,
deduplicated on the shared member `RT-300`: 13 boosters, same cross-fitted SCDF
calibration, same folds, same equal-weight mean.

**Control.** The incumbent seven specialists, measured in the same run.

| composition | n | mean OOF | per fold |
|---|---|---|---|
| `RT-420` specialists | 7 | **0.62581** | 0.63828 / 0.62040 / 0.63392 / 0.61750 / 0.61894 |
| `RT-421` seed clones | 7 | 0.62164 | 0.63817 / 0.61667 / 0.63020 / 0.61437 / 0.60879 |
| `RT-422` union | 13 | 0.62486 | 0.63967 / 0.61973 / 0.63303 / 0.61673 / 0.61513 |

**Result: −0.00095, positive on 1 of 5 folds**, paired series bootstrap
−0.00094 with 95% CI [−0.00202, +0.00007], 4% of replicates positive. Every
pre-registered condition failed. **The gains do not stack — they anti-stack.**

**Mechanism.** The union is not "specialists plus bagging". Under an
equal-weight mean it is "specialists with the champion configuration up-weighted
seven-fold": 7 of the 13 members are the `RT-100R` configuration at different
seeds, so that configuration carries 54% of the blend weight instead of 14%.
The ensemble's value comes from averaging models that are wrong in *different*
directions, and this composition spends more than half its weight on one
direction. The bagging gain measured in W4-E1 was real, but it was measured on
an arm where nothing else competed for weight.

**What this does NOT license.** Weight-tuned variants, `k` clones per
specialist, or any best-of composition. Those are precisely the lattice W4-E6's
pre-registration excluded, and searching it after a negative result is how a
null becomes a false positive. The mechanism above is recorded as an
explanation, not as the seed of a follow-up.

**What it does establish.** The incumbent seven-stream composition is not
merely adequate, it is *better than the obvious enrichment of it*. That is a
stronger position for the champion than W4-E1 alone gave it.

---

## WAVE 5 / C2 — TWO FAMILIES REJECTED BEFORE ANY TRAINING RUN

Both were named as NOT RUN alpha families in `STATE_OF_RESEARCH_V4.md` §12.
Neither was built. Both were killed on replicated synthetic mechanism evidence
plus a redundancy measurement, at a cost of zero training runs and zero TS-AUC
values. Full audits: `research/reports/wave5_m04_redundancy_audit.md`,
`research/reports/wave5_m07_redundancy_audit.md`.

### W5-R1 — RESIDUAL CUSUMSQ / cumulative squared-innovation evidence — **REJECTED**

**Hypothesis.** Cumulative squared-residual evidence, in the scale-free
Brown–Durbin–Evans bridge form `C_k/C_t − (k+1)/(t+1)` maximised over `k`,
captures persistent variance and conditional-variance breaks *more cleanly* than
`m04_resid`'s recent-window summaries, and localises where the variance changed.

**What was measured** (causal prototype, 7 synthetic DGP families with per-series
AR coefficients, series-level statistics, 40 series per arm, 2,000-replicate
bootstrap):

| | |
|---|---|
| redundancy of the bridge magnitude, ridge R² on `m04` alone | **0.9358** |
| redundancy on `m04` + `m11_focus` | 0.9585 |
| permanent variance shift vs temporary spike — incumbent `cmb_e_abs` | **|d| = 1.979** |
| — CUSUMSQ bridge magnitude | 1.142 |
| — CUSUMSQ change-age | 0.054 |
| bootstrap, magnitude − incumbent | **−0.870, 95% CI [−1.497, −0.231]** |
| bootstrap, age − incumbent | **−1.835, 95% CI [−2.369, −1.306]** |

**The incumbent wins on the candidate's own target mechanism**, with confidence
intervals excluding zero, and 94% of the magnitude channel is already determined
by `m04_resid` alone.

**Why it failed.** The bridge's scale-freeness was meant to be its virtue and is
its defect. Normalising `C_k/C_t` discards the overall scale *level* — and the
level is exactly what separates a permanent 2× variance shift from a 12-point 4×
spike. Both are inhomogeneous, so both look alike to a bridge. `cmb_e_abs`
(expanding mean of |residual| after AR + EWMA normalisation, null-calibrated)
measures sustained scale elevation directly and is roughly twice as good at it.

**Two distinct failures, not one.** The magnitude channel is *redundant*
(R² = 0.94). The change-age channel is genuinely non-redundant (R² = 0.34) and
was rejected for a different reason: it carries almost no signal on its own
contrast (|d| = 0.054, indistinguishable from zero). Non-redundant and
uninformative are different things and this family managed one of each.

**Retry warranted?** Not in this form. Any future variance-localisation attempt
must beat `cmb_e_abs` on the permanent-vs-temporary contrast *before* it is
built, and must not normalise away the scale level.

### W5-R2 — ABSORBING-STATE BOCPD (`m14_absorb`) — **REJECTED, ALREADY EXISTS**

**Hypothesis.** Textbook BOCPD models "change now / run length", while the
competition asks whether a *persistent* break has already occurred, so an
absorbing PRE → POST hidden state with no return transition is the right
formulation — emitting `P(post by t)`, posterior change age, entropy,
hazard-weighted evidence, posterior persistence, predictive likelihood ratio,
max posterior tau and tau-posterior concentration.

**What the audit found.** `m07_bayes` is titled *"generative / sequential
Bayesian evidence for an **ABSORBING** break"* and its first construction bullet
is the absorbing recursion, exactly:

    L_t(th) = logaddexp(L_{t-1}(th), log h) + llr_t(th) - log(1-h)

with the module's own note that *"there is no dyadic approximation and no
candidate-changepoint scan: the absorbing structure integrates over every
changepoint exactly."*

**Nine of nine proposed outputs already exist** — `ab_lpo` (P(post by t) as log
posterior odds), `bo_mean_rel` and the mode channel (change age), `bo_ent`
(entropy), the hazard terms inside the recursion, `ab_post_lvr` / `ab_post_rho` /
`ab_fast_slow` (persistence), `bf0_*` and `ev_*` (predictive likelihood ratios),
`bo_p_lt10` / `bo_p_lt25` (tau concentration). Integration over tau exists in a
**stronger** form than the proposed max.

The proposal's premise — that BOCPD's resettable state is wrong for an absorbing
break — is an argument `m07_bayes` already makes and resolves, by carrying both
models deliberately: *"unlike the absorbing model it allows repeated resets, so
the two disagree exactly on transients."* That disagreement is already emitted.

**Cost of this decision: one file read.** Building it would have been the exact
failure mode that produced `tl_` (−0.00410) and `rk_` (−0.00273) — re-stating an
existing channel under a new name and paying the split-search variance cost.

**Retry warranted?** No. The only open item in this area is the per-block
attribution ablation of `m07_bayes` that this file already records as unspent,
and that is an ablation of an existing module, not a new family.
