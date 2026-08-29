
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

## W5-E1 — SMALL BAGGING COMPONENT IN THE SPECIALIST ENSEMBLE — **REJECTED**

**Pre-registered** in `research/WAVE5_PREREG.md` §6 at commit `5488644`, before
any wave-5 number existed. Grid fixed at four λ values and **not enlarged after
it returned a null**.

**Hypothesis.** W4-E6 rejected the 13-booster *union* and diagnosed why: under an
equal-weight mean the union hands the `RT-100R` configuration 54% of the blend
weight instead of 14%. A *small* bagging component does not do that — λ=0.90
gives it 22.9% — so W4-E6 does not answer whether a modest admixture helps.

**Implementation.** No training. `S` and `B` are the two W4-E1 arms, each
cross-fitted through `SCDF_NSEEN` before mixing, then `λ·S + (1−λ)·B`.

| λ | TS-AUC | Δ vs λ=1 | folds better | `RT-100R` weight |
|---|---|---|---|---|
| 1.00 | **0.62581** | — | — | 14.3% |
| 0.90 | 0.62577 | −0.00004 | 3/5 | 22.9% |
| 0.80 | 0.62564 | −0.00017 | 3/5 | 31.4% |
| 0.70 | 0.62541 | −0.00040 | 1/5 | 40.0% |

Paired series bootstrap on the best λ<1, 200 replicates, common random numbers:
**−0.00003, 95% CI [−0.00026, +0.00018], 36% of replicates positive.**

**Why it failed, and why the failure is informative.** The delta is **monotone
decreasing in bagging weight** — there is no interior optimum and no threshold
effect. W4-E6's mechanism (54% is too much weight on one configuration) is
therefore an incomplete explanation: **any** admixture of exchangeable seed-clone
mass into a heterogeneous blend is neutral-to-harmful, at every weight tested.
The two gains W4-E1 separated are not partially additive at small doses either.

The CI is tight around zero, so this is a **clean null, not an underpowered
test**. Equal weighting over the seven specialists now stands on two independent
experiments.

**Retry warranted?** No. The grid spans the region where a benefit could
plausibly hide and the trend through it has one sign. Searching λ ∈ (0.90, 1.00)
after seeing this is exactly how a null becomes a false positive.

---

## W5-E9a — TS-AUC-SHAPED PAIR WEIGHTING (`pairwise_w`) — **REJECTED**

**Pre-registered** in `research/WAVE5_PREREG.md` §6.

**Hypothesis, and it is still arithmetically correct.** The official metric pools
concordant pairs over `Σ_t n_pos(t)·n_neg(t)`, so every within-timestep
(positive, negative) pair counts equally. The incumbent `pairwise_t` objective
(stream `RT-123R`/`RT-413`) samples a fixed `m_neg = 8` negatives per positive
row, which makes every positive ROW count equally instead. W5-D1 measured how
large the mismatch is: timesteps with ~7,000 series alive get the same weight as
timesteps with ~1,200.

**The dispatch was verified before the arms were compared.** `RT-702` routes the
*incumbent* objective through the same wave-5 hook and reproduces `RT-413`'s
ledger row to five decimals on **all five folds**, so the arms differ only in
the loss and not in the mechanism that delivers it.

| arm | id | TS-AUC | per fold |
|---|---|---|---|
| incumbent | `RT-702` | **0.61481** | 0.63185 / 0.61326 / 0.62455 / 0.60031 / 0.60408 |
| `n_neg(t)`-weighted pairs | `RT-700` | 0.61334 | 0.63270 / 0.60824 / 0.62348 / 0.60039 / 0.60191 |
| delta | | **−0.00147** | +0.00085 / −0.00502 / −0.00107 / +0.00008 / −0.00217 |

**Positive on 2 of 5 folds. Rejected.**

**Why it failed.** Offered as a hypothesis, not a finding: the weighting pushes
gradient onto the timesteps with the most alive series, which D1 places at
t ≈ 200–700 — and those timesteps are also the *easiest* and already the
best-ranked, while a positive row's gradient variance rises with its weight. The
incumbent's flat `m_neg` acts as an implicit importance weighting toward the
sparse late timesteps, and it earns its keep. **Matching the evaluation's
weighting is not the same as spending training capacity well.**

**Retry warranted?** Not in this form. A version that weights by `n_neg(t)` while
*capping* the weight, or that spends the extra pairs on late timesteps instead,
is a different experiment and would need its own pre-registration; nothing here
licenses tuning `m_neg` or the weight exponent against these folds.

---

## W5-E2 — `m10_persist`, OUTLIER-DRIVEN vs BULK SCALE CHANGE — **REJECTED**

**Pre-registered** in `research/WAVE5_PREREG.md` §6. Its *design* was chosen
after the W5-D2 forensics reported, which is stated in the ledger — the bar was
not moved for it.

**Hypothesis, and it was well aimed.** W5-D2 found the champion's highest-ranked
no-break series are dominated by heavy-tail/outlier and variance-burst mechanisms
(`tail_rate_online` +1.89 IQR in the top 1%; heavy-tail 40% of that group against
a 25.9% base rate). Nothing in the incumbent bank computes a trimmed statistic,
an energy-concentration statistic or an exceedance run length, so a scale
excursion driven by two points and one driven by the bulk arrive at the booster
looking alike. The load-bearing column was the contrast `*_gap` = calibrated
untrimmed scale − calibrated trimmed scale.

| arm | id | TS-AUC | per fold |
|---|---|---|---|
| control | `RT-301` | 0.61257 | 0.62567 / 0.60719 / 0.62018 / 0.60506 / 0.60475 |
| + `m10_persist` | `RT-740` | 0.61459 | 0.62401 / 0.61363 / 0.62767 / 0.60776 / 0.59986 |
| delta | | +0.00202 | −0.00166 / +0.00644 / +0.00749 / +0.00270 / −0.00489 — **3/5** |

**And the comparison that decides:**

| blend | TS-AUC | vs control |
|---|---|---|
| `RT-301` + `RT-303` (seed clone, no information) | 0.61753 | +0.00496 |
| `RT-301` + `RT-740` | 0.61713 | +0.00456 |
| **candidate − seed clone** | | **−0.00041**, 2/5 folds |

**It loses to a seed clone.** The whole blend gain, and more, is ordinary
variance reduction.

**Why it failed — and the answer was already in the repository.** The wave-1
taxonomy measured exactly the right thing and I did not weight it heavily enough
when designing this: **17.0% of no-break series contain a break-lookalike
transient in their online segment, and 15.6% of equally long break-free
HISTORICAL windows contain one too.** Transients are a property of the DGP, not
of the online period. So a per-series historical null — which every column in
this project is already calibrated against — has *already* priced the series'
own propensity to throw outliers. Measuring the outlier-vs-bulk split more
sharply does not help, because the champion was never confused about which
series are outlier-prone; it is confused about the same series the null is.

The age profile confirms the mechanism did not fire where it was aimed:

| age | Δ vs control |
|---|---|
| 0–5 | **−0.00094** |
| 5–10 | **−0.00075** |
| 10–20 | **−0.00398** |
| 20–50 | −0.00373 |
| 50–100 | +0.00270 |
| 100+ | +0.00427 |

All of the aggregate gain is at 50+ and it makes ages 0–50 worse — the same
shape as `m09_back`, which is what a module that is adding smoothing rather than
discrimination looks like under this metric's weighting.

**Retry warranted?** Not against no-break tails. The evidence says the residual
false-positive problem is not "the model cannot tell an outlier from a scale
break" but "for these series the historical null and the online segment are drawn
from distributions that genuinely overlap". A different attack — for example
conditioning the null on the series' own tail index rather than sharpening the
online statistic — would be a new hypothesis and needs its own pre-registration.

---

## W5-E9b — WEIGHTED SQUARED HINGE (`pairwise_h`) — **REJECTED**

Same pairs and same `n_neg(t)` weighting as W5-E9a; only the loss shape changes.
`HINGE_LR_MATCH = 0.25` scales the gradient so that at margin `d = 0` the squared
hinge matches the logistic's per-pair gradient — without it the arm would have run
at 4× the effective learning rate and the comparison would have measured step
size, not loss shape.

| arm | id | TS-AUC |
|---|---|---|
| incumbent `pairwise_t` | `RT-702` | 0.61481 |
| `pairwise_h` | `RT-701` | 0.61005 |
| delta | | **−0.00476** |

The argument for it was that a hinge stops pushing a pair once it is ranked
correctly by a margin, so gradient goes to pairs still inverted rather than to
widening already-correct margins — which for a purely pairwise-ranking metric
sounds right. It is worse by three times the margin W5-E9a lost by. Both W5-E9
variants fail and the incumbent `pairwise_t` stream stands unchanged.

---

## W5-E10 — UNION OF ALL THREE NEW FEATURE BLOCKS — **REJECTED**

**Pre-registered** in `research/WAVE5_PREREG.md` §6.

**Hypothesis.** The three blocks target different mechanisms and, as stage C
showed, different break ages — `m11_focus` is a young-break module, `m12_rdep` a
mature-break one. If their contributions are even partly additive the union
should beat the best single block.

| arm | standalone | Δ vs `RT-301` | folds | blend vs seed clone |
|---|---|---|---|---|
| `RT-750` `m12_rdep` alone, 57 cols | 0.61736 | **+0.00479** | 4/5 | **+0.00141** |
| `RT-760` all three, 175 cols | 0.61468 | +0.00211 | 2/5 | +0.00093 |

**175 new columns do less than half of what 57 do.** And the union is worse than
every single block at every age bucket below 50 — −0.00424 at age 0–5 where
`m11_focus` alone is +0.00308.

**Mechanism, same shape as W4-E6's.** A fixed budget spread over more things.
`feature_fraction = 0.5` now samples 675 columns instead of 557, so each tree
sees a smaller fraction of the block that actually works; and the blocks overlap
more than their titles suggest, since `m11_focus`'s residual channels and
`m12_rdep`'s residual CUSUM/CUSUMSQ paths are both reading AR-residual path
geometry. Wave 4 found gains anti-stacking across ensemble MEMBERS; this is the
same phenomenon across FEATURE BLOCKS.

**What this does NOT license.** Pairwise unions, block subsets, per-block feature
sampling, or any best-of composition — the lattice the pre-registration excluded.

**Retry warranted?** Only as part of a different design: if `m12_rdep` is
promoted, the question "does `m11_focus` add to it" should be asked once, with
`feature_fraction` held at a value that keeps the effective per-tree column count
constant, and pre-registered as its own experiment.

---

## W6-E2 / `RT-900` — VOID: LABEL LEAK VIA THE TRUE-τ MISSINGNESS MASK

*2026-08-22. This is not a rejected hypothesis. It is an **invalid measurement**,
and it is kept here so nobody re-derives it.*

**What it claimed.** 0.86552 TS-AUC against the champion's 0.61605 — **+0.24947,
5/5 folds**, by-age deltas from +0.40 at age 0–5 down to +0.19 at 100+. Under
the pre-registered §4.3 rule that reads "localisation is the lever."

**What it actually was.** The oracle block is `NaN` for every row with
`t < cut`, because there is no post-cut segment to compute over. For a break
series `cut = tau`, so the block's **missingness mask is the row-level target**.
LightGBM splits on missingness natively.

| diagnostic | value |
|---|---:|
| TS-AUC of the bare indicator `1[t >= cut]`, nothing else | **0.81442** |
| share of dev rows where the oracle block is `NaN` | 49.1% |
| **share of those `NaN` rows that are negatives** | **100.00%** |

An indicator that never looks at the data beats the champion by +0.198.

**Why the placebo cut did not save it.** Giving no-break series a placebo cut
drawn from the positives' relative-τ distribution is the wave-1 taxonomy's
construction and the oracle-frontier study's, and it is correct **at the series
level**: "does this series contain a break?" is not answered by knowing where
the boundary is. It cannot work **at the row level**, because "has the break
happened by now?" is answered by the boundary *exactly*. Same cut, different
question.

**Why no patch is authorised.** Not imputation, not dropping `or_elapsed` or
`or_frac`, not a missing indicator, not zero/null fill, not column masking, not
changing LightGBM's missing handling. Removing the explicit timing columns kills
the `NaN` mask, but the same information returns through `or_frac = 1.0` exactly
when `t < cut` and through the segment length the null calibration matches on.
Under TS-AUC, "give the model τ" is degenerate: **τ is the label.** The failure
is in the experimental design, not in the code.

**Blast radius: none.** `w6oracle` was never registered with
`sbr.features.base.load_all()`, never in a production manifest, never in a
`crunch test`, never in an ensemble, never in a submission. `RT-900` carries
`status=VOID` in `research/RESULTS.csv` and appears in no comparison table as
valid alpha.

**What replaced it.** `W6-E2R` (`research/WAVE6_PREREG.md` §18) — the same
question at **one row per series**, scored with **series ROC AUC**, against the
prior oracle-information-frontier control it must first reproduce. Standing rule
in `research/PROTOCOL.md` §1; regression test in `tests/test_no_tau_leakage.py`.

**The general lesson, stated so it is reusable.** A control that is valid at one
unit of analysis is not thereby valid at another. Before trusting any oracle
study, ask what the *unit* is and whether the oracle quantity determines the
*target at that unit*. And treat an implausibly large effect as a bug signal
first and a discovery second — +0.249 on a saturated ensemble was never going to
be real.

---

## RT-1200 -- Pilot 2 relay score-state transform -- KILL (2026-08-24)

**Mechanism.** Protective-relay logic applied as a causal transform of RT-600
evidence: operate/reset integral, thermal-replica state, picked-up dwell, and
pickup-to-reset cycle count.

**Variant.** Single preregistered fold-0 F-arm score-state candidate from
`research/reports/new_avenues_2026/PILOTS_01_03_PREREG.md`.

**Result.** Fold 0 integration:

| arm | TS-AUC |
|---|---:|
| RT600 | 0.638276 |
| RT600 + RT-401 seed clone | 0.638586 |
| RT600 + RT-1200 | 0.638380 |

Binding marginal vs clone: **-0.000207**. Gain vs RT600 alone: **+0.000103**.
Standalone whole-fold candidate TS-AUC was 0.619611. Dominant-cell candidate
AUC was 0.658230 versus RT-600 at 0.677711; within-t correlation with RT-600
was +0.7746; sampled dominant-cell pair-flow net was -318.

**Binding gate.** KILL because `marginal_vs_clone < +0.0010`.

**What this falsifies.** A simple relay-style state transform of the existing
RT-600 score path does not add competition-useful marginal ensemble alpha.
The explicit reset/count state is not enough to beat an exchangeable seed clone.

**What this does not falsify.** Relay logic on lower-level evidence channels is
not fully falsified by this F-arm result. A future A-mode feature block would
need a fresh preregistration and ID, and must still beat the same seed-clone
gate.

**Causality and status.** Prefix-state verification passed on 8 series / 29
prefixes. Post-cleanup reproduction after `aca2c4f` matched the original
metrics exactly at tolerance `1e-12` (`marginal_vs_clone=-0.000206633`);
final status is **FINAL SCREEN RESULT**.

Report: `research/reports/new_avenues_2026/pilot02_relay_logic.{md,json}`.

---

## RT-1201 -- Pilot 3 IM2 matched-length run null + dwell bank -- KILL (2026-08-24)

**Mechanism.** Path-functional null calibration for contiguous AR(2)-residual
scale excursions: matched-length maximum-run percentile, excursion-mass
percentile, and growth proxy over windows 32, 64, and 128.

**Variant.** Single preregistered fold-0 scalar candidate from
`research/reports/new_avenues_2026/PILOTS_01_03_PREREG.md`.

**Result.** Fold 0 integration:

| arm | TS-AUC |
|---|---:|
| RT600 | 0.638276 |
| RT600 + RT-401 seed clone | 0.638586 |
| RT600 + RT-1201 | 0.638888 |

Binding marginal vs clone: **+0.000301**. Gain vs RT600 alone: **+0.000612**.
Standalone whole-fold candidate TS-AUC was 0.583578. Dominant-cell candidate
AUC was 0.610158 versus RT-600 at 0.677711; mature-vs-never was 0.613459;
within-t correlation with RT-600 was +0.3817; sampled dominant-cell pair-flow
net was -1151.

**Binding gate.** KILL because `marginal_vs_clone < +0.0010`.

**What this falsifies.** This direct-score IM2/dwell scalar does not add enough
competition-useful marginal ensemble alpha to justify 5-fold confirmation. Low
correlation and visible standalone dwell separation are not sufficient when the
score is much weaker than RT-600 in the same same-t ranking geometry.

**What this does not falsify.** It does not fully falsify dwell information as
a future feature-block input to a trained specialist, nor joint size-duration
rarity. It does falsify this preregistered cheap direct scalar as a promotion
candidate.

**Causality and status.** Prefix verification passed on 8 series / 29 prefixes.
The run-length null uses exact interval-union counting of matched-length
historical segments. The mass companion uses a history-only episode-length
empirical null. Post-cleanup reproduction after `aca2c4f` matched the original
metrics exactly at tolerance `1e-12` (`marginal_vs_clone=+0.000301470`);
final status is **FINAL SCREEN RESULT**.

Report: `research/reports/new_avenues_2026/pilot03_im2_dwell.{md,json}`.

---

## RT-1202 / RT-1203 -- Pilot 4 trajectory geometry -- KILL (2026-08-24)

**Mechanism.** Shape-similarity trajectory geometry from z-normalized
subsequences over windows 16 and 64: nearest-neighbour provenance
(`d_hist` versus prior non-overlapping online neighbour) plus a cheap
history-boundary arc-rate proxy. `RT-1203` is the preregistered shuffled-history
control: same provenance distances, seed-0 permutation of historical arc order.

**Variant.** Single preregistered fold-0 real scalar plus shuffled-order control
from `research/reports/new_avenues_2026/PILOT04_PREREG.md`.

**Result.** Fold 0 integration:

| arm | TS-AUC |
|---|---:|
| RT600 | 0.638276 |
| RT600 + RT-401 seed clone | 0.638586 |
| RT600 + RT-1202 | 0.636000 |
| RT600 + RT-1203 | 0.635545 |

`RT-1202` binding marginal vs clone: **-0.002587**. Gain vs RT600 alone:
**-0.002277**. Standalone whole-fold candidate TS-AUC was 0.499324;
dominant-cell candidate AUC was 0.500308 versus RT-600 at 0.677711;
within-t correlation with RT-600 was +0.0039; sampled dominant-cell pair-flow
net was -2974.

`RT-1203` shuffled-control marginal vs clone was **-0.003042**; real minus
shuffled marginal was only +0.000455, and both arms damaged the ensemble.

**Binding gate.** KILL because `RT-1202 marginal_vs_clone < +0.0010`.

**What this falsifies.** This cheap direct trajectory-geometry scalar does not
carry usable same-t ranking signal. It is almost random standalone and actively
hurts the RT-600 ensemble after the same seed-clone gate used for the previous
pilots.

**What this does not falsify.** It does not prove that all temporal-order
information is absent. It falsifies this sampled-reference NN provenance plus
boundary arc-rate scalar as a promotion candidate. A learned sequence model or a
trained feature block would need a new preregistration and a materially
different reason to expect signal.

**Causality and status.** Prefix verification passed on 8 series / 29 prefixes.
The prior-online nearest-neighbour search excludes overlapping windows ending
after `t-m`. Post-cleanup reproduction after `aca2c4f` matched the original
metrics exactly at tolerance `1e-12` (`RT-1202 marginal_vs_clone=-0.002586857`,
`RT-1203 marginal_vs_clone=-0.003041629`); final status is
**FINAL SCREEN RESULT**.

Report: `research/reports/new_avenues_2026/pilot04_trajectory_geometry.{md,json}`.

---

## RT-1204 / RT-1205 -- Pilot 5 scale-survival coarse-graining -- KILL (2026-08-24)

**Mechanism.** Dyadic causal coarse-graining of the AR(2) residual-square stream
at scales `{1,2,4,8,16,32}`. Each online trailing block mean is compared against
the same-scale historical block-mean empirical null. `RT-1204` emits the five
cross-scale functionals: q05/q01 survival counts, q05/q01 largest surviving
scale code, and log-scale surprise slope. `RT-1205` is the binding control with
the six individual per-scale surprises only.

**Variant.** Preregistered Mode-A fold-0 ABL feature-addition screen from
`research/reports/new_avenues_2026/PILOT05_PREREG.md`. Both arms use the same
LightGBM architecture, fold, row budget, seed, and base legal bank; only the
Pilot-5 feature block differs.

**Result.** Fold 0 integration:

| arm | TS-AUC |
|---|---:|
| RT600 | 0.638276 |
| RT600 + RT-401 seed clone | 0.638586 |
| RT600 + RT-1204 | 0.638350 |
| RT600 + RT-1205 | 0.637880 |

`RT-1204` binding marginal vs clone: **-0.000236**. Gain vs RT600 alone:
**+0.000074**. Standalone whole-fold TS-AUC was 0.629444; dominant-cell AUC was
0.670218 versus RT-600 at 0.677711; within-t correlation with RT-600 was
+0.8860; sampled dominant-cell pair-flow net was -131.

`RT-1205` individual-scale control marginal vs clone was **-0.000707**.
Summary minus individual-control marginal was only **+0.000471**, just below
the preregistered +0.0005 distinguishability floor.

**Binding gate.** KILL because `RT-1204 marginal_vs_clone < +0.0010`. The
second scientific gate also failed: the summary arm did not beat the individual
per-scale control by the preregistered +0.0005 floor.

**What this falsifies.** Explicit scale-survival count / largest-surviving-scale
code / log-scale decay summaries on this frozen dyadic AR(2) residual-square
coarse-grained stream do not add enough competition-useful marginal ensemble
alpha to justify confirmation.

**What this does not falsify.** It does not falsify all multiscale
representations or all residual-scale detectors. It specifically falsifies this
small E1/E4 cross-scale functional, under the fixed thresholds and Mode-A
fold-0 ABL screen, as a promotion candidate.

**Causality and status.** `harness.verify()` passed for both summary and
individual mechanisms. Registered feature modules passed prefix invariance at
`atol=0.0`; first-valid/NaN semantics, future-mutation prefix stability, and
deterministic replay checks passed. RT600 sentinel reproduced exactly in the
expected tolerance (`mean=0.625811`, `pooled=0.625627`, dominant
`0.664277`, fold-0 E0 `0.638276`, E1 `0.638586`). Final status is
**FINAL SCREEN RESULT**.

Report: `research/reports/new_avenues_2026/pilot05_scale_survival.{md,json}`.

---

## RT-1206 / RT-1207 -- Pilot 6 spectral impulsiveness contrast -- KILL (2026-08-24)

**Mechanism.** Four Goertzel bands from the existing `m03_dyn` dyadic frequency
bank, with length-32 trailing segment envelopes and an adaptive trailing
half-prefix online window. `RT-1206` emits eight product-contrast columns:
band `energy_z` times negative spectral-kurtosis z and negative robust-negentropy
z. `RT-1207` is the binding control with the four matched plain `energy_z`
columns only.

**Variant.** Preregistered Mode-A fold-0 ABL feature-addition screen from
`research/reports/new_avenues_2026/PILOT06_PREREG.md`. Both arms use the same
LightGBM architecture, fold, row budget, seed, and base legal bank; only the
Pilot-6 feature block differs.

**Result.** Fold 0 integration:

| arm | TS-AUC |
|---|---:|
| RT600 | 0.638276 |
| RT600 + RT-401 seed clone | 0.638586 |
| RT600 + RT-1206 | 0.638233 |
| RT600 + RT-1207 | 0.638775 |

`RT-1206` binding marginal vs clone: **-0.000353**. Gain vs RT600 alone:
**-0.000043**. Standalone whole-fold TS-AUC was 0.627636; dominant-cell AUC was
0.667578 versus RT-600 at 0.664277; within-t correlation with RT-600 was
+0.8809; sampled dominant-cell pair-flow net was -249.

`RT-1207` plain-energy control marginal vs clone was **+0.000189**. Control
standalone whole-fold TS-AUC was 0.631084; dominant-cell AUC was 0.674571;
sampled dominant-cell pair-flow net was -20.

Contrast minus plain-energy control marginal was **-0.000542**.

**Binding gate.** KILL because `RT-1206 marginal_vs_clone < +0.0010`. The
second scientific gate also failed: the plain-energy control exceeded the
contrast arm.

**What this falsifies.** The F1/F6 spectral impulsiveness product contrast, as
implemented with fixed dyadic Goertzel bands, SEG=32 envelopes, robust
historical-null calibration, and the Mode-A fold-0 ABL screen, does not add
competition-useful marginal ensemble alpha. The result also argues that the
plain band-energy component accounts for any useful signal in this construction.

**What this does not falsify.** It does not falsify all spectral
representations, learned frequency-domain features, or other envelope statistics.
It specifically falsifies this low-impulsiveness product contrast under the
preregistered fixed bands and control.

**Causality and status.** `harness.verify()` passed for both contrast and
energy mechanisms. Registered feature modules passed prefix invariance at
`atol=0.0`; first-valid/NaN semantics, future-mutation prefix stability, and
deterministic replay checks passed. RT600 sentinel reproduced exactly in the
expected tolerance (`mean=0.625811`, `pooled=0.625627`, dominant
`0.664277`, fold-0 E0 `0.638276`, E1 `0.638586`). Final status is
**FINAL SCREEN RESULT**.

Report: `research/reports/new_avenues_2026/pilot06_spectral_impulse.{md,json}`.

---

## RT-1208 / RT-1209 -- Pilot 7 ordinal transition divergence and time irreversibility -- KILL (2026-08-24)

**Mechanism.** Order-3 ordinal patterns with the same tie convention as
`m03_dyn`. `RT-1208` emits expanding and trailing-half-prefix 6x6 transition
KL divergence plus signed Ramsey-Rothman increment-asymmetry features, each
with matched-count historical-null calibration. `RT-1209` is the binding
control with the matching permutation-entropy-only features.

**Variant.** Preregistered Mode-A fold-0 ABL feature-addition screen from
`research/reports/new_avenues_2026/PILOT07_PREREG.md`. Both arms use the same
LightGBM architecture, fold, row budget, seed, and base legal bank; only the
Pilot-7 feature block differs.

**Result.** Fold 0 integration:

| arm | TS-AUC |
|---|---:|
| RT600 | 0.638276 |
| RT600 + RT-401 seed clone | 0.638586 |
| RT600 + RT-1208 | 0.638551 |
| RT600 + RT-1209 | 0.638079 |

`RT-1208` binding marginal vs clone: **-0.000036**. Gain vs RT600 alone:
**+0.000274**. Standalone whole-fold TS-AUC was 0.629141; dominant-cell AUC
was 0.668268 versus RT-600 at 0.664277; within-t correlation with RT-600 was
+0.8780; sampled dominant-cell pair-flow net was -75.

`RT-1209` entropy-only control marginal vs clone was **-0.000508**. Control
standalone whole-fold TS-AUC was 0.626836; dominant-cell AUC was 0.663077;
sampled dominant-cell pair-flow net was -276.

Candidate minus entropy-control marginal was **+0.000472**, just below the
preregistered +0.0005 distinguishability floor.

**Binding gate.** KILL because `RT-1208 marginal_vs_clone < +0.0010`. The
second scientific gate also failed: the transition/asymmetry arm did not beat
the entropy-only control by the preregistered +0.0005 floor.

**What this falsifies.** Order-3 ordinal transition divergence and the
preregistered time-irreversibility block failed to add marginal ensemble alpha
beyond the entropy-only control under the fixed tie convention, transition KL,
Ramsey-Rothman asymmetry, matched-count null grid, and Mode-A fold-0 ABL
screen.

**What this does not falsify.** It does not falsify all ordinal methods, all
time-asymmetry statistics, visibility-graph statistics, or nonlinear dynamics
representations. It specifically falsifies this L3/L4 construction and control
comparison.

**Causality and status.** `harness.verify()` passed for both candidate and
entropy-control mechanisms. Registered feature modules passed prefix invariance
at `atol=0.0`; first-valid/NaN semantics, future-mutation prefix stability, and
deterministic replay checks passed. Pilot 7 does not recompute the known
`m07_bayes::bo_p_lt25_z` stream-parity path. RT600 sentinel reproduced exactly
in the expected tolerance (`mean=0.625811`, `pooled=0.625627`, dominant
`0.664277`, fold-0 E0 `0.638276`, E1 `0.638586`). Final status is
**FINAL SCREEN RESULT**.

Report:
`research/reports/new_avenues_2026/pilot07_ordinal_irreversibility.{md,json}`.

---

## RT-1210 / RT-1211 -- Pilot 10 joint size-duration rarity -- KILL (2026-08-24)

**Mechanism.** AR(2) residual-square rolling means over windows 32, 64, and
128 with the Pilot-3 q90 excursion band. `RT-1210` emits live and running-max
one-sided joint rarity of excursion peak excess and live duration under a
history-only endpoint table. `RT-1211` is the binding matched control with the
same endpoint table and duration-only dwell rarity.

**Variant.** Preregistered Mode-A fold-0 ABL feature-addition screen from
`research/reports/new_avenues_2026/PILOT10_PREREG.md`. Both arms use the same
LightGBM architecture, fold, row budget, seed, and base legal bank; only the
Pilot-10 feature block differs.

**Result.** Fold 0 integration:

| arm | TS-AUC |
|---|---:|
| RT600 | 0.638276 |
| RT600 + RT-401 seed clone | 0.638586 |
| RT600 + RT-1210 | 0.638065 |
| RT600 + RT-1211 | 0.638350 |

`RT-1210` binding marginal vs clone: **-0.000522**. Gain vs RT600 alone:
**-0.000211**. Standalone whole-fold TS-AUC was 0.626121; dominant-cell AUC
was 0.663758 versus RT-600 at 0.664277; within-t correlation with RT-600 was
+0.8779; sampled dominant-cell pair-flow net was -237.

`RT-1211` dwell-only control marginal vs clone was **-0.000237**. Control
standalone whole-fold TS-AUC was 0.627141; dominant-cell AUC was 0.664654;
within-t correlation with RT-600 was +0.8704; sampled dominant-cell pair-flow
net was -205.

Candidate minus dwell-control marginal was **-0.000285**. Candidate minus
dwell-control dominant-cell AUC was **-0.000896**.

**Binding gate.** KILL because `RT-1210 marginal_vs_clone < +0.0010`. The
mechanism-specific control gates also failed: the dwell-only control exceeded
the joint-rarity candidate on marginal ensemble value and on dominant-cell AUC.

**What this falsifies.** Joint size-duration rarity failed to add marginal
ensemble alpha beyond matched dwell-only rarity under the fixed AR(2)
residual-square channel, windows 32/64/128, q90 excursion band, historical
endpoint joint null, `1/(2*n_endpoints)` floor, and Mode-A fold-0 ABL screen.

**What this does not falsify.** It does not falsify all large-deviation,
scan-statistic, first-passage, or observer-residual approaches. It specifically
falsifies this I1 endpoint-null construction and its matched dwell-control
comparison.

**Causality and status.** `harness.verify()` passed for both candidate and
dwell-control mechanisms. Registered feature modules passed prefix invariance
at `atol=0.0`; first-valid/NaN semantics, future-mutation prefix stability,
deterministic replay, and the joint-surprise >= dwell-surprise invariant passed.
Pilot 10 does not recompute the known `m07_bayes::bo_p_lt25_z` stream-parity
path. RT600 sentinel reproduced exactly in the expected tolerance
(`mean=0.625811`, `pooled=0.625627`, dominant `0.664277`, fold-0 E0
`0.638276`, E1 `0.638586`). Final status is **FINAL SCREEN RESULT**.

Report: `research/reports/new_avenues_2026/pilot10_joint_rarity.{md,json}`.

---

## RT-1212 / RT-1213 -- Pilot 9(i) scalar historical difficulty gate -- KILL (2026-08-24)

**Mechanism.** One series-constant scalar from the 23 history-only fingerprints,
trained nested/fold-pure to predict RT-600 dominant-cell pair loss rate.
`RT-1212` uses the real nested scalar. `RT-1213` uses the binding within-fold
deranged scalar control, preserving each fold's scalar marginal distribution
while breaking the series-specific fingerprint match.

**Variant.** Preregistered Mode-A fold-0 ABL feature-addition screen from
`research/reports/new_avenues_2026/PILOT09_PREREG.md`. Both arms use the same
LightGBM architecture, fold, row budget, seed, and base legal bank; only the
single scalar column differs.

**Result.** Fold 0 integration:

| arm | TS-AUC |
|---|---:|
| RT600 | 0.638276 |
| RT600 + RT-401 seed clone | 0.638586 |
| RT600 + RT-1212 | 0.638750 |
| RT600 + RT-1213 | 0.638855 |

`RT-1212` binding marginal vs clone: **+0.000164**. Gain vs RT600 alone:
**+0.000474**. Standalone whole-fold TS-AUC was 0.629133; dominant-cell AUC
was 0.666635 versus RT-600 at 0.664277; within-t correlation with RT-600 was
+0.8629; sampled dominant-cell pair-flow net was -71.

`RT-1213` deranged-control marginal vs clone was **+0.000269**. Control
standalone whole-fold TS-AUC was 0.630245; dominant-cell AUC was 0.666912;
within-t correlation with RT-600 was +0.8712; sampled dominant-cell pair-flow
net was -226.

Candidate minus deranged-control marginal was **-0.000105**. The scalar target
itself was learnable on fold 0 as a diagnostic (`Spearman=+0.137`, `p=8.8e-7`),
but the learned scalar was not load-bearing in ensemble integration.

**Binding gate.** KILL because the deranged scalar control exceeded the real
scalar. The primary marginal gate also failed: `RT-1212 marginal_vs_clone`
was below `+0.0010`.

**What this falsifies.** The nested one-scalar J1 construction failed to add
load-bearing marginal ensemble alpha beyond a within-fold derangement control
under the fixed 23 fingerprint columns, RT-600 dominant-cell loss-rate target,
fold-pure scalar training, and Mode-A fold-0 ABL screen.

**What this does not falsify.** It does not falsify all historical-DGP
conditioning, tail-conditioned nulls, observer-conditioned representations, or
future reweighting schemes. It specifically falsifies this single scalar as an
eighth feature-stream conditioner because the fold-wise marginal/deranged
control explained the observed gain.

**Causality and status.** Fingerprint names matched the preregistered 23-column
history-only bank. Nested fold-purity audit passed; no scalar model used an
outer validation fold target for the scalar used on that fold. The derangement
had no fixed points inside folds and preserved each fold's scalar multiset.
The scalar was constant within checked series and filled zero lockbox rows.
Pilot 9(i) does not recompute the known `m07_bayes::bo_p_lt25_z` stream-parity
path. RT600 sentinel reproduced exactly in the expected tolerance
(`mean=0.625811`, `pooled=0.625627`, dominant `0.664277`, fold-0 E0
`0.638276`, E1 `0.638586`). Final status is **FINAL SCREEN RESULT**.

Report: `research/reports/new_avenues_2026/pilot09_difficulty_gate.{md,json}`.

---

## RT-1214 / RT-1215 -- Pilot 3 / IM3 individual observer residuals -- KILL (2026-08-24)

**Mechanism.** Two separately scored frozen per-series observers fit on history
only. `RT-1214` is an AR(2)-state Kalman observer with fixed history-only
`q x r` noise-grid selection and emits NIS accumulation, windowed NIS excess,
and normalized-innovation whiteness. `RT-1215` is a Hankel-DMD observer with
delay 16, rank 4, horizons 1 and 5, historical-subspace residual, and
effective-rank monitors.

**Variant.** Preregistered Mode-A fold-0 ABL feature-addition screen from
`research/reports/new_avenues_2026/PILOT03_OBSERVERS_PREREG.md`. The base
legal bank already includes `m04_resid`, so the binding marginal tests whether
either observer adds ensemble value beyond existing scalar residual monitors
and beyond the `RT-401` seed clone. The two arms are not unioned.

**Result.** Fold 0 integration:

| arm | TS-AUC |
|---|---:|
| RT600 | 0.638276 |
| RT600 + RT-401 seed clone | 0.638586 |
| RT600 + RT-1214 | 0.638722 |
| RT600 + RT-1215 | 0.638812 |

`RT-1214` binding marginal vs clone: **+0.000135**. Gain vs RT600 alone:
**+0.000446**. Standalone whole-fold TS-AUC was 0.630817; dominant-cell AUC
was 0.670067 versus RT-600 at 0.664277; within-t correlation with RT-600 was
+0.8836; sampled dominant-cell pair-flow net was -98.

`RT-1215` binding marginal vs clone: **+0.000226**. Gain vs RT600 alone:
**+0.000536**. Standalone whole-fold TS-AUC was 0.631331; dominant-cell AUC
was 0.673238 versus RT-600 at 0.664277; within-t correlation with RT-600 was
+0.8859; sampled dominant-cell pair-flow net was -11.

**Binding gate.** KILL for both arms because each missed the primary
`+0.0010` marginal-vs-clone gate. Hankel-DMD also failed the preregistered C1
redundancy guard because within-t rank correlation with RT-600 exceeded the
0.85 ceiling.

**What this falsifies.** These exact frozen observer constructions failed to
add enough marginal ensemble alpha under the fixed AR(2)-state Kalman grid,
fixed Hankel delay/rank/horizon design, history-only null calibration,
Mode-A fold-0 ABL screen, and existing `m04_resid` base-bank boundary.

**What this does not falsify.** It does not falsify all state-space,
innovation, delay-embedding, subspace-tracking, or adaptive observer methods.
It specifically kills these two preregistered non-adaptive observer arms as
eighth streams in this screen.

**Causality and status.** `harness.verify()` passed for both observer
mechanisms. Registered feature modules passed prefix invariance at `atol=0.0`;
Kalman first-valid/NaN semantics, future-mutation prefix stability,
deterministic replay, and history-only fit replay checks passed. Pilot 3
observers do not recompute the known `m07_bayes::bo_p_lt25_z`
stream-parity path. RT600 sentinel reproduced exactly in the expected
tolerance (`mean=0.625811`, `pooled=0.625627`, dominant `0.664277`,
fold-0 E0 `0.638276`, E1 `0.638586`). Final status is
**FINAL SCREEN RESULT**.

Report: `research/reports/new_avenues_2026/pilot03_observers.{md,json}`.

---

## RT-1216 / RT-1217 / RT-1218 -- Pilot 8 weighted conformal test martingale -- KILL (2026-08-24)

**Mechanism.** `RT-1216` adds eight conformal test martingale channels using
historical-PIT tail, dispersion, dependence, volatility-cluster, and Vovk
power-martingale payoffs. Its betting fraction is scaled by a predictable
recent benign-tail weight that excludes the current row. `RT-1217` is the
matched unweighted CTM control with the same payoffs and fixed weight 1.
`RT-1218` is a parameter-free direct score: the log-average of the five
preregistered `m07_bayes` e-process log-capitals, with no learned weights.

**Variant.** Preregistered Mode-A fold-0 ABL feature-addition screen from
`research/reports/new_avenues_2026/PILOT08_PREREG.md`. H1 tested the weighted
CTM feature block against the matched unweighted CTM control. H2 tested direct
e-value aggregation against RT600 + `RT-401` seed-clone marginal value.

**Result.** Fold 0 integration:

| arm | TS-AUC |
|---|---:|
| RT600 | 0.638276 |
| RT600 + RT-401 seed clone | 0.638586 |
| RT600 + RT-1216 | 0.639523 |
| RT600 + RT-1217 | 0.638799 |
| RT600 + RT-1218 | 0.636373 |

`RT-1216` binding marginal vs clone: **+0.000937**. Gain vs RT600 alone:
**+0.001247**. Standalone whole-fold TS-AUC was 0.634279; dominant-cell AUC
was 0.673986 versus RT-600 at 0.664277; mature-vs-never AUC was 0.670290;
within-t correlation with RT-600 was +0.8653; sampled dominant-cell pair-flow
net was -35.

`RT-1217` matched unweighted control marginal vs clone was **+0.000212**.
Control standalone whole-fold TS-AUC was 0.629803; dominant-cell AUC was
0.670833; mature-vs-never AUC was 0.668155; within-t correlation with RT-600
was +0.8683; sampled dominant-cell pair-flow net was -45.

Weighted minus unweighted mature-vs-never AUC was **+0.002135**, so the H1
split gate passed. Weighted minus unweighted marginal was **+0.000724**.

`RT-1218` direct e-value aggregation marginal vs clone was **-0.002214**.
Standalone whole-fold TS-AUC was 0.525678; dominant-cell AUC was 0.536757;
mature-vs-never AUC was 0.540505; within-t correlation with RT-600 was
+0.1341; sampled dominant-cell pair-flow net was -2167.

**Binding gate.** KILL for `RT-1216` because marginal_vs_clone was below the
primary `+0.0010` gate. The H1 split gate passed, but it was secondary and
cannot promote an arm that misses the primary marginal floor. KILL for
`RT-1218` because marginal_vs_clone was negative and below `+0.0010`.

**What this falsifies.** The exact weighted CTM construction failed to clear
the continuation floor despite beating its unweighted split control. The
parameter-free log-average of existing `m07_bayes` e-process capitals was
actively harmful in ensemble integration under the fold-0 screen.

**What this does not falsify.** It does not falsify all conformal martingales,
all e-values, or all anytime-valid aggregations. It specifically falsifies the
fixed historical-PIT payoff bank, fixed benign-tail weight formula, fixed power
grid, no-learned-weight H2 aggregation, and Mode-A fold-0 ABL integration path
used here.

**Causality and status.** `harness.verify()` passed for both weighted and
unweighted CTM mechanisms. Registered feature modules passed prefix invariance
at `atol=0.0`; finite/peak checks, omega predictability, future-mutation
prefix stability, deterministic replay, exact H2 m07-column selection, and
zero-lockbox-fill checks passed. RT600 sentinel reproduced exactly in the
expected tolerance (`mean=0.625811`, `pooled=0.625627`, dominant `0.664277`,
fold-0 E0 `0.638276`, E1 `0.638586`). Final status is
**FINAL SCREEN RESULT**.

Report: `research/reports/new_avenues_2026/pilot08_weighted_ctm.{md,json}`.

**First-sweep status.** Pilot 8 was the remaining planned first-sweep
mechanism. No arm cleared continuation, so the New Avenues first-sweep queue is
exhausted with no 5-fold confirmation candidate.

---

## RT-1219 / RT-1220 / RT-1221 / RT-1222 -- Second Sweep SS-01 repair-damage arbiter -- KILL (2026-08-25)

**Hypothesis.** A fold-pure constrained action policy can use frozen
first-sweep killed arms as causal sensors to identify when a small correction
repairs an RT600 same-t pair inversion without also damaging RT600-correct
pairs.

**Variant.** Execution preregistered in
`research/reports/new_avenues_2026/second_sweep/SS01_EXECUTION_PREREG.md` at
`32b5427`, following the program preregistration in
`research/reports/new_avenues_2026/SECOND_SWEEP_PREREG.md`. `RT-1219` is the
candidate arbiter. `RT-1220` is the global average frozen-sensor control.
`RT-1221` is the shuffled repair/damage-target arbiter control. `RT-1222` is
the train-fold-selected single-best killed-arm blend control.

**Result.** Fold 0 integration:

| arm | TS-AUC | marginal vs clone |
|---|---:|---:|
| RT600 | 0.638276 | |
| RT600 + RT-401 seed clone | 0.638586 | |
| RT600 + RT-1219 | 0.638276 | -0.000310 |
| RT600 + RT-1220 | 0.638900 | +0.000314 |
| RT600 + RT-1221 | 0.638276 | -0.000310 |
| RT600 + RT-1222 | 0.638904 | +0.000317 |

Dominant-cell pair-flow for `RT-1219`: repairs `0`, damage `0`, net `0`,
RT600-right damage rate `0.0000`. It retained `0.0000` of the original
first-sweep dominant repair reservoir (`0 / 14868`) while rejecting `1.0000`
of the original candidate-union damage (`0 / 28505`). The policy selected
RT600-only for all `806334` fold-0 validation rows, leaving `0` non-RT600
action rows and `0` contributing sensor families.

**Binding gate.** KILL because the candidate missed the `+0.0015`
marginal-vs-clone gate, produced no positive dominant-cell net pair lift, had
fewer than two contributing sensor families, and failed all three control-gap
requirements: candidate minus `RT-1220` was `-0.000624`, candidate minus
`RT-1221` was `+0.000000`, and candidate minus `RT-1222` was `-0.000627`.

**What this falsifies.** Under the frozen state representation and fixed
bounded-correction action set, first-sweep killed arms should not be reused as
production arbitration sensors. The first-sweep repair reservoir is not
separable from damage by this preregistered causal arbiter.

**What this does not falsify.** It does not falsify residual pair-ranking over
the incumbent 500-feature bank, negative-side calibration, or specialist
disagreement routing. Those are distinct Second Sweep mechanisms with their
own preregistered gates.

**Causality and status.** Preflight reproduced the RT600 sentinel exactly in
the expected tolerance (`mean=0.625811`, `pooled=0.625627`, dominant
`0.664277`, fold-0 E0 `0.638276`, E1 `0.638586`) and reproduced the canonical
dominant pair sample (`50540` pairs, `16193` RT600-wrong, `34347`
RT600-right). Frozen sensors were finite on dev rows and zero finite on
lockbox rows; new OOF lockbox finite counts were `0` for `RT-1219` through
`RT-1222`. No lockbox/test/submission/production path was used. Final status:
**FINAL SCREEN RESULT**.

Report:
`research/reports/new_avenues_2026/second_sweep/ss01_repair_damage_arbiter.{md,json}`.

---

## RT-1223 / RT-1224 -- Second Sweep SS-02 dominant-cell residual ranker -- KILL (2026-08-25)

**Hypothesis.** A bounded correction model trained directly against RT600
residual same-t pair errors over the incumbent 500-feature bank can repair
dominant-cell inversions without damaging already-correct RT600 rankings.

**Variant.** Execution preregistered in
`research/reports/new_avenues_2026/second_sweep/SS02_EXECUTION_PREREG.md` at
`3b39954`, after SS-01 was killed. `RT-1223` is the candidate. `RT-1224` is
the shuffled residual-offset/weight control. Inputs are the existing legal
500-feature bank plus frozen RT600 specialist and seed-clone score state; no
first-sweep killed sensor prediction is used.

**Result.** Fold 0 integration:

| arm | TS-AUC | marginal vs clone |
|---|---:|---:|
| RT600 | 0.638276 | |
| RT600 + RT-401 seed clone | 0.638586 | |
| RT600 + RT-1223 | 0.638297 | -0.000290 |
| RT600 + RT-1224 | 0.638384 | -0.000203 |

Candidate pair-flow:

| split | repairs | damage | net | RT600-right damage rate |
|---|---:|---:|---:|---:|
| whole fold | 494 | 587 | -93 | 0.0142 |
| dominant cell | 330 | 438 | -108 | 0.0128 |
| mature-vs-never | 376 | 453 | -77 | 0.0130 |
| mature-vs-prebreak | 328 | 430 | -102 | 0.0130 |

The shuffled residual control had dominant-cell net `-105` and
marginal_vs_clone `-0.000203`, so the real residual target did not separate
from its shuffled control.

**Binding gate.** KILL because the candidate missed the `+0.0015`
marginal-vs-clone gate, missed the dominant-cell `+300` net-pair gate with
`-108`, failed the never-break/pre-break pair-flow clause (`-77` and `-102`),
and missed the shuffled-control gap (`-0.000087` observed versus `+0.000500`
required).

**What this falsifies.** After SS-01 had already closed first-sweep sensor
arbitration, SS-02 falsifies this preregistered bounded residual-pair
correction over the incumbent 500-column causal bank. The current causal
feature representation did not provide a stable row-level correction signal for
RT600's dominant same-t residual inversions under the fixed loss, damage
penalty, correction magnitude, and training budget.

**What this does not falsify.** It does not falsify negative-side null-state
calibration or specialist-disagreement routing, which are the next distinct
Second Sweep mechanisms. It also does not justify tuning the residual loss,
damage penalty, correction cap, row budget, or feature subset.

**Causality and status.** Preflight reproduced the RT600 sentinel exactly in
the expected tolerance and reproduced the canonical dominant pair sample
(`50540` pairs, `16193` RT600-wrong, `34347` RT600-right). The script loaded
the 500 incumbent causal feature-bank columns, seven frozen specialist OOF
scores, and `RT-401`; it loaded no first-sweep killed sensor IDs. New OOF
lockbox finite counts were `0` for `RT-1223` and `RT-1224`. No
lockbox/test/submission/production path was used. Final status:
**FINAL SCREEN RESULT**.

Report:
`research/reports/new_avenues_2026/second_sweep/ss02_residual_ranker.{md,json}`.

---

## RT-1225 / RT-1226 / RT-1227 / RT-1228 / RT-1229 -- Second Sweep SS-03 negative-side null calibrator -- KILL (2026-08-25)

**Hypothesis.** A fixed fold-pure null-state calibration of RT600 can reduce
never-break false positives by conditioning the negative/null CDF on causal
score state, specialist dispersion, maturity, and weighted CTM benign-tail
suppression, without damaging pre-break ordering.

**Variant.** Execution preregistered in
`research/reports/new_avenues_2026/second_sweep/SS03_EXECUTION_PREREG.md` at
`8a5f41a`, after SS-01 and SS-02 were killed. `RT-1225` is the candidate.
`RT-1226` is the global RT600 null-SCDF calibration control. `RT-1227` is the
deranged weighted-CTM null-state partition control. `RT-1228` is the
unweighted CTM state partition control. `RT-1229` is the frozen Pilot-9 scalar
difficulty state control.

The candidate used one early state plus eight mature states with fixed bits:
`rt600_cal >= 0.50`, outer-train median specialist dispersion, and outer-train
median `(uctm_tail_log - wctm_tail_log)`. It fit negative-only
`SCDF_NSEEN` maps on outer-training rows and applied a bounded correction
`0.20 * clip(F_null_state - F_all, -0.25, +0.25)`, so the maximum absolute
score movement was `0.05`.

**Result.** Fold 0 integration:

| arm | TS-AUC | marginal vs clone |
|---|---:|---:|
| RT600 | 0.638276 | |
| RT600 + RT-401 seed clone | 0.638586 | |
| RT600 + RT-1225 | 0.638287 | -0.000299 |
| RT600 + RT-1226 | 0.638276 | -0.000310 |
| RT600 + RT-1227 | 0.638238 | -0.000348 |
| RT600 + RT-1228 | 0.638296 | -0.000290 |
| RT600 + RT-1229 | 0.638251 | -0.000335 |

Candidate pair-flow:

| split | repairs | damage | net | RT600-right damage rate |
|---|---:|---:|---:|---:|
| whole fold | 722 | 866 | -144 | 0.0210 |
| dominant cell | 659 | 773 | -114 | 0.0225 |
| mature-vs-never | 678 | 727 | -49 | 0.0209 |
| mature-vs-prebreak | 618 | 658 | -40 | 0.0199 |

**Binding gate.** KILL because the candidate missed the mature-vs-never net
pair-flow gate (`-49` versus `>0`), missed the `+0.0010` marginal-vs-clone
gate (`-0.000299`), exceeded the prebreak RT600-right damage-rate cap
(`0.0199` versus `0.0150`), failed to separate from the deranged partition by
the required marginal gap (`+0.000049` versus `+0.000500`), and was slightly
exceeded by the unweighted CTM state control (`RT-1228` marginal
`-0.000290`).

**What this falsifies.** The fixed null-state partition and bounded
conditional-null SCDF correction are not useful production calibration under
the current RT600 score state and Pilot-8 CTM suppression state. The result
also says the weighted-vs-unweighted CTM distinction is not load-bearing in
this calibration role.

**What this does not falsify.** It does not falsify specialist-disagreement
micro-routing, which is the next distinct Second Sweep mechanism. It also does
not falsify future null models with materially different causal state
representations or external data. It does not authorize SS-03b, partition
search, threshold adjustment, correction-scale tuning, CTM retuning, or a new
Pilot-9 scalar variant.

**Causality and status.** Preflight reproduced the RT600 sentinel in the
expected tolerance and reproduced the canonical dominant pair sample (`50540`
pairs, `16193` RT600-wrong, `34347` RT600-right). Required frozen OOF scores
and CTM feature caches were finite on all `4032524` dev rows and zero finite
on lockbox rows. New OOF lockbox finite counts were `0` for `RT-1225` through
`RT-1229`. No lockbox/test/submission/production path was used. Final status:
**FINAL SCREEN RESULT**.

Report:
`research/reports/new_avenues_2026/second_sweep/ss03_null_calibrator.{md,json}`.

---

## RT-1230 / RT-1231 / RT-1232 / RT-1233 -- Second Sweep SS-04 specialist-disagreement action router -- KILL (2026-08-25)

**Hypothesis.** A bounded fold-pure specialist-disagreement router can identify
same-t dominant-cell rows where one of the seven frozen specialists is more
trustworthy than the RT600 blend, especially in majority-correct and near-split
specialist configurations, without learning a global reweighting or replaying
the failed Pilot-1 static selector.

**Variant.** Execution preregistered in
`research/reports/new_avenues_2026/second_sweep/SS04_EXECUTION_PREREG.md` at
`25f40ac`, after SS-01, SS-02, and SS-03 were killed. `RT-1230` is the
candidate. `RT-1231` is the global logistic specialist reweighting control.
`RT-1232` is the Pilot-1 `exc_max_run64` static history-fingerprint selector
replay control. `RT-1233` is the shuffled disagreement-target router control.

The candidate trained a fold-pure row-level action router on exposed
outer-training same-t pairs where RT600 was wrong and at least one frozen
specialist was right. It could either keep RT600 or move a row toward one
specialist by a bounded amount. The binding fold-0 score used only
pre-registered router outputs and no post-score thresholding.

**Result.** Fold 0 integration:

| arm | TS-AUC | marginal vs clone |
|---|---:|---:|
| RT600 | 0.638276 | |
| RT600 + RT-401 seed clone | 0.638586 | |
| RT600 + RT-1230 | 0.638275 | -0.000312 |
| RT600 + RT-1231 | 0.638145 | -0.000441 |
| RT600 + RT-1232 | 0.638031 | -0.000555 |
| RT600 + RT-1233 | 0.638276 | -0.000310 |

Candidate pair-flow:

| split | repairs | damage | net | RT600-right damage rate |
|---|---:|---:|---:|---:|
| dominant cell | 0 | 0 | 0 | 0.0000 |
| majority-correct subgroup | 0 | 0 | 0 | 0.0000 |
| near-split subgroup | 0 | 0 | 0 | 0.0000 |

The global logistic specialist reweighting control lost more than the
candidate, but the gap was only `+0.000130`. The Pilot-1 static selector replay
control gap was `+0.000243`, and the shuffled-target router gap was
`-0.000002`. None reached the required `+0.000500` control gap.

**Binding gate.** KILL because the candidate missed the `+0.0010`
marginal-vs-clone gate (`-0.000312`), missed the majority-correct net-pair
gate (`0` versus `>0`), missed the near-split net-pair gate (`0` versus `>0`),
and failed all three control-separation gates.

**What this falsifies.** The preregistered bounded specialist-disagreement
router did not expose a usable residual action surface inside RT600's dominant
same-t failure reservoir. The result also rules out the specific global
logistic specialist reweighting control and the preregistered Pilot-1
`exc_max_run64` selector replay as rescue mechanisms under the same fold-0
binding screen.

**What this does not falsify.** It does not falsify future research using
materially different state representations, new data, different base models,
or a new preregistered research program. It does not authorize SS-04b,
threshold tuning, action-cap tuning, relabeling of the disagreement target, or
another Second Sweep mechanism.

**Causality and status.** Preflight reproduced the RT600 sentinel in the
expected tolerance and reproduced the canonical dominant pair sample (`50540`
pairs, `16193` RT600-wrong, `34347` RT600-right). Required frozen OOF scores
were finite on all `4032524` dev rows and zero finite on lockbox rows. New OOF
lockbox finite counts were `0` for `RT-1230` through `RT-1233`. No
lockbox/test/submission/production path was used. Final status:
**FINAL SCREEN RESULT**. With SS-04 killed, the preregistered Second Sweep is
**EXHAUSTED**.

Report:
`research/reports/new_avenues_2026/second_sweep/ss04_specialist_router.{md,json}`.

## RT-1234 / RT-1235 -- CRF-01 NNCSR null-normalised causal sequence ranker -- KILL (2026-08-26)

**Program preregistration** `research/reports/causal_representation_frontier/CRF_PROGRAM_PREREG.md` @ `85d121f`.
**Execution preregistration** `research/reports/causal_representation_frontier/CRF01_EXECUTION_PREREG.md` @ `afba958`.
**Preflight** `ed05902`. **Report** `research/reports/causal_representation_frontier/crf01_nncsr.{md,json}`.

**Hypothesis.** The 500-column bank discards temporal order beyond lag-2 products
and has no representation of excursion contiguity or excursion-growth rate. A
causal sequence encoder reading a small, **null-normalised** channel set, trained
under an objective that matches the metric's within-timestep comparison, can
represent those functionals and produce a same-`t` ranking direction RT-600 does
not already contain.

**What was done.** Eight history-fitted null-normalised channels -- `pit` (256-knot
history-ECDF normal score), `inn` (Yule-Walker AR(5) innovation, history-tail lag
initialisation), `inn_pit`, `abs_inn_pit`, `vol_norm` (strictly lagged half-life-32
EWMA, `0.25*mad_H` floor), `surp` (history `|innovation|` tail surprise), `exceed`
(soft `q90_H` exceedance, slope 4), `lag1_pit` -- into the `RT-970` causal dilated
TCN shell verbatim (hidden 32, kernel 3, dilations 1/2/4/8/16/32, six residual
blocks, 35,649 parameters, measured receptive field 253), trained under the same-`t`
pairwise logistic objective with `m_neg = 8` and uniform pair weighting. **No
`elapsed` channel, no `t` channel, no 500-column bank, no RT-600, in any arm.**
Seed 0, 20 epochs, AdamW 3e-3 / 1e-2, cosine, batch 32 series, fold 0.

**The number.** Fold-0 standalone whole-fold TS-AUC **0.592762** at within-`t` rho
vs RT600 **+0.4460**. The preregistered cheap abandon gate -- standalone `< 0.600`
**and** rho `<= 0.60` -- **fired**, so the candidate was abandoned before any
five-fold spend, before C2, and before any ensemble integration.
`marginal_vs_clone` was **never computed**: it requires a fold-pure five-fold OOF
because the cross-fitted SCDF fits fold 0's map on folds 1-4, and those folds were
deliberately not trained. That is the gate working, not a missing measurement.

Dominant-cell standalone `0.621422`. Pair flow negative in **every** cell:
whole-fold net `-2,862`, dominant net `-2,798`, mature-vs-never net `-2,962`,
mature-vs-pre-break net `-2,820`. Pre-break damage rate on RT600-correct pairs
**0.2626 against a 0.0150 cap** -- seventeen times over.

`RT-1235`, the mandatory matched BCE control (byte-identical channels,
architecture, optimiser, schedule, epochs, batches and seeds; only the loss
differs): standalone `0.570543`, dominant-cell `0.584322`, rho `+0.3631`, dominant
pair net `-5,019`.

**Why it failed.** Not detection. The candidate repairs 6,395 of RT-600's 16,193
sampled dominant-cell mistakes -- a 39.5 % repair rate, with a repair Jaccard of
only 0.174 against the seed clone, so the repairs really are its own -- while
damaging 26.8 % of the pairs RT-600 already had right. This is
`FIRST_SWEEP_SYNTHESIS.md` H1 recurring in a new representation: **the bottleneck is
repair-versus-damage arbitration, not the detector.** CRF-01 is a much better
detector than anything in the first sweep and arbitrates no better; SS-01, whose
whole purpose was arbitration, was itself KILL.

**What this closes.** The representation x objective factorial is now complete and
empty. `RT-970` is the same shell on the same folds, so the ladder isolates one
factor at a time:

| arm | channels | objective | fold-0 whole TS-AUC |
|---|---|---|---:|
| `RT-970` | location/scale + `elapsed` | BCE | 0.52618 |
| `RT-1235` | eight null-normalised, no `elapsed` | BCE | 0.57054 |
| `RT-1234` | eight null-normalised, no `elapsed` | same-`t` pairwise | 0.59276 |

**Representation effect `+0.0444`** at fixed objective. **Objective effect
`+0.0222`** whole-fold, `+0.0371` dominant-cell, at fixed representation. Wave 6's
own report said `RT-970` could not distinguish "family wrong" from "objective
wrong". Both were partly wrong, both are now fixed and measured, **and their sum is
still short of a threshold set deliberately below the fitted `+0.0030` contour.**

Closed by this result: null-normalised temporal representation combined with the
same-`t` ranking objective failed under the frozen `RT-970` shell, which closes the
representation x objective lane defined by `CRF_PROGRAM_PREREG.md` §1; learned
causal-prefix representations as an ensemble alpha source; and objective mismatch as
a live explanation for the W7-D3R gap. The objective effect is real and now
quantified -- it is simply an order of magnitude too small to matter.

`RT-1234` additionally sets a **new best standalone-at-low-redundancy point** for
this project -- `0.59276` at rho `0.446` against `RT-1201`'s prior `0.58358` -- so
the `corr(standalone, rho) = +0.983` frontier moved and the answer did not change.
That is the strongest available form of this negative: the gate was not cleared past
by a weak attempt.

**Retry warranted?** **No, and specifically not by escalation.** No `CRF-01b`. No
Transformer, GRU, SSM, RF-255, longer receptive field, wider hidden layer, extra
channel, alternative PIT, different `q90`, or different pair weighting -- all
forbidden by `CRF_PROGRAM_PREREG.md` §0.8/§0.9 and §40 of the execution brief, and
all unsupported by the evidence: the age profile (0.5205 at age 0-5 rising
monotonically to 0.6225 at age 100+) is a detector whose problem is arbitration, not
memory length. `RT-1236` (C2 temporal shuffle) was reserved and **not run**; the id
is not recycled.

**What is NOT closed.** `CRF-02` ACGN, which `CRF_PROGRAM_PREREG.md` §2 declares
unconditional and independent. CRF-01 is a statement about extracting more break
evidence from the legal prefix; CRF-02 asks the different question of whether the
never-break false-positive mass is a conditional-**null misspecification** problem.
`CRF-03` does not open on CRF-01, whose opening rule needs
`marginal_vs_clone >= +0.0015` and which has no marginal at all.

## RT-1237 / RT-1238 / RT-1239 -- CRF-02 ACGN -- **VOID, SUPERSEDED** (2026-08-26)

> **THIS ENTRY IS VOID. Do not cite its numbers.** The run it describes silently
> loaded a 24-series, 1-epoch, `HWIN = 128` null written by a unit test instead of
> the preregistered one. The ids are retired. The corrected run is
> `RT-1240`/`RT-1241`/`RT-1242`, filed below, and it reaches the same verdict
> (KILL) **for a different reason**: the derangement control PASSES on the real
> null, so the failure is amortization, not an inert bottleneck. Kept, not
> deleted, because the record is the record. See `EXPERIMENT_ID_MAP.md` and
> `crf02_acgn.md` §7.

The original text follows unchanged.

**Program preregistration** `research/reports/causal_representation_frontier/CRF_PROGRAM_PREREG.md` @ `85d121f`.
**Execution preregistration** `research/reports/causal_representation_frontier/CRF02_EXECUTION_PREREG.md` @ `9a3d3c7`.
**Preflight** `ad6ecd7`, **pre-score correction** `89149d5`.
**Report** `research/reports/causal_representation_frontier/crf02_acgn.{md,json}`.

**Hypothesis.** 73.99 % of dominant-cell loss is never-break negatives whose loss
rate is predicted by heavy tails, long memory, and histories that produce few
excursions and then wander -- the signature of **null misspecification**. Every null
this project ships is a fixed rolling-mean historical calibration with AR(2) shared
context. A **global, amortized, nonlinear, distributional** null learned from
break-free histories only should price "normal" better, and the sequential departure
of the online stream from it should be a different discriminative direction.

**What was done.** Per outer fold, a generative null `q(x_t | h_i, x_{t-1..t-R})`
pretrained on the **training-fold series' break-free histories only** -- no online
row, no label, no future -- with the frozen `RT-970` TCN body, an 8-dimensional
history bottleneck from mean-pooling the same encoder over the last 1024 points of
`H_i`, and 21 monotone quantile knots (base + softplus increments) at the
preregistered levels under pinball loss. The null was then **frozen**, five
preregistered strictly causal online signals plus their running peaks computed
(predictive PIT, predictive log score, running Anderson-Darling-weighted uniformity
discrepancy, cumulative predictive surprise, encoder latent-state shift), and only
then a 10->32->1 same-`t` pairwise ranking head fitted on training folds. Fold 0.

**The numbers.** Candidate `RT-1237` standalone whole-fold TS-AUC **0.527807** at
within-`t` rho **+0.2212**; dominant-cell **0.539309**.

**Three independent mandatory gates failed:**

1. **Cheap abandon gate** -- `0.527807 < 0.600` AND rho `0.2212 <= 0.60`. FIRED.
2. **Learned-null isolation** -- candidate minus C1 = **`-0.052779`** whole-fold
   (`-0.060600` dominant-cell) against a required `+0.000500`.
3. **Derangement** -- candidate minus C2 = **`-0.000981`**. The `m05_ctx` rule kills
   the arm regardless of its headline number.

`RT-1238`, the **fixed**-null control (AR(5) + 256-knot history residual ECDF,
per series, feeding **identical** downstream statistics into an **identical** ranking
head): **0.580586** standalone, **0.599909** dominant-cell. `RT-1239`, the deranged
`h_i` control: `0.528788` / `0.540154`. A declared no-id diagnostic refitting the
candidate's head on the eight features the fixed-null control also has reached
`0.531864`, **above** the 10-feature candidate.

Pair flow negative in every cell for every arm; candidate pre-break damage rate on
RT600-correct pairs `0.4144` against the `0.0150` cap.

**Why it failed -- and this is the load-bearing part.** The learned null lost to the
project's existing fixed apparatus by **0.0528**, and the candidate's own C2 control
explains why. Permuting `h_i` across series moves the score by `-0.00098`, i.e. by
nothing, so the 8-dimensional history bottleneck carries **no usable series-specific
information at all**. The learned null is therefore in effect a **population-average**
predictive distribution applied to every series alike, while the fixed null is a
**per-series** fit -- five AR coefficients and a 256-knot empirical residual
distribution -- paid for by that series' own break-free history at zero
generalisation cost, because the history is complete at `t = 0`.

Series heterogeneity in this data is large; that is the whole reason the project's
foundation is per-series historical calibration. **Amortizing across series LOSES
information here rather than adding it**, and an 8-float bottleneck is not a wide
enough channel to recover what a per-series fit gets for free.

Note this is **not** the usual `m05_ctx` failure. C2 did not win because `h_i` was a
memorised series identifier; it won because `h_i` was doing nothing, so destroying it
cost nothing. Both readings kill the arm, but the mechanism is what closes the lane.

**What this closes.** Exactly what `CRF_PROGRAM_PREREG.md` 2.9 wrote before any
number existed: **learned amortized generative nulls are closed for this problem.**
The per-series historical calibration the project already ships **is** the right
null -- now measured against a matched learned alternative rather than assumed. The
never-break false-positive mass is **not** a conditional-null misspecification
problem in the sense CRF-02 hypothesized: a strictly better-specified null was built
and it did not help. `h_i`-style amortized conditioning is closed as a route.

Combined with CRF-01, this closes **H-A** (representation saturation), **H-B**
(objective mismatch) and **H-E** (learned-null misspecification), leaving **H-D**.

**Retry warranted?** **No.** No `CRF-02b`. No MDN, no normalising flow, no wider
bottleneck, no deeper head, no different quantile grid, no second epoch count, no
seed re-roll -- all forbidden by `CRF_PROGRAM_PREREG.md` 0.8/0.9 and none supported
by the evidence: the null was not under-trained (pinball converged cleanly from
`0.364706` to `0.243881`), it was mis-conceived. A wider bottleneck is a different
experiment and the program does not authorise one.

**One pre-score failure, recorded.** The first fold-0 attempt stopped at the live
isolation assert and produced **no score of any kind**. It was a false positive --
stale pinball gradients on the frozen null, not a label leak -- fixed by clearing
them at freeze time, tightening the assert to its real contract, and adding a
regression test for that exact case. Nothing frozen changed and the correction is
its own pushed pre-score commit.

**CRF-03 does not open** and the CRF program is exhausted. See `CRF_FINAL.md`.

## RT-1256 -- CAT-410 specialist reimplementation -- KILL (2026-08-27)

**Program.** CatBoost Specialist Activation 2026. Full report:
`research/reports/catboost_specialist_2026/FINAL.md`.

**Hypothesis.** Reimplementing RT-410's evidence-recursion-heavy specialist setup
with the frozen RT-1251 CatBoost learner would add more ensemble alpha than a
matched exchangeable LightGBM seed clone.

**The number.** Standalone CAT-410 improved on incumbent RT-410
(`0.610328251` vs `0.605116528`, delta `+0.005211723`) and had rho
`+0.680825`, but the binding fixed-slot ensemble marginal was only
`+0.000753095` with `4/5` folds positive, below the preregistered `+0.0010`
gate. E2-E0 was `+0.000929590`; dominant-cell net vs E0 was `+17`.

**Why it failed.** Standalone CatBoost gain did not translate into enough
replacement-vs-clone ensemble value. The actual target was CatBoost specialist
ensemble alpha over matched LightGBM replacement, not standalone improvement.

**Retry warranted?** No CatBoost tuning, loss search, seed search, row-cap
change, or RT-410 variant is authorized by this program.


## RT-1250 / RT-1251 / RT-1252 -- Learner Diversity 2026 -- CLOSED, NO PROMOTION (2026-08-27)

**Program preregistration** `research/reports/learner_diversity_2026/PREREG.md`
@ `c91950b`.
**Final report** `research/reports/learner_diversity_2026/FINAL.md`.

**Hypothesis.** A non-LightGBM tabular learner might produce useful same-`t`
pair orderings beyond an exchangeable LightGBM `RT-401` replacement in the
RT600 seven-member ensemble.

**What was done.** Exactly three learner families were authorized: TabM
(`RT-1250`), CatBoost (`RT-1251`), and RealMLP (`RT-1252`). All used the frozen
500-column causal feature bank, canonical folds, `RT-401`-matched row sampler,
and seven-member replacement protocol. No hyperparameter search, feature
engineering, new data, router, stacker, lockbox/test access, or production
change was made.

**Result.** TabM and RealMLP were recorded INFEASIBLE before scoring under the
installed official/default full-scale configurations. CatBoost scored
standalone `0.620407665` with rho vs matched `RT-401` `+0.736467811`.
Replacement integration: E0 RT600 `0.625811342`, E1 six specialists plus
`RT-401` `0.624980472`, E2 six specialists plus CatBoost `0.626090272`.
Primary `marginal_vs_clone = +0.001109800`, positive folds `5/5`, dominant-cell
pair net `+88`.

**Why it is closed.** CatBoost clears INTERESTING (`>= +0.0010`) and therefore
falsifies the strict "LightGBM-only has no family-diversity residual" claim.
It does not clear SERIOUS (`>= +0.0030`), and only one learner crossed the
combination-opening threshold, so `RT-1253` remains unused. The result is a weak
measured family signal, not a deployable promotion.

**Retry warranted?** Not under this program. Any CatBoost HPO, shrunken TabM or
RealMLP run, altered neural default, new feature preprocessing, stacker, router,
or blend-weight tuning would be a new experiment outside the frozen final
learner-diversity audit.

## RT-1243 / RT-1244 -- LA-01 specialist replacement salvage -- KILL (2026-08-26)

**Program preregistration** `research/reports/leaderboard_alpha_2026/PROGRAM_PREREG.md`
@ `ed84d00`.
**Execution preregistration** `research/reports/leaderboard_alpha_2026/LA01_EXECUTION_PREREG.md`
@ `73677fc`.

**Hypothesis.** `m11_focus` and/or `m12_rdep` contained useful specialization
that prior ordinary integration erased because it added streams instead of
replacing a redundant RT600 specialist.

**What was done.** Full five-fold artifact-only nested replacement. For each
outer fold, the replaced RT600 specialist and replacement candidate were chosen
using only the other four folds under fold-pure SCDF calibration. `RT-1243`
allowed `RT-731` (`m11_focus`) or `RT-751` (`m12_rdep`); `RT-1244` used the same
nested rule with seed clone `RT-401`.

**Result.** E0 RT600 mean TS-AUC `0.625811342`; nested seed replacement
`0.624980472`; nested m11/m12 replacement `0.624975063`.
Primary `marginal_vs_clone = -0.000005408`, with only 2/5 folds positive.
Fold deltas vs clone: `+0.001354`, `+0.000078`, `-0.000171`, `-0.000960`,
`-0.000328`.

**Pair flow.** Candidate vs E0, 64 same-`t` pairs per time point: whole dev net
`-92`, dominant-cell repairs/damage/net `828/838/-10`, mature-vs-never net
`-38`, mature-vs-prebreak net `-99`.

**Why it failed.** The replacement rule mostly selected `m12_rdep`, but its
outer-fold gains did not survive the seed-replacement control and did not repair
dominant residual pairs on net. Replacement also lowered mean TS-AUC vs the
untouched RT600 seven-specialist ensemble by `-0.000836`.

**Retry warranted?** No. The preregistered full five-fold gate failed on all
binding dimensions: marginal below `+0.0015`, fewer than 4/5 positive folds, and
dominant-cell net not positive. Specialist-replacement salvage is closed.


## RT-1245 / RT-1246 -- LA-02 counterfactual synthetic augmentation -- KILL (2026-08-26)

**Program preregistration** `research/reports/leaderboard_alpha_2026/PROGRAM_PREREG.md`
@ `ed84d00`.
**Execution preregistration** `research/reports/leaderboard_alpha_2026/LA02_EXECUTION_PREREG.md`
@ `8438d93`.

**Hypothesis.** Fold-pure history-generated paired counterfactuals would teach
the fixed RT600 architecture to separate persistent break trajectories from
transient hard negatives, improving training data rather than changing the model.

**What was done.** Full five-fold real-validation-only retraining of the seven
RT600 specialist configurations. `RT-1245` added same-count fixed-null synthetic
non-break rows. `RT-1246` added paired persistent-positive and transient-negative
synthetic rows at the preregistered `0.33` ratio. Generator distributions were
estimated from outer-training folds only.

**Result.** C0 RT600 mean TS-AUC `0.625811264`; `RT-1245` C1 mean
`0.617350841`; `RT-1246` candidate mean `0.622508549`. The primary
candidate-vs-clone marginal is `+0.005157709` with `4/5` folds positive, which
is a MAJOR-sized metric signal. But the candidate remains below untouched RT600
by `-0.003302715`.

**Pair flow.** Candidate vs C0, 64 same-`t` pairs per time point: whole dev net
`-341`, dominant-cell repairs/damage/net `2853/3179/-326`,
mature-vs-never net `-366`, mature-vs-prebreak net `-246`.

**Why it failed.** Synthetic persistent/transient labels are informative relative
to null-only synthetic rows, but the intervention shifts the RT600 architecture
in the wrong residual direction: it sacrifices the dominant and mature-vs-never
pairs that the preregistered gate required it to repair. This is not a deployable
alpha claim; it is evidence that the counterfactual generator changes the learner
but does not improve the incumbent decision surface.

**Retry warranted?** No LA-02b. No ratio sweep, mechanism reweighting, synthetic
validation, or learner swap is authorised by the frozen program. Because LA-02
did not become SERIOUS under the full gate, LA-03 opens.


## RT-1247 / RT-1248 / RT-1249 -- LA-03 per-series history adaptation -- KILL (2026-08-26)

**Program preregistration** `research/reports/leaderboard_alpha_2026/PROGRAM_PREREG.md`
@ `ed84d00`.
**Execution preregistration** `research/reports/leaderboard_alpha_2026/LA03_EXECUTION_PREREG.md`
@ `3d88512`.

**Hypothesis.** A tiny per-series affine adapter fitted only on `H_i` would keep
the useful global predictive-null structure while recovering local series
calibration that the CRF amortized-null bottleneck lost.

**What was done.** Full five-fold training of one 10-feature pairwise LightGBM
head per arm. `RT-1247` used an outer-fold global AR(5) and pooled residual
ECDF. `RT-1248` used the fixed per-series AR(5)+history-residual-ECDF null.
`RT-1249` used the global AR(5) plus a two-parameter per-series affine adapter
`a_i * g_t + b_i`, fitted on `H_i` with fixed ridge `lambda=32`.

**Result.** Standalone heads: global `0.528175061`, fixed-null `0.564744385`,
adapted `0.543249095`. The adapted head beats the global clone by
`+0.015074034` but loses to the fixed-null control by `-0.021495290`, failing
the mandatory isolation gate.

**Integration.** E0 RT600 `0.625811264`; E1 `RT600+RT-1247` `0.624411329`;
E2 `RT600+RT-1249` `0.626087395`. Primary `marginal_vs_clone = +0.001676065`
with `5/5` folds positive, a WEAK-sized metric signal. The fixed-null diagnostic
`RT600+RT-1248` was stronger than the candidate at `0.626869923`.

**Pair flow.** E2 vs E0, 64 same-`t` pairs per time point: whole dev net `-69`,
dominant-cell repairs/damage/net `1166/1203/-37`, mature-vs-never net `-92`,
mature-vs-prebreak net `-74`.

**Why it failed.** The affine adapter repairs the deliberately weak global
control, but it still throws away information captured by the fixed per-series
AR(5)+residual-ECDF null. The small positive integrated marginal is largely a
comparison against a harmed clone, not evidence that the adapted representation
beats the project's existing legal null.

**Retry warranted?** No LA-03b. The frozen program forbids adapter-size, width,
architecture, loss, and RT600-conditioned variants. No Leaderboard Alpha
mechanism survived, so the combination rule does not open.


## RT-1240 / RT-1241 / RT-1242 -- CRF-02 ACGN amortized conditional generative null -- KILL (2026-08-26, CORRECTED RUN)

**Supersedes the void `RT-1237`/`RT-1238`/`RT-1239` entry above.** Same frozen
execution preregistration (`CRF02_EXECUTION_PREREG.md` @ `9a3d3c7`, unchanged); the
defect was that the earlier run did not execute it. **Report**
`research/reports/causal_representation_frontier/crf02_acgn.{md,json}`.

**Hypothesis.** Unchanged: a global, amortized, nonlinear, distributional null
learned from break-free histories only prices "normal" better than the project's
fixed per-series historical calibration, and the sequential departure of the online
stream from it is a different discriminative direction.

**The numbers.** Candidate `RT-1240`: standalone whole-fold TS-AUC **0.559140** at
within-`t` rho **+0.2887**, dominant-cell **0.579573**. Null pretrained on 6,383
training-fold series' break-free histories, 10 epochs, pinball `0.364706 ->
0.243881`, 1,364.9 s, state `e58bc20e05d9`, provenance fingerprint verified.

**Two gates failed, one passed:**

1. **Cheap abandon gate** -- `0.559140 < 0.600` AND rho `0.2887 <= 0.60`. FIRED.
2. **Learned-null isolation** -- candidate minus C1 = **`-0.021446`** whole-fold
   (`-0.020337` dominant) against a required `+0.000500`. FAILED.
3. **Derangement** -- candidate minus C2 = **`+0.003377`** whole-fold
   (`+0.008391` dominant). **PASSED.**

`RT-1241`, the **fixed**-null control (AR(5) + 256-knot history residual ECDF, per
series, feeding identical downstream statistics into an identical ranking head):
**0.580586** / **0.599909**. It reproduced **bitwise** from the void run, which is
itself confirmation that its code path never touches the neural null. `RT-1242`,
deranged `h_i`: `0.555763` / `0.571182`.

Pair flow negative in every cell for every arm; candidate pre-break damage rate on
RT600-correct pairs `0.3929` against the `0.0150` cap.

**Why it failed -- and the passing derangement control is what makes this sharp.**
The conditioning **works**: permuting `h_i` across series costs the candidate
`0.0034` whole-fold and `0.0084` on the dominant cell, so the 8-dimensional history
bottleneck carries real series-specific information and the model is not memorising
a series identifier. **And the fixed null still beats it by 0.0214.** The learned
null is conditioning correctly and losing anyway.

The reason is capacity placement, not capability. The fixed null has, in effect,
five AR coefficients plus a 256-knot empirical residual distribution **per series**,
paid for by that series' own break-free history at zero generalisation cost -- the
history is complete at `t = 0` and carries no label. The learned null compresses all
of that into **8 floats** shared across a population whose heterogeneity is the whole
reason this project's foundation is per-series historical calibration.
**Amortizing the null across series loses more than the learned nonlinearity gains.**

The declared no-id shared-8 diagnostic makes the conclusion conservative rather than
flattering: refit on the eight features the fixed-null control also has, the
candidate reaches `0.551432` and loses to `RT-1241` by `-0.029154`, worse than the
headline `-0.021446`.

**What this closes.** Exactly what `CRF_PROGRAM_PREREG.md` 2.9 wrote before any
number existed: **learned amortized generative nulls are closed for this problem.**
The per-series historical calibration the project already ships **is** the right
null -- now measured against a matched, correctly-conditioned learned alternative
rather than assumed. The never-break false-positive mass is **not** a
conditional-null misspecification problem in CRF-02's sense. Because the derangement
control passed, this is a statement about **amortization**, not about a broken
implementation.

Combined with CRF-01, this closes **H-A** (representation saturation), **H-B**
(objective mismatch) and **H-E** (learned-null misspecification), leaving **H-D**.

**Retry warranted?** **No `CRF-02b`.** No MDN, no flow, no deeper head, no different
quantile grid, no second epoch count, no seed re-roll -- forbidden by
`CRF_PROGRAM_PREREG.md` 0.8/0.9 and unsupported: the null converged cleanly, so it
was not under-trained. A **wider bottleneck** is the one thing this result genuinely
motivates, and it is a **different experiment** that this program does not
authorise. It is recorded as a hypothesis in `CRF_FINAL.md`, not run.

**CRF-03 does not open** and the CRF program is exhausted. See `CRF_FINAL.md`.

## Submission #15 -- RT-1257 first cloud attempt -- FAILED TO RUN (2026-08-29)

**Program.** Deep Ensemble Frontier 2026, promotion of `RT-1257`. Not a research
arm: no RT ID was consumed, no model was refit, no score was produced. Recorded
here because two latent packaging defects were exposed and both would otherwise
be rediscovered by the next agent who builds a submission.

**What happened.** `RT-1257` was uploaded as submission #15 (message `r15`,
created 2026-08-29 07:40:47) and exited 1 during `train`, before any scoring:

```
File "/context/code/submissions/RT1257_deployable.py", line 21, in <module>
  os.makedirs(_WORK, exist_ok=True)
PermissionError: [Errno 13] Permission denied: '/context/code/_sbr_payload_110'
```

**Defect 1 -- payload unpacked into the code tree.** The self-extracting
entrypoint generated by `research/scripts/build_submission.py` set
`_WORK = os.path.abspath(f"./_sbr_payload_{os.getpid()}")`. `"./"` is the process
working directory, which the cloud runner mounts **read-only** at `/context/code`.
This was latent in every artifact this builder has ever produced.

**Defect 2 -- inference dependency never declared.** Found by re-testing after
Defect 1 was fixed, i.e. it was queued behind it and would have failed the *next*
submission:

```
ModuleNotFoundError: No module named 'catboost'
  sbr/production/model.py:92  from catboost import CatBoostClassifier
```

`catboost` was declared only in `requirements-rt1257.txt`, a non-standard filename
the runner has no reason to read. `requirements.txt`, which does ship in the code
payload, declared neither `catboost` nor `lightgbm`. RT-600 scored 0.6268 with
seven **undeclared** LightGBM boosters, so the runner image evidently supplies
lightgbm -- nothing indicates it supplies catboost, and `RT-1257` is the first
artifact to need it.

**Why local qualification passed anyway -- this is the real finding.**
`CRUNCH_TEST.json` records `result: PASSED`, `determinism_check: passed`, 02:46
duration and a clean 2.2775 ms/pt benchmark for this exact artifact. It certified
determinism, memory and runtime, and it certified none of what actually broke,
because **`crunch test` runs against a writable working directory and whatever
interpreter happens to be active.** It cannot certify the runner environment. A
green local test is evidence about the model, not about the deployment.

**Fix.** `733c727` -- `_WORK = tempfile.mkdtemp(prefix=f"_sbr_payload_{os.getpid()}_")`,
which honours `TMPDIR` and falls back to `/tmp`. `e50098a` -- declare
`lightgbm==4.7.0` and `catboost==1.2.10` in `requirements.txt`. The lightgbm pin is
deliberate: `FINAL_ARCHITECTURE_FREEZE.md` records that the RT-125R/GOSS stream is
only bitwise reproducible under 4.7.0.

**Verification.** Defect 1 is verified by importing the rebuilt entrypoint with the
working directory `chmod 555`: import succeeds, `_WORK` resolves into `TMPDIR`
outside the tree, all eight model files unpack, `train`/`infer` present. **Defect 2
is reasoned, not verified** -- neither package is importable in the interpreter
available on this machine, which is itself how the gap survived qualification.

**Nothing about the model changed.** Across all three rebuilds the embedded source
zip (`c8e3585...`), model zip (`952422d...`) and feature manifest (`1646c3b...`)
sha256 are byte-identical. Packaging only.

**Cost.** Effectively zero quota: #15 failed at import, before `infer()`. Under
best-kept leaderboard scoring the external anchor 0.6268 was never at risk.

**Retry warranted?** Already retried as submission #16 with both fixes. No model,
feature, hyperparameter or blend change is authorized by any of this.

**Standing recommendation for the next builder.** Fold the read-only import check
into `build_submission.py` as a build-time gate -- unpack the freshly built
entrypoint from a `chmod 555` directory and refuse to emit the artifact if it
raises. It costs about a second and would have caught Defect 1 before the first
submission. More generally: do not treat a passing `crunch test` as deployment
qualification.
