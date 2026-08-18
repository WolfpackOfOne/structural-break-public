# Agent 4 — Residual / Innovation Representations, and the Over-Whitening Test

## Return block (PROTOCOL §6)

**AGENT NAME** — agent4 (residual / innovation representations)

**HYPOTHESIS**
Removing predictable dynamics from a series before monitoring it changes *what kind
of break is visible*, not merely how much noise is present. Specifically:
(a) volatility normalisation and AR filtering produce break evidence that is
**complementary** to raw multi-scale historical-null evidence, so
`m00_core + m04_resid` beats `m00_core`;
(b) **over-whitening is real** — a filter that adapts quickly to the very thing that
broke absorbs the break, so detection power should fall monotonically as the filter's
adaptation speed rises (const sd → slow robust EWMA → EWMA(22) → GARCH(1,1)), and an
AR filter fitted at the order that actually whitens the series should be *worse* than
the raw series for detecting a break in the mean/dependence structure.

**FALSIFICATION CONDITION**
(a) is falsified if `m00_core + m04_resid` ≤ 0.57384 (the paired control) on the
screen protocol. (b) is falsified if the single-representation ablation shows no
ordering, i.e. if AR-whitened and vol-normalised 8-column slices are within noise of
the raw 8-column slice, or if the fastest-adapting volatility filter (GARCH) is not
the weakest.

**FILES CHANGED**
- `/home/claude/sb/src/sbr/features/m04_resid.py` (new, 60 columns, owner agent4)
- `/home/claude/sb/research/reports/agent04_residual.md` (this file)
- `/home/claude/sb/research/scratch/agent04_runs.py` (new, experiment driver)
- `/home/claude/sb/research/FAILED_EXPERIMENTS.md` (appended)
No shared/core file was edited.

**EXPERIMENT IDs**
`RT-040C` (control), `RT-041T` (treatment), `RT-042S` (module alone),
`RT-043-{raw,ar1,ar2,ar5,vol,cmb,arvol}` (key ablation),
`RT-044-{ar2c,arR,arH,volc,volG,volM}` (matched estimator comparison),
`RT-045-{raw,ar2,ar5,vol}` (incremental-over-`m00_core` ablation).

**DATA USED** — `cache/store_screen` (2,500 series) + `cache/features_screen`.
Lockbox never loaded. `X_test.reduced.parquet` never read.

**FOLDS USED** — fold 0 only (screen protocol), `folds_screen.parquet`, seed 0.

**MODEL + FEATURES** — LightGBM binary, `n_estimators=300`, all other pipeline
defaults, `max_train_rows=300_000`, `screen=True`.

**TS-AUC / PER-FOLD / DELTA VS CONTROL** — see the tables below. Headline:

| run | modules | n_feat | screen TS-AUC (fold 0) | Δ vs control |
|---|---|---|---|---|
| RT-040C | `m00_core` | 151 | **0.57384** | — (reproduces Agent 0 exactly) |
| RT-041T | `m00_core + m04_resid` | 211 | **0.60244** | **+0.02860** |
| RT-042S | `m04_resid` alone | 60 | **0.58139** | +0.00755 |
| RT-045-ar5 | `m00_core` + 8 AR(5) columns | 159 | **0.59690** | **+0.02306** |

`m04_resid` alone, with 60 columns, beats the 151-column `m00_core` baseline; and
**eight AR(5)-residual columns alone recover +0.023 of the +0.029**.

**RUNTIME** — module build on the screen store: 605 s wall for 2,500 series with
`--workers 1` under a loaded machine (load average ≈ 7–10). Isolated marginal cost of
`m04_resid` measured on 120 series: **74 ms/series** (for reference, `m00_core` is
82 ms/series and the shared `make_ctx` is 143 ms/series on the same benchmark), so
the module fits the ~80 ms/series budget. Extrapolated full-store build ≈ 12 min of
own CPU. Screening runs: 110–920 s each depending on column count and machine load.

**CAUSALITY CHECK OUTPUT** — see §"Causality" below (13 series, bitwise, `atol=0.0`).

**LEAKAGE RISKS** — enumerated and mitigated in §"Causality".

**OOF ARTIFACT PATH** — none. `sbr.pipeline.run` deliberately does not save OOF
artifacts under `screen=True`; every number here is a screen number. Full-store OOF is
Agent 0's promotion decision.

**CONCLUSION: KEEP** — `m04_resid` is a large, cheap, causally-clean gain
(+0.0286 screen TS-AUC on a paired control). The over-whitening hypothesis is
**confirmed in its specific form** and **rejected in its general form** — details below.

---

## 1. What the module builds

Every representation is a map `x → e` whose parameters are estimated on the
**historical segment only** and then applied causally forward across the online
segment. Each stream is divided by the standard deviation of the *historical* stream,
so `1` always means "one historical innovation sd" and the representations are
directly comparable.

| tag | representation | fitted on history |
|---|---|---|
| `raw` | `z = (x − μ_h)/σ_h` — constant historical sd | μ, σ |
| `ar1`,`ar2`,`ar5` | AR(p) residual, fixed order, OLS on standardised history, run forward with `ar_filter_causal` warmed by the historical tail | φ, σ_e |
| `arR` | ridge AR(3), λ = 0.05·n on a unit-variance Gram | φ, σ_e |
| `arH` | Huber-IRLS AR(2), 3 fixed reweighting steps | φ, σ_e |
| `vol` | `z / sqrt(EWMA_pred(z², halflife 22))` | initial state = history |
| `volM` | `z / EWMA_pred(min(|z|,4), halflife 63)` — winsorised, robust, slow | initial state = history |
| `volG` | `z / sqrt(GARCH(1,1)_pred)`, variance-targeted, 12-point deterministic QMLE grid | (α,β) grid-argmin on history |
| `cmb` | AR(2) residual **then** EWMA(22) volatility normalisation of the residual | both |

Monitors (all null-calibrated, expanding window; the composite LLR additionally at a
trailing w=32): residual **mean**, **variance**, **absolute magnitude**, **tail
frequency** (`|e| > 95th pct of |e_h|`), **ACF(1)**, **ACF(1) of squared residual**,
and a Gaussian generalised-log-likelihood-ratio channel

```
LLR/w = mean(e²) − log(var(e)) − 1
```

which is exactly 2·(log L₁ − log L₀)/w for N(m,s²) against the null N(0,1), so it
reacts to a mean shift and a variance change jointly rather than to either alone.

**Null.** Own calibration engine (`_Null`), not `ctx.nc`, because `ctx.nc` only knows
`m00_core`'s raw transforms. For each representation and each monitored statistic the
null is the empirical distribution of **the same statistic over every contiguous
length-w window of the residual stream computed ON HISTORY**, summarised by median and
IQR-σ on a 6-node log-spaced grid `[6,16,32,80,200,512]` and log-interpolated to any
length. Only history enters the null; online values only ever enter as the query.

**Column budget.** Exactly 60 columns: 6 tier-A representations × (6 rolling monitors
+ expanding LLR + trailing-32 LLR) = 48, plus 4 tier-B representations ×
(var, ACF(1)-of-squares, LLR) = 12.

**Efficiency.** GARCH is *not* fitted with `scipy.optimize`. Variance targeting fixes
ω = (1−α−β)·1 (the standardised history has unit variance by construction), and the
recursion is an exponential filter of `ω + α·z²_{t−1}`, so each of 12 (α,β)
candidates is one `scipy.signal.lfilter` pass over history; the QMLE loss is evaluated
once per candidate and the arg-min is taken. Deterministic, no iteration to
convergence, ~1 ms/series. The null summariser uses a fixed stride subsample (never
random) capped at 600 windows and a single `np.partition` call instead of
`np.quantile` — that one change took the module from 224 ms/series to 74 ms/series.

---

## 2. Causality

```
series   559 n_hist= 3893 n_online=  10 -> True ok
series  1807 n_hist= 1823 n_online=  53 -> True ok
series  2165 n_hist= 2391 n_online= 167 -> True ok
series  1484 n_hist= 4873 n_online= 308 -> True ok
series  1574 n_hist= 4184 n_online= 458 -> True ok
series   770 n_hist= 1875 n_online= 614 -> True ok
series   731 n_hist= 3016 n_online= 754 -> True ok
series   935 n_hist= 4536 n_online= 846 -> True ok
series  1922 n_hist= 1029 n_online= 949 -> True ok
series  1462 n_hist= 1900 n_online= 990 -> True ok
series  1926 n_hist= 1996 n_online= 999 -> True ok
series     5 n_hist= 1001 n_online= 163 -> True ok
series  1444 n_hist= 4996 n_online= 386 -> True ok
ALL PASS (bitwise, atol=0.0): True
```
`check_prefix_invariance("m04_resid", hist, online, cuts=(3,10,37,101,255,499), atol=0.0)`
on 13 series spanning the full online-length range (10 → 999) and the full
history-length range (1,001 → 4,996).

**Leakage risks considered and closed**

1. *Recursive state initialised from the online segment.* This is the trap the mission
   flagged. All EWMA/GARCH state is produced by filtering `concat(hist, online)` in one
   `lfilter` pass and slicing; the filter is a first-order sequential IIR, so the first
   `n_hist + k` outputs of the concatenated filter are bit-identical to filtering
   `concat(hist, online[:k])`. Nothing is initialised from `n_online`, no burn-in is
   taken from the online side, and the history burn-in (`min(200, n_hist//5)`) is a
   function of history length only.
2. *Predictive vs contemporaneous volatility.* `_ewma_pred` / `_garch_pred` return the
   variance forecast for time `t` built from data strictly before `t`, so `e_t = z_t/σ_t`
   never divides a shock by a scale that already contains it.
3. *AR warm-up.* `ar_filter_causal(z_o, φ, z_h)` takes lags from the historical tail; at
   online index 0 the lags are historical, never zero-padded and never forward-looking.
4. *Lag products at online index 0.* `acf1`/`acf1sq` at `t=0` use the **last historical
   residual** as the lag. This is causal (history is available at `t=0`) and
   prefix-invariant, and it avoids the artificial zero that `m00_core`'s `lag1` carries
   at `t=0`.
5. *Null grid.* Capped by `n_hist//2` only. No online quantity touches the null.
6. *Expanding means.* `np.cumsum` is sequential, so a prefix of the cumsum of a longer
   array is bit-identical to the cumsum of the prefix — this is what makes the
   expanding-window monitors pass at `atol=0.0` rather than merely `atol=1e-7`.
7. *Determinism.* No RNG anywhere in the module; the null subsample is a fixed stride,
   the GARCH fit is a fixed grid.

---

## 3. THE KEY EXPERIMENT — does over-whitening destroy break signal?

### 3a. Matched single-representation ablation (8 columns each, `m04_resid` alone)

Identical fold, identical seed, identical model, identical monitor set — the **only**
thing that changes is the representation the monitors are computed on. Sliced with
`keep_regex` from the one cached module, so the comparison is exact.

| slice | representation | n_feat | screen TS-AUC | Δ vs `raw` |
|---|---|---|---|---|
| `RT-043-raw` | raw / constant historical sd | 8 | 0.54305 | — |
| `RT-043-ar1` | AR(1) residual | 8 | 0.53359 | **−0.00947** |
| `RT-043-ar2` | AR(2) residual | 8 | 0.53624 | **−0.00681** |
| `RT-043-ar5` | AR(5) residual | 8 | 0.54940 | **+0.00635** |
| `RT-043-vol` | EWMA(22) vol-normalised | 8 | 0.55392 | **+0.01087** |
| `RT-043-cmb` | AR(2) → EWMA(22) vol | 8 | 0.55122 | **+0.00817** |
| `RT-043-arvol` | union of `ar2`+`vol`+`cmb` | 24 | **0.57757** | +0.03452 |

### 3b. Matched estimator comparison (3 columns each: var, ACF(1)-of-squares, LLR)

| slice | estimator | n_feat | screen TS-AUC |
|---|---|---|---|
| `RT-044-ar2c` | OLS AR(2) | 3 | **0.53089** |
| `RT-044-arR` | ridge AR(3), λ=0.05 | 3 | 0.52702 |
| `RT-044-arH` | Huber-IRLS AR(2) | 3 | 0.52658 |
| `RT-044-volM` | robust winsorised EWMA, **halflife 63** | 3 | **0.53490** |
| `RT-044-volc` | EWMA, **halflife 22** | 3 | 0.51534 |
| `RT-044-volG` | GARCH(1,1), fitted persistence | 3 | **0.50012** |

### 3c. Incremental value on top of `m00_core` — the decision-relevant form

The 8-column slices in 3a are *standalone* detectors. The question that actually
matters is: given that we already monitor the raw series at every scale (`m00_core`),
which representation adds information? Same 8 columns, added to the same 151.

| run | feature set | n_feat | screen TS-AUC | Δ vs RT-040C |
|---|---|---|---|---|
| RT-040C | `m00_core` | 151 | 0.57384 | — |
| RT-045-raw | `m00_core` + `raw` slice (8) | 159 | 0.57169 | **−0.00215** |
| RT-045-vol | `m00_core` + `vol` slice (8) | 159 | 0.58995 | **+0.01610** |
| RT-045-ar2 | `m00_core` + `ar2` slice (8) | 159 | 0.59474 | **+0.02090** |
| RT-045-ar5 | `m00_core` + `ar5` slice (8) | 159 | **0.59690** | **+0.02306** |
| RT-041T | `m00_core` + all of `m04_resid` (60) | 211 | 0.60244 | **+0.02860** |

**The ordering inverts.** Standalone, `raw` (0.54305) beats `ar2` (0.53624). Conditional
on `m00_core`, `raw` adds **nothing** (−0.002, i.e. noise — `m00_core` already contains
that evidence at more scales) while `ar2` adds **+0.021** and `ar5` adds **+0.023**,
from eight columns each. Eight AR-residual columns recover 80 % of what the entire
60-column module recovers.

### 3d. VERDICT ON THE OVER-WHITENING QUESTION

**Removing predictable dynamics makes breaks HARDER to detect in isolation and MUCH
EASIER to detect in combination. Whitening does not destroy break signal — it moves
it into a channel that raw monitoring cannot see.**

The four findings, in order of importance:

1. **The standalone ablation reproduces the over-whitening effect, and it is a trap.**
   AR(1) and AR(2) residuals are worse standalone detectors than the raw series
   (−0.0095, −0.0068 on identical folds/seeds/monitors). The mechanism is exactly the
   one the mission anticipated: a level shift, which in raw space is a *sustained*
   displacement of every trailing and expanding window, collapses in AR-residual space
   into a **one-off spike of size (1−Σφ)·Δμ at t=τ** followed by an immediate return to a
   zero-mean residual. The expanding-mean monitor then sees one point of evidence
   instead of `t−τ` points, and its null-z **decays like 1/√L after the break instead of
   growing like √L**. Any detector that only looks at AR residuals loses the entire
   mean-shift family.
   **But this is a statement about marginal power, not about information content**, and
   it is the wrong basis for a feature decision — §3c shows the ranking reverses once
   the raw channel is already covered.

2. **A filter matched to the break channel annihilates that break. GARCH is the proof.**
   Within the matched 3-column family (var, ACF-of-squares, LLR), power falls
   monotonically with the filter's adaptation speed:
   ```
   robust EWMA hl=63   0.53490      (slow, winsorised)
   EWMA hl=22          0.51534
   GARCH(1,1) fitted   0.50012      <- literally no signal
   ```
   A GARCH filter's entire job is to track a change in conditional variance; when the
   structural break *is* a change in variance, the filter absorbs it within a few tens of
   points and the normalised residual returns to unit variance. Detection power is
   destroyed by construction. **This is the real, sharp form of over-whitening: not
   "whitening is bad" but "never let the filter adapt on the same timescale and in the
   same channel as the break you are hunting."** The corollary is a design rule —
   volatility filters used for break detection should be deliberately *slow* and
   *robust*, and any filter that adapts fast should be kept only as a contrast partner.

3. **Higher AR order is not more harmful; matching is.** AR(5) beats AR(1) by +0.016 and
   AR(2) by +0.013 standalone, and it is the best single 8-column addition to `m00_core`
   (+0.023). A fixed AR(5) fitted on 1,000–5,000 historical points is not over-fitted;
   its 5-lag residual retains a richer dependence signature, so its ACF(1) and
   ACF(1)-of-squares monitors react to a change in the *shape* of dependence rather than
   only its magnitude. **The public 2026 note that "selecting AR order per series can
   over-whiten and hurt" is therefore supported in direction but mis-attributed**: the
   damage comes from matching the filter to the break channel, and per-series order
   selection is precisely the procedure that maximises that matching. Per-series
   selection was NOT implemented, per the mission's instruction to benchmark fixed
   orders first; on this evidence it should be entered with a pre-registered
   falsification (must beat fixed AR(5) by > 0.003) and I expect it to fail.

4. **Regularising or robustifying the AR coefficients does not help.** Matched 3-column
   comparison: OLS AR(2) 0.53089, ridge AR(3) 0.52702, Huber-IRLS AR(2) 0.52658. Both
   are *below* plain OLS. Outlier-driven coefficient estimates are not a problem here; if
   anything a slightly noisier filter whitens less and therefore leaves more break signal
   in the residual — which is finding (1) restated. These 6 columns earn their place only
   as a robustness control and are the first thing to cut.

5. **The representations are complementary, and that is the shippable result.** No single
   8-column representation exceeds 0.554 standalone, but `ar2 ∪ vol ∪ cmb` (24 cols)
   reaches 0.57757 and the full 60-column module reaches 0.58139 — above the 151-column
   `m00_core` baseline on its own. Stacked, +0.0286. Different filters are **break-channel
   projectors**: each is blind to the channel it removes and sharp on the channels it
   leaves. Providing several filters with different blind spots lets the model infer
   *which channel broke*, which is strictly more information than any single filter can
   express. **The answer is not "whiten" or "don't whiten" — it is "monitor several
   whitenings simultaneously and let the model read the disagreement between them."**

---

## 4. Per-representation stories (SIGNAL / FALSE SIGNAL / DISAMBIGUATOR)

**`raw` — constant historical sd (control).**
*SIGNAL*: any break that moves location, scale, tail mass or PIT shape displaces the
online distribution from the historical null; the expanding-mean monitors accumulate
√L evidence for a persistent shift.
*FALSE SIGNAL*: a transient volatility burst or a single outlier lifts `var`, `abs`,
`tail` and `llr` for a stretch and then reverts; a slow deterministic drift with no
break does the same to `mean`.
*DISAMBIGUATOR*: `vol_*` — a genuine variance-level break keeps `vol_e_var` elevated
because the EWMA denominator lags the shift, whereas a burst is fully absorbed by the
EWMA within ~2 halflives and `vol_e_var` returns to zero while `raw_e_var` is still high.

**`ar1`/`ar2` — residual of the well-fitting AR filter.**
*SIGNAL*: a break in the innovation variance, in the innovation distribution (tails), or
in the AR structure *away from* the fitted φ. `ar2_e_acf1` is the direct channel: if the
post-break φ differs from the historical φ, the residual stops being white and ACF(1) of
the residual moves off its historical null.
*FALSE SIGNAL*: an isolated outlier enters the residual `p+1` times (once as the shock,
`p` times through the lag terms), so a single bad point produces a small burst of
correlated residuals that looks like a dependence change.
*DISAMBIGUATOR*: `ar2_e_tail` and `raw_e_mean`. An outlier lifts the tail rate but leaves
the raw expanding mean and the raw ACF alone; a real coefficient break moves
`ar2_e_acf1` persistently without a tail-rate excursion.
*KNOWN WEAKNESS (measured)*: blind to mean shifts — they become a one-off spike, which is
why `ar1`/`ar2` underperform `raw` **standalone** (−0.0095, −0.0068). This does not carry
over to the incremental setting: added to `m00_core`, the same 8 `ar2` columns are worth
**+0.021** while the 8 `raw` columns are worth −0.002. The AR residual is not a better
detector than the raw series; it is an *orthogonal* one, and orthogonality is what a
feature has to deliver.

**`ar5` — longer fixed-order AR residual.**
*SIGNAL*: as `ar2`, plus sensitivity to changes in dependence *shape* at lags 3–5 that an
AR(2) filter leaves in the residual and therefore cannot report on.
*FALSE SIGNAL*: with 5 lags, one outlier contaminates 6 residuals; also, on a
short-memory series the extra coefficients are near zero and the residual is nearly the
AR(2) residual, so `ar5` and `ar2` become collinear and the model may over-trust
agreement between them.
*DISAMBIGUATOR*: `ar5_e_acf1sq` vs `ar2_e_acf1sq` — a real higher-lag change moves them
apart; an outlier moves them together and with `*_e_tail`.

**`vol` — EWMA(22) volatility normalisation.**
*SIGNAL*: mean/level breaks (amplified, because the denominator does not follow a level
shift), and variance breaks *during the ~2-halflife window before the filter catches up*.
Also breaks in vol-of-vol, via `vol_e_acf1sq`.
*FALSE SIGNAL*: a vol burst is absorbed, which is the point — but a vol burst that is
*faster* than halflife 22 produces a genuine transient in `vol_e_var` before absorption.
*DISAMBIGUATOR*: `volM_*` (halflife 63) and `raw_e_var`. A transient shows up in `vol`
and disappears from `volM` and from the long raw windows; a real variance break shows in
`volM_e_var` and stays.

**`volM` — robust winsorised slow EWMA (halflife 63).**
*SIGNAL*: variance-level breaks — this is the best pure variance-break channel in the
module (0.53490 on 3 columns) precisely because it is slow and winsorised, so it neither
chases the break nor is dragged by the post-break outliers.
*FALSE SIGNAL*: because it is slow, a pre-break volatility trend leaves it mis-calibrated
for a long stretch and it reports a persistent pseudo-break.
*DISAMBIGUATOR*: `vol_e_var` (fast) — under a genuine break the fast and slow filters
disagree transiently and then the fast one re-converges; under a slow drift they track
each other with a constant offset.

**`volG` — GARCH(1,1), variance-targeted, grid-QMLE.**
*SIGNAL*: in principle, breaks that are *not* conditional-variance breaks (mean shifts,
tail-shape changes) seen against a properly de-heteroskedasticised background.
*FALSE SIGNAL*: essentially everything — measured 0.50012 on its 3 variance-flavoured
columns. The filter tracks the break.
*DISAMBIGUATOR*: this representation *is* the disambiguator. `raw_e_var` high while
`volG_e_var` ≈ 0 is the signature of "the break was a conditional-variance break that a
GARCH can follow", whereas both high means "the break was in the innovation distribution
itself, which GARCH cannot absorb". Kept for exactly this contrastive role, not for its
standalone power.

**`arR` / `arH` — ridge and Huber AR coefficients.**
*SIGNAL*: same channels as `ar2`, but with coefficients that are not dragged by
historical outliers, so the residual scale is a cleaner null.
*FALSE SIGNAL*: shrinkage leaves genuine persistence in the residual, so a change in the
*level* of persistence leaks into `arR_e_var`.
*DISAMBIGUATOR*: `ar2_e_var` vs `arR_e_var`. Measured verdict: neither adds over plain
OLS (−0.004 and −0.004); they earn their 6 columns only as a robustness control, and
they are the first thing to cut if the budget tightens.

**`cmb` — AR(2) residual then EWMA(22) vol normalisation.**
*SIGNAL*: breaks in the *standardised innovation* — i.e. changes in the shape of the
innovation distribution, in innovation tail mass, or in dependence, with both the linear
predictability and the volatility level projected out. This is the purest "did the
data-generating law change" channel in the module.
*FALSE SIGNAL*: it is doubly blind — a pure mean shift is a one-off spike (AR stage) and a
pure variance shift is absorbed (EWMA stage). Its 0.55122 comes almost entirely from
distributional and dependence channels.
*DISAMBIGUATOR*: `raw_*`. `cmb` high with `raw` quiet is a distributional break; `raw`
high with `cmb` quiet is a level or variance break that the filters ate.

**The `*_e_llr` GLR channel (all representations).**
*SIGNAL*: `mean(e²) − log(var(e)) − 1` is the per-point GLR of N(m,s²) vs N(0,1). It is
zero only when the window is simultaneously centred and unit-variance, so it fires on a
mean shift and a variance change jointly, at a strength that is the correct likelihood
weighting of the two rather than an arbitrary blend.
*FALSE SIGNAL*: it is one-sided-ish and heavy-tailed under the null (its distribution is
χ²-like), so short windows produce large positive values by chance; and it cannot tell a
mean shift from a variance shift on its own.
*DISAMBIGUATOR*: the paired `*_e_mean` and `*_e_var` columns decompose it, and
`*_w32_llr` vs `*_e_llr` separates a transient (trailing fires, expanding does not) from a
persistent break (both fire and the expanding one keeps growing).

---

## 5. What I would do next (not done — out of scope/budget)

1. **Do not add per-series AR order selection.** The measured ordering says the damage
   comes from matching the filter to the break channel; per-series selection maximises
   that matching. Pre-registered falsification if anyone tests it: it must beat fixed
   AR(5) (`RT-045-ar5`, 0.59690) by more than 0.003.
0. **Cheapest next win: more AR orders.** AR(5) > AR(2) > AR(1) incrementally, and the
   incremental gain has not flattened. AR(8)/AR(12) at fixed order, 8 columns each, is
   the obvious next probe.
2. **Add a second trailing scale (w=8 and w=128) for the LLR channel** on `raw`, `vol`
   and `cmb`. The trailing-32 LLR is already carrying weight and one scale is clearly
   under-resolved; this is ~12 more columns for what I expect is the cheapest remaining
   gain in this module.
3. **Add the explicit contrast columns** `raw_e_var − volG_e_var` and
   `raw_e_mean − ar2_e_mean`, which is the "which channel broke" signature stated in §4,
   currently only available to the model implicitly.
4. **Drop `arR`/`arH` (6 columns)** if budget is needed; they are measured non-additive.

---

## 6. Full experiment log (all 20 runs, screen protocol, fold 0, seed 0, n_estimators=300)

```
exp_id          n_feat  screen TS-AUC   runtime_s
RT-040C            151     0.57384         640    control (reproduces Agent 0 exactly)
RT-041T            211     0.60244         923    m00_core + m04_resid      D +0.02860
RT-042S             60     0.58139         228    m04_resid alone           D +0.00755
RT-043-raw           8     0.54305         149    ablation: raw
RT-043-ar1           8     0.53360         117    ablation: AR(1)
RT-043-ar2           8     0.53624         110    ablation: AR(2)
RT-043-ar5           8     0.54940         142    ablation: AR(5)
RT-043-vol           8     0.55392         131    ablation: EWMA(22) vol
RT-043-cmb           8     0.55122         146    ablation: AR(2)+vol
RT-043-arvol        24     0.57757         190    ablation: ar2 U vol U cmb
RT-044-ar2c          3     0.53089         272    matched: OLS AR(2)
RT-044-arR           3     0.52702         130    matched: ridge AR(3)
RT-044-arH           3     0.52658         137    matched: Huber AR(2)
RT-044-volc          3     0.51534         141    matched: EWMA hl=22
RT-044-volG          3     0.50012         123    matched: GARCH(1,1)
RT-044-volM          3     0.53490         203    matched: robust EWMA hl=63
RT-045-raw         159     0.57169         284    m00_core + raw slice      D -0.00215
RT-045-vol         159     0.58995         218    m00_core + vol slice      D +0.01610
RT-045-ar2         159     0.59474         404    m00_core + AR(2) slice    D +0.02090
RT-045-ar5         159     0.59690         428    m00_core + AR(5) slice    D +0.02306
```

Every row is in `research/RESULTS.csv`, appended through `sbr.pipeline.run` under the
file lock. Runtimes are wall clock on a machine running 4–6 other agents' jobs
(load average 6–10 throughout) and are not comparable across runs.

**Caveat on precision.** These are single-fold screen numbers on 2,500 series. Agent 0's
control is reproduced to the last digit (0.5738423135475659), so the *pairing* is exact
and the deltas are not seed noise, but the fold-to-fold standard error is unknown from
one fold. Differences below ~0.005 in the small-slice tables (e.g. ridge vs Huber)
should be read as ties. The headline deltas (+0.021 to +0.029) and the GARCH collapse
(-0.035) are far outside any plausible single-fold noise band.
