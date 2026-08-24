# RED TEAM 4 — Generator artifacts in the 500-column feature bank

**Assignment.** Assume TRUE: *"the feature bank contains columns that predict the
label through a property of how the data was SYNTHESISED rather than through
genuine structural-break evidence, so 0.615 is partly an artifact score that will
not transfer."* Go find the evidence.

**Verdict: REJECTED** for the bank as built, with one large caveat that is not
about the bank at all (§1). Every direct test of the hypothesis came back at
chance. The bank does not know `has_break` before a break happens, does not
detect fake breaks, is bitwise causal at cut lengths the shipped harness never
exercises, and reproduces from source exactly. The real artifact in this dataset
is enormous, is *outside* the bank, and the bank is verified clean of it.

All work on dev folds (`fold >= 0`, 8,000 series). Lockbox never loaded, never
scored. `X_test.reduced.parquet` never read. Nothing outside this file written.
Scratch code in `/tmp/rt4/`.

---

## 1. HEADLINE — the biggest artifact in this dataset is `n_online`, and it is worth more than the entire champion

`tau_index` is ~uniform on the online segment, so at timestep `t` the probability
a still-alive series has already broken is `min(1, (t+1)/n_online)`. That makes
series length a direct label oracle *within* every timestep — exactly the object
TS-AUC scores.

Measured on dev rows with the official metric:

| score | dev TS-AUC |
|---|---|
| champion `RT-100` (500 columns, 600 trees) | **0.61500** |
| 7-stream ensemble `RT-131` (for reference, from STATE) | 0.62541 |
| **illegal oracle: `-n_online`** | **0.62948** |
| **illegal oracle: `(t+1)/n_online`** | **0.62948** |
| 50/50 within-timestep rank blend of champion + oracle | **0.66625** |

One integer that a legal `infer()` cannot observe beats 500 engineered columns by
+0.0145 and beats the seven-stream ensemble by +0.0041. Per-timestep, the oracle's
series-level AUC is 0.875 at t=0, 0.766 at t=20, 0.723 at t=50, 0.694 at t=100,
0.650 at t=200, 0.602 at t=400.

This is known in principle — `runner_semantics.md` F4 records that the organisers
invalidated all pre-2026-06-08 predictions over exactly this leak — but it had
never been priced. **The price is +0.0145 to +0.0513.** Two consequences the team
should act on:

1. **The guard is load-bearing and must be quantitative, not just a code review.**
   Any new feature that acquires even a weak correlation with `n_online` gets a
   large spurious OOF gain and will be selected for. A feature is not safe because
   nobody wrote `n_online` in the code; it is safe when its correlation with
   `n_online` is measured at zero.
2. **I verified the current bank is clean of it** (below). That verification should
   become a standing gate, run on every new stream, not a one-off.

**The champion has not learned `n_online`.** Among not-yet-broken dev series at a
fixed `t`, the champion's score has essentially no information about series length:

| t | n series | AUC(score → `n_online` > median) | Spearman(score, `n_online`) |
|---|---|---|---|
| 0 | 7,975 | 0.5037 | −0.0004 |
| 20 | 7,570 | 0.5142 | +0.0124 |
| 50 | 6,993 | 0.4930 | −0.0098 |
| 100 | 6,286 | 0.4905 | −0.0140 |
| 200 | 5,118 | 0.4879 | −0.0174 |
| 400 | 3,319 | 0.5083 | −0.0052 |

And the *history* — the only thing available at t=0 — carries no `n_online`
information either: a 5-fold LightGBM on 26 history-only statistics predicts
`n_online > median` at series-level AUC **0.4981** (n = 8,000). So the leak is not
reachable through the front door either.

**Category: (a)-shaped but unreachable.** The structure exists in the organiser's
test set too (same generator), so it is not a training-only artifact. It is simply
illegal to use under the single-pass runner. Nothing to keep, nothing to fix — but
a permanent measurement obligation.

---

## 2. Line of attack 4 — column-level leakage screen at t = 0. **CLEAN.**

At t = 0 a break can have produced at most one observation, so anything predicting
eventual `has_break` there is either history leakage or generator bookkeeping.

I extracted the full 500-column cross-section at t ∈ {0, 1, 4, 19, 99} for all
8,000 dev series and computed series-level AUC for `has_break`, both over all
series and restricted to series whose break has *not yet* occurred (`tau < 0` or
`tau > t`) — the pure future-leakage version.

**Univariate.** Max |AUC − 0.5| over all 500 columns:

| t | max &#124;AUC−0.5&#124; | mean &#124;AUC−0.5&#124; | cols > 0.53 | cols > 0.55 |
|---|---|---|---|---|
| 0 | 0.0208 | 0.0035 | 0 | 0 |
| 1 | 0.0222 | 0.0041 | 0 | 0 |
| 4 | 0.0216 | 0.0051 | 0 | 0 |
| 19 | 0.0151 | 0.0038 | 0 | 0 |
| 99 | 0.0195 | 0.0051 | 0 | 0 |

Null SE of a series-level AUC at n₁ ≈ n₀ ≈ 4,000 is ≈ 0.0065, and the expected
maximum of 500 draws is ≈ 3.1 SE ≈ 0.020. The observed maximum is 0.021. **This is
the null distribution, to two significant figures.** The most extreme column at
t = 0 is `m07_bayes::ev_vc_z` at AUC 0.4792 (gain rank 102) — wrong-signed and
inside noise.

**Multivariate**, which is the test that actually matters (a leak can be spread
across columns). LightGBM, 15 leaves, 200 rounds, 5-fold series-level CV on the
existing folds, all 500 columns of a single timestep:

| t | OOF series-AUC for `has_break` | restricted to not-yet-broken |
|---|---|---|
| 0 | **0.4952** | 0.4950 |
| 4 | **0.4974** | 0.4964 |
| 19 | **0.4915** | 0.4955 |

All at or below chance. **The bank cannot see the future.** This is the single
cleanest refutation of the hypothesis I obtained.

**History-only control** (stronger than Wave 1's, which only tested `n_hist`):
26 statistics of the historical segment — 7 autocorrelations, 4 squared-series
autocorrelations, absolute-value ACF, 7 quantiles, min/max/range/MAD, kurtosis,
skew, tail mass, zero-crossing rate, first-half/second-half variance ratio.
Best univariate |AUC − 0.5| = 0.0115 (`acf3`, 0.5115). Multivariate 5-fold OOF
series-AUC for `has_break` = **0.5106** (n = 8,000, SE 0.0065 → 1.6σ). Consistent
with `m05_ctx`'s rejection at full scale. **No history-only label leakage.**

---

## 3. Lines 1 + 3 — τ / sample-size encoding, tested by placebo. **CLEAN.**

The claim to kill: features encode `n_post` (how many post-break points exist)
rather than what changed, so the model is a τ-detector.

**The structural argument first.** TS-AUC stratifies by `t`. Inside a timestep,
every alive series has seen exactly `t+1` online points. Total sample size is
therefore *matched by the metric itself*, and a pure sample-size artifact carries
zero weight. What survives stratification is only "does this series' prefix look
like it contains a regime boundary" — which is detection, not artifact. The
τ-encoding warning in `STATE_OF_RESEARCH.md` is a warning about *row-AUC* and
about uncalibrated statistics; it is largely defused by the metric.

**The empirical test.** I ran a placebo directly on the champion's OOF, which is
the only version of line 3 that is meaningful here (the feature modules never see
τ, so you cannot "recompute features with a fake τ" — you can only relabel):

- take the 4,025 **no-break** dev series (2,015,775 rows) — no break exists anywhere;
- draw a fake `τ_i ~ Uniform{0, …, n_online_i − 1}`, the real τ distribution;
- relabel `y[t] = 1[t ≥ τ_i]` and score the champion's own OOF with the official metric;
- 20 independent draws.

> **Placebo TS-AUC = 0.49904, sd 0.00649, range [0.48512, 0.51420].**

Compare: real dev TS-AUC 0.61500. The bank does **not** detect fake breaks. It is
not measuring elapsed-time-since-an-arbitrary-point, and it is not measuring
sample size. Note this placebo is deliberately hard: the fake labels inherit the
full `n_online` structure of §1 (fake-broken ⟺ short series), so had the bank
carried *any* `n_online` signal the placebo would have printed well above 0.5. It
did not — an independent confirmation of §1.

**τ distribution.** Relative τ mean 0.4872; KS against uniform D = 0.02627,
p = 0.0081 (reproduces the recorded 0.0263). Relative τ is flat across `n_online`
quintiles (0.481 / 0.504 / 0.469 / 0.492 / 0.489), and break rate is flat too
(0.485 / 0.508 / 0.492 / 0.495 / 0.505). AUC(`n_online` → `has_break`) = 0.5066,
AUC(`n_hist` → `has_break`) = 0.4930. No series-level label structure.

---

## 4. My own attacks (line 6)

### 4.1 Independent causality audit at cut lengths the shipped harness never reaches. **CLEAN — but the harness has a gap.**

`check_prefix_invariance` defaults to `cuts=(3, 10, 37)`. Every long-window and
grid-truncated code path is dormant at those lengths: `m00_core` w128/w256, `m06_loc`
`fx_*_384`, and in particular `m06_loc._hist_null`, which takes `n_online` as an
argument and truncates its candidate grid to `K = #{m : mult·m ≤ n_online}`. At a
cut of 3 or 10, `K = 0` and the function returns `None` for both the full and the
truncated build, so NaN == NaN and the test passes *vacuously*. **The one place in
the codebase where `n_online` is read was never actually tested.**

I re-ran the audit myself at **cuts (60, 150, 300, 450)** on 12 dev series with
`n_online ≥ 500`, all 7 modules, `atol = 0.0`:

| module | prefix invariance | max abs diff |
|---|---|---|
| m00_core / m01_seq / m02_dist / m03_dyn / m04_resid / m06_loc / m07_bayes | **PASS (bitwise)** | 0.000e+00 |

The `m06_loc` grid-truncation argument holds: row `t` only consults nodes with
`need ≤ t+1 ≤ n_online`, the null sample is pinned to the full grid constant, and
nothing changes. **Recommendation: raise the harness default cuts** (or add a long
cut set) so this is covered by the standing test rather than by this report.

### 4.2 Cache reproducibility. **CLEAN.**

Recomputed all 7 modules from source for the same 12 series and compared against
`cache/features/<m>.npy` rows: **max |recompute − cache| = 0.000e+00** and zero
NaN-pattern mismatches, every module. The model was trained on features that
current source reproduces exactly.

### 4.3 Cross-fold near-duplicate series. **CLEAN.**

If the generator drew from a small parameter pool, near-duplicate series
straddling folds would inflate OOF and vanish on the organiser's set. 38-dimensional
history fingerprint (25 quantiles, ACF at lags 1/2/3/5/8/13/21, squared-series ACF
at 1/2/3/5, kurtosis, skew), z-scored, nearest neighbour restricted to a
*different* fold:

- cross-fold NN distance: min 0.496, p1 0.777, median 1.390 — no cluster near zero;
- exact head/tail duplicate histories among 8,000 dev series: **0**;
- label agreement with cross-fold NN: **0.4942** overall; 0.5500 (n = 40) in the
  closest 0.5 %, 0.4875 (n = 160) in the closest 2 %, 0.4825 (n = 800) in the
  closest 10 %.

Chance is 0.500. No memorisable duplication.

### 4.4 The exact standardisation (line 2). **REAL, BOUNDED, TRANSFERS.**

Confirmed on 4,000 dev series: history mean is 0 to 2.3e-09 and **population** sd
(`ddof=0`) is 1 to ±2.6e-08; `Σh² / n_hist ∈ [0.99999995, 1.00000005]`. The
generator used `ddof=0`. Therefore the `ddof=1` sample sd is *exactly*
`sqrt(n/(n−1))` — sd computed with `ddof=1` is a deterministic bijection of
`n_hist`, spread 1.0001–1.0005. `n_hist` is already cleared as a label predictor
(0.493 here), so this is inert, but any future feature that ratios an online
`ddof=1` sd against a historical `ddof=1` sd is silently encoding `n_hist`.

**Crucially, the online segment is NOT separately standardised** — I checked this
because it would have been catastrophic (a no-break series' full-online sd pinned
to exactly 1 is a label leak). Online sd: no-break mean 0.982 / median 0.983 / max
3.83; break mean 1.058 / median 0.992 / max 44.5. Pre-break segment sd of break
series: median 0.969. All continuous, all consistent with the documented downward
small-sample bias. **No online standardisation leak.**

Boundary continuity: corr(hist[−1], online[0]) = +0.0425 (no-break, n = 2,027) vs
+0.0759 (break, n = 1,973) — a 1.1σ difference, not a finding, and consistent with
the t = 0 screen showing nothing.

This standardisation is a property of the *problem as delivered*, identical in the
organiser's test set. **Category (a): keep, it is not a transfer risk.**

---

## 5. Line 5 — degenerate / saturated / clock columns

400,000 dev rows from 1,200 dev series. Full table in `/tmp/rt4/degen.parquet`
and `/tmp/rt4/colstats.parquet`.

### 5.1 The NaN structure is a pure clock, and it inflates gain

Many high-gain columns are heavily NaN: `m06_loc::fx_h_rsq_384` 62.0 %,
`m00_core::w256_*` 44.7 %, `m06_loc::fx_p_rsq_256` 76.2 %, `m03_dyn::hw_hurst`
12.3 %. I checked whether NaN-ness is series-dependent (which could encode
something) — **it is not**. At every fixed `t` the NaN fraction across the entire
series cross-section is exactly 0.0 or exactly 1.0, and the pooled NaN fraction
matches `P(t < window)` to three decimals (`P(t<256) = 0.4463` vs w256's 0.4469;
`P(t<384) = 0.6203` vs `fx_*_384`'s 0.6198).

So NaN-ness is a deterministic function of `t`. LightGBM routes NaN down its own
branch, i.e. these columns hand the model an exact `t ≥ w` indicator. That is
**not a leak** (`t` is known at inference) and contributes **exactly zero**
within-timestep discrimination — but it does mean gain importance for the long-window
block is partly payment for clock-splitting.

### 5.2 Pure clocks with large gain

Per-column diagnostics on 700 whole dev series (333,766 rows): `clockR2` = fraction
of a column's variance explained by `t` alone (10-step bins); `within-t AUC` =
the official `ts_auc_flat` applied to the raw column.

| column | gain rank | gain share | clockR2 | within-t AUC |
|---|---|---|---|---|
| `m00_core::t_online` | 2 | 2.87 % | **0.9998** | **0.5000** |
| `m07_bayes::bo_mean_rel` | 6 | 1.81 % | **0.9923** | 0.5172 |
| `m00_core::log_t_online` | 16 | 1.05 % | **0.9938** | **0.5000** |
| `m07_bayes::ab_fast_slow` | 114 | 0.29 % | 0.9121 | 0.4909 |

Four columns with `clockR2 > 0.9` absorb **6.01 % of total champion gain**, and two
of them have *exactly* 0.5000 within-timestep AUC — zero standalone metric value by
construction. More broadly, **65 columns (13 %) have |within-t AUC − 0.5| < 0.005
and carry 9.88 % of gain; 125 columns (25 %) are under 0.01 and carry 19.4 % of
gain.** Spearman(gain, |within-t AUC − 0.5|) = **0.163**.

This is not an artifact — `t`-conditioning is legitimate and transfers — but it is
a live methodological hazard: **gain importance on this problem is only weakly
related to metric value, and RT-140's importance-ranked pruning was ranking on the
wrong quantity.** A top-k selected by within-timestep AUC-adjusted importance is a
different, probably better, top-k. `m07_bayes::bo_mean_rel` at gain rank 6 being
99.2 % clock is the sharpest example.

### 5.3 Degenerate and saturated columns

- **35 columns have gain exactly 0.0** — never split on. All are short-window
  `m00_core::w8_*`/`w16_*`/`w32_*` and `m06_loc::fx_*_16` variants plus
  `m01_seq::{glz,cz50}_slp`, `m01_seq::xc_n_hot`. Dead weight, not artifacts.
- **Near-constant:** `m07_bayes::xb_n_hot` (8 unique values, 49.2 % at 0, gain rank
  299); `m03_dyn::sp_dom` (6 unique values, 12.3 % NaN, gain rank 316);
  `m00_core::xs_min_tail_x` (90.4 % at the value 0, gain rank 448).
- **Saturated at a clip boundary, but high gain** — the `m01_seq` peak-ratio family:
  `sq25_pkr` 80.5 % at 0 (rank 36), `srz_pkr` 72.5 % (rank 26), `sq50_per` 72.8 %
  (rank 96), `cz25_pkr` 70.6 % (rank 20), `ce25_pkr` 68.4 % (rank 28), `phu_pkr`
  57.8 % (rank 13), `phd_pkr` 56.7 % (rank 10), `gle_pkr` 44.5 % (rank 7).
  `m00_core::w8_z_tail_x` is 85.6 % at its minimum. The `_pkr` mass at zero is
  "no peak has occurred yet", which is a legitimate and informative state, and their
  within-t AUCs are genuinely off 0.5 (0.514–0.544). **Not artifacts** — but they are
  fragile: a test set whose transient rate differs shifts a large point mass.

---

## 6. Ranked SUSPICIOUS COLUMNS

Ranked by *how much of their importance is not metric-bearing*, not by "leak" —
because I found no leaking column.

| # | column | gain rank | evidence | class |
|---|---|---|---|---|
| 1 | `m07_bayes::bo_mean_rel` | **6** (1.81 % of gain) | clockR2 **0.9923**; within-t AUC 0.5172; also the most extreme t=4 `has_break` AUC (0.4784). 6th most important feature is 99 % a time index. | clock, transfers, importance grossly overstated |
| 2 | `m00_core::t_online` | **2** (2.87 %) | clockR2 0.9998; within-t AUC **exactly 0.5000** | explicit clock, transfers, zero standalone metric value |
| 3 | `m00_core::log_t_online` | 16 (1.05 %) | clockR2 0.9938; within-t AUC **exactly 0.5000**; monotone in #2 so also redundant with it | clock + redundant |
| 4 | `m07_bayes::ab_fast_slow` | 114 (0.29 %) | clockR2 0.9121; within-t AUC 0.4909 (wrong side) | clock |
| 5 | `m06_loc::fx_*_384`, `fx_p_*_256`, `m00_core::w256_*` (26 cols) | 86–304 | 44.7–76.2 % NaN; NaN-ness is a *pure function of t* (0/1 across the cross-section at every t); gain partly paid for clock-splitting | t-indicator, transfers |
| 6 | `m03_dyn::az_*_exp_l*_z` family (10 cols) | 23–52 (~1.3 M gain) | within-t AUC 0.484–0.510, clockR2 < 0.015 — high gain, no univariate metric signal, no clock explanation. Either pure interaction value or fitted noise. **The most worthwhile ablation target.** | unexplained |
| 7 | `m02_dist::tl_exp_asym01` / `tl_exp_asym05` | 18, 60 (1.39 % combined) | within-t AUC 0.5037 / 0.5050, clockR2 ≈ 0.001 | unexplained |
| 8 | `m01_seq::sq25_pkr`, `srz_pkr`, `sq50_per`, `cz25_pkr` | 20–96 | 70–81 % of rows sit on the 0 boundary | fragile, not artifactual |
| 9 | `m07_bayes::xb_n_hot`, `m03_dyn::sp_dom`, `m00_core::xs_min_tail_x` | 299, 316, 448 | 6–8 unique values / 90 % modal mass | degenerate, negligible gain |
| 10 | 35 columns with gain 0.0 | 466–500 | never split on | dead weight |

**Not one column on this list predicts the label through a synthesis property.**
Every entry is either a clock, a degeneracy, or an unexplained-but-not-suspicious
interaction term.

---

## 7. Transfer sensitivity — what *would* move the number

Not an artifact, but the honest answer to "will 0.615 transfer". Champion OOF,
metric recomputed within strata (sub-cross-sections are thinner, so these are
mix-sensitivities, not comparable to 0.615 directly):

| stratum | TS-AUC | spread |
|---|---|---|
| `n_online` tertile low / mid / high | 0.58167 / 0.60354 / 0.63976 | **0.058** |
| historical-kurtosis quartile q0…q3 | 0.62623 / 0.61685 / 0.58758 / 0.61986 | 0.039 |
| `n_hist` tertile low / mid / high | 0.61082 / 0.60657 / 0.62746 | 0.021 |

If the organiser's test set has a different length or tail-heaviness mix, ±0.02–0.03
is on the table from composition alone — comparable to the recorded 0.0116
dev→lockbox haircut and larger than most of the wins in `RESULTS.csv`. Worth a
composition check against whatever `X_test` metadata is legally observable.

---

## 8. Lines of attack I dropped, and why

- **Placebo splits recomputed in raw feature space** (line 3, literal form). The
  feature modules never receive τ, so "recompute the bank with a fake τ" is not a
  defined operation — the only meaningful placebo is relabelling, which I ran (§3).
  The raw-statistic placebo (does a pre/post split of a no-break series look like a
  break) was already done in `break_taxonomy.md` (0.054σ vs 0.053σ mean shift, AUC
  0.4998; 17.0 % vs 15.6 % transient rate). Re-running it would have burned compute
  to reproduce a recorded negative.
- **Per-column τ-conditional decomposition** (line 1, fine-grained form). Dropped
  after the structural argument in §3: TS-AUC matches total sample size inside every
  timestep, so the decomposition asks a question the metric has already answered,
  and the placebo tests the residual directly at 0.499.
- **Ablation retrains.** Every "does the model actually lean on this" question ends
  in a retrain. At 2 contended cores I could not run the ~6.5 min/fold champion, and
  a 150k-row 200-tree proxy has a fold-noise floor around ±0.005 — wider than most
  effects on the suspicious list. I measured leakage instead of ablating it, which
  is the stronger test anyway.
- **Wave 1 repeats** (series id, store offset, row order, `n_hist`, DGP-cluster
  break rate) — cleared already; I re-ran only `n_hist`/`n_online` because they were
  free by-products.

---

## 9. What I could NOT test given the compute limit

1. **Any retrain-based ablation.** No number in this report says "removing column X
   costs Y TS-AUC". Item 6 on the suspicious list (`m03_dyn::az_*_exp_l*_z`, ~1.3 M
   gain, no univariate metric signal, no clock explanation) is unresolved for exactly
   this reason.
2. **The other six ensemble streams.** Everything here is measured on `RT-100R`'s
   OOF and the shared 500-column bank. `RT-121`/`RT-122`/`RT-123`/`RT-124`/`RT-125`
   were not screened for `n_online` correlation. Given §1, they should be, and it is
   cheap — one AUC per stream per timestep band.
3. **Causality at full coverage.** 12 series × 7 modules × 4 cuts, not 8,000 × 7.
   Bitwise-zero on 12 long series is strong, but a rare data-dependent branch could
   hide in the other 7,988.
4. **Lockbox anything** (spent, off limits) and **test-set composition** (frozen).
5. **The degeneracy screen** ran on 1,200 series / 400k rows and the column
   diagnostics on 700 series / 334k rows — modal fractions and clockR2 carry ~±0.01
   sampling error, and columns that are degenerate only in a rare regime would be missed.

---

## 10. Verdict and next steps

> **HYPOTHESIS REJECTED.** The 500-column bank contains no column that predicts the
> label through a synthesis property. It is blind to `has_break` at t = 0 (500-column
> multivariate OOF AUC **0.4952**), blind to it from history alone (**0.5106**),
> blind to fake breaks (placebo TS-AUC **0.49904 ± 0.00649**), blind to `n_online`
> (**0.488–0.514** across six timestep bands), bitwise causal at cut lengths the
> shipped harness never exercised, and bitwise reproducible from source. There are no
> cross-fold duplicates. **The 0.615 is not an artifact score, and I expected to find
> that it partly was.**

Two real results came out of the exercise anyway, and neither is what I was sent to find:

- **The `n_online` oracle is worth 0.62948 alone and 0.66625 blended** — the largest
  exploitable structure in the dataset, illegal, already the subject of an organiser
  invalidation, and currently guarded only by convention. Make the guard a measured
  gate on every stream.
- **Gain importance is only weakly aligned with metric value** (Spearman 0.163 against
  within-timestep AUC; 6 % of gain in four pure clocks; 19 % of gain in columns with
  no within-timestep signal). RT-140's pruning study ranked on gain and should be
  re-read with that in mind.

**With more compute, in priority order:**

1. `n_online`-correlation gate across all seven streams and any future stream —
   AUC(score → `n_online` > median) at t ∈ {0,20,50,100,200,400}, must stay in
   [0.48, 0.52]. Cost: minutes. Highest value per cycle in this report.
2. Ablate `m03_dyn::az_*_exp_l*_z` (10 columns, ~1.3 M gain, no univariate metric
   signal) at full scale, paired seeds. The only genuinely unexplained mass I found.
3. Re-run RT-140's selection with importance re-weighted by within-timestep AUC
   rather than raw gain; drop `log_t_online` (redundant with `t_online`) and test.
4. Raise `check_prefix_invariance` default cuts to include ≥ 300 so the `m06_loc`
   grid-truncation path is covered by the standing test, and extend the audit to a
   few hundred series.
5. Composition check of dev vs whatever test metadata is legally observable, given
   the 0.058 `n_online`-tertile spread.

---

*Scratch code: `/tmp/rt4/{a_std,b_t0,c_auc,d_multi,e_degen,f_placebo,g_causal,h_quick,i_hist,j_col,k_dup,l_oracle}.py`.
No repository file other than this report was modified. Lockbox untouched.*
