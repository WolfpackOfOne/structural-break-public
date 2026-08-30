# AGENT 12 — BAYESIAN / SEQUENTIAL INFERENCE (`m07_bayes`)

**CONCLUSION: KEEP.** Standalone screen TS-AUC **0.59916** with 50 columns
(m00_core = 0.57384 with 151), within-timestep rank correlation with the
m00_core prediction stream **0.187**, unweighted rank-average ensemble delta
**+0.00971** over the better of the two alone, LightGBM stack delta
**+0.03968** over the m00_core control.

---

## RETURN BLOCK (PROTOCOL §6)

**AGENT NAME** — agent12, Bayesian / sequential inference.

**HYPOTHESIS** — The break process is a latent state NOT-BROKEN → BROKEN that
is *absorbing* (the label `y_t = 1` for all `t >= tau` is literally that state).
Writing that generative model down and reporting posterior/e-value quantities
from it — rather than the shape of a frequentist detector path — produces a
prediction stream with information the moment/occupancy feature bank does not
carry, because the absorbing prior integrates over every changepoint exactly
and the mixture over post-break parameters can be aimed at the families the
forensics says exist (scale, dependence) instead of the one it says does not
(location).

**FALSIFICATION CONDITION** — (pre-registered, in the ledger) standalone screen
TS-AUC ≤ 0.53, and/or screen TS-AUC of `m00_core + m07_bayes` ≤ the `m00_core`
control. Secondary success bar from the mission brief: prediction correlation
with the m00_core stream must be low and the simple-blend ensemble delta
positive. **None of these fired.**

**FILES CHANGED**
- `/home/claude/sb/src/sbr/features/m07_bayes.py` (new, owned by me, 50 columns)
- `/home/claude/sb/scripts/agent12_blend.py` (new)
- `/home/claude/sb/research/reports/agent12_bayesian.md` (this file)
- `/home/claude/sb/research/artifacts/agent12_blend.json`,
  `/home/claude/sb/research/artifacts/agent12_screen_fold0_preds.npy` (new)
- `cache/features_screen/m07_bayes.npy` + `.cols.json` (new cache)
- Nothing under `src/sbr/` outside my own module was touched.

**EXPERIMENT IDs** — `RT-120C` (control, m00_core), `RT-120A` (m07_bayes alone),
`RT-120B` (m00_core + m07_bayes). All three appended to `research/RESULTS.csv`
via `sbr.pipeline.run`, same session, same seed 0, same params.

**DATA USED** — `cache/store_screen` (2,500 series) only. The lockbox
(`fold == -1`) was never loaded. `X_test.reduced.parquet` was never read. No
split was created; `research/folds/folds_screen.parquet` was used as shipped.

**FOLDS USED** — fold 0 as validation, folds 1–4 as training (screen protocol).

**MODEL + FEATURES** — LightGBM binary, `n_estimators=300`, `learning_rate=0.05`,
`num_leaves=63`, `min_data_in_leaf=200`, `feature_fraction=0.7`,
`bagging_fraction=0.7`, `lambda_l2=5.0`, `max_bin=127`, `num_threads=2`,
`max_train_rows=300_000`, `seed=0`. Feature sets as in the table below.

**TS-AUC (screen, fold 0)**

| exp | feature set | n_feat | TS-AUC | Δ vs control |
|---|---|---|---|---|
| RT-120C | `m00_core` (control) | 151 | **0.57384** | — |
| RT-120A | `m07_bayes` alone | 50 | **0.59916** | **+0.02532** |
| RT-120B | `m00_core + m07_bayes` | 201 | **0.61352** | **+0.03968** |

The control reproduces the published screen baseline **0.57384 exactly**
(`0.5738423135475659`), so the comparison is a true paired one.

**PER-FOLD** — screen protocol is single-fold by construction: fold 0 only,
`fold_std = 0`. No multi-fold claim is made here; promotion to the 5-fold
protocol is Agent 0's call.

**DELTA VS CONTROL** — +0.03968 TS-AUC for the stacked model (RT-120B − RT-120C).
The module also beats the control **standalone** by +0.02532 while using a third
as many columns.

**RUNTIME** — module build over the 2,500-series screen store: **639 s**,
`--workers 1`, on a 2-core box carrying five other agents' jobs at load ~8.
Isolated per-series cost of `m07_bayes.build` alone (context already built, as
the driver shares it across modules): **73 ms/series** measured over 80 series
(`m00_core` = 20 ms, `make_ctx` = 168 ms shared). Extrapolated full-store module
cost ≈ **12 min single-core**, inside the ~80 ms/series budget.
Training runtimes: control 701 s, m07 alone 376 s, stack 470 s (contended box).

**CAUSALITY CHECK OUTPUT** — `check_prefix_invariance("m07_bayes", …)`, `atol=0.0`
(bitwise), cuts `(1, 2, 3, 10, 37, 79, 150, 257, 401, 600)`, on 12 series chosen
to span the online-length and history-length ranges:

```
series   559  n_hist= 3893 n_online=  10  -> True ok
series  1514  n_hist= 2268 n_online= 144  -> True ok
series   259  n_hist= 3565 n_online= 264  -> True ok
series  2390  n_hist= 4434 n_online= 511  -> True ok
series  1949  n_hist= 3532 n_online= 755  -> True ok
series  1926  n_hist= 1996 n_online= 999  -> True ok
series     5  n_hist= 1001 n_online= 163  -> True ok
series  1444  n_hist= 4996 n_online= 386  -> True ok
series  1043  n_hist= 2953 n_online= 390  -> True ok
series  1965  n_hist= 4038 n_online= 343  -> True ok
series   737  n_hist= 2039 n_online= 180  -> True ok
series  1801  n_hist= 1948 n_online= 669  -> True ok
ALL PASS
```

The cut list deliberately includes 257 and 401, which straddle the BOCPD
run-length truncation (`R_MAX = 80`), the calibration position grid
(max 256) and the e-process window grid (max 486), so the truncation/fold logic
is exercised on both sides.

**DETERMINISM** — no RNG is used anywhere in the module. Two independent
processes (the ledger runs in `/tmp/run12.py` and `scripts/agent12_blend.py`)
trained the same models from the same cache and reproduced
`0.5738423135475659` and `0.5991560270675718` to the last digit.

**LEAKAGE RISKS** (stated honestly)
1. *Length leakage* — the single most dangerous failure mode for a recursion is
   an initialisation that depends on `n_online` (e.g. hazard `= 1/n`, or a null
   normalised by the online length). Every constant here — hazard rates
   `1/250` and `1/40`, `R_MAX`, the position/window grids, the NIG prior — is a
   literal or a function of `hist` only. The bitwise prefix check at 10 cut
   points is what proves it, and it is the check that would have caught it.
2. *Null construction* — all three nulls (`_PosNull`, `_AddNull`, `_MargNull`)
   are built exclusively from the historical segment, which the competition
   guarantees break-free. Online values only ever enter as the query.
3. *AR coefficients / residual ECDF / normal scores* — fitted on `hist` only;
   the online filter is warm-started from the tail of `hist`
   (`ar_filter_causal`), never from online points at or after `t`.
4. *t-dependent columns* — several columns (`bo_mean_rel`, `bf0_rate`,
   `ab_lpo_slp`) are explicit functions of elapsed online time. TS-AUC compares
   series only *within* a fixed online index, so a quantity that is constant
   across series at fixed `t` cannot move the ranking on its own; it can only
   act as a conditioning variable, exactly as `m00_core::t_online` does.
5. *Residual risk I cannot rule out on a screen* — a 50-column block that beats
   a 151-column block standalone is unusual enough that Agent 0 should re-run
   `RT-120A` on the full 5-fold protocol before believing the magnitude. The
   sign is safe; the size is one fold.

**OOF ARTIFACT PATH** — `sbr.pipeline.run` does not persist OOF under
`screen=True`, so the screen fold-0 validation predictions are saved by my own
script instead:
`/home/claude/sb/research/artifacts/agent12_screen_fold0_preds.npy`
(shape `(246205, 2)`, columns = m00_core-alone, m07_bayes-alone, rows in
`Data(screen=True).rows_for([0])` order) and the blend metrics are in
`/home/claude/sb/research/artifacts/agent12_blend.json`. No `research/oof/`
artifact exists because no full-protocol run was made.

---

## PRIMARY DELIVERABLE — CORRELATION AND ENSEMBLE DELTA

Both models are trained by `scripts/agent12_blend.py` on exactly the screening
code path (same `Data(screen=True)`, same `rows_for`, same seed-0 subsample,
same params), so the predictions correspond one-for-one to `RT-120C` and
`RT-120A`.

### Prediction correlation, m07_bayes-alone vs m00_core-alone

| measure | value |
|---|---|
| Pearson, raw probabilities | **0.5589** |
| Pearson, logits | **0.6451** |
| Spearman, global ranks | **0.6083** |
| **Pearson of within-timestep ranks** | **0.1871** |

The last row is the number that matters. TS-AUC never compares two rows at
different online indices; it compares series against each other *inside* a fixed
online index `t`. The global correlation is inflated by the shared, strongly
t-dependent component of both scores (both models learn "later online rows are
more likely post-break", worth a lot of global rank agreement and exactly zero
TS-AUC). Once that shared component is removed by ranking within each timestep,
the two streams agree at **ρ = 0.19** — they are close to orthogonal in the
space the metric actually scores. That is the decorrelation the mission asked
for, and it is why the blend gains as much as it does.

### Ensemble delta (better-alone = m07_bayes at 0.59916)

| blend | TS-AUC | Δ vs better alone | Δ vs m00_core |
|---|---|---|---|
| m00_core alone | 0.57384 | −0.02532 | 0 |
| m07_bayes alone | 0.59916 | 0 | +0.02532 |
| unweighted **rank average** (no fitting) | **0.60887** | **+0.00971** | +0.03503 |
| unweighted **within-t rank average** | 0.60537 | +0.00621 | +0.03153 |
| unweighted **logit average** | **0.61222** | **+0.01306** | +0.03838 |
| **LightGBM on both blocks (RT-120B)** | **0.61352** | **+0.01437** | **+0.03968** |

Weight sweep on the global rank blend (`w` = weight on m07_bayes), reported as a
curve rather than a selected value, since selecting `w` on the same fold would
be in-sample:

| w | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TS-AUC | .5738 | .5838 | .5927 | .6000 | .6056 | .6089 | **.6095** | .6084 | .6061 | .6030 | .5992 |

The curve is flat and single-peaked between `w = 0.4` and `w = 0.8` — every
weight in that band beats both components, so the gain is not a knife-edge
artifact of a tuned weight. The unweighted `w = 0.5` point is within 0.0006 of
the maximum.

**Verdict against the mission's own bar:** the bar was "0.55 standalone with a
+0.005 ensemble delta is a success". This is 0.599 standalone with a +0.0097
unweighted-rank-average delta (+0.0144 when the trees do the blending), on a
within-timestep prediction correlation of 0.19. It clears the bar on both axes
at once.

---

## WHAT IS IN THE MODULE

Primary observation stream (this is load-bearing, see the false-signal section):

```
raw → standardise by historical (mu, sd)
    → AR(6) whitening with historical coefficients (causal, warm-started from
      the tail of hist)                       [forensics: AR(6) > AR(2) for scale]
    → NORMAL SCORES against the historical residual ECDF
```

After the last step the null distribution of the stream is **exactly N(0,1) per
series by construction**, which is what makes a log Bayes factor comparable
across series with different volatility and tail weight — the whole game under
TS-AUC. It also removes, by construction, the generator artifact the forensics
report flags as item 7 (the raw variance ratio is biased by tail heaviness,
because the transform is rank-based and therefore blind to it). A second,
deliberately *un*-Gaussianised channel (`abz_*`) runs the same recursion on the
raw standardised series so genuine tail information is not thrown away.

### 1. Absorbing two-state posterior (`ab_*`, `abz_*`, 21 columns)

Because the state is absorbing, the exact filter for
`P(broken at t | x_1:t)` needs one log-space accumulator per mixture component:

```
L_t(θ) = logaddexp(L_{t-1}(θ), log h) + llr_t(θ) − log(1−h),   L_{−1}(θ) = −∞
LPO_t  = logsumexp_θ ( log w_θ + L_t(θ) )
```

with `L_t(θ) = log[ P(broken, params θ, x_1:t) / P(not broken, x_1:t) ]`. This
integrates over **every** changepoint exactly, in O(1) per step per component —
no dyadic candidate grid and no O(t) scan, which is the structural difference
from m01_seq's GLR. Because each θ evolves independently, one kernel pass gives
the combined posterior odds, the per-family Bayes factors, *and* the posterior
over θ given broken, which is emitted as "what kind of break is this"
(`ab_post_lvr` = posterior mean log variance ratio, `ab_post_rho` = posterior
mean dependence change, `ab_pfam_var`/`ab_pfam_dep` = family probabilities).

Mixture, aimed by the forensics: variance ratio grid 0.5×…2.0× (weight 0.45),
AR(1)-coefficient change ±0.2/±0.4 (0.20), ARCH-type conditional-variance
dependence (0.15), mean shift ±0.2/±0.4/±0.8 (0.20 — present, deliberately not
dominant, since the forensics puts location-break AUC at 0.4998). Two hazards
are run (`1/250` ≈ 1 / mean online length, and a fast `1/40`); their difference
`ab_fast_slow` is a pure recency channel.

- **SIGNAL** — after a real break the post-break likelihood ratio drifts upward
  at a constant rate, so `LPO` grows ~linearly in elapsed-since-break and never
  comes back. The absorbing prior has exactly the shape of the target.
- **FALSE SIGNAL** — the posterior is a running integral, so it cannot decay: an
  isolated outlier or a transient vol burst pushes it up permanently.
- **DISAMBIGUATOR** — `ab_lpo_rel` (current − running max, normalised) and
  `ab_lpo_slp` (16-step slope in units of the null spread). A transient leaves
  `rel < 0` and slope ≈ 0; a break keeps `rel ≈ 0` and the slope positive.
  `ab_loc_z` is deliberately retained as an internal control: the location
  family should be inert, and a model that leans on it is fitting noise.

### 2. BOCPD run-length posterior (`bo_*`, 10 columns)

Adams–MacKay with a Normal-Inverse-Gamma observation model **whose prior is the
historical segment** (`κ0 = α0 = 25`, `β0 = 24·var_hist`, `μ0 = mean_hist`),
hazard `1/250`, run-length grid truncated at `R = 80` with the tail folded into
the top bucket (so the longest-run hypothesis becomes a trailing-window one).
Bucket `r = t+1` is exactly the "no change since the online segment started"
hypothesis. Emitted: `P(r<10)`, `P(r<25)`, posterior mean run length relative to
`t+1`, entropy, no-change mass, log-odds(changed vs never changed) and its
running peak, plus historical-null-calibrated versions of three of them.

- **SIGNAL** — after a break the run-length posterior collapses onto short runs
  and the never-changed mass decays to zero.
- **FALSE SIGNAL** — a single outlier also resets the run-length posterior for a
  few steps.
- **DISAMBIGUATOR** — BOCPD *recovers*: it is not absorbing, so after a transient
  the posterior re-concentrates on long runs. The **disagreement** between the
  absorbing posterior (stays high) and BOCPD (comes back) is the transient
  signature, and both are emitted so the trees can read it.

### 3. E-values / test martingales (`ev_*`, 12 columns)

Betting capital processes against the per-series historical null, in log space
so they cannot overflow, on four **bounded** payoff functions with the null mean
`m0` taken from history: tail occupancy `1{|u−½|>0.4}`, PIT dispersion
`(u−½)²` (the scale channel), sign-agreement of consecutive centred PITs (the
dependence channel), and joint tail occupancy of consecutive points (the
volatility-clustering channel). Each is emitted as a mixture over a fixed grid
of betting fractions (no tuning) and, for the two headline payoffs, with a
predictable plug-in (GRAPA-style) fraction whose value at `t` uses only
`h[:t]`. Plus Vovk's simple-mixture **power martingale** on two-sided conformal
p-values (both `p` and `1−p`, so a variance *decrease* is also detected).

- **SIGNAL** — capital compounds only while the alternative keeps paying, and by
  Ville's inequality a large value is evidence at *any* stopping time. That is
  the correct object for a stream scored at every index without a
  multiple-testing correction — a genuinely different construction from both the
  likelihood-ratio bank and the frequentist detector bank.
- **FALSE SIGNAL** — a run of moderately unusual points that reverts.
- **DISAMBIGUATOR** — the payoffs are bounded, so one enormous observation can
  win at most one bet, where a Gaussian likelihood ratio would score it as
  overwhelming evidence. `ev_*` high with `ab_var` low ⇒ sustained scale change;
  `ab_var` high with `ev_*` low ⇒ a fat-tail artifact. That contrast is the
  intended use.

### 4. Sequential Bayes factors from `t = 0` (`bf0_*`, 5 columns)

The mixture likelihood ratio accumulated from the first online point (the
"break at the start" alternative), overall and for the variance family alone,
plus the per-observation evidence rate `bf0/(t+1)` which is directly comparable
across online indices.

### 5. Calibration — every quantity raw *and* against the per-series null

- **`_AddNull` (exact, length-matched).** The additive channels (sequential BF,
  mixture e-processes) have the property that the statistic over a length-`L`
  window is a window sum of the same increments — so *every* length-`L`
  historical window is one exact draw from the null. Online step `t` is scored
  against the historical window length nearest `t+1` in log space.
- **`_PosNull` (length-matched, restart blocks).** The absorbing posterior is
  path-dependent, so its null is built by **restarting the identical recursion**
  at a dense grid of historical offsets (stride 32) and recording the path at a
  log-spaced grid of elapsed positions (1…256). This matters: `LPO` drifts with
  elapsed time under the null, so scoring a 20-step-old online path against a
  converged historical path would make every early row look artificially quiet.
- **`_MargNull` (marginal).** BOCPD summaries are already probabilities and
  roughly comparable; they get the marginal distribution of the same summaries
  over a historical BOCPD pass.

Raw values are emitted alongside every calibrated one, as instructed, so the
trees can use the per-series scale where it happens to be informative.

---

## DOCUMENTED APPROXIMATIONS (all deliberate, all cost-driven)

1. **BOCPD run-length truncation at `R = 80`.** Mass that would reach `r = R` is
   folded into `r = R−1`, whose sufficient statistics are a trailing window of
   `R−1` points rather than the full run. Effect: for `t > 80` the "no change"
   hypothesis has an 80-point memory. Deterministic and strictly forward, so
   prefix invariance is unaffected.
2. **Position grid for `_PosNull` caps at 256.** Online rows with `t > 256`
   (≈ 20 % of rows) are calibrated against the 256-step null. `LPO` drifts like
   `log t` under the null, so the residual bias is a smooth shared function of
   `t` and cannot re-rank the cross-section at fixed `t`.
3. **Coarse window grid `(6, 18, 54, 162, 486)` for `_AddNull`** — factor-3
   spacing, so the worst-case length mismatch is √3 ≈ 1.7×. Chosen purely to fit
   the per-series budget; a denser grid cost ~4× and I did not measure a benefit.
4. **Null sample cap 500 draws per window** — enough for a robust z (median/IQR),
   marginal for an exact percentile; only `ab_lpo_sur` uses a percentile and it
   is derived from the ~100-block `_PosNull`.
5. **The e-process null mean `m0` is estimated on `hist`** rather than known.
   With `n_hist ≥ 1000` the estimation error is ≪ the alternative it tests, but
   it does mean these are e-processes against an *estimated* null, so their
   anytime-valid guarantee is approximate. They are used as features, not as
   tests, so this is a statement about interpretation, not validity.

---

## WHAT I WOULD DO NEXT (not done, out of budget)

1. Re-run `RT-120A` / `RT-120B` under the full 5-fold protocol. A 50-column
   block beating a 151-column block standalone deserves a second fold before the
   magnitude is believed. The *sign* is safe; the size is one fold.
2. Ablate the four blocks (`ab_*`, `bo_*`, `ev_*`, `bf0_*`) against each other.
   I have no per-column importance for the screen run (the pipeline does not
   persist it under `screen=True`), so I cannot say which block carries the
   0.599, and I have deliberately not guessed in this report.
3. Test whether the location family (`MU_GRID`, 20 % of the prior mass) earns
   its place. The forensics says location breaks do not exist; if `ab_loc_z` is
   inert the mass should be moved to variance/dependence, which is free.
4. The AR(6)+normal-scores stream is likely useful to other modules
   independently of the Bayesian machinery; if the ablation shows the stream
   rather than the recursions is doing the work, that is an important and
   cheaper finding for everyone.
