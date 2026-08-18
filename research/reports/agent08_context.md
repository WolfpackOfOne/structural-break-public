# AGENT 8 — HISTORICAL CONTEXT / DGP CLASSIFICATION / MIXTURE-OF-EXPERTS

## HEADLINE

**H1 is REJECTED. H2 is SUPPORTED — but only when context is used as a GATE, not as raw columns.**

| arm | screen TS-AUC (fold 0) | Δ vs control |
|---|---|---|
| `m00_core` control (RT-A08-C) | **0.57384** | — |
| `m05_ctx` ALONE (RT-A08-H1) | **0.50143** | −0.07241 |
| `m00_core + m05_ctx` (RT-A08-H2) | **0.57476** | **+0.00092** |
| `m00_core + m05_ctx_perm` (RT-A08-PERM) | **0.51662** | −0.05723 |
| **PERMUTATION CONTROL: true − permuted** | | **+0.05814** |
| 6 DGP specialists, hard gate, **equal capacity** (6×50 = 300 trees) | **0.60632** | **+0.03248** |
| same, but **permuted cluster assignment** (control) | 0.56054 | −0.01330 |
| **GATING CONTROL: true clusters − permuted clusters** | | **+0.04578** |

The single most important number is the permutation control: **+0.05814**
(true context beats within-fold-deranged context). It is real, but it is *not*
evidence that raw context columns help — see §3, it mostly says that 50
series-constant columns are overfitting fuel when they are mismatched. The
number that actually matters for the leaderboard is the equal-capacity gating
gain: **+0.03248 TS-AUC over the global model, +0.04578 over the
permuted-cluster control.**

---

## §6 RETURN BLOCK

**AGENT NAME** — agent8 (historical context / DGP classification / mixture-of-experts)

**HYPOTHESIS**
- **H1 (level / prior):** the break-free historical segment alone predicts whether
  and when a break occurs, because the generator that produced a given history is
  a priori more or less break-prone. A context feature is constant along a
  trajectory, so it can only act on TS-AUC by shifting a whole series' score level.
- **H2 (interaction):** the historical segment tells the model *how to read* the
  online evidence — a 3σ excursion is unremarkable in a heavy-tailed,
  vol-clustered DGP and decisive in a thin-tailed IID one — so context should
  modulate `m00_core`'s calibrated evidence rather than add to it.

**FALSIFICATION CONDITION**
- H1 falsified if `m05_ctx` alone scores TS-AUC ≤ 0.505 **and** a series-level
  `has_break` classifier on context sits inside the label-permuted null band.
- H2 falsified if `m00_core + m05_ctx` ≤ `m00_core + m05_ctx_perm`, i.e. the gain
  survives destroying the series↔context match, which would make it a pure
  level/prior artifact.
- Gating falsified if per-cluster specialists at **equal total tree budget** do
  not beat the single global model, or do not beat specialists trained on
  randomly permuted cluster labels.

**FILES CHANGED**
- `/home/claude/sb/src/sbr/features/m05_ctx.py` (NEW, owned, 50 columns)
- `/home/claude/sb/research/reports/agent08_context.md` (this file)
- `/home/claude/sb/research/scratch/agent08_perm.py` (own driver: builds `m05_ctx_perm` cache)
- `/home/claude/sb/research/scratch/agent08_series.py` (own driver: series-level H1 + shortcut audit)
- `/home/claude/sb/research/scratch/agent08_gating.py`, `agent08_gating2.py` (own drivers: gating)
- Caches written: `cache/features_screen/m05_ctx.{npy,cols.json}`,
  `cache/features_screen/m05_ctx_perm.{npy,cols.json}` (new module names, nothing overwritten)
- No shared source file was edited.

**EXPERIMENT IDs** — `RT-A08-C`, `RT-A08-H1`, `RT-A08-H2`, `RT-A08-PERM`
(all four appended to `research/RESULTS.csv` via `sbr.pipeline.run`). The gating
arms were run through an own driver because the KMeans gate has to be fit on
TRAIN folds only, which `pipeline.run` cannot express; the driver's global arm
reproduces `RT-A08-C` **bit for bit (0.57384)**, which is the proof it is the
same protocol.

**DATA USED** — `cache/store_screen` (2,500 series, 1,270,042 online rows),
`research/folds/folds_screen.parquet`. Lockbox never loaded.
`X_test.reduced.parquet` never read.

**FOLDS USED** — `folds=(0,)` (screen protocol). Fold 0 = 500 series / 246,205
validation rows; train = folds 1–4 subsampled to 300,000 rows, seed 0.
(I started at `folds=(0,1)` but the box was at load-average 11 on 2 cores; I
dropped to fold 0 so every arm is paired on identical validation series.)

**MODEL + FEATURES** — LightGBM binary, `n_estimators=300`, otherwise pipeline
defaults (lr 0.05, 63 leaves, min_data 200, ff/bf 0.7, l2 5, max_bin 127,
2 threads). `m00_core` = 151 cols, `m05_ctx` = 50 cols.

**TS-AUC (screen)** — table at the top of this report.

**PER-FOLD** — single fold, so per-fold == pooled for every arm. `RT-A08-C`
fold 0 = 0.57384, identical to Agent 0's reference control, so the arms are
directly comparable to the rest of the org's screen numbers.

**DELTA VS CONTROL** — `m05_ctx` as raw columns: **+0.00092** (noise).
`m05_ctx` as a 6-way DGP gate selecting specialist models: **+0.03248** at equal
capacity.

**RUNTIME** — module build on screen store 292 s (`--workers 1`); the four
pipeline runs ≈ 45 min wall under heavy contention (≈ 30 s each unloaded);
gating drivers 1,909 s + 444 s. Total ≈ 3 h wall, almost all of it queueing
behind other agents.

**CAUSALITY CHECK OUTPUT** — pasted verbatim in §1 below. 8 series, `atol=0.0`,
bitwise, ALL PASS.

**LEAKAGE RISKS** — enumerated in §5. Summary: none found. `n_hist` is emitted
and is a declared shortcut suspect but carries no signal (series-level AUC
0.4887 univariate / 0.4948 as a classifier). `n_online` is **not emitted**.

**SIGNAL STORY / FALSE-SIGNAL STORY** — §4.

**OOF ARTIFACT PATH** — none. `pipeline.run` deliberately does not persist OOF
for `screen=True` runs; all four experiments are screen runs. Per-series context
vectors are kept at `research/scratch/agent08_ctx_series.npy` (2500×50) and the
derangement index at `research/scratch/agent08_perm_idx.npy` so any of this is
reproducible without recomputing the module.

**CONCLUSION**
- `m05_ctx` **as a feature block: REJECT** (+0.00092, inside noise, and it costs
  50 columns that the permutation control shows are actively dangerous).
- `m05_ctx` **as a DGP gate: KEEP / PROMISING — strongest single result I
  produced.** +0.03248 TS-AUC at equal capacity with a clean permuted-cluster
  control. Recommend Agent 0 promote the gating architecture (not the raw
  columns) to the full 5-fold protocol.
- **H1: REJECT.** No generator artifact of the "some DGPs are more break-prone"
  kind exists in this data. That is a *good* finding about the challenge.

---

## 1. CAUSALITY CHECK (verbatim)

```
CAUSALITY CHECK: check_prefix_invariance("m05_ctx", ...) atol=0.0, cuts=(3,10,37)
  series    30 n_hist= 1001 n_online= 163 -> True ok
  series  4380 n_hist= 1761 n_online= 566 -> True ok
  series  3056 n_hist= 2540 n_online=  87 -> True ok
  series  4166 n_hist= 3373 n_online= 176 -> True ok
  series  1968 n_hist= 4182 n_online=  70 -> True ok
  series  5813 n_hist= 4996 n_online= 386 -> True ok
  series  2244 n_hist= 3893 n_online=  10 -> True ok
  series   655 n_hist= 3377 n_online= 999 -> True ok
ALL PASS: True
n_cols 50
```

Prefix invariance is trivially satisfied here and that is the point: every column
is a function of `ctx.hist` only, broadcast down the online axis. The permutation
driver independently re-verified constancy within every one of the 2,500 series
(`verified: context is constant within every series`), which is the property the
whole H1/H2 argument rests on.

Columns (50, budget ≤ 50):

```
h_log_sd h_log_sd_over_mad h_skew h_log_kurt h_bowley h_moors h_hill_hi h_hill_lo
h_qr_99_95 h_qr_95_iqr h_entropy h_frac_gt3
h_acf1 h_acf2 h_acf3 h_pacf1 h_pacf2 h_pacf3 h_ar3_log_unexpl h_acf_abs1
h_acf_sq1 h_acf_sq5 h_lb_sq_log
h_vr10 h_vr50 h_adf_nrho h_dfa
h_logband0 h_logband1 h_logband2 h_logband3 h_spec_entropy h_spec_dom
h_spec_centroid h_wav_l1 h_wav_l2 h_wav_slope
h_r100_log_vol_ratio h_r100_mean_z h_r250_log_vol_ratio h_r250_mean_z
h_r500_log_vol_ratio h_r500_mean_z h_r500_kurt_diff h_r500_acf1_diff
h_r250_tail_diff h_r500_log_iqr_ratio h_r500_trend h_r100_max_absz
h_n_hist
```

---

## 2. H1 — DOES HISTORY ALONE PREDICT BREAK OCCURRENCE? **NO.**

**Trajectory level.** `RT-A08-H1`, `m05_ctx` alone: **TS-AUC 0.50143**. Chance.
This is the cleanest possible statement of the geometry in the brief: a
series-constant feature cannot detect anything within a series, and at a fixed
online index it evidently cannot rank series either — because the level it would
have to shift carries no information.

**Series level.** LightGBM on the 2,500 per-series context vectors predicting
`has_break`, with the permanent series-level screen folds:

```
  context 50 cols                            pooledAUC=0.5068  perfold=0.5047 0.5062 0.5028 0.5257 0.4980
  context 49 cols (no n_hist)                pooledAUC=0.5020  perfold=0.4815 0.5072 0.5004 0.5282 0.4990
  n_hist alone                               pooledAUC=0.4948
  id + offset + row_order (pure metadata)    pooledAUC=0.4988
  label-permuted null (3 draws)              AUC=[0.5063, 0.5155, 0.5163]
```

The context classifier (0.5068) sits **below the middle of its own
label-permuted null band** (0.506–0.516). There is no series-level break prior.

**Third, independent confirmation.** Break rate inside each of the 6 KMeans DGP
clusters: `[0.5055, 0.5114, 0.4785, 0.5049, 0.4908, 0.5357]` against a global
0.4960. Flat.

**Verdict on H1: rejected on three independent tests.** No generator artifact of
the form "the simulator only injects breaks into certain DGP families". Stated
plainly because the brief asked for it plainly: **I looked hard for this red flag
and it is not there.** The 2026 data appears to assign breaks independently of
the historical generator, which is the right way to build the benchmark and
means nobody should expect a free lunch from a DGP prior.

---

## 3. H2 — DOES HISTORY HELP *INTERPRET* ONLINE EVIDENCE? **YES — VIA GATING.**

### 3.1 As raw feature columns: no.

`m00_core + m05_ctx` = 0.57476 vs control 0.57384 → **+0.00092**. One fold,
246k validation rows; this is inside noise and I will not claim it.

### 3.2 The permutation control (the headline number)

`m05_ctx_perm` is `m05_ctx` with each series' 50-vector reassigned to a
**different series in the same fold** (derangement, seed 8080, verified no fixed
points and no fold crossing). The fold-conditional marginal distribution of the
block is preserved exactly; only the series↔context match is destroyed.

```
m00_core + m05_ctx        0.57476
m00_core + m05_ctx_perm   0.51662
true − permuted          +0.05814
```

**Read this carefully, because the naive reading is wrong.** The decision rule I
was given says "if true beats permuted, the gain is real". True beats permuted by
+0.05814, so the rule is satisfied — but the honest mechanism is:

- The permuted arm is not a neutral baseline. It is **−0.05723 vs the control**,
  i.e. mismatched context is *catastrophically harmful*, far worse than having no
  context at all. Fifty series-constant, near-continuous columns give each
  training series an almost unique 50-dim signature. LightGBM splits on that
  signature and memorises series-level idiosyncrasies; when the signature is
  random with respect to the online evidence, everything it learned is noise at
  validation time.
- So +0.05814 decomposes as (tiny real interaction, ≈ +0.001) + (large avoided
  damage, ≈ +0.057). The permutation control's real message is a **warning about
  the encoding**, not a licence to ship the block.
- It also independently confirms H1's rejection: a genuine level/prior effect
  operates through the *marginal*, which the derangement preserves, so it would
  have survived permutation and shown up as a gain in the permuted arm. It did
  not. There is no prior to exploit.

### 3.3 Where the interaction actually lives

If context matters by modulating the reading of online evidence, then the right
encoding is a **low-cardinality gate**, not 50 raw columns. Compressing the same
information to a 6-way DGP cluster turns +0.0009 into +0.032:

| arm (fold 0, screen, same 300k train rows, seed 0) | TS-AUC | Δ vs global |
|---|---|---|
| global `m00_core`, 300 trees (reproduces RT-A08-C exactly) | 0.57384 | — |
| global + cluster id + 6 soft memberships, 300 trees | 0.57903 | +0.00519 |
| 6 specialists × 300 trees, **soft** blend | 0.58264 | +0.00880 |
| 6 specialists × 300 trees, **hard** gate | 0.60310 | +0.02926 |
| 0.5·global + 0.5·soft blend | 0.58772 | +0.01388 |
| **6 specialists × 50 trees (EQUAL CAPACITY), hard gate** | **0.60632** | **+0.03248** |
| 6 specialists × 50 trees, soft blend | 0.58394 | +0.01010 |
| PERMUTED clusters × 50 trees, hard | 0.56054 | −0.01330 |
| PERMUTED clusters × 300 trees, hard | 0.56098 | −0.01286 |
| PERMUTED clusters × 300 trees, soft | 0.55697 | −0.01687 |

**Verdict on H2: supported, and the effect is an order of magnitude larger than
anything the raw block delivered.** The same 50 numbers that are worth +0.0009 as
columns are worth +0.032 as a router.

---

## 4. GATING RESULT IN DETAIL

**Setup.** KMeans, k = 6, on robustly standardised (median / IQR-scaled,
clipped ±8, train-median imputed) context vectors, **fit on the 2,000 train-fold
series only**; validation series are only ever *assigned*, never fitted. Cluster
sizes over all 2,500 series: `[182, 352, 395, 204, 1255, 112]`. Soft assignment
is `softmax(−d²/2σ²)` with σ² = median within-cluster distance on train series.

**Equal capacity is genuinely equal.** 6 specialists × 50 rounds = 300 trees,
identical to the global model's 300. The specialists *partition* the same 300,000
training rows (20.8k / 40.3k / 46.1k / 27.6k / 151.5k / 13.7k), so no arm sees
more data either. Result: **0.60632 vs 0.57384, +0.03248.** Capacity is not the
explanation — note the equal-capacity arm actually *beats* the 6×-capacity arm
(0.60632 > 0.60310), i.e. the deep specialists were mildly overfitting.

**It is not ensembling.** Permuting the cluster assignment (same 6 cluster sizes,
random series↔cluster match) collapses the gain to **below** the global model:
0.56054 at equal capacity, 0.56098 at 6× capacity. **True − permuted =
+0.04578.** The gain requires the router to send a series to the specialist
trained on *its own DGP family*.

**Hard beats soft, and that is informative.** 0.60632 (hard) vs 0.58394 (soft).
Soft blending averages in specialists trained on the wrong regime, which dilutes
exactly the regime-specific reading that produces the gain. It also means the
gain is *not* coming from cross-model score-scale differences acting as a
disguised per-cluster prior — cluster break rates are flat (§2), so there is no
prior for a scale offset to encode, and a miscalibration story would predict hard
assignment to *hurt* under a cross-sectional metric, not help by +0.03.

**Interpretation.** This is H2 in its purest form. The optimal mapping from
"m00_core calibrated evidence" to "probability of break" is *different for
different DGP families* — a 2.5σ expanding-window z means something else in a
long-memory, heavily vol-clustered regime than in a near-IID one — and a single
global tree ensemble is forced to average those mappings. Splitting the model on
the DGP recovers the difference. `m00_core` already calibrates the *scale* of
evidence per series (that is what the historical null does); what it cannot do is
change the *functional form* of the evidence→probability map. Gating can.

**Caveats I will not hide.** (i) One fold, 500 validation series; the gating
delta is ~25× the raw-block delta so I believe the sign and rough magnitude, but
the exact value needs the 5-fold protocol. (ii) k = 6 was not tuned; no k-sweep
was run, so 0.606 is not an optimum, it is the first thing I tried. (iii) The
KMeans fit is train-fold-only for fold 0 specifically; a promoted version must
refit the gate inside every fold. (iv) Small clusters (112 series) give thin
specialists; a shrink-to-global blend per cluster is the obvious next move and I
did not have compute to test it.

---

## 5. SHORTCUT AUDIT

Univariate series-level AUC against `has_break`, n = 2,500 (95 % null band
0.5 ± 0.023):

```
META::series_id                0.4920   |d|=0.0080   inside null
META::store_offset             0.4920   |d|=0.0080   inside null
META::row_order                0.4920   |d|=0.0080   inside null
META::n_hist                   0.4887   |d|=0.0113   inside null
h_n_hist  (my column)          0.4887   |d|=0.0113   inside null
META::n_online (FORBIDDEN,
   measured for diagnosis only) 0.5132  |d|=0.0132   inside null
```

Multivariate: `id + store_offset + row_order` → AUC 0.4988 (chance).
`n_hist` alone → 0.4948 (chance).

**Findings.**
1. **No ordering artifact.** Series id, store offset and row order are all
   uninformative. Nothing about how the data was generated or laid out leaks the
   label.
2. **`n_hist` is clean.** I emitted it as instructed and flagged it as a shortcut
   suspect; the audit says it is not one on this data. It is still the column I
   would delete first if `m05_ctx` were ever promoted, because it is metadata
   rather than physics and there is no reason to expect its (absent) relationship
   to be stable on a private set. It also ranks 12th of 50 by univariate |AUC|,
   i.e. it is not doing work.
3. **`n_online` is NOT emitted.** For the record I measured it: univariate AUC
   0.5132, |d| = 0.0132, which is inside the null band and in any case much
   weaker than the "training shortcut" the public 2026 write-ups describe. Either
   way it is unusable — at online step *t* a competitor cannot know how much
   longer the series runs — and it is absent from the module.
4. The largest single univariate context effect anywhere is `h_r500_acf1_diff`
   at AUC 0.4710 (|d| = 0.029), marginally outside the null band and the only
   column that clears it; with 50 columns tested that is expected by chance
   (~2.5 expected at the 5 % level, I got 1–2). This is the "recent-history
   features are unexpectedly useful" claim from public 2026 work, **interrogated
   as instructed: it does not reproduce as a standalone break prior.** The recent
   vs whole family earns its place in this module through the gate (it is a
   strong DGP discriminator), not through any direct relationship to `has_break`.

---

## 6. FEATURE-FAMILY STORIES (protocol §4)

Every column here is blind to the break by construction, so the usual
"false signal" framing inverts: the failure mode is not a spurious detection, it
is a spurious *prior*. I give the intended interaction, the way it can go wrong,
and the disambiguator.

**Moments / tails / entropy** (`h_log_kurt`, `h_hill_*`, `h_qr_*`, `h_entropy`,
`h_frac_gt3`, `h_bowley`, `h_moors`, `h_skew`).
*Signal:* fixes the price of an exceedance. In a Student-t(3) history a 4σ online
point is routine; in a near-Gaussian one it is a break. This is exactly the
quantity `m00_core` cannot supply, because its null calibration fixes the scale
of the evidence but not how a heavy tail should discount it.
*False signal:* `h_log_sd` and `h_log_kurt` are the two most series-identifying
columns in the block and are prime memorisation handles — the permutation control
(§3.2) is what caught this.
*Disambiguator:* the gate. Quantising to 6 clusters keeps the tail regime and
throws away the series fingerprint.

**AR / PACF / vol clustering** (`h_acf*`, `h_pacf*`, `h_ar3_log_unexpl`,
`h_acf_sq*`, `h_lb_sq_log`).
*Signal:* effective sample size. A window of 64 online points in a φ = 0.9 AR(1)
carries maybe 8 independent observations, so the same z-score is much weaker
evidence and the model should demand more before calling a break.
*False signal:* strong persistence also makes ordinary mean-reversion look like a
sustained level shift, so these columns correlate with *false* online alarms as
well as with weak evidence — they point both ways at once and a linear model
cannot use them. Only a gate/interaction can.
*Disambiguator:* `h_vr10` / `h_vr50` separate "persistent but stationary" from
"near-unit-root drift".

**Spectral / wavelet / DFA** (`h_logband*`, `h_spec_*`, `h_wav_*`, `h_dfa`,
`h_adf_nrho`).
*Signal:* low-frequency power and DFA α > 0.5 mean the series wanders on its own.
Wandering is the single biggest confounder of a mean-break detector.
*False signal:* a short history (1,000 points) makes almost anything look
long-memory; `h_dfa` is biased upward at small n, which is precisely where
`h_n_hist` could sneak back in as a proxy.
*Disambiguator:* `h_wav_slope` (energy decay across Haar levels) is far less
n-sensitive than DFA and the two should agree.

**Recent vs whole** (`h_r{100,250,500}_*`, the diff/ratio columns).
*Signal:* the tail of history is the immediate pre-online regime. If it already
differs from the bulk (vol ratio ≠ 1, kurtosis diff ≠ 0, AR diff ≠ 0), then the
null `m00_core` calibrates against is a mixture and the online segment starts
off-centre through nobody's fault — the model should be more sceptical of early
online evidence in that series.
*False signal:* a single outlier in the last 100 points moves
`h_r100_log_vol_ratio` and `h_r100_max_absz` hard; a genuine slow regime change
in late history moves the 500-point versions too.
*Disambiguator:* the 100/250/500 ladder — a real late-history regime shift is
monotone across the ladder, an outlier is not. `h_r100_max_absz` is included
precisely to let the model identify and discount the outlier case.

**Metadata** (`h_n_hist`). No signal story. Emitted because the brief asked for
it, audited in §5, found inert, and recommended for deletion.

---

## 7. RECOMMENDATIONS TO AGENT 0

1. **Do not promote `m05_ctx` as a feature block.** +0.00092 for 50 columns, and
   the permutation control shows those columns are memorisation fuel that costs
   −0.057 the moment the match is broken. On a private set with a different DGP
   mix, that downside is the more likely outcome.
2. **Do promote the gating architecture.** +0.03248 TS-AUC at equal capacity with
   a clean permuted-cluster control (+0.04578 over it) is the largest verified
   effect I found. It is architectural, so it composes with whatever m01–m04
   deliver rather than competing with them.
3. **Suggested next steps** (I ran out of compute): sweep k ∈ {3, 4, 6, 8, 12};
   shrink small-cluster specialists toward the global model; try gating on a
   supervised low-dim projection of context rather than KMeans; re-run gating on
   the best combined feature set rather than `m00_core` alone; then 5-fold.
4. **Reassuring finding for the whole org:** there is no DGP→break prior in this
   data (three independent tests, §2), and no id/offset/ordering/`n_hist`
   shortcut (§5). Nobody should spend more time hunting for one.
