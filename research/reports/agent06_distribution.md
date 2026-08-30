# Agent 6 — Distributional / PIT / Rank Detection (`m02_dist`)

## Return block (PROTOCOL §6)

**AGENT NAME** — agent6 (distributional / PIT / rank detection)

**HYPOTHESIS** — Under stability the historical-ECDF PIT `u_t = F_hist(x_t)` of the
online segment is Uniform(0,1). A structural break destroys that uniformity in ways
that mean/variance/moment monitoring (`m00_core`) cannot see: shape changes that
preserve the first two moments, tail-weight changes, quantile-local regime shifts,
rank-structure drift. Trailing/expanding **occupancy of the historical quantile
bins**, turned into the classical two-sample divergences (chi-square, Jensen-Shannon,
Hellinger, TV, Cramér–von Mises, KS, 1-D Wasserstein, energy distance) and calibrated
against a **per-series historical null of the same statistic at the same window
length**, adds TS-AUC on top of `m00_core`.

**FALSIFICATION CONDITION** — screen TS-AUC of `m00_core + m02_dist` ≤ the paired
`m00_core` control (0.57384) on fold 0 with identical seed/settings.

**FILES CHANGED**
- `/home/claude/sb/src/sbr/features/m02_dist.py` (new, 59 columns, owner agent6)
- `/home/claude/sb/research/reports/agent06_distribution.md` (this file)
- `/home/claude/sb/research/FAILED_EXPERIMENTS.md` (appended)
- feature caches: `/home/claude/sb/cache/features_screen/m02_dist.{npy,cols.json}`
- ledger rows appended via `sbr.pipeline.run` only.

**EXPERIMENT IDs** — RT-060C (control), RT-061 (treatment), RT-062 (module alone),
RT-063-divall / RT-063-tl / RT-063-rk (within-module ablations).

**DATA USED** — `cache/store_screen` (2,500 series) only. Lockbox never loaded.
`X_test.reduced.parquet` never read.

**FOLDS USED** — fold 0 as validation, folds 1–4 as train (screen protocol),
`max_train_rows=300_000`, `seed=0`, `params={"n_estimators":300}` for every run.

**MODEL + FEATURES** — LightGBM binary, pipeline defaults except `n_estimators=300`.
Control 151 features (`m00_core`); treatment 210 (`m00_core` + 59 `m02_dist`).

### TS-AUC (screen, fold 0)

| exp | modules / slice | n_feat | TS-AUC | Δ vs control |
|---|---|---|---|---|
| RT-060C | `m00_core` (control) | 151 | **0.57384** | — |
| RT-061 | `m00_core` + `m02_dist` | 210 | **0.58698** | **+0.01313** |
| RT-062 | `m02_dist` alone | 59 | 0.54674 | −0.02710 |
| RT-063-divall | `m00_core` + `m02_dist::(dv,oc,qd,ab,rs)_` | 198 | 0.58555 | +0.01171 |
| RT-063-tl | `m00_core` + `m02_dist::tl_` | 157 | 0.56974 | −0.00410 |
| RT-063-rk | `m00_core` + `m02_dist::rk_` | 157 | 0.57111 | −0.00273 |

**PER-FOLD** — single screening fold, so per-fold == pooled == the numbers above;
`fold_std = 0` by construction. No 5-fold claim is made here.

**DELTA VS CONTROL** — **+0.01313 TS-AUC**. The control was re-run by me in this
session with the same seed and settings and reproduced Agent 0's number to all
printed digits (0.5738423135475659), so the pairing is exact.

**RUNTIME**
- module build cost: **~39 ms/series marginal** (measured over 120 screen series,
  excluding the shared `make_ctx`, which is ~162 ms and amortised across all modules).
  Inside the ~60 ms/series budget.
- full screen-store build (`--workers 1`, 2 cores shared with 5 other agents):
  **752 s** for 2,500 series → `m02_dist.npy` (1,270,042 × 59).
- each screening run: 8–15 min wall under heavy contention (load average 6–9).

**CAUSALITY CHECK OUTPUT** — `check_prefix_invariance("m02_dist", …, atol=0.0)`,
cuts `(1,2,3,5,10,37,63,129,257)`, 10 series spanning the online-length range:

```
series   559 n_hist= 3893 n_online=  10 -> True ok
series  1807 n_hist= 1823 n_online=  53 -> True ok
series  2165 n_hist= 2391 n_online= 167 -> True ok
series  1484 n_hist= 4873 n_online= 308 -> True ok
series  1574 n_hist= 4184 n_online= 458 -> True ok
series   770 n_hist= 1875 n_online= 614 -> True ok
series   731 n_hist= 3016 n_online= 754 -> True ok
series   326 n_hist= 2309 n_online= 901 -> True ok
series  1462 n_hist= 1900 n_online= 990 -> True ok
series  1926 n_hist= 1996 n_online= 999 -> True ok
```

Bitwise identical on every truncation. (10 series, ≥ the 8 required.)

**LEAKAGE RISKS** (audited, all negative)
- Every reference object — bin edges (implicit in the PIT), the `|x−med|` reference
  ECDF, the AR-residual reference ECDF, every null distribution, every CUSUM null —
  is built from `ctx.hist` / `ctx.hist_tr` **only**, once per series, before any
  online point is touched.
- All trailing statistics are cumsum differences over `online[:t+1]`; all expanding
  statistics are prefix sums; the rank CUSUM uses `np.maximum.accumulate`, which is a
  running (causal) max.
- The expanding-window nulls are selected by bucketing `L = t+1` to the nearest null
  length in log space. That mapping is a function of `t` alone, identical for every
  series, so it cannot smuggle in online information and cannot change a
  cross-sectional ranking at fixed `t`.
- Null subsampling uses `np.linspace` (deterministic), never an RNG, so nothing
  depends on call order.
- No column uses any online quantity computed across rows (no global mean, no
  full-sample quantile, no normalisation by the online segment).
- Residual PIT drops the first `len(ar_coef)` historical residuals, which
  `build_transforms` pads with exact zeros — otherwise a spurious atom at 0 would sit
  in the reference ECDF. History-only fix, no causality implication.

**OOF ARTIFACT PATH** — none. `sbr.pipeline.run(screen=True)` deliberately does not
persist OOF arrays or importances; all six runs are screening runs. Ledger rows carry
the numbers.

**CONCLUSION: KEEP** — `m02_dist` as a whole, at 59 columns and ~39 ms/series, buys
+0.0131 screen TS-AUC over the strongest existing module. **Recommended trim if
column budget is tight: drop the `tl_` and `rk_` families** (12 of 59 columns); they
are individually negative on top of `m00_core` and the remaining 47 columns retain
+0.0117 of the +0.0131.

---

## Design: what is actually computed

One object underlies almost everything: the occupancy vector of the online PIT over
`B` equal-probability historical bins. Under stability the reference is exactly
uniform-over-bins *by construction of the PIT*, so no reference histogram has to be
estimated or stored, and every divergence below is a closed form in the occupancy
vector. Rolling occupancy is `B` cumsum differences; nothing is re-sorted per step.

Streams (each an independent PIT against a history-only reference ECDF):
- `u` — raw PIT `F_hist(x)`, at `B = 20` (`dv_`, `qd_`) and `B = 10` (`oc_`).
- `ab` — PIT of `|x − med_hist|` against the historical ECDF of `|h − med_hist|`
  (a pure scale/shape channel that is blind to sign).
- `rs` — PIT of the causal AR(2) residual `ctx.tr["res_mean"]` against the historical
  residual ECDF built here from `ctx.hist_tr["res_mean"]` (innovation-law channel;
  the AR filter removes the autocorrelation that would otherwise make a level shift
  masquerade as a shape change).

Windows: trailing 32 and 128 (plus 64 for the tail family) and expanding.

Calibration: for every statistic, the *same* statistic is evaluated over contiguous
historical windows of the matching length (subsampled to ≤ 500 start positions by
`np.linspace`, which costs nothing given we floor p at 1/2m), sorted once, and the
online value converted by `searchsorted` into a one-sided `−log10 p` (for
"bigger = more different") or a signed two-sided surprise (for directional
statistics). Expanding windows are calibrated against the nearest of three null
lengths {16, 64, 256} in log space. `ctx.nc` is not used because none of these
statistics is the rolling mean of a registered transform — with the single exception
of the tail-rate family, whose nulls are built with the same cumsum machinery.

Closed forms used (`q` = occupancy, `F` = its CDF, reference uniform):
chi-square `Σ(C−w/B)²/(w/B)`; JS `½KL(q‖m)+½KL(u‖m)`; Hellinger `1−Σ√q/√B`;
TV `½Σ|q−1/B|`; KS `√w·max|F−k/B|`; CvM `(w/B)Σ(F−k/B)²`; W₁ `(1/B)Σ|F−k/B|`;
energy `2·mean(u²−u+½) − 2∫F(1−F) − ⅓` (the `E|U−U′|` term comes straight from the
binned CDF, the `E|U−V|` term from exact rolling means of `u` and `u²`). Quantile
deviations are read off the binned CDF by linear inversion at levels
0.1/0.25/0.5/0.75/0.9; in PIT space the historical quantile at level `p` *is* `p`,
so the deviation is already scaled by the historical spread — no explicit IQR
division is needed, and it is the correct scale-free version of the requested
"minus historical quantile, divided by historical IQR".

---

## Per-family stories

### `dv_` — divergences on the raw PIT (16 cols): chi², JS, KS, CvM, W₁, energy at w=32, w=128, expanding
- **SIGNAL** — any change in the marginal law moves occupancy away from uniform.
  These six differ in *where* they put their weight: chi² and JS are sensitive to
  low-occupancy bins being over-filled (new tail mass, a new mode); Hellinger/TV are
  bounded and robust; KS and CvM are CDF-shape statistics that respond to a coherent
  shift of probability mass to one side; W₁ and energy weight *how far* the mass
  moved, so a shift from bin 10 to bin 20 registers more than a shift to bin 11.
  Emitting the family rather than one member lets the tree learn which geometry of
  departure is diagnostic.
- **FALSE SIGNAL** — (a) a single outlier adds one count to one extreme bin; at w=32
  that is a +1 deviation on an expectation of 1.6, which is a large chi² and a large
  max-deviation. (b) A transient volatility burst symmetrically inflates both extreme
  bins and depletes the centre; chi², JS and Hellinger cannot tell that from a
  permanent variance break. (c) Serially dependent series (persistent volatility)
  produce genuinely large short-window divergences with no break at all.
- **DISAMBIGUATOR** — (a) is separated by scale: an outlier moves `dv_u_w32_*` and
  leaves `dv_u_w128_*` and `dv_u_exp_*` alone, and it moves chi²/max-dev far more
  than W₁/energy (one point cannot move much *mass*). (b) is separated by
  `tl_*_asym05` (a burst is symmetric, most real breaks are not) and by the
  expanding-window persistence. (c) is exactly what the per-series historical-null
  calibration removes: a series whose history already produces JS = 0.05 over 32-point
  windows has a null centred there, and the surprise stays near zero.

### `oc_` — occupancy shape (10 cols): entropy, max standardised bin deviation, argmax bin, Hellinger, TV
- **SIGNAL** — entropy of the occupancy is maximal (log B) under stability. Any
  concentration — a variance *collapse*, a censoring/clipping regime, a series that
  starts sticking to one level — drops it, and drops it in a way that a divergence
  statistic partly confounds with tail inflation. `argmax` bin says *which* part of
  the historical distribution is over-occupied, which is the closest thing here to a
  break-type label the model can condition on.
- **FALSE SIGNAL** — entropy is also depressed by short windows purely from sampling
  noise (32 points into 10 bins is sparse), and `argmax` is nearly uniform noise on
  short windows.
- **DISAMBIGUATOR** — the null is length-matched, so sampling-noise depression is
  exactly what the null encodes and the surprise is ~0. `argmax` is only emitted at
  w=128 and expanding, never at w=32 (that column was dropped in design for this
  reason).

### `qd_` — PIT-space quantile deviations (11 cols): Q50−½, (Q75−Q25)−½, (Q90−Q10)−0.8
- **SIGNAL** — this is the interpretable decomposition of the divergence: a pure
  location break moves `q50` and leaves the spreads; a pure variance break moves
  `qiqr` and `q9010` together; a tail-only break (fatter tails, same body) moves
  `q9010` and leaves `qiqr`. That last one is precisely the break `m00_core` is worst
  at, because it barely moves the mean or the standardised second moment.
- **FALSE SIGNAL** — a temporary level excursion moves `q50` at w=128 identically to
  a level break; heavy-tailed series make `q9010` noisy.
- **DISAMBIGUATOR** — the raw and calibrated versions are both emitted, and both the
  w=128 and expanding versions: a temporary excursion moves w=128 and not expanding,
  a break moves both and keeps moving them. Heavy-tail noise is absorbed by the
  per-series null.

### `ab_` / `rs_` — alternative representations (10 cols)
- **SIGNAL** — `ab_` (PIT of `|x − med|`) is a sign-blind scale/shape channel: a
  symmetric variance break shows up there as a *location* shift of a uniform, which
  is the single most powerful thing a rank statistic can detect, instead of as a
  U-shaped occupancy departure which is a weaker alternative. `rs_` (PIT of the AR(2)
  residual) isolates the innovation law: a change in innovation kurtosis or in the
  noise distribution with unchanged AR structure moves `rs_` and almost nothing else.
- **FALSE SIGNAL** — `ab_` fires on any variance change including transient bursts.
  `rs_` fires when the AR coefficients change (the residuals then stop being white),
  which is a dependence break, not a marginal-law break — worth detecting, but it
  means `rs_` is not a clean "innovation distribution" reading.
- **DISAMBIGUATOR** — `ab_` vs `dv_u_*` separates symmetric scale change from
  asymmetric/shape change; `rs_*` vs `dv_u_*` separates a dependence break (residual
  channel fires, raw channel quiet — an AR change barely moves the marginal) from a
  marginal break (both fire). `m01_seq`/`m03_dyn`/`m04_resid` own the dependence
  channel proper; `rs_` here is only the distributional read of it.

### `tl_` — tail occupancy & asymmetry (6 cols)  → **negative in isolation**
- **SIGNAL** — asymmetric tail inflation (`P(u>0.95) − P(u<0.05)`, same at 0.99/0.01)
  is the signature of a one-sided regime shift; centre depletion `P(0.25<u<0.75)` is
  the signature of a variance break.
- **FALSE SIGNAL** — a *symmetric* vol burst leaves the asymmetry at zero while
  hammering both tail rates; a single outlier moves the 0.01/0.99 asymmetry by a full
  1/w. Tail rates are the sparsest statistics here and therefore the noisiest.
- **DISAMBIGUATOR** — asymmetry vs total tail rate (`m00_core::*tail_x*`) separates
  one-sided from symmetric; the w=64 vs expanding contrast separates transient from
  persistent.
- **VERDICT** — measured Δ = **−0.00410** on top of `m00_core`. `m00_core` already
  monitors `tail_hi`, `tail_lo`, `tail_x` and `center` at every window with the same
  null engine, so the only genuinely new content is the *asymmetry contrast*, and 6
  sparse noisy columns cost more in fit variance than that contrast is worth. Drop.

### `rk_` — rank location/scale/CUSUM (6 cols)  → **negative in isolation**
- **SIGNAL** — rolling variance of `u` is exactly 1/12 under stability regardless of
  the series' tails, so it is a completely tail-free scale monitor. The rank CUSUM
  `max_{k≤t}|Σ(u_i−½)| / √(t/12)` is the classical Kolmogorov/Darling–Erdős
  distribution-free change-point statistic: it detects a rank-location shift *and
  localises it*, and its running max means late-window evidence never erases early
  evidence.
- **FALSE SIGNAL** — the CUSUM's running max is monotone-ish in `t`, so it drifts up
  even with no break; it is also inflated by any positive serial dependence in ranks
  (which a persistent-vol series has). Rank variance is depressed by a *narrowing*
  distribution and by short-window sampling noise alike.
- **DISAMBIGUATOR** — the length-matched historical CUSUM null (computed over
  historical windows of the same length with the same running-max definition) removes
  both the deterministic drift and the series' own dependence level; `rk_cusum_r`
  (residual stream) versus `rk_cusum_u` (raw stream) separates a genuine rank-location
  break from one induced by autocorrelation.
- **VERDICT** — measured Δ = **−0.00273**. The raw CUSUM columns are the *strongest
  single columns in the module* univariately (see below), but conditional on
  `m00_core`'s expanding PIT-mean surprise channels they are largely redundant. Drop
  unless the divergence family is also present, where the combination is mildly
  positive (+0.0131 full vs +0.0117 divergence-only).

---

## Within-module ablation (RT-063-*)

The **divergence/occupancy complex carries essentially all of the gain**:

```
m00 + dv,oc,qd,ab,rs  (47 cols)   0.58555   +0.01171   (89% of the full gain)
m00 + tl              ( 6 cols)   0.56974   -0.00410
m00 + rk              ( 6 cols)   0.57111   -0.00273
m00 + everything      (59 cols)   0.58698   +0.01313
```

Note the non-additivity: `tl_` and `rk_` are each negative alone yet the full module
beats the divergence-only slice by +0.00142. Two readings, not separated by this
experiment: (i) the extra 12 columns are genuinely complementary *once the occupancy
channel is present* (the disambiguator stories predict exactly this — asymmetry only
means something conditional on a divergence firing), or (ii) +0.00142 is inside
single-fold noise. **Recommended for promotion: the 47-column `(dv|oc|qd|ab|rs)`
slice**, which is the defensible, cheaper, robust result; the extra 12 columns should
only ride along if Agent 0 re-tests them at 5 folds.

## On the KS warning in the report

The brief flagged that KS-type statistics were *not* automatically useful in public
2026 work. Tested rather than skipped: `dv_u_w32_ks`, `dv_u_w128_ks`, `dv_u_exp_ks`,
`ab_w128_ks`, `rs_w128_ks` are in the module and inside the winning slice. Their
univariate |TS-AUC − ½| (0.024–0.027 for the calibrated expanding version) sits in the
same band as JS and chi², *not* below it. My read: the public finding is about **raw**
KS applied without a per-series null, where KS is dominated by the series' own tail
weight and dependence. Calibrated against a length-matched historical null, KS is
neither better nor worse than its siblings — it is one more geometry of departure,
worth its column but not worth privileging. This module does not test raw KS in
isolation, so the public claim is neither confirmed nor refuted, only bounded.

## An honest anomaly worth flagging to Agent 0

Univariate TS-AUC of individual columns on fold-0 validation rows is **sign-inverted**
for most "evidence-of-difference" statistics — e.g. `dv_u_exp_js` scores 0.4731,
`dv_u_exp_chi2` 0.4748, `oc_u_exp_maxdev` 0.4742, i.e. *more* divergence is
univariately associated with *fewer* breaks at fixed `t`.

This is **not** specific to `m02_dist`: the same probe on `m00_core` gives
`exp_absz_u2` = 0.4486, `exp_z_mean` = 0.4562, `w128_sur_mean` = 0.4785, while
`w128_z_sq` = 0.5322 and `exp_absz_mean` = 0.5107. So the reference module has the
identical pattern, and both modules nevertheless produce strongly positive TS-AUC once
a tree combines them. The signal in this dataset is therefore largely **interactive /
conditional**, not marginal — a raw divergence level is a proxy for a series-level
property (its own dependence and tail weight) that is negatively associated with
break-labelling in this population, and it only becomes evidence once the model
conditions on the companion channels. Practical consequences: (1) no one should screen
features on this dataset by univariate AUC; (2) whatever the conditioning variable is,
it is worth finding explicitly — a well-chosen series-level "how noisy is this series"
control could plausibly recover more than the +0.013 here. Strongest single columns by
|AUC−½|: `rk_cusum_u` 0.5413, `rk_cusum_u_raw` 0.5385, `rk_cusum_r_raw` 0.5368,
`rk_exp_var` 0.5358, `ab_exp_chi2` 0.4696, `qd_exp_qiqr` 0.5304.

## Cost / budget compliance

- 59 columns (budget ≤ 60).
- ~39 ms/series marginal (budget ~60 ms). The dominant cost is null construction:
  18 null blocks per series (4 stream×bin combinations × 4–5 window lengths), each a
  ≤500 × B occupancy matrix. Subsampling starts to ≤500 is what buys the ~5× headroom.
- `--workers 1` throughout; never more than one CPU-heavy job at a time.
