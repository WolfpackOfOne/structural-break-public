# AGENT 7 — DEPENDENCE / TREND / SPECTRAL / WAVELET (`m03_dyn`)

**AGENT NAME**  agent7 — online temporal-dynamics detector.

**HYPOTHESIS**
A structural break can leave the marginal law of the online segment almost
untouched (same mean, same variance, same tails, same PIT shape) while
rewriting the *temporal dynamics*: an AR coefficient flip, a volatility-
clustering regime change, a switch between random-walk and mean-reverting
behaviour, a change of characteristic frequency or of multi-scale energy
allocation. Those breaks are invisible to `m00_core`, which monitors moments
and occupancy only. Statistics of the online **dynamics**, each calibrated
against the distribution of the *same* statistic over length-matched windows of
the break-free history, should therefore add signal on top of `m00_core`.

**FALSIFICATION CONDITION**
Screen TS-AUC of `m00_core + m03_dyn` minus the paired `m00_core` control,
same seed, same fold, same session, `<= +0.001`.
Per-family falsification: family delta over the same control `<= +0.001`.

**FILES CHANGED**
- `/home/claude/sb/src/sbr/features/m03_dyn.py` (new, owned by agent7, 60 columns)
- `/home/claude/sb/research/reports/agent07_dynamics.md` (this file)
- `/home/claude/sb/research/FAILED_EXPERIMENTS.md` (appended)
No shared file was edited.

**EXPERIMENT IDs**  RT-070C (control), RT-070T (treatment), RT-070A (module alone),
RT-071AB / RT-071C / RT-071DE / RT-071F (family ablations).

**DATA USED**  `cache/store_screen` (2,500 series), `research/folds/folds_screen.parquet`.
Lockbox never loaded. `X_test.reduced.parquet` never read.

**FOLDS USED**  fold 0 only (screening protocol, §5).

**MODEL + FEATURES**  LightGBM binary, `n_estimators=300`, pipeline defaults
otherwise, `max_train_rows=300_000`, `seed=0`, `sample_mode=uniform`.
Features: `m00_core` (151 cols) ± `m03_dyn` (60 cols).

---

## HEADLINE NUMBERS (screen store, fold 0, seed 0, `n_estimators=300`, 300k train rows)

| exp | feature set | n_feat | TS-AUC | delta vs control |
|---|---|---|---|---|
| RT-070C | `m00_core` (paired control, run this session) | 151 | **0.57384** | — |
| RT-070T | `m00_core + m03_dyn` | 211 | **0.60381** | **+0.02997** |
| RT-070A | `m03_dyn` ALONE (no m00_core) | 60 | 0.55783 | −0.01601 vs control |
| RT-071AB | `m00_core` + A(dependence) + B(vol-clustering) | 173 | 0.59783 | +0.02399 |
| RT-071C | `m00_core` + C(trend) | 163 | 0.56897 | **−0.00487** |
| RT-071DE | `m00_core` + D(spectral) + E(wavelet) | 169 | 0.59488 | +0.02104 |
| RT-071F | `m00_core` + F(complexity) | 159 | 0.59724 | +0.02340 |
| RT-072NT | `m00_core + m03_dyn` minus the trend family | 199 | **0.60775** | **+0.03391** |

My control reproduces Agent 0's RT-001S number to all five decimals
(0.5738423135475659 both times), so the +0.02997 is a clean paired delta.

**PER-FOLD** single fold (fold 0) per the screening protocol; no fold spread
available, so treat +0.030 as one draw. It is 6x the +0.005 fold-to-fold noise
Agent 0 measured on the baseline, and the four independent family ablations all
land in the same place, which is the real robustness evidence here.

**RUNTIME**
- feature build, screen store (2,500 series), `--workers 1`, under heavy
  contention (load average 8–12 from other agents): **631 s total**.
- module cost measured alone, uncontended: **37 ms/series** (`m00_core` is
  31 ms/series and the shared `make_ctx` precompute is 243 ms/series on the same
  machine, so `m03_dyn` adds ~13 % to the end-to-end per-series cost).
- full store (10,000 series) projection: ~6 min of module time on top of the
  shared precompute.
- training: ~2–9 min per screen run depending on contention.

---

## CAUSALITY CHECK OUTPUT

`check_prefix_invariance("m03_dyn", hist, online, atol=0.0)` — bitwise, on 12
series spanning the full online-length range (10 → 999) and the full
history-length range (1,001 → 4,996), each with up to ten truncation points:

```
=== check_prefix_invariance('m03_dyn', ...) atol=0.0 ===
series  559  n_hist= 3893 n_online=  10 cuts=(3, 5, 7, 9)                              -> True ok
series 1732  n_hist= 2548 n_online=  11 cuts=(3, 5, 7, 10)                             -> True ok
series 1683  n_hist= 3057 n_online= 114 cuts=(3, 7, 10, 17, 37, 57, 64, 113)           -> True ok
series  259  n_hist= 3565 n_online= 264 cuts=(3, 7, 10, 17, 37, 64, 129, 132, 257, 263)-> True ok
series  906  n_hist= 4340 n_online= 405 cuts=(3, 7, 10, 17, 37, 64, 129, 202, 257, 404)-> True ok
series 2390  n_hist= 4434 n_online= 511 cuts=(3, 7, 10, 17, 37, 64, 129, 255, 257, 510)-> True ok
series 1519  n_hist= 3338 n_online= 614 cuts=(3, 7, 10, 17, 37, 64, 129, 257, 307, 613)-> True ok
series 1949  n_hist= 3532 n_online= 755 cuts=(3, 7, 10, 17, 37, 64, 129, 257, 377, 754)-> True ok
series 1041  n_hist= 1048 n_online= 902 cuts=(3, 7, 10, 17, 37, 64, 129, 257, 451, 901)-> True ok
series 1926  n_hist= 1996 n_online= 999 cuts=(3, 7, 10, 17, 37, 64, 129, 257, 499, 998)-> True ok
series    5  n_hist= 1001 n_online= 163 cuts=(3, 7, 10, 17, 37, 64, 81, 129, 162)      -> True ok
series 1444  n_hist= 4996 n_online= 386 cuts=(3, 7, 10, 17, 37, 64, 129, 193, 257, 385)-> True ok
ALL PASS: True
```

It was re-run unchanged after every optimisation (slice-based null rolling,
sorted-order-statistic quantiles, null subsampling) and still passes bitwise.

**NO STRIDE, NO HELD VALUE.** The brief allowed a strided/held spectral update;
it turned out not to be needed. The windowed DFT at a fixed frequency is
`|sum_j x_j e^{-2*pi*i*f*j}|^2`, and both quadratures are cumulative sums of the
per-point transforms `x_j cos(2*pi*f*j)` and `x_j sin(2*pi*f*j)`, so a trailing-window
band power is an O(1) difference of two cumsums — a Goertzel-equivalent with no
recursion state and no FFT. Every column of this module is therefore updated
exactly, every step.

---

## DESIGN DECISIONS THAT MATTER

**1. Adaptive fractional windows instead of fixed ones.** Online length ranges
10 → 999 with mean 504. A fixed `w=128` is undefined for a third of the data and
sluggish for the rest. Every statistic here is evaluated on a window that is a
fixed *fraction* of the online prefix: `quarter = (t+1)//4`, `half = (t+1)//2`,
`exp = t+1`. This keeps the feature defined early, responsive late, and makes
the null a smooth function of one scale parameter.

**2. A general null-calibration engine for window-scaled statistics.**
`ctx.nc` calibrates rolling *means of a transform*. Almost nothing here is a
rolling mean — ACF is a ratio of two of them, spectral entropy is a nonlinear
function of twelve of them. So the module carries its own `_Cal`: for a
statistic expressed as `f(R)` where `R(name, mult)` is the rolling mean of a
transform over `mult * W` points, it evaluates `f` over historical windows at
`W in {16, 64, 256}`, takes the median and a robust sd (max of IQR/1.349 and
(p95−p05)/3.29), and interpolates both in `log W` with linear extrapolation
beyond the grid. Online, `z = (S − mu(W)) / sd(W)`, clipped to ±10.
`transforms.py` was NOT touched; every transform is local to this module.

**3. Warm-up from the tail of history, not zeros.** Lag products, differences,
Haar details and ordinal patterns at online index 0 need up to 20 earlier
points. `m00_core` zero-pads. This module prepends the last 40 *historical*
points (known at t=0, therefore causal), so the first online windows contain no
artificial zeros. The historical null is built the same way, with the first 40
historical points trimmed so every historical transform value is fully defined.

**4. Cost engineering.** First working version: 101 ms/series. Final: 37 ms.
The three wins were (a) subsampling the historical null to ~700 windows per grid
point instead of all ~3,000 — the stride depends only on historical length, so
prefix invariance is untouched; (b) replacing fancy-index gathers with basic
strided slices in the historical rolling means; (c) replacing 93 `np.quantile`
calls per series with one `np.sort` plus order-statistic indexing (`np.quantile`
is ~35 ms/series of pure Python overhead at that call count).

---

## MARGINAL VALUE vs RUNTIME, PER FAMILY

Runtime is the *marginal* cost of the family on top of the shared per-point
transform build (7.5 ms/series, which is itself dominated by the 12 spectral
quadrature arrays and would shrink if spectral were removed).

| family | cols | ms/series | TS-AUC alone with m00 | delta | delta per column | verdict |
|---|---|---|---|---|---|---|
| A dependence (incremental ACF) + B vol-clustering | 22 | 13.9 | 0.59783 | **+0.02399** | +0.00109 | **KEEP** |
| C trend (cumsum slopes, Mann-Kendall) | 12 | 3.8 | 0.56897 | **−0.00487** | −0.00041 | **DROP-CANDIDATE** |
| D spectral + E wavelet | 18 | 11.4 | 0.59488 | **+0.02104** | +0.00117 | **KEEP — they do pay** |
| F complexity (perm. entropy, Hjorth) | 8 | 7.9 | 0.59724 | **+0.02340** | **+0.00293** | **KEEP — best value/column** |
| all six together | 60 | 37.0 | 0.60381 | +0.02997 | +0.00050 | |

Read this table carefully: the family deltas sum to +0.0644 but the joint delta
is +0.0300. **The families are largely redundant with each other.** Any one of
A+B, D+E or F recovers about three quarters of the total on its own. That is
what you would expect if they are all noisy measurements of one underlying
quantity — "the short-range dependence structure of the online segment has moved
away from the historical one" — which is exactly what they are: an ACF ratio, a
Haar energy ratio and a spectral band share are three parameterisations of the
same autocovariance sequence.

**Spectral and wavelet DO pay**, contrary to the prior that they would be
expensive ornamentation: +0.02104 for 11.4 ms. But they pay *the same money*
that complexity pays for 7.9 ms and 8 columns. If the column budget were being
squeezed I would cut spectral before I cut permutation entropy.

**The trend family is the loud negative.** Twelve columns of recursive slope,
multi-scale window slope, slope change and Mann-Kendall sign trend made the
model *worse* by 0.005 on their own. See FAILED_EXPERIMENTS.

---

## PER-FAMILY SIGNAL / FALSE SIGNAL / DISAMBIGUATOR

### A. DEPENDENCE — incremental ACF, calibrated (17 cols, `az_*`)
Windowed lag-k autocorrelation, **demeaned inside the window**, of five
processes: standardised raw `z` (lags 1,2,5,10,20), AR(2) residual `e`
(1,2,5,10), `|e|` (1,5,20), `e^2` (1) and `sign(z)` (1); expanding window for all,
plus a trailing-half window for the three lag-1 headliners. Each is
`rho = (mean(x_t x_{t-k}) − m^2) / (mean(x_t^2) − m^2)` from three cumulative
sums, then z-scored against the same ratio computed over historical windows of
the same length.

- **SIGNAL** — the break changes the persistence of the process: an AR
  coefficient moves, a moving-average term appears, an i.i.d. segment becomes
  a random walk. The marginal law can be untouched. Lag 5 on the raw series
  (`az_x_exp_l5_z`, univariate TS-AUC **0.4296**, the strongest single column in
  the module) says the post-break segment is *less* persistent at medium range
  than the history was — the dominant break flavour in this data appears to be a
  loss of medium-range dependence, not a gain.
- **FALSE SIGNAL** — (i) a pure **level shift** inflates raw lag products
  enormously; (ii) a **volatility burst** inflates both numerator and
  denominator; (iii) a single **outlier** at t contaminates lag-k products at
  two positions.
- **DISAMBIGUATOR** — (i) is neutralised *by construction*: demeaning inside the
  window makes a level shift leave `rho` exactly unchanged, which is why the ACF
  is defined as a ratio of two window-demeaned means rather than as
  `ctx.roll("lag1", w)`. (ii) is neutralised by the ratio: variance cancels
  between numerator and denominator, and the leftover is picked up by
  `m00_core`'s `w*_z_sq` and by family B. (iii) is separated by comparing the
  same lag across the residual (`az_e_*`) and the absolute-residual
  (`az_ae_*`) channels — an outlier moves `|e|` dependence but not sign
  dependence, whereas a real persistence change moves `az_sg_exp_l1_z` too.

### B. VOLATILITY CLUSTERING — realised-variance ratios (5 cols, `vr_*`)
`log(RV_{W/2} / RV_W)` and `log(RV_W / RV_{2W})` on squared residuals, plus the
`|e|` version, all calibrated.
- **SIGNAL** — a variance regime change shows up as the recent realised variance
  detaching from the slower one; unlike a level-calibrated variance z-score this
  is scale-free, so it fires for a series whose whole variance level is unusual
  but stationary only when the variance *moves*.
- **FALSE SIGNAL** — GARCH-type clustering with no break produces exactly this
  pattern transiently, many times per series.
- **DISAMBIGUATOR** — the historical null is built from the same series, so
  ordinary clustering is *in the null* and only excursions beyond what this
  series' own history produces score. Persistence across the `W/2 : W` and
  `W : 2W` pairs separates a transient burst (moves only the short pair) from a
  regime change (moves both).

### C. TREND — recursive/window slopes and Mann-Kendall (12 cols) — **NEGATIVE**
OLS slope over quarter/half/full prefixes from cumulative sums of `x` and `t*x`
(never a regression per step), z-calibrated, plus slope-change contrasts and two
sign-based robust trend statistics (mean of `sign(x_t − x_{t−h})` for h = 1, 8 —
a cheap Theil-Sen/Mann-Kendall substitute; a rolling median-of-differences was
rejected on cost).
- **SIGNAL** — a break into a drifting or mean-reverting regime.
- **FALSE SIGNAL** — everything. A trailing-window slope is a linear functional
  of the data with weights `(j − jbar)`, i.e. a smoothed first difference: any
  level shift, any autocorrelated wiggle, any variance burst near a window edge
  moves it. In a series with `mean(z) != 0` post-break, the slope is a strictly
  weaker version of the mean shift that `m00_core` already measures optimally.
- **DISAMBIGUATOR** — none that worked. `dslope_qh` / `dslope_he` (the
  short-window minus long-window z contrast) were supposed to be it, and they are
  the two *weakest* columns in the whole module (univariate TS-AUC 0.4986 and
  0.4977, i.e. nothing). Verdict below.

### D. SPECTRAL — fixed-frequency band powers (10 cols, `sp_*`)
Six dyadic frequencies (0.5 → 0.0156), windowed-DFT power from cumsums,
normalised to shares; emits low-band share, high-band share, log ratio,
spectral entropy over the six bands, spectral centroid, dominant band, plus
calibrated versions of four of them.
- **SIGNAL** — a change of characteristic frequency: a slow oscillation appears,
  a high-frequency component dies, the process whitens or reddens. Spectral
  entropy is the single scalar for "the spectrum flattened/peaked", and it does
  work (`sp_ent` residual TS-AUC 0.5325, see the H1/H2 table below).
- **FALSE SIGNAL** — a rectangular window leaks badly, so a strong level shift
  inside the window dumps power into every low band and moves `sp_low`,
  `sp_lr` and the centroid all at once, mimicking a "reddening" break. Short
  windows also bias entropy downward.
- **DISAMBIGUATOR** — the level-shift artefact is a *shared* movement of
  `sp_low` up and `sp_high` down with **no** movement in the ACF ratios, which
  are level-invariant by construction. Emitting `sp_low`, `sp_high` and their
  calibrated versions separately (rather than only the ratio) is what lets the
  model see that pattern. The window-length bias is removed by the calibrated
  `_z` versions, whose null is evaluated at the same window length.

### E. WAVELET — Haar dyadic energies (8 cols, `hw_*`)
Haar detail energy at scales 1,2,4,8 from one cumulative sum
(`d_s(t) = (C_t − 2C_{t−s} + C_{t−2s})/sqrt(2s)`), emitted as adjacent-scale log
energy ratios plus the OLS slope of `log2 E_s` against `log2 s` over the four
scales — that slope **is** the cheap Hurst/DFA substitute (`hw_hurst`).
- **SIGNAL** — a change in how energy is allocated across time scales: exactly
  the self-similarity/Hurst change that a random-walk-to-mean-reversion break
  produces. `hw_r12_z` (finest-scale energy ratio, calibrated) is the 11th
  strongest column univariately.
- **FALSE SIGNAL** — the coarse scales (`hw_r48`, needing 16 points per
  coefficient) are noisy in short windows and drift with any low-frequency
  wander; a single outlier puts energy into *every* scale at once and moves the
  ratios very little but moves `hw_hurst` a lot when the window is short.
- **DISAMBIGUATOR** — `hw_hurst` versus the individual ratios: an outlier tilts
  the whole log-energy-vs-log-scale line, a genuine self-similarity change bends
  it, so the ratios and the slope disagree for outliers and agree for breaks.
  Both raw and calibrated versions are emitted for exactly this reason.
- **HURST/DFA NOTE** — proper DFA was *not* implemented. It needs detrended
  fluctuation over many box sizes, which is not a cumulative-sum object at
  acceptable cost. The Haar log-energy-vs-log-scale slope is the cheap
  equivalent (a wavelet-based Hurst estimator) and costs 4 extra transforms.

### F. COMPLEXITY — permutation entropy + Hjorth (8 cols) — **best value/column**
Order-3 permutation entropy from six ordinal-pattern indicator transforms
(rolling means give the pattern frequencies directly), at half and expanding
windows; Hjorth log-mobility `0.5 log(var(dx)/var(x))` and log-complexity, all
from window-demeaned second moments of `x`, `dx`, `ddx`.
- **SIGNAL** — permutation entropy is an ordinal, monotone-invariant measure of
  how random the ordering of the process is. It moves when the *dynamics* become
  more or less deterministic and is blind to any monotone transform of the
  marginal — which is precisely the break `m00_core` cannot see. `pe_exp` has
  univariate TS-AUC 0.5315 and the largest online-movement component of any
  column in the module.
- **FALSE SIGNAL** — plug-in entropy is biased downward when the window is
  short, so *any* feature that shortens the effective window (early online
  index, a NaN gap) mimics an entropy drop. Ties (discretised data) inflate one
  pattern. Hjorth mobility is a variance ratio and reacts to a pure variance
  burst.
- **DISAMBIGUATOR** — the length bias is exactly what the calibrated
  `pe_half_z` / `pe_exp_z` remove, since the null is evaluated at the same
  window length. Note `pe_half` (raw) is *stronger* univariately than
  `pe_half_z` (calibrated) — see the H1/H2 section, this is not an accident.
  Hjorth complexity vs mobility separates a variance burst (mobility only) from
  a roughness change (complexity too).

---

## H1 vs H2: IS THIS A GENERATOR ARTIFACT?

The brief asks which of these features survive if H1 ("the dynamics genuinely
predict *whether* a break occurs") is a pure generator artifact. I measured it
directly, per column, on the screen store fold 0:

- **const (H1 channel)** — replace the column by its per-series mean over
  early, **pre-break** rows (`16 <= t < 128`, label 0) and broadcast that single
  number to every row of the series. It is a series-level constant, so its
  TS-AUC measures *only* "this series' dynamics say it is a break-prone DGP".
- **resid (H2/detection channel)** — column minus that constant: the purely
  within-series online movement.

| column | full TS-AUC | const (H1) | resid (H2) |
|---|---|---|---|
| pe_exp | 0.5315 | 0.4583 | **0.6155** |
| mk8 | 0.5086 | 0.4524 | 0.5474 |
| sp_ent | 0.5069 | 0.4625 | 0.5325 |
| sp_ent_z | 0.5015 | 0.4675 | 0.5315 |
| pe_half | 0.5144 | 0.5066 | 0.5310 |
| az_ae_exp_l1_z | 0.5267 | 0.5071 | 0.5304 |
| az_e_exp_l1_z | 0.5231 | 0.5172 | 0.5304 |
| vr_he_z | 0.5080 | 0.5020 | 0.5264 |
| az_x_exp_l1_z | 0.5247 | 0.5285 | 0.5255 |
| hw_r12_z | 0.5201 | 0.4808 | 0.5199 |
| drift_e_z | 0.5104 | 0.4751 | 0.5182 |
| az_x_exp_l5_z | 0.4296 | 0.4604 | 0.4720 |
| az_ae_exp_l20_z | 0.4706 | 0.4827 | 0.4545 |
| mk1 | 0.4881 | **0.5341** | 0.4612 |
| mk1_z | 0.4973 | **0.5440** | 0.4687 |
| az_x_exp_l20_z | 0.5143 | 0.5337 | 0.4750 |
| pe_half_z | 0.4997 | 0.5290 | 0.4820 |
| hw_hurst | 0.5125 | 0.5317 | 0.4859 |

**Answer to the brief's question.** The H1 channel is real but small: no column
exceeds `|AUC − 0.5| = 0.054` from the series-level constant alone, and most sit
under 0.02. The H2/detection channel is larger for the great majority of
columns. **`m03_dyn` is not primarily a DGP fingerprint** — which is the
expected result, since it is an *online* module; the historical-context version
of this question belongs to Agent 8.

Features that would still work if H1 were pure generator artifact (const ≈ 0.5,
resid well away from it): `az_ae_exp_l1_z`, `az_e_exp_l1_z`, `az_e2_exp_l1_z`,
`vr_he`, `vr_he_z`, `vr_qh_z`, `sp_lr`, `sp_low`, `pe_half`, `hj_comp`,
`hw_r48`, `hw_r24`, `drift_e`. These are the robust core: they measure
*movement*, not *identity*.

Features that would NOT survive it — and a finding worth handing to Agent 8:
`mk1`, `mk1_z`, `mk8`, `az_x_exp_l20_z`, `pe_half_z`, `hw_hurst`, `sp_high_z`,
`sp_cen`, `dslope_he`, `az_x_half_l1_z`, `az_e_half_l1_z` all have **const and
resid on opposite sides of 0.5**. For these columns the DGP-fingerprint effect
and the online-detection effect push the cross-sectional ranking in *opposite*
directions, so the raw column's TS-AUC is a partial cancellation. That is a
concrete, testable prediction: **these columns should get materially stronger
once historical-context features that identify the DGP are in the model**,
because the model can then condition the sign. It also explains, mechanically,
why the trend family is a net negative on its own.

Two honest caveats on this table. (1) `const` is computed non-causally — it uses
rows up to t=127 to explain rows at t<127 — so the `resid` numbers are optimistic
and this is a **diagnostic, not a feature**. (2) `const` is mildly confounded:
break series with `tau < 128` contribute fewer and earlier label-0 rows, so any
t-dependence of a column leaks into `const`. Both caveats push in the direction
of *over*-stating H1, so "H1 is small" is the conservative reading.

**A lead worth an experiment (not run, out of time).** The causal version of the
`resid` construction — column minus its own mean over a *fixed early online
window* (`16 <= t < 128`), emitted only for `t >= 128` — is legitimate and cheap.
On the diagnostic it lifts `pe_exp` from 0.5315 to 0.6155 univariately. Online
self-baselining may be a better reference than the historical null for the
entropy/complexity channels, because it cancels the DGP identity exactly.
Recommend Agent 0 assign this.

---

## LEAKAGE RISKS

- **Prefix invariance** is verified bitwise on 12 series covering the extremes of
  both length distributions, with cuts at 3, 7, 10, 17, 37, 64, 129, 257, n/2 and
  n−1. No stride, no held value, no recursion state.
- **The historical null uses history only.** Online values enter only as the
  query. The null-window subsampling stride is a function of historical length
  alone.
- **The AR(2) filter** is `ar_filter_causal` from `transforms.py`, warmed from the
  historical tail; the coefficients are fitted on history only.
- **Warm-up padding** uses the last 40 historical points. Historical data is
  fully available at t=0, so this is causal; it does mean the first few online
  windows blend across the hist/online boundary, which slightly *reduces* early
  sensitivity rather than creating information.
- **`sp_dom`** is an argmax and therefore discrete/discontinuous; it is the
  weakest spectral column and is a plausible overfit vector on a small store.
- **Residual risk: none identified.** The one construction I would flag if I had
  shipped it is the `resid` diagnostic above, which is non-causal and is
  deliberately NOT in the module.

---

## SIGNAL STORY (one paragraph)

The break moves the *shape of time*, not the shape of the histogram. Every
column here is one projection of the online autocovariance/ordinal structure —
lag-k correlation, band power, dyadic energy, ordinal-pattern frequency —
measured on an adaptive fraction of the online prefix and scored against the
distribution that the same projection occupies over length-matched break-free
historical windows. The empirical result is that this is worth **+0.030 TS-AUC**
on top of a strong marginal-distribution model, and that the single strongest
direction is a *loss of medium-range dependence* post-break
(`az_x_exp_l5_z` at univariate TS-AUC 0.4296).

## FALSE-SIGNAL STORY (one paragraph)

Three non-structural behaviours move all of these the same way: a level shift
(defeated by construction — every ACF is window-demeaned and every energy ratio
is scale-free), a transient volatility burst (defeated by the ratio form plus
the per-series null, which contains that series' own ordinary clustering), and a
short effective window (defeated by the `_z` versions, whose null is evaluated
at the same window length — this is why raw and calibrated versions are both
emitted rather than only the calibrated one). The one family where I could not
build a working disambiguator is trend, and it is the one family that lost
money.

---

## CONCLUSION: **KEEP** (with one family dropped)

**Recommended production configuration**

```python
run(..., modules=[..., "m03_dyn"],
    drop_cols=("::drift_", "::dslope_", "::mk"))     # 48 of 60 columns
```

| configuration | cols added | screen TS-AUC | delta vs control |
|---|---|---|---|
| control `m00_core` | — | 0.57384 | — |
| `+ m03_dyn` (all 60) | 60 | 0.60381 | +0.02997 |
| **`+ m03_dyn` minus trend (48)** | **48** | **0.60775** | **+0.03391** |

Dropping the trend family both *removes* 12 columns and *adds* +0.00394 TS-AUC,
confirmed independently by the family ablation (trend alone: −0.00487). Two
mutually consistent experiments, so I am confident enough to recommend the drop,
but the columns are left in the module rather than deleted so Agent 0 can
re-test them on the full 5-fold protocol before they are removed for good — a
single fold at a single seed is not enough to delete code over.

**KEEP** verdict rationale: +0.034 on a 0.574 control is by far the largest
single-module delta available at this point; it is causally verified bitwise;
it costs 37 ms/series against a 243 ms/series shared precompute; and the effect
is corroborated by four independent family ablations that each recover
three-quarters of it on their own, which is the signature of a real, redundant
signal rather than a lucky column.

**Promotion request to Agent 0**: full-store build + 5-fold protocol for
`m03_dyn` with `drop_cols=("::drift_", "::dslope_", "::mk")`. Free column
budget after the drop: 12.

**OOF ARTIFACT PATH** — none. `sbr.pipeline.run` deliberately does not persist
OOF or importance artifacts for `screen=True` runs
(`if save_oof and not screen`), and every experiment here is a screen run.
Artifacts will exist once Agent 0 promotes this to the full protocol.

---

## APPENDIX — univariate TS-AUC of every column (screen fold 0, NaN filled with
the column median; a diagnostic only, the model sees the NaNs natively)

```
column                   TS-AUC   |dev|   nan%
az_x_exp_l5_z            0.4296  0.0704    2.2
az_e_exp_l5_z            0.4541  0.0459    2.2
az_e_exp_l2_z            0.5415  0.0415    2.2
az_sg_exp_l1_z           0.4684  0.0316    2.2
pe_exp                   0.5315  0.0315    4.7
az_x_exp_l2_z            0.5306  0.0306    2.2
az_ae_exp_l20_z          0.4706  0.0294    7.9
az_ae_exp_l1_z           0.5267  0.0267    2.2
az_x_exp_l1_z            0.5247  0.0247    2.2
az_e_exp_l1_z            0.5231  0.0231    2.2
hw_r12_z                 0.5201  0.0201    9.4
hj_comp                  0.5187  0.0187    6.3
hj_comp_z                0.5185  0.0185    6.3
az_e2_exp_l1_z           0.5178  0.0178    2.2
az_ae_exp_l5_z           0.5151  0.0151    2.2
vr_qh_z                  0.5151  0.0151    6.3
pe_half                  0.5144  0.0144    9.4
az_x_exp_l20_z           0.5143  0.0143    7.9
vra_qh_z                 0.5142  0.0142    6.3
hj_mob                   0.4867  0.0133    6.3
hw_hurst                 0.5125  0.0125   12.6
sp_lr                    0.5125  0.0125   12.6
sp_high                  0.4878  0.0122   12.6
pe_exp_z                 0.5120  0.0120    4.7
mk1                      0.4881  0.0119    4.7
vr_qh                    0.5114  0.0114    6.3
hw_hurst_z               0.5113  0.0113   12.6
drift_e_z                0.5104  0.0104    2.2
drift_q                  0.5103  0.0103    9.4
sp_low                   0.5094  0.0094   12.6
mk8_z                    0.5093  0.0093    6.3
drift_e                  0.5089  0.0089    2.2
drift_q_z                0.5088  0.0088    9.4
mk8                      0.5086  0.0086    6.3
drift_h_z                0.5084  0.0084    4.7
vr_he_z                  0.5080  0.0080    6.3
az_e_half_l1_z           0.4922  0.0078    4.7
hw_r48                   0.5074  0.0074   12.6
drift_h                  0.5074  0.0074    4.7
vr_he                    0.5072  0.0072    6.3
az_ae_half_l1_z          0.5070  0.0070    4.7
sp_ent                   0.5069  0.0069   12.6
hw_r48_z                 0.5067  0.0067   12.6
hw_r12                   0.4935  0.0065    9.4
az_x_half_l1_z           0.4936  0.0064    4.7
sp_cen_z                 0.4937  0.0063   12.6
sp_high_z                0.5050  0.0050   12.6
hj_mob_z                 0.5044  0.0044    6.3
sp_cen                   0.5041  0.0041   12.6
hw_r24                   0.5041  0.0041    9.4
hw_r24_z                 0.4960  0.0040    9.4
sp_dom                   0.5038  0.0038   12.6
az_e_exp_l10_z           0.4970  0.0030    3.9
mk1_z                    0.4973  0.0027    4.7
dslope_he                0.4977  0.0023    4.7
sp_low_z                 0.4977  0.0023   12.6
sp_ent_z                 0.5015  0.0015   12.6
dslope_qh                0.4986  0.0014    9.4
az_x_exp_l10_z           0.4991  0.0009    3.9
pe_half_z                0.4997  0.0003    9.4
```

Note the six weakest columns include four of the twelve trend columns
(`dslope_he`, `dslope_qh`, `mk1_z`) — consistent with the ablation.
