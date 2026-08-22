# C2-A — `m04_resid` REDUNDANCY AUDIT

**2026-08-22, branch `research/wave5-alpha`. No competition data. No TS-AUC.**

Purpose: decide, **before spending any training compute**, whether the proposed
dependence-likelihood and residual-CUSUMSQ families carry information
`m04_resid` does not already have. Two of the three prior wave-1 negatives
(`tl_` −0.00410, `rk_` −0.00273) were individually strong statistics that failed
purely because they re-stated an existing channel. This audit exists so that
failure mode costs nothing this time.

---

## 1. WHAT `m04_resid` ACTUALLY IS

**60 columns = 10 representations × their monitor set.**

| tier | representations | monitors | cols |
|---|---|---|---|
| A | `raw`, `ar1`, `ar2`, `ar5`, `vol`, `cmb` | `mean`, `var`, `abs`, `tail`, `acf1`, `acf1sq`, `llr`, `w32_llr` | 6 × 8 = 48 |
| B | `arR`, `arH`, `volG`, `volM` | `var`, `acf1sq`, `llr` | 4 × 3 = 12 |

**Representations** (every map fitted on HISTORY only, applied causally forward,
each divided by its own historical sd so all sit on a "1 = historical innovation
sd" scale):

| name | definition | state |
|---|---|---|
| `raw` | `z = (x − mu_h)/sd_h` | none (constant historical scale) |
| `ar1`/`ar2`/`ar5` | AR(p) residual, OLS on standardised history, filtered forward warmed by the historical tail | fixed coefficient vector |
| `arR` | ridge AR(3), λ = 0.05 on a unit-variance Gram | fixed coefficients |
| `arH` | Huber-IRLS AR(2), 3 fixed reweight steps | fixed coefficients |
| `vol` | `z / sqrt(EWMA_pred(z², halflife 22))` | 1 EWMA state |
| `volM` | `z / EWMA_pred(min(|z|,4), halflife 63)` | 1 EWMA state (winsorised) |
| `volG` | `z / sqrt(GARCH(1,1)_pred)`, variance-targeted, deterministic 12-point QMLE grid on history | 1 variance state |
| `cmb` | AR(2) residual then EWMA(22) volatility normalisation | AR + EWMA |

**Monitors** — each is the rolling mean of a per-point transform, null-calibrated
against the empirical distribution of the SAME statistic over contiguous
length-matched windows of the residual stream computed **on history**:

| monitor | per-point transform | break class targeted |
|---|---|---|
| `e_mean` | `e` | residual mean / level |
| `e_var` | `clip(e)²` | residual variance |
| `e_abs` | `|clip(e)|` | robust residual scale |
| `e_tail` | `1[|clip(e)| > q95_hist]` | residual tail rate |
| `e_acf1` | `e_t · e_{t−1}` | **lag-1 dependence** |
| `e_acf1sq` | `(e²−1)(lag²−1)` | ARCH-type squared dependence |
| `e_llr` | `mean(e²) − log(var(e)) − 1` | joint mean+variance Gaussian GLR |
| `w32_llr` | same, trailing window 32 | same, one short scale |

## 2. COVERAGE CLASSIFICATION

| class | covered? | by what |
|---|---|---|
| residual mean | **yes** | `e_mean` × 6 |
| residual variance | **yes** | `e_var` × 10 |
| residual autocorrelation | **yes**, lag-1 only | `e_acf1` × 6 |
| lag product | **yes** | `e_acf1`, `e_acf1sq` |
| coefficient drift | **NO** | AR coefficients are frozen from history and never re-estimated or compared online |
| prediction-error change | **yes** | `e_llr`, `w32_llr` |
| cumulative evidence | **partly** | expanding means, but no CUSUM path, no reset, no running max |
| max-over-tau evidence | **NO** | nothing in `m04` maximises over a candidate changepoint |
| recent-vs-history evidence | **yes** (history only) | every null is historical; **no adaptive online baseline** |

**The structural finding: 54 of 60 columns are EXPANDING-window.** Only the six
`w32_llr` columns use a trailing window. There is **no** max-over-τ statistic, **no**
adaptive online baseline, and **no** scale-free (bridge-type) statistic anywhere
in the module.

## 3. NOVELTY MATRIX

| proposed mechanism | already in m04? | mathematically equivalent? | strongly related? | genuinely missing? |
|---|---|---|---|---|
| lag-1 dependence level | **yes** (`e_acf1`) | — | — | no |
| ARCH / squared dependence | **yes** (`e_acf1sq`) | — | — | no |
| residual variance level | **yes** (`e_var`) | — | — | no |
| residual scale, robust | **yes** (`e_abs`) | — | — | no |
| joint mean+var GLR | **yes** (`e_llr`) | — | — | no |
| cumulative squared-residual **magnitude** | no | **near** (R²=0.9358 from m04 alone) | yes | **NO** |
| CUSUMSQ **localisation** (change age) | no | no (R²=0.34 from m04) | weakly | yes, but see §5 |
| **AR-coefficient change, max over τ** | **NO** | no (R²=0.6886 from m04) | partly | **YES** |
| adaptive online dependence baseline | **NO** | no | no | **YES** |

## 4. METHOD — AND A CORRECTION TO THE FIRST ANALYSIS

Prototypes of each candidate were computed alongside the full existing bank
(`m04_resid` 60 cols + `m11_focus` 21 cols) over 7 synthetic DGP families
(no-break, mean shift, permanent variance shift, variance spike, permanent
dependence change, dependence burst, tail cluster), with AR coefficients drawn
per series so redundancy is not measured on one narrow regime. Redundancy is
ridge-regression R² of the candidate on the existing bank.

**The first mechanism comparison pooled ROWS and gave unstable answers** — the
dependence candidate beat the incumbent on one draw (1.08 vs 1.32 → 1.16 vs
0.84 on the next). Pooling rows conflates within-series and between-series
variance, and TS-AUC is a **series-ranking** metric. Every comparison below was
therefore recomputed on a **series-level statistic** with 40 series per arm and
a 2,000-replicate bootstrap. This correction reversed nothing in the final
verdicts, but it turned two unstable readings into decided ones, and it is the
same lesson `FAILED_EXPERIMENTS` already records: *one draw is not a result.*

## 5. VERDICTS

### C2-B — DEPENDENCE LIKELIHOOD: **PROCEED**

| | |
|---|---|
| redundancy vs m04 alone | R² = 0.6886 |
| redundancy vs m11 alone | R² = 0.3007 |
| redundancy vs both | R² = 0.7325 |
| own mechanism test (permanent dependence change vs dependence burst), series level, n=40/arm | incumbent `ar1_e_acf1` **|d| = 1.048**; candidate **|d| = 2.728** |
| paired bootstrap, candidate − incumbent | **+1.680, 95% CI [+1.151, +2.324], 100% of replicates positive** |

27% of the candidate is unexplained by the entire existing bank, and on the
falsification test the family exists to pass it beats the incumbent decisively.
**The missing mechanism is coefficient drift under max-over-τ**: `m04` freezes
its AR coefficients from history and never asks whether the *coefficient* has
changed — only whether the lag-product *level* has. Those differ exactly when
the innovation variance also moves, because an autocovariance changes when
either the coefficient or the variance changes, while a regression coefficient
does not.

### C2-C — RESIDUAL CUSUMSQ: **REJECTED BEFORE TRAINING**

| | |
|---|---|
| redundancy of the bridge magnitude vs m04 alone | **R² = 0.9358** |
| own mechanism test (permanent variance shift vs temporary spike), series level, n=40/arm | incumbent `cmb_e_abs` **|d| = 1.979**; CUSUMSQ magnitude **1.142**; CUSUMSQ age **0.054** |
| bootstrap, magnitude − incumbent | **−0.870, 95% CI [−1.497, −0.231]** |
| bootstrap, age − incumbent | **−1.835, 95% CI [−2.369, −1.306]** |

**The incumbent wins on the candidate's own target mechanism, with CIs excluding
zero, and 94% of the magnitude channel is already determined by `m04` alone.**

*Why, mechanistically.* The bridge's scale-freeness was supposed to be its
virtue. It is the defect: normalising `C_k/C_t` discards the overall scale
level, and the overall scale level is precisely what separates a permanent 2×
variance shift from a 12-point 4× spike. Both look inhomogeneous to a bridge.
Meanwhile `cmb_e_abs` — expanding mean of |residual| after AR+EWMA
normalisation, null-calibrated — measures sustained scale elevation directly.
The hypothesis that cumulative squared-innovation evidence captures persistent
variance breaks *more cleanly* than current recent-window summaries is
**falsified**: the current summaries are roughly twice as good.

*What is NOT claimed.* The change-age channel is genuinely non-redundant
(R² = 0.34). It is rejected not for redundancy but because it carries almost no
signal on its own target contrast (|d| = 0.054, indistinguishable from zero).
Non-redundant and uninformative are different failures; this one is the second.

## 6. COST

Zero training runs. Zero TS-AUC values. Zero competition data. One family
promoted, one family rejected, on replicated synthetic mechanism evidence.
