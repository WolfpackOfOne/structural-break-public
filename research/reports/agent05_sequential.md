# Agent 5 — Multi-Scale Sequential Detector Bank (`m01_seq`)

**AGENT NAME** agent5 — multi-scale sequential detector bank

**HYPOTHESIS**
A bank of classical sequential change detectors (CUSUM, CUSUM-SQ, Page-Hinkley,
Shiryaev-Roberts, dyadic GLR / mixture-GLR, multi-half-life EWMA), run on the
historically-standardised raw stream *and* the AR-residual stream, calibrated
against a **per-series historical-null replay of the identical recursion**, and
exposed through the *shape* of each evidence path (running peak, decayed peak,
time-since-peak, persistence, slope, current/peak ratio) rather than a collapsed
score, adds information that a window-based evidence module (`m00_core`) does
not already carry. The specific claim: window statistics tell you *how far* the
online segment currently is from the null; a sequential detector's path shape
tells you *whether the departure accumulated and stayed*, which is the
permanent-vs-transient distinction the label actually encodes.

**FALSIFICATION CONDITION**
Screen fold-0 TS-AUC of `m00_core + m01_seq` improves by **≤ +0.002** over the
paired `m00_core`-only control run in the same session with the same seed.
Secondary falsification: the shape channels are redundant, i.e. a levels-only
variant of `m01_seq` gets within 0.002 of the full variant.

**FILES CHANGED**
- `/home/claude/sb/src/sbr/features/m01_seq.py` (new, owned by agent5, 60 columns)
- `/home/claude/sb/research/reports/agent05_sequential.md` (this file)
- `/home/claude/sb/research/FAILED_EXPERIMENTS.md` (appended)
No shared/core file was edited.

**EXPERIMENT IDs**
| ID | Feature set | m01_seq cols used | screen fold-0 TS-AUC | Δ vs control |
|---|---|---|---|---|
| RT-010C | `m00_core` (control) | — | **0.57384** | — |
| RT-010T | `m00_core + m01_seq` | 60 | **0.58809** | **+0.01425** |
| RT-010A | `m01_seq` alone | 60 | **0.54392** | −0.02992 |
| RT-010N | `m00_core + m01_seq` levels-only | 18 | **0.58229** | +0.00845 |

Control reproduces Agent 0's published `m00_core` screen fold-0 number
(0.5738423135475659) to the last digit, so the pairing is exact.

**DATA USED** `cache/store_screen` (2,500 series, 1,270,042 online rows),
features cached to `cache/features_screen/m01_seq.npy`. Lockbox never loaded;
`X_test.reduced.parquet` never read.

**FOLDS USED** screen fold 0 as validation, screen folds 1–4 as train
(`research/folds/folds_screen.parquet`, unchanged).

**MODEL + FEATURES** LightGBM binary, `n_estimators=300`, lr 0.05, 63 leaves,
`min_data_in_leaf=200`, `feature_fraction=0.7`, `bagging 0.7/1`, `lambda_l2=5`,
`max_bin=127`, `num_threads=2`, seed 0, `max_train_rows=300_000` (uniform).
Identical in every run; only the feature set changes.

**TS-AUC (screen)** control 0.57384 → treatment **0.58809**.
**PER-FOLD** single screen fold (fold 0) per protocol §5; `fold_std = 0`.
**DELTA VS CONTROL** **+0.01425 TS-AUC** (≈ +7.1 % of the control's excess over 0.5).
Shape-only contribution: RT-010T − RT-010N = **+0.00580**.

**RUNTIME**
- module build, screen store, `--workers 1`: **479 s** for 2,500 series
  (≈ 190 ms/series wall under heavy contention; isolated timing is
  **7.5 ms/series** for `m01_seq` itself, on top of the ~72 ms shared
  `make_ctx` precompute that every module pays). Comfortably inside the
  ~60 ms/series module budget.
- training runs: ~8–10 min each under load average ~10 on 2 cores.

---

## CAUSALITY CHECK OUTPUT

`check_prefix_invariance("m01_seq", hist, online, cuts=(1,2,3,7,10,37,101,313,777), atol=0.0)`
on 12 series spanning the whole online-length range (10 → 999) and history
range (1,358 → 4,873):

```
series   559  n_hist= 3893 n_online=  10  ->  True  ok
series  1807  n_hist= 1823 n_online=  53  ->  True  ok
series  2165  n_hist= 2391 n_online= 167  ->  True  ok
series  1484  n_hist= 4873 n_online= 308  ->  True  ok
series  1574  n_hist= 4184 n_online= 458  ->  True  ok
series   770  n_hist= 1875 n_online= 614  ->  True  ok
series   201  n_hist= 1358 n_online= 707  ->  True  ok
series  1651  n_hist= 4829 n_online= 802  ->  True  ok
series   326  n_hist= 2309 n_online= 901  ->  True  ok
series  1729  n_hist= 4074 n_online= 973  ->  True  ok
series  1462  n_hist= 1900 n_online= 990  ->  True  ok
series  1926  n_hist= 1996 n_online= 999  ->  True  ok
ALL PASS: True
```

Bitwise (`atol=0.0`), 9 truncation points per series, 108 truncation tests.

**One real bug was caught by this harness and fixed.** The first version of the
mixture-GLR normalised by a *global* count of dyadic window lengths that fit in
`n_online`, so the value at row `t` depended on how long the series turned out
to be. It failed at prefix 1 on every series. The fix makes the mixture
row-local (only windows with `L ≤ t+1` contribute at row `t`). This is exactly
the class of leak the protocol's §3 harness exists to catch, and it would not
have shown up in any screening number.

---

## HOW THE CALIBRATION WORKS (the load-bearing part)

`ctx.nc` calibrates *rolling means of transforms*. CUSUM, Page-Hinkley,
Shiryaev-Roberts and GLR are **path dependent** — they are not rolling means of
anything — so `ctx.nc` cannot be used on them directly. Instead, for every
channel:

1. **Replay the identical recursion over history.** The historical segment is
   guaranteed break-free, so it is a valid null path. The online stream and the
   historical stream are standardised by the *same* `ctx.hp` constants
   (`mean` → `(x−μ)/σ`, `res_mean` → AR(2) residual / `ar_sigma`, `sq`
   standardised by the historical mean/sd of `z²`), so the recursion sees
   statistically identical inputs.
2. **Marginal null** — sort the historical detector path (after a burn-in of
   `min(64, H/4)`) and take the exact upper-tail empirical p-value of the
   current online value, as `−log10 P(null ≥ x)`.
3. **Running-peak null** — trailing rolling maxima of the historical detector
   path over dyadic windows 32/128/512, computed by **doubling**
   (`M_{2w}[i] = max(M_w[i], M_w[i−w])`), i.e. O(H log H) with a handful of
   vector ops and no Python loop. The online running peak at step `t` is scored
   against the grid window nearest `t` in log space — a function of `t` only, so
   it stays causal. This is the right null for "the largest value this detector
   has reached in `t` steps".
4. **Threshold / scale constants** — `q50`, `q99`, `max` of the historical path
   give the persistence threshold and the slope/peak-ratio normaliser.

Total extra cost per series is O(n_hist) plus a handful of sorts.

**Tail-resolution extension (matters more than it looks).** An empirical
p-value cannot resolve below `1/N`, so every value above the historical maximum
would collapse onto one capped number — precisely where the breaks live, and
with a cap that varies with `n_hist` (3.27 for H=1,000 vs 4.00 for H=5,000),
destroying cross-series comparability exactly in the decisive region. Each
surprise is therefore extended above the historical maximum by
`log1p((x − max_hist)/scale_hist)`, which keeps the ordering informative and
roughly comparable across series. The `*_pk` / `*_pkr` columns that dominate the
importance table live almost entirely in this extended region.

**Robust scale floor.** The first version floored the historical spread at
`1e-9`; for series whose AR-residual CUSUM null is near-degenerate this produced
`ce50_slp` values of ±3×10⁷ (column mean −2357, sd 244k) — a single column of
pure garbage. The floor is now
`max(q99−q50, 0.1·(max−q50), 1e-3·max(|q50|,1))` and every derived ratio is
clipped. Worth stating plainly: the first build of this module had a column
that was numerically meaningless, and only a per-column distribution audit
found it.

**Documented approximations**
- **GLR uses a dyadic candidate-change-point grid** (`L ∈ {1,2,…,512}`) instead
  of a full O(t) scan every step: O(log n) per point. A true change point
  falling between grid nodes is recovered with a segment length off by at most a
  factor 2, i.e. the GLR amplitude is recovered to within ~1/√2. Given that we
  hand the model the *shape* of the GLR path rather than a threshold crossing,
  this loss is immaterial; the exact scan would have cost ~500× more.
- The mixture-GLR is the row-local log-mean-exp over the same dyadic set.
- The running-peak null comes from a historical path that is not reset to zero
  at each window start, while the online peak does start from zero. This is
  mildly conservative, smooth in window length, and identical in direction for
  every series.
- The marginal null is the *stationary* detector distribution while the online
  detector at step `t` has only run `t` steps. This makes early rows look quiet.
  Because TS-AUC compares series only at a fixed online index, a `t`-monotone
  distortion shared by all series cannot change any within-`t` ranking.

**Implementation notes.** CUSUM/Page-Hinkley use the closed form
`S_t = C_t − min_{j≤t} C_j` (one `cumsum` + one `minimum.accumulate`) — fully
vectorised, no online loop. The decayed peak uses
`log P_t = t·logρ + cummax_s(log D_s − s·logρ)`, which is exact and cannot
overflow. EWMA uses `scipy.signal.lfilter` (a sequential IIR, so prefix-stable)
with a `1−(1−a)^t` bias correction. The **only** sequential loop is the
Shiryaev-Roberts recursion `logR_t = l_t + softplus(logR_{t−1})`, which has no
closed form; it is a numba `njit(cache=True, fastmath=False)` kernel (IEEE
deterministic; a pure-Python fallback is present and gives identical results).
Every time-axis reduction in the module is a sequential numpy accumulator; every
cross-channel reduction is row-wise. That is *why* the bitwise check passes.

---

## PER-CHANNEL STORIES

### 1. CUSUM on the standardised raw stream — `cz50_*` (k=0.5, full shape), `cz25_*`, `cz100_*`, `cz50_up`, `cz50_dn`
- **SIGNAL** A permanent shift in the mean makes `z_t − k` positive on average,
  so the reflected random walk stops returning to zero and grows ~linearly in
  the time since the break. `k` selects the shift size the detector is tuned to:
  `k=0.25` catches small persistent drifts (needs many post-break points),
  `k=1.0` catches large abrupt shifts almost immediately. Both directions are
  emitted separately (`_up`, `_dn`) plus the two-sided max, because the model
  should be able to learn asymmetric behaviour without us assuming it.
- **FALSE SIGNAL** A single large outlier injects one big increment and the
  CUSUM jumps once; a temporary level excursion (a "blip regime" that reverts)
  drives it up and then it decays back to zero as the negative slack `−k`
  accumulates. Heavy tails make both far more frequent than Gaussian intuition
  suggests — and the historical-null replay handles this automatically, because
  a heavy-tailed series produces a heavy-tailed *historical CUSUM path*, so the
  same jump is unsurprising for that series.
- **DISAMBIGUATOR** `cz50_rel = current/peak` and `cz50_tsp = (t − argmax)/t`.
  A break keeps `rel ≈ 1` and `tsp ≈ 0` indefinitely; an outlier gives one high
  peak followed by `rel → 0` and `tsp → 1`. `cz50_per` (fraction of steps above
  the historical q99) separates "one big spike" from "pinned high for 200 steps".

### 2. CUSUM on the AR-residual stream — `ce50_*` (full shape), `ce25_*`, `ce50_up/dn`
- **SIGNAL** A break in the *dependence* structure (AR coefficient change) or a
  mean shift in a strongly autocorrelated series. Whitening with the historical
  AR(2) before running the recursion removes the serial correlation that
  otherwise inflates the raw CUSUM's variance and makes its null wide and
  useless.
- **FALSE SIGNAL** A pure variance break also inflates the residual stream, and
  a *misspecified* historical AR fit leaves structure in the residuals that
  mimics drift.
- **DISAMBIGUATOR** The variance channel `sq50_*` moves for variance breaks and
  the raw channel `cz50_*` moves for level breaks; `ce50` moving while both are
  quiet is the dependence-break signature. The importance ranking supports this:
  `ce50_pk` and `ce25_pkr` are separately valuable alongside `cz50_pk`, i.e. the
  residual channel is not a copy of the raw channel.

### 3. Variance CUSUM (CUSUM-SQ) — `sq50_*` (full shape), `sq25_*`, `sq50_up/dn`
- **SIGNAL** Volatility regime change. Run on `(z² − E_hist[z²]) / sd_hist[z²]`
  so it is a proper standardised stream; `_up` = volatility increase,
  `_dn` = volatility collapse (a real and often-missed break type).
- **FALSE SIGNAL** A transient vol burst (a few clustered large moves) — very
  common in financial-style series with GARCH dynamics — and, for `_dn`, a quiet
  patch that is just a fluctuation of a heteroskedastic process.
- **DISAMBIGUATOR** `sq50_per` (persistence above the historical q99) and
  `sq50_tsp`. A GARCH burst has a short high-persistence window and then decays;
  a variance break holds `per` climbing linearly. `sq50_tsp` is the third most
  useful `tsp` channel in the model.

### 4. Page-Hinkley — `phu_*`, `phd_*`
- **SIGNAL** Same target as CUSUM but referenced to the *running online mean*
  rather than the historical mean. This makes it insensitive to a constant
  historical-vs-online offset (calibration error, slow drift already present in
  history) and sensitive to a change *within* the online segment.
- **FALSE SIGNAL** Because the reference itself adapts, a slow drift can be
  partially absorbed and PH under-reacts; conversely, in a short online segment
  the running mean is noisy and PH over-reacts early.
- **DISAMBIGUATOR** The CUSUM channels, which use the fixed historical
  reference. PH firing while CUSUM is quiet = a change relative to the online
  segment's own level. `phd_pkr` and `phu_pkr` are both top-10 features, so the
  running-reference variant is genuinely non-redundant with the fixed-reference
  one.

### 5. Shiryaev-Roberts — `srz_cur`, `srz_pkr`
- **SIGNAL** The Bayes-optimal (in average-detection-delay) quasi-stationary
  statistic: it *integrates* over all possible change points instead of
  maximising, so it accumulates weak, diffuse evidence that the max-type
  statistics discard. Kept in log space, it grows linearly in `t − τ` after a
  break of the assumed size (δ = 0.5).
- **FALSE SIGNAL** Because it integrates, it also accumulates a slow positive
  bias in the presence of any small systematic mismatch between the historical
  and online mean — including a benign calibration offset. It is the *least*
  robust channel to a wrong historical `μ`.
- **DISAMBIGUATOR** Page-Hinkley (which removes the online-mean offset by
  construction) and the `_pkr` shape. `srz_pkr` earns 1.7 % of total model gain;
  `srz_cur` earns essentially nothing, i.e. the peak is what matters.

### 6. Dyadic GLR / mixture-GLR — `glz_*` (full shape), `glz_mix`, `gle_*`
- **SIGNAL** No assumed shift magnitude: at each `t` it maximises
  `(Σ last L)²/(2L)` over dyadic `L`, which is a *scale-free* mean-shift
  detector. It is the fastest channel to react to a large break and the only one
  that simultaneously reports "how far back the change looks like it started".
  `gle_*` is the same detector on the AR-residual stream.
- **FALSE SIGNAL** The maximum over many candidate lengths is exactly a
  multiple-testing statistic: under the null it is systematically inflated, and
  its inflation grows with `t` and with the tail weight of the series. Any use
  of a fixed threshold here would be badly miscalibrated.
- **DISAMBIGUATOR** The historical-null replay, which reproduces *exactly the
  same* multiple-testing inflation on break-free data for the same series — this
  is the single clearest example of why the replay calibration is required
  rather than nice-to-have. Plus `glz_per`/`glz_tsp` for transient vs permanent.
  `gle_pkr` is the **highest-gain feature in the entire 211-column model after
  `t_online`**, and `glz_pk` is fourth.

### 7. EWMA bank — `ew8_z`, `ew32_z`, `ew128_z`, `ew32_pkrel`, `ew32_slp`, `ewv32_z`
- **SIGNAL** A signed, bounded-memory view of the level at three half-lives
  (8/32/128), robust-z'd against the historical EWMA path so the scale is
  per-series. Short half-life reacts fast, long half-life confirms. `ewv32_z` is
  the EWMA variance-ratio channel: `log(EWMA[z²]/E_hist[z²])`, a signed
  volatility-regime meter.
- **FALSE SIGNAL** The short half-life is essentially a smoothed outlier
  detector; the long half-life lags a real break by ~128 points and is still
  contaminated by pre-break data for a long time after `τ`.
- **DISAMBIGUATOR** Cross-half-life agreement (short *and* long both displaced
  = persistent; short only = transient), plus `ew32_pkrel`.
- **HONEST VERDICT** This family was the weakest. `ew128_z` and `ew32_pkrel` earn
  small but non-zero gain; `ew8_z`, `ew32_z` and `ew32_slp` earn essentially
  nothing, because `m00_core`'s trailing-window bank at w=8…256 already spans
  the same information with better calibration. See FAILED_EXPERIMENTS.

### 8. Cross-channel shape — `xc_max_cur`, `xc_max_pk`, `xc_n_hot`
- **SIGNAL** Do independent detectors *agree*? Several channels simultaneously
  extreme is much stronger evidence than one.
- **FALSE SIGNAL** The channels are far from independent (they share the same
  underlying stream), so agreement is partly mechanical.
- **DISAMBIGUATOR** The individual channels, which the model also has.
- **VERDICT** `xc_max_pk` is a top-10 feature (3.1 % gain); `xc_n_hot`
  (threshold count) is dead — a hard threshold throws away exactly the
  information the continuous max keeps.

---

## WHICH CHANNELS EARNED THEIR PLACE

`m01_seq` takes **48.7 % of total model gain with 28.4 % of the columns** in
RT-010T. The per-column ranking is unambiguous about *what kind* of column pays:

**Earned it (keep):**
| Column family | Gain share | Comment |
|---|---|---|
| `gle_pkr`, `glz_pk` | 3.86 %, 3.56 % | dyadic GLR peak — the single best sequential channel |
| `cz100_pkr`, `cz25_pkr`, `cz50_pk` | 3.51 %, 1.77 %, 2.19 % | raw CUSUM peaks; all three slacks pay separately |
| `ce50_pk`, `ce25_pkr`, `ce50_per` | 3.19 %, 2.09 %, 1.29 % | AR-residual CUSUM is *not* redundant with raw |
| `xc_max_pk` | 3.13 % | cross-channel peak agreement |
| `phd_pkr`, `phu_pkr` | 2.98 %, 2.41 % | running-reference Page-Hinkley, both directions |
| `sq50_pk`, `sq25_pkr`, `sq50_tsp`, `sq50_per` | 2.94 %, 1.18 %, 1.02 %, 0.61 % | variance CUSUM incl. shape |
| `srz_pkr` | 1.73 % | Shiryaev-Roberts peak |
| `*_tsp`, `*_per` (cz50, ce50, glz) | ~0.8–1.0 % each | the transient-vs-permanent discriminators |
| `*_up` / `*_dn` split (sq50, ce50, cz50) | 0.4–1.0 % each | direction genuinely matters |

**Did not earn it (drop candidates):**
- **All four `*_slp` columns and `ew32_slp`: exactly zero gain.** The local slope
  of a monotone-ish nonnegative path is nearly rank-equivalent to
  `current − decayed peak`, which the model already has, and it is the noisiest
  encoding of it.
- **`xc_n_hot`: zero gain.** Hard-threshold counting destroys the ordering that
  `xc_max_cur`/`xc_max_pk` preserve.
- **`*_cur` (calibrated current level) columns: near-zero gain across the board**
  (`cz50_cur` 0.0015 %, `glz_cur` 0.005 %, `ce50_cur` 0.006 %). This is the most
  interesting negative: the *level* of the sequential evidence is almost
  entirely subsumed by `m00_core`'s window bank. **What `m01_seq` contributes is
  the peak and the persistence, not the level.** Note the levels are *not*
  worthless in isolation — RT-010N (levels only) still beats the control by
  +0.00845 — they are worthless *given* `m00_core`.
- `cz50_rel`, `ce50_rel` ≈ 0 (though `sq50_rel`, `glz_rel` are non-zero);
  `ew8_z`, `ew32_z` ≈ 0.
- `glz_mix` (mixture-GLR) 0.008 % — dominated by the max-GLR.

A trimmed ~45-column version (dropping the 5 slope columns, `xc_n_hot`, the 8
weakest `*_cur`, `glz_mix`, `ew8_z`, `ew32_z`, `cz50_rel`, `ce50_rel`) would
almost certainly hold the gain at ~25 % lower build and training cost. **I have
deliberately not trimmed it in this session**, because dropping columns on the
basis of one fold's gain importance is a selection decision that belongs to
Agent 0 and needs its own paired control. Flagging it as the obvious next move.

---

## SHAPE ABLATION (the secondary falsification)

RT-010N restricts `m01_seq` to the 18 "level" columns (`*_cur`, `ew*_z`,
`ewv32_z`, `glz_mix`) and drops every peak / decayed-peak / time-since-peak /
persistence / slope / ratio column.

```
control  m00_core                          0.57384
levels   m00_core + m01_seq levels-only    0.58229   (+0.00845)
full     m00_core + m01_seq                0.58809   (+0.01425)
```

**The shape channels are worth +0.00580 TS-AUC on their own — 41 % of the
module's total contribution.** The secondary falsification (levels within 0.002
of full) is rejected decisively. This is the direct empirical confirmation of
the module's core thesis: what a sequential detector adds over a window bank is
the *history of its own path*, not its current reading.

---

## LEAKAGE RISKS

1. **Time-axis reductions.** Every one is a sequential accumulator
   (`cumsum`, `minimum.accumulate`, `maximum.accumulate`, `lfilter`, the numba
   SR loop). Any *data-dependent* block reduction — e.g. a stabilised
   block-wise cumulative log-sum-exp — would have broken bitwise prefix
   invariance because the block maximum changes when the series is truncated. I
   rejected that design for the Shiryaev-Roberts statistic for exactly this
   reason and used the exact sequential recursion instead. Verified at atol=0.0.
2. **The peak-window selector depends on `t` only** (`argmin |log(t+1) − log(w)|`
   over a fixed grid), never on any online value. If it had been chosen by the
   data it would have leaked.
3. **All nulls are built from `hist` only**; online values enter solely as
   `searchsorted` queries. All standardisation constants come from `ctx.hp` /
   `ctx.hist_tr`.
4. **Residual risk: none identified after the bitwise check.** The mixture-GLR
   bug (fixed) was the one real leak and it was caught by the harness, not by
   inspection — which is the argument for running the harness before every
   number.
5. **Non-leakage risk worth naming:** `t_online` is the highest-gain feature in
   the whole model. It is constant within a timestep so it cannot change any
   within-`t` ranking under TS-AUC, but it lets the model condition every
   `m01_seq` shape feature on how much online evidence exists — which is
   probably *why* the peak channels work so well (a peak surprise of 3 at t=20
   means something very different from the same value at t=800). If a future
   metric or ensembling step is not strictly within-`t`, this interaction needs
   re-examination.

**OOF ARTIFACT PATH** none — `sbr.pipeline.run` does not persist OOF for
`screen=True` runs by design. All four runs are recorded in
`research/RESULTS.csv` (RT-010C, RT-010T, RT-010A, RT-010N). Importance table saved at
`research/artifacts/agent05_m01_seq_importance.csv`.

**Note for Agent 0 (not acted on):** `research/RESULTS.csv` is now ragged —
`screen=True` runs append an extra `protocol` column that is not in the header,
so `pd.read_csv` fails with "Expected 26 fields, saw 27". Per protocol §1 I did
not edit the file or `pipeline.py`; flagging only.

---

## CONCLUSION: **KEEP**

+0.01425 screen fold-0 TS-AUC over an exact paired control, from 60 columns,
7.5 ms/series, with bitwise-verified causality on 12 series × 9 truncations.
The module is complementary rather than duplicative: alone it scores only
0.54392, but it takes 48.7 % of the gain when combined, and 41 % of its
contribution comes specifically from the path-shape channels that no
window-based module can produce.

Recommended follow-ups, in priority order:
1. Promote to the full 5-fold protocol (Agent 0's call) — the effect is ~7× the
   typical fold-to-fold noise scale seen in RT-000, but one screen fold is one
   screen fold.
2. Trim to ~45 columns (list above) and re-screen with a paired control; this is
   free performance if it holds.
3. Extend the peak-null grid downward (windows 8 and 16). The current smallest
   peak-null window is 32, so online steps `t < 23` are scored against a
   too-wide null. Given how much of the module's value is in `*_pk`, and that
   early online steps are the hardest and most valuable part of the trajectory,
   this is the most promising unexplored direction.
