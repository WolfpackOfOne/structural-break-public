# WAVE 5 — LIVE STATUS

**Branch `research/wave5-alpha`, parent `research/wave3-integration` @ `17bb5df`.**
macOS/arm64, Python 3.11.6, numpy 2.4.6, pandas 3.0.5, scipy 1.17.1,
scikit-learn 1.9.0, lightgbm 4.7.0, numba 0.67.0 — the freeze environment.
Machine: 10 cores, 16 GB. Feature cache shared read-only with the wave-3
worktree; new modules written locally.

Updated as runs land. `research/STATE_OF_RESEARCH_V5.md` is the final document.

---

## 0. THE EXTERNAL ANCHOR

| | |
|---|---|
| LB-001, Crunch public | **0.6268** |
| RT-600 development architecture, canonical partition | 0.62581 |
| transfer | flat to slightly positive |

Wave 5 is therefore an **alpha-discovery** project, not a validation-repair one.

## 1. BASELINES RE-ESTABLISHED ON THIS PLATFORM (same session, same folds)

| arm | id | TS-AUC | per fold | matches |
|---|---|---|---|---|
| A single control | `RT-300` | **0.61605** | 0.62903 / 0.61061 / 0.62688 / 0.61223 / 0.60152 | V4 §3 exactly |
| B seven seed clones | `RT-421` | **0.62164** | 0.63817 / 0.61667 / 0.63020 / 0.61437 / 0.60879 | V4 §5 exactly |
| C seven specialists (**the RT-600 architecture**) | `RT-420` | **0.62581** | 0.63828 / 0.62040 / 0.63393 / 0.61751 / 0.61894 | V4 §4 exactly |

Decomposition reproduced independently: bagging **+0.00559**, specialisation
**+0.00417**. Paired series bootstrap of specialists − seed clones:
**+0.00409**, 95% CI **[+0.00199, +0.00614]**, 200/200 replicates positive —
the same three digits W4-E1 reported.

## 2. W5-E1 — SPECIALIST / BAGGING MIXTURE: **REJECTED**

Pre-registered grid, four values, not enlarged.

| λ | score = λ·S + (1−λ)·B | Δ vs λ=1 | folds better | `RT-100R` blend weight |
|---|---|---|---|---|
| 1.00 | **0.62581** | — | — | 14.3% |
| 0.90 | 0.62577 | −0.00004 | 3/5 | 22.9% |
| 0.80 | 0.62564 | −0.00017 | 3/5 | 31.4% |
| 0.70 | 0.62541 | −0.00040 | 1/5 | 40.0% |

Bootstrap on the best λ<1: **−0.00003, CI [−0.00026, +0.00018], 36% positive.**

**The delta is monotone decreasing in bagging weight.** This is stronger than
W4-E6, which rejected only the 13-way union and explained it by that
composition handing the champion configuration 54% of the blend weight. The
mixture reaches that weight nowhere in the grid and still never helps: the
result is not a threshold effect at 54%, it is that **any** admixture of
seed-clone mass into the specialist blend is neutral-to-harmful. The tight CI
makes this a clean null rather than an underpowered one. **Equal weighting over
the seven specialists stands, and now stands on two independent tests.**

## 3. W5-D1 — WHERE THE METRIC'S WEIGHT ACTUALLY SITS

The official weight is `n_pos(t)·n_neg(t)` per timestep, and it is **not**
concentrated early:

| online index t | share of total pair weight | mean series alive |
|---|---|---|
| 0–10 | **0.2%** | 8000 |
| 10–25 | 1.0% | 7923 |
| 25–50 | 2.7% | 7750 |
| 50–100 | 7.9% | 7454 |
| 100–200 | 19.5% | 6850 |
| 200–400 | **36.6%** | 5647 |
| 400–700 | 27.7% | 3636 |
| 700–1000 | 4.3% | 1211 |

25% of the weight is at t ≤ 168, 50% at t ≤ 293, 90% at t ≤ 600.

By post-break age, on the positive side of each pair:

| age | share of positive rows | share of pair weight |
|---|---|---|
| 0–5 | 1.9% | **2.9%** |
| 5–10 | 1.9% | 2.8% |
| 10–20 | 3.6% | 5.3% |
| 20–50 | 10.0% | 13.9% |
| 50–100 | 14.4% | 18.4% |
| 100+ | 68.3% | **56.7%** |

**This corrects a premise the wave-5 brief states twice.** "We care enormously
about early evidence because real-time TS-AUC weights every timestep" is true
about timesteps and false about *weight*: ages 0–20 carry **11%** of the pair
weight and age 100+ carries **57%**. Young-break detection is worth roughly a
fifth of mature-break ranking under this metric. It also retrospectively
explains `m09_back`: a module that helped 100+ and hurt 0–20 produced a positive
aggregate, exactly as this weighting predicts, and was still correctly rejected
— on its seed-clone control, not on its age profile.

## 4. W5-D2 — WHAT FOOLS THE RT-600 ARCHITECTURE

Severity = each no-break series' mean within-timestep percentile rank under the
specialist ensemble. 4,025 no-break dev series; mean rank 0.4731 for negatives
against 0.5387 for positives.

Top 1% hardest negatives (n=40, mean rank **0.9221** against 0.0515 for the
easiest 1%) — descriptor gap against all negatives, in IQR units:

| descriptor | gap |
|---|---|
| `tail_rate_online` | **+1.89** |
| `shock_max_absz` | **+1.50** |
| `burst_ratio` | **+1.38** |
| `level_shift_end` | +0.54 |
| `tail_kurt_hist` | +0.41 |

Heuristic mechanism mix, top 1% against the base rate over all negatives:
heavy-tail/outlier **40.0% vs 25.9% (+14.1pp)**, variance burst 25.0% vs 20.8%
(+4.2pp), and every other mechanism at or below its base rate — local trend
−8.3pp, spectral −4.8pp, persistent displacement −2.8pp, dependence −2.4pp.

**Answer: the champion's false positives are dominated by no-break series whose
online segment carries more extreme values than their own history predicts.**
Tail occupancy, not trend, not dependence, not spectrum.

Two honest caveats. The taxonomy is a heuristic argmax over standardised
descriptors and is *labelled as such*; the `transient_shock` bucket takes 0% of
assignments because its two descriptors are near-collinear with the heavy-tail
pair, so the taxonomy separates those two mechanisms poorly and the honest
reading is "outlier/tail-driven excursions, transient or otherwise" as one
class. And `level_shift_end` at +0.54 says a minority of hard negatives really
do end displaced from their historical centre — those are not model errors so
much as unlabelled-looking series.

## 5. W5-D3 — AGE PROFILE OF THE THREE ARMS

| age | A single | B seed clones | S specialists | S−B | S−A | n_pos |
|---|---|---|---|---|---|---|
| 0–5 | 0.51077 | 0.51283 | 0.51517 | +0.00234 | +0.00440 | 19,675 |
| 5–10 | 0.52601 | 0.53092 | 0.53408 | +0.00316 | +0.00806 | 19,212 |
| 10–20 | 0.53870 | 0.54573 | 0.54880 | +0.00306 | +0.01009 | 37,061 |
| 20–50 | 0.56925 | 0.57364 | 0.57702 | +0.00338 | +0.00777 | 102,861 |
| 50–100 | 0.59597 | 0.60019 | 0.60516 | +0.00497 | +0.00919 | 148,574 |
| 100+ | 0.65086 | 0.65724 | 0.66151 | +0.00427 | +0.01065 | 705,859 |

Specialisation beats bagging at **every** age, +0.0023 to +0.0050, with the
largest gains at 50–100 and 100+ — which is also where the weight is. Young
breaks sit near chance (0.515 at age 0–5) for all three arms.

## 6. CAUSALITY GATES ON THE NEW MODULES

All at `atol=0.0`, bitwise, on 7 series spanning `n_online` 10 → 914 and
including **both** length-10 series in the dataset, with prefix cuts
(1, 3, 10, 37, 113):

| module | cols | causality | ms/series | full-store build |
|---|---|---|---|---|
| `m11_focus` | 76 | **PASS** | 32.4 | 264 s |
| `m12_rdep` | 57 | **PASS** | 64.4 | 436 s |
| `m10_persist` | 42 | **PASS** | 70.2 | ~700 s |

No degenerate columns in any built cache (0 of 76, 0 of 57); overall NaN share
1.0% and 6.3%, which is the correct value for "not enough online points yet".

**The gate earned its keep twice.** The first `m12_rdep` sized its expanding
nulls by `n_online` — the single forbidden input in this competition — and the
check failed it on all 7 series before any score was taken. The first
`m11_focus` was an exact O(t) scan at 526 ms/series; the convex-hull functional
pruning that replaced it is **bit-identical on the maximum, its age and the
anchored statistic** (verified against a brute-force reference on 30 random
cases including mean shifts and heavy tails) and 16× faster.

A third near-miss is worth recording because it would not have been caught by
any test: the stage-C queue's readiness check tested the feature cache file's
size, and the driver preallocates it with `open_memmap(mode="w+")`, so the file
is full-size and mostly zeros from the first second. Two runs of RT-740 were
started on a 15%-filled cache and killed; neither reached the ledger or wrote an
OOF vector. **A number computed from that cache would have looked entirely
normal.**

## 7. W5-E7 — `m11_focus`, EXACT MAXIMISATION OVER τ: STAGE C **SURVIVES**

ABL protocol, 400k rows, 5 canonical folds, seed 0 — the identical arms wave 3
used for `m09_back`, so the two families are directly comparable.

| arm | id | TS-AUC | per fold |
|---|---|---|---|
| control, 7 modules | `RT-301` | 0.61257 | 0.62567 / 0.60719 / 0.62018 / 0.60506 / 0.60475 |
| **+ `m11_focus`** | `RT-730` | **0.61574** | 0.62806 / 0.61394 / 0.62226 / 0.60762 / 0.60684 |
| delta | | **+0.00317** | +0.00239 / +0.00675 / +0.00208 / +0.00256 / +0.00209 — **5/5** |

And the comparison that killed `m09_back`, the two-model deployable blend
against the same blend built from a seed clone carrying no information:

| blend | TS-AUC | vs control |
|---|---|---|
| `RT-301` + `RT-303` (seed clone) — **the bar** | 0.61753 | +0.00496 |
| `RT-301` + `RT-730` (`m11_focus`) | **0.61808** | +0.00551 |
| **candidate − seed clone** | | **+0.00055**, 4/5 folds |

**`m11_focus` clears the gate `m09_back` failed** — it is the first new feature
family in this project to blend better than its own seed clone. Side by side:

| | `m09_back` (wave 3) | `m11_focus` (wave 5) |
|---|---|---|
| standalone delta | +0.00156 | **+0.00317** |
| folds positive | 4/5 | **5/5** |
| blend vs seed clone | **−0.00126** | **+0.00055** |
| age 0–5 | −0.00092 | **+0.00308** |
| age 5–10 | −0.00035 | **+0.00465** |
| age 100+ | +0.00349 | +0.00443 |

**The mechanism did what it was built to do.** `m09_back` was built for late
breaks and helped only mature ones; `m11_focus` was built on the argument that
fixed windows dilute early evidence, and its largest relative gains are at ages
0–5 and 5–10 — the buckets `m09_back` damaged. There is a real dip at 20–50
(−0.00139) which is recorded, not smoothed over.

**But +0.00055 over the seed clone is a fifth of the +0.0030 promotion bar.**
Under §28's bands that is noise-to-exploratory, not a candidate. Stage D — the
CHAMP-protocol eight-member test `(S + RT-731)` against `(S + RT-401)`, which is
the comparison §4 actually binds on — is what decides, and a two-model ABL blend
is a weak proxy for it in both directions.

### 7a. `m11_focus` — ensemble contribution, not gain alone (§22)

| module | % of total gain | % of columns | gain per column |
|---|---|---|---|
| `m07_bayes` | 23.11 | 8.68 | 2.66 |
| `m01_seq` | 14.46 | 10.42 | 1.39 |
| `m03_dyn` | 13.68 | 10.42 | 1.31 |
| `m00_core` | 13.05 | 26.22 | 0.50 |
| `m04_resid` | 12.39 | 10.42 | 1.19 |
| **`m11_focus`** | **11.52** | **13.19** | **0.87** |
| `m02_dist` | 8.07 | 10.24 | 0.79 |
| `m06_loc` | 3.71 | 10.42 | 0.36 |

For scale: `m09_back` took **1.25%** of gain from 9.3% of columns. `m11_focus`
takes nearly ten times the share, ahead of two production modules per column,
and its best column ranks **8th of 576**. The booster is genuinely using it.

**Within-timestep rank correlation with the control: `m11_focus` 0.7814,
seed clone 0.7846.** A seed change *still* decorrelates marginally more than 76
columns of new statistics. Wave 3's lesson holds exactly, and it is why the
blend-versus-clone comparison and not the correlation is what decided this.

**A mechanistic caveat, stated because the same shape of finding demoted
`m09_back`.** The columns the booster leans on hardest are `*_pk` — the running
high-water mark of the calibrated maximised statistic — across all six channels,
then `*_agefrac` (the inferred change-point age as a fraction of elapsed time),
then `*_anc_z` (the τ=0 **anchored** statistic, which is not maximised at all).
The raw `*_z` maximised statistic and the `*_gain` contrast — "what maximisation
over τ actually bought over not maximising" — appear lower. Gain importance is
precisely the metric §22 says not to trust alone, `_pk` is a monotone function of
`_z` so the maximisation is upstream of it either way, and the `_anc_z` and
`_gain` columns are both used. But the honest reading is that **part of this
module's contribution is peak-and-age geometry rather than the maximisation
itself**, and the clean way to settle it is a within-module ablation
(`_z`/`_gain` columns only, against `_pk`/`_age` columns only) which is listed
as a next experiment rather than run here, because it cannot change the
promotion decision and would spend degrees of freedom that the promotion
decision needs.

## 8. W5-E9a — TS-AUC-SHAPED PAIR WEIGHTING: **REJECTED**

The hypothesis was clean and, I still think, correct about the arithmetic: the
official metric pools concordant pairs over `Σ_t n_pos(t)·n_neg(t)`, so every
within-timestep (positive, negative) pair counts equally, whereas the incumbent
`pairwise_t` objective draws a fixed `m_neg = 8` negatives per positive row and
therefore makes every positive ROW count equally instead. D1 showed the
mis-weighting is not small: timesteps with 7,000 series alive are weighted the
same as timesteps with 1,200.

**The dispatch is verified byte-identical first.** `RT-702` is the incumbent
objective routed through the wave-5 hook, and it reproduces `RT-413`'s ledger row
to five decimals on **all five folds** — 0.63185 / 0.61326 / 0.62455 / 0.60031 /
0.60408. So the arms differ only in the loss.

| arm | id | TS-AUC | per fold |
|---|---|---|---|
| incumbent `pairwise_t` | `RT-702` | **0.61481** | 0.63185 / 0.61326 / 0.62455 / 0.60031 / 0.60408 |
| `pairwise_w`, pairs weighted by `n_neg(t)` | `RT-700` | 0.61334 | 0.63270 / 0.60824 / 0.62348 / 0.60039 / 0.60191 |
| delta | | **−0.00147** | +0.00085 / −0.00502 / −0.00107 / +0.00008 / −0.00217 — **2/5** |

**Aligning the training objective with the metric's own pair weighting makes the
model worse.** The most likely mechanism, stated as a hypothesis and not a
finding: the weighting concentrates gradient on the timesteps with the most
alive series, which D1 places at t ≈ 200–700 — but those are also the timesteps
whose ranking is *easiest* and already best served, while the estimator variance
of a positive row's gradient rises with its weight. Matching the evaluation
weighting is not the same as spending training capacity well, and this is a case
where the two come apart.

The incumbent's flat `m_neg` is, in effect, an implicit importance weighting
towards sparse late timesteps, and it earns its keep.

## 9. W5-D4 — PERFORMANCE BY BREAK FAMILY (**heuristic taxonomy, labelled as such**)

The wave-1 taxonomy built a proper artifact with a placebo-null threshold, but
`research/artifacts/break_taxonomy.parquet` did not survive the container that
produced it — its own report says so. This is a lighter reconstruction of its
`lead_family` column: an argmax over standardised family effect sizes, no
threshold, POST vs PRE where ≥60 pre-break online points exist and POST vs HIST
otherwise. No-break series get a placebo cut from the break series' relative-τ
distribution, so the shares have a matched null. **It uses τ and the post-break
segment and is a diagnostic only.**

| family | break % | placebo % | **excess** |
|---|---|---|---|
| location | 16.5 | 20.0 | −3.5 |
| **scale** | 32.6 | 25.4 | **+7.1** |
| tails | 0.1 | 0.1 | −0.0 |
| dependence | 0.3 | 0.1 | +0.1 |
| trend | 0.2 | 0.1 | +0.0 |
| spectral | 50.4 | 54.2 | −3.8 |

**Scale is the only family with a positive excess over the placebo null**, which
reproduces the wave-1 taxonomy's finding #3 independently. Read the *excess*
column, not the share: the argmax is dominated by "spectral" because my
high-frequency-energy statistic is the noisiest of the six, and a bucket with a
**negative** excess is one the classifier is filling with noise. That is a real
limitation of this reconstruction and it is why nothing is concluded from the
shares alone.

| lead family | n_pos | A single | B seed clones | S specialists | S−B | S−A |
|---|---|---|---|---|---|---|
| location | 154,551 | 0.55642 | 0.55996 | 0.56431 | +0.00435 | +0.00789 |
| **scale** | 373,783 | 0.68608 | 0.69360 | **0.69702** | +0.00342 | +0.01095 |
| dependence | 1,206 | 0.61099 | 0.60639 | 0.63417 | +0.02778 | +0.02317 |
| spectral | 490,741 | 0.58313 | 0.58811 | 0.59236 | +0.00424 | +0.00922 |

**The champion is a scale detector.** On scale-led breaks it reaches 0.697; on
location-led breaks 0.564, barely above the 0.4998 that the wave-1 forensics
measured for raw mean-shift separability. The dependence row is 1,206 rows and
its +0.0278 should not be read as anything but small-sample noise.

Specialisation beats bagging on **every** family, +0.0034 to +0.0044 on the three
populated ones — the same margin as the aggregate, so the specialist advantage is
not concentrated in one break type.

## 10. W5-E4/E5/E6 — `m12_rdep`: STAGE C **SURVIVES**, AND IT IS THE STRONGEST BLOCK

Same ABL arms as everything else.

| arm | id | TS-AUC | per fold |
|---|---|---|---|
| control | `RT-301` | 0.61257 | 0.62567 / 0.60719 / 0.62018 / 0.60506 / 0.60475 |
| **+ `m12_rdep`** | `RT-750` | **0.61736** | 0.62882 / 0.61509 / 0.62561 / 0.61390 / 0.60337 |
| delta | | **+0.00479** | +0.00315 / +0.00790 / +0.00543 / +0.00884 / −0.00138 — **4/5** |

| blend | TS-AUC | vs control | **vs seed clone** |
|---|---|---|---|
| `RT-301` + `RT-303` (seed clone) | 0.61753 | +0.00496 | — |
| `RT-301` + `RT-730` (`m11_focus`) | 0.61808 | +0.00551 | **+0.00055**, 4/5 |
| `RT-301` + `RT-740` (`m10_persist`) | 0.61713 | +0.00456 | **−0.00041**, 2/5 ✗ |
| **`RT-301` + `RT-750` (`m12_rdep`)** | **0.61894** | **+0.00637** | **+0.00141**, 4/5 |

**`m12_rdep` is the largest genuinely-new-information result this project has
produced** — 2.5× `m11_focus`'s margin over the seed clone, and against a
standard that has already rejected three candidates.

**The two survivors are complementary in age, which is the interesting part:**

| age | Δ `m11_focus` | Δ `m12_rdep` |
|---|---|---|
| 0–5 | **+0.00308** | −0.00228 |
| 5–10 | **+0.00465** | +0.00042 |
| 10–20 | **+0.00255** | −0.00039 |
| 20–50 | −0.00139 | −0.00054 |
| 50–100 | +0.00288 | +0.00304 |
| 100+ | +0.00443 | **+0.00778** |

`m11_focus` is a young-break module — maximising over τ recovers evidence that a
fixed window dilutes, exactly as designed. `m12_rdep` is a mature-break module,
and D1 says 57% of the metric's pair weight lives at age 100+, which is why it
scores higher despite doing nothing for early detection. **This is what makes
W5-E10, the union, worth its run rather than a formality.**

## 11. W5-E9b — WEIGHTED SQUARED HINGE: **REJECTED**

| arm | id | TS-AUC | delta |
|---|---|---|---|
| incumbent `pairwise_t` | `RT-702` | 0.61481 | — |
| `pairwise_h` weighted squared hinge | `RT-701` | 0.61005 | **−0.00476** |

Rejected. The gradient scale was matched to the logistic at `d = 0` precisely so
this would measure the loss shape and not the step size, and the loss shape is
worse. A hinge stops pushing a pair once it is ranked correctly by a margin; for
a metric that is *entirely* pairwise ranking that sounded right, and the data
disagrees. Both W5-E9 variants fail, and the incumbent `pairwise_t` stream stands
unchanged.

### 10a. Which of W5-E4 / E5 / E6 actually carried `m12_rdep`?

The module bundles three pre-registered sub-blocks, so the ledger can separate
them. Within `RT-750`'s `m12_rdep` gain:

| sub-block | % of module gain | columns | gain per column |
|---|---|---|---|
| **W5-E5 residual CUSUM / CUSUMSQ paths** | **72.8%** | 20 | **3.64** |
| W5-E6 dependence LR (variance profiled out) | 12.9% | 10 | 1.29 |
| W5-E4b robust two-sample scale | 6.9% | 9 | 0.77 |
| W5-E4 residual distribution distances | **7.5%** | 18 | 0.42 |

`m12_rdep` as a whole takes 12.17% of `RT-750`'s total gain from 10.23% of its
columns. Its E5 sub-block's gain-per-column of **3.64** is higher than every
production module except `m07_bayes` (2.45).

**This answers three of the brief's questions separately, and they do not have
the same answer:**

* **Does residual CUSUMSQ add new information? Yes — it is the whole result.**
  `RT-907`, the CUSUM/CUSUMSQ-on-regression-residuals translation, is the one
  2025 idea that paid.
* **Does a dependence likelihood ratio add new information?** Modestly. 12.9%
  from 10 columns; `dl_exp_phi` is the module's 6th-best column.
* **Do robust distribution distances add new information? Least of the three.**
  18 columns for 7.5% of the gain — the lowest gain-per-column in the module.
  `m02_dist` already computes this family on the raw PIT and residualising it
  buys little, which is the outcome the pre-registration named as its own
  falsification ("fully redundant with `m02_dist` under leave-one-block-out").

**A cross-cutting finding.** In *both* new modules the columns the booster leans
on are the **calibrated path geometry** — running peak (`*_pk`), persistence
(`*_per`), time-since-peak (`*_tsp`) — not the instantaneous statistic. The top
four `m12_rdep` columns are `rcdn_pk`, `rqup_pk`, `rqdn_pk`, `rcup_pk`; the top
six `m11_focus` columns are all `*_pk`. Whatever the underlying detector, what
survives into the model is *how high the evidence has ever been and how long it
stayed there*. That is a design lesson for the next module, and it is consistent
with `m01_seq` — the module whose entire thesis is detector-path shape — being
the second-largest gain contributor in the bank.

## 12. W5-E10 — THE UNION OF ALL THREE BLOCKS: **REJECTED**

| arm | id | standalone | Δ vs `RT-301` | folds | blend vs **seed clone** | folds |
|---|---|---|---|---|---|---|
| `m11_focus` | `RT-730` | 0.61574 | +0.00317 | **5/5** | **+0.00055** | 4/5 |
| `m10_persist` | `RT-740` | 0.61459 | +0.00202 | 3/5 | −0.00041 ✗ | 2/5 |
| **`m12_rdep`** | `RT-750` | **0.61736** | **+0.00479** | 4/5 | **+0.00141** | 4/5 |
| **union of all three** | `RT-760` | 0.61468 | +0.00211 | 2/5 | +0.00093 | 4/5 |

**175 new columns do worse than 57.** The union's standalone delta (+0.00211) is
**less than half** `m12_rdep`'s alone (+0.00479), and its blend margin over the
seed clone (+0.00093) is below `m12_rdep`'s (+0.00141). Its age profile is worse
than every single block at every bucket under 50:

| age | `m11_focus` | `m12_rdep` | **union** |
|---|---|---|---|
| 0–5 | +0.00308 | −0.00228 | **−0.00424** |
| 5–10 | +0.00465 | +0.00042 | **−0.00287** |
| 10–20 | +0.00255 | −0.00039 | **−0.00572** |
| 100+ | +0.00443 | +0.00778 | +0.00492 |

**The blocks anti-stack, exactly as W4-E6's ensemble members did**, and the
mechanism is the same shape: a fixed budget spread over more things. Here it is
`feature_fraction = 0.5` over 675 columns instead of 557 — every tree sees a
smaller fraction of the productive block — plus genuine overlap, since
`m11_focus`'s residual channels and `m12_rdep`'s residual CUSUM paths are both
reading AR-residual path geometry.

**What this does NOT license.** Pairwise unions, block subsets, per-block
`feature_fraction`, or any best-of composition. That is the lattice the
pre-registration excluded, and searching it after a negative result is how a null
becomes a false positive. The mechanism above is an explanation, not a follow-up.

## 13. DEPLOYMENT COST OF THE NEW MODULES

Measured on 40 random series, splitting the one-off per-series null construction
from the marginal per-point work.

| module | setup, ms/series | marginal, ms/point | amortised ms/pt | engine total | vs 15 h budget |
|---|---|---|---|---|---|
| incumbent 7 boosters | — | — | — | 1.734 | 8.0 h combined |
| + `m11_focus` | 29.5 | 0.0111 | +0.0696 | 1.804 | +4.0% |
| + `m10_persist` | 52.8 | 0.0036 | +0.1084 | 1.842 | +6.3% |
| + `m12_rdep` | 78.1 | −0.0007 | +0.1544 | 1.888 | +8.9% |

**None of them threatens the budget.** The cost is almost entirely the per-series
historical null, paid once at `fit_historical`, not per point — `m12_rdep`'s
marginal per-point cost is indistinguishable from zero. At 1.888 ms/pt the
projection is ~8.7 h against 15 h. Runtime is not the binding constraint on any
of these; the seed-clone control is.

**But a promoted module also needs a bitwise streaming twin** (`s_m12_rdep.py`
alongside the seven existing ones, plus a manifest change and an artifact
rebuild). That engineering cost is real and is the reason the promotion bar
matters more here than the runtime one.

## 14. THE NEXT THREE EXPERIMENTS, AND WHY THESE THREE

Ranked by evidence produced in this wave, not by appeal.

### W6-A — an expanded residual-PATH module (highest expected value)

W5-E5's sub-block delivered **72.8% of `m12_rdep`'s gain from 20 of its 57
columns**, at a gain-per-column of **3.64** — higher than every production module
except `m07_bayes`. And it is barely explored: those 20 columns are **one** drift
constant (`k = 0.5`), **one** residual representation (the shared context's
AR(2)), two statistics (CUSUM, CUSUMSQ) and two signs.

`m04_resid` already ships **eight** residual representations — `ar1`, `ar2`,
`ar5`, ridge `arR`, Huber `arH`, EWMA `vol`, winsorised `volM`, GARCH `volG`,
combined `cmb` — every one fitted on history only and applied causally forward,
and **none of them has a CUSUM path**. `m06_loc`'s forensics also measured AR(6)
residuals as ~0.9 AUC points better than AR(2) for scale localisation, and this
module used AR(2).

Cross {3–4 residual representations} × {2–3 drift constants} × {CUSUM, CUSUMSQ}
with the identical calibrated path geometry. This is a known-productive vein
being mined at one point.

### W6-B — `m11_focus` ablation, then a slim version

The module's gain concentrates in `*_pk`, `*_agefrac` and `*_anc_z`, not in the
maximised statistic `*_z` or the `*_gain` contrast that was the hypothesis. Train
two arms — `_z`+`_gain` columns only, and `_pk`+`_age` columns only — and find
out which half carries it. W5-E10 established that **column count is a real
cost** (175 columns did less than half of what 57 did), so a 20-column
`m11_focus` that keeps the contribution would be strictly better than the
76-column one, and might survive the union test that the full module failed.

### W6-C — calibration anchor placement (cheapest, and never once measured)

`SmoothTimeCDFCal` uses 12 **log-spaced** anchors, 256-point grids and
`min_n = 400`. The architecture freeze says plainly: *"Anchors, grid size and
`min_n` are frozen and were never tuned."* Meanwhile W4-E1 measured the
calibration family as worth **+0.00267** on the specialist arm — one of the
largest single effects in the project — and W5-D1 measured where the metric's
weight actually is: **50% between t = 168 and t = 451**, a narrow band that
log-spacing deliberately under-resolves, because log anchors crowd near t = 1
where **0.2%** of the pair weight lives.

This needs **no training at all** — it recomputes from OOF vectors already on
disk, which is why it belongs first in wall-clock order even though W6-A has the
higher ceiling. It must be pre-registered with a tiny structured grid (say three
anchor schemes: log, uniform-in-weight, and hybrid) and no continuous search,
because this is exactly the surface where tuning against OOF would be easiest and
most damaging.

## 15. STAGE D — AND A PROTOCOL REVERSAL THAT CHANGES HOW STAGE C SHOULD BE READ

`RT-731` is the champion configuration — CHAMP protocol, 1,000,000 rows, seed 0 —
with `m11_focus` added and nothing else changed. Its ABL twin `RT-730` was
**+0.00317 on 5/5 folds**.

| arm | id | protocol | TS-AUC | vs its own control |
|---|---|---|---|---|
| control | `RT-301` | ABL, 400k | 0.61257 | — |
| + `m11_focus` | `RT-730` | ABL, 400k | 0.61574 | **+0.00317, 5/5** |
| control | `RT-300` | CHAMP, 1M | 0.61605 | — |
| + `m11_focus` | `RT-731` | CHAMP, 1M | 0.61483 | **−0.00123, 2/5** |

per fold: −0.00212 / +0.00376 / −0.00262 / **−0.01087** / +0.00571

**The sign flips with the protocol.** A block that is +0.00317 on 5/5 folds at
400k training rows is −0.00123 on 2/5 at 1,000,000 rows, same folds, same seed,
same columns.

**This is the most methodologically important result in wave 5, and it indicts
the stage-C design — mine and wave 3's.** Two plausible mechanisms, both
consistent with what else is known:

1. **Capacity substitution.** At 400k rows the incumbent 500-column bank is not
   yet saturated, so 76 columns of a genuinely different statistic buy real
   accuracy. At 1M rows the bank has already extracted what those columns carry,
   and they add nothing to substitute for.
2. **Column dilution.** `feature_fraction = 0.5` samples 288 of 576 columns per
   tree instead of 250 of 500, so every tree sees a smaller share of the columns
   that were already working. This is the same mechanism that sank W5-E10's
   union, at a smaller dose.

**What follows for the record.** `m09_back` was rejected on ABL evidence and
`m11_focus` was promoted to stage D on ABL evidence; this says an ABL stage-C
result is a statement about the 400k-row regime and not about the deployed one.
The stage gates in `research/WAVE5_PREREG.md` §5 should have required the
protocol the champion actually uses before any promotion claim, and the ABL rung
should be read as a **screen for the absence of an effect, not as evidence for
one**. Wave 4 already documented the neighbouring version of this trap — forcing
specialist streams onto ABL collapses them toward the champion — and I did not
extend the lesson far enough.

## 16. STAGE D VERDICT — A NEW INTERNAL MAXIMUM, AND IT DOES NOT CLEAR THE BAR

CHAMP protocol, seed 0, seven production modules plus the block. Matched control
`RT-401` = the champion configuration at seed 1, differing from `RT-300` by the
seed and nothing else.

**Standalone streams:**

| stream | id | TS-AUC | vs `RT-300` | folds |
|---|---|---|---|---|
| champion | `RT-300` | 0.61605 | — | — |
| seed clone | `RT-401` | 0.61661 | +0.00056 | — |
| **+ `m12_rdep`** | `RT-751` | **0.61726** | **+0.00121** | 3/5 |
| + `m11_focus` | `RT-731` | 0.61483 | −0.00123 | 2/5 |

**Eight-member compositions:**

| composition | TS-AUC | vs `S` | folds |
|---|---|---|---|
| `S` — the RT-600 architecture | 0.62581 | — | — |
| **`S` + `RT-751`** | **0.62629** | **+0.00048** | 4/5 |
| `S` + `RT-731` | 0.62596 | +0.00015 | 4/5 |
| `S` + `RT-401` (seed clone, no information) | 0.62584 | +0.00003 | 4/5 |

**BINDING COMPARISON: `(S + RT-751) − (S + RT-401)` = +0.00046 on 3/5 folds.**
Against a §4 bar of +0.0030 on ≥4/5. **REJECTED as an eighth member.**

**0.62629 is a new internal maximum for this project** — +0.00048 over the
architecture that scored 0.6268 externally. It is also, on its own evidence, not
worth submitting: a fifteenth of the promotion bar, on three folds of five.

Two things this does establish:

* **`m12_rdep` does not reverse with protocol, and `m11_focus` does.** At CHAMP
  the residual-path block is +0.00121 standalone while the max-over-τ block is
  −0.00123. Whatever `m11_focus` was buying at 400k rows, the bank already has at
  1M; `m12_rdep`'s residual CUSUM/CUSUMSQ paths survive the transition.
* **`RT-751` is genuinely more decorrelated than a seed clone** — within-timestep
  rank correlation with `S` of **0.8133** against the clone's **0.8509**. This is
  the first stream in the project to beat a seed clone on that measure. Per wave
  3's lesson it is **not** a promotion credential, and it is reported here
  precisely because the blend delta it accompanies is +0.00046.

The eighth-member framing cannot resolve this block either way — an eighth
exchangeable member is worth +0.00003, so there is almost no room in the
composition for a new member to demonstrate anything. **W5-E11, now running,
rebuilds all seven streams with `m12_rdep` and is the test that decides.**

## 17. THE DEPLOYMENT PATH FOR `m12_rdep` IS BUILT AND PROVEN

A promoted feature module cannot ship without a **bitwise streaming twin** — the
Crunch runner is series-sequential and single-pass, and `ProductionModel` hard-
errors if the engine's columns do not hash to what the model was trained on.
The seven existing twins are 320–1,076 lines each; this was the real blocker on
any Wave-5 submission, so it was built while the architecture rebuild ran.

`src/sbr/stream/s_m12_rdep.py` + `tests/test_stream_parity_m12_rdep.py`:

| | |
|---|---|
| parity vs the batch module | **0 mismatches over 423,111 values**, 14 real series, `atol = 0` |
| edge cases covered | `n_online = 1`, both length-10 series, short histories, outlier / scale / dependence / heavy-tail |
| throughput | **225 µs/observation** (415 before tabulating the nulls) |
| engine projection | 1.734 → **1.959 ms/pt**, ~**9.0 h** of a 15 h budget |
| tests | 36 passing, every fast path pinned by a differential fuzz test |

**Two parity traps, both real, both caught by the test rather than by reasoning:**

1. **The AR-sigma round trip.** Batch computes `e = ctx.ar_online / hp.ar_sigma`
   where `ar_online` was already multiplied by `ar_sigma`. That is not the
   identity in floating point, so recomputing the AR filter in the twin differs
   in the last ulp. Reading `ctx.tr["res_mean"]` — which `build_transforms`
   defines by exactly that expression on exactly that input — is what makes it
   exact. `StreamCtx`'s own docstring flags the same trap.

2. **Reduction width.** Batch reduces a `(bins, n)` occupancy array along
   **axis 0** — numpy's strided path. A `(bins, 1)` array is contiguous and
   takes the **pairwise** path instead, and the two disagree in the last ulp on
   **97 of 300** random columns. Widths ≥ 2 all agree, so one padding column
   restores it. The first parity run failed on exactly this, at 4×10⁻¹⁶, in one
   cell of one column. It is now asserted in
   `test_reduction_width_invariance` rather than described in a comment.

**RT-600 is provably unaffected.** `MODULE_ORDER` is appended to, never
reordered, and `StreamEngine`'s default is now an explicit `PRODUCTION_MODULES`
tuple — the shipped seven — so a registered-but-unshipped module cannot leak
into a default engine. Verified three ways: the seven-module manifest still
hashes to `1646c3b9…cced` from the freeze; the first 500 columns of the
eight-module manifest are byte-identical to it; and **the actual shipped
`models/final10k_ensemble` loads through its hard manifest gate and streams
finite scores in [0, 1]**. `test_rt600_manifest_sha_is_unchanged` pins it.

### A pre-existing defect found on the way, and it is not mine

`tests/test_stream_engine_parity.py::test_engine_parity_real` **fails on
`research/wave3-integration` itself**, identically, before any wave-5 change:
1–2 cells per series out of ~200,000 differ between the batch and streaming
`m07_bayes`, in `bo_p_lt25_z`, `bo_lo_change_z` and `bo_ent`. The shipped
RT-600 artifact therefore has a known, tiny streaming drift in three
`m07_bayes` columns. At ≤2 cells per series it is very unlikely to move a
score, and it was not introduced or worsened here — but it is a real defect in
the deployed system and nothing in the wave-3 or wave-4 documents mentions it.
