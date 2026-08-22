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
