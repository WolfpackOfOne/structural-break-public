# RESEARCH DEGREES-OF-FREEDOM LEDGER

Cumulative count of everything that could have been chosen differently. It exists
so that "+0.0005" can be read against the number of chances we gave ourselves to
find a +0.0005.

## Cumulative counts

| | wave 1 | wave 2 (this wave) | total |
|---|---|---|---|
| logged experiments in `RESULTS.csv` | 78 | see below | — |
| distinct feature modules built | 8 (one rejected) | 0 new | 8 |
| model architectures / objectives tried | ~9 | 0 new | ~9 |
| hyperparameter studies | 1 sweep + per-stream hand tuning | 0 | — |
| ensemble compositions compared | ~12 (subsets, stacks, weightings) | 5 deployable calibrations + oracle | ~18 |
| fold partitions in existence | 1 canonical + 1 screen | +3 alternative (robustness only) | 6 |
| lockbox inspections | 2 | **0** | 2 |

## Attribution status of wave-1 rows

**66 of 78 wave-1 rows carry `git_sha = nogit`** and are therefore not
independently attributable to a code state. Wave 2 does not delete them — the
record is the record — but they may not be cited as confirmed results.

Directly re-executed in wave 2 from a clean checkout of
`research-checkpoint-20260818-1`:

| experiment | wave-1 | wave-2 | status |
|---|---|---|---|
| `RT-100` | 0.615103 | `RT-100R` 0.615103 | **reproduces, delta exactly 0.0** |
| `RT-123` | 0.614499 | `RT-123R` fold 0 0.63073 vs 0.63073 | reproduces (config matched) |

### Correction to an earlier wave-2 claim
`RT-121R`, `RT-122R` and `RT-124R` were initially described in this wave as
reproductions and their deltas (+0.0029, +0.0009, +0.0008) read as evidence that
the `nogit` rows are irreproducible. **That inference was wrong.** A file
rewrite failed silently and those three runs used reconstructed configurations,
not the wave-1 ones (`agent0_diversity.py` uses different `n_estimators`,
`learning_rate`, `min_data_in_leaf`, `lambda_l2` and `max_bin`). They are
legitimate diverse streams and are kept as such under the `R` suffix, but they
measure **configuration, not reproducibility**, and the ledger says so. The two
experiments whose configuration did match both reproduced exactly.

`RT-125R` additionally required a parameter change: LightGBM 4.7.0 rejects
`boosting=goss` alongside the pipeline's default bagging, which the wave-1
LightGBM accepted. That is a genuine environment-dependence finding.

## Promotion thresholds (tighten as this ledger grows)

| stage | bar |
|---|---|
| screen (features only) | +0.002 vs its own paired control |
| full dev (discovery) | positive on a **majority of folds** |
| promotion | paired bootstrap CI materially favourable, **or** a demonstrated ensemble delta under a **deployable** blend |
| architectural change | the above, plus stability across alt partitions and seeds |

A claimed gain below +0.001, after this many experiments, is "indistinguishable".

## Standing risks

1. **The champion feature bank was selected on the dev folds.** The lockbox put
   an upper bound of −0.0072 on what that selection cost. That bound is now two
   waves old and cannot be refreshed without spending the lockbox.
2. **The screen store is a subset of the dev folds**, so screen-driven choices
   are not independent of dev-fold scores.
3. **Wave 2 added no new held-out data.** Nested CV and alternative partitions
   bound the *variance* of our estimates; they cannot remove the *bias* from
   having chosen the architecture on this sample.
4. **The battery protocol is 400k training rows, not 1M.** Deltas within the
   battery are paired and valid; absolute levels are not champion scores.

---

# NEW AVENUES 2026 PILOTS -- PREREGISTERED 2026-08-24

Branch: `research/new-avenues-pilots-2026`.
Starting base: `origin/research/current` at
`6c37cb122f83012c39706f1f182ce8e3182461be`.
Pre-registration:
`research/reports/new_avenues_2026/PILOTS_01_03_PREREG.md`.

Degrees of freedom prospectively allocated before scoring:

| pilot | scored IDs | variants | fold screen | notes |
|---|---|---:|---|---|
| Specialist competence + failure manifolds | none | 0 | all dev folds descriptive | Diagnostic only; fixed fingerprint quintiles, `k in {2,3}`, one permutation seed. |
| Relay score-state | `RT-1200` | 1 | fold 0 | Fixed pickup/dropout quantiles, fixed charge/cool constants, no threshold tuning. |
| IM2+dwell bank | `RT-1201` | 1 | fold 0 | Fixed AR(2) residual-square channel, windows `{32,64,128}`, q90 band, fixed scalar mean. |
| Trajectory geometry | `RT-1202`, `RT-1203` | 1 primary + 1 shuffled-order control | fold 0 | Fixed windows `{16,64}`, 2,000 evenly spaced history references, non-overlap online neighbour rule, seed-0 history-order control. |
| Scale survival / coarse-graining exponent | `RT-1204`, `RT-1205` | 1 summary + 1 individual-scale control | fold 0 | Fixed AR(2) residual-square channel, scales `{1,2,4,8,16,32}`, two-sided empirical-null thresholds q05/q01, summary-control gap gate +0.0005. |

Continuation to 5 folds requires the preregistered marginal-vs-clone gates.

Pilot 5 result filed 2026-08-24:
`RT-1204` summary marginal_vs_clone `-0.000236`; `RT-1205` individual-control
marginal_vs_clone `-0.000707`; summary-control gap `+0.000471`.
Consumed the two preregistered fold-0 variants and stopped with no 5-fold
continuation because the primary `+0.0010` marginal gate failed and the
summary-control gap missed the `+0.0005` distinguishability floor.

Pilot 6 preregistered 2026-08-24 before scoring:

| pilot | scored IDs | variants | fold screen | notes |
|---|---|---:|---|---|
| Spectral impulsiveness contrast | `RT-1206`, `RT-1207` | 1 contrast + 1 plain-energy control | fold 0 | Fixed m03 dyadic Goertzel frequencies collapsed into four bands; segment length 32; adaptive half-prefix window; historical-null grid `{16,64,256}`; contrast-control gap gate +0.0005. |

Pilot 6 result filed 2026-08-24:
`RT-1206` contrast marginal_vs_clone `-0.000353`; `RT-1207` plain-energy
control marginal_vs_clone `+0.000189`; contrast-control gap `-0.000542`.
Consumed the two preregistered fold-0 variants and stopped with no 5-fold
continuation because the primary `+0.0010` marginal gate failed and the
plain-energy control exceeded the contrast arm.

Pilot 7 preregistered 2026-08-24 before scoring:

| pilot | scored IDs | variants | fold screen | notes |
|---|---|---:|---|---|
| Ordinal transition divergence + time irreversibility | `RT-1208`, `RT-1209` | 1 transition/asymmetry candidate + 1 entropy-only control | fold 0 | Fixed order-3 ordinal codes with `m03_dyn` tie convention; 6x6 transition KL; Ramsey-Rothman increment asymmetry; matched-count null grid `{16,64,256}`; entropy-control gap gate +0.0005. |

Pilot 7 result filed 2026-08-24:
`RT-1208` transition/asymmetry marginal_vs_clone `-0.000036`; `RT-1209`
entropy-control marginal_vs_clone `-0.000508`; candidate-control gap
`+0.000472`. Consumed the two preregistered fold-0 variants and stopped with
no 5-fold continuation because the primary `+0.0010` marginal gate failed and
the entropy-control gap missed the `+0.0005` distinguishability floor.

Pilot 10 preregistered 2026-08-24 before scoring:

| pilot | scored IDs | variants | fold screen | notes |
|---|---|---:|---|---|
| Joint size-duration rarity | `RT-1210`, `RT-1211` | 1 joint rarity candidate + 1 dwell-only rarity control | fold 0 | Fixed AR(2) residual-square channel, windows `{32,64,128}`, q90 band, historical endpoint joint null with `1/(2*n_endpoints)` floor, candidate-control gap gate +0.0005 and dominant-cell control gate. |

Pilot 10 result filed 2026-08-24:
`RT-1210` joint-rarity marginal_vs_clone `-0.000522`; `RT-1211`
dwell-control marginal_vs_clone `-0.000237`; candidate-control gap
`-0.000285`, and candidate-control dominant-cell AUC gap `-0.000896`.
Consumed the two preregistered fold-0 variants and stopped with no 5-fold
continuation because the primary `+0.0010` marginal gate failed and the
dwell-only control exceeded the joint-rarity arm.

Pilot 9(i) preregistered 2026-08-24 before scoring:

| pilot | scored IDs | variants | fold screen | notes |
|---|---|---:|---|---|
| Scalar historical-difficulty gate | `RT-1212`, `RT-1213` | 1 nested scalar candidate + 1 within-fold deranged control | fold 0 | Fixed 23 history-only fingerprints; target is RT-600 dominant-cell pair loss rate only; nested fold-pure scalar construction; derangement-control gap gate +0.0005. |

Pilot 9(i) result filed 2026-08-24:
`RT-1212` nested scalar marginal_vs_clone `+0.000164`; `RT-1213`
deranged-control marginal_vs_clone `+0.000269`; candidate-control gap
`-0.000105`. Consumed the two preregistered fold-0 variants and stopped with
no 5-fold continuation because the deranged control exceeded the real scalar
and the real scalar missed the primary `+0.0010` marginal gate.

---

# WAVE 3 — CLAUDE ALPHA LANE (`RT-3xx`)

Pre-registration is written **before** the run. Each block states the
hypothesis, the falsification condition, the matched control, and the number of
configurations tried, so the search degrees of freedom are counted honestly.

## Environment note — wave 3 runs on macOS, not the container

Wave 1 and wave 2 ran on Linux x86 (2 cores, 7 GB). Wave 3 runs on the user's
Mac (10 cores, 16 GB, arm64), same LightGBM 4.7.0. Floating-point summation
order in LightGBM's histogram construction is architecture- and thread-count
dependent, so **wave-3 absolute levels are not guaranteed bitwise-comparable to
the wave-1/2 ledger**. `RT-300` measures that gap once so every later wave-3
delta can be read against a wave-3 control rather than against a wave-2 number.

Portability changes made to shared code (no research semantics changed):
`SBR_ROOT`/`SBR_FEATURES` env overrides for the hardcoded `/home/claude/sb`
paths, and `load_all()` inside the feature-driver worker because macOS
multiprocessing uses `spawn`, not `fork`. The container defaults are preserved
as fallbacks.

## RT-300 — environment reproduction anchor  (PRE-REGISTERED 2026-08-20)

* **Hypothesis** The champion configuration `RT-100` reproduces on macOS/arm64
  from a clean checkout to within seed-scale noise (SD 0.0012).
* **Configuration** Exactly `RT-100`: 7 modules, 500 columns, CHAMP protocol
  (5 canonical folds, 1M training rows, seed 0, `n_estimators=600`,
  `learning_rate=0.05`, `num_leaves=63`, `min_data_in_leaf=300`,
  `feature_fraction=0.5`, `bagging_fraction=0.7`, `lambda_l2=5.0`,
  `max_bin=127`), `num_threads=2` to match the wave-2 run.
* **Falsification** |mean OOF − 0.615103| > 0.0050 (the partition-draw SD).
  A miss that large means the environment is not comparable and every wave-3
  delta must be re-based before anything can be promoted.
* **Search degrees of freedom** 1 configuration, 0 variants, 0 tuning. This is
  a reproduction, not a search.
* **Label** CONFIRMATION if it lands; otherwise the whole wave-3 lane is
  re-based on `RT-300` as its own control.

## RT-301 / RT-302 — backward-CUSUM late-break specialist  (PRE-REGISTERED 2026-08-20)

Tier-1 item 1 of the brief. **ID note:** the wave-3 brief describes an earlier
`RT-301`/`RT-302` pair (the `m08_chan` transformed detector bank). No such
module, ledger row or write-up exists anywhere in this repository or in any
branch's history, so those IDs are unallocated in the ledger and are used here.
The brief's `m08_chan` result is recorded as unverifiable in
`FAILED_EXPERIMENTS.md` rather than silently inherited.

* **Hypothesis** The metric loses most of its mass on **late** breaks, where few
  post-break observations exist. Evidence read *backwards* from the current
  point — reverse-time CUSUM and short-horizon variance/GLR tests over the
  observed prefix, calibrated against a length-matched historical null — detects
  a change in the last few observations that forward, expanding-window
  statistics cannot yet see, because a forward statistic averages the break
  away against a long pre-break prefix.
* **Matched control** `RT-301` = the 7 production modules, ABL protocol
  (5 canonical folds, 400k training rows, seed 0, battery hyperparameters).
  `RT-302` = the identical call plus the new module. Same folds, same rows,
  same seed, same params, same session.
* **Falsification (both pre-registered, either one rejects)**
  1. mean ABL delta ≤ 0 across the 5 folds, **or**
  2. the delta is positive on ≤ 2 of the 5 folds (not a majority).
* **Secondary, reported either way** TS-AUC by post-break age bucket
  (0–5, 5–10, 10–20, 20–50, 50–100 observations after τ), standalone TS-AUC,
  within-timestep rank correlation with the champion, and deployable ensemble
  delta.
* **Search degrees of freedom** 1 module design, 1 configuration, fixed window
  grid chosen a priori from `m00_core`'s existing grid — **no window tuning on
  validation**. Column budget ≤ 60.

## RT-303 — negative control for the RT-302 ensemble claim  (PRE-REGISTERED 2026-08-20)

* **Why** `RT-302` failed the bootstrap route to promotion (CI [−0.0019, +0.0049]
  straddles zero) but passed the *ensemble* route (+0.00371 deployable logit
  blend over `RT-301`). VALIDATION_V2 section 7 allows promotion on a
  demonstrated deployable ensemble delta — but only if the delta is attributable
  to the candidate.
* **Control** `RT-303` = the 7 production modules, ABL protocol, **seed 1**.
  Identical information content to `RT-301`; only the bagging and
  feature-sampling draws differ.
* **Falsification** the `RT-301`+`RT-303` logit blend gains within 0.0005 of what
  the `RT-301`+`RT-302` blend gains. Then the m09_back ensemble delta is generic
  two-model variance reduction and the promotion route closes.
* **Search degrees of freedom** 1 configuration, 0 variants.

## WAVE-3 OUTCOMES (recorded 2026-08-20, after the runs)

| id | what | result | label |
|---|---|---|---|
| `RT-300` | RT-100 config on macOS/arm64, CHAMP protocol | **0.61605** vs ledger 0.615103, delta **+0.00095** | **CONFIRMATION** — inside the pre-registered 0.0050 bound; 4 of 5 folds reproduce to the last printed digit, fold 3 moves +0.00476 |
| `RT-301` | ABL control, 7 modules, seed 0 | 0.61257 (wave-2 `RT-200` was 0.61282; 4 of 5 folds identical) | control |
| `RT-302` | `RT-301` + `m09_back` | 0.61413, delta **+0.00156**, positive on 4/5 folds | survives its own falsification, **NOT PROMOTED** — see `FAILED_EXPERIMENTS.md` |
| `RT-303` | seed-clone control, seed 1 | 0.61488 (wave-2 `RT-210` seed 1 was 0.61485) | negative control — **closed the ensemble route** |

**Search degrees of freedom actually spent in wave 3:** 1 new feature module,
1 configuration of it, 0 variants selected on validation, 0 hyperparameter
studies, 0 window-grid tuning, 4 logged full-protocol runs. No result was
selected from a set of alternatives, so the wave-3 rows carry no multiplicity
discount beyond the cumulative ledger.

**Two instrument findings, both re-usable:**

1. **This machine is deterministic; the ledger gap is architectural.** Fold 3 of
   the champion re-run three times gives 0.6122259125582563 every time, with
   bitwise-identical prediction vectors (`research/reports/wave3_determinism_fold3.json`).
   Adding `deterministic=True, force_row_wise=True` changes nothing. So the
   single-fold divergences from the wave-1/2 ledger are x86-vs-arm64 floating
   point tipping a borderline split, not run-to-run instability, and the shipped
   parameters already satisfy the competition's 1e-8 re-run eligibility condition.
2. **A seed change decorrelates more than a new feature family.** Within-timestep
   rank correlation with the same control: 0.7846 for a seed clone, 0.8219 for a
   51-column new module. Low rank correlation with the champion is therefore
   **not** evidence of new information, and no future stream may be promoted on a
   blend delta that has not been measured against a seed-clone control. Wave 2's
   seven-stream ensemble was never given this control and its diversity claim is
   correspondingly unaudited.

---

# WAVE 4 — PRE-REGISTRATION

**Written 2026-08-20, BEFORE the first wave-4 run. Nothing below was edited after
a result was seen; corrections appear as dated amendments, never as rewrites.**

Integration commit: `c4fb01e` on `research/wave3-integration`
(parents `bfcb232` research + `24675a6` engineering).

## W4-E1 — SPECIALIST DIVERSITY vs ORDINARY BAGGING

**The question.** The deployable seven-stream ensemble scores 0.62589 against the
single champion's 0.61510, a reported +0.01057. Wave 3 proved that a seed clone —
a model containing *zero* new information — can produce a *larger* blend delta and
a *lower* within-timestep rank correlation than a genuine new feature family. The
seven-stream ensemble's diversity claim therefore rests on two instruments that
are now known to be uncalibrated. This experiment calibrates them.

**Hypothesis (H1).** The seven wave-2 specialist streams — differing in feature
modules, tree depth, row-sampling policy, objective and boosting type — carry
information that seven seed clones of the champion do not, so under an identical
deployable calibration the specialist ensemble beats the seed-clone ensemble.

**Null (H0).** Most of the ensemble gain is ordinary bagging. Seven seed clones
land within noise of the seven specialists.

**Falsification of H1, fixed in advance.** H1 is rejected unless

    specialist SCDF ensemble  -  seed-clone SCDF ensemble  >  +0.0030

as a mean over the five canonical folds, **and** the difference is positive on at
least 4 of 5 folds. A difference in `(0.0010, 0.0030]` is declared INDETERMINATE
and resolved in favour of the simpler system. A difference `<= 0.0010` is noise
and H0 is accepted.

**Why +0.0030.** Wave 3 measured fold-to-fold SD 0.0085, partition-draw SD 0.0050
and seed SD 0.0012. 0.0030 is above seed noise and below partition noise; it is
also the brief's own "deserves investigation" line. It is chosen before the data.

**Arms — both seven members, both sharing member 1.**

| set | members | varies |
|---|---|---|
| SPECIALIST | `RT-300` (=`RT-100R`), `RT-410`..`RT-415` (=`RT-120R`..`RT-125R`) | modules, leaves, rows, sampling, objective, boosting, seed |
| SEED CLONE | `RT-300` (seed 0), `RT-401`..`RT-406` | **seed only** |

**PRE-REGISTERED SEED LIST — 0, 1, 7, 42, 2026, 31415, 271828.** Fixed here
before the first run. No substitution, no "best seven of ten", no dropping a
weak seed. Sharing `RT-300` between the arms is deliberate: it pairs the
comparison at member 1 and removes one run's worth of platform noise.

**Held constant across both arms.** The 10,000-series store, the canonical
`folds.parquet`, the 500-column feature cache, `num_threads=2`, macOS/arm64,
lightgbm 4.7.0, the SCDF calibration family and its cross-fitting scheme, the
equal-weight average, and the evaluation code.

**Analysis, fixed in advance.** For each arm: individual fold scores; the mean;
within-timestep and global pairwise correlations; and four blends — raw mean,
logit mean, global-CDF mean, smooth-time-CDF mean — each CROSS-FITTED so that
fold *k*'s calibration is fitted only on folds != *k*. The headline comparison is
SCDF-vs-SCDF. Paired series-level bootstrap, 200 replicates, common random
numbers across arms.

**Degrees of freedom this spends.** 12 new full-protocol runs, 2 ensemble
compositions, 0 hyperparameter searches, 0 seed selection, 0 calibration tuning.
The four blend families are all reported, not selected from.

**Stopping rule.** The comparison is made once, on the five canonical folds, and
then the alternate partitions (W4-E2) are read. No re-run of an arm with a
different seed if the answer is unwelcome.

## W4-E2 — DOES THE ENSEMBLE DELTA SURVIVE THE PARTITION DRAW?

**Hypothesis.** The ensemble delta (winner-of-W4-E1 minus single champion) is a
property of the method, not of the canonical fold draw, so it stays positive
under `folds_alt1`, `folds_alt2` and `folds_alt3`.

**Falsification.** The delta is negative on any alternate partition, or its
across-partition SD exceeds its canonical mean.

**Candidates declared BEFORE any alternate score is read:** exactly three — the
single champion, the seven-way seed ensemble, and the seven-way specialist
ensemble. The alternate partitions are a robustness diagnostic and will not be
used to select anything.

## W4-E3 — SCDF TIME COORDINATE, n_seen = t + 1

**Hypothesis.** The competition's online index is zero-based, so the calibrator's
`log(max(t,1))` maps t=0 and t=1 to the same anchor position. `log(t+1)` is the
mathematically clean coordinate and should be at least as good.

**Falsification / adoption rule, fixed in advance.** This is a BUGFIX CANDIDATE,
not a tuning knob. It is adopted only if, cross-fitted on the same folds and the
same streams, it does not regress by more than 0.0005. If the two are
indistinguishable the corrected definition wins on cleanliness. Anchor count,
grid size and `min_n` are NOT tuned — they stay at 12 / 256 / 400.

## W4-E4 — FOLD-MODEL DEPLOYMENT COST

**Hypothesis.** Because one shared feature engine dominates inference cost,
deploying 35 fold-boosters instead of 7 full-data boosters costs far less than
5x end-to-end.

**Falsification.** Measured ms/point with 35 boosters exceeds the 15-hour budget,
or exceeds 7-booster cost by more than 2x end-to-end.

**Measured before any score is read**: 1, 7, 14 and 35 boosters on the same
feature stream, p50 and p95 ms/point, model load time, memory, artifact size.

---

# WAVE 4 — RESULTS AGAINST THE PRE-REGISTRATION

## W4-E1 — **H1 CONFIRMED.** Specialist diversity beats bagging, but bagging is more than half the gain.

Bar fixed before the first run: `> +0.0030` mean **and** positive on `>= 4 of 5`
folds. Measured: **+0.00417, positive on 5 of 5.** Both conditions met.

| | mean OOF | per fold |
|---|---|---|
| single champion `RT-300` | 0.61605 | 0.62903 / 0.61061 / 0.62688 / 0.61223 / 0.60152 |
| seven seed clones, SCDF | 0.62164 | 0.63817 / 0.61667 / 0.63020 / 0.61437 / 0.60879 |
| seven specialists, SCDF | **0.62581** | 0.63828 / 0.62040 / 0.63392 / 0.61750 / 0.61894 |

Paired series bootstrap, 200 replicates, common random numbers:

| contrast | mean | 95% CI | positive |
|---|---|---|---|
| specialist − seed clone | +0.00409 | [+0.00199, +0.00614] | 200/200 |
| specialist − single | +0.00957 | [+0.00662, +0.01227] | 200/200 |
| seed clone − single | +0.00548 | [+0.00285, +0.00810] | 200/200 |

**The decomposition, which is the actual answer to the question asked:**

```
single champion                    0.61605
  + ordinary bagging (7 seeds)     +0.00559   <- 57% of the total
  + genuine specialist diversity   +0.00417   <- 43% of the total
= seven-stream deployable          0.62581
```

So the seven-stream architecture is **not** a bagging illusion — it clears its
own pre-registered bar on every fold with a bootstrap CI well clear of zero.
But **the majority of its advertised advantage is reproducible by training one
model seven times with different seeds**, and no wave-2 document says so. The
honest headline is "+0.0042 for specialisation on top of +0.0056 for bagging",
not "+0.0106 for a seven-stream architecture".

**Cross-platform corroboration.** This macOS specialist ensemble scores 0.62581;
the Linux wave-2 `RT-250` scored 0.62589. A 0.00008 gap across two platforms,
two rebuilds of every stream, and a complete loss of the original OOF vectors.
`RT-250` is now independently reproduced.

### Four things the arms say that were not asked for

1. **Calibration only matters when the members disagree about scale.** On the
   specialist arm the family spread is large — raw 0.62314, logit 0.62479,
   global CDF 0.62550, SCDF 0.62581, a +0.00267 spread. On the seed-clone arm
   every family lands within 0.00016. The SCDF machinery is not a general
   improvement; it is specifically a fix for heterogeneous score scales, which
   is exactly what the seven specialists have and seven seed clones do not.

2. **The legal calibration is not "recovering a fraction of the oracle" — it
   matches it.** Specialist SCDF 0.62581 vs the illegal within-timestep rank
   oracle 0.62580; seed-clone SCDF 0.62164 vs oracle 0.62160. In both arms the
   deployable transform is *at or above* the ceiling it was supposed to be
   approximating. The oracle framing has outlived its usefulness.

3. **Correlation did separate the arms, but only in the aggregate.** Specialist
   pairwise within-t rank correlation averages 0.6362 over [0.400, 0.782];
   seed clones average 0.7996 over [0.795, 0.803]. Wave 3's warning stands — the
   *top* of the specialist range (0.782) is indistinguishable from a seed clone,
   so per-stream correlation is still not a promotion credential. What separates
   the arms is the spread, not any single number.

4. **The specialists are individually WORSE.** Specialist members average
   0.61227; seed clones average 0.61544. The specialist arm wins the blend while
   losing on every member-quality measure — which is the ensemble effect working
   as designed, and a reminder that stream-level TS-AUC is the wrong thing to
   optimise for a member.

## W4-E3 — corrected SCDF time coordinate: **NEUTRAL, adopt for cleanliness**

`log(n_seen) = log(t+1)` vs the incumbent `log(max(t,1))`, cross-fitted,
identical anchors/grid/min_n:

| arm | incumbent | corrected | delta |
|---|---|---|---|
| specialist | 0.625814 | 0.625815 | +0.000001 |
| seed clone | 0.621640 | 0.621640 | 0.000000 |

Inside the pre-registered no-regression band by three orders of magnitude. The
adoption rule fixed in advance says the corrected definition wins ties, so it is
adopted: it is the coordinate that does not collapse t=0 onto t=1 and does not
score t=0 rows against a grid they were excluded from building. The gain is
correctness at the youngest online index, not TS-AUC — and TS-AUC at age 0–5 is
0.513, so there was never much there to win.

### Degrees of freedom spent in W4-E1/E3

17 full-protocol runs, 2 pre-declared ensemble compositions, 5 blend families all
reported and none selected from, 1 seed list fixed in advance and not revised,
0 hyperparameter searches, 0 calibration tuning. The hybrid composition
(4 specialists + 3 seed clones) was **not** tested, because it was not
pre-registered and testing it after seeing both arms is exactly the selection
this experiment exists to avoid.

---

# WAVE 4 — PRE-DECLARED FINAL-FIT POLICY

**Written 2026-08-20, BEFORE the partition study reported and BEFORE any
final-fit number exists. Recorded here so that no post-freeze choice can be made
in response to a post-freeze result.**

## The training-row budget rule

The competition allows all 10,000 labelled series. Development used 8,000; the
other 2,000 were the lockbox, which is **spent** for selection (two inspections)
and is therefore ordinary training data once the architecture is frozen.

**Rule, fixed now:** every stream's training-row budget scales by the series
ratio, `10000 / 8000 = 1.25`, rounded to the nearest thousand.

| stream | dev budget | final budget |
|---|---|---|
| `RT-100R` | 1,000,000 | 1,250,000 |
| `RT-120R`, `RT-121R`, `RT-122R` | 900,000 | 1,125,000 |
| `RT-123R`, `RT-124R`, `RT-125R` | 700,000 | 875,000 |

This keeps **rows sampled per series** constant, which is the quantity the
configurations were actually tuned around, and it applies identically to the
cross-fitted OOF pass (each fold trains on 8,000 of 10,000 series, itself 1.25x
the dev CV's 6,400) and to the final full-data boosters. One rule, both places,
no free parameter.

**No alternative was evaluated.** Row-budget scaling is not a knob to be tuned
after the fact; if it were tuned on anything measurable post-freeze that would
be selection, and there is nothing left to select on.

## The final calibration construction

`research/folds/folds_final10k.parquet`, generated by
`research/scripts/make_folds_final10k.py`, is a 5-fold stratified partition over
**all 10,000** series. `sha256(id,fold)` =
`6e9ebaf7cffada5bbf9e1d1f930f34c0b1fb2ce05287b46ab1303f6e89d8da65`.
Balance: 1,986–2,021 series per fold, break rate 0.4945–0.4998, ~1.00–1.02M
online rows per fold.

The canonical `folds.parquet` is untouched and the script refuses to overwrite
an existing partition file.

Calibration payloads will be fitted on cross-fitted OOF produced **under this
partition**, so the score distribution the CDF grids are estimated from comes
from models trained on 8,000 series while the deployed boosters are fitted on
10,000. That residual mismatch is smaller than the status quo — the wave-2
artifact's grids come from models trained on 6,400 series while its boosters saw
8,000 — but it is **not zero**, and it is not going to be closed by a correction
tuned after freeze. It is stated, bounded and accepted.

**Nothing scored under `folds_final10k` is evidence for any model choice.** By
the time it runs there are no model choices left to make. Any number it produces
is a FINAL-FIT diagnostic and is labelled as such.

---

# W4-E6 — PRE-REGISTRATION: DOES THE CHAMPION WANT BOTH?

**Written 2026-08-20, AFTER W4-E1 reported and BEFORE this experiment was run.
The provenance is stated plainly because it matters: W4-E1's decomposition is
what motivates the hypothesis. That is legitimate — a result suggesting the next
experiment is how research works — but it means this test carries the
multiplicity of having been chosen with knowledge of W4-E1, and its bar is set
accordingly.**

**The observation.** W4-E1 separated the seven-stream ensemble's gain into
+0.00559 from ordinary bagging and +0.00417 from specialist diversity. The
deployed champion currently harvests the second and only incidentally the first:
its seven members differ in seed, but each *configuration* appears exactly once.
If the two effects are even partly additive, a system carrying both should beat
one carrying mainly the second.

**Hypothesis (H1).** Blending the seven specialists together with the six extra
seed clones — 13 boosters over the same shared feature engine — beats the seven
specialists alone.

**Null (H0).** The two gains overlap. Once seven heterogeneous streams are
averaged, additional exchangeable members add nothing that averaging the
specialists has not already done.

**Compositions declared now, before any of them is scored.** Exactly three, and
they are unions, not selections — no member is chosen, dropped or reordered on
the basis of a score:

| id | composition | n |
|---|---|---|
| `RT-420` | the seven specialists (the incumbent) | 7 |
| `RT-421` | the seven seed clones | 7 |
| `RT-422` | **the union of both, deduplicated on `RT-300`** | 13 |

**Falsification of H1, fixed in advance.** H1 is rejected unless

    RT-422  -  RT-420  >  +0.0020   over the five canonical folds
    AND positive on at least 4 of 5 folds
    AND the paired series bootstrap's 95% CI excludes zero.

The threshold is lower than W4-E1's +0.0030 because this is a strictly cheaper
change — no new feature module, no new configuration, no new research surface,
just more of a thing already proven to work — but the extra bootstrap condition
is added because the hypothesis was chosen after seeing W4-E1.

**Cost condition, and it is binding.** A promotion also requires W4-E4 to show
that 13 boosters fit the runtime budget with margin. If 13 boosters cost more
than the budget allows, H1 being true is irrelevant and the incumbent stands.
The cost measurement is independent of the score measurement and neither is
allowed to move the other's threshold.

**What is NOT tested, and why.** Every intermediate composition — 4 specialists
+ 3 clones, 7 specialists + 2 clones, best-k of anything — is out of scope.
Those are exactly the selections this ledger exists to prevent, and testing the
union costs one number while testing the lattice costs a false positive.

**Degrees of freedom spent.** 0 new training runs, 1 new composition, 1
threshold fixed in advance, 0 members selected.

---

# WAVE 5 — PRE-REGISTRATION AND DEGREES-OF-FREEDOM RECORD

**Branch `research/wave5-alpha`, forked from `research/wave3-integration` @
`17bb5df`. Written 2026-08-21. The full design is `research/WAVE5_PREREG.md`,
committed at `5488644` BEFORE the first wave-5 number existed.**

**The external anchor.** LB-001 = **0.6268** on the Crunch public board, from the
RT-600 artifact whose development architecture scores 0.62581 on the canonical
partition. Internal → external transfer was flat to slightly positive, so wave 5
is an alpha-discovery project, not a validation-repair project.

**The binding standard, for every candidate.** Not "beats the champion blend".
`(S + candidate) − (S + seed clone)` ≥ +0.0030, on ≥4/5 canonical folds, with a
supportive paired bootstrap and directionally stable alternate partitions. Wave 3
(`m09_back`) and wave 4 (W4-E6) each produced a candidate that beat `S` and lost
to a same-strength stream carrying no information at all.

**Validation surface.** Canonical 8,000-series development folds and the three
alternate partitions. `RT-500`..`RT-506` (the all-10k OOF vectors),
`folds_final10k`, fold −1, `X_test.reduced` and the leaderboard are **not**
selection surfaces and are not reachable from `research/scripts/wave5_lib.py`.

## Experiments declared before any was scored

| id | question | new training runs | thresholds fixed in advance |
|---|---|---|---|
| W5-E1 | does a SMALL bagging component help, where the 13-way union did not? | **0** | λ grid {1.00, 0.90, 0.80, 0.70}, fixed at four values; screening bar +0.0010 on ≥4/5 |
| W5-E2 | `m10_persist` — outlier-driven vs bulk scale change | 1 | §4 bar |
| W5-E3 | hard-negative curriculum | 3 + 20 mining models | §4 bar; must not degrade ages 0–20 |
| W5-E4/5/6 | `m12_rdep` — residual distances, residual CUSUM/CUSUMSQ, dependence LR | 1 | §4 bar |
| W5-E7 | `m11_focus` — exact maximisation over candidate τ | 1 (+1 if it survives) | §4 bar |
| W5-E8 | absorbing-state BOCPD | **0 — already implemented** | see below |
| W5-E9 | TS-AUC-shaped pair weighting | 3 | §4 bar |
| W5-E10 | union of the surviving blocks | 1 | §4 bar |

**W5-E8 is declared NOT RUN, with a reason rather than an excuse.**
`src/sbr/features/m07_bayes.py` already implements the absorbing-state posterior
the brief asks for — its docstring reads "the latent state NOT-BROKEN → BROKEN is
ABSORBING, so the exact filtering recursion for P(broken at t | x_1:t) collapses
to one log-space accumulator per alternative parameter value", and it ships 50
columns of it inside the RT-600 artifact. Building a second one would have
measured the seed, not the mechanism.

## Degrees of freedom spent in wave 5

* **Seed lists**: none chosen. Every control reuses the seed list fixed in wave 4
  before its first run.
* **Hyperparameter searches**: 0. Every arm runs the ABL or CHAMP protocol
  verbatim from `research/scripts/wave2_lib.py`.
* **Calibration tuning**: 0. `SCDF_NSEEN`, 12 anchors, 256-point grids,
  `min_n=400`, frozen.
* **Ensemble weight optimisation**: 0. Only the four-point λ grid of W5-E1,
  declared in advance and not enlarged after it returned a null.
* **Curriculum constants** (`ALPHA=3.0`, `POWER=2.0`, `HARD_FRAC=0.10`,
  `OVER_K=4`): fixed in `research/scripts/wave5_e3_hardneg.py` before the first
  arm ran, one value each, no sweep.
* **A design chosen after seeing a result, stated plainly**: `m10_persist` was
  specified *after* the W5-D2 false-positive forensics reported, and targets the
  mechanism those forensics found rather than the generic family the brief
  listed. That is legitimate — a diagnostic suggesting the next experiment is how
  research works — but it means W5-E2 carries the multiplicity of having been
  chosen with knowledge of D2, and it is judged against the same unmoved §4 bar.
* **Causality**: every new module passes `check_prefix_invariance` at `atol=0.0`
  on 7 series including both length-10 series in the dataset. The gate earned its
  keep: the first `m12_rdep` sized its expanding nulls by `n_online`, the single
  forbidden input, and the check failed it on every series before any score was
  taken from it.

---

# W5-E11 — PRE-REGISTRATION: IS THE BLOCK AN EIGHTH MEMBER, OR A BETTER ARCHITECTURE?

**Written 2026-08-21, AFTER stage C reported and BEFORE any W5-E11 arm was
trained. The provenance is stated plainly, as W4-E6's was: stage C is what
motivates this, so W5-E11 carries the multiplicity of having been chosen with
knowledge of it, and its bar is fixed here.**

**The observation that forces it.** A null test of the promotion battery —
`W5-NULLTEST`, candidate `RT-402`, control `RT-401`, both seed clones of the
champion — measured what an EIGHTH exchangeable member is worth:

| composition | TS-AUC | vs `S` |
|---|---|---|
| `S` (seven specialists) | 0.62581 | — |
| `S` + `RT-401` (8th seed clone) | 0.62584 | **+0.00003** |
| `S` + `RT-402` (a different 8th seed clone) | 0.62530 | −0.00051 |

**The seven-member equal-weight blend is saturated in members.** This is now the
third independent test saying so — W4-E6 (13-way union, −0.00095), W5-E1 (λ
mixture, monotone decreasing), and this. A new feature block evaluated as an 8th
member is therefore being asked to move a blend that an 8th member cannot move:
its column weight is 1/8, and the entire seven-member specialisation effect is
+0.0042. §4's +0.0030 bar, read that way, is not a high bar — it is close to an
impossible one, and passing or failing it would say more about the composition
than about the block.

**Hypothesis (H1).** The strongest surviving block, `m12_rdep`, improves the
ARCHITECTURE rather than adding a member: rebuilding all seven specialist streams
with the block available to each beats the incumbent seven.

**Null (H0).** The block's information is already reachable by the incumbent
seven-stream bank, and adding 57 columns to each stream changes nothing that
averaging them has not already done.

**Arms, declared now.** Seven streams, each the EXACT incumbent configuration
from `research/scripts/wave2_streams.py` — same seed, same rows, same leaves,
same sampling, same objective — with `m12_rdep` appended to its module list and
**nothing else changed**:

| new id | rebuilds | modules |
|---|---|---|
| `RT-751` | `RT-300` / `RT-100R` | 7 production + `m12_rdep` |
| `RT-811` | `RT-410` / `RT-120R` | m00, m01, m07 + `m12_rdep` |
| `RT-812` | `RT-411` / `RT-121R` | m02, m03, m04, m06 + `m12_rdep` |
| `RT-813` | `RT-412` / `RT-122R` | 7 production + `m12_rdep` |
| `RT-814` | `RT-413` / `RT-123R` | 7 production + `m12_rdep` |
| `RT-815` | `RT-414` / `RT-124R` | m07, m06, m01 + `m12_rdep` |
| `RT-816` | `RT-415` / `RT-125R` | 7 production + `m12_rdep` |

`S'` is the equal-weight cross-fitted SCDF blend of those seven. The control is
`S`, measured in the same session on the same folds. **No member is selected,
dropped, reordered or reweighted; the only change is one module appended to every
stream.**

**Falsification of H1, fixed in advance.** H1 is rejected unless

    S' - S  >  +0.0030   over the five canonical folds
    AND positive on at least 4 of 5 folds
    AND the paired series bootstrap's 95% CI excludes zero.

The threshold is §4's, unchanged, because this is the comparison §4 was written
for — a candidate against the strongest relevant matched control, which here is
the incumbent architecture itself.

**Why no seed-clone control is named for this arm.** `S'` differs from `S` by 57
columns per stream and by nothing else — not by member count, not by weight, not
by seed. There is no bagging channel for the gain to arrive through, which is the
whole reason this comparison is cleaner than the eighth-member one.

**What is NOT tested, and why.** Adding `m11_focus` or `m10_persist` to the same
rebuild; any subset of streams; any reweighting. `m10_persist` failed stage C and
is out. `m11_focus` survived stage C and a combined rebuild is the obvious next
step — which is exactly why it is NOT run here: testing one block costs one
number, testing the lattice costs a false positive.

**Degrees of freedom spent.** 6 new training runs (`RT-751` is already declared
under stage D), 1 composition, 1 threshold fixed in advance, 0 members selected,
0 hyperparameters touched.

---

## WAVE 6

### W6-E1 — calibration anchor placement · **RESOLVED, FALSIFIED**

3 declared schemes + 1 random-anchor null, 0 selected, 1 threshold fixed in
advance (+0.0010), 0 hyperparameters touched, 0 submissions. Total spread across
every scheme including random placement: **0.00006**. Closed; may not be
reopened by a re-parameterisation of the same idea
(`research/WAVE6_STOPPING_RULE_AMENDMENT.md` §3(a)).

### W6-E2 / `RT-900` — **ATTEMPTED BUT VOID**

**This row counts as a spent degree of freedom, not as evidence.** One training
run (1,000,000 rows, CHAMP protocol, 906 s) was executed and produced an invalid
number. Recording it as "nothing happened" would understate the search; recording
its 0.86552 as a result would be fraud. It is therefore filed as **attempted but
void**: it consumed budget, it selected nothing, and it may not appear on either
side of any comparison.

| | |
|---|---|
| runs spent | 1 |
| results contributed | **0** |
| selection events | **0** |
| multiplicity charged | 1 attempt |

### W6-E2R — the corrected series-level oracle diagnostic

| | count |
|---|---|
| arms declared in advance | 6 (`A_rich`, `A_basic`, `B_causal`, `B_causal_withpos`, `C_nobound`, `AB`) |
| arms selected | **0** — this is a diagnostic and can promote nothing |
| leakage sentinels declared in advance | 7 |
| pseudo-τ seeds | 5, the prior study's, unchanged |
| thresholds fixed in advance | 3 (±0.010 reproduction, 0.005 case B, 0.010 case C) |
| hyperparameters it may select | **0** — explicitly forbidden from touching any neural hyperparameter (`WAVE6_PREREG.md` §18.8) |
| Crunch submissions | **0** |

**Falsification, fixed in advance.** The instrument is declared uncalibrated —
and nothing is inferred — if `A_rich` on the unfiltered population misses the
prior study's 0.6497 by more than ±0.010, or if any sentinel exceeds its
frontier counterpart materially (metadata 0.5379, permuted 0.5075,
random-boundary 0.5821).

**Why the learner is held fixed across arms.** Same 150-tree LGBM, same
hyperparameters, same cross-fit, same `pseudo_seed + 17` learner seed. Only the
representation changes, so a difference is attributable to representation or to
nothing. Changing the learner and the representation together would have made
the result uninterpretable in exactly the way `WAVE6_PREREG.md` §7 forbids.

### W6-N — the neural track

Declared in advance: 2 families now (`N1` MLP, `N2` TCN), 1 conditional (`N3`
GRU), ≤ 2 architecture sizes and ≤ 2 regularisation settings per family, 1 loss
(BCE) plus at most 1 pre-registered metric-aligned alternative. No architecture
search, no loss sweep, no seed fishing. Each family costs its runs whether or not
it works, and each is charged against multiplicity at declaration time, not at
publication time.

---

## WAVE 7

### W7-D0 — exact RT-600 pairwise-inversion loss cube

**Diagnostic only. No training, no candidate, nothing selected.** Pure
recombination and exact re-scoring of `research/oof/wave5_S_specialist.npy`
(RT-600's development architecture, `RT-420`) using the official metric's own
rank/pair machinery (`sbr.metric.ts_auc_flat`, `return_per_step=True`),
decomposed by current-t bucket × positive-age bucket × negative type
(never-break / pre-break).

| | count |
|---|---:|
| runs spent | 0 (recombination of existing OOF, one scoring pass) |
| results contributed | 1 (the exact loss cube) |
| selection events | **0** |
| hyperparameters touched | **0** |
| Crunch submissions | **0** |

**Reproduction check, before any cube number was trusted.** Re-scored the
exact champion OOF vector against the committed `RT-420` numbers (0.62581
mean, folds 0.63828/0.62040/0.63392/0.61750/0.61894): delta +0.000001 mean,
max per-fold delta +0.000009. PASS.

**Result.** Dominant cell (t≥200, age≥100): 50.50% of pair weight, **45.29%**
of exact inversion loss (prior inferred estimate was 45.7% — approximately
confirmed), cell AUC 0.66428. Inside that cell, never-break negatives carry
74.0% of the loss, pre-break only 26.0% — the open problem is separating
mature persistent weak breaks from truly stable series, not baseline
heterogeneity. Full writeup: `research/WAVE7_RT600_EXACT_ALPHA_BUDGET.md`.
Evidence: `research/reports/wave7_rt600_exact_alpha_budget.json`.

### W7-D3R — same-prefix vs. full-sequence three-arm diagnostic

Pre-registered `research/WAVE7_D3R_PREREG.md` (committed before any arm was
trained). Population fixed by W7-D0 (t≥200, age≥100, all negatives). 2 new
trainings (`RT-990` Arm B, `RT-991` Arm C); Arm A reuses `RT-300`.

| | count |
|---|---:|
| arms declared in advance | 3 (A reused, B and C trained) |
| arms selected | **0** — diagnostic, promotes nothing |
| hyperparameters swept | **0** — B's and C's configs are fixed in the prereg, not tuned |
| Crunch submissions | **0** |

**Result, cell TS-AUC (t≥200, age≥100):** A 0.65341, B 0.64749, C 0.71859.
**B beats A on 1/5 folds** (mean Δ −0.00592 — stronger tree capacity on the
identical 500 legal columns/rows does not help, consistent with the same
sign on the whole-dev-set comparison, B 0.61177 vs A 0.61605). **C beats B on
5/5 folds** (mean Δ +0.07110, every fold ≥+0.053) — each series' own final
online-row feature vector (full training-sequence info, no true tau) recovers
enormous discrimination the causal prefix cannot see.

**VERDICT: CASE 2 — future-information limit.** Extraction capacity is not
the bottleneck in this cell; the online prefix is information-limited.
**Next lane: teacher/distillation (Priority 1)**, not a horizon specialist —
Arm B already tested "more capacity, same information" on this exact cell
and it did not help. Full writeup: `research/reports/wave7_d3r.md`.
Evidence: `research/reports/wave7_d3r.json`.

### W7 teacher/distillation — one-fold pilot

Pre-registered `research/WAVE7_TEACHER_PREREG.md` (committed before any
teacher-target diagnostic or student score existed). Teacher target `Q` =
`research/oof/RT-991.npy` (W7-D3R Arm C), **reused, not retrained** — already
series-level cross-fitted, so no inner cross-fitting was needed. Fold 0 only,
500-column causal bank unmodified, same rows/capacity as `RT-990` (reused as
the hard-label control `T0`, no retraining needed). 2 new trainings
(`RT-992` T1 pure distillation, `RT-993` T2 hard+teacher 0.5/0.5 blend).

| | count |
|---|---:|
| arms declared in advance | 3 (T0 reused, T1 and T2 trained) |
| arms selected | **0** — one-fold screen, continuation gate only, no promotion |
| hyperparameters swept | **0** — capacity fixed identical to `RT-990`; blend weight fixed 0.5/0.5, no grid |
| Crunch submissions | **0** |

**Pre-flight checks, both passed before any student trained.** Teacher-target
diagnostics (`research/reports/wave7_teacher_diagnostics.md`): same-t AUC of
`Q` alone reproduces Arm C exactly (0.71989 pooled, 0.71859 cell — bit-for-bit
cross-check that `Q` loaded correctly); `corr(Q,y)=0.408`, cell-positive
`std=0.346` — nowhere near the pre-registered stop-and-redesign trigger
(`|corr|>0.98` and `std<0.02`). `xentropy`/`binary` hard-label parity check
(`research/reports/wave7_teacher_parity_check.json`): correlation
`0.9999999999999999`, AUC identical to machine precision — `xentropy` trusted
as the soft-label-safe objective (LightGBM's built-in, not a new
implementation; none existed in this repo before this pilot).

**Result, fold 0 only, dominant cell (t≥200, age≥100) vs `T0`:**

| arm | cell TS-AUC | Δ cell | whole-fold0 Δ | translated aggregate Δ | distillation efficiency |
|---|---:|---:|---:|---:|---:|
| T0 (`RT-990`) | 0.66381 | — | — | — | — |
| T1 (`RT-992`) | 0.68169 | +0.01788 | +0.01621 | +0.00903 | 33.9% |
| T2 (`RT-993`) | 0.68439 | +0.02058 | +0.01671 | +0.01039 | 39.0% |

(Distillation efficiency = student cell gain ÷ D3R's own fold-0 Arm C−Arm B
gap, +0.05277 — the smallest of the five D3R fold-level gaps, so this is a
conservative denominator, not a favorable one.) **Both arms clear all three
pre-registered continuation gates** (cell Δ≥+0.010 with no whole-fold
damage; whole-fold0 Δ≥+0.003; translated aggregate Δ≥+0.004) by a wide
margin. **VERDICT: CONTINUE.** This is a single-fold screen only — it has
**not** cleared the full promotion battery (≥4/5 folds, bootstrap CI,
alternate partitions) and licenses no submission. Full writeup:
`research/reports/wave7_teacher_pilot.md`. Evidence:
`research/reports/wave7_teacher_pilot.json`.

### W7 teacher/distillation — NESTED outer-fold-pure correction, 5-fold

**The pilot above was found to be outer-fold contaminated** (post-hoc
review): `Q` (global `RT-991` OOF) is cross-fitted per-row/series but not
per-*outer-student-validation-fold* — a fold-1 training row's `Q` came from
a teacher trained on fold 0, the pilot's own outer validation fold. Standard
nested-CV meta-feature leakage, not a deployment-causality failure (the
student's inference path stays strictly causal). Pre-registered the fix
(`research/WAVE7_TEACHER_NESTED_PREREG.md`, committed before any nested
score existed): for outer fold `f`, every inner teacher trains on folds
excluding `{f, g}` (`g` = its target fold), never `f`. A mechanical
fold-purity sentinel (pure set arithmetic, no training) confirmed 0/20
violations on the new scheme and — proving it actually catches the known
defect — 20/20 contaminated checks on the old scheme, run and committed
before any inner teacher trained.

20 inner teacher fits (no IDs, internal machinery) + `T1`/`T2` trained per
outer fold on nested `Q`, evaluated on the untouched outer fold, merged into
full 5-fold OOF vectors `RT-994`/`RT-995`.

| | count |
|---|---:|
| arms declared in advance | 2 (`T1`, `T2`; `T0`=`RT-990` unaffected, reused) |
| arms selected | **0** — measurement only, promotion legs checked but not acted on |
| hyperparameters swept | **0** — identical capacity/rows/features to the pilot |
| Crunch submissions | **0** |

**Result, whole-dev TS-AUC vs `T0` (`RT-990`), all 5 outer folds:**

| outer fold | T0 | T1 (`RT-994`) | T2 (`RT-995`) | T1−T0 | T2−T0 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.62656 | 0.61931 | 0.63103 | −0.00725 | +0.00447 |
| 1 | 0.60617 | 0.61792 | 0.62008 | +0.01175 | +0.01391 |
| 2 | 0.61753 | 0.61242 | 0.62658 | −0.00512 | +0.00904 |
| 3 | 0.60318 | 0.62065 | 0.61679 | +0.01747 | +0.01361 |
| 4 | 0.60581 | 0.60642 | 0.61190 | +0.00061 | +0.00609 |
| **mean** | | | | **+0.00349** | **+0.00943** |

**Paired series bootstrap (200 reps):** `T1−T0` mean +0.00308, CI95
`[−0.00468, +0.01089]` (crosses zero, 78% of reps positive). `T2−T0` mean
+0.00922, CI95 `[+0.00425, +0.01373]` (**entirely above zero**, 100% of reps
positive).

**`T1` (pure distillation): 3/5 folds positive, mean +0.0035, bootstrap CI
crosses zero → FAILS promotion legs 2 and 3. Does not clear promotion.**
Once outer-fold-pure, pure distillation on the teacher's raw score is not a
reliable signal — the contaminated pilot's apparent +0.0090 translated gain
for this arm was mostly an artifact.

**`T2` (hard+teacher 0.5/0.5 blend): 5/5 folds positive, mean +0.0094,
bootstrap CI entirely above zero → CLEARS all three measured promotion
legs** (magnitude ≥+0.0030, ≥4/5 folds, bootstrap CI>0). Reading: **major
breakthrough** by the pre-registered scale. Improvement is broad-based, not
narrowly concentrated in the dominant cell — positive across every age
bucket and every current-`t` bucket measured
(`research/reports/wave7_teacher_nested.json` §age_buckets/§t_buckets).

**Contamination comparison, fold 0 only (the pilot's only evaluated fold):**
`T1` was overstated by **+0.0235** (contaminated +0.0162 vs. true clean
**−0.0072**, i.e. the sign flips); `T2` was overstated by **+0.0122**
(contaminated +0.0167 vs. clean +0.0045). The contamination was large and
directionally misleading for `T1`, but a smaller, real, robust effect
survived the correction for `T2`.

**Leg 4 (alternate-partition stability) is authorized but has not run.** No
promotion decision, no ensemble-integration test against `RT-600`'s existing
seven specialists, and no submission until it does. Full writeup:
`research/reports/wave7_teacher_nested.md`. Evidence:
`research/reports/wave7_teacher_nested.json`,
`research/reports/wave7_teacher_nested_fold_purity_test.json`.

## WAVE 7 — T2 PROMOTION BATTERY, LEG 4/ENSEMBLE-INTEGRATION (`research/wave7-t2-promotion`)

Leg 4 (alternate-partition confirmation) was **not run** — a resource
decision, made explicit rather than silently skipped: the ensemble-
integration leg below (the binding measurement for competition value)
already gates T2 at MOSTLY REDUNDANT, and alternate partitions would cost
~15h of additional compute to re-confirm a standalone number that has
already been shown not to translate into ensemble value. Fold files
(`research/folds/folds_alt{1,2,3}.parquet`) remain ready if this leg is
wanted for the written record.

**Ensemble-integration result (`RT-1100`, pre-registered
`research/WAVE7_T2_INTEGRATION_PREREG.md`):** E0 (RT-600 seven specialists,
fold 0) = 0.63828; E1 (+ matched seed clone `RT-401`) = 0.63859; E2 (+ `T2`)
= 0.63882. **E2 − E1 = +0.00024** → **MOSTLY REDUNDANT** by the
pre-registered bands. T2 correlates 0.71–0.91 with the seven specialists
individually and 0.91 with the seed clone itself — statistically
indistinguishable from an exchangeable eighth stream. Same-t pair-repair
vs the E0 blend is net-negative for both T2 (−15) and the seed clone (−66),
uncalibrated.

T2's standalone `T2 − T0` (+0.00943, clears its own promotion legs against
a matched single-model control) does **not** transfer to ensemble value —
the gap between the two numbers (+0.00943 vs +0.00024, ~40×) is itself the
finding. T2 is redundant, not disqualified: it remains the only positive
ensemble marginal measured anywhere in this project (Wave 8's five
mechanisms were all ≈ −0.0003), just too small to matter on its own.

Full writeup: `research/reports/wave7_t2_promotion_final.md`. Evidence:
`research/reports/wave7_t2_ensemble_integration.{md,json}`,
`research/reports/wave7_t2_pairflow.{md,json}`,
`research/reports/wave7_t2_promotion_final.json`.
