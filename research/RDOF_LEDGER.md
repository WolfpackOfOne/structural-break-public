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

# WAVE 5 — OPENING ENTRY (`research/wave5-alpha`)

**Written 2026-08-22, before any Wave-5 experiment exists.**
Branch parent `research/wave3-integration` @ `17bb5df`. Charter:
`research/WAVE5_CHARTER.md`.

## Degrees of freedom spent so far: none

| | |
|---|---|
| training runs | **0** |
| candidates scored | **0** |
| TS-AUC numbers produced | **0** |
| hypotheses tested | **0** |
| ensemble compositions compared | **0** |
| hyperparameter studies | **0** |
| leaderboard submissions consulted for selection | **0** |
| packages audited | 8 |

Wave 5 opens with an environment and capability audit only
(`research/PACKAGE_CAPABILITY_MATRIX.md`,
`research/reports/wave5_package_probe.json`). No model was fitted, so no
selection occurred and no multiplicity discount is owed.

## New external information admitted this wave

RT-600 scored **0.6268** on the public Crunch leaderboard against a comparable
internal **0.62581** — a transfer of **+0.0010**.

**This is a fact, and it is not a parameter.** It may be cited as evidence that
the internal validation framework is directionally useful. It may **not** be used
to select, tune, weight, or rank anything. `VALIDATION_V2.md` §8 prohibits using
the leaderboard as the optimiser, and the arrival of a favourable number is
exactly when that rule earns its keep.

It also **falsifies, in the favourable direction, a pre-registered expectation**:
`FINAL_ARCHITECTURE_FREEZE.md` §4 put the base case at ~0.615 and the optimistic
case at ~0.625. The observed value beat the optimistic case. Recorded because a
pre-registration that turns out wrong is evidence about our calibration, and the
error was **pessimism about level transfer**, not optimism. One observation does
not establish a transfer law in either direction.

## Standing constraint on this environment

**No competition data is present in the container** (2026 feature store absent,
2025 parquet absent). No Wave-5 experiment can execute here until the store is
mounted or rebuilt, or the work moves to where the data lives. TabPFN
additionally requires a sideloaded checkpoint: `huggingface.co` is blocked by
network policy and the 8.4.0 default repo is gated.

## Bars that carry forward unchanged

Every bar in `VALIDATION_V2.md` §7, plus the wave-3 seed-clone rule: **a new
stream must beat a seed-clone control**, and low within-timestep rank correlation
is not a diversity credential. A new *model family* is subject to this
identically — XGBoost or CatBoost being a different library is not, by itself,
evidence of new information.

---

# WAVE 5 — PRE-REGISTRATION: `RT-500x` ALPHA LANE

**Written 2026-08-22, BEFORE any Wave-5 score exists — indeed before the
competition data is even present in the environment. Nothing below was written
with knowledge of a result, because no result is obtainable yet.**

**ID note.** `RT-500`..`RT-506` are ALREADY ALLOCATED to the post-freeze
cross-fitted all-10k OOF vectors (`FINAL_ARCHITECTURE_FREEZE.md` §"final
calibration construction"). Wave 5 therefore allocates from **`RT-520`
upward**, verified unused on every branch before allocation.

## W5-E1 — `m10_perm`: TRANSIENT vs PERMANENT

* **Hypothesis (H1).** TS-AUC by post-break age is 0.513 at age 0–5 against
  0.646 at 100+ (measured in the `m09_back` post-mortem). At small post-break
  age a genuine level shift and a short outlier burst produce the same
  trailing-window mean, because a window mean is invariant to the ARRANGEMENT
  of the points inside the window. Statistics of that arrangement — crossing
  counts, run lengths, mass concentration, and a mean-vs-median contrast —
  carry information no window-mean bank can express at any window length, and
  adding them raises TS-AUC on the young-break buckets.

* **Null (H0).** `m00_core`'s six trailing windows plus the expanding window
  already capture everything about the online segment that predicts a break;
  arrangement is redundant with level once the level is calibrated.

* **Matched control.** `RT-520` = the 7 production modules, ABL protocol
  (5 canonical folds, 400k training rows, seed 0, battery hyperparameters).
  `RT-521` = the identical call plus `m10_perm`. Same folds, rows, seed,
  params, session.

* **Falsification, both fixed in advance, either one rejects.**
  1. mean ABL delta ≤ 0 over the five folds, **or**
  2. the delta is positive on ≤ 2 of the 5 folds.

* **THE SEED-CLONE BAR — binding, and it is what killed `m09_back`.**
  `RT-522` = the 7 production modules at **seed 1**, a stream containing zero
  new information. If the deployable blend `RT-520`+`RT-521` does not beat the
  blend `RT-520`+`RT-522` by at least **+0.0010**, the ensemble route to
  promotion is CLOSED, whatever the ABL delta says. Low within-timestep rank
  correlation is not admissible as evidence.

* **Secondary, reported either way.** TS-AUC by post-break age bucket
  (0–5, 5–10, 10–20, 20–50, 50–100, 100+) — this is the hypothesis' own
  diagnostic and the place H1 must show up if it is true; standalone TS-AUC;
  rank correlation with the champion; per-family gain share
  (`xn` / `mx` / `cn` / `md`).

* **Pre-committed reading of a mixed result.** If the aggregate delta is
  positive but the age 0–5 and 5–10 buckets do **not** improve, H1 is **REJECTED
  even if the aggregate clears the bar** — that is exactly the pattern that
  falsified `m09_back` (it made young breaks worse and mature breaks better,
  the opposite of its stated mechanism), and it will not be re-read as a success
  here.

* **Search degrees of freedom.** 1 module design, 1 configuration, window grid
  (8,16,32,64,128,256) taken verbatim from `m00_core` with **no tuning**;
  `MGRID` (8,16,32,64) fixed by a stated mechanism argument before any score.
  24 columns. 0 hyperparameter searches.

### Work already completed under W5-E1 (no scores, none obtainable)

* Module built: `src/sbr/features/m10_perm.py`, 24 columns, 40.5 ms/series
  against the ~80 ms budget.
* **Bitwise prefix-invariant at `atol=0.0`** across 6 length profiles × 4 break
  kinds × 7 truncation points. Gate is permanent:
  `tests/test_m10_perm_causal.py` (27 tests, all passing, no data required).
* Per-column audit clean: no exploded scale, no constant column.
* Mechanism diagnostic (NOT selection, no TS-AUC): on a burst and a sustained
  shift carrying identical cumulative displacement, 14 of 24 columns separate at
  |Cohen d| > 1.0; concentration `cn256` at d = −11.5, `md64` at −7.27.
  Crossings collapse for the sustained shift and not for the burst, which is the
  documented direction.

### A NEGATIVE RESULT ESTABLISHED WITHOUT DATA

The module's first design carried a **mean-vs-robust-mean** contrast. It is
**identically zero and can never carry information**: `(x−mu)/sd` and
`(x−med)/mad` are affine images of one another, a window mean of an affine map
is affine in the window mean, and null-standardising each arm against its own
null removes any affine map exactly. Measured max difference **1.8e-15** on
heavy-tailed t(3) input; the median-based contrast measures 4.0 on the same
input. The dead family was removed before it could be scored, and
`tests/test_m10_perm_causal.py::test_affine_robust_contrast_would_be_dead`
prevents its return.

**Any future "robust versus plain" contrast in this project must use a
non-affine robust statistic or it is a column of zeros.** This cost zero
compute and no degrees of freedom to establish.

### Candidate REMOVED from the wave-5 lane before any compute was spent

**"Robust distribution distances"** is named as NOT RUN in
`STATE_OF_RESEARCH_V4.md` §12. On inspection `m02_dist` **already** emits
1-D Wasserstein, energy distance, Kolmogorov–Smirnov, Cramér–von Mises,
Jensen–Shannon, Hellinger, total variation and chi-square, each against a
length-matched historical null, as closed-form functions of the PIT occupancy
vector. A new module would be a re-statement of an existing channel — the exact
failure mode that killed `tl_` (−0.00410) and `rk_` (−0.00273), both of which
were individually strong and redundant with `m00_core`. **Dropped, with the
reason recorded, rather than run.**

## Degrees of freedom spent in Wave 5 to date

0 training runs · 0 candidates scored · 0 TS-AUC values · 0 hyperparameter
searches · 1 module built and certified · 1 sub-family removed on a proof ·
1 named candidate withdrawn on a redundancy audit.
