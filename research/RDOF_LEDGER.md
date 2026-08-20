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
