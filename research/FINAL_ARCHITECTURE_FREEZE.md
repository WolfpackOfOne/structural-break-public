# FINAL ARCHITECTURE FREEZE

**Frozen 2026-08-21. Branch `research/wave3-integration`.**

**NO FURTHER MODEL SELECTION AFTER THIS POINT.**

Everything below is fixed. The only remaining actions are: fit on all 10,000
labelled series, build the artifact, run the official `crunch test`, and
document. No number produced after this document is committed may be used to
revisit any choice in it — not a fold score, not a calibration diagnostic, not a
runtime measurement, not a leaderboard position.

---

## 1. THE INVESTMENT-COMMITTEE REVIEW

Adversarial review of the champion before freezing, against the questions in the
brief. The champion is not the client; the answers are what they are.

**What is the alpha mechanism?** Seven LightGBM boosters read overlapping slices
of one 500-column causal feature vector. They differ in feature subset, tree
depth, row-sampling policy, training-row budget, objective and boosting type, so
they are wrong in different directions. Averaging their *time-conditionally
calibrated* scores cancels part of that error. TS-AUC compares series within a
timestep, so the members must be on a common scale before averaging — that is
what the calibration is for, and W4-E1 proved it: the calibration family is
worth +0.00267 when the members are heterogeneous and 0.00016 when they are seed
clones.

**What is the false-signal mechanism?** Ensembling always helps a little.
Wave 3 showed a seed clone decorrelates *more* than a genuine new feature family
and blends *better*, so "low correlation + positive blend delta" is not evidence
of new information.

**What is the matched control, and does it beat one?** A seven-way seed-clone
ensemble, same configuration, same protocol, same calibration, seeds fixed in
advance. **Specialists beat it by +0.00417 on 5/5 folds**, bootstrap CI
[+0.00199, +0.00614], 200/200 replicates positive. Pre-registered bar was
+0.0030 and ≥4/5. Cleared.

**Is the gain bigger than research resolution?** Fold SD 0.0085, partition SD
0.0039 (measured, single-model levels), seed SD 0.0011. The total ensemble delta
is +0.0083 mean across four partitions with SD 0.0011 — comfortably above seed
noise and, critically, an order of magnitude more stable than the level.

**Alternate partitions?** All twelve deltas positive across canonical/alt1/alt2/
alt3. **But canonical is the most favourable of the four**, and on alt1 the
specialisation delta (+0.00254) would not have cleared W4-E1's own bar. Recorded,
not hidden. The effect is real and smaller than one draw made it look.

**Is it a feature contribution or a bagging contribution?** **Mostly bagging.**
+0.00499 of the +0.00832 mean total delta is reproducible by training one model
seven times with different seeds. Specialisation adds +0.00333 on top. No wave-2
document said this.

**Would a simpler model do the same thing?** Partly — and that is the honest
finding. A seven-seed ensemble of one configuration gets ~60% of the gain for a
much simpler story. It is not chosen because the specialists genuinely beat it,
on every partition, with a bootstrap CI clear of zero.

**Would a more complex one do better?** No. W4-E6 tested the union of both arms
(13 boosters) and it scored **worse** than the seven specialists (−0.00095,
1/5 folds). More members is not better; the incumbent composition beats its own
obvious enrichment.

**Platform deterministic?** Yes. Fold 3 re-run three times gives bitwise
identical predictions. Four of five champion folds reproduce the Linux ledger to
the last printed digit. `crunch test`'s 1e-8 determinism check passed.

**Causal? `n_online` safe?** Every module is bitwise prefix-invariant at
`atol=0`. `infer` is gated by a test that hands it iterables raising on `len()`,
indexing, reversal and re-iteration, plus a counter asserting score *k* is
emitted after exactly *k+1* points.

**Runtime?** Measured, not asserted: shared engine 1.069 ms/pt, marginal booster
0.078 ms/pt, seven boosters 1.734 ms/pt. The 15-hour budget is not close to
binding.

**How many variants were tried, and does the conclusion survive them?** Wave 4
spent 56 training runs, 4 ensemble compositions, 5 calibration families (all
reported, none selected from), 1 seed list fixed in advance, 0 hyperparameter
searches, 0 calibration tuning. Two of the four compositions were rejected. The
surviving claim was pre-registered with its threshold before the first run.

**Was the result anticipated before it was measured?** Yes, and the *direction*
of the surprise is recorded: the pre-registration expected either a clean win or
a clean null. What actually happened — a win that is real but half the size the
headline implied — was not anticipated and is the reason the expected-performance
estimate below is lower than any development number.

---

## 2. THE FROZEN ARCHITECTURE

**Ensemble.** Seven LightGBM boosters over one shared streaming feature engine.
Equal-weight arithmetic mean of seven calibrated scores. No weights, no stacking,
no gating.

**Feature bank.** 500 columns, 7 modules, in this exact order — the order is part
of the freeze because the model manifest indexes into it:

    m00_core, m01_seq, m02_dist, m03_dyn, m04_resid, m06_loc, m07_bayes

`feature_manifest_sha256 = 1646c3b9e09d8a7fb3c564483a1d1d999680caeefe848b092f907a6cac80cced`

**Streams.** Configurations verbatim from `research/scripts/wave2_streams.py`;
`max_train_rows` scaled by the pre-declared 1.25x rule.

| # | stream | modules | cols | seed | rows (final) | sampling | notes |
|---|---|---|---|---|---|---|---|
| 0 | `RT-100R` | all 7 | 500 | 0 | 1,250,000 | uniform | 63 leaves |
| 1 | `RT-120R` | m00, m01, m07 | 261 | 0 | 1,125,000 | uniform | 127 leaves, ff 0.35 |
| 2 | `RT-121R` | m02, m03, m04, m06 | 239 | 1 | 1,125,000 | uniform | 31 leaves, ff 0.7 |
| 3 | `RT-122R` | all 7 | 500 | 7 | 1,125,000 | per_series | 255 leaves, extra_trees |
| 4 | `RT-123R` | all 7 | 500 | 0 | 875,000 | uniform | objective `pairwise_t` |
| 5 | `RT-124R` | m07, m06, m01 | 170 | 3 | 875,000 | uniform | 63 leaves, ff 0.6 |
| 6 | `RT-125R` | all 7 | 500 | 11 | 875,000 | uniform | `boosting=goss` |

Shared: `learning_rate 0.05`, `n_estimators 600`, `lambda_l2 5.0`,
`max_bin 127`, `num_threads 2`, `objective binary` (except stream 4).
**`num_threads=2` is part of the freeze** — determinism depends on it.

**Seeds are per-stream and fixed:** 0, 0, 1, 7, 0, 3, 11. Not chosen; inherited
from the configurations that were validated.

**Calibration.** `sbr.production.calibration.SmoothTimeCDFCal`, `kind="scdf"`,
**`time_coord="log_n_seen"`** (the corrected `t+1` coordinate — W4-E3 found it
neutral to six decimals and the pre-registered tie-break adopts it for
correctness at `t=0`). 12 log-spaced anchors, 256-point quantile grids,
`min_n=400`. Fitted on cross-fitted OOF over `folds_final10k`. **Anchors, grid
size and `min_n` are frozen and were never tuned.**

**Training data.** All **10,000** labelled series, partition
`research/folds/folds_final10k.parquet`,
`sha256(id,fold) = 6e9ebaf7cffada5bbf9e1d1f930f34c0b1fb2ce05287b46ab1303f6e89d8da65`.
The former 2,000-series lockbox is **training data** here, not an evaluation set.
It was spent for selection in waves 1–2 and no wave-4 score was taken from it.

**Deployment.** `INFER_PARALLELISM = 1`. P=4 segfaulted LightGBM in two workers
on the official macOS runner across three mitigation attempts; single-worker
projects ~4.9 h on the private set against a 15 h budget. Reliability over
parallelism.

**Environment.** Python 3.11.6, numpy 2.4.6, pandas 3.0.5, scipy 1.17.1,
scikit-learn 1.9.0, lightgbm 4.7.0, numba 0.67.0, pyarrow 25.0.1, macOS/arm64.

---

## 3. REJECTED ALTERNATIVES

| candidate | measured | why rejected |
|---|---|---|
| single model `RT-100` | 0.61605 | −0.0098 vs the ensemble, on every partition |
| seven seed clones | 0.62164 | −0.00417, 5/5 folds, CI clear of zero |
| union of both, 13 boosters (W4-E6) | 0.62486 | −0.00095; equal weighting gives the champion config 54% of the blend |
| 35 fold-models (W4-E4) | not scored | 3.927 ms/pt → 11.0 h projected, ~26% margin; no evidence of a gain; rejected on runtime risk |
| wave-1 original streams (W4-E5) | 0.62602 | +0.00021 vs incumbent — inside noise; no case for churning a Crunch-tested configuration |
| `m09_back` (wave 3) | +0.00156 | failed its own seed-clone control (+0.00496) |
| `m08_chan` | unverifiable | exists in no commit on any branch |
| within-timestep rank oracle `RT-131` | 0.62600 | **illegal** — needs the live cross-section; also no longer better than the legal blend |
| logit-mean / global-CDF / raw-mean blends | 0.62479 / 0.62550 / 0.62314 | all worse than SCDF on the specialist arm |
| `INFER_PARALLELISM=4` | — | segfaulted the official runner three times |

---

## 4. EXPECTED EXTERNAL PERFORMANCE

**Do not quote a development OOF as an expected leaderboard score.**

Development OOF is 0.62581 on the canonical partition. Adjustments:

* the canonical partition is the most favourable of four → mean across
  partitions is the better estimator of the delta;
* the private set is a different draw of series, with an unknown composition,
  tail mix and online-length distribution;
* the final model trains on 25% more series, which should help a little;
* every development number has been through a long selection process.

| scenario | estimate | reasoning |
|---|---|---|
| optimistic | **~0.625** | private set resembles the training distribution; the +20% training data offsets the favourable-draw component |
| **base** | **~0.615** | single-model level moves 0.0092 across *internal* partitions alone; an external draw should move at least as much, with the +0.0083 ensemble delta riding on top of a lower base |
| conservative | **~0.605** | composition shift, heavier tails, or a shorter online-length mix that concentrates weight in the young-break regime where TS-AUC is 0.513 |

The **delta** is the durable asset: +0.0083 ± 0.0011 across four partitions.
The **level** is not.

---

## 5. WHAT REMAINS UNKNOWN

* Private-set composition, tail mix and online-length distribution. Untestable
  by construction; `X_test.reduced` was never inspected for research.
* Whether the specialisation delta holds on an external draw. Four internal
  partitions all agree, and one of them sits below W4-E1's bar.
* Whether any of the eight untested alpha families would clear a seed-clone
  control. **NOT RUN** — see `STATE_OF_RESEARCH_V4.md`.
* The residual calibration mismatch: grids from 8,000-series models, boosters
  from 10,000-series models. Smaller than the status quo, not zero.

---

## 6. SIGN-OFF

Frozen at commit: recorded in `research/FINAL_REPRODUCIBILITY_MANIFEST.json`
after the final fit, which is the first action taken under this freeze.

**NO FURTHER MODEL SELECTION AFTER THIS POINT.**
