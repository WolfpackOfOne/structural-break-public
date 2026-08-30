# W4-E2 — DOES THE ENSEMBLE DELTA SURVIVE THE PARTITION DRAW?

**Answer: yes, and the study also shows why wave 2's version of it could not
have told us.**

39 training runs, every stream at its **own native configuration** (an earlier
draft would have forced them onto the reduced ABL protocol, which overrides
`num_leaves` / `min_data_in_leaf` / `feature_fraction` — the knobs that make a
specialist a specialist — and would have measured nothing). Only the fold
partition changes. Each partition uses only its own OOF vectors and its own
cross-fitted calibration; no number crosses a partition boundary.

Candidates were declared before any alternate score was read: exactly three.

---

## 1. THE NUMBERS

| partition | single | seed-clone 7 | specialist 7 | bagging Δ | specialisation Δ | total Δ |
|---|---|---|---|---|---|---|
| canonical | 0.61605 | 0.62164 | 0.62581 | +0.00559 | +0.00417 | +0.00975 |
| alt1 | 0.60985 | 0.61526 | 0.61780 | +0.00541 | +0.00254 | +0.00795 |
| alt2 | 0.61904 | 0.62364 | 0.62756 | +0.00460 | +0.00392 | +0.00852 |
| alt3 | 0.61314 | 0.61752 | 0.62020 | +0.00438 | +0.00269 | +0.00707 |
| **mean** | | | | **+0.00499** | **+0.00333** | **+0.00832** |
| **SD** | | | | 0.00059 | 0.00083 | 0.00113 |

**All twelve deltas are positive.** Neither pre-registered falsification
condition fired: no delta is negative on any partition, and no delta's
across-partition SD approaches its canonical value.

## 2. THE POINT OF THE WHOLE EXERCISE

| quantity | spread across partitions |
|---|---|
| single model **level** | 0.00918 (SD 0.00394) |
| total ensemble **delta** | 0.00268 (SD 0.00113) |
| specialisation **delta** | 0.00163 (SD 0.00083) |

**The level moves nearly four times as much as the total delta, and more than
five times as much as the specialisation delta.** A single model's absolute
score is dominated by which series happened to land in which fold; the ensemble
delta is a property of the method. Wave 2 measured the first quantity
(`RT-221/222/223`) and reported it as evidence about the second. It is not:
0.60985 on alt1 versus 0.61904 on alt2 says almost nothing about whether the
ensemble helps, and the delta says it clearly.

## 3. WHAT THIS COSTS THE HEADLINE — READ THIS BEFORE QUOTING ANY NUMBER

**The canonical partition is the most favourable of the four for the
specialisation claim.** Its +0.00417 is the maximum; the mean is **+0.00333**
and alt1 gives +0.00254.

Two honest consequences:

1. **On alt1 the specialisation delta would NOT have cleared W4-E1's
   pre-registered +0.0030 bar.** The W4-E1 verdict stands — it was a legitimate
   test on the pre-declared partition, decided on the pre-declared threshold,
   5/5 folds, with a bootstrap CI clear of zero. But had the canonical fold draw
   been alt1, that experiment would have returned INDETERMINATE. The effect is
   real and it is smaller than one partition made it look.

2. **Expected out-of-sample gain should be quoted as +0.0083, not +0.0098.**
   The mean across four partitions is the better estimator of what transfers;
   the canonical figure carries a favourable-draw component of about +0.0014.

**Bagging is the more robust half of the gain.** Its delta has the *smallest*
across-partition SD of the three (0.00059) and is the larger effect on every
partition. The seven-stream architecture's advantage over a single model is
therefore mostly attributable to the most boring mechanism available, and that
mechanism is also the most reliable one.

## 4. WHAT THIS DOES NOT LICENSE

The alternate partitions were used as a diagnostic and selected nothing. No
architecture, hyperparameter, stream, calibration family or composition was
chosen on the basis of an alternate-partition score, and none may be. In
particular alt2 — the most favourable partition on every measure — is not
evidence for anything.
