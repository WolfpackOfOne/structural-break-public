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
