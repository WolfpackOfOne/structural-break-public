# Phase 0.8 — Full-scale local pilot

Date: 2026-08-31. Plan: `research/RT1320_PROMOTION_PLAN.md` §0.8.
Status: **PASS. Local execution is viable, and Phase 1 is ~3.7 h, not 39.5 h.**

Timing and memory only. Wrote nothing to `research/oof`; verified afterwards that
`nested_Q_outer0_inner1.npy` still carries its 2026-08-23 20:26 mtime.

## What was run

One full-scale inner teacher — 1,000,000 rows × 1000 columns × 900 rounds on the
frozen `ARM_B_PARAMS`, `num_threads=8` — with all three §0.7 levers and both
§0.8 memory changes applied, on the canonical partition.

The booster was then used to predict onto **both** fold 1 and fold 0,
demonstrating §0.7 lever 1 in situ: a single teacher serving both the
`(outer=0, inner=1)` and `(outer=1, inner=0)` roles.

## Correctness gate, run before the pilot

Chunked prediction vs whole-matrix prediction: **bitwise equal,
`max|diff| = 0.000e+00`**. The script aborts if this fails.

The full equivalence chain is therefore:

| change | verified | where |
|---|---|---|
| chunked in-place `augmented_stack` | identical sha256 at 250k and 1M | §0.7 lever 3 |
| `num_threads` 2 → 8 | predictions bitwise equal | §0.7 lever 2 |
| free X after `ds.construct()` | cannot affect results — LightGBM has already binned | by construction |
| chunked prediction | bitwise equal | this run |
| teacher `(f,g)` ≡ `(g,f)` | identical model string, predictions equal | §0.7 lever 1 |

**Not verified end-to-end at full scale.** Each link is checked individually;
no 1M × 900 run of the original path was made to diff against, because that is
the 39.5 min / 7.45 GB run the changes exist to avoid. The chain is sound but it
is a chain.

## Measured

| stage | wall | peak RSS |
|---|---:|---:|
| stack (chunked) | 15.2 s | 7.34 GB |
| Dataset construct + free X | 14.6 s | 7.56 GB |
| **train, 900 rounds** | **383.6 s** (0.426 s/round) | 7.56 GB |
| predict fold 1 (812,939 rows) | 17.5 s | 7.56 GB |
| predict fold 0 (806,334 rows) | 16.8 s | 7.56 GB |
| **total, one teacher serving both roles** | **7.5 min** | **7.56 GB** |

Swap was stable throughout and finished slightly better than it started
(8.44 → 8.19 GB used; free 774 MB → 1030 MB). The machine stayed usable.

## Projection vs reality

§0.5 projected **39.5 min** per inner teacher (0.640 s/round at 250k rows,
extrapolated linearly to 1M, at the frozen `num_threads=2`). Measured here:
**6.4 min of training**, 7.5 min including both predictions.

That is roughly **6×**, where §0.7 predicted about 2× from threads alone. Two
things are mixed together and this pilot cannot separate them:

1. thread scaling is much better at 1M rows than at the 120k rows §0.7 measured
   — more parallelisable work per tree, so 8 threads pay off far more; and
2. §0.5's linear row-extrapolation from 250k probably **over**-estimated the
   `num_threads=2` cost at 1M.

Separating them needs a `num_threads=2` run at 1M, which was not made because it
costs ~38 min to answer a question that does not change the plan.

**Correction to the record.** §0.7 stated the thread lever was worth 2.0×, and
corrected §0.5's earlier "~5×" as wrong. At full scale the 2.0× figure is itself
too pessimistic — it was measured at 120k rows and does not transfer. The
honest statement is that the *combination* of levers delivers ~6× against the
§0.5 projection, and the per-lever split at full scale is not established.

## Revised Phase 1 cost

| | measured |
|---|---:|
| one teacher, serving both its roles | 7.5 min |
| 10 distinct teachers, one partition | **1.2 h** |
| three alt partitions | **3.7 h** |

Against the original §0.5 figure of 39.5 h, that is a **~10.5× reduction**, all
of it from changes verified not to move predictions.

Still to add, and not measured here:

- student fits, 5 outer folds per partition at ~5 min each ≈ **0.4 h/partition**
- CatBoost refits for the champion lane — **not measured**, still unknown

So the teacher+student work is roughly **1.6 h per partition, ~4.9 h for all
three**. That fits one local evening, and no Crunch quota need be spent.

## Caveats

1. Run on the **canonical** partition. The alt partitions have near-identical row
   counts (folds are 806k / 813k / 807k / 803k / 804k) so the cost should carry,
   but it has not been demonstrated on `folds_alt*`.
2. Peak RSS 7.56 GB is above the ~4.7 GB §0.8 estimated. The estimate counted
   anonymous memory; the measurement is RSS, which includes evictable memmap
   pages. The machine held, but a second heavy process running alongside would
   not be safe.
3. One run, one machine, no repeat. Timing is not a promotion-relevant quantity,
   so this is proportionate — but do not quote 7.5 min as a tight figure.
4. `cmd_inner_teacher` still lacks its refuse-if-exists guard (§0.5). The pilot
   sidesteps it by writing nothing; a real Phase 1 run cannot.

## Artifacts

- `PHASE0_PILOT.json`, `PHASE0_PILOT.log`
- `research/scripts/rt1320_phase0_pilot.py`
