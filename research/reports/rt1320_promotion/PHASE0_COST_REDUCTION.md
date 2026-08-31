# Phase 0.7 — Three levers that make Phase 1 lighter

Date: 2026-08-31. Plan: `research/RT1320_PROMOTION_PLAN.md` §0.7.
Status: **two verified exact, one verified exact but less useful than hoped.**

Engineering only. Every lever here is required to leave predictions **bitwise
unchanged**; none of them is permitted to alter the science. Levers that would
alter it are listed in §4 and rejected.

## Lever 1 — the nested teacher fits every model twice (exact 2×)

`train_inner_teacher(outer_f, inner_g)` computes

```python
train_folds = [x for x in FOLDS if x not in (outer_f, inner_g)]
```

which depends only on the **set** `{outer_f, inner_g}`. The row subsample uses a
fixed `default_rng(0)`, and `ARM_B_PARAMS` carries no per-fold seed. So the
teacher for `(0,1)` and the teacher for `(1,0)` are the same model; only
`va_rows = d.rows_for([inner_g])` differs.

Verified directly:

```
train_folds identical : True        ([2, 3, 4] for both)
train rows identical  : True        (120,000 rows)
model string identical: True
predictions bitwise equal: True     max|diff| = 0.000e+00
```

Across the scheme `build_nested_Q` runs over all 20 ordered `(outer, inner)`
pairs, but those contain only **C(5,2) = 10 distinct training sets**, each fitted
twice:

| trained on | fitted for |
|---|---|
| [2,3,4] | (0,1) and (1,0) |
| [1,3,4] | (0,2) and (2,0) |
| [1,2,4] | (0,3) and (3,0) |
| [1,2,3] | (0,4) and (4,0) |
| [0,3,4] | (1,2) and (2,1) |
| [0,2,4] | (1,3) and (3,1) |
| [0,2,3] | (1,4) and (4,1) |
| [0,1,4] | (2,3) and (3,2) |
| [0,1,3] | (2,4) and (4,2) |
| [0,1,2] | (3,4) and (4,3) |

**Fix:** fit the 10 distinct teachers once each and predict each onto *both* of
its two held-out folds. **13.2 h → 6.6 h per partition, output bit-identical.**

**Care required.** This refactors a fold-purity-critical function. The purity
sentinel (`--fold-purity-test`) must still pass afterwards, and the §0.5
refuse-if-exists guard on `nested_Q_*` writes still applies. The saving is real
but it is not a one-line change.

## Lever 2 — thread count (2×, and it does not move predictions)

`ARM_B_PARAMS` pins `num_threads=2`, matching the 2-CPU Linux research box
recorded in `REPRODUCIBILITY_MANIFEST.json`. Measured at 120k rows × 25 rounds:

| num_threads | train | speedup |
|---|---:|---:|
| 2 (frozen) | 16.0 s | 1.00× |
| 4 | 9.2 s | 1.74× |
| 8 | 7.9 s | **2.02×** |
| 10 | 7.9 s | 2.03× |

Saturates at 8; the 9th and 10th cores buy nothing.

**Determinism.** Predictions are **bitwise equal** to the frozen `num_threads=2`
at 4, 8 and 10 (`max|diff| = 0.000e+00`). `model_to_string()` differs, but the
diff is exactly two lines — `[num_threads: 2]` vs `[num_threads: 8]` — i.e.
recorded metadata, not tree structure.

This is the demonstration `PROTOCOL_CHAMPION_2026.md`'s determinism gate
requires. **Two caveats before relying on it:** it was run at reduced scale
(120k × 25, not 1M × 900), and on one machine. Repeat at full scale before it
feeds a decision.

**Correction to the record.** §0.5 stated the thread change was worth "~5x".
Measured, it is **2.0×**. The earlier figure was inferred from core count and
was wrong.

## Lever 3 — memory: chunked in-place stack (exact, but not the whole story)

`augmented_stack` ends in `np.concatenate([own, fut], axis=1)`, which holds
`own` (n×500) + `fut` (n×500) + the result (n×1000) simultaneously — 7.45 GB of
**anonymous** memory at n=1,000,000.

Replacing it with a preallocated result filled in 100k-row blocks removes the
transient. Pure data movement, no arithmetic, so equality is exact:

| n=250,000 | time | peak RSS | sha256 |
|---|---:|---:|---|
| original | 20.7 s | 4.18 GB | `c1f9dbcc…f7b8` |
| chunked | 14.5 s | 4.10 GB | `c1f9dbcc…f7b8` |

Identical hash, and ~30% faster.

**But peak RSS barely moved, and that is the finding.** At 250k rows, RSS is
dominated by resident **memmapped feature-bank pages**, not by the array. Those
are clean, file-backed and evictable; the concatenate transient is anonymous and
is what actually drives swap. The saving is real but invisible at this scale.

At full scale, chunked:

```
n=1,000,000  shape (1000000, 1000) float32  array 3.73 GB
time 25.6s   PEAK_RSS 5.85 GB
```

Under the 7.45 GB the original would reach — but **swap went from 4.84/6.14 GB
used to 8.90/9.22 GB, 318 MB free**, and macOS grew the swapfile to cope. That is
the stack *alone*, before LightGBM's Dataset or the prediction matrix.

## What this means for running locally

Honest budget for one full inner teacher on this 16 GB machine:

| stage | anonymous memory |
|---|---:|
| feature matrix (chunked) | 3.73 GB |
| + Dataset construction, before freeing X | ~4.7 GB |
| after `ds.construct()` and `del X` | ~1.0 GB |
| prediction matrix over ~806k held-out rows | ~3.0 GB |

Peak ~5 GB anonymous plus evictable memmap residency. **Viable but tight**, on a
machine already swapping. Two further changes would make it comfortable, and
neither touches the science:

1. **Free the float32 matrix once the Dataset is built.** After
   `ds.construct()`, `del X` — LightGBM has already binned to ~1 byte/value at
   `max_bin=127`.
2. **Chunk the prediction** over `va_rows` instead of materialising a second
   3.0 GB matrix.

The deeper fix, if those are not enough, is to build the Dataset from
`lgb.Sequence` batches so the 3.73 GB float32 array never exists at once. That
is more invasive and should not be attempted before the two above are tried.

## Combined arithmetic

| | per partition | three alt partitions |
|---|---:|---:|
| as written (§0.5) | 13.2 h | 39.5 h |
| + lever 1 (dedup, exact) | 6.6 h | 19.8 h |
| + lever 2 (8 threads, exact) | **~3.3 h** | **~9.9 h** |

Both figures are extrapolations from reduced-scale measurements and should be
treated as targets to confirm, not as facts. **Confirm with one full-scale
timed inner teacher before committing to a batch.**

## 4. Levers considered and rejected

- **Reduce `MAX_TRAIN_ROWS` below 1,000,000** — would make the alt partitions
  non-comparable to canonical, which used 1M. Canonical would have to be re-run
  at the reduced budget too, which costs more than it saves.
- **Reduce `n_estimators` below 900** — same objection.
- **Reduce the inner-fold count** — breaks the fold-purity contract the entire
  RT-1320 result rests on. Not available at any price.
- **Use the RTX 4090** — irrelevant. The teacher is CPU LightGBM; using the GPU
  means `device=gpu`, a protocol change with its own determinism burden.

## Artifacts

- `research/scripts/rt1320_phase0_dedup.py`, `PHASE0_DEDUP.log`
- `research/scripts/rt1320_phase0_threads.py`, `PHASE0_THREADS.log`
- `research/scripts/rt1320_phase0_stack_mem.py`
