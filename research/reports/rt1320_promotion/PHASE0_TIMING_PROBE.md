# Phase 0.5 — What does one inner teacher fit cost?

Date: 2026-08-31. Plan: `research/RT1320_PROMOTION_PLAN.md` §0.5.
Status: **DONE — and the answer changes Phase 1's shape.**

Timing and memory only. **Produces no OOF vector, no score, and writes nothing
to `research/oof`.** Nothing here may be quoted as a result.

## Why this was run before the real thing

Phase 1 needs 20 nested inner teacher fits per alternate partition, 60 across
alt1/alt2/alt3. No timing evidence for `wave7_teacher_nested.py` existed anywhere
in the repo. The plan required measuring one before authorising sixty.

## Safety note — a real hazard found on the way

`cmd_inner_teacher` ends in

```python
np.save(f"{OOFDIR}/nested_Q_outer{outer_f}_inner{inner_g}.npy", Q)
```

with **no existence check and no `--force`**. In this worktree `research/oof` is a
symlink into `structural-break-wave8`, so running `--inner-teacher 0 1` as-is
would have silently overwritten the existing canonical
`nested_Q_outer0_inner1.npy` — a 40 MB gitignored file with no version-control
copy, and an input to RT-1320's entire evidence base.

The probe therefore never calls `cmd_inner_teacher`. It reconstructs the fit from
`ARM_B_PARAMS` + `augmented_stack` directly and saves nothing. Verified after the
run: `nested_Q_outer0_inner1.npy` mtime is unchanged at 2026-08-23 20:26.

**Before Phase 1 runs for real, `cmd_inner_teacher` needs a refuse-if-exists
guard.** This is a one-line fix and it should land before, not after, the first
production invocation.

## Method

The frozen architecture, at reduced scale, extrapolated linearly.

- Params: `ARM_B_PARAMS` unmodified — 900 estimators, 127 leaves,
  `min_data_in_leaf` 150, `feature_fraction` 1.0, `bagging_fraction` 0.8,
  `max_bin` 127, **`num_threads` 2**.
- Real fit: 1,000,000 rows × 1000 columns × 900 rounds.
- Probe fit: 250,000 rows × 1000 columns × 60 rounds, teacher `outer=0 inner=1`.
- Driver: `research/scripts/rt1320_phase0_timing_probe.py`.

Population, for the record: 5,036,517 rows total; folds are 806,334 / 812,939 /
806,691 / 802,506 / 804,054. Three training folds give 2,413,251 rows, which
`MAX_TRAIN_ROWS` subsamples to exactly 1,000,000.

## Measured

| quantity | probe (250k × 60) |
|---|---:|
| `augmented_stack` | 17.1 s |
| matrix | (250000, 1000) float32, 0.93 GB |
| train, 60 rounds | 38.4 s |
| **per round** | **0.640 s** |
| peak RSS | 4.22 GB |

## Projected to the real fit

| quantity | projection |
|---|---:|
| one inner teacher | **39.5 min** (stack 1.1 + train 38.4) |
| 20 inner teachers, one partition | **13.2 h** |
| 60 inner teachers, three alt partitions | **39.5 h** |
| feature matrix at full scale | 3.73 GB |
| peak during `np.concatenate` | **~7.45 GB** |

That 39.5 h is the **inner teachers alone**. It excludes the five outer-fold
student fits per partition, the RT-1320 student re-runs (~5 min per outer fold,
already measured historically), and the CatBoost refits.

## Two conclusions Phase 1 has to absorb

**1. This is not an afternoon.** The alternate-partition leg is 40+ hours of
single-machine compute at the frozen configuration. Open decision 4 in the plan
("where do the teacher refits run") is therefore not a convenience question, it
is a blocker, and it should be settled before Phase 1 is authorised.

**2. The RT-600-lane fallback does not avoid this cost.** The plan offered
running the alt leg on the RT-600 lane only, since every specialist and seed-clone
vector already exists there. That fallback avoids the *CatBoost* refits — it does
**not** avoid the teacher refits, because the student's training target is derived
from the nested Arm-C teacher on whichever partition it is fitted. The 39.5 h is
common to both lanes. Choosing the cheaper lane saves the CatBoost work and
nothing else.

## Memory — the harder constraint

Peak is ~7.45 GB for the feature matrix alone (3.73 GB result plus a 3.73 GB
`own`+`fut` transient during `np.concatenate`), before LightGBM's Dataset. The
same peak recurs at prediction time when `Xva` is built over the ~806k held-out
fold rows.

This machine has **16 GB total with swap already at 5.56 GB of 7.17 GB**. The
full-scale fit would sit close to the edge and would likely swap, which would
both invalidate the timing and make the machine painful to use. That is why this
probe was run at 250k rows rather than 1M.

`REPRODUCIBILITY_MANIFEST.json` records the research-era environment as a 2-CPU
Linux box, which is consistent with `num_threads=2` being the frozen value.

Two things follow, and they are different questions:

- **RAM.** The full-scale fit wants materially more headroom than 16 GB gives
  once swap is accounted for. This is the binding constraint.
- **Threads.** `num_threads=2` on a 10-core machine leaves ~5× on the table. But
  raising it is a **protocol change, not a free win**: this program's
  determinism discipline (`wave3_determinism.py`, the bitwise-replay gate in
  `PROTOCOL_CHAMPION_2026.md`) means a thread-count change has to be shown not
  to move predictions before it is used for anything that feeds a decision. Do
  not treat the 5× as available until that is demonstrated.

## Caveats on the projection

1. **It is an extrapolation, not a measured full fit.** LightGBM histogram
   construction is close to linear in rows, but at 1M rows × 1000 features with
   2 threads, cache behaviour can degrade, so the true cost is more likely to be
   above 39.5 min than below it. Treat 39.5 h as a floor for the 60 fits.
2. Per-round cost was measured over 60 rounds. Early rounds can be cheaper than
   late ones as trees deepen, which again biases the estimate low.
3. The probe subsamples with a different RNG draw than the real fit. Irrelevant
   for timing, and it produces no score, so it cannot leak into any result.

## Environment

python 3.11.6, numpy 2.4.6, pandas 3.0.5, lightgbm 4.7.0, macOS-26.5.2-arm64,
10 cores, 16 GB. Machine checked clear of other jobs before launch (load 1.18,
no python/lightgbm/catboost processes). Swap unchanged across the run.

## Artifacts

- `PHASE0_TIMING_PROBE.json` — measured and projected figures
- `PHASE0_TIMING_PROBE.log` — full console output
- `research/scripts/rt1320_phase0_timing_probe.py` — the driver
