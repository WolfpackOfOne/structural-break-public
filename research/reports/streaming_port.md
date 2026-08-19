# Streaming port — status

**The submission blocker.** Every feature module is batch-first: it computes a
whole trajectory from cumulative sums. The runner is series-sequential and
single-pass and wants one score per observation. Until that gap closes, the
0.6254 is unshippable.

## Done

**1. A provably-correct streaming path exists.** Prefix invariance — which every
module already passes bitwise — means recomputing the batch builder on the
growing online prefix and taking the last row is *by definition* what the batch
pipeline produced. `ReferenceStreamer` does exactly that and is **bitwise
identical to the batch matrix** on every series tested (n_online 34, 416, 811).
It is far too slow to ship, and it is the parity oracle for everything faster.

**2. Historical state is separated from per-step work.** `HistParams` and the
null-calibration engine depend only on the historical segment, so they are built
once per series (~150 ms) instead of per observation. Verified bit-identical to
the previous single-call construction for all 8 modules.

**3. The context is dual-mode.** `SeriesCtx.pos` switches `roll`/`expand`/`idx`
between full-trajectory and single-row evaluation, backed by incrementally
maintained cumulative sums. **The module code is unchanged and identical in both
modes**, so batch/stream parity is structural rather than a property to be
re-established by hand for every feature.

**4. `m00_core` (151 columns) is ported**, bitwise identical to the batch matrix
under the O(1) path on three series including an 811-point one.

**5. Tests.** `tests/test_streaming.py` — reference parity at n_online ∈ {1,2,5,40},
incremental parity, and a single-online-point smoke test across all 8 modules.

**6. A real bug fixed.** The lag-2 transform was built by concatenation and
returned length 2 instead of length 1 when the online segment had a single point
— invisible in batch mode (every series has ≥10 online points), an immediate
crash on the first streamed observation. Now built by explicit assignment, and
verified bit-identical to the cached features on 12 series.

## Where the time goes

Measured on a 416-point series against a 1,192-point history, all 7 champion
modules:

| component | share | ms/point |
|---|---|---|
| `m02_dist` | 29.7 % | 27.59 |
| `m07_bayes` | 19.0 % | 17.63 |
| `m04_resid` | 17.9 % | 16.60 |
| `m03_dyn` | 12.3 % | 11.46 |
| `m06_loc` | 9.8 % | 9.14 |
| `m00_core` | 5.7 % | 5.29 |
| `m01_seq` | 5.1 % | 4.78 |
| context rebuild | 0.4 % | 0.40 |
| **total (reference)** | | **92.88** |

The context rebuild is negligible; essentially all of it is modules recomputing
their whole trajectory at every step.

For `m00_core`, the O(1) path takes 6.23 → 2.94 ms/point. **Only 2×** — and the
profile explains why: the remaining cost is not algorithmic but **per-column
numpy call overhead**. Per emitted row, `m00_core` alone makes ~48 `nullcal.z`
calls, ~30 `nullcal.pct` calls and ~129 `clip` calls, each on a length-1 array.
The arithmetic is trivial; the dispatch is the cost.

## What remains

- **Six modules still to port.** They read `ctx.tr` directly (which single-row
  mode does not populate) and several carry genuine sequential recursions —
  running peaks, CUSUM paths, Bayesian filters — which need explicit state rather
  than the stateless trick. `m06_loc` additionally fails parity at row 7 today.
- **Batch the per-column operations.** One vectorised `z`/`pct` call per step over
  all (transform, window) pairs instead of ~80 separate ones, or a numba-compiled
  row builder. This is where the remaining order of magnitude is.
- **Decide the stream count first.** Each of the seven ensemble members multiplies
  the inference bill. Three streams at ~0.624 may be the better trade than seven
  at ~0.625 — that choice should be made *before* the porting effort is spent.
- **The runtime cap is still unknown**, so there is no target to optimise against.

**Honest completion estimate: roughly the first fifth.** The foundation, the
correctness oracle, the parity harness and one of seven modules are done; the
expensive modules are untouched.
