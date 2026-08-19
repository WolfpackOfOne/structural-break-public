# Platform constraints — from the official docs (checked 2026-08-19)

Sources: [competition docs](https://docs.crunchdao.com/competitions/competitions/structural-break-real-time)
and the [forum thread on cross-series state](https://forum.crunchdao.com/t/structural-break-real-time-may-infer-use-prior-completed-series-state/1186).

## 1. The metric is confirmed — our implementation is correct

> `TS-AUC = (Σ_t w(t)·AUC(t)) / Σ_t w(t)` where `w(t) = n_pos(t) · n_neg(t)`

This is **exactly** what `sbr/metric.py` implements, including the pair-count
weighting. The deep-research report flagged that the weighting could not be
verified from public material and warned that selecting against an unverified
objective would compromise everything downstream. **That risk is now closed.**
Every number in this project is measured against the right objective.

## 2. Runtime: 15 hours per week

> "The execution time of your solution should not exceed the platform's time
> limits: 15 hours per week."

Test set per the docs: 10,000 series public, 10,000 private. At the training
set's mean online length (~504 points), that is ~5.0M scoring points for the
public set and ~10.1M across both.

Budget = 54,000 s. Allowed cost per point, at parallelism `P`:

| test size | points | P=1 | P=4 | P=6 | P=8 |
|---|---|---|---|---|---|
| 10,000 series | 5.04 M | 10.7 ms | 42.9 ms | 64.3 ms | 85.7 ms |
| 20,000 series | 10.08 M | 5.4 ms | 21.4 ms | 32.1 ms | 42.9 ms |

**Measured today: 92.9 ms/point** for the seven champion modules on the correct-
but-slow reference streamer.

**Required speedup: ~2.2× at P=4 on the public set; ~4.3× at P=4 across both.**
At P=6 the reference streamer is already within ~1.4× of the public-set budget.

This is a **far easier target than assumed**. The plan was written around needing
roughly two orders of magnitude; the real requirement is single digits. The
dual-mode context already delivers 2× on `m00_core` without touching the
expensive modules, and `m02_dist`, `m07_bayes` and `m04_resid` — 67 % of the cost
between them — have not been optimised at all.

**Consequence: we probably do not have to cut ensemble members for speed.** The
"three streams instead of seven" trade should be re-examined only if the port
stalls.

## 3. Parallelism

Set a global `INFER_PARALLELISM = n`; the model starts `n` times in separate
processes, each handling a partition with isolated memory. Each process must
still process every point one at a time.

RAM scales with it — the docs' own example: "if you want 6 processes and your
model consumes 4 GB of RAM, the runtime must have 4 * 6 = 12 GB". Our per-series
state is the null-calibration engine at roughly a few MB, so per-process RAM is
dominated by the seven boosters, not by the features.

## 4. Determinism — and it independently confirms the blend decision

> "when re-run on 10% of the data, the predicted values should be the same
> (within a tolerance of 1e-8)"

Non-deterministic solutions are **ineligible for rewards**.

The forum thread asked whether `infer` may carry a summary of already-completed
series forward and use it on later ones. The official answer is that it is
mechanically allowed — but that it "will likely result in your code not being
deterministic based on when it starts", because the parallelism offsets differ
between the full run and the determinism-validation run, so predictions diverge
past tolerance.

This matters to us specifically: carrying cross-series state is the *only* way
one could approximate the within-timestep rank normalisation that `RT-131` used.
So the rank-average blend is not merely awkward to implement — pursuing it via
accumulated state would put the submission's reward eligibility at risk.
**The logit-average blend (`RT-160`) is the right answer on two independent
grounds**, and it costs nothing (0.62544 vs 0.62524).

Our inference path has no RNG and no cross-series state, so determinism holds by
construction. It still needs an explicit test.

## 5. What this changes

1. Metric-parity verification — **done, and it passed**. Highest-value cheap item
   on the roadmap, now closed.
2. Speed target — from "unknown, assume brutal" to **~2–4× at P=4**.
3. Stream count — no longer obviously a speed/accuracy trade. Keep all seven
   unless the port says otherwise.
4. New requirement — a **determinism test**: re-run inference on a 10 % sample and
   assert agreement to 1e-8.
