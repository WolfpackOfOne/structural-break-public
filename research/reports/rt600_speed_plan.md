# RT-600 inference speed plan

**Status: PLAN ONLY. Nothing here has been implemented.** Written while LB-003
(submission #3) was still running, at 2 h 20 m elapsed. It must not be acted on
while a scoring run is in flight.

**Scope discipline.** Every option below is a *performance* change. None of them
may alter a single score. The seven boosters, the 500 features, the SCDF
calibration and every seed stay exactly as frozen. The acceptance test for all of
this work is bitwise equality, not "close enough" — see §5.

---

## 1. Measured baseline

Profiled locally on the frozen artifact (6 series × 300 online points, model load
excluded), plus a direct predict benchmark. Numbers are this Mac; the cloud core
is slower, but the *proportions* are what matter.

| component | share of per-point cost | measured |
| --- | --- | --- |
| streaming feature engine (`engine.step`) | **~43%** | 2.28 s / 1800 pts |
| 7× LightGBM predict, one row at a time | **~24%** | **0.453 ms/point** |
| calibration + glue | remainder | |
| per-series historical warm-up (`fit_historical`) | — | **~0.23 s per series** |

Two facts dominate everything that follows.

**Fact 1 — we are using one core out of ~14.** The cloud job sits at a constant
~7% CPU, which is one saturated core on a ~14-vCPU runner. `INFER_PARALLELISM = 1`
is leaving roughly 93% of the machine idle. This is by far the largest available
win and it is not an optimisation at all — it is switching on capacity we are
already paying for.

**Fact 2 — single-row LightGBM predict is nearly all overhead.** Predicting one
row at a time costs 0.453 ms across the seven boosters. The identical work in
batches:

| batch | ms/point | speedup |
| --- | --- | --- |
| 1 (current) | 0.4530 | 1.0× |
| 64 | 0.0992 | 4.6× |
| 256 | 0.0758 | 6.0× |
| 1024 | 0.0628 | 7.2× |

At batch 256 the same trees over the same features cost **6× less per row**. The
difference is pure Python/ctypes call overhead — `__inner_predict_np2d` was the
single hottest frame in the profile.

**Memory, which constrains option A.** A loaded model costs **~162 MB**; a worker
process carrying interpreter + numpy + model measured **~380 MB RSS**.

---

## 2. Option A — process parallelism across series  ★ recommended first

Set `INFER_PARALLELISM` above 1 and let the runner fan series across workers.

**Why it is safe in principle.** Series are independent by construction: there is
no cross-series state anywhere in the engine, which is already asserted by the
production contract tests. Splitting series across processes cannot change any
score, because no score ever depended on another series.

**Why it is cheap.** It touches only the generated wrapper. `INFER_PARALLELISM`
lives in `CELL_ENTRY` in `build_submission.py`, not in `src/sbr` — so **the source
zip and model zip hashes come out bit-identical**, exactly as they did for the
boot-cell fix. That is a machine-checkable proof that the model did not change:

> source zip `199db8c9…` and model zip `6c8960dd…` unchanged ⇒ packaging changed,
> model did not.

**Expected gain:** near-linear in workers. P=4 → ~4×; P=8 → ~7–8×. A 2 h 10 m
inference becomes ~30 min at P=4, ~17 min at P=8.

**Risks, in order of seriousness:**

1. **Memory.** ~380 MB per worker. P=4 ≈ 1.5 GB, P=8 ≈ 3.0 GB, P=14 ≈ 5.3 GB —
   *plus* whatever the runner holds for a 216 MB test parquet. Local P=1 peaked at
   1.4 GB. We do not know the container's limit. An OOM here would look exactly
   like the stall I misdiagnosed earlier, so this is the risk to respect.
2. **The macOS segfault.** P=1 exists because "the official macOS Crunch runner
   segfaults LightGBM under forked P=4 workers". That is a macOS fork + OpenMP
   hazard and very likely does not apply to the Linux runner — but *likely* is not
   *verified*, and it cannot be verified locally on a Mac. This is the crux: the
   only machine that can answer it is the cloud.
3. Model loading is already lazy (`_load_model` on first use), so each forked
   worker builds its own copy after the fork rather than inheriting a half-initialised
   LightGBM handle. That is the correct shape for fork safety and is worth keeping.

**Recommendation: P=4, not P=14.** Four workers is a ~4× win for ~1.5 GB, well
inside any plausible container. Raise it only after a successful P=4 run proves
both the memory headroom and that Linux forking is safe. We have had three
deployment failures from assuming the runner resembles this laptop; this is the
same class of assumption and deserves the same suspicion.

---

## 3. Option B — batched prediction via series lockstep

Buffer K series, advance them in lockstep one time index at a time, and predict a
K-row batch per step instead of K single rows. Emit in the original order.

**Gain:** predict drops from 24% of runtime to ~4% — roughly a **20% overall**
reduction, on top of whatever Option A gives.

**Correctness:** each row in the batch still sees only its own series' prefix, so
per-series causality is untouched and output is bitwise identical.

**Costs and caveats:**
- Touches `src/sbr`, so the **source zip hash changes**. Still not a model change
  (model zip unchanged), but it forfeits the cheap proof and demands the full
  parity suite.
- K series of engine state held simultaneously.
- It reads series ahead of emitting their scores. No information flows between
  series, but it does bend the "real-time" streaming shape, and that is a
  judgement call worth making deliberately rather than by accident.

**Verdict: hold.** Do this only if Option A alone is insufficient. A 20% gain does
not justify restructuring the inference loop while a 4–8× gain sits untaken.

---

## 4. Option C — GPU offloading

**Recommendation: do not pursue this.** It is the wrong tool for this workload,
for three independent reasons:

1. **LightGBM's GPU support is for training, not inference.** There is no CUDA
   predict path that beats CPU for models of this size.
2. **The work is latency-bound, not throughput-bound.** After Option B, a batched
   predict step costs ~76 µs on CPU. A GPU kernel launch alone is ~5–10 µs and the
   host↔device transfer of a 256×500 float64 batch would cost more than the
   prediction saves. GPUs win on arithmetic intensity; tree traversal has almost
   none.
3. **The feature engine cannot use a GPU as written.** It is a sequential
   recurrence — each point's O(1) state update depends on the previous point's
   state. The only parallel dimension available is *across series*, which is
   precisely what Option A already exploits on the CPU, with vastly less risk.

Making a GPU pay would mean rewriting all eight feature modules as batch-vectorised
kernels and re-proving bitwise parity for every one of them, to chase a dimension
we can already exploit for free. The correct GPU decision here is not to make one.

---

## 5. Verification protocol — non-negotiable

Any change from this document must clear all of the following before submission:

1. **Bitwise identity.** Score the same series through the old and new paths and
   assert `atol=0`, not `allclose`. The repo already holds parity tests at exactly
   this standard; extend them rather than inventing a weaker bar.
2. **Hash discriminant.** Rebuild and compare. Source zip `199db8c9…` and model zip
   `6c8960dd…` unchanged proves the model is untouched. If either moves without a
   deliberate reason, stop.
3. **The existing release gates**, all of them: `validate_rt600_release.py` at 30/30,
   the four no-`n_online` causality gates actually running, the read-only-cwd gate,
   and the requirements-coverage gate.
4. **`crunch test`** with determinism check green at the new parallelism, six-file
   sentinel confirmed.
5. **A new submission**, recorded as LB-004 with its own entry in
   `rt600_baseline_submission.md`.

## 6. Staged rollout

| stage | change | expected | proves |
| --- | --- | --- | --- |
| 1 | `INFER_PARALLELISM = 4` | ~4×, ~30 min | Linux fork is safe; memory headroom |
| 2 | raise to 8 if stage 1 is clean | ~7–8×, ~17 min | headroom at scale |
| 3 | Option B, only if still needed | +20% | — |
| — | Option C | — | not pursued |

**Do not start stage 1 while LB-003 is running.** If LB-003 completes and scores,
we have our calibration point and speed becomes a convenience. If it is killed by a
wall-clock cap, stage 1 becomes the fix — and the timeout, not the leaderboard,
will be the reason. That distinction matters: changing parallelism because a job
timed out is an engineering response to an engineering failure, and it selects
nothing about the model.
