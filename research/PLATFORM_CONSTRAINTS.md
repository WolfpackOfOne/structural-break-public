<!-- Recovered 2026-09-01 from the legacy `structural-break` worktree, where it
had sat untracked on branch research/investigation-20260829-unresolved-questions
since 2026-08-19. It existed nowhere on main. Content is unchanged from the
original; only this note was added. -->

# Platform constraints — from the official docs (checked 2026-08-19)

Sources: [competition docs](https://docs.crunchdao.com/competitions/competitions/structural-break-real-time)
and the [forum thread on cross-series state](https://forum.crunchdao.com/t/structural-break-real-time-may-infer-use-prior-completed-series-state/1186).

## 0. Scoring structure — the public feedback set is NOT the prize set

**Added 2026-09-03.** Source: clarification from **Emanuele Olivetti (ADIA)** on
the competition's scoring process, relayed by the repository owner. This is not
a URL that has been fetched and verified in-repo; it is recorded here as owner-
supplied authoritative information because it is the single most consequential
operational fact in the project and was previously documented nowhere.

**The process:**

1. During the competition, **repeated submissions are scored on the same public
   test set.** This is leaderboard feedback.
2. When the competition closes, **each participant's SELECTED submission is
   evaluated once on the unseen private test set.**
3. **That private score determines the prizes.**
4. **It is NOT automatically your latest submission.** Selection is a required,
   explicit action.

### Consequence 1 — selection is the highest-leverage action in the project

An unset or stale selection forfeits more than every remaining modelling gain
combined. The measured external ladder is RT-600 0.6268 → RT-1257 0.6290 →
RT-1320 0.6303; the difference between selecting the best and the worst of those
is 0.0035, larger than any candidate now in reach. **Verify the selection
explicitly. Do not rely on a default.**

### Consequence 2 — public deltas are PAIRED, and are better evidence than they look

Because every submission is scored on *the same* public series, a difference
between two submissions is a paired comparison on identical data and the
series-sampling noise largely cancels. RT-1320's +0.0013 over RT-1257 is
therefore a more precise estimate of the ordering than an unpaired comparison of
two independent 10,000-series draws would be.

### Consequence 3 — but a fixed, repeatedly-queried set is a selection trap

The corresponding hazard is not noise in a single delta; it is **multiplicity**.
A fixed public set that can be queried repeatedly is exactly the configuration in
which climbing the public leaderboard by selecting on that draw's idiosyncrasies
produces a drop on the private set. Every submission is another look at the same
data.

**The existing protocol is the correct defence and must not be relaxed.** Every
promotion in this project was decided on dev folds *before* the external number
existed — RT-1320 passed its four-partition gate on 2026-09-01, two days before
submission #19 scored on 2026-09-03. `STATUS.md`'s standing instruction, "one
useful external calibration point, not a license to tune on the leaderboard", is
the rule that protects the private score. The project currently carries
essentially zero public-set selection debt; that is an asset, not an accident.

**Decision rule going forward:** decide on dev, use public only to confirm. A
candidate that is flat on dev and strongly positive on public is the signature of
public-set overfitting and must not be selected over an incumbent.

### Consequence 4 — what transfers to private is the DELTA, not the level

`FINAL_ARCHITECTURE_FREEZE.md` §4 already states this for the earlier lane: "The
delta is the durable asset. The level is not." Its base-case estimate for the
*level* on an unseen draw was ~0.615, well below the public 0.6303. A lower
private level is expected and is not evidence of a problem; the ranking is what
pays.

RT-1320's delta over RT-1257 is stable across every population it has been
measured on — canonical +0.0015159, alt1 +0.0021146, alt2 +0.0014825, alt3
+0.0019582, public (release 234) +0.0013. Five draws, five positive, spread
~0.0006.

### Consequence 5 — a further candidate is a free option, not a risk

Because only the *selected* submission is privately scored, building another
candidate cannot damage the final score. Build it, decide it on dev, submit it,
and select it only if dev and public agree; otherwise select RT-1320 and lose
nothing. The costs are compute and submission quota only.

### Terminology note — this resolves an inconsistency in the repo

§2 below quotes the docs as "10,000 series public, 10,000 private", while
`FINAL_ARCHITECTURE_FREEZE.md` §4 discusses the external score as though it were
already the private set. **§0 is authoritative on the process**: external scores
recorded in this repository (0.6268, 0.6290, 0.6303) are **public-set** figures
and are leaderboard feedback. No private-set score has been observed, and none
will be until the competition closes.

### Still unverified

- The submission quota, and whether selection can be changed after it is set.
- Whether the private set is a fresh 10,000-series draw or a rolling accumulation
  over data releases. Both submissions #16 and #19 ran against `data release 234`,
  so all comparisons in this repo are same-release and clean.

---

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
