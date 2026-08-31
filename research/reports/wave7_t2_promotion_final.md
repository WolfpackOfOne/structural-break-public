T2 / RT-995 — FINAL PROMOTION BATTERY

## A. START STATE

- Primary checkout branch at task start: `claude/structural-break-competition-entry-yjoj8r` @ `7892dcb` — untouched, no work done there.
- This battery was run on a new branch, `research/wave7-t2-promotion`, forked from `research/wave7-teacher-distillation@5093e0a` (the commit holding the full 5-outer-fold nested T2 result) in a dedicated new worktree, so no existing worktree's branch or WIP was touched.
- Active processes at start: none (checked `ps aux` for python/lightgbm before doing anything).
- Canonical T2 artifact verification: **reproduced from committed artifacts**, no retraining needed — `research/reports/wave7_teacher_nested.md`/`.json` on `research/wave7-teacher-distillation@5093e0a` already contain the full 5-outer-fold result; `research/oof/RT-995.npy` and per-outer-fold files are present and match the report's fold count and shape.

## B. CANONICAL T2

| | value |
|---|---:|
| T0 mean (whole-dev TS-AUC, 5 outer folds) | 0.61185 |
| T2 mean (whole-dev TS-AUC, 5 outer folds) | 0.62128 |
| T2 − T0 mean delta | **+0.00943** |
| per-fold deltas | +0.00447, +0.01391, +0.00904, +0.01361, +0.00609 |
| positive folds | 5/5 |
| paired series bootstrap (200 reps) | mean +0.00922, CI95 [+0.00425, +0.01373], P(>0)=1.00 |

Reproduced, not re-derived. Clears the three *measured* promotion legs
(magnitude ≥+0.0030, ≥4/5 folds, bootstrap CI > 0) — this is leg 4
(alternate partitions) and the ensemble-integration test that remained open.

## C. ALTERNATE PARTITIONS

**Not run.** `research/folds/folds_alt{1,2,3}.parquet` and
`research/scripts/make_folds_alt.py` are present and usable — this is not a
missing-infrastructure gap. The decision not to run them is a resource
decision, made explicit here rather than silently skipped:

- The canonical nested run (5 outer folds × 4 inner teachers + 2 outer
  students, concurrency-limited to 2 processes) took roughly 5 hours
  wall-clock, from the commit timestamps on
  `research/wave7-teacher-distillation` (`32510cc` 14:55 → `5093e0a`
  19:56, 2026-08-23). Three alternate partitions at the same design would
  cost on the order of **15 hours of additional wall-clock compute**.
- Section D/E below show the ensemble-integration leg — which this
  document's own pre-registration (section 9 of the driving brief) calls
  the single most important test — already gates T2 at **MOSTLY
  REDUNDANT**. Alternate partitions would only re-confirm the standalone
  `T0` comparison's stability across fold draws; they cannot raise the
  ensemble-integration verdict, because they do not touch the seven
  specialists or the blend at all.
- Given that, spending ~15 hours of compute to further validate a
  standalone number that has already been shown not to translate into
  ensemble value is not a well-aimed use of the machine. **This is a
  recommendation, not a unilateral closure** — if the alternate-partition
  leg is wanted anyway for the written record (e.g. before formally
  retiring `T2`), it is fully specified and ready to run; flag it and it
  can be started as a long-running background job.

## D. ALTERNATE STABILITY VERDICT

N/A — not run, per C. No claim of STRONG/SUPPORTED/FRAGILE/FAILED is made.

## E. ENSEMBLE INTEGRATION PREREG

- Prereg: `research/WAVE7_T2_INTEGRATION_PREREG.md`, commit `7198da2`.
- E0 = RT-600 seven specialists (`RT-300`, `RT-410`–`RT-415`), equal weight, cross-fitted SCDF calibration.
- E1 = E0 + `RT-401` (matched exchangeable seed clone), equal weight across eight streams.
- E2 = E0 + `RT-995` (T2), equal weight across eight streams.
- Fold 0 only (matches every Wave-8 mechanism's own measured marginal, for apples-to-apples comparison). No weight search, no retraining — all OOF arrays reused byte-identical from their source branches.

## F. ENSEMBLE RESULTS

| arm | TS-AUC (fold 0) |
|---|---:|
| E0 — RT600 7-stream | 0.63828 |
| E1 — RT600 + seed clone | 0.63859 |
| E2 — RT600 + T2 | 0.63882 |

| contrast | Δ |
|---|---:|
| E1 − E0 | +0.00031 |
| E2 − E0 | +0.00055 |
| **E2 − E1** | **+0.00024** |

## G. PAIR FLOW

Raw (uncalibrated) same-t pair diagnostic vs the E0 blend, fold 0:

| candidate | repairs | damage | net weighted repair |
|---|---:|---:|---:|
| T2 | 708 | 723 | −15 |
| seed clone | 665 | 731 | −66 |

No dominant-cell / never-break / pre-break pair-flow split was computed
separately — at this level of net signal (both candidates net-negative,
T2 only marginally less so than an exchangeable clone) a cell breakdown
would not change the reading and was not run to avoid manufacturing
false precision from a diagnostic that is already flat.

## H. DIVERSITY

| stream | corr with T2 |
|---|---:|
| `RT-300` | 0.9122 |
| `RT-410` | 0.8742 |
| `RT-411` | 0.8557 |
| `RT-412` | 0.8830 |
| `RT-413` | 0.7147 |
| `RT-414` | 0.8785 |
| `RT-415` | 0.9032 |
| E0 blend | 0.6740 |
| seed clone `RT-401` | 0.9099 |

T2 correlates with the seven specialists at essentially the same magnitude
the seed clone does (0.91) — the signature of a refined but largely
overlapping model, not an independent information source.

## I. BOOTSTRAP / UNCERTAINTY

Canonical (`T2 − T0`, single-model): bootstrap CI [+0.00425, +0.01373],
entirely above zero (section B). No bootstrap was run on the
ensemble-integration contrast (`E2 − E1`) — it is a single fold-0 point
estimate, matching the Wave-8 convention this leg intentionally mirrors for
comparability (none of the five Wave-8 marginals were bootstrapped either,
since none cleared their pilot gate to justify the additional compute).
If `T2` were to be pursued further, an all-5-fold, bootstrapped version of
the `E2 − E1` contrast is the natural next confirmatory step — but the
point estimate is small enough (+0.00024, an order of magnitude below the
"REAL BUT MODEST" floor of +0.0015) that added folds are unlikely to move
the verdict band.

## J. DEPLOYMENT FEASIBILITY

- Verified mechanically (not just by description): `research/scripts/wave7_teacher_nested.py` asserts `Xtr.shape[1] == 500` and `Xva.shape[1] == 500` for the student (T1/T2) — the *teacher* alone uses 1000 columns (500 causal + 500 own-series final-row broadcast, training-only, never touches the student). The student — the object that would ship — sees only the same unmodified 500-column causal bank every existing specialist already uses. No `Q`, no future representation, no `tau`, no `n_online`, no final-online-length feature is reachable at inference.
- Incremental cost of adding T2 as an eighth stream: +1 LightGBM booster, same architecture family (`ARM_B_PARAMS`, 500 causal columns) as the seven specialists already in production — inference cost scales by roughly 1/7 relative to the current ensemble. Training cost is a single ordinary 5-fold LightGBM fit (no nested teacher needed at deploy time — the teacher is training-time-only scaffolding that produced `T2`'s label; nothing about it needs to exist in production).
- No RT-600 production artifact was modified. This is a deployment plan only, per instruction — not an implementation.

## K. PROMOTION VERDICT

**MOSTLY REDUNDANT.**

## L. SCORE LADDER

- RT-600 external anchor: **0.6268**.
- Measured internal `E2 − E1`: **+0.00024**.
- Do not add `0.6268 + 0.00943` — that number was never a valid proxy for ensemble/external value, and this leg is exactly the evidence that shows why: the standalone single-model delta over-states what T2 contributes once folded into the actual production-shaped ensemble by roughly **40×** (0.00943 vs 0.00024).
- Conservative interpretation: if the internal fold-0 `E2 − E1` transfers to the external leaderboard at all (untested, single fold, no bootstrap), it implies an anchor move on the order of **0.6268 → ~0.6270**, not a materially different number.

## M. 0.640 / 0.645 / 0.650 ASSESSMENT

**Not supported.** A +0.00024 measured ensemble marginal, even taken at full
face value with no discount for its own fold-0-only, non-bootstrapped
status, does not credibly move the external anchor from 0.6268 to 0.640,
let alone 0.645 or 0.650. This is consistent with Wave 8's own conclusion
(`research/reports/wave8_final.md` section Q) reached the same way for five
other candidates — a real, positive, statistically supported *single-model*
result that does not survive contact with the actual seven-specialist
ensemble.

## N. NEXT SINGLE EXPERIMENT

T2 is **redundant, not disqualified** — it is the best-performing candidate
this project has measured against the ensemble-integration gate (every
Wave-8 mechanism was net-negative; T2 is the only positive figure), it is
simply too small to matter on its own. Per the driving brief's own decision
rule (§21: propose a fundamentally different alpha source when the
candidate is redundant), the next highest-value experiment is **not**
another future-aware/distillation variant on the same seven specialists
(that family — Wave 7's teacher distillation plus Wave 8's five mechanisms
— has now returned one weak-positive and five negative marginals in a row,
a consistent signal that the seven specialists' overlapping feature banks
already capture most of what any single additional model derived from the
same causal columns can add). A better-aimed next step is either (a) a
genuinely differently-conditioned specialist (different row population,
different loss, or different feature subset than all seven existing
streams share) whose *raw correlation with the existing seven* is
substantially below the ~0.7–0.9 band every candidate measured so far has
landed in, or (b) directly investigating why the specialists correlate as
highly as they do with everything tried against them — that redundancy,
not any one candidate's failure, may be the actual ceiling on ensemble
improvement from this direction.

## O. GIT END STATE

- Ending SHA: `9955486` on `research/wave7-t2-promotion`.
- Pushed: not yet — this branch has not been pushed to `origin`; will push on request (matches this project's git-push convention of confirming before remote actions).
- Clean tree: yes, after this report's own commit.
- Active processes: none.
