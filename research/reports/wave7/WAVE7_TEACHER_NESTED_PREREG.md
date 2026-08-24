# WAVE 7 — TEACHER/DISTILLATION, NESTED OUTER-FOLD-PURE, PRE-REGISTRATION

**Written 2026-08-23 on `research/wave7-teacher-distillation`, committed
BEFORE any nested teacher or nested student score exists.** This document
**amends** `research/WAVE7_TEACHER_PREREG.md` — it does not silently replace
it. The prior document and its result
(`research/reports/wave7_teacher_pilot.{md,json}`, `RT-992`/`RT-993`) remain
on disk, unmodified in substance, and are now marked
**MECHANISM-POSITIVE SCREEN, OUTER-FOLD CONTAMINATED, NOT PROMOTION
EVIDENCE** at the top of both files.

## 0. THE DEFECT

`research/WAVE7_TEACHER_PREREG.md` §1.1 argued that `Q` (`RT-991`'s OOF) was
"already cross-fitted at the series level" because a row's `Q` value never
came from a teacher trained on that row's own series. **This is true but
insufficient once `Q` becomes a training LABEL for a second, outer-CV model
(the student).** The relevant guarantee for a nested estimator is not "the
teacher never trained on this row's series" — it is **"no teacher that
generated any label used in the student's training set trained on the
student's OUTER VALIDATION fold."** `RT-991` was cross-fitted only against
its own row/series, not against the student's outer split, and both use the
*same* 5-fold partition. Concretely, for outer fold 0 (the pilot's only
evaluated fold): a fold-1 row used in student training received `Q` from
`RT-991`'s fold-1 teacher, itself trained on folds `{0,2,3,4}` — including
outer fold 0. Fold 0 information reached the student's training signal
before the student was ever scored on fold 0. This is the standard
stacked-ensemble meta-feature leakage pattern (base-model OOF must be
re-cross-fitted against an outer split before it can safely become a
second-level model's *feature*, and the same logic applies when it becomes a
second-level model's *label*), not a production-causality failure — the
trained student never sees `Q` at inference, only the 500 causal columns.

## 1. THE FIX — NESTED (DOUBLE) CROSS-FITTING

For every **outer** validation fold `f ∈ {0,1,2,3,4}`:

* `outer_train` = the other four folds.
* For every **inner** held-out fold `g ∈ outer_train` (four folds):
  * train an Arm-C-architecture teacher (`wave7_d3r.augmented_stack` /
    `last_row_lookup`, `ARM_B_PARAMS`, `objective="binary"`, same
    `MAX_TRAIN_ROWS=1,000,000` cap, `seed=0`) on folds
    `{0,1,2,3,4} \ {f, g}` — **three** folds, never `f` and never `g`;
  * predict `Q` for fold `g`'s rows with that teacher.
* Concatenate the four inner predictions into `Q_outer_train`, covering every
  row of `outer_train` (one Q value per row, from whichever of the four inner
  teachers excluded that row's own fold).
* Train `T1` (label `Q_outer_train`) and `T2` (label
  `0.5·y + 0.5·Q_outer_train`) on `outer_train`, identical inputs/capacity to
  `RT-992`/`RT-993` (500-column causal bank, `ARM_B_PARAMS` with
  `objective="xentropy"`, `MAX_TRAIN_ROWS` cap, `seed=0`). Evaluate on the
  untouched outer fold `f`.

Guarantee: no teacher that produced a label used in `T1`/`T2`'s training set
for outer fold `f` ever trained on fold `f`. 20 inner teacher fits total
(4 × 5 outer folds; no individual IDs — internal machinery, like
`Ctx.crossfit_streams`' per-fold calibration maps, never scored standalone or
written to `RESULTS.csv`) plus 10 student fits (`T1`+`T2` × 5 outer folds).

`T0` (`RT-990`) is unaffected and reused unchanged — it is ordinary
single-level 5-fold CV on hard labels, with no nested-model dependency, so it
was never contaminated and remains the matched control.

## 2. MECHANICAL SENTINEL — RUN AND MUST PASS BEFORE ANY SCORE

`fold_purity_test()` in `research/scripts/wave7_teacher_nested.py`: for every
outer fold `f` and every `Q` value generated for target fold `g` (`g ∈
outer_train`), assert

```
teacher_training_folds(f, g) ∩ {f, g} == ∅
```

where `teacher_training_folds(f, g) = {0,1,2,3,4} \ {f, g}` under the new
scheme. **The same test is run against the OLD scheme**
(`teacher_training_folds(g) = {0,1,2,3,4} \ {g}`, independent of `f`) and
**must fail** — demonstrating the sentinel actually catches the defect this
document exists to fix, not just that it passes trivially on the new
construction. Both checks are pure set arithmetic over fold indices (no
training, seconds to run) and are committed as `PASS`/`FAIL` evidence before
any inner teacher trains.

## 3. WHAT IS NOT CHANGED

Student inputs (500 unmodified causal columns), capacity (`ARM_B_PARAMS`),
row cap, seed, blend weight (`0.5`/`0.5`, no grid), and the teacher's own
architecture (Arm-C style: own 500 causal columns + own series' final-row
causal columns broadcast) are all identical to the contaminated pilot — only
the *cross-fitting structure that produces `Q`* changes. No hyperparameter
search on any inner teacher or outer student. No new features.

## 4. IDS

`RT-994` (nested `T1`) and `RT-995` (nested `T2`). These IDs were briefly
allocated to an *old-scheme* full-5-fold run that was started, then killed
before any fold completed once this contamination was found — no
`RESULTS.csv` row or OOF array was ever written under them, so they are
reused for the corrected design rather than retired. `RT-992`/`RT-993`
(the contaminated pilot) are kept as-is, relabeled, never overwritten.

## 5. EVALUATION AND REPORTING

Primary table, per outer fold `f`: `T0`, `T1`, `T2` whole-dev TS-AUC on fold
`f`, and `T1−T0`, `T2−T0`. Also, per fold: dominant-cell AUC delta,
never-break-only and pre-break-only cell delta, age-bucket and current-t
bucket breakdowns (`Ctx.score_by_age`, `AGE_BUCKETS` from `wave5_lib.py`).
Mean 5-fold delta, count of positive folds, paired series-level bootstrap
(`Ctx.bootstrap`, 200 reps) on `T1−T0` and `T2−T0`. **Contaminated-vs-clean
comparison**: fold-0 nested result vs. the original `RT-992`/`RT-993`
fold-0 pilot result, reported explicitly as `Δ(contaminated) − Δ(clean)` so
the size of the contamination's effect is itself measured, not just
asserted.

## 6. PROMOTION READING (fixed before any nested score is read)

* Nested-clean `T2` mean 5-fold delta vs `T0` **≥ +0.006** → **major
  breakthrough**, treat as a leadership candidate pending the remaining
  promotion legs.
* **+0.004 to +0.006** → **leaderboard candidate**.
* **+0.003 to +0.004** → **serious candidate**, proceed cautiously.
* **< +0.003** → **do not promote**, regardless of what the contaminated
  pilot showed.

Alternate-partition confirmation
(`research/folds/folds_alt{1,2,3}.parquet`) runs **only if** the nested-clean
canonical result clears **all** of: mean ≥ +0.003, ≥4/5 folds positive,
bootstrap CI entirely above zero. No submission until the nested-clean result
and the full promotion battery (this document's bar, plus alternate
partitions) both pass — this pre-registration authorizes measurement, not a
submission decision.

## 7. RUNNER

`research/scripts/wave7_teacher_nested.py`:

* `--fold-purity-test` — §2, before anything else.
* `--outer-fold F` (`F ∈ {0,1,2,3,4}`) — runs one outer fold's complete
  pipeline (4 inner teachers + `T1` + `T2`), writes partial per-fold OOF
  slices (`research/oof/RT-994_outer{F}.npy`,
  `research/oof/RT-995_outer{F}.npy`) so outer folds are independently
  resumable and can run as separate concurrent processes (machine limit
  unchanged: at most two concurrent, `research/HANDOFF_WAVE6.md` §2.4 — here
  interpreted at the process level, matching how every prior multi-fold
  runner in this project has used the limit).
* `--merge` — combines the five per-fold partial files into
  `research/oof/RT-994.npy` / `RT-995.npy` and appends one `RESULTS.csv` row
  per ID (`protocol="nested_full"`).
* `--analyze-nested` — §5–6, writes
  `research/reports/wave7_teacher_nested.{md,json}`.

## GIT

| | |
|---|---|
| starting SHA (pilot, contaminated) | `30a6647` |
| this amendment pre-registration commit | see `git log -1` after commit |
| first inner-teacher or nested-student score | strictly after this commit |
