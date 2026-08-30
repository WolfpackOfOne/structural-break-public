# WAVE 7 — TEACHER NESTED (OUTER-FOLD-PURE) RESULTS

Pre-registered: `research/WAVE7_TEACHER_NESTED_PREREG.md`. Replaces outer-fold-contaminated `RT-992`/`RT-993` (`research/reports/wave7_teacher_pilot.md`) with `RT-994`/`RT-995`, trained under nested (double) cross-fitting so no teacher that labels an outer fold's training data ever trained on that fold.

## Primary result table (whole-dev TS-AUC per outer fold)

| outer fold | T0 | T1 | T2 | T1−T0 | T2−T0 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.62656 | 0.61931 | 0.63103 | -0.00725 | +0.00447 |
| 1 | 0.60617 | 0.61792 | 0.62008 | +0.01175 | +0.01391 |
| 2 | 0.61753 | 0.61242 | 0.62658 | -0.00512 | +0.00904 |
| 3 | 0.60318 | 0.62065 | 0.61679 | +0.01747 | +0.01361 |
| 4 | 0.60581 | 0.60642 | 0.61190 | +0.00061 | +0.00609 |

**Mean Δ**: T1 +0.00349 (3/5 folds positive), T2 +0.00943 (5/5 folds positive).

## Dominant-cell (t≥200, age≥100) TS-AUC per outer fold

| outer fold | T0 cell | T1 cell | T2 cell | T0 never-brk | T1 never-brk | T2 never-brk | T0 pre-brk | T1 pre-brk | T2 pre-brk |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.66381 | 0.65508 | 0.66795 | 0.66384 | 0.65251 | 0.66814 | 0.66374 | 0.66223 | 0.66743 |
| 1 | 0.64176 | 0.66316 | 0.66295 | 0.64713 | 0.66377 | 0.66520 | 0.62579 | 0.66135 | 0.65625 |
| 2 | 0.66119 | 0.65851 | 0.67498 | 0.65233 | 0.65414 | 0.66758 | 0.68655 | 0.67103 | 0.69618 |
| 3 | 0.63478 | 0.65740 | 0.65585 | 0.64292 | 0.66521 | 0.66280 | 0.61149 | 0.63506 | 0.63596 |
| 4 | 0.63536 | 0.63921 | 0.64553 | 0.63883 | 0.64600 | 0.65149 | 0.62516 | 0.61928 | 0.62805 |

## Paired series bootstrap (200 reps)

| contrast | mean | CI95 | fraction > 0 |
|---|---:|---|---:|
| T1-T0 | +0.00308 | [-0.00468, +0.01089] | 0.78 |
| T2-T0 | +0.00922 | [+0.00425, +0.01373] | 1.00 |

## Contaminated-vs-clean (fold 0 only, whole-dev)

| arm | contaminated Δ | clean (nested) Δ | contamination inflation |
|---|---:|---:|---:|
| T1 | +0.01621 | -0.00725 | +0.02346 |
| T2 | +0.01671 | +0.00447 | +0.01224 |

## Reading (research/WAVE7_TEACHER_NESTED_PREREG.md §6)

T1: mean Δ +0.00349 → **SERIOUS CANDIDATE**  

T2: mean Δ +0.00943 → **MAJOR BREAKTHROUGH**


**Clears promotion legs 1-3 (magnitude ≥+0.003, ≥4/5 folds, bootstrap CI>0):** T1=False, T2=True


**Alternate-partition confirmation authorized:** True
