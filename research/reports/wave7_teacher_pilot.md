# WAVE 7 — TEACHER PILOT RESULTS (fold 0 only)

> **STATUS: MECHANISM-POSITIVE SCREEN. OUTER-FOLD CONTAMINATED. NOT PROMOTION
> EVIDENCE.** Post-hoc review found that `Q` (the global `RT-991` OOF, reused
> as-is) is cross-fitted only with respect to the row/series it predicts, not
> with respect to the *outer student validation fold*. Concretely: this
> pilot's outer fold is 0; fold-1 student-training rows received `Q` from
> `RT-991`'s fold-1 teacher, which itself trained on folds `{0,2,3,4}` —
> i.e. on fold 0, the outer validation fold. Fold 0 therefore had an
> indirect path into student training via the teacher's learned weights,
> before the student was ever scored on fold 0. This is a nested-CV
> meta-feature leakage pattern (the same failure mode a stacked ensemble has
> when base-model OOF isn't re-cross-fitted against the outer split), not a
> deployment-causality failure — the trained student's *inference* path
> remains strictly causal (500 unmodified columns, no privileged input). The
> numbers below are **not deleted** (they are a real, reproducible mechanism
> signal — the direction and rough size of "does teacher supervision help"
> is still informative) but must not be read as a clean generalization
> estimate or cited as promotion evidence. See
> `research/WAVE7_TEACHER_NESTED_PREREG.md` for the corrected design and
> `research/reports/wave7_teacher_nested.md` for the outer-fold-pure result
> and the contaminated-vs-clean comparison.

Pre-registered: `research/WAVE7_TEACHER_PREREG.md`. Teacher = `RT-991` (W7-D3R Arm C, reused). Fold 0 only -- not a full 5-fold result.

D3R fold-0 reference: Arm B cell AUC 0.66381, Arm C cell AUC 0.71658, future gap +0.05277.

## Arms

| arm | exp id | cell TS-AUC | never-break-only | pre-break-only | whole-fold0 TS-AUC |
|---|---|---:|---:|---:|---:|
| T0_RT990 | `RT-990` | 0.66381 | 0.66384 | 0.66374 | 0.62656 |
| T1_RT992 | `RT-992` | 0.68169 | 0.68018 | 0.68592 | 0.64277 |
| T2_RT993 | `RT-993` | 0.68439 | 0.68523 | 0.68206 | 0.64327 |

## Deltas vs T0

| arm | cell Δ | bucket | whole-fold0 Δ | translated aggregate Δ | distillation efficiency |
|---|---:|---|---:|---:|---:|
| T1_RT992 | +0.01788 | meaningful | +0.01621 | +0.00903 | 33.9% |
| T2_RT993 | +0.02058 | large | +0.01671 | +0.01039 | 39.0% |

## Gates (best arm: T2_RT993)

A (cell ≥+0.010, no material damage): True  
B (whole-fold0 ≥+0.003): True  
C (translated ≥+0.004): True


## VERDICT: **CONTINUE**
