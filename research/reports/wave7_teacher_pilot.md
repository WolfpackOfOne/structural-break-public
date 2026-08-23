# WAVE 7 — TEACHER PILOT RESULTS (fold 0 only)

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
