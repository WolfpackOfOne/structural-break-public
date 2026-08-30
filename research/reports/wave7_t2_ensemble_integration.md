# WAVE 7 — T2 (RT-995) ENSEMBLE-INTEGRATION RESULT

Pre-registered: `research/WAVE7_T2_INTEGRATION_PREREG.md`. Fold 0, matching
every Wave-8 mechanism's own measured marginal (`SST`/`ORR`/`PCFB`/`CFEP`/
`TGMC`, all ≈ −0.0003). Equal-weight, cross-fitted SCDF calibration
(`wave5_lib.Ctx.crossfit_blend`), no weight search, no retraining — E0/E1/E2
computed directly from existing OOF arrays via
`wave8_common.ensemble_marginal`.

## Primary result table

| arm | definition | fold-0 TS-AUC |
|---|---|---:|
| E0 | RT-600 seven specialists, equal weight | 0.63828 |
| E1 | E0 + `RT-401` (matched exchangeable seed clone) | 0.63859 |
| E2 | E0 + `RT-995` (T2) | 0.63882 |

| contrast | Δ |
|---|---:|
| E1 − E0 (free eighth-stream benefit) | +0.00031 |
| E2 − E0 (T2's raw gain over the base ensemble) | +0.00055 |
| **E2 − E1 (T2's marginal over an exchangeable eighth stream)** | **+0.00024** |

## Reading against the pre-registered bands

`E2 − E1 = +0.00024` falls in **0 to +0.0015 → MOSTLY REDUNDANT**.

T2 does add a hair more than a generic eighth stream would (+0.00024 vs the
+0.00031 an exchangeable clone already provides — T2's *own* contribution
over the clone's contribution is a small fraction of the clone's own free
variance-reduction gain), but it is nowhere near the standalone signal
(`+0.00943` T2−T0) or even the "REAL BUT MODEST" band's floor
(`+0.0015`). Context: it is still the only *positive* marginal figure among
every candidate this branch (and Wave 8) ever measured against
`RT600+seedclone` — the five Wave-8 mechanisms all scored ≈ **−0.0003**. T2
is not redundant to the point of hurting the ensemble; it is redundant to
the point of barely helping it.

## Standalone context (not the binding number)

For reference only — `T0`/`T2` single-model comparison, not ensemble value:
`T2 − T0` mean 5-fold delta = **+0.00943** (5/5 folds positive, bootstrap CI
`[+0.00425, +0.01373]`, from `research/reports/wave7_teacher_nested.md`).
The gap between this number and the measured `+0.00024` ensemble marginal is
the entire finding of this leg — see
`research/reports/wave7_t2_pairflow.md` for why.
