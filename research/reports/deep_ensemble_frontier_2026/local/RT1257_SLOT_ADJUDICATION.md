# RT-1257 Residual Slot Adjudication

Date: 2026-08-31
Branch: `research/rt1257-slot-adjudication`
Git SHA: `8adb33f889d6dc547739c0f0e71542a979e1b99e`

This is a zero-training adjudication of the remaining CSA-04 CatBoost slot candidates after
`RT-1257` has already installed `CAT-300` (`RT-1255`) and `CAT-413` (`RT-1254`).
It reads frozen OOF vectors only; no lockbox/test rows are filled or evaluated.

## Arms

- `E0`: RT-1257 fixed champion composition.
- `E1`: RT-1257 with the target residual slot replaced by matched seed clone `RT-401`.
- `E2`: RT-1257 with the target residual slot replaced by the CatBoost candidate.

Primary endpoint: `E2-E1`. Secondary deployment endpoint: `E2-E0`.
Continuation requires both endpoints to clear `max(SE, 0.0011)`, at least 4/5 positive folds
against RT-1257, and positive dominant-cell plus mature-vs-never pair-flow nets.

## Harness Checks

- RT-600 mean TS-AUC: 0.625811342 (expected 0.625811342; passed=True)
- RT-1257 mean TS-AUC: 0.627837664 (expected 0.627837664; passed=True)

## Summary

| priority | candidate | slot | primary E2-E1 | secondary E2-E0 | folds E2>E0 | boot 95% CI E2-E0 | dominant net | mature-never net | recommendation |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | RT-1261 / CAT-412 | RT-412 | +0.001087276 | +0.000338234 | 3/5 | [-0.000631306, +0.001358457] | -31 | 39 | RESEARCH_ALIVE_WEAK |
| 2 | RT-1263 / CAT-415 | RT-415 | +0.000283602 | +0.000175228 | 3/5 | [-0.000756519, +0.001139340] | -16 | -3 | RESEARCH_ALIVE_WEAK |
| 3 | RT-1262 / CAT-414 | RT-414 | +0.001057324 | +0.000178858 | 4/5 | [-0.000729204, +0.001056890] | -84 | 11 | RESEARCH_ALIVE_WEAK |
| 4 | RT-1260 / CAT-411 | RT-411 | +0.001317358 | +0.000133942 | 2/5 | [-0.000455188, +0.000748957] | -5 | 5 | PARK_NO_PROMOTION |

## Details

### RT-1261 / CAT-412

- Slot: `RT-412` replaced by `RT-1261`.
- Standalone delta vs incumbent: +0.009575300; within-t rho vs incumbent: 0.687397.
- Control movement (`E1-E0`): -0.000749042 (bootstrap SE 0.000474844, 95% CI [-0.001660128, +0.000191940]).
- Primary (`E2-E1`): +0.001087276 (5/5 folds; bootstrap SE 0.000479109, 95% CI [+0.000120550, +0.002001007]).
- Secondary (`E2-E0`): +0.000338234 (3/5 folds; bootstrap SE 0.000518804, 95% CI [-0.000631306, +0.001358457]).
- Pair flow vs RT-1257: dominant net -31; mature-vs-never net 39; mature-vs-prebreak net 5.

| fold | E1-E0 | E2-E1 | E2-E0 |
|---:|---:|---:|---:|
| 0 | -0.001053516 | +0.000437536 | -0.000615980 |
| 1 | +0.000894489 | +0.001112418 | +0.002006908 |
| 2 | -0.000375188 | +0.000918170 | +0.000542982 |
| 3 | -0.000590155 | +0.001648688 | +0.001058533 |
| 4 | -0.002620839 | +0.001319568 | -0.001301271 |

### RT-1263 / CAT-415

- Slot: `RT-415` replaced by `RT-1263`.
- Standalone delta vs incumbent: +0.002747386; within-t rho vs incumbent: 0.719175.
- Control movement (`E1-E0`): -0.000108375 (bootstrap SE 0.000377353, 95% CI [-0.000835803, +0.000647302]).
- Primary (`E2-E1`): +0.000283602 (3/5 folds; bootstrap SE 0.000472576, 95% CI [-0.000623453, +0.001196195]).
- Secondary (`E2-E0`): +0.000175228 (3/5 folds; bootstrap SE 0.000487608, 95% CI [-0.000756519, +0.001139340]).
- Pair flow vs RT-1257: dominant net -16; mature-vs-never net -3; mature-vs-prebreak net 44.

| fold | E1-E0 | E2-E1 | E2-E0 |
|---:|---:|---:|---:|
| 0 | -0.000167474 | +0.000562168 | +0.000394694 |
| 1 | -0.000107268 | +0.000660580 | +0.000553312 |
| 2 | -0.000365673 | +0.001180263 | +0.000814590 |
| 3 | -0.000094363 | -0.000251708 | -0.000346072 |
| 4 | +0.000192905 | -0.000733291 | -0.000540386 |

### RT-1262 / CAT-414

- Slot: `RT-414` replaced by `RT-1262`.
- Standalone delta vs incumbent: +0.004281980; within-t rho vs incumbent: 0.741628.
- Control movement (`E1-E0`): -0.000878466 (bootstrap SE 0.000582849, 95% CI [-0.002042670, +0.000208528]).
- Primary (`E2-E1`): +0.001057324 (4/5 folds; bootstrap SE 0.000591525, 95% CI [-0.000141396, +0.002192808]).
- Secondary (`E2-E0`): +0.000178858 (4/5 folds; bootstrap SE 0.000462471, 95% CI [-0.000729204, +0.001056890]).
- Pair flow vs RT-1257: dominant net -84; mature-vs-never net 11; mature-vs-prebreak net -14.

| fold | E1-E0 | E2-E1 | E2-E0 |
|---:|---:|---:|---:|
| 0 | +0.001301372 | -0.000901904 | +0.000399468 |
| 1 | -0.001266508 | +0.001689743 | +0.000423236 |
| 2 | -0.000000048 | +0.000636340 | +0.000636292 |
| 3 | -0.000450959 | +0.001282430 | +0.000831470 |
| 4 | -0.003976187 | +0.002580011 | -0.001396176 |

### RT-1260 / CAT-411

- Slot: `RT-411` replaced by `RT-1260`.
- Standalone delta vs incumbent: +0.001566454; within-t rho vs incumbent: 0.801134.
- Control movement (`E1-E0`): -0.001183416 (bootstrap SE 0.000517570, 95% CI [-0.002189766, -0.000132807]).
- Primary (`E2-E1`): +0.001317358 (5/5 folds; bootstrap SE 0.000545774, 95% CI [+0.000226905, +0.002403473]).
- Secondary (`E2-E0`): +0.000133942 (2/5 folds; bootstrap SE 0.000305040, 95% CI [-0.000455188, +0.000748957]).
- Pair flow vs RT-1257: dominant net -5; mature-vs-never net 5; mature-vs-prebreak net 3.

| fold | E1-E0 | E2-E1 | E2-E0 |
|---:|---:|---:|---:|
| 0 | -0.002020861 | +0.001630618 | -0.000390243 |
| 1 | -0.000594573 | +0.000538874 | -0.000055699 |
| 2 | -0.000997057 | +0.001161634 | +0.000164577 |
| 3 | -0.002544058 | +0.002469432 | -0.000074626 |
| 4 | +0.000239469 | +0.000786232 | +0.001025701 |

## Interpretation

None of the residual slots earns promotion from this champion-relative check unless its secondary
`E2-E0` clears the measured uncertainty/noise bar. Positive standalone or clone-relative movement is
treated as insufficient when the deployment composition does not improve RT-1257 by a distinguishable amount.

Recommended next work:

1. Do not open new training lanes for CAT-411, CAT-414, or CAT-415 from these frozen OOF results.
2. Keep CAT-412 as the only residual slot worth a narrow follow-up, and only if a preregistered alt-partition
   adjudication is desired despite the sub-noise RT-1257 lift.
3. If CAT-412 is followed up, run alt partitions with the same RT-1257-relative `E0/E1/E2` contract and
   require a deployment endpoint above the noise floor before changing the champion.

## Repro

```bash
/home/user/anaconda3/bin/python research/scripts/rt1257_slot_adjudication.py --artifact-root ../structural-break-deep-ensemble-frontier-local-2026 --bootstrap-reps 2000
```
