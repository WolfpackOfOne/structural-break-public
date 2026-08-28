# CSA-04 LOCAL CATBOOST SLOT SWEEP -- FINAL

Date: 2026-08-28
Branch: `research/deep-ensemble-frontier-local-2026`
Scoring SHA: `766ecf0`

## Single-Slot Replacements

| arm | RT ID | slot | standalone | rho | marginal_vs_clone | E2-E0 | folds positive | dominant net | mature-never net | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| CAT-413 | `RT-1254` | `RT-413` | 0.620440244 | 0.714196 | +0.001087151 | +0.001244347 | 5/5 | 93 | 80 | INTERESTING |
| CAT-300 | `RT-1255` | `RT-300` | 0.620242061 | 0.737786 | +0.001029459 | +0.001126876 | 5/5 | 75 | 80 | INTERESTING |
| CAT-410 | `RT-1256` | `RT-410` | 0.610328251 | 0.680825 | +0.000753095 | +0.000929590 | 4/5 | 17 | 76 | KILL |
| CAT-411 | `RT-1260` | `RT-411` | 0.609378436 | 0.801134 | +0.001567468 | +0.000362939 | 5/5 | 36 | -9 | PROMOTION_WORTHY |
| CAT-412 | `RT-1261` | `RT-412` | 0.623511276 | 0.687397 | +0.001630380 | +0.001029593 | 5/5 | 92 | 54 | PROMOTION_WORTHY |
| CAT-414 | `RT-1262` | `RT-414` | 0.615326823 | 0.741628 | +0.001588999 | +0.000552119 | 4/5 | 41 | 49 | PROMOTION_WORTHY |
| CAT-415 | `RT-1263` | `RT-415` | 0.619831816 | 0.719175 | +0.001001378 | +0.000907324 | 5/5 | 81 | 50 | INTERESTING |

## Hybrid Curve

RT-1257 regression check: observed `+0.002407204670070`, expected `+0.002407204670070`, diff `+0.000e+00`, passed `True`.

| k | RT ID | ordered survivor slots | marginal_vs_clone | E2-E0 | folds positive | dominant net | mature-never net | verdict |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 2 | `` | CAT-412, CAT-414 | +0.003601682 | +0.001439569 | 5/5 | 76 | 113 | SERIOUS |
| 3 | `` | CAT-412, CAT-414, CAT-411 | +0.004827562 | +0.001671713 | 5/5 | 117 | 115 | SERIOUS |
| 4 | `` | CAT-412, CAT-414, CAT-411, CAT-413 | +0.005623906 | +0.002239698 | 5/5 | 114 | 76 | MAJOR |
| 5 | `RT-1264` | CAT-412, CAT-414, CAT-411, CAT-413, CAT-300 | +0.005934643 | +0.002378478 | 5/5 | 104 | 33 | MAJOR |
| 6 | `` | CAT-412, CAT-414, CAT-411, CAT-413, CAT-300, CAT-415 | +0.005896893 | +0.001864143 | 5/5 | 100 | 41 | MAJOR |

Best `k`: `5` with survivors `CAT-412, CAT-414, CAT-411, CAT-413, CAT-300`.
H2 composition for CRUNCH: `RT-1255, RT-410, RT-1260, RT-1261, RT-1254, RT-1262, RT-415`.

## Prediction Adjudication

P1 breadth: not supported. CAT-412/CAT-415 margins were +0.001630380/+0.001001378; CAT-411/CAT-414 margins were +0.001567468/+0.001588999.

P2 idiosyncrasy: not supported. The wide idiosyncratic slots CAT-412 and CAT-415 did not both fail the +0.0010 gate.

P3 interior maximum: supported. Best k was 5; largest evaluated k was 6.

No lockbox, test, production, feature, router, stacker, hyperparameter, seed, or blend-weight change was made.
