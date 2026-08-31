# LA-01 -- Specialist Replacement Salvage Result

Date: 2026-08-26

Execution commit: `1dd8be0`

Program preregistration: `PROGRAM_PREREG.md`

Execution preregistration: `LA01_EXECUTION_PREREG.md`

Artifacts:

- `la01_nested_replacement.json`
- `la01_nested_replacement_summary.csv`

## Verdict

**KILL.** Specialist-replacement salvage is closed.

The nested candidate replacement did not beat the nested seed-clone replacement:

| arm | mean TS-AUC | pooled TS-AUC | fold TS-AUC |
|---|---:|---:|---|
| E0 RT600 seven specialists | 0.625811342 | 0.625626935 | 0.638276 / 0.620402 / 0.633930 / 0.617508 / 0.618941 |
| RT-1244 seed-clone replacement | 0.624980472 | 0.624723814 | 0.637842 / 0.620122 / 0.633561 / 0.617005 / 0.616374 |
| RT-1243 m11/m12 replacement | 0.624975063 | 0.624730380 | 0.639195 / 0.620200 / 0.633390 / 0.616044 / 0.616046 |

Primary `marginal_vs_clone = RT-1243 - RT-1244 = -0.000005408`.

Fold deltas vs clone:

- fold 0: `+0.001353828`
- fold 1: `+0.000078310`
- fold 2: `-0.000170716`
- fold 3: `-0.000960269`
- fold 4: `-0.000328194`

Only 2/5 folds were positive, below the required 4/5.

## Pair Flow

Pair-flow diagnostics are candidate vs E0 with 64 same-`t` pairs per time point,
seed `20260826`.

| split | repairs | damage | net |
|---|---:|---:|---:|
| whole dev | 1113 | 1205 | -92 |
| dominant cell | 828 | 838 | -10 |
| mature-vs-never | 802 | 840 | -38 |
| mature-vs-prebreak | 625 | 724 | -99 |

Dominant-cell net is not positive, so the LA-01 gate fails independently of the
primary marginal.

## Correlation

Within-`t` rank correlation with E0:

- RT-1243 vs E0, dev: `0.989999`
- RT-1243 vs E0, dominant cell: `0.992112`
- RT-1244 vs E0, dev: `0.991061`
- RT-1244 vs E0, dominant cell: `0.992932`

The replacement arms are nearly rank-identical to RT600.

## Nested Choices

Candidate arm:

| outer fold | replaced | replacement | inner mean |
|---:|---|---|---:|
| 0 | RT-413 | RT-751 (`m12_rdep`) | 0.623289 |
| 1 | RT-410 | RT-751 (`m12_rdep`) | 0.628084 |
| 2 | RT-300 | RT-751 (`m12_rdep`) | 0.624663 |
| 3 | RT-300 | RT-731 (`m11_focus`) | 0.628542 |
| 4 | RT-410 | RT-751 (`m12_rdep`) | 0.629131 |

Seed-clone control:

| outer fold | replaced | replacement | inner mean |
|---:|---|---|---:|
| 0 | RT-413 | RT-401 | 0.623006 |
| 1 | RT-410 | RT-401 | 0.627453 |
| 2 | RT-300 | RT-401 | 0.623981 |
| 3 | RT-300 | RT-401 | 0.628116 |
| 4 | RT-410 | RT-401 | 0.628398 |

## Notes

The first process was interrupted before it emitted any score or artifact
because it recomputed identical SCDF maps. Commit `1dd8be0` cached those maps
without changing the preregistered math. The completed run wrote metrics but
failed at `RESULTS.csv` append due a CSV-reader bug; the two ledger rows were
then appended from the frozen JSON artifact without recomputing the score.

Runtime for the completed scored run: `185.0 s`; calibration cache entries:
`245`.

