# LA-03 -- Per-Series History Adaptation Result

Date: 2026-08-26

Program preregistration: `PROGRAM_PREREG.md` at `ed84d00`.
Execution preregistration: `LA03_EXECUTION_PREREG.md` at `3d88512`.

## Verdict

`RT-1249` is **KILL**.

The integrated candidate is WEAK by marginal size: `RT600 + RT-1249` beats the
`RT600 + RT-1247` global no-adaptation clone by `+0.001676065` on `5/5` folds.
But the mandatory standalone isolation gate fails because the adapted head loses
to the fixed per-series null control by `-0.021495290`. The fixed-null diagnostic
also beats the candidate when integrated with RT600.

## Standalone Heads

| arm | mean TS-AUC | pooled TS-AUC | fold TS-AUC |
|---|---:|---:|---|
| `RT-1247` global no-adaptation | `0.528175061` | `0.527857309` | `0.530879358;0.526279830;0.523294043;0.531493524;0.528928552` |
| `RT-1248` fixed per-series null | `0.564744385` | `0.564551631` | `0.567625194;0.565032819;0.552958173;0.567479407;0.570626331` |
| `RT-1249` affine-adapted candidate | `0.543249095` | `0.543070103` | `0.550251532;0.541945090;0.546929767;0.530516770;0.546602316` |

Isolation:

- `RT-1249 - RT-1247 = +0.015074034` PASS.
- `RT-1249 - RT-1248 = -0.021495290` FAIL.

## RT600 Integration

| arm | mean TS-AUC | pooled TS-AUC | fold TS-AUC | delta |
|---|---:|---:|---|---:|
| E0 RT600 | `0.625811264` | `0.625626926` | `0.638276303;0.620401850;0.633930229;0.617507505;0.618940432` | -- |
| E1 `RT600 + RT-1247` clone | `0.624411329` | `0.624249199` | `0.636799459;0.619096493;0.631075607;0.616324108;0.618760979` | `-0.001399935` vs E0 |
| fixed diagnostic `RT600 + RT-1248` | `0.626869923` | `0.626647130` | `0.639068337;0.621161199;0.632916261;0.619603867;0.621599953` | `+0.001058660` vs E0 |
| E2 `RT600 + RT-1249` candidate | `0.626087395` | `0.625864075` | `0.639271127;0.619953970;0.634668911;0.616716518;0.619826447` | `+0.000276131` vs E0 |

Primary `marginal_vs_clone = E2 - E1 = +0.001676065`.
Fold deltas vs clone:

`+0.002472; +0.000857; +0.003593; +0.000392; +0.001065`

## Pair Flow

E2 vs E0, 64 same-`t` pairs per time point, seed `20260826`:

| split | repairs | damage | net | damage rate |
|---|---:|---:|---:|---:|
| whole dev | `1524` | `1593` | `-69` | `0.025208` |
| dominant cell | `1166` | `1203` | `-37` | `0.023842` |
| mature-vs-never | `1173` | `1265` | `-92` | `0.025073` |
| mature-vs-prebreak | `924` | `998` | `-74` | `0.022665` |

Within-`t` rank correlation vs E0:

| arm | dev | dominant cell |
|---|---:|---:|
| E1 global clone | `0.984904` | `0.986906` |
| E2 adapted candidate | `0.985206` | `0.987105` |

## Gate Accounting

| gate | result |
|---|---|
| adapted beats global standalone by `>= +0.0005` | PASS |
| adapted beats fixed-null standalone by `>= +0.0005` | FAIL |
| integrated `marginal_vs_clone >= +0.0015` | PASS |
| at least 4/5 folds positive | PASS |

The fixed-null control remains the better legal per-series representation.
Per-series affine adaptation is closed; no LA-03b is authorised.

Runtime: `1,189.8 s` (`19.8 min`).

Metrics: `la03_per_series_adaptation.json` and
`la03_per_series_adaptation_summary.csv`.
