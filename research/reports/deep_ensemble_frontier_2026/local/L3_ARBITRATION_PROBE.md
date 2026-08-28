# L3 ARBITRATION PROBE -- H3 VERDICT

Date: 2026-08-28
Branch: `research/deep-ensemble-frontier-local-2026`
Scoring SHA: `ac82624`

Verdict: `FAIL_NO_RETENTION_MECHANISM`.

This is fold-0 only. `RT-1234` and `RT-1235` each have 806,334 finite rows, so this is a descriptive gate and cannot promote a model.

| rule | action rows | dominant repairs | dominant damage | dominant net | retained repairs | mature-prebreak damage rate | fold0 E2-E1 | success |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `unconditional_equal` | 806334 | 3392 | 3819 | -427 | 1.000 | 0.1170 | +0.010920585 | False |
| `confidence_q05` | 80707 | 399 | 405 | -6 | 0.118 | 0.0111 | +0.003642304 | False |
| `confidence_q10` | 161373 | 850 | 834 | 16 | 0.251 | 0.0221 | +0.007600188 | False |
| `confidence_q20` | 322725 | 1541 | 1606 | -65 | 0.454 | 0.0484 | +0.009676960 | False |
| `confidence_q30` | 483931 | 2226 | 2434 | -208 | 0.656 | 0.0751 | +0.010880062 | False |
| `rt600_boundary_q10` | 80641 | 416 | 372 | 44 | 0.123 | 0.0132 | +0.000372912 | False |
| `rt600_boundary_q20` | 161464 | 745 | 617 | 128 | 0.220 | 0.0208 | +0.000912298 | False |
| `rt600_boundary_q30` | 241846 | 1049 | 879 | 170 | 0.309 | 0.0279 | +0.001328097 | False |
| `agreement_direction` | 508120 | 1531 | 1651 | -120 | 0.451 | 0.0517 | +0.000048803 | False |
| `dominant_cell_only` | 477025 | 3392 | 3819 | -427 | 1.000 | 0.1170 | +0.009598226 | False |
| `dominant_confidence_q10` | 95741 | 850 | 834 | 16 | 0.251 | 0.0221 | +0.005312419 | False |
| `dominant_confidence_q20` | 190944 | 1541 | 1606 | -65 | 0.454 | 0.0484 | +0.007139983 | False |
| `three_way_abstain_q10` | 423871 | 2264 | 5187 | -2923 | 0.667 | 0.1504 | +0.011550063 | False |
| `three_way_abstain_q20` | 536696 | 2802 | 5048 | -2246 | 0.826 | 0.1461 | +0.010888317 | False |

Successful rules: `none`.
Best dominant net rule: `rt600_boundary_q30`.

The prior was poor: SS-01 through SS-04 were all KILL. The only reason to run this probe was the much lower RT-1234 rho and its distinct repair set. A negative result here is direct evidence against opening another neural detector without a new retention mechanism.
