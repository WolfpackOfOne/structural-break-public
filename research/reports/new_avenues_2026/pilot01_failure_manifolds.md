# PILOT 1 -- SPECIALIST COMPETENCE AND FAILURE MANIFOLDS

Generated: 2026-08-24 on `research/new-avenues-pilots-2026`.

## RT-600 Anchor

* Dev mean TS-AUC: `0.625811`.
* Dev pooled TS-AUC: `0.625627`.
* Dominant-cell AUC: `0.664277`.

## Primary Diagnostic Verdict

Verdict: **WEAK**.

Best fold-held history-only selector headroom: `-0.023559` dominant-cell AUC via `exc_max_run64`.
Best fold-preserving permutation-control selector headroom: `-0.019020`.

The diagnostic does not allocate an RT ID and does not update `RESULTS.csv`.

## Specialist Selector Headroom

| fingerprint | delta vs RT600 | AUC | selected specialists |
|---|---:|---:|---|
| `exc_max_run64` | -0.023559 | 0.640718 | RT-300, RT-412, RT-413, RT-415 |
| `turn_rate` | -0.023712 | 0.640565 | RT-300, RT-412, RT-413, RT-414, RT-415 |
| `exc_n64` | -0.024350 | 0.639928 | RT-300, RT-412, RT-413, RT-414, RT-415 |
| `hill` | -0.024717 | 0.639560 | RT-300, RT-412, RT-413, RT-415 |
| `q_ratio` | -0.027902 | 0.636375 | RT-300, RT-412, RT-413, RT-415 |
| `max_logvr128` | -0.029748 | 0.634529 | RT-300, RT-412, RT-413, RT-414, RT-415 |
| `ar3` | -0.030096 | 0.634181 | RT-300, RT-412, RT-413, RT-415 |
| `max_absz64` | -0.030214 | 0.634063 | RT-300, RT-412, RT-413, RT-414, RT-415 |

## Strongest Monotone Fingerprint Signals

| fingerprint | population | rho with RT600 loss | rho with best specialist advantage |
|---|---|---:|---:|
| `kurt` | never_break | +0.1419 | +0.1529 |
| `q_ratio` | never_break | +0.1425 | +0.1493 |
| `hill` | never_break | -0.1281 | -0.1349 |
| `exc_n64` | never_break | -0.1287 | -0.1131 |
| `n_hist` | never_break | -0.0954 | -0.0949 |
| `kurt` | all | +0.0772 | +0.0941 |
| `perm_ent` | never_break | +0.0945 | +0.0913 |
| `q_ratio` | all | +0.0693 | +0.0894 |
| `skew` | never_break | +0.0889 | +0.0826 |
| `hill` | all | -0.0708 | -0.0793 |

## Hard Never-Break Strata

| fingerprint | top-decile SMD | rho with loss |
|---|---:|---:|
| `n_hist` | -0.3138 | -0.0954 |
| `q_ratio` | +0.2879 | +0.1425 |
| `exc_n64` | -0.2435 | -0.1287 |
| `hill` | -0.1750 | -0.1281 |
| `max_logvr128` | +0.1712 | +0.0449 |
| `acf1_abs` | +0.1365 | +0.0189 |
| `acf1_sq` | +0.1106 | +0.0161 |
| `ar3` | +0.0860 | +0.0274 |
| `perm_ent` | +0.0667 | +0.0945 |
| `spec_slope` | -0.0578 | -0.0760 |

## Failure-Manifold Clusters

| k | silhouette | eta2 loss | eta2 best advantage | cluster summaries |
|---:|---:|---:|---:|---|
| 2 | 0.9916 | 0.0000 | 0.0003 | c0: n=6369, nb=0.51, loss=0.340, adv=+0.1050, mode=RT-411; c1: n=7, nb=0.43, loss=0.361, adv=+0.1497, mode=RT-411 |
| 3 | 0.9753 | 0.0001 | 0.0004 | c0: n=6351, nb=0.51, loss=0.340, adv=+0.1050, mode=RT-411; c1: n=5, nb=0.60, loss=0.284, adv=+0.1274, mode=RT-411; c2: n=20, nb=0.55, loss=0.346, adv=+0.1347, mode=RT-410 |

## Specialist Pair Flow Vs RT-600, Dominant Cell

| specialist | corr vs RT600 | repairs | damage | net |
|---|---:|---:|---:|---:|
| `RT-300` | +0.9104 | 812 | 1012 | -200 |
| `RT-410` | +0.8112 | 1150 | 1562 | -412 |
| `RT-411` | +0.7457 | 1444 | 1751 | -307 |
| `RT-412` | +0.8690 | 1011 | 1145 | -134 |
| `RT-413` | +0.9064 | 853 | 999 | -146 |
| `RT-414` | +0.7937 | 1243 | 1613 | -370 |
| `RT-415` | +0.9110 | 795 | 1027 | -232 |

## Interpretation

History-only fingerprints do carry weak monotone information about where RT-600 fails, but the fold-held selector headroom is read against a same-procedure permutation control. This pilot opens a future gate only if the real selector headroom and specialist-advantage monotonicity clear the preregistered thresholds.

Runtime: `81.8s`.
