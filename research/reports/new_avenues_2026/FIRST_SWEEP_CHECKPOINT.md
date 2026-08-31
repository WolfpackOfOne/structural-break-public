# FIRST-SWEEP CHECKPOINT -- NEW AVENUES 2026

Branch: `research/new-avenues-pilots-2026`.

This table is a compact research checkpoint, not a replacement for
`research/RESULTS.csv`. Values are fold-0 screen diagnostics unless noted.

| Pilot | RT ID | mechanism | control RT ID | standalone | dominant cell | mature-vs-never | rho RT600 | repairs | damage | net pair lift | E0 | E1 | E2 | marginal_vs_clone | control gap | runtime | verdict | confirmation |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | none | specialist/failure-manifold diagnostic | permutation | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.638276 | n/a | n/a | n/a | n/a | 81.8s | WEAK | no scored candidate |
| 2 | RT-1200 | relay score-state transform | RT-401 | 0.619611 | 0.658230 | 0.655859 | 0.7746 | 1158 | 1476 | -318 | 0.638276 | 0.638586 | 0.638380 | -0.000207 | n/a | 64.3s | KILL | none |
| 3 | RT-1201 | IM2 dwell run-null scalar | RT-401 | 0.583578 | 0.610158 | 0.613459 | 0.3817 | 2021 | 3172 | -1151 | 0.638276 | 0.638586 | 0.638888 | +0.000301 | n/a | 244.3s | KILL | none |
| 4 | RT-1202 | trajectory geometry scalar | RT-1203 | 0.499324 | 0.500308 | 0.501410 | 0.0039 | 2460 | 5434 | -2974 | 0.638276 | 0.638586 | 0.636000 | -0.002587 | +0.000455 | 311.4s | KILL | none |
| 4 | RT-1203 | shuffled trajectory control | n/a | 0.493824 | 0.489650 | 0.486356 | 0.0074 | 2357 | 5487 | -3130 | 0.638276 | 0.638586 | 0.635545 | -0.003042 | n/a | 311.4s | negative control | none |
| 5 | RT-1204 | scale-survival summary | RT-1205 | 0.629444 | 0.670218 | 0.666092 | 0.8860 | 874 | 1005 | -131 | 0.638276 | 0.638586 | 0.638350 | -0.000236 | +0.000471 | 1998.6s | KILL | none |
| 5 | RT-1205 | individual-scale surprise control | n/a | 0.625238 | 0.664388 | 0.662921 | 0.8870 | 871 | 1046 | -175 | 0.638276 | 0.638586 | 0.637880 | -0.000707 | n/a | 1998.6s | negative control | none |
| 6 | RT-1206 | spectral impulsiveness contrast | RT-1207 | 0.627636 | 0.667578 | 0.664795 | 0.8809 | 892 | 1141 | -249 | 0.638276 | 0.638586 | 0.638233 | -0.000353 | -0.000542 | 1586.9s | KILL | none |
| 6 | RT-1207 | plain spectral-energy control | n/a | 0.631084 | 0.674571 | 0.673311 | 0.8851 | 942 | 962 | -20 | 0.638276 | 0.638586 | 0.638775 | +0.000189 | n/a | 1586.9s | control exceeded candidate | none |
| 7 | RT-1208 | ordinal transition KL + irreversibility | RT-1209 | 0.629141 | 0.668268 | 0.664570 | 0.8780 | 925 | 1000 | -75 | 0.638276 | 0.638586 | 0.638551 | -0.000036 | +0.000472 | 1627.8s | KILL | none |
| 7 | RT-1209 | permutation-entropy-only control | n/a | 0.626836 | 0.663077 | 0.663400 | 0.8850 | 863 | 1139 | -276 | 0.638276 | 0.638586 | 0.638079 | -0.000508 | n/a | 1627.8s | negative control | none |
| 10 | RT-1210 | joint size-duration rarity | RT-1211 | 0.626121 | 0.663758 | 0.659593 | 0.8779 | 902 | 1139 | -237 | 0.638276 | 0.638586 | 0.638065 | -0.000522 | -0.000285 | 1580.6s | KILL | none |
| 10 | RT-1211 | dwell-only rarity control | n/a | 0.627141 | 0.664654 | 0.662531 | 0.8704 | 941 | 1146 | -205 | 0.638276 | 0.638586 | 0.638350 | -0.000237 | n/a | 1580.6s | control exceeded candidate | none |
| 9(i) | RT-1212 | nested scalar difficulty conditioner | RT-1213 | 0.629133 | 0.666635 | 0.663818 | 0.8629 | 983 | 1054 | -71 | 0.638276 | 0.638586 | 0.638750 | +0.000164 | -0.000105 | 1176.1s | KILL | none |
| 9(i) | RT-1213 | within-fold deranged scalar control | n/a | 0.630245 | 0.666912 | 0.664874 | 0.8712 | 902 | 1128 | -226 | 0.638276 | 0.638586 | 0.638855 | +0.000269 | n/a | 1176.1s | control exceeded candidate | none |
| 3/IM3 | RT-1214 | frozen AR(2)-state Kalman/NIS observer | RT-401 | 0.630817 | 0.670067 | 0.667936 | 0.8836 | 927 | 1025 | -98 | 0.638276 | 0.638586 | 0.638722 | +0.000135 | n/a | 4120.5s | KILL | none |
| 3/IM3 | RT-1215 | frozen Hankel-DMD observer | RT-401 | 0.631331 | 0.673238 | 0.669922 | 0.8859 | 920 | 931 | -11 | 0.638276 | 0.638586 | 0.638812 | +0.000226 | n/a | 4120.5s | KILL; rho>0.85 | none |
| 8 | RT-1216 | weighted conformal test martingale | RT-1217 | 0.634279 | 0.673986 | 0.670290 | 0.8653 | 992 | 1027 | -35 | 0.638276 | 0.638586 | 0.639523 | +0.000937 | +0.002135 | 1651.5s | KILL | none |
| 8 | RT-1217 | matched unweighted CTM control | n/a | 0.629803 | 0.670833 | 0.668155 | 0.8683 | 999 | 1044 | -45 | 0.638276 | 0.638586 | 0.638799 | +0.000212 | n/a | 1651.5s | negative control | none |
| 8 | RT-1218 | parameter-free e-value aggregation | RT-401 | 0.525678 | 0.536757 | 0.540505 | 0.1341 | 2202 | 4369 | -2167 | 0.638276 | 0.638586 | 0.636373 | -0.002214 | n/a | 1651.5s | KILL | none |

First-sweep status: exhausted. No Pilot 8 arm cleared continuation; no planned
New Avenues first-sweep candidate cleared a 5-fold confirmation gate.
