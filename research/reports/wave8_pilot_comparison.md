# WAVE 8 -- FIVE-PILOT COMPARISON (fold 0)

| mechanism | matched control | standalone Δ | dominant-cell Δ | marginal Δ vs clone | repairs | damage | net pair lift | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| SST | RT-990 | -0.00331 | -0.00060 | -0.00031 | 719 | 830 | -111 | NOT CLEARED |
| ORR* | RT-990 | +0.01167 | +0.01386 | -0.00032 | 769 | 694 | 75 | NOT CLEARED |
| PCFB | RT-990 | -0.00218 | -0.00270 | -0.00031 | 733 | 784 | -51 | NOT CLEARED |
| CFEP | RT-990 | -0.00490 | -0.00109 | -0.00031 | 711 | 808 | -97 | NOT CLEARED |
| TGMC | RT-990 | -0.00416 | -0.00170 | -0.00031 | 920 | 846 | 74 | NOT CLEARED |

*ORR's `RT-1021` is a blend on the RT600 7-stream ENSEMBLE score, not a single-model score like every other row's candidate -- its standalone/cell deltas here compare ensemble-scale to single-model-scale and read misleadingly large. The honest ORR comparison is RT600-alone vs RT600+repair, both at ensemble scale: whole -0.00005, dominant-cell -0.00004 (`research/reports/wave8_orr.json`). Its `marginal_vs_clone` and `net_pair_lift` columns above are computed consistently with the other rows and remain valid.