# DATA_FORENSICS_2026 -- FINAL REPORT

Date: 2026-08-28
Branch: `research/data-forensics-2026`
Scoring/analysis SHA: `d44f1d4`

## Scope

No model was trained, no feature module was created, no OOF vector was written, and no RT ID was consumed. All row-level diagnostics use canonical dev folds `0..4`; lockbox labels, predictions, and values are not summarized.

## Headline Read

Dev population: `8000` series, `4032524` online rows, `3975` break series, `1033242` positive rows.
Reconstructed RT600 mean TS-AUC: `0.625811342`. Reconstructed RT-1264 mean TS-AUC: `0.628189820`. Delta: `+0.002378478`.
Fold deltas RT-1264 minus RT600: `+0.001170538, +0.002432860, +0.002120324, +0.004772096, +0.001396572`.

The fold-mean delta is positive on all five folds, so the `RT-1264` gain is not a single-fold artifact. The largest fold contribution is fold 3, but the sign is stable. Slice tables below use pooled TS-AUC inside each slice, so `whole_dev` does not numerically equal the fold-mean headline.

## Largest Slice Deltas

| slice | rows | positives | RT600 | RT-1264 | delta |
|---|---:|---:|---:|---:|---:|
| `rel_00_10` | 406082 | 11323 | 0.528812535 | 0.535919076 | +0.007106541 |
| `mature_vs_prebreak` | 1104487 | 667695 | 0.660950889 | 0.666104898 | +0.005154009 |
| `hist_len_low` | 1337481 | 348087 | 0.622637499 | 0.627416697 | +0.004779198 |
| `fold_3` | 802506 | 205339 | 0.617507588 | 0.622279684 | +0.004772096 |
| `online_len_low` | 461851 | 115264 | 0.591526369 | 0.595738393 | +0.004212024 |
| `t_100_199` | 684972 | 116925 | 0.609629884 | 0.613841600 | +0.004211716 |
| `online_len_mid` | 1345654 | 344506 | 0.613543647 | 0.617266550 | +0.003722904 |
| `rel_10_25` | 603000 | 56342 | 0.590051529 | 0.586419533 | -0.003631996 |

## Fixed Cells

| cell | RT600 | RT-1264 | delta | pair net | repairs | damage |
|---|---:|---:|---:|---:|---:|---:|
| `whole_dev` | 0.625626935 | 0.627893473 | +0.002266539 | 249 | 6094 | 5845 |
| `dominant_cell` | 0.664277099 | 0.667255183 | +0.002978083 | 137 | 4241 | 4104 |
| `mature_vs_never` | 0.665430880 | 0.667654188 | +0.002223308 | 299 | 4297 | 3998 |
| `mature_vs_prebreak` | 0.660950889 | 0.666104898 | +0.005154009 | 291 | 3441 | 3150 |
| `near_boundary` | 0.569857250 | 0.572282330 | +0.002425080 | 143 | 5584 | 5441 |
| `late_never_vs_mature` | 0.673649197 | 0.676005932 | +0.002356735 | 251 | 3077 | 2826 |

The strongest interpretable cell is `mature_vs_prebreak`: `RT-1264` gains `+0.005154009` pooled TS-AUC and `+291` sampled pair net. That is the part of the problem previous arbitration attempts kept damaging. The weak spot is early relative time: `rel_10_25` loses `-0.003631996`, and pair flow there is `-114`. Deployment review should check whether the live stream distribution over early relative positions matches dev.

## High-Score Mass

| model | threshold | rows | positive capture | negative share | never-break share | prebreak share |
|---|---|---:|---:|---:|---:|---:|
| `RT600` | `top_1` | 40332 | 0.029757 | 0.237677 | 0.135228 | 0.102450 |
| `RT600` | `top_5` | 201668 | 0.116715 | 0.402012 | 0.239731 | 0.162282 |
| `RT600` | `top_10` | 403292 | 0.196963 | 0.495378 | 0.307663 | 0.187715 |
| `RT1264` | `top_1` | 40334 | 0.029830 | 0.235831 | 0.136535 | 0.099296 |
| `RT1264` | `top_5` | 201654 | 0.116863 | 0.401212 | 0.240992 | 0.160220 |
| `RT1264` | `top_10` | 403286 | 0.198122 | 0.492400 | 0.304739 | 0.187661 |

At the top 10% within each `t`, `RT-1264` captures 1,198 more positive rows than RT600 while reducing negative share from `0.495378` to `0.492400`. Prebreak share is essentially unchanged at top 10 and lower at top 1/top 5. Never-break share is mixed: slightly higher at top 1/top 5, lower at top 10. This does not look like a broad false-positive explosion, but never-break top-score mass remains the production-risk cell to watch.

## Raw Process

| slice | rows | z mean | abs(z) mean | abs(z) p95 | abs(z)>3 |
|---|---:|---:|---:|---:|---:|
| `phase_never_break` | 2015775 | 0.003347 | 0.769489 | 1.934797 | 0.006302 |
| `phase_prebreak_far` | 661307 | -0.000601 | 0.768609 | 1.929044 | 0.006012 |
| `phase_prebreak_near` | 322200 | 0.001503 | 0.775377 | 1.950434 | 0.007232 |
| `phase_postbreak_early` | 327383 | 0.007740 | 0.815590 | 2.048107 | 0.012869 |
| `phase_postbreak_mature` | 705859 | 0.170803 | 0.987321 | 2.061535 | 0.013109 |

The raw series itself says why simple thresholding has been hard. Never-break, far-prebreak, and near-prebreak rows have nearly identical historical-z summaries. Early postbreak rows move only modestly. Mature postbreak rows show a clearer mean/absolute-z shift, but the tail-rate separation is still small (`abs(z)>3` is `0.013109` for mature postbreak versus `0.006302` for never-break). The remaining signal is not a one-dimensional amplitude anomaly.

## Feature Bank Audit

Deterministic sample rows: `80651` (`dev_rows[::50]`). For the full 500-column bank, participation-ratio effective rank is `21.282`, top eigenvalue share is `0.182415`, and mean absolute pairwise correlation is `0.120197`.

| module | cols | mean finite | near constant | effective rank | top eigen share | mean abs corr |
|---|---:|---:|---:|---:|---:|---:|
| `m00_core` | 151 | 0.906579 | 0 | 9.963 | 0.275060 | 0.182515 |
| `m01_seq` | 60 | 0.997307 | 0 | 6.430 | 0.357558 | 0.300046 |
| `m02_dist` | 59 | 0.890905 | 0 | 11.297 | 0.228151 | 0.152789 |
| `m03_dyn` | 60 | 0.930614 | 0 | 19.901 | 0.134325 | 0.071715 |
| `m04_resid` | 60 | 0.991583 | 0 | 7.111 | 0.304695 | 0.234410 |
| `m06_loc` | 60 | 0.861340 | 0 | 21.836 | 0.140066 | 0.086451 |
| `m07_bayes` | 50 | 0.999354 | 0 | 6.839 | 0.326454 | 0.275833 |

The frozen 500-column bank has participation-ratio effective rank `21.282`, not anything close to 500. `m03_dyn` and `m06_loc` are the most internally diverse modules by this audit; `m01_seq`, `m04_resid`, and `m07_bayes` are much more compressed. This supports the recent empirical pattern: more learner or mechanism diversity is likelier to matter than adding near-duplicate columns inside the same transform family.

## Interpretation

- This report is descriptive only. It does not authorize a model, threshold, router, feature, or production change.
- Any future experiment motivated by these slices needs its own preregistration before scoring.
- `RT-1264` remains an internal OOF result pending separate deployment feasibility and confirmation work.
- For deployment review, the concrete checks are early-relative-position behavior and never-break top-score mass, not global mean AUC.
