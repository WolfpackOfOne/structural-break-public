# Oracle Information Frontier 2026

**ORACLE / DIAGNOSTIC -- NOT DEPLOYABLE.** This study uses true/pseudo
boundaries and future post-boundary observations. It does not modify production
features, submission notebooks, champion selection, `src/sbr/production/`, the
old lockbox, or reduced test data.

## Executive Result

The 2026 training data does **not** show a 2025-like `~0.90` known-boundary
information frontier under this fixed diagnostic pipeline. The strongest 2026
known-boundary result is `0.6861` at `h=500`, but that uses only one third of
the dev series. The full-post known-boundary result on all dev series is
`0.6497 +/- 0.0035` across pseudo-tau seeds, bootstrap CI `[0.6327, 0.6569]`
on the primary seed.

This points to the 2026 data/information structure being materially harder than
the 2025 offline task. There is some headroom in mature/full-post settings, but
the measured headroom is not huge at the horizons carrying most real-time
decision mass.

## Data And Protocol

- Branch base: `12ea3266a2e8bd43fea4606e4b1754b8515226a7`.
- 2026 series used: canonical dev folds `0..4`, 8,000 series.
- 2026 excluded: fold `-1`, `X_test.reduced`, `y_test.reduced`,
  `y_test_index.reduced`, leaderboard data.
- 2026 store:
  `/path/to/workspace/structural-break-claude-wave3/cache/store`.
- 2026 current legal OOF:
  `/path/to/workspace/structural-break-claude-wave3/research/oof/RT-300.npy`.
- Pseudo-tau seeds: `0, 1, 7, 42, 2026`.
- No-break pseudo tau: drawn from positive tau distribution inside `n_online`
  strata, subject to `pseudo_tau + h <= n_online` for finite horizons and
  `pseudo_tau < n_online` for `FULL`.
- Model capacity: fixed 150-tree LGBM for the 2026 frontier. The 2025 benchmark
  used 500 trees. No hyperparameter sweep was run.
- Runtime: 6,989.7 seconds for the full 2026 frontier.

Hashes:

| artifact | sha256 |
|---|---|
| 2026 `cache/store/meta.parquet` | `2c6aab9b65fc30740bf0ec1109569567af7c5a6ee938264da9f6c05920bd3971` |
| 2026 `cache/store/values.npy` | `10c22b007a6d0084e76a596f37f2d460ed5bc6143fada7e9605eeed0924bc06b` |
| 2026 `research/folds/folds.parquet` | `ba4f71fee8fcb30cc808234584a924ff0b0f0529e5bbd1fcbd0b4b780ebcc312` |
| 2026 `research/oof/RT-300.npy` | `bd3e6456fef300a7d0fefacec92095724ab2401fbe046c28733c9600b96879d3` |

## Frontier Table

All AUCs are ordinary series ROC AUC on one row per eligible series. Values are
means across pseudo-tau seeds; `std` columns are in
`oracle_information_frontier_2026.csv`.

| h | n_pos | n_neg | single | logistic | LGBM basic | LGBM rich | current RT-300 | headroom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3975 | 4025 | 0.5149 | 0.5178 | 0.5128 | 0.5181 | 0.5223 | -0.0041 |
| 2 | 3956 | 4025 | 0.5251 | 0.5129 | 0.5192 | 0.5246 | 0.5245 | +0.0001 |
| 3 | 3932 | 4025 | 0.5358 | 0.5266 | 0.5246 | 0.5292 | 0.5267 | +0.0024 |
| 5 | 3897 | 4025 | 0.5366 | 0.5306 | 0.5266 | 0.5313 | 0.5306 | +0.0008 |
| 10 | 3809 | 4025 | 0.5363 | 0.5348 | 0.5430 | 0.5431 | 0.5463 | -0.0032 |
| 20 | 3640 | 3979 | 0.5397 | 0.5379 | 0.5511 | 0.5552 | 0.5610 | -0.0058 |
| 30 | 3492 | 3928 | 0.5392 | 0.5410 | 0.5594 | 0.5746 | 0.5709 | +0.0037 |
| 50 | 3249 | 3843 | 0.5498 | 0.5525 | 0.5884 | 0.5855 | 0.5895 | -0.0040 |
| 75 | 2976 | 3752 | 0.5507 | 0.5573 | 0.5984 | 0.6048 | 0.6057 | -0.0010 |
| 100 | 2730 | 3636 | 0.5692 | 0.5653 | 0.6024 | 0.6161 | 0.6180 | -0.0019 |
| 150 | 2319 | 3427 | 0.5652 | 0.5881 | 0.6329 | 0.6373 | 0.6397 | -0.0025 |
| 200 | 1988 | 3243 | 0.5861 | 0.5894 | 0.6418 | 0.6640 | 0.6530 | +0.0110 |
| 300 | 1431 | 2808 | 0.6113 | 0.5832 | 0.6413 | 0.6641 | 0.6637 | +0.0004 |
| 500 | 646 | 2021 | 0.6170 | 0.6311 | 0.6643 | 0.6861 | 0.7145 | -0.0284 |
| FULL | 3975 | 4025 | 0.5917 | 0.5892 | 0.6418 | 0.6497 | 0.6100 | +0.0396 |

Threshold crossings for LGBM rich:

| threshold | first h |
|---:|---|
| 0.60 | 75 |
| 0.65 | 200 |
| 0.70 | not reached |
| 0.75 | not reached |
| 0.80 | not reached |
| 0.85 | not reached |
| 0.90 | not reached |

## Pseudo-Tau Sensitivity

Pseudo-tau assignment matters, but it does not change the conclusion. LGBM-rich
seed standard deviation ranges from about `0.0023` to `0.0086`. At `FULL`, seed
range is `0.6443..0.6541`; at `h=200`, seed range is `0.6590..0.6684`.

## Current Legal Model Headroom

The current model comparison is horizon-matched: positives are scored at
`tau + h - 1`, no-break series at `pseudo_tau + h - 1`, and `FULL` at the final
online observation.

| h | oracle AUC | current RT-300 AUC | headroom |
|---|---:|---:|---:|
| 5 | 0.5313 | 0.5306 | +0.0008 |
| 20 | 0.5552 | 0.5610 | -0.0058 |
| 50 | 0.5855 | 0.5895 | -0.0040 |
| 100 | 0.6161 | 0.6180 | -0.0019 |
| 200 | 0.6640 | 0.6530 | +0.0110 |
| 500 | 0.6861 | 0.7145 | -0.0284 |
| FULL | 0.6497 | 0.6100 | +0.0396 |

Bootstrap on the primary seed:

| h | oracle CI | headroom CI |
|---|---|---|
| 5 | `[0.5227, 0.5496]` | `[-0.0130, 0.0222]` |
| 20 | `[0.5387, 0.5627]` | `[-0.0224, 0.0100]` |
| 50 | `[0.5685, 0.5931]` | `[-0.0244, 0.0087]` |
| 100 | `[0.6088, 0.6365]` | `[-0.0109, 0.0199]` |
| 200 | `[0.6503, 0.6799]` | `[-0.0000, 0.0309]` |
| FULL | `[0.6327, 0.6569]` | `[0.0192, 0.0500]` |

## Known-Boundary Advantage

Unknown-boundary scans were run on the key horizons only, using a fixed 17-point
candidate grid and simple max-statistic features. This is a lightweight
diagnostic, not a tuned offline changepoint system.

| h | true-boundary oracle | unknown-boundary scan | penalty |
|---|---:|---:|---:|
| 5 | 0.5362 | 0.5008 | 0.0354 |
| 20 | 0.5521 | 0.5044 | 0.0477 |
| 50 | 0.5799 | 0.5398 | 0.0401 |
| 100 | 0.6214 | 0.5463 | 0.0751 |
| 200 | 0.6648 | 0.5867 | 0.0781 |
| 500 | 0.6905 | 0.6077 | 0.0828 |
| FULL | 0.6443 | 0.5154 | 0.1289 |

The penalty grows with mature/full information. Known tau is valuable, but even
with known tau the rich model does not approach `0.90`.

## Leakage Controls

Primary-seed controls on key horizons:

| h | metadata-only AUC | permuted-label AUC | random-boundary-both AUC |
|---|---:|---:|---:|
| 5 | 0.5381 | 0.5087 | 0.5088 |
| 20 | 0.5327 | 0.4904 | 0.5155 |
| 50 | 0.5417 | 0.5222 | 0.5389 |
| 100 | 0.5618 | 0.4955 | 0.5690 |
| 200 | 0.5868 | 0.4937 | 0.5987 |
| 500 | 0.6319 | 0.5152 | 0.6704 |
| FULL | 0.5379 | 0.5075 | 0.5821 |

Permutation controls are near chance. Metadata-only and random-boundary controls
rise at long horizons, especially `h=500`, which means composition/length effects
are material there. Interpret long-horizon finite results with that caveat.

## Break-Family Frontier

Family labels are simple diagnostic full-post effect categories, not a new
validated taxonomy. Counts among break series: scale 2,258; trend 708;
distribution 504; dependence 192; location 37; weak/unclassified 276.

| family | h=5 | h=20 | h=50 | h=100 | h=200 | FULL |
|---|---:|---:|---:|---:|---:|---:|
| dependence | 0.5217 | 0.5686 | 0.6194 | 0.6575 | 0.7677 | 0.6911 |
| distribution | 0.5741 | 0.5750 | 0.6137 | 0.6539 | 0.6858 | 0.6334 |
| location | 0.5337 | 0.4264 | 0.4359 | 0.6803 | n=1 | 0.7040 |
| scale | 0.5304 | 0.5528 | 0.5828 | 0.6264 | 0.6739 | 0.6587 |
| trend | 0.5308 | 0.5380 | 0.5441 | 0.5837 | n=4 | 0.6903 |
| weak/unclassified | 0.5378 | 0.5262 | 0.5108 | 0.5236 | 0.5364 | 0.3881 |

Dependence and distribution categories are easiest at medium/mature horizons;
scale is the largest family and improves steadily but remains far from `0.90`.
Weak/unclassified is intrinsically hard under this feature bank.

## Score-Gap Decomposition

Diagnostic, not causal accounting:

| component | estimate |
|---|---:|
| 2025 local rich benchmark A | 0.6643 |
| 2026 FULL known-boundary B | 0.6497 |
| A - B, local apples-to-apples gap | +0.0146 |
| 2026 h=100 known-boundary C | 0.6161 |
| B - C, limited-post penalty vs h=100 | +0.0336 |
| 2026 FULL unknown-boundary D | 0.5154 |
| B - D, known-boundary advantage | +0.1289 |
| 2026 FULL current RT-300 E | 0.6100 |
| B - E, model-extraction gap with full future info | +0.0396 |

Because the local 2025 pipeline achieves only `0.6643`, it is **not** a valid
reproduction of the historical `~0.90` winning stack. Therefore the most
defensible comparison is not "our code gets 0.90 on 2025 but 0.65 on 2026"; it
is "the same diagnostic feature/model family is weak on both local 2025 and 2026,
and the 2026 true-boundary/full-post result still does not reveal hidden
0.90-class signal."

## Main Conclusion

The evidence supports **Case C / mixed-low**: the 2026 DGP and real-time
information structure appear materially harder, and the current `~0.62` TS-AUC
is plausibly close to the reachable real-time frontier for early and medium
horizons under this research stack. There is measurable recoverable headroom in
full-future diagnostics and at `h=200`, but not enough to explain a `0.90` to
`0.62` collapse as merely poor model extraction.

## Research Consequences

1. Improve localization/unknown-boundary evidence before adding more
   true-boundary features; the FULL known-vs-unknown gap is the largest measured
   penalty.
2. Distill offline oracle scores into legal causal models, but expect the gain
   to concentrate in mature/full-post regimes rather than `h <= 100`.
3. Focus mature-break specialists around dependence, distribution, and robust
   scale evidence; weak/unclassified breaks remain hard even with future data.
4. Audit long-horizon composition effects before trusting `h=500` conclusions;
   metadata-only and random-boundary controls are high there.
5. Do not chase 2025 `~0.90` as the direct target for 2026 unless a stronger
   public 2025 pipeline first reproduces high-0.80s on the actual 2025 data.

## Limitations

- 2026 LGBM used 150 trees for runtime control, not the suggested 500-1000.
- Unknown-boundary search used a fixed lightweight 17-candidate scan, not a
  tuned changepoint model.
- Controls were run on the primary pseudo seed for key horizons, not all seeds.
- The break-family labels are simple diagnostic categories derived from
  full-post effects; they are not a validated stable taxonomy.
- Finite large horizons have strong eligibility/composition changes.
- The 2025 benchmark validates local data discovery but not 2025 winner-level
  implementation quality.
