# LA-01 -- Specialist Replacement Salvage Execution Preregistration

Date: 2026-08-26

Program preregistration: `research/reports/leaderboard_alpha_2026/PROGRAM_PREREG.md`
at `ed84d00`.

Pre-score execution commit: this file's commit.

## Allocated IDs

| ID | arm |
|---|---|
| `RT-1243` | LA-01 headline nested replacement by `m11_focus`/`m12_rdep` candidate |
| `RT-1244` | LA-01 secondary nested replacement by exchangeable seed clone `RT-401` |

No fixed-replacement diagnostic receives an experiment ID unless it is later
promoted to a scored arm under a new preregistration.

## Frozen Inputs

Use existing canonical OOF arrays only. No retraining.

- RT600 specialists: `RT-300`, `RT-410`, `RT-411`, `RT-412`, `RT-413`,
  `RT-414`, `RT-415`
- seed-clone control: `RT-401`
- candidate replacement streams:
  - `RT-731`: champion config plus `m11_focus`
  - `RT-751`: champion config plus `m12_rdep`

The artifact root is the existing local Wave5 worktree containing
`cache/store`, `research/folds/folds.parquet`, and `research/oof`.

## Calibration and Scoring

Use the existing canonical smooth time-conditioned calibration convention:
`wave5_lib.CANON_CAL`, currently `SCDF_NSEEN`.

Every composition contains exactly seven streams with equal weights.

For each outer fold `f`:

1. Define outer-training folds as `{0,1,2,3,4} \ {f}`.
2. For each candidate replacement pair `(replaced specialist, replacement
   candidate)`, score the composition on the four outer-training folds by inner
   cross-fitting:
   - for each inner held-out fold `g` in the outer-training set, fit each
     stream's SCDF map on `outer_train \ {g}` only,
   - apply it to fold `g`,
   - compute that fold's TS-AUC,
   - average the four inner-fold TS-AUC values.
3. Select the highest inner mean. Ties are resolved lexicographically by
   `(replacement_id, replaced_id)` to avoid discretionary choice.
4. Freeze the selected replacement and evaluate on outer fold `f`, fitting SCDF
   maps on the four outer-training folds only and applying them to fold `f`.

The seed-clone control follows the same nested rule, except the only replacement
candidate is `RT-401`.

E0 is the untouched RT600 seven-specialist ensemble evaluated by the same
outer-fold SCDF convention.

Primary metric:

`marginal_vs_clone = mean_ts_auc(RT-1243) - mean_ts_auc(RT-1244)`.

Also report `RT-1243 - E0`, `RT-1244 - E0`, per-fold deltas, pooled dev TS-AUC,
dominant-cell pair repairs/damage/net, mature-vs-never pair net,
mature-vs-prebreak pair net, within-`t` rank correlation with E0, and runtime.

Pair-flow sampling is fixed at 64 same-`t` pairs per time point, seed
`20260826`, candidate-vs-E0.

## Gate

KEEP/WEAK only if:

- `marginal_vs_clone >= +0.0015`
- at least four of five outer folds have positive `RT-1243 - RT-1244`
- dominant-cell net pair lift is positive

SERIOUS if `marginal_vs_clone >= +0.0030` and at least four folds are positive.

KILL if the primary marginal is below `+0.0015`. If KILL, specialist-replacement
salvage is closed.

