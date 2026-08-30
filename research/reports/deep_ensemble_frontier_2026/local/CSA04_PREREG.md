# CSA-04 PREREGISTRATION -- COMPLETE CATBOOST SLOT SWEEP

Date: 2026-08-28
Branch: `research/deep-ensemble-frontier-local-2026`
Base: `research/deep-ensemble-frontier-2026@22d6633`
Lane: LOCAL

This preregistration authorizes only the LOCAL-lane CatBoost specialist sweep described in
`research/reports/deep_ensemble_frontier_2026/LANE_LOCAL.md` section L2. It does not authorize
lockbox or test-data access, production changes, feature generation, hyperparameter search, seed
search, blend-weight optimization, routers, stackers, or neural models.

## Binding Context

Production remains `RT-600` with external Crunch score `0.6268`.

The live positive mechanism is learner-family replacement inside existing RT-600 specialist slots.
The already measured CatBoost specialist arms are:

| Arm | Slot | Cols | `marginal_vs_clone` | Verdict |
|---|---|---:|---:|---|
| CAT-413 | `RT-413` | 500 | +0.001087151 | INTERESTING |
| CAT-300 | `RT-300` | 500 | +0.001029459 | INTERESTING |
| CAT-410 | `RT-410` | 261 | +0.000753095 | KILL |
| HYBRID `RT-1257` | `RT-413` + `RT-300` | -- | +0.002407205 | PROMOTION_WORTHY |

The original CSA program stopped after three slots because that was the preregistered scope, not
because `RT-411`, `RT-412`, `RT-414`, or `RT-415` failed. CSA-04 tests exactly those four
remaining slots.

## Preregistered Predictions

**P1 -- breadth hypothesis.** `RT-412` and `RT-415` (both 500 cols) yield `marginal_vs_clone`
near the approximately `+0.00105` observed for `RT-413`/`RT-300`. `RT-411` (239 cols) and
`RT-414` (170 cols) yield materially less, plausibly below the `+0.0010` gate as `RT-410`
(261 cols) did.

**P2 -- idiosyncrasy hypothesis, which contradicts P1 on the same two slots.** The alpha comes
from *heterogeneity*, so replacing the slots whose LightGBM idiosyncrasy is *hardest for CatBoost
to reproduce* should **destroy** more existing diversity than it adds. `RT-412`
(`extra_trees=True` + `sample_mode="per_series"`) and `RT-415` (`boosting="goss"`) are precisely
those slots. Under P2 they underperform P1's expectation and may be negative.

**P1 and P2 make opposite predictions about `RT-412` and `RT-415`.** That is what makes CSA-04 a
real experiment rather than a sweep. The outcome discriminates between "CatBoost is simply a
better learner on wide feature banks" and "the alpha is irreducibly about family mixing." Record
both before scoring.

**P3 -- interior maximum.** The `k`-slot hybrid curve is **non-monotone with an interior
maximum**. At `k = 7` every slot is CatBoost, the ensemble is single-family, and the mixing alpha
that produced `RT-1257` is gone by construction. Therefore some `k* < 7` maximizes
`marginal_vs_clone`. **Locating `k*` is the primary scientific deliverable of CSA-04**, and it is
a question no amount of additional single-slot testing answers.

## Frozen Arms

Use the exact `RT-1251` CatBoost learner settings:

`iterations=600`, `learning_rate=0.05`, `depth=6`, `l2_leaf_reg=5.0`,
`loss_function=Logloss`, `eval_metric=Logloss`, `bootstrap_type=Bernoulli`,
`subsample=0.7`, `rsm=0.7`, `border_count=127`, `thread_count=2`,
`allow_writing_files=False`, `verbose=False`.

The only seed used for each arm is the incumbent specialist's training seed. This preserves the
existing row-sampling convention and is not a seed search.

| ID | Arm | Replaced slot | Modules | Cols | `max_train_rows` | `random_seed` | Sampler |
|---|---|---|---|---:|---:|---:|---|
| `RT-1260` | CAT-411 | `RT-411` | `m02_dist,m03_dyn,m04_resid,m06_loc` | 239 | 900,000 | 1 | uniform |
| `RT-1261` | CAT-412 | `RT-412` | `m00_core,m01_seq,m02_dist,m03_dyn,m04_resid,m06_loc,m07_bayes` | 500 | 900,000 | 7 | `per_series` |
| `RT-1262` | CAT-414 | `RT-414` | `m07_bayes,m06_loc,m01_seq` | 170 | 700,000 | 3 | uniform |
| `RT-1263` | CAT-415 | `RT-415` | `m00_core,m01_seq,m02_dist,m03_dyn,m04_resid,m06_loc,m07_bayes` | 500 | 700,000 | 11 | uniform |
| `RT-1264` | CSA-04 best-`k` hybrid | selected survivor slots | frozen OOF streams | 7 streams | n/a | 20260827 pair-flow seed | frozen OOF |

## Open Questions Resolved Before Scoring

1. Control-array path: repository code confirms the submitted tree expects
   `<root>/research/oof/{ID}.npy`. `research/scripts/gpu_tabular/common.py::load_control_oof`
   and `submissions/H_gpu_tabular_full_oof.py::_try_binding_replacement_test` use that path.
2. CAT-412 sampling: the CatBoost runner must preserve `sample_mode="per_series"`. The local
   implementation has an explicit per-series branch in `sample_train_rows`; CAT-412 is invalid if
   this branch is not used.
3. CAT-415 GOSS caveat: GOSS is a LightGBM boosting mechanism with no CatBoost analogue. CAT-415
   will use the frozen Bernoulli bootstrap like all other CatBoost arms, will be reported under
   this caveat, and will still be eligible for the hybrid if it clears `+0.0010`. Excluding it only
   after seeing the number would be an unauthorized degree of freedom.
4. Hybrid ordering: add slots in descending measured single-slot `marginal_vs_clone`, across the
   six tested slots CAT-413, CAT-300, CAT-410, CAT-411, CAT-412, CAT-414, and CAT-415. Dominant
   pair net is reported as a diagnostic only, not an ordering rule.
5. RT-411 descriptive check: no persisted CatBoost feature-importance artifact is present in the
   current repository state. Prediction P1 therefore remains based on the frozen width/module
   contrast, and the final report will state this limitation.

## Execution Protocol

Train `RT-1260`, `RT-1261`, `RT-1262`, and `RT-1263` across all five canonical folds before any
of the four arms is evaluated. Training logs may contain fold number, runtime, row counts, and
checkpoint status. Training logs must not contain TS-AUC, pair flow, replacement gain, or rank
correlation.

OOF arrays are written under ignored `research/oof/` and are not committed.

No arm may stop early because of a predictive score. An arm may stop only for unrecoverable
technical failure, which must be recorded.

## Evaluation Protocol

Stage 1: evaluate four independent single-slot replacements under the nested slot-replacement
battery:

- `E0`: original seven-specialist RT-600.
- `E1`: RT-600 with the target slot replaced by matched seed clone `RT-401`.
- `E2`: RT-600 with the target slot replaced by the CatBoost candidate.
- Primary metric: `marginal_vs_clone = E2 - E1`.

Stage 2: let `S` be every slot with single-slot `marginal_vs_clone >= +0.0010` across CAT-413,
CAT-300, CAT-410, CAT-411, CAT-412, CAT-414, and CAT-415. Build hybrids for every `k` from `2`
through `|S|`, adding slots in descending measured single-slot `marginal_vs_clone`.

For each hybrid:

- `E2_k`: seven-member ensemble with the top-`k` surviving slots CatBoost-replaced.
- `E1_k`: matched control with the same `k` slots replaced by seed clones from `RT-401..RT-406`
  in frozen specialist order.
- Calibration: equal-weight fold-pure `SCDF_NSEEN`.
- Pair-flow diagnostics: 64 same-`t` pairs per time point, seed `20260827`, splits `whole`,
  `dominant_cell`, `mature_vs_never`, and `mature_vs_prebreak`.

`RT-1264` is allocated to the best-`k` hybrid by `marginal_vs_clone`. No other hybrid receives an
RT ID.

Mandatory regression check: if the frozen ordering puts CAT-413 and CAT-300 first, the `k = 2`
hybrid must reproduce `RT-1257`'s `marginal_vs_clone = +0.002407205` exactly. If it does not, halt
before interpreting any CSA-04 result.

## Gates

The frozen ladder from `PROGRAM_PLAN.md` applies:

| Verdict | Condition |
|---|---|
| `KILL` | `marginal_vs_clone < +0.0010` |
| `INTERESTING` | `>= +0.0010` |
| `PROMOTION_WORTHY` | `>= +0.0015`, `>=4/5` folds positive, dominant pair net `> 0` |
| `SERIOUS` | `>= +0.0030`, `>=4/5` positive, dominant pair net `> 0`, mature-vs-never pair net `> 0` |
| `MAJOR` | `>= +0.0050` |

Standalone AUC and low correlation are diagnostics only.

## Required Reporting

The final CSA-04 report must state all single-slot results, the full hybrid curve, the selected
best-`k` composition, the `k=2` regression-check result, and explicit adjudication of P1, P2, and
P3, including predictions that were wrong.
