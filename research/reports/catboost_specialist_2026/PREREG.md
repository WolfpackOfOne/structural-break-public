# CATBOOST SPECIALIST ACTIVATION 2026 — PREREGISTRATION

Date: 2026-08-27
Branch: `research/catboost-specialist-2026`
Base: `research/learner-diversity-2026@7e5ee4c`

No lockbox, test, production, feature-generation, feature-selection,
hyperparameter-search, seed-search, blend-weight, router, or stacker change is
authorized. `AGENTS.md` is absent from the base SHA; `research/PROTOCOL.md` is
binding.

## Frozen CatBoost Learner

Use the exact `RT-1251` CatBoost learner settings:

`iterations=600`, `learning_rate=0.05`, `depth=6`, `l2_leaf_reg=5.0`,
`loss_function=Logloss`, `eval_metric=Logloss`, `bootstrap_type=Bernoulli`,
`subsample=0.7`, `rsm=0.7`, `border_count=127`, `thread_count=2`,
`allow_writing_files=False`, `verbose=False`.

For each specialist reimplementation, the only seed used is the incumbent
specialist training seed. This preserves the existing specialist row-sampling
convention and is not a seed search.

## Existing Controls And Integration

Original RT600 is the frozen seven-member specialist set:
`RT-300,RT-410,RT-411,RT-412,RT-413,RT-414,RT-415`.

Canonical integration is fold-pure equal-weight `SCDF_NSEEN`. Fold `k`
calibration is fitted only on folds other than `k`.

Matched LightGBM replacement control for a single CatBoost specialist is the
existing exchangeable seed clone `RT-401` inserted into the same specialist
slot. For a conditional hybrid with multiple replacements, the matched control
uses the first unused seed clones from `RT-401..RT-406` assigned in frozen
specialist order.

Pair-flow diagnostics use 64 same-`t` pairs per time point, seed `20260827`,
on the same split definitions used in Learner Diversity 2026: whole,
dominant-cell, mature-vs-never, and mature-vs-prebreak.

## IDs

| ID | arm |
|---|---|
| `RT-1254` | CSA-01 primary CAT-413: RT-413 specialist setup reimplemented with frozen CatBoost learner. |
| `RT-1255` | CSA-02 conditional CAT-300: RT-300 setup reimplemented with frozen CatBoost learner. |
| `RT-1256` | CSA-02 conditional CAT-410: RT-410 setup reimplemented with frozen CatBoost learner. |
| `RT-1257` | CSA-03 conditional hybrid seven-member ensemble replacing all surviving incumbent slots. |

CSA-00 is descriptive and ID-free because it uses existing OOF vectors only.

## CSA-00 — Zero-Training 8th Member

Use existing OOF vectors only:

- E0: original seven-specialist RT600.
- E1: original seven specialists plus `RT-401`.
- E2: original seven specialists plus `RT-1251` CatBoost.

Report `E2-E1`, `E2-E0`, per-fold deltas, pair flow, and mature pair nets. This
cannot stop CSA-01.

## CSA-01 — CAT-413 Primary

Train exactly one CatBoost specialist with RT-413's specialist setup:

- modules: `m00_core,m01_seq,m02_dist,m03_dyn,m04_resid,m06_loc,m07_bayes`
- rows: canonical folds 0..4, no lockbox, uniform sampler, `700000` train rows
  per fold cap
- labels: binary row labels
- seed convention: incumbent seed `0`
- learner: frozen CatBoost settings above, with `random_seed=0`

Primary comparison:

- E0: original RT600.
- E1: RT600 with RT-413 replaced by `RT-401`.
- E2: RT600 with RT-413 replaced by CAT-413.

Binding metric: `marginal_vs_clone = E2 - E1`.

Gates:

- KILL: `< +0.0010`
- INTERESTING: `>= +0.0010`
- PROMISING: `>= +0.0015`, at least `4/5` folds positive, and dominant-cell
  pair net versus E0 positive
- SERIOUS: `>= +0.0030`, at least `4/5` folds positive, and dominant plus
  mature pair flow versus E0 positive

If CSA-01 is KILL, stop. Do not train CAT-300 or CAT-410.

## CSA-02 — Conditional CAT-300 And CAT-410

Run only if CSA-01 `marginal_vs_clone >= +0.0010`.

Train exactly:

- CAT-300: RT-300 modules, rows, labels, folds, uniform row sampler,
  `1000000` train row cap, incumbent seed `0`, frozen CatBoost learner.
- CAT-410: RT-410 modules, rows, labels, folds, uniform row sampler,
  `900000` train row cap, incumbent seed `0`, frozen CatBoost learner.

Evaluate each with the same fixed-slot replacement-vs-`RT-401` protocol.

## CSA-03 — Conditional Hybrid

Run only if at least two of CAT-413, CAT-300, CAT-410 have
`marginal_vs_clone >= +0.0010`.

Construct one seven-member hybrid replacing only surviving incumbent slots with
their CatBoost versions. Keep equal weights and canonical fold-pure SCDF. Do
not optimize weights.

Binding metric: hybrid `marginal_vs_clone` against the matched LightGBM clone
control. Promotion-worthy requires `>= +0.0015`, at least `4/5` positive folds,
dominant-cell net positive, and mature-vs-never net positive. SERIOUS is
`>= +0.0030`; MAJOR is `>= +0.0050`.

## Reporting

Create only `results.json`, `results.csv`, `FINAL.md`, and minimal runner code
after this preregistration. Append `research/RESULTS.csv` only through the
runner. Update experiment map, RDOF ledger, failed-experiment ledger, and
status as required. Record training runtime and production inference
implications. Run research hygiene, `git diff --check`, and `py_compile`.
