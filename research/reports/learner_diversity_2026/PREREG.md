# Learner Diversity 2026 Preregistration

Date: 2026-08-26
Branch: `research/learner-diversity-2026`
Base: `research/leaderboard-alpha-2026@d92860d`

## Question

Can a different tabular learner produce useful same-`t` pair orderings that
LightGBM does not, measured as marginal contribution to the RT-600 ensemble?

This is not a standalone leaderboard search. A learner matters only if it beats
an exchangeable LightGBM replacement.

## Prior Audit

One prior CatBoost row exists: `RT-A09-FAM-cat-B`, fold 0 screen, 2,500-series
screen store, `m00_core,m03_dyn`, 211 features. It is not a full canonical
five-fold, 500-feature, RT600 replacement test, so CatBoost is not duplicated in
the sense barred here. No prior TabM or RealMLP full replacement test was found.

## IDs

| ID | learner | status before score |
|---|---|---|
| `RT-1250` | TabM | full-scale infeasible under fixed default config |
| `RT-1251` | CatBoost | authorized full five-fold score |
| `RT-1252` | RealMLP | full-scale infeasible under fixed default config |

If at least two scored learners individually reach `marginal_vs_clone >= +0.0010`,
one combination gets `RT-1253`. No combination otherwise.

## Shared Protocol

- Data: 8,000-series dev population only, canonical folds
  `research/folds/folds.parquet`; no fold `-1`, no reduced/test files.
- Features: existing legal causal 500-column bank:
  `m00_core,m01_seq,m02_dist,m03_dyn,m04_resid,m06_loc,m07_bayes`.
- Training rows: same CHAMP row sampler as the matched LightGBM clone,
  `max_train_rows=1,000,000`, uniform row sample, seed `1`, replayed fold by
  fold exactly as `sbr.pipeline.run`.
- Matched LightGBM control: existing `RT-401` OOF artifact, the CHAMP full-bank
  seed-1 clone. No new LightGBM is trained.
- Labels/folds: existing row labels and canonical five outer folds.
- Calibration/integration: existing fold-pure `wave5_lib.CANON_CAL`
  (`SCDF_NSEEN`), seven members, equal weight. No blend-weight tuning.
- No RT600 score as model input. No new data generation. No feature columns. No
  routers, stackers, LA-04, production change, lockbox, or test data.

## Package Versions

Installed before any score: `catboost 1.2.10`, `tabm 0.0.3`,
`rtdl_num_embeddings 0.0.12`, `pytabkit 1.7.3`, `torch 2.13.0`,
`lightgbm 4.7.0`, `numpy 2.4.6`, `scipy 1.17.1`, `scikit-learn 1.9.0`.

## Learner Configurations

### LD-01 TabM / `RT-1250`

Official installed default from `pytabkit.models.sklearn.TabM_D_Classifier`:
`arch_type=tabm`, `tabm_k=32`, `num_emb_type=none`, `batch_size=256`,
`lr=2e-3`, `weight_decay=0`, `n_epochs=1_000_000_000`, `patience=16`,
`d_block=512`, `n_blocks=auto`, `dropout=0.1`, `tfms=[quantile_tabr]`,
`allow_amp=False`, seed `1`.

Full-scale infeasibility is declared before scoring: at 1,000,000 rows/fold,
this is at least 3,907 batches/epoch and at least 17 validation epochs per fold
before patience can fire, over five folds. Shrinking rows, `k`, width, epochs or
patience would be a forbidden retune, so LD-01 is recorded INFEASIBLE.

### LD-02 CatBoost / `RT-1251`

`CatBoostClassifier`: `iterations=600`, `learning_rate=0.05`, `depth=6`,
`l2_leaf_reg=5.0`, `loss_function=Logloss`, `eval_metric=Logloss`,
`bootstrap_type=Bernoulli`, `subsample=0.7`, `rsm=0.7`, `border_count=127`,
`random_seed=1`, `thread_count=2`, `allow_writing_files=False`. No early
stopping. Native CatBoost missing-value handling.

### LD-03 RealMLP / `RT-1252`

Official installed default from `pytabkit.models.sklearn.RealMLP_TD_Classifier`:
`hidden_sizes=[256,256,256]`, `act=selu`, parametric activations, `p_drop=0.15`,
`wd=2e-2`, `lr=4e-2`, `tfms=[one_hot,median_center,robust_scale,smooth_clip,embedding]`,
`num_emb_type=pbld`, `n_epochs=256`, `batch_size=256`, seed `1`.

Full-scale infeasibility is declared before scoring: 1,000,000 rows/fold implies
about 3,907 batches/epoch/fold and roughly five million batch steps over five
folds, plus full-matrix preprocessing/copies. Shrinking epochs or rows would be
a forbidden retune, so LD-03 is recorded INFEASIBLE.

## Binding Ensemble Test

For each scored learner and each outer fold:

1. Using only the other four folds, score all seven possible replacements of one
   RT600 specialist by the candidate.
2. Freeze the best replacement slot by inner mean TS-AUC.
3. Evaluate the frozen seven-member equal-SCDF blend on the held-out fold.

Headline arms:

- `E0`: original RT600 seven specialists.
- `E1`: six RT600 specialists plus `RT-401`, with the same fold-pure replacement
  selection rule.
- `E2`: six RT600 specialists plus the non-LightGBM candidate.

Primary metric: `marginal_vs_clone = E2 - E1`.

## Gates

- KILL: `marginal_vs_clone < +0.0010`.
- INTERESTING: `>= +0.0010`.
- SERIOUS: `>= +0.0030`, at least `4/5` folds positive, and dominant pair net
  `> 0`.
- MAJOR: `>= +0.0050`.
- Also KILL if the candidate is materially worse standalone, has `rho > 0.95`
  versus matched LightGBM, and adds no unique pair repairs.

Pair-flow diagnostics use 64 same-`t` pairs per time point, seed `20260827`,
reported for whole dev, dominant cell, mature-vs-never, and
mature-vs-prebreak.
