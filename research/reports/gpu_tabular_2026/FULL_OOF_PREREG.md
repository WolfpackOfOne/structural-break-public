# GPU TABULAR 2026 — FULL 5-FOLD OOF PREREGISTRATION

Date: 2026-08-27
Branch: `research/gpu-tabular-2026`
Base: `research/catboost-specialist-2026` lineage (`0ca415d`), plus this
branch's own hardware-benchmark history (`e39a69c`, `0bcb7ae`).

`research/PROTOCOL.md` is binding. No lockbox, test, production, feature
generation, feature selection, architecture tuning, seed search, blend
tuning, CatBoost tuning, partial-fold TS-AUC, or predictive-score-based
setting choice is authorized. No combination with `RT-1257` or with each
other (TabM + RealMLP) is authorized in this program.

## Scientific question

Can full-scale TabM or RealMLP trained on the same legal causal information
produce useful same-`t` ranking alpha beyond an exchangeable LightGBM
replacement?

## Frozen basis

This program is authorized specifically because the RTX 4090 cloud hardware
benchmark (Crunch submission `76357`, task `run-3e834e0f`) proved the full,
un-shrunk TabM and RealMLP configurations fit the competition compute budget:
TabM projects to `3.63606` hours for five folds, RealMLP to `0.90924` hours,
combined `4.545298927912005` hours, against a 15-hour/2-learner/0.90-fraction
quota. No batch size, epoch cap, `k`, `d_block`, hidden width, or layer count
reduction was required. Both configurations are frozen exactly as benchmarked
in `research/reports/gpu_tabular_2026/FROZEN_GPU_CONFIG.json`
(commit `d506aa6`, pushed).

This is a distinct authorization from the earlier `RT-1250`/`RT-1252` Learner
Diversity 2026 arms, which were ruled **INFEASIBLE** on
`research/learner-diversity-2026` because the installed official/default
full-scale configuration did not fit the compute budget there and shrinking
was not authorized. The RTX 4090 benchmark on this branch removes that
blocker: the same default/official-scale configurations now fit without any
shrink, so a fresh, separately-numbered arm is preregistered rather than
reopening `RT-1250`/`RT-1252`.

## Candidates

| key | label | implementation | RT ID |
|---|---|---|---|
| `tabm` | GPU-01 TabM | official `tabm.TabM` package | `RT-1258` |
| `realmlp` | GPU-02 RealMLP | `pytabkit.RealMLP_TD_Classifier` | `RT-1259` |

Both `RT-1258` and `RT-1259` were verified unused on every local branch,
every remote branch, and full reachable git history before allocation.

Both candidates MUST complete full 5-fold OOF training before either is
evaluated. Neither may be stopped early because the other scores well; a
candidate may only stop early on an unrecoverable technical failure
(implementation bug, serialization bug, CUDA/package API mismatch — see
Technical Failure Rule below), never on a predictive-score basis.

## Matched contract

| item | value |
|---|---|
| feature bank | `m00_core,m01_seq,m02_dist,m03_dyn,m04_resid,m06_loc,m07_bayes` |
| feature count | 500 |
| folds | canonical `research/folds/folds.parquet`, folds `0..4` |
| max training rows | `1,000,000` per outer fold |
| row sampler | RT-401 sequential RNG convention, seed `1` |
| calibration | fold-pure `SCDF_NSEEN` |
| controls | RT600 specialists `RT-300,RT-410..RT-415` and matched clone `RT-401` |
| scoring | `sbr.metric.ts_auc_flat`, only in `evaluate_gpu_oof.py` after all OOF folds exist |

No neural candidate may consume RT600 score, LightGBM score, CatBoost score,
`tau` as a feature, future prefix information, the final online horizon, test
data, or lockbox data.

## Training order

TabM folds `0..4`, then RealMLP folds `0..4` (or an equally deterministic
resumable order). Fold completion logs may contain only fold number, runtime,
epochs, training/inner-validation diagnostics, RAM, VRAM, and checkpoint
status — never TS-AUC, pair flow, replacement gain, or outer-fold rank
correlation. No outer-fold TS-AUC is computed or printed after individual
folds; `evaluate_gpu_oof.py` is the only script in this package that computes
TS-AUC, and it refuses to run on an incomplete OOF vector.

## Evaluation (only after both candidates have complete 5-fold OOF)

Standalone: mean/pooled/per-fold TS-AUC, fold std. Diversity: within-`t` rho
vs `RT-401`. Diagnostics: dominant-cell AUC. Pair flow: whole, dominant,
mature-vs-never, mature-vs-prebreak repairs/damage/net.

Binding nested replacement test, outer-fold pure (fold `k`'s replaced
specialist is chosen using only the other four folds, then frozen and
evaluated on fold `k`):

- E0: original seven-specialist RT600.
- E1: six RT600 specialists + exchangeable matched RT-401 LightGBM clone.
- E2: six RT600 specialists + neural candidate.

Primary metric: `marginal_vs_clone = E2 - E1`. Also report `E2 - E0`, fold
deltas `E2-E1`, and folds positive.

## Gates

| verdict | condition |
|---|---|
| `KILL` | `marginal_vs_clone < +0.0010` |
| `INTERESTING` | `marginal_vs_clone >= +0.0010` |
| `PROMOTION_WORTHY` | `>= +0.0015`, `>=4/5` folds positive, dominant pair net `>0` |
| `SERIOUS` | `>= +0.0030`, `>=4/5` positive, dominant pair net `>0`, mature-vs-never pair net `>0` |
| `MAJOR` | `>= +0.0050` |

Low correlation alone is never success. A model with weaker standalone
performance may still survive if ensemble marginal is strong.

## No combination

No combination of TabM + RealMLP, TabM + RT-1257, RealMLP + RT-1257, TabM +
CatBoost, or RealMLP + CatBoost is authorized in this program, even if
results are excellent. Any heterogeneous combination is a separate
preregistered experiment.

## Technical failure rule

If a frozen model fails because of an implementation bug, serialization bug,
or CUDA/package API mismatch, the implementation may be fixed while
preserving the scientific configuration exactly (no frozen hyperparameter may
change). If OOM occurs despite the benchmark having succeeded, that learner
stops, is reported `TECHNICAL_FAILURE`, and is not silently reduced; the
other learner continues if legally possible.

## Infrastructure note: evaluation environment split

`evaluate_gpu_oof.py`'s binding nested-replacement test requires the RT600
specialist and `RT-401` clone OOF vectors (`research/oof/*.npy`). Those paths
are `.gitignore`d (`research/oof/`, `*.npy`) and are not present in a cold
Crunch-cloud checkout, only in this local/persistent research environment.
Consequently the Crunch submission (`submissions/H_gpu_tabular_full_oof.py`)
computes and prints only what is derivable from the materialized dev
population itself during `train()` — standalone TS-AUC (mean/pooled/per-fold/
std) and dominant-cell AUC, which need only the candidate's own OOF
predictions plus labels/`t`/age already present in the materialized store —
and checkpoints the two candidates' fold OOF predictions into
`model_directory_path`. The binding nested-replacement test (`E0`/`E1`/`E2`,
`marginal_vs_clone`, pair flow, verdict) is computed as a separate local
post-run step by copying the two `*_oof.npy` files back into
`research/oof/gpu_tabular_2026/` and running
`research/scripts/gpu_tabular/evaluate_gpu_oof.py` in this checkout, where
the control artifacts already exist. This does not touch
`research/RESULTS.csv`; that ledger remains append-only via `sbr.pipeline.run`
and is updated only in a separate post-run bookkeeping step after the cloud
result exists.

## Reporting

Create only `FULL_OOF_PREREG.md` (this file), `EXPERIMENT_ID_MAP.md` /
`RDOF_LEDGER.md` allocation entries, `submissions/H_gpu_tabular_full_oof.py`,
and minimal supporting code after this preregistration. Append
`research/RESULTS.csv` only after the cloud OOF result exists, and only
through the existing append-only convention — not from this preregistration
and not from the Crunch training job itself.
