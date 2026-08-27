# GPU TABULAR 2026 PLAN

Date: 2026-08-27  
Branch: `research/gpu-tabular-2026`  
Base: `origin/research/catboost-specialist-2026` (`0ca415d`), the latest scientific lineage containing learner-diversity and CatBoost-specialist results plus RT600 research artifacts.

## Question

Test whether neural tabular learners trained on the same legal causal 500-feature bank and the same RT-401-matched training rows add ranking diversity beyond LightGBM/CatBoost.

Learners:

| key | label | implementation |
|---|---|---|
| `tabm` | GPU-01 TabM | official `tabm.TabM` package |
| `realmlp` | GPU-02 RealMLP | `pytabkit.RealMLP_TD_Classifier` |

No new features, synthetic data, validation-score tuning, test data, lockbox data, RT ID allocation, production branch edit, or `research/RESULTS.csv` update is authorized here.

## Matched Contract

The scripts reuse:

| item | value |
|---|---|
| feature bank | `m00_core,m01_seq,m02_dist,m03_dyn,m04_resid,m06_loc,m07_bayes` |
| feature count | 500 |
| folds | canonical `research/folds/folds.parquet`, folds `0..4` |
| max training rows | `1,000,000` per outer fold |
| row sampler | same sequential RNG sampler as RT-401, seed `1` |
| calibration | fold-pure `SCDF_NSEEN` |
| controls | RT600 specialists `RT-300,RT-410..RT-415` and matched clone `RT-401` |
| scoring | `sbr.metric.ts_auc_flat`, only in `evaluate_gpu_oof.py` after all OOF folds exist |

## Stage A: Hardware Benchmark, No Score

Run on the RTX 4090 only:

```bash
bash research/scripts/gpu_tabular/bootstrap_gpu.sh
python research/scripts/gpu_tabular/benchmark_gpu.py --learner tabm
python research/scripts/gpu_tabular/benchmark_gpu.py --learner realmlp
```

The benchmark loads one outer-training fold, materializes the cached feature matrix, preprocesses once, trains 2-3 epochs, and records preprocessing time, seconds/epoch, batches/sec, RAM, VRAM, CUDA device, package versions, and projected five-fold runtime.

No TS-AUC, replacement score, pair flow, or validation-fold ranking metric is computed in Stage A.

`benchmark_gpu.py` writes `research/reports/gpu_tabular_2026/FROZEN_GPU_CONFIG.json`. Commit and push that file before scoring:

```bash
git add research/reports/gpu_tabular_2026/FROZEN_GPU_CONFIG.json
git commit -m "Freeze GPU tabular benchmark config"
git push
```

The training wrappers refuse to run scored OOF training unless the frozen config is tracked, clean, and contained in the upstream branch.

## Compute Selection Rule

Configuration changes are allowed before any TS-AUC is computed and only from runtime/VRAM evidence.

Order:

| learner | reduction order |
|---|---|
| TabM | batch size for GPU utilization/OOM, epoch cap, `k`, `d_block` |
| RealMLP | batch size for GPU utilization/OOM, epoch cap with early stopping, hidden width/layers |

The default quota assumption is 15 total hours, split across two learners with a 0.90 training budget fraction.

## Scored Remote Run

After the frozen config commit is pushed:

```bash
bash research/scripts/gpu_tabular/run_tabm.sh
bash research/scripts/gpu_tabular/run_realmlp.sh
python research/scripts/gpu_tabular/evaluate_gpu_oof.py
```

If cached data and OOF controls are in another checkout or mounted artifact root:

```bash
export SBR_ARTIFACT_ROOT=/path/to/artifact/root
```

Fold checkpoints and assembled OOF files are written under `research/oof/gpu_tabular_2026` by default.

## Resume

The runners save `fold_<k>_pred.npy` and `fold_<k>_meta.json` immediately after each completed fold and assemble `<learner>_oof.npy` after every fold. `run_tabm.sh` and `run_realmlp.sh` pass `--resume`; completed folds with finite predictions and matching validation-row length are skipped.

Use `--force` only to overwrite completed folds.

## Evaluation

`evaluate_gpu_oof.py` refuses partial OOF. After all five folds are present, it reports for each learner:

| metric family | outputs |
|---|---|
| standalone | mean TS-AUC, pooled TS-AUC, per-fold TS-AUC, fold std |
| diversity | within-`t` rho vs `RT-401` |
| cell diagnostics | dominant-cell AUC |
| pair flow | whole, dominant, mature-vs-never, mature-vs-prebreak repair/damage/net |
| nested replacement | E0 original RT600, E1 six specialists + `RT-401`, E2 six specialists + neural candidate |

Replacement selection is outer-fold pure: each outer fold chooses the replaced specialist using only the other four folds.

Primary metric: `marginal_vs_clone = E2 - E1`.

Gates:

| verdict | condition |
|---|---|
| `KILL` | `marginal_vs_clone < +0.0010` |
| `INTERESTING` | `marginal_vs_clone >= +0.0010` |
| `PROMOTION_WORTHY` | `>= +0.0015`, at least `4/5` positive folds, dominant pair net `> 0` |
| `SERIOUS` | `>= +0.0030`, at least `4/5` positive folds, dominant and mature-vs-never pair flow `> 0` |
| `MAJOR` | `>= +0.0050` |

Low correlation alone is not success. Combination with CatBoost/RT1257 is not part of this package.
