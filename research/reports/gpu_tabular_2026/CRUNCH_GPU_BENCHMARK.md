# Crunch GPU Benchmark Submission

Date: 2026-08-27

This package is a Crunch-cloud benchmark submission for GPU-01 TabM and GPU-02
RealMLP. It is benchmark-only and must not be used as a scored RT model.

## Entrypoint

Use:

```bash
submissions/G_gpu_tabular_benchmark.py
```

The file exposes the same public Real-Time Crunch interface as the deployed
RT600 submission:

```python
train(datasets, model_directory_path)
infer(datasets, model_directory_path)
INFER_PARALLELISM
```

`train()` runs the GPU hardware benchmark. `infer()` returns deterministic dummy
probabilities only so the submission remains contract-valid.

## Contract

The benchmark uses:

| item | value |
|---|---|
| feature bank | `m00_core,m01_seq,m02_dist,m03_dyn,m04_resid,m06_loc,m07_bayes` |
| feature count | 500 legal causal features |
| fold | canonical fold `0` as the held-out fold |
| training population | folds `1,2,3,4` only |
| max training rows | `1,000,000` |
| row sampler | RT-401 convention, seed `1` |
| learners | TabM and RealMLP |
| benchmark epochs | `2` or `3` only |

The benchmark does not use test labels, lockbox data, reduced-test data, RT IDs,
TS-AUC, five-fold scored training, or validation ranking metrics.

## Output

The final benchmark JSON is printed between these exact markers:

```text
=== GPU_TABULAR_FREEZE_JSON_BEGIN ===
...
=== GPU_TABULAR_FREEZE_JSON_END ===
```

It is also written to `GPU_TABULAR_FREEZE.json` in `model_directory_path`.

Required runtime fields include GPU name, CUDA version, package versions,
preprocessing seconds, seconds per epoch, batches per second, rows per second,
peak RAM, peak VRAM, projected five-fold hours, projected combined hours, and
`score_computed: false`.

## Cloud Notes

Dependencies are declared in `requirements.txt` for Crunch's whitelist install
phase. The benchmark does not run `pip` or `apt` from `train()`.

Run on a Crunch GPU size with CUDA available. The wrapper fails fast if
`torch.cuda.is_available()` is false.
