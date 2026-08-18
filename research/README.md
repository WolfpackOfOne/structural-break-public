# Research workspace — 2026 Real-Time Edition

This directory holds the research machine built for the ADIA Lab / CrunchDAO
Structural Break Challenge **Real-Time Edition**, and the record of what it found.
The reusable library lives in [`../src/sbr`](../src/sbr); everything here is the
experiment layer on top of it.

**Start with [`STATE_OF_RESEARCH.md`](STATE_OF_RESEARCH.md).** It is the summary of
record: current champion, per-fold scores, lockbox confirmation, what won, what
failed, and the ranked list of what to do next.

## Layout

| path | what it is |
|---|---|
| `STATE_OF_RESEARCH.md` | the summary of record |
| `PROTOCOL.md` | the binding research protocol — validation rules, the causal contract, file ownership |
| `RESULTS.csv` | experiment ledger, one row per run, appended under a file lock |
| `FAILED_EXPERIMENTS.md` | negative results, so nobody rediscovers them |
| `folds/folds.parquet` | **permanent** series-level folds + lockbox. Never regenerate |
| `reports/` | per-agent reports and machine-readable diagnostics |
| `scripts/` | data preparation, per-agent experiment drivers, analysis |

## Reproducing

The pipeline needs the official competition data, which is **not redistributed
here**. With `X_train.parquet`, `y_train.parquet` and `y_train_index.parquet` in
place:

```bash
pip install -r requirements-research.txt
python research/scripts/build_store.py        # parquet -> float32 memmap store
python research/scripts/make_folds.py         # ONLY if folds/folds.parquet is absent
python research/scripts/make_screen_store.py  # 2,500-series store for cheap iteration
PYTHONPATH=src python -m sbr.features.driver --modules m00_core,m01_seq,m02_dist,m03_dyn,m04_resid,m06_loc,m07_bayes
python research/scripts/agent0_full_runs.py   # the champion 5-fold run
```

**Paths are absolute** (`/home/claude/sb/...`) throughout the scripts, inherited
from the container the research ran in. Point `SBR_STORE` and the `ROOT`
constants at your own working directory before running anything.

## The rules that produced these numbers

- Splits are **series-level and permanent**. No row-wise random splits, ever: two
  adjacent prefixes of one series must never straddle a train/validation boundary.
- A **2,000-series lockbox** was held out of every selection decision and opened
  twice, at the very end, to measure research overfit.
- `X_test.reduced.parquet` was never read.
- Every feature module must pass **bitwise prefix invariance** (`atol=0.0`):
  rebuilding on a truncated online segment must reproduce the surviving rows
  exactly. A module that fails is looking into the future. This check caught two
  real leaks during development.
- Model selection is **held-out trajectory TS-AUC**, never row-level AUC and never
  synthetic-data performance.
