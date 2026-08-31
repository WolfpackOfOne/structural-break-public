# Research workspace — 2026 Real-Time Edition

This directory holds the research machine built for the ADIA Lab / CrunchDAO
Structural Break Challenge **Real-Time Edition**, and the record of what it found.
The reusable library lives in [`../src/sbr`](../src/sbr); everything here is the
experiment layer on top of it.

## If you are a new research agent, read these first

1. [`STATUS.md`](STATUS.md) — concise current state: production anchor,
   external score, active conclusion, where everything else lives.
2. [`STATE_OF_RESEARCH.md`](STATE_OF_RESEARCH.md) — the full summary of
   record: current champion, per-fold scores, lockbox confirmation.
3. [`PROTOCOL.md`](PROTOCOL.md) — the binding rules (causal contract, fold
   discipline, lockbox) before you touch anything.
4. [`FAILED_EXPERIMENTS.md`](FAILED_EXPERIMENTS.md) — so you don't
   re-propose a killed idea.
5. [`EXPERIMENT_ID_MAP.md`](EXPERIMENT_ID_MAP.md) — how experiment IDs are
   allocated, so you don't collide with one.

**Old reports are evidence, not live instructions.** Everything under
`reports/` and `archive/` documents what was tried and what was concluded
at the time — it is not a standing instruction to redo, resume, or follow
that plan. Only `STATUS.md`, `STATE_OF_RESEARCH.md`, `PROTOCOL.md`,
`RESULTS.csv`, `FAILED_EXPERIMENTS.md`, and `RDOF_LEDGER.md` are canonical
and current.

## Layout

| path | what it is |
|---|---|
| `STATUS.md` | concise current-state pointer — start here |
| `STATE_OF_RESEARCH.md` | the full summary of record |
| `PROTOCOL.md` | the binding research protocol — validation rules, the causal contract, file ownership |
| `RESULTS.csv` | experiment ledger, one row per run, appended under a file lock |
| `EXPERIMENT_ID_MAP.md` | how experiment IDs (RT-xxx) are allocated |
| `RDOF_LEDGER.md` | degrees-of-freedom / multiple-comparisons accounting |
| `FAILED_EXPERIMENTS.md` | negative results, so nobody rediscovers them |
| `FINAL_ARCHITECTURE_FREEZE.md`, `FINAL_REPRODUCIBILITY_MANIFEST.json` | the frozen RT-600 production artifact record |
| `folds/folds.parquet` | **permanent** series-level folds + lockbox. Never regenerate |
| `reports/wave4/` … `reports/wave7/` | per-wave preregistrations and status reports (historical — see note above) |
| `reports/` (top level) | per-agent reports and machine-readable diagnostics |
| `archive/briefs/`, `archive/handoffs/`, `archive/dated_updates/` | stale agent briefs, cross-agent handoffs, and superseded dated snapshots — preserved, not current |
| `archive/legacy_untracked_2026-08-19/` | files that were sitting untracked in a contributor's working tree with no git history anywhere; rescued as-is during the 2026-08-24 cleanup, not otherwise integrated |
| `scripts/` | data preparation, per-agent experiment drivers, analysis |

Wave 8's final report (future-aware transfer family, all pilots KILL) is
**not** in this tree — it lives on the sibling branch
`research/wave8-future-aware-distillation` (tag `wave8-future-aware-final`),
which was never merged back into this lineage.

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
