# Weekend model-search harness

A resumable driver loop for trying many models against the ADIA Lab /
CrunchDAO Structural Break Challenge and tracking every result, built to run
unattended for hours without losing progress.

## Quick start

Smoke-test against a synthetic dataset shaped like the real competition data
(no downloads needed):

```bash
python scripts/run_experiments.py --synthetic
```

Run against the real data once it's placed under `data/` (the official
`X_train` is typically split into several Parquet parts due to GitHub's
100 MB limit):

```bash
python scripts/run_experiments.py \
  --x-train data/X_train.part1.parquet data/X_train.part2.parquet data/X_train.part3.parquet \
  --y-train data/y_train.parquet
```

Re-running the same command is safe and expected — experiments already
recorded with `status=ok` in `leaderboard.csv` are skipped, so you can stop
and restart the loop (or let it survive a crash/reboot) without redoing work.

## What it does

1. Loads the competition data: a `(series id, time)`-indexed frame with
   `value` and `period` (`0` before the series' boundary point, `1` after)
   columns, and a per-series binary label (`has_structural_break`).
2. Splits series ids into a train/validation set (`--val-fraction`, default
   20%) and precomputes shared feature matrices **once** — statistical
   before/after features (`src/structural_break/series_features.py`) and the
   existing CUSUM/rolling-z-score/PELT detectors' scores, applied per series.
3. Runs every experiment registered in
   `src/structural_break/experiment_registry.py`, in order, skipping ones
   already marked `ok` in the leaderboard.
4. Records every attempt to `experiments/leaderboard.csv`: id, family, status
   (`ok` / `skipped` / `timeout` / `error`), validation AUC, runtime, config,
   and notes. A bad config never crashes the loop — see
   `structural_break.experiment_runner.run_experiment`.
5. Saves the best-so-far model (`experiments/models/best_model.joblib`, git-
   ignored) and its config (`experiments/best_config.json`, tracked)
   whenever a new experiment beats the recorded best AUC.
6. Commits `experiments/` (and pushes, unless `--no-push`) on an interval —
   by experiment count (`--checkpoint-every`, default every experiment) and/or
   elapsed time (`--checkpoint-minutes`, default 30) — so a killed process
   loses at most one checkpoint's worth of work.
7. Appends a short summary to `experiments/progress.md` on every new best and
   at the end of the run — read that first when checking in.

## Extending the search (this is the main weekend task)

Add new experiments in `src/structural_break/experiment_registry.py` with the
`@register(name, family, description, configs=[...])` decorator — see the
working examples (`rf_stat_features`, `xgboost_combined_features`, ...) for
the pattern. Every experiment function receives an `ExperimentContext` with
precomputed `train_stat_features` / `val_stat_features` (before/after
statistics per series), `train_detector_features` / `val_detector_features`
(CUSUM/rolling-z-score/PELT scores per series), and the raw
`train_X`/`train_y`/`val_X`/`val_y` if you need row-level access instead.

Four families are stubbed out (`hmm_regime`, `bayesian_changepoint`,
`deep_learning_sequence_model`, `stacking_ensemble`) — they currently raise
`NotImplementedError` with a TODO describing the intended approach, and the
runner records them as `skipped` rather than failing. Implementing these is
the highest-value work for a long unattended run: they're the parts of the
model space nothing here covers yet. A missing optional dependency
(`xgboost`, `lightgbm`, `torch`, `hmmlearn`, ...) should raise `ImportError`
with an install hint, same as the two gradient-boosting experiments already
do — the runner treats that the same as a stub (`skipped`, not `error`), so
`pip install`-ing the dependency and re-running picks it up automatically.

## Useful flags

| Flag | Purpose |
| --- | --- |
| `--max-experiments N` / `--max-minutes N` | Cap a single invocation, e.g. for a quick check between longer runs. |
| `--timeout-per-experiment N` | Per-experiment wall-clock budget in seconds (`0` disables; default 900). Best-effort (`SIGALRM`), Unix only. |
| `--no-push` | Commit checkpoints locally without pushing (e.g. no configured remote credentials). |
| `--seed`, `--val-fraction` | Control the train/validation split — keep these fixed across a sweep so experiments stay comparable. |

## A note on evaluation discipline

The train/validation split here is a fast, single split for ranking
experiments against each other quickly — not rigorous cross-validation. It's
good enough to tell a real improvement from noise on a large sweep, but a
score that looks great on one split can still be an artifact of that split.
Sanity-check anything you're about to rely on (e.g. re-score the top few
candidates with a different `--seed`) before trusting it as the final answer.
