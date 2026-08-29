# Experiment progress log

Running notes from `scripts/run_experiments.py`, appended automatically on
each new best model and at the end of every run. This is the first thing to
read when checking in on an unattended run.

See `experiments/leaderboard.csv` for the full ranked history of every
experiment tried, and `experiments/best_config.json` for the current best
model's identity (the model artifact itself lives at
`experiments/models/best_model.joblib`, git-ignored since it's reproducible
from the config).

## Log
- New best: `logistic_stat_features` (family=linear_baseline) AUC=0.995238 at 2026-08-29T14:30:36+00:00.
- New best: `rf_stat_features__max_depth=None,n_estimators=200` (family=ml_baseline) AUC=1.0 at 2026-08-29T14:30:36+00:00.

## Run finished at 2026-08-29 14:30:39 UTC
- Ran 12 experiment(s) in 0.1 minutes.
- Best this session: rf_stat_features__max_depth=None,n_estimators=200 AUC=1.0.
- See `/home/user/structural-break/experiments/leaderboard.csv` for the full ranked history.
