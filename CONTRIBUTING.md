# Contributing

Thanks for your interest in improving this project. These notes cover local setup,
the checks we run, and a few conventions. They are written to be useful to both
human contributors and AI coding agents (Claude, Codex, etc.) working on the repo.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt  # runtime deps + pytest + ruff
pip install -e .                     # install the package (editable)
```

## Running the checks

```bash
ruff check .   # lint
pytest         # tests (synthetic data only)
```

Both commands run in CI on every pull request and on pushes to `main`
(`.github/workflows/ci.yml`). Please make sure they pass locally before opening a
PR.

## Running the code

```bash
# Baseline ML workflow
python scripts/baseline.py --train data/train.csv --test data/test.csv \
  --output outputs/submission.csv

# Detector comparison on synthetic data with known breaks
python scripts/compare_methods.py --dataset mean_shift \
  --output outputs/method_comparison.csv
```

## Conventions

- **Type hints and docstrings** on public functions and classes.
- **Tests for new behaviour.** New detectors go behind the `BreakDetector`
  interface in `src/structural_break/detectors.py` and need tests that use only
  synthetic data (see `structural_break.synthetic`) — never the competition data.
- **Keep changes focused.** Prefer small, reviewable PRs with a clear description.
- Lint and formatting are enforced by Ruff; see `[tool.ruff]` in `pyproject.toml`.

## What not to commit

Please do **not** commit any of the following (they are git-ignored, but double
check):

- Virtual environments (`.venv/`, `venv/`, `env/`).
- Generated outputs: `outputs/` (predictions, comparison tables, figures).
- Model artifacts (`*.pkl`, `*.joblib`) and large/raw data (`data/raw/`).
- Licensed or competition-restricted datasets of any kind.
- Notebook outputs — clear them before committing.

The only committed figure is the curated example at
`docs/images/example_break_detection.png`. Regenerate it with
`scripts/compare_methods.py --figure ...` if the detectors change.

## Reporting issues

Open a GitHub issue describing the problem, what you expected, and a minimal way to
reproduce it (a small synthetic series is ideal).
