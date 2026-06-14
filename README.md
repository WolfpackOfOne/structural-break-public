# Structural Break Detection

[![CI](https://github.com/WolfpackOfOne/structural-break/actions/workflows/ci.yml/badge.svg)](https://github.com/WolfpackOfOne/structural-break/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Detect structural breaks in time-series data using a reproducible baseline workflow built for the ADIA Lab Structural Break Challenge hosted by CrunchDAO.

Structural breaks are abrupt changes in the data-generating process of a time series. In finance and economics, these changes can correspond to regime shifts, policy changes, market stress, volatility transitions, or changes in relationships between variables.

This repository contains a machine-learning baseline plus a set of classical and modern change-point detectors, all behind a small, tested Python package.

## Current status

This is an active research project. Implemented today:

- A supervised baseline: engineered time-series features + a Random Forest classifier.
- Statistical / change-point detectors on a shared interface: **CUSUM**, **rolling z-score**, and **PELT** (via `ruptures`).
- Synthetic data generators with known break points, a method-comparison workflow, and a plotting helper.
- An importable package under `src/structural_break/`, a pytest suite, and GitHub Actions CI.

Planned upgrades (see [Roadmap](#professionalization-roadmap)):

- HMM regime detection and Bai-Perron-style multiple-break tests.
- Experiment tracking and richer visual diagnostics.
- A notebook walkthrough on the official challenge data.

## Repository layout

```text
structural-break/
├── src/
│   └── structural_break/   # Importable package
│       ├── data.py         # CSV loading + column validation
│       ├── features.py     # Baseline feature engineering
│       ├── models.py       # scikit-learn pipeline builder
│       ├── detectors.py    # CUSUM / rolling z-score / PELT detectors
│       ├── synthetic.py    # Synthetic series with known break points
│       ├── evaluation.py   # Per-row + point-based metrics
│       ├── visualization.py# Optional plotting helper
│       └── predict.py      # Submission/prediction helpers
├── scripts/
│   ├── baseline.py         # Thin argparse CLI around the package
│   └── compare_methods.py  # Detector comparison workflow (synthetic data)
├── data/                   # Small synthetic sample data (see data/README.md); raw files stay untracked
│   ├── train.csv
│   ├── test.csv
│   └── README.md
├── outputs/                # Generated predictions; created at runtime and git-ignored
├── baseline.ipynb          # Competition quickstarter notebook (requires crunch-cli)
├── pyproject.toml          # Package metadata / build configuration
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT license
└── README.md
```

> Note: The repository previously included local virtual environment files. New virtual environments should be created locally and left untracked.

## Setup

Clone the repository:

```bash
git clone https://github.com/WolfpackOfOne/structural-break.git
cd structural-break
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows PowerShell
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Optionally, install the package itself (editable) so you can `import structural_break`
from anywhere:

```bash
pip install -e .
```

## Data

The repository ships with **small synthetic sample data** under `data/` so the
baseline runs immediately after cloning — no downloads required. These samples
each contain one ordered series with a single injected structural break and are
**not** the official challenge data. See [`data/README.md`](data/README.md) for
the schema and details.

The baseline expects the following columns:

- `timestamp`
- `value`
- `has_structural_break` in the training data

This project was built around the ADIA Lab Structural Break Challenge hosted by
CrunchDAO. The official competition data is **not redistributed** here. To run
against it, place the real files at `data/train.csv` / `data/test.csv`
(replacing the samples) or keep large raw files under the git-ignored
`data/raw/`.

## Run the baseline

The baseline is a thin CLI around the `structural_break` package. Run it from the
repository root:

```bash
python scripts/baseline.py \
  --train data/train.csv \
  --test data/test.csv \
  --output outputs/submission.csv
```

All three paths default to the values shown above, so `python scripts/baseline.py`
works out of the box against the bundled synthetic samples. The command engineers
baseline features, trains a Random Forest classifier, prints training metrics, and
writes a submission CSV with `timestamp` and `has_structural_break` columns. Missing
files or missing columns are reported with a clear error message and a non-zero exit
code.

To use the package directly:

```python
from structural_break import (
    create_baseline_features,
    build_random_forest_baseline,
    build_submission,
)
```

## Development

Install the development dependencies (runtime requirements plus `pytest` and `ruff`)
and run the checks locally:

```bash
pip install -r requirements-dev.txt
pip install -e .

ruff check .   # lint
pytest         # tests (synthetic data only — no competition data required)
```

The same two commands run in [GitHub Actions](.github/workflows/ci.yml) on every
push to `main` and every pull request.

## Methods

The project pairs a supervised ML baseline with classical/modern change-point
detectors. Every detector shares one interface — `detector.detect(df)` returns a
DataFrame with `timestamp`, `break_score`, `has_structural_break`, and `method`:

```python
from structural_break import CusumDetector, PeltDetector, RollingZScoreDetector

result = PeltDetector().detect(df)   # df has 'timestamp' and 'value' columns
```

| Method | Detects | Strength | Limitation |
| ------ | ------- | -------- | ---------- |
| ML baseline (Random Forest) | Per-row break label from engineered features | Learns from labelled data | Needs labels; not a true change-point model |
| CUSUM | A single dominant mean shift | Simple, interpretable, parameter-light | Finds only the strongest break; mean-only |
| Rolling z-score | Local deviations / transitions / outliers | Transparent, no training | Flags transition points, not whole regimes |
| PELT (`ruptures`) | One or more mean shifts (penalised segmentation) | Handles multiple breaks well | Penalty tuning; mean-shift focused (`l2`) |

The **ML baseline** engineers `rolling_mean_3`, `rolling_std_3`, `lag_1`, `lag_2`,
`diff_1`, and `diff_2` from the `value` column and fits a `StandardScaler` +
`RandomForestClassifier` pipeline. Each detector documents its scoring and break
semantics in its class docstring in `src/structural_break/detectors.py`.

### Compare methods

`scripts/compare_methods.py` runs all detectors on a synthetic series with known
break points and scores each with windowed point-based precision/recall/F1:

```bash
python scripts/compare_methods.py \
  --dataset mean_shift \
  --output outputs/method_comparison.csv \
  --figure outputs/figures/example_break_detection.png
```

`--dataset` accepts `mean_shift`, `multiple`, or `variance`.

### Results (synthetic)

The numbers below are computed by the command above on the **synthetic**
`mean_shift` dataset (single mean shift at index 60, tolerance ±5). They are not
official competition results — they demonstrate detector behaviour on data with a
known ground truth.

| Method | Precision | Recall | F1 | Notes |
| ------ | --------: | -----: | -: | ----- |
| CUSUM | 1.00 | 1.00 | 1.00 | Single dominant mean shift |
| Rolling z-score | 1.00 | 1.00 | 1.00 | Flags the transition |
| PELT | 1.00 | 1.00 | 1.00 | Penalised segmentation |
| ML baseline | 1.00 | 1.00 | 1.00 | Trained on an independent series |

On harder cases the tradeoffs show: on the `multiple` dataset CUSUM recovers only
the dominant break (recall 0.5) while PELT recovers both; on the `variance`
dataset the mean-based detectors miss the volatility shift that the rolling
z-score still flags. Results against the official ADIA Lab challenge data are
**pending** and will be reported only when reproducible.

## Professionalization roadmap

### Phase 1 — Repository hygiene

- Remove committed virtual environment files
- Keep generated outputs out of version control
- Ensure a fresh clone can run the baseline
- Add a concise, accurate README

### Phase 2 — Package the project

Move reusable logic out of notebooks/scripts and into:

```text
src/structural_break/
├── features.py
├── models.py
├── evaluation.py
└── io.py
```

### Phase 3 — Testing and CI

Add:

- `pytest`
- Unit tests for feature engineering
- Unit tests for prediction output shape
- GitHub Actions for linting and tests

### Phase 4 — Research methods

Add and compare:

- CUSUM
- PELT
- Bai-Perron-style multiple break tests
- Bayesian change-point detection
- Hidden Markov Models for regime detection

## Why this project matters

Structural-break detection is directly relevant to quantitative research, risk management, macroeconomic analysis, and financial machine learning. A polished version of this repository can serve as a public research example showing how to move from a competition baseline to a maintainable research workflow.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
