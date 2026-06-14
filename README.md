# Structural Break Detection

Detect structural breaks in time-series data using a reproducible baseline workflow built for the ADIA Lab Structural Break Challenge hosted by CrunchDAO.

Structural breaks are abrupt changes in the data-generating process of a time series. In finance and economics, these changes can correspond to regime shifts, policy changes, market stress, volatility transitions, or changes in relationships between variables.

This repository currently contains a baseline machine-learning approach and notebook artifacts. The project is being cleaned up into a more professional research codebase with reproducible scripts, tests, and documented modeling workflows.

## Current status

This is an active research project. The current baseline uses engineered time-series features and a Random Forest classifier to predict whether each observation contains a structural break.

Planned upgrades include:

- Statistical change-point methods such as CUSUM and PELT
- Cleaner package structure under `src/structural_break/`
- Unit tests and GitHub Actions CI
- Better experiment tracking and model comparison tables
- Example charts showing detected break points

## Repository layout

```text
structural-break/
├── data/                  # Small synthetic sample data (see data/README.md); raw files stay untracked
│   ├── train.csv
│   ├── test.csv
│   └── README.md
├── outputs/               # Generated predictions; created at runtime and git-ignored
├── scripts/
│   └── baseline.py        # Current baseline training/prediction script
├── baseline.ipynb         # Competition quickstarter notebook (requires crunch-cli)
├── requirements.txt       # Python dependencies
├── .gitignore             # Local files, virtualenvs, and generated artifacts
├── LICENSE                # MIT license
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

From the repository root:

```bash
cd scripts
python baseline.py
```

The script reads from `../data/train.csv` and `../data/test.csv`, trains a Random Forest baseline, and writes predictions to `../outputs/submission.csv`.

A near-term refactor should replace these hardcoded relative paths with command-line arguments, for example:

```bash
python scripts/baseline.py \
  --train data/train.csv \
  --test data/test.csv \
  --output outputs/submission.csv
```

## Baseline methodology

The current baseline creates simple time-series features from the `value` column:

- Rolling mean and standard deviation
- Lag features
- First and second differences

It then fits a scikit-learn pipeline containing:

- `StandardScaler`
- `RandomForestClassifier`

This is a useful benchmark, but it should not be considered a final structural-break methodology. Future versions should compare the machine-learning baseline against classical change-point detection methods.

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
