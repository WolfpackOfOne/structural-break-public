# Structural Break Detection — Real-Time Edition

[![CI](https://github.com/WolfpackOfOne/structural-break-public/actions/workflows/ci.yml/badge.svg)](https://github.com/WolfpackOfOne/structural-break-public/actions/workflows/ci.yml)

Research and deployment work for the [**ADIA Lab Structural Break Challenge: Real-Time Edition**](https://hub.crunchdao.com/competitions/structural-break-real-time), hosted by CrunchDAO.

The system emits a continuous structural-break score at each online time step using only information available at that prefix. Inference must be causal, streaming, and independent across series; future-data leakage is a correctness failure, not a modelling choice.

## Results and repository status

| System | Role | Public-set TS-AUC | Status date |
| --- | --- | ---: | --- |
| **RT-1320** | Best public-set score recorded in this repository | **0.6303** | 2026-09-03 |
| **RT-1257** | Formal production anchor | **0.6290** | 2026-09-02 |
| RT-600 | Homogeneous-LightGBM reference | 0.6268 | — |

RT-1320 added an eighth residual-student member to RT-1257 and improved this program's recorded public-set score by **+0.0013**. Submission #19 was selected on 2026-09-04. The 0.6303 value is a public-set score, not the competition's final private-set result.

The default `main` branch is an earlier consolidated snapshot. It records RT-1320 as a research candidate, but predates RT-1257's formal production promotion on 2026-09-02 and RT-1320's public score and submission selection on 2026-09-03–04. Those later status changes are documented on public branches:

- [RT-1320 external-champion status](https://github.com/WolfpackOfOne/structural-break-public/blob/engineering/rt1320-promotion-prep/research/STATUS.md)
- [RT-1257 production-promotion status](https://github.com/WolfpackOfOne/structural-break-public/blob/production/rt1257/research/STATUS.md)

## Architecture

The `main` snapshot contains the RT-1257 seven-member specialist ensemble:

```text
500 causal streaming features
        |
        +-- five LightGBM specialists
        +-- two CatBoost specialists
        |
fold-pure, time-conditional CDF calibration
        |
      equal blend
        |
      RT-1257
```

RT-1320 retains that system and adds a residual-student model as an eighth member. The feature bank covers historical-null, sequential-memory, distributional, dynamic/dependence, residual, location-related, and Bayesian modules under `src/sbr/features/`.

## My role

I defined the system objectives, causal constraints, experiment protocol, review gates, and promotion decisions. I used coding agents for implementation and accepted work only after it passed the relevant tests and evidence gates.

## Engineering evidence

- **Causal streaming controls:** batch/stream parity checks and dedicated tests against `tau` and `n_online` leakage (`tests/test_stream_*`, `tests/test_no_tau_leakage.py`, and `tests/test_no_n_online_leakage.py`).
- **Testing and CI:** [GitHub Actions](.github/workflows/ci.yml) runs ruff, research-ledger hygiene checks, and pytest; a separate [research-hygiene workflow](.github/workflows/research-hygiene.yml) protects experiment IDs.
- **Experiment governance:** matched-control ensemble tests, an append-only [results ledger](research/RESULTS.csv), a [degrees-of-freedom ledger](research/RDOF_LEDGER.md), and explicit model classifications.
- **Reproducibility metadata:** versioned configuration; fold seeds, summaries, and hashes; dependency records; build manifests; and artifact hashes.
- **Deployment provenance:** tracked build tooling in [`research/scripts/build_submission.py`](research/scripts/build_submission.py), RT-1257 build metadata in [`submissions/RT1257_deployable.build.json`](submissions/RT1257_deployable.build.json), and qualification evidence under [`engineering/reports/rt1257_deployment/`](engineering/reports/rt1257_deployment/).
- **Negative-result preservation:** failed, parked, and superseded approaches remain searchable through the [negative-results index](research/NEGATIVE_RESULTS_INDEX.md) and [lessons learned](research/LESSONS_LEARNED.md).

## Reproducibility boundaries

Raw competition data, large feature caches, out-of-fold arrays, trained-model directories, and the generated RT-1257/RT-1320 deployment entrypoints and payloads are not distributed in this public repository. The repository does include the earlier RT-600 deployment notebook and Python entrypoint under `submissions/`, together with their build metadata, for audit and provenance.

The exact permanent fold-assignment artifact is also not distributed. The repository records its seeds, summaries, and hashes; exact reproduction requires the external artifact matching the recorded SHA-256. The tracked code and metadata support audit and reconstruction, but the public clone alone cannot reproduce the exact submitted deployment artifact.

## Quickstart

### Compact baseline and focused tests

The following runs the original Random Forest, CUSUM, rolling-z, and PELT examples plus their focused 52-test subset:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
pytest tests/test_detectors.py tests/test_cli.py tests/test_features.py \
  tests/test_models.py tests/test_change_point_eval.py \
  tests/test_prediction_output.py tests/test_visualization.py
python scripts/compare_methods.py --dataset mean_shift --output outputs/method_comparison.csv
```

The comparison and baseline-training examples use small synthetic samples committed under `data/`:

```bash
python scripts/baseline.py \
  --train data/train.csv \
  --test data/test.csv \
  --output outputs/submission.csv
```

Official competition data must be obtained separately and adapted to the schema described in [`docs/data.md`](docs/data.md).

### Research checks used by CI

On macOS, install the OpenMP runtime before importing LightGBM:

```bash
brew install libomp
```

The following reproduces the GitHub Actions checks:

```bash
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -r requirements-research.txt
pip install -e .
python -c "import lightgbm, numba, pyarrow, catboost; import sbr"
ruff check src tests scripts
python research/scripts/check_research_hygiene.py
pytest -q
```

The frozen RT-1257 deployment environment is recorded separately in [`requirements-rt1257.txt`](requirements-rt1257.txt) and was built with Python 3.11.6. CI does not install that pinned file. Research requiring the official competition store or caches must explicitly skip or report missing external data rather than substitute synthetic data for a binding result.

## Research lessons

- Standalone model quality was not enough; candidates had to add complementary information to the existing ensemble against a matched control.
- Several plausible deep-learning, transfer, and feature directions failed their binding ensemble-marginal tests and were preserved rather than hidden.
- Internal estimates closely anticipated the external improvements for RT-1257 and RT-1320, supporting the experiment protocol without treating the public leaderboard as a tuning oracle.

## Research navigation

The `main`-branch research record is preserved as an auditable historical snapshot:

- [`research/STATUS.md`](research/STATUS.md) — snapshot status
- [`research/MODEL_REGISTRY.md`](research/MODEL_REGISTRY.md) — model classifications
- [`research/INDEX.md`](research/INDEX.md) — research table of contents
- [`research/RESULTS.csv`](research/RESULTS.csv) — experiment ledger
- [`research/EXPERIMENT_ID_MAP.md`](research/EXPERIMENT_ID_MAP.md) — experiment provenance
- [`research/FAILED_EXPERIMENTS.md`](research/FAILED_EXPERIMENTS.md) — detailed negative evidence
- [`docs/MAIN_CONSOLIDATION_REPORT_2026.md`](docs/MAIN_CONSOLIDATION_REPORT_2026.md) — consolidation provenance

## Repository layout

```text
structural-break-public/
├── src/sbr/                  # causal research and streaming pipeline
├── src/structural_break/     # compact baseline package
├── research/                 # protocols, ledgers, reports, and scripts
├── engineering/reports/      # deployment and reliability evidence
├── submissions/              # selected historical deployment artifacts and build metadata
├── tests/
├── docs/
└── .github/workflows/
```
