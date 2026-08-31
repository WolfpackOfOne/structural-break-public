# Structural Break Detection — 2026 Real-Time Edition

[![CI](https://github.com/WolfpackOfOne/structural-break/actions/workflows/ci.yml/badge.svg)](https://github.com/WolfpackOfOne/structural-break/actions/workflows/ci.yml)

Research and deployment repository for the **2026 ADIA Lab / CrunchDAO Structural Break Challenge — Real-Time Edition**.

The system must emit a continuous structural-break score at each online time step using only information available at that prefix. Inference is causal, streaming, and independent across series.

## Current result

| System | Role | Official TS-AUC |
| --- | --- | ---: |
| **RT-1257** | **Current external champion** | **0.6290** |
| RT-600 | Formal production anchor / pure-LightGBM reference | 0.6268 |
| Difference | RT-1257 vs RT-600 | **+0.0022** |

RT-1257 is the first recorded external improvement beyond the RT-600 anchor in this research program. It was selected using internal evidence before the external score was known; the leaderboard result is treated as validation, not as a tuning oracle.

**Production-status nuance:** RT-1257 is the best externally measured deployable system, but RT-600 remains the formal production anchor during consolidation. The build/manifest source gap has been verified as packaging/dependency/report-only at the tracked-code level, a fresh local post-build Crunch test passed on 2026-08-31 against entrypoint SHA `05eafb66f4589f5b426f4af43779f428f7a90f2d64b423530d6f315a798e888b`, and PR #14 CI passed. Formal RT-1257 production promotion is deferred to a separate owner tag/status action.

## RT-1257 architecture

RT-1257 preserves the seven-slot RT-600 specialist architecture and changes exactly two learner implementations:

```text
500 causal streaming features
        |
        +-- RT-1255 / CAT-300     (replaces RT-300 LightGBM)
        +-- RT-410 LightGBM
        +-- RT-411 LightGBM
        +-- RT-412 LightGBM
        +-- RT-1254 / CAT-413     (replaces RT-413 LightGBM)
        +-- RT-414 LightGBM
        +-- RT-415 LightGBM
        |
fold-pure smooth time-conditional CDF calibration
        |
      equal blend
        |
      RT-1257
        |
official Crunch TS-AUC 0.6290
```

CAT-300 and CAT-413 are replacement slots inside the seven-member system; they are not extra eighth/ninth members.

The mature causal feature bank contains 500 columns across the core historical-null, sequential-memory, distributional, dynamic/dependence, residual, location-related, and Bayesian modules under `src/sbr/features/`.

## Research survivors

The corrected current-state model set is maintained in [`research/MODEL_REGISTRY.md`](research/MODEL_REGISTRY.md).

The main research-alive candidates beyond RT-1257 are:

- **RT-1261 / CAT-412** — strongest residual single-slot candidate. A three-slot CAT-413 + CAT-300 + CAT-412 composition is nominally only +0.000338 over RT-1257, below the 0.0011 paired-bootstrap noise floor.
- **RT-1263 / CAT-415** — positive individual evidence, no demonstrated corrected multi-slot improvement over RT-1257.
- **RT-1262 / CAT-414** — same status.
- **RT-1260 / CAT-411** — survives the original individual gate, with weaker corrected evidence.
- **RT-995 / T2** — parked rather than killed: strong standalone teacher-distillation signal, but mostly redundant when integrated with the ensemble.

### Important correction: RT-1264 is not live

RT-1264 was the original five-slot CatBoost best-k hybrid. The later CSA-04R reanalysis found that its selection endpoint was inflated by deterioration in the matched clone control as k grew. The corrected fixed E2-E0 analysis is registered as **RT-1265** and selects k=2: CAT-413 + CAT-300, exactly RT-1257.

RT-1264 remains immutable historical evidence but is classified **SUPERSEDED**, not as a current upside candidate.

## What did not work

This repository deliberately preserves negative results. Start with:

- [`research/NEGATIVE_RESULTS_INDEX.md`](research/NEGATIVE_RESULTS_INDEX.md) — fast searchable summary.
- [`research/FAILED_EXPERIMENTS.md`](research/FAILED_EXPERIMENTS.md) — detailed hypothesis, result, why-it-failed, and retry guidance.
- [`research/LESSONS_LEARNED.md`](research/LESSONS_LEARNED.md) — cross-program synthesis.

Major completed negative directions include:

- TabM and RealMLP full five-fold GPU arms: **KILL** at the binding ensemble-marginal endpoint.
- CAT-410 replacement: **KILL**.
- Causal Representation Frontier: no surviving model.
- Leaderboard Alpha: no surviving mechanism.
- Wave 8 future-aware transfer: all five mechanisms killed.
- New Avenues executed pilot sweeps: no confirmation candidate.
- Many intuitive sequential/trend/threshold summaries: redundant or harmful conditional on the mature bank.

### TabM / RealMLP experiment IDs

The repository contains two ID generations that must not be misread as two independent KILL tests:

- **RT-1250 TabM / RT-1252 RealMLP** — original Learner Diversity arms, recorded **INFEASIBLE** under their frozen original compute contract; they did not produce binding predictive scores.
- **RT-1258 TabM / RT-1259 RealMLP** — distinct fresh GPU-authorized arms after RTX 4090 benchmarking removed the compute blocker without shrinking the intended full-scale configurations. These completed five folds and are the binding **KILL** results.

## Research navigation

Start here:

- [`research/STATUS.md`](research/STATUS.md) — concise current state.
- [`research/MODEL_REGISTRY.md`](research/MODEL_REGISTRY.md) — models that matter now.
- [`research/INDEX.md`](research/INDEX.md) — research-program table of contents.
- [`research/RESULTS.csv`](research/RESULTS.csv) — quantitative experiment ledger.
- [`research/EXPERIMENT_ID_MAP.md`](research/EXPERIMENT_ID_MAP.md) — ID/provenance map.
- [`research/RDOF_LEDGER.md`](research/RDOF_LEDGER.md) — degrees-of-freedom accounting.
- [`research/NEGATIVE_RESULTS_INDEX.md`](research/NEGATIVE_RESULTS_INDEX.md) — failed-idea lookup.
- [`research/LESSONS_LEARNED.md`](research/LESSONS_LEARNED.md) — synthesis.

The exact pre-consolidation branch and model state is preserved under [`research/archive/2026-08-30/`](research/archive/2026-08-30/).

## Research standard going forward

The primary question is no longer simply “does this model have good standalone TS-AUC?”

It is:

> **Does this candidate add complementary information to RT-1257?**

For serious candidates, prefer a matched ensemble test:

```text
C0 = candidate standalone
C1 = matched control standalone

E0 = RT-1257
E1 = RT-1257 with an exchangeable matched control
E2 = RT-1257 with the candidate

primary endpoint: E2 - E1
secondary:        E2 - E0
```

Also examine fold consistency, pair-flow repair/damage, dominant-cell behavior, mature-vs-never behavior, within-t correlation, causality, runtime, memory, and deployment complexity.

RT-600 remains a permanent homogeneous-LightGBM reference because it isolates the value of learner-family diversity.

## Repository layout

```text
structural-break/
├── src/
│   ├── sbr/                         # 2026 causal research/production pipeline
│   └── structural_break/            # original compact baseline package
├── research/
│   ├── STATUS.md
│   ├── MODEL_REGISTRY.md
│   ├── INDEX.md
│   ├── RESULTS.csv
│   ├── EXPERIMENT_ID_MAP.md
│   ├── NEGATIVE_RESULTS_INDEX.md
│   ├── FAILED_EXPERIMENTS.md
│   ├── LESSONS_LEARNED.md
│   ├── RDOF_LEDGER.md
│   ├── reports/
│   ├── scripts/
│   └── archive/
├── engineering/
│   └── reports/                     # production/deployment evidence
├── submissions/                     # reproducible submission builders/artifact metadata
├── tests/
├── docs/
└── .github/workflows/
```

The original Random Forest / CUSUM / rolling-z / PELT package remains in the repository as an accessible baseline and general change-point example. It is no longer the headline competition architecture.

## Reproducibility and artifact policy

Raw competition data, large feature caches, OOF arrays, trained-model directories, and rebuildable generated payloads are not intended to be committed merely for convenience. Reproducibility is based on versioned code/configuration, permanent fold assignments, manifests/hashes, final reports, and explicit experiment provenance.

Historical experiment rows are not rewritten when a later analysis changes their interpretation. Corrections are recorded in the ID map, model registry, reports, and later ledger entries while the original measurement remains intact.

## Consolidation status

The research-to-production consolidation is being validated on `release/2026-research-consolidation`. `main` must not move until the consolidation report's merge gates are satisfied. See [`docs/MAIN_CONSOLIDATION_REPORT_2026.md`](docs/MAIN_CONSOLIDATION_REPORT_2026.md).

## Development

Create an environment and install the core/development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

Optional GPU-tabular dependencies for reproducing the completed RT-1258/RT-1259 research are isolated in:

```bash
pip install -r research/requirements-gpu-tabular.txt
```

Run repository checks with the commands defined in `.github/workflows/ci.yml`. Research that requires the official competition store/caches must not silently substitute synthetic data for the binding result; such tests should explicitly skip or report missing external data.
