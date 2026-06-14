# Data

This directory holds the time-series data used by the baseline workflow.

## What is committed here

`train.csv` and `test.csv` are **small synthetic samples**, not the official
challenge data. They exist so that a fresh clone can run the baseline end-to-end
without any external downloads. Each contains a single ordered series with one
deliberately injected level shift (a structural break), generated with a fixed
random seed.

Expected columns:

| File        | Columns                                   |
| ----------- | ----------------------------------------- |
| `train.csv` | `timestamp`, `value`, `has_structural_break` |
| `test.csv`  | `timestamp`, `value`                      |

- `timestamp` — ISO date (`YYYY-MM-DD`), one row per observation, ordered in time.
- `value` — the observed numeric series.
- `has_structural_break` — binary label (`1` once the series has entered the
  shifted regime, `0` otherwise). Present only in the training file.

## Official challenge data

This project was built around the
[ADIA Lab Structural Break Challenge](https://www.crunchdao.com/) hosted by
CrunchDAO. The official competition data is **not redistributed** in this
repository because of its licensing terms.

To run against the real data, obtain it through the competition and place it
here as `data/train.csv` / `data/test.csv` (replacing the synthetic samples), or
keep large raw files under `data/raw/`, which is git-ignored. Do not commit
licensed or competition-restricted datasets.
