# Data

This note describes the data the project uses, what is and is not included in the
repository, and how to supply your own data.

## Challenge context

The project was built around the **ADIA Lab Structural Break Challenge**, hosted by
[CrunchDAO](https://www.crunchdao.com/). The task is to decide, for a time series
split at a boundary point, whether the data-generating process changes across that
boundary — i.e. whether a structural break occurs.

## What is included in this repository

Only **small synthetic samples** are committed, so a fresh clone runs end-to-end
with no downloads:

- `data/train.csv`, `data/test.csv` — tiny seeded samples, each a single ordered
  series with one injected level shift. See [`data/README.md`](../data/README.md).
- Programmatic generators in `structural_break.synthetic`
  (`make_mean_shift`, `make_variance_shift`, `make_multiple_breaks`), which return a
  DataFrame plus the known break indices. These power the tests and
  `scripts/compare_methods.py`.

These samples exist for demonstration and testing. They are **not** the official
challenge data and should not be interpreted as representative of it.

## Can the official data be redistributed?

**No.** The official competition data is subject to the challenge's terms and is
**not** redistributed here. Obtain it through the competition itself.

## Expected local layout

Place data under `data/`:

```text
data/
├── train.csv      # columns: timestamp, value, has_structural_break
└── test.csv       # columns: timestamp, value
```

Large or raw files should go under `data/raw/`, which is git-ignored, so they are
never committed by accident.

## Expected columns

| Column | Files | Description |
| ------ | ----- | ----------- |
| `timestamp` | train, test | ISO date (`YYYY-MM-DD`), one row per observation, ordered in time |
| `value` | train, test | Observed numeric series |
| `has_structural_break` | train | Binary label: `1` in the post-break regime, `0` otherwise |

## Schema assumptions

- The data is treated as a **single ordered series**. If your data contains
  several independent series, add a series identifier and group by it before
  feature engineering or detection, so windows and segments do not span series
  boundaries.
- `has_structural_break` is read as the per-row training label for the ML baseline.
  The change-point detectors are unsupervised and use only `timestamp` and `value`.
- If the official challenge schema differs from the columns above, map it into this
  format in a small adapter rather than changing the core package.

## Output data

Running the baseline or comparison scripts writes to `outputs/`, which is
git-ignored. Generated predictions, comparison tables, and figures are not tracked;
the only committed figure is the curated example at
`docs/images/example_break_detection.png`.
