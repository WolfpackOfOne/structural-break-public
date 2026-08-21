# Public 2025 Reported Metrics

## Search Scope

Inspected the public repository at pinned commit `6316693333edc5831c2408ca5b155ffa24c302bd`: `README.md` and committed `submission.ipynb` outputs.

## Extracted Metrics

| quantity | value | source | protocol understood? |
|---|---:|---|---|
| reported local validation AUC | not reported | README/notebook search for AUC/CV/fold/validation/score | no |
| reported OOF ROC AUC | not reported | notebook outputs | no |
| reported Crunch/local test score | no score reported | `crunch.test(...)` output only reports local test execution | no |
| reported leaderboard score | not reported | README/notebook | no |
| reported rank/strength | second-place solution | GitHub repository description and README context | partly |
| local Crunch test duration | 00:06:13 | notebook output, final cell | yes: runtime only |
| local Crunch test memory consumed | 9.39 GB | notebook output, final cell | yes: memory only |

## Dataset Evidence In Notebook Output

The notebook output shows Crunch data release 146 downloads/already-exists checks with these byte lengths:

| file | notebook byte length |
|---|---:|
| `X_train.parquet` | 204327238 |
| `X_test.reduced.parquet` | 2380918 |
| `y_train.parquet` | 61003 |
| `y_test.reduced.parquet` | 2655 |

No historical `~0.90` AUC is documented in this public repository.  It should not be used as a sourced comparison value from this repo.
