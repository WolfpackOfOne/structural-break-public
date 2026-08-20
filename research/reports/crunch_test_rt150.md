# RT-150 official Crunch test report

Date: 2026-08-20 UTC

## Verdict

`submissions/C_ensemble_deployable.ipynb` was rebuilt for RT-150 and passed:

- local isolated notebook harness: `PASS`
- official `crunch test`, with determinism check enabled: `passed`
- final `INFER_PARALLELISM`: `1`

The final artifact uses one Crunch inference worker because the official macOS
runner repeatedly segfaulted LightGBM with `INFER_PARALLELISM = 4`. The measured
single-worker local speed still projects below the 15-hour weekly budget.

## Tested artifacts

- Notebook: `submissions/C_ensemble_deployable.ipynb`
  - size: `28,024,236` bytes
  - sha256: `9faa0d851b8645c75f15bf3bf3390d9258aacb5c4c9abf74bec9f040899c917b`
- CLI entrypoint for this local Crunch CLI version: `submissions/C_ensemble_deployable.py`
  - size: `28,021,454` bytes
  - sha256: `2b6c89e9e0e439315d9b90f34995933230d76983a315f71175a4e5c648f2f44a`

The `.py` file exists because `crunch test --main-file ...ipynb` fails in
Crunch CLI `11.11.0` with `AttributeError: 'NoneType' object has no attribute
'loader'`. It was generated mechanically by concatenating the notebook code
cells. `crunch convert` was not used because it commented out the embedded
payload variables and unpack calls.

## Environment

- branch worktree: `codex/wave3-engineering`
- base commit: `8e76ad205c68a67b219ff6b781dd9d696533d984`
- Python: `3.11.6`
- platform: `macOS-26.5.2-arm64-arm-64bit`
- machine: `arm64`
- Crunch CLI: `11.11.0`
- numpy: `2.4.6`
- pandas: `3.0.5`
- pyarrow: `25.0.1`
- scipy: `1.17.1`
- scikit-learn: `1.9.0`
- lightgbm: `4.7.0`
- numba: `0.67.0`

`pip install -r requirements-research.txt` initially failed under sandboxed DNS
for `pypi.org`; rerunning with approved network access installed the missing
runtime pieces, including `lightgbm==4.7.0`, `numba==0.67.0`, and
`llvmlite==0.49.0`.

## Model manifest

Model source archive copied from:

`/path/to/workspace/structural-break/wave2/models_rt150_ensemble.tar.gz`

Extracted to:

`models/rt150_ensemble`

Manifest summary:

- modules: `m00_core,m01_seq,m02_dist,m03_dyn,m04_resid,m06_loc,m07_bayes`
- features: `500`
- boosters: `7`
- calibration kind: `scdf`
- feature manifest sha256: `1646c3b9e09d8a7fb3c564483a1d1d999680caeefe848b092f907a6cac80cced`
- model manifest sha256: `5b2acbe34cc074b3d482b8c8230b848f97e5d61597c16d5ddc0af884564d52b4`
- manifest `code_git_sha`: `b5ea9d1d9cd87f06f574c5a78e4c850f41ef852f`
- `lockbox_touched`: `False`
- model built at: `2026-08-19T12:52:43Z`

Fold hash check:

- `research/folds/folds.parquet` sha256:
  `ba4f71fee8fcb30cc808234584a924ff0b0f0529e5bbd1fcbd0b4b780ebcc312`
- canonical `id,fold` CSV content sha256:
  `6e114f80240c0ff069266c6d65b660e500723ce9bf2ca7633596d9e392d6b9e9`

The brief's expected hash corresponds to the canonical `id,fold` content hash,
not the parquet container hash.

## Build commands

Store build used the real training parquet files from the original checkout,
because the isolated worktree contains placeholder data files:

```bash
env SBR_ROOT='/path/to/workspace/structural-break-codex-wave3' \
    SBR_RAW_DIR='/path/to/workspace/structural-break/data' \
    '/path/to/workspace/structural-break/.venv/bin/python' \
    research/scripts/build_store.py
```

Store build output:

- series: `10,000`
- total rows: `35,036,464`
- online points: `5,036,517`
- break rate: `0.4967`

Notebook build:

```bash
env SBR_ROOT='/path/to/workspace/structural-break-codex-wave3' \
    '/path/to/workspace/structural-break/.venv/bin/python' \
    research/scripts/build_submission.py \
    --model '/path/to/workspace/structural-break-codex-wave3/models/rt150_ensemble' \
    --out '/path/to/workspace/structural-break-codex-wave3/submissions/C_ensemble_deployable.ipynb' \
    --title 'RT-150 deployable ensemble (7 boosters, smooth time CDF)'
```

Build output:

- source zip sha256: `e51c70ab3609b9ce17983cf59c2d9c60ef0ea78a804692cf2202533d01aaabff`
- model zip sha256: `4f3d58a676dc098e15fb001a7d0c3970dfceb381271643878b714549bf636ae2`
- feature manifest: `1646c3b9e09d8a7fb3c564483a1d1d999680caeefe848b092f907a6cac80cced`

CLI `.py` generation:

```bash
'/path/to/workspace/structural-break/.venv/bin/python' -c 'import json, pathlib; nb_path=pathlib.Path("submissions/C_ensemble_deployable.ipynb"); out_path=pathlib.Path("submissions/C_ensemble_deployable.py"); nb=json.loads(nb_path.read_text()); cells=["".join(c["source"]) for c in nb["cells"] if c.get("cell_type")=="code"]; out_path.write_text("\n\n".join(cells)+"\n")'
```

## Local isolated harness

Command:

```bash
env SBR_ROOT='/path/to/workspace/structural-break-codex-wave3' \
    SBR_STORE='/path/to/workspace/structural-break-codex-wave3/cache/store' \
    '/path/to/workspace/structural-break/.venv/bin/python' \
    research/scripts/test_submission_notebook.py \
    --nb '/path/to/workspace/structural-break-codex-wave3/submissions/C_ensemble_deployable.ipynb'
```

Result:

```json
{
  "runtime_s": 29.047447681427002,
  "n_series": 20,
  "n_points": 10476,
  "ms_per_point": 2.7727613288876483,
  "deterministic": true,
  "order_independent": true,
  "future_poison_safe": true,
  "all_finite": true,
  "in_range": true,
  "ts_auc_diagnostic_in_sample_do_not_quote": 0.970255244897131,
  "PASS": true
}
```

The in-sample TS-AUC is only a diagnostic smoke value from the local fixture and
must not be quoted as validation performance.

Budget projection from local one-worker speed:

- 5.0M public points: `3.85` hours
- 10.1M private points: `7.78` hours
- 15-hour capacity: `19.48M` points

## Official `crunch test`

Final command:

```bash
'/path/to/workspace/structural-break/.venv/bin/crunch' test \
  --main-file '/path/to/workspace/structural-break-codex-wave3/submissions/C_ensemble_deployable.py' \
  --model-directory '/path/to/workspace/structural-break-codex-wave3/resources'
```

Final result:

```text
00:44:09 started
00:44:09 running local test
00:44:09 internet access isn't restricted, no check will be done
00:44:09 executing - command=train
00:44:18 executing - command=get_parallelism
00:44:18 using a parallelism of 1
00:44:18 executing - command=infer
00:46:48 checking determinism by executing the inference again with 10% of the data (tolerance: 1e-08)
00:46:48 executing - command=infer
00:46:59 save prediction - path=prediction
00:46:59 determinism check: passed
00:46:59 ended
00:46:59 duration - time=00:02:50
00:46:59 memory - before="272.04 MB" after="445.09 MB" consumed="173.05 MB"
```

The official runner also performed data file existence checks and reported
`already exists, file length match` for `X_train`, `X_test.reduced`, `y_train`,
`y_test.reduced`, `y_test_index.reduced`, and `y_train_index`. No reduced test
data was manually inspected or used outside the official Crunch test invocation.

Earlier official attempts:

- `.ipynb` as `--main-file`: failed before user code with
  `AttributeError: 'NoneType' object has no attribute 'loader'`.
- generated `.py`, P=4, shared `_sbr_payload`: hung after four `infer` workers.
- generated `.py`, P=4, process-specific payload and lazy LightGBM import:
  failed after `00:00:12` with two worker segfaults, `exit codes: 0=-11, 1=-11`.
- generated `.py`, P=4, added native thread caps and delayed NumPy import:
  same two-worker segfault, `exit codes: 0=-11, 1=-11`.

The final one-worker artifact is the tested artifact.
