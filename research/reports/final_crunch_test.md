# FINAL OFFICIAL `crunch test` — PASSED

**Experiment `RT-600` — the artifact to submit.**
Date 2026-08-21. Branch `research/wave3-integration`.

## Verdict

| | |
|---|---|
| official `crunch test` | **PASSED**, exit code 0 |
| determinism re-run (10% of data, tol 1e-08) | **passed** |
| `INFER_PARALLELISM` | 1 |
| duration | **00:01:40** |
| memory consumed | 2.55 GB (212 MB → 2.77 GB) |
| local isolated harness | **PASS**, 1.784 ms/point |
| Crunch CLI | 11.11.0 |

## Provenance — the defect that this replaces

The previous artifact's manifest carried
`code_git_sha = b5ea9d1d9cd87f06f574c5a78e4c850f41ef852f`, which resolves in no
repository and on no remote (`git log -1 b5ea9d1…` → `fatal: bad object`). The
provenance chain was broken and could not be repaired by editing the manifest.

| | |
|---|---|
| model trained at | `41ab0695a906834361298b4a61c3636909ceb79c` — reachable, clean |
| artifact built at | `61eec56f745eef77e10a23d7c7b5233a9c5146ff` — reachable, embedded source clean |
| `src/sbr` identical between them | **yes**, `git diff 41ab069 61eec56 -- src/sbr` is empty |

The two SHAs differ only because `research/scripts/build_submission.py` was
committed between training and building. The *embedded* source — the thing the
`code_git_sha` is a claim about — is byte-identical.

## Hashes

| artifact | sha256 |
|---|---|
| source zip | `e3efa5fc81a50fb4abdcf12f6072ade046b33b581e50434326bc05cef7c87cb7` |
| model zip | `6bbab7ed5a0ccef8c4b3c9258c319d93e8a639907872cc342901fc3f6f65329b` |
| model manifest | `1483a59a268ded18b6af804a5f0eb0b191376a1dc86526f286751bf1613ec940` |
| feature manifest | `1646c3b9e09d8a7fb3c564483a1d1d999680caeefe848b092f907a6cac80cced` |
| notebook (28,049,853 B) | `8753659f908673ce1f1f7ddd8ab087c9efa2c6e048200ba53f902c14db64910b` |
| Python entrypoint (28,047,061 B) | `f61871aaf6a8c0b961115b7f2a1e9a29e0e40baddf5dfadcaeee5a6b69d4a47c` |
| `folds_final10k.parquet` | `bf0cdf642bde018a632663ae7d211714a173fb64ef824b15416caf2649e0c716` |

Notebook and `.py` are now emitted from the **same** `code_cells` list by
`build_submission.py`. The `.py` is required because CLI 11.11.0 fails on an
`.ipynb` `--main-file` with `AttributeError: 'NoneType' object has no attribute
'loader'`, and `crunch convert` comments out the payload variables — producing a
submission with no model in it. Previously the `.py` was made by a shell
one-liner run afterwards; two artifacts, one of them not built by the builder.

**The artifacts are not bit-reproducible across builds**: the boot cell embeds a
`built at` UTC timestamp, so two builds of identical source and model differ in
the notebook and `.py` hashes. The `source_zip` and `model_zip` hashes *are*
stable and are the ones to verify.

## THE FAILURE THAT NEARLY SHIPPED A TEST OF THE WRONG COMPETITION

The first run of this test **failed**, and how it failed matters more than that
it failed.

```
project: found /home/user
...
ValueError: could not convert string to float: 'value'
```

The Crunch CLI resolves its project by walking *up* from the working directory
looking for `.crunchdao/project.json`. This worktree had none, so the walk
reached `$HOME` and found one dated **May 2025**:

```json
{"competitionName": "structural-break", ...}      <-- the 2025 competition
{"competitionName": "structural-break-real-time", ...}   <-- what we are in
```

The CLI then validated the **2025** data release — four files — and handed
`infer` a pandas DataFrame whose iteration yields the column name `'value'`.

The tell is in the file list. A correct run validates **six** files:

```
X_train, X_test.reduced, y_train, y_test.reduced,
y_test_index.reduced, y_train_index          <-- the last two are real-time only
```

A wrong-competition run validates four. **Any `crunch test` log that does not
list `y_test_index.reduced` and `y_train_index` was run against the wrong
competition**, whatever it says about passing.

Fixed by placing the correct `.crunchdao` in the worktree. `.crunchdao/` is now
in `.gitignore` — it contains an auth token — with a comment recording this trap,
because the failure mode is a silent wrong-competition test rather than an error.

## Exact command and output

```bash
cd "/path/to/workspace/structural-break-claude-wave3"
env SBR_ROOT="$PWD" .../.venv/bin/crunch test \
  --main-file "$PWD/submissions/C_ensemble_deployable.py" \
  --model-directory "$PWD/resources"
```

```text
02:56:43 started
02:56:43 running local test
02:56:44 executing - command=train
02:56:48 executing - command=get_parallelism
02:56:48 using a parallelism of 1
02:56:48 executing - command=infer
02:58:16 checking determinism by executing the inference again with 10% of the data (tolerance: 1e-08)
02:58:16 executing - command=infer
02:58:23 save prediction - path=prediction
02:58:23 determinism check: passed
02:58:23 ended
02:58:23 duration - time=00:01:40
02:58:23 memory - before="212.07 MB" after="2.77 GB" consumed="2.55 GB"

project: found /path/to/workspace/structural-break-claude-wave3
data/X_train.parquet: already exists, file length match
data/X_test.reduced.parquet: already exists, file length match
data/y_train.parquet: already exists, file length match
data/y_test.reduced.parquet: already exists, file length match
data/y_test_index.reduced.parquet: already exists, file length match
data/y_train_index.parquet: already exists, file length match
```

No reduced test data was inspected outside the official runner's own mechanical
access.

## Runtime budget

Local isolated speed **1.784 ms/point** (7 boosters), corroborated by the W4-E4
bench at 1.734 ms/pt.

| set | points | projection |
|---|---|---|
| public | 5.0M | **2.5 h** |
| private | 10.1M | **5.0 h** |
| combined | 15.1M | **7.5 h** |

Against a **15-hour** weekly budget, with the determinism re-run included in the
runner's own accounting. Comfortable margin. The previous artifact measured
2.77 ms/pt; this one is faster because the boosters are the same size and the
harness ran on an idle machine.

`INFER_PARALLELISM` stays at **1**. P=4 segfaulted LightGBM in two workers on
this runner across three separate mitigation attempts (shared payload;
process-local payload with lazy LightGBM import; native thread caps). There is
nothing to buy — 7.5 h of a 15 h budget — and a submission to lose.

## Prior failed attempts

| attempt | outcome |
|---|---|
| `.ipynb` as `--main-file` | `AttributeError: 'NoneType' object has no attribute 'loader'`, before user code |
| generated `.py`, P=4, shared payload | hung after four `infer` workers |
| generated `.py`, P=4, process-local payload, lazy import | two worker segfaults, `exit codes: 0=-11, 1=-11` |
| generated `.py`, P=4, native thread caps | same two-worker segfault |
| generated `.py`, P=1, **no `.crunchdao` in worktree** | ran against the **2025** competition; `could not convert string to float: 'value'` |
| generated `.py`, P=1, correct project | **PASSED** |
