# LB-001 / LB-002 — RT-600 external calibration baseline

> **STATUS: the live submission is LB-002 (submission #2).** LB-001 (submission #1)
> uploaded cleanly but **failed on the cloud runner at import** with a
> `PermissionError` and never produced a score. LB-002 is the same model with a
> fixed boot cell — the source and model zip hashes are bit-identical. Sections 1–13
> below describe LB-001 and remain accurate for it; **section 14 is the live record.**

**PURPOSE: EXTERNAL CALIBRATION BASELINE — NOT FINAL MODEL SELECTION.**

This submission exists to obtain our first real 2026 leaderboard number so we can
measure internal-to-external transfer. Its score is a *calibration point*, not a
hyperparameter oracle. No parameter inside RT-600 may be selected using it.

| field | value |
| --- | --- |
| date (UTC) | 2026-08-21 |
| competition | `structural-break-real-time` (2026 Real-Time Edition) |
| release label | LB-001 |
| artifact | RT-600 |
| git branch | `research/wave3-integration` |
| git SHA | `17bb5dfe651738f46766d27a77c14db2c476a5a1` |
| checkout | clean worktree, `git status` empty at release time |

## 1. Authority of this record

> The previous `research/reports/final_crunch_test.md` documents a **prior build**.
> LB-001 was retested after the deterministic ZIP fix
> (`73b566202bc0670d114c25ec9d0abcb277443a2f`, which moved packing from
> filesystem-mtime-dependent archives to deterministic content-addressed ZIPs);
> the hashes below are the **exact submitted artifact**.

`final_crunch_test.md` is preserved unchanged as historical evidence. It is not
wrong — it describes a different, earlier build. This file is the authoritative
release record for the artifact actually submitted as LB-001.

## 2. Artifact identity — expected vs actual

Entrypoint: `submissions/C_ensemble_deployable.py`
Model/resource directory: `resources/` (byte-identical to `models/final10k_ensemble/`)

| artifact | expected sha256 | actual sha256 | result |
| --- | --- | --- | --- |
| source zip (embedded) | `199db8c9…a413a0` | `199db8c9f5db7e1429f6ae09fa018d43cc58c5eb47b5c623a017799537a413a0` | **MATCH** |
| model zip (embedded) | `6c8960dd…99ea8c` | `6c8960ddc7331afecfc6980974f2879401a1eb583f15f5cad34c2ea03299ea8c` | **MATCH** |
| Python entrypoint | `660c88c1…ba2647` | `660c88c104c990626a5758f8d36af49c87273a587e399d86bbc353cc9bba2647` | **MATCH** |
| notebook | `8332b698…a73cc5` | `8332b698b6dbe91376c79d2051c5d294f515f77dea835a278b858b45bda73cc5` | **MATCH** |
| feature manifest | `1646c3b9…80cced` | `1646c3b9e09d8a7fb3c564483a1d1d999680caeefe848b092f907a6cac80cced` | **MATCH** |
| model manifest | `1483a59a…613ec940` | `1483a59a268ded18b6af804a5f0eb0b191376a1dc86526f286751bf1613ec940` | **MATCH** |

Verified independently of the artifact's own header comment: the base64 payloads
were decoded out of the `.py` and hashed directly. The embedded `model/manifest.json`
hashes to `1483a59a…`, i.e. **byte-identical to the on-disk model manifest** — the
artifact carries exactly the model in `resources/`.

`resources/` was diffed file-by-file against `models/final10k_ensemble/`: identical.

## 3. Provenance

Read out of the model manifest **embedded in the submitted artifact**:

| check | required | observed | result |
| --- | --- | --- | --- |
| `trained_on.n_series` | 10000 | 10000 | PASS |
| `trained_on.partition` | `folds_final10k` | `folds_final10k` | PASS |
| `n_features` | 500 | 500 | PASS |
| `calibration.kind` | `scdf` | `scdf` | PASS |
| `calibration.time_coord` | `log_n_seen` | `log_n_seen` (all 7 models) | PASS |
| SCDF anchors | 12 | 12 per model | PASS |
| `INFER_PARALLELISM` | 1 | 1 | PASS |
| streams | 7 | 7 boosters, 7 calibrators | PASS |
| `code_git_sha` | `41ab0695…` | `41ab0695a906834361298b4a61c3636909ceb79c` | PASS |
| training SHA reachable in git | yes | yes, ancestor of release HEAD | PASS |

Seven deployed specialist streams:

| stream | seed | n_columns | max_train_rows |
| --- | --- | --- | --- |
| RT-100R | 0 | 500 | 1,250,000 |
| RT-120R | 0 | 261 | 1,125,000 |
| RT-121R | 1 | 239 | 1,125,000 |
| RT-122R | 7 | 500 | 1,125,000 |
| RT-123R | 0 | 500 | 875,000 |
| RT-124R | 3 | 170 | 875,000 |
| RT-125R | 11 | 500 | 875,000 |

**Source identity.** The 33-file `src/sbr` tree embedded in the artifact was hashed
file-by-file against the git trees at the model's training SHA `41ab0695`, the build
SHA `73b56620`, and release HEAD `17bb5dfe`. All three: **zero content differences,
zero files missing on either side.** The deployed implementation is byte-identical to
the code the model claims it was trained with.

## 4. Release-critical tests

Run with `SBR_MODEL_DIR=<worktree>/models/final10k_ensemble` so model-dependent
tests execute rather than skip, in the environment matching the recorded dependency
manifest (Python 3.11.6, numpy 2.4.6, pandas 3.0.5, scipy 1.17.1, scikit-learn 1.9.0,
lightgbm 4.7.0, numba 0.67.0, pyarrow 25.0.1).

    tests/test_no_n_online_leakage.py
    tests/test_calibration_time_coord.py
    tests/test_production_contract.py

**94 passed, 0 failed, 0 skipped.**

The no-`n_online` causality gate **actually ran** — it did not skip. All four of its
cases passed by name:

- `test_infer_never_measures_the_online_length` — PASSED
- `test_score_k_is_emitted_after_exactly_k_plus_one_points` — PASSED
- `test_a_prefix_scores_identically_however_the_series_continues` — PASSED
- `test_the_gate_itself_catches_a_length_peek` — PASSED (the gate's own self-test)

Covered by the passing set: no `len(x_online)`, no indexing of `x_online`, no second
iteration, score *k* emitted after exactly *k+1* observations, prefix invariance,
future continuation cannot change prefix scores, all scores finite, all scores in [0,1].

## 5. Competition verification

| check | value |
| --- | --- |
| `.crunchdao/project.json` `competitionName` | **`structural-break-real-time`** |
| project resolved by CLI | `/path/to/workspace/structural-break-claude-wave3` |
| `.crunchdao/` gitignored | yes (`.gitignore:73`) — token never committed |

**The wrong-competition trap is real and was checked, not assumed.** An upward walk
from the release checkout finds a *second* config at `~/.crunchdao/project.json` whose
`competitionName` is **`structural-break`** — the 2025 competition. That is what an
earlier test accidentally ran against. The nearest config wins, so `crunch test` and
`crunch push` must be invoked with the working directory at the worktree root. The
test log confirms the CLI resolved the correct project.

### Six-file real-time sentinel: **PASS**

The run recognised six dataset files, which only the 2026 Real-Time Edition has
(data release 234):

    X_train.parquet
    X_test.reduced.parquet
    y_train.parquet
    y_test.reduced.parquet
    y_test_index.reduced.parquet
    y_train_index.parquet

A four-file layout would have meant the 2025 competition and an immediate stop.

## 6. Official `crunch test`

Command (exact CLI syntax taken from `crunch test --help`, not guessed):

    env SBR_ROOT="$PWD" .../.venv/bin/crunch test \
      --main-file "$PWD/submissions/C_ensemble_deployable.py" \
      --model-directory "$PWD/resources"

| field | value |
| --- | --- |
| start (UTC) | 2026-08-21T14:02:33Z |
| finish (UTC) | 2026-08-21T14:04:22Z |
| duration | 00:01:43 |
| memory | before 213.22 MB → after 1.61 GB, consumed 1.4 GB |
| Crunch CLI version | 11.11.0 |
| Python | 3.11.6 |
| parallelism | 1 |
| determinism tolerance | 1e-08 |
| determinism result | **passed** |
| exit code | **0** |
| payload self-check | `payload ok; source and model verified by sha256` |

The artifact was **not** rebuilt before testing.

## 7. Artifact did not change across the test

| | sha256 |
| --- | --- |
| PRE-TEST `C_ensemble_deployable.py` | `660c88c104c990626a5758f8d36af49c87273a587e399d86bbc353cc9bba2647` |
| POST-TEST `C_ensemble_deployable.py` | `660c88c104c990626a5758f8d36af49c87273a587e399d86bbc353cc9bba2647` |
| **MATCH** | **true** |

Model payload post-test: `resources/manifest.json` = `1483a59a…613ec940`, still
byte-identical to `models/final10k_ensemble/`. The only filesystem residue was the
run's self-extracted `_sbr_payload_<pid>/` scratch directory, removed after the run.

**The artifact tested is the artifact submitted.**

## 8. Internal reference performance

Internal reference level: **RT-420 / comparable development specialist ensemble
≈ 0.62581 TS-AUC** (cross-validated, folds 0.63828 / 0.62040 / 0.63392 / 0.61750 /
0.61894).

**This is the development-architecture score.** It is *not* a measured OOF score of
the deployed final 10k model, and must not be quoted as one. The final model is fitted
on all 10,000 labelled series and therefore has no held-out score of its own; 0.62581
is used purely as the internal reference level against which external transfer is
measured.

## 9. Predeclared interpretation bins

Fixed **before** the score was seen, and not to be altered after:

| public TS-AUC | classification |
| --- | --- |
| < 0.605 | SEVERE external generalization gap |
| 0.605 – <0.612 | MATERIAL external haircut |
| 0.612 – <0.618 | CONSISTENT with conservative/base expectation |
| 0.618 – <0.624 | GOOD transfer |
| 0.624 – 0.628 | EXCELLENT transfer / validation well calibrated |
| > 0.628 | BETTER THAN EXPECTED transfer |

## 10. Submission record

| field | value |
| --- | --- |
| submitted | **YES** |
| competition | `structural-break-real-time` |
| project | `project/8776/treaming-detector-v1` |
| **submission ID** | **#1** |
| dashboard URL | https://hub.crunchdao.com/competitions/structural-break-real-time/projects/8776/treaming-detector-v1/submissions/1 |
| message | `RT-600 baseline calibration` |
| upload start (UTC) | 2026-08-21T14:10:52Z |
| upload finish (UTC) | 2026-08-21T14:15:43Z |
| `crunch push` exit code | 0 |
| CLI confirmation | `submission #1 succesfully uploaded!` |
| bundle | code 210 files + model 45.53 MB (7 boosters + manifest) |
| status | **SUBMITTED / PENDING SCORING** |

Command (relative paths — see below):

    env SBR_ROOT="$PWD" .../.venv/bin/crunch push \
      -m "RT-600 baseline calibration" \
      --main-file submissions/C_ensemble_deployable.py \
      --model-directory resources

### Note: `push` requires relative paths, `test` does not

The first push attempt used the same **absolute** paths that `crunch test` accepts
and was rejected server-side before any submission was created:

    "code": "RelativePath", "property": "mainFilePath",
    "message": "path must be relative"

Exit code 1, **no submission created** — submission #1 is the only submission, and it
is the one described by this document. Retried with paths relative to the checkout
root and it succeeded. Worth remembering for the next release: `test` and `push`
disagree about path style.

The upload bundle was audited: it contained no `data/`, no `.crunchdao/`, no `.env`,
and no `prediction/` — the competition data and the auth token were not transmitted.

### Artifact identity across the whole release

The entrypoint hashed `660c88c104c990626a5758f8d36af49c87273a587e399d86bbc353cc9bba2647`
at every checkpoint — before `crunch test`, after `crunch test`, immediately before
`crunch push`, and after `crunch push`. The model manifest hashed
`1483a59a268ded18b6af804a5f0eb0b191376a1dc86526f286751bf1613ec940` at all four.
Nothing was rebuilt, retrained, recalibrated, or repacked at any point.

## 11. Score

**PENDING.** Scoring is asynchronous; `crunch push` returned no score, and the CLI
exposes no submission-status command. No score is recorded here because none was
returned — nothing is estimated or inferred.

When the public TS-AUC arrives, record it as `LB-001_PUBLIC_TS_AUC` and compute:

    external_gap = LB-001_PUBLIC_TS_AUC - 0.62581

reported in both raw AUC and TS-AUC percentage points, then classify it using the
predeclared bins in section 9 — which are fixed and must not be adjusted to fit the
result. Rank may be estimated only against the previously captured leaderboard
snapshot (#1 65.10%, #10 64.10%, #25 63.60%, #50 62.91%, #99 61.80%), and must be
labelled *approximate rank against snapshot, not necessarily current live rank*.

## 12. Standing rule for this submission

LB-001 is an external **calibration point**, not a hyperparameter oracle.

The score **may** change research priorities — e.g. "internal validation transfers
poorly, prioritise robustness and distribution-shift diagnostics", or "validation
transfers well, continue searching for new causal alpha".

The score **may not** be used to select anything inside RT-600. No seed, no SCDF
anchor count, no ensemble weight, no feature, and no LightGBM parameter may be
changed because of it, and the response to a disappointing number is not a spread of
minor variants. Tuning to this number would destroy the only thing it is good for.

## 13. Reproducing this verification

    python3 research/scripts/validate_rt600_release.py

Read-only; 30 gates covering checkout, artifact hashes, embedded-payload identity,
provenance, competition config, and the release-critical tests. It fails loudly rather
than skipping. Result at release time: **30/30 PASSED**.


---

# 14. LB-002 — the cloud-runnable rebuild (LIVE)

## 14.1 What happened to LB-001

LB-001 uploaded successfully and then died on the cloud runner **at import**, before
`train()` or `infer()` was ever called:

    File "/context/code/submissions/C_ensemble_deployable.py", line 21, in <module>
      os.makedirs(_WORK, exist_ok=True)
    PermissionError: [Errno 13] Permission denied: '/context/code/_sbr_payload_96'

The boot cell unpacked its payload into `./_sbr_payload_<pid>` — relative to the
current working directory. That is writable on a laptop and **read-only** at
`/context/code` on the runner.

**No local gate could have caught this.** `crunch test` runs with a writable cwd, so
the local run didn't fail — it silently created `_sbr_payload_56927/` in the repo
root. That directory was noted during the LB-001 release and deleted as harmless
residue. It was not residue; it was the symptom, and it was visible in the LB-001
record before the submission went out.

## 14.2 The fix

Two changes, both in the generated wrapper, none in the model:

1. **Payload extraction** — `research/scripts/build_submission.py`, boot cell:

       -_WORK = os.path.abspath(f"./_sbr_payload_{os.getpid()}")
       +_WORK = os.environ.get("SBR_PAYLOAD_DIR") or tempfile.mkdtemp(prefix="sbr_payload_")

   `tempfile` honours `TMPDIR` and falls back to `/tmp`; `SBR_PAYLOAD_DIR` is an
   explicit operator override.

2. **A second failure that was queued behind the first** — `train()` copied 45 MB
   into `model_directory_path` and raised if it couldn't. A read-only model directory
   would have produced a fresh traceback one step later. `train()` now degrades to a
   printed warning, and `_load_model()` falls back to the embedded, sha256-verified
   payload when the handed-in directory has no `manifest.json`. Both are the same
   bytes, so the scored function is identical either way; what disappears is the
   dependency on anything outside the artifact being writable.

## 14.3 Why this is not a model change

| identity | LB-001 | LB-002 | |
| --- | --- | --- | --- |
| **source zip sha256** | `199db8c9…a413a0` | `199db8c9…a413a0` | **UNCHANGED** |
| **model zip sha256** | `6c8960dd…99ea8c` | `6c8960dd…99ea8c` | **UNCHANGED** |
| **feature manifest** | `1646c3b9…80cced` | `1646c3b9…80cced` | **UNCHANGED** |
| **model manifest** | `1483a59a…613ec940` | `1483a59a…613ec940` | **UNCHANGED** |
| python entrypoint | `660c88c1…ba2647` | `acc16684…7312d16` | changed (wrapper) |
| notebook | `8332b698…a73cc5` | `830ae6f2…ab75d47` | changed (wrapper) |

The boot cell is in **neither** embedded zip, so a wrapper fix cannot move the two
content-addressed identities — which is precisely what the deterministic-ZIP change
in `73b5662` was built to guarantee. The rule to read this table by:

> If the source or model zip hash moves, the model changed.
> If only the entrypoint and notebook move, the packaging changed.

Re-verified against the manifest embedded in the **new** build, not carried over from
the old one: 10,000 series, 500 features, SCDF/`log_n_seen` on all 7 calibrators,
seven streams RT-100R/120R/121R/122R/123R/124R/125R, `INFER_PARALLELISM = 1`, and the
33-file `src/sbr` tree still byte-identical to the tree at training SHA `41ab069`.

**LB-002 is therefore the same calibration experiment as LB-001, not a new one.** No
seed, feature, weight, or calibration parameter was touched, and nothing was tuned to
any leaderboard result — there was no leaderboard result to tune to.

## 14.4 New release gate

`tests/test_artifact_readonly_cwd.py` imports the built artifact with cwd set to a
directory the process cannot write to, reproducing the runner's constraint directly.

- Against the **LB-001** artifact (`660c88c1…`): all three cases **FAIL** with the
  same `PermissionError: [Errno 13] … /_sbr_payload_<pid>` the cloud produced.
- Against the **LB-002** artifact (`acc16684…`): all three **PASS**.

It also asserts the artifact writes *nothing* into its cwd, and that
`SBR_PAYLOAD_DIR` is honoured. It is now in the validator's release-critical set,
because this is a failure class `crunch test` structurally cannot see.

## 14.5 Verification of LB-002

    python3 research/scripts/validate_rt600_release.py    →  30/30 PASSED

Release-critical tests (`test_no_n_online_leakage`, `test_calibration_time_coord`,
`test_production_contract`, `test_artifact_readonly_cwd`), with `SBR_MODEL_DIR` set:
**97 passed, 0 failed, 0 skipped.** The four no-`n_online` causality gates ran.

Official `crunch test`:

| field | value |
| --- | --- |
| start / finish (UTC) | 2026-08-21T14:39:20Z → 14:41:12Z |
| duration | 00:01:47 |
| memory | before 211.91 MB → after 605.09 MB, consumed 393.18 MB |
| parallelism | 1 |
| determinism | **passed** (tolerance 1e-08) |
| exit code | **0** |
| dataset files | **6** — the real-time sentinel |
| competition | `structural-break-real-time` |
| cwd after the run | **clean** — no `_sbr_payload_*` directory, unlike LB-001 |

Artifact hash `acc166842ecb31e1de3bdf1363a2ea119f78a018a248bbc9d17cf097b7312d16` before
the test, after the test, before the push, and after the push. Model manifest
`1483a59a…613ec940` at all four.

## 14.6 Submission record

| field | value |
| --- | --- |
| submitted | **YES** |
| **submission ID** | **#2** |
| competition | `structural-break-real-time` |
| dashboard URL | https://hub.crunchdao.com/competitions/structural-break-real-time/projects/8776/treaming-detector-v1/submissions/2 |
| upload start / finish (UTC) | 2026-08-21T14:41:34Z → 14:46:22Z |
| `crunch push` exit code | 0 |
| CLI confirmation | `submission #2 succesfully uploaded!` |
| git SHA | `fcf21aa2b521ed38be3d3bd854ddbfc7180b32ab` |
| branch | `claude/rt600-baseline-submission` |
| status | **SUBMITTED / PENDING SCORING** |

## 14.7 Cloud training setting

Leave **training enabled**. `train()` is a copy, not a fit: it installs the embedded
pre-trained ensemble and ignores `datasets` entirely, taking ~6 seconds. Enabling it
reproduces the locally validated path and is now self-sufficient in both directions —
if the model directory is unwritable, `train()` warns and `infer()` uses the embedded
copy; if `train()` never runs, `infer()` still finds a model.

## 14.8 Standing rule, restated

Section 12 applies unchanged to LB-002. It is an external calibration point. It may
redirect research priorities; it may not select anything inside RT-600.
