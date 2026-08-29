# Submission #16 — RT-1257 first clean cloud run — SCORED 0.6290

Successor to submission #15, which never reached scoring
(`research/FAILED_EXPERIMENTS.md`, "Submission #15"). #16 is the first RT-1257
artifact to execute end to end on the Crunch runner, and it **scored
TS-AUC 0.6290**, against the `RT-600` anchor of 0.6268.

**This is a deployment record, not a research result.** No RT ID is consumed and
no model was refit. The score is a calibration point: nothing inside RT-1257 may
be selected on it, and promoting RT-1257 to production anchor is an owner
decision that this record does not make — see §5.

## 1. Identity

| field | value |
| --- | --- |
| date (UTC) | 2026-08-29 |
| competition | `structural-break-real-time` (2026 Real-Time Edition) |
| runner task | `run-8cd3656e` |
| submission id | 76781 |
| model id | 73384 |
| data release | 234 |
| artifact | `RT-1257` (two-slot CatBoost/LightGBM hybrid) |
| git branch | `engineering/rt1257-deployment-2026` |
| code git SHA | `e50098a42f28b1007746cb20bbabc157e285b036` |
| built at | 2026-08-29T12:52:38Z |

Hashes, from `submissions/RT1257_deployable.build.json` and re-verified against
the on-disk entrypoint:

| artifact | sha256 | bytes |
| --- | --- | --- |
| entrypoint `submissions/RT1257_deployable.py` | `05eafb66f4589f5b426f4af43779f428f7a90f2d64b423530d6f315a798e888b` | 24,006,505 |
| notebook `submissions/RT1257_deployable.ipynb` | `2461c7002fce3d0a0057f99acfa6749940bc7e2d3fc764351ee21e6bcb3de2e1` | 24,009,226 |
| embedded source zip | `c8e35854675b038103cec6b6da827e1186bf8235bef6a8d3d8d6e00691429676` | — |
| embedded model zip | `952422d7ddb5a5d60719e909d1f0669c612952c72a26fc1859035dfae0381f0a` | — |
| feature manifest | `1646c3b9e09d8a7fb3c564483a1d1d999680caeefe848b092f907a6cac80cced` | — |
| model manifest | `ba4bee741d3d16ccc874724757167b9a7478005c14587bb329b845efca126d4c` | — |

The 24 MB entrypoint stays untracked and is identified by sha256, per the
convention set by the #15 record.

## 2. What the runner did

| stage | outcome |
| --- | --- |
| payload verification | `source and model verified by sha256` at every stage |
| `train` | ok — installs the pre-trained model, does not fit |
| `get_parallelism` | 1, matching `INFER_PARALLELISM = 1` in the entrypoint |
| `infer` | ok |
| determinism re-check | re-ran `infer` on 10% of the data, tolerance 1e-8 — **passed** |
| prediction upload | `prediction.parquet`, 31,252,136 bytes |
| model upload | 8 files, 38,637,862 bytes |
| terminal state | `result submitted` → `CLEANING` |

Both packaging defects that killed #15 are confirmed fixed in the live runner
environment, not merely in local test: the payload unpacked without a
`PermissionError` under the read-only `/context/code` mount (defect 1, fixed in
`733c727`), and `infer` reached completion, which requires the `catboost` import
in `src/sbr/production/model.py` to have resolved (defect 2, fixed in `e50098a`).

## 3. Model files round-tripped

Eight files down, the same eight back up:

| file | bytes |
| --- | --- |
| `manifest.json` | 773,863 |
| `model.cbm.0` | 696,904 |
| `model.cbm.4` | 696,872 |
| `model.txt.1` | 8,500,943 |
| `model.txt.2` | 2,155,913 |
| `model.txt.3` | 17,303,945 |
| `model.txt.5` | 4,235,622 |
| `model.txt.6` | 4,273,800 |

Two CatBoost slots (0, 4) and five LightGBM slots (1, 2, 3, 5, 6) — the RT-1257
two-slot hybrid over the RT-600 seven-stream base, as promoted.

**`has_changed=True` on the model upload is expected and is not a refit.**
`train()` in the entrypoint copies the embedded pre-trained model into
`model_directory_path` with `shutil.copy2`, so the resource is rewritten on every
run and the platform re-uploads it. The uploaded set is byte-identical in
composition to the downloaded set.

## 4. Open items this run exposes

1. **The tracked local-test record is stale.**
   `engineering/reports/rt1257_deployment/CRUNCH_TEST.json` is dated 2026-08-27
   and therefore predates both packaging fixes and the 2026-08-29 build that
   actually shipped. It records `result: PASSED` for an artifact that is not this
   one, and it carries no entrypoint sha256 to make that checkable. Any future
   reader comparing the two will be misled. `crunch test` should be re-run
   against the shipped build and the JSON regenerated with the entrypoint hash
   included.
2. **The build manifest reports a split provenance.**
   `manifest_matches_build_sha: false` — the model manifest was produced at
   `6b4fafacd30711bebf2bf977c3c390641f3ae522`, the code at `e50098a`. Expected,
   since the model was fitted before the packaging fixes, but it means the
   deployed artifact pairs a manifest and a source tree from two different
   commits. Confirm the gap is packaging-only before citing #16 as the
   reproducible deployment baseline.
3. **The submitted payload contained untracked files.**
   `crunch push` uploads the working tree, not `HEAD`. The runner log shows it
   downloading `engineering/reports/rt1257_deployment/oof/RT-1254.final10k.json`
   and `RT-1255.final10k.json`, which are untracked in git. Nothing about the run
   depends on them, but the submitted payload is therefore not reconstructible
   from `e50098a` alone.
4. **The payload ships the whole research tree.** Several hundred files under
   `research/` and `engineering/` are downloaded into `/context/code` on every
   run. Inert, but it is download weight and it widens the artifact's surface.
   Excluding both directories from the source zip in
   `research/scripts/build_submission.py` would not change predictions.

## 5. Result

| field | value |
| --- | --- |
| external score | **0.6290** TS-AUC (62.90%) |
| prior anchor | 0.6268 TS-AUC, `RT-600`, labelled LB-001 in `STATUS.md` |
| external delta | **+0.0022** |

### 5.1 The delta landed where the promotion battery said it would

RT-1257 was promoted on internal evidence alone. Those numbers, from
`reports/catboost_specialist_2026/FINAL.md` via `STATUS.md`:

| internal estimate | value |
| --- | --- |
| `marginal_vs_clone` | +0.002407205 |
| E2−E0 (ensemble vs base) | +0.002026322 |
| positive folds | 5/5 |

The realised external delta of **+0.0022 sits between the two**. That is a
closer agreement than this program has any right to expect from a single read,
and it is the first external movement since 0.6268 was set — Wave 5 ran ten
experiments and moved the score by zero.

### 5.2 What this does and does not license

It does **not** establish an internal-to-external transfer law. This is one
observation, from one pair of leaderboard reads, with no error bar on either.
The honest statement is that the internal battery did not mislead on this
artifact, not that it is calibrated.

Three specific limits on the comparison:

1. **Neither score has a confidence interval.** Both are single reads on the
   same held-out test set. The pairing helps — same data, same metric — but a
   +0.0022 move is not a significance test, and the battery classified RT-1257
   as PROMOTION_WORTHY *but not SERIOUS* precisely because the internal effect
   was small.
2. **Both figures are rounded.** 0.6290 and 0.6268 are given to four
   significant figures, so the delta carries roughly ±0.0001 of rounding slack.
   It does not change the sign or the order of magnitude.
3. **The comparison is not a perfectly controlled A/B.** The model function
   differs in exactly the two swapped slots (RT-300 and RT-413 → CatBoost), but
   the deployment around it also changed between the RT-600 uploads and #16:
   payload unpack location, declared requirements, and the entrypoint build. None of those
   should touch predictions — the determinism check passed at 1e-8 — but they
   were not held fixed, so "the two slots caused +0.0022" is an inference, not a
   measurement.

### 5.3 Promotion status

`RT-1257` now beats the anchor on the only measurement that is not internal.
Promotion is nonetheless **not taken here**: `RT-600` is frozen on
`production/rt600` under tag `rt600-production-0.6268`, and moving the anchor
means a production merge and a new tag, which is an owner-authorised action.
`STATUS.md` records the new external best and leaves the anchor where it is.

Before any promotion, close open items 1 and 2 in §4 — the stale
`CRUNCH_TEST.json` and the unverified `manifest_matches_build_sha: false` split.
An anchor whose local test record describes a different artifact is not an
anchor.
