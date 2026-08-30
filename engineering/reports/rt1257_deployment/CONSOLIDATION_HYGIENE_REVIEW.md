# RT-1257 Consolidation Hygiene Review

Date: 2026-08-30
Purpose: determine what can be verified from repository/GitHub evidence before RT-1257 is called the formal production anchor.

## Executive status

**External result:** VERIFIED — submission #16 is recorded at official TS-AUC 0.6290, versus RT-600 0.6268.

**Exact seven-slot composition:** VERIFIED from the deployment record — CAT-300 and CAT-413 occupy slots 0 and 4; the other five slots remain LightGBM.

**Manifest/build SHA gap is packaging-only in tracked source:** VERIFIED at Git-diff level. The model-manifest code SHA is `6b4fafacd30711bebf2bf977c3c390641f3ae522`; the shipped-build code SHA is `e50098a42f28b1007746cb20bbabc157e285b036`. The latter is three tracked commits ahead. The compare changes engineering evidence, `requirements.txt`, `research/scripts/build_submission.py`, and generated build metadata; it does **not** change `src/sbr` model, feature, calibration, metric, or inference source.

**Fresh Crunch test against shipped/post-packaging build:** **UNVERIFIED / REQUIRED.** The tracked `engineering/reports/rt1257_deployment/CRUNCH_TEST.json` predates the packaging fixes and does not include the entrypoint SHA recorded by the later build manifest.

**Formal production promotion:** **BLOCKED** until the fresh Crunch test is captured. RT-600 therefore remains the formal production anchor during consolidation.

## Artifact identity evidence

`submissions/RT1257_deployable.build.json` records:

- build time: 2026-08-29T12:52:38Z
- code_git_sha: `e50098a42f28b1007746cb20bbabc157e285b036`
- embedded_source_clean: true
- source zip SHA-256: `c8e35854675b038103cec6b6da827e1186bf8235bef6a8d3d8d6e00691429676`
- model zip SHA-256: `952422d7ddb5a5d60719e909d1f0669c612952c72a26fc1859035dfae0381f0a`
- feature manifest SHA-256: `1646c3b9e09d8a7fb3c564483a1d1d999680caeefe848b092f907a6cac80cced`
- model manifest SHA-256: `ba4bee741d3d16ccc874724757167b9a7478005c14587bb329b845efca126d4c`
- entrypoint SHA-256: `05eafb66f4589f5b426f4af43779f428f7a90f2d64b423530d6f315a798e888b`
- manifest_code_git_sha: `6b4fafacd30711bebf2bf977c3c390641f3ae522`
- manifest_matches_build_sha: false

The split is therefore explicit rather than hidden.

## Manifest-code -> build-code diff review

GitHub comparison from `6b4faf...` to `e50098a...` shows three commits. The tracked changed-file set is limited to:

- `engineering/reports/rt1257_deployment/CRUNCH_TEST.json`
- `engineering/reports/rt1257_deployment/RT1257_BENCHMARK.json`
- `engineering/reports/rt1257_deployment/RT1257_FIT.json`
- `engineering/reports/rt1257_deployment/RT1257_OOF.json`
- `engineering/reports/rt1257_deployment/RT1257_VALIDATE.json`
- `engineering/reports/rt1257_deployment/crunch_test.log`
- two CatBoost model-summary JSON records
- `requirements.txt`
- `research/scripts/build_submission.py`
- `submissions/RT1257_deployable.build.json`

No `src/sbr` predictive source changed. The dependency/build changes add the deployment requirements and packaging behavior needed for the hybrid artifact. This supports the statement that the tracked SHA gap is packaging/deployment-only.

This verification does **not** prove that the exact post-packaging artifact has passed `crunch test`; it only narrows what changed in version control.

## Stale Crunch-test record

The tracked `CRUNCH_TEST.json` records:

- competition: `structural-break-real-time`
- result: PASSED
- determinism_check: passed
- duration: 00:02:46
- reported memory consumed: 1.37 GB
- prediction rows: 50,983
- prediction SHA-256: `6241c1ebc76f1719830fde3478a9bbd9edd13dc74089ad5f667ab5a2a7c0f11a`

However, `SUBMISSION_16.md` explicitly states that this test record predates the final packaging fixes and 2026-08-29 build, and that it lacks the entrypoint hash required to tie it to the shipped artifact. Therefore its PASS must not be cited as a post-build RT-1257 qualification.

## Submitted working-tree provenance

`SUBMISSION_16.md` records that `crunch push` uploaded the working tree and that the payload included untracked OOF JSON files. The run did not depend on them, but their presence means the exact pushed source payload is not reconstructible solely from `e50098a`.

Status: **KNOWN PROVENANCE IMPERFECTION.** This does not invalidate the external score, but it is another reason to rebuild/test a clean tracked artifact before establishing a production tag.

## GitHub CI dependency review

The consolidation review was asked to verify whether the RT-600/advanced test tree would fail to import because CI installed only `requirements-dev.txt` rather than `requirements-research.txt`.

Current repository evidence does **not** support that specific failure theory:

- `requirements-dev.txt` includes `requirements.txt`.
- the advanced RT-1257 lineage's `requirements.txt` declares LightGBM and CatBoost; the RT-600 runtime generation also declares LightGBM/Numba/PyArrow through its runtime requirements.
- a real GitHub Actions run triggered by consolidation PR #13 completed the dependency-install step successfully on Ubuntu.

That run then failed at the repository-wide Ruff step before pytest, so full test execution remains **UNVERIFIED** until the CI lint scope is corrected and pytest is reached.

## Required promotion gate

Before changing the formal production anchor from RT-600 to RT-1257:

1. Build RT-1257 from a clean tracked checkout using the canonical build script.
2. Record the new source/build/model/manifest/entrypoint hashes.
3. Run `crunch test` against that exact build.
4. Regenerate `engineering/reports/rt1257_deployment/CRUNCH_TEST.json` so it identifies the entrypoint hash/build manifest it tested.
5. Re-run deterministic/prefix/series-independence/streaming-parity/production-contract tests in the clean environment.
6. Ensure GitHub CI reaches and passes the intended production test set.
7. Only then create the production promotion tag and change `STATUS.md` from “external champion” + “RT-600 formal anchor” to “RT-1257 production champion.”

Until those gates are evidenced, no consolidation document may claim RT-1257 is CI-verified or formally production-promoted.
