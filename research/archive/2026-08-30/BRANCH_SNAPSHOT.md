# Pre-consolidation repository snapshot — 2026-08-30

This snapshot was created before importing any research or production lineage into `release/2026-research-consolidation`. The branch itself was created directly from `main` at `cef5458da2d51fbb8513ad71177fca426616ee28`.

## Scope

- 32 remote branch heads were inventoried before consolidation.
- 16 existing milestone tags were inventoried before consolidation.
- One open pull request existed: #12, `chore/repo-cleanup-2026` -> `main`, from the 2026-08-24 cleanup pass. It predates the RT-1257 / Deep Ensemble work and was not merged or closed during this inventory.
- No branch was deleted, rebased, force-pushed, or moved during the read-only inventory.
- `main` was not modified.

See `BRANCH_HEADS.csv` for the exact branch-head snapshot and `RESULTS_SNAPSHOT.csv` for the exact pre-consolidation `research/current` ledger blob.

## Existing tags observed

- `lb-003-running`
- `multi-agent-2026-final`
- `oracle-information-frontier-2026-study`
- `reproduce-2025-public-solution-audit`
- `rt600-production-0.6268`
- `rt600-production-reliability-2026`
- `wave2-2026-final`
- `wave3-engineering-rt150`
- `wave3-integration-final`
- `wave5-alpha-c1c2c3-superseded`
- `wave5-alpha-final`
- `wave7-d3r-information-frontier`
- `wave7-early-attempt-superseded`
- `wave7-t2-final`
- `wave7-t2-promotion-mostly-redundant`
- `wave8-future-aware-final`

No RT-1257 production tag existed at the time of this snapshot.

## Material findings established before consolidation

1. `research/current` is stale as a canonical results trunk: its `research/RESULTS.csv` contains no `RT-1250` row and therefore omits the later learner-diversity / CatBoost champion lineage.
2. The sibling ledgers are generations, not five identical copies of the complete RT-1250..RT-1265 range. `research/catboost-specialist-2026`, `research/deep-ensemble-frontier-crunch-2026`, and `research/multi-agent-frontier-20260829` share the same ledger blob SHA `644684a5c622d52c472b22dbac6f9b8bc633a2a6`, carrying the RT-1250/1251/1252 and RT-1254..RT-1257 generation. `research/deep-ensemble-frontier-local-2026` has a later ledger blob SHA `4b196a1acdf3fa3df1846b254e8ed52fefa3d64d` that adds RT-1260..RT-1265. The completed RT-1258/RT-1259 GPU binding verdicts are documented in reports/commits but were not found as rows in the inspected branch-head ledgers.
3. CSA-04R supersedes the original RT-1264 best-k conclusion. RT-1265 selects k*=2, CAT-413 + CAT-300, exactly RT-1257, under the corrected fixed E2-E0 endpoint and parsimony/noise rule. RT-1264 is historical evidence, not a live candidate.
4. The specific CI dependency concern that `production/rt600` omits LightGBM/Numba/PyArrow is not supported by the current files: `requirements-dev.txt` includes `requirements.txt`, and `requirements.txt` pins those runtime packages. However, no GitHub Actions run or check-run exists on the current `production/rt600` head, so CI execution remains UNVERIFIED until an actual clean PR run executes the advanced tree.

## Preservation rule

This directory records the repository as it existed before consolidation. Do not rewrite these snapshot files to match later conclusions. Later corrections belong in the consolidation report.