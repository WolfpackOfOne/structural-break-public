# Void-run safeguards — the CRF-02 lesson, applied to production

`RT-1237` / `RT-1238` / `RT-1239` were voided because a run silently loaded a
24-series, 1-epoch null written by a **unit test** instead of the preregistered
one. The checkpoint's checksum was valid. It was caught by **compute
accounting** — `pretrain_runtime_s = 0.2` against a real 1,284.6 s — not by any
integrity check. It mattered scientifically, not just procedurally: the toy null
made the derangement control *tie*, which would have supported the wrong
mechanism.

## The four general rules

### 1. Integrity is not provenance
A checksum proves *the bytes are the ones we stored*. It says nothing about
**which** artifact this is — what config produced it, on what data, for how
long. Every load of a production-critical artifact must assert the *identity*
of the thing, not just its intactness.

### 2. Unit-test artifacts must be physically unable to collide with real ones
Not "named differently by convention" — isolated by construction, in a
temporary directory the test cannot escape. A module-level `CACHE` global is a
collision waiting for the next test that touches it.

### 3. Runtime accounting exposes impossible artifacts
A model that "trained" in 0.2 s did not train. Cheap, coarse resource
expectations (wall clock, row counts, file sizes) catch substitutions that no
hash will, because the wrong artifact is usually the *cheap* one.

### 4. Metadata must prove the fit population and config
Fold, seed, series set (hashed), epochs, window, architecture, code version.
If the metadata cannot distinguish the real artifact from a toy one, it is not
provenance metadata.

## Application to RT-600 production assets

| # | Check | Status | Where |
|---|---|---|---|
| 1 | Model directory asserts fit population, not just feature bank | **ADDED** | `ProductionModel._check_provenance` — pins `n_series=10000`, `partition=folds_final10k`, `folds_sha256`, `n_boosters=7`, calibration kind and time coordinate. Refuses to load otherwise. |
| 2 | The pin cannot drift from the frozen record | **ADDED** | `test_expected_provenance_matches_the_reproducibility_manifest` cross-checks every field against `research/FINAL_REPRODUCIBILITY_MANIFEST.json`. |
| 3 | The gate actually refuses the CRF-02 failure mode | **ADDED** | `test_model_with_wrong_provenance_is_refused` — parameterised over a 24-series model, an 8,000-series dev model, the wrong partition, a different fold assignment, a rejected calibration blend, and the pre-W4-E3 time coordinate. All six load cleanly through `_check_manifest` and are caught only by the provenance gate. |
| 4 | Deliberate research loads remain possible | **ADDED** | `SBR_ALLOW_UNPINNED_MODEL=1`, documented, tested, and never set in production. |
| 5 | Feature-bank identity | **ALREADY PRESENT** | `_check_manifest` hard-errors on `feature_manifest_sha256` mismatch. |
| 6 | CRF-02 test cache isolation | **ALREADY PRESENT** | `tests/test_crf02_causality.py::_isolate_cache` (autouse) plus the checkpoint fingerprint in `crf02_acgn.null_fingerprint`. |
| 7 | CRF-01 test cache isolation | **ADDED (defence in depth)** | `tests/test_crf01_causality.py::_isolate_cache`. No live leak: the CRF-01 tests build their own channel matrix and never reach `build_channels`/`emit`. But `K.CACHE` is a module global and the next test calling either would write `cache/crf01/` for real. |
| 8 | Runtime accounting on production inference | **PRESENT, NOT AUTOMATED** | `research/FINAL_REPRODUCIBILITY_MANIFEST.json.local_harness` records `1.909 ms/point` over 20 series. Re-measured this audit at 1.72–1.87 ms/pt. There is no automatic assertion that a run took a plausible amount of time — see the residual risk below. |

## Residual risks (not fixed here; owner decision)

* **`build_channels` cache validation is shape-only.** `research/scripts/crf01_nncsr.py`
  accepts a cached `channels.npy` if `n_rows` and the channel-name list match.
  That is integrity plus shape, not provenance. CRF is a closed program and this
  is research code, so it is recorded rather than changed.
* **No runtime-plausibility assertion in the production path.** Rule 3 is applied
  by hand, not by code. A cheap `assert elapsed > floor` in the submission
  harness would close it; it is out of scope for this audit because it changes
  the shipped entry point.
* **The provenance pin is a constant in source.** If the owner ever re-fits
  RT-600, `EXPECTED_PROVENANCE` and the reproducibility manifest must move
  together — the cross-check test enforces that they agree, which is the point.
