# RT-600 Reliability Release 2026 — clean production integration

**Status: `APPROVED_FOR_PRODUCTION_MERGE`**

> ## THIS IS NOT A MERGE OF THE RESEARCH LINEAGE.
>
> `engineering/rt600-final-reliability-2026` is rooted on the CRF research tip
> (`8165876`) and carries **152 commits** that `production/rt600` does not have.
> `git merge` would have imported ten post-freeze research feature modules, the
> CRF/wave-5..8 training code and test suites, the research `RESULTS.csv` and the
> research known-failure ledger. None of that is production. The audited fixes
> were transplanted **path by path** onto a branch that descends from
> `production/rt600`, and the research lineage was not merged.

| | |
| --- | --- |
| production base SHA | `9aaa9b046609eb94d9fdd09f7ae07d9249fe2f65` (`production/rt600`) |
| engineering source SHA | `d47937718ba91c2327aafa4528bdba39ee06661f` (`engineering/rt600-final-reliability-2026`) |
| clean release branch | `release/rt600-reliability-2026` |
| clean release artifact SHA | `f250777d065f57e7128f53c221644cfda777a0f4` — *Rebuild RT600 reliability release artifact*; this is the commit the shipped payload was built from |
| clean release branch head | the *Validate clean RT600 production release* commit that carries this report; see `git log release/rt600-reliability-2026` |
| merge base of the two | `17bb5df` — 152 commits ahead on engineering, 13 ahead on production |
| owner authorisation | 2026-08-26, *APPROVE RT-600 RELEASE*, clean production-descended lineage |

---

## 1. What was transplanted, and why each file

Every path below was checked against the release commits before it was taken:
the diff between `production/rt600` and the engineering tip for each of these
files is **exactly** the release patch, line for line, with no research drift
mixed in. Method was `git checkout d479377 -- <path>` for whole files, and a
single-commit patch apply where the engineering file also carried unrelated
change.

### Runtime source (9 files)

| path | from | why it is required |
| --- | --- | --- |
| `src/sbr/features/m07_bayes.py` | `b41da11` | `_bocpd_ct` extracted as the ONE source of the BOCPD Student-t log-normalising table. numba's `math.lgamma` and CPython's disagree by up to 512 ULP on this grid; two independently computed tables put batch-trained and stream-served features on permanently different constants. Train/serve skew, not float noise. |
| `src/sbr/stream/s_m07_bayes.py` | `b41da11` | imports that table instead of recomputing it. |
| `src/sbr/stream/_fp.py` | `03e4637` | **new.** Dekker two-product / two-sum emulation of a correctly-rounded FMA. `scipy.signal.lfilter` contracts its recursion into an FMA on arm64 and rounds once; a Python `a*b + c` rounds twice and disagrees by 1 ULP on ~20 % of steps, which a stateful IIR carries forward. `math.fma` is 3.13+; the frozen environment is 3.11.6. |
| `src/sbr/stream/s_m01_seq.py` | `03e4637` | `_Ewma` now carries `lfilter`'s `z` state and fuses the multiply-add. |
| `src/sbr/stream/s_m04_resid.py` | `03e4637` | `_fma` on the vol / volM / cmb / AR EWMA recursions. `volG` deliberately **not** fused — batch builds it as two numpy passes, i.e. two roundings. |
| `src/sbr/stream/s_m02_dist.py` | `03e4637` | `mL * mL`, never `mL ** 2`. |
| `src/sbr/stream/s_m06_loc.py` | `03e4637` | `mw * mw`, never `mw ** 2`. A scalar `**2` goes through libm `pow`; numpy's array `**2` is a squaring multiply. m06's rolling variance is a cumsum difference that nearly cancels, so one ULP survives into the emitted float32. **This is the one owner-accepted prediction change.** |
| `src/sbr/production/model.py` | `3e7a9a2` | `_check_provenance`. `_check_manifest` proves the feature bank matches — that is INTEGRITY and says nothing about WHICH model this is. Pins `n_series`, `partition`, `folds_sha256`, seven boosters, `scdf` / `log_n_seen`. `SBR_ALLOW_UNPINNED_MODEL=1` is the documented escape hatch; production never sets it. |

### Tests required for release verification (3 files)

| path | from | why |
| --- | --- | --- |
| `tests/test_stream_fp.py` | `03e4637` | **new.** Verifies `_fp.fma` exact against `fractions.Fraction` on 200,000 random triples. |
| `tests/test_stream_parity_m07_bayes.py` | `b41da11` | pure append: the `ct` single-sourcing guard and a bitwise stream-vs-compiled-kernel comparison. |
| `tests/test_production_contract.py` | `3e7a9a2` (that commit's hunk only) | six wrong-provenance refusals, the escape hatch, and a field-by-field cross-check of the pin against `FINAL_REPRODUCIBILITY_MANIFEST.json`. Applied as a patch rather than checked out whole, because the engineering copy of this file also carries research drift. |

### Release artifacts (4 files, all **rebuilt**, none copied)

`submissions/C_ensemble_deployable.ipynb`, `.py`, `.build.json` and the
`clean_production_release` block appended to
`research/FINAL_REPRODUCIBILITY_MANIFEST.json`. The existing top-level hashes in
that manifest are the record of the 2026-08-21 freeze and were **left
unchanged**.

### Release-validation tooling (not runtime; not packaged)

`production/release/packaged_release_check.py`, `compare_predictions.py`,
`source_tree_predict.py`, `PARITY_SAMPLE.json`, plus a new
`packaged_provenance_check.py` written here for gate E. Nothing in the shipped
payload imports any of it and none of it is in the source zip. One line of
`packaged_release_check.py` differs from the engineering copy: the audited
64-series parity sample is read from `PARITY_SAMPLE.json` beside it instead of
from `engineering/reports/.../FULL_PARITY.json`, which does not exist on this
lineage. The series identities are identical, so it is the same battery.

---

## 2. What was deliberately excluded

| excluded | why |
| --- | --- |
| `src/sbr/stream/engine.py` | The engineering diff registers `StreamM12Rdep` and splits `MODULE_ORDER` from `PRODUCTION_MODULES`. **No release commit touches this file** — it is research drift. On this lineage `MODULE_ORDER` *is* the frozen seven and `StreamEngine` already defaults to it. |
| `tests/test_stream_engine_parity.py` | same drift — the append-only guard for `m12_rdep`. Nothing was appended here. |
| `m10_persist`, `m11_focus`, `m12_rdep`, `m13_scale_survival`, `m14_spectral_impulse`, `m15_ordinal_irrev`, `m16_joint_rarity`, `m17_observers`, `m18_weighted_ctm`, `s_m12_rdep` | post-freeze research modules. They never existed on this lineage, so nothing was added and **no deletion commit was made**. The engineering artifact ships them inert; this one simply does not contain them. |
| `research/scripts/audit_causal_representation_frontier.py`, `tests/test_causal_representation_frontier_audit.py` | the CRF stale-`RESULTS.csv`-hash fix. The CRF audit code is not here and is not required here. Production does not acquire CRF infrastructure to carry its audit fix. |
| `research/known_failures.json`, `research/scripts/known_failure_gate.py`, `research/archive/known_failures/*` | research infrastructure; neither has ever existed on this lineage. See §7. |
| `tests/test_crf01_causality.py`, `test_crf02_causality.py`, `test_neural_causality.py` | CRF / neural research suites. |
| `research/RESULTS.csv` | **no research history merge, no RT ID, no experiment row.** The production copy is untouched. |
| `research/scripts/build_submission.py` | see §3 — the engineering copy is a **regression**. |
| ~320 other research files | wave-5..8 scripts, novel-stream pilots, second-sweep code, CRF reports, research test suites. |

---

## 3. The builder: production's, not engineering's

This is the finding that would have broken the release had the engineering
`submissions/` files been copied across.

`production/rt600` is **13 commits ahead** of the engineering branch on the
deployment wrapper. The research lineage forked before LB-002…LB-004, so
engineering's `research/scripts/build_submission.py` is *older*:

| | engineering copy | production copy (used here) |
| --- | --- | --- |
| payload directory | `./_sbr_payload_<pid>` | `tempfile.mkdtemp()`, `SBR_PAYLOAD_DIR` override |
| `INFER_PARALLELISM` | 1 | **4** |
| `_load_model` | no fallback | falls back to the embedded, sha256-verified payload |
| `train()` on a read-only model dir | raises | survives |

The engineering wrapper fails **at import** on the cloud runner, whose cwd
`/context/code` is read-only — that is exactly how LB-002 died. Copying the
engineering artifact into production would have shipped that regression. The
production builder was used unmodified and is not in the transplant set.

**Known pre-existing inaccuracy, deliberately not fixed:** `build.json` records
`"infer_parallelism": 1` as a literal in the builder while `CELL_ENTRY` ships
`INFER_PARALLELISM = 4`. That mismatch is already on `production/rt600` and
predates this release; correcting it is builder work, not a release transplant.

---

## 4. Artifact hashes

| artifact | production freeze | engineering release | **clean release** |
| --- | --- | --- | --- |
| source zip sha256 | `199db8c9f5db7e14…` | `e5381d518faa51cc…` | **`28b3f95f403dc40f5a1063a890dc714ba95c5d7b8ee994017b61c2c4647a3c9e`** |
| **model zip sha256** | `6c8960ddc7331afe…` | `6c8960ddc7331afe…` | **`6c8960ddc7331afecfc6980974f2879401a1eb583f15f5cad34c2ea03299ea8c`** |
| feature manifest sha256 | `1646c3b9e09d8a7f…` | `1646c3b9e09d8a7f…` | **`1646c3b9e09d8a7fb3c564483a1d1d999680caeefe848b092f907a6cac80cced`** |
| model manifest sha256 | `1483a59a268ded18…` | `1483a59a268ded18…` | **`1483a59a268ded18b6af804a5f0eb0b191376a1dc86526f286751bf1613ec940`** |
| notebook sha256 | `840a94f744e12d06…` | `a32f638c3ad2483d…` | **`5843458e15b0f4601f95f627432929ade77d3414b21d9703d977b908fdc2ba2c`** |
| `.py` sha256 | `74894ed1cd774c5e…` | `42668f521ebcd7df…` | **`36d5e75e5bfcdaff0c8a2ad12a8a6d1dfea9816f02b2eeb78fdc4eec4ef2d0a9`** |
| notebook bytes | 28,051,870 | 28,104,109 | **28,056,514** |

**Is the model artifact unchanged? YES — byte-identical.** `6c8960dd…` matches
both the frozen record and the expected engineering-release hash exactly, so the
stop-and-investigate condition of §9 did not trigger. The feature manifest is
unchanged too, which is why this model still loads: `_check_manifest` refuses a
model whose engine emits different columns.

The source zip differs from both, as it must: it is production's source plus the
audited fixes, without the ten research modules engineering's payload carries.

The notebook and `.py` are not bit-reproducible across builds — the boot cell
embeds a build timestamp. The zip hashes are the reproducible identities.

---

## 5. Source content audit — the packaged clean-lineage zip

**34 members** (engineering's payload has 44).

| must be present | |
| --- | --- |
| `sbr/stream/_fp.py` | present |
| `_bocpd_ct` in `features/m07_bayes.py` | present |
| `s_m07_bayes` imports it | present |
| `_fma` in `s_m01_seq`, `s_m04_resid` | present |
| `mL * mL` / `mw * mw` in `s_m02_dist`, `s_m06_loc` | present |
| `_check_provenance` in `production/model.py` | present |
| shipped module set | `m00_core, m01_seq, m02_dist, m03_dyn, m04_resid, m06_loc, m07_bayes` — **exactly the frozen seven** |

On this lineage `MODULE_ORDER` **is** the frozen seven and `StreamEngine`
defaults to it. The `MODULE_ORDER` / `PRODUCTION_MODULES` split exists only on
the research lineage, where a module was appended; there is nothing to split
here, so the shipped default module tuple is the frozen seven by construction
rather than by filtering.

| must be absent | found |
| --- | --- |
| m10–m18 research modules | none |
| `s_m12_rdep` | none |
| CRF training code | none |
| New Avenues / pilot / second-sweep code | none |
| `import torch` anywhere | none |
| experimental checkpoints | none |
| unit tests or test caches | none |
| `__pycache__`, `.pyc`, `.pyo`, numba `.nbc` / `.nbi` | none |
| `.npy` / `.npz` / `.parquet` / `.pkl` | none |
| user-local paths (`/Users/…`) | none |

**One known non-finding.** `sbr/store.py`, `sbr/pipeline.py` and
`sbr/features/driver.py` carry `/home/claude/sb` as an `os.environ.get`
**default**. Pre-existing on `production/rt600`, untouched by this release
(`git diff` over those three files across the whole release is empty),
overridden by `SBR_STORE` / `SBR_ROOT` / `SBR_FEATURES`, and not read on the
inference path.

---

## 6. Release gates — run against the PACKAGED clean-lineage artifact

Every gate below executed the notebook's own code cells in a scratch directory
with this repository removed from `sys.path`, and asserted `import sbr` resolved
inside the unpacked payload. Records are the `PACKAGED_*.json` files beside this
one.

| | gate | measurement | result |
| --- | --- | --- | --- |
| A | batch/stream parity | canonical **201**-series battery, all seven production modules, **51,460,500** cells | **0 mismatches — PASS** |
| B | prefix invariance | 14 series × 14 prefixes (1…256), stream-truncated and batch-recomputed, **7,261,000** cells | **0 mismatches — PASS** |
| C | series independence | features: 8 series × **1,146,000** cells with foreign series interleaved; predictions: **2,292** predictions, reordered + interleaved through `infer()` | **0 mismatches / 0 differences — PASS** |
| D | deterministic replay | 30 series, 12,467 points, 5 repeats | **1 distinct hash** `0b58afb014c0ab23` — **PASS** |
| E | artifact provenance | payload sha256 self-check + **6/6** wrong-provenance variants refused through the payload's own `ProductionModel`; escape hatch works | **PASS** |
| F | output contract | 12,467 rows, expected 12,467 | 0 NaN, 0 Inf, 0 outside [0,1], min 0.00734, max 0.99291, exact keys, per-series row counts correct, no duplicates or missing rows — **PASS** |

Order independence and future-poison safety were green in the same run.

**Packaging neutrality.** The same 67-series battery run from `src/sbr` directly
and from the notebook's unpacked payload: **38,723 predictions, 0 differing,
max |Δ| = 0.0.** Packaging contributes exactly zero, so every prediction
difference is attributable to the audited source fixes and not to the build.

**Full test suite on the clean tree:** 623 passed, 1 skipped
(`test_visualization.py` — matplotlib absent), **0 failed**, single process.

---

## 7. Known-failure policy

Record: `KNOWN_FAILURE_STATUS.json`. Classification: **release-validation
provenance, not a runtime requirement.**

The production release does not run `known_failure_gate.py` and does not read
`research/known_failures.json` — not at build, not at load, not at inference.
Neither file has ever existed on this lineage. The gate's audited fix
(`TORCH_TESTS` widened to the CRF test files) is about a torch/LightGBM `libomp`
collision in test files production does not have; importing the gate would add
research infrastructure to production purely to carry an audit fix for
infrastructure production lacks. It was not imported.

What is carried forward is the evidence:

| | |
| --- | --- |
| old fingerprint sha256 | `0bd576da7bb7dadb0e7ab10f29aa5de822838de4e1d014c82aabbc237f127a73` |
| archived record | `research/archive/known_failures/known_failures_2026-08-22_0bd576da7bb7.json` (on the research lineage), content hash verified equal to the fingerprint |
| old counts | 15 failed, 650 passed, 1 skipped — all fifteen were stream-parity tests |
| reason for regeneration | the fifteen pinned failures were resolved by verified engineering corrections (`b41da11` lgamma single-sourcing, `03e4637` arm64 FMA and scalar-square parity); no test removed, xfailed or loosened, no tolerance or expected output changed |
| authorisation | repository owner, 2026-08-26 |
| new count | 0 known failures |

**Independently verified here.** All fifteen pinned node ids live in test files
that already exist on `production/rt600`, so they were re-run verbatim on this
branch without importing anything: **15 passed, 0 failed.** This branch needs no
two-process split, because it contains no test that imports torch — which is the
entire reason the research gate splits.

---

## 8. `RESULTS.csv`

Untouched. The production copy at `research/RESULTS.csv` is the one that was
already there. **No research `RESULTS.csv` import. No RT ID. No experiment row.
No research history merge.** This is a production engineering release.

---

## 9. The one prediction change — `OWNER_ACCEPTED_ENGINEERING_CORRECTION`

Record: `PREDICTION_CHANGE_ROOTCAUSE.json`,
`PREDICTION_CHANGE_VS_PRODUCTION.json`.

Reproduced independently on this branch against the pre-fix production artifact,
same 67-series battery:

| | |
| --- | --- |
| predictions compared | 38,723 |
| changed | **1** (0.0026 %) |
| max absolute delta | **5.580e-04** |
| same-t pair order flips | **0** across **953,123** pairs |

Root cause, confirmed here at the feature level on series 5509 — all 500 columns
× 908 rows compared, **1 differing cell of 454,000**:

| | |
| --- | --- |
| column | `m06_loc::loc_p_rsq_stab`, row 42 |
| old served value | `4.2146848e-08` |
| clean stream value | `0.0` |
| batch reference | `0.0` |
| training cache `cache/features/m06_loc.npy` | `0.0` |

The corrected value agrees with the batch reference **and** with the feature
cache the frozen model was fitted on; the old served value agreed with neither.
The change restores intended train/serve semantics.

Per the owner's directive: **no TS-AUC computed, no leaderboard comparison, no
tuning around this row, no attempt to preserve the old prediction.**

---

## 10. Engineering artifact vs clean artifact — prediction equivalence

Record: `PREDICTION_EQUIVALENCE.json`. This is a packaging/transplant check, not
a performance comparison.

Both artifacts were run through the identical 67-series battery from their own
unpacked payloads:

| | |
| --- | --- |
| artifact A | engineering, notebook `a32f638c…` (the audited release candidate, unmodified) |
| artifact B | clean lineage, notebook `5843458e…` |
| predictions compared | **38,723** |
| **predictions differing** | **0** |
| max absolute delta | **0.0** |
| same-t pair order flips | **0** of 953,123 |

**Functionally identical.** The clean transplant missed nothing and altered
nothing. No TS-AUC computed for either side.

---

## 11. Runtime

Record: `PACKAGED_RUNTIME.json`. Measured on the packaged clean artifact, 12
series across four size regimes, least-squares fit of marginal + fixed cost.

| | audit reference | clean release | |
| --- | --- | --- | --- |
| marginal | ~1.537 ms/point | **1.513 ms/point** | −1.6 % |
| fixed | ~128.9 ms/series | **127.94 ms/series** | −0.7 % |
| projected 10k | ~2.51 h | **2.472 h** | −1.5 % |
| harness peak RSS | ~620 MB | **613.2 MB** | −1.1 % |

Every figure is at or slightly better than the declared tolerance. No regression.

---

## 12. Crunch execution test

Record: `CRUNCH_TEST.json`, `CRUNCH_OUTPUT_CONTRACT.json`,
`crunch_test_linux.log`. **One** clean-lineage run.

Run in `research/docker/Dockerfile.linux-verify` — **production/rt600's own
documented harness**, kept for exactly this purpose in commit `7746205`.
`python:3.12-slim` built from `requirements.txt` alone plus `crunch-cli 11.11.0`,
which doubles as the end-to-end `requirements.txt` check that LB-002 failed.

*Why not macOS:* `production/rt600` ships `INFER_PARALLELISM = 4`, and the macOS
`crunch test` runner segfaults LightGBM under forked workers. The macOS run was
executed **first** and reproduced the documented SIGSEGV exactly
(`2 worker(s) died (exit codes: 0=-11, 1=-11)`); its log is kept beside this file
as `crunch_test_macos_p4_segfault.log`, evidence that this is the known
pre-existing macOS fork+OpenMP hazard recorded in
`research/reports/rt600_baseline_submission.md` §16.3 and not a release
regression. The branch's own policy is to settle parallelism claims in the
container, which is what was done.

| | LB-004 record | **clean release** |
| --- | --- | --- |
| exit | 0 | **0** |
| parallelism | 4 | **4** |
| worker deaths | none | **none** |
| determinism check | passed | **passed** (tol 1e-08, 10 % re-run) |
| duration | 00:00:58 | **00:00:49** |
| memory consumed | 203.04 MB | **202.85 MB** |
| prediction rows | 50,983 | **50,983** |

Payload self-check: *payload ok; source and model verified by sha256*.
Competition verified: 6 data files including `y_test_index.reduced` and
`y_train_index`, i.e. the real-time competition, not the 2025 one.

**Output contract:** 50,983 rows, index `(id, time)`, column `prediction`, 100
series, 0 NaN, 0 Inf, 0 outside [0,1], **0 duplicate index rows**, 0 missing.

**Reference run (not a second release test).** The pre-fix production artifact
(`74894ed1`) was run through the identical container immediately afterwards:
`prediction.parquet` is **bitwise identical** to the clean release run's, all
50,983 rows, md5 `6f38a1b8067425812fc932d3160f5115` for both. On the
competition's reduced test set the accepted `m06_loc` correction changes nothing
at all — the one changed prediction lives on training-store series 5509, which
is not in that set.

`crunch test` is a local execution test and returns no score. Nothing was
submitted, no model variants were tested, and no returned metric was used for
model selection — there was no score available to select on even in principle.

---

## 13. Gate summary

| gate | result |
| --- | --- |
| batch/stream parity (201 series, 7 modules, 51.5 M cells) | **PASS** — 0 |
| prefix invariance (7.26 M cells) | **PASS** — 0 |
| series independence (features + predictions) | **PASS** — 0 / 0 |
| deterministic replay (5 repeats) | **PASS** — 1 hash |
| artifact provenance (6 wrong variants) | **PASS** — 6/6 refused |
| output contract | **PASS** |
| packaging neutrality | **PASS** — bitwise |
| engineering vs clean prediction equivalence | **PASS** — 0 of 38,723 |
| runtime within declared tolerance | **PASS** |
| Crunch execution test | **PASS** — 00:00:49, determinism passed |
| model artifact unchanged | **PASS** — byte-identical |
| source content audit | **PASS** |
| full test suite | **PASS** — 623/0/1 |
| known-failure status | **PASS** — 15/15 pinned failures green |

**All clean-lineage gates pass → `APPROVED_FOR_PRODUCTION_MERGE`.**

---

## 14. Confirmations

- No research `RESULTS.csv` import — the production copy is untouched.
- No new RT ID, no experiment row, no score recorded.
- No TS-AUC computed anywhere in this release, for any artifact.
- No lockbox access and no model-search activity of any kind.
- No post-freeze research module imported — none was needed and none was added.
- No deletion commit for files that never existed on production.
- The source zip was **not** hand-filtered; the canonical production builder
  packed `src/sbr` whole and the clean tree defined the contents.
- Neither `production/rt600` nor `engineering/rt600-final-reliability-2026` was
  rewritten. `main` and `research/current` were not touched.
- No force push.
