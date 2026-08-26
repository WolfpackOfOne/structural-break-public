# RT-600 deployment readiness

**Branch:** `engineering/rt600-final-reliability-2026`
**Base:** `81658764c95a7d91f3ca1f80ff25ad8c2b9b33e2` (CRF tip, `CRF_PROGRAM_EXHAUSTED`)
**Scope:** engineering only. No experiment ID consumed, no RESULTS.csv change, no
candidate score, no lockbox or test data read, no model search.

**STATUS: READY_FOR_OWNER_REVIEW** (was `NOT_READY`; superseded 2026-08-26).

The two gates below that required an owner decision were approved as
engineering-only release actions and are now done. The known-failure
fingerprint is regenerated and the shipped artifact is rebuilt, re-validated
and Crunch-tested. **See "Release candidate — RT-600-RC1" at the end of this
document, which supersedes gates 13 and 15 and corrects gate 12.**

Nothing is deployed and nothing is merged to `production/rt600`. The owner
still makes the merge decision.

The audit below is left as written, as the record of what was found before the
release actions were authorised.

---

## Gate matrix

| # | Gate | Verdict | Evidence |
|---|---|---|---|
| 1 | Scientific freeze | **PASS** | `CRF_PROGRAM_EXHAUSTED` at `8165876`. Model search stopped. RT-600 remains champion at external **0.6268**. The wider-learned-bottleneck hypothesis was **not** executed. |
| 2 | Batch/stream parity | **PASS** | All 7 frozen modules, 201 series, **53,203,000 cells, 0 mismatches** at `atol=0`. Was 4/5 seeds failing. `STREAM_PARITY_REPRO.md`, `FULL_PARITY.json`. |
| 3 | Prefix invariance | **PASS** | **7,261,000 cells, 0 mismatches** over 14 series x 14 boundary prefixes, in both modes (truncated stream, and batch rebuilt on the truncated online array). Plus 37 dedicated prefix/poison/`n_online` tests. |
| 4 | Series independence | **PASS** | Feature level: **1,146,000 cells, 0 mismatches**. Prediction level: 2,964 predictions re-run under reordering plus foreign-series interleaving, **0 differ**. |
| 5 | Deterministic replay | **PASS** | 5 repeats of the full inference path → 1 distinct output hash. Feature sweeps over 64 series give identical SHA-256 across 4 processes: idle ×2, `VECLIB_MAXIMUM_THREADS=1`, and under 8-way CPU load. BLAS is Accelerate; no thread-count sensitivity found. |
| 6 | Artifact provenance | **PASS (hardened)** | `ProductionModel._check_provenance` now pins fit population, partition, fold SHA, booster count and calibration. Six wrong-provenance variants that pass the old feature-manifest check are refused. `VOID_RUN_CHECKLIST.md`. |
| 7 | Environment reproducibility | **PASS** | Running environment matches `FINAL_REPRODUCIBILITY_MANIFEST.json` exactly: Python 3.11.6, numpy 2.4.6, pandas 3.0.5, scipy 1.17.1, sklearn 1.9.0, lightgbm 4.7.0, numba 0.67.0, macOS-26.5.2-arm64. |
| 8 | Production artifact reproducibility | **PASS** | `model_zip_sha256` = `6c8960dd…` reproduces bit-exactly. `source_zip_sha256` = `199db8c9…` reproduces bit-exactly from all three recorded commits (`73b56620`, `9ab674bf`, `9aaa9b04`). See the note on `41ab0695` below. |
| 9 | Runtime | **PASS** | 1.633 ms marginal per point + 129 ms fixed per series → **2.64 h** for 10,000 series at mean online 504, against a 15 h budget. Frozen record: 1.734 ms/pt. No regression from the fixes. |
| 10 | Memory | **PASS** | Peak RSS **525 MB** for the seven-booster ensemble. `crunch test` previously recorded 1.81 GB consumed. No leak across series. |
| 11 | Output contract | **PASS** | 13,615 predictions: 0 NaN, 0 inf, 0 outside [0,1], 0 row-count mismatches, range [0.0306, 0.9861]. Row count, ordering and series mapping correct. |
| 12 | OpenMP / library conflict | **PASS (defect fixed; see correction in §9 of the release-candidate section — the venv *does* contain torch)** | The documented torch+LightGBM `libomp` segfault is real and reproduced. The split-process gate that guards it **was broken** and is repaired — see gate 13. The production venv has no torch at all, which is the safest configuration. |
| 13 | Known-failure gate | **RESOLVED 2026-08-26** (was FAIL — owner decision) | All **15** pinned failures now pass. By the file's own policy a disappearance is an alert requiring a human. This is intended and evidenced, but it is not mine to sign off. Details below. |
| 14 | CI | **PASS (correctly does not run; re-checked 2026-08-26)** | Neither workflow triggers on `engineering/**`: `ci.yml` fires on `main` pushes and PRs; `research-hygiene.yml` on `research/**` **and** a change to `RESULTS.csv`, which is unchanged. No workflow was edited. Run `32984351317` (CRF, `9c5a352`) is **queued** — unrelated to this branch. |
| 15 | Submission package reproducibility | **RESOLVED 2026-08-26** (was BLOCKED — owner decision) | The shipped notebook embeds a *frozen source zip*. The deployed artifact therefore does **not** contain these fixes, and rebuilding changes `source_zip_sha256` and requires a fresh `crunch test`. Details below. |

---

## A. What was actually wrong

Four independent train/serve skews, all the same shape: the batch pipeline wrote
`cache/features`, the frozen model was trained on that, and production infers
through `StreamEngine` — which did not reproduce it.

| Module | Primitive | Mechanism |
|---|---|---|
| `m07_bayes` | `math.lgamma` | numba's and CPython's disagree by up to **512 ULP**. Used only to build the BOCPD Student-t constant table, computed independently on each side; 57/80 entries differed. It enters every run length's predictive likelihood, so the error persisted from `t=1` and never reconverged. |
| `m04_resid`, `m01_seq` | `scipy.signal.lfilter` | Contracts `b0*x + z` into a **fused multiply-add** on arm64 — one rounding where Python's `b0*x + z` rounds twice, disagreeing on ~20 % of steps. The EWMA/GARCH recursions are stateful, so it accumulated: `vol`/`volM`/`cmb` differed at 84/111/29 of 568 steps on series 7395. |
| `m06_loc`, `m02_dist` | `x ** 2` | A Python scalar `**2` goes through libm `pow`; numpy's array `**2` is a squaring multiply. They differ on **688/500,000** doubles. `m06`'s rolling variance is a cumsum difference that cancels to ~1e-14 on a near-constant window, so one ULP moved the emitted float32 by 15 %. |

Ruled out by controlled experiment, not assumption: `math.log`, `math.exp`,
`math.log1p` and FMA contraction under numba are all bitwise identical
(0/200,000 each); `np.cumsum` is bitwise identical to a sequential loop
(0/400 arrays); and `_BocpdStream` was proven a bitwise-exact re-expression of
the batch algorithm (`_bocpd.py_func` vs `_BocpdStream`: 0/6,400 cells).

**No tolerance was loosened.** No feature semantics, hazard constant,
calibration choice or tuned value changed. The batch path is bit-identical
before and after: `cache/features/m07_bayes.npy` still reproduces exactly, so
no refit is implied.

### Intended semantics — resolved without reference to any score

The fix direction was **not** chosen. It is fixed by the frozen artifact:

* the training cache matches the **batch** path bitwise (0 differing cells on
  every failing series) and disagreed with the stream;
* every streaming module's own docstring states its contract is to reproduce the
  batch module bitwise — the stream is a *port*, the batch is the reference;
* `tests/test_stream_parity_m07_bayes.py` states outright that the batch output
  "is the parity target";
* `FINAL_ARCHITECTURE_FREEZE.md` §2 freezes the bank the batch driver built.

No TS-AUC was computed for either variant, and none is needed.

### The counter-argument, stated fairly

`research/known_failures.json` records the opposite decision: *"repairing the
twin would change the values the STREAMING engine emits, i.e. frozen production
semantics and therefore RT-600's live predictions. RT-600 scored 0.6268 on the
leaderboard through this exact streaming path."*

That is a real risk-management position and it was right to record it. Two things
have changed since:

1. It was written believing the streaming rewrite was itself the defect. It is
   not — the divergence is a compiler/libm difference, and the stream was a
   faithful port that three specific primitives betrayed.
2. The risk it was protecting against is now **measured**. See §D.

---

## B. Prefix invariance

Prefixes `1, 2, 3, 5, 10, 16, 20, 32, 50, 64, 100, 128, 200, 256` over 14 series,
in two modes: truncating the stream, and rebuilding the **batch** modules on the
truncated online array. Both must reproduce the surviving rows bitwise.

**7,261,000 cells checked, 0 mismatches.** (`PREFIX_INDEPENDENCE.json`.)

The historical→online boundary is covered at `L = 1`, and every module's `t = 0`
row is inside the 53.2M-cell parity sweep. 37 dedicated tests also pass:
`test_future_poison_cannot_change_an_emitted_row`, `test_no_n_online_leakage`,
and each module's `test_prefix_poison`.

*Harness note.* An earlier run of this check reported a large mismatch count.
That was a defect in the **check**, not the code: the reference row block was
built as `StreamEngine().fit_historical(h).step(x) for x in o`, which constructs
a fresh engine per observation and so compares against 300 independent
first-steps rather than one streamed run. Corrected, it reports zero. Recorded
because a scary intermediate number that turned out to be the measuring
instrument is exactly the kind of thing this report should not quietly drop.

## C. Series independence

Feature level: 8 target series re-run in reverse order with 4 foreign series
streamed in between — **1,146,000 cells, 0 mismatches**. Prediction level: 2,964
predictions under reordering plus interleaving, **0 differ**.
`INFER_PARALLELISM = 1`, so no shared cross-series state exists in the deployed
path; independence is nevertheless asserted rather than assumed.

## D. Prediction impact — BUGFIX-PREDICTION-CHANGING

Classified per the audit brief and quantified **without** any TS-AUC.

Sample: 67 series — the 7 with known pre-fix feature divergence plus 60 drawn at
random. Base commit vs fixed, same frozen model artifact, same store.

| Quantity | Value |
|---|---|
| Rows compared | **37,795** |
| Rows whose prediction changed | **0** |
| Series with any change | **0** |
| Max absolute score difference | **0.0** |
| Mean absolute score difference | **0.0** |
| Cross-series pairs compared at matched `t` | **912,922** |
| Pair-order flips (the structure TS-AUC scores) | **0** |

The features change; the predictions do not. LightGBM bins at `max_bin = 127`,
and a 1-ULP float32 move essentially never crosses a bin boundary.

This is measured on 37,795 rows, not proven for all inputs — but it is the
strongest available evidence, and it is the number the `known_failures.json`
rationale was implicitly guessing at.

**No TS-AUC was computed for either version.** Which one scores better is
deliberately unknown, and must stay unknown: choosing on that basis would be
model selection under a freeze.

---

## E. Test results

Run with the full artifact environment, two processes (torch and LightGBM cannot
share one).

| Result | Count |
|---|---|
| Passed | **794** |
| Failed | **1** |
| Skipped | 5 |

Targeted suites: parity / contract / stream — **550 passed, 0 failed**.
Production contract — **88 passed**.

### The one failure

`tests/test_causal_representation_frontier_audit.py::test_results_csv_is_untouched_by_the_design_task`

**Pre-existing at the base commit `8165876`; unrelated to this branch.** Verified
by running it there. The CRF *design-task* audit pinned RESULTS.csv at
`5b34c564…`, and the CRF *execution* then legitimately appended its own result
rows, moving it to `8a0a8832…`. The pin was never updated. RESULTS.csv is
correct and untouched; the stale constant is CRF research bookkeeping and is left
for the owner rather than edited here.

### Two engineering defects found in the test infrastructure itself

1. **The split-process gate was broken.** It excluded only
   `tests/test_neural_causality.py`, but the CRF program added
   `test_crf01_causality.py` and `test_crf02_causality.py`, which import torch.
   Running the gate as documented **segfaulted inside `lightgbm/basic.py`**
   instead of reporting — reproduced this session. Fixed; `TORCH_TESTS` is now
   the complete tuple.
2. **The parity sample was too narrow.** The 15-series sample in
   `test_engine_parity_real` missed the `m04`/`m06`/`m02` defects entirely; they
   only appeared at 64+ series. The m01 and m06 cases were visible all along as
   pinned "known failures" on synthetic inputs.

---

## What blocks deployment

### BLOCKER 1 — the known-failure gate needs an owner decision

All 15 pinned failures now pass. The file's policy is explicit: *"A DISAPPEARANCE
IS AN ALERT, NOT A CELEBRATION… Any NEW failure, any DISAPPEARANCE, and any
RENAME is a gate failure requiring a human decision."*

`known_failures.json` has deliberately **not** been regenerated. Regenerating it
is exactly the act the policy forbids doing unilaterally, and leaving the gate
red is the honest state. The owner needs to decide whether to re-pin it to an
empty set on the strength of §A and §D.

### BLOCKER 2 — the shipped artifact does not contain these fixes

The submission notebook embeds a frozen zip of `src/sbr` built at
`production/rt600`. Editing `src/sbr` on this branch does not change the deployed
artifact. Deploying the corrected engine requires:

1. rebuilding the submission (`research/scripts/build_submission.py`), which
   changes `source_zip_sha256` away from `199db8c9…`;
2. a fresh `crunch test` including the 1e-8 determinism check;
3. updating `FINAL_REPRODUCIBILITY_MANIFEST.json` and the freeze record.

None of that was done here: it modifies production, and the brief forbids
touching production until the owner has reviewed these gates.

**Nothing was merged. The branch is pushed and clean.**

---

## Residual observations (recorded, not fixed)

* `ruff check .` reports **1,972** pre-existing errors repo-wide, unchanged by
  this branch (new files are clean). `ci.yml` runs `ruff check .` and would fail
  on `main` today. Not touched — out of scope, and fixing it by editing the
  workflow is forbidden.
* `41ab0695` (the manifest's `model_code_git_sha`) cannot reproduce
  `source_zip_sha256` **using its own `build_submission.py`**, because the
  deterministic-packer fix landed later and the old packer encoded checkout
  mtimes. `src/sbr` is git-identical between `41ab0695` and `73b56620`, and the
  current packer reproduces `199db8c9…` from that content. The frozen record is
  consistent; the caveat is about which packer you run.
* `research/scripts/crf01_nncsr.build_channels` validates its cache on row count
  and channel names only — integrity and shape, not provenance. CRF is a closed
  program and this is research code, so it is recorded rather than changed.
* There is no automatic runtime-plausibility assertion in the production path.
  Rule 3 of the void-run checklist is applied by hand.

---

# Release candidate — RT-600-RC1

**Date:** 2026-08-26 · **Branch:** `engineering/rt600-final-reliability-2026`
**Release commit:** `60b399e3b845de8ee14a94f0b23a33f06f2889ea`
**STATUS: READY_FOR_OWNER_REVIEW.** Not deployed. Not merged to `production/rt600`.

The two blockers above were owner decisions, not failures. The owner approved both
as engineering-only release actions, and this section records what was done and
what was measured. Nothing here consumed an RT id, wrote to `RESULTS.csv`,
computed a TS-AUC, tuned anything, or selected between implementations.

`research/RESULTS.csv` is byte-identical before and after:
`8a0a88324ea269cb3d6710c76a1e2bfc994d0bc562642a4bae901ba32be73e21`.

## Owner approvals

| # | Action | Authorisation |
|---|---|---|
| 1 | Regenerate the pinned known-failure fingerprint | owner, 2026-08-26, engineering only |
| 2 | Rebuild the shipped artifact and run a fresh Crunch execution test | owner, 2026-08-26, engineering only |

Both are recorded in the artifacts themselves, not only here: the fingerprint
carries a `regeneration.authorised_by` field, and
`FINAL_REPRODUCIBILITY_MANIFEST.json.release_candidate` carries `authorised_by`.

## 1. Known-failure fingerprint regenerated

| | |
|---|---|
| prior pinned failures | **15** |
| observed now | **0** — 795 passed, 0 failed, 5 skipped |
| old fingerprint sha256 | `0bd576da7bb7dadb0e7ab10f29aa5de822838de4e1d014c82aabbc237f127a73` |
| new fingerprint sha256 | `f6df37d1d3f63eda4fbc3a4408a7dab4f5fc83cb462e170dc576cc53342516e4` |
| old fingerprint archived to | `research/archive/known_failures/known_failures_2026-08-22_0bd576da7bb7.json` (byte-identical copy) |

The file's own policy — *a disappearance is an alert, not a celebration* — was
applied rather than waived. The gate was run first, from a clean invocation, and
reported all fifteen gone **before** anything was rewritten. The cause is the one
the policy text does not enumerate: the defect they pinned was repaired
(`b41da11`, `03e4637`). No test was removed, xfailed or renamed; no tolerance was
loosened; no expected output was edited to obtain green.

Provenance is preserved in four places, not overwritten: the verbatim archive
above, a `previous_fingerprint` block inside the new file (prior sha256, prior
counts, all fifteen node ids, prior policy text), a `resolved` block naming the
commit that fixed each pinned class, and the original `why_not_fixed` rationale
kept **unedited** in `CLASSES` so a reviewer can read the position that was
overturned. `--generate` now archives before writing, so this cannot be skipped
by a future regeneration.

Against an empty pinned set the gate is strictly stronger than before: every
failure is now a new failure.

## 2. CRF stale RESULTS-hash check — repaired (`a18ed73`)

**The issue.** `test_results_csv_is_untouched_by_the_design_task` pinned the
whole-file sha256 of `research/RESULTS.csv` at `b47b22ad` (`5b34c564…`) and
compared it against current HEAD.

**Which assertion was intended — resolved from the repository, not from prose.**
The audit script's own docstring says *"the CRF **design task** must leave this
untouched"*. The ledger's hash across the program:

| commit | RESULTS.csv sha256 | rows | what it was |
|---|---|---|---|
| `b47b22a` | `5b34c564…` | 245 | Second Sweep tip, design-task base |
| `b219baf` | `5b34c564…` | 245 | CRF audit — design task |
| `85d121f` | `5b34c564…` | 245 | CRF prereg — design task |
| `afba958` | `5b34c564…` | 245 | CRF-01 execution prereg |
| `b6f27e3` | `a21bc538…` | 247 | CRF-01 **evaluated** — rows appended |
| `9a5ecc0` | `cd99bf9a…` | 250 | CRF-02 evaluated |
| `9c5a352` | `8a0a8832…` | 253 | CRF-02 corrected run |

Interpretation **A**. The design task changed nothing and the invariant held; the
CRF *execution* then legitimately appended its own result rows. Nobody updated
the pin, so the assertion silently changed meaning into *"HEAD must forever equal
a pre-execution hash"* — which no correct repository state can satisfy.

**The fix.** The assertion is re-scoped to the historical invariant it was meant
to be: `RESULTS.csv` is **append-only** with respect to the design task, i.e. its
first 155,924 bytes must still hash to `5b34c564…`. That fails on any edit,
reorder or removal of a pre-CRF row — strictly stronger than the old check about
the thing the old check existed to protect — while permitting appends. It needs
no git and runs in any checkout. The `no CRF- experiment id` assertion is
unchanged, and the whole-file hash is still computed and reported as a fact
rather than as a gate.

**No RESULTS.csv row was touched and no CRF scientific history was rewritten.**

## 3. Rebuilt artifact

Built with the canonical builder. No file was assembled by hand.

```
python research/scripts/build_submission.py \
  --model <models/final10k_ensemble> \
  --out submissions/C_ensemble_deployable.ipynb \
  --title "Structural Break Real-Time: 7-stream deployable ensemble, all 10,000 series"
```

| item | value |
|---|---|
| source commit | `60b399e3b845de8ee14a94f0b23a33f06f2889ea` |
| embedded source clean | **yes** — payload is reachable at that commit |
| built at | 2026-08-26T18:28:02Z |
| **source_zip sha256** | `e5381d518faa51ccadb36898bfab987e788c8a3fcafb91920b9af8b779f83d6d` (was `199db8c9…`) |
| **model_zip sha256** | `6c8960ddc7331afecfc6980974f2879401a1eb583f15f5cad34c2ea03299ea8c` — **BYTE-IDENTICAL** |
| feature manifest sha256 | `1646c3b9e09d8a7fb3c564483a1d1d999680caeefe848b092f907a6cac80cced` — **unchanged** |
| model manifest sha256 | `1483a59a268ded18b6af804a5f0eb0b191376a1dc86526f286751bf1613ec940` — unchanged |
| notebook sha256 | `a32f638c3ad2483d8205789e370e828078639b9a4781fb5bc35699a19188b433` (28,104,109 B) |
| `.py` entrypoint sha256 | `42668f521ebcd7df0f05423dbeba5617e2217a39ee2f9dbabf23cbb92ea35ac2` (28,101,317 B) |

**The model artifacts did not change.** `model_zip_sha256` is byte-identical to
the frozen record, so the §6 stop condition did not trigger. The feature manifest
is also unchanged, which is why this model still loads at all:
`ProductionModel._check_manifest` refuses a model whose engine emits different
columns.

Environment of the build, matching `FINAL_REPRODUCIBILITY_MANIFEST.json`
exactly: Python 3.11.6, numpy 2.4.6, pandas 3.0.5, scipy 1.17.1, sklearn 1.9.0,
lightgbm 4.7.0, numba 0.67.0, pyarrow 25.0.1, macOS-26.5.2-arm64.

The notebook and `.py` are **not** bit-reproducible across builds — the boot cell
embeds a build timestamp. The zip hashes are the reproducible identities.

### What is in the payload — 44 files

**Changed (all eight are the audited fixes):** `sbr/features/m07_bayes.py`,
`sbr/production/model.py`, `sbr/stream/engine.py`, `sbr/stream/s_m01_seq.py`,
`sbr/stream/s_m02_dist.py`, `sbr/stream/s_m04_resid.py`,
`sbr/stream/s_m06_loc.py`, `sbr/stream/s_m07_bayes.py`.

**Added:** `sbr/stream/_fp.py` (the arm64 FMA emulation the parity fix needs),
plus `m10_persist`, `m11_focus`, `m12_rdep`, `m13_scale_survival`,
`m14_spectral_impulse`, `m15_ordinal_irrev`, `m16_joint_rarity`, `m17_observers`,
`m18_weighted_ctm` and `s_m12_rdep`.

**Removed:** nothing.

Every audited fix is confirmed present in the shipped zip:

| fix | where |
|---|---|
| BOCPD constant-table single sourcing | `m07_bayes._bocpd_ct`, imported by `s_m07_bayes._BocpdStream` |
| arm64 fused-multiply-add parity | `stream/_fp.fma`, used by `s_m01_seq` and `s_m04_resid` |
| scalar-square parity | `s_m02_dist`, `s_m06_loc` |
| production provenance hardening | `production/model.py` `_check_provenance` |
| split-process known-failure gate repair | `research/scripts/known_failure_gate.py` (repo-side, not payload) |

**On the ten extra feature modules — stated plainly.** They are not part of the
reliability work. They arrived in `src/sbr` with the branch base (`8165876`, the
CRF tip) and the canonical builder packs `src/sbr` whole. Filtering it by hand is
exactly the class of error the builder exists to prevent, and the brief forbids
hand-assembly, so they are shipped. They do not change inference:
`PRODUCTION_MODULES` is still the frozen seven, `MODULE_ORDER` was appended to
rather than reordered (asserted by `test_rt600_manifest_sha_is_unchanged`), and
the RT-600 manifest sha is unchanged. Only `m12_rdep`/`s_m12_rdep` are even
imported at run time, via `sbr.stream.engine`, and neither is instantiated. None
imports torch or any dependency not already required. Cost is import time and
~90 KB of payload.

### What is NOT in the payload — audited, not assumed

The zip was scanned member by member:

| checked for | found |
|---|---|
| CRF training code as a runtime dependency | **none** — CRF code lives in `research/scripts/`, which is not packed |
| `import torch` anywhere | **none** |
| research caches, `.npy`, `.npz`, parquet | **none** — the zip holds 43 `.py` files and `stream/CONTRACT.md`, nothing else |
| `__pycache__` / `.pyc` / `.pyo` | **none** |
| unit-test checkpoints | **none** — the only `tests/` matches are docstrings citing test names |
| experimental model artifacts (`.pkl`, `.joblib`, `.pt`, `.ckpt`) | **none** |
| local absolute paths | 4 hits, all `/home/claude/sb` **defaults behind `SBR_*` env overrides** in `pipeline.py`, `store.py`, `features/driver.py` — training-side helpers, unchanged from the shipped zip, inert at inference |

## 4. Gates re-run FROM the rebuilt artifact

Every number below was measured by executing the notebook's own code cells in a
scratch directory with this repository removed from `sys.path`, asserting that
`import sbr` resolves inside the unpacked payload, and running the gate against
*that*. Harness: `release_candidate/packaged_release_check.py`.

| gate | population | result |
|---|---|---|
| **Batch/stream parity** | 7 frozen modules, **201 series** | **51,460,500 cells, 0 mismatches** at `atol=0` |
| **Prefix invariance** | 14 series × 14 boundary prefixes, both modes | **7,261,000 cells, 0 mismatches** |
| **Series independence (features)** | 8 series reordered + 4 foreign interleaved | **1,146,000 cells, 0 mismatches** |
| **Series independence (predictions)** | 8 series, reordered + interleaved | **2,292 predictions, 0 differ** |
| **Deterministic replay** | 5 repeats of the full inference path | **1 distinct output hash** (`0b58afb014c0ab23`) |
| **Order independence** | full shuffle via the Crunch-contract harness | **PASS** |
| **Future poisoning** | tail rewritten past the cut | **PASS** |
| **Provenance** | payload sha256 self-check + `ProductionModel.load` | **PASS** |
| **Output contract (local)** | 30 series, 12,467 predictions | 0 NaN, 0 inf, 0 out of `[0,1]`, row counts exact |

Per-module parity, 201 series: `m00_core` 15,541,071 · `m01_seq` 6,175,260 ·
`m02_dist` 6,072,339 · `m03_dyn` 6,175,260 · `m04_resid` 6,175,260 ·
`m06_loc` 6,175,260 · `m07_bayes` 5,146,050 — **0 mismatches in every one.**

The 201-series population strictly contains the audit's 64-series sample, plus
137 more drawn deterministically (`rng(0)`) from the rest of the store. The cell
count differs slightly from the audit's quoted 53,203,000 because the extension
sample is not the same one; the mismatch count is the load-bearing number and it
is 0.

*Note on `FULL_PARITY.json`.* That file records the **discovery** run —
64 series, 15,120,500 cells, `total_bad: 2` (`m04_resid` series 7395
`vol_e_abs`, `m06_loc` series 2575 `loc_h_rsq_stab`) — and its
`prefix_invariance` block is the **defective harness** the audit describes in
§B, reporting 15,536,597 mismatches. It was committed at `3e7a9a2`, after the
fixes, as evidence of what was wrong. It is not a post-fix measurement and should
not be read as one. The post-fix numbers are the table above and
`PREFIX_INDEPENDENCE.json`.

## 5. Prediction equivalence — one change, root-caused

Same development-only battery, now shipped-artifact vs rebuilt-artifact, 67
series (the 6 with evidenced pre-fix feature divergence — 2575, 4746, 5002, 5111,
7395, 8581 — plus 61 drawn at random). **No TS-AUC was computed for either
side.**

| quantity | value |
|---|---|
| predictions compared | **38,723** |
| predictions changed | **1** (0.0026 %) |
| max absolute prediction delta | **5.580e-04** |
| mean absolute prediction delta | 1.441e-08 |
| series with any change | **1** |
| same-`t` cross-series pairs compared | 953,123 |
| **same-`t` pair-order flips** | **0** |

This is nonzero where the audit measured zero, so per §9 it was investigated
before proceeding.

**Packaging is ruled out.** The same 67 series were run twice — once from
`src/sbr` directly, once from the notebook's unpacked payload —
**38,723 predictions, bitwise identical, 0 differing.** The build contributes
exactly nothing.

**The change is the bugfix, and it moves toward the trained-on semantics.**
Series 5509 streamed through the shipped-era source tree (`73b56620`) and through
HEAD, all 500 columns × 908 rows:

| | |
|---|---|
| cells compared | 454,000 |
| cells differing | **1** |
| column | `m06_loc::loc_p_rsq_stab`, row 42 |
| shipped stream value | `4.2146848e-08` |
| rebuilt stream value | `0.0` |
| **batch reference** | **`0.0`** |
| **`cache/features/m06_loc.npy` — the array RT-600 was fitted on** | **`0.0`** |

No column outside the audited fix set differs. This is the `m06_loc`
scalar-square defect exactly as root-caused: `m06`'s rolling variance is a cumsum
difference that very nearly cancels, so one ULP is visible in the emitted
float32. The prediction moves `0.6489955357142857 → 0.6484375` — one feature cell
crossing one LightGBM bin boundary. The rebuilt artifact agrees with the batch
reference **and** with the training cache; the shipped artifact does not.

**This revises the audit's headline.** *"Rows whose prediction changed: 0"* was
true of the audit's sample and is not a general claim. The rate here is 1 in
38,723. The mechanism the audit gave still holds — LightGBM bins at
`max_bin = 127` and a 1-ULP float32 move almost never crosses a boundary — but
*almost never* is not *never*, and this is what it looks like when it does. The
classification `BUGFIX-PREDICTION-CHANGING` was correct and remains correct.

Which version scores better is deliberately unknown and must stay unknown.

## 6. Runtime — tolerance stated before interpretation

**Tolerance declared in advance:** a regression is unacceptable if marginal cost
or the 10k projection exceeds the audited value by more than **+20 %**, or if
peak RSS exceeds it by more than **+25 %**. Anything inside those bands is noise
on a shared laptop.

| quantity | audited | rebuilt artifact | Δ | verdict |
|---|---|---|---|---|
| marginal ms / online point | 1.633 | **1.537** | −5.9 % | **faster** |
| fixed ms / series | 129 | **128.9** | −0.05 % | unchanged |
| projected 10,000 series | 2.64 h | **2.51 h** | −4.9 % | **faster**, against a 15 h budget |
| peak RSS | 525 MB | **619.7 MB** | +18.0 % | within tolerance |

Per-bucket, packaged: `hist~1000/online~1000` 1.578 ms/pt · `hist~5000/online~1000`
1.749 · `hist~3000/online~500` 1.840 · `short_online_10_20` 8.719 (fixed cost
dominating 17 points, as before).

The RSS rise is within tolerance and is mostly the measuring harness: a packaged
run holds the 28 MB notebook JSON plus the base64 source and model strings
resident, which a source-tree run does not. The deployment-relevant memory number
is the Crunch runner's own, below, and it went **down**.

## 7. Fresh Crunch execution test — PASSED

One release candidate, one run. Not a research evaluation: no variant
submissions, no model comparison, nothing tuned against any result.

```
cd <structural-break-claude-wave3>   # the worktree holding .crunchdao, data/ and
                                     # the correct project.json
env SBR_ROOT="$PWD" <.venv>/bin/crunch test \
  --main-file <rt600-reliability>/submissions/C_ensemble_deployable.py \
  --model-directory <scratch>/crunch_resources
```

| | |
|---|---|
| CLI | 11.11.0 |
| result | **PASSED** |
| duration | **00:01:47** (freeze record 00:01:40) |
| `infer` wall | 00:01:34 |
| determinism re-run, tolerance 1e-08 | **passed** |
| `INFER_PARALLELISM` | 1 |
| memory | 278.43 MB → 2.55 GB, **consumed 2.27 GB** (freeze record 2.55 GB) |
| payload self-check | `payload ok; source and model verified by sha256` |
| environment-specific failures | **none** |
| score returned | **none** — `crunch test` is a local execution test and returns no score. Nothing was submitted; there was no number available to select on even in principle. |

**Right competition, verified by the documented tell.** Six data files were
validated, including `y_test_index.reduced.parquet` and `y_train_index.parquet`,
which are real-time only. A run that lists four files was run against the 2025
competition, whatever it says about passing.

No reduced test data was inspected outside the official runner's own mechanical
access.

## 8. Output contract on the real deployment output

`prediction/prediction.parquet` as produced by the Crunch runner:

| check | result |
|---|---|
| rows produced | **50,983** |
| rows expected (`period == 2`, the online segment of `X_test.reduced`) | **50,983** |
| key set `(id, time)` identical to expected | **yes** |
| series/timestep mapping and ordering identical | **yes** |
| missing rows | **0** |
| extra rows | **0** |
| duplicate keys | **0** |
| series | 100 of 100 |
| NaN / inf | **0 / 0** |
| outside `[0, 1]` | **0** |
| range | `[0.00390625, 0.99609375]` |

## 9. Environment and the OpenMP hazard

Running environment matches `FINAL_REPRODUCIBILITY_MANIFEST.json` field for
field: Python 3.11.6, numpy 2.4.6, pandas 3.0.5, scipy 1.17.1, sklearn 1.9.0,
lightgbm 4.7.0, numba 0.67.0, pyarrow 25.0.1, macOS-26.5.2-arm64.

**Correction to gate 12 above.** That gate states *"the production venv has no
torch at all, which is the safest configuration."* That is no longer true, and
may not have been checked at the time: the frozen venv contains **torch 2.13.0**
at `.venv/lib/python3.11/site-packages/torch`. The mitigation therefore matters
rather than being belt-and-braces, and it works — the full suite ran twice today
through the corrected split-process path, 795 tests, with no segfault. The gate
script raises on `Fatal Python error` / `Segmentation fault` rather than parsing
a crash as a failure, so a broken split would be loud. `KMP_DUPLICATE_LIB_OK` was
not set. Note this is the *research* environment; the deployed Crunch runner's
environment is the organisers' and contains no torch.

## 10. CI

| | |
|---|---|
| `ci.yml` | triggers on `main` pushes and pull requests. This branch has no open PR, so it correctly does not run. |
| `research-hygiene.yml` | triggers on `research/**` **and** a change to `research/RESULTS.csv`. Neither applies. |
| workflow YAML edited | **no** |
| runs on this branch | **none**, as expected |
| required checks run locally instead | known-failure gate (795 passed, 0 failed), `check_research_hygiene.py` (**OK: 253 experiment rows, no duplicate IDs**), CRF audit script (**AUDIT: PASS**) |

**Run `32984351317`** (Research hygiene, CRF branch, `9c5a352`): **still
`queued`**, created 2026-08-26T15:11:58Z, not updated since, no conclusion. It is
**not** successful and is not described as such. It is unrelated to this branch
and to this release. The identical committed check passes locally, so it is not
treated as a blocker for the RT-600 engineering release — but it remains an open
item on the CRF branch.

## Remaining blockers

**None for the engineering release.** Every mandatory gate passes.

Open for the owner, none of which this task is authorised to close:

1. **The production merge itself.** `production/rt600`, `research/current` and
   `main` are untouched. No force push, no rebase of published history.
2. **The one prediction change.** Real, measured, root-caused to the audited fix
   and to the trained-on semantics, with 0 pair-order flips. It is the owner's
   call whether shipping a bit-for-bit different prediction stream needs anything
   beyond this evidence.
3. **The ten extra feature modules in the payload.** Inert and audited, but they
   ship. Removing them means changing `src/sbr` on the release commit, not
   filtering the builder.
4. **GitHub Actions run `32984351317`** is stuck queued on the CRF branch.
5. Everything under *Residual observations* below is unchanged by this work.
