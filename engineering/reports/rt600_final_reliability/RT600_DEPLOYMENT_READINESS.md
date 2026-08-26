# RT-600 deployment readiness

**Branch:** `engineering/rt600-final-reliability-2026`
**Base:** `81658764c95a7d91f3ca1f80ff25ad8c2b9b33e2` (CRF tip, `CRF_PROGRAM_EXHAUSTED`)
**Scope:** engineering only. No experiment ID consumed, no RESULTS.csv change, no
candidate score, no lockbox or test data read, no model search.

**STATUS: NOT_READY — two gates require an owner decision before deployment.**

The engineering work is complete and every measured gate passes. What blocks
deployment is not a failure; it is that two decisions are the owner's to make and
this audit is not authorised to make them. They are stated in full at the bottom.

---

## Gate matrix

| # | Gate | Verdict | Evidence |
|---|---|---|---|
| 1 | Scientific freeze | **PASS** | `CRF_PROGRAM_EXHAUSTED` at `8165876`. Model search stopped. RT-600 remains champion at external **0.6268**. The wider-learned-bottleneck hypothesis was **not** executed. |
| 2 | Batch/stream parity | **PASS** | All 7 frozen modules, 201 series, **53,203,000 cells, 0 mismatches** at `atol=0`. Was 4/5 seeds failing. `STREAM_PARITY_REPRO.md`, `FULL_PARITY.json`. |
| 3 | Prefix invariance | **PASS** | See §B below — streamed row `t` is unchanged by how much data follows, and a batch rebuild on the truncated online segment reproduces the surviving rows bitwise. |
| 4 | Series independence | **PASS** | Feature level (§C) and prediction level: 2,964 predictions re-run under reordering plus foreign-series interleaving, **0 differ**. |
| 5 | Deterministic replay | **PASS** | 5 repeats of the full inference path → 1 distinct output hash. Feature sweeps over 64 series give identical SHA-256 across 4 processes: idle ×2, `VECLIB_MAXIMUM_THREADS=1`, and under 8-way CPU load. BLAS is Accelerate; no thread-count sensitivity found. |
| 6 | Artifact provenance | **PASS (hardened)** | `ProductionModel._check_provenance` now pins fit population, partition, fold SHA, booster count and calibration. Six wrong-provenance variants that pass the old feature-manifest check are refused. `VOID_RUN_CHECKLIST.md`. |
| 7 | Environment reproducibility | **PASS** | Running environment matches `FINAL_REPRODUCIBILITY_MANIFEST.json` exactly: Python 3.11.6, numpy 2.4.6, pandas 3.0.5, scipy 1.17.1, sklearn 1.9.0, lightgbm 4.7.0, numba 0.67.0, macOS-26.5.2-arm64. |
| 8 | Production artifact reproducibility | **PASS** | `model_zip_sha256` = `6c8960dd…` reproduces bit-exactly. `source_zip_sha256` = `199db8c9…` reproduces bit-exactly from all three recorded commits (`73b56620`, `9ab674bf`, `9aaa9b04`). See the note on `41ab0695` below. |
| 9 | Runtime | **PASS** | 1.633 ms marginal per point + 129 ms fixed per series → **2.64 h** for 10,000 series at mean online 504, against a 15 h budget. Frozen record: 1.734 ms/pt. No regression from the fixes. |
| 10 | Memory | **PASS** | Peak RSS **525 MB** for the seven-booster ensemble. `crunch test` previously recorded 1.81 GB consumed. No leak across series. |
| 11 | Output contract | **PASS** | 13,615 predictions: 0 NaN, 0 inf, 0 outside [0,1], 0 row-count mismatches, range [0.0306, 0.9861]. Row count, ordering and series mapping correct. |
| 12 | OpenMP / library conflict | **PASS (defect fixed)** | The documented torch+LightGBM `libomp` segfault is real and reproduced. The split-process gate that guards it **was broken** and is repaired — see gate 13. The production venv has no torch at all, which is the safest configuration. |
| 13 | Known-failure gate | **FAIL — OWNER DECISION** | All **15** pinned failures now pass. By the file's own policy a disappearance is an alert requiring a human. This is intended and evidenced, but it is not mine to sign off. Details below. |
| 14 | CI | **PASS (correctly does not run)** | Neither workflow triggers on `engineering/**`: `ci.yml` fires on `main` pushes and PRs; `research-hygiene.yml` on `research/**` **and** a change to `RESULTS.csv`, which is unchanged. No workflow was edited. Run `32984351317` (CRF, `9c5a352`) is **queued** — unrelated to this branch. |
| 15 | Submission package reproducibility | **BLOCKED — OWNER DECISION** | The shipped notebook embeds a *frozen source zip*. The deployed artifact therefore does **not** contain these fixes, and rebuilding changes `source_zip_sha256` and requires a fresh `crunch test`. Details below. |

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

Prefixes `1, 2, 3, 5, 10, 16, 20, 32, 50, 64, 100, 128, 200, 256, 500, 512, 1000`,
in two modes: truncating the stream, and rebuilding the **batch** module on the
truncated online array. Both must reproduce the surviving rows bitwise.

Results in `FULL_PARITY.json` (`prefix_invariance`). The historical→online
boundary is covered at `L = 1`, and every module's `t = 0` row is inside the
53.2M-cell parity sweep. `test_future_poison_cannot_change_an_emitted_row` and
`test_no_n_online_leakage` both pass.

## C. Series independence

Feature level: the same target series re-run under three orderings with foreign
series interleaved before and after. Prediction level: 2,964 predictions, 0
differ. `INFER_PARALLELISM = 1`, so no shared cross-series state exists in the
deployed path; independence is nevertheless asserted rather than assumed.

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
