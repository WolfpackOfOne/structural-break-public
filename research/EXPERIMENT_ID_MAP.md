# EXPERIMENT ID MAP

**Purpose.** The `RT-*` namespace was allocated by three agents on three branches
over three waves, and two informal names in the handoff documents do not resolve
to the rows they appear to name. This file is the single place that says which
string means which experiment. It maps; it does **not** rename. Historical rows in
`research/RESULTS.csv` keep the IDs they were written with, forever.

Written 2026-08-20 on `research/wave3-integration` (`c6da270`), after searching
`research/RESULTS.csv` on every branch (`main`, `research/wave2-2026`,
`codex/wave3-engineering`, `research/multi-agent-2026`,
`claude/structural-break-competition-entry-yjoj8r`), every file under `research/`,
and every commit reachable from any ref.

---

## 1. THE COLLISION THAT MATTERED

| informal name | what people meant by it | what `RESULTS.csv` actually holds under that string |
|---|---|---|
| "RT-150" | the deployable seven-stream ensemble, 0.62589 | `RT-150_f1` — a **rejected DGP-gating** experiment, 0.60619, present identically on `research/wave2-2026` **and** `research/multi-agent-2026` |

The deployable ensemble had **no ledger row at all**. Its number lived only in
`research/reports/deployable_ensemble_v2.json` and in prose in
`STATE_OF_RESEARCH_V2.md` §B1.5. Two of its five reported fold scores appear
nowhere else. A system that has been through an official `crunch test` and is the
project's headline result cannot be identified by a string that resolves to a
rejected experiment.

**Resolution: the deployable seven-stream ensemble is now `RT-250`.**
`RT-250` was verified unused on every branch and in every reachable commit before
allocation. `RT-150_f1` is untouched and still means the rejected gating run.

---

## 2. CANONICAL IDS

| canonical ID | system | status |
|---|---|---|
| `RT-100` | single champion, 500 cols, 7 modules, LightGBM binary, 1M rows — Linux, wave 1 | CONFIRMATION, 0.615103 |
| `RT-100R` | byte-for-byte reproduction of `RT-100` from a clean checkout — Linux, wave 2 | CONFIRMATION, 0.615103, delta exactly 0.0 |
| `RT-300` | the same configuration on macOS/arm64 — wave 3 | CONFIRMATION, 0.61605 |
| `RT-131` | within-timestep cross-sectional rank average of 7 streams | **ORACLE / ILLEGAL**, 0.62541 — not deployable, never a champion |
| **`RT-250`** | **7 streams + frozen cross-fitted smooth time-conditional CDF, equal average** | **DEPLOYABLE**, 0.62589 — the Crunch-tested system |
| `RT-150_f1` | DGP-gated blend | REJECTED, 0.60619 — **not** the ensemble, despite the name |

### `RT-250` provenance

* **Score** 0.6258948544909153, mean over the five canonical folds.
* **Per fold** 0.63806 / 0.62066 / 0.63439 / 0.61685 / 0.61952.
* **Streams** `RT-100R`, `RT-120R`, `RT-121R`, `RT-122R`, `RT-123R`, `RT-124R`, `RT-125R`.
* **Calibration** `scdf` — `SmoothTimeCDFCal`, 12 log-spaced anchors, 256-point
  quantile grids, cross-fitted so fold *k*'s map is fitted on folds != *k*.
* **Selection** the calibration *family* was chosen by nested CV, which picked
  `scdf` on all five outer folds; 0.62589 is therefore the nested-confirmation
  number, not a best-of-five.
* **Paired vs `RT-100`** +0.010567, series bootstrap 95% CI [+0.00763, +0.01320],
  200/200 replicates positive.
* **Evidence on disk** `research/reports/deployable_ensemble_v2.json`.
* **Platform** Linux/x86-64, wave 2. Not directly comparable to a macOS number.
* **Built artifact** `models/rt150_ensemble` (name retained — it is what the
  Crunch-tested tarball is called), tested in `research/reports/crunch_test_rt150.md`.
* **Known defect** its OOF stream vectors did not survive the container that
  produced them, so before wave 4 the number could not be recomputed from
  anything in the repository. See §4.

---

## 3. WAVE-4 ALLOCATIONS (this session, macOS/arm64)

`RT-4xx` was verified unused on every branch before allocation.

| ID | is | alias of |
|---|---|---|
| `RT-300` | champion config, seed 0, CHAMP protocol | `RT-100R` — **member 1 of both wave-4 arms** |
| `RT-401` | seed clone, seed 1 | — |
| `RT-402` | seed clone, seed 7 | — |
| `RT-403` | seed clone, seed 42 | — |
| `RT-404` | seed clone, seed 2026 | — |
| `RT-405` | seed clone, seed 31415 | — |
| `RT-406` | seed clone, seed 271828 | — |
| `RT-410` | specialist stream A, macOS reconstruction | `RT-120R` |
| `RT-411` | specialist stream B, macOS reconstruction | `RT-121R` |
| `RT-412` | specialist stream C, macOS reconstruction | `RT-122R` |
| `RT-413` | specialist stream D, macOS reconstruction | `RT-123R` |
| `RT-414` | specialist stream E, macOS reconstruction | `RT-124R` |
| `RT-415` | specialist stream F, macOS reconstruction | `RT-125R` |

Derived (computed from OOF vectors, no new training):

| ID | is |
|---|---|
| `RT-420` | SPECIALIST seven-stream SCDF ensemble, macOS |
| `RT-421` | SEED-CLONE seven-model SCDF ensemble, macOS |

---

## 4. THE `R`-SUFFIX TRAP — READ BEFORE COMPARING ANY `RT-1xxR` TO `RT-1xx`

`research/scripts/wave2_streams.py` states it in its own docstring and it is easy
to miss: **`RT-120R`..`RT-125R` are not reproductions of `RT-120`..`RT-125`.**
They differ in `n_estimators`, `learning_rate`, `min_data_in_leaf`, `lambda_l2`
and `max_bin`. Only `RT-100R` and `RT-123R` carry their wave-1 configuration.

| wave-1 | wave-2 `R` | same config? | wave-1 score | wave-2 `R` score |
|---|---|---|---|---|
| `RT-100` | `RT-100R` | **yes** | 0.615103 | 0.615103 |
| `RT-120` | `RT-120R` | no | 0.608839 | 0.605595 |
| `RT-121` | `RT-121R` | no | 0.604881 | 0.607812 |
| `RT-122` | `RT-122R` | no | 0.613155 | 0.614083 |
| `RT-123` | `RT-123R` | **yes** | 0.614499 | 0.614499 |
| `RT-124` | `RT-124R` | no | 0.610868 | 0.611651 |
| `RT-125` | `RT-125R` | no | 0.613558 | 0.617411 |

So a claim of the form "the deployable calibration recovers 99.7% of `RT-131`'s
oracle gain" is measured over the **`R` streams**, not over the original
`RT-120`..`RT-125` that `RT-131` itself was built from. That number is a
statement about the wave-2 stream set. See `research/reports/rt131_original_audit.md`.

---

## 5. OTHER NAMES THAT DO NOT MEAN WHAT THEY LOOK LIKE

* **`RT-160` / `RT-170`–`RT-172` / `RT-180`–`RT-182` / `RT-190`** exist **only** on
  `research/multi-agent-2026`. `RT-160` = 0.62544 is a logit-average blend and is
  the genuine independent deployable corroborator of `RT-250`.
* **The "independent re-measure, 0.62524"** cited in `HANDOFF_WAVE3.md` §1 is
  `RT-131`'s pooled OOF — the **illegal** within-timestep rank average. It does
  not corroborate a deployable number. `RT-160` does.
* **`RT-2410`, `RT-2411`, `RT-2412`, `RT-2420`, `RT-2421`** are four-digit IDs
  (random-column-drop controls), not `RT-241`/`RT-242` with a suffix.
* **`m08_chan`** is named in `HANDOFF_WAVE3.md` §5 as a completed rejected
  experiment. It exists in no commit on any branch and its `RT-30x` IDs were
  unallocated; wave 3 reused them for `m09_back`. Anything attributed to
  `m08_chan` is unsourced.
