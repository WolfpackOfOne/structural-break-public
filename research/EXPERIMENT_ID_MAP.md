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

---

## 6. WAVE-5 ALLOCATIONS (`research/wave5-alpha`, macOS/arm64, 2026-08-21)

`RT-7xx` and `RT-8xx` were verified unused on every branch and in every
reachable commit before allocation. `RT-6xx` is **reserved for the shipped
artifact** — RT-600 is the Crunch-tested submission (LB-001 = 0.6268) and
nothing in wave 5 may take an `RT-6xx` id.

### Objective arms — W5-E9 (protocol: CHAMP modules, 700k rows, seed 0)

| ID | objective | note |
|---|---|---|
| `RT-702` | `pairwise_t` | **CONTROL.** The incumbent objective routed through the wave-5 hook. Reproduces `RT-413`'s ledger row to five decimals on all five folds, which is what licenses the comparison |
| `RT-700` | `pairwise_w` | pairs weighted by `n_neg(t)` — REJECTED, −0.00147 |
| `RT-701` | `pairwise_h` | weighted squared hinge — REJECTED, −0.00476 |

### Curriculum arms — W5-E3 (protocol: CHAMP)

| ID | weighting | note |
|---|---|---|
| `RT-710` | uniform | **CONTROL.** A `wbinary` custom objective starts from raw score 0, not the label prior, so the control must be a `wbinary` run and **not** `RT-300` |
| `RT-711` | smooth reweighting | `w_neg = 1 + 3 r²` on nested fold-pure hardness |
| `RT-712` | oversampling | hardest 10% of negatives entered 4× in the training pool |

### Feature-block arms — W5-E2 / E4 / E5 / E6 / E7 / E10 (protocol: ABL 400k, arms identical to the wave-3 `m09_back` test)

| ID | block | verdict |
|---|---|---|
| `RT-730` | `+ m11_focus` | survives stage C, **+0.00055** vs the seed clone |
| `RT-740` | `+ m10_persist` | REJECTED, −0.00041 vs the seed clone |
| `RT-750` | `+ m12_rdep` | survives stage C, **+0.00141** vs the seed clone — the strongest |
| `RT-760` | `+ all three` | REJECTED, +0.00093 — worse than `m12_rdep` alone |

Controls reused from wave 3, not re-run: `RT-301` (ABL control) and `RT-303`
(its seed clone).

### Stage-D ensemble-stream arms (protocol: CHAMP, matched control `RT-401`)

| ID | is | verdict |
|---|---|---|
| `RT-731` | champion config + `m11_focus` | −0.00123 standalone; **sign flips** vs its ABL twin `RT-730` |
| `RT-751` | champion config + `m12_rdep` | +0.00121 standalone; member 1 of `S'` |

### W5-E11 architecture rebuild — every specialist stream + `m12_rdep`

Each is the incumbent configuration **verbatim** from
`research/scripts/wave2_streams.py` with one module appended and nothing else
changed.

| ID | rebuilds | which wave-2 stream |
|---|---|---|
| `RT-751` | `RT-300` | `RT-100R` (also the stage-D arm; member 1 of `S'`) |
| `RT-811` | `RT-410` | `RT-120R` |
| `RT-812` | `RT-411` | `RT-121R` |
| `RT-813` | `RT-412` | `RT-122R` |
| `RT-814` | `RT-413` | `RT-123R` |
| `RT-815` | `RT-414` | `RT-124R` |
| `RT-816` | `RT-415` | `RT-125R` |

`S'` is the equal-weight cross-fitted SCDF blend of those seven; `S` is the
incumbent seven (`RT-420`).

### Partition suffixes

`RT-301.alt1`, `RT-303.alt1`, `RT-750.alt1` (and `.alt2`) are the ABL arms
re-run on an alternate fold partition. **Robustness diagnostic only** — the
partitions select nothing, and only the DELTA is compared across them, never
the level (W4-E2: levels move ~0.009 across partitions, deltas ~0.003).

### Names that are NOT experiments

* **`W5-NULLTEST`** is a null test of the promotion battery itself — candidate
  `RT-402` against control `RT-401`, both seed clones. Its purpose was to
  measure what an eighth exchangeable member is worth (**+0.00003**), which is
  what motivated W5-E11. It is not a candidate and has no ledger row.
* **`W5-D1`..`W5-D4`** are diagnostics, not experiments: metric geometry,
  false-positive forensics, age profile, break-family mix. D2 and D4 use `tau`
  and post-break data and are **never** readable by production code.
