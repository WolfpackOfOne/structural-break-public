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

---

## Wave 6

### `RT-900` — **VOID**, and not reusable

| ID | is | verdict |
|---|---|---|
| `RT-900` | champion config + the `w6oracle` true-τ block | **VOID — LABEL LEAK VIA THE MISSINGNESS MASK.** 0.86552 TS-AUC is not alpha; the block is `NaN` exactly when `t < cut`, and for a break series `cut = tau`, so the mask *is* `y[t]`. The bare indicator `1[t>=cut]` scores 0.81442 alone. Never in a comparison table, an ensemble, a promotion decision, feature selection, production or a submission. See `research/FAILED_EXPERIMENTS.md`. |

**The ID is retired.** It is not reused, reassigned or recycled. `w6oracle` was
never a registered module and never reachable from `load_all()`.

### `RT-940`–`RT-944` — W6-E2R, the corrected **series-level** oracle diagnostic

| ID | arm | representation |
|---|---|---|
| `RT-940` | `A_rich` | the oracle-frontier study's generic known-boundary bank — **the reproduction control**, target 0.6497 |
| `RT-941` | `A_basic` | the same study's basic bank, target 0.6418 |
| `RT-942` | `B_causal` | **our 500 production columns at the boundary split** — the candidate representation |
| `RT-943` | `C_nobound` | our 500 columns at the end of the series, no boundary — the matched representation control |
| `RT-944` | `AB` | `A_rich ++ B_causal`, complementarity |

**These have NO `research/RESULTS.csv` row, deliberately.** `RESULTS.csv` is the
row-level TS-AUC ledger; a series-ROC-AUC number sitting in it is exactly how a
diagnostic gets mistaken for alpha six weeks later. They live in
`research/reports/wave6_corrected_oracle.{md,json,csv}`.
Runner: `research/scripts/wave6_e2r.py`. Pre-registration:
`research/WAVE6_PREREG.md` §18.

### `RT-960` / `RT-970` / `RT-980` — the neural track

| ID | is |
|---|---|
| `RT-960` | W6-N1 — compact MLP on the existing 500 causal features (**learner** capacity) |
| `RT-970` | W6-N2 — compact causal dilated TCN on compact legal channels (**representation** learning) |
| `RT-980` | W6-N3 — GRU, **only** if N1/N2 justify escalation |

Pre-registration: `research/WAVE6_NEURAL_PREREG.md` and `WAVE6_PREREG.md` §20.

### IDs deliberately NOT allocated

`RT-901`–`RT-908`. `RT-903`, `RT-906`, `RT-907` and `RT-908` already name
2025-reproduction *ideas* throughout `research/WAVE5_PREREG.md` and
`research/STATE_OF_RESEARCH_V5.md`; re-using them as experiment IDs would
collide in text even though it would not collide in the ledger.

## 7. WAVE 7

### `W7-D0` — no new ID allocated

The exact pairwise-inversion loss cube re-scores `RT-420` (`research/oof/wave5_S_specialist.npy`,
the existing RT-600 development architecture OOF) in different row slices. It
trains nothing and produces no new prediction vector, so it does not receive
an `RT-*` ID. See `research/RDOF_LEDGER.md` "WAVE 7" and
`research/WAVE7_RT600_EXACT_ALPHA_BUDGET.md`.

### `RT-990` / `RT-991` — W7-D3R, Arms B and C

| ID | is |
|---|---|
| `RT-990` | W7-D3R Arm B — `RT-300`'s 500 legal causal columns/rows, stronger LightGBM capacity only (127 leaves, ff 1.0, 900 trees). CAUSAL, but never promoted — diagnostic only. |
| `RT-991` | W7-D3R Arm C — Arm B's capacity plus 500 columns holding each row's own series' feature vector at that series' FINAL online row. **NOT CAUSAL. OFFLINE DIAGNOSTIC ONLY. NEVER a production candidate.** |

Arm A reuses `RT-300` (no new ID). Pre-registration:
`research/WAVE7_D3R_PREREG.md`. Result: `research/reports/wave7_d3r.{md,json}`.

### `RT-992` / `RT-993` — W7 teacher/distillation, one-fold pilot (T1, T2)

> **OUTER-FOLD CONTAMINATED. MECHANISM-POSITIVE SCREEN, NOT PROMOTION
> EVIDENCE.** `Q` (global `RT-991` OOF) is cross-fitted per-row/series but
> not per-*outer-validation-fold*: this pilot's outer fold-0 student training
> used fold-1 rows labeled by a teacher that itself trained on fold 0. See
> the nested-clean correction below (`RT-994`/`RT-995`) and
> `research/WAVE7_TEACHER_NESTED_PREREG.md`. **The IDs and numbers are kept,
> not deleted or overwritten** — they remain a real mechanism-direction
> signal, just not a clean generalization estimate.

| ID | is |
|---|---|
| `RT-992` | W7 teacher pilot T1 — `RT-990`'s unmodified 500 legal causal columns/rows/capacity, fold 0 only, label = `Q` (`RT-991` OOF, reused as teacher, clipped to (1e-6, 1-1e-6)), objective `xentropy`. **CAUSAL AT INFERENCE** — only privileged input is the training label, never a feature. **Outer-fold contaminated, see above — not promotable as-is.** |
| `RT-993` | W7 teacher pilot T2 — identical to `RT-992` except label = `0.5*y + 0.5*Q`, fixed blend, no grid. Same causal-at-inference note and same contamination caveat as `RT-992`. |

`T0` (the matched hard-label control) reuses `RT-990`'s existing fold-0 OOF
slice — same rows, columns and capacity, so no new ID was needed; `RT-990`
itself has no nested-CV exposure (ordinary single-level 5-fold CV on hard
labels) and remains a valid control throughout. Pre-registration:
`research/WAVE7_TEACHER_PREREG.md`. Result:
`research/reports/wave7_teacher_pilot.{md,json}`,
`research/reports/wave7_teacher_diagnostics.{md,json}`,
`research/reports/wave7_teacher_parity_check.json`.

### `RT-994` / `RT-995` — W7 teacher/distillation, NESTED outer-fold-pure, 5-fold (T1, T2)

**Corrected design.** Fixes the `RT-992`/`RT-993` contamination: for every
outer validation fold `f`, `Q` for each outer-training row now comes from an
*inner* teacher trained only on folds excluding `{f, g}` (`g` = that row's
own fold), never on `f`. 20 inner teacher fits (no separate IDs — internal
machinery, not standalone experiments, never scored or saved to
`RESULTS.csv`), then `T1`/`T2` trained per outer fold on the nested `Q` and
evaluated on the untouched outer fold. Same student inputs/capacity/rows as
`RT-992`/`RT-993` — only the teacher's cross-fitting structure changed.

**These IDs were briefly allocated, then aborted, under the *old*
(contaminated) full-5-fold design before the contamination was found** — that
run was killed before any fold completed, so it never produced a
`RESULTS.csv` row or a saved OOF array; nothing under those IDs was ever
scored. They are reused here for the corrected nested design rather than
retired, since no contaminated artifact exists under them.

| ID | is |
|---|---|
| `RT-994` | W7 teacher NESTED T1 — pure distillation, nested outer-fold-pure `Q`, 5 outer folds. |
| `RT-995` | W7 teacher NESTED T2 — hard+teacher 0.5/0.5 blend, nested outer-fold-pure `Q`, 5 outer folds. |

**RESULT (complete, all 5 outer folds, 2026-08-23).** Mean Δ vs `RT-990`:
`RT-994` (T1) **+0.00349**, 3/5 folds positive, bootstrap CI `[-0.0047,
+0.0109]` (crosses zero) — reading "serious candidate" by magnitude alone,
but **fails promotion legs 2 and 3** (fold-consistency, bootstrap), so **does
not clear promotion**. `RT-995` (T2) **+0.00943**, 5/5 folds positive,
bootstrap CI `[+0.0043, +0.0137]` (entirely above zero) — **clears all three
measured promotion legs** (magnitude ≥+0.0030, ≥4/5 folds, bootstrap CI>0),
reading **"major breakthrough"**. Alternate-partition confirmation (leg 4)
is now authorized but not yet run — **no promotion or submission decision
until it is**. Contamination comparison (fold 0 only): the original
contaminated pilot overstated `T1` by **+0.0235** (true clean effect on that
fold was *negative*) and `T2` by **+0.0122** — the contamination was large,
but a smaller, real, robust effect survived for `T2`.

Pre-registration: `research/WAVE7_TEACHER_NESTED_PREREG.md`. Result:
`research/reports/wave7_teacher_nested.{md,json}`.

### `RT-1100` — T2 ensemble-integration blend (E2 arm)

| ID | is |
|---|---|
| `RT-1100` | RT-600 seven specialists + `RT-995` (T2), equal weight, cross-fitted SCDF calibration, fold 0. E2 arm of `research/WAVE7_T2_INTEGRATION_PREREG.md`. Blend, not a retrained model. |

E0 (seven specialists alone) and E1 (+`RT-401` matched seed clone) are
reused, not newly retrained or ID'd (E1 is the same construction Wave 8 used
under `rt600_plus_seedclone` for every one of its five mechanisms). Result:
`E2 − E1 = +0.00024` → MOSTLY REDUNDANT. Full writeup:
`research/reports/wave7_t2_promotion_final.md`. Pre-registration:
`research/WAVE7_T2_INTEGRATION_PREREG.md`.

Alternate-partition leg (`RT-994`/`RT-995`'s leg 4) was not run — see
`research/RDOF_LEDGER.md` for the resource-cost rationale.

---

# NEW AVENUES 2026 PILOTS (`research/new-avenues-pilots-2026`)

Reserved prospectively on 2026-08-24 before any New Avenues pilot score was
produced on this branch. Pre-registration:
`research/reports/new_avenues_2026/PILOTS_01_03_PREREG.md`.

| ID | is |
|---|---|
| `RT-1200` | Pilot 2 relay score-state transform of RT-600 evidence, fold-0 screen candidate. |
| `RT-1201` | Pilot 3 IM2 matched-length empirical run-null plus excursion dwell-bank scalar, fold-0 screen candidate. |
| `RT-1202` | Pilot 4 real trajectory-geometry scalar, fold-0 screen candidate. |
| `RT-1203` | Pilot 4 shuffled-history trajectory-geometry control scalar, fold-0 screen control. |

Pilot 1, specialist-competence and failure-manifold diagnostics, is
diagnostic-only and consumes no RT ID.

Reserved prospectively on 2026-08-24 before any Pilot 5 score was produced.
Pre-registration:
`research/reports/new_avenues_2026/PILOT05_PREREG.md`.

| ID | is |
|---|---|
| `RT-1204` | Pilot 5 scale-survival cross-scale summary features on dyadic AR(2) residual-square coarse-graining, fold-0 screen candidate. |
| `RT-1205` | Pilot 5 six individual per-scale surprise features on the same dyadic residual-square coarse-graining, fold-0 binding control. |

Result filed 2026-08-24:
`research/reports/new_avenues_2026/pilot05_scale_survival.{md,json}`.
`RT-1204` KILL: marginal_vs_clone `-0.000236`; `RT-1205` control
marginal_vs_clone `-0.000707`; summary-control gap `+0.000471`, below the
preregistered `+0.0005` distinguishability floor.

Reserved prospectively on 2026-08-24 before any Pilot 6 score was produced.
Pre-registration:
`research/reports/new_avenues_2026/PILOT06_PREREG.md`.

| ID | is |
|---|---|
| `RT-1206` | Pilot 6 spectral impulsiveness contrast features: band energy with low spectral-kurtosis / low robust-negentropy impulse evidence, fold-0 screen candidate. |
| `RT-1207` | Pilot 6 matched plain band-energy features on the same four frequency bands, fold-0 binding control. |

Result filed 2026-08-24:
`research/reports/new_avenues_2026/pilot06_spectral_impulse.{md,json}`.
`RT-1206` KILL: marginal_vs_clone `-0.000353`; `RT-1207` plain-energy
control marginal_vs_clone `+0.000189`; contrast-control gap `-0.000542`.
Primary marginal gate failed, and the control exceeded the contrast.

Reserved prospectively on 2026-08-24 before any Pilot 7 score was produced.
Pre-registration:
`research/reports/new_avenues_2026/PILOT07_PREREG.md`.

| ID | is |
|---|---|
| `RT-1208` | Pilot 7 ordinal transition divergence plus time-irreversibility features, fold-0 screen candidate. |
| `RT-1209` | Pilot 7 matched permutation-entropy-only ordinal control, fold-0 binding control. |

Result filed 2026-08-24:
`research/reports/new_avenues_2026/pilot07_ordinal_irreversibility.{md,json}`.
`RT-1208` KILL: marginal_vs_clone `-0.000036`; `RT-1209` entropy-only
control marginal_vs_clone `-0.000508`; candidate-control gap `+0.000472`,
below the preregistered `+0.0005` distinguishability floor.

Reserved prospectively on 2026-08-24 before any Pilot 10 score was produced.
Pre-registration:
`research/reports/new_avenues_2026/PILOT10_PREREG.md`.

| ID | is |
|---|---|
| `RT-1210` | Pilot 10 joint size-duration rarity features on AR(2) residual-square excursions, fold-0 screen candidate. |
| `RT-1211` | Pilot 10 matched dwell-only rarity features on the same excursion states, fold-0 binding control. |

Result filed 2026-08-24:
`research/reports/new_avenues_2026/pilot10_joint_rarity.{md,json}`.
`RT-1210` KILL: marginal_vs_clone `-0.000522`; `RT-1211` dwell-only
control marginal_vs_clone `-0.000237`; candidate-control gap `-0.000285`.
The primary marginal gate failed, the dwell-only control exceeded the joint
candidate, and the candidate lost the dominant-cell control comparison by
`-0.000896` AUC.

Reserved prospectively on 2026-08-24 before any Pilot 9(i) score was produced.
Pre-registration:
`research/reports/new_avenues_2026/PILOT09_PREREG.md`.

| ID | is |
|---|---|
| `RT-1212` | Pilot 9(i) nested scalar historical-difficulty conditioner predicting RT-600 dominant-cell loss propensity, fold-0 screen candidate. |
| `RT-1213` | Pilot 9(i) within-fold deranged scalar control preserving fold-wise scalar marginals, fold-0 binding control. |

Result filed 2026-08-24:
`research/reports/new_avenues_2026/pilot09_difficulty_gate.{md,json}`.
`RT-1212` KILL: marginal_vs_clone `+0.000164`; `RT-1213` deranged-control
marginal_vs_clone `+0.000269`; candidate-control gap `-0.000105`. The
deranged control exceeded the real scalar, and the real scalar also failed the
primary `+0.0010` marginal gate.

Reserved prospectively on 2026-08-24 before any Pilot 3 observer score was
produced. Pre-registration:
`research/reports/new_avenues_2026/PILOT03_OBSERVERS_PREREG.md`.

| ID | is |
|---|---|
| `RT-1214` | Pilot 3 frozen AR(2)-state Kalman/NIS observer features, fold-0 screen arm. |
| `RT-1215` | Pilot 3 frozen Hankel-DMD observer features, fold-0 screen arm. |

Result filed 2026-08-24:
`research/reports/new_avenues_2026/pilot03_observers.{md,json}`.
`RT-1214` KILL: marginal_vs_clone `+0.000135`. `RT-1215` KILL:
marginal_vs_clone `+0.000226`; the Hankel-DMD arm also failed the
preregistered redundancy gate with within-t rank correlation `+0.8859` versus
RT-600, above the `0.85` ceiling. Neither arm cleared the `+0.0010` primary
marginal gate.

Reserved prospectively on 2026-08-24 before any Pilot 8 score was produced.
Pre-registration:
`research/reports/new_avenues_2026/PILOT08_PREREG.md`.

| ID | is |
|---|---|
| `RT-1216` | Pilot 8 weighted conformal test martingale feature arm, fold-0 screen candidate. |
| `RT-1217` | Pilot 8 matched unweighted conformal test martingale feature control, fold-0 binding control. |
| `RT-1218` | Pilot 8 parameter-free e-value aggregation direct-score arm, fold-0 screen candidate. |

Result filed 2026-08-24:
`research/reports/new_avenues_2026/pilot08_weighted_ctm.{md,json}`.
`RT-1216` KILL: marginal_vs_clone `+0.000937`, below the `+0.0010`
primary gate. The weighted arm did beat the matched unweighted CTM on the
preregistered mature-vs-never split (`+0.002135`), but that split gate cannot
override the primary marginal miss. `RT-1218` KILL: marginal_vs_clone
`-0.002214`. No Pilot 8 arm cleared continuation; the planned New Avenues
first-sweep queue is exhausted.

Second Sweep SS-01 reserved prospectively on 2026-08-25 before any SS-01 score
was produced. Program pre-registration:
`research/reports/new_avenues_2026/SECOND_SWEEP_PREREG.md`. Execution
pre-registration:
`research/reports/new_avenues_2026/second_sweep/SS01_EXECUTION_PREREG.md`
at `32b5427`.

| ID | is |
|---|---|
| `RT-1219` | SS-01 constrained repair-damage arbiter over frozen first-sweep sensors, fold-0 screen candidate. |
| `RT-1220` | SS-01 global average of the same frozen first-sweep sensors, fold-0 binding control. |
| `RT-1221` | SS-01 shuffled repair/damage-target arbiter, fold-0 binding control. |
| `RT-1222` | SS-01 single-best killed-arm blend selected on train folds only, fold-0 binding control. |

Result filed 2026-08-25:
`research/reports/new_avenues_2026/second_sweep/ss01_repair_damage_arbiter.{md,json}`.
`RT-1219` KILL: marginal_vs_clone `-0.000310`; dominant-cell repairs `0`,
damage `0`, net `0`; contributing sensor families `0`; repair-reservoir
retention `0.0000`; first-sweep candidate-union damage rejected `1.0000`.
Controls did not lose: `RT-1220` marginal_vs_clone `+0.000314`, `RT-1221`
`-0.000310`, and `RT-1222` `+0.000317`. SS-01 failed the preregistered
mandatory gates and no SS-01b is authorized.

Second Sweep SS-02 reserved prospectively on 2026-08-25 before any SS-02 score
was produced. Program pre-registration:
`research/reports/new_avenues_2026/SECOND_SWEEP_PREREG.md`. Execution
pre-registration:
`research/reports/new_avenues_2026/second_sweep/SS02_EXECUTION_PREREG.md`
at `3b39954`.

| ID | is |
|---|---|
| `RT-1223` | SS-02 bounded residual same-t pair correction over the incumbent 500-feature bank, fold-0 screen candidate. |
| `RT-1224` | SS-02 shuffled residual-offset/weight control, fold-0 binding control. |

Result filed 2026-08-25:
`research/reports/new_avenues_2026/second_sweep/ss02_residual_ranker.{md,json}`.
`RT-1223` KILL: marginal_vs_clone `-0.000290`; dominant-cell repairs `330`,
damage `438`, net `-108`; mature-vs-never net `-77`; mature-vs-prebreak net
`-102`; RT600-right dominant damage rate `0.0128`. The shuffled residual
control `RT-1224` had marginal_vs_clone `-0.000203`, so the candidate-control
gap was `-0.000087` instead of the required `+0.000500`. SS-02 failed the
mandatory gates and no SS-02b is authorized.

Second Sweep SS-03 reserved prospectively on 2026-08-25 before any SS-03 score
was produced. Program pre-registration:
`research/reports/new_avenues_2026/SECOND_SWEEP_PREREG.md`. Execution
pre-registration:
`research/reports/new_avenues_2026/second_sweep/SS03_EXECUTION_PREREG.md`
at `8a5f41a`.

| ID | is |
|---|---|
| `RT-1225` | SS-03 weighted-CTM null-state conditional calibration of RT600, fold-0 screen candidate. |
| `RT-1226` | SS-03 global RT600 null-SCDF calibration control. |
| `RT-1227` | SS-03 deranged weighted-CTM null-state partition control. |
| `RT-1228` | SS-03 unweighted CTM state partition control. |
| `RT-1229` | SS-03 Pilot-9 scalar difficulty state partition control. |

Result filed 2026-08-25:
`research/reports/new_avenues_2026/second_sweep/ss03_null_calibrator.{md,json}`.
`RT-1225` KILL: marginal_vs_clone `-0.000299`; dominant-cell repairs `659`,
damage `773`, net `-114`; mature-vs-never net `-49`; mature-vs-prebreak net
`-40`; prebreak RT600-right damage rate `0.0199`, above the preregistered
`0.0150` cap. The deranged partition control `RT-1227` lost only by
`+0.000049` marginal versus the required `+0.000500`, and the unweighted CTM
state control `RT-1228` slightly exceeded the candidate. SS-03 failed the
mandatory gates and no SS-03b is authorized.

Second Sweep SS-04 reserved prospectively on 2026-08-25 before any SS-04 score
was produced. Program pre-registration:
`research/reports/new_avenues_2026/SECOND_SWEEP_PREREG.md`. Execution
pre-registration:
`research/reports/new_avenues_2026/second_sweep/SS04_EXECUTION_PREREG.md`
at `25f40ac`.

| ID | is |
|---|---|
| `RT-1230` | SS-04 bounded row-level specialist action router, fold-0 screen candidate. |
| `RT-1231` | SS-04 global logistic specialist reweighting control. |
| `RT-1232` | SS-04 Pilot-1 `exc_max_run64` static history-fingerprint selector replay control. |
| `RT-1233` | SS-04 shuffled disagreement-target router control. |

Result filed 2026-08-25:
`research/reports/new_avenues_2026/second_sweep/ss04_specialist_router.{md,json}`.
`RT-1230` KILL: marginal_vs_clone `-0.000312`; dominant-cell repairs `0`,
damage `0`, net `0`; majority-correct specialist-disagreement net `0`;
near-split net `0`. Controls did not lose by the required `+0.000500` gap:
`RT-1231` candidate-control gap `+0.000130`, `RT-1232` gap `+0.000243`, and
`RT-1233` gap `-0.000002`. SS-04 failed the mandatory gates, no SS-04b is
authorized, and the preregistered Second Sweep is exhausted.

---

# CAUSAL REPRESENTATION FRONTIER (`research/causal-representation-frontier-2026`)

Program preregistration:
`research/reports/causal_representation_frontier/CRF_PROGRAM_PREREG.md` at
`85d121f`. It freezes three designs and allocates no id; each experiment
requires its own execution preregistration committed before any score exists.

## CRF-01 · NNCSR

Reserved prospectively on 2026-08-26 **before any CRF-01 score, OOF vector or
TS-AUC existed**. Execution preregistration:
`research/reports/causal_representation_frontier/CRF01_EXECUTION_PREREG.md`.

| ID | is |
|---|---|
| `RT-1234` | CRF-01 candidate — eight null-normalised history-fitted channels (`pit`, `inn`, `inn_pit`, `abs_inn_pit`, `vol_norm`, `surp`, `exceed`, `lag1_pit`) into the `RT-970` causal dilated TCN shell (hidden 32, 35,649 params, measured receptive field 253), trained under the same-`t` pairwise logistic objective with `m_neg = 8`. No `elapsed` channel, no 500-column bank, no RT600. Fold-0 screen candidate. |
| `RT-1235` | CRF-01 C1 mandatory control — byte-identical channels, architecture, optimiser, schedule, epochs, batches and seeds, masked rowwise BCE loss. Isolates the **objective**. |
| `RT-1236` | CRF-01 C2 conditional control — identical arm and objective with online channels temporally shuffled within each series (one per-series permutation applied to all eight channels, label left at its original `t`). Isolates the **temporal representation**. Declared and reserved in advance; runs only if the candidate clears the preregistered cheap abandon gate. Diagnostic only — not causal at inference, never a deployment candidate. |

`RT-401` and the seven RT-600 specialists `RT-300`, `RT-410`–`RT-415` are reused
as frozen OOF vectors and consume no id.

Collision audit performed before allocation: the highest previously allocated id
in the `RT-12xx` band is `RT-1233`; `RT-1234`/`RT-1235`/`RT-1236` appear in no
`RESULTS.csv` on any local or remote ref, in no tracked file, and in no commit
reachable from any ref (`git log --all -S`). None is a recycled killed, void,
abandoned, contaminated or reserved id.

### CRF-01 result, filed 2026-08-26

`research/reports/causal_representation_frontier/crf01_nncsr.{md,json}`.

**`RT-1234` KILL — abandoned at the preregistered cheap abandon gate.** Fold-0
standalone whole-fold TS-AUC `0.592762` (below the `0.600` necessary condition)
with within-`t` ρ vs RT600 `+0.4460` (at or below the `0.60` ceiling), so
`CRF_PROGRAM_PREREG.md` §0.2 fires and the candidate is abandoned before any
five-fold spend. `marginal_vs_clone` was **never computed** — that is the gate's
intended effect, not an omission, since the fold-0 marginal requires a fold-pure
five-fold OOF and folds 1–4 were deliberately not trained. Dominant-cell standalone
`0.621422`; dominant pair net `-2,798`; mature-vs-never net `-2,962`;
mature-vs-pre-break net `-2,820`; pre-break damage rate on RT600-correct pairs
`0.2626` against the `0.0150` cap.

**`RT-1235` C1 mandatory BCE control** ran on fold 0 and is preserved: standalone
`0.570543`, dominant-cell `0.584322`, ρ `+0.3631`, dominant pair net `-5,019`. The
candidate beats it by `+0.02222` whole-fold and `+0.03710` dominant-cell, so the
same-`t` pairwise objective is **materially better than BCE on a learned
representation** — the first clean isolation of that factor in this project, and a
real positive finding that does not resurrect the candidate
(`CRF_PROGRAM_PREREG.md` §1.10).

**`RT-1236` was NOT run and its id is NOT consumed.** `CRF_PROGRAM_PREREG.md` §1.7
opens C2 only if the candidate clears the abandon gate; it did not. The id remains
reserved to CRF-01's C2 arm and must not be recycled or reassigned to anything else.

Ladder against `RT-970` (same shell, same folds): `RT-970` 0.52618 →
`RT-1235` 0.57054 (**channel effect +0.0444** at fixed objective) →
`RT-1234` 0.59276 (**objective effect +0.0222** at fixed channels). Both Wave-6
hypotheses were partly right and their sum is still insufficient.

`RT-1234` and `RT-1235` OOF vectors are fold-0 only by construction: finite on the
806,334 fold-0 rows, NaN on folds 1–4, zero finite lockbox rows. Their
`mean_oof_ts_auc` / `pooled_oof_ts_auc` / `per_fold_ts_auc` columns in
`research/RESULTS.csv` hold the arm's **standalone** fold-0 TS-AUC, **not** an
ensemble score — unlike the SS-0x rows, which hold `E2`. The `notes` column says so
on both rows.

## CRF-02 · ACGN

Reserved prospectively on 2026-08-26 **before any CRF-02 score, OOF vector or
TS-AUC existed**. Execution preregistration:
`research/reports/causal_representation_frontier/CRF02_EXECUTION_PREREG.md`.
CRF-02 is **unconditional and independent of CRF-01's outcome**
(`CRF_PROGRAM_PREREG.md` §2); it is not a response to CRF-01's kill.

| ID | is |
|---|---|
| `RT-1237` | CRF-02 candidate — learned amortized conditional generative null `q(x_t \| h_i, x_{t-1..t-R})`, pretrained on the **training-fold series' break-free histories only** for that outer fold, 21 monotone quantile knots via base + softplus increments, pinball loss, no label; then **frozen**; then five preregistered causal online signals plus their running peaks into a 10→32→1 same-`t` pairwise ranking head. Fold-0 screen candidate. |
| `RT-1238` | CRF-02 C1 mandatory control — matched **fixed** null (AR(5) + history residual ECDF) feeding identical downstream statistics and an identical ranking head. Isolates learned vs fixed conditional null. |
| `RT-1239` | CRF-02 C2 mandatory control — history embeddings `h_i` deranged across series within fold, everything else identical. Isolates useful conditioning vs series identity/memorisation. If C2 matches or beats the candidate the arm is KILL regardless of its headline number (the `m05_ctx` rule). |

**`RT-1236` is deliberately skipped, not recycled** — it remains reserved to
CRF-01's C2 temporal-shuffle arm, which was gate-blocked and never run.

Collision audit before allocation: highest allocated id `RT-1235`; `RT-1237`,
`RT-1238`, `RT-1239` appear in no `RESULTS.csv` on any ref, in no tracked file, and
in no commit reachable from any ref.

### CRF-02 result, filed 2026-08-26

`research/reports/causal_representation_frontier/crf02_acgn.{md,json}`.

**`RT-1237` KILL on three independent mandatory grounds.**

1. **Cheap abandon gate fired**: fold-0 standalone whole-fold TS-AUC `0.527807`
   (below `0.600`) at within-`t` ρ `+0.2212` (below `0.60`).
2. **Learned-null isolation failed**: candidate − C1 = **`-0.052779`** whole-fold
   (`-0.060600` dominant-cell) against a required `+0.000500`. The **fixed**
   per-series null `RT-1238` — AR(5) + 256-knot history residual ECDF — reached
   `0.580586` / `0.599909` through **identical** downstream statistics and an
   **identical** ranking head, beating the learned null by `0.0528`.
3. **Derangement gate failed**: candidate − C2 = **`-0.000981`**. Permuting `h_i`
   across series within fold changes essentially nothing, so the 8-dimensional
   history bottleneck carries no usable series information — the `m05_ctx` rule,
   which kills the arm regardless of its headline number.

`marginal_vs_clone` was never computed for any arm: the abandon gate fired and
folds 1–4 were deliberately not trained, so the isolation comparisons are on
standalone fold-0 TS-AUC exactly as `CRF02_EXECUTION_PREREG.md` §8 stage S1
preregistered. Both mandatory controls were run **before** the gate was read so the
scientific comparison would survive a headline failure — which is what happened.

The declared shared-8 diagnostic (no RT id) refit the candidate's head on the eight
features the fixed-null control also has and reached `0.531864`, **above** the
10-feature candidate: the isolation gate was not flattered by the candidate's two
extra encoder-derived features.

Pair flow negative in every cell for every arm; pre-break damage rate on
RT600-correct pairs `0.4144` (candidate) against the `0.0150` cap.

**Reading.** The learned null is effectively a **population-average** predictive
distribution, because C2 shows `h_i` conditions on nothing usable. The fixed null is
a **per-series** fit paid for by that series' own break-free history at zero
generalisation cost. Amortization across series therefore *loses* information rather
than adding it, and an 8-float bottleneck is not a wide enough channel to recover
what a per-series fit gets for free.

`RT-1237`/`RT-1238`/`RT-1239` OOF vectors are fold-0 only: finite on the 806,334
fold-0 rows, NaN on folds 1–4, zero finite lockbox rows. Their `mean_oof_ts_auc` /
`pooled_oof_ts_auc` / `per_fold_ts_auc` columns hold the arm's **standalone** fold-0
TS-AUC, not an ensemble score; the `notes` column says so on all three rows.

**CRF-03 does not open.** `CRF_PROGRAM_PREREG.md` §3.1 requires a fold-0
`marginal_vs_clone ≥ +0.0015` from CRF-01 or CRF-02 plus a passing mandatory
isolation control. Both primaries are KILL and neither has a marginal. No further
CRF id is allocated. `RT-1236` remains reserved and unconsumed.

### `RT-1237` / `RT-1238` / `RT-1239` — **VOID**, and not reusable

**Defect found 2026-08-26, immediately after the CRF-02 result was filed and
pushed at `9a5ecc0`, during routine compute accounting for `CRF_FINAL.md`.**

The reported CRF-02 fold-0 run did **not** train the preregistered generative
null. It silently **loaded a checkpoint written by a unit test** —
`fit_series = 24`, `pretrain_epochs = 1`, `HWIN = 128`, pinball `2.1786`, state
`d70f793fe464…` — instead of the preregistered 6,383-series, 10-epoch,
`HWIN = 1024` null. The tell was `pretrain_runtime_s = 0.2` against the 1,284.6 s
the real pretraining had taken on the earlier attempt.

**Cause.** A null checkpoint was added at `89149d5` so a downstream failure would
not cost a refit. `tests/test_crf02_causality.py` exercises the **real**
`pretrain_null` (that is the point of gates P3 and P6b) and `CACHE` resolved to the
production `cache/crf02/`, so the test's toy null was written there and the next
real run loaded it. The load-time assertion compared the checkpoint's `state_sha256`
against its own recorded value — which proves **integrity, not provenance**, and a
toy null is perfectly self-consistent.

**What is and is not affected.**

* `RT-1237` (candidate, learned null) — **VOID.** Produced by the toy null.
* `RT-1239` (deranged `h_i`) — **VOID.** Same toy null.
* `RT-1238` (fixed AR(5)+ECDF null) — **structurally unaffected**: its code path
  never touches the neural null. It is voided anyway, because an arm's whole
  purpose here is the *comparison*, and a control retained from a voided
  comparison invites exactly the confusion this file exists to prevent. It is
  re-run and is expected to reproduce bitwise, which is itself a check.
* **CRF-01 (`RT-1234`/`RT-1235`) is unaffected and stands.** Verified directly: its
  emitted metadata records `n_train_series = 6383`, `n_val_series = 1617`, 20
  epochs, 906.6 s / 884.2 s. `crf01_nncsr.py` writes no checkpoint, and its tests
  never call `emit` or `build_channels`, so nothing in `cache/crf01/` was
  test-written.

**The three ids are retired.** They are not reused, reassigned or recycled, per the
`RT-900` precedent. Their `research/RESULTS.csv` rows are **left exactly as
written** — the ledger is append-only and the record is the record — and they must
never appear in a comparison table, an ensemble, a promotion decision, feature
selection, production or a submission. `research/oof/RT-123{7,8,9}.npy` are the toy
null's output and are equally void.

**Two fixes, both asserted in code.**

1. A null checkpoint now carries a **provenance fingerprint** — fold, seed,
   `PRETRAIN_EPOCHS`, `HWIN`, `BATCH_SERIES`, `HIDDEN`, `BOTTLENECK`, `n_levels`,
   `lr`, `wd`, `n_fit_series` and the **sha256 of the sorted fit-series ids** — and a
   mismatch raises `CHECKPOINT PROVENANCE MISMATCH` and **stops the run**. It is
   loud rather than a silent fall-through to retraining, because a mismatch is
   evidence that something is wrong.
2. Every test now gets a throwaway `CACHE` via an autouse fixture, so no test can
   write into the production cache at all.

`test_ckpt_provenance_mismatch_is_refused_loudly` reproduces the exact defect —
writes a 24-series 1-epoch checkpoint, then asks for the real fit set — and
requires the refusal. **27 gates now pass.**

### CRF-02 corrected allocation

| ID | is |
|---|---|
| `RT-1240` | CRF-02 candidate, **corrected run** — the learned amortized conditional generative null, executing `CRF02_EXECUTION_PREREG.md` @ `9a3d3c7` **unchanged**. |
| `RT-1241` | CRF-02 C1 mandatory control, corrected run — fixed AR(5) + history residual ECDF null. |
| `RT-1242` | CRF-02 C2 mandatory control, corrected run — deranged `h_i`. |

**No design changed.** The execution preregistration is the one frozen at `9a3d3c7`,
before any CRF-02 number existed; the defect was that the run did not execute it.
The corrected run does. Verified unused before allocation on every ref, in every
tracked file, and in every reachable commit.

### CRF-02 corrected result, filed 2026-08-26

`research/reports/causal_representation_frontier/crf02_acgn.{md,json}`.

**`RT-1240` KILL.** Two mandatory gates failed and one passed.

1. **Cheap abandon gate fired**: fold-0 standalone whole-fold TS-AUC `0.559140`
   (below `0.600`) at within-`t` ρ `+0.2887` (below `0.60`).
2. **Learned-null isolation failed**: candidate − C1 = **`-0.021446`** whole-fold
   (`-0.020337` dominant-cell) against a required `+0.000500`. The **fixed**
   per-series null `RT-1241` — AR(5) + 256-knot history residual ECDF — reached
   `0.580586` / `0.599909` through **identical** downstream statistics and an
   **identical** ranking head.
3. **Derangement gate PASSED**: candidate − C2 = **`+0.003377`** whole-fold
   (`+0.008391` dominant-cell). The 8-dimensional history bottleneck carries real
   series-specific information; the model is **not** memorising a series identifier.
   The `m05_ctx` rule was **not** triggered.

**The passing derangement control is what makes this negative sharp.** The learned
null conditions correctly and loses anyway, so the failure is **amortization**, not
a broken implementation. The fixed null holds five AR coefficients plus a 256-knot
empirical residual distribution *per series*, paid for by that series' own break-free
history at zero generalisation cost; the learned null compresses all of it into
**8 floats** shared across a population whose heterogeneity is the reason per-series
historical calibration is this project's foundation.

`RT-1241` reproduced the void `RT-1238` **bitwise**, confirming its code path never
touched the neural null. The declared no-id shared-8 diagnostic reached `0.551432`,
**below** the 10-feature candidate, so the isolation gate is conservative in the
candidate's favour: on the eight shared features the candidate loses by `-0.029154`.

`marginal_vs_clone` was never computed for any arm — the abandon gate fired and folds
1–4 were deliberately not trained — so the isolation comparisons are on standalone
fold-0 TS-AUC exactly as `CRF02_EXECUTION_PREREG.md` §8 stage S1 preregistered. Both
mandatory controls ran **before** the gate was read.

Generative null provenance, now verified by fingerprint: 6,383 training-fold series'
break-free histories, 10 epochs, `HWIN = 1024`, pinball `0.364706 → 0.243881`,
1,364.9 s, state `e58bc20e05d9…`.

`RT-1240`/`RT-1241`/`RT-1242` OOF vectors are fold-0 only: finite on the 806,334
fold-0 rows, NaN on folds 1–4, zero finite lockbox rows. Their `mean_oof_ts_auc` /
`pooled_oof_ts_auc` / `per_fold_ts_auc` columns hold the arm's **standalone** fold-0
TS-AUC, not an ensemble score.

**CRF-03 does not open.** `CRF_PROGRAM_PREREG.md` §3.1 requires a fold-0
`marginal_vs_clone ≥ +0.0015` from CRF-01 or CRF-02 plus a passing mandatory
isolation control. Both primaries are KILL and neither produced a marginal. No
further CRF id is allocated. `RT-1236` remains reserved and unconsumed;
`RT-1237`/`RT-1238`/`RT-1239` remain void and retired.

---

## Leaderboard Alpha 2026

Program preregistration:
`research/reports/leaderboard_alpha_2026/PROGRAM_PREREG.md` at `ed84d00`.

### LA-01 · Specialist replacement salvage

Reserved prospectively on 2026-08-26 before any LA-01 score, OOF vector,
or pair-flow diagnostic existed on `research/leaderboard-alpha-2026`.
Execution preregistration:
`research/reports/leaderboard_alpha_2026/LA01_EXECUTION_PREREG.md`.

| ID | arm |
|---|---|
| `RT-1243` | LA-01 headline nested seven-member replacement arm. For each outer fold, select on the other four folds which RT600 specialist is replaced by either `RT-731` (`m11_focus`) or `RT-751` (`m12_rdep`), then evaluate that frozen seven-member equal-SCDF blend on the outer fold. |
| `RT-1244` | LA-01 secondary control. Same nested replacement protocol, but the replacement candidate is the exchangeable seed clone `RT-401`. |

### LA-01 result, filed 2026-08-26

`RT-1243` is **KILL**. Full report:
`research/reports/leaderboard_alpha_2026/LA01_NESTED_REPLACEMENT.md`; metrics:
`la01_nested_replacement.{json,csv}`.

E0 RT600 seven-specialist mean TS-AUC `0.625811342`; `RT-1244` nested
seed-replacement control `0.624980472`; `RT-1243` nested `m11_focus`/`m12_rdep`
replacement `0.624975063`. Primary `marginal_vs_clone = -0.000005408`; positive
folds `2/5`; dominant-cell repairs/damage/net `828/838/-10`; mature-vs-never
net `-38`; mature-vs-prebreak net `-99`.

The result closes specialist-replacement salvage. Continue to LA-02.

### LA-02 · Counterfactual synthetic augmentation

Reserved prospectively on 2026-08-26 before any LA-02 score, OOF vector, or
pair-flow diagnostic existed. Execution preregistration:
`research/reports/leaderboard_alpha_2026/LA02_EXECUTION_PREREG.md`.

| ID | arm |
|---|---|
| `RT-1245` | LA-02 C1 same-count synthetic fixed-null-only control. Seven RT600 stream configs trained on real rows plus synthetic null-only rows; validation remains real only. |
| `RT-1246` | LA-02 candidate paired counterfactual augmentation. Seven RT600 stream configs trained on real rows plus paired persistent-positive/transient-hard-negative synthetic rows; validation remains real only. |

### LA-02 result, filed 2026-08-26

`RT-1246` is **KILL** under the preregistered survival gate. Full report:
`research/reports/leaderboard_alpha_2026/LA02_COUNTERFACTUAL_AUGMENTATION.md`;
metrics: `la02_counterfactual_augmentation.{json,csv}`.

C0 RT600 mean TS-AUC `0.625811264`; `RT-1245` null-only synthetic control
`0.617350841`; `RT-1246` paired counterfactual candidate `0.622508549`.
Primary `marginal_vs_clone = +0.005157709`; positive folds `4/5`; metric class
by magnitude **MAJOR**. The final verdict is still **KILL** because two binding
pair-flow gates fail: dominant-cell repairs/damage/net `2853/3179/-326`, and
mature-vs-never net `-366`. Candidate vs C0 is also negative at `-0.003302715`.

Because LA-02 did not become SERIOUS under the full gate, LA-03 opens.

### LA-03 · Per-series history adaptation

Reserved prospectively on 2026-08-26 before any LA-03 score, OOF vector, or
pair-flow diagnostic existed. Execution preregistration:
`research/reports/leaderboard_alpha_2026/LA03_EXECUTION_PREREG.md`.

| ID | arm |
|---|---|
| `RT-1247` | LA-03 C0 global no-adaptation predictive-null head. One outer-fold global AR(5), pooled residual ECDF, same ten residual/PIT features, same LightGBM pairwise head. Also the E1 RT600+head clone in the final marginal comparison. |
| `RT-1248` | LA-03 C1 fixed per-series AR(5)+history-residual-ECDF null head. Mandatory isolation control and integrated diagnostic. |
| `RT-1249` | LA-03 candidate global AR(5) plus per-series affine adapter (`a_i`, `b_i`) fitted only on `H_i`, same ten residual/PIT features, same LightGBM pairwise head. Also the E2 RT600+head candidate in the final marginal comparison. |

### LA-03 result, filed 2026-08-26

`RT-1249` is **KILL**. Full report:
`research/reports/leaderboard_alpha_2026/LA03_PER_SERIES_ADAPTATION.md`;
metrics: `la03_per_series_adaptation.{json,csv}`.

Standalone heads: `RT-1247` global no-adaptation `0.528175061`; `RT-1248`
fixed per-series null `0.564744385`; `RT-1249` affine-adapted candidate
`0.543249095`. The adapted head beats the global clone by `+0.015074034` but
fails the fixed-null isolation gate by `-0.021495290`.

Integrated with RT600: E0 `0.625811264`; E1 `RT600+RT-1247` `0.624411329`;
E2 `RT600+RT-1249` `0.626087395`; primary `marginal_vs_clone = +0.001676065`
with `5/5` folds positive, i.e. WEAK by metric size. Final verdict remains
KILL because the mandatory fixed-null isolation gate failed. The fixed-null
diagnostic `RT600+RT-1248` was stronger than the candidate at `0.626869923`.

Pair flow for E2 vs E0 was also negative: dominant-cell net `-37`,
mature-vs-never net `-92`, mature-vs-prebreak net `-74`.

No Leaderboard Alpha mechanism survived; the combination rule does not open.

## Learner Diversity 2026

Reserved prospectively on 2026-08-26 before any Learner Diversity score, OOF
vector, replacement analysis, or pair-flow diagnostic existed. Execution
preregistration:
`research/reports/learner_diversity_2026/PREREG.md`.

| ID | arm |
|---|---|
| `RT-1250` | LD-01 TabM. Authorized learner-family arm using the same 500 causal columns, canonical folds, RT-401-matched row sampler, and fold-pure SCDF replacement protocol. Recorded INFEASIBLE before scoring because the installed official default full-scale run is not feasible without shrinking or retuning. |
| `RT-1251` | LD-02 CatBoost. Authorized full five-fold CatBoost candidate using the same 500 causal columns, canonical folds, RT-401-matched row sampler, and fold-pure seven-member replacement protocol. |
| `RT-1252` | LD-03 RealMLP. Authorized learner-family arm using the same 500 causal columns, canonical folds, RT-401-matched row sampler, and fold-pure SCDF replacement protocol. Recorded INFEASIBLE before scoring because the installed official default full-scale run is not feasible without shrinking or retuning. |
| `RT-1253` | Conditional one-combination slot. Opens only if at least two learner-family candidates individually achieve `marginal_vs_clone >= +0.0010`; otherwise unused. |

### Learner Diversity result, filed 2026-08-27

Full report: `research/reports/learner_diversity_2026/FINAL.md`; metrics:
`learner_results.{json,csv}`.

`RT-1250` TabM and `RT-1252` RealMLP are **INFEASIBLE** under the frozen
installed official/default full-scale configurations. Shrinking rows, epochs,
width, `k`, or patience would be a new experiment and was not authorized.

`RT-1251` CatBoost is **INTERESTING** but not SERIOUS. Standalone TS-AUC
`0.620407665`; matched `RT-401` standalone `0.616609144`; within-`t` rho
`+0.736467811`. Replacement test: E0 RT600 `0.625811342`; E1 six specialists
plus `RT-401` `0.624980472`; E2 six specialists plus CatBoost `0.626090272`.
Primary `marginal_vs_clone = +0.001109800`, positive folds `5/5`,
dominant-cell pair net `+88`. This clears the INTERESTING threshold
(`+0.0010`) but not the SERIOUS threshold (`+0.0030`). Only one learner cleared
the combination-opening threshold, so `RT-1253` remains unused.

## CatBoost Specialist Activation 2026

Reserved prospectively on 2026-08-27 before any CatBoost specialist score, OOF
vector, replacement analysis, or pair-flow diagnostic existed on this branch.
Execution preregistration:
`research/reports/catboost_specialist_2026/PREREG.md`.

| ID | arm |
|---|---|
| `RT-1254` | CSA-01 primary CAT-413. RT-413 specialist training setup reimplemented with the frozen `RT-1251` CatBoost learner; fixed-slot replacement against `RT-401`. |
| `RT-1255` | CSA-02 conditional CAT-300. Opens only if CSA-01 reaches `marginal_vs_clone >= +0.0010`; fixed-slot replacement against `RT-401`. |
| `RT-1256` | CSA-02 conditional CAT-410. Opens only if CSA-01 reaches `marginal_vs_clone >= +0.0010`; fixed-slot replacement against `RT-401`. |
| `RT-1257` | CSA-03 conditional hybrid seven-member ensemble. Opens only if at least two CatBoost specialists reach `marginal_vs_clone >= +0.0010`. |

### CatBoost Specialist Activation result, filed 2026-08-27

Full report: `research/reports/catboost_specialist_2026/FINAL.md`; metrics:
`results.{json,csv}`.

CSA-00 zero-training 8th member was descriptive only: E2-E1
`+0.000975838`, E2-E0 `+0.001002808`, dominant-cell net vs E0 `+70`.

`RT-1254` CAT-413 is **INTERESTING**: standalone `0.620440244`, rho vs
incumbent RT-413 `+0.714196`, fixed-slot `marginal_vs_clone = +0.001087151`,
E2-E0 `+0.001244347`, `5/5` folds positive, dominant-cell net `+93`.

`RT-1255` CAT-300 is **INTERESTING**: standalone `0.620242061`, rho vs
incumbent RT-300 `+0.737786`, fixed-slot `marginal_vs_clone = +0.001029459`,
E2-E0 `+0.001126876`, `5/5` folds positive, dominant-cell net `+75`.

`RT-1256` CAT-410 is **KILL** despite standalone improvement: fixed-slot
`marginal_vs_clone = +0.000753095`, below the `+0.0010` gate.

`RT-1257` HYBRID replaces only RT-300 and RT-413 with their surviving CatBoost
versions. It is **PROMOTION_WORTHY** but not SERIOUS/MAJOR:
E0 `0.625811342`; matched clone control E1 `0.625430459`; CatBoost hybrid E2
`0.627837664`; primary `marginal_vs_clone = +0.002407205`; E2-E0
`+0.002026322`; `5/5` folds positive; dominant-cell net `+158`;
mature-vs-never net `+73`.

## GPU Tabular 2026 — Full 5-Fold OOF

Reserved prospectively on 2026-08-27 before any full-scale TabM or RealMLP
score, OOF vector, replacement analysis, or pair-flow diagnostic existed on
`research/gpu-tabular-2026`. Execution preregistration:
`research/reports/gpu_tabular_2026/FULL_OOF_PREREG.md`. Frozen benchmark
basis: `research/reports/gpu_tabular_2026/FROZEN_GPU_CONFIG.json`, RTX 4090
cloud hardware benchmark (Crunch submission `76357`, task `run-3e834e0f`).

| ID | arm |
|---|---|
| `RT-1258` | GPU-01 TabM full 5-fold OOF, official `tabm.TabM` package, frozen exactly as RTX-4090-benchmarked (`k=32,d_block=512,n_blocks=3`, no shrink). Fixed-slot nested replacement against `RT-401`. |
| `RT-1259` | GPU-02 RealMLP full 5-fold OOF, `pytabkit.RealMLP_TD_Classifier`, frozen exactly as RTX-4090-benchmarked (`hidden_sizes=[256,256,256]`, no shrink). Fixed-slot nested replacement against `RT-401`. |

These are **distinct** from `RT-1250` (TabM) and `RT-1252` (RealMLP) on
`research/learner-diversity-2026`, which were ruled **INFEASIBLE**: the
installed official/default full-scale configuration did not fit the compute
budget there, and shrinking rows/epochs/width/`k`/patience was not
authorized. The RTX 4090 hardware benchmark on this branch removed that
blocker — the same default/official-scale configurations fit the 15-hour
budget without any shrink (TabM `3.63606h`, RealMLP `0.90924h`, combined
`4.545298927912005h` for both learners' five folds) — so `RT-1258`/`RT-1259`
are freshly numbered rather than reopening `RT-1250`/`RT-1252`.

Both candidates must complete full 5-fold OOF before either is evaluated.
Result not yet filed; no score exists at allocation time.

## Deep Ensemble Frontier 2026 -- LOCAL Lane CSA-04

Reserved prospectively on 2026-08-28 before any CSA-04 score, OOF vector,
replacement analysis, hybrid-curve result, or pair-flow diagnostic existed on
`research/deep-ensemble-frontier-local-2026`. Execution preregistration:
`research/reports/deep_ensemble_frontier_2026/local/CSA04_PREREG.md`.

| ID | arm |
|---|---|
| `RT-1260` | CSA-04 CAT-411. Reimplement `RT-411` with the frozen `RT-1251` CatBoost learner; fixed-slot replacement against `RT-401`. |
| `RT-1261` | CSA-04 CAT-412. Reimplement `RT-412` with the frozen `RT-1251` CatBoost learner, preserving `sample_mode="per_series"`; fixed-slot replacement against `RT-401`. |
| `RT-1262` | CSA-04 CAT-414. Reimplement `RT-414` with the frozen `RT-1251` CatBoost learner; fixed-slot replacement against `RT-401`. |
| `RT-1263` | CSA-04 CAT-415. Reimplement `RT-415` with the frozen `RT-1251` CatBoost learner under the preregistered GOSS caveat; fixed-slot replacement against `RT-401`. |
| `RT-1264` | CSA-04 best-`k` hybrid over all CatBoost specialist survivors from CAT-413, CAT-300, CAT-410, CAT-411, CAT-412, CAT-414, and CAT-415, ordered by descending single-slot `marginal_vs_clone`. |

`RT-1265` through `RT-1269` remain unallocated LOCAL-lane contingency IDs.
