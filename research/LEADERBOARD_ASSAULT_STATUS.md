# LEADERBOARD ASSAULT STATUS

**Written 2026-08-23 on `research/wave6-alpha`, end of the W7-D0/D3R session.
Updated 2026-08-23 on `research/wave7-teacher-distillation`, first after the
W7 teacher/distillation one-fold pilot, then again after that pilot was
found outer-fold contaminated and replaced by a nested (double)
cross-fitted 5-fold correction.**
Read this before starting any further Wave-7 work.

---

## WHERE WE ARE

* **External Crunch score: 0.6268** (unchanged this session — no submission
  was made; nothing here cleared the promotion bar).
* **Champion: RT-600** (`RT-420` in the ledger, seven-specialist SCDF blend),
  development OOF **0.62581** on the canonical 5 dev folds. Re-verified exact
  this session: observed 0.625811, Δ +0.000001 — reproduction PASSED.
* **W7-D0 (exact loss cube):** the dominant remaining-loss region is current
  `t ≥ 200` AND positive break age `≥ 100` — **45.29% of exact remaining
  pairwise inversion loss** (prior inferred estimate 45.7%, approximately
  confirmed), 50.50% of pair weight, cell AUC 0.66428. Inside that cell,
  **never-break negatives carry 74.0% of the loss**, pre-break only 26.0%.
  Detail: `research/WAVE7_RT600_EXACT_ALPHA_BUDGET.md`.
* **W7-D3R (same-prefix vs. full-sequence diagnostic):** run on exactly that
  cell. **Arm B (same info, more tree capacity) did not beat Arm A** (1/5
  folds, mean −0.00592 cell AUC — consistent with the whole-dev-set
  comparison, B 0.61177 vs A 0.61605). **Arm C (same capacity + each series'
  own final-row features) beat Arm B on 5/5 folds**, mean **+0.07110** cell
  AUC, every fold ≥+0.053. **VERDICT: CASE 2 — future-information limit.**
  Detail: `research/reports/wave7_d3r.md`.
* **W7 teacher/distillation, one-fold pilot (fold 0) — `RT-992`/`RT-993`,
  OUTER-FOLD CONTAMINATED, NOT PROMOTION EVIDENCE.** Its apparent gains
  (`T1` +0.01621, `T2` +0.01671 whole-fold0 Δ) came partly from a nested-CV
  meta-feature leak: `Q` (global `RT-991` OOF) was cross-fitted per-row but
  not per outer-validation-fold. Kept on disk, relabeled, not deleted. See
  `research/WAVE7_TEACHER_NESTED_PREREG.md` §0.
* **W7 teacher/distillation, NESTED outer-fold-pure, full 5-fold —
  `RT-994`/`RT-995`, THE CURRENT AUTHORITATIVE RESULT.** Fold-purity
  sentinel confirmed 0/20 violations on the corrected scheme and 20/20 on
  the old one (proving it catches the exact defect) before any score was
  read. Whole-dev TS-AUC vs `T0` (`RT-990`), mean of 5 outer folds: `T1`
  (pure distillation) **+0.00349**, 3/5 folds positive, bootstrap CI
  `[−0.0047, +0.0109]` (crosses zero) — **fails promotion**. `T2` (0.5
  hard + 0.5 teacher blend) **+0.00943**, **5/5 folds positive**, bootstrap
  CI `[+0.0043, +0.0137]` (**entirely above zero**) — **clears all three
  measured promotion legs (magnitude, fold-consistency, bootstrap),
  reading MAJOR BREAKTHROUGH.** Contamination inflated the fold-0-only
  pilot by +0.0235 (`T1`, sign-flipping) and +0.0122 (`T2`) — real, but
  smaller than first measured. Improvement is broad-based (every age
  bucket, every current-`t` bucket), not narrowly concentrated. **Leg 4
  (alternate-partition stability) is authorized but not yet run — no
  promotion, no ensemble-integration test against `RT-600`'s specialists,
  no submission until it is.** Detail:
  `research/reports/wave7_teacher_nested.md`.

## WHAT THIS MEANS

The dominant loss cell is **not** a representation/extraction problem — more
tree capacity on the existing 500-column legal causal bank was tested
directly on this exact cell and made things mildly worse. It **is** an
information problem: each series' own eventual trajectory carries the signal
the causal prefix cannot see at `t≥200` with a break that is still only
weakly separated from noise.

## NEXT LANE

**`T2` (`RT-995`) is the funded priority.** It cleared 3 of 4 promotion
legs on the full nested 5-fold measurement (magnitude, fold-consistency,
bootstrap). Two things stand between it and a submission decision:

1. **Alternate-partition confirmation (leg 4)** — retrain/re-evaluate on
   `research/folds/folds_alt{1,2,3}.parquet` to confirm the effect isn't an
   artifact of the canonical partition. Authorized by
   `research/WAVE7_TEACHER_NESTED_PREREG.md` §6 (all three other legs
   cleared) but **not yet run** — comparable compute cost to the nested run
   just completed (~4-5 hours wall clock at this environment's pace).
2. **Ensemble-integration test.** `T2`'s standalone pooled TS-AUC (0.62108)
   is a single-model number, below `RT-600`'s seven-specialist ensemble
   (0.62581) — expected, not a red flag. The open question this project's
   own Rule (`research/HANDOFF_WAVE6.md` §3.2) requires answering before any
   promotion: does `T2` add ensemble-level alpha over `RT-600`'s existing
   specialists (or over a matched seed-clone), not just over the single
   `RT-990` control it was measured against here.

**Deprioritised — pure distillation (`T1`/`RT-994`) as a standalone
candidate.** The nested-clean result shows it doesn't reliably generalize
(3/5 folds, bootstrap crosses zero) — the contaminated pilot's apparent
strength was mostly measurement artifact.

**Deprioritised — horizon specialist / more capacity on existing features.**
Arm B already answered this question for this exact cell: no.

**Deprioritised — XGBoost/CatBoost.** Same reasoning as above; a different
tree library on the same 500 causal columns is unlikely to reverse a
capacity-doesn't-help finding, and the teacher lane is now the funded
priority.

## WHAT HAS NOT BEEN DONE

* **No alternate-partition confirmation** for `T2` (leg 4 — authorized, not
  run).
* **No ensemble-integration test** of `T2` against `RT-600`'s specialists or
  a matched seed clone.
* No XGBoost/CatBoost run (deprioritised — see above).
* No submission. `RT-990`/`RT-991` remain diagnostics (`RT-991` explicitly
  non-causal, never deployable). `RT-992`/`RT-993` are outer-fold
  contaminated, kept only as a measured mechanism-direction signal.
  `RT-994` does not clear promotion. `RT-995` clears 3/4 legs, pending leg 4
  and ensemble integration — **still no promotion or submission decision.**

## FILES THIS SESSION ADDED

```
research/scripts/wave7_d0_exact_loss_cube.py
research/WAVE7_RT600_EXACT_ALPHA_BUDGET.md
research/reports/wave7_rt600_exact_alpha_budget.json
research/WAVE7_D3R_PREREG.md
research/scripts/wave7_d3r.py
research/reports/wave7_d3r.md
research/reports/wave7_d3r.json
research/oof/RT-990.npy, RT-990.importance.csv
research/oof/RT-991.npy
```

**Added in the teacher-pilot follow-up session (`research/wave7-teacher-distillation`):**

```
research/WAVE7_TEACHER_PREREG.md
research/scripts/wave7_teacher_pilot.py
research/reports/wave7_teacher_diagnostics.{md,json}
research/reports/wave7_teacher_parity_check.json
research/reports/wave7_teacher_pilot.{md,json}
research/oof/RT-992.npy
research/oof/RT-993.npy
```

**Added after the outer-fold contamination was found and corrected (same branch):**

```
research/WAVE7_TEACHER_NESTED_PREREG.md
research/scripts/wave7_teacher_nested.py
research/reports/wave7_teacher_nested_fold_purity_test.json
research/reports/wave7_teacher_nested_outer{0,1,2,3,4}_{t1,t2}.json
research/reports/wave7_teacher_nested.{md,json}
research/oof/RT-994.npy, RT-995.npy
research/oof/nested_Q_outer{0..4}_inner{...}.npy  (20 inner-teacher checkpoints)
```
Ledger and ID map updated: `research/RDOF_LEDGER.md`, `research/EXPERIMENT_ID_MAP.md`.

**T2 promotion battery closed out (`research/wave7-t2-promotion`, forked from
`research/wave7-teacher-distillation@5093e0a`):** ensemble-integration test
against the actual RT-600 seven-specialist ensemble run (the leg every prior
status note flagged as outstanding). Result: `E2 − E1 = +0.00024`, MOSTLY
REDUNDANT — T2's standalone `+0.00943` single-model edge does not survive
contact with the ensemble; T2 correlates 0.71–0.91 with the seven
specialists, ~0.91 with a plain exchangeable seed clone. Alternate-partition
confirmation (leg 4) was not run given this result (see
`research/RDOF_LEDGER.md`). T2 is not promoted, not submitted. Full report:
`research/reports/wave7_t2_promotion_final.md`.
