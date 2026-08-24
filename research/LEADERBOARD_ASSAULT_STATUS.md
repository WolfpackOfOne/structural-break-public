# LEADERBOARD ASSAULT STATUS

**Written 2026-08-23 on `research/wave6-alpha`, end of the W7-D0/D3R session.
Updated 2026-08-23 on `research/wave7-teacher-distillation` after the W7
teacher/distillation one-fold pilot.**
**Updated 2026-08-23 on `research/wave8-future-aware-distillation`: all five
Wave-8 mechanisms (SST/ORR/PCFB/CFEP/TGMC) ran real fold-0 pilots and were
all KILLED at their pre-registered gates. External score is still 0.6268 --
unchanged, no submission attempted. Full report: `research/reports/
wave8_final.md`. The only standing positive result anywhere in the pipeline
remains T2 (`RT-995`, +0.00943 dev-fold marginal, 3/4 promotion legs
cleared) from `research/wave7-teacher-distillation` -- not re-derived here,
cited by SHA. Highest-value next step: finish T2's remaining two legs
(alternate-partition, ensemble-integration) before proposing new
future-aware mechanisms.**
Read this before starting any further Wave-7 or Wave-8 work.

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
* **W7 teacher/distillation, one-fold pilot (fold 0):** teacher `Q` = `RT-991`
  reused (no retraining). `T1` (pure distillation) +0.01788 cell AUC / +0.01621
  whole-fold0 / +0.00903 translated aggregate (33.9% of D3R's fold-0
  future-information gap captured). `T2` (0.5 hard + 0.5 teacher) +0.02058
  cell / +0.01671 whole-fold0 / +0.01039 translated aggregate (39.0%
  captured). **Both clear all three pre-registered continuation gates by a
  wide margin. VERDICT: CONTINUE — one fold only, full 5-fold promotion
  battery not yet run.** Detail: `research/reports/wave7_teacher_pilot.md`.

## WHAT THIS MEANS

The dominant loss cell is **not** a representation/extraction problem — more
tree capacity on the existing 500-column legal causal bank was tested
directly on this exact cell and made things mildly worse. It **is** an
information problem: each series' own eventual trajectory carries the signal
the causal prefix cannot see at `t≥200` with a break that is still only
weakly separated from noise.

## NEXT LANE

**The one-fold teacher pilot cleared its continuation gate.** Recommended
next step: fund a full 5-fold run of `T1` and/or `T2` under new IDs
(`RT-994` onward — `RT-992`/`RT-993` are the pilot IDs and are not reused),
then run the full promotion battery
(`research/HANDOFF_WAVE6.md` §3.3 / `research/WAVE5_PREREG.md` §4: ≥+0.0030
mean TS-AUC over the strongest matched control, ≥4/5 folds positive, paired
series bootstrap CI supportive, alternate partitions stable) before this is
anywhere near a submission decision. **A single strong fold is evidence, not
proof** — fold 0 was, if anything, the *smallest* of D3R's five per-fold
future-information gaps, so it is not a cherry-picked favorable fold, but
5-fold confirmation is still required before any promotion claim.

**Deprioritised — horizon specialist / more capacity on existing features.**
Arm B already answered this question for this exact cell: no.

**Deprioritised — XGBoost/CatBoost.** Same reasoning as above; a different
tree library on the same 500 causal columns is unlikely to reverse a
capacity-doesn't-help finding, and the teacher lane is now the funded
priority.

## WHAT HAS NOT BEEN DONE

* **No full 5-fold teacher run.** `RT-992`/`RT-993` are a fold-0-only screen.
* No promotion battery (bootstrap, alternate partitions, ≥4/5 folds) run on
  any teacher arm.
* No XGBoost/CatBoost run (deprioritised — see above).
* No submission. `RT-990`/`RT-991` remain diagnostics (`RT-991` explicitly
  non-causal, never deployable); `RT-992`/`RT-993` are causal-at-inference but
  have cleared only a one-fold screen, not a promotion decision.

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
Ledger and ID map updated: `research/RDOF_LEDGER.md`, `research/EXPERIMENT_ID_MAP.md`.
