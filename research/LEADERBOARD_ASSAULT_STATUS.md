# LEADERBOARD ASSAULT STATUS

**Written 2026-08-23 on `research/wave6-alpha`, end of the W7-D0/D3R session.**
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

## WHAT THIS MEANS

The dominant loss cell is **not** a representation/extraction problem — more
tree capacity on the existing 500-column legal causal bank was tested
directly on this exact cell and made things mildly worse. It **is** an
information problem: each series' own eventual trajectory carries the signal
the causal prefix cannot see at `t≥200` with a break that is still only
weakly separated from noise.

## NEXT LANE

**Priority 1 — teacher / distillation**, targeting a continuous
`full_sequence_break_confidence` target built from each series' complete
training sequence (never true `tau`, never explicit boundary, never true
age — see `research/WAVE7_D3R_PREREG.md` §2 Arm C for the exact legal
boundary already exercised). `RT-991`'s OOF is a ready-made starting
representation for that teacher.

**Priority 2 — targeted simulation / probabilistic evidence accumulation**
for the same slow-evidence-buildup mechanism.

**Deprioritised — horizon specialist / more capacity on existing features.**
Arm B already answered this question for this exact cell: no.

**Before any teacher score counts for anything**, it must clear the
project's own pilot bar (one fold, ≥+0.003 fold TS-AUC, per
`research/WAVE7_PROPOSAL_metric_aligned_transition.md` and brief §24) before
earning a full 5-fold run, and any eventual candidate needs the full
promotion battery in `research/FINAL_ARCHITECTURE_FREEZE.md` §1 (≥+0.0030
mean, ≥4/5 folds, bootstrap CI clear of zero, alternate-partition support)
before it is anywhere near a Crunch submission.

## WHAT HAS NOT BEEN DONE

* No teacher target has been built or trained.
* No XGBoost/CatBoost run (deprioritised by the Arm B result — a different
  tree library on the same 500 columns is unlikely to reverse a
  capacity-doesn't-help finding).
* No submission. `RT-990` and `RT-991` are diagnostics; `RT-991` is
  explicitly non-causal and can never be one.

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
Ledger and ID map updated: `research/RDOF_LEDGER.md`, `research/EXPERIMENT_ID_MAP.md`.
