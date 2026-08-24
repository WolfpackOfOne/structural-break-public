# WAVE 7 — W7-D0: EXACT RT-600 PAIRWISE-INVERSION LOSS CUBE

**Written 2026-08-22 on `research/wave6-alpha`.** Diagnostic only — no training,
no new model, no candidate selected. Pure recombination and exact re-scoring of
`research/oof/wave5_S_specialist.npy`, the RT-600 development architecture's
cross-fitted OOF vector (documented as `RT-420` / "seven specialists, SCDF" in
`research/HANDOFF_WAVE6.md` and `research/RDOF_LEDGER.md`, 0.62581 canonical).

Where this document and `research/WAVE7_ALPHA_BUDGET.md` (if one is later
written from the inferred/approximate method) disagree, **this document is
authoritative** — it is computed from the official pairwise rank machinery in
`src/sbr/metric.py`, not from a marginal-bucket-AUC approximation.

Runner: `research/scripts/wave7_d0_exact_loss_cube.py`.
Evidence: `research/reports/wave7_rt600_exact_alpha_budget.json`.

---

## A. DATA AVAILABILITY

Real store: **YES**. `cache/store` (symlinked from
`structural-break-claude-wave3`), `research/folds/folds.parquet` (canonical 5
dev folds), `research/oof/wave5_S_specialist.npy` all present and used as-is.
No synthetic data anywhere in this result.

## B. RT-600 REPRODUCTION

| | mean | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---:|---:|---:|---:|---:|---:|
| expected (`RT-420`, HANDOFF_WAVE6 / RDOF_LEDGER) | 0.62581 | 0.63828 | 0.62040 | 0.63392 | 0.61750 | 0.61894 |
| observed (this run) | 0.625811 | 0.638285 | 0.620401 | 0.633929 | 0.617507 | 0.618939 |
| delta | **+0.000001** | +0.000005 | +0.000001 | +0.000009 | +0.000007 | -0.000001 |

**PASS** (tolerance ±1e-4). This is the exact array the champion's headline
number was computed from; the cube below is built on it with no re-derivation.

## C. EXACT REMAINING LOSS (pooled canonical dev, folds 0–4 combined)

Computed from `sbr.metric.ts_auc_flat`'s own per-timestep numerator/denominator
(`num_t` = concordant-pair-equivalent count with exact mid-rank tie handling,
`den_t = n_pos(t)·n_neg(t)`), summed over every eligible timestep across all
five folds — i.e. the exact complement of the official metric, not an
approximation of it.

| | value |
|---|---:|
| pooled TS-AUC | 0.625627 |
| total pair weight (`Σ den_t`) | 3,381,921,039 |
| total inversion-equivalent loss (`Σ (den_t − num_t)`) | 1,266,100,176.5 |
| loss fraction | 0.374373 |

## D. EXACT LOSS BY CURRENT t AND POSITIVE AGE (pair weight, per neg-type)

Full 6×6×2 cube (pair weight per cell) — see table in the script's stdout and
`cell_rows` in the JSON for pair weight, inversion loss and cell AUC per cell.
Cross-check: cube totals reproduce the pooled scorer exactly
(`3,381,921,039` weight, `1,266,100,176.5` loss — every dev pair falls in
exactly one cell, confirmed to <1 pair-unit).

The two largest single age×negtype rows by weight are both **age 100+**:
`never_break × 100+` (1,412M pair-weight-units across all t) and
`pre_break × 100+` (521M). Age 100+ alone carries roughly 57% of total pair
weight, consistent with the project's earlier finding that 68%+ of positive
rows are age 100+.

## E. EXACT t × AGE × NEGATIVE-TYPE CUBE — DOMINANT CELL

**Region: current t ≥ 200 AND positive break age ≥ 100.**

| | value |
|---|---:|
| fraction of total pair weight | **0.5050** |
| fraction of total inversion loss | **0.4529** |
| cell AUC | 0.66428 |
| perfect-repair ceiling, pooled TS-AUC | 0.79518 (+0.16956 over observed) |

**Never-break vs pre-break split inside the dominant cell:**

| | loss share | weight share |
|---|---:|---:|
| never-break negative | **0.7399** | 0.7425 |
| pre-break negative | 0.2601 | 0.2575 |

## F. THE 45.7% CLAIM

Prior inferred value: 45.7% of remaining weighted loss.
**Exact measured value: 45.29% of remaining inversion loss** (50.50% of pair
weight, at cell AUC 0.66428).

**Verdict: APPROXIMATELY CONFIRMED.** The exact figure (45.29%) sits within
0.4 points of the inferred 45.7% — well inside the ±2-point band that would
count as a clean confirmation, and nowhere near a rejection. The earlier
marginal-AUC-based estimate was not wrong in magnitude; the joint cell really
is the single largest reservoir of remaining loss, at almost exactly the scale
previously believed.

## G. NEVER-BREAK VS PRE-BREAK ATTRIBUTION

**Never-break negatives dominate the dominant cell's loss (74.0%)**, not
pre-break negatives (26.0%). Per the interpretation rule this project already
committed to: **the model still cannot separate genuine persistent weak shifts
from truly stable series, even after the break has been present for 100+
online observations.** This is not primarily a baseline-heterogeneity /
hard-stable-prefix problem (which would show up as pre-break dominance); it is
a discrimination problem against series that never break at all. That argues
for representation/evidence-accumulation work (long-horizon persistence
features, cumulative weak-signal channels) over baseline-normalisation work.

## H. LEADERBOARD CAPTURE REQUIRED (external anchor 0.6268, dev pooled 0.625627)

Fraction of the dominant cell's own inversions that would need to be repaired
to reach each target, holding everything else fixed:

| target | needed Δ TS-AUC | % of dominant-cell inversions | within cell ceiling? |
|---|---:|---:|---|
| 0.630 | +0.00320 | **1.89%** | yes |
| 0.635 | +0.00820 | **4.84%** | yes |
| 0.640 | +0.01320 | **7.79%** | yes |
| 0.645 | +0.01820 | **10.73%** | yes |
| 0.650 | +0.02320 | **13.68%** | yes |

All five targets sit far inside the cell's 100% ceiling (which itself would be
worth +0.1696 pooled TS-AUC if achieved perfectly, i.e. wildly unrealistic).
The practical reading: even a small, real capture rate inside this one cell —
on the order of a few percent of its inversions — is competitively meaningful.
This is the number D3R exists to bound: how much of that few percent is
actually reachable with information already available at time `t`, versus how
much requires future information.

## I. WHAT THIS DOES NOT YET ANSWER

W7-D0 is a measurement of *where* the loss is, not of *whether* it is
repairable. Per the brief's own gate: no horizon specialist, teacher,
XGBoost, or other lane may start before **W7-D3R** (three-arm same-prefix vs.
full-sequence diagnostic) is pre-registered and run on exactly this cell
(t≥200, age≥100, weighted toward never-break negatives per §G). That
pre-registration is the next deliverable, not yet written as of this
document.

---

## GIT

| | |
|---|---|
| starting SHA (`research/wave6-alpha` before this work) | `33fc210` |
| this commit | see `git log -1` after commit |
| tree state | clean except this wave's new files |
