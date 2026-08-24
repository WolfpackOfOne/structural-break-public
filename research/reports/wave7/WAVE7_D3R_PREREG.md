# WAVE 7 — W7-D3R PRE-REGISTRATION

**Written 2026-08-22 on `research/wave6-alpha`, committed BEFORE any D3R score
exists.** Population, arms, thresholds and interpretation are fixed here.
Nothing below may be changed after the first score is read, per the
commit-boundary discipline this project has followed since wave 5
(`research/WAVE5_PREREG.md`, `research/WAVE6_PREREG.md`,
`research/WAVE6_NEURAL_PREREG.md`).

**Question.** Is the dominant loss cell found in W7-D0 (t≥200, age≥100,
45.29% of exact remaining inversion loss) repairable with information already
legally available at time `t` — or does closing it require information the
online, causal, prefix-only system can never see?

---

## 1. POPULATION — FIXED FROM W7-D0, NOT RE-CHOSEN HERE

The evaluation population is committed to exactly the cell W7-D0 measured,
using the pair-weight machinery already built and verified in
`research/scripts/wave7_d0_exact_loss_cube.py`:

```
mask = (t >= 200) & ((y == 1 & age >= 100) | (y == 0))
```

i.e. every canonical dev row (folds 0–4) at online index `t ≥ 200`, keeping
**only mature positives** (post-break age ≥ 100) among the positives, and
**every negative** at that `t` (both never-break and pre-break — W7-D0 §G
showed both matter: never-break carries 74.0% of the cell's loss, pre-break
26.0%). All arms are scored with `sbr.metric.ts_auc_flat` restricted to this
mask, per canonical fold and pooled, and additionally split by negative type
for diagnosis. This is the exact same restriction, not a re-derivation — the
cell was fixed by W7-D0 before this document existed and is not being tuned
to flatter any arm.

**Cell pair-weight fraction of the whole dev set: 0.5050** (from W7-D0). This
is the translation factor used in §6: a cell-level TS-AUC delta ΔAUC_cell
implies an aggregate pooled delta of approximately `0.5050 × ΔAUC_cell`,
holding every other cell fixed.

---

## 2. ARMS

### Arm A — legal current-prefix baseline. **`RT-300`, reused, no new training.**

The existing single-champion OOF (`research/oof/RT-300.npy`): 500-column
causal bank (`m00_core, m01_seq, m02_dist, m03_dyn, m04_resid, m06_loc,
m07_bayes`), CHAMP protocol (1,000,000 train rows/fold, `num_leaves 63,
min_data_in_leaf 300, feature_fraction 0.5, bagging_fraction 0.7,
n_estimators 600, learning_rate 0.05, lambda_l2 5.0, max_bin 127,
num_threads 2`, `objective binary`), cross-fitted over the 5 canonical folds.
No future, no tau, no `n_online`. **Reused rather than retrained because it
was frozen and scored long before this cell existed as a question** — its
score on this cell carries no selection bias toward this diagnostic.

### Arm B — same information, stronger extraction. **`RT-990`, new training.**

**Identical input**: the same 500 causal columns, the same rows
(1,000,000/fold, same sampling), the same 5 canonical folds, the same seed
(0). **Only capacity changes**, to test whether the champion's own protocol
is leaving legal signal on the table:

| param | Arm A (`RT-300`) | Arm B (`RT-990`) |
|---|---:|---:|
| `num_leaves` | 63 | **127** |
| `min_data_in_leaf` | 300 | **150** |
| `feature_fraction` | 0.5 | **1.0** |
| `bagging_fraction` | 0.7 | **0.8** |
| `n_estimators` | 600 | **900** |
| `learning_rate` | 0.05 | 0.05 |
| `lambda_l2` | 5.0 | 5.0 |
| `max_bin` | 127 | 127 |
| `num_threads` | 2 | 2 |

No future, no tau, no `n_online`, no new columns. This is the "stronger
offline learner" arm from the brief (§15), kept to LightGBM with higher
capacity because XGBoost/CatBoost are not installed in the frozen environment
and installing a new package mid-diagnostic is out of scope for a ten-minute
screen — if B clears its bar, XGBoost becomes the natural §25 follow-up, not
a prerequisite for this diagnostic.

### Arm C — full training-sequence information, no true tau. **`RT-991`, new training.**

**Arm B's exact learner config**, applied to an **augmented 1000-column
input**: Arm B's 500 causal columns, **plus 500 more columns holding each
row's own series' feature vector AT THAT SERIES' FINAL ONLINE ROW**,
broadcast to every row of the series.

* Computed directly from the already-cached causal feature memmaps
  (`cache/features/m0*.npy`) — for series `s` with online rows
  `[r_0 .. r_last]`, every row `r_i` of series `s` receives the extra block
  `X[r_last, :500]` (`r_i`'s own 500 causal columns are untouched, unshifted,
  at their usual position).
* **This is legal for an offline diagnostic and illegal for production**: it
  uses the complete TRAINING sequence for a series (the whole online segment
  the causal engine has walked by the last row), but it never touches true
  `tau`, never encodes an explicit break boundary, never encodes true
  post-break age, and never reveals `n_online` beyond what "this is the last
  row" implies through the broadcast values themselves (not through a length
  feature). Same-t comparisons at the metric's core are unaffected — this
  only asks whether the destination the series' evidence eventually reaches
  helps rank the CURRENT row.
* **Held fixed vs Arm B**: same 5 canonical folds, same sampling target
  (1,000,000 rows/fold; row *count* is unchanged, only column count grows
  500→1000), same seed, same tree hyperparameters. The only lever that moves
  between B and C is the extra 500 columns, so `AUC(C) − AUC(B)` is
  attributable to future information, not to capacity or data volume.

---

## 3. WHAT IS NOT DONE

No true `tau`. No true post-break age as a feature. No `n_online` or any
function of final online length. No boundary indicator. No hyperparameter
search on any arm (B's and C's settings are fixed above, not swept). No
promotion decision — a diagnostic can select nothing; the outcome only routes
the next lane (§7).

---

## 4. FALSIFICATION / SCALE, FIXED BEFORE ANY SCORE IS READ

Cell-level AUC-delta thresholds (brief §18, adopted verbatim):

| Δ AUC (cell) | reading |
|---|---|
| < +0.003 | negligible |
| +0.003 to +0.010 | small/moderate |
| +0.010 to +0.020 | meaningful |
| > +0.020 | large |

Every cell-level delta is translated to an implied pooled TS-AUC delta via
`0.5050 × Δ AUC_cell` (§1) before being compared against the project's
existing promotion bar (+0.0030 pooled, ≥4/5 folds) — translation only,
**no promotion decision is made from this diagnostic**; a real promotion
requires the full battery in `FINAL_ARCHITECTURE_FREEZE.md` §1.

---

## 5. INTERPRETATION MATRIX, PRE-REGISTERED

| case | pattern | reading | next lane |
|---|---|---|---|
| **1** | B ≫ A, C ≈ B | current prefix already contains the signal; extraction is underpowered | horizon specialist → tree extraction (XGBoost/CatBoost) → teacher |
| **2** | B ≈ A, C ≫ B | current prefix is information-limited; future data resolves it | teacher/distillation → simulation; deprioritise horizon specialist |
| **3** | B ≫ A, C ≫ B | both extraction and future information matter | horizon specialist + teacher, controlled parallel |
| **4** | B ≈ A, C ≈ B | dominant cell is near-irreducible under available data | abandon the cell; recompute next-largest actionable class from W7-D0's cube |

---

## 6. WHAT GETS COMPUTED, WHATEVER THE ANSWER

For each arm: pooled cell TS-AUC, per-fold cell TS-AUC, cell AUC split by
negative type (never-break / pre-break), translated pooled-TS-AUC-equivalent
delta. `B − A` and `C − B`, both raw and translated. The CASE verdict follows
mechanically from §5 with no discretion once the numbers are in.

---

## 7. RUNNER

`research/scripts/wave7_d3r.py` — trains `RT-990` and `RT-991` via
`sbr.pipeline.run` (Arm A is not retrained), then scores all three arms on
the fixed §1 mask using the same `ts_auc_flat`/cube machinery as W7-D0.
Machine limit: at most two trainers concurrently (`research/HANDOFF_WAVE6.md`
§2.4) — B and C are run as two background trainers, not three, since A needs
no training.

---

## GIT

| | |
|---|---|
| starting SHA (after W7-D0) | `55643d0` |
| this pre-registration commit | see `git log -1` after commit |
| first D3R score generated | strictly after the pre-reg commit, never before |
