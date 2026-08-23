# WAVE 7 — PRE-REGISTRATION

**Committed before any wave-7 TS-AUC exists. Training runs at commit time: 0.**

Deliberately short. A ten-minute pilot does not deserve a four-hour protocol
(brief §49). What is fixed here is what would otherwise be chosen after seeing a
score: the arms, the bars, the routing cutpoints and the controls.

Diagnostics that gate the ordering: `research/WAVE7_ALPHA_BUDGET.md` (W7-D1, W7-D2).
Both were computed and committed before any lane was designed, from
version-controlled artifacts only.

---

## Fixed for the whole wave

* **Learner configuration** is the champion's, verbatim: `learning_rate 0.05`,
  `num_leaves 63`, `min_data_in_leaf 200`, `feature_fraction 0.7`,
  `bagging_fraction 0.7`, `bagging_freq 1`, `lambda_l2 5.0`, `max_bin 127`,
  400 rounds. Held identical in every arm of every lane, so a delta is
  attributable to the lane. `tests/test_wave7_lanes.py` reads the pipeline's AST
  and fails if the two ever drift apart.
* **Features** are the seven production modules, 500 columns, unchanged.
* **Folds** are the canonical partition. Fold −1 (lockbox) and
  `folds_final10k.parquet` are not touched by any wave-7 object.
* **Scorer** is `sbr.metric.ts_auc_flat`, `weighting="pairs"`.
* **Age-bucket reporting is mandatory** for every candidate, not an appendix.
* **Seed-clone bar**: nothing is promoted on a blend delta that a same-strength
  LightGBM seed clone also produces. Wave 6 established this against a model with
  0.21 within-time rank correlation worth +0.0001.
* **No leaderboard-directed tuning.** 0.6268 is a calibration point, not an optimiser.

## Lane A — teacher → causal student (`wave7_a_distill.py`)

**Arms**, one canonical fold, no target-weight sweep:

| arm | target |
|---|---|
| A0 | `y[t] = 1[t ≥ τ]` — control |
| A1 | teacher `evidence` path |
| A2 | `0.5·y + 0.5·evidence` |

**Teacher bank, declared and closed**: `evidence` (graded break-by-now evidence
from full-sequence magnitude), `mag_total` (full-sequence severity), `permanence`
(does the displacement at `t` survive). Adding a target after a score exists is a
protocol violation. `SCALE = 4.0` in the evidence path and the 1/1/1 weighting in
`mag_total` are declared here and never fitted.

**Ordering, against brief §12**: evidence and magnitude lead; permanence is last.
Permanence is a young-age mechanism and ages 0–20 hold 13.6% of the weighted loss;
at age 100+, which holds 51.3%, permanence is not in doubt. Recorded with the
numbers in `WAVE7_ALPHA_BUDGET.md` §8.2.

**Legality.** Teacher quantities are labels, never features. Teacher targets are
computed from the full series of **training-fold series only** — asserted against
the fold table, not assumed. `--prove-deletion` deletes every teacher artifact
after training and asserts the student's predictions are bit-identical.

**Bar.** ≥ +0.003 aggregate on the pilot fold, or ≥ +0.005 in ages 0–20 with
credible aggregate potential. +0.0015 kills the lane.

## Lane B — horizon specialists (`wave7_b_horizon.py`)

**Regimes, log-spaced, fixed here and never re-cut against TS-AUC:**
H1 `[0,16)` · H2 `[16,41)` · H3 `[41,101)` · H4 `[101,251)` · H5 `[251,501)` · H6 `[501,∞)`.

**Pilot region: H4**, against brief §18's "early/mid where TS-AUC is weakest".
Weakest is not where the loss is — `t < 50` is 4.7% of the loss with a +0.018
ceiling. H4 is the earliest regime with material mass (20.3%) and enough rows for
a stable fit. H5 and H6 follow if H4 clears.

**Row-budget control.** The specialist gets the same row budget as the global arm,
so a gain is specialisation and not extra data.

**Overlapping routing**, if used, is the analytic Gaussian kernel in `log(1+t)`
in `wave7_lib.horizon_weights` with `width = 0.5` declared here. **No blending
weight is searched on validation.** Routing reads the current online index and
nothing else — no estimated break age, no inferred τ.

**Bar.** ≥ +0.005 conditional TS-AUC inside the pilot region before the six-regime
bank is built. Full-system promotion needs ≥ +0.003 aggregate on canonical CV with
≥ 4/5 folds positive; ≥ +0.005 preferred.

## Lane C — XGBoost (`wave7_c_xgb.py`)

Same features, rows, folds, labels, scorer. Only the learner changes. Parameters
declared once to **match** the LightGBM control's capacity (depth 6 ≈ 63 leaves,
same lr, same subsampling, same L2, same rounds), not to win a tuning contest. No
Optuna, no grid. Blends are equal-weight rank averages; no weight is searched.

**Bar.** ≥ +0.003 beyond the matched **seed-clone** control, or a standalone score
that materially beats LightGBM. A slower LightGBM is killed. CatBoost opens only
after this screen.

## Lane D — pairwise neural

Ten minutes, one fold, `RT-961` architecture unchanged, same-`t` pairwise
objective. Two separate verdicts required: did metric alignment repair neural
training, and is the repaired model remotely competitive. `0.58 → 0.60` closes the
competition lane regardless of how interesting the mechanism is.

## W7-D3 — dominant-cell diagnostic (added by the budget, runs first)

Not a model. Take the W6-E2R series-level protocol, restrict it to the dominant
joint cell (`t ≥ 200`, age ≥ 100 — 45.7% of remaining loss), remove the boundary,
ask the question row-wise, and measure how much of W6-E2R's +0.02354 survives.
Decides whether the wave is chasing +0.005 or +0.015. No promotion decision hangs
on it directly.

## Stopping rule

If lanes A, B and C all fail to produce a credible +0.003, **stop adding models**.
Return to the budget and ask whether the training sample's information limit has
been reached. Do not descend into calibration tuning, feature-subset searches or
0.0003 ensemble optimisation.

## Submission rule

≥ +0.004 robust internal improvement, ≥ 4/5 folds, seed/control check passed, alt
partitions supporting the direction. A materially different architecture with
unusually clean robustness may qualify at +0.003. Nothing below +0.003 consumes a
submission.
