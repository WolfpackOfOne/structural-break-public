# WAVE 5 — PRE-REGISTRATION

**Branch `research/wave5-alpha`, parent `research/wave3-integration` @ `17bb5df`.**
Written 2026-08-21, **before any Wave-5 number was computed**. Binding.

`research/PROTOCOL.md` and `research/VALIDATION_V2.md` remain in force.

---

## 0. WHAT WAVE 5 IS FOR

LB-001 is **0.6268** on the Crunch public board, from the RT-600 artifact whose
development architecture scored **0.62581** on the canonical partition. Internal
→ external transfer was flat-to-positive. **Validation is not the problem any
more. Missing alpha is.**

So every Wave-5 candidate answers one question, and standalone AUC is not it:

> **Does this add information beyond a same-strength seed clone?**

Wave 3 (`m09_back`) and Wave 4 (W4-E6) both died on that question. A blend delta
is not evidence. Low within-timestep rank correlation is not evidence — a seed
change buys more decorrelation than a new feature family did.

## 1. IMMUTABLE

`research/wave3-integration`, RT-600, `submissions/C_ensemble_deployable.py`,
`models/final10k_ensemble`, `research/FINAL_REPRODUCIBILITY_MANIFEST.json`.
Not retrained, not rewritten, not merged into. Wave 5 does not merge itself.

## 2. THE VALIDATION SURFACE

**Permitted:** `research/folds/folds.parquet` folds 0–4 (8,000 dev series),
`folds_alt1/2/3`, `sbr.metric.ts_auc_flat` at official `pairs` weighting.

**Forbidden as a selection surface:** `RT-500..RT-506` (the all-10k OOF vectors —
those exist only for post-freeze calibration and are **not** a validation set),
`folds_final10k`, fold −1, `X_test.reduced`, `y_test.reduced`,
`y_test_index.reduced`, and the leaderboard.

## 3. REFERENCE ARMS (fixed here, not re-chosen later)

| symbol | set | members |
|---|---|---|
| `A` | single control | `RT-300` |
| `B` | seed-clone ensemble | `RT-300`, `RT-401`–`RT-406` |
| `S` | specialist ensemble (**the RT-600 architecture**) | `RT-300`, `RT-410`–`RT-415` |

Calibration for every arm: cross-fitted `SCDF_NSEEN` (`time_coord="log_n_seen"`),
12 anchors / 256-point grids / `min_n=400`, equal-weight mean. Frozen; not tuned.

## 4. PROMOTION STANDARD — applies to every family below

A candidate `C` is promoted only if **all four** hold:

1. `S+C` − (**strongest matched control**) ≥ **+0.0030** TS-AUC;
2. positive on ≥ **4/5** canonical folds;
3. paired series bootstrap (200 reps, common random numbers) CI supportive;
4. alternate partitions positive, or at minimum directionally stable.

**The matched control is not `S`.** It is `S` plus a same-strength stream
carrying no new information — a seed clone of a champion-configuration booster,
or where the candidate is a feature block, `S` plus a booster trained on the
incumbent 500 columns with a fresh seed. A candidate that beats `S` but not
`S + seed clone` has demonstrated bagging, not alpha, and is REJECTED.

Interpretation bands (statistical controls override the labels):
`<+0.001` noise · `+0.001..+0.003` exploratory · `+0.003..+0.006` meaningful ·
`+0.006..+0.010` major · `>+0.010` breakthrough.

## 5. STAGE GATES

**A** mechanism/causality/runtime on cached or screen data → **B** 1–2 folds →
**C** full canonical 5-fold OOF → **D** matched seed control → **E** alternate
partitions → **F** promotion. Every rejection is recorded, in
`research/FAILED_EXPERIMENTS.md`, with its number.

**Causality is a gate, not a claim.** Every new module must pass
`check_prefix_invariance` at `atol=0.0` on ≥8 series of different lengths,
including the shortest in the dataset, and the output is pasted into the report.

## 6. THE PRE-REGISTERED EXPERIMENTS

### W5-E1 — specialist / bagging mixture *(no training; cheap, runs first)*
**Hypothesis.** W4-E6 rejected the 13-booster *union*, whose mechanism was
established: under an equal-weight mean the union gives the `RT-100R`
configuration 54% of the blend weight. A *small* bagging component does not do
that (λ=0.9 gives it 22.9%), so the question W4-E6 answered is not this one.
**Mechanism.** Variance reduction from `B` on top of the specialists' error
diversity, at a weight that does not swamp the composition.
**Grid, fixed here, not to be enlarged:** λ ∈ {1.00, 0.90, 0.80, 0.70},
score = λ·`S` + (1−λ)·`B`, both arms cross-fitted before mixing.
**Falsification.** No λ<1 beats λ=1 by ≥ +0.0010 on ≥4/5 folds → reject, move on.
**Promotion to the artifact** still requires the §4 bar.

### W5-E2 — `m10_persist`: transient shock vs persistent break
**Hypothesis.** The competition asks whether a break *has occurred and persists*.
A transient outlier, a volatility burst or a temporary level shift moves the
incumbent one-sample evidence bank the same way a permanent shift does, and
nothing in the bank distinguishes them. Features that measure whether evidence
*stayed* elevated should reject exactly those negatives.
**Mechanism.** Peak-vs-current evidence, time since peak, decay rate after peak,
fraction of trailing windows still elevated, short-vs-long horizon agreement,
sign consistency, post-alarm persistence, recovery toward the historical centre,
run length, repeated exceedance counts — all causal, all from the online prefix.
**False signal.** A genuine break detected late looks "recently peaked"; a slow
drift never peaks. Companion: break-age-conditioned diagnostics.
**Falsification.** Fails the §4 bar, or the gain does not sit in the negatives
the persistence features were built to reject (checked against W5-D2).

### W5-E3 — hard-negative curriculum
**Hypothesis.** Ranking loss is dominated by a minority of no-break series that
the champion scores high. Increasing their training emphasis improves the
within-timestep ranking more than it costs on young true breaks.
**Mechanism.** Cross-fitted mining: hardness for a series in fold *k* is
estimated **only** from models that never saw fold *k*. Labels are never changed.
**Variants, fixed here:** uniform (control) · reweighting · oversampling.
**Falsification.** §4 bar; and it must not degrade the 0–20 age buckets.
**Leakage risk, and the gate for it.** Mining that sees its own validation fold
manufactures a result. Fold purity is asserted in code, not assumed.

### W5-E4 — robust distribution distances (`RT-903/906` translation)
Wasserstein · energy · KS/PIT · Cramér-von Mises · Brown–Forsythe-like robust
scale · tail-probability shift · quantile displacement · |x| distribution shift.
Online prefix and trailing windows vs the historical distribution, plus
residualised variants. Streaming/incremental only — **no O(n²) recomputation**.
Repeated-testing behaviour null-calibrated on historical windows.
**Falsification.** §4 bar, or fully redundant with `m02_dist` under leave-one-
block-out.

### W5-E5 — residual CUSUM / CUSUMSQ / AR break (`RT-907` translation)
Residual CUSUM, CUSUMSQ, residual-variance LR, AR-coefficient drift proxy,
coefficient stability, lag-product cumulative evidence, prediction-error variance
change, residual ACF change. **AR parameters fitted on historical break-free data
only**; fixed small AR family or pooled order — never per-series online selection.

### W5-E6 — dependence likelihood ratio
Compact LR bank for changes in lag-1/lag-2 correlation, a short AR vector,
innovation variance and persistence: instantaneous LR, cumulative LR, decayed
peak, persistence, and a cheap candidate-τ max/softmax. Null-calibrated on
historical subsegments.

### W5-E7 — AR(p)-FOCuS / sequential localisation
**Hypothesis.** The incumbent bank uses *fixed windows*, which dilutes early
evidence relative to maximising over candidate τ.
Exact/near-exact max over τ of the normalised likelihood improvement — mean
first, then scale / residual mean / lag product. Functional pruning if
practical, otherwise a transparent coarse grid **with its approximation error
quantified**. Outputs: max evidence, inferred changepoint age, second-best
evidence, localisation confidence, evidence persistence. **Runtime benchmarked;
a detector that breaks the inference budget is not promoted.**

### W5-E8 — absorbing-state BOCPD
**Not** textbook BOCPD. A textbook run-length posterior *resets* after a change
and therefore answers "is there a change at t?", when the question is **"has a
break already occurred by t?"** The target state is conceptually absorbing:
pre-break → post-break. Expose `P(break has occurred by t)`, hazard, run-length
posterior, heavy-tailed predictive, variance-change sensitivity.

### W5-E9 — pairwise / TS-AUC objective
One or two principled alternatives only, **no objective HPO**: sample
positive–negative pairs at the same `t`, weight by the official `n_pos·n_neg`
structure, optimise a ranking margin / logistic loss. Compared against the
incumbent `binary` and `pairwise_t` streams under the exact official scorer.

### W5-E10 — best combined candidate
Only families that individually cleared §4. No best-of composition search; the
W4-E6 lattice stays excluded.

## 7. DIAGNOSTICS REQUIRED OF EVERY PROMOTED CANDIDATE

* **Break age** 0–5 / 5–10 / 10–20 / 20–50 / 50–100 / 100+, baseline vs
  candidate vs delta, same eligible samples. Early evidence matters and must not
  be bought by wrecking mature ranking.
* **Hard negatives** top 1% / 5% / 10% highest-scoring no-break series, with a
  mechanism taxonomy (transient shock, heavy tail/outlier, variance burst, local
  trend, dependence fluctuation, spectral change, other). Transparent summaries
  and representative cases — no hand-labelling of thousands of series.
* **Break family** on the existing taxonomy, labelled heuristic where it is.
* **Ensemble contribution**, not gain importance: standalone strength,
  leave-one-block-out, time-bucketed permutation, specialist blend delta,
  **matched seed-control delta**, correlation with existing streams.

## 8. WHAT IS FORBIDDEN

`n_online` or final online length in any form · future data · cross-series live
state · within-timestep rank oracles in a production candidate · validation-fold
leakage · global feature selection using validation labels ·
leaderboard-directed fitting · continuous optimisation of ensemble weights
(equal weighting is the incumbent virtue; only the §6 W5-E1 grid is licensed) ·
`nogit` or undocumented runs · post-hoc experiment IDs.

## 9. LEADERBOARD POLICY

LB-001 = **0.6268**, immutable. **LB-002 is earned, not spent.** It requires
either (A) ≥ +0.0030 canonical, ≥4/5 folds, seed control passed, alternate
partitions confirming; or (B) overwhelming evidence for a theoretically
important architecture. Then: freeze → fit all 10k → build artifact → causality
gates → official `crunch test` → submit **once**. Wave-5 design is not revised
retrospectively on what LB-002 returns.

## 10. ID NAMESPACE

`W5-E*` for experiments; `RT-6xx` reserved (RT-600 is the shipped artifact), new
canonical training runs take **`RT-7xx`**, verified unused before allocation.
Every run reaches `research/RESULTS.csv` through `sbr.pipeline.run` with a real
git SHA and a real seed.
