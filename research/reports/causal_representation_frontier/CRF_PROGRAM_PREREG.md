# CRF PROGRAM PREREGISTRATION — CAUSAL REPRESENTATION FRONTIER

**STATUS: `DESIGNED_NOT_EXECUTED`.**

No experiment below has been run. No `RT-xxx` id is allocated by this document.
No score, no OOF vector, no TS-AUC, no `marginal_vs_clone` exists for any CRF
candidate. `research/RESULTS.csv` is byte-identical to its state at the base
commit (sha256 `5b34c564e69f502c4a54d4ba1b702b400893358073e1897cd82453c83215512c`).

Base: `origin/research/new-avenues-pilots-2026` @
`b47b22ad7e0b85fe977cb65453ae8531227d414d`.
Evidence base: `CAUSAL_REPRESENTATION_FRONTIER.md`,
`prior_representation_audit.csv`, `representation_collision_matrix.csv`,
`SOURCES.md`, all in this directory.

**Each experiment below requires its own EXECUTION preregistration commit,
committed before any score exists, allocating ids from
`research/EXPERIMENT_ID_MAP.md` — exactly as `SS01_EXECUTION_PREREG.md` did.
This document freezes the design; it does not authorise a run.**

---

## 0. PROGRAM-LEVEL RULES (bind all three experiments)

### 0.1 Screen thresholds

| band | `marginal_vs_clone` (fold 0, vs `E1 = RT600 + RT-401`) |
|---|---:|
| KILL | `< +0.0015` |
| WEAK | `+0.0015` … `+0.0030` |
| SERIOUS | `≥ +0.0030` |
| MAJOR | `≥ +0.0050` |

Serious confirmation (required before any promotion discussion):
mean five-fold `marginal_vs_clone ≥ +0.0030`; **≥ 4/5 folds positive**; positive
dominant-cell pair flow; no control failure; deployable runtime measured, not
asserted.

### 0.2 Cheap abandon gate — evaluated FIRST, before any five-fold spend

A candidate whose fold-0 **standalone whole-fold TS-AUC < 0.600 while
within-timestep ρ vs the RT600 blend ≤ 0.60** is abandoned immediately; no
further folds, no controls beyond those already run.

Justification, fixed here before any CRF score exists: over the 17 scored arms
with both quantities recorded, `corr(standalone, ρ) = +0.983`, and a descriptive
OLS places the `+0.0030` contour at standalone ≈ 0.608 (ρ = 0.2) to ≈ 0.641
(ρ = 0.5). `RT-1201` measured 0.584 at ρ = 0.38 and returned `+0.000301`. This is
a **necessary-condition filter set deliberately below the fitted contour**, not a
promotion criterion, and it may not be relaxed after a score is seen.

### 0.3 Mandatory reporting for every CRF candidate

Every arm reports, on the canonical fold-0 sample, before any verdict is written:
whole-fold repairs / damage / net; dominant-cell repairs / damage / net;
mature-vs-never net; mature-vs-prebreak net; within-`t` rank correlation with
RT600; unique repair coverage; damage rate on RT600-correct pairs; and
standalone whole-fold and dominant-cell TS-AUC.

Pair flow is a **binding** diagnostic, not colour: `FIRST_SWEEP_SYNTHESIS.md`
measured dominant-cell pair net as the strongest non-tautological correlate of
`marginal_vs_clone` (Pearson 0.879). Low ρ alone is not a credential
(`RT-970`: ρ 0.21, value +0.0001).

### 0.4 Label / τ rules

True `τ` is **never** an inference feature. It is used only to construct the
training label `y[t] = 1[t ≥ τ]` and for post-hoc age-bucket diagnostics, both of
which `PROTOCOL.md` permits. Post-break **age** may be used for training-time
diagnostics and for reporting; it may **not** weight the loss, route a row, gate a
blend, or reach a scorer in any form. `n_online`, the final horizon, any
boundary-conditioned availability, and any quantity whose *missingness* depends on
`τ` are forbidden — `RT-900` is the standing example (0.86552 of pure leak through
a missingness mask).

### 0.5 Fold purity — what may be fitted where

| object | fitted on |
|---|---|
| history-derived per-series constants (ECDF knots, AR coefficients, residual-ECDF knots, exceedance threshold, EWMA seed) | **that series' own history only.** Legal at inference (the history is complete at `t = 0`) and trivially fold-pure. |
| global channel clipping scales / any global standardiser | **training-fold rows only**, frozen, then applied everywhere. Asserted in code, not by convention. |
| the sequence encoder and its head | **training folds only** for that outer fold |
| CRF-02's null pretraining | **training-fold series' histories only**, per outer fold. It may NOT be pretrained once globally and reused across folds. |
| CRF-02's history embedding `h_i` | produced by the outer-fold-pure encoder; for validation series, produced by the training-fold-fitted encoder applied to that series' own history (a forward pass, never a fit) |

**Learn from the Wave-7 nested-teacher contamination incident.** A fold-purity
sentinel must run and pass **before** any CRF number is read, in the shape of
`tests/test_wave8_causality.py`'s nested-scheme purity test: set arithmetic *and*
a live call through the actual fitting path, plus a positive control that
reproduces the contaminated scheme and shows it differs.

Additional purity assertions required by this program:

1. `assert_prefix_invariance(channels, atol=0.0)` over ≥ 8 series of different
   lengths, bitwise — the `PROTOCOL.md` §3 contract.
2. `assert_no_forbidden_columns` over every input name, reusing
   `wave8_common.assert_no_forbidden_columns`.
3. A truncation test: rebuilding the channels and re-running the encoder on a
   truncated online segment reproduces the surviving rows to ≤ 1e-8.
4. A batch-composition test: a series' score does not depend on which series share
   its minibatch (Wave-6 Gate 5, already implemented).
5. `finite(lockbox rows) == 0` on every emitted OOF vector.

### 0.6 Calibration

Standalone TS-AUC is computed on the **raw** sequence score. No calibration is
applied and none is needed: TS-AUC compares only within a timestep and is
invariant to any monotone transform applied identically inside a timestep.

For ensemble integration, the canonical cross-fitted
`sbr.production.calibration.SmoothTimeCDFCal`, `kind="scdf"`,
`time_coord="log_n_seen"`, 12 log-spaced anchors, 256-point quantile grids,
`min_n=400` is applied — **frozen exactly as the seven production streams use it,
and applied identically to the candidate and to every control.** No new
calibration layer is invented. If a candidate appears to need one, that is a
finding to report, not a change to make.

### 0.7 Ensemble integration rule

Unchanged from every prior wave, so the numbers stay comparable:

```
E0 = RT600 seven specialists, equal weight, cross-fitted SCDF
E1 = E0 + RT-401 (matched exchangeable seed clone), equal weight over eight
E2 = E0 + candidate,                                equal weight over eight
marginal_vs_clone = E2 − E1
```

No weight search. No stacking. No subset selection. All three are closed lanes.

### 0.8 Compute budget

≤ 15 h total screening compute before any serious confirmation. One primary
architecture per hypothesis. **No Optuna, no architecture search, no width sweep,
no learning-rate sweep, no seed fishing, no re-rolling a diverged fold.**

### 0.9 Prohibited post-score changes (applies to all three)

After the first CRF number exists, the following may not be changed for that
experiment: architecture, hidden dimension, layer count, dilation schedule,
receptive field, channel set, optimizer, learning rate, weight decay, schedule,
epochs, batch construction, pair sampler, `m_neg`, pair weighting, seed, early
stopping rule, calibration, ensemble rule, or any gate threshold in this document.
A change to any of them creates a **new experiment with a new id and a new
execution preregistration**, and the original result stands in `RESULTS.csv`.

No `SS-0Xb`-style repair arms. No "one more variant". If an arm fails, it fails.

### 0.10 What the program may not do

No submission. No lockbox read. No `X_test.reduced` read. No modification to
`production/rt600` or any production artifact. No merge into `research/current` or
`research/new-avenues-pilots-2026`. No edit to an existing `RESULTS.csv` row. No
repair of the 15 pinned known failures to make a suite green.

---

## 1. CRF-01 · NNCSR — NULL-NORMALIZED CAUSAL SEQUENCE RANKER

**Unconditional. Primary. The designated stop experiment.**

### 1.1 Central hypothesis

The 500-column bank discards temporal order beyond lag-2 products
(`NEW_AVENUES_2026.md` §C.4) and has no representation of excursion contiguity or
excursion-growth rate (§C.1, D4/D5). A causal sequence encoder reading a **small,
null-normalised** channel set, trained under an objective that matches the
metric's within-timestep comparison, can represent those functionals and produce a
same-`t` ranking direction that RT-600 does not already contain.

### 1.2 Information channel

Temporal structure of the history-referenced innovation stream — specifically the
*duration and growth* of null-band exceedances, which under stationarity grows
like `log t` and under a persistent regime change grows like `t`
(`NEW_AVENUES_2026.md` §E.4).

### 1.3 Exactly why prior attempts do not falsify it

| prior | why it does not close this |
|---|---|
| `RT-970/971` W6 TCN | same architecture, but channels were `(x − median H)/IQR H` — **location and scale only**, no PIT, no innovation, no exceedance, history compressed to two scalars; **plus** an `elapsed = log1p(t)/7` channel handing the network the exact rowwise shortcut; **plus** BCE fixed by preregistration. The Wave-6 report states in §L that it cannot distinguish "family wrong" from "objective wrong". |
| `RT-111`, `RT-700`, `RT-701`, `RT-123`, `RT-A09-*` | all same-`t` ranking over a **fixed** 500-dimensional static vector with gradient-boosted trees. None learned a representation. `RT-700` specifically closes *metric-shaped pair weighting*, which this design does not use. |
| `RT-1042` CFEP | frozen encoder, future-summary target, 16 columns appended to the saturated bank, LightGBM BCE head. CRF-01 has no pretraining, no future target, no static bank, and the encoder **is** the model. |
| `RT-1202` Pilot 4 | zero learned parameters; hand-specified nearest-neighbour distances collapsed to one fixed scalar. |
| `RT-1201` Pilot 3 IM2 | hand-specified nine-feature dwell scalar. It sets the **bar** (0.610 cell AUC at ρ 0.38 → +0.0003), not the ceiling; the hypothesis here is that a learned run functional beats a hand-specified one, which D6 independently supports (the naive Erdős–Rényi normalisation was *worse* than the raw longest run). |
| `RT-1223` SS-02 | a bounded correction on RT600 whose top two features by gain were `rt600_cal` and `rt600_logit`. CRF-01 never sees RT600. |
| W7-D3R Arm B | more tree capacity on the static bank. Different representation entirely. |

### 1.4 Inputs — FROZEN

Eight channels, all fitted on the series' own break-free history `H_i` and
therefore bitwise prefix-invariant by construction:

| # | name | definition |
|---|---|---|
| 1 | `pit` | `clip(Φ⁻¹(F̂_H(x_t)), ±4)`, `F̂_H` = 256-knot linear-interpolated history ECDF with `(r − 0.5)/n` plotting positions |
| 2 | `inn` | `(x_t − Σ_{k=1..5} φ_k x_{t−k}) / σ_H`, with `φ` and `σ_H` from Yule-Walker on `H` only; rows `t < 5` use the historical tail as the lag source |
| 3 | `inn_pit` | `clip(Φ⁻¹(Ĝ_H(inn_t)), ±4)`, `Ĝ_H` = 256-knot ECDF of the AR(5) residuals **on the history** |
| 4 | `abs_inn_pit` | `|inn_pit|` |
| 5 | `vol_norm` | `clip(inn_t / max(EWMA_{hl=32}(|inn|)_{t−1}, 0.25·mad_H), ±8)`; the EWMA is seeded from `H`'s residual MAD and is strictly lagged |
| 6 | `surp` | `clip(−log(1 − Ĝ^{abs}_H(|inn_t|) + 1/(n_H+1)), 0, 12)` |
| 7 | `exceed` | `σ(4·(|inn_t| − q90_H)/mad_H)`, a soft exceedance indicator against the history's own 90th percentile |
| 8 | `lag1_pit` | `clip(pit_t · pit_{t−1}, ±16)`, with `pit_{-1} = 0` |

**Forbidden and unreachable from this input path:** raw `x` other than through the
above; any of the 500 columns; RT600 or any specialist score; any first- or
second-sweep candidate score; true `τ`; `n_online`; final horizon; `elapsed` or any
monotone function of `t`; any cross-sectional quantity.

`elapsed` is removed from **both** the candidate and the control, so the pair
remains matched.

### 1.5 Sequence representation and architecture — FROZEN

Identical to `RT-970` except the input channel count:

dilated causal TCN · kernel 3 · dilations 1/2/4/8/16/32 · six residual blocks ·
GELU · weight-normed left-padded causal convolutions · dropout 0.1 ·
hidden 32 · per-timestep linear head to one scalar · receptive field **127** ·
≈ 35.7k parameters · **no BatchNorm, no time-axis normalisation, no recurrent
layer** · CPU only, `torch.use_deterministic_algorithms(True)`, 6 threads,
`PYTHONHASHSEED=0`, seed 0.

Optimisation, also identical to `RT-970`: AdamW lr 3e-3, weight decay 1e-2, cosine
schedule over all steps, batch = 32 series (length-bucketed, legal per Gate 5),
20 epochs, gradient clip 1.0, no early stopping.

Rationale for holding the architecture exactly: it makes CRF-01 vs `RT-970` a
clean two-factor comparison (channels, objective) and it inherits five already-
passing causality gates and a measured runtime. Receptive-field extension and SSM
architectures are **follow-ups conditional on evidence that RF 127 binds**, not
part of this screen.

### 1.6 Training target and objective — FROZEN

Target: `y[t] = 1[t ≥ τ]`, the standard row-level label. Permitted as the
supervised target by `PROTOCOL.md`.

**Objective (candidate arm): same-`t` pairwise logistic.**

- Group = online index `t`. Pairs drawn **only inside a group** and **only from
  training-fold series**.
- `m_neg = 8` negatives per positive, matching the existing
  `sbr.pipeline._make_pairwise_t` convention.
- Loss `= mean over sampled pairs of softplus(−(s_pos − s_neg))`. Plain logistic
  on the score difference. **No margin term, no tie-aware term** — TS-AUC uses
  mid-ranks and continuous network outputs make exact ties measure-zero.
- **Uniform pair weighting.** The metric's own `n_pos(t)·n_neg(t)` weighting is
  *not* used: `RT-700` tested exactly that on trees and lost 0.00147. A weighted
  variant is **not** a declared arm and may not be run under this preregistration.
- Timesteps with only one class present contribute zero weight, as in the metric.
- Minimum group occupancy: a timestep contributes only if it has ≥ 1 positive and
  ≥ 8 negatives among the batch's training series at that `t`.
- Batch construction: series-major (as `RT-970`), with pairs formed **within the
  batch** at each `t`. Batch size 32 series is unchanged, so group occupancy is
  the binding constraint and is reported.

**Objective (control arm): masked BCE, uniform over rows** — identical
architecture, identical channels, identical optimiser, identical seed. This is the
single mandatory control and it isolates the objective.

### 1.7 Controls

| control | isolates | status |
|---|---|---|
| **C1 — same architecture + same channels + BCE** | **objective** | **MANDATORY** |
| C2 — same architecture + same objective + temporally shuffled channels within each series | temporal representation | **DECLARED, CONDITIONAL** — runs only if the candidate clears §0.2's abandon gate, and only on fold 0 |
| `RT-401` exchangeable seed clone | the ensemble bar | already exists; no compute |

No third control. Per the brief's §31, one or two decisive controls maximum.

C2's construction, fixed now so it cannot be chosen later: permute the online time
index within each series with a per-series seed derived from the series id,
applying the *same* permutation to all eight channels, leaving the label at each
original `t` in place. This destroys temporal order while preserving every
channel's marginal distribution.

### 1.8 Folds, streaming state, runtime

Folds: canonical `research/folds/folds.parquet`, series-level, never regenerated.
Fold-0 screen first; five folds only on clearing §0.2 and a fold-0
`marginal_vs_clone ≥ +0.0015`.

Streaming state per series ≈ **6 KB** (8 channels × 127-sample ring buffer plus
history constants). `fit_history(H)` computes the ECDF knots, AR(5) coefficients,
residual-ECDF knots, `q90_H`, `mad_H`, EWMA seed — **once**, at `t = 0`.
`update(x_t)` costs eight channel updates plus one incremental causal pass.
Implemented against `research/scripts/novel_streams/harness.py::StreamingMechanism`.

Budget: channel build ≈ 0.8 h; fold-0 candidate + C1 ≈ 1.0 h; five folds ≈ 2.5 h.

### 1.9 Gates

| gate | threshold |
|---|---|
| abandon (checked first) | standalone fold-0 whole TS-AUC ≥ 0.600 at ρ ≤ 0.60 |
| primary screen | fold-0 `marginal_vs_clone ≥ +0.0015` |
| objective isolation | candidate − C1 (BCE) on `marginal_vs_clone` ≥ **+0.0010** |
| pair flow | dominant-cell net **> 0** and mature-vs-never net **> 0** |
| damage cap | damage rate on RT600-correct pre-break pairs ≤ **0.0150** (the cap SS-03 failed at 0.0199) |
| serious | mean five-fold `marginal_vs_clone ≥ +0.0030`, ≥ 4/5 folds positive, dominant pair net > 0, C1 gap ≥ +0.0010, deployable runtime measured |

### 1.10 Negative interpretation — the lane this closes

If CRF-01 fails the primary screen, then in combination with `RT-970/971`
(sequence + BCE), `RT-111`/`700`/`701`/`123` (static + rank), W7-D3R Arm B
(static + capacity) and SS-01…04 (routing around RT600), **all four cells of the
representation × objective factorial are filled and none produces marginal
ensemble alpha.**

Closed on that outcome: learned causal prefix representations as an alpha source;
objective mismatch as a live explanation; and — unless CRF-02 says otherwise —
the whole "extract more from the legal prefix" lane. Belief in **H-D** (the
practical limit is the legal prefix itself; the W7-D3R gap is predominantly
post-`t` information) rises to the point where the project should turn to
deployment robustness rather than model research.

If CRF-01 fails but **C1 (BCE) fails by more**, that is a real finding about the
objective and must be reported as such — it does not resurrect the candidate.

---

## 2. CRF-02 · ACGN — AMORTIZED CONDITIONAL GENERATIVE NULL

**Unconditional. Independent of CRF-01.**

### 2.1 Central hypothesis

`73.99 %` of dominant-cell loss is never-break negatives (W7-D0 §E/G), and D3
shows their loss rate is predicted by heavy tails, long memory, and histories that
themselves produce *few* excursions and then wander — the signature of **null
misspecification**, not of missed break evidence. Every null this project ships is
a fixed rolling-mean historical calibration with AR(2) shared context. A **global,
amortized, nonlinear, distributional** null learned from all break-free histories
prices "normal" better, and the resulting predictive PIT / log-score sequence is a
different discriminative direction.

### 2.2 Information channel

The conditional predictive distribution of the next observation under the
break-free regime, and the sequential departure of the realised online stream from
it.

### 2.3 Exactly why prior attempts do not falsify it

| prior | why it does not close this |
|---|---|
| `m04_resid` | scalar AR / volatility residual **summaries**; no predictive distribution anywhere |
| `RT-1214` Kalman/NIS | per-series, **frozen**, **linear**, fixed order 2, Gaussian; mean + variance only; added as a feature block to a bank that already contains `m04_resid`; ρ 0.88 |
| `RT-1215` Hankel-DMD | per-series, frozen, linear delay-embedded operator; ρ 0.886, failed its own ρ ≤ 0.85 guard |
| GARCH(1,1) | scored **0.50012** standalone — literally zero, because a per-series adaptive filter adapts on the same timescale as the break. ACGN is fitted on **break-free histories only** and therefore cannot adapt to a break it never saw. |
| `m07_bayes` / BOCPD | a posterior over *change* with a memoryless geometric hazard; not a predictive density for `x_t` |
| `RT-1042` CFEP | differs on all three required axes — training population (break-free histories only vs all online segments incl. post-break rows), target object (full predictive density of the present vs a future handcrafted summary), downstream role (base model vs a frozen 16-column addendum to the saturated bank) |
| `RT-1225` SS-03 | a **fixed** post-hoc null-state SCDF correction applied to RT600's output. ACGN is a learned generative null **upstream** of any score and never sees RT600. |
| `m05_ctx` / `RT-150` | 50 raw series-constant columns / a 6-way hard gate. ACGN's history enters as an **8-dimensional bottleneck conditioning the null**, never as an additive predictor, and carries a mandatory derangement control. |

### 2.4 Inputs and design — FROZEN

**Pretraining population.** Break-free historical segments `H_i` of the
**training-fold series of that outer fold only**. No online row, no label, no
future. ≈ 30M points across 8,000 series at 5 folds.

**Model.** `q_θ(x_t | h_i, x_{t−1..t−R})`, where the conditioning path is the same
frozen TCN shell as CRF-01 (hidden 32, RF 127) reading the standardised series,
and `h_i` is an **8-dimensional bottleneck** produced by mean-pooling the same
encoder over `H_i`.

**Head.** 21 monotone quantile knots at levels
`0.01, 0.05, 0.10, …, 0.90, 0.95, 0.99`, parameterised as a base level plus
softplus increments so monotonicity holds by construction. **Pinball (quantile)
loss.** A mixture-density head is explicitly **not** used — MDNs are the fragile
end of the family and this program forbids post-score tuning.

**Online signals** (all strictly causal, all from `H_i` and `x_{<t}`):
predictive PIT `u_t` by interpolation across the knots; predictive log score;
cumulative PIT-uniformity discrepancy (a running Anderson-Darling-flavoured
statistic against `U(0,1)`); cumulative predictive-surprise; the encoder's own
latent-state shift against its history-pooled value.

**Head to score.** A small ranking head (one hidden layer, 32 units) over those
five signals plus their running peaks, trained with the **same same-`t` pairwise
logistic objective as CRF-01** (§1.6), on training folds only. The generative
model is **frozen** before the ranking head is fitted, so the label cannot rewrite
the null.

### 2.5 Controls

| control | isolates | status |
|---|---|---|
| **C1 — matched FIXED null**: AR(5) + historical residual ECDF (CRF-01's channels 2–3) feeding *identical* downstream statistics and an *identical* ranking head | **learned vs fixed conditional null** | **MANDATORY** |
| **C2 — deranged `h_i`**: history embeddings permuted across series within fold, everything else identical | **conditioning vs series memorisation** (the `m05_ctx` control) | **MANDATORY** |

Two controls, both decisive, both required by prior failures.

### 2.6 Purity

The pretraining is **outer-fold-scoped**: five separate null models, one per outer
fold, each fitted on that fold's training series' histories only. It may not be
fitted once globally and reused — that is precisely the Wave-7 nested-teacher
contamination pattern. The fold-purity sentinel of §0.5 runs first and must pass.

For a validation series, `h_i` is produced by applying the training-fold-fitted
encoder to that series' own history — a forward pass, never a fit. This is legal
because the history is fully available at `t = 0` and contains no label.

### 2.7 Streaming state and runtime

State ≈ **7 KB/series**: CRF-01's ring buffer plus `h_i` (8 floats) plus the five
accumulators and their peaks. `fit_history(H)` runs one encoder pass over `H` at
`t = 0`. `update(x_t)` adds one 21-knot head evaluation and a PIT interpolation.

Budget: pretraining ≈ 3.0 h; online scoring + C1 + C2 ≈ 2.0 h.

### 2.8 Gates

As §0.1 and §0.2, plus:

| gate | threshold |
|---|---|
| **learned-null isolation** | candidate − C1 (fixed null) on `marginal_vs_clone` ≥ **+0.0005** |
| **derangement** | candidate − C2 (deranged `h_i`) on `marginal_vs_clone` ≥ **+0.0005**. If C2 matches or beats the candidate, the history embedding is a series identifier and the arm is KILL regardless of its headline number — this is the `m05_ctx` rule. |
| never-break focus | mature-vs-never pair net **> 0** |
| damage cap | pre-break damage rate ≤ 0.0150 |

### 2.9 Negative interpretation

If ACGN fails, and specifically if it does not beat its **fixed**-null control by
+0.0005, then learned generative nulls are closed for this problem: the per-series
historical calibration the project already ships is the right null, and the
never-break false-positive mass is not a null-misspecification problem. Combined
with a CRF-01 failure, this closes **H-E** as well as H-A and H-B, leaving H-D.

---

## 3. CRF-03 · NNCSR-G — CONDITIONAL INTEGRATION

**CONDITIONAL. Not independent. Opens only on an interim bar.**

### 3.1 Opening condition

CRF-03 opens if and only if **CRF-01 or CRF-02 reaches at least WEAK**
(`marginal_vs_clone ≥ +0.0015` on fold 0) **and** passes its own mandatory
isolation control. If both primaries are KILL, CRF-03 **does not open** and the
program terminates — a failed primary followed by an architecture combination is
fishing, and this is the same rule that kept `RT-980` closed in Wave 6.

### 3.2 Design

CRF-02's learned predictive-PIT and log-score channels replace CRF-01's fixed-null
channels 1–3 and 6; `h_i` enters the encoder by FiLM conditioning at the input
projection; trained end-to-end under the same-`t` pairwise objective. Control: the
better of CRF-01/CRF-02 alone, i.e. the integration must beat its own best
component by ≥ +0.0010.

Budget ≈ 4.0 h. **If compute is constrained, run CRF-01 and CRF-02 and stop.**

---

## 4. FIRST EXECUTION — EXACTLY ONE

**CRF-01 · NNCSR, fold-0 pilot with its mandatory C1 (BCE) control.**

It is the only design that separates representation failure from objective failure
in a single run, it fills the last empty cell of the factorial, it is
RT600-independent, and it can close an entire research lane on failure.

**Not executed under this task.** Executing it requires:

1. an execution preregistration `CRF01_EXECUTION_PREREG.md` in this directory,
   committed **before** any score exists;
2. `RT-xxx` ids allocated from `research/EXPERIMENT_ID_MAP.md` for the candidate,
   the C1 control, and (conditionally) the C2 shuffle control — no id may be
   reused, including from voided experiments;
3. the §0.5 purity sentinel and the §1.x causality gates run and passing, with
   their output pasted into the execution preregistration, **before** the first
   TS-AUC is read.

Practical execution commands, recorded so the environment is not re-derived:

```bash
# torch and LightGBM segfault sharing a process on macOS/arm64 (Wave-6 finding).
# Keep them in separate processes.  Do NOT use KMP_DUPLICATE_LIB_OK.
W="/path/to/workspace/structural-break-causal-representation-frontier"
export SBR_ROOT="$W" PYTHONPATH="$W/src:$W/research/scripts"
V="/path/to/workspace/structural-break-wave8/.venv/bin/python"   # torch 2.13.0

# process 1 (torch): channels, training, OOF emission
"$V" research/scripts/crf01_nncsr.py --build-channels
"$V" research/scripts/crf01_nncsr.py --arm candidate --fold 0
"$V" research/scripts/crf01_nncsr.py --arm bce_control --fold 0

# process 2 (no torch): ensemble integration and pair flow over frozen .npy arrays
python research/scripts/crf01_integrate.py --fold 0
```

Neither script exists yet. Neither may be written with a training path until the
execution preregistration is committed.

---

## 5. WHAT THIS PROGRAM WILL HAVE ESTABLISHED EITHER WAY

**If CRF-01 clears:** the legal prefix contains materially more information than
the 500-vector extracts, and the route to it is a metric-aligned learned
representation over null-normalised channels. That reopens a lane the project has
had closed since Wave 6 on the strength of one crippled experiment.

**If CRF-01 fails:** the representation × objective factorial is complete and
empty. Combined with SS-01's degenerate arbiter, the diffuseness of the loss, and
the post-`t` reading of D3R Arm C, the project will have strong, structured
evidence that **0.6268 is close to what the legal causal prefix supports**, and
the remaining budget belongs to deployment reliability rather than to model
search.

Both outcomes are worth ~2 hours. The current state — assuming the answer from a
Wave-6 experiment that its own report says cannot answer it — is not.
