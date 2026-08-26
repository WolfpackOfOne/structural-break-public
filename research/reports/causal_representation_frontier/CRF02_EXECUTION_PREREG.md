# CRF-02 EXECUTION PREREGISTRATION — ACGN, AMORTIZED CONDITIONAL GENERATIVE NULL

**STATUS AT COMMIT TIME: `PREREGISTERED_NOT_EXECUTED`.**

No CRF-02 training path exists yet. No `RT-1237`/`RT-1238`/`RT-1239` score, OOF
vector, TS-AUC, `marginal_vs_clone` or pair-flow number exists.

This file freezes every remaining implementation detail of CRF-02 before any CRF-02
number can be produced or read. The scientific contract is `CRF_PROGRAM_PREREG.md`
§0 and §2, which this file does not modify, weaken or reinterpret.

Program preregistration: `CRF_PROGRAM_PREREG.md` @ `85d121f`.
CRF-01 result (KILL, abandoned at the cheap abandon gate): `crf01_nncsr.md`,
filed at `b6f27e3`. **CRF-02 is unconditional and independent of that outcome**
(`CRF_PROGRAM_PREREG.md` §2), and is not a response to it: its hypothesis is about
the *negative* side of the loss, not about extracting more break evidence.
Branch: `research/causal-representation-frontier-2026`. Base: `b6f27e3`.

---

## 1. QUESTION

`73.99 %` of dominant-cell loss is never-break negatives (W7-D0 §E/G), and D3 shows
their loss rate is predicted by heavy tails, long memory, and histories that
themselves produce few excursions and then wander — the signature of **null
misspecification**, not of missed break evidence. Every null this project ships is a
fixed rolling-mean historical calibration with AR(2) shared context.

Does a **global, amortized, nonlinear, distributional** null — learned from
break-free histories only — price "normal" better, and does the resulting sequential
departure of the realised online stream from it constitute a different
discriminative direction?

CRF-01 answered "can we extract more break evidence from the legal prefix?" with a
measured no. **This is a different question and its failure modes are different.**

---

## 2. EXPERIMENT IDs — ALLOCATED PROSPECTIVELY

| ID | arm | status |
|---|---|---|
| `RT-1237` | CRF-02 candidate: learned amortized conditional generative null + frozen downstream signals + same-`t` pairwise ranking head | **CANDIDATE** |
| `RT-1238` | C1 mandatory control: **fixed** null (AR(5) + history residual ECDF) feeding identical downstream statistics and an identical ranking head. Isolates **learned vs fixed conditional null**. | **MANDATORY CONTROL** |
| `RT-1239` | C2 mandatory control: history embeddings `h_i` **deranged across series within fold**, everything else identical. Isolates **useful conditioning vs series memorisation**. | **MANDATORY CONTROL** |

Both controls are **mandatory** and both run on fold 0 regardless of the candidate's
headline number (unless the §7 abandon gate fires, which stops the whole arm).

`RT-401` and the seven RT-600 specialists are reused as frozen OOF vectors and
consume no id.

### 2.1 Collision audit

Verified before allocation: the highest allocated id is `RT-1235`; `RT-1236` is
**reserved to CRF-01's unrun C2 arm and is deliberately skipped, not recycled**;
`RT-1237`, `RT-1238` and `RT-1239` appear in no `RESULTS.csv` on any local or remote
ref, in no tracked file, and in no commit reachable from any ref
(`git log --all -S`). None is a recycled killed, void, abandoned, contaminated or
reserved id.

### 2.2 Why prior work does not falsify this

Unchanged from `CRF_PROGRAM_PREREG.md` §2.3 and not re-argued here: `m04_resid`
(scalar summaries, no predictive distribution), `RT-1214` Kalman/NIS (per-series,
frozen, linear, Gaussian, mean+variance only, ρ 0.88), `RT-1215` Hankel-DMD
(per-series, frozen, linear, ρ 0.886), GARCH(1,1) (0.50012 standalone — a per-series
adaptive filter adapts on the same timescale as the break; ACGN is fitted on
break-free histories only and therefore *cannot* adapt to a break it never saw),
`m07_bayes`/BOCPD (a posterior over change, not a predictive density),
`RT-1042` CFEP (differs on training population, target object and downstream role),
`RT-1225` SS-03 (a fixed post-hoc correction to RT600's output; ACGN is a learned
generative null upstream of any score and never sees RT600),
`m05_ctx`/`RT-150` (raw series-constant columns / a hard gate; ACGN's history enters
as an 8-dimensional bottleneck with a **mandatory derangement control**).

CRF-01 does not falsify it either: CRF-01 is a discriminative sequence ranker trained
on labels, with a **fixed** per-series null baked into its channels. CRF-02 learns
the null itself, from unlabelled break-free history, and the label never touches it.

---

## 3. DATA, FOLDS, POPULATION

Identical to `CRF01_EXECUTION_PREREG.md` §3 and not restated: canonical full store,
`research/folds/folds.parquet`, dev folds `0..4`, **fold `-1` lockbox never loaded**,
no `X_test.reduced`, no production artifact, no submission. Binding screen is
**fold 0**.

**Pretraining population (frozen).** Break-free **historical** segments `H_i` of the
**training-fold series of that outer fold only**. No online row, no label, no future
value, and no validation-fold series' history enters the fit. Per
`CRF_PROGRAM_PREREG.md` §2.6 the null is **outer-fold-scoped**: a separate null per
outer fold. It may **not** be fitted once globally and reused — that is exactly the
Wave-7 nested-teacher contamination pattern.

For a validation series, `h_i` is produced by applying the **training-fold-fitted**
encoder to that series' own history — a forward pass, never a fit. Legal because the
history is complete at `t = 0` and contains no label.

---

## 4. THE MODEL — FROZEN

`q_θ(x_t | h_i, x_{t−1..t−R})`, a monotone quantile predictor.

**Per-series standardisation.** `z = (x − μ_H)/σ_H` where `μ_H`, `σ_H` are the mean
and `ddof=1` std of that series' own history, floored at `1e-9`. History only, legal
at `t = 0`, identical to CRF-01's convention.

**History window (frozen).** `Hwin_i` = the **last 1024 points** of `H_i`, or all of
`H_i` if shorter. 1024 is ≈ 4× the encoder's measured receptive field of 253, so the
window is not receptive-field-limited. The **same** window is used at training and at
inference, so there is no train/inference pooling mismatch to reason about later.

**Body.** The frozen CRF-01 / `RT-970` TCN shell, `n_in = 1`, hidden 32, kernel 3,
dilations `1,2,4,8,16,32`, six residual blocks, GELU, dropout 0.1, weight-normed
left-padded causal convolutions, no BatchNorm, no time-axis normalisation. Its
per-timestep output is 32-dimensional. **One body, used for both passes** —
`CRF_PROGRAM_PREREG.md` §2.4's "the same encoder".

**History bottleneck.** `p_i = mean over time of body(z[Hwin_i])` (32-d), then
`h_i = Linear(32 → 8)(p_i)`. Exactly the preregistered **8-dimensional bottleneck
produced by mean-pooling the same encoder over `H_i`**.

**Strict-past shift (load-bearing).** A causal TCN at position `t` sees `z_t` itself.
The null must predict `x_t` from `x_{t−1..t−R}`, so the predictive pass is fed the
sequence **shifted right by one**: position `t` receives `z_{t−1}`, and the online
segment's position 0 receives the **last history point**, not a zero. The output at
`t` is therefore a function of the strict past and the history alone. This is
asserted by the prefix, truncation and shift sentinels of §10, not argued.

**Head.** Per timestep, concatenate `body(z_shift)(t)` (32-d) with `h_i` (8-d,
broadcast over time) → 40-d, then one `1×1` linear map to 21 parameters:
`q_1 = base`, `q_{k+1} = q_k + softplus(δ_k)` for `k = 1..20`. **Monotonicity holds
by construction.** No MDN, no flow, no head search — `CRF_PROGRAM_PREREG.md` §2.4
forbids all three.

Conditioning is by **concatenation at the head**, not FiLM at the input projection.
FiLM is CRF-03's declared mechanism (`CRF_PROGRAM_PREREG.md` §3.2) and is
deliberately left unused here so the two experiments stay distinct.

**Quantile levels — the 21 fixed by the program preregistration:**

```
0.01, 0.05,
0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
0.95, 0.99
```

**Objective: pinball (quantile) loss**, `mean over levels and valid positions of
max(τ_k(z_t − q_k), (τ_k − 1)(z_t − q_k))`. No label, no `τ`, no online row.

### 4.1 Pretraining optimisation — frozen

Inherited from CRF-01 wherever it applies, so nothing is newly chosen that does not
have to be:

| item | value |
|---|---|
| optimiser | `AdamW`, lr `3e-3`, weight decay `1e-2` |
| schedule | `CosineAnnealingLR` over all steps |
| epochs | **10** |
| batch | 32 series, length-bucketed, chunk order shuffled, same `_batches` helper |
| gradient clip | `1.0` |
| seed | `0`; per-fold `seed·1000 + fold`, as CRF-01 |
| threads / determinism | 6 threads, `torch.use_deterministic_algorithms(True)` |

**Why 10 and not CRF-01's 20**, fixed here before any CRF-02 number exists: each
epoch sees `6,383 × 1024 ≈ 6.5M` history points against CRF-01's `≈ 3.2M` online
points, so 10 epochs is already a larger token budget than CRF-01's 20, and the
program's `≤ 15 h` screening cap (§0.8) with `0.50 h` already spent by CRF-01 does
not permit more. **This is a compute-budget choice made in advance, not a tuned
one:** no other epoch count will be tried, and a loss curve that "looks strange" is
not grounds to change it (`CRF_PROGRAM_PREREG.md` §0.9).

---

## 5. ONLINE SIGNALS — FROZEN, FIVE PLUS THEIR PEAKS

The generative model is **frozen** before any of this is computed, and frozen again
before the ranking head is fitted, so the supervised label can never rewrite the
null. All five are strictly causal functions of `H_i` and `x_{<t}` plus `x_t`.

Let `Q_t = (q_1 … q_21)` be the predicted knots at `t` and `z_t` the realised
standardised value.

| # | name | definition |
|---|---|---|
| 1 | `pit` | `u_t = interp(z_t; Q_t, levels)`, linear between knots, clamped to `[0.005, 0.995]` outside the knot range |
| 2 | `neglog` | `−log(max(dens_t, 1e-8))` where `dens_t = (τ_{k+1} − τ_k)/max(q_{k+1} − q_k, 1e-6)` in the bin containing `z_t`, and the extreme bin's density outside; clipped `[−5, 20]` |
| 3 | `ad` | running Anderson–Darling-weighted uniformity discrepancy, `mean_{s ≤ t}[ −log(u_s) − log(1 − u_s) ] − 2`. Expectation **0** under `U(0,1)`, `O(1)` to update, and it carries the AD tail weighting. Clipped `[−5, 20]` |
| 4 | `surp` | `mean_{s ≤ t}[ neglog_s ] − neglog_base_i`, where `neglog_base_i` is the mean `neglog` the frozen model assigns to that series' **own `Hwin_i`** — computed at `t = 0`, history only. Clipped `[−10, 10]` |
| 5 | `lshift` | `‖runmean_{s ≤ t}(body(z_shift)(s)) − p_i‖₂ / √32`, the encoder's latent-state shift against its own history-pooled value. Clipped `[0, 20]` |

**Peaks.** `peak_k(t) = max_{s ≤ t} v_k(s)`, with `v_1(s) = |u_s − 0.5|` (the
extremeness of the PIT, since the running max of `u` itself is not a discrepancy)
and `v_k = signal_k` for `k = 2..5`.

**Ranking-head input: 10 features** — the five signals and their five peaks, in that
fixed order. Nothing else. No `t`, no `elapsed`, no `n_online`, no `τ`, no age, no
RT600, no 500-column bank, no cross-sectional quantity.

---

## 6. RANKING HEAD — FROZEN

`MLP(10 → 32 → 1)`, one hidden layer of 32 units, GELU, no dropout — exactly
`CRF_PROGRAM_PREREG.md` §2.4's "one hidden layer, 32 units". Trained with the **same
same-`t` pairwise logistic objective as CRF-01** (`m_neg = 8`, uniform pair
weighting, minimum group occupancy ≥ 1 positive and ≥ 8 negatives, negatives drawn
uniformly with replacement, resampled every step), on **training folds only**.

Optimisation: AdamW lr `3e-3`, wd `1e-2`, cosine, 20 epochs, batch 32 series,
gradient clip 1.0, seed 0, per-fold seed `seed·1000 + fold`, pair rng
`2026082610 + fold`.

**Feature standardisation — the one genuinely fitted global object in CRF-02.**
Median and IQR of each of the 10 features computed on **training-fold rows only**,
frozen, then applied to every row including validation; a feature whose training-fold
IQR is `≤ 1e-12` gets scale 1; values clipped to `±20` after standardising. This is
the `CRF_PROGRAM_PREREG.md` §0.5 "global standardiser fitted on training folds only"
case, it is **asserted in code**, and §10's purity gate P5 tests it non-vacuously
(unlike CRF-01, where no global standardiser existed at all).

**No joint end-to-end fine-tuning of the null under the label.** The generative
parameters are frozen and are asserted to receive no gradient while the ranking head
trains. That would violate the scientific isolation the experiment exists to
establish.

---

## 7. CHEAP ABANDON GATE — EVALUATED FIRST

`CRF_PROGRAM_PREREG.md` §0.2, unchanged and unrelaxable, with the same frozen
definitions as `CRF01_EXECUTION_PREREG.md` §7 (raw score, `ts_auc_flat` on fold-0
validation rows; ρ = `diagnostic_pack(...)["within_t_rank_corr_rt600"]` on fold-0
dominant-cell rows against `harness.rt600_blend`):

> abandon if fold-0 standalone whole-fold TS-AUC **< 0.600** **AND** within-`t` ρ vs
> the RT600 blend **≤ 0.60**.

If it fires the candidate is abandoned: no further folds and no ensemble integration.
Controls already run are preserved and reported in full, and — because CRF-02's
scientific content is the **comparison** — `RT-1238` and `RT-1239` are run on fold 0
**before** the gate is read, so the learned-vs-fixed and conditioning-vs-memorisation
questions get answered whatever the headline number does.

---

## 8. EXECUTION STAGING — FROZEN

| stage | what runs | what may be read |
|---|---|---|
| **S0** | purity + causality preflight (§10) | test pass/fail only, **no TS-AUC of any arm** |
| — | **`Validate CRF-02 purity and causality preflight` committed and pushed** | — |
| **S1** | fold-0 generative null (candidate); signals for candidate, C1 and C2; three ranking heads | fold-0 standalone TS-AUC and ρ for all three arms → **§7 gate**, and the §11 isolation comparisons on standalone |
| **S2** | only if §7 does not fire: folds 1–4 for all three arms | full fold-0 evaluation, `marginal_vs_clone`, gates (§11) |
| **S3** | only if §11 passes: **no new training** | five-fold confirmation from the same OOF vectors |

As in CRF-01, S2 exists because the cross-fitted calibration that produces the
**fold-0** marginal fits fold 0's map on folds 1–4, so a fold-pure five-fold OOF is
required before any marginal exists.

---

## 9. IMPLEMENTATION CONTRACT

Same two-process split as CRF-01, same interpreters, `KMP_DUPLICATE_LIB_OK`
forbidden:

```
research/scripts/crf02_acgn.py       TORCH PROCESS.  pretrains the fold's null,
                                     freezes it, emits the 10 online signals for
                                     every arm, trains the ranking heads, emits
                                     frozen raw OOF score arrays + metadata.
                                     Imports no lightgbm.
research/scripts/crf01_integrate.py  NO-TORCH PROCESS.  reused unchanged, with
                                     CRF-02's ids registered.  Canonical SCDF,
                                     E0/E1/E2, pair flow, gates.
```

Streaming form reported as for CRF-01: `fit_history(H)` = one encoder pass over
`Hwin_i` giving `h_i`, `p_i` and `neglog_base_i`; `update(x_t)` = one incremental
causal pass, one 21-knot head evaluation, one PIT interpolation and five `O(1)`
accumulator updates; measured state bytes per series and wall-clock reported.

---

## 10. PRE-SCORE GATES — MUST PASS BEFORE ANY SCORE

In `tests/test_crf02_causality.py`, run to green before any CRF-02 number is read.

| gate | assertion |
|---|---|
| **P1** | the null's fitted series set is exactly `FOLDS \ {f}`; set arithmetic over all five outer folds |
| **P2** | positive control: the **globally pretrained** null (one null over all 10k histories, reused across outer folds) is reproduced and shown to intersect `{f}` for every `f` — the Wave-7 pattern, detected not assumed |
| **P3** | live path: pretraining asserts its series set internally, and a validation series' `h_i` is produced by a **forward pass** through the training-fold-fitted encoder with `requires_grad` off and no optimiser step |
| **P4** | `μ_H`, `σ_H`, `Hwin_i`, `p_i`, `h_i` and `neglog_base_i` depend on that series' own history alone, and are bitwise unchanged by any perturbation of its online segment |
| **P5** | the 10-feature standardiser's median/IQR are computed from training-fold rows only; perturbing validation rows leaves them bitwise unchanged; and the contaminated variant (fit on all rows) is shown to differ |
| **P6** | the frozen generative parameters receive **no gradient** while the ranking head trains (`grad is None` or exactly zero for every generative parameter after a backward pass) |
| **C1** | bitwise prefix invariance, `atol = 0.0`, ≥ 8 real series of different lengths, over the 10-feature signal block |
| **C2** | `assert_no_forbidden_columns` over every signal name and metadata key |
| **C3** | truncation: signals recomputed on a truncated online segment reproduce surviving rows ≤ 1e-8 |
| **C4** | batch composition, `float64`, ≤ 1e-8 |
| **C5** | deterministic replay: identical state-dict sha256 and bitwise identical signals |
| **C6** | `finite(oof[fold == −1]) == 0` for every emitted vector |
| **C7** | no final online length referenced; the strict-past shift is verified directly — the predicted knots at `t` are bitwise unchanged when `z_t` and everything after it is replaced, i.e. **the null never sees the value it is pricing** |
| **C8** | single-series inference reproduces the in-batch score ≤ 1e-8 — no cross-sectional quantity at inference |
| **C9** | monotonicity: `q_1 ≤ q_2 ≤ … ≤ q_21` everywhere, by construction and by assertion on real data |
| **C10** | C2 derangement is a genuine derangement within fold group, is deterministic, and leaves every other input bitwise unchanged |

Any failure stops the experiment. No score is read.

---

## 11. BINDING FOLD-0 SCREEN GATES

| gate | threshold |
|---|---|
| abandon (checked first, §7) | standalone fold-0 whole TS-AUC `≥ 0.600` **or** ρ `> 0.60` |
| **primary screen** | fold-0 `marginal_vs_clone ≥ +0.0015` |
| **learned-null isolation** | candidate − C1 (fixed null) on `marginal_vs_clone` **`≥ +0.0005`** |
| **derangement** | candidate − C2 (deranged `h_i`) on `marginal_vs_clone` **`≥ +0.0005`**. If C2 matches or beats the candidate the history embedding is a series identifier and the arm is **KILL regardless of its headline number** — the `m05_ctx` rule. |
| **never-break focus** | mature-vs-never pair net **> 0** |
| **damage cap** | pre-break damage rate on RT600-correct pairs **≤ 0.0150** |

If any fails, **CRF-02 = KILL**. No `CRF-02b`. No tuning.

SERIOUS requires mean five-fold `marginal_vs_clone ≥ +0.0030`, `≥ 4/5` folds
positive, positive pair flow, both isolation controls passing, and measured feasible
runtime. If SERIOUS, broad execution stops and CRF-03 is **not** started.

### 11.1 One declared diagnostic, not an arm

Because C1 has no encoder, its `lshift` signal and peak are identically **zero**,
which leaves the candidate two informative features that the control lacks. To make
sure the isolation gate is not flattered by that, a **secondary diagnostic** refits
the candidate's ranking head on the **8 shared features only** (signals 1–4 and their
peaks), on the same frozen generative model. It consumes **no RT id** and is not a
candidate — the same treatment `W7-D0` received — and it is reported alongside the
gate rather than substituted for it.

---

## 12. EVALUATION, FILING, BUDGET, SAFETY

Calibration, ensemble rule, mandatory reporting and the canonical deterministic pair
sample are **identical to `CRF01_EXECUTION_PREREG.md` §12** and are not restated or
altered. Result filing follows §15 there. `unique repair coverage` keeps the
definition fixed at `CRF01_EXECUTION_PREREG.md` §18.15.

Budget: fold-0 screen expected ≈ 1.0 h (pretrain ≈ 0.6 h, signal generation ≈ 0.1 h,
three ranking heads + the §11.1 diagnostic ≈ 0.3 h). Program cap `≤ 15 h`, of which
CRF-01 consumed `0.50 h`. No Optuna, no architecture search, no width sweep, no
learning-rate sweep, no seed fishing, no re-rolled fold.

Safety: no lockbox read, no test data, no `X_test.reduced`, no submission, no
`production/rt600` edit, no promotion, no force push, no rebase of pushed history, no
recycled RT id, no post-score tuning.

---

## 13. PREFLIGHT COMPLETED

*Empty at Stage A by design. Filled in the Stage B commit
`Validate CRF-02 purity and causality preflight` with actual test output, pushed
**before** the first CRF-02 score exists.*
