# WAVE 6 — NEURAL TRACK PRE-REGISTRATION

**Written 2026-08-22, before any neural number exists. Binding.**
Subordinate to `research/PROTOCOL.md`, `research/VALIDATION_V2.md`,
`research/WAVE6_PREREG.md` §2 (the promotion bar) and
`research/WAVE6_STOPPING_RULE_AMENDMENT.md` §3.

---

## 0. WHY THIS TRACK EXISTS, STATED HONESTLY

Wave 5 ran ten experiments and promoted nothing. Its central measurement was not
"the ideas were bad" — it was that `m12_rdep`'s advantage **shrinks monotonically
as the comparison approaches the deployed system**:

| comparison | Δ |
|---|---:|
| standalone vs the single control | **+0.0038** ± 0.0007, 3 partitions |
| two-model blend vs the seed clone | +0.0010 ± 0.0009, sign flips |
| eighth ensemble member | +0.0005 |
| full architecture rebuild | **−0.0027** |

The information was real **and redundant**. Four independent measurements say
this ensemble is saturated for hand-designed statistics that overlap the 500
columns it already computes.

**That is a statement about a search direction, not about the problem.** Every
one of those measurements was taken inside one model family: gradient-boosted
trees on a fixed causal feature bank. A learned causal sequence representation
is the strongest materially different family the project has never run, and
Amendment 1 established that the model class was never the constraint we assumed
it was.

**What would make this track wrong.** If the 500-column bank is already a
sufficient statistic for what a causal model can extract, then a neural model
reading the same information will find the same thing and the track ends at §7's
terminating condition. That is a real possibility and the experiment is designed
to detect it cheaply, which is why N1 (learner capacity, on the *existing*
features) runs before N2 (representation learning, on *raw channels*).

---

## 1. THE TWO INPUT TRACKS — NEVER CONFLATED

| track | input | what it isolates |
|---|---|---|
| **A** | the existing 500 causal columns → MLP | **learner** capacity |
| **B** | compact causal raw/derived channels → TCN | **representation** learning |

Reading, fixed in advance:

* A helps, B does not → **the learner is the lever**; the bank is a sufficient
  statistic and the gain is in how it is combined.
* B helps materially beyond A → **learned temporal representation is the lever**;
  the bank is lossy and the wave-7 programme is representation.
* neither helps → §7 fires.
* B helps and A does not → the bank is lossy *and* trees were not the
  bottleneck; report it, do not rationalise it.

---

## 2. EXPERIMENTS — THE WHOLE FAMILY, DECLARED NOW

| id | model | input | when |
|---|---|---|---|
| `RT-960` | MLP `500 → 256 → 128 → 32 → 1` | track A | first, unconditionally |
| `RT-961` | the same MLP, second regularisation setting | track A | with `RT-960` |
| `RT-970` | causal dilated TCN, hidden 32 | track B | second, unconditionally |
| `RT-971` | causal dilated TCN, hidden 64 | track B | with `RT-970` |
| `RT-980` | GRU, hidden 64, unidirectional | track B | **only if** §6 escalation is met |

**Two architecture sizes per family, two regularisation settings, and nothing
else.** No width sweep, no depth sweep, no learning-rate sweep, no seed fishing.
A transformer is **not authorised** by this document; authorising one requires a
new pre-registration that names what N1/N2/N3 measured that motivates it.

### 2.1 Exact architectures — fixed here so they cannot be tuned later

**MLP (`RT-960` / `RT-961`).** `500 → 256 → 128 → 32 → 1`, GELU, LayerNorm after
each hidden layer, dropout `p` — `RT-960` `p = 0.1`, `RT-961` `p = 0.3` — sigmoid
output. AdamW, lr `1e-3`, weight decay `1e-2` (`RT-960`) / `1e-1` (`RT-961`),
cosine decay, batch 4096, 12 epochs, gradient clip 1.0. Inputs standardised by
**training-fold** median/IQR computed on the training folds only and frozen;
non-finite inputs → 0 after standardisation, with a companion indicator channel
per column group (not per column).

**TCN (`RT-970` / `RT-971`).** Causal dilated 1-D convolutions, kernel 3,
dilations `1, 2, 4, 8, 16, 32`, six residual blocks, hidden 32 / 64, GELU,
dropout 0.1, weight-normed convs, per-timestep linear head → sigmoid. Receptive
field 127 online steps; longer-range evidence enters through the causal running
channels of §3, not through depth. AdamW, lr `3e-3`, weight decay `1e-2`, batch
32 **series**, 20 epochs, gradient clip 1.0.

**GRU (`RT-980`).** Unidirectional, hidden 64, 2 layers, dropout 0.1, per-timestep
head. Same optimiser block as the TCN.

All models: `torch.manual_seed(seed)`, `torch.use_deterministic_algorithms(True)`,
single-threaded data order, seeds `0` and one replicate seed for the promoted
candidate only.

---

## 3. TRACK B INPUT CHANNELS — THE LEGAL SET, FIXED

Per online index `t`, computed from `hist` and `online[:t+1]` only:

| channel | definition |
|---|---|
| `z` | `(x_t − med_hist) / iqr_hist`, clipped to ±8 |
| `z2` | `z²` |
| `absz` | `|z|` |
| `sgn` | `sign(z)` |
| `lag1` | `z_t · z_{t−1}` (0 at `t = 0`) |
| `run_mean` | expanding mean of `z` over `online[:t+1]` |
| `run_var` | expanding variance of `z`, log1p-compressed |
| `ewma_fast` | EWMA of `z`, half-life 8 |
| `ewma_slow` | EWMA of `z`, half-life 64 |
| `elapsed` | `log1p(t)` — an elapsed **counter**, not a length |

Ten channels. `med_hist` and `iqr_hist` come from the historical segment, which
is break-free by construction and fully observed before the online stream
starts. Every other channel is a causal running statistic.

**`elapsed` is legal and `n_online` is not.** `log1p(t)` is known at time `t`.
The final online length is not, and never appears — the existing
`tests/test_no_n_online_leakage.py` gate applies unchanged.

---

## 4. HARD CONSTRAINTS — ANY VIOLATION VOIDS THE RUN

**Forbidden in any deployable neural input:** `tau`; any future observation;
future segment length; final online length; boundary-conditioned availability or
missingness; any oracle feature. This is the RT-900 lesson made permanent.

**Forbidden architecturally:** bidirectional RNNs; unmasked or non-causal
attention; any convolution that reads `t' > t`; full-sequence normalisation;
BatchNorm over the time axis; a padding mask that encodes final sequence length.

**Normalisation.** Historical, causal-running, or a fixed training-set transform.
**Never** full-online mean/std, final min/max, or future-aware batch statistics.
This is audited explicitly, not assumed — it is the single most likely way this
track fakes a result, because "standardise the sequence" is the default habit
everywhere in deep learning and it is a future leak here.

**Padding.** Sequences are **left-aligned at `t = 0`** and padded on the right.
Right padding after a causal model cannot influence earlier outputs. The mask is
used only to zero the loss. Left padding is forbidden.

### 4.1 Gates, run before any score is read

| gate | requirement |
|---|---|
| **prefix invariance** | two series sharing a prefix with different futures produce predictions on the shared prefix agreeing to `<= 1e-8` (float64 eval), in the same batch composition and alone |
| **no-`n_online` dependence** | truncating the online segment leaves surviving predictions unchanged to `<= 1e-8` |
| **no-τ dependence** | two series identical except for where the break is produce identical predictions on the shared pre-break prefix |
| **fold purity** | every standardisation constant, every epoch, every early-stop decision uses training folds only; asserted in code |
| **determinism** | two runs at the same seed agree to `<= 1e-8` |

A failed gate voids the run. It is not patched into passing after the score is
known.

---

## 5. PROTOCOL — MATCHED TO EVERY OTHER STREAM

| item | value |
|---|---|
| validation surface | `research/folds/folds.parquet` folds 0–4, 8,000 dev series |
| forbidden | `RT-500`–`RT-506`, `folds_final10k`, fold −1, `X_test.reduced`, `y_test.reduced`, the leaderboard |
| metric | `sbr.metric.ts_auc_flat`, official `pairs` weighting |
| cross-fit | the same 5-fold loop as `sbr.pipeline.run`; fold `k` never sees fold `k` |
| calibration | the canonical cross-fitted `SCDF_NSEEN`, `time_coord="log_n_seen"` — identical to every shipped stream |
| ledger | `research/RESULTS.csv` via `append_result`, never hand-edited |
| runner | `research/scripts/wave6_neural.py`, reusing `Data`, `load_features`, `rows_for`, `evaluate_scores`, `append_result` so the fold loop, the row sampling and the scorer stay byte-identical to the LightGBM streams |

**Controls, all four, none optional:**

| control | what it is |
|---|---|
| `A` | `RT-300`, the single-model control · 0.61605 |
| `B` | the seed-clone ensemble · 0.62164 |
| `S` | the RT-600 architecture, seven specialists · 0.62581 |
| **seed clone** | **the binding control: `S` + one more exchangeable member, worth +0.00003 (W5-NULLTEST)** |

---

## 6. THE PROMOTION BAR — §2's, PLUS THE NEURAL-SPECIFIC CONTROLS

A neural candidate advances only if **all** of these hold:

1. aggregate Δ ≥ **+0.0030** TS-AUC over the strongest matched control;
2. positive on **≥ 4 of 5** canonical folds;
3. paired series bootstrap 95% CI on the contrast **excludes zero**;
4. **incremental value over `S`** as an eighth member, not merely over `RT-300`;
5. **the gain exceeds a matched seed clone's** — an eighth exchangeable member
   is worth +0.00003, so anything that looks like bagging is bagging;
6. no regression in the causality, prefix-invariance or `n_online` gates.

Major architectural promotion additionally requires **alternate partitions**
(`folds_alt1/2/3`) to support the direction. Deltas, never levels, are compared
across partitions — levels move ~0.009 between partitions and ~0.003 for deltas
(W4-E2).

**Beating one LightGBM model is not sufficient and never was.** Low correlation
with the ensemble is not sufficient. A positive blend delta is not sufficient.
The binding comparison is **candidate information contribution vs matched
seed-clone diversity**, exactly as in Wave 5.

### 6.1 Escalation to `RT-980` (GRU)

Authorised **only** if `RT-970` or `RT-971` clears (1), (2) and (3) against the
single control **and** shows a positive eighth-member delta against `S`, i.e. it
reaches the last hurdle rather than the first. A TCN that fails standalone does
not earn a GRU; that would be architecture fishing with extra steps.

---

## 7. TERMINATING CONDITION — FIXED NOW

Wave 6 stops adding when **both** input tracks have been run once each, under
these controls, and **neither** clears §6 against a matched seed clone. At that
point the recommendation is: ship `RT-600`, write the wave up, open no further
Wave-6 candidates.

This condition is written before a single neural number exists precisely so that
it cannot be renegotiated after a near-miss. A Δ of +0.0028 on 3/5 folds is a
failure, not "promising".

---

## 8. THE OFFLINE-TEACHER ROUTE — A NEURAL MODEL NEED NOT SHIP

`WAVE6_PREREG.md` §14 stays open and is **not** gated on §6. If a neural model is
informative but too expensive or not whitelisted for deployment, it may be used
offline to produce: a probability, an embedding, a hard-negative score, an
auxiliary target, or motif clusters — distilled into LightGBM, XGBoost or a small
MLP.

Distillation is a **separate experiment with its own controls**, not a rescue for
a failed neural candidate. The distilled student faces §6 unchanged, and the
teacher's own targets must be cross-fitted fold-pure — a teacher trained on fold
`k` may not produce targets for fold `k`. That is exactly the trap W5-E3's
hard-negative miner had to be built around, and the same `_assert_fold_pure`
discipline applies.

**W5-E3 is the standing warning here.** The last time this project reweighted
training rows toward "hard" cases, it lost 0.00701 and 0.01339 on its two arms
and worsened the young-break buckets it was designed to help. Teacher signals are
row weights by another name.

---

## 9. ENVIRONMENT — TWO DIFFERENT QUESTIONS, NOT ONE

Amendment 1's correction was that "not installed locally" and "not permitted in
the competition" are different facts, and I previously conflated them. Both are
recorded here so the conflation cannot recur:

| question | status |
|---|---|
| is `torch` installed in the research venv? | **no** — `torch`, `xgboost`, `catboost` are all absent locally. A local install is a prerequisite for *running* N1/N2. |
| is `torch` on the competition whitelist? | **W6-E0, outstanding.** Dependencies are declared in `requirements.txt` and installed by the platform; packages not on the whitelist can be requested. |

**Neither gates the research question.** The local install is a chore. The
whitelist gates *deployment*, and §8 exists precisely because a model can be
valuable without shipping. What the whitelist does gate is whether a promoted
neural candidate can be a *direct* submission or must go through distillation —
so W6-E0 is resolved before any promotion decision, not before the experiments.

Local budget for planning: 10 cores, 16 GB RAM, CPU only. The MLP at 1,000,000
sampled rows is roughly half an hour for five folds; the TCN over 6,400 training
series per fold is the expensive one. If a fold exceeds three hours, the run is
reported as infeasible at this budget rather than quietly shrunk.

---

## 10. DEGREES OF FREEDOM

| | count |
|---|---|
| families declared | 2 unconditional (`N1`, `N2`), 1 conditional (`N3`) |
| runs declared | 4 unconditional (`RT-960/961/970/971`), 1 conditional (`RT-980`) |
| architecture sizes per family | 2, fixed in §2.1 |
| regularisation settings | 2, fixed in §2.1 |
| losses | 1 (BCE), plus at most 1 pre-registered alternative (same-timestep pairwise ranking), which must be declared before it is run |
| seeds | 1 per configuration; a replicate seed **only** for a candidate that has already cleared §6 |
| hyperparameters selectable by W6-E2R | **0** |
| Crunch submissions authorised | **0** |

Every run is charged against multiplicity at declaration time, not at
publication time.
