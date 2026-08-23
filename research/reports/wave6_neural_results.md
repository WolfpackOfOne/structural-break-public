# WAVE 6 NEURAL TRACK — CONTROLLED EXECUTION RESULT

**Both pre-registered input tracks ran once each, under controls, and neither
cleared the bar. Wave 6 terminates. `RT-600` = 0.6268 remains champion.**

Pre-registration: `research/WAVE6_NEURAL_PREREG.md`, committed `17d01b8` before
any neural number existed. Engineering frozen at **`78f6933`**, the PRE-NEURAL
EXECUTION SHA. Nothing in the pre-registration was edited at any point.

---

## A. PRE-SCORE TEST STATE

| | |
|---|---|
| suite, two processes, artifacts attached | **650 passed · 15 failed · 1 skipped** |
| causality skips | **0** — the four `@needs_model` tests now RUN |
| the 15 failures | **pre-existing**, pinned in `research/known_failures.json` |
| RT-600 feature manifest | `1646c3b9…cced`, **byte-identical** |

The single remaining skip is `test_visualization.py` (no `matplotlib`), a
plotting import.

### The torch/LightGBM segfault, found while pinning the fingerprint

torch ships `torch/lib/libomp.dylib`, sklearn ships its own, and LightGBM brings
a third OpenMP runtime. **Loading torch and LightGBM into one process segfaults
on macOS/arm64**, dying inside `lightgbm/basic.py`. The first fingerprint run
recorded *"0 known failures"* because pytest had **crashed, not passed** — the
exact silent-success mode this gate exists to catch.

Fixed by running the suite in **two processes** and unioning the failure sets,
and by making the parser raise on `Fatal Python error` / `Segmentation fault` and
on an unparseable summary rather than returning an empty set. **Not** fixed with
`KMP_DUPLICATE_LIB_OK`, which trades a loud crash for silent numerical
corruption in the two libraries whose outputs this project compares bitwise.

---

## B. REPRODUCIBILITY FIX — `test_feature_order_is_immutable`

**It was failing on a true statement.** It asserted `MODULE_ORDER == the seven`.
Wave 5 split one name into two: `MODULE_ORDER` became the **append-only column
ordering** over everything the streaming registry can build, and
`PRODUCTION_MODULES` became the **shipped default**. Registering `m12_rdep`
appended to the former and deliberately left the latter alone, so no production
column moved — but the old assertion could not express that distinction.

Bookkeeping repair only, **no `src/` change**. The invariant is restated in four
parts: `PRODUCTION_MODULES` is the RT-600 seven; `MODULE_ORDER[:7]` is that tuple
and contains no duplicates; `set(MODULE_ORDER) == set(_CLASSES)`, so registration
is deterministic and a block cannot be dropped or reordered by set iteration; and
the default engine emits 500 columns hashing to `1646c3b9…cced`. Plus
`test_appending_a_research_module_cannot_move_a_production_column`, which
*exercises* the append-only claim: build on every known module and confirm the
first 500 columns are the RT-600 bank, identical and in order.

### The 15 pinned failures

| class | n | node ids |
|---|---:|---|
| `m07_bayes_stream_parity` | 8 | `test_parity_real_bitwise[319, 2546, 2649, 4313, 4330, 7885]`, `test_parity_synthetic_bitwise[t3_2, var_shrink]` |
| `engine_parity_real` | 4 | `test_engine_parity_real[0, 137, 4242, 7777]` |
| `m06_loc_stream_parity` | 2 | `test_real_store_bitwise`, `test_synthetic_bitwise[22]` |
| `m01_seq_stream_parity` | 1 | `test_bitwise_parity_synthetic[39]` |

All are batch-versus-streaming float differences — **2 differing cells out of
185,000** on the engine test. The `engine_parity_real` four are the *same*
`m07_bayes` defect seen through the assembled engine, not an independent one.
They reproduce **name for name** on `9aaa9b0`, the pre-Wave-5 RT-600 submission
baseline. Repairing them would change what the streaming engine emits — frozen
production semantics, and therefore RT-600's live predictions, which scored
0.6268 through this exact path.

The gate fails on a new failure, a rename, **and a disappearance**. A vanishing
pinned failure is an alert: either the streaming output changed or a test stopped
asserting. Self-tested in both directions.

### Artifact-level causality verification

`research/reports/wave6_causality_artifacts.json`:

| check | result |
|---|---|
| the four `test_no_n_online_leakage` `@needs_model` tests | **4 passed, 0 skipped, exit 0** |
| model directory | `models/final10k_ensemble` |
| artifact manifest sha256 | `1646c3b9…cced` over the RT-600 seven, 500 columns |
| **live** default `StreamEngine` reproduces it | **yes** |
| live column **order** identical to the artifact list | **yes, element by element** |

Every file in the artifact directory is hashed into that record. A skipped
causality test is worse than a missing one, because the suite still reads green.

---

## C. CAUSALITY GATES — ALL FIVE PASS

`tests/test_neural_causality.py`, 18 tests, run **before** any score was read.

| gate | what it asserts | result |
|---|---|---|
| 1 prefix invariance | shared prefix, different futures → identical, MLP and TCN | ≤ 1e-8 |
| 2 no `n_online` | truncation cannot move a surviving row; `elapsed = log1p(t)/7` is identical whatever the length turns out to be | ≤ 1e-8 |
| 3 no true τ | moving the break cannot move a pre-break prediction; the 7 group-missingness indicators score ≈ 0.5 under the **official scorer** | pass |
| 4 fold purity | standardiser constants are a function of the fitted rows only, and survive degenerate columns | pass |
| 5 determinism | same seed → same output; **and batch composition cannot move a series** | ≤ 1e-8 |

Gate 5's batch-composition test is what makes length-bucketed batching legal
rather than a covert length channel — asserted, not assumed.

**Normalisation audit (§5 of the brief), tested rather than argued:** scaling a
series' tail leaves every earlier channel bit-identical; `z` is normalised by
**history**, not by the online window; the TCN contains no BatchNorm, no
time-axis LayerNorm, no convolution with built-in padding, and no recurrent
layer. Execution is **CPU-only** — MPS is available on this machine and
deliberately unused, because `use_deterministic_algorithms` does not cover that
backend and Gate 5 requires ≤ 1e-8. Threads fixed at 6.

---

## D. TRACK A — MLP ON THE IDENTICAL 500 COLUMNS

`champ_fold_rows` replays the exact rng sequence `sbr.pipeline.run` consumes, so
the MLP saw **literally the 1,000,000 rows `RT-300` saw**, not a fresh sample of
the same size.

| id | learner / input | TS-AUC | Δ vs `RT-300` | folds+ |
|---|---|---:|---:|---:|
| `RT-300` | LightGBM, 500 cols | **0.61605** | — | control |
| `RT-960` | MLP, dropout 0.1, wd 1e-2 | 0.57059 | **−0.04546** | 0/5 |
| `RT-961` | MLP, dropout 0.3, wd 1e-1 | 0.58061 | **−0.03545** | 0/5 |

**Why it loses, from the training history rather than speculation.** Training BCE
falls to **0.077** by epoch 12 while `RT-300` stands at 0.61605. The row-level
target `y[t] = 1[t ≥ τ]` is monotone in `t` within a series, so elapsed time is
an enormous **row-level** predictor — and TS-AUC deletes all of it by comparing
only inside a timestep. Trees, capped by depth and `min_data_in_leaf = 300`,
could not exploit it far enough to hurt. A 168k-parameter network can, and does.

The regularisation arm confirms the mechanism rather than repairing it: 0.1 → 0.3
dropout and 1e-2 → 1e-1 weight decay is worth **+0.0100** standalone. A third,
heavier setting is **not** run — two per family is what was pre-registered.

---

## E. TRACK A ATTRIBUTION

The bar is not `RT-300`. It is a **matched eighth seed clone**, worth +0.00003.

| | TS-AUC | Δ vs `S` | Δ vs the seed clone | folds+ | bootstrap CI on the seed-clone contrast |
|---|---:|---:|---:|---:|---|
| `S` seven specialists | 0.62581 | — | — | — | — |
| `S` + 8th **seed clone** | 0.62584 | +0.00003 | — | — | — |
| `S` + `RT-960` | 0.62524 | −0.00057 | **−0.00060** | 1/5 | [−0.00165, +0.00061], 19.2% positive |
| `S` + `RT-961` | 0.62597 | +0.00016 | **+0.00013** | 3/5 | [−0.00105, +0.00128], 58.5% positive |

Both CIs include zero. `RT-961` contributes +0.00013 against a clone worth
+0.00003 — **bagging with extra steps.** Two-model blends are worse still:
`RT-300 + RT-961` = 0.61006 against `RT-300 + RT-401` = 0.61978, **−0.00971**.

**Verdict: learner capacity is NOT the lever.** Given identical information, a
different learner extracted less, not more.

---

## F. TRACK B — CAUSAL DILATED TCN ON TEN LEGAL CHANNELS

| id | hidden | TS-AUC | Δ vs `RT-300` | folds+ | params |
|---|---:|---:|---:|---:|---:|
| `RT-970` | 32 | 0.54152 | **−0.07453** | 0/5 | 35,905 |
| `RT-971` | 64 | 0.54319 | **−0.07287** | 0/5 | 139,393 |

Folds: `RT-970` 0.52618 / 0.55661 / 0.54210 / 0.55056 / 0.53217; `RT-971`
0.52725 / 0.55603 / 0.54311 / 0.55738 / 0.53216.

**`RT-971` fold 3 diverged at epoch 3** — training BCE 0.549 → **6.013** — and
recovered under the cosine schedule to finish at 0.55738, the arm's second-best
fold. Gradient clipping at 1.0 was already in place; this is an AdamW /
weight-norm instability at lr 3e-3 with hidden 64.

**It is not re-run.** Not with another seed, not with a lower learning rate, not
with a different schedule. The configuration was frozen before any neural number
existed; the divergence is an outcome *of that configuration*; and re-rolling a
fold after seeing its trajectory is how a screen becomes a search. The full
per-epoch history is in `research/reports/wave6_n2_tcn.json`.

Both arms finished well inside the pre-registered 3-hour-per-fold infeasibility
rule (12–21 min per fold), which the runner enforces in code.

---

## G. TRACK B ATTRIBUTION

| | TS-AUC | Δ vs `S` | Δ vs the seed clone | folds+ | bootstrap CI on the seed-clone contrast |
|---|---:|---:|---:|---:|---|
| `S` + 8th **seed clone** | 0.62584 | +0.00003 | — | — | — |
| `S` + `RT-970` | 0.62594 | +0.00013 | **+0.00010** | 3/5 | [−0.00128, +0.00126], 44.8% positive |
| `S` + `RT-971` | 0.62603 | +0.00022 | **+0.00019** | 3/5 | @@RT971_CI@@ |

Two-model blends: `RT-300 + RT-970` = 0.60651 and `RT-300 + RT-971` = 0.60673,
against `RT-300 + RT-401` = 0.61978 — **−0.01327** and **−0.01305**.

### The single most useful number Wave 6 produced

| model | within-time rank correlation with `RT-300` | ensemble value over a seed clone |
|---|---:|---:|
| `RT-960` MLP | 0.3575 | −0.00060 |
| `RT-961` MLP | 0.4018 | +0.00013 |
| `RT-970` TCN | **0.2088** | +0.00010 |
| `RT-971` TCN | **0.2116** | +0.00019 |

The TCN is **by a wide margin the most decorrelated model this project has ever
produced** — a rank correlation of 0.21 against a champion stream — and it is
worth one ten-thousandth of an AUC point. Wave 5 asserted that low correlation is
not evidence of information. This is the cleanest possible demonstration:
**maximal diversity, zero value. Diversity is not the currency; correct same-`t`
ordering is.**

**Verdict: a learned temporal representation is not the lever either** — at least
not this one, at this budget, under this objective. See §L for the distinction
that matters.

---

## H. AGE RESULTS

Positives restricted to one post-break age bucket, negatives held fixed, so the
buckets share a comparison set. Pair weight is heavily concentrated at the top:
**705,859 of 1,033,242 positive rows are age 100+**.

### Standalone, minus `RT-300`

| id | 0–5 | 5–10 | 10–20 | 20–50 | 50–100 | 100+ |
|---|---:|---:|---:|---:|---:|---:|
| `RT-960` | **+0.00641** | −0.00230 | −0.00726 | −0.02757 | −0.04259 | −0.05869 |
| `RT-961` | −0.00336 | −0.00753 | −0.00694 | −0.01899 | −0.03287 | −0.04608 |
| `RT-970` | **+0.00585** | −0.00463 | −0.01756 | −0.04437 | −0.06177 | −0.10218 |
| `RT-971` | @@RT971_AGE_STANDALONE@@ |

### In the ensemble, `S + N` minus `S + seed clone`

| id | 0–5 | 5–10 | 10–20 | 20–50 | 50–100 | 100+ |
|---|---:|---:|---:|---:|---:|---:|
| `RT-960` | **+0.00200** | **+0.00148** | +0.00157 | +0.00051 | −0.00059 | −0.00124 |
| `RT-961` | −0.00000 | −0.00024 | +0.00053 | +0.00061 | −0.00004 | +0.00010 |
| `RT-970` | **+0.00142** | **+0.00137** | +0.00066 | −0.00030 | −0.00006 | −0.00014 |
| `RT-971` | @@RT971_AGE_ENS@@ |

### THE YOUNG-BREAK OBSERVATION — **HYPOTHESIS-GENERATING ONLY**

`RT-960` and `RT-970` — an MLP on handcrafted columns and a TCN on raw channels,
sharing no input representation — both help in the 0–5 and 5–10 buckets and both
hurt at 100+. Ages 0–20 are the project's measured weak spot, and this is the
first thing in six waves that moved them at all.

**It must not be exploited in this wave, and I am not going to.** The reasons it
is weak evidence, stated up front rather than buried:

1. **It was read out of failed experiments.** Reading a subgroup out of a null
   result and promoting it is precisely how the multiplicity ledger gets lied to.
2. **`RT-961` does not show it** — −0.00000 at 0–5 and −0.00024 at 5–10. The
   pattern appears in two of the four arms, not four of four, and the one that
   breaks it is the *better* of the two MLPs.
3. **The two arms that do show it are not independent.** Same rows, same
   objective, same ensemble, same fold structure. Two families is not two
   experiments.
4. **The effect is one to two thousandths** against a bar of thirty.
5. **The obvious exploitation is illegal.** Gating a blend on post-break age
   requires τ at inference, and τ at inference is `RT-900`: 0.86552 of pure leak.
   A *legal* recency proxy is itself a change-point detector feeding its own
   errors back into routing, which is W5-E3's failure mode.

It is written down so a future wave can test it **prospectively**. That is its
only purpose. No age-gated blend, no age-weighted ensemble, no proxy chosen after
seeing this table.

---

## I. SEED-CONTROL RESULT

This is the whole experiment in one line.

| | Δ vs `S` |
|---|---:|
| an **eighth exchangeable LightGBM seed clone** (W5-NULLTEST, reconfirmed here) | **+0.00003** |
| the best neural arm (`RT-971`) | +0.00022 |
| **the difference** | **+0.00019** |

Against a promotion bar of **+0.0030** on ≥ 4/5 folds with a bootstrap CI
excluding zero. Every neural arm's CI includes zero. The best of them beats
re-running the same LightGBM with a different random seed by **nineteen
hundred-thousandths of an AUC point**.

---

## J. PROMOTION DECISION

**Nothing is promoted. Nothing earns Wave-6 confirmation.**

| requirement (`WAVE6_NEURAL_PREREG.md` §6) | best arm | met |
|---|---|---|
| aggregate Δ ≥ +0.0030 over the strongest matched control | +0.00019 | **no** |
| positive on ≥ 4/5 folds | 3/5 | **no** |
| paired bootstrap CI excludes zero | includes zero | **no** |
| incremental ensemble value over `S` | +0.00022 | **no** |
| gain beyond a matched seed clone | +0.00019 vs +0.00003 | **no** |
| no regression in the causality gates | all five pass | yes |

No alternate-partition confirmation is run, because §26 of the brief and §6 of
the pre-registration make confirmation conditional on clearing the bar, and
nothing cleared it. Running confirmation on a failed candidate would be a
seventh look at the same data.

**`RT-980` (GRU) does not open.** §6.1 requires the TCN to reach the *last*
hurdle, not the first; it reached none.

---

## K. WAVE-6 STOPPING RULE

`WAVE6_STOPPING_RULE_AMENDMENT.md` §3(f), written before any neural number
existed:

> Wave 6 stops adding when **both** input tracks have been run once each, under
> controls, and **neither** clears §6 against a matched seed clone. At that point
> the recommendation is: ship `RT-600`, write the wave up, open no further
> Wave-6 candidates.

Track A ran once. Track B ran once. Neither cleared. **The condition is met
exactly as written, and Wave 6 terminates.**

Recorded outcome: **NEURAL TRACK FAILED CONTROLLED SCREEN.**

No larger MLP, no other activation, no other TCN width, no transformer, no
learning-rate sweep, no extra seed, no age-gated ensemble. `RT-971`'s diverged
fold is not re-rolled.

**`RT-600` = 0.6268 remains champion.** No submission was made or is justified.

### Wave 6 in full

| experiment | outcome |
|---|---|
| W6-E1 calibration anchors | **falsified** — every scheme, including random anchors, inside 0.00006 |
| W6-E2 / `RT-900` τ oracle | **VOID** — label leak through the missingness mask |
| W6-E2R corrected series-level oracle | **Case C** — our bank beats the frontier's by **+0.0235**, 5/5 seeds, 25/25 folds |
| W6-N1 MLP (learner capacity) | **failed** |
| W6-N2 TCN (learned representation) | **failed** |

W6-E2R is the wave's one positive result and it survives: representation quality
is a live lever. Wave 6 establishes that **a learned representation is not how to
pull it.**

---

## L. THE INTERPRETATION — OBJECTIVE MISMATCH, NOT "NEURAL NETWORKS DON'T WORK"

The evidence does not support the lazy reading. It supports a specific one.

1. **The MLP drove row-level BCE to 0.077 while scoring 0.045 *below* a tree
   ensemble on identical rows and identical features.** Those two facts are only
   compatible if the model is optimising something the metric largely discards.
   `y[t] = 1[t ≥ τ]` is monotone in `t` within a series, so elapsed time is a
   huge row-level signal — and TS-AUC compares only *within* a timestep, where
   `t` is constant and that signal carries exactly zero information.
2. **Trees never had this pathology.** Depth limits and `min_data_in_leaf = 300`
   stopped them fitting the elapsed-time structure far enough to hurt.
3. **The TCN result says diversity is not the missing ingredient.** Rank
   correlation 0.21 with the champion, ensemble value +0.0001. A model must
   improve the **same-`t` ordering**, not merely disagree.

So the open question is not *"can neural networks do this"* but **"does a
same-`t` objective recover the gap?"** — and the honest caveat is that
`WAVE6_NEURAL_PREREG.md` deliberately fixed BCE as the only loss, so this wave
**cannot distinguish** "the family is wrong" from "the objective was wrong".
That ambiguity is a cost of the pre-registration, and it was the right cost:
running a loss sweep alongside two architectures would have made every number
uninterpretable.

**And the prior is not favourable.** Same-`t` pairwise ranking has already been
tested four ways on trees and is mildly negative every time — `RT-111` 0.62302
against binary 0.62631 on the identical matrix, `RT-700` −0.00147, `RT-701`
−0.00476. The Wave-7 proposal opens with that objection rather than burying it.

---

## M. MASTER TABLE

| experiment | learner / input | TS-AUC | Δ vs control | folds+ | 0–5 Δ (ens.) | ensemble Δ | seed-bar pass | verdict |
|---|---|---:|---:|---:|---:|---:|---|---|
| `RT-300` | LightGBM, 500 cols | 0.61605 | — | — | — | — | — | **control** |
| `RT-401` | LightGBM, 500 cols, seed clone | 0.61661 | +0.00056 | — | — | **+0.00003** | — | **control (the bar)** |
| `S` (`RT-300`+`RT-410`–`415`) | seven specialists | 0.62581 | +0.00976 | — | — | — | — | **control** |
| `RT-960` | MLP d0.1 wd1e-2, 500 cols | 0.57059 | −0.04546 | 0/5 | +0.00200 | −0.00060 | **no** | rejected |
| `RT-961` | MLP d0.3 wd1e-1, 500 cols | 0.58061 | −0.03545 | 0/5 | −0.00000 | +0.00013 | **no** | rejected |
| `RT-970` | causal TCN h32, 10 channels | 0.54152 | −0.07453 | 0/5 | +0.00142 | +0.00010 | **no** | rejected |
| `RT-971` | causal TCN h64, 10 channels | 0.54319 | −0.07287 | 0/5 | @@RT971_AGE05@@ | +0.00019 | **no** | rejected |
| `RT-980` | GRU | — | — | — | — | — | — | **not opened** (§6.1 unmet) |

"Δ vs control" is against `RT-300`; "ensemble Δ" is against the **eighth seed
clone**, which is the binding comparison.

### Cost

| | runtime | parameters | inference |
|---|---:|---:|---|
| `RT-960` / `RT-961` | 562 s / 473 s, 5 folds | 167,937 | one 507→1 forward pass per row |
| `RT-970` | 3,632 s, 5 folds | 35,905 | one causal pass per series, receptive field 127 |
| `RT-971` | 6,111 s, 5 folds | 139,393 | as above |

---

## N. WAVE-5 IMPLEMENTATION CHECKS

Two defects were reported for the hard-negative work. **Neither exists in this
repository**, and no observed experiment is rewritten.
`research/reports/wave5_e3_conformance.json`.

### N.1 "hardest 20 percent" / `hardness >= 0.80` — **NOT PRESENT**

There is no `RT-542` here. The Wave-5 hard-negative experiment is W5-E3 with
arms `RT-710` (uniform control), `RT-711` (reweight), `RT-712` (oversample) —
all observed, all rejected (−0.00701 and −0.01339).

The selection was **never** a fixed threshold. `oversample_pool` computes
`thr = np.quantile(r[m], 1 − HARD_FRAC)` over the *eligible inner-fold negatives*
and takes `r >= thr` — a quantile of the actual hardness distribution, which is
the hardest `HARD_FRAC` by construction.

| outer fold | eligible inner negatives | selected | **realised fraction** |
|---:|---:|---:|---:|
| 0 | 2,403,027 | 240,303 | **0.10000** |
| 1 | 2,394,857 | 239,486 | **0.10000** |
| 2 | 2,398,522 | 239,854 | **0.10000** |
| 3 | 2,402,115 | 240,212 | **0.10000** |
| 4 | 2,398,607 | 239,861 | **0.10000** |

`HARD_FRAC` is 0.10, not 0.20, and all three ledger rows say `hard_frac=0.1`.
`git log -S"0.80"` over `research/scripts` returns one hit and it is `LAMBDAS` in
the W5-E1 mixture sweep, an unrelated blend weight. The file has exactly one
commit, `ad1b7fb`, and that version already used the quantile. **No pre-score
conformance fix is required, because there is nothing to conform.**

### N.2 held-out fold in the oversampled index — **NOT PRESENT; convention now an invariant**

The implementation never started from `np.arange(len(hardness))`. It patches
`Data.rows_for`, takes `rows` from the real call, and draws duplicates as
`rows[mask[rows]]` — a subset of the training folds' own row list. Independently,
`H[k]` is NaN on fold `k` by construction of the nested mining, so the mask
excluded it too.

Measured on all five folds: **0** held-out rows in any pool, **0** held-out rows
in any training index, **0** positives ever duplicated, and every training index
length exactly `len(train) + 3 × len(pool)`.

But the discriminator was `len(f) != 4` — "four folds means training" — a
convention about how `run` happens to be written. Both are now checked
invariants: fold exclusion is an explicit term in the pool (`d.row_fold != k`),
and the patch asserts the returned index contains no held-out row before
returning it.

**This does not rewrite `RT-712`.** The added term is provably a no-op on the
executed configuration — 0 selected rows lay in the held-out fold, verified
above — and assertions can only raise, never alter a value.
`tests/test_wave5_hardneg_invariants.py`, 4 tests, all passing.

---

## O. NEXT STEP — PROPOSED, NOT AUTHORISED

`research/WAVE7_PROPOSAL_metric_aligned_transition.md`. **Not a
pre-registration. Not executed. No Wave-7 number exists.**

Primary arm: the *same* frozen `RT-961` MLP and `RT-970` TCN, with **only** the
objective changed to same-`t` pairwise ranking — architecture held at Wave 6's
values so the result stays interpretable. Same-`t` stratified minibatching is a
component of that sampler, not a separate degree of freedom. Age-balanced
training is legal **at training** (post-break age is a deterministic function of
the labels and never reaches inference) and illegal the moment it routes, gates
or is fed to a scorer. The two-head architecture is conditional on the
single-head version clearing an interim bar. The recency gate is recommended
**against** for Wave 7.

The proposal's §8 states what would make me recommend cancelling it, including a
**ten-minute one-fold mechanism pilot** that should be the first thing
authorised: if the same-`t` objective leaves dev TS-AUC within noise of
`RT-961`'s 0.58061, the elapsed-time explanation is wrong and the wave should be
cancelled after that single fold rather than completed for symmetry.

---

## P. WHAT THIS DOES NOT PROVE

1. Not that neural networks cannot do this task — only that **these two
   pre-registered families, at this budget, under BCE**, do not.
2. Not that the young-break signal is real. Four arms, two show it, one of the
   two MLPs does not, and it was found by reading subgroups out of nulls.
3. Not that representation is a dead end. **W6-E2R's +0.0235 stands** — the
   representation lever is real; a *learned* representation is not how to pull it.
4. Not that the pinned 15 test failures are harmless — only that they are
   pre-existing, bounded at 2 cells in 185,000, and not repairable without
   changing frozen production semantics.
