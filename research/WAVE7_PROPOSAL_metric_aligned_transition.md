# WAVE 7 — PROPOSAL: METRIC-ALIGNED TRANSITION SPECIALISTS

**STATUS: PROPOSAL. NOT PRE-REGISTERED. NOT AUTHORISED. NOT EXECUTED.**
Written 2026-08-22 at the close of Wave 6. No Wave-7 number exists, and none may
be produced until this document is converted into a signed pre-registration and
committed on its own branch.

Parent evidence: `research/reports/wave6_neural_results.md`,
`research/WAVE6_NEURAL_PREREG.md`, `research/WAVE6_STOPPING_RULE_AMENDMENT.md`.

---

## 0. THE HONEST OBJECTION, FIRST

**Same-t pairwise ranking is not a new idea in this project. It has been tested
repeatedly on gradient-boosted trees and it is mildly NEGATIVE.**

| id | objective | dev TS-AUC | against |
|---|---|---:|---|
| `RT-111` | pairwise logistic, groups = `t` | 0.62302 | **binary 0.62631 on the identical 800k matrix, −0.00330** |
| `RT-413` / `RT-123R` | `pairwise_t`, ensemble stream D | 0.61481 | the ensemble's weakest-but-useful member |
| `RT-700` | pairs weighted by `n_neg(t)` — the metric's own weighting | 0.61334 | `RT-702` control 0.61481, **−0.00147** |
| `RT-701` | weighted squared hinge | 0.61005 | `RT-702` control, **−0.00476** |

Wave 5 wrote the conclusion down: *"a metric that is entirely pairwise ranking
sounded right, and the data disagrees."* Any Wave-7 proposal that opens with
"align the loss with the metric" is proposing something this project has already
falsified once, and must say so in its first paragraph. This one does.

**So what is actually new?** The argument is not that pairwise ranking is good.
It is that **trees never had the pathology pairwise ranking fixes, and the
neural models do.**

| | training BCE reached | dev TS-AUC |
|---|---:|---:|
| `RT-300` LightGBM, binary, same rows | — | **0.61605** |
| `RT-960` MLP, binary, same rows | **0.077** | 0.57059 |
| `RT-961` MLP, binary, heavier regularisation | 0.077 | 0.58061 |
| `RT-970` / `RT-971` TCN, masked BCE | 0.537 | 0.54152 / 0.54319 |

A 168k-parameter MLP drove row-level BCE to 0.077 — near-perfect row-level
separation — while scoring **0.045 BELOW** a tree ensemble on the metric. The
row-level target `y[t] = 1[t >= tau]` is monotone in `t` within a series, so
**elapsed time is an enormous row-level predictor and TS-AUC deletes all of it**
by comparing only inside a timestep. Trees, constrained by depth and by
`min_data_in_leaf = 300`, could not exploit it far enough to hurt. A network can
and does.

That makes the two situations genuinely different. For trees, pairwise ranking
replaced a working objective and cost a little. For a network, the binary
objective is measurably solving the wrong problem. **This is the one condition
under which retrying a falsified intervention is legitimate: the mechanism it
targets has been shown to be present in the new setting and absent in the old.**

**And it is still probably not enough.** Section 8 states what would falsify the
whole wave before it starts.

---

## 1. WHAT WAVE 6 ESTABLISHED THAT THIS PROPOSAL RESTS ON

1. **Diversity is not the currency.** The TCN's within-time rank correlation
   with `RT-300` is **0.21**, the most decorrelated model this project has
   produced, and its ensemble contribution is **+0.0001** against a seed clone
   worth +0.00003. Low correlation buys nothing. The model has to improve the
   **same-`t` ordering**, not merely differ.
2. **Both neural families helped in the same narrow place.** Independently:

   | age bucket | `S+RT-960` − `S+clone` | `S+RT-970` − `S+clone` |
   |---|---:|---:|
   | 0–5 | +0.00200 | +0.00142 |
   | 5–10 | +0.00148 | +0.00137 |
   | 10–20 | +0.00157 | +0.00066 |
   | 50–100 | −0.00059 | −0.00006 |
   | 100+ | −0.00124 | −0.00014 |

   **HYPOTHESIS-GENERATING ONLY.** Two families is not two experiments; both
   were trained on the same rows with the same objective against the same
   ensemble, so the agreement is weaker evidence than it looks. It was found by
   reading subgroups out of failed experiments, which is precisely the procedure
   that manufactures findings. It is written here so that a future wave can test
   it **prospectively**, and for no other purpose.
3. **The metric's weight is where the neural models are worst.** 705,859 of
   1,033,242 positive rows are age 100+. A model that is +0.002 at age 0–5 and
   −0.001 at 100+ loses on aggregate by construction.

---

## 2. THE WAVE-7 QUESTION

> Does a causal model trained against a **same-`t`** objective retain the early
> post-break signal the Wave-6 neural models showed, **without** the mature-break
> loss that cancelled it — and does the result add information to the
> seven-specialist ensemble beyond a matched seed clone?

Two sub-questions, to be separated as Wave 6 separated learner from
representation:

* **Q1 — objective.** Does replacing row-level BCE with a same-`t` objective
  recover the gap between the neural models and `RT-300` on the *same* inputs?
* **Q2 — specialisation.** Does an architecture and a sampling scheme that
  explicitly target *recent transitions* beat a same-`t` model that does not?

---

## 3. THE FIVE COMPONENTS, ASSESSED ON THE RESEARCH CASE

The brief asks for the case for each. Assessed honestly, they are not equally
strong, and a pre-registration should not run all five.

### 3.1 Same-`t` pairwise ranking loss — **STRONGEST. The primary arm.**

Directly optimises `P(score(pos_t) > score(neg_t))` at the same online index,
which is what TS-AUC integrates. The pair sampler already exists and is already
metric-correct: `sbr.pipeline._make_pairwise_t` groups by online index `t`,
draws `m_neg = 8` negatives per positive **inside the group**, and is the
objective stream D ships with.

* **Case for:** it removes exactly the elapsed-time channel that the BCE
  training history shows the network exploiting. Within a timestep, `t` is
  constant, so it carries literally zero gradient.
* **Case against:** already negative on trees, four separate ways (§0).
* **Ties.** TS-AUC uses **mid-ranks**; exact score ties contribute 0.5.
  Continuous network outputs make exact ties measure-zero, so a plain logistic
  pairwise loss is consistent with the metric. A pre-registration must state
  this rather than assume it, and must **not** add a tie-aware margin term
  without declaring it as a separate arm.
* **Falsifiable prediction, to be fixed before running:** if the mechanism in §0
  is right, the same-`t` MLP's **training** BCE-equivalent will stop collapsing
  and its dev TS-AUC will rise by **more than +0.02** over `RT-961`'s 0.58061.
  Less than that and the elapsed-time story is wrong regardless of the final
  score, and the wave should be reported as such.

### 3.2 Same-`t` stratified minibatching — **STRONG, but it is a component of 3.1, not an arm.**

A minibatch built by sampling *timesteps* and then rows within them, rather than
rows uniformly. Necessary for 3.1 to have well-populated groups; on its own,
with a BCE loss, it changes nothing about what is optimised. **Recommendation:
fold it into the primary arm's sampler and do not spend a degree of freedom on
it separately.**

### 3.3 Positive-age-balanced training — **MODERATE, and it needs a legality ruling stated explicitly.**

Reweight or resample positive rows so the age distribution seen in training is
flatter than 68% age-100+.

* **Legality: LEGAL AT TRAINING, and here is the exact reason.** Post-break age
  `t − tau` is a deterministic function of the *training labels* — `y` is the
  step function whose transition is `tau`, so age is recoverable from `y` alone
  with no extra information. Weighting training rows by a function of the label
  is ordinary class balancing. `tau` never enters a feature, a normalisation
  constant, a mask, or a hidden state, and **at inference the model receives no
  age input at all**.
* **The illegal cousin, stated so the line cannot blur.** Using true age or true
  `tau` to *route* a row to a model, to *gate* a blend, or as an input at
  inference is forbidden — `tau` is unknown at inference and that is the whole
  problem. RT-900 is the standing example of what happens when boundary
  knowledge reaches a row-level scorer: 0.86552 of pure leak.
* **Case against:** it optimises a different objective from the metric. TS-AUC's
  weight *is* concentrated at high age (57% of pair weight at 100+), so flattening
  the training age distribution deliberately down-weights where the score is won.
  This arm is a specialisation bet, not a general improvement, and it should only
  be run as an ensemble MEMBER candidate, never as a champion candidate.

### 3.4 Two-head persistent + recent-transition architecture — **MODERATE. Second arm at most.**

One head trained on mature positives, one on young positives, combined by a
learned, causal, input-dependent mixture.

* **Case for:** it is the prospective test of §1.2, and it keeps the mature head
  from being damaged by the young objective.
* **Case against:** the combination weight is where this becomes illegal by
  accident. If the mixture is fitted using age, it is 3.3's illegal cousin
  wearing a different hat. The gate must be that **the mixture depends only on
  the causal state at time `t`** and is trained end-to-end against the same-`t`
  objective, never against an age target.
* **Recommendation:** run only if the primary arm clears §7's interim bar. A
  two-head model whose single-head version failed is architecture fishing.

### 3.5 Causal transition-recency gate — **WEAKEST AS PROPOSED, and the most dangerous.**

An observable proxy for "a transition happened recently".

* **Case for:** it is the legal counterpart of the illegal age gate, and the
  500-column bank already contains causal change-recency evidence — `m11_focus`
  computes the exact `argmax` over candidate `tau` and its inferred age.
* **Case against, and it is serious:** *any* estimate of transition recency is
  itself a change-point detector, so gating on it means the pipeline's own
  errors feed back into which model scores the row. W5-E3 is the standing
  warning: the last time this project let a mined score decide training
  emphasis, it lost **0.00701** and **0.01339** and damaged the young buckets it
  was designed to help.
* **Recommendation: DO NOT run in Wave 7.** If §1.2 survives a prospective test,
  a recency gate becomes a Wave-8 question with its own controls. Proposing it
  now, on the strength of a subgroup read out of two failed experiments, is
  exactly the sequence that produces a fake result.

---

## 4. PROPOSED ARMS — DELIBERATELY FEW

| id | model | objective | tests |
|---|---|---|---|
| `W7-P1` | the `RT-961` MLP, **unchanged** | same-`t` pairwise, `m_neg = 8` | Q1 on identical inputs |
| `W7-P2` | the `RT-970` TCN, **unchanged** | same-`t` pairwise | Q1 on learned representation |
| `W7-P3` | `RT-961` MLP | same-`t` pairwise + age-balanced positive sampling | 3.3, ensemble-member candidate only |
| `W7-P4` | two-head | same-`t` pairwise | 3.4, **conditional** on `W7-P1` or `W7-P2` clearing §7 |

**Architectures are held at Wave 6's frozen values.** Changing the objective and
the architecture together makes the result uninterpretable, which is the rule
`WAVE6_PREREG.md` §7 already binds. Four arms, one conditional. No transformer,
no width sweep, no learning-rate sweep, no loss sweep beyond the single declared
objective, no seed fishing.

---

## 5. WHAT A PRE-REGISTRATION MUST FIX BEFORE ANY SCORE

1. **Sampling.** Group = online index `t`. `m_neg` per positive. Whether groups
   are weighted by `n_pos(t)·n_neg(t)` (the metric's weighting) or uniformly —
   **note that `RT-700` tested exactly that weighting on trees and it lost
   0.00147**, so the default must be uniform and the weighted version is a
   separate declared arm or nothing. Minimum group occupancy. Behaviour when a
   timestep has one class only (contributes zero weight, as in the metric).
2. **Loss.** Pairwise logistic on the score difference. Clipping. Tie handling
   (§3.1). Exactly one alternative may be declared, and it must be declared
   before running.
3. **Fold purity.** Pairs drawn only from training folds; no pair may span the
   validation fold; standardisation constants from training rows only; asserted
   in code with `_assert_fold_pure`, not by convention. The five Wave-6 gates
   apply unchanged and run before any score.
4. **Acceptance thresholds.** §6-of-Wave-6's bar, unchanged: ≥ +0.0030 aggregate
   over the strongest matched control, ≥ 4/5 folds, paired bootstrap CI
   excluding zero, incremental ensemble value over `S`, and **gain beyond a
   matched seed clone**. Alternate partitions for architectural promotion.
5. **Falsification, per arm**, including §3.1's mechanism test (+0.02 over
   `RT-961` or the elapsed-time explanation is wrong).
6. **Seed controls.** One seed per configuration; a replicate seed only for an
   arm that has already cleared the bar. The matched seed clone remains the
   binding control.
7. **Stopping rule.** Fixed before running, in the shape of Wave 6's: if `W7-P1`
   and `W7-P2` both fail, `W7-P3` and `W7-P4` do not open, and the neural
   programme closes rather than escalating.
8. **Age-bucket reporting** is mandatory and goes in the body, not an appendix.

---

## 6. WHAT WAVE 7 MAY NOT DO

* No blend weighted by true `tau` or true post-break age. No age-gated ensemble.
  No proxy for age **chosen after seeing Wave 6's subgroup table** — a proxy is
  legal only if it is declared in the pre-registration and justified
  independently of §1.2.
* No re-running of any Wave-6 arm with a different seed, learning rate,
  schedule, or architecture. `RT-960`, `RT-961`, `RT-970`, `RT-971` are closed,
  including `RT-971`'s diverged fold 3.
* No submission. `RT-600` = **0.6268** remains LB-001 and remains champion until
  a pre-registered candidate clears the bars and passes confirmation.
* No change to production feature or streaming semantics, and no repair of the
  15 pinned known failures (`research/known_failures.json`) to make a suite
  green.

---

## 7. INTERIM BAR FOR OPENING THE CONDITIONAL ARMS

`W7-P4` opens only if `W7-P1` or `W7-P2` clears **all** of: aggregate delta
≥ +0.0030 against `RT-300` on identical inputs, ≥ 4/5 folds, bootstrap CI
excluding zero, **and** a non-negative eighth-member delta against the matched
seed clone. Reaching the last hurdle, not the first.

---

## 8. WHAT WOULD MAKE ME RECOMMEND NOT RUNNING THIS AT ALL

Stated now, while it is still cheap to say.

* If a **one-fold pilot** of `W7-P1` does not move the mechanism indicator — if
  the same-`t` objective leaves the model's dev TS-AUC within noise of
  `RT-961`'s 0.58061 — then the elapsed-time explanation is wrong, the neural
  deficit is something else, and Wave 7 should be cancelled after that single
  fold rather than completed for symmetry. **A one-fold mechanism pilot should be
  the first thing the pre-registration authorises, and its cost is about ten
  minutes.**
* The prior is not favourable. Trees closed a 0.045 gap that networks could not,
  on identical rows and identical features, and four separate pairwise-objective
  experiments have already come back negative. The most likely Wave-7 outcome is
  that the same-`t` objective recovers part of the gap and still does not reach
  `RT-300`, let alone add to a seven-member ensemble.
* If that is the outcome, the right conclusion is the one Wave 6 already
  supports: **this problem is solved by causal feature engineering plus
  gradient-boosted trees, and the remaining headroom is not in the model
  family.** W6-E2R's +0.0235 says the representation lever is real; it says
  nothing about a *learned* representation being the way to pull it, and Wave 6
  is evidence that it is not.
