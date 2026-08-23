# WAVE 7 — THE ALPHA BUDGET

**Branch `claude/project-setup-github-20g6pt`, parent `research/wave6-alpha` @ `33fc210`
merged with the LB-001..005 release line @ `9aaa9b0`.**

Producers: `research/scripts/wave7_alpha_budget.py` (W7-D1) and
`research/scripts/wave7_teacher_privilege.py` (W7-D2).
Machine-readable output: `research/reports/wave7_alpha_budget.json`,
`research/reports/wave7_teacher_privilege.json`.

> **TS-AUC VALUES OBSERVED IN WAVE 7 = 0 · TRAINING RUNS = 0 · SUBMISSIONS = 0.**
> The 2026 store is absent from this container and `crunchdao.com` is refused at
> the egress proxy (403 at CONNECT, recorded in the proxy's own failure log). No
> pilot in this wave has been run. Everything below is computed from
> **version-controlled artifacts** and is reproducible here today.

---

## 1. WHAT IS EXACT AND WHAT IS IMPORTED

This distinction governs how much weight each number can carry, so it comes first.

| quantity | status | source |
|---|---|---|
| every **pair weight** `w_t = n_pos(t)·n_neg(t)`, and its split across current-`t` buckets, post-break-age buckets, their **joint** cells, tau-fraction quartiles and online-length buckets | **EXACT, computed here** | `research/folds/folds.parquet` — 8,000 dev series with their true `tau_index`, `n_online` and fold |
| per-bucket **AUC** | **IMPORTED, not recomputed** | `research/WAVE5_STATUS.md` §5 (age, RT-600 architecture) and `research/reports/champion_diagnostics.json` (current-`t`, tau quartile, online length, RT-300-class) |

Scoring the AUCs afresh needs OOF prediction vectors, which need the store. So
the budget is: **exact weights × measured AUCs.**

### The reconstruction gate

`Σ_B share_B · AUC_B = TS-AUC` is an identity. The script refuses to emit a
budget whose parts do not rebuild the whole:

| decomposition | reconstructed | independently reported | error | |
|---|---:|---:|---:|---|
| by current `t` | 0.61500 | 0.61500 | **0.00000** | PASS |
| by post-break age | 0.62563 | 0.62581 | 0.00018 | PASS |

The `t` reconstruction is exact to five decimals against a number computed on a
different machine, in a different session, from data this container has never
seen. That is the strongest available evidence that the weight geometry is right.
The age residual is rounding in the published 5-decimal AUCs.

**Arm caveat, stated once and not repeated:** the age budget is the **RT-600
architecture** (0.62581). The `t`, tau-fraction and online-length budgets are the
**RT-300-class single model** (0.61500), because no RT-600 by-`t` decomposition
has ever been committed. RT-600 sits +0.0108 above it, so its `t`-bucket losses
are somewhat smaller than shown. The **shape** across `t` — which is all the
budget is used for — is what transfers.

---

## 2. THE BUDGET BY POST-BREAK AGE — RT-600 ARCHITECTURE

Total remaining weighted loss: **0.37437**.

| age | weight share | AUC | loss | **% of all loss** | perfect-repair ceiling | one whole architecture generation was worth |
|---|---:|---:|---:|---:|---:|---:|
| **100+** | 0.5672 | 0.66151 | 0.19198 | **51.3%** | +0.19198 | +0.01065 |
| 50–100 | 0.1839 | 0.60516 | 0.07261 | 19.4% | +0.07261 | +0.00919 |
| 20–50 | 0.1392 | 0.57702 | 0.05889 | 15.7% | +0.05889 | +0.00777 |
| 10–20 | 0.0527 | 0.54880 | 0.02379 | 6.4% | +0.02379 | +0.01009 |
| 0–5 | 0.0290 | 0.51517 | 0.01408 | 3.8% | +0.01408 | +0.00440 |
| 5–10 | 0.0279 | 0.53408 | 0.01302 | 3.5% | +0.01302 | +0.00806 |

The last column is the yardstick that makes the ceilings readable: moving from a
single LightGBM to the seven-specialist ensemble — a whole architecture
generation, the largest single step this project has taken — bought between
+0.0044 and +0.0107 per bucket. **Ceilings are three to forty times larger than
anything the project has ever actually captured in one step.** Treat a ceiling as
a disqualifier, never as a forecast.

## 3. THE BUDGET BY CURRENT ONLINE INDEX — RT-300-CLASS

| current `t` | weight share | AUC | loss | **% of all loss** | ceiling |
|---|---:|---:|---:|---:|---:|
| **200–400** | 0.3658 | 0.61608 | 0.14045 | **36.5%** | +0.14045 |
| **400–1000** | 0.3203 | 0.64731 | 0.11297 | **29.3%** | +0.11297 |
| 100–200 | 0.1953 | 0.59960 | 0.07818 | 20.3% | +0.07818 |
| 50–100 | 0.0792 | 0.55387 | 0.03533 | 9.2% | +0.03533 |
| 20–50 | 0.0312 | 0.54599 | 0.01418 | 3.7% | +0.01418 |
| 0–20 | 0.0082 | 0.52466 | 0.00389 | 1.0% | +0.00389 |

**86% of remaining loss sits at `t ≥ 100`. Everything before `t = 50` is 4.7% of
the loss with a perfect-repair ceiling of +0.018.**

## 4. THE JOINT TABLE — WHERE THE LOSS ACTUALLY LIVES

Share of total pair weight. Rows are the current online index, columns are the
post-break age of the positive. Blank cells are impossible (age cannot exceed `t`).

| `t` \ age | 0–5 | 5–10 | 10–20 | 20–50 | 50–100 | 100+ | **row** |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–20 | 0.37% | 0.24% | 0.20% | · | · | · | 0.82% |
| 20–50 | 0.47% | 0.48% | 0.90% | 1.28% | · | · | 3.12% |
| 50–100 | 0.51% | 0.53% | 1.06% | 3.27% | 2.55% | · | 7.92% |
| 100–200 | 0.65% | 0.65% | 1.32% | 3.99% | 6.71% | 6.21% | 19.52% |
| 200–400 | 0.62% | 0.62% | 1.25% | 3.70% | 6.31% | **24.08%** | 36.58% |
| 400–1000 | 0.28% | 0.28% | 0.55% | 1.68% | 2.82% | **26.43%** | 32.03% |
| **col** | 2.90% | 2.79% | 5.27% | 13.92% | 18.39% | **56.72%** | 100% |

**Two cells hold 50.5% of all weighted pairs**, and on the flat-AUC-within-age
assumption ≈45.7% of all remaining loss:

> a break that happened **more than 100 observations ago**, in a series observed
> for **more than 200 online steps** — and the model still inverts roughly a third
> of those pairs against a series that never breaks at all.

That is the competition, stated in one sentence. It is not a young-break problem
and it is not a transient-shock problem.

## 5. NAMED ERROR CLASSES

Overlapping by construction; ranked by measured loss.

| error class | weight | **% of loss** | perfect-repair Δ | currently targeted by | priority |
|---|---:|---:|---:|---|---|
| mature persistent break vs no-break, late online (`t≥200`, age≥100) | 0.505 | **45.7%** | +0.17096 | **nothing in the wave-7 brief** | **1** |
| mid-age break (20 ≤ age < 100) | 0.323 | 35.1% | +0.13150 | lane B (horizon specialists) | 2 |
| young break (age < 20) — transient-vs-persistent territory | 0.110 | 13.6% | +0.05089 | lane A permanence target, W6 neural young-age effect | 3 |
| short online segment (`n_online < 200`) | 0.076 | 9.1% | +0.03406 | lane B H1–H3 | 4 |
| late break (`tau > 0.74` of the segment) | 0.040 | 4.5% | +0.01703 | — | 5 |
| heavy-tail / outlier false alarm (top 1% hardest negatives) | 0.010 | 1.5% | +0.00922 | W5-D2 diagnosis; no module | 6 |

**The heavy-tail class deserves a specific note, because it is the failure mode
this project has diagnosed most confidently and it is nearly worthless.** W5-D2
found the top 1% of no-break series sit at mean within-timestep percentile rank
0.9221 against 0.4731 for negatives overall — an unambiguous, mechanistically
clean finding. Sized against the metric: **fixing those series perfectly buys at
most +0.0092, and the excess they carry over an average negative is +0.0055.**
A confident diagnosis of a 1.5% error class is not a research programme.

Classes that **cannot** be sized without OOF vectors, listed rather than guessed:
variance-only break, dependence-only break, weak magnitude, online-vs-history
offset. The last of these matters most: `by_break_class` puts **92.8% of break
series (3,673 of 3,958) in `weak_unclassified` at AUC 0.6005**, against 0.828 for
`mixed`, 0.697 `dep_dominant`, 0.691 `shape_dominant`, 0.687 `trend_dominant`.
Every *identifiable* break family is already handled well. The budget is almost
entirely made of breaks too weak to classify — which is the same population as
the dominant joint cell, seen from a different angle.

## 6. W7-D2 — WHAT A FULL-SEQUENCE TEACHER ACTUALLY KNOWS

Lane A's ceiling is bounded by how much series the student has **not** seen where
the metric puts its weight. Every unseen point of a positive row is post-break, so
the teacher's advantage on the detection question is exactly how many more times
it observes the post-break regime.

**Teacher privilege ratio** — post-break points the teacher holds ÷ post-break
points the student holds:

| `t` \ age | 0–5 | 5–10 | 10–20 | 20–50 | 50–100 | 100+ |
|---|---:|---:|---:|---:|---:|---:|
| 0–20 | 73.5× | 26.0× | 15.0× | · | · | · |
| 20–50 | 93.6× | 36.4× | 18.7× | 9.8× | · | · |
| 50–100 | 95.2× | 36.5× | 19.7× | 9.3× | 5.5× | · |
| 100–200 | 103.6× | 39.0× | 20.6× | 9.6× | 5.0× | 3.4× |
| 200–400 | 96.5× | 37.1× | 19.7× | 9.1× | 4.8× | **2.4×** |
| 400–1000 | 70.6× | 26.9× | 14.3× | 6.8× | 3.7× | **1.6×** |

Overall: the teacher holds **261 more points on average, 42.6% of the online
segment**, and 73.6% of the metric's weight sits where at least a quarter of the
segment is still unseen. So the teacher is genuinely well informed — Lane A is
not bounded away by information geometry.

**But privilege and weight are almost perfectly anti-correlated.** Where privilege
is transformative (age 0–5: 70–100×) the weight is 2.9%. In the two cells that
own half the loss it is **1.6× and 2.4×** — the teacher sees the same regime for
somewhat longer, which under any likelihood-ratio detector is a √2-ish
improvement in effective evidence, not a new kind of knowledge.

## 7. HOW MUCH LOSS MUST BE CONVERTED TO WIN

| external target | gap from LB-001 0.6268 | internal-equivalent gap from 0.62581 | **share of remaining weighted loss** |
|---|---:|---:|---:|
| 0.630 | +0.0032 | +0.0042 | **1.12%** |
| 0.635 | +0.0082 | +0.0092 | **2.45%** |
| 0.640 | +0.0132 | +0.0142 | **3.79%** |
| 0.645 | +0.0182 | +0.0192 | 5.13% |
| 0.650 | +0.0232 | +0.0242 | 6.46% |

This is the most encouraging table in the document. **Winning territory does not
require solving the problem. It requires converting between one and four percent
of the inversions that remain.** Every ceiling in §2–§5 is an order of magnitude
larger than that. The constraint is not headroom; it is capture rate.

---

## 8. WHAT THE BUDGET CHANGES IN THE WAVE-7 PLAN

Three corrections, each with the number behind it. All were made before any
wave-7 score existed, because none exists.

### 8.1 Lane B's pilot region moves — brief §18

The brief recommends piloting the horizon specialist in the "EARLY/MID regime
where current TS-AUC is weakest". Weakest is not where the loss is: `t < 50` is
**4.7% of the loss with a +0.018 ceiling**. A perfect early-regime specialist
cannot deliver 0.635.

**Pilot H4 (`t ∈ [101, 251)`) instead** — 20.3% of the loss, enough rows for a
stable fit, and the earliest regime with material mass. H5 and H6 (65.8% of the
loss between them) are the follow-ups if H4 clears. Encoded in
`research/scripts/wave7_b_horizon.py`, default `--regime H4`.

### 8.2 Lane A's target ordering reverses — brief §12

The brief makes **permanence** the priority teacher target. Permanence
distinguishes a transient disturbance from a persistent shift, which is a young-age
question: ages 0–20 hold **13.6% of the loss**. And at age 100+, where 51% of the
loss lives, permanence is not in doubt — the break has been visible for a hundred
observations.

**Prioritise the graded oracle-evidence target instead**, with full-sequence
magnitude behind it. Its mechanism applies at every age: the hard label
`y[t] = 1[t ≥ τ]` says a break happened, not that it is *visible*. A weak break at
age 300 and a large break at age 300 carry the same label and are worlds apart
under TS-AUC. Since 92.8% of break series are `weak_unclassified` at AUC 0.6005,
the hard label is systematically overconfident exactly where the budget is.
Permanence stays in the declared bank, ranked last.

### 8.3 The young-break lane is capped, not closed

Wave 6 saw the MLP and TCN help at ages 0–5 and 5–10 and correctly refused to
build an age-gated blend. The budget puts a number on the restraint: **the entire
young-break lane has a perfect-repair ceiling of +0.0509**, and realistic capture
against an architecture-generation yardstick of ~+0.008 per bucket is one to two
thousandths. It is a legitimate +0.002 lane and it is not a route to 0.635.

## 9. THE UNCOMFORTABLE READING

The dominant error class — a persistent break, a hundred-plus observations old,
in a long series, still inverted against a no-break series — is the one where the
model has the **most** data and the **fewest** excuses. The teacher's privilege
there is 1.6–2.4×. The break family is unclassifiable. Nothing in lanes A, B or C
is aimed at it directly.

Two readings are consistent with the evidence, and this wave cannot yet separate
them:

1. **Information limit.** These breaks are genuinely near-indistinguishable in
   the training sample, and 0.66 at age 100+ is close to Bayes. If so, lanes A–C
   compete for the 13.6% young + 35.1% mid-age remainder, and the honest target
   is 0.630–0.635, not 0.640+.
2. **Representation limit.** W6-E2R is the direct evidence against reading 1:
   at the FULL horizon our causal bank beat the oracle frontier's generic bank by
   **+0.02354 series ROC AUC**, 25/25 folds positive, with the gap *widening*
   under matched column width. That study says the representation was
   underpowered, not saturated — measured on precisely the mature, whole-series
   question that the dominant cell asks.

**Reading 2 is better supported, and it points somewhere none of the three lanes
go: at what distinguishes a weak persistent break from noise after 100+
observations.** The single highest-value experiment this budget can name is
therefore neither the teacher nor the specialist bank — it is to take the
W6-E2R series-level protocol, restrict it to the dominant joint cell, and measure
how much of the +0.0235 survives when the boundary is removed and the question is
asked row-wise. That is a diagnostic, not a model, and it is cheap.

## 10. REPRODUCING THIS

```
python research/scripts/wave7_alpha_budget.py        # W7-D1, ~2 s, no store needed
python research/scripts/wave7_teacher_privilege.py   # W7-D2, ~20 s, no store needed
```

Both refuse to emit a budget whose bucket decomposition does not reconstruct the
reported aggregate.
