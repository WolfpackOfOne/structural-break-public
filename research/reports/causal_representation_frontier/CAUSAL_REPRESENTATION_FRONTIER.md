# CAUSAL REPRESENTATION FRONTIER

**Status: AUDIT AND DESIGN. NO EXPERIMENT WAS EXECUTED. NO SCORE WAS PRODUCED.**

Branch `research/causal-representation-frontier-2026`, forked from
`origin/research/new-avenues-pilots-2026` at
**`b47b22ad7e0b85fe977cb65453ae8531227d414d`** ("Evaluate SS-04 specialist router"),
the exhausted Second Sweep tip.

Forbidden-action check: no model was trained, no candidate OOF vector was produced,
no TS-AUC was computed for any candidate, no `RT-xxx` id was allocated,
`research/RESULTS.csv` was not modified (sha256
`5b34c564e69f502c4a54d4ba1b702b400893358073e1897cd82453c83215512c` before and after),
the lockbox was not opened, `X_test.reduced.parquet` was not read, and no production
artifact was touched.

Everything below is verified against repository state, not against handoff prose.
Where a source document and a prior summary disagree, the source wins and the
disagreement is stated.

---

## A. CURRENT EMPIRICAL BOUNDARY

Verified from the repository, not assumed.

| quantity | value | source |
|---|---:|---|
| RT-600 external Crunch leaderboard (LB-001) | **0.6268** | `STATUS.md`, `EXPERIMENT_ID_MAP.md` |
| RT-600 dev mean TS-AUC | **0.625811** | SS-01/02/03/04 sentinels, `wave7_rt600_exact_alpha_budget.json` |
| RT-600 dev pooled TS-AUC | **0.625627** | same |
| RT-600 dev dominant-cell TS-AUC | **0.664277** | W7-D0 (`0.66428` in the report) |
| fold-0 E0 (RT600 7-stream) | 0.638276 | every second-sweep report |
| fold-0 E1 (RT600 + `RT-401` exchangeable clone) | 0.638586 | same |
| ⇒ an inert candidate scores | **−0.000310** marginal_vs_clone | arithmetic; observed exactly by SS-01 and SS-04 |
| dominant cell = current `t ≥ 200` **and** positive age `≥ 100` | **50.50 %** of pair weight, **45.29 %** of inversion loss | W7-D0 |
| never-break share of dominant-cell loss | **73.99 %** | W7-D0 §E |
| total inversion loss fraction | 0.374373 | W7-D0 §C |

All four figures quoted in the task brief reproduce exactly. The one worth
restating precisely: **an exchangeable seed clone is worth +0.000310**, so
`marginal_vs_clone` is measured against a bar that a no-op candidate misses by
exactly that much. Every "−0.00031" in the Wave-8 and Second-Sweep tables is a
candidate that did nothing at all.

**Runtime envelope** (`FINAL_ARCHITECTURE_FREEZE.md` §1–2): shared feature engine
1.069 ms/pt, seven boosters 1.734 ms/pt total, `INFER_PARALLELISM = 1`, projected
~4.9 h against a 15 h budget. Roughly **10 h / ≈3.5 ms per point of unused
inference budget**. That is a real constraint but a generous one for a small
streaming sequence model.

---

## B. WHAT THE FIRST SWEEP FALSIFIED

Twelve scored mechanisms, all KILL (`FIRST_SWEEP_SYNTHESIS.md`,
`first_sweep_matrix.csv`). The important content is not the list of deaths but
the **shape** of them.

Sorted by within-timestep rank correlation with RT-600:

| arm | mechanism | standalone whole AUC | ρ vs RT600 | marginal_vs_clone |
|---|---|---:|---:|---:|
| `RT-1202` | trajectory geometry | 0.4993 | 0.004 | −0.002587 |
| `RT-1218` | direct e-value aggregation | 0.5257 | 0.134 | −0.002214 |
| `RT-1201` | IM2 dwell / run-length null | 0.5836 | **0.382** | **+0.000301** |
| `RT-1200` | relay score-state | 0.6196 | 0.775 | −0.000207 |
| `RT-1212` | scalar difficulty | 0.6291 | 0.863 | +0.000164 |
| `RT-1216` | weighted CTM | 0.6343 | 0.865 | +0.000937 |
| `RT-1208` | ordinal irreversibility | 0.6291 | 0.878 | −0.000036 |
| `RT-1210` | joint rarity | 0.6261 | 0.878 | −0.000522 |
| `RT-1206` | spectral impulse | 0.6276 | 0.881 | −0.000353 |
| `RT-1214` | Kalman / NIS | 0.6308 | 0.884 | +0.000135 |
| `RT-1215` | Hankel-DMD | 0.6313 | 0.886 | +0.000226 |
| `RT-1204` | scale survival | 0.6294 | 0.886 | −0.000236 |

**Signal and redundancy have been perfectly confounded for this project's entire
history.** Over the 17 scored arms for which both numbers exist (twelve first-sweep
candidates, four Wave-6 neural arms, T2), the Pearson correlation between standalone
AUC and within-timestep ρ against the RT600-family blend is **+0.983**. Nothing this
project has ever built has been simultaneously good and different.

The exact size of the empty region, from a descriptive OLS of `marginal_vs_clone`
on (standalone AUC, ρ) over those 17 arms (`R² = 0.61`, n = 17 — an extrapolation
outside the observed hull, quoted as a scale not a prediction):

| ρ vs RT600 | standalone whole-dev AUC needed for +0.0030 |
|---:|---:|
| 0.2 | 0.608 |
| 0.3 | 0.619 |
| 0.4 | 0.630 |
| 0.5 | 0.641 |

Best standalone AUC ever achieved at ρ ≤ 0.60: **0.58358** (`RT-1201`). The gap
to the region that would pay is roughly **+0.035 AUC**, and no mechanism has ever
come close to it. That number is the CRF program's target and its risk statement
in one line.

**The repair reservoir.** The descriptive oracle found ≥1 first-sweep candidate
repairs `14,868 / 16,193 = 91.8 %` of sampled dominant RT600 mistakes — and the
same candidate set damages `28,505 / 34,347 = 83.0 %` of sampled RT600-correct
dominant pairs. That asymmetry is what motivated the Second Sweep.

---

## C. WHAT THE SECOND SWEEP FALSIFIED

Four preregistered mechanisms, all KILL, all measured against the same sentinels.

| arm | mechanism | acted? | marginal_vs_clone | dominant repairs/damage/net |
|---|---|---|---:|---|
| `SS-01` `RT-1219` | repair-damage arbiter over 12 frozen first-sweep sensors | **no** — 0 non-RT600 action rows | −0.000310 | 0 / 0 / 0 |
| `SS-02` `RT-1223` | RT600 residual same-t pair ranker over the 500-bank | yes | −0.000290 | 330 / 438 / **−108** |
| `SS-03` `RT-1225` | negative-side null-state SCDF calibrator | yes | −0.000299 | 659 / 773 / **−114** |
| `SS-04` `RT-1230` | specialist-disagreement micro-router | **no** — 0 dominant repairs | −0.000312 | 0 / 0 / 0 |

**The strongest conclusion the Second Sweep supports is not "these four
mechanisms failed."** It is this:

> Given the frozen 500-column state representation and RT-600's own scores as
> input, a fold-pure learner with an explicit repair-vs-damage objective **cannot
> find a decision rule that separates repair from damage at all**. Two of the four
> arms converged to the identity — they learned that acting is worse than not
> acting — and the two that did act produced net-negative pair flow.

The 91.8 % repair reservoir is therefore **not causally separable in this
representation**. It is an artefact of unfiltered noise: any sufficiently
different score fixes ~92 % of the mistakes and breaks ~83 % of the correct
pairs, and nothing in the state vector tells you which is which. `SS-01`'s
degenerate solution is the cleanest evidence in the repository that the
information required to make that call is **not present in the 500-vector plus
RT600's own score state**.

That is a statement about the representation, not about arbitration. It is the
reason this program does not build `SS-01b`.

---

## D. W7-D3R INFORMATION FRONTIER — AND THE PART THAT IS USUALLY MISREAD

The three arms (`wave7_d3r.md`, `research/scripts/wave7_d3r.py`):

| arm | inputs | dominant-cell AUC |
|---|---|---:|
| A `RT-300` | 500 legal causal columns at row `t` | 0.65341 |
| B `RT-990` | identical columns, 900 trees × 127 leaves, ff 1.0 | 0.64749 (**−0.00592**, beats A on 1/5) |
| C `RT-991` | Arm B **plus the same 500 columns evaluated at that series' FINAL online row** | **0.71859** (+0.07110, 5/5 folds) |

Two readings, and the second is the one that governs this program.

**Reading 1 (correct and standard).** Extraction capacity is not the bottleneck;
future observations resolve what the causal bank cannot see. FUTURE INFORMATION
EXISTS.

**Reading 2 (verified at source level, and materially constrains H-A).** Arm C's
extra 500 columns are **the identical functional form, evaluated on more data**.
It is not a raw-future oracle and it is not a richer representation — it is the
same summary bank read at time `T` instead of time `t`. So D3R does **not** show
that the summary bank's functional form discards the information. It shows the
prefix at `t` is evidence-starved relative to the prefix at `T`.

In the dominant cell (`t ≥ 200`, age `≥ 100`) the prefix is already long. Arm C's
+0.071 there is therefore mostly information arriving **after** `t` — from the
`n_online − t` observations the model will never legally see. That is a direct
argument for H-D and against a naive reading of "Level 1 → Level 4 headroom is
huge, therefore Level 1 → Level 3 headroom must be large too."

Caveat recorded rather than buried: Arm C's broadcast block is 500 *series-constant*
columns, the exact encoding that made `m05_ctx` memorise series identity
(`FAILED_EXPERIMENTS.md:266`). Unlike `m05_ctx`'s history block, Arm C's block is
genuinely label-informative, so the gain is real — but no derangement control was
run on it, and the +0.071 should be read as an upper bound on future information,
not a clean estimate of it.

---

## E. PRIOR NEURAL / SEQUENCE AUDIT

Machine-readable: `prior_representation_audit.csv`.

**Only four torch models have ever been scored in this project** — `objective`
and `model` columns of `RESULTS.csv` over all 245 rows contain exactly
`torch_mlp` ×2 and `torch_tcn_h32` / `torch_tcn_h64` ×1 each. CFEP's `RT-1041` /
`RT-1042` are recorded as `lgbm` because the neural part was a frozen 16-column
addendum. There is **no** transformer, no recurrent model, no state-space model,
and no contrastive model anywhere in the ledger. `RT-980` (GRU) was never opened.

### E.1 W6-N2 TCN — the exact audit (§10 of the brief)

Source: `research/scripts/wave6_n2_tcn.py`,
`research/scripts/wave6_neural_lib.py:172–260`, `reports/wave6_neural_results.md`.

| question | answer |
|---|---|
| **channels consumed** | exactly ten: `z`, `z²`, `|z|`, `sign(z)`, `lag1 = z_t·z_{t−1}`, expanding `run_mean`, `log1p(run_var)`, `EWMA(hl=8)`, `EWMA(hl=64)`, `elapsed = log1p(t)/7` |
| **raw / standardized / null-normalized?** | `z = clip((x − median(H)) / IQR(H), ±8)`. **Location and scale only.** Not a PIT, not a rank/ECDF transform, not an innovation, not a residual, not a null percentile. |
| **history representation** | **two scalars per series** — `median(H)` and `IQR(H)`. Nothing else about the 1,000–5,000-point break-free history reaches the network. |
| **per-series null identity** | **absent.** No conditional null, no history ECDF, no exceedance-percentile channel, no per-series dependence structure. |
| **receptive field** | **127** (kernel 3, dilations 1/2/4/8/16/32, six residual blocks). Online lengths run to 999. Long memory reaches the net only through the three handcrafted expanding/EWMA channels. |
| **architecture size** | 35,905 params (h32) / 139,393 (h64); GELU, weight-normed left-padded causal convs, dropout 0.1, no BatchNorm, no time-axis norm |
| **target** | `y[t] = 1[t ≥ τ]`, per-timestep head |
| **objective** | masked **BCE**, uniform over rows. Fixed by preregistration; **no ranking arm was permitted** |
| **score produced independently per series?** | yes — one causal pass per series, batch composition proven not to move a prediction (Gate 5) |
| **calibration** | none for standalone TS-AUC (a sigmoid is monotone within a timestep, so the metric is invariant); canonical `SCDF_NSEEN` was applied for the ensemble analysis |
| **data volume / row sampling** | **all** online rows of all 6,400 training series per fold, 20 epochs, batch = 32 series — *more* data than `RT-300`'s 1M sampled rows |
| **why 0.5415 / 0.5432** | see below |
| **failure attributable to** | **input representation AND objective, jointly. Not architecture, not optimisation, not data volume.** |

Why that attribution is defensible rather than convenient:

1. **The `elapsed` channel is the pathology, handed over explicitly.** The MLP on
   identical rows drove training BCE to **0.077** while scoring 0.045 *below* a
   tree ensemble. `y[t] = 1[t ≥ τ]` is monotone in `t` within a series, so elapsed
   time is an enormous row-level predictor that TS-AUC deletes entirely by
   comparing only *inside* a timestep. The TCN was given `log1p(t)/7` as channel 10
   under exactly that objective.
2. **The channel set contains no null-referenced quantity at all.** Every winning
   idea in this project is per-series historical-null calibration
   (`STATE_OF_RESEARCH.md`: *"the single most reliable improvement"*), and
   specifically normal-scoring an AR-whitened stream against the historical
   residual ECDF. The TCN received neither. It got the two crudest moments of the
   history and nothing else.
3. **Optimisation and data are ruled out.** 12–21 min/fold, well inside the 3 h
   budget; loss curves recorded per epoch; both widths agree to within 0.002; the
   h64 arm's fold-3 divergence recovered to that arm's second-best fold. Doubling
   capacity moved the score by +0.0017.
4. **The report says so itself.** §L: *"`WAVE6_NEURAL_PREREG.md` deliberately
   fixed BCE as the only loss, so this wave cannot distinguish 'the family is
   wrong' from 'the objective was wrong'."* That ambiguity has never been resolved,
   because the Wave-7 proposal that would have resolved it (`W7-P1`/`W7-P2`) was
   **never preregistered and never executed** — there is no `W7-P` row in
   `RESULTS.csv`.

**Conclusion: "we tried a TCN and neural networks don't work" is not supportable.**
What was tried is a location/scale-normalised raw-channel TCN with an elapsed-time
shortcut channel, trained on the one objective the same document identifies as
metric-misaligned. The one number it does establish beyond doubt is the negative
one: within-timestep ρ of **0.21** with the champion and an ensemble contribution
of **+0.0001** — *maximal diversity, zero value*.

### E.2 W6-N1 MLP

Closes "learner capacity on the identical 500 columns is the lever" — decisively,
in the negative direction (−0.045 standalone). It closes nothing about sequence
representations, because it has none.

### E.3 Pilot 4 — trajectory geometry (§12 of the brief)

What it tested (`PILOT04_PREREG.md` §Exact Algorithm): for `m ∈ {16, 64}`,
z-normalised online windows matched by **Euclidean distance** against ≤2,000
evenly-spaced z-normalised historical windows; a provenance component
`log1p(d_hist) − log1p(d_online)`; a FLOSS-style arc-crossing rate; the four
components collapsed to **one scalar by a fixed z-scored arithmetic mean**.
Zero learned parameters. Result: whole AUC **0.499324**, ρ **0.0039**, marginal
−0.002587, real-minus-shuffled **+0.000455**.

**What it falsifies:** hand-specified nearest-neighbour shape matching, compressed
to a fixed scalar, carries no usable label information here.

**What it does NOT falsify, and must not be read as falsifying:**

- learned predictive state (nothing was learned);
- conditional likelihood or density-ratio evidence (no probability model existed);
- any representation optimised for same-`t` ranking (the scalar was unsupervised);
- "temporal order beyond lag-2 is useless" — Pilot 4 tested *one* order-sensitive
  functional, and `NEW_AVENUES_2026.md` §C.4 separately records that the 500-column
  bank discards temporal order beyond lag-2 products entirely, which is a statement
  about the bank, not evidence that the order is uninformative.

The distinction the program must hold: **nearest-neighbour shape matching ≠
learned predictive state ≠ conditional likelihood ≠ a representation trained for
same-`t` ranking.** Pilot 4 tests only the first.

### E.4 Pilot 3 / IM3 — Kalman and Hankel-DMD

Both are **per-series, frozen, linear, fixed-order** observers whose outputs were
appended as feature blocks to a base bank that already contains `m04_resid`.
`RT-1214` AR(2)-state Kalman with a history-only `q×r` noise grid; `RT-1215`
delay-16 rank-4 Hankel-DMD. Both landed at ρ ≈ 0.88 — squarely in the incumbent
residual-scale direction — and `RT-1215` failed its own preregistered ρ ≤ 0.85
redundancy guard. The report's own wording is correct and load-bearing: *"it does
not falsify all state-space or delay-embedding observer ideas."*

They close: per-series frozen linear second-moment observers as an added block.
They do not close: a **global, amortized, nonlinear, distributional** null model.

### E.5 Pilot 3 / IM2 — the most important negative in the repository

`RT-1201`: dominant-cell AUC **0.610158**, mature-vs-never **0.613459**, ρ
**0.3817** — the good-signal / low-redundancy quadrant this project has been
hunting since Wave 4 — and `marginal_vs_clone = +0.000301` with dominant pair net
**−1151**.

**The missing quadrant was reached, and at cell AUC 0.61 it paid three
ten-thousandths.** Any program that promises ≥ +0.003 by "occupying the missing
quadrant" must explain why it will land materially above 0.61, not merely below
ρ = 0.5. This report treats `RT-1201` as the bar, not as encouragement.

---

## F. WAVE-8 REPRESENTATION-LEARNING AUDIT — CFEP AT SOURCE LEVEL (§11)

Source: `research/scripts/wave8_cfep.py` and `wave8_cfep_classify.py` on
`origin/research/wave8-future-aware-distillation`.

| question | answer |
|---|---|
| **exact sequence input** | the **identical ten Wave-6 channels**, including `elapsed`. `from wave6_n2_tcn import build_channels`. No new channel was introduced. |
| **exact future target** | the 8 cached SST **handcrafted structural channels at `t + 200`**, robust-standardised (median/IQR fitted on outer-train eligible rows), masked to eligible rows, MSE. Not the raw future series, not a predictive distribution — *a future value of the same handcrafted family*. |
| **raw or normalized channels** | location/scale-normalised, as W6 |
| **encoder causal?** | yes — same left-padded dilated convs, RF 127 |
| **representation learned** | 16-dim `Conv1d(hidden→16, k=1)` bottleneck |
| **frozen or fine-tuned?** | **FROZEN.** `--train-embedding`, then `--extract` to disk, then a **separate process** runs LightGBM. The supervised stage can never adjust the encoder. |
| **downstream** | `concat(500 causal columns, 16 embedding dims)` → LightGBM, `objective="binary"` |
| **matched BCE control** | `RT-1041`: same architecture, same bottleneck, same downstream, BCE pretraining. Binding contrast is **C − B**. |
| **result** | C − B = **−0.00374** whole fold-0, **−0.00375** dominant cell; marginal_vs_clone −0.00031 |

**Why the future-predictive objective lost.** Four candidate explanations; the
evidence discriminates.

- *Poorly predictable?* Partly — but two real bugs were found and fixed en route
  (an unstandardised ±6000 target that NaN'd training, and `NaN × mask ≠ 0`), and
  the reported numbers are post-fix, so raw untrainability is not the story.
- *Easy but label-irrelevant?* This is the best-supported reading. The target is a
  **future handcrafted summary** whose *present* value is already one of 500
  columns the downstream model receives. Predicting `SST(t+200)` well is largely
  predicting `SST(t)` plus drift — a task solvable without any break-relevant
  structure.
- *Dominated by nuisance?* Plausible and untested: the 8 SST channels are
  scale-dominated and per-series scale is exactly the nuisance every module
  calibrates away.
- *Simply redundant?* Certain, and structurally so. The embedding had to earn
  16 columns' worth of gain against a 500-column bank that D3R Arm B had just
  shown to be capacity-saturated, using a frozen encoder and a rowwise-BCE tree
  head. **This design cannot express "the representation is the base model."**

**Standing rule this establishes:** do not propose self-supervised future
prediction again unless the new design differs in a load-bearing way from CFEP on
at least **training population**, **target object**, and **downstream role**.
CRF-02 is required to differ on all three (§L).

Other Wave-8 arms — SST, ORR, PCFB, TGMC — are all static-bank or
future-teacher mechanisms with no learned temporal representation; all five
marginals sit at ≈ −0.0003, i.e. inert. ORR is the informative one: its repair
mechanism was **real** (`P(repair correct | RT600 wrong, teacher confident) =
0.737`) and moved the aggregate metric by −0.00005, because *the loss is diffuse*
(`NEW_AVENUES_2026.md` §E.1: the worst 5 % of series carry only 0.16 of cell loss).
**Coverage beats precision** is not a slogan here; it is a measured constraint.

---

## G. PAIRWISE-OBJECTIVE AUDIT (§17, §18)

Every same-`t` ranking experiment in this project's history, without exception,
was **LightGBM on the static 500-column vector**:

| id | variant | result |
|---|---|---|
| `RT-A09-OBJ-pairwise-t-m8-B` | pairwise logistic, groups = `t`, screen | 0.61395 (screen; screen is not evidence for objectives) |
| `RT-A09-OBJ-lambdarank-B` | lambdarank | 0.60731 |
| `RT-A09-OBJ-xendcg-B` | XE-NDCG | 0.59262 |
| `RT-111` | pairwise logistic, **paired**, identical 800k matrix | 0.62302 vs binary 0.62631 (**−0.00330**) |
| `RT-123` / `RT-123R` | pairwise-`t`, 5 folds, 30 % fewer rows | 0.61450 — a dead heat; **a permanent RT-600 stream** |
| `RT-700` | pairs weighted by `n_neg(t)`, the metric's own weighting | −0.00147 vs control `RT-702` |
| `RT-701` | weighted squared-hinge margin | −0.00476 vs control |
| `RT-1223` (SS-02) | pair objective as a bounded **correction on RT600** | −0.000290, dominant net −108 |

Two things follow, and they point in opposite directions.

**Against a new ranking experiment:** the objective has been tested seven ways and
is neutral-to-negative every time. Metric-shaped pair weighting specifically
loses. Any proposal opening with "align the loss with the metric" is proposing
something already falsified once.

**For a new ranking experiment, and this is the load-bearing asymmetry:** trees
never had the pathology that pairwise ranking removes. `min_data_in_leaf = 300`
and depth limits stopped LightGBM exploiting elapsed time far enough to hurt; a
168k-parameter network drove BCE to 0.077 doing exactly that. **For trees, the
ranking loss replaced a working objective and cost a little. For a network, the
binary objective is measurably solving the wrong problem.**

This is the one condition under which retrying a falsified intervention is
legitimate: *the mechanism the intervention targets has been shown present in the
new setting and absent in the old.*

**Explicit collision statement.** A new ranking arm is legitimate **only** as:

> OLD: pairwise objective over a **fixed** 500-dimensional static vector, gradient-boosted trees.
> NEW: pairwise objective **jointly learning a causal sequence representation** from a null-normalised prefix, no static bank, no RT600.

It is **not** legitimate as: the same 500-vector, the same booster, different pair
weights. `RT-700` already is that experiment.

---

## H. REPRESENTATION COLLISION MATRIX

`representation_collision_matrix.csv` — past and proposed mechanisms against
21 representation/objective axes, with `closest_prior` and `collision_severity`.

The matrix makes one fact visually obvious: the columns
`uses_pit`, `uses_innovations`, `conditional_density_model` and
`same_t_pairwise_objective` have **never once been simultaneously true with
`learned_temporal_representation`**. Every learned-sequence row
(`RT-970/971`, `RT-1041/1042`) has `uses_pit = NO`, `uses_innovations = NO`,
`same_t_pairwise_objective = NO`. Every ranking row
(`RT-111/700/701/123`, `RT-1223`) has `learned_temporal_representation = no`.

---

## I. WHAT THE 500-VECTOR DISCARDS

From `NEW_AVENUES_2026.md` §C, restricted to what survives the first and second
sweeps as still-plausible:

1. **Temporal order beyond lag-2 products.** Every column is a marginal or
   low-order-moment functional of the prefix. §C.4 states it flatly. Pilot 4
   attacked this with one handcrafted scalar and found nothing; nothing has
   attacked it with a learned representation.
2. **Contiguity and excursion-growth.** No column computes the length of the
   *current* contiguous excursion, its running maximum, the episode count, or the
   inter-episode gap. D4 measured `res64_maxrun90` at **cell AUC 0.60791, ρ 0.367**,
   and D5 identified the actual state variable: *"is the longest excursion growing
   faster than log t, for this series' own dependence structure"* — under
   stationarity the longest run grows like `log t`; under a persistent regime
   change it grows like `t`. `RT-1201` priced this as a nine-feature scalar and
   got +0.0003.
3. **Path-functional nulls.** `NullCal` calibrates rolling means. There is no
   per-series empirical null for any *path* functional — longest run, excursion
   area, first-passage time. D6 showed the naive Erdős–Rényi normalisation is
   *worse* than the raw longest run (0.60028 vs 0.60791) because a single
   historical order statistic is far too noisy a scale estimate.
4. **A learned conditional null.** Everything is calibrated against the
   distribution of rolling means over historical windows, with AR(2) shared
   context. Nothing models `P(x_t | H, x_<t)` as a distribution.

Items 1, 2 and 4 are exactly what a null-normalised causal sequence encoder is
positioned to represent, and item 3 is why a *learned* run functional should beat
a hand-specified one.

---

## J. WHAT TEMPORAL INFORMATION REMAINS PLAUSIBLY AVAILABLE

The four-level hierarchy the brief asks for (§24), instantiated against measured
numbers:

| level | object | measured |
|---|---|---|
| **L0** | raw current observation only | `RT-000` handcrafted noisy-OR 0.52051 |
| **L1** | handcrafted static causal summary bank (500 cols) | `RT-300` 0.61605 / RT-600 0.625811 / dominant cell 0.664277 |
| **L2** | frozen history-normalised raw sequence | `RT-970/971` 0.5415 / 0.5432 — **but with location/scale-only channels, an elapsed shortcut and BCE** |
| **L3** | learned causal prefix representation, null-normalised, metric-aligned | **NEVER MEASURED** |
| **L4** | own-series full-sequence oracle | `RT-991` dominant cell 0.71859 |

W7-D3R measures **L1 → L4 = +0.071** in the dominant cell. It does *not* measure
L1 → L3, and — per §D — a large share of L1 → L4 is post-`t` information that no
causal model can reach. **The open quantity is L1 → L3, and this program exists to
measure it.**

L2 as measured is not a fair estimate of L2: the one L2 experiment was crippled on
both the input and the objective side. A properly-built L2/L3 arm is the single
missing cell of the ablation in §M.

---

## K. NULL-NORMALIZED SEQUENCE HYPOTHESIS (§15)

The paradigm:

```
guaranteed break-free history H_i
    → fit a per-series null (deterministic, history-only)
    → transform the online prefix into a null-normalised sequence
    → causal learned representation
    → one scalar score per t
```

The point is to **preserve temporal information while removing per-series
nuisance**. The project's entire empirical record says the second half of that is
the thing that matters (`STATE_OF_RESEARCH.md`: per-series historical-null
calibration at matched window length is "the single most reliable improvement");
W6-N2 did the first half and skipped the second.

Selected channel set for CRF-01 — deliberately **eight**, not five hundred. All
fitted on `H_i` alone, so all are bitwise prefix-invariant by construction:

| # | channel | rationale, sourced |
|---|---|---|
| 1 | `pit` — `Φ⁻¹(F̂_H(x_t))` clipped ±4 | distributional normalisation, not location/scale. W6 had none. |
| 2 | `inn` — AR(5) predictive residual on history-fitted coefficients, ÷ historical residual sd | `STATE_OF_RESEARCH.md`: AR(6)-residual log-sd ratio is *the strongest single statistic in the project* (0.603), and *"AR order matters in the opposite direction to the public folklore"*. Shared context is AR(2); 5 is the documented better order. |
| 3 | `inn_pit` — normal score of `inn` against the **historical residual ECDF** | listed verbatim as EXPERIMENT THAT WON #7. |
| 4 | `abs_inn_pit` | magnitude surprise, the scale channel the taxonomy says dominates |
| 5 | `vol_norm` — `inn` ÷ causal EWMA(hl=32) of `|inn|`, floored at historical scale | heteroskedasticity normalisation without GARCH's adapt-to-the-break failure |
| 6 | `surp` — `−log(1 − Ĝ_H(|inn|))` clipped | per-point conditional p-value in log space |
| 7 | `exceed` — soft indicator `|inn|` above the historical 90th percentile | **the contiguity substrate.** D4/D5 say the missing state variable is excursion-growth-rate; giving the network the exceedance indicator lets it *learn* the run functional rather than have it hand-specified — which is precisely where `RT-1201` and D6 failed. |
| 8 | `lag1_pit` — product of consecutive normal scores | dependence, the second-ranked break family (post-vs-pre AUC 0.538) |

`elapsed` is **removed**. Under a same-`t` objective it carries zero gradient by
construction; under the matched BCE control it is the documented pathology, so it
is removed from both arms to keep them matched.

Explicitly excluded: raw `x`, the 500 columns, RT600, any specialist score, true
`τ`, `n_online`, any cross-sectional quantity.

---

## L. CONDITIONAL GENERATIVE-NULL HYPOTHESIS (§19)

Learn `P(x_t | h_i, x_<t)` where `h_i` is a bottlenecked embedding of the
break-free history, with a model materially more expressive than AR(2), Kalman,
DMD, GARCH or a fixed historical window.

**Eligibility check against the audit — all four must hold, and do:**

| prior | why CRF-02 is materially distinct |
|---|---|
| `m04_resid` | scalar AR/volatility residual *summaries*; no predictive distribution |
| `RT-1214` Kalman/NIS | per-series, **frozen**, **linear**, fixed order 2, Gaussian, mean+variance only. CRF-02 is global, amortized, nonlinear, and emits full predictive quantiles. |
| `RT-1215` Hankel-DMD | per-series, frozen, linear delay-embedded operator; failed its own ρ ≤ 0.85 guard |
| `m07_bayes` / BOCPD | a *posterior over change*, with a memoryless geometric hazard; not a predictive density for `x_t` |
| Wave-8 future-state methods | all target the **future**; CRF-02 targets the **present conditional under the null** and never sees a future row |
| `RT-1042` CFEP | differs on all three required axes: training population (break-free histories only, vs all online segments), target object (predictive density, vs a future handcrafted summary), downstream role (base model, vs 16 frozen columns bolted onto the saturated bank) |

**Head choice: a 21-knot monotone quantile head under pinball loss**, not a
mixture-density network. Reason: MDNs are unstable on CPU without tuning, and this
program forbids tuning after a score; a quantile head yields the PIT directly by
interpolation, which is the object the downstream statistics need.

**Inference signals:** predictive log score, predictive PIT `u_t`, cumulative
PIT-uniformity discrepancy, cumulative predictive-surprise, and the learned latent
state's own shift. Never a future value.

**The `m05_ctx` constraint is binding on `h_i`.** The 2026 generator assigns
breaks **independently of the historical DGP** — three independent tests agree
(series-level AUC 0.5068 against a permuted null band of 0.5063–0.5163; break rate
flat across six DGP clusters). So `h_i` has **no substrate as a break prior** and
may enter **only** as a conditioner of the null. It is therefore constrained to an
**8-dimensional bottleneck**, and the arm carries a **within-fold derangement
control** as a mandatory gate — the control that caught `m05_ctx` and that
`FAILED_EXPERIMENTS.md` explicitly instructs the next agent to run.

---

## M. METRIC-ALIGNED OBJECTIVE HYPOTHESIS — THE ABLATION (§32)

Filled from prior experiments wherever the repository already answers the cell.
**Only the shaded cells need to be executed.**

| representation | rowwise BCE | same-`t` RANK |
|---|---|---|
| **static 500-vector, trees** | KNOWN: `RT-300` 0.61605 (the incumbent) | KNOWN NEGATIVE: `RT-111` −0.00330 paired; `RT-700` −0.00147; `RT-701` −0.00476; `RT-123` a dead heat |
| **static 500-vector, MLP** | KNOWN NEGATIVE: `RT-960/961` −0.045 / −0.035 | never run — *and not worth running*: no temporal representation is being learned, so it only re-tests the objective on a saturated vector |
| **learned sequence, location/scale channels + `elapsed`** | KNOWN NEGATIVE: `RT-970/971` 0.5415 / 0.5432, marginal +0.0001 / +0.0002 | **EMPTY** |
| **learned sequence, null-normalised channels** | **EMPTY → CRF-01 control** | **EMPTY → CRF-01 candidate** |

Three empty cells; CRF-01 fills two of them with one run pair, and the third
(`sequence + weak channels + rank`) is deliberately left empty because it is
dominated — nobody should spend compute proving that bad channels plus a good
objective still fail.

**This is why CRF-01 is the decisive stop experiment.** After it, the
representation × objective factorial is complete.

---

## N. CANDIDATE ARCHITECTURES (§20) — AND WHY NOT A TRANSFORMER

Selection criteria, applied honestly: sequence length ≤ 999 online; CPU-only
deterministic execution at ≤ 1e-8; **constant** streaming state per series;
per-series independence; 10k-series throughput; training stability without a
sweep; one score emitted per `t`.

| architecture | streaming state | verdict |
|---|---|---|
| **small dilated causal TCN** | ring buffer of RF samples × C channels | **SELECTED** |
| GRU / LSTM | O(1) hidden state — the best streaming story | strong, but BPTT over 999 steps on CPU is slower and less stable, and no causality-gate infrastructure exists for it here |
| compact causal Transformer | KV cache **grows with `t`**; O(t²) or O(t) memory per point | **REJECTED** — violates the constant-state requirement; `WAVE6_PREREG.md` §410–411 already priced it as infeasible at 999 points × ~2,000 series |
| structured state-space (S4/Mamba-style) | constant state, unbounded horizon — theoretically the best fit | **REJECTED FOR NOW**: new dependency, determinism unverified against Gate 5, no implementation in-repo. Named as the follow-up if CRF-01's receptive field binds. |
| reservoir / random-feature state | cheap, constant | **retained as a control**, not as a candidate: a frozen random encoder is the natural "is the representation learned or just wide?" ablation |

**The TCN is selected for a reason beyond fashion, and beyond it being what was
tried before.** `tests/test_neural_causality.py` already gates this exact shell on
five properties — prefix invariance ≤1e-8, no `n_online` leakage, no true-`τ`
leakage under the official scorer, fold purity of standardiser constants, and
determinism *including* batch-composition invariance. Runtime is measured
(12 min/fold at h32). Reusing it holds architecture **exactly constant** against
`RT-970`, so that the only things that change are the input representation and the
objective — which is the entire point of the ablation in §M. Changing the
architecture at the same time would make the result uninterpretable, which is the
rule `WAVE6_PREREG.md` §7 already binds.

**Frozen architecture for CRF-01, identical to `RT-970` except channel count:**
hidden 32, kernel 3, dilations 1/2/4/8/16/32, six residual blocks, GELU, weight
norm, dropout 0.1, AdamW lr 3e-3 wd 1e-2, cosine schedule, batch 32 series,
20 epochs, grad clip 1.0, CPU, 6 threads, seed 0. Receptive field 127.
Approximately 35.7k parameters at 8 input channels.

No width sweep. No learning-rate sweep. No Optuna. No architecture search.

---

## O. CAUSAL / STREAMING CONSTRAINTS (§21)

Every proposed mechanism must implement the existing
`research/scripts/novel_streams/harness.py::StreamingMechanism` interface —
`fit_history(H)`, `initialize_online(state)`, `update(state, x_t)`,
`current_features(state)` — which already exists on this branch and is regression-tested
by `tests/test_novel_streams_harness.py`.

| mechanism | `fit_history(H)` | state per series | ops / timestep | training cost | inference cost |
|---|---|---|---|---|---|
| **CRF-01** | 256-knot history ECDF; AR(5) coefficients; residual-ECDF knots; 90th-percentile threshold; EWMA seed | 8 channels × 127-sample ring buffer (≈ 4 KB) + ~1.5 KB scalars ≈ **6 KB** | 8 channel updates + one incremental TCN pass over the buffer; ≈ 35.7k MACs upper bound, far less with cached dilated activations | ≈ 25 min/fold measured-analogue (`RT-970`: 12 min/fold at 10 channels) | dominated by the TCN pass; budget headroom is ≈ 3.5 ms/pt against a model of this size, comfortably inside |
| **CRF-02** | same, plus one forward pass over `H` to produce `h_i` (8 dims), done **once** at `t = 0` | as CRF-01 + 8 floats + quantile-head accumulators ≈ **7 KB** | one quantile-head evaluation (21 knots) + PIT interpolation + accumulator updates | pretraining ≈ 30M history points, small model, few epochs | one extra small head per point |
| **CRF-03** | union of the above | ≈ 8 KB | union | conditional | conditional |

Rejected on runtime grounds without being run: anything requiring attention over
all 5,000 historical points plus all online points at every update. The history is
consumed **once**, at `t = 0`, into fixed-size sufficient statistics — which is
also what makes the history-derived normalisation trivially prefix-invariant.

---

## P. COMPUTE FEASIBILITY (§38, §39)

**Measured local environment:** Apple M2 Pro, 10 cores, 16 GB RAM, macOS/arm64
(Darwin 25.5.0). Python 3.11.6. torch 2.13.0 present in the
`structural-break-wave8` venv; lightgbm 4.7.0; numpy 2.4.6. MPS is available and
**deliberately unused** — `torch.use_deterministic_algorithms` does not cover that
backend and the causality gates require ≤ 1e-8 run-to-run. CPU threads fixed at 6.

**The OpenMP collision is real and already solved in-repo.** torch ships
`torch/lib/libomp.dylib`, sklearn ships its own, LightGBM brings a third; loading
torch and LightGBM into one process **segfaults on macOS/arm64**, dying inside
`lightgbm/basic.py`. The Wave-6 fingerprint run once recorded "0 known failures"
because pytest had *crashed, not passed*. The fix in use is **two processes**, not
`KMP_DUPLICATE_LIB_OK` (which trades a loud crash for silent numerical corruption
in exactly the two libraries this project compares bitwise). CFEP follows the same
pattern: torch training and extraction in one process, LightGBM in a separate
subprocess.

**Every CRF execution plan must respect this split**, and CRF-01/02's primary arms
avoid it entirely by not using LightGBM at all in the candidate path — the
candidate is a torch model emitting an OOF vector; only ensemble integration
(which reads frozen `.npy` arrays and calls `wave8_common.ensemble_marginal`)
touches the LightGBM-adjacent stack, and it does so in its own process.

**Budget: ≤ 15 h screening before any serious confirmation.**

| stage | estimate |
|---|---:|
| CRF-01 channel build over the full store | ≈ 0.8 h (Pilot-3 observers took 2,899 s for a comparable per-series build) |
| CRF-01 fold-0 candidate + matched BCE control | ≈ 1.0 h |
| CRF-01 five-fold, if the fold-0 pre-screen clears | ≈ 2.5 h |
| CRF-02 null pretraining on break-free histories | ≈ 3.0 h |
| CRF-02 online scoring + fixed-null control + derangement control | ≈ 2.0 h |
| CRF-03 (conditional) | ≈ 4.0 h |
| **total** | **≈ 13.3 h** |

---

## Q. TOP 3 PROGRAM EXPERIMENTS

Full preregistration: `CRF_PROGRAM_PREREG.md`. Priority table:
`crf_candidate_priority.csv`.

### CRF-01 · NNCSR — Null-Normalized Causal Sequence Ranker *(primary, unconditional)*

Eight history-fitted null-normalised channels → the frozen `RT-970` TCN shell →
one scalar per `t`, trained with a **same-`t` pairwise logistic** objective.
No RT600, no 500 columns, no future teacher, no `τ`, no `n_online`.
Matched control: identical architecture, identical channels, **BCE**.

Closest priors: `RT-970` (architecture), `RT-111`/`RT-700` (objective).
Load-bearing difference: it changes **both** axes at once relative to each prior,
and it is the only unfilled cell of §M's factorial.

### CRF-02 · ACGN — Amortized Conditional Generative Null *(unconditional, independent)*

A global quantile-head sequence model of `P(x_t | h_i, x_<t)` trained **only on
guaranteed break-free histories**, with `h_i` an 8-dim bottleneck of the history
and a mandatory derangement control. Online, the model emits predictive PIT and
log-score sequences; a small ranking head over that state produces the score.
Controls: (a) the matched **fixed** null (AR(5) + historical residual ECDF —
i.e. CRF-01's channels 2–3 with identical downstream statistics), isolating
"learned conditional null" from "fixed null"; (b) the within-fold derangement of
`h_i`, isolating conditioning from series memorisation.

Why it is independent rather than an ablation of CRF-01: it attacks a different
population by construction. **73.99 % of dominant-cell loss is never-break
negatives**, and D3 shows their loss rate is predicted by heavy tails, long
memory and *calm* histories that later wander — the signature of null
misspecification, not of missed break evidence. CRF-01 improves what the model
sees; CRF-02 improves what "normal" means.

### CRF-03 · NNCSR-G *(conditional — opens only on an interim bar)*

CRF-02's learned PIT/log-score channels plus FiLM conditioning on `h_i`, fed to
CRF-01's encoder, trained end-to-end under the same-`t` objective.

**Stated plainly: CRF-03 is not independent.** It is the integration of the two,
and the brief's §25 prefers three only when all three are independent. It is
carried as slot 3 because if either primary clears its interim bar the
integration is the obvious highest-value follow-up, and because reserving the slot
now prevents an unregistered follow-up later. **If compute is constrained, run
CRF-01 and CRF-02 and stop.**

---

## R. LANES NOW CLOSED

Closed unless a separate preregistration reopens them with a materially different
target. Each carries the evidence that closes it.

| lane | closed by |
|---|---|
| more LightGBM capacity on the same 500 features | W7-D3R Arm B: −0.00592 cell, 1/5 folds |
| more seed bagging / an eighth exchangeable member | W5-NULLTEST +0.00003; RT-401 +0.000310 |
| generic stacking, weighting, subset selection | LOFO subset −0.0008, logistic stack −0.0001, LGBM stack −0.0038 |
| generic RT600 residual routers / correctors | SS-02 −0.000290 (net −108) |
| specialist routing / disagreement micro-routing | SS-04 −0.000312, 0 dominant repairs; `RT-1232` static selector worse still |
| repair-vs-damage arbitration over first-sweep sensors | SS-01: learned not to act; reservoir retention 0.0000 |
| static DGP gates / fingerprint selectors | `RT-150` −0.0219 at full scale; `m05_ctx` −0.0177; Pilot 9 deranged control beat the real scalar |
| fixed null-state calibrators over RT600 | SS-03 −0.000299, prebreak damage 0.0199 > 0.0150 cap |
| CTM tuning / `RT-1216` variants | closed explicitly by `FIRST_SWEEP_SYNTHESIS.md`; SS-03's `RT-1228` unweighted control matched it |
| simple anomaly-feature additions without arbitration | eleven first-sweep arms |
| dwell / scale / spectral / ordinal / trajectory feature families **as handcrafted scalars** | `RT-1201`/`1204`/`1206`/`1208`/`1202` |
| basic Kalman / DMD observer blocks | `RT-1214` +0.000135; `RT-1215` +0.000226 and ρ-guard failure |
| **raw / light-input small neural classifiers under BCE** | `RT-960/961/970/971` |
| future-distillation variants on the same causal student substrate | Wave 8 ×5 all ≈ −0.0003; T2 +0.00024 |
| self-supervised **future-summary** pretraining with a frozen embedding appended to the bank | CFEP C−B = −0.00374 |
| learner-capacity substitution on identical columns | W6-N1 −0.045 |

**Nearby but explicitly still OPEN, stated precisely so the boundary cannot blur:**

- a learned causal sequence representation over **null-normalised** channels
  (PIT / innovation / exceedance), which has never been built;
- a same-`t` ranking objective applied to a **jointly learned** representation,
  which has never been run in any wave;
- a **global, amortized, nonlinear, distributional** conditional null, as opposed
  to per-series frozen linear observers;
- a **bottlenecked** history embedding used as a *null conditioner* under a
  derangement control, as opposed to 50 raw series-constant columns;
- structured state-space architectures, if and only if the TCN's receptive field
  is shown to bind.

---

## S. STRONG STOP CONDITIONS (§30)

**The designated stop experiment is CRF-01.**

> If a causal sequence model reading a properly null-normalised prefix, trained
> under a metric-aligned same-`t` objective, with a matched BCE control and no
> access to RT600 or the 500-column bank, cannot beat an exchangeable RT600 clone
> by even **+0.0015**, then:
>
> 1. all four cells of the representation × objective factorial (§M) are filled
>    and none produces marginal ensemble alpha;
> 2. preserving more prefix temporal detail is **insufficient**, and the burden of
>    proof shifts decisively onto anyone proposing another representation;
> 3. **the learned-causal-prefix-representation lane closes**, together with
>    "objective mismatch" as a live explanation;
> 4. belief in **H-D** — that the practical limit is the legal prefix itself and
>    the W7-D3R gap is predominantly post-`t` information — rises to the point
>    where continued model research is not the best use of the remaining budget,
>    and the project should turn to deployment robustness, streaming parity, and
>    protecting the 0.6268 anchor.

Secondary stop, applicable to CRF-02: if the learned conditional null does not
beat its **fixed** AR(5)+ECDF null control by +0.0005 on marginal_vs_clone, then
learned generative nulls are closed for this problem and item 4 above is
strengthened; the per-series calibration the project already ships is the right
null.

Cheap abandon gate, preregistered here and binding **before** any five-fold spend:
a candidate whose fold-0 standalone whole-fold TS-AUC is **< 0.600 at ρ ≤ 0.60**
is abandoned without further folds. Justification: the descriptive frontier in §B
places the +0.0030 contour at standalone ≈ 0.608–0.641 across that ρ range, and
`RT-1201` demonstrated empirically that 0.584 at ρ = 0.38 buys +0.0003. This gate
is a *necessary*-condition filter set deliberately below the regression contour;
it is not a promotion criterion.

---

## T. RECOMMENDED FIRST EXECUTION

**CRF-01 · NNCSR, fold-0 pilot with its matched BCE control.**

It maximises scientific information × plausible alpha ÷ compute:

- it is the **only** experiment that separates representation failure from
  objective failure, because it runs both cells of §M's bottom row in one pass;
- it is the last empty cell of a factorial the project has already paid for three
  quarters of;
- it is RT600-independent by construction, so it can create a genuinely new
  prediction direction rather than another correction;
- it costs ≈ 1.8 h including the channel build, against a 15 h program budget;
- and it is capable of closing an entire research lane if it fails.

**Do not execute it under this task.** It requires its own execution
preregistration commit (per `AGENTS.md`'s preregistration-before-score rule) and
an `RT-xxx` allocation from `EXPERIMENT_ID_MAP.md`, neither of which this
design task is authorised to create.

---

## HONEST STATEMENT OF PRIOR

This program is more likely to close a lane than to find alpha.

The evidence for that: D3R Arm C's gain is mostly post-`t` information; the loss
is diffuse (worst 5 % of series carry 0.16 of cell loss, so no targeted repair can
work); 92.4 % of break series are individually indistinguishable from a placebo
split; the good-signal/low-redundancy quadrant was already reached by `RT-1201`
and paid +0.0003; and seventeen consecutive scored candidates across four waves
have landed within ±0.001 of an exchangeable seed clone.

The evidence that it is nonetheless worth running: the one cell of the
representation × objective factorial that has never been filled is exactly the
cell where the two strongest mechanistic arguments in the repository — that the
500-vector discards temporal order beyond lag-2, and that rowwise BCE lets a
network optimise a channel the metric deletes — would both have to be wrong for
the answer to be null. Measuring that is worth ~2 hours. Not measuring it and
continuing to guess is not.
