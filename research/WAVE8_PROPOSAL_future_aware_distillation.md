# WAVE 8 — PROPOSAL: FUTURE-AWARE DISTILLATION ROUTES

**STATUS: PROPOSAL. NOT PRE-REGISTERED. NOT AUTHORISED. NOT EXECUTED.**
Written 2026-08-23 on `research/wave8-future-aware-distillation`, branched from
`research/wave7-teacher-distillation` @ `399ea6f`. No Wave-8 `RT-` number exists,
and none may be produced until one of the routes below is converted into a
signed pre-registration on its own commit, per the discipline this project has
followed since wave 5 (`WAVE5_PREREG.md`, `WAVE6_PREREG.md`, `WAVE6_NEURAL_PREREG.md`,
`WAVE7_D3R_PREREG.md`, `WAVE7_TEACHER_NESTED_PREREG.md`).

Parent evidence: `research/WAVE7_RT600_EXACT_ALPHA_BUDGET.md` (W7-D0),
`research/reports/wave7_d3r.md` (W7-D3R), `research/WAVE7_TEACHER_NESTED_PREREG.md`
and `research/reports/wave7_teacher_nested_outer{0,1,2}_t{1,2}.json` (W7 nested
teacher, in progress), `research/LEADERBOARD_ASSAULT_STATUS.md`.

---

## 0. THE HONEST OBJECTION, FIRST

**This proposal exists to spend compute after Wave 7's teacher run finishes, not
instead of it.** Three things need saying before any of the five routes below is
taken seriously.

**First, Wave 7 is not done, and its result is not a promotion.** The nested
`T2` checkpoint (`399ea6f`) shows outer folds 0–2 of 5:

| fold | `T1 − T0` | `T2 − T0` |
|---|---:|---:|
| 0 | −0.00725 | +0.00447 |
| 1 | +0.01175 | +0.01391 |
| 2 | −0.00511 | +0.00905 |

`T2` (hard label + teacher blend) is positive on 3/3 folds so far, provisional
mean **+0.00914**. `T1` (pure distillation) is inconsistent once outer-fold-pure
— sign-flips twice — which is itself informative: the contaminated one-fold
pilot (`RT-992`/`RT-993`, since marked contaminated) reported a misleadingly
strong +0.016/+0.018 that the honest nested design does not reproduce for `T1`.
Folds 3–4 are outstanding, and no bootstrap CI, alternate-partition check, or
promotion-battery pass (`RDOF_LEDGER.md`'s bar: ≥+0.0030 mean over the strongest
matched control, ≥4/5 folds positive, paired bootstrap CI supportive, stable
across alt partitions) has run. **Nothing below should be read as "the teacher
lane is confirmed and here is what comes next."** It should be read as "if `T2`
clears the promotion battery, these are the highest-value follow-on
mechanisms — and if it doesn't, most of them are moot, because they all lean on
the same underlying premise `T2` is currently testing."

**Second, this project has already falsified the naive version of "get the
future to help."** Wave 5's pairwise-ranking rows (`RT-111`, `RT-700`, `RT-701`)
showed that aligning the training objective with the metric's own pairwise
structure makes LightGBM *worse*, not better — see
`WAVE7_PROPOSAL_metric_aligned_transition.md` §0 for the table. Any route below
that reduces to "add a same-`t` ranking loss" inherits that prior and must beat
it, not merely restate it. Route 3 (Oracle Repair Ranker) is the one most
exposed to this — it is explicitly a pairwise mechanism — so it should be judged
against the wave-5 pairwise-ranking failure as its baseline expectation, not
against zero.

**Third, `Q` is a single reused teacher (`RT-991`), and this project has one
data point on its reliability, not a validated distillation pipeline.** `RT-991`
is offline-only, sees each series' own final-row features broadcast to every row
of that series, and has never been checked for whether the parts of its signal
that transfer to a causal student are the parts that are actually true (as
opposed to parts that are artifacts of its own construction, e.g. sensitivity to
final online length distribution). None of the routes below should assume `Q`,
or the mechanism that produced it, is beyond scrutiny.

With that stated, the dominant finding W7-D3R actually established is durable
regardless of how the nested run finishes: **the loss cell that matters most
(`t≥200`, positive age≥100 — 50.50% of pair weight, 45.29% of exact remaining
loss, `WAVE7_RT600_EXACT_ALPHA_BUDGET.md` §E) is information-limited, not
extraction-limited** — more tree capacity on the existing 500-column causal bank
made it *worse* (Arm B vs A, −0.00592 cell AUC), while the offline oracle that
sees each series' own eventual trajectory closed **+0.07110 cell AUC, 5/5 folds,
every fold ≥+0.053** (Arm C vs B). That gap is real and it is large. The
question this wave asks is which of several ways to chase it are worth the
compute, and in what order.

---

## 1. FIVE ROUTES, RANKED BY EXPECTED MARGINAL VALUE AFTER `T2`

Ranked by estimated marginal causal contribution **on top of `T2`**, not
standalone novelty — a mechanism largely redundant with `T2` is worth less than
one that repairs different pairs, even if its solo number looks better. Ranges
are this proposal's priors, not measured results; nothing here has run.

| rank | mechanism | core lever | prior Δ pooled TS-AUC (realistic) | prior Δ (upside) | overlap with `T2` | cost | confidence |
|---|---|---|---:|---:|---|---|---|
| 1 | **Structural Successor Targets (SST)** | Forecast a small vector of future structural statistics (location/scale/dependence displacement) at horizons +50/+100/+200, causally, from the current 500-column state | +0.003 to +0.006 | +0.007 to +0.010 | moderate | low | medium-high |
| 2 | **Predictability-Constrained Bottleneck (PCFB)** | Distill Arm-C's 500-column augmented state into an 8–32-dim subspace selected for *predictability from the prefix*, not just oracle information (PLS/reduced-rank, not PCA) | +0.004 to +0.007 | +0.008 to +0.012 | high | moderate | medium |
| 3 | **Oracle Repair Ranker (ORR)** | Use `RT-991` to find the specific same-`t` pairs where `RT-600` is wrong and the teacher is confident, train a residual correction on only those pairs | +0.002 to +0.004 | +0.006 to +0.009 | low (relational, not pointwise) | low | medium, penalised by §0's wave-5 prior |
| 4 | **Causal Future-Embedding Pretraining (CFEP)** | Small causal dilated-conv encoder trained to predict a latent representation of future structural windows (JEPA/CPC-style), fed into LightGBM alongside the 500 columns | +0.002 to +0.004 | +0.006 to +0.010 | low-moderate | high | low-medium |
| 5 | **Teacher-Guided Matched Counterfactuals (TGMC)** | Bootstrap paired stable/broken continuations from real break-free histories, keep only pairs where `RT-991` separates confidently and `RT-600` still fails, train on the union | +0.001 to +0.003 | +0.004 to +0.007 | low (changes training support) | moderate | low-medium

Do not sum this column. 1 and 2 target the same information and will overlap
substantially if both work; 2 is partially redundant with `T2` itself, since
`T2`'s label already *is* a scalar compression of the same Arm-C state PCFB
would decompose. 3, 4 and 5 are the more independent bets, and are ranked lower
mainly because each carries a specific, already-known risk in this project (3:
pairwise objectives on trees; 4: this project's one prior neural attempt failed
outright, see below; 5: synthetic/real domain mismatch, never tested here).

**Route 4 carries a project-specific warning the original brief this proposal
is adapted from did not have access to.** Wave 6 already ran a neural track on
this exact data (`WAVE6_NEURAL_PREREG.md`) and closed it as a clean failure: an
MLP drove training BCE to 0.077 (near-perfect row-level separation) while
scoring **0.045 below** the tree ensemble on TS-AUC, because the row-level label
is monotone in `t` within a series and elapsed time is a powerful row-level
predictor that the metric specifically deletes by comparing only within a
timestep. A TCN with masked BCE did no better (0.541–0.543 vs the champion's
0.626). CFEP's causal dilated-conv encoder is architecturally similar to the
Wave-6 TCN. If it is pursued, it must be evaluated exactly the way `WAVE7_
PROPOSAL_metric_aligned_transition.md` argues neural nets must be evaluated
here — on the metric, per fold, against a matched control — and must not be
credited for row-level or training-loss improvement, which Wave 6 already
proved is not informative here.

---

## 2. THE THREE EXPERIMENTS WORTH PRE-REGISTERING NEXT (IN ORDER)

All three are gated on `T2` finishing its 5-fold nested run and the promotion
battery being applied to it. If `T2` fails the battery, re-read §0 before
running any of these — the premise weakens for all of them, most severely for
1 and 2.

### 2.1 SST pilot — cheapest, highest information-per-compute

**Hypothesis.** Some of Arm C's +0.07110 cell-AUC advantage is not "new
information that only arrives after `t`" but future structural consequences
that are already statistically forecastable from the current 500-column causal
state — i.e. the prefix contains anticipatory structure the existing 500→1
classifier never gets to use because it is never asked to predict anything
about the future.

**Design, one canonical fold (fold 0), reusing existing infrastructure.**
Construct roughly 8 structural target channels per horizon `h ∈ {50, 100, 200}`
(subset of: robust location displacement, log variance ratio, dependence-change
statistic, cumulative evidence delta, no-break-compatibility statistic — all
derivable from the already-cached `cache/features/m0*.npy` memmaps, no new raw
data). For rows where `t + h` exceeds the series' online length, the target is
undefined and the row is excluded from that horizon's *target* construction —
never masked-and-imputed, since an availability mask is exactly the τ-leakage
pattern `PROTOCOL.md` §0 already forbids for a different reason (a mask that
depends on whether break age matters is legal here only because eligibility is
gated on `t + h` vs. `n_online`, which is independent of `y`; this must be
checked empirically before training, not assumed).

Three arms, matched rows/capacity to `RT-300`:

```
A  control:  500 causal columns alone
B  oracle:   500 + true future structural targets (offline diagnostic only)
C  legal:    500 + cross-fitted OOF-predicted structural targets
```

Report `R = ΔAUC(C) / ΔAUC(B)` — the fraction of the oracle target's value that
survives becoming a causal, predicted feature. This is the single most useful
number this pilot can produce regardless of which way it comes out.

**Cross-fitting.** Same nested structure `WAVE7_TEACHER_NESTED_PREREG.md`
established for `Q`: outer fold `f`, inner folds within `outer_train`, OOF
target predictions used for the final classifier's training rows, no arm ever
sees a value derived from its own outer validation fold.

**Continuation gate (one fold).** Continue to the remaining 4 folds only if
legal arm C clears **≥+0.003 fold-0 TS-AUC** over control with visible movement
concentrated in the dominant cell, **or** `RT-600 + SST` beats `RT-600 +
seed-clone` by **≥+0.0015**. Kill otherwise.

### 2.2 PCFB bake-off — the sharpest test of the core thesis

**Hypothesis.** `T2`'s scalar teacher target discards most of Arm C's
information (it is a single sigmoid-bounded number derived from a 1000-column
input), but the raw 500 extra Arm-C columns contain too much future nuisance to
distill wholesale. A shared, predictability-constrained subspace — PLS between
the 500 causal columns and the 500 Arm-C-only columns, not a PCA of the Arm-C
columns alone — should sit in between.

**Design.** Same fold-0 pilot structure as §2.1. Fit PLS latent directions
`Z = B_PLS^T F_arm_c` for `d ∈ {8, 16, 32}` using only `outer_train` rows,
predict `Ẑ` from the 500 causal columns via small LightGBM regressors, nest
exactly as in §2.1. Compare against a PCA-of-Arm-C control at matched `d` to
confirm the gain is from predictability-selection and not merely
dimensionality reduction.

**Continuation gate.** `≥+0.004` fold-0 standalone over the PCA control at the
same `d`, **and** `≥+0.002` for `RT-600 + PCFB` over `RT-600 + seed-clone`.

**Note on sequencing.** Run 2.1 before 2.2. If SST's `R` ratio (oracle-to-legal
retention) is very low — the future targets are highly informative but not
statistically predictable from the prefix — that is strong evidence PCFB will
fail for the same reason, and should deprioritise it before spending the extra
2–3× compute PCFB costs over SST.

### 2.3 ORR pilot — cheapest complement, but judged against a stacked prior

**Hypothesis.** A specific, teacher-confirmed subset of `RT-600`'s same-`t`
inversions (where `RT-991`'s margin is confidently right and `RT-600`'s margin
is wrong) is recoverable from the existing 500-column state even though global
pointwise distillation is not the right way to reach it.

**Design.** Reuse the nested inner-fold `Q`/`RT-600` OOF caches already being
produced by the Wave-7 teacher run — no new teacher training required. Mine
same-`t` pairs within each inner holdout where `m_R < 0` and `m_T` is in the
top teacher-margin quantile; train a shallow LightGBM repair-propensity
regressor; blend `S = S_RT600 + β·r`, `β` selected on inner folds only.

**Why this is graded on a curve.** §0 already showed same-`t` pairwise
objectives are mildly negative for LightGBM on this data (`RT-111`, `RT-700`,
`RT-701`, roughly −0.001 to −0.005). ORR is not exactly that experiment — it
trains a *regression* target (repair propensity) rather than a pairwise loss,
and it operates on a pre-filtered subset rather than all pairs — but it is
close enough that a positive result must be checked against "did this simply
avoid the specific failure mode of the earlier pairwise attempts, or did it
happen to work for an unrelated reason." Report the repair-pair accuracy
*and* damage to previously-correct pairs, not just aggregate TS-AUC.

**Continuation gate.** `≥+0.002` marginal `RT-600 + ORR` over `RT-600 +
seed-clone` on the fold-0 pilot, with repair-pair causal recoverability
(`P[r_p > r_n | m_R<0, m_T>δ]`) meaningfully above 0.5 — not just the blend
number, since a blend can move even if the repair mechanism itself is closer to
noise.

---

## 3. WHAT WOULD CHANGE THE PRIORITY ORDER

**If `T2` fails its 5-fold promotion battery:** do not run any of §2 as
written. Re-diagnose why the teacher signal did not transfer (calibration?
teacher-student mismatch of the kind Stanton et al. describe for KD generally —
fidelity and downstream generalisation are not the same thing, a pattern this
project already has one data point for in `T1`'s inconsistency vs `T2`'s
consistency) before proposing a second distillation-adjacent mechanism.

**If SST's oracle arm (B) itself shows little dominant-cell lift:** that would
mean the 8 chosen structural channels are not the right decomposition of Arm
C's advantage, not that the advantage is unreachable — expand the channel set
before concluding the route is dead.

**If SST's legal arm (C) matches its oracle arm (B) closely:** the current
500-column prefix already contains the anticipatory signal and the classifier
simply never sees it framed usefully. This would argue for spending compute on
better feature *framing* (SST-style auxiliary targets folded into the existing
7-stream ensemble) over anything neural, and would deprioritise CFEP further —
if a shallow regressor already recovers most of the oracle gain, a dilated-conv
encoder is solving a problem that does not need solving.

**If SST's legal arm badly underperforms its oracle arm across every channel
tried:** that is the CFEP-relevant result — it would suggest the causal
information is not absent from the *value* of the prefix but from its
*representation*, which is the one argument for accepting Route 4's cost and
its Wave-6 failure risk. Even then, run it as a strict ablation against a
BCE-trained encoder of matched capacity (Wave 6's `RT-960`/`RT-970` architecture
family, reused rather than re-invented) so the result isolates the *predictive*
training objective, not "having a neural net."

---

## 4. WHAT THIS PROPOSAL DOES NOT CLAIM

No leaderboard projection is offered here. `LEADERBOARD_ASSAULT_STATUS.md`
already states the external score (0.6268) has not moved since `RT-600`, and
`RDOF_LEDGER.md`'s promotion bar exists precisely because this project has
repeatedly seen dev-fold gains shrink or invert once stacked against a real
matched control, an alternate partition, or the lockbox. Any of the five routes
above that clears its fold-0 gate still owes: the full 5-fold run, the
paired-bootstrap CI, the alternate-partition check, and — only after all of
that, and only if the gain looks ensemble-additive rather than redundant with
`T2` — a lockbox measurement, which this project spends exactly twice per
finding and has already spent twice on the current champion.
