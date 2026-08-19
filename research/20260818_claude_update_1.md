# Claude update 1 — 2026-08-18

ADIA Lab / CrunchDAO **Structural Break Challenge, Real-Time Edition**.
One session, 78 logged experiments, 8 research agents, waves 1–6.
Competition deadline: **17 September 2026** — 30 days out.

---

## 1. Where the code is

### Headline numbers

All figures are the official Time-Stratified AUC on **series-level** folds. The
2,000-series lockbox was never used for any selection decision.

| | mean OOF (5 folds) | lockbox |
|---|---|---|
| Shipped EWMA/CUSUM/variance noisy-OR detector (`RT-000`) | 0.52051 | — |
| `m00_core` alone — calibrated null evidence + LightGBM (`RT-101`) | 0.56349 | — |
| Best single model, 500 causal features (`RT-100`) | 0.61510 | 0.60791 |
| Four-stream ensemble (`RT-130`) | 0.62394 | **0.61214** |
| Seven-stream **rank**-average ensemble (`RT-131`) — *not implementable, see below* | 0.62541 | — |
| **Seven-stream logit-average ensemble (`RT-160`) — current champion** | **0.62561** | not re-measured |

Per-fold for the champion: 0.63653 / 0.61998 / 0.63622 / 0.61762 / 0.61772
(std 0.00868).

> **Correction logged 2026-08-19.** `RT-131` blended streams by averaging
> within-timestep rank percentiles across series. The crunch runner is
> series-sequential and single-pass, so that cross-section does not exist at
> inference and `RT-131` could never have been submitted. Averaging logits is a
> per-series function, is deployable, and scores 0.62544 against the rank
> average's 0.62524 — deployability costs nothing. Found by reading the inference
> contract before starting the streaming port rather than after. Paired series-level bootstrap against the shipped detector:
**+0.0951, 95 % CI [+0.0841, +0.1059], 120/120 replicates positive.**

For context, the strongest public 2026 approaches described in the deep-research
report sit around 0.575–0.579. We clear that by ~0.046 on OOF and ~0.033 on the
lockbox — though that is a reconstruction on our folds, not a replication of
their code.

### What exists, physically

```
src/sbr/                     the reusable library
  store.py                   per-series float32 memmap store (10,000 series, 35M points)
  metric.py                  official TS-AUC, one lexsort, exact mid-rank ties
  nullcal.py                 per-series historical-null calibration engine
  transforms.py              historical characterisation + per-point transforms
  pipeline.py                leakage-safe OOF training/evaluation + ledger
  features/                  8 causal feature modules behind a registry
research/                    folds, ledger, reports, negative results
tests/test_sbr_metric.py     metric parity + prefix invariance, 5 passing
```

Eight feature modules, 550 columns built, 500 used by the champion:

| module | cols | what it monitors | +Δ on `m00_core` (screen) |
|---|---|---|---|
| `m00_core` | 151 | calibrated multi-scale online-vs-null evidence | baseline 0.57384 |
| `m01_seq` | 60 | CUSUM / Page-Hinkley / GLR / Shiryaev-Roberts + peak & persistence shape | +0.0143 |
| `m02_dist` | 59 | PIT occupancy, divergences, tails, ranks | +0.0131 |
| `m03_dyn` | 60 | ACF, volatility clustering, spectral, wavelet, complexity | +0.0300 |
| `m04_resid` | 60 | AR(1/2/3/5), ridge/robust AR, EWMA/GARCH volatility normalisation | +0.0286 |
| `m05_ctx` | 50 | historical context (series-constant) | **rejected** |
| `m06_loc` | 60 | online change-point localisation, τ̂-anchored segment statistics | +0.0315 |
| `m07_bayes` | 50 | absorbing-state posterior, BOCPD, e-processes, Bayes factors | +0.0397 |

Champion gain share: `m07_bayes` 24.6 %, `m01_seq` 18.0 %, `m00_core` 16.3 %,
`m03_dyn` 14.8 %, `m04_resid` 13.9 %, `m02_dist` 8.9 %, `m06_loc` 3.7 %.

### The thing that is not done

**Nothing here can be submitted.** Every module is batch-first: it computes a
whole trajectory from cumulative sums. The competition hands you one observation
at a time and wants one score back. There is no streaming port, no
batch/stream parity test, and no submission notebook wired to the new library.
The 0.6254 is real but currently unshippable.

---

## 2. What is finished from the original brief

The brief asked for 20 minimum deliverables plus 7 stretch items.

**Delivered (17 of 20):**

1. Official metric parity — implemented from spec, verified against a slow
   sklearn reference to 1e-12 including ties, protected by a test.
2. Permanent series-level folds — 5 dev folds + 2,000-series lockbox, stratified
   on break/no-break × τ quartile × history tertile × online tertile, hashed.
3. Leakage-safe OOF framework — series-level throughout; row subsampling applies
   to training only, validation always scores every online row.
4. Causal training-row / feature generation system — registry + driver, with the
   causal contract enforced bitwise.
5. 2026 break taxonomy — all 8,000 dev series, 560-column artifact, detectability
   curves, transient rates, generator-artifact audit.
6. Calibrated historical-null prototype — became the backbone of every module.
7. Residualisation prototype — `m04_resid`, with the over-whitening question
   answered directly.
8. Multi-scale detector bank — `m01_seq`.
9. Distribution/PIT prototype — `m02_dist`.
10. Historical-context analysis — `m05_ctx`, with H1 vs H2 settled.
11. Supervised LightGBM benchmark — many.
13. Direct-ranking experiment — pairwise-t, lambdarank, XE-NDCG.
14. OOF prediction files — 7 full-scale streams saved.
15. Prediction-correlation / ensemble analysis — including the within-timestep
    correction, which turned out to matter.
17. Populated experiment ledger — 78 rows, appended under a file lock.
18. Ranked research roadmap — in `STATE_OF_RESEARCH.md`.
19. Tests confirming causal inference — prefix invariance across all 8 modules.
20. No unverified claims — with one exception I had to retract; see §3.

**Partial (1):**

12. CatBoost / XGBoost comparisons — run on the screen store only (XGBoost
    0.5988, CatBoost 0.5723 against a LightGBM control of 0.6068), never at full
    scale, and never measured for **blend delta**, which is the number that
    actually matters for a portfolio.

**Not delivered (2):**

16. Public-solution → actionable experiment map. The read-only research agent was
    never spawned. The deep-research report substituted for it, which is thinner
    than what was asked for.
20a. Several stretch items: **deep temporal models, synthetic augmentation, hard
    negatives, 2025→2026 transfer, and symbolic/automatic feature search were all
    skipped.** Two reasons, both real: this container has 2 CPU cores and 7 GB of
    RAM, and three of the four wave-3 agents were killed mid-task by a monthly
    spend limit.

**Stretch items delivered:** Bayesian sequential model (`m07_bayes`, the single
best family), specialist mixture-of-experts (built and rejected), and a full OOF
stack.

---

## 3. What did not work, and why

### The pattern worth internalising

**Three of three screen-level *architectural* wins reversed at full scale. Every
screen-level *feature* addition transferred.** All three reversals share one
mechanism: they help a data-starved model and stop helping once the model is not
data-starved.

| idea | screen (2,000 series, 151–211 cols) | full scale (6,400 train series, 500 cols) |
|---|---|---|
| Historical-context block as features | +0.0009 | **−0.0177** |
| Pairwise-t ranking objective | +0.0072 | dead heat (see below) |
| DGP-cluster gated specialists | +0.032 | **−0.0219** on both folds |

The protocol now states it plainly: the screen store is valid triage for new
features and is **no evidence at all** for objectives, architectures, or anything
that repartitions the training data.

### Individual failures

**Historical context as features.** Agent 8's permutation control is what exposed
it: context vectors deranged *within* a fold score 0.057 *below* using no context
at all. Fifty near-continuous series-constant columns give each training series a
near-unique fingerprint that LightGBM memorises. Separately, H1 ("history predicts
whether a break occurs") was rejected on three independent tests — context alone
scores 0.50143, a series-level `has_break` classifier scores 0.5068 against a
label-permuted null of 0.506–0.516, and break rate is flat across DGP clusters.

**Gated specialists.** The clusters carry real information — gating beats its own
permutation control by +0.0064 — but splitting 6,400 series into six groups costs
far more than the routing gains, and the largest cluster holds ~47 % of series
while the smallest holds ~3 %. A 500-column, 600-tree global model already learns
the conditioning that an explicit router had to supply for a 151-column one.

**Ensemble weighting and subset selection.** Every scheme lost to the plain equal
average: honest leave-one-fold-out greedy subset selection −0.0008, LOFO logistic
stack −0.0001, LOFO LightGBM stack −0.0038. The best subset chosen *in hindsight*
beat the average by 0.0003 — and the gap between that 0.0003 and the honest
−0.0008 is exactly the self-deception on offer. Seven streams inside a 0.010 band
with a 0.011 per-fold spread means any weighting fits fold noise.

**Feature selection.** No free lunch: all-500 = 0.62689, top-300 = 0.62645,
top-200 = 0.62111, top-60 = 0.60952. The columns earn their place collectively
even though most are individually weak. Top-300 is a deployment trade (40 % less
inference cost for −0.0004), not a research gain.

**GARCH(1,1) volatility normalisation.** 0.50012 standalone — literally zero
signal. Power falls monotonically with filter adaptation speed (EWMA hl=63 0.535
→ hl=22 0.515 → GARCH 0.500). Never let a filter adapt on the same timescale and
in the same channel as the break you are hunting.

**Trend features, KS statistics, tail-asymmetry, rank-CUSUM, slope-of-evidence
channels, mixture-GLR, EWMA level banks** — all zero or negative marginal value.
The trend result is consistent with the forensics: trend breaks barely exist here.

**Alternative targets** — soft ramp −0.012, log-hazard regression −0.018,
`scale_pos_weight=4` −0.009, XE-NDCG −0.014. Plain binary `1[t ≥ τ]` survived
every attack.

### A claim I had to retract

I reported the pairwise-t objective as "failed to promote, −0.0033" on the
strength of a single carefully-paired fold. Over five folds it scores **0.61444
against the champion's 0.61510 on 30 % fewer training rows** — a dead heat. The
same objective moved 0.008 *on the same fold* between two runs. **One fold is not
a result when the per-fold spread is 0.011**, and neither the screen's +0.0072 nor
my −0.0033 was real. It is now a permanent ensemble member. The retraction is in
`FAILED_EXPERIMENTS.md`.

### Process failures worth recording

- The causality harness caught **two genuine look-ahead leaks** during
  development — a mixture-GLR normaliser that depended on total series length,
  and a candidate grid spaced by `n_online` rather than by `t`. Neither was
  visible in any score.
- A separate distribution audit caught a column exploding to ±3e7 from an
  unfloored null spread. It passed prefix invariance. **Causality checks do not
  substitute for sanity checks.**
- Two runs were lost to OOM at 6 GB before I noticed an arm was duplicating both
  feature matrices.
- Syncing the research workspace into the git clone silently reverted a fix that
  existed only in the clone. Now applied to both trees.

---

## 4. Five paths forward

### Path A — Streaming port and ship it

Turn the 8 modules into incremental `update(x)` form, parity-test each against
the batch reference, benchmark ms/point, wire the submission notebook.

**Pros.** It is the only path that converts 0.6254 into a leaderboard position;
everything else is worth zero without it. Bitwise prefix invariance means a
correct-but-slow reference already exists for free — recompute on the growing
prefix and emit the last row — so every optimised module has an exact parity
oracle. Risk is engineering risk, which is the kind you can schedule.

**Cons.** Largest single block of work, and it buys **no** accuracy. Naive prefix
recompute is O(n²), roughly 125 s per series at n=1000, so real incremental state
is needed for the expensive modules. We still do not know the competition's
runtime cap, which makes it hard to know when to stop optimising.

**Verdict: this is the only genuinely mandatory path.** The question is not
whether but when.

---

### Path B — More ensemble streams

Add a DART model, a different-seed champion, a soft-gated model, CatBoost and
XGBoost as members rather than replacements.

**Pros.** The only thing that has transferred cleanly to the lockbox: 4 streams
+0.0087, 7 streams +0.0102, and the lockbox confirmed +0.0062 of it. Mechanically
simple — each stream is one `run()` call and the blend rule is parameter-free, so
there is nothing to tune and nothing to overfit. Diminishing but still positive.

**Cons.** Returns are shrinking (+0.0087 → +0.0102 for three more streams), each
stream is ~35 minutes of compute on 2 cores, and **every stream multiplies the
streaming-port cost**, since each must run causally at inference. A 7-model
ensemble is a 7× inference bill.

---

### Path C — 2025→2026 transfer and synthetic augmentation

Convert known-boundary 2025 series into pseudo-real-time examples; fit DGPs to
2026 histories and generate hard negatives — transients that revert.

**Pros.** The only remaining idea that adds **data** rather than model variety,
and everything else we have tried is a variation on re-reading the same 10,000
series. The forensics give it a sharp target: 17 % of no-break series contain
break-lookalike transients, and the disambiguator must be persistence rather than
amplitude — exactly what hard negatives would teach. Highest ceiling of anything
untried.

**Cons.** Highest uncertainty. The 2025 DGP may not match 2026, and the brief's
own rule is that synthetic performance is not evidence. Substantial build effort
before the first number arrives. Validation stays on real 2026 folds, so a
negative result costs a lot of time to establish.

---

### Path D — Localisation v2

`m06_loc` was built by an agent that ran out of budget before tuning it, and still
delivered +0.0315. The forensics measured a ~3 AUC-point gap between oracle-window
and prefix-anchored comparison at 160–320 observations elapsed.

**Pros.** A measured, quantified opportunity rather than a hunch — we know how
much is on the table. Attacks the single largest structural weakness: every other
feature dilutes post-break evidence with pre-break online points by a factor that
depends on τ. Self-contained work in one module.

**Cons.** τ̂ is noisy and the honest risk is that better localisation buys nothing
beyond a denser window bank — the falsification test for this was specified but
never run to completion. Adds inference cost to a module that already needs a
streaming port.

---

### Path E — Consolidate and de-risk

No new alpha. Seed and fold-assignment stability study, byte-level metric parity
against the live Crunch scorer, deploy the top-300 subset, tidy the repo, carve a
fresh holdout to replace the spent lockbox.

**Pros.** The champion rests on **one seed and one fold assignment** — we have no
idea how much of the 0.0089 fold spread is fold-assignment luck. The metric is
verified against a reference implementation but never against the organiser's
actual scorer, and the deep-research report explicitly flags that the pair-count
weighting could not be confirmed from public material. **If the weighting is
wrong, every number in this project is measuring the wrong thing.** Cheapest path
by far.

**Cons.** Adds zero accuracy and does not move the submission forward. Easy to
spend a week feeling productive while the leaderboard position stays at nil.

---

### Recommended sequence

**A → E(metric parity only) → B → D**, with C as the stretch if the port lands
early.

The reasoning: A is mandatory and its cost is knowable, so it should start now
rather than at the end of a month. The metric-parity check inside E is a few hours
and protects everything, so it rides along. B is cheap accuracy but should wait
until the port exists, because the port's cost scales with stream count — better
to know that price before adding members. D is the best-quantified remaining
alpha. C has the highest ceiling and the highest chance of consuming two weeks for
nothing, which is a bet worth making only from a position where something is
already shipping.

---

## 5. Open risks

- **Nothing is submittable.** Thirty days to deadline.
- **Metric weighting is unconfirmed** against the live scorer.
- **The lockbox is spent** — two touches. Further selection needs a fresh holdout
  carved from the dev folds.
- **Single seed, single fold assignment.** No stability estimate beyond the 0.0089
  fold spread.
- **Screen-to-full transfer is unreliable** for anything architectural — the rule
  is now written down, but it was learned three times the hard way.
- **Compute.** 2 CPU cores and 7 GB of RAM shaped every decision in this project.
  Anything involving deep models or large searches needs a different machine.
