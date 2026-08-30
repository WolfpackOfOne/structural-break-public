# Claude Code Research Prompt

## What this file is, and how it relates to the other prompt

You are AGENT_ID, one of five independent senior quantitative research agents.

`chatgpt_research_prompt.md` is the **execution brief**. It owns the independence
rule, the repository-forensics requirement, the causal protocol, the forbidden
low-value answers, and the ten-section output contract. Follow it for all of
that. **Nothing in this file changes your output format or your output path.**
When saving your report, your filename must still include your unique
agent/model identifier, as required by the execution brief.

This file is the **evidence brief**. It exists so that five agents do not each
spend their first hour re-deriving the same numbers, and so that none of you
reasons from a stale summary. It gives you:

1. the competition's hard rules and the real compute envelope (§1),
2. the measured score history and the measured failure record (§2, §3),
3. a detailed decomposition of the 2025 winning solution (§4),
4. four adversarial questions the execution brief does not ask (§5).

**Every number below is a REPORT-CLAIM until you verify it.** They were read on
2026-08-29 from `STATE_OF_RESEARCH.md`, `STATUS.md`, `FAILED_EXPERIMENTS.md`,
`LEADERBOARD_ASSAULT_STATUS.md`, `research/reports/`, and the 2026-08-29
`unresolved-quant-questions` investigation — several of which live in sibling
working directories rather than on your checked-out branch. Some are known to be
stale or historically scoped: the canonical `RESULTS.csv` includes an early
wave-1 block with 77 `git_sha=nogit` rows and still preserves the historical
illegal `RT-131` "NEW CHAMPION" ledger row, even though later rows and
`STATUS.md` supersede it. Treat this section as a map of where to look, not as a
substitute for looking. If your forensics contradict a number here, **the
repository wins — report the discrepancy explicitly in your Repository Evidence
section.**

---

## §1 — RULES AND THE REAL COMPUTE ENVELOPE

The execution brief states the causal protocol qualitatively. Here are the
binding quantities.

**You submit code, not predictions.** The Crunch cloud runs your module through
`train(datasets, model_directory_path)`, `infer(datasets, model_directory_path)`,
and a module-level `INFER_PARALLELISM`.

**Inference is series-sequential and single-pass, one point at a time.** Series
`i`'s score at step `t` is emitted before series `i+1` is seen. The cross-section
does not exist at inference.

**Metric.** `TS-AUC = Σ_t w(t)·AUC(t) / Σ_t w(t)`, `w(t) = n_pos(t)·n_neg(t)`.
Verified against the official docs; our implementation is parity-verified.
Only within-timestep ordering is scored, so any monotone transform applied
identically inside a timestep is free, and **elapsed time — an enormous row-level
predictor — is worth exactly zero.** Several failures in §3 reduce to that fact.

**Runtime: 15 hours per week = 54,000 s.** Test is 10,000 public + 10,000 private
series at ~504 mean online length → 5.04M scored points public, 10.08M across
both. Allowed cost per point:

| test size | points | P=1 | P=4 | P=6 | P=8 |
|---|---|---|---|---|---|
| 10,000 series | 5.04 M | 10.7 ms | 42.9 ms | 64.3 ms | 85.7 ms |
| 20,000 series | 10.08 M | 5.4 ms | 21.4 ms | 32.1 ms | 42.9 ms |

Measured: the reference streamer runs the seven champion modules at **92.9
ms/point**; the deployed RT-600 path runs at **64.9 ms/point** (62.6 with the
ported `m00_core`), fitting the P=6 public budget. `INFER_PARALLELISM = n` starts
n processes with isolated memory, so **RAM multiplies by n**.

**Determinism is a hard eligibility rule.** Re-run on 10% of the data,
predictions must match to **1e-8** or the solution is ineligible for rewards.
The forum confirms carrying state across completed series is *mechanically*
permitted but breaks determinism, because parallelism offsets differ between the
full run and the validation run. **This is what makes cross-series state illegal
in practice.** Any mechanism you propose that requires cross-series state or any
RNG in the inference path is disqualified — check yours before you write it down.

**Compute actually available:**
- Wave-1/2 agent containers: **2 CPU cores, 7 GB RAM**, shared.
- Local development: **Mac, 10 cores, 16 GB, arm64, CPU-only.** MPS is present
  and deliberately unused — `use_deterministic_algorithms` does not cover that
  backend and the causality gates require ≤1e-8. Threads pinned at 6.
- **RTX 4090 via the Crunch cloud** benchmark path; used once, to benchmark TabM
  and RealMLP.
- Training: best single model ≈ **6.5 min/fold on 2 cores**; feature build
  ≈ 0.25 s/series; a full nested 5-fold run ≈ **4–5 h wall clock**.
- Environment trap: **torch + LightGBM in one process segfaults on macOS/arm64**
  (three competing OpenMP runtimes). The suite runs in two processes and unions
  failure sets. `KMP_DUPLICATE_LIB_OK` was rejected — it trades a loud crash for
  silent numerical corruption in the two libraries this project compares bitwise.

Your §10 final recommendation is explicitly scoped to 15 hours. Price your
proposals against the numbers above, not against an imagined cluster.

---

## §2 — WHAT THE DATA CONTAINS, AND WHAT THE SYSTEM SCORES

**Break taxonomy, measured on 8,000 dev series:**
- **Location breaks essentially do not exist.** Median |mean shift| 0.054σ for
  break series vs 0.053σ for placebo splits; series-level AUC 0.4998. The
  classical change-point literature targets a break type this data lacks.
- Family strength: scale 0.559, Wasserstein 0.558, dependence 0.538, shape 0.522,
  trend 0.518. Strongest single statistic found anywhere: **AR(6)-residual
  log-sd ratio, 0.603**.
- **92.4% of break series are individually indistinguishable from a placebo
  split** at p<0.01 on any family — a weak-signal aggregation problem, not a
  detection problem.
- 17.0% of no-break series contain a break-lookalike transient; so do 15.6% of
  matched break-free windows. **Transients are the DGP.** The disambiguator must
  be persistence, not amplitude.

**Representation.** Seven causal modules, 500 columns: `m00_core` (calibrated
multi-scale null evidence, 151), `m01_seq` (sequential detector bank with
peak/persistence channels, 60), `m02_dist` (PIT/occupancy/divergence, 59),
`m03_dyn` (dependence/spectral/wavelet/complexity, 60), `m04_resid` (AR and
volatility residuals, 60), `m06_loc` (online change-point localisation, 60),
`m07_bayes` (absorbing-state posterior, BOCPD, e-processes, 50). All pass bitwise
prefix-invariance at `atol=0.0`.

**Score history** (5-fold series-level OOF TS-AUC unless noted):

| id | what | score |
|---|---|---|
| `RT-000` | shipped EWMA+CUSUM+variance noisy-OR baseline | 0.52051 |
| `RT-101` | `m00_core` alone | 0.56349 |
| — | strongest public 2026 approaches in the prior-art report | ~0.575–0.579 |
| `RT-100` | best single LightGBM, 500 cols | 0.61500 |
| `RT-130` | 4-stream ensemble | 0.62374 |
| `RT-131` | 7-stream **rank** average | 0.62524 (**never implementable**) |
| `RT-160` | 7-stream **logit** average (deployable) | 0.62544 |
| `RT-600` | seven-specialist SCDF blend, production anchor | 0.62581 dev |
| `RT-600` | **external Crunch leaderboard** | **0.6268** |
| `RT-1257` | CatBoost two-slot hybrid, submission #16, 2026-08-29 | **0.6290 external** |

**Estimand warning — this is where the ledger is most dangerous.** These numbers
are not all the same quantity. Pooled-vs-mean OOF, whole-population vs
dominant-cell, `E2−E1` (against a deteriorating clone control) vs `E2−E0`
(against the fixed deployment counterfactual), and single-split series AUC vs
five-fold row-level TS-AUC all appear in the reports, sometimes adjacent.
`RT-600`'s **fold-0** E0 score is quoted as ≈0.63828 in the Wave-7 batteries;
its five-fold development mean is 0.62581 and its pooled dev TS-AUC is 0.62563.
All three are real, and they are different estimands.
The retired claim "RT-1264 adds +0.00593 over RT-600" is an `E2−E1` artifact; the
fixed-baseline figure is +0.00203. **Do not mix these. Verify which estimand any
number you cite actually is.**

**Lockbox.** The 4-stream blend scored **0.61214** on 2,000 untouched series
against a 0.62374 dev estimate — a **−0.0116 haircut**. That is the only clean
read on dev→held-out transfer in the program, and it is the number to price any
claimed dev gain against.

---

## §3 — THE MEASURED FAILURE RECORD

The execution brief tells you not to re-run an ordinary model search. This is why.

**Neural learners (Wave 6) — outright failure, with a diagnosed mechanism.** MLP
on the *identical* 500 columns and the *identical* 1M sampled rows scored 0.57059
and 0.58061 against a 0.61605 LightGBM control, **0/5 folds positive**. The
mechanism was read off the training history: training BCE falls to 0.077 while
the control sits at 0.61605, because the row target `y[t] = 1[t ≥ τ]` is monotone
in `t`, so the network learns elapsed time — which TS-AUC deletes entirely. Trees
capped by depth and `min_data_in_leaf=300` could not exploit it far enough to
hurt. The regularisation arm confirmed the mechanism rather than repairing it.
**Any sequence model you propose must say explicitly how it avoids this trap.**

**Teacher/distillation (Wave 7) — real single-model alpha, no ensemble alpha.**
The first pilot showed +0.0162/+0.0167 and was **outer-fold contaminated**: `Q`
was cross-fitted per-row but not per outer-validation-fold. Contamination was
worth +0.0122 to +0.0235. The corrected nested run gave T2 **+0.00943 standalone,
5/5 folds, CI [+0.0043,+0.0137]** — but an ensemble marginal over a *seed clone*
of only **+0.00024**. It learned something real that RT-600 already knew. A
fold-purity sentinel now exists and was validated in both directions (0/20
violations on the corrected scheme, 20/20 on the old one).

**Wave 8 / New Avenues — exhausted.** Five full-population ensemble marginals
≈ **−0.00031**. Second sweep SS-01…SS-04 all **KILL** (−0.00029, −0.00030,
−0.00031; SS-03 also failed the pre-break damage cap, 0.0199 > 0.0150). 79
catalogued mechanisms across 15 families produced no confirmation candidate.

**Learner Diversity 2026.** CatBoost INTERESTING but not SERIOUS
(`marginal_vs_clone` +0.00111); **TabM and RealMLP INFEASIBLE** at frozen full
scale. The later two-slot CatBoost hybrid `RT-1257` is the one promotion, at
`E2−E0` +0.002026, 5/5 positive folds. In that fixed E2-E0 contrast, folds 3-4
supply about 64% of the summed delta; the clone-control marginal has a different
per-fold shape. No alternate-partition evidence exists.

**Feature-level negatives, each with a mechanism.** Slope-of-evidence channels:
gain exactly 0.0 (rank-equivalent to `current − decayed_peak`, already present).
Hard-threshold cross-channel counts: gain 0.0 — discretising for a GBM is
self-harm, since the tree builds its own threshold from the continuous max.
Calibrated *current levels* of sequential statistics: near-zero **conditional on**
`m00_core`, though worth +0.00845 standalone. EWMA bank: dominated by the
length-matched-null rectangular bank. Mixture-GLR: dominated by max-GLR.

**Every attempt to be clever about combination lost to the plain average.**
Honest LOFO greedy subset selection −0.0008; LOFO logistic stack 0.62514; LOFO
LightGBM stack 0.62144; raw probability averaging 0.62288 — all against 0.62544
for the parameter-free logit average. Feature pruning: top-300 −0.0004, top-200
−0.0058, top-60 −0.0174.

**Two process failures that cost more than any negative result.** (1) `RT-131`
was champion for a week while being unimplementable: it averaged within-timestep
rank percentiles, a function of the cross-section, which does not exist at
inference. It was nearly shipped. (2) The canonical ledger rotted, and canonical
HEAD is a disconnected 16-commit harness lineage while the real research lives on
shared-object-database branches, tags, and worktrees.

**The structural diagnosis the program ended on.** The exact loss cube: dominant
remaining loss is `current t ≥ 200 AND break age ≥ 100` — **45.29% of exact
remaining pairwise inversion loss, 50.50% of pair weight, cell AUC 0.66428**, and
inside that cell **never-break negatives carry 74.0% of the loss**. On that exact
cell, Arm B (same information, more tree capacity) *lost* (1/5 folds, −0.00592),
while Arm C (same capacity plus each series' own **final-row** features) beat Arm
B on **5/5 folds, mean +0.07110** cell AUC. The recorded verdict at the time
was a **future-information limit, not a representation or extraction limit**.

**But that verdict is no longer settled.** The Arm-C prediction vector
(`RT-991.npy`) exists under the sibling Wave 8 OOF artifacts, so the stale
missing-file claim is false. `research/scripts/armc_residualization.py` first
decomposed Arm C's +0.07110 into endpoint/horizon structure and residual signal:
after a cross-fitted fixed-`t` projection of `logit(RT-991)` on `{n_online,
n_online-t, t/n_online}`, the residual still scored **0.685794** on the dominant
cell, **+0.038303** over Arm B, with within-t rank rho **0.457016** vs Arm B.
Then `research/scripts/armc_residual_student.py` trained the requested nested
fold-pure causal student of that residual on the existing 500-column prefix bank
only, over the full dev population with no `t+h` eligibility filter. The student
kept useful signal: raw dominant-cell TS-AUC **0.656890** (**+0.009398** vs Arm
B), `RT600 + residual_student` pooled whole-dev gain **+0.001973**, marginal vs
seed-clone blend **+0.001951**, never-break net rate **+0.003409**, and pre-break
damage **0.013672 < 0.015000**. A light combination with the current CatBoost
hybrid is also positive: `RT1257 + residual_student` is **+0.001396** mean
whole-dev TS-AUC over `RT-1257`. This is a **confirmation-only score candidate**,
not a new RT ID and not a production change. Reports:
`research/reports/armc_residualization.md` and
`research/reports/armc_residual_student/armc_residual_student.md`.

---

## §4 — HOW THE 2025 EDITION WAS WON

The execution brief summarises the Alphabot solution as a process example. Here
is the decomposition, so your §3 mechanisms can engage with specifics.

Source: *"Structural Break — Alphabot's Team Solution"*, Humberto Brandão, João
Peinado, Mario Filho, Rafael Alencar (Alphabot), October 2025.

**Two caveats first.** That was the **batch edition**: the series arrives already
split into `period=0` / `period=1`, the boundary is **given**, and one prediction
is emitted per series. Nearly every Alphabot feature is a two-sample pre-vs-post
test at a known boundary, which does not exist here — the execution brief's
warning against copying illegal offline features stands. Second, **the write-up
reports no scores and no final rank.** It is a method document. Treat the method
as the evidence and cite no number from it.

**Their method: stacking, built on enforced independence.**
- Four members worked **completely independently** on features *and* models,
  exchanging nothing until the end. Their stated reason: early exchange *"can
  harm the quality of our solutions, since team members' thought processes start
  to be influenced by one another's opinions, preventing new perspectives from
  emerging."* This exercise is a deliberate reconstruction of that mechanism.
- **Humberto — 58 classical-statistics features.** Welch t on |x| across local
  spans; Fisher aggregation of p-values across window scales; Fligner–Killeen;
  F-variance; BDS independence; supF step/hinge scan; min-p Wilcoxon;
  Jensen–Shannon; Hellinger; differential entropy on central quantile bands;
  Expected Shortfall at 0.95; ACF Fisher-z difference tests; and **run-length
  statistics for quiet-band persistence**.
- **Mario — two stages, ending in a random search.** Stage 1: per-regime
  descriptors (moments, quantiles, ACF at lags 1/2/5/10/20, Sharpe, Sortino,
  profit factor, CVaR at 1/2.5/5/10%, Gini, max drawdown, regression coefs), then
  **cross-regime fusion operators** — `diff`, `absdiff`, `product`, `ratio` — plus
  KS, Wasserstein, Euclidean. Stage 2: **287 composite interaction features found
  by semi-automated random search** — sample 2–4 stage-1 variables, apply a random
  operator and a random nonlinearity (sin, cos, tanh, log1p, sqrt), **keep only if
  validation improves**. Series transforms `{x, |x|, log(1+x), cumsum,
  cumprod(1+x)}`.
- **Rafael — many representations of one signal.** Raw; IQR-filtered at
  λ∈{1.0,1.75,2.0} with absolute-value and sign-flipped variants; Symlet-2
  wavelet denoising; cumsum; pct-change. Windowed stats over windows {20…240}.
  Eight distributional distances (Canberra, Bray–Curtis, Hamming,
  Jensen–Shannon, Rogers–Tanimoto, Dice, Yule, Cityblock) at cuts
  {30,120,150,170,200}. KS full, windowed, **and within-pre placebo splits**
  (first half vs second half) — a null-calibration idea that *is* legal here.
  477 engineered interactions plus 22 alternates.
- **João — meta-features over first-level predictions.** Combined the first-level
  models' *outputs* with statistical descriptors, merging up to six values into
  one via subtraction, product, ratio, arithmetic mean, geometric mean, harmonic
  mean.
- **Eight first-level models, all tree-based** — XGBoost classifiers and
  regressors per member's feature group, one five-seed bagged, a Random Forest on
  combined features, one XGBoost over everything — feeding a **second-level model
  on ~100 features**.

**Two observations for your analysis.** Their feature emphasis — dispersion,
tails, distributional distance, dependence, persistence — matches our measured
taxonomy (scale 0.559 dominant, location 0.4998 dead), which is weak external
corroboration that the taxonomy is right. And both programs independently
converged on trees over neural networks.

---

## §5 — FOUR QUESTIONS THE EXECUTION BRIEF DOES NOT ASK

Answer these **inside the sections the execution brief already defines** — most
fit §1 Executive diagnosis, §6 Arbitration, §7 Oracle/information-ceiling, and §8
What should we stop doing. Do not invent new top-level sections for them.

**A. Resolve the stacking contradiction.** Alphabot won on stacking. We measured
every stack and every weighting as *worse than doing nothing* — LOFO logistic
0.62514, LOFO LightGBM 0.62144, honest greedy −0.0008, all against 0.62544 for a
parameter-free average. Both findings cannot be generally true. Which is local to
its setting? Is it the metric, the per-step correlation structure, the fact that
our seven streams share one representation and therefore have nothing to stack,
or the fact that their stack combined **four disjoint feature philosophies**
while ours combined seven hyperparameter variants? **Fold your answer into §6 —
it bears directly on whether we need a better detector or better arbitration.**

**B. Falsify the frame.** For eight waves the frame has been *richer causal
representation → better within-timestep ranking*. What would have to be true for
that frame itself to be wrong? Specifically: is this problem label-noise-limited
or Bayes-limited near ~0.63–0.65? If it is, the correct remaining work is
variance reduction, calibration, lockbox economics and submission strategy — not
signal, and §8 should say so. **Give the cheapest observation, using artifacts
that already exist, that separates "near the Bayes limit" from "stuck in a local
frame."** If you think the frame is right, say what evidence makes it right
rather than merely unfalsified. This belongs in §7.

**C. Import from a field that has solved this shape of problem.** Somewhere,
someone must emit an irrevocable per-step judgment about a weak, persistent,
absorbing-state change, scored against a cohort they cannot observe. Sequential
clinical monitoring. Seismology foreshock discrimination. Predictive maintenance
and remaining-useful-life. Survival analysis under interval censoring.
Statistical process control. Fraud and account-takeover scoring. Astronomical
transient alert brokers. **Name the field, name the method precisely, and state
what breaks in transfer** — against no mean shifts, 92.4% individually
undetectable, within-timestep-only comparison, single-pass causal inference, the
15 h/week budget, and 1e-8 determinism. A transfer that dies on one constraint is
still a useful answer if you say which one and why. At least one of your five
mechanisms should come from outside this repository's intellectual lineage.

**D. The highest-value move that is neither a feature nor a learner.** The whole
program lives in {features} × {learners} × {blend}. Consider and go beyond:
*generative* — the data is 92.4% individually undetectable and plausibly
synthetic from a known family, so reverse-engineer the DGP and attack it with
simulation-based inference / amortised posterior estimation on unlimited
simulated series with known `τ`; *objective surgery* — train directly on the
pairwise within-timestep comparisons TS-AUC actually scores, rather than on row
labels; *cohort distillation* — the cross-section is unavailable at *inference*
but fully available at *training*, and was only ever used illegally at inference
(`RT-131`), never distilled into a legal per-series student; *boundary
reconstruction* — Alphabot's entire feature set presumes a known `τ`, so ask what
an internal `τ̂` posterior would have to look like before those 822 features
become computable online within 64 ms/point. **At least one of your five
mechanisms should live outside that product space.**

---

## §6 — STANDARD OF EVIDENCE

The execution brief's MEASURED FACT / INFERENCE / HYPOTHESIS discipline applies
to everything here. Additionally:

- A number in this file is a **REPORT-CLAIM**. Promote it to MEASURED FACT only
  after you verify it against branch evidence, and say where you verified it.
- Report any discrepancy you find between this file and the repository. Finding
  one is a result, not an inconvenience.
- Do not treat the external leaderboard as a tuning signal. There are two
  external reads in this program's history (0.6268, then 0.6290). One observation
  is a calibration point, not a transfer law.
- Price every dev-measured gain against the **−0.0116 lockbox haircut** before
  calling it material.
- Any mechanism requiring cross-series state or inference-path RNG is
  disqualified by the 1e-8 rule. Say so yourself if one of your ideas dies there.
- Prefer diagnostics on existing artifacts over anything that retrains. Where you
  must retrain, price it against §1 and state the kill criterion first.
