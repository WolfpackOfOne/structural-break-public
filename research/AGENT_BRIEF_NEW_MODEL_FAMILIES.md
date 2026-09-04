# BRIEF: find a new model family for the 2026 Structural Break Challenge

You are a research agent. Your job is to propose **new model families** for a
structural-break detection problem that is deep into diminishing returns. The
programme has run roughly 40 distinct mechanisms and killed almost all of them.
Your value is in proposing something *structurally different*, not in retrying a
variant of what is listed below as closed.

Read the whole brief before proposing. A proposal that collides with a closed
lane is worse than no proposal, because it costs a training run to rediscover.

---

## 1. THE PROBLEM

**Competition:** 2026 ADIA Lab / CrunchDAO Structural Break Challenge — Real-Time
Edition (`structural-break-real-time`).

Each **series** is a univariate numeric time series split into two parts:

- a **history** segment, guaranteed break-free, fully visible from the first step;
- an **online** segment, streamed one point at a time.

Some series contain exactly one structural break at an unknown index `tau`
somewhere in the online segment. Others never break.

**Row-level target:** for online index `t`, `y[t] = 1[t >= tau]`. So a broken
series emits negatives before its break and positives after it. A never-break
series emits negatives forever. There are therefore **two kinds of negative** and
the distinction matters enormously (see §4):

- **never-break negatives** — series that never break;
- **pre-break negatives** — series that will break, but have not yet at time `t`.

**Scale:** 10,000 training series, ~504 online points each, **5,036,517 total
rows**. Development partition is 8,000 series / **4,032,524 rows** in 5 fixed
series-level folds. A further 2,000-series lockbox existed and is now spent (all
10,000 are used for the final fit).

**Test:** 10,000 series public (repeated submissions scored here, leaderboard
feedback) + 10,000 series private (your *selected* submission is scored once at
close; that decides prizes).

---

## 2. THE METRIC — read this carefully, it shapes everything

**Time-Stratified AUC.** For each online index `t`, take every series still alive
at `t`, compute the cross-sectional ROC AUC between series already broken
(`t >= tau`) and series not broken, and weight that timestep by the pair count:

```
TS_AUC = ( sum_t w(t) * AUC(t) ) / ( sum_t w(t) ),    w(t) = n_pos(t) * n_neg(t)
```

Consequences you must internalise:

1. **It is a within-timestep, cross-sectional RANKING metric.** Only the ordering
   of series *against each other at the same `t`* matters. Absolute calibration
   is irrelevant except insofar as it changes that ordering.
2. **A feature constant along a series' trajectory can still matter** — it shifts
   that series' whole ranking — but it can never act as a within-series detector.
3. **The unit of loss is an inverted (positive, negative) pair at the same `t`.**
   All diagnostics below are in "pair repairs vs pair damage" because that is the
   metric's own currency. A mechanism can have excellent standalone AUC and still
   damage more pairs than it repairs.

---

## 3. HARD LEGALITY AND DEPLOYMENT CONSTRAINTS

These are not stylistic. Violating them invalidates a result or forfeits prize
eligibility.

- **Causality.** Row `t` of any feature may use only the history and
  `online[:t+1]`. This is enforced by a bitwise prefix-invariance harness at
  `atol=0.0`. A mechanism that has not passed it cannot be scored.
- **No true `tau`, directly or indirectly.** Anything whose *availability,
  support, length, missingness, denominator, calibration window or segment
  boundary* depends on true `tau` is invalid — an earlier experiment scored 0.866
  this way and was voided as a label leak. Placebo cuts do not repair it.
- **No `n_online`.** Total online length is a forbidden input (it leaks how long
  a series survives).
- **Determinism required at 1e-8** on a 10% replay. Non-deterministic solutions
  are ineligible for rewards.
- **No cross-series state at inference.** Carrying a summary of completed series
  forward into later ones is mechanically permitted, but it is incompatible with
  our reward-eligibility standard unless it survives parallelism-invariant
  deterministic replay; platform guidance says it will likely fail that test.
  **Treat the entire "use the cross-section at time t" family as closed. Do not
  propose it.**
- **Runtime: 15 hours per week.** Current champion measures 2.28 ms per online
  point, ~3.7 h per 10,000 series. There is real headroom, but a mechanism costing
  10x the current per-point budget is not deployable.

---

## 4. WHERE THE REMAINING LOSS ACTUALLY IS

Exact pairwise-inversion loss decomposition of the **RT-600 reference
architecture**, which remains the programme's canonical diagnostic surface
(computed from the metric's own numerator/denominator, not approximated):

**Dominant cell** — `t >= 200` AND positive post-break age `>= 100`:
- **50.50%** of all pair weight, **45.29%** of all remaining loss, cell AUC 0.664.
- Within the cell: never-break negatives are 74.25% of weight and 73.99% of loss;
  pre-break negatives are 25.75% of weight and 26.01% of loss.
- **Difficulty is essentially uniform across negative type** (loss/weight ratios
  0.9965 and 1.0100). The often-quoted "74% of loss is never-break" says only
  where the *weight* is. Do not read it as a weakness.

**Performance by post-break age** (positives restricted to the bucket, negatives
held fixed):

| age | 0-5 | 5-10 | 10-20 | 20-50 | 50-100 | 100+ |
|---|---|---|---|---|---|---|
| TS-AUC | 0.515 | 0.534 | 0.549 | 0.577 | 0.605 | 0.662 |

Young breaks are near chance. This is plausibly information-limited physics
(detecting a break 3 observations after it happens), not a modelling gap — but
nobody has proven that.

**The oracle bound.** A diagnostic arm ("Arm C") given each series' own *final*
online state scores **0.71859** in the dominant cell against a matched legal arm's
0.64749 — a gap of **+0.07110**, 5/5 folds. It is non-causal and undeployable; it
is a ceiling on how much signal the full sequence contains, not an achievable
target. A preregistered analysis (below) found most of that gap is **not** visible
from the legal prefix.

---

## 5. THE CURRENT SYSTEM

**Feature bank: 500 causal columns in 7 modules**, all computed streaming and
prefix-verified:

| module | cols | content |
|---|---|---|
| `m00_core` | 151 | core evidence recursion, tail/level monitors |
| `m01_seq` | 60 | CUSUM / CUSUM-SQ detector paths, peak/decay/persistence geometry |
| `m02_dist` | 59 | distribution distances on the historical-ECDF PIT (chi2, JS, Hellinger, TV, CvM, KS, Wasserstein, energy) |
| `m03_dyn` | 60 | ACF/PACF-style z-scores, dynamics |
| `m04_resid` | 60 | AR/GARCH residual moment monitors vs a history null |
| `m06_loc` | 60 | location-shift family |
| `m07_bayes` | 50 | Bayesian online changepoint posterior channels |

Every module fits its parameters on the **history only** and freezes them —
adaptive filters were tested and erase the break they are hunting.

**Champion architecture (`RT-1320`, 8 members):** seven diverse boosted-tree
streams over different module subsets and hyperparameters (LightGBM, plus two
CatBoost slots), each calibrated by a fold-pure smooth time-conditional CDF
(`SCDF_NSEEN`), then **equal-weight averaged**, plus an eighth member that is a
distilled "Arm-C residual student". Streams train on 700k-900k sampled rows each,
not the full 4M.

**Score ladder (public test set, same data release):**

| model | dev OOF | public |
|---|---|---|
| RT-600 (7 LightGBM) | 0.625811 | 0.6268 |
| RT-1257 (2 CatBoost slots) | 0.627838 | 0.6290 |
| **RT-1320 (+ student, current)** | **0.629254** | **0.6303** |

**Current status:** RT-1320 is the external champion and selected submission #19
at 0.6303. RT-1257 remains the formal production anchor; those roles are
intentionally distinct.

Dev estimates have predicted external gains to ~0.0001 twice running. **The
dev-fold machinery is a trustworthy instrument. Decide on dev; the leaderboard
only confirms.**

---

## 6. HOW A CANDIDATE IS JUDGED — the bar you must clear

Standalone AUC is **not** a promotion criterion. The binding test is the
**ensemble marginal against a matched control**.

The historically validated addition contract that promoted RT-1320 was:

- `E0` = RT-1257.
- `E1` = RT-1257 + one extra member that is a seed clone of an existing member
  (same features, different seed).
- `E2` = RT-1257 + the Arm-C residual student.
- In that specific eighth-member experiment, adding an ordinary exchangeable
  member was worth only about **+0.00003**. That is historical evidence from the
  RT-1257-to-RT-1320 contract, **not** a universal clone prior.

For **new** research, the ultimate baseline is the current external champion,
RT-1320. Define the matched contract before training:

- for an **additive candidate**, `E0 = RT-1320`, `E1 = RT-1320 + matched extra
  control`, and `E2 = RT-1320 + candidate` — this is a ninth-member test;
- for a **replacement candidate**, `E1` must replace the same RT-1320 slot with a
  genuinely exchangeable matched control and `E2` must replace that slot with the
  candidate.

The candidate must beat `E1`, not merely `E0`, and `E2-E0` must be inspected as a
secondary deployment endpoint. **Do not assume the historical +0.00003 control
lift carries over to RT-1320 or to a ninth-member ensemble.**

- **Noise floor: 0.0011** (paired series-level bootstrap). Below that is not a result.
- Requires fold consistency (4/5 or 5/5) and positive dominant-cell pair flow.
- Robustness across four alternate fold partitions is expected for promotion.

**Two failure modes that have killed almost everything, learn them:**

1. **Beating a weak control.** Several candidates beat a *degraded* control
   (seed clone, synthetic clone, global no-adaptation baseline) and lost to the
   actual incumbent. If your control is worse than the champion, clearing it means
   nothing.
2. **Standalone signal with zero marginal.** One candidate scored *better than the
   RT-600 reference in the dominant cell standalone* (0.6732 vs 0.6643) and added
   nothing, because its within-`t` rank correlation with RT-600 was 0.886.

---

## 7. WHAT HAS BEEN TRIED AND KILLED

Do not propose these. If you believe one deserves revival you must state a new
falsifiable reason the prior failure mechanism no longer applies.

**Feature families built as modules and rejected:** `m09_back`, `m10_persist`,
`m11_focus`, `m12_rdep` (residualised distribution distances, residual
CUSUM/CUSUMSQ paths, AR-coefficient likelihood ratio), `m13_scale_survival`,
`m14_spectral_impulse`, `m15_ordinal_irrev`, `m16_joint_rarity`, `m17_observers`,
`m18_weighted_ctm`.

**Notable individual kills, with the number that killed them:**

| mechanism | result |
|---|---|
| Hankel-DMD observers on the raw prefix (delay 16, rank 4, rolling windows; reconstruction error, **subspace-angle residual**, **effective-rank monitors**) | marginal +0.000226, **rho vs RT-600 0.886**, cell pair flow net −11. **This is "PCA/SSA/dynamic PCA/Koopman as a break detector" and it is CLOSED.** |
| Frozen Kalman/NIS observer residuals | marginal +0.000135, rho 0.884 |
| IM2 matched-length run null + dwell bank incl. **max-run growth exponent** | marginal +0.000301, cell pair flow **net −1151**, rho **0.38** — the most decorrelated candidate ever produced, still strongly negative |
| Trajectory geometry (NN provenance, arc-rate) | marginal −0.002587 |
| Spectral impulse, ordinal irreversibility, joint rarity, weighted CTM, difficulty gating, failure-manifold routing, relay logic, scale survival | all KILL |
| `m12_rdep` as a full architecture rebuild (7 streams rebuilt with it) | **−0.00268 vs incumbent**, 1/5 folds |
| Linear bottleneck of the 500-col bank: PCA(16) / PLS(16) → LightGBM | PCA −0.00148, PLS −0.00270 cell AUC. **Note: this compressed the handcrafted bank; it is a different hypothesis from the DMD row above.** |
| Future-aware / teacher-distillation: SST, ORR, PCFB, CFEP, TGMC | all KILL |
| Teacher distillation T1 / T2 | T2 has real standalone alpha (+0.00943) but ~+0.000237 ensemble marginal — mostly redundant |
| Neural: MLP, TCN on the same channels | failed |
| GPU tabular: TabM (standalone 0.601, rho 0.585, marginal +0.00004), RealMLP (standalone 0.560, marginal −0.0032) | KILL. "Different architecture + decent standalone" is demonstrably insufficient |
| Learned representation: NNCSR, corrected ACGN | KILL — a fixed per-series null beat the learned representation |
| Residual CatBoost slots | **LANE CLOSED.** CAT-410 is KILL; CAT-411/412/414/415 all finished below the RT-1257-relative 0.0011 deployment floor. CAT-412 was best at only **E2-E0 +0.000338234** (3/5; dominant-cell net −31); CAT-414 +0.000178858, CAT-415 +0.000175228, CAT-411 +0.000133942. CAT-412's optional alt-partition rescue was scoped and declined. |
| Objective variants: pairwise_w, pairwise_h, hinge | worse than incumbent pairwise_t |
| Ensemble: learned stacking, weighting, subset selection, static DGP/fingerprint routers, specialist disagreement micro-routing, repair-damage arbitration | all lost to the equal-weight average |
| Per-series history adaptation (affine, AR(5)) | beat a global clone but lost to the fixed per-series null by −0.0215 |
| Counterfactual synthetic augmentation | beat a synthetic clone, negative pair flow, below champion |
| Calibration anchor placement | total spread across all schemes including random: 0.00006 |
| Within-timestep rank averaging | ILLEGAL — cross-sectional, forfeits determinism |

**Two preregistered structural findings that bound the search:**

- **Bank saturation.** 72 fitted OOF vectors over this bank have a participation
  ratio of **2.730** — roughly three effective dimensions, 60% of variance in one
  component. A cross-fitted convex blend of all 72 cannot reliably beat the
  incumbent (mean +0.0022, 95% CI containing zero, needing 22 members).
- **The oracle gap is largely not shadowable.** On the pairs the oracle repairs
  and the legal model misses, three causality-verified excursion channels that
  discriminate at 0.561/0.604/0.562 across the cell generally score
  **0.5023/0.4999/0.5027** — indistinguishable from chance, 99% CIs containing
  0.50. The information the oracle uses is not visible in legal excursion
  evidence. The preregistered H1a analysis was **STRENGTHENED**: T2's corrected
  pairs strongly overlap Arm C's repair set (`R_lift = 2.540`, 5/5 folds), and
  **69.8%** of everything T2 repairs lies inside Arm C's repair set, yet the
  declared legal prefix channels are essentially chance on the exact residual
  pairs Arm C fixes. Teacher transfer is reaching the oracle's problem region;
  most of the remaining oracle information is still not legally observable.

**The two-sided squeeze you must escape.** High-correlation channels (rho
0.86–0.89) are redundant and add nothing. The single most decorrelated channel
ever produced (rho 0.38) still damaged 1.6 pairs per repair. Being *different* is
not sufficient and being *good standalone* is not sufficient. You need a reason
your mechanism repairs pairs the bank actually gets wrong, without damaging the
ones it gets right.

---

## 8. WHAT IS GENUINELY STILL OPEN

Short list. Be sceptical of all of it.

1. **A second, better Arm-C-derived student.** This is the only mechanism with a
   *confirmed external gain* (+0.0013). The existing one is a single unoptimised
   design with a negative fold 0, and its gain is entirely a never-break-cut
   false-positive repair (never-break pair net +0.0034, pre-break +0.0004 ≈ 0).
   The pre-break cut is untargeted — but it is only 13% of total pair weight, so a
   proportional gain there prices at roughly +0.0006, below the noise floor.
   **This does not reopen generic teacher distillation.** A second student needs a
   materially different residual target or pair-repair mechanism, ideally one
   that repairs errors beyond the current never-break false-positive cut; ordinary
   "use the future as teacher" variants remain closed.
2. **Explicitly nonlinear representation learning on the raw prefix** — kernel
   PCA and similar. The Hankel-DMD arm was *linear*, so this is not strictly
   covered. But note the linear version failed by *redundancy*, which a nonlinear
   version would also have to escape.
3. **Anything that genuinely changes the information set** while staying inside
   {history, online prefix}, per-series, causal, deterministic. Nobody has found
   such a thing in ~40 attempts.

---

## 9. COMPUTE AVAILABLE

- **Local:** Apple M2 Pro, 10 cores, **16 GB RAM** (the binding constraint), ~43 GB
  free disk. Feature builds over the full store run 3–50 minutes per module. A
  single 5-fold LightGBM stream on ~900k sampled rows is tens of minutes.
- **GPU:** NVIDIA RTX 4090, 24 GB VRAM, CUDA 12.8, available for neural work.
- **Deployment budget:** 15 h/week inference; current champion uses ~3.7 h per
  10,000 series, so roughly a 4x per-point headroom exists.
- Full 5-fold training of one new stream plus the ensemble marginal battery is
  approximately half a day end to end. Assume you can fund **a handful** of
  serious arms, not dozens.

---

## 10. WHAT TO DELIVER

For each proposal, give:

1. **The mechanism**, precisely enough to implement, and the *physical* reason it
   should separate a post-break series from both a never-break and a pre-break
   series at the same `t`.
2. **Why it is not any row in §7.** Name the closest prior experiment and say
   what is materially different. Search the implementation family, not the idea
   name — "PCA as a detector" is filed as Hankel-DMD, "excursion growth" as a
   dwell bank. Idea-name searches return false negatives.
3. **Why it will have low rank correlation with a 500-column bank containing
   CUSUM paths, PIT distances, AR-residual monitors and a Bayesian changepoint
   posterior** — and why that decorrelation will convert into *repaired pairs*
   rather than damage, given that a rho-0.38 channel already failed.
4. **Causality argument**: how every column is computable from history +
   `online[:t+1]` with no dependence on `tau` or `n_online`.
5. **Integration contract and repair target, declared before training.** State
   whether this is a ninth-member addition, an RT-1320 member replacement, or a
   residual correction; define `E0/E1/E2` and the matched control explicitly.
   Name the pair population you expect to repair — never-break, pre-break,
   dominant-cell mature breaks, or another predeclared slice — and why.
6. **Cost estimate** against §9, and the cheapest experiment that could kill it.
7. **A preregistered falsification**: the number, the threshold, the control
   (which must be the incumbent-relevant one), and the fold-consistency
   requirement — written before running.

Rank your proposals by (probability of clearing +0.0011 champion-relative
marginal vs matched control) / (cost). Prefer one well-argued structurally novel
mechanism over five variants.

**It is a legitimate and useful answer to conclude that no proposal clears the
bar.** A well-evidenced "the information set is exhausted" is worth more than a
speculative mechanism that costs a week. Do not manufacture optimism.