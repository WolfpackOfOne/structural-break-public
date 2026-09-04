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

**Measured 2026-09-04, and the warning above was right.** `RT-1321` ran that
ninth-member control for the first time: `E1 - E0 = **−0.000554**`. It is not
+0.00003 and it is not zero — it is *negative*, an order of magnitude larger in
absolute value than the eighth-member figure, and with the opposite sign. A
candidate scoring `E2 - E1 = +0.0005` against it has not added +0.0005 to
anything; `RT-1321`'s `E2 - E0` was −0.000034. **Report `E1 - E0` on the same
line as your primary endpoint.** See §7's saturation finding.

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
| **LS-KD / RT-1321** — lag-space characteristic-kernel discrepancy on the raw prefix (Gaussian RBF via random Fourier features, R=32, depths 3/5/8, half-lives 32/128, history-only median-heuristic bandwidth, MMD of an online EWMA kernel mean against a frozen historical kernel mean, on raw-PIT and AR(2)-residual-PIT delay vectors) | 9th-member contract on RT-1320: **E2−E1 +0.000519** (4/5, floor +0.0011), **E2−E0 −0.000034**, **E1−E0 −0.000554**, bootstrap 95% CI [−0.000116, +0.001069] contains zero. Block **rho 0.162** — more decorrelated than anything except RT-1202 — and **0.4712 on the incumbent's own inverted pairs, below chance**. Mature-vs-never pair flow −25 vs E1 and **−54 vs E0**; mature-vs-pre −32 vs E0. **This is "kernel PCA / MMD / NEWMA / Scan-B / kernel two-sample as a break detector" and it is CLOSED**, together with avenues G5 and H3. Tested against a *coordinate-separable* kernel control, not a seed clone, so "joint lag structure beyond nonlinear marginal structure" is what was measured — and it is worth nothing here. |
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

**Structural findings that bound the search.** The first two are preregistered;
the third is a single measured observation and is labelled as such.

- **Bank saturation.** 72 fitted OOF vectors over this bank have a participation
  ratio of **2.730** — roughly three effective dimensions, 60% of variance in one
  component. A cross-fitted convex blend of all 72 cannot reliably beat the
  incumbent (mean +0.0022, 95% CI containing zero, needing 22 members).
- **RT-1320 may be saturated with respect to ordinary additive ninth members.**
  In the RT-1321 contract the *matched control* ninth member — an ordinary
  production-configuration LightGBM on the 500-column bank plus a feature block —
  scored `E1 − E0 = **−0.000554**`. Adding an exchangeable ninth member made the
  champion **worse**. Treat this as evidence, not a theorem: it is one matched
  control in one experiment, and the historical RT-1257→RT-1320 eighth-member
  contract measured an ordinary added clone at only about +0.00003, which is
  small but not negative. Two consequences you must act on anyway:
  **(a)** any candidate whose primary endpoint is `E2 − E1` must report
  `E1 − E0` next to it, because a degrading control manufactures a positive-
  looking primary — this has now happened three times (RT-1260/CAT-411,
  RT-1264/CSA-04, RT-1321), and in RT-1321 the entire +0.00052 headline was the
  control's own damage; **(b)** justify *why your idea belongs as a ninth
  member at all* rather than as a slot **replacement**, a **residual corrector**
  on the champion's own errors, or a **fundamentally different inference
  mechanism**. "Add one more member" is now the weakest available integration
  contract, and it is the one that has to argue for itself.
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

**The two-sided squeeze you must escape.** This is the single most important
paragraph in the brief, and it now rests on a monotone sequence in rho rather
than on two endpoints:

| mechanism | rho vs incumbent | what happened |
|---|---:|---|
| Hankel-DMD (`RT-1215`) | **0.886** | redundant; marginal +0.000226, cell pair flow −11 |
| Frozen Kalman/NIS (`RT-1214`) | 0.884 | redundant; marginal +0.000135 |
| GPU TabM (`RT-1258`) | 0.585 | different learner, standalone 0.601, marginal +0.00004 |
| IM2 dwell/growth (`RT-1201`) | **0.38** | decorrelated; cell pair flow **−1151**, 1.6 damaged per repair |
| **LS-KD block (`RT-1321`)** | **0.162** | most decorrelated channel with real construction discipline; **below chance (0.4712) on the incumbent's own inverted pairs** |
| Trajectory geometry (`RT-1202`) | 0.004 | orthogonal and worthless; marginal −0.002587 |

(The rho column is not all against the same incumbent — the pre-2026-09 rows are
within-`t` rank correlation against **RT-600**, the `RT-1321` rows against
**RT-1320**. The two incumbents correlate very highly with each other, so the
ordering is safe, but do not quote these as five decimal places of the same
quantity.)

Read down that table. Redundancy is not the binding constraint and decorrelation
is not the missing ingredient: **as rho falls, the marginal does not rise.**
`RT-1321` is the cleanest statement of it, because its block was measured
directly on the pairs the champion gets wrong and scored *worse than a coin*.

There is a second lesson inside the same experiment. The LS-KD *block* has rho
0.162, but the *ninth member trained on the bank plus that block* has rho
**0.835** — the learner maps most of the novelty back onto an incumbent-like
ranking, and still returns `E2 − E0 = −0.000034`. So "my channel is decorrelated"
is not even a durable property once a booster has seen it alongside the bank:
what survives training is the part that already agrees with the bank, and what
is discarded is the part that was different.

Being *different* is not sufficient, being *good standalone* is not sufficient,
and being *different after training* is not something you get to assume. You need
a reason your mechanism repairs pairs the bank actually gets wrong, without
damaging the ones it gets right — and the cheapest honest test of that is the one
`RT-1321` ran early and should have been believed on: **score your channel on the
incumbent's own inverted pairs before you train anything.** It costs no training
run — you need only the champion's OOF vector and your raw channel — and in
`RT-1321` it returned 0.4712 in the Stage-1 screen and correctly predicted the
Stage-2 verdict hours before the matched contract confirmed it.

Note this is a *different and stricter* population from the oracle-shadowing
analysis above, which scored three excursion channels at 0.5023/0.4999/0.5027 on
the pairs **Arm C** repairs. That one bounds what the legal prefix can see; this
one bounds whether *your* channel is aimed at the errors the deployed model
actually makes. Run both; they can disagree.

---

## 8. WHAT IS GENUINELY STILL OPEN

Short list, be sceptical of all of it — and it got shorter on 2026-09-04.
**There are now two entries, not three.**

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
2. **Something that genuinely changes the legal information-extraction
   mechanism**, while staying inside {history, online prefix}, per-series, causal,
   deterministic — and beyond the families already tested. Nobody has found such a
   thing in ~40 attempts. Note the emphasis: it is no longer enough to change the
   *statistic*. Delay embeddings, kernel two-sample distances, spectral,
   ordinal, dwell, conformal, Bayesian, observer, distributional and residual
   statistics have all been tried on the same extraction mechanism — a frozen
   history-fitted null, a causal online statistic, and a boosted tree over the
   result. What is untested is a different *mechanism*, not a 41st statistic
   inside this one.

**REMOVED 2026-09-04 — "Explicitly nonlinear representation learning on the raw
prefix (kernel PCA and similar)".** That lane is **CLOSED** by `RT-1321` / LS-KD.
It was open only because the Hankel-DMD kill (`RT-1215`) was explicitly linear;
`RT-1321` ran the explicitly nonlinear characteristic-kernel version of the same
hypothesis, against a coordinate-separable control that isolated exactly the
nonlinear-joint-versus-nonlinear-marginal question, and got `E2 − E0 = −0.000034`.
The delay-embedding lane is now shut from both ends: the linear version failed by
**redundancy** (rho 0.886), the nonlinear version failed by producing genuinely
novel information (rho 0.162) that **does not repair pairs**. Do not reopen it
without a reason that defeats both failure modes at once, and note that
"escape the redundancy" — the obvious response to `RT-1215` — is precisely what
`RT-1321` did successfully and it changed nothing.

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
   rather than damage, given that a rho-0.38 channel already failed and a
   rho-0.162 channel scored *below chance* on the incumbent's own errors. State
   your channel's expected AUC **on the pairs the champion currently inverts**;
   that number, not rho, is the one that has predicted every outcome so far, and
   it costs no training run to measure.
4. **Causality argument**: how every column is computable from history +
   `online[:t+1]` with no dependence on `tau` or `n_online`.
5. **Integration contract and repair target, declared before training.** State
   whether this is a ninth-member addition, an RT-1320 member replacement, or a
   residual correction; define `E0/E1/E2` and the matched control explicitly.
   Name the pair population you expect to repair — never-break, pre-break,
   dominant-cell mature breaks, or another predeclared slice — and why.
   **If you choose "ninth-member addition", justify it.** §7 records
   `E1 - E0 = −0.000554` for an ordinary matched ninth member: that contract now
   starts from a deficit your candidate has to pay off before it adds anything,
   and a slot replacement, a residual corrector on the champion's own errors, or
   a different inference mechanism may be strictly better places to put the same
   idea. Whichever you choose, commit to reporting `E1 - E0` alongside `E2 - E1`.
   Prefer a **mechanism-matched** control to a seed clone whenever your
   hypothesis names a mechanism — `RT-1321`'s coordinate-separable control is the
   worked example: it turned "does joint lag structure add anything beyond
   nonlinear marginal structure" from a rhetorical question into a measurement.
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