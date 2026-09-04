# RT-1321 — LS-KD (Lag-Space Characteristic-Kernel Discrepancy) — PREREGISTRATION

**Written before any RT-1321 feature was evaluated against a label.**
Branch: `research/rt1321-lskd`. Briefing commit: `ebaff8a49903d75d5a59f7996dd308c803de19ad`.
Frozen incumbent: **RT-1320** (8 members, dev mean OOF TS-AUC 0.629254, public 0.6303).

## 0. ID allocation

`RT-1321` verified unallocated on this branch: it appears exactly once in the
tree, in `research/EXPERIMENT_ID_MAP.md` line 1111 ("`RT-1321` through `RT-1329`
remain unallocated lane contingency IDs"), and nowhere in `RESULTS.csv`,
`research/oof/`, `research/reports/` or any source file.

## 1. Hypothesis (exact, falsifiable)

The 500-column bank represents scalar marginal change (m02_dist, m06_loc),
recursive evidence (m00_core, m01_seq), low-order dependence (m03_dyn),
residual moments (m04_resid), Bayesian evidence (m07_bayes) and — via the
killed `RT-1215` arm — *linear* delay-space structure.

**H1.** There exists predictive information in a *characteristic-kernel*
comparison of the JOINT law of short consecutive states
`P(U_t, U_{t-1}, …, U_{t-d+1})` against the frozen historical joint law, which
is not recoverable from (a) the existing 500-column bank and (b) a
mechanism-matched *coordinate-separable* kernel discrepancy that sees only the
same nonlinear marginal information.

**H0.** Any apparent LS-KD signal is explained by nonlinear marginal
distribution distance plus what the bank already carries. Under H0,
`E2 - E1 ≈ 0`.

### Why this is not a closed lane

* **`RT-1215` Hankel-DMD** (`src/sbr/features/m17_observers.py`, delay 16, rank 4)
  is a **linear** delay-subspace observer: reconstruction error, subspace angle,
  effective rank. It measures whether the online delay cloud stays inside a
  *linear* subspace fitted on history. LS-KD measures the *distribution* of the
  delay cloud in an RKHS, which separates laws that share a linear subspace
  (e.g. a transition-law change with unchanged marginal and unchanged principal
  directions). `NEGATIVE_RESULTS_INDEX.md` explicitly records for that row:
  "Genuinely NOT covered: **kernel PCA** and other explicitly nonlinear variants".
* **`RT-1202` trajectory geometry** is NN provenance / arc-rate on z-normalised
  subsequences — a *geometric* statistic on windows, not a two-sample kernel
  distance against a frozen historical law.
* **`RT-1201` IM2 dwell/growth** is run-length / excursion mass. No delay space.
* **W8-PCFB** compressed the *handcrafted bank*, not the raw prefix.
* **`m02_dist` / `m12_rdep`** compute chi2/JS/Hellinger/TV/CvM/KS/W1/energy — all
  on the **1-D** PIT (raw and AR-residual). Energy distance is an MMD special
  case with the distance kernel, but it is applied to the *marginal*. The repo's
  own avenue ledger says so verbatim: `new_avenues_2026.csv` row **G5** —
  "every incumbent distance is on the MARGINAL (1-D); the delay-embedded cloud
  is joint over d lags" — rated priority HIGH, status "after Pilot 4", never
  executed. Row **H3** (sequential kernel MMD) is explicitly filed as collapsing
  *into* G5. **RT-1321 is the execution of G5.**

## 2. Feature definition (frozen)

### 2.1 Input streams (2)

* `u` — historical-ECDF PIT of the raw series. Repository canonical:
  `HistParams.pit`, surfaced as `ctx.tr["u"]` / `ctx.hist_tr["u"]`.
* `r` — AR-residual historical-ECDF PIT, constructed by importing
  `sbr.features.m02_dist._pit_against` and reproducing that module's own block
  verbatim (`sref = sort(hist_tr["res_mean"][p_ar:])`, PIT of online
  `tr["res_mean"]` against it, with m02_dist's `nh - p_ar > 50` fallback to `u`).

**Preregistered deviation from the task brief, declared before any label was
touched.** The brief names an "AR(5)-residual" stream. The repository's canonical
residual is **AR(2)**: `HistParams(ar_order=2)` is the default in
`sbr/features/base.py::make_ctx`, in `sbr/stream/ctx.py::StreamCtx` and in
`sbr/stream/engine.py::StreamEngine`, and `StreamCtx._AR_PAD/_AR_WIN` are sized
for it. Building an AR(5) stream would require a second residual pipeline in
both the batch context and `StreamCtx`, i.e. exactly the "subtly different
version of the AR residual normalization" the brief forbids, and would put the
bitwise batch/stream parity contract at risk. **Smallest defensible alteration:
use the repository's canonical AR(2) residual PIT stream.** The hypothesis is
about the *joint lag law*, not the AR order.

### 2.2 Delay vectors

`z_t^(d) = [u_t, u_{t-1}, …, u_{t-d+1}]`, depths **d ∈ {3, 5, 8}**.
For online `t < d-1` the vector is completed from the **tail of the historical
PIT stream**, which physically precedes `online[0]` in the same series (the store
holds one contiguous series split into `hist | online`). This is history-only,
so every online row `t ≥ 0` is defined and causal. No NaN rows.

### 2.3 Random Fourier features

`R = 32`, seed **1321**, `numpy.random.RandomState(1321)` (the legacy generator,
whose stream NEP-19 guarantees across NumPy versions — `default_rng` does not).
Drawn once at import in a fixed order, for `d` in `(3, 5, 8)`:

```
W[d] = rs.standard_normal((R, d))      # base directions
b[d] = rs.uniform(0, 2*pi, R)          # joint phases     (CANDIDATE)
B[d] = rs.uniform(0, 2*pi, (R, d))     # per-coordinate phases (CONTROL)
```

A unit test pins a SHA-256 over the concatenated base arrays.

**CANDIDATE (joint):**  `phi_j(z) = sqrt(2/R) * cos( (W[d][j] / sigma)·z + b[d][j] )`

**CONTROL (coordinate-separable / marginal):**
`phi_j(z) = sqrt(2/R) * (1/sqrt(d)) * sum_c cos( (W[d][j,c]/sigma) * z_c + B[d][j,c] )`

The control's induced kernel is `k(z,z') = (1/d) * sum_c k_1(z_c, z'_c)` —
additive across lag coordinates, therefore it carries coordinatewise marginal
distribution information and **no cross-coordinate interaction terms**. It is
fully causal; it is not a shuffled or future-aware control.

### 2.4 Bandwidth (history-only, deterministic)

Per (series, stream, depth): take the history delay-vector matrix, subsample at
most **192 rows at deterministic `np.linspace` positions**, compute the exact
median of all pairwise Euclidean distances, and clamp
`sigma = clip(median_dist, 0.05*sqrt(d), 2.0*sqrt(d))` (delay vectors live in
`[0,1]^d`, so `sqrt(d)` is the diameter; the clamp only fires on degenerate
histories). Frozen for the whole series. Nothing about `tau`, `n_online` or the
online segment enters.

### 2.5 Frozen historical reference

`mu_H = mean over ALL valid history delay vectors of phi(z)`. No subsampling
(the same rule everywhere, batch and stream). No online observation ever
updates it.

### 2.6 Online recursion

`lambda_h = 1 - 2^(-1/h)` for half-lives **h ∈ {32, 128}**.
`mu_{t,h} = (1 - lambda_h) * mu_{t-1,h} + lambda_h * phi(z_t)`, initialised at
`mu_{-1,h} = mu_H`.
`D_{t,d,h} = || mu_{t,h} - mu_H ||_2^2`.

### 2.7 History-only normalization

Replay the historical delay-vector sequence causally through the identical
recursion, initialised at `mu_H`, giving `D^H_j`. Discard a burn-in of
`min(4*h, L//4)` leading replay points (deterministic, history-only) so the null
is stationary while the online path is compared at its own stationarity. Take
`g = log1p(D^H)`, `m = median(g)`, `s = 1.4826 * median|g - m|` floored at 1e-9.
Frozen. Emit `S_{t,d,h} = clip((log1p(D_{t,d,h}) - m) / s, -8, 8)`.

### 2.8 Column count — 28 per block

* **12 core channels** `kd_{u,r}_d{3,5,8}_h{32,128}` — the primitive discrepancy,
  identifiable and un-buried.
* **4 aggregate paths** `A^{stream,h}_t = mean over d of S`, each emitting 4
  generic summaries in the repository's existing idiom (cf. `m12_rdep`'s
  `_cur/_pk/_dpk/_per`): `page` (`log1p` of positive Page accumulation with drift
  `k=1.0`), `pk` (running max), `dd` (peak minus current), `per` (fraction of
  elapsed online steps above 2.0). **16 columns.**

Candidate block `m19_lskd` and control block `m19_lskm` have **identical shape,
identical column names modulo the module prefix, and identical everything except
`phi`**.

### 2.9 Cost

Per online point: `2 streams × 3 depths × 32` cosines ≈ 192 transcendental ops
plus ≈ 2·10^3 multiply-adds for the EWMA/norm. Bounded state:
`2×3×2×32` floats for the EWMAs, `2×3×32` for `mu_H`, `d_max-1 = 7` trailing PIT
values per stream, 4 Page/peak scalars per aggregate path. **No prefix rescan.**

## 3. Causality contract

LS-KD inference reads only `hist` and `online[:t+1]`. It never reads `tau`,
anything derived from `tau`, `n_online`, the final online state, future
observations, cross-series state, or any label-dependent support. Enforced by
`sbr.features.base.check_prefix_invariance` and
`novel_streams.harness.verify` at **atol = 0.0** with `equal_nan=True`.

## 4. STAGE 1 — information screen (cheapest credible test), canonical fold 0

Anchor: the frozen **RT-1320** eight-member blend (`E0`), rebuilt with
`armc_residual_student`'s own SCDF_NSEEN cross-fitted calibration and equal-weight
blend. Both LS-KD blocks are reduced to a single standalone score by fitting one
LightGBM **on the kernel block alone** (no bank columns), fold-pure, so the
screen measures the block's own information.

Reported for **both** candidate and control: standalone whole-fold TS-AUC;
dominant-cell AUC (`t ≥ 200`, positive post-break age `≥ 100`); mature-positive
vs never-break; mature-positive vs pre-break; within-`t` rank correlation to
RT-1320; candidate/control correlation; incumbent pair repairs, damages and
repair−damage on each cut; conditional performance on RT-1320's hard pairs.
Diagnostic cuts use `tau` **for reporting only** and are never reachable from
inference.

### 4.1 STAGE-1 KILL RULE (binding, declared now)

KILL before any member training if **any** of:

1. `dominant-cell (repairs − damage)` of the **joint** candidate is **not greater
   than** that of the **marginal control**, on the fixed-perturbation test of §5;
2. mature-positive vs pre-break net pair flow of the joint candidate is clearly
   negative (net rate ≤ −0.002 at every preregistered epsilon);
3. the joint candidate's advantage over the control vanishes once marginal
   effects are controlled (i.e. §5 shows no epsilon at which joint − marginal
   dominant-cell net is positive);
4. `rho(candidate, RT-1320) > 0.90` **and** the candidate shows no clear
   advantage over the marginal control on dominant-cell repair−damage;
5. legality or determinism cannot be made exact without changing the hypothesis.

High standalone AUC alone does **not** pass. Low correlation alone does **not**
pass.

## 5. Fixed-perturbation pair-flow diagnostic (not the production ensemble)

Add `epsilon * (within-t rank of the block score − 0.5)` to the within-`t` rank
of RT-1320, for the **preregistered grid `epsilon ∈ {0.01, 0.02, 0.04}`**, and
report dominant-cell / mature-vs-never / mature-vs-pre repairs, damages and net
for **every** epsilon, for **both** candidate and control. The full grid is
reported; no epsilon is selected after the fact.

## 6. STAGE 2 — matched ninth-member contract

`E0` = RT-1320 (8 members, unchanged, artifacts never modified in place).
`E1` = RT-1320 + ninth member trained on **500 bank columns + m19_lskm** (control).
`E2` = RT-1320 + ninth member trained on **500 bank columns + m19_lskd** (candidate).

The ninth member is the repository's canonical production **pairwise_t**
LightGBM specialist configuration (`wave2_streams.py::RT-123R`, i.e. the
`RT-413` slot config): `n_estimators=600, learning_rate=0.05, num_leaves=63,
min_data_in_leaf=300, feature_fraction=0.5, bagging_fraction=0.7,
bagging_freq=1, lambda_l2=5.0, max_bin=127, objective=pairwise_t`,
`max_train_rows=700_000`, `sample_mode="uniform"`, `seed=0`.
E1 and E2 share fold, sampled training rows, row-sampling seed, model seed,
hyperparameters, objective, boosting, calibration family (fold-pure
`SCDF_NSEEN`), and ninth-member weight (equal-weight nine-member mean).
**No objective, hyperparameter or weight tuning.**

### 6.1 FOLD-0 AUTHORIZATION GATE (binding)

Proceed to five folds only if **all** hold on canonical fold 0:

* `E2 − E1 ≥ +0.0008` TS-AUC;
* dominant-cell repair − damage `> 0` (E2 vs E1);
* mature-positive vs pre-break repair − damage `> 0` (E2 vs E1);
* no substantial mature-positive vs never-break deterioration
  (net rate not worse than −0.002).

`E2 − E0` is inspected as a secondary deployment endpoint.

### 6.2 FIVE-FOLD PROMOTION GATE (binding)

Primary endpoint: mean OOF TS-AUC `E2 − E1`. Promote only if **all**:

* `E2 − E1 ≥ +0.0011` (the paired series-level bootstrap noise floor);
* `E2 − E0 > 0`;
* at least **4/5** canonical folds positive on `E2 − E1`;
* aggregate dominant-cell repair − damage `> 0`;
* aggregate mature-positive vs pre-break pair flow `> 0`;
* mature-positive vs never-break pair flow nonnegative or convincingly positive;
* paired series-level bootstrap supports the improvement.

If the canonical result reaches threshold, run the four alternate-partition
diagnostics (`folds_alt1/2/3` plus canonical) and require **≥ 3 of 4 positive**.

**Goalposts are frozen here.** No threshold, control, cut or endpoint may be
changed after any result is seen.

## 7. Failure interpretation (declared in advance)

* High standalone AUC + high rho to RT-1320 + `E2−E1 ≈ 0` → **rediscovery. KILL.**
* Low rho + weak standalone + negative pair flow → **different but useless. KILL**
  (RT-1201/RT-1202 shape).
* Joint beats marginal in raw pair flow but `E2−E1 ≈ 0` → **information exists
  but the bank/learner already recovers it. KILL for champion integration.**
* `E2−E1 > 0` driven only by never-break repairs with mature-vs-pre negative →
  treat with suspicion; still requires the full `+0.0011` and positive dominant
  flow.
* `E2−E1 ≥ +0.0011`, 4/5 or 5/5, positive mature-vs-pre and dominant flow,
  robust alternate partitions → **PROMOTE**, then build and push a Crunch
  submission. Selection of the private entry remains a separate manual act and
  RT-1320 stays selected unless dev and public agree.

## 8. Compute budget

Feature build over the full 10,000-series store: two blocks. Stage-1 screen:
two block-only fold-0 LightGBMs. Stage 2: two 5-fold LightGBM streams on 700k
sampled rows. This is the "handful of serious arms" the brief funds.
