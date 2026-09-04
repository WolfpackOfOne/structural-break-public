# RT-1321 — LS-KD (Lag-Space Characteristic-Kernel Discrepancy) — **KILL**

**Verdict: KILL. No Crunch submission was built and none was pushed.**
Branch `research/rt1321-lskd`. Preregistration `RT1321_PREREG.md`, frozen before
any label was evaluated. Incumbent RT-1320 was never modified; it was rebuilt
read-only from its frozen OOF vectors to mean fold TS-AUC **0.629253722**,
matching the recorded `0.6292537222254164` to nine decimals.

---

## 1. Hypothesis

That a characteristic-kernel comparison of the **joint** law of short
consecutive states `P(U_t, …, U_{t-d+1})` against the frozen historical joint
law carries predictive information not recoverable from (a) the 500-column bank
and (b) a mechanism-matched **coordinate-separable** kernel discrepancy that
sees only the same nonlinear *marginal* information.

This is the repository's own avenue **G5** (`new_avenues_2026.csv`: "every
incumbent distance is on the MARGINAL (1-D); the delay-embedded cloud is joint
over d lags", priority HIGH, never executed), and it is the "kernel PCA and
other explicitly nonlinear variants" that `NEGATIVE_RESULTS_INDEX.md` records as
**genuinely not covered** by the killed linear Hankel-DMD arm `RT-1215`.

## 2. Exact feature definition

Two 28-column blocks, identical in every respect except the feature map.

**Inputs (2 streams).** `u` = historical-ECDF PIT (`HistParams.pit`, via
`ctx.tr["u"]`). `r` = AR-residual historical-ECDF PIT, built by importing
`m02_dist._pit_against` and reproducing that module's own block — one definition
of each, no parallel construction.

**Preregistered deviation.** The task brief named an AR(5) residual stream; the
repository's canonical residual is AR(2) (`make_ctx`, `StreamCtx`, `StreamEngine`
all default to `ar_order=2`, and `StreamCtx._AR_PAD/_AR_WIN` are sized for it).
Building AR(5) would have required a second residual pipeline in both the batch
context and the streaming context — exactly the "subtly different version of the
AR residual normalization" the brief forbids — and would have put the bitwise
`StreamCtx` parity contract at risk. Declared in `RT1321_PREREG.md` §2.1 before
any label was touched. The hypothesis concerns the joint lag law, not the AR
order.

**Delay vectors.** `z_t^(d) = [u_t, u_{t-1}, …, u_{t-d+1}]`, `d ∈ {3, 5, 8}`.
For `t < d-1` the vector is completed from the tail of the historical PIT, which
physically precedes `online[0]` in the same series — the same warm start
`ar_filter_causal` already uses. Every online row is defined; no NaN rows.

**RFF.** `R = 32`, seed `1321`, `numpy.random.RandomState` (NEP 19 guarantees
that stream across NumPy versions; `default_rng` does not). Drawn once at import
in fixed depth order `(3, 5, 8)`:
`W[d] = rs.standard_normal((R,d))`, `b[d] = rs.uniform(0,2π,R)`,
`B[d] = rs.uniform(0,2π,(R,d))`.
Basis SHA-256 pinned at **`ef89c1cc0834ce13442a321fafcb853a8cbf67205f63456bd37b933e6efd0580`**
by `tests/test_m19_lskd.py::test_rff_basis_is_frozen`.

* **CANDIDATE `m19_lskd`** (joint):
  `phi_j(z) = sqrt(2/R) · cos( (W[d][j]/sigma)·z + b[d][j] )`
* **CONTROL `m19_lskm`** (coordinate-separable / marginal):
  `phi_j(z) = sqrt(2/R) · (1/sqrt(d)) · sum_c cos( (W[d][j,c]/sigma)·z_c + B[d][j,c] )`,
  whose induced kernel is `k(z,z') = (1/d) sum_c k_1(z_c, z'_c)` — additive
  across lag coordinates, therefore coordinatewise marginal information with **no
  cross-coordinate interaction terms**. Fully causal; not a shuffled or
  future-aware control.

**Bandwidth (history-only, deterministic).** Per (series, stream, depth): at most
**192** history delay vectors at `np.linspace` positions, exact median of all
pairwise Euclidean distances, clamped to `[0.05·sqrt(d), 2.0·sqrt(d)]`
(delay vectors live in `[0,1]^d`). Frozen for the series.

**Historical reference.** `mu_H` = mean of `phi` over **all** valid history delay
vectors — no subsampling, the same rule batch and stream. No online observation
updates it.

**Online recursion.** `lambda_h = 1 - 2^(-1/h)`, `h ∈ {32, 128}`;
`mu_t = (1-lambda) mu_{t-1} + lambda phi(z_t)`, `mu_{-1} = mu_H`;
`D_{t,d,h} = ||mu_t - mu_H||²`.

**History-only normalization.** Replay history through the identical recursion
from `mu_H`; drop a burn-in of `min(4h, L/4)`; `m = median(log1p(D^H))`,
`s = 1.4826·MAD`; emit `S = clip((log1p(D) - m)/s, -8, 8)`. Frozen; no online
adaptation; no label ever enters.

**Columns (28).** 12 core `kd_{u,r}_d{3,5,8}_h{32,128}` (the primitive
discrepancy, identifiable and un-buried) plus 4 aggregate paths
`A = mean over d` per (stream, half-life), each emitting `page` (log1p of a
positive Page accumulation, drift 1.0), `pk` (running max), `dd` (peak − current)
and `per` (fraction of elapsed steps above 2.0) — 16 columns, in the `m12_rdep`
`_cur/_pk/_dpk/_per` idiom.

**Complexity.** Fit: `O(n_hist · R · d)` per (stream, depth) plus a `192²`
median. Step: `2·3·R = 192` cosines and `2·3·2·R` EWMA updates —
`O(sum_d R·d)`, no prefix rescan, bounded state
(`2·3·2·32` EWMA floats + `2·3·32` frozen means + 7 trailing PIT values per
stream + 4 scalars per aggregate path ≈ 4 kB per series).

## 3. Causality proof and test results

| gate | result |
|---|---|
| bitwise prefix invariance, `atol=0`, `equal_nan` | **PASS** — 12 real store series (cuts 1/3/10/37/111) and 10 synthetic families × 2 modules |
| future-mutation: poison `online[cut:]`, rows `< cut` must not move | **PASS** (batch and streaming) |
| no `tau` reachable (signature, column names, source tokens) | **PASS** |
| `n_online` gate: longer online sharing a prefix must not move shared rows | **PASS** |
| deterministic replay | **PASS** |
| series-order independence | **PASS** |
| parallel 5-worker build == serial rebuild | **PASS** (18 random series) |
| **batch ↔ streaming bitwise parity**, `atol=0` | **PASS** — 15 synthetic families and 30 real store series × 2 modules |
| RFF basis SHA pinned | **PASS** |
| full unit suite | 51 + 38 tests pass |

Every column is a function of `hist` and `online[:t+1]` only. Bandwidth, kernel
mean and normalization constants are history-only and frozen before the first
online point. There is no cross-series state, no RNG at inference, and no
dependence on the online horizon.

**One implementation fact worth recording.** The joint projection is accumulated
as `d` explicit rank-1 updates rather than `Z @ W.T`, and the squared norm and
EWMA are one in-place recursion shared by both paths. This is load-bearing:
BLAS `gemm` regroups the length-`d` inner sum differently for one row than for
many and disagreed in the last ulp on **~25%** of entries (max 4.4e-16), which
would have made batch/stream parity unreachable. This repository has already had
a single-ulp feature defect move a shipped prediction (`m06_loc`), so the batch
module was rewritten rather than the parity tolerance relaxed. The change was
made after Stage 1 and is **numerically inert on the science**: every Stage-1
number is identical to six decimals before and after
(`STAGE1_SCREEN.preparity.json` vs `STAGE1_SCREEN.json`).

## 4. Stage 1 — information screen (canonical fold 0, anchor RT-1320)

Standalone TS-AUC of one LightGBM on the 28-column block alone:

| cut | RT-1320 | JOINT | MARGINAL control |
|---|---:|---:|---:|
| whole fold | 0.640281 | **0.542467** | 0.544396 |
| dominant cell | 0.680891 | **0.557594** | 0.567139 |
| mature vs never-break | 0.679726 | **0.561493** | 0.572684 |
| mature vs pre-break | 0.684141 | **0.546716** | 0.551667 |

Within-`t` rank rho vs RT-1320 (dominant cell): joint **0.1623**, marginal 0.1397.
Candidate/control rho (dominant cell): **0.3032**.

Conditional AUC on the same-`t` pairs RT-1320 **inverts** (dominant cell):
joint **0.4712**, marginal 0.4892 — both *below chance*.

Fixed-perturbation grid (`rank(RT-1320) + eps·(rank(block) − 0.5)`), dominant
cell, full preregistered grid, nothing selected after the fact:

| eps | joint net | joint ΔAUC | marginal net | marginal ΔAUC |
|---:|---:|---:|---:|---:|
| 0.01 | +7 | +0.000107 | −3 | +0.000200 |
| 0.02 | +8 | +0.000197 | +3 | +0.000402 |
| 0.04 | +6 | +0.000305 | +19 | +0.000724 |

Mature-vs-pre net rate, joint: −0.00045 / −0.00060 / −0.00082.

**Adjudication (`STAGE1_ADJUDICATION.md`): no preregistered Stage-1 kill
condition fired unambiguously**, so the experiment proceeded to the binding
matched contract rather than being killed on an ambiguous reading of its own
screen. The discouraging evidence was recorded *before* Stage 2 so it could not
be reinterpreted afterwards: the joint candidate was already worse than its own
marginal control on every standalone cut, the marginal control already bought a
larger dominant-cell ΔAUC at every epsilon, and the joint block was already
below chance on exactly the pairs the incumbent gets wrong.

## 5. Stage 2 — matched ninth-member contract

`E0` = RT-1320 (8 members, untouched). `E1` = RT-1320 + ninth member on
{500 bank columns + `m19_lskm`}. `E2` = RT-1320 + ninth member on
{500 bank columns + `m19_lskd`}. Identical fold, sampled training rows,
row-sampling seed, model seed, hyperparameters, objective (`pairwise_t`),
boosting, fold-pure `SCDF_NSEEN` calibration and equal-weight ninth-member
contract. Nothing was tuned.

Learner: the repository's canonical production pairwise_t specialist
configuration (`wave2_streams.py` RT-123R / the RT-413 slot) —
`n_estimators=600, lr=0.05, num_leaves=63, min_data_in_leaf=300,
feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0,
max_bin=127`, `max_train_rows=700_000`, `sample_mode=uniform`, `seed=0`.

### Absolute scores and deltas

| arm | mean OOF TS-AUC | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---:|---:|---:|---:|---:|---:|
| **E0** RT-1320 | **0.629253722** | 0.640281 | 0.623244 | 0.636034 | 0.624462 | 0.622247 |
| **E1** + control | 0.628699976 | 0.640990 | 0.622363 | 0.636819 | 0.623230 | 0.620098 |
| **E2** + candidate | 0.629219268 | 0.641471 | 0.623520 | 0.636036 | 0.624192 | 0.620877 |

| endpoint | value |
|---|---:|
| **PRIMARY `E2 − E1`** | **+0.000519292** (4/5 folds positive) |
| per fold | +0.000481, +0.001158, **−0.000783**, +0.000962, +0.000779 |
| **SECONDARY `E2 − E0`** | **−0.000034454** |
| per fold | +0.001190, +0.000276, +0.000002, −0.000270, −0.001370 |
| **control lift `E1 − E0`** | **−0.000553746** |
| paired series bootstrap `E2 − E1` (400 reps) | mean +0.000525, **95% CI [−0.000116, +0.001069]**, fraction positive 0.9425 |

### Cell TS-AUC (all dev folds)

| cut | E0 | E1 | E2 | E2−E1 | **E2−E0** |
|---|---:|---:|---:|---:|---:|
| whole dev | 0.628998 | 0.628475 | 0.628984 | +0.000509 | **−0.000014** |
| dominant cell | 0.669369 | 0.668563 | 0.669109 | +0.000547 | **−0.000260** |
| mature vs never-break | 0.670353 | 0.669657 | 0.670190 | +0.000533 | **−0.000162** |
| mature vs pre-break | 0.666535 | 0.665408 | 0.665994 | +0.000586 | **−0.000541** |

### Pair flow (20 pairs per `t`, seed 0)

`E2` vs `E1`:

| cut | repairs | damage | net | net rate |
|---|---:|---:|---:|---:|
| whole dev | 234 | 238 | **−4** | −0.000201 |
| dominant cell | 174 | 143 | **+31** | +0.001950 |
| mature vs never-break | 154 | 179 | **−25** | −0.001572 |
| mature vs pre-break | 152 | 135 | **+17** | +0.001156 |

`E2` vs `E0` — the endpoint that actually decides deployment:

| cut | repairs | damage | net | net rate |
|---|---:|---:|---:|---:|
| whole dev | 227 | 210 | +17 | +0.000854 |
| dominant cell | 156 | 161 | **−5** | −0.000314 |
| mature vs never-break | 123 | 177 | **−54** | −0.003396 |
| mature vs pre-break | 118 | 150 | **−32** | −0.002177 |

### Member-level diagnostics

| quantity (dominant cell) | value |
|---|---:|
| within-`t` rho, candidate ninth member vs RT-1320 | 0.8346 |
| within-`t` rho, control ninth member vs RT-1320 | 0.8359 |
| within-`t` rho, candidate vs control member | 0.8492 |
| candidate member standalone mean TS-AUC | 0.614838 |
| control member standalone mean TS-AUC | 0.611858 |

## 6. Gate adjudication — what failed

**Fold-0 authorization gate (§6.1).** `E2 − E1 ≥ +0.0008` on fold 0:
observed **+0.000481**. **FAILED on its first binding condition.**

*(Recorded honestly: this gate could not save compute. The fold-pure
`SCDF_NSEEN` calibration derives fold 0's map from the member's out-of-fold
predictions on folds 1–4, so evaluating fold 0 at all requires all five fits.
The gate was still applied first and is still binding — it is reported as the
authorization failure it is, not overridden by the five-fold mean.)*

**Five-fold promotion gate (§6.2).**

| requirement | observed | verdict |
|---|---|---|
| `E2 − E1 ≥ +0.0011` | +0.000519 | **FAIL** (under half the floor) |
| `E2 − E0 > 0` | −0.000034 | **FAIL** |
| ≥ 4/5 folds positive on `E2 − E1` | 4/5 | pass |
| dominant-cell repair − damage > 0 | +31 | pass |
| mature-vs-pre pair flow > 0 | +17 | pass |
| mature-vs-never nonnegative | −25 | **FAIL** |
| paired bootstrap supports the improvement | 95% CI [−0.000116, +0.001069] contains zero, upper bound below the floor | **FAIL** |

Four binding requirements fail, including both the primary endpoint and the
deployment endpoint. **The alternate-partition leg was not run**: §6.2 makes it
conditional on the canonical result reaching threshold, and it did not.

## 7. The decisive fact: the control degrades the champion

`E1 − E0 = −0.000554`. **Adding an ordinary matched ninth LightGBM member makes
RT-1320 worse.** The nominally positive `E2 − E1 = +0.00052` is therefore almost
entirely *"the candidate degrades the champion less than the control does"*, not
*"the candidate adds information"*. `E2 − E0 = −0.000034` states the same thing
directly: **adding the joint lag-space kernel member to RT-1320 adds nothing, and
is very slightly negative on every cut measured.**

This is precisely the inflation mode `PROTOCOL_CHAMPION_2026.md` names — "never
interpret E2-E1 without inspecting E0, E1 and E2 separately; RT-1264/CSA-04
demonstrated that a degrading E1 control can manufacture an inflated marginal
headline" — and the brief's failure mode 1, "beating a weak control". Quoting
`+0.00052 at 4/5 folds` as a near-miss would be exactly that error. It is not
reported as promising, because it is not.

There is a second, independent reading of `E1 − E0 < 0` worth keeping: at eight
members RT-1320 appears to be **saturated with respect to ordinary additive
LightGBM members**, consistent with the bank-saturation finding (participation
ratio 2.730 over 72 fitted OOF vectors).

## 8. Failure classification

Against the preregistered interpretation table this is **CASE C with a CASE B
component**:

* **CASE C** — the joint block does beat its marginal control in raw pair flow
  (`E2 − E1` dominant-cell net +31; `E2 − E1` positive on 4/5 folds; candidate
  member standalone 0.6148 vs control 0.6119) and this **does not convert into
  any ensemble marginal against the champion** (`E2 − E0 ≈ 0`). Whatever joint
  lag-distribution information exists is already recoverable by the existing
  500-column bank and learner.
* **CASE B** — the block itself is weak (standalone 0.5425 vs the incumbent's
  0.6403) and decorrelated (rho 0.16), and scores **below chance (0.4712)** on
  exactly the pairs RT-1320 gets wrong. It is not aimed at the incumbent's
  errors. `RT-1201` already established at rho 0.38 that decorrelation alone
  converts to damage rather than repair; RT-1321 is another point on that curve.

It is **not** CASE A: rho is 0.16, not 0.89, so this is not the `RT-1215`
redundancy signature. The joint kernel genuinely sees something the bank does
not — it is just not something that repairs pairs.

## 9. Cost and runtime

| quantity | measured |
|---|---|
| batch feature build, both blocks, 10,000 series / 5,036,517 rows | **200 s** (5 workers) |
| batch cost per online point (both blocks) | ~40 µs |
| **streaming cost per observation** | **211 µs** (`m19_lskd`), 225 µs (`m19_lskm`) |
| streaming state per series | **5.0 kB** of live state + 16 kB frozen residual ECDF reference = 20.6 kB, measured; independent of `t`, nothing grows with the prefix |
| ninth-member training | 113 s per fold, 700k rows × 528 columns |
| total experiment compute | ~1.5 h wall clock |

Projected deployment impact had it promoted: +211 µs on the champion's
2.28 ms/point is +9.3%, i.e. ~4.0 h per 10,000 series against the 15 h ceiling —
deployable, and the per-step cost is dominated by Python overhead on tiny arrays
and would have been reduced by fusing the step into a single numba kernel. Moot:
**no production fit, no artifact and no submission was built.**

## 10. Scientific conclusion

**Did nonlinear lag-space distribution information exist beyond the current
500-feature bank? Yes — but not usefully, and not where the loss is.**

The joint kernel is genuinely distinct: within-`t` rho of only 0.16 against
RT-1320, and it measurably beats its own coordinate-separable control both as a
ninth member (standalone 0.6148 vs 0.6119) and in dominant-cell pair flow
(+31 net). So the *mechanism distinction* the experiment was designed to test —
joint lag law versus nonlinear marginal — is real and detectable.

It is also worthless for this champion. The block's standalone alpha is 0.5425
against an incumbent at 0.6403; its conditional AUC on the incumbent's own
inverted pairs is 0.4712, *below chance*; and adding it to RT-1320 as a matched
ninth member moves the ensemble by **−0.000034**. The information is real,
small, mis-aimed, and already recoverable by the existing bank.

Avenue **G5** is now closed. Together with `RT-1215` (linear delay-subspace,
KILL, rho 0.886) this closes the delay-embedding lane from both ends: the linear
version failed by redundancy, and the explicitly nonlinear characteristic-kernel
version fails by having decorrelated information that does not repair pairs.
`NEGATIVE_RESULTS_INDEX.md`'s note that "kernel PCA and other explicitly
nonlinear variants" were not covered by `RT-1215` is answered: they are covered
now, and the answer is no.

The two-sided squeeze the brief describes held once more. RT-1321 is the fourth
strongly decorrelated channel (rho 0.38, 0.585, 0.16 …) to produce no marginal
value, and the third experiment in this programme whose apparent gain was an
artifact of a control that degrades the incumbent.

## 11. Artifacts

* `RT1321_PREREG.md` — preregistration, frozen before any label.
* `STAGE1_SCREEN.json`, `STAGE1_SCREEN.preparity.json`, `STAGE1_ADJUDICATION.md`.
* `CONTRACT.json` — E0/E1/E2, per-fold deltas, pair flow, paired bootstrap.
* `MEMBER_*.json`, `MEMBER_DIAGNOSTICS.json`.
* `src/sbr/features/m19_lskd.py`, `src/sbr/stream/s_m19_lskd.py`.
* `tests/test_m19_lskd.py`, `tests/test_stream_parity_m19_lskd.py`.
* `research/scripts/rt1321_lskd.py`.

The implementation is committed and reproducible: the RFF basis is SHA-pinned,
the store and folds are the canonical ones, and both blocks rebuild bitwise from
the recorded code.
