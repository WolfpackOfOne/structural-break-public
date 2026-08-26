# CRF-01 EXECUTION PREREGISTRATION — NNCSR, NULL-NORMALIZED CAUSAL SEQUENCE RANKER

**STATUS AT COMMIT TIME: `PREREGISTERED_NOT_EXECUTED`.**

No CRF-01 training path exists yet. No `RT-1234`/`RT-1235`/`RT-1236` score, OOF
vector, TS-AUC, `marginal_vs_clone` or pair-flow number exists. `research/RESULTS.csv`
is byte-identical to its program-design state, sha256
`5b34c564e69f502c4a54d4ba1b702b400893358073e1897cd82453c83215512c` (246 lines,
245 experiment rows), verified immediately before this file was written.

This file freezes **every remaining implementation detail** of CRF-01 before any
CRF-01 number can be produced or read. The scientific contract is
`CRF_PROGRAM_PREREG.md` §0 and §1, which this file does **not** modify, weaken or
reinterpret. Where the two disagree in wording, `CRF_PROGRAM_PREREG.md` governs
the science and this file governs only the mechanical detail it left open.

Program preregistration: `CRF_PROGRAM_PREREG.md` at `85d121f`
("Preregister causal representation frontier program").
Branch: `research/causal-representation-frontier-2026`.
Base commit for this execution: `85d121f`.

---

## 1. QUESTION

Does a causal sequence encoder that reads a small **null-normalised** channel set
and is trained under a **same-`t` pairwise ranking** objective produce a ranking
direction that the frozen 500-column bank and the RT-600 seven-stream ensemble do
not already contain?

This is the last empty cell of the representation × objective factorial:

| | rowwise BCE objective | same-`t` ranking objective |
|---|---|---|
| **static 500-col bank** | `RT-300` etc. (incumbent) | `RT-111`, `RT-700/701/702`, `RT-123` |
| **learned causal sequence** | `RT-970`/`RT-971` | **CRF-01 — empty until now** |

`RT-970` is not this experiment: its channels were location/scale only
(`(x − med H)/IQR H` and running functions of it), it carried an
`elapsed = log1p(t)/7` rowwise shortcut channel, and its objective was BCE fixed
by preregistration. CRF-01 changes exactly two factors — the **channel set**
(null-normalised, `elapsed` removed) and the **objective** (same-`t` pairwise) —
and holds the architecture, optimiser, batching, seed and evaluation identical, so
`RT-970` (BCE + weak channels), `RT-1235` (BCE + CRF channels) and `RT-1234`
(pairwise + CRF channels) form an interpretable ladder.

---

## 2. EXPERIMENT IDs — ALLOCATED PROSPECTIVELY

| ID | arm | status |
|---|---|---|
| `RT-1234` | CRF-01 candidate: 8 null-normalised channels + RT-970 TCN shell + same-`t` pairwise logistic | **CANDIDATE** |
| `RT-1235` | C1 mandatory control: byte-identical arm with masked rowwise BCE. Isolates **objective**. | **MANDATORY CONTROL** |
| `RT-1236` | C2 conditional control: identical arm and objective, online channels temporally shuffled within each series. Isolates **temporal representation**. | **DECLARED, CONDITIONAL** — runs only if the candidate clears the §7 cheap abandon gate |

`RT-401` (matched exchangeable seed clone) and the seven RT-600 specialists
`RT-300`, `RT-410`, `RT-411`, `RT-412`, `RT-413`, `RT-414`, `RT-415` are reused as
frozen OOF vectors. They consume no new ID and are not retrained.

### 2.1 ID collision audit

`RT-1234`, `RT-1235` and `RT-1236` were verified unused before allocation:

* every `research/RESULTS.csv` reachable from every local and remote ref was
  scanned; the highest allocated id in the `RT-12xx` band is `RT-1233` (SS-04
  shuffled-target router control);
* `grep` over every tracked `*.md`, `*.csv`, `*.py`, `*.json` in this worktree
  returns no occurrence of `RT-1234`, `RT-1235` or `RT-1236`;
* `git log --all -S 'RT-1234'` returns no commit.

None of these ids is a recycled killed / void / abandoned / contaminated /
reserved id. `RT-900` remains retired. `RT-6xx` remains reserved for the shipped
production artifact and is not touched.

---

## 3. DATA, FOLDS, POPULATION

* Canonical full development store only: `sbr.pipeline.Data()` over `cache/store`
  (10,000 series, 5,036,517 online rows) with `research/folds/folds.parquet`.
  Folds are **never** regenerated.
* Development folds `0,1,2,3,4` (8,000 series, 4,032,524 rows).
* Fold `-1` (the 2,000-series lockbox, 1,003,993 rows) is **never** loaded for
  training, calibration, scoring, selection or diagnostics. Every emitted OOF
  vector is asserted to have **zero finite lockbox entries**.
* `X_test.reduced.parquet`, `y_test.reduced`, `folds_final10k.parquet`, the
  `RT-500..RT-506` all-10k vectors, and any production artifact are **not read**.
* No submission is built. `production/rt600` is not touched.
* The binding screen is **fold 0**. Folds 1–4 are trained only to complete the
  fold-pure OOF vector that the cross-fitted calibration of the fold-0 marginal
  requires (see §12.1), and become the confirmation surface only if §14 opens.

---

## 4. INPUT CHANNELS — FROZEN, EIGHT, NO OTHERS

All eight are computed per series from that series' own break-free history `H_i`
and `online[:t+1]`, and from nothing else. They are bitwise prefix-invariant by
construction: every per-series constant is a function of `H_i` alone, and every
online recursion is strictly left-to-right.

**Notation.** `n_H = len(H)`; `μ_H = mean(H)`; `σ_x,H = std(H, ddof=1)` floored at
`1e-9`; `z_H = (H − μ_H)/σ_x,H`; `z_t = (x_t − μ_H)/σ_x,H` for online `x_t`.

**AR(5) fit (frozen).** Yule–Walker on `z_H` only. Biased autocovariances
`r_k = (1/n_H) Σ_{i} z_H[i] z_H[i+k]`, `k = 0..5`. Solve the symmetric Toeplitz
system `T(r_0..r_4) φ = (r_1..r_5)` with a `1e-10 · r_0` ridge on the diagonal for
numerical safety. If `n_H < 60` (= `10·p + 10`, the guard `sbr.transforms._fit_ar`
already uses) then `φ = 0`. This is Yule–Walker as `CRF_PROGRAM_PREREG.md` §1.4
specifies and is **deliberately not** `sbr.transforms._fit_ar`, which is OLS; the
difference is recorded here rather than silently resolved.

**History innovations.** `e_H = z_H[5:] − Σ_k φ_k z_H[5−k−1 : n_H−k−1]`;
`σ_H = std(e_H, ddof=1)` floored at `1e-9`; `inn_H = e_H / σ_H` (unit variance
under the null by construction).

**Derived history constants**, all from `inn_H`:
* `mad_H = 1.4826 · median(|inn_H − median(inn_H)|)`, floored at `1e-9` — the
  scaled-MAD convention `sbr.transforms.HistParams.mad` already uses.
* `q90_H = quantile(|inn_H|, 0.90)`.

**256-knot ECDF construction (frozen, used three times).** For a sample `v` of
length `n`: sort ascending; take `idx = round(linspace(0, n−1, 256))`; knot
abscissae `v_sorted[idx]`, knot probabilities `(idx + 0.5)/n` — the `(r − 0.5)/n`
plotting position. Duplicate abscissae are collapsed to their **maximum**
probability so the knot sequence is strictly increasing. Evaluation is
`np.interp`, i.e. linear interpolation, constant outside the knot range. If
`n < 256` all `n` order statistics are used. Three instances:
`F̂_H` on `H`; `Ĝ_H` on `inn_H`; `Ĝ^abs_H` on `|inn_H|`.

**Online innovation.** With `warm` = the last 5 entries of `z_H`,
`inn_t = (z_t − Σ_{k=1..5} φ_k z_{t−k}) / σ_H`, where lag sources for `t < 5` come
from `warm` — the historical tail, exactly the convention
`sbr.transforms.ar_filter_causal` implements. This is what makes rows `t < 5`
legal and prefix-invariant.

| # | name | definition | clip |
|---|---|---|---|
| 1 | `pit` | `Φ⁻¹(F̂_H(x_t))` | `±4` |
| 2 | `inn` | as above | none (unbounded by design; see §4.1) |
| 3 | `inn_pit` | `Φ⁻¹(Ĝ_H(inn_t))` | `±4` |
| 4 | `abs_inn_pit` | `|inn_pit|` | inherits `±4` → `[0,4]` |
| 5 | `vol_norm` | `inn_t / max(E_{t−1}, 0.25·mad_H)` | `±8` |
| 6 | `surp` | `−log(1 − Ĝ^abs_H(|inn_t|) + 1/(n_H+1))` | `[0, 12]` |
| 7 | `exceed` | `sigmoid(4·(|inn_t| − q90_H)/mad_H)` | none (already in `(0,1)`) |
| 8 | `lag1_pit` | `pit_t · pit_{t−1}`, with `pit_{−1} = 0` | `±16` |

**Channel 5's EWMA is strictly lagged and history-seeded.**
`a = 1 − 0.5^{1/32}` (half-life 32). `E_{−1} = mad_H`. For each `t` in order:
the denominator uses `E_{t−1}`; **then** `E_t = E_{t−1} + a·(|inn_t| − E_{t−1})`.
`inn_t` never influences its own denominator.

**Post-construction rule.** Non-finite entries are set to `0.0` after all clips —
the `wave6_neural_lib.causal_channels` convention. Output dtype `float32`, shape
`(n_online, 8)`.

### 4.1 No global standardiser is fitted

Every clip bound above is a **constant written in this preregistration**, and every
per-series constant is a function of that series' own history. Therefore **no
global channel scale, standardiser or clip is estimated from any row of any fold.**
`CRF_PROGRAM_PREREG.md` §0.5's requirement that a global standardiser be fitted on
training-fold rows only is satisfied vacuously and this fact is **asserted in
code**, not assumed (§10 gate P3). Channel 2 `inn` is intentionally left unclipped:
it is already `σ_H`-normalised, and clipping it would remove the excursion-magnitude
signal the hypothesis is about.

### 4.2 Forbidden and unreachable

Not computed, not stored, not reachable from the CRF-01 input path: raw `x` other
than through the eight channels above; any of the 500 bank columns; `RT600` or any
specialist / seed-clone score; any first- or second-sweep candidate score; true
`τ`; post-break age; `n_online`; the final online length; any boundary-conditioned
availability or missingness; `elapsed`, `t`, or any monotone function of `t`; any
cross-sectional quantity computed across series at a shared `t`.

**There is no `elapsed` channel and no `t` channel, in the candidate or in either
control.** This is the single most important difference from `RT-970` on the input
axis and it is applied symmetrically so the ladder stays matched.

---

## 5. ARCHITECTURE — RT-970 SHELL, UNCHANGED

Built by `wave6_neural_lib.CausalTCN.build(hidden=32, dropout=0.1, seed=…,
device=cpu, n_in=8)` — the `RT-970` shell called verbatim, with the input channel
count as the only argument that differs.

* causal dilated TCN, kernel 3, dilations `1,2,4,8,16,32`, six residual blocks
* each block: two weight-normed **left-padded** causal `Conv1d` at that block's
  dilation, GELU, dropout 0.1, plus a `1×1` residual projection when the channel
  count changes
* per-timestep `1×1` linear head to one scalar
* no BatchNorm, no time-axis normalisation, no recurrence, no attention, no SSM
* **35,649 parameters** at `n_in = 8, hidden = 32` — measured, and matching
  `CRF_PROGRAM_PREREG.md` §1.5's "≈ 35.7k"
* CPU only; `torch.use_deterministic_algorithms(True)`; `torch.set_num_threads(6)`;
  `PYTHONHASHSEED=0`; `numpy` seeded; MPS/CUDA unused

### 5.1 Receptive field — a documented discrepancy, resolved in favour of the architecture

`CRF_PROGRAM_PREREG.md` §1.5 states receptive field **127** *and* "identical to
`RT-970`" *and* "≈ 35.7k parameters". These three are not simultaneously
satisfiable: `127 = 1 + 2·(3−1)·(1+2+4+8+16+32)` is the arithmetic for **one**
convolution per dilation, while the `RT-970` block contains **two**, which is also
what makes the parameter count 35,649 rather than ~23k.

Measured before any CRF-01 number existed, by perturbing a single input timestep in
`float64` and recording which outputs move (`t = 300`, `T = 700`):

```
span 300..552   receptive field = 253   nothing before t moves (causal)
```

**Resolution: the architecture is binding, the derived number is not.** CRF-01 uses
the `RT-970` shell verbatim — the instruction repeated three times in the program
preregistration — and its true receptive field is **253**, recorded here before any
score exists. Nothing about the design changes; only the description is corrected.
No layer, dilation or width is altered to chase 127, because that would be a
post-hoc architecture change to an experiment whose whole value is being a clean
two-factor ablation against `RT-970`.

---

## 6. TRAINING — FROZEN

Identical for `RT-1234`, `RT-1235` and `RT-1236` except where a row says otherwise.

| item | value |
|---|---|
| optimiser | `AdamW` |
| learning rate | `3e-3` |
| weight decay | `1e-2` |
| schedule | `CosineAnnealingLR`, `T_max = EPOCHS · ceil(n_train_series / 32)`, stepped every optimiser step |
| epochs | `20`, no early stopping, no patience, no best-epoch selection |
| gradient clipping | `clip_grad_norm_(…, 1.0)` |
| batch | **32 series**, length-bucketed, chunk order shuffled |
| batch construction | `wave6_n2_tcn._batches(tr_series, n_on, rng, 32)` verbatim |
| batch rng | `np.random.default_rng(seed·1000 + fold)` |
| init / torch seed | `torch.manual_seed(seed·1000 + fold)` before `build`, exactly as `RT-970` |
| seed | `0` (so fold `f` uses seed `f`) |
| dtype | `float32` |
| threads | 6 |
| padding | right zero-padding to the batch's max length, with a `(B,T)` validity mask; padded positions contribute zero loss and are never scored |

Because the batch rng, the init seed, the epoch count and the step count are
identical functions of the fold index, the candidate and C1 see **the same series
in the same batches in the same order with the same initial weights**. The only
difference is the loss.

### 6.1 Candidate objective (`RT-1234`) — same-`t` pairwise logistic

* Groups are online index `t`. Pairs are drawn **only within a group** and **only
  from training-fold series** (the batch contains training series only).
* For each optimiser step, inside the packed batch: for each `t` present in the
  batch, let `P_t` be the valid rows with `y = 1` and `N_t` the valid rows with
  `y = 0`. A timestep contributes only if `|P_t| ≥ 1` **and** `|N_t| ≥ 8`
  (the minimum group occupancy of `CRF_PROGRAM_PREREG.md` §1.6).
* For each positive in a contributing `t`, draw `m_neg = 8` negatives **uniformly
  with replacement** from `N_t` — the `sbr.pipeline._make_pairwise_t` convention,
  which resamples on every call.
* Loss `= mean over all sampled pairs of softplus(−(s_pos − s_neg))`.
* **Uniform pair weighting.** No `n_pos(t)·n_neg(t)` metric-shaped weighting
  (`RT-700` closed that lane at −0.00147), no margin, no tie term, no age or `τ`
  weighting, no hard-negative mining, no RT600 residual target, no custom weights.
* Pair rng: `np.random.default_rng(2026082600 + fold)`, advanced once per
  optimiser step. It is separate from the batch rng, so batch order is bitwise
  identical to C1's.
* Group occupancy (contributing timesteps per epoch, pairs per epoch) is **recorded
  and reported**, since batch size 32 makes it the binding constraint.

### 6.2 C1 objective (`RT-1235`) — masked rowwise BCE

`binary_cross_entropy_with_logits(logits, y, reduction="none")`, then
`(l · mask).sum() / mask.sum()` — uniform over rows, byte-identical to the
`RT-970` loss expression. Nothing else differs from the candidate: same channels,
same architecture, same optimiser, same schedule, same epochs, same batches, same
seeds, same emission, same calibration, same evaluation.

### 6.3 C2 construction (`RT-1236`) — conditional, frozen now

Per series `i` with online length `n`, draw
`perm = np.random.default_rng(2026082601 + i).permutation(n)` and replace the
channel matrix `C` by `C[perm]`, i.e. **the same permutation applied to all eight
channels**. The label stays at its original `t`. This destroys temporal order while
preserving every channel's per-series marginal distribution exactly. The shuffle is
applied to training **and** validation series so the arm is internally consistent.
Objective, architecture, optimiser and seeds are the candidate's.

C2 is a diagnostic control and is **not causal at inference** by construction; it is
never a deployment candidate and never enters an ensemble that is claimed
deployable. It runs **only** if the candidate clears §7, and only as described here.

---

## 7. CHEAP ABANDON GATE — EVALUATED FIRST

`CRF_PROGRAM_PREREG.md` §0.2, unmodified and unrelaxable:

> abandon if fold-0 **standalone whole-fold TS-AUC < 0.600** **AND**
> within-`t` ρ vs the RT600 blend **≤ 0.60**.

Frozen definitions:

* **standalone whole-fold TS-AUC** = `sbr.metric.ts_auc_flat` on the **raw** score
  over all fold-0 validation rows. No calibration (§0.6: TS-AUC is invariant to any
  monotone transform applied identically inside a timestep).
* **within-`t` ρ** = `novel_streams.harness.diagnostic_pack(...)["within_t_rank_corr_rt600"]`
  on fold 0 — Pearson correlation of `within_t_rank(candidate)` against
  `within_t_rank(rt600_blend)` over **dominant-cell** fold-0 rows
  (`t ≥ 200` and (`y = 0` or post-break age `≥ 100`)). This is the canonical
  quantity the `+0.983` audit correlation and the `+0.0030` contour were fitted on;
  it is not redefined here.
* **RT600 blend** = `novel_streams.harness.rt600_blend(c)` — the equal-weight
  cross-fitted `SCDF_NSEEN` blend of the seven specialists.

If the gate fires: **stop.** No further folds for the candidate, no C2, no ensemble
integration for the candidate. C1 fold 0, if already trained, is preserved and
reported. The negative is filed in full (§15) and the program continues to CRF-02,
which `CRF_PROGRAM_PREREG.md` §2 declares unconditional and independent.

---

## 8. EXECUTION STAGING — FROZEN

| stage | what runs | what may be read |
|---|---|---|
| **S0** | purity + causality preflight (§10, §11) | test pass/fail only, **no TS-AUC of any arm** |
| — | **`Validate CRF-01 purity and causality preflight` committed and pushed** | — |
| **S1** | candidate fold-0 model, C1 fold-0 model | candidate fold-0 standalone TS-AUC, dominant-cell TS-AUC, ρ → **§7 gate** |
| **S2** | only if §7 does not fire: candidate + C1 folds 1–4; C2 all five folds | full fold-0 evaluation (§12), gates (§13) |
| **S3** | only if §13 passes: **no new training** | five-fold confirmation from the same OOF vectors (§14) |

Stage S2 trains folds 1–4 because the cross-fitted calibration that produces the
**fold-0** marginal fits fold 0's map on folds 1–4, so the candidate must carry
fold-pure scores there. This is the same construction every stream in this
repository uses and the same one `SS-02`…`SS-04` used for their fold-0 screens. It
is stated here so that "fold-0 screen" is not mistaken for "one model".

---

## 9. IMPLEMENTATION CONTRACT — TWO PROCESSES

`torch` and `lightgbm` segfault when sharing a process on macOS/arm64 (Wave-6
finding). They are kept in **separate processes**. `KMP_DUPLICATE_LIB_OK` and every
equivalent hack are **forbidden**.

```
research/scripts/crf01_nncsr.py      TORCH PROCESS.  builds the 8 channels,
                                     trains a fold model, emits frozen raw OOF
                                     score arrays + metadata JSON.
                                     Imports no lightgbm, performs no ensemble
                                     integration, computes no marginal.

research/scripts/crf01_integrate.py  NO-TORCH PROCESS.  loads the frozen arrays,
                                     applies the canonical cross-fitted SCDF,
                                     computes E0/E1/E2, TS-AUC, pair flow, rho,
                                     writes the experiment report.
                                     Imports no torch.
```

Interpreters, verified present before this file was committed:

```
TORCH   /path/to/workspace/structural-break-wave8/.venv/bin/python
        python 3.11.6 · torch 2.13.0 · numpy 2.4.6 · scipy 1.17.1 · lightgbm 4.7.0 (never imported)
NOTORCH /path/to/workspace/structural-break-new-avenues-pilots/.venv/bin/python
        python 3.11.6 · numpy 2.4.6 · lightgbm 4.7.0 · torch ABSENT
HOST    macOS arm64 (Darwin 25.5.0), 10 cores, 16 GB
SBR_ROOT = this worktree; cache/store and cache/features are symlinks into
           structural-break-claude-wave3
```

Channels are cached once to `cache/crf01/channels.npy` (`.gitignore`d, as
`cache/` already is) keyed by row count and channel-name list, so the candidate and
both controls read **the same bytes**.

### 9.1 Streaming form

`crf01_nncsr.py` exposes the `novel_streams.harness.StreamingMechanism` interface —
`fit_history(H)` (one pass: ECDF knots, `φ`, `σ_H`, residual-ECDF knots, `q90_H`,
`mad_H`, EWMA seed), `initialize_state()`, `update(x_t)`, `emit_score()` — and
reports measured **state bytes per series** and **full evaluation wall-clock**.
Target feasibility: 10,000 independent series at online length ≤ 1,000.

---

## 10. FOLD-PURITY SENTINELS — MUST PASS BEFORE ANY SCORE

Implemented as real tests in `tests/test_crf01_causality.py`, in the style of
`tests/test_wave8_causality.py`. Prose is not sufficient.

| gate | assertion |
|---|---|
| **P1 set arithmetic** | for every outer fold `f`, the fitted series set is exactly `FOLDS \ {f}` and its intersection with `{f}` is empty |
| **P2 positive control** | the *contaminated* scheme (fit on all five folds) is reproduced and shown to intersect `{f}` for every `f` — proving the sentinel detects the defect rather than passing vacuously |
| **P3 live fitting path** | called through the real training entry point with a tiny budget: the model that scores fold `f` never saw a fold-`f` series, asserted inside the function, and the emitted vector is NaN everywhere outside fold `f` |
| **P4 per-series constants** | every history constant (`μ_H`, `σ_x,H`, `φ`, `σ_H`, ECDF knots, `mad_H`, `q90_H`, EWMA seed) is bitwise unchanged when every *other* series in the store is replaced by noise — i.e. it depends on that series' history alone |
| **P5 no global standardiser** | the channel builder's output for a series is bitwise unchanged under any change to the fold assignment, and the builder is asserted to hold no fitted global state |
| **P6 validation rows do not train** | perturbing a validation series' online values leaves every training-fold gradient bitwise unchanged |

---

## 11. CAUSALITY SENTINELS — MUST PASS BEFORE ANY SCORE

| gate | assertion |
|---|---|
| **C1 prefix invariance** | `harness.verify` / `check_prefix_invariance` style, **atol = 0.0**, ≥ 8 real store series of different lengths, cuts `(3, 10, 37, 111)` — channels rebuilt on a truncated online segment reproduce the surviving rows **bitwise**, NaN pattern included |
| **C2 forbidden columns** | `wave8_common.assert_no_forbidden_columns` over all eight channel names and every emitted metadata key |
| **C3 truncation** | rebuild channels **and re-run the trained encoder** on a truncated online segment; surviving score rows reproduce to `≤ 1e-8` |
| **C4 batch composition** | a series' emitted score is unchanged (`≤ 1e-8`) by which other series share its minibatch — Wave-6 Gate 5, re-asserted for this shell and channel set |
| **C5 deterministic replay** | re-running a fold from scratch reproduces the state-dict sha256 and the score vector bitwise |
| **C6 lockbox** | `finite(oof[fold == −1]).sum() == 0` for every emitted vector |
| **C7 no final length** | no code path reads `n_online`, the final row, or any length-dependent normalisation; asserted by construction review **and** by C1 (a length-dependent constant cannot survive bitwise prefix invariance) |
| **C8 no cross-sectional inference** | the scoring function is exercised on a **single series in isolation** and reproduces its in-batch score to `≤ 1e-8` — the model needs no other series at inference (`CRF_PROGRAM_PREREG.md` label rules; same-`t` grouping is legal for **training** only) |

**Any failure stops the experiment. No score is read. Correctness is fixed first,
and if a fix changes anything frozen above it becomes a new id and a new execution
preregistration.**

---

## 12. EVALUATION — FROZEN

### 12.1 Calibration and ensemble integration

Canonical and unmodified: `wave5_lib.Ctx.crossfit_blend` with
`CANON_CAL = wave4_cal.SCDF_NSEEN` — `SmoothTimeCDFCal`, `kind="scdf"`,
`time_coord="log_n_seen"`, 12 log-spaced anchors, 256-point quantile grids,
`min_n = 400`, fold `k`'s map fitted on folds `≠ k`. Applied **identically** to the
candidate and to every control. No new calibration layer is invented; if a
candidate appears to need one, that is a finding to report, not a change to make.

```
E0 = seven RT-600 specialists,  equal weight, cross-fitted SCDF
E1 = E0 + RT-401 seed clone,    equal weight over eight
E2 = E0 + candidate,            equal weight over eight
marginal_vs_clone = E2 − E1        (fold 0 rows)
```

Computed by `wave8_common.ensemble_marginal`, unchanged. No weight search, no
stacking, no subset selection — all three are closed lanes.

Before any arm is evaluated, `rt600_sentinel`-equivalent expected values are
re-checked and must agree within `0.005`:
`mean 0.625811`, `pooled 0.625627`, `dominant_cell 0.664277`,
`fold0_e0 0.638276`, `fold0_e1_seedclone 0.638586`.

### 12.2 Mandatory reporting — every arm, canonical fold-0 sample

Pair flow uses `second_sweep_ss01.pair_flow_pack` with the **same deterministic
sample for every arm** — `PAIR_EVAL_SEED` unchanged, `EVAL_PAIRS_PER_T = 64`,
dominant cell `t ≥ 200` and age `≥ 100`. **The sample is never re-drawn per arm.**

Reported for the candidate and for every control that runs:
whole repairs / damage / net · dominant repairs / damage / net ·
mature-vs-never net · mature-vs-prebreak net · unique repair coverage ·
damage rate on RT600-correct pairs · pre-break damage rate on RT600-correct pairs ·
within-`t` ρ vs RT600 · standalone whole-fold TS-AUC · standalone dominant-cell
TS-AUC · `E0` / `E1` / `E2` / `marginal_vs_clone` · training and evaluation runtime ·
streaming state bytes per series.

ρ is **reported, never optimised for.** Low ρ is not a credential: `RT-970` reached
ρ 0.21 for `+0.0001`. The target is good standalone signal **and** positive pair
flow **and** low enough redundancy **and** broad coverage.

---

## 13. BINDING FOLD-0 SCREEN GATES

All applicable gates must pass. Thresholds are fixed here and may not be changed
after any CRF-01 number is seen.

| gate | threshold |
|---|---|
| abandon (checked first, §7) | standalone fold-0 whole TS-AUC `≥ 0.600` **or** ρ `> 0.60` |
| **primary screen** | fold-0 `marginal_vs_clone` **`≥ +0.0015`** |
| **objective isolation** | candidate `marginal_vs_clone` − C1 `marginal_vs_clone` **`≥ +0.0010`** |
| **pair flow** | dominant-cell net **`> 0`** and mature-vs-never net **`> 0`** |
| **damage cap** | damage rate on RT600-correct **pre-break** pairs **`≤ 0.0150`** |

Bands (`CRF_PROGRAM_PREREG.md` §0.1): KILL `< +0.0015`; WEAK `+0.0015 … +0.0030`;
SERIOUS `≥ +0.0030`; MAJOR `≥ +0.0050`.

**If any required gate fails, CRF-01 is KILL.** No tuning. No `CRF-01b`. No
architecture, channel, objective, seed or threshold change. The result stands in
`RESULTS.csv` and the program proceeds to CRF-02.

If CRF-01 fails the primary screen but **C1 fails by more**, that is a real finding
about the objective and is reported as such — it does not resurrect the candidate.

---

## 14. FIVE-FOLD CONFIRMATION — ONLY IF §13 PASSES

No architecture change, no hyperparameter change, no seed change, no re-roll.

SERIOUS requires **all** of:
* mean five-fold `marginal_vs_clone ≥ +0.0030`
* `≥ 4/5` folds positive
* positive dominant-cell pair flow
* C1 gap `≥ +0.0010`
* deployable runtime **measured**, not asserted

If SERIOUS: **broad execution stops.** CRF-02 is not started. CRF-01 is fully
documented and returned — a real representation breakthrough matters more than
completing the queue.

If fold 0 clears but five-fold confirmation does not: the result is preserved and
the program continues to CRF-02.

---

## 15. RESULT FILING

Regardless of verdict:

* `research/reports/causal_representation_frontier/crf01_nncsr.md` and `.json`
* one **appended** `research/RESULTS.csv` row per executed RT arm — never an edit,
  never a deletion, never a reused id
* `research/EXPERIMENT_ID_MAP.md` — allocation and result
* `research/RDOF_LEDGER.md` — degrees of freedom consumed
* `research/FAILED_EXPERIMENTS.md` — if killed, with the **exact** falsification,
  not "neural model failed"
* `research/STATUS.md` — only if the scientific state changes
* `python research/scripts/check_research_hygiene.py` after filing, with before /
  after row counts recorded
* GitHub Actions `research-hygiene` / `check-results-csv` verified on the actual
  result-commit run via `gh run list` + `gh run watch --exit-status`

---

## 16. COMPUTE BUDGET

`RT-970` measured 3,631.9 s for five folds at hidden 32 / 10 channels / 20 epochs
(≈ 726 s per fold). CRF-01 expectations:

| stage | expected |
|---|---|
| channel build (10,000 series) | ≤ 0.5 h |
| S1: candidate fold 0 + C1 fold 0 | ≈ 0.5 h |
| S2: candidate + C1 folds 1–4 | ≈ 1.6 h |
| S2: C2 five folds, if opened | ≈ 1.1 h |

Program screening budget `≤ 15 h` before any serious confirmation. No Optuna, no
architecture search, no width sweep, no learning-rate sweep, no seed fishing, no
re-rolling a diverged fold. If a fold exceeds a 3 h budget it is reported as
infeasible rather than shrunk — the `RT-970` rule.

---

## 17. SAFETY DECLARATIONS

No lockbox read. No test data. No `X_test.reduced`. No Crunch submission. No
`production/rt600` edit. No deployment promotion. No merge into `research/current`
or `research/new-avenues-pilots-2026`. No force push. No rebase of pushed history.
No recycled RT id. No post-score tuning. Five-fold confirmation is still
development research; production evaluation comes later and only after a serious
candidate.

---

## 18. PREFLIGHT COMPLETED

*This section is empty at Stage A by design. It is filled in the Stage B commit
`Validate CRF-01 purity and causality preflight` with the actual test commands and
their actual output, and that commit is pushed **before** the first CRF-01 score
exists.*
