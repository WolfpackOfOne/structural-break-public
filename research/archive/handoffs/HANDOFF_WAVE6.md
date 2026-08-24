# HANDOFF — WAVE 5 → WAVE 6

**2026 ADIA Lab / CrunchDAO Structural Break Challenge (Real-Time Edition)**

Written 2026-08-21 at the end of wave 5, on branch `research/wave5-alpha`
(parent `research/wave3-integration` @ `17bb5df`).

**Read this file completely before running anything.** It is written for an
agent with no memory of this session. Where it and the repository disagree, the
repository wins — every number below is on disk and the path is given.

---

## PART 0 — THE THIRTY-SECOND VERSION

* The competition score is **0.6268** (Crunch public, LB-001). It has not moved.
* The deployed system is **RT-600**: seven LightGBM boosters over one shared
  500-column causal streaming feature engine, equal-weight mean of
  time-conditionally calibrated scores. Development OOF **0.62581**.
* **Wave 5 promoted nothing.** It ran nine pre-registered experiments, built
  three new causal feature modules, and rejected all of them against a control
  that most of them beat on the first framing tried.
* The single most useful output is not a feature. It is a **quantitative map of
  how little headroom this ensemble leaves**, measured four independent ways,
  plus three corrections to premises the project had been carrying.
* **Do not submit anything.** No wave-5 candidate cleared the promotion bar.

---

## PART 1 — GROUND TRUTH, AND WHAT YOU MUST NOT TOUCH

### 1.1 The numbers that anchor everything

| | value | where |
|---|---|---|
| LB-001, Crunch public leaderboard | **0.6268** | external, from the user |
| RT-600 development OOF, canonical partition | **0.62581** | `research/reports/wave5_e1_mixture.json` |
| single-model control `RT-300` | 0.61605 | `research/RESULTS.csv` |
| seven seed clones `RT-421` | 0.62164 | same |
| **best internal number ever produced** | **0.62629** | `S + RT-751`, and it fails its own control |
| leaderboard snapshot | #1 65.10 · #10 64.10 · #25 63.60 · #50 62.91 | user-supplied |
| approximate standing | ~rank 59 | |

Internal → external transfer was **flat to slightly positive** (0.62581 →
0.6268). The validation framework is sound. Do not rebuild it.

### 1.2 Immutable refs — verified unmoved at the end of wave 5

| ref | SHA | rule |
|---|---|---|
| `research/wave3-integration` | `17bb5df` | the trunk. Do not force-push, do not rewrite |
| `claude/rt600-baseline-submission` | `9aaa9b0` | the shipped artifact's branch |
| `codex/reproduce-2025-public-solution` | `422e4b2` | READ ONLY. Never merge |
| `codex/oracle-information-frontier-2026` | `5a3b8a0` | READ ONLY. Never merge |

`submissions/C_ensemble_deployable.py`, `models/final10k_ensemble` and
`research/FINAL_REPRODUCIBILITY_MANIFEST.json` are frozen. RT-600 is the
fallback champion and the thing you lose if you are careless.

### 1.3 The forbidden list (this is not boilerplate — one of these was violated)

* **`n_online`** or the final online length, in any form, for any purpose —
  a grid, a null length, a buffer size, a branch. A feature built from
  `(t+1)/n_online` scores ~0.6295 by itself, *better than the legitimate
  champion*, and would invalidate the entry. **The first `m12_rdep` sized its
  expanding nulls by `n_online` and `check_prefix_invariance` caught it.**
* future data · cross-series live state · within-timestep rank oracles in a
  production candidate · validation-fold leakage · global feature selection
  using validation labels · leaderboard-directed fitting.
* `research/folds/folds_final10k.parquet`, fold `-1` (the spent lockbox),
  `RT-500`..`RT-506` (all-10k OOF vectors), `X_test.reduced`, `y_test.reduced`
  — **none of these is a validation surface.** `RT-5xx` exists only for
  post-freeze calibration.

---

## PART 2 — HOW TO SET UP AND RUN ANYTHING

### 2.1 Worktrees on this machine

```
structural-break                    main working dir, branch claude/structural-break-competition-entry-yjoj8r
structural-break-claude-wave3       RT-600's branch; OWNS THE 10 GB FEATURE CACHE and cache/store
structural-break-codex-wave3        codex/wave3-engineering
structural-break-oracle             codex/oracle-information-frontier-2026   (read only)
structural-break-reproduce-2025     codex/reproduce-2025-public-solution     (read only)
structural-break-wave5              research/wave5-alpha   <- wave 5 lives here
```

### 2.2 The environment

**Use this interpreter and nothing else.** It matches the architecture freeze
exactly: Python 3.11.6, numpy 2.4.6, pandas 3.0.5, scipy 1.17.1,
scikit-learn 1.9.0, lightgbm 4.7.0, numba 0.67.0.

```
PY="/path/to/workspace/structural-break/.venv/bin/python"
```

The system python (anaconda) has a numpy 1.x/2.x ABI conflict and will fail on
`import pandas`. If you see `_ARRAY_API not found`, you used the wrong python.

### 2.3 Setting up a new research worktree (the cache-sharing trick)

The feature cache is **10 GB** and the store is 134 MB. Never rebuild or copy
them. Symlink per-file so new modules are written locally and the shared cache
stays read-only:

```bash
cd "/path/to/workspace/structural-break"
git branch research/wave6-<name> research/wave5-alpha       # or wave3-integration
git worktree add "../structural-break-wave6" research/wave6-<name>
W6="/path/to/workspace/structural-break-wave6"
SRC="/path/to/workspace/structural-break-claude-wave3"
mkdir -p "$W6/cache/features" "$W6/research/oof" "$W6/research/reports"
ln -sfn "$SRC/cache/store" "$W6/cache/store"
for f in "$SRC"/cache/features/*; do ln -sf "$f" "$W6/cache/features/$(basename "$f")"; done
for f in "$SRC"/research/oof/*;   do ln -sf "$f" "$W6/research/oof/$(basename "$f")";   done
```

Then **also** symlink the wave-5 modules' caches if you want them
(`m10_persist`, `m11_focus`, `m12_rdep` live in
`structural-break-wave5/cache/features/`).

Every script needs `SBR_ROOT` set to the worktree root. The feature driver also
needs `PYTHONPATH`:

```bash
export SBR_ROOT="$W6"
"$PY" -u research/scripts/<script>.py                       # research scripts
SBR_ROOT="$W6" PYTHONPATH="$W6/src" SBR_FEATURES="$W6/cache/features" \
  SBR_STORE="$W6/cache/store" "$PY" -u -m sbr.features.driver --modules <mod> --workers 3
```

### 2.4 Machine limits — this bit down hard

10 cores, **16 GB RAM**, 10 GB swap. A CHAMP-protocol arm needs ~4.6 GB for its
training and validation matrices, and the feature memmaps are 10 GB so each
trainer's page-cache working set is several GB.

**Run at most TWO trainers concurrently.** With three, swap saturated
(9.9 GB of 10 GB used, 26M swapouts) and every arm slowed by roughly **5×**.
Use `sysctl -n vm.swapusage` to check.

A single ABL 5-fold arm is ~10 min alone, ~20 min with contention. A CHAMP arm
is ~20 min alone. A CHAMP arm with a Python custom objective is ~25–30 min.

---

## PART 3 — THE VALIDATION FRAMEWORK, AND THE ONE RULE THAT MATTERS

### 3.1 Protocols

Both are defined verbatim in `research/scripts/wave2_lib.py`. **Do not invent a
third.**

* **`CHAMP`** — 1,000,000 training rows, 5 canonical folds, `num_leaves 63`,
  `min_data_in_leaf 300`, `feature_fraction 0.5`, `bagging_fraction 0.7`,
  `n_estimators 600`, `learning_rate 0.05`, `lambda_l2 5.0`, `max_bin 127`,
  `num_threads 2`. This is the champion's own protocol.
* **`ABL`** — the same but 400,000 rows. For ablations and stability studies.
  **Every arm of a comparison must use the same protocol**; absolute levels sit
  below CHAMP and must never be quoted as a champion score.

**`num_threads=2` is part of the architecture freeze** — determinism depends
on it.

### 3.2 THE RULE

> **A candidate must beat a same-strength stream that contains no new
> information at all.**

Not "beats the champion". Not "the blend improved". Not "it decorrelates". The
control is a **seed clone** — the champion configuration with only the random
seed changed.

Three candidates have now died on this rule and it has never once been wrong:

| candidate | beat the champion? | beat a seed clone? |
|---|---|---|
| `m09_back` (wave 3) | yes, +0.00371 blend | **no, −0.00126** |
| 13-way union (wave 4, W4-E6) | — | **no, −0.00095** |
| `m10_persist` (wave 5, W5-E2) | yes, +0.00456 blend | **no, −0.00041** |

And **within-timestep rank correlation is not a diversity credential.** A seed
change decorrelates *more* than 51 columns of new statistics did (0.7846 vs
0.8219 for `m09_back`) and more than 76 columns did (0.7846 vs 0.7814 for
`m11_focus`).

### 3.3 The promotion bar (`research/WAVE5_PREREG.md` §4)

All four, or it is not promoted:

1. ≥ **+0.0030** TS-AUC over the strongest matched control;
2. positive on ≥ **4/5** canonical folds;
3. paired series bootstrap (200 reps, common random numbers) CI supportive;
4. alternate partitions positive or at minimum directionally stable.

**All four were needed in wave 5.** `m12_rdep` passed the first framing tried
and failed conditions 1 and 4. See Part 5.

### 3.4 Files that define the surface

```
research/folds/folds.parquet         PERMANENT canonical 5 dev folds + lockbox(-1). NEVER regenerate
research/folds/folds_alt{1,2,3}.parquet   alternate partitions, robustness only
research/RESULTS.csv                 the ledger. Append ONLY via sbr.pipeline.run
research/oof/<EXP>.npy               OOF prediction vectors
src/sbr/metric.py                    the official TS-AUC. Verified against sklearn
```

---

## PART 4 — WHAT WAVE 5 RAN, WITH EVERY NUMBER

Nine pre-registered experiments (`research/WAVE5_PREREG.md`, committed at
`5488644` **before** the first number existed) plus one added mid-wave with its
provenance stated (`W5-E11`, in `research/RDOF_LEDGER.md`).

### 4.1 Baselines — reproduced exactly, same session, same folds

| arm | id | TS-AUC | matches |
|---|---|---|---|
| A single | `RT-300` | 0.61605 | V4 §3 exactly |
| B seven seed clones | `RT-421` | 0.62164 | V4 §5 exactly |
| C seven specialists = **RT-600** | `RT-420` | 0.62581 | V4 §4 exactly |

Bagging **+0.00559**, specialisation **+0.00417**. Bootstrap of specialists −
seed clones: **+0.00409, CI [+0.00199, +0.00614], 200/200 positive** — W4-E1's
digits, reproduced independently.

### 4.2 Every experiment

| id | what | result | verdict |
|---|---|---|---|
| **W5-E1** | λ·S + (1−λ)·B mixture, λ ∈ {1.0, 0.9, 0.8, 0.7} | −0.00004 / −0.00017 / −0.00040, **monotone decreasing** | REJECTED |
| **W5-E2** | `m10_persist` — outlier-vs-bulk scale, designed from D2 | standalone +0.00202 (3/5); **−0.00041 vs seed clone** | REJECTED |
| **W5-E3** | hard-negative curriculum, nested fold-pure mining | `RT-711` −0.00701 (1/5); `RT-712` −0.01339 (0/5) vs `RT-710` | REJECTED |
| **W5-E4/5/6** | `m12_rdep` — residual distances, residual CUSUM/CUSUMSQ, dependence LR | standalone **+0.00479** (4/5); **+0.00141 vs seed clone** | strongest, still rejected |
| **W5-E7** | `m11_focus` — exact max over candidate τ | ABL +0.00317 (5/5); **CHAMP −0.00123 (2/5)** | REJECTED, sign flips |
| **W5-E8** | absorbing-state BOCPD | **NOT RUN — already shipped** in `m07_bayes` | n/a |
| **W5-E9a** | `pairwise_w`, pairs weighted by `n_neg(t)` | −0.00147 (2/5) | REJECTED |
| **W5-E9b** | `pairwise_h`, weighted squared hinge | −0.00476 | REJECTED |
| **W5-E10** | union of all three blocks (175 cols) | +0.00211 — **less than half** of `m12_rdep`'s 57 cols alone | REJECTED |
| **W5-E11** | every stream rebuilt with `m12_rdep` | **S′ − S = −0.00268 on 1/5** | REJECTED |

`W5-E8` deserves a sentence because it is a "not run" that is not a gap:
`src/sbr/features/m07_bayes.py` already implements the absorbing-state
posterior the brief asked for — its docstring says so — and it ships 50 columns
of it inside RT-600, taking **23.1%** of total model gain, the largest
per-column contribution in the bank. Building a second one would have measured
the seed.

### 4.3 W5-E3 — the hard-negative curriculum

The mining is the part worth understanding, because the obvious implementation
leaks. "Hardness" is a function of the labels, so scoring it from the existing
`RT-300` OOF vector means a row in fold *j* gets its hardness from a model
trained on every fold except *j* — **and that model saw fold *k***, the fold
whose training weights you are about to build. Fold *k*'s labels reach fold
*k*'s training run. The effect is small and entirely capable of manufacturing a
+0.002.

The implementation instead runs, for each outer fold *k*, an **inner 4-fold
cross-fit entirely inside folds ≠ k** — 20 cheap models (`m00_core` only, 250k
rows, 200 trees), 286 s total. `_assert_fold_pure` checks it rather than
trusting it, and the test in this handoff's Part 8 verifies that `H[k]` is NaN
on every row of fold *k*.

The three arms differ **only in the weight vector**:
* `RT-710` uniform — **the control**. A custom objective starts from raw score 0
  rather than the label prior, so the control must be a `wbinary` run and
  **not** `RT-300`. (Measured: `RT-710` fold 0 = 0.62834 vs `RT-300`'s 0.62903.)
* `RT-711` smooth reweighting, `w_neg = 1 + 3r²` on the within-t percentile rank
  of nested hardness; positives `w = 1`. Measured on fold-0 training rows: mean
  weight 1.745, negatives 2.000, positives 1.000, max 4.000.
* `RT-712` genuine row duplication — hardest 10% of negatives enter the sampling
  pool 4×, at the same total row budget (+22.3% pool). Validation and dev row
  sets verified unchanged.

The result is not close:

| arm | TS-AUC | vs `RT-710` | folds |
|---|---|---|---|
| `RT-710` uniform custom-objective control | **0.61472** | — | — |
| `RT-711` smooth reweighting | 0.60770 | **−0.00701** | 1/5 |
| `RT-712` oversampling | 0.60132 | **−0.01339** | 0/5 |

Age 0–20 also moves the wrong way: `RT-710` 0.53149, `RT-711` 0.53007
(−0.00142), `RT-712` 0.52130 (−0.01019). The oversampling arm violates the
pre-registered young-break guardrail by itself, and both treated arms miss the
primary control badly. **W5-E3 is rejected.** Evidence:
`research/reports/wave5_e3_hardneg.json` and
`research/reports/wave5_executive.csv`.

---

## PART 5 — THE FOUR THINGS WAVE 5 ESTABLISHED

These are the transferable results. If you read nothing else, read this part.

### 5.1 The metric does not reward early detection the way this project assumed

The official weight is `n_pos(t)·n_neg(t)` per timestep.
`research/reports/wave5_diagnostics.json`, section D1:

| online index t | share of pair weight | | post-break age | share of pair weight |
|---|---|---|---|---|
| 0–10 | **0.2%** | | 0–5 | **2.9%** |
| 10–50 | 3.7% | | 5–20 | 8.1% |
| 50–200 | 27.4% | | 20–100 | 32.3% |
| 200–700 | **64.3%** | | **100+** | **56.7%** |

25% of the weight is at t ≤ 168, 50% at t ≤ 293, 90% at t ≤ 600.

**Ages 0–20 carry 11% of the weight; age 100+ carries 57%.** The wave-5 brief
states the opposite premise twice ("we care enormously about early evidence
because real-time TS-AUC weights every timestep") — true about timesteps, false
about weight. This retrospectively explains `m09_back`, whose gain was entirely
mature-break: that is what this weighting rewards.

**Consequence for wave 6: do not spend a wave chasing young-break detection.**
It is worth about a fifth of mature-break ranking. `m11_focus` did improve ages
0–10 by +0.0031/+0.0047 — genuinely, as designed — and it did not matter.

### 5.2 The ensemble is saturated, three independent ways

| test | result |
|---|---|
| W4-E6, 13-booster union | −0.00095 |
| W5-E1, λ mixture at every weight | monotone decreasing, best −0.00004 |
| **W5-NULLTEST, an 8th seed clone** | **+0.00003** |

That last one is the useful number. `research/reports/wave5_nulltest_8th_member.log`:
adding an eighth exchangeable member to the seven-specialist blend is worth
**three hundred-thousandths**. A different eighth clone is worth −0.00051.

**Consequence: never evaluate a feature block as an 8th ensemble member.** Its
weight is 1/8, the whole seven-member specialisation effect is +0.0042, and the
control contributes ~0. The framing cannot resolve anything. Evaluate at the
architecture level (rebuild every stream — that is W5-E11) or standalone.

### 5.3 A mature ensemble leaves almost no headroom for an overlapping statistic

This is the wave's central measurement. `m12_rdep`'s advantage shrinks
monotonically as the comparison approaches the deployed system:

```
single booster, standalone         +0.0038  ± 0.0007   STABLE across 3 partitions, 4/5 folds each
two-model ABL blend vs seed clone  +0.0010  ± 0.0009   sign flips; negative on 1 of 3 partitions
eight-member ensemble              +0.0005             3/5 folds
full architecture rebuild          -0.0027             1/5 folds
```

Per-partition detail (`research/reports/wave5_alt_partitions.json`):

| partition | standalone | folds | vs seed clone | folds |
|---|---|---|---|---|
| canonical | +0.00479 | 4/5 | +0.00141 | 4/5 |
| alt1 | +0.00344 | 4/5 | **−0.00025** | 2/5 |
| alt2 | +0.00327 | 4/5 | +0.00190 | 4/5 |

**Both rows are true.** `m12_rdep` genuinely makes a single booster better — one
of the most stable effects in this project. Its marginal value over ordinary
seed diversity is not resolved by this framework, because the seven-stream bank
already contains `m01_seq`'s CUSUM paths, `m07_bayes`'s absorbing posterior and
`m04_resid`'s residual monitors, and averaging seven heterogeneous models over
500 columns recovers what the 57 new ones add — while the columns still cost
their dilution.

**Consequence: a wave-6 feature family must attack a mechanism the bank does not
already cover, not compute an existing mechanism better.**

### 5.4 An ABL result is not a CHAMP result

`m11_focus`: **+0.00317 on 5/5 folds at ABL (400k rows), −0.00123 on 2/5 at
CHAMP (1M rows)** — same folds, same seed, same columns, sign flipped.

Either the 500-column bank has already extracted what those columns carry once
it has 1M rows (capacity substitution), or `feature_fraction=0.5` over 576
columns dilutes what was already working. Both are consistent with W5-E10, where
175 new columns did less than half of what 57 did.

**Consequence: the ABL rung is a screen for the ABSENCE of an effect, not
evidence for one.** Wave 3 rejected `m09_back` on ABL evidence and wave 5
promoted `m11_focus` on it; only the second was a mistake, but both were
unlicensed. Require CHAMP before any promotion claim.

### 5.5 Two corrections to the record

* **The Codex 2025 reproduction failed its own calibration gate.** `polars`,
  `lightgbm`, `shap`, `tabpfn` were never installed; every rung R25-010…R25-050
  is `blocked`; the public repo reports no AUC of its own. **There is no
  verified strong 2025 teacher.** The consequential measurement is on the
  sibling branch: a **boundary-aware oracle** — told the true τ and given h
  post-break points — is **matched or beaten by the legal causal `RT-300` at
  every horizon through h = 150**. Teacher distillation is not justified, and
  not for lack of time: the only regime with real teacher advantage is `FULL`,
  which needs the entire post-break segment. Full review:
  `research/reports/wave5_codex2025_implications.md`.
* **RT-600 has a pre-existing streaming defect.**
  `tests/test_stream_engine_parity.py::test_engine_parity_real` fails on
  `research/wave3-integration` itself, identically, before any wave-5 change:
  1–2 cells per series out of ~200,000 differ between batch and streaming
  `m07_bayes`, in `bo_p_lt25_z`, `bo_lo_change_z`, `bo_ent`. Almost certainly
  score-irrelevant at that magnitude. **No wave-3 or wave-4 document mentions
  it.** Worth 20 minutes to close before any future rebuild.

---

## PART 6 — FILES YOU MAY NOT CHANGE

This is the list to read before writing code. Violating any of these either
invalidates historical numbers or stops the shipped artifact from loading.

### 6.1 Frozen competition artifacts — never edit, never regenerate

```
research/FINAL_ARCHITECTURE_FREEZE.md
research/FINAL_REPRODUCIBILITY_MANIFEST.json
<rt600-worktree>/submissions/C_ensemble_deployable.py
<rt600-worktree>/submissions/C_ensemble_deployable.ipynb
<rt600-worktree>/submissions/C_ensemble_deployable.build.json
<rt600-worktree>/models/final10k_ensemble/manifest.json
<rt600-worktree>/models/final10k_ensemble/model.txt.{0..6}
```
where `<rt600-worktree>` = `structural-break-claude-wave3`.

### 6.2 Core library — read it, never edit it

```
src/sbr/store.py          src/sbr/metric.py        src/sbr/transforms.py
src/sbr/nullcal.py        src/sbr/pipeline.py
src/sbr/features/base.py  src/sbr/features/driver.py
src/sbr/stream/ctx.py
```

`metric.py` and `pipeline.py` are the worst: editing either silently invalidates
every number in `RESULTS.csv`, including RT-600's. If you believe one is wrong,
**say so in your report and do not fix it** — that is the standing rule from
`research/PROTOCOL.md` §1 and it has held for five waves.

### 6.3 Shipped feature modules — these ARE the frozen manifest

```
src/sbr/features/m00_core.py   m01_seq.py   m02_dist.py   m03_dyn.py
                  m04_resid.py   m06_loc.py    m07_bayes.py
```

Any edit — even a comment that changes a column name — moves
`feature_manifest_sha256` away from
`1646c3b9e09d8a7fb3c564483a1d1d999680caeefe848b092f907a6cac80cced`
and `ProductionModel.load` **hard-errors rather than predicting**. That gate
exists because a model and a feature bank that disagree is the failure mode that
silently costs a competition.

### 6.4 Shipped streaming twins

```
src/sbr/stream/s_m00_core.py   s_m01_seq.py   s_m02_dist.py   s_m03_dyn.py
                 s_m04_resid.py   s_m06_loc.py    s_m07_bayes.py
```

### 6.5 `src/sbr/stream/engine.py` — APPEND-ONLY, and there is a trap

You must edit this to register a new streaming module. Three rules:

1. `MODULE_ORDER` may be **appended to**, never reordered or inserted into.
2. The `StreamEngine.__init__` default must stay **`PRODUCTION_MODULES`** (the
   shipped seven). **I broke this in wave 5**: registering `m12_rdep` in
   `MODULE_ORDER` changed the default engine from 7 modules to 8, and four
   existing tests caught it immediately. A registered-but-unshipped module must
   not leak into a default engine.
3. Re-run `tests/test_stream_engine_parity.py::test_rt600_manifest_sha_is_unchanged`
   after touching it.

### 6.6 Validation surfaces — never regenerate, never score on

```
research/folds/folds.parquet              PERMANENT. NEVER regenerate
research/folds/folds_alt{1,2,3}.parquet   robustness only; select nothing with them
research/folds/folds_final10k.parquet     NOT a validation set
research/folds/folds_screen.parquet
research/RESULTS.csv                      append ONLY via sbr.pipeline.run (file-locked)
data/X_test.reduced.parquet
data/y_test.reduced.parquet
data/y_test_index.reduced.parquet
```

Also forbidden as selection surfaces even though they are not files you would
edit: fold `-1` (the spent lockbox) and OOF vectors `RT-500`..`RT-506`.

### 6.7 Protocol definitions — copy them, do not modify them

```
research/scripts/wave2_lib.py       CHAMP and ABL protocol dicts, alt_folds()
research/scripts/wave2_streams.py   the seven specialist stream configurations
```

Wave 4 documented what happens if you ignore this: forcing specialist streams
onto the ABL protocol overrides `num_leaves`, `min_data_in_leaf` and
`feature_fraction` — exactly the knobs that make a specialist a specialist — and
every stream collapses toward the champion. The experiment then measures nothing.

### 6.8 Append-only, never rewrite history

```
research/RDOF_LEDGER.md          research/FAILED_EXPERIMENTS.md
research/EXPERIMENT_ID_MAP.md    research/RESULTS.csv
```

Historical rows keep the IDs they were written with, forever. If a name is
ambiguous, **map it** in `EXPERIMENT_ID_MAP.md`; do not rename it.

### 6.9 What you are free to create

```
src/sbr/features/m1x_<name>.py        new batch feature module
src/sbr/stream/s_m1x_<name>.py        its bitwise twin (required before shipping)
research/scripts/wave6_*.py           new experiment drivers
tests/test_*.py                       new tests
research/reports/*.{md,json,csv}      new reports
```

Wave 5's rejected modules are kept and are useful scaffolding:
`m10_persist.py`, `m11_focus.py`, `m12_rdep.py`. **Reuse `m12_rdep`'s
`_GridNull` / `_WinNull` / `_cum` / `_roll` null machinery by importing it**
rather than reimplementing — that is what made its streaming twin bitwise.

---

## PART 7 — THE TRAPS, EACH OF WHICH COST REAL TIME

1. **`n_online` leakage through a null length.** The first `m12_rdep` sized its
   expanding nulls by the online length. `check_prefix_invariance(..., atol=0.0)`
   failed it on all 7 test series. **Run that gate before taking any number**,
   on ≥8 series of different lengths including both length-10 series in the
   dataset, with cuts `(1, 3, 10, 37, 113)`.

2. **The feature-cache readiness check.** `sbr.features.driver` preallocates the
   output with `np.lib.format.open_memmap(mode="w+")`, so the `.npy` is
   **full-size and mostly zeros from the first second**. Testing existence or
   size proves nothing. Wait for the driver's own `total <n>s` line. Two runs of
   `RT-740` started on a 15%-filled cache; both were killed before reaching the
   ledger, and **a number computed from that cache would have looked entirely
   normal**.

3. **Silent `str.replace` no-ops.** A patch that does not match its pattern
   fails silently and leaves the old code running. This bit twice (the queue
   script, the `_CLASSES` registration). **Assert the pattern was found, then
   grep the file to confirm.**

4. **Three trainers saturate the machine.** 16 GB RAM and 10 GB swap; the third
   concurrent arm pushed swap to 9.9/10 GB with 26M swapouts and slowed
   everything ~5×. Two maximum. Check `sysctl -n vm.swapusage`.

5. **Floating-point parity traps in streaming twins** (both real, both caught
   only by the bitwise test):
   * **The AR-sigma round trip.** Batch computes `e = ctx.ar_online / hp.ar_sigma`
     where `ar_online` was *already multiplied* by `ar_sigma`. That is not the
     identity in IEEE arithmetic. Read `ctx.tr["res_mean"]` — which
     `build_transforms` defines by exactly that expression on exactly that input
     — instead of recomputing the filter.
   * **Reduction width.** Batch reduces a `(bins, n)` array along **axis 0**,
     numpy's strided path. A `(bins, 1)` array is contiguous and takes the
     **pairwise** path, disagreeing in the last ulp on **97 of 300** random
     columns. Pad to width ≥ 2. See `_pad2` and
     `test_reduction_width_invariance`.

6. **Duplicate background jobs.** Chained `until ...; do sleep; done` waiters
   that then launch work will launch it again if you re-arm them. Check
   `ps -Ao etime,command | grep "[p]ython -u"` before assuming the machine is
   idle.

7. **Use the right python.** `/path/to/workspace/structural-break/.venv/bin/python`.
   The anaconda python fails on `import pandas` with `_ARRAY_API not found`.

---

## PART 8 — WHAT WAVE 5 BUILT THAT YOU CAN USE

### 8.1 Reusable machinery

| file | what it gives you |
|---|---|
| `research/scripts/wave5_lib.py` | `Ctx` — folds, scoring, `score_by_age`, cross-fitted SCDF per stream **with an on-disk cache**, equal/weighted blending, paired series bootstrap. The stream cache is the reason a composition costs an array op instead of a ten-minute refit |
| `research/scripts/wave5_stream_eval.py` | the promotion battery: `(S+C)` vs `(S+N)`, age buckets, rank correlations, bootstrap |
| `research/scripts/wave5_abl_compare.py` | the wave-3 comparison that rejected `m09_back`, reusable for any block |
| `research/scripts/wave5_obj.py` | custom objective / per-row weight hook that routes through `pipeline.run`'s existing dispatch, so the fold loop, sampling, seeds and ledger stay byte-identical. `RT-702` proves it: it reproduces `RT-413` to five decimals on all five folds |
| `research/scripts/wave5_e3_hardneg.py` | nested fold-pure hardness mining + genuine row oversampling via a `rows_for` patch that touches only the 4-fold training call |
| `research/scripts/wave5_summary.py` | rebuilds the executive table from `RESULTS.csv` and the report JSONs, so the write-up cannot drift from disk |
| `research/scripts/wave5_partitions.py`, `wave5_alt_blocks.py` | alternate-partition runners |
| `src/sbr/features/m12_rdep.py` | `_GridNull`, `_WinNull` — length-interpolated and window nulls built from history only. **Import these; do not rewrite them** |
| `src/sbr/stream/s_m12_rdep.py` | a worked example of a bitwise twin, including the two float traps and the `_ScalarGrid` tabulation that took it 415 → 225 µs/obs |

### 8.2 Tests worth running before you claim anything

```bash
export SBR_ROOT="$PWD" SBR_STORE="$PWD/cache/store" PYTHONPATH="$PWD/src"
"$PY" -m pytest tests/test_wave5_modules.py -q            # causality + no n_online, 24 tests
"$PY" -m pytest tests/test_stream_parity_m12_rdep.py -q   # bitwise twin, 12 tests
"$PY" -m pytest tests/test_stream_engine_parity.py -q     # RT-600 manifest guard
"$PY" -m pytest tests/test_no_n_online_leakage.py -q      # the gate that protects the entry
```

`test_engine_parity_real` fails **on the parent branch too** — see §5.5. Do not
spend time chasing it thinking you caused it; do consider fixing it.

---

## PART 9 — THE THREE HIGHEST-VALUE NEXT EXPERIMENTS

Ranked by evidence this wave produced, not by appeal.

### W6-A — an expanded residual-PATH module (highest ceiling)

W5-E5's sub-block delivered **72.8% of `m12_rdep`'s gain from 20 of its 57
columns**, at gain-per-column **3.64** — higher than every production module
except `m07_bayes` (2.45). And it is barely explored: those 20 columns are
**one** drift constant (`k = 0.5`), **one** residual representation (the shared
context's AR(2)), two statistics (CUSUM, CUSUMSQ), two signs.

`m04_resid` already ships **eight** residual representations — `ar1`, `ar2`,
`ar5`, ridge `arR`, Huber `arH`, EWMA `vol`, winsorised `volM`, GARCH `volG`,
`cmb` — every one fitted on history only and applied causally forward, and
**none has a CUSUM path**. `m06_loc`'s forensics separately measured AR(6)
residuals as ~0.9 AUC points better than AR(2) for scale localisation, and this
module used AR(2).

Cross {3–4 residual representations} × {2–3 drift constants} × {CUSUM, CUSUMSQ}
with the identical calibrated path geometry.

**A cross-cutting design lesson to carry into it:** in *both* new wave-5 modules
the columns the booster leaned on were the **calibrated path geometry** —
running peak `*_pk`, persistence `*_per`, time-since-peak `*_tsp` — not the
instantaneous statistic. The top four `m12_rdep` columns are `rcdn_pk`,
`rqup_pk`, `rqdn_pk`, `rcup_pk`; the top six `m11_focus` columns are all `*_pk`.
Whatever the detector, what survives into the model is *how high the evidence
has ever been and how long it stayed there*.

**But read §5.3 first.** The bank already contains `m01_seq`'s CUSUM paths on
the raw series. W6-A is a bet that doing it on *residual* streams the bank does
not whiten is different enough. If it comes back at +0.001 over a seed clone,
that is the third time this framework has said "real and redundant", and the
right response is to stop adding statistics.

### W6-B — `m11_focus` ablation, then a slim version

The module's gain concentrates in `*_pk`, `*_agefrac` and `*_anc_z` (the τ=0
**anchored** statistic, which is not maximised at all) — not in the maximised
`*_z` or the `*_gain` contrast that was the hypothesis. Train two arms: `_z` +
`_gain` only, and `_pk` + `_age` only. W5-E10 established that **column count is
a real cost** (175 columns did less than half what 57 did), so a 20-column
`m11_focus` that keeps the contribution would be strictly better than the
76-column one — and might survive the union test the full module failed.

### W6-C — calibration anchor placement (cheapest, and never once measured)

`SmoothTimeCDFCal` uses 12 **log-spaced** anchors, 256-point grids,
`min_n = 400`. The freeze says plainly: *"Anchors, grid size and `min_n` are
frozen and were never tuned."* Meanwhile W4-E1 measured the calibration family
as worth **+0.00267** on the specialist arm — one of the largest single effects
in the project — and W5-D1 measured where the metric's weight actually is:
**50% between t = 168 and t = 451**, a narrow band that log-spacing deliberately
under-resolves, because log anchors crowd near t = 1 where **0.2%** of the pair
weight lives.

**This needs no training at all** — it recomputes from OOF vectors already on
disk, which is why it should go first in wall-clock order even though W6-A has
the higher ceiling. Pre-register a tiny structured family (say: log,
uniform-in-pair-weight, hybrid) and **no continuous search** — this is the
surface where tuning against OOF would be easiest and most damaging.

---

## PART 10 — WHAT NOT TO DO

* **Do not submit.** Nothing cleared the bar. LB-001 = 0.6268 stands.
* **Do not evaluate a feature block as an 8th ensemble member.** §5.2.
* **Do not promote on an ABL result.** §5.4.
* **Do not treat low rank correlation as evidence.** §3.2.
* **Do not search the composition lattice** — pairwise unions, block subsets,
  best-of-k, per-block `feature_fraction`. W4-E6 and W5-E10 both excluded it in
  advance, and searching it after a negative result is how a null becomes a
  false positive.
* **Do not chase young-break detection as a priority.** §5.1.
* **Teachers: see the correction.** §5.5 originally said "do not build a
  teacher study". That was too broad — corrected in `WAVE6_PREREG.md` §12. The
  oracle result rules out the **τ-knowledge** lever given generic features; it
  says nothing about the **representation** lever, which no experiment here has
  ever tested. An offline neural teacher is legitimate and is `W6-E4`.
* **Do not rebuild the CV framework.** It passed external calibration.
* **Do not tune anything against these folds** without pre-registering the grid
  in `research/RDOF_LEDGER.md` first, with a real git SHA and a real seed.

---

## PART 11 — REPRODUCING THE WAVE-5 HEADLINES

```bash
cd "/path/to/workspace/structural-break-wave5"
export SBR_ROOT="$PWD"
PY="/path/to/workspace/structural-break/.venv/bin/python"

"$PY" -u research/scripts/wave5_e1_mixture.py      # baselines A/B/C + the λ grid   (~21 min)
"$PY" -u research/scripts/wave5_diagnostics.py     # D1 metric geometry, D2 false positives, D3 age
"$PY" -u research/scripts/wave5_break_family.py    # D4 break family
"$PY" -u research/scripts/wave5_abl_compare.py RT-730 RT-740 RT-750 RT-760
"$PY" -u research/scripts/wave5_e11_eval.py        # the architecture verdict
"$PY"    research/scripts/wave5_summary.py         # the executive table
```

Every headline number in the wave-5 documents is regenerated by one of these.

---

## PART 12 — GIT AND PROVENANCE

| | |
|---|---|
| branch | `research/wave5-alpha` |
| parent | `research/wave3-integration` @ `17bb5df` |
| relationship | strictly ahead; **behind by nothing**; never merged into itself |
| machine | macOS/arm64, 10 cores, 16 GB |
| feature cache | shared **read-only** from `structural-break-claude-wave3`; the wave-3 worktree still holds exactly its original eight modules and no `RT-7xx`/`RT-8xx`/`wave5_*` file |

**Merge recommendation.** Wave 5 promoted nothing, so there is no model change
to merge. What is worth keeping is the *record and the machinery*: the
pre-registration, the ledger entries, the four negative results with mechanisms,
the diagnostics, the three modules, the streaming twin, and the tests. Merging
`research/wave5-alpha` into `research/wave3-integration` is safe **because the
RT-600 manifest is asserted unchanged by a test** — but it is not urgent, and
leaving it as a research branch is equally defensible.

**The honest one-line summary of wave 5:** it spent its compute discovering that
this ensemble has almost no headroom left for statistics that overlap what it
already computes, measured that four independent ways, and corrected three
premises the project had been carrying — while moving the score not at all.

---

## PART 13 — MEASURED 2026-08-22: THE CALIBRATION IS AIMED AT THE WRONG PLACE

W6-C was ranked third on cost. This measurement moves it to first on evidence.

The shipped `SmoothTimeCDFCal` uses **12 log-spaced anchors**, frozen and, per
the architecture freeze, **never tuned**:

    1, 2, 4, 7, 12, 23, 43, 81, 152, 285, 533, 999

Against where the official `n_pos(t)·n_neg(t)` weight actually sits:

| anchor band | share of pair weight |
|---|---|
| [1, 2) | 0.01% |
| [2, 4) | 0.03% |
| [4, 7) | 0.08% |
| [7, 12) | 0.21% |
| [12, 23) | 0.71% |
| [23, 43) | 2.01% |
| [43, 81) | 5.52% |
| [81, 152) | 13.12% |
| **[152, 285)** | **26.62%** |
| **[285, 533)** | **35.90%** |
| **[533, 999)** | **15.78%** |

**Nine of twelve anchors sit at t ≤ 168, covering 25% of the pair weight.
Exactly ONE anchor (t = 285) lies inside t ∈ (168, 451], the band carrying 50%
of it. The first seven anchors together cover 1.05% of the metric.**

Over half the calibration's resolution is spent on one percent of what is
scored. This is not a tuning opinion — it is a measured mismatch between where
the calibration can express a change in the score distribution and where the
metric weighs one.

It matters because the calibration family is not a minor term: W4-E1 measured it
at **+0.00267** on the specialist arm, one of the largest single effects in the
project, and it is the mechanism that makes seven heterogeneous streams
comparable at a fixed `t` — which is what TS-AUC rewards.

**Do this first.** It needs **no training**: the anchor scheme is applied to OOF
vectors already on disk, so the whole family is minutes of compute.

**And pre-register it tightly, because this is the most dangerous surface in the
project.** It is a direct knob on the evaluation metric's own geometry, fitted
on the same folds that judge it. Declare a **tiny structured family in advance**
— for example exactly three schemes: incumbent log-spaced, uniform-in-
pair-weight, and a hybrid — with **no continuous search, no anchor-count sweep,
no grid-size sweep**, and the same +0.0030/≥4-of-5/bootstrap/alternate-partition
bar as everything else. A calibration tuned against these folds would transfer
worse than a feature does, and the +0.0027 that makes this attractive is exactly
the size of gain that a tuned-in artefact would fake.
