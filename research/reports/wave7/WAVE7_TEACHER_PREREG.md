# WAVE 7 — TEACHER/DISTILLATION PILOT, PRE-REGISTRATION

**Written 2026-08-23 on `research/wave7-teacher-distillation`, committed BEFORE
any teacher-target diagnostic or student score exists.** Population, teacher
construction, student arms, thresholds and the interpretation rule are fixed
here. Nothing below may change after the first number is read, per the
commit-boundary discipline this project has followed since wave 5
(`research/WAVE5_PREREG.md`, `research/WAVE6_PREREG.md`,
`research/WAVE7_D3R_PREREG.md`).

**Question.** W7-D3R (`research/reports/wave7_d3r.md`, verdict **CASE 2 —
future-information limit**) established that the dominant loss cell (t≥200,
positive age≥100, 50.50% of dev pair weight, 45.29% of remaining RT-600 loss,
`research/WAVE7_RT600_EXACT_ALPHA_BUDGET.md`) is not fixed by more tree
capacity on the existing 500 legal causal columns (Arm B − Arm A = −0.00592
cell AUC, 4/5 folds negative), but is largely resolved by each series' own
future (Arm C − Arm B = +0.07110 cell AUC, 5/5 folds, +0.05277 to +0.08180 per
fold). Arm C is not causal and can never ship. **Can a strictly causal student
be taught, at inference time, to recover some fraction of that future-information
advantage from a privileged training-time teacher signal?**

---

## 1. TEACHER — REUSE ARM C, DO NOT RETRAIN

Per the brief's own instruction to prefer a mechanism that already demonstrated
the gain over an arbitrary invented oracle: the teacher target `Q` is built
directly from the existing `research/oof/RT-991.npy` vector (W7-D3R Arm C),
**not from a newly trained model.**

```
Q = clip(RT-991.npy, 1e-6, 1 - 1e-6)
```

`RT-991` was trained with `objective="binary"`, so its raw predictions are
already sigmoid-bounded, continuous, and rank-informative — no further
transform is applied before diagnostics (§4). No rank transform, no
per-t CDF calibration: `Q` is used purely as a continuous supervision *target*
for a regression-style objective, not as a score being combined with other
scores by addition, so the global-vs-per-t calibration-geometry concern that
applies to blend weights does not apply here (monotonicity of `Q` in the raw
Arm-C score is all that is used, and a single teacher's label values are never
summed against another stream's).

### 1.1 Cross-fitting — already satisfied structurally, not re-implemented

Folds in this project are series-level (`sbr.pipeline.Data.row_fold`, one fold
per series, never split within a series). `RT-991`'s OOF value for any row in
fold `f` came from a model trained on folds `≠ f`, i.e. **a model that never
saw that row's series** — the standard cross-fitting guarantee is already
satisfied by construction, at the series level, exactly as required by the
brief's §12. No inner cross-fitting loop is added. If a future teacher
iteration trains a *new* model (rather than reusing `RT-991`), inner
cross-fitting would be required and must be pre-registered separately before
that arm is scored.

### 1.2 Teacher quantities are training labels only, never features

`Q` (and `RT-991.npy` itself) is used exclusively as the `label=` argument to
`lgb.Dataset`. It is never concatenated into a student's feature matrix. The
student's `X` is built by the exact same `sbr.pipeline.load_features(FULL)` +
`sbr.pipeline._stack(...)` path used by `RT-300`/`RT-990` (the unmodified
500-column causal bank), with **no additional columns** — a shape assertion
(`Xtr.shape[1] == 500`) is run before every training call as a mechanical
guard against accidentally reusing Arm C's 1000-column `augmented_stack`.

---

## 2. TEACHER TARGET DIAGNOSTICS — REQUIRED BEFORE ANY STUDENT SCORE

Runner: `research/scripts/wave7_teacher_pilot.py --diagnostics`. Computed and
committed (`research/reports/wave7_teacher_diagnostics.{md,json}`) **before**
`--train` is ever invoked. Reports, over the canonical dev partition:

* `Q` mean/std/quantiles, split by `y` (positive/negative) and by dominant-cell
  membership (in-cell / outside-cell) — 4 groups.
* Same-`t` discriminatory power of `Q` itself: pooled dev TS-AUC and dominant-cell
  TS-AUC (this reproduces Arm C's own already-known numbers — 0.71989 pooled,
  0.71859 cell — as a sanity check that `Q` was loaded correctly, not a new
  result).
* Pearson correlation of `Q` with: `y`, `t`, post-break age (positives only),
  `tau_index` (positives only), `n_online`, `has_break` (among negatives, i.e.
  never-break vs pre-break separation).

**Stop-and-redesign condition (fixed here, not after seeing the numbers):** if
`|corr(Q, y)| > 0.98` **and** `Q` restricted to positives shows near-zero
variance (std < 0.02), `Q` is not meaningfully richer than a hard label and
the target must be redesigned before any student trains. Any other diagnostic
pattern (including strong correlation with age/tau among positives — expected,
since Arm C's advantage over Arm B is largest exactly where age is largest) is
not, by itself, disqualifying: the target is *supposed* to encode future
confirmation, which is naturally correlated with maturity.

---

## 3. STUDENT ARMS — ONE CANONICAL FOLD (FOLD 0), SAME INPUTS/CAPACITY/ROWS

All three arms share: the 500-column causal bank (`FULL` from
`wave2_lib.py`), fold 0 as validation, folds {1,2,3,4} as the training pool,
`MAX_TRAIN_ROWS = 1_000_000`, `rng = np.random.default_rng(0)` used for
exactly one `rng.choice` call before any other draw (reproducing `RT-990`'s
fold-0 training-row sample bit-for-bit, since `RT-990`/Arm B's own loop makes
its first and only relevant `rng.choice` call at `f=0`), and Arm B's tree
capacity (`num_leaves 127, min_data_in_leaf 150, feature_fraction 1.0,
bagging_fraction 0.8, bagging_freq 1, n_estimators 900, learning_rate 0.05,
lambda_l2 5.0, max_bin 127, num_threads 2`). **Only the training label and
objective change between arms.** This isolates supervision as the sole lever,
per the brief's §15.

| arm | exp id | label | objective | training |
|---|---|---|---|---|
| **T0** — hard-label control | `RT-990` (reused) | `y` | `binary` | **not retrained** — `RT-990`'s existing fold-0 OOF slice is bit-identical to what this arm would produce, since it is the same rows/columns/capacity/seed |
| **T1** — pure teacher distillation | `RT-992` | `Q` | `xentropy` | new, fold 0 only |
| **T2** — hard + teacher, fixed blend | `RT-993` | `0.5·y + 0.5·Q` | `xentropy` | new, fold 0 only |

No target-weight grid. `0.5/0.5` is the only blend scored in this pilot,
chosen for symmetry between legal ground truth and privileged confirmation,
not tuned. `RT-992`/`RT-993` are the next unallocated IDs
(`research/EXPERIMENT_ID_MAP.md` §7 — highest allocated is `RT-991`; `RT-901`–
`RT-908` remain deliberately unallocated and are not touched).

### 3.1 Objective validity — `xentropy` hard-label parity, checked before T1/T2

LightGBM's built-in `objective="xentropy"` (continuous cross-entropy) is used
rather than a hand-rolled soft-label objective, since none exists in this repo
(confirmed by search — no `assert_causal_names`/custom soft objective/banned-
token gate exists anywhere in this codebase; an earlier external brief
describing such code was written against a different, non-existent repo
state). Before T1/T2 train, `research/scripts/wave7_teacher_pilot.py
--parity-check` runs a reduced-scale spot check (100,000 train rows, 150
trees, same fold-0 split) comparing `objective="xentropy"` against
`objective="binary"` **on hard `y` labels**: booster predictions must have
Pearson correlation ≥ 0.999 and cell/pooled TS-AUC within 0.0005 of each
other. If this fails, `xentropy` is not trusted and T1/T2 do not run under
this pre-registration.

---

## 4. WHAT IS NOT DONE

No hyperparameter search on T1/T2 (capacity is fixed, identical to `RT-990`).
No blend-weight grid on T2. No true `tau`, true age, `n_online`, or boundary
indicator as a student **feature** (only the 500 unmodified causal columns).
No promotion decision from this pilot — a one-fold screen can only clear or
fail a continuation gate (§6), never itself justify a submission.

---

## 5. EVALUATION

For T0/T1/T2, on fold 0 only: dominant-cell TS-AUC (pooled, never-break-only,
pre-break-only — same mask as `wave7_d3r.py:cell_mask`), and fold-0 whole-dev
TS-AUC (`Ctx.rows[0]`, all rows, not just the cell). Deltas `T1−T0` and
`T2−T0`, cell and whole-fold.

**Distillation efficiency**, matched to fold 0 specifically (not the pooled
5-fold D3R numbers, since this pilot is fold-0 only):

```
distillation_efficiency = (T1_cell_auc − T0_cell_auc) / (ArmC_fold0_cell_auc − ArmB_fold0_cell_auc)
                         = (T1_cell_auc − T0_cell_auc) / 0.05277
```

(`ArmC_fold0_cell_auc − ArmB_fold0_cell_auc = 0.71658 − 0.66381 = +0.05277`,
`research/reports/wave7_d3r.json`, fold 0 — the smallest of the five D3R
fold-level C−B gaps, so this is a conservative denominator.)

**Translated aggregate delta**: `0.5050 × ΔAUC_cell` (`research/
WAVE7_RT600_EXACT_ALPHA_BUDGET.md` §E pair-weight fraction), reported for
whichever of T1/T2 has the larger cell delta.

---

## 6. ONE-FOLD CONTINUATION GATE — FIXED BEFORE ANY SCORE

Continue to a full 5-fold run (new IDs, not `RT-992`/`RT-993` re-used) if
**any** of:

* **A.** dominant-cell TS-AUC delta (T1 or T2 vs T0) ≥ **+0.010**, with no
  material (> −0.003) whole-fold-0 damage outside the cell;
* **B.** whole-fold-0 aggregate TS-AUC delta ≥ **+0.003**;
* **C.** translated aggregate delta (§5) ≥ **+0.004**.

Kill (do not fund a 5-fold run under this pre-registration; return to the
`research/EXPERIMENT_ID_MAP.md` allocation for any follow-up) if the
translated aggregate delta is **≤ +0.0005**; treat **≤ +0.001** as "probably
kill" pending a specific, named reason to expect fold 0 was unusually hard for
this mechanism (checked against D3R's own fold-0 being the *smallest* of its
five C−B gaps, §5 — fold 0 is if anything a conservative pilot fold, not a
favorable one).

Full 5-fold promotion (if the pilot clears and a full run is later funded)
still requires the project's standing bar
(`research/HANDOFF_WAVE6.md` §3.3 / `research/WAVE5_PREREG.md` §4): **≥
+0.0030** TS-AUC over the strongest matched control, positive on **≥4/5**
folds, paired series bootstrap CI supportive, alternate partitions stable.
This pre-registration does not shortcut that bar — it only decides whether a
5-fold run is worth funding at all.

---

## 7. LEAKAGE SENTINELS

* Feature-matrix shape assertion (`Xtr.shape[1] == Xva.shape[1] == 500`) before
  every `lgb.train` call — guards against accidentally reusing Arm C's
  1000-column `augmented_stack`.
* `Q`/`RT-991.npy` never appears in any `_stack(...)` call or `names` list
  used to build `X` — checked by construction (the training script only ever
  loads `FULL`'s causal modules for `X`; `Q` is loaded once, separately, and
  used only as `label=`).
* No true `tau_index`, post-break age, `n_online`, or boundary indicator is
  read anywhere in the student training path (`train_student()` in
  `wave7_teacher_pilot.py` takes no `Ctx`/`meta.parquet` argument at all —
  only `PL.Data()` + `FULL` features + the pre-built label array).
* After the pilot, `RT-992`/`RT-993` are marked, in
  `research/EXPERIMENT_ID_MAP.md`, exactly like `RT-990`/`RT-991`: what
  privileged information their *training* used, and that the trained student
  boosters themselves take no privileged input at inference (only the 500
  causal columns) — this is the actual causality claim, and it is true by
  construction here (unlike `RT-990`/`RT-991`, `RT-992`/`RT-993` **are**
  legal to deploy as trained artifacts if promoted, since only their labels,
  never their inputs, used privileged information).

---

## 8. RUNNER

`research/scripts/wave7_teacher_pilot.py`:

* `--diagnostics` — §2, before anything else.
* `--parity-check` — §3.1, before T1/T2.
* `--train t1` / `--train t2` — §3, fold 0 only.
* `--analyze` — §5–6, writes `research/reports/wave7_teacher_pilot.{md,json}`.

Machine limit unchanged: at most two trainers concurrently
(`research/HANDOFF_WAVE6.md` §2.4). T1 and T2 are run sequentially in this
pilot (current swap usage was already elevated before this work started —
`sysctl -n vm.swapusage` showed ~7.8/9.2 GB used with zero trainers running —
so contention is avoided rather than relied on to stay under the limit).

---

## GIT

| | |
|---|---|
| starting SHA (preserved from `research/wave6-alpha`) | `a084956` |
| this pre-registration commit | see `git log -1` after commit |
| first teacher-target diagnostic | strictly after the pre-reg commit |
| first student score | strictly after the diagnostic and parity-check |
