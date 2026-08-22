# WAVE 6 — PRE-REGISTRATION AND EXECUTION PLAN

**Branch to create: `research/wave6-alpha`, from `research/wave5-alpha` @ `7ef9118`.**
Written 2026-08-22, **before any Wave-6 number exists**. Binding.

`research/PROTOCOL.md`, `research/VALIDATION_V2.md` and
`research/HANDOFF_WAVE6.md` remain in force. Read the handoff first — in
particular Part 6 (files you may not change) and Part 7 (the traps).

---

## 0. WHAT WAVE 6 IS FOR, AND WHEN TO STOP

LB-001 = **0.6268**. Wave 5 ran ten experiments and moved the score by **zero**,
while establishing — four independent ways — that this ensemble has almost no
headroom left for statistics that overlap what it already computes.

**Wave 6 is therefore not another feature wave.** It spends its budget on the
two things Wave 5 measured as unexamined, in this order:

1. a **measured mismatch** between where the calibration has resolution and
   where the metric has weight (Part 13 of the handoff);
2. a **measured headroom** of +0.0396 at the FULL horizon that nobody has
   priced — is it worth knowing τ, or is it something else?

**THE STOPPING RULE, FIXED NOW.** If **W6-E1 and W6-E2 both return inside
noise** (< +0.0010 and < +0.010 respectively, definitions in §3 and §4), Wave 6
**stops adding** and the recommendation becomes: ship RT-600, write the wave up,
and do not open W6-E3. Four independent measurements already say this system is
saturated for the kind of work Wave 5 did. A fifth negative is information; a
sixth is not.

Approximate gaps for scale, not targets: top-50 **+0.0023**, top-25 +0.0092,
#1 +0.0242.

---

## 1. VALIDATION SURFACE — UNCHANGED

**Permitted:** `research/folds/folds.parquet` folds 0–4 (8,000 dev series),
`folds_alt1/2/3`, `sbr.metric.ts_auc_flat` at official `pairs` weighting.

**Forbidden as a selection surface:** `RT-500`..`RT-506`, `folds_final10k`,
fold −1, `X_test.reduced`, `y_test.reduced`, `y_test_index.reduced`, the
leaderboard.

**Reference arms, unchanged from Wave 5** (OOF vectors already on disk):

| symbol | set | members |
|---|---|---|
| `A` | single control | `RT-300` |
| `B` | seed-clone ensemble | `RT-300`, `RT-401`–`RT-406` |
| `S` | **the RT-600 architecture** | `RT-300`, `RT-410`–`RT-415` |

`S` = 0.62581. `B` = 0.62164. `A` = 0.61605.

---

## 2. THE PROMOTION BAR — UNCHANGED, PLUS ONE NEW CONTROL

A candidate is promoted only if **all four** hold:

1. ≥ **+0.0030** TS-AUC over the strongest matched control;
2. positive on ≥ **4/5** canonical folds;
3. paired series bootstrap (200 reps, common random numbers) CI supportive;
4. alternate partitions positive, or at minimum directionally stable.

**The Wave-5 lesson that must not be forgotten:** the strongest single framing
of a candidate is usually the most favourable one. `m12_rdep` read +0.00141,
+0.00046, −0.00025 and −0.00268 across four legitimate framings. **Report every
framing you compute, and lead with the one closest to the deployed system.**

### 2.1 The calibration null control — the seed-clone analogue for W6-E1

A calibration change has no seed-clone analogue, so one is defined here, in
advance, because without it W6-E1 cannot be interpreted.

**`NULL-ANCHOR`: twelve anchors drawn log-uniformly at random on [1, 999],
sorted, seeds fixed below.** It is a *different* anchor placement carrying **no
information about where the metric's weight is**. If a random re-placement gains
as much as a weight-aware one, then the gain is "any perturbation of a frozen
grid helps" — i.e. the incumbent placement is merely unlucky, not
systematically wrong — and the weight-aware story is unsupported.

**Seeds, fixed now, not to be substituted or extended:
`0, 1, 7, 42, 2026`.** Five draws; report all five; the control value is their
**mean**, not their best.

---

## 3. W6-E1 — CALIBRATION ANCHOR PLACEMENT

**Zero training. Runs on OOF vectors already on disk. Do this first.**

### 3.1 Hypothesis

`SmoothTimeCDFCal` places 12 log-spaced anchors at
`1, 2, 4, 7, 12, 23, 43, 81, 152, 285, 533, 999`. Measured against the official
`n_pos(t)·n_neg(t)` weight:

* the first seven anchors together span **1.05%** of the pair weight;
* **nine of twelve** sit at t ≤ 168, spanning the first **25%**;
* **exactly one** (t = 285) lies inside t ∈ (168, 451], which carries **50%**.

The calibration is the mechanism that makes seven heterogeneous streams
comparable at a fixed `t` — which is precisely what TS-AUC rewards — and W4-E1
priced the calibration family at **+0.00267** on the specialist arm. The
architecture freeze states that anchors, grid size and `min_n` were **never
tuned**. H1: placing the same twelve anchors where the metric's weight is
improves the specialist ensemble.

**H0:** anchor placement is immaterial once there are twelve of them, and any
apparent gain is reproduced by `NULL-ANCHOR`.

### 3.2 The schemes — exactly three, declared now, not to be extended

Everything else is **frozen at the shipped values**: 12 anchors, 256-point
quantile grids, `min_n = 400`, `time_coord = "log_n_seen"`, cross-fitted so
fold *k*'s map is fitted on folds ≠ *k*, equal-weight mean over streams.

| id | scheme | anchors |
|---|---|---|
| `CAL-LOG` | **incumbent / control** | 12 log-spaced on [1, max t] |
| `CAL-WT` | uniform-in-pair-weight | the 12 quantiles of the cumulative `n_pos·n_neg` distribution over t |
| `CAL-HYB` | hybrid | 6 log-spaced on [1, 168] + 6 weight-quantiles above 168 |

`CAL-HYB` exists because `CAL-WT` may starve small `t` of any anchor at all,
and rows at small `t` still have to be scored even though they carry little
weight; the hybrid keeps a floor of early resolution.

### 3.3 The leakage rule — this is the part that can silently fake a result

The pair-weight distribution is a function of the **labels**
(`n_pos(t)`, `n_neg(t)`). Placing anchors from it is legitimate *model fitting*
— SCDF already fits its grids from training OOF — **but it must be cross-fitted
exactly as the grids are: fold *k*'s anchor placement is computed from the pair
weights of folds ≠ *k* only.** Computing one global placement over all five
folds and applying it everywhere leaks fold *k*'s labels into fold *k*'s ranking
and is expressly forbidden here.

Assert it in code (`_assert_anchor_fold_pure`), do not trust it.

*(This is training-time fitting and is unrelated to the `n_online` prohibition,
which governs features at inference. At inference the calibration only looks up
`t`.)*

### 3.4 Falsification

H1 is rejected unless

    best of {CAL-WT, CAL-HYB}  −  CAL-LOG   >=  +0.0030   over the five canonical folds
    AND positive on at least 4 of 5 folds
    AND the paired series bootstrap 95% CI excludes zero
    AND the same delta is positive on alt1 and alt2
    AND it exceeds the MEAN of the five NULL-ANCHOR draws by at least +0.0020

**Screening threshold for the stopping rule (§0):** if the best scheme beats
`CAL-LOG` by **< +0.0010**, W6-E1 counts as "inside noise".

### 3.5 Cost and outputs

Five stream-calibrations per scheme × 5 folds ≈ **30 s per scheme** with the
warm cache; eight evaluations total (3 schemes + 5 null draws) ≈ **10 min**,
plus ~15 min bootstrap for the winner, plus alt1/alt2 re-runs ≈ 10 min.
**Under an hour, no training.**

Write `research/reports/wave6_e1_anchors.json` and record every scheme,
including the null draws individually.

---

## 4. W6-E2 — PRICE THE LOCALISATION LEVER (ORACLE DIAGNOSTIC)

**This experiment can never produce a production model. Its output is a number
that decides what W6-E3 is.**

### 4.1 Hypothesis

`codex/oracle-information-frontier-2026` measured a boundary-aware oracle as
matched or beaten by the legal `RT-300` at every horizon through h = 150, and
**+0.0396 ahead at FULL**. That is the only large headroom anywhere in this
problem, and 57% of the metric's pair weight sits at post-break age 100+, the
same regime. But the oracle used a *generic* 150-tree bank while `RT-300` uses
500 purpose-built columns, so the gap could be either:

* **(a)** the oracle knows **τ** and we must estimate it, or
* **(b)** the oracle's features are better in that regime.

**These imply opposite Wave-6 programmes and nobody has separated them.**

### 4.2 Design

Train the champion configuration (CHAMP protocol, seed 0, the seven production
modules) **plus a small oracle block** giving the true break location:

    true post-break age            t - tau      (and log1p of it)
    is-post-break indicator        1[t >= tau]
    trailing mean / robust scale / lag-1 product of the segment (tau, t]
    the same three vs their historical-null z

`RT-900` = champion + oracle block. Control = `RT-300`, already on disk.

**Report the delta by post-break age bucket**, because the aggregate is not the
question — the 100+ bucket is.

### 4.3 The decision rule, fixed in advance

    RT-900 - RT-300 at age 100+  >=  +0.010   ->  knowing tau is the lever.
                                                 W6-E3 = causal localisation.
    RT-900 - RT-300 at age 100+  <   +0.010   ->  tau knowledge is NOT the gap.
                                                 W6-E3 = model-class substitution,
                                                 and the oracle-frontier FULL
                                                 headroom is recorded as
                                                 unreachable rather than untried.

**Screening threshold for the stopping rule (§0):** < +0.010 at age 100+ counts
as "inside noise" for W6-E2's own purpose — but note it still *decides* the
branch, so W6-E2 is never wasted.

### 4.4 Discipline

Label every artifact `ORACLE / DIAGNOSTIC — NOT DEPLOYABLE`, exactly as
`wave5_break_family.py` and the oracle-frontier study do. The module lives in
`research/scripts/`, **never** in `src/sbr/features/`, so it cannot be picked up
by `load_all()` and cannot reach a production manifest.

Cost: one small feature build (~10 min) + one CHAMP run (~20 min).

---

## 5. W6-E3 — THE CONDITIONAL BRANCH

**Do not start this until W6-E1 and W6-E2 have both reported.** Which arm runs
is determined by §4.3, not by preference.

### 5.1 Arm A — causal localisation (if τ knowledge is worth ≥ +0.010)

`m11_focus` already maximises **exactly** over candidate τ and is bit-identical
to brute force on the maximum. It gained +0.00317 at ABL and **lost 0.00123 at
CHAMP**. So the mechanism is not dead — the delivery was. The two things Wave 5
learned about it, and the shape of the retry:

* its gain concentrated in `*_pk`, `*_agefrac` and `*_anc_z`, **not** in the
  maximised statistic or the `*_gain` contrast that was the hypothesis;
* 175 columns did less than half of what 57 did, so **column count is a real
  cost**.

Pre-register a **slim** `m11_focus` — the ablation in handoff Part 9 (W6-B),
run first as two arms (`_z`+`_gain` only, `_pk`+`_age` only), then a single
~20-column module built from whichever half carries it. Same bar. **No column
search beyond those two arms.**

### 5.2 Arm B — model-class substitution (if τ knowledge is not the gap)

Every member of `S` is LightGBM. The +0.0042 specialisation gain comes from
configurations "wrong in different directions"; a different model *class* is
wrong in more different directions, and this axis has never been tested.

**Substitute, do not add.** An eighth member is worth **+0.00003** whatever it
is (`W5-NULLTEST`). Replace exactly one member and keep seven.

* **Member to replace, fixed now: `RT-410`** (0.60512, the weakest specialist).
  Not chosen by search — it is the minimum, stated before the run.
* **Replacement class, fixed now: `sklearn.ensemble.HistGradientBoostingClassifier`.**
  `xgboost`, `catboost`, `torch` and `tabpfn` are **not installed** in the frozen
  environment; `sklearn` 1.9.0 is. HistGB differs from LightGBM in binning,
  growth policy and regularisation while remaining a gradient-boosted tree, so
  it is a genuine but not reckless change of class.
* Same seven modules, same folds, same row budget, `max_iter` and `max_leaf_nodes`
  set to LightGBM's `n_estimators`/`num_leaves`; **no hyperparameter search**.

**The engineering cost is real and must be priced before promotion:**
`ProductionModel` loads members with `lgb.Booster(model_file=...)`. A sklearn
member requires extending the artifact's serialisation and its streaming
`step()`. **Do not begin that work until the research arm clears the bar.**

### 5.3 Arm C — the residual-path module (only if E1 and E2 both fail AND you overrule §0)

`W6-A` in handoff Part 9. It has the best per-column evidence in the project
(72.8% of `m12_rdep`'s gain from 20 of 57 columns, gain-per-column 3.64) and it
is a **bet against §5.3 of the handoff**. It is listed here so that the decision
to run it is explicit and recorded, not drifted into.

---

## 6. EXECUTION SCHEDULE

| # | task | depends on | compute | wall clock |
|---|---|---|---|---|
| 0 | branch `research/wave6-alpha`, worktree, symlink caches (handoff §2.3) | — | none | 10 min |
| 1 | commit this file **before any number** | 0 | none | — |
| 2 | **W6-E1** three schemes + five null draws, canonical | 1 | none | ~10 min |
| 3 | W6-E1 bootstrap + alt1/alt2 for the winner | 2 | none | ~25 min |
| 4 | **W6-E2** oracle block build + `RT-900` CHAMP run | 1 | 1 trainer | ~35 min |
| 5 | **DECISION GATE** — apply §0 stopping rule and §4.3 branch | 3, 4 | — | — |
| 6 | W6-E3 arm A or B per the gate | 5 | 1–2 trainers | 2–4 h |
| 7 | alternate partitions for anything that clears | 6 | 2 trainers | ~1 h |
| 8 | `STATE_OF_RESEARCH_V6.md`, ledger, failed-experiments entries | all | none | 1 h |

Steps 2–4 are independent and 2 needs no CPU, so **run W6-E2's training while
W6-E1 computes**. Two trainers maximum, always (handoff §2.4).

**Total to the decision gate: about one hour of wall clock.** That is the point
of this ordering — the cheapest experiment is also the best-evidenced one.

---

## 7. DEGREES OF FREEDOM DECLARED IN ADVANCE

| | |
|---|---|
| anchor schemes | **3**, named in §3.2, not to be extended |
| null-anchor draws | **5**, seeds `0, 1, 7, 42, 2026` fixed in §2.1; control is their **mean** |
| calibration knobs touched | **anchor placement only.** Anchor count (12), grid size (256), `min_n` (400) and `time_coord` stay frozen |
| oracle-block columns | ~8, listed in §4.2 |
| member substituted in arm B | **1**, `RT-410`, named before the run |
| replacement class | **1**, HistGB, named before the run |
| hyperparameter searches | **0** |
| ensemble weight optimisation | **0** — equal weighting stands on W4-E6 and W5-E1 |

**Every run reaches `research/RESULTS.csv` through `sbr.pipeline.run` with a
real git SHA and a real seed.** New canonical IDs: `RT-9xx`, verified unused
before allocation.

---

## 8. EXPLICITLY EXCLUDED

Anchor-count sweeps · grid-size sweeps · `min_n` sweeps · continuous
optimisation of anchor positions · more than one substituted member · best-of-k
model classes · pairwise unions of feature blocks · teacher distillation
(handoff §5.5) · young-break-targeted families (handoff §5.1) · any use of
`n_online`, future data, cross-series state or the leaderboard · rebuilding the
CV framework.

**And the one that will be most tempting:** if `CAL-WT` gains +0.002, do **not**
try 16 anchors, or 20, or a different grid size, to push it over +0.0030. That
is the lattice this section exists to close. The bar was set before the number.
