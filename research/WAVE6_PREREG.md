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

---

# AMENDMENT 1 — 2026-08-22: THE MODEL CLASS WAS NEVER THE CONSTRAINT

**§5.2 above contains an error and this amendment supersedes it.**

## 9. THE ERROR

§5.2 restricted the model-family arm to `sklearn` on the grounds that
"`xgboost`, `catboost`, `torch` and `tabpfn` are **not installed** in the frozen
environment". That conflated two unrelated things:

* **what is in the local research venv** — a research friction, fixed by
  `pip install`, and not a statement about anything;
* **what is deployable** — governed by the CrunchDAO **whitelist** for
  `structural-break-real-time`, with dependencies declared in `requirements.txt`
  and installed by the platform before the model runs, plus a
  *request-whitelisting* route for anything absent.

The local `requirements.txt` is itself proof the two differ: it lists `ruptures`
and `matplotlib`, neither of which is in the frozen environment.

**The consequence is not cosmetic.** It narrowed a whole research track to one
`sklearn` estimator on a premise I did not check, and the correct constraint —
whitelist, runtime, determinism, streaming — is both different and looser.

## 10. W6-E0 — ESTABLISH THE ACTUAL CONSTRAINT (DE-RISKING, NOT BLOCKING)

**Corrected 2026-08-22.** An earlier draft called this "blocking". That was
overstated in two ways:

* **The research question does not need the whitelist.** "Does an XGBoost member
  beat a seed clone?" is answerable today by installing it in the research venv.
  That result is valid whether or not the library can ship. What needs the
  whitelist is *committing to a deployment path*, which only matters once
  something clears the bar — and on wave-5 base rates, serialising on it would
  cost more than it saves.
* **The risk is concentrated at the exotic end.** XGBoost and CatBoost are
  mainstream, and Crunch staff have confirmed a top-level `import torch` is
  detected and installed. Rungs 1–3 carry low risk of wasted work. The genuine
  uncertainty is TabPFN-class and specialised time-series libraries, where the
  answer may be "request whitelisting" with lead time.

**So: W6-E0 is required before PROMOTING anything, and does not gate STARTING
rungs 1–3.** Run it in parallel. It matters most for rung 4 and the teacher
track, where the library choice is open and the build cost is highest.

Steps:

1. Read **Resources → Whitelisted Libraries** on the `structural-break-real-time`
   competition page and record the list verbatim in
   `research/reports/wave6_whitelist.md`, with the date and the URL.
2. Record the declaration mechanism for *this* competition's submission type —
   `requirements.txt` for scripts; confirm what the notebook/`.py` artifact we
   ship actually uses, since `submissions/C_ensemble_deployable.py` currently
   declares nothing beyond the frozen stack.
3. Note the *request whitelisting* route and its criteria (on PyPI, established
   ~6 months, documented, reasonably popular, verified PyPI details) for
   anything worth asking for.
4. **Do not infer the whitelist from the local venv. Ever again.**

## 11. THE CONSTRAINTS THAT ARE REAL WHATEVER THE WHITELIST SAYS

"Allowed" and "sensible to deploy" are different, and four gates bind
independently of the library list. Any candidate model family must pass all
four **before** it is worth research time:

| gate | requirement | why it bites |
|---|---|---|
| **determinism** | `crunch test` runs a determinism check at **1e-8**. Two runs must agree | the shipped artifact passed this; a neural model needs fixed seeds, single-threaded or deterministic kernels, `torch.use_deterministic_algorithms(True)`, and no nondeterministic reductions. Achievable on CPU, not free |
| **streaming contract** | per-series, per-point, single pass, **no `n_online`**, no cross-series state | an RNN/GRU is a natural fit — it *is* a carried state. A TCN needs a bounded receptive-field buffer. A transformer over the full prefix is O(t²) per point unless KV-cached, and must never see beyond `t` |
| **runtime** | current artifact 1.734 ms/pt → ~2h11m at P=4 on 16 vCPU / 64 GB, against a 15 h budget | roughly **5–7× headroom**. That is real room for a small network, and not room for a large transformer at 999 points × ~2,000 series |
| **artifact size** | persisted `resources/` capped at **10 GB** | irrelevant for a small net; relevant for TabPFN-style or ensemble-of-nets designs |

## 12. CORRECTION TO A WAVE-5 CONCLUSION — TEACHERS ARE NOT RULED OUT

`research/STATE_OF_RESEARCH_V5.md` §7 and `research/HANDOFF_WAVE6.md` §5.5 and
Part 10 say **"teacher distillation is NOT justified"**. That was stated too
broadly and is corrected here.

**What the evidence actually shows.** The oracle-frontier study gave a model the
**true τ** plus `h` post-break points and found the legal causal `RT-300`
matches or beats it at every horizon through h = 150. But that oracle was a
**fixed 150-tree LightGBM over a generic feature bank**. So the finding is:

> Knowing where the break is buys nothing at h ≤ 150 **given generic features**.

That constrains the **τ-knowledge** lever. It says **nothing** about the
**representation** lever — whether some function of the raw prefix `x_{1:t}`
carries information that 500 hand-built causal columns do not encode. A
TCN/GRU/transformer reading the raw series attacks exactly that, and **no
experiment in this project has ever tested it.**

**This also raises the value of W6-E2**, which now does more than pick a branch:

    tau knowledge worth >= +0.010 at age 100+   ->  the LOCALISATION lever is live
    tau knowledge worth <  +0.010               ->  localisation is exhausted, and any
                                                    remaining headroom must live in the
                                                    REPRESENTATION -> the neural track is
                                                    the best-motivated thing in wave 6

## 13. REVISED W6-E3 — MODEL FAMILY (supersedes §5.2)

Still **substitution, not addition** — an eighth member is worth **+0.00003**
(`W5-NULLTEST`) whatever it is. Still exactly one member replaced: **`RT-410`**,
the weakest specialist at 0.60512, named before any run.

Run in this order, cheapest and least deployment-risk first, and **stop at the
first one that clears the bar** — this is a ladder, not a search:

| rung | family | why | deployment cost |
|---|---|---|---|
| 1 | **XGBoost** | different tree construction, split-finding and regularisation; drop-in on the same 500-column matrix | low — another tree serialisation |
| 2 | **CatBoost** | ordered boosting, different dynamics again | low–medium |
| 3 | **PyTorch MLP** on the existing 500 causal features | learns *nonlinear combinations* trees approximate axis-wise; genuinely different inductive bias | medium — determinism + a `step()` path |
| 4 | **GRU / TCN** on the raw prefix | the **representation** lever of §12; carries explicit sequential state, natural streaming fit | high — the real test of §11's four gates |

Rungs 1–2 need the whitelist confirmed; 3–4 need it confirmed **and** a
determinism proof before any promotion claim.

**Each rung is judged by the same unmoved bar** (§2): `S` with `RT-410`
substituted, against `S` unchanged and against the seed-clone framing, ≥ +0.0030
on ≥4/5 folds, bootstrap supportive, alternate partitions stable.

## 14. W6-E4 — THE OFFLINE TEACHER TRACK (new)

The strategically interesting property: **the teacher never has to ship.**
Offline compute is unbounded; the deployed system stays a fast streaming
engine. Four exit routes, in increasing deployment cost:

1. **diagnostic only** — does the teacher's score contain information beyond
   `S`? Measured the same way every candidate is: does `S + teacher` beat
   `S + seed clone`?
2. **feature discovery** — if it does, what are the series and timesteps where
   it wins, and what mechanism do they share? That is a specification for a
   causal feature, which is the cheapest possible way to buy the gain.
3. **distillation** — regress the teacher's output on the existing causal
   feature bank with a small student. Ships as one more LightGBM stream, zero
   new deployment risk.
4. **direct deployment** — only if §11's four gates pass with margin.

**The pre-registered order is 1 → 2 → 3, and 4 only on overwhelming evidence.**
Route 2 is the one this project is best set up to exploit and the one most
likely to survive contact with the streaming contract.

**Teacher training may use future data. Its OUTPUT may not reach any production
path except through route 3, where the student sees only causal features.**
Label every artifact `TEACHER / DIAGNOSTIC — NOT DEPLOYABLE`, keep it in
`research/scripts/`, never in `src/sbr/features/`, exactly as the oracle studies
do.

## 15. WHAT DOES NOT CHANGE

The discipline is unchanged and applies to every rung and route above:
substitution not addition · the seed-clone control · ≥ +0.0030 on ≥4/5 folds,
bootstrap, alternate partitions · report **every** framing computed and lead
with the one closest to the deployed system · no hyperparameter search beyond a
single named configuration per rung · every run through `sbr.pipeline.run` with
a real SHA and seed · the §0 stopping rule still governs W6-E1 and W6-E2.

**And the §8 exclusions still hold.** A neural family is not a licence to sweep
architectures. One named configuration per rung, declared before it runs.

---

# AMENDMENT 2 — 2026-08-22: W6-E2 IS VOID; THE ORACLE QUESTION MOVES TO THE SERIES LEVEL

**Written after RT-900 was diagnosed and voided, and BEFORE any W6-E2R number
exists.** The corrected experiment had not been run when this section was
committed; the git history is the proof.

---

## 16. W6-E2 / RT-900 IS PERMANENTLY VOID

`RT-900` scored **0.86552** against the champion's 0.61605, +0.24947, 5/5 folds.
Under §4.3 that reads "localisation is the lever". **It is not evidence of
anything.**

The oracle block is `NaN` for every row with `t < cut` — there is no post-cut
segment to compute over. For a break series `cut = tau`, so the block's
missingness mask **is** the row-level target `y[t] = 1[t >= tau]`, and LightGBM
splits on missingness natively.

| diagnostic | value |
|---|---|
| TS-AUC of the bare indicator `1[t >= cut]`, nothing else | **0.81442** |
| share of dev rows where the oracle block is `NaN` | 49.1% |
| **share of those `NaN` rows that are negatives** | **100.00%** |

`RT-900` is retained in `research/RESULTS.csv` with `status=VOID`, in
`research/EXPERIMENT_ID_MAP.md`, in `research/FAILED_EXPERIMENTS.md` and in
`research/RDOF_LEDGER.md`. It is never to enter a model-comparison table as
valid alpha, an ensemble, a promotion decision, feature selection, production or
a submission. **No production path was touched:** `w6oracle` was never a
registered module, never reachable from `sbr.features.base.load_all()`, never in
a manifest, never in a `crunch test`.

**It is not patchable, and no repair is authorised.** Not by imputing the NaNs,
not by dropping `or_elapsed` or `or_frac`, not by adding a missing indicator,
not by zero- or null-filling, not by masking columns, not by changing LightGBM's
missing handling. The placebo cut — the wave-1 taxonomy's construction and the
oracle-frontier study's — is the right protection at the **series** level and
cannot work at the **row** level, because the series-level question ("does this
series contain a break?") is not answered by the boundary while the row-level
question ("has the break happened by now?") is answered by it exactly.

---

## 17. STANDING RULE — TRUE τ UNDER ROW-LEVEL TS-AUC

**For the row-level real-time target, any experiment that gives the learner true
τ, directly or indirectly, is invalid for predictive-performance measurement.**

"Indirectly" includes any feature whose **availability, support, length,
missingness, denominator, calibration window or segment boundary** depends on
true τ in a way visible to the learner.

True τ may be used only for: post-hoc diagnostics; age-bucket evaluation;
offline oracle studies under a **non-degenerate** protocol; teacher analysis
where target leakage is explicitly quarantined. It may never be exposed to a
real-time row classifier. Mirrored into `research/PROTOCOL.md` §1 and enforced
by `tests/test_no_tau_leakage.py`.

---

## 18. W6-E2R — THE CORRECTED, SERIES-LEVEL, KNOWN-BOUNDARY REPRESENTATION TEST

### 18.1 Why the series level is non-degenerate

At **one row per series** there is no before/after target for the boundary to
encode. Knowing τ does not tell you whether the series is positive; it tells you
*where to look*. This is precisely the prior oracle-information-frontier
protocol, so W6-E2R inherits an instrument that has already passed its own
permutation, metadata and random-boundary controls.

### 18.2 The question

> At the FULL horizon — the only horizon where the frontier measured real
> headroom (+0.0396 over legal `RT-300`) — does **our 500-column causal bank**,
> given the same boundary, beat the frontier's **generic 150-tree bank** at
> **0.6497**?

This is a **representation-capacity** question, nothing else.

### 18.3 Protocol — matched to the prior study, item by item

| item | value |
|---|---|
| population | canonical dev folds 0–4, 8,000 series, fold −1 excluded |
| store | `structural-break-claude-wave3/cache/store`, sha256 `2c6aab9b…3971` / `10c22b00…c06b` |
| folds | `research/folds/folds.parquet`, sha256 `ba4f71fe…c312` |
| unit | **one row per series** (asserted in code) |
| label | series `has_break` |
| metric | **series ROC AUC** — never row-level TS-AUC |
| horizon | `FULL` only |
| boundary | positives: true `tau_index`. negatives: `assign_pseudo_taus` verbatim |
| pseudo seeds | `0, 1, 7, 42, 2026` |
| learner | `LGBMClassifier(n_estimators=150, lr=0.04, num_leaves=31, min_child_samples=20, subsample=0.85/freq 1, colsample 0.85, reg_lambda 2.0, class_weight balanced)` behind a median `SimpleImputer` |
| learner seed | `pseudo_seed + 17` for every rich arm — identical across arms |
| cross-fit | the frontier's `crossfit_model` over folds 0–4, unmodified |
| runner | `research/scripts/wave6_e2r.py`, which **imports the original frontier script by path and does not modify it** |

The frontier module is loaded from
`structural-break-oracle/research/scripts/oracle_information_frontier.py` at
sha256 `0b56baae4fb4331ee6d95035a9230ba7c8e4d0aa5ac8e45edce4aa541cf27bc8`, and
its sha is recorded in the output JSON.

### 18.4 The arms

| arm | representation |
|---|---|
| `A_rich` | the frontier's generic known-boundary bank — **the reproduction target, 0.6497** |
| `A_basic` | the frontier's basic bank — secondary reproduction target, 0.6418 |
| **`B_causal`** | **our 500 production columns, evaluated at the boundary split** |
| `B_causal_withpos` | the same, keeping `t_online`/`log_t_online` — diagnostic only |
| `C_nobound` | our 500 columns at the end of the series, **no boundary at all** — the matched representation control |
| `AB` | `A_rich ++ B_causal` — are the two banks complementary |

**How arm B is given the boundary.** The engine runs **unmodified** on a
re-split series: `hist' = hist ++ online[:boundary]`, `online' =
online[boundary:]`, and the **last row** of its `(n_online', 500)` output is the
series vector. This is the exact analogue of the frontier's
`compare_segment(post, hist)`: the post-boundary segment against a pre-boundary
reference. No feature is invented, no module is modified, nothing is fitted to
the label.

**`t_online` and `log_t_online` are dropped from the primary arm.** At the last
row those two columns **are** `post_len`, which the frontier excludes from every
model via `MODEL_EXCLUDE_COLUMNS`. Keeping them would hand arm B a metadata
channel arm A does not have. `B_causal_withpos` reports the undropped variant so
the size of that channel is visible rather than assumed.

**Eligibility: `post_len >= 10`.** The shipped store's shortest online segment
is 10 points, so a re-split with fewer post-boundary points is outside anything
the modules were built for — several blocks are literally undefined there and
the engine raises. The filter is applied **identically to every head-to-head
arm**, and `A_rich` is **also** scored on the unfiltered population, which is
the reproduction check against 0.6497. Both numbers are reported; the dropped
counts are reported per class.

### 18.5 Leakage sentinels — run and read BEFORE the arms are interpreted

| id | uses only | must show |
|---|---|---|
| `S1` | `boundary`, `rel_boundary`, `n_hist`, `n_online`, `post_len` | no material prediction |
| `S2` | the **missingness pattern** of arm B (and of arm A) | no material prediction |
| `S3` | valid-column counts, pre length, post length | no material prediction |
| `S4` | arm B with **permuted labels** | ≈ 0.50 |
| `S5` | arm B and arm A with a **placebo boundary drawn for both classes** | no large AUC |

The frontier's own FULL-horizon controls are the calibration for "material":
metadata-only **0.5379**, permuted **0.5075**, random-boundary-both **0.5821**.
A sentinel materially above its frontier counterpart invalidates the run.

**S5 will not be zero and is not expected to be.** The frontier's own
random-boundary control at FULL is 0.5821: a randomly placed cut still splits a
break series into segments that differ on average. S5 is read as a *floor*, and
the reportable quantity is `B_causal − S5_placebo_B` alongside
`A_rich − S5_placebo_A`.

### 18.6 Pre-registered interpretation — fixed now

Let `Δ = mean(B_causal) − mean(A_rich)` over the five pseudo seeds, both on the
eligible population.

| case | condition | reading | consequence |
|---|---|---|---|
| **A** | `|Δ| < 0.005` | the frontier's generic bank was already near representation saturation; the remaining real-time gap is dominated by unknown τ, early evidence, localisation and sequential uncertainty | **lowers** the priority of large new representation models |
| **B** | `Δ >= +0.005`, robust | representation quality is a meaningful lever and the generic bank was underpowered | **strengthens** the case for learned sequence representation |
| **C** | `Δ >= +0.010`, robust | representation is clearly still live | proceed into the neural track aggressively |
| **D** | reproduction fails, or any sentinel fires | the instrument is uncalibrated | **infer nothing**; do not report a frontier |

"Robust" means: positive on **at least 4 of the 5 pseudo seeds**, and the paired
series bootstrap (400 resamples over series, common random numbers) 95% CI on
`B − A` excludes zero.

**Case D is checked first.** Reproduction is declared successful only if
`A_rich` on the unfiltered population lands within **±0.010** of the prior
`0.6497` — roughly three times the prior study's own across-seed std of 0.0035.
If it does not, the comparison is not run against a stale scalar; the run is
reported as a failed reproduction and nothing is concluded.

### 18.7 What this experiment cannot prove — binding on the write-up

Whatever number arm B returns, it is **not** an information ceiling, **not**
Bayes-optimal performance, **not** an information-theoretic limit. It is the
performance of *one representation* and *one learner* under *known-boundary
series-level* evaluation. The write-up uses "representation diagnostic",
"known-boundary benchmark", "measuring instrument". It does not use "true
information frontier".

It also says nothing about the real-time row-level task directly: the series
question and the row question are different questions, which is the entire
lesson of RT-900.

### 18.8 Discipline

The oracle diagnostic **may** decide whether representation research looks
promising. It **may not** select TCN depth, hidden size, kernel width, dropout,
learning rate or sequence length. Those belong to §20.

No Crunch submission during the diagnostic. `RT-600` = 0.6268 remains LB-001.

---

## 19. ID ALLOCATION — RT-900 IS NOT REUSED

`RT-901`–`RT-908` are **not** allocated: `RT-903`, `RT-906`, `RT-907`, `RT-908`
already name 2025-reproduction ideas throughout `research/WAVE5_PREREG.md` and
`research/STATE_OF_RESEARCH_V5.md`, and re-using them would collide in text.

| new id | is | namespace |
|---|---|---|
| `RT-940` | W6-E2R arm `A_rich` — frontier bank reproduction | series-level diagnostic |
| `RT-941` | W6-E2R arm `A_basic` | series-level diagnostic |
| `RT-942` | W6-E2R arm `B_causal` — our bank at the boundary | series-level diagnostic |
| `RT-943` | W6-E2R arm `C_nobound` — our bank, no boundary | series-level diagnostic |
| `RT-944` | W6-E2R arm `AB` union | series-level diagnostic |
| `RT-960` | W6-N1 — MLP on the 500 causal features | row-level TS-AUC candidate |
| `RT-970` | W6-N2 — causal dilated TCN | row-level TS-AUC candidate |
| `RT-980` | W6-N3 — GRU, only if N1/N2 justify it | row-level TS-AUC candidate |

**`RT-940`–`RT-944` get no `research/RESULTS.csv` row.** That file is the
row-level TS-AUC ledger, and putting a series-ROC-AUC number in it is exactly
how a diagnostic gets mistaken for alpha later. They are filed in
`research/reports/wave6_corrected_oracle.{md,json,csv}` and indexed in
`research/EXPERIMENT_ID_MAP.md`.

---

## 20. W6-N — THE NEURAL TRACK

Pre-registered in full in **`research/WAVE6_NEURAL_PREREG.md`**, which is
binding and must be committed before any neural number exists. Summary of the
parts that are fixed here:

**Permission.** The neural track is **allowed**, and its permission does not
depend on the W6-E2R outcome (§18.6 sets its *priority*, not its legality).
Justification: a causal learned sequence representation is the strongest
materially different model family the project has never tested, and Amendment 1
established that the model class was never the constraint.

**Order — no jumping ahead.** `N1` compact MLP on the existing 500 causal
features → `N2` compact causal dilated TCN → `N3` GRU only if N1/N2 justify
escalation. A transformer is not authorised by this amendment.

**Two input tracks, never conflated.** Track A (500-feature vector → MLP) tests
**learner** capacity. Track B (compact causal channels → TCN) tests
**representation** learning. If A helps and B does not, the learner is the
lever; if B helps materially beyond A, learned temporal representation is.

**Budget.** At most two architecture sizes per family and at most two
regularisation settings. BCE first; at most one pre-registered metric-aligned
alternative (same-timestep pairwise ranking loss). No loss sweep, no
architecture search, no seed fishing.

**Hard constraints.** No τ, no future observations, no `n_online`, no final
online length, no boundary-conditioned availability or missingness, no oracle
features in any deployable neural input. No bidirectional RNNs, no unmasked
attention, no full-sequence normalisation, no padding mask encoding final
length. Normalisation must be historical / causal-running / a fixed training-set
transform — never full-online or future-aware batch statistics. Prefix
invariance is required to `<= 1e-8` on shared prefixes with different futures.

**Promotion bar (in addition to §2, not instead of it).** Robust positive
aggregate delta; positive on ≥ 4/5 folds; paired bootstrap support; incremental
value over the specialist ensemble; **gain beyond a matched seed clone**; no
regression in the causality/prefix tests. Major architectural promotion also
requires alternate partitions to support the direction. Beating one LightGBM
model is not sufficient and never was.

**Deployment is a separate question.** A neural model does not have to ship. The
offline-teacher route of §14 stays open: probability, embedding, hard-negative
score, auxiliary target or motif cluster, distilled into LightGBM / XGBoost / a
small MLP.

---

## 21. DEGREES OF FREEDOM ADDED BY THIS AMENDMENT

| | count |
|---|---|
| W6-E2R arms | 6 (5 declared + 1 diagnostic variant), 0 selected |
| W6-E2R sentinels | 7, all pre-declared, none selectable |
| pseudo seeds | 5, the prior study's, unchanged |
| thresholds fixed in advance | 3 (`±0.010` reproduction, `0.005` case B, `0.010` case C) |
| neural families authorised | 2 now (`N1`, `N2`), 1 conditional (`N3`) |
| hyperparameters the diagnostic may select | **0** |
| Crunch submissions authorised | **0** |

`RT-900` counts in the ledger as **attempted but void** — not as evidence, and
not as nothing.
