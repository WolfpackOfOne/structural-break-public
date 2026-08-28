# DEEP ENSEMBLE FRONTIER 2026 — PROGRAM PLAN

**Status:** `PLAN_ONLY — NO EXPERIMENT AUTHORIZED BY THIS DOCUMENT`
**Branch:** `research/deep-ensemble-frontier-2026`
**Base:** `research/gpu-tabular-2026@0c1a7df` ("Persist only fold checkpoints, not the 8 GB feature cache")
**Worktree:** `/path/to/workspace/structural-break-deep-ensemble-frontier-2026`
**Written:** 2026-08-28
**Author context:** written after auditing `deep-research-report (4).md` against live repository state.

---

## 0. HOW TO READ THIS DOCUMENT, AND WHAT IT IS NOT

This is a **program plan**, not a preregistration. It is deliberately verbose because
the failure mode this project keeps hitting is not "we lacked an idea" — it is "we ran
an idea whose falsification condition, control, and cost were only half-specified, and
then had to argue afterwards about what the number meant." Every workstream below is
written out to the point where the corresponding `*_PREREG.md` can be produced almost
mechanically from it.

**This document authorizes nothing.** Under `AGENTS.md` §"Preregistration-before-score
rule", no run that can produce a headline number may begin until a separate
preregistration file exists, is committed, and the commit predates the score. This plan
is the input to those preregistrations, not a substitute for them.

Three things this document is explicitly *not*:

1. **Not a promotion decision.** Production remains `RT-600` on `production/rt600`
   (external Crunch leaderboard **0.6268**). Nothing here changes that, and nothing here
   may change it without clearing the promotion battery defined in `AGENTS.md`
   §"What constitutes a valid promotion".
2. **Not a re-litigation of closed hypotheses.** `H-A` (representation saturation),
   `H-B` (objective mismatch) and `H-E` (learned-null misspecification) are closed by the
   Causal Representation Frontier program. Where this plan touches near them
   (§9, neural), it does so under an explicitly different question, with the closure
   stated and respected.
3. **Not an endorsement of `deep-research-report (4).md`.** §3 records precisely which of
   that report's claims survived verification and which did not. One of its central
   operational claims is **wrong in a way that inverts the next action**, and the next
   agent must not act on it. See §3.2.

**Ordering note.** Workstreams are presented in **descending expected value per unit of
compute and per unit of research degree-of-freedom spent**, not in order of intellectual
interest. Neural sequence models are deliberately **last** (§9). That placement is an
evidence-driven ranking, not a dismissal, and §9.1 states exactly what would move them up.

---

## 1. VERIFIED STATE OF THE WORLD

Everything in this section was read off live repository state on 2026-08-28, not quoted
from a prior report. `AGENTS.md` §"Verify branch/SHA instead of trusting stale prompts"
requires this and it caught a real error (§3.2).

### 1.1 Production and scoring anchors

| Item | Value | Source |
|---|---|---|
| Competition | ADIA Lab / CrunchDAO Structural Break Challenge — **Real-Time Edition** | `research/STATUS.md` |
| Metric | Time-Stratified AUC (`sbr.metric.ts_auc_flat`); one prediction per online step | `research/STATUS.md` |
| Causality constraint | prediction at `t` may use only data at or before `t`; bitwise prefix invariance at `atol=0.0` | `AGENTS.md` §"Causality rules" |
| Production anchor | `RT-600`, seven-specialist SCDF blend, `production/rt600` | `research/STATUS.md` |
| External score | **0.6268** (LB-001) | `research/STATUS.md`, `EXPERIMENT_ID_MAP.md` |
| RT-600 dev OOF (canonical 5 folds) | 0.62581, re-verified exact (Δ +0.000001) | `LEADERBOARD_ASSAULT_STATUS.md` |
| RT-600 as `E0` in the replacement battery | **0.638276** | `CRF_FINAL.md` §B |
| Matched exchangeable clone `E1` | 0.638586 | `CRF_FINAL.md` §B |

Note the two different RT-600 numbers. **0.62581** is whole-dev pooled OOF. **0.638276**
is `E0` in the nested-replacement battery population. They are not interchangeable and
mixing them is a recurring source of confusion in prior reports. **Every number in this
plan that carries a `marginal_vs_clone` is on the battery population.**

### 1.2 The seven-specialist roster (the object every workstream operates on)

Reconstructed from `research/scripts/wave2_streams.py::JOBS` and
`research/scripts/wave4_lib.py::SPECIALIST_ALIAS`. `FULL` = the 500-column bank
(`m00_core, m01_seq, m02_dist, m03_dyn, m04_resid, m06_loc, m07_bayes`).

| Slot | Wave-2 alias | Modules | Breadth | Seed | `max_train_rows` | Distinguishing LightGBM idiosyncrasy | CatBoost status |
|---|---|---|---|---:|---:|---|---|
| `RT-300` | `RT-100R` | FULL | 7 modules / 500 cols | 0 | 1,000,000 | the champion configuration itself | **tested** — CAT-300, `+0.001029459` |
| `RT-410` | `RT-120R` | `m00_core,m01_seq,m07_bayes` | 3 modules | 0 | 900,000 | `num_leaves=127`, `feature_fraction=0.35` — deep, narrow-feature | **tested** — CAT-410, `+0.000753095` **KILL** |
| `RT-411` | `RT-121R` | `m02_dist,m03_dyn,m04_resid,m06_loc` | 4 modules | 1 | 900,000 | `num_leaves=31`, `feature_fraction=0.7` — shallow, shape/dynamics | **UNTESTED** |
| `RT-412` | `RT-122R` | FULL | 7 modules / 500 cols | 7 | 900,000 | `num_leaves=255`, `extra_trees=True`, **`sample_mode="per_series"`** | **UNTESTED** |
| `RT-413` | `RT-123R` | FULL | 7 modules / 500 cols | 0 | 700,000 | **`objective="pairwise_t"`**, groups = online index `t` | **tested** — CAT-413, `+0.001087151` (best) |
| `RT-414` | `RT-124R` | `m07_bayes,m06_loc,m01_seq` | 3 modules / ~170 cols | 3 | 700,000 | no window-bank features at all | **UNTESTED** |
| `RT-415` | `RT-125R` | FULL | 7 modules / 500 cols | 11 | 700,000 | **`boosting="goss"`**, keeps large-gradient rows | **UNTESTED** |

Per-module column counts are not tabulated anywhere in the ledgers; `PROTOCOL.md` caps a
module at ~60 columns, and the `RT-414` ≈170 figure comes from the `wave2_streams.py`
stream-E note. Module *count* is used as the breadth proxy in §6.2 and §6.3 for that reason —
if exact column counts matter to the P1 test, derive them before preregistering rather than
estimating.

**This table is the single most important object in this plan.** Four of seven slots
have never been tested with a second learner family, and the untested four are not a
random sample — they include the two structurally *most* idiosyncratic LightGBM
configurations in the ensemble (`RT-412` extra-trees/per-series, `RT-415` GOSS).

### 1.3 The loss geometry we are trying to move

From `WAVE7_RT600_EXACT_ALPHA_BUDGET.md` via `LEADERBOARD_ASSAULT_STATUS.md`:

- **Dominant cell** = current `t ≥ 200` **AND** positive break age `≥ 100`.
- It carries **45.29%** of exact remaining pairwise inversion loss.
- It carries **50.50%** of total pair weight.
- Cell AUC is **0.66428**.
- **Inside** that cell, **never-break negatives carry 74.0%** of the loss; pre-break only 26.0%.

Consequence for translation arithmetic: `pooled_delta ≈ 0.5050 × cell_delta`, holding all
other cells fixed. Any claim of the form "this fixes the dominant cell" must be divided by
roughly two before it becomes a pooled TS-AUC claim.

### 1.4 The information frontier (why we are not hunting for new signal)

`W7-D3R` (`reports/wave7_d3r.md`), run on exactly the dominant cell:

| Arm | Description | Cell TS-AUC |
|---|---|---:|
| A — `RT-300` | legal prefix baseline | 0.65341 |
| B — `RT-990` | **same information, more tree capacity** | 0.64749 |
| C — `RT-991` | full sequence, each series' own future (offline diagnostic only) | **0.71859** |

- **B − A = −0.00592.** More capacity on the identical legal columns buys *nothing*, and
  is mildly negative on 4/5 folds. Verdict: the legal causal feature bank, trained harder,
  has hit its ceiling.
- **C − B = +0.07110**, positive on 5/5 folds, every fold ≥ +0.053. This is a **ceiling,
  not a target** — Arm C reads each series' own future and can never be deployed.
- Recorded verdict: **CASE 2, future-information limit.**

### 1.5 What has actually produced positive marginal alpha (the base rate)

This is the honest prior any new proposal must be scored against.

| Program | Best `marginal_vs_clone` | Verdict |
|---|---:|---|
| **CatBoost Specialist Activation (RT-1257 hybrid)** | **+0.002407205** | **PROMOTION_WORTHY** |
| CatBoost single slot (CAT-413 / `RT-1254`) | +0.001087151 | INTERESTING |
| CatBoost single slot (CAT-300 / `RT-1255`) | +0.001029459 | INTERESTING |
| CSA-00 zero-training 8th member (`RT-1251` added) | +0.000975838 | descriptive |
| LA-03 per-series history adaptation | +0.001676065 | **KILL** — failed fixed-null isolation gate at `−0.021495290` |
| RT-1216 weighted conformal test martingale | +0.000937 | KILL (closest first-sweep miss) |
| T2 / `RT-995` teacher distillation | +0.00024 | MOSTLY REDUNDANT |
| CAT-410 | +0.000753095 | KILL |
| SS-01 repair-damage arbiter | −0.000310 | KILL |
| SS-02 dominant-cell residual ranker | −0.000290 | KILL |
| SS-03 negative-side null calibrator | −0.000299 | KILL |
| SS-04 specialist-disagreement micro-router | −0.000312 | KILL |
| LA-01 specialist replacement salvage | −0.000005408 | KILL |
| First sweep: `RT-1200/1201/1202/1204/1206/1208/1210/1212/1214/1215/1218` | all below gate | all KILL |
| CRF-01 NNCSR (`RT-1234`) | **never computed** — abandoned at standalone gate | KILL |
| CRF-02 ACGN (`RT-1240`) | **never computed** — abandoned at standalone gate | KILL |

**Read this table before proposing anything.** Across two exhausted preregistered sweeps,
a neural representation program, a distillation program, and a leaderboard-alpha program,
**exactly one mechanism has produced material positive marginal ensemble alpha: swapping
the learner family in an existing specialist slot.** That mechanism is currently
**3/7 explored**.

### 1.6 Diversity is not the bottleneck — arbitration is

From `CRF_FINAL.md` §F, "Diagnosis: the bottleneck is arbitration, not detection":

- CRF-01 **repairs 39.5%** of RT-600's sampled dominant-cell mistakes.
- Repair **Jaccard against the seed clone `RT-401` is only 0.174** — the repairs are
  genuinely its own, not rediscoveries.
- It **damages 26.8%** of what RT-600 already had right.
- Pre-break damage rate on RT600-correct pairs: **0.2626** against a **0.0150** cap.

And from `FIRST_SWEEP_SYNTHESIS.md` §H1, measured across the whole first sweep:

> Any candidate repairs **91.8%** of sampled dominant RT600 mistakes, but any candidate
> damages **83.0%** of sampled RT600-correct dominant pairs.

Also recorded there: **dominant-cell pair net is the strongest non-tautological correlate
of `marginal_vs_clone`, Pearson 0.879.** And the program-audit fact:
**`corr(standalone, ρ) = +0.983` across 17 scored arms** — nothing this project has ever
built has been both good and different.

**Implication that governs this entire plan:** we do not have a detection problem. We have
a *retention* problem. A candidate that finds new repairs it cannot keep is worth nothing,
and we have now built several of those.

---

## 2. BINDING CONSTRAINTS

These are not negotiable within this program. Sources: `AGENTS.md`, `research/PROTOCOL.md`,
and the frozen conventions inherited from `FULL_OOF_PREREG.md` / CatBoost `PREREG.md`.

### 2.1 Data and causality

- **No lockbox.** The 2,000-series lockbox is opened only for a final confirmation read,
  never for iterative selection. If in doubt whether a check is "selection", it is.
- **No test data.** `X_test.reduced.parquet` or any path containing `reduced`/`test` is
  never read during research.
- **True `tau` is forbidden**, directly or as a proxy, as a feature or as any input to
  online inference.
- **Bitwise prefix invariance at `atol = 0.0`** for every feature module. CRF-01's
  preflight standard is the bar: 24 gates green, prefix invariance bitwise, truncation
  `2.220e-16`, batch composition `1.665e-16`, fold purity proven *with a positive control
  that reproduces a known-contaminated scheme*.
- **Future information only in an explicitly authorized teacher-only role**, never online.
- **Nested cross-fitting mandatory** wherever an auxiliary model is fit on data overlapping
  the evaluation fold. Run the fold-purity sentinel. Outer-fold contamination is a real,
  previously-caught defect in this repo (`W7` teacher pilot, `RT-992`/`RT-993`).

### 2.2 Folds and calibration

- Canonical **series-level, permanent** folds from `research/folds/folds.parquet`, folds
  `0..4`. Never regenerate. Never row-wise split. Never let two prefixes of one series
  straddle a boundary.
- Calibration is **fold-pure `SCDF_NSEEN`**: fold `k`'s calibration is fitted only on the
  other four folds.
- Ensemble integration is **equal weight**. Blend-weight optimization is not authorized
  anywhere in this program without its own separate preregistration.

### 2.3 The binding evaluation protocol

Every candidate in this program is judged by **nested slot replacement**, outer-fold pure
(fold `k`'s replaced specialist is chosen using only the other four folds, then frozen and
evaluated on fold `k`):

- **`E0`** — original seven-specialist RT-600.
- **`E1`** — six RT-600 specialists + exchangeable matched LightGBM clone in the target slot.
- **`E2`** — six RT-600 specialists + the candidate in the target slot.
- **Primary metric: `marginal_vs_clone = E2 − E1`.**
- Also report `E2 − E0`, per-fold deltas of `E2 − E1`, and folds positive.

For multi-slot hybrids, the matched control uses the first unused seed clones from
`RT-401..RT-406` assigned in **frozen specialist order** — the convention already used by
CSA-03.

**Replacement, never addition.** `W4-E6` tested the union of both arms (13 boosters) and it
scored **−0.00095 (1/5 folds)** against the seven specialists. More members is not better.
This is why the report's insistence on nested replacement over "unconditional eighth-model
averaging" is correct and is adopted here.

### 2.4 Frozen gate ladder

Inherited unchanged from `FULL_OOF_PREREG.md` and CatBoost `PREREG.md`. **Do not move these
after seeing a result.**

| Verdict | Condition |
|---|---|
| `KILL` | `marginal_vs_clone < +0.0010` |
| `INTERESTING` | `≥ +0.0010` |
| `PROMOTION_WORTHY` | `≥ +0.0015`, `≥4/5` folds positive, dominant pair net `> 0` |
| `SERIOUS` | `≥ +0.0030`, `≥4/5` positive, dominant pair net `> 0`, mature-vs-never pair net `> 0` |
| `MAJOR` | `≥ +0.0050` |

**Low correlation alone is never success.** A weaker-standalone model may still survive if
ensemble marginal is strong — and conversely.

### 2.5 Pair-flow diagnostic convention

64 same-`t` pairs per time point, seed `20260827`, on splits: `whole`, `dominant_cell`,
`mature_vs_never`, `mature_vs_prebreak`. Reused unchanged from Learner Diversity 2026 /
CatBoost Specialist Activation 2026 / GPU Tabular 2026 so that every number in this
program is directly comparable to those.

### 2.6 Bookkeeping obligations (mandatory, not cleanup)

- **`research/RESULTS.csv`** — append one row per scoring run. Never edit or delete an
  existing row, including voided ones.
- **`research/EXPERIMENT_ID_MAP.md`** — allocate every `RT-xxxx` here first. **A voided
  experiment keeps its ID.** IDs are never reused or renumbered.
- **`research/RDOF_LEDGER.md`** — record the degrees of freedom frozen before each score.
- **`research/FAILED_EXPERIMENTS.md`** — mandatory entry for every rejected hypothesis:
  what was tried, the falsification condition, why it failed.
- **`research/STATUS.md`** — update only when the production anchor, external score, or
  active research conclusion changes. Keep it to one screen.
- **Do not commit** `.npy` OOF arrays, checkpoints, caches, model binaries, logs, or
  notebook outputs.

### 2.7 Experiment ID allocation

Highest allocated ID anywhere in the repository is **`RT-1259`** (GPU-02 RealMLP).
Verified 2026-08-28 by scanning `EXPERIMENT_ID_MAP.md`, `RESULTS.csv`, `RDOF_LEDGER.md`
and every local worktree: **`RT-1260` and above are entirely unused.**

Proposed allocation (to be confirmed at preregistration time, not before):

| ID | Arm |
|---|---|
| `RT-1260` | CSA-04 CAT-411 |
| `RT-1261` | CSA-04 CAT-412 |
| `RT-1262` | CSA-04 CAT-414 |
| `RT-1263` | CSA-04 CAT-415 |
| `RT-1264` | CSA-05 best-`k` mixed hybrid |
| `RT-1265`–`RT-1269` | reserved for Workstream C (cross-family), unallocated until C opens |
| `RT-1270`+ | reserved for Workstream E (neural), unallocated until E opens |

`RT-1258` / `RT-1259` are **already allocated and must not be reused** — they belong to
GPU-01 / GPU-02 whether or not those runs are ever scored.

---

## 3. AUDIT OF `deep-research-report (4).md`

Recorded here because the next agent will read that report, and it contains one error that
would send them to the wrong place.

### 3.1 What verified correct

| Claim in report | Verification |
|---|---|
| RT-1257 `+0.002407205`, `E2−E0 +0.002026322`, 5/5 folds, dominant net `+158`, mature-vs-never `+73` | **exact** — `reports/catboost_specialist_2026/results.csv` |
| RT-600 production, external 0.6268 | **exact** — `STATUS.md` |
| CRF-01 ladder: representation `+0.0444`, objective `+0.0222` | **exact** — `CRF_FINAL.md` §E |
| CRF-02 fixed-null isolation `−0.021446` | **exact**, and correctly *not* conflated with LA-03's separate `−0.021495290` |
| RT-1258/RT-1259 unresolved, `BINDING_RESULT_PENDING` | **confirmed** — no ledger row, no `results.json`, no OOF on disk |
| SS-01…SS-04 all KILL; RT-1216 `+0.000937`; T2 `+0.00024` | **exact** — `STATUS.md` |
| Nested slot replacement, not eighth-model averaging | **correct**, and independently supported by `W4-E6` (13 boosters, −0.00095) |
| No branch was created, no model trained, no ID consumed | **confirmed** — honest reporting of its own limits |

The report is a genuinely careful piece of work and its refusal to fabricate a branch SHA
or a commit count is the right instinct.

### 3.2 **The error that inverts the next action**

The report states that the binding nested-replacement test for `RT-1258`/`RT-1259` is
blocked because the `RT-600` and `RT-401` control OOF arrays are gitignored and absent
from a cold checkout.

**This is backwards.** The controls are present on this machine:

```
structural-break-learner-diversity-2026/research/oof/
    RT-300.npy  RT-410.npy  RT-411.npy  RT-412.npy
    RT-413.npy  RT-414.npy  RT-415.npy          <- all seven specialists
    RT-401.npy … RT-406.npy                      <- all six seed clones
    RT-1251.npy RT-1254.npy RT-1255.npy RT-1256.npy  <- CatBoost arms
```

Each is `(5036517,) float32`, verified loadable. And
`research/scripts/gpu_tabular/common.py::load_control_oof()` already takes an
`artifact_root` argument, so the controls resolve with a command-line flag.

**What is actually missing is the other side of the comparison:**
`research/oof/gpu_tabular_2026/{tabm,realmlp}/fold_{0..4}_pred.npy` do not exist anywhere
on this machine. A repository-wide search for `fold_*_pred.npy` returns nothing.

So the blocker is not "recover the controls from a local checkout." It is **"bring the
TabM and RealMLP fold predictions back from the Crunch cloud run."** These are different
tasks with different owners and different failure modes. §5 is written accordingly.

### 3.3 Other corrections

**(a) "CRF-01 was correctly killed because it damaged incumbent-correct pairs" — half true.**
CRF-01 was killed at a **standalone** cheap-abandon gate: whole-fold TS-AUC `0.592762`
against a `0.600` necessary condition. `CRF_FINAL.md` states plainly: **no CRF arm ever
produced a `marginal_vs_clone` at all** — folds 1–4 were deliberately never trained. Pair
flow is the *supporting* argument, not the trigger. This matters: the neural family has
never been measured on the binding endpoint, and any future claim about it must say so.

**(b) The report's "novel" damage-penalty loss is not novel — it is SS-02, and SS-02 is KILL.**
The report ranks "train the ranker to penalize reversal of incumbent-correct pairs" as
design variable #4 and experiment FNSR-02, presented as untried. `SS-02`
(`RT-1223`/`RT-1224`) trained a same-`t` residual pair ranker with an explicit
`damage_penalty: 2.0` and dominant-pair weighting. Result: `marginal_vs_clone −0.000290`,
dominant repairs/damage/net `330/438/−108`, mature-vs-never net `−77`, shuffled-control gap
`−0.000087`. The mechanism differs (shallow residual ranker on the incumbent bank vs. a
penalty inside a deep sequence model's loss), so this is not a full closure — but it is
prior art the report does not cite, and it substantially lowers the prior.

**(c) The report reopens hypotheses the repository formally closed.** FNSR is `H-A`
(representation) plus `H-B` (objective) with a longer memory horizon. The CRF factorial
closed both, measured each effect (`+0.0444`, `+0.0222`), and recorded that **their sum is
still an order of magnitude short**. Proposing a longer-memory variant is legitimate
science, but it must be framed as reopening a closed hypothesis with new evidence, not as
a fresh direction.

**(d) The report contradicts the repository's own stated diagnosis and does not engage it.**
`CRF_FINAL.md` says the bottleneck is **arbitration, not detection**. The report's headline
proposal is a better detector. It does not address §1.6 at all.

**(e) The report misses the cheapest live lead entirely.** The CatBoost slot sweep is
**3/7 complete** and the untested four were excluded by a priori preregistration design,
not by evidence. The report does not mention `RT-411`, `RT-412`, `RT-414` or `RT-415` once.
This is Workstream B and it is the highest-EV item in the program.

**(f) Minor: the Mamba / xLSTM / MixturePFN citations are architectural motivation only.**
The report says this itself and is correct to. They are not competition-performance evidence
and carry no weight in the gate ladder.

---

## 4. PROGRAM STRUCTURE AND SEQUENCING

Five workstreams. **A and B run first and are independent of each other** — they can
proceed in parallel if compute allows, because they touch different artifacts and neither
gates the other.

```
  A. Close GPU-01/02 (RT-1258/1259)        ── bookkeeping, ~1h, no training
       │
       │  (result is path-defining for E)
       ▼
  B. CSA-04: complete the CatBoost slot sweep   ── ~1.8 CPU-h, highest EV
       │
       ├─ B passes ──▶ C. Cross-family slot mixing (XGBoost / GPU learners)
       │
       └─ (independent) ──▶ D. Arbitration probe on existing OOF  ── zero training
                                  │
                                  │  (D is the gate for E)
                                  ▼
                            E. Neural sequence models  ── §9, last, gated
```

**Decision rule at each junction:**

| Junction | If | Then |
|---|---|---|
| After A | TabM `marginal_vs_clone ≥ +0.0010` | a third learner family exists → feeds C; raises prior for E |
| After A | both KILL | strong evidence against "new architecture, same 500 features" → **lowers prior for E substantially** |
| After B | best-`k` hybrid `≥ +0.0030` | SERIOUS reached → move to deployment feasibility, then confirmation |
| After B | best-`k` hybrid `< +0.0015` | learner-family mixing is exhausted at 2 families → C becomes the only ensemble lane |
| After D | gating retains repairs under the damage cap | E opens with a concrete mechanism |
| After D | gating cannot retain repairs | **E does not open.** See §9.1 |

---

## 5. WORKSTREAM A — CLOSE GPU-01 / GPU-02 (`RT-1258` / `RT-1259`)

**Type:** evaluation of an already-preregistered, already-executed experiment.
**Not new research.** No new ID is consumed. No model is trained.
**Cost:** ~1 hour, dominated by data transfer.
**Priority:** first, because it is cheap, because it is an outstanding obligation, and
because its result changes the prior on Workstream E.

### 5.1 Why this is an obligation, not an option

`FULL_OOF_PREREG.md` preregistered `RT-1258` (TabM) and `RT-1259` (RealMLP) on 2026-08-27,
allocated both IDs, froze both configurations, and recorded the authorization basis (RTX
4090 benchmark, Crunch submission `76357`, task `run-3e834e0f`; TabM projects to 3.63606 h
for five folds, RealMLP 0.90924 h, combined 4.545298927912005 h against a
15-h/2-learner/0.90-fraction quota). `RDOF_LEDGER.md` records the program with
**"Result not yet filed."**

Leaving a preregistered experiment unfiled is exactly the failure mode preregistration
exists to prevent. It also silently biases the record: an unfiled result is disproportionately
likely to be a null one.

### 5.2 The actual blocker

Per §3.2: the **controls are present**, the **candidate predictions are not**.

Required and missing:
```
research/oof/gpu_tabular_2026/tabm/fold_{0,1,2,3,4}_pred.npy
research/oof/gpu_tabular_2026/tabm/fold_{0,1,2,3,4}_meta.json
research/oof/gpu_tabular_2026/realmlp/fold_{0,1,2,3,4}_pred.npy
research/oof/gpu_tabular_2026/realmlp/fold_{0,1,2,3,4}_meta.json
```

`common.py::load_oof()` validates each fold's prediction shape against
`d.rows_for([fold])` and raises `SystemExit` on mismatch, so a partial or misaligned
recovery will fail loudly rather than silently corrupt the result. Good.

### 5.3 Steps

1. **Recover the fold artifacts** from the Crunch cloud run. The run was engineered
   specifically to survive this: commits `e80f863` ("Isolate each TabM/RealMLP fold in its
   own subprocess after SIGSEGV crash"), `6c45c67` and `0c1a7df` establish per-fold
   checkpointing under `model_directory_path`, persisting fold checkpoints but not the
   8 GB feature cache. Retrieve into the paths above.
2. **Verify completeness before scoring.** All five folds must exist for a learner before
   that learner is evaluated. `evaluate_gpu_oof.py` refuses to run on an incomplete OOF
   vector — do not work around this.
3. **Verify the fold logs contain no forbidden content.** `FULL_OOF_PREREG.md` restricts
   fold completion logs to fold number, runtime, epochs, training/inner-validation
   diagnostics, RAM, VRAM, checkpoint status — **never** TS-AUC, pair flow, replacement
   gain, or outer-fold rank correlation. If a recovered log contains a score, record that
   as a protocol deviation in the final report; do not quietly drop it.
4. **Run the binding evaluation**, pointing at the learner-diversity checkout for controls:
   ```
   python research/scripts/gpu_tabular/evaluate_gpu_oof.py \
       --artifact-root "../structural-break-learner-diversity-2026"
   ```
   Confirm the exact flag name against `evaluate_gpu_oof.py`'s argument parser before
   running; `configure_roots()` is the function that consumes it.
5. **Both candidates must complete before either is evaluated.** Neither may be stopped or
   judged early because the other scored well.
6. **File the results:** `RESULTS.csv` rows for `RT-1258` and `RT-1259`,
   `reports/gpu_tabular_2026/FINAL.md`, `results.json`, `results.csv`, `RDOF_LEDGER.md`
   closure, and — if KILL — a `FAILED_EXPERIMENTS.md` entry each.

### 5.4 What to report

For each of `RT-1258`, `RT-1259`:

- Standalone: mean / pooled / per-fold TS-AUC, fold standard deviation.
- Diversity: within-`t` ρ against `RT-401`.
- Diagnostics: dominant-cell AUC.
- Pair flow: whole, dominant, mature-vs-never, mature-vs-prebreak — repairs / damage / net.
- Binding: `E0`, `E1`, `E2`, `marginal_vs_clone`, `E2 − E0`, per-fold `E2 − E1`, folds positive.
- Verdict against the §2.4 ladder.

### 5.5 Interpretation guide, written in advance

The user-supplied standalone landmarks are **TabM ≈ 0.60065** and **RealMLP ≈ 0.55991**.
These are *not* the answer to the ensemble question and must not be treated as one — T2 /
`RT-995` is the standing proof that strong standalone can collapse to `+0.00024` marginal.

But one comparison is worth stating up front, before the number exists, so it cannot be
constructed after the fact:

**TabM's ≈0.60065 standalone is above the 0.600 bar that CRF-01 failed at 0.592762.** If
TabM nonetheless produces a negligible or negative `marginal_vs_clone`, that is direct
evidence that the 0.600 standalone screen is *not* a sufficient predictor of ensemble value
and, more importantly, that a new architecture on the same 500 features does not buy
retention. That is the single most informative thing this workstream can tell us about
Workstream E, and it is why A precedes E.

### 5.6 Hard constraint carried over

`FULL_OOF_PREREG.md` authorizes **no combination** in that program: not TabM + RealMLP, not
TabM + `RT-1257`, not RealMLP + `RT-1257`, not either + CatBoost — **even if results are
excellent.** Any heterogeneous combination is a separate preregistered experiment. If A
produces a survivor, it becomes an input to Workstream C, not an immediate blend.

---

## 6. WORKSTREAM B — CSA-04: COMPLETE THE CATBOOST SLOT SWEEP

**This is the highest expected-value item in the program.**
**Type:** new experiment, requires preregistration.
**Cost:** ~1.8 CPU-hours of training (estimated from measured runtimes) plus evaluation.
**IDs:** `RT-1260`–`RT-1263` (four slots), `RT-1264` (best-`k` hybrid).

### 6.1 Why this exists — the gap in CSA

The CatBoost Specialist Activation 2026 preregistration deliberately fixed **three** slots
a priori and made two of them conditional:

- `CSA-01` primary: CAT-413 only.
- `CSA-02` conditional on CSA-01 ≥ +0.0010: CAT-300 and CAT-410 only.
- `CSA-03` conditional on ≥2 survivors: hybrid over **surviving slots only**.

That design was correct — it is a proper staged, gated sweep. But its scope was three of
seven slots, chosen before any evidence existed about which slots would respond.
**`RT-411`, `RT-412`, `RT-414` and `RT-415` were never trained with CatBoost.** They were
not excluded by a result; they were excluded by the shape of the original prereg. There is
no killed-idea barrier here.

### 6.2 The measured evidence that this is worth doing

| Arm | Slot replaced | `marginal_vs_clone` | Standalone | ρ vs RT600 | Dominant net | Runtime | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| CAT-413 | `RT-413` (FULL, `pairwise_t`) | **+0.001087151** | 0.620440244 | 0.714196 | 93 | 1527.1 s | INTERESTING |
| CAT-300 | `RT-300` (FULL, champion) | **+0.001029459** | 0.620242061 | 0.737786 | 75 | 2208.1 s | INTERESTING |
| CAT-410 | `RT-410` (3 modules) | +0.000753095 | 0.610328251 | 0.680825 | 17 | 1132.1 s | KILL |
| **HYBRID `RT-1257`** | 413 + 300 | **+0.002407205** | — | — | **158** | — | **PROMOTION_WORTHY** |

Two facts do the work here:

**Fact 1 — the hybrid was slightly super-additive.**
Sum of the two component marginals: `0.001087151 + 0.001029459 = 0.002116610`.
Measured hybrid: `0.002407205`.
**Excess: +0.000290595**, a ratio of **1.137**. Two independent-family replacements did not
merely add; they reinforced. Dominant net behaved the same way: `93 + 75 = 168` against a
measured `158` — slightly *sub*-additive on pair flow while super-additive on the metric,
which is itself worth explaining in the final report.

**Fact 2 — per-slot alpha appears to scale with the module breadth of the slot replaced.**

| Slot | Modules | Cols | `marginal_vs_clone` |
|---|---|---:|---:|
| `RT-413` | FULL | 500 | +0.001087 |
| `RT-300` | FULL | 500 | +0.001029 |
| `RT-410` | 3 modules | narrow | +0.000753 |

Both FULL-bank slots landed at ≈ +0.00105; the narrow slot landed 29% lower and failed the
gate. **`n = 3` is far too small to call this a law**, and it must be preregistered as a
falsifiable prediction rather than assumed — but it generates a concrete, testable ordering
over the untested slots, which is exactly what a good prereg needs.

### 6.3 Preregistered prediction (falsifiable, stated before any CSA-04 score exists)

**Prediction P1.** Slots whose incumbent LightGBM reads the FULL 500-column bank
(`RT-412`, `RT-415`) will yield `marginal_vs_clone` in the neighbourhood of the ≈+0.00105
observed for `RT-413`/`RT-300`. Slots reading a restricted module set (`RT-411` 4 modules,
`RT-414` 3 modules / ~170 cols) will yield materially less, plausibly below the `+0.0010`
gate as `RT-410` did.

**Prediction P2 (competing, and the reason P1 might fail).** The alpha comes from
*heterogeneity*, so replacing the slots whose LightGBM idiosyncrasy is *hardest for
CatBoost to reproduce* should **destroy** more existing diversity than it adds.
`RT-412` (`extra_trees=True` + `sample_mode="per_series"`) and `RT-415` (`boosting="goss"`)
are precisely those slots. Under P2, `RT-412` and `RT-415` underperform P1's expectation and
may be negative.

**P1 and P2 make opposite predictions about the same two slots.** That is what makes CSA-04
a real experiment rather than a sweep. Record both before scoring; the outcome discriminates
between "CatBoost is simply a better learner on wide feature banks" and "the alpha is
irreducibly about family mixing."

**Prediction P3.** The `k`-slot hybrid curve is **non-monotone with an interior maximum**.
At `k = 7` every slot is CatBoost, the ensemble is single-family, and the mixing alpha that
produced `RT-1257` is gone by construction. Therefore there exists some `k* < 7` maximizing
`marginal_vs_clone`. Locating `k*` is the primary scientific deliverable of CSA-04, and it
is a question no amount of additional single-slot testing answers.

### 6.4 Design

**Learner (frozen, zero degrees of freedom).** The exact `RT-1251` CatBoost settings, copied
verbatim from CatBoost `PREREG.md`:

```
iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=5.0,
loss_function=Logloss, eval_metric=Logloss, bootstrap_type=Bernoulli,
subsample=0.7, rsm=0.7, border_count=127, thread_count=2,
allow_writing_files=False, verbose=False
```

**No CatBoost tuning is authorized.** No `depth`, `iterations`, `learning_rate`,
`l2_leaf_reg`, `rsm`, `subsample`, or `border_count` search. This is the same learner that
produced `RT-1254`/`RT-1255`/`RT-1256`, which is what makes CSA-04's numbers directly
comparable to CSA-01/02 rather than a fresh, incomparable sweep.

**Per-arm contract.** For each arm, take the incumbent slot's modules, rows, labels, folds,
row-sampler mode, `max_train_rows` cap and **incumbent training seed** exactly as recorded
in §1.2, and vary **only the learner family**.

| ID | Arm | Slot | Modules | `max_train_rows` | `random_seed` | Sampler |
|---|---|---|---|---:|---:|---|
| `RT-1260` | CAT-411 | `RT-411` | `m02_dist,m03_dyn,m04_resid,m06_loc` | 900,000 | 1 | uniform |
| `RT-1261` | CAT-412 | `RT-412` | FULL | 900,000 | 7 | **`per_series`** |
| `RT-1262` | CAT-414 | `RT-414` | `m07_bayes,m06_loc,m01_seq` | 700,000 | 3 | uniform |
| `RT-1263` | CAT-415 | `RT-415` | FULL | 700,000 | 11 | uniform |

**Open specification item — resolve before preregistering, not after.** `RT-412` uses
`sample_mode="per_series"` and `RT-415` uses GOSS. The row-sampling mode is part of the
*data contract* and must be preserved (CAT-412 gets per-series sampling). GOSS is a
*LightGBM boosting mechanism* with no CatBoost equivalent and is therefore simply not
reproduced — CAT-415 uses the frozen Bernoulli/`subsample=0.7` bootstrap like every other
CatBoost arm. **State this explicitly in the prereg**, because it means CAT-415 is a less
faithful slot reimplementation than CAT-411/412/414, and that asymmetry must be disclosed
before its number exists rather than invoked afterwards to explain a bad result.

Note also that CSA-01/02 used a **uniform** row sampler for all three arms including
CAT-300, whereas the GPU program used the **RT-401 sequential sampler, seed 1**. CSA-04
follows the CatBoost convention (uniform, incumbent seed) for comparability with
`RT-1254`–`RT-1256`. Do not mix conventions within a program.

**Training order.** `RT-1260`, `RT-1261`, `RT-1262`, `RT-1263`, all five folds each, in a
deterministic resumable order. **All four must complete before any of the four is
evaluated.** No arm may be stopped early on a predictive-score basis; only on unrecoverable
technical failure, which must be recorded.

**Log hygiene.** Fold logs may record fold number, runtime, rows, RAM, and checkpoint status.
**No TS-AUC, no pair flow, no replacement gain, no rank correlation** in any per-fold log.

### 6.5 Evaluation

**Stage 1 — four independent single-slot replacements.** For each arm, the standard
`E0`/`E1`/`E2` battery of §2.3, with the matched control being `RT-401` inserted into the
same slot. Report the full §5.4 metric set. Apply the §2.4 ladder.

**Stage 2 — the hybrid curve (the actual deliverable).** Let `S` be the set of slots with
`marginal_vs_clone ≥ +0.0010` across **all six tested slots** (CAT-300, CAT-410, CAT-413
from CSA, plus the four new). Construct hybrids at every `k` from 2 up to `|S|`, adding
slots in descending order of measured single-slot marginal. For each `k`:

- `E2_k` = seven-member ensemble with the top-`k` surviving slots CatBoost-replaced.
- `E1_k` = matched control with the **same `k` slots** replaced by seed clones drawn from
  `RT-401..RT-406` in frozen specialist order.
- Report `marginal_vs_clone`, `E2 − E0`, per-fold deltas, folds positive, and all four
  pair-flow splits.

The hybrid at `k = 2` **must reproduce `RT-1257`'s `+0.002407205` exactly** if the slot
ordering puts 413 and 300 first. **Treat that as a mandatory regression check**: if it does
not reproduce, something in the evaluation harness has drifted and no CSA-04 number is
trustworthy until it is explained.

Allocate **`RT-1264`** to the best-`k` hybrid.

**No blend-weight optimization at any `k`.** Equal weight, fold-pure `SCDF_NSEEN`, always.

### 6.6 Expected value, stated honestly with its downside

**Upside case.** If P1 holds and `RT-412`/`RT-415` deliver ≈ +0.00105 each, a 4-slot hybrid
at the observed 1.137 super-additivity ratio lands near
`(0.001087 + 0.001029 + 0.00105 + 0.00105) × 1.137 ≈ +0.0048` — inside `MAJOR` territory
(`≥ +0.0050`) or immediately below it, and comfortably past `SERIOUS` (`+0.0030`).

**Central case.** Super-additivity decays as slots are consumed (P3), the narrow slots
(`RT-411`, `RT-414`) fail the gate as `RT-410` did, and `k*` lands at 3–4 slots for a hybrid
around **+0.0032 to +0.0040**. This still crosses `SERIOUS` and would be the **largest
measured marginal in the project's recorded history** — roughly 1.5× `RT-1257`.

**Downside case.** P2 dominates: replacing `RT-412`/`RT-415` destroys the extra-trees and
GOSS idiosyncrasies that were carrying real diversity, all four new slots come in below the
gate, `k* = 2`, and CSA-04 confirms `RT-1257` is already the peak. **This is still a valuable
result** — it converts "we stopped at three slots because the prereg said three" into
"learner-family mixing is measured and exhausted at two slots," which is a real closure and
directly justifies redirecting budget to Workstream C or E.

**There is no outcome of CSA-04 that leaves us where we started.** That is the property
that makes it the right first experiment.

### 6.7 Cost and risk

- **Compute:** measured CSA runtimes were 1527.1 s, 2208.1 s, 1132.1 s for three arms
  (4867.3 s total). Four more arms at comparable scale ≈ **1.6–1.9 CPU-hours**. Trivial.
- **Research degrees of freedom:** four new arms, all configurations inherited, **zero
  tuning knobs**. This is about as DOF-cheap as an experiment gets, which matters given the
  `RDOF_LEDGER.md` framing that "+0.0005" must be read against the number of chances taken.
- **Risk — multiple comparisons.** Four more single-slot tests against a `+0.0010` gate.
  Mitigation: the gate is fixed in advance, all four are trained before any is scored, and
  the **binding deliverable is the hybrid curve**, not any individual slot. A single slot
  squeaking over `+0.0010` is not a result; the hybrid is.
- **Risk — deployment cost.** Every surviving CatBoost slot adds one CatBoost model to
  inference. `reports/catboost_specialist_2026/FINAL.md` already flags that `RT-1257` needs
  separate deployment inference benchmarking, and that work is **not done**. A 4-slot hybrid
  quadruples that exposure. **Deployment feasibility must be measured before any promotion
  claim**, and the streaming/online inference budget is a hard constraint, not a footnote.
  See §10.

---

## 7. WORKSTREAM C — CROSS-FAMILY SLOT MIXING

**Type:** new experiment, requires preregistration.
**Status:** **conditional.** Opens only on B, and shaped by A.
**IDs:** `RT-1265`–`RT-1269` reserved, unallocated until it opens.

### 7.1 Rationale

If B confirms the mechanism, the natural generalization is that the ensemble is not a set of
seven LightGBM models to be upgraded, but **seven functional slots to be filled by whichever
family is most complementary in that slot.** `RT-1257` is a 2-family mixture. There is a
`research/xgb-gpu-2026` branch in the remote snapshot, and Workstream A may deliver a third
and fourth family (TabM, RealMLP).

The design question is a **per-slot family assignment**, and the search space is combinatorially
large — which is exactly why it needs a tight prereg rather than an open sweep.

### 7.2 Opening conditions

C opens **only if both**:

1. B's best-`k` hybrid reaches `≥ +0.0015` with `≥4/5` folds positive and positive dominant
   pair net, **and**
2. At least one non-LightGBM, non-CatBoost family has a measured single-slot
   `marginal_vs_clone ≥ +0.0010` (from A, or from a separate XGBoost arm).

If B's hybrid is KILL, C does not open — the mechanism is falsified at its source and adding
families to a dead mechanism is fishing.

### 7.3 Constraint on scope

C must be preregistered as a **fixed, enumerated set of assignments** — not a search. The
`RDOF_LEDGER.md` discipline is the whole reason this project's numbers mean anything, and a
free per-slot family search over 7 slots × 4 families is 16,384 configurations. Any such
search would need its own multiplicity correction and would almost certainly not survive it.

A defensible C-scope: assign each slot the family with the highest *measured single-slot*
marginal for that slot, evaluate that one assignment plus its matched clone control, and stop.
One configuration, chosen by pre-existing measurements, zero search.

---

## 8. WORKSTREAM D — ARBITRATION PROBE ON EXISTING OOF

**Type:** descriptive diagnostic on existing artifacts. **Zero training.**
**Cost:** near-zero — minutes of CPU.
**IDs:** none consumed if it stays descriptive (same basis as CSA-00, which was
"descriptive and ID-free because it uses existing OOF vectors only").
**Purpose:** this is the **gate for Workstream E.**

### 8.1 The question

§1.6 established that the measured bottleneck is retention, not detection. CRF-01 repairs
39.5% of dominant-cell mistakes with a repair-Jaccard of 0.174 against the seed clone — those
repairs are real and are its own — while damaging 26.8% of RT-600-correct pairs against a
0.0150 cap.

**The question D answers:** does there exist *any* gating, abstention, or confidence-weighted
combination of CRF-01's existing predictions with RT-600 that retains a material fraction of
those repairs while bringing the damage rate near the cap?

This is a question about a **combination rule over frozen prediction vectors**, not about a
model. It costs nothing to answer.

### 8.2 Why it is answerable right now

`structural-break-causal-representation-frontier/research/oof/` contains:

- `RT-1234.npy` — CRF-01 NNCSR candidate, `(5036517,)` float32, **806,334 finite rows
  (16.01%) = fold 0 only.** Folds 1–4 were deliberately never trained, so this is a
  fold-0-only study by construction.
- `RT-1235.npy` — the matched BCE-objective control, same coverage.
- `RT-1240.npy`, `RT-1241.npy`, `RT-1242.npy` — CRF-02 candidate, fixed-null control,
  deranged control.
- `RT-300.npy`, `RT-401.npy`, `RT-410.npy`–`RT-415.npy` — the full RT-600 control set.

Everything needed is already on disk.

### 8.3 Design

**Population:** fold 0 only. **State this limitation everywhere the result is cited.** A
single-fold descriptive diagnostic is not a confirmation and cannot promote anything.

**Arms — a small, fixed, preregistered set of gating rules.** Illustrative, to be finalized
in the prereg:

1. **Baseline:** unconditional equal-weight blend of `RT-1234` into the RT-600 slot
   (reproduces the known-bad behaviour; establishes the reference).
2. **Confidence-gated:** use `RT-1234` only where its calibrated score is in the top/bottom
   `q` quantiles of its own within-`t` distribution; otherwise defer entirely to RT-600.
   Sweep `q` over a **preregistered fixed grid**.
3. **Agreement-gated:** use `RT-1234` only where it agrees in sign of deviation with RT-600,
   or only where RT-600's own within-`t` score is near its decision boundary.
4. **Cell-restricted:** apply `RT-1234` only inside the dominant cell (`t ≥ 200`, break age
   `≥ 100`), deferring to RT-600 everywhere else.
5. **Abstention:** an explicit three-way output — repair, defer, abstain — with the abstain
   region tuned to the 0.0150 damage cap.

**Primary readout.** For each rule, on fold 0, in the dominant cell:
repair count, damage count, net, **damage rate on RT600-correct pairs against the 0.0150
cap**, and the resulting fold-0 `E2 − E1`.

**Success criterion, fixed in advance.** A rule "works" if it retains **≥ 50%** of the
unconditional repair count while bringing the pre-break damage rate on RT600-correct pairs
**below 0.05** — still over the 0.0150 cap, but within an order of magnitude of it rather
than the 17× overshoot currently measured. This is a deliberately generous bar because D is
a *screen*, not a confirmation.

### 8.4 Prior art and honest expectations

**The prior here is poor and must be stated.** `SS-01` (repair-damage arbitration),
`SS-02` (residual ranking with a `2.0` damage penalty), `SS-03` (null calibration) and
`SS-04` (specialist-disagreement routing) were all built to solve arbitration and all four
are KILL, at `−0.000310`, `−0.000290`, `−0.000299`, `−0.000312`. `FIRST_SWEEP_SYNTHESIS.md`
§H4 records "static error-manifold routing is weak."

**What is different here:** all four SS arms arbitrated over candidates built on the
*incumbent 500-column bank*, at ρ 0.86–0.89 against RT-600. `RT-1234` sits at **ρ 0.446** with
repair-Jaccard 0.174 — genuinely different error geometry, which is the one input the SS
family never had. That is a real distinction, and it is also the *only* argument for D. If D
fails, the honest reading is that arbitration is hard **regardless** of how different the
candidate is, and that is decisive information about Workstream E.

**D is cheap enough that its poor prior does not matter.** It costs a few minutes and it is
the difference between funding §9 on evidence and funding it on architecture enthusiasm.

---

## 9. WORKSTREAM E — NEURAL SEQUENCE MODELS

**Placed last deliberately. Read §9.1 before reading §9.3.**

### 9.1 Why this is last, and exactly what would move it up

This is not a dismissal of neural models. It is a ranking derived from what has been
measured in this repository.

**The evidence against funding a new neural architecture right now:**

1. **The representation × objective factorial is complete and empty.** `CRF_FINAL.md` §E
   fills every cell. Representation effect `+0.0444`, objective effect `+0.0222`, total
   `+0.0666` over `RT-970`'s 0.52618 — reaching **0.59276**, still below the `0.600`
   necessary condition and roughly `0.043` below the fitted `+0.0030` contour at ρ ≈ 0.45.
   `H-A` and `H-B` are closed.
2. **Learned nulls are closed.** `H-E`. The fixed AR(5) + 256-knot per-series residual ECDF
   beat a correctly-conditioning learned null by `0.021446`, with the derangement control
   **passing** (`+0.003377` whole, `+0.008391` dominant) — so the learned null was not broken;
   it conditioned correctly and lost anyway. The failure is amortization.
3. **The information frontier says the missing signal is post-`t`.** `W7-D3R` Arm B, same
   information with more capacity, was **negative**. `H-D` — the practical limit is the legal
   prefix, and the residual gap is predominantly post-`t` — is what survives.
4. **The measured bottleneck is arbitration, not detection** (§1.6). A better detector does
   not address it.
5. **`corr(standalone, ρ) = +0.983` across 17 arms.** Nothing this project has built has ever
   been both good and different, and `RT-1234` moving that frontier by `+0.0092` did not
   change the answer.
6. **The base rate is bad and the cost is high.** CRF-01 required 24 green preflight gates,
   bitwise prefix-invariance proof, a fold-purity positive control reproducing a known
   contaminated scheme, and a checkpoint provenance fingerprint added after `RT-1237/1238/1239`
   were voided by a unit-test null. That is the correct standard, and it is the real cost of a
   neural arm — the training compute is the cheap part.

**What would move E up, concretely:**

| Trigger | Effect |
|---|---|
| **D succeeds** — a gating rule retains ≥50% of CRF-01's repairs under a <0.05 damage rate | **E opens.** We would have a demonstrated retention mechanism, and a better detector feeding it becomes worth building. This is the main path. |
| **A returns TabM `≥ +0.0010`** | Raises the prior that a non-tree family can extract retainable alpha from the same features. Not sufficient alone. |
| **B returns KILL across all four slots and `k* = 2`** | Learner-family mixing is exhausted; the ensemble lane closes and E becomes the *least bad* remaining option by elimination rather than by evidence. Say so explicitly if this happens. |
| A returns both KILL **and** D fails | **E should not be funded.** Two independent lines say new architectures on this information do not retain. Redirect to deployment robustness, per `CRF_FINAL.md`'s own recommendation. |

### 9.2 Standing constraints on any neural arm

Any arm in E, whatever the architecture, inherits all of §2 plus:

- **Preserve the fixed per-series null.** `H-E` is closed. The AR(5) + 256-knot historical
  residual ECDF is paid for by each series' own break-free history at zero generalization
  cost. Expose its residual / PIT / exceedance streams directly as input channels. **Do not
  compress it into a small learned global embedding** — CRF-02 measured that trade at
  `−0.021446` and it is not to be repeated.
- **Same-`t` pairwise ranking objective**, with negative sampling matching the TS-AUC
  comparison structure. The `+0.0222` whole / `+0.0371` dominant objective effect is
  measured and real; use it, do not re-derive it.
- **Full CRF-01-grade preflight before any score.** Bitwise prefix invariance at `atol=0.0`,
  truncation and batch-composition invariance, a fold-purity positive control, a checkpoint
  provenance fingerprint that halts the run on mismatch, and test-isolated caches.
- **The cheap abandon gate stays.** Fold-0 standalone `< 0.600` → abandon, folds 1–4 not
  trained. This gate has now fired twice and saved substantial compute both times.
- **Binding endpoint is `marginal_vs_clone` under nested replacement**, not standalone
  TS-AUC and not low ρ. Note explicitly that **no CRF arm ever reached this endpoint** —
  an E arm that clears the abandon gate would be the first neural candidate ever measured
  on it.

### 9.3 Candidate architectures, ranked

Reproduced from the deep-research report with the corrections of §3.3 applied.

**E-1 — Fixed-Null Selective State-Space Ranker (Mamba-family).**
*The report's headline proposal.* Causal selective SSM over the null-normalized sequence
channels, input-dependent state parameters allowing content-dependent retention and
forgetting, linear sequence scaling, trained under the same-`t` pairwise objective on top of
the preserved fixed per-series null.

*The case for:* CRF-01's TCN had receptive field ≈ 253. If the missing structure is
long-horizon excursion evolution conditional on a strong individualized null, a finite
convolutional field cannot see it and a selective recurrence can. Mamba / Mamba-3 are
architectural motivation only — **not competition-performance evidence**, and they carry no
weight in the gate ladder.

*The case against, which must be in the prereg:* this is `H-A` reopened. `W7-D3R` Arm B says
more capacity on the same information is negative. `H-D` says the residual gap is post-`t`,
and no amount of *causal* memory reaches post-`t` information. And §1.6 says the failure was
retention, which memory horizon does not obviously fix.

*Required framing:* E-1 must be preregistered as **"does longer effective memory change the
retention geometry?"** — with pair flow and damage rate as co-primary readouts alongside
standalone — not as "does a better representation raise standalone TS-AUC?" That second
question is answered and the answer is no.

**E-2 — Memory isolation control.**
Same channels, same objective, same head; deliberately memory-truncated recurrence versus the
full-state version. This is what establishes whether "long memory" is actually load-bearing
or whether E-1's result (either sign) came from something else. **If E-1 runs, E-2 is not
optional** — the CRF program's whole value came from its matched ladders
(`RT-970 → RT-1235 → RT-1234`), and an E-1 without its isolation control would be a strictly
weaker piece of evidence than CRF-01 was.

**E-3 — Repair-preserving objective term.**
An explicit penalty on reversal of incumbent-correct training-fold pairs, inside the loss.
**Note the prior art the report missed:** `SS-02` already ran a same-`t` residual pair loss
with `damage_penalty: 2.0` and dominant-pair weighting, scoring `−0.000290` with dominant
net `−108`. E-3 is not novel; it is SS-02's mechanism relocated into a deep sequence model's
training loop. That relocation is a real difference and may matter — but the prereg must cite
SS-02 and say why this time is different, and E-3 should only run if E-1 clears the abandon
gate with a non-trivial standalone signal, holding representation fixed.

*Constraint:* outer-fold RT-600 predictions may enter only where the protocol permits, for
training-objective and control construction, under strict nested fold purity. They must never
leak validation ranks or future information. Run the fold-purity sentinel.

**E-4 — xLSTM / state-augmented recurrent family.** Lower priority. The evidence alignment to
this task is weaker than for selective SSMs and there is no mechanism-level argument specific
to this problem. Would only be worth running as a second architecture *after* E-1 established
that the memory axis matters at all.

**E-5 — Mixture / sparse-expert tabular in-context learning (MixturePFN family).** Treat as a
research comparator, not a bet. The MixturePFN literature itself flags scaling difficulty as
dataset size grows, and this competition has million-row folds and hard streaming inference
constraints. It is unlikely to survive deployment feasibility even if it scored well.

### 9.4 Explicitly out of scope for E

Carried forward from the report's "five lanes least worth tuning" and the repository's own
`FAILED_EXPERIMENTS.md`:

- Raw CRF-01-style TCN width / depth / epoch sweeps. The ladder is measured; scaling it is
  not a new hypothesis.
- Tiny amortized learned-null embeddings. `H-E` is closed at `−0.021446`.
- Shallow or static post-hoc repair routers. SS-01 through SS-04, all KILL.
- Another iteration of future-aware distillation. Wave 8's five pilots (ORR, TGMC, SST, PCFB,
  CFEP) are all KILL on the full population.
- Generic TabM / RealMLP hyperparameter search **before** `RT-1258`/`RT-1259`'s binding
  replacement result exists. This is Workstream A's entire point.
- `CRF-04`. `CRF_PROGRAM_PREREG.md` §0.8/§0.9 forbid inventing one, and `CRF-03` did not open
  because neither primary produced a `marginal_vs_clone` at all. Anything in E is a **new
  program with a new preregistration**, not a CRF continuation.

---

## 10. DEPLOYMENT FEASIBILITY — THE CONSTRAINT EVERY WORKSTREAM SHARES

Currently **unmeasured**, and it gates every promotion claim in this plan.

`reports/catboost_specialist_2026/FINAL.md` closes with: *"inference adds one CatBoost model
per surviving replacement and needs separate deployment benchmarking"* and *"Next action:
review surviving CatBoost specialist(s) against deployment cost before any production work."*
**That review has not happened.** `RT-1257` is PROMOTION_WORTHY and undeployed.

This matters more, not less, as B succeeds: a 4-slot hybrid quadruples the CatBoost inference
exposure relative to `RT-1257`. And it applies to E with far greater force — a selective SSM
running causally at every online time step is a fundamentally different inference profile from
a gradient-boosted tree, and the `research/xgb-gpu-2026` / GPU-tabular benchmarking work
exists precisely because compute budgets in this competition are binding.

**Rule for this program:** no arm is described as promotable until its **online, streaming,
per-timestep inference cost** has been measured against the competition budget, using the same
methodology as the RTX 4090 hardware benchmark (Crunch submission `76357`, task
`run-3e834e0f`). A model that cannot run online is not a candidate regardless of its
`marginal_vs_clone`.

---

## 11. RISK REGISTER

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Multiple comparisons across 4 new CSA-04 slots manufacture a false `+0.0010` | Medium | High | Gate frozen in advance; all four trained before any scored; **hybrid curve is the binding deliverable**, not any single slot; RDOF logged |
| R2 | CSA-04 hybrid does not reproduce `RT-1257` at `k = 2` | Low | **Critical** | Mandatory regression check (§6.5). If it fails, **halt** — no CSA-04 number is trustworthy until explained |
| R3 | GPU fold artifacts unrecoverable from the cloud run | Medium | Medium | Per-fold checkpointing was engineered for exactly this (`e80f863`, `6c45c67`, `0c1a7df`). If truly lost, the honest outcome is a recorded `UNRECOVERABLE` in `RESULTS.csv` and `RDOF_LEDGER.md`, **not** a re-run under a quietly amended prereg |
| R4 | A recovered GPU fold log contains a forbidden TS-AUC | Low | Medium | Inspect logs before evaluation; record any deviation in the final report rather than dropping it |
| R5 | Deployment cost kills a scientifically successful hybrid | **Medium-High** | High | Measure §10 **early**, ideally in parallel with B's training, not after |
| R6 | Neural arm consumes weeks and reproduces CRF-01 | Medium | High | E is gated on D (§9.1); E-2 isolation control mandatory; cheap abandon gate at 0.600 retained |
| R7 | Slot-replacement alpha is partly a bagging artifact rather than family heterogeneity | Medium | Medium | `FINAL_ARCHITECTURE_FREEZE.md` already measured that `+0.00499` of the `+0.00832` specialist delta is reproducible by seed variation alone. The `E1` seed-clone control is exactly the right control for this and is already in the protocol — but say so in the report rather than assuming the reader knows |
| R8 | A voided run silently uses a wrong artifact (the `RT-1237/1238/1239` failure mode) | Low | **Critical** | Checkpoint provenance fingerprints; compute accounting sanity check (`pretrain_runtime_s = 0.2` vs a real 1,364.9 s is what caught it); test-isolated caches |
| R9 | Two agents work concurrently in different worktrees and diverge | Medium | Medium | `AGENTS.md` §"Branch / worktree rules"; `RESULTS.csv` appends under a file lock; never force-push |
| R10 | This program's own scope creeps into an unbounded family × slot search | Medium | High | §7.3 caps Workstream C at one enumerated assignment; every workstream needs its own prereg |

---

## 12. DEFINITION OF DONE

The program is complete when **all** of the following hold:

- [ ] `RT-1258` and `RT-1259` have filed rows in `research/RESULTS.csv` with a
      `marginal_vs_clone`, or a recorded `UNRECOVERABLE` status with the reason.
- [ ] `reports/gpu_tabular_2026/FINAL.md` exists with the full §5.4 metric set for both arms.
- [ ] `CSA04_PREREG.md` is committed **before** any CSA-04 score exists, containing
      predictions P1, P2 and P3 as written in §6.3.
- [ ] `RT-1260`–`RT-1263` are trained on all five folds, all four complete before any is
      scored, and all four have `RESULTS.csv` rows.
- [ ] The `k = 2` hybrid regression check reproduces `+0.002407205`.
- [ ] The full hybrid curve over `k` is reported, `k*` identified, and `RT-1264` allocated
      to the best-`k` hybrid.
- [ ] P1 / P2 / P3 are each explicitly adjudicated in the final report — including the ones
      that were wrong.
- [ ] Deployment inference cost is measured for the best surviving configuration (§10).
- [ ] `RDOF_LEDGER.md` records every frozen degree of freedom for every arm.
- [ ] `FAILED_EXPERIMENTS.md` has an entry for every KILL, with the falsification condition
      and why it failed.
- [ ] `STATUS.md` updated **only if** the production anchor, external score, or active
      research conclusion actually changed.
- [ ] Workstream D executed and reported, and the E-opening decision recorded **with its
      reasoning**, whichever way it goes.
- [ ] No `.npy`, checkpoint, cache, or notebook output committed.

---

## 13. OPEN QUESTIONS TO RESOLVE BEFORE PREREGISTERING

These are genuine unknowns, not rhetorical. Each should be answered in the relevant prereg.

1. **Where exactly do the GPU fold artifacts live**, and is the Crunch cloud run's output
   still retrievable? This determines whether Workstream A is one hour or a dead end.
2. **Confirm `evaluate_gpu_oof.py`'s argument name** for the control root — read the parser
   rather than assuming `--artifact-root`.
3. **CAT-415 and GOSS.** Confirmed above that GOSS has no CatBoost analogue and the frozen
   Bernoulli bootstrap will be used. Does that make CAT-415 a materially unfaithful slot
   reimplementation, and if so, should it be reported under a caveat or excluded from the
   hybrid? **Decide before scoring.**
4. **CAT-412 and `per_series` sampling.** Confirm the CatBoost runner actually honours
   `sample_mode="per_series"`; if it does not, that is a specification change and must be
   disclosed in the prereg, not patched afterwards.
5. **Hybrid slot ordering.** §6.5 adds slots in descending single-slot marginal. Is that the
   right ordering, or should it be by dominant pair net (which `FIRST_SWEEP_SYNTHESIS.md`
   found correlates with `marginal_vs_clone` at Pearson 0.879)? **Pick one and freeze it** —
   trying both is a degree of freedom.
6. **Does `RT-411`'s module set overlap with what CatBoost is good at?** `RT-411` reads
   `m02_dist, m03_dyn, m04_resid, m06_loc` — no `m00_core`, no `m07_bayes`. Worth a
   descriptive look at the CSA feature-importance record before predicting its outcome.
7. **What is the actual competition deadline and remaining submission budget?** Not recorded
   in `STATUS.md` or `PROTOCOL.md`. This determines how many of these workstreams are
   realistically fundable and should be established before committing to C or E.
8. **Is `research/xgb-gpu-2026` a live branch with usable artifacts**, or a stub? It appears
   in the remote snapshot but has no local worktree. Workstream C's viability depends on it.

---

## 14. ONE-PARAGRAPH SUMMARY FOR THE NEXT AGENT

Production is `RT-600` at external **0.6268** and has not moved. The best measured research
result is `RT-1257`, a **two-slot** LightGBM/CatBoost hybrid at `marginal_vs_clone
+0.002407205`, 5/5 folds positive, dominant net `+158` — PROMOTION_WORTHY, not SERIOUS, and
not yet deployment-benchmarked. **Do three things, in this order.** First, recover the TabM
and RealMLP fold predictions from the Crunch cloud run and file `RT-1258`/`RT-1259` — the
control OOF vectors are **not** missing, they are in
`structural-break-learner-diversity-2026/research/oof/`, and the deep-research report is
wrong about this. Second, preregister and run **CSA-04**: the CatBoost slot sweep is only
**3 of 7 complete**, `RT-411`/`RT-412`/`RT-414`/`RT-415` were excluded by prereg design rather
than by evidence, it costs under two CPU-hours, and the real deliverable is the hybrid curve
over `k` and the location of its interior maximum. Third, run the **zero-cost arbitration
probe** on `RT-1234`'s existing fold-0 OOF, because the repository's own diagnosis is that the
bottleneck is retention rather than detection, and that probe is the gate on whether any new
neural architecture is worth funding at all. **Do not start a Mamba.** Not because it is a bad
idea — because `H-A`, `H-B` and `H-E` are closed, the measured failure is that candidates
cannot keep the repairs they find, and there is a two-CPU-hour experiment sitting untouched
that has already returned `+0.0024` from half of its slots.
