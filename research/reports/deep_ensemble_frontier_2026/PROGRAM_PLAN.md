# DEEP ENSEMBLE FRONTIER 2026 — PROGRAM PLAN (SHARED CORE)

**Status:** `PLAN_ONLY — NO EXPERIMENT AUTHORIZED BY THIS DOCUMENT`
**Branch:** `research/deep-ensemble-frontier-2026`
**Base:** `research/gpu-tabular-2026@0c1a7df` ("Persist only fold checkpoints, not the 8 GB feature cache")
**Worktree:** `/path/to/workspace/structural-break-deep-ensemble-frontier-2026`
**Written:** 2026-08-28

> **This program runs as TWO PARALLEL LANES with two agents.**
> This file is the **shared core**: verified state, binding constraints, the audit of the
> deep-research report, the lane split, and the coordination protocol. **Both agents read
> this file in full.** Then each agent reads exactly one lane file and works only in its own
> worktree:
>
> | Lane | Agent reads | Works in | Hardware |
> |---|---|---|---|
> | **LOCAL** | [`LANE_LOCAL.md`](LANE_LOCAL.md) | `research/deep-ensemble-frontier-local-2026` | this M2 Pro — 10 cores, 16 GB, 12.61 GB feature cache on disk |
> | **CRUNCH** | [`LANE_CRUNCH.md`](LANE_CRUNCH.md) | `research/deep-ensemble-frontier-crunch-2026` | Crunch cloud — RTX 4090, 15-h/2-learner quota |
>
> **Neural sequence models are the last item in the CRUNCH lane** (`LANE_CRUNCH.md` §C4)
> and are gated on a local result. See §6.3 and `LANE_CRUNCH.md` §C4.1.

---

## 0. HOW TO READ THIS DOCUMENT, AND WHAT IT IS NOT

This is a **program plan**, not a preregistration. It is deliberately verbose because the
failure mode this project keeps hitting is not "we lacked an idea" — it is "we ran an idea
whose falsification condition, control, and cost were only half-specified, and then had to
argue afterwards about what the number meant." Every workstream in the two lane files is
written out to the point where its `*_PREREG.md` can be produced almost mechanically.

**This document authorizes nothing.** Under `AGENTS.md` §"Preregistration-before-score rule",
no run that can produce a headline number may begin until a separate preregistration file
exists, is committed, and the commit predates the score.

Three things this is explicitly *not*:

1. **Not a promotion decision.** Production remains `RT-600` on `production/rt600` (external
   Crunch leaderboard **0.6268**). Nothing here changes that without clearing the promotion
   battery in `AGENTS.md` §"What constitutes a valid promotion".
2. **Not a re-litigation of closed hypotheses.** `H-A` (representation saturation), `H-B`
   (objective mismatch) and `H-E` (learned-null misspecification) are closed by the Causal
   Representation Frontier program. Where the CRUNCH lane touches near them it does so under
   an explicitly different question, with the closure stated and respected.
3. **Not an endorsement of `deep-research-report (4).md`.** §3 records which of its claims
   survived verification and which did not. One central operational claim is **wrong in a way
   that inverts the next action** — see §3.2. Do not act on that report without reading §3.

---

## 1. VERIFIED STATE OF THE WORLD

Read off live repository state on 2026-08-28, not quoted from a prior report. `AGENTS.md`
§"Verify branch/SHA instead of trusting stale prompts" requires this, and it caught a real
error (§3.2).

### 1.1 Production and scoring anchors

| Item | Value | Source |
|---|---|---|
| Competition | ADIA Lab / CrunchDAO Structural Break Challenge — **Real-Time Edition** | `research/STATUS.md` |
| Metric | Time-Stratified AUC (`sbr.metric.ts_auc_flat`); one prediction per online step | `research/STATUS.md` |
| Causality constraint | prediction at `t` uses only data at or before `t`; bitwise prefix invariance at `atol=0.0` | `AGENTS.md` §"Causality rules" |
| Production anchor | `RT-600`, seven-specialist SCDF blend, `production/rt600` | `research/STATUS.md` |
| External score | **0.6268** (LB-001) | `research/STATUS.md`, `EXPERIMENT_ID_MAP.md` |
| RT-600 dev OOF (canonical 5 folds) | 0.62581, re-verified exact (Δ +0.000001) | `LEADERBOARD_ASSAULT_STATUS.md` |
| RT-600 as `E0` in the replacement battery | **0.638276** | `CRF_FINAL.md` §B |
| Matched exchangeable clone `E1` | 0.638586 | `CRF_FINAL.md` §B |

**0.62581** is whole-dev pooled OOF. **0.638276** is `E0` on the nested-replacement battery
population. They are not interchangeable, and mixing them is a recurring source of confusion
in prior reports. **Every `marginal_vs_clone` in this program is on the battery population.**

### 1.2 The seven-specialist roster — the object both lanes operate on

Reconstructed from `research/scripts/wave2_streams.py::JOBS` and
`research/scripts/wave4_lib.py::SPECIALIST_ALIAS`. Column counts derived from the feature
cache manifests (§1.7); `FULL` = 500 columns exactly.

| Slot | Alias | Modules | Cols | Seed | `max_train_rows` | Distinguishing LightGBM idiosyncrasy | CatBoost status |
|---|---|---|---:|---:|---:|---|---|
| `RT-300` | `RT-100R` | FULL | **500** | 0 | 1,000,000 | the champion configuration itself | **tested** — CAT-300, `+0.001029459` |
| `RT-410` | `RT-120R` | `m00_core,m01_seq,m07_bayes` | **261** | 0 | 900,000 | `num_leaves=127`, `feature_fraction=0.35` | **tested** — CAT-410, `+0.000753095` **KILL** |
| `RT-411` | `RT-121R` | `m02_dist,m03_dyn,m04_resid,m06_loc` | **239** | 1 | 900,000 | `num_leaves=31`, `feature_fraction=0.7` — shallow | **UNTESTED** |
| `RT-412` | `RT-122R` | FULL | **500** | 7 | 900,000 | `num_leaves=255`, `extra_trees=True`, **`sample_mode="per_series"`** | **UNTESTED** |
| `RT-413` | `RT-123R` | FULL | **500** | 0 | 700,000 | **`objective="pairwise_t"`**, groups = online index `t` | **tested** — CAT-413, `+0.001087151` (best) |
| `RT-414` | `RT-124R` | `m07_bayes,m06_loc,m01_seq` | **170** | 3 | 700,000 | no window-bank features at all | **UNTESTED** |
| `RT-415` | `RT-125R` | FULL | **500** | 11 | 700,000 | **`boosting="goss"`** | **UNTESTED** |

Column arithmetic: `m00_core` 151, `m01_seq` 60, `m02_dist` 59, `m03_dyn` 60, `m04_resid` 60,
`m06_loc` 60, `m07_bayes` 50 → FULL = **500** exactly. `RT-414` = 50+60+60 = **170**, which
matches the independent "170 cols" note in `wave2_streams.py`. That agreement validates the
derivation; the counts are used quantitatively in `LANE_LOCAL.md` §L2.

**Four of seven slots have never been tested with a second learner family**, and the untested
four are not a random sample — they include the two structurally *most* idiosyncratic
LightGBM configurations (`RT-412` extra-trees/per-series, `RT-415` GOSS).

### 1.3 The loss geometry we are trying to move

From `WAVE7_RT600_EXACT_ALPHA_BUDGET.md` via `LEADERBOARD_ASSAULT_STATUS.md`:

- **Dominant cell** = current `t ≥ 200` **AND** positive break age `≥ 100`.
- **45.29%** of exact remaining pairwise inversion loss; **50.50%** of total pair weight.
- Cell AUC **0.66428**. Inside the cell, **never-break negatives carry 74.0%** of the loss.
- Translation: `pooled_delta ≈ 0.5050 × cell_delta`. Any "this fixes the dominant cell" claim
  is halved before it becomes a pooled TS-AUC claim.

### 1.4 The information frontier

`W7-D3R` (`reports/wave7_d3r.md`), run on exactly the dominant cell:

| Arm | Description | Cell TS-AUC |
|---|---|---:|
| A — `RT-300` | legal prefix baseline | 0.65341 |
| B — `RT-990` | **same information, more tree capacity** | 0.64749 |
| C — `RT-991` | full sequence, each series' own future (offline diagnostic only) | **0.71859** |

**B − A = −0.00592**, negative on 4/5 folds: more capacity on identical legal columns buys
nothing. **C − B = +0.07110** on 5/5 folds — a **ceiling, not a target**; Arm C reads each
series' own future and can never be deployed. Recorded verdict: **CASE 2, future-information
limit.**

### 1.5 What has actually produced positive marginal alpha — the base rate

| Program | Best `marginal_vs_clone` | Verdict |
|---|---:|---|
| **CatBoost hybrid `RT-1257`** | **+0.002407205** | **PROMOTION_WORTHY** |
| CAT-413 (`RT-1254`) | +0.001087151 | INTERESTING |
| CAT-300 (`RT-1255`) | +0.001029459 | INTERESTING |
| CSA-00 zero-training 8th member | +0.000975838 | descriptive |
| LA-03 per-series history adaptation | +0.001676065 | **KILL** — failed fixed-null isolation at `−0.021495290` |
| RT-1216 weighted conformal test martingale | +0.000937 | KILL (closest first-sweep miss) |
| T2 / `RT-995` teacher distillation | +0.00024 | MOSTLY REDUNDANT |
| CAT-410 (`RT-1256`) | +0.000753095 | KILL |
| SS-01 / SS-02 / SS-03 / SS-04 | −0.000310 / −0.000290 / −0.000299 / −0.000312 | all KILL |
| LA-01 specialist replacement salvage | −0.000005408 | KILL |
| First sweep — eleven arms `RT-1200`…`RT-1218` | all below gate | all KILL |
| CRF-01 (`RT-1234`), CRF-02 (`RT-1240`) | **never computed** — abandoned at standalone gate | KILL |

**Read this before proposing anything.** Across two exhausted preregistered sweeps, a neural
representation program, a distillation program and a leaderboard-alpha program, **exactly one
mechanism has produced material positive marginal ensemble alpha: swapping the learner family
in an existing specialist slot.** That mechanism is **3/7 explored**.

### 1.6 Diversity is not the bottleneck — arbitration is

`CRF_FINAL.md` §F, "Diagnosis: the bottleneck is arbitration, not detection":

- CRF-01 **repairs 39.5%** of RT-600's sampled dominant-cell mistakes.
- Repair **Jaccard vs the seed clone `RT-401` is 0.174** — the repairs are genuinely its own.
- It **damages 26.8%** of what RT-600 already had right.
- Pre-break damage rate on RT600-correct pairs **0.2626** against a **0.0150** cap.

`FIRST_SWEEP_SYNTHESIS.md` §H1, across the whole first sweep: *any* candidate repairs **91.8%**
of sampled dominant RT600 mistakes but damages **83.0%** of RT600-correct dominant pairs. Also
recorded: dominant-cell pair net is the strongest non-tautological correlate of
`marginal_vs_clone` (**Pearson 0.879**), and **`corr(standalone, ρ) = +0.983` across 17 arms** —
nothing this project has built has ever been both good and different.

**This governs the whole program: we do not have a detection problem, we have a retention
problem.** It is why the neural lane is gated on an arbitration result rather than funded up
front.

### 1.7 Local compute inventory (verified 2026-08-28)

This determines the lane split and was measured, not assumed.

| Item | Value |
|---|---|
| Machine | Apple **M2 Pro**, **10 cores**, **16 GB** RAM |
| Competition data | `structural-break-claude-wave3/data/X_train.parquet` (218 MB), also mirrored in other worktrees |
| **Prebuilt feature cache** | `structural-break-claude-wave3/cache/features/` — **12.61 GB**, per-module `.npy` + `.cols.json`, memory-mappable |
| FULL bank on disk | 10.08 GB across the seven modules (`m00_core` 3.04 GB is the bulk) |
| Rows | **5,036,517** per module, `float32` |
| Control OOF vectors | `structural-break-learner-diversity-2026/research/oof/` — all 7 specialists, `RT-401`–`RT-406`, `RT-1251`, `RT-1254`–`RT-1256`. **All are symlink chains** (→ `wave5` → `claude-wave3`); each resolves to 19.2 MiB, `(5036517,) float32`, 4,032,524 finite (80.0657%). `research/oof/` is `.gitignore`d (line 60) — which is why they never reached the cloud |
| CRF OOF vectors | `structural-break-causal-representation-frontier/research/oof/` — incl. `RT-1234`/`RT-1235` (fold 0, 806,334 finite rows = 16.01%) |
| Measured CatBoost runtimes | 1527.1 s / 2208.1 s / 1132.1 s per arm at `thread_count=2` (4867.3 s for three) |

**Consequence: the entire CatBoost slot sweep needs no feature rebuild.** Modules are
memory-mapped, and the largest training subset is 1,000,000 rows × 500 cols × 4 B = **2.0 GB**
resident. That fits the 16 GB machine with room to spare. This is what makes the LOCAL lane
real rather than aspirational.

**Warning for both agents:** `X_test.reduced.parquet` and `y_test.reduced.parquet` are present
in several worktree `data/` directories. `AGENTS.md` forbids reading any path containing
`reduced` or `test` during research. Their presence on disk is not permission.

---

## 2. BINDING CONSTRAINTS — BOTH LANES

Not negotiable. Sources: `AGENTS.md`, `research/PROTOCOL.md`, and the frozen conventions of
`FULL_OOF_PREREG.md` and the CatBoost `PREREG.md`.

### 2.1 Data and causality

- **No lockbox.** The 2,000-series lockbox opens only for a final confirmation read, never for
  iterative selection. If in doubt whether a check is "selection", it is.
- **No test data.** See the warning in §1.7.
- **True `tau` is forbidden**, directly or as a proxy, as a feature or as any input to online
  inference.
- **Bitwise prefix invariance at `atol = 0.0`** for every feature module. CRF-01's preflight is
  the standard: 24 gates green, prefix invariance bitwise, truncation `2.220e-16`, batch
  composition `1.665e-16`, fold purity proven **with a positive control that reproduces a
  known-contaminated scheme**.
- **Future information only in an explicitly authorized teacher-only role**, never online.
- **Nested cross-fitting mandatory** wherever an auxiliary model is fit on data overlapping the
  evaluation fold. Run the fold-purity sentinel. Outer-fold contamination is a real,
  previously-caught defect here (`W7` teacher pilot, `RT-992`/`RT-993`).

### 2.2 Folds and calibration

- Canonical **series-level, permanent** folds from `research/folds/folds.parquet`, folds `0..4`.
  Never regenerate. Never row-wise split. Never let two prefixes of one series straddle a
  boundary.
- Calibration is **fold-pure `SCDF_NSEEN`**: fold `k`'s calibration fitted only on the other four.
- Ensemble integration is **equal weight**. Blend-weight optimization is not authorized anywhere
  in this program without its own preregistration.

### 2.3 The binding evaluation protocol

Every candidate in both lanes is judged by **nested slot replacement**, outer-fold pure (fold
`k`'s replaced specialist is chosen using only the other four folds, then frozen and evaluated
on fold `k`):

- **`E0`** — original seven-specialist RT-600.
- **`E1`** — six RT-600 specialists + exchangeable matched LightGBM clone in the target slot.
- **`E2`** — six RT-600 specialists + candidate in the target slot.
- **Primary metric: `marginal_vs_clone = E2 − E1`.**
- Also report `E2 − E0`, per-fold deltas of `E2 − E1`, and folds positive.

Multi-slot hybrids use the first unused seed clones from `RT-401..RT-406` in **frozen specialist
order** — the CSA-03 convention.

**Replacement, never addition.** `W4-E6` tested the union of both arms (13 boosters) and scored
**−0.00095 (1/5 folds)**. More members is not better.

### 2.4 Frozen gate ladder — identical in both lanes

Inherited unchanged from `FULL_OOF_PREREG.md` and CatBoost `PREREG.md`. **Do not move these
after seeing a result.**

| Verdict | Condition |
|---|---|
| `KILL` | `marginal_vs_clone < +0.0010` |
| `INTERESTING` | `≥ +0.0010` |
| `PROMOTION_WORTHY` | `≥ +0.0015`, `≥4/5` folds positive, dominant pair net `> 0` |
| `SERIOUS` | `≥ +0.0030`, `≥4/5` positive, dominant pair net `> 0`, mature-vs-never pair net `> 0` |
| `MAJOR` | `≥ +0.0050` |

**Low correlation alone is never success.**

### 2.5 Pair-flow diagnostic convention

64 same-`t` pairs per time point, seed `20260827`, splits `whole`, `dominant_cell`,
`mature_vs_never`, `mature_vs_prebreak`. Unchanged from Learner Diversity / CatBoost / GPU
Tabular so every number in this program is directly comparable to those.

### 2.6 Bookkeeping obligations — mandatory, not cleanup

- **`research/RESULTS.csv`** — append one row per scoring run. Never edit or delete a row.
- **`research/EXPERIMENT_ID_MAP.md`** — allocate every `RT-xxxx` here first. **A voided
  experiment keeps its ID.** IDs are never reused or renumbered.
- **`research/RDOF_LEDGER.md`** — record degrees of freedom frozen before each score.
- **`research/FAILED_EXPERIMENTS.md`** — mandatory entry per rejected hypothesis: what was
  tried, the falsification condition, why it failed.
- **`research/STATUS.md`** — update only when the production anchor, external score, or active
  research conclusion changes. One screen maximum.
- **Do not commit** `.npy` OOF arrays, checkpoints, caches, model binaries, logs, or notebook
  outputs.

See §5.3 for how two concurrent agents share these files without corrupting them.

---

## 3. AUDIT OF `deep-research-report (4).md`

Both agents will encounter this report. It contains one error that would send an agent to the
wrong place.

### 3.1 What verified correct

| Claim | Verification |
|---|---|
| RT-1257 `+0.002407205`, `E2−E0 +0.002026322`, 5/5, dominant `+158`, mature-never `+73` | **exact** — `reports/catboost_specialist_2026/results.csv` |
| RT-600 production, external 0.6268 | **exact** |
| CRF-01 ladder: representation `+0.0444`, objective `+0.0222` | **exact** — `CRF_FINAL.md` §E |
| CRF-02 fixed-null isolation `−0.021446` | **exact**, correctly *not* conflated with LA-03's `−0.021495290` |
| RT-1258/RT-1259 unresolved | **confirmed** — no ledger row, no `results.json`, no OOF on disk |
| SS-01…04 KILL; RT-1216 `+0.000937`; T2 `+0.00024` | **exact** |
| Nested slot replacement over eighth-model averaging | **correct**, independently supported by `W4-E6` |
| No branch created, no model trained, no ID consumed | **confirmed** — honest about its own limits |

It is a careful piece of work and its refusal to fabricate a branch SHA or commit count is the
right instinct.

### 3.2 **The blocker, and the direction that resolves it**

The report states the binding test for `RT-1258`/`RT-1259` is blocked because the `RT-600` and
`RT-401` control OOF arrays are gitignored and absent, and concludes that *"that test must occur
later in the persistent local research checkout."*

**The first half is right. The second half is backwards, and the direction matters more than the
diagnosis.**

**What is true.** The controls are absent **from the Crunch cloud environment** — they are
`.gitignore`d at `research/oof/` (line 60) and therefore were never part of the submitted tree.
The submission's own docstring says exactly this, and it is correct.

**What the report gets wrong.** It concludes the fix is to bring the neural predictions *down* to
the Mac. **Crunch cloud-generated model artifacts are not intended to be downloaded back to a
local machine**, so a workflow that depends on pulling `tabm_oof.npy` / `realmlp_oof.npy` locally
is built on something the platform does not offer.

**The neural predictions are not missing — they were generated correctly.** The completed cloud
run produced complete five-fold OOF for both learners and uploaded them as model artifacts:

```
resources/gpu_tabular_oof/tabm_oof.npy          resources/gpu_tabular_oof/realmlp_oof.npy
gpu_tabular_oof/tabm/fold_{0..4}_pred.npy       gpu_tabular_oof/realmlp/fold_{0..4}_pred.npy
```

The absence of `fold_*_pred.npy` from this machine is therefore **expected and correct**, not
evidence of a failed run.

**The resolution inverts the handoff: ship the controls UP, evaluate IN Crunch, bring a small
JSON back.** And the mechanism already exists — this needs no new scientific code.
`submissions/H_gpu_tabular_full_oof.py::_try_binding_replacement_test()` calls
`load_control_oof(_REPO_ROOT)`, which looks for exactly eight arrays at
`_REPO_ROOT/research/oof/{RT-300,RT-401,RT-410,RT-411,RT-412,RT-413,RT-414,RT-415}.npy`. When
they are present it computes the full `E0`/`E1`/`E2` battery **in the cloud** via
`evaluate_gpu_oof.evaluate_learner()` and prints it between
`=== GPU_TABULAR_OOF_RESULTS_BEGIN ===` and `=== GPU_TABULAR_OOF_RESULTS_END ===`. When they are
absent it returns `{"computed": False, ...}` and the verdict
`PENDING_LOCAL_BINDING_EVALUATION` — which is precisely the state the last run ended in.

So the missing piece was never the neural predictions and never the scoring logic. **It was eight
`.npy` files that never left this Mac.**

**This is why the program splits into two lanes, but the flow is the opposite of what the report
implies:** packaging and validating the eight controls is a local task (`LANE_LOCAL.md` §L1);
arranging for them to reach the cloud environment alongside the existing neural artifacts, and
running the evaluation there, is a Crunch task (`LANE_CRUNCH.md` §C1).

**Do not attempt to circumvent the artifact restriction** — no printing arrays to logs, no base64
encoding of predictions, no extraction workarounds. The small result JSON is sufficient for every
research decision this program needs to make.

**One related non-issue, recorded so nobody investigates it twice.** The GPU research submission
scored **0.500** on the leaderboard. That was expected and correct: `infer()` deliberately returns
constant 0.5 probabilities because the submission is labelled `RESEARCH OOF ONLY - NOT A
LEADERBOARD CANDIDATE`. The meaningful output was the training-set OOF computed inside `train()`,
never `prediction.parquet`. **The next objective is not another leaderboard submission.**

### 3.3 Other corrections

**(a) "CRF-01 was correctly killed because it damaged incumbent-correct pairs" — half true.**
It was killed at a **standalone** cheap-abandon gate: whole-fold TS-AUC `0.592762` against a
`0.600` necessary condition. `CRF_FINAL.md` states plainly that **no CRF arm ever produced a
`marginal_vs_clone` at all** — folds 1–4 were deliberately never trained. Pair flow is the
supporting argument, not the trigger. The neural family has never been measured on the binding
endpoint, and any future claim about it must say so.

**(b) The report's "novel" damage-penalty loss is not novel — it is SS-02, and SS-02 is KILL.**
The report ranks "penalize reversal of incumbent-correct pairs" as design variable #4 /
experiment FNSR-02, presented as untried. `SS-02` (`RT-1223`/`RT-1224`) trained a same-`t`
residual pair ranker with an explicit `damage_penalty: 2.0` and dominant-pair weighting:
`marginal_vs_clone −0.000290`, dominant repairs/damage/net `330/438/−108`, mature-vs-never
`−77`, shuffled-control gap `−0.000087`. The mechanism differs (shallow residual ranker on the
incumbent bank vs. a penalty inside a deep sequence model's loss), so this is not a full
closure — but it is uncited prior art that substantially lowers the prior.

**(c) It reopens closed hypotheses.** FNSR is `H-A` plus `H-B` with a longer memory horizon.
The CRF factorial closed both, measured each (`+0.0444`, `+0.0222`), and recorded that **their
sum is still an order of magnitude short**.

**(d) It contradicts the repository's own diagnosis without engaging it.** `CRF_FINAL.md` says
the bottleneck is **arbitration, not detection**; the report's headline proposal is a better
detector. It does not address §1.6.

**(e) It misses the cheapest live lead entirely.** The CatBoost slot sweep is **3/7 complete**
and the untested four were excluded by a priori prereg scope, not by evidence. The report does
not mention `RT-411`, `RT-412`, `RT-414` or `RT-415` once. That is now `LANE_LOCAL.md` §L2 and
it is the highest-EV item in the program.

**(f) Minor.** Mamba / xLSTM / MixturePFN citations are architectural motivation only — the
report says so itself and is correct. They carry no weight in the gate ladder.

---

## 4. THE TWO LANES

### 4.1 Why this particular split

The split is not "hard things go to the cloud." It follows one criterion: **does the task
require a GPU or the Crunch platform runtime?** Verified against §1.7:

| Task | Needs GPU / platform? | Lane |
|---|---|---|
| Validate + package the eight control OOF arrays (153.6 MiB, hashing) | No | LOCAL |
| Train CatBoost, `depth=6`, `iterations=600`, `thread_count=2`, ≤1M rows | No — measured at ~1500 s/arm on this machine | LOCAL |
| Arbitration probe over frozen `.npy` prediction vectors | No — minutes of CPU | LOCAL |
| Read a result JSON out of Crunch logs and file it in the ledgers | No | LOCAL |
| Determine whether a prior run's model artifacts can be mounted/reused | **Yes — platform semantics** | CRUNCH |
| **Run the binding E0/E1/E2 where the neural OOF lives** | **Yes — artifacts stay in Crunch** | CRUNCH |
| Re-run TabM / RealMLP full 5-fold OOF (4.545 GPU-hours) if reuse is impossible | **Yes — RTX 4090** | CRUNCH |
| Measure online per-timestep inference cost in the real runtime | **Yes — must be the platform's own runtime** | CRUNCH |
| XGBoost GPU arm | **Yes** | CRUNCH |
| Train a selective SSM / neural sequence model | **Yes** | CRUNCH |

**The controlling constraint:** the neural OOF predictions exist as Crunch model artifacts and
**are not intended to be downloaded** (§3.2). Therefore the analysis goes to the data, not the
reverse. The eight control arrays travel **up**; a small result JSON comes **back**.

### 4.2 Lane contents at a glance

**LOCAL** — `LANE_LOCAL.md`, worktree `structural-break-deep-ensemble-frontier-local-2026`:

| ID | Task | Cost | Blocked by |
|---|---|---|---|
| **L1** | **Validate + package the eight control OOF arrays for upload** (manifest with sha256, shape, dtype, alignment) | ~1 h | nothing |
| **L2** | **CSA-04 — complete the CatBoost slot sweep + hybrid curve** | ~1.8 CPU-h | nothing |
| **L3** | Arbitration probe on existing fold-0 OOF | minutes | nothing |
| **L4** | File the returned result JSON for `RT-1258`/`RT-1259`; ledger custody | ~1 h | **C1** (cross-lane) |

**CRUNCH** — `LANE_CRUNCH.md`, worktree `structural-break-deep-ensemble-frontier-crunch-2026`:

| ID | Task | Cost | Blocked by |
|---|---|---|---|
| **C1** | **Get controls + neural OOF into one environment and run the binding test.** First investigate artifact reuse; only re-run if reuse is impossible | evaluation-only session, **or** 4.545 GPU-h if regeneration is forced | **L1** (needs the validated package) |
| **C2** | **Deployment inference benchmark — `RT-1257` and the best-`k` hybrid** | platform session | nothing for `RT-1257`; **L2** for the hybrid |
| **C3** | XGBoost GPU cross-family arm | GPU session | **L2** passing |
| **C4** | **Neural sequence models — last, gated** | days | **L3** (see §6.3) |

### 4.3 Both lanes start immediately with unblocked work

**LOCAL agent starts on L1, L2 and L3.** None is blocked. L1 is short and unblocks the other
agent's main task, so **do L1 first**, then L2 (the highest-EV item in the program) and L3 (the
gate for the entire neural lane).

**CRUNCH agent starts on C2 and on the C1 investigation.** C2 on `RT-1257` is unblocked *today*
and is an outstanding obligation — `reports/catboost_specialist_2026/FINAL.md` closes with
"review surviving CatBoost specialist(s) against deployment cost before any production work" and
**that review has never happened**. `RT-1257` is PROMOTION_WORTHY and undeployed.

**The C1 investigation is also unblocked and must come before the C1 run:** determine whether
Crunch can mount or reuse the previous submission's uploaded model artifacts. **Do not assume a
new submission inherits the previous run's artifacts.** That answer decides whether C1 costs an
evaluation-only session or a 4.545-GPU-hour regeneration, and it can be answered while L1 is
still packaging.

Neither agent waits on the other to begin.

---

## 5. COORDINATION PROTOCOL

Two agents on one repository is exactly the situation `AGENTS.md` §"Branch / worktree rules"
warns about. This section is what keeps them from destroying each other's work.

### 5.1 Worktree isolation — the hard rule

```
research/deep-ensemble-frontier-2026            <- this plan; both agents READ, neither WRITES after setup
  ├── research/deep-ensemble-frontier-local-2026   <- LOCAL agent only
  └── research/deep-ensemble-frontier-crunch-2026  <- CRUNCH agent only
```

Create both from the plan branch:

```
git worktree add -b research/deep-ensemble-frontier-local-2026 \
    "../structural-break-deep-ensemble-frontier-local-2026"  research/deep-ensemble-frontier-2026
git worktree add -b research/deep-ensemble-frontier-crunch-2026 \
    "../structural-break-deep-ensemble-frontier-crunch-2026" research/deep-ensemble-frontier-2026
```

- **Never touch a worktree you were not assigned.** Not to read a file, not to "just check" —
  use `git show <branch>:<path>` if you need to see the other lane's content.
- **Never `git push --force`.** If a push is rejected for divergence, **stop and investigate**.
  `docs/repository_cleanup_audit.md` §C documents a real case where `research/wave5-alpha`
  diverged between two sessions and neither side was overwritten.
- **Never rebase pushed history**, never `git filter-repo`/BFG. Preregistration commits,
  experiment IDs and negative results must stay reachable exactly as committed.
- Before anything that could discard work, run `git status` **and** `git worktree list` first.

### 5.2 Experiment-ID partition — the anti-collision mechanism

Highest allocated ID anywhere in the repository is **`RT-1259`**. Verified 2026-08-28 across
`EXPERIMENT_ID_MAP.md`, `RESULTS.csv`, `RDOF_LEDGER.md` and every local worktree: **`RT-1260`
and above are entirely unused.**

**The ID space is partitioned by lane so collision is structurally impossible:**

| Range | Owner | Purpose |
|---|---|---|
| `RT-1260` – `RT-1269` | **LOCAL only** | CSA-04 arms + hybrid + local contingency |
| `RT-1270` – `RT-1289` | **CRUNCH only** | XGBoost cross-family, deployment arms, contingency |
| `RT-1290` – `RT-1319` | **CRUNCH only** | reserved for neural (`C4`), unallocated until C4 opens |

**Neither agent may allocate outside its own range**, even if the other range looks unused.
Proposed LOCAL allocation (confirm at prereg time, not before):

| ID | Arm |
|---|---|
| `RT-1260` | CSA-04 CAT-411 |
| `RT-1261` | CSA-04 CAT-412 |
| `RT-1262` | CSA-04 CAT-414 |
| `RT-1263` | CSA-04 CAT-415 |
| `RT-1264` | CSA-05 best-`k` mixed hybrid |
| `RT-1265`–`RT-1269` | LOCAL contingency, unallocated |

`RT-1258`/`RT-1259` are **already allocated** to GPU-01/GPU-02 and must not be reused whether or
not those runs are ever scored.

### 5.3 Shared-file custody

These files exist on both branches and will conflict if both agents write them freely.

| File | Rule |
|---|---|
| `research/RESULTS.csv` | **Append-only, both lanes.** Each agent appends only rows for IDs in its own range. On merge, conflicts are resolved by **keeping both sides** and sorting by ID. Never edit or delete an existing row. |
| `research/EXPERIMENT_ID_MAP.md` | Append-only within your own ID range. Same merge rule. |
| `research/RDOF_LEDGER.md` | Append-only; each lane appends its own program section. |
| `research/FAILED_EXPERIMENTS.md` | Append-only; one entry per KILL, in your own range. |
| **`research/STATUS.md`** | **SINGLE WRITER — LOCAL agent only.** This file is rewritten rather than appended, so concurrent edits corrupt it. The CRUNCH agent reports status changes to the LOCAL agent (or the user) and does **not** edit it. |
| `reports/deep_ensemble_frontier_2026/*` | Each lane writes only its own subdirectory and its own lane file. |

Report subdirectories, to keep them disjoint:

```
reports/deep_ensemble_frontier_2026/
    PROGRAM_PLAN.md     <- shared core (this file)
    LANE_LOCAL.md       <- LOCAL agent's plan
    LANE_CRUNCH.md      <- CRUNCH agent's plan
    local/              <- LOCAL agent writes only here
    crunch/             <- CRUNCH agent writes only here
```

### 5.4 Cross-lane handoffs

Exactly four handoffs. Everything else is independent.

| # | From → To | Artifact | Consumer |
|---|---|---|---|
| **H1** | LOCAL `L1` → CRUNCH `C1` | the **validated eight-array control package** + its manifest (sha256, shape, dtype, finite count, provenance) | CRUNCH places it at `research/oof/` in the submitted tree so `load_control_oof()` finds it |
| **H2** | LOCAL `L2` → CRUNCH `C2` | the best-`k` hybrid composition (which slots are CatBoost) | CRUNCH benchmarks its online inference cost |
| **H3** | LOCAL `L3` → CRUNCH `C4` | the arbitration-probe verdict | **gates whether the neural lane opens at all** |
| **H4** | CRUNCH `C1` → LOCAL `L4` | the **result JSON** copied out of the Crunch logs between the `GPU_TABULAR_OOF_RESULTS_BEGIN/END` markers | LOCAL files `RT-1258`/`RT-1259` in the ledgers |

**Handoff mechanics.**

- **H1 is the one that carries real data, and `.npy` files are never committed** (`AGENTS.md`).
  The package moves through the filesystem into the Crunch submission tree; only the **manifest**
  is committed, under `reports/deep_ensemble_frontier_2026/local/`. **The arrays are symlink
  chains on this machine** (§1.7) — packaging must dereference (`cp -L`, `tar -h`, `rsync -L`) or
  the cloud receives 82-byte dangling links.
- **H4 carries no arrays — deliberately.** The neural OOF stays in Crunch. A small JSON is
  sufficient for every research decision this program makes. **Do not attempt to extract the full
  arrays** by printing, base64-encoding, or any other workaround (§3.2).
- H2 and H3 are small enough to commit directly as markdown/JSON.

### 5.5 Sequencing

```
  t=0   LOCAL  ──▶ L1 (validate+package 8 controls, ~1h) ──▶ H1 ──┐
        LOCAL  ──▶ L2 (CSA-04, ~1.8 CPU-h) ──┬──▶ H2 ──▶ CRUNCH C2 (hybrid benchmark)
        LOCAL  ──▶ L3 (probe, minutes) ──────┼──▶ H3 ──▶ gates CRUNCH C4 (neural)
                                             └──▶ L2 pass ──▶ CRUNCH C3 (XGBoost)
                                                                 │
  t=0   CRUNCH ──▶ C2 (RT-1257 benchmark — unblocked today)      │
        CRUNCH ──▶ C1 investigation: can prior artifacts be reused? ◀┘
                     │
                     ├─ reuse possible ──▶ evaluation-only run ──▶ H4 ──▶ LOCAL L4
                     └─ reuse impossible ─▶ technical recovery run (frozen config,
                                            no tuning, controls packaged first) ──▶ H4
```

Decision rules at each junction:

| Junction | If | Then |
|---|---|---|
| C1 investigation | prior model artifacts are mountable/reusable | **evaluation-only recovery run** — no neural retraining |
| C1 investigation | they are not | **technical recovery run**: regenerate `RT-1258`/`RT-1259` from the frozen config with controls packaged from the start. **Not a new experiment — no new IDs, no tuning** |
| After L4 | TabM `marginal_vs_clone ≥ +0.0010` | a third learner family exists → feeds C3; raises the prior for C4 |
| After L4 | both KILL | strong evidence against "new architecture, same 500 features" → **lowers the prior for C4 substantially** |
| After L2 | best-`k` hybrid `≥ +0.0030` | SERIOUS reached → H2 to C2 for deployment feasibility, then confirmation |
| After L2 | best-`k` hybrid `< +0.0015` | learner-family mixing exhausted at two families → C3 becomes the only ensemble lane |
| After L3 | gating retains repairs under the damage cap | **C4 opens** with a concrete mechanism |
| After L3 | gating cannot retain repairs | **C4 does not open.** See §6.3 |

---

## 6. CROSS-CUTTING CONCERNS

### 6.1 Deployment feasibility gates every promotion, in both lanes

Currently **unmeasured**. `reports/catboost_specialist_2026/FINAL.md`: *"inference adds one
CatBoost model per surviving replacement and needs separate deployment benchmarking"* and
*"Next action: review surviving CatBoost specialist(s) against deployment cost before any
production work."* **That review has not happened.**

This matters more as L2 succeeds — a 4-slot hybrid quadruples `RT-1257`'s CatBoost inference
exposure — and far more for C4, where a selective SSM running causally at every online timestep
is a fundamentally different inference profile from a gradient-boosted tree.

**Rule for this program: no arm is described as promotable until its online, streaming,
per-timestep inference cost has been measured against the competition budget**, using the
methodology of the RTX 4090 hardware benchmark (Crunch submission `76357`, task
`run-3e834e0f`). A model that cannot run online is not a candidate regardless of its
`marginal_vs_clone`. This is CRUNCH `C2` and it is why C2 starts immediately rather than waiting
for L2.

### 6.2 Multiple comparisons across two lanes

Two agents running in parallel roughly doubles the rate at which this program consumes chances
to find a `+0.0010`. `RDOF_LEDGER.md` exists precisely so that "+0.0005" can be read against the
number of chances taken.

Mitigations, binding on both lanes:

- Gates are frozen in §2.4 **before** any arm runs and do not move.
- Within a lane, **all arms of a sweep are trained before any of them is scored.**
- The binding deliverable of L2 is the **hybrid curve**, not any individual slot squeaking over
  `+0.0010`.
- Each lane logs its own degrees of freedom in `RDOF_LEDGER.md` as it goes, not retrospectively.
- Neither lane may re-run a killed arm under an amended prereg. If a result forces a change of
  plan mid-run, register a **new** prereg and say so explicitly in the commit message — see
  `f0276ae` for the pattern.

### 6.3 Why neural is last and gated — summary

Full argument in `LANE_CRUNCH.md` §C4.1. In brief:

1. The representation × objective factorial is **complete and empty** (`H-A`, `H-B` closed).
2. Learned nulls are **closed** (`H-E`, `−0.021446` against the fixed per-series null).
3. `W7-D3R` Arm B says more capacity on the same information is **negative**; the residual gap
   is predominantly **post-`t`** (`H-D`), which no causal model reaches.
4. The measured bottleneck is **arbitration, not detection** (§1.6) — a better detector does not
   address it.
5. `corr(standalone, ρ) = +0.983` across 17 arms.
6. The real cost of a neural arm is the **preflight**, not the training: CRF-01 needed 24 green
   gates, bitwise prefix-invariance proof, a fold-purity positive control reproducing a known
   contaminated scheme, and a checkpoint provenance fingerprint added only after
   `RT-1237/1238/1239` were voided by a unit-test null.

**C4 opens if and only if L3 demonstrates a retention mechanism**, or if L1 and L2 both close
the ensemble lane and C4 becomes the least-bad remaining option by elimination — in which case
the report must say so explicitly rather than dressing elimination up as evidence.

---

## 7. RISK REGISTER

| # | Risk | Lane | Likelihood | Impact | Mitigation |
|---|---|---|---|---|---|
| R1 | Multiple comparisons across 4 new CSA-04 slots manufacture a false `+0.0010` | LOCAL | Medium | High | Gate frozen in advance; all four trained before any scored; **hybrid curve is the binding deliverable**; RDOF logged |
| R2 | CSA-04 hybrid does not reproduce `RT-1257` at `k = 2` | LOCAL | Low | **Critical** | Mandatory regression check (`LANE_LOCAL.md` §L2.5). If it fails, **halt** — no CSA-04 number is trustworthy until explained |
| R3 | Prior model artifacts cannot be mounted, forcing a 4.545-GPU-h regeneration | CRUNCH | Medium | Medium | Investigate reuse **before** committing GPU time (§4.3). If forced, it is a **technical recovery** under the frozen config — no tuning, no new IDs, controls packaged first. Record the regeneration in `RDOF_LEDGER.md` as recovery, not as a new arm |
| R3b | A regeneration quietly changes a TabM/RealMLP parameter | CRUNCH | Low | **Critical** | `FROZEN_GPU_CONFIG.json` is enforced by content hash in `train()`. Do not weaken that check. A changed parameter makes it a new experiment needing a new prereg and new IDs |
| R4 | A recovered GPU fold log contains a forbidden TS-AUC | CRUNCH | Low | Medium | Inspect logs before evaluation; record the deviation rather than dropping it |
| **R4b** | **Someone tries to extract the neural OOF arrays out of Crunch** | CRUNCH | Medium | High | Explicitly out of bounds (§3.2): no printing arrays, no base64, no log encoding. The result JSON is sufficient. If a decision seems to need the raw arrays, that decision is misspecified — re-read §2.3 |
| R5 | Deployment cost kills a scientifically successful hybrid | CRUNCH | **Med-High** | High | C2 starts at t=0 in parallel, not after L2 |
| R6 | Neural arm consumes weeks and reproduces CRF-01 | CRUNCH | Medium | High | C4 gated on L3; isolation control mandatory; cheap abandon gate at 0.600 retained |
| R7 | Slot-replacement alpha is partly a bagging artifact, not family heterogeneity | LOCAL | Medium | Medium | `FINAL_ARCHITECTURE_FREEZE.md` measured that `+0.00499` of the `+0.00832` specialist delta is reproducible by seed variation alone. The `E1` seed-clone control is exactly the right control and is already in the protocol — state this in the report rather than assuming the reader knows |
| R8 | A run silently uses a wrong artifact (the `RT-1237/1238/1239` failure mode) | Both | Low | **Critical** | Checkpoint provenance fingerprints; compute-accounting sanity check (`pretrain_runtime_s = 0.2` against a real 1,364.9 s is what caught it); test-isolated caches |
| **R9** | **Two agents collide on `RESULTS.csv` / IDs / `STATUS.md`** | **Both** | **Medium** | **High** | §5.2 ID partition makes ID collision structurally impossible; §5.3 makes ledgers append-only and `STATUS.md` single-writer; §5.1 forbids cross-worktree writes |
| **R10** | **An agent reads or writes the other lane's worktree** | **Both** | Medium | High | §5.1; use `git show <branch>:<path>` to inspect, never a direct path |
| **R11** | **H1 control package is corrupt, truncated, misaligned, or arrives as dangling symlinks** | Both | **Medium-High** | **Critical** | The arrays are symlink chains locally (§1.7) — **dereference when packaging**. LOCAL commits a manifest with sha256/shape/dtype/finite-count; CRUNCH re-verifies every hash **before** any GPU time is spent. All eight must show `5,036,517` rows, `float32`, and `4,032,524` finite — a mismatch on any one fails the run early rather than producing a wrong `marginal_vs_clone` |
| R12 | Scope creep into an unbounded family × slot search | Both | Medium | High | `LANE_CRUNCH.md` §C3 caps cross-family at one enumerated assignment; every workstream needs its own prereg |
| R13 | Both agents idle waiting on the other | Both | Low | Medium | §4.3 — each lane has unblocked work at t=0 by construction |

---

## 8. DEFINITION OF DONE — WHOLE PROGRAM

**LOCAL lane:**

- [ ] `CSA04_PREREG.md` committed **before** any CSA-04 score exists, containing predictions
      P1, P2, P3 as written in `LANE_LOCAL.md` §L2.3.
- [ ] `RT-1260`–`RT-1263` trained on all five folds, all four complete before any is scored,
      all four with `RESULTS.csv` rows.
- [ ] The `k = 2` hybrid regression check reproduces `+0.002407205`.
- [ ] Full hybrid curve over `k` reported, `k*` identified, `RT-1264` allocated to best-`k`.
- [ ] P1 / P2 / P3 each explicitly adjudicated — **including the ones that were wrong.**
- [ ] L1 control package built with a committed manifest; all eight arrays verified at
      `5,036,517` rows, `float32`, `4,032,524` finite; H1 delivered.
- [ ] `RT-1258`/`RT-1259` filed from the H4 result JSON with a `marginal_vs_clone`, or a
      recorded `UNRECOVERABLE`.
- [ ] L3 arbitration probe executed and reported; H3 verdict delivered to CRUNCH.
- [ ] `STATUS.md` updated **only if** the anchor, external score, or active conclusion changed.

**CRUNCH lane:**

- [ ] Artifact-reuse question answered in writing **before** any GPU time was spent.
- [ ] C1 resolved: binding test computed in-cloud with all eight controls present, result JSON
      captured between the markers and delivered as H4 — or `UNRECOVERABLE` recorded.
- [ ] If regeneration was required: recorded as a **technical recovery**, frozen config hash
      verified, no parameter changed, no new RT ID allocated.
- [ ] `RT-1257` online per-timestep inference cost measured against the competition budget.
- [ ] Best-`k` hybrid inference cost measured once H2 arrives.
- [ ] C3 and C4 each either opened under their own preregistration, or **explicitly declined
      in writing with the gate result that declined them.**

**Both:**

- [ ] `RDOF_LEDGER.md` records every frozen degree of freedom for every arm.
- [ ] `FAILED_EXPERIMENTS.md` has an entry per KILL with the falsification condition.
- [ ] No `.npy`, checkpoint, cache, or notebook output committed.
- [ ] Both lane branches merged back to `research/deep-ensemble-frontier-2026` with ledger
      conflicts resolved by **keeping both sides**.

---

## 9. OPEN QUESTIONS TO RESOLVE BEFORE PREREGISTERING

Genuine unknowns. Each belongs in the relevant lane's prereg. Lane ownership marked.

1. **[CRUNCH — answer first, before any GPU time]** How does Crunch handle
   `model_directory_path` and the `resources` directory **across** runs? **Do not assume a new
   submission inherits the previous submission's uploaded model artifacts.** Can the completed
   `RT-1258`/`RT-1259` artifacts be mounted or reused by a subsequent run? Yes → evaluation-only
   recovery. No → technical regeneration at 4.545 GPU-hours.
2. **[LOCAL]** Confirm the exact path and filename convention the submitted tree expects for the
   control arrays. `load_control_oof()` reads `<root>/research/oof/{ID}.npy`; the agent-facing
   sketch used `resources/control_oof/`. **Follow the repository's existing convention rather
   than inventing a new one** — and if a `resources/`-style staging path is required by the
   platform, the shim that moves them to `research/oof/` must be explicit and committed.
3. **[LOCAL]** **CAT-415 and GOSS.** GOSS is a LightGBM boosting mechanism with no CatBoost
   analogue, so CAT-415 will use the frozen Bernoulli/`subsample=0.7` bootstrap. That makes it a
   less faithful slot reimplementation than CAT-411/412/414. **Decide before scoring** whether
   it is reported under a caveat or excluded from the hybrid — never after.
4. **[LOCAL]** **CAT-412 and `per_series` sampling.** Confirm the CatBoost runner honours
   `sample_mode="per_series"`. If it does not, that is a specification change to be disclosed in
   the prereg, not patched afterwards.
5. **[LOCAL]** **Hybrid slot ordering.** Add slots by descending single-slot marginal, or by
   dominant pair net (which correlates with `marginal_vs_clone` at Pearson 0.879)? **Pick one
   and freeze it** — trying both is a degree of freedom.
6. **[LOCAL]** Does `RT-411`'s module set (`m02_dist,m03_dyn,m04_resid,m06_loc` — no `m00_core`,
   no `m07_bayes`) overlap with what CatBoost is good at? Worth a descriptive look at the CSA
   feature-importance record before predicting its outcome.
7. **[BOTH]** **What is the competition deadline and remaining submission budget?** Not recorded
   in `STATUS.md` or `PROTOCOL.md`. This determines how much of C3/C4 is realistically fundable
   and should be established before either opens.
8. **[CRUNCH]** Is `research/xgb-gpu-2026` a live branch with usable artifacts or a stub? It
   appears in the remote snapshot but has no local worktree. C3's viability depends on it.

---

## 10. ONE-PARAGRAPH SUMMARY FOR EACH AGENT

**LOCAL agent.** Production is `RT-600` at external **0.6268**. The best measured research result
is `RT-1257`, a **two-slot** LightGBM/CatBoost hybrid at `marginal_vs_clone +0.002407205`, 5/5
folds positive. Do **L1 first** and it is short: validate and package the eight control OOF arrays
(`RT-300`, `RT-401`, `RT-410`–`RT-415`, 153.6 MiB dereferenced) with a sha256 manifest, and hand
them to the CRUNCH agent. Those eight files never leaving this Mac is the **entire** reason the
GPU experiment is unfiled — the neural predictions were generated correctly and the scoring code
already computes the binding test in-cloud when the controls are present. Then your headline job:
**CSA-04**. The CatBoost slot sweep is only **3 of 7 complete**, `RT-411`/`RT-412`/`RT-414`/`RT-415`
were excluded by the original prereg's a priori scope rather than by any result, the 12.61 GB
feature cache is already on disk so nothing needs rebuilding, and the whole sweep costs under two
CPU-hours. The real deliverable is the hybrid curve over `k` and the location of its interior
maximum. Also run the **zero-cost arbitration probe** on `RT-1234`'s existing fold-0 OOF — minutes
of CPU, and it decides whether the other agent may start a neural model at all. Details:
`LANE_LOCAL.md`.

**CRUNCH agent.** Two things are unblocked today. First, **answer the artifact-reuse question**:
can the completed `RT-1258`/`RT-1259` model artifacts from submission `76357` / task
`run-3e834e0f` be mounted or reused by a subsequent run? Do not assume they are inherited. That
answer is the difference between an evaluation-only session and a 4.545-GPU-hour regeneration, and
you can settle it while the LOCAL agent packages the controls. Second, **benchmark `RT-1257`'s
online inference cost** — the CatBoost final report named it as the next action, it has never been
done, and no arm in this program can be called promotable until it exists. Your main task, C1, is
**not** a new experiment: it is putting eight already-existing control arrays into the same
environment as already-existing neural predictions and running already-existing scoring code. Do
not tune anything, do not allocate new RT IDs, do not combine with `RT-1257` yet, and do not try
to pull the neural arrays out of Crunch — a small result JSON between the
`GPU_TABULAR_OOF_RESULTS_BEGIN/END` markers is sufficient for every decision here. And do **not**
start a neural model: `H-A`, `H-B` and `H-E` are closed, the measured failure is that candidates
cannot keep the repairs they find, and a zero-cost local probe is about to tell you whether any
retention mechanism exists at all. Details: `LANE_CRUNCH.md`.
