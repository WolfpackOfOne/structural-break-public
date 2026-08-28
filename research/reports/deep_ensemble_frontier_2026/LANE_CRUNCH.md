# DEEP ENSEMBLE FRONTIER 2026 — **CRUNCH LANE**

**Agent:** CRUNCH
**Branch:** `research/deep-ensemble-frontier-crunch-2026`
**Worktree:** `/path/to/workspace/structural-break-deep-ensemble-frontier-crunch-2026`
**Hardware:** Crunch cloud — RTX 4090, 15-h / 2-learner / 0.90-fraction quota
**ID range:** **`RT-1270` – `RT-1289`, plus `RT-1290` – `RT-1319` reserved for neural — you may
not allocate outside these ranges**
**Status:** `PLAN_ONLY — NO EXPERIMENT AUTHORIZED BY THIS DOCUMENT`

> **Read [`PROGRAM_PLAN.md`](PROGRAM_PLAN.md) in full first.** It holds the verified state, the
> binding constraints, the gate ladder, the audit of the deep-research report, and the
> coordination protocol. This file assumes all of it and does not repeat it.
>
> **You do not edit `research/STATUS.md`.** The LOCAL agent is its single writer. Report status
> changes to them.

---

## C0. THE ONE THING TO UNDERSTAND BEFORE YOU START

Your first task looks like a GPU job and is not one.

**`RT-1258` and `RT-1259` already ran. The neural predictions already exist.** The completed
cloud run produced complete five-fold OOF for both learners and uploaded them as model artifacts:

```
resources/gpu_tabular_oof/tabm_oof.npy          resources/gpu_tabular_oof/realmlp_oof.npy
gpu_tabular_oof/tabm/fold_{0..4}_pred.npy       gpu_tabular_oof/realmlp/fold_{0..4}_pred.npy
```

**The scoring code already exists too**, and it already knows how to run in the cloud.
`submissions/H_gpu_tabular_full_oof.py::_try_binding_replacement_test()` calls
`load_control_oof(_REPO_ROOT)` and, when the controls are present, computes the entire
`E0`/`E1`/`E2` battery via `evaluate_gpu_oof.evaluate_learner()` and prints it between
`=== GPU_TABULAR_OOF_RESULTS_BEGIN ===` and `=== GPU_TABULAR_OOF_RESULTS_END ===`.

**The only missing input was eight `.npy` files** — the RT-600 specialists and the RT-401 matched
clone — which live at `research/oof/`, are `.gitignore`d (line 60), and therefore were never part
of the submitted tree. Absent them, the function returns `{"computed": False, ...}` and the run
ends at `PENDING_LOCAL_BINDING_EVALUATION`. That is exactly where the last run ended.

**So C1 is not a new experiment.** It is putting eight already-existing control arrays into the
same environment as already-existing neural predictions, and running already-existing scoring
code. The LOCAL agent is packaging those eight files for you right now (handoff **H1**).

Two consequences, both binding:

1. **Analysis goes to the data, not the reverse.** Crunch cloud-generated model artifacts are not
   intended to be downloaded back to a local machine. **Do not build a workflow that depends on
   pulling `tabm_oof.npy` / `realmlp_oof.npy` down.**
2. **A small result JSON is sufficient.** You return the block between the markers; the LOCAL
   agent files it. You do not need — and must not attempt — to extract the full arrays.

**A related non-issue, recorded so nobody investigates it twice.** The GPU research submission
scored **0.500** on the leaderboard. That was expected and correct: `infer()` deliberately returns
constant 0.5 probabilities because the submission is labelled `RESEARCH OOF ONLY - NOT A
LEADERBOARD CANDIDATE`. The meaningful output was the training-set OOF computed inside `train()`,
never `prediction.parquet`. **The next objective is not another leaderboard submission.**

---

## C1. GET CONTROLS AND NEURAL OOF INTO ONE ENVIRONMENT, AND RUN THE BINDING TEST

**Type:** technical recovery / evaluation of an already-preregistered, already-executed
experiment. **Not new research.**
**IDs:** **none.** `RT-1258`/`RT-1259` are already allocated. **Do not allocate a new ID**, even
if regeneration is required.
**Cost:** an evaluation-only session if artifact reuse works; **4.545 GPU-hours** if it does not.
**Blocked by:** **H1** from LOCAL `L1` for the run itself. **The investigation in §C1.1 is not
blocked — start there.**

### C1.1 First, and before any GPU time: the artifact-reuse question

**Investigate exactly how Crunch handles `model_directory_path` and the `resources` directory
across cloud runs.**

**Do not assume a new submission automatically receives the previous submission's uploaded model
artifacts.** That assumption is the difference between a cheap evaluation and a 4.5-GPU-hour
regeneration, and getting it wrong in the optimistic direction wastes a run.

Determine specifically:

- Can the `RT-1258`/`RT-1259` model artifacts from the completed run (submission `76357`, task
  `run-3e834e0f`) be **mounted or reused** by a subsequent run?
- If so, by what mechanism, and does the reused artifact arrive at a path the submission can read?
- Does `resources/` persist across runs, or is it rebuilt from the submitted tree each time?

Write the answer to `reports/deep_ensemble_frontier_2026/crunch/C1_ARTIFACT_REUSE.md` before
launching anything. This is open question **#1** and it is yours.

### C1.2 Branch A — reuse is possible: evaluation-only recovery run

Preferred outcome. No neural retraining.

1. Receive **H1**: the eight-array control package plus
   `local/H1_CONTROL_MANIFEST.json` from the LOCAL agent.
2. **Re-verify every sha256 against the manifest before spending any compute.** All eight must
   show `5,036,517` rows, `float32`, and **`4,032,524` finite** (80.0657% — the five canonical dev
   folds; the remaining ~20% is the lockbox, correctly `NaN`). **A mismatch on any single array
   fails the run before it starts.** This is R11 and it is critical: a misaligned control produces
   a wrong `marginal_vs_clone` rather than an error.
3. **Place them where the code looks.** `load_control_oof()` reads `<root>/research/oof/{ID}.npy`
   for `RT-300`, `RT-401`, `RT-410`–`RT-415`. **Follow the repository's existing convention rather
   than inventing new filenames.** If the platform requires a `resources/`-style staging path, the
   shim that relocates them must be explicit and committed, not improvised.
4. **Watch the symlink hazard.** The arrays are symlink chains on the LOCAL machine. Verify the
   files you received are ~19.2 MiB each, **not 82 bytes**.
5. Mount or reuse the existing neural artifacts, and run the evaluation.
6. Capture the block between `=== GPU_TABULAR_OOF_RESULTS_BEGIN ===` and
   `=== GPU_TABULAR_OOF_RESULTS_END ===`. Deliver it as **H4**.

### C1.3 Branch B — reuse is impossible: technical recovery run

Only if §C1.1 says the artifacts cannot be reused.

**This is a technical recovery of an existing experiment, not a new scientific experiment.** That
distinction is not a formality — it determines what you are allowed to change, which is nothing.

- **Regenerate `RT-1258` and `RT-1259` using their exact frozen configurations.** TabM:
  `k=32, d_block=512, n_blocks=3, dropout=0.1, lr=0.002, weight_decay=0.0003, grad_clip_norm=1.0`.
  RealMLP: `hidden_sizes=[256,256,256], lr=0.04`. Both from `FROZEN_GPU_CONFIG.json`, commit
  `d506aa6`, enforced by content hash in `train()` — **do not weaken that check** (R3b).
- **Under no circumstances change any TabM or RealMLP parameter.** No batch size, epoch cap, `k`,
  `d_block`, hidden width, or layer count. A changed parameter makes this a new experiment
  requiring a new preregistration and new IDs.
- **Package all eight control arrays before the run starts**, so the binding test computes in the
  same session rather than deferring again.
- Matched contract, unchanged: 500-column bank
  (`m00_core,m01_seq,m02_dist,m03_dyn,m04_resid,m06_loc,m07_bayes`), canonical folds `0..4`,
  1,000,000 max training rows per outer fold, RT-401 sequential sampler seed `1`, fold-pure
  `SCDF_NSEEN`.
- **Training order and completion rule:** TabM folds `0..4`, then RealMLP folds `0..4`, or an
  equally deterministic resumable order. **Both candidates must complete before either is
  evaluated.** Neither may stop early because the other scored well — only on unrecoverable
  technical failure, which must be recorded.
- **Log hygiene.** Fold completion logs may contain only fold number, runtime, epochs,
  training/inner-validation diagnostics, RAM, VRAM, checkpoint status. **Never** TS-AUC, pair
  flow, replacement gain, or outer-fold rank correlation. No outer-fold TS-AUC is computed or
  printed after individual folds.
- Record the regeneration in `RDOF_LEDGER.md` as a **technical recovery**, not as a new arm.

### C1.4 The evaluation itself — do not rewrite it

**Run the existing `evaluate_gpu_oof.py` logic. Do not rewrite the scientific scoring logic unless
you find an implementation bug** — and if you do, report it rather than silently patching it.

The binding structure (`PROGRAM_PLAN.md` §2.3):

- **`E0`** = original seven-specialist RT-600.
- **`E1`** = six RT-600 specialists + the matched exchangeable `RT-401` LightGBM clone.
- **`E2`** = six RT-600 specialists + the candidate neural model.
- **Binding metric: `marginal_vs_clone = E2 − E1`.**

**The specialist being replaced must be selected outer-fold pure** — for each held-out fold `k`,
the replacement decision uses only the other four folds, is then frozen, and is evaluated on fold
`k`. This is what makes the number honest, and the per-fold choice must appear in the output JSON
so the purity claim is verifiable.

`marginal_vs_clone` answers the question the program actually cares about: does TabM or RealMLP
contribute information beyond what we would get merely by swapping an incumbent specialist for
another exchangeable LightGBM model?

### C1.5 The result JSON — what it must contain

Print a small JSON to the logs between the markers. Required fields, per learner:

- `standalone TS-AUC` (mean, pooled, per-fold, fold std)
- `dominant-cell AUC`
- `rho` vs `RT-401`; `rho` vs RT-600 where available
- `E0`, `E1`, `E2`
- `marginal_vs_clone`, `E2_minus_E0`
- per-fold `E2 − E1` deltas, and the positive-fold count
- whole pair repairs / damage / net
- dominant-cell repairs / damage / net
- mature-vs-never repairs / damage / net
- mature-vs-prebreak repairs / damage / net
- **the replacement specialist selected for each fold**
- the final verdict against the frozen ladder

The dominant cell is the diagnostic region `t ≥ 200`, with all negative rows at those times and
positive rows whose break age is at least 100. It matters because prior forensic work showed it
carries **50.50%** of official same-time pair weight and **45.29%** of RT-600's remaining ranking
loss (`PROGRAM_PLAN.md` §1.3).

**But dominant-cell AUC is a diagnostic, not the optimization criterion.** The final criterion
remains whole-development incremental ensemble value.

### C1.6 Gates — frozen, unchanged

| Verdict | Condition |
|---|---|
| `KILL` | `marginal_vs_clone < +0.0010` |
| `INTERESTING` | `≥ +0.0010` |
| `PROMOTION_WORTHY` | `≥ +0.0015`, `≥4/5` folds positive, dominant pair net `> 0` |
| `SERIOUS` | `≥ +0.0030`, `≥4/5` positive, dominant pair net `> 0`, mature-vs-never pair net `> 0` |
| `MAJOR` | `≥ +0.0050` |

**Standalone AUC is not the promotion criterion.** A model with mediocre standalone can be
valuable if it fixes errors RT-600 makes. **Low correlation is not sufficient either** — earlier
neural work established that a model can be very different from RT-600 and still provide no
ensemble alpha. CRF-01 is the standing proof: ρ `0.446`, repair-Jaccard `0.174`, and uniformly
negative pair flow. **What matters is complementary correct pair ordering.**

### C1.7 Hard prohibitions for C1

- **Do not** attempt to extract the neural OOF arrays from Crunch — no printing arrays, no base64
  encoding, no log-encoding of files, no workarounds. It is unnecessary and out of bounds (R4b).
- **Do not** tune anything during this recovery.
- **Do not** allocate new RT IDs.
- **Do not** combine with `RT-1257` yet. `FULL_OOF_PREREG.md` authorizes **no combination** —
  not TabM + RealMLP, not either with `RT-1257` or CatBoost — **even if results are excellent.**
  Any heterogeneous combination is a separate preregistered experiment (that is `C3`).
- **Do not** change the ensemble, touch test or lockbox data, or modify production.

**Establish the exact binding ensemble value of `RT-1258` and `RT-1259` first.** Only after that
result exists should anyone decide whether TabM, RealMLP, or neither deserves parameter tuning or
heterogeneous combination.

---

## C2. DEPLOYMENT INFERENCE BENCHMARK

**Type:** engineering measurement, not a scientific experiment.
**Blocked by:** nothing for `RT-1257`. **Start this immediately, in parallel with the C1
investigation.**
**IDs:** none unless a new arm is trained.

### C2.1 Why this is overdue

`reports/catboost_specialist_2026/FINAL.md` closes with: *"inference adds one CatBoost model per
surviving replacement and needs separate deployment benchmarking"* and *"Next action: review
surviving CatBoost specialist(s) against deployment cost before any production work."*

**That review has never happened.** `RT-1257` is `PROMOTION_WORTHY` at `marginal_vs_clone
+0.002407205` with 5/5 folds positive — and undeployed, with unmeasured inference cost.

**Rule for this program: no arm is described as promotable until its online, streaming,
per-timestep inference cost has been measured against the competition budget.** A model that
cannot run online is not a candidate regardless of its `marginal_vs_clone`. This gates every
promotion in both lanes, which is why it starts at t=0 rather than waiting for L2.

### C2.2 Scope

1. **`RT-1257` as it stands** — the two-slot hybrid (CatBoost in the `RT-300` and `RT-413` slots,
   LightGBM elsewhere). Measure per-timestep online inference cost in the platform's own runtime,
   using the methodology of the RTX 4090 hardware benchmark (submission `76357`, task
   `run-3e834e0f`) so the numbers are comparable to the existing budget projections.
2. **The best-`k` hybrid**, when handoff **H2** arrives from LOCAL `L2`. If `k*` is 3 or 4, that
   is three to four CatBoost models in the inference path rather than two — the exposure scales
   with `k` and the measurement must be redone, not extrapolated.

Report against the competition's streaming budget explicitly: not just "it takes X ms" but
"X ms against a budget of Y, at k=2 and k=k\*".

### C2.3 What this is not

This is not a leaderboard submission and not a promotion. It produces an engineering number that
gates a promotion decision someone else makes later.

---

## C3. CROSS-FAMILY SLOT MIXING

**Type:** new experiment. **Requires its own preregistration.**
**Status:** **conditional — opens only on results from both lanes.**
**IDs:** `RT-1270`–`RT-1289`, unallocated until it opens.

### C3.1 Rationale

If LOCAL `L2` confirms the mechanism, the natural generalization is that the ensemble is not seven
LightGBM models to upgrade but **seven functional slots to be filled by whichever family is most
complementary in that slot.** `RT-1257` is a two-family mixture. There is a `research/xgb-gpu-2026`
branch in the remote snapshot, and C1 may deliver a third and fourth family (TabM, RealMLP).

### C3.2 Opening conditions

C3 opens **only if both**:

1. `L2`'s best-`k` hybrid reaches `≥ +0.0015` with `≥4/5` folds positive and positive dominant
   pair net, **and**
2. at least one non-LightGBM, non-CatBoost family has a measured single-slot
   `marginal_vs_clone ≥ +0.0010` — from C1, or from a separate XGBoost arm.

If `L2`'s hybrid is KILL, **C3 does not open.** The mechanism is falsified at its source and
adding families to a dead mechanism is fishing.

### C3.3 Scope constraint — this is the one that keeps C3 honest

C3 must be preregistered as a **fixed, enumerated set of assignments — not a search.** A free
per-slot family search over 7 slots × 4 families is 16,384 configurations; any such search needs a
multiplicity correction it would almost certainly not survive, and `RDOF_LEDGER.md` exists
precisely so that "+0.0005" can be read against the number of chances taken.

A defensible C3 scope: assign each slot the family with the highest **measured single-slot**
marginal for that slot, evaluate that one assignment plus its matched clone control, and stop. One
configuration, chosen by pre-existing measurements, zero search.

### C3.4 Open question

Is `research/xgb-gpu-2026` a live branch with usable artifacts, or a stub? It appears in the remote
snapshot but has no local worktree. C3's viability depends on it. This is open question **#8** and
it is yours.

---

## C4. NEURAL SEQUENCE MODELS

**Placed last deliberately. Read §C4.1 before §C4.3.**
**Status:** **gated. Does not open on your judgment — it opens on a LOCAL result (H3).**
**IDs:** `RT-1290`–`RT-1319` reserved, **unallocated until C4 opens.**

### C4.1 Why this is last, and exactly what would move it up

This is not a dismissal of neural models. It is a ranking derived from what has been measured in
this repository.

**The evidence against funding a new neural architecture right now:**

1. **The representation × objective factorial is complete and empty.** `CRF_FINAL.md` §E fills
   every cell. Representation effect `+0.0444`, objective effect `+0.0222`, total `+0.0666` over
   `RT-970`'s 0.52618 — reaching **0.59276**, still below the `0.600` necessary condition and
   roughly `0.043` below the fitted `+0.0030` contour at ρ ≈ 0.45. **`H-A` and `H-B` are closed.**
2. **Learned nulls are closed (`H-E`).** The fixed AR(5) + 256-knot per-series residual ECDF beat
   a correctly-conditioning learned null by `0.021446`, with the derangement control **passing**
   (`+0.003377` whole, `+0.008391` dominant) — so the learned null was not broken; it conditioned
   correctly and lost anyway. The failure is amortization.
3. **The information frontier says the missing signal is post-`t`.** `W7-D3R` Arm B — same
   information, more capacity — was **negative** (`−0.00592`, 4/5 folds). **`H-D`** is what
   survives, and no *causal* model reaches post-`t` information.
4. **The measured bottleneck is arbitration, not detection** (`PROGRAM_PLAN.md` §1.6). CRF-01
   repairs **39.5%** of dominant-cell mistakes at repair-Jaccard **0.174** — genuinely its own
   repairs — and damages **26.8%** of what RT-600 already had right, against a **0.0150** cap. A
   better detector does not address that.
5. **`corr(standalone, ρ) = +0.983` across 17 scored arms.** Nothing this project has built has
   ever been both good and different, and `RT-1234` moving that frontier by `+0.0092` did not
   change the answer.
6. **The real cost is the preflight, not the training.** CRF-01 required 24 green preflight gates,
   bitwise prefix-invariance proof, a fold-purity positive control reproducing a known
   contaminated scheme, and a checkpoint provenance fingerprint added only after
   `RT-1237/1238/1239` were voided by a unit-test null — a defect caught by compute accounting
   (`pretrain_runtime_s = 0.2` against a real 1,364.9 s), not by the state-sha check that existed.

**What would move C4 up, concretely:**

| Trigger | Effect |
|---|---|
| **H3 succeeds** — LOCAL's probe finds a gating rule retaining ≥50% of CRF-01's repairs under a <0.05 damage rate | **C4 opens.** A retention mechanism would be demonstrated, and a better detector feeding it becomes worth building. **This is the main path.** |
| **C1 returns TabM `≥ +0.0010`** | Raises the prior that a non-tree family can extract retainable alpha from the same features. **Not sufficient alone.** |
| **L2 returns KILL across all four slots and `k* = 2`** | Learner-family mixing is exhausted; the ensemble lane closes and C4 becomes the *least bad* remaining option **by elimination rather than by evidence**. If this is why C4 opens, **say so explicitly** — do not dress elimination up as evidence. |
| **C1 returns both KILL and H3 fails** | **C4 should not be funded.** Two independent lines say new architectures on this information do not retain. Redirect to deployment robustness, which is `CRF_FINAL.md`'s own recommendation. |

### C4.2 Standing constraints on any neural arm

Any C4 arm, whatever the architecture, inherits all of `PROGRAM_PLAN.md` §2 plus:

- **Preserve the fixed per-series null.** `H-E` is closed. The AR(5) + 256-knot historical residual
  ECDF is paid for by each series' own break-free history at zero generalization cost. Expose its
  residual / PIT / exceedance streams directly as input channels. **Do not compress it into a
  small learned global embedding** — CRF-02 measured that trade at `−0.021446`.
- **Same-`t` pairwise ranking objective**, with negative sampling matching the TS-AUC comparison
  structure. The `+0.0222` whole / `+0.0371` dominant objective effect is measured and real; use
  it, do not re-derive it.
- **Full CRF-01-grade preflight before any score.** Bitwise prefix invariance at `atol=0.0`,
  truncation and batch-composition invariance, a fold-purity positive control, a checkpoint
  provenance fingerprint that halts the run on mismatch, and test-isolated caches.
- **The cheap abandon gate stays.** Fold-0 standalone `< 0.600` → abandon, folds 1–4 not trained.
  It has fired twice and saved substantial compute both times.
- **Binding endpoint is `marginal_vs_clone` under nested replacement**, not standalone TS-AUC and
  not low ρ. Note explicitly: **no CRF arm ever reached this endpoint.** A C4 arm clearing the
  abandon gate would be the first neural candidate ever measured on it.
- **Deployment profile is part of the result, not an afterthought (§C2).** A selective SSM running
  causally at every online timestep is a fundamentally different inference profile from a
  gradient-boosted tree.

### C4.3 Candidate architectures, ranked

From the deep-research report, with `PROGRAM_PLAN.md` §3.3's corrections applied.

**C4-1 — Fixed-Null Selective State-Space Ranker (Mamba-family).** *The report's headline
proposal.* A causal selective SSM over the null-normalized sequence channels — input-dependent
state parameters allowing content-dependent retention and forgetting, linear sequence scaling —
trained under the same-`t` pairwise objective on top of the preserved fixed per-series null.

*The case for:* CRF-01's TCN had receptive field ≈ 253. If the missing structure is long-horizon
excursion evolution conditional on a strong individualized null, a finite convolutional field
cannot see it and a selective recurrence can. Mamba / Mamba-3 are **architectural motivation only,
not competition-performance evidence**, and carry no weight in the gate ladder.

*The case against, which must appear in the prereg:* this is `H-A` reopened. `W7-D3R` Arm B says
more capacity on the same information is negative. `H-D` says the residual gap is post-`t`. And
§C4.1(4) says the failure was retention, which memory horizon does not obviously fix.

*Required framing:* C4-1 must be preregistered as **"does longer effective memory change the
retention geometry?"** — with pair flow and damage rate as co-primary readouts alongside
standalone — **not** as "does a better representation raise standalone TS-AUC?" That second
question is answered and the answer is no.

**C4-2 — Memory isolation control.** Same channels, same objective, same head; a deliberately
memory-truncated recurrence against the full-state version. This establishes whether long memory is
actually load-bearing or whether C4-1's result came from something else. **If C4-1 runs, C4-2 is
not optional.** The CRF program's entire value came from its matched ladders
(`RT-970 → RT-1235 → RT-1234`); a C4-1 without its isolation control would be strictly weaker
evidence than CRF-01 already was.

**C4-3 — Repair-preserving objective term.** An explicit penalty on reversal of incumbent-correct
training-fold pairs, inside the loss. **Prior art the report missed:** `SS-02`
(`RT-1223`/`RT-1224`) already ran a same-`t` residual pair loss with `damage_penalty: 2.0` and
dominant-pair weighting, scoring `−0.000290` with dominant net `−108`. C4-3 is not novel; it is
SS-02's mechanism relocated into a deep sequence model's training loop. That relocation is a real
difference and may matter — but **the prereg must cite SS-02 and say why this time is different.**
C4-3 runs only if C4-1 clears the abandon gate with non-trivial standalone signal, holding
representation fixed.

*Constraint:* outer-fold RT-600 predictions may enter only where the protocol permits, for
training-objective and control construction, under strict nested fold purity. They must never leak
validation ranks or future information. **Run the fold-purity sentinel.**

**C4-4 — xLSTM / state-augmented recurrent family.** Lower priority. Evidence alignment to this
task is weaker than for selective SSMs and there is no mechanism-level argument specific to this
problem. Worth running only as a second architecture *after* C4-1 established that the memory axis
matters at all.

**C4-5 — Mixture / sparse-expert tabular in-context learning (MixturePFN family).** A research
comparator, not a bet. The MixturePFN literature itself flags scaling difficulty as dataset size
grows, and this competition has million-row folds and hard streaming inference constraints. It is
unlikely to survive C2 even if it scored well.

### C4.4 Explicitly out of scope for C4

Carried from the report's "five lanes least worth tuning" and the repository's own
`FAILED_EXPERIMENTS.md`:

- Raw CRF-01-style TCN width / depth / epoch sweeps. The ladder is measured; scaling it is not a
  new hypothesis.
- Tiny amortized learned-null embeddings. `H-E` closed at `−0.021446`.
- Shallow or static post-hoc repair routers. SS-01 through SS-04, all KILL.
- Another iteration of future-aware distillation. Wave 8's five pilots (ORR, TGMC, SST, PCFB,
  CFEP) are all KILL on the full population.
- **Generic TabM / RealMLP hyperparameter search before C1's binding result exists.** This is C1's
  entire point. Only after `RT-1258`/`RT-1259` have a `marginal_vs_clone` should anyone decide
  whether either deserves tuning.
- `CRF-04`. `CRF_PROGRAM_PREREG.md` §0.8/§0.9 forbid inventing one, and `CRF-03` did not open
  because neither primary produced a `marginal_vs_clone` at all. Anything in C4 is a **new program
  with a new preregistration**, not a CRF continuation.

---

## C5. YOUR OPEN QUESTIONS

From `PROGRAM_PLAN.md` §9, the ones you own:

- **#1 — answer before any GPU time.** How does Crunch handle `model_directory_path` and
  `resources` across runs? Can the completed `RT-1258`/`RT-1259` artifacts be mounted or reused?
  **Do not assume inheritance.**
- **#7 (shared)** Competition deadline and remaining submission budget — not recorded in
  `STATUS.md` or `PROTOCOL.md`. This constrains how much of C3/C4 is realistically fundable.
- **#8** Is `research/xgb-gpu-2026` a live branch with usable artifacts or a stub?

---

## C6. START HERE

1. Create your worktree from the plan branch (`PROGRAM_PLAN.md` §5.1). Work nowhere else — use
   `git show <branch>:<path>` to inspect the other lane, never a direct path.
2. **Answer open question #1** (§C1.1) and write it up. This is unblocked right now and it decides
   whether C1 is cheap or expensive.
3. **Start C2 on `RT-1257`** in parallel. Also unblocked, also overdue, and it gates every
   promotion in the program.
4. When **H1** arrives, **re-verify all eight sha256 hashes and the `4,032,524` finite count before
   spending compute.** A mismatch fails the run early; a missed mismatch produces a wrong number.
5. Run C1 — Branch A if reuse works, Branch B if it does not. **No tuning, no new IDs, no
   combination.**
6. Capture the result JSON between the markers and deliver **H4** to the LOCAL agent.
7. **Do not start a neural model.** C4 opens on **H3** from the LOCAL agent's arbitration probe —
   a result that costs them minutes and could save you days.
