# AGENT BRIEF — DEEP ENSEMBLE FRONTIER 2026, **CRUNCH LANE**

*Paste everything below the line into the agent as a single prompt.*

---

You are taking the **CRUNCH lane** of the Deep Ensemble Frontier 2026 research program in the
`structural-break` repository (ADIA Lab / CrunchDAO Structural Break Challenge — Real-Time
Edition). A second agent is working the **LOCAL lane** concurrently against the same repository.
Read this brief completely before running anything.

## 0. FIRST: YOU MUST WORK IN YOUR OWN GIT WORKTREE

**This is not optional and it is the first thing you do.**

This repository routinely has 15–20 active worktrees, one per branch, laid out as sibling
directories under `/path/to/workspace/`. Each belongs to a different research
session. `AGENTS.md` states the rule plainly: **never touch a worktree or branch you were not
asked to work in** — modifying, rebasing, or force-pushing another worktree's branch destroys
someone else's in-progress work. A second agent is actively working the LOCAL lane right now, so
this is a live concern, not a hypothetical one.

Create your own worktree and work **only** inside it:

```bash
cd "/path/to/workspace/structural-break-deep-ensemble-frontier-2026"
git fetch origin
git worktree add -b research/deep-ensemble-frontier-crunch-2026 \
    "/path/to/workspace/structural-break-deep-ensemble-frontier-crunch-2026" \
    research/deep-ensemble-frontier-2026
cd "/path/to/workspace/structural-break-deep-ensemble-frontier-crunch-2026"
git branch --show-current   # must print research/deep-ensemble-frontier-crunch-2026
```

Rules that follow, all binding:

- **Your worktree is `structural-break-deep-ensemble-frontier-crunch-2026`. Do not write anywhere
  else.** You will need to *read* from other worktrees; reading is fine, writing is not.
- **Do not write into the LOCAL agent's worktree** (`...-local-2026`). To inspect their work, use
  `git show research/deep-ensemble-frontier-local-2026:<path>`, never a direct filesystem path.
- **Never `git push --force`.** If a push is rejected for divergence, stop and investigate.
  `docs/repository_cleanup_audit.md` §C documents a real case where two sessions diverged and
  neither side was overwritten — that is the standard.
- **Never rebase pushed history, never `git filter-repo` or BFG.** Preregistration commits,
  experiment IDs and negative results must stay reachable exactly as committed.
- Before anything that could discard work, run `git status` **and** `git worktree list` first.
- **You do NOT edit `research/STATUS.md`.** The LOCAL agent is its single writer for this program
  — it gets rewritten rather than appended, so concurrent edits corrupt it. Report status changes
  to them.

## 1. READ THESE, IN THIS ORDER, BEFORE DOING ANY WORK

1. `AGENTS.md` (repo root) — canonical agent instructions. Binding. If it conflicts with anything
   in this brief, **`AGENTS.md` and current repository state win.**
2. `research/PROTOCOL.md` — feature module and causality protocol. Binding.
3. `research/reports/deep_ensemble_frontier_2026/PROGRAM_PLAN.md` — the shared program core:
   verified state, binding constraints, the gate ladder, an audit of a prior research report, and
   the two-lane coordination protocol. **Read in full**, especially §3.2 on the handoff direction.
4. `research/reports/deep_ensemble_frontier_2026/LANE_CRUNCH.md` — **your lane, in detail.** This
   brief summarises it; that file is the authority.
5. `research/reports/gpu_tabular_2026/FULL_OOF_PREREG.md` — the preregistration your first task
   completes. Binding on what you may and may not change.
6. `submissions/H_gpu_tabular_full_oof.py` — **read the docstring and
   `_try_binding_replacement_test()` before planning anything.** It already does most of C1.
7. `research/FAILED_EXPERIMENTS.md` — read before proposing anything resembling a killed idea.

Old reports are evidence, not live instructions. Verify claims against current repository state.

## 2. THE ONE THING TO UNDERSTAND BEFORE YOU START

**Your first task looks like a GPU job and is not one.**

`RT-1258` (TabM) and `RT-1259` (RealMLP) **already ran to completion.** The cloud run produced
complete five-fold OOF for both learners and uploaded them as model artifacts:

```
resources/gpu_tabular_oof/tabm_oof.npy          resources/gpu_tabular_oof/realmlp_oof.npy
gpu_tabular_oof/tabm/fold_{0..4}_pred.npy       gpu_tabular_oof/realmlp/fold_{0..4}_pred.npy
```

**The scoring code already exists too, and already knows how to run in the cloud.**
`submissions/H_gpu_tabular_full_oof.py::_try_binding_replacement_test()` calls
`load_control_oof(_REPO_ROOT)` and, when the controls are present, computes the entire `E0`/`E1`/`E2`
battery via `evaluate_gpu_oof.evaluate_learner()` and prints it between
`=== GPU_TABULAR_OOF_RESULTS_BEGIN ===` and `=== GPU_TABULAR_OOF_RESULTS_END ===`.

**The only missing input was eight `.npy` files** — the RT-600 specialists `RT-300`,
`RT-410`–`RT-415` and the matched clone `RT-401`. They live at `research/oof/`, which is
`.gitignore`d at line 60, so they were never part of the submitted tree. Absent them the function
returns `{"computed": False, ...}` and the run ends at `PENDING_LOCAL_BINDING_EVALUATION`. **That
is exactly where the last run stopped.**

**So C1 is not a new experiment.** It is putting eight already-existing control arrays into the
same environment as already-existing neural predictions, and running already-existing scoring
code. The LOCAL agent is packaging those eight files for you now — that is handoff **H1**.

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

## 3. CONTEXT IN ONE PARAGRAPH

Production is `RT-600`, a seven-specialist LightGBM ensemble, external Crunch score **0.6268**,
unchanged. Scoring is Time-Stratified AUC; inference is causal and streaming. The best measured
research result is `RT-1257`, a two-slot LightGBM/CatBoost hybrid at `marginal_vs_clone =
+0.002407205`, 5/5 folds positive — `PROMOTION_WORTHY`, and **undeployed with unmeasured inference
cost**. Across two exhausted preregistered sweeps, a neural representation program, a distillation
program and a leaderboard-alpha program, exactly one mechanism has ever produced material positive
marginal ensemble alpha: swapping the learner family in an existing specialist slot. The LOCAL
agent is finishing that sweep. Your jobs are to close out the GPU experiment, measure the
deployment cost that gates every promotion in the program, and — only if a gate opens — take the
neural lane.

## 4. HARD CONSTRAINTS — READ TWICE

- **No lockbox. No test data.** Never read `X_test.reduced.parquet`, `y_test.reduced.parquet`, or
  any path containing `reduced` or `test` in a competition-data directory. Their presence on disk
  is not permission.
- **True `tau` is forbidden** as a feature or any input to inference, directly or via a proxy.
- **Canonical folds only**, series-level and permanent, `research/folds/folds.parquet`, folds
  `0..4`. Never regenerate.
- **Preregistration before score.** Anything that can produce a headline number needs its
  hypothesis, falsification condition and protocol committed *before* the number exists. C1 is the
  exception only because it is completing an existing preregistration — do not treat that as a
  precedent for C3 or C4.
- **Your experiment-ID ranges are `RT-1270`–`RT-1289` and `RT-1290`–`RT-1319`** (the latter
  reserved for neural, unallocated until C4 opens). **Do not allocate outside them.** The LOCAL
  agent owns `RT-1260`–`RT-1269`. `RT-1258`/`RT-1259` are already allocated — **C1 allocates
  nothing.** A voided experiment keeps its ID; IDs are never reused.
- **Ledgers are append-only.** Never edit or delete an existing row in `RESULTS.csv`,
  `EXPERIMENT_ID_MAP.md`, `RDOF_LEDGER.md`, or `FAILED_EXPERIMENTS.md`.
- **You do not edit `STATUS.md`.** (§0)
- **Do not commit** `.npy` arrays, checkpoints, caches, model binaries, logs, or notebook outputs.
- **Write only under `research/reports/deep_ensemble_frontier_2026/crunch/`** for your own reports,
  so the two lanes stay disjoint and merge cleanly.

## 5. THE BINDING EVALUATION PROTOCOL

- **`E0`** = original seven-specialist RT-600.
- **`E1`** = six RT-600 specialists + the matched exchangeable `RT-401` LightGBM clone.
- **`E2`** = six RT-600 specialists + the candidate.
- **Binding metric: `marginal_vs_clone = E2 − E1`.**

**The replaced specialist must be selected outer-fold pure** — for each held-out fold `k`, the
decision uses only the other four folds, is then frozen, and is evaluated on fold `k`. The per-fold
choice must appear in the output JSON, or the purity claim is unverifiable.

Integration is equal-weight, fold-pure `SCDF_NSEEN`. Pair-flow diagnostics: 64 same-`t` pairs per
time point, seed `20260827`, splits `whole`, `dominant_cell`, `mature_vs_never`,
`mature_vs_prebreak`.

**Frozen gate ladder — do not move these after seeing a result:**

| Verdict | Condition |
|---|---|
| `KILL` | `marginal_vs_clone < +0.0010` |
| `INTERESTING` | `≥ +0.0010` |
| `PROMOTION_WORTHY` | `≥ +0.0015`, `≥4/5` folds positive, dominant pair net `> 0` |
| `SERIOUS` | `≥ +0.0030`, `≥4/5` positive, dominant pair net `> 0`, mature-vs-never pair net `> 0` |
| `MAJOR` | `≥ +0.0050` |

**Standalone AUC is not the promotion criterion**, and low correlation is not sufficient. A prior
neural arm reached ρ 0.446 with genuinely novel repairs (repair-Jaccard 0.174) and was worthless
because it could not keep them. What matters is complementary correct pair ordering. The
**dominant cell** (`t ≥ 200`, all negatives at those times, positives with break age ≥ 100) carries
50.50% of pair weight and 45.29% of RT-600's remaining ranking loss — but it is a **diagnostic, not
the optimization criterion.** The final criterion is whole-development incremental ensemble value.

## 6. YOUR FOUR TASKS

### TASK C1 — Get controls and neural OOF into one environment, run the binding test

**Allocates no IDs. Changes no parameters. This is a technical recovery, not new science.**

#### C1.1 First, before any GPU time: the artifact-reuse question

**Investigate how Crunch handles `model_directory_path` and the `resources` directory across
runs.** Determine specifically:

- Can the `RT-1258`/`RT-1259` model artifacts from the completed run (Crunch submission `76357`,
  task `run-3e834e0f`) be **mounted or reused** by a subsequent run?
- If so, by what mechanism, and does the reused artifact land at a path the submission can read?
- Does `resources/` persist across runs, or is it rebuilt from the submitted tree each time?

**Do not assume a new submission automatically inherits the previous submission's uploaded model
artifacts.** That assumption is the difference between a cheap evaluation session and a
4.545-GPU-hour regeneration, and being wrong in the optimistic direction wastes a run.

Write the answer to `research/reports/deep_ensemble_frontier_2026/crunch/C1_ARTIFACT_REUSE.md`
before launching anything. **This is unblocked right now — start here while waiting on H1.**

#### C1.2 Branch A — reuse works: evaluation-only recovery run

Preferred. No neural retraining.

1. Receive **H1** from the LOCAL agent: the eight-array control package plus
   `local/H1_CONTROL_MANIFEST.json`.
2. **Re-verify every sha256 against the manifest before spending any compute.** All eight must
   show shape `(5036517,)`, dtype `float32`, and **finite count exactly `4,032,524`** (80.0657% —
   the five canonical dev folds; the remaining ~20% is the lockbox, correctly `NaN`). **A mismatch
   on any single array fails the run before it starts.** This matters more than it looks: a
   misaligned control produces a *wrong `marginal_vs_clone`* rather than an error.
3. **Check the symlink hazard.** The arrays are symlink chains on the LOCAL machine. Verify what
   you received is ~19.2 MiB per file, **not 82 bytes**. Total payload 153.6 MiB.
4. **Place them where the code looks:** `load_control_oof()` reads `<root>/research/oof/{ID}.npy`
   for `RT-300`, `RT-401`, `RT-410`–`RT-415`. **Follow the repository's existing convention rather
   than inventing filenames.** If the platform requires a `resources/`-style staging path, the shim
   that relocates them must be explicit and committed, not improvised inside the submission.
5. Mount or reuse the existing neural artifacts and run the evaluation.
6. Capture the block between the markers. Deliver it as handoff **H4**.

#### C1.3 Branch B — reuse is impossible: technical recovery run

Only if C1.1 says the artifacts cannot be reused. **This is a technical recovery of an existing
experiment, not a new one** — which determines what you may change, and the answer is nothing.

- **Regenerate `RT-1258` and `RT-1259` from their exact frozen configurations.** TabM:
  `k=32, d_block=512, n_blocks=3, dropout=0.1, lr=0.002, weight_decay=0.0003, grad_clip_norm=1.0`.
  RealMLP: `hidden_sizes=[256,256,256], lr=0.04`. Both from `FROZEN_GPU_CONFIG.json` (commit
  `d506aa6`), enforced by content hash in `train()` — **do not weaken that check.**
- **Under no circumstances change any TabM or RealMLP parameter.** No batch size, epoch cap, `k`,
  `d_block`, hidden width, or layer count. A changed parameter makes this a new experiment
  requiring a new preregistration and new IDs.
- **Package all eight controls before the run starts**, so the binding test computes in the same
  session rather than deferring again.
- Matched contract unchanged: 500-column bank
  (`m00_core,m01_seq,m02_dist,m03_dyn,m04_resid,m06_loc,m07_bayes`), canonical folds `0..4`,
  1,000,000 max training rows per outer fold, RT-401 sequential sampler seed `1`, fold-pure
  `SCDF_NSEEN`.
- **Both candidates must complete before either is evaluated.** Neither may stop early because the
  other scored well — only on unrecoverable technical failure, which must be recorded.
- **Log hygiene.** Fold logs may contain only fold number, runtime, epochs, training/inner-validation
  diagnostics, RAM, VRAM, checkpoint status. **Never** TS-AUC, pair flow, replacement gain, or
  outer-fold rank correlation. No outer-fold TS-AUC after individual folds.
- Record it in `RDOF_LEDGER.md` as a **technical recovery**, not as a new arm.

Budget reference: TabM projects to 3.63606 h for five folds, RealMLP to 0.90924 h, combined
**4.545298927912005 h** against a 15-h / 2-learner / 0.90-fraction quota. No shrink was required.

#### C1.4 The evaluation — do not rewrite it

**Run the existing `evaluate_gpu_oof.py` logic. Do not rewrite the scientific scoring logic unless
you find an implementation bug** — and if you do, report it rather than silently patching it.

#### C1.5 The result JSON

Print a small JSON between `=== GPU_TABULAR_OOF_RESULTS_BEGIN ===` and
`=== GPU_TABULAR_OOF_RESULTS_END ===`. Required per learner: standalone TS-AUC (mean, pooled,
per-fold, fold std); dominant-cell AUC; ρ vs `RT-401` and vs RT-600 where available; `E0`, `E1`,
`E2`; `marginal_vs_clone`; `E2_minus_E0`; per-fold `E2 − E1` deltas and positive-fold count;
whole / dominant-cell / mature-vs-never / mature-vs-prebreak repairs, damage and net; **the
replacement specialist selected for each fold**; and the final verdict.

Deliver that block to the LOCAL agent as **H4**. They file `RT-1258`/`RT-1259` in the ledgers.

#### C1.6 Prohibitions specific to C1

- **Do not** extract the neural OOF arrays from Crunch — no printing arrays, no base64, no
  log-encoding, no workarounds. Unnecessary and out of bounds.
- **Do not** tune anything during this recovery.
- **Do not** allocate new RT IDs.
- **Do not** combine with `RT-1257` yet. `FULL_OOF_PREREG.md` authorizes **no combination** — not
  TabM + RealMLP, not either with `RT-1257` or CatBoost — **even if results are excellent.** Any
  heterogeneous combination is a separate preregistered experiment (that is C3).
- **Do not** change the ensemble, touch test or lockbox data, or modify production.

**Establish the exact binding ensemble value of `RT-1258` and `RT-1259` first.** Only after that
result exists should anyone decide whether TabM, RealMLP, or neither deserves tuning or
combination.

### TASK C2 — Deployment inference benchmark *(unblocked today; start in parallel with C1.1)*

`reports/catboost_specialist_2026/FINAL.md` closes with: *"inference adds one CatBoost model per
surviving replacement and needs separate deployment benchmarking"* and *"Next action: review
surviving CatBoost specialist(s) against deployment cost before any production work."*

**That review has never happened.** `RT-1257` is `PROMOTION_WORTHY` at `+0.002407205` with 5/5
folds positive — and undeployed, with unmeasured inference cost.

**Program rule: no arm is described as promotable until its online, streaming, per-timestep
inference cost has been measured against the competition budget.** A model that cannot run online
is not a candidate regardless of its `marginal_vs_clone`. This gates every promotion in both lanes,
which is why it starts now rather than waiting.

Scope:

1. **`RT-1257` as it stands** — CatBoost in the `RT-300` and `RT-413` slots, LightGBM elsewhere.
   Measure per-timestep online inference cost in the platform's own runtime, using the methodology
   of the RTX 4090 hardware benchmark (submission `76357`, task `run-3e834e0f`) so the numbers are
   comparable to existing budget projections.
2. **The best-`k` hybrid** when handoff **H2** arrives from the LOCAL agent. If `k*` is 3 or 4,
   that is three to four CatBoost models in the inference path instead of two — the exposure scales
   with `k` and must be **measured, not extrapolated**.

Report against the streaming budget explicitly: not "it takes X ms" but "X ms against a budget of
Y, at k=2 and at k=k\*". This is an engineering measurement, not a promotion and not a leaderboard
submission.

### TASK C3 — Cross-family slot mixing *(conditional; do not open on your own judgment)*

**IDs: `RT-1270`–`RT-1289`, unallocated until it opens. Requires its own preregistration.**

If the LOCAL agent's CSA-04 confirms the mechanism, the generalization is that the ensemble is not
seven LightGBM models to upgrade but **seven functional slots to be filled by whichever family is
most complementary in each.** `RT-1257` is a two-family mixture; C1 may deliver a third and fourth.

**C3 opens only if both:**

1. The LOCAL agent's best-`k` hybrid reaches `≥ +0.0015` with `≥4/5` folds positive and positive
   dominant pair net, **and**
2. at least one non-LightGBM, non-CatBoost family has a measured single-slot
   `marginal_vs_clone ≥ +0.0010` — from C1, or from a separate XGBoost arm.

If the hybrid is KILL, **C3 does not open.** The mechanism is falsified at its source and adding
families to a dead mechanism is fishing.

**Scope constraint — this is what keeps C3 honest.** C3 must be preregistered as a **fixed,
enumerated set of assignments, not a search.** A free per-slot family search over 7 slots × 4
families is 16,384 configurations; it would need a multiplicity correction it would not survive,
and `RDOF_LEDGER.md` exists precisely so "+0.0005" can be read against the number of chances taken.
A defensible scope: assign each slot the family with the highest **measured single-slot** marginal
for that slot, evaluate that one assignment plus its matched clone control, and stop.

Open question you own: is `research/xgb-gpu-2026` a live branch with usable artifacts, or a stub?
It appears in the remote snapshot but has no local worktree. C3's viability depends on it.

### TASK C4 — Neural sequence models *(last, and gated on a result from the other agent)*

**IDs: `RT-1290`–`RT-1319` reserved, unallocated until C4 opens. Requires a new preregistration —
it is a new program, not a continuation of the prior neural work.**

**Do not start this on your own judgment. C4 opens on handoff H3 from the LOCAL agent.**

#### C4.1 Why it is last

Not a dismissal — a ranking derived from what has been measured in this repository:

1. **The representation × objective factorial is complete and empty.** Representation effect
   `+0.0444`, objective effect `+0.0222`, total `+0.0666` over the `RT-970` baseline — reaching
   **0.59276**, still below the `0.600` necessary condition and ~`0.043` below the fitted `+0.0030`
   contour at ρ ≈ 0.45. Hypotheses `H-A` and `H-B` are **closed**.
2. **Learned nulls are closed (`H-E`).** The fixed AR(5) + 256-knot per-series residual ECDF beat a
   correctly-conditioning learned null by `0.021446`, with the derangement control **passing** — so
   the learned null was not broken; it conditioned correctly and lost anyway. The failure is
   amortization.
3. **The missing signal is post-`t`.** The `W7-D3R` arm with the same information and more capacity
   was **negative** (`−0.00592`, 4/5 folds). `H-D` survives, and no *causal* model reaches post-`t`
   information.
4. **The measured bottleneck is arbitration, not detection.** The prior neural arm repairs **39.5%**
   of dominant-cell mistakes at repair-Jaccard **0.174** — genuinely its own repairs — and damages
   **26.8%** of what RT-600 already had right, against a **0.0150** cap. A better detector does not
   address that.
5. **`corr(standalone, ρ) = +0.983` across 17 scored arms.** Nothing this project has built has ever
   been both good and different.
6. **The real cost is the preflight, not the training.** The prior neural arm required 24 green
   preflight gates, bitwise prefix-invariance proof, a fold-purity positive control reproducing a
   known contaminated scheme, and a checkpoint provenance fingerprint added only after three IDs
   were voided by a run that silently loaded a unit-test artifact — caught by compute accounting
   (`pretrain_runtime_s = 0.2` against a real 1,364.9 s), not by the integrity check that existed.

**What opens C4:**

| Trigger | Effect |
|---|---|
| **H3 succeeds** — the LOCAL probe finds a gating rule retaining ≥50% of the prior arm's repairs under a <0.05 damage rate | **C4 opens.** A retention mechanism would be demonstrated. **This is the main path.** |
| **C1 returns TabM `≥ +0.0010`** | Raises the prior that a non-tree family can extract retainable alpha. **Not sufficient alone.** |
| **CSA-04 returns KILL across all four slots and `k* = 2`** | The ensemble lane closes and C4 becomes the least-bad option **by elimination, not by evidence**. If this is why it opens, **say so explicitly.** |
| **C1 both KILL and H3 fails** | **C4 should not be funded.** Two independent lines say new architectures on this information do not retain. Redirect to deployment robustness. |

#### C4.2 Standing constraints on any neural arm

- **Preserve the fixed per-series null.** Expose its residual / PIT / exceedance streams directly.
  **Do not compress it into a small learned global embedding** — that trade was measured at
  `−0.021446`.
- **Same-`t` pairwise ranking objective**, negative sampling matching the TS-AUC comparison
  structure. The `+0.0222` whole / `+0.0371` dominant objective effect is measured — use it, do not
  re-derive it.
- **Full preflight before any score:** bitwise prefix invariance at `atol=0.0`, truncation and
  batch-composition invariance, a fold-purity positive control, a checkpoint provenance fingerprint
  that halts the run on mismatch, test-isolated caches.
- **The cheap abandon gate stays:** fold-0 standalone `< 0.600` → abandon, folds 1–4 not trained.
  It has fired twice and saved substantial compute both times.
- **Binding endpoint is `marginal_vs_clone`**, not standalone and not low ρ. Note that **no prior
  neural arm ever reached this endpoint** — an arm clearing the abandon gate would be the first.
- **Deployment profile is part of the result (C2).** A selective SSM running causally at every
  online timestep is a fundamentally different inference profile from a gradient-boosted tree.

#### C4.3 Candidates, ranked

- **C4-1 — Fixed-Null Selective State-Space Ranker (Mamba-family).** Causal selective SSM over
  null-normalized channels, input-dependent state parameters, linear scaling, same-`t` pairwise
  objective on top of the preserved fixed null. *For:* the prior TCN had receptive field ≈ 253; if
  the missing structure is long-horizon excursion evolution, a finite convolutional field cannot see
  it. Mamba/Mamba-3 are **architectural motivation only, not competition evidence.** *Against:* this
  is `H-A` reopened, and §C4.1(4) says the failure was retention, which memory horizon does not
  obviously fix. **Required framing:** preregister as *"does longer effective memory change the
  retention geometry?"* with pair flow and damage rate as co-primary readouts — **not** as "does a
  better representation raise standalone TS-AUC?" That question is answered and the answer is no.
- **C4-2 — Memory isolation control.** Deliberately memory-truncated recurrence vs the full-state
  version, channels and objective fixed. **If C4-1 runs, C4-2 is not optional** — the prior
  program's entire value came from its matched ladders.
- **C4-3 — Repair-preserving objective term.** **Prior art the source report missed:** a prior
  experiment (`SS-02`) already ran a same-`t` residual pair loss with `damage_penalty: 2.0` and
  dominant-pair weighting, scoring `−0.000290` with dominant net `−108`. C4-3 is that mechanism
  relocated into a deep model's training loop — a real difference, but **the prereg must cite SS-02
  and say why this time differs.** Runs only if C4-1 clears the abandon gate.
- **C4-4 — xLSTM / state-augmented recurrent.** Lower priority; weaker evidence alignment. Only
  after C4-1 establishes the memory axis matters.
- **C4-5 — Mixture / sparse-expert tabular ICL (MixturePFN family).** A comparator, not a bet. The
  literature flags scaling difficulty as dataset size grows; million-row folds and streaming
  inference make it unlikely to survive C2.

#### C4.4 Out of scope for C4

Raw TCN width/depth/epoch sweeps; tiny amortized learned-null embeddings; shallow or static
post-hoc repair routers (SS-01…SS-04, all KILL); another future-aware distillation iteration (Wave
8's five pilots, all KILL); **generic TabM/RealMLP hyperparameter search before C1's binding result
exists**; and a `CRF-04` — the prior program's preregistration forbids inventing one, so anything
here is a new program with a new preregistration.

## 7. ORDER OF WORK

1. Create the worktree (§0). Verify the branch name.
2. Read §1's documents, especially `PROGRAM_PLAN.md` §3.2 and the submission's
   `_try_binding_replacement_test()`.
3. **C1.1 — answer the artifact-reuse question and write it up.** Unblocked now; it decides whether
   C1 is cheap or expensive.
4. **C2 on `RT-1257`** in parallel. Also unblocked, also overdue, and it gates every promotion.
5. When **H1** arrives: **re-verify all eight sha256 hashes and the `4,032,524` finite count before
   spending compute.**
6. Run C1 — Branch A if reuse works, Branch B if not. No tuning, no new IDs, no combination.
7. Capture the result JSON between the markers; deliver **H4** to the LOCAL agent.
8. When **H2** arrives, benchmark the best-`k` hybrid's inference cost.
9. C3 and C4 only if their gates open — and if they do not, **decline them in writing with the gate
   result that declined them.**
10. Push your branch. Report.

## 8. WHAT NOT TO DO

- Do not work outside your worktree, or write into the LOCAL agent's worktree or branch.
- Do not edit `research/STATUS.md`.
- Do not force-push, rebase pushed history, or rewrite history by any means.
- Do not allocate an RT ID outside `RT-1270`–`RT-1289` / `RT-1290`–`RT-1319`, and allocate none at
  all for C1.
- Do not change any TabM or RealMLP parameter, or weaken the frozen-config hash check.
- Do not attempt to pull the neural OOF arrays out of Crunch by any means.
- Do not combine TabM, RealMLP, `RT-1257` or CatBoost in C1, even if results are excellent.
- Do not touch lockbox or test data, or modify production.
- Do not start a neural model until H3 opens C4.
- Do not treat the 0.500 leaderboard score as a problem to solve.
- Do not delete a report, branch, or `RESULTS.csv` row because a result was negative.

## 9. REPORT BACK

State plainly what ran, what the numbers were, and what is blocked. If a control array fails
verification, report it and stop rather than proceeding — a misaligned control produces a wrong
answer, not an error. If artifact reuse turns out to be impossible, say so before spending
4.5 GPU-hours, so the cost is a decision rather than a surprise.
