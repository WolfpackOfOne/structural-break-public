# CODEX BRIEF — DEEP ENSEMBLE FRONTIER 2026, **LOCAL LANE**

*Paste everything below the line into Codex as a single prompt.*

---

You are taking the **LOCAL lane** of the Deep Ensemble Frontier 2026 research program in the
`structural-break` repository (ADIA Lab / CrunchDAO Structural Break Challenge — Real-Time
Edition). A second agent is working the **CRUNCH lane** concurrently against the same repository.
Read this brief completely before running anything.

## 0. FIRST: YOU MUST WORK IN YOUR OWN GIT WORKTREE

**This is not optional and it is the first thing you do.**

This repository routinely has 15–20 active worktrees, one per branch, laid out as sibling
directories under `/path/to/workspace/`. Each belongs to a different research
session. `AGENTS.md` states the rule plainly: **never touch a worktree or branch you were not
asked to work in** — modifying, rebasing, or force-pushing another worktree's branch destroys
someone else's in-progress work. Right now a second agent is actively working the CRUNCH lane, so
this is a live concern, not a hypothetical one.

Create your own worktree and work **only** inside it:

```bash
cd "/path/to/workspace/structural-break-deep-ensemble-frontier-2026"
git fetch origin
git worktree add -b research/deep-ensemble-frontier-local-2026 \
    "/path/to/workspace/structural-break-deep-ensemble-frontier-local-2026" \
    research/deep-ensemble-frontier-2026
cd "/path/to/workspace/structural-break-deep-ensemble-frontier-local-2026"
git branch --show-current   # must print research/deep-ensemble-frontier-local-2026
```

Rules that follow from this, all binding:

- **Your worktree is `structural-break-deep-ensemble-frontier-local-2026`. Do not write anywhere
  else.** You will need to *read* files from other worktrees (the feature cache and the control
  OOF arrays live elsewhere) — reading is fine, writing is not.
- **Never `git push --force`.** If a push is rejected for divergence, stop and investigate. There
  is a documented case in `docs/repository_cleanup_audit.md` §C where two sessions diverged and
  neither side was overwritten — that is the standard.
- **Never rebase pushed history, never `git filter-repo` or BFG.** Preregistration commits,
  experiment IDs and negative results must stay reachable exactly as committed.
- Before anything that could discard work (`git checkout --`, `git restore`, `git reset --hard`,
  `git clean -fd`, removing a worktree), run `git status` **and** `git worktree list` first and
  stop if anything looks active that you did not create.
- Commit and push to **your own branch** as you go. Push is fine; force-push is not.

## 1. READ THESE, IN THIS ORDER, BEFORE DOING ANY WORK

1. `AGENTS.md` (repo root) — canonical agent instructions. Binding. If it conflicts with anything
   in this brief, **`AGENTS.md` and current repository state win.**
2. `research/PROTOCOL.md` — feature module and causality protocol. Binding.
3. `research/reports/deep_ensemble_frontier_2026/PROGRAM_PLAN.md` — the shared program core:
   verified state, binding constraints, the gate ladder, an audit of a prior research report, and
   the two-lane coordination protocol. **Read in full.**
4. `research/reports/deep_ensemble_frontier_2026/LANE_LOCAL.md` — **your lane, in detail.** This
   brief is a summary of it; that file is the authority.
5. `research/STATUS.md` and `research/reports/catboost_specialist_2026/FINAL.md` — current state
   and the result you are extending.
6. `research/FAILED_EXPERIMENTS.md` — read before proposing anything resembling a killed idea.

Do not act on any older report, brief, or handoff without verifying it against current repository
state. Old reports are evidence, not live instructions.

## 2. CONTEXT IN ONE PARAGRAPH

Production is `RT-600`, a seven-specialist LightGBM ensemble, external Crunch score **0.6268**,
unchanged. Scoring is Time-Stratified AUC; inference is causal and streaming. The best measured
research result is `RT-1257`, a **two-slot** LightGBM/CatBoost hybrid: `marginal_vs_clone =
+0.002407205`, `E2−E0 = +0.002026322`, 5/5 folds positive, dominant-cell pair net `+158`,
mature-vs-never `+73`. It is `PROMOTION_WORTHY` but not `SERIOUS`, and it is undeployed. Across
two exhausted preregistered sweeps, a neural representation program, a distillation program and a
leaderboard-alpha program, **exactly one mechanism has ever produced material positive marginal
ensemble alpha: swapping the learner family in an existing specialist slot.** That mechanism is
currently **3 of 7 explored**. Your main job is to finish it.

## 3. HARD CONSTRAINTS — READ TWICE

- **No lockbox. No test data.** Never read `X_test.reduced.parquet`, `y_test.reduced.parquet`, or
  any path containing `reduced` or `test` in a competition-data directory. These files are present
  in several worktrees; **their presence on disk is not permission.**
- **True `tau` (the actual break location) is forbidden** as a feature or as any input to
  inference, directly or via a proxy.
- **Canonical folds only.** Series-level, permanent, from `research/folds/folds.parquet`, folds
  `0..4`. Never regenerate them, never use a row-wise split, never let two prefixes of one series
  straddle a train/validation boundary.
- **Preregistration before score.** Before running anything that can produce a headline number,
  write the hypothesis, the falsification condition and the exact protocol, and **commit it before
  the number exists.** Amending a prereg after seeing a result destroys the point. If a result
  forces a mid-run change of plan, register a new prereg and say so explicitly in the commit
  message.
- **Your experiment-ID range is `RT-1260` – `RT-1269`. Do not allocate outside it.** The CRUNCH
  agent owns `RT-1270`–`RT-1289` and `RT-1290`–`RT-1319`. `RT-1258`/`RT-1259` are already
  allocated to an existing experiment. A voided experiment keeps its ID; IDs are never reused.
- **Ledgers are append-only.** `research/RESULTS.csv`, `research/EXPERIMENT_ID_MAP.md`,
  `research/RDOF_LEDGER.md`, `research/FAILED_EXPERIMENTS.md`. Never edit or delete an existing
  row, including for rejected or voided experiments — the negative result is the point.
- **You are the single writer of `research/STATUS.md`** for this program. The CRUNCH agent does
  not edit it. Update it **only** if the production anchor, external score, or active research
  conclusion actually changes, and keep it to one screen.
- **Do not commit** `.npy` OOF arrays, checkpoints, caches, model binaries, logs, large plots, or
  notebook outputs. Commit markdown reports, compact JSON summaries, ledgers, preregistrations,
  source code, small manifests.
- **No tuning of any kind** is authorized in this lane: no hyperparameter search, no seed search,
  no blend-weight optimization, no feature selection, no router, no stacker.

## 4. THE BINDING EVALUATION PROTOCOL

Every candidate is judged by **nested slot replacement**, outer-fold pure — for each held-out fold
`k`, the replacement decision uses only the other four folds, is then frozen, and is evaluated on
fold `k`:

- **`E0`** = original seven-specialist RT-600.
- **`E1`** = six RT-600 specialists + an exchangeable matched LightGBM seed clone in the target slot.
- **`E2`** = six RT-600 specialists + the candidate in the target slot.
- **Primary metric: `marginal_vs_clone = E2 − E1`.** Also report `E2 − E0`, per-fold `E2 − E1`,
  and folds positive.

Multi-slot hybrids use the first unused seed clones from `RT-401..RT-406` in frozen specialist
order. Integration is **equal-weight, fold-pure `SCDF_NSEEN`** — fold `k`'s calibration fitted only
on the other four folds. **Replacement, never addition:** a prior experiment (`W4-E6`) tested 13
boosters against the seven specialists and scored **−0.00095**.

**Frozen gate ladder — do not move these after seeing a result:**

| Verdict | Condition |
|---|---|
| `KILL` | `marginal_vs_clone < +0.0010` |
| `INTERESTING` | `≥ +0.0010` |
| `PROMOTION_WORTHY` | `≥ +0.0015`, `≥4/5` folds positive, dominant pair net `> 0` |
| `SERIOUS` | `≥ +0.0030`, `≥4/5` positive, dominant pair net `> 0`, mature-vs-never pair net `> 0` |
| `MAJOR` | `≥ +0.0050` |

Pair-flow diagnostics: 64 same-`t` pairs per time point, seed `20260827`, splits `whole`,
`dominant_cell`, `mature_vs_never`, `mature_vs_prebreak`. Unchanged from prior programs so numbers
stay comparable.

**Standalone AUC is not the promotion criterion**, and low correlation is not sufficient. A prior
neural arm reached ρ 0.446 against RT-600 with genuinely novel repairs and was still worthless
because it could not keep them. What matters is complementary correct pair ordering.

## 5. YOUR FOUR TASKS

### TASK L1 — Package and validate eight control OOF arrays *(do this first; it is short)*

**The CRUNCH agent is blocked on this.** A completed cloud run trained TabM (`RT-1258`) and
RealMLP (`RT-1259`) over all five folds, but could not compute the binding replacement test
because eight control arrays are `.gitignore`d and never reached the cloud. The scoring code
already handles this in-cloud once the files are present. Your job is to make them travel.

The eight arrays live in `structural-break-learner-diversity-2026/research/oof/`:
`RT-300`, `RT-401`, `RT-410`, `RT-411`, `RT-412`, `RT-413`, `RT-414`, `RT-415`.

**Critical hazard: they are symlink chains** (`learner-diversity` → `wave5` → `claude-wave3`) and
list as 82 bytes. A plain `cp`, `git archive`, or `tar` will ship dangling links. **Dereference:**
`cp -L`, `tar -h`, or `rsync -L`. Verify each copy is ~19.2 MiB, **not 82 bytes**. Total payload
153.6 MiB.

**Validate every array before packaging.** All eight must show: shape `(5036517,)`, dtype
`float32`, and **finite count exactly `4,032,524`** (80.0657% — the five canonical dev folds; the
remaining ~20% is the held-out lockbox, correctly `NaN`). The identical finite count across all
eight is the canonical-alignment evidence. Reference sha256 values are tabulated in
`LANE_LOCAL.md` §L1.2 — **recompute and compare**; if any has changed, stop and find out why
before shipping it.

**If any control is missing or malformed, fail here** — before the other agent spends GPU time.

Write `research/reports/deep_ensemble_frontier_2026/local/H1_CONTROL_MANIFEST.json` with per-file
ID, resolved source path, bytes, shape, dtype, finite count, sha256, and the provenance chain.
**Commit the manifest; never commit the `.npy` files.** Then tell the CRUNCH agent the manifest
path — that is handoff **H1**.

Confirm the exact destination path convention the submitted tree expects rather than inventing
one: the loader reads `<root>/research/oof/{ID}.npy`.

### TASK L2 — CSA-04: complete the CatBoost slot sweep *(your headline job)*

**IDs: `RT-1260`–`RT-1263` for the four arms, `RT-1264` for the best-`k` hybrid.**

A prior program (CatBoost Specialist Activation 2026) tested CatBoost in **three** of the seven
specialist slots and stopped — not because the rest failed, but because its preregistration fixed
three slots a priori. Measured results:

| Arm | Slot | Cols | `marginal_vs_clone` | Verdict |
|---|---|---:|---:|---|
| CAT-413 | `RT-413` | 500 | +0.001087151 | INTERESTING |
| CAT-300 | `RT-300` | 500 | +0.001029459 | INTERESTING |
| CAT-410 | `RT-410` | 261 | +0.000753095 | KILL |
| **HYBRID `RT-1257`** | 413+300 | — | **+0.002407205** | **PROMOTION_WORTHY** |

The two-slot hybrid was **super-additive**: components sum to `+0.002116610`, measured
`+0.002407205`, excess `+0.000290595`, ratio **1.137**.

**The four untested slots**, to be run with the *incumbent* slot's modules, rows, labels, folds,
sampler mode, row cap and training seed — varying **only** the learner family:

| ID | Arm | Slot | Modules | Cols | `max_train_rows` | seed | sampler |
|---|---|---|---|---:|---:|---:|---|
| `RT-1260` | CAT-411 | `RT-411` | `m02_dist,m03_dyn,m04_resid,m06_loc` | 239 | 900,000 | 1 | uniform |
| `RT-1261` | CAT-412 | `RT-412` | FULL (500) | 500 | 900,000 | 7 | **`per_series`** |
| `RT-1262` | CAT-414 | `RT-414` | `m07_bayes,m06_loc,m01_seq` | 170 | 700,000 | 3 | uniform |
| `RT-1263` | CAT-415 | `RT-415` | FULL (500) | 500 | 700,000 | 11 | uniform |

`FULL` = `m00_core,m01_seq,m02_dist,m03_dyn,m04_resid,m06_loc,m07_bayes` = exactly 500 columns.

**Frozen learner — zero degrees of freedom.** Use the exact settings that produced the existing
arms, so your numbers stay comparable:

```
iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=5.0,
loss_function=Logloss, eval_metric=Logloss, bootstrap_type=Bernoulli,
subsample=0.7, rsm=0.7, border_count=127, thread_count=2,
allow_writing_files=False, verbose=False
```

**No feature rebuild is needed.** A 12.61 GB prebuilt cache exists at
`structural-break-claude-wave3/cache/features/` as per-module `.npy` + `.cols.json`. Memory-map it
(`mmap_mode='r'`) and slice rows. Largest resident training subset is 900,000 × 500 × 4 B = 1.8 GB
on a 16 GB machine.

**Preregister these three predictions verbatim before any score exists** (full text in
`LANE_LOCAL.md` §L2.3):

- **P1 (breadth).** The two 500-column slots (`RT-412`, `RT-415`) land near the ≈+0.00105 observed
  for the other 500-column slots; the narrow slots (`RT-411` 239 cols, `RT-414` 170 cols) land
  materially lower, plausibly below the gate as the 261-column `RT-410` did.
- **P2 (idiosyncrasy) — contradicts P1 on the same two slots.** The alpha comes from
  *heterogeneity*, so replacing the slots whose LightGBM idiosyncrasy CatBoost can least reproduce
  should *destroy* diversity. `RT-412` (`extra_trees=True` + per-series sampling) and `RT-415`
  (`boosting="goss"`) are exactly those. Under P2 they underperform and may be negative.
- **P3 (interior maximum).** The `k`-slot hybrid curve is non-monotone with a maximum at some
  `k* < 7`, because at `k=7` the ensemble is single-family and the mixing alpha is gone by
  construction. **Locating `k*` is the primary deliverable.**

**Two specification questions to settle in the prereg, not after:**
(a) `RT-415` uses GOSS, a LightGBM boosting mechanism with **no CatBoost analogue** — CAT-415 will
use the frozen Bernoulli bootstrap, making it a less faithful slot reimplementation. Decide *before
scoring* whether it is reported under a caveat or excluded from the hybrid.
(b) Confirm the CatBoost runner honours `sample_mode="per_series"` for CAT-412; if not, that is a
specification change to disclose in the prereg.

**Execution rules.** Train all four arms over all five folds. **All four must complete before any
of the four is scored.** No arm stops early on a predictive-score basis — only on unrecoverable
technical failure, which must be recorded. Fold logs may contain fold number, runtime, rows, RAM,
checkpoint status — **never** TS-AUC, pair flow, replacement gain, or rank correlation.

**Evaluation.** Stage 1: four independent single-slot replacements under the §4 protocol. Stage 2
— the actual deliverable — build hybrids at every `k` from 2 up to the number of slots clearing
`+0.0010` across all six tested slots, adding slots in descending measured single-slot marginal
(freeze that ordering choice in the prereg). For each `k`, the matched control replaces **the same
`k` slots** with seed clones.

**MANDATORY REGRESSION CHECK: if the ordering puts 413 and 300 first, the `k = 2` hybrid must
reproduce `+0.002407205` exactly. If it does not, HALT** — the evaluation harness has drifted and
no CSA-04 number is trustworthy until that is explained.

Expected cost: prior arms ran 1527 s / 2208 s / 1132 s at `thread_count=2`, so roughly **1.6–1.9
CPU-hours** for four.

Adjudicate P1, P2 and P3 explicitly in the final report — **including the ones that were wrong.**

### TASK L3 — Arbitration probe *(zero training, minutes, and it gates the other agent's biggest decision)*

The repository's own diagnosis is that the bottleneck is **arbitration, not detection**. A prior
neural arm (`RT-1234`) repairs **39.5%** of RT-600's dominant-cell mistakes with repair-Jaccard
**0.174** against the seed clone — the repairs are genuinely its own — while damaging **26.8%** of
what RT-600 already had right, against a **0.0150** cap.

**Question:** does *any* gating, abstention, or confidence-weighted combination of `RT-1234`'s
**existing** predictions with RT-600 retain a material fraction of those repairs while bringing
damage near the cap?

Everything needed is already on disk in
`structural-break-causal-representation-frontier/research/oof/`: `RT-1234.npy`, `RT-1235.npy`, and
the full RT-600 control set. **Note `RT-1234` is fold-0 only** — 806,334 finite rows (16.01%),
because folds 1–4 were deliberately never trained. This is a fold-0 descriptive diagnostic; state
that limitation everywhere it is cited. It cannot promote anything.

Preregister a small fixed set of gating rules (unconditional baseline, confidence-gated over a
fixed quantile grid, agreement-gated, dominant-cell-restricted, three-way abstention) and report
per rule: repair count, damage count, net, **damage rate on RT600-correct pairs against the 0.0150
cap**, and fold-0 `E2 − E1`.

**Success criterion, fixed in advance:** a rule works if it retains **≥50%** of the unconditional
repair count while bringing pre-break damage rate on RT600-correct pairs **below 0.05**.

**The prior is poor and you must say so.** Four prior arbitration experiments (SS-01 through
SS-04) are all KILL at −0.000310, −0.000290, −0.000299, −0.000312. What is different is that all
four arbitrated over candidates at ρ 0.86–0.89 against RT-600, whereas `RT-1234` sits at ρ 0.446 —
genuinely different error geometry, the one input those arms never had. **Report a negative as
clearly as a positive**: a negative here is the most budget-saving result available in this
program, because it tells the CRUNCH agent not to spend days on a neural model.

Write the verdict to `research/reports/deep_ensemble_frontier_2026/local/L3_ARBITRATION_PROBE.md`
and deliver it as handoff **H3**.

### TASK L4 — File the returned GPU result, and keep the ledgers

When the CRUNCH agent returns handoff **H4** — a small JSON block copied out of the Crunch logs
between `=== GPU_TABULAR_OOF_RESULTS_BEGIN ===` and `=== GPU_TABULAR_OOF_RESULTS_END ===` — file
`RT-1258` and `RT-1259`. **That JSON is sufficient; you do not need the neural OOF arrays**, and
nobody should attempt to extract them from Crunch.

Check the JSON contains, per learner: standalone TS-AUC, dominant-cell AUC, ρ vs `RT-401`, `E0`,
`E1`, `E2`, `marginal_vs_clone`, `E2_minus_E0`, per-fold deltas, positive fold count, all four
pair-flow splits, **the replacement specialist selected for each fold**, and the verdict. That last
item matters: the outer-fold-pure claim is unverifiable without it, and you should say so rather
than filing silently.

Then append `RESULTS.csv` rows, write `reports/gpu_tabular_2026/FINAL.md`, close the "Result not
yet filed" line in `RDOF_LEDGER.md`, and add `FAILED_EXPERIMENTS.md` entries for any KILL.

Ongoing: append `RDOF_LEDGER.md` degrees-of-freedom entries **before** each score, not
retrospectively; write a `FAILED_EXPERIMENTS.md` entry for every KILL with its falsification
condition; and update `STATUS.md` only on a real change of anchor, score, or conclusion.

## 6. ORDER OF WORK

1. Create the worktree (§0). Verify you are on `research/deep-ensemble-frontier-local-2026`.
2. Read §1's documents.
3. **L1** — package and validate the eight controls, commit the manifest, send **H1**. Short, and
   the other agent is blocked on it.
4. Settle the L2 specification questions (GOSS handling, per-series sampling, hybrid ordering).
5. Write and commit `CSA04_PREREG.md` with P1, P2, P3. **Before any score exists.**
6. Train `RT-1260`–`RT-1263`. All four before scoring any.
7. **L3** in parallel with training — minutes of CPU. Send **H3** as soon as you have it.
8. Evaluate CSA-04: single slots, then the hybrid curve. **`k=2` regression check first.**
9. Send **H2** — the best-`k` hybrid composition — to the CRUNCH agent as soon as `k*` is known;
   they need it for deployment inference benchmarking. Do not wait for your final report.
10. File **H4** when it arrives (L4).
11. Final report with P1/P2/P3 adjudicated, ledgers updated, branch pushed.

## 7. WHAT NOT TO DO

- Do not work outside your worktree, or write to any other branch.
- Do not force-push, rebase pushed history, or rewrite history by any means.
- Do not allocate an RT ID outside `RT-1260`–`RT-1269`.
- Do not touch lockbox or test data, or `X_test.reduced.parquet`.
- Do not tune CatBoost, search seeds, optimize blend weights, or add a router or stacker.
- Do not run `evaluate_gpu_oof.py` locally against neural predictions — they stay in Crunch.
- Do not attempt to extract arrays from Crunch by printing, base64, or any workaround.
- Do not start a neural model. That is the CRUNCH lane's C4 and it is gated on your L3 result.
- Do not promote anything to production. `RT-600` stays the anchor.
- Do not delete a report, branch, or `RESULTS.csv` row because a result was negative.

## 8. REPORT BACK

State plainly what ran, what the numbers were, which predictions were wrong, and what is blocked.
If tests fail or a step was skipped, say so with the output. If the `k=2` regression check fails,
report that immediately rather than continuing — it invalidates everything downstream of it.
