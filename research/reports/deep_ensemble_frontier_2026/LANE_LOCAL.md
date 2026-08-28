# DEEP ENSEMBLE FRONTIER 2026 — **LOCAL LANE**

**Agent:** LOCAL
**Branch:** `research/deep-ensemble-frontier-local-2026`
**Worktree:** `/path/to/workspace/structural-break-deep-ensemble-frontier-local-2026`
**Hardware:** Apple M2 Pro, 10 cores, 16 GB RAM
**ID range:** **`RT-1260` – `RT-1269` — you may not allocate outside this range**
**Status:** `PLAN_ONLY — NO EXPERIMENT AUTHORIZED BY THIS DOCUMENT`

> **Read [`PROGRAM_PLAN.md`](PROGRAM_PLAN.md) in full first.** It holds the verified state,
> the binding constraints, the gate ladder, the audit of the deep-research report, and the
> coordination protocol. This file assumes all of it and does not repeat it.
>
> **You own `research/STATUS.md`.** You are its single writer for this program. The CRUNCH
> agent does not edit it.

---

## L0. WHY THIS LANE MATTERS MORE THAN ITS HARDWARE SUGGESTS

The instinct that the powerful hardware does the important work is wrong here, and it is worth
saying plainly before you start.

**The single highest-expected-value experiment in this program runs on your laptop in under two
CPU-hours** (§L2). It completes the one mechanism that has ever produced material positive
marginal ensemble alpha in this project, and it was left half-finished for reasons of prereg scope
rather than evidence.

**The gate on the entire expensive neural lane is a zero-training probe you run in minutes**
(§L3). Whether the CRUNCH agent spends days on a selective state-space model depends on a result
you produce from `.npy` files already sitting on disk.

**And the GPU program — 4.5 GPU-hours already spent, both learners trained, results unfiled — is
blocked on eight files on your filesystem** (§L1). Not on compute, not on scoring code, not on the
neural predictions. Eight `.npy` files that are `.gitignore`d and therefore never travelled.

Your lane is not the support lane. Plan accordingly.

---

## L1. PACKAGE AND VALIDATE THE EIGHT CONTROL OOF ARRAYS

**Type:** data packaging and validation for handoff **H1**. **No experiment, no score, no ID.**
**Cost:** ~1 hour.
**Blocked by:** nothing. **Do this first** — it is short and it unblocks the CRUNCH agent's main
task.

### L1.1 Why this is the whole blocker

`FULL_OOF_PREREG.md` preregistered `RT-1258` (TabM) and `RT-1259` (RealMLP) on 2026-08-27,
allocated both IDs, and froze both configurations. The authorization basis is recorded: the RTX
4090 hardware benchmark (Crunch submission `76357`, task `run-3e834e0f`) showed TabM projects to
**3.63606 h** for five folds and RealMLP to **0.90924 h**, combined **4.545298927912005 h**,
against a 15-h/2-learner/0.90-fraction quota. No shrink was required; both configurations are
frozen in `FROZEN_GPU_CONFIG.json` (commit `d506aa6`).

**The run executed and produced correct five-fold OOF for both learners.** `RDOF_LEDGER.md`
nonetheless records the program with **"Result not yet filed"**, and the run reported verdict
`PENDING_LOCAL_BINDING_EVALUATION`.

The reason is narrow and entirely fixable by you. `submissions/H_gpu_tabular_full_oof.py::
_try_binding_replacement_test()` calls `load_control_oof(_REPO_ROOT)`, which requires exactly
**eight** arrays at `<root>/research/oof/`:

```
RT-300  RT-401  RT-410  RT-411  RT-412  RT-413  RT-414  RT-415
```

`research/oof/` is `.gitignore`d (line 60), so those eight files were never part of the submitted
tree. When they are absent the function returns `{"computed": False, ...}` and the binding test is
skipped. **When they are present it computes the entire `E0`/`E1`/`E2` battery in the cloud** and
prints it between `=== GPU_TABULAR_OOF_RESULTS_BEGIN ===` and `=== GPU_TABULAR_OOF_RESULTS_END ===`.

So: **the scoring logic already exists, the neural predictions already exist, and the only missing
input is eight files that never left this Mac.** Your job is to make them travel.

Leaving a preregistered experiment unfiled is exactly what preregistration exists to prevent, and
it biases the record — an unfiled result is disproportionately likely to be a null one.

### L1.2 The arrays, verified 2026-08-28

Canonical location: `structural-break-learner-diversity-2026/research/oof/`. **All eight are
symlink chains** (`learner-diversity` → `wave5` → `claude-wave3`); real payload **8 × 19.2 MiB =
153.6 MiB**.

| ID | Rows | dtype | Finite | Finite frac | sha256 (of the resolved file) |
|---|---:|---|---:|---:|---|
| `RT-300` | 5,036,517 | float32 | 4,032,524 | 0.800657 | `bd3e6456fef300a7d0fefacec92095724ab2401fbe046c28733c9600b96879d3` |
| `RT-401` | 5,036,517 | float32 | 4,032,524 | 0.800657 | `16e1a10e75f3a2adf12d0efdfe8f4dd9391cfb53f871909a3f7392d2f175f57e` |
| `RT-410` | 5,036,517 | float32 | 4,032,524 | 0.800657 | `cbe1282006585cdcceb020b8c67ba62e2cc8be551bb93b81d9a19bac8ec628cb` |
| `RT-411` | 5,036,517 | float32 | 4,032,524 | 0.800657 | `f08248b3c47683046e4531637179a198df5cdcb024539ddc098eb88ca1158612` |
| `RT-412` | 5,036,517 | float32 | 4,032,524 | 0.800657 | `2684cec6917c22b44af0b37f5e5b055e6159d7d452b0989bdc21cee0f73a3ae0` |
| `RT-413` | 5,036,517 | float32 | 4,032,524 | 0.800657 | `f101827affa4394e83b007c22cadaa06f1f4021771615a6db02c17c50b6ba4ac` |
| `RT-414` | 5,036,517 | float32 | 4,032,524 | 0.800657 | `31e98d95f7453d2a9bdf22ad877df6c1e8a032a9b7d12464c76d77885921972f` |
| `RT-415` | 5,036,517 | float32 | 4,032,524 | 0.800657 | `04e6cc609a4791021d164f8037a1f2df8f8bca9618d408a97cdd7253ee0127b6` |

**The identical finite count across all eight is the alignment evidence.** 4,032,524 / 5,036,517 =
80.0657% is exactly the five canonical dev folds; the remaining ~20% is the held-out lockbox,
correctly `NaN`. If any array in your package disagrees on that count, it is not canonically
aligned and the run must fail before GPU time is spent.

These hashes are the reference values for the H1 manifest. **Recompute them at packaging time and
compare** — if one has changed since 2026-08-28, stop and find out why before shipping it.

### L1.3 Steps

1. **Dereference when copying. This is the main hazard.** The eight files are symlinks; a plain
   `cp`, `git archive`, or `tar` without dereferencing ships 82-byte dangling links. Use
   `cp -L`, `tar -h`, or `rsync -L`, then **verify the copies are ~19.2 MiB each**, not 82 bytes.
2. **Validate every array before packaging**, per the agent-facing checklist: expected length
   (5,036,517), canonical alignment (finite count 4,032,524), finite required predictions, correct
   RT ID, sha256, fold/provenance, dtype and shape. **If any control is missing or malformed, fail
   here — before any GPU work begins.**
3. **Resolve open question #2:** confirm the exact path the submitted tree expects.
   `load_control_oof()` reads `<root>/research/oof/{ID}.npy`. **Follow the repository's existing
   convention rather than inventing a new one.** If the platform requires a `resources/`-style
   staging path, the shim that relocates them to `research/oof/` must be explicit and committed,
   not improvised in the submission.
4. **Write the manifest** to `reports/deep_ensemble_frontier_2026/local/H1_CONTROL_MANIFEST.json`:
   per-file ID, resolved source path, bytes, shape, dtype, finite count, sha256, plus the
   provenance chain (which worktree each resolves into) and the date. **Commit the manifest. Never
   commit the `.npy` files** (`AGENTS.md`).
5. **Hand off H1** and tell the CRUNCH agent the manifest path. They re-verify every hash before
   spending GPU time (R11).

### L1.4 What you are *not* doing here

- **You are not running `evaluate_gpu_oof.py` locally.** The neural OOF stays in Crunch; the
  binding test runs where the data is.
- **You are not pulling arrays down from Crunch.** Not by download, not by log printing, not by
  base64. See `PROGRAM_PLAN.md` §3.2 — this is explicitly out of bounds and unnecessary.
- **You are not allocating an RT ID.** `RT-1258`/`RT-1259` are already allocated. This is filing
  an existing experiment, not opening a new one.

### L1.5 Interpretation guide for the result when it returns (H4 → §L4)

The user-supplied standalone landmarks are **TabM ≈ 0.60065** and **RealMLP ≈ 0.55991**. These
are **not** the answer to the ensemble question — T2 / `RT-995` is the standing proof that a
strong standalone can collapse to `+0.00024` marginal.

One comparison is worth fixing in advance so it cannot be constructed afterwards:

**TabM's ≈0.60065 standalone clears the 0.600 bar that CRF-01 failed at 0.592762.** If TabM
nonetheless returns a negligible or negative `marginal_vs_clone`, that is direct evidence that
(a) the 0.600 standalone screen is not a sufficient predictor of ensemble value, and more
importantly (b) **a new architecture on the same 500 features does not buy retention.** That is
the most informative thing this task can contribute to the C4 decision, and it is why the result
must reach the CRUNCH agent whichever way it lands.

### L1.6 Constraint carried over

`FULL_OOF_PREREG.md` authorizes **no combination** in that program — not TabM + RealMLP, not
either with `RT-1257`, not either with CatBoost — **even if results are excellent.** Any
heterogeneous combination is a separate preregistered experiment. A survivor becomes an input to
CRUNCH `C3`, not an immediate blend.

Likewise: **standalone AUC is not the promotion criterion, and low correlation is not sufficient.**
A model with mediocre standalone can be valuable if it fixes errors RT-600 makes; a model that is
very different from RT-600 can still provide no ensemble alpha — CRF-01 is the standing proof.
What matters is complementary correct pair ordering. And **dominant-cell AUC is a diagnostic, not
the optimization criterion** — the final criterion remains whole-development incremental ensemble
value.

---

## L2. **CSA-04 — COMPLETE THE CATBOOST SLOT SWEEP**

**This is the highest expected-value item in the entire program.**
**Type:** new experiment. **Requires preregistration before any score.**
**Cost:** ~1.6–1.9 CPU-hours of training plus evaluation.
**IDs:** `RT-1260`–`RT-1263` (four slots), `RT-1264` (best-`k` hybrid).
**Blocked by:** nothing. **Start here.**

### L2.1 The gap in CSA

The CatBoost Specialist Activation 2026 preregistration fixed **three** slots a priori and made
two conditional:

- `CSA-01` primary: **CAT-413 only.**
- `CSA-02`, conditional on CSA-01 ≥ +0.0010: **CAT-300 and CAT-410 only.**
- `CSA-03`, conditional on ≥2 survivors: hybrid over **surviving slots only**.

That was a correct staged, gated design. But its scope was three of seven slots, chosen before
any evidence existed about which would respond. **`RT-411`, `RT-412`, `RT-414` and `RT-415` were
never trained with CatBoost.** They were not excluded by a result — they were excluded by the
shape of the original preregistration. **There is no killed-idea barrier here**, which matters
because `AGENTS.md` requires reading `FAILED_EXPERIMENTS.md` before proposing anything that
resembles a killed idea. This does not resemble one.

### L2.2 The measured evidence

| Arm | Slot replaced | Cols | `marginal_vs_clone` | Standalone | ρ vs RT600 | Dominant net | Runtime | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| CAT-413 | `RT-413` (FULL, `pairwise_t`) | 500 | **+0.001087151** | 0.620440244 | 0.714196 | 93 | 1527.1 s | INTERESTING |
| CAT-300 | `RT-300` (FULL, champion) | 500 | **+0.001029459** | 0.620242061 | 0.737786 | 75 | 2208.1 s | INTERESTING |
| CAT-410 | `RT-410` (3 modules) | 261 | +0.000753095 | 0.610328251 | 0.680825 | 17 | 1132.1 s | KILL |
| **HYBRID `RT-1257`** | 413 + 300 | — | **+0.002407205** | — | — | **158** | — | **PROMOTION_WORTHY** |

**Fact 1 — the hybrid was super-additive on the metric.**
Sum of components: `0.001087151 + 0.001029459 = 0.002116610`.
Measured hybrid: `0.002407205`. **Excess +0.000290595, ratio 1.137.**
Two independent-family replacements did not merely add; they reinforced.

**Fact 2 — but sub-additive on pair flow.** Dominant net: `93 + 75 = 168` against a measured
`158`. Super-additive on the metric, sub-additive on pair flow. **This deserves an explanation
in the final report rather than being passed over** — it is a hint about how the two replacements
interact, and pair net is the strongest known correlate of `marginal_vs_clone` (Pearson 0.879).

**Fact 3 — per-slot alpha appears to scale with the column breadth of the slot replaced.**

| Slot | Cols | `marginal_vs_clone` |
|---|---:|---:|
| `RT-413` | 500 | +0.001087 |
| `RT-300` | 500 | +0.001029 |
| `RT-410` | 261 | +0.000753 |

Both 500-column slots landed at ≈+0.00105; the 261-column slot landed 29% lower and failed the
gate. **`n = 3` is far too small to call this a law.** It must be preregistered as a falsifiable
prediction, not assumed. But it generates a concrete testable ordering over the untested slots,
which is exactly what a good prereg needs.

Untested slots by breadth: `RT-412` **500**, `RT-415` **500**, `RT-411` **239**, `RT-414` **170**.

### L2.3 Preregistered predictions — commit these before any CSA-04 score exists

**P1 — breadth hypothesis.** `RT-412` and `RT-415` (both 500 cols) yield `marginal_vs_clone`
near the ≈+0.00105 observed for `RT-413`/`RT-300`. `RT-411` (239 cols) and `RT-414` (170 cols)
yield materially less, plausibly below the `+0.0010` gate as `RT-410` (261 cols) did.

**P2 — idiosyncrasy hypothesis, which contradicts P1 on the same two slots.** The alpha comes
from *heterogeneity*, so replacing the slots whose LightGBM idiosyncrasy is *hardest for CatBoost
to reproduce* should **destroy** more existing diversity than it adds. `RT-412`
(`extra_trees=True` + `sample_mode="per_series"`) and `RT-415` (`boosting="goss"`) are precisely
those slots. Under P2 they underperform P1's expectation and may be negative.

**P1 and P2 make opposite predictions about `RT-412` and `RT-415`.** That is what makes CSA-04 a
real experiment rather than a sweep. The outcome discriminates between "CatBoost is simply a
better learner on wide feature banks" and "the alpha is irreducibly about family mixing." Record
both before scoring.

**P3 — interior maximum.** The `k`-slot hybrid curve is **non-monotone with an interior
maximum**. At `k = 7` every slot is CatBoost, the ensemble is single-family, and the mixing alpha
that produced `RT-1257` is gone by construction. Therefore some `k* < 7` maximizes
`marginal_vs_clone`. **Locating `k*` is the primary scientific deliverable of CSA-04**, and it is
a question no amount of additional single-slot testing answers.

### L2.4 Design

**Learner — frozen, zero degrees of freedom.** Exact `RT-1251` CatBoost settings, verbatim from
CatBoost `PREREG.md`:

```
iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=5.0,
loss_function=Logloss, eval_metric=Logloss, bootstrap_type=Bernoulli,
subsample=0.7, rsm=0.7, border_count=127, thread_count=2,
allow_writing_files=False, verbose=False
```

**No CatBoost tuning is authorized.** No `depth`, `iterations`, `learning_rate`, `l2_leaf_reg`,
`rsm`, `subsample` or `border_count` search. This is the same learner that produced
`RT-1254`/`RT-1255`/`RT-1256`, which is what makes CSA-04's numbers directly comparable to
CSA-01/02 rather than a fresh, incomparable sweep.

**Per-arm contract.** Take the incumbent slot's modules, rows, labels, folds, row-sampler mode,
`max_train_rows` cap and **incumbent training seed** exactly as in `PROGRAM_PLAN.md` §1.2, and
vary **only the learner family**.

| ID | Arm | Slot | Modules | Cols | `max_train_rows` | `random_seed` | Sampler |
|---|---|---|---|---:|---:|---:|---|
| `RT-1260` | CAT-411 | `RT-411` | `m02_dist,m03_dyn,m04_resid,m06_loc` | 239 | 900,000 | 1 | uniform |
| `RT-1261` | CAT-412 | `RT-412` | FULL | 500 | 900,000 | 7 | **`per_series`** |
| `RT-1262` | CAT-414 | `RT-414` | `m07_bayes,m06_loc,m01_seq` | 170 | 700,000 | 3 | uniform |
| `RT-1263` | CAT-415 | `RT-415` | FULL | 500 | 700,000 | 11 | uniform |

**Two specification items to settle in the prereg, not after (open questions #3 and #4):**

- **`RT-412` / per-series sampling.** Row-sampling mode is part of the *data contract* and must
  be preserved, so CAT-412 gets per-series sampling. **Confirm the CatBoost runner actually
  honours `sample_mode="per_series"`.** If it does not, that is a specification change to
  disclose in the prereg.
- **`RT-415` / GOSS.** GOSS is a **LightGBM boosting mechanism with no CatBoost analogue**, so it
  is simply not reproduced — CAT-415 uses the frozen Bernoulli/`subsample=0.7` bootstrap like
  every other CatBoost arm. **State this explicitly**, because it makes CAT-415 a less faithful
  slot reimplementation than CAT-411/412/414, and that asymmetry must be disclosed before its
  number exists rather than invoked afterwards to explain a bad result.

**Sampler convention.** CSA-01/02 used a **uniform** row sampler with the incumbent seed; the GPU
program used the **RT-401 sequential sampler, seed 1**. CSA-04 follows the **CatBoost
convention** for comparability with `RT-1254`–`RT-1256`. **Do not mix conventions within a
program.**

**Training order.** `RT-1260`, `RT-1261`, `RT-1262`, `RT-1263`, five folds each, deterministic
and resumable. **All four must complete before any of the four is evaluated.** No arm stops early
on a predictive-score basis; only on unrecoverable technical failure, which must be recorded.

**Log hygiene.** Fold logs may record fold number, runtime, rows, RAM, checkpoint status. **No
TS-AUC, no pair flow, no replacement gain, no rank correlation** in any per-fold log.

### L2.5 Evaluation

**Stage 1 — four independent single-slot replacements.** For each arm, the standard `E0`/`E1`/`E2`
battery, matched control `RT-401` in the same slot, full §L1.4 metric set, frozen ladder applied.

**Stage 2 — the hybrid curve, which is the actual deliverable.** Let `S` be the set of slots with
`marginal_vs_clone ≥ +0.0010` across **all six tested slots** (CAT-300, CAT-410, CAT-413 from CSA
plus the four new). Build hybrids at every `k` from 2 to `|S|`, adding slots in descending order
of measured single-slot marginal — **or by dominant pair net; pick one and freeze it before
scoring** (open question #5). For each `k`:

- `E2_k` = seven-member ensemble with the top-`k` surviving slots CatBoost-replaced.
- `E1_k` = matched control with **the same `k` slots** replaced by seed clones from
  `RT-401..RT-406` in frozen specialist order.
- Report `marginal_vs_clone`, `E2 − E0`, per-fold deltas, folds positive, and all four pair-flow
  splits.

**MANDATORY REGRESSION CHECK.** If the slot ordering puts 413 and 300 first, the `k = 2` hybrid
**must reproduce `RT-1257`'s `+0.002407205` exactly**. If it does not, something in the evaluation
harness has drifted and **no CSA-04 number is trustworthy until it is explained. Halt and
investigate.** This is R2 and it is marked critical.

Allocate **`RT-1264`** to the best-`k` hybrid. **No blend-weight optimization at any `k`** —
equal weight, fold-pure `SCDF_NSEEN`, always.

### L2.6 Expected value, with its downside stated

**Upside.** If P1 holds and `RT-412`/`RT-415` deliver ≈+0.00105 each, a 4-slot hybrid at the
observed 1.137 super-additivity ratio lands near
`(0.001087 + 0.001029 + 0.00105 + 0.00105) × 1.137 ≈ +0.0048` — at or just below `MAJOR`
(`≥ +0.0050`) and comfortably past `SERIOUS` (`+0.0030`).

**Central.** Super-additivity decays as slots are consumed (P3), the narrow slots fail the gate as
`RT-410` did, `k*` lands at 3–4, hybrid ≈ **+0.0032 to +0.0040**. Still crosses `SERIOUS`, and
would be **the largest measured marginal in the project's recorded history** — roughly 1.5×
`RT-1257`.

**Downside.** P2 dominates: replacing `RT-412`/`RT-415` destroys the extra-trees and GOSS
idiosyncrasies that were carrying real diversity, all four new slots fall below the gate,
`k* = 2`, and CSA-04 confirms `RT-1257` is already the peak. **This is still a valuable result** —
it converts "we stopped at three slots because the prereg said three" into "learner-family mixing
is measured and exhausted at two slots," a real closure that justifies redirecting budget to
CRUNCH `C3` or `C4`.

**No outcome of CSA-04 leaves us where we started.** That is what makes it the right first
experiment.

### L2.7 Cost, memory, and risk

- **Compute.** Measured CSA runtimes 1527.1 s, 2208.1 s, 1132.1 s (4867.3 s for three arms at
  `thread_count=2`). Four more at comparable scale ≈ **1.6–1.9 CPU-hours**.
- **Memory.** No feature rebuild — the 12.61 GB cache at
  `structural-break-claude-wave3/cache/features/` is memory-mappable per module. Largest training
  subset is 1,000,000 × 500 × 4 B = **2.0 GB** resident; CSA-04's largest is 900,000 × 500 × 4 B =
  **1.8 GB**. Comfortable on 16 GB. Prefer `mmap_mode='r'` and slice rows rather than loading
  whole modules.
- **Research degrees of freedom.** Four new arms, all configurations inherited, **zero tuning
  knobs.** About as DOF-cheap as an experiment gets — which matters given `RDOF_LEDGER.md`'s
  framing that "+0.0005" must be read against the number of chances taken.
- **Risk — multiple comparisons (R1).** Four more single-slot tests against a `+0.0010` gate.
  Mitigation: gate fixed in advance, all four trained before any scored, and **the binding
  deliverable is the hybrid curve**, not any individual slot. A single slot squeaking over
  `+0.0010` is not a result.
- **Risk — bagging confound (R7).** `FINAL_ARCHITECTURE_FREEZE.md` measured that `+0.00499` of the
  `+0.00832` specialist delta is reproducible by seed variation alone. The `E1` seed-clone control
  is exactly the right control and is already in the protocol — **say so in the report** rather
  than assuming the reader knows.
- **Risk — deployment cost (R5).** Every surviving CatBoost slot adds one CatBoost model to
  inference. A 4-slot hybrid quadruples `RT-1257`'s exposure. **Send H2 to the CRUNCH agent as
  soon as `k*` is known** — do not wait for the final report.

---

## L3. ARBITRATION PROBE — THE GATE ON THE NEURAL LANE

**Type:** descriptive diagnostic on existing artifacts. **Zero training.**
**Cost:** minutes of CPU.
**IDs:** none consumed if it stays descriptive — same basis as CSA-00, "descriptive and ID-free
because it uses existing OOF vectors only."
**Blocked by:** nothing.
**Consumer:** handoff **H3** to CRUNCH `C4`. **This decides whether the neural lane opens at all.**

### L3.1 The question

`PROGRAM_PLAN.md` §1.6 established that the measured bottleneck is retention, not detection.
CRF-01 repairs **39.5%** of dominant-cell mistakes with repair-Jaccard **0.174** against the seed
clone — those repairs are real and are its own — while damaging **26.8%** of RT-600-correct pairs
against a **0.0150** cap.

**The question:** does there exist *any* gating, abstention, or confidence-weighted combination of
CRF-01's **existing** predictions with RT-600 that retains a material fraction of those repairs
while bringing the damage rate near the cap?

This is a question about a **combination rule over frozen prediction vectors**, not about a model.
It costs nothing to answer.

### L3.2 Everything needed is already on disk

`structural-break-causal-representation-frontier/research/oof/`:

| File | Contents | Coverage |
|---|---|---|
| `RT-1234.npy` | CRF-01 NNCSR candidate | `(5036517,)` float32, **806,334 finite (16.01%) = fold 0 only** |
| `RT-1235.npy` | matched BCE-objective control | same coverage |
| `RT-1240/1241/1242.npy` | CRF-02 candidate, fixed-null control, deranged control | same coverage |
| `RT-300.npy`, `RT-401.npy`, `RT-410`–`RT-415.npy` | full RT-600 control set | complete |

Folds 1–4 of the CRF arms were deliberately never trained, so **this is a fold-0-only study by
construction.** State that limitation everywhere the result is cited. A single-fold descriptive
diagnostic is not a confirmation and cannot promote anything.

### L3.3 Design

**Arms — a small, fixed, preregistered set of gating rules.** Illustrative; finalize in the prereg:

1. **Baseline.** Unconditional equal-weight blend of `RT-1234` into the RT-600 slot. Reproduces
   the known-bad behaviour and establishes the reference.
2. **Confidence-gated.** Use `RT-1234` only where its calibrated score is in the top/bottom `q`
   quantiles of its own within-`t` distribution; otherwise defer entirely to RT-600. Sweep `q`
   over a **preregistered fixed grid**.
3. **Agreement-gated.** Use `RT-1234` only where it agrees in sign of deviation with RT-600, or
   only where RT-600's own within-`t` score is near its decision boundary.
4. **Cell-restricted.** Apply `RT-1234` only inside the dominant cell (`t ≥ 200`, break age
   `≥ 100`), deferring to RT-600 everywhere else.
5. **Abstention.** Explicit three-way output — repair / defer / abstain — with the abstain region
   set against the 0.0150 damage cap.

**Primary readout.** Per rule, on fold 0, in the dominant cell: repair count, damage count, net,
**damage rate on RT600-correct pairs against the 0.0150 cap**, and the resulting fold-0 `E2 − E1`.

**Success criterion, fixed in advance.** A rule "works" if it retains **≥ 50%** of the
unconditional repair count while bringing the pre-break damage rate on RT600-correct pairs **below
0.05** — still over the cap, but within an order of magnitude rather than the current 17×
overshoot. Deliberately generous: this is a **screen**, not a confirmation.

### L3.4 Prior art and honest expectations

**The prior is poor and must be stated.** `SS-01` (repair-damage arbitration), `SS-02` (residual
ranking with a `2.0` damage penalty), `SS-03` (null calibration) and `SS-04`
(specialist-disagreement routing) were all built to solve arbitration and all four are KILL, at
`−0.000310`, `−0.000290`, `−0.000299`, `−0.000312`. `FIRST_SWEEP_SYNTHESIS.md` §H4 records "static
error-manifold routing is weak."

**What is different:** all four SS arms arbitrated over candidates built on the *incumbent
500-column bank*, at ρ 0.86–0.89 against RT-600. `RT-1234` sits at **ρ 0.446** with repair-Jaccard
**0.174** — genuinely different error geometry, the one input the SS family never had. That is a
real distinction and it is the *only* argument for L3.

**If L3 fails,** the honest reading is that arbitration is hard **regardless** of how different the
candidate is — and that is decisive information about the neural lane, not a disappointment.

**L3 is cheap enough that its poor prior does not matter.** Minutes of CPU, and it is the
difference between funding a multi-day GPU program on evidence versus on architecture enthusiasm.

### L3.5 Deliver H3 either way

Write the verdict to `reports/deep_ensemble_frontier_2026/local/L3_ARBITRATION_PROBE.md` and tell
the CRUNCH agent. **Report a negative as clearly as a positive** — a negative L3 is the single
most budget-saving result available in this program.

---

## L4. FILE THE RETURNED RESULT JSON, AND LEDGER CUSTODY

**Blocked by:** handoff **H4** from CRUNCH `C1`. **Do not wait idle — L1, L2 and L3 come first.**

### L4.1 Filing `RT-1258` / `RT-1259` from the H4 JSON

The CRUNCH agent copies the block between `=== GPU_TABULAR_OOF_RESULTS_BEGIN ===` and
`=== GPU_TABULAR_OOF_RESULTS_END ===` out of the Crunch logs and commits it. **That JSON is
sufficient** — you do not need the neural OOF arrays to decide whether either learner survives.

Expected contents per learner, which you should check are all present before filing: standalone
TS-AUC, dominant-cell AUC, ρ vs `RT-401`, `E0`, `E1`, `E2`, `marginal_vs_clone`, `E2_minus_E0`,
per-fold deltas, positive fold count, whole/dominant/mature-never/mature-prebreak
repairs/damage/net, **the replacement specialist selected for each fold**, and the verdict.

The per-fold replacement choice matters: `PROGRAM_PLAN.md` §2.3 requires it be made outer-fold
pure, using only the other four folds. If the JSON does not record it, the purity claim is
unverifiable and you should say so rather than filing silently.

Then: `RESULTS.csv` rows for `RT-1258` and `RT-1259`, `reports/gpu_tabular_2026/FINAL.md`,
`RDOF_LEDGER.md` closure of the "Result not yet filed" line, and a `FAILED_EXPERIMENTS.md` entry
for each KILL. If C1 had to regenerate the OOF, record that it was a **technical recovery** under
the frozen config — not a new arm, no new ID.

Apply the frozen ladder (`PROGRAM_PLAN.md` §2.4) and the §L1.5 interpretation guide. Send the
verdict onward: a TabM survivor feeds CRUNCH `C3`, and either outcome informs `C4`.

### L4.2 Ongoing ledger custody

You are the **single writer of `research/STATUS.md`** for this program (`PROGRAM_PLAN.md` §5.3).
The CRUNCH agent reports status changes to you rather than editing it.

- Append `RESULTS.csv` rows **only for IDs in `RT-1260`–`RT-1269`**, plus `RT-1258`/`RT-1259` on
  H4 (those are already-allocated IDs being filed, not new allocations).
- Append `EXPERIMENT_ID_MAP.md` entries within your range.
- Append your program's degrees-of-freedom section to `RDOF_LEDGER.md` **before** each score, not
  retrospectively.
- Write a `FAILED_EXPERIMENTS.md` entry for every KILL, with the falsification condition and why
  it failed. `AGENTS.md` calls this mandatory, not optional cleanup — it is what stops the next
  agent re-running a killed idea.
- Update `STATUS.md` **only if** the production anchor, external score, or active research
  conclusion actually changes. Keep it to one screen.
- On merge back to `research/deep-ensemble-frontier-2026`, resolve ledger conflicts by **keeping
  both sides** and sorting by ID. Never resolve by choosing one side.

**Do not commit** `.npy` arrays, checkpoints, caches, model binaries, logs, or notebook outputs.

---

## L5. YOUR OPEN QUESTIONS

From `PROGRAM_PLAN.md` §9, the ones you own:

- **#2** Confirm the exact path/filename convention the submitted tree expects for the eight
  control arrays. `load_control_oof()` reads `<root>/research/oof/{ID}.npy`. **Follow the
  repository's existing convention rather than inventing one.**
- **#3** CAT-415 / GOSS: caveat or exclude from the hybrid? **Decide before scoring.**
- **#4** CAT-412: does the CatBoost runner honour `sample_mode="per_series"`?
- **#5** Hybrid slot ordering: descending single-slot marginal, or dominant pair net? **Pick one
  and freeze it** — trying both is a degree of freedom.
- **#6** Does `RT-411`'s module set (`m02_dist,m03_dyn,m04_resid,m06_loc` — no `m00_core`, no
  `m07_bayes`) overlap with what CatBoost is good at? A descriptive look at the CSA
  feature-importance record before predicting its outcome is worth the ten minutes.
- **#7 (shared)** Competition deadline and remaining submission budget — not recorded in
  `STATUS.md` or `PROTOCOL.md`. Establish it; it constrains both lanes.

---

## L6. START HERE

1. Create your worktree from the plan branch (`PROGRAM_PLAN.md` §5.1). Work nowhere else.
2. **Do §L1 first — it is short and the CRUNCH agent is waiting on it.** Dereference the
   symlinks, validate all eight arrays against the §L1.2 reference table, write and commit the
   manifest, send **H1**.
3. Resolve open questions **#3, #4, #5** — they change the CSA-04 prereg text.
4. Write and commit `CSA04_PREREG.md` with predictions **P1, P2, P3** verbatim from §L2.3.
   **Commit it before any score exists.**
5. Train `RT-1260`–`RT-1263`, all five folds each, **all four before scoring any.**
6. In parallel with training, run **§L3** — minutes of CPU, and it unblocks the other agent's
   biggest decision. Send **H3** as soon as you have it.
7. Evaluate CSA-04: single slots, then the hybrid curve. **Check the `k = 2` regression against
   `+0.002407205` first — halt if it fails.**
8. Send **H2** (best-`k` composition) to CRUNCH as soon as `k*` is known.
9. When **H4** arrives, file `RT-1258`/`RT-1259` per §L4.1.
10. Adjudicate P1, P2 and P3 in writing — **including the ones that were wrong.**
