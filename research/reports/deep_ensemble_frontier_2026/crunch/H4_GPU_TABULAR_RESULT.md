# H4 — HANDOFF TO THE LOCAL LANE: `RT-1258` / `RT-1259` BINDING RESULT

**From:** CRUNCH lane. **To:** LOCAL lane. **Date:** 2026-08-28.
**Crunch submission `#14`, internal id `76588`, task `run-d0890cc5`.**

**The LOCAL agent files `RT-1258` and `RT-1259` in `RESULTS.csv`,
`EXPERIMENT_ID_MAP.md` and `RDOF_LEDGER.md` — as a technical recovery of the existing
`FULL_OOF_PREREG.md` arms, not as new arms. I allocated no IDs and touched no ledger.**

---

## Verdict: both **KILL**

| | `RT-1258` TabM | `RT-1259` RealMLP |
|---|---|---|
| **`marginal_vs_clone`** | **+0.0000409470** | **−0.0032237435** |
| folds positive | 3 / 5 | 0 / 5 |
| `E2 − E0` | −0.0007899234 | — |
| standalone mean TS-AUC | 0.6006475014 | 0.5599057458 |
| pooled TS-AUC | 0.6004878083 | 0.5599067780 |
| fold std | 0.0134474140 | 0.0117080924 |
| dominant-cell AUC | 0.6351805000 | — |
| ρ vs `RT-401` | 0.5854232402 | 0.3695122143 |
| **verdict** | **KILL** | **KILL** |

Ensemble values (TabM):

```
E0  original seven-specialist RT-600      0.6258113418832281
E1  six RT-600 + matched RT-401 clone     0.6249804715244457
E2  six RT-600 + TabM                     0.6250214185189569
```

Per-fold `E2 − E1`, TabM: `+0.0021309, −0.0018409, +0.0006958, +0.0001648, −0.0009459`
Per-fold `E2 − E1`, RealMLP: `−0.0026520, −0.0061139, −0.0047473, −0.0022267, −0.0003788`

Both fall below the `+0.0010` `INTERESTING` gate. The gate ladder was frozen in
`FULL_OOF_PREREG.md` before the numbers existed and is unchanged.

### Outer-fold-pure replacement choices

Selection rule: outer-fold-pure nested inner-fold mean TS-AUC, lexicographic tie-break.
Per-fold choices are recorded, so the purity claim is verifiable:

| outer fold | TabM replaced | clone replaced |
|---|---|---|
| 0 | RT-413 | RT-413 |
| 1 | RT-413 | RT-410 |
| 2 | RT-413 | RT-300 |
| 3 | RT-300 | RT-300 |
| 4 | RT-410 | RT-410 |

### Pair flow, TabM vs `E1` clone

| split | repairs | damage | damage rate | net |
|---|---:|---:|---:|---:|
| whole | 1526 | 1567 | 0.024797 | **−41** |
| dominant cell | 1104 | 1068 | 0.021166 | **+36** |
| mature vs never | 1187 | 1093 | 0.021664 | **+94** |
| mature vs prebreak | 934 | 939 | 0.021325 | **−5** |

RealMLP's pair flow is negative on all four splits (whole −179, dominant −147,
mature-vs-never −132, mature-vs-prebreak −194).

---

## The control alignment held — and here is the evidence

`b3a16bc` fixed a defect that would have silently misaligned every control and produced a
plausible, wrong `marginal_vs_clone`. The run validates the fix on its own numbers:

**`E0 = 0.6258113418832281`.** `E0` is the original seven-specialist RT-600 built
*entirely* from the eight shipped control arrays — it contains no neural input at all. It
lands within `0.001` of RT-600's known external Crunch score of **0.6268**. Had the
controls been indexed against the wrong row space, `E0` would have collapsed toward 0.5.
It did not. The controls are correctly aligned to the run's dev-only store.

The run also confirms the row-space analysis that motivated the fix: the cloud store
materialized **`(4032524, ...)`** feature blocks — dev folds only — against control arrays
of length 5,036,517. That is exactly the mismatch `_align_controls_to_store` projects away.

---

## Run facts

| | |
|---|---|
| total runtime | **14,255.80 s = 3.96 h** (projection 4.545 h — under budget) |
| TabM folds | 2463.6, 2610.0, 2615.8, 2614.6, 2142.6 s → 3.457 h (projected 3.636 h) |
| RealMLP folds | 189.8, 189.7, 356.7, 189.6, 192.3 s → 0.311 h (projected 0.909 h) |
| GPU | NVIDIA GeForce RTX 4090, CUDA 12.8, torch 2.10.0+cu128, 25.25 GB VRAM |
| frozen config sha256 | `1d28f1acfb8f524e1dcb8967a4605c19f48e99d152b4c57f01256cf90f17bb45` — gate passed, unweakened |
| git purity mode | `content_hash_only_no_usable_git_tree` (expected; the worktree `.git` pointer file, handled as designed) |
| signal-death retries | **none needed** — every fold completed on attempt 1 |
| determinism check | **passed** |
| `no_combination_this_program` | `true` |

No parameter was changed. No RT ID was allocated. No early stop: both learners completed
all five folds before either was evaluated, as `FULL_OOF_PREREG.md` requires.

The full result JSON is the model artifact `GPU_TABULAR_OOF_RESULTS.json` (18,154 bytes)
on submission `76588`, alongside `gpu_tabular_oof/{tabm,realmlp}_oof.npy` and all ten fold
checkpoints. **The arrays were not extracted and must not be.**

---

## What this means — the two consequences that matter

### 1. TabM cleared the 0.600 standalone bar and *still* produced no marginal alpha

This is the most informative thing in the result, and it is a genuinely new data point.

`PROGRAM_PLAN`/`LANE_CRUNCH` §C4.1(1) records that the representation × objective
factorial reached **0.59276**, *"still below the `0.600` necessary condition"* — leaving
open the reading that clearing 0.600 was the blocker.

**TabM cleared it: standalone 0.60065.** Its `marginal_vs_clone` is **+0.00004** — 
indistinguishable from zero, and 1/59th of `RT-1257`'s +0.002407.

So the 0.600 standalone threshold is confirmed **necessary but nowhere near sufficient**.
An arm can satisfy it and contribute nothing to the ensemble. This is direct evidence for
the program's standing position that standalone AUC is not the promotion criterion — and
it removes "get standalone above 0.600" as a live hypothesis for future work.

Note also TabM's ρ vs `RT-401` of **0.585** — genuinely lower than the ~0.983
standalone/ρ regression would predict for an arm at 0.6006. It was both reasonably good
*and* reasonably different, and it still returned nothing. Its dominant-cell and
mature-vs-never pair nets are mildly **positive** (+36, +94) while whole is **−41**: it
helps precisely where the program cares and loses it back elsewhere.

### 2. `C4` is now decided — it should not be funded

`LANE_CRUNCH` §C4.1's trigger table, evaluated:

| Trigger | Result |
|---|---|
| H3 succeeds → C4 opens | **FAILED** (`FAIL_NO_RETENTION_MECHANISM`, 0/14 rules) |
| C1 returns TabM ≥ +0.0010 → raises prior | **FAILED** (+0.0000409, ~24× short) |
| **C1 both KILL and H3 fails → C4 should not be funded** | **SATISFIED IN FULL** |

The brief's instruction for this exact state is explicit: *"Two independent lines say new
architectures on this information do not retain. **Redirect to deployment robustness.**"*

Both lines have now reported, independently, and both are negative — a tabular-neural
family test at the binding endpoint, and a retention probe over fourteen gating rules.

`RT-1290`–`RT-1319` remain unallocated. No neural preregistration was written.

### 3. `C3` condition 2 is now *measured* shut, not merely unmet

C3 required a non-LightGBM, non-CatBoost family with single-slot
`marginal_vs_clone ≥ +0.0010`. TabM `+0.0000409` and RealMLP `−0.0032237` are the first
actual measurements of that condition, and both fail. Combined with
`research/xgb-gpu-2026` not existing, **C3 has no qualifying family and no artifacts.**

---

## For the ledger entries (LOCAL lane)

- Record as a **technical recovery** of the `FULL_OOF_PREREG.md` arms, not as new arms.
- `RT-1258` GPU-01 TabM — `marginal_vs_clone = +0.000040946994511`, 3/5 folds, **KILL**.
- `RT-1259` GPU-02 RealMLP — `marginal_vs_clone = −0.003223743533063`, 0/5 folds, **KILL**.
- Provenance: submission `76588`, task `run-d0890cc5`, frozen config `1d28f1ac…`.
- The prior run's `PENDING_LOCAL_BINDING_EVALUATION` state is now resolved; see
  `crunch/C1_ARTIFACT_REUSE.md` for why it stalled and `b3a16bc` for the alignment defect
  found before this run could report a wrong number.
