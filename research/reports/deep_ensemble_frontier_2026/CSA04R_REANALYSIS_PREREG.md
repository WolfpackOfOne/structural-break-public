# CSA-04R — HYBRID CURVE RE-ANALYSIS: PREREGISTRATION

**Program:** Deep Ensemble Frontier 2026
**Branch to execute on:** `research/deep-ensemble-frontier-local-2026` (LOCAL lane; merge this
file in from `research/deep-ensemble-frontier-2026`)
**Written:** 2026-08-28
**Status:** `PREREGISTERED — NOT YET EXECUTED`
**Type:** re-analysis of existing OOF vectors. **No training. No new model. No new data.**
**ID:** `RT-1265` (LOCAL range `RT-1260`–`RT-1269`), allocated **only if** the selected
composition differs from `RT-1264`'s. If the rule re-selects the same composition, no ID is
consumed and `RT-1264` stands.

---

## 0. HONESTY DECLARATION — READ FIRST

**The data this prereg governs already exists and has already been looked at.** `CSA04_RESULTS.json`
and `CSA04_RESULTS.csv` are committed at `766ecf0` on the LOCAL branch, and the `E2 − E0` column
that this re-analysis promotes to primary endpoint is visible in them right now.

This document therefore **cannot** and **does not** claim to be a blind preregistration. Pretending
otherwise would be worse than not writing it. Its purpose is narrower and still worth doing:

**To fix the decision rule, the tie-break, the noise threshold and the expected outcome in writing
BEFORE applying them — so that the conclusion cannot be reverse-engineered from whichever number
looks best.**

Under `AGENTS.md` §"Preregistration-before-score rule", amending an existing prereg after seeing a
result destroys the point. So this is registered as a **new, separately-named analysis** (CSA-04R)
rather than an amendment to `CSA04_PREREG.md`. CSA-04's own record stands unaltered: its arms, its
verdicts, its `RT-1264` allocation and its P1/P2/P3 adjudications are **not** retracted, edited or
deleted. §7 states exactly what CSA-04 got right.

**§6 states the outcome I expect before running this.** If the executed result contradicts §6, that
is a real finding. If it matches, the rule was not tuned to produce it.

---

## 1. THE DEFECT THIS CORRECTS

### 1.1 What happened

CSA-04 selected `k*` by maximising `marginal_vs_clone = E2 − E1`, where `E1` replaces **the same
`k` slots** with seed clones drawn from `RT-401..RT-406`. That control construction is inherited
from the CatBoost and GPU preregistrations, where it was applied at `k = 1` and once at `k = 2`.

**`E1` is not a fixed baseline. It degrades monotonically as `k` grows**, because replacing `k`
*heterogeneous* specialists with `k` near-identical seed clones destroys ensemble diversity by
construction:

| k | `E0` | `E1` | `E2` | `E1 − E0` | `E2 − E1` (CSA-04 endpoint) | `E2 − E0` |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.625811 | 0.623649 | 0.627251 | −0.002162 | +0.003602 | +0.001440 |
| 3 | 0.625811 | 0.622655 | 0.627483 | −0.003156 | +0.004828 | +0.001672 |
| 4 | 0.625811 | 0.622427 | 0.628051 | −0.003384 | +0.005624 | +0.002240 |
| 5 | 0.625811 | 0.622255 | 0.628190 | −0.003556 | **+0.005935** | +0.002378 |
| 6 | 0.625811 | 0.621779 | 0.627675 | −0.004033 | +0.005897 | +0.001864 |

### 1.2 Why this invalidates the ladder at high `k`, in one line

**At `k = 5`, a candidate that does nothing at all — `E2` exactly equal to `E0` — scores
`marginal_vs_clone = E0 − E1 = +0.003556`, which the frozen ladder labels `SERIOUS`.** At `k = 6`
a do-nothing candidate scores `+0.004033`, approaching `MAJOR`.

The thresholds (`+0.0030` SERIOUS, `+0.0050` MAJOR) were calibrated in a regime where `E1 ≈ E0`.
They do not survive translation to `k = 5`. The `k = 5` `MAJOR` verdict decomposes into
**`+0.003556` of control collapse and `+0.002378` of genuine gain over RT-600.**

### 1.3 The selection consequence, which is the substantive damage

Ordering the survivor slots by the two endpoints gives **near-opposite orderings**:

| rank | by `marginal_vs_clone` | by `E2 − E0` |
|---:|---|---|
| 1 | CAT-412 (+0.001630) | **CAT-413 (+0.001244)** |
| 2 | CAT-414 (+0.001589) | **CAT-300 (+0.001127)** |
| 3 | CAT-411 (+0.001567) | CAT-412 (+0.001030) |
| 4 | CAT-413 (+0.001087) | CAT-415 (+0.000907) |
| 5 | CAT-300 (+0.001029) | CAT-414 (+0.000552) |
| 6 | CAT-415 (+0.001001) | CAT-411 (+0.000363) |

CAT-411 moves from 3rd to last. CAT-413 moves from 4th to 1st. **The greedy curve built on the
second ordering starts with CAT-413 + CAT-300 — which is exactly `RT-1257`.**

The reversal already shows up in the committed numbers: CSA-04's `k = 2` (CAT-412 + CAT-414) has a
**higher** `marginal_vs_clone` than `RT-1257` (+0.003602 vs +0.002407) but a **lower** `E2 − E0`
(+0.001440 vs +0.002026). Ordering by the inflated endpoint put the weaker pair first.

A third signal points the same way and was not used at all: the three slots that ranked top on
`marginal_vs_clone` have the **lowest** mature-vs-never pair nets (CAT-412 `54`, CAT-414 `49`,
CAT-411 **`−9`**), while CAT-413 and CAT-300 have `80` each.

### 1.4 Attribution

**This defect is in the plan, not in CSA-04's execution.** `LANE_LOCAL.md` §L2.5 specified both the
`E1_k` control construction and the frozen ladder, and did not flag that the thresholds stop
meaning the same thing once `k` grows. The LOCAL agent executed the specification exactly as
written, froze the ordering choice in advance as instructed, and passed the mandatory `RT-1257`
regression check at `diff = 0.000e+00`. Holding the ladder fixed after seeing results was correct
behaviour, not a mistake.

---

## 2. WHAT CHANGES AND WHAT DOES NOT

**Minimal intervention. Exactly two things change.**

| Element | CSA-04 | CSA-04R | Changed? |
|---|---|---|---|
| Per-slot admission test | `marginal_vs_clone ≥ +0.0010` at `k = 1` | **identical** | **no** |
| Admitted survivor set | CAT-412, CAT-414, CAT-411, CAT-413, CAT-300, CAT-415 (CAT-410 excluded at `+0.000753`) | **identical** | **no** |
| Slot ordering for the greedy curve | descending `marginal_vs_clone` | **descending `E2 − E0`** | **YES** |
| Endpoint selecting `k*` | `marginal_vs_clone` | **`E2 − E0`** | **YES** |
| Reference for `E2 − E0` | — | `E0` = original seven-specialist RT-600, fixed across all `k` | new, fixed |
| Calibration / integration | equal-weight, fold-pure `SCDF_NSEEN` | identical | no |
| Folds | canonical `0..4` | identical | no |
| Pair-flow convention | 64 pairs/`t`, seed `20260827` | identical | no |
| OOF vectors | as trained | **identical — nothing retrained** | no |

**Why the per-slot test does NOT change.** `E2 − E1` at `k = 1` answers a genuine and different
scientific question: *does CatBoost carry ranking information an exchangeable LightGBM clone does
not, in this slot?* At `k = 1` the control degrades only mildly and the test is sound. That
question is already answered — six of seven slots, yes — and CSA-04R does not revisit it.
**CAT-410 stays excluded** at `+0.000753` even though its `E2 − E0` (`+0.000930`) would rank it 4th;
re-admitting it on the new endpoint would be exactly the post-hoc rule-shopping this document
exists to prevent.

**Why `E2 − E0` is the right endpoint for selecting `k`.** The question `k*` answers is a
deployment question — *ship the hybrid, or keep what we ship?* — and the counterfactual to
replacing `k` slots is replacing **none**, which is `E0`. `E0` is fixed across all `k`, so values
are comparable along the curve. `E1` is not fixed and is not the relevant counterfactual for that
question.

---

## 3. PROCEDURE

Pure OOF arithmetic on vectors already on disk. Expected runtime: minutes.

1. **Reproduce CSA-04 exactly first, as a harness check.** Recompute the committed
   `marginal_vs_clone` and `E2 − E0` for all seven single slots and all five hybrid points. **Every
   value must match `CSA04_RESULTS.json` to at least 1e-9.** If any does not, **HALT** and report —
   nothing downstream is trustworthy. This is the same discipline as CSA-04's own `RT-1257`
   regression check.
2. **Re-verify the `RT-1257` regression check** at `+0.002407204670070`. Must remain exact.
3. **Place `RT-1257` on the `E2 − E0` curve as the incumbent reference point.** Its
   `E2 − E0 = +0.002026322` is the number every CSA-04R composition must beat.
4. **Re-order the six admitted survivors by descending single-slot `E2 − E0`** (§1.3, right column).
   This ordering is **frozen by this document** and may not be re-derived after seeing curve
   results.
5. **Build the greedy nested curve** at every `k` from 1 to 6 under that ordering. For each `k`
   report:
   - `E0`, `E1`, `E2` (report `E1` for continuity with CSA-04, but it is **not** the endpoint)
   - **`E2 − E0`** — primary
   - per-fold `E2 − E0`, and folds positive
   - `E2 − E0` **relative to `RT-1257`**, per fold and pooled, with folds positive
   - pair flow vs `E0`: whole, dominant-cell, mature-vs-never, mature-vs-prebreak
     (repairs / damage / net)
6. **Paired bootstrap** (§4) for every comparison used in a decision.
7. **Apply the selection rule** (§5). Report `k*` and its composition.
8. **Descriptive appendix, cannot select `k*`:** enumerate all 63 non-empty subsets of the six
   admitted slots and report their `E2 − E0`. This costs nothing and is informative about the
   surface's shape. **It is explicitly barred from selecting `k*` or from supporting any promotion
   claim** — 63 comparisons on 5 folds would overfit, and `RDOF_LEDGER.md` exists so that "+0.0005"
   can be read against the number of chances taken. If the enumerated maximum exceeds the greedy
   `k*`, **report the gap and do not act on it**; treat it as a hypothesis for a future,
   independently-designed experiment.

---

## 4. NOISE MODEL — FIXED BEFORE ANY DECISION

Differences along this curve are small and the folds are shared, so paired inference is mandatory.

**Bootstrap.** Paired bootstrap resampling **series** (not rows — folds are series-level and rows
within a series are dependent), `B = 2000` replicates, seed `20260828`, fixed before execution.
Report the bootstrap SE and a 95% percentile interval for:

- each `k`'s `E2 − E0`
- `k*` candidate vs `RT-1257`
- adjacent curve points (`k` vs `k+1`) around the maximum

**Prior noise scale, from `FINAL_ARCHITECTURE_FREEZE.md`.** That document records, for the
specialist-vs-clone ensemble delta, `SD = 0.0011` across four data partitions (with fold SD 0.0085
and partition SD 0.0039 at single-model level). It is a related but not identical contrast, so it
is used only as a **floor**, not as the estimate.

**Operative threshold:**

```
δ_noise = max( paired bootstrap SE of the contrast , 0.0011 )
```

Two `E2 − E0` values differing by less than `δ_noise` are **indistinguishable** and may not be
ranked against each other.

---

## 5. SELECTION RULE AND VERDICT LADDER — FIXED BEFORE ANY DECISION

### 5.1 Selecting `k*`

1. Let `M = max_k (E2 − E0)` over the greedy curve, `k = 1..6`.
2. Let `K_tied = { k : M − (E2 − E0)_k < δ_noise }`.
3. **`k* = min(K_tied)`.**

**The parsimony tie-break is preregistered and binding.** Rationale, stated in advance so it cannot
be argued away later: each CatBoost slot adds **+0.2420 ms/pt** of online inference (measured, C2);
each additional replaced slot is another model to train, ship, version and monitor; and a smaller
`k` spends fewer degrees of freedom. When the metric cannot distinguish two compositions, the
cheaper and simpler one wins.

### 5.2 Verdict ladder on `E2 − E0`

Thresholds are anchored to the **incumbent** (`RT-1257`, `E2 − E0 = +0.002026322`) and to the noise
floor, not to any observed CSA-04 value:

| Verdict | Condition |
|---|---|
| **`SUPERSEDES_RT1257`** | `(E2 − E0)_{k*} − 0.002026322 ≥ δ_noise`, **and** ≥4/5 folds positive vs `RT-1257`, **and** dominant-cell pair net vs `E0` `> 0`, **and** mature-vs-never pair net vs `E0` `> 0` |
| **`NOT_DISTINGUISHABLE`** | `k*`'s `E2 − E0` exceeds `RT-1257`'s but by `< δ_noise`, or a pair-flow condition fails |
| **`INFERIOR`** | `k*`'s `E2 − E0` `<` `RT-1257`'s |

**Only `SUPERSEDES_RT1257` licenses a promotion conversation.** Neither other verdict does, and
neither retracts CSA-04's per-slot science (§7).

**The mature-vs-never condition is retained deliberately.** CAT-411's single-slot mature-vs-never
net is **`−9`**, and CSA-04's `k = 5` hybrid carried only `+33` against `k = 4`'s `+76`. A
composition that buys pooled AUC while going negative on mature-vs-never is a composition to
distrust.

### 5.3 What may not be done

- **The ordering may not be re-derived** after seeing curve results, on any endpoint.
- **The gate ladder in §5.2 may not be moved** after seeing results.
- The 63-subset enumeration **may not select `k*`** (§3.8).
- No blend-weight optimisation, no re-training, no re-admission of CAT-410, no new slot.
- `E1` may not be reintroduced as a selection endpoint at any `k > 1`.

---

## 6. EXPECTED OUTCOME — STATED BEFORE EXECUTION

Recorded so the conclusion cannot be shaped after the fact. From the committed CSA-04 numbers:

**P4 — the re-ordered curve starts at `RT-1257`.** Under `E2 − E0` ordering the first two slots are
CAT-413 and CAT-300, so the `k = 2` point **is** `RT-1257`'s composition and must return
`E2 − E0 = +0.002026322`. **This doubles as a second harness check** — if `k = 2` does not
reproduce that value, HALT.

**P5 — `k*` will be small, and the verdict will be `NOT_DISTINGUISHABLE`.** The largest `E2 − E0`
anywhere in CSA-04's committed curve is `+0.002378` (at its `k = 5`), which exceeds `RT-1257`'s
`+0.002026322` by **`+0.000352`** — roughly **0.32 ×** the `0.0011` noise floor, before any
bootstrap. I expect the re-ordered curve to land in the same region, the tie-break to select a
small `k`, and the verdict to be `NOT_DISTINGUISHABLE`.

**P6 — CSA-04's `k = 5` composition will not survive.** Its lead over `RT-1257` is inside noise and
its mature-vs-never net (`+33`) is the weakest of the higher-`k` points.

**If P5 is wrong** — if some composition beats `RT-1257` by more than `δ_noise` with clean pair
flow — that is a genuine and welcome result, and the `SUPERSEDES_RT1257` path is the one that was
preregistered to catch it.

**What P5 being right would mean, stated plainly now:** CSA-04's real contribution is that **six of
seven specialist slots respond to CatBoost replacement** — a broad, robust, previously-unknown fact
— while **no multi-slot composition beats the two-slot `RT-1257` by a distinguishable margin.**
That is a substantially more interesting scientific result than "MAJOR", and it is not a failure.

---

## 7. WHAT CSA-04 GOT RIGHT — NOT RETRACTED

Recorded explicitly so this re-analysis is not misread as invalidating the experiment.

- **The `RT-1257` regression check passed at `diff = 0.000e+00`.** The harness is verified.
- **All four new slots cleared the per-slot `+0.0010` admission gate on the sound `k = 1` test.**
  CAT-412 `+0.001630`, CAT-414 `+0.001589`, CAT-411 `+0.001567`, CAT-415 `+0.001001`. **This is the
  headline finding and it stands unchanged.**
- **P1 (breadth) and P2 (idiosyncrasy) were both falsified, and reported as falsified.** Both were
  my predictions; neither survived. The agent adjudicated them honestly rather than quietly
  dropping them.
- **P3 (interior maximum) held** on the endpoint it was tested against, and holds on `E2 − E0` too
  (max at an interior `k`, decline at `k = 6`).
- **All four arms trained to completion before any was scored**, per protocol.
- **`RT-1264` keeps its ID** whatever CSA-04R concludes. A voided or superseded experiment keeps
  its identifier (`AGENTS.md`); it is not freed for reuse.

---

## 8. DELIVERABLES

- `research/reports/deep_ensemble_frontier_2026/local/CSA04R_REANALYSIS.md` — full curve on
  `E2 − E0`, bootstrap intervals, `k*`, verdict, adjudication of **P4, P5, P6** including any that
  were wrong.
- `research/reports/deep_ensemble_frontier_2026/local/CSA04R_REANALYSIS.json` — machine-readable,
  including the 63-subset descriptive appendix clearly flagged as non-selecting.
- `research/RESULTS.csv` — a row for `RT-1265` **only if** `k*`'s composition differs from
  `RT-1264`'s. Append only; never edit CSA-04's rows.
- `research/RDOF_LEDGER.md` — a CSA-04R section recording: 1 endpoint change, 1 ordering change,
  0 tuning knobs, 0 models trained, 1 bootstrap seed fixed in advance, and the 63-subset appendix
  logged as descriptive-only.
- `research/reports/deep_ensemble_frontier_2026/local/CSA04_FINAL.md` — **append** a pointer to
  CSA-04R. **Do not rewrite its numbers or its verdicts.**
- `research/STATUS.md` — update **only if** the verdict is `SUPERSEDES_RT1257`. Under
  `NOT_DISTINGUISHABLE` or `INFERIOR`, `RT-1257` remains the best measured result and production
  `RT-600` (external **0.6268**) remains the anchor, so nothing in `STATUS.md` changes.

---

## 9. OUT OF SCOPE

- Any re-training, any hyperparameter or seed change, any blend-weight optimisation.
- Re-admitting CAT-410, or admitting any slot on the new endpoint.
- Reconsidering the GPU lane. `RT-1258` (`+0.0000409`) and `RT-1259` (`−0.003224`) are KILL at
  `k = 1`, where `marginal_vs_clone` is sound. **This defect does not touch them.**
- Reopening C4. `L3` returned `FAIL_NO_RETENTION_MECHANISM` with 0 of 14 rules succeeding; that
  gate is shut on its own evidence and is unaffected by anything here.
- Lockbox, test data, production configuration.

---

## 10. ONE-PARAGRAPH SUMMARY

CSA-04 selected the hybrid size `k` by maximising `E2 − E1`, but `E1` is not a fixed baseline — it
degrades from `−0.002162` to `−0.004033` as `k` goes from 2 to 6, because replacing more
heterogeneous specialists with near-identical seed clones destroys diversity by construction. At
`k = 5` that inflation is large enough that a candidate identical to RT-600 would score `SERIOUS`.
CSA-04R changes exactly two things — the ordering and the selection endpoint, both to `E2 − E0`,
which is fixed across `k` and is the actual deployment counterfactual — holds everything else
including the sound per-slot `k = 1` admission test constant, adds a paired series bootstrap with a
`0.0011` noise floor, and preregisters a parsimony tie-break so that indistinguishable compositions
resolve to the cheaper one. I expect the answer to be that `RT-1257` was already at the peak and
that CSA-04's real contribution is the discovery that six of seven slots respond to CatBoost at
all. That expectation is written here, before execution, so that it cannot be assembled afterwards.
