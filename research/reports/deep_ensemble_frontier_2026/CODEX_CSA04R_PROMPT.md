# AGENT BRIEF — CSA-04R HYBRID CURVE RE-ANALYSIS (LOCAL LANE)

*Paste everything below the line into the local agent as a single prompt.*

---

You previously executed the LOCAL lane of Deep Ensemble Frontier 2026 (CSA-04, the L3 arbitration
probe, and the H1 control package). This is a follow-up task on the same branch and in the same
worktree. It is **a re-analysis of OOF vectors you already produced. You will not train anything.**

## 0. WHERE TO WORK

Return to your existing worktree and branch — **do not create a new one, and do not touch any other
worktree**:

```bash
cd "/path/to/workspace/structural-break-deep-ensemble-frontier-local-2026"
git branch --show-current    # must print research/deep-ensemble-frontier-local-2026
git status --porcelain       # must be clean before you start
```

The preregistration governing this task lives on the shared plan branch. Merge it in:

```bash
git fetch origin
git merge --no-ff origin/research/deep-ensemble-frontier-2026 -m "Merge CSA-04R preregistration"
```

Then **read `research/reports/deep_ensemble_frontier_2026/CSA04R_REANALYSIS_PREREG.md` in full.
It is the authority. This brief summarises it; where they differ, the prereg wins.**

Standing rules from your original brief still apply: never force-push, never rebase pushed history,
work only in your own worktree, ledgers are append-only, you remain the single writer of
`research/STATUS.md`, and do not commit `.npy` arrays.

## 1. WHY THIS TASK EXISTS

CSA-04 selected `k*` by maximising `marginal_vs_clone = E2 − E1`. **`E1` is not a fixed baseline.**
It replaces the same `k` slots with near-identical seed clones, so it degrades monotonically as `k`
grows — this is visible in your own committed `CSA04_RESULTS.json`:

| k | `E1 − E0` | `E2 − E1` (what CSA-04 maximised) | `E2 − E0` |
|---:|---:|---:|---:|
| 2 | −0.002162 | +0.003602 | +0.001440 |
| 3 | −0.003156 | +0.004828 | +0.001672 |
| 4 | −0.003384 | +0.005624 | +0.002240 |
| 5 | −0.003556 | **+0.005935** | +0.002378 |
| 6 | −0.004033 | +0.005897 | +0.001864 |

**At `k = 5`, a candidate identical to RT-600 — `E2 = E0`, adding nothing whatsoever — would score
`marginal_vs_clone = +0.003556` and be labelled `SERIOUS`.** The frozen thresholds were calibrated
where `E1 ≈ E0`, i.e. at `k = 1` and the single `k = 2` hybrid. They do not translate to `k = 5`.

The substantive damage is to **selection**, not just labelling. Ordering the six admitted survivors
by the two endpoints gives near-opposite orderings — CAT-411 moves from 3rd to last, CAT-413 from
4th to 1st — and the reversal already shows in your numbers: CSA-04's `k = 2` has a *higher*
marginal than `RT-1257` (+0.003602 vs +0.002407) but a *lower* `E2 − E0` (+0.001440 vs +0.002026).

**This defect is in the plan's specification, not in your execution.** `LANE_LOCAL.md` §L2.5
specified both the `E1_k` control and the frozen ladder without noting that the thresholds stop
meaning the same thing as `k` grows. You followed it exactly, froze the ordering in advance as
instructed, and passed the mandatory `RT-1257` regression check at `diff = 0.000e+00`. Holding the
ladder fixed after seeing results was correct behaviour.

## 2. WHAT CHANGES — EXACTLY TWO THINGS

| Element | Changed? |
|---|---|
| Per-slot admission test (`marginal_vs_clone ≥ +0.0010` at `k = 1`) | **NO — keep it** |
| Survivor set: CAT-412, CAT-414, CAT-411, CAT-413, CAT-300, CAT-415 | **NO — identical** |
| CAT-410 excluded at `+0.000753` | **NO — stays excluded** |
| Folds, calibration, pair-flow convention, OOF vectors | **NO — identical** |
| **Slot ordering for the greedy curve** | **YES → descending single-slot `E2 − E0`** |
| **Endpoint selecting `k*`** | **YES → `E2 − E0`** |

**Do not re-admit CAT-410** even though its `E2 − E0` (+0.000930) would rank it 4th. Re-admitting a
slot on the new endpoint is exactly the post-hoc rule-shopping this exercise exists to prevent.

**Why the `k = 1` test is kept:** at `k = 1` the clone control degrades only mildly and `E2 − E1`
soundly answers a real question — *does CatBoost carry ranking information an exchangeable LightGBM
clone does not, in this slot?* That question is answered (six of seven slots, yes) and is not
revisited. **Why `E2 − E0` selects `k`:** `k*` is a deployment question, the counterfactual to
replacing `k` slots is replacing none (`E0`), and `E0` is fixed across `k` so the curve is
comparable along its length.

## 3. THE FROZEN SLOT ORDERING

Use this ordering. **It is frozen by the prereg and may not be re-derived after you see results.**

| rank | slot | single-slot `E2 − E0` |
|---:|---|---:|
| 1 | CAT-413 (`RT-1254`) | +0.001244347 |
| 2 | CAT-300 (`RT-1255`) | +0.001126876 |
| 3 | CAT-412 (`RT-1261`) | +0.001029593 |
| 4 | CAT-415 (`RT-1263`) | +0.000907324 |
| 5 | CAT-414 (`RT-1262`) | +0.000552119 |
| 6 | CAT-411 (`RT-1260`) | +0.000362939 |

## 4. PROCEDURE

**Step 1 — reproduce CSA-04 exactly, as a harness check.** Recompute `marginal_vs_clone` and
`E2 − E0` for all seven single slots and all five committed hybrid points. **Every value must match
`CSA04_RESULTS.json` to at least 1e-9. If any does not, HALT and report** — nothing downstream is
trustworthy.

**Step 2 — re-verify the `RT-1257` regression check** at `+0.002407204670070`. Must remain exact.

**Step 3 — place `RT-1257` on the curve as the incumbent reference.** Its
`E2 − E0 = +0.002026322` is the number every CSA-04R composition must beat.

**Step 4 — build the greedy nested curve** at `k = 1..6` under the §3 ordering. Per `k` report:
`E0`, `E1`, `E2` (report `E1` for continuity — it is **not** the endpoint); **`E2 − E0`** as
primary; per-fold `E2 − E0` and folds positive; `E2 − E0` **relative to `RT-1257`** per fold and
pooled with folds positive; and pair flow vs `E0` on all four splits (whole, dominant-cell,
mature-vs-never, mature-vs-prebreak) as repairs / damage / net.

**Step 5 — paired bootstrap.** Resample **series**, not rows (folds are series-level and rows
within a series are dependent). `B = 2000`, **seed `20260828`**. Report bootstrap SE and 95%
percentile intervals for: each `k`'s `E2 − E0`; the `k*` candidate vs `RT-1257`; and adjacent curve
points around the maximum.

**Step 6 — apply the selection rule.** With
`δ_noise = max(paired bootstrap SE of the contrast, 0.0011)`:

1. `M = max_k (E2 − E0)` over `k = 1..6`.
2. `K_tied = { k : M − (E2 − E0)_k < δ_noise }`.
3. **`k* = min(K_tied)`.**

The parsimony tie-break is binding and preregistered: when the metric cannot distinguish two
compositions, the cheaper one wins — each CatBoost slot costs a measured **+0.2420 ms/pt** of
online inference (C2, crunch lane).

**Step 7 — verdict**, on `E2 − E0`, anchored to `RT-1257`'s `+0.002026322`:

| Verdict | Condition |
|---|---|
| `SUPERSEDES_RT1257` | `(E2−E0)_{k*} − 0.002026322 ≥ δ_noise`, **and** ≥4/5 folds positive vs `RT-1257`, **and** dominant-cell pair net vs `E0` > 0, **and** mature-vs-never pair net vs `E0` > 0 |
| `NOT_DISTINGUISHABLE` | exceeds `RT-1257` but by `< δ_noise`, or a pair-flow condition fails |
| `INFERIOR` | below `RT-1257` |

The mature-vs-never condition is deliberate: CAT-411's single-slot mature-vs-never net is **−9**,
and CSA-04's `k = 5` carried only `+33` against `k = 4`'s `+76`.

**Step 8 — descriptive appendix.** Enumerate all 63 non-empty subsets of the six admitted slots and
report their `E2 − E0`. **This is descriptive only and is BARRED from selecting `k*` or supporting
any promotion claim.** If its maximum exceeds the greedy `k*`, **report the gap and do not act on
it** — treat it as a hypothesis for a future, independently-designed experiment. 63 comparisons on
5 folds would overfit.

## 5. ADJUDICATE THESE THREE PREDICTIONS

They are recorded in the prereg §6 **before** execution. Report each explicitly, **including any
that were wrong.**

- **P4 — the re-ordered curve starts at `RT-1257`.** Under the §3 ordering the first two slots are
  CAT-413 + CAT-300, so `k = 2` **is** `RT-1257`'s composition and must return
  `E2 − E0 = +0.002026322`. **This is a second harness check — if it does not reproduce, HALT.**
- **P5 — `k*` will be small and the verdict will be `NOT_DISTINGUISHABLE`.** The largest `E2 − E0`
  in CSA-04's committed curve exceeds `RT-1257` by only `+0.000352`, about 0.32× the 0.0011 noise
  floor.
- **P6 — CSA-04's `k = 5` composition will not survive.** Its lead is inside noise and its
  mature-vs-never net (+33) is the weakest of the higher-`k` points.

If P5 is wrong — if some composition beats `RT-1257` by more than `δ_noise` with clean pair flow —
that is a genuine and welcome result and `SUPERSEDES_RT1257` is the path preregistered to catch it.

## 6. WHAT YOU MAY NOT DO

- **Do not re-derive the ordering** after seeing curve results, on any endpoint.
- **Do not move the verdict ladder** after seeing results.
- **Do not let the 63-subset appendix select `k*`.**
- **Do not reintroduce `E1` as a selection endpoint at any `k > 1`.**
- **Do not retrain, retune, re-seed, or optimise blend weights.** No model is fit in this task.
- **Do not re-admit CAT-410 or admit any new slot.**
- **Do not edit or delete any CSA-04 output.** `CSA04_FINAL.md`, `CSA04_RESULTS.json`,
  `CSA04_RESULTS.csv` and their `RESULTS.csv` rows stand as written. `RT-1264` keeps its ID
  whatever CSA-04R concludes — a superseded experiment keeps its identifier.
- **Do not touch lockbox or test data**, or modify production.
- **Do not reconsider `RT-1258`/`RT-1259`** — KILL at `k = 1`, where the metric is sound; this
  defect does not touch them. **Do not reopen C4** — `L3` returned `FAIL_NO_RETENTION_MECHANISM`
  with 0 of 14 rules succeeding, and that gate is shut on its own evidence.

## 7. DELIVERABLES

- `research/reports/deep_ensemble_frontier_2026/local/CSA04R_REANALYSIS.md` — full curve on
  `E2 − E0`, bootstrap intervals, `k*`, verdict, and adjudication of P4/P5/P6.
- `research/reports/deep_ensemble_frontier_2026/local/CSA04R_REANALYSIS.json` — machine-readable,
  with the 63-subset appendix clearly flagged as non-selecting.
- `research/RESULTS.csv` — a row for **`RT-1265`** *only if* `k*`'s composition differs from
  `RT-1264`'s. If the rule re-selects the same composition, **no ID is consumed.** Append only.
- `research/RDOF_LEDGER.md` — a CSA-04R section: 1 endpoint change, 1 ordering change, 0 tuning
  knobs, 0 models trained, bootstrap seed fixed in advance, 63-subset appendix logged as
  descriptive-only.
- `research/reports/deep_ensemble_frontier_2026/local/CSA04_FINAL.md` — **append** a pointer to
  CSA-04R. **Do not rewrite its numbers or verdicts.**
- `research/STATUS.md` — update **only if** the verdict is `SUPERSEDES_RT1257`. Otherwise `RT-1257`
  remains the best measured result, production `RT-600` (external 0.6268) remains the anchor, and
  nothing in `STATUS.md` changes.

Commit and push to your own branch. Expected runtime: minutes.

## 8. REPORT BACK

State the curve, `k*`, `δ_noise`, the verdict, and which of P4/P5/P6 were wrong. If the Step 1 or
Step 2 harness checks fail, report that immediately and stop — it invalidates everything
downstream, including CSA-04's committed numbers.

If the verdict is `NOT_DISTINGUISHABLE`, say so plainly and do not soften it. That outcome means
CSA-04's real contribution is that **six of seven specialist slots respond to CatBoost replacement**
— a broad, robust, previously-unknown fact — while no multi-slot composition beats the two-slot
`RT-1257` by a distinguishable margin. That is a more interesting result than "MAJOR", and it is
not a failure.
