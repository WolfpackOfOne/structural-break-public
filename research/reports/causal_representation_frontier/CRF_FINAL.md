# CAUSAL REPRESENTATION FRONTIER — EXECUTION FINAL

## STATUS: **`CRF_PROGRAM_EXHAUSTED`**

Both primaries are KILL. `CRF-03` did not open. **Model search under this program is
stopped.** No `CRF-04` is invented — `CRF_PROGRAM_PREREG.md` §0.8/§0.9 forbid it and
the evidence does not support it.

---

## A. GIT

| | |
|---|---|
| branch | `research/causal-representation-frontier-2026` |
| starting SHA | `85d121f` — *Preregister causal representation frontier program* |
| ending SHA | this commit |
| working tree | clean |
| origin | synced, every commit pushed, **no force push, no rebase of pushed history** |

Commits, in order — the preregistration chronology is intact and unsquashed:

| SHA | commit |
|---|---|
| `afba958` | Preregister CRF-01 NNCSR execution |
| `ed05902` | Validate CRF-01 purity and causality preflight |
| `b6f27e3` | Evaluate CRF-01 null-normalized sequence ranker |
| `9a3d3c7` | Preregister CRF-02 ACGN execution |
| `ad6ecd7` | Validate CRF-02 purity and causality preflight |
| `89149d5` | Correct CRF-02 isolation sentinel **before any score** |
| `9a5ecc0` | Evaluate CRF-02 amortized conditional null — **later voided** |
| `fb24c39` | Void `RT-1237/1238/1239` and reserve the corrected CRF-02 run |
| `9c5a352` | Evaluate CRF-02 amortized conditional null (corrected run) |

No training implementation predates its execution preregistration. No score predates
its preflight commit. Nothing was rewritten.

---

## B. CRF-01 · NNCSR — NULL-NORMALIZED CAUSAL SEQUENCE RANKER

**KILL — abandoned at the preregistered cheap abandon gate.**

| | |
|---|---|
| RT IDs | `RT-1234` candidate, `RT-1235` C1 (BCE); `RT-1236` C2 **reserved, not run, not recycled** |
| preflight | 24 gates green. Prefix invariance `atol = 0.0` **bitwise**; truncation `2.220e-16`; batch composition `1.665e-16`; fold purity proven with a positive control that reproduces the Wave-7 contaminated scheme; purity assertion proven reachable; zero finite lockbox rows |
| standalone whole-fold TS-AUC | **0.592762** |
| standalone dominant-cell TS-AUC | 0.621422 |
| within-`t` ρ vs RT600 | **+0.4460** |
| `E0` / `E1` / `E2` | 0.638276 / 0.638586 / — |
| `marginal_vs_clone` | **not computed** — gate fired, folds 1–4 deliberately never trained |
| C1 marginal | not computed, same reason |
| candidate − C1, on standalone | **+0.022219** whole, **+0.037100** dominant |
| C2 | not run (gate-blocked) |
| whole pair net | **−2,862** |
| dominant pair net | **−2,798** |
| mature-vs-never net | **−2,962** |
| mature-vs-pre-break net | **−2,820** |
| pre-break damage rate on RT600-correct | **0.2626** vs a 0.0150 cap |
| unique repair coverage / Jaccard vs `RT-401` | 0.3088 / 0.1740 |
| runtime | 906.6 s + 884.2 s = **0.497 h** |
| **verdict** | **KILL** |

---

## C. CRF-02 · ACGN — AMORTIZED CONDITIONAL GENERATIVE NULL

**KILL — abandon gate fired and the learned-null isolation gate failed.**

| | |
|---|---|
| RT IDs | `RT-1240` candidate, `RT-1241` C1 (fixed null), `RT-1242` C2 (deranged `h_i`); `RT-1237/1238/1239` **void and retired** |
| preflight | 27 gates green, including the strict-past shift verified **bitwise** (the null never sees the value it is pricing), the label gradient proven never to reach the null, the forbidden global-pretraining scheme reproduced and shown to contaminate 5/5 outer folds, and a checkpoint provenance guard |
| standalone whole-fold TS-AUC | **0.559140** |
| standalone dominant-cell TS-AUC | 0.579573 |
| within-`t` ρ vs RT600 | **+0.2887** |
| `marginal_vs_clone` | **not computed** — gate fired |
| C1 fixed null | **0.580586** / 0.599909 — **beats the candidate** |
| candidate − C1 | **−0.021446** whole, −0.020337 dominant (needs ≥ +0.0005) — **FAIL** |
| C2 deranged `h_i` | 0.555763 / 0.571182 |
| candidate − C2 | **+0.003377** whole, **+0.008391** dominant — **PASS** |
| whole pair net | **−5,633** |
| dominant pair net | **−5,273** |
| mature-vs-never net | **−5,176** |
| mature-vs-pre-break net | **−6,734** |
| pre-break damage rate on RT600-correct | **0.3929** vs a 0.0150 cap |
| runtime | 1,364.9 s pretrain + signals + 4 heads = **0.411 h** |
| **verdict** | **KILL** |

**The passing derangement control is the load-bearing detail.** The 8-dimensional
history bottleneck carries real series-specific information — permuting it costs
0.0034 whole-fold and 0.0084 on the dominant cell — so the model is *not* memorising
a series identifier and the `m05_ctx` rule was not triggered. **The learned null
conditions correctly and loses anyway.** The failure is **amortization**: the fixed
null holds five AR coefficients plus a 256-knot empirical residual distribution *per
series*, paid for by that series' own break-free history at zero generalisation
cost, while the learned null compresses all of it into 8 floats shared across a
population whose heterogeneity is the reason per-series historical calibration is
this project's foundation.

### The void run

`RT-1237/1238/1239` (`9a5ecc0`) silently loaded a 24-series, 1-epoch, `HWIN = 128`
null written by a **unit test** instead of the preregistered one. Caught by compute
accounting for this report — `pretrain_runtime_s = 0.2` against a real 1,364.9 s.
The state-sha check that existed proved *integrity, not provenance*.

It mattered scientifically, not merely procedurally: with the toy null the
derangement control **tied** (`−0.000981`), which reads as "the bottleneck carries
nothing". Same verdict, **wrong mechanism**. Fixed with a checkpoint provenance
fingerprint (fold, seed, epochs, `HWIN`, batch, widths, level count, lr, wd,
fit-series count, and the sha256 of the sorted fit-series ids) that **stops the run**
on mismatch, plus test-isolated caches, plus a regression test that reproduces the
defect and requires the refusal. CRF-01 was verified unaffected: it writes no
checkpoint, its tests never call `emit` or `build_channels`, and its metadata records
the full 6,383-series, 20-epoch runs.

---

## D. CRF-03 · NNCSR-G — **DID NOT OPEN**

`CRF_PROGRAM_PREREG.md` §3.1 opens CRF-03 **if and only if** CRF-01 or CRF-02
reaches `marginal_vs_clone ≥ +0.0015` on fold 0 **and** passes its own mandatory
isolation control.

Both primaries are KILL and **neither produced a `marginal_vs_clone` at all**. The
condition is not met, CRF-03 is not run, and no id is allocated for it. This is the
same rule that kept `RT-980` closed in Wave 6: a failed primary followed by an
architecture combination is fishing.

---

## E. REPRESENTATION × OBJECTIVE FACTORIAL — NOW EMPIRICALLY FILLED

| | **rowwise BCE** | **same-`t` ranking** |
|---|---|---|
| **static 500-column bank** | `RT-300` incumbent, 0.61605 | `RT-111`, `RT-700/701/702`, `RT-123` — dead heat |
| **learned causal sequence** | `RT-970` 0.52618, **`RT-1235` 0.57054** | **`RT-1234` 0.59276** |

Because `RT-970` is the same shell on the same fold partition, the ladder isolates
one factor at a time for the first time in this project:

* **representation effect, objective held fixed: `+0.0444`**
  (`RT-970 → RT-1235`: null-normalised channels, `elapsed` removed)
* **objective effect, representation held fixed: `+0.0222`** whole-fold,
  **`+0.0371`** dominant-cell (`RT-1235 → RT-1234`: BCE → same-`t` pairwise)
* **total vs `RT-970`: `+0.0666`**

Wave 6's own report said `RT-970` could not distinguish "family wrong" from
"objective wrong". **Both were partly wrong. Both are now fixed and measured. Their
sum is still not enough.**

**The null cell is also filled.** CRF-02 adds the generative-null axis: a learned,
correctly-conditioned amortized null loses to the fixed per-series null by `0.0214`.

---

## F. INFORMATION FRONTIER CONCLUSION

### **PRACTICAL LEGAL-PREFIX CEILING NOW STRONGLY SUPPORTED**

Three hypotheses close:

* **H-A, representation saturation** — closed. The best low-redundancy
  representation this project has ever produced still fails.
* **H-B, objective mismatch** — closed. The effect is real (`+0.0222`) and an order
  of magnitude too small.
* **H-E, learned-null misspecification** — closed. A more flexible,
  demonstrably-conditioning null prices "normal" *worse* than the fixed per-series
  calibration already shipped.

**H-D** — the practical limit is the legal prefix itself, and the residual W7-D3R gap
is predominantly **post-`t`** information — is what remains.

### Two independent lines of evidence, stated at their real strength

**1. The frontier moved and the answer did not change.** The program audit's binding
fact was `corr(standalone, ρ) = +0.983` across 17 scored arms — nothing this project
has built has ever been both good and different — with the best standalone at
`ρ ≤ 0.60` being `RT-1201`'s **0.58358**, worth `+0.000301`. `RT-1234` reached
**0.59276 at ρ 0.446**, a new best point on that frontier by `+0.0092`, and still
failed a necessary condition deliberately set *below* the fitted `+0.0030` contour.

**2. Pair flow — and this is the stronger argument, because it does not rest on an
extrapolation.** `FIRST_SWEEP_SYNTHESIS.md` measured dominant-cell pair net as the
strongest non-tautological correlate of `marginal_vs_clone` (Pearson **0.879**).
Every CRF arm is **negative in every pair-flow cell**: CRF-01 dominant net `−2,798`,
CRF-02 `−5,273`, and both are negative on mature-vs-never and mature-vs-pre-break
too. Pre-break damage rates on RT600-correct pairs are `0.2626` and `0.3929` against
a `0.0150` cap — one and two orders of magnitude over. A candidate that damages a
quarter to two-fifths of what RT-600 already gets right does not have a small
positive marginal hiding behind a threshold.

### Stated honestly: what was *not* measured

**No CRF arm ever produced a `marginal_vs_clone`.** The abandon gate fired first in
both cases, exactly as designed, and folds 1–4 were deliberately never trained. So
the conclusion rests on a **necessary-condition filter plus pair flow**, not on a
direct ensemble measurement.

The fair version of the counter-argument, and its answer: the `0.600` bar is a
descriptive OLS over `n = 17` arms whose own recorded caveat is "R² moderate, ρ
baselines heterogeneous across waves". Taking that fit at face value, the `+0.0030`
contour at `ρ ≈ 0.45` sits near standalone `≈ 0.635`; `RT-1234` is `0.043` below it.
So even under the most generous reading of the extrapolation, CRF-01 was never near
SERIOUS — it might have been weakly positive, and its uniformly negative pair flow
argues it would not have been. That is a bounded uncertainty on the *size* of a
missed effect, not on its sign or its materiality.

### Diagnosis: the bottleneck is arbitration, not detection

CRF-01 repairs **39.5 %** of RT-600's sampled dominant-cell mistakes, with a repair
Jaccard against the seed clone of only **0.174** — the repairs really are its own.
It cannot keep them: it damages **26.8 %** of what RT-600 already had right. This is
`FIRST_SWEEP_SYNTHESIS.md` H1 recurring in a genuinely new representation. The
project now has a much better detector than the first sweep produced and it
arbitrates no better; and `SS-01`, whose entire purpose was arbitration, was itself
KILL.

---

## G. BEST MEASURED CRF RESULT

**`RT-1234`, CRF-01 candidate: fold-0 standalone whole-fold TS-AUC `0.592762` at
within-`t` ρ `+0.4460`; dominant-cell `0.621422`.**

A **new best standalone-at-low-redundancy point** for this project, beating
`RT-1201`'s `0.58358` by `+0.0092`. No CRF arm produced a `marginal_vs_clone`.

Production `RT-600` is unchanged at an external **0.6268**.

---

## H. NEXT ACTION

### **`CRF_PROGRAM_EXHAUSTED` — STOP MODEL SEARCH.**

The remaining budget belongs to **deployment robustness**, not to model research.

Explicitly not authorised by this program and not to be started as a continuation of
it: `CRF-04`, Transformer, GRU, SSM, longer receptive field, RF-255, wider hidden
layer, more channels, an alternative PIT, a different `q90`, another null head, or
different pair weighting.

**One hypothesis is recorded, not run.** CRF-02's passing derangement control shows
the conditioning works and the *width* is what fails — 8 floats against what a
per-series fit gets for free. A materially wider bottleneck is therefore the one
thing this program's evidence genuinely motivates. It is a **different experiment**
requiring its own program-level preregistration and its own justification against the
compute budget, and it should be weighed against the fact that its ceiling is the
fixed null it is trying to beat, which is already shipped and already free.

---

## I. GITHUB ACTIONS

| commit | run | job | conclusion |
|---|---|---|---|
| `b6f27e3` Evaluate CRF-01 | `32972488395` | `check-results-csv` | **success** |
| `9a5ecc0` Evaluate CRF-02 (later voided) | `32980033144` | `check-results-csv` | **success** |
| `9c5a352` Evaluate CRF-02 (corrected) | `32984351317` | `check-results-csv` | **queued — not concluded at the time of writing** |

Each run was matched to its commit by `headSha` before being reported; no run was
claimed on title alone.

**Run `32984351317` is reported as queued, not as passing.** It was created at
`2026-08-26T15:11:58Z` for `headSha 9c5a352`, matched to this branch's HEAD, and had
not been assigned a runner after ~30 minutes of polling. No other run in the
repository was competing, so this is GitHub-side runner availability rather than
anything about the commit. It is **not** claimed as a success, because it has not
concluded.

What *is* verified: the identical check the workflow runs,
`python research/scripts/check_research_hygiene.py`, was executed locally against the
committed `research/RESULTS.csv` and reports
`OK: RESULTS.csv has 253 experiment rows, no duplicate IDs`. The two earlier result
commits were each verified through a concluded run matched by `headSha`. Commits that do not touch `research/RESULTS.csv`
(`afba958`, `ed05902`, `9a3d3c7`, `ad6ecd7`, `89149d5`, `fb24c39`) correctly trigger
no run — the workflow is path-filtered on `research/RESULTS.csv`.

`research/RESULTS.csv`: **245 → 253** experiment rows. Append-only throughout; no row
was edited or deleted, including the three void rows. `check_research_hygiene.py`
reports no duplicate IDs after every filing.

---

## J. SAFETY

Confirmed for the whole program:

* **no lockbox read** — fold `−1` never loaded; every emitted OOF vector asserted to
  have zero finite lockbox rows, at emission and again at load
* **no test data** — no `X_test.reduced`, no `y_test.reduced`, no
  `folds_final10k.parquet`
* **no production change** — `production/rt600` untouched, no promotion, no artifact
  rebuilt
* **no submission**
* **no force push, no rebase of pushed history, no squashed chronology**
* **no recycled RT ID** — `RT-1236` reserved and unconsumed;
  `RT-1237`/`RT-1238`/`RT-1239` void and retired; `RT-900` still retired
* **no post-score tuning** — no gate threshold, architecture, channel set, objective,
  optimiser, seed, calibration or ensemble rule was changed after any number was seen
* **no merge** into `research/current` or `research/new-avenues-pilots-2026`
* **no repair of the 15 pinned known failures**

Total compute: **0.93 h** of the 15 h screening budget — CRF-01 `0.497 h`, CRF-02
corrected `0.411 h`, plus the `0.36 h` void run, counted against the budget because
compute spent is compute spent.

---

## K. WHAT THE PROGRAM ESTABLISHED

`CRF_PROGRAM_PREREG.md` §5 wrote both outcomes in advance. This is the one that
happened:

> **If CRF-01 fails:** the representation × objective factorial is complete and
> empty… the project will have strong, structured evidence that **0.6268 is close to
> what the legal causal prefix supports**, and the remaining budget belongs to
> deployment reliability rather than to model search.

Two experiments, six scored arms, two mandatory isolation controls, one declared
no-id diagnostic, one void run caught and corrected, 51 pre-score gates across the
two batteries, and 0.93 h of compute. Both hypotheses were **worth testing and both
are now closed with reasons rather than assumptions** — including one, the objective
effect, that turned out to be real, positive, cleanly isolated for the first time,
and still an order of magnitude too small to matter.

The program's own instruction for this outcome, written before any number existed:
that failure **is evidence**, and the correct response is not architecture number 12.
