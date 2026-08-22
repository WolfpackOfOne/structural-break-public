# WAVE 6 — LIVE STATUS

**Branch `research/wave6-alpha`, parent `research/wave5-alpha` @ `26b01f6`.**
Pre-registration: `research/WAVE6_PREREG.md`, committed before any number.

---

## W6-E1 — CALIBRATION ANCHOR PLACEMENT: **INSIDE NOISE. HYPOTHESIS FALSIFIED.**

**The parity check passed first**, which is what makes the rest readable:
`CAL-LOG` — the incumbent anchors run through the new explicit-anchor class —
reproduces the shipped `SCDF_NSEEN` to **all five decimals on every fold**
(0.62581, 0.63828 / 0.62040 / 0.63393 / 0.61751 / 0.61894). So the
generalisation is faithful and every delta below is pure anchor **placement**.

| scheme | TS-AUC | Δ vs `CAL-LOG` | folds | anchors (fold 0) |
|---|---|---|---|---|
| `CAL-LOG` incumbent | **0.62581** | — | — | 1, 2, 4, 7, 12, 23, 43, 81, 152, 285, 533, 999 |
| `CAL-WT` weight quantiles | 0.62579 | **−0.00002** | 2/5 | 53, 105, 149, 190, 230, 272, 317, 366, 421, 487, 570, 704 |
| `CAL-HYB` hybrid | 0.62581 | **−0.00001** | 2/5 | 1, 3, 8, 22, 60, 168, 200, 262, 329, 407, 506, 661 |
| `NULL-0` | 0.62581 | +0.00000 | | 1, 6, 43, 66, 81, 154, 275, 280, 547, 638 |
| `NULL-1` | 0.62575 | −0.00006 | | 1, 3, 9, 17, 19, 34, 41, 45, 182, 304, 701, 710 |
| `NULL-7` | 0.62577 | −0.00004 | | 1, 5, 7, 8, 25, 75, 212, 246, 291, 417, 491 |
| `NULL-42` | 0.62580 | −0.00001 | | 2, 13, 21, 22, 124, 192, 210, 228, 376, 602, 844 |
| **NULL-ANCHOR mean — the control** | 0.62578 | −0.00003 | | |

**Best scheme: −0.00001. Screening threshold was +0.0010. Bar was +0.0030.**

**Every placement scores the same.** Anchors packed into `[53, 704]` where the
weight is, anchors spread log-uniformly at random, and the incumbent's
front-loaded set all land within **0.00006** of each other. Zero grid fallbacks
in every scheme, so this is not a starvation artefact — the calibration is
simply insensitive to where its knots sit.

### The measurement was right and the inference was wrong

Handoff Part 13 is descriptively correct: nine of the twelve shipped anchors do
sit at t ≤ 168, covering 25% of the pair weight, and the first seven do span
1.05% of it. That is a real mismatch between where the calibration has
resolution and where the metric has weight.

**It does not bind.** `SmoothTimeCDFCal` interpolates smoothly in `log(t+1)`
between anchors, and the score→percentile map evidently varies slowly enough
with `t` that twelve knots *anywhere* capture it. Resolution was never the
constraint; W4-E1's +0.00267 for the calibration family is bought by having a
time-conditional map **at all**, not by where it is sampled.

This is a good reminder that a measured mismatch is a hypothesis, not a finding.
It cost under ten minutes and no training to close, which is exactly why it was
scheduled first.

**Counts toward the §0 stopping rule.**

Evidence: `research/reports/wave6_e1_anchors.json`.

---

## W6-E2 — **VOID. THE EXPERIMENT WAS ILL-POSED AND I SHOULD HAVE CAUGHT IT.**

`RT-900` scored **0.86552** against `RT-300`'s 0.61605 — an aggregate delta of
**+0.24946**, positive on 5/5 folds, with by-age deltas from +0.40 (age 0–5) to
+0.19 (age 100+). Under §4.3's rule that reads "LOCALISATION is the lever".

**It is not a finding. It is a leak, and the number should never be quoted.**

### The diagnosis

The oracle block is `NaN` for every row with `t < cut`, because there is no
post-cut segment to compute statistics over. Test the mask on its own:

| | |
|---|---|
| TS-AUC of the single indicator `1[t >= cut]`, nothing else | **0.81442** |
| share of rows where the block is `NaN` | 49.1% |
| **of those, share that are NEGATIVES** | **100.00%** |

For a break series `cut = tau`, so `1[t >= cut]` **is** the online target
`y[t] = 1[t >= tau]`. For a no-break series `cut` is a placebo and `y = 0`
throughout. So `NaN ⟹ y = 0` with certainty, and LightGBM splits on missingness
natively. A bare indicator beats the champion by **+0.198** without looking at
the data at all.

### Why patching it does not work

The placebo cut was supposed to be the protection, and it is the right idea — it
is what the wave-1 taxonomy and the oracle-frontier study both use. It fails
here for a reason specific to this metric:

* the **series-level** question is "does this series contain a break?", and
  knowing the boundary does not answer it — which is why the oracle frontier's
  design is sound;
* the **real-time row-level** question is "has the break happened *by now*?",
  and **knowing τ answers it exactly**.

Dropping `or_elapsed` and `or_frac` and back-filling the segment with the full
prefix removes the `NaN` mask, but the same information returns through
`or_frac = 1.0` exactly when `t < cut`, and through the segment length that the
null calibration is matched on. **Under TS-AUC, "give the model τ" is
degenerate: τ is the label.** There is no non-degenerate patch, so the
experiment is voided rather than repaired.

### What this cost, and what it did not

15 minutes of training and one feature build. `RT-900` stays in the ledger
flagged **VOID — LABEL LEAK VIA THE MISSINGNESS MASK**, because deleting a
result is worse than recording why it was wrong. No production path touched:
`w6oracle` was never a registered module and cannot reach a manifest.

**The §4.3 branch is NOT decided.** Wave 6 does not get to claim "localisation
is the lever" on the back of a leak, and the §0 stopping rule has one arm
resolved (`W6-E1`, inside noise) and one arm **unresolved**.

### The corrected question, for whoever runs it next

The decidable version is the oracle frontier's own, sharpened:

> At the **FULL** horizon, where the frontier measured its only real headroom
> (+0.0396 over `RT-300`), does **our 500-column causal bank** — given the true
> boundary, at the frontier's series-level protocol, one row per series — beat
> the frontier's **generic 150-tree bank** at the same 0.6497?

* if **our features match ~0.6497**, the frontier's oracle was already
  representation-saturated and the gap to `RT-300` is about τ-knowledge;
* if **our features beat it materially**, representation is the live lever and
  the neural track is the best-motivated thing in wave 6.

It is series ROC AUC, not TS-AUC, which is precisely what makes it non-degenerate
— and it is the metric the frontier already reports, so the comparison is direct.

---

## W6-E2R — **CASE C. REPRESENTATION IS STILL A LIVE LEVER.**

Pre-registered in `research/WAVE6_PREREG.md` §18 at `768204e`, **before this run
reported**. Full report: `research/reports/wave6_corrected_oracle.md`.

### The reproduction gate passed first, which is what makes the rest readable

| | series ROC AUC |
|---|---:|
| prior oracle-frontier study, `lgbm_rich`, FULL | 0.6496854 |
| this run, same protocol, unfiltered 8,000 | **0.6496899** |
| delta | **+0.0000045** (gate was ±0.010) |

All four input hashes match the prior manifest byte for byte. Every sentinel is
clean: boundary-metadata 0.51374 (the frontier's own was 0.5379), missingness
0.51348 / 0.51650, support-length 0.52624, permuted labels **0.50152**.

### The result

| arm | mean series AUC | |
|---|---:|---|
| `A_rich` frontier generic bank + true boundary | 0.65037 | the prior study's arm |
| **`B_causal` our 500 causal columns + the same boundary** | **0.67391** | **Δ = +0.02354** |
| `C_nobound` our 500 columns, **no boundary** | 0.64529 | |
| `AB` union | 0.67848 | |
| `RT-300` shipped row-level stream at the final row | 0.61939 | reference |

**5 of 5 pseudo-τ seeds positive. 25 of 25 folds positive. Every bootstrap CI
excludes zero**, worst lower bound +0.00567. The pre-registered Case C threshold
was +0.010.

Two controls that matter more than the headline:

* **Matched column width.** Arm B has 498 columns to arm A's 280. Cut at random
  to exactly 280, three draws: Δ becomes **+0.0308** (seed 0) and **+0.0205**
  (seed 2026). *Larger*, not smaller. And `C_nobound` — 498 of our columns with
  no boundary — scores *below* 280 of theirs with one. Width does not buy it.
* **Placebo boundary.** With a random cut for both classes, arm A falls to
  0.59470 and arm B to 0.62312: the gap is **+0.0284**, slightly wider than with
  the true boundary. Arm B is not exploiting the boundary harder; it is a better
  representation either way.

### What the boundary is worth, and the part of the frontier's headroom that was never real

| | series AUC | step |
|---|---:|---:|
| `RT-300` row-level stream, final row | 0.61939 | — |
| our same 500 columns refit for the **series** question, no boundary | 0.64529 | **+0.02589** |
| … plus the true boundary | 0.67391 | **+0.02863** |
| … plus the frontier's bank on top | 0.67848 | +0.00457 |

**+0.0259 of the prior study's +0.0396 "model-extraction gap" is a
question-mismatch artefact**, not headroom: it is the difference between reading
a row-level model's last prediction and fitting a series-level classifier on the
same columns. It is **not available to the real-time task**. τ-knowledge is
worth about +0.029 on top of that, and the two banks retain +0.005 of mutual
complementarity.

### The §4.3 branch, finally decided — and decided the other way

The voided RT-900 would have said "localisation is the lever". The corrected
experiment says **both levers are real and comparable in size at FULL** — τ is
worth +0.029, representation +0.024 — but only one of them is legal. We cannot
be given τ. We *can* build a better representation, and the measurement says the
generic bank that Wave 6 was about to reason from was **underpowered, not
saturated**.

### Consequences

* `WAVE6_STOPPING_RULE_AMENDMENT.md` §3(e): Case C ⇒ neural-track priority
  **HIGH**, and `RT-980` (GRU) becomes reachable.
* Nothing about the real-time score changes. **+0.0235 series AUC is not
  +0.0235 TS-AUC** and §H of the report forbids that arithmetic.
* No submission. `RT-600` = 0.6268 remains LB-001.

### What is next

`W6-N1` (`RT-960`/`RT-961`) — MLP on the existing 500 columns, testing **learner**
capacity — then `W6-N2` (`RT-970`/`RT-971`) — causal dilated TCN on ten legal
channels, testing **representation** learning. Pre-registered in full at
`research/WAVE6_NEURAL_PREREG.md` (`17d01b8`). `torch` is absent from the local
venv and must be installed; W6-E0, the competition whitelist, is still
outstanding and gates *deployment*, not the experiment — §8 of that document is
why.
