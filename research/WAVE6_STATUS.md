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
