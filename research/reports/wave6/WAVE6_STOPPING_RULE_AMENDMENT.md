# WAVE 6 — STOPPING-RULE AMENDMENT

**Dated 2026-08-22.** Written after `RT-900` was voided and after Amendment 1
corrected the model-family assumption, and **before** any W6-E2R or neural
number exists.

The original rule is reproduced verbatim below and is **not deleted**. This
document supersedes it going forward; it does not rewrite the record.

---

## 1. THE ORIGINAL RULE, VERBATIM

From `research/WAVE6_PREREG.md` §0, written 2026-08-22 before any Wave-6 number
existed:

> **THE STOPPING RULE, FIXED NOW.** If **W6-E1 and W6-E2 both return inside
> noise** (< +0.0010 and < +0.010 respectively, definitions in §3 and §4), Wave 6
> **stops adding** and the recommendation becomes: ship RT-600, write the wave up,
> and do not open W6-E3. Four independent measurements already say this system is
> saturated for the kind of work Wave 5 did. A fifth negative is information; a
> sixth is not.

That rule was sound when written. It is being amended because **one of its two
input arms turned out not to exist**, and because a premise it silently relied
on was wrong.

---

## 2. WHY THE EVIDENCE PATH CHANGED

### 2.1 One arm resolved as designed

**W6-E1 — calibration anchor placement — is FALSIFIED and stands.** Every
scheme, including a random-anchor null, lands inside 0.00006 of the shipped
`SCDF_NSEEN`:

| scheme | dev TS-AUC | Δ vs shipped |
|---|---:|---:|
| `CAL-LOG` (parity check, must reproduce the incumbent) | 0.62581 | — |
| `CAL-WT` metric-weight anchors | 0.62579 | −0.00002 |
| `CAL-HYB` hybrid | 0.62581 | −0.00001 |
| `NULL-ANCHOR` random placement, mean | 0.62578 | −0.00003 |

Zero grid fallbacks in every scheme; the parity check reproduced the shipped
calibration to five decimals on all five folds. The Part-13 measurement that
motivated E1 (9 of 12 anchors covering 25% of the metric's pair weight) was
descriptively correct; the **inference that it mattered was wrong**, because
SCDF interpolates smoothly in `log(t+1)` and twelve knots anywhere recover the
map. This arm is resolved: **inside noise**, exactly as the rule's first
condition anticipated.

### 2.2 The other arm never produced a measurement

**W6-E2 / `RT-900` is VOID.** It did not return "inside noise" and it did not
return "outside noise". It returned **0.86552**, which is not a measurement of
localisation value at all — it is the row-level label arriving through the
oracle block's missingness mask. The bare indicator `1[t >= cut]` scores
**0.81442** TS-AUC by itself; 100.00% of the rows where the block is `NaN` are
negatives.

The stopping rule is a conjunction over two measurements. **One of its two
conjuncts is undefined.** A rule cannot fire on an operand that does not exist,
and it must not fire on a fabricated one. Triggering "stop adding" here would
mean stopping on the strength of a leak.

### 2.3 A premise of the rule was factually wrong

The original rule was written on the belief — recorded in the pre-amendment
`§5.2` — that the deployable model family was effectively LightGBM and
scikit-learn. **Amendment 1 established that this was never the constraint.**
CrunchDAO's rule is a per-competition dependency whitelist declared through
`requirements.txt`, with a request route for packages not yet on it; PyTorch,
XGBoost and CatBoost are all live candidates for evaluation.

"Four independent measurements say this system is saturated" was and remains
true — **for the kind of work Wave 5 did**, which was adding statistics to a
LightGBM feature bank that already computes overlapping statistics. It is not
evidence about a model family that has never been run. The rule generalised from
a saturated *search direction* to a saturated *problem*, and those are not the
same claim.

### 2.4 What has NOT changed

Nothing about the difficulty of the task, the validation surface, the promotion
bar, or the leaderboard policy. Wave 5's central finding stands in full:
`m12_rdep`'s advantage shrinks monotonically as the comparison approaches the
deployed system (+0.0038 standalone → +0.0010 two-model → +0.0005 eighth member
→ −0.0027 architecture rebuild). **The information is real and redundant.** That
finding is the reason the neural track is framed as *a different family*, not as
*more features*.

---

## 3. THE AMENDED RULE

Effective 2026-08-22, replacing `WAVE6_PREREG.md` §0's stopping rule:

> **AMENDED STOPPING RULE.**
>
> **(a) W6-E1 is resolved and closed.** Calibration-anchor placement is
> falsified. No further anchor, knot, coordinate or window-width experiment is
> authorised in Wave 6. This arm may not be reopened by a new parameterisation
> of the same idea.
>
> **(b) W6-E2 produced no measurement.** `RT-900` is void and contributes
> **zero** evidence in either direction. The corrected experiment **W6-E2R**
> (`WAVE6_PREREG.md` §18) replaces it and must reproduce the prior
> oracle-frontier control before anything is read off it.
>
> **(c) Wave 6 stops adding *statistical feature blocks*.** The Wave-5 evidence
> is sufficient: this ensemble is saturated for hand-designed statistics that
> overlap what its 500 columns already compute. `m11_focus`, `m10_persist`,
> block unions and any successor of the same shape are closed for Wave 6. This
> half of the original rule is **retained and strengthened**.
>
> **(d) The neural track is NOT blocked by (c) and is NOT gated on W6-E2R.** It
> is a different model family, not another feature block, and it was never
> among the four measurements the original rule generalised from. It proceeds
> under `research/WAVE6_NEURAL_PREREG.md` with the promotion bar of
> `WAVE6_PREREG.md` §2 plus §20's additional controls, applied unchanged.
>
> **(e) W6-E2R sets the neural track's PRIORITY, not its permission.**
>   * Case B/C (representation headroom, `Δ >= +0.005` robust) → priority
>     **HIGH**; N1 and N2 both run, N3 becomes reachable.
>   * Case A (representation saturated, `|Δ| < 0.005`) → priority **MODERATE**;
>     **N1 and N2 still run, once each**, because the real-time row-level task
>     is not the series-level diagnostic and a null on the second does not
>     settle the first. N3 does not open.
>   * Case D (failed reproduction or a fired sentinel) → the instrument is
>     uncalibrated; nothing is inferred from it, and the neural track proceeds
>     at MODERATE priority on its own justification.
>
> **(f) The new terminating condition.** Wave 6 stops adding when **both**
> neural input tracks (§20 Track A and Track B) have been run once each, under
> controls, and **neither** clears the §2 promotion bar against a matched seed
> clone. At that point the recommendation becomes: ship `RT-600`, write the wave
> up, and open no further Wave-6 candidates.
>
> **(g) Unconditional, and not amendable by a later amendment of mine.** No
> Crunch submission is authorised by anything in Wave 6 without canonical CV, a
> seed control, alternate partitions, a final freeze, the final-10k fit and
> release validation. `RT-600` = **0.6268** remains LB-001. Void results never
> enter a promotion decision.

---

## 4. WHAT THIS AMENDMENT DELIBERATELY DOES NOT DO

* It does not reopen feature engineering. (c) is stricter than the original.
* It does not license architecture fishing. §20 caps the search at two sizes and
  two regularisation settings per family, with the loss fixed in advance.
* It does not lower the promotion bar by a single basis point.
* It does not treat `RT-900`'s 0.86552 as evidence of anything except a leak.
* It does not claim W6-E2R will find headroom. The interpretation table in §18.6
  was fixed before the run and includes the null.

---

## 5. AUTHORSHIP AND HONESTY NOTE

The rule being amended is one I wrote, and the leak being worked around is one I
designed. Both facts are why this document exists rather than a silent edit to
§0: an amendment that quietly relaxes a constraint after the constraint became
inconvenient is worth nothing. The original text is preserved above in full so
that anyone can check whether the reasoning here is a correction or a
rationalisation.

The strongest argument *against* this amendment is that a researcher who has
just spent a wave finding nothing, and then invalidated their own headline
result, is exactly the researcher most motivated to find a reason to keep going.
That argument is real. The answer to it is (c) and (f): the amendment closes the
direction Wave 5 exhausted, and it fixes a terminating condition for the new
direction **before** the new direction has produced a single number.
