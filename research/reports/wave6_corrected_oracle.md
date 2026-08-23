# W6-E2R — THE CORRECTED SERIES-LEVEL KNOWN-BOUNDARY REPRESENTATION TEST

**ORACLE / DIAGNOSTIC — NOT DEPLOYABLE.** Uses true and pseudo boundaries and
post-boundary observations. It modifies no production feature, no submission
notebook, no champion selection, nothing under `src/sbr/production/`, and it
touches neither the old lockbox nor the reduced test data.

**Metric: series ROC AUC, one row per series. Never row-level TS-AUC.**

Pre-registered in `research/WAVE6_PREREG.md` §18, committed at `768204e`
**before** this run reported. Runner `research/scripts/wave6_e2r.py`; artefacts
`wave6_corrected_oracle.{json,csv}` and `wave6_corrected_oracle_width.json`.

---

## Executive result

**Representation is still a live lever.** Given the same boundary, the same
learner, the same folds and the same scorer, our 500-column causal bank beats
the prior oracle study's generic bank by **+0.0235 series AUC** — **5 of 5
pseudo-τ seeds, 25 of 25 folds, every bootstrap CI excluding zero**. The
pre-registered **Case C** threshold (+0.010) is cleared by more than a factor of
two, and the advantage **grows** when arm B is cut down to arm A's exact column
count.

The instrument is calibrated: the prior study's `0.6497` reproduced to
**0.64969**, a delta of **+0.00000**. All five leakage sentinels are clean.

---

## A. RT-900 POSTMORTEM

| | |
|---|---:|
| `RT-900` TS-AUC | **0.86552** |
| champion `RT-300` | 0.61605 |
| apparent delta | **+0.24947**, 5/5 folds |
| **status** | **VOID — LABEL LEAK VIA THE MISSINGNESS MASK** |

The oracle block was `NaN` for every row with `t < cut`, because there is no
post-cut segment to compute over. For a break series `cut = tau`, so the block's
**missingness mask is the row-level target** `y[t] = 1[t >= tau]`, and LightGBM
splits on missingness natively.

| diagnostic | value |
|---|---:|
| TS-AUC of the bare indicator `1[t >= cut]`, nothing else | **0.81442** |
| share of dev rows where the block is `NaN` | 49.1% |
| **share of those `NaN` rows that are negatives** | **100.00%** |

An indicator that never looks at the data beats the champion by +0.198.

**No production path was touched.** `w6oracle` was never registered with
`sbr.features.base.load_all()`, never in a manifest, never in a `crunch test`,
never in an ensemble, never in a submission. The row is retained with
`status=VOID` in `research/RESULTS.csv` and appears in no comparison table as
valid alpha.

**The design error was mine, and it was avoidable.** A +0.249 delta on an
ensemble that four independent Wave-5 measurements had shown to be saturated was
never plausible. The correct first move on an implausibly large effect is to try
to break it, not to interpret it.

---

## B. WHY ROW-LEVEL TRUE τ IS DEGENERATE — AND WHY NO PATCH EXISTS

The placebo cut — no-break series receive a cut drawn from the positives'
relative-τ distribution — is the wave-1 taxonomy's construction and the
oracle-frontier study's, and it is **correct at the series level**: "does this
series contain a break?" is not answered by knowing where the boundary is.

It cannot work **at the row level**, because "has the break happened by now?" is
answered by the boundary *exactly*. Same cut, different question, different unit
of analysis.

Removing the explicit timing columns does not repair it. Dropping `or_elapsed`
and `or_frac` kills the `NaN` mask, but the same information returns through
`or_frac = 1.0` exactly when `t < cut`, and through the segment length the null
calibration matches on. Under the row-level target, **τ *is* the label**, so
every channel that carries τ carries the answer.

**Standing rule**, now in `research/PROTOCOL.md` §1: for the row-level
real-time target, any experiment that gives the learner true τ — directly, or
through a feature's **availability, support, length, missingness, denominator,
calibration window or segment boundary** — is invalid for predictive-performance
measurement. True τ remains legal for post-hoc diagnostics, age-bucket
evaluation, series-level oracle studies and quarantined teacher analysis.

**Regression tests:** `tests/test_no_tau_leakage.py`, 7 gates. The load-bearing
one scores every production column's `NaN` mask **with the official scorer** over
a panel of series with different τ. That distinction is the test: a fixed
warm-up mask correlates with `y` at 0.99 when rows are pooled and carries
*exactly zero* ranking information within a timestep, while RT-900's mask carried
all of it. A pooled-correlation test flags `m03_dyn.drift_q` and misses the
point entirely — I wrote that version first and it failed for the wrong reason.

---

## C. CORRECTED DESIGN

One row per series. Series label. Series ROC AUC. FULL horizon.

| item | value |
|---|---|
| population | canonical dev folds 0–4, 8,000 series; fold −1 excluded |
| store | sha256 `2c6aab9b…3971` (meta), `10c22b00…c06b` (values) |
| folds | sha256 `ba4f71fe…c312` |
| frontier module | `oracle_information_frontier.py` sha256 `0b56baae…7bc8`, **imported by path and not modified** |
| boundary | positives `tau_index`; negatives `assign_pseudo_taus` verbatim |
| pseudo-τ seeds | 0, 1, 7, 42, 2026 |
| learner | LGBM 150 trees, the frontier's hyperparameters, median imputer |
| learner seed | `pseudo_seed + 17` — **identical across every rich arm** |
| eligibility | `post_len >= 10`, applied identically to every head-to-head arm |

**How arm B is given the boundary.** The engine runs **unmodified** on a re-split
series: `hist' = hist ++ online[:boundary]`, `online' = online[boundary:]`, and
the **last row** of its `(n_online', 500)` output is the series vector. This is
the exact analogue of the frontier's `compare_segment(post, hist)`. No feature
is invented, no module is modified, nothing is fitted to the label.

**`t_online` / `log_t_online` are dropped from the primary arm** — at the last
row those two columns *are* `post_len`, which the frontier excludes from every
model. The undropped variant is reported so the size of that channel is visible.

**Eligibility cost.** 8,000 → ~7,724 series; ~166 of the dropped are positives
(τ within 10 points of the end) and the rest negatives whose pseudo-τ landed
near the end. `A_rich` is **also** scored on the unfiltered 8,000, which is the
reproduction check.

---

## D. LEAKAGE-SENTINEL RESULTS — ALL CLEAN

Read **before** the arms, as required. The prior study's own FULL-horizon
controls are the calibration.

| sentinel | uses only | ours | frontier's | verdict |
|---|---|---:|---:|---|
| `S1` boundary metadata | boundary, rel_boundary, n_hist, n_online, post_len | **0.51374** | 0.5379 | clean, and **below** the frontier's |
| `S2` missingness, arm B | the `NaN` pattern of our 498 columns | **0.51348** | — | clean |
| `S2` missingness, arm A | the `NaN` pattern of the frontier's 280 | **0.51650** | — | clean |
| `S3` support / length | valid-column counts, pre length, post length | **0.52624** | — | clean |
| `S4` permuted labels, arm B | our 498 columns, labels shuffled | **0.50152** | 0.5075 | chance, as required |
| `S5` placebo boundary, arm B | our 498 columns, random cut for **both** classes | 0.62312 | — | see below |
| `S5` placebo boundary, arm A | the frontier's 280, random cut for both classes | 0.59470 | 0.5821 | consistent with the prior study |

**S5 is not a failure and was pre-registered not to be.** A randomly placed cut
still splits a break series into segments that differ on average, so a nonzero
placebo AUC is expected — the frontier measured 0.5821 for its own bank and we
measure 0.5947 for the same bank on our population. The informative quantity is
the *gap* between the two banks under a placebo boundary:

| | true boundary | placebo boundary | boundary is worth |
|---|---:|---:|---:|
| arm A, frontier bank | 0.65037 | 0.59470 | +0.0557 |
| arm B, our bank | 0.67391 | 0.62312 | +0.0508 |
| **B − A** | **+0.0235** | **+0.0284** | — |

Our bank's advantage **survives, and is slightly larger, under a boundary that
carries no information at all**. Whatever arm B is doing, it is not exploiting
the boundary more aggressively than arm A — it is a better representation of the
series either way.

---

## E. PRIOR FRONTIER REPRODUCTION — EXACT

| | series ROC AUC |
|---|---:|
| prior study, `lgbm_rich`, FULL, mean over 5 pseudo seeds | 0.6496854 |
| **this run, `A_rich` on the unfiltered 8,000** | **0.6496899** |
| **delta** | **+0.0000045** |

Per-seed: 0.64431 · 0.65130 · 0.64712 · 0.65164 · 0.65405, std 0.00390 against
the prior study's 0.0035. Secondary control `A_basic` = 0.64034 against the
prior 0.6418.

All four input hashes match the prior study's manifest byte for byte. The
reproduction gate was **±0.010**; it came in at **±0.000005**. The instrument is
calibrated and the comparison is against a live control, not a stale scalar.

---

## F. CURRENT REPRESENTATION RESULT

| arm | seed 0 | seed 1 | seed 7 | seed 42 | seed 2026 | mean | std |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A_rich_full_population` (reproduction) | 0.64431 | 0.65130 | 0.64712 | 0.65164 | 0.65405 | **0.64969** | 0.00390 |
| `A_rich` frontier bank + boundary | 0.64863 | 0.65139 | 0.64735 | 0.65059 | 0.65389 | **0.65037** | 0.00253 |
| `A_basic` frontier basic bank | 0.63782 | 0.64178 | 0.63137 | 0.64813 | 0.64261 | **0.64034** | 0.00622 |
| **`B_causal` our bank + boundary** | 0.68178 | 0.67537 | 0.67323 | 0.66692 | 0.67226 | **0.67391** | 0.00539 |
| `B_causal_withpos` (+ position cols) | 0.67737 | 0.67332 | 0.67568 | 0.66922 | 0.67764 | 0.67465 | 0.00349 |
| `C_nobound` our bank, **no boundary** | 0.64470 | 0.64872 | 0.64088 | 0.64367 | 0.64846 | **0.64529** | 0.00333 |
| `AB` union | 0.68495 | 0.67860 | 0.67617 | 0.67409 | 0.67862 | **0.67848** | 0.00408 |
| `legal_RT300_no_boundary` (reference) | 0.61928 | 0.61887 | 0.61980 | 0.61946 | 0.61955 | 0.61939 | 0.00035 |

### Robustness of `B − A`

| | |
|---|---|
| mean Δ | **+0.02354** |
| per seed | +0.03314 · +0.02398 · +0.02588 · +0.01633 · +0.01837 — **5/5 positive** |
| per fold | **25 of 25 positive**, smallest +0.00716 |
| paired series bootstrap, 400 reps | CI excludes zero on **every** seed; worst lower bound **+0.00567**; 99.8–100% of resamples positive |

### The column-count control (post-hoc, declared as post-hoc)

Arm B carries 498 columns against arm A's 280, which is a real alternative
explanation and was not anticipated in the pre-registration. Arm B was randomly
subsampled to arm A's **exact width**, three independent draws:

| pseudo seed | `A_rich` (280 cols) | `B_causal` (498) | `B_causal` at **280** cols | Δ at matched width |
|---|---:|---:|---:|---:|
| 0 | 0.64863 | 0.68178 | 0.67885 · 0.68025 · 0.67923 → **0.67944** | **+0.03081** |
| 2026 | 0.65389 | 0.67226 | 0.67473 · 0.67204 · 0.67641 → **0.67439** | **+0.02050** |

**The advantage is larger at matched width, not smaller.** Arm B loses almost
nothing when 44% of its columns are discarded at random, which is what a
genuinely richer representation looks like and is not what a
more-columns-wins artefact looks like. `C_nobound` says the same thing from the
other side: 498 of our columns **without** the boundary (0.64529) score
*slightly below* 280 of the frontier's **with** it (0.65037), so width alone
does not buy the gap.

### What the boundary is actually worth, decomposed

Mean over five pseudo seeds, eligible population, each row adding one thing:

| | series AUC | step |
|---|---:|---:|
| `RT-300`, the shipped row-level stream, read at the final online row | 0.61939 | — |
| our **same 500 columns**, refit for the series question, **no boundary** | 0.64529 | **+0.02589** |
| … plus the **true boundary** | 0.67391 | **+0.02863** |
| … plus the frontier's generic bank on top | 0.67848 | +0.00457 |
| *(for reference: the frontier's generic bank + boundary)* | *0.65037* | |

Three things fall out of this, and the second is the one that changes the map.

1. **Knowing τ is worth about +0.029** to our own representation at FULL. Real,
   and roughly the same size as the representation gap.
2. **+0.0259 of the prior study's +0.0396 "model-extraction gap" is not a
   representation gap at all.** It is the difference between reading a
   *row-level* model's last prediction and fitting a *series-level* classifier
   on the same 500 columns. No new information, no boundary — just answering the
   question that was actually asked. **This portion is not available to the
   real-time task**, because the real-time metric never asks the series
   question. See §H.
3. **The two banks are partly complementary**: +0.0046 remains in the frontier's
   generic bank after ours is present. Small, but not zero — its `pre{20..500}`
   windowed contrasts are computed against references our streaming engine does
   not maintain.

---

## G. REPRESENTATION-HEADROOM VERDICT

**CASE C — representation is clearly still live.** Pre-registered thresholds:
Case B at +0.005, Case C at +0.010. Measured **+0.0235**, 5/5 seeds, 25/25
folds, bootstrap CI excluding zero on every seed, **+0.0257 at matched column
width**, and **+0.0284 under a placebo boundary** where the boundary carries
nothing.

The prior study's generic 150-tree oracle bank was **underpowered**, not
saturated. Its conclusion — "the same diagnostic feature/model family is weak on
both local 2025 and 2026" — was a statement about *that bank*, and this run
shows a materially better bank exists inside our own repository. Its main
finding, that the 2026 data does not hide a 0.90-class signal, is untouched:
0.67848 is a long way from 0.90.

Under `WAVE6_STOPPING_RULE_AMENDMENT.md` §3(e), Case C sets the neural track's
priority to **HIGH** and makes `RT-980` (GRU) reachable.

---

## H. WHAT THIS DOES **NOT** PROVE

Binding on every downstream write-up.

1. **0.678 is not a ceiling.** Not Bayes-optimal, not an information-theoretic
   limit, not a frontier. It is the performance of *one representation* and *one
   learner* under *known-boundary series-level* evaluation. The correct words
   are **representation diagnostic**, **known-boundary benchmark**, **measuring
   instrument**. This document does not use "true information frontier" and
   neither should any successor.

2. **It does not transfer to the real-time score.** The series question ("does
   this series contain a break, judged at the end?") and the row question ("has
   the break happened by time `t`?") are different questions, which is the entire
   lesson of RT-900. **+0.0235 series AUC at FULL is not +0.0235 TS-AUC** and
   nothing here licenses that arithmetic. In particular, the +0.0259 step in §F
   is a question-mismatch artefact and is unavailable to the real-time task by
   construction.

3. **It does not say a neural model will work.** It says the 500-column bank
   contains materially more extractable series-level structure than the generic
   bank did, at a horizon where the generic bank was the basis for concluding
   the problem was near-saturated. That removes an argument *against* the
   representation programme; it supplies no evidence *for* any particular
   architecture, and it is explicitly forbidden from selecting a single neural
   hyperparameter (`WAVE6_PREREG.md` §18.8).

4. **It does not license true-τ features anywhere near production.** Arm B is an
   offline diagnostic and can promote nothing. Every rule in §B stands.

5. **The comparison is bank-to-bank, not method-to-method.** Arm A is the
   frontier's bank at the frontier's capacity (150 trees, no hyperparameter
   sweep, as its own Limitations section notes). A better-tuned generic bank
   would close some of the gap. What the run establishes is that the *specific*
   scalar 0.6497, which Wave 6 was about to reason from, is not the level a good
   causal representation reaches.

6. **FULL only.** The frontier showed legal `RT-300` already matching or beating
   its oracle at every horizon through `h = 150`. Nothing here was measured at
   short horizons and nothing should be claimed there.

---

## Provenance

| | |
|---|---|
| branch | `research/wave6-alpha` |
| pre-registration | `research/WAVE6_PREREG.md` §18, committed `768204e` **before** this run reported |
| runner | `research/scripts/wave6_e2r.py` |
| post-hoc width control | `research/scripts/wave6_e2r_width.py` |
| report assembly | `research/scripts/wave6_e2r_report.py` |
| artefacts | `wave6_corrected_oracle.json` · `.csv` · `_width.json` |
| runtime | 1,745 s for the main run (30,669 distinct `(id, boundary)` featurisations) |
| IDs | `RT-940` `A_rich` · `RT-941` `A_basic` · `RT-942` `B_causal` · `RT-943` `C_nobound` · `RT-944` `AB` |
| `RESULTS.csv` rows | **none, by design** — that file is the row-level TS-AUC ledger |
| Crunch submissions | **none.** `RT-600` = 0.6268 remains LB-001 |
