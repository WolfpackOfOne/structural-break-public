# LEADERBOARD ASSAULT — LIVE STATUS

**Wave 7. Branch `claude/project-setup-github-20g6pt`** (the designated development
branch for this session; it now carries both the wave-6 research line and the
LB-001..005 release line, merged at wave-7 open).

| | |
|---|---|
| parent research branch | `research/wave6-alpha` @ `33fc210` |
| parent release branch | `claude/rt600-baseline-submission` @ `9aaa9b0` |
| wave-7 starting SHA | the merge commit that opens this branch |
| **TS-AUC values observed in wave 7** | **0** |
| **training runs in wave 7** | **0** |
| **leaderboard submissions in wave 7** | **0** |

---

## A. CURRENT CHAMPION

| | | |
|---|---:|---|
| **external** | **0.6268** | LB-001, RT-600, Crunch public. Unchanged. |
| **internal** | **0.62581** | RT-600 architecture, canonical 5-fold, seven equal-weight specialists |
| internal, single model | 0.61605 | RT-300 — the control every lane is measured against |
| internal, seven seed clones | 0.62164 | RT-421 — the bagging control that separates diversity from specialisation |

**RT-600 is untouched.** No refit, no calibration tweak, no dependency change, no
production mutation (brief §48). Wave-7 objects live entirely in
`research/scripts/wave7_*.py` and cannot reach a manifest.

## B. THE HARD BLOCKER

**The 2026 store is absent from this container and cannot be fetched here.**

```
api.crunchdao.com:443   403 at CONNECT (organization egress policy)
hub.crunchdao.com:443   403 at CONNECT
huggingface.co:443      403 at CONNECT
```

Recorded in the agent proxy's own failure log (`$HTTPS_PROXY/__agentproxy/status`).
Per the proxy's documented policy a 403 is an organization denial and is reported,
not routed around. pypi.org and files.pythonhosted.org **are** reachable, so
`xgboost` and `catboost` can be installed the moment they are needed.

Consequence: **every lane in this wave is data-bound.** The five pilots are coded,
gated, tested and pre-registered; each one hard-stops with the remedy printed
rather than failing deep inside a fold:

```
HARD STOP: the 2026 store is absent at <path>.
Wave 7's pilots are all data-bound.  Set SBR_STORE to a built store,
or build one, then re-run this script unchanged.
```

## C. GAP TO EACH TARGET

Expressed the way the assault should be planned — as a share of the inversions
that remain, not as an abstract delta.

| target | gap from 0.6268 external | internal-equivalent gap | **share of remaining weighted loss to convert** |
|---|---:|---:|---:|
| 0.630 | +0.0032 | +0.0042 | **1.12%** |
| 0.635 | +0.0082 | +0.0092 | **2.45%** |
| 0.640 | +0.0132 | +0.0142 | **3.79%** |
| 0.645 | +0.0182 | +0.0192 | 5.13% |
| 0.650 | +0.0232 | +0.0242 | 6.46% |

Total remaining weighted loss is 0.37437. Every named error class has a ceiling
larger than any of these gaps. **Headroom is not the constraint; capture rate is.**

## D. ACTIVE LANES — ranked by the budget, not by the brief's ordering

| rank | lane | targets which error class | class's % of loss | ceiling | pilot | cost |
|---|---|---|---:|---:|---|---|
| **1** | **W7-D3 dominant-cell diagnostic** (new; see §G) | mature persistent break, late online | **45.7%** | +0.171 | series-level, one protocol | ~30 min |
| 2 | **Lane B** horizon specialists, pilot **H4** `t∈[101,251)` | mid-age + mature, mid horizon | 20.3% at H4; 65.8% at H5+H6 | +0.078 at H4 | `wave7_b_horizon.py` | 1 fold |
| 3 | **Lane A** teacher → causal student, target **oracle evidence** | all ages; strongest where the label is most overconfident | up to 45.7% | teacher privilege 1.6–2.4× in the dominant cells | `wave7_a_distill.py` | 1 fold × 3 arms |
| 4 | **Lane C** XGBoost on the identical causal state | none specifically — a learner-family bet | n/a | n/a | `wave7_c_xgb.py` | 5 folds |
| 5 | Lane D pairwise-neural 10-minute diagnostic | mechanism question only | n/a | n/a | one-line objective swap in `wave6_n1_mlp.py` | 10 min |
| — | Lane C′ CatBoost | only after the XGBoost screen | n/a | n/a | not written | — |

Two of these differ from the wave-7 brief, on measured grounds recorded in
`research/WAVE7_ALPHA_BUDGET.md` §8:

* **Lane B pilots H4, not the early regime.** `t < 50` is 4.7% of the loss with a
  +0.018 ceiling; a perfect early specialist cannot reach 0.635.
* **Lane A leads with oracle evidence, not permanence.** Permanence is a young-age
  mechanism and ages 0–20 hold 13.6% of the loss; at age 100+, where half the loss
  sits, permanence is not in doubt.

## E. KILLED / FROZEN LANES

| lane | why | evidence |
|---|---|---|
| GRU / transformer / larger TCN / larger MLP | both neural input tracks failed the controlled screen; the terminating rule fired exactly as written | W6-N1/N2: 0.5406–0.5806 vs a 0.61661 seed clone; every bootstrap CI on the seed-clone contrast includes zero |
| age-gated neural blend | needs true age at inference, which is τ, which is the label | W6-E2 void |
| any oracle/τ block as a **feature** | under TS-AUC "give the model τ" is degenerate — a bare `1[t≥cut]` indicator scores 0.81442 | W6-E2 VOID, label leak via the missingness mask |
| calibration anchor placement | every placement, including random, scores within 0.00006 | W6-E1, best scheme −0.00001 against a +0.0030 bar |
| specialist/bagging mixture | delta monotone decreasing in bagging weight; tight CI, clean null | W5-E1 |
| residual CUSUMSQ, absorbing-state BOCPD, robust distribution distances, mean-vs-robust-mean | redundant or algebraically dead, rejected before any compute | W5-R1/R2, C2 freeze §4 |
| **heavy-tail false-alarm repair as a headline lane** | **new in wave 7** — the most confidently diagnosed failure mode is worth at most +0.0092, realistically +0.0055 | W7-D1 §5 |
| **early-horizon specialisation as the lane-B entry point** | **new in wave 7** — 4.7% of loss, +0.018 ceiling | W7-D1 §3 |
| calibration tuning, SCDF knots, ensemble-weight searches, seed fishing, recency gates, hard-negative threshold sweeps | frozen for this wave by brief §27 | — |

## F. ALPHA BUDGET BY ERROR CLASS

Full derivation and provenance: `research/WAVE7_ALPHA_BUDGET.md`.

| error class | fraction of weighted inversions | theoretical perfect-repair Δ | currently targeted by | priority |
|---|---:|---:|---|---|
| mature persistent break vs no-break, late online (`t≥200`, age≥100) | **45.7%** | +0.17096 | **nothing in the original plan** → W7-D3 | **1** |
| mid-age break (20 ≤ age < 100) | 35.1% | +0.13150 | lane B | 2 |
| young break (age < 20) | 13.6% | +0.05089 | lane A permanence; W6 neural young-age effect | 3 |
| short online segment (`n_online < 200`) | 9.1% | +0.03406 | lane B H1–H3 | 4 |
| late break (`τ > 0.74`) | 4.5% | +0.01703 | — | 5 |
| heavy-tail / outlier false alarm (top 1% negatives) | 1.5% | +0.00922 | diagnosed, no module | 6 |

Unsized without OOF vectors: variance-only break, dependence-only break, weak
magnitude, online-vs-history offset.

## G. NEXT EXPERIMENT — SINGLE HIGHEST VALUE

**W7-D3 — the dominant-cell diagnostic. Runs before any of the three lane pilots.**

45.7% of remaining loss is one phenomenon: a break more than 100 observations
old, in a series observed more than 200 online steps, still inverted against a
no-break series. No lane in the wave-7 brief is aimed at it, and the two readings
of it lead to opposite research programmes:

* **information limit** — 0.66 at age 100+ is near Bayes for this sample, the
  honest ceiling is ~0.635, and lanes A–C fight over the remainder;
* **representation limit** — W6-E2R measured our causal bank beating the oracle
  frontier's generic bank by **+0.02354 series ROC AUC**, 25/25 folds positive,
  *widening* under matched column width, on exactly the mature whole-series
  question this cell asks.

The experiment: take the W6-E2R series-level protocol, restrict it to the
dominant joint cell, remove the boundary, and ask the question row-wise. Measure
how much of the +0.0235 survives. It is a diagnostic, not a model; it reuses
`research/scripts/wave6_e2r.py` wholesale; and it decides whether the wave is
chasing +0.005 or +0.015.

**Then**, in order: lane B H4 pilot → lane A one-fold three-arm pilot → lane C
XGBoost screen → lane D ten-minute diagnostic. Kill aggressively at each gate.

## H. SUBMISSION DECISION

**NO. Nothing in wave 7 has been scored.**

The policy stands unchanged from brief §35: a candidate needs ≥ +0.004 robust
internal improvement, ≥ 4/5 folds, a passing same-strength seed-clone control and
supporting alt partitions before a submission is prepared. Nothing has met the
first condition because nothing has met the precondition of having been run.

## I. END-OF-STAGE TABLE

| lane | pilot Δ | mechanism evidence | theoretical alpha budget | cost | continue? |
|---|---:|---|---:|---:|---|
| teacher / distillation (A) | **not run — store absent** | privilege is real (261 unseen points, 42.6% of segment) but 1.6–2.4× where the weight is; the hard label is measurably overconfident (92.8% of breaks are `weak_unclassified` at AUC 0.6005) | +0.171 in principle; the *evidence-target* mechanism is unquantified | 1 fold × 3 arms | **YES — coded, gated, pre-registered, blocked on data** |
| t-specialists (B) | **not run — store absent** | trees must share capacity across regimes whose weight differs 45× (`t` 0–20 = 0.82% vs 200–400 = 36.6%) | +0.078 at H4, +0.253 at H5+H6 | 1 fold | **YES — pilot region corrected to H4** |
| XGBoost (C) | **not run — store absent** | none specific; a learner-family bet against the wave-6 lesson that decorrelation alone is worth +0.0001 | n/a | 5 folds | **YES, but ranked 4th** |
| CatBoost | not run | same | n/a | — | **only after the XGBoost screen** |
| pairwise neural (D) | **not run — store absent** | W6 showed BCE 0.077 alongside a score 0.045 below trees, which is objective mismatch; but same-`t` pairwise has already lost on trees | must close most of the 0.035–0.045 MLP gap to matter | 10 min | **10 minutes only** |
| **dominant-cell diagnostic (W7-D3)** | **not run — store absent** | W6-E2R: +0.02354 series AUC over the oracle frontier's bank, 25/25 folds, widening under matched width | decides between a +0.005 and a +0.015 wave | ~30 min | **YES — run it first** |

## J. WHAT IS READY TO RUN THE MOMENT A STORE EXISTS

```
export SBR_STORE=/path/to/cache/store        # also SBR_FEATURES for the built bank
python research/scripts/wave7_b_horizon.py  --fold 0 --regime H4
python research/scripts/wave7_a_distill.py  --fold 0 --arms A0,A1,A2 --prove-deletion
python research/scripts/wave7_c_xgb.py      --folds 0,1,2,3,4
```

Gates that already pass here, with no data present: `tests/test_wave7_lanes.py`,
**15 passed** — teacher-leak refusal (including the gate's own self-test), the
deliberate non-prefix-invariance of teacher labels, prefix-identical horizon
routing, deterministic analytic blend weights, and a proof that the bucket report
used to gate every candidate is the exact decomposition of the official metric
rather than an approximation of it.

Repository-wide: **607 passed, 19 failed, 15 skipped.** Every failure is a
`FileNotFoundError` against the absent store or one of the two synthetic
`m07_bayes` parity cases already pinned in `research/known_failures.json`. No
failure is in a file this wave touched.
