# STATE OF RESEARCH — V5

**2026 ADIA Lab / CrunchDAO Structural Break Challenge (Real-Time Edition)**
Written 2026-08-21. Branch `research/wave5-alpha`, parent `research/wave3-integration` @ `17bb5df`.

`VALIDATION_V2.md` and `research/WAVE5_PREREG.md` are binding. Where this file
and the repository disagree, **the repository wins**.

---

## 0. THE HEADLINE

Wave 5 found new causal information, and it is not enough to submit.

Three new feature blocks were built and gated; two of them beat a same-strength
seed clone, which no new feature family in this project had ever done before —
`m09_back` failed that test in wave 3 and the 13-way union failed it in wave 4.
The strongest, `m12_rdep`, adds **+0.00479** standalone on 4/5 folds and beats
its seed-clone blend by **+0.00141**. That is real and it is roughly half of what
the §4 promotion bar demands.

Two things were *disproved* that mattered more than another 0.001:

* **The metric does not reward early detection the way this project assumed.**
  Ages 0–20 carry **11%** of the official pair weight; age 100+ carries **57%**.
  Two of the brief's own premises rest on the opposite belief.
* **Matching the training objective to the metric's pair weighting makes the
  model worse** (−0.00147). So does a hinge (−0.00476). So does any admixture of
  bagging into the specialist blend, at every weight tested.

And the most consequential number in the repository was not produced by wave 5
at all. `codex/oracle-information-frontier-2026` measured that **the legal causal
model already matches or beats a model told where the break is, at every horizon
through h = 150.** The 2025 reproduction that was supposed to calibrate against a
strong teacher **failed its own gate** — its dependencies were never installed
and every rung of its ladder is `blocked`. Teacher distillation is not justified,
and the reason is not lack of time.

---

## 1. WHERE THE EXTERNAL SCORE STANDS

| | |
|---|---|
| LB-001, Crunch public | **0.6268** |
| RT-600 development architecture, canonical partition | 0.62581 |
| internal → external transfer | **flat to slightly positive** |
| leaderboard snapshot supplied | #1 65.10% · #10 64.10% · #25 63.60% · #50 62.91% |
| approximate standing | ~rank 59 |

The validation framework passed its first real external test. That is why wave 5
spent its compute on alpha and not on rebuilding CV.

---

## 2. WHAT IS IMMUTABLE, AND VERIFIED SO

| ref | SHA | state |
|---|---|---|
| `research/wave3-integration` | `17bb5df` | unmoved |
| `claude/rt600-baseline-submission` | `9aaa9b0` | unmoved, worktree clean |
| `codex/reproduce-2025-public-solution` | `422e4b2` | unmoved, read only |
| `codex/oracle-information-frontier-2026` | `5a3b8a0` | unmoved, read only |

`research/wave5-alpha` is strictly ahead of its parent and behind it by nothing.
The 10 GB feature cache is shared **read-only**: the wave-3 worktree still holds
exactly its original eight modules, every wave-5 artifact is local to the wave-5
worktree, and no `RT-7xx` or `wave5_*` file exists in the wave-3 OOF directory.
Wave 5 does not merge itself.

---

## 3. BASELINES, RE-ESTABLISHED ON THIS PLATFORM

Same session, same folds, same scorer.

| arm | id | TS-AUC | matches |
|---|---|---|---|
| A single control | `RT-300` | **0.61605** | V4 §3 exactly |
| B seven seed clones | `RT-421` | **0.62164** | V4 §5 exactly |
| C seven specialists (**RT-600's architecture**) | `RT-420` | **0.62581** | V4 §4 exactly |

Bagging **+0.00559**, specialisation **+0.00417**. Paired bootstrap of
specialists − seed clones: **+0.00409, CI [+0.00199, +0.00614], 200/200
positive** — W4-E1's three digits, reproduced independently.

---

## 4. WHAT THE METRIC ACTUALLY REWARDS (W5-D1)

The official weight is `n_pos(t)·n_neg(t)` per timestep. It is **not** early.

| online index t | share of pair weight | | post-break age | share of pair weight |
|---|---|---|---|---|
| 0–10 | **0.2%** | | 0–5 | **2.9%** |
| 10–50 | 3.7% | | 5–20 | 8.1% |
| 50–200 | 27.4% | | 20–100 | 32.3% |
| 200–700 | **64.3%** | | 100+ | **56.7%** |
| 700–1000 | 4.3% | | | |

25% of the weight sits at t ≤ 168, 50% at t ≤ 293, 90% at t ≤ 600.

**This corrects a premise the wave-5 brief states twice.** "We care enormously
about early evidence because real-time TS-AUC weights every timestep" is true
about timesteps and false about *weight*: ages 0–20 carry **11%** and age 100+
carries **57%**. Young-break detection is worth about a fifth of mature-break
ranking here. It also retrospectively explains `m09_back`, whose aggregate gain
was entirely mature-break — that is what this weighting rewards, and it was
still correctly rejected, on its seed-clone control rather than its age profile.

## 5. WHAT FOOLS THE CHAMPION (W5-D2), AND WHAT IT IS GOOD AT (W5-D4)

Severity = each no-break series' mean within-timestep percentile rank under the
specialist ensemble. Top 1% hardest negatives (n = 40, mean rank **0.9221**
against 0.0515 for the easiest 1%), descriptor gap in IQR units:
`tail_rate_online` **+1.89**, `shock_max_absz` **+1.50**, `burst_ratio`
**+1.38**, `level_shift_end` +0.54. Heuristic mechanism mix: heavy-tail/outlier
**40.0% vs a 25.9% base rate**; every other mechanism at or below base.

**The champion's false positives are no-break series whose online segment
carries more extreme values than their own history predicts.** Not trend, not
dependence, not spectrum.

And by break family (heuristic taxonomy, labelled as such — the wave-1 artifact
did not survive its container):

| lead family | n_pos | A single | B seed clones | S specialists |
|---|---|---|---|---|
| location | 154,551 | 0.55642 | 0.55996 | 0.56431 |
| **scale** | 373,783 | 0.68608 | 0.69360 | **0.69702** |
| spectral | 490,741 | 0.58313 | 0.58811 | 0.59236 |

**The champion is a scale detector** — 0.697 on scale-led breaks against 0.564
on location-led ones. Scale is also the only family with a positive excess over
a matched placebo null (+7.1pp), reproducing the wave-1 taxonomy independently.
Specialisation beats bagging on every family, +0.0034 to +0.0044.

## 6. CAUSALITY, AND THE GATE EARNING ITS KEEP

Every new module passes `check_prefix_invariance` at `atol = 0.0`, bitwise, on
7 series spanning `n_online` 10 → 914 including **both** length-10 series in the
dataset, at prefix cuts (1, 3, 10, 37, 113).

The gate caught three things that no score would have revealed:

1. **The first `m12_rdep` sized its expanding nulls by `n_online`** — the single
   forbidden input in this competition — and the check failed it on all 7 series
   before any number was taken from it.
2. **The first `m11_focus` was an exact O(t) scan at 526 ms/series.** The convex
   -hull functional pruning that replaced it is **bit-identical on the maximum,
   its inferred age and the anchored statistic** — verified against a brute-force
   reference on 30 random cases — and 16× faster.
3. **The stage-C queue's readiness check tested the feature cache's file size**,
   and the driver preallocates it with `open_memmap(mode="w+")`, so the file is
   full-size and mostly zeros from the first second. Two runs of `RT-740` were
   started on a 15%-filled cache and killed; neither reached the ledger. **A
   number computed from that cache would have looked entirely normal.**

## 7. IMPLICATIONS OF THE CODEX 2025 REPRODUCTION

Full review in `research/reports/wave5_codex2025_implications.md`. In short:

**The reproduction failed its own calibration gate.** `polars`, `lightgbm`,
`shap` and `tabpfn` were never installed, every rung R25-010…R25-050 is
`blocked`, and the public repository reports no AUC of its own. There is no
verified strong 2025 teacher anywhere in this project.

**The consequential measurement is on the sibling branch.**
`codex/oracle-information-frontier-2026` gave a model the true break boundary
and `h` post-break points and compared it to `RT-300`:

| h | boundary-aware oracle | legal `RT-300` | headroom |
|---|---|---|---|
| 20 | 0.5552 | 0.5610 | **−0.0058** |
| 100 | 0.6161 | 0.6180 | **−0.0019** |
| 150 | 0.6373 | 0.6397 | **−0.0025** |
| FULL | 0.6497 | 0.6100 | +0.0396 |

**At every horizon through h = 150 the legal causal model already matches or
beats a model that is told where the break is.** Two honest qualifications: the
oracle used a fixed 150-tree LGBM over a generic bank while `RT-300` is 600 trees
over 500 purpose-built columns, and it is series ROC AUC rather than TS-AUC. But
it reframes wave 5 entirely — the easy known-boundary information at the horizons
carrying the decision mass is already extracted.

**Teacher distillation is therefore NOT justified**, and not for lack of time:
the only regime with real teacher advantage is `FULL`, which requires the entire
post-break segment — exactly the future information that cannot be distilled
into a causal prefix feature.

**What the review did produce:** three of the wave-5 blocks exist because of it.
`RT-906` (residualised distances), `RT-907` (residual CUSUM/CUSUMSQ) and
`RT-908` (AR breakpoint LR) were the genuine gaps; every other 2025 idea was
checked against the production modules and found already present, redundant, or
impossible online. `RT-907` is the one that paid.

## 8. DEPLOYMENT PATH

`m12_rdep` has a bitwise streaming twin: **0 mismatches over 423,111 values** on
14 real series at `atol = 0`, **225 µs/observation**, projecting **1.959 ms/pt**
and ~9.0 h of a 15 h budget. RT-600 is provably unaffected — the seven-module
manifest still hashes to `1646c3b9…cced`, the first 500 columns of the
eight-module manifest are byte-identical, and the shipped
`models/final10k_ensemble` loads through its hard manifest gate and streams.
Detail and the two floating-point parity traps: `research/WAVE5_STATUS.md` §17.

**A pre-existing defect found on the way.** `test_engine_parity_real` fails on
`research/wave3-integration` itself, identically, before any wave-5 change: 1–2
cells per series out of ~200,000 differ between the batch and streaming
`m07_bayes`, in `bo_p_lt25_z`, `bo_lo_change_z` and `bo_ent`. The shipped RT-600
artifact has a known tiny streaming drift that no wave-3 or wave-4 document
mentions. Very unlikely to move a score at that magnitude; recorded because it
is real.

## 9. FINAL HARD-NEGATIVE RESULT

W5-E3 landed after the main state draft and did not change the Wave-5 verdict.
The nested, fold-pure miner was correct; the training intervention was not.

| arm | TS-AUC | vs control | folds |
|---|---|---|---|
| `RT-710` uniform `wbinary` control | **0.61472** | — | — |
| `RT-711` hard-negative reweighting | 0.60770 | **−0.00701** | 1/5 |
| `RT-712` hard-negative oversampling | 0.60132 | **−0.01339** | 0/5 |

Age 0–20 also moved down: −0.00142 for reweighting and −0.01019 for
oversampling. The curriculum path is therefore rejected, not carried forward to
Wave 6. Full working detail remains in `research/WAVE5_STATUS.md` §21 and
`research/reports/wave5_e3_hardneg.json`.
