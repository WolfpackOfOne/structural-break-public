# HANDOFF — 2026 ADIA Lab / CrunchDAO Structural Break Challenge (Real-Time Edition)
**Written 2026-08-20 for whoever picks this up next. Read it before touching anything.**

Competition deadline: **17 September 2026**. Objective: maximise true out-of-sample
**Time-Stratified AUC**.

---

## 0. THE THREE THINGS THAT WILL WASTE YOUR TIME IF YOU DON'T READ THEM

1. **There are two divergent branches and they both claim "RT-150".** They fork at
   `e98f5b9`. `research/wave2-2026` is the one to use. See §2.
2. **The submission notebook does not exist.** `.gitignore` line 65 excludes
   `submissions/*.ipynb` as a build artifact, and the trained model directory and
   all wave-2 OOF vectors are uncommitted too. The wave-2 doc reports it passing at
   4.57 ms/point; **that is unverifiable from the repository**. See §4.
3. **The official `crunch test` cannot run in a cloud container.**
   `api.hub.crunchdao.com` returns 403 at the egress proxy. It can only run on the
   user's Mac. See §4.

---

## 1. SCOREBOARD — with provenance, because provenance is the whole game

| model | what it is | score | label |
|---|---|---|---|
| `RT-000` | shipped handcrafted EWMA/CUSUM/variance noisy-OR | 0.52051 | DISCOVERY, wave 1 |
| `RT-100` / `RT-100R` | 7 causal feature modules, 500 cols, LightGBM binary, 1M rows | **0.61510** | **CONFIRMATION** — reproduced from a clean checkout with delta exactly 0.0 |
| `RT-100` on lockbox | same, trained on all dev, scored on 2,000 never-touched series | 0.60791 | lockbox, **spent** |
| `RT-131` | within-timestep cross-sectional rank average | ~0.6254 | **ILLEGAL** — diagnostic upper bound only |
| `RT-150` (wave 2) | 7 streams + frozen cross-fitted smooth time-conditional CDF, equal average | 0.62589 | DEPLOYABLE, wave-2 reported |
| `RT-160` / `RT-190` (other branch) | logit-average blends | 0.62544 / 0.62368 | DEPLOYABLE, other branch |
| independent re-measure | my 7 OOF streams under wave-2's calibration code | **0.62524** | corroborates RT-150 to ~0.0007 |

Per-fold for `RT-100`: `0.62903 / 0.61061 / 0.62688 / 0.60747 / 0.60152`.

**Experimental resolution — memorise these.** fold-to-fold SD ≈ 0.0085, partition-draw
SD ≈ 0.0050, seed SD ≈ 0.0012. **Anything under +0.0005 is noise.** +0.005 is
potentially meaningful. +0.010 is material.

---

## 2. REPOSITORY STATE — the fork

```
                                    ┌── ea5e2f2 ── 23cb4da   research/multi-agent-2026
65b0e21 ── 46583f5 ── e98f5b9 ──────┤
   (wave 1)          (checkpoint)   └── 8e76ad2              research/wave2-2026   ← USE THIS
```

* **`research/wave2-2026`** (HEAD `8e76ad2`) — reproducibility gate, a full bitwise
  streaming engine (`src/sbr/stream/s_m0*.py`, all 7 modules, with parity tests),
  `src/sbr/production/` (model, calibration, submission entry points),
  `VALIDATION_V2.md`, `RDOF_LEDGER.md`, alternative fold partitions, negative
  controls, red-team generator audit, `PUBLIC_IDEA_MAP.md`. **This is the trunk.**
* **`research/multi-agent-2026`** (HEAD `23cb4da`) — the same wave-1 base plus an
  independent, *less complete* streaming port and the platform-constraints research.
  **Its `RT-150` means something completely different** (a rejected gating
  experiment, −0.0219). Its `RT-160`/`RT-190` are logit-average blends. Do not mix
  the ID spaces.
* Immutable checkpoint tag: `research-checkpoint-20260818-1` = `e98f5b9`.
* Both branches are pushed to origin.

**Unmerged value on the multi-agent branch** worth cherry-picking: the platform-constraints
research (§3), the deployability finding (a blend must be a per-series function),
and `research/reports/platform_constraints.md`.

---

## 3. PLATFORM CONSTRAINTS — verified from official docs, 2026-08-19

From [the competition docs](https://docs.crunchdao.com/competitions/competitions/structural-break-real-time)
and [the forum](https://forum.crunchdao.com/t/structural-break-real-time-may-infer-use-prior-completed-series-state/1186):

* **Metric confirmed:** `TS-AUC = Σ_t w(t)·AUC(t) / Σ_t w(t)`, `w(t) = n_pos(t)·n_neg(t)`.
  This is exactly what `sbr/metric.py` implements. **The largest open validation risk
  in the project is closed** — we are optimising the right objective.
* **Runtime: 15 hours per week.** Test set 10,000 public + 10,000 private series
  (~5.0M / ~10.1M scoring points). Allowed cost per point: 10.7 ms at parallelism 1,
  42.9 ms at P=4, 64.3 ms at P=6 (public set).
* **Parallelism** via a global `INFER_PARALLELISM = n`; RAM scales per process.
* **Determinism is a reward-eligibility condition**: re-run on 10 % of the data must
  match to 1e-8. Non-deterministic ⇒ ineligible for rewards.
* **Cross-series state is mechanically allowed but breaks determinism** under the
  platform's parallelism (organiser's own warning). This is why any blend must be a
  **fixed per-series function** of that series' own scores — a within-timestep rank
  across the live cross-section is both unavailable at inference and, if faked via
  accumulated state, a reward-eligibility risk.

---

## 4. PHASE-0 BLOCKERS (attempted, documented, unresolved)

**Official `crunch test`: NOT RUN — environment cannot reach Crunch.**
`crunch-cli` installs fine and the binary exists; `crunch ping` fails with
`api.hub.crunchdao.com … Tunnel connection failed: 403 Forbidden`. The user's Mac VM
also has no network. **Only the user's macOS environment can run it.**

**The submission artifact is missing.** `submissions/C_ensemble_deployable.ipynb`
is not in the repo (gitignored build artifact). Nor is the model directory, nor the
wave-2 OOF `.npy` vectors. Rebuilding requires re-running the 7 production boosters
(hours). Until then, RT-150's reported 4.57 ms/point and harness PASS are
**unverified from a clean checkout**.

**Hardware.** Cloud container: 2 vCPU, 7 GB RAM, ~14 GB free disk, **no GPU**.
User's Mac VM (via the file bridge): 4 cores, 3.9 GB RAM, **no network, no LightGBM/
scipy/sklearn/pyarrow/numba, and the repo `.venv` is a macOS venv unusable in it**.
⇒ **Alpha Teams 17 & 18 (deep temporal, self-supervised) are NOT RUN, not tested.**
Anything needing a GPU or >4 GB RAM must go to the user's macOS environment.

---

## 5. WAVE-3 WORK ACTUALLY COMPLETED

Only **one** of the 25 alpha teams was executed. Everything else is NOT RUN.

### W3-A1 — transformed detector bank (`m08_chan`) — **REJECTED**

Pre-registered in `RDOF_LEDGER.md` before running (1 configuration, 0 variants
selected on validation). CUSUM / Page-Hinkley / Shiryaev-Roberts plus peak,
time-since-peak, persistence and matched-length null calibration, on six channels:
`z²`, `|z|`, `z_t z_{t-1}`, `sgn·sgn`, `e²`, `e_t e_{t-1}`. 72 columns,
140 ms/series, **bitwise prefix-invariant on 10 series**.

| arm | mean OOF (ABL, 400k rows, 5 folds, same seed) |
|---|---|
| `RT-301` control, 7 modules | 0.61282 |
| `RT-302` + `m08_chan` | 0.61209 |
| **delta** | **−0.00073** |
| deployable ensemble delta | **+0.00023** |
| standalone | 0.61201 |

Per-fold delta `+0.0003, +0.0069, +0.0005, −0.0061, −0.0052` — sd 0.0047, **six
times the mean**. Both pre-registered falsification conditions met. The mechanism
was sound; the marginal information was already covered by `m01_seq` (recursive
memory) plus `m00_core`/`m04_resid`/`m07_bayes` (derived channels). Full write-up in
`research/FAILED_EXPERIMENTS.md`. **Do not retry as written** — only the two
genuinely novel channels (`sgn·sgn`, `e_t e_{t-1}`) are worth isolating.

### Deliverable 3 — `PUBLIC_IDEA_MAP.md` corrected

Wave 2 asserted no public 2026 real-time solution *exists*. That overstates the
evidence. Corrected to a four-state vocabulary — VERIFIED / ATTESTED / FAILED TO
RE-ACCESS / NOT FOUND — and the two 2026 implementations attested by
`research/deep_research_report.md` (the 0.518→0.533→0.557→0.575 residual/AR ladder,
and the 648→200-feature LightGBM at ~0.579) are restored as **ATTESTED**, not
erased. **New and unexploited:** two *2025* solution repos wave 2 believed
unreachable are publicly indexed —
`aParsecFromFuture/ADIA-Lab-Structural-Break-Challenge-Solution` (2nd place) and
`StefanConstantin707/adia-lab-structural-break-challenge`. Nobody has read them.

---

## 6. WHAT IS *NOT* DONE — the honest list

**Not run at all:** AR(p)-FOCuS · localisation v2 · backward CUSUM / late-break
specialist · transient-vs-permanent · robust BOCPD · e-processes/martingales ·
dependence v2 · distribution/PIT v2 · spectral/wavelet v2 · 2025→2026 transfer ·
real hard negatives · synthetic augmentation · likelihood-ratio learning ·
ranking v2 · multi-task supervision · deep temporal (no GPU) · self-supervised
(no GPU) · automated interaction discovery · alternative tabular learners ·
distillation · adversarial/robustness slices · calibration v2 · temporal
specialists · break-type specialists · Optuna hyperparameter search · row-sampling
research · survival/hazard formulation · Bayesian posterior over tau · null
calibration v2 · **all six red teams** · `STATE_OF_RESEARCH_V3.md`.

---

## 7. HARD-WON TRAPS — every one of these actually happened

* **Screen→full reversals.** Three of three screen-store *architectural* wins
  reversed at full scale (context block +0.0009→−0.0177; pairwise objective
  +0.0072→tie; DGP gating +0.032→−0.0219). Every *feature* addition transferred.
  **The screen store is triage for features and is NOT evidence for objectives,
  architectures, or anything that repartitions training data.**
* **One fold is not a result.** I promoted a "pairwise objective fails, −0.0033"
  conclusion from a single well-paired fold; over five folds it was a dead heat, and
  the same objective moved 0.008 on the *same fold* between two runs. Per-fold spread
  is 0.011.
* **The deployability trap.** A blend of within-timestep ranks scored beautifully and
  **could never have been submitted** — the runner is series-sequential and
  single-pass. Check what the inference interface can observe *before* optimising
  anything that assumes more.
* **Silent `str.replace` no-ops.** Two optimisation edits and (per wave 2) one file
  rewrite silently did nothing; the code kept running the old path and I "measured"
  a speedup that wasn't there. **Assert your edit landed.**
* **`pkill -f <pattern>` kills your own shell** if the pattern appears in your command line.
* **OOM at ~6 GB.** Materialising train + valid matrices plus a copy blows the box.
  Keep to ≤700k rows × ~570 cols, and never `column_stack` a second full copy.
* **Causality checks are not sanity checks.** A column exploding to ±3e7 passed
  prefix invariance; only a per-column distribution audit caught it.
* **Supervision is the real cost.** 60.5M effective tokens this session: 42 % the
  ten research subagents, 26 % re-reading context, 21 % running experiments. Compute
  is nearly free; *watching* it is not. **Queue jobs and check back rarely.**

---

## 8. RECOMMENDED NEXT ACTIONS, IN ORDER

1. **Rebuild and run the official `crunch test` on the user's Mac.** It is the only
   unverified link in the deployment chain and the only machine that can do it.
   `python3 research/scripts/build_submission.py` then `crunch test`.
2. **Commit the built notebook and model artifacts** (or a checksum manifest) so the
   claim is reproducible. Right now the production system exists only as a report.
3. **Read the two 2025 repos** and translate each idea into its causal 2026 form.
   Cheapest genuine novelty available; nobody has looked.
4. **Then Tier-1 alpha**, each with a matched control and a pre-registration:
   backward CUSUM / late-break, localisation v2, transient-vs-permanent, robust
   BOCPD, hard-negative sampling. ~1.3 h per fully-controlled experiment on 2 cores.
5. **Do heavy compute on the user's macOS environment, not in a container** — more
   cores, more RAM, and it is the only place `crunch test` works.

**If you must submit today:** the safest artifact is `RT-100` (single booster,
reproduced exactly from a clean checkout, 0.61510 OOF / 0.60791 lockbox). The
highest-alpha is `RT-150`, contingent on someone actually building and testing its
notebook.

---

## 9. GROUND RULES THAT ARE NOT NEGOTIABLE

Read `research/VALIDATION_V2.md` and obey it. Never use `X_test.reduced.parquet` for
selection. The original lockbox is **spent** — two inspections — so there is no
pristine holdout left; protection now comes from nested CV, alternate partitions,
seed stability, paired comparisons, negative controls and the RDOF ledger. Label
every number DISCOVERY, CONFIRMATION or DEPLOYABLE. No row-wise splits, no
cross-series state, no `n_online`, no future observations, never regenerate the
canonical folds. Record every experiment's degrees of freedom *before* running it.
