# WAVE 5 — ALPHA CHARTER

**Opened 2026-08-22. Branch `research/wave5-alpha`.**

`VALIDATION_V2.md` remains binding. Where this file and the repository disagree,
**the repository wins.**

---

## 1. PROVENANCE

| | |
|---|---|
| new branch | `research/wave5-alpha` |
| parent branch | `research/wave3-integration` — the scientific trunk |
| parent SHA | `17bb5dfe651738f46766d27a77c14db2c476a5a1` |
| parent commit | "STATE_OF_RESEARCH_V4: the champion survived its control, and it changed" |
| read-only reference | `claude/rt600-baseline-submission` @ `9aaa9b046609eb94d9fdd09f7ae07d9249fe2f65` |
| read-only reference | `codex/reproduce-2025-public-solution` @ `422e4b2db989051c39d517f03a947ae475f36a5c` |
| machine | Linux x86_64, 4 vCPU, 15 GB RAM, **no GPU** |
| environment | Python 3.12.3; research venv `.venv-wave5` |
| research deps | `research/requirements-wave5.txt` |

**Why not branch from the release branch.** `claude/rt600-baseline-submission`
carries packaging fixes and submission history — a read-only source of runtime
evidence, not a development trunk. It was read for the LB-001..LB-004 record and
nothing was taken from it into this branch.

**Three side branches are not contained in RT-600's history** and hold unmerged
work: `codex/reproduce-2025-public-solution` (2 commits),
`codex/oracle-information-frontier-2026` (1), `codex/wave3-engineering` (1).
None was merged here.

---

## 2. RT-600 IS AN IMMUTABLE FALLBACK CHAMPION

External result: **Crunch TS-AUC = 0.6268**.

Not to be modified, retrained, improved in place, rewritten, or tuned against its
own public score. Frozen by identity, so that any drift is detectable:

| artifact | sha256 |
|---|---|
| source zip | `199db8c9f5db7e1429f6ae09fa018d43cc58c5eb47b5c623a017799537a413a0` |
| model zip | `6c8960ddc7331afecfc6980974f2879401a1eb583f15f5cad34c2ea03299ea8c` |
| feature manifest | `1646c3b9e09d8a7fb3c564483a1d1d999680caeefe848b092f907a6cac80cced` |
| model manifest | `1483a59a268ded18b6af804a5f0eb0b191376a1dc86526f286751bf1613ec940` |

The rule from the release record still reads the situation correctly: *if the
source or model zip hash moves, the model changed; if only the entrypoint and
notebook move, the packaging changed.*

---

## 3. IMMUTABLE EXTERNAL FACTS

Recorded as given. These are observations, not estimates, and Wave 5 accepts them
as premises.

**A.** RT-600 public Crunch TS-AUC = **0.6268**.

**B.** The same RT-600 architecture scored **≈0.62581** under comparable internal
validation (`RT-420`, seven specialists, SCDF, five canonical folds).

**C.** Internal → external transfer was therefore **≈ +0.0010**. **No large
collapse was observed.**

**D.** A **P=4** RT-600 cloud submission ran successfully on **16 vCPU / 64 GB**
and produced the same predictions and the same 0.6268.

**E.** Cloud P=4 succeeded although local macOS P=4 segfaulted. The segfault was
**platform-specific**, not architectural.

**F.** The successful cloud run took **≈2 h 11 m** against a 15 h budget.
**Production runtime is not the bottleneck. Wave 5 does not optimise it.**

### 3.1 What fact C actually costs us — and it is not nothing

`STATE_OF_RESEARCH_V4.md` §18 and `FINAL_ARCHITECTURE_FREEZE.md` §4 both
pre-registered an expected external performance, and both were **wrong in the
same direction**:

| scenario | predicted | actual |
|---|---|---|
| optimistic | ~0.625 | |
| **base (the stated expectation)** | **~0.615** | |
| conservative | ~0.605 | |
| **observed** | | **0.6268** |

The outcome landed **above the optimistic case**. The reasoning behind the base
case — "the single-model level moves 0.0092 across internal partitions alone, so
an external draw should move at least as much" — did not describe reality.

The honest reading, stated before any Wave-5 experiment leans on it:

* The claim "**the delta is the durable asset, the level is not**" is **partly
  falsified**. The level transferred, essentially intact, across an external draw.
* This is **one observation**. One leaderboard point cannot establish a transfer
  law, and reading +0.0010 as "internal gains transfer 1:1" would repeat exactly
  the error the base case made in the other direction.
* What it does license: **the internal validation framework is directionally
  useful and its numbers are not fiction.** An internal +0.010 is worth chasing.
* What it does not license: **using 0.6268, or any future leaderboard number, to
  select a parameter.** That prohibition is unchanged and absolute.

---

## 4. VALIDATION RULE FOR WAVE 5

Unchanged from `VALIDATION_V2.md`, restated because Wave 5 introduces new model
families and the temptation to relax it scales with novelty.

**Selection runs on:** the canonical 8,000-series development set, the canonical
folds, the exact TS-AUC scorer, and alternate partitions (`alt1/alt2/alt3`) when
promoting a candidate.

**Selection may NOT use:**

* `RT-500`..`RT-506` all-10k OOF vectors — generated **after** the freeze for
  deployment calibration. They are not a research validation set.
* the former lockbox — spent.
* `X_test.reduced` / `y_test.reduced` / private labels — never read.
* **the public leaderboard score as a parameter-selection objective.**

**Promotion bars, inherited and still binding:**

| stage | bar |
|---|---|
| screen (features only) | +0.002 vs its own paired control |
| full dev (discovery) | positive on a majority of folds |
| promotion | paired bootstrap CI materially favourable, **or** a deployable ensemble delta |
| architectural change | the above, plus stability across alt partitions and seeds |

**And the one Wave 3 bought with a failed experiment — non-negotiable:**

> Any new stream must beat a **seed-clone control**. A deployable ensemble delta
> is not evidence unless it beats a same-strength stream known to contain nothing
> new. Low within-timestep rank correlation is **not** a diversity credential: a
> seed change decorrelates *more* (0.7846) than 51 columns of new statistics did
> (0.8219).

`m09_back` and the W4-E6 13-model union both died on this bar. A new model family
— XGBoost, CatBoost, a distilled teacher — is subject to it identically. **A
different library is not automatically new information.**

---

## 5. WHAT WAVE 5 INHERITS AS SETTLED

Do not re-run these. Full detail in `FAILED_EXPERIMENTS.md`.

| result | status |
|---|---|
| seed bagging is worth **+0.0050** (57% of the ensemble gain) | established |
| specialist diversity is worth **+0.0033** on top | established, all 4 partitions |
| `m09_back` backward CUSUM | REJECTED — failed its seed-clone control |
| 13-model equal-weight union (W4-E6) | REJECTED — −0.00095; over-weights `RT-100R` 7× |
| ensemble weighting / subset selection | REJECTED — honest selection *loses* 0.0008 |
| DGP-cluster gated specialists | REJECTED at full scale (−0.0219) |
| `m05_ctx` context block as raw features | REJECTED — no substrate; permutation control caught it |
| GARCH normalisation, trend/slope family, tail-asymmetry, rank-CUSUM | REJECTED, each with a mechanism |
| pairwise-t objective | equivalent within noise; kept as a stream |
| `m08_chan` | **unsourced** — exists in no commit; not inherited |

**The pattern worth carrying into a wave that adds model families:** three
screen-level wins reversed at full scale (context block, pairwise objective, DGP
gating), all the same failure mode — they help a *data-starved* model and stop
helping once it is not. **The screen store may triage features. It may not
promote objectives, architectures, or model families.** Those are tested at full
scale from the start.

---

## 6. THE OPPORTUNITY — STATED AS THE RECORD STATES IT

`STATE_OF_RESEARCH_V4.md` §12 names eight alpha families as **NOT RUN**, and calls
this "the largest remaining opportunity and the honest gap in this work":

transient vs permanent · hard-negative sampling · dependence likelihood v2 ·
robust distribution distances · residual CUSUMSQ / AR regression break ·
robust BOCPD · AR(p)-FOCuS · TabPFN / deep / teacher

Wave 4 spent its compute auditing the champion instead. That audit was worth it —
it is why we know bagging is 57% of the gain — but it means **no new alpha family
has been tested since wave 3**, and the one that was (`m09_back`) failed.

Wave 5's mandate is these eight plus the broadened model family, under the bars in
§4.

---

## 7. PRODUCTION / RESEARCH SEPARATION

Root `requirements.txt` **is** the cloud runner environment: `crunch push` uploads
it and the runner builds from it. LB-002 died in `infer()` because lightgbm was
declared only in a file the runner does not read.

**It is not modified in this wave.** Research dependencies live in
`research/requirements-wave5.txt` and install into a separate venv. Production
dependencies change only when a model that needs them is actually promoted toward
submission, and then through the measured-import-closure procedure that
`tests/test_requirements_cover_runtime.py` enforces.

Capability audit of the new families: `research/PACKAGE_CAPABILITY_MATRIX.md`.
Raw evidence: `research/reports/wave5_package_probe.json`.

---

## 8. THE BINDING CONSTRAINT ON THIS SESSION — STATED FIRST, NOT BURIED

**This container holds no competition data of either year.**

| asset | state |
|---|---|
| 2026 feature store (`/home/claude/sb/cache/store`, `SBR_STORE`) | **absent** |
| 2026 raw competition parquet | **absent** |
| 2025 competition parquet | **absent** (the 2025 audit read it from the user's Mac) |
| `data/` in this repo | 3 synthetic demo files, ~16 KB |

Consequence: **no training run, no OOF vector, no TS-AUC, and no fold score can be
produced here.** Everything in this wave so far is environment and capability
work. No alpha claim is made, and none may be inferred from the fact that new
libraries now import.

Corroborating detail: 17 of the 19 failing tests in the suite fail with
`FileNotFoundError` on the missing store, which is the same constraint seen from
the other side.

**To start Wave-5 experiments, one of these must happen:** mount or rebuild the
2026 feature store in this environment; or run the experiments where the store
lives. For the TabPFN rungs specifically, a checkpoint must additionally be
sideloaded — `huggingface.co` is blocked here and the 8.4.0 default repo is gated.

---

## 9. DEGREES OF FREEDOM SPENT IN WAVE 5 SO FAR

| | |
|---|---|
| training runs | **0** |
| candidates scored | **0** |
| TS-AUC numbers produced | **0** |
| hypotheses tested | **0** |
| packages audited | 8 |
| leaderboard submissions | **0** |

Nothing has been selected, so nothing is owed a multiplicity discount. The ledger
opens clean.
