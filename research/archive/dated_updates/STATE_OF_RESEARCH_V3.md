# STATE OF RESEARCH — V3
**2026 ADIA Lab / CrunchDAO Structural Break Challenge (Real-Time Edition)**
Written 2026-08-20. Deadline **17 September 2026**. Objective: true out-of-sample
Time-Stratified AUC.

This document supersedes nothing. `VALIDATION_V2.md` is still binding,
`STATE_OF_RESEARCH_V2.md` is still the wave-2 record, and where this file and
the repository disagree, **the repository wins**.

---

## 0. THE HEADLINE

Wave 3 ran one alpha experiment to completion with a matched control, a
pre-registration, a paired bootstrap, a mechanism diagnostic and a negative
control. **It did not promote.** The score to beat is unchanged.

What wave 3 *did* buy is worth more than one feature module would have been:

1. **The research machine now runs on the user's Mac**, reproduces the champion,
   and is deterministic. Before this wave the entire `sbr` stack existed only as
   code — no store, no feature cache, no OOF vectors, and every path hardcoded
   to a container that no longer exists.
2. **A methodological hole was found and closed.** Ensemble deltas and
   within-timestep rank correlations — the two instruments wave 2 used to justify
   its seven-stream champion — were never controlled against a stream with no new
   information. When that control is run, it beats the new module. See section 3.

---

## 1. SCOREBOARD — wave 3 additions only

All wave-3 rows carry a real `git_sha` (`8e76ad2`), unlike the 66 `nogit` rows of
wave 1.

| id | what | mean OOF | label |
|---|---|---|---|
| `RT-300` | `RT-100` config, CHAMP protocol, macOS/arm64 | **0.61605** | **CONFIRMATION** of the environment (ledger: 0.615103, delta +0.00095) |
| `RT-301` | ABL control, 7 modules, seed 0 | 0.61257 | control (wave-2 `RT-200`: 0.61282) |
| `RT-302` | `RT-301` + `m09_back` (51 new columns) | 0.61413 | **NOT PROMOTED** |
| `RT-303` | seed-clone control, seed 1 | 0.61488 | negative control |

Per-fold `RT-300`: 0.62903 / 0.61061 / 0.62688 / **0.61223** / 0.60152.
Per-fold `RT-100`: 0.62903 / 0.61061 / 0.62688 / **0.60747** / 0.60152.
Four folds identical to the last printed digit; fold 3 moves +0.00476.

**The deployable champion is unchanged**: the wave-2 seven-stream ensemble with a
frozen cross-fitted time-conditional CDF, **0.62589** — recorded in
`STATE_OF_RESEARCH_V2.md` section B1.5 and `research/reports/deployable_ensemble_v2.json`,
but see section 5 for what is wrong with how that number is filed.

---

## 2. WHAT WAS BUILT, AND WHAT IT COST

| artifact | state |
|---|---|
| 10,000-series store | verified **bitwise against the raw parquet** on 6 random series; break rate 0.4967 and 5,036,517 online rows both match the protocol |
| feature cache, 7 production modules, 500 columns | built, 10 GB, 293 s on 6 workers |
| `m09_back` cache, 51 columns | built, 120 s |
| all 8 modules | re-verified **bitwise prefix-invariant**, atol=0.0 |
| OOF vectors `RT-300..303` | on disk, gitignored by repo policy |

Portability changes to shared code, **no research semantics altered**:
`SBR_ROOT` / `SBR_FEATURES` / `SBR_STORE` environment overrides for the
hardcoded `/home/claude/sb` paths in `pipeline.py`, `features/driver.py` and
`wave2_lib.py` (container defaults preserved as fallbacks), and a `load_all()`
call inside the feature-driver worker because macOS multiprocessing uses `spawn`
rather than `fork` and workers were starting with an empty registry.

`research/scripts/bg.sh` launches any job detached under `caffeinate -ims -w`, so
long runs survive the machine idling.

---

## 3. THE FINDING THAT MATTERS MOST

`RT-302` beat its control by +0.00156 and its deployable logit blend beat the
control by +0.00371 — which under `VALIDATION_V2.md` section 7 is a promotion
route. So the route was tested with a control that should have been run in wave 2
and never was:

| deployable logit blend | mean | delta vs control |
|---|---|---|
| control + `m09_back` (51 new columns of new statistics) | 0.61627 | +0.00371 |
| **control + a seed clone of itself (zero new information)** | **0.61753** | **+0.00496** |

And on diversity, measured the way wave 2 measured it:

| stream | within-timestep rank correlation with the control |
|---|---|
| `RT-302`, 51 new columns | 0.8219 |
| `RT-303`, same features, different seed | **0.7846** |

**Changing the random seed decorrelated more than adding a new feature family,
and blended better.** Two consequences, both now binding in `RDOF_LEDGER.md`:

* No stream may be promoted on a blend delta that has not been measured against a
  seed-clone control of the same standalone strength.
* Low within-timestep rank correlation with the champion is not a diversity
  credential. It is what a different bagging draw produces for free.

Wave 2's seven-stream ensemble (`RT-131` / the 0.62589 champion) was justified on
exactly these two instruments — "within-timestep rank correlations 0.40–0.69" and
a +0.0088 ensemble delta — and was never given this control. **That does not make
the champion wrong**: 0.40–0.69 is far below 0.78, and seven streams differing in
features, depth, row sampling and boosting type are not seed clones. But the
*size* of its diversity claim is unaudited, and a seven-way seed-clone ensemble is
now the obvious missing baseline. It is the first item in section 6.

---

## 4. TIER-1 ITEM 1 — CLOSED, AND WHY

`m09_back` was the brief's backward-CUSUM / late-break specialist. Full write-up
in `FAILED_EXPERIMENTS.md`. The one-line version: **it makes young breaks worse
and mature breaks better**, which is the opposite of its design.

| post-break age | control | + m09_back | delta |
|---|---|---|---|
| 0–5 | 0.51278 | 0.51186 | −0.00092 |
| 5–10 | 0.52594 | 0.52560 | −0.00035 |
| 10–20 | 0.54279 | 0.54063 | −0.00215 |
| 20–50 | 0.56866 | 0.56564 | −0.00302 |
| 50–100 | 0.59150 | 0.59252 | +0.00102 |
| 100+ | 0.64569 | 0.64917 | +0.00349 |

The mechanism failure is instructive for the rest of Tier 1: for a *young* break
the suffix is short, so a suffix-vs-prefix contrast is dominated by prefix noise
and is a **worse** instrument than a one-sample test against a 3,000-point
history. The pre-break online segment only becomes the better reference once the
break is old — by which time the problem is already solved. **Any Tier-1 idea
that estimates something from the post-break segment inherits this
problem**, localisation v2 included: at age 0–5 there is nothing to estimate
from. The honest read is that the 0–20 age band may not be reachable by better
statistics at all, and the metric mass there (76k of 1.03M positive rows) is
small enough that it is not where the remaining alpha is.

Note also the base rates: TS-AUC at age 0–5 is 0.513 and at 100+ it is 0.646.
Nearly all discriminating power in this problem comes from mature breaks.

---

## 5. TWO BOOKKEEPING DEFECTS IN THE INHERITED RECORD

Both were found by cross-checking the wave-3 brief against the repository, and
both are stated in `FAILED_EXPERIMENTS.md` and `RDOF_LEDGER.md`.

1. **W3-A1 / `m08_chan` cannot be verified.** The brief and handoff describe a
   completed, rejected experiment with numbers, a module "on disk" and a write-up
   "in `FAILED_EXPERIMENTS.md`". None of it exists in the working tree or in any
   commit of any branch, and `RT-30x` appears nowhere in `research/`. Its IDs were
   therefore treated as unallocated and reused. Its derived advice ("only
   `sgn·sgn` and `e_t e_{t-1}` are worth isolating") is not inherited.
2. **The 0.62589 champion has no ledger row and no experiment ID.** It is real —
   `STATE_OF_RESEARCH_V2.md` section B1.5 and `deployable_ensemble_v2.json` both
   carry it — but it is absent from `RESULTS.csv`. The name "RT-150" for it was
   coined in the handoff; in `RESULTS.csv` on **both** branches `RT-150_f1` is the
   *rejected* DGP-gating run (0.60619). The handoff's warning that the two
   branches use "RT-150" for different things is backwards: that row is identical
   on both. The real divergence is `RT-160`/`RT-190`, which exist only on
   `research/multi-agent-2026`.
   Related: the handoff's corroborating "independent re-measure, 0.62524" is
   `RT-131`'s pooled OOF — the *within-timestep rank average*, which the same
   documents label ILLEGAL and non-deployable (`deployable_blend.json` calls that
   exact value `rank_average_not_deployable`). The genuine independent deployable
   corroborator is `RT-160` = **0.62544**. The conclusion survives; the cited row
   was the wrong one.

---

## 6. RANKED NEXT ACTIONS

1. **Seven-way seed-clone ensemble baseline.** Blend seven seed-clones of the
   champion under the same deployable calibration as the 0.62589 system. If it
   lands near 0.625, the seven-stream architecture is buying little over
   ensembling and the project's headline gain needs restating. This is one script
   and ~2 hours, and it is the highest-information experiment available.
2. **Give the champion a ledger row**, with its provenance, and rebuild its OOF
   vectors so any future candidate can be measured against it. Right now no
   wave-3 candidate can be blended with the actual champion, only with `RT-300`.
3. **Run the official `crunch test` on this Mac** — still the one unverified link
   in the deployment chain, and this machine is the only one that can reach the
   API. Codex owns the notebook; the environment is now proven here.
4. **Transient vs permanent (Tier-1 item 2)**, which does not depend on the
   post-break segment being long and is therefore not subject to the section-4
   failure mode.
5. **The `_pre` family as 6 columns**, not 51, against a control that already has
   `m00_core`'s expanding block — the only part of `m09_back` that earned its gain.
6. **Hard-negative row sampling (Tier-1 item 5)** and **dependence v2 with a
   likelihood framing (item 6)**, in that order.
7. **Read the two 2025 solution repos.** Still nobody has. Cheapest genuine
   novelty available, and Codex's translation map is the input.

**Not attempted, and honestly marked NOT RUN:** deep temporal and self-supervised
models (no GPU), AR(p)-FOCuS, e-processes, robust BOCPD, distribution/PIT v2,
break-type and temporal specialists, alternative tabular learners, Optuna,
survival/hazard formulation, and all six red teams.

---

## 7. WHAT A READER SHOULD DISTRUST IN THIS DOCUMENT

The `RT-302` numbers are a single ABL protocol at one seed on the canonical
partition. They were not run on alternative partitions and not seed-swept,
because the candidate was rejected and spending four more runs on a rejected
candidate is not a good use of the deadline. If someone wants to revive the
`_pre` family, those runs have to happen.

The environment-reproduction claim rests on four folds matching exactly and one
moving +0.00476. That is strong evidence the pipeline is identical and weak
evidence that *every* number transfers; a fold-level divergence of that size on a
different fold could change a marginal delta's sign. Wave-3 deltas are all
computed against wave-3 controls run on this machine, which is what makes them
safe — never compare a wave-3 arm to a wave-2 number directly.
