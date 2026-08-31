# RT-1320 Promotion Plan — Draft Preregistration

Status: **DRAFT, not preregistered.** Nothing in this file binds until it is
committed and its endpoints are frozen. No result described here has been run.

Subject: **RT-1320 / M1 Arm-C residual student** — the first candidate in the
program that *adds* an eighth member to RT-1257 rather than swapping a slot.

Governing protocol: `PROTOCOL_CHAMPION_2026.md`. Status vocabulary:
`MODEL_REGISTRY.md`.

---

## 1. Where the candidate actually stands

Under the addition contract (E1 = RT-1257 + one matched added seed clone
`RT-403`), from
`reports/armc_residual_student_confirm_s20260901/E2_E1_addition_contract.json`:

| quantity | value |
|---|---:|
| E0 (RT-1257) | 0.627838 |
| E1 (RT-1257 + RT-403 clone) | 0.627738 |
| E2 (RT-1257 + student) | 0.629254 |
| **Primary E2−E1** | **+0.001516** |
| primary folds positive | **4/5** (fold 0 = −0.001363) |
| Secondary E2−E0 | +0.001416 (4/5, fold 0 = −0.000150) |
| control lift E1−E0 | −0.000100 |
| paired-bootstrap noise floor | 0.0011 |

Reproduced at a second seed within ~1% of the noise floor.

**Honest reading.** The primary endpoint clears the declared noise floor by
about 38%, on four of five folds, with a well-behaved (flat) E1 control — which
is exactly the failure mode RT-1264/CSA-04 was killed for, and this candidate
does not have it. That is a real result. But the margin is thin, the gain is
concentrated in folds 1 and 3, and fold 0 is negative on **both** endpoints.
The widely quoted 5/5 figure belongs to the RT-600 lane (+0.001965), not to the
champion lane, and must not be used to describe this candidate.

**Scope of the gain.** From `reports/armc_residual_student/`: this is a
**never-break-cut** gain. Never-break pair net vs RT-600 is +0.003409; pre-break
pair net is +0.000363, i.e. approximately zero. The T-orthogonal residual scores
*below* Arm B on the pre-break cut. Do not plan, report, or promote this as a
broad improvement — it is a null-model / false-positive-repair mechanism.

---

## 2. Ordering principle: cheapest kill first

`MODEL_REGISTRY.md` lists five outstanding items. Executed in listed order, the
program would spend production-engineering effort on a candidate whose primary
scientific risk has not yet been tested. This plan reorders them by
**probability of killing the candidate, divided by cost**:

1. **Alternate-partition leg** — most likely to kill, partially pre-paid. First.
2. **Causality gate** — cheap, mostly inherited, but a hard protocol gate; and
   until it passes, no predictive score above may be *interpreted* at all.
3. **Final-10k teacher target regeneration** — expensive, and pointless if (1)
   fails.
4. **Production artifact + manifest** — small, mechanical, last.
5. **Crunch test + external score** — gated on RT-1257's own promotion, which is
   an independent workstream.

Phases 1 and 2 are the decision. Phases 3–5 are execution of a decision already
made.

---

## 3. Phase 0 — Preconditions (no compute)

Before any fit, establish the ground truth this plan assumes.

**0.1 Confirm the alt-partition inventory.** Verified present in
`../structural-break-wave8/research/oof/`, canonical + `.alt1/.alt2/.alt3`:

- specialists `RT-300, RT-410, RT-411, RT-412, RT-413, RT-414, RT-415`
- seed clones `RT-401`, `RT-403`

Verified **missing** on alt partitions:

- `RT-991` (Arm-C teacher) — canonical only
- `nested_Q_outer{f}_inner{g}` (20 files) — canonical only
- `RT-1254` (CAT-413), `RT-1255` (CAT-300) — canonical only, in
  `../structural-break-deep-ensemble-frontier-local-2026/research/oof/`

**This corrects `MODEL_REGISTRY.md` item 1**, which states the seven specialists
must be refit under `folds_alt*`. They already are. Update that line when this
plan is committed.

**0.2 Confirm the alt vectors are trustworthy. — DONE 2026-08-31, PASS.**
Recomputed W4-E2 from the 52 vectors on disk using
`wave4_partition_analyse.py` unchanged, and diffed against the committed
`reports/ensemble_partition_stability.json`. **Worst absolute discrepancy across
all 24 levels and deltas: `0.000e+00`.** All 52 SHA-256 values distinct, so no
alt vector is a stale copy of its canonical counterpart. Summary reproduces the
freeze exactly (total +0.00832 / sd 0.00113; specialisation +0.00333; single
level spread sd 0.00394; 12/12 deltas positive).

The vectors are **not stale**; Phase 1 does not need to refit the seven
specialists or the seed clones. It also re-confirms from the vectors themselves
the fact that motivates Phase 1: canonical is the most favourable of the four
partitions on all three deltas, and alt1's specialisation delta (+0.00254) is
below W4-E1's own +0.0030 bar.

Caveat carried forward: this is **integrity, not provenance** — identical bytes,
not proof those bytes came from the claimed configurations. Scope is the RT-600
lane only; `RT-991`, `nested_Q_*`, `RT-1254` and `RT-1255` still have no alt
vectors and remain the real cost of Phase 1.

Full record: `reports/rt1320_promotion/PHASE0_ALT_REVERIFICATION.md`.

**0.3 Plumbing delta.** `wave5_lib.Ctx` already accepts `folds_file=`;
`armc_residual_student.py` already accepts `--folds-path`. Only
`wave7_teacher_nested.py` lacks a partition flag. One CLI argument threaded to
`Ctx`, plus partition-suffixed output names — not a redesign.

**0.4 Code is self-contained on this branch; data is not.** Verified: every
script this plan needs is tracked on `main` under `research/scripts/` —
`wave7_teacher_nested.py`, `wave5_lib.py`, `wave4_partitions.py`,
`armc_residual_student.py`, `armc_e2_e1_addition_contract.py`,
`armc_residualization.py`, `verify_causality_artifacts.py` — and all four
partitions are tracked under `research/folds/` (`folds.parquet`,
`folds_alt1/2/3.parquet`, `folds_final10k.parquet`).

The **OOF vectors are not**. They are `*.npy`, gitignored, and live outside this
worktree:

| what | where |
|---|---|
| specialists + seed clones, canonical and `.alt1/2/3` | `../structural-break-wave8/research/oof/` |
| `RT-991`, `nested_Q_*` (canonical only) | `../structural-break-wave8/research/oof/` |
| `RT-1254`, `RT-1255` (canonical only) | `../structural-break-deep-ensemble-frontier-local-2026/research/oof/` |
| RT-1320 student OOF + per-fold checkpoints | `../structural-break-multi-agent-frontier-20260829/research/reports/armc_residual_student/` |
| 500-column feature bank | `../structural-break-wave8/cache/features/` (symlinks into wave3/wave5) |

This branch therefore cannot reproduce anything on its own. Record the resolved
absolute paths and the SHA-256 of each input vector in every report this plan
produces, per `REPRODUCIBILITY_MANIFEST.json` discipline — the sibling worktrees
are mutable and unversioned, which is exactly the condition that voided
RT-1237/1238/1239.

**0.5 Measure one fit before committing to sixty. — DONE 2026-08-31.**
Measured at 250k rows x 60 rounds on the frozen `ARM_B_PARAMS`: **0.640 s/round**,
peak RSS 4.22 GB. Extrapolated to the real 1,000,000 x 900 fit:

| | |
|---|---:|
| one inner teacher | **39.5 min** |
| 20 inner teachers, one partition | **13.2 h** |
| 60 inner teachers, three alt partitions | **39.5 h** |
| feature matrix / peak during concatenate | 3.73 GB / **~7.45 GB** |

Two consequences Phase 1 must absorb:

- **The alt leg is 40+ hours of compute, not an afternoon** — and that is inner
  teachers alone, excluding student fits and CatBoost refits. Open decision 4
  below is therefore a **blocker**, not a convenience.
- **The RT-600-lane fallback does not avoid this cost.** It avoids the CatBoost
  refits only. The teacher refits are common to both lanes, because the student's
  target derives from the nested teacher on whichever partition it is fitted.

**Memory is the binding constraint.** ~7.45 GB peak on a 16 GB machine whose swap
is already at 5.56/7.17 GB. The probe ran at 250k rows precisely to avoid
thrashing. `num_threads=2` leaves ~5x on the table on a 10-core box, but raising
it is a protocol change that must first be shown not to move predictions, per the
determinism gate — not a free win.

**Hazard found and not yet fixed:** `cmd_inner_teacher` calls `np.save` on
`nested_Q_outer{f}_inner{g}.npy` with no existence check and no `--force`. With
`research/oof` symlinked to a sibling worktree, invoking it would silently
overwrite the existing canonical vector, which is gitignored and unrecoverable.
**Add a refuse-if-exists guard before Phase 1's first production invocation.**
The probe avoided `cmd_inner_teacher` entirely and wrote nothing; verified.

Full record: `reports/rt1320_promotion/PHASE0_TIMING_PROBE.md`.

---

**0.6 Close the Crunch cloud gaps before booking the run.** §0.5 made venue a
blocker; the venue is settled in principle (see open decision 4) but three
things must be checked first, and all three are cheap.

**0.6.1 CPU core count — the one that changes the arithmetic.**
`reports/gpu_tabular_2026/FROZEN_GPU_CONFIG.json` records GPU, VRAM, OS, python,
torch and peak RAM for the RT-1258/RT-1259 cloud runs, but **no CPU core count**.
The teacher is a CPU LightGBM fit at `num_threads=2`; the RTX 4090 is irrelevant
to it. Core count therefore decides whether the 13.2 h per partition is real or
whether a thread increase makes it materially cheaper. Record it on the next
cloud job — it is one line of `os.cpu_count()` in a run that is happening anyway.

Note the dependency: a `num_threads` change is a **protocol change**, gated on a
determinism demonstration (§0.5). Cores make it *worth* doing; they do not make
it *allowed*.

**0.6.2 Pin provenance before the run, not after.** The repo already records
that the Crunch-uploaded code tree **had no `.git` directory**, so the GPU
benchmark's reported `git_sha` came from a stale embedded fallback pointing at a
commit that does not contain the benchmarked file —
`FROZEN_GPU_CONFIG.json` carries two long notes untangling it. That was
survivable for a diagnostic. Phase 1 feeds a promotion decision, so the run must
carry an explicitly injected commit SHA, the folds SHA-256, and the feature-bank
manifest hash as data, and assert them on arrival. Reconstructing provenance
afterwards is what `_check_provenance` exists to prevent.

**0.6.3 Confirm the feature bank rebuilds in-cloud.**
`feature_preparation_benchmark.feature_build_seconds = 272` says the cloud job
built the 500-column bank itself in ~4.5 min rather than needing a ~10 GB upload.
Confirm that path still works for the modules the teacher needs (`FULL` =
m00–m04, m06, m07) and that it reproduces the tracked feature-manifest SHA. If
it does, data access is a non-issue: `folds_alt*.parquet` are tracked in git and
travel with the code.

---
## 4. Phase 1 — Alternate-partition leg (the decision)

**Why this is the gate.** `FINAL_ARCHITECTURE_FREEZE.md` records that the
canonical partition was **the most favourable of the four**, and that on alt1 the
RT-600 specialisation delta (+0.00254) would not have cleared W4-E1's own bar.
Measured partition SD is 0.0039 at single-model level. RT-1320's entire evidence
base is one draw from the most flattering partition, and its margin over the
noise floor is +0.0004. This is the single most likely place for the candidate
to die, and finding that out is much cheaper than the alternative.

**Execution is staged: alt1 first, then a decision, then maybe alt2/alt3.**
§0.5 measured the leg at 39.5 h of inner teachers alone, and §0.6 establishes
that the realistic venue is Crunch cloud at 15 h/week. One partition is 13.2 h
and fits a week; three do not. alt1 is also the partition **most likely to kill
the candidate** — it is the least favourable of the four, and the RT-600
specialisation delta there (+0.00254) already falls below W4-E1's own +0.0030
bar. So alt1 first is both the only affordable order and the correct one.

**1.1 Regenerate the teacher, alt1 first.** Per partition: 20 nested inner
teacher fits + the outer merge, producing `RT-991.altK.npy` and
`nested_Q_outer{f}_inner{g}.altK.npy`. Fold-purity sentinel
(`--fold-purity-test`) must pass on the partition before any score is read.

Two preconditions, both from §0.5: `cmd_inner_teacher` needs its
refuse-if-exists guard **before** the first production invocation, and the
output paths must be partition-suffixed so an alt run can never collide with a
canonical vector.

**1.2 Regenerate the student on the same partition.** Five outer fits via
`--train-outer F --folds-path research/folds/folds_altK.parquet`, then
`--merge-analyze`. Uses the existing 500-column causal bank; adds no features.
~5 min per outer fold, negligible against 1.1.

**1.3 Regenerate the two CatBoost members on the same partition.** Required for
the champion lane. Without `RT-1254.altK` / `RT-1255.altK` there is no E0 on alt
partitions and only the RT-600 lane can be evaluated.

If this refit is judged too expensive, the fallback is explicit and must be
recorded as a limitation, not quietly substituted: run the alt leg on the
**RT-600 lane only**, and accept that champion-lane partition sensitivity
remains untested.

**Note the fallback is smaller than it looks (§0.5).** It avoids the CatBoost
refits *only*. The teacher refits — the 13.2 h — are common to both lanes,
because the student's target derives from the nested Arm-C teacher on whichever
partition it is fitted. Choosing the cheaper lane does not make the leg cheap.

**1.4 Endpoint — a two-stage gate, declared before looking.** For each
partition K, compute the addition contract E0/E1/E2 exactly as in
`armc_e2_e1_addition_contract.py`, with E1 = RT-1257 + one matched added seed
clone.

Staging the compute forces staging the rule. A single-partition result is
**weaker** evidence than the four-partition mean originally drafted here, and
that must be stated in the rule rather than discovered afterwards.

**Stage 1 — alt1 alone (13.2 h). A kill gate only; it cannot pass anything.**

- **KILL** — E2−E1 on alt1 is negative, *or* below +0.0000 on 3 or more folds.
  The candidate's whole case is that it survives an unfavourable draw. It does
  not. Stop; do not buy alt2/alt3.
- **CONTINUE** — anything else. This is *not* a pass. It licenses spending the
  next 26 h, nothing more.

Stage 1 is deliberately asymmetric: alt1 can end the program but cannot
promote. A positive alt1 on its own is one draw from a partition chosen for
being pessimistic, which is a weak basis for a promotion and a strong basis for
continuing.

**Stage 2 — alt2 + alt3 (26.4 h), evaluated on all four partitions together.**

- **PASS** — mean E2−E1 across the four partitions ≥ 0.0011, *and* E2−E1 > 0 on
  at least 3 of 4 partitions, *and* no partition worse than −0.0011.
- **KILL** — mean across partitions below the noise floor, or two or more
  partitions negative.
- **INCONCLUSIVE** — anything else. Report as such; do not select the favourable
  partitions.

**Anti-gaming clause.** Stage 1's CONTINUE threshold is fixed here, before alt1
is run. If alt1 lands marginal, the response is Stage 2 or a recorded
INCONCLUSIVE — **not** a revised Stage 1 threshold, and not a decision to stop
at alt1 and quote it as support. Quota pressure is not a scientific argument.

Report every partition run and all per-fold vectors whatever the outcome, and
report which stage the program stopped at. `FAILED_EXPERIMENTS.md` and
`NEGATIVE_RESULTS_INDEX.md` get the row either way.

**1.5 Fold-0 diagnosis (runs alongside, does not gate).** Fold 0 is negative on
both endpoints on canonical. Establish whether that is partition-specific noise
or a property of the mechanism. If fold 0 is negative on multiple partitions, the
mechanism has a regime where it does damage, and that must be characterised —
by online-horizon bucket and by never-break/pre-break cut — before promotion is
arguable regardless of the mean.

---

## 5. Phase 2 — Causality gate

`PROTOCOL_CHAMPION_2026.md`: no predictive score is interpreted until this
passes. The ledger currently records `causal_verified=no`.

The student reuses the already-verified 500-column causal bank and adds no
features, so the feature-level gate is largely inherited. **None of it has been
run against the student's own inference path**, which is the new surface.

Required, against the student's inference path specifically:

- prefix invariance
- no dependence on total online horizon
- no future / known-τ / label-derived state at inference
- per-series inference independence
- feature-missingness audit for label leakage
- output shape / range / NaN contract

Two candidate-specific notes:

- The student is a **`regression`-objective booster**. Its raw output is a
  residual prediction, not a probability. It reaches a common scale through the
  same rank-based SCDF calibration as every other member, which tolerates that —
  but the output contract check must be written against a residual-valued
  member, not copied from a probability-valued one.
- The training **target** is teacher-derived. The teacher is nested and
  fold-pure (sentinel: 20/20 contamination caught under the old scheme, nested
  passes). Labels do not enter inference. Record this explicitly; it is the
  obvious place a reviewer will look for leakage.

Existing harness: `research/scripts/verify_causality_artifacts.py`, tracked on
this branch.

---

## 6. Phase 3 — Final-10k target regeneration

Only if Phases 1 and 2 pass.

Production fits use `research/folds/folds_final10k.parquet` over 10,000 series
(`sha256(id,fold) = 6e9eba…`). The nested Arm-C teacher that generates RT-1320's
label exists **only over the 8,000 dev series**. The target itself must be
regenerated at 10k before an artifact can be built.

Note the asymmetry deliberately: the dev evidence is 8k, the artifact is 10k, and
there is no held-out set left to check the 10k student against —
`FINAL_ARCHITECTURE_FREEZE.md` records the former 2,000-series lockbox as spent
and now training data. The 10k fit is an act of faith licensed by the 8k
evidence, exactly as RT-600 and RT-1257 were. Say so in the report rather than
implying the 10k artifact was validated.

The freeze also flags an existing residual calibration mismatch (grids from
8,000-series models, boosters from 10,000-series models). Adding an eighth
member does not fix it and should not be described as making it worse; check
whether the student's SCDF grid inherits the same mismatch and record the answer.

---

## 7. Phase 4 — Production artifact and manifest

Mechanical. `src/sbr/production/model.py` loads members from
`manifest["model_files"]` and slices from `manifest["booster_columns"]`, so
member count is manifest-driven — an eighth member is an artifact and
calibration-payload change, not a code redesign.

Two concrete things the registry does not mention:

**7.1 The provenance gate hardcodes seven.**
`ProductionModel.EXPECTED_PROVENANCE` contains `"n_boosters": 7`. An 8-member
artifact will raise `MODEL PROVENANCE MISMATCH` on load. The gate is overridable
per-artifact via `manifest["expected_provenance"]`, which is the correct
mechanism — set `n_boosters: 8` **in the manifest**, and do not edit the class
default, and do not set `SBR_ALLOW_UNPINNED_MODEL=1`. That escape hatch exists
for research loads; the CRF-02 incident that motivated the gate (a 24-series
unit-test checkpoint scored as production, voiding RT-1237/1238/1239) is the
reason it must not be used here.

**7.2 Runtime is not a risk.** From `RT1257_BENCHMARK.json`, RT-1257 projects
**3.69 h** against the **15 h** quota (RT-600: 4.04 h). Per-point cost is
dominated by the shared streaming engine, not by member prediction, so an eighth
booster adds roughly 0.3 h. Benchmark it anyway and record the number — but do
not plan around a runtime constraint that does not bind.

Also required: dependency delta (none expected — LightGBM regression booster),
peak RSS, and deterministic replay on the packaged artifact.

---

## 8. Phase 5 — External validation, and the RT-1257 dependency

RT-1320 stacks on top of a champion that is **not yet the formal production
anchor**. RT-600 still holds that role; RT-1257's promotion is deferred to a
separate owner tag/status action.

This is a sequencing dependency, not a scientific one, and it should be resolved
in parallel from day one rather than discovered at submission time. Promoting
RT-1257 is small, independent, and already evidenced (fresh local Crunch test
PASS 2026-08-31, PR #14 CI green, manifest/build-SHA gap shown to be packaging
rather than `src/sbr` model/feature changes).

**Recommendation: start RT-1257's formal promotion now, in parallel with
Phase 1.** It is on the critical path for Phase 5 and off the critical path for
everything else.

Then: fresh Crunch test on the 8-member artifact, and one external submission.

**Leaderboard discipline.** Per protocol, the external score validates or
falsifies a decision already made on internal evidence. It may not select
composition size, calibration, or retry. If the external score disappoints,
the recorded outcome is that the internal evidence was falsified — not an
invitation to re-tune.

---

## 9. What would make this KILL, stated in advance

So the outcome is not negotiated after the fact:

- **Stage 1:** E2−E1 on alt1 negative, or non-positive on 3 or more folds; or
- **Stage 2:** mean E2−E1 across four partitions below +0.0011; or
- **Stage 2:** two or more partitions negative; or
- fold 0 negative on a majority of partitions run, with an identifiable damage
  regime; or
- any causality-gate failure on the student's inference path that is not a
  test-harness bug.

Stopping at Stage 1 with a CONTINUE and no Stage 2 is **not** a pass. If Stage 2
is never funded, the recorded outcome is INCONCLUSIVE and RT-1320 stays
`RESEARCH_ALIVE` — it does not become promotable by default.

Any of these ends the promotion attempt. The candidate returns to
`RESEARCH_ALIVE` or moves to `KILL` per the evidence, the negative result is
filed, and the next question is whether the never-break repair mechanism can be
delivered some other way — see `NEW_AVENUES_2026.md` §E.2, which already
identifies never-break false positives as null-model errors.

---

## 10. Open decisions

1. **Alt-partition scope.** Full champion lane (adds the CatBoost refit per
   partition) or RT-600 lane only? Note §0.5: the fallback saves the CatBoost
   work only — the 13.2 h teacher cost is common to both lanes, so this is a
   smaller decision than it first appeared.
2. **Decision rule.** Is the two-stage §1.4 gate the one to freeze, and is
   Stage 1's CONTINUE threshold right? Separately: should the noise floor be
   re-derived by paired bootstrap on the pooled multi-partition distribution
   rather than reusing the canonical 0.0011?
3. **RT-1257 promotion.** Start it in parallel now, as recommended?
4. **Compute — RESOLVED IN PRINCIPLE, constrained in practice.** Run it on
   **Crunch cloud**, packaged as a submission, exactly as RT-1258/RT-1259 were
   (`submissions/G_gpu_tabular_benchmark.py`, submission 76357 / task
   `run-3e834e0f`).

   *What it fixes:* the memory blocker. Observed peak RAM there was **18.96 GB**,
   against this machine's 16 GB total with swap at 5.56/7.17 GB. The ~7.45 GB
   matrix fits with headroom. It is also Linux x86_64 / python 3.11, closer to
   `REPRODUCIBILITY_MANIFEST.json`'s recorded research environment than this Mac.

   *What it does not fix:* the quota. Crunch cloud bills the **same 15 h/week**
   the scoring runs use — the GPU arms budgeted against it explicitly
   (`quota_hours_total 15.0`, `quota_fraction_for_scored_training 0.9`,
   `quota_learners 2`, `per_learner_budget_seconds 24300`). Phase 1's 39.5 h is
   ~3 weeks of the entire quota, and spending it also costs submission capacity
   in those weeks. **The RTX 4090 buys nothing here**: the teacher is CPU
   LightGBM, and the GPU is only relevant if LightGBM is switched to
   `device=gpu`, which is a determinism-gated protocol change.

   *Consequence, now folded into §1.4:* run **alt1 alone first** at 13.2 h — one
   week's quota, and the partition most likely to kill the candidate.

   *Still open:* whether to spend one week's quota on alt1 at all, given it also
   displaces a scoring submission that week.

5. **Thread count.** Is a `num_threads` change in scope? It is worth ~5x here,
   but only after a determinism demonstration; otherwise the 13.2 h per
   partition stands. Blocked on §0.6.1 (cloud core count is unrecorded).

6. **Stage-2 funding.** If alt1 returns CONTINUE, is the further 26.4 h — two
   more weeks of quota — authorised in advance, or does it return for a decision?
   Deciding now avoids the §1.4 anti-gaming failure mode where a marginal alt1
   gets quoted as support because Stage 2 was never funded.
