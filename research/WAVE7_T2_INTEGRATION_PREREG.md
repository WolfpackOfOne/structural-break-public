# WAVE 7 — T2 (RT-995) ENSEMBLE-INTEGRATION PRE-REGISTRATION

**Written 2026-08-24 on `research/wave7-t2-promotion` (forked from
`research/wave7-teacher-distillation@5093e0a`), committed BEFORE any E0/E1/E2
score exists.** This is the promotion battery's leg that Wave 8's own final
report (`research/reports/wave8_final.md` section R, inherited via
`research/wave8-future-aware-distillation@589e1db`) named as the single
highest-value next experiment: does `T2` add ranking information to the
seven-specialist RT-600 ensemble beyond what an eighth exchangeable model
would add for free.

T2 is frozen (architecture, features, blend weight, seed, row cap,
preprocessing — see `research/WAVE7_TEACHER_NESTED_PREREG.md`). This document
freezes only the **integration** protocol: which arms are scored, on what
data, with what calibration, before any of those scores are read.

## 1. Source of truth, not re-derived

* `T2` = `RT-995`, canonical (non-alternate) partition, whole-dev OOF array
  `research/oof/RT-995.npy`, copied byte-identical from
  `research/wave7-teacher-distillation@5093e0a`. Not retrained here.
* Seven specialists (`RT-300`, `RT-410`–`RT-415`) and the matched seed clone
  (`RT-401`) — copied byte-identical from the same source branch's
  `research/oof/`. Not retrained here.
* Evaluation harness — `research/scripts/wave8_common.py`
  (`ensemble_marginal`, `pair_repair_stats`), cherry-picked verbatim from
  `research/wave8-future-aware-distillation@fca489a` (commit `4b04983` on
  this branch). No new evaluation code is written; existing framework reused
  per instruction.

## 2. Arms (frozen)

* **E0** — `rt600_7stream`: cross-fitted, equal-weight `crossfit_blend` of
  the seven specialists only. This is the existing RT-600 research ensemble
  score, unmodified.
* **E1** — `rt600_plus_seedclone`: E0's seven specialists plus `RT-401`
  (an exchangeable matched-seed clone of an existing specialist), equal
  weight across all eight streams.
* **E2** — `rt600_plus_t2`: E0's seven specialists plus `T2` (`RT-995`),
  equal weight across all eight streams.

Primary comparison: **E2 − E1** (does T2 beat the free variance-reduction
benefit of an eighth exchangeable stream). Secondary: **E2 − E0**, **E1 −
E0**.

## 3. Calibration (frozen, not chosen after seeing a score)

`wave5_lib.Ctx.crossfit_blend`: for each of the 5 canonical folds `k`, every
stream's calibration map is fit on folds `!= k` only (`CANON_CAL`, the
project's existing smooth time-conditioned calibrator — no global rank
transform, no validation-row-trained map, no cross-series live rank, no
final-horizon information), then applied to fold `k`. Equal weight
(`weights=None`) across streams in every arm, per repository convention
(the same default `wave8_common.ensemble_marginal` used for all five Wave-8
pilot mechanisms) — no weight search on T2.

## 4. Fold scope

`ensemble_marginal(..., fold=0)` — fold 0, matching the fold every Wave-8
mechanism was scored on (`RT-990`, `RT-991`, and all five Wave-8 arms are
fold-0 pilots). This keeps E2 vs the five already-measured Wave-8 marginals
(`SST`/`ORR`/`PCFB`/`CFEP`/`TGMC`, all ≈ −0.0003) an apples-to-apples
comparison. Extending to all 5 folds is a natural follow-on but is not
required to answer the primary question and is not pre-registered here to
avoid multiplying comparisons after seeing fold 0.

## 5. Pair-flow and diversity (frozen)

`wave8_common.pair_repair_stats(base=E0_blend, candidate=E2_blend, ...)` for
repairs/damage vs the E0 baseline, `n_pairs_per_t=10, seed=0` (matching
Wave-8 convention). Within-t rank correlation of `T2` against each of the
seven specialists individually and against the E0 blend, via
`np.corrcoef` on fold-0 rows where both are non-NaN — arithmetic only, no
new modeling choice.

## 6. What this does NOT do

No hyperparameter or weight search on T2 or on the blend. No retraining of
any stream. No test-set or final10k access. No submission decision — this
measures E2−E1, nothing else. Alternate-partition confirmation (T2 retrained
under `folds_alt{1,2,3}`) is a separate, much more expensive leg (nested
double-cross-fit retraining, ~5h wall-clock per partition based on the
canonical run's timing) and is reported separately, gated on whether this
cheap integration result makes that compute worth spending.

## 7. Reading (fixed before any score is read)

Per `research/WAVE7_TEACHER_NESTED_PREREG.md`-style pre-committed bands,
applied to **E2 − E1**:

* `>= +0.0050` → MAJOR COMPETITION ALPHA
* `+0.0030` to `+0.0050` → STRONG / PROMOTABLE
* `+0.0015` to `+0.0030` → REAL BUT MODEST
* `0` to `+0.0015` → MOSTLY REDUNDANT
* `< 0` → REJECT AS ENSEMBLE ADDITION

## GIT

| | |
|---|---|
| starting SHA (this branch, before infra pull) | `5093e0a` |
| infra pull commit | `4b04983` |
| this pre-registration commit | see `git log -1` after commit |
| first E0/E1/E2 score | strictly after this commit |
