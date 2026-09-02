# Phase 1, alt2 + alt3 — Stage 2, all four partitions

Date: 2026-09-01. Plan: `RT1320_PROMOTION_PLAN.md` §1.4 Stage 2, §9.
Verdict: **NOT RECORDED HERE.** Every §4 PASS condition is met and no KILL
condition fires, but the verdict against a preregistered gate is an owner
action. This document reports; it does not decide.

## The gate, as frozen before alt1 was run

- **PASS** — mean E2−E1 across the four partitions ≥ 0.0011, *and* E2−E1 > 0 on
  at least 3 of 4 partitions, *and* no partition worse than −0.0011.
- **KILL** — mean below the noise floor, or two or more partitions negative.
- **INCONCLUSIVE** — anything else.

## Result — champion lane, addition contract, all four partitions

E0 = RT-1257 on the partition. E1 = RT-1257 + one matched added seed clone
(RT-403). E2 = RT-1257 + the RT-1320 student.

| partition | E2−E1 | folds + | E2−E0 | fold 0 | control E1−E0 |
|-----------|------------|-------|------------|-------------|-------------|
| canonical | +0.0015159 | 4 / 5 | +0.0014161 | **−0.0013627** | −0.0000998 |
| alt1      | +0.0021146 | 5 / 5 | +0.0018707 | +0.0021428 | −0.0002439 |
| alt2      | +0.0014825 | 5 / 5 | +0.0012849 | +0.0010623 | −0.0001976 |
| alt3      | +0.0019582 | 4 / 5 | +0.0017287 | +0.0005588 | −0.0002295 |

Per-fold primary vectors are in the four `E2_E1_addition_contract.json` records
(`PHASE1_ALT1/ALT2/ALT3_...` here, canonical in
`armc_residual_student_confirm_s20260901/`).

## Against the frozen rule

```
mean E2-E1 across 4 partitions : +0.0017678   rule: >= +0.0011      MET
partitions with E2-E1 > 0      : 4/4          rule: >= 3 of 4       MET
worst partition                : +0.0014825   rule: none < -0.0011  MET
partitions negative            : 0            KILL if >= 2          NOT TRIGGERED
```

## Fold 0 — §1.5

Fold 0 is negative on **canonical only, 1 of 4**. §9's kill criterion requires
it negative on a *majority* of partitions run, which would be 3. Not triggered.

§1.5's conditional therefore never fires: the horizon-bucket and
never-break/pre-break characterisation is required only *if* fold 0 is negative
on multiple partitions. There is no damage regime to characterise.

Record: `PHASE1_FOLD0_DIAGNOSIS.json`, produced by
`research/scripts/rt1320_fold0_diagnosis.py`.

**Known gap in that record.** The diagnostic recomputes fold 0 from the stored
vectors and could only evaluate alt1/alt2/alt3 — canonical's student outer
`.npy` files do not exist in this worktree, only committed JSON, so it is listed
under `partitions_skipped`. Canonical's fold-0 value above comes from its
endpoint JSON, not from a recomputation. Its `n_negative: 0` therefore reads one
lower than reality. The conclusion is unchanged, but a ~29 min canonical student
refit would make the §1.5 artifact self-contained.

## What the leg was designed to expose, and what it found

`FINAL_ARCHITECTURE_FREEZE.md` records canonical as the **most favourable** of
the four partitions, and the whole point of this leg was that RT-1320's evidence
was one draw from that flattering partition. The result is the opposite:
canonical is the **weakest** of the four and the only one with a negative fold.
The candidate is stronger on all three alternates.

The E1 control is flat-to-slightly-negative on every partition (−0.0001 to
−0.0002). The RT-1264 / CSA-04 inflation shape — where the apparent gain comes
from a control that degrades — does not appear anywhere in this leg.

## Caveat that applies to every number above

Phase 2's causality gate passed at research stage (`PHASE2_CAUSALITY.md`), and
the artifact-level gate passed on the assembled 8-member artifact
(`engineering/reports/rt1320_promotion_prep/ARTIFACT_CAUSALITY.json`, on
`engineering/rt1320-promotion-prep`). Anything quoted from this document should
still carry its provenance: these are development-partition OOF endpoints, not
an external score. `external_score` remains an open blocker and needs Crunch
quota.

## What is NOT done here

Per §9 the outcome must be recorded whichever way it goes, including rows in
`FAILED_EXPERIMENTS.md` and `NEGATIVE_RESULTS_INDEX.md`, and a status change in
`MODEL_REGISTRY.md`. Those are deliberately **not** written by this document,
because each presumes a verdict, and the verdict has not been recorded. They
follow the owner's §4 call.

## Provenance

- alt2: nested teachers already on disk from earlier work (20 vectors);
  CatBoost members `RT-1254.alt2` / `RT-1255.alt2` fitted this session
  (10 folds, ~72 min); student 5 outer fits + merge (~19 min); endpoint
  `armc_e2_e1_addition_contract.py --oof-suffix .alt2`.
- alt3: nested teachers fitted this session, 10 pair fits → 20 vectors
  (14:18–16:17, ~2 h); fold-purity sentinel PASS; CatBoost members
  `RT-1254.alt3` / `RT-1255.alt3` (~63 min); student 5 outer fits + merge
  (~29 min); endpoint `--oof-suffix .alt3`.
- Fold-purity sentinel passed on both partitions before any score was read
  (§1.1): 0/20 failures on the nested scheme, 20/20 contamination correctly
  detected on the old global-OOF scheme.
- Sequenced by `research/scripts/rt1320_phase1_stage2.sh`, which mirrors the
  alt1 Stage 1 invocations recorded in `PHASE1_ALT1_STAGE1.md §Provenance`.
- Threads: 8. Phase 0.7 Lever 2 established predictions are bitwise equal
  across thread counts, so this does not confound the partition comparison.
