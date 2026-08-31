# Phase 0.2 — Are the stored alternate-partition OOF vectors stale?

Date: 2026-08-31. Plan: `research/RT1320_PROMOTION_PLAN.md` §0.2.
Status: **PASS.** Diagnostic only; selects nothing, promotes nothing.

## Question

`RT1320_PROMOTION_PLAN.md` §0.1 established that the seven RT-600 specialists and
both seed clones already have `.alt1/.alt2/.alt3` OOF on disk, which is why the
alternate-partition leg (Phase 1) is much cheaper than `MODEL_REGISTRY.md`
implies. Those vectors date from the wave-4 partition study (W4-E2).

Before Phase 1 is planned around them, they must be shown to still be the
vectors that produced the recorded result. If they are stale, Phase 1 is a full
refit and its cost estimate changes completely.

## Method

Recompute W4-E2 from the `.npy` files on disk **today**, using the tracked
analysis code unchanged, and diff against the committed record.

- Analysis code: `research/scripts/wave4_partition_analyse.py`, unmodified.
- Driver: `research/scripts/rt1320_phase0_alt_reverify.py`.
- Committed record under test:
  `research/reports/ensemble_partition_stability.json`.
- Inputs: 13 streams (`RT-300`; seed clones `RT-401`–`RT-406`; specialists
  `RT-410`–`RT-415`) × 4 partitions = **52 vectors**, resolved through the
  symlink `research/oof -> ../structural-break-wave8/research/oof`.
- The driver redirects the analysis output to the scratchpad so the tracked
  record it is being compared against is never overwritten.

Every input vector's absolute path, SHA-256, size and mtime are recorded in
`PHASE0_ALT_REVERIFY_INPUTS.json`, per the plan's §0.4 discipline.

## Result

All 52 vectors present. All 52 SHA-256 values **distinct** — no alt vector is a
duplicate of its canonical counterpart, which is the way a stale copy would most
plausibly have presented.

Recomputed, diffed against the committed record across three arms and three
deltas on four partitions — 24 quantities:

| partition | single | seed clone | specialist | bagging | specialisation | total |
|---|---:|---:|---:|---:|---:|---:|
| canonical | 0.61605365 | 0.62163961 | 0.62580863 | +0.00558597 | +0.00416901 | +0.00975498 |
| alt1 | 0.60985298 | 0.61525820 | 0.61780255 | +0.00540521 | +0.00254436 | +0.00794957 |
| alt2 | 0.61903740 | 0.62363587 | 0.62755959 | +0.00459846 | +0.00392372 | +0.00852219 |
| alt3 | 0.61313587 | 0.61751581 | 0.62020438 | +0.00437995 | +0.00268857 | +0.00706852 |

**Worst absolute discrepancy vs the committed record, across all 24 quantities:
`0.000e+00`.** Not "within tolerance" — identical.

Summary statistics likewise reproduce:

| delta | mean | sd | range | all positive |
|---|---:|---:|---|---|
| bagging | +0.00499 | 0.00059 | [+0.00438, +0.00559] | yes |
| specialisation | +0.00333 | 0.00083 | [+0.00254, +0.00417] | yes |
| total | +0.00832 | 0.00113 | [+0.00707, +0.00975] | yes |

Single-model level spread across partitions: 0.00918 (sd 0.00394).

These match `FINAL_ARCHITECTURE_FREEZE.md` §1 as written: total delta +0.0083
mean with SD 0.0011; partition SD 0.0039 at single-model level; alt1
specialisation delta +0.00254; canonical specialisation +0.00417; twelve of
twelve deltas positive.

## Verdict

**The stored alternate-partition vectors are not stale.** Phase 1 may be planned
on the assumption that the seven specialists and both seed clones do not need
refitting under `folds_alt*`.

This also re-confirms, from the vectors rather than from prose, the fact that
motivates Phase 1 in the first place: **canonical is the most favourable of the
four partitions** on every one of the three deltas, and alt1's specialisation
delta (+0.00254) sits below W4-E1's own +0.0030 preregistered bar.

## Limitations — read before quoting this

1. **This is integrity, not provenance.** Bitwise reproduction proves the files
   are the same bytes that produced the committed record. It does *not* prove
   those bytes were produced by the configurations claimed for them. That is the
   same distinction `ProductionModel._check_provenance` is built around, and the
   CRF-02 incident that voided RT-1237/1238/1239 was a provenance failure that
   every checksum passed.
2. **Scope is the RT-600 lane only.** This says nothing about `RT-991`,
   `nested_Q_*`, `RT-1254` or `RT-1255`, none of which have alt-partition
   vectors at all. Those remain the actual cost of Phase 1.
3. **A zero discrepancy is weaker evidence of environment-independence than it
   looks.** It is consistent with the committed record having been produced on
   this machine in this venv, in which case the run re-executed identical code
   over identical bytes and identity is the expected outcome rather than a
   robustness finding. It was not run cross-platform.
4. Diagnostic only. Nothing here selects a model, a composition, or a partition.

## Environment

python 3.11.6, numpy 2.4.6, pandas 3.0.5, pyarrow 25.0.1, scipy 1.17.1,
scikit-learn 1.9.0, lightgbm 4.7.0, macOS-26.5.2-arm64.

Matches `FINAL_REPRODUCIBILITY_MANIFEST.json` `release_candidate.environment`
exactly. Interpreter: `structural-break/.venv/bin/python`.

Wall time ~5 min single-threaded, no GPU. Machine checked clear of other jobs
before launch (load 1.63, no python/lightgbm/catboost processes).

## Artifacts

- `PHASE0_ALT_REVERIFY_RECOMPUTED.json` — the recomputation
- `PHASE0_ALT_REVERIFY_INPUTS.json` — 52 input paths with SHA-256
- `PHASE0_ALT_REVERIFY.log` — full console output including the diff
- `research/scripts/rt1320_phase0_alt_reverify.py` — the driver
