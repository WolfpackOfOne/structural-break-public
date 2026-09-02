# Phase 2 — RT-1320 causality gate

Date: 2026-08-31. Plan: `RT1320_PROMOTION_PLAN.md` §5.
Status: **research-stage gate PASSES. Artifact-level gate NOT RUN — see limits.**

`PROTOCOL_CHAMPION_2026.md` states no predictive score is interpreted before the
causality gate passes, and the ledger recorded `causal_verified=no`. Every
number in `PHASE1_ALT1_STAGE1.md` was produced before this ran, and was reported
carrying that caveat.

Implementation: `tests/test_rt1320_causality.py`, seven gates, following the
`test_wave8_causality.py` pattern already used for research-stage arms.

## Gates

| # | gate | result |
|---|---|---|
| 1 | inputs are exactly the 500-column causal bank | PASS |
| 2 | no forbidden token reachable as a feature name | PASS |
| 3 | the sentinel **rejects** a planted forbidden column | PASS |
| 4 | nested purity, and the sentinel still rejects the old global-OOF scheme | PASS |
| 5 | no label-derived state enters inference | PASS |
| 6 | per-row independence and deterministic replay | PASS |
| 7 | output contract | PASS |

GATE 3 and GATE 4's second half matter more than they look: both check the
sentinel still *catches a defect* rather than passing on clean data. A gate that
cannot fail proves nothing, which is the lesson RT-992/RT-993 paid for.

GATE 5 checks the inference input path structurally — `stack_features` takes
matrices and row indices only, and its source contains no `d.y`, `has_break`,
`tau_index` or `nested_Q`. The teacher-derived target is training-only.

GATE 7 records what the plan predicted: the student's output range is
**[-4.325093, 8.671623]** over 4,032,524 covered rows. It is a
`regression`-objective booster and its raw output is a residual, **not** a
probability. The range is recorded, deliberately not asserted to [0, 1] — that
would be the wrong contract for this member and would only ever pass by
accident.

## Limits — do not read this as a full causality clearance

**Streaming prefix-invariance through a production inference path is NOT
covered.** RT-1320 has no production artifact; that is Phase 4. The artifact-level
harness `verify_causality_artifacts.py` requires `SBR_MODEL_DIR` and cannot be
pointed at a research-stage booster.

Prefix invariance is **inherited** here: the student is a pure row-wise function
of the causal bank, and that bank's prefix invariance is what GATE 1 pins. That
is a sound argument, not an independent measurement of the student's own
streaming path. **It must be re-run directly once a production artifact exists.**

Ledger status should therefore read `causal_verified=research-stage`, not `yes`.

## A platform note worth keeping

GATE 6's first draft segfaulted LightGBM when the full suite ran (exit 139),
while passing standalone. Cause: twenty-five single-row `predict` calls, each
spinning up a thread pool. `sbr/production/model.py` already documents exactly
this — it calls `predict(..., validate_features=False, num_threads=1)` because
"the default path rebuilds a pandas-style feature-name check and spins up a
thread pool for a single row" — and the repo previously hit the same fragility
at `INFER_PARALLELISM=4`, which is why that constant is 1.

The fix was to match production's calling convention, not to weaken the test.

## Suite

`718 passed, 56 skipped` (711 before these seven), exit 0. The 4 errors in
`test_wave5_hardneg_invariants.py` are pre-existing and data-dependent.
