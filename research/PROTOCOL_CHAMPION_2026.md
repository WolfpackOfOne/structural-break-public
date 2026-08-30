# Champion-Relative Research Protocol — 2026 Consolidated Standard

This file supplements the historical `PROTOCOL.md` after the external validation of RT-1257. Historical preregistrations remain governed by the protocol that existed when they were run; this document defines the default standard for new post-consolidation research.

## Baselines

- **E0-CHAMP:** RT-1257, current external champion, 0.6290 official TS-AUC.
- **E0-LGBM:** RT-600, permanent homogeneous seven-LightGBM reference, 0.6268 official TS-AUC.

A new production candidate must ultimately demonstrate complementary information relative to RT-1257. RT-600 remains useful for diagnosing whether a gain comes from learner-family diversity or from a mechanism that also improves the homogeneous reference.

## Candidate evaluation

Where a candidate can replace or add an exchangeable slot, use a matched-control design:

- C0: candidate standalone.
- C1: matched control standalone.
- E0: RT-1257 unchanged.
- E1: RT-1257 with the matched exchangeable control under the same replacement/addition contract.
- E2: RT-1257 with the candidate.

Primary endpoint: **E2 - E1**.

Secondary endpoint: **E2 - E0**.

Never interpret E2-E1 without inspecting E0, E1, and E2 separately. RT-1264/CSA-04 demonstrated that a degrading E1 control can manufacture an inflated marginal headline as composition size grows.

## Required supporting evidence

For serious promotion decisions, report at minimum:

- per-fold E2-E1
- folds positive
- E2-E0
- standalone candidate and matched-control performance
- within-t rank correlation against relevant incumbent/control
- whole-population pair repairs/damage/net
- dominant-cell pair repairs/damage/net
- mature-vs-never repairs/damage/net
- mature-vs-prebreak repairs/damage/net
- runtime and memory
- feature/inference causality status
- prefix invariance
- series independence
- deterministic replay
- deployment complexity and dependency delta

A candidate with strong standalone AUC, low correlation, or a positive dominant-cell result can still fail if whole-system marginal value is absent or offsetting damage dominates.

## Noise and parsimony

Use a predeclared paired-bootstrap/noise rule when choosing among nested composition sizes or closely related candidates. If a more complex candidate's improvement over a simpler candidate is below the declared noise floor, choose the simpler candidate.

Do not select composition size from a descriptive exhaustive-subset appendix unless that selection rule was preregistered before the subset outcomes were inspected.

## Causality gate

No predictive score is interpreted before:

- prefix invariance passes,
- no feature depends on total online horizon,
- no future/known-tau/label-derived state enters inference,
- per-series inference remains independent,
- feature missingness is audited for label leakage,
- output shape/range/NaN contracts pass.

Oracle/full-sequence diagnostics may be used only as information-frontier evidence and must be explicitly labeled `ORACLE_NONCAUSAL` or `DIAGNOSTIC`.

## Leaderboard discipline

Official leaderboard results are external validation points. They may not select model parameters, feature blocks, composition size, calibration settings, or retry decisions.

A model is promoted on preregistered internal evidence; an external score may validate or falsify that promotion decision but must not become the next hyperparameter objective.

## Status discipline

Use the status vocabulary in `MODEL_REGISTRY.md`:

- EXTERNAL_CHAMPION
- REFERENCE
- ACTIVE_COMPONENT
- RESEARCH_ALIVE
- PARKED
- SUPERSEDED
- KILL
- INFEASIBLE
- DIAGNOSTIC
- ORACLE_NONCAUSAL
- INVALID

Do not collapse INFEASIBLE, PARKED, SUPERSEDED, and KILL into the same label; they imply different evidence and different retry logic.
