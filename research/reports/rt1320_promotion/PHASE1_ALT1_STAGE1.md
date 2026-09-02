# Phase 1, alt1 — Stage 1 gate

Date: 2026-08-31. Plan: `RT1320_PROMOTION_PLAN.md` §1.4 Stage 1.
Verdict: **CONTINUE.** Not a pass — Stage 1 cannot pass anything.

## The gate, as frozen before the run

- **KILL** — E2−E1 on alt1 negative, *or* non-positive on 3 or more folds.
- **CONTINUE** — anything else. Licenses spending Stage 2; nothing more.

## Result — champion lane, addition contract

E0 = RT-1257 (alt1). E1 = RT-1257 + one matched added seed clone (RT-403, alt1).
E2 = RT-1257 + RT-1320 student (alt1).

| | E0 | E1 | E2 |
|---|---:|---:|---:|
| mean TS-AUC | 0.6213025 | 0.6210586 | 0.6231732 |

| endpoint | value | folds positive |
|---|---:|---:|
| **PRIMARY E2−E1** | **+0.0021146** | **5/5** |
| SECONDARY E2−E0 | +0.0018707 | 5/5 |
| control lift E1−E0 | −0.0002439 | — |

Primary per fold: +0.002143, +0.003580, +0.001273, +0.001517, +0.002060.
Every fold clears zero; the weakest is +0.00127. Noise floor 0.0011 — the mean
clears it by ~1.9x.

## Against canonical

| | canonical | alt1 |
|---|---:|---:|
| PRIMARY E2−E1 | +0.001516 | **+0.002115** |
| primary folds positive | 4/5 | **5/5** |
| SECONDARY E2−E0 | +0.001416 | +0.001871 |
| secondary folds positive | 4/5 | 5/5 |
| control lift E1−E0 | −0.000100 | −0.000244 |

Canonical per fold: −0.001363, +0.002705, +0.001458, +0.003487, +0.001292.

**The candidate is stronger on alt1 than on canonical, and fold 0 — negative on
both canonical endpoints — is positive here (+0.002143).** That is the opposite
of what this leg was designed to expose. `FINAL_ARCHITECTURE_FREEZE.md` records
alt1 as the least favourable of the four partitions for the RT-600
specialisation delta, and the plan's §4 rationale rested on it being where the
candidate was most likely to die.

The E1 control is flat-to-slightly-negative on both partitions (−0.000100,
−0.000244), so neither result carries the RT-1264/CSA-04 inflation shape where a
degrading control manufactures the headline.

## What this does and does not license

**Does:** spending Stage 2 — alt2 and alt3.

**Does not:** promotion, or any claim that RT-1320 has cleared the
alternate-partition leg. Two partitions is not four, and §1.4's PASS rule is
defined on the four-partition mean with at least 3 of 4 positive and none worse
than −0.0011. Stage 1 is a kill gate by construction: a positive alt1 is one
draw from a partition chosen for being pessimistic, which is a weak basis for
promotion and a strong basis for continuing.

Per §9, stopping here with a CONTINUE and no Stage 2 is recorded as
**INCONCLUSIVE**, not a pass, and RT-1320 stays `RESEARCH_ALIVE`.

## Caveat that applies to every number above

**The causality gate has not run.** `PROTOCOL_CHAMPION_2026.md` states that no
predictive score is interpreted before prefix invariance, horizon independence,
per-series independence, deterministic replay and the output contract pass
against the candidate's own inference path. The ledger still records
`causal_verified=no`, and Phase 2 is outstanding.

These scores are therefore **provisional**. They are reported because the gate
decision is about whether to spend more compute, not about promotion — but they
must carry `causal_verified=no` wherever they are quoted, and Phase 2 should
run before any further interpretation.

## Provenance

- alt1 CatBoost members fitted this session:
  `RT-1255.alt1.npy` (CAT-300), `RT-1254.alt1.npy` (CAT-413), ~63 min for both,
  `catboost_specialist_2026.py --partition alt1 --train-only`, no ledger rows.
- alt1 nested teachers: 10 pair fits, 20 vectors, 71.5 min.
- alt1 student: 5 outer fits + merge, 29 min.
- Endpoint: `armc_e2_e1_addition_contract.py --oof-suffix .alt1`.
- Raw: `PHASE1_ALT1_E2_E1_addition_contract.json`.
