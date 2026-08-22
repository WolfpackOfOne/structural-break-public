# WAVE 5 — STATUS

**Updated 2026-08-22, branch `research/wave5-alpha`.**
Charter: `research/WAVE5_CHARTER.md` · Freeze: `research/WAVE5_C2_UNSCORED_FREEZE.md`

> **TRAINING RUNS = 0 · NEW TS-AUC VALUES OBSERVED = 0 · LEADERBOARD SUBMISSIONS = 0**
>
> The zero-score state is intact and deliberate. Every mechanism in the bank was
> fixed before any evidence about whether it works.

---

## PHASES

| phase | scope | state |
|---|---|---|
| **C1** | open the branch, audit the broadened model family, build the first two mechanisms | **COMPLETE** |
| **C2** | redundancy-audit the remaining NOT-RUN families, build only what is genuinely new, pre-register, freeze | **COMPLETE** |
| **C3** | staged scoring against the pre-registration | **BLOCKED — no competition data in this container** |

## THE BANK

| family | cols | ms/series | nearest existing | novelty | mechanism certified | scored? |
|---|---:|---:|---|---|---|---|
| `m10_perm` | 24 | 40.5 | `m00_core` window means | **NEW** | yes | **NO** |
| `m11_focus` | 21 | 62.3 | `m01_seq` dyadic max-GLR | **NEW** | yes | **NO** |
| `m12_deplr` | 14 | 55.1 | `m04_resid` `e_acf1` | **NEW** (R² 0.73 on the full bank) | yes | **NO** |
| hard-negative training | n/a | n/a | uniform sampling | protocol, not a module | fold purity proven | **NO** |

## REJECTED BEFORE SCORING — four ideas, zero compute

| idea | reason |
|---|---|
| mean-vs-robust-mean contrast | algebraically dead (1.8e-15) |
| generic robust distribution distances | already in `m02_dist` |
| residual CUSUMSQ (`m13_rcsq`) | loses to the incumbent on its own mechanism; 94% redundant |
| absorbing-state BOCPD (`m14_absorb`) | `m07_bayes` already is one |

## WHAT REMAINS FROM THE ORIGINAL EIGHT NOT-RUN FAMILIES

| family | disposition |
|---|---|
| transient vs permanent | **built** — `m10_perm` |
| dependence likelihood v2 | **built** — `m12_deplr` |
| robust distribution distances | **withdrawn**, redundant with `m02_dist` |
| residual CUSUMSQ / AR regression break | **rejected**, W5-R1 |
| robust BOCPD | **rejected**, W5-R2 — `m07_bayes` already absorbing |
| hard-negative sampling | **protocol pre-registered and gated**, unscored |
| AR(p)-FOCuS | **built** — `m11_focus` (adaptive-baseline variant) |
| TabPFN / deep / teacher | **blocked** — no GPU, HuggingFace refused, checkpoint gated |

Seven of eight are resolved. The eighth is blocked on environment, not on ideas.

## BLOCKERS

| blocker | detail |
|---|---|
| 2026 store absent | `cache/store` missing; `crunchdao.com` refused at CONNECT, so the container cannot self-serve. Remedy: `research/scripts/wave5_ingest_store.py` with a URL on a reachable host (~134 MB, verified byte-exact) |
| TabPFN weights | `huggingface.co` refused; `Prior-Labs/tabpfn_3` gated. Blocks 2025 rungs R25-040/050 and any TabPFN teacher |
| GPU | none present; torch runs CPU-only on 4 cores |

## NEXT ACTION

Ingest and verify the store, then run C3 **stage 1 only** (baseline/control
reproduction on this platform) before any candidate is scored. Scoring order and
the binding pre-commitments are in `WAVE5_C2_UNSCORED_FREEZE.md` §5.
