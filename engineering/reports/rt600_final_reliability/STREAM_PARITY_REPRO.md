# RT-600 batch/stream parity defect — reproduction and root cause

**Status:** REPRODUCED, ROOT-CAUSED, ROOT CAUSE IS AN IMPLEMENTATION DEFECT
(a train/serve skew), not an unavoidable floating-point ordering artifact.

This document records the state of the defect **before** any fix, per the
"reproduce first, fix second" rule.

---

## 1. Environment of record

| item | value |
|---|---|
| repo | `WolfpackOfOne/structural-break` |
| branch | `engineering/rt600-final-reliability-2026` |
| base commit | `81658764c95a7d91f3ca1f80ff25ad8c2b9b33e2` (CRF tip, `CRF_PROGRAM_EXHAUSTED`) |
| platform | macOS 26.5.2, arm64 |
| Python | 3.11.6 (conda-forge, Clang 15.0.7) |
| numpy | 2.4.6 |
| pandas | 3.0.5 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.0 |
| lightgbm | 4.7.0 |
| numba | 0.67.0 |
| torch | not installed in the research venv |
| store | `cache/store` (10,000 series, 5,036,517 online rows) |

Matches `research/FINAL_ARCHITECTURE_FREEZE.md` §2 "Environment" exactly.

## 2. Reproduction

```
SBR_STORE=$PWD/cache/store pytest tests/test_stream_engine_parity.py -q
```

Result: **4 of 5 seeds fail**, exactly as previously recorded.

| seed | series | n_hist | n_online | tau | cells differing | of |
|---|---|---|---|---|---|---|
| 0    | 8506 | 3787 | 149 | — | 0 | 74,500 |
| 0    | 6369 | 4763 | 261 | — | 0 | 130,500 |
| 0    | **5111** | 4132 | 370 | -1 | **2** | 185,000 |
| 137  | 4911 | 1343 | 355 | — | 0 | 177,500 |
| 137  | **8581** | 4489 | 442 | -1 | **1** | 221,000 |
| 137  | 2349 | 2671 | 780 | — | 0 | 390,000 |
| 999  | 8138 / 7788 / 1742 | — | — | — | 0 | — |
| 4242 | **4746** | 2167 | 877 | -1 | **2** | 438,500 |
| 4242 | 7822 / 8558 | — | — | — | 0 | — |
| 7777 | **5002** | 4785 | 357 | 26 | **1** | 178,500 |
| 7777 | 9419 / 4971 | — | — | — | 0 | — |

### Differing cells (batch = cache/training value, stream = production value)

| series | t | column | batch | stream | abs diff | rel diff |
|---|---|---|---|---|---|---|
| 5111 | 188 | `m07_bayes::bo_p_lt25_z`    | -0.2561489    | -0.25614887   | 2.980e-08 | 1.16e-07 |
| 5111 | 296 | `m07_bayes::bo_p_lt25_z`    | -0.028020771  | -0.02802077   | 1.863e-09 | 6.65e-08 |
| 8581 | 36  | `m07_bayes::bo_lo_change_z` | -3.4722543    | -3.472254     | 2.384e-07 | 6.87e-08 |
| 4746 | 735 | `m07_bayes::bo_lo_change_z` | -0.11939903   | -0.119399026  | 7.451e-09 | 6.24e-08 |
| 4746 | 876 | `m07_bayes::bo_ent`         | 1.9279711     | 1.927971      | 1.192e-07 | 6.18e-08 |
| 5002 | 145 | `m07_bayes::bo_p_lt25_z`    | -0.011449598  | -0.011449599  | 9.313e-10 | 8.13e-08 |

**Correction to the prior characterisation.** The defect was previously recorded
as confined to `m07_bayes::bo_p_lt25_z`. It is not. It also reaches
`bo_lo_change_z` and — decisively — `bo_ent`, which is a **raw BOCPD output**
with no z-transform, null calibration or clipping applied. That rules out the
null-calibration / clipping family of explanations and localises the defect to
the BOCPD recursion itself.

Every difference is exactly 1 ULP in float32. The columns are emitted as float32;
the underlying float64 disagreement is ~1e-13 relative and only becomes visible
when it straddles a float32 rounding boundary. This is why so few cells differ:
**the defect is present on essentially every row and is almost always masked by
the float32 cast.**

## 3. First causal divergence

Isolating the BOCPD kernel on series 5111 (`SER=5111`), comparing three
implementations of the *same* recursion:

| comparison | online `g` (370x8) | historical `ghh` (800x8) |
|---|---|---|
| `_bocpd` (numba JIT) vs `_BocpdStream` | 2530 / 2960 differ | 5537 / 6400 differ |
| `_bocpd.py_func` (same source, un-JIT'd) vs `_BocpdStream` | **0 / 2960** | **0 / 6400** |
| `_bocpd` (JIT) vs `_bocpd.py_func` (same source) | 2530 / 2960 differ | 5537 / 6400 differ |

First divergence: `t=1`, column 5 (`bo_ent`), abs diff 1.249e-16.
Divergence **persists** for the rest of the path and does **not** reconverge —
it is a persistent state perturbation, not a transient.

**`_BocpdStream` is a bitwise-exact re-expression of the batch algorithm.**
Its docstring claim ("bitwise == `_bocpd`", "fuzz-verified at atol=0") is
correct *at the source level*. The streaming rewrite is not the defect.

The entire disagreement is between **numba-compiled** and **CPython-interpreted**
execution of identical source.

## 4. Root cause

Probing each libm call used by the kernel, 200,000 random inputs each,
numba 0.67.0 `@njit(cache=True, fastmath=False)` vs CPython 3.11.6:

| operation | cells differing |
|---|---|
| `math.log` | 0 / 200,000 |
| `math.exp` | 0 / 200,000 |
| `math.log1p` | 0 / 200,000 |
| `a*b + c` (FMA contraction probe) | 0 / 200,000 |
| **`math.lgamma`** | **104,332 / 200,000** |

`math.lgamma` is the *only* divergent primitive. Example:
`x = 3.271197060801792` → numba `0x3feea358bed5c8a7`, CPython `0x3feea358bed5c8ac`.

`lgamma` appears in exactly two places in the whole feature bank:

* `src/sbr/features/m07_bayes.py:287` — batch `_bocpd`, building `ct`
* `src/sbr/stream/s_m07_bayes.py:397` — `_BocpdStream.__init__`, building `self.ct`

Both compute the same Student-t log-normalising constant table:

```
nu[r]  = 2.0 * (BO_AL0 + 0.5*r)                       # = 50 + r
ct[r]  = lgamma(0.5*(nu[r]+1.0)) - lgamma(0.5*nu[r]) - 0.5*log(nu[r]*pi)
```

This table is **80 data-independent constants**. Comparing the two:

* **57 / 80 entries differ**
* max abs diff 5.684e-14, max rel diff 6.173e-14, **max ULP distance 512**

`ct` enters the predictive log-likelihood of every run length at every step, so
a constant-table difference perturbs the entire posterior forever after — which
is exactly the observed signature (persistent from t=1, never reconverging).

**Root cause: the batch kernel's `ct` table is built by numba's `lgamma`, the
streaming kernel's by CPython's `lgamma`, and the two libm implementations
disagree by up to 512 ULP.**

## 5. Why this is a defect and not acceptable float noise

The batch path is not merely "another implementation" — it is the path that
**built the training data**. Checking the shipped feature cache
(`cache/features/m07_bayes.npy`, 5,036,517 x 50, the array RT-600 was fit on)
against both implementations:

| series | cache vs batch | cache vs stream | batch vs stream | cells |
|---|---|---|---|---|
| 5111 | **0** | 2 | 2 | 18,500 |
| 5002 | **0** | 1 | 1 | 17,850 |
| 8581 | **0** | 1 | 1 | 22,100 |
| 4746 | **0** | 2 | 2 | 43,850 |

The training cache matches the **batch** path bitwise and disagrees with the
**stream** path. `sbr.production.model` performs inference through
`StreamEngine` (`src/sbr/production/model.py:23,79`).

Therefore the frozen RT-600 model is **trained on batch features and served
stream features**. That is a train/serve skew: small, but a real engineering
defect with a definite correct answer, not a tolerance question.

## 6. Intended semantics (resolved without reference to any score)

Per `research/PROTOCOL.md` and `AGENTS.md`, intended semantics are resolved from
frozen production documents, the shipped artifact and earliest implementation
history — never by which variant scores better. No TS-AUC was computed for
either variant, and none is needed:

* `research/FINAL_ARCHITECTURE_FREEZE.md` §2 freezes the 500-column feature bank
  and pins `feature_manifest_sha256 = 1646c3b9...cced`.
* The frozen model artifact was fit on `cache/features/*.npy`, produced by
  `sbr/features/driver.py` calling the **batch** module functions.
* `_BocpdStream`'s own docstring states its design contract is to be **bitwise
  equal to `_bocpd`** — the streaming class was written to mirror the batch
  kernel, so the batch kernel is by construction the reference.

**The batch values are the intended semantics. The streaming implementation must
be corrected to reproduce them.** The fix direction is fixed by the frozen
artifact, not chosen.

## 7. Required fix

Make both implementations obtain the `ct` table from a **single source of
truth** compiled the same way, so that parity holds whether or not numba is
present (`m07_bayes.py` degrades to a no-op `njit` when numba is unavailable —
in that configuration both sides must equally use CPython's `lgamma`).

The correction is confined to the constant table. No change to the recursion,
the feature semantics, the calibration, the hazard constants, or any tuned
value.
