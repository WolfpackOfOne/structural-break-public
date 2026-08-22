# C3 STAGE 1 — INSTRUMENT VALIDATION AND HARD STOP

**2026-08-22, branch `research/wave5-alpha`, pre-score SHA `95bd8dd`.**

> **C3 SCORING DID NOT START. TRAINING RUNS = 0. NEW TS-AUC VALUES = 0.**
>
> The 2026 feature store is not present in this container and no source for it
> was supplied. Under §10 of the C3 brief this is a hard stop, not something to
> work around.

---

## 1. WHAT WAS VALIDATED (everything the store is not required for)

Stage 1 asks whether the folds, the scorer, the calibration and the feature-store
semantics behave as expected. Three of the four could be checked without data,
and were.

### 1.1 Scorer parity — **PASS, exact to 1e-12**

`sbr.metric.ts_auc_flat` collapses the per-timestep AUC into a single global rank
ratio for speed. That is only legitimate if it reproduces the documented metric
exactly, so it was checked against a literal implementation — per online index
`t`, ROC AUC across eligible series via `sklearn`, timesteps with a single class
skipped, weighted by `n_pos * n_neg` — on the **real series geometry** (true
`tau_index`, `n_online` and `has_break` from `research/folds/folds.parquet`;
201,876 rows over 400 dev series, 54,451 positives, max `t` = 994).

| score pattern | `ts_auc_flat` | brute force | match |
|---|---|---|---|
| random | 0.501222408608 | 0.501222408608 | **yes** |
| informative | 0.920444401931 | 0.920444401931 | **yes** |
| heavy ties | 0.872772332667 | 0.872772332667 | **yes** |
| all identical | 0.500000000000 | 0.500000000000 | **yes** |

Tie-heavy and fully degenerate cases match too, which matters because the metric
uses mid-ranks and the hard-negative hardness estimator was just aligned to that
convention. Gate is permanent: `tests/test_scorer_parity_real_geometry.py`
(6 tests), which also asserts single-class timesteps carry **zero weight** rather
than silently counting 0.5, and that perfect and inverted scores bracket [0, 1].

### 1.2 Fold metadata — **PRESENT AND INTACT**

| file | rows | per-fold | note |
|---|---|---|---|
| `folds.parquet` | 10,000 | −1: 2,000 · 0–4: 1,617/1,609/1,601/1,592/1,581 | fold −1 is the spent lockbox; dev = 8,000 |
| `folds_alt1/2/3.parquet` | 8,000 each | identical per-fold counts | robustness partitions, stratification preserved |
| `folds_final10k.parquet` | 10,000 | 2,021/2,008/1,997/1,988/1,986 | **matches the recorded balance "1,986–2,021 per fold" exactly** |

`folds.parquet` carries `has_break`, `tau_index`, `n_hist`, `n_online`, so labels
and break ages are available for age-bucket analysis the moment scores exist.
Break rate on dev is 0.4969, consistent with the record.

### 1.3 Module causality — **PASS**

All three frozen candidates re-verified bitwise prefix-invariant at `atol = 0.0`
across 6 length profiles × 4 break kinds × 7 truncation points, with clean column
audits and exact-vs-brute-force checks on both changepoint scans.

### 1.4 What could NOT be validated

| item | why |
|---|---|
| ABL / seed-clone / specialist baseline reproduction | **needs the store** |
| calibration behaviour on real OOF | needs trained models |
| feature-store semantics | needs the store |
| RT-600 artifact causality gates (`test_no_n_online_leakage`, 4 tests) | `@needs_model`; `models/` holds build artifacts and is not in git. **They skip with an explicit reason, not silently** |

## 2. THE HARD STOP

| check | result |
|---|---|
| `cache/store/values.npy` | **ABSENT** |
| `cache/store/meta.parquet` | **ABSENT** |
| `SBR_STORE` / `SBR_ROOT` | unset |
| `/mnt/attach`, `/mnt/user-data`, `/media` | empty |
| filesystem-wide search for `values.npy` / `meta.parquet` / `X_train.parquet` | no hits outside site-packages |
| `api.crunchdao.com` | **403 at CONNECT** — network policy |
| source path supplied with the task | **none** |

`crunch-cli` installs from PyPI but has no reachable API, so the container cannot
self-serve. This is the same blocker recorded in `WAVE5_UNBLOCK_OPTIONS.md`.

**No substitute was used.** Per §10, a store that does not match its manifest —
or a different version of the data — is not an acceptable basis for scoring, and
fabricating one would make every downstream number meaningless.

## 3. WHAT UNBLOCKS STAGE 1

Only the primary store has to move: **140,145,984 bytes** of `values.npy` plus a
small `meta.parquet`. The ~10 GB feature cache is derived and rebuilds locally.

```
python research/scripts/wave5_ingest_store.py --values-url URL --meta-url URL
```

It refuses to certify anything that does not hash to `10c22b00…` /
`d09ec076…` at exactly that byte count, then prints the rebuild sequence. Disk
headroom is 21 GB against a ~10 GB cache; measured training cost on this box is
9.4–11.8 min/fold at 1M rows × 500 features, so the full C3 slate of ten
experiment-level runs is a matter of hours, not days.

## 4. STATE AT STOP

Everything that could be done before evidence has been done, and the pre-score
boundary is committed at `95bd8dd`:

* metric-aligned hard-negative hardness, fold purity strengthened, 17 tests;
* package-matrix deployability wording corrected;
* scorer proven exact on real geometry, 6 tests;
* three feature modules frozen and byte-identical.

**129 tests pass, 4 skip (artifact-level, explicit reason), 0 fail.**

The next action is data ingestion. Nothing else in C3 can proceed without it, and
nothing in C3 has been prejudged.
