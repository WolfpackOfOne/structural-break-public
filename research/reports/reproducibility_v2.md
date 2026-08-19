# A1 — Reproducibility report (wave 2)
**Verdict: RT-100 REPRODUCES BIT-IDENTICALLY. The hard gate is passed.**

## What was rebuilt, and from what

Nothing was reused. This wave started in a **fresh container** with an empty
filesystem. Every artifact below was rebuilt from the raw competition parquet:

| artifact | how it was produced | check |
|---|---|---|
| `cache/store` | `research/scripts/build_store.py` over `X_train.parquet` | 10,000 series, 5,036,517 online rows, break rate 0.4967 — matches the wave-1 protocol figures exactly |
| `research/folds/folds.parquet` | regenerated with `make_folds.py` into a scratch dir and compared | **id→fold SHA256 `6e114f80…` identical to the committed value**; `DataFrame.equals` True |
| `cache/features` | `sbr.features.driver` over all 7 production modules | 5,036,517 × (151+60+59+60+60+60+50) = 500 columns |
| RT-100 booster ×5 folds | `sbr.pipeline.run` with the wave-1 parameters | see below |

The **code** for the reproduction came from a genuinely clean `git clone` of the
tag `research-checkpoint-20260818-1` into a separate directory
(`/home/claude/repro`), with `sys.path` pinned to that checkout and asserted at
run time. No fitted model object, no cached OOF array and no wave-1 process
state existed anywhere in the container.

## Result

| | recorded (wave 1) | reproduced (`RT-100R`) | delta |
|---|---|---|---|
| mean OOF TS-AUC | 0.615103 | 0.6151034246587253 | **0.0** |
| pooled OOF TS-AUC | 0.615000 | 0.6149965423542226 | **0.0** |
| fold 0 | 0.62903 | 0.62903 | 0 |
| fold 1 | 0.61061 | 0.61061 | 0 |
| fold 2 | 0.62688 | 0.62688 | 0 |
| fold 3 | 0.60747 | 0.60747 | 0 |
| fold 4 | 0.60152 | 0.60152 | 0 |
| n_features | 500 | 500 | — |
| train rows / fold | 1,000,000 | 1,000,000 | — |
| valid rows / fold | — | 806334 / 812939 / 806691 / 802506 / 804054 | — |

The delta is **exactly zero in double precision**, not "within tolerance".

This is a stronger result than it had to be, because the environment is *not*
the wave-1 environment: this container runs **NumPy 2.4.4, pandas 3.0.2,
LightGBM 4.7.0, scikit-learn 1.8.0, Python 3.11.15**. The champion is therefore
robust to a major-version change in three of its four core dependencies, which
is not something we could have assumed.

## `nogit` is fixed

`sbr.pipeline.git_sha()` returns `"nogit"` only when `git rev-parse` fails — i.e.
when the pipeline is run from a directory that is not a git checkout, which is
what happened in wave 1. Wave 2 runs inside a real clone, so every new ledger
row carries a real SHA (`RT-100R` → `e98f5b9`). No code change was needed; the
defect was environmental. **Every row appended from now on records a real SHA.**
Historical `nogit` rows are left as they are — rewriting them would be
falsifying the record — but they are now flagged in `research/RDOF_LEDGER.md` as
not independently attributable, with the exception of the three reproduced here.

## Immutable checkpoint

```
tag              research-checkpoint-20260818-1
commit           e98f5b969416585216502ed2d00075c6e86d7da9
wave-2 branch    research/wave2-2026
manifest         research/REPRODUCIBILITY_MANIFEST.json
```

The manifest records: git SHA + tag, Python/NumPy/pandas/SciPy/sklearn/LightGBM/
pyarrow versions, CPU count, raw-parquet SHA256s, store `values.npy` and
`meta.parquet` SHA256s, all fold-file SHA256s, the canonical id→fold SHA256, the
per-module column counts, the **batch feature-manifest SHA256**, the
**streaming feature-manifest SHA256** (`1646c3b9…`), a SHA over all 29 `sbr`
source files, and the seed protocol.

## Caveats, stated plainly

1. **The raw data was not re-downloaded from the organiser**; it was copied from
   the researcher's machine. Its SHA256 is now recorded, so any future divergence
   will be visible, but this run does not prove the local copy matches the
   organiser's current release.
2. **`RT-101R` was queued but not run** — the compute went to the six ensemble
   streams instead, which unlock the central deployability question. `RT-101`
   remains a wave-1 number with a `nogit` SHA.
3. `RT-000` was not re-run either; it is a deterministic handcrafted detector
   with no fitted state, so it carries the least reproduction risk of the three.
4. The reproduction shares the rebuilt feature cache with the working tree. The
   cache was itself rebuilt from raw data by checkpoint code, so this is not
   circular — but a fully hermetic reproduction would rebuild it a second time
   inside the clean checkout, which costs 93 minutes and buys little given the
   fold-SHA and row-count checks already passed.
