# VALIDATION V2 — bias control after 78 adaptive experiments
**Research Director · wave 2 · binding on every agent and every future experiment.**

## 1. What has already been seen — the honest inventory

| data | series | how it has been used | contamination |
|---|---|---|---|
| dev folds 0–4 | 8,000 | **78 logged experiments**, most of them adaptive: features chosen, modules promoted, streams selected, blends compared | **HEAVY.** Any number computed here is a *discovery* number. |
| screen store (subset of folds 0–4) | 2,500 | the exploratory triage instrument for most of those 78 | heavy, and *inside* the dev folds — screen results are not independent of dev results |
| original lockbox (fold −1) | 2,000 | **inspected exactly twice**: RT-100 → 0.60791, RT-130 → 0.61214 | **SPENT for selection.** Two scalars only — see §5. |
| `X_test.reduced.parquet` | — | never read | frozen, stays frozen |

**There is no untouched subset of the training data left.** Every one of the
10,000 series has participated in a scored evaluation. This document does not
pretend otherwise, and no agent may describe any partition of the training data
as an "untouched holdout" from here on.

## 2. The three score concepts — use these words, always

1. **DISCOVERY score.** Computed on data that participated in choosing the thing
   being scored. Everything on folds 0–4 for any candidate whose features,
   modules or hyperparameters were picked using folds 0–4. Optimistically biased
   by an unknown amount. Useful for ranking candidates against each other under
   an identical protocol; useless as an estimate of competition performance.
2. **CONFIRMATION score.** Computed by a procedure in which candidate selection
   and evaluation are separated *within the run* — nested CV (§3), where for
   every outer fold `k` the entire selection pipeline is refit using only folds
   `≠ k`. This is the honest internal estimate.
3. **DEPLOYABLE score.** A confirmation score produced by an inference procedure
   that can actually execute under the Crunch streaming interface
   (`research/reports/runner_semantics.md`). A number that needs the test
   cross-section, or `n_online`, or a second pass, is **not** a deployable score
   and must be labelled **ORACLE / DIAGNOSTIC**.

A result may only be called "champion" if it has a deployable confirmation score.

## 3. Nested cross-validation — the confirmation protocol

Because there is no clean holdout, confirmation is bought with nesting, not with
fresh data.

```
for k in outer folds 0..4:                       # 5 outer folds, canonical partition
    inner = folds != k                           # 4 folds, 6,400 series
    (a) all feature selection, all thresholding, all calibration fitting,
        all blend construction, all hyperparameter choice
        happens using ONLY `inner`, with its own inner CV where a held-out
        score is needed
    (b) refit the selected pipeline on `inner`
    (c) score fold k, once
CONFIRMATION = mean over k of the fold-k scores  (report per fold, always)
```

Binding consequences:
- **Never** rank features on all five folds and then quote a five-fold CV number
  for the selected subset. That is the single most common way to launder
  selection bias, and it is what makes RT-140's 0.62689 a discovery number.
- **Never** fit a calibration map on fold `k`'s own score distribution. All
  calibration is cross-fitted: fold `k` is transformed by a map built on `≠ k`.
- **Never** choose an ensemble subset using the fold it is evaluated on.
- A hyperparameter search may run inside `inner` only. Any search that touches
  fold `k` invalidates fold `k` as confirmation for that candidate, permanently.

## 4. Alternative fold partitions — stability, not tuning

`research/scripts/make_folds_alt.py` builds three alternative grouped partitions
(`alt1/alt2/alt3`) of the *same* 8,000 dev series, each preserving the canonical
stratification (has_break × τ quartile × n_hist tertile × n_online tertile).
Label agreement with the canonical partition is ≈0.20, i.e. genuinely different.

Rules:
- The canonical partition is **never** regenerated or changed.
- Alternative partitions answer one question — *does this conclusion depend on
  the partition?* — and are reported as a **distribution**, never as a mean to
  be maximised.
- It is forbidden to run a candidate on alt partitions, pick the best, and
  report it. It is forbidden to add a fourth partition because the first three
  were unfavourable. Pre-declare which partitions a study will use.
- Alt partitions are for: RT-100 robustness, the ensemble gain, and major module
  ablations. Nothing else without Director sign-off.

## 5. The original lockbox — spent, and what that means precisely

The 2,000-series lockbox has been observed twice. Both observations were single
aggregate scalars confirming a decision already made; nothing was selected on it.
The information leak is therefore small in absolute terms — but it is not zero,
and the mandate for this wave is explicit:

> **The lockbox is closed. It may not be used for model selection, for deciding
> whether a new idea works, or for scoring RT-131 or any successor.**

One exception exists and requires explicit written authorisation from the
Research Director (Graham) **before** the score is computed: a **single,
pre-registered, final confirmation** of exactly one already-chosen deployable
production system, reported whatever it says. No agent may take this decision.
Until that authorisation exists, `lockbox_touched` stays `no` on every row.

## 6. Uncertainty and comparison

Point estimates are not evidence at this stage.
- **Paired series-level bootstrap** for every model comparison: resample whole
  *series* (not rows) with replacement, recompute TS-AUC for both models on the
  same resample, report the distribution of the *difference*. Report the CI and
  the fraction of replicates favouring the candidate.
- **Seed stability** for any promoted architecture: seeds {0, 1, 7, 42, 2026};
  report mean, std, min, max. The best seed is never selected.
- **Fold stability**: canonical + alt1/alt2/alt3, reported as four numbers.

## 7. Research degrees-of-freedom ledger

`research/RDOF_LEDGER.md` records, cumulatively: hypotheses tested, model
architectures tried, feature variants, hyperparameter studies, and blends
compared. Promotion thresholds tighten as the ledger grows:

| stage | data | bar |
|---|---|---|
| SCREEN (feature triage only) | screen store, fold 0 | +0.002 vs its own paired control, or a strong complementarity argument |
| FULL DEV (discovery) | canonical 5-fold | positive on a **majority of folds**, not just on the mean |
| PROMOTION | nested confirmation | paired bootstrap CI materially favourable, **or** a demonstrated ensemble delta under a *deployable* blend |
| ARCHITECTURAL CHANGE | nested confirmation + alt-fold + seed sweep | all of the above, and stable across partitions and seeds |

Screen results may triage **features**. Screen results may **not** promote
objectives, architectures, gating, stacking, model families, or target changes —
three of three such screen wins reversed at full scale in wave 1.

A claimed improvement smaller than +0.001 after this many experiments is not a
result. Say "indistinguishable" and move on.

## 8. Prohibited, without exception
test/future/fold leakage · reusing the spent lockbox · hindsight relabelling ·
selective reporting · cherry-picked folds or seeds · promoting from one fold ·
changing folds after seeing results · tuning on confirmation folds · comparing
models on different fold definitions · feature selection using validation rows ·
selecting an ensemble on its own evaluation fold · using the leaderboard as the
optimiser · any inference-time use of the test cross-section or `n_online`.
