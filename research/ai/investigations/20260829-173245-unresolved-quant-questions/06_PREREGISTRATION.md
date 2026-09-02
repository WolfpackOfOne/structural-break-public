# Experiment Preregistration

## Experiment ID

`ANALYSIS-ID-TBD` — allocate a non-colliding analysis ID from the ledger before execution. Do not reuse an `RT-*` ID.

## Scientific Question

Is the frozen CatBoost k=2 hybrid's canonical-partition gain over RT-600 reproducible from existing OOF artifacts, free of material score-level `n_online` association, and not wholly dependent on its strongest folds or known damaged relative-time slice?

## Hypothesis

The fixed CAT-413 + CAT-300 replacement has a small positive E2-E0 effect that does not derive from `n_online` correlation.

## Mechanism

CatBoost's ordered-boosting/split policy extracts complementary rankings from the same causal bank. Length-correlated scores or an effect confined to one fold/slice would contradict this interpretation.

## Baseline

`E0`: frozen seven-specialist RT-600 development ensemble, assembled exactly under CSA-04R's equal-weight, fold-pure `SCDF_NSEEN` procedure from `RT-300.npy` and `RT-410.npy`–`RT-415.npy`.

## Candidate

`E2(k=2)`: replace RT-600 slots 413 and 300 with CAT-413 (`RT-1254.npy`) and CAT-300 (`RT-1255.npy`). No other replacement, subset, weight, calibration, or threshold may be tried.

## Exact Inputs

- Worktree: `/path/to/workspace/structural-break-deep-ensemble-frontier-local-2026/`.
- OOF vectors: `research/oof/RT-300.npy`, `RT-410.npy`–`RT-415.npy`, `RT-1254.npy`, `RT-1255.npy`.
- Folds: `research/folds/folds.parquet`.
- Row labels, `t`, series ID, and `n_online`: the same frozen development-store metadata used by CSA-04R. Execution must record its resolved path and checksum. If it cannot be provenance-matched to the 5,036,517-position vectors, stop as `INVALID`.
- Reference: `research/reports/deep_ensemble_frontier_2026/local/CSA04R_REANALYSIS.md` at analysis SHA `98a1a9a` and its machine-readable output.
- Development population only. No lockbox, leaderboard, test labels, or test metadata.

## Exact Training Population

None. No model fitting or retraining is permitted. Prediction inputs must remain byte-identical.

## Exact Fold Logic

Use canonical outer folds 0–4. Preserve each vector's held-out-series OOF predictions. Do not refit, repartition, or impute.

## OOF Assembly

Require vector length 5,036,517 and identical finite-row masks. Recreate E0 and E2 exactly with CSA-04R assembly. E0 must equal 0.625811342 and E2-E0 must equal 0.002026321728670 within `1e-9`; otherwise halt as `INVALID`.

## Calibration

Use only the frozen fold-pure `SCDF_NSEEN` transforms and equal weights embodied by CSA-04R. Do not fit or alter calibration.

## Primary Endpoint

Paired whole-development TS-AUC difference `E2-E0` on common finite OOF rows, using official within-time pair weighting.

## Secondary Endpoints

1. Per-fold E2-E0 and folds positive.
2. Paired series-bootstrap CI and SE.
3. Within fixed `t` bands `{0–19,20–49,50–99,100–199,200–399,400+}`, AUC of `E0`, `E2`, and `E2-E0` for predicting `n_online` above the band-specific median.
4. E2-E0 excluding fold 3 and, separately, fold 4; descriptive sensitivity only.
5. E2-E0 in relative-time 10–25% and its complement.
6. Dominant-cell E2-E0 (`t>=200`, break age `>=100`).

## Evaluation Cells

Whole finite dev OOF; outer folds 0–4; six fixed `t` bands; dominant cell; relative-time 10–25% and complement.

## Primary Success Criterion

All must hold:

1. Exact assembly checks pass.
2. E2-E0 is positive in 5/5 folds.
3. Paired series-bootstrap 95% percentile CI excludes zero.
4. No E2 score-level `n_online` AUC is outside `[0.48,0.52]` in any fixed `t` band.
5. Excluding fold 3 and excluding fold 4 each leaves E2-E0 positive.

Success means **clean provisional canonical-partition alpha**, not alternate-partition, lockbox, or external confirmation.

## Failure Criterion

Any failed provenance or assembly check is `INVALID`. Otherwise, failure is any unmet primary-success condition. A purity failure overrides a positive aggregate delta.

## Fold Consistency Criterion

E2-E0 must be strictly positive in all five folds. Fold-3- and fold-4-excluded pooled sensitivities must each remain positive.

## Statistical Test

Paired series-level bootstrap percentile interval for E2-E0. No row bootstrap, multiplicity-driven selection, or post-hoc endpoint substitution.

## Bootstrap / Resampling Procedure

Resample series with replacement within outer fold using multinomial counts; retain all rows of sampled series; recompute exact weighted TS-AUC. Use `B=10,000`, seed `20260829`; report observed delta, bootstrap SE, and 2.5/97.5 percentiles. This independently audits CSA-04R's B=2,000 result.

## Leakage Checks

- Confirm no lockbox/test inputs are loaded.
- Verify vector length, common finite mask, fold alignment, and series-row alignment.
- Run fixed score-to-`n_online` tests by `t` band.
- Confirm no calibration or threshold is fitted on evaluation labels.
- Record checksums of all prediction, fold, and metadata inputs.

## Required Ablations

No model ablation or training. Fixed sensitivities: exclude fold 3; exclude fold 4; relative-time 10–25% versus complement; dominant cell versus whole population. None may select a revised candidate.

## Expected Compute

CPU-only OOF arithmetic, minutes to a few hours, dominated by 10,000 bootstrap replicates. Zero training and leaderboard calls.

## Expected Output Artifacts

- JSON with checksums, regression checks, endpoints, fold results, bootstrap summary, purity tables, and status.
- Human-readable report with the same results and deviations.
- No modified OOF vectors, production files, model files, or historical ledgers.

## Decision Rule

- `POSITIVE_PROVISIONAL` only if every success condition passes.
- `NEGATIVE_OR_UNVERIFIED` if validly executed but any condition fails.
- `INVALID` for provenance, alignment, mask, or exact-reproduction failure.

No outcome authorizes training or promotion. A positive result preserves CatBoost k=2 as the leading existing-artifact candidate while leaving alternate-partition and external confirmation unresolved.

## What Result Would Falsify the Mechanism

Material score-level `n_online` association, a bootstrap interval including zero, a nonpositive fold, or loss of the aggregate effect when either strongest fold is excluded would make stable complementary learner bias unlikely. Slice concentration alone localizes but does not falsify the mechanism.
