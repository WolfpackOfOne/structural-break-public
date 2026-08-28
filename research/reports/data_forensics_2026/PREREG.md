# DATA_FORENSICS_2026 -- PREREGISTRATION

Date: 2026-08-28
Branch: `research/data-forensics-2026`
Base: `research/deep-ensemble-frontier-local-2026@21b2bae`
Agent: `codex-data-forensics`

## Scope

This is a descriptive data/OOF anatomy program. It trains no model, creates no
feature module, tunes no threshold, writes no OOF vector, and consumes no RT ID.
It may use development labels only after this preregistration is committed.
It may not read test/reduced data. It may not summarize lockbox labels, lockbox
predictions, or lockbox values. All reported row-level diagnostics are restricted
to canonical dev folds `0..4`.

## Fixed Inputs

- Data/cache root: `../structural-break-learner-diversity-2026`.
- Current LOCAL OOF source: `../structural-break-deep-ensemble-frontier-local-2026/research/oof`.
- Incumbent OOF fallback source: `../structural-break-learner-diversity-2026/research/oof`.
- Folds: canonical `research/folds/folds.parquet`, using folds `0..4` only.
- Existing streams:
  - RT600: `RT-300`, `RT-410`, `RT-411`, `RT-412`, `RT-413`, `RT-414`, `RT-415`.
  - RT-1264 best-`k=5` hybrid streams: `RT-1255`, `RT-410`, `RT-1260`,
    `RT-1261`, `RT-1254`, `RT-1262`, `RT-415`.
- Calibration: frozen fold-pure `SCDF_NSEEN`, fit on the other four dev folds
  for each evaluated fold.
- Pair diagnostics: 128 same-`t` positive/negative pairs per time point, random
  seed `20260828`.
- Feature-cache audit: frozen 500-column RT600 bank
  `m00_core,m01_seq,m02_dist,m03_dyn,m04_resid,m06_loc,m07_bayes`; deterministic
  dev-row sample `dev_rows[::50]`.

## Frozen Diagnostics

### DF-01 Population and Label Anatomy

Report dev-only series counts, row counts, break prevalence, positive-row
prevalence, history/online length summaries, tau-position summaries, fold
balance, `t`-bin class balance, relative-online-position class balance, and
post-hoc phase counts. Phase labels are diagnostic only:

- `never_break`
- `prebreak_far` (`tau - t > 100`)
- `prebreak_near` (`0 < tau - t <= 100`)
- `postbreak_early` (`0 <= t - tau < 100`)
- `postbreak_mature` (`t - tau >= 100`)

### DF-02 RT600 vs RT-1264 Residual Anatomy

Reconstruct fold-pure calibrated RT600 and RT-1264 vectors from existing OOF
streams. Report overall TS-AUC and fixed slice TS-AUC/delta for:

- fold `0..4`;
- `t` bins `[0,49]`, `[50,99]`, `[100,199]`, `[200,399]`, `[400,699]`,
  `[700,999]`;
- relative-online-position bins `[0,.1)`, `[.1,.25)`, `[.25,.5)`,
  `[.5,.75)`, `[.75,.9)`, `[.9,1]`;
- `hist_len` tertiles and `online_len` tertiles computed on dev series only;
- fixed composite cells: `whole_dev`, `dominant_cell`, `mature_vs_never`,
  `mature_vs_prebreak`, `near_boundary`, and `late_never_vs_mature`.

No slice threshold may be changed after seeing results.

### DF-03 Pair-Flow Localization

Using the same fixed pair sampler, report RT-1264-vs-RT600 repairs, damage, net
pair lift, repair rate among RT600-wrong sampled pairs, and damage rate among
RT600-right sampled pairs for the DF-02 slices.

### DF-04 High-Score Mass and False-Positive Anatomy

Within each `t`, rank calibrated scores. For top-score percentiles `top_1`,
`top_5`, and `top_10`, report positive capture, never-break negative mass,
prebreak negative mass, and postbreak composition for RT600 and RT-1264. The
percentiles are frozen before scoring.

### DF-05 Raw Process Anatomy

For dev series only, standardize each online value by its own historical mean
and standard deviation. Report row-weighted summaries of `z`, `abs(z)`,
`1[abs(z)>2]`, and `1[abs(z)>3]` by phase and by `t` bin. This uses true `tau`
only as a post-hoc diagnostic label; it cannot become an online feature.

### DF-06 Feature Bank Redundancy / Missingness Audit

On the deterministic `dev_rows[::50]` sample, report for each of the 500 frozen
feature columns: finite rate, mean, standard deviation, and `p01/p50/p99`.
Also report per-module finite rate, near-constant feature count, participation
ratio effective rank, top eigenvalue share, mean absolute within-module
correlation, and mean absolute between-module correlations.

## Falsification / Action Conditions

This program cannot promote a model. Its conclusions are descriptive:

- If RT-1264's gain is concentrated in one fold, one late-time bin, or one
  length stratum, mark the CatBoost hybrid as fragile for deployment review.
- If RT-1264 repairs mature-vs-prebreak pairs while increasing never-break
  high-score mass, flag false-positive risk for deployment review.
- If feature-bank effective rank is low inside or across modules, record
  redundancy as a reason future research should prefer mechanism diversity over
  more columns in the same family.
- If raw process phase summaries show that near-boundary and mature-postbreak
  rows are not separable by simple standardized magnitude/tail occupancy, record
  that the remaining problem is not a one-dimensional amplitude anomaly.

## Outputs

- `research/reports/data_forensics_2026/DATA_FORENSICS_REPORT.md`
- `research/reports/data_forensics_2026/data_forensics_results.json`
- Compact CSV tables under `research/reports/data_forensics_2026/`

No `RESULTS.csv` row is written. `STATUS.md` is updated only if the active
research conclusion materially changes.
