# RAW_DATA_ATLAS_2026 -- PREREGISTRATION

Date: 2026-08-28
Branch: `research/raw-data-atlas-2026`
Base: `research/data-forensics-2026@6786096`
Agent: `codex-raw-data-atlas`

## Scope

This is a dev-only visual/data atlas of the actual stored time-series values.
It trains no model, creates no feature module, tunes no threshold, writes no OOF
vector, consumes no RT ID, and appends no `RESULTS.csv` row.

The atlas may use true `tau` only as an offline diagnostic annotation in plots
and tables. It may use existing OOF predictions only for deterministic example
selection and score overlays. It must not read test/reduced data and must not
summarize lockbox labels, lockbox predictions, or lockbox values. All plotted
series and row summaries are restricted to canonical development folds `0..4`.

## Fixed Inputs

- Data/cache root: `../structural-break-learner-diversity-2026`.
- OOF source for current LOCAL artifacts:
  `../structural-break-deep-ensemble-frontier-local-2026/research/oof`.
- OOF fallback source: `../structural-break-learner-diversity-2026/research/oof`.
- Raw values: `cache/store/values.npy`, sliced only for dev-fold series.
- Metadata: `cache/store/meta.parquet` and canonical `research/folds/folds.parquet`.
- Existing calibrated scores:
  - RT600 streams: `RT-300`, `RT-410`, `RT-411`, `RT-412`, `RT-413`, `RT-414`, `RT-415`.
  - RT-1264 streams: `RT-1255`, `RT-410`, `RT-1260`, `RT-1261`, `RT-1254`, `RT-1262`, `RT-415`.
- Calibration: fold-pure `SCDF_NSEEN`, fit on the other four dev folds for
  each evaluated fold.

## Fixed Raw Normalization

For each selected dev series, plot:

- last 300 historical observations, standardized by that series' historical
  mean and standard deviation;
- full online segment, standardized by the same historical mean and standard
  deviation;
- true `tau` as an offline vertical annotation for break series;
- optional calibrated RT600 and RT-1264 score overlays for model-anchored
  examples.

No online statistic or plotted value may use future rows for normalization.

## Fixed Shape Metrics

For every dev series:

- `online_mean_z`, `online_abs_z_mean`, `online_max_abs_z`, `online_tail2_rate`,
  `online_tail3_rate`;
- phase means/rates for `never_break`, `prebreak_far`, `prebreak_near`,
  `postbreak_early`, `postbreak_mature`, using the same phase definitions as
  `DATA_FORENSICS_2026`;
- if a break series has at least 25 rows before and after `tau`, compute
  `pre100` and `post100` windows capped at 100 rows each and report:
  `level_shift_100 = post_mean_z - pre_mean_z`,
  `scale_log_100 = log((post_std_z + 1e-6) / (pre_std_z + 1e-6))`,
  `tail2_shift_100 = post_tail2_rate - pre_tail2_rate`.

Break-shape family is frozen as:

- `boundary_limited` if either side of `tau` has fewer than 25 rows;
- `subtle` if `abs(level_shift_100) < 0.20`, `abs(scale_log_100) < 0.20`, and
  `abs(tail2_shift_100) < 0.03`;
- otherwise whichever of `abs(level_shift_100)`, `abs(scale_log_100)`, and
  `4 * abs(tail2_shift_100)` is largest, with sign retained as
  `level_up`, `level_down`, `scale_up`, `scale_down`, `tail_up`, or `tail_down`.

## Fixed Example Selection

Within each gallery, ties are broken by ascending `series_id`. If the top series
has already been selected for that gallery, take the next series under the same
rule.

### Raw Archetype Gallery

Select one dev series for each:

- `never_stable`: never-break, `n_online >= 400`, smallest distance to the
  never-break median `online_abs_z_mean` and `online_mean_z`.
- `never_tail_outlier`: never-break, largest `online_max_abs_z`.
- `never_level_drift`: never-break, largest `abs(online_mean_z)`.
- `prebreak_near_tail`: break series with at least 25 near-prebreak rows,
  largest `prebreak_near_abs_z_mean`.
- `postbreak_level_up`: break series with at least 50 mature-postbreak rows,
  largest `postbreak_mature_mean_z`.
- `postbreak_level_down`: break series with at least 50 mature-postbreak rows,
  smallest `postbreak_mature_mean_z`.
- `postbreak_scale_tail`: break series with at least 50 mature-postbreak rows,
  largest `postbreak_mature_abs_z_mean - prebreak_far_abs_z_mean`.
- `postbreak_subtle`: break series classified `subtle`, smallest shape strength.

### Model-Anchored Raw Gallery

Select one dev series for each:

- `rt1264_mature_lift`: break series with at least 50 mature-postbreak rows,
  largest mean `RT1264 - RT600` calibrated score delta on mature-postbreak rows.
- `rt1264_mature_drop`: same population, smallest mean score delta.
- `rt1264_never_score_reduction`: never-break series, smallest mean score delta.
- `rt1264_never_score_increase`: never-break series, largest mean score delta.
- `early_rel_loss`: at least 20 rows in relative-online-position `[.10,.25)`,
  smallest mean score delta there.
- `early_rel_gain`: same population, largest mean score delta there.
- `near_boundary_lift`: at least 40 near-boundary rows, largest mean score delta.
- `near_boundary_drop`: same population, smallest mean score delta.

## Fixed Figures

- `figures/raw_archetype_gallery.png`: standardized raw traces only.
- `figures/model_anchored_gallery.png`: standardized raw traces plus RT600 and
  RT-1264 score overlays.
- `figures/break_shape_bands.png`: median and IQR standardized online value
  around `tau`, offsets `[-100, +199]`, by break-shape family.
- `figures/phase_absz_box.png`: deterministic sampled `abs(z)` distributions
  by offline phase.
- `figures/raw_score_links.png`: descriptive series-level raw-shape metrics
  against RT-1264 minus RT600 score deltas.

## Outputs

- `research/reports/raw_data_atlas_2026/RAW_DATA_ATLAS_REPORT.md`
- `research/reports/raw_data_atlas_2026/raw_data_atlas_results.json`
- `research/reports/raw_data_atlas_2026/series_raw_metrics.csv`
- `research/reports/raw_data_atlas_2026/selected_series.csv`
- `research/reports/raw_data_atlas_2026/shape_family_summary.csv`
- `research/reports/raw_data_atlas_2026/raw_score_correlations.csv`
- the fixed PNG figures above

## Interpretation Rules

This program is descriptive only. It cannot promote `RT-1264`, reject it, or
authorize a router/threshold/feature. If the atlas shows visually separable raw
regimes, record them as hypotheses for future preregistration. If it shows that
repair/damage examples look ambiguous in raw space, record that as evidence
against simple hand-built raw rules.
