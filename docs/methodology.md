# Methodology

This document explains the ideas behind the detectors in this repository in plain
language. It is meant to be readable by students and non-specialist reviewers, not
only time-series experts.

## What is a structural break?

A **structural break** is a point in time where the process generating a series
changes. The series may shift its mean, change its volatility, or alter the
relationship between variables. In finance and economics these correspond to
regime shifts, policy changes, market-stress episodes, or volatility transitions.

A simple example: a sensor reads around 10 for a month, then jumps to around 15 and
stays there. The jump is a structural break in the mean.

## Classification vs. change-point detection

There are two distinct ways to frame the problem, and this repository includes
both.

- **Classification (supervised).** Treat each observation as a labelled example
  (`has_structural_break` = 0 or 1) and train a model to predict the label from
  features. This needs labelled training data and learns whatever patterns
  correlate with the label. It is *not* inherently a model of "the process
  changed" — it is a generic classifier applied to the task.
- **Change-point detection (unsupervised / statistical).** Look directly for the
  point(s) where the statistical properties of the series change. These methods
  need no labels and are purpose-built for the question "did the data-generating
  process change, and where?"

The change-point detectors are the more domain-appropriate tools; the ML baseline
is a useful, familiar reference point.

## Baseline feature engineering

The ML baseline (`create_baseline_features`) derives a handful of local features
from the raw `value` column:

- `rolling_mean_3`, `rolling_std_3` — local level and local variability.
- `lag_1`, `lag_2` — recent past values.
- `diff_1`, `diff_2` — first and second differences (local rate of change).

These give a `RandomForestClassifier` enough local context to separate "stable"
from "post-break" rows on simple data. The series is sorted by `timestamp` and
treated as a single ordered sequence; if you have multiple independent series,
group by the series identifier before engineering features so windows do not leak
across series boundaries.

## CUSUM intuition

CUSUM (cumulative sum) tracks how far the series has drifted from its overall mean.
Define `S_k = sum_{i<=k} (x_i - mean)`. If the mean is constant, `S_k` wanders
around zero. If the mean shifts up partway through, the deviations become
systematically positive after the shift, so `S_k` trends in one direction and
reaches an extreme at the change point.

The detector estimates the change point as `argmax |S_k|` and reports a break there
when the normalised statistic `max|S_k| / (sigma * sqrt(n))` exceeds a threshold.
It is simple, interpretable, and almost parameter-free — but it finds only the
single most prominent mean shift and ignores changes in variance.

## Rolling z-score intuition

The rolling z-score is a transparent local-anomaly detector. For each row it
computes how many standard deviations the current value sits from a trailing
window's mean. When a series steps to a new level, the first values in the new
regime are far from the trailing mean of the old regime, so their z-scores spike —
flagging the *transition*. Once the window fills with new-regime values, the
z-score returns to normal.

This makes the rolling z-score good at spotting transitions and isolated outliers,
but it does not label an entire regime, and it can fire on one-off spikes that are
not true structural breaks.

## PELT / change-point segmentation intuition

PELT (Pruned Exact Linear Time) frames detection as **segmentation**: split the
series into contiguous segments so that each segment is well described by a simple
model (here, a constant mean with `l2` cost), while paying a fixed **penalty** for
each additional break. Too few breaks leaves obvious shifts unmodelled; too many
overfits noise. The penalty controls that trade-off; a larger penalty yields fewer
breaks.

PELT naturally handles **multiple** change points, which is its main advantage over
the single-shift CUSUM estimator. It is backed by the
[`ruptures`](https://centre-borelli.github.io/ruptures-docs/) library.

## Evaluation considerations

Change-point methods return break *locations*, not a clean label per row, so we
score them with **point-based, windowed** metrics:

- A predicted break is a **true positive** if it lies within a tolerance window of
  a true break.
- Each true break can be matched at most once (greedy nearest match).
- Precision, recall, and F1 follow from the true-positive / false-positive /
  false-negative counts.

Why a tolerance window? Detectors legitimately localise a break a few steps early
or late (CUSUM, for instance, reports the accumulation point). Demanding exact
index equality would penalise essentially-correct detections. The tolerance makes
the comparison fair across methods with different localisation behaviour.

## Common failure modes

- **Variance-only changes.** Mean-based detectors (CUSUM, PELT with `l2`) miss a
  shift in volatility when the mean is unchanged; the rolling z-score may still
  catch the transition. See the synthetic `variance` dataset.
- **Multiple breaks.** CUSUM reports only the dominant break; prefer PELT when
  several regimes are expected.
- **Penalty / threshold sensitivity.** PELT's penalty and CUSUM's threshold trade
  precision against recall; defaults here are tuned for clear synthetic shifts and
  should be re-tuned for real data.
- **Cross-series leakage.** Concatenating independent series without grouping
  creates spurious "breaks" at the joins.
- **Tiny samples.** Rolling and lag features, and segmentation costs, are unstable
  on very short series; treat small-sample results with caution.
