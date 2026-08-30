# Grok M4 — Explicit-Duration Absorbing Filter

Date: `2026-08-30 14:34:08`  ·  git `6ae2177`

The screen established the premise: excursion dwell is not memoryless. This is
the filter that premise licenses, run through Grok's Experiment C.

## What was implemented

, stated precisely so the report does not overclaim. This is
an explicit-duration two-component model of the CURRENT elevated episode, solved
in closed form, not a full multi-state HSMM forward pass. The duration
distribution is explicit and non-geometric, which is the entire scientific
content of M4; what is not modelled is transitions between multiple elevated
regimes. Given an ongoing elevated episode of length r:

    p(absorbing | r) = pi_A S_A(r) / (pi_A S_A(r) + (1 - pi_A) S_T(r))

S_T is this series' own empirical survival of historical excursion lengths, so
the transient prior is per-series and causal. S_A is a Weibull survival with
increasing hazard and globally frozen hyper-parameters -- never fitted to y on
the scored fold. The emission magnitude cancels between the two components
because both condition on "elevated", which is deliberate: magnitude is what
m07_bayes already prices, and duration is the object M4 claims is missing.

Frozen hyper-parameters, never fitted to `y` on any fold: `weibull_shape=1.5`, `weibull_scale=512.0`, `prior_absorbing=0.05`, `smooth_window=64`.

Dev coverage: `0.8424`.

## Pre-conditions

| gate | observed | requirement | passed |
|---|---:|---|---|
| posterior not monotone in `ab_fast` | 0.137037 | \|rho\| <= 0.90 | True |
| disagreement has univariate cell signal | 0.539881 | >= 0.52 | True |

Univariate dominant-cell AUCs:

| channel | AUC |
|---|---:|
| hsmm_post_absorbing_cell | 0.558605 |
| hsmm_logbf_vs_geometric_cell | 0.543333 |
| disagreement_vs_ab_fast_cell | 0.539881 |
| disagreement_vs_ab_fast_mature_vs_never | 0.542571 |
| ab_fast_cell | 0.601677 |

## Experiment C

Three fold-pure specialists on the same `m00_core + m07_bayes` base, differing
only in what is appended. The dwell-scalar arm exists so that any gain is
attributed to the posterior rather than to re-adding dwell in any form.

| arm | marginal vs clone | positive folds | standalone whole | standalone cell |
|---|---:|---:|---:|---:|
| control_m00_m07 | +0.001028 | 3/5 | 0.604067 | 0.641663 |
| dwell_scalar | +0.000545 | 3/5 | 0.602496 | 0.641756 |
| hsmm | +0.001149 | 4/5 | 0.604288 | 0.643385 |

| contrast | value |
|---|---:|
| hsmm_minus_dwell_scalar | +0.000604 |
| hsmm_minus_control | +0.000122 |
| dwell_scalar_minus_control | -0.000482 |
| hsmm_minus_control_positive_folds | +2.000000 |

Per-fold increment over the `{m00 + m07}` control: +0.001556, -0.000043, -0.000585, +0.000742, -0.001060 (2/5 positive).

### Why the primary contrast is the increment, not the absolute marginal

Both conditions are satisfiable by a channel that adds nothing. The absolute marginal is carried by the base specialist (+0.001028 with no dwell channel), and the dwell-scalar comparator is negative (-0.000482), so clearing it by +0.0005 requires only not being harmful.

Experiment C is specified as "specialist on {m00 + m07_old + m07_hsmm} vs
{m00 + m07_old}", so the increment is the contrast the experiment names. Both of
the spec's numeric conditions are reported above for completeness, and both are
met (`1.000000`), but
they are not the gate.

## Verdict

KILL. The filter works and the premise holds -- the posterior is not monotone in ab_fast (rho 0.1370) and carries real univariate cell signal (0.558605) -- but it adds +0.000122 over the {m00 + m07} control at only 2/5 positive folds, against a +0.000500 requirement. Note that the spec's two literal conditions ARE met (marginal +0.001149 >= +0.0010; over dwell scalar +0.000604 >= +0.0005), and they are misleading here: the absolute marginal is carried by the base specialist, which scores +0.001028 with no dwell channel at all, and the dwell-scalar comparator is negative (-0.000482), so beating it requires only not being harmful. The dwell premise is real; the posterior does not convert it into ensemble marginal.

