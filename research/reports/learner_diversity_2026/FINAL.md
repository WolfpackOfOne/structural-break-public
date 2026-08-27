# LEARNER DIVERSITY 2026 — COMPLETE

Date: 2026-08-27
Branch: `research/learner-diversity-2026`
Base: `research/leaderboard-alpha-2026@d92860d`
Preregistration and scoring SHA: `c91950b`

No lockbox, test, production, feature, data-generation, router, stacker, LA-04,
or blend-weight change was made.

## Binding Result

| learner | IDs | standalone | rho vs LGBM | marginal_vs_clone | folds positive | dominant pair net | runtime | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| TabM | `RT-1250` | n/a | n/a | n/a | n/a | n/a | 0.0 s | INFEASIBLE |
| CatBoost | `RT-1251` | 0.620407665 | 0.736467811 | +0.001109800 | 5/5 | +88 | 10316.9 s | INTERESTING |
| RealMLP | `RT-1252` | n/a | n/a | n/a | n/a | n/a | 0.0 s | INFEASIBLE |

`RT-1251` CatBoost is the only scored learner and the best learner. It beats
the matched `RT-401` standalone control by `+0.003798521` and, more importantly,
beats the exchangeable `RT-401` seven-member replacement by
`marginal_vs_clone = +0.001109800` with all five folds positive.

## Ensemble Context

E0 original RT600: `0.625811342`.
E1 six RT600 specialists plus `RT-401`: `0.624980472`.
E2 six RT600 specialists plus CatBoost: `0.626090272`.

Fold deltas E2-E1: `+0.001910647`, `+0.000656868`, `+0.001327185`,
`+0.000519347`, `+0.001134955`.

Pair flow for E2 vs E0 is mixed but passes the dominant-cell direction check:
whole net `-1`, dominant-cell net `+88`, mature-vs-never net `+61`,
mature-vs-prebreak net `+61`.

## Combination

No combination was run. The preregistered combination opened only if at least
two learner-family candidates individually reached `marginal_vs_clone >=
+0.0010`. Only CatBoost reached that threshold; TabM and RealMLP were
full-scale infeasible under the frozen official/default configurations.
Conditional `RT-1253` remains unused.

## Interpretation

The strict "LightGBM-only is all there is" claim is closed: CatBoost produced a
measured non-LightGBM marginal above the preregistered INTERESTING threshold.

The deployable research lane is not promoted: the measured marginal is below the
SERIOUS threshold of `+0.0030`, and no combination opened. Production `RT-600`
remains unchanged.
