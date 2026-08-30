# CATBOOST SPECIALIST ACTIVATION 2026 — COMPLETE

Date: 2026-08-27
Branch: `research/catboost-specialist-2026`
Scoring SHA: `a360dfd`

| arm | RT ID | standalone | rho | marginal_vs_clone | E2-E0 | folds positive | dominant net | runtime | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| CAT-413 | `RT-1254` | 0.620440244 | 0.714196 | +0.001087151 | +0.001244347 | 5/5 | 93 | 1527.1s | INTERESTING |
| CAT-300 | `RT-1255` | 0.620242061 | 0.737786 | +0.001029459 | +0.001126876 | 5/5 | 75 | 2208.1s | INTERESTING |
| CAT-410 | `RT-1256` | 0.610328251 | 0.680825 | +0.000753095 | +0.000929590 | 4/5 | 17 | 1132.1s | KILL |
| HYBRID | `RT-1257` |  |  | +0.002407205 | +0.002026322 | 5/5 | 158 | 0.0s | PROMOTION_WORTHY |

CSA-00: E2-E1 `+0.000975838`; E2-E0 `+0.001002808`; dominant net vs E0 `70`.
Best specialist: `CAT-413` with `marginal_vs_clone = +0.001087151`.
Hybrid: run, verdict `PROMOTION_WORTHY`, marginal_vs_clone `+0.002407205`.
Best measured gain over original RT600: `+0.002026322`.
Best marginal_vs_clone: `+0.002407205`.
Production feasibility: Training was 4867.3s for three CatBoost specialists in this run; inference adds one CatBoost model per surviving replacement and needs separate deployment benchmarking.
Next action: review surviving CatBoost specialist(s) against deployment cost before any production work.

No lockbox, test, production, feature, router, stacker, hyperparameter, seed, or blend-weight change was made.
