# LA-03 -- Per-Series History Adaptation Execution Preregistration

Date: 2026-08-26

Program preregistration: `PROGRAM_PREREG.md` at `ed84d00`.

LA-02 result: final-gate KILL at `f4319d1`; LA-03 opens because LA-02 did not
become SERIOUS under the full preregistered gate.

## Allocated IDs

| ID | arm |
|---|---|
| `RT-1247` | LA-03 C0 global no-adaptation predictive-null head |
| `RT-1248` | LA-03 C1 fixed per-series AR(5)+history-residual-ECDF null head |
| `RT-1249` | LA-03 candidate global AR(5) plus per-series affine-adapted predictive-null head |

## Frozen Design

All arms train the same small LightGBM same-`t` pairwise ranking head on the
same ten causal residual/PIT features. The model never receives RT600 scores,
specialist scores, `tau`, post-break age, final online length, test data, or any
cross-sectional feature.

Feature rows are generated from a predictive null that prices `x_t` using only
`H_i` and `x_{<t}`; the current value `x_t` enters only as the realised residual
being priced.

Common row-level features, in fixed order:

1. `resid_z`
2. `resid_pit`
3. `abs_resid_pit`
4. `surp`
5. `run_mean_resid_z`
6. `run_mean_abs_resid_z`
7. `run_mean_surp`
8. `run_max_abs_resid_pit`
9. `exceed90`
10. `lag1_pit`

Training head:

- `wave2_train_ensemble.DEFAULT` LightGBM parameters
- `600` boosting rounds
- same-`t` pairwise objective from `sbr.pipeline._make_pairwise_t`
- `1,000,000` training-row cap per outer fold and arm, uniform sampled with seed
  `2026082603 + fold`
- full canonical folds `0,1,2,3,4` from the start

## Arms

C0 `RT-1247`: global no-adaptation control.

- Every series is standardised by its own history mean and standard deviation.
- One global AR(5) predictor is fitted per outer fold using only training-fold
  histories.
- The residual ECDF and residual scale are pooled from training-fold histories.
- No per-series predictive coefficient, residual-scale, or residual-ECDF
  adaptation is used.

C1 `RT-1248`: fixed per-series null control.

- Per-series AR(5) coefficients are fitted by Yule-Walker on `H_i`.
- The residual scale and residual ECDF are fitted on that same `H_i`.
- This is the fixed per-series AR(5)+history-residual-ECDF control family that
  beat the learned CRF null.

Candidate `RT-1249`: per-series affine-adapted predictive null.

- Starts from the same outer-fold global AR(5) predictor as C0.
- For each series, fit only a two-parameter affine adapter on `H_i`:
  `z_t = a_i * g_t + b_i + e_t`, where `g_t` is the strict-past global AR
  prediction.
- Ridge prior is fixed at `lambda = 32` toward `a_i = 1`, `b_i = 0`; fitted
  coefficients are clipped to `[-3, 3]`.
- Residual scale and residual ECDF are fitted from the adapted history residuals.
- No full independent per-series model is fitted.

## Isolation and Final Metric

Mandatory representation isolation gate, measured on standalone five-fold OOF
heads before reading the ensemble result:

- `RT-1249 - RT-1247 >= +0.0005`
- `RT-1249 - RT-1248 >= +0.0005`

If either isolation comparison fails, LA-03 is KILL regardless of ensemble
scores.

Final competition protocol:

- E0: existing RT600 seven-specialist ensemble.
- E1: RT600 plus `RT-1247` under the existing equal cross-fitted SCDF blend.
- E2: RT600 plus `RT-1249` under the same blend.

Primary metric:

`marginal_vs_clone = E2 mean TS-AUC - E1 mean TS-AUC`.

Also report candidate minus E0, E1 minus E0, C1 integrated diagnostic, fold
deltas, dominant-cell repairs/damage/net, mature-vs-never net,
mature-vs-prebreak net, within-`t` rank correlation with RT600, and runtime.

## Gate

LA-03 survives only if:

- both standalone isolation gates pass,
- five-fold `marginal_vs_clone >= +0.0015`,
- at least 4/5 folds have positive `E2 - E1`.

`>= +0.0030` with at least 4/5 positive folds is SERIOUS.

No LA-03b, width sweep, adapter-size sweep, alternate architecture, loss
variant, or RT600-score-conditioned version is authorised.
