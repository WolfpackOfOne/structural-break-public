# Moonshot Kill Gate — Reconstructed Generator

Date: `2026-08-30 15:03:07`  ·  git `6ae2177`

Grok and Kimi independently proposed the same moonshot and independently
specified the same kill gate. This runs it once and discharges both:

- agent_05_grok46.md section 5 (SNPE / ABC-SMC on a reconstructed generator)
- agent_02_kimi_k3.md section 5 (DGP reverse-engineering -> simulation-based amortized inference)

**No amortized posterior was trained.** That is the purpose of the gate: the
expensive stage is entered only against a validated generator.

## Artifact battery

Bounded search over `36` simulator configurations. The simulator is
allowed to pick its best configuration against the real battery *before* the
transfer test, so a transfer failure cannot be blamed on untuned parameters.

| statistic | real | best simulator | tolerance | matched |
|---|---:|---:|---:|---|
| location_auc | 0.4760 | 0.4861 | 0.04 | yes |
| scale_auc | 0.5613 | 0.7666 | 0.06 | no |
| dependence_auc | 0.5424 | 0.5201 | 0.06 | yes |
| shape_auc | 0.5188 | 0.4888 | 0.06 | yes |
| ar6_resid_logsd_auc | 0.5663 | 0.5259 | 0.06 | yes |
| transient_rate_mean | 0.0813 | 0.0826 | 0.08 | yes |
| hist_kurt_median | 0.2119 | 0.4188 | 1.5 | yes |

Matched `6/7`; required `6`.

## Transfer test

Train on simulated series only, score on real fold 0. Series-level, because a
simulated corpus shares no row space with the real one — which is the protocol
both agents' kill gates name.

| model | AUC on real fold 0 |
|---|---:|
| trained on 1200 simulated series | **0.5321** |
| identical features and learner, trained on real data | 0.6117 |

Threshold `0.55`, row-level TS-AUC on `806334` real fold-0 rows.

The real-trained reference is the control that makes the result readable: it
separates 'these summary features are too weak' from 'the simulator is not the
DGP'. Only the second explanation kills the moonshot, and only the second is
consistent with a real-trained model doing well on the same rows and features.

## Verdict

KILL both moonshots. Battery match 6/7 (required 6); simulator-trained transfer AUC on real fold 0 is 0.5321 against the 0.55 threshold both agents set, while the identical features and learner trained on REAL data reach 0.6117 on the same rows, so the simulator transfers 0.29 of the achievable lift against a 0.60 requirement. The features are not the problem; the generator is not the DGP. The expensive amortization stage is never entered.

