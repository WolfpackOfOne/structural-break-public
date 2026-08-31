# Negative Results Index

Fast searchable index of major failed, invalid, redundant, superseded, or infeasible directions. This is not a replacement for `FAILED_EXPERIMENTS.md`; detailed evidence remains there and in the linked program reports.

| Family | Experiment / mechanism | Outcome | Why it did not promote | Retry guidance / nuance |
| --- | --- | --- | --- | --- |
| Sequential | slope-of-evidence channels | DEAD | zero LightGBM gain; noisy/redundant with current-minus-decayed-peak information already present | Low priority; only as replacement with much longer lag |
| Sequential | hard threshold cross-channel count `xc_n_hot` | DEAD | thresholding destroyed rank information a tree could exploit directly from continuous features | No |
| Sequential | detector current-level channels | REDUNDANT | levels have standalone signal but add almost nothing conditional on `m00_core`; recursive peaks carry the useful new memory | Do not re-add as ordinary extra columns |
| Sequential | short EWMA level bank | MOSTLY REDUNDANT | rectangular multi-scale bank plus length-matched null calibration already covers the information better | Only long/variance forms with better calibration if justified |
| Sequential | mixture GLR | DEAD | smoothed restatement of max GLR; lost tail resolution | No in current form |
| Process | mixture-GLR global normalizer | INVALID implementation | depended on total online length; prefix-invariance caught look-ahead | Fixed historically; lesson is mandatory causality harness before scoring |
| Distribution | RT-063-tl tail asymmetry | KILL / negative | sparse/noisy tail rates largely repeat `m00_core` tail monitoring | Only conditional disambiguator if a promoted divergence family warrants it |
| Distribution | RT-063-rk rank-CUSUM family | KILL / negative | strong univariate rank information but redundant with expanding PIT mean/dispersion channels | Not standalone |
| Dynamics | RT-071C trend family | KILL | noisier linear restatement of level shifts; removing it improved the full module | No ordinary revival |
| Wave 5 objectives | RT-700 pairwise_w | REJECTED | worse than matched pairwise_t control | No without a new mechanism-level hypothesis |
| Wave 5 objectives | RT-701 pairwise_h | REJECTED | materially worse than control | No |
| Wave 5 features | RT-740 `m10_persist` | REJECTED | negative marginal vs seed clone | No simple revival |
| Wave 5 combination | RT-760 all three feature blocks | REJECTED | combination weaker than strongest single block | Do not assume more feature blocks = more alpha |
| Wave 6 oracle | RT-900 `w6oracle` | INVALID | missingness pattern leaked the break label; 0.8655 was not alpha | Never reuse; ID retired |
| Teacher distillation | RT-994 T1 | NO PROMOTION | mean improvement but only 3/5 positive and uncertainty crossed zero | Superseded by T2 work |
| Teacher distillation | RT-995 T2 | PARKED, not KILL | strong standalone improvement but only ~+0.000237 marginal vs matched clone when added to RT-600 | Revisit only with a mechanism for extracting complementary residual information |
| Future-aware transfer | Wave 8 ORR | KILL | no sufficient full-population marginal alpha | Closed with Wave 8 program |
| Future-aware transfer | Wave 8 TGMC | KILL | no sufficient full-population marginal alpha | Closed |
| Future-aware transfer | Wave 8 SST | KILL | no sufficient full-population marginal alpha | Closed |
| Future-aware transfer | Wave 8 PCFB | KILL | no sufficient full-population marginal alpha | Closed |
| Future-aware transfer | Wave 8 CFEP | KILL | no sufficient full-population marginal alpha | Closed |
| Causal representation | CRF-01 NNCSR / RT-1234 | KILL | standalone too weak and pair-flow evidence negative; program abandon gate fired | Do not revive same representation/objective |
| Causal representation | CRF-02 corrected ACGN family | KILL | fixed per-series null outperformed learned representation; no sufficient promotable model | Program exhausted |
| Leaderboard Alpha | LA-02 | KILL | large clone-relative headline but mandatory pair-flow gates negative | Do not promote on E2-E1 alone |
| Leaderboard Alpha | LA-03 / RT-1249 | KILL | +0.001676 clone-relative, 5/5 positive, but fixed-null isolation failed and pair flow was negative | No current survivor |
| New Avenues | first/second pilot sweeps | KILL / exhausted | no confirmation candidate survived the binding continuation rules | Historical mechanism library remains useful, but executed arms are not live |
| Learner diversity | RT-1250 TabM | INFEASIBLE, no predictive verdict | original frozen full-scale execution did not fit compute budget; shrinking/retuning forbidden | Later distinct GPU arm RT-1258 removed the hardware blocker and supplied the binding score |
| Learner diversity | RT-1252 RealMLP | INFEASIBLE, no predictive verdict | same original compute-contract problem | Later distinct GPU arm RT-1259 supplied the binding score |
| GPU tabular | RT-1258 TabM | KILL | standalone 0.60065 and rho 0.585 were not enough; marginal_vs_clone only +0.000040947, E2-E0 negative | Strong evidence that “different + >0.600 standalone” is insufficient |
| GPU tabular | RT-1259 RealMLP | KILL | standalone 0.55991; marginal_vs_clone -0.003223744; 0/5 positive | No |
| CatBoost | RT-1256 / CAT-410 | KILL | marginal_vs_clone +0.000753095, below +0.0010 gate | Not part of current survivor set |
| CatBoost composition | RT-1264 five-slot hybrid | SUPERSEDED, not KILL | original best-k selection used E2-E1 while clone control worsened with k; corrected CSA-04R selection changes the current conclusion | Preserve historical row; use RT-1265 corrected analysis |
| CatBoost composition | k=3 CAT-413+CAT-300+CAT-412 | NOT_DISTINGUISHABLE | nominal +0.000338234 over RT-1257 is below 0.0011 paired-bootstrap noise floor | CAT-412 remains the most interesting residual single-slot research question |
| Historical ensemble | RT-131 within-timestep rank average | ORACLE / ILLEGAL | uses cross-sectional information unavailable to independent streaming inference | Diagnostic only |
| Offline/full sequence | oracle/frontier models | ORACLE_NONCAUSAL | use future/full-sequence or known-boundary information | Information-bound diagnostics only |

## Important distinctions

**INFEASIBLE is not KILL.** RT-1250/RT-1252 never produced binding predictive results under their original frozen execution contract. RT-1258/RT-1259 were distinct GPU-authorized arms after hardware removed that blocker and are the actual scored KILL results.

**PARKED is not KILL.** RT-995/T2 contains real standalone teacher-distillation signal. It is parked because that signal is mostly redundant with the ensemble.

**SUPERSEDED is not KILL.** RT-1264 remains a historical experiment with real measured outputs. What is superseded is its use as the current selection conclusion: RT-1265/CSA-04R corrected the endpoint and selects RT-1257's two-slot composition.

## Rule for future agents

Before funding a mechanism that resembles an entry above, read the detailed section in `FAILED_EXPERIMENTS.md` and the linked final program report. A retry needs a new falsifiable reason the prior failure mechanism no longer applies; changing a seed or lightly retuning the same idea is not enough.
