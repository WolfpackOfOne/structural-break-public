# Negative Results Index

Fast searchable index of major failed, invalid, redundant, superseded, or infeasible directions. This is not a replacement for `FAILED_EXPERIMENTS.md`; detailed evidence remains there and in the linked program reports.

| Family | Experiment / mechanism | Outcome | Why it did not promote | Retry guidance / nuance |
| --- | --- | --- | --- | --- |
| Ensemble slots | Residual CatBoost specialists CAT-411 / CAT-412 / CAT-414 / CAT-415 (RT-1260 / RT-1261 / RT-1262 / RT-1263) | LANE CLOSED 2026-09-01 | RT-1257-relative adjudication put every deployment endpoint below the 0.0011 paired-bootstrap noise floor: CAT-412 +0.000338234 (the best of them), CAT-414 +0.000178858, CAT-415 +0.000175228, CAT-411 +0.000133942. CAT-412's optional alt-partition leg was scoped, costed at ~a day of partition plumbing, and declined against a sub-floor lift. | Only with a new mechanism, not a new partition. Reviving CAT-412 needs partition support in `deep_ensemble_local_2026.py` and `rt1257_slot_adjudication.py`, then the same RT-1257-relative E0/E1/E2 contract with a deployment endpoint above the floor. |
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
| Future-aware transfer | Wave 8 PCFB (RT-1030..RT-1033) — **linear bottleneck of the EXISTING 500-column bank**: 500 handcrafted causal features -> PCA(16) / PLS(16) -> LightGBM | KILL | PCA16 ~-0.00148 dominant-cell AUC, PLS16 ~-0.00270, against a seed-clone ensemble marginal of ~-0.00031 — i.e. materially worse than adding an ordinary extra model. PLS was worse than PCA despite explicitly seeking a future-predictive subspace. | **Scope matters: this falsified linear compression of the handcrafted bank, NOT dimensionality methods applied to the raw prefix as a detector.** For that distinct hypothesis see the Hankel-DMD row below (RT-1215), which tested it and also failed. Do not treat PCFB as covering both. |
| Future-aware transfer | Wave 8 CFEP | KILL | no sufficient full-population marginal alpha | Closed |
| Causal representation | CRF-01 NNCSR / RT-1234 | KILL | standalone too weak and pair-flow evidence negative; program abandon gate fired | Do not revive same representation/objective |
| Causal representation | CRF-02 corrected ACGN family | KILL | fixed per-series null outperformed learned representation; no sufficient promotable model | Program exhausted |
| Leaderboard Alpha | LA-02 | KILL | large clone-relative headline but mandatory pair-flow gates negative | Do not promote on E2-E1 alone |
| Leaderboard Alpha | LA-03 / RT-1249 | KILL | +0.001676 clone-relative, 5/5 positive, but fixed-null isolation failed and pair flow was negative | No current survivor |
| New Avenues | first/second pilot sweeps (aggregate) | KILL / exhausted | no confirmation candidate survived the binding continuation rules | Historical mechanism library remains useful, but executed arms are not live. Individual mechanisms are indexed separately below — the aggregate row alone made them invisible to lookup. |
| Representation / subspace | **Pilot 3 / IM3, `RT-1215` — Hankel-DMD observers on the RAW prefix.** Search aliases: PCA as a break detector, dynamic PCA, PCA on lag/Hankel embeddings, singular spectrum analysis (SSA), subspace-angle change, PCA reconstruction error, rolling eigenvalue / effective-rank shift, Koopman/DMD. Implementation: `src/sbr/features/m17_observers.py`, delay 16, rank 4, horizons 1 and 5, windows 32/64, fitted history-only and frozen. | KILL | marginal vs clone **+0.000226** (gate +0.0010) and **rho vs RT-600 = 0.8859**, failing the 0.85 redundancy gate; dominant-cell pair flow 920 repairs / 931 damage, net **-11**. Standalone was *better than the champion in the cell* (0.673238 vs 0.664277) and still added nothing — the subspace, reconstruction-error and effective-rank information is already present in the bank under other names. Causality fully verified (prefix check, deterministic replay, operator/subspace replay equal). | **This is the entry to read before proposing any PCA/SSA/subspace break detector.** The mechanism is filed under DMD, not PCA, which is why lookups for "PCA" miss it and land on W8-PCFB instead — a different hypothesis. Genuinely NOT covered: **kernel PCA** and other explicitly nonlinear variants; and this is one design point (delay 16 / rank 4), so per the pilot's own wording it "does not falsify all state-space or delay-embedding observer ideas". Weigh a revival against the two-sided result: high-rho channels like this one are redundant, and the one very low-rho channel (RT-1201, rho 0.38) was still net -1151. |
| Representation / observers | Pilot 3 / IM3, `RT-1214` — frozen Kalman/NIS observer (AR(2) state, history-only q x r grid) | KILL | marginal vs clone +0.000135 (gate +0.0010); rho vs RT-600 0.8836; dominant-cell net -98 | State-space observer residuals beyond `m04_resid` are redundant. Same caveat as RT-1215 on the family not being exhausted. |
| Excursion / persistence | Pilot 3, `RT-1201` — IM2 matched-length run null + dwell bank: matched-length run percentile, episode-mass percentile and **max-run growth proxy**, windows 32/64/128 | KILL | marginal vs clone **+0.000301** (gate +0.0010); dominant-cell pair flow 2021 repairs / 3172 damage, net **-1151** — it damages 1.6 pairs per repair in the cell carrying half the metric weight. **rho vs RT-600 = 0.3817**, the most decorrelated candidate the programme ever produced, and still net-negative. | **This closes the excursion-GROWTH-RATE channel**, not merely excursion length. `NEW_AVENUES_2026.md` identifies growth as the real discriminator (never-break p90 run 56->94 as t goes 200->1000; mature-break 94->239) and `RT-1201` is the arm that tested it. It is also the strongest single refutation of "decorrelation is the missing ingredient". |
| Wave 5 features / architecture | **W5-E11 — `m12_rdep` as an ARCHITECTURE rebuild** (`RT-751`, `RT-811`..`RT-816`): all seven specialist streams rebuilt verbatim with `m12_rdep` appended, nothing else changed | KILL / H0 ACCEPTED | `S'` 0.623133406 vs `S` 0.625811264 = **-0.002678**, 1/5 folds positive, paired bootstrap 95% CI [-0.00611, +0.00079], fraction positive 0.09. | **Do not revive on the "+0.00141 beat a seed clone" figure.** That compares against `B` (seed clones, 0.621640484), which is itself -0.00417 worse than the incumbent `S`; beating a degraded control is not evidence of value, and W5-E11's own preregistration ruled the clone control inapplicable to an architecture rebuild in advance. The floor moving 0.0030 -> 0.0011 is irrelevant: both are positive bars and the measurement is -0.00268. Full adjudication: `reports/rt1320_promotion/Q1_M12_RDEP_ADJUDICATION.md`. |
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

**A MECHANISM IS INDEXED UNDER ITS IMPLEMENTATION NAME, NOT ITS IDEA NAME.** This has
now caused two false "open lane" reports in one session. "PCA as a break detector"
is filed as Hankel-DMD (`RT-1215`); "excursion growth exponent" is filed as the IM2
dwell bank (`RT-1201`); "add `m12_rdep` to the bank" is filed as W5-E11. Searching
for the idea returns nothing, or returns a *differently-scoped* experiment that
shares the keyword — the W8-PCFB / `RT-1215` pair is the canonical trap, since both
involve dimensionality reduction but one compresses the handcrafted bank and the
other detects on the raw prefix. **Before concluding a lane is untested, search for
the implementation family (DMD, Koopman, observer, dwell, run-length, residual
distance), not the idea.** Add search aliases to any row you write.

## Rule for future agents

Before funding a mechanism that resembles an entry above, read the detailed section in `FAILED_EXPERIMENTS.md` and the linked final program report. A retry needs a new falsifiable reason the prior failure mechanism no longer applies; changing a seed or lightly retuning the same idea is not enough.
