# Pre-consolidation model snapshot — 2026-08-30

This file records the model state established before consolidation changes. Historical experiment rows remain authoritative evidence; this file is a current-state classification, not a rewrite of the ledger.

| Model | Family / role | Pre-consolidation classification | Evidence summary |
| --- | --- | --- | --- |
| RT-1257 | Hybrid 5x LightGBM + CAT-300 + CAT-413 | EXTERNAL_BEST / PROMOTION_PENDING_HYGIENE | Official submission #16 = 0.6290 vs RT-600 0.6268; +0.0022 external. Formal production anchor not yet moved because deployment hygiene remained open. |
| RT-600 | 7-slot LightGBM specialist ensemble | REFERENCE / FORMAL_PRODUCTION_ANCHOR | Official external 0.6268; frozen production lineage. |
| RT-1254 / CAT-413 | CatBoost replacement of RT-413 | ACTIVE_COMPONENT | Component of RT-1257; positive fixed-slot evidence. |
| RT-1255 / CAT-300 | CatBoost replacement of RT-300 | ACTIVE_COMPONENT | Component of RT-1257; positive fixed-slot evidence. |
| RT-1261 / CAT-412 | CatBoost replacement of RT-412 | RESEARCH_ALIVE | Strongest remaining additional single-slot candidate; corrected k=3 curve is only +0.000338234 over RT-1257, below 0.0011 noise floor. |
| RT-1263 / CAT-415 | CatBoost replacement of RT-415 | RESEARCH_ALIVE | Positive individual gate, but no corrected multi-slot evidence that it improves RT-1257. |
| RT-1262 / CAT-414 | CatBoost replacement of RT-414 | RESEARCH_ALIVE | Positive individual gate, but no corrected multi-slot evidence that it improves RT-1257. |
| RT-1260 / CAT-411 | CatBoost replacement of RT-411 | RESEARCH_ALIVE | Positive clone-relative individual gate; corrected E2-E0 is small and mature-vs-never is weak. |
| RT-995 / T2 | Teacher-distilled causal LightGBM | PARKED | Strong standalone teacher signal; final RT-600 integration showed only small marginal ensemble alpha. |
| RT-1264 | Original CSA-04 five-slot CatBoost hybrid | SUPERSEDED | Historical CSA-04 result used a control-collapse-sensitive E2-E1 selection endpoint. CSA-04R / RT-1265 corrects the selection rule; RT-1264 is not a live upside candidate. |
| RT-1265 | CSA-04R corrected reanalysis identity | NOT_DISTINGUISHABLE / SUPERSEDING_ANALYSIS | Corrected fixed E2-E0 + parsimony selection chooses k*=2 = CAT-413 + CAT-300, exactly RT-1257; no new OOF vector. |
| RT-1256 / CAT-410 | CatBoost replacement of RT-410 | KILL | Marginal-vs-clone below gate. |
| RT-1250 | TabM local/full-scale feasibility record | INFEASIBLE_HISTORICAL | Original learner-diversity arm did not complete under frozen full-scale local/default setup. Not an independent scientific KILL from RT-1258. |
| RT-1252 | RealMLP local/full-scale feasibility record | INFEASIBLE_HISTORICAL | Original learner-diversity arm did not complete under frozen full-scale local/default setup. Not an independent scientific KILL from RT-1259. |
| RT-1258 | TabM GPU/cloud technical recovery of original arm | KILL | Completed five-fold OOF recovery; marginal_vs_clone +0.000040947 and E2-E0 negative. Binding scored verdict for the TabM arm. |
| RT-1259 | RealMLP GPU/cloud technical recovery of original arm | KILL | Completed five-fold OOF recovery; marginal_vs_clone -0.003223744. Binding scored verdict for the RealMLP arm. |

## ID relationship that must be preserved

RT-1250 (TabM) and RT-1252 (RealMLP) are the original learner-diversity full-scale feasibility records. They did not complete under the frozen local/default setup and were recorded INFEASIBLE. RT-1258 and RT-1259 are later GPU/cloud technical recovery/completion records for those same preregistered learner arms, not independent hypotheses or a second scientific test. The binding scored verdicts are RT-1258/RT-1259 (both KILL); RT-1250/RT-1252 remain historical feasibility records and must not be presented as a second pair of independent kills.

## RT-1264 correction

The original RT-1264 row remains immutable historical evidence. Current-state registries must classify it SUPERSEDED by the corrected CSA-04R analysis, RT-1265. The descriptive 63-subset appendix is not permitted to select a champion. CAT-412 / RT-1261, not RT-1264, is the most interesting remaining single-slot research candidate.
