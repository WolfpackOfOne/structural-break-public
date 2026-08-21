# W4-E5 — THE APPLES-TO-APPLES RT-131 AUDIT

**Question.** `sbr/production/calibration.py` claimed the deployable transform
"recovers 99.7% of the oracle rank-average's gain over the best single model".
That number was measured over the wave-2 `R` streams. `RT-131` — the oracle it
claims to recover — was built from the **wave-1** streams, five of which have
different hyperparameters (`research/EXPERIMENT_ID_MAP.md` §4). So the headline
described a stream set the oracle was never computed on.

**Method.** Rebuild the exact wave-1 streams on this machine from the `agent0`
scripts that produced their `RESULTS.csv` rows, then compute — for *both* stream
sets, on one platform, with identical cross-fitted calibration — the single
champion, the illegal within-timestep rank oracle, and every legal blend.

---

## 1. WERE THE ORIGINAL STREAMS REPRODUCIBLE? YES — ALL SEVEN

| wave-1 | wave-4 rebuild | wave-1 score (Linux) | rebuild (macOS) | delta |
|---|---|---|---|---|
| `RT-100` | `RT-300` | 0.615103 | 0.616054 | +0.00095 |
| `RT-120` | `RT-430` | 0.608839 | 0.609148 | +0.00031 |
| `RT-121` | `RT-431` | 0.604881 | 0.604881 | **0.000000** |
| `RT-122` | `RT-432` | 0.613155 | 0.613398 | +0.00024 |
| `RT-123` | `RT-413` | 0.614499 | 0.614810 | +0.00031 |
| `RT-124` | `RT-433` | 0.610868 | 0.610526 | −0.00034 |
| `RT-125` | `RT-434` | 0.613558 | 0.615795 | +0.00224 |

**No stream needed an "UNREPRODUCIBLE UNDER CURRENT ENVIRONMENT" label.**
`RT-121` reproduces to six decimals across a platform change.

`RT-125` / GOSS was the one flagged as version-sensitive, and the flag was
justified but the outcome benign. Two wave-1 scripts disagree about whether
bagging is disabled alongside GOSS — `agent0_streams2.py` leaves
`bagging_fraction` unset, `agent0_stream_f.py` sets `bagging_fraction=1.0,
bagging_freq=0`. Under lightgbm 4.7.0 **both forms train to identical
predictions**, because GOSS ignores `bagging_fraction` entirely. The ambiguity
is immaterial. Its +0.00224 is the largest platform gap in the table and is
consistent with GOSS's exact-gradient row selection being the most sensitive of
the seven configurations to floating-point tie-breaking, not with a
mis-specification.

## 2. THE ANSWER

### Wave-1 original streams — the apples-to-apples set

| | mean OOF | gain over single | % of oracle gain |
|---|---|---|---|
| single champion `RT-300` | 0.61605 | — | — |
| ORACLE within-t rank avg (**this is `RT-131`**) | 0.62600 | +0.00995 | 100% by definition |
| legal SCDF (corrected `t+1`) | **0.62602** | +0.00996 | **100.1%** |
| legal SCDF (legacy coordinate) | 0.62602 | +0.00996 | 100.1% |
| legal global CDF | 0.62573 | +0.00968 | 97.3% |
| legal logit mean | 0.62501 | +0.00896 | 90.0% |
| legal raw mean | 0.62345 | +0.00740 | 74.3% |

Corroboration: the rebuilt oracle scores 0.62600 against the wave-1 ledger's
`RT-131` = 0.625411, a +0.0006 platform gap in line with the individual streams.

### Wave-2 `R` streams — the set the 99.7% was measured on

| | mean OOF | gain over single | % of oracle gain |
|---|---|---|---|
| ORACLE within-t rank avg | 0.62580 | +0.00975 | — |
| legal SCDF | 0.62581 | +0.00976 | **100.1%** |
| legal global CDF | 0.62550 | +0.00945 | 96.9% |
| legal logit mean | 0.62479 | +0.00874 | 89.6% |
| legal raw mean | 0.62314 | +0.00709 | 72.7% |

## 3. VERDICT

**The concern was legitimate and the answer survives it.** Both stream sets give
**100.1%**. The 99.7% figure was, if anything, marginally conservative, and the
fact that it was measured on the wrong set turns out not to matter — the two
sets are equivalent for this purpose.

**But the framing should be retired.** "Recovers 100.1% of the oracle gain" is a
confusing way to say what is actually happening: **the legal transform matches
the illegal one.** A frozen per-series `F_m(s | t)` fitted on training folds is
not an approximation to within-timestep cross-sectional ranking — on this
problem it is an equal substitute, and on the seed-clone arm of W4-E1 it was
strictly better (0.62164 vs 0.62160). The oracle was never a ceiling. Wording in
`calibration.py` has been changed accordingly.

**The two stream sets are interchangeable.** Wave-1 originals blend to 0.62602,
wave-2 `R` streams to 0.62581 — a gap of 0.00021, an order of magnitude below
the fold-to-fold SD of 0.0085 and below the seed SD of 0.0011. The `R` streams'
hyperparameter drift from the wave-1 configurations cost nothing. There is
**no case for switching the deployed ensemble to the wave-1 configurations**,
and no case for claiming the drift was harmful.

**Where the calibration actually earns its keep.** Raw mean recovers only 74%.
The +0.0026 between raw and SCDF is the whole reason the deployable system
approaches the oracle at all — and W4-E1 showed that margin collapses to 0.0002
when the members are seed clones on a common scale. The transform is a **scale
reconciliation for heterogeneous members**, not a general blending improvement.
