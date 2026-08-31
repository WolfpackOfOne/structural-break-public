# STATE OF RESEARCH — V4

**2026 ADIA Lab / CrunchDAO Structural Break Challenge (Real-Time Edition)**
Written 2026-08-21. Branch `research/wave3-integration`.

`VALIDATION_V2.md` remains binding. Where this file and the repository disagree,
**the repository wins**.

---

## 0. THE HEADLINE

The champion survived its own control — and the control changed what the
champion means.

The seven-stream deployable ensemble beats a seven-way seed-clone ensemble by
**+0.00417** on 5/5 folds, and the delta stays positive on all four fold
partitions. It is real. But **57% of its advantage over a single model is
ordinary bagging**, reproducible by training one model seven times with
different seeds, and no wave-2 document said so. The defensible claim is
"+0.0033 for specialisation on top of +0.0050 for bagging", not "+0.0106 for a
seven-stream architecture".

The artifact is built, trained on all 10,000 labelled series, and has passed the
official `crunch test` with the determinism check. Its provenance chain — broken
in the inherited artifact — is repaired and verified from a clean clone.

---

## 1. INTEGRATION STATUS

| | |
|---|---|
| research parent | `bfcb232` (`research/wave2-2026`) |
| engineering parent | `24675a6` (`codex/wave3-engineering`) |
| common base | `8e76ad2` — the two are **siblings**, neither contains the other |
| integration commit | **`c4fb01e`** |
| conflicts | 1 file, `research/scripts/wave2_lib.py`, 2 hunks |

Resolved best-of-each-side, not ours/theirs: `ROOT` from the engineering branch
(repo-relative fallback; the research side still fell back to a dead container
path), `cols_of()` from the research branch (honours `SBR_FEATURES`, which is
what lets a worktree share the 10 GB feature cache). `src/sbr/pipeline.py` made
the identical change on both branches and merged silently.

**No research semantics changed.** Every engineering hunk was a path fix. Full
audit in `research/reports/wave3_integration_audit.md`. Both source branches are
untouched on `origin`.

## 2. RESEARCH RECORD REPAIR

The deployable champion **had no ledger row at all**. It was called "RT-150"
informally — a string that in the same `RESULTS.csv` resolves to `RT-150_f1`,
the *rejected* DGP-gating run at 0.60619, identically on both branches.

* Canonical ID assigned: **`RT-250`**, verified unused on every branch and in
  every reachable commit before allocation.
* `research/EXPERIMENT_ID_MAP.md` maps every ambiguous name. **Nothing was
  renamed**; historical rows keep their IDs.
* The **`R`-suffix trap** is documented: `RT-120R`..`RT-125R` are *not*
  reproductions of `RT-120`..`RT-125`. Only `RT-100R` and `RT-123R` carry their
  wave-1 configuration.
* `m08_chan`, described in `HANDOFF_WAVE3.md` as a completed rejected
  experiment, **exists in no commit on any branch**. Unsourced.

## 3. CURRENT SINGLE-MODEL CONTROL

`RT-300`, macOS/arm64: **0.61605**, folds 0.62903 / 0.61061 / 0.62688 / 0.61223
/ 0.60152. Linux ledger `RT-100`: 0.615103. Four folds reproduce to the last
printed digit; fold 3 moves +0.00476. Deterministic on re-run (bitwise identical
predictions three times). **Never compare a macOS arm to a Linux number.**

## 4. SPECIALIST ENSEMBLE (`RT-420`)

Seven boosters, cross-fitted SCDF calibration, equal-weight mean:
**0.62581** — folds 0.63828 / 0.62040 / 0.63392 / 0.61750 / 0.61894.

Against the Linux `RT-250`'s 0.62589: **0.00008 apart**, after rebuilding every
stream on a different platform with the original OOF vectors permanently lost.
`RT-250` is independently reproduced.

## 5. SEED-CLONE ENSEMBLE (`RT-421`)

Seeds fixed before the first run: **0, 1, 7, 42, 2026, 31415, 271828**.

| seed | id | OOF | | blend | mean |
|---|---|---|---|---|---|
| 0 | `RT-300` | 0.61605 | | raw mean | 0.62180 |
| 1 | `RT-401` | 0.61661 | | logit mean | 0.62172 |
| 7 | `RT-402` | 0.61357 | | global CDF | 0.62166 |
| 42 | `RT-403` | 0.61687 | | **SCDF** | **0.62164** |
| 2026 | `RT-404` | 0.61485 | | oracle (illegal) | 0.62160 |
| 31415 | `RT-405` | 0.61498 | | | |
| 271828 | `RT-406` | 0.61517 | | | |

Members 0.61544 ± 0.00106. Within-t rank correlation **0.7996** [0.795, 0.803];
specialists **0.6362** [0.400, 0.782].

## 6. SPECIALIST VS BAGGING — THE CONCLUSION

```
single champion                  0.61605
  + ordinary bagging             +0.00559     57%
  + specialist diversity         +0.00417     43%
= seven-stream deployable        0.62581
```

Paired series bootstrap, 200 replicates, common random numbers:

| contrast | mean | 95% CI | positive |
|---|---|---|---|
| specialist − seed clone | +0.00409 | [+0.00199, +0.00614] | 200/200 |
| specialist − single | +0.00957 | [+0.00662, +0.01227] | 200/200 |
| seed clone − single | +0.00548 | [+0.00285, +0.00810] | 200/200 |

Three findings nobody asked for:

1. **SCDF is a scale fix, not a blending improvement.** Calibration family moves
   the specialist arm +0.00267 and the seed arm 0.00016.
2. **The legal blend matches the illegal oracle** (0.62581 vs 0.62580; on the
   seed arm it is strictly better). "Percent of oracle gain recovered" is retired.
3. **The specialists are individually worse** (0.61227 vs 0.61544) and win the
   blend. Member-level TS-AUC is the wrong thing to optimise.

## 7. PARTITION STABILITY

| partition | single | seed 7 | spec 7 | bagging Δ | specialisation Δ | total Δ |
|---|---|---|---|---|---|---|
| canonical | 0.61605 | 0.62164 | 0.62581 | +0.00559 | +0.00417 | +0.00975 |
| alt1 | 0.60985 | 0.61526 | 0.61780 | +0.00541 | +0.00254 | +0.00795 |
| alt2 | 0.61904 | 0.62364 | 0.62756 | +0.00460 | +0.00392 | +0.00852 |
| alt3 | 0.61314 | 0.61752 | 0.62020 | +0.00438 | +0.00269 | +0.00707 |
| **mean ± SD** | | | | **+0.00499 ± 0.00059** | **+0.00333 ± 0.00083** | **+0.00832 ± 0.00113** |

All twelve deltas positive. **The single model's LEVEL moves 0.00918 across
partitions while the total DELTA moves 0.00268** — levels are dominated by which
series landed in which fold; deltas are a property of the method. Wave 2's
`RT-221/222/223` measured the first and was read as evidence about the second.

**Against our own headline:** canonical is the *most favourable* of the four,
and on alt1 the specialisation delta (+0.00254) would **not** have cleared
W4-E1's pre-registered +0.0030 bar. The effect is real *and* smaller than one
draw made it look. **Bagging is both the larger and the more stable half.**

## 8. EXACT RT-131 AUDIT

All seven wave-1 streams reproduced; **none needed an "UNREPRODUCIBLE" label**.
`RT-121` reproduces to six decimals across a platform change. `RT-125`/GOSS —
the flagged one — is fine: the two wave-1 scripts disagree about disabling
bagging alongside GOSS, but under lightgbm 4.7.0 both forms train to identical
predictions because GOSS ignores `bagging_fraction`.

| stream set | oracle (= `RT-131`) | legal SCDF | recovered |
|---|---|---|---|
| wave-1 originals (apples-to-apples) | 0.62600 | **0.62602** | **100.1%** |
| wave-2 `R` streams | 0.62580 | 0.62581 | 100.1% |

The 99.7% claim was measured on the wrong set but was, if anything, marginally
conservative. The two sets are interchangeable (0.62602 vs 0.62581, inside
noise), so the `R` streams' hyperparameter drift cost nothing.

## 9. CALIBRATION AUDIT

| coordinate | specialist | seed clone |
|---|---|---|
| `log(max(t,1))` — incumbent | 0.625814 | 0.621640 |
| `log(t+1)` — corrected | 0.625815 | 0.621640 |

Neutral to six decimals; the pre-registered tie-break adopts the corrected one.
The incumbent maps t=0 and t=1 to the same anchor **and** scores t=0 rows
against a grid built from a window that excluded them. **Canonical is now
`time_coord="log_n_seen"`.** Payloads record their coordinate; a payload without
the field is treated as legacy, so the Crunch-tested wave-2 artifact still loads
correctly — verified: rebuilding it against the new source gives diagnostic
`0.970255244897131`, bit-identical to the original report.

## 10. FOLD-MODEL DEPLOYMENT

| boosters | ms/point | p95 | private projection |
|---|---|---|---|
| 1 | 1.148 | 0.883 | — |
| 7 | 1.734 | 1.490 | 4.9 h |
| 14 | 2.285 | 2.067 | 6.4 h |
| 35 | 3.927 | 3.799 | **11.0 h** |

Shared engine **1.069 ms/pt**; marginal booster **0.078 ms/pt**. 35 vs 7 is
**2.26×, not the asserted 5×** — the shared feature engine dominates exactly as
the architecture predicts. **Rejected anyway**: 11.0 h leaves ~26% margin, and
there is no evidence 35 fold-models score better. Now a measurement, not an
assertion.

## 11. PROVENANCE AUDIT

| | |
|---|---|
| old, broken | `b5ea9d1d9cd87f06f574c5a78e4c850f41ef852f` — `fatal: bad object`, on no remote |
| model trained at | `41ab069` — reachable, clean tree |
| artifact built at | `73b5662` — reachable, embedded source clean |
| `src/sbr` between them | **identical** (`git diff` empty) |

**Clean-clone reproduction: PASSED.** A fresh clone of the branch re-packs
`src/sbr` to the shipped `source_zip_sha256` exactly, the model zip extracted
from the committed notebook matches `model_zip_sha256`, and `41ab069` is
reachable in the clone.

That check initially **failed**, and the failure was worth more than the pass:
`pack_dir` used `z.write()`, which stores file mtimes, so byte-identical source
hashed differently depending on when it was checked out. `source_zip_sha256`
could never have verified a rebuild. Zips are now content-addressed (1980 epoch,
fixed permissions, sorted order). The notebook and `.py` remain *not*
bit-reproducible — the boot cell embeds a build timestamp — so the zip hashes
are the identities to verify.

## 12. NEW ALPHA RESEARCH

| family | status | verdict |
|---|---|---|
| union of specialists + seed clones (W4-E6) | **RUN** | **REJECTED**, −0.00095, 1/5 folds |
| `m09_back` backward CUSUM (wave 3) | RUN | REJECTED — failed its seed-clone control |
| transient vs permanent | **NOT RUN** | — |
| hard-negative sampling | **NOT RUN** | — |
| dependence likelihood v2 | **NOT RUN** | — |
| robust distribution distances | **NOT RUN** | — |
| residual CUSUMSQ / AR regression break | **NOT RUN** | — |
| robust BOCPD | **NOT RUN** | — |
| AR(p)-FOCuS | **NOT RUN** | — |
| TabPFN / deep / teacher | **NOT RUN** | no GPU |

**W4-E6 is the one substantive alpha result and it is negative.** The union of
both arms (13 boosters) scores **0.62486** against the specialists' 0.62581.
Mechanism: under an equal-weight mean the union is *"specialists with the
champion configuration up-weighted sevenfold"* — 7 of 13 members are `RT-100R`
at different seeds, taking 54% of the blend weight instead of 14%. Ensembles
work by averaging models wrong in *different* directions.

The eight untested families are marked NOT RUN because the wave-4 compute went
to auditing the champion instead, and because W4-E1 raised the bar: any new
stream must now beat a seed-clone control, which `m09_back` and the union both
failed. **This is the largest remaining opportunity and the honest gap in this
work.**

## 13. NEGATIVE RESULTS

`m09_back` (+0.00156, but a seed clone gave +0.00496) · W4-E6 union (−0.00095) ·
35 fold-models (feasible, no evidence of gain, 26% margin) · `INFER_PARALLELISM=4`
(segfaulted three ways) · raw/logit/global-CDF blends (all below SCDF) ·
switching to wave-1 original streams (+0.00021, inside noise).

## 14. BIAS AUDIT

* **Selection bias.** W4-E1's bar (+0.0030) and seed list were fixed before the
  first run. W4-E6's hypothesis was chosen *after* seeing W4-E1 — stated in the
  ledger, and given an extra bootstrap condition for it.
* **Fold contamination.** Every calibration cross-fitted; fold *k*'s map never
  sees fold *k*.
* **Multiple testing.** Wave 4 spent 63 training runs, 4 compositions, 5
  calibration families (all reported, none selected from), 0 hyperparameter
  searches, 0 calibration tuning. Two of four compositions rejected.
* **Partition variance.** Measured, and it costs the headline: canonical is the
  most favourable draw.
* **Seed variance.** SD 0.00106 across seven seeds.
* **Platform variance.** Four of five folds bitwise; one moves +0.00476.
* **Leaderboard bias.** None — no leaderboard submission was made or consulted.
* **Lockbox.** Not scored in wave 4. Used only as FINAL-FIT training data after
  the freeze was committed.
* **`X_test.reduced`.** Never inspected. Only the official runner touched it.

## 15. FINAL ARCHITECTURE

Seven LightGBM boosters over one shared streaming engine, 500 causal features in
7 modules, equal-weight mean of `SmoothTimeCDFCal(time_coord="log_n_seen")`
outputs. Seeds 0, 0, 1, 7, 0, 3, 11. `num_threads=2`. `INFER_PARALLELISM=1`.
Full specification and rejected alternatives:
`research/FINAL_ARCHITECTURE_FREEZE.md`, committed **before** the 10k fit.

## 16. FINAL 10K FIT

All **10,000** labelled series (the spent lockbox folded back in as training
data), partition `folds_final10k`, row budget **1.25× dev** per the rule
pre-declared before the freeze: 1,250,000 / 1,125,000 ×3 / 875,000 ×3.
Calibration from cross-fitted OOF `RT-500`..`RT-506` over the same partition.

Residual mismatch, stated not hidden: grids come from 8,000-series models while
the boosters see 10,000. Smaller than the status quo (6,400 vs 8,000). Not zero.

## 17. FINAL SUBMISSION

| | |
|---|---|
| `crunch test` | **PASSED**, determinism passed at 1e-08 |
| duration / memory | 00:01:43 / 1.81 GB |
| local harness | PASS, 1.909 ms/point |
| projection | 2.7 h public, 5.4 h private, **8.0 h combined** vs 15 h |
| source zip | `199db8c9f5db7e1429f6ae09fa018d43cc58c5eb47b5c623a017799537a413a0` |
| model zip | `6c8960ddc7331afecfc6980974f2879401a1eb583f15f5cad34c2ea03299ea8c` |
| notebook | `8332b698b6dbe91376c79d2051c5d294f515f77dea835a278b858b45bda73cc5` |
| Python entrypoint | `660c88c104c990626a5758f8d36af49c87273a587e399d86bbc353cc9bba2647` |
| feature manifest | `1646c3b9e09d8a7fb3c564483a1d1d999680caeefe848b092f907a6cac80cced` |

**A `crunch test` run can silently target the wrong competition.** The CLI walks
*up* for `.crunchdao/project.json`; with none in the worktree it found a May-2025
config in `$HOME` and validated the **2025** data release. The tell is the file
list: a correct real-time run validates **six** files including
`y_test_index.reduced` and `y_train_index`. Any log missing those two was run
against the wrong competition. Full detail in
`research/reports/final_crunch_test.md`.

## 18. EXPECTED EXTERNAL PERFORMANCE

**Do not quote 0.62581 as an expected leaderboard score.**

| scenario | estimate | reasoning |
|---|---|---|
| optimistic | ~0.625 | private set resembles training; +25% data offsets the favourable draw |
| **base** | **~0.615** | the single-model level moves 0.0092 across *internal* partitions alone; an external draw should move at least as much, with +0.0083 riding on a lower base |
| conservative | ~0.605 | composition shift, heavier tails, or an online-length mix concentrating weight where TS-AUC is 0.513 |

**The delta is the durable asset: +0.0083 ± 0.0011 across four partitions. The
level is not.**

## 19. REMAINING RISKS

Private-set composition, tail mix and online-length distribution — untestable by
construction. Whether specialisation holds on an external draw (four internal
partitions agree; one sits below W4-E1's bar). The eight untested alpha
families. The residual calibration training-regime mismatch. Platform: all
wave-4 numbers are macOS/arm64; the runner is Linux.

## 20. MERGE PLAN

| branch | SHA | action |
|---|---|---|
| `research/wave3-integration` | HEAD | **the trunk.** Merge to `main` when ready |
| `research/wave2-2026` | `bfcb232` | keep; fully contained in the integration |
| `codex/wave3-engineering` | `24675a6` | keep; fully contained in the integration |
| `research/multi-agent-2026` | `23cb4da` | **not** merged; still holds `RT-160`/`RT-190` and the platform-constraints research |

Recommended: PR `research/wave3-integration` → `main`. Do not delete the source
branches — they are the provenance of the integration commit.

## 21. IF THE COMPETITION ENDED TODAY

Submit **`submissions/C_ensemble_deployable.py`** (with `.ipynb` as the
equivalent notebook form), model directory `models/final10k_ensemble`.

Trained on all 10,000 labelled series, causal, deterministic to 1e-08,
`crunch test` passed, 8.0 h of a 15 h budget, provenance verified from a clean
clone.
