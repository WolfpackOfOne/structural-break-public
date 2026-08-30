# STATUS

Concise pointer to current state. [`STATE_OF_RESEARCH.md`](STATE_OF_RESEARCH.md)
is an archival Wave 1-6 summary whose opening "current champion" section is
stale; use this file for live production-anchor, external-score, and active
research conclusions.

- **Competition:** ADIA Lab / CrunchDAO Structural Break Challenge —
  **Real-Time Edition** (causal/streaming inference; break location unknown,
  scored one prediction per online time step with Time-Stratified AUC).
- **Current production anchor:** `RT-600`, frozen on the `production/rt600`
  branch (tag `rt600-production-0.6268`). Seven-stream deployable ensemble,
  all 10,000 labelled series. Environment build, LB-002/003/004 verification
  runs, and submission #5 record are on that branch.
- **Current external score:** **0.6290** on the official Crunch leaderboard —
  `RT-1257`, submission #16, 2026-08-29. Beats the prior external best of
  **0.6268** (`RT-600`, LB-001, confirmed in
  [`EXPERIMENT_ID_MAP.md`](EXPERIMENT_ID_MAP.md)) by **+0.0022**, the first
  external movement in this program since 0.6268 was set. The realised delta
  sits between the two internal estimates that licensed the promotion
  (`marginal_vs_clone` +0.002407, E2−E0 +0.002026). Full record:
  [`engineering/reports/rt1257_deployment/SUBMISSION_16.md`](../engineering/reports/rt1257_deployment/SUBMISSION_16.md).
  **This is a calibration point, not an oracle** — no parameter may be selected
  on it, and the anchor has not moved (see above): promotion of `RT-1257` to
  production is an owner decision, and two deployment-hygiene items are open
  against the artifact first (§4 of that record).
- **Production anchor:** still **`RT-600`** (see above) — `RT-1257` has not been
  promoted, so the anchor has not moved.
- **Best-scoring internal result not yet promoted:** `RT-1257` two-slot hybrid,
  E2 mean OOF TS-AUC **0.627837664**, E2-E0 **+0.002026322** over the fixed
  `RT-600` counterfactual. Deployable in the sense that it qualified as an
  artifact and reached scoring; *not* the current anchor. For `RT-600` itself,
  the canonical E0 scores are mean OOF **0.625811342** and pooled dev
  **0.625626926**; **0.638276** is fold 0 only, not pooled OOF.
- **Current research conclusion:** **CATBOOST SPECIALIST ACTIVATION 2026** is
  complete on `research/catboost-specialist-2026`. CAT-413 (`RT-1254`) and
  CAT-300 (`RT-1255`) are INTERESTING, CAT-410 (`RT-1256`) is KILL, and the
  two-slot hybrid (`RT-1257`, replacing RT-300 and RT-413 only) is
  **PROMOTION_WORTHY** but not SERIOUS: `marginal_vs_clone = +0.002407205`,
  E2-E0 `+0.002026322`, `5/5` positive folds, dominant-cell net `+158`,
  mature-vs-never net `+73`. Full report:
  `reports/catboost_specialist_2026/FINAL.md`. Production `RT-600` remains
  unchanged. **Externally confirmed 2026-08-29:** submission #16 scored 0.6290,
  +0.0022 over the RT-600 anchor — the internal battery did not mislead on this
  artifact. One observation, no error bar; not a transfer law.
- **Prior research conclusion:** T2/RT-995 promotion battery —
  **MOSTLY REDUNDANT**. T2 clears 3/4 promotion legs and has real
  single-model alpha, but E2 (RT600 + T2 ensemble, ≈0.63882) beats E1
  (RT600 + seed clone, ≈0.63859) by only +0.00024 — essentially no marginal
  ensemble alpha. The first New Avenues sweep is now
  **FIRST_SWEEP_EXHAUSTED**. Second Sweep `SS-01` repair-damage arbitration,
  `SS-02` dominant-cell residual ranking, `SS-03` negative-side null-state
  calibration, and `SS-04` specialist-disagreement micro-routing are all
  **KILL**. SS-01 selected no first-sweep sensor actions; SS-02 produced
  negative pair flow (`dominant net=-108`) and negative marginal_vs_clone
  (`-0.000290`); SS-03 produced negative mature-vs-never net pair flow (`-49`),
  negative marginal_vs_clone (`-0.000299`), and failed the prebreak damage-rate
  cap (`0.0199 > 0.0150`); SS-04 produced no dominant or targeted
  specialist-disagreement repairs on the canonical sample and negative
  marginal_vs_clone (`-0.000312`). The preregistered Second Sweep is
  **EXHAUSTED** with no confirmation candidate. Full detail:
  `reports/wave7/` and tag
  `wave7-t2-promotion-mostly-redundant`, plus
  `reports/new_avenues_2026/FIRST_SWEEP_SYNTHESIS.md`,
  `reports/new_avenues_2026/SECOND_SWEEP_PREREG.md`,
  `reports/new_avenues_2026/second_sweep/ss01_repair_damage_arbiter.md`, and
  `reports/new_avenues_2026/second_sweep/ss02_residual_ranker.md`, and
  `reports/new_avenues_2026/second_sweep/ss03_null_calibrator.md`, and
  `reports/new_avenues_2026/second_sweep/ss04_specialist_router.md`.
- **Latest major negative result:** Wave 8 future-aware transfer family —
  five pilots (ORR, TGMC, SST, PCFB, CFEP), all KILL on the full population.
  Lives on the sibling branch `research/wave8-future-aware-distillation`
  (tag `wave8-future-aware-final`); not merged into this lineage.
- **Standing major positive result:** W7-D3R same-prefix-vs-full-sequence
  diagnostic (tag `wave7-d3r-information-frontier`) established the
  future-information limit (CASE 2) that later Wave 7/8 work is measured
  against. Still the reference point, but **CASE 2 is now only partially
  right**: residualizing Arm C against endpoint/horizon variables shows a real
  orthogonal residual on the never-break cut, so the limit was not purely
  future information. See the Grok follow-up below.
- **Latest Grok-response score follow-up:** Arm-C horizon residualization and
  nested causal student are complete on this branch. First, `RT-991.npy` was
  found in the Wave 8 OOF artifacts and the cross-fitted projection of
  `logit(RT-991)` on `{n_online, n_online-t, t/n_online}` showed a real
  residual: **0.685794** on the W7-D0 dominant cell, **+0.038303** over Arm B
  (`RT-990`), with within-t rank rho **0.457016** vs Arm B. That lift is
  **never-break-cut only** — on the pre-break cut the residual is
  **−0.006822** against Arm B, so Arm C's raw pre-break lift is endpoint
  information, not transferable structure. Second, a nested fold-pure
  500-causal-feature student of that residual gives
  `RT600 + residual_student` pooled whole-dev gain **+0.001973**, marginal vs
  seed-clone blend **+0.001951** (**5/5 positive folds**, clearing the
  **+0.0015** bar), never-break net rate **+0.003409**, and pre-break damage
  rate **0.013672 < 0.015000** on the dominant pre-break cut. A light
  combination with the current CatBoost hybrid is also positive:
  `RT1257 + residual_student` is **+0.001396** mean whole-dev TS-AUC over
  `RT-1257`, **+0.003422** over `RT-600`. Caveats that must travel with these
  numbers: the gain is concentrated in folds 1 and 3 (t = 2.56 on five folds);
  the standalone student-vs-Arm-B contrast is **2/5 positive** and not stable;
  and the whole-dev damage rate is **0.015175**, above the gate value, which is
  scoped to the pre-break cut only. This is **PROMOTE for confirmation only**,
  not a production change or new RT ID. Reports:
  [`reports/armc_residualization.md`](reports/armc_residualization.md),
  [`reports/armc_residual_student/armc_residual_student.md`](reports/armc_residual_student/armc_residual_student.md),
  and the roll-up
  [`reports/grok_response_followup.md`](reports/grok_response_followup.md).
- **Active research branch:** this worktree is
  `grok-response-issues-20260830`, a follow-up branch for addressing
  `agent_05_grok46.md`. It was branched from
  `research/multi-agent-frontier-20260829`, whose checked-in research state was
  based on the CatBoost-specialist lineage. The canonical chain remains
  `research/current` (wave2 → wave3-integration → wave5-alpha → wave6-alpha →
  wave7-teacher-distillation → wave7-t2-promotion). Standalone sibling
  branches still worth checking: `research/wave8-future-aware-distillation`
  (killed transfer family), `research/multi-agent-2026` and
  `codex/wave3-engineering` (an earlier, parallel RT-000..RT-213-numbered
  track predating the Wave 5 restart), `codex/oracle-information-frontier-2026`
  and `codex/reproduce-2025-public-solution` (standalone studies that
  motivated the Wave 5 restart), `claude/project-setup-github-20g6pt` (an
  abandoned early Wave 7 opening attempt).
- **Experiment ledger:** [`RESULTS.csv`](RESULTS.csv), IDs explained in
  [`EXPERIMENT_ID_MAP.md`](EXPERIMENT_ID_MAP.md), degrees-of-freedom
  accounting in [`RDOF_LEDGER.md`](RDOF_LEDGER.md).
- **Failed experiments:** [`FAILED_EXPERIMENTS.md`](FAILED_EXPERIMENTS.md) —
  read before proposing anything that resembles a killed idea.
- **Next-wave research direction:**
  [`NEW_AVENUES_2026.md`](NEW_AVENUES_2026.md) — broad-exploration survey
  written 2026-08-24 after the Wave-7/8 dead end: nine functional classes
  absent from the 500-column bank, 79 source-backed mechanisms in 15
  families ([`new_avenues_2026.csv`](new_avenues_2026.csv)), six descriptive
  diagnostics `D1`–`D6`
  ([`reports/new_avenues_2026_diagnostics.json`](reports/new_avenues_2026_diagnostics.json)),
  and an 11-experiment queue. Reusable plugin harness in
  [`scripts/novel_streams/`](scripts/novel_streams/) — imports with no
  environment set up; the shared ensemble/pair-flow helpers it reuses are
  already on this branch at `scripts/wave8_common.py`, so no sibling branch
  or worktree is needed. Regression-tested by
  `tests/test_novel_streams_harness.py`. The cleanup completed on
  `research/current` at `aca2c4f` and is infrastructure-only: no pilot run,
  no new score, no promotion, and RT-600 still the production anchor.
- **First New Avenues execution branch:**
  `research/new-avenues-pilots-2026` has completed the planned first sweep:
  Pilots 1-8, 10, 9(i), and the Pilot 3 / IM3 observer arms. Pilot 1
  specialist/failure-manifold diagnostics are **WEAK**. Every scored
  first-sweep candidate is **KILL** or a negative/control result: `RT-1200`
  relay score-state, `RT-1201` IM2+dwell scalar, `RT-1202` trajectory
  geometry, `RT-1204` scale survival, `RT-1206` spectral impulsiveness,
  `RT-1208` ordinal transition/irreversibility, `RT-1210` joint
  size-duration rarity, `RT-1212` scalar historical-difficulty gate,
  `RT-1214` Kalman/NIS observer, `RT-1215` Hankel-DMD observer, `RT-1216`
  weighted conformal test martingale, and `RT-1218` parameter-free e-value
  aggregation all failed the primary continuation gate. `RT-1216` was the
  closest miss at marginal_vs_clone `+0.000937` after beating its unweighted
  CTM control on mature-vs-never AUC by `+0.002135`; `RT-1218` was negative
  at `-0.002214`. `RT-1215` also failed the preregistered Hankel redundancy
  guard (`rho=+0.8859 > 0.85`). No 5-fold confirmation is warranted from the
  first sweep; the remaining planned first-sweep mechanisms are exhausted.
  Post-hoc descriptive synthesis is complete in
  `reports/new_avenues_2026/FIRST_SWEEP_SYNTHESIS.md`, with normalized matrix
  `reports/new_avenues_2026/first_sweep_matrix.csv`, repair/overlap
  diagnostics, and a source manifest. The second sweep is preregistered in
  `reports/new_avenues_2026/SECOND_SWEEP_PREREG.md`. `SS-01`
  Repair-Damage Arbiter (`RT-1219` plus controls `RT-1220`/`RT-1221`/`RT-1222`)
  is **KILL**: marginal_vs_clone `-0.000310`, dominant repairs/damage/net
  `0/0/0`, repair-reservoir retention `0.0000`, and `0` contributing sensor
  families. `SS-02` Dominant-Cell Residual Ranker (`RT-1223` plus shuffled
  control `RT-1224`) is also **KILL**: marginal_vs_clone `-0.000290`,
  dominant repairs/damage/net `330/438/-108`, mature-vs-never net `-77`, and
  shuffled-control gap `-0.000087`. `SS-03` Negative-Side Null Calibrator
  (`RT-1225` plus controls `RT-1226`/`RT-1227`/`RT-1228`/`RT-1229`) is
  **KILL**: marginal_vs_clone `-0.000299`, dominant repairs/damage/net
  `659/773/-114`, mature-vs-never net `-49`, mature-vs-prebreak net `-40`,
  prebreak damage rate `0.0199`, deranged-control marginal gap `+0.000049`,
  and unweighted-CTM control marginally exceeded the candidate. First-sweep
  killed-arm arbitration, incumbent-bank residual pair ranking, and the fixed
  null-state calibration are closed under their frozen screens. `SS-04`
  Specialist Disagreement Micro-Router (`RT-1230` plus controls
  `RT-1231`/`RT-1232`/`RT-1233`) is **KILL**: marginal_vs_clone `-0.000312`,
  dominant repairs/damage/net `0/0/0`, majority-correct net `0`, near-split
  net `0`, and all three controls failed the `+0.0005` gap requirement. The
  preregistered Second Sweep is exhausted with no 5-fold confirmation
  candidate; production `RT-600` remains unchanged.
- **Sweep status:** `FIRST_SWEEP_EXHAUSTED` and `SECOND_SWEEP_EXHAUSTED`.
  Both preregistered New Avenues sweeps are closed with no confirmation
  candidate. Production `RT-600` (external **0.6268**) is unchanged.
- **Current research program:** **CATBOOST SPECIALIST ACTIVATION 2026** —
  **`CATBOOST_SPECIALIST_ACTIVATION_2026_COMPLETE`** on
  `research/catboost-specialist-2026`. `RT-1257` is the best measured arm:
  hybrid `marginal_vs_clone = +0.002407205`, E2-E0 `+0.002026322`, `5/5`
  positive folds, dominant-cell net `+158`, mature-vs-never net `+73`.
  This is PROMOTION_WORTHY by the frozen gate but below the SERIOUS threshold
  and requires separate deployment inference benchmarking. Final report:
  [`reports/catboost_specialist_2026/FINAL.md`](reports/catboost_specialist_2026/FINAL.md).
  Production `RT-600` remains unchanged.
- **Completed prior research program:** **LEADERBOARD ALPHA 2026** —
  **`LEADERBOARD_ALPHA_2026_EXHAUSTED`** on
  `research/leaderboard-alpha-2026`. All three preregistered experiments are
  **KILL** and no combination opens. LA-01 specialist replacement salvage
  (`RT-1243`/`RT-1244`) failed at `-0.000005408` marginal vs clone. LA-02
  counterfactual synthetic augmentation (`RT-1245`/`RT-1246`) had metric-class
  **MAJOR** signal vs its synthetic clone (`+0.005157709`) but failed pair-flow
  gates and remained below RT600. LA-03 per-series history adaptation
  (`RT-1247`/`RT-1248`/`RT-1249`) had WEAK integrated marginal vs the global
  clone (`+0.001676065`, `5/5` positive folds), but failed the mandatory
  fixed-null isolation gate (`RT-1249 - RT-1248 = -0.021495290` standalone).
  Final report:
  [`reports/leaderboard_alpha_2026/LEADERBOARD_ALPHA_2026_FINAL.md`](reports/leaderboard_alpha_2026/LEADERBOARD_ALPHA_2026_FINAL.md).
  Production `RT-600` remains unchanged.
- **Completed prior research program:** **CAUSAL REPRESENTATION FRONTIER (CRF)** —
  **`CRF_PROGRAM_EXHAUSTED`**. Both primaries are KILL, `CRF-03` did not open, and
  model search under this program is stopped. Full account in
  [`reports/causal_representation_frontier/CRF_FINAL.md`](reports/causal_representation_frontier/CRF_FINAL.md);
  design and evidence in
  [`reports/causal_representation_frontier/`](reports/causal_representation_frontier/).
  - **`CRF-01` NNCSR** (`RT-1234`, control `RT-1235`; `RT-1236` reserved and **not**
    consumed) — **KILL**, abandoned at the cheap abandon gate: fold-0 standalone
    whole-fold TS-AUC `0.592762` (below the `0.600` necessary condition) at
    within-`t` ρ `+0.4460`. Pair flow negative in every cell (dominant net
    `-2,798`, mature-vs-never `-2,962`); pre-break damage rate on RT600-correct
    pairs `0.2626` against a `0.0150` cap. Because `RT-970` is the same shell on
    the same folds, the ladder `RT-970` 0.52618 → `RT-1235` 0.57054 → `RT-1234`
    0.59276 isolates both factors for the first time: **representation effect
    `+0.0444`**, **objective effect `+0.0222`**. Wave 6 could not tell "family
    wrong" from "objective wrong"; both were partly wrong and their sum is still
    short. `RT-1234` set a **new best standalone-at-low-redundancy point**
    (0.59276 at ρ 0.446, prior record `RT-1201`'s 0.58358) — the frontier moved
    and the answer did not change.
  - **`CRF-02` ACGN** (`RT-1240`, controls `RT-1241`/`RT-1242`) — **KILL**. The
    abandon gate fired (`0.559140` at ρ `+0.2887`) and the **learned-null isolation
    gate failed at `-0.021446`**: the **fixed** per-series null (AR(5) + 256-knot
    history residual ECDF), through identical downstream statistics and an identical
    ranking head, reached `0.580586` and beat the learned null. The **derangement
    control PASSED** (`+0.003377` whole, `+0.008391` dominant), so the 8-dimensional
    history bottleneck does carry real series information and the model is not
    memorising a series identifier — which makes the negative sharp rather than
    ambiguous: **the learned null conditions correctly and loses anyway.** The
    failure is **amortization**. The fixed null holds five AR coefficients plus a
    256-knot empirical residual distribution *per series*, paid for by that series'
    own break-free history at zero generalisation cost; the learned null compresses
    all of it into 8 floats shared across a population whose heterogeneity is the
    reason per-series historical calibration is this project's foundation. The null
    the project already ships **is** the right null — now measured against a matched,
    correctly-conditioned learned alternative rather than assumed.
  - **One void run, recorded.** `RT-1237`/`RT-1238`/`RT-1239` are **VOID and
    retired**: that run silently loaded a 24-series, 1-epoch null written by a unit
    test instead of the preregistered one. Caught by compute accounting
    (`pretrain_runtime_s = 0.2` against a real 1,284.6 s). It mattered
    scientifically, not just procedurally — the toy null made the derangement
    control *tie*, which would have supported the wrong mechanism. Fixed with a
    checkpoint **provenance fingerprint** that stops the run on mismatch, and
    test-isolated caches. `CRF-01` was verified unaffected. Detail in
    `EXPERIMENT_ID_MAP.md` and `crf02_acgn.md` §7.
  - **Reading at the time.** The representation × objective factorial is complete and
    empty, and learned generative nulls are closed. **H-A** (representation
    saturation), **H-B** (objective mismatch) and **H-E** (null misspecification)
    are closed under the CRF tests. CRF left **H-D** as the working explanation:
    the practical limit is the legal prefix itself and the residual W7-D3R gap is
    predominantly **post-`t`** information.
  - **2026-08-29 update.** The Arm-C horizon residualization follow-up challenges
    that last inference: after removing a within-t endpoint/horizon projection,
    the residual remains **+0.038303** dominant-cell AUC over Arm B.
  - **2026-08-30 update.** A nested fold-pure causal student did shadow part of
    that residual from the existing prefix feature bank, giving `RT600 +
    residual_student` **+0.001973** pooled whole-dev gain and clearing the
    pre-break damage cap (`0.013672 < 0.015000`). H-D is no longer sufficient as
    a complete explanation; the remaining work is confirmation and deployment
    costing, not feature expansion.
- **Final Wave reports:** [`reports/wave4/`](reports/wave4/),
  [`reports/wave5/`](reports/wave5/), [`reports/wave6/`](reports/wave6/),
  [`reports/wave7/`](reports/wave7/); Wave 8's report is on the
  `research/wave8-future-aware-distillation` branch, not here.
- **Frozen architecture record:**
  [`FINAL_ARCHITECTURE_FREEZE.md`](FINAL_ARCHITECTURE_FREEZE.md),
  [`FINAL_REPRODUCIBILITY_MANIFEST.json`](FINAL_REPRODUCIBILITY_MANIFEST.json).

_Last updated: 2026-08-30 on `grok-response-issues-20260830`: RT-1257 remains
the best external read; production `RT-600` remains unchanged; Arm-C horizon
residualization plus the nested residual student produced a confirmation-only
score candidate. Update
this file whenever the production anchor, external score, or active research
conclusion changes — see `AGENTS.md` at the repo root for the update rule._
