# STATUS

Concise pointer to current state. For the full picture (per-fold scores,
lockbox confirmation, what won/failed in detail) see
[`STATE_OF_RESEARCH.md`](STATE_OF_RESEARCH.md) — this file is deliberately
short and gets updated whenever that changes materially.

- **Competition:** ADIA Lab / CrunchDAO Structural Break Challenge —
  **Real-Time Edition** (causal/streaming inference; break location unknown,
  scored one prediction per online time step with Time-Stratified AUC).
- **Current production anchor:** `RT-600`, frozen on the `production/rt600`
  branch (tag `rt600-production-0.6268`). Seven-stream deployable ensemble,
  all 10,000 labelled series. Environment build, LB-002/003/004 verification
  runs, and submission #5 record are on that branch.
- **Current external score:** **0.6268** on the official Crunch leaderboard
  (LB-001), confirmed in [`EXPERIMENT_ID_MAP.md`](EXPERIMENT_ID_MAP.md).
- **Current best deployable internal result:** RT-600 seven specialists,
  pooled OOF ≈ 0.63828 (E0 in the Wave 7 T2 promotion battery).
- **Current research conclusion:** T2/RT-995 promotion battery —
  **MOSTLY REDUNDANT**. T2 clears 3/4 promotion legs and has real
  single-model alpha, but E2 (RT600 + T2 ensemble, ≈0.63882) beats E1
  (RT600 + seed clone, ≈0.63859) by only +0.00024 — essentially no marginal
  ensemble alpha. The first New Avenues sweep is now
  **FIRST_SWEEP_EXHAUSTED**. Second Sweep `SS-01` repair-damage arbitration and
  `SS-02` dominant-cell residual ranking are both **KILL**. SS-01 selected no
  first-sweep sensor actions; SS-02 produced negative pair flow
  (`dominant net=-108`) and negative marginal_vs_clone (`-0.000290`). The
  dominant remaining hypothesis is negative-side/null-state calibration; next
  authorized action is `SS-03`. Full detail: `reports/wave7/` and tag
  `wave7-t2-promotion-mostly-redundant`, plus
  `reports/new_avenues_2026/FIRST_SWEEP_SYNTHESIS.md`,
  `reports/new_avenues_2026/SECOND_SWEEP_PREREG.md`,
  `reports/new_avenues_2026/second_sweep/ss01_repair_damage_arbiter.md`, and
  `reports/new_avenues_2026/second_sweep/ss02_residual_ranker.md`.
- **Latest major negative result:** Wave 8 future-aware transfer family —
  five pilots (ORR, TGMC, SST, PCFB, CFEP), all KILL on the full population.
  Lives on the sibling branch `research/wave8-future-aware-distillation`
  (tag `wave8-future-aware-final`); not merged into this lineage.
- **Latest major positive result:** W7-D3R same-prefix-vs-full-sequence
  diagnostic (tag `wave7-d3r-information-frontier`) established the
  future-information limit (CASE 2) that later Wave 7/8 work is measured
  against.
- **Active research branch:** this worktree is
  `research/new-avenues-pilots-2026`; the canonical chain remains
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
  shuffled-control gap `-0.000087`. First-sweep killed-arm arbitration and
  incumbent-bank residual pair ranking are both closed under their frozen
  screens. Proceed next to `SS-03`; production `RT-600` remains unchanged.
- **Final Wave reports:** [`reports/wave4/`](reports/wave4/),
  [`reports/wave5/`](reports/wave5/), [`reports/wave6/`](reports/wave6/),
  [`reports/wave7/`](reports/wave7/); Wave 8's report is on the
  `research/wave8-future-aware-distillation` branch, not here.
- **Frozen architecture record:**
  [`FINAL_ARCHITECTURE_FREEZE.md`](FINAL_ARCHITECTURE_FREEZE.md),
  [`FINAL_REPRODUCIBILITY_MANIFEST.json`](FINAL_REPRODUCIBILITY_MANIFEST.json).

_Last updated: 2026-08-25, after SS-02 was executed and killed on its
preregistered fold-0 screen. Update this file whenever the production anchor,
external score, or active research conclusion changes — see
`AGENTS.md` at the repo root for the update rule._
