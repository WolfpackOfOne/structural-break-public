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
  ensemble alpha. Full detail: `reports/wave7/` and tag
  `wave7-t2-promotion-mostly-redundant`.
- **Latest major negative result:** Wave 8 future-aware transfer family —
  five pilots (ORR, TGMC, SST, PCFB, CFEP), all KILL on the full population.
  Lives on the sibling branch `research/wave8-future-aware-distillation`
  (tag `wave8-future-aware-final`); not merged into this lineage.
- **Latest major positive result:** W7-D3R same-prefix-vs-full-sequence
  diagnostic (tag `wave7-d3r-information-frontier`) established the
  future-information limit (CASE 2) that later Wave 7/8 work is measured
  against.
- **Active research branch:** `research/current` (this branch — tracks the
  canonical chain: wave2 → wave3-integration → wave5-alpha → wave6-alpha →
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
- **Final Wave reports:** [`reports/wave4/`](reports/wave4/),
  [`reports/wave5/`](reports/wave5/), [`reports/wave6/`](reports/wave6/),
  [`reports/wave7/`](reports/wave7/); Wave 8's report is on the
  `research/wave8-future-aware-distillation` branch, not here.
- **Frozen architecture record:**
  [`FINAL_ARCHITECTURE_FREEZE.md`](FINAL_ARCHITECTURE_FREEZE.md),
  [`FINAL_REPRODUCIBILITY_MANIFEST.json`](FINAL_REPRODUCIBILITY_MANIFEST.json).

_Last updated: 2026-08-24, during the repository cleanup pass. Update this
file whenever the production anchor, external score, or active research
conclusion changes — see `AGENTS.md` at the repo root for the update rule._
