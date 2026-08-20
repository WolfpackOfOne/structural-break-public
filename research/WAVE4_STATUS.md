# WAVE 4 — LIVE STATUS

**Living document. Updated as results land. If you are picking this up cold,
read this file first, then `research/EXPERIMENT_ID_MAP.md`, then
`research/RDOF_LEDGER.md` (wave-4 pre-registration at the bottom).**

Branch: `research/wave3-integration`, pushed to `origin`.
Started 2026-08-20. Deadline 2026-09-17.

---

## HOW TO RUN ANYTHING HERE

```bash
WT="/path/to/workspace/structural-break-claude-wave3"
# every job goes through this; it sets SBR_ROOT/SBR_STORE/SBR_FEATURES,
# detaches, and holds the machine awake for the job's lifetime
"$WT/research/scripts/bg.sh" <logname> research/scripts/<script>.py [args...]
tail -f "$WT/logs/<logname>.log"
```

Interpreter: `/path/to/workspace/structural-break/.venv/bin/python`
(**not** the anaconda `python3` on PATH — its numpy/pandas pairing is broken).
Machine: 10 cores, 16 GB. Three concurrent CHAMP runs is the safe ceiling;
four concurrent is fine at the ABL protocol.

## WHAT EXISTS ON THIS MACHINE THAT IS NOT IN GIT

* `cache/store` — the 10,000-series store, 5,036,517 online rows, break rate 0.4967
* `cache/features` — 10 GB, 8 modules, 500 production columns + 51 `m09_back`
* `research/oof/*.npy` — OOF score vectors, gitignored by repo policy
* `models/rt150_ensemble` (in the **codex** worktree) — the Crunch-tested boosters

**The wave-2 stream OOF vectors did not survive the container that made them.**
That is why wave 4 has to retrain the seven specialist streams before it can say
anything about the champion at all.

---

## PHASE 1 — INTEGRATION — **DONE**

| | |
|---|---|
| integration branch | `research/wave3-integration` |
| research parent | `bfcb232` (`research/wave2-2026`) |
| engineering parent | `24675a6` (`codex/wave3-engineering`) |
| integration commit | `c4fb01e` |
| conflicts | 1 file, `research/scripts/wave2_lib.py`, 2 hunks |
| resolution | best-of-both, not ours/theirs — see `research/reports/wave3_integration_audit.md` |

Both source branches are untouched and still on `origin`.

## BOOKKEEPING REPAIR — **DONE** (`4ffc411`)

* The deployable champion now has a canonical ID: **`RT-250`**. It previously had
  no row at all, and the informal name "RT-150" resolves in the same CSV to
  `RT-150_f1`, a **rejected** DGP-gating run at 0.60619.
* `research/EXPERIMENT_ID_MAP.md` maps every ambiguous name. Nothing was renamed.
* `research/reports/rt250_provenance.json` records the two real defects: the
  manifest's `code_git_sha` `b5ea9d1` does not resolve anywhere, and the stream
  OOF vectors are gone.
* Verified in passing: this worktree's canonical `id,fold` hash is
  `6e114f80…b9e9`, matching the Crunch test report. Same fold partition.

## PHASE 2 — CHAMPION AUDIT — **IN FLIGHT**

| experiment | what | status |
|---|---|---|
| W4-E1 | seven specialists vs seven seed clones, identical SCDF calibration | **RUNNING** — 12 trainings, logs `w4a/w4b/w4c` |
| W4-E3 | SCDF time coordinate `log(t+1)` vs `log(max(t,1))` | code done and parity-checked, awaits E1's OOF |
| W4-E5 | exact wave-1 streams for the RT-131 apples-to-apples audit | configs staged as `RT-430`..`RT-434`, not launched |
| W4-E2 | ensemble delta under `folds_alt1/2/3` | not launched |
| W4-E4 | 1 / 7 / 14 / 35 booster inference cost | not launched |

### Already established before any of it runs

**The SCDF incumbent has a real bug.** `log(max(t,1))` maps t=0 and t=1 to the
same anchor position, and t=0 rows are scored against an anchor grid they were
excluded from building (the fit cuts windows on raw t, the evaluation clamps).
`research/scripts/wave4_cal.py` implements both coordinates; `SCDF_T` reproduces
the shipped `SmoothTimeCDFCal` to `max|diff| = 0.0` including t=0..39, bug and
all, because the control arm has to be the thing that shipped.

**RT-125 / GOSS is reproducible.** Under lightgbm 4.7.0 the two conflicting
wave-1 forms train to identical predictions; GOSS ignores `bagging_fraction`.
No stream needs an "UNREPRODUCIBLE" label on that account.

**The original streams' diversity was never as high as advertised.** Wave 2 cited
"within-timestep rank correlations 0.40–0.69" for its seven streams. Those are
stream-vs-stream pairs. Against the *champion*, `research/reports/portfolio.json`
gives 0.617 / 0.617–0.780 — and wave 3 measured a pure seed clone at 0.785. The
gap that was supposed to separate a specialist from a seed clone is much smaller
than the wave-2 narrative implies. W4-E1 is what settles it.

## PHASES 3–8 — NOT STARTED

---

## IF YOU HAVE TO STOP RIGHT NOW

Submit `models/rt150_ensemble` via `submissions/C_ensemble_deployable.py`. It has
passed the official `crunch test` with the determinism check, at
`INFER_PARALLELISM = 1`, 2.77 ms/point locally, projecting 7.8 h on the private
set against a 15 h budget. Its weaknesses are provenance (unreachable
`code_git_sha`) and that it is trained on 8,000 of the 10,000 labelled series —
**not** correctness. It is a legal, tested, deterministic artifact.
