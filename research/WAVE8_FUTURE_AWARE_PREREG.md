# WAVE 8 — PRE-REGISTRATION: FUTURE-AWARE DISTILLATION (SST / ORR / PCFB / CFEP / TGMC)

**STATUS AT WRITING: PRE-REGISTERED, NOT YET SCORED.** Nothing under any ID in
§4 may be read before this file is committed. This document converts
`research/WAVE8_PROPOSAL_future_aware_distillation.md` (proposal only, no `RT-`
number, no arms frozen) into a binding pre-registration, per the discipline
this project has followed since `WAVE5_PREREG.md`.

## 0. START STATE

- Working branch: `research/wave8-future-aware-distillation`, worktree
  `/path/to/workspace/structural-break-wave8` (a dedicated
  `git worktree`, not the primary checkout — the primary checkout at
  `/path/to/workspace/structural-break` is on an unrelated
  branch, `claude/structural-break-competition-entry-yjoj8r`, with unrelated
  untracked WIP; a worktree avoids disturbing it).
- Branch HEAD before this file: `902d232` ("Pre-register nothing yet: propose
  Wave 8 future-aware distillation routes").
- `cache/` (feature memmaps + store, 486 MB) and `research/oof/` (1.3 GB,
  includes `RT-300`, `RT-401`, `RT-410`–`RT-415`, `RT-990`, `RT-991`, `RT-994`,
  `RT-995`) are gitignored and were not present in a fresh worktree checkout;
  both were copied read-only from the `research/wave7-teacher-distillation`
  worktree (APFS clone, no data risk) so this branch can train and evaluate
  without retraining artifacts that already exist and are unchanged by this
  wave. Python env: symlinked `.venv` to the primary checkout's venv
  (lightgbm 4.7.0, torch 2.13.0, already installed).
- No active training process found anywhere on the machine
  (`ps aux | grep -i python/lightgbm/torch` empty) and no lock file indicates
  a live run. Safe to proceed.

## 1. T2 GATE (§5 of the execution brief)

`research/reports/wave7_teacher_nested.md` / `.json` on
`research/wave7-teacher-distillation` @ `5093e0a` is the **final**, not WIP,
5-outer-fold nested (double-cross-fitted) result — supersedes the `399ea6f`
checkpoint (folds 0–2 only) this branch's own proposal doc cited.

| | `T1` (`RT-994`, pure distillation) | `T2` (`RT-995`, hard+teacher 0.5/0.5) |
|---|---:|---:|
| mean Δ vs `RT-990` | +0.00349 | **+0.00943** |
| folds positive | 3/5 | **5/5** |
| bootstrap CI (paired, series-level) | [-0.0047, +0.0109] (crosses 0) | **[+0.0043, +0.0137]** (entirely > 0) |
| clears legs 1–3 (magnitude/folds/bootstrap) | **no** | **yes** — reads MAJOR BREAKTHROUGH |
| leg 4 (alt-partition) | not run | not run |
| ensemble-integration vs RT-600 7-stream | not run | not run |

**Reading:** T2 did not fail. It cleared every promotion leg that has been
measured; the two outstanding legs (alternate-partition, ensemble-integration)
are unrun, not failed. Per §5/§25 of the execution brief this licenses Wave 8
to proceed, using `RT-995` (`T2`) as the standing distillation baseline whose
marginal-alpha accounting (§6/§27) Wave 8 must not double-count against.
Contamination note carried forward: the original one-fold pilot overstated
`T2` by +0.0122 — a reminder that every Wave-8 continuation gate below is
judged on outer-fold-pure numbers only, never on a pilot that reuses an
in-sample teacher.

**Historical-ledger note:** `RT-994`/`RT-995` rows live on
`research/wave7-teacher-distillation`'s `RESULTS.csv`, a branch this worktree
did not fork from at that commit. They are **not** copied into this branch's
`RESULTS.csv` (that would misattribute a different commit's git_sha); T2 is
used here only as *evidence*, cited by SHA, per §25's requirement to measure
`RT600 + T2 + M` combinations from the existing `RT-995.npy` OOF array (copied
in §0) rather than re-deriving T2.

## 2. PILOT FOLD

**Fold 0**, unconditionally, for all five mechanisms, per the proposal's own
choice and the execution brief §11. Not reselected after seeing any result.

## 3. INFRASTRUCTURE THIS PREREG COMMITS TO BUILDING

`research/scripts/wave8_common.py`, providing, and used identically by every
mechanism below:

- `future_row_index(d, h)` — for each row `r` with series `s=sidx[r]`,
  `t=t[r]`, returns the row index of the same series at `t+h`, or `-1` if
  `t+h` exceeds that series' online length. **Verified empirically** (not
  assumed): physical row order is series-major with `t` contiguous `0..n-1`
  per series (`row_of(s,t) = first_row[s] + t`), checked on the first 10
  series and via a global `argsort(sidx)==identity` check before this
  pre-registration was written. This makes future-row lookup an O(1) offset,
  not a search.
- `eligibility_diagnostic(d, h)` — computes `P(t+h exists | y=1, t)` vs
  `P(t+h exists | y=0, t)` at matched `t`, **before** any SST/PCFB target is
  trained on. If these differ materially (>0.05 absolute at any populated `t`
  bucket) that is reported and the affected horizon is flagged; per §13 of
  the execution brief, target *availability* itself never becomes a training
  feature regardless of this diagnostic's outcome — rows without `t+h`
  support are simply excluded from that horizon's target construction, both
  for training and for the diagnostic's own denominator.
- `nested_oof_regressor(...)` — generalises
  `research/scripts/wave7_teacher_nested.py`'s double-cross-fit pattern
  (`build_nested_Q` / `train_inner_teacher`) from a single binary teacher
  target to an arbitrary real-valued (possibly multi-column) LightGBM
  regression target: for outer fold `f`, every outer-training row's predicted
  target comes from an inner model trained on folds `{0..4} \ {f, g}` where
  `g` is that row's own fold — never `f`. Used by SST (structural targets),
  PCFB (latent Z), and ORR (repair propensity, degenerate case: teacher `Q` is
  already nested from Wave 7's own artifacts, reused not rebuilt).
- `fold_purity_test()` — the exact set-arithmetic sentinel from
  `wave7_teacher_nested.py`, reused verbatim (already proven in that file to
  both pass on the nested scheme and correctly reject the old contaminated
  scheme on all 20/20 `(f,g)` pairs); re-run once for Wave 8's own nested
  target construction to confirm the same contract holds for regression
  targets, not just the binary teacher.
- `assert_no_forbidden_columns(names)` — asserts none of: true `tau_index`,
  true post-break age, `n_online`, final online length, any target-*
  availability* flag, or final/terminal padding indicator appear in a
  student's feature names. Run before every training call in every Wave-8
  script.
- `ensemble_marginal(candidate_oof, fold=0)` — the one ensemble contract used
  everywhere: equal-weight calibrated blend of `SPECIALISTS` (`RT-300`,
  `RT-410`..`RT-415`) vs the same set **+ `RT-401`** (the pre-existing,
  already-trained exchangeable seed clone used in `RT-813`'s W5-NULLTEST
  citation — reused, not retrained) vs the same seven **+ candidate**, all
  through `wave5_lib.Ctx.crossfit_blend` (legal, cross-fitted,
  time-conditioned `SCDF_NSEEN` calibration — **never** a global rank
  transform, per execution-brief §12). Fold-0-only for the pilot stage.

Every mechanism script additionally reuses, unmodified: `wave5_lib.Ctx`
(row/fold bookkeeping, age buckets, `bootstrap`), `wave7_d3r.FULL` (the 500
causal-column module list), `sbr.pipeline.Data` / `_stack` / `run` /
`append_result`.

## 4. EXPERIMENT ID ALLOCATION

Verified free: `RT-1000`–`RT-2409` (highest allocated ID anywhere on any
branch's `RESULTS.csv`/`EXPERIMENT_ID_MAP.md` below 2410 is `RT-995`; `RT-2410`
–`RT-2421` are already taken by an unrelated ablation battery — avoided).
**Reserved block for Wave 8: `RT-1000`–`RT-1099`.**

Controls reused with **no new ID** (already-trained, unchanged artifacts):
matched 500-causal-column control = `RT-990` fold-0 OOF slice; RT-600
ensemble = `SPECIALISTS`; RT-600+seed-clone = `SPECIALISTS + RT-401`.

| ID | mechanism / arm |
|---|---|
| `RT-1000` | SST-B oracle, h=50 (offline, fold-0 pilot) |
| `RT-1001` | SST-B oracle, h=100 |
| `RT-1002` | SST-B oracle, h=200 |
| `RT-1003` | SST-C legal (nested OOF), h=50 |
| `RT-1004` | SST-C legal, h=100 |
| `RT-1005` | SST-C legal, h=200 |
| `RT-1006` | SST-C legal, all 3 horizons combined — **primary SST candidate** |
| `RT-1007` | SST-B oracle, all 3 horizons combined (ceiling reference) |
| `RT-1020` | ORR repair-propensity regressor (fold-0 pilot) |
| `RT-1021` | ORR blended score `RT600_score + β·repair` (fold-0 pilot) |
| `RT-1030` | PCFB PCA16, oracle Z |
| `RT-1031` | PCFB PCA16, legal predicted Ẑ |
| `RT-1032` | PCFB PLS16, oracle Z |
| `RT-1033` | PCFB PLS16, legal predicted Ẑ — **primary PCFB candidate** |
| `RT-1041` | CFEP-B, matched-capacity BCE embedding control |
| `RT-1042` | CFEP-C, future-predictive embedding — **primary CFEP candidate** |
| `RT-1051` | TGMC-R, real + random matched synthetic pairs |
| `RT-1052` | TGMC-T, real + teacher-guided synthetic pairs — **primary TGMC candidate** |

(CFEP-A and TGMC-0 controls are the same `RT-990` fold-0 slice reused above —
identical rows/columns/capacity, no new ID needed, same convention `T0` used
in `WAVE7_TEACHER_PREREG.md`.) IDs promoted to the full 5-fold protocol reuse
the same number with `folds="0,1,2,3,4"` and `protocol="full"` in a new ledger
row, exactly as this project's convention allows (`RESULTS.csv` is
append-only; an ID's pilot and full rows coexist, distinguished by `protocol`
and `folds`). `RT-1008`–`RT-1019`, `RT-1022`–`RT-1029`, `RT-1034`–`RT-1040`,
`RT-1043`–`RT-1050`, `RT-1053`–`RT-1059` are reserved headroom for the same
mechanisms' full-CV/bootstrap/redundancy follow-on runs if any promote;
`RT-1060`–`RT-1099` reserved for T2-integration and combined-mechanism
ensemble tests (§25/§26 of the execution brief). Will be updated in
`research/EXPERIMENT_ID_MAP.md` as each ID is actually used.

## 5. MECHANISM SPECIFICATIONS (FROZEN, PER EXECUTION BRIEF)

### 5.1 SST — Structural Successor Targets
8 channels, frozen from the existing causal feature bank (no new raw
computation, no search over columns):

| # | channel meaning | column | source module |
|---|---|---|---|
| 1 | long-window location displacement | `cz100_cur` | `m01_seq` |
| 2 | robust/median location displacement | `loc_h_rl1_q` | `m06_loc` |
| 3 | log variance ratio | `vr_qh_z` | `m03_dyn` |
| 4 | robust scale / IQR ratio | `qd_exp_qiqr` | `m02_dist` |
| 5 | dependence / AR displacement | `ar2_e_acf1` | `m04_resid` |
| 6 | cumulative evidence / GLR-like state | `glz_cur` | `m01_seq` |
| 7 | long-vs-short evidence persistence | `sl_mean_16_128` | `m00_core` |
| 8 | historical no-break compatibility | `bo_nochange` | `m07_bayes` |

Horizons `h ∈ {50, 100, 200}`. Target for row `r` at horizon `h` = the value
of these 8 columns at `future_row_index(d,h)[r]`; rows where that index is
`-1` are excluded from that horizon's target construction (not imputed, not
masked-as-feature). Arms A (control, reused `RT-990`)/B (oracle, offline
only)/C (legal, nested-OOF) exactly as execution-brief §13. Continuation
gate: legal `RT-1006` ≥+0.003 fold-0 dominant-cell TS-AUC over `RT-990`, or
`RT600+SST` ≥+0.0015 over `RT600+RT-401`.

### 5.2 PCFB — Predictability-Constrained Future Bottleneck
Frozen to horizon `t+200` only, `d=16` only, methods `{PCA16, PLS16}` only —
no `d∈{8,32}`, no other horizon, in this pilot. Future source: the same
500-causal-column vector (`wave7_d3r.FULL`) evaluated at `future_row_index(d,
200)`. PCA16 fit on Arm-C future vectors in `outer_train` only; PLS16 fit
between current 500 cols and future 500 cols in `outer_train` only. Continuation
gate: PLS16 legal (`RT-1033`) ≥+0.004 fold-0 standalone over the PCA16 control
at matched `d`, and ≥+0.002 for `RT600+PCFB` over `RT600+RT-401`.

### 5.3 ORR — Oracle Repair Ranker
Reuses Wave 7's already-computed nested inner-fold `Q` checkpoints
(`research/oof/nested_Q_outer0_inner*.npy`, copied in §0) and `RT-600`'s
per-specialist OOF — **no new teacher training**. Same-`t` pairs mined only
within compatible inner-held-out sets. Repair target: a regression
(propensity), not a pairwise loss — explicitly checked against the project's
prior negative pairwise-ranking results (`RT-111`, `RT-700`, `RT-701`) as the
baseline expectation, not zero. `β` selected on inner folds only. Continuation
gate: `RT600+ORR` (`RT-1021`) ≥+0.002 over `RT600+RT-401`, and repair-pair
recoverability `P(repair_pos>repair_neg | RT600 wrong, teacher confident)`
meaningfully above 0.50 (report exact value; "meaningfully" read as ≥0.55
given this project's noise floor on fold-0-only pair counts).

### 5.4 CFEP — Causal Future-Embedding Pretraining
One frozen architecture: causal dilated 1D conv, 32 channels, kernel 3, 6
residual blocks (dilations 1/2/4/8/16/32 — reusing `RT-970`/`RT-971`'s
already-validated causal TCN shell from `research/scripts/wave6_n2_tcn.py`
rather than re-inventing one), 16-dim embedding head. No architecture search.
Objective: non-contrastive latent prediction of a future structural window
(reusing SST's 8-channel representation at h=200 as the prediction target,
so CFEP and SST share a ground-truth definition and are directly comparable)
— **not** BCE. Trained in a **separate subprocess** from any LightGBM call
(`PROTOCOL.md`/commit `1068b96`: torch and LightGBM segfault sharing a
process). Controls: CFEP-A = `RT-990` reused; CFEP-B (`RT-1041`) = identical
architecture trained with masked BCE instead (matched-capacity ablation of
the *objective*, reusing Wave 6's TCN family per the proposal's own
instruction not to re-run an unmatched BCE net). Binding comparison is
CFEP-C vs CFEP-B, not vs CFEP-A. Continuation gate: CFEP-C (`RT-1042`)
≥+0.002 over CFEP-B, and either standalone ≥+0.003 over CFEP-A or
`RT600+CFEP-C` ≥+0.0015 over `RT600+RT-401`.

### 5.5 TGMC — Teacher-Guided Matched Counterfactuals
One intervention family only: weak persistent location displacement.
Simulation parameters (shift magnitude, noise, dependence) estimated from
**outer-training data only**. Synthetic pairs scored offline by `RT-990`-style
causal model and the outer-pure nested teacher `Q`; teacher-guided accepted
pairs require confident `Q(B)>Q(A)` while `RT-990` is ambiguous/wrong.
TGMC-0 = `RT-990` reused; TGMC-R (`RT-1051`) = real + random matched pairs;
TGMC-T (`RT-1052`) = real + teacher-guided pairs. Evaluation only on real
outer-validation rows; synthetic accuracy is diagnostic-only. Continuation
gate: TGMC-T > TGMC-R **and** `RT600+TGMC-T` ≥+0.0015 over `RT600+RT-401`
(or an equivalent dominant-cell gain); killed outright if synthetic accuracy
is high but real TS-AUC is flat, without simulator tuning.

## 6. LEAKAGE / CAUSALITY TEST SUITE (BEFORE ANY SCORE)

`tests/test_wave8_causality.py` (top-level `tests/`, matching
`test_no_tau_leakage.py`/`test_neural_causality.py`'s existing location and
pytest convention), run to green before §7:
outer-fold purity (fold-purity sentinel, both nested-target and nested-teacher
variants; must also demonstrate it *rejects* the old contaminated
global-OOF scheme, 20/20, exactly as `wave7_teacher_nested.py` already proved
for the binary case), inner target-fold purity, second-level OOF construction
(`Z_hat` never trained on its own outer-validation fold), no true `tau`/age/
`n_online`/final-length/availability-flag/terminal-padding as a student
feature (`assert_no_forbidden_columns`), deterministic fold assignment and
target construction (same seed → identical output, checked by hash), scorer
parity against `sbr.metric.ts_auc_flat` (already the project's verified
reference — no reimplementation), `RT-600` production artifact integrity
(`RT-300`/`RT-410`–`415` OOF files load and reproduce their ledger scores).
CFEP additionally: prefix-only normalisation, outer-validation series absent
from SSL pretraining, future target never reachable by the deployed encoder
at inference. TGMC additionally: simulator metadata absent from features,
outer-validation series absent from simulator parameter fitting.

## 7. EXECUTION ORDER

SST → ORR → PCFB → CFEP → TGMC, per execution-brief §19 (information-per-
compute, then cheap-and-orthogonal, then the mechanism most likely to lean on
SST's retention finding, then the highest-risk/cost route, then the route
that changes training support rather than representation). All five receive
a real fold-0 pilot regardless of earlier results, per §18; only mechanisms
clearing their gate proceed to full 5-fold CV, bootstrap, and alternate
partitions.

## 8. WHAT THIS PREREG DOES NOT AUTHORISE

No leaderboard submission at any stage (execution-brief §32). No arm beyond
those frozen in §5 without a new, separately committed amendment. No
selection using `X_test.reduced`, the lockbox (fold `-1`), or
`folds_final10k.parquet`. No d∈{8,32} for PCFB, no architecture search for
CFEP, no second intervention family for TGMC, in this pilot wave.
