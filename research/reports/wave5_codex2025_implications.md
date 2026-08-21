# Implications of the Codex 2025 reproduction — READ-ONLY REVIEW

**Reviewed branch `codex/reproduce-2025-public-solution` @
`422e4b2db989051c39d517f03a947ae475f36a5c`, read only. Nothing was merged,
cherry-picked or modified. Also reviewed: `codex/oracle-information-frontier-2026`
@ `5a3b8a0`, whose frontier study is the more consequential of the two.**

---

## 1. THE HEADLINE: THE REPRODUCTION FAILED ITS OWN GATE

`research/reports/public_2025_reproduction.md` states its verdict in its first
line: **`CALIBRATION GATE: FAIL`**, and answers its own question
"may we now interpret the 2026 strong oracle as an information-frontier
approximation?" with **`NO`**.

The reason is mundane and total: `polars`, `lightgbm`, `shap` and `tabpfn` were
not installed and installing them was refused, so **every rung of the
reproduction ladder R25-010 … R25-050 is `blocked`, with no AUC**. The public
repository publishes no validation AUC, no OOF AUC and no leaderboard score
either, so there is no reported number to compare against.

**There is therefore no verified strong 2025 teacher anywhere in this project.**
Any wave-5 plan that depended on one — teacher/student distillation above all —
is unsupported, and this report does not pretend otherwise.

What the branch *did* establish, and it is worth keeping:

* the local 2025 dataset is byte-identical to the public notebook's Crunch
  release-146 file sizes (four files, hashes recorded), so the data identity is
  strong though not cryptographically proven;
* the public solution's feature construction was read from source and
  manifested: 2,171 source-derived features against a README claim of 2,408;
* the pinned public commit is `6316693333edc5831c2408ca5b155ffa24c302bd`.

## 2. THE RESULT THAT ACTUALLY MATTERS IS ON THE OTHER BRANCH

`codex/oracle-information-frontier-2026` measured something far more useful than
a 2025 reproduction: **how much information is left in the 2026 data at each
horizon**, by giving a model the true break boundary and `h` post-break points —
an oracle the legal system cannot have — and comparing it to `RT-300`.

| h | LGBM-rich oracle (knows τ, sees h post-break points) | legal `RT-300` | headroom |
|---:|---:|---:|---:|
| 5 | 0.5313 | 0.5306 | +0.0008 |
| 20 | 0.5552 | 0.5610 | **−0.0058** |
| 50 | 0.5855 | 0.5895 | **−0.0040** |
| 100 | 0.6161 | 0.6180 | **−0.0019** |
| 150 | 0.6373 | 0.6397 | **−0.0025** |
| 200 | 0.6640 | 0.6530 | +0.0110 |
| 300 | 0.6641 | 0.6637 | +0.0004 |
| FULL | 0.6497 | 0.6100 | +0.0396 |

**At every horizon up to 150, the legal causal model already MATCHES OR BEATS a
model that is told where the break is.** That is the single most important
number either Codex branch produced, and it reframes wave 5 entirely: the
feature bank is not leaving easy known-boundary information on the table at the
horizons that carry the metric's decision mass.

Two honest qualifications, because this is a load-bearing claim:

1. the oracle used a **fixed 150-tree LightGBM over a generic feature bank**,
   while `RT-300` is 600 trees over 500 purpose-built causal columns. "Headroom
   ≈ 0" means "our model already matches *this* oracle", not "no information
   remains";
2. it is series ROC AUC on one row per series, **not TS-AUC**, so it is a
   diagnostic about information content, not a forecast of leaderboard movement.

The 2025-vs-2026 benchmark on the same branch adds the other half: the local
2025 task scores **0.6643** under a comparable pipeline, against a widely
repeated `~0.90` folklore figure that the branch **could not reproduce and
explicitly declines to endorse**. The 2026 task is materially harder than the
2025 one, and the gap is dominated not by feature engineering but by
`B − D = +0.1289`, the value of *knowing where the boundary is* — which the
real-time edition does not give anyone.

## 3. CLASSIFICATION OF EVERY 2025 IDEA

Using the categories the brief asks for, over the translations in
`research/reports/codex_2025_translation.md` (RT-901…RT-913), checked against
what the seven production modules actually contain:

| 2025 technique | class | evidence |
|---|---|---|
| multi-window pre/post stats and correlations (RT-902) | **already present in 2026** | `m00_core` runs 6 trailing windows + expanding over 7 transforms, null-calibrated; `m06_loc` localises over a geometric segment bank |
| F-test / Levene / KS on absolute values (RT-903) | **already present**, partly | `m02_dist` emits KS, chi², JS, Hellinger, TV, CvM, Wasserstein, energy on the raw PIT; the *robust scale two-sample* was the genuine gap and is `m12_rdep`'s `rs_*` block |
| residualised distribution distances (RT-906) | **legally causalizable — and was the real gap** | `m02_dist` is raw-PIT only. Built as `m12_rdep`'s `rd_*` block (W5-E4) |
| CUSUM / CUSUMSQ on regression residuals (RT-907) | **legally causalizable — real gap** | `m01_seq` runs the paths on the raw series, `m04_resid` monitors residual moments without a recursion. Built as `m12_rdep`'s `rc_*`/`rq_*` blocks (W5-E5) |
| Chow-style AR breakpoint regression (RT-908) | **legally causalizable** | built as `m12_rdep`'s `dl_*` dependence LR with the innovation variance profiled out (W5-E6), and as `m11_focus`'s exact max over τ (W5-E7) |
| ACF/PACF distances, Ljung-Box (RT-911) | **already present** | `m03_dyn` `az_*` lag-z bank |
| spectral centroid / rolloff / KL (RT-909) | **already present** | `m03_dyn` `sp_*`, exact Goertzel-equivalent from cumsums |
| realized vol / vol-of-vol / downside vol (RT-910) | **likely redundant** | `m00_core` `sq`/`abs`, `m04_resid` `vol`/`volM`/`volG` |
| transform bank: rank, cumsum, rolling mean/std (RT-901) | **likely redundant** | the PIT/rank view is `m02_dist`'s whole basis |
| SHAP / gain top-200/500 selection (RT-905) | **non-causal as used in 2025** | the public pipeline selects features globally, before the final fit, on all labelled data — which inflates its own validation and is exactly what `research/PROTOCOL.md` forbids |
| **TabPFN OOF meta-feature (RT-904)** | **teacher-only, and unverified** | never executed here; no GPU; and the reproduction that would have priced it failed its gate |
| offline PELT / binary segmentation (RT-912) | **impossible online** | these see both sides of the break by construction |
| CNN autoencoder / LSTM / attention (RT-913) | **teacher-only** | no GPU, no runtime budget, no verified benefit |

## 4. IS TEACHER DISTILLATION JUSTIFIED? **NO**

The brief conditions a `W5-TEACHER` study on the Codex reproduction "showing a
credible strong 2025 reproduction". It does not: the gate failed, and no
reproduction AUC exists at any rung of the ladder.

The oracle frontier makes the case worse rather than merely unproven. A teacher
is worth distilling when it knows something the student cannot learn. Here the
*strongest available teacher* — one told the true break location and given the
post-break data — is **already matched or beaten by the legal student at every
horizon through h=150**. The only regime with real teacher advantage is `FULL`
(+0.0396), which requires the entire post-break segment, i.e. exactly the future
information that can never be distilled into a causal prefix feature.

**Verdict: teacher distillation is NOT justified, and the reason is not "we ran
out of time" — it is that the measured teacher advantage lives entirely in
information that is unavailable by construction.** If a GPU and an executable
2025 environment appear later, the first thing to run is the reproduction ladder,
not the distillation.

## 5. WHAT WAVE 5 TOOK FROM THIS REVIEW

Three of the wave-5 feature blocks exist *because* of this review rather than
despite it: the residualised distances, the residual CUSUM/CUSUMSQ paths and the
AR-breakpoint likelihood ratio are RT-906, RT-907 and RT-908 translated into
strictly causal streaming form. Every other 2025 idea was checked against the
production modules first and found already present, redundant, or impossible —
which is the useful outcome of reading someone else's solution carefully, and is
cheaper than rebuilding it.
