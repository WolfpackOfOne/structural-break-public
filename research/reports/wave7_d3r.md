# WAVE 7 — W7-D3R RESULTS

Pre-registered: `research/WAVE7_D3R_PREREG.md`, committed at `e672c06` before
any arm was trained. Population: t≥200, positive age≥100, all negatives —
fixed by W7-D0, not re-derived here. Cell pair-weight fraction of dev:
**0.5050**.

## Arms

| arm | exp id | pooled cell TS-AUC | never-break-only | pre-break-only |
|---|---|---:|---:|---:|
| A — legal prefix baseline | `RT-300` | 0.65341 | 0.65413 | 0.65131 |
| B — same info, stronger extraction | `RT-990` | 0.64749 | 0.64912 | 0.64280 |
| C — full sequence, no true tau | `RT-991` | **0.71859** | 0.71439 | 0.73070 |

(Whole-dev, non-cell-restricted pooled scores for context: A 0.61605, B
0.61177, C 0.71989 — B's higher capacity mildly overfits everywhere, not just
in this cell; C's lift is present dev-wide, as expected from an arm with
access to each series' own future.)

## Deltas, per fold

| fold | A | B | B−A | C | C−B |
|---|---:|---:|---:|---:|---:|
| 0 | 0.66753 | 0.66381 | −0.00372 | 0.71658 | +0.05277 |
| 1 | 0.64600 | 0.64176 | −0.00424 | 0.72222 | +0.08046 |
| 2 | 0.67721 | 0.66119 | −0.01603 | 0.72707 | +0.06588 |
| 3 | 0.64695 | 0.63478 | −0.01217 | 0.71081 | +0.07604 |
| 4 | 0.62862 | 0.63536 | +0.00674 | 0.71716 | +0.08180 |

**B beats A on 1/5 folds. C beats B on 5/5 folds**, by a large and consistent
margin (+0.053 to +0.082) on every one of them.

## Pooled deltas and translation

| contrast | cell AUC Δ | classification | translated pooled Δ |
|---|---:|---|---:|
| B − A | **−0.00592** | flat (negative — no extraction headroom found) | −0.00299 |
| C − B | **+0.07110** | large | +0.03591 |

Translation: `pooled_delta ≈ 0.5050 × cell_delta` (the cell's exact pair-weight
share of the whole dev set, from W7-D0), holding every other cell fixed. The
C−B translated figure (+0.0359) is **not achievable in production** — Arm C
is an offline diagnostic that uses each series' own future — it is a ceiling
on how much signal exists in the full sequence for a teacher to try to
transfer, not a deployable gain.

## Interpretation

**Arm B (same information, more tree capacity) does not beat Arm A — it is
mildly negative, and consistently so (4/5 folds).** More leaves, no feature
subsampling and 50% more trees on the identical 500 legal causal columns and
identical rows buys nothing in this cell; if anything it overfits slightly.
This is the same qualitative finding as the whole-dev-set comparison (B
0.61177 vs A 0.61605), so it is not a cell-specific artefact — the current
legal causal feature bank, trained harder with more capacity, has hit its
ceiling here.

**Arm C (same capacity as B, plus each series' own final-online-row feature
vector) is enormous and universal — +0.071 cell AUC, 5/5 folds, no fold below
+0.053.** The mechanism this project already has words for from
`WAVE7_PROPOSAL_metric_aligned_transition.md`: at long horizons a persistent
weak break is a low-SNR signal that only resolves with enough evidence: the
series' own eventual state carries almost all of the discriminating power
this cell is missing, and none of it is legally visible at time `t`.

**Never-break vs pre-break inside Arm C**: 0.71439 vs 0.73070 — the
full-sequence signal resolves BOTH negative types, slightly better against
pre-break negatives. This does not change the routing decision (§ next lane)
but is worth carrying into teacher-target design: the oracle target should
not assume never-break series are the only hard case.

## Verdict

**CASE 2 — future-information limit.**

`B ≈ A` (in fact slightly negative, 4/5 folds) and `C ≫ B` (5/5 folds, large
and uniform). Per the pre-registered interpretation matrix
(`WAVE7_D3R_PREREG.md` §5): the current online prefix is information-limited
for this cell; extraction capacity is not the bottleneck; future observations
resolve what the causal feature bank cannot see. This is the polar opposite
of Case 1 — pouring more tree capacity into the existing 500-column bank is
not the next lane.

## Next lane (per `WAVE7_D3R_PREREG.md` §5 / brief §19)

* **Priority 1 — teacher / distillation.** Build a continuous
  `full_sequence_break_confidence` target from each series' complete online
  trajectory (never true `tau`, never explicit boundary/age) and distill it
  into a causal student. Arm C's exact feature (each row + its series' final
  state) is the natural first teacher representation to start from — it is
  already trained and already an OOF vector (`RT-991`).
* **Priority 2 — targeted simulation / probabilistic evidence accumulation**
  for the same mechanism (slow evidence build-up for weak persistent shifts).
* **Deprioritised — horizon specialist / more tree capacity.** Arm B already
  tested "same information, more capacity" directly on this exact cell and
  it did not help. A capacity-only horizon specialist should not be the next
  spend without a reason to expect a different result than Arm B got.
* **XGBoost/CatBoost (brief §25)**: still secondary. Arm B's result argues
  against "more capacity, same features" as a lever here regardless of which
  library supplies the capacity, so a different tree library is unlikely to
  change this verdict on its own.

## What this does not license

No promotion. No submission. Arm C is not a production candidate — the
final-row broadcast is explicitly non-causal. A teacher/distillation lane
must clear its own pilot bar (`WAVE7_PROPOSAL_metric_aligned_transition.md`,
brief §24: one fold, ≥+0.003, before earning full 5-fold training) before any
number from that lane can be compared against the champion.
