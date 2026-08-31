# CRF-01 · NNCSR — NULL-NORMALIZED CAUSAL SEQUENCE RANKER

## VERDICT: **KILL — ABANDONED AT THE PREREGISTERED CHEAP ABANDON GATE**

| | |
|---|---|
| candidate | `RT-1234` |
| C1 mandatory control (BCE) | `RT-1235` |
| C2 conditional control (temporal shuffle) | `RT-1236` — **reserved, NOT run** (gate-blocked) |
| program preregistration | `CRF_PROGRAM_PREREG.md` @ `85d121f` |
| execution preregistration | `CRF01_EXECUTION_PREREG.md` @ `afba958` |
| preflight commit | `ed05902` |
| folds trained | fold 0 only |
| compute | 0.497 h |
| lockbox / test / production / submission | none |

---

## 1. THE GATE THAT FIRED

`CRF_PROGRAM_PREREG.md` §0.2, fixed before any CRF score existed and evaluated
**first**, before any five-fold spend:

> abandon if fold-0 standalone whole-fold TS-AUC **< 0.600** **AND** within-`t`
> ρ vs the RT600 blend **≤ 0.60**.

| quantity | measured | threshold | fires? |
|---|---:|---:|---|
| standalone whole-fold TS-AUC, fold 0 | **0.592762** | ≥ 0.600 | **below** |
| within-`t` ρ vs RT600 (dominant cell) | **+0.4460** | > 0.60 | **at or below** |

Both conditions hold. **The gate fires.** Per the preregistration the candidate is
abandoned immediately: no further folds, no C2, no ensemble integration, and
`marginal_vs_clone` is never computed. The gate was not relaxed, reinterpreted or
re-derived after the number was seen — its 0.600 / 0.60 thresholds are the ones
committed at `85d121f`, three commits before this score existed.

`marginal_vs_clone` is **NOT COMPUTED**, and that is the intended behaviour rather
than a gap: the fold-0 marginal needs a fold-pure five-fold OOF vector, because the
cross-fitted SCDF fits fold 0's calibration map on folds 1–4. Those folds were
deliberately never trained. The gate exists precisely to avoid paying for them.

---

## 2. MANDATORY REPORT — BOTH ARMS, CANONICAL FOLD-0 SAMPLE

Identical deterministic pair sample for both arms; never re-drawn.
`t ≥ 200` and post-break age `≥ 100` define the dominant cell.

| | **`RT-1234` candidate** | **`RT-1235` C1 (BCE)** | RT-600 |
|---|---:|---:|---:|
| standalone whole-fold TS-AUC | **0.592762** | 0.570543 | 0.638276 |
| standalone dominant-cell TS-AUC | **0.621422** | 0.584322 | 0.677710 |
| mature-vs-never-break AUC | 0.620650 | 0.585700 | 0.676800 |
| mature-vs-pre-break AUC | 0.623570 | 0.580470 | 0.680250 |
| within-`t` ρ vs RT600 | **+0.4460** | +0.3631 | — |
| whole-fold repairs / damage / **net** | 8,890 / 11,752 / **−2,862** | 9,100 / 13,897 / **−4,797** | — |
| dominant repairs / damage / **net** | 6,395 / 9,193 / **−2,798** | 6,722 / 11,741 / **−5,019** | — |
| mature-vs-never **net** | **−2,962** | **−5,217** | — |
| mature-vs-pre-break **net** | **−2,820** | **−4,389** | — |
| unique repair coverage vs `RT-401` | 0.3088 | 0.3311 | — |
| repair Jaccard vs `RT-401` | 0.1740 | 0.1625 | — |
| damage rate on RT600-correct pairs (dominant) | 0.2677 | 0.3418 | — |
| **pre-break** damage rate on RT600-correct pairs | **0.2626** | 0.3295 | cap 0.0150 |
| `marginal_vs_clone` | not computed (gate) | not computed (gate) | — |
| training runtime, fold 0 | 906.6 s | 884.2 s | — |
| parameters | 35,649 | 35,649 | — |

RT-600 baseline sentinel reproduced exactly through the real integration path:
fold-0 `E0 = 0.638276`, `E1 = 0.638586`.

### Age and current-`t` profile (candidate)

| age bucket | 0–5 | 5–20 | 20–50 | 50–100 | 100+ |
|---|---:|---:|---:|---:|---:|
| candidate | 0.5205 | 0.5315 | 0.5479 | 0.5736 | 0.6225 |

| `t` bucket | 0–20 | 20–50 | 50–100 | 100–200 | 200–400 | 400+ |
|---|---:|---:|---:|---:|---:|---:|
| candidate | 0.5655 | 0.5592 | 0.5737 | 0.5881 | 0.5837 | 0.6155 |

The signal is real and grows monotonically with post-break age and with elapsed
online length — the shape a genuine sequence detector should have, not the shape of
an artifact. It simply never reaches the level RT-600 already holds.

---

## 3. THE FACTORIAL, FILLED

`RT-970` is the same shell on the same folds, so the three arms form a clean ladder
in which one factor moves at a time.

| arm | channels | objective | fold-0 whole TS-AUC |
|---|---|---|---:|
| `RT-970` | location/scale + `elapsed` | masked BCE | 0.52618 |
| `RT-1235` | **eight null-normalised, no `elapsed`** | masked BCE | 0.57054 |
| `RT-1234` | eight null-normalised, no `elapsed` | **same-`t` pairwise** | 0.59276 |

* **representation effect, objective held fixed: `+0.0444`**
* **objective effect, representation held fixed: `+0.0222`** whole-fold,
  **`+0.0371`** on the dominant cell
* **total vs `RT-970`: `+0.0666`**

Both hypotheses the Wave-6 report said it could not separate — "family wrong" versus
"objective wrong" — are now separated, and **both were partly right**. Null
normalisation is a materially better representation than location/scale. The
same-`t` pairwise objective is materially better than rowwise BCE *on a learned
representation*, which no prior experiment could show: every earlier ranking arm
(`RT-111`, `RT-700`, `RT-701`, `RT-123`) ran the objective over the **fixed** 500-vector
with gradient-boosted trees, where it was a dead heat.

**And the sum of both real effects is still not enough.** That is the finding.

---

## 4. THE FRONTIER MOVED, AND THE ANSWER DID NOT CHANGE

The program audit's binding fact was that across 17 scored arms `corr(standalone, ρ)
= +0.983` — nothing this project has built has ever been both good and different —
and that the best standalone ever reached at `ρ ≤ 0.60` was `RT-1201`'s **0.58358**,
which returned `+0.000301`.

CRF-01 reaches **0.59276 at ρ = 0.446**. That is a **new best point on the
signal/redundancy frontier for this project**, beating the previous record by
`+0.0092` at comparable redundancy — and it still fails the necessary condition set
*below* the fitted `+0.0030` contour.

This is the strongest possible form of the negative result. The gate was not
calibrated against a weak attempt; it was cleared past by the best low-redundancy
representation the project has ever produced, and the answer was still no.

---

## 5. WHY IT FAILS — THE PAIR FLOW IS UNAMBIGUOUS

Every pair-flow cell is **negative** for both arms. On the dominant cell the
candidate repairs 6,395 of RT-600's 16,193 sampled mistakes — a 39.5 % repair rate,
which is genuinely high — while damaging 9,193 of the pairs RT-600 already had
right, a **26.8 %** damage rate. The pre-break damage rate on RT600-correct pairs is
**0.2626 against a 0.0150 cap**: seventeen times over, not marginally over.

This is the First Sweep's diagnosis in a new form. `FIRST_SWEEP_SYNTHESIS.md` H1
measured that the candidate union repaired 91.8 % of RT-600's dominant mistakes
while damaging 83.0 % of what it had right, and concluded the bottleneck is
repair-versus-damage arbitration rather than detection. CRF-01 is a much better
detector than anything in that sweep — and it arbitrates no better. A learned causal
representation does not, on its own, solve the arbitration problem; and SS-01, whose
entire purpose was to arbitrate, was itself KILL.

The candidate's unique repair coverage against the seed clone is 0.3088 with a
repair Jaccard of only 0.1740 — the repairs really are largely its own, not the
clone's. It simply cannot keep them without paying more in damage than they are
worth.

---

## 6. WHAT THIS CLOSES

The representation × objective factorial is now **complete and empty**:

| | rowwise BCE | same-`t` ranking |
|---|---|---|
| **static 500-column bank** | `RT-300` (incumbent) | `RT-111`, `RT-700/701/702`, `RT-123` — dead heat |
| **learned causal sequence** | `RT-970`/`RT-971`, `RT-1235` | **`RT-1234` — this experiment** |

Closed by this result, under `CRF_PROGRAM_PREREG.md` §1.10:

* **learned causal-prefix representations as an ensemble alpha source.** The best
  one this project can build, with the right objective and no rowwise shortcut
  channel, does not reach the necessary condition.
* **objective mismatch as a live explanation for the W7-D3R gap.** The objective
  effect is real, isolated and measured at `+0.0222` standalone — and it does not
  close the gap.
* the `RT-970` reading. Wave 6 could not distinguish "family wrong" from "objective
  wrong"; both were wrong, both are now fixed, and the family still loses.

**Not closed by this result:** CRF-02. `CRF_PROGRAM_PREREG.md` §2 declares the
amortized conditional generative null **unconditional and independent of CRF-01**,
and its hypothesis is about the *negative side* — that 73.99 % of dominant-cell loss
is never-break negatives whose loss rate is a null-**misspecification** signature.
CRF-01's failure is a statement about extracting more break evidence from the prefix;
it is not a statement about pricing "normal" better. The program continues to CRF-02.

`CRF-03` does **not** open on CRF-01: its opening rule requires
`marginal_vs_clone ≥ +0.0015` on fold 0, and CRF-01 has no marginal at all.

---

## 7. WHAT WAS NOT DONE, AND WILL NOT BE

No tuning. No `CRF-01b`. No architecture, channel, hyperparameter, seed or threshold
change. No re-roll of the seed. No relaxation of the abandon gate. No fishing for a
fold on which it looks better. `RT-1236` remains **reserved and unconsumed** — the
id is not recycled and not reassigned. The receptive field was not extended to chase
the failure, and no Transformer, GRU, SSM, RF-255 or extra channel was tried:
`CRF_PROGRAM_PREREG.md` §0.8/§0.9 forbid exactly that, and the age profile in §2
shows a detector whose problem is arbitration, not memory length.

---

## 8. REPRODUCTION

```bash
W=<this worktree>
export SBR_ROOT="$W" PYTHONPATH="$W/src:$W/research/scripts"
TORCH="…/structural-break-wave8/.venv/bin/python"      # torch 2.13.0, no lightgbm imported
NOTORCH="…/structural-break-new-avenues-pilots/.venv/bin/python"   # no torch present

"$TORCH"   research/scripts/crf01_nncsr.py --build-channels          #    8 s
"$TORCH"   research/scripts/crf01_nncsr.py --arm candidate   --fold 0  # 906.6 s
"$TORCH"   research/scripts/crf01_nncsr.py --arm bce_control --fold 0  # 884.2 s
"$NOTORCH" research/scripts/crf01_integrate.py --stage abandon
"$NOTORCH" research/scripts/crf01_integrate.py --stage report_fold0 --arms candidate,bce_control
```

Fold-0 model state sha256: candidate `ab737b3f80eb…`, C1 `1cf9d01d9696…`.
Streaming state measured at **~20.0 KiB/series** (3 × 256-knot ECDFs plus an 8 × 253
ring buffer) — reported, not optimised, and larger than the program
preregistration's ~6 KB estimate because the true receptive field is 253, not 127
(`CRF01_EXECUTION_PREREG.md` §5.1).

Evidence: `crf01_nncsr.json`, `crf01_abandon.json`, `crf01_report_fold0.json`,
`crf01_sentinel.json`, `research/oof/RT-1234.npy`, `research/oof/RT-1235.npy`
(fold-0 rows finite, folds 1–4 NaN, lockbox empty).
