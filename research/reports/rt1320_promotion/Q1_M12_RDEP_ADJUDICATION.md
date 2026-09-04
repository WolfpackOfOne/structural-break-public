# m12_rdep — adjudicated CLOSED on existing evidence. No new test is warranted.

Date: 2026-09-03. Analysis ID **AN-M12-ADJ-20260903**. Nothing trained, refitted
or re-scored; this reads committed artifacts against a criterion that was fixed
in advance in 2026-08.

## Why this document exists

`Q1_H1B_RESOLVED.md` named `m12_rdep` as the one concrete candidate still able to
move the score, and characterised it this way:

> the only block ever to beat a seed clone (+0.00141). It failed a **+0.0030** bar
> that the programme no longer uses; the current floor is 0.0011. Whether that
> reversal is legitimate or is re-reading an old negative under a friendlier
> standard must be settled by a preregistered test, not by noticing the number
> now clears.

That framing is wrong, and this document corrects it. The test it calls for
**already exists, was preregistered, and was run.** It is W5-E11.

## W5-E11 already is that test

`research/RDOF_LEDGER.md` §"W5-E11 — PRE-REGISTRATION", written **2026-08-21,
"AFTER stage C reported and BEFORE any W5-E11 arm was trained"**, with its bar
fixed in the same document and its multiplicity explicitly charged.

Its design is exactly what a new preregistration would have specified: seven
streams, each the incumbent configuration verbatim from
`research/scripts/wave2_streams.py` — same seed, same rows, same leaves, same
sampling, same objective — with `m12_rdep` appended to the module list and
**nothing else changed**.

| new id | rebuilds | | new id | rebuilds |
|---|---|---|---|---|
| `RT-751` | `RT-300` | | `RT-814` | `RT-413` |
| `RT-811` | `RT-410` | | `RT-815` | `RT-414` |
| `RT-812` | `RT-411` | | `RT-816` | `RT-415` |
| `RT-813` | `RT-412` | | | |

All seven were trained and are in `research/RESULTS.csv` (2026-08-21, 20:58–22:29,
`status=recorded`). The result is `research/reports/wave5_e11_architecture.json`
and `.log`.

## The result

| | mean TS-AUC | per fold |
|---|---:|---|
| `S` — incumbent seven | **0.625811264** | 0.63828 0.62040 0.63393 0.61751 0.61894 |
| `S'` — all seven + `m12_rdep` | **0.623133406** | 0.63761 0.61797 0.62948 0.62074 0.60987 |
| `B` — seven seed clones | 0.621640484 | 0.63817 0.61667 0.63020 0.61437 0.60879 |

```
S' - S    = -0.002677858     1/5 folds positive
            per fold  -0.00067  -0.00244  -0.00445  +0.00324  -0.00907
paired series bootstrap (200 reps, common random numbers)
            mean -0.00275   median -0.00294
            95% CI [-0.00611, +0.00079]   fraction positive 0.09
```

Recorded outcome, in the file: **"H0 ACCEPTED: the block adds nothing the bank did
not already reach."**

Member by member the block is mixed — 3 of 7 rebuilt streams improve
(`+0.00121`, `+0.00194`, `+0.00127`) and 4 degrade (`−0.00132`, `−0.00610`,
`−0.00316`, `−0.00133`) — and the blend of them lands below the incumbent blend.

## Why the "+0.00141 beat a seed clone" line does not survive contact with this

Look at the third row of the table. **The seed clone `B` is itself −0.00417 worse
than the incumbent `S`.**

So `S' − B = +0.00149` and `S' − S = −0.00268` are the same experiment. The block
clears the clone only because the clone is a degraded control. "Beats a seed
clone" is not evidence of value against anything the programme would actually
ship.

The W5-E11 preregistration anticipated precisely this and ruled the clone
comparison inapplicable **before the numbers existed**:

> **Why no seed-clone control is named for this arm.** `S'` differs from `S` by 57
> columns per stream and by nothing else — not by member count, not by weight, not
> by seed. There is no bagging channel for the gain to arrive through, which is the
> whole reason this comparison is cleaner than the eighth-member one.

The clone control exists to calibrate an **eighth-member addition**, where bagging
diversity is a confound. An architecture rebuild has no such channel, so the
correct control is `S`, and `S` is the control the preregistration named.

## The floor argument does not rescue it

The 0.0030 → 0.0011 floor change is irrelevant here, and it is worth being exact
about why. Both are **positive** bars. The measured quantity against the
preregistered control is **−0.00268**. It clears no positive bar — not 0.0030, not
0.0011, not zero. There is no floor at which this result becomes a pass.

The only way to make `m12_rdep` look like a pass is to switch from the control its
own preregistration named (`S`) to the one that preregistration explicitly
excluded (`B`). That substitution is the exact failure mode `Q1_H1B_RESOLVED.md`
warned against — re-reading an old negative under a friendlier standard — applied
to the wrong number.

## What is NOT claimed here

- Not that `m12_rdep` is a bad module. Three of seven rebuilt streams improved,
  and its stated gaps in the bank (residual-PIT distances, residual CUSUM/CUSUMSQ
  paths, an AR-coefficient LR with σ profiled out) are real and correctly
  identified. It is registered in `MODULE_ORDER`, carries
  `tests/test_stream_parity_m12_rdep.py`, and passes `check_prefix_invariance` at
  `atol=0.0`. The infrastructure is sound; the measured effect on the blend is
  negative.
- Not that today's H1a result closed it. It did not. `Q1_H1A.md` bounds an
  excursion-duration/persistence channel family; `m12_rdep` targets dependence
  structure (a φ change with σ profiled out), a different mechanism. `m12_rdep` is
  closed by W5-E11, on its own evidence, and would have been closed with or
  without H1a.
- Not that W5-E11 was underpowered in a way that hides a positive. The bootstrap
  CI does include zero at its upper edge (`+0.00079`), so the honest reading is
  "no detectable gain, point estimate negative" rather than "proven harmful". That
  is still a fail: the preregistered rule required `> +0.0030`, `≥4/5` folds, and
  a CI excluding zero, and the result misses all three.

## Verdict

**`m12_rdep` is CLOSED.** The question was preregistered, tested with seven
training runs, and answered negative against the control its own preregistration
named. Re-testing it would be re-running a completed, clean, negative experiment
in the hope of a different answer. No new preregistration is warranted, and no
training spend is justified.

## Programme state after this

Every lane is now closed on evidence rather than exhaustion:

| lane | status | authority |
|---|---|---|
| same-bank candidates | closed — bank ≈ 3 effective dimensions | `Q1_H1B_RESOLVED.md` / `Q3_CEILING.json` |
| a new feature family chasing Arm C's oracle gain | closed — no prefix analogue exists to name | `Q1_H1A.md` (AN-Q1-D1) |
| `m12_rdep`, the bank-gap candidate | closed — `S' − S = −0.00268`, 1/5 folds | W5-E11, this document |

`RT-1320` is submitted and its external score is pending; that number is outside
this analysis and nothing here bears on it.

~~**Recorded as finished for this information set.**~~ **See the correction
below — this call was premature.**

---

## CORRECTION — 2026-09-03 — "finished for this information set" was premature

**What happened.** Hours after this document was written, submission #19 scored
**0.6303**, beating the RT-1257 champion's 0.6290 by **+0.0013**. The model is
`RT-1320` — RT-1257 plus the M1 Arm-C residual student.

**What survives unchanged.** Nothing in the measurements above is retracted:

- H1b binding (bank ≈ 3 effective dimensions) stands.
- H1a strengthened (Arm C's oracle gain is largely not prefix-visible) stands.
- `m12_rdep` closed by W5-E11 at `S' − S = −0.00268` stands.
- Pilot 3 / `RT-1201` killing the growth-exponent channel stands.

**What was wrong.** The *inference* drawn from them — that the score itself
could no longer be moved — was overreach. The three closed lanes concern
**new feature families and same-bank additions**. `RT-1320` is neither: it is a
student distilled from Arm C, and it was a live, gated candidate the whole time.

**The analysis above actually predicted this and it was read backwards.**
`Q1_H1A.md` measured that a legal student recovers **39.5%** of Arm C's corrected
pairs, and that **56.2%** of those are pairs E0 already wins. That was presented
as the explanation for why distillation *fails*. The complement — the roughly
17% of Arm C's repair set that is both transferable and not already held — is
precisely what `RT-1320` converted into external alpha. The measurement was
right; the framing around it was pessimistic.

Three independent measurements agree on the mechanism, which is worth recording
because it is now the only evidenced live lane:

| source | finding |
|---|---|
| `RT-1320` promotion plan | gain is never-break-cut: never-break pair net +0.003409, pre-break +0.000363 (~0) |
| `Q1_H1A.md` locality | Arm C's corrected pairs are **78.4%** never-break negatives |
| W7-D0 exact loss cube | never-break negatives carry **74.0%** of dominant-cell loss — but that is **73.99% of loss against 74.25% of weight**, i.e. proportional. See the precision note below. |

**Precision note (2026-09-03).** The "74% of loss" figure identifies where the
*pair weight* is, not a disproportionate weakness. Loss-to-weight ratios inside
the dominant cell are never-break **0.9965** and pre-break **1.0100** — the
champion is very slightly *better* than proportional on never-break and very
slightly worse on pre-break. Difficulty is essentially uniform across negative
type. Targeting the never-break cut is justified because it is the largest pool
(37.5% of total pair weight), not because it is the weakest, and this document
should not be read as claiming the latter.

**External calibration, now twice confirmed.** `RT-1320`'s secondary endpoint
E2−E0 against RT-1257 predicted **+0.001416**; realized **+0.0013**. `RT-1257`
predicted +0.0020/+0.0024 and realized +0.0022. The dev-fold addition contract
and the 0.0011 noise floor are externally validated instruments, not proxies.

**Revised state.** The ceiling is real and close, but **not yet reached**. The
external ladder is 0.6268 → 0.6290 (+0.0022) → 0.6303 (+0.0013) — decelerating,
and a further step on the same trend lands near +0.0007, *below* the 0.0011
noise floor. That is the honest quantitative statement: one or possibly two more
steps of diminishing size, not an open frontier and not a closed one.

The single evidenced live lane is **a second Arm-C student aimed specifically at
the never-break cut**, preregistered under the same addition contract with its
bar fixed first, and treated as probably the last one. `RT-1320` is a single
unoptimised design with a negative fold 0 on both endpoints; the never-break
pool it draws from is the largest remaining block of loss.

Documents superseded by this correction: the "Programme state after this"
table's implication that all lanes are closed, and the answer given in session
that the score could not be improved by modelling. The per-lane closures in that
table are individually correct and stand.

## Residual open items, stated so the record is honest

- **H1c** (pair-weight dilution) was never settled by its own falsification, D2 —
  the cell decomposition of T2's standalone +0.00943. `Q1_H1A.md` gives indirect
  evidence against it (Arm C's corrected pairs are diffuse, not concentrated in a
  niche; and 56.2% of the transferred slice is pairs E0 already wins, which points
  at redundancy rather than dilution). That is inference, not D2.
- **H1d** (pilot underpower) is untouched. D3 would attach CIs to the five Wave-8
  fold-0 KILL marginals. It is cheap, but its value is now retrospective: any
  candidate it resurrected would be a same-bank candidate, which H1b has already
  finished.
- **Scope of the H1a channel bound**: one family — excursion duration, persistence
  and mass at `W=64` on the AR(2) residual. A mechanism from elsewhere is not
  excluded by it, merely no longer indicated.

## Provenance

Reads `research/reports/wave5_e11_architecture.json` / `.log`, the W5-E11
preregistration in `research/RDOF_LEDGER.md`, the arm rows in
`research/RESULTS.csv`, `src/sbr/stream/engine.py:42,52`, and
`src/sbr/features/m12_rdep.py`. Nothing was trained, refitted or re-scored. No
lockbox, test-label or leaderboard data touched. `research/RESULTS.csv` not
edited. No `RT-*` ID allocated.

Incidental cross-check: W5-E11's `S` mean is `0.6258112639463216`. The E0 blend
computed independently today in `Q1_H1A.json` is `0.6258112639463216` — identical
to the last digit, across two analyses two weeks apart.
