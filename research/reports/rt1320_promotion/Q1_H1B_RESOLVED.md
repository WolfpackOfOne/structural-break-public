# Q1 / H1b resolved by the Q3 ceiling measurement

Date: 2026-09-03. No new computation — this reads an already-committed result
against an already-preregistered criterion.

## What was preregistered

The recovered investigation
(`research/ai/investigations/20260829-173245-unresolved-quant-questions/02_PRIMARY_RESEARCH.md`)
poses **Q1: what does W7-D3R Arm C's +0.07110 dominant-cell advantage measure**,
and offers three competing explanations for why distilling it recovers so little:

- **H1a** non-shadowability — the advantage is future information, not legally visible at time `t`
- **H1b** redundancy with the bank — the recovered signal overlaps what the seven specialists already carry
- **H1c** pair-weight dilution — the gain lives in too small a slice to move the weighted metric

H1b carries an explicit falsification criterion, written before any of this was run:

> **Falsification:** effective-rank analysis of the existing OOF library; if rank
> ≈ 2–3 and a cross-fitted convex blend of existing vectors cannot beat E1, H1b is
> confirmed as binding.

## What Q3 measured

`Q3_CEILING.json` (commit `72a4651`) computed both halves over 72 legal OOF
vectors on 4,032,524 canonical dev rows.

**Part 1 — rank ≈ 2–3.**

```
participation ratio          2.730      <- inside the preregistered [2, 3]
entropy effective rank       9.040
top eigenvalue share         0.601
mean off-diagonal correlation 0.569
```

72 vectors carrying a participation ratio of 2.73, with 60% of variance in a
single component.

**Part 2 — a convex blend cannot beat the incumbent.**

```
greedy equal-weight blend, per fold  +0.00036 +0.00475 +0.00301 +0.00514 -0.00223
mean                                 +0.002208
95% CI                               [-0.001665, +0.006081]   <- contains zero
```

The interval spans zero, and one fold is negative. The blend does not reliably
beat the incumbent.

Two points make this *conservative* rather than generous:

- The baseline was **RT-1257 (E0)**, not E1. E1 carries a matched clone that
  degrades it by −0.000198, so E1 is the easier target. Failing against E0
  implies failing against E1.
- The blend needed **15 additions to a 7-member base — 22 members** — to reach
  that unreliable +0.0022, which is the signature of fitting selection noise, not
  of finding structure.

**Both halves of the criterion are met. H1b is confirmed as binding.**

## What this means

The distillation gap is not primarily a mystery about the future. It is
**saturation**: the 500-column causal bank, as expressed through 72 fitted
vectors, contains roughly three effective dimensions, and a new member drawn from
that same bank overlaps what is already there almost regardless of how it is
fitted.

That explains a pattern the programme kept rediscovering the hard way and
attributing to individual candidate weakness:

| exhibit | result |
|---|---|
| W5-NULLTEST, an 8th exchangeable member | +0.00003 |
| W4-E6, 13-booster union | rejected, worse than best component |
| `m09_back` | decorrelated *less* than a seed clone (0.8219 vs 0.7846), lost to it |
| RT-995 / T2 | standalone +0.00943, ensemble marginal ~+0.000237 |
| RT-1258 / RT-1259 GPU arms | KILL |
| CAT-411/412/414/415 | all sub-floor, lane closed |

None of those were bad models. They were drawing on an information set that was
already spent.

## What it does not resolve

**H1a and H1c remain open.** Confirming redundancy as binding does not measure how
much of Arm C's +0.07110 is legally shadowable in principle. It says only that
*whatever* is shadowable is largely already present in the bank. A student fitted
on this bank cannot recover much, but that is a statement about the bank, not
about the ceiling of legal causal information in general.

The investigation's own framing of Q1 asked for the split between endpoint/
`n_online` information and future break-path evidence. That decomposition — its
D5/D6/D7 sub-experiments — is untouched and still the way to settle H1a.

## Consequence for what to do next

Same-bank candidates are finished. Q3's decision rule already said the ceiling is
binding "**for this bank**"; H1b now says *why*. The only paths that can move the
score are ones that change the information set:

1. ~~**Add a module to the bank.** `m12_rdep` is the concrete candidate…~~
   **WITHDRAWN 2026-09-03 — this bullet was factually wrong. See the correction
   below.**
2. **Settle H1a** — how much of Arm C's advantage is shadowable at all. If little,
   the bank is near the true ceiling and the programme is finished. If much, a new
   feature family is worth building.
   **RESOLVED 2026-09-03: H1a STRENGTHENED — `Q1_H1A.md` (AN-Q1-D1).** Little is
   shadowable. Arm C's corrected pairs are prefix-inaccessible: three
   causality-verified excursion channels that discriminate at 0.561/0.604/0.562
   across the dominant cell score 0.5023/0.4999/0.5027 on exactly those pairs, all
   99% CIs containing 0.50. No prefix analogue exists to name, so no new feature
   family is indicated.

---

## CORRECTION — 2026-09-03 — the `m12_rdep` bullet above was wrong

**What this document said.** That `m12_rdep` is "the only block ever to beat a
seed clone (+0.00141)", that it "failed a +0.0030 bar that the programme no
longer uses", and that whether the reversal under the current 0.0011 floor is
legitimate "must be settled by a preregistered test".

**What is actually true.** That preregistered test already existed when this
document was written. It is **W5-E11** (`research/RDOF_LEDGER.md`, written
2026-08-21 before any arm was trained), it was executed as seven training runs
(`RT-751`, `RT-811`–`RT-816`, all in `research/RESULTS.csv`), and it answered the
question negative:

```
S  incumbent seven            0.625811264
S' seven rebuilt + m12_rdep   0.623133406
B  seven seed clones          0.621640484

S' - S = -0.002677858   1/5 folds positive
         bootstrap 95% CI [-0.00611, +0.00079], fraction positive 0.09
recorded outcome: "H0 ACCEPTED: the block adds nothing the bank did not
                   already reach"
```

**Why the error mattered.** The "+0.00141 beat a seed clone" figure is a
comparison against `B`, and `B` is itself −0.00417 *worse* than the incumbent `S`.
"Beats a seed clone by +0.0015" and "loses to the incumbent by −0.0027" are the
same experiment. W5-E11's own preregistration ruled the clone control
inapplicable to an architecture rebuild in advance, precisely because there is no
bagging channel for a gain to arrive through — the correct control is `S`.

The floor change is a red herring: 0.0030 and 0.0011 are both **positive** bars,
and the measured quantity against the preregistered control is **−0.00268**. No
floor makes that a pass. Reaching the clone comparison to make it clear would
have been the exact error this bullet warned against, aimed at the wrong number.

**Consequence.** `m12_rdep` is closed, not open. Path 1 above is withdrawn; path
2 is resolved. Full adjudication: `Q1_M12_RDEP_ADJUDICATION.md`. Nothing in the
rest of this document — the H1b analysis, the rank measurement, the exhibit
table — depends on the withdrawn bullet, and none of it is affected.

## Provenance

Reads `Q3_CEILING.json` at commit `72a4651` against the criterion in
`02_PRIMARY_RESEARCH.md`, recovered at commit `2c285cc`. Nothing was trained,
refitted, or re-scored. No lockbox, test, or leaderboard data was touched.
