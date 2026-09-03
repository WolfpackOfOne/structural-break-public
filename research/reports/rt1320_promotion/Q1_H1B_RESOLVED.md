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

1. **Add a module to the bank.** `m12_rdep` is the concrete candidate — already in
   the streaming engine's `MODULE_ORDER` with parity tests, excluded from
   `PRODUCTION_MODULES` by a one-line tuple, and the only block ever to beat a
   seed clone (+0.00141). It failed a **+0.0030** bar that the programme no longer
   uses; the current floor is 0.0011. Whether that reversal is legitimate or is
   re-reading an old negative under a friendlier standard must be settled by a
   preregistered test, not by noticing the number now clears.
2. **Settle H1a** — how much of Arm C's advantage is shadowable at all. If little,
   the bank is near the true ceiling and the programme is finished. If much, a new
   feature family is worth building.

## Provenance

Reads `Q3_CEILING.json` at commit `72a4651` against the criterion in
`02_PRIMARY_RESEARCH.md`, recovered at commit `2c285cc`. Nothing was trained,
refitted, or re-scored. No lockbox, test, or leaderboard data was touched.
