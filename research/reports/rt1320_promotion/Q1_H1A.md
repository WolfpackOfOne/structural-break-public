# Q1 / H1a — is Arm C's +0.07110 legally shadowable, or imported future information?

Analysis ID: **AN-Q1-D1-20260903**. No `RT-*` ID allocated: nothing is trained,
refitted or re-scored. Pure set algebra plus descriptive channel accuracy over
OOF vectors that already exist.

Diagnostic **D1** of
`research/ai/investigations/20260829-173245-unresolved-quant-questions/02_PRIMARY_RESEARCH.md`,
extended with the channel arm (that document's candidate experiment 4,
"oracle-gain anatomy").

---

## PART 1 — DECISION RULE

**This section was written and committed before any statistic in Part 2 was
computed.** Commit history is the evidence.

### Population

The canonical development partition: folds 0–4, **4,032,524** finite rows inside
the **5,036,517**-row canonical row space. Fold −1 (the spent lockbox) is not an
evaluation surface here and is untouched; all five vectors carry zero finite
values on it, which is checked and reported.

Primary population is the **W7-D3R dominant cell** — `t >= 200`, positives with
post-break age `>= 100`, all negatives — because that is the cell in which
+0.07110 was measured. Whole-dev is the declared secondary population.

### Pair universe

Pairs are drawn by the repository's own `pair_repair_stats` sampler, reused
verbatim (`research/scripts/catboost_specialist_2026.py`, itself ported from
`wave8_common.py`): within each distinct `t`, sample `k = min(64, n_pos, n_neg)`
positives and negatives without replacement and pair them elementwise, at the
canonical `PAIRS_PER_T = 64`, `PAIR_SEED = 20260827`.

The sampler's draw depends only on `(rows, y, t, pairs_per_t, seed)` and never on
the scores. Holding those fixed therefore yields **one common pair universe of
size N** on which all four sets below are evaluated, which is what makes the set
algebra well defined. The only extension to the shipped function is that it also
returns the sampled row indices; the sampling loop itself is unmodified.

### The four sets

| set | definition | vectors |
|---|---|---|
| `a` | Arm C ranks the pair correctly, Arm B does not | RT-991 vs RT-990 |
| `b` | T2 ranks the pair correctly, T0 does not | RT-995 vs RT-990 |
| `c` | seed clone ranks the pair correctly, champion does not | RT-401 vs RT-300 |
| `d` | E0 ranks the pair correctly | cross-fitted SCDF_NSEEN equal-weight blend of RT-300, RT-410…RT-415 |

`c` is the chance baseline: a seed clone differs from its champion only by noise,
so `|c ∩ a|` measures how much overlap with Arm C's repair set arises for free.

Ranking is compared **within `t`**, and `SCDF_NSEEN` calibration is monotone
within a fixed `t`, so raw and calibrated vectors induce the same pair ordering.
Raw vectors are therefore used for the single-model contrasts and the cross-fitted
blend for `d`, with no inconsistency.

### Statistics

```
R_raw  = |b ∩ a| / |c ∩ a|
lift_x = |x ∩ a| / ( |x| · |a| / N )          for x in {b, c}
R_lift = lift_b / lift_c
```

`R_raw` is reported because the brief requires it. It is on its own **confounded
by repair-set size**: under complete independence a repair set twice as large
intersects `a` twice as often, so `R_raw` can exceed 1 with no shared information
whatsoever. `R_lift` divides that out. Both are reported and the verdict requires
them to agree; where they disagree the result is inconclusive by construction.

### Declared thresholds

Stated now, before the numbers:

- **overlap clearly above chance** — `R_lift >= 1.25` **and** `R_raw >= 1.25`
  **and** per-fold `R_lift > 1` on **5/5** folds.
- **overlap at chance** — `R_lift <= 1.10` **or** per-fold `R_lift > 1` on
  **≤ 3/5** folds.
- anything else — the overlap arm is inconclusive.

### Prefix-analogue channels — a closed list of three, declared in advance

The criterion names "excursion duration, persistence" as the channels whose
prefix analogues would weaken H1a. The probe is the existing
`m20_dwell_probe` block — `res64_run90` (current excursion dwell),
`res64_maxrun90` (running maximum dwell = persistence), `res64_mass90`
(excursion mass) — an AR(2)-residual excursion block computed causally over the
full row space and **passed through the repository's own bitwise prefix-invariance
check** (`sbr.features.base.check_prefix_invariance`, `atol=0.0`). It is legally
visible at time `t` by the repo's own standard.

**No other channel is examined, and no threshold, window or quantile is scanned.**
The list is three columns, fixed before computing.

For each channel `x`, pair-accuracy on Arm C's corrected pairs:

```
A_x = mean over pairs in `a` of  1[ x(positive) > x(negative) ]
```

- A channel **has a demonstrable prefix analogue** on `a` iff
  `A_x >= 0.55` and its cluster-bootstrap CI excludes 0.50 at the corrected level.
- The pairs are **prefix-inaccessible** iff every channel has `A_x` inside
  `0.50 ± 0.03` or a CI containing 0.50.

0.50 is a **conservative** null on `a`: `a` is by construction the set Arm B got
wrong, so any channel correlated with Arm B is biased *below* 0.50 there. A
channel clearing 0.55 on `a` has cleared a bar tilted against it.

**Channel validity precondition.** A channel is an informative probe only if its
accuracy on the *full* sampled pair universe of the dominant cell is `>= 0.55`.
A channel that is near 0.50 everywhere is simply weak, and its null result on `a`
says nothing about prefix accessibility. Such a channel is reported but is not
allowed to support a "prefix-inaccessible" conclusion — if all three fail the
precondition, the channel arm is uninformative and the verdict cannot be
STRENGTHENED.

### Multiplicity

**Five declared inferential comparisons**: `R` on the dominant cell, `R` on whole
dev, and the three channel accuracies. Bonferroni at family α = 0.05 gives
**α = 0.01 each — all intervals below are 99% intervals.**

Uncertainty is by **series-level cluster bootstrap**, 2,000 replicates, seed
20260903: pairs sharing a series are not independent, so a binomial interval
would be far too narrow. Series are resampled with replacement and the statistic
recomputed.

Per-fold values (5) and the seed-stability spread (5 auxiliary pair seeds:
20260827, 1, 7, 42, 2026) are **stability descriptors, not additional tests**. No
maximum or best case over folds, seeds or channels is ever promoted to the
reported statistic; the reported statistic is fixed in advance as the pooled
dominant-cell value at the canonical seed. This is the specific failure the Q3
analysis made — a greedy search over 72 vectors reported as a verdict while its
95% CI contained zero — and it is why the search space here is closed before the
first number.

### Verdicts

- **H1a STRENGTHENED** — the overlap arm is *clearly above chance* and stable
  (5/5 folds), **and** the overlapping pairs are *prefix-inaccessible* under
  channels that passed the validity precondition.
  → Arm C's advantage is imported future information; the bank is near the true
  legal ceiling; the programme is finished for this information set.
- **H1a WEAKENED** — at least one declared channel has a demonstrable prefix
  analogue on `a` at the corrected level. → a new feature family is worth
  building; the report names the channels.
- **INCONCLUSIVE** — anything else.

The two arms are logically independent: the channel arm can weaken H1a whatever
the overlap arm shows, because it speaks to Arm C's gain directly rather than to
T2's transfer route. Where the arms land in a configuration the preregistered
rule does not name — for instance overlap at chance *together with*
prefix-inaccessibility — the verdict is **INCONCLUSIVE**, and the sub-arm
findings are reported separately and explicitly labelled as not constituting a
verdict. The rule is not rewritten after the fact to accommodate them.
