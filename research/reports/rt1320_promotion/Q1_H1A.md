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

---

## PART 2 — RESULT

# VERDICT: **H1a STRENGTHENED**

Both preregistered conditions are met, and neither is marginal.

1. **The overlap is clearly above chance and stable.** `R_raw = 3.216`
   (99% CI 2.764–3.759), `R_lift = 2.540` (99% CI 2.254–2.886), per-fold
   `R_lift > 1` on **5/5** folds, range 2.31–2.57, and 2.40–2.46 across four
   auxiliary pair seeds. T2's corrected pairs are a large, stable subset of Arm
   C's: **69.8%** of everything T2 repairs lands inside Arm C's repair set.
2. **Those pairs are prefix-inaccessible.** All three declared excursion
   channels pass the validity precondition (0.561, 0.604, 0.562 on the full
   dominant-cell pair universe, every CI excluding 0.50) yet land at
   **0.5023, 0.4999, 0.5027** on Arm C's corrected pairs — every one inside
   0.50 ± 0.03, every 99% CI containing 0.50. The persistence channel that
   discriminates at 0.604 across the cell generally discriminates at **0.4999**
   on exactly the pairs Arm C repairs.

That is the criterion's own language satisfied literally: the corrections are
"concentrated where prefix excursion lengths are statistically indistinguishable
across classes."

**Arm C's +0.07110 is imported future information. The bank is near the true
legal ceiling for this information set, and the programme is finished on this
question — not stalled on it.**

### Sanity: the anchor reproduces exactly

| arm | vector | published cell TS-AUC | recomputed |
|---|---|---:|---:|
| A — legal prefix baseline | RT-300 | 0.65341 | **0.65341** |
| B — same info, more capacity | RT-990 | 0.64749 | **0.64749** |
| C — full sequence, no true tau | RT-991 | 0.71859 | **0.71859** |

C − B per fold: `+0.05277 +0.08046 +0.06588 +0.07604 +0.08180`, pooled
**+0.0710969** vs the published **+0.07110**. Every digit of `wave7_d3r.md`
reproduces from the artifacts. Dominant cell = 2,392,213 dev rows.

### The counts

Dominant cell, pooled over the five folds, `N = 50,458` sampled pairs:

| quantity | count | as a share |
|---|---:|---|
| `\|a\|` Arm C fixes vs Arm B | **8,510** | 16.9% of pairs |
| `\|b\|` T2 fixes vs T0 | **4,817** | 9.5% |
| `\|c\|` seed clone fixes vs champion | **3,804** | 7.5% |
| `\|d\|` E0 already wins | 33,674 | 66.7% |
| **`\|b ∩ a\|`** | **3,361** | 69.8% of `b`; 39.5% of `a` |
| **`\|c ∩ a\|`** | **1,045** | 27.5% of `c` |
| **`\|b ∩ a ∩ d\|`** | **1,890** | **56.2% of `b ∩ a`** |

Per fold:

| fold | N | `\|a\|` | `\|b\|` | `\|c\|` | `\|b∩a\|` | `\|c∩a\|` | `\|b∩a∩d\|` | `R_raw` | `R_lift` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 48,041 | 7,303 | 3,965 | 3,618 | 2,710 | 989 | 1,551 | 2.740 | 2.500 |
| 1 | 47,911 | 8,244 | 4,273 | 3,402 | 2,945 | 951 | 1,594 | 3.097 | 2.466 |
| 2 | 47,797 | 8,034 | 4,442 | 3,400 | 3,195 | 1,059 | 1,852 | 3.017 | 2.309 |
| 3 | 48,319 | 8,621 | 4,849 | 3,926 | 3,541 | 1,222 | 1,896 | 2.898 | 2.346 |
| 4 | 48,197 | 9,093 | 5,051 | 3,893 | 3,647 | 1,092 | 2,074 | 3.340 | 2.574 |

Whole dev (the declared secondary population) agrees: `R_raw = 2.712`
(2.413–3.072), `R_lift = 2.280` (2.082–2.522), 5/5 folds.

### The ratio against its declared threshold

| statistic | value | 99% CI | declared bar | |
|---|---:|---|---|---|
| `R_raw` | **3.216** | 2.764 – 3.759 | ≥ 1.25 | clears |
| `R_lift` | **2.540** | 2.254 – 2.886 | ≥ 1.25 | clears |
| folds with `R_lift > 1` | **5/5** | — | 5/5 | clears |

Note `lift_c = 1.629`, not 1.0. A seed clone's repairs already overlap `a` well
above independence, because `a` is by construction a set of *hard* pairs and any
repair set concentrates there. This is exactly why the brief insisted the
baseline be a ratio against `c` rather than a raw count: measured against
independence the transfer would look like a 4.14× lift, and most of that is an
artefact of pair difficulty. Against the honest seed-clone baseline it is 2.54×
— smaller, and real.

### The channel arm

`m20_dwell_probe` — the AR(2)-residual excursion block, causal by the
repository's own bitwise prefix-invariance check. Accuracy is scored on
mid-ranks (ties = 0.5), the convention of `sbr.metric.ts_auc_flat`. The shipped
`pair_repair_stats` uses strict `>`, which is right for continuous OOF scores
but mis-scores a discrete run length, where 66% of pairs tie; scoring those ties
as losses would have reported a spurious 0.23.

| channel | all pairs (validity) | **on `a`** | on `b ∩ a` | on `a \ b` | on `c` |
|---|---:|---:|---:|---:|---:|
| `res64_run90` (dwell) | 0.5614 [.549,.574] | **0.5023 [.487,.518]** | 0.5540 | 0.4685 | 0.4993 |
| `res64_maxrun90` (persistence) | 0.6038 [.579,.627] | **0.4999 [.467,.534]** | 0.5925 | 0.4395 | 0.5285 |
| `res64_mass90` (excursion mass) | 0.5616 [.550,.575] | **0.5027 [.488,.518]** | 0.5552 | 0.4684 | 0.5005 |

Every probe is valid (≥ 0.55 on the full universe, CI excluding 0.50), so the
null on `a` is a measurement, not a weak instrument. No channel reaches 0.55 on
`a`; all three sit inside 0.50 ± 0.03 with CIs containing 0.50. **No channel has
a demonstrable prefix analogue on Arm C's corrected pairs.**

The 0.50 null is conservative here. `a` is the set Arm B ranked *wrong*, so any
channel correlated with Arm B is biased below 0.50 on it; a channel clearing
0.55 there would have cleared a bar tilted against it. None came close. And the
seed clone's noise-driven repairs (`c`, 0.5285 on persistence) are *more*
prefix-visible than Arm C's — Arm C is repairing pairs that legal excursion
evidence cannot see at all.

### Where the corrected pairs sit

`a` is diffuse, not a niche. Against the pair universe it is mildly tilted to
shorter horizons (t 200–399: 33.3% vs 25.4%), shorter series (n_online < 500:
19.9% vs 12.1%), younger post-break ages (100–199: 31.3% vs 27.0%), and slightly
more pre-break negatives (21.6% vs 18.8%). Every tilt is a few points. There is
no concentrated subpopulation where a targeted legal harvest is waiting.

### What the distillation actually transferred

Reading `a` against `b` and `d` explains the programme's most confusing number —
T2's standalone **+0.00943** collapsing to an ensemble marginal of **~+0.000237**:

- **39.5%** of Arm C's repairs were recovered by T2 — the transfer was real, not
  coincidence. The apparent paradox in the investigation ("T2 *did* transfer
  +0.00943, so the future is not entirely inscrutable") is resolved: it
  transferred a real slice.
- **56.2%** of that slice is pairs **E0 already wins**. More than half of what
  the distillation carried across was already in the bank, so it could not
  appear as an ensemble marginal. This is H1b's redundancy, now visible pair by
  pair rather than inferred from an effective rank.
- The slice T2 recovered is the prefix-visible slice (persistence 0.5925 on
  `b ∩ a`); the 60.5% it could not recover is prefix-*anti*-visible (0.4395 on
  `a \ b`). Arm C's repair set is not homogeneous — it splits into a thin
  legally-shadowable part that a legal student does find and the bank mostly
  already had, and a majority that no prefix excursion evidence reaches.

  **Caveat, stated because it limits this specific point:** `b ∩ a` is selected
  on T2 — a legal model — being right, so conditioning on it selects
  prefix-visible pairs partly by construction. The split is offered as a
  description of the mechanism, not as independent evidence. The verdict does
  not rest on it; it rests on `a` as a whole measuring 0.4999.

### Multiplicity

Five comparisons were declared before computing: `R` on the dominant cell, `R`
on whole dev, and three channel accuracies. Bonferroni at family α = 0.05 gives
α = 0.01, so every interval above is a **99%** interval from a series-level
cluster bootstrap (2,000 replicates, seed 20260903, pairs weighted by
`count[series(pos)] × count[series(neg)]` — pairs sharing a series are not
independent and a binomial interval would be far too narrow).

Nothing was scanned. The channel list was three columns fixed in advance; no
window, quantile or threshold was searched. Per-fold and per-seed figures are
stability descriptors and no maximum over them is reported as the result. The
headline `R_lift = 2.540` clears its 1.25 bar with a lower 99% bound of 2.254 —
it does not depend on the correction, which is the difference between this and
the Q3 greedy search whose 95% CI contained zero.

### Verdict in one line

**H1a STRENGTHENED.** The distillation genuinely touched Arm C's information
(2.54× the seed-clone baseline, 5/5 folds), and the information it touched is
not legally visible at time `t`: causal excursion-duration and persistence
evidence discriminates Arm C's corrected pairs at 0.50. The +0.07110 is imported
future information. **No new feature family is indicated by this evidence, and
the ceiling is real rather than an artefact of a spent bank.**

### Consequence

Read with `Q1_H1B_RESOLVED.md`, the two live hypotheses now both resolve, and
they resolve compatibly rather than competing:

- **H1b binding** — the bank carries ~3 effective dimensions, so same-bank
  candidates are finished.
- **H1a strengthened** — the headroom the oracle demonstrates is not legally
  reachable, so a *different* bank does not obviously recover it either.

The second was the open question that kept the programme alive. Answering it
closes the "build a new feature family to chase Arm C" lane: this analysis found
no channel with a prefix analogue to name, which is precisely the output the
H1a-WEAKENED branch would have required. **This is a confirmed ceiling for this
information set, and it is a result.**

Two things this does *not* close, stated so the record is honest:

- It bounds one named channel family — excursion duration, persistence and mass
  at `W=64` on the AR(2) residual — because that is the family the criterion
  named and the only one with a prereg-clean, causality-verified artifact. A
  channel from an entirely different mechanism is not excluded by this test. It
  is, however, no longer *indicated*: nothing here points at one.
- It does not speak to RT-1320, whose external score is still pending, nor to
  `m12_rdep`, which is a bank-membership question that `Q1_H1B_RESOLVED.md`
  already routed to its own preregistered test.

### Reproduction

| vector | resolved path | sha256 |
|---|---|---|
| RT-990 (Arm B / T0) | `structural-break-kimi-response-issues-20260830/research/oof/RT-990.npy` | `872902e7ac6525a98787997607348d8d54f31d1d7e3002aad971cec2ac1b7c41` |
| RT-991 (Arm C) | `structural-break-wave8/research/oof/RT-991.npy` | `e0b726eaa360fb612e4b983a1c23823f69e6fd999bac5a97ce225a3b155940e5` |
| RT-995 (T2) | `structural-break-wave7-promotion/research/oof/RT-995.npy` | `988eb7c4665088a74fcc4617527d6289df8b969163bb04af900c73745090f144` |
| RT-401 (seed clone) | `structural-break-rt1257-slot-adjudication/research/oof/RT-401.npy` | `16e1a10e75f3a2adf12d0efdfe8f4dd9391cfb53f871909a3f7392d2f175f57e` |
| RT-300 (champion / Arm A) | `structural-break-rt1257-slot-adjudication/research/oof/RT-300.npy` | `bd3e6456fef300a7d0fefacec92095724ab2401fbe046c28733c9600b96879d3` |

RT-995's declared location
(`structural-break-causal-representation-frontier/research/oof/RT-995.npy`) is a
symlink chain resolving to `structural-break-wave7-promotion`; the resolved path
is what was hashed and loaded.

All five are `float32`, shape `(5036517,)`, finite on exactly the **4,032,524**
canonical dev rows and finite on **zero** lockbox rows — provenance-matched to
the canonical row space, so no heuristic alignment was needed and no INVALID
condition arose. Store `meta.parquet`
`2c6aab9b65fc30740bf0ec1109569567af7c5a6ee938264da9f6c05920bd3971` and
`folds.parquet`
`ba4f71fee8fcb30cc808234584a924ff0b0f0529e5bbd1fcbd0b4b780ebcc312` are identical
across every worktree used. `m20_dwell_probe.npy`
`773bf82fc52a12b5285095dbdeae3b0f12d704ef152188c3799d2ce72ade1185`.

E0 is the cross-fitted `SCDF_NSEEN` equal-weight blend of RT-300 and
RT-410…RT-415, mean five-fold TS-AUC **0.625811**.

### Constraints

Nothing was trained, refitted or re-scored. Development population only; the
lockbox (fold −1) was never an evaluation surface and every vector is empty on
it. No test labels or leaderboard data touched. `research/RESULTS.csv` not
edited. No `RT-*` ID allocated — this is `AN-Q1-D1-20260903`. `crunch push` and
`crunch setup` not run.

Machine-readable form: `Q1_H1A.json`.
