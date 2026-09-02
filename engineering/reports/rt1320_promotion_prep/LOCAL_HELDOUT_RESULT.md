# Local held-out read of RT-1320: result

Date: 2026-09-02. Rule declared in `LOCAL_HELDOUT_PREREG.md`, committed `8b402c0`
**before** these numbers were computed.

## Result

100 series (ids 10000–10099, zero overlap with the 10,000 training series),
50,983 rows, 13,188 positives (25.9%). Official pair-weighted TS-AUC via
`sbr.metric.ts_auc_flat`.

| system | TS-AUC |
|---|---|
| **E0** — RT-1257, 7 members | 0.600504 |
| **E2** — RT-1257 + RT-1320, 8 members | 0.592694 |
| **E2 − E0** | **−0.007809** |

## Declared interpretation, applied

The rule fixed beforehand:

- below −0.010 → informative, investigate before spending quota
- **−0.010 to +0.010 → says nothing**
- above +0.010 → says nothing on its own

**−0.007809 falls inside the band. By the declared rule this read says nothing,
and is recorded as saying nothing.** It does not support RT-1320 and it does not
count as evidence against it. RT-1320's status is unchanged: `RESEARCH_ALIVE`,
blocked on `external_score`.

## Stated plainly, without relitigating the rule

The delta is **negative**, and at −0.0078 it sits about 78% of the way to the
informative threshold. That is a fact worth recording rather than burying, and it
is the direction that would matter if it held up.

It is equally a fact that this is exactly the regime the band was drawn for.
100 series is ~1/80th of the ~8,000 behind the 0.0011 dev noise floor; scaling by
√80 gives ≈0.0098, so a ±0.010 band is roughly one standard error. A result of
this size is what sampling noise alone produces here. The threshold was set
before the number was seen, and is not being moved now that the number is
uncomfortable — moving it in either direction is the failure mode the
preregistration exists to prevent.

Note also that both absolute values (0.600, 0.593) sit well below the ~0.629 dev
figures. That is expected: a different 100-series draw, not a comparable
population.

Per the prereg, this is **one read**. The slice is not re-queried.

## A methodological correction

The first pass also attempted a per-series decomposition — how many of the 100
series improve versus worsen. It returned "0 improve, 0 worsen, 100
tied/degenerate", which is not a finding but a category error on my part.

TS-AUC is **cross-sectional**: at each timestep `t` it ranks series against one
another and weights by `n_pos(t)·n_neg(t)`. Within a single series there is no
cross-section at any `t`, so every per-series value is degenerate by construction.
The metric does not decompose that way, and the prereg's "summarise per-series
direction" line was not well posed. Recorded here rather than silently dropped.

## What would actually settle this

An external score. The one thing this cannot substitute for, and the thing
`external_score` has been blocked on throughout: a real Crunch submission on the
full test population, scored once, against a leaderboard that cannot be re-read.
