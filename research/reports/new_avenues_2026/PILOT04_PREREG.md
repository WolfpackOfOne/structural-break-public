# PILOT 4 -- TRAJECTORY GEOMETRY PREREGISTRATION

Written 2026-08-24 on branch `research/new-avenues-pilots-2026`, before any
Pilot 4 score was produced.

Starting base for this branch remains `origin/research/current` at
`6c37cb122f83012c39706f1f182ce8e3182461be`. At preregistration time,
`origin/research/current` had not advanced with the parallel novel-stream
cleanup commit, so Pilot 4 results will be marked PRE-CLEANUP / PROVISIONAL if
scored before that merge is available.

## Hypothesis

The existing 500-column RT-600 bank does not compare the online trajectory to
the history as a shape. If the current online subsequence stops finding close
neighbours in history and instead finds its closest shapes in the recent online
prefix, that is a trajectory-geometry signal distinct from marginal moments,
window statistics, and lag-2 dependence.

## Information Channel

G1/G3 from `NEW_AVENUES_2026.md`: FLOSS-style history/online arc-crossing plus
nearest-neighbour provenance. This is the direct test of whether temporal order
beyond low-order moments carries independent label information.

## Reserved IDs

| ID | candidate |
|---|---|
| `RT-1202` | real trajectory-geometry scalar |
| `RT-1203` | shuffled-history control scalar |

## Exact Inputs

* Raw history and online arrays from `cache/store`, dev folds only.
* History-fitted robust normalization via median/MAD.
* No true tau, labels, final online length, future rows, or test/lockbox data in
  feature construction.
* RT-600 OOF streams only for diagnostics and ensemble integration.

## Exact Algorithm

For each series independently and for each subsequence length `m in {16, 64}`:

1. Build a z-normalized historical subsequence reference set from all
   length-`m` historical windows. If more than 2,000 historical windows exist,
   choose 2,000 evenly spaced windows. No label or online information is used.
2. For online row `t`, if `t + 1 < m`, emit neutral/NaN component values that
   aggregate to scalar 0 for the candidate.
3. Let `q_t` be the z-normalized online window ending at `t`.
4. Compute `d_hist(t)`: Euclidean distance from `q_t` to the nearest sampled
   historical subsequence.
5. Compute `d_online(t)`: Euclidean distance from `q_t` to the nearest prior
   online subsequence ending at or before `t - m`. This non-overlap rule avoids
   trivial self/near-self matches. If no prior online subsequence exists, emit
   NaN for this component.
6. Provenance component:
   `prov(t) = log1p(d_hist(t)) - log1p(d_online(t))`; positive means recent
   online shapes explain the current window better than history.
7. Arc-crossing component:
   find the nearest historical neighbour index `h_t` of `q_t`. Map online
   endpoint `t` to combined index `H + t`, where `H` is the historical
   subsequence count. Maintain the number of arcs `(h_u, H+u)` for `u <= t`
   whose historical endpoint lies within the last `H/4` historical subsequence
   indices. This is a cheap boundary-crossing proxy: if current online shapes
   still attach near the end of history, arcs cross the history/online boundary;
   if they attach far away or switch to online neighbours, the proxy drops.
   Normalize by elapsed comparable online windows, yielding `arc_rate(t)`.
8. History-order control for `RT-1203`: reuse the same distance/provenance
   values, but deterministically permute historical neighbour indices with
   seed 0 before computing the arc-rate component. This preserves shape
   distances and destroys historical ordering.
9. Build one scalar for each candidate as a fixed z-scored arithmetic mean of
   the four real components: `prov16`, `arc16`, `prov64`, `arc64`; z centers
   and scales are fitted on folds 1..4 only. `RT-1203` uses the same scalar
   construction with shuffled arc components.

## Online State

Per `m`: sampled historical reference matrix, nearest-neighbour historical
indices, prior online z-normalized subsequences, provenance distance state,
boundary arc count, elapsed comparable-window count.

## Parameter-Selection Rule

All parameters are fixed here:

* `m = {16, 64}`
* max historical references = 2,000, evenly spaced
* nearest prior online windows require non-overlap, `end <= t - m`
* boundary band = last `H/4` historical subsequence indices
* history permutation seed = 0
* scalar = equal mean of z-scored components

No validation-label threshold tuning, no weight search, no extra windows after
seeing fold 0.

## Primary Candidate

`RT-1202`, real trajectory-geometry scalar.

## Control Candidate

`RT-1203`, shuffled-history control scalar. It must not match the real
candidate if ordering/trajectory geometry is the active channel.

## Population And Fold

Fold 0 is the headline screen. Folds 1..4 may be used only to fit scalar z
centers/scales. No lockbox.

## Causal Constraints

At online row `t`, only history and online rows `<= t` may affect the state.
The prior-online nearest-neighbour search excludes overlapping windows ending
after `t - m`. Final online length is not used.

## Ensemble Integration

Use `wave8_common.ensemble_marginal` for each scored ID:

* `RT600`
* `RT600 + RT-401 seed clone`
* `RT600 + RT-1202`
* `RT600 + RT-1203`

No global rank transform and no candidate-specific blend weight.

## Metrics

For both `RT-1202` and `RT-1203` report:

* whole-fold TS-AUC
* dominant-cell AUC
* mature-break vs never-break
* mature-break vs pre-break
* t buckets and age buckets
* within-t rank correlation with RT-600
* sampled pair repairs/damage/net
* marginal vs seed clone
* runtime

## Kill Gate

KILL if `RT-1202 marginal_vs_clone < +0.0010`.

Also KILL if the shuffled-history control matches or exceeds the real
candidate's marginal-vs-clone result within `0.0002`, because that means the
measured effect is distance/provenance without usable ordering information.

## Continuation Gate

Run 5-fold confirmation only if:

* `RT-1202 marginal_vs_clone >= +0.0020`, and
* `RT-1202 marginal_vs_clone - RT-1203 marginal_vs_clone >= +0.0005`.

Run paired bootstrap only after a positive five-fold result with
`marginal_vs_clone >= +0.0030` and positive sign on at least 4/5 folds.

## Runtime Ceiling

2.5 hours for feature build, both scored candidates, diagnostics, and report.

## Code Paths

Create:

* `research/scripts/novel_streams/pilots/pilot04_trajectory_geometry.py`
* `research/reports/new_avenues_2026/pilot04_trajectory_geometry.md`
* `research/reports/new_avenues_2026/pilot04_trajectory_geometry.json`
