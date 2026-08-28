# L3 ARBITRATION PROBE PREREGISTRATION

Date: 2026-08-28
Branch: `research/deep-ensemble-frontier-local-2026`
Lane: LOCAL

This preregistration authorizes a descriptive, fold-0-only arbitration probe over existing OOF
vectors. It authorizes no training, no new RT ID, no lockbox or test-data access, no model
selection outside the fixed rules below, and no neural-model launch.

## Question

Does any fixed gating, abstention, or confidence-weighted combination of CRF-01 `RT-1234` with
RT-600 retain at least half of the unconditional repair count while reducing damage on
RT600-correct dominant-cell/pre-break pairs to below `0.05`?

`RT-1234` is fold-0 only: `806,334` finite rows, `16.01%` coverage. This probe is therefore a
single-fold descriptive diagnostic and cannot promote any model.

## Fixed Inputs

- Candidate: `RT-1234.npy`, existing CRF-01 NNCSR fold-0 OOF.
- Matched control: `RT-1235.npy`, existing CRF-01 BCE control fold-0 OOF.
- RT-600 controls: `RT-300`, `RT-401`, and `RT-410` through `RT-415` from existing OOF arrays.
- Calibration: equal-weight fold-pure `SCDF_NSEEN` where full-fold controls exist; fold-0-only
  candidate/control score vectors are used only on their finite fold-0 support.
- Pair-flow sample: 64 same-`t` pairs per time point, seed `20260827`.

## Fixed Rules

All rules use only frozen prediction vectors and time index/label metadata for evaluation.

1. `unconditional_equal`: equal-weight average of calibrated RT-600 and calibrated `RT-1234` on
   the finite fold-0 rows.
2. `confidence_q05`, `confidence_q10`, `confidence_q20`, `confidence_q30`: use `RT-1234` only
   where its within-`t` rank percentile is in the top or bottom `q`; otherwise defer to RT-600.
3. `rt600_boundary_q10`, `rt600_boundary_q20`, `rt600_boundary_q30`: use `RT-1234` only where
   RT-600's within-`t` rank percentile is within `q / 2` of 0.5; otherwise defer to RT-600.
4. `agreement_direction`: use `RT-1234` only where `RT-1234` and RT-600 have the same sign of
   deviation from their within-`t` medians; otherwise defer to RT-600.
5. `dominant_cell_only`: use the unconditional equal blend only inside the dominant cell and
   defer to RT-600 everywhere else.
6. `dominant_confidence_q10`, `dominant_confidence_q20`: apply the corresponding confidence gate
   only inside the dominant cell and defer to RT-600 everywhere else.
7. `three_way_abstain_q10`, `three_way_abstain_q20`: move toward `RT-1234` only at top/bottom
   confidence tails, defer to RT-600 in the middle, and replace low-confidence disagreements with
   the within-`t` median RT-600 score. The abstain region is fixed by the same quantile threshold;
   it is not fitted to the observed damage rate.

No additional thresholds, quantiles, cells, weights, rules, or best-of variants may be added after
seeing the result.

## Readout

For each rule, report on fold 0:

- dominant-cell repair count, damage count, net, and damage rate on RT600-correct pairs;
- mature-vs-never and mature-vs-prebreak repair/damage/net/damage-rate splits;
- fold-0 `E2 - E1` against the matched `RT-1235` control, with the same rule applied to the
  candidate and to `RT-1235`.

Primary success criterion: a rule works if it retains `>= 50%` of the unconditional dominant-cell
repair count while bringing mature-vs-prebreak damage rate below `0.05`.

## Prior

The prior is poor. Four prior arbitration experiments, SS-01 through SS-04, are all KILL at
`-0.000310`, `-0.000290`, `-0.000299`, and `-0.000312`. The only distinction here is that those
arbitrated over high-correlation candidates (`rho` about `0.86` to `0.89`), while `RT-1234` has
rho `0.446` and repair-Jaccard `0.174` against the seed clone. A negative result is therefore a
valuable gate against spending GPU time on another neural detector.
