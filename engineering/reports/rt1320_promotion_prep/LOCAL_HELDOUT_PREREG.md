# Local held-out read of RT-1320: interpretation rule, declared before computing

Date: 2026-09-02. **Written before the numbers were computed.** Nothing in this
document was informed by the result.

## What is being computed

TS-AUC (`sbr.metric.ts_auc_flat`, the official pair-weighted metric) of two
already-existing prediction sets over the same 100-series reduced test slice:

- **E0** — RT-1257, the 7-member production anchor. Predictions from the gate
  item 1 Crunch run, `crunch_run1/prediction.parquet`, 50,983 rows.
- **E2** — RT-1257 + RT-1320, the 8-member system. Predictions from the issue #20
  Crunch run, `crunch_test_run1/prediction.parquet`, 50,983 rows.

Reported quantity: **E2 − E0** on identical rows. Paired, so series-level
variance largely cancels.

Labels come from `y_test.reduced.parquet` (50,983 rows). Series ids are
**10000–10099**, disjoint from the training ids 0–9999 — verified, zero overlap.
Genuinely out of sample.

No model is refitted. Both prediction sets already exist; this is arithmetic over
committed artifacts.

## Why the rule has to be written first

This slice has two properties that make it a selection surface if used casually:

1. **It is tiny.** The 0.0011 paired-bootstrap noise floor was established on
   ~8,000 dev series. This is 100 — about 1/80th. Sampling error here is expected
   to be *larger than the effect*, which is +0.0018 on the dev partitions.
2. **It is re-readable.** Unlike the leaderboard, it can be queried repeatedly.
   `PROTOCOL_CHAMPION_2026.md` exists to stop exactly that, and
   `NEGATIVE_RESULTS_INDEX.md` already records candidates killed for endpoints
   that inflated against a degrading control.

The lockbox is also already spent — `folds_final10k` absorbed all 2,000 former
lockbox series — so this is the only out-of-sample data left, and burning its
credibility by reading it without a declared rule would cost more than the read
is worth.

## The rule

**This read cannot promote anything, and cannot alter RT-1320's status.**
RT-1320 stays `RESEARCH_ALIVE` blocked on `external_score` whatever comes back.
`ACTIVE_COMPONENT` requires membership of the external champion, which requires a
real Crunch score. Nothing here substitutes for that.

Declared interpretations, fixed now:

- **Large negative — E2 − E0 below −0.010.** Informative. It would mean something
  is wrong that four dev partitions and the causality gates all missed. Trigger:
  investigate before spending quota on an external submission. This is the only
  outcome that changes what happens next.
- **Anything between −0.010 and +0.010.** **Says nothing.** Record it as
  uninformative. Do not quote it as support, do not quote it as concern, do not
  re-read the slice hoping for a cleaner number.
- **Large positive — above +0.010.** Also says nothing on its own. On 100 series
  it is as likely to be sampling as signal, and treating a favourable read as
  confirmation while dismissing an unfavourable one is precisely the asymmetry
  the protocol forbids.

The ±0.010 band is set from the ~80× reduction in series count against a dev
noise floor of 0.0011, i.e. roughly √80 ≈ 9× wider, rounded up. It is deliberately
wide.

**One read.** The comparison is run once. If it is re-run for a legitimate
engineering reason — a corrected input, say — that re-run gets recorded here with
its reason, and the original number stays visible.

## What will be reported

E0, E2, the delta, the number of rows and series compared, and the declared
interpretation applied. Per-series deltas will be summarised (how many series
improve, how many worsen) but explicitly not mined for subsets — no "it helps on
the long-horizon ones" conclusions from 100 series.
