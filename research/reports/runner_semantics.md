# A4 — Official runner semantics and what `infer()` may legally use
**Status: RESOLVED. This closes the RT-131 legality question.**

## Source of truth
1. `crunch-cli==11.11.0` (`crunch/runner/{runner,local,unstructured}.py`, `crunch/tester.py`),
   read directly from the wheel.
2. The competition quickstarter contract as reproduced in this repository's
   `realtime_submission.ipynb` (which was generated from the official quickstarter).
3. Organiser statements on the competition forum / docs (see `research/PUBLIC_IDEA_MAP.md`).

## The interface

```python
def train(datasets: List[Tuple[int, List[float], List[float], Optional[int]]],
          model_directory_path: str) -> None: ...

def infer(datasets: Iterable[Tuple[List[float], Iterable[float]]],
          model_directory_path: str):
    model = load(model_directory_path)
    yield                                  # readiness handshake, no value
    for x_historical, x_online in datasets:
        state = init(x_historical)
        for point in x_online:             # single-pass iterator
            yield float(score)             # exactly one score per online point
```

Plus a module-level `INFER_PARALLELISM` integer that the runner honours.

## Findings, stated as binding constraints

**F1 — `datasets` is single-pass, and so is every series' `x_online`.**
You may not rewind, look ahead, or buffer the whole test set. At online index `t`
of series `i` you have seen: this series' full history, this series' online
points `0..t`, and (if `INFER_PARALLELISM == 1`) whatever earlier series you
already consumed. You have *not* seen any later series at all.

**F2 — there is no simultaneous cross-section.** Series are consumed one at a
time to completion, not in lockstep across `t`. The set of scores of all alive
series at index `t` — the object RT-131's rank average needs — **does not exist
at any point during a legal `infer()` run.**

**F3 — `INFER_PARALLELISM > 1` makes cross-series state non-deterministic**, and
the runner re-runs `infer()` and compares outputs (tolerance ~1e-8; determinism
check is on by default in the cloud, `--no-determinism-check` only affects local
`crunch test`). Organiser statements confirm that cross-series accumulation is
treated as defeated by parallelism + the determinism recheck, and that
persisting state to disk between runs is disqualifying.

**F4 — `n_online` is unknowable at time `t`** and must never be a feature. This
is not a theoretical concern: the organisers invalidated all pre-2026-06-08
predictions over exactly this leak. Our feature bank has never used it — keep it
that way, and keep the guard test in the production suite.

**F5 — anything learned from TRAINING data is legal at inference.** `train()`
receives the full labelled training set and may persist arbitrary artifacts to
`model_directory_path`. A calibration map, a CDF table, a per-time-bucket
quantile grid, a distilled student — all legal, because they are frozen
functions of training data, not of the test cross-section.

## Verdict on RT-131

> **RT-131's within-timestep cross-sectional rank average is NOT a legal
> inference procedure.** Its 0.62541 is an offline ORACLE/DIAGNOSTIC number and
> an upper bound on what a legal blend of those seven streams can achieve. It is
> not a competition score and must never be reported as the champion.

The legal replacement is a **per-series transform learned entirely from training
data** applied to each stream before averaging (F5). Because TS-AUC only cares
about the ordering across series *within* a timestep, what the oracle rank does
is put every stream on a common, time-conditional scale. A frozen
`F_m(score | t)` estimated cross-fitted on training OOF is the deployable
approximation of exactly that object. That is experiment family B1/B2/B3.

## What a legal harness must therefore test
- single-pass `datasets` and single-pass `x_online` (wrap both in one-shot iterators that raise on re-iteration)
- exactly one yielded score per online point, in order, finite, in [0,1]
- readiness `yield` before the first score
- byte-identical outputs across two runs (determinism)
- identical outputs under `INFER_PARALLELISM` 1 vs 4 (no cross-series coupling)
- prefix/poison invariance: mutating `x_online[t+1:]` must not change the score at `t`
